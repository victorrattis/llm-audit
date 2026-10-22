// The opencode 2.x entry of llm-audit.
//
// 2.x does not route a model call through a `fetch` a plugin can wrap: the
// session runner talks to the provider through its own HTTP client. It hands a
// plugin each model request and its response through the session's HTTP hooks
// instead, so this file listens there and gives what it hears to the recorder
// in `../llm-audit.ts` — the same one, so the turn files, the index and the
// folder layout come out exactly as they do on 1.x.
//
// Why a folder and not a file: 2.x drops a configured plugin whose path is a
// file ("configured plugin path must be a directory") and loads a folder's
// `server.ts`. Keep this folder beside `llm-audit.ts`, which it imports. Load
// only this folder on 2.x, and only `llm-audit.ts` on 1.x.
//
// What the hooks give that the wire does not carry: the session the request
// belongs to (on 2.x the session headers carry the cache-routing id, which for
// a subagent is its parent's and for a fork its source's), the agent, the model
// and its reasoning variant (`model.request`), the catalog price of the model
// (`ctx.model.list()`, the tiers `rates_per_million` is picked from), and the
// name of the conversation (the session events). No hook writes to its event,
// so arming the audit cannot change what opencode sends or receives.
//
// What it cannot see: a provider that 2.x drives over a WebSocket (OpenAI and
// xAI, unless the provider's `transport` setting says "http") sends its frames
// through the `experimental.ws.*` hooks, not through HTTP, and is not recorded.
//
// Nothing is imported at runtime but `../llm-audit.ts`: the plugin object is
// written out rather than built with `Plugin.define` from `@opencode/plugin`,
// which is the identity, so the file loads whatever resolves that package.
await import("../llm-audit.ts")

/** What `../llm-audit.ts` leaves on `globalThis` for this entry. */
type Core = {
  resolve: (options: Record<string, unknown> | undefined) => { enabled: boolean; dir: string }
  shared: (settings: { enabled: boolean; dir: string }) => any
  watchExit: (state: any) => void
  capture: (
    state: any,
    request: Wire,
    inflight: Promise<Response>,
    started: number,
    known?: { session?: string },
  ) => Promise<unknown>
  modelBody: (url: string, raw: string) => Record<string, unknown> | undefined
  redactHeaders: (input: unknown) => Record<string, string>
  MODEL_BODY: RegExp
}
const core = (globalThis as Record<string, unknown>)["__opencodeLLMAuditCore"] as Core

type Wire = { method: string; url: string; raw: string; body: Record<string, unknown>; headers: Record<string, string> }

// The slice of the 2.x plugin context this file uses, as `@opencode/plugin`
// declares it. Written out so type-checking does not need that package.
type ModelRef = { id: string; providerID: string; variant?: string }
type Kind = "primary" | "compaction" | "title" | "generate"
type Scope = { sessionID: string; agent: string; model: ModelRef; kind: Kind }
type Hooks = {
  prompt: { sessionID: string; messageID: string }
  "model.request": Scope
  "http.request": Scope & { request: Request }
  "http.response": Scope & { request: Request; response: Response }
}
type Context = {
  options: Record<string, unknown>
  session: {
    hook: <Name extends keyof Hooks>(name: Name, callback: (event: Hooks[Name]) => Promise<void> | void) => Promise<unknown>
    get: (input: { sessionID: string }) => Promise<unknown>
  }
  model: { list: () => Promise<unknown> }
  event: { subscribe: (options?: { signal?: AbortSignal }) => AsyncIterable<unknown> }
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value)
const record = (value: unknown) => (isRecord(value) ? value : undefined)
const str = (value: unknown) => (typeof value === "string" && value ? value : undefined)
const message = (error: unknown) => (error instanceof Error ? error.message : String(error))

// The requests that are the conversation. A `title` or `generate` request is
// opencode naming or summarizing it on the side, on a model of its own; 1.x
// never put those in the folder either.
const RECORDED = new Set<Kind>(["primary", "compaction"])

// How long a request may wait for its response before it is written down as
// one that never got one. The response hook fires when the headers arrive, not
// when the stream ends, so this is far longer than any provider takes to start.
const ABANDONED_MS = 10 * 60 * 1000

/** A request seen going out, waiting for its response. */
type Waiting = {
  sessionID: string
  startedAt: number
  settle: (response: Response) => void
  fail: (error: Error) => void
  // False once the body turned out not to be a model call: its response is not
  // cloned, because a clone nobody reads holds the whole body in memory.
  wanted: boolean
}

export default {
  id: "llm-audit",
  async setup(ctx: Context) {
    const settings = core.resolve(ctx.options)
    if (!settings.enabled) return
    const state = core.shared(settings)
    // The state outlives a reload of the plugin — it is kept on `globalThis` —
    // so one built by an older copy comes back without the newer maps on it.
    state.titles ??= new Map()
    state.messages ??= new Map()
    core.watchExit(state)

    // The model catalog, read once and off the request's path. The price is not
    // on the request, and 2.x lists it as an array of tiers, base first, which
    // the recorder resolves against the size of each prompt.
    let catalog: Promise<Array<Record<string, unknown>>> | undefined
    let warned = false
    const listed = () =>
      (catalog ??= ctx.model.list().then(
        (reply) => {
          const items = Array.isArray(reply) ? reply : Array.isArray(record(reply)?.["data"]) ? record(reply)!["data"] : []
          return (items as unknown[]).filter(isRecord)
        },
        (error) => {
          catalog = undefined
          if (!warned) console.warn(`[llm-audit] model catalog lookup failed; turns go without rates: ${message(error)}`)
          warned = true
          return []
        },
      ))
    void listed()

    /** A new thing the user typed: the boundary the turns are grouped by. */
    await ctx.session.hook("prompt", (event) => {
      state.requests.set(event.sessionID, (state.requests.get(event.sessionID) ?? 0) + 1)
      if (event.messageID) state.messages.set(event.sessionID, event.messageID)
    })

    await ctx.session.hook("model.request", (event) => {
      // A title request runs beside the turn on another model; letting it in
      // would file the turn in flight under the title model's name and price.
      if (!RECORDED.has(event.kind)) return
      const model = event.model
      const context: Record<string, unknown> = {
        agent: str(event.agent),
        providerID: str(model?.providerID),
        modelID: str(model?.id),
        variant: str(model?.variant) ?? "default",
      }
      state.recent = event.sessionID
      state.contexts.set(event.sessionID, context)
      // Filled in when the catalog answers — long before the turn is written,
      // which is when the price is read — so the request never waits on it.
      void listed().then((items) => {
        const entry = items.find((item) => item["providerID"] === model?.providerID && item["id"] === model?.id)
        if (!entry) return
        context.cost = entry["cost"]
        const limit = record(entry["limit"])?.["context"]
        if (typeof limit === "number" && limit > 0) context.contextLimit = limit
      })
      void refreshTitle(event.sessionID)
    })

    // Held by the request object, which is the one thing the two HTTP hooks
    // share: opencode hands the response hook the very request it handed the
    // request hook.
    const waiting = new Map<Request, Waiting>()
    const abandon = (key: Request, held: Waiting, why: string) => {
      waiting.delete(key)
      held.fail(new Error(why))
    }

    await ctx.session.hook("http.request", (event) => {
      if (!RECORDED.has(event.kind)) return
      const request = event.request
      if (request.method.toUpperCase() !== "POST") return
      const startedAt = Date.now()
      for (const [key, held] of waiting) {
        // A session sends one model request at a time, so one still waiting
        // when the next goes out failed before any response came back.
        if (held.sessionID === event.sessionID)
          abandon(key, held, "no response arrived: the request failed before the provider answered")
        else if (held.startedAt < startedAt - ABANDONED_MS)
          abandon(key, held, "no response arrived within ten minutes of the request")
      }
      let copy: Request
      try {
        copy = request.clone()
      } catch {
        return
      }
      let settle!: (response: Response) => void
      let fail!: (error: Error) => void
      const inflight = new Promise<Response>((resolve, reject) => {
        settle = resolve
        fail = reject
      })
      inflight.catch(() => undefined)
      const held: Waiting = { sessionID: event.sessionID, startedAt, settle, fail, wanted: true }
      waiting.set(request, held)
      // Read beside the request, not in front of it: the hook returns at once,
      // and the bytes are judged and parsed while they are already on the wire.
      void copy.text().then(
        (raw) => {
          // The same cheap gate as on 1.x: one scan of the bytes, before any parse.
          const body = core.MODEL_BODY.test(raw) ? core.modelBody(request.url, raw) : undefined
          if (!body) {
            held.wanted = false
            return
          }
          const wire: Wire = { method: "POST", url: request.url, raw, body, headers: core.redactHeaders(request.headers) }
          void core.capture(state, wire, inflight, startedAt, { session: event.sessionID }).catch(() => undefined)
        },
        () => {
          held.wanted = false
        },
      )
    })

    await ctx.session.hook("http.response", (event) => {
      const held = waiting.get(event.request)
      if (!held) return
      waiting.delete(event.request)
      if (!held.wanted) return
      // Cloned here, before the hook returns: opencode starts reading the body
      // as soon as it does, and a body already being read cannot be cloned.
      try {
        held.settle(event.response.clone())
      } catch (error) {
        held.fail(error instanceof Error ? error : new Error(message(error)))
      }
    })

    /** What a conversation is called: `session.created` and `session.renamed` carry it. */
    const controller = new AbortController()
    void (async () => {
      for await (const incoming of ctx.event.subscribe({ signal: controller.signal })) {
        const data = record(record(incoming)?.["data"]) ?? record(record(incoming)?.["properties"]) ?? {}
        const info = record(data["info"])
        const id = str(info?.["id"]) ?? str(data["sessionID"])
        const title = str(info?.["title"]) ?? str(data["title"])
        if (id && title) state.titles.set(id, title)
      }
    })().catch(() => undefined)

    /** A resumed conversation is named already; nothing will announce it again. */
    async function refreshTitle(sessionID: string) {
      if (state.titles.has(sessionID)) return
      try {
        const info = record(await ctx.session.get({ sessionID }))
        const title = str(info?.["title"]) ?? str(record(info?.["data"])?.["title"])
        if (title) state.titles.set(sessionID, title)
      } catch {
        // best effort: the session events will name it
      }
    }

    return () => {
      controller.abort()
      // Unloaded with requests in flight: their responses go to the next copy's
      // hooks, which never saw them go out. Write them down now, as unanswered.
      for (const [key, held] of waiting) abandon(key, held, "the audit plugin was unloaded while this request was in flight")
    }
  },
}
