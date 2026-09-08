// A wire log for LLM provider traffic.
//
// One file per model request, holding exactly two things: what went out and
// what came back, verbatim. It answers "what did opencode send this turn, and
// what did the provider reply?", and answers it without anything to join: no
// reuse diff, no piece store, no summaries, no accounting of its own beyond
// what the provider itself reported.
//
// That is not a smaller audit, it is the same one paid for elsewhere. Every
// second-order reading — how the prompt splits across system, tools and
// messages, how far this turn is still the previous one, what each tool's
// output costs on every turn after the one it arrived in — is a reading of the
// body, and the body is here whole. `harness-audit-viewer` derives all of them
// from it, once, on the turn you open, instead of on every request the agent
// makes.
//
// Self-contained: drop it in `.opencode/plugins/` (project) or
// `~/.config/opencode/plugins/` (global) of a stock opencode install.
//
// Enable with `OPENCODE_LLM_AUDIT=1` (or point `OPENCODE_LLM_AUDIT_DIR` at a
// directory), or from config:
//
//   { "plugin": [["./llm-audit.ts", { "dir": "/tmp/audit" }]] }
//
// A "turn" is one HTTP request to the model. A single thing you typed usually
// spans several, because each tool call round-trips again.
//
//   <dir>/<timestamp>-<session>/turn_001.json
//   <dir>/<timestamp>-<session>/turn_002.json
//
// and each of those is:
//
//   { turn, session, agent, provider, model, started_at, elapsed_seconds, ...,
//     "usage":    { input, cache_read, cache_write, output, prompt, total, source },
//     "reply":    { text, reasoning, tool_calls, stop_reason },
//     "request":  { method, url, dialect, chars, cache, body, headers },
//     "response": { status, id, model, finish_reason, request_id, bytes, headers } }
//
// That layout is the one `harness-audit-viewer/viewer.html` reads, and it is
// the reason the bill and the answer sit at the top of the file rather than
// inside `response`: the viewer keys the whole decoded format off
// `request.body`, and reads `usage`, `reply`, `cache_hit_rate`, `error`,
// `incomplete` and `note` off the turn itself. Nothing here exists only for the
// viewer — every field is one an audit asks for — but where a name or a
// position was free, it is the one the viewer already knows.
//
// The request keeps every byte that went out, in `body`, exactly as it was
// sent — the system prompt, the tool schemas, every message, the sampling
// settings and the cache markers, in the dialect's own shape rather than a
// redistribution of it. The repetition between turns is the point rather than
// noise: the model holds no state, so each turn resends the whole conversation,
// and that is precisely what the prompt cache is charging for. Beside the body
// go the three things a reader would otherwise have to derive, and only those:
//
//   dialect    which API the body speaks — "anthropic", "openai-chat",
//              "openai-responses", "gemini", "bedrock" —, read off its keys and
//              not off the URL: gateways and proxies rewrite paths, the body
//              keys are the contract
//   cache      what the request asked to cache, under one name: the `key`
//              (`prompt_cache_key`, `cachedContent`) or the `breakpoints`
//              (Anthropic's `cache_control`, Bedrock's positional `cachePoint`),
//              each saying which part and which index it sits at
//   chars      the characters the body occupied on the wire, which is what shows
//              the prompt growing turn after turn. Taken from the bytes, not
//              measured: the split across the three parts is a reading of the
//              body, and the body is right there to read
//
// The recorder runs inside every `fetch` the process makes, so the cost of
// deciding *not* to record is paid far more often than the cost of recording.
// Hence the first rule the code keeps: nothing is parsed, serialized or walked
// until a scan of the raw bytes says this request is worth it. The cache markers
// are then walked in the parsed structure rather than in a second copy of it as
// text, and the size of the prompt is read off the string instead of summed.
//
// The second rule is that none of it happens before the request does. The gate
// is the method and one scan of the bytes; then the call goes out, and only then
// is the body parsed, the folder opened and the turn described — beside the
// request rather than in front of it. Nothing opencode is waiting on waits on
// this file.
//
// The third is that bytes are never written twice. The body goes into the turn
// file exactly as it went onto the wire, spliced in as the string it already
// was, because serializing the parse of a body is writing the same bytes a
// second time — and in a second arrangement, which is not what was sent.
//
// The response is the opposite: the frames are folded into the answer they
// describe, because a stream is the same reply written twice, in pieces, in a
// dialect. What is kept is what an audit asks of a reply, under one set of
// names whatever the provider:
//
//   id, model     the response's own id, and the model actually served, which
//                 is not always the model that was asked for
//   finish_reason "stop" | "length" | "tool-calls" | "content-filter" |
//                 "error" | "unknown" — opencode's own vocabulary (see
//                 `packages/llm/src/schema/ids.ts`). The provider's own word
//                 for it is on the reply, as `reply.stop_reason`, once
//   usage         the token bill, normalized so the prompt counters never
//                 overlap: prompt = input + cache_read + cache_write, and
//                 reasoning is part of output rather than added to it. The
//                 provider's own usage object is kept verbatim as `native`,
//                 so a mapping that is wrong is still a mapping you can redo
//   reply         what the model said: text, reasoning, the tool calls with
//                 their ids, names and parsed arguments, and why it stopped.
//                 The tool call id is the join with the tool result that comes
//                 back in the next turn's request
//
// `raw` comes back only when none of that could be decoded, so a dialect this
// file does not know still leaves its bytes behind rather than an empty turn.
// A rejected request keeps its error body verbatim, where the reply would be.
//
// Credentials in headers and query params are redacted; bodies are stored
// verbatim, so treat the directory as sensitive.
//
// How it hooks in: it wraps the process's `fetch` and records the calls whose
// shape says "model request" (see `modelBody`), so it covers every provider —
// the bytes recorded are the bytes on the wire. The one hook it uses,
// `chat.params`, only reads the session id and never writes to its output, so
// arming this cannot change what opencode does.
import type { Plugin, PluginOptions } from "@opencode-ai/plugin"
import fs from "node:fs"
import os from "node:os"
import path from "node:path"

const REDACTED = "<redacted>"
const SENSITIVE_NAME =
  /authorization|api[-_]?key|access[-_]?token|refresh[-_]?token|id[-_]?token|token|secret|credential|cookie|signature/i
const SHORT_QUERY_NAME = /^(key|sig)$/i
// Headers that only look credential-shaped. `x-ratelimit-remaining-tokens` and
// `anthropic-ratelimit-input-tokens-limit` are counters, and the rule above
// matches the bare word `token` inside them — redacting the very numbers that
// say how much room was left.
const PUBLIC_NAME = /ratelimit|rate-limit|retry-after|token-count/i
const TEXT_CONTENT = /^(text\/|application\/(json|x-ndjson|jsonl|.*\+json))/i

// The two spellings of a cache marker, in one pass over the bytes. A body that
// carries neither is scanned to its end, and that is the common case — all of
// the OpenAI and Gemini traffic — so testing the two separately would scan the
// whole prompt twice to conclude the same nothing.
const CACHE_MARK = /"cache(_control|Point)"/

// Paths the model APIs use: OpenAI-style completions and responses, Anthropic
// messages, Gemini generateContent, Bedrock converse/invoke, Cohere-style chat.
const MODEL_PATH =
  /(chat\/completions|\/completions|\/responses|\/messages|generatecontent|\/converse|\/invoke|\/predict|\/chat)(\b|$)/i
/**
 * The cheap gate: a POST whose bytes spell none of these is not a model call,
 * and is turned away before anything parses it.
 *
 * Two things in one pass, because a body that fails is scanned to its end and
 * the traffic that fails is the common case — a second scan would double the
 * cost of every request this file does not want. The first alternative is the
 * key that means "this is a prompt", whatever the dialect calls it; the second
 * is the model named in the body, which is what still catches the OpenAI
 * Responses dialect (it names the model but calls its prompt `input`) and a
 * gateway that rewrote the path out of recognition.
 */
const MODEL_BODY = /"(messages|contents|system_instruction|systemInstruction)"\s*:|"model"\s*:\s*"/

// Turns that arrive without a session id still have to be written somewhere.
const NO_SESSION = "sem-sessao"

// The stand-in `request.body` carries through the turn object, so the bytes that
// went out can be spliced into the file exactly as they were sent instead of
// being serialized a second time from the parse of themselves. Random per
// process and carrying a NUL, so no body can contain it by accident.
const BODY_MARK = `\u0000llm-audit:body:${Math.random().toString(36).slice(2)}\u0000`

// The folder's index: one line per turn, next to the turns themselves.
const INDEX_FILE = "turns.json"

// Where a stream stops being latency and starts being generation: the first
// frame carrying a piece of the answer. Matched on the bytes rather than on
// parsed frames because the point is to know *when* it arrived, and the parse
// only happens once the whole body is in. Deliberately loose — every dialect
// spells its delta differently, and being a frame early or late costs a few
// milliseconds on a number whose interesting range is hundreds.
const FIRST_TOKEN = /content_block_delta|"delta"\s*:|\.delta"|"candidates"/

// The provider's own id for the request, under the six spellings the providers
// use for it (`packages/llm/src/route/executor.ts:80`). It is what you quote
// when you open a ticket with them.
const REQUEST_ID = ["x-request-id", "request-id", "x-amzn-requestid", "x-amz-request-id", "x-goog-request-id", "cf-ray"]

// The rate limit headers, in the two shapes the providers publish them: OpenAI
// puts what is limited last, Anthropic puts it in the middle
// (`executor.ts:112`). Both end up under the same three names here.
const OPENAI_LIMIT = /^x-ratelimit-(limit|remaining|reset)-(.+)$/
const ANTHROPIC_LIMIT = /^anthropic-ratelimit-(.+)-(limit|remaining|reset)$/

// The statuses the provider is telling you to come back later on, rather than
// that the request was wrong (`executor.ts:90`).
const RETRYABLE = new Set([429, 503, 504, 529])

// Where the providers put their token counts. Cohere hangs them off `meta`.
const USAGE_KEY = /^(usage|usage_?metadata|billed_units|token_?usage)$/i

// Keys worth reading text out of, for the Bedrock blocks that carry their
// reasoning under a shape of their own.
const TEXT_KEYS = ["text", "thinking", "reasoning", "content", "parts", "output", "result", "response", "summary"]
const CODE_KEYS = ["input", "arguments", "args", "functionCall", "functionResponse", "toolUse", "toolResult"]

/**
 * Every provider's word for why the model stopped, mapped onto opencode's own
 * six (`packages/llm/src/schema/ids.ts`). Keyed lowercase, because Gemini
 * shouts its reasons and the rest do not.
 *
 * The point of normalizing is that "was this answer cut off?" is one question,
 * and `max_tokens`, `length`, `MAX_TOKENS` and `incomplete` are four ways of
 * saying yes.
 */
const FINISH = new Map([
  // Anthropic, and Bedrock Converse.
  ["end_turn", "stop"],
  ["stop_sequence", "stop"],
  ["max_tokens", "length"],
  ["tool_use", "tool-calls"],
  ["refusal", "content-filter"],
  ["content_filtered", "content-filter"],
  ["guardrail_intervened", "content-filter"],
  // OpenAI chat completions.
  ["stop", "stop"],
  ["length", "length"],
  ["tool_calls", "tool-calls"],
  ["function_call", "tool-calls"],
  ["content_filter", "content-filter"],
  // OpenAI Responses, which reports the response's status rather than a reason.
  ["completed", "stop"],
  ["incomplete", "length"],
  ["failed", "error"],
  ["cancelled", "error"],
  // Gemini.
  ["max_tokens_reached", "length"],
  ["safety", "content-filter"],
  ["recitation", "content-filter"],
  ["blocklist", "content-filter"],
  ["prohibited_content", "content-filter"],
  ["spii", "content-filter"],
  ["image_safety", "content-filter"],
  ["malformed_function_call", "error"],
  ["other", "unknown"],
])

type FetchLike = typeof globalThis.fetch

/**
 * One outgoing model request, as the recorder sees it: the bytes that went out
 * and the body they parse to, parsed once at the door and carried from there.
 */
type Wire = {
  method: string
  url: string
  raw: string
  body: Record<string, unknown>
  headers: Record<string, string>
}

type Settings = { enabled: boolean; dir: string }

/**
 * One request's token bill, normalized across the dialects.
 *
 * `input` excludes what was read from or written to the cache, so the three
 * prompt-side counters never overlap and `prompt` is their sum. `reasoning` is
 * the thinking part of `output`, not an addition to it — that is how it is
 * billed.
 *
 * A counter the dialect does not report is left out rather than written as
 * zero: a model with no prompt cache and a cache that missed are opposite
 * findings, and a `0` cannot tell them apart.
 */
type Usage = {
  input: number
  cache_read?: number
  cache_write?: number
  output: number
  reasoning?: number
  prompt: number
  total: number
  // Where these numbers came from. Always "provider" here: this file only ever
  // transcribes a count the provider reported, and drops the `usage` object
  // entirely when it reported none — it never estimates. Written all the same,
  // because a reader that cannot tell a measured number from a guessed one
  // treats every log as measured.
  source: "provider"
  // The second-order counters: the ones that qualify the bill rather than being
  // it. Absent when the provider reported none of them.
  details?: {
    // The TTL split of the cache write, when the provider reports it: the two
    // are billed at different multipliers.
    cache_write_5m?: number
    cache_write_1h?: number
    // The provider's own total, and how far it sits from the sum above. Both
    // written only when the two disagree — silence means they agree, and a
    // number here means a counter this file did not read, so the total above is
    // short by exactly that much.
    reported_total?: number
    total_mismatch?: number
  }
  // The provider's own usage object, exactly as it reported it. The normalized
  // fields above are a reading of this; this is the evidence.
  native: Record<string, unknown>
}

/** What the model said, assembled out of however many frames carried it. */
type Reply = {
  text: string
  reasoning: string
  // `id` is the provider's tool call id: what the next turn's tool result
  // points back at, and the join between what the model asked for and what
  // running it actually did.
  tool_calls: { id?: string; name?: string; arguments?: unknown }[]
  // The provider's own word for why it stopped, before normalization.
  stop_reason?: string
}

/** Which response this was, folded out of whatever frame first named it. */
type Identity = { id?: string; model?: string }

/**
 * The token bill under construction. `found` is not `merged` being non-empty: a
 * `usage` object the provider sent and this file could read nothing out of is
 * still a bill that was reported, and a different finding from none at all.
 */
type Bill = { merged: Record<string, unknown>; found: boolean }

/** A reply under construction, while the frames are still arriving. */
type Draft = {
  text: string[]
  reasoning: string[]
  calls: Map<string, { id?: string; name?: string; args: string }>
  stop?: string
}

/** How the answer arrived, in milliseconds from the request going out. */
type Timing = {
  first_byte_ms: number
  // Only for a stream: a response delivered whole has no such moment, and a
  // number there would invite comparing it with one that means something else.
  first_token_ms?: number
  last_byte_ms: number
  stream_ms: number
  chunks: number
}

/**
 * What the hooks know and the wire does not: which agent is running, which model
 * was asked for, at what reasoning effort, and what that model costs.
 *
 * Read structurally out of the hook's argument rather than off its declared
 * type, because the generated types and what the session actually passes have
 * drifted before (`variant` is on the value and not on `UserMessage`).
 */
type Context = {
  agent?: string
  providerID?: string
  modelID?: string
  variant?: string
  cost?: Record<string, unknown>
  contextLimit?: number
}

/** One conversation, and one folder. */
type Session = {
  id: string
  dir: string
  turns: number
  // The `turns.json` index in memory, read back from the file when the folder is
  // reopened so a restart appends to the index instead of starting it over.
  rows: Record<string, unknown>[]
  // Whether a write of the index is already queued, so a burst of turns costs
  // one rewrite of the file instead of one per turn.
  indexQueued?: boolean
}

type State = {
  settings: Settings
  // The session `chat.params` last named, for requests that do not say which
  // session they belong to on the wire.
  recent?: string
  // What the hooks learned, by session id.
  contexts: Map<string, Context>
  // Child session -> parent, learned from `x-parent-session-id`, so a subagent's
  // turns are filed with the conversation whose work they are.
  parents: Map<string, string>
  // How many things the user has typed, per session id, and the id of the last
  // one. Kept here and not on the folder because the folder a session writes to
  // is only known once a request goes out and names its parent, while these are
  // known one hook earlier — a subagent's first message arrives before anything
  // has said whose subagent it is.
  requests: Map<string, number>
  messages: Map<string, string>
  sessions: Map<string, Session>
  // Turns whose response has not landed yet, by path, so a request still in
  // flight when the process exits leaves a record all the same.
  pending: Map<string, { head: Record<string, unknown>; request: unknown; raw: string }>
  queue: Promise<void>
}

export const LLMAuditPlugin: Plugin = async (_input, options) => {
  const settings = resolve(options)
  if (!settings.enabled) return {}
  const state = shared(settings)
  install(state)

  // The two hooks exist only to learn what the wire cannot carry. Neither writes
  // to its `output`, so arming the audit cannot change what opencode does.
  return {
    async "chat.params"(input) {
      state.recent = input.sessionID
      const model = record(input.model)
      state.contexts.set(input.sessionID, {
        agent: input.agent,
        providerID: str(model?.["providerID"]),
        modelID: str(model?.["id"]),
        variant: variantOf(input.message),
        cost: record(model?.["cost"]),
        contextLimit: num(record(model?.["limit"])?.["context"]) || undefined,
      })
    },

    /**
     * A new thing the user typed: the boundary the turns are grouped by. No
     * folder is opened here — which conversation this session writes to is not
     * settled until a request goes out and names its parent.
     */
    async "chat.message"(input) {
      state.requests.set(input.sessionID, (state.requests.get(input.sessionID) ?? 0) + 1)
      if (input.messageID) state.messages.set(input.sessionID, input.messageID)
    },
  }
}

/**
 * The reasoning variant the message asked for, or "default" when it asked for
 * none — and the two are not the same finding: "default" says a variant was
 * resolved and it was none, where a missing field says the log predates this
 * being recorded. A reader that cannot tell them apart reports every old turn
 * as running at default effort.
 */
function variantOf(message: unknown): string {
  const variant = record(record(message)?.["model"])?.["variant"]
  return typeof variant === "string" && variant ? variant : "default"
}

/**
 * The conversation a session's turns belong to: itself, or whoever spawned it.
 * A subagent can spawn a subagent; the guard is against a cycle, not depth.
 */
function rootOf(state: State, session: string | undefined): string {
  if (!session) return NO_SESSION
  let current = session
  for (let step = 0; step < 8; step++) {
    const parent = state.parents.get(current)
    if (!parent || parent === current) break
    current = parent
  }
  return current
}

function resolve(options?: PluginOptions): Settings {
  const env = process.env
  const flag = env["OPENCODE_LLM_AUDIT"]
  const dir = (options?.["dir"] as string) ?? env["OPENCODE_LLM_AUDIT_DIR"]
  return {
    enabled:
      (options?.["enabled"] as boolean) ??
      (flag === undefined ? Boolean(dir) : flag !== "" && flag !== "0" && flag.toLowerCase() !== "false"),
    dir:
      dir ??
      path.join(
        env["XDG_DATA_HOME"] ?? path.join(os.homedir(), ".local", "share"),
        "opencode",
        "log",
        "llm-audit",
      ),
  }
}

/** One state per process, however many plugin instances opencode creates. */
function shared(settings: Settings): State {
  const global = globalThis as Record<string, unknown>
  const existing = global["__opencodeLLMAuditState"] as State | undefined
  if (existing) return existing
  const state: State = {
    settings,
    contexts: new Map(),
    parents: new Map(),
    requests: new Map(),
    messages: new Map(),
    sessions: new Map(),
    pending: new Map(),
    queue: Promise.resolve(),
  }
  global["__opencodeLLMAuditState"] = state
  return state
}

/** Wraps the process's fetch, once. */
function install(state: State) {
  const global = globalThis as Record<string, unknown>
  if (global["__opencodeLLMAuditFetch"]) return
  const original = globalThis.fetch
  const patched = (input: Parameters<FetchLike>[0], init?: Parameters<FetchLike>[1]) => {
    // The cheap gate, in the order that turns traffic away for the least work:
    // the method first, because a GET can never be a model call and `rawBody`
    // would decode a whole binary body to find that out; then one scan of the
    // bytes for the key that means "this is a prompt".
    const method = (init?.method ?? methodOf(input) ?? "GET").toUpperCase()
    if (method !== "POST") return original(input, init)
    const raw = rawBody(init?.body)
    if (typeof raw !== "string" || !MODEL_BODY.test(raw)) return original(input, init)

    // Past here nothing is allowed to hold the request back. The clock is read,
    // the call goes out, and the body is parsed while it is already on its way:
    // the parse, the redaction, the folder and the three synchronous `fs` calls
    // that open it all used to sit between opencode and the provider. What is
    // recorded is unchanged — the tests below are the same ones, on the same
    // parsed keys, only now they run beside the request instead of before it.
    const started = Date.now()
    const url = urlOf(input)
    const inflight = original(input, init)

    const body = modelBody(url, raw)
    if (!body) return inflight
    // Only a model call pays for the headers: redacting them builds a `Headers`
    // and walks it, and by here the traffic is the traffic worth recording.
    const request: Wire = { method, url, raw, body,
                            headers: redactHeaders(init?.headers ?? headersOf(input)) }
    return capture(state, request, inflight, started)
  }
  // Carries over anything the runtime hangs off fetch (Bun's `preconnect`).
  global["__opencodeLLMAuditFetch"] = true
  globalThis.fetch = Object.assign(patched, original) as FetchLike
  // Written synchronously: an exit handler is too late for the queue.
  process.on("exit", () => flush(state))
}

/**
 * The parsed body of a model call, or nothing — which is also the answer to
 * "should this request be recorded at all?".
 *
 * A model call POSTs a JSON body shaped like a prompt. Both halves matter: the
 * shape alone would catch opencode's own APIs, and the path alone would catch
 * unrelated traffic to the same hosts.
 *
 * The order is the point. `fetch` is wrapped for the whole process, so this runs
 * on every call the process makes, and the parse is the expensive step —
 * proportional to a body that, for a warm conversation, is most of a megabyte.
 * So the bytes are scanned first, in `install`, and only a body that already
 * looks like a prompt or already names a model ever reaches here to be parsed.
 * Everything else — telemetry, opencode's own APIs, whatever else the process
 * talks to — is turned away for the cost of one scan of the bytes, and the
 * request that does reach here is already on the wire by the time it does.
 */
function modelBody(url: string, raw: string) {
  const body = record(json(raw))
  if (!body) return undefined
  const shape =
    "messages" in body || "contents" in body || "prompt" in body || ("input" in body && "model" in body)
  if (!shape) return undefined
  const named = MODEL_PATH.test(pathOf(url)) || typeof body["model"] === "string"
  return named ? body : undefined
}

async function capture(state: State, request: Wire, inflight: Promise<Response>, started: number) {
  // The bytes that went out, kept as the string they arrived as: it is what the
  // turn file writes back out verbatim, and holding it instead of the parsed
  // body lets that object graph go once the head below has been read off it.
  const raw = request.raw
  // opencode names the session on every request it makes, and a subagent also
  // names whoever spawned it — its turns are that conversation's work, so they
  // are filed in that folder and numbered in the same sequence.
  const wire = sessionOf(state, request)
  const parent = headerOf(request, "x-parent-session-id")
  if (wire !== NO_SESSION && parent) state.parents.set(wire, parent)
  const sess = openSession(state, rootOf(state, wire))
  if (!sess) return inflight
  const turn = ++sess.turns
  const file = path.join(sess.dir, `turn_${String(turn).padStart(3, "0")}.json`)
  const context = state.contexts.get(wire) ?? (state.recent ? state.contexts.get(state.recent) : undefined)
  const sent = readRequest(request)
  // What the hooks and the wire know before the answer comes back. A field the
  // hooks never filled in is left out rather than written as null: the absence
  // says "not known", where a value would claim something.
  const head = {
    turn,
    session: wire === NO_SESSION ? undefined : wire,
    parent_session: parent,
    agent: context?.agent,
    provider: context?.providerID ?? hostOf(request.url),
    model: context?.modelID ?? modelOf(request),
    variant: context?.variant,
    request_index: state.requests.get(wire),
    // From the hook, or from the header the opencode-hosted providers carry it
    // in — the same id, by two roads.
    user_message: state.messages.get(wire) ?? headerOf(request, "x-opencode-request"),
    started_at: new Date(started).toISOString(),
  }
  state.pending.set(file, { head, request: sent, raw })

  // The rest of the head is only knowable once the answer is in: how long it
  // took, and which pricing tier this prompt fell into.
  const finish = (response: Record<string, unknown>, timing?: Timing) =>
    write(
      state,
      sess,
      file,
      {
        ...head,
        elapsed_seconds: round2((Date.now() - started) / 1000),
        timing,
        ...ratesOf(context, num(record(response["usage"])?.["prompt"])),
        context_limit: context?.contextLimit,
      },
      sent,
      response,
      raw,
    )

  try {
    const response = await inflight
    const status = response.status
    const headers = redactHeaders(response.headers)
    const type = response.headers.get("content-type") ?? ""
    // Read from a clone, so the response opencode gets is untouched. The turn is
    // written once the body has finished arriving, off the request's own path.
    let read: Promise<{ buffer?: ArrayBuffer; timing?: Timing }>
    try {
      read = timedBody(response.clone(), started).catch(() => ({}))
    } catch {
      read = Promise.resolve({})
    }
    // `headers` last: what the answer was comes first in the file, and the
    // transport noise after it.
    void read.then(
      ({ buffer, timing }) => finish({ status, ...describeResponse(type, buffer, status, headers), headers }, timing),
      (error) => finish({ status, error: describeError(error), headers }),
    )
    return response
  } catch (error) {
    finish({ error: describeError(error) })
    throw error
  }
}

/**
 * The response body, with the clock kept while it arrives.
 *
 * A buffered read resolves only once the stream has ended, so every moment
 * inside it is lost — and a streamed answer is nearly all inside it. Reading
 * the clone frame by frame costs one pass over bytes that were being decoded
 * anyway, and buys the measurement that separates a slow provider from a long
 * answer.
 */
async function timedBody(response: Response, started: number): Promise<{ buffer?: ArrayBuffer; timing?: Timing }> {
  const body = response.body
  // A body already in hand, or a runtime that does not expose a stream: fall
  // back to the buffered read, and report no timing rather than a made-up one.
  if (!body) return { buffer: await response.arrayBuffer().catch(() => undefined) }

  const reader = body.getReader()
  const chunks: Uint8Array[] = []
  const decoder = new TextDecoder("utf-8", { fatal: false })
  let size = 0
  let first: number | undefined
  let token: number | undefined
  let last = 0
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    if (!value?.byteLength) continue
    const at = Date.now() - started
    if (first === undefined) first = at
    last = at
    chunks.push(value)
    size += value.byteLength
    // `stream: true` so a multi-byte character split across two chunks does not
    // come back as a replacement character and hide the marker behind it.
    if (token === undefined && FIRST_TOKEN.test(decoder.decode(value, { stream: true }))) token = at
  }

  const buffer = new Uint8Array(size)
  let offset = 0
  for (const chunk of chunks) {
    buffer.set(chunk, offset)
    offset += chunk.byteLength
  }
  return {
    buffer: buffer.buffer,
    timing:
      first === undefined
        ? undefined
        : {
            first_byte_ms: first,
            // Only meaningful when the answer came in pieces. One chunk is a
            // response assembled before it was sent, whose first token never had
            // a moment of its own.
            first_token_ms: chunks.length > 1 ? token : undefined,
            last_byte_ms: last,
            stream_ms: last - first,
            chunks: chunks.length,
          },
  }
}

/**
 * The tarif that applied to this turn, already resolved to its tier.
 *
 * opencode picks the tier by the *inclusive* prompt — `cost.tiers` first (the
 * largest whose `size` the prompt passed), then `experimentalOver200K` over
 * 200.000 (`session.ts:379`). Writing only the base rate would misprice every
 * long conversation, and the prompt is known by the time the turn is written,
 * so the tier is resolved here rather than left for the reader to guess.
 *
 * The rate is written, never the amount: the amount is arithmetic anyone can
 * redo, while the rate is a fact about the moment the request went out.
 */
function ratesOf(context: Context | undefined, prompt: number) {
  const cost = context?.cost
  if (!cost) return {}
  let rates = cost
  let tier = "base"
  const tiers = arrayOf(cost["tiers"])
    .map((item) => record(item))
    .filter((item): item is Record<string, unknown> => {
      const size = num(record(item?.["tier"])?.["size"])
      return record(item?.["tier"])?.["type"] === "context" && prompt > size
    })
    .sort((a, b) => num(record(b["tier"])?.["size"]) - num(record(a["tier"])?.["size"]))
  const over = record(cost["experimentalOver200K"])
  if (tiers[0]) {
    rates = tiers[0]
    tier = `context>${num(record(tiers[0]["tier"])?.["size"])}`
  } else if (over && prompt > 200_000) {
    rates = over
    tier = "over_200k"
  }
  const cache = record(rates["cache"])
  return {
    rates_per_million: {
      input: num(rates["input"]),
      output: num(rates["output"]),
      cache_read: num(cache?.["read"]),
      cache_write: num(cache?.["write"]),
    },
    rates_tier: tier,
  }
}

/**
 * The turn file, assembled.
 *
 * What the answer *was* sits at the top of the file, beside what was asked —
 * the bill, the reply, the failure —, and only the transport stays under
 * `response`: the status, the ids, the bytes, the headers. That is the order a
 * reader wants, and it is the order `harness-audit-viewer` reads, which looks
 * for `usage`, `reply`, `cache_hit_rate`, `error`, `incomplete` and `note` on
 * the turn itself.
 *
 * A field the response never carried is left out rather than written as null:
 * the absence says "did not happen", where a null claims it did and came back
 * empty. A request that never became a response leaves no `response` at all.
 */
function turnOf(head: Record<string, unknown>, request: unknown, response: Record<string, unknown>) {
  const { usage, reply, error, incomplete, note, ...transport } = response
  const bill = record(usage)
  return {
    ...head,
    ...(bill
      ? {
          usage: bill,
          // `cache_read / prompt`, written once and on the turn rather than
          // inside the usage object it is derived from.
          cache_hit_rate:
            num(bill["prompt"]) && bill["cache_read"] !== undefined
              ? round2(num(bill["cache_read"]) / num(bill["prompt"]))
              : undefined,
        }
      : {}),
    ...(reply !== undefined ? { reply } : {}),
    ...(error !== undefined ? { error } : {}),
    ...(incomplete !== undefined ? { incomplete, note } : {}),
    request,
    ...(Object.keys(transport).length ? { response: transport } : {}),
  }
}

/** The turn file, and the line about it in the folder's index. */
function write(
  state: State,
  sess: Session,
  file: string,
  head: Record<string, unknown>,
  request: unknown,
  response: Record<string, unknown>,
  raw: string,
) {
  state.pending.delete(file)
  // The row is built from the response as it came, before `turnOf` splits it:
  // the index is the plugin's own, and nothing in it moved.
  insertRow(sess.rows, rowOf(head, path.basename(file), request, response))
  state.queue = state.queue
    .then(() => fs.promises.writeFile(file, text(turnOf(head, request, response), raw)))
    .catch(() => undefined)
  queueIndex(state, sess)
}

/**
 * The index, written once per drain rather than once per turn.
 *
 * The file is the whole list every time, so writing it after every turn rewrites
 * a file that grows with the conversation — quadratic over a session, to produce
 * a state only the last one of which anybody reads. A row is in `sess.rows`
 * before this is called, so a write already queued and not yet started will
 * carry it, and a burst of turns collapses into a single write. `flush` writes
 * it once more at exit, which is what makes the final state exact.
 */
function queueIndex(state: State, sess: Session) {
  if (sess.indexQueued) return
  sess.indexQueued = true
  state.queue = state.queue
    .then(() => {
      // Cleared before the list is read, so a row landing during the write
      // queues another one rather than being left out of the file.
      sess.indexQueued = false
      return fs.promises.writeFile(path.join(sess.dir, INDEX_FILE), text(index(sess)))
    })
    .catch(() => undefined)
}

/**
 * Keeps the rows in turn order as they land, so the index never has to be
 * sorted to be written. Turns mostly finish in the order they started, so the
 * scan back from the end stops immediately; a turn that overtook another walks
 * back only as far as it has to.
 */
function insertRow(rows: Record<string, unknown>[], row: Record<string, unknown>) {
  const turn = num(row["turn"])
  let at = rows.length
  while (at > 0 && num(rows[at - 1]?.["turn"]) > turn) at--
  rows.splice(at, 0, row)
}

/**
 * One line per turn: the head, plus the numbers anyone reads first.
 *
 * Enough to price the whole conversation without opening a single turn file —
 * which is why the rate travels on the line too, since a conversation mixes
 * models (the `title` agent runs on a small one, at another price).
 */
function rowOf(head: Record<string, unknown>, file: string, request: unknown, response: Record<string, unknown>) {
  const usage = record(response["usage"])
  const reply = record(response["reply"])
  const timing = record(head["timing"])
  return {
    turn: head["turn"],
    file,
    started_at: head["started_at"],
    seconds: head["elapsed_seconds"],
    session: head["session"],
    parent_session: head["parent_session"],
    agent: head["agent"],
    provider: head["provider"],
    model: head["model"],
    variant: head["variant"],
    request_index: head["request_index"],
    status: response["status"],
    finish_reason: response["finish_reason"],
    first_token_ms: timing?.["first_token_ms"],
    // Written as an explicit `null` when the provider reported nothing, never
    // left out: a reader summing these lines must not read "not reported" as
    // "zero tokens". `native` stays out — the line is for scanning.
    usage: usage ? { ...usage, native: undefined } : null,
    rates_per_million: head["rates_per_million"],
    rates_tier: head["rates_tier"],
    chars: num(record(request)?.["chars"]) || undefined,
    tools: reply ? arrayOf(reply["tool_calls"]).map((call) => record(call)?.["name"]) : undefined,
    // A turn that never came back may have cost tokens nobody can count, and a
    // turn the provider rejected bought nothing. Both have to be visible as
    // something other than a zero.
    incomplete: response["incomplete"] === true ? true : undefined,
    error: response["error"] !== undefined ? true : undefined,
  }
}

/** The folder's index, turns in order however the answers came back. */
function index(sess: Session) {
  return {
    session: sess.id === NO_SESSION ? undefined : sess.id,
    updated_at: new Date().toISOString(),
    turns: sess.rows,
  }
}

/** The index a previous run left, so a restart appends rather than restarts. */
function readIndex(dir: string): Record<string, unknown>[] {
  try {
    const parsed = JSON.parse(fs.readFileSync(path.join(dir, INDEX_FILE), "utf8"))
    return arrayOf(record(parsed)?.["turns"]).filter((row): row is Record<string, unknown> => isRecord(row))
  } catch {
    return []
  }
}

/** Records the turns still in flight, so a request is never silently lost. */
function flush(state: State) {
  const touched = new Set<Session>()
  for (const [file, { head, request, raw }] of state.pending) {
    const response = {
      incomplete: true,
      note: "the process exited while this request was still in flight, so nothing came back to record",
    }
    const sess = [...state.sessions.values()].find((entry) => entry.dir === path.dirname(file))
    if (sess) {
      insertRow(sess.rows, rowOf(head, path.basename(file), request, response))
      touched.add(sess)
    }
    try {
      fs.writeFileSync(file, text(turnOf(head, request, response), raw))
    } catch {
      // Nothing useful to do at exit.
    }
  }
  state.pending.clear()
  // The index last, and only for the folders that changed: an exit handler is
  // too late for the queue, so these writes are synchronous.
  for (const sess of touched) {
    try {
      fs.writeFileSync(path.join(sess.dir, INDEX_FILE), text(index(sess)))
    } catch {
      // Nothing useful to do at exit.
    }
  }
}

/**
 * Whose conversation this request is. opencode names the session on every
 * request it makes; the header is exact, and the last session `chat.params`
 * named is the fallback for traffic that carries none.
 */
function sessionOf(state: State, request: { headers: Record<string, string> }) {
  return (
    headerOf(request, "x-session-id") ??
    headerOf(request, "x-session-affinity") ??
    headerOf(request, "x-opencode-session") ??
    state.recent ??
    NO_SESSION
  )
}

/**
 * The folder a session writes into: the one it already has if it has one, so
 * resuming a conversation appends to it rather than scattering the same
 * conversation over a folder per run. The turn numbering picks up from the
 * highest turn already on disk.
 */
function openSession(state: State, id: string): Session | undefined {
  const existing = state.sessions.get(id)
  if (existing) return existing
  try {
    const dir = sessionDir(state.settings.dir, id)
    fs.mkdirSync(dir, { recursive: true })
    const created: Session = {
      id,
      dir,
      turns: lastTurn(dir),
      rows: readIndex(dir),
    }
    state.sessions.set(id, created)
    return created
  } catch {
    return undefined
  }
}

/**
 * Where a conversation lives. The name leads with the time the session was
 * first seen, so sorting folder names reads them in the order the conversations
 * started, and ends with the session id, which is what finds it again.
 */
function sessionDir(base: string, id: string) {
  const suffix = short(id)
  const pattern = new RegExp(`^\\d{8}_\\d{6}-${escaped(suffix)}$`)
  for (const name of entriesOf(base).sort()) if (pattern.test(name)) return path.join(base, name)
  return path.join(base, `${stamp(new Date())}-${suffix}`)
}

function entriesOf(dir: string): string[] {
  try {
    return fs.readdirSync(dir)
  } catch {
    return []
  }
}

/** The highest turn already written to a folder. */
function lastTurn(dir: string) {
  let highest = 0
  for (const name of entriesOf(dir)) {
    const match = /^turn_(\d+)/.exec(name)
    if (match) highest = Math.max(highest, Number(match[1]))
  }
  return highest
}

function short(id: string) {
  if (id === NO_SESSION) return NO_SESSION
  return id.replace(/^ses_/, "").slice(0, 10) || NO_SESSION
}

function escaped(value: string) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
}

function stamp(date: Date) {
  const pad = (value: number) => String(value).padStart(2, "0")
  return [
    date.getFullYear(),
    pad(date.getMonth() + 1),
    pad(date.getDate()),
    "_",
    pad(date.getHours()),
    pad(date.getMinutes()),
    pad(date.getSeconds()),
  ].join("")
}

/**
 * The request: the body as it went out, and the little that has to be derived.
 *
 * The body itself is not carried here but stood in for: `BODY_MARK` marks the
 * place, and `text` puts the bytes back into it when the file is written. The
 * body was already in hand as the string that went on the wire, so serializing
 * the parse of it would be writing the same bytes a second time, in a second
 * arrangement — and it is the wire bytes, not a re-rendering of them, that this
 * file promises. It also lets the parsed object graph go here, rather than
 * being held in `pending` until the answer lands.
 *
 * The body is stored whole and untouched — the system prompt, the tool schemas,
 * every message, the sampling settings, the cache markers, in the dialect's own
 * shape. The repetition between turns is the point of the log, not noise: the
 * model holds no state, so each turn resends the whole conversation, and that is
 * precisely what the prompt cache is charging for.
 *
 * The three parts are located rather than moved. Splitting the body into
 * `system`/`tools`/`messages` and a leftover bag of settings was the older shape
 * here, and it cost twice: it wrote the same bytes in a second arrangement, and
 * the arrangement was lossy — the OpenAI dialects carry the system prompt as a
 * message with a privileged role, so lifting it out loses where it sat in the
 * list. Locating the parts is enough to measure them (`chars`) and to say where
 * the cache markers are (`cache`); a reader that wants the parts themselves
 * finds them in the body, by `dialect`.
 */
function readRequest(request: Wire) {
  const head = { method: request.method, url: redactUrl(request.url) }
  const body = record(request.body)
  const dialect = body ? dialectOf(body) : "unknown"
  // Nothing is derived about a dialect this file does not know: the body still
  // goes in whole, the way an undecodable response keeps its bytes.
  if (!body || dialect === "unknown")
    return { ...head, dialect, body: BODY_MARK, headers: request.headers }

  return {
    ...head,
    dialect,
    // The characters this turn put on the wire, taken from the bytes rather than
    // measured: `raw.length` is free and exact, where summing the parts costs a
    // serialization of the whole prompt and still misses the settings keys and
    // the JSON punctuation around everything. The split across system, tools and
    // messages is not written — nothing read it, and a reader who wants it has
    // the body right here to take it from.
    chars: request.raw.length,
    cache: cacheView(body, dialect, request.raw),
    body: BODY_MARK,
    headers: request.headers,
  }
}

/**
 * Which API dialect the body speaks. Checked by shape rather than by URL: the
 * same provider is reachable through gateways and proxies that rewrite paths,
 * but the body keys are the contract.
 */
function dialectOf(body: Record<string, unknown>) {
  if ("contents" in body) return "gemini"
  if ("toolConfig" in body || "inferenceConfig" in body || "additionalModelRequestFields" in body) return "bedrock"
  if ("input" in body && !("messages" in body)) return "openai-responses"
  if ("messages" in body) {
    if ("anthropic_version" in body || "system" in body) return "anthropic"
    const first = arrayOf(body["tools"])[0]
    if (isRecord(first) && "input_schema" in first) return "anthropic"
    return "openai-chat"
  }
  return "unknown"
}

/**
 * The system prompt, wherever the dialect keeps it.
 *
 * The OpenAI dialects carry it as an ordinary message with a privileged role —
 * `developer` in the Responses traffic opencode actually sends, which has no
 * `instructions` field at all — so it is lifted out of the message list. That
 * is what makes the three parts comparable across providers.
 */
function systemItems(body: Record<string, unknown>, dialect: string): unknown[] {
  if (dialect === "gemini") {
    const instruction = body["systemInstruction"] ?? body["system_instruction"]
    return instruction ? [instruction] : []
  }
  if (dialect === "openai-chat" || dialect === "openai-responses") {
    const items: unknown[] = []
    if (body["instructions"]) items.push(body["instructions"])
    for (const item of messageList(body, dialect)) if (isSystemRole(record(item)?.["role"])) items.push(item)
    return items
  }
  const system = body["system"]
  if (typeof system === "string") return system ? [system] : []
  return arrayOf(system)
}

/** The conversation, minus whatever was lifted into `system`. */
function messageItems(body: Record<string, unknown>, dialect: string): unknown[] {
  const list = messageList(body, dialect)
  if (dialect === "openai-chat" || dialect === "openai-responses")
    return list.filter((item) => !isSystemRole(record(item)?.["role"]))
  return list
}

function messageList(body: Record<string, unknown>, dialect: string): unknown[] {
  if (dialect === "gemini") return arrayOf(body["contents"])
  if (dialect === "openai-responses") return arrayOf(body["input"])
  return arrayOf(body["messages"])
}

function isSystemRole(role: unknown) {
  return role === "system" || role === "developer"
}

/** The tool schemas, out of the envelope each dialect wraps them in. */
function toolItems(body: Record<string, unknown>, dialect: string): unknown[] {
  if (dialect === "bedrock")
    return arrayOf(record(body["toolConfig"])?.["tools"]).map((tool) => unwrap(tool, "toolSpec"))
  const tools = arrayOf(body["tools"])
  if (dialect === "gemini") {
    const declarations: unknown[] = []
    for (const tool of tools) {
      const entry = record(tool)
      const list = entry?.["functionDeclarations"] ?? entry?.["function_declarations"]
      // Only when the wrapper holds the declarations and nothing else.
      if (Array.isArray(list) && entry && Object.keys(entry).length === 1) declarations.push(...list)
      else declarations.push(tool)
    }
    return declarations
  }
  return tools.map((tool) => unwrap(tool, "function"))
}

/**
 * `{type:"function", function:{...}}` -> the schema inside it.
 *
 * Only when the wrapper carries nothing but the wrapping, so what is dropped is
 * reconstructible from `dialect` alone. A wrapper with a field of its own is
 * left exactly as it came.
 */
function unwrap(tool: unknown, name: string) {
  const entry = record(tool)
  if (!entry || !isRecord(entry[name])) return tool
  return Object.keys(entry).some((field) => field !== name && field !== "type") ? tool : entry[name]
}

/**
 * What this request asked the provider to cache, in one shape.
 *
 * The four dialects say it four structurally different ways: Anthropic marks a
 * `cache_control` inside a tool, a system block or a content block; Bedrock
 * inserts a positional `cachePoint` element after what it wants cached; the
 * OpenAI dialects pass a `prompt_cache_key` and let the provider decide; Gemini
 * points at a `cachedContent` resource it uploaded earlier. The markers stay
 * where they were — this only says where they are, so "where does the cached
 * prefix end" is one question rather than four conventions to know.
 */
function cacheView(body: Record<string, unknown>, dialect: string, raw: string) {
  const key = str(body["prompt_cache_key"]) ?? str(body["cachedContent"]) ?? str(body["cached_content"])
  const breakpoints: { in: string; index: number; type: string; ttl?: string }[] = []
  // One scan of the bytes before any structure is located, let alone walked. A
  // body that never spells a marker has none to find — which is all of the
  // OpenAI and Gemini traffic, and every request that asked for no caching — and
  // for it neither the search below nor the work of finding the three parts to
  // search is done at all.
  if (CACHE_MARK.test(raw)) {
    const prompt: Record<string, unknown[]> = {
      system: systemItems(body, dialect),
      tools: toolItems(body, dialect),
      messages: messageItems(body, dialect),
    }
    for (const [part, items] of Object.entries(prompt))
      items.forEach((item, index) => {
        const marker = cacheOf(item)
        if (marker) breakpoints.push({ in: part, index, ...marker })
      })
  }
  // No key and no breakpoint is a request that asked for no caching at all,
  // which is a finding of its own — hence an absent field rather than an empty
  // one, so it reads as "nothing asked" and not as "nothing found".
  if (!key && !breakpoints.length) return undefined
  return { key, breakpoints: breakpoints.length ? breakpoints : undefined }
}

/**
 * The cache marker inside one prompt item, if it carries one.
 *
 * Walked over the parsed item and stopped at the first hit, rather than read out
 * of the item serialized: a message is worth kilobytes and a marker twenty
 * characters, and building the former to find the latter is the whole cost of
 * this file's cache view. The depth limit is against a cycle, not against
 * nesting — a marker sits shallow, on the block that carries it.
 */
function cacheOf(item: unknown, depth = 0): { type: string; ttl?: string } | undefined {
  if (depth > 6 || item === null || typeof item !== "object") return undefined
  if (Array.isArray(item)) {
    for (const child of item) {
      const found = cacheOf(child, depth + 1)
      if (found) return found
    }
    return undefined
  }
  const entry = item as Record<string, unknown>
  const control = record(entry["cache_control"])
  // Anthropic names the kind of cache it wants; a Bedrock cache point carries
  // only `"type":"default"`, which says nothing its position does not.
  if (control) return { type: str(control["type"]) ?? "ephemeral", ttl: str(control["ttl"]) }
  const point = record(entry["cachePoint"])
  if (point) return { type: "point", ttl: str(point["ttl"]) }
  for (const child of Object.values(entry)) {
    const found = cacheOf(child, depth + 1)
    if (found) return found
  }
  return undefined
}

/**
 * The response, read.
 *
 * The frames are parsed to be folded into the reply and the usage, and then
 * dropped: they are the same answer a second time, in pieces. What survives is
 * the answer itself, under names that mean the same thing on every provider.
 */
function describeResponse(
  type: string,
  buffer: ArrayBuffer | undefined,
  status: number,
  headers: Record<string, string>,
): Record<string, unknown> {
  // What the transport says, whatever the body turns out to be — and on a
  // rejected turn it is most of what there is to know.
  const http = {
    request_id: REQUEST_ID.map((name) => headers[name]).find(Boolean),
    rate_limit: rateLimitOf(headers),
    retryable: status >= 400 ? RETRYABLE.has(status) : undefined,
  }
  if (!buffer) return http
  const bytes = buffer.byteLength
  if (type && !TEXT_CONTENT.test(type) && !type.includes("event-stream"))
    return { ...http, bytes, encoding: "binary" }
  const body = new TextDecoder().decode(buffer)
  const streamed = type.includes("event-stream") || body.startsWith("data:") || body.startsWith("event:")
  // A rejected request has no reply to assemble: what it has instead is the
  // reason, which is the whole of what the turn is worth reading for.
  if (status >= 400) return { ...http, error: errorBody(body, streamed), bytes }
  const payload = streamed ? sse(body) : json(body)
  const { reply, usage, id, model } = foldPayload(payload)
  return {
    request_id: http.request_id,
    id,
    model,
    finish_reason: finishReason(reply?.stop_reason, Boolean(reply?.tool_calls.length)),
    usage,
    // The provider's own word for why it stopped rides on the reply and nowhere
    // else. `finish_reason` above is this word normalized, not a second copy of
    // it — writing the raw one twice would let the two disagree.
    reply,
    bytes,
    rate_limit: http.rate_limit,
    // Nothing was understood: no text, no tool call, no token count. Either the
    // provider said nothing or this file does not know its dialect, and in both
    // cases the bytes are the only record there is going to be.
    raw: reply || usage ? undefined : body,
  }
}

/**
 * What the provider said about the room left, and about waiting.
 *
 * Two shapes, same three questions: OpenAI puts what is limited at the end of
 * the header name, Anthropic in the middle. Whatever the suffix says — requests,
 * tokens, input-tokens — is kept as the key, so a counter this file has never
 * heard of still lands in the record.
 */
function rateLimitOf(headers: Record<string, string>) {
  const limit: Record<string, string> = {}
  const remaining: Record<string, string> = {}
  const reset: Record<string, string> = {}
  const into = (kind: string) => (kind === "limit" ? limit : kind === "remaining" ? remaining : reset)
  for (const [name, value] of Object.entries(headers)) {
    const openai = OPENAI_LIMIT.exec(name)
    if (openai?.[2]) into(openai[1]!)[openai[2]] = value
    const anthropic = ANTHROPIC_LIMIT.exec(name)
    if (anthropic?.[1]) into(anthropic[2]!)[anthropic[1]] = value
  }
  const wait = retryAfterOf(headers)
  const some = (value: Record<string, string>) => (Object.keys(value).length ? value : undefined)
  if (wait === undefined && !some(limit) && !some(remaining) && !some(reset)) return undefined
  return { retry_after_ms: wait, limit: some(limit), remaining: some(remaining), reset: some(reset) }
}

/** The wait the provider imposed, in milliseconds, however it spelled it. */
function retryAfterOf(headers: Record<string, string>) {
  const millis = Number(headers["retry-after-ms"])
  if (Number.isFinite(millis)) return Math.max(0, millis)
  const value = headers["retry-after"]
  if (!value) return undefined
  const seconds = Number(value)
  if (Number.isFinite(seconds)) return Math.max(0, seconds * 1000)
  // The header is also allowed to be an HTTP date.
  const date = Date.parse(value)
  return Number.isNaN(date) ? undefined : Math.max(0, date - Date.now())
}

/**
 * Which response this was, and which model actually answered.
 *
 * The model is worth reading off the reply rather than the request: an alias, a
 * fallback after an overload, or a silent downgrade all show up here and
 * nowhere else.
 */
function absorbIdentity(found: Identity, event: unknown) {
  const entry = record(event)
  if (!entry) return
  // Anthropic carries it inside `message`, the Responses dialect inside
  // `response`, and the rest put it at the top of the body.
  const holder = record(entry["message"]) ?? record(entry["response"]) ?? entry
  found.id ??= str(holder["id"]) ?? str(holder["responseId"])
  found.model ??= str(holder["model"]) ?? str(holder["modelVersion"])
}

/**
 * One pass over the frames, for everything that has to be read out of them.
 *
 * A stream is thousands of frames, and the reply, the token bill and the
 * response's identity were each a walk over all of them. None of the three
 * depends on the others, and none depends on the order the frames arrive in —
 * `merge` keeps the larger of each counter and the identity keeps the first of
 * each name — so one loop does what three did.
 */
function foldPayload(payload: unknown) {
  const draft: Draft = { text: [], reasoning: [], calls: new Map() }
  const bill: Bill = { merged: {}, found: false }
  const found: Identity = {}
  // A payload that is not a structure never had a reply in it. The other two
  // readers find nothing in one either, and say so themselves.
  const foldable = payload !== undefined && payload !== null && typeof payload !== "string"
  const list = Array.isArray(payload)
  // A frame of a list sits one level inside it, which is the depth the token
  // search was already budgeted against when it walked the list itself.
  const depth = list ? 1 : 0
  const one = (event: unknown) => {
    if (foldable) absorb(draft, event)
    absorbUsage(bill, event, depth)
    absorbIdentity(found, event)
  }
  if (list) (payload as unknown[]).forEach(one)
  else one(payload)
  return {
    reply: foldable ? replyOf(draft) : undefined,
    usage: bill.found ? normalize(bill.merged) : undefined,
    id: found.id,
    model: found.model,
  }
}

/** The provider's reason for stopping, in opencode's vocabulary. */
function finishReason(native: string | undefined, hasToolCalls: boolean) {
  const mapped = native ? (FINISH.get(native.toLowerCase()) ?? "unknown") : undefined
  // Some dialects only report that the response finished, which says nothing
  // about why; the tool call in the answer is what says it. A `length` or a
  // `content-filter` is never overwritten — a reply cut off mid tool call is
  // truncated first and a tool call second.
  if (hasToolCalls && (mapped === undefined || mapped === "stop")) return "tool-calls"
  return mapped
}

function describeError(error: unknown) {
  if (error instanceof Error) return { name: error.name, message: error.message, cause: causeOf(error.cause) }
  return { message: typeof error === "string" ? error : JSON.stringify(error) }
}

/**
 * The body of a rejected turn, read for what it says rather than for how it was
 * announced.
 *
 * A provider that streams its answers keeps the streaming content type on the
 * rejection too, but the rejection itself is one plain JSON object with no
 * `data:` frame to find. Read as SSE it yields nothing, and a 400 whose message
 * is the entire reason to keep the turn gets written as `[]`. So the frames only
 * count if there were frames, and the body is kept whatever shape it turns out
 * to have: text this file cannot parse is still the reason the request failed.
 */
function errorBody(body: string, streamed: boolean): unknown {
  if (streamed) {
    const events = sse(body)
    if (events.length) return events
  }
  const payload = json(body)
  // `json` hands back the string it could not parse. An empty one says nothing
  // the status has not already said.
  return typeof payload === "string" && !payload.trim() ? undefined : payload
}

// Server-sent events as the providers emit them: one JSON payload per `data:`
// line, terminator frames dropped.
function sse(body: string) {
  const events: unknown[] = []
  for (const frame of body.split(/\r?\n\r?\n/)) {
    const data = frame
      .split(/\r?\n/)
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).trim())
      .join("\n")
    if (!data || data === "[DONE]") continue
    events.push(json(data))
  }
  return events
}

/**
 * The reply, closed out once every frame has been folded into the draft.
 *
 * Nothing at all is a reply that never happened, and an absent `reply` says so
 * — where an empty one would claim the model answered with silence.
 */
function replyOf(draft: Draft): Reply | undefined {
  const text = draft.text.join("")
  const reasoning = draft.reasoning.join("")
  const calls = [...draft.calls.values()].filter((call) => call.name || call.args)
  if (!text && !reasoning && calls.length === 0 && !draft.stop) return undefined
  return {
    text,
    reasoning,
    tool_calls: calls.map((call) => ({
      id: call.id,
      name: call.name,
      // Parsed when the arguments are valid JSON, which they are unless the
      // answer was cut off mid call — and then the fragment is the finding.
      arguments: call.args ? json(call.args) : undefined,
    })),
    stop_reason: draft.stop,
  }
}

/**
 * Folds one frame, or one whole body, into the draft.
 *
 * One pass for every dialect, because they overlap: the same block shapes show
 * up in a streaming frame, in a non-streaming `content` array and in a
 * Responses `output` item.
 */
function absorb(draft: Draft, event: unknown) {
  const value = record(event)
  if (!value) return
  const type = str(value["type"]) ?? ""

  // Anthropic streaming.
  if (type === "content_block_start") {
    absorbBlock(draft, key(value["index"]), record(value["content_block"]))
    return
  }
  if (type === "content_block_delta") {
    const delta = record(value["delta"]) ?? {}
    push(draft.text, delta["text"])
    push(draft.reasoning, delta["thinking"])
    if (typeof delta["partial_json"] === "string") call(draft, key(value["index"])).args += delta["partial_json"]
    return
  }
  if (type === "message_start") {
    absorb(draft, value["message"])
    return
  }
  if (type === "message_delta") {
    draft.stop = str(record(value["delta"])?.["stop_reason"]) ?? draft.stop
    return
  }

  // OpenAI Responses streaming. Only the deltas are folded in; the terminal
  // `response.completed` frame repeats the whole output and would double it.
  if (type.startsWith("response.")) {
    if (type.endsWith(".delta") && typeof value["delta"] === "string") {
      const delta = value["delta"]
      if (type.includes("function_call_arguments"))
        call(draft, key(value["output_index"] ?? value["item_id"])).args += delta
      else if (type.includes("reasoning")) draft.reasoning.push(delta)
      else draft.text.push(delta)
      return
    }
    if (type === "response.output_item.added" || type === "response.output_item.done") {
      // Message and reasoning items repeat text that already arrived as deltas;
      // only the function call is worth picking up, for its name.
      const item = record(value["item"])
      if (str(item?.["type"])?.includes("function_call")) absorbBlock(draft, key(value["output_index"]), item)
      return
    }
    if (type === "response.completed" || type === "response.incomplete" || type === "response.failed") {
      const response = record(value["response"])
      // The status is all this dialect reports; `finishReason` reads the tool
      // calls to tell a plain finish from a finish that is waiting on a tool.
      draft.stop = str(record(response?.["incomplete_details"])?.["reason"]) ?? str(response?.["status"]) ?? draft.stop
      return
    }
    return
  }

  // OpenAI chat completions, streamed or not.
  const choices = arrayOf(value["choices"])
  if (choices.length) {
    for (const choice of choices) {
      const entry = record(choice)
      if (!entry) continue
      draft.stop = str(entry["finish_reason"]) ?? draft.stop
      const message = record(entry["delta"]) ?? record(entry["message"])
      if (!message) continue
      push(draft.text, message["content"])
      if (Array.isArray(message["content"]))
        for (const part of message["content"]) absorbBlock(draft, "0", record(part))
      push(draft.reasoning, message["reasoning_content"] ?? message["reasoning"])
      arrayOf(message["tool_calls"]).forEach((item, position) => {
        const use = record(item)
        if (!use) return
        const target = call(draft, key(use["id"] ?? use["index"] ?? position))
        const fn = record(use["function"]) ?? use
        target.name = str(fn["name"]) ?? target.name
        target.id = str(use["id"]) ?? target.id
        if (typeof fn["arguments"] === "string") target.args += fn["arguments"]
      })
    }
    return
  }

  // Gemini, streamed or not.
  const candidates = arrayOf(value["candidates"])
  if (candidates.length) {
    for (const candidate of candidates) {
      const entry = record(candidate)
      if (!entry) continue
      draft.stop = str(entry["finishReason"]) ?? draft.stop
      arrayOf(record(entry["content"])?.["parts"]).forEach((part, position) => {
        const item = record(part)
        if (!item) return
        // Gemini marks reasoning by flagging an ordinary text part.
        if (typeof item["text"] === "string")
          push(item["thought"] === true ? draft.reasoning : draft.text, item["text"])
        const fn = record(item["functionCall"]) ?? record(item["function_call"])
        if (fn) {
          const target = call(draft, key(str(fn["name"]) ?? position))
          target.name = str(fn["name"]) ?? target.name
          target.args = safeStringify(fn["args"] ?? fn["arguments"] ?? {})
        }
      })
    }
    return
  }

  // Anthropic non-streaming, and Bedrock Converse.
  if (Array.isArray(value["content"])) {
    value["content"].forEach((block, position) => absorbBlock(draft, key(position), record(block)))
    draft.stop = str(value["stop_reason"]) ?? draft.stop
    return
  }
  if (Array.isArray(value["output"])) {
    value["output"].forEach((item, position) => absorbBlock(draft, key(position), record(item)))
    draft.stop = str(value["status"]) ?? draft.stop
    return
  }
  const output = record(value["output"])
  if (output) {
    absorb(draft, output["message"] ?? output)
    draft.stop = str(value["stopReason"]) ?? draft.stop
  }
}

function absorbBlock(draft: Draft, id: string, block: Record<string, unknown> | undefined) {
  if (!block) return
  const type = str(block["type"]) ?? ""
  if (type === "text" || type === "output_text" || type === "input_text") {
    push(draft.text, block["text"])
    return
  }
  if (type === "thinking" || type === "reasoning" || type === "reasoning_content") {
    push(draft.reasoning, block["thinking"] ?? block["text"])
    for (const item of arrayOf(block["summary"]).concat(arrayOf(block["content"])))
      push(draft.reasoning, record(item)?.["text"])
    return
  }
  if (type === "redacted_thinking") {
    push(draft.reasoning, REDACTED)
    return
  }
  if (type === "tool_use" || type === "server_tool_use" || type === "function_call") {
    // Keyed by position, never by the block's own id: the frames that carry the
    // arguments identify the call by its index in the stream. The id is kept as
    // a field instead — it is what the next turn's tool result points back at.
    const target = call(draft, id)
    target.name = str(block["name"]) ?? target.name
    // `call_id` before `id`: the Responses dialect carries both, and it is the
    // `call_id` that the next turn's `function_call_output` points back at —
    // the item's own `fc_...` id joins with nothing. The dialects that have one
    // id call it `id` (Anthropic, chat completions) or `toolUseId` (Bedrock).
    target.id = str(block["call_id"] ?? block["id"] ?? block["toolUseId"]) ?? target.id
    const input = block["input"] ?? block["arguments"]
    // While streaming, the opening frame carries empty arguments and the real
    // ones arrive as deltas, so never overwrite what is already there.
    if (typeof input === "string") {
      if (input && !target.args) target.args = input
    } else if (isRecord(input) && Object.keys(input).length) target.args = safeStringify(input)
    return
  }
  if (type === "message") {
    for (const part of arrayOf(block["content"])) absorbBlock(draft, id, record(part))
    return
  }
  // Bedrock blocks have no `type`; the key is the type.
  if (!type) {
    push(draft.text, block["text"])
    const use = record(block["toolUse"])
    if (use) absorbBlock(draft, id, { type: "tool_use", ...use })
    const reasoning = record(block["reasoningContent"])
    if (reasoning) push(draft.reasoning, textOf(reasoning))
  }
}

function call(draft: Draft, id: string) {
  const existing = draft.calls.get(id)
  if (existing) return existing
  const created: { id?: string; name?: string; args: string } = { args: "" }
  draft.calls.set(id, created)
  return created
}

function push(target: string[], value: unknown) {
  if (typeof value === "string" && value) target.push(value)
}

function key(value: unknown) {
  return typeof value === "string" || typeof value === "number" ? String(value) : "0"
}

function textOf(value: unknown, depth = 0): string {
  if (typeof value === "string") return value
  if (depth > 8 || value === null || typeof value !== "object") return ""
  if (Array.isArray(value))
    return value
      .map((item) => textOf(item, depth + 1))
      .filter(Boolean)
      .join("\n")
  const entry = record(value)
  if (!entry) return ""
  const parts: string[] = []
  for (const name of TEXT_KEYS) if (name in entry) parts.push(textOf(entry[name], depth + 1))
  for (const name of CODE_KEYS) if (name in entry) parts.push(safeStringify(entry[name]))
  return parts.filter(Boolean).join("\n")
}

/**
 * The token bill, wherever the dialect keeps it.
 *
 * A stream reports usage in more than one frame — an opening count and a final
 * one — so the frames are merged, taking the larger of each counter, before
 * anything is read out of them.
 */
function absorbUsage(bill: Bill, value: unknown, depth: number) {
  if (depth > 8 || value === null || typeof value !== "object") return
  if (Array.isArray(value)) return value.forEach((item) => absorbUsage(bill, item, depth + 1))
  for (const [name, item] of Object.entries(value)) {
    if (USAGE_KEY.test(name) && isRecord(item)) {
      merge(bill.merged, item)
      bill.found = true
      continue
    }
    absorbUsage(bill, item, depth + 1)
  }
}

function merge(target: Record<string, unknown>, source: Record<string, unknown>) {
  for (const [name, value] of Object.entries(source)) {
    if (typeof value === "number" && Number.isFinite(value)) target[name] = Math.max(num(target[name]), value)
    else if (isRecord(value)) {
      const nested = record(target[name]) ?? {}
      merge(nested, value)
      target[name] = nested
    }
  }
}

/**
 * One shape out of every dialect.
 *
 * The prompt counters are the subtle part. Anthropic reports `input_tokens`
 * already net of the cache, and names the cached tokens separately. The OpenAI
 * dialects and Google fold the cached tokens into the prompt total and say so
 * in a details object — so the cached part is subtracted back out, which is
 * also why the absence of that details object is what distinguishes the two
 * conventions. Bedrock uses the same counters under camelCase names.
 *
 * The output counter is the other one. Anthropic and OpenAI already count
 * thinking inside the completion tokens; Gemini reports `thoughtsTokenCount`
 * on the side and bills it on top of the candidates, so it is added there.
 */
function normalize(value: Record<string, unknown>): Usage | undefined {
  const inputDetails = record(value["input_tokens_details"]) ?? record(value["prompt_tokens_details"]) ?? {}
  const outputDetails = record(value["output_tokens_details"]) ?? record(value["completion_tokens_details"]) ?? {}
  const creation = record(value["cache_creation"]) ?? {}

  // The cache counters the prompt total already contains. Read is named three
  // ways and never twice at once, so the spellings are alternatives rather than
  // a sum. Write only shows up nested, on the dialects that fold it in.
  const cachedRead =
    Math.max(num(inputDetails["cached_tokens"]), num(value["cached_tokens"])) + num(value["cachedContentTokenCount"])
  const cachedWrite = num(inputDetails["cache_write_tokens"]) + num(inputDetails["cache_creation_input_tokens"])
  const inclusive = cachedRead + cachedWrite
  const prompt =
    num(value["input_tokens"]) +
    num(value["inputTokens"]) +
    num(value["prompt_tokens"]) +
    num(value["promptTokenCount"])
  const write5m = num(creation["ephemeral_5m_input_tokens"])
  const write1h = num(creation["ephemeral_1h_input_tokens"])
  const thoughts = num(value["thoughtsTokenCount"])

  // Whether this dialect counts cache and thinking at all. A counter it never
  // reports is left out of the record rather than written as zero.
  const counts = (...items: unknown[]) => items.some((item) => typeof item === "number")
  const cached = counts(
    inputDetails["cached_tokens"],
    value["cached_tokens"],
    value["cachedContentTokenCount"],
    value["cache_read_input_tokens"],
    value["cacheReadInputTokens"],
    value["cache_creation_input_tokens"],
    value["cacheWriteInputTokens"],
    inputDetails["cache_write_tokens"],
    inputDetails["cache_creation_input_tokens"],
    creation["ephemeral_5m_input_tokens"],
    creation["ephemeral_1h_input_tokens"],
  )
  const thinks = counts(outputDetails["reasoning_tokens"], value["reasoning_tokens"], value["thoughtsTokenCount"])
  const cacheRead = cachedRead + num(value["cache_read_input_tokens"]) + num(value["cacheReadInputTokens"])
  // The same number under three conventions — the top-level counter, the TTL
  // split beside it, and the nested one already inside the prompt total — so the
  // largest, not the sum. Adding them would bill one write twice.
  const cacheWrite = Math.max(
    num(value["cache_creation_input_tokens"]) + num(value["cacheWriteInputTokens"]),
    write5m + write1h,
    cachedWrite,
  )

  const usage: Usage = {
    // Gemini bills the prompt its built-in tools generate on top of the prompt.
    input: Math.max(0, prompt - inclusive) + num(value["toolUsePromptTokenCount"]),
    ...(cached ? { cache_read: cacheRead, cache_write: cacheWrite } : {}),
    output:
      num(value["output_tokens"]) +
      num(value["outputTokens"]) +
      num(value["completion_tokens"]) +
      num(value["candidatesTokenCount"]) +
      thoughts,
    ...(thinks ? { reasoning: num(outputDetails["reasoning_tokens"]) + num(value["reasoning_tokens"]) + thoughts } : {}),
    prompt: 0,
    total: 0,
    source: "provider",
    native: value,
  }
  usage.prompt = usage.input + (usage.cache_read ?? 0) + (usage.cache_write ?? 0)
  usage.total = usage.prompt + usage.output
  // Every counter came back zero, which is not a bill: it is a `usage` object
  // that held nothing this file knows how to read.
  if (usage.total === 0) return undefined

  // The provider's own total, kept only when it disagrees with ours. Silence
  // means the two agree; a number here means a counter this file did not read,
  // and it is the alarm that the sum above is short by exactly the difference.
  const reported = num(value["total_tokens"]) + num(value["totalTokens"]) + num(value["totalTokenCount"])
  const details = {
    // The cache write split, when the provider reports it. The two TTLs are
    // priced differently and opencode's own catalog has a single write rate, so
    // this is the one number that lets a reader be more exact than opencode is
    // — and it is also what says which window this turn's cache expires in.
    ...(write5m ? { cache_write_5m: write5m } : {}),
    ...(write1h ? { cache_write_1h: write1h } : {}),
    ...(reported && reported !== usage.total
      ? { reported_total: reported, total_mismatch: reported - usage.total }
      : {}),
  }
  if (Object.keys(details).length) usage.details = details
  return usage
}

function rawBody(body: unknown): string | undefined {
  if (typeof body === "string") return body
  if (body instanceof ArrayBuffer) return new TextDecoder().decode(body)
  if (ArrayBuffer.isView(body)) return new TextDecoder().decode(body as Uint8Array)
  return undefined
}

function urlOf(input: unknown) {
  if (typeof input === "string") return input
  if (input instanceof URL) return input.toString()
  if (isRecord(input) && typeof input["url"] === "string") return input["url"]
  return String(input)
}

function methodOf(input: unknown) {
  return isRecord(input) && typeof input["method"] === "string" ? input["method"] : undefined
}

function headersOf(input: unknown) {
  return isRecord(input) ? input["headers"] : undefined
}

/** A request header, if it survived redaction. */
function headerOf(request: { headers: Record<string, string> }, name: string) {
  const value = request.headers[name]
  return value && value !== REDACTED ? value : undefined
}

function pathOf(url: string) {
  return URL.canParse(url) ? new URL(url).pathname : url
}

function hostOf(url: string) {
  return URL.canParse(url) ? new URL(url).host : undefined
}

// Anthropic and the OpenAI dialects name the model in the body; Gemini and
// Bedrock put it in the path. Only the fallback for when the hooks said nothing.
function modelOf(request: Wire) {
  const named = str(request.body["model"])
  return named ?? /\/models?\/([^:/?]+)/.exec(request.url)?.[1]
}

function round2(value: number) {
  return Math.round(value * 100) / 100
}

function redactUrl(value: string) {
  if (!URL.canParse(value)) return value
  const url = new URL(value)
  url.searchParams.forEach((_, key) => {
    if (SENSITIVE_NAME.test(key) || SHORT_QUERY_NAME.test(key)) url.searchParams.set(key, REDACTED)
  })
  return url.toString()
}

function redactHeaders(input: unknown): Record<string, string> {
  const result: Record<string, string> = {}
  if (input === undefined || input === null) return result
  try {
    new Headers(input as HeadersInit).forEach((value, name) => {
      result[name.toLowerCase()] = SENSITIVE_NAME.test(name) && !PUBLIC_NAME.test(name) ? REDACTED : value
    })
  } catch {
    return result
  }
  return result
}

function json(value: string) {
  try {
    return JSON.parse(value)
  } catch {
    return value
  }
}

/**
 * The turn, as the file holds it: everything serialized, and the request body
 * put back where its placeholder sits, byte for byte as it went out.
 */
function text(value: unknown, raw?: string) {
  const out = serialize(value)
  if (raw === undefined) return out + "\n"
  const needle = JSON.stringify(BODY_MARK)
  const at = out.indexOf(needle)
  // No placeholder is not a failure: a turn written before the body was read,
  // or one whose body never made it into the object, simply has nothing to put
  // back. What is never done is guessing at where it would have gone.
  if (at < 0) return out + "\n"
  return out.slice(0, at) + raw + out.slice(at + needle.length) + "\n"
}

/**
 * The common path first: a replacer is called back on every node in the tree,
 * and a `bigint` is the only thing it exists for. So the plain serialization is
 * tried, and the replacer pays for itself only on the tree that actually holds
 * one — which is the tree that just threw.
 */
function serialize(value: unknown) {
  try {
    return JSON.stringify(value, null, 2)
  } catch {
    return JSON.stringify(value, (_key, item) => (typeof item === "bigint" ? String(item) : item), 2)
  }
}

function causeOf(cause: unknown) {
  if (cause instanceof Error) return cause.message
  return typeof cause === "string" ? cause : undefined
}

function safeStringify(value: unknown) {
  if (typeof value === "string") return value
  try {
    return JSON.stringify(value, (_key, item) => (typeof item === "bigint" ? String(item) : item)) ?? ""
  } catch {
    return "<unserializable>"
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
}

function record(value: unknown) {
  return isRecord(value) ? value : undefined
}

function arrayOf(value: unknown): unknown[] {
  return Array.isArray(value) ? value : []
}

function str(value: unknown) {
  return typeof value === "string" && value ? value : undefined
}

function num(value: unknown) {
  return typeof value === "number" && Number.isFinite(value) ? value : 0
}
