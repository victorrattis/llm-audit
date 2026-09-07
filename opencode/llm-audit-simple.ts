// A minimal wire log for LLM provider traffic.
//
// One file per model request, holding exactly two things: what went out and
// what came back, verbatim. No token accounting, no reuse diff, no piece store,
// no summaries — for those, use `llm-audit.ts` next to this file. This one
// answers only "what did opencode send this turn, and what did the provider
// reply?", and answers it without anything to join.
//
// Self-contained: drop it in `.opencode/plugins/` (project) or
// `~/.config/opencode/plugins/` (global) of a stock opencode install.
//
// Enable with `OPENCODE_LLM_AUDIT_SIMPLE=1` (or point
// `OPENCODE_LLM_AUDIT_SIMPLE_DIR` at a directory), or from config:
//
//   { "plugin": [["./llm-audit-simple.ts", { "dir": "/tmp/audit" }]] }
//
// A "turn" is one HTTP request to the model. A single thing you typed usually
// spans several, because each tool call round-trips again.
//
//   <dir>/<timestamp>-<session>/turn_001.json
//   <dir>/<timestamp>-<session>/turn_002.json
//
// and each of those is:
//
//   { "request":  { method, url, dialect, settings, chars, cache, prompt, headers },
//     "response": { status, id, model, finish_reason, usage, reply, headers } }
//
// The request keeps every byte that went out — the system prompt, the tool
// schemas and every message, exactly as they were sent, cache markers and all.
// The repetition between turns is the point rather than noise: the model holds
// no state, so each turn resends the whole conversation, and that is precisely
// what the prompt cache is charging for. Only the shape is normalized, because
// each dialect hides the same three things somewhere else:
//
//   prompt     `system`, `tools` and `messages`, whatever the provider calls
//              them — `input`/`instructions` on the Responses dialect (where
//              opencode sends the system prompt as a `developer` item),
//              `contents`/`systemInstruction` on Gemini, `toolConfig.tools` on
//              Bedrock. The items inside are untouched
//   settings   everything else in the body, by subtraction rather than by a
//              list of known keys — that is how `thinking`, `service_tier`,
//              `include`, `generationConfig` and whatever the next provider
//              invents stay in the record instead of being dropped
//   cache      what the request asked to cache, under one name: the `key`
//              (`prompt_cache_key`, `cachedContent`) or the `breakpoints`
//              (Anthropic's `cache_control`, Bedrock's positional `cachePoint`),
//              each saying which part and which index it sits at
//   chars      the size of each of the three parts, which is what shows the
//              system and the tools standing still while the messages grow
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
//                 `packages/llm/src/schema/ids.ts`), with the provider's word
//                 for it kept beside it as `finish_reason_native`
//   usage         the token bill, normalized so the prompt counters never
//                 overlap: prompt = input + cache_read + cache_write, and
//                 reasoning is part of output rather than added to it. The
//                 provider's own usage object is kept verbatim as `native`,
//                 so a mapping that is wrong is still a mapping you can redo
//   reply         what the model said: text, reasoning, and the tool calls
//                 with their ids, names and parsed arguments. The id is the
//                 join with the tool result that comes back in the next
//                 turn's request
//
// `raw` comes back only when none of that could be decoded, so a dialect this
// file does not know still leaves its bytes behind rather than an empty turn.
// A rejected request keeps its error body verbatim, where the reply would be.
//
// Credentials in headers and query params are redacted; bodies are stored
// verbatim, so treat the directory as sensitive.
//
// How it hooks in: it wraps the process's `fetch` and records the calls whose
// shape says "model request" (see `isModelCall`), so it covers every provider —
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

// Paths the model APIs use: OpenAI-style completions and responses, Anthropic
// messages, Gemini generateContent, Bedrock converse/invoke, Cohere-style chat.
const MODEL_PATH =
  /(chat\/completions|\/completions|\/responses|\/messages|generatecontent|\/converse|\/invoke|\/predict|\/chat)(\b|$)/i
// Body keys that mean "this is a prompt", across every dialect.
const PROMPT_KEY = /"(messages|contents|system_instruction|systemInstruction)"\s*:/

// Turns that arrive without a session id still have to be written somewhere.
const NO_SESSION = "sem-sessao"

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
  // The TTL split of the cache write, when the provider reports it: the two are
  // billed at different multipliers.
  cache_write_5m?: number
  cache_write_1h?: number
  output: number
  reasoning?: number
  prompt: number
  total: number
  // `cache_read / prompt`, kept next to the numbers it comes from.
  cache_hit_rate?: number
  // The provider's own total, written only when it disagrees with the sum
  // above — silence means the two agree.
  reported_total?: number
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
  pending: Map<string, { head: Record<string, unknown>; request: unknown }>
  queue: Promise<void>
}

export const LLMAuditSimplePlugin: Plugin = async (_input, options) => {
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
  const flag = env["OPENCODE_LLM_AUDIT_SIMPLE"]
  const dir = (options?.["dir"] as string) ?? env["OPENCODE_LLM_AUDIT_SIMPLE_DIR"]
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
        "llm-audit-simple",
      ),
  }
}

/** One state per process, however many plugin instances opencode creates. */
function shared(settings: Settings): State {
  const global = globalThis as Record<string, unknown>
  const existing = global["__opencodeSimpleAuditState"] as State | undefined
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
  global["__opencodeSimpleAuditState"] = state
  return state
}

/**
 * Wraps the process's fetch, once. The guard key is this plugin's own, so the
 * full `llm-audit.ts` can wrap the same fetch and both still record.
 */
function install(state: State) {
  const global = globalThis as Record<string, unknown>
  if (global["__opencodeSimpleAuditFetch"]) return
  const original = globalThis.fetch
  const patched = (input: Parameters<FetchLike>[0], init?: Parameters<FetchLike>[1]) => {
    const request = describeRequest(input, init)
    if (!isModelCall(request)) return original(input, init)
    return capture(state, request, () => original(input, init))
  }
  // Carries over anything the runtime hangs off fetch (Bun's `preconnect`).
  global["__opencodeSimpleAuditFetch"] = true
  globalThis.fetch = Object.assign(patched, original) as FetchLike
  // Written synchronously: an exit handler is too late for the queue.
  process.on("exit", () => flush(state))
}

/**
 * A model call POSTs a JSON body shaped like a prompt. Both halves matter: the
 * shape alone would catch opencode's own APIs, and the path alone would catch
 * unrelated traffic to the same hosts.
 */
function isModelCall(request: ReturnType<typeof describeRequest>) {
  if (request.method !== "POST" || typeof request.raw !== "string") return false
  if (!PROMPT_KEY.test(request.raw) && !isRecord(request.body)) return false
  if (!isRecord(request.body)) return false
  const shape =
    "messages" in request.body ||
    "contents" in request.body ||
    "prompt" in request.body ||
    ("input" in request.body && "model" in request.body)
  if (!shape) return false
  return MODEL_PATH.test(pathOf(request.url)) || typeof request.body["model"] === "string"
}

async function capture(state: State, request: ReturnType<typeof describeRequest>, send: () => Promise<Response>) {
  const started = Date.now()
  // opencode names the session on every request it makes, and a subagent also
  // names whoever spawned it — its turns are that conversation's work, so they
  // are filed in that folder and numbered in the same sequence.
  const wire = sessionOf(state, request)
  const parent = headerOf(request, "x-parent-session-id")
  if (wire !== NO_SESSION && parent) state.parents.set(wire, parent)
  const sess = openSession(state, rootOf(state, wire))
  if (!sess) return send()
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
  state.pending.set(file, { head, request: sent })

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
    )

  try {
    const response = await send()
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

/** The turn file, and the line about it in the folder's index. */
function write(
  state: State,
  sess: Session,
  file: string,
  head: Record<string, unknown>,
  request: unknown,
  response: Record<string, unknown>,
) {
  state.pending.delete(file)
  sess.rows.push(rowOf(head, path.basename(file), request, response))
  state.queue = state.queue
    .then(() => fs.promises.writeFile(file, text({ ...head, request, response })))
    .then(() => fs.promises.writeFile(path.join(sess.dir, INDEX_FILE), text(index(sess))))
    .catch(() => undefined)
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
    chars: num(record(record(request)?.["chars"])?.["total"]) || undefined,
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
    turns: [...sess.rows].sort((a, b) => num(a["turn"]) - num(b["turn"])),
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
  for (const [file, { head, request }] of state.pending) {
    const response = {
      incomplete: true,
      note: "the process exited while this request was still in flight, so nothing came back to record",
    }
    const sess = [...state.sessions.values()].find((entry) => entry.dir === path.dirname(file))
    if (sess) {
      sess.rows.push(rowOf(head, path.basename(file), request, response))
      touched.add(sess)
    }
    try {
      fs.writeFileSync(file, text({ ...head, request, response }))
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

function describeRequest(input: unknown, init: { method?: string; headers?: unknown; body?: unknown } | undefined) {
  const raw = rawBody(init?.body)
  return {
    method: (init?.method ?? methodOf(input) ?? "GET").toUpperCase(),
    url: urlOf(input),
    headers: redactHeaders(init?.headers ?? headersOf(input)),
    raw,
    body: typeof raw === "string" ? json(raw) : raw,
  }
}

/**
 * The request, in one shape whatever the dialect speaks.
 *
 * Every item is kept exactly as it was sent, cache markers and all — the
 * repetition between turns is the point of the log, not noise: the model holds
 * no state, so each turn resends the whole conversation, and that is precisely
 * what the prompt cache is charging for. What changes here is only *where* the
 * three parts live, because each dialect hides them somewhere else.
 */
function readRequest(request: ReturnType<typeof describeRequest>) {
  const head = { method: request.method, url: redactUrl(request.url) }
  const body = record(request.body)
  const dialect = body ? dialectOf(body) : "unknown"
  // Nothing is guessed about a dialect this file does not know: the body goes in
  // whole, the way an undecodable response keeps its bytes.
  if (!body || dialect === "unknown")
    return { ...head, dialect, body: request.body, headers: request.headers }

  const prompt = {
    system: systemItems(body, dialect),
    tools: toolItems(body, dialect),
    messages: messageItems(body, dialect),
  }
  const chars = {
    total: 0,
    system: sizeOf(prompt.system),
    tools: sizeOf(prompt.tools),
    messages: sizeOf(prompt.messages),
  }
  chars.total = chars.system + chars.tools + chars.messages

  return {
    ...head,
    dialect,
    settings: settingsOf(body, dialect),
    chars,
    cache: cacheView(body, prompt),
    prompt,
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

// The body keys each dialect spends on the prompt. Everything else is a
// setting, by subtraction — see `settingsOf`.
const PROMPT_KEYS: Record<string, string[]> = {
  anthropic: ["system", "messages", "tools"],
  "openai-chat": ["messages", "tools"],
  "openai-responses": ["input", "instructions", "tools"],
  gemini: ["contents", "systemInstruction", "system_instruction", "tools"],
  bedrock: ["messages", "system", "toolConfig"],
}

/**
 * Everything in the body that is not the prompt — model, sampling, cache key,
 * whatever else the provider takes.
 *
 * Taken by subtraction rather than from a list of known keys, and that is the
 * whole point: the keys worth having are exactly the ones a fixed list would
 * miss. `thinking` (Anthropic), `service_tier`, `include` and `prompt_cache_key`
 * (Responses), `generationConfig` (Gemini), `additionalModelRequestFields`
 * (Bedrock) — none of them are in the setting list the full plugin keeps, and
 * the next provider will invent one more. Subtraction cannot miss any of them.
 */
function settingsOf(body: Record<string, unknown>, dialect: string) {
  const spent = new Set(PROMPT_KEYS[dialect] ?? [])
  const settings: Record<string, unknown> = {}
  for (const [name, value] of Object.entries(body)) if (!spent.has(name)) settings[name] = value
  // Bedrock keeps `toolChoice` inside `toolConfig`, next to the tools that were
  // lifted out of it. Only the tools were spent, so the rest comes back.
  if (dialect === "bedrock") {
    const config = { ...record(body["toolConfig"]) }
    delete config["tools"]
    if (Object.keys(config).length) settings["toolConfig"] = config
  }
  return settings
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
function cacheView(body: Record<string, unknown>, prompt: Record<string, unknown[]>) {
  const key = str(body["prompt_cache_key"]) ?? str(body["cachedContent"]) ?? str(body["cached_content"])
  const breakpoints: { in: string; index: number; type: string; ttl?: string }[] = []
  for (const [part, items] of Object.entries(prompt))
    items.forEach((item, index) => {
      const marker = cacheOf(safeStringify(item))
      if (marker) breakpoints.push({ in: part, index, ...marker })
    })
  // No key and no breakpoint is a request that asked for no caching at all,
  // which is a finding of its own — hence an absent field rather than an empty
  // one, so it reads as "nothing asked" and not as "nothing found".
  if (!key && !breakpoints.length) return undefined
  return { key, breakpoints: breakpoints.length ? breakpoints : undefined }
}

function cacheOf(serial: string) {
  const control = /"cache_control"\s*:\s*\{([^}]*)\}/.exec(serial)?.[1]
  const point = /"cachePoint"\s*:\s*\{([^}]*)\}/.exec(serial)?.[1]
  const inner = control ?? point
  if (inner === undefined) return undefined
  return {
    // Anthropic names the kind of cache it wants; a Bedrock cache point carries
    // only `"type":"default"`, which says nothing its position does not.
    type: control ? (/"type"\s*:\s*"([^"]+)"/.exec(control)?.[1] ?? "ephemeral") : "point",
    ttl: /"ttl"\s*:\s*"([^"]+)"/.exec(inner)?.[1],
  }
}

/** The characters a list of prompt items occupies on the wire. */
function sizeOf(items: unknown[]) {
  return items.reduce((total: number, item) => total + safeStringify(item).length, 0)
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
  const payload = streamed ? sse(body) : json(body)
  // A rejected request has no reply to assemble: what it has instead is the
  // reason, which is the whole of what the turn is worth reading for.
  if (status >= 400) return { ...http, error: payload, bytes }

  const reply = readReply(payload)
  const usage = usageOf(payload)
  const { id, model } = identityOf(payload)
  return {
    request_id: http.request_id,
    id,
    model,
    finish_reason: finishReason(reply?.stop_reason, Boolean(reply?.tool_calls.length)),
    finish_reason_native: reply?.stop_reason,
    usage,
    // `stop_reason` is left off the reply itself: it is already above, twice,
    // and a number written in two places is a number that gets to disagree.
    reply: reply && { text: reply.text, reasoning: reply.reasoning, tool_calls: reply.tool_calls },
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
function identityOf(payload: unknown): { id?: string; model?: string } {
  const found: { id?: string; model?: string } = {}
  const visit = (value: unknown) => {
    const entry = record(value)
    if (!entry) return
    // Anthropic carries it inside `message`, the Responses dialect inside
    // `response`, and the rest put it at the top of the body.
    const holder = record(entry["message"]) ?? record(entry["response"]) ?? entry
    found.id ??= str(holder["id"]) ?? str(holder["responseId"])
    found.model ??= str(holder["model"]) ?? str(holder["modelVersion"])
  }
  if (Array.isArray(payload)) payload.forEach(visit)
  else visit(payload)
  return found
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
 * The reply, folded out of whatever the provider sent.
 *
 * One pass for every dialect, because they overlap: the same block shapes show
 * up in a streaming frame, in a non-streaming `content` array and in a
 * Responses `output` item.
 */
function readReply(payload: unknown): Reply | undefined {
  if (payload === undefined || payload === null || typeof payload === "string") return undefined
  const draft: Draft = { text: [], reasoning: [], calls: new Map() }
  if (Array.isArray(payload)) payload.forEach((event) => absorb(draft, event))
  else absorb(draft, payload)

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

/** Folds one frame, or one whole body, into the draft. */
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
function usageOf(payload: unknown): Usage | undefined {
  const merged: Record<string, unknown> = {}
  let found = false
  const visit = (value: unknown, depth: number) => {
    if (depth > 8 || value === null || typeof value !== "object") return
    if (Array.isArray(value)) return value.forEach((item) => visit(item, depth + 1))
    for (const [name, item] of Object.entries(value)) {
      if (USAGE_KEY.test(name) && isRecord(item)) {
        merge(merged, item)
        found = true
        continue
      }
      visit(item, depth + 1)
    }
  }
  visit(payload, 0)
  if (!found) return undefined
  return normalize(merged)
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
    // The cache write split, when the provider reports it. The two TTLs are
    // priced differently and opencode's own catalog has a single write rate, so
    // this is the one number that lets a reader be more exact than opencode is.
    ...(write5m ? { cache_write_5m: write5m } : {}),
    ...(write1h ? { cache_write_1h: write1h } : {}),
    output:
      num(value["output_tokens"]) +
      num(value["outputTokens"]) +
      num(value["completion_tokens"]) +
      num(value["candidatesTokenCount"]) +
      thoughts,
    ...(thinks ? { reasoning: num(outputDetails["reasoning_tokens"]) + num(value["reasoning_tokens"]) + thoughts } : {}),
    prompt: 0,
    total: 0,
    native: value,
  }
  usage.prompt = usage.input + (usage.cache_read ?? 0) + (usage.cache_write ?? 0)
  usage.total = usage.prompt + usage.output
  // Every counter came back zero, which is not a bill: it is a `usage` object
  // that held nothing this file knows how to read.
  if (usage.total === 0) return undefined
  usage.cache_hit_rate = usage.prompt && usage.cache_read !== undefined ? round2(usage.cache_read / usage.prompt) : undefined
  // The provider's own total, kept only when it disagrees with ours. Silence
  // means the two agree; a number here means a counter this file did not read,
  // and it is the alarm that the sum below is short.
  const reported = num(value["total_tokens"]) + num(value["totalTokens"]) + num(value["totalTokenCount"])
  if (reported && reported !== usage.total) usage.reported_total = reported
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
function modelOf(request: { url: string; body: unknown }) {
  const named = str(record(request.body)?.["model"])
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

function text(value: unknown) {
  return JSON.stringify(value, (_key, item) => (typeof item === "bigint" ? String(item) : item), 2) + "\n"
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
