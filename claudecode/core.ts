// Wire-level audit log for Claude Code, one record per model request.
//
// This is the decoding half of `llm-audit.ts` — the opencode plugin that wraps
// the process's `fetch` — carried over unchanged. Claude Code is closed and
// wraps nothing, so the capture half is gone: `report.ts` reads what Claude
// Code itself writes to disk (the session transcript, and the raw Messages API
// bodies that `OTEL_LOG_RAW_API_BODIES=file:<dir>` leaves behind) and calls
// into here.
//
// Everything below is the opencode file verbatim, minus the fetch wrapper and
// the HTTP redaction it needed. Keeping the functions and the types identical
// is the whole point: the folders this writes have the same shape as the ones
// opencode writes, so the same `jq` reads both.
import { createHash } from "node:crypto"
import fs from "node:fs"
import path from "node:path"

export const REDACTED = "<redacted>"

// The fallback ratio, used only for the turns where the provider reports no
// usage at all. Where it does report, the real count is distributed instead.
export const CHARS_PER_TOKEN = 4
const PREVIEW_CHARS = 240

// Turns that arrive without a session id and without a context to borrow one
// from still have to be written somewhere.
export const NO_SESSION = "sem-sessao"

// Ids that are different on every request by construction. Claude Code smuggles
// `x-anthropic-billing-header` in as the first system block, and it carries the
// id of the request before it: the provider does not read that block as prompt —
// the whole prefix comes back from the cache anyway — but a byte comparison would
// call the system block changed on every single turn. They come out whole, not
// masked in place: the first request of a session carries no `cc_prev_req` at
// all, and a field that is there on one turn and absent on the next would break
// the prefix just the same.
const VOLATILE_ID = /\s*\b(cc_prev_req|cc_prompt_id)=[^;"]*;?/g

// Block types whose size on the wire says nothing about their token cost.
const BINARY_KIND = /image|audio|video|document|file|inline_?data|blob|source/i

// Sampling knobs worth auditing, across dialects. Everything else in the body
// is either the prompt itself or plumbing.
const SETTING_KEYS = [
  "stream",
  "stream_options",
  "max_tokens",
  "max_completion_tokens",
  "max_output_tokens",
  "maxTokens",
  "temperature",
  "top_p",
  "top_k",
  "topP",
  "topK",
  "thinking",
  "reasoning",
  "reasoning_effort",
  "tool_choice",
  "toolChoice",
  "parallel_tool_calls",
  "stop_sequences",
  "stop",
  "service_tier",
  "generationConfig",
  "inferenceConfig",
  "truncation",
  "store",
  "metadata",
]

// Where the providers put their token counts. Cohere hangs them off `meta`.
const USAGE_KEY = /^(usage|usage_?metadata|billed_units|token_?usage)$/i

// Only the parts of the model this file uses. Declared locally because
// `@opencode-ai/plugin` does not export the model type, and the value we get
// from `chat.params` is structurally wider than this.
export type ModelInfo = {
  id: string
  providerID: string
  cost: { input: number; output: number; cache: { read: number; write: number } }
}

/**
 * One request's token bill, normalized across the dialects.
 *
 * `input` excludes what was read from or written to the cache, so the four
 * prompt-side counters never overlap and `prompt` is their sum. `reasoning` is
 * the thinking part of `output`, not an addition to it.
 */
export type Usage = {
  input: number
  cache_read: number
  cache_write: number
  output: number
  reasoning: number
  prompt: number
  total: number
  source: "provider" | "estimated"
  // Counters only some providers report: the cache TTL split, audio and
  // prediction tokens, built-in tool requests, the provider's own total (kept
  // as reported, to check ours against), and its own cost when it bills in
  // dollars rather than tokens.
  details?: Record<string, number>
}

export type Context = {
  sessionID: string
  agent: string
  providerID: string
  model: ModelInfo
}

// One decoded piece of the prompt: a system block, a tool schema, or a message.
// Sized in characters here; tokens are attributed later, once the provider has
// said what the whole prompt actually cost.
export type Piece = {
  index: number
  role?: string
  name?: string
  kinds?: string[]
  chars: number
  share?: number
  binary?: boolean
  description_chars?: number
  schema_chars?: number
  cache?: string
  preview?: string
  text?: string
  // Hash of the piece with the cache markers removed, for the reuse diff.
  stable?: string
}

export type Prompt = {
  dialect: string
  settings: Record<string, unknown>
  totals: {
    chars: number
    system_chars: number
    tools_chars: number
    messages_chars: number
    tool_count: number
    message_count: number
  }
  system: Piece[]
  tools: Piece[]
  messages: Piece[]
}

// The prompt with its tokens attributed: the JSON view, plus the per-message
// numbers the reuse diff needs.
export type Attributed = {
  view: Record<string, unknown>
  perSystem: number[]
  perTool: number[]
  perMessage: number[]
  system: number
  tools: number
  messages: number
  total: number
  measured: boolean
}

// What a turn looked like, kept so the next turn of the same session can be
// diffed against it.
export type Fingerprint = {
  turn: number
  agent?: string
  system: string
  tools: string
  messages: string[]
  chars: number[]
}

export type Reuse = {
  previous_turn: number
  system_changed: boolean
  tools_changed: boolean
  stable_messages: number
  changed_at: number | null
  added_messages: number
  dropped_messages: number
  prefix_stable: boolean
  resent_chars: number
  new_chars: number
}

export type Reply = {
  text: string
  reasoning: string
  tool_calls: { name?: string; arguments: unknown; chars: number }[]
  stop_reason?: string
  chars: { text: number; reasoning: number; tool_calls: number; total: number }
}

export type Draft = {
  text: string[]
  reasoning: string[]
  calls: Map<string, { name?: string; args: string }>
  stop?: string
}

// The fields that identify a turn, at the head of its record.
export type Head = {
  turn: number
  provider: string | undefined
  model: string | undefined
  session: string | undefined
  agent: string | undefined
  started_at: string
  elapsed_seconds: number
}

// Running totals. Kept per model and per agent as well as overall, because
// tokens from a title model and tokens from the coding model are neither the
// same money nor the same problem.
export type Bucket = {
  turns: number
  // Turnos aqui dentro sem contagem nenhuma — o provider recusou, ou a resposta
  // nunca chegou. Contam como turno, não como tokens.
  uncounted: number
  usage: Usage
  cost: number
  provider_cost: number
}

/**
 * One conversation, and one folder.
 *
 * A process serves however many conversations you open — `/new` does not start
 * a new process — and one conversation outlives however many processes, since
 * resuming it starts a new one. So the folder is neither: it belongs to the
 * session, is found again by id when the session is resumed, and keeps its turn
 * numbering and its totals across every run that writes to it. Everything is
 * counted here rather than on the run, which only indexes what it touched.
 *
 * Subagents (the `task` tool) are sessions of their own on the wire, with their
 * own id and a `parentID`, but they are part of the work of whoever spawned
 * them, so their turns are filed under the root session's folder.
 */
export type Session = {
  id: string
  dir: string
  started: number
  turns: number
  errors: number
  incomplete: number
  reported: Bucket
  estimated: Bucket
  byModel: Map<string, Bucket>
  byAgent: Map<string, Bucket>
  // Per session id filed here, so what a subagent cost is a number and not an
  // inference from the agent breakdown.
  bySession: Map<string, Bucket>
  // The reuse chains. Keeping them on the session makes diffing one
  // conversation's prompt against another's structurally impossible.
  history: Map<string, Fingerprint>
  // Every session id filed under this folder: the root and its subagents.
  members: Set<string>
  // Child -> parent among them, kept so the chain survives a restart: the wire
  // only ever names the immediate parent.
  parents: Map<string, string>
  prompt: { system: number; tools: number; messages: number; total: number }
  resent: number
  breaks: number
  // Turns whose response had not landed yet. A request still in flight when the
  // process exits would otherwise leave no trace at all.
  pending: Map<number, Record<string, unknown>>
  // The runs that have written here, oldest first. More than one means the
  // conversation was resumed.
  runs: string[]
}

export type Saved = {
  session: string
  started: number
  turns: number
  errors: number
  incomplete: number
  reported: Bucket
  estimated: Bucket
  byModel: Map<string, Bucket>
  byAgent: Map<string, Bucket>
  bySession: Map<string, Bucket>
  history: Map<string, Fingerprint>
  members: Set<string>
  parents: Map<string, string>
  prompt: Session["prompt"]
  resent: number
  breaks: number
  runs: string[]
}

/**
 * Where a conversation lives: the same folder every time, so coming back to a
 * session months later writes next to what it wrote then.
 *
 * The name leads with the time the session was first seen, so a reader sorting
 * folder names — which is what the viewer does — reads them in the order the
 * conversations started. The name is only a label, though: what identifies the
 * folder is the session id in its `state.json`, because ten characters of an id
 * are not a promise of uniqueness.
 *
 * A subagent asked for by its own id lands on the folder of whoever spawned it,
 * because that folder lists it as a member — the fallback that keeps a session's
 * work together even when the parent chain did not survive the restart. Turns
 * that arrive without a session id have nothing to be matched on at all, so they
 * always open a folder of their own.
 */
export function sessionDir(base: string, id: string): { dir: string; id: string } {
  if (id !== NO_SESSION) {
    const pattern = new RegExp(`^\\d{8}_\\d{6}-${escaped(short(id))}(-\\d+)?$`)
    const folders = []
    for (const name of entriesOf(base).sort()) {
      const dir = path.join(base, name)
      if (pattern.test(name)) {
        const saved = readState(dir)
        if (saved?.session === id) return { dir, id }
      } else if (/^\d{8}_\d{6}-/.test(name)) folders.push(dir)
    }
    for (const dir of folders) {
      const saved = readState(dir)
      if (saved && saved.session !== NO_SESSION && saved.members.has(id)) return { dir, id: saved.session }
    }
  }
  const first = `${stamp(new Date())}-${short(id)}`
  let name = first
  for (let i = 2; fs.existsSync(path.join(base, name)); i++) name = `${first}-${i}`
  return { dir: path.join(base, name), id }
}

export function entriesOf(dir: string): string[] {
  try {
    return fs.readdirSync(dir)
  } catch {
    return []
  }
}

export function lastTurn(dir: string) {
  let highest = 0
  for (const name of entriesOf(dir)) {
    const match = /^turn_(\d+)/.exec(name)
    if (match) highest = Math.max(highest, Number(match[1]))
  }
  return highest
}

/**
 * The folder as the last run left it. Anything unreadable is treated as absent:
 * starting the counts over is a wrong number in one folder, while trusting a
 * file we cannot parse would merge two conversations.
 */
export function readState(dir: string): Saved | undefined {
  let data: unknown
  try {
    data = JSON.parse(fs.readFileSync(path.join(dir, "state.json"), "utf8"))
  } catch {
    return undefined
  }
  if (!isRecord(data) || typeof data["session"] !== "string") return undefined
  const prompt = isRecord(data["prompt"]) ? data["prompt"] : {}
  return {
    session: data["session"],
    started: num(data["started"]) || Date.now(),
    turns: num(data["turns"]),
    errors: num(data["errors"]),
    incomplete: num(data["incomplete"]),
    reported: reviveBucket(data["reported"]),
    estimated: reviveBucket(data["estimated"]),
    byModel: reviveBuckets(data["by_model"]),
    byAgent: reviveBuckets(data["by_agent"]),
    bySession: reviveBuckets(data["by_session"]),
    history: reviveHistory(data["history"]),
    members: new Set(strings(data["members"])),
    parents: reviveParents(data["parents"]),
    prompt: {
      system: num(prompt["system"]),
      tools: num(prompt["tools"]),
      messages: num(prompt["messages"]),
      total: num(prompt["total"]),
    },
    resent: num(data["resent"]),
    breaks: num(data["breaks"]),
    runs: strings(data["runs"]),
  }
}

function reviveParents(value: unknown): Map<string, string> {
  const map = new Map<string, string>()
  if (!isRecord(value)) return map
  for (const [child, parent] of Object.entries(value)) if (typeof parent === "string") map.set(child, parent)
  return map
}

function reviveBucket(value: unknown): Bucket {
  const created = bucket()
  if (!isRecord(value)) return created
  created.turns = num(value["turns"])
  created.uncounted = num(value["uncounted"])
  created.cost = num(value["cost"])
  created.provider_cost = num(value["provider_cost"])
  const usage = isRecord(value["usage"]) ? value["usage"] : {}
  for (const key of ["input", "cache_read", "cache_write", "output", "reasoning", "prompt", "total"] as const)
    created.usage[key] = num(usage[key])
  const details = isRecord(usage["details"]) ? usage["details"] : undefined
  const kept: Record<string, number> = {}
  for (const [key, amount] of Object.entries(details ?? {})) if (typeof amount === "number") kept[key] = amount
  if (Object.keys(kept).length) created.usage.details = kept
  return created
}

function reviveBuckets(value: unknown): Map<string, Bucket> {
  const map = new Map<string, Bucket>()
  if (!isRecord(value)) return map
  for (const [key, entry] of Object.entries(value)) map.set(key, reviveBucket(entry))
  return map
}

function reviveHistory(value: unknown): Map<string, Fingerprint> {
  const map = new Map<string, Fingerprint>()
  if (!isRecord(value)) return map
  for (const [key, entry] of Object.entries(value)) {
    if (!isRecord(entry)) continue
    if (typeof entry["system"] !== "string" || typeof entry["tools"] !== "string") continue
    if (!Array.isArray(entry["messages"]) || !Array.isArray(entry["chars"])) continue
    map.set(key, {
      turn: num(entry["turn"]),
      agent: str(entry["agent"]),
      system: entry["system"],
      tools: entry["tools"],
      messages: strings(entry["messages"]),
      chars: entry["chars"].map(num),
    })
  }
  return map
}

function strings(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : []
}

function escaped(value: string) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
}

export function short(id: string) {
  if (id === NO_SESSION) return NO_SESSION
  return id.replace(/^ses_/, "").slice(0, 10) || NO_SESSION
}

/**
 * Where a turn is written. The name is the turn number — unless a file of that
 * name is already there, which means two processes are on the same conversation
 * at once, and the one arriving second keeps its copy rather than overwriting
 * the other's. The turn number inside the record is the one that counts.
 */
export function turnPath(sess: Session, turn: number) {
  const name = `turn_${String(turn).padStart(3, "0")}`
  let file = path.join(sess.dir, `${name}.json`)
  for (let i = 2; fs.existsSync(file); i++) file = path.join(sess.dir, `${name}-${i}.json`)
  return file
}

/**
 * What the next run needs to continue this folder: the counters in the shape
 * they are kept in memory, so resuming restores them exactly rather than
 * reconstructing them from the rounded, human-facing summary next to it.
 */
export function checkpoint(sess: Session) {
  return {
    version: 1,
    session: sess.id,
    started: sess.started,
    updated: Date.now(),
    turns: sess.turns,
    errors: sess.errors,
    incomplete: sess.incomplete,
    reported: sess.reported,
    estimated: sess.estimated,
    by_model: Object.fromEntries(sess.byModel),
    by_agent: Object.fromEntries(sess.byAgent),
    by_session: Object.fromEntries(sess.bySession),
    prompt: sess.prompt,
    resent: sess.resent,
    breaks: sess.breaks,
    members: [...sess.members],
    parents: Object.fromEntries(sess.parents),
    history: Object.fromEntries(sess.history),
    runs: sess.runs,
  }
}

// ---------------------------------------------------------------------------
// Totals.
// ---------------------------------------------------------------------------

export function tally(
  sess: Session,
  sessionID: string | undefined,
  context: Context | undefined,
  model: string | undefined,
  usage: Usage | undefined,
  spend: Spend | undefined,
  tokens: Attributed | undefined,
  reuse: Reuse | undefined,
) {
  if (tokens) {
    sess.prompt.system += tokens.system
    sess.prompt.tools += tokens.tools
    sess.prompt.messages += tokens.messages
    sess.prompt.total += tokens.total
  }
  if (reuse) {
    sess.resent += resent(reuse, tokens)
    if (!reuse.prefix_stable) sess.breaks += 1
  }
  const name = `${context?.providerID ?? "?"}/${model ?? context?.model.id ?? "?"}`
  // Todo turno entra nas quebras, inclusive o que não comprou nada: uma linha
  // cujos turnos não somam o total da pasta é um turno que o leitor não vê. O
  // que ele não comprou fica à parte, em `uncounted`, e nunca vira token.
  for (const target of [
    into(sess.byModel, name),
    into(sess.byAgent, context?.agent ?? "?"),
    into(sess.bySession, sessionID ?? "?"),
  ]) {
    if (usage) accumulate(target, usage, spend)
    else {
      target.turns += 1
      target.uncounted += 1
    }
  }
  if (!usage) return
  accumulate(usage.source === "provider" ? sess.reported : sess.estimated, usage, spend)
}

function accumulate(target: Bucket, usage: Usage, spend: Spend | undefined) {
  target.turns += 1
  add(target.usage, usage)
  target.cost += spend?.estimated_usd ?? 0
  target.provider_cost += spend?.provider_usd ?? 0
}

export function into(map: Map<string, Bucket>, key: string) {
  const existing = map.get(key)
  if (existing) return existing
  const created = bucket()
  map.set(key, created)
  return created
}

export function bucket(): Bucket {
  return { turns: 0, uncounted: 0, usage: zero("provider"), cost: 0, provider_cost: 0 }
}

export function summary(sess: Session) {
  const reported = sess.reported
  const estimated = sess.estimated
  return {
    session: sess.id === NO_SESSION ? undefined : sess.id,
    // The subagents filed here, when there are any: they are sessions of their
    // own on the wire but part of this one's work.
    sessions_included: sess.members.size > 1 ? [...sess.members] : undefined,
    turns: sess.turns,
    errors: sess.errors,
    // Turns that were sent but whose response never arrived: their tokens are
    // missing from everything below.
    incomplete_turns: sess.incomplete || undefined,
    // The exact bill, and — kept apart so it cannot contaminate it — what the
    // turns the provider stayed silent about are worth at chars / 4.
    tokens: shape(reported),
    estimated_tokens: estimated.turns ? shape(estimated) : undefined,
    // Denominador: todos os turnos. Contar só os que tiveram contagem faz um
    // turno recusado sumir da frase que existe justamente para achá-lo.
    coverage: `${reported.turns}/${sess.turns} turns counted by the provider`,
    // Turnos sem contagem nenhuma: recusados pelo provider, ou sem resposta.
    uncounted_turns: sess.turns - reported.turns - estimated.turns || undefined,
    cost_usd: round(reported.cost + estimated.cost),
    provider_cost_usd: reported.provider_cost ? round(reported.provider_cost) : undefined,
    // Where the prompt tokens went, summed over the run.
    prompt_tokens_by_part: sess.prompt,
    // Prompt the model saw again because it was an unchanged prefix: cheap when
    // the provider caches it, paid in full when the prefix keeps breaking.
    resent_tokens: sess.resent,
    // Turns whose cacheable prefix changed. Every one of these is a cache miss
    // on the whole history — the first thing to look at when costs are high.
    prefix_breaks: sess.breaks,
    // Both breakdowns count every turn, estimated ones included.
    by_model: Object.fromEntries([...sess.byModel].map(([name, value]) => [name, shape(value)])),
    by_agent: Object.fromEntries([...sess.byAgent].map(([name, value]) => [name, shape(value)])),
    // Per conversation filed here: the root, and a row for each subagent it
    // spawned — what the `task` tool actually cost, without leaving the folder.
    by_session:
      sess.bySession.size > 1
        ? Object.fromEntries([...sess.bySession].map(([name, value]) => [name, shape(value)]))
        : undefined,
    // Who spawned whom, for a reader putting the rows above in order.
    session_parents: sess.parents.size ? Object.fromEntries(sess.parents) : undefined,
    // The conversation's own clock: first turn to last, which for a session
    // that was resumed spans the time it sat idle in between.
    elapsed_seconds: (Date.now() - sess.started) / 1000,
    started_at: new Date(sess.started).toISOString(),
    updated_at: new Date().toISOString(),
    // The runs that wrote here. More than one means the session was resumed,
    // and that everything above is the total over all of them.
    runs: sess.runs.length > 1 ? sess.runs : undefined,
    path: sess.dir,
  }
}

export function shape(value: Bucket) {
  const usage = value.usage
  return {
    turns: value.turns,
    input: usage.input,
    cache_read: usage.cache_read,
    cache_write: usage.cache_write,
    prompt: usage.prompt,
    output: usage.output,
    reasoning: usage.reasoning,
    total: usage.total,
    cache_hit_rate: hitRate(usage),
    // Turnos da linha que não trouxeram contagem: estão no `turns`, fora dos
    // tokens. Sem isto uma linha com zero token parece um erro de conta.
    uncounted_turns: value.uncounted || undefined,
    cost_usd: round(value.cost),
    provider_cost_usd: value.provider_cost ? round(value.provider_cost) : undefined,
    details: usage.details && Object.keys(usage.details).length ? usage.details : undefined,
  }
}

// ---------------------------------------------------------------------------
// Usage: the token bill, normalized across the dialects.
// ---------------------------------------------------------------------------

/**
 * Pulls the token counts out of a response payload.
 *
 * Providers report usage under `usage` / `usageMetadata` / `meta.billed_units`,
 * at the top level for a single JSON response and spread across frames while
 * streaming (Anthropic sends the input counts in `message_start` and the final
 * output count in `message_delta`). The raw counters are merged first, keeping
 * the largest value seen for each, and normalized once at the end — normalizing
 * each frame on its own would misread a frame that carries a prompt total
 * before the cached-token detail that belongs with it.
 */
export function usageOf(payload: unknown): Usage | undefined {
  const merged: Record<string, unknown> = {}
  let found = false
  const visit = (value: unknown, depth: number) => {
    if (depth > 8 || value === null || typeof value !== "object") return
    if (Array.isArray(value)) return value.forEach((item) => visit(item, depth + 1))
    for (const [key, item] of Object.entries(value)) {
      if (USAGE_KEY.test(key) && isRecord(item)) {
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
  for (const [key, value] of Object.entries(source)) {
    if (typeof value === "number" && Number.isFinite(value)) target[key] = Math.max(num(target[key]), value)
    else if (isRecord(value)) {
      const nested = record(target[key]) ?? {}
      merge(nested, value)
      target[key] = nested
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
export function normalize(value: Record<string, unknown>): Usage | undefined {
  const inputDetails = record(value["input_tokens_details"]) ?? record(value["prompt_tokens_details"]) ?? {}
  const outputDetails = record(value["output_tokens_details"]) ?? record(value["completion_tokens_details"]) ?? {}
  const creation = record(value["cache_creation"]) ?? {}

  const inclusive = num(inputDetails["cached_tokens"]) + num(value["cachedContentTokenCount"])
  const prompt =
    num(value["input_tokens"]) +
    num(value["inputTokens"]) +
    num(value["prompt_tokens"]) +
    num(value["promptTokenCount"])
  const write5m = num(creation["ephemeral_5m_input_tokens"])
  const write1h = num(creation["ephemeral_1h_input_tokens"])
  const thoughts = num(value["thoughtsTokenCount"])

  const usage: Usage = {
    // Gemini bills the prompt its built-in tools generate on top of the prompt.
    input: Math.max(0, prompt - inclusive) + num(value["toolUsePromptTokenCount"]),
    cache_read: inclusive + num(value["cache_read_input_tokens"]) + num(value["cacheReadInputTokens"]),
    cache_write: Math.max(
      num(value["cache_creation_input_tokens"]) + num(value["cacheWriteInputTokens"]),
      write5m + write1h,
    ),
    output:
      num(value["output_tokens"]) +
      num(value["outputTokens"]) +
      num(value["completion_tokens"]) +
      num(value["candidatesTokenCount"]) +
      thoughts,
    reasoning: num(outputDetails["reasoning_tokens"]) + num(value["reasoning_tokens"]) + thoughts,
    prompt: 0,
    total: 0,
    source: "provider",
  }
  usage.prompt = usage.input + usage.cache_read + usage.cache_write
  usage.total = usage.prompt + usage.output
  if (usage.total === 0) return undefined

  const details: Record<string, number> = {}
  const note = (key: string, amount: number) => {
    if (amount) details[key] = amount
  }
  // The cache TTLs are priced differently, so the split has to survive.
  note("cache_write_5m", write5m)
  note("cache_write_1h", write1h)
  note("audio_input", num(inputDetails["audio_tokens"]))
  note("audio_output", num(outputDetails["audio_tokens"]))
  note("accepted_prediction", num(outputDetails["accepted_prediction_tokens"]))
  note("rejected_prediction", num(outputDetails["rejected_prediction_tokens"]))
  for (const [key, amount] of Object.entries(record(value["server_tool_use"]) ?? {})) note(key, num(amount))
  // Kept as reported, so a disagreement with the total above is visible rather
  // than smoothed over.
  note(
    "reported_total",
    num(value["total_tokens"]) + num(value["totalTokens"]) + num(value["totalTokenCount"]),
  )
  // Gateways that bill in money rather than tokens (OpenRouter) report it here.
  note("provider_cost", num(value["cost"]))
  if (Object.keys(details).length) usage.details = details
  return usage
}

export function guess(prompt: Prompt | undefined, reply: Reply | undefined): Usage | undefined {
  if (!prompt) return undefined
  const usage = zero("estimated")
  usage.input = est(prompt.totals.chars)
  usage.output = est(reply?.chars.total ?? 0)
  usage.reasoning = est(reply?.chars.reasoning ?? 0)
  usage.prompt = usage.input
  usage.total = usage.prompt + usage.output
  return usage.total ? usage : undefined
}

export type Spend = {
  estimated_usd: number
  provider_usd?: number
  model?: string
  per_million?: { input: number; output: number; cache_read: number; cache_write: number }
  note?: string
}

/**
 * What the turn is worth at the model's catalog price. Anthropic's one-hour
 * cache writes are billed at twice the base input rate where the catalog's
 * `cache.write` is the five-minute price, so they are priced apart.
 */
export function price(usage: Usage | undefined, context: Context | undefined): Spend | undefined {
  if (!usage) return undefined
  const provider = num(usage.details?.["provider_cost"])
  if (!context) return provider ? { estimated_usd: round(provider), provider_usd: round(provider) } : undefined
  const cost = context.model.cost
  const long = num(usage.details?.["cache_write_1h"])
  const short = Math.max(0, usage.cache_write - long)
  const estimated =
    (usage.input * cost.input +
      usage.output * cost.output +
      usage.cache_read * cost.cache.read +
      short * cost.cache.write +
      long * cost.input * 2) /
    1_000_000
  return {
    estimated_usd: round(estimated),
    provider_usd: provider ? round(provider) : undefined,
    model: context.model.id,
    per_million: {
      input: cost.input,
      output: cost.output,
      cache_read: cost.cache.read,
      cache_write: cost.cache.write,
    },
    note: usage.source === "estimated" ? "token counts estimated: the provider reported none" : undefined,
  }
}

export function hitRate(usage: Usage | undefined) {
  if (!usage || !usage.prompt) return undefined
  return round(usage.cache_read / usage.prompt)
}

export function add(target: Usage, value: Usage) {
  target.input += value.input
  target.cache_read += value.cache_read
  target.cache_write += value.cache_write
  target.output += value.output
  target.reasoning += value.reasoning
  target.prompt += value.prompt
  target.total += value.total
  if (!value.details) return
  const details = target.details ?? {}
  for (const [key, amount] of Object.entries(value.details)) {
    if (key === "reported_total" || key === "provider_cost") continue
    details[key] = num(details[key]) + amount
  }
  if (Object.keys(details).length) target.details = details
}

export function zero(source: Usage["source"]): Usage {
  return { input: 0, cache_read: 0, cache_write: 0, output: 0, reasoning: 0, prompt: 0, total: 0, source }
}

// ---------------------------------------------------------------------------
// Attribution: the provider's one prompt number, spread over the parts.
// ---------------------------------------------------------------------------

export function attribute(prompt: Prompt | undefined, usage: Usage | undefined): Attributed | undefined {
  if (!prompt) return undefined
  const pieces = [...prompt.system, ...prompt.tools, ...prompt.messages]
  const chars = pieces.map((piece) => piece.chars)
  const billed = usage?.source === "provider" ? usage.prompt : 0
  const measured = billed > 0 && chars.some(Boolean)
  const tokens = distribute(measured ? billed : est(prompt.totals.chars), chars)

  const cut = (from: number, count: number) => tokens.slice(from, from + count)
  const systemTokens = cut(0, prompt.system.length)
  const toolTokens = cut(prompt.system.length, prompt.tools.length)
  const messageTokens = cut(prompt.system.length + prompt.tools.length, prompt.messages.length)
  const sum = (values: number[]) => values.reduce((total, value) => total + value, 0)
  const total = sum(tokens)

  const rows = (list: Piece[], counts: number[]) =>
    list.map((piece, index) => ({
      ...piece,
      tokens: counts[index],
      // The text lives verbatim in `request.body`; repeating it here would
      // double the size of every record.
      text: undefined,
      stable: undefined,
    }))

  return {
    view: {
      dialect: prompt.dialect,
      settings: prompt.settings,
      tokens: {
        total,
        system: sum(systemTokens),
        tools: sum(toolTokens),
        messages: sum(messageTokens),
        // Exact and split proportionally, or a chars / 4 guess throughout.
        source: measured ? "provider total, distributed by characters" : `characters / ${CHARS_PER_TOKEN}`,
        chars_per_token: total ? round2(prompt.totals.chars / total) : undefined,
      },
      totals: prompt.totals,
      system: rows(prompt.system, systemTokens),
      tools: rows(prompt.tools, toolTokens),
      messages: rows(prompt.messages, messageTokens),
    },
    perSystem: systemTokens,
    perTool: toolTokens,
    perMessage: messageTokens,
    system: sum(systemTokens),
    tools: sum(toolTokens),
    messages: sum(messageTokens),
    total,
    measured,
  }
}

/**
 * Splits `total` across `weights` as whole tokens that still add up to `total`
 * exactly, giving the leftover to the largest remainders.
 */
export function distribute(total: number, weights: number[]): number[] {
  const sum = weights.reduce((carry, weight) => carry + weight, 0)
  if (!sum) return weights.map(() => 0)
  const exact = weights.map((weight) => (weight * total) / sum)
  const result = exact.map(Math.floor)
  let left = total - result.reduce((carry, value) => carry + value, 0)
  const order = exact
    .map((value, index) => ({ remainder: value - Math.floor(value), index }))
    .sort((a, b) => b.remainder - a.remainder)
  for (const entry of order) {
    if (left <= 0) break
    result[entry.index] += 1
    left -= 1
  }
  return result
}

export function view(prompt: Prompt | undefined) {
  if (!prompt) return undefined
  const rows = (list: Piece[]) => list.map((piece) => ({ ...piece, text: undefined, stable: undefined }))
  return {
    dialect: prompt.dialect,
    settings: prompt.settings,
    totals: prompt.totals,
    system: rows(prompt.system),
    tools: rows(prompt.tools),
    messages: rows(prompt.messages),
  }
}

/**
 * The reply with its output tokens attributed. Reasoning is taken from the
 * provider's own count rather than guessed; what is left over is split between
 * the prose and the tool-call arguments by size.
 */
export function replyView(reply: Reply | undefined, usage: Usage | undefined) {
  if (!reply) return undefined
  const measured = usage?.source === "provider" && usage.output > 0
  // Providers that count thinking separately (OpenAI, Gemini) hand us the
  // number. Anthropic folds it into `output` without breaking it out, so its
  // share is split by size along with everything else rather than silently
  // landing on the prose.
  const counted = measured ? Math.min(usage.reasoning, usage.output) : 0
  const budget = Math.max(0, measured ? usage.output - counted : est(reply.chars.total))
  const [textTokens, reasoningTokens, callTokens] = distribute(budget, [
    reply.chars.text,
    counted ? 0 : reply.chars.reasoning,
    reply.chars.tool_calls,
  ])
  const calls = distribute(
    callTokens,
    reply.tool_calls.map((call) => call.chars),
  )
  return {
    stop_reason: reply.stop_reason,
    tokens: {
      output: counted + budget,
      reasoning: counted + reasoningTokens,
      text: textTokens,
      tool_calls: callTokens,
      source: measured ? "provider" : `characters / ${CHARS_PER_TOKEN}`,
    },
    chars: reply.chars,
    text: reply.text || undefined,
    reasoning: reply.reasoning || undefined,
    tool_calls: reply.tool_calls.map((call, index) => ({ ...call, tokens: calls[index] })),
  }
}

// ---------------------------------------------------------------------------
// Prompt: what the harness sent, decoded out of the provider's dialect.
// ---------------------------------------------------------------------------

export function readPrompt(body: unknown): Prompt | undefined {
  if (!isRecord(body)) return undefined
  const dialect = dialectOf(body)
  const system = systemPieces(body, dialect)
  const tools = toolPieces(body, dialect)
  const messages = messagePieces(body, dialect)

  const sum = (pieces: Piece[]) => pieces.reduce((total, piece) => total + piece.chars, 0)
  const systemChars = sum(system)
  const toolsChars = sum(tools)
  const messagesChars = sum(messages)
  const chars = systemChars + toolsChars + messagesChars
  for (const piece of [...system, ...tools, ...messages]) piece.share = chars ? round(piece.chars / chars) : 0

  return {
    dialect,
    settings: settingsOf(body),
    totals: {
      chars,
      system_chars: systemChars,
      tools_chars: toolsChars,
      messages_chars: messagesChars,
      tool_count: tools.length,
      message_count: messages.length,
    },
    system,
    tools,
    messages,
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

function systemPieces(body: Record<string, unknown>, dialect: string): Piece[] {
  if (dialect === "gemini") {
    const instruction = record(body["systemInstruction"]) ?? record(body["system_instruction"])
    const parts = arrayOf(instruction?.["parts"])
    return parts.length ? parts.map((part, index) => piece(index, part)) : instruction ? [piece(0, instruction)] : []
  }
  if (dialect === "openai-responses") {
    const instructions = body["instructions"]
    return instructions ? [piece(0, instructions)] : []
  }
  if (dialect === "openai-chat") {
    // OpenAI carries the system prompt as ordinary messages; lifting them out
    // keeps the system/tools/messages split comparable across dialects.
    return arrayOf(body["messages"])
      .filter((message) => isSystemRole(record(message)?.["role"]))
      .map((message, index) => piece(index, record(message)?.["content"] ?? message))
  }
  const system = body["system"]
  if (typeof system === "string") return system ? [piece(0, system)] : []
  return arrayOf(system).map((block, index) => piece(index, block))
}

function toolPieces(body: Record<string, unknown>, dialect: string): Piece[] {
  return toolList(body, dialect).map((tool, index) => {
    const spec = record(tool) ?? {}
    const schema = spec["input_schema"] ?? spec["parameters"] ?? spec["inputSchema"] ?? spec["parametersJsonSchema"]
    const serial = safeStringify(tool)
    return {
      index,
      name: str(spec["name"]) ?? str(spec["type"]),
      chars: serial.length,
      description_chars: sizeOf(spec["description"]),
      schema_chars: sizeOf(schema),
      cache: cacheOf(serial),
      preview: preview(str(spec["description"]) ?? serial),
      text: str(spec["description"]),
      stable: stamped(tool),
    }
  })
}

function toolList(body: Record<string, unknown>, dialect: string): unknown[] {
  if (dialect === "bedrock")
    return arrayOf(record(body["toolConfig"])?.["tools"]).map((tool) => record(tool)?.["toolSpec"] ?? tool)
  const tools = arrayOf(body["tools"])
  if (dialect === "gemini") {
    const declarations: unknown[] = []
    for (const tool of tools) {
      const list = record(tool)?.["functionDeclarations"] ?? record(tool)?.["function_declarations"]
      if (Array.isArray(list)) declarations.push(...list)
      else declarations.push(tool)
    }
    return declarations
  }
  // OpenAI nests the schema under `function`; flatten so every dialect reads
  // the same downstream.
  return tools.map((tool) => (isRecord(tool) && isRecord(tool["function"]) ? tool["function"] : tool))
}

function messagePieces(body: Record<string, unknown>, dialect: string): Piece[] {
  const list =
    dialect === "gemini"
      ? arrayOf(body["contents"])
      : dialect === "openai-responses"
        ? arrayOf(body["input"])
        : arrayOf(body["messages"])

  const pieces: Piece[] = []
  for (const message of list) {
    const value = record(message)
    const role = str(value?.["role"]) ?? str(value?.["type"]) ?? "message"
    if (dialect === "openai-chat" && isSystemRole(role)) continue
    // Gemini keeps parts under `parts`; the Responses API has bare items with
    // no content wrapper at all, so fall back to the item itself.
    const content = value ? (value["content"] ?? value["parts"] ?? value) : message
    const serial = safeStringify(message)
    const body = textOf(content)
    const kinds = kindsOf(content)
    pieces.push({
      index: pieces.length,
      role,
      kinds,
      chars: serial.length,
      binary: kinds.some((kind) => BINARY_KIND.test(kind)) || undefined,
      cache: cacheOf(serial),
      preview: preview(body),
      text: body || undefined,
      stable: stamped(message),
    })
  }
  return pieces
}

function settingsOf(body: Record<string, unknown>) {
  const settings: Record<string, unknown> = {}
  for (const key of SETTING_KEYS) if (key in body) settings[key] = body[key]
  return settings
}

function piece(index: number, value: unknown, extra: Partial<Piece> = {}): Piece {
  const serial = typeof value === "string" ? value : safeStringify(value)
  const body = textOf(value) || (typeof value === "string" ? value : "")
  return {
    index,
    chars: serial.length,
    cache: cacheOf(serial),
    ...extra,
    preview: preview(body || serial),
    text: body || undefined,
    stable: stamped(value),
  }
}

function kindsOf(content: unknown): string[] {
  if (typeof content === "string") return ["text"]
  const kinds = new Set<string>()
  const collect = (part: unknown) => {
    if (typeof part === "string") {
      kinds.add("text")
      return
    }
    const value = record(part)
    if (!value) return
    const type = str(value["type"])
    if (type) {
      kinds.add(type)
      return
    }
    // Gemini and Bedrock name the block by its key rather than a `type` field.
    for (const key of Object.keys(value)) if (key !== "role") kinds.add(key)
  }
  if (Array.isArray(content)) content.forEach(collect)
  else collect(content)
  return [...kinds]
}

// Keys that hold something a human would read, plus the ones that hold tool
// arguments — those are JSON, but they are part of what the model was sent.
const TEXT_KEYS = ["text", "thinking", "reasoning", "content", "parts", "output", "result", "response", "summary"]
const CODE_KEYS = ["input", "arguments", "args", "functionCall", "functionResponse", "toolUse", "toolResult"]

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
  for (const key of TEXT_KEYS) if (key in entry) parts.push(textOf(entry[key], depth + 1))
  for (const key of CODE_KEYS) if (key in entry) parts.push(safeStringify(entry[key]))
  return parts.filter(Boolean).join("\n")
}

/** The prompt part as the model reads it, with the per-request noise stripped.
 *
 *  Two things change from one turn to the next without a word of the prompt
 *  changing, and a byte comparison calls both of them a broken prefix:
 *
 *    the rolling cache marker  it moves forward as the conversation grows, and
 *                              the client sends the message it lands on wrapped
 *                              in a text block and the very same message, one
 *                              turn later, as a bare string — same text, other
 *                              bytes
 *    the billing header        it carries the previous request's id, so it is
 *                              new on every turn by construction
 *
 *  So the fingerprint runs over this projection and not over the raw serial:
 *  markers dropped, content shape normalized, volatile ids masked. Everything
 *  else stays, including whatever the marker was sitting on. */
function canonical(value: unknown, depth = 0): unknown {
  if (typeof value === "string") return value.replace(VOLATILE_ID, "")
  if (depth > 12 || value === null || typeof value !== "object") return value ?? null
  if (Array.isArray(value)) return value.map((item) => canonical(item, depth + 1))
  const entry = value as Record<string, unknown>
  const out: Record<string, unknown> = {}
  // Sorted: the fingerprint must not depend on the order the client happened to
  // serialize the keys in.
  for (const key of Object.keys(entry).sort()) {
    if (key === "cache_control" || key === "cachePoint") continue
    const item = entry[key]
    out[key] =
      key === "content" && typeof item === "string"
        ? [{ text: canonical(item, depth + 1), type: "text" }]
        : canonical(item, depth + 1)
  }
  return out
}

function stamped(value: unknown) {
  return hash(safeStringify(canonical(value)))
}

function cacheOf(serial: string) {
  const match = /"cache_control"\s*:\s*\{\s*"type"\s*:\s*"([^"]+)"/.exec(serial)
  if (match) return match[1]
  if (serial.includes('"cachePoint"')) return "point"
  return undefined
}

function isSystemRole(role: unknown) {
  return role === "system" || role === "developer"
}

// ---------------------------------------------------------------------------
// Reuse: how much of this prompt is the previous prompt, byte for byte.
// ---------------------------------------------------------------------------

export function track(
  sess: Session,
  turn: number,
  sessionID: string | undefined,
  context: Context | undefined,
  model: string | undefined,
  prompt: Prompt | undefined,
): Reuse | undefined {
  if (!prompt) return undefined
  // The chains live on the session, so two conversations can never be diffed
  // against each other. Within one folder there is still a chain per session id
  // — a subagent files here but is its own conversation — and per model and
  // dialect, so a small model called for titles does not read as a break.
  const key = [sessionID ?? "?", model ?? context?.model.id ?? "", prompt.dialect]
  const next: Fingerprint = {
    turn,
    agent: context?.agent || undefined,
    system: hash(prompt.system.map((piece) => piece.stable).join("\n")),
    tools: hash(prompt.tools.map((piece) => piece.stable).join("\n")),
    messages: prompt.messages.map((piece) => piece.stable ?? ""),
    chars: prompt.messages.map((piece) => piece.chars),
  }
  const previous = sess.history.get(key.join("|"))
  sess.history.set(key.join("|"), next)
  if (!previous) return undefined
  // The agent splits the chain further, but only when both sides say which one
  // they are: opencode runs its side agents (`title`, `summarize`) on the
  // session's own id, and diffing the build agent's first prompt against the
  // title agent's produces a turn where everything diverged at message #0 and
  // the user's own message reads as new, having travelled twice. An unknown
  // agent is not a different one, though — that would drop a good diff.
  if (previous.agent && next.agent && previous.agent !== next.agent) return undefined
  return compare(previous, next)
}

export function compare(previous: Fingerprint, next: Fingerprint): Reuse {
  let stable = 0
  while (
    stable < previous.messages.length &&
    stable < next.messages.length &&
    previous.messages[stable] === next.messages[stable]
  )
    stable++
  const sum = (values: number[]) => values.reduce((total, value) => total + value, 0)
  const systemChanged = previous.system !== next.system
  const toolsChanged = previous.tools !== next.tools
  return {
    previous_turn: previous.turn,
    system_changed: systemChanged,
    tools_changed: toolsChanged,
    stable_messages: stable,
    changed_at: stable < previous.messages.length ? stable : null,
    added_messages: next.messages.length - stable,
    dropped_messages: Math.max(0, previous.messages.length - stable),
    // True when this turn only appended: everything the provider could have
    // cached is still there, unchanged, in the same order.
    prefix_stable: !systemChanged && !toolsChanged && stable >= previous.messages.length,
    resent_chars: sum(next.chars.slice(0, stable)),
    new_chars: sum(next.chars.slice(stable)),
  }
}

export function resolveReuse(reuse: Reuse | undefined, tokens: Attributed | undefined) {
  if (!reuse) return undefined
  return {
    ...reuse,
    resent_tokens: resent(reuse, tokens),
    new_tokens: tokens
      ? tokens.perMessage.slice(reuse.stable_messages).reduce((total, value) => total + value, 0)
      : est(reuse.new_chars),
  }
}

export function resent(reuse: Reuse, tokens: Attributed | undefined) {
  if (!tokens) return est(reuse.resent_chars)
  return tokens.perMessage.slice(0, reuse.stable_messages).reduce((total, value) => total + value, 0)
}

// ---------------------------------------------------------------------------
// Reply: the model's output, reassembled from the stream.
// ---------------------------------------------------------------------------

export function readReply(payload: unknown): Reply | undefined {
  if (payload === undefined || payload === null || typeof payload === "string") return undefined
  const draft: Draft = { text: [], reasoning: [], calls: new Map() }
  if (Array.isArray(payload)) payload.forEach((event) => absorb(draft, event))
  else absorb(draft, payload)

  const text = draft.text.join("")
  const reasoning = draft.reasoning.join("")
  const calls = [...draft.calls.values()].filter((call) => call.name || call.args)
  if (!text && !reasoning && calls.length === 0 && !draft.stop) return undefined
  const callChars = calls.reduce((total, call) => total + call.args.length, 0)
  return {
    text,
    reasoning,
    tool_calls: calls.map((call) => ({ name: call.name, arguments: json(call.args), chars: call.args.length })),
    stop_reason: draft.stop,
    chars: {
      text: text.length,
      reasoning: reasoning.length,
      tool_calls: callChars,
      total: text.length + reasoning.length + callChars,
    },
  }
}

/**
 * Folds one payload into the draft. Handles the streaming frames and the
 * single-shot bodies of every dialect in one place, because they overlap: the
 * same block shapes show up in `content_block_start`, in a non-streaming
 * `content` array, and in a Responses `output` item.
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
      draft.stop = str(response?.["status"]) ?? draft.stop
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
    // arguments identify the call by its index in the stream.
    const target = call(draft, id)
    target.name = str(block["name"]) ?? target.name
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
  const created: { name?: string; args: string } = { args: "" }
  draft.calls.set(id, created)
  return created
}

function push(target: string[], value: unknown) {
  if (typeof value === "string" && value) target.push(value)
}

function key(value: unknown) {
  return typeof value === "string" || typeof value === "number" ? String(value) : "0"
}

function json(value: string) {
  try {
    return JSON.parse(value)
  } catch {
    return value
  }
}

export function text(value: unknown) {
  return JSON.stringify(value, (_key, item) => (typeof item === "bigint" ? String(item) : item), 2) + "\n"
}

export function safeStringify(value: unknown) {
  if (typeof value === "string") return value
  try {
    return JSON.stringify(value, (_key, item) => (typeof item === "bigint" ? String(item) : item)) ?? ""
  } catch {
    return "<unserializable>"
  }
}

function sizeOf(value: unknown) {
  return typeof value === "string" ? value.length : value === undefined ? 0 : safeStringify(value).length
}

export function est(chars: number) {
  return Math.round(chars / CHARS_PER_TOKEN)
}

function preview(value: string) {
  if (!value) return undefined
  const flat = value.replace(/\s+/g, " ").trim()
  return flat.length > PREVIEW_CHARS ? flat.slice(0, PREVIEW_CHARS) + "…" : flat
}

function hash(value: string) {
  return createHash("sha1").update(value).digest("hex").slice(0, 16)
}

export function stamp(date: Date) {
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

export function round(value: number) {
  return Math.round(value * 1_000_000) / 1_000_000
}

function round2(value: number) {
  return Math.round(value * 100) / 100
}

export function num(value: unknown) {
  return typeof value === "number" && Number.isFinite(value) ? value : 0
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
}

export function record(value: unknown) {
  return isRecord(value) ? value : undefined
}

export function arrayOf(value: unknown): unknown[] {
  return Array.isArray(value) ? value : []
}

export function str(value: unknown) {
  return typeof value === "string" && value ? value : undefined
}

