#!/usr/bin/env bun
// Per-turn audit log for Claude Code, in the format `llm-audit.ts` writes for
// opencode.
//
// Claude Code is closed and wraps no `fetch`, so nothing here captures: it
// reads what Claude Code already puts on disk and turns it into the same
// folders the opencode plugin writes. Two sources, joined per turn:
//
//   the transcript   ~/.claude/projects/<slug>/<session>.jsonl — the exact
//                    token bill the provider reported (`message.usage`), the
//                    model, the effort, the reply with its thinking intact
//   the raw bodies   what `OTEL_LOG_RAW_API_BODIES=file:<dir>` leaves behind —
//                    the untruncated Messages API request (system prompt, tool
//                    schemas, every message, the sampling settings) and response
//
// The transcript alone is enough to produce a turn; the bodies are what add the
// decomposed prompt and the reuse diff. With the bodies switched off, turns
// still come out, carrying `prompt: undefined` and saying why.
//
// A turn here is one request to the model, which is what `requestId` names. A
// single thing you typed usually spans several, because each tool call
// round-trips again. Careful: one response is persisted as several transcript
// entries, one per content block, each repeating the same `usage` — so the
// entries are grouped by `requestId` and the bill is counted once.
//
// Run from a hook, which hands it the session on stdin:
//
//   "Stop":        one pass per turn
//   "SubagentStop": the same, so a subagent's turns land as they finish
//   "SessionEnd":  a last catch-up pass
//
// Reads `CLAUDE_LLM_AUDIT_DIR` (where to write, default
// `~/.local/share/claude-code/log/llm-audit`), `CLAUDE_LLM_AUDIT_BODIES` or
// `OTEL_LOG_RAW_API_BODIES` (where the raw bodies are), `CLAUDE_LLM_AUDIT=0`
// (off), and `CLAUDE_LLM_AUDIT_INLINE=0` (point at the raw request body on disk
// instead of copying it into every turn record — smaller folders, but the
// viewer needs the body inline).
//
// `--prune <days>` drops raw body files older than that many days and exits;
// they are the part that grows without bound.
//
// The bodies hold the whole conversation, system prompt included. Treat the
// directory as sensitive.
import fs from "node:fs"
import os from "node:os"
import path from "node:path"
import {
  NO_SESSION,
  attribute,
  checkpoint,
  bucket,
  hitRate,
  isRecord,
  lastTurn,
  normalize,
  num,
  price,
  readPrompt,
  readReply,
  readState,
  record,
  replyView,
  resolveReuse,
  sessionDir,
  str,
  summary,
  tally,
  text,
  track,
  turnPath,
  view,
  type Context,
  type ModelInfo,
  type Session,
  type Usage,
} from "./core.ts"

type Hook = {
  session_id?: string
  transcript_path?: string
  cwd?: string
  prompt_id?: string
  permission_mode?: string
  hook_event_name?: string
  agent_id?: string
  agent_type?: string
}

type Entry = Record<string, unknown>

/** One request to the model: every transcript entry that shares a `requestId`. */
type Group = {
  requestId: string
  entries: Entry[]
  content: unknown[]
  messageId?: string
  model?: string
  stopReason?: string
  usage?: Record<string, unknown>
  sidechain: boolean
  // The conversation this turn belongs to: the session, or a subagent of it.
  session: string
  agent: string
  // When the input was ready, when the first block landed, when the last did.
  askedAt?: string
  startedAt: string
  endedAt: string
  version?: string
  gitBranch?: string
  cwd?: string
  effort?: string
  uuid: string
}

type Settings = {
  enabled: boolean
  dir: string
  bodies?: string
  inline: boolean
}

type Body = { id: string; file: string; mtime: number }

const LOCK_STALE_MS = 120_000
const UUID = /[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/gi

function main() {
  const settings = resolve()
  try {
    if (process.argv.includes("--prune")) return prune(settings)
    if (!settings.enabled) return
    const hook = readHook()
    if (!hook.transcript_path || !fs.existsSync(hook.transcript_path)) return
    fs.mkdirSync(settings.dir, { recursive: true })
    relocateBodies(settings)
    collect(settings, hook)
  } catch (error) {
    // A hook that throws is a hook that breaks the turn it was watching. The
    // failure goes to a file and the process still exits 0.
    try {
      fs.mkdirSync(settings.dir, { recursive: true })
      fs.appendFileSync(
        path.join(settings.dir, "errors.log"),
        `${new Date().toISOString()} ${error instanceof Error ? (error.stack ?? error.message) : String(error)}\n`,
      )
    } catch {
      // Nothing useful to do.
    }
  }
}

function resolve(): Settings {
  const env = process.env
  const flag = env["CLAUDE_LLM_AUDIT"]
  const dir =
    env["CLAUDE_LLM_AUDIT_DIR"] ??
    path.join(env["XDG_DATA_HOME"] ?? path.join(os.homedir(), ".local", "share"), "claude-code", "log", "llm-audit")
  // Where the raw bodies are, in order of how much it can be trusted.
  //
  // `OTEL_LOG_RAW_API_BODIES` is where Claude Code was told to put them, but a
  // hook does not necessarily get to read it: only the telemetry variables
  // Claude Code classifies as safe reach a subprocess, and that one does not.
  // So the last resort is the conventional place — `bodies/` next to the
  // folders — which is where the settings file points it anyway.
  const raw = env["OTEL_LOG_RAW_API_BODIES"] ?? ""
  // Ao lado do diretório de auditoria, nunca dentro dele: os corpos são `.json`
  // de ~1 MB e o viewer varre recursivamente tudo que termina em `.json`.
  const sibling = `${dir}-bodies`
  const inside = path.join(dir, "bodies")
  const chosen =
    env["CLAUDE_LLM_AUDIT_BODIES"] ??
    (raw.startsWith("file:")
      ? raw.slice(5)
      : fs.existsSync(sibling)
        ? sibling
        : fs.existsSync(inside)
          ? inside
          : undefined)
  // Aimed at the audit directory itself, they are read from the sibling
  // instead, because that is where `relocateBodies` moves them every pass.
  const bodies = chosen && path.resolve(chosen) === path.resolve(dir) ? sibling : chosen
  return {
    enabled: flag === undefined ? true : flag !== "" && flag !== "0" && flag.toLowerCase() !== "false",
    dir,
    bodies,
    inline: env["CLAUDE_LLM_AUDIT_INLINE"] !== "0",
  }
}

/**
 * Moves raw bodies out of the audit directory, when that is where they landed.
 *
 * `OTEL_LOG_RAW_API_BODIES` is read once at startup and points wherever it was
 * pointed; aim it at the audit directory and every request drops a ~200 KB
 * `.json` next to the conversation folders, which is what the viewer's
 * recursive scan reads as a turn. So each pass sweeps them into `<dir>-bodies`
 * and reads them there — which is also what makes the fallback in `resolve()`
 * find them on the next run, without the variable a hook never sees.
 */
function relocateBodies(settings: Settings) {
  let names: string[]
  try {
    names = fs.readdirSync(settings.dir)
  } catch {
    return
  }
  const loose = names.filter((name) => name.endsWith(".request.json") || name.endsWith(".response.json"))
  if (!loose.length) return
  const bodies =
    settings.bodies && path.resolve(settings.bodies) !== path.resolve(settings.dir)
      ? settings.bodies
      : `${settings.dir}-bodies`
  fs.mkdirSync(bodies, { recursive: true })
  for (const name of loose) {
    try {
      fs.renameSync(path.join(settings.dir, name), path.join(bodies, name))
    } catch {
      // Still being written, or already gone: the next pass gets it.
    }
  }
  settings.bodies = bodies
}

function readHook(): Hook {
  let input = ""
  try {
    input = fs.readFileSync(0, "utf8")
  } catch {
    return {}
  }
  try {
    const value = JSON.parse(input)
    return isRecord(value) ? (value as Hook) : {}
  } catch {
    return {}
  }
}

// ---------------------------------------------------------------------------
// The pass: read the transcript, write whatever turns are not on disk yet.
// ---------------------------------------------------------------------------

function collect(settings: Settings, hook: Hook) {
  const entries = readTranscript(hook.transcript_path!)
  if (!entries.length) return
  const id = hook.session_id ?? str(entries[0]?.["sessionId"]) ?? NO_SESSION
  const version = str(entries[entries.length - 1]?.["version"])

  const found = sessionDir(settings.dir, id)
  fs.mkdirSync(found.dir, { recursive: true })
  const release = lock(found.dir)
  // Another hook is already on this folder. It reads the same transcript, so
  // it takes these turns with it; anything it misses the next Stop picks up.
  if (!release) return
  try {
    const sess = openSession(found, version)
    const extras = readExtras(found.dir)
    const pending = group(entries, extras.seen)
    if (!pending.length) return

    const bodies = settings.bodies ? index(settings.bodies) : { requests: [], responses: new Map<string, string>() }
    const claimed = new Set<string>()
    for (const turn of pending) {
      write(settings, sess, turn, hook, bodies, claimed)
      extras.seen.add(turn.requestId)
      extras.last = turn.uuid
    }
    save(sess, extras)
  } finally {
    release()
  }
}

function readTranscript(file: string): Entry[] {
  const entries: Entry[] = []
  for (const line of fs.readFileSync(file, "utf8").split("\n")) {
    if (!line.trim()) continue
    try {
      const value = JSON.parse(line)
      if (isRecord(value)) entries.push(value)
    } catch {
      // A half-written last line while the session is live. It arrives whole
      // on the next pass.
    }
  }
  return entries
}

/**
 * The turns a pass has to write: the requests whose id is not already on disk.
 *
 * One response is several transcript entries — one per content block, all
 * carrying the same `message.id`, the same `requestId` and the same `usage` —
 * so the entries are folded back into the request they came from and the bill
 * is taken once. `askedAt` is the timestamp of whatever preceded the group,
 * which is the closest a hook can get to how long the request took: Claude Code
 * reports the real duration only over OpenTelemetry.
 */
function group(entries: Entry[], seen: Set<string>): Group[] {
  const byUuid = new Map<string, Entry>()
  for (const entry of entries) {
    const uuid = str(entry["uuid"])
    if (uuid) byUuid.set(uuid, entry)
  }

  const groups = new Map<string, Group>()
  const order: string[] = []
  let previous: string | undefined
  for (const entry of entries) {
    const timestamp = str(entry["timestamp"])
    if (entry["type"] !== "assistant") {
      if (timestamp) previous = timestamp
      continue
    }
    const message = record(entry["message"])
    const key = str(entry["requestId"]) ?? str(message?.["id"]) ?? str(entry["uuid"])
    if (!key) continue
    if (seen.has(key)) {
      if (timestamp) previous = timestamp
      continue
    }
    const sidechain = entry["isSidechain"] === true
    let group = groups.get(key)
    if (!group) {
      const root = sidechain ? subagentRoot(entry, byUuid) : undefined
      group = {
        requestId: key,
        entries: [],
        content: [],
        messageId: str(message?.["id"]),
        model: str(message?.["model"]),
        stopReason: str(message?.["stop_reason"]),
        usage: record(message?.["usage"]),
        sidechain,
        session: root ? `${str(entry["sessionId"]) ?? NO_SESSION}#${root.slice(0, 8)}` : (str(entry["sessionId"]) ?? NO_SESSION),
        agent: sidechain ? "subagent" : "main",
        askedAt: previous,
        startedAt: timestamp ?? new Date().toISOString(),
        endedAt: timestamp ?? new Date().toISOString(),
        version: str(entry["version"]),
        gitBranch: str(entry["gitBranch"]),
        cwd: str(entry["cwd"]),
        effort: str(entry["effort"]),
        uuid: str(entry["uuid"]) ?? key,
      }
      groups.set(key, group)
      order.push(key)
    }
    group.entries.push(entry)
    for (const block of Array.isArray(message?.["content"]) ? (message["content"] as unknown[]) : []) group.content.push(block)
    // The last entry of the group carries the stop reason the turn ended on.
    group.stopReason = str(message?.["stop_reason"]) ?? group.stopReason
    if (timestamp) {
      group.endedAt = timestamp
      previous = timestamp
    }
    group.uuid = str(entry["uuid"]) ?? group.uuid
  }
  return order.map((key) => groups.get(key)!)
}

/**
 * Which subagent a sidechain entry belongs to: the topmost sidechain entry of
 * its parent chain. The transcript gives a subagent no id of its own, and two
 * subagents running at once would otherwise share one reuse chain and be
 * diffed against each other.
 */
function subagentRoot(entry: Entry, byUuid: Map<string, Entry>): string | undefined {
  let current = entry
  for (let step = 0; step < 1000; step++) {
    const parent = byUuid.get(str(current["parentUuid"]) ?? "")
    if (!parent || parent["isSidechain"] !== true) return str(current["uuid"])
    current = parent
  }
  return str(current["uuid"])
}

// ---------------------------------------------------------------------------
// One turn.
// ---------------------------------------------------------------------------

function write(
  settings: Settings,
  sess: Session,
  group: Group,
  hook: Hook,
  bodies: { requests: Body[]; responses: Map<string, string> },
  claimed: Set<string>,
) {
  const turn = ++sess.turns
  const usage = usageOfGroup(group)
  const model = group.model
  const speed = str(group.usage?.["speed"])
  const context = contextOf(group, model, speed)
  const priced = catalog(model, speed)

  const responseFile = bodies.responses.get(group.requestId)
  const requestFile = matchRequest(bodies, group, claimed, responseFile)
  const requestBody = requestFile ? read(requestFile.file) : undefined
  const prompt = readPrompt(requestBody)

  const reuse = track(sess, turn, group.session, context, model, prompt)
  const tokens = attribute(prompt, usage)
  const spend = priced ? price(usage, context) : undefined
  // The reply is assembled from the transcript rather than from the raw body:
  // the body has the thinking redacted out of it, the transcript does not.
  const reply = readReply({ content: group.content, stop_reason: group.stopReason })

  sess.members.add(group.session)
  if (group.sidechain && sess.id !== NO_SESSION) sess.parents.set(group.session, sess.id)
  tally(sess, group.session, context, model, usage, spend, tokens, reuse)

  const started = Date.parse(group.askedAt ?? group.startedAt)
  const elapsed = Number.isFinite(started) ? (Date.parse(group.startedAt) - started) / 1000 : 0

  const body: Record<string, unknown> = {
    turn,
    provider: "anthropic",
    model,
    session: group.session,
    agent: hook.agent_type ?? group.agent,
    started_at: group.startedAt,
    // Not the request duration Claude Code measures — it never hands that to a
    // hook — but the gap between the input being ready and the first block of
    // the answer landing, which is the same thing to within a write.
    elapsed_seconds: elapsed >= 0 ? elapsed : 0,
    parent_session: group.sidechain && sess.id !== NO_SESSION ? sess.id : undefined,
    root_session: sess.id === NO_SESSION ? undefined : sess.id,
    usage,
    cost: spend,
    cache_hit_rate: hitRate(usage),
    prompt: tokens?.view ?? view(prompt),
    prompt_note: prompt
      ? undefined
      : settings.bodies
        ? "no raw request body matched this turn, so the prompt could not be decomposed"
        : "raw bodies are off (set OTEL_LOG_RAW_API_BODIES=file:<dir>), so the prompt could not be decomposed",
    reuse: resolveReuse(reuse, tokens),
    reply: replyView(reply, usage),
    request: {
      // `POST` is the only thing about the envelope that is known: Claude Code
      // writes the body, never the URL or the headers it went out with.
      method: "POST",
      url: null,
      headers: null,
      // The body has to be here, not merely pointed at: the viewer keys the
      // whole decoded format off `request.body`, and without it reads `request`
      // itself as the body. When the turn has no raw body, this carries the one
      // thing about the request that is known from the transcript — never a
      // reconstruction of a prompt that was not observed.
      body: settings.inline ? (requestBody ?? { model }) : undefined,
      // Which `.request.json` this turn was matched to, and how sure that is:
      // "exact" when the response named it, "adjacent" when it is the closest
      // request written before the response, "none" when there was nothing.
      match: requestFile ? requestFile.match : "none",
      path: requestFile?.file,
    },
    response: {
      status: 200,
      headers: null,
      bytes: responseFile ? sizeOf(responseFile) : undefined,
      // The response as the transcript kept it — thinking included, which the
      // raw body redacts. `path` is the untruncated body on disk.
      body: {
        id: group.messageId,
        model,
        stop_reason: group.stopReason,
        content: group.content,
        usage: group.usage,
      },
      path: responseFile,
    },
    // What only Claude Code knows about the turn.
    claude_code: {
      request_id: group.requestId,
      message_id: group.messageId,
      prompt_id: hook.prompt_id,
      hook_event: hook.hook_event_name,
      effort: group.effort,
      speed,
      permission_mode: hook.permission_mode,
      sidechain: group.sidechain || undefined,
      agent_id: hook.agent_id,
      version: group.version,
      cwd: group.cwd ?? hook.cwd,
      git_branch: group.gitBranch,
      transcript_uuid: group.uuid,
      // The transcript format is internal to Claude Code and changes between
      // releases; the version above is what a future break is diagnosed with.
      joined_on: "requestId",
    },
  }

  const index = {
    turn,
    time: group.startedAt,
    session: group.session,
    parent_session: body["parent_session"],
    root_session: body["root_session"],
    agent: body["agent"],
    model,
    source: usage?.source,
    input_tokens: usage?.input,
    cache_read_tokens: usage?.cache_read,
    cache_write_tokens: usage?.cache_write,
    prompt_tokens: usage?.prompt,
    output_tokens: usage?.output,
    reasoning_tokens: usage?.reasoning,
    total_tokens: usage?.total,
    cache_hit_rate: hitRate(usage),
    system_tokens: tokens?.system,
    tools_tokens: tokens?.tools,
    messages_tokens: tokens?.messages,
    stable_messages: reuse?.stable_messages,
    prefix_stable: reuse?.prefix_stable,
    stop_reason: group.stopReason,
    tool_calls: reply?.tool_calls.map((call) => call.name).filter(Boolean),
    cost_usd: spend?.estimated_usd,
    seconds: body["elapsed_seconds"],
    request_id: group.requestId,
    prompt_id: hook.prompt_id,
    effort: group.effort,
    request_match: requestFile ? requestFile.match : "none",
  }

  fs.writeFileSync(turnPath(sess, turn), text(body))
  fs.appendFileSync(path.join(sess.dir, "turns.jsonl"), JSON.stringify(index) + "\n")
}

/**
 * The turn's token bill, taken off the transcript.
 *
 * The counts are the provider's own, so they go through `normalize` unchanged
 * — with one rename: Claude Code writes the thinking tokens as
 * `output_tokens_details.thinking_tokens`, where `normalize` reads
 * `reasoning_tokens`. Left alone, the reasoning count would silently be zero.
 */
function usageOfGroup(group: Group): Usage | undefined {
  const raw = group.usage
  if (!raw) return undefined
  const value: Record<string, unknown> = { ...raw }
  const details = record(raw["output_tokens_details"])
  if (details && details["reasoning_tokens"] === undefined && details["thinking_tokens"] !== undefined)
    value["output_tokens_details"] = { ...details, reasoning_tokens: details["thinking_tokens"] }
  // `iterations` repeats the same counters one level down, and merging them in
  // would double the turn.
  delete value["iterations"]
  return normalize(value)
}

/** The catalog price for a model, and for fast mode separately — it is billed apart. */
function catalog(model: string | undefined, speed: string | undefined): ModelInfo["cost"] | undefined {
  if (!model) return undefined
  const table = prices()
  return (speed === "fast" ? table[`${model}:fast`] : undefined) ?? table[model]
}

let cached: Record<string, ModelInfo["cost"]> | undefined
function prices(): Record<string, ModelInfo["cost"]> {
  if (cached) return cached
  cached = {}
  try {
    const parsed = JSON.parse(fs.readFileSync(new URL("./prices.json", import.meta.url), "utf8"))
    const models = record(parsed)?.["models"]
    for (const [name, entry] of Object.entries(record(models) ?? {})) {
      const cost = record(entry)
      const cache = record(cost?.["cache"])
      if (!cost || !cache) continue
      cached[name] = {
        input: num(cost["input"]),
        output: num(cost["output"]),
        cache: { read: num(cache["read"]), write: num(cache["write"]) },
      }
    }
  } catch {
    // No catalog, no cost. Never a made-up number.
  }
  return cached
}

/**
 * The context the counters are grouped by. Always built, even for a model the
 * catalog has never heard of, so an unpriced turn still lands in the right row
 * of `by_agent` and `by_session`; the zeroed cost is simply never priced.
 */
function contextOf(group: Group, model: string | undefined, speed: string | undefined): Context | undefined {
  if (!model) return undefined
  return {
    sessionID: group.session,
    agent: group.agent,
    providerID: "anthropic",
    model: {
      id: model,
      providerID: "anthropic",
      cost: catalog(model, speed) ?? { input: 0, output: 0, cache: { read: 0, write: 0 } },
    },
  }
}

// ---------------------------------------------------------------------------
// The raw bodies, and matching them to turns.
// ---------------------------------------------------------------------------

function index(dir: string) {
  const requests: Body[] = []
  const responses = new Map<string, string>()
  let names: string[]
  try {
    names = fs.readdirSync(dir)
  } catch {
    return { requests, responses }
  }
  for (const name of names) {
    const file = path.join(dir, name)
    if (name.endsWith(".response.json")) {
      // Named for the request id, which is what the transcript calls it too.
      responses.set(name.slice(0, -".response.json".length), file)
      continue
    }
    if (!name.endsWith(".request.json")) continue
    let mtime = 0
    try {
      mtime = fs.statSync(file).mtimeMs
    } catch {
      continue
    }
    requests.push({ id: name.slice(0, -".request.json".length), file, mtime })
  }
  requests.sort((a, b) => a.mtime - b.mtime)
  return { requests, responses }
}

/**
 * Which `.request.json` belongs to a turn.
 *
 * The response side is exact: Claude Code names that file for the request id,
 * which is the `requestId` on the transcript's assistant entries. The request
 * side is named for a client-generated uuid the transcript never records, so it
 * is matched two ways, and the record says which one was used:
 *
 *   exact     the response body names the uuid, so the pair is certain
 *   adjacent  it is the last unclaimed request written before the response,
 *             for the same model — right in ordinary use, a guess when several
 *             sessions are writing into the directory at once
 */
function matchRequest(
  bodies: { requests: Body[]; responses: Map<string, string> },
  group: Group,
  claimed: Set<string>,
  responseFile: string | undefined,
): (Body & { match: string }) | undefined {
  if (!bodies.requests.length) return undefined

  if (responseFile) {
    // The cheap exact path: any uuid in the response body that is also a
    // request file sitting in the directory.
    const raw = readText(responseFile)
    if (raw) {
      for (const candidate of raw.match(UUID) ?? []) {
        const hit = bodies.requests.find((entry) => entry.id === candidate.toLowerCase())
        if (hit && !claimed.has(hit.id)) {
          claimed.add(hit.id)
          return { ...hit, match: "exact" }
        }
      }
    }
  }

  const deadline = responseFile ? mtimeOf(responseFile) : Date.parse(group.endedAt)
  if (!Number.isFinite(deadline)) return undefined
  let best: Body | undefined
  for (const entry of bodies.requests) {
    if (claimed.has(entry.id)) continue
    if (entry.mtime > deadline) break
    // A request for a different model is a different turn.
    if (group.model && modelOf(entry.file) && modelOf(entry.file) !== group.model) continue
    best = entry
  }
  if (!best) return undefined
  claimed.add(best.id)
  return { ...best, match: "adjacent" }
}

const models = new Map<string, string | undefined>()
function modelOf(file: string) {
  if (models.has(file)) return models.get(file)
  // Only the head of the file: the model is the cheapest field to reach and
  // these bodies run to megabytes.
  let value: string | undefined
  try {
    const handle = fs.openSync(file, "r")
    const buffer = Buffer.alloc(4096)
    const read = fs.readSync(handle, buffer, 0, 4096, 0)
    fs.closeSync(handle)
    value = /"model"\s*:\s*"([^"]+)"/.exec(buffer.subarray(0, read).toString("utf8"))?.[1]
  } catch {
    value = undefined
  }
  models.set(file, value)
  return value
}

function sizeOf(file: string) {
  try {
    return fs.statSync(file).size
  } catch {
    return undefined
  }
}

function mtimeOf(file: string) {
  try {
    return fs.statSync(file).mtimeMs
  } catch {
    return NaN
  }
}

function read(file: string): unknown {
  const raw = readText(file)
  if (raw === undefined) return undefined
  try {
    return JSON.parse(raw)
  } catch {
    return undefined
  }
}

function readText(file: string): string | undefined {
  try {
    return fs.readFileSync(file, "utf8")
  } catch {
    return undefined
  }
}

/** Drops raw bodies older than `--prune <days>`; they are what grows forever. */
function prune(settings: Settings) {
  const days = Number(process.argv[process.argv.indexOf("--prune") + 1]) || 7
  if (!settings.bodies) return
  const cutoff = Date.now() - days * 86_400_000
  let dropped = 0
  for (const name of fs.readdirSync(settings.bodies)) {
    if (!name.endsWith(".request.json") && !name.endsWith(".response.json")) continue
    const file = path.join(settings.bodies, name)
    try {
      if (fs.statSync(file).mtimeMs < cutoff) {
        fs.unlinkSync(file)
        dropped++
      }
    } catch {
      // Gone already, or in use.
    }
  }
  process.stdout.write(`dropped ${dropped} body files older than ${days} days from ${settings.bodies}\n`)
}

// ---------------------------------------------------------------------------
// The folder: opened where the conversation left it, saved so the next pass
// picks it up.
// ---------------------------------------------------------------------------

function openSession(found: { dir: string; id: string }, version: string | undefined): Session {
  let prior = readState(found.dir)
  if (prior && prior.session !== found.id) prior = undefined
  // In opencode a run is a process. Claude Code's process is the app itself,
  // which outlives everything, so what is worth recording per folder is which
  // versions of Claude Code wrote it — that is what a future format break is
  // read against.
  const runs = [...(prior?.runs ?? [])]
  if (version && !runs.includes(version)) runs.push(version)
  return {
    id: found.id,
    dir: found.dir,
    started: prior?.started ?? Date.now(),
    // Never below what is on disk: a lost state file must not make this pass
    // overwrite the turns the last one wrote.
    turns: Math.max(prior?.turns ?? 0, lastTurn(found.dir)),
    errors: prior?.errors ?? 0,
    incomplete: prior?.incomplete ?? 0,
    reported: prior?.reported ?? bucket(),
    estimated: prior?.estimated ?? bucket(),
    byModel: prior?.byModel ?? new Map(),
    byAgent: prior?.byAgent ?? new Map(),
    bySession: prior?.bySession ?? new Map(),
    history: prior?.history ?? new Map(),
    members: prior?.members ?? new Set(),
    parents: prior?.parents ?? new Map(),
    prompt: prior?.prompt ?? { system: 0, tools: 0, messages: 0, total: 0 },
    resent: prior?.resent ?? 0,
    breaks: prior?.breaks ?? 0,
    pending: new Map(),
    runs,
  }
}

/** What the next pass needs beyond the counters: where it stopped reading. */
function readExtras(dir: string): { seen: Set<string>; last?: string } {
  try {
    const value = JSON.parse(fs.readFileSync(path.join(dir, "state.json"), "utf8"))
    const seen = record(value)?.["seen_requests"]
    return {
      seen: new Set(Array.isArray(seen) ? seen.filter((item): item is string => typeof item === "string") : []),
      last: str(record(value)?.["last_uuid"]),
    }
  } catch {
    return { seen: new Set() }
  }
}

function save(sess: Session, extras: { seen: Set<string>; last?: string }) {
  // core's checkpoint, so both sides read the same file, plus the two things
  // only an incremental reader needs: where it stopped.
  const state: Record<string, unknown> = { ...checkpoint(sess) }
  state["seen_requests"] = [...extras.seen]
  state["last_uuid"] = extras.last
  fs.writeFileSync(path.join(sess.dir, "summary.json"), text(summary(sess)))
  fs.writeFileSync(path.join(sess.dir, "state.json"), text(state))
}


/**
 * One writer per folder. `Stop`, `SubagentStop` and a second session on the
 * same conversation can all land at once, and two passes numbering turns from
 * the same counter would write over each other.
 */
function lock(dir: string): (() => void) | undefined {
  const file = path.join(dir, ".lock")
  for (let attempt = 0; attempt < 25; attempt++) {
    try {
      fs.mkdirSync(file)
      return () => {
        try {
          fs.rmdirSync(file)
        } catch {
          // Already gone.
        }
      }
    } catch {
      try {
        // A pass that was killed leaves its lock behind; after two minutes it
        // is not a writer, it is litter.
        if (Date.now() - fs.statSync(file).mtimeMs > LOCK_STALE_MS) fs.rmdirSync(file)
      } catch {
        // Someone else got there first.
      }
      sleep(200)
    }
  }
  return undefined
}

function sleep(ms: number) {
  Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, ms)
}

// Last, not first: the module's `let` bindings have to be initialized before
// the pass touches them.
main()
