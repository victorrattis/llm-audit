# llm-audit

A wire log for the traffic between a coding agent and its model provider. One
file per model request, holding what went out and what came back, verbatim: the
system prompt, the tool schemas, every message, the cache markers, the sampling
settings, and the reply with its token bill.

It records only what was observed. Every second-order reading — how the prompt
splits across system, tools and messages, how much of this turn is still the
previous one, what a tool's output costs on every turn after the one it arrived
in, what the whole thing cost — is a reading of the body, and the body is here
whole. [`harness-audit-viewer`](https://github.com/victorrattis/harness-audit-viewer)
derives them from it, on the turn you open, instead of on every request the
agent makes.

Two harnesses, one format, so the same `jq` reads both sides:

| | how it captures |
|---|---|
| **opencode** | a plugin that wraps the process's `fetch` — the bytes recorded are the bytes on the wire, whatever the provider |
| **Claude Code** | closed, and it wraps no `fetch`. There is no capture: `report.ts` is an incremental reader of what Claude Code already writes to disk — the transcript, for the provider's own token count, and the raw API bodies telemetry writes when asked |

A **turn** is one HTTP request to the model. A single thing you typed usually
spans several, because each tool call round-trips again. Output is one folder
per conversation:

```
<dir>/<YYYYMMDD_HHMMSS>-<session>/turn_001.json
                                 turn_002.json
                                 turns.json      # opencode: the index, one turn per line
```

The turn files are the same shape on both sides. The folder around them is not:
the Claude Code reader keeps a `turns.jsonl` index instead, plus a `summary.json`
for the conversation and a `state.json` telling the next pass where it stopped.

The bodies carry the entire conversation — system prompt, full history, every
tool result. **Treat the output directory as sensitive.** Credentials in headers
and query params are redacted; bodies are not.

## Install on opencode

One file, importing nothing at runtime — its only import is `import type`, which
is erased before it runs.

```bash
ln -s "$PWD/opencode/llm-audit.ts" ~/.config/opencode/llm-audit.ts
```

Then arm it in `~/.config/opencode/opencode.jsonc`, which also says where to
write. **Write the path out in full**: nothing here expands `~`, so a tilde in
`dir` becomes a directory named `~`.

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "plugin": [
    ["./llm-audit.ts", { "dir": "/Users/you/temp/opencode-audit-log" }]
  ]
}
```

For a single project, the same entry in `.opencode/opencode.jsonc` with the file
beside it. Named in the config like this, the `dir` is what arms it. Dropped into
a plugin folder instead, it loads but stays off until `OPENCODE_LLM_AUDIT=1` or
`OPENCODE_LLM_AUDIT_DIR=<dir>` says otherwise — armed by the flag alone, it
writes to `~/.local/share/opencode/log/llm-audit`.

`@opencode-ai/plugin` next to it, as `~/.config/opencode/package.json` declares
it, only buys you type-checking while editing the file.

| | |
|---|---|
| `OPENCODE_LLM_AUDIT` | `1` to arm, `0` to disable. The config's `enabled` wins over it |
| `OPENCODE_LLM_AUDIT_DIR` | where to write, when the config does not say |

## Install on Claude Code

Needs [bun](https://bun.sh) on the machine — `report.ts` is TypeScript and runs
as a hook.

```bash
mkdir -p ~/.claude/scripts/llm-audit
cp claudecode/{report.ts,core.ts,prices.json} ~/.claude/scripts/llm-audit/
```

`~/.claude/settings.json` does the two halves: `env` turns on the raw-body
telemetry Claude Code writes (telemetry variables apply at startup, from any
settings file), and the hooks call the reader.

```json
{
  "env": {
    "CLAUDE_CODE_ENABLE_TELEMETRY": "1",
    "OTEL_METRICS_EXPORTER": "none",
    "OTEL_LOG_RAW_API_BODIES": "file:/Users/you/temp/claude-code-audit-log-bodies",
    "CLAUDE_LLM_AUDIT_DIR": "/Users/you/temp/claude-code-audit-log",
    "CLAUDE_LLM_AUDIT_BODIES": "/Users/you/temp/claude-code-audit-log-bodies"
  },
  "hooks": {
    "Stop": [
      { "hooks": [{ "type": "command", "async": true, "timeout": 60,
          "command": "CLAUDE_LLM_AUDIT_DIR=/Users/you/temp/claude-code-audit-log CLAUDE_LLM_AUDIT_BODIES=/Users/you/temp/claude-code-audit-log-bodies ~/.bun/bin/bun ~/.claude/scripts/llm-audit/report.ts" }] }
    ],
    "SubagentStop": [
      { "hooks": [{ "type": "command", "async": true, "timeout": 60,
          "command": "CLAUDE_LLM_AUDIT_DIR=/Users/you/temp/claude-code-audit-log CLAUDE_LLM_AUDIT_BODIES=/Users/you/temp/claude-code-audit-log-bodies ~/.bun/bin/bun ~/.claude/scripts/llm-audit/report.ts" }] }
    ],
    "SessionEnd": [
      { "hooks": [{ "type": "command", "timeout": 60,
          "command": "CLAUDE_LLM_AUDIT_DIR=/Users/you/temp/claude-code-audit-log CLAUDE_LLM_AUDIT_BODIES=/Users/you/temp/claude-code-audit-log-bodies ~/.bun/bin/bun ~/.claude/scripts/llm-audit/report.ts" }] }
    ]
  }
}
```

The two variables repeated as a prefix on each command are not redundancy:
Claude Code only exports to a subprocess the telemetry variables it classifies as
"safe", and `OTEL_LOG_RAW_API_BODIES` is not one of them — a hook does not see
it. The `env` block is read before any shell, so its paths have to be written out
in full; the hook commands run through one, where `~` expands. `Stop` runs `async` so it adds no latency to the turn;
`SessionEnd` runs synchronously to guarantee the final `summary.json`.

The raw bodies land in `<dir>-bodies`, **beside** the audit folders and never
inside them: each is ~1 MB, and the viewer scans recursively for `.json`. They
are also the part that grows without limit:

```bash
bun ~/.claude/scripts/llm-audit/report.ts --prune 7
```

Set `CLAUDE_LLM_AUDIT=0` to turn the reader off.
[`claudecode/README.md`](claudecode/README.md) has the rest — how the two sources
are matched per turn, what the viewer requires, and where this side differs from
the opencode plugin.
