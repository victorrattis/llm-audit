

## Configure llm audit on OpenCode
```json
origins: [{ "spec": ["file:///Users/victorrattis/.config/opencode/llm-audit.ts",
                     {"dir": "/Users/victorrattis/temp/opencode-audit-log"}],
            "source": "/Users/victorrattis/.config/opencode", "scope": "global" }]
```


`~/.config/opencode/opencode.jsonc`
`~/.config/opencode/llm-audit.ts`





## Configure llm audit on Claude Code


```json
{
  "env": {
    "CLAUDE_CODE_ENABLE_TELEMETRY": "1",
    "OTEL_METRICS_EXPORTER": "none",
    "OTEL_LOG_RAW_API_BODIES": "file:/Users/victorrattis/temp/claude-code-audit-log-bodies",
    "CLAUDE_LLM_AUDIT_DIR": "/Users/victorrattis/temp/claude-code-audit-log",
    "CLAUDE_LLM_AUDIT_BODIES": "/Users/victorrattis/temp/claude-code-audit-log-bodies"
  },
  "hooks": {
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "CLAUDE_LLM_AUDIT_DIR=/Users/victorrattis/temp/claude-code-audit-log CLAUDE_LLM_AUDIT_BODIES=/Users/victorrattis/temp/claude-code-audit-log-bodies /Users/victorrattis/.bun/bin/bun /Users/victorrattis/.claude/scripts/llm-audit/report.ts",
            "timeout": 60,
            "async": true
          }
        ]
      }
    ],
    "SubagentStop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "CLAUDE_LLM_AUDIT_DIR=/Users/victorrattis/temp/claude-code-audit-log CLAUDE_LLM_AUDIT_BODIES=/Users/victorrattis/temp/claude-code-audit-log-bodies /Users/victorrattis/.bun/bin/bun /Users/victorrattis/.claude/scripts/llm-audit/report.ts",
            "timeout": 60,
            "async": true
          }
        ]
      }
    ],
    "SessionEnd": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "CLAUDE_LLM_AUDIT_DIR=/Users/victorrattis/temp/claude-code-audit-log CLAUDE_LLM_AUDIT_BODIES=/Users/victorrattis/temp/claude-code-audit-log-bodies /Users/victorrattis/.bun/bin/bun /Users/victorrattis/.claude/scripts/llm-audit/report.ts",
            "timeout": 60
          }
        ]
      }
    ]
  }
}
```
