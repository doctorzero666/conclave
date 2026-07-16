# conclave

> *Like a conclave of cardinals, but for AI models.*

Multi-LLM deliberation in your terminal. One command, multiple models, side-by-side comparison. No API keys, no per-token cost — uses your existing Claude Code / Codex CLI subscriptions.

## Install

```bash
pip install conclave-llm
```

### Prerequisites

conclave doesn't manage API keys. It calls LLMs through CLI tools you already have installed and authenticated:

| Backend | Install |
|---------|---------|
| Claude | `npm install -g @anthropic-ai/claude-code` then `claude login` |
| Codex  | `npm install -g @openai/codex` then `codex login` |

## Quick Start

```bash
# Ask Claude + Codex the same question
conclave "What's the best caching strategy for a Flask API?"

# Only Claude
conclave "Review app.py for SQL injection" --backends claude

# Diff view: see exactly where models agree and disagree
conclave "Should we use Redis or in-memory cache?" --diff

# JSON output for scripts
conclave "Summarize this in 3 bullet points" --format json

# Copy result to clipboard
conclave "Draft a PostgreSQL migration script" --copy
```

## Killer Use Cases

### 1. Code Security Review

```bash
$ conclave "Review auth.py — any security issues?" --backends claude,codex

## Panel: 'Review auth.py — any security issues?'

### claude (claude-sonnet-4) | 15.2s
⚠️ CRITICAL: JWT secret hardcoded on line 12. Use env var.
⚠️ No rate limiting on /login endpoint — brute force risk.

### codex (gpt-5.6-sol) | 8.7s
1. Line 12: hardcoded secret → os.environ.get("JWT_SECRET")
2. Missing password strength validation
3. Session tokens don't expire
```

Two models catch different issues. Don't choose — ask both.

### 2. Architecture Decision

```bash
$ conclave "Monolith → microservices: where to start?" --diff

--- claude (claude-sonnet-4)
+++ codex (gpt-5.6-sol)
-Start with the auth module — it's the most self-contained
+Start with notifications — lowest risk, highest impact
-Use strangler fig: route old→new gradually
+Run both in parallel, compare outputs
```

### 3. Documentation & Code Generation

```bash
$ conclave "Write a pydantic model for a Stripe webhook event" --copy
# → both models' versions on your clipboard
```

## All Options

```
conclave run [OPTIONS] PROMPT

Options:
  -b, --backends TEXT    Comma-separated backends (default: claude,codex)
  -f, --format [markdown|json]  Output format (default: markdown)
  --diff / --no-diff     Unified diff view (2 backends only)
  --no-cache             Skip cache, force fresh calls
  --copy                 Copy result to clipboard
  -t, --timeout INT      Per-backend timeout seconds (default: 180)
  -v, --verbose          Print subprocess command lines
  --help                 Show this message
```

Environment variables: `CONCLAVE_BACKENDS`, `CONCLAVE_FORMAT`, `CONCLAVE_DIFF`, `CONCLAVE_TIMEOUT`.

## Features

- ⚡ **Parallel execution** — all backends run simultaneously
- 📊 **Live progress** — see results as they arrive
- 🔍 **Diff view** — unified-diff between two model outputs
- 💾 **Smart cache** — same prompt within 5 minutes doesn't re-call models
- 📋 **One-copy** — `--copy` puts output on your clipboard
- 🛡️ **Safe subprocess** — no `shell=True`, no command injection
- 🎨 **Rich terminal UI** — color-coded status
- 🐧 **macOS + Linux** — v0.1 (Windows in v0.2)

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | All backends succeeded |
| 2 | Partial success (some failed) |
| 1 | All backends failed |

## Development

```bash
git clone https://github.com/doctorzero666/conclave
cd conclave
pip install -e ".[dev]"
pytest
```

## License

MIT
