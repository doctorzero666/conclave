# conclave v3

> *Multi-LLM deliberation protocol engine — like a conclave of AI cardinals.*

V3 is a complete rewrite. It's not just "ask two models and compare" — it's a **multi-round deliberation protocol** where models critique each other, revise their answers, and a judge synthesizes a consensus report.

## What's new in v3

- 🔄 **Multi-round deliberation** — Initial → Critique → Revise → Synthesize
- 🔌 **Plugin provider system** — CLI (Claude Code/Codex) + API (OpenAI/Anthropic/DeepSeek)
- 🧠 **Semantic synthesizer** — LLM-powered judge generates consensus/divergence/insight report
- 📡 **Streaming** — Real-time output for each provider
- ⚙️ **Config system** — `~/.conclave/config.yaml`
- 🧪 **Test suite** — 22 tests, CI via GitHub Actions

## Install

```bash
pip install conclave-llm
```

### Prerequisites

conclave doesn't manage API keys. It connects through:

| Provider Type | What you need |
|:--|:--|
| CLI | `claude` and/or `codex` CLI installed + logged in |
| OpenAI | `OPENAI_API_KEY` env var |
| Anthropic | `ANTHROPIC_API_KEY` env var |
| DeepSeek | `DEEPSEEK_API_KEY` env var |

## Quick Start

```bash
# Initialize config
conclave config init

# Ask 2 models the same question
conclave "Should we use Redis or Postgres for this caching layer?"

# Multi-round deliberation with synthesis
conclave "Design a rate-limiting strategy for our API" --rounds 3

# Use specific providers
conclave "Review this architecture" --providers claude,deepseek

# JSON output
conclave "Compare Kubernetes vs Nomad" --format json --rounds 2

# Prompt from a file (long / multiline specs — no shell pipes needed)
conclave run --prompt-file ./spec.md --providers claude,codex --rounds 1

# Persist result to disk (JSON is un-truncated; markdown is ANSI-free)
conclave run --prompt-file ./spec.md \
    --providers claude,codex \
    --format json --output ./deliberation.json
```

### `--prompt-file` and `--output`

- `--prompt-file PATH` reads the prompt from a UTF-8 file. Mutually exclusive
  with the positional `PROMPT`. Empty/missing/non-UTF-8/directory inputs raise
  a clear error.
- `--output PATH` writes the full result to disk.
  - `--format json` writes the complete `DeliberationResult` as JSON (response
    text is **not** truncated, unlike the terminal preview).
  - `--format markdown` writes plain, ANSI-free markdown.
  - If writing fails (permissions, missing parent dir, …) conclave prints a
    warning but keeps the deliberation exit code.
- Passing the same path to `--prompt-file` and `--output` is rejected so the
  input can never be clobbered.

### Migrating from `hermes_fusion.py`

The old fusion runner used `echo "$SPEC" | python3 hermes_fusion.py …`, which
Hermes Tirith blocks as `pipe_to_interpreter`. Replace it with a `--prompt-file`
call — no stdin pipe, no interactive TUI:

```bash
# ❌ blocked by Hermes and hangs on multiline prompts
echo "$SPEC" | python3 ~/.hermes/scripts/hermes_fusion.py --providers claude,codex

# ✅ conclave equivalent
conclave run --prompt-file ./spec.md \
    --providers claude,codex --rounds 1 \
    --format json --output ./out.json
```

The default Claude preset now runs `claude -p "{prompt}"`, so long or multiline
prompts no longer drop into the interactive TUI and hang.

## Deliberation Protocol

```
Round 1 (INITIAL):  All models answer independently
Round 2 (CRITIQUE):  Each model critiques others' answers
Round 3 (REVISE):    Models revise based on feedback
       ↓
    SYNTHESIS:       Judge generates consensus report:
                     ✅ Consensus points
                     🔀 Divergences
                     💡 Unique insights
```

## Provider System

conclave v3 supports any backend through a provider plugin system:

```yaml
# ~/.conclave/config.yaml
providers:
  claude:
    provider_type: cli
    model: claude-sonnet-4
    executable: claude
  openai:
    provider_type: openai
    model: gpt-5.6-sol
    api_key: ${OPENAI_API_KEY}
```

## Architecture

```
Layer 4: CLI (click + rich)
Layer 3: Protocol Engine (DeliberationEngine)
Layer 2: Scheduler (parallel ThreadPoolExecutor)
Layer 1: Providers (CLI / OpenAI / Anthropic / DeepSeek)
Layer 0: Config + Cache + Formatters
```

## Development

```bash
git clone https://github.com/doctorzero666/conclave
cd conclave
pip install -e ".[dev]"
pytest       # 22 tests
ruff check   # lint
```

## License

MIT — Zhichao Jiang
