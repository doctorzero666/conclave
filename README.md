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
```

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
