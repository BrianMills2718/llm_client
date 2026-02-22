# LLM Model Guide — February 2026

Practical reference for choosing models. Data from [Artificial Analysis](https://artificialanalysis.ai/) leaderboard + provider pricing pages.

**Price** = blended cost per 1M tokens, assuming 3:1 input:output ratio: `(3×input + 1×output) / 4`. For exact input/output splits, see [Provider Pricing](#provider-pricing) section.

**Intelligence** = Artificial Analysis quality index (0-100 scale). **Speed** = output tokens/sec. **Latency** = time to first token (seconds).

Models with thinking/effort levels show their best config. Duplicate entries removed.

---

## Best Model by Use Case

| Use Case | Model | Why |
|---|---|---|
| **Best overall quality** | Claude Opus 4.6 Adaptive | #1 intelligence (53), best instruction following |
| **Best value frontier** | GLM-5 | Intelligence 50 (= Opus 4.5) at $1.55 blended. Open-weight. |
| **Best coding (benchmark)** | GPT-5.2 Codex (xhigh) | #1 LiveCodeBench. Intelligence 49. |
| **Best coding (architecture/review)** | Claude Opus 4.5/4.6 | Superior reasoning about code, not just generating it |
| **Best mid-tier** | Gemini 3 Flash | Intelligence 46, $1.13, 207 t/s, 1M context |
| **Cheapest good model** | MiMo-V2-Flash | Intelligence 41, $0.15 (!), 166 t/s. Open-weight. |
| **Best value per intelligence** | DeepSeek V3.2 | Intelligence 42, $0.32. Open-weight. |
| **Best reliability for tool loops** | DeepSeek V3.2 (OpenRouter) + fallback | More stable in long agentic tool-call loops than Gemini OpenAI-compat paths in our runs |
| **Best long context** | Grok 4.1 Fast | 2M context, $0.28, 179 t/s |
| **Cheapest reasoning** | DeepSeek V3.2 (thinking) | $0.32 blended with CoT |
| **Best reasoning quality** | o3 | Intelligence 38 (reasoning-focused benchmarks much higher) |
| **Fastest inference** | gpt-oss-120B | 312 t/s, $0.26. Open-weight. |
| **Best open-weight** | GLM-5 | Intelligence 50, MIT license, 744B/44B active |
| **Cheapest OpenAI** | GPT-5 nano | $0.14, intelligence 27. Good for simple tasks. |
| **Best free tier** | Gemini 2.5 Flash | Free on Google dev platform |

---

## Reliability Notes (Production)

- Gemini models remain strong for many text/analysis tasks, but in long
  tool-calling loops they can intermittently return empty content (`200 OK`,
  no usable text/tool call payload) through compatibility layers.
- This is not always resolved by naive retries because some failures are
  deterministic for the current conversation/tool state.
- Recommended default for tool-heavy automation:
  1. Primary: `openrouter/deepseek/deepseek-chat`
  2. Fallback 1: `openrouter/openai/gpt-5-mini`
  3. Fallback 2: `gemini/gemini-2.5-flash` (optional, for cost/context)
- If you keep Gemini as primary, require typed failure classification and
  state-mutating retry/recovery before repeating the same request payload.

---

## Full Leaderboard

### Tier 1: Frontier (Intelligence 45+)

| Rank | Model | Provider | Intel | Price | Speed | Latency | Context | Open | Params |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Claude Opus 4.6 (Adaptive) | Anthropic | 53 | $10.00 | 64 | 1.69s | 200K | No | ? |
| 2 | GPT-5.2 (xhigh) | OpenAI | 51 | $4.81 | 98 | 46.57s | 400K | No | est 2-5T MoE |
| 3 | Claude Opus 4.5 | Anthropic | 50 | $10.00 | 84 | 1.53s | 200K | No | ? |
| 4 | **GLM-5** | Z AI (Zhipu) | 50 | $1.55 | 76 | 1.54s | 200K | **MIT** | 744B / 44B active |
| 5 | GPT-5.2 Codex (xhigh) | OpenAI | 49 | $4.81 | 91 | 25.87s | 400K | No | ? |
| 6 | Gemini 3 Pro Preview (high) | Google | 48 | $4.50 | 133 | 30.65s | 1M | No | ? |
| 7 | **Kimi K2.5** | Moonshot | 47 | $1.20 | 45 | 1.46s | 256K | **MIT** | 1T / 32B active |
| 8 | Gemini 3 Flash | Google | 46 | $1.13 | 207 | 12.10s | 1M | No | ? |
| 9 | Claude Opus 4.6 | Anthropic | 46 | $10.00 | — | — | 200K | No | ? |

### Tier 2: Strong (Intelligence 39-44)

| Rank | Model | Provider | Intel | Price | Speed | Latency | Context | Open | Params |
|---|---|---|---|---|---|---|---|---|---|
| 10 | Claude 4.5 Sonnet | Anthropic | 43 | $6.00 | 81 | 1.33s | 1M | No | ? |
| 11 | GPT-5.1 Codex (high) | OpenAI | 42 | $3.44 | 166 | 15.87s | 400K | No | ? |
| 12 | **GLM-4.7** | Z AI (Zhipu) | 42 | $0.94 | 128 | 0.73s | 200K | **Open** | 355B MoE |
| 13 | **MiniMax-M2.5** | MiniMax | 42 | $0.53 | 71 | 1.67s | 205K | **MIT*** | 230B / 10B active |
| 14 | **DeepSeek V3.2** | DeepSeek | 42 | $0.32 | 36 | 1.35s | 128K | **MIT** | 685B / 37B active |
| 15 | Grok 4 | xAI | 41 | $6.00 | 39 | 15.96s | 256K | No | ? |
| 16 | **MiMo-V2-Flash** | Xiaomi | 41 | $0.15 | 166 | 1.39s | 256K | **MIT** | 309B / 15B active |
| 17 | GPT-5 mini (high) | OpenAI | 41 | $0.69 | 127 | 60.75s | 400K | No | ? |
| 18 | Qwen3 Max Thinking | Alibaba | 40 | $2.40 | 38 | 1.75s | 256K | API only | ? |
| 19 | **MiniMax-M2.1** | MiniMax | 40 | $0.53 | 61 | 1.73s | 205K | **Open** | ? |
| 20 | Grok 4.1 Fast | xAI | 39 | $0.28 | 179 | 4.85s | **2M** | No | ? |
| 21 | GPT-5.1 Codex mini (high) | OpenAI | 39 | $0.69 | 114 | 12.88s | 400K | No | ? |

\* MiniMax-M2.5 uses modified MIT (must display branding in commercial products)

### Tier 3: Good (Intelligence 33-38)

| Rank | Model | Provider | Intel | Price | Speed | Latency | Context | Open | Params |
|---|---|---|---|---|---|---|---|---|---|
| 22 | o3 | OpenAI | 38 | $3.50 | 110 | 20.34s | 200K | No | ? |
| 23 | Claude 4.5 Haiku | Anthropic | 37 | $2.00 | 127 | 0.46s | 200K | No | ? |
| 24 | KAT-Coder-Pro V1 | KwaiKAT | 36 | $0.53 | 58 | 1.80s | 256K | Open | ? |
| 25 | Nova 2.0 Pro Preview | Amazon | 36 | $3.44 | 129 | 21.86s | 256K | No | ? |
| 26 | Gemini 2.5 Pro | Google | 34 | $3.44 | 152 | 38.10s | 1M | No | ? |
| 27 | **DeepSeek V3.2 Speciale** | DeepSeek | 34 | $0.42 | — | — | 128K | **MIT** | 685B / 37B active |
| 28 | **gpt-oss-120B** | OpenAI | 33 | $0.26 | **312** | 0.47s | 131K | **Open** | 117B / 5.1B active |
| 29 | Grok 3 mini Reasoning | xAI | 32 | $0.35 | 195 | 0.72s | 1M | No | ? |
| 30 | **K-EXAONE** | LG AI Research | 32 | $0.00 | 136 | 0.30s | 256K | **Open** | ? |

### Tier 4: Budget / Specialized (Intelligence 26-32)

| Rank | Model | Provider | Intel | Price | Speed | Latency | Context | Open | Params |
|---|---|---|---|---|---|---|---|---|---|
| 31 | Qwen3 Max | Alibaba | 31 | $2.40 | 27 | 2.37s | 262K | API only | ? |
| 32 | **GLM-4.7-Flash** | Z AI (Zhipu) | 30 | $0.15 | 43 | 0.72s | 200K | **Open** | ? |
| 33 | Nova 2.0 Lite | Amazon | 30 | $0.85 | 237 | 15.45s | 1M | No | ? |
| 34 | **Qwen3 235B A22B** | Alibaba | 29 | $2.63 | 45 | 1.41s | 256K | **Open** | 235B / 22B active |
| 35 | ERNIE 5.0 Thinking | Baidu | 29 | $0.00 | — | — | 128K | No | ? |
| 36 | Grok Code Fast 1 | xAI | 29 | $0.53 | **306** | 7.00s | 256K | No | ? |
| 37 | **Qwen3-Coder-Next** | Alibaba | 28 | $0.53 | 124 | 0.80s | 256K | **Apache 2.0** | 80B / 3B active |
| 38 | **Apriel-v1.6-15B** | ServiceNow | 28 | $0.00 | 146 | 0.21s | 128K | **Open** | 15B dense |
| 39 | Magistral Medium 1.2 | Mistral | 27 | $2.75 | 38 | 0.54s | 128K | Open | ? |
| 40 | **DeepSeek R1 0528** | DeepSeek | 27 | $2.36 | — | — | 128K | **MIT** | 685B / 37B active |
| 41 | GPT-5 nano (high) | OpenAI | 27 | $0.14 | 141 | 110.10s | 400K | No | ? |
| 42 | **Qwen3 Next 80B A3B** | Alibaba | 26 | $1.88 | 172 | 1.14s | 262K | **Open** | 80B / 3B active |

---

## Category Rankings

### Cheapest (Price per 1M tokens, blended)

| Model | Price | Intelligence | Open |
|---|---|---|---|
| K-EXAONE | $0.00 | 32 | Yes |
| Apriel-v1.6-15B | $0.00 | 28 | Yes |
| GPT-5 nano | $0.14 | 27 | No |
| MiMo-V2-Flash | $0.15 | 41 | Yes |
| GLM-4.7-Flash | $0.15 | 30 | Yes |
| gpt-oss-120B | $0.26 | 33 | Yes |
| Grok 4.1 Fast | $0.28 | 39 | No |
| DeepSeek V3.2 | $0.32 | 42 | Yes |

**Winner**: MiMo-V2-Flash — intelligence 41 at $0.15 is absurd value.

### Fastest (Output tokens/sec)

| Model | Speed | Intelligence | Price |
|---|---|---|---|
| gpt-oss-120B | 312 t/s | 33 | $0.26 |
| Grok Code Fast 1 | 306 t/s | 29 | $0.53 |
| Nova 2.0 Lite | 237 t/s | 30 | $0.85 |
| Gemini 3 Flash | 207 t/s | 46 | $1.13 |
| Grok 3 mini Reasoning | 195 t/s | 32 | $0.35 |
| Grok 4.1 Fast | 179 t/s | 39 | $0.28 |
| Qwen3 Next 80B A3B | 172 t/s | 26 | $1.88 |
| MiMo-V2-Flash | 166 t/s | 41 | $0.15 |
| GPT-5.1 Codex (high) | 166 t/s | 42 | $3.44 |

**Winner**: gpt-oss-120B for raw speed. Gemini 3 Flash for speed + intelligence combo.

### Lowest Latency (Time to first token)

| Model | TTFT | Intelligence | Price |
|---|---|---|---|
| Apriel-v1.6-15B | 0.21s | 28 | $0.00 |
| K-EXAONE | 0.30s | 32 | $0.00 |
| Claude 4.5 Haiku | 0.46s | 37 | $2.00 |
| gpt-oss-120B | 0.47s | 33 | $0.26 |
| GLM-4.7 | 0.73s | 42 | $0.94 |
| Grok 3 mini Reasoning | 0.72s | 32 | $0.35 |
| Qwen3-Coder-Next | 0.80s | 28 | $0.53 |

**Winner**: Claude 4.5 Haiku for latency + quality combo. Apriel for absolute lowest.

### Longest Context

| Model | Context | Intelligence | Price |
|---|---|---|---|
| Grok 4.1 Fast | 2M | 39 | $0.28 |
| Gemini 3 Pro Preview | 1M | 48 | $4.50 |
| Gemini 3 Flash | 1M | 46 | $1.13 |
| Claude 4.5 Sonnet | 1M | 43 | $6.00 |
| Gemini 2.5 Pro | 1M | 34 | $3.44 |
| Grok 3 mini Reasoning | 1M | 32 | $0.35 |
| Nova 2.0 Lite/Omni | 1M | 28-30 | $0.85 |

**Winner**: Grok 4.1 Fast for max context at lowest price. Gemini 3 Flash for context + quality.

### Best Open-Weight Models

| Model | Intelligence | Price | License | Params |
|---|---|---|---|---|
| GLM-5 | 50 | $1.55 | MIT | 744B / 44B active |
| Kimi K2.5 | 47 | $1.20 | MIT | 1T / 32B active |
| GLM-4.7 | 42 | $0.94 | Open | 355B MoE |
| MiniMax-M2.5 | 42 | $0.53 | MIT* | 230B / 10B active |
| DeepSeek V3.2 | 42 | $0.32 | MIT | 685B / 37B active |
| MiMo-V2-Flash | 41 | $0.15 | MIT | 309B / 15B active |
| gpt-oss-120B | 33 | $0.26 | Open | 117B / 5.1B active |
| Qwen3-Coder-Next | 28 | $0.53 | Apache 2.0 | 80B / 3B active |

**Takeaway**: Open-weight models now match Claude Opus 4.5 quality (GLM-5 at intelligence 50). The gap between open and proprietary has nearly closed.

---

## Provider Pricing (Input / Output per 1M tokens)

### Anthropic

| Model | Input | Output | Context |
|---|---|---|---|
| Claude 4.5 Haiku | $1.00 | $5.00 | 200K |
| Claude 4.5 Sonnet | $3.00 | $15.00 | 1M |
| Claude Opus 4.5 | $5.00 | $25.00 | 200K |
| Claude Opus 4.6 | $15.00 | $75.00 | 200K |

Cache reads: 90% off. Batch: 50% off.

### OpenAI

| Model | Input | Output | Context |
|---|---|---|---|
| GPT-OSS-20B | $0.03 | $0.14 | ? |
| GPT-5 Nano | $0.05 | $0.40 | 400K |
| GPT-4.1 Nano | $0.10 | $0.40 | ? |
| GPT-5-Mini | $0.25 | $2.00 | 400K |
| GPT-5.1-Codex-Mini | $0.25 | $2.00 | 400K |
| o4-mini | $1.10 | $4.40 | 200K |
| o3-mini | $1.10 | $4.40 | 200K |
| GPT-5 | $1.25 | $10.00 | 1M |
| GPT-5-Codex | $1.25 | $10.00 | ? |
| GPT-5.1 | $1.25 | $10.00 | ? |
| GPT-5.1-Codex | $1.25 | $10.00 | 400K |
| GPT-5.2 | $1.75 | $14.00 | 400K |
| GPT-5.2-Codex | $1.75 | $14.00 | 400K |
| o3 | $2.00 | $8.00 | 200K |
| gpt-oss-120B | $0.10 | $0.56 | 131K |

Batch: 50% off.

### Google

| Model | Input | Output | Context |
|---|---|---|---|
| Gemini 2.0 Flash-Lite | $0.075 | $0.30 | 1M |
| Gemini 2.5 Flash-Lite | $0.10 | $0.40 | ? |
| Gemini 2.0 Flash | $0.10 | $0.40 | 1M |
| Gemini 2.5 Flash | $0.30 | $2.50 | 1M |
| Gemini 3 Flash | ~$0.50 | ~$3.00 | 1M |
| Gemini 2.5 Pro | $1.25 | $10.00 | 1M |
| Gemini 3 Pro Preview | $2.00 | $12.00 | 1M |

>200K input doubles price on Pro models. Cache reads: 90% off. Batch: 50% off. Free tier on dev platform.

### DeepSeek

| Model | Input | Output | Context |
|---|---|---|---|
| DeepSeek V3.2 (chat) | $0.28 | $0.42 | 128K |
| DeepSeek V3.2 (thinking) | $0.28 | $0.42 | 128K |

Cache hits: $0.028 (90% off). Off-peak 50-75% discount (16:30-00:30 GMT). Open-weight, self-hostable.

### xAI

| Model | Input | Output | Context |
|---|---|---|---|
| Grok 4.1 Fast | $0.20 | $0.50 | 2M |
| Grok 4 | ~$3.00 | ~$15.00 | 256K |

---

## Decision Flowchart

```
Need cheapest possible?
  ├─ Yes, open-weight → MiMo-V2-Flash ($0.15, intel 41, MIT)
  ├─ Yes, API → DeepSeek V3.2 ($0.32, intel 42)
  └─ No
     Need frontier quality (intel 45+)?
       ├─ Budget → GLM-5 ($1.55, intel 50, open-weight)
       │           or Kimi K2.5 ($1.20, intel 47, open-weight)
       ├─ Quality → Claude Opus 4.6 Adaptive ($10, intel 53)
       └─ Balanced → Gemini 3 Flash ($1.13, intel 46, 207 t/s)
     Need long context (>256K)?
       ├─ Budget → Grok 4.1 Fast ($0.28, 2M context!)
       ├─ Quality → Gemini 3 Pro Preview ($4.50, 1M, intel 48)
       └─ Anthropic → Claude 4.5 Sonnet ($6.00, 1M, intel 43)
     Need maximum speed?
       ├─ Budget → gpt-oss-120B ($0.26, 312 t/s)
       └─ Quality → Gemini 3 Flash ($1.13, 207 t/s, intel 46)
     Need reasoning/thinking?
       ├─ Budget → DeepSeek V3.2 thinking ($0.32)
       ├─ Quality → o3 ($3.50)
       └─ Open-weight → Qwen3 Max Thinking (API $2.40)
     Need coding agent?
       ├─ Budget → GPT-5.1 Codex mini ($0.69, intel 39)
       ├─ Quality → GPT-5.2 Codex ($4.81, intel 49)
       └─ Open → Qwen3-Coder-Next ($0.53, 80B/3B, Apache 2.0)
     General purpose?
       ├─ Dirt cheap → MiMo-V2-Flash ($0.15) or DeepSeek V3.2 ($0.32)
       ├─ Budget → Gemini 3 Flash ($1.13)
       ├─ Mid-tier → GLM-5 ($1.55) or Kimi K2.5 ($1.20)
       └─ Premium → Claude Opus 4.6 ($10.00)
```

---

## Models to Avoid (Outclassed)

| Model | Why | Use Instead |
|---|---|---|
| GPT-4o ($2.50/$10) | GPT-5 is $1.25/$10 and smarter | GPT-5 |
| GPT-4o-mini ($0.15/$0.60) | MiMo-V2-Flash is cheaper ($0.15 blended) AND smarter (41 vs ~30). DeepSeek V3.2 is $0.32 at intel 42 | MiMo-V2-Flash or DeepSeek V3.2 |
| Gemini 2.5 Pro ($3.44 blended) | Gemini 3 Pro Preview is better (intel 48 vs 34) at similar price. Gemini 3 Flash is cheaper and better (intel 46 at $1.13) | Gemini 3 Flash or 3 Pro |
| o1 / o1-pro | Superseded by o3 / o4-mini | o3 or o4-mini |
| Claude Haiku 3.5 | Haiku 4.5 is same price, better | Claude 4.5 Haiku |
| Grok 4 ($6.00) | Worse than Sonnet 4.5 at same price, worse than GLM-5 at 4x the price | GLM-5 or Sonnet 4.5 |
| Nova 2.0 Pro ($3.44) | Intel 36, beaten by Gemini 3 Flash (intel 46 at $1.13) | Gemini 3 Flash |
| Magistral Medium ($2.75) | Intel 27, overpriced for quality. DeepSeek V3.2 is 8x cheaper at intel 42 | DeepSeek V3.2 |
| DeepSeek R1 0528 ($2.36) | Old reasoning model. V3.2 thinking mode is same quality, 7x cheaper | DeepSeek V3.2 (thinking) |

---

## Agentic Coding Models (CLI/SDK only)

These models are **not available via standard API** — they only work through Codex CLI (`codex -m`), Codex SDK, or Claude Code. Use `call_llm("codex/gpt-5.3-codex", ...)` via llm_client.

| Model | CLI Flag | Capability | Speed | Access |
|---|---|---|---|---|
| **gpt-5.3-codex** | `codex -m gpt-5.3-codex` | Best agentic coding | Moderate | CLI/SDK, app, Cloud. **No API.** |
| **gpt-5.3-codex-spark** | `codex -m gpt-5.3-codex-spark` | Text-only, near-instant | Fastest | CLI/SDK, app. ChatGPT Pro only. No Cloud, no API. |
| **gpt-5.2-codex** | `codex -m gpt-5.2-codex` | Previous gen, still strong | Moderate | CLI/SDK, app, Cloud. **Has API access.** |
| **gpt-5.1-codex** | `codex -m gpt-5.1-codex` | Older | Fast | CLI/SDK, app, Cloud. Has API. |
| **Claude Code (Opus 4.6)** | `claude` | Best instruction following | Slow (64 t/s) | CLI only. |
| **Claude Code (Sonnet 4.5)** | `claude --model sonnet` | Good balance | Moderate (81 t/s) | CLI only. |

Key distinction: gpt-5.3-codex is the most capable but has **no API access** — it can only be used through Codex CLI/SDK. If you need API access (e.g., litellm routing), use gpt-5.2-codex instead.

---

## Key Observations (Feb 2026)

1. **Open-weight caught up.** GLM-5 (intel 50, MIT) matches Opus 4.5. Kimi K2.5 (intel 47, MIT) beats Gemini 3 Flash. The moat is thin.

2. **Chinese models dominate value.** DeepSeek V3.2, MiMo-V2-Flash, GLM-5, Kimi K2.5, MiniMax-M2.5, Qwen3 — all open-weight, all dramatically cheaper than Western equivalents.

3. **MoE is the standard.** Nearly every competitive model uses Mixture-of-Experts. Active params range from 3B (Qwen3-Coder-Next) to 44B (GLM-5) regardless of total model size.

4. **GPT-4o-mini is dead.** Was the default "cheap model" — now outclassed on both price and quality by MiMo-V2-Flash, DeepSeek V3.2, gpt-oss-120B, and others.

5. **Context windows are real now.** Grok 4.1 Fast does 2M at $0.28 blended. Gemini 3 Flash does 1M at $1.13. Long-context is no longer premium.

6. **Speed king is gpt-oss-120B** at 312 t/s from API, but Cerebras does 2,500 t/s on Llama 4 Maverick if you use their inference platform.

7. **Latency varies wildly.** Claude models have sub-2s TTFT. Some OpenAI reasoning models take 20-110s to first token (thinking time). Matters a lot for interactive use.

---

## Notes

- **Cache discounts**: DeepSeek (90%), Anthropic (90%), Google (90%). Always enable prompt caching for repeated prefixes.
- **Batch discounts**: Google (50%), Anthropic (50%), OpenAI (50%). Use for non-time-sensitive bulk work.
- **Self-hosting**: DeepSeek V3.2, GLM-5, Kimi K2.5, MiMo-V2-Flash, gpt-oss-120B all open-weight MIT. Run on Cerebras/Groq for speed, or own GPUs for cost at scale.
- **litellm compatibility**: All API models work through litellm. Agent SDKs (Claude Code, Codex CLI) need llm_client agent routing.
- **Thinking token costs**: Reasoning models (o3, DeepSeek thinking, Qwen3 thinking) generate hidden thinking tokens billed at output rates. Actual cost per query can be 5-20x the listed price.

*Last updated: 2026-02-15. Data from Artificial Analysis leaderboard + provider pricing pages.*

## Sources

- [Artificial Analysis Leaderboard](https://artificialanalysis.ai/)
- [OpenAI Pricing](https://platform.openai.com/docs/pricing)
- [Anthropic Pricing](https://platform.claude.com/docs/en/about-claude/pricing)
- [Google Gemini Pricing](https://ai.google.dev/gemini-api/docs/pricing)
- [DeepSeek Pricing](https://api-docs.deepseek.com/quick_start/pricing)
- [xAI Grok 4.1](https://x.ai/news/grok-4-1)
- [GLM-5 on HuggingFace](https://huggingface.co/zai-org/GLM-5)
- [Kimi K2.5 on HuggingFace](https://huggingface.co/moonshotai/Kimi-K2.5)
- [MiMo-V2-Flash on HuggingFace](https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash)
- [MiniMax M2.5](https://www.minimax.io/news/minimax-m25)
- [Qwen3-Coder-Next](https://huggingface.co/Qwen/Qwen3-Coder-Next)
