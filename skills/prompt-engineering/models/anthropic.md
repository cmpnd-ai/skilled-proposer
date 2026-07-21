# Anthropic (Claude) — Model-Specific Prompting Guide

Covers Claude Sonnet 5, Opus 4.8, Fable 5, Mythos 5, and earlier Claude 4 models.

<!-- Sources: anthropic-claude-sonnet-5-prompting.md, anthropic-claude-opus-4.8-prompting.md, anthropic-claude-fable-5-prompting.md, anthropic-prompting-best-practices.md, anthropic-extended-thinking.md, anthropic-migrating-to-claude-4.md, 2025-11-10-claude-best-practices-prompt-engineering-2026.md, 2023-09-23-anthropic-long-context-prompting.md -->

## Models

| Model | Context | Adaptive Thinking | Key use |
|---|---|---|---|
| Claude Fable 5 / Mythos 5 | 1M (default) | Always on (can't disable) | Long-horizon autonomy, subagent delegation |
| Claude Opus 4.8 | 1M (default) | OFF by default (must enable) | Coding, agentic — start at `xhigh` effort |
| Claude Sonnet 5 | — | ON by default (can disable) | General-purpose, migration from Sonnet 4.6 |
| Claude Opus 4.5 / earlier Claude 4 | — | Manual extended thinking | Budget_tokens supported |

## Reasoning / Thinking Behavior

### Effort parameter (primary tuning lever)
`effort`: `low` / `medium` / `high` / `xhigh` / `max` — controls intelligence vs. token spend.

- **Opus 4.8**: Start at **`xhigh`** for coding/agentic; minimum **`high`** for intelligence-sensitive tasks. `max` can show diminishing returns/overthinking. "Effort is likely to be more important for this model than for any prior Opus."
- **Sonnet 5**: Defaults to **`high`** (same as Sonnet 4.6). Raise to `xhigh` for hardest coding/agentic. Cross-model mapping: **Sonnet 5 medium ≈ Sonnet 4.6 high; Sonnet 5 high ≈ Sonnet 4.6 max**. Match by observed thinking length when benchmarking, not effort name.
- **Fable 5**: Use `effort` to control thinking depth; higher for complex long-horizon, lower for simple lookups.

### Adaptive thinking (model decides when/how much to think)
- **Opus 4.8**: thinking is **OFF by default**; must explicitly set `thinking: {type: "adaptive"}` to enable
- **Sonnet 5**: adaptive thinking is **ON by default**. Disable with `thinking: {type: "disabled"}`
- **Fable 5 / Mythos 5**: adaptive thinking is **always on**; both `thinking: {type: "disabled"}` and manual extended thinking return a **400 error**

### Manual extended thinking (`budget_tokens`) — DEPRECATED/REMOVED on latest models
- **NOT supported (400 error)**: Fable 5, Mythos 5, Opus 4.8, Opus 4.7, Sonnet 5
- **Deprecated**: Opus 4.6, Sonnet 4.6
- **Supported**: Opus 4.5, Haiku 4.5, earlier Claude 4

### Steering thinking via prompt
- Add guidance: "Thinking adds latency and should only be used when it will meaningfully improve answer quality — typically for problems that require multistep reasoning. When in doubt, respond directly."
- If shallow reasoning on complex problems: **raise effort to high/xhigh rather than prompting around it**
- If running at low effort for latency, add: "This task involves multistep reasoning. Think carefully through the problem before responding."

### Interleaved thinking
On Claude 4+ models, thinking can interleave with tool use (reason between tool calls).

<!-- source: anthropic-extended-thinking.md, anthropic-claude-opus-4.8-prompting.md, anthropic-claude-sonnet-5-prompting.md -->

## Prompt Structure

### XML tags — core Anthropic convention
Claude is trained to recognize content within XML tags as distinct sections. Use `<context>`, `<instructions>`, `<output_format>`, `<example>` (multiple in `<examples>`). Improves parsing accuracy, reduces instruction-mixing errors.

### Be clear and direct
Claude is like "a brilliant but new employee who lacks context." **Golden rule**: show your prompt to a colleague with minimal context; if they'd be confused, Claude will be too. Use numbered lists/bullets when order/completeness matters.

### Add context/motivation
Explain *why* a behavior matters; Claude generalizes from the explanation.

### Positive examples > negative instructions
Show desired concision rather than telling the model what not to do.

<!-- source: 2025-11-10-claude-best-practices-prompt-engineering-2026.md, anthropic-prompting-best-practices.md -->

## Agentic / Tool-Use Guidance

- **Opus 4.8 favors reasoning over tool calls** (produces better results usually); raising effort increases tool-calling behavior
- **Fable 5 excels at subagent delegation** — dispatching and sustaining parallel subagents; provide clear delegation instructions
- **Interleaved thinking with tool use** on Claude 4+ (reason between tool calls)

## Context Handling

- **Opus 4.8**: 1M context window is the **default**
- **Fable 5 / Mythos 5**: 1M context window by default, up to 128k output tokens per request
- **Fable 5 longer turns by default**: individual requests on hard tasks can run many minutes at higher effort; autonomous runs can extend hours. Adjust client timeouts, streaming, progress indicators before migrating
- **Long-context techniques**: (1) extract reference quotes before answering, (2) few-shot of correctly-answered questions about other sections

## Known Pitfalls — BREAKING CHANGES

| Pitfall | Details |
|---|---|
| **DO NOT prefill** | Returns 400 error on Fable 5, Mythos 5, Sonnet 5, Opus 4.8+. Use system prompt instructions instead. |
| **DO NOT pass budget_tokens** | 400 error on Opus 4.8/4.7, Sonnet 5, Fable 5, Mythos 5. Use adaptive thinking + effort. |
| **Sampling params not accepted** | `temperature`, `top_p`, `top_k` are NOT accepted on Sonnet 5+ and latest models. |
| **thinking: {type: "disabled"}** | NOT supported on Mythos Preview, Fable 5, Mythos 5 (400 error). |
| **max_tokens is hard limit** | Includes thinking + response. Revisit when moving from no-thinking to adaptive-thinking workloads. |
| **Fable 5 safety classifiers** | Target offensive cybersecurity, biology/life sciences, and extraction of model's summarized thinking. Benign content may trigger. Configure fallback to Opus 4.8. |
| **Fable 5 overplanning** | On ambiguous tasks, add: "When you have enough information to act, act. Do not re-derive facts already established in the conversation." |
| **Mythos 5 data retention** | Requires 30-day data retention; not available under zero data retention (ZDR). |

<!-- source: anthropic-migrating-to-claude-4.md -->

## Migration

- **Claude 3.5/3.7 → Claude 4**: extended thinking introduced; prefill still supported on Claude 4 (removed later)
- **Claude 4 → Opus 4.8 / Sonnet 5 / Fable 5**: adaptive thinking becomes default; manual extended thinking deprecated/removed; **prefill removed (400 error)**; **sampling params not accepted**; new `effort` parameter replaces thinking budget; new tokenizer on Sonnet 5+ (token counts may differ)
- **Sonnet 4.6 → Sonnet 5**: adaptive thinking on by default (was off); if previously running thinking off, try thinking on with lower effort
- **Opus 4.8 performs well on existing Opus 4.7 prompts**; Sonnet 5 on existing Sonnet 4.6 prompts — mostly drop-in

## Unique Features

- **Adaptive thinking** — model decides when/how much to think (off on Opus 4.8, on by default on Sonnet 5, always on Fable 5/Mythos 5)
- **Thinking `signature` field** in thinking content blocks (encrypted reasoning trace)
- **Opus 4.8 response length auto-calibration** to perceived task complexity
- **Fable 5 long-horizon autonomy** — multiday goal-directed runs with strong instruction retention
- **Claude API migration skill**: in Claude Code run `/claude-api migrate this project to claude-opus-4-8` to auto-apply model ID swap, breaking parameter changes, prefill replacement, and effort calibration
