# xAI (Grok) — Model-Specific Prompting Guide

Covers Grok 4.5 and Grok prompt caching.

<!-- Sources: xai-grok-reasoning.md, xai-grok-prompt-caching-best-practices.md, xai-grok-prompts-repo.md -->

## Models

| Model | Type | Key feature |
|---|---|---|
| Grok 4.5 | Reasoning (always on) | `reasoning_effort` low/medium/high; reasoning can't be disabled |
| Grok 3 / 4 / 4.1 | Standard + reasoning variants | Official system prompts in xai-org/grok-prompts repo |

## Reasoning Behavior

**Grok 4.5 reasoning is always on — you cannot disable it.** You can only tune effort via the `reasoning_effort` parameter.

### `reasoning_effort` levels

| Setting | Description | Best for |
|---|---|---|
| `"low"` | Some reasoning tokens, still fast | Latency-sensitive agentic use, simple tool calling |
| `"medium"` | More thinking | Complex data analysis, long-context reasoning |
| `"high"` (default) | More reasoning tokens for deeper thinking | Very challenging problems, complex math, multi-step logic, competition-level tasks |

### CRITICAL: Don't add CoT to Grok reasoning models
- **Don't add "think step by step"** to prompts — Grok reasons internally; just state the problem and any constraints
- Use `reasoning_effort="low"` for latency-sensitive agentic/tool-calling tasks
- Use `"high"` for hard math/proof/multi-step-logic tasks

### Unsupported parameters with reasoning models
`presencePenalty`, `frequencyPenalty`, and `stop` **cannot be used** with reasoning models. Requests including them return an error.

<!-- source: xai-grok-reasoning.md -->

## Encrypted Reasoning Content

The reasoning content is encrypted by xAI and can be returned if you pass `include: ["reasoning.encrypted_content"]` to the Responses API. You can send the encrypted content back to provide more context to a previous conversation.

When using the Vercel AI SDK, encrypted reasoning content is automatically included under the hood (as long as `store: false` is not specified).

**Use this for multi-turn agentic flows** — pass encrypted reasoning back across turns to preserve thinking context.

## Prompt Caching Best Practices

### Stable conversation ID
- **Always set `x-grok-conv-id`** (or `prompt_cache_key` for Responses API) — routes requests to the same server, maximizing cache hits
- Use a UUID or your application's session ID

### Front-load static content
Place system prompts, few-shot examples, and reference documents at the beginning where they form a stable prefix:
```
system prompt → few-shot examples → reference documents → variable user query
```

### Never modify earlier messages
Only append new ones. Any edit, removal, or reorder breaks the cache.

### Other tips
- Monitor `cached_tokens` — if consistently 0, verify conversation ID and message ordering
- Handle cache misses gracefully — eviction and routing mean hits aren't guaranteed
- Caching doesn't affect output quality — only accelerates prompt processing
- Caching works with streaming, tool calls, and function calling

<!-- source: xai-grok-prompt-caching-best-practices.md -->

## Known Pitfalls

- **Reasoning cannot be disabled** — only tuned via `reasoning_effort`
- **Don't use `presence_penalty`, `frequency_penalty`, or `stop`** with reasoning models — they'll error
- **Don't add "think step by step"** — Grok reasons internally
- **Set a generous timeout** — xAI's own examples use 3600s for high-effort proofs

## Unique Features

- **Reasoning always on** (can't disable, only tune effort)
- **Encrypted reasoning content** — pass back across turns for multi-turn context preservation
- **Prompt caching with stable conversation IDs** — the main lever for cost control (since sampling params are unsupported on reasoning models)
- **Official system prompts repository** (xai-org/grok-prompts) — Grok 3, 4, 4.1 system-turn prompt templates (Jinja2) for thinking and non-thinking, with and without tools

## Prompt Ordering for Maximum Cache Efficiency

```
[System prompt]          ← static, cached
[Few-shot examples]      ← static, cached
[Reference documents]    ← static, cached
[Variable user query]    ← changes each turn
```

Treat earlier turns as immutable in multi-turn apps — append only. Use a consistent conversation ID per session so Grok routes to a warm cache.
