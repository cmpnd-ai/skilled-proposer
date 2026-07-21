# Qwen (Qwen3) — Model-Specific Prompting Guide

Covers Qwen3 Instruct-2507, Qwen3-Thinking-2507, and hybrid Qwen3-2504 models.

<!-- Sources: qwen3-quickstart-thinking-mode.md, qwen3-blog-hybrid-thinking.md -->

## Models

| Model | Mode | Key feature |
|---|---|---|
| Qwen3-Instruct-2507 | Non-thinking only | Does NOT generate `<think>` blocks |
| Qwen3-Thinking-2507 | Thinking only | Always reasons; parse `</think>` to split reasoning from answer |
| Qwen3-2504 (hybrid) | Both modes | `/think` and `/no_think` in-prompt toggles |

**Important**: The 2507 variants split thinking/non-thinking into two dedicated models. The original 2504 hybrid models support both modes in one model via toggle.

## Reasoning Behavior

### Qwen3-Instruct-2507 (non-thinking)
- Supports **only non-thinking mode** — does NOT generate `<think></think>` blocks
- Specifying `enable_thinking=False` is no longer required or supported
- **May use CoT automatically** for complex tasks (even without thinking mode)
- Recommended output length: **16,384 tokens** for most queries

### Qwen3-Thinking-2507 (thinking)
- Supports **only thinking mode** — always reasons before answering
- The default chat template automatically includes `<think>` — so the model's output may contain only `</think>` without an explicit opening `<think>` tag
- To parse: find `</think>` (token id 151668) and split — everything before is thinking content, everything after is the final answer
- Recommended output length: **32,768 tokens** (reasoning can be long)

### Qwen3-2504 (hybrid) — `/think` and `/no_think` toggles
- Add **`/no_think`** to a user message for fast, non-reasoning responses (simple Q&A, chitchat)
- Add **`/think`** (or omit the toggle, since thinking is the default) for complex reasoning, math, coding
- These toggles let you control per-turn whether the model reasons step-by-step or responds quickly

### CRITICAL: Don't add CoT to Thinking variants
- **Avoid adding "think step by step"** to the prompt for Thinking variants — they already reason; just state the task

<!-- source: qwen3-quickstart-thinking-mode.md -->

## Sampling Parameters

| Variant | temperature | top_p | top_k | min_p |
|---|---|---|---|---|
| Qwen3-Instruct-2507 | 0.7 | 0.8 | 20 | 0 |
| Qwen3-Thinking-2507 | 0.6 | 0.8 | 20 | 0 |

For supported frameworks, adjust `presence_penalty` between 0 and 2 to reduce repetitions. Higher values may cause language mixing and slight performance decrease.

## Known Pitfalls

- **Don't try to force thinking mode on the Instruct variant** — it only supports non-thinking
- **Don't try to disable thinking on the Thinking variant** — it only supports thinking
- **The Thinking variant may not emit an opening `<think>` tag** — the chat template injects it; parse by finding `</think>` (token id 151668)
- **Allow generous `max_new_tokens`** — 16,384 for Instruct, 32,768 for Thinking (reasoning can be long)
- **Don't add "think step by step"** to Thinking variants — they already reason internally

## Unique Features

- **Two dedicated model variants** (Instruct vs Thinking) instead of one toggleable model — pick the right variant for your task
- **`/think` and `/no_think` in-prompt toggles** for the original 2504 hybrid models — per-turn control of reasoning
- **`<think>` block parsing** — split reasoning from final answer using `</think>` token (id 151668)
- **119-language support**
- **Open-weight lineup**: MoE 235B/30B and dense 0.6B-32B
- **Thinking-budget control** (on hybrid models)

## Prompting Implications

- **Pick the right variant**: Use Qwen3-Instruct-2507 for fast general chat; use Qwen3-Thinking-2507 for hard reasoning
- **Use the recommended sampling params** for each variant
- **Parse `</think>`** to split reasoning from the final answer
- **For hybrid 2504 models**, use `/think` / `/no_think` toggles per turn instead of two models
- **Avoid adding "think step by step"** to the Thinking variant — just state the task

<!-- source: qwen3-quickstart-thinking-mode.md, qwen3-blog-hybrid-thinking.md -->
