# DeepSeek — Model-Specific Prompting Guide

Covers DeepSeek R1, V3.2, and V4-Pro (thinking mode).

<!-- Sources: deepseek-r1-prompting.md, deepseek-thinking-mode.md, deepseek-v3.2-release.md -->

## Models

| Model | Type | Key feature |
|---|---|---|
| DeepSeek V4-Pro | Thinking (toggleable) | `reasoning_effort` control, `reasoning_content` output |
| DeepSeek V3.2 / V3.2-Speciale | Reasoning-first | First DeepSeek model with thinking integrated into tool-use |
| DeepSeek R1 / R1-Zero | RL-trained reasoning | Emits reasoning before final answer; distilled variants available |

## Reasoning Behavior

**R1 reasons internally and emits reasoning before the final answer.** It produces natural `<think>...</think>`-style reasoning tokens before the response.

**V4-Pro thinking mode**: before outputting the final answer, the model first outputs a chain-of-thought reasoning to improve accuracy. The CoT content is returned via the `reasoning_content` parameter, at the same level as `content`.

### Thinking mode toggle and effort control

| Control | OpenAI Format | Anthropic Format |
|---|---|---|
| Thinking toggle | `{"thinking": {"type": "enabled/disabled"}}` | (same) |
| Effort control | `{"reasoning_effort": "high/max"}` | `{"output_config": {"effort": "high/max"}}` |

- Thinking toggle defaults to `enabled`
- Default effort is `high` for regular requests; `max` for complex agent requests (Claude Code, OpenCode)
- `low` and `medium` are mapped to `high`; `xhigh` is mapped to `max`

### CRITICAL: Don't add CoT prompts to DeepSeek reasoning models
- **Do NOT add "think step by step"** or similar CoT prompts — R1 and thinking-mode models already reason internally
- Adding CoT instructions can **interfere** with the model's trained reasoning pattern
- Just state the task clearly; let the model produce its natural reasoning

<!-- source: deepseek-r1-prompting.md, deepseek-thinking-mode.md -->

## Prompt Structure

- **Use a system prompt that sets role and constraints**, not one that tells the model *how* to think. The default R1 system prompt is minimal (e.g., "You are a helpful assistant.")
- **Don't force a specific reasoning format.** Let the model produce its natural reasoning tokens.
- **For math/code, request the final answer in a parseable form**: `\boxed{}`, a code block, or a JSON field
  - Official R1 directive: "Please reason step by step, and put your final answer within `\boxed{}`."

## Multi-Turn and Tool-Use Reasoning

### `reasoning_content` concatenation rules
- Between two `user` messages, if the model **did not perform a tool call**: the intermediate assistant's `reasoning_content` does **not** need to be in context. If passed back, it will be ignored.
- Between two `user` messages, if the model **performed a tool call**: the intermediate assistant's `reasoning_content` **must** be passed back to the API in all subsequent turns, or the model loses its reasoning context.

### V3.2: Thinking integrated into tool-use
- V3.2 is the **first DeepSeek model to integrate thinking directly into tool-use** — it reasons *while* calling tools
- Both thinking and non-thinking modes are supported for tool use
- V3.2-Speciale is reasoning-maxed but has **no tool use** — prompt it for pure reasoning/proof tasks (math olympiad, competition programming)

<!-- source: deepseek-v3.2-release.md, deepseek-thinking-mode.md -->

## Known Pitfalls

- **Don't add CoT prompts** — the model reasons internally; external CoT interferes
- **Avoid setting temperature/sampling params** in thinking mode — `temperature`, `top_p`, `presence_penalty`, `frequency_penalty` are ignored (no error but no effect)
- **Don't use `temperature=0`** for R1 — can cause repetition issues in some frameworks. Use **0.5-0.7** for R1.
- **Always pass `reasoning_content` back for tool-use turns** — otherwise the model loses reasoning context across the tool call
- **Parse `reasoning_content` separately from `content`** if you want to display or log reasoning

## Unique Features

- **`\boxed{}` directive** for math answers — the official recommended way to get parseable final answers from R1
- **`reasoning_content` output** — separate field for the model's chain-of-thought, alongside `content`
- **V3.2 thinking-in-tool-use** — first DeepSeek model to reason while calling tools (no need to choose between "reasoning" and "tool use")
- **Distilled variants** (R1-Distill-Qwen-1.5B/7B/14B/32B, R1-Distill-Llama-8B/70B) — inherit same prompting guidance
- **V3.2-Speciale** — reasoning-maxed for competition-level math/programming (API-only, no tool use)

## Temperature Recommendations

- **R1**: low temperature (0.5-0.7) to avoid incoherent reasoning loops
- **Thinking mode**: sampling params ignored — don't bother setting them
