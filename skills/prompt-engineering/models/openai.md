# OpenAI — Model-Specific Prompting Guide

Covers GPT-4.1, GPT-5/5.6, o3/o4-mini, and reasoning models.

<!-- Sources: openai-gpt-4.1-prompting-guide.md, openai-gpt-5-introducing.md, openai-gpt-5.6-model-guidance.md, openai-reasoning-best-practices.md, openai-reasoning-models-guide.md, openai-o3-o4-mini-function-calling-guide.md, openai-structured-outputs.md, openai-prompt-engineering-guide.md -->

## Models

| Model | Type | Context | Use for |
|---|---|---|---|
| GPT-5.6 (sol/terra/luna) | Reasoning (tiered) | — | Flagship reasoning; sol=highest intelligence, terra=lower cost, luna=high volume |
| GPT-5 / GPT-5-thinking | Unified router | — | Real-time router picks efficient vs. thinking model based on complexity |
| GPT-4.1 / mini / nano | Standard | 1M | Agentic workflows, literal instruction-following, workhorse |
| o3 / o4-mini | Reasoning | — | Math, science, engineering, planning, complex agentic |

## Reasoning Behavior

**GPT-5 unified router**: automatically decides between efficient and thinking model. Steer with intent phrases like **"think hard about this"** or **"think longer"** to trigger deeper reasoning.

**GPT-5.6 / reasoning models**: use internal **reasoning tokens** (billed, hidden). Control via `reasoning.effort`:
- `low` — fastest/cheapest
- `medium` — balanced (default for many workflows)
- `high` — thorough
- `pro` (gpt-5.6-sol only) — maximum intelligence

**Start with `gpt-5.6`** for most reasoning workloads. Use `gpt-5.6-sol` with `reasoning.mode: "pro"` for hardest problems.

**Multi-turn reasoning state**: include reasoning items from previous turns to preserve reasoning state across turns. The model builds on earlier reasoning without re-deriving.

**Responses API preferred** over Chat Completions for reasoning models (improved intelligence/performance).

### CRITICAL: Don't scaffold CoT on reasoning models
- **DO NOT** add "think step by step," "reason through this carefully," or similar CoT prompts to o-series, GPT-5-thinking, or GPT-5.6 reasoning models
- They reason internally already; explicit CoT can **degrade** performance
- Keep prompts simple and direct; provide goals and constraints, not reasoning steps
- Give reasoning models MORE context, not more step-by-step instructions

<!-- source: openai-reasoning-best-practices.md, openai-reasoning-models-guide.md -->

## Prompt Structure

- **Developer messages replace system messages** for o-series. Any system message is auto-converted to developer. Use `developer` role for cross-turn instructions.
- **Use delimiters** (XML tags, markdown headers, triple quotes) to separate instructions/context/data.
- **GPT-4.1 follows instructions more literally than GPT-4o** — a feature, not a bug. If behavior differs, "a single sentence firmly and unequivocally clarifying your desired behavior is almost always sufficient."
- **Place key instructions at top AND bottom** of long context (primacy + recency).

## Agentic / Tool-Use Guidance

### GPT-4.1 agentic system prompt — three key reminders:
1. **Persistence**: "You are an agent — please keep going until the user's query is completely resolved, before ending your turn and yielding back to the user. Only terminate your turn when you are sure that the problem is solved."
2. **Tool-calling**: "If you are not sure about file content or codebase structure pertaining to the user's request, use your tools to read files and gather the relevant information: do NOT guess or make up an answer."
3. **Planning** (optional): "You MUST plan extensively before each function call and double-check that the right parameters are being passed. Verify your approach iteratively, and only proceed after confirming it's the right path."

### Tool-call best practices:
- Give explicit ordering guidance and conditions for when tools should be called
- Ask for **search-and-replace block diffs** rather than full-file rewrites (reduces errors/tokens)
- Add restraint/confirmation gates: "Before taking any irreversible action, confirm with the user first."

### o3/o4-mini function calling:
- Use developer messages for context/role
- Write clear, concise function descriptions — **verbose is worse for reasoning models**
- Few-shot examples help for complex multi-step workflows but keep minimal
- Reasoning models handle tool orchestration natively in CoT

### GPT-5.6 Programmatic Tool Calling:
- Can write JavaScript to call eligible tools, pass results between calls, and process intermediate outputs in a hosted runtime
- For complex multi-step tool workflows benefiting from in-flight computation

## Context Handling

- **GPT-4.1: 1M token context window.** Place most important instructions at top AND bottom. Use clear structural markers.
- **Limit RAG context for reasoning models** — too much irrelevant context distracts and increases latency/cost.
- For RAG, provide relevant context rather than relying on the model to find it in a massive prompt.
- **Three difficulties of long context**: (1) recall of a specific fact = needle-in-haystack, (2) summarizing/document understanding, (3) reasoning over all context — each needs different handling.
- Create **"context distillates"** for retrieval rather than feeding raw long context.

## Structured Outputs

- **JSON Schema via `response_format`**: model constrained to valid JSON conforming to schema (constrained decoding)
- Use **Pydantic (Python)** or **Zod (JS)** for schema definition
- `strict: true` mode for hard guarantees
- Refusals are programmatically distinguishable from normal responses
- **Simpler prompting**: no need for "respond ONLY in JSON" — the schema enforces format
- Supports nested objects, enums, arrays, optionals

<!-- source: openai-structured-outputs.md -->

## Known Pitfalls

- **Don't add CoT to reasoning models** — they reason internally; scaffolding hurts
- **GPT-4.1 literalism can surprise** — it won't liberally infer intent like GPT-4o did; be explicit
- **Too much irrelevant context hurts reasoning models** — increases latency/cost and distracts

## Migration

- **GPT-4o → GPT-4.1**: requires prompt migration; GPT-4.1 follows instructions more literally
- **GPT-5.5/5.4 → GPT-5.6**: start with current reasoning effort setting, then test same setting AND one level lower. GPT-5.6 often maintains/improves quality with fewer tokens. Benchmark rather than assume.

## Unique Features

- **GPT-5 unified router** with explicit intent signals ("think hard about this")
- **GPT-5.6 tiered naming** (sol/terra/luna) and Programmatic Tool Calling (JS in hosted runtime)
- **GPT-5 improvements**: reduced hallucinations, reduced sycophancy, improved instruction following
- **Planner/workhorse pattern**: reasoning models plan, GPT models execute
