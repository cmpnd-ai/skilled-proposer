# Google (Gemini) — Model-Specific Prompting Guide

Covers Gemini 3, Gemini 2.5, and Gemini Image (Nano Banana).

<!-- Sources: google-gemini-3-developer-guide.md, google-gemini-prompt-design-strategies.md, google-gemini-thinking-mode.md, google-gemini-image-prompt-guide.md -->

## Models

| Model | Type | Key feature |
|---|---|---|
| Gemini 3 | Thinking (dynamic by default) | `thinking_level` parameter, thought signatures |
| Gemini 2.5 Pro / 2.5 series | Thinking | `thinkingBudget` parameter |
| Gemini 3.1 Flash Image (Nano Banana) | Image generation | "Thinks through" image prompts |

## Reasoning / Thinking Behavior

**Gemini 3 uses dynamic thinking by default** — reasons internally before responding. Control via **`thinking_level`** parameter (evolution of `thinkingBudget` from 2.5).

**Gemini 2.5 series**: use **`thinkingBudget`** — guides the model on specific number of thinking tokens. Setting `0` effectively disables thinking (where supported); higher budgets = more internal reasoning.

**Thought summaries** (not raw chain) returned as dedicated `thought` parts/steps, chronologically alongside function calls, user inputs, model outputs.

**Streaming with thinking**: thought chunks stream first, then final response content.

### CRITICAL: Don't add CoT to thinking models
- **Don't ask the model to "think step by step"** for Gemini thinking models — they already think internally
- Describe the task and constraints clearly; use thinking level/budget to trade latency/cost vs quality

<!-- source: google-gemini-thinking-mode.md -->

## Prompt Structure

### Gemini 3 core prompting principles
- **Clear and specific instructions**: well-structured input, explicit constraints, response format
- Named sections: Input, Constraints, Response format
- **Break prompts into components**: role/context, task, constraints, examples, output format
- **Delimiters**: XML tags, markdown headers to help model parse structure
- **Few-shot examples**: always include; even a single example can significantly improve performance
  - 1-2 for simple classification, 4+ for nuanced tasks (including edge cases and adversarial/negative examples)
  - Consistent example formatting

### Explicit reasoning instructions (Gemini 3)
"Reason step-by-step. Decompose the problem. Reflect before answering. For planning, enumerate the steps and verify each."

### Fallback responses
Define what the model should do when uncertain — say it doesn't know or ask for clarification.

<!-- source: google-gemini-prompt-design-strategies.md -->

## Agentic / Tool-Use Guidance

### Agentic system instruction template
Structured system instruction covering: Identity, Objective, Available tools, Constraints, Reasoning approach, Output expectations.

### Agentic workflow patterns
- **Reasoning and strategy**: prompt to plan before acting
- **Execution and reliability**: structured tool use and verification
- **Interaction and output**: formatting/surfacing results

### Gemini 3 features
- **Combine built-in tools and function calling** in the same request (search, code execution alongside developer-defined functions)
- **Structured Outputs with tools** — combine structured output schemas with function calling
- **Multimodal function responses** — tool results that include images
- **Code Execution with images** — run generated code over image inputs

### Multi-turn agentic flows
Use **thought signatures** to preserve reasoning context across tool calls.

## Context Handling

- Long context topic-specific guide linked separately in Gemini docs
- For multi-turn agentic flows, use thought signatures (stateful or stateless mode) to preserve reasoning context across turns

## Thought Signatures

Carry reasoning context across turns. Two modes:

- **Stateful mode (Recommended)**: API stores thoughts server-side keyed by interaction; you don't resend
- **Stateless mode**: you receive a signed/encrypted thought token and must pass it back in subsequent turns

<!-- source: google-gemini-thinking-mode.md, google-gemini-3-developer-guide.md -->

## Image Prompting (Nano Banana / Gemini Image)

Canonical structure: **style/medium → subject → setting → action → composition**

Example: "A photo (style) of a friendly and relaxed snow leopard (subject), walking through a rocky, snowy landscape with melting snow and small flowers, with a contrail in the sky (setting). It has one paw raised as it is walking towards us (action). A full body portrait (composition)."

Editing with prompts: change character, adjust composition, alter action, swap set — preserving other elements.

<!-- source: google-gemini-image-prompt-guide.md -->

## Known Pitfalls

- **Don't add "think step by step"** to thinking models
- Thinking tokens are billed
- `thinkingBudget` from 2.5 → `thinking_level` on Gemini 3 (parameter name change)

## Migration

- **Gemini 2.5 → Gemini 3**: parameter changes (`thinkingBudget` → `thinking_level`), behavioral differences
- **OpenAI-compatible endpoint** for Gemini 3 with documented mappings for reasoning effort and other parameters
- **Interactions API** is now GA; Google recommends it for all latest features/models

## Unique Features

- **Thought signatures** — stateful (server-stored) or stateless (encrypted token) reasoning context preservation
- **`thinking_level`** replacing `thinkingBudget`
- **Combine built-in tools + function calling** in same request
- **Image generation models "think through" image prompts**
- **Multimodal function responses** (tool results with images)
- **Code Execution with images**
