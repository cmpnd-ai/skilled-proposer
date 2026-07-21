# Meta (Llama) — Model-Specific Prompting Guide

Covers Llama 3, 3.1, 3.2, 3.3, and Llama 4 (Scout, Maverick).

<!-- Sources: meta-llama-3-prompt-format.md, meta-llama-3.1-prompt-format.md, meta-llama-3.2-prompt-format.md, meta-llama-3.3-prompt-format.md, meta-llama-4-herd-blog.md, meta-llama-prompt-engineering-guide.md -->

## Models

| Model | Context | Type | Notes |
|---|---|---|---|
| Llama 4 Scout | 10M | Multimodal MoE | 17B active, 16 experts, natively multimodal |
| Llama 4 Maverick | — | Multimodal MoE | 17B active, 128 experts, best performance-to-cost |
| Llama 3.3 70B | — | Text-only instruct | Supersedes 3.1 70B instruct |
| Llama 3.1 | 128K | Text instruct | Tool calling, structured output |
| Llama 3.2 (1B/3B) | 8K (quantized) | Lightweight | Strong system prompt helps small models |
| Llama 3.2 (11B/90B) | — | Vision multimodal | `<\|image\|>` token for images |
| Llama 3 | — | Text base/instruct | Original format |

## Reasoning Behavior

Llama models (3.x/4) are **standard chat/instruction models, NOT internal-reasoning models** in the OpenAI-o/DeepSeek-R1 sense. **CoT/step-by-step prompting IS a valid and documented technique** for Llama reasoning tasks — unlike reasoning models where it hurts.

Llama 4 is **natively multimodal** (trained from start on text + vision + audio for some variants); image and text tokens processed together within MoE layers. Mixture-of-Experts routing is internal; from a prompting standpoint, treat like a standard chat model.

<!-- source: meta-llama-prompt-engineering-guide.md, meta-llama-4-herd-blog.md -->

## Prompt Structure

### Special tokens (shared across 3/3.1/3.2/3.3)
- `<|begin_of_text|>` (BOS), `<|eot_id|>` (end of message), `<|start_header_id|>{role}<|end_header_id|>` (roles: system/user/assistant), `<|end_of_text|>` (EOS)
- Newlines (0x0A) are part of the format
- Instruct prompt ends with the assistant header to start completion
- **Single optional system message at start**, then alternating user/assistant, always ending with last user message + assistant header

### Use the chat template
Use `apply_chat_template` rather than hand-formatting. Don't free-form prompt the instruct model.

### Vision models (3.2 11B/90B)
Insert `<|image|>` token before the text instruction within the same user turn. Multiple images allowed (each with own `<|image|>`). Give a clear task: "Describe," "What is unusual about," "Extract all text from."

### Lightweight models (1B/3B)
Use a **strong explicit system prompt** (small models benefit more). Be concise — small models can drift with overly long prompts.

### General Llama prompting tips
Be clear and concise; use specific few-shot examples; break down complex tasks; provide context; iterate.

<!-- source: meta-llama-3-prompt-format.md, meta-llama-3.1-prompt-format.md, meta-llama-3.2-prompt-format.md -->

## Agentic / Tool-Use Guidance

- **Llama 3.1 introduced tool calling** (function-calling format), later superseded by the more flexible **Llama 3.2 format**
- For tool use on 3.1/3.2/3.3 models, **prefer the Llama 3.2 function-calling format** (functions can be provided in the user message)
- **Llama 3.3 zero-shot function calling** uses the Llama 3.2 format — JSON-schema-style function definitions inside the system message; model emits an assistant turn with special function-call syntax
- **Structured output** (JSON conforming to schema) supported from 3.1

## Context Handling

- **Llama 3.1**: up to **128K context**
- **Llama 3.2 quantized instruct models**: reduced context length of **8K** (vs. full lightweight models)
- **Llama 4 Scout**: industry-leading **10M context** — reconsider prompt caching/RAG strategies given the huge window

## Known Pitfalls

- **Llama 3.3 is text-only** — no vision (use 3.2 11B/90B or Llama 4 for multimodal)
- **Llama 3.2 quantized instruct models have only 8K context** (vs. full lightweight models)
- **Small/lightweight models drift with overly long prompts** — be concise
- **False refusals and "preachy" responses** are a known issue across Llama generations — mitigated by a good system prompt

## Unique Features

### Llama 4 refusal/preachy reduction via system prompt
Meta explicitly recommends a good system prompt to reduce false refusals and preachy outputs. This is a **first-class prompting consideration** for Llama 4 — if the model is refusing benign requests or producing preachy responses, improve the system prompt before trying other techniques.

### Llama 4 tone adaptation
The template instructs the model to adapt tone to user needs (humor, empathy, formality); a strong system prompt controls this.

### Llama 4 Scout 10M context
Industry-leading context window — rethink RAG/caching strategies.

### Natively multimodal MoE
Text + vision + audio tokens in same MoE layers (not bolted-on vision).

### `<|image|>` marker for vision prompts (3.2)

## Migration

- **Llama 3 prompts work unchanged in 3.1/3.2/3.3**, but Meta recommends updating to the new format for best results
- **Llama 3.3 70B supersedes instruction-tuned Llama 3.1 70B** — use `Llama-3.3-70B-Instruct` wherever you'd use 3.1 70B instruct
- **Llama 3.3 prompt format = Llama 3.1** (no changes); **function calling = Llama 3.2 format**
- **Llama 3.1/3.2 templates are applicable across all 3.1 and 3.2 models**
