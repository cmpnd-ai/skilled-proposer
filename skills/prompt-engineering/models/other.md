# Other Vendors — Cohere and Mistral

Covers Cohere Command A / Command R+ and Mistral Large 2.

<!-- Sources: cohere-command-a.md, cohere-command-r-plus.md, mistral-large-2-model-card.md -->

---

## Cohere

### Models

| Model | Context | Max Output | Use for |
|---|---|---|---|
| Command A (111B) | 256K | 8K | Tool use, RAG, agents, multilingual — successor to Command R+ |
| Command R+ (08-2024) | 128K | 4K | Complex RAG + multi-step tool use (agents) |

### CRITICAL: Override the chatty default
**Command A and Command R+ are "chatty" by default** — verbose, uses markdown, optimized for conversation. To get concise, non-markdown output (e.g., for downstream parsing), explicitly instruct the model in the system prompt:

> "Simply provide the answer and do not use markdown or code block markers."

This is the **most important prompting quirk** for Cohere models. If your output is too verbose or includes unwanted markdown, add this system instruction.

<!-- source: cohere-command-a.md, cohere-command-r-plus.md -->

### Agentic / Tool-Use Guidance

- **Command A excels at multi-step REACT agents** — structure prompts to encourage subgoal decomposition and active information seeking
- **Command A avoids unnecessarily calling tools** — you don't need to over-prompt it to be judicious about tool use, but do provide clear tool descriptions so it can decide well
- **Command R+ for complex RAG + multi-step tool use**; Command R for simpler RAG / single-step tool use
- The 08-2024 update improved instruction-following from the **system message** — use it for persona, tone, and constraints

### RAG Guidance

- **Provide documents in the prompt and ask for citations** — the model is trained to cite grounding documents
- **The model declines unanswerable questions** — phrase RAG prompts so the model knows it can say "I don't know" when documents don't contain the answer
- Use the full 256K context (Command A) for long-document RAG and financial/numerical extraction rather than chunking aggressively

### Multilingual

- Command A: trained for 23 languages (English, French, Spanish, Italian, German, Portuguese, Japanese, Korean, Chinese, Arabic, Russian, Polish, Turkish, Vietnamese, Dutch, Czech, Indonesian, Ukrainian, Romanian, Greek, Hindi, Hebrew, Persian)
- **Prompt in the user's language** — the model responds in kind, or follow an explicit instruction to output in a specific language
- Excels at cross-lingual tasks (translation, answering questions about content in other languages)

### Known Pitfalls

- **Chatty by default** — always override with system instruction for concise/non-markdown output
- **Match model to task**: Command R+ for complex RAG + agents; Command R for simpler tasks
- **Command A is successor to Command R+** — migrate for higher throughput and longer context; prompting patterns carry over

### Unique Features

- **Chatty default behavior** (needs explicit system instruction override for concise output)
- **REACT agent optimization** — tuned for multi-step tool use and subgoal decomposition
- **Judicious tool use** — avoids unnecessary tool calls
- **Citation generation for RAG** — trained to cite grounding documents
- **Safety modes** — granular control over output generation per context
- **Robust to whitespace/newline changes** — minor prompt formatting tweaks won't drastically change output

---

## Mistral

### Models

| Model | Context | Type | Notes |
|---|---|---|---|
| Mistral Large 2.0 (24-07) | — | Open-weight (MRL) | Flagship; links to prompting, function-calling, tokenization docs |
| Mistral Large 2.1 (24-11) | — | Open-weight | Updated version |
| Mistral Large 3 (25-12) | — | Open-weight | Latest |
| Magistral Medium (25-06) | — | Reasoning | Reasoning model variant |

### Prompt Structure

- **Strongly recommends using a system prompt** to set the assistant's persona, role, and behavioral guardrails. Placed in the `system` role of the chat template.
- **Control tokens**: `[INST] ... [/INST]` for instruction turns in older Mistral/Mixtral models; richer set for function calling in newer models
- **Use `apply_chat_template`** rather than hand-formatting prompts

### Function Calling — Five Steps

1. **Developer** specifies functions/tools and a system prompt (optional)
2. **User** queries the model powered with the new functions/tools
3. **Model** decides whether to call a tool and emits a tool call
4. **Tool** result is fed back
5. **Model** produces the final answer using the tool result

### Sampling Recommendations

- **General chat**: `temperature` ~0.7, `top_p` ~0.95-1.0
- **Structured/extractive tasks**: `temperature` 0.0-0.3 (lower for more deterministic output)

### Known Pitfalls

- **Use `apply_chat_template`** — don't hand-format with control tokens
- **Set a system prompt** — Mistral models benefit significantly from explicit persona/role/guardrail setting

### Unique Features

- **Open-weight** (MRL license) — can run locally
- **Five-step function calling pattern** — well-documented, structured approach
- **Magistral** — reasoning model variant (separate from standard Mistral Large)

### When to Consider Mistral

Mistral has **no major unique prompting quirks** beyond the standard chat model patterns (system prompt, control tokens, apply_chat_template). If the student model is Mistral:
- Use a strong system prompt for persona/role/constraints
- Use `apply_chat_template` for formatting
- Follow the five-step function calling pattern for tool use
- Adjust temperature per task type (0.7 general, 0.0-0.3 structured)
- **CoT prompting IS valid** — Mistral Large is a standard chat model, not an internal-reasoning model. Use "think step by step" if the task requires multi-step reasoning.
- For **Magistral** (reasoning variant), check if it reasons internally — if so, don't add CoT scaffolding (same rule as other reasoning models)
