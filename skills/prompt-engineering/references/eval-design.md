# Eval Design — How to Design Evals and Success Criteria for the Student Model

Synthesized from Promptfoo, Langfuse, Anthropic long-context eval, Gemini, and GPT-4.1 sources.

<!-- All sources from ~/prompt-engineering/ corpus -->

## The Eval-Driven Mindset

**AI engineering is inherently empirical.** LLMs are nondeterministic — the same prompt can produce different outputs on different runs. You cannot reason your way to a good prompt; you must test it. Build informative evals and iterate often.

<!-- source: 2025-04-14-openai-gpt-4-1-prompting-guide.md -->

## Building Eval Cases

### How many cases?

- **Minimum**: 10-15 representative cases covering normal, edge, and adversarial inputs
- **Robust**: 30-50 cases with diverse coverage
- **Production**: 100+ cases including synthetic variations and adversarial inputs

### What to cover

1. **Normal cases** — typical inputs the model will see in production
2. **Edge cases** — boundary conditions, unusual formats, minimal/maximal inputs
3. **Adversarial cases** — inputs designed to break the prompt (injection attempts, format manipulation, ambiguous requests)
4. **Regression cases** — inputs that failed in previous iterations (prevent regressions)

### Using synthetic data to expand coverage

When real data is limited, use LLMs to generate diverse inputs including adversarial query variations. This amplifies test coverage and helps prevent overfitting to narrow patterns.

<!-- source: 2025-11-12-langfuse-llm-evals-roadmap.md -->

## Pass Conditions: Deterministic + Probabilistic

Blend both types — deterministic alone misses semantic errors; probabilistic alone is noisy.

### Deterministic checks
- **Format validation**: output conforms to expected schema (JSON, XML, specific structure)
- **Key fact presence**: required information is present in the output
- **Length constraints**: output within expected length bounds
- **Regex matching**: output matches expected patterns
- **Tool call validity**: correct tool called with correct parameters

### Probabilistic checks (LLM-as-judge)
- **Semantic accuracy**: does the output correctly answer the question?
- **Faithfulness**: does the output stick to provided facts (no hallucination)?
- **Tone/style**: does the output match the desired tone?
- **Completeness**: does the output cover all required aspects?

**LLM-as-judge** provides scalable measurement for a fast development loop, especially for CI/CD. Use a stronger model as judge if available.

<!-- source: 2025-11-12-langfuse-llm-evals-roadmap.md, promptfoo-intro-evaluation-redteaming.md -->

## RAG-Specific Metrics

For retrieval-augmented generation:
- **Retrieval relevance**: were the right documents retrieved?
- **Answer faithfulness**: does the answer stick to retrieved facts?
- **Context completeness**: did the retrieved context contain all needed information?

<!-- source: 2025-11-12-langfuse-llm-evals-roadmap.md -->

## The Evaluation Flywheel

Evaluation is a continuous iterative loop, not a one-time gate:

```
Observability → Error Analysis → Testing → Synthetic Data → Experiments → back to Observability
```

1. **Observability**: log inputs, outputs, latencies, metadata. Turn the black box into an inspectable system.
2. **Error analysis**: review traces, classify issues (hallucinations, irrelevance, formatting errors), cluster similar errors, prioritize fixes.
3. **Testing**: formalize deterministic + probabilistic checks to prevent regressions.
4. **Synthetic data**: generate diverse inputs including adversarial variations.
5. **Experiments**: compare prompt/model variants with statistical interpretation.
6. **Back to observability**: monitor production performance continuously.

<!-- source: 2025-11-12-langfuse-llm-evals-roadmap.md -->

## Promptfoo: Test-Driven LLM Development

Promptfoo is an open-source CLI/library for test-driven LLM development:

- **Declarative YAML test cases** — no code, language-agnostic
- **Automatic scoring** — define metrics and expected outputs
- **Matrix views** — side-by-side comparison of prompts × inputs
- **Caching, concurrency, live reloading** for fast iteration
- **Red teaming** — scan for security vulnerabilities and compliance risks
- **CI/CD integration** — GitHub action to prevent regressions
- **Multi-provider** — test across multiple models/providers

**5-step workflow**: Define test cases → configure evaluation → run → analyze → feedback loop (expand test cases as you gather examples and user feedback).

<!-- source: promptfoo-intro-evaluation-redteaming.md -->

## Long-Context Eval Methodology (Anthropic)

For evaluating long-context recall:

1. **Randomized collage**: Split a document into sections. Generate 5 MCQs per section (3 wrong, 1 right). Reassemble randomized sets of sections into long documents.
2. **Discard too-difficult questions**: Remove the ~10% of questions the model can't answer even in short context — they're too difficult to test long context.
3. **Avoid eval-question pitfalls**:
   - Questions answerable without the document
   - Unintentional clues (right answer more detailed than wrong)
   - "This document" references (ambiguous after stitching)
   - Questions that contain the answer
4. **Use few-shot question-writing examples**: Two sample meeting chunks with hand-written questions as few-shot examples + writing guidelines to get the LLM to write evals in the sweet spot of difficulty.

<!-- source: 2023-09-23-anthropic-long-context-prompting.md -->

## Iteration Best Practices

### Change one element at a time
Change a single element, run evals, measure. Only then change the next. This attributes effects to specific changes and prevents confounding.

<!-- source: google-gemini-prompt-design-strategies.md -->

### Hold out a test set
Split eval cases into optimization set (80%) and held-out test set (20%). Only iterate against the optimization set. Check the held-out set periodically to detect overfitting.

### Hold a control prompt
Write a minimal baseline prompt (just the task description, no optimization). Compare every iteration against it. If an iteration's pass rate drops below the control, discard it.

### Track what's been tried
The iteration log prevents repeating dead ends and detects overfitting. See `templates/iteration-log.md`.

## When Automatic Optimization Helps

DSPy's systematic prompt optimization enhances performance best when instruction tuning and example selection are optimized together, but impact varies by task:
- Eval accuracy criterion: 46.2% → 64.0%
- Router agent: 85.0% → 90.0%
- Guardrails: only modest gains
- Hallucination detection: selective enhancements

**No one-size-fits-all.** Evaluate your specific use case. If you have a metric and a training set, automated optimization (DSPy/GEPA, TextGrad, APE) can find better prompts than manual iteration. If you don't have a metric, build one first.

<!-- source: 2025-07-04-dspy-prompts-as-code.md -->
