# Iteration Log Template

Use this template for each prompt optimization run. Copy it, fill it in, and append new iterations as you go. The log serves three purposes: (1) avoid repeating dead ends, (2) detect overfitting, (3) enable rollback.

---

## Run Metadata

- **Date**: YYYY-MM-DD
- **Student model**: <vendor/model name>
- **Task description**: <one sentence describing what the student model should do>
- **Success threshold**: <e.g., 95% of eval cases pass>
- **Control prompt**: <minimal baseline — just the task description, no optimization>
- **Eval case count**: <N optimization set + M held-out test set>

---

## Iteration 0: Baseline

- **Candidate**: <control prompt — minimal baseline>
- **Techniques applied**: None (baseline)
- **Eval results**: X/Y pass (Z%)
  - Pass: <list case IDs>
  - Fail: <list case IDs with failure reason>
- **Held-out results**: X/M pass (Z%)
- **Failure modes**: <list failure mode IDs from taxonomy>
- **Next move**: <which technique to try first, targeting which failure mode>

---

## Iteration 1

- **Candidate**: <full prompt text or diff from previous>
- **Technique applied**: <technique name from references/techniques.md>
- **Failure mode targeted**: <FM# from taxonomy>
- **Model-specific check**: <which models/<vendor>.md was consulted, any quirks noted>
- **Reasoning**: <why this technique for this failure mode>
- **Eval results**: X/Y pass (Z%) — [↑/↓/→ from previous]
  - Fixed: <case IDs that now pass>
  - Regressed: <case IDs that now fail but passed before>
  - Still failing: <case IDs still failing>
- **Held-out results**: X/M pass (Z%) — [↑/↓/→ from previous]
- **Overfitting check**: [✓/✗] held-out improvement matches optimization set improvement
- **Decision**: [Keep as new baseline / Revert / Try different technique]
- **Next move**: <what to try next, targeting which failure mode>

---

## Iteration 2

- **Candidate**: <full prompt text or diff from previous>
- **Technique applied**: <technique name>
- **Failure mode targeted**: <FM#>
- **Model-specific check**: <vendor file consulted>
- **Reasoning**: <why this technique>
- **Eval results**: X/Y pass (Z%) — [↑/↓/→ from previous]
  - Fixed: <case IDs>
  - Regressed: <case IDs>
  - Still failing: <case IDs>
- **Held-out results**: X/M pass (Z%) — [↑/↓/→]
- **Overfitting check**: [✓/✗]
- **Decision**: [Keep / Revert / Try different technique]
- **Next move**: <what to try next>

---

<!-- Continue adding iterations as needed -->

---

## Stop Criteria Checklist

- [ ] Pass rate meets threshold (≥ success threshold on optimization set)
- [ ] Held-out set performance matches optimization set (no overfitting)
- [ ] No regressions from previous baseline
- [ ] Control prompt still outperformed by current best
- [ ] OR: 3+ consecutive iterations showed <2% improvement (plateau reached)
- [ ] OR: diminishing returns — each iteration costs more than the quality gain justifies

## Summary

- **Total iterations**: N
- **Best variant**: Iteration #X
- **Best pass rate**: X/Y (Z%) on optimization set, X/M (Z%) on held-out
- **Control pass rate**: X/Y (Z%) — improvement of ΔZ%
- **Techniques that worked**: <list>
- **Techniques that didn't**: <list — avoid these for this task/model in future>
- **Key insight**: <what was the highest-impact change and why>
