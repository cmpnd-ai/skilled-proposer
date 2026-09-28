"""Run GEPA on the committee task under named ablations.

    uv run python -m benchmarks.committee.run --dry-run
    uv run python -m benchmarks.committee.run --ablations all --seeds 13 --max-metric-calls 1500

Keys and endpoints come from the environment. A .env file in the working
directory is loaded first. STUDENT_API_KEY is optional for LM Studio.
When CMPND_API_KEY is set, every student call, reflection call, and GEPA
run is traced to cmpnd under the tags below.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import dspy
from dotenv import load_dotenv

from benchmarks.committee.ablations import Ablation, resolve
from benchmarks.committee.data import DATA_PATH, load_rows, make_examples, split
from benchmarks.committee.metric import SELFTEST_CASES, committee_metric
from benchmarks.committee.recorder import TrajectoryRecorder
from benchmarks.committee.task import (
    DEFAULT_REFLECTION_MODEL,
    DEFAULT_STUDENT_API_BASE,
    DEFAULT_STUDENT_MODEL,
    build_program,
    build_reflection,
    build_student,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ablations", nargs="+", default=["all"])
    p.add_argument("--seeds", nargs="+", type=int, default=[13])
    p.add_argument("--max-metric-calls", type=int, default=1500)
    p.add_argument("--split", nargs=3, type=int, default=[556, 200, 200], metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--reflection-minibatch-size", type=int, default=15)
    p.add_argument("--num-threads", type=int, default=4)
    p.add_argument("--student-model", default=os.environ.get("STUDENT_MODEL", DEFAULT_STUDENT_MODEL))
    p.add_argument("--student-api-base", default=os.environ.get("STUDENT_API_BASE", DEFAULT_STUDENT_API_BASE))
    p.add_argument("--reflection-model", default=os.environ.get("REFLECTION_MODEL", DEFAULT_REFLECTION_MODEL))
    p.add_argument("--out", default="benchmarks/results")
    p.add_argument("--dry-run", action="store_true", help="Load data, check the metric, and exit.")
    return p.parse_args(argv)


def selftest_metric() -> None:
    for gold, pred, expected in SELFTEST_CASES:
        out = committee_metric(dspy.Example(committee=gold), dspy.Prediction(committee=pred))
        if expected is not None and abs(out.score - expected) > 1e-9:
            raise SystemExit(f"metric self-check failed for {gold!r} vs {pred!r}: {out.score}")


def instruction_words(program) -> int:
    """Word count of the instructions across the program's predictors."""
    return sum(len(pred.signature.instructions.split()) for _, pred in program.named_predictors())


def evaluate(program, examples, num_threads: int) -> float:
    result = dspy.Evaluate(devset=examples, metric=committee_metric, num_threads=num_threads,
                           display_progress=False)(program)
    return result.score / 100.0


def run_one(ablation: Ablation, seed: int, args, train, val, test, reflection_lm) -> dict:
    run_dir = Path(args.out) / ablation.name / str(seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    proposer = ablation.build_proposer(run_dir)
    recorder = TrajectoryRecorder()

    optimizer = dspy.GEPA(
        metric=committee_metric,
        reflection_lm=reflection_lm,
        instruction_proposer=proposer,
        max_metric_calls=args.max_metric_calls,
        reflection_minibatch_size=ablation.reflection_minibatch_size or args.reflection_minibatch_size,
        num_threads=args.num_threads,
        seed=seed,
        track_stats=True,
        gepa_kwargs=ablation.gepa_kwargs(extra_callbacks=[recorder]),
    )
    started = time.time()
    optimized = optimizer.compile(build_program(), trainset=train, valset=val)
    wall = time.time() - started
    optimized.save(str(run_dir / "program.json"))
    recorder.write_trajectory(run_dir / "trajectory.jsonl")

    summary = {
        "ablation": ablation.name,
        "seed": seed,
        "split": list(args.split),
        "student_model": args.student_model,
        "reflection_model": args.reflection_model,
        "max_metric_calls": args.max_metric_calls,
        "test_score": evaluate(optimized, test, args.num_threads),
        "best_instruction_words": instruction_words(optimized),
        "wall_seconds": round(wall, 1),
        **recorder.summary(),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


TRACE_TAGS = ["skilled-proposer", "committee-benchmark"]


def configure_tracing() -> bool:
    """Turn on cmpnd tracing when CMPND_API_KEY is set. Returns whether it is on."""
    if not os.environ.get("CMPND_API_KEY"):
        print("[trace] CMPND_API_KEY is not set, so tracing is off")
        return False
    import cmpnd

    cmpnd.configure(project_tags=TRACE_TAGS)
    cmpnd.auto_instrument()
    print(f"[trace] cmpnd tracing on, tags={TRACE_TAGS}")
    return True


def flush_tracing() -> None:
    import cmpnd

    cmpnd.flush_exporter()


def main(argv=None) -> None:
    load_dotenv()
    args = parse_args(argv)

    examples = make_examples(load_rows(DATA_PATH))
    train, val, test = split(examples, sizes=tuple(args.split), seed=args.seeds[0])
    print(f"[data] {len(train)} train / {len(val)} val / {len(test)} test from {DATA_PATH.name}")
    selftest_metric()
    print("[metric] metric self-check passed")
    if args.dry_run:
        return

    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)
    student = build_student(args.student_model, args.student_api_base, os.environ.get("STUDENT_API_KEY", ""))
    reflection = build_reflection(args.reflection_model)
    dspy.configure(lm=student, adapter=dspy.XMLAdapter())
    print(f"[lm] student={args.student_model} reflection={args.reflection_model}")
    tracing = configure_tracing()

    probe = reflection("Reply with the single word ready.")
    if not probe or not probe[0].strip():
        raise SystemExit("The reflection model returned nothing. Check the API key in .env.")

    try:
        for ablation in resolve(args.ablations):
            for seed in args.seeds:
                if len(args.seeds) > 1:
                    train, val, test = split(examples, sizes=tuple(args.split), seed=seed)
                print(f"[run] {ablation.name} seed={seed}")
                summary = run_one(ablation, seed, args, train, val, test, reflection)
                print(f"[done] {ablation.name} seed={seed} test={summary['test_score']:.3f} "
                      f"best_val={summary['best_valset_score']:.3f} calls_to_best={summary['metric_calls_to_best']}")
    finally:
        if tracing:
            flush_tracing()


if __name__ == "__main__":
    main()
