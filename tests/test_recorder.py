import json

from benchmarks.committee.recorder import TrajectoryRecorder


class State:
    program_candidates = [{"p": "seed"}]
    full_program_trace = [{"tasks": [{"subsample_scores": [0.0, 1.0], "new_subsample_scores": [1.0, 1.0]}]}]


def drive(recorder):
    recorder.on_valset_evaluated({"iteration": 0, "candidate_idx": 0, "candidate": {}, "average_score": 0.2,
                                  "is_best_program": True})
    # Iteration 1: rejected.
    recorder.on_budget_updated({"iteration": 1, "metric_calls_used": 5, "metric_calls_delta": 2,
                                "metric_calls_remaining": 95})
    recorder.on_proposal_start({"iteration": 1, "parent_candidate": {}, "components": ["p"], "reflective_dataset": {}})
    recorder.on_candidate_rejected({"iteration": 1, "old_score": 1.0, "new_score": 1.0, "reason": "tie"})
    recorder.on_iteration_end({"iteration": 1, "state": State(), "proposal_accepted": False})
    # Iteration 2: accepted with a better valset score.
    recorder.on_budget_updated({"iteration": 2, "metric_calls_used": 12, "metric_calls_delta": 7,
                                "metric_calls_remaining": 88})
    recorder.on_proposal_start({"iteration": 2, "parent_candidate": {}, "components": ["p"], "reflective_dataset": {}})
    recorder.on_valset_evaluated({"iteration": 2, "candidate_idx": 1, "candidate": {}, "average_score": 0.6,
                                  "is_best_program": True})
    recorder.on_candidate_accepted({"iteration": 2, "new_candidate_idx": 1, "new_score": 2.0, "parent_ids": [0]})
    recorder.on_iteration_end({"iteration": 2, "state": State(), "proposal_accepted": True})
    # Iteration 3: merge.
    recorder.on_budget_updated({"iteration": 3, "metric_calls_used": 20, "metric_calls_delta": 8,
                                "metric_calls_remaining": 80})
    recorder.on_merge_accepted({"iteration": 3, "new_candidate_idx": 2, "parent_ids": [0, 1]})
    recorder.on_iteration_end({"iteration": 3, "state": State(), "proposal_accepted": True})


def test_rows_one_per_iteration():
    r = TrajectoryRecorder()
    drive(r)
    assert [row["iteration"] for row in r.rows] == [1, 2, 3]
    first, second, third = r.rows
    assert first["verdict"] == "rejected" and first["metric_calls_used"] == 5
    assert first["best_valset_score"] == 0.2 and first["minibatch_before"] == 0.5 and first["minibatch_after"] == 1.0
    assert second["verdict"] == "accepted" and second["best_valset_score"] == 0.6
    assert second["reflection_calls"] == 2
    assert third["verdict"] == "merge_accepted" and third["is_merge"] is True


def test_summary_fields():
    r = TrajectoryRecorder()
    drive(r)
    s = r.summary()
    assert s["iterations"] == 3
    assert s["best_valset_score"] == 0.6
    assert s["metric_calls_to_best"] == 12
    assert s["accept_rate"] == 0.5
    assert s["reflection_calls"] == 2
    assert s["metric_calls_used"] == 20
    assert s["auc_valset_vs_calls"] > 0


def test_seed_best_counts_its_valset_cost():
    r = TrajectoryRecorder()
    r.on_valset_evaluated({"iteration": 0, "candidate_idx": 0, "candidate": {}, "average_score": 0.4,
                           "num_examples_evaluated": 3, "is_best_program": True})
    assert r.summary()["metric_calls_to_best"] == 3


def test_write_trajectory(tmp_path):
    r = TrajectoryRecorder()
    drive(r)
    path = tmp_path / "trajectory.jsonl"
    r.write_trajectory(path)
    lines = path.read_text().splitlines()
    assert len(lines) == 3
    assert json.loads(lines[1])["verdict"] == "accepted"
