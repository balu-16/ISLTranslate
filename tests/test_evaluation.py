from isl_slt.config import Config
from isl_slt.evaluation import generation
import pytest

from isl_slt.evaluation.analysis import aggregate_runs
from isl_slt.utils.io import fingerprint, write_json


def test_effective_generation_mismatch_cannot_aggregate(tmp_path):
    paths = []
    for seed, maximum in [(13, 64), (42, 128)]:
        effective = {"generation": {"max_new_tokens": maximum}}
        report = {"partition": "test", "membership_hash": "same", "comparison_fingerprint": "same",
                  "seed": seed, "effective_evaluation": effective,
                  "evaluation_fingerprint": fingerprint(effective), "metrics": {"scores": {"bleu4": 0}}}
        path = tmp_path / f"{seed}.json"
        write_json(path, report)
        paths.append(str(path))
    with pytest.raises(ValueError, match="identical"):
        aggregate_runs(paths)


def test_gold_subset_is_paired(monkeypatch):
    monkeypatch.setattr(generation, "score_predictions",
                        lambda predictions, references, *args: {"count": len(predictions)})
    rows = [{"uid": str(i), "source_id": "source", "prediction": "generated",
             "references": ["transcript"], "interpreter_reference": "interpreter" if i < 3 else None,
             "truncated": False} for i in range(5)]
    scores = generation.score_rows(rows, Config())
    assert scores["count"] == 5
    left, right = scores["transcript_on_gold_subset"], scores["interpreter_diagnostic"]
    assert left["count"] == right["count"] == 3
    assert left["uid_hash"] == right["uid_hash"]
