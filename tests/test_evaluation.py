from isl_slt.config import Config
from isl_slt.evaluation import generation


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
