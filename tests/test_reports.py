from isl_slt.training.reports import summarize_run
from isl_slt.utils.io import append_event


def test_summary_uses_only_measured_values(tmp_path):
    append_event(tmp_path / "events.jsonl", kind="update", updated=True, examples=4, frames=40,
                 update_seconds=2, peak_gpu_memory_bytes=None)
    append_event(tmp_path / "events.jsonl", kind="epoch", train_seconds=3, validation_seconds=1)
    report = summarize_run(tmp_path)
    assert report["update_examples_per_second"] == 2
    assert report["validation_seconds_per_epoch"] == 1
    assert report["peak_gpu_memory_bytes"] is None
