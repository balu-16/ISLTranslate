from isl_slt.training.checkpoint import guarded_improvement


def test_guard_uses_historical_chrf_maximum():
    assert guarded_improvement(11, 29.5, 10, 30)
    assert not guarded_improvement(11, 29.49, 10, 30)
    assert not guarded_improvement(10, 31, 10, 30)
    assert not guarded_improvement(float("nan"), 30, 10, 30)
