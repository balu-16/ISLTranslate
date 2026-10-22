from types import SimpleNamespace

import numpy as np

from isl_slt.features.statistics import apply_statistics, fit_statistics


def test_scaling_leaves_masks_and_invalid_values_unchanged():
    train = SimpleNamespace(values=np.array([[2., 1.], [4., 0.], [0., 1.]], dtype=np.float32),
        continuous_valid=np.array([[True, False], [True, False], [False, False]]),
        metadata={"continuous_channels": [0]})
    statistics = fit_statistics([train], "schema", "training-only")
    assert statistics["mean"] == [3., 0.]
    transformed = apply_statistics(train, statistics)
    np.testing.assert_array_equal(transformed[:, 1], train.values[:, 1])
    assert transformed[2, 0] == 0
    validation = SimpleNamespace(values=np.array([[1000000., 1.]]),
        continuous_valid=np.array([[True, False]]), metadata=train.metadata)
    apply_statistics(validation, statistics)
    assert statistics["mean"][0] == 3.
