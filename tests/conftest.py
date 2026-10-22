"""Offline fixtures. No network or pretrained model assets are needed."""

import pytest

from isl_slt.data.manifests import Record


@pytest.fixture
def record_factory(tmp_path):
    def make(uid="sample-1", **overrides):
        values = dict(uid=uid, source_id=uid.rsplit("-", 1)[0], source_verified=True,
                      source_provenance="synthetic-test", dataset_name="synthetic",
                      dataset_revision="test-v1", text="a synthetic test sentence",
                      pose_path=str(tmp_path / f"{uid}.pose"))
        return Record(**(values | overrides))
    return make
