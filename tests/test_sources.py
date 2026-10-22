import pytest

from isl_slt.config import Adapter
from isl_slt.data.manifests import resolve_source


@pytest.mark.parametrize("uid,parent", [
    ("1782bea75c7d-1", "1782bea75c7d"), ("UbhONRGm-90--61", "UbhONRGm-90-"),
    ("0EaLX2t1wPU_e1", "0EaLX2t1wPU"), ("0EaLX2t1wPU_w", "0EaLX2t1wPU"),
    ("garbage", None),
])
def test_candidates_never_claim_verification(uid, parent):
    row = resolve_source(uid, {}, Adapter().candidate_uid_patterns)
    assert row["source_id"] == parent
    assert row["source_verified"] is False


def test_authoritative_source_keeps_provenance():
    override = {"source_id": "physical-parent", "source_verified": True, "source_provenance": "reviewed-recording"}
    assert resolve_source("UbhONRGm-90--61", {"UbhONRGm-90--61": override},
                          Adapter().candidate_uid_patterns)["source_id"] == "physical-parent"
