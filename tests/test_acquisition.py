import io
import zipfile

import pytest

from isl_slt.data.acquisition import MultipartReader, extract_selected, pose_index


def test_split_zip_selective_extraction(tmp_path, record_factory):
    memory = io.BytesIO()
    with zipfile.ZipFile(memory, "w") as archive:
        archive.writestr("poses/wanted-1.pose", b"synthetic-pose")
        archive.writestr("poses/other-1.pose", b"not-requested")
    payload = memory.getvalue()
    parts = [tmp_path / "aa", tmp_path / "ab"]
    parts[0].write_bytes(payload[:17])
    parts[1].write_bytes(payload[17:])
    report = extract_selected(parts, [record_factory("wanted-1")], tmp_path / "output", inspect_sample=False)
    assert len(report["files"]) == 1
    assert (tmp_path / "output/wanted-1.pose").read_bytes() == b"synthetic-pose"
    assert not (tmp_path / "output/other-1.pose").exists()
    with MultipartReader(parts) as joined:
        joined.seek(-10, 2)
        assert joined.read() == payload[-10:]


def test_archive_traversal_rejected():
    memory = io.BytesIO()
    with zipfile.ZipFile(memory, "w") as archive:
        archive.writestr("../outside.pose", b"bad")
    memory.seek(0)
    with zipfile.ZipFile(memory) as archive, pytest.raises(ValueError, match="Unsafe"):
        pose_index(archive)
