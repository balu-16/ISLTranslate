from pathlib import Path

from isl_slt.data.context import SourceIndex
from isl_slt.features.cache import SignatureSession
from isl_slt.utils import io


def test_source_signatures_hash_each_asset_once(tmp_path, record_factory, monkeypatch):
    records = []
    for number in range(200):
        record = record_factory(f"parent-{number}", split="train")
        Path(record.pose_path).write_bytes(str(number).encode())
        record.asset_hashes = {"pose": io.file_hash(record.pose_path)}
        records.append(record)
    calls = []
    original = io.file_hash
    def counted(path):
        calls.append(str(path))
        return original(path)
    monkeypatch.setattr(io, "file_hash", counted)
    index, session = SourceIndex(records), SignatureSession()
    for record in records:
        for dependency in index.dependencies(record):
            session.signature(dependency)
    assert len(calls) == 200
