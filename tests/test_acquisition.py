import io
import zipfile
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
import threading

import pytest

from isl_slt.data.acquisition import MultipartReader, extract_selected, pose_index
from isl_slt.data.acquisition import plan_selection
from isl_slt.data.archives import ArchivePart, RangeFile, StagedParts


@contextmanager
def range_server(payloads, honor_ranges=True):
    transferred = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_GET(self):
            data = payloads[int(self.path.strip("/"))]
            header = self.headers.get("Range")
            if header and honor_ranges:
                start, end = map(int, header.removeprefix("bytes=").split("-"))
                body = data[start:end + 1]
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
            else:
                body = data
                self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (ConnectionResetError, BrokenPipeError):
                pass
            transferred.append(len(body))
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", transferred
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_remote_ranges_transfer_only_index_and_selected_member(tmp_path, record_factory):
    memory = io.BytesIO()
    with zipfile.ZipFile(memory, "w") as archive:
        archive.writestr("poses/wanted-1.pose", b"selected" * 200)
        archive.writestr("poses/unused-1.pose", os.urandom(2 * 1024**2))
    data = memory.getvalue()
    payloads = [data[start:end] for start, end in zip([0, 100, 400, 1000000], [100, 400, 1000000, len(data)])]
    with range_server(payloads) as (url, transferred):
        parts = [ArchivePart(f"part-{i}", len(payload), {"size": len(payload), "id": i},
                            lambda i=i, size=len(payload): RangeFile(f"{url}/{i}", size))
                 for i, payload in enumerate(payloads)]
        records, plan = plan_selection(parts, [record_factory("wanted-1")])
        extract_selected(parts, records, tmp_path / "out", inspect_sample=False)
        assert plan["compressed_bytes"] == 1600
        assert plan["uncompressed_bytes"] == 1600
        assert sum(transferred) < plan["compressed_bytes"] + 150000
        assert (tmp_path / "out/wanted-1.pose").read_bytes() == b"selected" * 200


def test_ranges_refuse_full_body_and_staged_fallback_is_bounded(tmp_path):
    payloads = [b"a" * 100, b"b" * 200, b"c" * 300, b"d" * 400]
    with range_server(payloads, honor_ranges=False) as (url, _):
        with pytest.raises(OSError, match="byte ranges"):
            RangeFile(f"{url}/0", 100).read(1)
        parts = [ArchivePart(str(i), len(data), {"size": len(data)}, None) for i, data in enumerate(payloads)]
        with StagedParts(parts, [f"{url}/{i}" for i in range(4)], tmp_path / "staging") as staging:
            for i in [3, 0, 1, 2, 3]:
                assert staging.open(i).read() == payloads[i]
                assert len(staging.cached) <= 2
            assert staging.peak_bytes <= 700
        assert not list((tmp_path / "staging").glob("*.part"))


def test_pretraining_plan_excludes_all_target_sources_and_budgets_groups(tmp_path, record_factory):
    target = [record_factory("target-1", text="  SAME Text  ", split="train")]
    pool = [record_factory("target-2", text="unique"), record_factory("other-1", text="same text"),
            record_factory("allowed-1", text="first unique"), record_factory("allowed-2", text="second unique")]
    archive = tmp_path / "selection.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        for row in pool:
            stream.writestr(f"poses/{row.uid}.pose", b"x" * 10)
    rows, report = plan_selection([archive], pool, target_records=target, budget_bytes=20)
    assert {r.uid for r in rows} == {"allowed-1", "allowed-2"}
    assert report["compressed_bytes"] == 20
    assert {r["reason"] for r in report["excluded"]} == {"target_parent", "duplicate_normalized_text"}


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
