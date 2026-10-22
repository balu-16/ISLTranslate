"""Explicit acquisition and selective extraction from one ZIP split into ordered byte parts."""

import argparse
import bisect
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import zipfile

from isl_slt.config import Features, load_config
from isl_slt.data.manifests import Record, load_manifest
from isl_slt.data.pose import decode_pose
from isl_slt.features.pose import selected_joints
from isl_slt.utils.io import atomic_open, file_hash, write_json


class MultipartReader(io.RawIOBase):
    """A seekable concatenation; zipfile can read the central directory across part boundaries."""
    def __init__(self, parts):
        super().__init__()
        self.streams, self.offsets, self.position = [], [0], 0
        try:
            for path in parts:
                stream = Path(path).open("rb")
                self.streams.append(stream)
                self.offsets.append(self.offsets[-1] + Path(path).stat().st_size)
        except BaseException:
            self.close()
            raise
        if not self.streams:
            raise ValueError("Supply ordered ZIP byte parts")

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset if whence == 0 else self.position + offset if whence == 1 else self.offsets[-1] + offset if whence == 2 else -1
        if position < 0:
            raise ValueError("Invalid seek")
        self.position = position
        return position

    def read(self, size=-1):
        remaining = self.offsets[-1] - self.position
        size = remaining if size is None or size < 0 else min(size, remaining)
        chunks = []
        while size > 0:
            index = bisect.bisect_right(self.offsets, self.position) - 1
            count = min(size, self.offsets[index + 1] - self.position)
            stream = self.streams[index]
            stream.seek(self.position - self.offsets[index])
            block = stream.read(count)
            if len(block) != count:
                raise OSError("ZIP part changed or was truncated during extraction")
            chunks.append(block)
            self.position += count
            size -= count
        return b"".join(chunks)

    def close(self):
        for stream in getattr(self, "streams", []):
            stream.close()
        super().close()


def pose_index(archive: zipfile.ZipFile) -> dict:
    index = {}
    for entry in archive.infolist():
        name = PurePosixPath(entry.filename)
        if name.is_absolute() or ".." in name.parts or "\\" in entry.filename or "\x00" in entry.filename:
            raise ValueError("Unsafe archive member path")
        if stat.S_ISLNK(entry.external_attr >> 16):
            raise ValueError("Archive symlinks are not accepted")
        if entry.is_dir() or name.suffix.lower() != ".pose":
            continue
        uid = name.stem
        if uid in index:
            raise ValueError(f"Ambiguous pose UID in archive: {uid}")
        index[uid] = {"member": entry.filename, "bytes": entry.file_size, "crc32": entry.CRC}
    return index


def inspect_pose_bytes(data: bytes, settings: Features | None = None) -> dict:
    from pose_format import Pose
    settings = settings or Features()
    pose = Pose.read(data)
    record = Record(uid="inspection", dataset_name="inspection", dataset_revision="provided-bytes",
                    text="", pose_path="provided.pose")
    sequence = decode_pose(pose, record, settings)
    required = selected_joints(settings)
    missing = [list(name) for name in required if name not in sequence.names]
    if missing:
        raise ValueError(f"Required pose landmarks are absent: {missing}")
    return {"frames": len(sequence.times), "fps": sequence.metadata["fps"],
            "width": sequence.width, "height": sequence.height,
            "components": sequence.metadata["components"], "points": [list(name) for name in sequence.names]}


def extract_selected(parts, records, output, *, settings=None, max_bytes=50 * 1024**3,
                     inspect_sample=True) -> dict:
    records = list(records)
    if not records or len({row.uid for row in records}) != len(records):
        raise ValueError("Extraction needs nonempty unique manifest UIDs")
    output = Path(output)
    with MultipartReader(parts) as combined, zipfile.ZipFile(combined) as archive:
        index = pose_index(archive)
        missing = [row.uid for row in records if row.uid not in index]
        if missing:
            raise ValueError(f"Manifest UIDs missing from archive: {missing}")
        selected = {row.uid: index[row.uid] for row in records}
        if sum(row["bytes"] for row in selected.values()) > max_bytes:
            raise ValueError("Selected extraction exceeds max_bytes")
        if any((output / f"{row.uid}.pose").exists() for row in records):
            raise ValueError("Extraction refuses to overwrite existing pose files")
        first = selected[records[0].uid]
        if first["bytes"] > 512 * 1024**2:
            raise ValueError("Sample pose exceeds inspection memory bound")
        sample = inspect_pose_bytes(archive.read(first["member"]), settings) if inspect_sample else None
        written = []
        for record in records:
            target = output / f"{record.uid}.pose"
            with archive.open(selected[record.uid]["member"]) as source, atomic_open(target) as stream:
                shutil.copyfileobj(source, stream, length=1024 * 1024)
            written.append({"uid": record.uid, "member": selected[record.uid]["member"],
                            "path": str(target), "sha256": file_hash(target)})
    report = {"parts": [str(path) for path in parts], "sample": sample, "files": written,
              "archive_index": index, "layout": "flattened uid.pose; raw annotations unchanged"}
    write_json(output / "extraction.json", report)
    return report


def fetch_main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="Exploration-Lab/iSign")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--out", default="data/raw/isign-acquisition")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--csv", action="store_true")
    parser.add_argument("--parts", action="store_true")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--archives", nargs="+", help="Ordered local aa, ab, ac, ad byte parts")
    parser.add_argument("--manifest")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--max-bytes", type=int, default=50 * 1024**3)
    args = parser.parse_args(argv)
    if args.archives:
        if not args.manifest:
            parser.error("--archives requires --manifest")
        report = extract_selected(args.archives, load_manifest(args.manifest), args.out,
                                  settings=load_config(args.config).features, max_bytes=args.max_bytes)
        print(json.dumps({"count": len(report["files"]), "report": str(Path(args.out) / "extraction.json")}))
        return
    if not args.allow_network or not (args.list or args.csv or args.parts):
        parser.error("Remote operations require --allow-network and an explicit --list, --csv or --parts")
    from huggingface_hub import HfApi, hf_hub_download
    token = os.environ.get("HF_TOKEN")
    info = HfApi(token=token).dataset_info(args.repo, revision=args.revision, files_metadata=True)
    listing = [{"name": row.rfilename, "bytes": row.size} for row in info.siblings]
    expected = [f"iSign-poses_v1.1_part_{suffix}" for suffix in ("aa", "ab", "ac", "ad")]
    names = {row["name"] for row in listing}
    if args.parts and not set(expected).issubset(names):
        raise ValueError("Expected ordered iSign ZIP parts are absent; inspect the repository listing")
    downloaded = []
    for row in listing:
        if (args.csv and row["name"].endswith(".csv")) or (args.parts and row["name"] in expected):
            downloaded.append(hf_hub_download(args.repo, row["name"], repo_type="dataset",
                revision=info.sha, token=token, local_dir=args.out))
    report = {"repository": args.repo, "resolved_revision": info.sha, "files": listing,
              "downloaded": downloaded, "pose_part_order": expected}
    write_json(Path(args.out) / "acquisition.json", report)
    print(json.dumps(report, indent=2))


def inspect_main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("pose")
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args(argv)
    print(json.dumps(inspect_pose_bytes(Path(args.pose).read_bytes(), load_config(args.config).features), indent=2))
