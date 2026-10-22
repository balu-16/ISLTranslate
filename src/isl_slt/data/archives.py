"""Sized seekable archive parts, bounded HTTP ranges and an explicit two-part disk fallback."""

import bisect
from collections import OrderedDict
from dataclasses import dataclass
from collections.abc import Callable
import hashlib
import io
from pathlib import Path
import re
from urllib.request import HTTPRedirectHandler, Request, build_opener
from urllib.error import HTTPError

from isl_slt.utils.io import atomic_open


class SafeRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, url):
        redirected = super().redirect_request(request, fp, code, message, headers, url)
        if redirected is not None:
            # Hub/CDN redirects carry their own signed authorization. Never forward the token.
            from urllib.parse import urlsplit
            if urlsplit(request.full_url).netloc != urlsplit(url).netloc:
                redirected.remove_header("Authorization")
        return redirected


@dataclass(frozen=True)
class ArchivePart:
    name: str
    size: int
    identity: dict
    open: Callable[[], io.IOBase]


def local_part(path) -> ArchivePart:
    path = Path(path)
    status = path.stat()
    return ArchivePart(path.name, status.st_size,
        {"path": str(path.resolve()), "size": status.st_size, "mtime_ns": status.st_mtime_ns, "inode": status.st_ino},
        lambda: path.open("rb"))


class RangeFile(io.RawIOBase):
    """Read exactly the requested range; reject HTTP 200 before reading any full body."""
    def __init__(self, url, size, token=None, opener=None):
        super().__init__()
        self.url, self.size, self.position = url, size, 0
        self.token, self.opener = token, opener or build_opener(SafeRedirect())
        self.transferred = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset if whence == 2 else -1
        if position < 0:
            raise ValueError("Invalid archive seek")
        self.position = position
        return position

    def read(self, size=-1):
        size = max(0, self.size - self.position) if size is None or size < 0 else min(size, max(0, self.size - self.position))
        if not size:
            return b""
        start, end = self.position, self.position + size - 1
        headers = {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        try:
            response = self.opener.open(Request(self.url, headers=headers), timeout=60)
        except HTTPError as error:
            raise OSError(f"Remote archive range request failed with HTTP {error.code}; check access/revision") from None
        with response:
            if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{self.size}":
                raise OSError("Remote server did not honor exact byte ranges; explicitly choose --transport staged")
            data = response.read(size + 1)
        if len(data) != size:
            raise OSError("Remote archive range changed or was truncated")
        self.position += size
        self.transferred += size
        return data


class StagedParts:
    """LRU staging: evict before download, so temporary bytes count toward the two-part limit."""
    def __init__(self, parts, urls, directory, token=None):
        self.parts, self.urls, self.directory = parts, urls, Path(directory)
        self.token, self.cached, self.peak_bytes = token, OrderedDict(), 0
        self.opener = build_opener(SafeRedirect())
        self.lock = None

    def __enter__(self):
        import fcntl
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = (self.directory / ".staging.lock").open("a+b")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.lock.close()
            raise ValueError("Staging directory is already in use") from None
        # Remove only names owned by this transport after taking exclusive ownership.
        for path in self.directory.iterdir():
            if re.fullmatch(r"(?:[0-9a-f]{64}\.part|\.[0-9a-f]{64}\.part\..+)", path.name) and path.is_file():
                path.unlink()
        return self

    def ensure(self, number):
        part = self.parts[number]
        if number in self.cached:
            self.cached.move_to_end(number)
            return self.cached[number]
        while len(self.cached) >= 2:
            _, old = self.cached.popitem(last=False)
            old.unlink()
        # Include identity in the name; never reuse another revision's partial download.
        name = hashlib.sha256(repr(part.identity).encode()).hexdigest()
        path = self.directory / f"{name}.part"
        if path.exists():
            path.unlink()
        headers = {"Accept-Encoding": "identity"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        count, digest = 0, hashlib.sha256()
        try:
            response = self.opener.open(Request(self.urls[number], headers=headers), timeout=60)
        except HTTPError as error:
            raise OSError(f"Staged archive request failed with HTTP {error.code}; check access/revision") from None
        with response:
            with atomic_open(path) as stream:
                while block := response.read(min(1024 * 1024, part.size - count + 1)):
                    count += len(block)
                    if count > part.size:
                        raise OSError("Staged part exceeds declared size")
                    stream.write(block)
                    digest.update(block)
                    self.peak_bytes = max(self.peak_bytes, count + sum(p.stat().st_size for p in self.cached.values()))
                if count != part.size or part.identity.get("sha256") not in (None, digest.hexdigest()):
                    raise OSError("Staged archive size/checksum mismatch")
        self.cached[number] = path
        return path

    def open(self, number):
        owner = self
        class StagedFile(RangeFile):
            def __init__(self):
                io.RawIOBase.__init__(self)
                self.size, self.position = owner.parts[number].size, 0
            def read(self, size=-1):
                with owner.ensure(number).open("rb") as stream:
                    stream.seek(self.position)
                    data = stream.read(size)
                self.position += len(data)
                return data
        return StagedFile()

    def __exit__(self, *args):
        for path in self.cached.values():
            path.unlink(missing_ok=True)
        self.cached.clear()
        if self.lock:
            self.lock.close()


class MultipartReader(io.RawIOBase):
    """Concatenate local paths or ArchiveParts (including HfFileSystem-compatible streams)."""
    def __init__(self, parts):
        super().__init__()
        self.parts = [part if isinstance(part, ArchivePart) else local_part(part) for part in parts]
        suffixes = [re.search(r"iSign-poses_v1\.1_part_([a-z]{2})$", part.name) for part in self.parts]
        if any(suffixes) and [match.group(1) if match else None for match in suffixes] != ["aa", "ab", "ac", "ad"]:
            raise ValueError("Supply all four iSign pose ZIP parts in aa, ab, ac, ad order")
        if not self.parts or any(part.size <= 0 for part in self.parts):
            raise ValueError("Supply nonempty sized archive parts")
        self.streams, self.offsets, self.position = {}, [0], 0
        for part in self.parts:
            self.offsets.append(self.offsets[-1] + part.size)

    @property
    def identity(self):
        return [part.identity for part in self.parts]

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
        remaining = max(0, self.offsets[-1] - self.position)
        size = remaining if size is None or size < 0 else min(size, remaining)
        chunks = []
        while size > 0:
            number = bisect.bisect_right(self.offsets, self.position) - 1
            count = min(size, self.offsets[number + 1] - self.position)
            if number not in self.streams:
                self.streams[number] = self.parts[number].open()
            stream = self.streams[number]
            stream.seek(self.position - self.offsets[number])
            block = stream.read(count)
            if len(block) != count:
                raise OSError("ZIP part changed or was truncated during extraction")
            chunks.append(block)
            self.position += count
            size -= count
        return b"".join(chunks)

    def close(self):
        for stream in getattr(self, "streams", {}).values():
            stream.close()
        super().close()
