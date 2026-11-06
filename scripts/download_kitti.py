"""Download selected KITTI odometry color frames without fetching the 69 GB ZIP."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import closing
import io
import json
import os
from pathlib import Path
import shutil
import struct
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import zipfile
import zlib


BASE = "https://s3.eu-central-1.amazonaws.com/avg-kitti"
COLOR_URL = f"{BASE}/data_odometry_color.zip"
CALIB_URL = f"{BASE}/data_odometry_calib.zip"
POSES_URL = f"{BASE}/data_odometry_poses.zip"
BLOCK_SIZE = 1024 * 1024


def request_bytes(url: str, start: int | None = None, stop: int | None = None) -> bytes:
    headers = {"User-Agent": "robust-visual-odometry/0.1"}
    if start is not None:
        headers["Range"] = f"bytes={start}-{stop}"
    for attempt in range(5):
        try:
            with urlopen(Request(url, headers=headers), timeout=90) as response:
                if start is not None:
                    expected = f"bytes {start}-{stop}/"
                    actual = response.headers.get("Content-Range", "")
                    if response.status != 206 or not actual.startswith(expected):
                        raise RuntimeError(f"HTTP range unsupported: {response.status} {actual}")
                return response.read()
        except (HTTPError, URLError, TimeoutError) as exc:
            if attempt == 4:
                raise RuntimeError(f"Download failed: {url}") from exc
            time.sleep(2**attempt)
    raise AssertionError("Unreachable")


class HTTPRangeFile(io.RawIOBase):
    """Seekable, block-cached view of an HTTP object with Range support."""

    def __init__(self, url: str, block_size: int = BLOCK_SIZE, cache_blocks: int = 8):
        self.url = url
        self.block_size = block_size
        self.cache_blocks = cache_blocks
        with urlopen(Request(url, method="HEAD"), timeout=30) as response:
            self.size = int(response.headers["Content-Length"])
        self.position = 0
        self.cache: OrderedDict[int, bytes] = OrderedDict()
        self.bytes_fetched = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_CUR:
            offset += self.position
        elif whence == os.SEEK_END:
            offset += self.size
        elif whence != os.SEEK_SET:
            raise ValueError(f"Invalid whence: {whence}")
        if offset < 0:
            raise ValueError("Negative seek")
        self.position = offset
        return offset

    def _block(self, index: int) -> bytes:
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        start = index * self.block_size
        stop = min(start + self.block_size, self.size) - 1
        data = request_bytes(self.url, start, stop)
        if len(data) != stop - start + 1:
            raise IOError(f"Short HTTP response at {start}")
        self.bytes_fetched += len(data)
        self.cache[index] = data
        if len(self.cache) > self.cache_blocks:
            self.cache.popitem(last=False)
        return data

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self.size - self.position
        size = min(size, self.size - self.position)
        parts = []
        while size > 0:
            index, offset = divmod(self.position, self.block_size)
            part = self._block(index)[offset : offset + size]
            if not part:
                raise IOError(f"Cannot read byte {self.position}")
            parts.append(part)
            self.position += len(part)
            size -= len(part)
        return b"".join(parts)


def archive_paths(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    result = {}
    for info in archive.infolist():
        if info.filename.startswith("dataset/") and not info.is_dir():
            result[info.filename.removeprefix("dataset/")] = info
    return result


def extract_one(archive: zipfile.ZipFile, info: zipfile.ZipInfo, destination: Path) -> bool:
    if destination.is_file() and destination.stat().st_size == info.file_size:
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    try:
        with archive.open(info) as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target)
        if temporary.stat().st_size != info.file_size:
            raise IOError(f"Size mismatch for {destination}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def extract_remote_member(url: str, archive_size: int, info: zipfile.ZipInfo,
                          destination: Path) -> bool:
    """Fetch one local ZIP record by range; validate its CRC before saving."""
    if destination.is_file() and destination.stat().st_size == info.file_size:
        return False
    start = info.header_offset
    stop = min(start + info.compress_size + 4095, archive_size - 1)
    record = request_bytes(url, start, stop)
    if len(record) < 30:
        raise IOError(f"Short ZIP record for {info.filename}")
    signature, _, flags, method, _, _, _, _, _, name_len, extra_len = struct.unpack_from(
        "<IHHHHHIIIHH", record
    )
    if signature != 0x04034B50 or flags & 1 or method != info.compress_type:
        raise ValueError(f"Unsupported ZIP record for {info.filename}")
    data_start = 30 + name_len + extra_len
    compressed = record[data_start : data_start + info.compress_size]
    if len(compressed) != info.compress_size:
        raise IOError(f"ZIP record exceeds range for {info.filename}")
    if method == zipfile.ZIP_DEFLATED:
        content = zlib.decompress(compressed, -15)
    elif method == zipfile.ZIP_STORED:
        content = compressed
    else:
        raise ValueError(f"Unsupported compression method {method}")
    if len(content) != info.file_size or zlib.crc32(content) != info.CRC:
        raise IOError(f"Size or CRC mismatch for {info.filename}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    try:
        temporary.write_bytes(content)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def small_archive(url: str, root: Path, needed: list[str]) -> None:
    print(f"Reading {url.rsplit('/', 1)[-1]}", flush=True)
    with zipfile.ZipFile(io.BytesIO(request_bytes(url))) as archive:
        entries = archive_paths(archive)
        for relative in needed:
            if relative not in entries:
                raise KeyError(f"Missing {relative} in {url}")
            extract_one(archive, entries[relative], root / relative)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/kitti"))
    parser.add_argument("--sequences", nargs="+", default=["00"])
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--stereo", action="store_true", help="Also download matching right-camera image_3 frames")
    args = parser.parse_args()
    if args.start < 0 or args.frames < 1:
        parser.error("--start must be >= 0 and --frames must be >= 1")
    sequences = [str(int(sequence)).zfill(2) for sequence in args.sequences]
    if any(int(sequence) > 10 for sequence in sequences):
        parser.error("Use sequences 00-10, which have ground truth poses")
    root = args.root
    small_archive(CALIB_URL, root, [f"sequences/{s}/calib.txt" for s in sequences])
    small_archive(POSES_URL, root, [f"poses/{s}.txt" for s in sequences])

    with closing(HTTPRangeFile(COLOR_URL)) as remote:
        print(f"Reading color ZIP directory ({remote.size / 1e9:.1f} GB remote)", flush=True)
        with zipfile.ZipFile(remote) as archive:
            entries = archive_paths(archive)
            for sequence in sequences:
                print(f"Sequence {sequence}: frames {args.start}-{args.start + args.frames - 1}", flush=True)
                jobs = []
                for frame in range(args.start, args.start + args.frames):
                    cameras = ("image_2", "image_3") if args.stereo else ("image_2",)
                    for camera in cameras:
                        relative = f"sequences/{sequence}/{camera}/{frame:06d}.png"
                        if relative not in entries:
                            raise KeyError(f"Missing {relative} in KITTI color archive")
                        jobs.append((entries[relative], root / relative))
                with ThreadPoolExecutor(max_workers=6) as pool:
                    futures = [pool.submit(extract_remote_member, COLOR_URL, remote.size, info, path)
                               for info, path in jobs]
                    for count, future in enumerate(as_completed(futures), 1):
                        future.result()
                        if count % 50 == 0 or count == len(jobs):
                            print(f"  {count}/{len(jobs)}", flush=True)

    inventory = {}
    for sequence_dir in sorted((root / "sequences").glob("[0-9][0-9]")):
        images = sorted((sequence_dir / "image_2").glob("*.png"))
        if images:
            inventory[sequence_dir.name] = {
                "frame_count": len(images),
                "first_frame": int(images[0].stem),
                "last_frame": int(images[-1].stem),
            }
    manifest = {
        "source": "KITTI odometry, official avg-kitti S3 archives",
        "urls": {"color": COLOR_URL, "calibration": CALIB_URL, "poses": POSES_URL},
        "camera": "image_2",
        "stereo": bool(args.stereo),
        "local_inventory": inventory,
        "last_request": {"sequences": sequences, "start_frame": args.start,
                         "frame_count": args.frames},
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Dataset ready at {root.resolve()}", flush=True)


if __name__ == "__main__":
    main()
