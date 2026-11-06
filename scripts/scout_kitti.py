"""Fetch sparse candidate frames to choose a contiguous experiment window."""

from __future__ import annotations

import argparse
from contextlib import closing
from pathlib import Path
import zipfile

import cv2
import numpy as np

from download_kitti import COLOR_URL, HTTPRangeFile, archive_paths, extract_remote_member


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--frames", nargs="+", type=int, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/scout"))
    args = parser.parse_args()
    sequence = str(int(args.sequence)).zfill(2)
    if any(frame < 0 for frame in args.frames):
        parser.error("Frame indices must be nonnegative")
    tiles = []
    with closing(HTTPRangeFile(COLOR_URL)) as remote, zipfile.ZipFile(remote) as archive:
        entries = archive_paths(archive)
        for frame in args.frames:
            relative = f"sequences/{sequence}/image_2/{frame:06d}.png"
            if relative not in entries:
                raise KeyError(f"No {relative} in KITTI color archive")
            local = args.output / sequence / f"{frame:06d}.png"
            extract_remote_member(COLOR_URL, remote.size, entries[relative], local)
            image = cv2.imread(str(local))
            if image is None:
                raise ValueError(f"Cannot decode {local}")
            tile = cv2.resize(image, (480, 145), interpolation=cv2.INTER_AREA)
            cv2.rectangle(tile, (0, 0), (160, 28), (0, 0, 0), -1)
            cv2.putText(tile, f"{sequence} / {frame:06d}", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(tile)
    contact_sheet = args.output / f"sequence_{sequence}.jpg"
    if not cv2.imwrite(str(contact_sheet), np.hstack(tiles)):
        raise IOError(f"Cannot write {contact_sheet}")
    print(contact_sheet.resolve())


if __name__ == "__main__":
    main()
