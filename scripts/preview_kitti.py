"""Create a compact contact sheet for inspecting downloaded KITTI windows."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from rvo.kitti import image_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/kitti"))
    parser.add_argument("--sequences", nargs="+", default=["00", "07"])
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("outputs/kitti_preview.jpg"))
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be >= 1")
    rows = []
    for sequence in args.sequences:
        paths = image_paths(args.root, sequence)
        if not paths:
            raise FileNotFoundError(f"No frames for sequence {sequence}")
        selected = np.linspace(0, len(paths) - 1, args.samples, dtype=int)
        tiles = []
        for index in selected:
            path = paths[index]
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError(f"Cannot decode {path}")
            tile = cv2.resize(frame, (480, 145), interpolation=cv2.INTER_AREA)
            cv2.rectangle(tile, (0, 0), (170, 28), (0, 0, 0), -1)
            cv2.putText(tile, f"{sequence} / {path.stem}", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(tile)
        rows.append(np.hstack(tiles))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), np.vstack(rows)):
        raise IOError(f"Cannot write {args.output}")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
