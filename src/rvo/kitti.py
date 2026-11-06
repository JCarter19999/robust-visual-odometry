"""KITTI odometry training data layout and basic integrity checks."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def image_paths(root: Path, sequence: str, camera: str = "image_2") -> list[Path]:
    return sorted((root / "sequences" / sequence / camera).glob("*.png"))


def projection_matrix(root: Path, sequence: str, camera: str = "image_2") -> np.ndarray:
    key = {"image_0": "P0", "image_1": "P1", "image_2": "P2", "image_3": "P3"}[camera]
    path = root / "sequences" / sequence / "calib.txt"
    for line in path.read_text(encoding="ascii").splitlines():
        if line.startswith(key + ":"):
            values = np.fromstring(line.partition(":")[2], sep=" ")
            if values.size != 12:
                raise ValueError(f"Expected 12 numbers in {path} for {key}")
            return values.reshape(3, 4)
    raise ValueError(f"Missing {key} in {path}")


def ground_truth_poses(root: Path, sequence: str) -> np.ndarray:
    path = root / "poses" / f"{sequence}.txt"
    rows = np.loadtxt(path, dtype=np.float64)
    if rows.ndim == 1:
        rows = rows[None, :]
    if rows.shape[1] != 12:
        raise ValueError(f"Expected 12 values per pose in {path}")
    poses = np.tile(np.eye(4, dtype=np.float64), (len(rows), 1, 1))
    poses[:, :3, :4] = rows.reshape(-1, 3, 4)
    return poses


def inspect(root: Path, sequence: str, camera: str = "image_2") -> dict:
    paths = image_paths(root, sequence, camera)
    if not paths:
        raise FileNotFoundError(f"No images in {root / 'sequences' / sequence / camera}")
    indices = [int(path.stem) for path in paths]
    if indices != list(range(indices[0], indices[-1] + 1)):
        raise ValueError(f"Frame gap in sequence {sequence}")
    poses = ground_truth_poses(root, sequence)
    if indices[-1] >= len(poses):
        raise ValueError(f"Frame {indices[-1]} has no ground truth pose")
    image = cv2.imread(str(paths[0]), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode {paths[0]}")
    P = projection_matrix(root, sequence, camera)
    if not np.isfinite(P).all() or P[0, 0] <= 0 or P[1, 1] <= 0:
        raise ValueError(f"Invalid projection matrix for sequence {sequence}")
    return {
        "sequence": sequence,
        "camera": camera,
        "frames": len(paths),
        "first": indices[0],
        "last": indices[-1],
        "image_shape": tuple(image.shape),
        "ground_truth_poses": len(poses),
        "fx": float(P[0, 0]),
        "fy": float(P[1, 1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--sequences", nargs="+", default=["00", "07"])
    parser.add_argument("--camera", default="image_2")
    args = parser.parse_args()
    for sequence in args.sequences:
        print(inspect(args.root, sequence, args.camera))


if __name__ == "__main__":
    main()
