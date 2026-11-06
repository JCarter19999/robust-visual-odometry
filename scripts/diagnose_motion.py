"""Inspect feature flow and ground truth displacement at selected VO frames."""

from __future__ import annotations

import argparse
from pathlib import Path
import tomllib

import cv2
import numpy as np
from ultralytics import YOLO

from rvo.kitti import ground_truth_poses
from rvo.vo import detect_and_track, detect_boxes, outside_boxes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/kitti_mvp.toml"))
    parser.add_argument("--sequence", default="07")
    parser.add_argument("--frames", nargs="+", type=int,
                        default=[642, 650, 660, 680, 700, 720, 740, 800])
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    root = Path(config["dataset"]["root"])
    image_dir = root / "sequences" / args.sequence / config["dataset"]["camera"]
    model = YOLO(config["filter"]["model"])
    classes = set(config["filter"]["classes"])
    confidence = float(config["filter"]["confidence"])
    padding = int(config["filter"]["box_padding_px"])
    ground_truth = ground_truth_poses(root, args.sequence)
    print("frame kept median_flow_px ground_truth_step_m")
    for frame in args.frames:
        previous = cv2.imread(str(image_dir / f"{frame - 1:06d}.png"))
        current = cv2.imread(str(image_dir / f"{frame:06d}.png"))
        if previous is None or current is None:
            raise FileNotFoundError(f"Missing frame pair ending at {frame}")
        before, after, _ = detect_and_track(
            cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(current, cv2.COLOR_BGR2GRAY), config["features"]
        )
        previous_boxes, _ = detect_boxes(model, previous, classes, confidence)
        current_boxes, _ = detect_boxes(model, current, classes, confidence)
        keep = outside_boxes(before, previous_boxes, padding)
        keep &= outside_boxes(after, current_boxes, padding)
        flow = np.linalg.norm(after[keep] - before[keep], axis=1)
        median_flow = float(np.median(flow)) if len(flow) else float("nan")
        gt_step = np.linalg.norm(ground_truth[frame, :3, 3] - ground_truth[frame - 1, :3, 3])
        print(f"{frame} {int(keep.sum())} {median_flow:.3f} {gt_step:.3f}")


if __name__ == "__main__":
    main()
