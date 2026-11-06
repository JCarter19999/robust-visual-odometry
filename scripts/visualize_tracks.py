"""Draw retained and removed VO tracks for selected KITTI frame pairs."""

from __future__ import annotations

import argparse
from pathlib import Path
import tomllib

import cv2
from ultralytics import YOLO

from rvo.vo import detect_and_track, detect_boxes, outside_boxes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/kitti_mvp.toml"))
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--frames", nargs="+", type=int, required=True,
                        help="Current frame of each pair; previous frame is current minus one")
    parser.add_argument("--output", type=Path, default=Path("outputs/diagnostics"))
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    root = Path(config["dataset"]["root"])
    image_dir = root / "sequences" / args.sequence / config["dataset"]["camera"]
    model = YOLO(config["filter"]["model"])
    classes = set(config["filter"]["classes"])
    confidence = float(config["filter"]["confidence"])
    padding = int(config["filter"]["box_padding_px"])
    args.output.mkdir(parents=True, exist_ok=True)
    for frame in args.frames:
        if frame < 1:
            parser.error("Frames must be >= 1")
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
        rendered = current.copy()
        for left, top, right, bottom in current_boxes.astype(int):
            cv2.rectangle(rendered, (left, top), (right, bottom), (0, 230, 255), 2)
        for index in range(0, len(after), 4):
            point = tuple(after[index].astype(int))
            color = (50, 210, 50) if keep[index] else (40, 40, 240)
            cv2.circle(rendered, point, 2, color, -1)
        label = (f"KITTI {args.sequence}/{frame:06d}: {int(keep.sum())} kept, "
                 f"{int((~keep).sum())} removed; green=kept red=removed yellow=boxes")
        cv2.rectangle(rendered, (0, 0), (min(rendered.shape[1], 950), 30), (0, 0, 0), -1)
        cv2.putText(rendered, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
        path = args.output / f"seq{args.sequence}_{frame:06d}.jpg"
        if not cv2.imwrite(str(path), rendered):
            raise IOError(f"Cannot write {path}")
        print(path.resolve())


if __name__ == "__main__":
    main()
