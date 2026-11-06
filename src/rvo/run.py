"""Run paired baseline and YOLO-filtered VO on configured KITTI windows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import tomllib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .evaluation import metrics
from .kitti import ground_truth_poses
from .vo import run_window


def save_poses(path: Path, poses: np.ndarray) -> None:
    np.savetxt(path, poses[:, :3, :4].reshape(len(poses), 12), fmt="%.9f")


def save_plot(path: Path, reference: np.ndarray, aligned: dict[str, np.ndarray],
              valid: dict[str, np.ndarray], title: str) -> None:
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.plot(reference[:, 0, 3], reference[:, 2, 3], label="Ground truth", color="black")
    colors = {"baseline": "#1f77b4", "filtered": "#d62728",
              "selective": "#2ca02c", "instance_mask": "#9467bd"}
    for mode, poses in aligned.items():
        axis.plot(poses[:, 0, 3], poses[:, 2, 3], label=mode.title(), color=colors[mode])
        failures = ~valid[mode]
        if np.any(failures):
            axis.scatter(poses[failures, 0, 3], poses[failures, 2, 3],
                         s=12, color=colors[mode], marker="x", alpha=0.6)
    axis.set(xlabel="X (m)", ylabel="Z (m)", title=title)
    axis.axis("equal")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/kitti_mvp.toml"))
    parser.add_argument("--output", type=Path, default=Path("outputs/kitti_mvp"))
    parser.add_argument("--sequences", nargs="+", help="Run only these configured sequences")
    parser.add_argument("--limit", type=int, help="Process at most this many frames per window")
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    root = Path(config["dataset"]["root"])
    for window in config["dataset"]["windows"]:
        sequence = window["sequence"]
        if args.sequences and sequence not in args.sequences:
            continue
        start = int(window["start_frame"])
        count = int(window["frame_count"])
        if args.limit is not None:
            count = min(count, args.limit)
        if count < 3:
            parser.error("A run needs at least three frames")
        output = args.output / f"seq{sequence}_{start:06d}_{start + count - 1:06d}"
        output.mkdir(parents=True, exist_ok=True)
        print(f"Running sequence {sequence}, frames {start}-{start + count - 1}", flush=True)
        records, poses, valid = run_window(root, sequence, start, count, config)
        reference = ground_truth_poses(root, sequence)[start : start + count]
        if len(reference) != count:
            raise ValueError(f"Missing ground truth for sequence {sequence}")
        summary = {"sequence": sequence, "start_frame": start, "frame_count": count,
                   "camera": config["dataset"]["camera"], "detector": config["filter"]["model"],
                   "selective_settings": config["selective"],
                   "diagnostic_settings": config["diagnostics"],
                   "mask_model": config["mask_filter"]["model"], "metrics": {}}
        aligned = {}
        common_valid = np.logical_and.reduce(list(valid.values()))
        summary["common_valid_poses"] = int(np.count_nonzero(common_valid))
        for mode in poses:
            mode_records = [row for row in records if row["mode"] == mode]
            result, aligned[mode] = metrics(poses[mode], reference, valid[mode],
                                            common_valid, mode_records)
            summary["metrics"][mode] = result
            save_poses(output / f"poses_{mode}.txt", poses[mode])
            save_poses(output / f"poses_{mode}_aligned.txt", aligned[mode])
            print(f"  {mode}: ATE {result['ate_rmse_m']:.2f} m; failures "
                  f"{result['pose_failures']}/{result['pose_attempts']}; "
                  f"mean {result['mean_total_ms']:.1f} ms/frame", flush=True)
        with (output / "per_frame.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        save_plot(output / "trajectory.png", reference, aligned, valid,
                  f"KITTI {sequence}, frames {start}-{start + count - 1}")
        print(f"  Saved {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
