"""Paired feature-based monocular VO with optional object-box filtering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from ultralytics import YOLO

from .kitti import projection_matrix


@dataclass
class Step:
    pose: np.ndarray
    valid: bool
    submitted: int
    ransac_inliers: int
    pose_inliers: int
    geometry_ms: float


def detect_and_track(previous: np.ndarray, current: np.ndarray, settings: dict) -> tuple[np.ndarray, np.ndarray, int]:
    corners = cv2.goodFeaturesToTrack(
        previous,
        maxCorners=int(settings["max_corners"]),
        qualityLevel=float(settings["quality_level"]),
        minDistance=float(settings["min_distance"]),
    )
    if corners is None:
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, empty, 0
    tracked, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, corners, None)
    if tracked is None or status is None:
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, empty, len(corners)
    before = corners.reshape(-1, 2)
    after = tracked.reshape(-1, 2)
    height, width = current.shape
    valid = (status.reshape(-1) == 1) & np.isfinite(after).all(axis=1)
    valid &= (after[:, 0] >= 0) & (after[:, 0] < width)
    valid &= (after[:, 1] >= 0) & (after[:, 1] < height)
    return before[valid], after[valid], len(corners)


def detect_boxes(model: YOLO, frame: np.ndarray, classes: set[str], confidence: float) -> tuple[np.ndarray, float]:
    start = perf_counter()
    result = model.predict(frame, device=0, conf=confidence, verbose=False)[0]
    elapsed_ms = (perf_counter() - start) * 1000
    boxes = result.boxes
    if len(boxes) == 0:
        return np.empty((0, 4), dtype=np.float32), elapsed_ms
    category = boxes.cls.cpu().numpy().astype(int)
    selected = [index for index, class_id in enumerate(category)
                if result.names[class_id] in classes]
    return boxes.xyxy.cpu().numpy()[selected], elapsed_ms


def outside_boxes(points: np.ndarray, boxes: np.ndarray, padding: int) -> np.ndarray:
    keep = np.ones(len(points), dtype=bool)
    for left, top, right, bottom in boxes:
        inside = ((points[:, 0] >= left - padding) & (points[:, 0] <= right + padding)
                  & (points[:, 1] >= top - padding) & (points[:, 1] <= bottom + padding))
        keep &= ~inside
    return keep


def selective_keep(before: np.ndarray, after: np.ndarray, previous_boxes: np.ndarray,
                   current_boxes: np.ndarray, padding: int, image_shape: tuple[int, int],
                   settings: dict) -> tuple[np.ndarray, int, bool]:
    """Reject a detected box only when its median flow differs from background."""
    hard_keep = outside_boxes(before, previous_boxes, padding)
    hard_keep &= outside_boxes(after, current_boxes, padding)
    flow = after - before
    background_points = flow[hard_keep] if np.count_nonzero(hard_keep) >= 50 else flow
    if len(background_points) == 0:
        return np.ones(len(before), dtype=bool), 0, True
    background_flow = np.median(background_points, axis=0)
    keep = np.ones(len(before), dtype=bool)
    active_boxes = 0
    for points, boxes in ((before, previous_boxes), (after, current_boxes)):
        for left, top, right, bottom in boxes:
            inside = ((points[:, 0] >= left - padding) & (points[:, 0] <= right + padding)
                      & (points[:, 1] >= top - padding) & (points[:, 1] <= bottom + padding))
            if np.count_nonzero(inside) < int(settings["min_box_tracks"]):
                continue
            difference = float(np.linalg.norm(np.median(flow[inside], axis=0) - background_flow))
            if difference >= float(settings["flow_difference_px"]):
                keep &= ~inside
                active_boxes += 1
    height, width = image_shape
    selected = before[keep]
    if len(selected):
        columns = np.clip((selected[:, 0] * int(settings["grid_cols"]) / width).astype(int),
                          0, int(settings["grid_cols"]) - 1)
        rows = np.clip((selected[:, 1] * int(settings["grid_rows"]) / height).astype(int),
                       0, int(settings["grid_rows"]) - 1)
        occupied = len(np.unique(rows * int(settings["grid_cols"]) + columns))
    else:
        occupied = 0
    fallback = len(selected) < int(settings["min_features"]) or occupied < int(settings["min_occupied_cells"])
    if fallback:
        keep[:] = True
    return keep, active_boxes, fallback


def estimate_step(before: np.ndarray, after: np.ndarray, K: np.ndarray,
                  prior_pose: np.ndarray, settings: dict) -> Step:
    start = perf_counter()
    submitted = len(before)
    minimum = int(settings["min_correspondences"])
    if submitted < minimum:
        return Step(prior_pose.copy(), False, submitted, 0, 0,
                    (perf_counter() - start) * 1000)
    cv2.setRNGSeed(0)
    essential, inlier_mask = cv2.findEssentialMat(
        before, after, K, method=cv2.RANSAC, prob=0.999,
        threshold=float(settings["ransac_threshold_px"]),
    )
    if essential is None or inlier_mask is None or essential.shape != (3, 3):
        return Step(prior_pose.copy(), False, submitted, 0, 0,
                    (perf_counter() - start) * 1000)
    ransac_inliers = int(np.count_nonzero(inlier_mask))
    if ransac_inliers < minimum:
        return Step(prior_pose.copy(), False, submitted, ransac_inliers, 0,
                    (perf_counter() - start) * 1000)
    pose_inliers, R, direction, _ = cv2.recoverPose(
        essential, before, after, K, mask=inlier_mask.copy()
    )
    if pose_inliers < minimum or not np.isfinite(R).all() or not np.isfinite(direction).all():
        return Step(prior_pose.copy(), False, submitted, ransac_inliers, int(pose_inliers),
                    (perf_counter() - start) * 1000)
    # recoverPose maps camera t to camera t+1. Store camera-to-world poses.
    relative_inverse = np.eye(4, dtype=np.float64)
    relative_inverse[:3, :3] = R.T
    relative_inverse[:3, 3] = -(R.T @ direction).reshape(3)
    pose = prior_pose @ relative_inverse
    return Step(pose, True, submitted, ransac_inliers, int(pose_inliers),
                (perf_counter() - start) * 1000)


def run_window(root: Path, sequence: str, start_frame: int, frame_count: int,
               config: dict) -> tuple[list[dict], dict[str, np.ndarray], dict[str, np.ndarray]]:
    if frame_count < 2:
        raise ValueError("A VO window needs at least two frames")
    dataset = config["dataset"]
    if dataset["camera"] != "image_2":
        raise ValueError("The paired runner currently uses KITTI image_2")
    K = projection_matrix(root, sequence, dataset["camera"])[:, :3]
    image_dir = root / "sequences" / sequence / dataset["camera"]
    paths = [image_dir / f"{frame:06d}.png"
             for frame in range(start_frame, start_frame + frame_count)]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} frames; first: {missing[0]}")
    model_path = Path(config["filter"]["model"])
    if not model_path.is_file():
        raise FileNotFoundError(f"Detector weights missing: {model_path}; run scripts/prepare_model.py")
    model = YOLO(str(model_path))
    classes = set(config["filter"]["classes"])
    confidence = float(config["filter"]["confidence"])
    padding = int(config["filter"]["box_padding_px"])

    previous_color = cv2.imread(str(paths[0]), cv2.IMREAD_COLOR)
    if previous_color is None:
        raise ValueError(f"Cannot read {paths[0]}")
    previous_gray = cv2.cvtColor(previous_color, cv2.COLOR_BGR2GRAY)
    # Exclude model loading and the first CUDA warmup from measured frame times.
    model.predict(previous_color, device=0, conf=confidence, verbose=False)
    previous_boxes, previous_detection_ms = detect_boxes(model, previous_color, classes, confidence)
    poses = {mode: [np.eye(4, dtype=np.float64)]
             for mode in ("baseline", "filtered", "selective")}
    valid = {mode: [True] for mode in poses}
    records: list[dict] = []

    for pair_number, path in enumerate(paths[1:], 1):
        read_start = perf_counter()
        current_color = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if current_color is None:
            raise ValueError(f"Cannot read {path}")
        current_gray = cv2.cvtColor(current_color, cv2.COLOR_BGR2GRAY)
        read_ms = (perf_counter() - read_start) * 1000

        tracking_start = perf_counter()
        before, after, corner_count = detect_and_track(previous_gray, current_gray,
                                                       config["features"])
        tracking_ms = (perf_counter() - tracking_start) * 1000
        current_boxes, current_detection_ms = detect_boxes(model, current_color,
                                                           classes, confidence)
        detection_ms = current_detection_ms + (previous_detection_ms if pair_number == 1 else 0.0)
        filtering_start = perf_counter()
        keep = outside_boxes(before, previous_boxes, padding)
        keep &= outside_boxes(after, current_boxes, padding)
        filtering_ms = (perf_counter() - filtering_start) * 1000
        filtered_count = int(np.count_nonzero(~keep))
        selective_start = perf_counter()
        selective_mask, active_boxes, coverage_fallback = selective_keep(
            before, after, previous_boxes, current_boxes, padding, current_gray.shape,
            config["selective"]
        )
        selective_ms = (perf_counter() - selective_start) * 1000
        selected_flow = np.linalg.norm(after[selective_mask] - before[selective_mask], axis=1)
        median_selected_flow = float(np.median(selected_flow)) if len(selected_flow) else float("inf")

        for mode in poses:
            mask = {"baseline": None, "filtered": keep, "selective": selective_mask}[mode]
            points_before, points_after = (before, after) if mask is None else (before[mask], after[mask])
            step = estimate_step(points_before, points_after, K, poses[mode][-1], config["geometry"])
            inlier_ratio = step.ransac_inliers / step.submitted if step.submitted else 0.0
            low_motion_degeneracy = (
                mode == "selective" and not step.valid
                and inlier_ratio >= float(config["diagnostics"]["min_ransac_inlier_ratio"])
                and median_selected_flow <= float(config["diagnostics"]["low_flow_px"])
            )
            poses[mode].append(step.pose)
            valid[mode].append(step.valid)
            total_ms = read_ms + tracking_ms + step.geometry_ms
            if mode == "filtered":
                total_ms += detection_ms + filtering_ms
            elif mode == "selective":
                total_ms += detection_ms + selective_ms
            records.append({
                "sequence": sequence, "frame": int(path.stem), "mode": mode,
                "corners": corner_count, "tracked": len(before),
                "removed": 0 if mask is None else int(np.count_nonzero(~mask)),
                "active_boxes": active_boxes if mode == "selective" else 0,
                "coverage_fallback": int(coverage_fallback) if mode == "selective" else 0,
                "median_selected_flow_px": median_selected_flow if mode == "selective" else 0.0,
                "low_motion_degeneracy": int(low_motion_degeneracy),
                "submitted": step.submitted, "ransac_inliers": step.ransac_inliers,
                "pose_inliers": step.pose_inliers, "valid": int(step.valid),
                "read_ms": read_ms, "tracking_ms": tracking_ms,
                "detection_ms": detection_ms if mode != "baseline" else 0.0,
                "filtering_ms": (filtering_ms if mode == "filtered" else
                                 selective_ms if mode == "selective" else 0.0),
                "geometry_ms": step.geometry_ms, "total_ms": total_ms,
            })
        previous_gray, previous_boxes = current_gray, current_boxes

    return records, {key: np.stack(value) for key, value in poses.items()}, {
        key: np.asarray(value, dtype=bool) for key, value in valid.items()
    }
