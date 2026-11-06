"""Object motion checks built on tracked image features."""

from __future__ import annotations

import cv2
import numpy as np


def _inside(points: np.ndarray, boxes: np.ndarray, padding: int = 0) -> np.ndarray:
    selected = np.zeros(len(points), dtype=bool)
    for x0, y0, x1, y1 in boxes:
        selected |= ((points[:, 0] >= x0 - padding) & (points[:, 0] <= x1 + padding)
                     & (points[:, 1] >= y0 - padding) & (points[:, 1] <= y1 + padding))
    return selected


def temporal_motion_keep(before: np.ndarray, after: np.ndarray, old_boxes: np.ndarray,
                         new_boxes: np.ndarray, state: dict, settings: dict,
                         padding: int) -> tuple[np.ndarray, dict, int]:
    """Reject object tracks only after motion relative to the background persists."""
    keep = np.ones(len(before), dtype=bool)
    flow = after - before
    background = _inside(before, old_boxes, padding) | _inside(after, new_boxes, padding)
    bg_flow = np.median(flow[~background], axis=0) if np.any(~background) else np.median(flow, axis=0) if len(flow) else np.zeros(2)
    prior = state.get("objects", {})
    next_objects: dict[int, dict] = {}
    used: set[int] = set()
    threshold = float(settings["flow_difference_px"])
    needed = int(settings["consecutive_frames"])
    active = 0
    for j, box in enumerate(new_boxes):
        best_id, best_iou = None, 0.0
        for key, record in prior.items():
            if int(key) in used:
                continue
            old = np.asarray(record["box"])
            ix0, iy0 = np.maximum(old[:2], box[:2]); ix1, iy1 = np.minimum(old[2:], box[2:])
            intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
            union = np.prod(np.maximum(0, old[2:] - old[:2])) + np.prod(np.maximum(0, box[2:] - box[:2])) - intersection
            iou = intersection / union if union else 0.0
            if iou > best_iou:
                best_iou, best_id = iou, int(key)
        old_box = None
        if best_id is None or best_iou < float(settings["min_iou"]):
            best_id = int(state.get("next_id", 0)); state["next_id"] = best_id + 1
            streak = 0
        else:
            used.add(best_id)
            streak = int(prior[best_id]["streak"])
            old_box = np.asarray(prior[best_id]["box"])
        points = _inside(before, old_box[None, :], padding) if old_box is not None else np.zeros(len(before), bool)
        points |= _inside(after, box[None, :], padding)
        if np.count_nonzero(points) >= int(settings["min_tracks"]):
            object_flow = np.median(flow[points], axis=0)
            moving = np.linalg.norm(object_flow - bg_flow) >= threshold
            streak = streak + 1 if moving else 0
        else:
            streak = 0
        next_objects[best_id] = {"box": box.tolist(), "streak": streak}
        if streak >= needed:
            keep &= ~points
            active += 1
    state["objects"] = next_objects
    return keep, state, active


def geometric_residual_keep(before: np.ndarray, after: np.ndarray, boxes0: np.ndarray,
                            boxes1: np.ndarray, K: np.ndarray, settings: dict,
                            padding: int) -> tuple[np.ndarray, int]:
    """Use robust background epipolar geometry to reject inconsistent object tracks."""
    keep = np.ones(len(before), dtype=bool)
    if len(before) < 8:
        return keep, 0
    cv2.setRNGSeed(0)
    E, inliers = cv2.findEssentialMat(before, after, K, method=cv2.RANSAC,
                                      prob=0.999, threshold=float(settings["ransac_threshold_px"]))
    if E is None or E.shape != (3, 3) or inliers is None:
        return keep, 0
    F = np.linalg.inv(K).T @ E @ np.linalg.inv(K)
    p0 = np.c_[before, np.ones(len(before))]
    p1 = np.c_[after, np.ones(len(after))]
    lines1 = (F @ p0.T).T
    lines0 = (F.T @ p1.T).T
    numerator = np.sum(p1 * lines1, axis=1) ** 2
    denom = lines1[:, 0]**2 + lines1[:, 1]**2 + lines0[:, 0]**2 + lines0[:, 1]**2
    residual = np.sqrt(numerator / np.maximum(denom, 1e-12))
    object_points = _inside(before, boxes0, padding) | _inside(after, boxes1, padding)
    reject = object_points & (residual > float(settings["max_epipolar_px"]))
    keep[reject] = False
    minimum = int(settings["min_features"])
    if np.count_nonzero(keep) < minimum:
        keep[:] = True
    return keep, int(np.count_nonzero(reject))


def disparity_map(left: np.ndarray, right: np.ndarray, settings: dict) -> np.ndarray:
    matcher = cv2.StereoSGBM_create(minDisparity=0, numDisparities=int(settings["num_disparities"]),
                                    blockSize=int(settings["block_size"]), P1=8*3*int(settings["block_size"])**2,
                                    P2=32*3*int(settings["block_size"])**2,
                                    uniquenessRatio=8, speckleWindowSize=80, speckleRange=2)
    return matcher.compute(left, right).astype(np.float32) / 16.0


def stereo_residual_keep(before: np.ndarray, after: np.ndarray, boxes0: np.ndarray,
                         boxes1: np.ndarray, disparity: np.ndarray, K: np.ndarray,
                         baseline_m: float, settings: dict, padding: int) -> tuple[np.ndarray, int]:
    """Estimate stereo-backed camera motion, then reject high-residual object features."""
    keep = np.ones(len(before), dtype=bool)
    if len(before) < 8:
        return keep, 0
    h, w = disparity.shape
    xy = np.rint(before).astype(int)
    xy[:, 0] = np.clip(xy[:, 0], 0, w-1); xy[:, 1] = np.clip(xy[:, 1], 0, h-1)
    d = disparity[xy[:, 1], xy[:, 0]]
    good = np.isfinite(d) & (d > float(settings["min_disparity_px"]))
    z = K[0, 0] * baseline_m / np.maximum(d, 1e-6)
    good &= (z > float(settings["min_depth_m"])) & (z < float(settings["max_depth_m"]))
    u, v = before[:, 0], before[:, 1]
    points3 = np.c_[(u-K[0,2])*z/K[0,0], (v-K[1,2])*z/K[1,1], z]
    valid_idx = np.flatnonzero(good)
    if len(valid_idx) < 8:
        return keep, 0
    success, rvec, tvec, _ = cv2.solvePnPRansac(points3[valid_idx].astype(np.float32),
        after[valid_idx].astype(np.float32), K, None, reprojectionError=float(settings["max_reprojection_px"]),
        iterationsCount=100, confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
    if not success:
        return keep, 0
    projected, _ = cv2.projectPoints(points3[valid_idx], rvec, tvec, K, None)
    error = np.linalg.norm(projected.reshape(-1, 2) - after[valid_idx], axis=1)
    object_points = _inside(before[valid_idx], boxes0, padding) | _inside(after[valid_idx], boxes1, padding)
    reject_idx = valid_idx[object_points & (error > float(settings["max_reprojection_px"]))]
    keep[reject_idx] = False
    if np.count_nonzero(keep) < int(settings["min_features"]):
        keep[:] = True
    return keep, len(reject_idx)
