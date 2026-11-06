"""Scale-aware evaluation for monocular KITTI camera trajectories."""

from __future__ import annotations

import numpy as np


def align_sim3(estimate: np.ndarray, reference: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit one proper similarity transform using valid estimated positions."""
    if np.count_nonzero(valid) < 3:
        raise ValueError("At least three valid poses are required for Sim(3) alignment")
    source = estimate[valid, :3, 3]
    target = reference[valid, :3, 3]
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    source_centered, target_centered = source - source_mean, target - target_mean
    source_variance = np.mean(np.sum(source_centered**2, axis=1))
    if source_variance < 1e-12:
        raise ValueError("Estimated trajectory has no translational spread")
    covariance = target_centered.T @ source_centered / len(source)
    U, singular_values, Vt = np.linalg.svd(covariance)
    sign = np.ones(3)
    sign[-1] = np.linalg.det(U @ Vt)
    rotation = U @ np.diag(sign) @ Vt
    scale = float(np.dot(singular_values, sign) / source_variance)
    translation = target_mean - scale * rotation @ source_mean
    aligned = estimate.copy()
    aligned[:, :3, 3] = scale * (rotation @ estimate[:, :3, 3].T).T + translation
    aligned[:, :3, :3] = rotation @ estimate[:, :3, :3]
    return aligned, scale


def rotation_error_degrees(first: np.ndarray, second: np.ndarray) -> float:
    cosine = np.clip((np.trace(first.T @ second) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(cosine)))


def metrics(estimate: np.ndarray, reference: np.ndarray, valid: np.ndarray,
            common_valid: np.ndarray, records: list[dict],
            interval: int = 10) -> tuple[dict, np.ndarray]:
    if len(estimate) != len(reference) or len(valid) != len(estimate) or len(common_valid) != len(estimate):
        raise ValueError("Pose and validity lengths differ")
    if np.any(common_valid & ~valid):
        raise ValueError("Common evaluation mask includes an invalid pose")
    aligned, scale = align_sim3(estimate, reference, common_valid)
    position_errors = np.linalg.norm(aligned[common_valid, :3, 3] - reference[common_valid, :3, 3], axis=1)
    all_mask = np.ones(len(estimate), dtype=bool)
    all_aligned, _ = align_sim3(estimate, reference, all_mask)
    all_position_errors = np.linalg.norm(all_aligned[:, :3, 3] - reference[:, :3, 3], axis=1)
    translation_errors, rotation_errors = [], []
    for first in range(len(estimate) - interval):
        second = first + interval
        if not (common_valid[first] and common_valid[second]):
            continue
        gt_rotation = reference[first, :3, :3].T @ reference[second, :3, :3]
        vo_rotation = aligned[first, :3, :3].T @ aligned[second, :3, :3]
        gt_translation = reference[first, :3, :3].T @ (
            reference[second, :3, 3] - reference[first, :3, 3]
        )
        vo_translation = aligned[first, :3, :3].T @ (
            aligned[second, :3, 3] - aligned[first, :3, 3]
        )
        translation_errors.append(float(np.linalg.norm(vo_translation - gt_translation)))
        rotation_errors.append(rotation_error_degrees(vo_rotation, gt_rotation))
    submitted = sum(row["submitted"] for row in records)
    inliers = sum(row["ransac_inliers"] for row in records)
    total_ms = np.array([row["total_ms"] for row in records])
    detection_ms = np.array([row["detection_ms"] for row in records])
    output = {
        "ate_rmse_m": float(np.sqrt(np.mean(position_errors**2))),
        "ate_all_frames_rmse_m": float(np.sqrt(np.mean(all_position_errors**2))),
        "rpe_interval_frames": interval,
        "rpe_pairs": len(translation_errors),
        "rpe_translation_mean_m": float(np.mean(translation_errors)) if translation_errors else None,
        "rpe_rotation_mean_deg": float(np.mean(rotation_errors)) if rotation_errors else None,
        "sim3_scale": scale,
        "sim3_aligned_common_poses": int(np.count_nonzero(common_valid)),
        "pose_failures": int(np.count_nonzero(~valid[1:])),
        "low_motion_degeneracy_frames": sum(row["low_motion_degeneracy"] for row in records),
        "pose_attempts": len(valid) - 1,
        "tracking_failure_rate": float(np.mean(~valid[1:])),
        "feature_inlier_ratio": inliers / submitted if submitted else None,
        "mean_total_ms": float(total_ms.mean()),
        "median_total_ms": float(np.median(total_ms)),
        "mean_detection_ms": float(detection_ms.mean()),
        "mean_tracked_features": float(np.mean([row["tracked"] for row in records])),
        "mean_removed_features": float(np.mean([row["removed"] for row in records])),
    }
    return output, aligned
