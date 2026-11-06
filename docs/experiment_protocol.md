# MVP experiment protocol

## Data and variants

- Use KITTI odometry training sequences with left color frames (`image_2`), matching `P2` calibration, and poses at the same integer frame indices.
- Record the exact sequence and contiguous frame range for every run. Initial comparison: sequence 00 frames 0-299 (quieter residential streets), sequence 07 frames 600-899 (crossing traffic and a near-stop), and a moving-camera slice of sequence 07 frames 800-899.
- Run the same Shi-Tomasi, Lucas-Kanade, essential-matrix RANSAC, and `recoverPose` settings in all variants. Fix random seeds where supported.
- Baseline: keep all valid tracked features. Filtered: discard a track if either endpoint falls within a padded YOLO box of a configured potentially moving class. Selective: reject a box only if median tracked flow inside it differs from background flow by at least the configured threshold. If the retained tracks do not meet count and image-grid coverage thresholds, use all tracks for that frame. A class label marks *potential* motion; parked cars may be wrongly discarded by the hard filter, while flow gating is only a heuristic for object motion.
- Do not use ground truth poses, frame-to-frame ground truth distance, or detection labels to estimate camera motion.

## Per-frame record

Save frame index, estimated 4x4 pose, number of detected corners, number of tracked features, number filtered, number passed to RANSAC, number of RANSAC inliers, whether pose estimation succeeded, a low-motion degeneracy flag, and timing for image read, detection, tracking, geometry, and total processing. For failure frames, record a failure and retain the last valid pose for trajectory plotting; do not count that retained pose as a successful estimate. The degeneracy flag is diagnostic and does not turn a failure into a success.

## Metrics

- **ATE:** root mean squared translation distance after a *single* Sim(3) alignment per estimated trajectory against ground truth. Fit and evaluate all variants on the same frames where every variant produced a valid pose. Report that common frame count and each variant's failures separately.
- **All-frame ATE:** fit one Sim(3) alignment on every frame, including carried poses at failures. This operational measure uses the same frame set for all variants.
- **RPE:** relative rotation and translation error over fixed frame intervals after the same sequence-level scale alignment. Use only pairs whose endpoints were valid in all variants.
- **Inlier ratio:** RANSAC inliers divided by correspondences submitted to RANSAC, aggregated with numerator and denominator totals.
- **Tracking failure rate:** failed pose-estimation frames divided by attempted frame pairs.
- **Processing time:** mean and median milliseconds per frame, including YOLO inference in the filtered run. Report GPU model and detector weights.

Compare all variants on identical frames and use the common valid frame set for ATE and RPE. Inspect annotated examples where filtering helps and hurts. Keep the original trajectory and metric files under `outputs/` so reported numbers can be reproduced.
