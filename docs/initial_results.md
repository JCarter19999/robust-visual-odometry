# Initial paired VO results

Run: `python -m rvo.run` with `configs/kitti_mvp.toml`, KITTI left color camera, YOLO11n, NVIDIA GeForce RTX 4070 Ti. These are development windows, not a broad KITTI benchmark. ATE uses one Sim(3) fit per variant on the same poses valid in both variants. RPE uses 10-frame intervals with valid endpoints in both variants. Each window has 300 frames and 299 pose attempts.

| Sequence / frames | Method | ATE RMSE (m) | 10-frame RPE translation (m) | 10-frame RPE rotation (deg) | Pose failures | Inlier ratio | Mean time (ms/frame) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 00 / 0-299 | Baseline | 5.16 | 1.75 | 0.302 | 0 | 0.870 | 41.3 |
| 00 / 0-299 | YOLO filtered | 5.11 | 1.76 | 0.340 | 0 | 0.877 | 53.0 |
| 07 / 600-899 | Baseline | 7.94 | 1.94 | 0.339 | 32 | 0.841 | 43.2 |
| 07 / 600-899 | YOLO filtered | 6.92 | 1.94 | 0.348 | 91 | 0.891 | 55.9 |

The sequence 07 ATE comparison uses 208 common valid poses. Filtering removes about 300 of 1,388 tracked features per frame on average. All 91 filtered pose failures in this run occurred after essential-matrix estimation when `recoverPose` returned fewer than the configured eight valid pose inliers. This may indicate insufficient usable parallax or another geometric degeneracy; the current run does not establish the cause. Filtering has a clear reliability cost here despite the lower ATE on common valid poses. The near-equal RPE values also limit any accuracy claim.

The time metric includes frame read, Shi-Tomasi/Lucas-Kanade tracking, geometry, and YOLO detection plus filtering for the filtered variant. It excludes model loading and CUDA warmup. Repeated timing runs and more scene windows are needed before drawing a speed conclusion.

## Visuals

![Sequence 00 aligned trajectories](figures/seq00_trajectory.png)

![Sequence 07 aligned trajectories](figures/seq07_trajectory.png)

Yellow boxes mark potentially moving objects; red points are removed from the pose estimate. The white van crosses the camera view in sequence 07 frame 650.

![Sequence 07 tracked feature filtering](figures/seq07_000650_tracks.jpg)
