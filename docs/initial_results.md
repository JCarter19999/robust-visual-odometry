# Initial VO results and failure analysis

Run: `python -m rvo.run` with `configs/kitti_mvp.toml`, KITTI left color camera, YOLO11n and YOLO11n-seg, NVIDIA GeForce RTX 4070 Ti. These are exploratory windows, not a broad KITTI benchmark. ATE uses one Sim(3) fit per variant on poses valid in all four variants. All-frame ATE uses every frame, including the carried pose at a failed estimate. RPE uses 10-frame intervals with endpoints valid in all variants.

| Sequence / frames | Method | Common-pose ATE (m) | All-frame ATE (m) | Pose failures | Mean time (ms/frame) |
| --- | --- | ---: | ---: | ---: | ---: |
| 00 / 0-299 | Baseline | 5.16 | 5.16 | 0 / 299 | 46.6 |
| 00 / 0-299 | Hard boxes | 5.11 | 5.11 | 0 / 299 | 60.8 |
| 00 / 0-299 | Selective flow gate | 5.14 | 5.14 | 0 / 299 | 61.5 |
| 00 / 0-299 | Instance masks | 5.06 | 5.06 | 0 / 299 | 64.1 |
| 07 / 600-899 | Baseline | 7.82 | 8.29 | 32 / 299 | 50.0 |
| 07 / 600-899 | Hard boxes | 6.99 | 5.93 | 91 / 299 | 61.8 |
| 07 / 600-899 | Selective flow gate | 7.30 | 6.16 | 84 / 299 | 62.8 |
| 07 / 600-899 | Instance masks | 7.01 | 5.80 | 81 / 299 | 64.8 |
| 07 / 800-899 | Baseline | 1.99 | 1.99 | 0 / 99 | 51.0 |
| 07 / 800-899 | Hard boxes | 1.99 | 1.99 | 0 / 99 | 64.8 |
| 07 / 800-899 | Selective flow gate | 2.00 | 2.00 | 0 / 99 | 65.9 |
| 07 / 800-899 | Instance masks | 1.99 | 1.99 | 0 / 99 | 67.9 |

The 600-899 ATE comparison uses 203 poses valid in all four variants; the other windows use every pose. In that window instance masks remove about 229 of 1,388 tracked features per frame, compared with about 300 for boxes. They reduce failures from 91 to 81, a modest improvement, while all-frame ATE falls from 5.93 m to 5.80 m. The spatial-coverage fallback did not trigger. The selective flow gate has 60 low-motion degeneracy frames (`median retained flow <= 4 px` and RANSAC inlier ratio `>= 0.8`); those remain failures.

The main failure cluster occurs around frames 643-738. The median ground-truth step in the 640-740 interval is about 0.009 m, while many failed filtered frames still have hundreds of RANSAC inliers and fewer than eight `recoverPose` inliers. This supports a near-stationary/low-parallax explanation. Ground truth was used for this diagnosis and evaluation only, never for pose estimation. Calling the carried pose a new successful estimate would not change the trajectory, so the runner does not do that.

On the moving-camera slice, frames 800-899, all methods complete every pair and their ATE is effectively equal. The experiments do not yet show a consistent accuracy benefit from object filtering. Instance masks add about 14-17 ms per frame here. Timing includes frame read, Shi-Tomasi/Lucas-Kanade tracking, geometry, and model inference plus filtering where applicable; model loading and CUDA warmup are excluded.

## Visuals

![Sequence 00 aligned trajectories](figures/seq00_trajectory.png)

![Sequence 07 aligned trajectories, including the near-stop](figures/seq07_trajectory.png)

![Sequence 07 moving-camera slice](figures/seq07_800_trajectory.png)

Yellow boxes mark potentially moving objects; red points are removed by hard filtering. The white van crosses the camera view in sequence 07 frame 650.

![Sequence 07 tracked feature filtering](figures/seq07_000650_tracks.jpg)
