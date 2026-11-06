# Initial VO results and failure analysis

Run: `python -m rvo.run` with `configs/kitti_mvp.toml`, KITTI left color camera, YOLO11n, NVIDIA GeForce RTX 4070 Ti. These are exploratory windows, not a broad KITTI benchmark. ATE uses one Sim(3) fit per variant on poses valid in all three variants. All-frame ATE uses every frame, including the carried pose at a failed estimate. RPE uses 10-frame intervals with endpoints valid in all variants.

| Sequence / frames | Method | Common-pose ATE (m) | All-frame ATE (m) | Pose failures | Mean time (ms/frame) |
| --- | --- | ---: | ---: | ---: | ---: |
| 00 / 0-299 | Baseline | 5.16 | 5.16 | 0 / 299 | 46.6 |
| 00 / 0-299 | Hard boxes | 5.11 | 5.11 | 0 / 299 | 60.8 |
| 00 / 0-299 | Selective flow gate | 5.14 | 5.14 | 0 / 299 | 61.5 |
| 07 / 600-899 | Baseline | 7.80 | 8.29 | 32 / 299 | 50.1 |
| 07 / 600-899 | Hard boxes | 6.97 | 5.93 | 91 / 299 | 64.7 |
| 07 / 600-899 | Selective flow gate | 7.28 | 6.16 | 84 / 299 | 65.7 |
| 07 / 800-899 | Baseline | 1.99 | 1.99 | 0 / 99 | 50.6 |
| 07 / 800-899 | Hard boxes | 1.99 | 1.99 | 0 / 99 | 64.2 |
| 07 / 800-899 | Selective flow gate | 2.00 | 2.00 | 0 / 99 | 65.0 |

The 600-899 ATE comparison uses 204 poses valid in all variants; the other windows use every pose. The selective filter removes about 263 of 1,388 tracked features per frame in that window, compared with about 300 for hard boxes. It reduces failures from 91 to 84, a modest change. Its spatial-coverage fallback did not trigger in any of the three windows. Sixty of its failures meet the low-flow degeneracy rule (`median retained flow <= 4 px` and RANSAC inlier ratio `>= 0.8`). Those frames remain **failures**, with the last pose carried only for plotting and all-frame ATE.

The main failure cluster occurs around frames 643-738. The median ground-truth step in the 640-740 interval is about 0.009 m, while many failed filtered frames still have hundreds of RANSAC inliers and fewer than eight `recoverPose` inliers. This supports a near-stationary/low-parallax explanation. Ground truth was used for this diagnosis and evaluation only, never for pose estimation. Calling the carried pose a new successful estimate would not change the trajectory, so the runner does not do that.

On the moving-camera slice, frames 800-899, all three methods complete every pair and their ATE is effectively equal. These windows do not yet show a consistent accuracy benefit from object filtering. The filter adds roughly 14-15 ms per frame in this run. Timing includes frame read, Shi-Tomasi/Lucas-Kanade tracking, geometry, and YOLO inference plus filtering where applicable; model loading and CUDA warmup are excluded.

## Visuals

![Sequence 00 aligned trajectories](figures/seq00_trajectory.png)

![Sequence 07 aligned trajectories, including the near-stop](figures/seq07_trajectory.png)

![Sequence 07 moving-camera slice](figures/seq07_800_trajectory.png)

Yellow boxes mark potentially moving objects; red points are removed by hard filtering. The white van crosses the camera view in sequence 07 frame 650.

![Sequence 07 tracked feature filtering](figures/seq07_000650_tracks.jpg)
