# Robust Visual Odometry Under Dynamic Objects

Compare monocular feature-based visual odometry using a baseline, hard YOLO box filtering, selective flow-gated filtering, and YOLO instance-mask filtering. The experiment uses a quieter window from KITTI odometry sequence 00 and traffic windows from sequence 07, left color camera (`image_2`), matching calibration, and ground truth poses.

## Layout

`configs/` holds experiment settings; `scripts/` holds data acquisition utilities; `src/rvo/` holds reusable code. `data/` and `outputs/` are ignored by Git.

## Environment

From this directory in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe scripts\check_environment.py
```

The CUDA wheel command above is published by [PyTorch](https://pytorch.org/get-started/previous-versions/). If the environment check reports `CUDA available: False`, use the [official PyTorch selector](https://pytorch.org/get-started/locally/) to choose a wheel compatible with the current driver, then rerun the check.

## KITTI data

The official [KITTI odometry page](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) describes the data and license. The downloader reads directly from KITTI's official S3 archives. It fetches only requested frames from the large color ZIP, plus the small calibration and pose ZIPs. The experiment uses frames 0-299 of sequence 00 and frames 600-899 of sequence 07, including a separate 800-899 analysis window.

```powershell
.\.venv\Scripts\python.exe scripts\download_kitti.py --sequences 00 --frames 300
.\.venv\Scripts\python.exe scripts\download_kitti.py --sequences 07 --start 600 --frames 300
.\.venv\Scripts\python.exe -m rvo.kitti data/kitti --sequences 00 07
.\.venv\Scripts\python.exe scripts\preview_kitti.py
.\.venv\Scripts\python.exe scripts\prepare_model.py
```

The downloader is resumable: existing frames are skipped. Rerun with a larger `--frames` value to extend a sequence, or `--start` for a later contiguous window. Downloaded data stays under `data/` and is not committed.

```
data/kitti/
  sequences/00/calib.txt
  sequences/00/image_2/000000.png
  poses/00.txt
```

KITTI sequences 00-10 include ground truth; sequences 11-21 do not. Use camera `P2` from `calib.txt` with `image_2`. Monocular translation has unknown scale, so trajectory metrics require one documented similarity alignment per run; per-frame ground truth scale must not enter pose estimation.

## First experiment

Settings live in `configs/kitti_mvp.toml`; the comparison rules and metric definitions are in `docs/experiment_protocol.md`. The baseline, hard-box, selective-flow, and instance-mask variants share tracking, RANSAC, and pose settings. Outputs include ATE, translation and rotation RPE, failure rate, processing time, trajectories, and annotated frames.

The 300-frame windows make iteration manageable. Additional local exploratory frames from sequences 01 and 05 are present on this machine; they are outside the first experiment. Record exact frame ranges with every result.

## Run paired visual odometry

```powershell
.\.venv\Scripts\python.exe -m rvo.run
```

The runner uses the configured windows and writes per-frame records, estimated poses, metric summaries, and trajectory plots under `outputs/kitti_mvp/`. For a shorter development run, add `--limit 30`; for one sequence, add `--sequences 07`. All variants use the same decoded frames and tracked correspondences before filtering.

The current results and failure analysis are summarized in `docs/initial_results.md` with trajectory plots and a feature-filtering example. To inspect flow and ground-truth motion at the failure frames, run `python scripts/diagnose_motion.py`; ground truth is used only for diagnosis and evaluation.

The instance-mask variant uses `yolo11n-seg.pt` and excludes a feature if either endpoint falls inside a padded object mask. Run `scripts/prepare_model.py` to fetch both detection and segmentation weights before running the comparison.

To inspect which features are removed near crossing traffic:

```powershell
.\.venv\Scripts\python.exe scripts\visualize_tracks.py --sequence 07 --frames 650 700 824
```
