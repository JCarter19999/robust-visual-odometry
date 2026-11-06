"""Fetch the pretrained detector and confirm it runs on the local GPU."""

from pathlib import Path
import shutil

import torch
from ultralytics import YOLO


def main() -> None:
    destination = Path("data/models/yolo11n.pt")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file():
        model = YOLO("yolo11n.pt")
        source = Path(model.ckpt_path)
        if source.resolve() != destination.resolve():
            shutil.move(str(source), str(destination))
    model = YOLO(str(destination))
    print(f"Model: {destination.resolve()} ({destination.stat().st_size / 1e6:.1f} MB)")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the project environment")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    frame = Path("data/kitti/sequences/00/image_2/000000.png")
    if frame.is_file():
        result = model.predict(str(frame), device=0, verbose=False)[0]
        print(f"Detector inference: {len(result.boxes)} boxes on {frame.name}")


if __name__ == "__main__":
    main()
