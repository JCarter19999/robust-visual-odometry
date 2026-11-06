"""Fetch the pretrained detector and confirm it runs on the local GPU."""

from pathlib import Path
import shutil

import torch
from ultralytics import YOLO


def main() -> None:
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the project environment")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    frame = Path("data/kitti/sequences/00/image_2/000000.png")
    for name in ("yolo11n.pt", "yolo11n-seg.pt"):
        destination = Path("data/models") / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.is_file():
            model = YOLO(name)
            source = Path(model.ckpt_path)
            if source.resolve() != destination.resolve():
                shutil.move(str(source), str(destination))
        model = YOLO(str(destination))
        print(f"Model: {destination.resolve()} ({destination.stat().st_size / 1e6:.1f} MB)")
        if frame.is_file():
            result = model.predict(str(frame), device=0, verbose=False)[0]
            extras = f", {len(result.masks) if result.masks is not None else 0} masks" if "-seg" in name else ""
            print(f"Inference: {len(result.boxes)} boxes{extras} on {frame.name}")


if __name__ == "__main__":
    main()
