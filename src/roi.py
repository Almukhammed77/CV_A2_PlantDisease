"""Torch-free helpers: file collection and ROI (mask) application."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

CLASS_NAMES = ["diseased", "healthy"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BG_COLOR = (124, 116, 104)  # ImageNet-mean grey, paints the removed background


def collect_image_paths(data_dir) -> list[tuple[Path, int]]:
    """Same logic as A1: sorted files, skip non-images / corrupt files."""
    data_dir = Path(data_dir)
    samples: list[tuple[Path, int]] = []
    for class_name in CLASS_NAMES:
        folder = data_dir / class_name
        for path in sorted(folder.iterdir()):
            if path.name.startswith(".") or path.suffix.lower() not in IMG_EXT:
                continue
            try:
                with Image.open(path) as im:
                    im.verify()
                with Image.open(path) as im:
                    im.convert("RGB")
                samples.append((path, CLASS_TO_IDX[class_name]))
            except Exception as exc:  # noqa: BLE001
                print(f"Skipped corrupt image {path}: {exc}")
    return samples


def mask_path_for(mask_dir, img_path) -> Path:
    img_path = Path(img_path)
    return Path(mask_dir) / img_path.parent.name / (img_path.stem + ".png")


def load_mask(path, size=None) -> np.ndarray:
    m = Image.open(path).convert("L")
    if size is not None and m.size != tuple(size):
        m = m.resize(tuple(size), Image.NEAREST)
    return np.asarray(m) > 127


def apply_setting(img: Image.Image, mask, setting: str, pad: float = 0.05) -> Image.Image:
    """raw: untouched | masked: background painted grey | crop: masked + bounding-box crop."""
    if setting == "raw" or mask is None:
        return img
    arr = np.asarray(img.convert("RGB"))
    bg = np.empty_like(arr)
    bg[:] = BG_COLOR
    out = Image.fromarray(np.where(mask[..., None], arr, bg).astype(np.uint8))
    if setting == "masked":
        return out
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return out
    h, w = mask.shape
    py = int((ys.max() - ys.min() + 1) * pad)
    px = int((xs.max() - xs.min() + 1) * pad)
    box = (max(xs.min() - px, 0), max(ys.min() - py, 0),
           min(xs.max() + 1 + px, w), min(ys.max() + 1 + py, h))
    return out.crop(box)


