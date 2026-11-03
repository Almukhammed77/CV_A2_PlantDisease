#!/usr/bin/env python3
"""Generate leaf masks for every dataset image (SAM2 by default).

Outputs:  <out>/masks/<class>/<stem>.png   (255 = leaf)
          <out>/seg_stats.json             (counts, fallback rate, mean area, mean time)
          <out>/figures/mask_examples.png  (image | mask overlay | ROI crop)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from roi import CLASS_NAMES, apply_setting, collect_image_paths  # noqa: E402
from segmenter import Segmenter  # noqa: E402


def overlay(img: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    out = img.astype(np.float32).copy()
    out[mask] = (1 - alpha) * out[mask] + alpha * np.array([0, 255, 0], np.float32)
    return out.astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--out", default="outputs/seg")
    ap.add_argument("--method", default="sam2", choices=["sam2", "grabcut", "auto"])
    args = ap.parse_args()

    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    seg = Segmenter(args.method)
    print("Segmenter:", seg.name)

    samples = collect_image_paths(args.data)
    infos, examples = [], {c: [] for c in CLASS_NAMES}
    for path, y in tqdm(samples, desc="segmenting"):
        img = Image.open(path).convert("RGB")
        mask, info = seg(img)
        mp = out / "masks" / path.parent.name / (path.stem + ".png")
        mp.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask.astype(np.uint8) * 255).save(mp)
        info["file"] = str(path)
        infos.append(info)
        if len(examples[CLASS_NAMES[y]]) < 2:
            examples[CLASS_NAMES[y]].append((img, mask))

    stats = {
        "method": seg.name,
        "n_images": len(infos),
        "n_fallback_full_frame": int(sum(i["fallback"] for i in infos)),
        "mean_area_fraction": float(np.mean([i["area_frac"] for i in infos])),
        "mean_time_s": float(np.mean([i["time_s"] for i in infos])),
    }
    (out / "seg_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))

    rows = [(c, e) for c in CLASS_NAMES for e in examples[c]]
    fig, axes = plt.subplots(len(rows), 3, figsize=(9, 3 * len(rows)))
    for r, (c, (img, mask)) in enumerate(rows):
        arr = np.asarray(img)
        panels = [arr, overlay(arr, mask), np.asarray(apply_setting(img, mask, "crop"))]
        for k, (p, t) in enumerate(zip(panels, ["image", "mask overlay", "ROI crop"])):
            axes[r, k].imshow(p)
            axes[r, k].axis("off")
            axes[r, k].set_title(f"{c}: {t}" if k == 0 else t, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "figures" / "mask_examples.png", dpi=130)
    print("Saved", out / "figures" / "mask_examples.png")


if __name__ == "__main__":
    main()
