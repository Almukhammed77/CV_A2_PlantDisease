#!/usr/bin/env python3
"""Explainability figure: image | SAM2 mask | ROI input | Grad-CAM, on test images."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLASS_NAMES, collect_image_paths, split_samples  # noqa: E402
from pipeline import LeafPipeline  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--setting", default=None)
    ap.add_argument("--method", default="auto")
    ap.add_argument("--per-class", type=int, default=2)
    args = ap.parse_args()

    pipe = LeafPipeline(args.out, args.setting, args.method)
    split = split_samples(collect_image_paths(args.data))
    paths, labels = split["test"]
    chosen = []
    for c in range(2):
        chosen += [(p, c) for p, y in zip(paths, labels) if y == c][: args.per_class]

    fig, ax = plt.subplots(len(chosen), 4, figsize=(11, 2.7 * len(chosen)))
    for r, (p, y) in enumerate(chosen):
        img = Image.open(p).convert("RGB")
        res = pipe.run(img)
        arr = np.asarray(img).copy()
        over = arr.copy()
        over[res["mask"]] = (0.5 * over[res["mask"]] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
        ok = res["label"] == CLASS_NAMES[y]
        panels = [(arr, f"true: {CLASS_NAMES[y]}"), (over, "SAM2 mask"),
                  (np.asarray(res["roi"]), f"ROI ({pipe.setting})"),
                  (res["cam_overlay"], f"Grad-CAM -> {res['label']} {'OK' if ok else 'WRONG'}")]
        for c, (im, t) in enumerate(panels):
            ax[r, c].imshow(im)
            ax[r, c].set_title(t, fontsize=9, color="black" if (c < 3 or ok) else "red")
            ax[r, c].axis("off")
    fig.tight_layout()
    path = Path(args.out) / "figures" / "gradcam_examples.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    print("Saved", path)


if __name__ == "__main__":
    main()
