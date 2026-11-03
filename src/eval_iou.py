#!/usr/bin/env python3
"""Mask quality: IoU / Dice of SAM2 masks (and GrabCut baseline) vs. hand-drawn masks.

Step 1 (pick images to annotate):   python src/eval_iou.py --prepare 14
    -> copies N stratified images into gt_todo/<class>/ . Paint the LEAF WHITE and the
       background BLACK (any editor, e.g. Paint / Photopea), save as PNG into
       gt_masks/<class>/ with the SAME file name stem (e.g. 12.jpeg -> 12.png).
Step 2 (evaluate):                  python src/eval_iou.py
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from roi import CLASS_NAMES, collect_image_paths  # noqa: E402
from segmenter import Segmenter, clean_mask, grabcut_mask  # noqa: E402


def iou_dice(pred: np.ndarray, gt: np.ndarray):
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    iou = inter / union if union else 1.0
    dice = 2 * inter / (pred.sum() + gt.sum()) if (pred.sum() + gt.sum()) else 1.0
    return float(iou), float(dice)


def prepare(data, n, todo):
    samples = collect_image_paths(data)
    rng = random.Random(0)
    todo = Path(todo)
    for c in CLASS_NAMES:
        files = [p for p, y in samples if CLASS_NAMES[y] == c]
        for p in rng.sample(files, min(n // 2, len(files))):
            dst = todo / c / p.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(p, dst)
    print(f"Copied images to {todo}/. Draw masks (leaf=white, bg=black) -> gt_masks/<class>/<same stem>.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--gt", default="gt_masks")
    ap.add_argument("--masks", default="outputs/seg/masks")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--prepare", type=int, default=0)
    ap.add_argument("--todo", default="gt_todo")
    args = ap.parse_args()

    if args.prepare:
        prepare(args.data, args.prepare, args.todo)
        return

    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    rows, viz = [], []
    samples = collect_image_paths(args.data)
    by_key = {(p.parent.name, p.stem): p for p, _ in samples}
    for gt_file in sorted(Path(args.gt).glob("*/*.png")):
        key = (gt_file.parent.name, gt_file.stem)
        if key not in by_key:
            print("No image for", gt_file)
            continue
        img = Image.open(by_key[key]).convert("RGB")
        gt = np.asarray(Image.open(gt_file).convert("L").resize(img.size, Image.NEAREST)) > 127
        sam_path = Path(args.masks) / key[0] / (key[1] + ".png")
        if not sam_path.exists():
            print("No SAM2 mask for", key)
            continue
        sam = np.asarray(Image.open(sam_path).convert("L").resize(img.size, Image.NEAREST)) > 127
        gc = clean_mask(grabcut_mask(np.asarray(img)))
        i1, d1 = iou_dice(sam, gt)
        i2, d2 = iou_dice(gc, gt)
        rows.append({"class": key[0], "file": by_key[key].name, "iou_sam2": i1, "dice_sam2": d1,
                     "iou_grabcut": i2, "dice_grabcut": d2})
        viz.append((img, gt, sam, gc, i1, i2))

    if not rows:
        print("No ground-truth masks found. Run with --prepare 14, draw masks, then rerun.")
        return
    df = pd.DataFrame(rows)
    df.to_csv(out / "iou_table.csv", index=False)
    summary = {
        "n_images": len(df),
        "sam2_iou_mean": float(df.iou_sam2.mean()), "sam2_iou_std": float(df.iou_sam2.std(ddof=0)),
        "sam2_dice_mean": float(df.dice_sam2.mean()),
        "grabcut_iou_mean": float(df.iou_grabcut.mean()), "grabcut_iou_std": float(df.iou_grabcut.std(ddof=0)),
        "grabcut_dice_mean": float(df.dice_grabcut.mean()),
        "per_class": {c: {"sam2_iou": float(g.iou_sam2.mean()), "grabcut_iou": float(g.iou_grabcut.mean()),
                          "n": int(len(g))} for c, g in df.groupby("class")},
    }
    (out / "iou_results.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    k = min(4, len(viz))
    fig, ax = plt.subplots(k, 4, figsize=(11, 2.8 * k), squeeze=False)
    for r, (img, gt, sam, gc, i1, i2) in enumerate(viz[:k]):
        for c, (im, t) in enumerate([(np.asarray(img), "image"), (gt, "manual GT"),
                                     (sam, f"SAM2 IoU={i1:.2f}"), (gc, f"GrabCut IoU={i2:.2f}")]):
            ax[r, c].imshow(im, cmap="gray")
            ax[r, c].set_title(t, fontsize=9)
            ax[r, c].axis("off")
    fig.tight_layout()
    fig.savefig(out / "figures" / "iou_examples.png", dpi=130)


if __name__ == "__main__":
    main()
