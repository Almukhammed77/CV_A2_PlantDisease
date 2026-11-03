#!/usr/bin/env python3
"""Average inference latency of the full pipeline (segmentation + classification + Grad-CAM)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import collect_image_paths, split_samples  # noqa: E402
from pipeline import LeafPipeline, environment_string  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--setting", default=None)
    ap.add_argument("--method", default="auto")
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()

    pipe = LeafPipeline(args.out, args.setting, args.method)
    paths = split_samples(collect_image_paths(args.data))["test"][0][: args.n]
    imgs = [Image.open(p).convert("RGB") for p in paths]
    for im in imgs[:3]:  # warm-up (CUDA init, cudnn autotune, SAM2 compile)
        pipe.run(im)
    runs = [pipe.run(im) for im in imgs]
    res = {
        "environment": environment_string(pipe.device),
        "segmenter": runs[0]["segmenter"],
        "setting": pipe.setting,
        "n_images": len(runs),
        "avg_seg_ms": float(np.mean([r["t_seg_ms"] for r in runs])),
        "avg_cls_gradcam_ms": float(np.mean([r["t_cls_ms"] for r in runs])),
        "avg_total_ms": float(np.mean([r["t_total_ms"] for r in runs])),
        "p95_total_ms": float(np.percentile([r["t_total_ms"] for r in runs], 95)),
    }
    res["fps"] = 1000.0 / res["avg_total_ms"]
    Path(args.out).mkdir(exist_ok=True)
    (Path(args.out) / "latency.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
