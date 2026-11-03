#!/usr/bin/env python3
"""Gradio demo: upload / camera -> SAM2 leaf mask -> healthy vs diseased -> Grad-CAM + latency.

Run:  python src/app.py --share        (share=True gives a public link, e.g. from Colab)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gradio as gr
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import LeafPipeline, environment_string  # noqa: E402
from roi import BG_COLOR  # noqa: E402,F401

NOTICE = (
    "**Notice.** This demo classifies photos of *plant leaves* (healthy vs diseased). "
    "No synthetic or generated media is used or produced. Please do not upload photos of people or faces. "
    "Uploaded images are processed in memory and are not stored. "
    "The model was trained on ~200 self-collected photos: treat the output as a demo, not as an agronomic diagnosis."
)


def build_app(pipe: LeafPipeline):
    env = environment_string(pipe.device)
    history: list[float] = []

    def predict(img):
        if img is None:
            return None, None, None, None, "Upload a leaf photo first."
        from PIL import Image
        pil = Image.fromarray(img) if isinstance(img, np.ndarray) else img
        r = pipe.run(pil)
        history.append(r["t_total_ms"])
        arr = np.asarray(pil.convert("RGB")).copy()
        over = arr.copy()
        over[r["mask"]] = (0.5 * over[r["mask"]] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
        stats = (
            f"**Segmenter:** {r['segmenter']}" + (" (no plausible mask - full frame used)" if r["seg_fallback"] else "")
            + f"  \n**Input variant:** `{pipe.setting}`"
            + f"  \n**Latency (this image):** segmentation {r['t_seg_ms']:.0f} ms + classification/Grad-CAM {r['t_cls_ms']:.0f} ms"
            f" = **{r['t_total_ms']:.0f} ms**"
            + f"  \n**Average over {len(history)} request(s):** {np.mean(history):.0f} ms"
            + f"  \n**Environment:** {env}"
        )
        return r["probs"], over, np.asarray(r["roi"]), r["cam_overlay"], stats

    with gr.Blocks(title="Leaf disease: SAM2 ROI + ResNet18 + Grad-CAM") as demo:
        gr.Markdown("# Plant leaf disease detection - SAM2 ROI + ResNet18 + Grad-CAM")
        gr.Markdown(NOTICE)
        with gr.Row():
            inp = gr.Image(label="Leaf photo (upload or camera)", sources=["upload", "webcam"], type="numpy")
            with gr.Column():
                label = gr.Label(label="Prediction", num_top_classes=2)
                btn = gr.Button("Analyse", variant="primary")
        with gr.Row():
            o_mask = gr.Image(label="SAM2 mask overlay (green = leaf)")
            o_roi = gr.Image(label="Model input (ROI)")
            o_cam = gr.Image(label="Grad-CAM (red = decisive regions)")
        info = gr.Markdown()
        btn.click(predict, inp, [label, o_mask, o_roi, o_cam, info])
    return demo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--setting", default=None, choices=[None, "raw", "masked", "crop"])
    ap.add_argument("--method", default="auto", choices=["auto", "sam2", "grabcut"])
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    pipe = LeafPipeline(args.out, args.setting, args.method)
    print("Segmenter:", pipe.segmenter.name, "| setting:", pipe.setting, "| device:", pipe.device)
    build_app(pipe).launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
