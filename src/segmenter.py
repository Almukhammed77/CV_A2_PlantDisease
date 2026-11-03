"""Leaf ROI segmentation: SAM2 (SOTA) with a classical GrabCut baseline / fallback.

SAM2 is prompted with a few positive points around the image centre and negative points in
the corners; among the 3 candidate masks the one with the highest predicted IoU and a
plausible area is kept, then cleaned (largest connected component, holes filled).
"""
from __future__ import annotations

import time

import cv2
import numpy as np
from PIL import Image

MAX_SIDE = 1024          # images are downscaled for the segmenter, mask is upscaled back
MIN_AREA, MAX_AREA = 0.05, 0.95  # plausible leaf area as a fraction of the frame
SAM2_IDS = ("facebook/sam2.1-hiera-small", "facebook/sam2-hiera-small")


def clean_mask(mask: np.ndarray) -> np.ndarray:
    """Keep the largest connected component and fill holes."""
    m = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return mask.astype(bool)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    m = (labels == largest).astype(np.uint8)
    # fill holes: flood-fill the background from the border, invert
    h, w = m.shape
    flood = np.pad(m, 1).copy()
    ff = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(flood, ff, (0, 0), 2)
    holes = (flood[1:-1, 1:-1] == 0)
    return (m.astype(bool) | holes)


def grabcut_mask(img: np.ndarray, iters: int = 5) -> np.ndarray:
    """Classical baseline: GrabCut initialised with a centred rectangle."""
    h, w = img.shape[:2]
    rect = (int(w * 0.05), int(h * 0.05), int(w * 0.90), int(h * 0.90))
    mask = np.zeros((h, w), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.grabCut(bgr, mask, rect, bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
    return (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)


class Segmenter:
    """method: 'sam2' | 'grabcut' | 'auto' (SAM2 if installed, otherwise GrabCut)."""

    def __init__(self, method: str = "auto", device: str | None = None, model_id: str | None = None):
        self.predictor = None
        self.name = "grabcut"
        self.device = device
        if method in ("sam2", "auto"):
            try:
                self._load_sam2(device, model_id)
            except Exception as exc:  # noqa: BLE001
                if method == "sam2":
                    raise RuntimeError(
                        "SAM2 could not be loaded. Install it with: "
                        "pip install git+https://github.com/facebookresearch/sam2.git"
                    ) from exc
                print(f"[Segmenter] SAM2 unavailable ({exc}); falling back to GrabCut.")

    # ------------------------------------------------------------------ SAM2
    def _load_sam2(self, device, model_id):
        import torch
        from sam2.build_sam import build_sam2_hf
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        last = None
        for mid in ([model_id] if model_id else list(SAM2_IDS)):
            try:
                model = build_sam2_hf(mid, device=device)
                self.predictor = SAM2ImagePredictor(model)
                self.name = f"sam2 ({mid.split('/')[-1]})"
                return
            except Exception as exc:  # noqa: BLE001
                last = exc
        raise last  # type: ignore[misc]

    def _sam2_mask(self, img: np.ndarray):
        import torch

        h, w = img.shape[:2]
        pts = np.array([
            [w * 0.50, h * 0.50], [w * 0.40, h * 0.50], [w * 0.60, h * 0.50],   # leaf (positive)
            [w * 0.03, h * 0.03], [w * 0.97, h * 0.03],
            [w * 0.03, h * 0.97], [w * 0.97, h * 0.97]], dtype=np.float32)       # background (negative)
        lab = np.array([1, 1, 1, 0, 0, 0, 0], dtype=np.int32)
        use_cuda = str(self.device).startswith("cuda")
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_cuda else torch.autocast("cpu", enabled=False)
        with torch.inference_mode(), ctx:
            self.predictor.set_image(img)
            masks, scores, _ = self.predictor.predict(
                point_coords=pts, point_labels=lab, multimask_output=True)
        order = np.argsort(-np.asarray(scores))
        for i in order:
            m = clean_mask(np.asarray(masks[i]) > 0.5)
            frac = m.mean()
            if MIN_AREA <= frac <= MAX_AREA:
                return m, float(scores[i])
        return None, float(np.max(scores))

    # ------------------------------------------------------------------ public
    def __call__(self, pil_img: Image.Image):
        """Returns (mask[H,W] bool at the ORIGINAL size, info dict)."""
        t0 = time.perf_counter()
        pil_img = pil_img.convert("RGB")
        W, H = pil_img.size
        scale = min(1.0, MAX_SIDE / max(W, H))
        small = pil_img.resize((max(1, int(W * scale)), max(1, int(H * scale)))) if scale < 1 else pil_img
        arr = np.asarray(small)
        mask, score, fallback = None, None, False
        if self.predictor is not None:
            mask, score = self._sam2_mask(arr)
        else:
            m = clean_mask(grabcut_mask(arr))
            if MIN_AREA <= m.mean() <= MAX_AREA:
                mask = m
        if mask is None:  # no plausible mask -> keep the whole frame
            mask, fallback = np.ones(arr.shape[:2], bool), True
        if mask.shape != (H, W):
            mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)) > 127
        info = {"method": self.name, "score": score, "fallback": fallback,
                "area_frac": float(mask.mean()), "time_s": time.perf_counter() - t0}
        return mask, info
