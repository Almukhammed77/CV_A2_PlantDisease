"""End-to-end inference: photo -> SAM2 leaf mask -> ROI -> ResNet18 -> Grad-CAM."""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from common import CLASS_NAMES, IMG_SIZE, MEAN, STD, build_resnet18, get_device
from gradcam import GradCAM, overlay_cam
from roi import apply_setting
from segmenter import Segmenter
from torchvision import transforms

_TF = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
                          transforms.Normalize(mean=MEAN, std=STD)])


def environment_string(device) -> str:
    s = f"{platform.system()} {platform.machine()} | Python {platform.python_version()} | torch {torch.__version__} | device: {device}"
    if torch.cuda.is_available():
        s += f" ({torch.cuda.get_device_name(0)})"
    return s


def pick_setting(out_dir="outputs", setting=None) -> str:
    """Use the explicit setting, else the best one by validation F1 from metrics_a2.json."""
    if setting:
        return setting
    mp = Path(out_dir) / "metrics_a2.json"
    if mp.exists():
        return json.loads(mp.read_text())["best_setting_by_val_f1"]
    return "crop"


class LeafPipeline:
    def __init__(self, out_dir="outputs", setting=None, method="auto"):
        self.device = get_device()
        self.setting = pick_setting(out_dir, setting)
        self.segmenter = Segmenter(method)
        self.model = build_resnet18(pretrained=False)
        w = Path(out_dir) / f"resnet18_{self.setting}.pt"
        if not w.exists():
            raise FileNotFoundError(f"{w} not found - run src/run_experiments.py first.")
        self.model.load_state_dict(torch.load(w, map_location="cpu"))
        self.model.to(self.device).eval()
        self.cam = GradCAM(self.model)

    @staticmethod
    def _sync(device):
        if device.type == "cuda":
            torch.cuda.synchronize()

    def run(self, pil_img: Image.Image, with_cam: bool = True) -> dict:
        pil_img = pil_img.convert("RGB")
        t0 = time.perf_counter()
        mask, info = self.segmenter(pil_img)
        self._sync(self.device)
        t1 = time.perf_counter()
        roi = apply_setting(pil_img, mask, self.setting)
        x = _TF(roi).unsqueeze(0).to(self.device)
        if with_cam:
            cam, probs, idx = self.cam(x)
        else:
            with torch.no_grad():
                probs = torch.softmax(self.model(x), 1)[0].cpu().numpy()
            cam, idx = None, int(np.argmax(probs))
        self._sync(self.device)
        t2 = time.perf_counter()
        roi_resized = np.asarray(roi.resize((IMG_SIZE, IMG_SIZE)))
        return {
            "label": CLASS_NAMES[idx],
            "probs": {c: float(p) for c, p in zip(CLASS_NAMES, probs)},
            "mask": mask,
            "roi": roi,
            "cam_overlay": overlay_cam(roi_resized, cam) if cam is not None else None,
            "segmenter": info["method"],
            "seg_fallback": info["fallback"],
            "t_seg_ms": (t1 - t0) * 1000,
            "t_cls_ms": (t2 - t1) * 1000,
            "t_total_ms": (t2 - t0) * 1000,
        }
