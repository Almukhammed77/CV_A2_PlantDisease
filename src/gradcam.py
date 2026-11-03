"""Minimal Grad-CAM for torchvision ResNet (target layer = last block of layer4)."""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer=None):
        self.model = model
        self.layer = target_layer if target_layer is not None else model.layer4[-1]
        self.acts = None
        self.grads = None
        self._h = self.layer.register_forward_hook(self._fwd)

    def _fwd(self, module, inp, out):
        self.acts = out.detach()
        out.register_hook(lambda g: setattr(self, "grads", g.detach()))

    def remove(self):
        self._h.remove()

    def __call__(self, x: torch.Tensor, class_idx: int | None = None):
        """x: [1,3,H,W]. Returns (cam[H,W] in 0..1, probs[C], class_idx)."""
        self.model.eval()
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)
            logits = self.model(x)
            probs = torch.softmax(logits, 1)[0].detach().cpu().numpy()
            idx = int(logits.argmax(1).item()) if class_idx is None else class_idx
            self.model.zero_grad()
            logits[0, idx].backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, probs, idx


def overlay_cam(img_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a JET heat-map over an RGB uint8 image (cam is resized to the image)."""
    h, w = img_rgb.shape[:2]
    cam = cv2.resize(cam.astype(np.float32), (w, h))
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return (alpha * heat + (1 - alpha) * img_rgb).astype(np.uint8)
