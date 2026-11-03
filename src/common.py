"""Shared utilities for Assignment 2: data, ROI preparation, model, training, metrics.

The split and the training recipe follow Assignment 1 (a1/train_and_evaluate.py) exactly,
so that A2 results are directly comparable with the A1 baseline.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from roi import (BG_COLOR, CLASS_NAMES, CLASS_TO_IDX, IMG_EXT, apply_setting,  # noqa: F401
                 collect_image_paths, load_mask, mask_path_for)

SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 16
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
# raw = A1 input | masked = background removed | crop = background removed + bbox crop
SETTINGS = ("raw", "masked", "crop")


# ----------------------------------------------------------------------------- basics
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def split_samples(samples):
    """A1 split: 80/20 (train+val vs test), then 80/20 (train vs val); stratified, seed 42."""
    paths = [p for p, _ in samples]
    labels = np.array([y for _, y in samples])
    tr_p, te_p, tr_y, te_y = train_test_split(
        paths, labels, test_size=0.2, random_state=SEED, stratify=labels)
    tr_p, va_p, tr_y, va_y = train_test_split(
        tr_p, tr_y, test_size=0.2, random_state=SEED, stratify=tr_y)
    return {
        "train": (tr_p, tr_y.tolist()),
        "val": (va_p, va_y.tolist()),
        "test": (te_p, te_y.tolist()),
    }


class LeafDataset(Dataset):
    def __init__(self, paths, labels, transform=None, setting="raw", mask_dir=None):
        self.paths, self.labels = paths, labels
        self.transform, self.setting, self.mask_dir = transform, setting, mask_dir

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = Path(self.paths[idx])
        img = Image.open(p).convert("RGB")
        if self.setting != "raw":
            mp = mask_path_for(self.mask_dir, p)
            if not mp.exists():
                raise FileNotFoundError(f"Mask not found: {mp}. Run src/segment_sam2.py first.")
            img = apply_setting(img, load_mask(mp, img.size), self.setting)
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def build_transforms(train: bool):
    norm = transforms.Normalize(mean=MEAN, std=STD)
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(25),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2),
            transforms.ToTensor(), norm])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(), norm])


# ----------------------------------------------------------------------------- model
def build_resnet18(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    if pretrained:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    else:
        model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate_loader(model, loader, criterion, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    ys, ps, probs = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss_sum += criterion(out, y).item() * x.size(0)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)
        ys += y.cpu().tolist()
        ps += pred.cpu().tolist()
        probs += torch.softmax(out, 1).cpu().tolist()
    return loss_sum / total, correct / total, np.array(ys), np.array(ps), np.array(probs)


def fit_stage(model, train_loader, val_loader, device, epochs, lr, name,
              patience=5, verbose=True):
    criterion = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val, best_state, left = float("inf"), None, patience
    for ep in range(1, epochs + 1):
        tl, ta = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl, va, *_ = evaluate_loader(model, val_loader, criterion, device)
        scheduler.step(vl)
        for k, v in zip(hist, (tl, vl, ta, va)):
            hist[k].append(v)
        if verbose:
            print(f"[{name}] epoch {ep:02d}/{epochs} train_loss={tl:.4f} acc={ta:.3f} "
                  f"| val_loss={vl:.4f} acc={va:.3f}")
        if vl < best_val:
            best_val, left = vl, patience
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            left -= 1
            if left == 0:
                if verbose:
                    print(f"[{name}] early stopping at epoch {ep}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return hist


def train_resnet_two_stage(train_loader, val_loader, device, head_epochs=6, ft_epochs=12,
                           verbose=True):
    """A1 recipe: (1) train fc head, lr 1e-3; (2) fine-tune layer4 + fc, lr 1e-4."""
    model = build_resnet18().to(device)
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True
    h1 = fit_stage(model, train_loader, val_loader, device, head_epochs, 1e-3, "head", verbose=verbose)
    for n, p in model.named_parameters():
        if n.startswith("layer4") or n.startswith("fc"):
            p.requires_grad = True
    h2 = fit_stage(model, train_loader, val_loader, device, ft_epochs, 1e-4, "finetune", verbose=verbose)
    return model, {k: h1[k] + h2[k] for k in h1}


# ----------------------------------------------------------------------------- metrics
def compute_metrics(y_true, y_pred) -> dict:
    """Weighted P/R/F1 (as in A1) + macro-F1 + confusion matrix."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def make_loader(split, key, setting, mask_dir, train, num_workers=2):
    paths, labels = split[key]
    ds = LeafDataset(paths, labels, build_transforms(train), setting, mask_dir)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=train, num_workers=num_workers)
