#!/usr/bin/env python3
"""Plant disease classification: baseline + transfer learning pipeline."""

from __future__ import annotations

import json
import os
import random
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "dataset"
FIG_DIR = ROOT / "figures"
OUT_DIR = ROOT / "outputs"
FIG_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS_SIMPLE = 20
EPOCHS_TL = 12
NUM_WORKERS = 0
CLASS_NAMES = ["diseased", "healthy"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def collect_image_paths() -> list[tuple[Path, int]]:
    samples: list[tuple[Path, int]] = []
    skipped = []
    for class_name in CLASS_NAMES:
        folder = DATA_DIR / class_name
        for path in sorted(folder.iterdir()):
            if path.name.startswith(".") or path.suffix.lower() not in {
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".webp",
            }:
                continue
            try:
                with Image.open(path) as im:
                    im.verify()
                with Image.open(path) as im:
                    im.convert("RGB")
                samples.append((path, CLASS_TO_IDX[class_name]))
            except Exception as exc:  # noqa: BLE001
                skipped.append((str(path), str(exc)))
    if skipped:
        print(f"Skipped {len(skipped)} corrupt images:")
        for p, e in skipped:
            print(f"  - {p}: {e}")
    return samples


def plot_class_distribution(samples: list[tuple[Path, int]]) -> None:
    counts = {name: 0 for name in CLASS_NAMES}
    for _, y in samples:
        counts[CLASS_NAMES[y]] += 1
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(list(counts.keys()), list(counts.values()), color=["#c44e52", "#4c72b0"])
    ax.set_title("Dataset class distribution")
    ax.set_ylabel("Number of images")
    for i, (k, v) in enumerate(counts.items()):
        ax.text(i, v + 1, str(v), ha="center")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "class_distribution.png", dpi=150)
    plt.close(fig)


def plot_sample_grid(samples: list[tuple[Path, int]], n_per_class: int = 4) -> None:
    fig, axes = plt.subplots(2, n_per_class, figsize=(12, 6))
    for row, class_name in enumerate(CLASS_NAMES):
        class_samples = [p for p, y in samples if CLASS_NAMES[y] == class_name]
        chosen = class_samples[:n_per_class]
        for col, path in enumerate(chosen):
            img = Image.open(path).convert("RGB")
            axes[row, col].imshow(img)
            axes[row, col].axis("off")
            if col == 0:
                axes[row, col].set_ylabel(class_name, fontsize=12)
            axes[row, col].set_title(path.name, fontsize=8)
    fig.suptitle("Sample images from the dataset", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sample_images.png", dpi=150)
    plt.close(fig)


def extract_color_histogram(path: Path, bins: int = 32) -> np.ndarray:
    img = np.asarray(Image.open(path).convert("RGB").resize((128, 128)))
    feats = []
    for c in range(3):
        hist, _ = np.histogram(img[:, :, c], bins=bins, range=(0, 256), density=True)
        feats.append(hist)
    return np.concatenate(feats).astype(np.float32)


def train_classical_baselines(
    train_paths: list[Path],
    train_labels: np.ndarray,
    test_paths: list[Path],
    test_labels: np.ndarray,
) -> dict:
    print("Extracting color histogram features...")
    X_train = np.stack([extract_color_histogram(p) for p in tqdm(train_paths)])
    X_test = np.stack([extract_color_histogram(p) for p in tqdm(test_paths)])

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = {}

    logreg = LogisticRegression(max_iter=2000, random_state=SEED)
    logreg.fit(X_train_s, train_labels)
    pred_lr = logreg.predict(X_test_s)
    results["logistic_regression"] = evaluate_predictions(
        test_labels, pred_lr, "Logistic Regression"
    )

    svm = LinearSVC(max_iter=5000, random_state=SEED)
    svm.fit(X_train_s, train_labels)
    pred_svm = svm.predict(X_test_s)
    results["svm"] = evaluate_predictions(test_labels, pred_svm, "Linear SVM")

    return results


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, title: str) -> dict:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "report": classification_report(
            y_true, y_pred, target_names=CLASS_NAMES, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
    }
    plot_confusion(y_true, y_pred, title)
    print(f"\n=== {title} ===")
    print(metrics["report"])
    return metrics


def plot_confusion(y_true, y_pred, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {title}")
    safe = title.lower().replace(" ", "_").replace("—", "").replace("__", "_")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"cm_{safe}.png", dpi=150)
    plt.close(fig)


class LeafDataset(Dataset):
    def __init__(self, paths: list[Path], labels: list[int], transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def build_transforms(train: bool):
    if train:
        return transforms.Compose(
            [
                transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
                transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(25),
                transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate_model(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
    return (
        running_loss / total,
        correct / total,
        np.array(all_labels),
        np.array(all_preds),
    )


def fit_torch_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs: int,
    lr: float,
    name: str,
    freeze_backbone: bool = False,
):
    if freeze_backbone and hasattr(model, "features") is False:
        # ResNet-style: freeze all except final fc
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val = float("inf")
    best_state = None
    patience_left = 5

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        va_loss, va_acc, _, _ = evaluate_model(model, val_loader, criterion, device)
        scheduler.step(va_loss)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)
        print(
            f"[{name}] Epoch {epoch:02d}/{epochs} "
            f"train_loss={tr_loss:.4f} acc={tr_acc:.3f} | "
            f"val_loss={va_loss:.4f} acc={va_acc:.3f}"
        )
        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_left = 5
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f"[{name}] Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def plot_history(history: dict, name: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title(f"{name} — Loss")
    axes[0].legend()
    axes[0].set_xlabel("Epoch")
    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title(f"{name} — Accuracy")
    axes[1].legend()
    axes[1].set_xlabel("Epoch")
    fig.tight_layout()
    safe = name.lower().replace(" ", "_")
    fig.savefig(FIG_DIR / f"history_{safe}.png", dpi=150)
    plt.close(fig)


def build_resnet18(num_classes: int = 2) -> nn.Module:
    model = models.resnet18(weights=None)
    ckpt = ROOT / ".cache" / "torch" / "hub" / "checkpoints" / "resnet18-f37072fd.pth"
    if ckpt.exists():
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    else:
        # Fallback to torchvision download if local checkpoint is missing
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def plot_comparison(all_metrics: dict) -> None:
    names = []
    accs = []
    f1s = []
    for name, m in all_metrics.items():
        names.append(name.replace("_", " ").title())
        accs.append(m["accuracy"])
        f1s.append(m["f1"])
    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - width / 2, accs, width, label="Accuracy")
    ax.bar(x + width / 2, f1s, width, label="F1-score")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison")
    ax.legend()
    for i, (a, f) in enumerate(zip(accs, f1s)):
        ax.text(i - width / 2, a + 0.02, f"{a:.2f}", ha="center", fontsize=8)
        ax.text(i + width / 2, f + 0.02, f"{f:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "model_comparison.png", dpi=150)
    plt.close(fig)


def plot_augmentation_examples(path: Path) -> None:
    img = Image.open(path).convert("RGB")
    aug = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomRotation(25),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        ]
    )
    fig, axes = plt.subplots(1, 5, figsize=(12, 3))
    axes[0].imshow(img.resize((IMG_SIZE, IMG_SIZE)))
    axes[0].set_title("Original")
    axes[0].axis("off")
    for i in range(1, 5):
        axes[i].imshow(aug(img))
        axes[i].set_title(f"Aug {i}")
        axes[i].axis("off")
    fig.suptitle("Data augmentation examples")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "augmentation_examples.png", dpi=150)
    plt.close(fig)


def main() -> None:
    set_seed(SEED)
    device = get_device()
    print(f"Device: {device}")

    samples = collect_image_paths()
    print(f"Valid images: {len(samples)}")
    plot_class_distribution(samples)
    plot_sample_grid(samples)

    paths = [p for p, _ in samples]
    labels = np.array([y for _, y in samples])

    train_paths, test_paths, train_y, test_y = train_test_split(
        paths,
        labels,
        test_size=0.2,
        random_state=SEED,
        stratify=labels,
    )
    train_paths, val_paths, train_y, val_y = train_test_split(
        train_paths,
        train_y,
        test_size=0.2,
        random_state=SEED,
        stratify=train_y,
    )

    split_info = {
        "train": int(len(train_paths)),
        "val": int(len(val_paths)),
        "test": int(len(test_paths)),
        "total": int(len(samples)),
    }
    print("Split:", split_info)
    plot_augmentation_examples(train_paths[0])

    all_metrics: dict = {}
    classical = train_classical_baselines(train_paths, train_y, test_paths, test_y)
    all_metrics.update(classical)

    train_ds = LeafDataset(train_paths, train_y.tolist(), build_transforms(True))
    val_ds = LeafDataset(val_paths, val_y.tolist(), build_transforms(False))
    test_ds = LeafDataset(test_paths, test_y.tolist(), build_transforms(False))

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # Baseline Simple CNN
    simple = SimpleCNN().to(device)
    hist_simple = fit_torch_model(
        simple,
        train_loader,
        val_loader,
        device,
        epochs=EPOCHS_SIMPLE,
        lr=1e-3,
        name="Simple CNN",
    )
    plot_history(hist_simple, "Simple CNN")
    _, _, y_true, y_pred = evaluate_model(
        simple, test_loader, nn.CrossEntropyLoss(), device
    )
    all_metrics["simple_cnn"] = evaluate_predictions(y_true, y_pred, "Simple CNN")
    torch.save(simple.state_dict(), OUT_DIR / "simple_cnn.pt")

    # Transfer learning ResNet18
    resnet = build_resnet18().to(device)
    # Stage 1: train head only
    for p in resnet.parameters():
        p.requires_grad = False
    for p in resnet.fc.parameters():
        p.requires_grad = True
    hist_tl_1 = fit_torch_model(
        resnet,
        train_loader,
        val_loader,
        device,
        epochs=6,
        lr=1e-3,
        name="ResNet18 head",
    )
    # Stage 2: fine-tune last blocks lightly
    for name, param in resnet.named_parameters():
        if name.startswith("layer4") or name.startswith("fc"):
            param.requires_grad = True
    hist_tl_2 = fit_torch_model(
        resnet,
        train_loader,
        val_loader,
        device,
        epochs=EPOCHS_TL,
        lr=1e-4,
        name="ResNet18 fine-tune",
    )
    hist_tl = {
        k: hist_tl_1[k] + hist_tl_2[k] for k in hist_tl_1
    }
    plot_history(hist_tl, "ResNet18 Transfer Learning")
    _, _, y_true, y_pred = evaluate_model(
        resnet, test_loader, nn.CrossEntropyLoss(), device
    )
    all_metrics["resnet18_transfer"] = evaluate_predictions(
        y_true, y_pred, "ResNet18 Transfer Learning"
    )
    torch.save(resnet.state_dict(), OUT_DIR / "resnet18_transfer.pt")

    plot_comparison(all_metrics)

    # Serializable summary (drop large arrays for JSON)
    summary = {
        "split": split_info,
        "device": str(device),
        "models": {},
    }
    for name, m in all_metrics.items():
        summary["models"][name] = {
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "confusion_matrix": m["confusion_matrix"],
            "report": m["report"],
        }

    with open(OUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    rows = []
    for name, m in summary["models"].items():
        rows.append(
            {
                "Model": name,
                "Accuracy": round(m["accuracy"], 4),
                "Precision": round(m["precision"], 4),
                "Recall": round(m["recall"], 4),
                "F1": round(m["f1"], 4),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "metrics_table.csv", index=False)
    print("\nFinal comparison:")
    print(df.to_string(index=False))
    print(f"\nSaved figures to {FIG_DIR}")
    print(f"Saved metrics to {OUT_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
