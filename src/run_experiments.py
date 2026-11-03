#!/usr/bin/env python3
"""Train/evaluate ResNet18 (A1 recipe) on three input variants and compare.

  raw     - original photo (reproduces the A1 baseline)
  masked  - SAM2 mask, background painted grey          (ablation)
  crop    - SAM2 mask, background removed + bbox crop   (our full ROI pipeline)

The A1 split (seed 42) is kept fixed; only the training seed varies, so the 40-image test
set is identical to A1. Outputs: outputs/metrics_a2.json, results_table.csv, weights, figures.
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
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CLASS_NAMES, SETTINGS, collect_image_paths, compute_metrics,  # noqa: E402
                    evaluate_loader, get_device, make_loader, set_seed, split_samples,
                    train_resnet_two_stage)


def plot_history(hist, title, path):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ep = range(1, len(hist["train_loss"]) + 1)
    ax[0].plot(ep, hist["train_loss"], label="train")
    ax[0].plot(ep, hist["val_loss"], label="val")
    ax[0].set_title("Loss")
    ax[1].plot(ep, hist["train_acc"], label="train")
    ax[1].plot(ep, hist["val_acc"], label="val")
    ax[1].set_title("Accuracy")
    for a in ax:
        a.set_xlabel("epoch (head stage -> fine-tune stage)")
        a.legend()
        a.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_cm(cm, title, path):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], CLASS_NAMES)
    ax.set_yticks([0, 1], CLASS_NAMES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--masks", default="outputs/seg/masks")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--settings", nargs="+", default=list(SETTINGS), choices=list(SETTINGS))
    ap.add_argument("--a1-metrics", default="a1/metrics.json")
    ap.add_argument("--head-epochs", type=int, default=6)
    ap.add_argument("--ft-epochs", type=int, default=12)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    device = get_device()
    print("Device:", device)

    samples = collect_image_paths(args.data)
    split = split_samples(samples)
    split_info = {k: len(v[0]) for k, v in split.items()}
    split_info["total"] = len(samples)
    print("Split:", split_info)

    runs: dict[str, list] = {s: [] for s in args.settings}
    crit = nn.CrossEntropyLoss()
    for setting in args.settings:
        for seed in args.seeds:
            print(f"\n=== setting={setting} seed={seed} ===")
            set_seed(seed)
            tr = make_loader(split, "train", setting, args.masks, True, args.workers)
            va = make_loader(split, "val", setting, args.masks, False, args.workers)
            te = make_loader(split, "test", setting, args.masks, False, args.workers)
            model, hist = train_resnet_two_stage(tr, va, device, args.head_epochs, args.ft_epochs)
            _, _, yv, pv, _ = evaluate_loader(model, va, crit, device)
            _, _, yt, pt, prob = evaluate_loader(model, te, crit, device)
            rec = {"seed": seed, "val": compute_metrics(yv, pv), "test": compute_metrics(yt, pt),
                   "epochs_run": len(hist["train_loss"])}
            runs[setting].append(rec)
            print(f"test acc={rec['test']['accuracy']:.3f} f1={rec['test']['f1']:.3f} "
                  f"| val f1={rec['val']['f1']:.3f}")
            if seed == args.seeds[0]:
                torch.save(model.state_dict(), out / f"resnet18_{setting}.pt")
                plot_history(hist, f"ResNet18 - {setting} (seed {seed})",
                             out / "figures" / f"history_{setting}.png")
                plot_cm(rec["test"]["confusion_matrix"], f"{setting} (seed {seed}, test)",
                        out / "figures" / f"cm_{setting}.png")
                pd.DataFrame({
                    "file": [str(p) for p in split["test"][0]],
                    "true": [CLASS_NAMES[i] for i in yt],
                    "pred": [CLASS_NAMES[i] for i in pt],
                    "p_diseased": prob[:, 0]}).to_csv(out / f"test_predictions_{setting}.csv", index=False)

    # ---- summary (mean +- std over seeds)
    def agg(setting, split_key, metric):
        v = [r[split_key][metric] for r in runs[setting]]
        return float(np.mean(v)), float(np.std(v))

    summary = {}
    for s in runs:
        summary[s] = {}
        for sk in ("val", "test"):
            for m in ("accuracy", "precision", "recall", "f1", "macro_f1"):
                mu, sd = agg(s, sk, m)
                summary[s][f"{sk}_{m}_mean"], summary[s][f"{sk}_{m}_std"] = mu, sd
    best = max(summary, key=lambda s: summary[s]["val_f1_mean"])  # choose on VAL, not test

    a1 = None
    if Path(args.a1_metrics).exists():
        a1m = json.loads(Path(args.a1_metrics).read_text())["models"]["resnet18_transfer"]
        a1 = {k: a1m[k] for k in ("accuracy", "precision", "recall", "f1")}

    result = {"split": split_info, "device": str(device), "seeds": args.seeds,
              "a1_baseline_resnet18": a1, "runs": runs, "summary": summary,
              "best_setting_by_val_f1": best}
    (out / "metrics_a2.json").write_text(json.dumps(result, indent=2))

    rows = []
    if a1:
        rows.append({"model": "A1 ResNet18 (reported, 1 run)", "acc": a1["accuracy"], "acc_std": "",
                     "f1": a1["f1"], "f1_std": "", "val_f1": ""})
    for s in runs:
        rows.append({"model": f"A2 ResNet18 - {s}", "acc": summary[s]["test_accuracy_mean"],
                     "acc_std": summary[s]["test_accuracy_std"], "f1": summary[s]["test_f1_mean"],
                     "f1_std": summary[s]["test_f1_std"], "val_f1": summary[s]["val_f1_mean"]})
    pd.DataFrame(rows).to_csv(out / "results_table.csv", index=False)

    # comparison bar chart
    fig, ax = plt.subplots(figsize=(6, 3.8))
    names = list(runs)
    mu = [summary[s]["test_f1_mean"] for s in names]
    sd = [summary[s]["test_f1_std"] for s in names]
    ax.bar(names, mu, yerr=sd, capsize=5, color=["#8da0cb", "#66c2a5", "#fc8d62"][:len(names)])
    if a1:
        ax.axhline(a1["f1"], color="k", ls="--", lw=1, label=f"A1 ResNet18 F1={a1['f1']:.3f}")
        ax.legend()
    ax.set_ylabel(f"test F1 (mean +- std, {len(args.seeds)} seeds)")
    ax.set_ylim(0.5, 1.02)
    ax.set_title("Ablation: input variant")
    fig.tight_layout()
    fig.savefig(out / "figures" / "ablation_f1.png", dpi=140)
    plt.close(fig)
    print("\nBest setting by validation F1:", best)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
