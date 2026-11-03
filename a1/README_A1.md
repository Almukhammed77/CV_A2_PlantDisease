# Plant Disease Detection — Computer Vision Assignment

Binary classification of plant leaves: **healthy** vs **diseased**.

## Results (test set, 40 images)

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| Logistic Regression | 0.875 | 0.876 | 0.875 | 0.875 |
| Linear SVM | 0.800 | 0.803 | 0.800 | 0.799 |
| Simple CNN | 0.875 | 0.876 | 0.875 | 0.875 |
| **ResNet18 Transfer** | **0.950** | **0.950** | **0.950** | **0.950** |

## Deliverables

| File | Description |
|------|-------------|
| `dataset/` / `dataset.zip` | Collected images (`healthy/`, `diseased/`) |
| `Plant_Disease_Classification.ipynb` | Full experiment notebook |
| `Report_Plant_Disease_Detection.docx` | 3–5 page report with figures |
| `train_and_evaluate.py` | End-to-end training script |
| `figures/` | Plots (samples, augmentation, CM, curves) |
| `outputs/metrics.json` | Numeric results |

## How to run

```bash
cd ComputerVisionProject
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Option A — script (saves figures + metrics)
python train_and_evaluate.py

# Option B — notebook
jupyter notebook Plant_Disease_Classification.ipynb
```

ResNet18 ImageNet weights should be at:
`.cache/torch/hub/checkpoints/resnet18-f37072fd.pth`
(downloaded automatically if SSL works; otherwise download with curl).
