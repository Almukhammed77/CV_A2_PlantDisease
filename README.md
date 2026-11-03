# CV Assignment 2 - Plant disease: SAM2 leaf ROI + ResNet18 + Grad-CAM + Gradio demo

Continuation of A1 (healthy vs diseased leaves). One SOTA component is added: **SAM2** segments the leaf, the
background is removed and ResNet18 (the A1 recipe) classifies the ROI. The app shows prediction, SAM2 mask,
Grad-CAM, latency and environment.

```
cv_a2/
  dataset/                 A1 images (diseased 100, healthy 98)
  a1/                      A1 metrics.json + train script (baseline: ResNet18 F1 0.950, 40 test images)
  src/
    segmenter.py           SAM2 (point prompts) + GrabCut baseline/fallback
    segment_sam2.py        masks for the whole dataset  -> outputs/seg/
    run_experiments.py     raw vs masked vs crop, 3 seeds -> outputs/metrics_a2.json, figures, weights
    eval_iou.py            IoU/Dice vs ~14 hand-drawn masks (SAM2 and GrabCut)
    make_figures.py        Grad-CAM example figure
    benchmark_latency.py   avg latency + environment -> outputs/latency.json
    app.py / pipeline.py   Gradio demo
  notebooks/A2_Colab.ipynb all steps for Colab (T4 GPU)
  report/build_report.py   fills the 8-page report (.docx) with your real numbers and figures
```

## Fastest way: Colab (about 30-40 min including the manual masks)
1. Upload `CV_A2_PlantDisease.zip` to Colab (Runtime -> **T4 GPU**), open `notebooks/A2_Colab.ipynb` or upload it.
2. Run the cells top to bottom (install -> SAM2 masks -> IoU masks -> experiments -> figures/latency -> app -> report).
3. Run the app cell, open the `gradio.live` link, test 3-4 leaves, take a screenshot ->
   `report/assets/app_screenshot.png`.
4. Rebuild the report with your repo / video / app links, open in Word, fix red `[...]` placeholders, export **PDF**.

## Local run (CPU works, slower)
```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/sam2.git   # needs Python>=3.10, torch>=2.5
python src/segment_sam2.py --data dataset --out outputs/seg      # add --method grabcut to test without SAM2
python src/run_experiments.py
python src/make_figures.py && python src/benchmark_latency.py
python src/app.py                                                # http://127.0.0.1:7860 (--share for a public link)
python report/build_report.py --repo-url ... --video-url ... --app-url ...
```
SAM2.1-small weights are downloaded automatically from Hugging Face on first run.

## IoU masks (needed for the "IoU" metric, ~15 min)
```bash
python src/eval_iou.py --prepare 14      # copies 14 images to gt_todo/
# paint leaf = white, background = black, save PNG with the same file name into gt_masks/<class>/
python src/eval_iou.py                   # IoU/Dice SAM2 vs GrabCut -> outputs/iou_results.json
```

## Notes on honesty / what to say at the defense
- The test set is 40 images (1 image = 2.5 pp). Differences of a few points are noise; the report says so.
  The variant used in the app is chosen by **validation** F1.
- If SAM2 does not beat the raw baseline, report it as is; the ablation, IoU, Grad-CAM and app are the graded parts.
- Synthetic share is 0 % (no generated media). You chose one SOTA method (SAM2), which the assignment allows.
- MedSAM2 is SAM2 fine-tuned on medical data; for leaves the general SAM2.1 weights are used.
- A1 inconsistency: `report.docx` from A1 describes Keras/MobileNetV2, the A1 code and metrics are PyTorch/ResNet18.
  The A2 baseline uses the PyTorch numbers (`a1/metrics.json`).

## Demo video (3-5 min) outline
0:00 topic + A1 baseline | 0:40 what SAM2 is (diagram from the report) | 1:20 live app: healthy leaf, diseased leaf,
a failure case; point at mask, Grad-CAM, latency | 2:50 results table + ablation + IoU | 3:40 ethics/limits |
4:10 repo + how to run.

## Submission checklist (Moodle, due 7 Nov 2026 23:59)
- [ ] Report PDF (6-8 pages) - [ ] App link / run guide - [ ] Repo link (push this folder; `.gitignore` skips weights)
- [ ] Video link - [ ] Be ready to run the app and explain every step at the defense.
