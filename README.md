# CV Assignment 2 - Plant disease: SAM2 leaf ROI + ResNet18 + Grad-CAM + Gradio demo

Continuation of A1 (healthy vs diseased leaves). One SOTA component is added: **SAM2** segments the leaf, the
background is removed and ResNet18 (the A1 recipe) classifies the ROI. The app shows prediction, SAM2 mask,
Grad-CAM, latency and environment.

```
cv_a2/
  dataset/                
  a1/                    
  src/
    segmenter.py           
    segment_sam2.py        
    run_experiments.py     
    eval_iou.py          
    make_figures.py        
    benchmark_latency.py   
    app.py / pipeline.py  
  notebooks/A2_Colab.ipynb 
  report/build_report.py   
```

## Fastest way: Colab 
1. Upload `CV_A2_PlantDisease.zip` to Colab (Runtime -> **T4 GPU**), open `notebooks/A2_Colab.ipynb` or upload it.
2. Run the cells top to bottom (install -> SAM2 masks -> IoU masks -> experiments -> figures/latency -> app -> report).
3. Run the app cell, open the `gradio.live` link, test 3-4 leaves, take a screenshot ->
   `report/assets/app_screenshot.png`.
4. Rebuild the report with your repo / video / app links, open in Word, fix red `[...]` placeholders, export **PDF**.

## Local run 
```bash
python -m venv .venv && source .venv/bin/activate        
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/sam2.git   
python src/segment_sam2.py --data dataset --out outputs/seg      #
python src/run_experiments.py
python src/make_figures.py && python src/benchmark_latency.py
python src/app.py                                                
python report/build_report.py --repo-url ... --video-url ... --app-url ...
```
SAM2.1-small weights are downloaded automatically from Hugging Face on first run.

## IoU masks 
```bash
python src/eval_iou.py --prepare 14      
# paint leaf = white, background = black, save PNG with the same file name into gt_masks/<class>/
python src/eval_iou.py                  
```

## Notes on honesty / what to say at the defense
- The test set is 40 images (1 image = 2.5 pp). Differences of a few points are noise; the report says so.
  The variant used in the app is chosen by **validation** F1.
- If SAM2 does not beat the raw baseline, report it as is; the ablation, IoU, Grad-CAM and app are the graded parts.
- Synthetic share is 0 % (no generated media). You chose one SOTA method (SAM2), which the assignment allows.
- MedSAM2 is SAM2 fine-tuned on medical data; for leaves the general SAM2.1 weights are used.
- A1 inconsistency: `report.docx` from A1 describes Keras/MobileNetV2, the A1 code and metrics are PyTorch/ResNet18.
  The A2 baseline uses the PyTorch numbers (`a1/metrics.json`).
