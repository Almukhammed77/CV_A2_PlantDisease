#!/usr/bin/env python3
"""Build the Assignment 2 report (.docx) from the experiment outputs.

Run after the experiments:   python report/build_report.py --repo-url ... --video-url ... --app-url ...
Anything that has not been produced yet is replaced by a RED placeholder, so the file
always builds; re-run it after each step. Export to PDF from Word for Moodle.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
ASSETS = Path(__file__).resolve().parent / "assets"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NEUROHIVE = "https://neurohive.io/en/state-of-the-art/medsam2-open-source-sota-3d-medical-image-and-video-segmentation-model/"


def jload(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


class R:
    def __init__(self, out_path: Path, args):
        self.doc = Document()
        self.out_path, self.args = out_path, args
        s = self.doc.sections[0]
        s.page_height, s.page_width = Cm(29.7), Cm(21.0)
        s.top_margin = s.bottom_margin = Cm(2.0)
        s.left_margin = s.right_margin = Cm(2.2)
        st = self.doc.styles["Normal"]
        st.font.name, st.font.size = "Calibri", Pt(11)
        self._page_number(s)
        self.fig_n = 0
        self.tab_n = 0

    # ---- helpers
    def _page_number(self, section):
        p = section.footer.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run()
        for t, txt in (("begin", None), (None, "PAGE"), ("end", None)):
            if t:
                el = OxmlElement("w:fldChar")
                el.set(qn("w:fldCharType"), t)
            else:
                el = OxmlElement("w:instrText")
                el.set(qn("xml:space"), "preserve")
                el.text = txt
            run._r.append(el)

    def h(self, text, level=1):
        self.doc.add_heading(text, level=level)

    def p(self, text="", bold=False, italic=False, size=None, align=None, after=6):
        para = self.doc.add_paragraph()
        r = para.add_run(text)
        r.bold, r.italic = bold, italic
        if size:
            r.font.size = Pt(size)
        if align:
            para.alignment = align
        para.paragraph_format.space_after = Pt(after)
        return para

    def todo(self, text):
        para = self.doc.add_paragraph()
        r = para.add_run(f"[{text}]")
        r.bold = True
        r.font.color.rgb = RGBColor(0xC0, 0x00, 0x00)
        return para

    def bullets(self, items):
        for it in items:
            para = self.doc.add_paragraph(style="List Bullet")
            para.add_run(it)
            para.paragraph_format.space_after = Pt(2)

    def fig(self, path: Path, caption: str, width_cm: float = 14.0):
        if not path.exists():
            self.todo(f"Missing figure {path.name} - run the corresponding step and rebuild the report")
            return
        self.fig_n += 1
        self.doc.add_picture(str(path), width=Cm(width_cm))
        self.doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        self.p(f"Figure {self.fig_n}. {caption}", italic=True, size=9.5, align=WD_ALIGN_PARAGRAPH.CENTER)

    def table(self, header, rows, caption=None, widths=None):
        self.tab_n += 1
        if caption:
            self.p(f"Table {self.tab_n}. {caption}", italic=True, size=9.5, after=2)
        t = self.doc.add_table(rows=1, cols=len(header))
        t.style = "Light Grid Accent 1"
        for i, hd in enumerate(header):
            t.rows[0].cells[i].text = hd
            for r in t.rows[0].cells[i].paragraphs[0].runs:
                r.bold = True
        for row in rows:
            cells = t.add_row().cells
            for i, v in enumerate(row):
                cells[i].text = str(v)
        for row in t.rows:
            for c in row.cells:
                for para in c.paragraphs:
                    para.paragraph_format.space_after = Pt(0)
                    for r in para.runs:
                        r.font.size = Pt(9.5)
        self.doc.add_paragraph().paragraph_format.space_after = Pt(2)

    def save(self):
        self.doc.save(self.out_path)
        print("Wrote", self.out_path)


def count_images(data: Path):
    out = {}
    for c in ("diseased", "healthy"):
        d = data / c
        out[c] = sum(1 for f in d.iterdir() if f.suffix.lower() in IMG_EXT and not f.name.startswith(".")) if d.exists() else None
    return out


def pm(mu, sd):
    return f"{mu:.3f} ± {sd:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs"))
    ap.add_argument("--data", default=str(ROOT / "dataset"))
    ap.add_argument("--file", default=str(ROOT / "Report_A2_Plant_Disease_SAM2.docx"))
    ap.add_argument("--repo-url", default=None)
    ap.add_argument("--video-url", default=None)
    ap.add_argument("--app-url", default=None)
    args = ap.parse_args()

    out = Path(args.out)
    fig = out / "figures"
    seg_fig = out / "seg" / "figures"
    M = jload(out / "metrics_a2.json")
    SEG = jload(out / "seg" / "seg_stats.json")
    IOU = jload(out / "iou_results.json")
    LAT = jload(out / "latency.json")
    A1 = jload(ROOT / "a1" / "metrics.json")
    a1 = A1["models"]["resnet18_transfer"] if A1 else None
    counts = count_images(Path(args.data))

    r = R(Path(args.file), args)

    # ------------------------------------------------------------------ title
    r.p("Astana IT University - Computer Vision", bold=True, size=12, align=WD_ALIGN_PARAGRAPH.CENTER, after=2)
    r.p("Assignment 2 (Continuation): Level-Up Your A1 Topic with SOTA CV + Demo", bold=True, size=16,
        align=WD_ALIGN_PARAGRAPH.CENTER, after=2)
    r.p("Plant disease detection: SAM2 leaf-ROI segmentation, ResNet18 and Grad-CAM", size=13,
        align=WD_ALIGN_PARAGRAPH.CENTER, after=2)
    r.p("Student: Telman Zhalgas Yerlanuly (SE-2411)   |   Instructor: Nurym Kaiyrkhan", italic=True,
        align=WD_ALIGN_PARAGRAPH.CENTER, after=2)
    links = []
    for name, val in (("Repository", args.repo_url), ("Demo video", args.video_url), ("Live app", args.app_url)):
        links.append(f"{name}: {val}" if val else f"{name}: [add link]")
    r.p("   |   ".join(links), size=9.5, align=WD_ALIGN_PARAGRAPH.CENTER, after=8)

    # ------------------------------------------------------------------ 1
    r.h("1. Introduction and goal")
    r.p("In Assignment 1 (A1) I built a binary classifier that separates healthy from diseased plant leaves on a "
        "self-collected dataset. The strongest A1 model was an ImageNet-pretrained ResNet18 fine-tuned in two stages"
        + (f", which reached accuracy {a1['accuracy']:.3f} and F1 {a1['f1']:.3f} on a 40-image test set."
           if a1 else "."))
    r.p("A1 classified the whole photo, so the network also sees pots, hands, windows and other plants. In Assignment 2 "
        "I upgrade the same topic with one state-of-the-art component: Segment Anything Model 2 (SAM2) is used to "
        "extract the leaf region of interest (ROI) before classification. The goal is to (i) integrate SAM2 into the "
        "pipeline, (ii) compare against the A1 baseline with an ablation, (iii) explain predictions with Grad-CAM and "
        "masks, and (iv) ship an interactive demo that reports latency.")

    # ------------------------------------------------------------------ 2
    r.h("2. Dataset update")
    nd, nh = counts["diseased"], counts["healthy"]
    total = (nd + nh) if (nd is not None and nh is not None) else (M["split"]["total"] if M else None)
    r.p(f"The A1 dataset is reused unchanged: {total if total else '198'} valid RGB photos "
        f"({nd if nd is not None else '100'} diseased, {nh if nh is not None else '98'} healthy; two stray .html files "
        "in the original folder were removed). Photos are real, self-collected or downloaded leaf images with very "
        "different backgrounds, sizes and lighting. The A1 stratified split is kept so that the test set is identical "
        f"to A1: train {M['split']['train'] if M else 126}, validation {M['split']['val'] if M else 32}, "
        f"test {M['split']['test'] if M else 40}.")
    r.p("Synthetic share: 0 %. No generated images or videos are used; the optional generative components of the "
        "assignment (TA-TiTok/MaskGen, ArtAug, Step-Video-T2V) were deliberately not used because the task allows one "
        "SOTA method and time was limited. Augmentation is unchanged from A1: random resized crop, horizontal/vertical "
        "flips, rotation ±25°, colour jitter. The only new data artefacts are the SAM2 masks and ROI crops.")
    if SEG:
        r.p(f"SAM2 segmentation statistics ({SEG['method']}): {SEG['n_images']} images, mean mask area "
            f"{SEG['mean_area_fraction']*100:.1f} % of the frame, {SEG['n_fallback_full_frame']} image(s) with no "
            f"plausible mask (the full frame is kept for those), mean segmentation time {SEG['mean_time_s']*1000:.0f} ms/image.")
    else:
        r.todo("Segmentation statistics missing - run src/segment_sam2.py")
    r.fig(ASSETS / "class_distribution.png", "Class distribution of the A1 dataset (unchanged).", 8.0)

    # ------------------------------------------------------------------ 3
    r.h("3. SOTA method: SAM2 for leaf ROI segmentation")
    r.p("What it is. Segment Anything Model 2 (SAM2, Ravi et al., Meta AI, 2024) is a promptable foundation model for "
        "segmentation. An image encoder (Hiera vision transformer) turns the photo into features; a prompt encoder "
        "embeds user hints (points, boxes or masks); a lightweight mask decoder combines both and returns several "
        "candidate masks with predicted IoU scores. For video, a memory attention module propagates masks across "
        "frames. MedSAM2 (Ma et al., 2025) is SAM2 fine-tuned on a large medical 3D/video dataset to segment organs "
        "and lesions; its description on Neurohive is the page this assignment points to. Because MedSAM2 is "
        "specialised for medical modalities and leaves are natural images, I use the general SAM2.1 weights that "
        "MedSAM2 is built on.")
    r.fig(ASSETS / "sam2_pipeline.png", "(a) SAM2 in a nutshell, (b) the ROI pipeline used in this work.", 15.5)
    r.p("Why it helps this topic. Disease symptoms (spots, necrosis, yellowing) live on the leaf, while the photo "
        "background (soil, pots, hands, other plants) is a nuisance that can also leak class information in a small "
        "dataset. Segmenting the leaf removes that nuisance, so the classifier can focus on lesion texture and colour, "
        "and the mask is itself an interpretable output for the user.")
    r.p("How it is used. For every image SAM2 is prompted with three positive points near the centre and four "
        "negative points in the corners (no training needed). Of the three candidate masks the one with the highest "
        "predicted IoU and a plausible area (5-95 % of the frame) is kept, cleaned (largest connected component, holes "
        "filled) and applied in three input variants that form the ablation: raw (A1 input), masked (background "
        "painted grey) and crop (background removed + bounding-box crop).")
    r.fig(seg_fig / "mask_examples.png", "SAM2 masks and ROI crops on example images.", 9.5)
    r.p("Sources: Neurohive, \"MedSAM2: Open Source SOTA 3D Medical Image and Video Segmentation Model\" - " + NEUROHIVE,
        size=9.5, italic=True)

    # ------------------------------------------------------------------ 4
    r.h("4. Experiments")
    r.p("Setup. ResNet18 (ImageNet weights) with the A1 recipe: stage 1 trains the new head (6 epochs, lr 1e-3), stage "
        "2 fine-tunes layer4 + head (up to 12 epochs, lr 1e-4), Adam, weight decay 1e-4, ReduceLROnPlateau, early "
        "stopping on validation loss. Each variant is trained with several random seeds on the same A1 split and "
        "reported as mean ± std. The variant used in the app is chosen by validation F1, not by test F1. Precision, "
        "recall and F1 are weighted averages, as in A1.")
    if M:
        S = M["summary"]
        rows = []
        if a1:
            rows.append(["A1 ResNet18 (reported, 1 run)", f"{a1['accuracy']:.3f}", f"{a1['precision']:.3f}",
                         f"{a1['recall']:.3f}", f"{a1['f1']:.3f}", "-"])
        names = {"raw": "A2 raw (A1 input, re-trained)", "masked": "A2 SAM2 masked (bg removed)",
                 "crop": "A2 SAM2 crop (bg removed + crop)"}
        for s in M["runs"]:
            rows.append([names[s], pm(S[s]["test_accuracy_mean"], S[s]["test_accuracy_std"]),
                         pm(S[s]["test_precision_mean"], S[s]["test_precision_std"]),
                         pm(S[s]["test_recall_mean"], S[s]["test_recall_std"]),
                         pm(S[s]["test_f1_mean"], S[s]["test_f1_std"]),
                         f"{S[s]['val_f1_mean']:.3f}"])
        r.table(["Model / input", "Test acc", "Test precision", "Test recall", "Test F1", "Val F1 (mean)"], rows,
                f"Baseline vs SAM2 variants on the 40-image test set (mean ± std over seeds {M['seeds']}).")
        raw, crop = S.get("raw"), S.get("crop")
        if raw and crop:
            d = crop["test_f1_mean"] - raw["test_f1_mean"]
            noise = max(raw["test_f1_std"], crop["test_f1_std"], 0.025)
            if abs(d) <= noise:
                verdict = ("no reliable difference: the change is within seed-to-seed noise and within one test image "
                           "(1 image = 2.5 percentage points)")
            elif d > 0:
                verdict = "an improvement larger than the seed-to-seed noise"
            else:
                verdict = "a drop larger than the seed-to-seed noise"
            r.p(f"Result. The SAM2 ROI crop changes the mean test F1 by {d:+.3f} relative to the re-trained raw baseline "
                f"({crop['test_f1_mean']:.3f} vs {raw['test_f1_mean']:.3f}); I interpret this as {verdict}. "
                f"The variant selected by validation F1 is '{M['best_setting_by_val_f1']}'. With only 40 test images, "
                "differences of a few points are not statistically meaningful, so these numbers should be read as "
                "indicative.")
        r.fig(fig / "ablation_f1.png", "Ablation: effect of the SAM2 input variant on test F1 (error bars: std over seeds).", 9.0)
        r.p("Ablation. The ablation compares three inputs with everything else fixed: raw photo, SAM2-masked photo (same "
            "framing, background removed) and SAM2 crop (background removed and zoomed on the leaf). It separates the "
            "effect of removing the background from the effect of zooming in.")
        r.fig(fig / f"history_{M['best_setting_by_val_f1']}.png",
              f"Training curves of the selected variant ('{M['best_setting_by_val_f1']}', first seed).", 12.0)
    else:
        r.todo("Metrics missing - run src/run_experiments.py (table, ablation and training curves appear here)")
    if IOU:
        r.p(f"Mask quality. On {IOU['n_images']} images with hand-drawn masks, SAM2 reaches IoU "
            f"{IOU['sam2_iou_mean']:.3f} ± {IOU['sam2_iou_std']:.3f} (Dice {IOU['sam2_dice_mean']:.3f}) versus "
            f"{IOU['grabcut_iou_mean']:.3f} ± {IOU['grabcut_iou_std']:.3f} (Dice {IOU['grabcut_dice_mean']:.3f}) for a "
            "classical GrabCut baseline.")
        r.fig(fig / "iou_examples.png", "Manual ground truth vs SAM2 and GrabCut masks.", 11.0)
    else:
        r.todo("IoU missing - annotate ~14 masks and run src/eval_iou.py")

    # ------------------------------------------------------------------ 5
    r.h("5. Explainability")
    r.p("Two complementary explanations are shown to the user: the SAM2 mask (what part of the photo is treated as the "
        "leaf) and a Grad-CAM heat-map of the last ResNet block (which regions of the ROI drive the decision). A "
        "reliable diseased prediction should highlight lesions rather than edges or leftover background.")
    r.fig(fig / "gradcam_examples.png", "Image | SAM2 mask | model input | Grad-CAM on test images (wrong predictions are marked in red).", 13.0)
    if M:
        b = M["best_setting_by_val_f1"]
        r.fig(fig / f"cm_{b}.png", f"Confusion matrix of the selected variant ('{b}', first seed, test set).", 6.0)

    # ------------------------------------------------------------------ 6
    r.h("6. Demo application and latency")
    r.p("The demo is a Gradio web app (src/app.py): the user uploads a leaf photo or uses the camera; the app shows the "
        "predicted class with probabilities, the SAM2 mask overlay, the ROI actually fed to the network and the "
        "Grad-CAM overlay, together with the latency of the request and the running average, and the environment "
        "(OS, Python, PyTorch, device). Heavy components run on the server, so there is no on-device model.")
    shot = ASSETS / "app_screenshot.png"
    if shot.exists():
        r.fig(shot, "Screenshot of the demo application.", 14.0)
    else:
        r.todo("Insert a screenshot of the running app as report/assets/app_screenshot.png and rebuild")
    if LAT:
        r.table(["Environment", "Segmenter", "Seg. (ms)", "Classify + Grad-CAM (ms)", "Total (ms)", "FPS"],
                [[LAT["environment"], LAT["segmenter"], f"{LAT['avg_seg_ms']:.0f}", f"{LAT['avg_cls_gradcam_ms']:.0f}",
                  f"{LAT['avg_total_ms']:.0f}", f"{LAT['fps']:.1f}"]],
                f"Average end-to-end latency over {LAT['n_images']} test images (after warm-up).")
    else:
        r.todo("Latency missing - run src/benchmark_latency.py")

    # ------------------------------------------------------------------ 7
    r.h("7. Ethics and limitations")
    r.bullets([
        "Data and consent: the dataset contains photos of plants only. No faces or people are the subject of any image; "
        "photos with hands in the frame, if any, are incidental. The app asks users not to upload photos of people, "
        "and uploads are processed in memory and not stored.",
        "Synthetic media: none is generated or used (synthetic share 0 %), so no watermarking is required; if "
        "generated leaves are added in future work they must be labelled as synthetic.",
        "Bias and coverage: the dataset is small (~200 photos), binary, and covers a few plant species and "
        "backgrounds; accuracy will drop on other species, camera types or lighting. The model detects visible "
        "damage, not pathogens, and must not replace an agronomist.",
        "Statistical limits: the test set has 40 images, so one image is worth 2.5 points; multi-seed means reduce "
        "training noise but not test-set noise.",
        "Segmentation failure modes: SAM2 may select a neighbouring leaf, a pot or the whole plant; in that case the "
        "classifier receives a wrong ROI. Frames with no plausible mask fall back to the full image.",
        "Misuse: automated results could be over-trusted in food-safety or insurance contexts; the app labels itself "
        "as a demo.",
        "Licences: SAM2 is released under Apache 2.0; ResNet18 weights come from torchvision (ImageNet).",
    ])

    # ------------------------------------------------------------------ 8
    r.h("8. Reproducibility")
    r.p("Environment: Python 3.10+, PyTorch ≥ 2.5, SAM2 (pip install git+https://github.com/facebookresearch/sam2.git); "
        "weights (SAM2.1-small) are downloaded automatically from Hugging Face. A GPU (Colab T4) is recommended for "
        "SAM2 and training; the app also runs on CPU.")
    for line in ["pip install -r requirements.txt  (+ SAM2, see README)",
                 "python src/segment_sam2.py --data dataset --out outputs/seg",
                 "python src/eval_iou.py --prepare 14   ->   draw masks into gt_masks/   ->   python src/eval_iou.py",
                 "python src/run_experiments.py",
                 "python src/make_figures.py && python src/benchmark_latency.py",
                 "python src/app.py --share",
                 "python report/build_report.py"]:
        para = r.p(line, size=9.5, after=1)
        for run in para.runs:
            run.font.name = "Consolas"
    r.p("Input: dataset/diseased and dataset/healthy (JPEG). Outputs: masks, metrics_a2.json, weights "
        "(resnet18_*.pt), figures and latency.json under outputs/. Random seeds are listed in Table 1.")

    # ------------------------------------------------------------------ 9
    r.h("9. References")
    r.bullets([
        "N. Ravi et al., \"SAM 2: Segment Anything in Images and Videos\", Meta AI, arXiv:2408.00714, 2024.",
        "J. Ma et al., \"MedSAM2: Segment Anything in 3D Medical Images and Videos\", arXiv:2504.03600, 2025.",
        "Neurohive: MedSAM2 - " + NEUROHIVE,
        "K. He et al., \"Deep Residual Learning for Image Recognition\", CVPR 2016.",
        "R. Selvaraju et al., \"Grad-CAM: Visual Explanations from Deep Networks\", ICCV 2017.",
        "C. Rother et al., \"GrabCut: Interactive Foreground Extraction using Iterated Graph Cuts\", SIGGRAPH 2004.",
    ])
    r.save()


if __name__ == "__main__":
    main()
