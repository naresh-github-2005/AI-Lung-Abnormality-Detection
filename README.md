# VinBigData Chest X-ray Abnormality Detection (Streamlit)

Streamlit app for VinBigData chest X-ray abnormality detection using a hybrid pipeline:
- Binary classification (normal vs abnormal)
- Multi-label classification (14 disease classes + No finding)
- Object detection (YOLOv8 at two input sizes)
- Fusion with refinement, NMS, and optional Grad-CAM

Live demo:
- https://ai-lung-abnormality-detection-mzihfjtfrnfa8yrktwpwgd.streamlit.app/

Datasets:
- Original competition: https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection
- Preprocessed dataset: https://www.kaggle.com/datasets/naresh26032005/vinbigdata-prep

## Architecture

```mermaid
flowchart LR
  U[User] --> UI[Streamlit UI (app.py)]
  UI --> IN[Upload image(s)]
  IN --> QC[Heuristic X-ray check (utils.py)]
  QC -- invalid --> R0[Invalid response]
  QC -- valid --> HP[Hybrid inference (fusion.py)]

  HP --> B[ResNet50 binary TTA]
  HP --> M[ResNet50 multi-label TTA]
  HP --> Y640[YOLOv8 640]
  HP --> Y1024[YOLOv8 1024]

  Y640 --> F[WBF + refine + NMS]
  Y1024 --> F
  B --> F
  M --> F

  F --> OUT[Detections + class probabilities]
  OUT --> VIS[Boxes + tables]
  OUT --> CAM[Grad-CAM (optional)]
  OUT --> XLS[Excel report (optional)]
```

## Repository structure

- app.py: Streamlit UI, uploads, inference orchestration, results display, Excel export
- fusion.py: Model loading, TTA, YOLO fusion, refinement, NMS, and Grad-CAM
- utils.py: Image IO (PNG/JPG/DICOM), preprocessing, ROI crop, utilities, drawing
- requirements.txt: Python dependencies
- models/: Trained weights and thresholds used at inference
- final_fusion_output/: Offline fusion outputs and metrics from final experiments
- Training Notebooks/: Training and experimentation notebooks

## Key features

- Multi-image batch upload
- Invalid/normal/abnormal triage
- Bounding boxes for detected regions
- Grad-CAM heatmaps on abnormal cases
- Excel export (summary, detections, class probabilities)
- CPU/GPU support (CUDA if available)

## How inference works (high level)

1. Read image (PNG/JPG/BMP or DICOM) and standardize to grayscale.
2. Heuristic validation to filter non X-ray images.
3. TTA binary classification for normal vs abnormal.
4. TTA multi-label classification for class probabilities.
5. YOLOv8 runs at 640 and 1024; results fused via WBF.
6. Per-box refinement using cropped multi-label predictions.
7. Classwise NMS and final selection.
8. Render boxes, show probabilities, and optional Grad-CAM.

## Setup

1. Create a virtual environment (recommended).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure the following files exist under models/:
   - resnet50_binary_best.pth
   - resnet50_multilabel_best.pth
   - best_class_thresholds_tta.npy
   - best_binary_threshold.txt
   - best_yolov8m_640.pt
   - yolov8m_1024_final.pt

## Run locally

```bash
streamlit run app.py
```

## Outputs

- final_fusion_output/fusion_metrics.json: final metrics summary
- final_fusion_output/fusion_summary.json: final experiment summary
- final_fusion_output/fusion_submission.csv: competition submission file
- final_fusion_output/final_val_predictions.csv: validation predictions

## Training notebooks

Notebooks in Training Notebooks/ cover:
- Data preparation
- YOLOv8 training at 640 and 1024
- ResNet50 binary and multi-label training
- Final fusion experiments

## Notes

- DICOM handling uses VOI LUT when present.
- If CUDA is available, torch/YOLO will use GPU automatically.
- Grad-CAM is computed from the ResNet50 backbone last conv block.
