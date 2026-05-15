# VinBigData Chest X-ray Abnormality Detection (Streamlit)

Streamlit app for VinBigData chest X-ray abnormality detection using a hybrid pipeline:
- Binary classification: normal vs abnormal
- Multi-label classification: 14 disease classes + no finding
- Object detection: YOLOv8 at two input sizes
- Fusion with refinement, NMS, and optional Grad-CAM

## Live demo - [Streamlit demo](https://ai-lung-abnormality-detection-mzihfjtfrnfa8yrktwpwgd.streamlit.app/)

## Datasets

- Original competition dataset: VinBigData Chest X-ray Abnormalities Detection
- Preprocessed dataset: VinBigData prep dataset

## Architecture

```text
User -> Streamlit UI -> Upload image(s)
      -> Heuristic X-ray check
      -> If invalid -> Invalid response
      -> If valid -> Hybrid inference
           -> ResNet50 binary TTA
           -> ResNet50 multilabel TTA
           -> YOLOv8 640
           -> YOLOv8 1024
           -> WBF + refinement + NMS
           -> Final detections, class probabilities, Grad-CAM, Excel report
```

## Repository structure

- `app.py`: Streamlit UI, upload handling, inference orchestration, result display, Excel export
- `fusion.py`: Model loading, TTA, YOLO fusion, refinement, NMS, Grad-CAM, hybrid prediction logic
- `utils.py`: Image I/O for PNG/JPG/BMP/DICOM, preprocessing, ROI crop, visualization utilities
- `requirements.txt`: Python dependencies
- `models/`: Trained weights and thresholds used at inference
- `final_fusion_output/`: Offline fusion outputs and metrics from final experiments
- `Training Notebooks/`: Training and experimentation notebooks

## Notebook guide

### Notebook 1 — Data preparation
Builds the preprocessing dataset from the competition images, converts DICOM files to cached 640px images, creates label files, and writes train/validation splits.

### Notebook 2 — YOLOv8 640
Trains the first detector on full-size cached images. This model acts as the clean, precise anchor detector.

### Notebook 3 — ResNet50 advanced
Trains the binary classifier and multilabel classifier with ROI crops, 3-window preprocessing, TTA, and threshold tuning. This model corrects class confusion and boosts image-level recall.

### Notebook 4A — YOLOv8 1024 stage 1
Trains the higher-resolution detector on 1024px images, ROI crops, and class-balanced sampling to improve small-lesion recall.

### Notebook 4B — YOLOv8 1024 stage 2
Fine-tunes the 1024px detector using hard negative mining and hard positive mining so difficult classes are less likely to collapse into background.

### Notebook 5 — Final fusion
Combines YOLO640, YOLO1024, and the ResNet models with WBF, TTA, crop refinement, and class-aware suppression to produce the final predictions and submission file.

## Why these models were chosen

### ResNet50
- Strong transfer learning backbone for medical image classification
- Efficient enough to train and infer within Kaggle constraints
- Good balance between accuracy and speed
- Works well for binary gating and multilabel refinement

### YOLOv8
- Fast and reliable detector
- Strong support for custom training and augmentation
- Good choice for bounding-box localization of abnormalities
- The 640px version gives cleaner, more precise detections
- The 1024px version helps recover small lesions missed at lower resolution

### Two-stage detector design
- YOLO640 is used as the precise anchor detector
- YOLO1024 is used as the recall booster for tiny and subtle findings
- Fusion of both detectors gives better coverage than one model alone

## Why not only one model

A single classifier cannot localize lesions.
A single detector at one scale usually misses either small lesions or precise boundaries.
A hybrid ensemble gives:
- better recall
- better localization
- fewer false positives
- better robustness on rare classes

## How inference works

1. Read image (PNG/JPG/BMP or DICOM) and standardize to grayscale.
2. Heuristic validation filters obvious non-X-ray images.
3. ResNet50 binary TTA estimates normal vs abnormal probability.
4. ResNet50 multilabel TTA estimates disease probabilities.
5. YOLOv8 runs at 640 and 1024.
6. Outputs are fused with Weighted Box Fusion.
7. Each box is refined using cropped multilabel predictions.
8. Class-wise NMS and thresholding produce the final detections.
9. The app renders boxes, tables, Grad-CAM, and Excel output.

## Setup

1. Create a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure the following files exist under `models/`:
   - `resnet50_binary_best.pth`
   - `resnet50_multilabel_best.pth`
   - `best_class_thresholds_tta.npy`
   - `best_binary_threshold.txt`
   - `best_yolov8m_640.pt`
   - `yolov8m_1024_final.pt`

## Run locally

```bash
streamlit run app.py
```

## Outputs

- `final_fusion_output/fusion_metrics.json`: final metrics summary
- `final_fusion_output/fusion_summary.json`: final experiment summary
- `final_fusion_output/fusion_submission.csv`: competition submission file
- `final_fusion_output/final_val_predictions.csv`: validation predictions

## Streamlit UI features

- Batch upload of multiple images
- Invalid / normal / abnormal triage
- Bounding boxes on detected regions
- Grad-CAM heatmaps for abnormal images
- Excel export with summary, detections, and class probabilities
- CPU/GPU support

## Important implementation notes

- DICOM handling uses VOI LUT when present.
- If CUDA is available, Torch and YOLO will use GPU automatically.
- Grad-CAM is computed from the ResNet50 backbone last convolution block.
- The invalid-image check is heuristic and can later be replaced with a learned invalid-image classifier.

## FAQs

### 1. Why did you choose a hybrid pipeline instead of a single model?
A single model either detects lesions well or classifies well, but not both. The hybrid approach combines detection, classification, and refinement, which improves localization and recall.

### 2. Why use both YOLO640 and YOLO1024?
YOLO640 is cleaner and more precise, while YOLO1024 recovers smaller lesions that the 640 model may miss. Using both improves coverage across lesion sizes.

### 3. Why did you choose ResNet50 for classification?
ResNet50 is a strong pretrained backbone, easy to fine-tune, and efficient enough for Kaggle constraints. It works well for binary gating and multilabel refinement.

### 4. Why not use a deeper model like ResNet101 or EfficientNet-L2?
Deeper models usually need more time and compute. ResNet50 gives a good balance of speed, stability, and accuracy for this pipeline.

### 5. Why use TTA?
Test-time augmentation improves robustness by averaging predictions from the original and flipped image views. It helps stabilize both classification and detection confidence.

### 6. Why use Weighted Box Fusion?
WBF merges overlapping predictions from multiple detectors more intelligently than simple NMS, which helps keep true lesions while reducing duplicate boxes.

### 7. Why is Grad-CAM included?
Grad-CAM adds explainability by showing which regions influenced the classifier's decision. This is useful for medical visualization and debugging.

### 8. How do you handle rare classes?
Rare classes are boosted with ROI crops, hard negative mining, class-balanced sampling, adaptive thresholds, and fusion logic that lowers the chance of missing subtle findings.

### 9. Why do some images get marked as invalid?
The app uses a heuristic front-end filter to reject images that do not look like chest X-rays. This prevents obvious non-medical inputs from going through the pipeline.

### 10. Why use two detector stages?
The first detector focuses on precision, while the second detector focuses on recall for small lesions. Combining them makes the system more balanced.

### 11. What is the biggest challenge in VinBigData?
The hardest problem is that many subtle findings are predicted as background. The solution is multi-scale detection, ROI refinement, and fusion across detectors and classifiers.

### 12. How did you improve recall on background-heavy classes?
By using 1024px detection, ROI crops, hard positive mining, class-aware thresholds, and image-level class priors from the multilabel ResNet.

### 13. Why is the pipeline suitable for deployment?
It separates training from inference, uses cached final weights, supports batch upload, and provides both visualization and export. That makes it practical for a Streamlit front end.

### 14. How would you improve the system further?
I would add a learned invalid-image classifier, lung segmentation, tile-based inference for tiny lesions, and pseudo-labeling for further recall improvement.

## Notes

- The architecture section is intentionally written as plain text instead of Mermaid to avoid rendering issues on GitHub.
- The notebook descriptions are short and focused on the purpose of each stage.
- The interview Q&A is included to help explain the project in demos, reviews, and viva-style questions.

