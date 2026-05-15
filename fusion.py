from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from ultralytics import YOLO

from utils import (
    ensure_uint8,
    chest_roi_crop,
    cxr_to_3window,
    preprocess_tta_views,
    iou_xyxy,
    GradCAM,
)

try:
    from ensemble_boxes import weighted_boxes_fusion
    HAS_WBF = True
except Exception:
    HAS_WBF = False


CLASS_NAMES = [
    "Aortic enlargement",
    "Atelectasis",
    "Calcification",
    "Cardiomegaly",
    "Consolidation",
    "ILD",
    "Infiltration",
    "Lung Opacity",
    "Nodule/Mass",
    "Other lesion",
    "Pleural effusion",
    "Pleural thickening",
    "Pneumothorax",
    "Pulmonary fibrosis",
    "No finding",
]
NUM_CLASSES = len(CLASS_NAMES)
NO_FINDING_ID = 14

RARE_CLASS_IDS = {2, 8, 9, 11, 12, 13}
FOCUS_CLASS_IDS = {1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}


def find_model_file(root: Optional[Path], exact_name: Optional[str] = None, contains: Optional[str] = None, suffix: Optional[str] = None) -> Optional[Path]:
    if root is None or not root.exists():
        return None
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if exact_name is not None and p.name != exact_name:
            continue
        if contains is not None and contains.lower() not in p.name.lower():
            continue
        if suffix is not None and p.suffix.lower() != suffix.lower():
            continue
        return p
    return None


def get_model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cpu")


class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes=1, pretrained=False, dropout=0.25):
        super().__init__()
        try:
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            backbone = models.resnet50(weights=weights)
        except Exception:
            backbone = models.resnet50(pretrained=pretrained)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)


class ResNet50Binary(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.model = ResNet50Classifier(num_classes=1, pretrained=pretrained)

    def forward(self, x):
        return self.model(x).squeeze(1)


class ResNet50MultiLabel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, pretrained=False):
        super().__init__()
        self.model = ResNet50Classifier(num_classes=num_classes, pretrained=pretrained)

    def forward(self, x):
        return self.model(x)


def load_binary_model(path: Path, device: torch.device):
    m = ResNet50Binary(pretrained=False).to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    return m


def load_multi_model(path: Path, device: torch.device):
    m = ResNet50MultiLabel(pretrained=False).to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    return m


def load_all_models(models_root: Optional[Path] = None) -> Dict[str, Any]:
    """
    Searches recursively for:
      - resnet50_binary_best.pth
      - resnet50_multilabel_best.pth
      - best_class_thresholds_tta.npy
      - best_binary_threshold.txt
      - best_yolov8m_640.pt
      - yolov8m_1024_final.pt
    """
    if models_root is None:
        models_root = Path(__file__).resolve().parent / "models"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo_device = 0 if torch.cuda.is_available() else "cpu"

    binary_path = find_model_file(models_root, exact_name="resnet50_binary_best.pth", suffix=".pth")
    if binary_path is None:
        binary_path = find_model_file(models_root, contains="resnet50_binary_best", suffix=".pth")

    multi_path = find_model_file(models_root, exact_name="resnet50_multilabel_best.pth", suffix=".pth")
    if multi_path is None:
        multi_path = find_model_file(models_root, contains="resnet50_multilabel_best", suffix=".pth")

    thresh_path = find_model_file(models_root, exact_name="best_class_thresholds_tta.npy", suffix=".npy")
    if thresh_path is None:
        thresh_path = find_model_file(models_root, contains="best_class_thresholds_tta", suffix=".npy")

    bin_thr_path = find_model_file(models_root, exact_name="best_binary_threshold.txt", suffix=".txt")
    if bin_thr_path is None:
        bin_thr_path = find_model_file(models_root, contains="best_binary_threshold", suffix=".txt")

    yolo640_path = find_model_file(models_root, exact_name="best_yolov8m_640.pt", suffix=".pt")
    if yolo640_path is None:
        yolo640_path = find_model_file(models_root, contains="best_yolov8m_640", suffix=".pt")

    yolo1024_path = find_model_file(models_root, exact_name="yolov8m_1024_final.pt", suffix=".pt")
    if yolo1024_path is None:
        yolo1024_path = find_model_file(models_root, contains="yolov8m_1024_final", suffix=".pt")

    if not all([binary_path, multi_path, thresh_path, bin_thr_path, yolo640_path, yolo1024_path]):
        raise FileNotFoundError(
            f"Missing one or more model files in {models_root}. "
            "Need resnet models, thresholds, YOLO640 best.pt, YOLO1024 final.pt."
        )

    binary_model = load_binary_model(binary_path, device)
    multi_model = load_multi_model(multi_path, device)
    yolo640 = YOLO(str(yolo640_path))
    yolo1024 = YOLO(str(yolo1024_path))
    class_thresholds = np.load(str(thresh_path))
    if len(class_thresholds) != NUM_CLASSES:
        class_thresholds = np.full(NUM_CLASSES, 0.5, dtype=np.float32)

    try:
        best_binary_threshold = float(Path(bin_thr_path).read_text().strip())
    except Exception:
        best_binary_threshold = 0.5

    return {
        "device": device,
        "yolo_device": yolo_device,
        "binary_model": binary_model,
        "multi_model": multi_model,
        "yolo640": yolo640,
        "yolo1024": yolo1024,
        "class_thresholds": class_thresholds,
        "best_binary_threshold": best_binary_threshold,
        "paths": {
            "binary": str(binary_path),
            "multi": str(multi_path),
            "thresh": str(thresh_path),
            "bin_thr": str(bin_thr_path),
            "yolo640": str(yolo640_path),
            "yolo1024": str(yolo1024_path),
        },
    }


def tta_predict_binary(gray_img: np.ndarray, binary_model, device) -> float:
    views = preprocess_tta_views(gray_img, img_size=224, use_roi=True)
    probs = []
    with torch.no_grad():
        for view in views:
            img3 = cxr_to_3window(view)
            x = torch.from_numpy(img3).permute(2, 0, 1).float() / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            x = ((x - mean) / std).unsqueeze(0).to(device)
            logit = binary_model(x)
            probs.append(torch.sigmoid(logit).item())
    return float(np.mean(probs))


def tta_predict_multilabel(gray_img: np.ndarray, multi_model, device) -> np.ndarray:
    views = preprocess_tta_views(gray_img, img_size=224, use_roi=True)
    probs_list = []
    with torch.no_grad():
        for view in views:
            img3 = cxr_to_3window(view)
            x = torch.from_numpy(img3).permute(2, 0, 1).float() / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            x = ((x - mean) / std).unsqueeze(0).to(device)
            logits = multi_model(x)
            probs_list.append(torch.sigmoid(logits).squeeze(0).cpu().numpy())
    return np.mean(probs_list, axis=0)


def yolo_predict_single(model, img_bgr, imgsz=640, conf=0.2, iou=0.45, device=None):
    result = model.predict(source=img_bgr, imgsz=imgsz, conf=conf, iou=iou, verbose=False, device=device)[0]
    dets = []
    if result.boxes is None or len(result.boxes) == 0:
        return dets
    for b in result.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        dets.append({
            "bbox": [x1, y1, x2, y2],
            "conf": float(b.conf[0].item()),
            "class_id": int(b.cls[0].item()),
        })
    return dets


def flip_boxes_back(dets, width):
    out = []
    for d in dets:
        x1, y1, x2, y2 = d["bbox"]
        out.append({
            "bbox": [width - x2, y1, width - x1, y2],
            "conf": d["conf"],
            "class_id": d["class_id"],
        })
    return out


def yolo_tta_dets(model, gray_img, imgsz=640, conf=0.2, iou=0.45, device=None):
    img_bgr = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    H, W = gray_img.shape[:2]

    dets_all = []
    dets_all.extend(yolo_predict_single(model, img_bgr, imgsz=imgsz, conf=conf, iou=iou, device=device))

    flip_bgr = cv2.flip(img_bgr, 1)
    flip_dets = yolo_predict_single(model, flip_bgr, imgsz=imgsz, conf=conf, iou=iou, device=device)
    dets_all.extend(flip_boxes_back(flip_dets, W))

    return dets_all, (H, W)


def run_wbf(boxes_list, scores_list, labels_list, iou_thr=0.55, skip_box_thr=0.0001, weights=None):
    if HAS_WBF:
        boxes, scores, labels = weighted_boxes_fusion(
            boxes_list,
            scores_list,
            labels_list,
            weights=weights,
            iou_thr=iou_thr,
            skip_box_thr=skip_box_thr,
            conf_type="avg",
        )
        return boxes, scores, labels

    all_boxes, all_scores, all_labels = [], [], []
    for b, s, l in zip(boxes_list, scores_list, labels_list):
        all_boxes.extend(b)
        all_scores.extend(s)
        all_labels.extend(l)
    return all_boxes, all_scores, all_labels


def fuse_two_yolos(gray_img, yolo640, yolo1024, conf640=0.10, conf1024=0.08, device=None):
    H, W = gray_img.shape[:2]
    dets640, _ = yolo_tta_dets(yolo640, gray_img, imgsz=640, conf=conf640, iou=0.45, device=device)
    dets1024, _ = yolo_tta_dets(yolo1024, gray_img, imgsz=1024, conf=conf1024, iou=0.45, device=device)

    boxes_list, scores_list, labels_list = [], [], []

    for dets in [dets640, dets1024]:
        boxes, scores, labels = [], [], []
        for d in dets:
            x1, y1, x2, y2 = d["bbox"]
            boxes.append([x1 / W, y1 / H, x2 / W, y2 / H])
            scores.append(d["conf"])
            labels.append(d["class_id"])
        boxes_list.append(boxes)
        scores_list.append(scores)
        labels_list.append(labels)

    weights = [1.0, 1.15]
    fused_boxes, fused_scores, fused_labels = run_wbf(
        boxes_list,
        scores_list,
        labels_list,
        iou_thr=0.55,
        skip_box_thr=0.0001,
        weights=weights,
    )

    dets = []
    for box, score, label in zip(fused_boxes, fused_scores, fused_labels):
        dets.append({
            "bbox": [box[0] * W, box[1] * H, box[2] * W, box[3] * H],
            "conf": float(score),
            "class_id": int(label),
        })

    return dets, (H, W)


def crop_from_bbox(img_gray, bbox, pad=0.08):
    H, W = img_gray.shape[:2]
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    x1 = int(max(0, x1 - pad * bw))
    y1 = int(max(0, y1 - pad * bh))
    x2 = int(min(W - 1, x2 + pad * bw))
    y2 = int(min(H - 1, y2 + pad * bh))
    crop = img_gray[y1:y2, x1:x2]
    return crop if crop.size > 0 else img_gray


def crop_probs_two_stage(crop_gray, multi_model, device):
    crop_gray = ensure_uint8(crop_gray)
    crop_gray = cv2.resize(crop_gray, (224, 224), interpolation=cv2.INTER_AREA)

    views = [crop_gray, cv2.flip(crop_gray, 1)]
    probs_list = []

    with torch.no_grad():
        for view in views:
            img3 = cxr_to_3window(view)
            x = torch.from_numpy(img3).permute(2, 0, 1).float() / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            x = ((x - mean) / std).unsqueeze(0).to(device)
            logits = multi_model(x)
            probs_list.append(torch.sigmoid(logits).squeeze(0).cpu().numpy())

    return np.mean(probs_list, axis=0)


def adaptive_class_thresholds(base_thresholds, binary_prob, global_probs):
    base = np.array(base_thresholds, dtype=np.float32).copy()

    rescue_floor = np.array([
        0.22,  # Aortic enlargement
        0.12,  # Atelectasis
        0.10,  # Calcification
        0.22,  # Cardiomegaly
        0.10,  # Consolidation
        0.10,  # ILD
        0.10,  # Infiltration
        0.10,  # Lung Opacity
        0.10,  # Nodule/Mass
        0.10,  # Other lesion
        0.12,  # Pleural effusion
        0.10,  # Pleural thickening
        0.10,  # Pneumothorax
        0.10,  # Pulmonary fibrosis
        0.65,  # No finding
    ], dtype=np.float32)

    thr = 0.55 * base + 0.45 * rescue_floor

    normal_prob = float(global_probs[NO_FINDING_ID])
    abnormal_prob = float(binary_prob)

    if normal_prob > 0.80 and abnormal_prob < 0.35:
        thr[:14] += 0.04
    elif abnormal_prob > 0.65 or normal_prob < 0.55:
        thr[:14] -= 0.03

    for c in RARE_CLASS_IDS:
        thr[c] -= 0.02

    thr[:14] = np.clip(thr[:14], 0.05, 0.55)
    thr[NO_FINDING_ID] = np.clip(thr[NO_FINDING_ID], 0.50, 0.85)
    return thr


def refine_detection(gray_img, det, binary_prob, global_probs, multi_model, device, base_thresholds):
    crop1 = crop_from_bbox(gray_img, det["bbox"], pad=0.06)
    crop2 = crop_from_bbox(gray_img, det["bbox"], pad=0.16)

    probs1 = crop_probs_two_stage(crop1, multi_model, device)
    probs2 = crop_probs_two_stage(crop2, multi_model, device)
    probs = 0.65 * probs1 + 0.35 * probs2

    yolo_cls = int(det["class_id"])
    yolo_conf = float(det["conf"])
    global_top = int(np.argmax(global_probs[:14]))
    adaptive_thr = adaptive_class_thresholds(base_thresholds, binary_prob, global_probs)

    candidates = [c for c in range(14) if probs[c] >= adaptive_thr[c]]

    if len(candidates) > 0:
        def candidate_score(c):
            score = (
                0.50 * probs[c] +
                0.25 * global_probs[c] +
                0.25 * (1.0 if c == yolo_cls else 0.0)
            )
            if c in RARE_CLASS_IDS:
                score *= 1.08
            if c == global_top:
                score *= 1.04
            return score

        cls_id = max(candidates, key=candidate_score)
        conf = float(candidate_score(cls_id))
    else:
        crop_top = int(np.argmax(probs[:14]))
        crop_top_prob = float(probs[crop_top])

        if probs[yolo_cls] >= max(0.08, adaptive_thr[yolo_cls] * 0.75):
            cls_id = yolo_cls
            conf = float(0.55 * yolo_conf + 0.45 * probs[yolo_cls])
        elif crop_top_prob >= 0.10:
            cls_id = crop_top
            conf = float(0.55 * yolo_conf + 0.45 * crop_top_prob)
        else:
            return None

    normal_prob = float(global_probs[NO_FINDING_ID])
    if normal_prob > 0.82 and binary_prob < 0.40 and conf < 0.25:
        return None

    conf = 0.45 * yolo_conf + 0.35 * float(probs[cls_id]) + 0.20 * float(global_probs[cls_id])
    if cls_id == yolo_cls:
        conf *= 1.05
    if cls_id == global_top:
        conf *= 1.04
    if cls_id in RARE_CLASS_IDS:
        conf *= 1.08
    if cls_id in FOCUS_CLASS_IDS:
        conf *= 1.02
    if normal_prob > 0.78:
        conf *= 0.92
    elif binary_prob > 0.65:
        conf *= 1.05

    min_keep = adaptive_thr[cls_id] * 0.75
    if conf < min_keep:
        return None

    return {
        "bbox": det["bbox"],
        "class_id": cls_id,
        "class_name": CLASS_NAMES[cls_id],
        "conf": float(conf),
    }


def class_iou_thr(cls_id):
    if cls_id in {2, 8, 9, 11, 12, 13}:
        return 0.35
    if cls_id in {1, 4, 5, 6, 7, 10}:
        return 0.40
    return 0.42


def filter_by_classwise_nms(dets):
    if len(dets) == 0:
        return dets
    dets = sorted(dets, key=lambda x: x["conf"], reverse=True)
    keep = []
    for d in dets:
        skip = False
        for k in keep:
            if d["class_id"] == k["class_id"] and iou_xyxy(d["bbox"], k["bbox"]) >= class_iou_thr(d["class_id"]):
                skip = True
                break
        if not skip:
            keep.append(d)
    return keep


def soft_nms_like_filter(preds, iou_thr=0.4):
    if len(preds) == 0:
        return preds
    preds = sorted(preds, key=lambda x: x[4], reverse=True)
    keep = []
    while preds:
        best = preds.pop(0)
        keep.append(best)
        remain = []
        for p in preds:
            if p[5] != best[5] or iou_xyxy(p[:4], best[:4]) < iou_thr:
                remain.append(p)
        preds = remain
    return keep


def hybrid_predict(
    gray_img: np.ndarray,
    binary_model,
    multi_model,
    yolo640,
    yolo1024,
    class_thresholds,
    best_binary_threshold: float,
    conf640=0.18,
    conf1024=0.15,
    max_refine_boxes=12,
    yolo_device=None,
):
    raw = ensure_uint8(gray_img)

    device = get_model_device(binary_model)
    binary_prob = tta_predict_binary(raw, binary_model, device)
    global_probs = tta_predict_multilabel(raw, multi_model, device)
    normal_prob = float(global_probs[NO_FINDING_ID])

    if normal_prob >= class_thresholds[NO_FINDING_ID] and binary_prob < best_binary_threshold:
        return {
            "valid": True,
            "message": "NORMAL — No abnormalities detected",
            "detections": [],
            "raw": raw,
            "binary_prob": binary_prob,
            "global_probs": global_probs,
            "status": "NORMAL",
        }

    dets, _ = fuse_two_yolos(raw, yolo640, yolo1024, conf640=conf640, conf1024=conf1024, device=yolo_device)
    dets = sorted(dets, key=lambda x: x["conf"], reverse=True)[:max_refine_boxes]

    refined = []
    for d in dets:
        r = refine_detection(raw, d, binary_prob, global_probs, multi_model, device, class_thresholds)
        if r is not None:
            refined.append(r)

    refined = filter_by_classwise_nms(refined)
    refined_xyxy = [r["bbox"] + [r["conf"], r["class_id"]] for r in refined]
    refined_xyxy = soft_nms_like_filter(refined_xyxy, iou_thr=0.40)
    refined = [
        {"bbox": r[:4], "conf": float(r[4]), "class_id": int(r[5]), "class_name": CLASS_NAMES[int(r[5])]}
        for r in refined_xyxy
    ]

    if len(refined) == 0 and (binary_prob > 0.60 or normal_prob < 0.50) and len(dets) > 0:
        d0 = dets[0]
        cls_id = int(d0["class_id"])
        if cls_id != NO_FINDING_ID:
            refined.append({
                "bbox": d0["bbox"],
                "conf": float(d0["conf"]),
                "class_id": cls_id,
                "class_name": CLASS_NAMES[cls_id],
            })

    refined = sorted(refined, key=lambda x: x["conf"], reverse=True)[:10]

    if len(refined) == 0:
        msg = "NORMAL — No abnormalities detected" if normal_prob >= 0.5 else "No confident disease found"
        status = "NORMAL"
    else:
        msg = f"Detected {len(refined)} abnormal region(s)"
        status = "ABNORMAL"

    return {
        "valid": True,
        "message": msg,
        "detections": refined,
        "raw": raw,
        "binary_prob": binary_prob,
        "global_probs": global_probs,
        "status": status,
    }


def generate_gradcam_overlay(gray_img: np.ndarray, multi_model, class_idx: int):

    import torch
    import cv2
    import numpy as np

    device = get_model_device(multi_model)

    roi = chest_roi_crop(ensure_uint8(gray_img))
    roi = cv2.resize(roi, (224, 224), interpolation=cv2.INTER_AREA)

    img3 = cxr_to_3window(roi)

    x = torch.from_numpy(img3).permute(2, 0, 1).float() / 255.0

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    x = ((x - mean) / std).unsqueeze(0).to(device)

    # FIXED TARGET LAYER
    target_layer = multi_model.model.backbone.layer4[-1].conv3

    activations = []
    gradients = []

    def forward_hook(module, inp, out):
        activations.append(out)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    multi_model.eval()

    logits = multi_model(x)

    score = logits[:, class_idx].sum()

    multi_model.zero_grad()

    score.backward(retain_graph=True)

    acts = activations[0]
    grads = gradients[0]

    weights = grads.mean(dim=(2, 3), keepdim=True)

    cam = (weights * acts).sum(dim=1)

    cam = torch.relu(cam)

    cam = cam.squeeze().detach().cpu().numpy()

    cam = cv2.resize(cam, (224, 224))

    cam = cam - cam.min()

    if cam.max() > 0:
        cam = cam / cam.max()

    heatmap = np.uint8(255 * cam)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)

    overlay = cv2.addWeighted(
        roi_rgb,
        0.6,
        heatmap,
        0.4,
        0
    )

    overlay = cv2.resize(
        overlay,
        (gray_img.shape[1], gray_img.shape[0])
    )

    fh.remove()
    bh.remove()

    return overlay