import re
from io import BytesIO
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np
import pydicom
import torch
from pydicom.pixel_data_handlers.util import apply_voi_lut


def ensure_uint8(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    img = img - img.min()
    mx = img.max()
    if mx > 0:
        img = img / mx
    return (img * 255.0).clip(0, 255).astype(np.uint8)


def read_dicom_from_bytes(file_bytes: bytes) -> np.ndarray:
    ds = pydicom.dcmread(BytesIO(file_bytes))
    img = ds.pixel_array
    try:
        img = apply_voi_lut(img, ds)
    except Exception:
        pass

    img = img.astype(np.float32)

    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        img = np.max(img) - img

    if np.ptp(img) > 0:
        img = (img - img.min()) / (img.max() - img.min())
    else:
        img = np.zeros_like(img, dtype=np.float32)

    return (img * 255.0).clip(0, 255).astype(np.uint8)


def read_uploaded_image(uploaded_file) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Returns:
        rgb_display, gray_image, kind
    """
    name = uploaded_file.name.lower()
    raw = uploaded_file.getvalue()

    if name.endswith((".dcm", ".dicom")):
        gray = read_dicom_from_bytes(raw)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        return rgb, gray, "dicom"

    buf = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Unable to decode image")

    if img.ndim == 2:
        gray = ensure_uint8(img)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        return rgb, gray, "image"

    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return rgb, ensure_uint8(gray), "image"


def entropy_gray(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    p = hist / (hist.sum() + 1e-9)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def colorfulness_rgb(img: np.ndarray) -> float:
    if img.ndim != 3 or img.shape[2] < 3:
        return 0.0
    b, g, r = cv2.split(img.astype(np.float32))
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)
    return float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))


def is_probably_xray(img_rgb: np.ndarray, img_gray: np.ndarray) -> Tuple[bool, float, str]:
    """
    Heuristic only. Returns (is_xray_like, score, reason)
    """
    if img_gray is None or img_gray.size == 0:
        return False, 0.0, "empty image"

    h, w = img_gray.shape[:2]
    if min(h, w) < 128:
        return False, 0.0, "too small"

    score = 0.0
    reasons = []

    if img_rgb.ndim == 2:
        score += 0.25
        reasons.append("grayscale")
    else:
        cf = colorfulness_rgb(img_rgb)
        if cf < 15:
            score += 0.20
            reasons.append("low colorfulness")
        else:
            score -= 0.25
            reasons.append("colorful")

    gray = ensure_uint8(img_gray)
    e = entropy_gray(gray)
    if 4.5 <= e <= 7.8:
        score += 0.20
        reasons.append("good entropy")
    else:
        score -= 0.10
        reasons.append("bad entropy")

    p5, p95 = np.percentile(gray, [5, 95])
    contrast = float(p95 - p5)
    if contrast >= 35:
        score += 0.20
        reasons.append("good contrast")
    else:
        score -= 0.15
        reasons.append("low contrast")

    thr = np.percentile(gray, 25)
    body = gray > thr
    body_ratio = float(body.mean())
    if 0.15 <= body_ratio <= 0.95:
        score += 0.20
        reasons.append("body ratio ok")
    else:
        score -= 0.25
        reasons.append("body ratio odd")

    ch, cw = h // 4, w // 4
    center = gray[ch:3 * ch, cw:3 * cw]
    if center.size > 0:
        c_ratio = float((center > np.percentile(gray, 30)).mean())
        if 0.20 <= c_ratio <= 0.95:
            score += 0.10
            reasons.append("center ok")

    return (score >= 0.15), float(score), ", ".join(reasons)


def chest_roi_crop(img: np.ndarray, pad: float = 0.06) -> np.ndarray:
    thr = max(5, int(np.percentile(img, 5)))
    mask = (img > thr).astype(np.uint8) * 255
    coords = cv2.findNonZero(mask)
    if coords is None:
        return img

    x, y, w, h = cv2.boundingRect(coords)
    H, W = img.shape[:2]

    x1 = max(0, int(x - pad * w))
    y1 = max(0, int(y - pad * h))
    x2 = min(W, int(x + w + pad * w))
    y2 = min(H, int(y + h + pad * h))

    crop = img[y1:y2, x1:x2]
    return crop if crop.size > 0 else img


def lung_window(img: np.ndarray) -> np.ndarray:
    lo = np.percentile(img, 2)
    hi = np.percentile(img, 70)
    out = np.clip((img - lo) / (hi - lo + 1e-6), 0, 1)
    return (out * 255).astype(np.uint8)


def mediastinal_window(img: np.ndarray) -> np.ndarray:
    lo = np.percentile(img, 10)
    hi = np.percentile(img, 90)
    out = np.clip((img - lo) / (hi - lo + 1e-6), 0, 1)
    return (out * 255).astype(np.uint8)


def detail_window(img: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)


def cxr_to_3window(img_gray: np.ndarray) -> np.ndarray:
    img_gray = ensure_uint8(img_gray)
    return np.stack(
        [lung_window(img_gray), mediastinal_window(img_gray), detail_window(img_gray)],
        axis=-1
    )


def preprocess_tta_views(gray_img: np.ndarray, img_size=224, use_roi=True):
    img = ensure_uint8(gray_img)
    if use_roi:
        img = chest_roi_crop(img)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return [img, cv2.flip(img, 1)]


def iou_xyxy(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - inter + 1e-9
    return inter / union


def overlay_heatmap_on_image(img_rgb: np.ndarray, cam: np.ndarray, alpha=0.40) -> np.ndarray:
    heatmap = (cam * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = cv2.resize(heatmap, (img_rgb.shape[1], img_rgb.shape[0]))
    out = cv2.addWeighted(img_rgb, 1 - alpha, heatmap, alpha, 0)
    return out


def draw_detections(img_rgb: np.ndarray, detections):
    img = img_rgb.copy()
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    rng = np.random.default_rng(42)
    class_colors = {i: tuple(int(x) for x in rng.integers(40, 255, size=3)) for i in range(14)}

    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        cls_id = int(det["class_id"])
        conf = float(det["conf"])
        color = class_colors.get(cls_id, (0, 255, 0))
        label = f"{det['class_name']} {conf:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_text = max(0, y1 - th - baseline)
        cv2.rectangle(img, (x1, y_text), (x1 + tw, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return img


def format_submission_string(detections):
    if not detections:
        return "14 1 0 0 1 1"
    parts = []
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        parts.extend([
            str(int(d["class_id"])),
            f"{float(d['conf']):.4f}",
            f"{float(x1):.1f}",
            f"{float(y1):.1f}",
            f"{float(x2):.1f}",
            f"{float(y2):.1f}",
        ])
    return " ".join(parts)


def sanitize_sheet_name(name: str) -> str:
    name = re.sub(r"[\[\]\:\*\?\/\\]", "_", name)
    return name[:31] if len(name) > 31 else name


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        logits = self.model(input_tensor)
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())
        score = logits[:, class_idx].sum()
        self.model.zero_grad(set_to_none=True)
        score.backward(retain_graph=True)

        grads = self.gradients
        acts = self.activations
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam, logits.detach().cpu()