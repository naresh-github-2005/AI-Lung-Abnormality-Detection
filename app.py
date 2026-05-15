from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

from utils import (
    read_uploaded_image,
    is_probably_xray,
    draw_detections,
    sanitize_sheet_name,
)
from fusion import (
    load_all_models,
    hybrid_predict,
    generate_gradcam_overlay,
    CLASS_NAMES,
    NUM_CLASSES,
    NO_FINDING_ID,
)

st.set_page_config(
    page_title="VinBigData X-ray Predictor",
    page_icon="🩻",
    layout="wide",
)

st.title("🩻 VinBigData Chest X-ray Predictor")
st.caption("Multiple uploads • INVALID / NORMAL / ABNORMAL • Boxes • Grad-CAM • Excel export")


@st.cache_resource(show_spinner=True)
def get_resources():
    return load_all_models()


resources = get_resources()

binary_model = resources["binary_model"]
multi_model = resources["multi_model"]
yolo640 = resources["yolo640"]
yolo1024 = resources["yolo1024"]
class_thresholds = resources["class_thresholds"]
best_binary_threshold = resources["best_binary_threshold"]
yolo_device = resources["yolo_device"]

st.sidebar.header("Inference controls")
conf640 = st.sidebar.slider("YOLO640 confidence", 0.01, 0.50, 0.18, 0.01)
conf1024 = st.sidebar.slider("YOLO1024 confidence", 0.01, 0.50, 0.15, 0.01)
max_refine_boxes = st.sidebar.slider("Max boxes to refine", 1, 30, 12, 1)
show_gradcam = st.sidebar.checkbox("Show Grad-CAM", value=True)
create_excel = st.sidebar.checkbox("Create Excel report", value=True)

st.sidebar.write(f"Binary threshold: `{best_binary_threshold:.3f}`")


def build_excel_bytes(results):

    from io import BytesIO
    import pandas as pd

    summary_rows = []
    det_rows = []
    prob_rows = []

    for res in results:

        image_id = res["image_id"]

        global_probs = res["global_probs"]

        top_cls = int(np.argmax(global_probs[:14]))

        top_conf = float(global_probs[top_cls])

        summary_rows.append({
            "image_id": image_id,
            "status": res["status"],
            "message": res["message"],
            "binary_prob": float(res["binary_prob"]),
            "top_class": CLASS_NAMES[top_cls],
            "top_class_conf": top_conf,
            "num_detections": len(res["detections"]),
        })

        # FIXED DETECTIONS LOOP
        for rank, d in enumerate(
            sorted(
                res["detections"],
                key=lambda x: x["conf"],
                reverse=True
            ),
            start=1
        ):

            x1, y1, x2, y2 = d["bbox"]

            det_rows.append({
                "image_id": image_id,
                "rank": rank,
                "class_name": d["class_name"],
                "confidence": float(d["conf"]),
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
            })

        # CLASS PROBABILITIES
        probs_temp = []

        for cls_id in range(NUM_CLASSES):

            probs_temp.append({
                "image_id": image_id,
                "class_name": CLASS_NAMES[cls_id],
                "confidence": float(global_probs[cls_id]),
            })

        probs_temp = sorted(
            probs_temp,
            key=lambda x: x["confidence"],
            reverse=True
        )

        for rank, row in enumerate(probs_temp, start=1):

            row["rank"] = rank

            prob_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    det_df = pd.DataFrame(det_rows)

    prob_df = pd.DataFrame(prob_rows)

    bio = BytesIO()

    with pd.ExcelWriter(
        bio,
        engine="openpyxl"
    ) as writer:

        summary_df.to_excel(
            writer,
            index=False,
            sheet_name="Summary"
        )

        det_df.to_excel(
            writer,
            index=False,
            sheet_name="Detections"
        )

        prob_df.to_excel(
            writer,
            index=False,
            sheet_name="Class_Probabilities"
        )

    bio.seek(0)

    return bio.getvalue()


uploaded_files = st.file_uploader(
    "Upload one or more chest X-ray images",
    type=["png", "jpg", "jpeg", "bmp", "dcm", "dicom"],
    accept_multiple_files=True,
)

if uploaded_files:
    results = []
    invalid_count = 0
    normal_count = 0
    abnormal_count = 0

    progress = st.progress(0)
    for idx, uploaded in enumerate(uploaded_files, start=1):
        try:
            rgb, gray, kind = read_uploaded_image(uploaded)
            valid, valid_score, valid_reason = is_probably_xray(rgb, gray)

            if not valid:
                result = {
                    "image_id": uploaded.name,
                    "status": "INVALID",
                    "message": f"INVALID — not like a chest X-ray ({valid_reason})",
                    "detections": [],
                    "raw": gray,
                    "binary_prob": 0.0,
                    "global_probs": np.zeros(NUM_CLASSES, dtype=np.float32),
                    "gradcam": None,
                    "display_rgb": rgb,
                }
                invalid_count += 1
            else:
                result = hybrid_predict(
                    gray,
                    binary_model=binary_model,
                    multi_model=multi_model,
                    yolo640=yolo640,
                    yolo1024=yolo1024,
                    class_thresholds=class_thresholds,
                    best_binary_threshold=best_binary_threshold,
                    conf640=conf640,
                    conf1024=conf1024,
                    max_refine_boxes=max_refine_boxes,
                    yolo_device=yolo_device,
                )
                result["image_id"] = uploaded.name
                result["display_rgb"] = rgb
                result["gradcam"] = None

                if result["status"] == "NORMAL":
                    normal_count += 1
                else:
                    abnormal_count += 1

                if show_gradcam and result["status"] == "ABNORMAL" and len(result["detections"]) > 0:
                    try:
                        top_cls = int(np.argmax(result["global_probs"][:14]))
                        result["gradcam"] = generate_gradcam_overlay(gray, multi_model, top_cls)
                    except Exception:
                        result["gradcam"] = None

            results.append(result)

        except Exception as e:
            results.append({
                "image_id": uploaded.name,
                "status": "INVALID",
                "message": f"INVALID — could not read file ({e})",
                "detections": [],
                "raw": None,
                "binary_prob": 0.0,
                "global_probs": np.zeros(NUM_CLASSES, dtype=np.float32),
                "gradcam": None,
                "display_rgb": None,
            })
            invalid_count += 1

        progress.progress(idx / len(uploaded_files))

    st.subheader("Batch summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Invalid", invalid_count)
    c2.metric("Normal", normal_count)
    c3.metric("Abnormal", abnormal_count)

    if create_excel:
        excel_bytes = build_excel_bytes(results)
        st.download_button(
            label="Download Excel report",
            data=excel_bytes,
            file_name="xray_predictions.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    for res in results:
        with st.expander(f"{res['image_id']} — {res['status']}", expanded=False):
            st.write(res["message"])

            left, right = st.columns(2)
            if res["display_rgb"] is not None:
                if len(res["detections"]) > 0:
                    annotated = draw_detections(res["display_rgb"], res["detections"])
                else:
                    annotated = res["display_rgb"]
                left.image(annotated, caption="Predicted image with boxes", use_container_width=True)

            if res["gradcam"] is not None:
                right.image(res["gradcam"], caption="Grad-CAM", use_container_width=True)
            else:
                right.info("Grad-CAM not shown for this image.")

            if res["status"] == "INVALID":
                st.error("INVALID image")
            elif res["status"] == "NORMAL":
                st.success("NORMAL — no abnormalities detected")
            else:
                st.warning("ABNORMAL — detections found")

            dets = sorted(res["detections"], key=lambda x: x["conf"], reverse=True)
            det_df = pd.DataFrame([
                {
                    "rank": i + 1,
                    "class_name": d["class_name"],
                    "confidence": float(d["conf"]),
                    "x1": float(d["bbox"][0]),
                    "y1": float(d["bbox"][1]),
                    "x2": float(d["bbox"][2]),
                    "y2": float(d["bbox"][3]),
                }
                for i, d in enumerate(dets)
            ])
            st.dataframe(det_df, use_container_width=True)

            probs = res["global_probs"]
            prob_df = pd.DataFrame([
                {"class_name": CLASS_NAMES[i], "confidence": float(probs[i])}
                for i in range(NUM_CLASSES)
            ]).sort_values("confidence", ascending=False)

            st.markdown("### Class probabilities")
            st.dataframe(prob_df, use_container_width=True)

else:
    st.info("Upload one or more X-ray images to start prediction.")