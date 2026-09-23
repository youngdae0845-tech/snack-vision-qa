import os
import gc

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from rembg import new_session, remove
from supabase import create_client


st.set_page_config(
    page_title="Snack Vision QC",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def get_config(name, default=None):
    try:
        return st.secrets[name]
    except Exception:
        return os.environ.get(name, default)


def get_int_config(name, default):
    value = get_config(name, str(default))
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def get_float_config(name, default):
    value = get_config(name, str(default))
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


SUPABASE_URL = get_config("SUPABASE_URL")
SUPABASE_KEY = get_config("SUPABASE_KEY")

# Accuracy-first defaults for paid Railway resources.
MASK_DIM = get_int_config("MASK_DIM", 2200)
ANALYSIS_DIM = get_int_config("ANALYSIS_DIM", 2600)

# u2net is less likely than isnet-general-use to keep only a few "salient" pieces.
REMBG_MODEL = get_config("REMBG_MODEL", "u2net")

# Low alpha threshold keeps weakly detected snack regions instead of deleting them.
MASK_THRESHOLD = get_int_config("MASK_THRESHOLD", 24)
MASK_KERNEL_SIZE = get_int_config("MASK_KERNEL_SIZE", 5)
MASK_CLOSE_ITER = get_int_config("MASK_CLOSE_ITER", 1)
MASK_ERODE_ITER = get_int_config("MASK_ERODE_ITER", 0)

# The color mask rescues yellow/orange snack pixels when AI segmentation is too selective.
USE_COLOR_RESCUE_MASK = get_config("USE_COLOR_RESCUE_MASK", "true").lower() != "false"
COLOR_MASK_MIN_AREA_RATIO = get_float_config("COLOR_MASK_MIN_AREA_RATIO", 0.02)

REVIEW_THRESHOLD = get_float_config("REVIEW_THRESHOLD", 10.0)
NG_THRESHOLD = get_float_config("NG_THRESHOLD", 20.0)
HEATMAP_MAX_DELTA_E = get_float_config("HEATMAP_MAX_DELTA_E", 30.0)

TARGET_L = get_config("TARGET_L")
TARGET_A = get_config("TARGET_A")
TARGET_B = get_config("TARGET_B")


def get_target_lab():
    if TARGET_L is None or TARGET_A is None or TARGET_B is None:
        return None
    try:
        return np.array([float(TARGET_L), float(TARGET_A), float(TARGET_B)], dtype=np.float32)
    except ValueError:
        return None


@st.cache_resource
def get_supabase_client():
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    return create_client(SUPABASE_URL, SUPABASE_KEY)


@st.cache_resource
def load_ai_model(model_name):
    return new_session(model_name)


def resize_by_max_dim(img, max_dim):
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img

    scale = max_dim / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def normalize_kernel_size(size):
    size = max(3, int(size))
    return size if size % 2 == 1 else size + 1


def largest_components_mask(mask, min_area_ratio=0.0002):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    image_area = mask.shape[0] * mask.shape[1]
    min_area = max(24, int(image_area * min_area_ratio))
    cleaned = np.zeros_like(mask)

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == label] = 255

    return cleaned


def create_snack_color_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    bgr_float = img.astype(np.float32) / 255.0
    lab = cv2.cvtColor(bgr_float, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    a_channel = lab[:, :, 1]
    b_channel = lab[:, :, 2]

    orange_hue = (h >= 3) & (h <= 48) & (s >= 28) & (v >= 35)
    warm_lab = (l_channel >= 18) & (a_channel >= -8) & (b_channel >= 8) & (s >= 18)
    bright_yellow = (h >= 12) & (h <= 58) & (s >= 18) & (v >= 65)

    color_mask = (orange_hue | warm_lab | bright_yellow).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    color_mask = largest_components_mask(color_mask)

    return color_mask


def create_precise_masks(original_img, analysis_img, session):
    mask_input_img = resize_by_max_dim(original_img, MASK_DIM)
    no_bg_img = remove(mask_input_img, session=session)

    if no_bg_img.ndim < 3 or no_bg_img.shape[2] < 4:
        raise ValueError("배경 제거 결과에 alpha 채널이 없습니다.")

    alpha = no_bg_img[:, :, 3].astype(np.uint8)
    resized_alpha = cv2.resize(
        alpha,
        (analysis_img.shape[1], analysis_img.shape[0]),
        interpolation=cv2.INTER_CUBIC,
    )

    _, ai_mask = cv2.threshold(
        resized_alpha,
        MASK_THRESHOLD,
        255,
        cv2.THRESH_BINARY,
    )

    display_mask = ai_mask
    color_rescue_mask = None
    if USE_COLOR_RESCUE_MASK:
        color_rescue_mask = create_snack_color_mask(analysis_img)
        color_area_ratio = np.count_nonzero(color_rescue_mask) / color_rescue_mask.size

        if color_area_ratio >= COLOR_MASK_MIN_AREA_RATIO:
            display_mask = cv2.bitwise_or(display_mask, color_rescue_mask)

    kernel_size = normalize_kernel_size(MASK_KERNEL_SIZE)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    if MASK_CLOSE_ITER > 0:
        display_mask = cv2.morphologyEx(
            display_mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=MASK_CLOSE_ITER,
        )

    display_mask = largest_components_mask(display_mask)

    analysis_mask = display_mask.copy()
    if MASK_ERODE_ITER > 0:
        analysis_mask = cv2.erode(analysis_mask, kernel, iterations=MASK_ERODE_ITER)

    return display_mask, analysis_mask, ai_mask, color_rescue_mask


def calculate_status(mean_delta_e):
    if mean_delta_e >= NG_THRESHOLD:
        return {
            "status": "NG",
            "label": "관리 한계 초과",
            "tone": "danger",
            "message": "색상 편차가 높습니다. 시즈닝 뭉침, 과열, 조명 조건을 확인하세요.",
        }
    if mean_delta_e >= REVIEW_THRESHOLD:
        return {
            "status": "REVIEW",
            "label": "검토 필요",
            "tone": "warning",
            "message": "색상 편차가 기준 근처입니다. LOT 샘플 재확인을 권장합니다.",
        }
    return {
        "status": "OK",
        "label": "정상 범위",
        "tone": "success",
        "message": "색상 균일도가 안정적입니다. 현재 기준에서는 정상으로 판단됩니다.",
    }


def analyze_image(original_img, session):
    analysis_img = resize_by_max_dim(original_img, ANALYSIS_DIM)
    display_mask, analysis_mask, ai_mask, color_rescue_mask = create_precise_masks(
        original_img,
        analysis_img,
        session,
    )

    core_pixels = analysis_mask > 0
    if np.count_nonzero(core_pixels) == 0:
        raise ValueError("스낵 영역을 감지하지 못했습니다.")

    visual_masked_img = cv2.bitwise_and(analysis_img, analysis_img, mask=display_mask)
    core_masked_img = cv2.bitwise_and(analysis_img, analysis_img, mask=analysis_mask)

    ai_masked_img = cv2.bitwise_and(analysis_img, analysis_img, mask=ai_mask)
    if color_rescue_mask is None:
        color_rescue_img = np.zeros_like(analysis_img)
    else:
        color_rescue_img = cv2.bitwise_and(analysis_img, analysis_img, mask=color_rescue_mask)

    bgr_float = analysis_img.astype(np.float32) / 255.0
    lab_img = cv2.cvtColor(bgr_float, cv2.COLOR_BGR2LAB)
    lab_pixels = lab_img[core_pixels]
    bgr_pixels = analysis_img[core_pixels]

    mean_lab = lab_pixels.mean(axis=0)
    mean_bgr = bgr_pixels.mean(axis=0)

    target_lab = get_target_lab()
    if target_lab is None:
        reference_lab = mean_lab
        analysis_mode = "내부 균일도 기준"
    else:
        reference_lab = target_lab
        analysis_mode = "기준 Lab 색상 대비"

    delta_values = np.linalg.norm(lab_pixels - reference_lab, axis=1).astype(np.float32)
    mean_delta_e = float(delta_values.mean()) if delta_values.size else 0.0
    std_delta_e = float(delta_values.std()) if delta_values.size else 0.0
    p95_delta_e = float(np.percentile(delta_values, 95)) if delta_values.size else 0.0

    delta_map = np.zeros(core_pixels.shape, dtype=np.float32)
    delta_map[core_pixels] = delta_values

    heatmap_scale = max(HEATMAP_MAX_DELTA_E, 1.0)
    heatmap_gray = np.clip((delta_map / heatmap_scale) * 255.0, 0, 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_TURBO)
    heatmap_blend = cv2.addWeighted(analysis_img, 0.52, heatmap_bgr, 0.48, 0)
    heatmap_masked = cv2.bitwise_and(heatmap_blend, heatmap_blend, mask=display_mask)

    result = {
        "analysis_img": analysis_img,
        "visual_masked_img": visual_masked_img,
        "core_masked_img": core_masked_img,
        "ai_masked_img": ai_masked_img,
        "color_rescue_img": color_rescue_img,
        "heatmap_masked": heatmap_masked,
        "delta_values": delta_values,
        "mean_bgr": mean_bgr,
        "mean_lab": mean_lab,
        "reference_lab": reference_lab,
        "analysis_mode": analysis_mode,
        "avg_l": float(mean_lab[0]),
        "avg_a": float(mean_lab[1]),
        "avg_b": float(mean_lab[2]),
        "mean_delta_e": mean_delta_e,
        "std_delta_e": std_delta_e,
        "p95_delta_e": p95_delta_e,
        "sample_pixels": int(delta_values.size),
        "mask_area_ratio": float(np.count_nonzero(display_mask) / display_mask.size),
        "ai_area_ratio": float(np.count_nonzero(ai_mask) / ai_mask.size),
        "color_area_ratio": 0.0 if color_rescue_mask is None else float(np.count_nonzero(color_rescue_mask) / color_rescue_mask.size),
    }

    del bgr_float, lab_img, lab_pixels, bgr_pixels, delta_map, heatmap_bgr, heatmap_blend
    del display_mask, analysis_mask, ai_mask, color_rescue_mask, core_pixels
    gc.collect()
    return result


# ---------------------------------------------------------------------------
# Presentation helpers — visual layer only. None of these touch analysis data.
# ---------------------------------------------------------------------------

STATUS_ICON = {"success": "✓", "warning": "!", "danger": "✕"}


def render_kpi(label, value, helper, tone="neutral"):
    st.markdown(
        f"""
        <div class="kpi-card kpi-{tone}">
            <div class="kpi-top">
                <span class="kpi-label">{label}</span>
                <span class="kpi-dot kpi-dot-{tone}"></span>
            </div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-helper">{helper}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_panel(meta, mean_delta_e, std_delta_e, p95_delta_e):
    icon = STATUS_ICON.get(meta["tone"], "•")
    st.markdown(
        f"""
        <div class="status-panel status-{meta["tone"]}">
            <div class="status-top">
                <span class="status-eyebrow">Final QA Decision</span>
                <span class="status-icon">{icon}</span>
            </div>
            <div class="status-main">{meta["status"]}</div>
            <div class="status-label">{meta["label"]}</div>
            <div class="status-copy">{meta["message"]}</div>
            <div class="status-grid">
                <div><span>Mean Delta E</span><strong>{mean_delta_e:.2f}</strong></div>
                <div><span>Std Dev</span><strong>{std_delta_e:.2f}</strong></div>
                <div><span>P95 Delta E</span><strong>{p95_delta_e:.2f}</strong></div>
                <div><span>NG Threshold</span><strong>{NG_THRESHOLD:.1f}</strong></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_color_chip(mean_bgr, avg_l, avg_a, avg_b):
    r = int(np.clip(mean_bgr[2], 0, 255))
    g = int(np.clip(mean_bgr[1], 0, 255))
    b = int(np.clip(mean_bgr[0], 0, 255))

    st.markdown(
        f"""
        <div class="color-chip-card">
            <div class="color-swatch" style="background: rgb({r}, {g}, {b});"></div>
            <div>
                <div class="color-title">Average Product Color</div>
                <div class="color-values">L {avg_l:.1f} &nbsp;/&nbsp; a {avg_a:.1f} &nbsp;/&nbsp; b {avg_b:.1f}</div>
                <div class="color-helper">정밀 분석 마스크 내부 픽셀 기준</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_panel_title(title, caption):
    st.markdown(
        f"""
        <div class="panel-title">
            <div class="panel-heading">{title}</div>
            <div class="panel-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_media_label(title):
    st.markdown(f'<div class="media-label">{title}</div>', unsafe_allow_html=True)


def safe_table_columns(df, columns):
    existing = [column for column in columns if column in df.columns]
    return df[existing] if existing else df


supabase = get_supabase_client()

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

        :root {
            --bg: #f2f3f9;
            --surface: #ffffff;
            --surface-soft: #f8f9fd;
            --border: #e4e7f1;
            --border-strong: #d8dcec;
            --ink: #11132a;
            --ink-secondary: #5b6083;
            --ink-muted: #9296b0;
            --navy: #10162e;
            --navy-soft: #1c2648;
            --brand: #4338ca;
            --brand-strong: #352cad;
            --brand-soft: #edeafc;
            --success: #0ca34a;
            --success-strong: #0a7a38;
            --success-soft: #e6f7ec;
            --warning: #d97706;
            --warning-strong: #b45309;
            --warning-soft: #fef3e2;
            --danger: #dc2626;
            --danger-strong: #b91c1c;
            --danger-soft: #fde9e9;
            --radius-lg: 18px;
            --radius-md: 14px;
            --radius-sm: 10px;
            --shadow-sm: 0 1px 2px rgba(17, 19, 42, 0.04);
            --shadow-md: 0 10px 26px rgba(17, 19, 42, 0.07);
            --shadow-lg: 0 22px 48px rgba(17, 19, 42, 0.10);
        }

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        .stApp {
            background:
                radial-gradient(1100px 420px at 8% -8%, rgba(67, 56, 202, 0.07), transparent 60%),
                var(--bg);
            color: var(--ink);
        }

        .block-container {
            max-width: 1380px;
            padding-top: 1.6rem;
            padding-bottom: 3.5rem;
        }

        #MainMenu, footer, header {
            visibility: hidden;
        }

        h1, h2, h3, h4, h5, strong {
            color: var(--ink);
        }

        /* ---------- Header ---------- */

        .report-header {
            position: relative;
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 28px;
            padding: 30px 32px 26px;
            margin-bottom: 22px;
            background: linear-gradient(180deg, #ffffff 0%, #fbfbfe 100%);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            box-shadow: var(--shadow-lg);
            overflow: hidden;
        }

        .report-header::before {
            content: "";
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 4px;
            background: linear-gradient(90deg, var(--brand) 0%, #7c6ff0 45%, #22c1a1 100%);
        }

        .brand-mark {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            font-size: 12px;
            font-weight: 800;
            color: var(--brand);
            text-transform: uppercase;
            letter-spacing: .09em;
            margin-bottom: 10px;
        }

        .brand-mark::before {
            content: "";
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: var(--brand);
            box-shadow: 0 0 0 4px var(--brand-soft);
        }

        .report-title {
            font-size: 30px;
            line-height: 1.2;
            font-weight: 800;
            color: var(--navy);
            letter-spacing: -0.01em;
            margin-bottom: 10px;
        }

        .report-subtitle {
            font-size: 14px;
            line-height: 1.6;
            color: var(--ink-secondary);
            max-width: 760px;
        }

        .spec-chips {
            display: flex;
            flex-direction: column;
            gap: 8px;
            min-width: 280px;
            padding: 16px 18px;
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            background: var(--surface-soft);
        }

        .spec-chips-title {
            font-size: 11px;
            font-weight: 800;
            color: var(--ink-muted);
            text-transform: uppercase;
            letter-spacing: .08em;
            margin-bottom: 2px;
        }

        .spec-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 12.5px;
            padding: 5px 0;
            border-top: 1px dashed var(--border);
        }

        .spec-row:first-of-type { border-top: none; }

        .spec-row span:first-child {
            color: var(--ink-secondary);
        }

        .spec-row span:last-child {
            font-weight: 700;
            color: var(--navy);
            font-variant-numeric: tabular-nums;
        }

        .spec-pill {
            display: inline-block;
            padding: 2px 9px;
            border-radius: 999px;
            font-size: 11px;
            font-weight: 800;
        }

        .spec-pill-on { background: var(--success-soft); color: var(--success-strong); }
        .spec-pill-off { background: var(--surface); color: var(--ink-muted); border: 1px solid var(--border-strong); }

        /* ---------- Upload zone ---------- */

        .upload-note {
            position: relative;
            background: var(--surface);
            border: 1.5px dashed var(--border-strong);
            border-radius: var(--radius-md);
            padding: 24px 22px;
            margin: 4px 0 18px;
            transition: border-color .15s ease;
        }

        .upload-note strong {
            display: block;
            color: var(--navy);
            font-size: 17px;
            font-weight: 800;
            margin-bottom: 6px;
        }

        .upload-note span {
            color: var(--ink-secondary);
            font-size: 13px;
            line-height: 1.55;
        }

        /* ---------- Section titles / cards ---------- */

        .panel-title {
            margin: 10px 0 14px;
        }

        .panel-heading {
            font-size: 15px;
            font-weight: 800;
            color: var(--navy);
            letter-spacing: -0.01em;
        }

        .panel-caption {
            font-size: 12.5px;
            color: var(--ink-muted);
            margin-top: 3px;
        }

        .section-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            padding: 22px;
            box-shadow: var(--shadow-md);
        }

        .empty-state {
            display: flex;
            flex-direction: column;
            align-items: flex-start;
            gap: 10px;
            min-height: 260px;
            justify-content: center;
        }

        .empty-state-badge {
            width: 46px;
            height: 46px;
            border-radius: 13px;
            background: var(--brand-soft);
            color: var(--brand);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
            font-weight: 800;
            margin-bottom: 6px;
        }

        /* ---------- KPI cards ---------- */

        .kpi-card {
            min-height: 122px;
            background: var(--surface);
            border: 1px solid var(--border);
            border-left: 4px solid var(--border-strong);
            border-radius: var(--radius-md);
            padding: 17px 18px;
            box-shadow: var(--shadow-sm);
            transition: box-shadow .15s ease, transform .15s ease;
        }

        .kpi-card:hover {
            box-shadow: var(--shadow-md);
            transform: translateY(-1px);
        }

        .kpi-success { border-left-color: var(--success); }
        .kpi-warning { border-left-color: var(--warning); }
        .kpi-danger  { border-left-color: var(--danger); }
        .kpi-blue    { border-left-color: var(--brand); }

        .kpi-top {
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .kpi-label {
            font-size: 11.5px;
            color: var(--ink-muted);
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: .05em;
        }

        .kpi-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: var(--border-strong);
        }

        .kpi-dot-success { background: var(--success); }
        .kpi-dot-warning { background: var(--warning); }
        .kpi-dot-danger  { background: var(--danger); }
        .kpi-dot-blue    { background: var(--brand); }

        .kpi-value {
            font-size: 27px;
            line-height: 1.2;
            font-weight: 800;
            color: var(--ink);
            margin-top: 10px;
            letter-spacing: -0.01em;
        }

        .kpi-helper {
            font-size: 12px;
            color: var(--ink-muted);
            margin-top: 8px;
        }

        /* ---------- Status panel ---------- */

        .status-panel {
            position: relative;
            min-height: 330px;
            border-radius: var(--radius-lg);
            padding: 24px 24px 22px;
            color: #ffffff;
            box-shadow: var(--shadow-lg);
            overflow: hidden;
        }

        .status-success { background: linear-gradient(155deg, #0b7a3d 0%, #0a5c30 100%); }
        .status-warning { background: linear-gradient(155deg, #b45309 0%, #8a3e07 100%); }
        .status-danger  { background: linear-gradient(155deg, #c62828 0%, #931f1f 100%); }

        .status-top {
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .status-eyebrow {
            font-size: 11px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: .09em;
            opacity: .8;
        }

        .status-icon {
            width: 30px;
            height: 30px;
            border-radius: 50%;
            background: rgba(255, 255, 255, .16);
            border: 1px solid rgba(255, 255, 255, .28);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 15px;
            font-weight: 800;
        }

        .status-main {
            font-size: 52px;
            line-height: 1;
            font-weight: 900;
            margin-top: 20px;
            letter-spacing: -0.02em;
        }

        .status-label {
            font-size: 17px;
            font-weight: 700;
            margin-top: 8px;
            opacity: .96;
        }

        .status-copy {
            font-size: 13px;
            line-height: 1.6;
            opacity: .88;
            margin-top: 14px;
        }

        .status-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 10px;
            margin-top: 20px;
        }

        .status-grid div {
            padding: 11px 12px;
            background: rgba(255, 255, 255, .12);
            border: 1px solid rgba(255, 255, 255, .16);
            border-radius: 10px;
        }

        .status-grid span {
            display: block;
            font-size: 11px;
            opacity: .8;
        }

        .status-grid strong {
            display: block;
            font-size: 19px;
            margin-top: 4px;
            color: #ffffff;
            font-variant-numeric: tabular-nums;
        }

        /* ---------- Color chip ---------- */

        .color-chip-card {
            display: grid;
            grid-template-columns: 104px 1fr;
            gap: 16px;
            align-items: center;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            padding: 16px;
            min-height: 136px;
            box-shadow: var(--shadow-sm);
        }

        .color-swatch {
            width: 104px;
            height: 104px;
            border-radius: 12px;
            border: 1px solid rgba(17, 19, 42, .10);
            box-shadow: inset 0 0 0 3px rgba(255, 255, 255, .6), var(--shadow-sm);
        }

        .color-title {
            font-size: 12.5px;
            color: var(--ink-muted);
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: .05em;
        }

        .color-values {
            font-size: 20px;
            font-weight: 800;
            color: var(--ink);
            margin-top: 8px;
            font-variant-numeric: tabular-nums;
        }

        .color-helper {
            font-size: 12px;
            color: var(--ink-muted);
            margin-top: 7px;
        }

        /* ---------- Media gallery ---------- */

        .media-label {
            display: inline-flex;
            align-items: center;
            gap: 7px;
            font-size: 12.5px;
            font-weight: 800;
            color: var(--navy);
            margin: 2px 0 8px;
        }

        .media-label::before {
            content: "";
            width: 6px;
            height: 6px;
            border-radius: 2px;
            background: var(--brand);
        }

        [data-testid="stImage"] {
            border-radius: var(--radius-md);
            overflow: hidden;
            border: 1px solid var(--border);
            box-shadow: var(--shadow-sm);
        }

        [data-testid="stImage"] img {
            display: block;
        }

        /* ---------- Tabs ---------- */

        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            margin-bottom: 14px;
            border-bottom: none;
        }

        .stTabs [data-baseweb="tab"] {
            height: 44px;
            padding: 0 20px;
            border-radius: 999px;
            background: var(--surface);
            border: 1px solid var(--border);
            color: var(--ink-secondary);
            font-weight: 700;
            transition: all .15s ease;
        }

        .stTabs [data-baseweb="tab"]:hover {
            border-color: var(--border-strong);
            color: var(--ink);
        }

        .stTabs [aria-selected="true"] {
            color: #ffffff !important;
            background: var(--navy) !important;
            border-color: var(--navy) !important;
            box-shadow: var(--shadow-sm);
        }

        .stTabs [data-baseweb="tab-highlight"],
        .stTabs [data-baseweb="tab-border"] {
            display: none;
        }

        /* ---------- Inputs & buttons ---------- */

        .stTextInput input, .stTextInput > div > div {
            border-radius: var(--radius-sm) !important;
        }

        .stButton > button {
            width: 100%;
            min-height: 44px;
            border-radius: var(--radius-sm);
            border: 1px solid var(--navy);
            background: var(--navy);
            color: #ffffff;
            font-weight: 800;
            letter-spacing: .01em;
            transition: all .15s ease;
        }

        .stButton > button:hover {
            border-color: var(--brand);
            background: var(--brand);
            color: #ffffff;
            transform: translateY(-1px);
            box-shadow: var(--shadow-md);
        }

        .stButton > button:active {
            transform: translateY(0);
        }

        [data-testid="stExpander"] {
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            background: var(--surface);
            box-shadow: var(--shadow-sm);
        }

        [data-testid="stDataFrame"] {
            border-radius: var(--radius-md);
            overflow: hidden;
            border: 1px solid var(--border);
        }

        .stAlert {
            border-radius: var(--radius-sm);
        }

        hr {
            border-color: var(--border);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


target_lab = get_target_lab()
target_text = "이미지 평균" if target_lab is None else f"L {target_lab[0]:.1f} / a {target_lab[1]:.1f} / b {target_lab[2]:.1f}"
color_rescue_pill = (
    '<span class="spec-pill spec-pill-on">ON</span>' if USE_COLOR_RESCUE_MASK
    else '<span class="spec-pill spec-pill-off">OFF</span>'
)

st.markdown(
    f"""
    <div class="report-header">
        <div>
            <div class="brand-mark">Snack Vision QC</div>
            <div class="report-title">고정밀 완제품 색차 분석 리포트</div>
            <div class="report-subtitle">
                AI 마스킹과 스낵 색상 기반 보조 마스크를 결합해 제품 영역 누락을 줄이고,
                float Lab 색공간 기준으로 Delta E 분포와 P95 편차를 산출합니다.
            </div>
        </div>
        <div class="spec-chips">
            <div class="spec-chips-title">Precision Profile</div>
            <div class="spec-row"><span>Mask resolution</span><span>{MASK_DIM}px</span></div>
            <div class="spec-row"><span>Color analysis</span><span>{ANALYSIS_DIM}px</span></div>
            <div class="spec-row"><span>Segmentation model</span><span>{REMBG_MODEL}</span></div>
            <div class="spec-row"><span>Color rescue mask</span>{color_rescue_pill}</div>
            <div class="spec-row"><span>Reference</span><span>{target_text}</span></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


tab1, tab2 = st.tabs(["실시간 색차 분석", "누적 품질 통계"])


with tab1:
    left_col, right_col = st.columns([0.76, 1.24], gap="large")

    with left_col:
        st.markdown(
            """
            <div class="upload-note">
                <strong>검사 이미지 업로드</strong>
                <span>정확도를 위해 배경은 단순하게, 조명은 일정하게 촬영하는 것을 권장합니다.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "이미지 파일",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

        lot_input = st.text_input("LOT NUMBER", "LOT-2026-005")

        render_panel_title("분석 조건", "Railway Variables에서 기준값과 해상도를 조정할 수 있습니다.")
        config_cols = st.columns(2)
        with config_cols[0]:
            render_kpi("Review", f">= {REVIEW_THRESHOLD:.1f}", "검토 필요 기준", "warning")
        with config_cols[1]:
            render_kpi("NG", f">= {NG_THRESHOLD:.1f}", "관리 한계 기준", "danger")

        st.caption(
            "isnet이 제품을 일부만 잡으면 REMBG_MODEL=u2net, MASK_THRESHOLD=24, MASK_ERODE_ITER=0 조합을 권장합니다."
        )

    with right_col:
        if uploaded_file is None:
            st.markdown(
                """
                <div class="section-card">
                    <div class="empty-state">
                        <div class="empty-state-badge">QC</div>
                        <div class="panel-heading">분석 결과 대기</div>
                        <div class="panel-caption">
                            이미지를 업로드하면 고정밀 마스킹, Delta E 분포, 평균 Lab, 판정 결과가 표시됩니다.
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            try:
                file_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
                original_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                if original_img is None:
                    st.error("이미지를 읽지 못했습니다. JPG 또는 PNG 파일을 다시 업로드해주세요.")
                    st.stop()

                with st.spinner("AI 마스킹과 색상 보조 마스크를 결합해 분석 중입니다..."):
                    ai_session = load_ai_model(REMBG_MODEL)
                    result = analyze_image(original_img, ai_session)

                analysis_img = result["analysis_img"]
                visual_masked_img = result["visual_masked_img"]
                core_masked_img = result["core_masked_img"]
                ai_masked_img = result["ai_masked_img"]
                color_rescue_img = result["color_rescue_img"]
                heatmap_masked = result["heatmap_masked"]
                delta_values = result["delta_values"]
                mean_bgr = result["mean_bgr"]
                avg_l = result["avg_l"]
                avg_a = result["avg_a"]
                avg_b = result["avg_b"]
                mean_delta_e = result["mean_delta_e"]
                std_delta_e = result["std_delta_e"]
                p95_delta_e = result["p95_delta_e"]
                sample_pixels = result["sample_pixels"]
                mask_area_ratio = result["mask_area_ratio"]
                ai_area_ratio = result["ai_area_ratio"]
                color_area_ratio = result["color_area_ratio"]
                analysis_mode = result["analysis_mode"]
                status_meta = calculate_status(mean_delta_e)

                kpi_cols = st.columns(4)
                with kpi_cols[0]:
                    render_kpi("Final Status", status_meta["status"], status_meta["label"], status_meta["tone"])
                with kpi_cols[1]:
                    render_kpi("Mean Delta E", f"{mean_delta_e:.2f}", analysis_mode, "blue")
                with kpi_cols[2]:
                    render_kpi("P95 Delta E", f"{p95_delta_e:.2f}", "상위 5% 편차 경계", "neutral")
                with kpi_cols[3]:
                    render_kpi("Mask Coverage", f"{mask_area_ratio * 100:.1f}%", f"AI {ai_area_ratio * 100:.1f}% / Color {color_area_ratio * 100:.1f}%", "blue")

                st.write("")
                status_col, color_col = st.columns([1, 1], gap="large")
                with status_col:
                    render_status_panel(status_meta, mean_delta_e, std_delta_e, p95_delta_e)
                with color_col:
                    render_color_chip(mean_bgr, avg_l, avg_a, avg_b)

                    fig, ax = plt.subplots(figsize=(4.8, 3.2))
                    ax.hist(delta_values, bins=80, color="#4338ca", edgecolor="#ffffff", linewidth=0.35, alpha=0.92)
                    ax.axvline(mean_delta_e, color="#d97706", linewidth=2, label="Mean")
                    ax.axvline(p95_delta_e, color="#dc2626", linewidth=2, label="P95")
                    ax.set_xlabel("Delta E", fontsize=9, color="#5b6083")
                    ax.set_ylabel("Pixel Count", fontsize=9, color="#5b6083")
                    ax.tick_params(colors="#9296b0", labelsize=8)
                    ax.grid(axis="y", linestyle="--", alpha=0.25, color="#d8dcec")
                    ax.legend(frameon=False, fontsize=8, labelcolor="#11132a")
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    ax.spines["left"].set_color("#d8dcec")
                    ax.spines["bottom"].set_color("#d8dcec")
                    fig.patch.set_facecolor("#ffffff")
                    ax.set_facecolor("#ffffff")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                render_panel_title("시각 검증", "AI 마스크와 색상 보조 마스크가 합쳐진 최종 제품 영역을 확인합니다.")
                img_col1, img_col2 = st.columns(2, gap="medium")
                with img_col1:
                    render_media_label("Raw Image")
                    st.image(cv2.cvtColor(analysis_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                with img_col2:
                    render_media_label("Final Product Mask")
                    st.image(cv2.cvtColor(visual_masked_img, cv2.COLOR_BGR2RGB), use_container_width=True)

                img_col3, img_col4 = st.columns(2, gap="medium")
                with img_col3:
                    render_media_label("Analysis Core Mask")
                    st.image(cv2.cvtColor(core_masked_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                with img_col4:
                    render_media_label("Delta E Heatmap")
                    st.image(cv2.cvtColor(heatmap_masked, cv2.COLOR_BGR2RGB), use_container_width=True)

                with st.expander("마스크 진단 보기"):
                    diag_col1, diag_col2 = st.columns(2, gap="medium")
                    with diag_col1:
                        render_media_label("AI Mask Only")
                        st.image(cv2.cvtColor(ai_masked_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                    with diag_col2:
                        render_media_label("Color Rescue Mask Only")
                        st.image(cv2.cvtColor(color_rescue_img, cv2.COLOR_BGR2RGB), use_container_width=True)

                st.write("")
                save_col, note_col = st.columns([0.35, 0.65])
                with save_col:
                    save_clicked = st.button("검사 결과 DB 저장")
                with note_col:
                    st.caption("저장 시 LOT 번호, 평균 Lab, Mean/P95 Delta E, 판정 상태가 Supabase에 기록됩니다.")

                if save_clicked:
                    if supabase is None:
                        st.error("Supabase 환경변수가 설정되지 않았습니다. Railway Variables를 확인해주세요.")
                    else:
                        log_data = {
                            "lot_number": lot_input,
                            "avg_l": round(avg_l, 2),
                            "avg_a": round(avg_a, 2),
                            "avg_b": round(avg_b, 2),
                            "delta_e": round(mean_delta_e, 2),
                            "defect_type": "색상 편차 관리 필요" if status_meta["status"] == "NG" else "정상",
                            "status": status_meta["status"],
                            "image_path": "web_upload.jpg",
                        }
                        try:
                            supabase.table("snack_color_logs").insert(log_data).execute()
                            st.success("Supabase에 검사 결과가 저장되었습니다.")
                        except Exception as e:
                            st.error(f"저장 실패: {e}")

                del original_img, analysis_img, visual_masked_img, core_masked_img
                del ai_masked_img, color_rescue_img, heatmap_masked, delta_values, result
                gc.collect()

            except Exception as e:
                st.error("분석 중 오류가 발생했습니다.")
                st.exception(e)


with tab2:
    top_cols = st.columns([0.7, 0.3])
    with top_cols[0]:
        render_panel_title("누적 품질 통계", "Supabase에 저장된 LOT 검사 이력을 기준으로 품질 추세를 확인합니다.")
    with top_cols[1]:
        if st.button("DB 동기화"):
            st.rerun()

    if supabase is None:
        st.warning("Supabase 환경변수가 설정되지 않아 통계 DB를 불러올 수 없습니다.")
    else:
        try:
            response = (
                supabase.table("snack_color_logs")
                .select("*")
                .order("created_at", desc=False)
                .execute()
            )
            data = response.data

            if not data:
                st.info("DB에 저장된 검사 데이터가 없습니다.")
            else:
                df = pd.DataFrame(data)

                total_count = len(df)
                ng_count = len(df[df["status"] == "NG"]) if "status" in df.columns else 0
                review_count = len(df[df["status"] == "REVIEW"]) if "status" in df.columns else 0
                action_rate = ((ng_count + review_count) / total_count) * 100 if total_count > 0 else 0
                avg_delta_e = df["delta_e"].mean() if "delta_e" in df.columns else 0
                recent_status = df.iloc[-1]["status"] if "status" in df.columns else "-"

                stat_cols = st.columns(4)
                with stat_cols[0]:
                    render_kpi("Total Inspections", f"{total_count:,}", "누적 검사 LOT", "blue")
                with stat_cols[1]:
                    render_kpi("Action Rate", f"{action_rate:.1f}%", "REVIEW + NG 비율", "warning" if action_rate > 0 else "success")
                with stat_cols[2]:
                    render_kpi("Average Delta E", f"{avg_delta_e:.2f}", "누적 평균 색차", "neutral")
                with stat_cols[3]:
                    render_kpi("Latest Status", str(recent_status), "최근 LOT 판정", "success" if recent_status == "OK" else "warning")

                st.write("")
                if "lot_number" in df.columns and "delta_e" in df.columns:
                    chart_df = df[["lot_number", "delta_e"]].set_index("lot_number")
                    st.line_chart(chart_df, color="#4338ca")

                st.write("")
                st.dataframe(
                    safe_table_columns(
                        df,
                        [
                            "created_at",
                            "lot_number",
                            "avg_l",
                            "avg_a",
                            "avg_b",
                            "delta_e",
                            "defect_type",
                            "status",
                        ],
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

        except Exception as e:
            st.error(f"데이터를 불러오지 못했습니다: {e}")
