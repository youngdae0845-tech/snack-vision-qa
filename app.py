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
    page_icon="SV",
    layout="wide",
    initial_sidebar_state="expanded",
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

MASK_DIM = get_int_config("MASK_DIM", 2200)
ANALYSIS_DIM = get_int_config("ANALYSIS_DIM", 2600)
REMBG_MODEL = get_config("REMBG_MODEL", "isnet-general-use")

MASK_THRESHOLD = get_int_config("MASK_THRESHOLD", 128)
MASK_KERNEL_SIZE = get_int_config("MASK_KERNEL_SIZE", 5)
MASK_CLOSE_ITER = get_int_config("MASK_CLOSE_ITER", 1)
MASK_ERODE_ITER = get_int_config("MASK_ERODE_ITER", 1)

REVIEW_THRESHOLD_DEFAULT = get_float_config("REVIEW_THRESHOLD", 5.0)
NG_THRESHOLD = get_float_config("NG_THRESHOLD", 20.0)
HEATMAP_MAX_DELTA_E = get_float_config("HEATMAP_MAX_DELTA_E", 30.0)
EXCELLENT_THRESHOLD = get_float_config("EXCELLENT_THRESHOLD", 3.0)

TARGET_L_DEFAULT = get_float_config("TARGET_L", 165.0)
TARGET_A_DEFAULT = get_float_config("TARGET_A", 149.0)
TARGET_B_DEFAULT = get_float_config("TARGET_B", 165.0)


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

    _, display_mask = cv2.threshold(
        resized_alpha,
        MASK_THRESHOLD,
        255,
        cv2.THRESH_BINARY,
    )

    kernel_size = normalize_kernel_size(MASK_KERNEL_SIZE)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    if MASK_CLOSE_ITER > 0:
        display_mask = cv2.morphologyEx(
            display_mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=MASK_CLOSE_ITER,
        )

    analysis_mask = display_mask.copy()
    if MASK_ERODE_ITER > 0:
        analysis_mask = cv2.erode(analysis_mask, kernel, iterations=MASK_ERODE_ITER)

    return display_mask, analysis_mask


def calculate_status(mean_delta_e, review_threshold):
    if mean_delta_e >= NG_THRESHOLD:
        return {
            "status": "NG",
            "label": "관리 한계 초과",
            "tone": "danger",
            "grade": "불량",
            "grade_desc": "뭉침/편중",
            "message": "색상 편차가 높습니다. 시즈닝 뭉침, 과열, 조명 조건을 확인하세요.",
        }
    if mean_delta_e >= review_threshold:
        return {
            "status": "REVIEW",
            "label": "검토 필요",
            "tone": "warning",
            "grade": "경계",
            "grade_desc": "시즈닝 편중 의심",
            "message": "색상 편차가 기준 근처입니다. LOT 샘플 재확인을 권장합니다.",
        }
    if mean_delta_e < EXCELLENT_THRESHOLD:
        return {
            "status": "OK",
            "label": "정상 범위",
            "tone": "success",
            "grade": "우수",
            "grade_desc": "ΔE < 3",
            "message": "색상 균일도가 매우 안정적입니다.",
        }
    return {
        "status": "OK",
        "label": "정상 범위",
        "tone": "success",
        "grade": "합격",
        "grade_desc": "기준 이내",
        "message": "색상 균일도가 안정적입니다. 현재 기준에서는 정상으로 판단됩니다.",
    }


def analyze_image(original_img, session, target_lab, review_threshold):
    analysis_img = resize_by_max_dim(original_img, ANALYSIS_DIM)
    display_mask, analysis_mask = create_precise_masks(original_img, analysis_img, session)

    core_pixels = analysis_mask > 0
    if np.count_nonzero(core_pixels) == 0:
        raise ValueError("스낵 영역을 감지하지 못했습니다.")

    visual_masked_img = cv2.bitwise_and(analysis_img, analysis_img, mask=display_mask)
    core_masked_img = cv2.bitwise_and(analysis_img, analysis_img, mask=analysis_mask)

    bgr_float = analysis_img.astype(np.float32) / 255.0
    lab_img = cv2.cvtColor(bgr_float, cv2.COLOR_BGR2LAB)
    lab_pixels = lab_img[core_pixels]
    bgr_pixels = analysis_img[core_pixels]

    mean_lab = lab_pixels.mean(axis=0)
    mean_bgr = bgr_pixels.mean(axis=0)

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
    min_delta_e = float(delta_values.min()) if delta_values.size else 0.0
    max_delta_e = float(delta_values.max()) if delta_values.size else 0.0

    delta_map = np.zeros(core_pixels.shape, dtype=np.float32)
    delta_map[core_pixels] = delta_values

    heatmap_scale = max(HEATMAP_MAX_DELTA_E, 1.0)
    heatmap_gray = np.clip((delta_map / heatmap_scale) * 255.0, 0, 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_TURBO)
    heatmap_blend = cv2.addWeighted(analysis_img, 0.52, heatmap_bgr, 0.48, 0)
    heatmap_masked = cv2.bitwise_and(heatmap_blend, heatmap_blend, mask=display_mask)

    status_meta = calculate_status(mean_delta_e, review_threshold)

    result = {
        "analysis_img": analysis_img,
        "visual_masked_img": visual_masked_img,
        "core_masked_img": core_masked_img,
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
        "min_delta_e": min_delta_e,
        "max_delta_e": max_delta_e,
        "sample_pixels": int(delta_values.size),
        "status_meta": status_meta,
        "review_threshold": review_threshold,
    }

    del bgr_float, lab_img, lab_pixels, bgr_pixels, delta_map, heatmap_bgr, heatmap_blend
    del display_mask, analysis_mask, core_pixels
    gc.collect()
    return result


def safe_table_columns(df, columns):
    existing = [column for column in columns if column in df.columns]
    return df[existing] if existing else df


# ---------------------------------------------------------------------------
# Rendering helpers (dark instrument-panel theme)
# ---------------------------------------------------------------------------

def render_number_heading(number, title):
    st.markdown(
        f"""
        <div class="section-heading">
            <span class="section-number">{number}</span>
            <span class="section-title">{title}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(label, value, helper, tone="neutral"):
    st.markdown(
        f"""
        <div class="kpi-card kpi-{tone}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-helper">{helper}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_pill(text, active=False):
    dot_class = "dot-active" if active else "dot-idle"
    st.markdown(
        f"""
        <div class="status-pill">
            <span class="{dot_class}"></span>{text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_color_tile(mean_bgr=None, sample_pixels=None):
    if mean_bgr is None:
        st.markdown(
            """
            <div class="tile-card">
                <div class="tile-swatch tile-swatch-empty">이미지 대기 중</div>
                <div class="tile-meta">
                    <div><span>RGB</span><strong>—</strong></div>
                    <div><span>PIXELS</span><strong>—</strong></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    r = int(np.clip(mean_bgr[2], 0, 255))
    g = int(np.clip(mean_bgr[1], 0, 255))
    b = int(np.clip(mean_bgr[0], 0, 255))
    st.markdown(
        f"""
        <div class="tile-card">
            <div class="tile-swatch" style="background: rgb({r}, {g}, {b});"></div>
            <div class="tile-meta">
                <div><span>RGB</span><strong>{r}, {g}, {b}</strong></div>
                <div><span>PIXELS</span><strong>{sample_pixels:,}</strong></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_legend():
    st.markdown(
        """
        <div class="legend-row">
            <span class="legend-item"><span class="legend-dot legend-green"></span>우수 (ΔE&lt;3)</span>
            <span class="legend-item"><span class="legend-dot legend-blue"></span>합격</span>
            <span class="legend-item"><span class="legend-dot legend-amber"></span>경계 (시즈닝 편중 의심)</span>
            <span class="legend-item"><span class="legend-dot legend-red"></span>불량 (뭉침/편중)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer_stats(n=None, min_v=None, max_v=None, threshold=None):
    n_text = f"{n:,}" if n is not None else "—"
    min_text = f"{min_v:.2f}" if min_v is not None else "—"
    max_text = f"{max_v:.2f}" if max_v is not None else "—"
    th_text = f"{threshold:.1f}" if threshold is not None else "—"
    st.markdown(
        f"""
        <div class="footer-stats">
            <span>N <strong>{n_text}</strong> px</span>
            <span>MIN <strong>{min_text}</strong></span>
            <span>MAX <strong>{max_text}</strong></span>
            <span>THRESHOLD <strong>{th_text}</strong></span>
        </div>
        """,
        unsafe_allow_html=True,
    )


supabase = get_supabase_client()

st.markdown(
    """
    <style>
        :root {
            --bg: #0a0e16;
            --panel: #10141f;
            --panel-2: #141926;
            --line: #232a3a;
            --ink: #e6e9f2;
            --muted: #7c8598;
            --navy: #0f2742;
            --blue: #3b82f6;
            --amber: #d97706;
            --green: #16a34a;
            --red: #ef4444;
        }

        .stApp {
            background: var(--bg);
            color: var(--ink);
        }

        .block-container {
            max-width: 1360px;
            padding-top: 1.4rem;
            padding-bottom: 3rem;
        }

        #MainMenu, footer, header {
            visibility: hidden;
        }

        section[data-testid="stSidebar"] {
            background: var(--panel);
            border-right: 1px solid var(--line);
        }

        section[data-testid="stSidebar"] .block-container {
            padding-top: 1.6rem;
        }

        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span {
            color: var(--ink) !important;
        }

        .sidebar-eyebrow {
            font-size: 11px;
            font-weight: 800;
            letter-spacing: .08em;
            text-transform: uppercase;
            color: var(--muted);
            margin-bottom: 6px;
        }

        .sidebar-group-title {
            font-size: 11px;
            font-weight: 800;
            letter-spacing: .06em;
            text-transform: uppercase;
            color: var(--muted);
            margin: 18px 0 10px;
            border-top: 1px solid var(--line);
            padding-top: 16px;
        }

        .report-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 24px;
            padding: 22px 26px;
            margin-bottom: 18px;
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 10px;
        }

        .brand-mark {
            font-size: 11px;
            font-weight: 800;
            color: var(--blue);
            text-transform: uppercase;
            letter-spacing: .1em;
            margin-bottom: 8px;
        }

        .report-title {
            font-size: 26px;
            line-height: 1.2;
            font-weight: 800;
            color: var(--ink);
            margin-bottom: 8px;
        }

        .report-subtitle {
            font-size: 13px;
            color: var(--muted);
            max-width: 720px;
            line-height: 1.6;
        }

        .status-pill {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 14px;
            border: 1px solid var(--line);
            border-radius: 999px;
            font-size: 12px;
            font-weight: 700;
            color: var(--ink);
            white-space: nowrap;
        }

        .dot-idle, .dot-active {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            display: inline-block;
        }

        .dot-idle { background: var(--muted); }
        .dot-active { background: var(--green); box-shadow: 0 0 6px var(--green); }

        .section-heading {
            display: flex;
            align-items: center;
            gap: 10px;
            margin: 4px 0 12px;
        }

        .section-number {
            width: 22px;
            height: 22px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border: 1px solid var(--line);
            border-radius: 6px;
            font-size: 11px;
            font-weight: 800;
            color: var(--muted);
        }

        .section-title {
            font-size: 14px;
            font-weight: 800;
            color: var(--ink);
        }

        .kpi-card {
            min-height: 110px;
            background: var(--panel);
            border: 1px solid var(--line);
            border-top: 3px solid #3a4256;
            border-radius: 10px;
            padding: 16px;
        }

        .kpi-success { border-top-color: var(--green); }
        .kpi-warning { border-top-color: var(--amber); }
        .kpi-danger { border-top-color: var(--red); }
        .kpi-blue { border-top-color: var(--blue); }

        .kpi-label {
            font-size: 11px;
            color: var(--muted);
            font-weight: 750;
            text-transform: uppercase;
            letter-spacing: .06em;
        }

        .kpi-value {
            font-size: 26px;
            line-height: 1.2;
            font-weight: 800;
            color: var(--ink);
            margin-top: 8px;
            font-variant-numeric: tabular-nums;
        }

        .kpi-helper {
            font-size: 12px;
            color: var(--muted);
            margin-top: 8px;
        }

        .section-card {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 10px;
            padding: 18px;
        }

        .tile-card {
            background: var(--panel-2);
            border: 1px solid var(--line);
            border-radius: 10px;
            padding: 14px;
            min-height: 260px;
            display: flex;
            flex-direction: column;
        }

        .tile-swatch {
            width: 100%;
            flex: 1;
            min-height: 170px;
            border-radius: 8px;
            border: 1px solid var(--line);
        }

        .tile-swatch-empty {
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--muted);
            font-size: 13px;
            background: repeating-linear-gradient(45deg, #0d1220, #0d1220 10px, #10141f 10px, #10141f 20px);
        }

        .tile-meta {
            display: flex;
            justify-content: space-between;
            margin-top: 12px;
            font-size: 12px;
        }

        .tile-meta span {
            display: block;
            color: var(--muted);
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: .05em;
        }

        .tile-meta strong {
            display: block;
            color: var(--ink);
            margin-top: 4px;
            font-size: 13px;
        }

        .heatmap-frame {
            background: var(--panel-2);
            border: 1px solid var(--line);
            border-radius: 10px;
            padding: 14px;
            min-height: 260px;
        }

        .heatmap-empty {
            min-height: 340px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--muted);
            font-size: 13px;
            background-image:
                linear-gradient(var(--line) 1px, transparent 1px),
                linear-gradient(90deg, var(--line) 1px, transparent 1px);
            background-size: 28px 28px;
            border-radius: 8px;
            border: 1px solid var(--line);
        }

        .legend-row {
            display: flex;
            flex-wrap: wrap;
            gap: 18px;
            margin-top: 14px;
            font-size: 12px;
            color: var(--muted);
        }

        .legend-item {
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }

        .legend-dot {
            width: 9px;
            height: 9px;
            border-radius: 50%;
            display: inline-block;
        }

        .legend-green { background: var(--green); }
        .legend-blue { background: var(--blue); }
        .legend-amber { background: var(--amber); }
        .legend-red { background: var(--red); }

        .footer-stats {
            display: flex;
            gap: 24px;
            margin-top: 14px;
            padding-top: 12px;
            border-top: 1px solid var(--line);
            font-size: 12px;
            color: var(--muted);
        }

        .footer-stats strong {
            color: var(--ink);
            margin-left: 4px;
        }

        .status-summary {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 10px;
            margin-top: 6px;
        }

        .status-summary div {
            padding: 10px;
            background: var(--panel-2);
            border: 1px solid var(--line);
            border-radius: 8px;
        }

        .status-summary span {
            display: block;
            font-size: 11px;
            color: var(--muted);
        }

        .status-summary strong {
            display: block;
            font-size: 18px;
            margin-top: 4px;
            color: var(--ink);
        }

        .placeholder-card {
            background: var(--panel);
            border: 1px dashed var(--line);
            border-radius: 10px;
            padding: 60px 24px;
            text-align: center;
            color: var(--muted);
        }

        .placeholder-card strong {
            display: block;
            color: var(--ink);
            font-size: 17px;
            margin-bottom: 8px;
        }

        [data-testid="stImage"] {
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid var(--line);
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 6px;
            margin-bottom: 14px;
            border-bottom: 1px solid var(--line);
        }

        .stTabs [data-baseweb="tab"] {
            height: 42px;
            padding: 0 16px;
            background: transparent;
            color: var(--muted);
            font-weight: 700;
            font-size: 13px;
        }

        .stTabs [aria-selected="true"] {
            color: var(--blue) !important;
            border-bottom: 2px solid var(--blue);
        }

        .stButton > button {
            width: 100%;
            min-height: 42px;
            border-radius: 8px;
            border: 1px solid var(--blue);
            background: var(--blue);
            color: #ffffff;
            font-weight: 750;
        }

        .stButton > button:hover {
            border-color: #60a5fa;
            background: #2563eb;
            color: #ffffff;
        }

        .stButton > button:disabled {
            border-color: var(--line);
            background: var(--panel-2);
            color: var(--muted);
        }

        div[data-testid="stTextInput"] input,
        div[data-testid="stNumberInput"] input,
        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
        div[data-testid="stFileUploader"] section {
            background: var(--panel-2) !important;
            border: 1px solid var(--line) !important;
            color: var(--ink) !important;
            border-radius: 8px !important;
        }

        div[data-testid="stFileUploader"] section button {
            background: var(--panel) !important;
            color: var(--ink) !important;
            border: 1px solid var(--line) !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Sidebar — TARGET LAB / thresholds / image upload / run
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown('<div class="sidebar-eyebrow">SNACK QC · MULTI-MODULE INSTRUMENT</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-group-title" style="margin-top:0;border-top:none;padding-top:0;">기준값 · TARGET LAB</div>', unsafe_allow_html=True)

    use_target_lab = st.checkbox("기준 Lab 값 사용 (해제 시 이미지 평균 기준)", value=True)

    target_l = st.slider("L* 명도", 0.0, 255.0, float(TARGET_L_DEFAULT), 1.0)
    target_a = st.slider("a* 적색도", 0.0, 255.0, float(TARGET_A_DEFAULT), 1.0)
    target_b = st.slider("b* 황색도", 0.0, 255.0, float(TARGET_B_DEFAULT), 1.0)

    st.markdown('<div class="sidebar-group-title">판정 기준</div>', unsafe_allow_html=True)
    review_threshold = st.number_input(
        "경고 알람 (Mean ΔE)",
        min_value=0.0,
        value=float(REVIEW_THRESHOLD_DEFAULT),
        step=0.5,
    )
    st.caption(f"평균 ΔE가 이 값을 초과하면 관리 요망(REVIEW)으로, {NG_THRESHOLD:.1f} 이상이면 불량(NG)으로 판정합니다.")

    st.markdown('<div class="sidebar-group-title">분석 이미지</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "이미지 파일",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    run_clicked = st.button("분석 실행", disabled=uploaded_file is None)


if "result" not in st.session_state:
    st.session_state.result = None
if "result_error" not in st.session_state:
    st.session_state.result_error = None

if run_clicked and uploaded_file is not None:
    try:
        file_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
        original_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if original_img is None:
            st.session_state.result = None
            st.session_state.result_error = "이미지를 읽지 못했습니다. JPG 또는 PNG 파일을 다시 업로드해주세요."
        else:
            target_lab = (
                np.array([target_l, target_a, target_b], dtype=np.float32)
                if use_target_lab
                else None
            )
            with st.spinner("고정밀 AI 마스킹과 색차 분석을 실행 중입니다..."):
                ai_session = load_ai_model(REMBG_MODEL)
                result = analyze_image(original_img, ai_session, target_lab, review_threshold)
            st.session_state.result = result
            st.session_state.result_error = None
            del original_img
            gc.collect()
    except Exception as e:
        st.session_state.result = None
        st.session_state.result_error = str(e)


result = st.session_state.result
has_result = result is not None

target_text = (
    "image mean" if not use_target_lab else f"L {target_l:.1f} / a {target_a:.1f} / b {target_b:.1f}"
)

header_left, header_right = st.columns([0.8, 0.2])
with header_left:
    st.markdown(
        f"""
        <div class="report-header" style="margin-bottom:0;">
            <div>
                <div class="brand-mark">SNACK QC · MULTI-MODULE INSTRUMENT</div>
                <div class="report-title">스낵 품질 분석 계기판</div>
                <div class="report-subtitle">
                    무광 검은색 배경에서 촬영된 스낵 이미지를 기준 LAB 값과 비교해 색상 편차(ΔE)를
                    픽셀 단위로 계산합니다.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with header_right:
    st.markdown("<div style='height: 22px'></div>", unsafe_allow_html=True)
    render_status_pill("분석 완료" if has_result else "분석 대기", active=has_result)

st.write("")

tab1, tab2, tab3 = st.tabs(["01 색차 분석", "02 펠릿 크기 분석", "03 기록 관리"])


with tab1:
    if st.session_state.result_error:
        st.error(st.session_state.result_error)

    kpi_cols = st.columns(3)
    if has_result:
        status_meta = result["status_meta"]
        with kpi_cols[0]:
            render_kpi_card("MEAN ΔE", f"{result['mean_delta_e']:.2f}", "평균 색차", "blue")
        with kpi_cols[1]:
            render_kpi_card("ΔE STD DEV", f"{result['std_delta_e']:.2f}", "색차 표준편차", "neutral")
        with kpi_cols[2]:
            render_kpi_card("GRADE", status_meta["grade"], status_meta["grade_desc"], status_meta["tone"])
    else:
        with kpi_cols[0]:
            render_kpi_card("MEAN ΔE", "—", "평균 색차", "neutral")
        with kpi_cols[1]:
            render_kpi_card("ΔE STD DEV", "—", "색차 표준편차", "neutral")
        with kpi_cols[2]:
            render_kpi_card("GRADE", "—", "품질 판정", "neutral")

    st.write("")

    # 기록 저장
    st.markdown('<div class="section-heading"><span class="section-number">+</span><span class="section-title">기록 저장</span></div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        f_cols = st.columns(5)
        with f_cols[0]:
            product_name = st.text_input("제품", placeholder="예: 새우깡", label_visibility="visible")
        with f_cols[1]:
            line_name = st.text_input("생산라인", placeholder="예: 후라잉라인")
        with f_cols[2]:
            batch_no = st.text_input("배치번호", placeholder="예: B240912-1")
        with f_cols[3]:
            shift = st.selectbox("조", ["조간", "야간"])
        with f_cols[4]:
            worker = st.text_input("작업자", placeholder="이름")
        note = st.text_input("메모", placeholder="선택 입력")

        save_col, help_col = st.columns([0.3, 0.7])
        with save_col:
            save_clicked = st.button("기록에 저장", disabled=not has_result)
        with help_col:
            if not has_result:
                st.caption("분석을 먼저 실행하세요")
            else:
                st.caption("저장 시 LOT 정보, 평균 Lab, Mean ΔE, 판정 결과가 DB에 기록됩니다.")
        st.markdown('</div>', unsafe_allow_html=True)

        if save_clicked and has_result:
            if supabase is None:
                st.error("Supabase 환경변수가 설정되지 않았습니다. Railway Variables를 확인해주세요.")
            else:
                log_data = {
                    "lot_number": batch_no or None,
                    "product": product_name or None,
                    "line": line_name or None,
                    "shift": shift,
                    "worker": worker or None,
                    "note": note or None,
                    "avg_l": round(result["avg_l"], 2),
                    "avg_a": round(result["avg_a"], 2),
                    "avg_b": round(result["avg_b"], 2),
                    "delta_e": round(result["mean_delta_e"], 2),
                    "defect_type": "색상 편차 관리 필요" if result["status_meta"]["status"] == "NG" else "정상",
                    "status": result["status_meta"]["status"],
                    "image_path": "web_upload.jpg",
                }
                try:
                    supabase.table("snack_color_logs").insert(log_data).execute()
                    st.success("Supabase에 검사 결과가 저장되었습니다.")
                except Exception as e:
                    st.error(f"저장 실패: {e}")

    st.write("")

    tile_col, map_col = st.columns([1, 1.6], gap="large")
    with tile_col:
        render_number_heading("01", "평균 색상 타일")
        render_color_tile(result["mean_bgr"] if has_result else None, result["sample_pixels"] if has_result else None)

    with map_col:
        render_number_heading("02", "편차 오버레이 맵")
        st.markdown('<div class="heatmap-frame">', unsafe_allow_html=True)
        if has_result:
            st.image(cv2.cvtColor(result["heatmap_masked"], cv2.COLOR_BGR2RGB), use_container_width=True)
        else:
            st.markdown('<div class="heatmap-empty">이미지를 업로드하고 분석을 실행하세요</div>', unsafe_allow_html=True)
        render_legend()
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")

    render_number_heading("03", "색상 편차 분포 (ΔE 히스토그램)")
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    if has_result:
        fig, ax = plt.subplots(figsize=(9.6, 3.2))
        fig.patch.set_facecolor("#10141f")
        ax.set_facecolor("#10141f")
        ax.hist(result["delta_values"], bins=80, color="#3b82f6", edgecolor="#0a0e16", linewidth=0.3)
        ax.axvline(result["mean_delta_e"], color="#d97706", linewidth=2, label="Mean")
        ax.axvline(result["p95_delta_e"], color="#ef4444", linewidth=2, label="P95")
        ax.set_xlabel("Delta E", fontsize=9, color="#7c8598")
        ax.set_ylabel("Pixel Count", fontsize=9, color="#7c8598")
        ax.tick_params(colors="#7c8598", labelsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.15, color="#7c8598")
        legend = ax.legend(frameon=False, fontsize=8, labelcolor="#e6e9f2")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#232a3a")
        ax.spines["bottom"].set_color("#232a3a")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        render_footer_stats(
            n=result["sample_pixels"],
            min_v=result["min_delta_e"],
            max_v=result["max_delta_e"],
            threshold=result["review_threshold"],
        )
    else:
        st.markdown('<div class="heatmap-empty" style="min-height:220px;">이미지를 업로드하고 분석을 실행하세요</div>', unsafe_allow_html=True)
        render_footer_stats()
    st.markdown('</div>', unsafe_allow_html=True)

    if has_result:
        st.write("")
        status_meta = result["status_meta"]
        st.markdown(
            f"""
            <div class="section-card">
                <div class="section-title">최종 판정: {status_meta["status"]} · {status_meta["label"]}</div>
                <div class="kpi-helper" style="margin-top:6px;">{status_meta["message"]}</div>
                <div class="status-summary">
                    <div><span>Mean Delta E</span><strong>{result['mean_delta_e']:.2f}</strong></div>
                    <div><span>P95 Delta E</span><strong>{result['p95_delta_e']:.2f}</strong></div>
                    <div><span>Review Threshold</span><strong>{result['review_threshold']:.1f}</strong></div>
                    <div><span>NG Threshold</span><strong>{NG_THRESHOLD:.1f}</strong></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("시각 검증 (원본 · 전체 마스크 · 분석 코어 마스크)"):
            img_col1, img_col2, img_col3 = st.columns(3, gap="medium")
            with img_col1:
                st.markdown("**Raw Image**")
                st.image(cv2.cvtColor(result["analysis_img"], cv2.COLOR_BGR2RGB), use_container_width=True)
            with img_col2:
                st.markdown("**Full AI Mask**")
                st.image(cv2.cvtColor(result["visual_masked_img"], cv2.COLOR_BGR2RGB), use_container_width=True)
            with img_col3:
                st.markdown("**Analysis Core Mask**")
                st.image(cv2.cvtColor(result["core_masked_img"], cv2.COLOR_BGR2RGB), use_container_width=True)


with tab2:
    st.markdown(
        """
        <div class="placeholder-card">
            <strong>펠릿 크기 분석 모듈</strong>
            이 모듈은 준비 중입니다. 곧 펠릿(스낵 조각) 크기 균일도 분석 기능이 추가될 예정입니다.
        </div>
        """,
        unsafe_allow_html=True,
    )


with tab3:
    top_cols = st.columns([0.7, 0.3])
    with top_cols[0]:
        render_number_heading("03", "기록 관리")
    with top_cols[1]:
        if st.button("DB 동기화"):
            st.rerun()

    if supabase is None:
        st.warning("Supabase 환경변수가 설정되지 않아 기록을 불러올 수 없습니다.")
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
                    render_kpi_card("Total Inspections", f"{total_count:,}", "누적 검사 LOT", "blue")
                with stat_cols[1]:
                    render_kpi_card("Action Rate", f"{action_rate:.1f}%", "REVIEW + NG 비율", "warning" if action_rate > 0 else "success")
                with stat_cols[2]:
                    render_kpi_card("Average Delta E", f"{avg_delta_e:.2f}", "누적 평균 색차", "neutral")
                with stat_cols[3]:
                    render_kpi_card("Latest Status", str(recent_status), "최근 LOT 판정", "success" if recent_status == "OK" else "warning")

                st.write("")
                if "lot_number" in df.columns and "delta_e" in df.columns:
                    chart_df = df[["lot_number", "delta_e"]].set_index("lot_number")
                    st.line_chart(chart_df, color="#3b82f6")

                st.write("")
                st.dataframe(
                    safe_table_columns(
                        df,
                        [
                            "created_at",
                            "lot_number",
                            "product",
                            "line",
                            "shift",
                            "worker",
                            "avg_l",
                            "avg_a",
                            "avg_b",
                            "delta_e",
                            "defect_type",
                            "status",
                            "note",
                        ],
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

        except Exception as e:
            st.error(f"데이터를 불러오지 못했습니다: {e}")
