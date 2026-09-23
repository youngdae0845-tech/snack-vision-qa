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

MASK_DIM = get_int_config("MASK_DIM", 2200)
ANALYSIS_DIM = get_int_config("ANALYSIS_DIM", 2600)
REMBG_MODEL = get_config("REMBG_MODEL", "isnet-general-use")

MASK_THRESHOLD = get_int_config("MASK_THRESHOLD", 128)
MASK_KERNEL_SIZE = get_int_config("MASK_KERNEL_SIZE", 5)
MASK_CLOSE_ITER = get_int_config("MASK_CLOSE_ITER", 1)
MASK_ERODE_ITER = get_int_config("MASK_ERODE_ITER", 1)

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
    }

    del bgr_float, lab_img, lab_pixels, bgr_pixels, delta_map, heatmap_bgr, heatmap_blend
    del display_mask, analysis_mask, core_pixels
    gc.collect()
    return result


def render_kpi(label, value, helper, tone="neutral"):
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


def render_status_panel(meta, mean_delta_e, std_delta_e, p95_delta_e):
    st.markdown(
        f"""
        <div class="status-panel status-{meta["tone"]}">
            <div class="status-eyebrow">Final QA Decision</div>
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
                <div class="color-values">L {avg_l:.1f} / a {avg_a:.1f} / b {avg_b:.1f}</div>
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


def safe_table_columns(df, columns):
    existing = [column for column in columns if column in df.columns]
    return df[existing] if existing else df


supabase = get_supabase_client()

st.markdown(
    """
    <style>
        :root {
            --bg: #f5f7fb;
            --panel: #ffffff;
            --ink: #111827;
            --muted: #667085;
            --line: #e5e7eb;
            --navy: #0f2742;
            --blue: #2563eb;
            --amber: #d97706;
            --green: #15803d;
            --red: #dc2626;
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

        .report-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 24px;
            padding: 24px 26px;
            margin-bottom: 18px;
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 10px;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.06);
        }

        .brand-mark {
            font-size: 12px;
            font-weight: 800;
            color: var(--blue);
            text-transform: uppercase;
            letter-spacing: .08em;
            margin-bottom: 8px;
        }

        .report-title {
            font-size: 30px;
            line-height: 1.15;
            font-weight: 850;
            color: var(--navy);
            margin-bottom: 8px;
        }

        .report-subtitle {
            font-size: 14px;
            color: var(--muted);
            max-width: 820px;
        }

        .model-strip {
            min-width: 300px;
            padding: 14px 16px;
            border: 1px solid #dbe4f0;
            border-radius: 8px;
            background: #f8fafc;
        }

        .model-strip div:first-child {
            font-size: 12px;
            font-weight: 800;
            color: var(--navy);
            margin-bottom: 8px;
        }

        .model-strip span {
            display: block;
            font-size: 12px;
            color: var(--muted);
            line-height: 1.7;
        }

        .upload-note {
            background: #ffffff;
            border: 1px dashed #b6c2d2;
            border-radius: 10px;
            padding: 22px;
            margin: 8px 0 16px;
        }

        .upload-note strong {
            display: block;
            color: var(--navy);
            font-size: 17px;
            margin-bottom: 6px;
        }

        .upload-note span {
            color: var(--muted);
            font-size: 13px;
        }

        .panel-title {
            margin: 8px 0 12px;
        }

        .panel-heading {
            font-size: 15px;
            font-weight: 850;
            color: var(--navy);
        }

        .panel-caption {
            font-size: 12px;
            color: var(--muted);
            margin-top: 2px;
        }

        .section-card {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 10px;
            padding: 18px;
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.05);
        }

        .kpi-card {
            min-height: 118px;
            background: #ffffff;
            border: 1px solid var(--line);
            border-top: 4px solid #94a3b8;
            border-radius: 10px;
            padding: 16px;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
        }

        .kpi-success { border-top-color: var(--green); }
        .kpi-warning { border-top-color: var(--amber); }
        .kpi-danger { border-top-color: var(--red); }
        .kpi-blue { border-top-color: var(--blue); }

        .kpi-label {
            font-size: 12px;
            color: var(--muted);
            font-weight: 750;
            text-transform: uppercase;
            letter-spacing: .04em;
        }

        .kpi-value {
            font-size: 28px;
            line-height: 1.2;
            font-weight: 850;
            color: var(--ink);
            margin-top: 8px;
        }

        .kpi-helper {
            font-size: 12px;
            color: var(--muted);
            margin-top: 8px;
        }

        .status-panel {
            min-height: 330px;
            border-radius: 10px;
            padding: 22px;
            color: #ffffff;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.12);
        }

        .status-success { background: #14532d; }
        .status-warning { background: #92400e; }
        .status-danger { background: #991b1b; }

        .status-eyebrow {
            font-size: 11px;
            font-weight: 850;
            text-transform: uppercase;
            letter-spacing: .08em;
            opacity: .78;
        }

        .status-main {
            font-size: 54px;
            line-height: 1;
            font-weight: 900;
            margin-top: 18px;
        }

        .status-label {
            font-size: 18px;
            font-weight: 800;
            margin-top: 8px;
        }

        .status-copy {
            font-size: 13px;
            line-height: 1.55;
            opacity: .9;
            margin-top: 12px;
        }

        .status-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 10px;
            margin-top: 20px;
        }

        .status-grid div {
            padding: 10px;
            background: rgba(255, 255, 255, .13);
            border: 1px solid rgba(255, 255, 255, .18);
            border-radius: 8px;
        }

        .status-grid span {
            display: block;
            font-size: 11px;
            opacity: .75;
        }

        .status-grid strong {
            display: block;
            font-size: 20px;
            margin-top: 4px;
        }

        .color-chip-card {
            display: grid;
            grid-template-columns: 106px 1fr;
            gap: 14px;
            align-items: center;
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 10px;
            padding: 14px;
            min-height: 136px;
        }

        .color-swatch {
            width: 106px;
            height: 106px;
            border-radius: 8px;
            border: 1px solid rgba(15, 23, 42, .14);
            box-shadow: inset 0 0 0 1px rgba(255, 255, 255, .25);
        }

        .color-title {
            font-size: 13px;
            color: var(--muted);
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: .04em;
        }

        .color-values {
            font-size: 21px;
            font-weight: 850;
            color: var(--ink);
            margin-top: 8px;
        }

        .color-helper {
            font-size: 12px;
            color: var(--muted);
            margin-top: 6px;
        }

        [data-testid="stImage"] {
            border-radius: 10px;
            overflow: hidden;
            border: 1px solid var(--line);
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            margin-bottom: 10px;
        }

        .stTabs [data-baseweb="tab"] {
            height: 44px;
            padding: 0 18px;
            border-radius: 8px;
            background: #ffffff;
            border: 1px solid var(--line);
            color: var(--muted);
            font-weight: 750;
        }

        .stTabs [aria-selected="true"] {
            color: #ffffff !important;
            background: var(--navy);
            border-color: var(--navy);
        }

        .stButton > button {
            width: 100%;
            min-height: 44px;
            border-radius: 8px;
            border: 1px solid var(--navy);
            background: var(--navy);
            color: #ffffff;
            font-weight: 800;
        }

        .stButton > button:hover {
            border-color: #1d4ed8;
            background: #1d4ed8;
            color: #ffffff;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


target_lab = get_target_lab()
target_text = "image mean" if target_lab is None else f"L {target_lab[0]:.1f} / a {target_lab[1]:.1f} / b {target_lab[2]:.1f}"

st.markdown(
    f"""
    <div class="report-header">
        <div>
            <div class="brand-mark">Snack Vision QC</div>
            <div class="report-title">고정밀 완제품 색차 분석 리포트</div>
            <div class="report-subtitle">
                고해상도 AI 마스킹과 float Lab 색공간 계산으로 평균색, Delta E 분포, P95 편차를 산출합니다.
                마스크 경계 픽셀은 분석에서 제외해 배경 혼입을 줄였습니다.
            </div>
        </div>
        <div class="model-strip">
            <div>Precision Profile</div>
            <span>Mask resolution: {MASK_DIM}px</span>
            <span>Color analysis: {ANALYSIS_DIM}px</span>
            <span>Segmentation model: {REMBG_MODEL}</span>
            <span>Reference: {target_text}</span>
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
            "기준색 비교가 필요하면 TARGET_L, TARGET_A, TARGET_B 값을 Railway Variables에 입력하세요."
        )

    with right_col:
        if uploaded_file is None:
            st.markdown(
                """
                <div class="section-card">
                    <div class="panel-heading">분석 결과 대기</div>
                    <div class="panel-caption" style="margin-top: 8px;">
                        이미지를 업로드하면 고정밀 마스킹, Delta E 분포, 평균 Lab, 판정 결과가 표시됩니다.
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

                with st.spinner("고정밀 AI 마스킹과 색차 분석을 실행 중입니다..."):
                    ai_session = load_ai_model(REMBG_MODEL)
                    result = analyze_image(original_img, ai_session)

                analysis_img = result["analysis_img"]
                visual_masked_img = result["visual_masked_img"]
                core_masked_img = result["core_masked_img"]
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
                    render_kpi("Sample Pixels", f"{sample_pixels:,}", "정밀 마스크 내부 픽셀", "blue")

                st.write("")
                status_col, color_col = st.columns([1, 1], gap="large")
                with status_col:
                    render_status_panel(status_meta, mean_delta_e, std_delta_e, p95_delta_e)
                with color_col:
                    render_color_chip(mean_bgr, avg_l, avg_a, avg_b)

                    fig, ax = plt.subplots(figsize=(4.8, 3.2))
                    ax.hist(delta_values, bins=80, color="#0f2742", edgecolor="white", linewidth=0.35)
                    ax.axvline(mean_delta_e, color="#d97706", linewidth=2, label="Mean")
                    ax.axvline(p95_delta_e, color="#dc2626", linewidth=2, label="P95")
                    ax.set_xlabel("Delta E", fontsize=9)
                    ax.set_ylabel("Pixel Count", fontsize=9)
                    ax.grid(axis="y", linestyle="--", alpha=0.24)
                    ax.legend(frameon=False, fontsize=8)
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    fig.patch.set_facecolor("#ffffff")
                    ax.set_facecolor("#ffffff")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                render_panel_title("시각 검증", "원본, 전체 마스크, 분석 코어 마스크, 실제 Delta E 히트맵을 비교합니다.")
                img_col1, img_col2 = st.columns(2, gap="medium")
                with img_col1:
                    st.markdown("**Raw Image**")
                    st.image(cv2.cvtColor(analysis_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                with img_col2:
                    st.markdown("**Full AI Mask**")
                    st.image(cv2.cvtColor(visual_masked_img, cv2.COLOR_BGR2RGB), use_container_width=True)

                img_col3, img_col4 = st.columns(2, gap="medium")
                with img_col3:
                    st.markdown("**Analysis Core Mask**")
                    st.image(cv2.cvtColor(core_masked_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                with img_col4:
                    st.markdown("**Delta E Heatmap**")
                    st.image(cv2.cvtColor(heatmap_masked, cv2.COLOR_BGR2RGB), use_container_width=True)

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
                del heatmap_masked, delta_values, result
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
                    st.line_chart(chart_df, color="#0f2742")

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
