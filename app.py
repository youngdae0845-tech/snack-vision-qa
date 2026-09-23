import os
import gc
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from rembg import new_session, remove
from supabase import create_client


st.set_page_config(page_title="스낵 완제품 색차분석기", layout="wide")

# Railway Variables에서 조절 가능
MASK_DIM = int(os.environ.get("MASK_DIM", "900"))          # AI 누끼용 해상도
ANALYSIS_DIM = int(os.environ.get("ANALYSIS_DIM", "1600")) # 색차 분석용 해상도
REMBG_MODEL = os.environ.get("REMBG_MODEL", "u2net")


def get_secret(name):
    try:
        return st.secrets[name]
    except Exception:
        return os.environ.get(name)


SUPABASE_URL = get_secret("SUPABASE_URL")
SUPABASE_KEY = get_secret("SUPABASE_KEY")


@st.cache_resource
def get_supabase_client():
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    return create_client(SUPABASE_URL, SUPABASE_KEY)


@st.cache_resource
def load_ai_model():
    return new_session(REMBG_MODEL)


def resize_by_max_dim(img, max_dim):
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img

    scale = max_dim / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def create_large_mask(original_img, analysis_img, session):
    mask_input_img = resize_by_max_dim(original_img, MASK_DIM)

    no_bg_img = remove(mask_input_img, session=session)

    if no_bg_img.ndim < 3 or no_bg_img.shape[2] < 4:
        raise ValueError("배경 제거 결과에 alpha 채널이 없습니다.")

    small_alpha = no_bg_img[:, :, 3]

    large_mask = cv2.resize(
        small_alpha,
        (analysis_img.shape[1], analysis_img.shape[0]),
        interpolation=cv2.INTER_LINEAR
    )

    _, large_mask = cv2.threshold(large_mask, 127, 255, cv2.THRESH_BINARY)

    # 가장자리 배경 픽셀이 색차 분석에 섞이는 것을 줄임
    kernel = np.ones((3, 3), np.uint8)
    large_mask = cv2.erode(large_mask, kernel, iterations=1)

    return large_mask


def analyze_image(original_img, session):
    analysis_img = resize_by_max_dim(original_img, ANALYSIS_DIM)
    mask = create_large_mask(original_img, analysis_img, session)

    if np.count_nonzero(mask) == 0:
        raise ValueError("스낵 영역을 감지하지 못했습니다.")

    black_bg_img = cv2.bitwise_and(analysis_img, analysis_img, mask=mask)

    lab_img = cv2.cvtColor(analysis_img, cv2.COLOR_BGR2LAB)
    mean_lab = cv2.mean(lab_img, mask=mask)
    mean_bgr = cv2.mean(analysis_img, mask=mask)

    avg_l = mean_lab[0] * (100 / 255)
    avg_a = mean_lab[1] - 128
    avg_b = mean_lab[2] - 128

    lab_float = lab_img.astype(np.float32)

    delta_e_map = np.sqrt(
        ((lab_float[:, :, 0] - mean_lab[0]) * (100 / 255)) ** 2 +
        (lab_float[:, :, 1] - mean_lab[1]) ** 2 +
        (lab_float[:, :, 2] - mean_lab[2]) ** 2
    )

    valid_pixels = delta_e_map[mask > 0]

    mean_delta_e = float(np.mean(valid_pixels)) if len(valid_pixels) > 0 else 0
    std_delta_e = float(np.std(valid_pixels)) if len(valid_pixels) > 0 else 0

    a_channel = lab_float[:, :, 1] - 128
    min_redness, max_redness = 10.0, 28.0

    ratio = np.clip((a_channel - min_redness) / (max_redness - min_redness), 0, 1)
    hue = np.uint8(60 - (ratio * 60))
    sat = np.full_like(hue, 255)
    val = np.full_like(hue, 255)

    custom_bgr = cv2.cvtColor(cv2.merge([hue, sat, val]), cv2.COLOR_HSV2BGR)
    blended_heatmap = cv2.addWeighted(black_bg_img, 0.6, custom_bgr, 0.4, 0)
    heatmap_masked = cv2.bitwise_and(blended_heatmap, blended_heatmap, mask=mask)

    result = {
        "analysis_img": analysis_img,
        "black_bg_img": black_bg_img,
        "heatmap_masked": heatmap_masked,
        "valid_pixels": valid_pixels,
        "mean_bgr": mean_bgr,
        "avg_l": avg_l,
        "avg_a": avg_a,
        "avg_b": avg_b,
        "mean_delta_e": mean_delta_e,
        "std_delta_e": std_delta_e,
    }

    del lab_img, lab_float, delta_e_map, custom_bgr, blended_heatmap, mask
    gc.collect()

    return result


supabase = get_supabase_client()


st.markdown("""
<style>
    .stApp {
        background-color: #F4F6F9;
    }
    h1 {
        color: #112A46;
        font-weight: 800;
        text-align: center;
        padding-bottom: 18px;
        border-bottom: 3px solid #112A46;
        margin-bottom: 28px;
    }
    div[data-testid="metric-container"] {
        background-color: white;
        border: 1px solid #E2E8F0;
        padding: 18px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(15, 23, 42, 0.06);
        border-left: 5px solid #112A46;
    }
    [data-testid="stImage"] {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 8px rgba(15, 23, 42, 0.08);
    }
</style>
""", unsafe_allow_html=True)


st.title("🏭 스낵 완제품 색차분석기")

tab1, tab2 = st.tabs(["📸 스낵 색차 분석실", "📈 누적 품질 통계"])


with tab1:
    st.markdown("##### 📌 스낵 완제품 색차분석기")

    uploaded_file = st.file_uploader(
        "스낵 사진을 업로드하거나 촬영하세요",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        try:
            file_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
            original_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if original_img is None:
                st.error("이미지를 읽지 못했습니다. JPG 또는 PNG 파일을 다시 업로드해주세요.")
                st.stop()

            with st.spinner("AI가 스낵 영역과 색차(ΔE)를 분석 중입니다..."):
                ai_session = load_ai_model()
                result = analyze_image(original_img, ai_session)

            analysis_img = result["analysis_img"]
            black_bg_img = result["black_bg_img"]
            heatmap_masked = result["heatmap_masked"]
            valid_pixels = result["valid_pixels"]
            mean_bgr = result["mean_bgr"]
            avg_l = result["avg_l"]
            avg_a = result["avg_a"]
            avg_b = result["avg_b"]
            mean_delta_e = result["mean_delta_e"]
            std_delta_e = result["std_delta_e"]

            st.caption(
                f"누끼 해상도: {MASK_DIM}px / 색차 분석 해상도: {ANALYSIS_DIM}px / 모델: {REMBG_MODEL}"
            )

            st.markdown("<br>", unsafe_allow_html=True)

            img_col1, img_col2 = st.columns(2)

            with img_col1:
                st.markdown("**RAW IMAGE**")
                st.image(cv2.cvtColor(analysis_img, cv2.COLOR_BGR2RGB), use_container_width=True)

            with img_col2:
                st.markdown("**AI MASKING**")
                st.image(cv2.cvtColor(black_bg_img, cv2.COLOR_BGR2RGB), use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True)

            col_tile, col_dist, col_map, col_metrics = st.columns([1.2, 2, 1.5, 1])

            with col_tile:
                st.markdown("**AVG COLOR TILE**")
                avg_color_tile = np.zeros((300, 200, 3), dtype=np.uint8)
                avg_color_tile[:] = mean_bgr[:3]
                tile_rgb = cv2.cvtColor(avg_color_tile, cv2.COLOR_BGR2RGB)
                st.image(tile_rgb, use_container_width=True)
                st.markdown(
                    f"<div style='text-align: center; margin-top: 10px;'>"
                    f"L: {avg_l:.0f} &nbsp;&nbsp; a: {avg_a:.0f} &nbsp;&nbsp; b: {avg_b:.0f}"
                    f"</div>",
                    unsafe_allow_html=True
                )

            with col_dist:
                st.markdown("**ΔE DISTRIBUTION**")
                fig, ax = plt.subplots(figsize=(4.5, 3.8))
                ax.hist(valid_pixels, bins=60, color="#112A46", edgecolor="white", linewidth=0.5)
                ax.set_xlabel("Color Difference (ΔE)", fontsize=9)
                ax.set_ylabel("Pixel Count", fontsize=9)
                ax.grid(axis="y", linestyle="--", alpha=0.3)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                fig.patch.set_facecolor("#F4F6F9")
                ax.set_facecolor("#F4F6F9")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            with col_map:
                st.markdown("**HEATMAP**")
                st.image(cv2.cvtColor(heatmap_masked, cv2.COLOR_BGR2RGB), use_container_width=True)

            with col_metrics:
                st.markdown("**QA METRICS**")

                dev_level = (
                    "Very High (NG)"
                    if mean_delta_e >= 20
                    else ("High (Review)" if mean_delta_e >= 10 else "Normal (OK)")
                )

                status_color = "#E53E3E" if mean_delta_e >= 20 else "#38A169"

                st.markdown(f"""
                <div style='line-height: 2.2; font-size: 15px; margin-top: 30px; background-color: white; padding: 15px; border-radius: 8px; border: 1px solid #E2E8F0; box-shadow: 0 2px 4px rgba(0,0,0,0.05);'>
                    <b>Mean ΔE:</b><br>
                    <span style='font-size: 20px; color: #112A46; font-weight: bold;'>{mean_delta_e:.1f}</span><br>
                    <b>Std Dev:</b><br>{std_delta_e:.1f}<br>
                    <b>Status:</b><br>
                    <span style='color: {status_color}; font-weight: bold;'>{dev_level}</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            lot_input = st.text_input("LOT NUMBER", "LOT-2026-005")

            status = "NG" if mean_delta_e >= 20 else "OK"
            defect_type = "색상 불균일(시즈닝 뭉침)" if status == "NG" else "정상"

            if st.button("💾 클라우드 DB에 검사 결과 전송"):
                if supabase is None:
                    st.error("Supabase 환경변수가 설정되지 않았습니다. Railway Variables를 확인해주세요.")
                else:
                    log_data = {
                        "lot_number": lot_input,
                        "avg_l": round(avg_l, 2),
                        "avg_a": round(avg_a, 2),
                        "avg_b": round(avg_b, 2),
                        "delta_e": round(mean_delta_e, 2),
                        "defect_type": defect_type,
                        "status": status,
                        "image_path": "web_upload.jpg"
                    }

                    try:
                        supabase.table("snack_color_logs").insert(log_data).execute()
                        st.success("✅ 수파베이스에 기록이 완료되었습니다.")
                    except Exception as e:
                        st.error(f"저장 실패: {e}")

            del original_img, analysis_img, black_bg_img, heatmap_masked, valid_pixels, result
            gc.collect()

        except Exception as e:
            st.error("분석 중 오류가 발생했습니다.")
            st.exception(e)


with tab2:
    st.markdown("##### 📊 실시간 누적 품질 통계")

    if st.button("🔄 실시간 DB 동기화"):
        st.rerun()

    if supabase is None:
        st.warning("Supabase 환경변수가 설정되지 않아 통계 DB를 불러올 수 없습니다.")
    else:
        try:
            response = (
                supabase
                .table("snack_color_logs")
                .select("*")
                .order("created_at", desc=False)
                .execute()
            )
            data = response.data

            if data:
                df = pd.DataFrame(data)

                total_count = len(df)
                ng_count = len(df[df["status"] == "NG"])
                ng_rate = (ng_count / total_count) * 100 if total_count > 0 else 0
                avg_delta_e = df["delta_e"].mean()

                metric1, metric2, metric3 = st.columns(3)
                metric1.metric("Total Inspections", f"{total_count} LOTs")
                metric2.metric("Defect Rate (NG)", f"{ng_rate:.1f}%")
                metric3.metric("Average ΔE", f"{avg_delta_e:.1f}")

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**📈 LOT별 색상 편차(ΔE) 추이**")
                st.line_chart(df.set_index("lot_number")["delta_e"], color="#112A46")

                st.markdown("**📋 전체 검사 로그 DB**")
                st.dataframe(
                    df[[
                        "created_at",
                        "lot_number",
                        "avg_l",
                        "avg_a",
                        "avg_b",
                        "delta_e",
                        "defect_type",
                        "status"
                    ]],
                    use_container_width=True
                )
            else:
                st.info("DB에 저장된 데이터가 없습니다.")

        except Exception as e:
            st.error(f"데이터를 불러오지 못했습니다: {e}")
