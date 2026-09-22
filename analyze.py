import cv2
import numpy as np
import matplotlib.pyplot as plt
from rembg import remove
from supabase import create_client, Client

# ==========================================
# 1. 수파베이스 연결 설정 (본인 정보 입력)
# ==========================================
SUPABASE_URL = "https://yxetrhoifimlblursgbi.supabase.co"
SUPABASE_KEY = "sb_publishable_IB_TxAiC7W46BviCFrtVyQ_NEPdsI_V"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# 2. 이미지 불러오기
image_path = "sample.jpg" 
img = cv2.imread(image_path)

if img is None:
    print(f"오류: [{image_path}] 사진을 찾을 수 없습니다.")
else:
    print("종합 분석 및 DB 자동 저장을 시작합니다...")

    # 3. AI 누끼 따기
    no_bg_img = remove(img)
    alpha_channel = no_bg_img[:, :, 3]
    _, mask = cv2.threshold(alpha_channel, 10, 255, cv2.THRESH_BINARY)
    black_bg_img = cv2.bitwise_and(img, img, mask=mask)

    # 4. 평균 색상값 계산
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    mean_lab = cv2.mean(lab_img, mask=mask)
    mean_bgr = cv2.mean(img, mask=mask)

    avg_l = mean_lab[0] * (100 / 255)
    avg_a = mean_lab[1] - 128
    avg_b = mean_lab[2] - 128

    print(f"[분석 완료] L: {avg_l:.1f}, a: {avg_a:.1f}, b: {avg_b:.1f}")

    # 기능 1: Average Color Tile 
    avg_color_tile = np.zeros((300, 300, 3), dtype=np.uint8)
    avg_color_tile[:] = mean_bgr[:3]

    # 5. 전체 색상 편차(Delta E) 계산 
    lab_float = lab_img.astype(np.float32)
    delta_e_map = np.sqrt(((lab_float[:,:,0] - mean_lab[0]) * (100/255))**2 + 
                          (lab_float[:,:,1] - mean_lab[1])**2 + 
                          (lab_float[:,:,2] - mean_lab[2])**2)
    
    valid_pixels = delta_e_map[mask > 0]
    mean_delta_e = float(np.mean(valid_pixels)) if len(valid_pixels) > 0 else 0

    # 합격/불합격 판정 (기준치 예시: 편차 30 이상이면 불량)
    status = "OK"
    defect_type = "정상"
    if mean_delta_e >= 30:
        status = "NG"
        defect_type = "색상 불균일"

    # ==========================================
    # 6. 수파베이스 DB에 데이터 전송
    # ==========================================
    log_data = {
        "lot_number": "LOT-0515-A", # 임의의 로트 번호
        "avg_l": round(avg_l, 2),
        "avg_a": round(avg_a, 2),
        "avg_b": round(avg_b, 2),
        "delta_e": round(mean_delta_e, 2),
        "defect_type": defect_type,
        "status": status,
        "image_path": image_path
    }

    try:
        supabase.table("snack_color_logs").insert(log_data).execute()
        print(">>> 수파베이스 DB 기록 완료! <<<")
    except Exception as e:
        print("DB 전송 오류:", e)

    # 기능 2: a* 강도 기반 자연스러운 열지도
    a_channel = lab_float[:,:,1] - 128
    min_redness = 10.0   
    max_redness = 28.0  

    ratio = np.clip((a_channel - min_redness) / (max_redness - min_redness), 0, 1)
    hue = np.uint8(60 - (ratio * 60))
    sat = np.full_like(hue, 255)
    val = np.full_like(hue, 255)

    custom_bgr = cv2.cvtColor(cv2.merge([hue, sat, val]), cv2.COLOR_HSV2BGR)
    blended_heatmap = cv2.addWeighted(black_bg_img, 0.6, custom_bgr, 0.4, 0)
    heatmap_masked = cv2.bitwise_and(blended_heatmap, blended_heatmap, mask=mask)

    # 창 띄우기
    cv2.imshow("1. AI Background Removed", cv2.resize(black_bg_img, (500, 350)))
    cv2.imshow("2. Average Color Tile", avg_color_tile)
    cv2.imshow("3. Natural Heatmap", cv2.resize(heatmap_masked, (500, 350)))
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 기능 3: Color Deviation Distribution 
    plt.figure(figsize=(8, 5))
    plt.hist(valid_pixels, bins=60, color='darkorange', edgecolor='black', alpha=0.7)
    plt.title('Color Deviation (Delta E) Distribution')
    plt.xlabel('Delta E (Difference from Average)')
    plt.ylabel('Pixel Count')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()