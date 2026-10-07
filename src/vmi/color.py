"""색 분석: HSV/LAB 통계, Hue 히스토그램, LAB k-means 대표색."""
import cv2
import numpy as np

HUE_BIN = 5  # OpenCV Hue 범위는 0~179 (360도를 절반으로 저장) → 5단위로 36개 bin


def lab_chroma(img: np.ndarray) -> np.ndarray:
    """픽셀별 LAB chroma = 회색 축(a=b=0)으로부터의 거리. 밝기로 나누지 않는다."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    a, b = lab[..., 1] - 128, lab[..., 2] - 128  # OpenCV 8-bit LAB는 a, b에 +128 offset
    return np.hypot(a, b)


def color_stats(img: np.ndarray, method: str, sat_threshold: int, chroma_threshold: float,
                dark_threshold: int) -> tuple[dict, np.ndarray]:
    """색 통계와 유채색 픽셀의 Hue 히스토그램(합=1)을 반환한다.

    method: 유채색 판정 기준
      "hsv" — S >= sat_threshold  (S = (max-min)/max 라서 어둡거나 따뜻한 회색에서 과대평가됨, EXP-002)
      "lab" — chroma >= chroma_threshold  (밝기와 독립)
    비교를 위해 두 방법의 achromatic 비율은 항상 함께 기록한다.
    Hue 값은 두 방법 모두 HSV의 H를 쓰고, 판정 기준(마스크)만 바뀐다.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    chroma = lab_chroma(img)

    colorful = {"hsv": s >= sat_threshold, "lab": chroma >= chroma_threshold}
    chromatic = colorful[method] & (v >= dark_threshold)  # 너무 어두운 픽셀의 Hue는 noise

    hist = np.bincount(h[chromatic] // HUE_BIN, minlength=180 // HUE_BIN).astype(np.float64)
    if hist.sum() > 0:
        hist /= hist.sum()

    stats = {
        "mean_s": float(s.mean()),
        "mean_v": float(v.mean()),
        "mean_chroma": float(chroma.mean()),
        "achromatic_hsv": float((~colorful["hsv"]).mean()),
        "achromatic_lab": float((~colorful["lab"]).mean()),
        "dark_ratio": float((v < dark_threshold).mean()),
        "chromatic_ratio": float(chromatic.mean()),
    }
    return stats, hist


def dominant_colors(img: np.ndarray, k: int, sample_size: int, seed: int) -> list[tuple[np.ndarray, float]]:
    """LAB 공간에서 k-means로 대표색 k개를 찾아 [(BGR 색, 비율), ...]를 비율 내림차순으로 반환한다."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    pixels = lab.reshape(-1, 3).astype(np.float32)

    # 모든 픽셀(약 40만 개) 대신 무작위 샘플로 계산 → 속도 확보, 비율은 샘플로 추정
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pixels), size=min(sample_size, len(pixels)), replace=False)
    sample = pixels[idx]

    cv2.setRNGSeed(seed)  # k-means 초기 중심 선택을 재현 가능하게
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(sample, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)

    ratios = np.bincount(labels.ravel(), minlength=k) / len(labels)
    centers_bgr = cv2.cvtColor(centers.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_LAB2BGR).reshape(-1, 3)

    order = np.argsort(-ratios)
    return [(centers_bgr[i], float(ratios[i])) for i in order]


def to_hex(bgr: np.ndarray) -> str:
    b, g, r = (int(c) for c in bgr)
    return f"#{r:02X}{g:02X}{b:02X}"


def render_color_panel(img, palette, hue_hist, stats) -> np.ndarray:
    """[원본 | 대표색 팔레트] 위에 Hue 히스토그램과 통계를 붙인 패널 이미지를 만든다."""
    h, w = img.shape[:2]
    pal_w, hist_h = 180, 120
    bg, fg = (20, 20, 20), (230, 230, 230)
    font = cv2.FONT_HERSHEY_SIMPLEX

    canvas = np.full((h + hist_h, w + pal_w, 3), bg, dtype=np.uint8)
    canvas[:h, :w] = img

    # 팔레트: 비율만큼의 높이로 색 블록을 쌓는다
    y = 0
    for i, (color, ratio) in enumerate(palette):
        y_end = h if i == len(palette) - 1 else y + round(ratio * h)
        cv2.rectangle(canvas, (w, y), (w + pal_w, y_end), color.tolist(), -1)
        b, g, r = (int(c) for c in color)
        text_color = (20, 20, 20) if 0.299 * r + 0.587 * g + 0.114 * b > 140 else fg
        if y_end - y >= 18:  # 너무 얇은 블록에는 글자를 쓰지 않는다
            cv2.putText(canvas, f"{to_hex(color)} {ratio:.0%}", (w + 10, y + 16), font, 0.45, text_color, 1, cv2.LINE_AA)
        y = y_end

    # Hue 히스토그램: 각 막대를 해당 Hue의 색으로 칠한다.
    # 높이에 chromatic_ratio를 곱해, 유채색 픽셀이 적은 사진은 막대도 낮게 보이도록 한다 (EXP-002 원인 2).
    total_w = w + pal_w
    n_bins = len(hue_hist)
    bar_w = total_w / n_bins
    base_y, max_bar = h + hist_h - 30, hist_h - 45
    peak = hue_hist.max()
    for i, value in enumerate(hue_hist):
        if peak == 0:
            break
        bar_h = round(value / peak * max_bar * stats["chromatic_ratio"])
        hue = i * HUE_BIN + HUE_BIN // 2
        bar_color = cv2.cvtColor(np.uint8([[[hue, 200, 230]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
        x0, x1 = round(i * bar_w) + 1, round((i + 1) * bar_w) - 1
        cv2.rectangle(canvas, (x0, base_y - bar_h), (x1, base_y), bar_color, -1)
    if peak == 0:
        cv2.putText(canvas, "no chromatic pixels", (10, base_y - 10), font, 0.45, fg, 1, cv2.LINE_AA)

    text = (f"chromatic {stats['chromatic_ratio']:.0%}  |  achromatic hsv {stats['achromatic_hsv']:.0%} "
            f"lab {stats['achromatic_lab']:.0%}  |  dark {stats['dark_ratio']:.0%}  V {stats['mean_v']:.0f}")
    cv2.putText(canvas, text, (10, h + hist_h - 10), font, 0.45, fg, 1, cv2.LINE_AA)
    return canvas
