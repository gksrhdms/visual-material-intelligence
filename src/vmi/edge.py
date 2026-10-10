"""Edge와 Contour 분석: Gaussian blur → Canny (fixed / auto threshold) → contour."""
import cv2
import numpy as np


def auto_canny_thresholds(gray: np.ndarray, sigma: float) -> tuple[int, int]:
    """밝기 중앙값(median) 기준으로 Canny threshold를 정한다: (1 ± sigma) × median."""
    median = float(np.median(gray))
    return int(max(0, (1 - sigma) * median)), int(min(255, (1 + sigma) * median))


def edge_analysis(img: np.ndarray, method: str, blur_ksize: int, fixed_low: int, fixed_high: int,
                  auto_sigma: float, min_contour_length: float) -> tuple[np.ndarray, list, dict]:
    """(선택한 방식의 edge map, 긴 contour 목록(길이 내림차순), 통계)를 반환한다.

    method: "fixed" | "auto" — contour와 패널에 사용할 방식.
    비교를 위해 두 방식의 edge density는 항상 함께 기록한다 (EXP-004).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.Canny는 내부에서 blur를 하지 않으므로 노이즈 제거를 직접 먼저 한다
    gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    thresholds = {
        "fixed": (fixed_low, fixed_high),
        "auto": auto_canny_thresholds(gray, auto_sigma),
    }
    edges = {name: cv2.Canny(gray, low, high) for name, (low, high) in thresholds.items()}
    selected = edges[method]

    # Edge(픽셀) → Contour(연결된 곡선). 길이를 재고 걸러낼 수 있게 된다.
    contours, _ = cv2.findContours(selected, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # 주의: 1픽셀 두께의 선을 한 바퀴 돌아 추적하므로 arcLength는 실제 선 길이의 약 2배다
    lengths = [cv2.arcLength(c, True) for c in contours]
    long_contours = sorted(
        (c for c, length in zip(contours, lengths) if length >= min_contour_length),
        key=lambda c: -cv2.arcLength(c, True),
    )

    stats = {
        "edge_density_fixed": float((edges["fixed"] > 0).mean()),
        "edge_density_auto": float((edges["auto"] > 0).mean()),
        "auto_low": thresholds["auto"][0],
        "auto_high": thresholds["auto"][1],
        "n_contours": len(contours),
        "n_contours_long": len(long_contours),
        "longest_contour": float(max(lengths, default=0.0)),
    }
    return selected, long_contours, stats


def render_edge_panel(img, edges, contours, stats, method) -> np.ndarray:
    """[원본 | edge map | 긴 contour] 패널. 가장 긴 contour 하나만 accent 색으로 강조한다."""
    h, w = img.shape[:2]
    bar_h = 40
    bg, fg, dim, accent = (20, 20, 20), (230, 230, 230), (95, 95, 95), (40, 80, 240)
    font = cv2.FONT_HERSHEY_SIMPLEX

    canvas = np.full((h + bar_h, w * 3, 3), bg, dtype=np.uint8)
    canvas[:h, :w] = img

    edge_view = canvas[:h, w:2 * w]
    edge_view[edges > 0] = fg

    contour_view = canvas[:h, 2 * w:]
    cv2.drawContours(contour_view, contours, -1, dim, 1, cv2.LINE_AA)
    cv2.drawContours(contour_view, contours[:5], -1, fg, 1, cv2.LINE_AA)
    cv2.drawContours(contour_view, contours[:1], -1, accent, 2, cv2.LINE_AA)

    text = (f"method {method}  |  density fixed {stats['edge_density_fixed']:.1%} "
            f"auto {stats['edge_density_auto']:.1%} (T {stats['auto_low']}/{stats['auto_high']})  |  "
            f"contours {stats['n_contours']} -> long {stats['n_contours_long']}  |  "
            f"longest {stats['longest_contour']:.0f}px")
    cv2.putText(canvas, text, (10, h + 26), font, 0.45, fg, 1, cv2.LINE_AA)
    return canvas
