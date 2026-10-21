"""Texture 분석: local standard deviation과 |Laplacian|을 사람 영역(mask) 안쪽에서 측정한다."""
import cv2
import numpy as np


def local_std(gray: np.ndarray, window: int) -> np.ndarray:
    """픽셀마다 window×window 주변의 밝기 표준편차.

    분산 = E[x²] − E[x]² 이므로 박스 필터(cv2.blur) 두 번으로 빠르게 계산한다.
    """
    mean = cv2.blur(gray, (window, window))
    mean_sq = cv2.blur(gray * gray, (window, window))
    return np.sqrt(np.maximum(mean_sq - mean * mean, 0))  # 부동소수 오차로 생기는 음수 방지


def texture_analysis(img: np.ndarray, mask: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """(local std map, |Laplacian| map, 측정 영역 mask, 통계)를 반환한다."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    std = local_std(gray, window)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))

    # mask 경계에서는 창 안에 사람과 배경이 섞여 std가 커진다 (옷감이 아니라 실루엣에 반응).
    # 창의 반지름만큼 erode해서 창 전체가 사람 안에 들어가는 픽셀만 측정한다.
    inner = cv2.erode(mask, np.ones((window, window), np.uint8))
    if not inner.any():  # 아주 가는 mask라면 erode 후 남는 게 없을 수 있다
        inner = mask
    keep = inner > 0

    stats = {
        "texture_std": float(std[keep].mean()),
        "texture_std_p90": float(np.percentile(std[keep], 90)),
        "texture_lap": float(lap[keep].mean()),
        "inner_ratio": float(keep.sum() / max((mask > 0).sum(), 1)),  # erode 후 남은 비율
    }
    return std, lap, inner, stats


def render_texture_panel(img, std, lap, inner, stats, std_scale: float, lap_scale: float) -> np.ndarray:
    """[측정 영역 | local std map | |Laplacian| map] 패널.

    이미지마다 최대값으로 정규화하지 않고 고정 스케일을 써서, 패널끼리 밝기를 직접 비교할 수 있게 한다.
    """
    h, w = img.shape[:2]
    bar_h = 40
    bg, fg = (20, 20, 20), (230, 230, 230)
    font = cv2.FONT_HERSHEY_SIMPLEX
    keep = inner > 0

    canvas = np.full((h + bar_h, w * 3, 3), bg, dtype=np.uint8)
    canvas[:h, :w][keep] = img[keep]
    for i, (value_map, scale) in enumerate([(std, std_scale), (lap, lap_scale)], start=1):
        gray = np.clip(value_map / scale * 255, 0, 255).astype(np.uint8)
        canvas[:h, i * w:(i + 1) * w][keep] = gray[keep][:, None]  # 측정 영역만 표시

    for i, label in enumerate(["measured region", f"local std (0-{std_scale:g})", f"|laplacian| (0-{lap_scale:g})"]):
        cv2.putText(canvas, label, (i * w + 10, 20), font, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
    text = (f"std mean {stats['texture_std']:.1f}  p90 {stats['texture_std_p90']:.1f}  |  "
            f"laplacian mean {stats['texture_lap']:.1f}  |  inner {stats['inner_ratio']:.0%} of mask")
    cv2.putText(canvas, text, (10, h + 26), font, 0.45, fg, 1, cv2.LINE_AA)
    return canvas
