"""Silhouette(사람 영역 mask) 추출: Otsu threshold(baseline) vs GrabCut."""
import time

import cv2
import numpy as np


def largest_component(mask: np.ndarray) -> np.ndarray:
    """연결된 덩어리 중 가장 큰 것 하나만 남긴다 (사람은 한 덩어리라는 가정)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:  # 0번은 배경. 전경 덩어리가 하나도 없는 경우
        return mask
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == biggest, 255, 0).astype(np.uint8)


def postprocess_mask(mask: np.ndarray, ksize: int) -> np.ndarray:
    """Closing(구멍 메우기) → Opening(잡티 제거) → 가장 큰 덩어리만 남기기."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return largest_component(mask)


def otsu_mask(img: np.ndarray, blur_ksize: int) -> np.ndarray:
    """밝기 히스토그램을 두 그룹으로 나누는 Otsu threshold. 테두리에 많은 쪽을 배경으로 본다."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Otsu는 "밝은 쪽 / 어두운 쪽"만 나눈다. 어느 쪽이 사람인지는 모르므로
    # 이미지 테두리 픽셀의 과반이 속한 쪽을 배경으로 정한다.
    border = np.concatenate([binary[0], binary[-1], binary[:, 0], binary[:, -1]])
    if (border > 0).mean() > 0.5:
        binary = cv2.bitwise_not(binary)
    return binary


def grabcut_mask(img: np.ndarray, margin_x: float, margin_y: float, iterations: int, seed: int,
                 scale: float = 1.0) -> np.ndarray:
    """중앙 사각형으로 초기화한 GrabCut. 사각형 바깥은 확실한 배경으로 취급된다.

    scale < 1이면 축소한 이미지에서 계산하고 mask만 원래 크기로 되돌린다.
    768px 원본에서는 장당 3~13초가 걸려서 속도를 위해 도입했다 (EXP-005).
    """
    full_h, full_w = img.shape[:2]
    if scale < 1.0:
        img = cv2.resize(img, (round(full_w * scale), round(full_h * scale)), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    x0, y0 = round(w * margin_x), round(h * margin_y)
    rect = (x0, y0, w - 2 * x0, h - 2 * y0)  # (x, y, width, height)

    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)  # GrabCut 내부에서 쓰는 배경/전경 색 모델 저장 공간
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.setRNGSeed(seed)
    cv2.grabCut(img, mask, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)

    # 결과 라벨 4종: 확실한 배경 / 확실한 전경 / 아마 배경 / 아마 전경 → 전경 2종만 사람으로
    is_fg = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
    result = np.where(is_fg, 255, 0).astype(np.uint8)
    # mask는 0/255 라벨이므로 값을 섞지 않는 INTER_NEAREST로 확대한다
    return cv2.resize(result, (full_w, full_h), interpolation=cv2.INTER_NEAREST)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """두 mask의 IoU = 교집합 / 합집합."""
    a, b = a > 0, b > 0
    union = (a | b).sum()
    return float((a & b).sum() / union) if union > 0 else 0.0


def silhouette_analysis(img: np.ndarray, cfg: dict, seed: int) -> tuple[dict, dict]:
    """두 방법의 mask({"otsu": ..., "grabcut": ...})와 통계를 반환한다. 방법별 시간도 따로 잰다."""
    masks, stats = {}, {}

    t0 = time.perf_counter()
    masks["otsu"] = postprocess_mask(otsu_mask(img, cfg["otsu_blur_ksize"]), cfg["morph_ksize"])
    stats["otsu_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    raw = grabcut_mask(img, cfg["grabcut_margin_x"], cfg["grabcut_margin_y"], cfg["grabcut_iterations"], seed,
                       cfg["grabcut_scale"])
    masks["grabcut"] = postprocess_mask(raw, cfg["morph_ksize"])
    stats["grabcut_ms"] = (time.perf_counter() - t0) * 1000

    stats["area_otsu"] = float((masks["otsu"] > 0).mean())
    stats["area_grabcut"] = float((masks["grabcut"] > 0).mean())
    # 정답과의 비교가 아니라 두 방법이 서로 얼마나 일치하는지다 (정확도가 아님)
    stats["agreement_iou"] = mask_iou(masks["otsu"], masks["grabcut"])
    return masks, stats


def render_silhouette_panel(img, masks, stats, method) -> np.ndarray:
    """[원본 | Otsu mask | GrabCut mask | 선택한 방법의 cutout] 패널."""
    h, w = img.shape[:2]
    bar_h = 40
    bg, fg = (20, 20, 20), (230, 230, 230)
    font = cv2.FONT_HERSHEY_SIMPLEX

    canvas = np.full((h + bar_h, w * 4, 3), bg, dtype=np.uint8)
    canvas[:h, :w] = img
    canvas[:h, w:2 * w][masks["otsu"] > 0] = fg
    canvas[:h, 2 * w:3 * w][masks["grabcut"] > 0] = fg
    selected = masks[method] > 0
    canvas[:h, 3 * w:][selected] = img[selected]

    for i, label in enumerate(["original", "otsu", "grabcut", f"cutout ({method})"]):
        cv2.putText(canvas, label, (i * w + 10, 20), font, 0.45, (150, 150, 150), 1, cv2.LINE_AA)

    text = (f"otsu {stats['otsu_ms']:.1f} ms area {stats['area_otsu']:.0%}  |  "
            f"grabcut {stats['grabcut_ms']:.0f} ms area {stats['area_grabcut']:.0%}  |  "
            f"agreement IoU {stats['agreement_iou']:.2f}")
    cv2.putText(canvas, text, (10, h + 26), font, 0.45, fg, 1, cv2.LINE_AA)
    return canvas
