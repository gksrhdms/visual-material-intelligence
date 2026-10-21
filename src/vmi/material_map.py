"""Material Map: CV 분석 결과를 하나의 시각 언어(포스터)로 합친다.

CV 출력 → 시각적 의미 → 디자인 언어
  silhouette        → 형태        → 점이 찍히는 영역 + 가는 외곽선
  texture(local std) → 질감의 양   → 점의 크기 (하프톤 망점)
  칸의 평균색        → 소재의 색   → 점의 색
  edge              → 구조/재단선  → 잉크색 가는 선
  수치 지표          → 측정값      → 고정 스케일 막대 (포스터끼리 비교 가능)
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from vmi.color import to_hex

PAPER = (238, 235, 229)   # RGB
INK = (21, 21, 21)
MUTED = (138, 135, 128)
RULE = (207, 203, 194)
TRACK = (220, 216, 208)


def cell_means(values: np.ndarray, mask: np.ndarray, cell: int) -> tuple[np.ndarray, np.ndarray]:
    """칸(cell×cell)마다 mask 안 픽셀의 평균값과 mask가 칸을 덮는 비율을 반환한다.

    INTER_AREA 축소 = 칸 단위 평균. mask 밖 픽셀을 빼려고 resize(값×mask) / resize(mask)로 계산한다.
    """
    gh, gw = mask.shape[0] // cell, mask.shape[1] // cell
    m = (mask[:gh * cell, :gw * cell] > 0).astype(np.float32)
    v = values[:gh * cell, :gw * cell].astype(np.float32)
    weighted = v * (m[..., None] if v.ndim == 3 else m)

    cover = cv2.resize(m, (gw, gh), interpolation=cv2.INTER_AREA)
    mean = cv2.resize(weighted, (gw, gh), interpolation=cv2.INTER_AREA)
    safe = np.maximum(cover, 1e-6)
    mean = mean / (safe[..., None] if mean.ndim == 3 else safe)
    return mean, cover


def render_field(img, mask, std_map, edges, size, cell, std_scale, show_edges, gt=None) -> np.ndarray:
    """MATERIAL FIELD: 사람 영역을 격자로 나누고 칸마다 점 하나를 찍는다 (BGR 이미지).

    gt가 주어지면 예측 mask와 칸 단위로 비교해 실패를 드러낸다 (분석에는 쓰지 않고 표시만 한다):
      예측 O, GT O → 채운 점 (정상)
      예측 O, GT X → 회색 빈 원 (배경을 사람으로 잘못 포함)
      예측 X, GT O → 작은 회색 점 (사람을 놓침)
    """
    # 사진의 배경 여백이 그대로 종이 여백이 되지 않도록 사람 영역(bbox + 여유 16px)으로 먼저 자른다
    region = mask > 0 if gt is None else (mask > 0) | (gt > 0)
    ys, xs = np.nonzero(region)
    pad = 16
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, mask.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, mask.shape[1])
    img, mask, std_map, edges = (a[y0:y1, x0:x1] for a in (img, mask, std_map, edges))
    if gt is not None:
        gt = gt[y0:y1, x0:x1]

    fw, fh = size
    h, w = img.shape[:2]
    s = min(fw / w, fh / h)  # 원본 → field 배율
    ox, oy = round((fw - w * s) / 2), round((fh - h * s) / 2)  # 가운데 정렬 여백

    field = np.full((fh, fw, 3), PAPER[::-1], np.uint8)

    if show_edges:  # 구조선: 사람 영역 안쪽의 edge만 잉크색 35%로 얹는다 (배경 edge는 사람의 데이터가 아님)
        e = ((edges > 0) & (mask > 0)).astype(np.uint8) * 255
        e = cv2.resize(e, (round(w * s), round(h * s)), interpolation=cv2.INTER_NEAREST) > 0
        region = field[oy:oy + e.shape[0], ox:ox + e.shape[1]]
        region[e] = (0.35 * np.array(INK[::-1]) + 0.65 * region[e]).astype(np.uint8)

    colors, cover = cell_means(img, mask, cell)
    texture, _ = cell_means(std_map, mask, cell)
    predicted = cover >= 0.5  # 칸의 절반 이상이 사람(예측)인 칸
    if gt is None:
        truth = predicted
    else:
        gh, gw = cover.shape
        gt01 = (gt[:gh * cell, :gw * cell] > 0).astype(np.float32)
        truth = cv2.resize(gt01, (gw, gh), interpolation=cv2.INTER_AREA) >= 0.5

    SHIFT = 4  # 서브픽셀 좌표로 원을 그리기 위한 고정소수점 비트 수 (cv2.circle의 shift)
    muted = MUTED[::-1]
    for gy, gx in zip(*np.nonzero(predicted | truth)):
        cx, cy = ox + (gx + 0.5) * cell * s, oy + (gy + 0.5) * cell * s
        center = (round(cx * 2 ** SHIFT), round(cy * 2 ** SHIFT))
        half = 0.5 * cell * s
        if predicted[gy, gx] and truth[gy, gx]:
            strength = min(texture[gy, gx] / std_scale, 1.0)
            r = round(half * (0.3 + 0.7 * strength) * 2 ** SHIFT)  # 질감이 없어도 형태가 보이도록 최소 30%
            color = colors[gy, gx]
            cv2.circle(field, center, r, color.tolist(), -1, cv2.LINE_AA, SHIFT)
            # 흰 옷의 점이 종이색에 묻히지 않도록, 자기 색보다 20% 어두운 테두리를 두른다
            cv2.circle(field, center, r, (color * 0.8).tolist(), 1, cv2.LINE_AA, SHIFT)
        elif predicted[gy, gx]:  # 배경을 사람으로 잘못 포함 (false positive)
            cv2.circle(field, center, round(half * 0.6 * 2 ** SHIFT), muted, 1, cv2.LINE_AA, SHIFT)
        else:  # 사람을 놓침 (false negative)
            cv2.circle(field, center, round(half * 0.18 * 2 ** SHIFT), muted, -1, cv2.LINE_AA, SHIFT)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    scaled = [(c.astype(np.float32) * s + (ox, oy)).astype(np.int32) for c in contours]
    cv2.drawContours(field, scaled, -1, INK[::-1], 1, cv2.LINE_AA)
    return field


def load_font(path: str, size: int):
    try:
        return ImageFont.truetype(path, size)
    except OSError:  # 폰트가 없는 환경에서도 실행은 되도록
        return ImageFont.load_default(size)


def render_material_map(img, mask, std_map, edges, palette, stats: dict, title: str, subtitle: str,
                        cfg: dict, gt=None, badge: str | None = None, footer: str | None = None) -> np.ndarray:
    """포스터 한 장(BGR)을 만든다. stats: mean_chroma, achromatic, texture_std, texture_lap, area (+ iou).

    gt가 있으면 silhouette 실패를 그래픽과 IoU 수치로 드러낸다.
    badge: 시스템 출력이 아닌 참고용 포스터임을 표시하는 검은 띠 (예: GT silhouette로 만든 이상적인 경우)
    footer: 하단 재현 정보 (없으면 cfg["footer"])
    """
    W, H = cfg["canvas"]
    margin, gap = 48, 40
    col_w = 304
    field_w, field_h = W - 2 * margin - gap - col_w, H - 2 * margin
    col_x = margin + field_w + gap

    canvas = np.full((H, W, 3), PAPER[::-1], np.uint8)
    canvas[margin:margin + field_h, margin:margin + field_w] = render_field(
        img, mask, std_map, edges, (field_w, field_h), cfg["cell"], cfg["std_scale"], cfg["show_edges"], gt)

    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    f = lambda size: load_font(cfg["font"], size)  # noqa: E731
    mono = lambda size: load_font(cfg["font_mono"], size)  # noqa: E731

    def rule(y):
        d.line([(col_x, y), (col_x + col_w, y)], fill=RULE, width=1)

    # --- 헤더 ---
    y = margin
    d.text((col_x, y), "VISUAL MATERIAL INTELLIGENCE", font=f(13), fill=MUTED)
    d.text((col_x - 4, y + 18), title, font=f(112), fill=INK)
    d.text((col_x, y + 150), subtitle.upper(), font=f(18), fill=INK)
    y += 196
    rule(y)
    if badge:  # 참고용 포스터 표시: 헤더 바로 아래, 가장 눈에 띄는 위치
        y += 16
        d.rectangle([col_x, y, col_x + col_w, y + 30], fill=INK)
        d.text((col_x + 10, y + 6), badge, font=f(16), fill=PAPER)
        d.text((col_x, y + 38), "reference image, not system output", font=f(12), fill=MUTED)
        y += 60
        rule(y)

    # --- 팔레트: 비율 막대 + 색상 목록 ---
    y += 18
    d.text((col_x, y), "PALETTE", font=f(13), fill=MUTED)
    y += 26
    x = col_x
    for i, (color, ratio) in enumerate(palette):
        x_end = col_x + col_w if i == len(palette) - 1 else x + round(ratio * col_w)
        d.rectangle([x, y, x_end, y + 28], fill=tuple(int(c) for c in color[::-1]))
        x = x_end
    y += 40
    for color, ratio in palette:
        d.rectangle([col_x, y + 3, col_x + 11, y + 14], fill=tuple(int(c) for c in color[::-1]))
        d.text((col_x + 22, y), to_hex(color), font=mono(15), fill=INK)
        d.text((col_x + col_w, y), f"{ratio:.0%}", font=mono(15), fill=INK, anchor="ra")
        y += 22
    y += 14
    rule(y)

    # --- 측정값: 고정 스케일 막대 ---
    y += 18
    d.text((col_x, y), "MEASUREMENTS", font=f(13), fill=MUTED)
    y += 28
    lap_per_std = stats["texture_lap"] / stats["texture_std"] if stats["texture_std"] else 0.0
    rows = [
        ("CHROMA", "mean, LAB", stats["mean_chroma"], 40, f"{stats['mean_chroma']:.1f}"),
        ("ACHROMATIC", "chroma < 15", stats["achromatic"], 1, f"{stats['achromatic']:.0%}"),
        ("TEXTURE", "local std 7px", stats["texture_std"], 40, f"{stats['texture_std']:.1f}"),
        ("FREQUENCY", "laplacian / std", lap_per_std, 8, f"{lap_per_std:.2f}"),
        ("SILHOUETTE", "area of frame", stats["area"], 0.6, f"{stats['area']:.0%}"),
    ]
    if gt is not None:
        rows.append(("SILHOUETTE", "IoU vs GT mask", stats["iou"], 1, f"{stats['iou']:.2f}"))
    for label, unit, value, scale, text in rows:
        d.text((col_x, y), label, font=f(16), fill=INK)
        d.text((col_x + 108, y + 3), unit, font=f(12), fill=MUTED)
        d.text((col_x + col_w, y), text, font=mono(16), fill=INK, anchor="ra")
        y += 24
        d.rectangle([col_x, y, col_x + col_w, y + 3], fill=TRACK)
        d.rectangle([col_x, y, col_x + round(min(value / scale, 1.0) * col_w), y + 3], fill=INK)
        y += 22
    y += 4
    rule(y)

    # --- 범례: 그래픽의 각 요소가 무슨 데이터인지 ---
    y += 18
    d.text((col_x, y), "LEGEND", font=f(13), fill=MUTED)
    y += 28
    for r in (3, 6, 9):  # 점 크기 = texture
        cx = col_x + 10 + (r - 3) * 4
        d.ellipse([cx - r, y + 10 - r, cx + r, y + 10 + r], fill=INK)
    d.text((col_x + 64, y + 2), "dot size = texture strength", font=f(14), fill=INK)
    y += 30
    d.ellipse([col_x + 4, y + 4, col_x + 16, y + 16], fill=MUTED)
    d.text((col_x + 64, y + 2), "dot color = mean color of 8px cell", font=f(14), fill=INK)
    y += 30
    d.line([(col_x, y + 10), (col_x + 40, y + 10)], fill=(130, 130, 130), width=1)
    d.text((col_x + 64, y + 2), "line = edges (Canny)", font=f(14), fill=INK)
    if gt is not None:
        y += 30
        d.ellipse([col_x + 3, y + 4, col_x + 17, y + 18], outline=MUTED, width=1)
        d.text((col_x + 64, y + 2), "background taken as person", font=f(14), fill=INK)
        y += 30
        d.ellipse([col_x + 8, y + 9, col_x + 12, y + 13], fill=MUTED)
        d.text((col_x + 64, y + 2), "person missed", font=f(14), fill=INK)
        d.text((col_x + 64, y + 22), "ring / small dot: compared with GT mask", font=f(12), fill=MUTED)

    # --- 하단: 재현 정보 ---
    d.text((col_x, H - margin - 34), footer or cfg["footer"], font=mono(11), fill=MUTED)
    d.text((col_x, H - margin - 16), "PHASE 1 — CLASSICAL CV", font=f(12), fill=INK)

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
