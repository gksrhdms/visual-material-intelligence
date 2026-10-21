"""PHASE 1 결과를 한눈에 보는 contact sheet를 만든다 (run_stage1.py 결과를 모으기만 한다).

1. results/sheets/material_map_grid.png — Material Map 포스터 전체 그리드
2. results/sheets/stages.png — 대표 이미지의 단계별 비교: 원본 → silhouette → texture → Material Map → GT 포스터

실행: python scripts/make_contact_sheet.py
      python scripts/make_contact_sheet.py --stems 002_plain 013_pattern 007_complexbg
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from vmi.io_utils import list_images, load_image, save_image  # noqa: E402
from vmi.material_map import INK, MUTED, PAPER, load_font  # noqa: E402

RESULTS = ROOT / "results"
FONT = "C:/Windows/Fonts/bahnschrift.ttf"


def fit(img: np.ndarray, box: tuple[int, int]) -> np.ndarray:
    """비율을 유지하며 box(w, h) 안에 맞추고, 남는 곳은 종이색으로 채운다."""
    bw, bh = box
    h, w = img.shape[:2]
    s = min(bw / w, bh / h)
    resized = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    out = np.full((bh, bw, 3), PAPER[::-1], np.uint8)
    y, x = (bh - resized.shape[0]) // 2, (bw - resized.shape[1]) // 2
    out[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return out


def add_text(canvas: np.ndarray, items: list[tuple[tuple[int, int], str, int, tuple]]) -> np.ndarray:
    """[(위치, 문자열, 크기, 색), ...]를 Pillow로 그린다 (OpenCV는 타이포 품질이 낮다)."""
    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    for xy, text, size, color in items:
        d.text(xy, text, font=load_font(FONT, size), fill=color)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def material_map_grid(cols: int = 5, cell: tuple[int, int] = (324, 405), gap: int = 12, margin: int = 48):
    posters = list_images(RESULTS / "material_map")
    rows = -(-len(posters) // cols)  # 올림 나눗셈
    header = 70
    W = 2 * margin + cols * cell[0] + (cols - 1) * gap
    H = 2 * margin + header + rows * cell[1] + (rows - 1) * gap
    canvas = np.full((H, W, 3), PAPER[::-1], np.uint8)
    for i, path in enumerate(posters):
        r, c = divmod(i, cols)
        x, y = margin + c * (cell[0] + gap), margin + header + r * (cell[1] + gap)
        canvas[y:y + cell[1], x:x + cell[0]] = fit(load_image(path), cell)
    return add_text(canvas, [
        ((margin, margin), "VISUAL MATERIAL INTELLIGENCE — PHASE 1", 28, INK),
        ((margin, margin + 36), f"{len(posters)} material maps · system output (GrabCut silhouette) · "
                                "hollow rings = background taken as person, vs GT", 15, MUTED),
    ])


def stages(stems: list[str], cell: tuple[int, int] = (300, 405), gap: int = 12, margin: int = 48):
    labels = ["ORIGINAL", "SILHOUETTE · GrabCut", "TEXTURE · local std", "MATERIAL MAP", "IDEAL · GT silhouette"]
    header, label_h = 70, 28
    W = 2 * margin + len(labels) * cell[0] + (len(labels) - 1) * gap
    H = 2 * margin + header + label_h + len(stems) * cell[1] + (len(stems) - 1) * gap
    canvas = np.full((H, W, 3), PAPER[::-1], np.uint8)
    texts = [((margin, margin), "PIPELINE — FROM PHOTO TO MATERIAL MAP", 28, INK),
             ((margin, margin + 36), "each column is a saved output of run_stage1.py", 15, MUTED)]
    for c, label in enumerate(labels):
        texts.append(((margin + c * (cell[0] + gap), margin + header), label, 14, MUTED))

    for r, stem in enumerate(stems):
        original = load_image(RESULTS / "original" / f"{stem}.jpg")
        h, w = original.shape[:2]
        mask = cv2.imread(str(RESULTS / "silhouette" / "mask" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        cutout = np.full_like(original, PAPER[::-1])
        cutout[mask > 0] = original[mask > 0]
        texture = load_image(RESULTS / "texture" / f"{stem}.png")[:h, w:2 * w]  # 패널의 가운데 열 = local std map
        tiles = [original, cutout, texture,
                 load_image(RESULTS / "material_map" / f"{stem}.png"),
                 load_image(RESULTS / "material_map_gt" / f"{stem}.png")]
        y = margin + header + label_h + r * (cell[1] + gap)
        for c, tile in enumerate(tiles):
            x = margin + c * (cell[0] + gap)
            canvas[y:y + cell[1], x:x + cell[0]] = fit(tile, cell)
    return add_text(canvas, texts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stems", nargs="+", default=["002_plain", "013_pattern", "017_metalic", "007_complexbg"])
    args = parser.parse_args()

    out = RESULTS / "sheets"
    save_image(material_map_grid(), out / "material_map_grid.png")
    save_image(stages(args.stems), out / "stages.png")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
