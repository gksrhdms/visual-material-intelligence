"""EXP-003: 유채색 판정 기준 비교 — HSV S>=40 vs LAB chroma>=T (T = 10, 15, 20).

이미지별 achromatic 비율을 Markdown 표로 출력한다. 표를 docs/experiments.md에 붙여넣어 기록한다.
실행: python scripts/exp003_chroma_threshold.py          # T별 achromatic 비율
      python scripts/exp003_chroma_threshold.py --dist   # 이미지별 chroma 분포
"""
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from vmi.color import color_stats, lab_chroma  # noqa: E402
from vmi.io_utils import list_images, load_image, resize_max_side  # noqa: E402

CHROMA_THRESHOLDS = [10, 15, 20]


def main():
    with open(ROOT / "configs" / "stage1.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    c = cfg["color"]

    header = ["file", f"hsv S>={c['sat_threshold']}"] + [f"lab T={t}" for t in CHROMA_THRESHOLDS]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))

    columns = [[] for _ in range(len(header) - 1)]
    for path in list_images(ROOT / cfg["input_dir"]):
        img = resize_max_side(load_image(path), cfg["resize"]["max_side"])

        values = []
        for i, t in enumerate(CHROMA_THRESHOLDS):
            stats, _ = color_stats(img, "lab", c["sat_threshold"], t, c["dark_threshold"])
            if i == 0:
                values.append(stats["achromatic_hsv"])  # HSV 값은 T와 무관하므로 한 번만
            values.append(stats["achromatic_lab"])

        for col, v in zip(columns, values):
            col.append(v)
        print(f"| {path.stem} | " + " | ".join(f"{v:.0%}" for v in values) + " |")

    print("| **mean** | " + " | ".join(f"**{sum(col) / len(col):.0%}**" for col in columns) + " |")


def chroma_distribution():
    """이미지별 chroma 구간 분포(%)를 출력한다.

    threshold는 픽셀이 몰려 있는 구간(peak)이 아니라 드문 구간(valley)에 두어야
    T를 조금 바꿨을 때 결과가 크게 흔들리지 않는다 (020: 10~12 구간에 64.8% 집중).
    """
    with open(ROOT / "configs" / "stage1.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    bins = [0, 5, 10, 12, 14, 16, 18, 20, 25, 30, 40, 256]
    header = ["file"] + [f"{a}-{b}" for a, b in zip(bins[:-1], bins[1:-1])] + [f"{bins[-2]}+"]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))

    rows = []
    for path in list_images(ROOT / cfg["input_dir"]):
        img = resize_max_side(load_image(path), cfg["resize"]["max_side"])
        hist, _ = np.histogram(lab_chroma(img), bins=bins)
        rows.append(hist / hist.sum())
        print(f"| {path.stem} | " + " | ".join(f"{v:.1%}" for v in rows[-1]) + " |")

    mean = np.mean(rows, axis=0)
    print("| **mean** | " + " | ".join(f"**{v:.1%}**" for v in mean) + " |")


if __name__ == "__main__":
    if "--dist" in sys.argv:
        chroma_distribution()
    else:
        main()
