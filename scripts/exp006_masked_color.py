"""EXP-006: color 통계를 어느 영역에서 재느냐 — 사진 전체 / GrabCut mask / GT mask.

- GrabCut과 GT의 차이 = silhouette 오차가 color 통계로 전파된 양 (error propagation)
- GT 사람 영역 안의 chroma 분포 → EXP-003의 T=15를 의상 기준으로 재확인

실행: python scripts/exp006_masked_color.py   (GrabCut 포함 약 10~20초)
"""
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from vmi.color import color_stats, lab_chroma  # noqa: E402
from vmi.io_utils import list_images, load_image, resize_max_side  # noqa: E402
from vmi.metrics import binary_metrics  # noqa: E402
from vmi.silhouette import grabcut_mask, postprocess_mask  # noqa: E402

BINS = [0, 5, 10, 12, 14, 16, 18, 20, 25, 30, 40, 256]  # EXP-003과 같은 구간


def main():
    with open(ROOT / "configs" / "stage1.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    c, s = cfg["color"], cfg["silhouette"]

    def achromatic(img, mask=None):
        stats, _ = color_stats(img, "lab", c["sat_threshold"], c["chroma_threshold"], c["dark_threshold"], mask)
        return stats["achromatic_lab"]

    print(f"## Achromatic ratio (LAB chroma < {c['chroma_threshold']}) by measured region\n")
    print("| file | whole | GrabCut mask | GT mask | GrabCut - GT | silhouette IoU |")
    print("|---|---|---|---|---|---|")

    errors, ious, person_chroma = [], [], []
    for gt_path in list_images(ROOT / "data" / "masks"):
        img = resize_max_side(load_image(ROOT / cfg["input_dir"] / f"{gt_path.stem}.jpg"), cfg["resize"]["max_side"])
        gt = (cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8) * 255
        pred = postprocess_mask(grabcut_mask(img, s["grabcut_margin_x"], s["grabcut_margin_y"],
                                             s["grabcut_iterations"], cfg["seed"], s["grabcut_scale"]),
                                s["morph_ksize"])

        whole, by_pred, by_gt = achromatic(img), achromatic(img, pred), achromatic(img, gt)
        iou = binary_metrics(pred, gt)["iou"]
        errors.append(abs(by_pred - by_gt))
        ious.append(iou)
        person_chroma.append(lab_chroma(img)[gt > 0])
        print(f"| {gt_path.stem} | {whole:.0%} | {by_pred:.0%} | {by_gt:.0%} | "
              f"{(by_pred - by_gt) * 100:+.0f}%p | {iou:.2f} |", flush=True)

    # silhouette이 좋을 때와 나쁠 때, color 오차가 얼마나 다른가
    good = [e for e, i in zip(errors, ious) if i >= 0.8]
    bad = [e for e, i in zip(errors, ious) if i < 0.8]
    print("\n## Error propagation: |GrabCut - GT| achromatic\n")
    print("| silhouette IoU | images | mean error | max error |")
    print("|---|---|---|---|")
    for label, group in [(">= 0.8", good), ("< 0.8", bad)]:
        if group:
            print(f"| {label} | {len(group)} | {statistics.mean(group) * 100:.1f}%p | {max(group) * 100:.1f}%p |")

    # GT 사람 영역 픽셀 전체의 chroma 밀도 (구간 폭으로 나눔 — EXP-003 정정 사항)
    hist, _ = np.histogram(np.concatenate(person_chroma), bins=BINS)
    density = hist / hist.sum() * 100 / np.diff(BINS)
    labels = [f"{a}-{b}" for a, b in zip(BINS[:-2], BINS[1:-1])] + [f"{BINS[-2]}+"]
    print("\n## Chroma density inside GT person masks (%/unit, all 25 images pooled)\n")
    print("| " + " | ".join(labels) + " |")
    print("|" + "---|" * len(labels))
    print("| " + " | ".join(f"{d:.2f}" for d in density) + " |")


if __name__ == "__main__":
    main()
