"""EXP-005: GT mask 기준으로 silhouette 방법들의 IoU / Precision / Recall / 시간을 비교한다.

실행: python scripts/evaluate_silhouette.py
      python scripts/evaluate_silhouette.py --scales 0.5 0.33   # GrabCut 해상도 배율 지정
GrabCut ×1.0은 장당 수 초가 걸리므로 전체 실행에 3~5분 걸린다.
결과: 터미널에 Markdown 표, results/eval_silhouette.csv에 전체 기록
"""
import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from vmi.dataset import group_of  # noqa: E402
from vmi.io_utils import list_images, load_image, resize_max_side  # noqa: E402
from vmi.metrics import binary_metrics  # noqa: E402
from vmi.silhouette import grabcut_mask, otsu_mask, postprocess_mask  # noqa: E402


def build_methods(cfg: dict, scales: list[float]) -> dict:
    """{방법 이름: img -> mask 함수}. run_stage1과 같은 후처리를 적용한다."""
    s = cfg["silhouette"]
    methods = {"otsu": lambda img: postprocess_mask(otsu_mask(img, s["otsu_blur_ksize"]), s["morph_ksize"])}
    for scale in scales:
        methods[f"grabcut x{scale}"] = lambda img, scale=scale: postprocess_mask(
            grabcut_mask(img, s["grabcut_margin_x"], s["grabcut_margin_y"], s["grabcut_iterations"],
                         cfg["seed"], scale),
            s["morph_ksize"])
    return methods


def print_table(header: list[str], rows: list[list[str]]) -> None:
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for row in rows:
        print("| " + " | ".join(row) + " |")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scales", type=float, nargs="+", default=[1.0, 0.5, 0.33])
    args = parser.parse_args()

    with open(ROOT / "configs" / "stage1.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    methods = build_methods(cfg, args.scales)
    raw_dir = ROOT / cfg["input_dir"]

    records = []  # 한 행 = (이미지, 방법)
    print("## IoU per image\n")
    print("| file | " + " | ".join(methods) + " |")
    print("|" + "---|" * (len(methods) + 1))
    for gt_path in list_images(ROOT / "data" / "masks"):
        img = resize_max_side(load_image(raw_dir / f"{gt_path.stem}.jpg"), cfg["resize"]["max_side"])
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE) > 127  # JPG 압축으로 생긴 회색 경계 픽셀 처리
        if gt.shape != img.shape[:2]:
            raise ValueError(f"GT 크기 {gt.shape}가 이미지 크기 {img.shape[:2]}와 다릅니다: {gt_path.name}")

        ious = []
        for name, fn in methods.items():
            t0 = time.perf_counter()
            pred = fn(img)
            ms = (time.perf_counter() - t0) * 1000
            m = binary_metrics(pred, gt)
            records.append({"file": gt_path.stem, "group": group_of(gt_path.stem),
                            "method": name, "ms": round(ms, 1), **{k: round(v, 4) for k, v in m.items()}})
            ious.append(f"{m['iou']:.3f}")
        print(f"| {gt_path.stem} | " + " | ".join(ious) + " |", flush=True)

    # --- 방법별 요약 ---
    print("\n## Summary\n")
    rows = []
    for name in methods:
        r = [x for x in records if x["method"] == name]
        rows.append([name,
                     f"{statistics.mean(x['iou'] for x in r):.3f}",
                     f"{statistics.median(x['iou'] for x in r):.3f}",
                     f"{statistics.mean(x['precision'] for x in r):.3f}",
                     f"{statistics.mean(x['recall'] for x in r):.3f}",
                     f"{sum(x['iou'] >= 0.8 for x in r)}/{len(r)}",
                     f"{statistics.median(x['ms'] for x in r):.0f}"])
    print_table(["method", "mean IoU", "median IoU", "mean precision", "mean recall", "IoU>=0.8", "median ms"], rows)

    # --- 조건(그룹)별 평균 IoU ---
    print("\n## Mean IoU by group\n")
    groups = sorted({x["group"] for x in records})
    rows = []
    for g in groups:
        row = [f"{g} ({len({x['file'] for x in records if x['group'] == g})})"]
        for name in methods:
            r = [x["iou"] for x in records if x["group"] == g and x["method"] == name]
            row.append(f"{statistics.mean(r):.3f}")
        rows.append(row)
    print_table(["group"] + list(methods), rows)

    out = ROOT / cfg["output_dir"] / "eval_silhouette.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
