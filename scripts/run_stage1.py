"""PHASE 1: Classical CV pipeline을 폴더 전체에 실행한다.

실행: python scripts/run_stage1.py
      python scripts/run_stage1.py --config configs/stage1.yaml
"""
import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))  # src/vmi를 import하기 위해 경로 추가

from vmi.dataset import find_gt_mask, group_of  # noqa: E402
from vmi.color import color_stats, dominant_colors, render_color_panel, to_hex  # noqa: E402
from vmi.edge import edge_analysis, render_edge_panel  # noqa: E402
from vmi.material_map import render_material_map  # noqa: E402
from vmi.metrics import binary_metrics  # noqa: E402
from vmi.io_utils import list_images, load_image, resize_max_side, save_image  # noqa: E402
from vmi.silhouette import render_silhouette_panel, silhouette_analysis  # noqa: E402
from vmi.texture import render_texture_panel, texture_analysis  # noqa: E402


def process_one(path: Path, cfg: dict, out_dir: Path) -> dict:
    """이미지 한 장을 처리하고 단계별 시간(ms)을 반환한다."""
    timing = {"file": path.name}

    t0 = time.perf_counter()
    img = load_image(path)
    timing["load_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    img = resize_max_side(img, cfg["resize"]["max_side"])
    timing["resize_ms"] = (time.perf_counter() - t0) * 1000

    h, w = img.shape[:2]
    timing["width"], timing["height"] = w, h

    save_image(img, out_dir / "original" / path.name)

    # --- Silhouette (Task 7) — 이후 단계가 사람 영역만 분석하도록 가장 먼저 실행 ---
    s = cfg["silhouette"]
    masks, stats = silhouette_analysis(img, s, cfg["seed"])  # 방법별 시간은 함수 안에서 잰다
    mask = masks[s["method"]]
    save_image(render_silhouette_panel(img, masks, stats, s["method"]), out_dir / "silhouette" / f"{path.stem}.png")
    save_image(mask, out_dir / "silhouette" / "mask" / f"{path.stem}.png")
    timing.update({key: round(value, 4) for key, value in stats.items()})

    # --- Color analysis (Task 3~4) — 사람 영역(mask) 안에서만 계산 (EXP-006) ---
    c = cfg["color"]
    t0 = time.perf_counter()
    stats, hue_hist = color_stats(img, c["method"], c["sat_threshold"], c["chroma_threshold"],
                                  c["dark_threshold"], mask)
    palette = dominant_colors(img, c["k"], c["sample_size"], cfg["seed"], mask)
    timing["color_ms"] = (time.perf_counter() - t0) * 1000

    whole, _ = color_stats(img, c["method"], c["sat_threshold"], c["chroma_threshold"], c["dark_threshold"])
    timing["achromatic_lab_whole"] = round(whole["achromatic_lab"], 3)  # 비교용: 사진 전체 기준

    cutout = np.full_like(img, 20)  # 패널에는 분석에 쓰인 사람 영역만 보여준다
    cutout[mask > 0] = img[mask > 0]
    save_image(render_color_panel(cutout, palette, hue_hist, stats), out_dir / "color" / f"{path.stem}.png")
    timing.update({key: round(value, 3) for key, value in stats.items()})
    timing["dominant_hex"] = to_hex(palette[0][0])

    # --- Texture (Task 8) — 사람 영역 안쪽에서만 측정 ---
    tx = cfg["texture"]
    t0 = time.perf_counter()
    std_map, lap_map, inner, stats = texture_analysis(img, mask, tx["window"])
    timing["texture_ms"] = (time.perf_counter() - t0) * 1000
    save_image(render_texture_panel(img, std_map, lap_map, inner, stats, tx["std_scale"], tx["lap_scale"]),
               out_dir / "texture" / f"{path.stem}.png")
    timing.update({key: round(value, 3) for key, value in stats.items()})

    # --- Edge & contour (Task 5~6) ---
    e = cfg["edge"]
    t0 = time.perf_counter()
    edges, contours, stats = edge_analysis(img, e["method"], e["blur_ksize"], e["fixed_low"], e["fixed_high"],
                                           e["auto_sigma"], e["min_contour_length"])
    timing["edge_ms"] = (time.perf_counter() - t0) * 1000

    save_image(render_edge_panel(img, edges, contours, stats, e["method"]), out_dir / "edge" / f"{path.stem}.png")
    timing.update({key: round(value, 4) for key, value in stats.items()})

    # --- Material Map (Task 9) — 앞 단계의 결과를 하나의 포스터로 ---
    # GT mask가 있으면 silhouette 실패를 포스터에 표시만 한다 (분석에는 쓰지 않음)
    gt, timing["gt_iou"] = None, None
    gt_path = find_gt_mask(ROOT / "data" / "masks", path.stem)
    if gt_path is not None:
        gt = np.where(cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE) > 127, 255, 0).astype(np.uint8)
        timing["gt_iou"] = round(binary_metrics(mask, gt)["iou"], 4)

    t0 = time.perf_counter()
    number, suffix = path.stem.split("_", 1)
    poster = render_material_map(
        img, mask, std_map, edges, palette,
        {"mean_chroma": timing["mean_chroma"], "achromatic": timing["achromatic_lab"],
         "texture_std": timing["texture_std"], "texture_lap": timing["texture_lap"],
         "area": timing[f"area_{s['method']}"], "iou": timing["gt_iou"]},
        gt=gt,
        title=number, cfg=cfg["material_map"],
        subtitle=suffix if group_of(path.stem) == suffix else f"{group_of(path.stem)} / {suffix}")
    timing["material_map_ms"] = (time.perf_counter() - t0) * 1000
    save_image(poster, out_dir / "material_map" / f"{path.stem}.png")
    return timing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "stage1.yaml"))
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    in_dir = ROOT / cfg["input_dir"]
    out_dir = ROOT / cfg["output_dir"]
    paths = list_images(in_dir)
    print(f"{len(paths)} images in {in_dir}")

    rows, failed = [], []
    for path in paths:
        try:
            row = process_one(path, cfg, out_dir)
        except Exception as e:  # 한 장이 실패해도 나머지는 계속 처리
            failed.append(path.name)
            print(f"  [FAIL] {path.name}: {e}")
            continue
        rows.append(row)
        print(f"  {row['file']:<20} grabcut {row['grabcut_ms']:6.1f} ms | texture {row['texture_ms']:5.1f} ms | "
              f"std {row['texture_std']:5.2f} (p90 {row['texture_std_p90']:5.2f}) | "
              f"laplacian {row['texture_lap']:5.2f} | map {row['material_map_ms']:5.1f} ms")

    if rows:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "timing.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        for key in [k for k in rows[0] if k.endswith("_ms")]:
            values = [r[key] for r in rows]
            # 첫 이미지의 warm-up 이상값이 mean을 끌어올리므로 median을 함께 본다 (EXP-001)
            print(f"{key:<10} mean {sum(values) / len(values):6.1f} ms | "
                  f"median {statistics.median(values):6.1f} ms | max {max(values):6.1f} ms")

    print(f"done: {len(rows)} ok, {len(failed)} failed -> {out_dir}")


if __name__ == "__main__":
    main()
