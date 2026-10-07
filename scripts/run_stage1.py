"""PHASE 1: Classical CV pipeline을 폴더 전체에 실행한다.

실행: python scripts/run_stage1.py
      python scripts/run_stage1.py --config configs/stage1.yaml
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))  # src/vmi를 import하기 위해 경로 추가

from vmi.color import dominant_colors, hsv_stats, render_color_panel, to_hex  # noqa: E402
from vmi.io_utils import list_images, load_image, resize_max_side, save_image  # noqa: E402


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

    # --- Color analysis (Task 3~4) ---
    c = cfg["color"]
    t0 = time.perf_counter()
    stats, hue_hist = hsv_stats(img, c["sat_threshold"], c["dark_threshold"])
    palette = dominant_colors(img, c["k"], c["sample_size"], cfg["seed"])
    timing["color_ms"] = (time.perf_counter() - t0) * 1000

    save_image(render_color_panel(img, palette, hue_hist, stats), out_dir / "color" / f"{path.stem}.png")
    timing.update({key: round(value, 3) for key, value in stats.items()})
    timing["dominant_hex"] = to_hex(palette[0][0])
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
        print(f"  {row['file']:<24} {row['width']}x{row['height']:<5} "
              f"load {row['load_ms']:6.1f} ms | resize {row['resize_ms']:5.1f} ms | "
              f"color {row['color_ms']:6.1f} ms | achromatic {row['achromatic_ratio']:.0%}")

    if rows:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "timing.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        for key in [k for k in rows[0] if k.endswith("_ms")]:
            values = [r[key] for r in rows]
            print(f"{key:<10} mean {sum(values) / len(values):6.1f} ms | max {max(values):6.1f} ms")

    print(f"done: {len(rows)} ok, {len(failed)} failed -> {out_dir}")


if __name__ == "__main__":
    main()
