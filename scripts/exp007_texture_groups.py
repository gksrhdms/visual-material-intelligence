"""EXP-007: texture 지표로 수집 조건(group)을 구분할 수 있는가?

run_stage1.py가 만든 results/timing.csv를 읽어 그룹별로 요약한다 (이미지를 다시 처리하지 않는다).
실행: python scripts/run_stage1.py → python scripts/exp007_texture_groups.py
"""
import csv
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from vmi.dataset import group_of  # noqa: E402

METRICS = ["texture_std", "texture_lap"]


def main():
    with open(ROOT / "results" / "timing.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["group"] = group_of(Path(r["file"]).stem)
        for m in METRICS:
            r[m] = float(r[m])
        r["lap_per_std"] = r["texture_lap"] / r["texture_std"] if r["texture_std"] else 0.0

    # 그룹별 범위: min~max가 다른 그룹과 겹치지 않아야 "구분 가능"
    print("## Texture by group (mean, min~max)\n")
    print("| group | n | local std | |laplacian| | lap / std |")
    print("|---|---|---|---|---|")
    for g in sorted({r["group"] for r in rows}):
        rs = [r for r in rows if r["group"] == g]
        cells = []
        for key in METRICS + ["lap_per_std"]:
            v = [r[key] for r in rs]
            cells.append(f"{statistics.mean(v):.2f} ({min(v):.2f}~{max(v):.2f})")
        print(f"| {g} | {len(rs)} | " + " | ".join(cells) + " |")

    # 이미지별 순위: 같은 그룹끼리 모여 있는지 눈으로 확인
    print("\n## Images sorted by local std\n")
    print("| rank | file | group | local std | |laplacian| | lap / std |")
    print("|---|---|---|---|---|---|")
    for i, r in enumerate(sorted(rows, key=lambda r: r["texture_std"]), start=1):
        print(f"| {i} | {Path(r['file']).stem} | {r['group']} | {r['texture_std']:.2f} | "
              f"{r['texture_lap']:.2f} | {r['lap_per_std']:.2f} |")


if __name__ == "__main__":
    main()
