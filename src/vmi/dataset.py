"""데이터셋 메타 정보: 파일명 → 수집 조건(group), GT mask 찾기."""
from pathlib import Path

# 파일명 접미사 → 수집 조건. 표에 없는 접미사(plain, complexbg, pattern)는 그대로 쓴다.
GROUPS = {"leather": "glossy", "metalic": "glossy", "satin": "glossy",
          "sheer": "sheer", "mesh": "sheer",
          "allblack": "mono", "monochrome": "mono", "allwhite": "mono"}


def group_of(stem: str) -> str:
    """'016_leather' → 'glossy'"""
    suffix = stem.split("_", 1)[1]
    return GROUPS.get(suffix, suffix)


def find_gt_mask(mask_dir: Path, stem: str) -> Path | None:
    """data/masks에서 같은 이름의 GT mask를 찾는다 (.png 우선, 없으면 .jpg). 없으면 None."""
    for ext in (".png", ".jpg"):
        path = mask_dir / f"{stem}{ext}"
        if path.exists():
            return path
    return None
