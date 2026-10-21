"""데이터셋 메타 정보: 파일명 → 수집 조건(group). data/DATASET.md의 6개 조건과 일치시킨다."""

# 파일명 접미사 → 수집 조건. 표에 없는 접미사(plain, complexbg, pattern)는 그대로 쓴다.
GROUPS = {"leather": "glossy", "metalic": "glossy", "satin": "glossy",
          "sheer": "sheer", "mesh": "sheer",
          "allblack": "mono", "monochrome": "mono", "allwhite": "mono"}


def group_of(stem: str) -> str:
    """'016_leather' → 'glossy'"""
    suffix = stem.split("_", 1)[1]
    return GROUPS.get(suffix, suffix)
