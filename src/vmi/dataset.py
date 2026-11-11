"""데이터셋 메타 정보: 파일명 → 수집 조건(group), GT mask 찾기·읽기."""
from pathlib import Path

import cv2
import numpy as np

# 파일명 접미사 → 수집 조건. 표에 없는 접미사(plain, complexbg, pattern)는 그대로 쓴다.
GROUPS = {"leather": "glossy", "metalic": "glossy", "satin": "glossy",
          "sheer": "sheer", "mesh": "sheer",
          "allblack": "mono", "monochrome": "mono", "allwhite": "mono"}


def group_of(stem: str) -> str:
    """'016_leather' → 'glossy'"""
    suffix = stem.split("_", 1)[1]
    return GROUPS.get(suffix, suffix)


def find_gt_mask(mask_dir: Path, stem: str) -> Path | None:
    """mask 폴더에서 GT mask를 찾는다. 없으면 None.

    찾는 순서: {stem}.png → {stem}.jpg → {stem}_mask.png (Photoroom 등 도구 출력 이름 그대로)
    """
    for name in (f"{stem}.png", f"{stem}.jpg", f"{stem}_mask.png"):
        path = mask_dir / name
        if path.exists():
            return path
    return None


def load_gt_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    """GT mask를 분석 해상도(shape = (h, w))의 0/255 이진 mask로 읽는다.

    도구마다 출력이 달라서(RGBA, 원본보다 작은 크기 등) 여기서 맞춘다:
    - 채널: 흑백으로 변환 (alpha가 실제로 쓰였다면 alpha를 사람 영역으로 본다)
    - 크기: 라벨이므로 값이 섞이지 않는 INTER_NEAREST로 맞춘다 (가로세로 비율이 같다는 전제)
    """
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise IOError(f"GT mask를 읽을 수 없습니다: {path}")
    if raw.ndim == 3 and raw.shape[2] == 4 and raw[..., 3].min() < 255:
        gray = raw[..., 3]  # 투명 배경 cutout: alpha = 사람 영역
    elif raw.ndim == 3:
        gray = cv2.cvtColor(raw[..., :3], cv2.COLOR_BGR2GRAY)  # 흰색 = 사람 영역
    else:
        gray = raw

    h, w = shape
    if abs(gray.shape[0] / gray.shape[1] - h / w) > 0.01:
        raise ValueError(f"GT mask 비율 {gray.shape}이 이미지 비율 {shape}과 다릅니다: {path.name}")
    if gray.shape != (h, w):
        gray = cv2.resize(gray, (w, h), interpolation=cv2.INTER_NEAREST)
    return np.where(gray > 127, 255, 0).astype(np.uint8)
