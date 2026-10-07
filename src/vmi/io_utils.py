"""이미지 입출력과 resize."""
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_images(folder: Path) -> list[Path]:
    """폴더 안의 이미지 파일 경로를 이름순으로 반환한다."""
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def load_image(path: Path) -> np.ndarray:
    """이미지를 BGR uint8 배열로 읽는다. 실패하면 예외를 던진다."""
    img = cv2.imread(str(path))
    if img is None:
        raise IOError(f"이미지를 읽을 수 없습니다: {path}")
    return img


def resize_max_side(img: np.ndarray, max_side: int) -> np.ndarray:
    """긴 변이 max_side보다 크면 비율을 유지하며 축소한다. 작으면 그대로 반환한다.

    확대하지 않는 이유: 저해상도 이미지를 키워도 디테일은 생기지 않고
    보간으로 만들어진 흐릿한 질감이 texture 분석을 왜곡한다.
    """
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return img
    new_size = (round(w * scale), round(h * scale))  # cv2는 (width, height) 순서
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)


def save_image(img: np.ndarray, path: Path) -> None:
    """상위 폴더가 없으면 만들고 이미지를 저장한다.

    cv2.imwrite는 실패해도 False만 반환하고 이유를 알려주지 않는다.
    인코딩(cv2.imencode)과 파일 쓰기(Python)를 분리해서, 쓰기 실패 시
    PermissionError 등 OS의 실제 원인이 에러 메시지에 나오도록 한다.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix, img)
    if not ok:
        raise IOError(f"이미지를 인코딩할 수 없습니다 ({path.suffix}): {path}")
    path.write_bytes(buf.tobytes())
