"""PHASE 0: 개발 환경과 입력 데이터를 확인한다.

실행: python scripts/check_env.py
"""
import platform
import sys
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def check_packages():
    print("== Packages")
    for name in ["cv2", "numpy", "matplotlib", "yaml"]:
        try:
            module = __import__(name)
            print(f"  {name:<11} {module.__version__}")
        except ImportError:
            print(f"  {name:<11} NOT INSTALLED")


def check_images():
    print(f"== Images in {RAW_DIR}")
    if not RAW_DIR.exists():
        print("  data/raw 폴더가 없습니다.")
        return

    paths = sorted(p for p in RAW_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    print(f"  count: {len(paths)}  (목표: 20장 이상)")

    import cv2  # 패키지 확인 이후에 import (미설치여도 위 출력은 보이도록)

    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            # Windows 한글 경로/파일명은 cv2.imread가 실패할 수 있다
            print(f"  [READ FAIL] {p.name}")
            continue
        h, w = img.shape[:2]
        print(f"  {p.name:<30} {w}x{h}")


if __name__ == "__main__":
    print("== System")
    print(f"  Python   {sys.version.split()[0]}  ({sys.executable})")
    print(f"  OS       {platform.platform()}")
    print(f"  CPU      {platform.processor()}")
    check_packages()
    check_images()
