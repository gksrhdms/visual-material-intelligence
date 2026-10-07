# Visual Material Intelligence

사진 속 인물과 의상을 Computer Vision으로 분석하고, 영역·형태·질감·색 정보를 새로운 **Visual Material Map**으로 변환하는 시스템.

> 🚧 Work in progress — 현재 **PHASE 0 (Environment)**. 진행 상황은 [docs/ROADMAP.md](docs/ROADMAP.md) 참고.

## Problem
(PHASE 1에서 작성)

## Approach
1. **Classical CV baseline** — OpenCV만으로 color / edge / texture / silhouette 분석
2. **Deep Learning** — pretrained detection·segmentation으로 baseline의 한계를 개선
3. **Real-time & Deployment** — 웹캠 실시간 분석, CPU 최적화, 웹 데모

## Dataset
Stage 1 테스트 세트 25장 — 수집 기준·출처·라이선스는 [data/DATASET.md](data/DATASET.md).
원본 이미지는 저장소에 포함하지 않는다.

## Installation
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage
```powershell
# 환경 및 입력 데이터 확인
python scripts/check_env.py
```

## Environment
| | |
|---|---|
| OS | Windows 11 |
| CPU | Intel Core i7-1260P |
| GPU | Intel Iris Xe (NVIDIA GPU 없음 → CPU 추론 기준) |
| RAM | 16 GB |
| Python | 3.11.4 |
