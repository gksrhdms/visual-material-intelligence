# Visual Material Intelligence

사진 속 인물과 의상을 Computer Vision으로 분석하고, 영역·색·질감·구조 정보를 하나의 시각 언어인 **Material Map**으로 변환하는 시스템.

> **Status — PHASE 1 (Classical CV) 완료.** 딥러닝 없이 OpenCV만으로 만든 end-to-end baseline.
> 다음 단계(PHASE 2)는 이 baseline의 측정된 한계를 pretrained 모델로 개선하는 것. 로드맵: [docs/ROADMAP.md](docs/ROADMAP.md)

<!-- DEMO: 현재 결과 이미지는 라이선스 미확인(Pinterest) 사진에서 생성되어 공개하지 않는다.
     공개용 사진(직접 촬영 / Unsplash·Pexels)으로 재생성한 뒤 contact sheet를 이 위치에 넣는다. -->

## Problem

패션 이미지에서 "옷이 어떤 색이고, 어떤 질감이며, 어떤 형태인가"를 자동으로 측정하려면, 먼저 **측정 대상(사람/의상)과 배경을 분리**해야 한다.
이 프로젝트의 실험에서 이 전제가 가장 중요하다는 것이 반복해서 확인됐다:

- 사진 전체로 색을 재면 020(금색 드레스)은 "74% 무채색"이었지만, 사람 영역만 재면 **8%** — 베이지 배경이 옷의 색을 가리고 있었다 (EXP-006).
- silhouette IoU가 0.8 미만인 이미지는 color 통계 오차가 **약 6배** 크다 (2.1%p → 12.6%p).

## Approach

```
Image ─► Resize ─► Silhouette ─┬─► Color   (LAB chroma, k-means palette)  ─┐
         (≤768px)   (GrabCut)   ├─► Texture (local std, |Laplacian|)       ├─► Material Map
                                └─► Edge    (Canny, contour)              ─┘    (poster)
```

| 단계 | 방법 | 왜 이 방법인가 (근거 실험) |
|---|---|---|
| Resize | 긴 변 768px, `INTER_AREA`, 축소만 | 픽셀 단위 파라미터(kernel, threshold)를 이미지 간 일관되게. 확대는 가짜 질감을 만든다 (EXP-001) |
| Silhouette | GrabCut, 계산 해상도 ×0.33 | Otsu 대비 IoU 2배. 원본 해상도보다 빠르고 오히려 정확 (EXP-005) |
| Color | LAB chroma ≥ 15, mask 안에서만 | HSV 채도는 어둡거나 따뜻한 회색을 유채색으로 오판 (EXP-002, 003) |
| Edge | Canny fixed (100, 200) | 밝기 median 기반 auto threshold는 밝은/어두운 배경에서 양극단으로 실패 (EXP-004) |
| Texture | 7×7 local std, \|Laplacian\|, mask를 erode해 경계 제외 | edge(blur 후)는 가는 줄무늬를 놓치지만 Laplacian은 잡는다 (EXP-007) |

## Dataset

- 테스트 세트 25장, 6개 조건: plain 5 / complexbg 5 / pattern 5 / glossy 4 / sheer 3 / mono 3 — **실패할 사진을 의도적으로 포함**
- Silhouette GT mask 25장 (AI 배경 제거 도구로 생성 후 전수 검수, 1장 수정)
- 상세: [data/DATASET.md](data/DATASET.md). 원본은 라이선스 미확인으로 저장소에 포함하지 않는다.

## Baseline & Results

**Silhouette (GT 25장 기준)**

| method | mean IoU | median IoU | precision | recall | IoU≥0.8 | median ms |
|---|---|---|---|---|---|---|
| Otsu | 0.379 | 0.324 | 0.527 | 0.521 | 5/25 | 5 |
| GrabCut ×1.0 | 0.700 | 0.757 | 0.741 | 0.940 | 9/25 | 4,467 |
| GrabCut ×0.5 | 0.749 | 0.815 | 0.814 | 0.916 | 13/25 | 797 |
| **GrabCut ×0.33 (채택)** | **0.759** | **0.850** | 0.843 | 0.896 | **15/25** | **270** |

×0.5와 ×0.33의 IoU 차이는 2~3장의 결과 뒤집힘에서 나온 것으로 의미 있는 차이가 아니다. ×0.33의 채택 근거는 **품질이 같고 3배 빠르다는 것**이다.

**조건별 silhouette IoU (GrabCut ×0.33)**

| plain | complexbg | pattern | glossy | sheer | mono |
|---|---|---|---|---|---|
| 0.923 | **0.435** | 0.700 | 0.882 | 0.746 | 0.969 |

**처리 시간 (이미지 1장, CPU i7-1260P, median)**

| load | resize | silhouette (GrabCut) | color | texture | edge | Material Map | **total** |
|---|---|---|---|---|---|---|---|
| 13.2 | 2.5 | **283.4** | 27.2 | 9.0 | 5.6 | 93.1 | **458** ms |

GrabCut이 전체의 62%를 차지한다. (total에는 비교용 Otsu 4.6 ms 포함)

## Error Analysis

| 실패 유형 | 사례 | 원인 | 해결 방향 |
|---|---|---|---|
| 배경을 사람으로 포함 | 003, 015, 007 | GrabCut 초기 사각형이 화면의 90%를 덮어 전경 색 모델에 배경색이 섞임 | 사람 위치를 아는 tight bbox (person detector) |
| 사각형 가정 위반 | 009 (누운 자세), 008 (2명) | "사람은 중앙에 1명" 가정 | person detector |
| 색만으로 분리 불가 | 007 (흰 드레스 + 흰 건물) | GrabCut은 색 분포로만 판단 | 학습된 segmentation |
| 피부와 옷이 섞임 | 019 (핑크 새틴), 020 (금색 시스루) | 옷의 hue가 피부와 같음, 시스루는 픽셀 자체가 혼합색 | human parsing (피부/머리/옷 분할) |
| texture 값의 교란 | 019, 011 (저해상도) | 같은 옷도 해상도만 바꾸면 local std가 최대 44% 변함 | 사람 키 기준 스케일 정규화 |

더 자세한 분석과 모든 실험 기록: [docs/experiments.md](docs/experiments.md) (EXP-001 ~ 007)

## Creative Direction — Material Map

CV 출력을 장식 없이 그대로 시각 언어로 옮긴다. 데이터 시각화 × 패션 그래픽.

| CV 출력 | 시각적 의미 | 디자인 언어 |
|---|---|---|
| Silhouette | 형태 | 점이 찍히는 영역 + 가는 외곽선 |
| Texture (local std) | 질감의 양 | 점의 크기 — 하프톤 망점 |
| 8px 칸의 평균색 | 소재의 색 | 점의 색 (강조색 없음, 옷에서 나온 색만) |
| Edge | 구조, 재단선 | 잉크색 가는 선 |
| 측정값 | 수치 | 고정 스케일 막대 — 포스터끼리 비교 가능 |
| Silhouette 오차 (GT 대비) | 실패 | 회색 빈 원 = 배경을 사람으로 포함, 작은 점 = 놓친 사람 |

실패를 숨기지 않는다: GT가 있는 이미지는 시스템이 틀린 칸을 그래픽으로 드러내고, GT silhouette로 만든 "IDEAL" 포스터를 별도 폴더에 참고용으로 둔다 (PHASE 2의 목표치).

## Technical Stack

Python 3.11 · OpenCV 5.0 · NumPy · Pillow (타이포그래피) · PyYAML (실험 설정) — CPU only

## Future Work (PHASE 2)

- **문제**: complexbg silhouette IoU 0.435, 실패 원인은 "사람 위치를 모름"과 "색만으로 분리 불가"
- **후보**: person detector로 GrabCut 초기화 / 학습된 person segmentation / human parsing (피부·옷 분리)
- **검증**: 같은 GT 25장, 같은 `evaluate_silhouette.py`로 baseline 0.759 대비 비교. 도입 전 모델별 정확도·속도·메모리·라이선스 비교

## Installation

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```powershell
python scripts/check_env.py                 # 환경 및 입력 이미지 확인
python scripts/run_stage1.py                # 전체 pipeline → results/
python scripts/make_contact_sheet.py        # results/sheets/ (포스터 그리드, 단계별 비교)
python scripts/evaluate_silhouette.py       # GT 대비 silhouette 평가 (data/masks 필요)
```

실험 재현 스크립트: `exp003_chroma_threshold.py`, `exp006_masked_color.py`, `exp007_texture_groups.py`
설정: `configs/stage1.yaml` — 모든 파라미터와 그 근거 실험 번호가 주석으로 기록되어 있다.

| 출력 폴더 | 내용 |
|---|---|
| `results/original` | resize된 입력 |
| `results/silhouette` | Otsu / GrabCut 비교 패널, `mask/` |
| `results/color` | 사람 영역 대표색 팔레트, Hue 히스토그램 |
| `results/texture` | local std / \|Laplacian\| 맵 |
| `results/edge` | edge map, contour |
| `results/material_map` | Material Map 포스터 (시스템 출력) |
| `results/material_map_gt` | GT silhouette 참고 포스터 (시스템 출력 아님) |
| `results/timing.csv` | 이미지별 단계 시간과 모든 측정값 |

## Environment

| | |
|---|---|
| OS | Windows 11 |
| CPU | Intel Core i7-1260P |
| GPU | Intel Iris Xe (NVIDIA GPU 없음 → CPU 기준) |
| RAM | 16 GB |
| Python | 3.11.4 |
