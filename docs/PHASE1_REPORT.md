# PHASE 1 Report — Classical CV Prototype

기간: 2026-10-07 ~ 2026-10-21 · 실험 기록: [experiments.md](experiments.md) (EXP-001 ~ 007)

## 1. 구현한 것

- OpenCV만으로 동작하는 end-to-end pipeline: resize → silhouette → color → texture → edge → Material Map
- 25장 일괄 처리, 이미지별 단계 시간과 모든 측정값을 `results/timing.csv`에 기록
- Silhouette GT 25장과 정량 평가 스크립트 (IoU / precision / recall)
- Material Map 포스터: CV 출력을 점의 크기·색·선으로 옮긴 시각 언어, GT 대비 실패 영역 표시
- 실험 재현 스크립트 4종, contact sheet 2종, 설정 파일 하나로 모든 파라미터 관리 (seed 고정)

## 2. 배운 CV 개념

| 개념 | 어디서 체득했나 |
|---|---|
| 보간법의 선택 (`INTER_AREA` 축소 vs `INTER_NEAREST` 라벨) | EXP-001, GrabCut mask 복원, Material Map 칸 평균 |
| 색공간: HSV 채도의 한계, LAB chroma | EXP-002, 003 |
| threshold 선택: anchor case, valley, 구간 폭 보정, precision–recall trade-off | EXP-003, 004 |
| Canny 4단계, OpenCV Canny에 blur가 없다는 점 | EXP-004 |
| edge(경계) vs region(영역) 기반 분할 | EXP-004 → 005 |
| Otsu, GrabCut, morphology(closing/opening), connected components | EXP-005 |
| IoU / precision / recall, mean vs median, 작은 표본의 차이를 과신하지 않기 | EXP-005 |
| 오차 전파: 앞 단계의 IoU가 뒤 단계의 오차를 완전히 설명하지 못함 | EXP-006 |
| local std(박스 필터 분산), Laplacian(2차 미분, 노이즈 증폭), 해상도 교란 변수 | EXP-007 |
| warm-up, median latency | EXP-001 이후 모든 시간 측정 |

## 3. 성능

| 항목 | 값 |
|---|---|
| Silhouette mean IoU (GrabCut ×0.33, GT 25장) | 0.759 (median 0.850, IoU≥0.8: 15/25) |
| Otsu baseline | 0.379 |
| 가장 약한 조건 | complexbg 0.435 |
| 이미지당 처리 시간 (median) | 458 ms — GrabCut 283 ms (62%) |
| Color 오차 전파 | silhouette IoU < 0.8에서 achromatic 오차 12.6%p (≥ 0.8: 2.1%p) |

## 4. 문제점

1. 복잡한 배경에서 silhouette 실패 (007: 0.24, 009: 0.30, 006: 0.39, 015: 0.40)
2. 피부와 옷을 색으로 분리할 수 없음 (019, 020)
3. texture 값이 해상도·옷 조각 경계·노이즈에 교란됨 → 소재 분류 기준으로는 부족
4. chroma 무채색 판정의 threshold(T=15)는 데이터상 자연 경계가 없어 임의적
5. GrabCut이 처리 시간의 62%이고, 입력이 조금 바뀌어도 결과가 뒤집히는 불안정한 방법
6. 데이터 라이선스: 25장 모두 Pinterest — 공개 결과물에 사용 불가

## 5. 다음 단계 — PHASE 2

문제 → 기술 → 실험의 연결:
- 문제 1(사람 위치 모름, 색만으로 분리 불가) → person detector / 학습된 person segmentation
- 문제 2(피부·옷 혼합) → human parsing
- 검증: 같은 GT 25장, `evaluate_silhouette.py`, baseline 0.759 / complexbg 0.435와 비교
- 도입 전: 후보 모델별 정확도·CPU 속도·메모리·라이선스 비교표와 도입 정당화 문서

## 6. Git

Phase 1 마무리 commit: `docs: complete phase 1 with README, contact sheets and report`

## 7. 포트폴리오에 기록할 내용

- **문제 정의를 데이터로 했다**: "배경이 측정을 오염시킨다"는 것을 EXP-002~004에서 반복 관찰 → silhouette을 pipeline 맨 앞으로 이동 → EXP-006에서 정량 확인 (020: 74% → 8%)
- **직관이 틀린 것을 측정으로 발견**: GrabCut은 원본 해상도가 가장 느리고(4.5초) 가장 부정확(IoU 0.700). ×0.33이 16배 빠르고 IoU 0.759
- **널리 쓰이는 경험칙을 검증 후 기각**: median 기반 auto Canny threshold가 밝은/어두운 배경에서 실패하는 원인(밝기 ≠ gradient)을 설명
- **실패를 숨기지 않는 시각화**: Material Map에 GT 대비 오차를 그래픽으로 표시, IDEAL 포스터로 PHASE 2 목표를 시각화
- **이력서 문장 초안** (실측 수치만):
  "OpenCV 기반 패션 이미지 분석 pipeline을 구축하고, GT 25장으로 silhouette 방법을 정량 비교해 Otsu(IoU 0.38) 대비 GrabCut(IoU 0.76)을 채택했으며, 계산 해상도 실험으로 정확도 손실 없이 처리 시간을 4.5초에서 0.27초로 단축."
