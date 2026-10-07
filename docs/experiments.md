# Experiment Log

형식: 무엇을 바꿨는가 → 왜 → 결과(숫자) → 결론

---

## EXP-001 — Resize 정책과 load/resize 시간 (PHASE 1, Task 1~2)

**설정**
- `resize.max_side = 768`, `cv2.INTER_AREA`, 축소만 하고 확대하지 않음
- 25장, Intel i7-1260P, CPU, 단일 실행 (warm-up 없음)

**왜**
blur kernel, Canny threshold, texture window는 픽셀 단위이므로 해상도를 맞춰야 이미지 간 결과를 비교할 수 있다.
저해상도 이미지(011, 019)를 확대하면 보간으로 생긴 가짜 질감이 texture 분석을 왜곡하므로 확대하지 않는다.

**결과**
| 단계 | mean | max |
|---|---|---|
| load (디스크 읽기 + JPEG 디코딩) | 10.0 ms | 19.9 ms |
| resize (INTER_AREA) | 1.9 ms | 10.0 ms |

- 원본 픽셀 수가 많을수록 load가 길다 (019 285×427: 1.7 ms / 008 1200×1799: 19.9 ms).
- resize가 0.0 ms인 이미지(011, 019, 022)는 긴 변이 768 이하라 resize가 생략된 경우.
- resize max 10.0 ms는 첫 번째 이미지(001)에서만 발생 → OpenCV 첫 호출 초기화 비용(warm-up)으로 추정. 나머지 24장은 1.3~2.6 ms.

**결론**
- 전처리 비용은 디코딩이 지배적이고 resize는 무시할 수준.
- 단일 실행 + warm-up 없는 측정은 첫 호출 이상값이 섞인다. 이후 latency 측정은 warm-up 후 반복 측정하고 median을 보고한다.
