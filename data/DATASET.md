# Dataset Card — Stage 1 (Classical CV)

## Purpose
Phase 1 Classical CV pipeline 검증용 소규모 이미지 세트. 학습용이 아니라 **pipeline이 어떤 조건에서 성공/실패하는지 보기 위한 테스트 세트**다.

## Summary
| 항목 | 값 |
|---|---|
| Number of images | 25 |
| Sources | Pinterest 25장 (원작자/라이선스 미확인 — 내부 실험 전용) |
| Resolution | 최소 285×427 (019) ~ 최대 1200×1912 (020), 전부 세로 비율 |
| Conditions | plain 5 / complexbg 5 / pattern 5 / glossy(leather·metallic·satin) 4 / sheer·mesh 3 / black·white·mono 3 |
| Annotation | Silhouette GT mask 25장 (`data/masks/`, 이진 mask, `results/original` 해상도) — 아래 Annotation 절 참고 |
| Split | 없음 (평가 전용) |

## 수집 기준 — 일부러 다양하게
결과가 "잘 되는 사진"만 모으면 error analysis를 할 수 없다. 아래 조건을 골고루 섞는다.

| 조건 | 목표 장수 | 이유 |
|---|---|---|
| 단색 배경 + 전신 | 5 | 쉬운 케이스 (baseline 성공 기준) |
| 복잡한 배경 (거리, 실내) | 5 | silhouette 실패 유도 |
| 패턴 의상 (체크, 스트라이프, 프린트) | 4 | texture 분석 |
| 광택 소재 (가죽, 새틴, 메탈릭) | 3 | specular highlight |
| 투명/시스루 소재 | 3 | 가장 어려운 케이스 |
| 검은 옷 / 흰 옷 | 3 | 색 정보 부족, 노출 문제 |

## Image List
파일명은 영문으로 짓는다 (OpenCV `imread`는 Windows 한글 경로에서 실패할 수 있음).
예: `001_plainbg_full.jpg`

| File | Source | Author / URL | License | Condition |
|---|---|---|---|---|
| 001_plain.jpg | pinterest | | | 단색 배경 |
| 002_plain.jpg | pinterest | | | 단색 배경 |
| 003_plain.jpg | pinterest | | | 단색 배경 |
| 004_plain.jpg | pinterest | | | 단색 배경 |
| 005_plain.jpg | pinterest | | | 단색 배경 |
| 006_complexbg.jpg | pinterest | | | 복잡한 배경 |
| 007_complexbg.jpg | pinterest | | | 복잡한 배경 |
| 008_complexbg.jpg | pinterest | | | 복잡한 배경 |
| 009_complexbg.jpg | pinterest | | | 복잡한 배경 |
| 010_complexbg.jpg | pinterest | | | 복잡한 배경 |
| 011_pattern.jpg | pinterest | | | 패턴 (저해상도 424px) |
| 012_pattern.jpg | pinterest | | | 패턴 |
| 013_pattern.jpg | pinterest | | | 패턴 |
| 014_pattern.jpg | pinterest | | | 패턴 |
| 015_pattern.jpg | pinterest | | | 패턴 |
| 016_leather.jpg | pinterest | | | 광택 — 가죽 |
| 017_metalic.jpg | pinterest | | | 광택 — 메탈릭 |
| 018_leather.jpg | pinterest | | | 광택 — 가죽 |
| 019_satin.jpg | pinterest | | | 광택 — 새틴 (저해상도 285px ⚠) |
| 020_sheer.jpg | pinterest | | | 투명 — 시스루 |
| 021_mesh.jpg | pinterest | | | 투명 — 메쉬 |
| 022_sheer.jpg | pinterest | | | 투명 — 시스루 |
| 023_allblack.jpg | pinterest | | | 올블랙 |
| 024_monochrome.jpg | pinterest | | | 모노톤 |
| 025_allwhite.jpg | pinterest | | | 올화이트 |

## Annotation — Silhouette GT
- 도구: Photoroom (AI 배경 제거 기반 웹 도구)으로 생성
- 검수: 원본 위 overlay로 25장 전수 확인. 009에서 소파 전체가 전경으로 잡힌 오류를 발견해 수정 (면적 55% → 14%)
- 형식: JPG로 저장되어 경계에 회색 픽셀이 0.01~0.26% 존재 → 평가 시 127 기준 이진화. 이후 mask는 무손실 PNG로 저장한다.
- 라벨 규칙:
  | 대상 | 규칙 |
  |---|---|
  | 머리카락, 피부, 옷, 신발 | 포함 |
  | 들고 있는 가방·액세서리 | 포함 |
  | 반투명 천 (021 튤 등) | 포함 (옷의 일부) |
  | 사진 속 모든 인물 (008: 2명) | 포함 |
  | 바닥 그림자, 신체 사이의 배경 | 제외 |
- 한계: GT 자체가 AI 모델의 출력에서 출발했으므로, 그 모델이 체계적으로 틀리는 영역(반투명 천 경계, 잔머리)은 사람이 확인한 만큼만 정확하다.

## Potential Bias
(수집 후 작성: 성별/체형/피부톤/포즈/조명의 치우침)

## License Notes
- **현재 25장은 모두 Pinterest 출처로 원작자 저작권이 확인되지 않았다.** pipeline 개발·내부 평가에만 사용하며, README·demo 영상 등 공개 결과물에는 사용하지 않는다. 공개용 결과는 직접 촬영 또는 Unsplash/Pexels 이미지로 교체해 생성한다.
- Unsplash / Pexels: 무료 사용 가능, 출처 표기는 권장 사항이지만 포트폴리오에서는 항상 기록한다.
- 원본 이미지는 git에 올리지 않는다 (`.gitignore`의 `data/raw/`).
- 인물이 식별되는 직접 촬영 사진은 공개 전 촬영 대상의 동의를 받는다.
