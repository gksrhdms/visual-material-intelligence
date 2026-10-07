# Dataset Card — Stage 1 (Classical CV)

## Purpose
Phase 1 Classical CV pipeline 검증용 소규모 이미지 세트. 학습용이 아니라 **pipeline이 어떤 조건에서 성공/실패하는지 보기 위한 테스트 세트**다.

## Summary
| 항목 | 값 |
|---|---|
| Number of images | 25 |
| Sources | (a) 직접 촬영, (b) Unsplash / Pexels |
| Resolution | 최소 285×427 (019) ~ 최대 1200×1912 (020), 전부 세로 비율 |
| Conditions | plain 5 / complexbg 5 / pattern 5 / glossy(leather·metallic·satin) 4 / sheer·mesh 3 / black·white·mono 3 |
| Annotation | 없음 (Phase 1 후반에 silhouette GT mask 10장 추가) |
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

## Potential Bias
(수집 후 작성: 성별/체형/피부톤/포즈/조명의 치우침)

## License Notes
- Unsplash / Pexels: 무료 사용 가능, 출처 표기는 권장 사항이지만 포트폴리오에서는 항상 기록한다.
- 원본 이미지는 git에 올리지 않는다 (`.gitignore`의 `data/raw/`).
- 인물이 식별되는 직접 촬영 사진은 공개 전 촬영 대상의 동의를 받는다.
