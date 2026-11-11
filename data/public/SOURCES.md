# Public Image Set — 포트폴리오 · README 공개용

실험(25장, Pinterest)은 라이선스가 확인되지 않아 공개 결과물에 쓰지 않는다.
이 폴더의 이미지는 공개해도 되는 사진으로, 포트폴리오와 README의 시각 결과물을 만드는 데만 쓴다.
수치(IoU, 처리 시간 등)는 실험 세트 기준이며, 이 세트로 다시 측정한 값이 아니다.

- 원본 사진은 저장소에 올리지 않는다 (`.gitignore`). 생성한 결과 이미지만 공개한다.
- 파일명 규칙: `p01_plain.jpg` … (접미사는 실험 세트와 같은 조건 이름)

## Images

| File | Source | Condition | GT mask |
|---|---|---|---|
| p01_plain.jpg | 본인 사진 — 빈티지숍 모델 활동 중 촬영 | plain | — |
| p02_plain.jpg | 본인 사진 — 빈티지숍 모델 활동 중 촬영 | plain | — |
| p03_plain.jpg | [Pexels 34794171](https://www.pexels.com/ko-kr/photo/34794171/) | plain | — |
| p04_complexbg.jpg | [Pexels 40297470](https://www.pexels.com/ko-kr/photo/40297470/) | complexbg | ✓ |
| p05_complexbg.jpg | [Pexels 40297468](https://www.pexels.com/ko-kr/photo/40297468/) | complexbg | ✓ |
| p06_pattern.jpg | [Pexels 40654032](https://www.pexels.com/ko-kr/photo/40654032/) | pattern | — |

## License Notes

- Pexels 사진: [Pexels License](https://www.pexels.com/license/) — 무료 사용·수정 가능. 출처 표기는 선택이지만 항상 기록한다. 원본을 그대로 재배포하지 않는다.
- p01, p02: 사진 속 인물은 본인. 사진 저작권은 촬영자에게 있으므로 공개 전 촬영자의 사용 동의를 받는다.

## GT Masks (`data/public_masks/`, 저장소 미포함)

- p04, p05: Photoroom으로 생성 → 원본 위 overlay로 확인 (머리카락·손가락·체인 벨트 포함 정확)
- Photoroom 출력(RGBA, 2000px 폭, `*_mask.png`)을 pipeline 해상도(긴 변 768px)의 이진 PNG로 변환해 사용 (`INTER_NEAREST`)
- 라벨 규칙은 실험 세트와 동일 ([DATASET.md](../DATASET.md) Annotation 절)
