# Visual Material Intelligence — Roadmap

> 규칙: 현재 PHASE의 Definition of Done(DoD)을 모두 만족하고, 직접 실행해서 확인하기 전에는 다음 PHASE로 넘어가지 않는다.
> 새 기술/모델은 "문제 → 기술 → 실험 → 결과"의 연결이 설명될 때만 추가한다.

| PHASE | 이름 | 상태 | 예상 기간 |
|---|---|---|---|
| 0 | Environment | 🔄 진행 중 | 2~3일 |
| 1 | Classical CV Prototype | ⏳ | 2주 |
| 2 | Deep Learning (pretrained inference) | ⏳ | 2주 |
| 3 | Evaluation | ⏳ | 1주 |
| 4 | Fine-tuning | ⏳ | 3주 |
| 5 | Real-time | ⏳ | 1~2주 |
| 6 | Optimization | ⏳ | 1~2주 |
| 7 | Deployment | ⏳ | 1~2주 |
| 8 | Portfolio | ⏳ | 1~2주 |

---

## PHASE 0 — Environment

**Goal**
재현 가능한 개발 환경과 Phase 1용 입력 데이터(20장 이상)를 준비한다.

**Tasks**
1. Python 3.11 venv 생성 (`.venv`) ✅
2. Stage 1 dependency 설치
3. `requirements.txt` 고정 (버전 명시)
4. 폴더 구조 생성
5. `.gitignore` 작성
6. `scripts/check_env.py` — 환경 정보 출력 스크립트
7. 이미지 20~30장 수집 → `data/raw/`
8. `data/DATASET.md` 작성 (출처·라이선스·장수·해상도·편향)
9. `git init` + 첫 commit

**Files**
`requirements.txt`, `.gitignore`, `README.md`(초안), `scripts/check_env.py`, `data/DATASET.md`, `docs/ROADMAP.md`

**Dependencies**
opencv-python, numpy, matplotlib, pyyaml

**Expected Output**
`python scripts/check_env.py` 실행 시 Python/OpenCV/NumPy 버전, CPU, 이미지 개수 출력.

**Evaluation**
- 다른 PC에서 `pip install -r requirements.txt`만으로 같은 환경 재현 가능한가?
- `data/raw` 이미지 수, 해상도 분포가 DATASET.md에 기록되었는가?

**Definition of Done**
- [ ] venv에서 `import cv2` 성공
- [ ] `requirements.txt`에 버전 고정
- [ ] `check_env.py`가 에러 없이 실행
- [ ] `data/raw`에 20장 이상
- [ ] `DATASET.md`에 모든 이미지의 출처/라이선스 기록
- [ ] 첫 commit 완료

---

## PHASE 1 — Classical CV Prototype

**Goal**
딥러닝 없이 OpenCV만으로 이미지 → 6종 결과(Original / Edge / Color / Texture / Silhouette / Material Map)를 자동 생성하는 end-to-end pipeline. 이후 모든 PHASE의 **baseline**이 된다.

**Tasks**
1. 이미지 입력 2. resize 3. HSV/LAB 변환 4. color analysis(히스토그램, k-means 팔레트) 5. edge(Canny) 6. contour 7. texture(local std, Laplacian) 8. silhouette(GrabCut vs Otsu) 9. visualization(Material Map, contact sheet) 10. batch processing + 처리시간 기록

**Files**
`configs/stage1.yaml`, `src/vmi/{io_utils,color,edge,texture,silhouette,material_map}.py`, `scripts/run_stage1.py`, `docs/experiments.md`

**Dependencies**
Phase 0과 동일 (+ texture 단계에서 scikit-image 검토 — 추가 전 정당화)

**Expected Output**
`results/{original,edge,color,texture,silhouette,material_map}/` 에 이미지별 결과, `results/contact_sheet.png`, `results/timing.csv`

**Evaluation**
- 이미지당 평균/최대 처리시간 (ms)
- Silhouette: 직접 그린 GT mask 10장 기준 IoU (GrabCut vs Otsu)
- 정성 평가: 실패 사례 5장 이상 분류

**Definition of Done**
- [ ] 20개 이상의 이미지 처리 가능
- [ ] 모든 결과 자동 저장
- [ ] 실행 명령어가 README에 존재
- [ ] 에러 없이 전체 pipeline 실행
- [ ] 결과 이미지 확인 가능 (contact sheet)
- [ ] 최소 3개의 visual output 비교 가능
- [ ] Silhouette IoU, 처리시간 수치 기록

---

## PHASE 2 — Deep Learning (Pretrained Inference)

**Goal**
Phase 1 silhouette의 한계(복잡한 배경, 사람/배경 색이 비슷한 경우)를 pretrained detection + segmentation으로 해결하는지 검증한다.

**Tasks**
1. 기술 도입 정당화 문서 작성 (후보: YOLO person detection, SAM2 segmentation)
2. PyTorch CPU 설치
3. person detection → bbox
4. bbox를 prompt로 segmentation
5. Phase 1 pipeline의 silhouette 모듈 교체 가능하게 연결
6. CPU 추론 시간 측정

**Files**
`src/vmi/detection.py`, `src/vmi/segmentation_dl.py`, `scripts/run_stage2.py`

**Dependencies**
torch (CPU), 모델 라이브러리 — 도입 전 비교표(정확도/속도/메모리/라이선스)로 결정

**Expected Output**
Phase 1과 같은 결과 폴더 구조 + DL silhouette, 모델별 inference time

**Evaluation**
같은 GT 10장으로 IoU 비교: GrabCut vs DL, CPU latency(ms), 메모리

**Definition of Done**
- [ ] 기술 정당화 문서 존재
- [ ] 20장 전체 실행
- [ ] Phase 1 대비 IoU / latency 비교표

---

## PHASE 3 — Evaluation

**Goal**
"잘 된다"가 아닌 숫자와 실패 유형으로 현재 시스템을 설명한다.

**Tasks**
1. GT mask 확장 (10 → 30장)
2. 평가 스크립트 (IoU, Precision, Recall)
3. Error analysis — 실패 유형 분류 (투명 소재, 검은 옷, 복잡한 배경, 가림 등)
4. 결과표 + 정성 비교 그리드

**Files**
`src/vmi/metrics.py`, `scripts/evaluate.py`, `docs/error_analysis.md`

**Evaluation / DoD**
- [ ] `python scripts/evaluate.py` 한 번으로 모든 방법의 표 생성
- [ ] 실패 유형별 개수와 예시 이미지
- [ ] Phase 4에서 해결할 문제 1~2개 선정

---

## PHASE 4 — Fine-tuning

**Goal**
Phase 3에서 찾은 약점(예: material 구분)을 custom dataset으로 개선한다.

**Tasks**
1. class 정의 (예: transparent / shiny / matte / pattern / fabric) — 최소 개수로 시작
2. 라벨링 도구 선정, annotation (format, split 기록)
3. Colab/Kaggle GPU에서 학습 (seed 고정)
4. 실험: Baseline → Augmentation → Fine-tuning → Hyperparameter

**Files**
`data/DATASET.md` 갱신, `notebooks/train_colab.ipynb`, `configs/train_*.yaml`, `docs/experiments.md`

**Evaluation**
mAP50 / Precision / Recall / IoU, class별 성능, confusion matrix, overfitting 판단(train vs val curve)

**DoD**
- [ ] 실험표 4행 이상 (모두 실측)
- [ ] 재현 가능한 학습 config + seed
- [ ] class imbalance 분석

---

## PHASE 5 — Real-time

**Goal**
웹캠/영상에서 Camera → Detection → Segmentation → Material Analysis → Visualization을 실시간으로 수행한다.

**Tasks**
1. 영상 입력 루프 2. 프레임별 latency 측정(단계별) 3. 프레임 스킵/해상도 조절 4. Visual Mode 1~2개 실시간 렌더링

**Files**
`scripts/run_realtime.py`, `src/vmi/realtime.py`

**Evaluation**
FPS, 단계별 latency(ms), CPU 사용률, 메모리, tracking 안정성(mask 깜빡임)

**DoD**
- [ ] 30초 이상 끊김 없이 실행
- [ ] 단계별 latency 표

---

## PHASE 6 — Optimization

**Goal**
현재 하드웨어(Intel CPU/iGPU)에서 inference 속도를 개선한다.

**Tasks**
1. PyTorch → ONNX export 2. ONNX Runtime(CPU) 3. OpenVINO(Intel) 4. 입력 해상도/양자화 실험 5. (선택) Colab에서 TensorRT 참고 측정

**Evaluation**
| Runtime | Latency | FPS | Model Size | Memory | IoU/mAP 변화 |

**DoD**
- [ ] 3개 이상 runtime 비교표 (실측)
- [ ] 정확도 손실 여부 확인

---

## PHASE 7 — Deployment

**Goal**
분석 결과를 브라우저에서 보는 실시간 데모 형태로 만든다.

**Tasks**
1. FastAPI 서버 2. WebSocket으로 결과 스트리밍 3. 웹 visual (Visual Mode) 4. Docker

**DoD**
- [ ] `docker run` 한 줄로 데모 실행
- [ ] end-to-end latency 측정

---

## PHASE 8 — Portfolio

**Goal**
채용 담당자가 5분 안에 문제·접근·결과·창의성을 이해하는 결과물.

**Tasks**
1. README 전체 구조 작성 (Problem ~ Usage)
2. 20~60초 demo video
3. Baseline → Improvement → Final 표
4. Creative Direction (Visual Mode 01~05)
5. 이력서 문장, 면접 Q&A

**DoD**
- [ ] README의 모든 숫자가 실측값
- [ ] Demo video 링크
- [ ] 포트폴리오 리뷰 모드 평가 7점 이상
