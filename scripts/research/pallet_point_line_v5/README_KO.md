# 팔레트 점·선 공동 자세 코어 v5

## 이 패키지가 만드는 것

`영상 특징 + 주어진 가로·세로·높이 + 보정된 카메라`
→ `점 관측 + Deep Hough 선 관측 + 영상 기반 복수 초기 자세`
→ `미분 가능한 공동 기하 계산`
→ `하나의 R,t와 그 박스에서 나온 코너·모서리`

기존 v4의 64개 고정 키포인트 배치 scorer를 다시 이름만 바꾼 것이 아니다. 주어진 치수의 강체를 점과 선으로 함께 설명하는 per-instance pose 코어다. 프로그램이 실제 작동하는지와 팔레트에서 정확한지는 별개다.

**현재 상태: 코드 작성 및 CPU 검증. 실제 팔레트 데이터·GPU 실험은 하지 않았다.**
검증 결과는 `evidence/LOCAL_VALIDATION.json`을 읽는다. 생성한 검사 데이터의 정확도/손실을 논문용 팔레트 성능으로 쓰지 않는다.

## 구현과 남은 연결

|구성|상태|
|---|---|
|C1/C2/C4와 3D 코너/edge 순열, 중심 보존, 직사각 C4 거부|구현/테스트|
|명시적 raw→feature affine, predicted ROI, 두 해상도 결합|구현/테스트|
|치수 FiLM, 점 분기, 직접 선 기준, 실제 DHT 선 분기|구현/테스트|
|다중 line mode, 양 끝점의 동일 mode, robust 공동 pose solver|구현/테스트|
|영상 기반 복수 초기 pose, 국소 refinement, GT 없는 선택|구현/테스트|
|점·선·자세·초기값 손실의 한 물체/한 대칭 선택|구현/테스트|
|prepared feature cache 실제 학습/저장/추론/오프라인 평가|구현/통합 테스트|
|RGB toy backbone→ROI→head→solver 실제 역전파/업데이트|생성 fixture로 검증|
|원래 YOLO checkpoint·detector·데이터·canonical evaluator 연결|로컬 CLI 작업 필요|
|실제 소스 이미지와 feature의 물리 정렬|여기서는 미확인; 로컬 독립 검사 필요|
|실사 성능, 6D benchmark, GPU latency, Jetson export|미실행|

## 반드시 구별할 한계

- 이 새 point head는 원래 YOLO keypoint head와 초기 출력이 같지 않다. R0 재사용/초기 parity를 주장하지 않는다. backbone/검출기 재사용과 point head 재사용은 다르다.
- `point`, `direct`, `hough`는 같은 치수/카메라/초기 pose/solver 틀을 사용하지만 실제 활성 파라미터 수가 다르다. 정확한 수를 기록한다. H 대 D도 집계+readout 구조 비교이지 Hough 연산 하나만의 인과 ablation은 아니다.
- `runner train`은 고정 특징 학습이다. online backbone 학습을 했다고 부르지 않는다. `adapters.py`의 `detach=False` 경로와 toy RGB demo는 online 연결 가능성을 보여주지만 실제 YOLO 연결 완료는 아니다.
- `PointLineRefiner`는 robust IRLS/EM + 감쇠 Gauss–Newton + 고정 예산 line search다. EPro-PnP 재현, 글로벌 최적해, 확률적으로 보정된 uncertainty가 아니다.
- Hough의 상위 mode 선택, window/NMS, solver step 수락, 최종 pose 선택은 불연속이다. 선택된 경로의 미분과 학습은 가능하지만 모든 선택을 매끄럽게 미분한다고 하지 않는다.
- 선 표준편차와 결합 계수는 현재 고정 엔지니어링 설정이다. 모델이 confidence를 회피적으로 0으로 만들지는 않지만, 적절한 uncertainty나 상관 처리의 증명도 아니다.
- 선은 amodal cuboid support다. v>0를 물리적 가시성으로 해석하지 않는다. 현재 line loss는 ROI와 교차하는 지원 GT 선만 감독하며, 그 구간의 물리 경계 존재를 인증하지 않는다.
- 대칭은 외부에서 확인된 계약을 요구한다. 정사각 치수만으로 C4를 부여하지 않는다. 직사각형 W/D를 바꾸는 좌표계 재표현도 이 버전에서 자동 허용하지 않는다.
- prepared manifest가 검출된 instance만 담고 있으면 전체 이미지 검출 성능을 설명할 수 없다. 누락 검출·negative는 원 detector의 전체 모집단 ledger와 별도로 연결해야 한다.
- 치수·pose frame이 잘못되면 강체 출력은 잘못된 형상을 강요한다. raw GT를 투영점으로 덮어써 맞추지 않는다.

## 바로 실행

기존 PyTorch 환경에서 압축을 새 폴더에 풀고:

```bash
cd /실제/압축해제/pallet_point_line_v5
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python verify_bundle.py
python -m pytest tests -q
python -m plpose_v5.demo --output /새로운/경로/v5_demo --steps 8 --device cpu
python -m plpose_v5.diagnostics --output /새로운/경로/geometry_probe.json
python -m plpose_v5.runner --help
python -m plpose_v5.assessment --help
```

`demo`만 OpenCV가 필요하다. 필요한 라이브러리가 이미 있으면 재설치하지 않는다. CUDA용 torch를 pip 명령으로 CPU용으로 교체하지 않는다.

**실제 저장소에서 진행할 CLI 지시문은 `CLI_INSTRUCTIONS_KO.md`다.**
`docs/METHOD_AND_CONTRACT_KO.md`, `docs/PRIOR_WORK_KO.md`를 함께 읽어야 한다.

## 파일 지도

- `plpose_v5/geometry.py`: 물체 좌표, 투영, Jacobian, line 변환, 실제 샘플링
- `contracts.py`: 관측/GT 분리, 물체별 대칭 계약
- `hough.py`: content-normalized DHT, direct line 기준, 복수 mode decode
- `solver.py`: 점·선 공동 자세, 다중 시작점, validity/조건수 진단
- `model.py`: 치수 조건부 단일 per-instance 그래프
- `objective.py`: 모든 손실 항이 공유하는 한 대칭 선택
- `adapters.py`: 검증된 기존 모듈 hook, 두 해상도 ROI, 명시적 좌표 변환
- `io.py`, `runner.py`: SHA·초기값·계획·실제 업데이트·checkpoint·GT-free 추론
- `assessment.py`: 동일 대칭 평가, 결측 실패 분모, session paired interval
- `diagnostics.py`: 동일 초기화의 점/예측선/정답선 비교 함수
- `fixtures.py`, `demo.py`, `tests/`: 소프트웨어 검증용 생성 데이터

작성 코드는 논문 공식 코드를 복사한 것이 아니다. 출처와 차이는 `docs/PRIOR_WORK_KO.md`에 적었다. novelty·성능은 검증되지 않았다.
