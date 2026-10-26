# V270 공개 패키지 CPU 재현 안내

이 패키지는 저장된 출력·라벨의 지표 계산과 실제 KP head의 작은 CPU 재생을 지원한다. 원본 RGB, SYN 전체 neck feature 약 3GB, GPU, 새 학습은 필요하지 않다. 전체 SYN 추론이나 처음부터 재학습하는 명령은 아니다.

검증한 로컬 환경은 Python 3.10.20, PyTorch 2.1.1+cu118, NumPy 1.26.4, Ultralytics 8.4.60이다. 아래 명령은 CPU만 사용한다. 실제 head 재생은 포함된 원래 R0 체크포인트의 Ultralytics 클래스 정의가 필요하다. 숫자만 재계산하는 실사 지표 재생에는 모델 로드가 없다.

## 경로 설정

현재 작업 폴더를 `/tmp`로 바꿔도 실행된다. `TASK_DIR`는 공개 checkout의 **artifacts 안 V270 폴더**를 가리켜야 한다. 다른 컴퓨터에서는 두 절대 경로를 바꾼다. 세 출력 경로는 새 폴더여야 한다. 이미 존재하면 접미사를 바꾼다.

```bash
cd /tmp
MLPY=/home/minjae/anaconda3/envs/pallet-yolo26/bin/python
TASK_DIR=/home/minjae/Documents/pallet-visibility-publication-20261024/_docs/experiments/pallet-keypoint-visibility/V270_INITIAL_KP_JOINT_VISIBILITY_QUICKCHECK/artifacts/V270_INITIAL_KP_JOINT_VISIBILITY_QUICKCHECK
```

## 1. 실제 head 및 저장된 실사 판정 재생

```bash
"$MLPY" -B "$TASK_DIR/replay_initial_joint_v270_v2.py" --out /tmp/v270_head_check_001
```

사용 파일은 `weights/BASE_R0.pt`, 세 FINAL300 head 체크포인트, `fullframe_replay_v1/FORWARD_INPUTS5.npz`, 동결된 현재 예측·설정이다. 각 arm에서 이미지 5장의 **전체 neck pyramid**를 실제 head에 통과시켜 raw27/E9와 detector anchor를 확인한다. 원본 RGB와 backbone forward는 없다. 두 학습된 E arm에서는 저장된 1,784개 native8 logit의 sigmoid 및 고정 임계값 판정도 각각 확인한다. XY_ONLY의 E 출력은 학습되지 않았으므로 분류기로 평가하지 않는다.

결과는 `/tmp/v270_head_check_001/REPLAY_RESULT.json`이다. 실제 head의 절대 오차 허용값은 원래대로 `1e-3`이다. 저장된 CUDA float32 sigmoid와 NumPy float64 계산의 차이는 `1.2e-7` 이내여야 하며, 1,784개 판정은 각 arm에서 **완전히 같아야 한다**. 이 두 검증은 범위가 다르다: 실제 head의 직접 재생은 5장, 모든 실사의 readout 재생은 223장×8점이다.

기존 `replay_initial_joint_v270.py`/`cpu_replay_v1`은 당시 더 작은 `8e-8` 수치 허용값에서 실패한 기록이다. VIS_ONLY 4점의 최대 차이가 `8.336086310123392e-8`였고 판정 차이는 0이었다. 원래 실패·코드를 보존하고, 실행 전 `REPLAY_FLOAT32_SIGMOID_CONTRACT_V2.json`에 새 산술 허용값을 기록했다. 모델, `.5` 판정 임계값, 학습, 과학적 출력은 바꾸지 않았다. 재현에는 V2 명령을 사용한다.

## 2. 실사223장 평가 지표 재계산

```bash
"$MLPY" -B "$TASK_DIR/evaluate_initial_joint_v270.py" --replay "$TASK_DIR/evaluation_v4/portable_replay_v1" --output /tmp/v270_current_metrics_001
```

압축된 평가 입력과 저장 예측만으로 계산하고, 포함된 `REFERENCE_RESULTS.json`과 결과 사전 전체가 같은지 확인한다. 원본 실사 RGB·외부 annotation 경로·모델·GPU를 열지 않는다. 최종 평가본은 `evaluation_v4`다. 이전 evaluation 디렉터리는 실패·수정 이력이며 최종 결과를 대신하지 않는다.

결과는 `/tmp/v270_current_metrics_001/RESULTS.json`, 실행 확인은 `COMPLETE.json`이다. 비교의 주 대상은 동일한 square640 후보에서의 VIS_ONLY와 JOINT다. 좌표 변화의 기준은 동일한 전처리·후보로 생성한 `R0_MATCHED_SQUARE640`이다. 과거 rect=True/N3 좌표와 동일하다고 해석하지 않는다.

## 3. SYN DEV235 저장 출력·학습 범위 감사 재생

```bash
"$MLPY" -B "$TASK_DIR/analyze_syndev_joint_v270.py" --replay "$TASK_DIR/syndev_compact_inputs_v1" --out /tmp/v270_syndev_metrics_001
```

작은 numeric teacher·저장된 DEV 예측·로그·head 상태만 읽는다. 원본 RGB/M4, 전체 neck feature, 모델 forward, optimizer는 사용하지 않는다. `CLASSIFICATION.csv`, `LOCATION_SUMMARY.csv`, `PAIRED_LOCATION.csv`, `PAIRED_VISIBILITY.csv`, `NONDE_DISTRIBUTIONS.csv`, `TRAINING_AUDIT.json`, `RESULTS.json`을 생성한다. 학습 범위 감사는 세 arm의 동일 초기값·draws, 각 300회 갱신, 고정 파라미터 및 BN 상태 보존 등을 검산한다.

## 결과를 해석할 범위

이 실험은 frozen backbone/neck 위 원래 KP cv4와 새 E readout을 300회씩 학습한 제한된 pilot이다. 전체 네트워크 공동 학습의 가능·불가능을 판정한 실험이 아니다. 외부 가림 교사는 실제 물리 표면의 gold가 아니라 저장된 virtual-cuboid external-ray proxy이며 self/outside/partial/centroid는 E 학습에서 제외했다.

특히 FIT raw slot 2/3/6/7은 유효 교사 중 D가 0개이고 E가 각각 95/98/28/26개였다. DEV에도 이 네 slot의 D 교사가 없다. 두 모델은 실사223장 모두에서 이 네 slot을 E로 판정했다. JOINT는 slot0/1을 전부 D로 판정했으며, VIS_ONLY slot1은 24장을 E로 판정했다. 따라서 낮은 SYN loss나 높은 SYN 정확도만으로 이미지의 외부 가림 단서를 배웠다고 주장할 수 없다. slot별 교사 지원의 빈틈과 slot prior shortcut이 중요한 진단이다. 지면과 ray offset의 영향은 원래 first-hit 기록·동일 생성기가 없으므로 가능한 원인이며 확정 원인으로 쓰지 않는다.

실사 주 평가에서 JOINT는 VIS_ONLY 대비 D recall을 높이고 E recall을 낮췄다. 이것을 외부 가림 인식의 개선이나 새 방식 채택 근거로 제시하지 않는다. 반대로 이 결과만으로 joint learning 전반이 불가능하다고 결론내리지 않는다. 재현 PASS는 저장 산출물과 계산의 일관성을 확인하며, 라벨 의미의 타당성이나 새로운 독립 실사에 대한 일반화를 보증하지 않는다.
