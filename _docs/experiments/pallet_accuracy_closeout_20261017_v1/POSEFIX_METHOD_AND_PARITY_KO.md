# PoseFix+SubPix 공정 확장: 입력 계약과 참조 없는 재현

이번 비교는 synthetic-only PRIOR1 seed1 last6000의 **첫 uncapped 출력**을 사용한다. 기존 원95장의 두 고정 cap 정의를 모두 복원했으며, 어느 하나를 PRIMARY192 결과로 대표 선정하지 않는다. 판정은 `TWO_FIXED_COMPARATORS_NO_GLOBAL_WINNER`다. 이 비교는 사람 관절 원문 PoseFix 자체의 재현이 아니라 기존 팔레트용 포트와 기존 공통 F를 포함한 전체 시스템 비교다.

## 동일하게 복원한 실제 연산

1. 기존 `seed1_REAL_DEV.npz`에서 `points[selected_rows,0,1]`을 읽는다. chain0=RAW, pass1=첫 보정이며 RAW 보정량에 cap을 새로 걸지 않는다. checkpoint SHA256은 `b9670d1a42db8d4e80ae1efe55eb237f05f81f7bd089cefe4051bbedd7d0a581`, neural cache SHA256은 `683b7395860865f7e10cba5e1dfb7c04e84234dd296340416c4759b671c95f06`이다.
2. 원 RGB를 SHA로 확인하고 원해상도 grayscale에서 기존 SubPix를 한 번 적용한다. `winSize=(5,5)`, `zeroZone=(-1,-1)`, `maxCount=40`, `epsilon=.001`, 최소 영상 extent15, float32 OpenCV 입력 후 float64 결과다. unsupported, 결측, 영상 밖 초기점, OpenCV 예외, 비유한/영상 밖 결과의 원래 fallback 규칙을 유지한다.
3. **PF-anchor:** SubPix 추가 이동만 PF 출력 기준 원영상 대각선1%로 제한한다. **YOLO-anchor:** 동일한 native SubPix 결과를 원 YOLO 좌표 기준 대각선1%로 제한한다. 중심8과 unsupported slot은 원 YOLO와 동일하게 유지한다.
4. 기존 `pose.infer(...,source=False)`의 정의를 그대로 실행한다. prediction-only W/D 선택 후 native 유한8점의 SQPnP와 RefineLM, 원 물리 크기와 선택된 차원의 Q를 사용한다. 정사각 W/D 병합과 unavailable 처리도 원래 함수 그대로다. 사람 visibility, initial pose, 참조 pose는 제공하지 않는다.

주요 계약 함정은 추가143장의 `NATIVE_INPUTS143.frame.raw_points`가 원 RAW YOLO 좌표가 아니라 **N3+SubPix 출력**이라는 점이다. 이번 두 anchor의 원점·support는 prediction-only `pallet_posefix_replay_diagnosis_v1/INPUTS.json`의 `points`와 `valid`에서 가져왔다. 원95장의 anchor/support, neural tensor PASS0와 원 YOLO를 전부 원소별 검산했고 불일치는0이었다. 추가143에서는 native 입력의 K·등록 치수·RGB 메타데이터만 사용했다. 새 모델 forward, 학습, seed 변경은0이다.

## 채점 전에 수행한 parity

`posefix_replay infer95`는 원95장×다섯 경로=475행을 참조 접근 차단 상태에서 봉인했다. 다섯 경로는 RAW YOLO, N3+SubPix, PF RAW, PF-anchor, YOLO-anchor다. 별도 `parity95` 프로세스가 원95장 봉인 출력과 비교했다.

| 검증 | 결과 |
|---|---:|
| native 점 배열의 완전 동일 | 475/475 |
| 최종 pose 전체 dictionary 완전 동일 | 475/475 |
| 위치·회전·차원·재투영 수치 최대 절대 차이 | 0 |
| selected hypothesis / available 불일치 | 0 |
| 허용차 | 절대1e-10 |

이 gate가 통과한 다음에만 같은 두 정의를 238장(CAL46+TEST49+추가143)에 실행했다. PRIMARY192와 CAL46은 별도 gzip 파일로 봉인했다. 새 PF RAW의 추가143장 최종 pose는 기존 저장 RAW F와 모든 비교 필드에서 최대차이0이었다. 따라서 캐시, 공통 F, 치수·Q 변환의 다른 변형을 연결한 결과가 아니다. 정확도66/71/70 및 N0/E0 집계는 별도 후봉인 평가기에 맡겼다.

원95장의 당시 `compare.py`, 공통 F, SubPix, selector, geometry 소스 SHA가 당시 PROTOCOL에 기록된 SHA와 일치했다. `posefix_replay.py`는 GT loader를 포함한 모듈 전체 실행 대신 **그대로인 함수 정의만 AST로 추출**하며, W/D selector는 원 모듈을 사용한다. 기존95의 부수적인 joint curriculum 재실행은 이번 비교에 필요하지 않아 실행하지 않았다.

## I/O 경계와 실패 보존

예측 프로세스는 `FIXED_REFERENCES`, `TARGETS.json`, isolated scoring, GT annotation, metric/result/oracle 파일 읽기를 거부한다. 실제 경계 receipt에는 차단 probe3회가 남는다. native C로 수행되는 RGB 읽기는 직전에 파일 SHA와 실제 canvas를 직접 검증한다. 새 owned 출력 밖 쓰기도 금지한다. 평가 코드는 별도 프로세스이며, 예측·프로토콜 봉인을 확인한 뒤에만 고정 참조를 읽는다.

238장×세 PF 경로=714개의 F 결과는 모두 available였다. SubPix 영상 밖 초기점83개는 기존 규칙대로 유지했으며 삭제하거나 성공으로 처리하지 않았다. 238장의 실제 SubPix query는1,821점이었다. 이 점별 fallback과 최종 pose unavailable는 다른 단위다. 좌표 한 점이 유지됐다는 사실은 B/I 상태 판단이나 완전한 영상관측을 뜻하지 않는다.

`posefix_replay.py`에는 평가 참조나 성공 라벨을 읽는 코드가 없다. 역사적 예측 parity는 별도 command이며 성공/오차를 계산하지 않는다. original95 순서와 native143 순서를 그대로 저장했고, 평가기는 immutable ID로 canonical ledger와 결합한다. 순서 재배열이 있더라도 점·pose를 변경하지 않는다.

## 실제 실행량

| 단계 | wall(s) | CPU(s) | F 호출 | SubPix 이미지 / 점 query |
|---|---:|---:|---:|---:|
| predictor-only 입력 준비·봉인 | 0.118 | 0.118 | 0 | 0 / 0 |
| COMMON95 기술 재현 | 1.424 | 1.389 | 475 | 95 / 719 |
| 고정 정의238장 확장 | 2.630 | 2.569 | 714 | 238 / 1,821 |

프로세스 기록의 peak RSS는 각 단계1,080,456KiB였으며 BLAS/OpenCV thread는1이었다. 이 수치는 이미 존재하는 neural 출력을 재사용한 **후처리 실행량**이다. PoseFix backbone 추론 시간을 포함한 end-to-end 속도나 경량화 성능으로 주장하지 않는다. 95 parity와238 확장에95장이 겹치며, parity 검증 비용을 따로 보존했다.

## 재현 명령

로컬 기존 캐시·RGB가 있는 checkout에서 새 빈 private 디렉터리를 지정한다. 원본 사용자 작업 폴더·운영 설정·가중치·기존 예측은 바꾸지 않는다.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHON=/path/to/existing/pallet-yolo26/python
$PYTHON -B -m scripts.research.pallet_accuracy_closeout_20261017_v1.posefix_replay prepare --repo "$PWD" --private "$NEW_PRIVATE"
$PYTHON -B -m scripts.research.pallet_accuracy_closeout_20261017_v1.posefix_replay infer95 --repo "$PWD" --private "$NEW_PRIVATE"
$PYTHON -B -m scripts.research.pallet_accuracy_closeout_20261017_v1.posefix_replay parity95 --repo "$PWD" --private "$NEW_PRIVATE"
$PYTHON -B -m scripts.research.pallet_accuracy_closeout_20261017_v1.posefix_replay extend192 --repo "$PWD" --private "$NEW_PRIVATE"
$PYTHON -B -m scripts.research.pallet_accuracy_closeout_20261017_v1.posefix_replay parity143 --repo "$PWD" --private "$NEW_PRIVATE"
$PYTHON -B -m unittest scripts.research.pallet_accuracy_closeout_20261017_v1.test_posefix_replay
```

`NEW_PRIVATE`는 사전에 만든 새 빈 결과 디렉터리다. 각 출력은 exclusive-create여서 기존 시도를 덮어쓰지 않는다. 파일을 지우고 수치가 좋아질 때까지 다시 실행하는 절차가 아니다. 실제 신규238 실행은 한 번이며 두 anchor는 같은 native SubPix 출력에서 동시에 만들어졌다.
