# 실행 및 재현 기록

입력은 RGB 한 장과 기존 물리 W/H/D 치수, 기존 보정 K다. 원래 진단 계획의 무학습 범위와 이후 사용자가 요청한 지속 개선 목표를 구분했다. 이번 단계는 후속 개선 실험으로, 새 선형 선택기4개만 학습했다. DINO·R0·DIVERSE backbone/refiner는 재학습하지 않았다. 새 실사 GT·추가 촬영·시간 정보는 사용하지 않았다.

## 고정한 조건

- 이전271차원 입력·9개 해시·signed T/R 타깃·스케일을 보존하고 native 이미지385차원을 추가했다. DINO ViT-S/14의 기존 공식 로컬 가중치와 준비 코드를 그대로 썼다.
- source의 기존 반사 padding100을 특징 표본 위치에서 제외했다. 토큰 문맥에서 반사 영역의 영향이 완전히 사라진다는 뜻은 아니다.
- R0 유효 TRAIN 후보5,194개에서만 FP32 mean/std를 계산하고 std 하한1e−6을 적용했다. VAL/실사에서 정규화를 다시 계산하지 않는다.
- TRAIN2,598행·실패1행·유효 후보 보존, 0 초기화, bias0, ridge1e−4, 과소예측 Huber 추가계수1(비용2:1), sign계수1, 고정 Newton/Armijo를 유지했다. 임계값·계수·epoch·seed 탐색이나 재시작은 없다.
- 4개 모델이 수렴한 후 독립 검산을 통과해야 VAL 입력을 만든다. VAL 입력 추출과 후보 선택 고정, 독립 입력 검산, 품질 계산, 독립 route/오차 검산은 별도 단계다. source45조건 모두 통과해야 실사 평가 경로가 열린다.

## 실제 실행

| 단계 | 실행 식별자 | 결과 |
|---|---|---|
| native TRAIN 재집계 | 9164 | exit0, 기존 token 재사용·새 forward0 |
| native 독립 검산 | 86259 | exit0, 20,776개 descriptor 검산 PASS |
| PREFIT | 55759 | exit0, 기존9개 해시와 추가3개 해시·1,312계수 미분 검산 PASS |
| TRAIN 프로토콜 봉인 | 14426 | exit0, SHA `582ee54ffa121b9bd129b0091bac0b86291c4729efe5964bff968da5a67ccad9` |
| 실제4fits | 27725 | exit0, 312 objective 호출·84 승인 반복·네 모델 인증 PASS |
| 독립 TRAIN 검산 | 12572 | exit0, 재fit0·VAL/실사0 |
| SOURCE VAL 특징·선택 고정 | 97527 | exit0, RGB forward1,024회·모델별1,024행 선택 |
| SOURCE 입력 첫 독립 검산 | 36903 | exit1, binding의 선택적 bytes/논리 경로 형식 오류·완료 영수증 미생성 |
| SOURCE 입력 수정 후 독립 검산 | 71462 | exit0, 동일 이미지1,024개·descriptor8,192개 PASS |
| SOURCE 품질 계산 | 2925 | exit0, 43/45 통과·전체 FAIL |
| SOURCE 독립 검산 | 72069 | exit0, 선택4,096개·T/R16,384값·45판정 일치 |
| 실사 미실행 기록 | 종료 확인 | exit0, source 미달·금지 실사 산출물17개 부재 |

최종 가중치까지 R0_ONLY는54호출/19반복, UNION s1은62/19, s2는142/30, s3는54/16이었다. 최적화 구간 시간 합계는 약56.05초다. 파일 검증·입력 로딩·사전 검산 시간은 이 수치에 포함하지 않는다. 모든 시도와 Armijo 거절 호출은 trace에 보존했다.

GPU 입력 추출 과정의 xFormers 미설치·TypedStorage·CuDNN 우회 경고는 보존하며, 프로세스는 exit0으로 완료했다. 고정 DINO FP32 추론, TF32 off, cuDNN benchmark off, frame batch1, FP16 token 저장 조건을 유지했다. RGB/K를 다시 padding하거나 새 PnP를 계산하지 않았다.

첫 SOURCE 입력 검산은 실제 이미지 SHA가 일치했지만, 선택적 bytes 필드 및 symlink의 논리 경로와 실제 경로를 dict 전체 비교한 검산기 오류로 중단됐다. 검산기에서 SHA를 필수로, bytes는 존재할 때만 비교하도록 수정했다. 원본 검산 코드·해시와 오류를 보존했고 틀린 SHA/크기 거부 및 symlink 합성 검산 후 같은 산출물을 다시 확인했다. 추출·route·가중치·정규화·허용오차는 바꾸지 않았다. [오류 및 수정 기록](APPEARANCE_VERIFICATION_SCHEMA_INCIDENT_KO.md)

SOURCE 입력 최대 절대 오차는1.55049e−6이며 고정 atol=rtol=2e−6을 통과했다. source45조건 중 UNION_s1의 R 중앙값이 R0_ONLY/R0_GEO보다 엄격히 낮아야 하는2조건이 미달했다. 보고서는 독립 source 검산이 완료된 뒤에만 생성한다. [현재 결과](REPORT_KO.md), [TRAIN 검산](TRAIN_CONVERGENCE_KO.md), [SOURCE 입력 검산](SOURCE_VAL_APPEARANCE_VERIFICATION_KO.md), [SOURCE 품질 검산](SOURCE_VAL_VERIFICATION_KO.md), [실사 미실행](REAL_EVALUATION_NOT_RUN_KO.md)이 각각의 상태를 나타낸다.

## 재현 명령

아래 순서는 해당 원본 데이터·가중치·해시가 있는 로컬 환경용이다. 이미 완료된 namespace 산출물을 지우거나 덮어쓰지 않는다. 새 실행에는 별도로 봉인된 namespace가 필요하다. 공개 파라미터 파일은 마지막 승인 checkpoint의 바이트 동일 사본이다.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg MPLCONFIGDIR=/tmp/pallet-mpl
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -B -m scripts.research.pallet_pose_signed_axes_visual_20261001_v1.prefit_review --write
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -B -m scripts.research.pallet_pose_signed_axes_visual_20261001_v1.seal_training
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -B -m scripts.research.pallet_pose_signed_axes_visual_20261001_v1.convex_train all
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -B -m scripts.research.pallet_pose_signed_axes_visual_20261001_v1.verify_train verify
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -B -m scripts.research.pallet_pose_signed_axes_visual_20261001_v1.evaluate_source freeze
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -B -m scripts.research.pallet_pose_signed_axes_visual_20261001_v1.verify_appearance verify --scope SOURCE_VAL
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -B -m scripts.research.pallet_pose_signed_axes_visual_20261001_v1.evaluate_source score
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -B -m scripts.research.pallet_pose_signed_axes_visual_20261001_v1.verify_source verify
```

source 실패 시 `evaluate_real not_run`으로 미실행을 기록한다. source 통과 시에만 `seal_real` → `evaluate_real freeze` → `verify_appearance verify --scope REAL` → `evaluate_real score` → 독립 실사 검산 순서로 진행한다. 수렴·검산·GitHub 게시 자체는 안정적 T/R 개선의 완료 조건이 아니다.
