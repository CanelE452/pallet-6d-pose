# 현재 R0/PoseFix 출력에 맞춘 합성 GEO 보정의 실행 가능성

2026-10-01. 읽기 전용 감사와 저장 예측의 좌표·기하 특징 확인만 수행했다. 이미지 모델 추론0회, fit0회다. 이 문서는 실행 프로토콜이 아니다. 로컬 선행 실험 경로는 이번 공개에 포함된다는 보장이 없어 코드 표기로만 적었다. 측정한 경로·SHA·개수는 [SELECTOR_FEASIBILITY.json](SELECTOR_FEASIBILITY.json)에 있다.

**현재 DIVERSE 3개 seed의 기존 좌표·선택 박스·W/D 후보를 그대로 두는 선택기 재학습은 전체 목표를 달성할 수 없다.** [후보 하한](CANDIDATE_BOUNDS.json)의 자연99 T P90 최선값 평균도 **129.502732cm**로 허용 **126.494939cm**를 넘는다. PRIOR1/FULL125도 clean29 T 중앙값 하한 **3.198106/3.432615cm**가 **3.076355cm**를 넘는다. 따라서 이 후보 집합의 GEO 재학습을 전체 해결책으로 삼는 큰 추론·fit은 보류하는 것이 타당하다. R0의 하한은 clean T **2.929862cm**, 자연 T P90 **120.471370cm**여서 **R0-only 선택기 보정까지 배제되지는 않는다.** 단, R0-only 개선은 FULL/PoseFix 보정의 회복과 다른 결과이며 연구 범위를 바꾸어 부르면 안 된다. 하한을 통과하는 것은 공동 T/R·불확실성·recording 안정성이 달성 가능하다는 증명도 아니다.

## 재사용 가능한 입력과 캐시

현재 R0 checkpoint SHA는 `970a0913b38ed4c9e3662837abccbf9d91b8b0858deafae854c1055e477644f7`이다. `scripts/research/pallet_line_pose_v1/features.py:19,32`가 이 SHA를 강제한다. 이전 S0/S1은 이 checkpoint가 아니므로 그 예측을 현재 R0로 재명명할 수 없다.

| 실제 파일 | 확인 결과 | 사용 범위 |
|---|---|---|
| `data/pallet/results/pallet_line_pose_v1/cache/CACHE_MANIFEST.json` 및 `CACHE_COMPLETE.json` | 정확한 R0 SHA, 완료/PASS, 60,000행. completion에 고정된 manifest SHA 일치 | 현재 R0 좌표·박스·선택 index의 보조 무결성 기준 |
| 같은 cache의 `points.npy`, `boxes.npy`, `score.npy`, `selected_candidate_index.npy`, `input_shape.npy`, `record_index.npy` | source 전체와 selector6144의 ID를 모두 포함. points/boxes는 모델 입력 좌표계의 float32 | 키포인트 confidence9개와 전체 후보 풀이 없으므로 그대로 feature94를 복원할 수 없음. confidence=1 같은 대체 금지 |
| `data/pallet/results/pallet_dim_conditioned_p_v1/source_baseline/*.json` | 1,985개 파일 SHA 전부 `SYNTH_DETECTION_AUDIT.json`과 일치. 실제 전체 candidates에 score/box/xy/keypoints_conf/selected_index 보존 | selector TRAIN129/VAL49/TEST46와 교집합. TRAIN+VAL178장은 온전한 R0 입력 재사용 가능 |
| `_docs/experiments/pallet_dim_conditioned_p_v1/SYNTH_DETECTION_AUDIT.json` | SHA `cdd791f9b6b1b89c20fbbd41673ec38cad903d2f3c9074a741f8e065e8657e53`; utility-selector의 기존 SPLIT에도 같은 binding 확인 | 임의 파일 목록이 아닌 과거 동결된 예측 묶음 |
| `data/pallet/results/pallet_selector_recovery_v1/stage2_synth_scorer/SYNTH_INPUTS.json` | 기존 split lock의 SHA 일치. id/split/image/hw/pad/K/dims만 포함 | 타깃 연결 전 사용할 합성 입력 목록 |
| 같은 폴더의 `SYNTH_RECORDS.json`, `SYNTH_LABELS.npz`, 기존 `EXACT_LABEL_AUDIT.json` | records SHA 확인. exact Xcf width/depth parity 감독의 기존 계약 | 새 예측·특징 동결 뒤 TRAIN/VAL parity만 연결. TEST는 열거나 채점하지 않는 최소안 |

현재 감사 범위에서 TRAIN+VAL 5,120장 전체의 confidence를 보존하는 R0 캐시는 찾지 못했다. 확정 재사용178장을 제외한 **4,942장**은 현재 R0 frozen forward가 필요하다. 다른 캐시를 발견하면 checkpoint뿐 아니라 RGB SHA·좌표계·후보·confidence·runtime을 같은 수준으로 인증한 후에만 이 수를 줄일 수 있다. 부분 캐시를 채우는 데 GT 박스나 선택된 정답 객체를 쓰지 않는다.

## 분할과 노출의 실제 범위

기존 `SYNTHETIC_SPLIT_LOCK.json`의 records/inputs/source manifest/geometry sidetable 네 binding이 모두 일치했다. ID·이미지 SHA·scenario·renderer group 교집합은 TRAIN/VAL/TEST의 모든 쌍에서0이다.

| 분할 | N | renderer group | G38 | P0 | TEX | 완전한 R0 캐시 재사용 |
|---|---:|---:|---:|---:|---:|---:|
| TRAIN | 4096 | 7 | 2946 | 595 | 555 | 129 |
| VAL | 1024 | 1 | 0 | 504 | 520 | 49 |
| TEST | 1024 | 1 | 0 | 496 | 528 | 46 |

분할은 `synth_split.py:20-39`에서 G38 merged archive를 TRAIN에 넣고 P0/TEX의 같은 renderer shard를 함께 배정한 후 SHA 순으로 제한했다. 기존 S0/S1 replay512와 그 파생 scenario511개를 제외했다. VAL/TEST가 각각 renderer group1개라는 사실과 저양각 원천 편중을 유지해야 한다. selector에 대한 그룹 분리이지 넓은 원천으로 학습된 R0가 처음 보는 이미지라는 뜻이 아니다.

이번6개 refiner의 source order는 같은 `data/pallet/results/pallet_posefix_replay_v1/ORDERS.npz`이며 2,400노출/1,412고유 이미지다. 이 실제 학습 입력과 selector TRAIN/VAL/TEST 교집합은 **105/26/26장**이다. source probe의 held256과는 **18/6/7장**이 겹친다. 따라서 VAL1024는 새 GEO의 학습에서는 분리돼 있지만 frozen refiner까지 완전히 미노출인 검증이 아니다. 이것을 없애려고 결과를 본 뒤 분할을 다시 고르지 않는다. 새로운 미노출 평가가 필요하면 별도 사전 계약이어야 한다.

## 좌표·runtime·feature94 계약

`pallet_dim_conditioned_p_v1/paper_evaluate.py:59-78`의 source baseline은 이미100px reflect-padding된 RGB에 R0를 실행한 뒤 **xy와 box에서100을 빼서 저장**했다. 현재 selector K와 RGB는 prepared canvas 좌표이므로 재사용할 때 전체 후보의 xy/box에100을 더한다. 점 confidence, score, candidate 순서와 selected index는 그대로 둔다. 이미 padding된 RGB에100px를 또 붙이지 않는다.

이 어댑터를 TRAIN/VAL178장 전체의 저장 예측에만 적용했다. 원래60k float32 캐시를 역변환한 값과 차이는 최대 point **0.000053406px**, box **0.000045776px**였고 score/index는 일치했다. 원래 full JSON의 값이 더 직접적인 운영 예측이므로 그 좌표를 사용해야 한다. 역변환한 float32 좌표로 덮어쓰지 않는다. 실제 기존 `features.extract`를 거친178쌍 모두 **2×94 finite** 특징을 만들었다. RGB는 열지 않았고 이미지 모델 forward는 없었다.

R0 원래 cache runtime은 **batch1, cuDNN TF32=True, matmul TF32=False, FP32**다. 현재 PoseFix refiner는 **batch1, TF32=False, cuDNN benchmark=False/deterministic=True**다. 누락된 R0를 재생성하는 단계와 refiner 단계를 명시적으로 나누고 각 runtime을 다시 설정해야 한다. 같은 checkpoint만으로 다른 batch/TF32 설정의 수치적 동일성을 가정하지 않는다. 실제 재생성 전에는 결과에 무관하게 고정한 소수 source ID로 full JSON과 모든 후보·confidence·선택을 비교하는 smoke를 예산에 포함해야 한다.

현재 PoseFix 적용은 이전 단계 `infer.load_refiner`와 `CORE.predict(..., cap_fraction=None)`를 그대로 사용한다. crop1.25, 288×384, RGB mean, validity, inverse affine, 중심점8 보존과 나머지 detection contract를 유지한다. source GT를 입력으로 사용하는 기존 `SourceData.item()` 학습 어댑터를 inference 입력으로 재사용하지 않는다.

선택기의 특징은 `pallet_selector_recovery_v1/features.py:30`의 동일 `F.extract(pred,K,dims,hw)`다. score, confidence9개, 그 weighted residual과 원래 중심점까지94개에 포함된다. **특징 입력·투영·잔차는 중심점을 포함한9점이지만, PnP fit은 corner8점**이다. `challenge/evaluation_v2/pnp_selector.py:405-424`는 `object_points[:8]`, `points[:8]`로 SQPnP/LM을 실행하고 이후9점을 투영한다. 이를9점 PnP solver라고 부르면 안 된다. 현재 최종 pose도 기존 `D.candidates`의 corner8 SQPnP/LM physical-frame C2 계약을 유지한다. CPU에서 기존 checkpoint의 score/name tie-break와 캐시의 GEO_name을 먼저 비교하고, 새 checkpoint도 같은 adapter로 적용해야 한다.

## 필요한 경우에만 사전 고정할 최소 설계와 예산

아래는 구현 가능성을 위한 비용 상한 설계이며 실행 결정이 아니다. 모든 fit은 기존 Linear94/pairwise BCE, TRAIN-only 정규화, AdamW1e-3/weight_decay1e-4, batch256 source frames, max30epoch/patience5, earliest best synthetic VAL만 사용한다. 실사 DEV·wood 사례·TEST 점수로 epoch, model, threshold, seed를 고르지 않는다. 현재 hold-W/D나 T-best 선택은 감독 또는 runtime 입력으로 사용하지 않는다.

| 질문 | 새 R0 source forward | frozen PoseFix source forward | 새95-parameter selector fits | 실사 image forward |
|---|---:|---:|---:|---:|
| R0-only calibration의 3개 optimizer seed 반복 | 4942 + 고정 smoke상한8 | 0 | 3, 각 최대480 updates | 0; 기존 R0 후보·예측 재사용 |
| DIVERSE 3개 출력 각각 vs 공통 pooled GEO의 호환성 진단 | 동일 R0 입력 한 번 | 3×5120=15360 | own3 + shared1 =4 | 0 |
| 기존 paired SINGLE/DIVERSE 비교를 보존하는 확장 진단 | 동일 R0 입력 한 번 | 6×5120=30720 | own6 + shared1 =7 | 0 |

FULL125 또는 PRIOR1까지 source calibration의 입력 arm으로 추가하면 각각 **5,120 frozen refiner forward**가 더 필요하다. TEST source inference는 최소안에서0이다. 좌표/RGB 모델은 어느 안에서도 새로 학습하지 않는다. source에서 특징 추출 실패한 행 수는 반드시 기록하고, 실사는 원래173장과 모든 failure 분모를 유지한다. 시간은 측정된 최근 refiner의173장 약2.55~2.60초 forward 합계로 보면 모델별5120장 약75~77초의 순수 forward가 산술 추정되지만, RGB 읽기·해시·PNP·직렬화·모델 로딩이 제외돼 있어 완료 시간 보장이 아니다.

own-vs-shared의 공정한 최소 비교는 동일 source frame별 minibatch와 optimizer update 상한을 사용하고, pooled는 각 frame의 고정된 모델 view 손실을 먼저 평균해 weight1을 주는 것이다. own은 자기 모델 view 하나를 사용한다. 이렇게 하면 pooled의 view 수만큼 optimizer update 수가 늘어나는 과거 혼입을 피할 수 있다. invalid-view 처리와 TRAIN/VAL의 공통 유효 frame 규칙을 생성 결과 전에 고정하고 실제 제외 수를 공개해야 한다. own은 해당 모델 VAL, shared는 모델별 VAL 정확도의 사전 고정 평균으로 earliest best를 고르며 이 목적 차이도 보고한다. refiner seed1/2/3은 이미 고정한 세 모델 모두를 평가한다. own/shared 중 실사 최고 조합이나 최고 seed를 새 winner로 고르지 않는다.

R0-only의 optimizer seed3회는 refiner 학습seed3회와 다른 반복이다. 이 안을 선택한다면 같은 학습 데이터의 작은 선택기 최적화 반복이라는 범위와 비교 기준을 새 계약에 명시해야 한다. 단일 model-conditioned fit과 historical pooled GEO 비교만으로 “출력 분포 일치의 순수 효과”를 선언할 수도 없다.

현재 우선 판단은 **DIVERSE 전체 목표를 위해 선택기만 재학습하는 실행은 보류**, **R0-only 보정은 하한으로 배제되지 않았으나 별도의 좁은 가설**이다. 원래 FULL/self-training 개선을 조용히 R0-only 개선으로 대체하지 않는다. 학습을 다시 시작하기 전에 [하한 검토](BOUNDS_REVIEW_KO.md)와 검출/연속 좌표의 남은 제약을 함께 고려해야 한다.
