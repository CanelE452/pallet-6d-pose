# 팔레트 A/B/C 실행 인계 — 2026-10-06

[확인] 요청한 지시문 703행을 모두 읽고 `CanelE452/pallet-6d-pose`의 최신 원격 `88ee557e09e32ee2c7dd11b17309e05a325cb3c2`에서 별도 worktree와 `research/joint-action-handoff-20261006` branch를 만들었어. 원래 로컬 main의 HEAD `39d4219a`, 수정된 tracked 5파일, 변경·미추적 상태 491항목을 보존했어. 5파일의 내용 SHA와 diff/status는 검산했으며, 5,328개 미추적 파일 전체 내용을 각각 해시했다고 주장하지 않아.

[확인] 이번 범위는 등록 치수가 주어진 조건의 RGB(red, green, blue) 영상 국소 코너 정제야. 같은 참조 코너에서 재구성한 PnP(Perspective-n-Point) 자세 비교와 독립 물리 계측 정확도는 다른 증거야. 반복 사용한 개발자료(DEV, development) 319장/13세션, 기존 합성 평가 1,985장, 별도 정사각형 119장/1세션, 기존 12장/96점과 8,910행 영상 감사를 합친 새 시험집합을 만들지 않았어.

## 완료 상태와 해석

[확인] 고정 예산 A의 실용 개선은 확인하지 못했어. 실제 DEV와 기존 synthetic 개발 진단 모두 GEO−N3/PoseFix 정규화 코너 오차의 seed 평균 차이와 95% bootstrap 구간이 양수였어. 실제 GEO−PERM의 평균 차이는 음수지만 좋은 코너 손상 보존 조건을 통과하지 못했고, synthetic에서는 GEO가 PERM보다 악화됐어. 큰 oracle 여지를 배포 이익으로 바꾸지 못한 결과를 그대로 공개해.

아래 값은 seed1 / seed2 / seed3 순서야. 코너 med/P90은 매칭 관측 코너의 조건부 값이고, 전체 penalty/PCK·pose 평균 대비와 원행은 `A_RESULT_KO.md`, `results/A_summary.json`에 있어. 모든 방법의 실제 F 자세 coverage는 319/319야.

| 방법 | 코너 med px (세 seed) | P90 px (세 seed) | T med cm (세 seed) | R med ° (세 seed) |
|---|---|---|---|---|
| N3 | 5.7446 / 5.7677 / 5.8222 | 42.3204 / 41.8721 / 42.2087 | 7.0392 / 6.7902 / 7.3735 | 2.1104 / 2.0727 / 2.0281 |
| PoseFix | 5.5626 / 5.5284 / 5.5911 | 43.5346 / 44.0303 / 44.1571 | 7.1854 / 6.8438 / 6.8250 | 1.9540 / 1.9871 / 2.1422 |
| FROZEN_GEO_I | 6.5809 / 6.6776 / 6.6244 | 42.7268 / 43.2799 / 44.6366 | 8.1923 / 7.4698 / 8.2917 | 2.4993 / 2.3924 / 2.4717 |
| FROZEN_GEO_J | 6.7045 / 6.7207 / 6.6944 | 43.8900 / 43.8900 / 43.8900 | 7.8969 / 7.8936 / 7.8969 | 2.5389 / 2.5389 / 2.5192 |
| FROZEN_PERM_I | 6.5809 / 6.6776 / 6.6244 | 42.7268 / 43.2799 / 44.6366 | 8.1923 / 7.4698 / 8.2917 | 2.4993 / 2.3924 / 2.4717 |
| FROZEN_PERM_J | 6.7050 / 6.7181 / 6.7181 | 43.8900 / 43.8900 / 43.8900 | 7.8936 / 7.8969 / 7.8969 | 2.4717 / 2.5389 / 2.5356 |
| FIT_GEO_J | 6.4291 / 6.4614 / 6.4908 | 44.7775 / 43.0952 / 43.9856 | 7.2888 / 7.6630 / 7.8845 | 2.4026 / 2.4674 / 2.4918 |
| FIT_PERM_J | 6.6265 / 6.7655 / 6.5640 | 44.1047 / 43.5901 / 43.6749 | 8.2266 / 8.3237 / 7.8969 | 2.5502 / 2.5579 / 2.3581 |

[확인] 실제 DEV의 seed 평균 정규화 코너 거리 E_sym(승인된 전체 대칭 tuple의 평균 거리/원영상 대각선) 대비: GEO−N3 +0.000785698 [0.000454285, 0.001255931], GEO−PoseFix +0.000900928 [0.000500382, 0.001412779], GEO−RAW −0.000319115 [−0.000652309, −0.000054014], GEO−PERM −0.000191328 [−0.000398712, −0.000024607]. 같은 13세션 추출·10,000회·seed20260917의 탐색적 95% 구간이며 다중 대조 보정은 하지 않았어. GEO의 RAW 대비 <5→>10px 손상 코너는 6/6/5개, N3 대비 17/15/19개, PERM 대비 5/15/22개라 사전 보존 조건은 미충족이야. GEO−N3 frame 평균 회전 차이는 +2.908621° [0.819988, 4.779611]였어. 중앙값만으로 이 tail 손상을 숨기지 않아.

[확인] 동결 GEO의 J−I E_sym은 +0.000064348 [−0.000017404, 0.000182033], PERM은 +0.000068629 [−0.000009067, 0.000181559]로 실제 DEV의 공동 선택 이익을 확립하지 못했어. I는 더 큰 조합 공간, J는 같은 bank의 단일 후보여서 각각 맞는 oracle gap으로 해석해. 최종 J의 oracle gap은 허용오차 밖 음수 0건이었고 전 방법의 실패/공동 성공 분모는 전체 행에 남겼어.

[확인] A의 최종 실행 상태·세 seed 전체 결과는 `A_status.json`, `results/A_summary.json`과 최종 A 보고서에 연결해. 원래 N3 headline을 새 방법 결과로 덮어쓰지 않아. B는 취득 패킷·실행 가능한 import/validate/evaluate·검증 완료이며 실제 독립 물리 pair는 0개라 `BLOCKED_DATA`야. C는 본문/보충의 실제 문장·표·캡션과 참고문헌을 수정했고, 최종 A 영수증을 탐색적 보충 결과로 통합해.

[확인] source selection 1,031장의 원영상 9점으로 NoOp+최대 두 W/D(width/depth) 가설×100개 후보를 만들었어. NoOp는 원시 좌표의 정확한 복사이고 center8·instance·box·confidence·결측을 유지해. 회전은 카메라 축에서 물체 중심을 기준으로 적용하고, 회전 rad/이동 m의 투영 민감도로 방향을 정규화한 뒤 비선형 투영 root search로 원영상 대각선 1% 이동 제한을 적용해. 코너별 clipping으로 후보를 바꾸지 않아. 생성 자세가 아닌 같은 최종 함수 F(q)의 W/D 선택과 PnP 결과를 평가해.

[확인] source oracle은 ADD(Average Distance of Model Points, 모델 점 평균 거리) 계열의 ADDsym_m(허용 proper rotation 하의 대응하는 정준 8코너 평균 3D 거리, m)을 최소화하는 한 후보를 골랐어. surface ADD-S가 아니야. GEO의 비영 여지 954/1,031, 중앙 headroom 0.011915445 m, PERM의 1,003/1,031·0.013282408 m였고, 양쪽 raw/oracle 자세 산출은 1,030/1,031로 같았어. 정답을 쓰는 oracle은 오프라인 후보 여지 진단이며 배포 알고리즘이 아니야. PERM도 높은 상한을 가져 이 결과만으로 기하 결합의 우위를 주장할 수 없어.

[확인] 독립 코너 hard argmax(I)와 공동 후보 hard argmax(J)를 같은 bank·동결 N3 세 seed에서 비교해. 같은 코너의 비NoOp 후보를 고정 순열로 섞은 PERM은 후보 좌표·RGB patch 특징·metadata의 주변 multiset을 유지해. GEO/PERM의 새 학습은 같은 seed 초기 가중치·배치 순서·6,000 update만 사용하고, 체크포인트 선택·cap/loss/seed/모델 재탐색 없이 마지막 결과를 보고해. 실제 학습량과 직접 GEO−PERM/J−I 대비·손상·실패·oracle gap은 최종 영수증에 있어.

[확인] 기존 319 DEV에서 Base→N3 코너 중앙값 6.721→5.778 px, 위치 7.897→7.068 cm, 회전 2.539→2.070°는 보존했어. 치수 조건부 추가 변화 약 −0.1662 px, 대칭 추가 변화 약 +0.0004884 px를 전체 방법의 개선과 구분해. PoseFix 방식의 uncapped RAW 첫 pass 중앙값 5.561 px/6.951 cm/2.028°는 N3보다 낮고, N3 코너 90백분위(P90) 42.134 px는 PoseFix 43.907 px보다 낮아. 입력·감독·loss·학습/선택 예산이 다른 전체 방법 비교야. PCK(Percentage of Correct Keypoints, 정확히 예측한 키포인트 비율)@10px는 결측·매칭 실패를 포함한 전체 참조 코너 분모로 보고해. 실제 검출 후보와 최종 F 산출은 각각 319/319, 코너 정답 매칭은 311/319로 구분해.

[확인] `results/A_REAL_DEV_DESCRIPTIVE_SUBGROUPS.json/.csv`와 `A_REAL_DEV_HUMAN_CORNER_STATES.csv`는 최신 사람 등급 153/92/74장, 기존 거리 태그 near/mid/far/unknown 155/103/59/2장, 정준 참조 코너 2,499개에 연결한 관측 상태로 동일 A 행을 재집계해. 새 거리 경계·레이블·추론·자세 계산은 추가하지 않아. 태그의 물리 기준·좌표 출처·노출 시점 미확인을 그대로 남기고, 겹치는 관측 상태별 frame 집단을 합쳐 표본 수를 늘리지 않아.

## 실제 수정·검증 근거

[확인] `C_report_KO.md`, `C_MAIN_CHANGE_NOTES_KO.md`, `PAPER_REVISION.patch`가 실제 문장과 숫자의 출처를 설명해. Abstract/Introduction/Related Work/Method/Results/Discussion/Limitations/Conclusion과 표·그림 캡션을 수정했어. C2/C4는 정준 수직 Y축에 대한 180°/90° proper rotation이며, W=D가 외관·작업 대칭을 자동 승인하지 않아. Simple Baselines의 adaptation head 출처와 ResNet backbone 원전을 구분하고, BB8·후속 pose refinement를 생략한 일반화를 고쳤어.

[확인] 기존 bibliography 42개에서 현재 본문 45개+보충 1개=46개를 전수 감사했어. 기존 수정 항목 15개(서지 표현 12개 및 인용 주장 수정 포함), 필요한 신규 4개, 미확인 metadata 0개야. 전문/초록/metadata 확인 범위와 접근 제한을 `REFERENCES_AUDIT.tsv`에 따로 남겼어. 46개 전부의 구현 전문을 확인했다는 뜻은 아니야. 인용 순서·중복·미사용·undefined citation 및 실제 그림 의존성을 검산하고 main 15페이지/supplement 14페이지 PDF를 컴파일했고, 81개 실제 TeX/style/사용 그림 입력의 SHA를 묶었어. 한국어 검토본이며 최종 영문 투고·저자 소속 완성을 주장하지 않아.

[확인] Base/N3 seed1/PoseFix seed1은 같은 RTX 3080·환경·RAM의 원본 BGR(blue, green, red) 입력에서 전처리→검출/공유 특징→crop/보정/decoder→최종 PnP까지 동기화 wall-clock으로 쟀어. 26프레임/13세션·arm별 준비 20회·5반복이며 중앙값은 11.055908/14.469713/25.238526 ms, P90은 13.080935/17.169513/27.310544 ms야. 각 130/130 자세 산출, 모델 원출력 최대 차이는 0/0/0.000121719 px≤0.001 px야. decoding/모델 loading은 제외하고, process-resident memory를 방법별 독립 메모리로 비교하지 않아. 새 A 비용은 별도의 같은 입력 패널에서 Base/N3/Frozen J/GEO/PERM 중앙값 11.080369/14.432943/56.663289/56.541317/56.666900 ms, P90 13.428345/16.635635/59.234860/59.173654/59.837343 ms였어. 각 130/130 산출이며 모든 750호출의 좌표 차이 0px·action/최종 W/D/coverage 일치를 `RUNTIME_A_VERIFICATION.json`에서 확인했어. 두 측정 패널의 분포를 합치지 않았어.

[확인] 첫 runtime은 기존 YOLO의 cuDNN tensor float 32(TF32) 수치 계약을 끈 상태여서 parity 검증 실패 후 450회 전부 폐기했어. 원래 계약으로 다시 측정한 450회와 섞지 않았어. A bank 생성의 반복 npz materialization과 결측 NaN backward/J bank membership 문제도 정식 update 전에 수정하고 유효한 후보·oracle만 재사용했어. 새 A 첫 시간 측정도 fresh CPU/BLAS 재집계와 겹쳐 750호출의 시간을 폐기하고 조용한 환경에서 같은 규칙으로 재측정했어. 폐기·중단과 실행량은 영수증에 남겼어. 최종 계수는 정식 6fit/36,000 update/576,000 노출/2,383.508초, 별도 smoke4/64노출, 오프라인 후보 F 1,340,270회/실패3/234.709초, frozen+fit 평가232.996초, 실제 방법 최종 F 원행57,600개야. 전체경로 timing은 유효1,200호출+폐기1,200호출=2,400호출이고 유효 패널 wall11.232/31.198초야. 생성 초기화/fixture PnP는 이 평가 F 계수에 합치지 않으며 정확히 기록되지 않은 precompute 재시작 시간을 추정 합산하지 않아.

[확인] 작은 계약 시험 25개(A 4/B 16/C 5)가 통과했어(최종 0.756초). fixture는 수학·형식 검증이며 실제 물리 가림 성능이 아니야. 원래 radial API와 실제 동결 세 checkpoint의 출력은 별도 독립 검토에서 정확히 일치했어. 새 동적 좌표 변환의 FP32 반올림 logit 차이는 최대 약 4.5e−5로 구분해서 기록했어. 실제 실행 때 GPU sampling mask tensor를 저장하지 않았으므로 `A_SAMPLING_MASK_RECEIPT.json`은 고정 좌표·affine·32점 stencil로 CPU FP32에서 재구성한 2,304frame의 bitmask·공통 point support라는 범위를 명시해. 큰 bitmask와 재생성 bank는 외부 cache에 해시로 연결했어. 이 보완의 CNN/최종 F 채점/update는 0이며 실제 319bank 초기화 solvePnP 1,276회·RefineLM 1,276회, wall20.449초는 별도 기록이야. fresh 분석126.04초에서도 실제 bank957회 재생성의 개별 초기 PnP 수는 미계측으로 남기고 0회로 표시하지 않아. 기존 공유 증거 29,747개 검산/실패 0개 및 8,910행·12장/96점을 재사용했고, source 캐시 17배열 73,741,202,176 bytes는 전체 내용 SHA를 읽기 전용으로 남겼어. 추가 실제 입력 663개/436,658,695 bytes의 SHA도 `INPUT_DEPENDENCIES.json`에 있어.

## 없는 입력과 실제 재개 명령

[확인] B 실제 취득·독립 참조·사람 검수·물리 pair 평가는 `BLOCKED_DATA`야. 같은 팔레트/카메라의 clean/물리 가림 RGB pair, 프레임별 독립 canonical 6D 참조와 원계측 ID, 카메라·시간·외부계측 calibration, 구간 전체의 상대 자세 안정 기록, 측정 불확실도/공통 참조 cross covariance, 사람 검수와 recording/pose 단위 분리·노출 기록이 필요해. 해당 모델의 최종 F(q) JSONL 출력은 취득 후 의존 작업 `NOT_RUN_DEPENDENCY`야. 기존 9,029관측의 독립 T/R 참조 0개와 현재 자료가 이 의존성을 충족하지 않는다는 감사 결과를 유지해.

[확인] 필요 자료량은 목표 평균 차이의 신뢰구간 반폭 h와 pilot의 recording별 paired SD s_D에서 `ceil((1.96*s_D/h)^2)`를 초기 근거로 삼고 실제 cluster 구조로 다시 판단해. 현재 319 DEV의 세 seed 평균 N3−Base session mean 위치/회전 SD 5.014063 cm/3.496772°는 이 기존 대비의 변동이야. 미래 물리 가림 대비의 분산으로 바꾸지 않아. h·실제 pair pilot 분산은 미확정이며 미연결 126행을 임의 요구량으로 삼지 않았어.

아래는 확인한 실제 CLI야. 이 환경의 원데이터·checkpoint 경로를 그대로 읽고, 큰 생성 bank/새 checkpoint는 저장소 밖에 둬.

```bash
cd /home/minjae/Documents/github/pallet-pose-handoff-20261006
PALLET_PY=/home/minjae/anaconda3/envs/pallet-yolo26/bin/python

"$PALLET_PY" -m scripts.research.pallet_joint_action_handoff_20261006_v1.run --stage status
"$PALLET_PY" -m scripts.research.pallet_joint_action_handoff_20261006_v1.run --stage verify

# 완료된 fit/후보/기준값/runtime은 protocol·입력·code·checkpoint SHA가 같을 때 재사용해.
"$PALLET_PY" -m scripts.research.pallet_joint_action_handoff_20261006_v1.run \
  --stage resume \
  --source-root /home/minjae/Documents/github/pallet-pose \
  --cache-dir /tmp/pallet-joint-action-cache --workers 8 \
  --tex-engine /tmp/pallet-paper-tools/gnu15/tectonic
```

source 전체 배열 내용의 재검산은 `source_hashes --source-root ... --rehash`, 추가 입력의 재검산은 `input_dependencies --source-root ...` 모듈로 실행해. 실제 통합 resume은 exit0이며 기존 fit/checkpoint/예측/oracle/runtime/mask 40개 해시가 불변이었어(`RESUME_VERIFICATION.json`). 추가 CNN/최종 F/update/runtime은0이고 bank 생성용 초기 PnP는 별도 미계측이라고 표시했어. 다른 머신에서는 원입력·동일 checkpoint/cache·해당 CUDA/Python 환경·실제 Tectonic 경로를 먼저 복원해야 해. 의존성이 없으면 이미 공개한 행/원고/해시와 해당 상태를 확인할 수 있으며 새 학습을 완료했다고 표시하지 않아.

실제 pair 자료를 받으면 `measurement_packet/README_KO.md`의 `PAIR_DATA_ROOT`, `PAIR_SOURCE_MANIFEST`, `PAIR_OUTPUT_ROOT`, `PAIR_FINAL_POSES`를 실제 경로로 설정하고 다음 순서로 실행해. 이번 환경에 그 경로/자료가 존재한다는 뜻은 아니야.

```bash
python3 scripts/research/pallet_joint_action_handoff_20261006_v1/measurement.py import \
  --manifest "$PAIR_SOURCE_MANIFEST" --data-root "$PAIR_DATA_ROOT" \
  --output "$PAIR_OUTPUT_ROOT/imported_manifest.json"
python3 scripts/research/pallet_joint_action_handoff_20261006_v1/measurement.py validate \
  --manifest "$PAIR_OUTPUT_ROOT/imported_manifest.json" --data-root "$PAIR_DATA_ROOT" \
  --output "$PAIR_OUTPUT_ROOT/VALIDATION.json"
python3 scripts/research/pallet_joint_action_handoff_20261006_v1/measurement.py evaluate \
  --manifest "$PAIR_OUTPUT_ROOT/imported_manifest.json" --data-root "$PAIR_DATA_ROOT" \
  --predictions "$PAIR_FINAL_POSES" --split independent_test \
  --output "$PAIR_OUTPUT_ROOT/evaluation"
```

## 파일과 Git 증거

| 내용 | 실제 경로 |
|---|---|
| 개별 상태·정지한 의존성 | `status.json`, `A_status.json`, `B_status.json`, `C_status.json` |
| 예산·ID·입력·출력 해시 | `protocol.json`, `run_manifest.json`, `A_protocol.json`, `A_manifest.json`, `INPUT_DEPENDENCIES.json`, `SOURCE_CACHE_HASHES.json` |
| A 전체 행·대조·손상·실패·oracle gap | `results/A_*.json`, `A_fits/*.json`, `A_smoke/*.json` |
| 기존/새 A runtime 원행과 검산 | `RUNTIME_MATCHED.json`, `RUNTIME_VERIFICATION.json`, `RUNTIME_A.json`, `RUNTIME_A_VERIFICATION.json`, `results/runtime*_rows.json` |
| B 수집/불확실도/분리 schema | `measurement_packet/README_KO.md`, `measurement_packet/SCHEMA_KO.md`, `measurement_packet/ACQUISITION_KO.md` |
| 실제 개정 원고 | `paper_updated/main.tex`, `paper_updated/main.pdf`, `paper_updated/supplement.tex`, `paper_updated/supplement.pdf`, `PAPER_REVISION.patch` |
| 문헌의 실제 주장·확인 범위 | `REFERENCES_AUDIT.tsv`, `REFERENCES_AUDIT_SUMMARY.json`, `PRIMARY_SOURCE_READ_NOTES_KO.md` |
| 검증 원행·독립 검토 | `results/CONTRACT_TEST_OUTPUT.txt`, `results/INDEPENDENT_A_REVIEW.json`, `REUSED_EVIDENCE_VERIFICATION.json`, `C_VISUAL_REVIEW.json` |

[확인] raw/private 영상·source 특징 캐시·학습 checkpoint·타인 논문 PDF·무관한 원사용자 변경은 이 결과 commit에 포함하지 않아. 원고 PDF는 이번에 실제 빌드한 자체 원고야. 로컬/원격 최종 SHA는 파일 내부의 자기참조를 피하고 commit/정상 push 후 CLI 응답 및 저장소 밖 `/home/minjae/Documents/github/pallet-joint-action-push-20261006.json`에 기록해. `run --stage status`는 그 영수증이 현재 HEAD와 맞을 때 Git 상태를 `DONE`으로 표시해. 원격 작업 branch에 push한 것은 main 반영 완료를 뜻하지 않아.
