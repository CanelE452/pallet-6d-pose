실제 A~G 실행과 보완을 완료하되, C1 물리축 복원·C/D 공통 분할·Stage1 사전등록 조건의 실패를 통과로 바꾸지 않았다.

# 지시문 항목별 완료 대조

이 파일은 최초 게시 후 원행·실행 코드를 지시문 각 항목과 다시 대조한 기록이다. 산술 검산 PASS와 실험 계약 FAIL을 구분한다. 기존 결과·학습 설정을 보존하며 성능이 나빠 추가 seed·모델·하이퍼파라미터를 탐색하지 않았다. [확인] 최신 상태는 [EXPERIMENT_STATUS_COMPLETION.csv](EXPERIMENT_STATUS_COMPLETION.csv), 누적 비용은 [COMPLETION_COMPUTE_RECEIPT.json](COMPLETION_COMPUTE_RECEIPT.json), 산술 검산은 [SUPPLEMENT_INDEPENDENT_VERIFICATION.json](SUPPLEMENT_INDEPENDENT_VERIFICATION.json)이다.

| 지시문 항목 | 실제 실행·검증 | 결과·해석 제한 |
|---|---|---|
| §0 A~G 독립 진행 | 아래 모든 작업에 실제 실행/재사용/차단 근거 | 실패 때문에 다른 작업을 취소하지 않음 |
| §1.1 환경·dirty tree | [SOURCE_PRESERVATION.json](SOURCE_PRESERVATION.json), [COMPLETION_PREPUBLICATION_AUDIT.json](COMPLETION_PREPUBLICATION_AUDIT.json) | 원449항목·tracked diff byte 동일; 별도 worktree |
| §1.1 현 코드·과거 선택기 구분 | [CLAIM_AUDIT.md](CLAIM_AUDIT.md), [BC_INPUT_CONTRACT.json](BC_INPUT_CONTRACT.json) | 이전 GEO_IMG_MLP를 신규 기여로 이름만 바꾸지 않음 |
| §1.2 새 출력 | 코드·문서 모두 새 namespace | 기반 evaluator·solver·과거 자료·원고 수정0 |
| §1.3 사용 가능 실사 | [INPUT_CONTRACT.json](INPUT_CONTRACT.json) | REAL173/9세션, Severe63 포함; 보호 FINAL146 새 평가0 |
| §1.3 RECT245·SQUARE118 | [METRICS.json](METRICS.json), INPUT_CONTRACT | 허용 교집합110 기술 분석; SQUARE 별도 참조/C4/단일세션, 합산0·long/short NA |
| §1.3 과거 노출 | [BC_INPUT_CONTRACT.json](BC_INPUT_CONTRACT.json), [D_CROSS_TASK_EXPOSURE.json](D_CROSS_TASK_EXPOSURE.json) | N3의 과거 TRAIN 노출과 D의 C VAL253/TEST271 노출 공개 |
| §1.3 모든 모델 공통 split | D_CROSS_TASK_EXPOSURE | **FAIL / IMPLEMENTATION_OR_CONTRACT_ISSUE**. C/D 통합 독립 TEST·동등 학습 모집단 상대 우위는 BLOCKED_CONTRACT |
| §2 GPU/CPU/fits/updates/PnP | 원·최종 COMPUTE receipts, 고정 lock, 호출 ledger | fitting8+2·2000updates 유지; 실패 호출 포함·50,000 상한; 동결 재추론6144회 |
| §3 A1 주장 대조 | [CLAIM_AUDIT.json](CLAIM_AUDIT.json), CLAIM_AUDIT.md | source SHA·정의·분모·원값·상태 포함. 보호 자료 과거 공개 수치는 새 원행 검산과 구분 |
| §3 A1 margin·코너·선택 연결 | [A1_PAIRED_MARGIN_RESIDUAL_ROWS.jsonl.gz](A1_PAIRED_MARGIN_RESIDUAL_ROWS.jsonl.gz) | 173×4경로×3seed=2076행; GT margin은 runtime confidence가 아님 |
| §3 A2 세 oracle·현재 선택 | [A2_ORACLE_METRICS.json](A2_ORACLE_METRICS.json), [A2_ORACLE_ROWS.jsonl.gz](A2_ORACLE_ROWS.jsonl.gz) | 실제5arm·20760행; T/R/성공 최적 oracle 분리·UNCLEAR 포함 |
| §3 A3 exact·correlated residual·Gaussian | [A3_GEOMETRY_DIAGNOSTIC.json](A3_GEOMETRY_DIAGNOSTIC.json) | 고정16×α4·σ5×8draw; 실제5920호출; raw C1 signed-front180° 실패2/16 |
| §3 A3 canonical physical gate | [A3_PHYSICAL_ROUNDTRIP_AUDIT.json](A3_PHYSICAL_ROUNDTRIP_AUDIT.json) | **FAIL**. raw R/성공을 노이즈만의 한계로 해석 중단; 허용 대칭 수정0 |
| §3 A3 두 실제 비율·square | [A3_RATIO_AXIS_CORRECTION.json](A3_RATIO_AXIS_CORRECTION.json) | 원 wood 축 뒤집힘 공개·보존. 같은 K/R/t/H/area 등록 W/D projection-only48행 보완; 새6D NA/PnP0 |
| §4 B1 동일 N3 입력 치수 | [B1_RESULTS.json](B1_RESULTS.json), [B1_PERMUTATION_REPAIR_RESULTS.json](B1_PERMUTATION_REPAIR_RESULTS.json) | original CORRECT/MEAN/ZERO/해시donor 보존. 해시 중복172/173을 순열로 부르지 않음 |
| §4 B1 실제 고정 순열 | [B1_PERMUTATION_REPAIR_LOCK.json](B1_PERMUTATION_REPAIR_LOCK.json), [B1_PERMUTATION_REPAIR_DONOR_SEAL.json](B1_PERMUTATION_REPAIR_DONOR_SEAL.json) | seed20261011·TRAIN donor173/중복0; 추가1038호출; probability/좌표/cap/선택/6D 평가; GT 전 prediction seal |
| §4 B1 N0/N1/N2/N3 seed 절제 | [B1_ABLATION_REUSE.json](B1_ABLATION_REUSE.json) | 기존2076 per-ID 파일 재사용, 새fit/forward/PnP0; 동일 checkpoint 개입과 구분 |
| §4 B2 대칭 감독 | [B2_SYMMETRY.json](B2_SYMMETRY.json) | YOLO125/55915·DOPE134/44063·RN0/55806; 고유frame/optimizer노출/tie/mask/group 구분 |
| §5 C1 레이블·predictor 입력 | [C_LABEL_AUDIT.json](C_LABEL_AUDIT.json), [C_B1_VERIFICATION.json](C_B1_VERIFICATION.json) | canonical registry만 D 입력·GT/source/session/material정답 제외; C1 physical gate 실패는 별도 제한 |
| §5 C2 cache·분할 | [C_EXTRACTION_RECEIPT.json](C_EXTRACTION_RECEIPT.json) | 현재 frozen checkpoint P3/P4 공간cache 최대차0; 새고비용 REAL RGB추출0 |
| §5 C3 고정8arm | [C_FITS.json](C_FITS.json) | 8fit 1회씩·sweep0. 같은 크기 G/GV/GVD·zero 슬롯·학습 조건 유지 |
| §5 C4 선택·후보 | [C_FRAME_METRICS.jsonl.gz](C_FRAME_METRICS.jsonl.gz) | 5190행·후보/코너 불변·p=.5·tie/fallback·TAGGED 물리 순서 불변. 이름 없는 G슬롯교환 불변 구조는 없음 |
| §5 C5 실제6D·하위군 | [C_RESULTS.json](C_RESULTS.json), [C5_SUBGROUPS_SUPPLEMENT.json](C5_SUBGROUPS_SUPPLEMENT.json) | 전173·교집합110·가림/고도/margin/material/session/type 모두 보고. 등록type128/45. 크기 구간은 사전 경계 없음으로 NA |
| §5 C5 합성 고도 범위 | [C5_CONTRACT_DETAILS_SUPPLEMENT.json](C5_CONTRACT_DETAILS_SUPPLEMENT.json) | TEST≥30°0으로 해당 구간 NA; repartition/REAL 기반 선택 없음 |
| §6 D 국소 결합·matched control | [D_CONTRACT.json](D_CONTRACT.json), [D_TRAINING_RECEIPT.json](D_TRAINING_RECEIPT.json) | 두2305parameter·동일초기화/order/loss·각1000step·frozen gradient/weight 및 step0 parity |
| §6 D Stage1/SubPix 실제6D | [D_REPAIRED_ROWS.jsonl.gz](D_REPAIRED_ROWS.jsonl.gz), [D_REPAIRED_RESULTS.json](D_REPAIRED_RESULTS.json) | 10경로×173=1730행. K adapter 최초오류6055calls 보존·동일코너 복구2440calls 포함 |
| §6 D 치수·추가 학습 증분 분리 | [COMPARISON_TRANSITIONS_COMPLETE.csv](COMPARISON_TRANSITIONS_COMPLETE.csv) | Stage1 OLD_P78/CONTROL81/JOINT77; SP85/86/85. mean2/shuffle3 wrong-axis 손상ID 포함 |
| §7 E 보류·복수 가설 | [E_RISK_COVERAGE.json](E_RISK_COVERAGE.json), [E_PROBABILITY_RISK.json](E_PROBABILITY_RISK.json) | 24+27곡선·전체threshold·동률·50/80/90/95/100%·risk/AURC/ROC/전체성공·random 기준선 |
| §7 E runtime threshold | E_PROBABILITY_RISK | VAL-only 고정값 REAL1회 전이. GT threshold 선택0·안전 보장 주장0; YOLO두후보 모두유효/미산출0·UNCLEAR3 |
| §8 F 시간 계약 | [F_G_INPUT_AUDIT.json](F_G_INPUT_AUDIT.json) | 연속8910장 실제존재; motion/association/no-face-switch 독립 계약 없음→BLOCKED_CONTRACT. 임의binary vote·미래GT 사용0 |
| §9 G1 독립 floor | F_G_INPUT_AUDIT | target-matched 측정 calibration0·floor-contact UNKNOWN→BLOCKED_CALIBRATION |
| §9 G2 평면 OFAT | [G2_PLANE_SENSITIVITY.json](G2_PLANE_SENSITIVITY.json) | SYNTH64/REAL GT-peer61×3seed·7500행·40조건·all signs. GT FEASIBILITY_ONLY·zero조건선택불변; 새PnP0 |
| §9 G3 depth | [G3_DEPTH_GATE_REUSE.json](G3_DEPTH_GATE_REUSE.json), [G3_DEPTH_BIAS_ROWS.jsonl.gz](G3_DEPTH_BIAS_ROWS.jsonl.gz) | 기존231장 median17.179% gateFAIL 유지. 허용85×3bias join; 새depth추론/보정/S4policy0 |
| §10 지정4대비·8가설 | [REQUESTED_PRIMARY_PAIRED.json](REQUESTED_PRIMARY_PAIRED.json) | 지정Stage1 포함 전부 계산. 원lock은SP라 **사전등록 일치 FAIL**; 실행후 탐색 정렬·원family 보존, 소급등록 없음 |
| §10 seed·session·95%CI·Holm | [BOOTSTRAP_LOCK.json](BOOTSTRAP_LOCK.json), [SUPPLEMENT_INDEPENDENT_VERIFICATION.json](SUPPLEMENT_INDEPENDENT_VERIFICATION.json) | 9실제세션×10000paireddraw·seed는frame안, 원p/Holm·영상가중/세션동일가중 모두 보존 |
| §11 회복·손상·미산출 | 완전 C/D8131행, [CHANGED_B1_CANDIDATES_COMPLETE.csv](CHANGED_B1_CANDIDATES_COMPLETE.csv), [OLD_P_CHANGED_CANDIDATES_COMPLETE.csv](OLD_P_CHANGED_CANDIDATES_COMPLETE.csv) | 모든ID·양후보 포함. 원B1개입10+순열4전환/84calls, OLD_P15전환/12ID/72calls를 GT없이복구. unchanged OLD_P 미보존값만 NA. 원D front_depth 열은centroidZ로 정정 |
| §11 필수9산출물·그림 | [RESULTS_KO.md](RESULTS_KO.md), [REPRODUCE.md](REPRODUCE.md), 아래 publication proof | 원그림9개 보존+지정Stage1그림1개=PNG10. RGB/원참조/privatepath/대형weight 공개0 |
| §11 재현·잠금 | [C_CLEAN_REPLAY_PREPARATION.json](C_CLEAN_REPLAY_PREPARATION.json) | 새helper로현재codehash의clean guards 실제검증·추가fit0. 과거 수동 lock 생성의CLI누락을 보완; 전체추론을 다시한것은 아님 |
| §11 Git 정상 게시 | [COMPLETION_PREPUBLICATION_AUDIT.json](COMPLETION_PREPUBLICATION_AUDIT.json), 원격 후속receipt | main normalFF·force/자동merge0·기존449변경 보존·원격SHA/HTTP파일hash 검증 |

[확인] 최초 보고에서 누락·오표기된 부분은 실행/코드 오류의 위치를 남겨 보완했다. 최초 수치·checkpoint·과거 SHA는 유지했다. C/D 공통 split과 Stage1 사전등록을 사후 편집으로 복구할 수는 없으므로 해당 제한은 그대로 남긴다. [추정] 다음 연구는 C1 물리축 round-trip를 별도 승인 범위에서 고친 뒤 독립 조건으로 확인해야 하며, 이 pilot의 작은 개선 방향을 배포 결정 근거로 쓰지 않는다.
