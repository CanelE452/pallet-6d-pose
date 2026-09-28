# A–F 최소 대조 후보

문헌 검토2026-09-28, 최종 실행 상태 갱신2026-09-29. A/B/C 주cycle, 명목 seed43 재실행4fit 및 Wood 적용2fit이 모두 완료됐다. **12fit/3,840update를 사용했으나 유효한 학습 난수 변화 재현은 미완료**다. [기존 선택 기록](FINAL_SELECTION.json)은 당시 pending 상태 그대로 보존하고, 후속 [재현 유효성 정정](REPLICATION_VALIDITY_CORRECTION.md) 및 [독립 최종 해석](DISCUSSION_REVIEW.md)을 함께 읽는다. C는 재현 대상으로만 선택됐고 최종 승격하지 않는다. 이 문서는 새 fit을 승인하지 않는다. 미실행 D/E/F는 실패가 아니다.

공통 계약: Plastic 자연 Moderate+Severe99frame가 기본 주 발견 집합, 전체128와 severity/recording/Clean 손익 필수. Wood45는 별도 적용성이고 Severe는 NA. 같은 centroid/대칭/단위·공통 pose solver에서 T/R median/P90·coverage·paired 변화 확인. 차이는 NEW minus BASE이며 음수가 개선이다. NEW_REF−NEW_RAW, NEW_REF−OLD_REF, NEW_REF−R0를 분리한다. 반복 DEV 참조는 학습·mask·weight·membership 결정에 쓰지 않는다. 마지막4fit 상당은 대응 control/추가 seed 재현을 우선 예약한다.

| ID | 완료한 최소 진단 | 주cycle 후 disposition | 실제/예약 비용과 제한 |
| --- | --- | --- | --- |
| A |64real/64source preflight, 실제320 paired batch 및3,000 합성 계획 검사 | `COMPLETED_TRADEOFF_NOT_SELECTED`: OLD_REF 대비 T↓/R↑, matched NEW_RAW 대비만 두 median↓ | 실제2fit/640update; 반복 대상으로 미선택 |
| B | 실제 설치 criterion·assignment·head gradient, finite difference64/64, ignored gradient0, TRAIN 계수 고정 | `COMPLETED_NO_GAIN_NOT_SELECTED`: OLD_REF 대비 두 median↑; 근본 실패 원인 확정 아님 | 실제2fit/640; 큰 이득이 없어 추가 gradient-scale control 미실행 |
| C | A의 실제 가림/감독 노출량, covered 위치 신호, 빈도변경 preflight 및320 paired batch | `COMPLETED_NOT_PROMOTED`; effective repeat=`NOT_RUN_EFFECTIVE_TRAINING_VARIATION`. Plastic OLD_REF 대비만 미세 joint sign, R0/matched RAW 대비 둘 다↑; Wood 별도 절충 | 주cycle2fit/640; 명목 seed43 재실행4fit/1,280 + Wood 적용2fit/640 실제 완료. 재실행 비용도 모두 계상 |
| D | trainable branch/ignore 및 실제 graph, R3 분류와 R4 GHMR 수식·공식 구현 분리 | detector Focal `NOT_APPLICABLE_CURRENT_OBJECTIVE`; kobj Focal/GHMR `NOT_RUN_LOWER_PRIORITY`, 방법 실패 아님 | 실제fit0; 단순 좌표 B·노출 C·재현/control 시도 우선, 이후fit 상한 도달 |
| E | 실제 bilinear 함수의 CPU mass/경계/ignore gradient 검사6fixture | `CPU_SANITY_DONE_NOT_RUN_STUDENT_BUDGET_PRIORITY`; teacher/학생 품질 미측정 | 실제fit0; teacher 두 대조+학생 전달은 최소4fit/1,240 역사적 추정으로 이번에 예약하지 않음 |
| F | 기존 Replay 비용/입력계약, SHA 고정 실제 가림 TRAIN5쌍 CPU 검사 | `REUSED_ASSET_FEASIBLE_NOT_RUN_CURRENT_TRANSFER_BUDGET_PRIORITY`; 현재 교사/학생 품질 미측정 | 실제fit0/신규추론0; 최소4fit/1,240 + cache/probe보다 C의 대응 재현과 Wood 적용 우선 |

## 실행 판정의 정확한 의미

주집합99의 수치는 동일한 기준으로 `NEW_REF − 비교 arm`을 계산했다. ΔT는cm, ΔR은full-rotation degree이며 음수가 개선이다. 아래는 **median의 차이**다. matched frame 차이의 median이나 PCK/AUC로 선택 통계를 바꾸지 않는다.

| Cycle | NEW_REF T/R median | ΔT/ΔR vs OLD_REF | vs matched NEW_RAW | vs R0 | 현재 판단 |
| --- | --- | --- | --- | --- | --- |
| [A](cycles/A_INPUT_OCCLUSION/RESULTS_PLASTIC_S42.json) | 12.255720 / 8.730257 | −0.167291 / +0.035561 | −0.462948 / −0.276132 | +0.107798 / +2.692313 | TRADEOFF |
| [B](cycles/B_COORDINATE_SUPPLEMENT/RESULTS_PLASTIC_S42.json) | 12.430253 / 8.704029 | +0.007242 / +0.009333 | −0.037415 / −0.357588 | +0.282331 / +2.666084 | NO_GAIN |
| [C](cycles/C_EXPOSURE/RESULTS_PLASTIC_S42.json) | 12.387281 / 8.682595 | −0.035730 / −0.012101 | +0.064303 / +0.399305 | +0.239359 / +2.644651 | 부호상 유일한 OLD_REF joint 후보; 재현 전 승격 금지 |

C의 OLD_REF 대비 차이는 약0.36mm/0.012°에 불과하고 학습·reference 변동일 수 있다. 사전 규칙의 joint sign으로 **재현 대상을 고른 것**이지 실용 이득·R0 초과·독립 검증을 선언한 것이 아니다. 모든 결과는 재사용 DEV이며 처음부터 같은 사전학습 R0에서 시작했다. 최종 재실행에서 숫자는 정확히 같았으나 실제 loader 난수도 같아 변동성을 시험하지 못했다.

세 주cycle 실제6fit/1,920update 이후, [baseline repeat](cycles/BASELINE_REPEAT/PROTOCOL.json)와 [recipe repeat](cycles/RECIPE_REPEAT/PROTOCOL.json)에 trainer 설정 seed43 RAW/REF 네fit을 수행했다. 설정43은 실제 로그로 확인했지만 고정된 DataLoader generator와 worker seed 때문에 C42/43 complete trace 및879 tensor가 같았다. baseline42/43도879 tensor가 같았다. 설정 변경을 실제 stream 변화로 간주하고 fit 전에 검사하지 못한 실행 검증 누락이며, 결정론적 재실행을 독립 반복으로 인정하지 않는다. 과거 순서-only 반복과 다른 설정을 의도했더라도 이번 의도는 달성하지 못했다.

[Wood 적용](cycles/WOOD_APPLICABILITY/RESULTS_WOOD_S42.json)은 seed42 RAW/REF 두fit 완료다. 총 실제12fit/3,840update, teacher fit0, 신규 수동주석0이며 명목 반복4fit/1,280update도 비용에 모두 포함한다. 추가 corrective fit이나 재시작은 하지 않는다. 유효 반복의 미완료를 남긴 실행 해석은 `PARTIAL_BUDGET`이고 성능 승격은 없다.

아래 세부 카드는 사전 설계의 질문·반론·중단 조건을 보존하고 실행 후 disposition을 붙였다. 미실행 D/E/F는 이번 제한된 batch의 우선순위 결정이지 일반적 반증이 아니다.

## A — FILTERED_INPUT_TO_OCCLUDED

- 질문/출처: [R1 §3–4.1](https://arxiv.org/html/2011.12498v2), [R2 §3](https://arxiv.org/html/2303.04346v1)의 원본/easy 감독과 어려운 입력 분리. 방법절을 읽었고 공식README는 확인했지만 전체 알고리즘 재현은 아니다.
- 다른 전제: 현217/361 pool은 filtered unlabeled이지 verified CLEAN이 아니다. 고정teacher는 R1의 coupled heatmap collapse와 다르고, noise fill은 R2의 다른 이미지 limb paste가 아니다.
- 기존과 차이: 과거Clean19/87support·전층기본학습·lr1e−4·300평가의 S1/S2가 아니라 현재 main pool·pose/flow-only·lr1e−5에서 RAW/REF를 비교한다. C2 affineOFF와도 다르며 기본 affine/HSV/source는 유지한다.
- 필요정보/변경축: frozen 원본 targets/support와 predicted bbox, RGB만 필요. 기존 기본증강 후 RGB rectangle overlay만 추가. target/v는 덮어쓰지 않는다. 계획은 양 arm 공통 REF transformed corner에 조건부이며 RAW도 이 teacher 정보를 계획에 사용한다고 명시한다.
- 실제 구현 검토: [`occlusion.py`](../../../scripts/research/pallet_pose_objective_followup_v2/occlusion.py)는 reference dryrun의 RGB/bbox equality 확인 후 RNG를 원래 transform 종료상태로 돌린다. shape/fill의 기존 분포를 재사용하되 S2 edge/pair 조건은 제거한다. random 최대32개 첫 유효 위치, covered corner≥1/remaining≥2(따라서 최소3), center는 계획에서만 제외한다. 기존 transformed-native-canvas 한정과 달리 전체 model canvas를 사용한다. 이는 공개해야 할 sampling-domain 차이지 oldS1 bit-exact 재사용이 아니다.
- 최소 진단: [새 preflight](OCCLUSION_PREFLIGHT_PLASTIC.json)의64source/64real 검사. parent 실행기록상 RGB/bbox/RNG/타깃 보존PASS, real14/64 적용/REF22점 coverage, 기존 RAW/REF affine후 support차이1회 유지. 이는 전체320batch 보장을 대신하지 않는다. 실제 masked/remaining 감독과 pixel 예시를 fit trace로 확인한다.
- control/단순대안: 같은320update NEW_RAW_OCC/NEW_REF_OCC. 기존 전체pool RAW/REF의 기본변환·RNG parity가 확인되면 baseline 재fit을 줄인다. 새subset·support가 생기면 matched baseline이 필요하다. 복잡한 구조 mask보다 현재 random-placement 대조가 먼저다.
- 강한 반론: 가림이 이미 noisy한 target을 더 어려운 입력에 강제해 오류를 확대할 수 있다. rectangle은 자기가림/시점변화/실제occluder를 대변하지 않는다. 같은 mask여도 RAW/REF target 위치가 달라 covered count가 다를 수 있다.
- TRAIN/DEV·예산·중단: TRAIN은 적용률·가린점 gradient/손실/타깃추종, DEV는 자연가림 T/R. 2fit/640update. 참조누출·좌표/mask/RNG 불일치·비수치면 기술중단/기록. 적용이0이면 학습 전 feasibility 재판단; 미적용 occurrence는 삭제하지 않는다.
- 다음 결정: mask와 gradient가 실제 바뀌었는데 무차이/악화면 A 음성결과 보존 후 B/C의 새 근거로 이동. joint T/R 후보면 source/Clean/tails·R0비교를 보존하고 matched additional-seed 재현 또는 독립된 두번째 원리와 결합을 검토한다.
- 실행 후: [전수 입력 감사](AUDIT_A_INPUT_OCCLUSION.md)에서 real2,560회 중542회(21.17%) 적용, REF824/RAW781 감독점 가림, 원래 affine support 차이3batch를 확인했다. [A 보고서](cycles/A_INPUT_OCCLUSION/REPORT_KO.md)의 OLD_REF T/R tradeoff로 선택하지 않았다. 입력 변화와 국소 gradient가 존재한다는 것만으로 자연 가림 T/R 개선이 보장되지 않았다.

## B — 큰 오차 좌표항

- 질문/출처: [R5 Wing의 L1/SmoothL1 비교](https://arxiv.org/html/1711.06753v5), [R7 RLE §3](https://arxiv.org/html/2107.11291v3), [R12](https://github.com/ultralytics/ultralytics/blob/v8.4.60/ultralytics/utils/loss.py). 관련방법/로컬criterion을 읽었다. 현재 학생은 L2-only가 아니다.
- 기존과 차이/필요정보: 과거C1 Huber는 frozen pose 선택점수이지 학습loss 대조가 아니다. C3 큰오차는 stored-index TRAIN이며 physical identity를 확증하지 않는다. 최고score 배포검출과 실제assigner 출력부터 연결해야 한다. DEV는 사용하지 않는다.
- 최소 진단: [LOSS_SIGNAL_AUDIT.md](LOSS_SIGNAL_AUDIT.md)의 실제 설치 graph를 확인했다. 선택한 MAIN8/8 branch-context에서 RLE aggregate clamp가 켜져 RLE gradient0이었지만 location 신호는 남았다. C3의 문제 corner4에서 location 감쇠가 컸으나 RLE/combined 신호가 남았고 실제 표준화±100 clip은0건이었다. 실제 fixed-assignment finite difference64/64, leaf descent16/16 및 ignored gradient0을 통과했다. 이 소표본 진단을 전체TRAIN 빈도나 원인 확정으로 일반화하지 않는다.
- 단순대안/변경축: 포화 근거가 있으면 기존 supervised에만 detached bbox 정규화 L1 또는 SmoothL1 한 항을 추가. transition/weight는 TRAIN 단위와 gradient 규모에서 한 값 고정. sigma/clip/LR/학습부를 동시에 바꾸지 않는다. ignored 모든항 gradient0.
- control/반론: 동일recipe RAW/REF2fit/640update. 큰pseudo 잔차는 label noise일 수 있으며 추가항이 단순 effective scale로 작동할 수 있다. 큰 개선이면 동일gradient총배율 대조에 재현예산을 먼저 쓴다.
- TRAIN/DEV·중단·다음: TRAIN 신호/target추종과 DEV T/R을 분리한다. NaN/Inf는 보존된 기술실패. gradient가 이미 정상이라면 B를 원인으로 채택하지 않고 C노출/모델대응을 본다. 정상화가확인돼도 T/R무이득이면 성능성공으로 쓰지 않는다.
- 실행 후: [B SPEC](cycles/B_COORDINATE_SUPPLEMENT/SPEC.md)대로 z=(pred−target)/sqrt((2σ)²·detached area·2)의 각 좌표 SmoothL1(β1)을 추가했다. λ0.16500387762358062는 MAIN_R0 clear TRAIN probe에서 기존 weighted location head norm의10%로 고정했다. A 가림은 끄고 다른 기본 recipe는 보존했다. 2fit 결과 OLD_REF 대비 T/R 모두 소폭 악화되어 추가 loss를 채택하지 않았다. 전체 실패가 gradient 포화 때문이라는 가설의 인과 증명이 아니며, 큰 성능 이득이 없으므로 추가 gradient-scale matched fit을 사용하지 않았다.

## C — 노출/기여 균형

- 질문/출처: [R5 §5](https://arxiv.org/html/1711.06753v5), [R11 §4](https://arxiv.org/html/1604.03540v1). 원래 GT pose-bin balancing/annotated RoI mining과 현pseudo를 구분한다.
- 기존과 차이: source/real 고정512슬롯은 point/gradient 균형을 보장하지 않는다. oldS1/S2는 masked1695/2653으로 달랐다. 현재pool의 자연난도는 확인된metadata가 없으면 UNKNOWN이지 CLEAN이 아니다.
- 최소진단/필요정보: C0 uniform의 source/real·covered/remaining supervised수, assignment수, loss분모·gradient합 및 unique영상/recording multiplicity. generated mask가 알려진 그룹이며 pseudo잔차는 GT정오답 그룹이 아니다.
- 단순대안/변경축/control: 같은영상·source/real비·총update에서 masked 노출 또는 capped weight 하나만 조정한다. uniform matched RAW/REF가 control. 1:1은 가설일 뿐 필수비율 아님; 빈그룹을 조작해 채우지 않는다. total weight/denominator와 전체배율을 기록한다.
- 강한반론/역할: 어려움 oversampling은 noisy labels의 반복일 수 있고 적은recording에 과적합할 수 있다. 생성가림 빈도변경은 정확/부정확 타깃균형이 아니다. TRAIN 균형개선만으로 DEV T/R성공을 선언하지 않는다.
- 예산/중단/다음: 채택시RAW/REF2fit/640update와같은uniform대조. 유효그룹0·누출·기여량불명은 그branch를보류. 관측불균형이없으면 `LOWER_PRIORITY`; A에서covered점 기여부족이실제관측되면 한축개입근거가된다.
- 실행 후: [C SPEC](cycles/C_EXPOSURE/SPEC.md)은 A의 schedule probability0.5→1만 바꿨다. shape/aspect/fill/32회 위치후보, 원래 gate draw 소비 및 기존 적용 입력의 동일성, source/목록/target/손실/update를 보존했다. 실제 적용1,054/2,560=41.17%, REF masked1,559(총v2 21,823의7.14%)/RAW1,470으로 증가했다. A의 REF824점(3.78%)보다 노출이 늘었지만 100% 적용은 아니며 GT 정오답1:1 또는 자연 난도 균형이 아니다. C_REF−A_REF의 주 T/R median은 +0.131561cm/−0.047662°로 역시 절충이다. OLD_REF 대비 아주 작은 joint sign만으로 유일한 재현 대상으로 선택했고 현재 R0·matched RAW 초과 주장은 하지 않는다.
- 최종 확인: 명목 seed43 recipe−동일43 baseline 수치는 seed42와 동일하지만 무효 난수 개입이므로 재현 성공이 아니다. Plastic C_REF−OLD_REF의 paired median ΔR은 +0.019063°이고 R은31frame 개선/68악화; recording 하나 제외 시 joint 부호가 사라질 수 있다. Wood45 pooled OLD_REF 대비 ΔT/ΔR=−0.204365cm/−0.011585°, matched RAW 대비−0.150489/−0.039785지만 R0 대비−0.228069/+0.048722로 절충이다. Wood Clean38/Moderate7 각각 rotation median은 악화했고 Severe0은NA다. [독립 검토](DISCUSSION_REVIEW.md)는 각 통계와 모집단을 분리해 `NOT_PROMOTED`를 지지한다.

## D — Focal과 GHM을 분리

- 질문/출처: [R3 Focal §3](https://arxiv.org/html/1708.02002v2), [R4 GHM 방법](https://arxiv.org/pdf/1811.05181), [저자 GHMR 소스 전체](https://raw.githubusercontent.com/libuyu/mmdetection/master/mmdet/models/losses/ghm_loss.py)를 읽었다.
- 최소진단: trainable inventory에서 detector분류/backbone이동결임을 확인. kobj학습은있지만 v1은제외되므로 dense-background imbalance와같지않다. 실제kobj pos/neg/ignore·gradient기여는B진단에서확인한다.
- 판정/변경축: detector Focal은 `NOT_APPLICABLE_CURRENT_OBJECTIVE`. kobj Focal은 `NOT_RUN_LOWER_PRIORITY`이며 coordinate residual을p_t로넣지않는다. GHMR은 회귀용ASL1의detached 출력미분밀도이며 전체parameter norm이아니다. 향후 별도 분포진단이강하면 uniformASL1 vsGHMR의한축weighting대조가필요하다.
- 필요정보/단순대안/강한반론: TRAIN만으로 noise가능성을나누고 먼저C의uniform/cappedweight를검토. GHMR μ의단위·bin·EMA·valid/occupied-bin분모를고정해야하며 코드기본momentum0과논문.75는같지않다. 드문오답bin이증폭될수있어 outlier에항상최고비중을주는방법으로쓰지않는다.
- control/역할/예산: 같은RAW/REF, 같은base회귀항·총배율control. 최소2fit/640외필요control을예약하지못하면`NOT_RUN_BUDGET`. TRAIN histogram균등화는성능증거아니며DEV주T/R로판정. emptybin/nonfinite는안전처리·기록,유효근거없으면미실행유지.
- 실행 후 disposition: detector branch가 동결이므로 그 분류 Focal은 이번 목적의 변경축이 아니다. kobj Focal/GHMR은 원문과 branch/ignore/실제 loss 신호를 검토했으나 구현·fit하지 않았다. RLE aggregate clamp 관측은 GHMR density imbalance의 증거가 아니고 단순 보조 회귀 B도 T/R 무이득이었다. 이 조건에서 새 ASL1 family·bin/EMA·scale control을 추가하기보다 관측된 가림 노출 C와 대응 재현을 우선했다. `NOT_RUN_LOWER_PRIORITY`이며 GHMR 성능 실패가 아니다.

## E — refiner의 soft target distribution

- 질문/출처: [R6 AWing §4](https://arxiv.org/html/1904.07399v3), [R8 §3](https://arxiv.org/html/2310.00099v1), 현재prior_model. 방법절은읽었으나AWing공식training loss소스는확보못했다. 이카드는AWing재현이아니다.
- 기존과차이/필요정보: 입력Gaussian σ9crop-px/peak255와현재bilinear target최대4cell은다르다. output72×96, crop288×384, ratio4. 현재CE+coordL1+L2를보존해target만동일중심normalizedGaussian으로비교한다. 새studentheatmaphead는추가하지않는다.
- 최소진단: 실제bilinear함수+σ1grid 수학fixture6개 CPU검사완료([RELATED_WORK](RELATED_WORK.md)). mass·ignoregradient0·boundarybias확인. σ1은예시이고fit값아님. native폭은실제cropmatrix로연결해야한다. Gaussian예상좌표가경계에서중심과달라져coordinateL1와상충가능.
- 단순대안/변경축/control: 같은teacher시작·추가step·TRAIN·입력·loss의bilinear vsGaussian. AWing이나weightedmap동시변경금지. currentbilinear+coordL1가가장단순control이며원래있던softsupervision을새발명으로쓰지않는다.
- 강한반론/역할: 틀린중심을넓혀도중심오류는수정되지않는다. AWing은peak1intensity회귀로CE대체와다르다. teacherTRAIN감소가학생T/R개선은아니다.
- 예산/중단/다음: 역사적teacher300step을가정하면2teacherfit/600,전달까지2studentfit/640추가;정확cost는예약후확정. 전체경로부족하면`REFINER_ONLY_SCREEN/STUDENT_POSE_UNTESTED`로제한. 경계/ignore불일치면fit전수정명세;실험결과미확인상태로실패기록하지않는다.
- 실행 후 disposition: CPU 수학 sanity만 완료했다. Gaussian 분포에 대한 teacher fit이나 학생 전달은 하지 않았고 AWing의 실험 실패를 주장하지 않는다. 서로 다른 teacher 분포 대조와 학생 전달의 최소4fit 경로보다 현재 C의 baseline+recipe 재현4fit 및 Wood 적용을 우선하여 `NOT_RUN_STUDENT_BUDGET_PRIORITY`다.

## F — 실제 가림 초기점에 대한 teacher 적응

- 질문/출처: [R9 PoseFix](https://arxiv.org/html/1812.03595v2)의기존방법검토재사용+[R1/R2/R8]방법과현Replay train코드. 원논문성공과팔레트결과를분리한다.
- 기존과차이/필요정보: 단순정답근처독립noise만아니라 `I_occ`에서frozenR0가만든`q_init_occ,box_occ`를쓴다. `y_target`은원본의적격S1/S2/P. noGTcrop, U/center추가감독없음. 원영상정확hint를그대로넣으면이질문을검사하지못한다.
- 최소 진단: 기존300step/real8+source8/micro2/TFAdam1e−4/allteachertrainable/BNstatsfrozen 및 실측184.564s를 확인했다. [F 최소 CPU 진단](F_MINIMUM_DIAGNOSTIC.md)은 과거 저장된 실제 가림 TRAIN5쌍의 SHA·RGB 변화·실제 OCC-R0 초기점/예측bbox·crop matrix·공통support·원본 target roundtrip을 확인했다(fit0/신규추론0). 과거 DAY264는 현재217과 교집합0, target은 Replay+PnP 보완, 초기 teacher는 synthetic-only여서 현재 F의 품질 증거가 아니다. 현재 pool의 새 가림 결측률/source 유지/학생 전달은 미측정이다. 기존9/38은 clean-only가 아닌 teacher-exposed stored-index이며 physical-axis 확정GT가 아니다.
- 변경축/control/단순대안: 같은현재startingteacher의T_KEEP/T_OCC를같은300추가step으로적응(예시budget,최종SPEC필수). 기존frozen도참고선. 그뒤두teacher를freeze하고같은realmembership/common support로각학생320. frozen교사유지가가장싼대안이고교사추가학습량control생략불가.
- 강한반론: 가림detector의초기점/bbox실패가teacher에재입력되므로support가변한다. 적은9/38에과적합하거나syntheticprior를손상할수있다. source유지/teacher품질은학생자세성공과별도다.
- 역할/예산/중단/다음: 최소4fit/1,240update+캐시추론/probe;학생320을teacherstep으로복사하지않는다. 여기에재현reserve까지확보하지못하면`LOWER_PRIORITY/NOT_RUN_BUDGET`. GTcrop필요·공통support고갈·identity모순은해당경로만보류. 최종student-only RGB+solver를유지하고teacher-only향상을최종성과로부르지않는다.
- 실행 후 disposition: 실제 가림에서 얻은 초기점/box를 넣는 경로는 저장5쌍 CPU 검사에서 실행 가능했다. 그러나 다른 DAY264, Replay+PnP 보완 target, synthetic-only teacher 시작점은 현재 적응·전달 대조가 아니다. 새 T_KEEP/T_OCC 두teacher 및 두student의 최소4fit을 시작하지 않았고 teacher 추가학습량 대조를 생략해 fit 수를 줄이지도 않았다. 주cycle3개 이후 예약된 대응 재현/control을 우선하여 `NOT_RUN_CURRENT_TRANSFER_BUDGET_PRIORITY`; F의 실제 정확도나 학생전달 실패로 기록하지 않는다.

## 왜 A+B 또는 B+C를 결합하지 않았는가

두 독립 원리의 유효성이 확인되지 않았다. A는 OLD_REF 대비 T/R tradeoff, B는 둘 다 악화였고, C는 A의 가림 노출 빈도를 바꾼 같은 입력 원리의 최소 대조다. C의 작은 OLD_REF joint sign이 B의 무이득을 구제하지 않는다. 따라서 A+B/B+C의 결합으로 원인과 단독효과를 섞거나 실패한 loss를 되살리는 fit을 하지 않았다. 마지막 주cycle은 TRAIN에서 확인한 작은 실제 가림 노출을 늘리는 C로 썼다.

새 방법·강도·계수·threshold 탐색을 닫고 예약된 모든 fit/eval을 완료했다. 명목 seed43 재실행에서 수치는 같았지만 실제 난수 변화가 없어 독립 반복은 미완료로 정정했다. Wood의 별도 적용성은 이를 대신하지 않는다. 사전 선택 기준을 PCK/AUC나 cm+degree 임의합으로 바꾸지 않으며, C를 최종 승격하지 않는다. 기존 결과·비용·선택 당시 기록과 정정 이력을 모두 보존하고 추가 fit 없이12fit 상한에서 종료한다.
