# A–F 최소 대조 후보

검토일2026-09-28. 문헌·코드 조사와 학습 실행을 구분한다. 초기 fit 상태는 모두 `PENDING_FIT_SELECTION_OR_EXECUTION`이며 실제 cycle SPEC/RESULTS가 최종 실행 상태의 근거다. 이 문서는 새 fit을 승인하거나 예약하지 않는다. parent가 실제 진단·예산·사전 T/R 규칙으로 선택한다. 후보 미실행은 실패가 아니다.

공통 계약: Plastic 자연 Moderate+Severe99frame가 기본 주 발견 집합, 전체128와 severity/recording/Clean 손익 필수. Wood45는 별도 적용성이고 Severe는 NA. 같은 centroid/대칭/단위·공통 pose solver에서 T/R median/P90·coverage·paired 변화 확인. 차이는 NEW minus BASE이며 음수가 개선이다. NEW_REF−NEW_RAW, NEW_REF−OLD_REF, NEW_REF−R0를 분리한다. 반복 DEV 참조는 학습·mask·weight·membership 결정에 쓰지 않는다. 마지막4fit 상당은 대응 control/추가 seed 재현을 우선 예약한다.

| ID | 질문·현재 최소 진단 | 초기 결정 | 최소 fit/control 비용 |
| --- | --- | --- | --- |
| A | 원본의 고정 pseudo를 가린 입력이 따라가는가? 기존 mask code/audit 재사용+새64real/64source preflight | `IMPLEMENTED_PREFLIGHT_PASS_FIT_PENDING`; 유력 첫 학생 대조 | Plastic RAW/REF 각320,2fit/640update; baseline 계약 동일하면 재사용 |
| B | 실제 assignment된 큰 잔차에서 좌표 신호가 포화/clip되는가? 실제 loss graph·finite difference | `RUNTIME_DIAGNOSIS_PENDING`; 단순 좌표항 후보 | RAW/REF 각320,2fit/640; 큰 이득 시 동일gradient-scale control 우선 |
| C | covered/remaining·source/real의 노출/유효 gradient가 불균형한가? 기존 노출과 새 A진단 | `EXPOSURE_DIAGNOSIS_PENDING`; uniform 우선 | 한 exposure 또는 weight축 RAW/REF 2fit/640; uniform matched control 필요 |
| D | 불균형의 대상이 분류인가 회귀인가? trainable branch/ignore 읽기+R4 공식수식 확인 | detector Focal `NOT_APPLICABLE_CURRENT_OBJECTIVE`; GHMR `CONDITIONAL_PENDING_DIAGNOSIS` | 실행시 RAW/REF 2fit/640 외에 같은base regression/uniform control 확보 |
| E | 기존 bilinear target를 같은 중심의 Gaussian으로 바꾸는가? CPU grid·경계·ignore 검사 | `CPU_DIAGNOSTIC_DONE_FIT_PENDING_SELECTION`; 단일분포 대조 | teacher2fit/600을 역사기반 가정; 학생전달까지 최소4fit/1,240, 실제SPEC로 재예약 |
| F | 실제 가림 R0 오류에 teacher를 적응시키는가? 기존 Replay 비용·입력계약 검사 | `CONTRACT_REVIEW_DONE_PROBE_PENDING`; 전체control 비용 미예약 | T_KEEP/T_OCC 각300 가정+학생각320=4fit/1,240; inference/cache/probe 별도 |

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

## B — 큰 오차 좌표항

- 질문/출처: [R5 Wing의 L1/SmoothL1 비교](https://arxiv.org/html/1711.06753v5), [R7 RLE §3](https://arxiv.org/html/2107.11291v3), [R12](https://github.com/ultralytics/ultralytics/blob/v8.4.60/ultralytics/utils/loss.py). 관련방법/로컬criterion을 읽었다. 현재 학생은 L2-only가 아니다.
- 기존과 차이/필요정보: 과거C1 Huber는 frozen pose 선택점수이지 학습loss 대조가 아니다. C3 큰오차는 stored-index TRAIN이며 physical identity를 확증하지 않는다. 최고score 배포검출과 실제assigner 출력부터 연결해야 한다. DEV는 사용하지 않는다.
- 최소 진단: location의 e/exp(−e), RLE scale·clip·aggregate clamp, 각branch/좌표/scale/head gradient·finite difference·descent방향·supervised/ignore수. [LOSS_SIGNAL_AUDIT.md](LOSS_SIGNAL_AUDIT.md)가 생성되면 그 실제graph 측정이 우선한다. 정적수식만으로 원인 확정 금지.
- 단순대안/변경축: 포화 근거가 있으면 기존 supervised에만 detached bbox 정규화 L1 또는 SmoothL1 한 항을 추가. transition/weight는 TRAIN 단위와 gradient 규모에서 한 값 고정. sigma/clip/LR/학습부를 동시에 바꾸지 않는다. ignored 모든항 gradient0.
- control/반론: 동일recipe RAW/REF2fit/640update. 큰pseudo 잔차는 label noise일 수 있으며 추가항이 단순 effective scale로 작동할 수 있다. 큰 개선이면 동일gradient총배율 대조에 재현예산을 먼저 쓴다.
- TRAIN/DEV·중단·다음: TRAIN 신호/target추종과 DEV T/R을 분리한다. NaN/Inf는 보존된 기술실패. gradient가 이미 정상이라면 B를 원인으로 채택하지 않고 C노출/모델대응을 본다. 정상화가확인돼도 T/R무이득이면 성능성공으로 쓰지 않는다.

## C — 노출/기여 균형

- 질문/출처: [R5 §5](https://arxiv.org/html/1711.06753v5), [R11 §4](https://arxiv.org/html/1604.03540v1). 원래 GT pose-bin balancing/annotated RoI mining과 현pseudo를 구분한다.
- 기존과 차이: source/real 고정512슬롯은 point/gradient 균형을 보장하지 않는다. oldS1/S2는 masked1695/2653으로 달랐다. 현재pool의 자연난도는 확인된metadata가 없으면 UNKNOWN이지 CLEAN이 아니다.
- 최소진단/필요정보: C0 uniform의 source/real·covered/remaining supervised수, assignment수, loss분모·gradient합 및 unique영상/recording multiplicity. generated mask가 알려진 그룹이며 pseudo잔차는 GT정오답 그룹이 아니다.
- 단순대안/변경축/control: 같은영상·source/real비·총update에서 masked 노출 또는 capped weight 하나만 조정한다. uniform matched RAW/REF가 control. 1:1은 가설일 뿐 필수비율 아님; 빈그룹을 조작해 채우지 않는다. total weight/denominator와 전체배율을 기록한다.
- 강한반론/역할: 어려움 oversampling은 noisy labels의 반복일 수 있고 적은recording에 과적합할 수 있다. 생성가림 빈도변경은 정확/부정확 타깃균형이 아니다. TRAIN 균형개선만으로 DEV T/R성공을 선언하지 않는다.
- 예산/중단/다음: 채택시RAW/REF2fit/640update와같은uniform대조. 유효그룹0·누출·기여량불명은 그branch를보류. 관측불균형이없으면 `LOWER_PRIORITY`; A에서covered점 기여부족이실제관측되면 한축개입근거가된다.

## D — Focal과 GHM을 분리

- 질문/출처: [R3 Focal §3](https://arxiv.org/html/1708.02002v2), [R4 GHM 방법](https://arxiv.org/pdf/1811.05181), [저자 GHMR 소스 전체](https://raw.githubusercontent.com/libuyu/mmdetection/master/mmdet/models/losses/ghm_loss.py)를 읽었다.
- 최소진단: trainable inventory에서 detector분류/backbone이동결임을 확인. kobj학습은있지만 v1은제외되므로 dense-background imbalance와같지않다. 실제kobj pos/neg/ignore·gradient기여는B진단에서확인한다.
- 판정/변경축: detector Focal은 `NOT_APPLICABLE_CURRENT_OBJECTIVE`. kobjFocal은 `LOWER_PRIORITY_PENDING_SIGNAL`, coordinate residual을p_t로넣지않는다. GHMR은 회귀용ASL1의detached 출력미분밀도이며 전체parameter norm이아니다. 분포진단이강하면 uniformASL1 vsGHMR의한축weighting대조가필요하다.
- 필요정보/단순대안/강한반론: TRAIN만으로 noise가능성을나누고 먼저C의uniform/cappedweight를검토. GHMR μ의단위·bin·EMA·valid/occupied-bin분모를고정해야하며 코드기본momentum0과논문.75는같지않다. 드문오답bin이증폭될수있어 outlier에항상최고비중을주는방법으로쓰지않는다.
- control/역할/예산: 같은RAW/REF, 같은base회귀항·총배율control. 최소2fit/640외필요control을예약하지못하면`NOT_RUN_BUDGET`. TRAIN histogram균등화는성능증거아니며DEV주T/R로판정. emptybin/nonfinite는안전처리·기록,유효근거없으면미실행유지.

## E — refiner의 soft target distribution

- 질문/출처: [R6 AWing §4](https://arxiv.org/html/1904.07399v3), [R8 §3](https://arxiv.org/html/2310.00099v1), 현재prior_model. 방법절은읽었으나AWing공식training loss소스는확보못했다. 이카드는AWing재현이아니다.
- 기존과차이/필요정보: 입력Gaussian σ9crop-px/peak255와현재bilinear target최대4cell은다르다. output72×96, crop288×384, ratio4. 현재CE+coordL1+L2를보존해target만동일중심normalizedGaussian으로비교한다. 새studentheatmaphead는추가하지않는다.
- 최소진단: 실제bilinear함수+σ1grid 수학fixture6개 CPU검사완료([RELATED_WORK](RELATED_WORK.md)). mass·ignoregradient0·boundarybias확인. σ1은예시이고fit값아님. native폭은실제cropmatrix로연결해야한다. Gaussian예상좌표가경계에서중심과달라져coordinateL1와상충가능.
- 단순대안/변경축/control: 같은teacher시작·추가step·TRAIN·입력·loss의bilinear vsGaussian. AWing이나weightedmap동시변경금지. currentbilinear+coordL1가가장단순control이며원래있던softsupervision을새발명으로쓰지않는다.
- 강한반론/역할: 틀린중심을넓혀도중심오류는수정되지않는다. AWing은peak1intensity회귀로CE대체와다르다. teacherTRAIN감소가학생T/R개선은아니다.
- 예산/중단/다음: 역사적teacher300step을가정하면2teacherfit/600,전달까지2studentfit/640추가;정확cost는예약후확정. 전체경로부족하면`REFINER_ONLY_SCREEN/STUDENT_POSE_UNTESTED`로제한. 경계/ignore불일치면fit전수정명세;실험결과미확인상태로실패기록하지않는다.

## F — 실제 가림 초기점에 대한 teacher 적응

- 질문/출처: [R9 PoseFix](https://arxiv.org/html/1812.03595v2)의기존방법검토재사용+[R1/R2/R8]방법과현Replay train코드. 원논문성공과팔레트결과를분리한다.
- 기존과차이/필요정보: 단순정답근처독립noise만아니라 `I_occ`에서frozenR0가만든`q_init_occ,box_occ`를쓴다. `y_target`은원본의적격S1/S2/P. noGTcrop, U/center추가감독없음. 원영상정확hint를그대로넣으면이질문을검사하지못한다.
- 최소진단: 기존300step/real8+source8/micro2/TFAdam1e−4/allteachertrainable/BNstatsfrozen및실측184.564s를확인했다. 새가림R0출력의결측·bbox변화·source유지probe는아직PENDING. 현재9/38은혼합난도·teacherexposed stored-index이며physicalaxis확정GT아님.
- 변경축/control/단순대안: 같은현재startingteacher의T_KEEP/T_OCC를같은300추가step으로적응(예시budget,최종SPEC필수). 기존frozen도참고선. 그뒤두teacher를freeze하고같은realmembership/common support로각학생320. frozen교사유지가가장싼대안이고교사추가학습량control생략불가.
- 강한반론: 가림detector의초기점/bbox실패가teacher에재입력되므로support가변한다. 적은9/38에과적합하거나syntheticprior를손상할수있다. source유지/teacher품질은학생자세성공과별도다.
- 역할/예산/중단/다음: 최소4fit/1,240update+캐시추론/probe;학생320을teacherstep으로복사하지않는다. 여기에재현reserve까지확보하지못하면`LOWER_PRIORITY/NOT_RUN_BUDGET`. GTcrop필요·공통support고갈·identity모순은해당경로만보류. 최종student-only RGB+solver를유지하고teacher-only향상을최종성과로부르지않는다.

## 다음 판정 원칙

A의실제fit/eval과B/C의작은실제graph진단을우선한다. 유효한두원리가확인될때만세번째cycle에서단일/결합누락셀을채운다. 한축tradeoff도보존하고T/R을cm+degree임의합으로합치지않는다. 모든후보를구현하는것이목적이아니지만유력한미실행후보가남아있다면이문헌표만으로탐색종료를선언하지않는다.
