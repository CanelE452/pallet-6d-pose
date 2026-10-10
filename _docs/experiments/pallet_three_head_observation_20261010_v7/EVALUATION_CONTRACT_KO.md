# 수정된 세 head와 두 관측 공급의 고정 평가 계약

이 문서는 v7의 새 실사 점수 접근 전에 고정할 평가 정의다. 코드 준비와 source CAL 준비, 실제 정확도·시간 측정 완료는 서로 다른 상태이며, 이 문서 자체는 아직 실행되지 않은 단계의 PASS를 주장하지 않는다. 비교 결과가 좋지 않다는 이유로 head·seed·threshold·학습량을 고르거나 반복하지 않는다. 원본 영상·가중치·사용자 변경·기존319평가·v2–v6 결과와 frozen byte는 보존한다.

## 목적과 모집단

최상위 질문은 “가림 판단이 일부 틀려도 다른 유효 대응점으로 자세를 구하고 자기 가림 코너를 재투영했을 때, 기존 단순 대조보다 실제 위치·회전이 좋아졌는가?”다. v7은 이미 수정 감독으로 동일 초기화·배치 순서·각3000update/batch16으로 학습 완료한 GEOMETRY_ONLY, IMAGE_NO_ROLE, IMAGE_ROLE을 같은 source CAL 알고리즘·decoder·후단 point-only solver에 적용한다. 세 head의 새 학습은0, 새 RGB·annotation도0이다. 기존 corrected245장66-way decoder 평가와 현재 새로운 confidence/선교점 decoder의 세 head 비교를 구분한다.

기존 authority의 clean153·moderate92, 총245장/13session을 그대로 사용한다. severe74는 이번 정확도·runtime panel에서 제외한다. 원319자료를 삭제하거나 쉬움·중간을 모두 무가림으로 재정의하지 않는다. COHORT의 ID·난도와 source/input byte binding이 authority이며 새 성능으로 subset을 선택하지 않는다.

## head와 source-only calibration

원 corrected 마지막 step3000 checkpoint를 고정한다. checkpoint·TRAINING_COMPLETION·CHECKPOINT_METADATA·READY 감독 SHA·기존 cache manifest/family split/selected Base 좌표를 검증한다. 미수정 head나 좋은 source-test checkpoint로 대체하지 않는다.

| arm | 원 forward의 입력 처리 | calibration 실행 |
|---|---|---|
| GEOMETRY_ONLY | image/neck0:19를0으로, geometry19:25와 초기 virtual role25:28 유지 | 기존 source CAL128의8batch×16, head forward8회 |
| IMAGE_NO_ROLE | role25:28을0으로, image/neck0:19와 geometry19:25 유지 | 같은 CAL128의8batch×16, head forward8회 |
| IMAGE_ROLE | 기존28channel 모두 유지 | 기존 ROLE CAL의9개 파일 byte-exact 재사용, 새 forward0회 |

source CAL은 index768..895, calibration128family만 사용한다. train/source-test/실사 점수·GT로 threshold를 정하지 않는다. 두 새 arm의 예정 실제 head 호출은 총16batch/256image exposure다. ROLE 원 영수증의 과거8forward/128exposure는 보존하고 새 REUSE_RECEIPT의 current0과 분리한다. 새 source pass의 완료 여부는 별도 실제 COMPLETION이 authority다. 모델 입력 feature cache를 다시 만들거나 detector·neck·N3·초기pose·PnP·ray를 실행하지 않는다.

수치 calibration 본문은 변경 없는 v2 `calibrate(saved, source_CAL, READY_targets, original_v2_protocol)`이다. confidence는 `sigmoid(best_candidate_logit − NONE_logit) = p_best/(p_best+p_NONE)`, correctness는 certified source POS와 bestbin 오차≤2px, genuine NONE은 incorrect, IGNORE는 unknown이다. 최소confidence0.5 중 one-sided95% Wilson lower precision≥0.95를 만족하는 가장 덜 제한적인 cutoff를 tie 포함해 선택하며, 없으면 새 head query를 abstain한다. correlated scene queries와 cutoff scan 때문에 Wilson 수치를 실사 transfer guarantee로 주장하지 않는다. 각 head는 자기 CAL logits에서 계수를 정하며 accuracy로 가장 좋은 arm의 계수를 다른 arm에 복사하지 않는다.

지원된 source semantic edge, query uncertainty·line covariance·corner radius/외삽은 같은 기존 정의를 사용한다. source CAL 미지원은 `MODEL_CALIBRATION_UNSUPPORTED`이며 실사 physical absence가 아니다. virtual boundary/internal/unavailable role 특징은 실제 물리 경계의 소유권 classifier가 아니다. 기존 physical wire 지원의 virtual 교점도 실사 독립 physical corner truth 인증으로 확대하지 않는다.

## 사전에 고정한 두 관측 공급과 대조

원 Base detector·N3 seed1→cornerSubPix·등록 K/실제 치수·기존 Base query/feature rounding과 decoder를 고정한다. accuracy frame마다 detector/N3/초기 N3 pose/기존 Base query feature를 한 번 공유하고 세 head를 각각 한 번 forward한다. head logits·observations를 ensemble하거나 다수결하지 않는다.

`BOUNDARY_ONLY`는 실제 decoder의 두 관측 선 교점 중 기존 final-line radius consistency와 corner uncertainty≤8px admission을 통과한 sparse 좌표만 넣는다. native N3와 거리≤8px 또는 corner LOO를 새 gate로 적용하지 않는다. 없는 대응 좌표는 NaN이고 native 좌표로 fit pool을 채우지 않는다. center8은 출력 메타데이터로 유지하며 PnP corner ID로 쓰지 않는다. initial H는 이 sparse pool에서도 가설 생성·scoring·refit에서 제외한다.

`CORNERWISE_HYBRID`는 변경 없는 v4 선택기다. 같은 admission을 통과한 실제 교점을 native N3의 고정8px basin과 prior-free native point bank의 H∪{k} heldout 검증으로 비교한다. residual≤8px와 native squared residual 대비 strict improvement>1e−8px²가 확인된 교점만 대체하고 나머지는 유효 native RGB 관측을 유지한다. missing native의 기존 조건부 경로도 그대로 유지한다. 검증 projection을 새 관측 좌표로 사용하지 않는다. initial H/Base proposal 의존성은 남아 있으므로 조건부 numeric exclusion과 완전한 통계적 독립성을 구분한다.

| method | 의미 |
|---|---|
| IMAGE_ROLE_BOUNDARY_ONLY | 고정 primary, ROLE의 sparse 실제 boundary 교점만 |
| GEOMETRY_ONLY_BOUNDARY_ONLY | 같은 sparse 공급, GEOM head |
| IMAGE_NO_ROLE_BOUNDARY_ONLY | 같은 sparse 공급, NO_ROLE head |
| GEOMETRY_ONLY_CORNERWISE_HYBRID | 같은 v4 hybrid 공급, GEOM head |
| IMAGE_NO_ROLE_CORNERWISE_HYBRID | 같은 v4 hybrid 공급, NO_ROLE head |
| IMAGE_ROLE_CORNERWISE_HYBRID | 변경 없는 v4 ROLE hybrid geometry control |
| N3_INDEPENDENT_ROBUST_H | native N3 관측, 예측 H 제외, 같은 강건 solver |
| N3_INDEPENDENT_ROBUST_NO_MASK | native N3 관측, H 제외 없음, 같은 강건 solver |
| BASE / N3_SUBPIX | 새 같은 frame capture의 기존 단순 고정 출력 |

새8arm 모두 변경 없는 prior-free v4 finite4-ID bank를 쓴다. 최대8corner의70subset/치수 분기이며 exact 좌표·K·치수·image size의 가설을 재사용한다. 임의 dim0·초기 R,t/projection/dimension prior는 넣지 않는다. actual4point·배치/수치·cheirality·강건 합의·다중해 조건을 보존한다. native mask가 한 점 틀렸거나 fit 후 바뀐 사실을 frame 실패로 바꾸지 않는다. 점 부족·insufficient consensus·numeric·ambiguity와 fallback은 별도로 남긴다. 이 v7은 point-only 비교이며 관측 partial line을 추가 pose factor로 넣지 않는다.

H를 적용하는 여섯 head/supply와 nativeH 경로는 initial predicted H를 실제 fit에서 제외하며, 새 최종 R,t가 나오면 그 H의 초기 좌표를 최종 projection으로 교체한다. no-mask native 대조는 H=[]로 모든 유효 native 관측을 평가하며 predicted H는 진단 정보로 분리한다. projection을 다시 PnP 관측으로 넣지 않는다. 새 pose가 없으면 기존 full native N3 기본 출력을 반환하며 fallback은 NEW가 아니다. BOUNDARY_ONLY의 NEW에서 표시되는 unobserved non-H native 좌표도 독립 fit 관측이었다고 주장하지 않는다. `selected_corner_ids`는 admission/공급 단계의 proposal ID이며 solver used/fit/final inlier와 같다고 가정하지 않는다. 이 ID들, 실제 display 교체와 H projection을 각각 기록한다.

## 봉인·대조 parity와 후행 점수

GT-free OBS735행의 identity는 `(head_arm,id)`다. 새 geometry1960행의 identity는 `(method,id)`이며 fixed Base/N3490행을 더해 실제 점수2450행이다. 모든 arm마다 같은245ID가 정확히 한 번 있어야 한다. complete seal뿐 아니라 cleanup_error=None인 complete INFERENCE_RECEIPT와245/735/1960/490 count·SHA·GT-free field가 필요하다. prefix/실패/불완전 cleanup을 완성으로 처리하지 않는다.

실사 reference 전에 `IMAGE_ROLE_CORNERWISE_HYBRID`245행을 원 v4 ROLE GT-free geometry와 비교한다. 같은 input/output/phase/R,t/dimension/ID/status/selection decision에 categorical exact와 atol1e−7/rtol0 numeric parity를 적용한다. nonzero within-tolerance와 bitexact를 구분하며 실제 차이가 제한을 넘으면 영수증을 보존하고 점수화를 중단한다. 이 비교의 과거 데이터는 검산 전용이고 새로운 pose 선택에 넣지 않는다.

그 후에만 변경 없는 기존 scorer/reference mapping을 사용하며 scorer 단계의 model/PnP/optimizer fit을 금지한다. corner/human state는 fresh fixed N3의 같은 permutation phase로 계산한다. matched=False 또는 missing/invalid reference는 UNKNOWN이다. 유리한 GT phase·새 annotation·점수 기준을 선택하지 않는다. scoring 완료 receipt와 sealed byte가 맞아야 통계를 만든다.

## 전체 운용·통계·관측 품질

전체245/쉬움153/중간92 모두 T(cm),R(degree),ADDsym(cm)의 n·mean·sample variance(ddof1)·SD·median·P90·max를 원행에서 계산한다. 각 method에 operational/new-pose/fallback/no-pose·fixed-control count를 남긴다. 큰 오차가 실패로 빠져 평균이 좋아 보이지 않도록 전체 운용과 common operational을 먼저 보고, candidate-new와 both-new subset도 별도로 비교한다.

common.py의 고정16contrast는 세 head×두 공급 각각−N3의6개, 세 hybrid−boundary의3개, 각 공급의 NO_ROLE−GEOM/ROLE−NO_ROLE의4개, nativeH−no-mask와 primary−두native control의3개다. 같은13session의 기존10000×13 multiplicity draw를 재사용하고 새 draw·seed는 만들지 않는다.3stratum×16contrast×3scope=144paired group·432metric CI가 요청된다.0개 pair의 CI는None으로 남기며 빈 bootstrap resample 수를 보고한다. 이 수는 예정 검산량이지 실행 완료 claim이 아니다.

POSTHOC1960행은 saved observation/ref/solver만 읽는다. computed corner→admission→supply 선택→H 제외→NEW fit/inlier→실제 output을 분리하고 `BOUNDARY_ONLY`에 존재하지 않는 LOO residual은None으로 기록한다.8px reference agreement와 correct/incorrect/unknown·used pool 및 stored layout은 physical ownership truth가 아니다. wrong known mask인데 T/R 함께 개선한 경우, known mask가 맞는데 함께 악화한 경우를 분리한다. 직접 가시점 DIRECT의 손상(입력≤5→출력>10px)과 humanSELF/실제로 H 재투영한 subset의 before/after를 따로 보고한다.

독립 verify는 statistics.py를 import하지 않고 stdlib `fsum`·quantile·같은 frozen draw로 moments/CI/ID/pool/mask/좌표 품질을 다시 계산한다. publicly reviewable moments·CSV·H scalar projection은 source GT/모델/PnP를 다시 실행하지 않는다. saved numeric witness PASS를 reference의 물리적 진실성·실사 전이·정확도 성공으로 주장하지 않는다.

## 실제 비용과 보호

별도 quiet window에서 고정10경로(BASE,N3,*새8arm)의 각각20warmup+130measurement, 총1500fresh full path를 한 번 측정한다.26eligible panel의5repeat이며 source/head arm 선택은 사전 method에 따른다. head 경로의 feature+요청 head forward, 초기 자세·corner LOO·최종 robust pose·H projection을 전체 interval에 포함한다. 모델 load·RGB file/decode·GT scoring·journal/resource/parity의 timer 경계는 영수증에 명시한다. cached 좌표 재생이나 stage 평균 합산으로 비용을 대신하지 않는다.

실제 resource guard 실패/CLI 실패·prefix·cleanup·primitive counts·source calibration16/currentROLE0을 별도 영수증으로 남긴다. 학습량은 이전 corrected9000과 이번0을 분리한다. 미공개 source-neck/bootstrap 감사와 기존 frozen 결과·사용자 checkout을 해시로 보호한다. main 자동 merge·force push를 하지 않으며 parent가 전용 research branch의 정상 게시 SHA를 확인한다. 현재 method가 최상위 목표를 달성했는지는 실행된 원행/CI가 나온 뒤 답한다.
