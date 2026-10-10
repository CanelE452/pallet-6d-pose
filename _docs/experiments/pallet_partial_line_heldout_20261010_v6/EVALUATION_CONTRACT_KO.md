# v6 양끝점 heldout 부분 선 절제의 평가 계약

이 문서는 실제 v6 정확도 실행과 새 reference 접근 전에 고정할 방법·모집단·점수화 경계다. v5 결과에서 드러난 검증 대상 차이에 대응하는 **별도 구조 절제**이며, 현재 성공이 증명된 보정법이나 물리 소유권 수리라고 부르지 않는다. 기존 v5 코드·protocol·실행량·원행·실패 기록은 변경하지 않는다. 새 namespace의 formal PROTOCOL이 실제 code와 이 문서의 SHA를 묶는다.

## 검증할 질문과 동일 입력

같은 학습 head와 관측에서, 각 관측 선의 두 등록 끝점 a,b와 H를 fit에서 제외한 다른 native N3 점의 자세가 그 선을 지지하는지 확인하면, v5 C2의 자기 일관성만 이용한 선 선택보다 최종 위치·회전이 나아지는가를 검증한다. 성공 판단의 주 대조는 여전히 고정 N3→cornerSubPix이며 전체 운용 모집단에서 위치·회전 평균을 함께 낮추는지다. 새 구조의 성능을 본 뒤 기준을 바꿔 반복하지 않는다.

고정 Base, 기존 N3 seed1, 수정 감독 IMAGE_ROLE 마지막 step3000 checkpoint, 기존 source-only CAL과 좌표/line decoder를 그대로 사용한다. READY 수정 감독의 corrected9000 업데이트는 과거 완료된 학습이며 이번 update0이다. 원 RGB·K·물리 W/H/D registry·native ordering·query와 FP16 feature rounding 계약을 보존한다. source에서 감독된 wire는 실사의 실제 물리 소유권 인증과 구분한다. role은 초기 virtual hull / internal / 초기 자세 unavailable 특징이며 모든 실사 경계의 물리 소유권 정답이 아니다.

## 선별 검증의 실제 fit 입력

1. 입력 선은 같은 frozen IMAGE_ROLE에서 만들어진 관측 선이다. 최종 선 지지 반경 검사·semantic edge ID·등록 endpoint ID·normal/offset·query ID 중복 검사를 유지한다. 실제 채택한 boundary corner가 최종 허용 point pool에 있을 때 그 incident edge를 consumed로 제거한다. 같은 edge의 여러 query나 그 끝점 둘을 독립 선으로 세지 않는다.
2. 각 검증 대상 edge e=(a,b)에 대해 **native N3 좌표만** 사용하는 point bank로 `hidden=H`, `temporary_excluded={a,b}`를 명시해 실제 새 heldout 자세를 구한다. boundary 교점으로 대체한 점, 검사 선, 다른 선 factor, 선의 초기 endpoint2D좌표는 이 검증 fit/scoring에 넣지 않는다. 사용·scoring·generator·actualfit 및 저장 scored/refit branch의 ID가 H∪{a,b}와 겹치지 않는 증거를 남긴다.
3. heldout numeric solver에는 초기 R,t·8점 재투영·초기 dimension prior를 주지 않는다. 등록 두 치수 분기를 유지하고, fixed point-only finite4 subset SQPnP/IPPE·강건 합의·다중해·수치 부족 처리를 보존한다. 독립적으로 알려진 dimension이 없는데 dim0를 강제하지 않는다. 동일 좌표/K/치수/4점ID의 가설 재사용은 허용하되 제외 점의 영상 좌표를 scoring/equivalence 잔차에 사용하지 않는다. 전역 numeric cache에는 제외 ID가 포함된 다른 부분집합의 생성 결과가 있을 수 있다. 해당 검증에서 실제 허용된 generator·scoring·refit branch만 H∪{a,b}를 제외하므로, 모든 primitive 생성 호출에서 제외 좌표를 읽지 않았다는 주장은 하지 않는다. 기존 v4 equivalence의 8개 좌표는 후보 자세의 **모델 재투영끼리** 비교한 값이며 제외한 영상 관측과의 잔차가 아니다.
4. available NEW, unresolved ambiguity 없음, 기존 numeric pose/point consensus 조건을 통과한 heldout R,t로 **등록 3D endpoint**를 재투영한다. 관측 normal/offset을 정규화해 두 endpoint의 signed normal distance RMS를 구한다. 기존8px 기준을 그대로 적용한다. 이 scalar는 무한선 위치의 조건부 기하 일치이며 실제 visible ownership·along-edge3D 지지 fraction·RGB boundary의 독립 정답을 인증하지 않는다.

## 상태와 선 pool 정책

| 검증 상태 | 최종 C2 입력 | 의미 |
|---|---|---|
| 조건을 만족한 heldout 자세, 선 RMS≤8px | SUPPORTED_RETAIN | 다른 native N3 point의 조건부 geometry와 일치 |
| 조건을 만족한 heldout 자세, 선 RMS>8px | CONTRADICTED_REJECT | 같은 고정 기준에서 해당 다른점 geometry와 불일치 |
| 실제 점 수·배치 부족, numeric failure, 다중해, 새 heldout pose unavailable | UNVERIFIED_RETAIN | 독립 검증 근거 부족을 별도 기록하고 원래 유효 관측 선은 유지 |

UNVERIFIED를 물리적으로 대응 없음(NONE)이나 틀린 선으로 바꾸지 않는다. 원래 관측 계약이 invalid인 것은 valid UNVERIFIED 관측과 구분하고 기존 fail-closed 상태를 유지한다. 초기 H가 틀렸다는 이유, 하나의 mask 차이, unavailable 검증 선 하나 때문에 frame 전체를 실패로 만들지 않는다. source CAL unsupported는 모델 감독 coverage 제한이지 실사의 physical absence가 아니다.

native N3와 H, Base 기반 proposal은 같은 고정 추정기의 출력이다. numeric heldout 검사에서 endpoint raw coordinates를 뺐다는 것과 시스템 전체 통계적 독립성을 구분한다. 새로운 양끝점 검증은 v5 저장 packet 감사의17개 조건부 witness를 재사용한 척하는 것이 아니라 실제 새 point-only 경로이며 각 호출·사용ID·결과를 남긴다.

## 최종 자세·자기 가림 출력

선 gate 이외의 코너 LOO 선택, 후보 좌표, native 8px basin과 strict improvement admission은 v5와 동일하다. retained 관측의 최종 fit은 **변경 없는 v5 point+line factor 정책**을 따른다. 실제 point/edge당 한 factor, 최소 실제4점·최소 actual pointinlier4개, 같은 factor pool로 모든 가설 비교, 최대3개 시작·soft_l1/f_scale8/max_nfev50, local modeled-normal joint rank6·cheirality·다중해 처리와 점4 미만 unavailable을 유지한다. 고정8px를 성능에 맞춰 확대/축소하지 않는다.

같은 선에서 만든 채택 코너와 선의 중복 관측은 금지한다. 실제 fit point count, fit IDs, final point inlier, raw/consumed/gate-rejected/unverified-retained/used/final line edge, support query 수를 각각 보존한다. fallback의 raw candidate inlier는 승인된 NEW 최종 inlier로 부르지 않는다. 선 pool이 비면 기존 v4 point-only solver delegate를 유지한다. 이 결과가 선을 이용하는 v5 C2 control과 같아야 한다고 강제하지 않는다.

NEW 최종 R,t가 얻어지면 H의 초기2D좌표를 재투영으로 교체한다. 제외한 H 좌표를 최종 fit에 넣지 않고 재투영점을 다시 관측처럼 fit하지 않는다. 새 자세가 없으면 native N3 기본 출력 반환으로 기록한다. NEW / fallback / no_pose를 구별하며 관측 부족·수치 실패·다중해의 이유를 보존한다. center, 다른 detection 후보, score/box identity 계약은 보존한다.

## 고정 방법과 control

새 core `common.PRIMARY`와 PROTOCOL에 선언한 alias를 authority로 사용한다. 다음4방법 각각245행을 만들며 fixed Base/N3 각각245행을 따로 둔다.

| arm | 변경/목적 |
|---|---|
| `N3_INDEPENDENT_CORNERWISE_ROLE_ENDPOINT_VALIDATED_LINES` | 위 선 gate만 추가한 v6 PRIMARY |
| `N3_INDEPENDENT_CORNERWISE_ROLE_PARTIAL_LINES` | 변경 없는 v5 C2 control; endpoint gate 없음 |
| `N3_INDEPENDENT_ROBUST_H` | native N3만, 예측 H 제외, 같은 prior-free 강건 PnP |
| `N3_INDEPENDENT_ROBUST_NO_MASK` | native N3만, H mask 없음, 같은 prior-free 강건 PnP |
| `BASE` / `N3_SUBPIX` | fresh 고정 simple controls |

unchanged v5 C2 control은 과거 GT-free v5 geometry와 **reference 읽기 전에** 비교한다. old geometry는 parity 감사 입력일 뿐 신규 fit/선 gate의 입력이 아니다. numeric tolerance atol1e−7/rtol0와 categorical exact를 고정하고 모든 비영 차이·exactness·최대차이를 남긴다. 범위를 벗어나면 실패 receipt를 보존하며 자동 무시/반복하지 않는다. C2 linepool·consumed·finalinlier·point+line score도 control parity scope에 포함한다.

## 245장 실행과 seal 이후 점수화

최신 사용자 범위의 existing clean153+moderate92=245,13session만 사용한다. severe74는 새 정확도·시간 panel에서 제외하며 원래319 결과는 보존한다. middle label에 부분 가림이 있을 수 있다. GT/성능으로 frame·case·weight·설정을 선택하지 않는다. 같은245 ID의4arm geometry980+fixed490+observation245를 GT-free로 완성·봉인한다. complete inference receipt와 cleanup_error=None, code/input SHA·고정 cohort·각 method의 중복 없는245행·GT_input=False를 검사하기 전 reference를 읽지 않는다.

scoring은 fresh fixed N3 phase와 기존 target/permutation/human states만 사용하는 변경 없는 후단 metric이다. 새로운 모델/PnP/optimizer를 금지한다. fixed Base/N3 좌표 및 기존 score parity를 보존하며 v5 control의 pre-GT parity를 먼저 검사한다. 값·ID·상태와 output_source를 원행에 남긴다. NOMASK는 `mask_applied=False`와 known-mask error NA를 유지한다. 알려진 human label과의 mask차이는 posthoc 감사이며 배포 selection에 들어가지 않는다.

전체245 운용 출력과 공통산출집합, candidateNEW와 bothNEW subset을 모두 보고한다. T/R/ADDsym의 평균·표본분산ddof1·SD·중앙값·P90·최대,153/92 strata, 새 pose·fallback·완전실패를 원행에서 계산한다. 큰 오차 frame을 빼지 않는다. 기존10000×13 session draw를 재사용하고 새 draw0이다. 위치·회전 동시 개선 여부는 primary−fixedN3 전체 운용 결과로 판단하고 CI가0을 포함하면 유의 개선/악화라고 과장하지 않는다.

직접 가시 코너 손상, 사람 SELF와 실제 재투영 H, hidden 좌표 오차, 최종 pose를 분리한다. mask가 틀렸지만 두 pose metric이 나아진 경우와 mask가 known label에 맞지만 나빠진 경우를 분리한다. 선 heldout gate의 SUPPORTED/REJECT/UNVERIFIED 상태와 최종 solver inlier, posthoc proxy agreement를 같은 의미로 부르지 않는다.

## 비용·재현·보고 경계

실제 runtime은 detector/N3/초기pose/featurepose/ROLE/코너LOO/새 선 양끝점heldout PnP/최종C2/H재투영을 모두 포함한 fresh API 경로로 별도 quiet window에서 측정한다. cache좌표 재생·기존평균합산은 사용하지 않는다. model load·RGB decode·GT scoring·parity·resource journal의 타이머 경계를 명시한다. 실제 model/OpenCV/optimizer/heldout 호출·resource snapshot·실패/cleanup 시도를 남긴다.

v6 core의 code·입력·정책 freeze는 새 정확도/reference 접근 전이다. 이후 저장행 품질 감사는 별도 code/input을 own arithmetic 전에 freeze할 수 있으나 accuracy 사전등록 또는 score 이전 freeze로 쓰지 않는다. 기존 source 레이 miss→NONE 수리는 historical `targets()`를 재작성한 것이 아니라 READY target·loader를 사용한 corrected 학습 경로이며, 고정 main N3→cornerSubPix 오류로 혼동하지 않는다.

새 학습·RGB·촬영·수동annotation·seed·CLIP/DINO·대형분할·PnP역전파·6Dloss·N3학습은0이다. 원본 사용자 checkout·이전 publication·v5실패/receipt·원행 바이트를 보존한다. 이 계약 작성은 code/doc static 작업이며 모델·PnP·numeric 성능실험·GT score 실행0이다. 실제 결과가 나온 뒤에만 검산과 개선/목표미달 상태를 보고한다.
