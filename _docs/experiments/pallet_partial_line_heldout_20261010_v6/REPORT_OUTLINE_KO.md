# v6 실제 receipt 기반 보고서 작성 순서

이 파일은 formal PROTOCOL 이후 작성한 **unbound 보고서 준비 메모**다. 새 v6 scored 결과를 읽거나 수학·모델·PnP·GT 점수화를 실행하지 않았다. root가 complete geometry와 score를 확인한 뒤에만 아래 빈 결과 부분을 실제 receipt와 원행 통계로 채운다. frozen 코드·PROTOCOL·EVALUATION_CONTRACT·STATIC_REUSE_AUDIT는 수정하지 않는다.

formal PROTOCOL SHA는 `1de41781e280963c16486e828990ba9d4c4e1556c45faedd5ddc7edd2b55ce2c`다. 이 메모 작성 시 root가 actual245 inference 시작을 알렸고, 새로운 score 완료와 성공 여부는 확인하지 않았다. 아래 결과 항목은 계획 또는 필요한 증거이며 이미 실행한 것으로 쓰지 않는다.

## 최종 RESULT 첫 문장과 판정

“가림 판단이 일부 틀려도 다른 유효 대응점으로 자세를 구하고 자기 가림 코너를 재투영했을 때, 기존 단순 대조보다 실제 위치·회전이 좋아졌는가?”

완료 후 PRIMARY와 fixed N3_SUBPIX의 전체245 운용 T/R 평균·paired CI를 먼저 제시하고 둘 다 개선됐는지 답한다. v5 C2 대비 변화와 고정 N3 대비 목표 충족을 따로 쓴다. 새 산출률만 증가하거나 숨은 좌표만 좋아진 것은 위치·회전 개선으로 부르지 않는다. CI가0을 포함하는 contrast를 유의 개선/악화로 표현하지 않는다.

## 실제로 바뀐 관측 선택 단계

- same frozen corrected ROLE와 Base anchor, 기존 source CAL·decoder·코너 LOO·native8px admission을 유지했다.
- PRIMARY만 unused 선 e=(a,b)의 native N3-only point bank에서 H∪{a,b}를 빼고 prior-free 실제 자세를 구한다. endpoint를 뺀다는 주장과 모든 cached primitive에서 endpoint를 전혀 생성하지 않았다는 주장은 구분한다.
- checkable NEW의 normalized endpoint RMS≤8은 SUPPORTED_RETAIN, >8은 CONTRADICTED_REJECT, unavailable은 UNVERIFIED_RETAIN이다. UNVERIFIED를 physical NONE으로 바꾸지 않는다.
- retained 선을 기존 v5 C2 factor 정책으로 fit한다. consumed edge 중복 금지, actual4point minimum, local rank·다중해·NEW/fallback, 새 R,t로 H 교체를 보존한다.
- 초기 H와 Base proposal 의존성, 모델끼리 all8 equivalence, 잘못된 남은 N3 합의 가능성은 남는다. 실제 경계의 물리 소유권을 인증하는 방식이라고 쓰지 않는다.

구체 코드 anchor는 frozen EVALUATION_CONTRACT_KO.md와 STATIC_REUSE_AUDIT_KO.md에 있다. 새 성공 결과가 나오더라도 role은 초기 virtual hull/internal/unavailable 특징이라는 설명을 바꾸지 않는다.

## 실행 봉인·대조군·원행

complete inference와 cleanup_error=None, exact245 ID, geometry980+fixed490+OBS245, GT-free fields·SHA를 receipt로 확인한다. v5 C2 control parity는 reference 접근 전에 actual geometry를 비교한 결과를 써야 하며 허용오차 PASS를 bitexact로 바꾸지 않는다. 이후 score1470과 fixed N3 reference phase·matched guard, scoring fits0을 확인한다. 중단·resume·CLI·posthoc 실패가 있으면 모두 실제 시도 수와 함께 남긴다.

원행·cohort·seal·각 receipt·독립 verify와 public review의 실제 파일을 연결한다. large raw가 archive로 게시되면 존재하지 않는 원 gzip 링크 대신 archive manifest와 restore 방법을 연결하고 실제 복원 검산 결과를 구별한다. root의 publication·원격 SHA 확인 전에는 push 완료나 main 보존 원격 SHA를 임의로 주장하지 않는다.

## 전체245 및153/92 성능표

네 새 arm과 fresh BASE/N3_SUBPIX의 T/R/ADDsym 평균·표본분산ddof1·SD·중앙값·P90·최대, NEW/fallback/no_pose를 넣는다. easy153·moderate92·combined245는 현재 frozen cohort의 범위다. severe74와 원319는 새 성능에 합치지 않는다. 각 contrast의 common operational/candidate NEW/both NEW와 pair IDs·denominator·기존10000×13 draw CI를 같이 설명한다. primary−v5C2, primary−nativeH, primary−nomask의 결과를 가림·line gate 효과와 연결하되 association과 causal ownership을 구별한다.

## 관측 품질과 실패 이유

실제 gate 기록에서 raw/consumed/unused/SUPPORTED/CONTRADICTED/UNVERIFIED/retained/finalinlier edge 수, available/insufficient/numeric/ambiguous 검증 자세, remaining correct/incorrect/unknown point·layout·dimension branch를 따로 제시한다. 지원 query 수를 독립3D 선 수로 세지 않는다. gate와 pose 성능의 관계는 저장 행의 posthoc association이며 heldout witness가 틀릴 수 있다. 선 삭제와 성공 frame 수만으로 효과를 단정하지 않는다.

wrong known mask+T/R 둘 개선과 correct known mask+둘 악화를 나누고 혼합·unknown/no-mask를 유지한다. DIRECT damage와 SELF/실제 H 재투영의 before/after, final point/line inlier와 fallback raw candidate를 분리한다. proxy endpoint RMS는 geometric proxy model-line agreement이며 실제 RGB 경계 ownership 또는 along-edge physical visibility 정답이 아니다.

## 실제 full-path 비용과 그림

실제600 fresh API의4arm×(20warmup+130measured),26 eligible panel을 receipt에서 확인한다. primary의 detector/N3/초기pose/featurepose/ROLE/코너LOO/새 dualendpointLOO/finalC2/H재투영이 측정 구간에 포함됐는지 쓴다. mean/SD/median/P90/max와 actual call ledger·resource quiet 결과를 제시한다. model initialization·RGB decode·GT/parity/journal 경계는 runtime receipt대로 분리하며 cached timing이나 stage 평균 합산을 사용하지 않는다.

고정6사례와 분포 그림은 frozen case ID를 유지한다. 그림의 GT proxy·실제 출력·채택 corner·H reprojection·retained/finalinlier·gate rejected 선의 legend를 명확히 한다. 완료 figure 검토 receipt가 생기기 전에는 이미지 inspect PASS를 쓰지 않는다.

## 이전 실패·수리와 아직 입증하지 않은 것

historical training.targets의 ray miss→NONE 감독 오류는 legacy 함수를 덮어쓴 것이 아니다. READY target 수리와 corrected loader 경로로3×3000=9000 수정 학습을 완료했으며 고정 main N3→SubPix 오류로 혼동하지 않는다. 이번 v6 새 training0/RGB0이다.

기존 authoritative receipt의 완료 사실은 다음과 같다. 원래 미수정3head는 full319×3 observation957·6posepath1914를 평가했고 GEOM0NEW/319fallback, NO_ROLE3/316, ROLE0/319였다. **수정 감독3head**는 최신 범위245×3 observation735·6path1470를 완료했다. corrected GEOM236NEW/9fallback, NO_ROLE193/52, ROLE220/25, 모두0완전실패다. corrected checkpoint SHA와 TRAINING_COMPLETION/OBSERVATION_SEAL/SUBSET_EVALUATE_ADAPTER_RECEIPT는 `pallet_kp_corrected_supervision_20261010_v1`에 연결한다. corrected245를 full319라고 쓰지 않는다.

위3condition의 후단 자세 평가는 원래66-way decoder에서 이미 완료됐다. 이후 새 confidence/line/cornerwise/partial-line decoder는 ROLE만 사용하므로 현재 decoder에서 GEOM/NO_ROLE 대비 ROLE 필요성은 입증하지 않았다. 실사 성능을 보고3head 중 새 head를 선택하지 않는다. source controlled4variants 미완료와 source role3 semantic replay의 초기 featurepose 미보유 제한도 현재 실행이 해결한 것으로 쓰지 않는다.

결론은 root가 실제 결과를 확인한 뒤 작성한다. 구조 수정의 실행·검산 완료와 최상위 위치·회전 목표 달성을 구분하고, 필요한 다음 검사만 근거와 함께 적는다. 이 메모 자체의 새 accuracy/model/PnP/GT score·numeric audit 실행은0이다.
