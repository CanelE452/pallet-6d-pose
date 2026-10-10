# V8: 동일 관측의 원래 C3 솔버·마스크 대조

이 문서는 실제 V8 실행 전에 고정할 평가 계약이다. 실행 결과나 성공 판정은 별도의 보고서에 기록한다. 코드 작성이나 CPU fixture 통과를 실사 평가 완료로 취급하지 않는다.

첨부 지시문의 C3는 같은 IMAGE_ROLE 관측으로 마스크 없는 강건 PnP와 일반 PnP를 비교하도록 요청했다. 이전 corrected v1의 66-way decoder에서 그 대조를 이미 실행했다. V2와 V7의 최신 sparse boundary-only 관측에서는 마스크 없는 경로가 native/hybrid 관측을, 일반 PnP 경로가 native N3 관측을 사용하거나 일반 PnP 경로가 없었다. 따라서 이번 진단은 **최신 V7 관측을 그대로 공급하는 누락된 대조**다. 기존에 완료한 실험을 미실행으로 바꾸지 않는다.

사용자의 최신 실제 실사 범위는 Clean153 + Moderate92 = 245장이다. 원래 첨부 문서의 전체319 조건과 최신 범위를 구분한다. Severe74는 이번 새 진단에서 제외하고 과거319 결과를 그대로 보존한다. V7의 실사 결과는 이미 알려져 있으므로 독립된 미지 시험집합이라고 주장하지 않는다. V8 코드·정책·대조·집계 방식은 V8 새 실사 솔버 또는 GT 평가를 실행하기 전에 고정한다.

## 고정 입력과 세 경로

각 프레임에서 V7 `IMAGE_ROLE_BOUNDARY_ONLY`의 sparse `input_points`, K, 물리적 xyz, 영상 크기, selected index와 detector metadata, 최초 N3 H, 최초 N3 자세, 관측 계약을 그대로 사용한다. 원래 native N3는 V7 `OBSERVATIONS`의 `IMAGE_ROLE.native_N3_points`에서 가져온다. V7 parent row의 `native_points`는 이미 보정·표시 처리된 출력이므로 초기 native 입력으로 쓰지 않는다.

| 경로 | 최종 fit의 H | 솔버 | 의미 |
|---|---|---|---|
| ROLE_BOUNDARY_H_ROBUST | 원래 H 제외 | 기존 V4 강건 PnP | 기존 결과의 실제 재실행 및 parity 기준 |
| ROLE_BOUNDARY_H_STANDARD | 같은 H 제외 | 기존 V4 일반 PnP | 동일 관측에서 합의 솔버의 효과 진단 |
| ROLE_BOUNDARY_NO_MASK_ROBUST | H=[] | 같은 V4 강건 PnP | 동일 관측에서 H 제외의 fit 효과 진단 |

한 프레임의 동일 q/K/xyz/크기에 **하나의 PoseBank**를 사용한다. 같은 4-ID 수치 가설을 마스크 사이에서 재사용하며, refit과 LM의 실제 호출은 별도로 센다. 최초 H+robust 경로를 실제로 풀고 V7의 자세·상태·좌표와 고정 `atol=1e-7, rtol=0` parity를 확인한 뒤 다른 두 경로를 푼다. 이 replay도 실행량에 포함한다. 정상 완료 예정량은 245개 bank, 735개 실제 solve 경로이며, 그중 기존 대조 replay 245개와 신규 진단 490개를 구분한다.

Head, CAL, query decoding, 관측 admission, 초기 자세, 가림 분류를 다시 계산하지 않는다. 결측 sparse 좌표를 native 좌표로 메워 fit하지 않는다. native N3와 초기 자세는 기존 표시·기본 출력 반환에만 사용하며 최종 수치 prior가 아니다. 일반 경로는 전체 active U의 SSE·LM·rank로 판정한다. 8px diagnostic inlier가 4개 미만이라는 이유로 일반 PnP를 실패시키지 않는다. 강건 경로의 NEW에는 4개 이상의 final inlier와 필요한 수치 rank가 요구된다. 4·5점과 표준 해법·다중해 처리는 기존 솔버 및 별도의 CPU fixture에서 검증하며 6점 조건의 숫자를 임의로 바꾸지 않는다.

H를 적용한 경로는 H 초기 좌표를 최종 fit에서 제외한다. NEW R,t가 구해지면 H를 해당 자세의 재투영으로 교체하고 재투영 좌표를 다시 fit하지 않는다. no-mask 경로는 명시적 진단이며 원래 예측 H를 별도의 diagnostic 필드에 보존한다. 마스크의 불일치나 변화 자체로 프레임을 실패시키지 않는다. 점 수·배치 부족, 수치 실패, 부족한 합의, rank 부족, 다중해, N3 기본 출력 반환, 완전 실패를 분리한다.

## 사전 증거와 실행 경계

V7 protocol, full geometry1960, observations735, fixed geometry490, GT-free seal, 성공한 inference/cleanup receipt, 고정 cohort와 parity를 SHA256·bytes 및 전체 population/order로 검증한다. 모든 세 head와 여덟 방법의 population을 최초 V8 bank 생성 전에 끝까지 확인한다. 원래 native N3, 최초 pose/H, center 및 metadata는 같은 프레임의 OBS·fixed·primary 사이에서 대조한다. 한 프레임의 완전한 witness만 메모리에 보유하고 전체 원행을 축약하거나 제거하지 않는다.

CPU `CONTROL_CHECKS_ATTEMPT_B.json`의 성공과 `CPU_TEST_PROTOCOL.json`의 코드 binding을 전제조건으로 사용한다. 원래 실패한 `CONTROL_CHECKS.json` 및 실패 이유도 보존·binding하며 두 실행의 실제 호출을 감사 자료에서 분리한다. 환경 변수가 누락된 실패를 없던 실행으로 처리하지 않는다. controls/test는 CPU protocol 이후 변경하지 않는다.

V8 실행 직전에 독립 quiet preflight와 실제 실행의 resource snapshot을 남긴다. 다른 학습·평가·시간 측정과 경합하면 진행하지 않고 기록을 보존한다. 원본 영상·가중치·사용자 checkout 및 게시된 V7 증거를 보호한다. 새 사진, 수동 라벨, 합성, 학습, seed, 6D loss, PnP 역전파 또는 임계값 재선택을 수행하지 않는다.

실제 entry counter, solver의 전체 후보·best/per-dimension/alternative, singular/rank/Jacobian witness, operation/cache counts, 원래 native 좌표, 프레임별 shared-bank ledger를 저장한다. Jacobian fit ID 감사 metadata는 일반 경로의 `used`, 강건 경로의 `final_inliers`로 표시하며 Jacobian이 없는 경우 빈 목록으로 둔다. 이 metadata는 실제 Jacobian을 대체하는 증거가 아니다. 스트림 close/fsync/publish와 환경·canary·counter 복구가 끝난 성공 receipt를 먼저 만들고 그 receipt를 참조하는 GT-free seal을 만든다. 실패한 prefix와 실제 호출량을 보존하며 자동 재시도하지 않는다.

## 고정 비교와 실제 평가

비교는 각 세 경로와 BASE, N3_SUBPIX의 여섯 쌍, H_STANDARD−H_ROBUST, NO_MASK_ROBUST−H_ROBUST의 두 쌍으로 총 여덟 쌍이다. 차이는 앞−뒤이며 음수가 개선이다. 결과를 보고 비교의 방향이나 기준을 바꾸지 않는다.

성공한 V8 seal과 cleanup receipt를 모두 검증한 뒤 별도 `validation_checks.py`의 protocol을 freeze하고 실행한다. 그 독립 `VALIDATION_CHECKS.json`이 PASS이며 현재 seal·원행·ledger·receipt·core protocol에 SHA로 연결된 것을 확인한 이후에만 기존 실사 reference adapter로 실제 자세 평가를 수행한다. 기존 GT는 geometry-reconstructed DEV reference이며 독립적인 새 물리적 6D 측정이 아니다. GT나 사람 가시성은 관측·마스크·후보·자세 선택에 입력하지 않는다.

각 원행의 위치·회전·ADDsym에 대해 평균·표본 분산·표준편차·중앙값·P90을 계산한다. NEW, N3 기본 출력 반환, 완전 실패를 분리하고 산출률과 분모를 명시한다. 공통 산출집합과 전체 운용 결과를 모두 비교하며 완전 실패를 평균에서 조용히 제거하지 않는다. 원래 BASE/N3 reference와 부모 Hrobust replay parity, 직접 가시 코너 손상, 숨은 코너 재투영 오차 및 실제 최종 자세를 구분한다. 위치와 회전이 둘 다 좋아졌는지 답하며 숨은 좌표만 개선된 결과를 자세 개선으로 표현하지 않는다. 기존 고정 bootstrap draw와 reference convention을 유지한다.

Hrobust와 no-mask의 실제 U가 같으면 수치 결과의 parity를 검사하고 H의 최종 표시 재투영 차이와 fit의 차이를 구분한다. 동일 U에서 mask fit 효과가 0이라는 사실은 ROLE feature의 조건부 H/Base 의존성이나 가림 판단 전체가 필요 없다는 인과 주장으로 확대하지 않는다. 실제로 제외된 sparse ID와 남은 올바른 대응의 진단은 평가 이후에만 보고한다.

이번 경로는 sealed q를 재사용하는 PnP 진단이다. 그 wall seconds나 cache 시간은 처리시간 비교에 사용하지 않는다. 새로운 fresh 전체 경로 시간 측정은 수행하지 않으며 이미 별도로 완료한 V7 fresh 경로의 공식 시간과 명확히 구분한다.

## 공개 감사와 한계

최종 보고서는 실제 문제, 기존 수정, 이번 검증 범위, 실패하거나 미검증인 범위를 분리하고 코드·원행·독립 검산·실행량·그림과 연결한다. 평가 전의 이 계약 및 재현 안내에는 나중의 실제 결과를 덧붙이지 않는다. 결과는 별도 보고서에 저장한다. 전용 연구 브랜치만 사용하고 main 변경·자동 merge·force push를 하지 않는다.

공개 그림은 기존 고정 visual-case protocol의 동일한 여섯 프레임으로 24개 panel과 총8개 PNG를 만든다. 원본 RGB는 성공한 후단 평가 이후의 공개 검토를 위해서만 읽는다. NEW로 수락된 실제 fit에만 fit annotation을 붙이고 일반 경로의 8px inlier는 diagnostic으로 표시한다. 이미지를 읽어 최종 경로를 재선택하거나 threshold를 바꾸지 않는다.

이 진단은 물리적 edge ownership, source-to-real correspondence domain gap, 네 합성 통제 변형 또는 독립 역할 의미 검증을 새로 해결하지 않는다. 그 문제를 새 버그로 단정하거나 모든 코너별 방법의 불가능성으로 결론내리지 않는다. 관측이 충분하지 않으면 강건 PnP도 새 자세를 만들 수 없다는 한계는 남는다.
