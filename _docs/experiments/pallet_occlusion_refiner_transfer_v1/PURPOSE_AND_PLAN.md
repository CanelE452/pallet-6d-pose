# PURPOSE AND PLAN — occlusion refiner transfer v1

- [확인] 사용자 지시: 2026-09-21 첨부 `c86f83f5-0534-4d94-94ae-8503726761b5/pasted-text.txt`. 실행 범위는 저장소 감사 → E1 무학습 진단 → 무결성 gate를 통과한 경우에만 E2 2×2 보정기 파일럿이다. 자동 commit/push 및 E3~E6 실행은 금지한다.
- [확인] 시작 branch `main`, HEAD `73bfe38a259b3c846e49a98fb82c44578f6e2248`. 시작 시 tracked diff 없음. 기존 untracked 연구 자료는 보존한다. 새 namespace에만 기록하며 기존 모델·라벨·표·split·완료 artifact를 변경하지 않는다.
- [확인] 목적: frozen RGB 모델을 유지한 채 clean 실사 수도레이블로 보정기만 적응시켜, 정상 코너를 보존하면서 가림 코너 복원이 가능한지 판정한다.

## 왜 이 행동을 하는가

- [추정] A. 실제 whole-object 후보의 oracle headroom은 candidate 생성 병목과 선택 병목을 구분하는 진단이 된다. GT oracle은 진단 전용이며 배포 방법이 아니다.
- [추정] B. visible evidence로 hidden point를 복원할 수 있는지 확인해야 camera/dimension/keypoint contract 오류를 학습 문제로 오해하지 않을 수 있다.
- [추정] C. 전체 검출기 대신 refiner만 적응시키면 target-domain 보정 학습 효과를 분리하기 쉽다.
- [추정] D. 가린 RGB에서 frozen R0를 실제로 다시 실행한 initial points를 입력하면 clean points 복사의 shortcut을 줄일 수 있다. 이는 기존 pose-only 인공 가림 실험과 다르다.
- [추정] E. 같은 real exposure에 synthetic replay를 추가하는 대조는 source forgetting/정상점 보존의 역할을 진단한다. compute-matched라고 주장하지 않는다.

## 실행 계획과 중단 조건

1. [확인] 지정된 문서/코드의 실제 schema와 checkpoint·데이터 provenance를 읽는다. 기준 SHA 차이는 감사하고 checkout하지 않는다. CUDA는 host에서 확인하며 시스템을 변경하지 않는다.
2. [확인] E1-A는 frozen prediction을 먼저 고정하고 동일 모집단에서 whole-object candidate oracle과 실제 후보를 채점한다. 평가 실패를 분모에서 제거하지 않는다.
3. [확인] E1-B는 실제 paired physical occlusion 자료가 검증된 경우에만 physical recovery를 주장한다. 없으면 기존 자기 가림 sanity·치수 ±5% 민감도만 보고한다.
4. [확인] E1-C는 평가/Replay 수동 학습 세션과 겹치지 않고 기존 verified condition으로 clean임을 확인할 수 있는 pool만 허용한다. 라벨/태그 없음은 clean으로 간주하지 않는다. 임계값과 teacher 경로는 기존 고정값을 재사용한다.
5. [확인] E2는 leakage/contract/coordinate parity/고정 target path/unique clean ≥8 조건을 모두 만족해야 실행한다. 조건 미달은 STOP으로 기록하며 CAD eval을 임의로 학습 전환하거나 태그를 추정해 채우지 않는다.
6. [확인] 허용되면 synthetic-only PoseFix에서 seed1, 기존 verified Replay 설정으로 네 arm을 학습한다. CLEAN/OCC × source 없음/있음; 같은 real targets/masks/order/exposure/update/BN policy. OCC는 반드시 A(I)에서 R0 재추론한다. final checkpoint만 평가하고 새 selector는 학습하지 않는다.
7. [확인] 모든 출력과 checkpoint를 고정한 뒤 GT 채점, improvement/damage/random 갤러리, 회귀검사와 최종 감사/보고를 작성한다. gate에서 중단된 단계는 수행한 것처럼 표시하지 않는다.

## 예상 결과의 의미

- [추정] E1 oracle도 낮으면 selection보다 representation/candidate 생성 병목일 가능성이 있다.
- [추정] GT-visible geometry도 실패하면 camera/axis/keypoint/GT contract를 먼저 의심해야 한다.
- [추정] E2 train만 좋아지면 artificial-real gap 또는 clean target 다양성이 병목일 수 있다.
- [추정] occlusion 개선과 clean 손상이 함께 나타나면 복원/보존 trade-off다.
- [추정] source replay에서만 안정적이면 forgetting control이 필요할 수 있다.
- [확인] E2 PASS가 아니면 E3~E6을 실행하지 않는다. 결과를 보고 seed/lr/threshold/학습량을 바꿔 구제 실험하지 않는다.

## 사전 고정한 E2 판정

- [확인] 무결성 위반 0; R0 <5px → 출력 >10px damage ≤1%; A11 occlusion PCK10 ≥R0+3pp 및 동일 set N2/N3/Replay 최고보다 나쁘지 않음; primary 기존 comparator 대비 P90 ≤1.05배; clean 및 source-clean PCK10 ≥−1pp, source P90 ≤1.10배. 비교 기준을 실행 전 protocol에 명시한다. 이는 통계적 유의성/비열등성 검정이 아닌 pilot threshold다.
