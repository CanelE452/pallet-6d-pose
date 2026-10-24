# V222 C65 연속 학습 준비

C66/C67은 같은 C65 최종 state를 strict-load하고 optimizer를 초기화한다. C65를 고른 이유는 최신 SOURCE recipe 연속성이며 보호 평가 성적이 아니다. 모델은 그대로118359개 파라미터/RAW/native+mask+q8이다.

공통 old192+point32+naturalweak16=240행과 RNG1810/1811/1812를 두 팔에서 동일하게 재생한다. C67만 독립 RNG1813으로 FIT 부모4개를 uniform-with-replacement 추출해 BEFORE/COVER/SHAM12행을 더한다. 새 가지는 `.25*mean_parent(.5*BCE(COVER,E)+.25*BCE(BEFORE,D)+.25*BCE(SHAM,D))`이며 hinge/새 contrastive 목적은 없다. 자연 weak도 양팔.25이다. 추가자료+loss branch bundle의 효과를 보며 같은 예제 수·주소화 단독 인과라고 주장하지 않는다.

probe는 각팔32 disposable 업데이트만, optimizer 구간 walltime과 로그/로드 I/O를 분리한다. 두팔 중 느린 step_s로 `max(200,min(2000,100*floor((60/max(step_s))/100)))`를 고정한다. probe 상태는 버리고 최종 fit은 C65를 다시 읽으며 중간 checkpoint선택 없이 두 final을 모두 보존한다. 지금 budget은미정/실제실행0이다.

원 C65 exact 선택6 source cache/동일MASK/owner조건부정규화와 RGB/255·GroupNorm을 재사용한다. 원입력q8와 GT-derived naturalweak q8는 권한이 다르며 새 pose/ROI/mask feature는 넣지 않는다. CF도 GT-family D48의 같은q/L/native/q8/valid에서 만든 finite-part weak이며 native point target은−1이다. 원query9는사용하지 않는다.

학습 전에는 natural FIT45와 CF FIT PNG만 decode한다. 자연57의 정확 source authority는 V218 `joined_weak_intended_part_targets_v1/ALL360_CHECKED_WEAK_TARGET_CANDIDATES.jsonl`(52ac5bb8…)의 각 `generated_input.RGB96`이다. SOURCE_DEV12는별도 postfreeze whitelist이며 CF DEV role manifest/PNG도 두 새 final을 봉인한 뒤에만 연다. 원본 fullRGB/GT/inputbank 경로를추가로따르지 않는다.

기하실패는 BEFORE D를보존하고 미생성C/S는 UNKNOWN/inputU로남긴다. 이후 저장 SOURCE평가는 all144상태·known·성공triplet 분모를구분하고 D/E CM과 B_D/C_E/S_D, 양방향 logit반응·SHAM오탐 및 role/component/native를 재집계할수있게 전체행을저장한다. 실사전이해결·물리gold·성능향상·채택주장은0이다.

실제 SOURCE입력·모델·PNG/probe/fit은열거나실행하지 않았다. pure3그룹과가상logit미분만확인했으며 head/modelforward는0이다. 실제rootentry에필요한 정확binding필드는 `READINESS_AND_EXACT_INPUT_INTERFACE.json`, 절차는단일protocol, 기본승인은false인entrytemplate에있다.
