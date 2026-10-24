# V198 native 조건부 가설의 독립 코드 검토

검토한 V175/176 및 V181→183→189→194 계보에는 categorical native ID를 neural forward에 주는 경로가 없었습니다. native/part 감독을 만들 때 쓰는 선언 permutation과, 모델 입력으로 native를 주는 것은 다릅니다. V160/C45/C46의 정확한 코드 포인터는 없어 프로젝트 전체에서 처음이라는 결론은 내리지 않습니다.

현재 head는 RGB·valid·고정 crop-relative XY와 선택적 SAM만 받습니다. native/q/L은 저장·검산용 META입니다. 공간 경로는 96→48→24→12→3×3→shared64이며 1×1 전역 평균은 아닙니다. 마지막 Linear에 onehot만 붙이면 native별 bias가 추가될 뿐, 영상과 native의 상호작용을 읽는 경로가 되지는 않습니다.

Root가 제안한 RAW 대조는 원 backbone/shared64/base direct를 유지하고, Linear(72,16)→SiLU→Linear(16,1) residual의 마지막 층을 0으로 초기화하는 방식입니다. 동일 구조·seed에서 zero8과 원 native onehot8의 초기 출력은 같습니다. 비선형 층은 영상×native 상호작용을 표현할 수 있어 논리상 차단 문제는 없습니다. 다만 zero8에서는 condition 열의 gradient가 0이므로, 결론은 이 설계에서 native 정보를 제공한 효과로 제한해야 합니다. 이 문서는 구현·fit·모델 채택이 아닙니다.

온전한 D/E/D triplet을 native별로 유지하면 raw native-only E 비율은 모두 1/3입니다. 16D/16E 균형 direct sampling의 기대 native-only p(E)는 모두 1/2이므로, native prior만으로 같은 native의 BEFORE/COVER/SHAM을 구별할 수 없습니다. 유한 draw 차이, 보조 owner/public loss의 prior와 영상의 source/factor shortcut까지 배제되는 것은 아닙니다. 이미 봉인한 TRAIN109 triplet의 native별 수는 35/26/13/22/6/7/0/0이며, 6·7의 direct 감독 부재와 모든 UNKNOWN 입력을 그대로 유지해야 합니다. 109는 독립 frame 수가 아닙니다.

native 슬롯은 물리 부위의 불변 이름이 아닙니다. 선언 permutation과 관찰 방향을 구분해야 하며, GT canonical 부위·endpoint·enclosure·pose를 모델에 주어서는 안 됩니다. 현재 근거는 weak 끝점/합성 구성 권한이며 native physical gold가 아닙니다.

원 q8를 추가하는 경우에는 별도 계약이 필요합니다. SELF_SCENE15는 원 scene의 q8와 개별 jittered crop q/L을 따로 보존하고 새 simultaneous jittered8은 만들지 않았습니다. 원 active q와 현재 q/L을 함께 주면 증강 dx/dy/scale을 답처럼 알 수 있습니다. 미래 q8 계약은 현재 selected q를 self0으로 두고 원 active anchor를 배제해야 하지만, 그것만으로 다른 7점의 간접 shortcut까지 제거됐다고 할 수 없습니다. native ID만 쓰는 대조는 이 추가 문맥 없이 먼저 사전 정의할 수 있습니다.

검토 기준은 V197 TRAIN 입력·목표 계약과 코드입니다. HOLDOUT 또는 실사 성적으로 이 후보를 고르지 않았으며, 새 데이터/배열/체크포인트/평가 결과를 읽거나 모델을 실행하지 않았습니다. 기존 q/L·initializer·분할·loss·draw·8000step·threshold·U를 바꾸지 않는 두 팔을 별도 scope로 동결한 뒤에만 실제 실험할 수 있습니다. 조건부 이득, feature 사용, 미세 cue 손실과 실사 전이는 아직 측정하지 않았습니다.

6개 순수 scalar 가상 검사는 통과했습니다. 정확한 코드 SHA·검토 범위·가상 입력/검사 소스는 같은 폴더의 REVIEW.json에 보존했습니다.
