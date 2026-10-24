# V218: 실제 part/query 타깃 후보 구성

V159 사전은 query별 라벨이 아니라 camera-dynamic 8역할 정의다. top0/1/4/5는 상판 끝, bottom2/3/6/7은 하단 runner 끝이다. front/rear 이름만으로 보임/자기가림을 정하지 않는다.

원96행·U11·FIT64/DEV32를 보존했다. native D/E/S는 전96행 UNKNOWN이다. 기존 weak9는 nativeD9가 아니라 같은 팔레트의 local part가 보인다는 약한 관찰이다.

이번에 실제로 만든 데이터는 96행 대응표, 원 q8/native/L/availability만으로 만든 96행 FP32 조건, 두 원 ROI의 정확 OR로 만든 9개 finite support 타깃(7FIT/2DEV), 같은 support를 위한 9개 COVER/SHAM 기하 배치안이다. ROI는 학습 타깃에만 사용하며 배포 입력에 넣지 않는다. ROI 바깥은 음성 물질/배경 정답이 아니라 IGNORE다.

|frame/native|원역할|예상 native 부위|현재 대응 상태|
|---|---|---|---|
|0 / 6|SOURCE_FIT|outer bottom-runner end corner|DECK_VS_BOTTOM_RUNNER_UNRESOLVED|
|2 / 0|SOURCE_FIT|outer deck-board end corner|WEAK_TOP_ROLE_PART_ALIGNMENT_REVIEW_REQUIRED|
|2 / 4|SOURCE_FIT|outer deck-board end corner|WEAK_TOP_ROLE_PART_ALIGNMENT_REVIEW_REQUIRED|
|4 / 0|SOURCE_FIT|outer deck-board end corner|WEAK_TOP_ROLE_PART_ALIGNMENT_REVIEW_REQUIRED|
|5 / 4|SOURCE_DEV|outer deck-board end corner|WEAK_TOP_ROLE_PART_ALIGNMENT_REVIEW_REQUIRED|
|5 / 7|SOURCE_DEV|outer bottom-runner end corner|DECK_VS_BOTTOM_RUNNER_UNRESOLVED|
|6 / 4|SOURCE_FIT|outer deck-board end corner|F6_LINKED_BOARD_FEATURE_NOT_NATIVE_ADDRESSABLE|
|6 / 5|SOURCE_FIT|outer deck-board end corner|F6_LINKED_BOARD_FEATURE_NOT_NATIVE_ADDRESSABLE|
|6 / 6|SOURCE_FIT|outer bottom-runner end corner|F6_LINKED_BOARD_FEATURE_NOT_NATIVE_ADDRESSABLE|

f6의 3개 board/edge witness는 native 주소화와 분리했다. f0n6·f5n7은 상판 하단 윤곽 관찰과 사전의 bottom-runner가 다르므로 미해결이다. 나머지 top-role4행도 약한 부위 대응 검토 후보이며 native gold나 자동 native D가 아니다. 완벽한 3D mesh 인증이나 q와 참조점의 정확 일치를 일괄 선결조건으로 세운 것은 아니다.

COVER는 원 두 ROI 합집합 전체를 가리고 SHAM은 같은 면적·패턴을 다른 위치에 두는 계획이다. 현재 수치는 bbox 기하 가능성뿐이다. actual alpha all-valid, COVER 실제 변경, SHAM finite support 불변, 같은 part 맥락 보존을 root 승인 이미지 단계에서 검증해야 한다. q가 바뀌었는지와 feature를 가렸는지는 다른 축이다. 배치가 없으면 factual 관찰만 남기며 ROI를 줄이지 않는다.

두 SOURCE_DEV 후보와 모든 파생형은 모든 학습 loss에서 제외한다. 이전 appearance 노출과 옛 SOURCE 학습 이력은 사라지지 않으며 독립 신규 검증이라고 부르지 않는다. 이번에는 새 이미지·GT payload·모델·fit·paired image를 만들거나 읽지 않았고 training admission은 모두 false다.

다음 한 단계는 [고정14 PNG 바인딩](EXACT9_CLEAN96_AND5_CONTEXT_BINDINGS_NO_IMAGE_ACCESS.json)과 별도로 필요한 valid96 정확 바인딩을 root가 고정한 뒤, 이 [9개 계획](ALL9_FINITE_WITNESS_COVER_SHAM_EXAMPLE_PLANS.json)의 실제 crop/valid/alpha/RGB proof를 수행하는 것이다. native part 대응이 미확정이어도 finite local-part auxiliary 타깃 자체는 보존한다.

데이터: [원96 CSV](ALL96_TARGET_CANDIDATE_READOUT.csv), [finite9](ALL9_EXPLICIT_FINITE_LOCAL_PART_TARGETS.json), [native 부위 분리표](WEAK9_NATIVE_ANATOMY_CORRESPONDENCE_SEPARATED.json), [계약](TARGET_AND_CONDITION_CONTRACT.json), [요약](SUMMARY.json).
