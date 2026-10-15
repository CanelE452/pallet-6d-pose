# 추가143 중 저장된 후보로 5cm에 도달하지 못한 24장

현재 전체192장 진단에서 TEST49장 모두, 추가143장 중119장은 저장된 후보 안에 참조 위치 오차 5cm 이하 자세가 있습니다. 이 문서는 남은 추가24장만 원인을 분류하며 나머지119장·TEST49장을 평가에서 제외하지 않습니다. 참조는 기존 annotation 기반 derived pose이며 독립 물리 측정이 아닙니다. 정답으로 실행 후보를 고르지 않았고 새 추론·PnP·학습은 없습니다.

- 4장은 B/I가 2점 미만이라 모든 초기 솔버가 진입을 유보했고 저장된 자세 후보가 없습니다. 기존 coarse fallback도 계속 평가됩니다.
- 20장은 후보가 있으나 가장 가까운 위치도 5cm를 넘습니다. 이 중15장은 사후 참조 자세가 기존 모든 hard guard를 통과하는 W/D·C2 표현이 있습니다. 후보 탐색 범위·최적화·관측 likelihood를 추가로 구분해야 하며, 단순 재선택으로는 없는 자세를 만들 수 없습니다.
- 나머지5장은 모든 참조 표현이 hard guard를 위반하며, 모두 적어도 하나의 O 원 밖에 있습니다. 일부는 U 범위·partial 접선 범위·가시 기하 조건도 위반합니다. 정확한 참조 한 점의 실패가 모든 5cm 근방 자세의 불가능을 증명하지는 않습니다.

24장 중 MODERATE16·CLEAN8이며 D 처리 후 완전한 2D B/I가 2점 이하인 사진은13장입니다. 반면 완전한 B/I Jacobian의 local rank6도11장이어서 rank만으로 영상 코너 의미·크기 대응·위치 편향이 해결되지는 않습니다. 초기 투영 폭 중앙값181.94px, 범위67.30–456.49px로 작은 먼 물체만의 문제도 아닙니다.

[대표 실제 RGB 4장](REPRESENTATIVE_RGBS.png)은 작은 콘 가림005174, 어두운 적재/그림자033721, 왼쪽 절단014632, 먼 야간031344를 보여줍니다. 사진만으로 native별 정답 가시성을 확정하지 않았고 S 좌표·정답 자세를 그리지 않았습니다. 033721은 완전한 B/I5점·rank6인데 저장된 최선 위치 오차가41.18cm입니다. 031344는 B/I 보관 좌표7개가 D에서 완전한2D2개로 줄며, 저장된 최선은7.99cm입니다.

빈 후보집합4장 모두 봉인된 SAM mask와 supported global outline이 있습니다. 기존 coarse fallback 위치 오차는005174 11.25cm,030410 8.64cm,032043 13.85cm,032645 10.64cm입니다. 새 actual NMS chain 대안은 각각0/2/0/3개지만 이 대조군에서 인증된 factor는0입니다. 현재 observer/factor 연결은 기존 partial normal을 대체하는 범위에 한정되며 S mate도 제외합니다. 영상 선이 존재해도 native 3D edge 인증이나 충분한 자세 관측을 뜻하지 않습니다.

현재 O 영역은 **raw N3 중심 원 ∩ 영상 canvas**입니다. 실제 occluder mask와 교차한 영역이 아닙니다. 원 반경은 `max(8px, nonS raw 대각선 × .0864559283)`이며 CAL의 plastic O7점·5사진·2세션에서 정해졌습니다. wood O 전용 표본은 없습니다. source는 `ot_domains_v2.py:100–160,261–273`이며 model 내부 해시는6548c604…입니다. 미래/동시 coverage 또는 독립 physical occlusion support를 보장하지 않습니다.

- [R24 전체 수치·stage·rank·기존 reference guard](R24_MISSING_COVERAGE_DIAGNOSIS_V1.json)
- [5장 O 원/참조 투영과 4장 mask·line 상세](O_DOMAIN_AND_EMPTY_BANK_DETAILS_V1.json)
- [사용되지 않은 chain과 factor 인증 구분](EMPTY_BANK_AUTHENTICATION_NOTE_V1.json)
- [다음 관측 계약 복구안 A/B, 구현·실행 전](OBSERVATION_CONTRACT_REPAIR_PROPOSAL_V1.json)

A는 기존 CAL264 비S 위치 범위 `.5447507542×scale`를 O에 빌리는 hard range 대조군입니다. O 전용 확률이나 occluder support로 부르지 않으며 raw OOB의 기존 broad canvas를 줄이지 않습니다. B는 O의 위치 원을 soft/유보하고 실제 canvas 조건을 분리하는 방식입니다. soft loss에는 추가 가중치 계약이 필요하므로, 매개변수 없는 위치 hard-bound 유보가 더 단순한 원인 대조군입니다. 어떤 경우도 pallet mask의 음성 영역을 occluder로 간주하지 않으며 기본 e 결과를 바꾸지 않습니다.
