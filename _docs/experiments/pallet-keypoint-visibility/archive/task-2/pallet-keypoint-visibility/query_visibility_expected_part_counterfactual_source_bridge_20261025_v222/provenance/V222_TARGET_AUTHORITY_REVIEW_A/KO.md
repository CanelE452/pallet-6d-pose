# V222 타깃 권위·추가 제어쌍 독립 검토 기준

상태: **준비 기준 고정**. producer/trainer의 새 프로토콜·코드와 실제 산출물은 아직 검토하지 않았다. 새 SOURCE 픽셀·GT·배열·weights·보호 점수 접근 및 학습은 0이다. 질문은 V219 C65의 기존 SOURCE recipe에 **query-addressed expected finite-part CF weak branch**를 더하는 한 요인의 효과다. C65 연속 warmstart는 최신 SOURCE 연속성으로 사전 지정하며 V220 성적으로 선택한 것으로 해석하지 않는다.

| 필수 확인 | 코드·프로토콜 및 이후 실제 증거에서 닫을 조건 |
|---|---|
| 타깃·family 권위 | `PUBLIC_SOURCE_GT_QUERY_JITTER_V1`의 support-checked D48, FIT39/DEV9만 후보로 사용한다. GT-derived training q8를 원 initializer q8라고 부르지 않는다. 동일 후보·native가 주소화한 **관찰 finite-part 약영역**이며 exact native point/자연 외부가림 gold가 아니다. 원 query9의 `native_DE_target=-1` 자료는 이 branch에 사용 0. 목재 presence·owner·mask 부재·q OFF를 D/E로 복사하지 않는다. |
| 같은 입력의 triplet | BEFORE/COVER/SHAM의 원 q/L/native/q8/condition32, availability/U, valid와 lineage를 exact 비교한다. 새 q·GT recenter·PnP·permutation search는 없다. 이미 학습한 FIT39 BEFORE는 새 독립 자료가 아니며 재사용·중복가중 방식을 명시한다. 성공 수를 미리 강제하지 않고 실패·missing을 보존한다. |
| exact finite target | 두 reader inclusive ROI의 **집합 OR**를 보존한다. bbox 대체·shrink·유리한 ROI 재선정은 없다. ROI는 배경도 포함하는 weak bag/region이며 모든 내부 픽셀을 pallet segmentation 정답으로 삼지 않는다. ROI는 감독 selector일 뿐 배포 feature가 아니다. |
| 실제 COVER/SHAM proof | 같은 occluder RGB/alpha pattern·support·질량을 위치만 옮긴다. COVER는 exact target OR 전체를 불투명하게 덮고 SHAM은 OR와 disjoint이며 target RGB를 보존한다. 모든 alpha support는 원 strict-valid 안이어야 한다. 원본 대비 alpha 밖 RGB 불변, 실제 RGB 변화, SHAM target 불변을 각각 검산한다. alpha 픽셀 수와 실제 changed 픽셀 수를 구분하며 no-sham/무변화/invalid 실패를 강제 성공으로 바꾸지 않는다. 이는 post-RAW96 제어이며 물리 렌더·native E·query-support 정답으로 승격하지 않는다. |
| role·분모 | family/row/부모 component identity로 FIT39/DEV9 및 모든 파생형을 join한다. SOURCE_DEV와 연결 descendants는 **모든 old auxiliary·point·weak·CF loss 및 normalization에서 제외**한다. DEV를 FIT으로 바꾸거나 원 role를 덮어쓰지 않는다. nominal/jitter/variant 상관과 부모 frame/component 수를 행 수와 분리한다. |
| 한 요인 학습 대조 | 양팔은 같은 C65 checkpoint/init, reset optimizer, 공통 old/point/기존 weak draw·예산·문턱을 사용한다. treatment만 명시된 CF weak branch를 더하며 weight·batch·RNG·BEFORE 중복 정책은 actual 전에 고정한다. checkpoint/초기 state와 공통 draw identity를 증명한다. SOURCEDEV나 보호 실사 성적으로 모델·타깃·문턱을 선택하지 않는다. |

독립 검토 순서는 새 producer/trainer의 동결 code/protocol 확인 → root의 별도 actual 승인 후 exact metadata/pixel proof 검산이다. 지금 파일은 실행 승인이나 최종 PASS가 아니다. 유한 부위의 약한 감독은 완벽한 mesh/물리 corner 인증을 일괄 선결조건으로 삼지 않고 사용할 수 있다. 동시에 약한 region 라벨 정의가 바뀐 것을 실사 전이 성공이나 원인 증명으로 주장하지 않는다.

근거는 기존 SOURCE 텍스트 두 개뿐이다. V221 C `source_readiness_c/KO.md` SHA `2cb0b45de26402f0f5d6fbd92bd945c8b2048d17c7e84bccf51aaa018f0fbc82`와 A `V221_SOURCE_HISTORY_AUDIT_A/KO.md` SHA `6c3982c40e312886df4390395f548f72956a2b9790c181421e8ff0d54f36e48f`. V95의 cue-set COVER/SHAM 학습과 V163의 무학습 ROI 관찰을 새 발상으로 반복하지 않고, 이번 차이는 expected finite-part와 query/native/q8 주소 권위를 정렬하는 것이다.
