# V216 weak visible-part / query-support 계약 독립 검토

완벽한 mesh·정확3D corner gold 없이도 공개 SOURCE의 **선택된 instance와 의도 부위에 대한 약한 관찰**은 진행할 수 있다. 단, 보이는 임의 local feature를 덮은 라벨을 현재 native head에 복사해서는 안 된다. 핵심은 그 witness가 원 q/native와 같은 frame 맥락에서 **어떤 부위를 묻는지 주소화되는가**이다.

[기존 weak44](../../query_visibility_dense_point_weak_bridge_source_training_20261025_v212/fit_preparation_a/WEAK44_INPUT_METADATA_AND_FUTURE_PNG_WHITELIST.json)는3frame의 factual8+SHAM18 D와 COVER18 weakE다. 권위는 `ASSISTANT_SOURCE_LOCAL_VISIBLE_FEATURE_WEAK_NOT_NATIVE_GOLD`이며, native0..4만 있다. [기존 loss](../../query_visibility_dense_point_weak_bridge_source_training_20261025_v212/fit_preparation_a/source_target_contract.py)는 point와 weak를 같은 direct logit에서 `old5 + point + .25 weak`로 학습했다. 이 weak bridge의 존재나 가중치 분리가 두 라벨의 의미가 같다는 증명은 아니다. 반대로 이번 text 검토가 기존 모든 weak 관찰을 오류로 판정한 것도 아니다.

[원 V207 연산자](../../query_visibility_source_train_witness_conditioned_controls_20261025_v207/raw96_control_preparation_b/PROTOCOL_BEFORE_RAW96_OR_DONOR_PAYLOADS.json)는 sampler 이후96에서 두 ROI의 **정확 pixel union**을 COVER alpha로 전부 덮고 SHAM alpha는0으로 만든다. 같은 donor 변환·RGB/alpha pattern·면적을 쓰고 valid·outside-alpha·actual RGB change를 검증한다. 이는 local-feature covered/preserved 대조이지 physical native 외부가림 gold가 아니다. ROI는 q 포함·중심·최소면적을 요구하지 않는다.

## 분리할 관찰

|축|사람이 기록할 수 있는 약근거|그것만으로 결정할 수 없는 것|
|---|---|---|
|instance|fullimage에서 같은 대상의 설명·대략 영역·식별 이유|pose/3D 동일성 인증, q8이 다른 물체에 모일 때 intended target 확정|
|intended part|선택 instance의 특정 코너·끝·블록 등 설명과 native 역할 대응 이유|generic wood/owner를 native D/E로 변환|
|local witness|원96에서 실제 보이는 feature의 inclusive ROI·kind·양성 근거|corrected q, 정확 point0, full part mask|
|주소화 가능성|원q/native+맥락으로 그 feature가 지정되는지, ROI selector가 추가로 필요한지|임의 grain ROI를 모르는 native head가 같은 문제를 푼다고 가정|
|query support|ON/NEAR/OFF/UNKNOWN과 같은 intended part에 대한 시각 근거|ROI box 포함=part support, GT 좌표오차/수정좌표|
|외부/자기 가림|선택 부위와 연결된 외부 전경·동일 instance self 근거|다른 물체·바닥·OFF·부분 미식별을 자동 E로 변환|

`LOCAL_FEATURE_ONLY` 또는 `EXPLICIT_ROI_SELECTOR_NEEDED`도 관찰로 보존한다. 별도 ROI-conditioned branch나 spatial auxiliary loss 후보가 될 수 있지만 **현재 조건 없는 native direct head의 primary D/E 라벨로 복사할 수 없다.** 현재 입력으로 특정 intended feature가 지정된다는 근거가 있다면, q가 ROI에 정확히 들어오지 않더라도 weak part 관찰의 후보가 될 수 있다. 물리 gold를 추가 선결조건으로 만들지 않는다.

query-support와 witness 라벨도 다르다. 가상의 q=(47.5,47.5), witness=[60,40,70,50]에서 COVER가 witness만 덮으면 q 근방은 그대로일 수 있다. 반대로 SHAM을 [42,42,52,52]에 두면 witness는 보존하면서 q 근방을 바꿀 수 있다. 이는 대수적 설명이며 새 image/label 증거가 아니다. **각 variant의 query support는 별도로 관찰·증명해야 한다.**

## 12frame 확장을 위한 최소 규칙

[기존 V215](../../query_visibility_public_train_visible_part_coverage_expansion_20261025_v215/ROOT_REPORT_KO.md)의 root6+B6은 서로 다른12frame을 본 것이다. 같은96행에 대한 두 reader의 독립 합의가 아니다. 새 관찰은 같은 original fullimage/q8/clean96/valid와 q/L/native/U를 두 reader가 독립 확인한 뒤 봉인한다. instance·의도 부위·같은 witness 정체성의 일치를 먼저 확인한다. 같은 feature-kind와 ROI overlap만으로 동일 부위나3Dgold라고 할 수 없다. 불일치/모호성은 UNKNOWN으로 보존하고 q 이동·쉬운 reader 선택으로 맞추지 않는다.

전체96행/U11을 유지한다. available85는 semantic known85가 아니다. Fullimage에만 보이고 원crop에는 없는 부위는 원입력의 visible witness를 증명하지 못한다. E에는 선택 부위에 대한 양성 외부 전경 근거가 필요하며, SELF와 위치 불확실성은 별도다. 완전히 숨겨져 대응을 설명할 수 없다면 UNKNOWN을 허용한다.

paired 개입은 원q/L/native/valid·같은 donor 변환을 고정하고 actual alpha/RGB proof를 남긴다. 두 witness union의 전부 COVER/전부 SHAM 미침범, alpha-all-valid, outside-alpha 원본 동일, SHAM feature RGB 동일, 각 witness의 실제 COVER 변화가 필요하다. geometry/actual-change 실패는 보존하고 라벨이나 donor를 바꿔 통과시키지 않는다. 주변 intended-part 맥락이 남는지도 기록하되 이를 완벽 mesh/고정 면적 gate로 만들지는 않는다.

A가 제안한 prefix별2FIT+1DEV(8/4frame)는 **기존 outer TRAIN을 바꾸지 않는 내부 개발 역할**로 사전 고정해야 한다. DEV는 이후 모든 loss branch와 augmentation descendant에서 제외한다. 기존 owner/support 노출이나 이미 본 영상 이력이 사라지지는 않으므로 blind 일반화 평가라고 부를 수 없다. 관찰 후 class 수를 보고 split을 고치거나 제외된 FIT/jitter의 여집합을 DEV로 만드는 것은 피한다.

지금 완료한 것은 text-only 설계 검토다. 새 annotation·학습 admission·이미지 view·GT·모델/점수·배열·Git은0이다. 정확한 fields·보존 조건·근거 SHA는 [REVIEW.json](REVIEW.json)에 있으며 A 최종 schema의 구현 검사는 별도 요청 시 한다.
