# 단일 수정 후보를 검토하고 보류한 근거

**`NO_SUPPORTED_CHANGE` — 확인된 구현 오류0, 관측모델 불일치 확인, 실제 물리 owner를 결정할 증거 부족.** 기존 E0를 유지하며 새 E variant를 PRIMARY192에서 평가하지 않는다. 미평가를 향상0 또는 새로운 경로의114/192로 바꾸어 쓰지 않는다.

검토한 한 가지 변경은 “후속 RGB 회복 이후 현재 source/conditionalΣ에 원 고정 CAL adapter를 다시 적용한다”였다. 새 모델이나 weight sweep은 아니지만, 원 adapter가 학습하지 않은 `B_ACTUAL_BOUNDARY_RECOVERY`를 기존 corrected 그룹과 같은 total-error 분포로 취급하는 새 가정이다. 이 가정을 오류 수정으로 지지하지 못해 **opt-in selector 구현·192 재선택 이전에 기각**했다.

1. **원 코드 위치.** `boundary_partial_observations_v4_4.fixed_fit_measurements:128`은 “Use recovered image coordinates with the original fixed fit scales.”라고 명시하고 새 xy/source/conditionalΣ에 원 fitΣ/reliability를 복사한다. 원 adapter는 `statewise_uncertainty_calibration_v4.apply_calibrated_measurements:315`에서 현재 source/state와 nonS scale로 effectiveΣ를 만든다. 구현 누락이 아니라 실험의 원 고정 scale 정책이다.
2. **실제 관측 불일치.** CAL267점 중10점/9영상, PRIMARY985점 중29점/22영상이 fallback→RGB recovery로 바뀌어도 원 fallback fit scale을 유지한다. PRIMARY 중27개는 full likelihood,2개는 철회된 partial archive다. 전39행의 실제/가상 scale·reliability를 [전이 표](CAL_PROVENANCE_ALL39_TRANSITIONS.csv)에 보존한다. 모든 fitΣ−현재 conditionalΣ는 PSD이며 원 override 복사와 whitening에 산술 위반0이다.
3. **발생 기전·예상 영향과 직접 반증.** 원 CAL floor를 현재 관측 source에 재적용하면 일반적으로 새 recovery점의 scale을 줄이고 reliability를 올려 full BI가 더 강해진다. 이는 선/outline과 충돌할 때 recovery점을 자동으로 정답 쪽으로 옮기는 변경이 아니다. source가 달라도 같은 잡음이라 가정해야 한다. 현재 source가 authentic native physical corner라는 확인도 없다. 특히 직전 full_BI 충돌 사례030898의 full5점은 B RGB4점+I RGB1점, fallback→recovery0이어서 이 재적용안으로 전5점과 full_BI 비용이 모두 불변이다. 따라서 처음 이 변경을 고려하게 한 구체적 충돌 자체를 바꾸지 못한다.
4. **반례.** CAL `wood_184309:000372/k3`의 mapped annotation/파생 reference 잔차416.470px는 raw RGB상 실제 포인트가416px 빗나갔다는 증거가 아니다. actual recovery(702.594,656.397)은 다른 기존 reference2 근처5.618px이고 고정 k3→ref6 등록과 충돌한다. 원 N3→ORACLE→fresh CAL의 점·pose·branch 전달은 exact, per-index geometry phase1 gap1.777e-16m, phase0 gap0.994m이며 RAW/N3 branch 혼합 오류도 발견하지 못했다. owner가 미확정인 채 σmax10.968→7.560px/reliability0.961476→0.981502로 강화하는 것은 물리적으로 정당화된 수리가 아니다. [실제 그림과 원 기록](audit_gallery/CAL_OWNER_PHASE_WITNESS.json).
5. **다른 오류 후보 검산.** proper Q/native XYZ/physical cuboid 보존, px² covariance/역 Cholesky whitening, full/partial 철회와 원본 archive exact, partial grouped radial robust와 reliability 한 번 적용에 오류를 확인하지 못했다. source의 `image_measurement=true`는 fallback에도 있으므로 이름만으로 독립 RGB점이라고 해석하지 않았다. 같은 RGB/mask의 점·선·outline 재사용은 통계적 독립성 미확인이라는 모델 한계다.
6. **사전 성공/실패 판정.** 우선 구현·전달 계약 위반이 실제 원 데이터에서 입증돼야 opt-in 코드와 평가 프로토콜을 봉인할 수 있다. 이 첫 요건을 통과하지 못했다. 새 covariance 숫자를 정하거나 CAL/여섯 사례/PRIMARY192에서 좋아질 때까지 반복하지 않는다. 과거 radial bias/scatter opt-in의 LOGO 악화를 숨기거나 재명명하지 않는다.

이 판단은 full_BI 결합 문제가 없다는 뜻이 아니다. 기존 가시경계 pilot의030898은 외곽이 성공 후보를 선호해도 full_BI를 더하면 순위가 뒤집혔고,005286은 외곽 자체가 실패를 선호했으며020955에는 family 선택·참조 ambiguity가 남았다. 한 항의 가중치를 바꿔 전체를 해결한다고 추론할 수 없다는 근거다. 최종 목적은 최고 pose 성공률이며 처리시간 최적화로 바꾸지 않는다.

다음에 필요한 한 가지는 **새로 격리한 한 세션에서 이미지 채널의 실제 물리 corner 대응을 독립적으로 식별하는 자료**다. marker/고정된 물리 corner 식별 등으로 W/D와 phase를 영상 밖 정보로 인증하고, 기존 기하 재구성 참조와 분리한 뒤 관측 잡음을 판단해야 한다. 현재 annotation을 재배열하거나 기존192장을 새 독립 test로 부르지 않는다.

**최종 판정: `REJECT` — 근거가 부족한 covariance adapter 재적용 수정안을 기각하고 원 E0를 보존한다.** 공정한192 비교의 성공 건수 차이와 후속 연구 판단은 최종 보고서에 별도로 기록한다.
