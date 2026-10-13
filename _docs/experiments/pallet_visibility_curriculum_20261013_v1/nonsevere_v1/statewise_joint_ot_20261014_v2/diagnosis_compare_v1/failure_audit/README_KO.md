# 봉인된 current v3 실패 원인 사후 진단

모든 예측과 기존 사후 평가가 봉인된 뒤 CAL/재사용 TEST를 열어 독립 산술 검사를 했습니다. 상태·B/I·영역·solver·모델은 수정하지 않았고 새 PnP fit, 학습, TEST 기반 튜닝을 하지 않았습니다. 원인 후보의 동시 발생은 인과 증명이 아닙니다.

“제약을 추가하면 좋아진다”는 전제부터 점검했습니다. 고정된 B/I 3σ budget, O/T Ω, 가시성·canvas 제약 안에 실제 표적 자세가 있어야 합니다. 여기서는 사람이 정한 native 좌표와 그로부터 만든 역사적 reference pose를 따로 검사합니다. Reference pose는 독립 물리 ground truth가 아닙니다. Ω의 경험적 반경과 B/I covariance도 확률적으로 보증된 범위가 아닙니다.

| 분할 | 프레임 / 모델 대응 호환 | 호환 reference pose가 모든 현재 제약 통과 | BI budget·geometry·canvas 통과 | OT 통과 | true O annotation 영역 포함 |
|---|---:|---:|---:|---:|---:|
| CAL | 46 / 44 | 4 | 4 | 41 | 7/7 |
| TEST | 49 / 42 | 8 | 10 | 38 | 11/11 |

모델 대응이 호환되지 않는 프레임은 고정 native phase와 등록된 물리 모델의 꼭짓점 위치가 맞지 않습니다. 그 프레임의 pose score는 역사적 대칭 평가대로 유지하되, native pixel 개선이나 reference pose가 current native solver에 들어갈 수 있다는 주장은 분리해야 합니다. 대응을 유리하게 바꾸지 않았습니다.

기존 성공→current 실패의 primary 4건:

| 프레임 | 원본 / BI-only / current cm | 원본 / BI-only / current deg | 상태 | 핵심 후보 |
|---|---|---|---|---|
| eval_cad:1778653018878592768 | 2.52 / 7.48 / 7.48 | 1.62 / 1.91 / 1.91 | B T T B U B S U | few_visible_anchors_exact_reprojection_does_not_establish_accurate_pose, derived_reference_pose_excluded_by_BI_covariance_budget |
| eval_night08:1779449501478488320 | 3.91 / 30.99 / 30.99 | 1.85 / 8.98 / 8.98 | T B B T U B S S | few_visible_anchors_exact_reprojection_does_not_establish_accurate_pose, derived_reference_pose_excluded_by_BI_covariance_budget, derived_reference_pose_excluded_by_OT_domain, state_semantic_mismatch |
| eval_pallet07:1778652170735118080 | 2.46 / 16.22 / 16.22 | 1.07 / 4.96 / 4.96 | I T T T B O S B | few_visible_anchors_exact_reprojection_does_not_establish_accurate_pose, derived_reference_pose_excluded_by_BI_covariance_budget |
| wood_183705:000706 | 2.77 / 9.42 / 4.74 | 4.02 / 102.05 / 5.92 | T B B T B O S U | few_visible_anchors_exact_reprojection_does_not_establish_accurate_pose, derived_reference_pose_excluded_by_BI_covariance_budget, state_semantic_mismatch, some_BI_image_correction_harmful_in_pixel_metric |

이 중 최종 자세가 BI-only와 동일한 사례는 OT가 정확도를 떨어뜨렸다고 해석할 수 없습니다. sparse/오차 있는 B/I 관측과 상태에 따라 제외된 DIRECT 점을 먼저 분리해야 합니다. 특히 3개 관측은 P3P가 잔차 0으로 맞춰도 정확한 물리 자세가 보증되지 않습니다. Wood 000706의 OT는 102° BI-only 회전 오류를 약 5.93°로 줄였지만 5° 성공 기준을 넘었습니다.

B/I 픽셀 보정은 호환 모델의 실제 DIRECT 점만 집계했습니다:

| 분할 | raw / corrected 평균 px | 유리 / 불리 / 변화 ≤0.1px |
|---|---|---|
| CAL | 18.377 / 17.361 | 107 / 30 / 105 |
| TEST | 6.159 / 5.449 | 96 / 43 / 85 |

점 평균 개선과 특정 sparse pose의 악화를 구분해야 합니다. 보정이 움직이지 않은 cad 사례도 악화했으므로 모든 악화를 보정기 하나의 인과 효과로 돌릴 수 없습니다. 최종 B/I residual이 작은 것은 입력 관측에 맞는다는 뜻이며 semantic corner 또는 역사적 reference pose에 맞는다는 증거는 아닙니다.

S는 초기 coarse geometry에서 조건부로 제외되었고 이후 영상 query가 없다는 보장은 그대로 통과합니다. 그러나 물리적으로 실제 self-occluded라는 증명은 아니며, human DIRECT/EXTERNAL과 충돌하는 false-S는 별도 기록했습니다. Natural T의 UNANNOTATED target은 독립 실제 절단 label로 단정하지 않았습니다. 실제 crop의 source-visible T4/T5 누락과 false T0/T3도 별도로 남겼습니다.

Reference 좌표의 정수화 정도와 reference pose 투영 오차를 프레임별 JSON에 기록했습니다. 좌표가 반올림되거나 실제 pallet의 모서리가 이상적 cuboid와 다르거나 고정 phase가 틀리면, 모든 2D 주석과 하나의 rigid pose가 완전히 일치하지 않을 수 있습니다. 이번 검사는 불일치를 측정했으며 그 원인을 독립적으로 확정하지 않았습니다.

다음 controlled ablation 후보는 상태 유지/교체, raw-BI/corrected-BI, BI covariance budget, raw-centered Ω의 표적 포함, sparse-P3P branch와 가족 representative 정책입니다. 이번 진단 수치로 TEST에 맞춰 모델을 조정하지 않았습니다.

실제 사진 비교 (S는 첫 두 패널에서 제외, 최종 pose 투영에만 표시):

- [eval_cad:1778653018878592768](eval_cad__1778653018878592768.png) · [SVG](eval_cad__1778653018878592768.svg)
- [eval_night08:1779449501478488320](eval_night08__1779449501478488320.png) · [SVG](eval_night08__1779449501478488320.svg)
- [eval_pallet07:1778652170735118080](eval_pallet07__1778652170735118080.png) · [SVG](eval_pallet07__1778652170735118080.svg)
- [wood_183705:000706](wood_183705__000706.png) · [SVG](wood_183705__000706.svg)
- [eval_night08:1779449488068910592](eval_night08__1779449488068910592.png) · [SVG](eval_night08__1779449488068910592.svg)
- [eval_pallet09:1778653661653195264](eval_pallet09__1778653661653195264.png) · [SVG](eval_pallet09__1778653661653195264.svg)

[프레임별 JSON](FAILURE_AUDIT.json) · [진단 protocol](PROTOCOL.json) · [봉인 및 source 불변 확인](DIAGNOSIS_SEAL.json)
