# C 표본 출처 보충: mask와 convex envelope를 구분

[본 그림/비용 보고서](REPORT_KO.md)와 [원 EVIDENCE](EVIDENCE.json)는 그대로 보존했다. 이 보충은 저장된 mask manifest와 생성 source만 확인했으며 이미지·mask 배열·모델을 다시 디코드하거나 추론하지 않았다.

031426 `outline_observation`은 **supported=True**, **global_mask_only=True**, **convex_envelope_is_not_actual_complete_surface=True**다. 명시적 method 문자열은 없고 keys와 source 호출 경로로 생성 방식을 확인했다. image SHA `782ca888…`, mask NPZ `wood_night_01__031426.npz`, file SHA `542a1f4e…`는 원 R143 manifest의 해당 행과 정확히 일치한다. opaque NPZ 바이트 해시도 확인했으며 배열은 열지 않았다. float64 mask-array SHA와 별도 canonical pixel digest는 서로 다른 규약이므로 억지로 같은 값이라고 비교하지 않았다.

실제 경로는 다음과 같다.

1. `observe_remainder_rgb_v4_14.py`가 기존 `GeometryPromptSAMCapture` full-mask 결과를 NPZ와 manifest로 보관했다.
2. `capture_boundary_recovery_masks_v4_6.py`는 MobileSAM vit_t box prompt에서 3개 중 max-quality mask를 고르고 component cleanup을 한다. 이 행은 index1, 품질0.9681885다. 초기 model projection은 역사적 box prompt 구성에 쓰였고 정밀 관측으로 복사됐다는 뜻은 아니다. point prompt는 사용하지 않았다.
3. `evaluate_fixed_wd_archived_tangent_v4_15.py`는 `pallet_mask_clean`을 읽어 V13 `observe_global_outline`에 준다. 이 함수는 mask≥0.5인 **전체 foreground**의 convex hull을 만든 뒤 canvas 경계를 처리하고 1px 간격 midpoint 표본을 생성한다. 이 함수 자체에는 초기 model projection 입력이 없다.
4. 그 보관 `task['outline']`이 후속 task를 통해 V32 packet의 `regularizers.outline_observation`으로 복사된다. 이번 그림은 이미 보관된 225개 표본만 썼다.

따라서 주황 표본은 “SAM이 직접 측정한 선점”이 아니다. **SAM-derived clean-mask의 후속 convex-envelope 표본**이라는 표현이 정확하다.

| 저장 component cleanup metadata | 상부 component | 주 component |
|---|---:|---:|
| foreground area (px) | 113 | 1810 |
| bbox [x,y,w,h] | [465,193,42,6] | [434,213,78,29] |
| retained | True | True |

원 cleanup은 largest×0.01=18.1px 이상을 유지하므로 상부 113px도 남았다. raw/clean foreground는 모두1923px이고 제거된 픽셀은0, component는2개다. mask 단계는 hole-fill/convex-hull/morphology를 하지 않았다고 저장돼 있다. **후속 global-outline 단계가 두 retained component를 함께 convex hull로 묶는다.** 저장 foreground hull 넓이는2788.5px²다. 이 기록은 상부 표본의 생성 위치와 envelope 연결을 설명하지만 그113px의 실제 의미가 파렛트/배경/그림자 중 무엇인지는 확정하지 않는다.

source의 경로·전체 SHA·행 범위, manifest/seal/current opaque bytes 일치와 원 observation의 모든 key/비배열 metadata는 [PROVENANCE_EVIDENCE_V2.json](PROVENANCE_EVIDENCE_V2.json)에 있다. 원 full observation의 점/선/polygon 배열은 기존 EVIDENCE에 보존돼 있다. 현재 mask/model/source/실험은 변경하지 않았고 추가 threshold/grid/NN 추론도 없다.
