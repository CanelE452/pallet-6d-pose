# SOURCE SYNTH native target 코드 계약 검토

기존 합성 코드에는 **선언된 native 대응과 anchor 투영, 점별 D/E/S/U**를 별도로 계산할 근거가 있다. 현재 DIRECT 학습 cache는 그 native target을 **−1로 유지**한다. finite-end의 0/1/0이나 crop-centre owner를 native 가시성으로 복사해서는 안 된다. 이번 PASS는 코드 추적 완료이며 새 GT/이미지 계산이나 전이 성능 인증이 아니다.

| 구분 | 확인한 정의 | 보장하지 않는 것 |
|---|---|---|
| native → canonical | `c=permCF_to_canonical[k]`. signed-quarter 회전 후 선언 corner의 정확 일치로 만든 input/CF slot→canonical 대응 | q-nearest/GT-best/PnP 결과가 아니며, 한 input slot이 항상 같은 canonical physical corner라는 뜻도 아님 |
| 등록 참조 | 새 procedural geometry의 `native_anchor3D[c] == native_witness_points[c][0]`; `projected_reference_xy=project[0]` | 실제 query, region 중심 또는 공개 영상의 physical native gold와 동일하다는 보장 없음 |
| 점 가시성 | `target_witness_states[0]`/`combined_witness_states[0]`: reference 거리에서 expected part D, 앞 foreign999 E, 앞 target surface S, tie/miss/불명확 U | whole-region `state`와 동일하지 않음 |
| region/end 감독 | facing faces·active witness 전체를 쓰는 region state, V180 비배타 support2, V189 inner0.2 finite-end는 서로 다른 정의 | 이들을 기존 native D/E/S 또는 centre-owner로 alias하지 않음 |
| 원 입력 | 원 TRAIN plan q/L/flags, exact V90 F64 bilinear/uint8-round crop와 all8 보존 guard | finite q가 곧 available이거나, 원 crop 안에서 해당 끝의 증거가 충분하다는 뜻 아님 |

## 정확한 코드 근거

- [V129 quarter_mapping / camera / config](../../query_visibility_registered_native_diversity_20261023_v129/render_preparation/corpus_math.py#L14)는 signed-quarter 기준의 대응과 이상적 anchor 투영을 만든다. canonical6/7과 input native6/7을 같은 역할로 간주할 수 없다. 이번에는 실제 config 행/coverage를 다시 열지 않았다.
- [V127 make_mesh / first_hit / classify_hits / region_teacher](../../query_visibility_spatial_stencil_control_20261023_v127/registered_native_teacher_feasibility/build_registered_pallet.py#L67)는 named primary part·anchor·37 witness와 점별 ray 판정을 정의한다. region state는 target-alone S 우선, 두 개 이상 facing face 및 active witness 전체 판정, 영역 OOF 등 추가 조건을 갖는다. `projected_reference_xy`는 점0 투영일 뿐 최종 region state의 의미를 바꾸지 않는다.
- [V189 finite_end / direct_targets](../../query_visibility_sampled_domain_direct_end_20261024_v189/direct_end_training_preparation_c/synthetic_direct_end_candidate.py#L17)는 `anchor + .2*(patch-anchor)`라는 별도 finite-end convention이다. [V189 population / crop / proof / packet](../../query_visibility_sampled_domain_direct_end_20261024_v189/direct_end_training_preparation_c/prepare_and_generate_synthetic_sampled_end.py#L97)는 원 q/L을 보존하고 native/quality/owner target을 −1로 저장한다.
- [V194 source_population / load_pixels](../../query_visibility_source_group_split_direct_end_training_20261024_v194/direct_cache_preparation_a/build_expanded_direct_control_cache.py#L80)는 원 all8 q/L/native/availability와 별도 direct authority를 검산한다. [V180 support operator](../../query_visibility_frozen_sam_source_baseline_20261024_v180/source_end_supervision_preparation_a/source_end_support.py#L139)는 자기 가림·모호성을 UNKNOWN으로 보존하며 native target을 만들지 않는다. [V90 original crop](../../query_visibility_public_cue_inputs_20261023_v90/producer/spatial_input_math.py#L74)는 실제 RGB 4-tap bilinear sampler다.

## 다음 SOURCE-only 확인 한 가지

사전 정의한 TRAIN-only join에서 **원 q/L/native/U를 그대로 둔 채** `k→perm[k]`, anchor 투영, point0 target/combined D/E/S/U, whole-region state, 그리고 원 raw96의 해당 feature 증거/좌표 차이를 서로 다른 열로 확인하면 된다. 저장된 point field가 없으면 같은 선언 geometry의 point-ray 감독을 별도로 사전 정의해 계산할 수 있다. crop에 증거가 부족한 현상을 외부 E로 바꾸거나, GT로 q를 옮기거나, finite-end/owner 라벨을 native로 복사하지 않는다.

새 procedural 자산 내부의 명시적 등록 권위는 활용할 수 있다. 공개 assistantweak 관찰에 완벽한 mesh/vertex 인증을 추가 선결조건으로 요구할 필요는 없다. 동시에 작은 공개 witness 표본이나 합성 exact registration은 SQUARE·다양한 pose·자연 가림 전이를 증명하지 않는다.

이번 읽기는 코드8개·프로토콜2개·코드 포인터용 완료 metadata1개와 root scope뿐이다. 실제 config/GT/JSONL 행, NPZ/NPY/RGB·헤더·자산 hash, ray/투영/렌더 실행, 모델·학습·PnP·평가/보호 셋·SOURCE HOLDOUT body는 0이다. 실제 인구 coverage와 upstream query 생성기는 이번 좁은 코드 경로에서 재감사하지 않았다. 원본과 기존 동결 산출물은 수정하지 않았다.
