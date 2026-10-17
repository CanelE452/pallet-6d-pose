# E0 관측·대응·공분산 계약 감사

**판정은 `CONFIRMED_BUG=0`, `OBSERVATION_MODEL_MISMATCH`, 의미 owner의 `INSUFFICIENT_EVIDENCE`다.** 저장된 238영상/1,252개 B/I 기록에서 Q·native 기하, CAL override 전달, whitening, full/partial 철회에 단순 구현 오류를 확인하지 못했다. 따라서 기존 adapter 재적용을 오류 수정으로 포장하거나 새 E 성능을 만들지 않았다. PRIMARY192 비교 완료와 별개로 C2 수정 경로는 `NO_SUPPORTED_CHANGE`다.

원 결과를 재검산하는 Phase A 통과 후 원 typed packet·원 archive·CAL 입력·기존 소스만 읽었다. 새 학습, 이미지 추론, 선 추출, covariance fit, PnP/최적화, 후보 재점수/선택은 없다. 별도 진단 그림에는 저장 관측과 기존 참조를 표시했다. CAL46은 원 학습/교정에 사용한 네 세션이고 독립 시험이 아니다.

## 무엇을 실제 관측으로 소비하는가

원 E0의 입력은 `TYPED_PACKETS.jsonl.gz/solver_input`의 native XYZ/K/상태/visible/domain/U/partial/outline/config이며, `activation`의 실제 full/partial 분할을 사용한다. `registered_family`가 같은 XYZ/Q를 제공한다. `provenance`는 원 보정 journal, 원 RGB/mask, 원 archive 행과 SHA를 연결한다. packet의 `e0_pose`는 원 결과이며 새 관측이나 prior로 대체하지 않았다.

| 상태·기록 | 원 소비 의미 | 감사 결과 |
|---|---|---|
| S | 영상점/선/지역 질의 없음; PnP 모델 투영·전체 depth guard만 가능 | B/I 1,252기록에 S 및 최초 hidden 0 |
| full B/I corrected | 원 예측과 RGB 선 교점의 보정 결과. 기본 v2는 raw와 교점을 blend하며 독립 RGB 센서점이 아님 | 실제 `source`로 fallback과 구분 |
| `VISIBLE_BACKBONE_FALLBACK` | N3+SubPix 점을 상속. `image_measurement=true`라는 저장 플래그만으로 RGB 교점이라 부르지 않음 | 실제 소스별 분리 집계 |
| partial B/I | 원 full 좌표/Σ는 archive·seed만 남고 final 2D 비용에서 철회; 1~2 normal + finite tangent interval | CAL75, PRIMARY341 모두 full과 교집합0; typed 원본 archive 숫자 exact |
| O/T | 위치영역 hard 제약; 정밀 point likelihood 없음 | CAL O9/T0, PRIMARY O124/T0. T의 새 효과 주장 없음 |
| U | 기존 nonS 예측 중심의 넓은 ellipse hard 제약; 정밀 점·line·canvas/visibility 비용 없음 | CAL24, PRIMARY198 |
| whole outline | 원 마스크의 convex envelope와 등록 cuboid 투영 hull 사이 양방향 arc 평균 거리 | CAL45/46 지원, PRIMARY192/192 지원. 마스크 지원은 실제 외곽·가시성 증명이 아님 |

소스 근거: [visible_bi_correction_v2.py:101](../../../scripts/research/pallet_visibility_curriculum_20261013_v1/visible_bi_correction_v2.py#L101), [typed partial 철회:100](../../../scripts/research/pallet_visibility_curriculum_20261013_v1/statewise_partial_BI_u_solver_v4_13.py#L100), [실제 full 제외:149](../../../scripts/research/pallet_visibility_curriculum_20261013_v1/statewise_partial_BI_u_solver_v4_13.py#L149), [whole-mask 관측:85](../../../scripts/research/pallet_visibility_curriculum_20261013_v1/statewise_partial_BI_outline_solver_v4_13.py#L85).

partial 한 corner의 두 normal은 `||r||²`를 묶어 radial pseudo-Huber 하나로 계산한다. 두 선을 독립 corner 두 개처럼 더하지 않는다. partial reliability를 한 번 곱하고 `boundary_factor_weight` metadata를 다시 곱하지 않는다. full BI는 점별 reliability를 곱한 radial 손실의 **합**, outline은 길이 정규화된 양방향 **평균**이다. 이 결합은 모델링 가정이며 Σ 누락이나 중복 계수라는 단순 산술 오류로 확인되지 않았다. normal variance 범위는 CAL11.088~472.749px², PRIMARY11.150~360.842px²이며 total-error coverage로 교정된 독립 선 잡음이 아니다. [normal grouped loss:108](../../../scripts/research/pallet_visibility_curriculum_20261013_v1/statewise_boundary_normal_solver_v4_4.py#L108), [full BI:164](../../../scripts/research/pallet_visibility_curriculum_20261013_v1/statewise_uncertain_solver_v4_bias.py#L164).

## 원 CAL adapter와 현재 override

원 `final_adapter_CAL46/MODEL.json` SHA256은 `e3fe7030ab90d7bcf0b4821137792d4914a6cb0a13f1858e73715ec634f2ff6e`다. 이 모델은 **회복 전** `fresh_observations_v2/CAL_OBSERVATIONS.jsonl.gz`(SHA `54240812a9354ea135021d1699808b60f9b307b6cca9a14ecfcf74c7f8520244`)에서 학습됐다. 원 CAL46 전체267점 중 고정 native 기하가 compatible인44영상의254점을 covariance fit에 사용했다. incompatible 두 영상 `wood_184309:000432`, `000492`의13점은 frame과 annotation 진단에 보존된다. 전체 human DIRECT254/SELF13과 fit eligible254는 우연히 같은 수이며 같은 모집단이 아니다. fit eligible는 DIRECT242/SELF12다.

| CAL 세션 | 영상 | 원 B/I 전체 / compatible | B / I | fallback / B RGB / I RGB |
|---|---:|---:|---:|---:|
| eval_cad | 18 | 104 / 104 | 96 / 8 | 39 / 65 / 0 |
| eval_night08 | 3 | 6 / 6 | 6 / 0 | 3 / 3 / 0 |
| eval_outside | 5 | 27 / 27 | 24 / 3 | 5 / 20 / 2 |
| wood_184309 | 20 | 130 / 117 | 119 / 11 | 68 / 58 / 4 |

| 원 fit group | compatible 점 / 세션 | 추가 floor px² | reliability |
|---|---:|---:|---:|
| B corrected | 145 / 4 | 39.850428 | 0.981502 |
| B fallback | 88 / 4 | 83.986218 | 0.961476 |
| I corrected | 6 / 2 | 35.324018 | 0.908537 |
| I fallback | 15 / 3 | 58.111134 | 0.808457 |

원 수식은 `fitΣ = original conditionalΣ + floor(group) × clipped(nonS_scale/reference_scale)² × I`다. 조건부 Σ와 floor 모두 원영상 px², scale ratio와 reliability는 무차원이다. reliability는 CAL human DIRECT 빈도를 session/frame/point 균형·원 pseudo-count로 부드럽게 반영한 값이며 검출 확률·물리 coverage 확률이 아니다. 불확실성 모듈이 없던 상태가 아니다. [group/등록 정합:100](../../../scripts/research/pallet_visibility_curriculum_20261013_v1/statewise_uncertainty_calibration_v4.py#L100), [원 fit:273](../../../scripts/research/pallet_visibility_curriculum_20261013_v1/statewise_uncertainty_calibration_v4.py#L273), [runtime 적용:315](../../../scripts/research/pallet_visibility_curriculum_20261013_v1/statewise_uncertainty_calibration_v4.py#L315).

이후 [fixed_fit_measurements:128](../../../scripts/research/pallet_visibility_curriculum_20261013_v1/boundary_partial_observations_v4_4.py#L128)는 **“Use recovered image coordinates with the original fixed fit scales.”**라고 명시하며 회복된 xy/source/conditionalΣ와 원 fitΣ/reliability를 조합한다. 이번 1,252기록에서 원 fitΣ/reliability 복사는 모두 exact이고, 복사 전 원 adapter의 재계산 차이도0이었다. 이 설계를 우발적인 override 누락으로 분류하지 않는다.

| 집단 | B/I 기록 | 현재 full / partial archive | 소스·조건부Σ 변경 | 변경 영상 | 변경 중 full / partial |
|---|---:|---:|---:|---:|---:|
| CAL46 | 267 | 192 / 75 | 10 | 9 | 10 / 0 |
| PRIMARY192 | 985 | 644 / 341 | 29 | 22 | 27 / 2 |

변경39점의 실제 소스는 `VISIBLE_BACKBONE_FALLBACK→B_ACTUAL_BOUNDARY_RECOVERY`다. 원 corrected 학습145점은 `B_TARGET_BACKGROUND_LINES`이며 새 회복 소스의 total-error calibration은 아니다. 현재 관측으로 같은 adapter를 다시 적용하면 이39점에서 Σ/reliability가 바뀐다. 하지만 새 소스를 원 corrected 그룹과 같은 잡음 분포로 간주한다는 새로운 가정이다. 새 점을 더 믿으면 좋아진다는 근거로 쓸 수 없다. 모든 원 fitΣ−현재 conditionalΣ는 PSD였고 음의 고유값 위반0이다.

전39변경의 원/현재 소스·conditional/fit σ·hypothetical adapter σ/reliability는 [CAL_PROVENANCE_ALL39_TRANSITIONS.csv](CAL_PROVENANCE_ALL39_TRANSITIONS.csv), 나머지1,213불변 기록을 포함한 전량은 [E_POINT_CONTRACT_ROWS.csv](E_POINT_CONTRACT_ROWS.csv)다. 원 CAL source/state/session별 점수·Σ·잔차의 평균/표본SD/median/P90/P95/max는 [CAL_PROVENANCE_DISTRIBUTIONS.csv](CAL_PROVENANCE_DISTRIBUTIONS.csv), 전 그룹 count는 [CAL_PROVENANCE_SOURCE_COUNTS.csv](CAL_PROVENANCE_SOURCE_COUNTS.csv)다. covariance 단위가 px²인지 확인하는 검사는 whitening `L=inv(cholesky(Σ))`, `LᵀL=Σ⁻¹`의 최대 차이4.163×10⁻¹⁷로 통과했다.

직전 결합 문제의 직접 사례 `wood_night_01:030898`은 **이번39점 전이와 무관**하다. 실제 full5점은 B RGB k0/2/3/4, I RGB k1이고 fallback→recovery0이다. total σmax는 각각3.580/3.521/3.532/3.543px와3.379px이며 기존 adapter를 현재 관측에 재적용해도 전5점이 그대로다. 따라서 이 재적용안은 직전 성공/실패 후보의 full_BI 순위 충돌을 바꾸지 않는다. 그 현상만으로 이 수정안을 선택하는 기전 근거도 없다.

## 416px를 실제 잘못 찍은 점으로 단정하지 않는다

`wood_184309:000372/k3`에서 원 N3 점은(706.959,667.021), RGB 회복은(702.594,656.397)이다. 직접 원1280×720 RGB를 열어 본 결과 두 점은 앞쪽 아래 코너 부근이다. 기존 annotation의 근처 reference2=(703,662)까지 거리는5.618px다. 그러나 고정 metadata는 k3→reference6=(1049.727,426.296)으로 연결한다. 해당 기존 annotation/파생 참조와의 잔차가416.470px다. **이는 416px 물리적 위치 오검출의 증명이 아니라 채널·W/D·owner 가설과 참조 등록 사이의 큰 불일치**다. 가까운 reference2로 정답을 바꾸거나 selector를 gating하지 않았다.

오래된 N3_SUBPIX journal의 점/R_cf/R_physical/중심이 ORACLE_CAL_INPUTS→fresh_CAL 입력에 정확히 전달됐고, 고정 `corner.branch=1`의 target permutation이 metadata mapping `[5,4,7,6,1,0,3,2]`와 일치한다. RAW의 branch를 N3에 잘못 복사한 증거는 없다. 원 reference alignment는 unordered 집합 검사가 아니라 **per-index 3D** 검사다. phase0 gap0.994032m, phase1 gap1.777×10⁻¹⁶m로 phase1이 고정된다. phase0의 k3 투영(467.150,419.023)도 관측 옆 reference2와 다르므로 단순 C2 위상 반전 하나로 회복된다고 할 수 없다. 초기 모델 자신의 k3 투영은(710.744,670.104)이다. compatible44는 등록된 기하 대응이 수학적으로 맞는다는 뜻이며 영상 채널의 실제 물리 owner가 맞다는 인증이 아니다.

이 점의 conditional σmax1.400px에 비해 현재 total σmax10.968px다. adapter 재적용은 σmax7.560px와 reliability0.981502로 강화한다(현재0.961476). 큰 mapped 잔차의 원인이 owner인지 등록/참조인지 분리되지 않은 상태에서 이를 정밀하고 인증된 물리 점으로 더 강하게 가중할 근거는 없다. 원 annotation·참조는 `independent_physical_pose_truth=false`이고 수정하지 않았다. 원/overlay를 실제 열어 확인했으며 [그림](audit_gallery/CAL_OWNER_PHASE_WITNESS.png)과 [전량 숫자·원 N3 SHA](audit_gallery/CAL_OWNER_PHASE_WITNESS.json)에 남겼다.

## native/Q·공통 RGB·남는 한계

원 [두 WD family](../../../scripts/research/pallet_visibility_curriculum_20261013_v1/statewise_two_wd_family_v4_9.py#L30)는 `SWAPPED_WD`에서 native extent를 바꾸고 `Q'=RY90×Q`, `R'=R×RY90ᵀ`를 명시적으로 운반한다. 같은 채널의 physical owner를 바꾸는 별도 가설이며 단순 번호 변환을 누락한 오류라고 판정하지 않았다. 최종 선택 family의 unordered 물리 vertex bijection gap 최대1.110×10⁻¹⁶m, proper Q·XYZ 바인딩 검사는 통과했다. CAL6영상/PRIMARY32영상이 SWAPPED_WD를 선택했고 그 안의 B/I 기록30/143개에서 원 family와 채널 physical owner가 다르다. 저장 원장이 명시한 `SEMANTIC_OWNER_UNVERIFIED`를 유지한다.

full 점과 partial normal은 같은 이미지·마스크와 초기 geometry로 안내된 RGB 선을 사용한다. outline도 같은 마스크의 hull이다. 기본 full 점은 N3와 RGB 교점의 blend라 upstream과도 상관될 수 있다. 한 corner full/partial 이중 비용은 철회로 막지만, 서로 다른 corner·선·outline 사이의 공동 잡음을 추정하거나 독립성을 인증하지는 않는다. 같은 프레임에서 관측을 재사용한다는 코드 근거는 분명하지만 다른 source SHA나 표본 개수만으로 독립 센서 정보라고 계산하지 않는다. 이 감사는 실제 공유 픽셀의 상관계수나 물리 edge owner를 새로 추정하지 않았다.

이전 radial total-error opt-in은 후속 V47 좌표를 CAL에서 다시 분석하고 bias/scatter/backoff까지 포함했다. 그 원 LOGO 결과는 reference norm 평균17.4896→17.5322px, 별도 DIRECT annotation 평균15.4759→15.5846px로 악화했고 채택되지 않았다. `CAL_radial_total_error_v4_9/INTERPRETATION_KO.json`, `IMPLEMENTATION_RECEIPT_V1.json`을 읽었으며 같은 시도를 새 방법으로 반복하지 않았다. 이번39점의 단순 adapter 재적용과 수학적으로 같은 방법은 아니지만, 이미 교정된 잡음 모듈의 존재와 소규모 네 세션 전이 한계를 보여 주는 선행 반례다.

감사 wall20.016초/CPU20.011초, peak RSS1,285,648KiB다. 추가 설명 그림/기하 추적은 별도 약1초 수준의 읽기·투영 연산이며 PnP/최적화가 아니다. 새 학습·신경망·후보 생성·새 E192 선택 성능은0회다. 원 E0·N0·두 PoseFix 경로의 공정 비교는 [REPORT_KO.md](REPORT_KO.md), 수정 판정은 [MECHANISM_AND_EDIT_PROPOSAL.md](MECHANISM_AND_EDIT_PROPOSAL.md)에서 분리한다.
