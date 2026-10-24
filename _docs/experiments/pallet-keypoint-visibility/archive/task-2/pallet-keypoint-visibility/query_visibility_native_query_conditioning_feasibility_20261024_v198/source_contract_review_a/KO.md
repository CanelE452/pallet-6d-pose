# Native ID·원 q8 문맥의 입력 계약 검토

**nativeID-only는 기존 네 cache 모두에서 계약상 가능하다.** 원 q8도 같은 scene/frame/control group의 원 좌표와 availability로 보존하는 코드가 있다. 현재 head는 이를 입력받지 않는다. **SELF jitter 뒤 새로 추론한 동시 q8은 없다.**

이번에는 기존 코드 8개만 읽고 해시를 기록했다. 실제 source metadata·배열·헤더·픽셀·가중치·점수·평가 결과는 열지 않았다. 아래 분모는 코드의 선언/검증 조건이며 새 전수 집계가 아니다.

| Population | 행수 | 저장 native ID | 저장 q8의 역할 |
|---|---:|---|---|
| SYNTH SELF_SCENE15 | 61,440 | 원 slot 유지 | 512 원 scene의 `original_input_metadata8`. 각 행의 crop q/L만 변환한다. q8은 불변 원문맥이며 jitter 후 initializer 출력이 아니다. |
| SYNTH BEFORE_ANCHOR4 | 4,096 | 원 slot 유지 | 원 BEFORE q8을 네 image variant에 고정한 anchor 문맥. image-original q8과 구분한다. |
| PUBLIC_TRAIN1480 | 1,480 | 원 slot 유지 | `sam_group_id=PUBLIC_FRAME_<frame>`의 원 8행 q/L/ID/availability. |
| CF3816 | 3,816 | 원 slot 유지 | `pair_id/variant`별 원 8행과 packet q8. 합성 RGB에서 새로 초기화한 값은 아니다. |
| DIRECT_CONTROL3144 | 3,144 | 원 slot 유지 | 같은 `group_id`의 8행과 `metadata8`; 원 source packet q8/L8/availability8 exact join. |

SELF의 기반은 원 4,096행/512 scene이다. nominal 4,096행도 15상태 안에 포함된다. 원 U를 보존하는 코드 조건은 SELF 1,200, BEFORE-anchor 64, PUBLIC 194, CF 360, DIRECT 112다. q8 중 unavailable slot은 사라지지 않으며 별도 마스크가 필요하다.

`INPUTS10`에는 행별 `native_slot/query_xy/L/original_query_available`가 있지만 `[8,2]` q8 tensor는 없다. 같은 원본 identity로 기존 ROWS/scene/group metadata를 join해야 한다. 현재 [old gather](../../query_visibility_query_jitter_paired_source_training_20261024_v183/source_fit_preparation/train_jitter_paired_source.py)의 234–246행과 [grouped trainer](../../query_visibility_source_group_split_direct_end_training_20261024_v194/grouped_fit_preparation_b/train_grouped_direct_end.py)의 62–71행은 RGB/valid와 선택적 SAM만 head에 넘긴다.

코드 근거:

- [원 raw producer](../../query_visibility_local_support_raw_head_20261024_v181/raw_cache_preparation/build_source_raw96_cache.py) 123–140, 226–252, 274–304행: 같은 scene/frame/pair-variant의 원 native8 q/L/availability join과 ROWS 저장.
- [증강 producer](../../query_visibility_query_jitter_paired_source_training_20261024_v183/augmentation_preparation_a/build_source_augmentations.py) 49–80, 155–159, 223행: 원 `original_input_metadata8` 저장, active crop만 변환, 원 scene/anchor identity와 `scene_metadata` 보존.
- [증강 compositor](../../query_visibility_query_jitter_paired_source_training_20261024_v183/compose_augmented_sam_inputs.py) 123–143, 160–164, 184행: SELF의 원 q8 prompt와 jitter crop q/L 분리, BEFORE anchor 사용.
- [DIRECT producer](../../query_visibility_source_group_split_direct_end_training_20261024_v194/direct_cache_preparation_a/build_expanded_direct_control_cache.py) 72–87, 93–111, 179–194행: 원 synthetic/real all8 authority와 packet join, group/control key 및 metadata 저장.

SELF의 다른 증강/native mate를 모으거나 모든 원 q에 active jitter를 더해 ‘동시 initializer8’을 만들면 기존 계약 밖이다. 가능한 범위는 **원 scene q8을 불변 문맥으로 명명하고 현재 crop q/L과 분리해 쓰는 것**이다. 또 active 원 anchor와 crop q/L을 함께 주면 dx/dy/scale이라는 증강량이 드러날 수 있다. active slot 노출/마스크와 상대좌표 기준은 prospective 계약에서 먼저 정해야 하며, 여기서는 선택하지 않았다.

numerical native slot은 deployment 입력 식별자다. 그 자체가 selected instance나 물리 끝부위 대응 인증은 아니며, 새 vertex certificate를 필수조건으로 올리지도 않는다. GTend/enclosure/pose/참조좌표/PnP/GT-best permutation은 조건 입력이 될 수 없다. 중첩 원 metadata에는 감독 정보가 있으므로 허용 scalar만 명시적으로 골라야 한다.

다음 한 단계는 별도 prospective 정의 후 **기존 metadata의 exact8 identity join 검증**이다. 원 q/L·null·U, SELF 원문맥 대 crop 좌표, BEFORE anchor 대 image 원점, source component TRAIN 역할을 검사하면 된다. 픽셀·타깃·점수·가중치나 새 initializer 실행은 필요하지 않다. 이번 검토는 모델 구조를 고르거나 학습·개선을 입증하지 않는다.
