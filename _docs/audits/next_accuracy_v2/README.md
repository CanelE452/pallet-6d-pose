# next_accuracy_v2 — 정리된 GT 에서 저앙각 supervision 이 실제로 localisation 을 움직이는가

지시문(§0~§37)에 대한 작업.  기준 커밋 `ddc0770`
(지시문이 전제한 `f2b2739` 와의 차이는 history 문서 248줄뿐 — 전제 유지).

읽기 전용 감사를 먼저 끝내고, 그 결과가 허락한 실험 하나만 돌린다.
`accuracy_root_cause_v1` 은 건드리지 않는다.

## 문서

| 문서 | 지시문 절 | 상태 |
|---|---|---|
| `DATA_CONTRACT_AUDIT.md` | §3 · §4 · §5 | 완료 |
| `DERIVED_DATA_AUDIT.md` | §6 | 완료 |
| `EVAL_AND_SYNTHETIC_LABEL_AUDIT.md` | §3 보강 | 완료 |
| `SOURCE_DIVERSITY_AUDIT.md` | §23~§25 근거 | 완료 |
| `GT_PARTITION.md` | §8 · §9 · §10 · §31 · §32 | 완료 |
| `CORRECTED_REAL_FT.md` | §11 · §12 | 완료 |
| `ELEVATION_COMPOSITION_ABLATION.md` | §13 · §14 · §15 | 완료 |
| `LOW_ANGLE_PL_QUALITY.md` | §17 · §18 | 완료 |
| `SELFTRAIN_DECISION.md` | §19~§22 | 완료 |
| `SYNTHETIC_DECISION.md` | §23~§25 | 완료 |
| `FINAL_DECISION.md` | §34 | 완료 |

기계 판독용 산출물은 `data/pallet/results/next_accuracy_v2/`,
스크립트는 `scripts/research/next_accuracy_v2/`.

> ⚠️ **기계 판독용 산출물은 git 에 들어가지 않는다.**  `.gitignore:11` 의 `data/**` 가
> 막는다 — 실측: `data/pallet/results/next_accuracy_v2` 추적 파일 0,
> `accuracy_root_cause_v1` 도 0.  저장소 관례를 따른 것이지 누락이 아니다.
> 따라서 아래 문서들이 가리키는 `data/pallet/results/...` 경로(METHOD_LOCK.json 포함)는
> **이 머신에만 있다.**  다른 머신에서 재현하려면 `scripts/research/next_accuracy_v2/`
> 의 스크립트를 다시 돌려야 한다 — 전부 읽기 전용이라 재실행 가능하다.
> 학습 산출물(`challenge/yolo_pose_one_model/next_accuracy_v2/`, 158MB)도
> `*.pt` 규칙으로 제외된다.

## 지금까지 뒤집힌 전제

지시문이 예상한 것과 실측이 다른 지점만 적는다.

1. **§4 의 sentinel 결함은 실재하고 재현됐지만 반경이 작다.**
   학습 소스 851장 중 2 프레임(6 keypoint).  합성 R0 60,000장에 0개,
   정본 평가셋 140장에 0개.  두 독립 경로가 같은 6개를 가리켰다.
2. **정본 평가셋의 `projected_cuboid` 규약 위반 0/140.**
   `gt_v2_canonical` 은 다른 프레임이 아니라 같은 140장의 사본(최대오차 0.00 px).
   → 지금까지의 평가 수치를 이 축 때문에 재해석할 필요가 없다.
3. **합성 GT 에 `[-1,-1]` sentinel 이 0개다.**
   `prepare_yolo_pose.py` 주석의 "renderer writes -1,-1" 은 이 데이터셋에서 0회.
   음수 좌표 16,551점은 상수가 아니라 실제 화면밖 투영(연속분포)이다.
   → fallback 경로는 고치지 않았다.
4. **파생본이 옛 규약으로 되돌아가는 방향이 반대였다.**
   기존 `live_gt_v4`/`v5` 의 flip 라벨은 696/696 정확한 미러(오차 0.0 px)다.
   그 데이터셋이 `f2b2739` 이전에 빌드돼 `projected_cuboid` 를 읽었고, flip 산출물의
   그 필드는 올바르게 미러돼 있었기 때문이다.  틀린 것은 **base 프레임** 쪽이다
   (696장 중 321장 = 46.1% 가 위반본에서).
   → "지금 재빌드하면 flip 이 틀어진다" 가 맞고 "기존 flip 이 틀렸다" 는 틀리다.
5. **§11 의 baseline 은 이미 한 번 돌았다** —
   `challenge_c4_track/clean_label/stage1_indexed`.  그러나 평가가 interleave(같은 세션)
   val 155 라 §11 이 금지한 구조다.  촬영단위 split 으로 다시 세웠다.
6. **§13 의 균형 arm 178 은 성립하지 않는다.**
   train pool 의 8-15도 층은 handheld 96.1%, <8도 층은 42.9% 다.
   두 arm 이 앙각만큼이나 촬영방식으로 달라진다.
   → arm 을 handheld 2 세션 안으로 좁혀 137장으로 바꿨다
   (세션 구성까지 글자 그대로 동일).

## 코드 변경

라벨 계약 (§4·§5)
- `challenge/yolo_pose_one_model/scripts/prepare_yolo_pose.py`
  — `load_kps` 가 `(x, y, known)` 을 돌려주고 `to_line` 이 상태를 본다.
    좌표 값으로 상태를 표현하지 않는다.
- `scripts/stage0/model_compare/mc_build_yolo_broad.py` — 3-튜플 대응.

파생 데이터 (§6)
- `challenge/scripts/dataset/keypoint_annotations_transform.py` (신규) — 공용 변환 헬퍼.
- 생성기 6개가 이걸 쓴다: `gen_truncation_crops.py` · `pad_truncation_crops.py` ·
  `augment_ratio_robust.py` · `augment_dataset.py`(hflip 인덱스 순열 추가) ·
  `make_pseudo_gt.py` · `gen_flip_noise_aug.py`.
- `challenge/yolo_pose_one_model/scripts/prepare_yolo_pose_from_live_gt.py`
  — `assert_derived_is_current()` 로 낡은 파생 폴더를 **거부**한다.

테스트
- `test_keypoint_field_contract.py` — 옛(결함) 계약을 고정하던 3개 갱신 (6 케이스).
- `test_label_contract_end_to_end.py` (신규 7) — 전체 변환 경로.
- `test_derived_artifact_invariants.py` (신규 5) — **디스크 산출물** 검사 + 빌더 가드.
  낡은 `flip_noise_aug_livegt` 에서 851/851 을 잡는 것으로 이빨을 확인했다.
- `test_derived_generators_preserve_keypoints.py` (신규, 함수 8 -> 18 케이스) — AST 검사.

## 새로 만든 데이터

- `challenge/data/03_derived/flip_noise_aug_livegt_v2` — flip/noise 재생성 1,702장.
- `challenge/yolo_pose_one_model/datasets/live_gt_contract_v2` — 정본 라벨, 촬영단위 split.
- `challenge/yolo_pose_one_model/datasets/live_gt_legacy_v2` — 대조군(학습 라벨만 legacy,
  val 라벨은 정본으로 통일).

기존 폴더는 하나도 덮어쓰지 않았다.
