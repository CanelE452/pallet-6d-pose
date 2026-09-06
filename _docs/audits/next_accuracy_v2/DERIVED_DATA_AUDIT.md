# 파생(derived) real 데이터 — keypoint 필드 감사

작성 2026-09-06 · 읽기 전용 감사(파일 수정·재생성 없음)

**[목적]** 파생 데이터 생성기가 어느 keypoint 필드를 읽고, 결과 JSON 에 무엇을 보존하는지 전수 확인.
**[판정지표]** 생성기마다 Q1~Q4 를 코드 라인 근거로 답하고, 디스크 산출물을 실측해 대조.

각 문장 끝의 `[확인]` = 코드/산출물에서 직접 봄, `[추정]` = 해석.

---

## 0. 기준 사실 (재확인)

- `challenge/yolo_pose_one_model/scripts/prepare_yolo_pose.py:61-76` — `load_kps()` 는
  `objects[0].keypoint_annotations` 가 리스트이고 길이 ≥ 9 면 **그것만** 쓰고 `return` 한다.
  그 조건이 아닐 때만 `projected_cuboid` (+ `projected_cuboid_centroid`) 로 떨어진다. [확인]
- LR 순서 스크린(camera-facing 0123 위반 판정) 규칙은
  `scripts/research/accuracy_root_cause_v1/real_label_audit.py:322` 의
  `not (u[0] < u[1] and u[3] < u[2])` 를 그대로 썼다 (u = 앞면 4점의 x). [확인]
- 이 감사에서 위 규칙으로 `live_capture_gt` 851장을 전수 재현했다:
  `projected_cuboid` 위반 **198/851 (23.3%)**, `keypoint_annotations` 위반 **0/851**.
  `REAL_LABEL_AUDIT.md` 의 수치와 일치한다. [확인]
- real GT 폴더별 `keypoint_annotations` 보유 (전수):
  `manual_gt` 0/574, `eval_canonical` 0/105, `gt_v2_canonical` 140/140,
  `live_capture_gt` 851/851, `pseudo_gt` 0/38, `augmented` 0/275 → 합 991/1,983. [확인]

---

## 1. 생성기별 표 (Q1~Q4)

| 생성기 (파일:라인) | Q1 읽는 필드 | Q2 `keypoint_annotations` 보존/변환 | Q3 `visibility`/`in_frame`/`source`/`reason` 갱신 | Q4 `load_kps` 경로 |
|---|---|---|---|---|
| `challenge/scripts/dataset/gen_flip_noise_aug.py`:69-89 | **둘 다** — `projected_cuboid`:69, `keypoint_annotations`:79 [확인] | **보존 + 변환 코드 있음**(`:80-89` `x'=w-1-x`, `FLIP_PERM_8`+centroid). ★단 이 코드는 커밋 `f2b2739`(2026-09-06 13:05)에 들어갔고 **디스크 산출물은 2026-09-04 18:12 생성분**이라 반영 안 됨 [확인] | **아니오** — `dict(entry)` 얕은 복사 후 `xy` 만 바꾼다(`:83-86`). 네 필드 모두 원본 그대로 [확인]. flip 은 좌우 반전이라 `in_frame`/`visibility` 는 의미상 불변이지만 `source` 는 `manual_click` 으로 남는다 [확인] | `keypoint_annotations` 경로 (길이 9) [확인] |
| `challenge/scripts/dataset/gen_truncation_crops.py`:84-88, 264-285 | `projected_cuboid` + `_centroid` **만** (`get_keypoints`:86-87) [확인] | **보존 안 함** — `write_output`:266-274 가 6개 키(`class,name,visibility,projected_cuboid,projected_cuboid_centroid,dimensions_m` + 있으면 `pose_transform`)로 **새 dict 를 만든다** [확인] | 해당 없음(필드가 없다). `visibility` 는 오브젝트 레벨 스칼라를 그대로 복사(`:269`) [확인] | `projected_cuboid` fallback [확인] |
| `challenge/scripts/dataset/pad_truncation_crops.py`:65-67, 140-163 | `projected_cuboid` + `_centroid` 만 [확인] | **보존 안 함** — 새 dict(`:142-148`) + `pose_transform/cuboid/location/quaternion_xyzw` 만 pass-through(`:150-153`) [확인] | 해당 없음 [확인] | `projected_cuboid` fallback [확인] |
| `challenge/scripts/dataset/augment_ratio_robust.py`:94-98, 133-157 | `projected_cuboid` + `_centroid` 만 [확인] | **보존 안 함** — 새 dict(`:135-142`) + pose 계열만 pass-through(`:144-147`) [확인] | 해당 없음. docstring `:24` 이 "keypoint 인덱스를 절대 순열하지 않는다(flip 이면 순열해야 하지만 이 스크립트는 flip 을 안 한다)" 고 명시 [확인] | `projected_cuboid` fallback [확인] |
| `challenge/scripts/dataset/augment_dataset.py`:5, 50-54, 79-118, 151 | **`manual_kps`** (`:151`) — `projected_cuboid` 도 `keypoint_annotations` 도 안 읽는다 [확인] | **보존 안 함** — `write_gt`:99-115 가 `projected_cuboid`/`_centroid`/`manual_kps` 를 새로 쓴다 [확인] | 해당 없음. 화면 밖은 `[-1,-1]` sentinel 로만 표시(`:82-86`) [확인] | `projected_cuboid` fallback [확인] |
| `challenge/scripts/dataset/make_pseudo_gt.py`:292-304 | 읽지 않음 — 모델 추론 + PnP 재투영 결과를 쓴다 [확인] | **보존 안 함** [확인] | 해당 없음 [확인] | `projected_cuboid` fallback [확인] |
| `challenge/yolo_pose_one_model/scripts/prepare_yolo_pose_from_live_gt.py`:31, 100-127 | 직접 안 읽음 — `prepare_yolo_pose.one/load_kps` 를 import 해 쓴다(`:31`) [확인] | 해당 없음(YOLO `.txt` 라벨을 낸다) [확인] | 해당 없음 [확인] | 원본은 `keypoint_annotations`, `--crop-dir`/`--aug-dir` 로 더한 파생본은 그 파일이 가진 필드에 따름 [확인] |
| `challenge/yolo_pose_one_model/scripts/prepare_real_ft.py`:50-70 | `projected_cuboid` **만** (`load_kps_real`:56) — `keypoint_annotations` 를 아예 안 본다 [확인] | 해당 없음(`.txt` 출력) [확인] | 해당 없음 [확인] | 자체 loader 라 `load_kps` 를 안 탄다. **위반본을 읽는다** [확인] |
| `scripts/self_training_yolo/real_ft_v1/build_real_ft_dataset.py`:58 | `keypoint_annotations` **만** (없으면 KeyError) [확인] | 해당 없음(`.txt` 출력) [확인] | 해당 없음 [확인] | 자체 loader [확인] |

### ★ 핵심 비대칭

`gen_flip_noise_aug.py` 하나만 `keypoint_annotations` 를 보존한다. 나머지 파생 생성기는 전부
**새 dict 를 만들어** 그 필드를 떨어뜨리므로, 산출물이 다시 `load_kps` 를 탈 때 반드시
`projected_cuboid` fallback 으로 간다 — 원본에 규약을 지키는 `keypoint_annotations` 가 있었어도
파생 단계에서 소실된다. [확인]

---

## 2. 디스크 실측 표 (전수 — 표본 아님)

LR 열은 앞면 4점이 sentinel 이 아닌 프레임만 판정했고, sentinel 프레임 수는 괄호로 적었다.

| 폴더 | JSON 수 | `keypoint_annotations` 보유 | `projected_cuboid` LR 위반 | 비고 |
|---|---|---|---|---|
| `challenge/data/03_derived/flip_noise_aug_livegt` | 1,702 (flip 851 + noise 851) | **1,702 / 1,702** | 396 / 1,702 (23.3%) | 부모 198 위반 × 2 = 396, 정확히 일치 [확인] |
| `challenge/data/03_derived/truncation_crops_livegt` | 1,206 | 0 / 1,206 | 318 / 1,203 (sentinel 3) | 부모 851장 중 402장 × 3 crop [확인] |
| `challenge/data/03_derived/truncation_crops` | 485 | 0 | 0 / 450 (sentinel 35) | 부모가 `manual_gt` 계열 — 애초에 `keypoint_annotations` 가 없다 [확인] |
| `challenge/data/03_derived/truncation_crops_dope` | 46,119 | 0 (3장 파싱 실패) | 2 / 46,065 (sentinel 52) | 합성 [확인] |
| `challenge/data/03_derived/truncation_crops_synth` | 394 | 0 | 0 / 394 | 합성(`mixed_v8_train_*`) [확인] |
| `challenge/data/03_derived/truncation_crops_palletobj` | 748 | 0 | 0 / 747 (sentinel 1) | 합성(`train_palletobj_v1_*`) [확인] |
| `challenge/data/03_derived/_train_pallet07_aug` | 276 | 0 | 19 / 157 (sentinel 119) | `augmented` 와 같은 내용 + 원본 1장 [확인] |
| `challenge/data/03_derived/_train_capturepallet07` | 26 | 0 | 0 / 25 (sentinel 1) | [확인] |
| `challenge/data/03_derived/_train_manual_pseudo` | 64 | 0 | 34 / 63 (sentinel 1) | `make_pseudo_gt` 산출(추론 PnP) [확인] |
| `challenge/data/03_derived/_eval_real_gt_merged` | 219 (심링크) | — | — | **438개 전 항목이 끊긴 심링크.** 2026-08-14 경로 재편 이전 `challenge/data/capturenight04_manual_gt/...` 를 가리킨다 [확인] |
| `challenge/data/03_derived/yolo_pose*` (6개) | 0 | — | — | `.txt` 라벨 폴더, JSON 없음 [확인] |
| `challenge/data/01_real/augmented/capturepallet07_augmented` | 275 | 0 | 19 / 157 (sentinel 118) | 아래 flip 검증 참조 [확인] |

### 2-1. flip 미러링 수치 검증 — `flip_noise_aug_livegt` (전수 851장)

각 파생본을 파일명으로 부모 프레임에 짝지어 (851/851 짝 성공) 다음을 계산했다.
기대값: `x' = W-1-x` (W=640), 코너 순열 `FLIP_PERM_8=(1,0,3,2,5,4,7,6)`, centroid 불변.

| 검사 | 결과 |
|---|---|
| `projected_cuboid` 가 기대 미러+순열과 일치 (오차 < 1e-6) | **851 / 851** [확인] |
| `keypoint_annotations` 가 기대 미러+순열과 일치 | **0 / 851** [확인] |
| `keypoint_annotations` 가 부모와 **글자 그대로 동일**(미변환) | **851 / 851** [확인] |
| `visibility`/`in_frame`/`source`/`reason` 이 부모와 동일 | 7,659 / 7,659 항목 (851×9) [확인] |
| `manual_kps` 가 부모와 동일(미변환) | 851 / 851 [확인] |
| `pose_transform` 이 `null` 로 바뀜 | 851 / 851 [확인] |
| noise 본의 `keypoint_annotations` 가 부모와 동일 (정상 — 노이즈는 좌표를 안 바꾼다) | 851 / 851 [확인] |

예: `flip_noise_aug_livegt/capture_20260902_008663_f.json` 의
`projected_cuboid[0] = [212.0, 245.0]` (부모 `[513.0, 314.0]` 의 미러+순열 결과) 인데
`keypoint_annotations[0].xy = [15.0, 311.0]` 로 부모와 완전히 같다. [확인]

**→ 디스크의 flip 851장은 이미지만 좌우로 뒤집히고 `keypoint_annotations` 는 안 뒤집힌 상태다.
`load_kps` 가 그 필드를 우선하므로, 지금 재빌드하면 뒤집힌 이미지에 안 뒤집힌 라벨이 붙는다.** [확인]
(`gen_flip_noise_aug.py:75-78` 주석이 이 실패 모드를 정확히 예고하고 있으나, 산출물은 그 수정 이전 것이다.)

### 2-2. flip 인덱스 순열 검증 — `01_real/augmented/capturepallet07_augmented` (전수 275장)

이 폴더는 `augment_dataset.py` 산출이고 프레임당 11개(원본 1 + flip 1 + crop 5 + shear 4)라
파일 인덱스 `i % 11 == 1` 이 flip 변형이다. [확인]

| 변형 슬롯 | LR OK | LR 위반 | sentinel |
|---|---|---|---|
| `i%11==0` (원본) | 19 | **0** | 6 |
| `i%11==1` (**hflip**) | **0** | **19** | 6 |
| `i%11==2..10` (crop·shear) | 100 | 0 | 116 |

즉 flip 변형 25장 중 비-sentinel 19장이 **전부** LR 순서 위반이다. 원인은 코드에 명시돼 있다 —
`augment_dataset.py:5` "horizontal flip (좌우 mirror, **ID swap 없음** → object-fixed convention 유지)",
`:50-54` `hflip()` 이 좌표만 미러하고 인덱스를 순열하지 않는다. object-fixed 규약에선 맞지만
현행 camera-facing 0123 규약에선 좌우 이름이 뒤바뀐 라벨이다. [확인]

### 2-3. truncation crop 의 규약 승계 — `truncation_crops_livegt` (전수 1,206장)

| 검사 | 결과 |
|---|---|
| 파일명으로 부모 프레임 확인 | 1,206 / 1,206 [확인] |
| 파생본 LR 판정이 부모 `projected_cuboid` 판정과 일치 | 1,017 (나머지는 sentinel), **불일치 0** [확인] |
| `keypoint_annotations` 보유 | 0 / 1,206 [확인] |

**→ crop 은 부모의 `projected_cuboid` 를 기하학적으로 올바르게 옮기지만, 규약 위반 여부도 그대로
승계한다. 규약을 지키는 `keypoint_annotations` 는 버려지므로 `load_kps` 가 위반본만 보게 된다.** [확인]

---

## 3. 소비처 (누가 이 파생본을 쓰는가)

- `prepare_yolo_pose_from_live_gt.py:186-195` 의 `--crop-dir`/`--aug-dir` 가 파생 폴더를
  **train 에만** 더한다(val 파생분은 `add_derived_jobs`:113-116 에서 제외). [확인]
- 릴리스 `challenge/yolo_pose_one_model/release/pallet-pose-yolo26n-livegt/dataset_manifest.json:20`
  은 `crop_dir = challenge/data/03_derived/truncation_crops_livegt` 로 빌드됐다
  (`aug_dir` 키 없음 → flip/noise 는 안 들어갔다). train ok=1,328 / val ok=70. [확인]
- `challenge/yolo_pose_one_model/challenge_c4_track/SOURCE_AUDIT.md:165-166` 은
  flip/noise 1,392장 + crop 999장을 쓴 빌드를 기록한다 — 그 빌드는 flip 산출물을 포함한다. [추정]
  (이 감사에서 해당 학습 산출물의 라벨 파일 자체는 열지 않았다 — **확인 못 함**.)

---

## 판정

| 폴더 | 판정 | 사유 |
|---|---|---|
| `challenge/data/03_derived/flip_noise_aug_livegt` (flip 851장) | **UNSAFE_CONVENTION_MISMATCH** | 이미지는 미러, `keypoint_annotations` 는 미변환. `load_kps` 가 그 필드를 우선하므로 라벨이 이미지와 좌우로 어긋난다(851/851). 생성기 코드는 이미 고쳐졌으나(f2b2739) **재생성 미실행** [확인] |
| `challenge/data/03_derived/flip_noise_aug_livegt` (noise 851장) | **SAFE_TO_USE** | 노이즈는 좌표를 안 바꾼다. `keypoint_annotations` 851/851 이 부모와 동일하고 그게 맞는 동작 [확인] |
| `challenge/data/03_derived/truncation_crops_livegt` | **STALE_NEEDS_REBUILD** | 기하 변환 자체는 정확(부모 판정과 불일치 0)하나 `keypoint_annotations` 를 버려 위반본 `projected_cuboid` 로 학습된다. 318/1,203 이 LR 위반 [확인] |
| `challenge/data/01_real/augmented/capturepallet07_augmented` | **UNSAFE_CONVENTION_MISMATCH** | hflip 변형 19/19 전부 LR 순서 위반(인덱스 순열 없음). crop/shear 100장은 무위반 [확인] |
| `challenge/data/03_derived/_train_pallet07_aug` | **UNSAFE_CONVENTION_MISMATCH** | 위와 같은 내용(19/157 위반) [확인] |
| `challenge/data/03_derived/_train_manual_pseudo` | **UNSAFE_CONVENTION_MISMATCH** | 34/63 LR 위반. 추론 PnP 재투영 산출이라 GT 가 아니다 [확인] |
| `challenge/data/03_derived/truncation_crops` | **SAFE_TO_USE** | LR 위반 0/450. 부모(`manual_gt` 계열)에 `keypoint_annotations` 가 애초에 없어 소실될 것이 없다 [확인] |
| `challenge/data/03_derived/truncation_crops_dope` | **SAFE_TO_USE** | 합성. LR 위반 2/46,065 (0.004%) [확인] |
| `challenge/data/03_derived/truncation_crops_synth` | **SAFE_TO_USE** | 합성. LR 위반 0/394 [확인] |
| `challenge/data/03_derived/truncation_crops_palletobj` | **SAFE_TO_USE** | 합성. LR 위반 0/747 [확인] |
| `challenge/data/03_derived/_train_capturepallet07` | **SAFE_TO_USE** | LR 위반 0/25 [확인] |
| `challenge/data/03_derived/_eval_real_gt_merged` | **STALE_NEEDS_REBUILD** | 438개 심링크 전부 끊김(2026-08-14 경로 재편 이전 대상) [확인] |
| `challenge/data/03_derived/yolo_pose*` (6개) | 판정 대상 아님 | JSON 없음(`.txt` 라벨 폴더). 이 감사는 라벨 `.txt` 를 열지 않았다 — **확인 못 함** [확인] |

### 확인 못 한 것

- `yolo_pose*` 6개 폴더의 `.txt` 라벨이 어느 필드에서 나왔는지 (감사 범위 밖).
- `challenge_c4_track` 학습 산출물이 실제로 어긋난 flip 라벨을 소비했는지 — 매니페스트만 봤고
  생성된 `.txt` 는 안 열었다.
- `truncation_crops_dope` 의 2건 LR 위반이 진짜 결함인지 경계 잡음인지 (개별 확인 안 함).


---

# 2026-09-06 후속 — 생성기 수정 (§6 규칙 적용)

위 감사는 **읽기 전용**이었고, 그 결과로 생성기 6개를 고쳤다.
규칙: *모든 derived real data 는 `SOURCE_FIELD = keypoint_annotations` 를 명시적으로
보존하고, frame metadata 에 `keypoint_source` / `parent_frame` / `transformation` 을 남긴다.*

## 공용 헬퍼

`challenge/scripts/dataset/keypoint_annotations_transform.py` (신규).
다섯 곳에 같은 코드를 베끼면 갈라지므로 한 벌만 둔다.

```text
transform_annotations(src_obj, M, w, h, perm=None)
    2x3 affine 으로 xy 를 옮기고 in_frame 을 새 캔버스로 갱신한다.
    xy=None 은 좌표가 아니라 상태이므로 그대로 None 으로 둔다.
    perm 은 좌우가 바뀌는 변환에서만 준다 (FLIP_PERM_8).
provenance(parent_frame, transformation)
    parent_frame 은 저장소 상대경로로 정규화한다 (Windows/Ubuntu 공유).
attach(obj, ...)  위 둘을 오브젝트에 얹는다.
```

## 생성기별 변경

| 파일 | 무엇을 고쳤나 | 실행 검증 |
|---|---|---|
| `gen_truncation_crops.py` | `gen_variant` 가 `win` 을 반환, `write_output` 이 crop affine `[[sx,0,-cx0·sx],[0,sy,-cy0·sy]]` 으로 정본 필드를 옮긴다 | 6장 생성, 부모 대비 최대오차 **1e-13** |
| `pad_truncation_crops.py` | `pad`·원본 크기를 받아 `[[sx,0,pad·sx],[0,sy,pad·sy]]` 적용 | 6장, 최대오차 **5.7e-14** |
| `augment_ratio_robust.py` | `apply_affine` 이 `M` 반환. trunc 경로는 crop+pad 를 **합성**하고 `_assert_matches` 로 자기검증 | squash 4장 / trunc 5장, 최대오차 **0.0** |
| `augment_dataset.py` | ★`hflip` 에 `FLIP_PERM_8` 추가(없어서 19/19 위반이었다). 부모에 필드가 없으면 `manual_kps` 에서 만든다 | 33장 생성, hflip LR **3/3 정상**(위반 0) |
| `make_pseudo_gt.py` | 추론 결과를 `source="pnp_projected"` 로 **출처를 밝혀** 쓴다 — fallback 으로 조용히 내려가지 않게 | 모델 필요로 실행 못 함. 변수 유효범위만 AST 로 확인 |
| `gen_flip_noise_aug.py` | 이미 필드는 보존했으나 provenance 가 없었다 — 추가 | 1,702장 재생성 |

## ★ hflip 규약 변경을 명시한다

`augment_dataset.py` 의 docstring 은 예전에 *"ID swap 없음 -> object-fixed convention
유지"* 라고 적었다.  그 object-frame 규약은 폐기됐고(v8), 이 저장소의 정본은
**camera-facing 0123** 이다(CLAUDE.md).  순열 없이 미러만 하면 0(왼쪽위)이 오른쪽으로
가서 규약을 어긴다.  실측이 그것을 보였고(19/19), 수정 후 위반 0 이다.

**따라서 기존 산출물 `challenge/data/01_real/augmented/capturepallet07_augmented` 와
`03_derived/_train_pallet07_aug` 는 여전히 UNSAFE 다.**  재생성해야 한다.
이번에는 재생성하지 않았다 — v1/v2 과제 데이터라 이번 실험이 쓰지 않는다.

## 테스트

`challenge/tests/test_derived_generators_preserve_keypoints.py`
(테스트 함수 8개 -> 생성기 6개에 parametrize 되어 **18 케이스**).
문자열 검색이 아니라 **AST** 로 본다 — 주석·docstring 에만 있는 것을 통과시키지 않기
위해서다(memory `forbidden-token-tests-must-use-ast`).
실제로 `gen_flip_noise_aug.py` 의 provenance 누락을 이 테스트가 잡아냈다.

디스크 산출물 검사는 `test_derived_artifact_invariants.py` 가 따로 맡는다 —
생성기를 고쳐도 낡은 산출물은 초록불로 통과하기 때문이다.

## 판정 갱신

```text
flip_noise_aug_livegt_v2          SAFE_TO_USE       (재생성 + provenance)
flip_noise_aug_livegt (구본)      STALE             격리 목록에 유지
truncation_crops_livegt (구본)    STALE_NEEDS_REBUILD  생성기는 고쳤으나 산출물 미갱신
augmented/capturepallet07_*       UNSAFE            hflip 순열 없이 만들어진 것
```
