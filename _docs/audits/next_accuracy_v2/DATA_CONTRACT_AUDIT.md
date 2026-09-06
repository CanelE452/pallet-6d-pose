# DATA_CONTRACT_AUDIT — 학습 라벨이 GT 의 어떤 상태를 잃어버리는가

지시문 §3·§4·§5 에 대한 답.  읽기 전용 감사 + 그 결과로 만든 최소 수정.
새 학습 0 · 새 추론 0.  모든 수치는 디스크 전수이고 표본이 아니다.

기준 커밋 `ddc0770` (지시문이 전제한 `f2b2739` 와의 차이는 history 문서 248줄뿐).

## 1. keypoint source contract — 실제로 무엇이 읽히는가

`challenge/yolo_pose_one_model/scripts/prepare_yolo_pose.py::load_kps` 는
`objects[0].keypoint_annotations` 가 9개 이상이면 그것을, 아니면
`objects[0].projected_cuboid` + `projected_cuboid_centroid` 를 읽는다 [확인].

real GT 전수 (`challenge/data/01_real/` 아래 숫자 이름 JSON):

```text
N_total_real_json                 1983
N_with_keypoint_annotations        991
N_fallback_to_projected_cuboid     992
N_unusable                           0
```

폴더별 [확인]:

```text
폴더                    total   kp_ann   fallback
live_capture_gt           851      851          0      <- 학습 소스
gt_v2_canonical           140      140          0      <- 평가셋의 v2 이관본
eval_canonical            105        0        105
manual_gt                 574        0        574
augmented                 275        0        275
pseudo_gt                  38        0         38
```

**fallback 은 예외가 아니라 절반이다(992/1983).**  그리고 fallback 하는 쪽이
정본 평가셋이 물리적으로 놓인 `eval_canonical`·`manual_gt` 다.
`projected_cuboid` 는 `live_capture_gt` 851장에서 camera-facing 0123 규약을
198장(23.3%) 어긴다 [확인, `REAL_LABEL_AUDIT.md` 재현].

산출물: `data/pallet/results/next_accuracy_v2/KEYPOINT_CONTRACT_CENSUS.json`

## 2. ★ `None -> sentinel -> padding -> valid point` — 재현됨

지시문 §4 가 지목한 경로를 **실제 변환기로 재현했다** [확인]:

```text
xy = None
  -> load_kps      (-0.5, -0.5)          # SENTINEL
  -> one()         (99.5, 99.5)          # + PAD(100)
  -> to_line       0 <= 99.5 < 840       # padded 캔버스 안
  -> YOLO txt      v = 2
```

재현 결과 (640x480 더미, index 3 만 `xy=None`):

```text
                       index 3 의 v    bbox (padded px)
수정 전                       2         x[ 99.5, 408.0]  y[ 99.5, 348.0]
수정 후                       0         x[400.0, 408.0]  y[340.0, 348.0]
정상값(손계산)                0         x[400.0, 408.0]  y[340.0, 348.0]
```

**피해는 v 하나가 아니다.**  그 점이 `to_line` 의 `inx` 에 들어가 bbox 까지
좌상단으로 끌려간다 — 위 예에서 8x8 px 상자가 308x248 px 이 됐다.

### 2.1 실제 영향 범위 — 과장하지 않는다

전수 측정 (`measure_contract_violation.py`, 991장 / 8,919 keypoint):

```text
계약 위반 keypoint            1262
그중 gt_v2_canonical          1260      <- 평가셋 이관본. 학습이 읽지 않는다
그중 live_capture_gt             2 프레임 / 6 keypoint   <- 학습 소스
bbox 가 1px 넘게 어긋난 프레임    2      (259.5 px, 255.5 px)
```

`gt_v2_canonical` 140장은 1,260개 keypoint 가 **전부** `visibility=0,
source=unknown` 이다 [확인].  이관 시 "좌표는 남기되 provenance 는 모름" 으로
표시된 것이고, 학습·데이터셋 빌드 코드 중 이 폴더를 읽는 것은 없다
(`git grep gt_v2_canonical` → 테스트·어노테이션 툴·감사 스크립트뿐) [확인].

따라서 **학습 소스 측 실피해는 851장 중 2장**이다.  결함은 실재하고 재현되지만
"학습이 대량으로 오염됐다" 고 말할 근거는 없다.

## 3. 계약 정본은 이미 저장소 안에 있었다

새로 설계한 것이 아니다.  `scripts/annotate/real_gt_v2_schema.py::
keypoint_annotations_to_ultralytics` 의 docstring 이 이미 규정한다 [확인]:

> A migrated visibility-0 point may retain legacy `xy` for audit/bbox geometry,
> but the training target must still be `[0, 0, 0]` so unknown provenance is not
> silently supervised.

즉 §4 가 요구한 세 상태 분리는 **이미 존재하는 계약**이고,
`prepare_yolo_pose.load_kps` 가 그 계약을 쓰지 않은 것이다.

같은 저장소의 `prepare_real_ft.py::load_kps_real` 은 또 다른 방식으로
같은 문제를 피했다 — `far = -10.0 * PAD` 로 캔버스 밖에 던진다 [확인].
셋(스키마 / real-FT 로더 / 메인 변환기)이 서로 다른 규약을 쓰고 있었다.

## 4. 수정

`prepare_yolo_pose.py`:

```text
load_kps  ->  (x, y, known) 3-튜플을 돌려준다
              known = xy is not None and visibility != 0
to_line   ->  (x, y) 와 (x, y, known) 을 모두 받는다
              v = 2 는 known 이고 캔버스 안일 때만
one()     ->  known 플래그를 그대로 넘긴다
```

상태를 **좌표 값으로 표현하지 않는다** — 이것이 §4 의 요구다.
세 상태 대응:

```text
ANNOTATED_VISIBLE       visibility 1/2, xy 있음        -> known=True,  캔버스 판정
ANNOTATED_OUT_OF_FRAME  xy 있음, 캔버스 밖             -> known=True,  v=0
UNKNOWN_OR_MISSING      xy None 또는 visibility 0      -> known=False, v=0, x=y=0
```

`visibility` 필드가 아예 없으면 `0`(모름)으로 본다.  이 경우 그 프레임은
감독 대상이 0개가 되어 `to_line` 이 `None` 을 돌려주고 `all_kp_outside` 로
집계된다 — **조용히 틀리는 대신 눈에 보이게 실패한다**.
현재 real GT 8,919개 항목 전부가 이 필드를 갖고 있다 [확인].

호출자 `scripts/stage0/model_compare/mc_build_yolo_broad.py:47` 도 함께 고쳤다.

### 4.1 이 수정이 학습 감독을 줄이는가 — 아니다

`live_capture_gt` 851장 전수 [확인]:

```text
visibility=0 이고 xy 가 있는 점    0     <- 0 이라 정당한 감독은 하나도 안 준다
visibility=0 이고 xy=None          6     <- 잘못 감독되던 점, 이제 v=0
visibility=1                    3720
visibility=2                    3933
```

`visibility=1`(pnp_projected 2,871 · centroid_auto 849)은 계속 감독된다.
감사한 ultralytics 8.4.60 loader 가 `v==0` 만 마스킹하고 1과 2를 동등하게
취급하므로, `to_line` 이 1을 "2" 로 쓰는 것은 동작상 차이가 없다 [확인, 스키마
docstring].  이것을 결함으로 보고하지 않는다.

## 5. end-to-end 테스트 (§5)

기존 `test_keypoint_field_contract.py` 는 loader 에서 끝나 이 결함을 못 잡았다.
그중 `test_load_kps_null_xy_becomes_sentinel` 은 오히려 **결함을 계약으로
고정하고 있었다** — 갱신했다.

새로 `challenge/tests/test_label_contract_end_to_end.py` (7개).
전부 JSON -> load_kps -> pad -> to_line -> YOLO txt 파싱을 실제로 통과한다.

```text
test_none_keypoint_stays_unsupervised_after_padding
test_visibility_zero_with_legacy_xy_is_not_supervised
test_known_out_of_frame_keypoint_follows_contract_after_padding
test_real_keypoint_annotations_survive_full_conversion
test_flip_full_conversion_preserves_index_contract
test_synthetic_fallback_still_converts
test_to_line_accepts_bare_pairs
```

`13 passed` (신규 7 + 기존 파일 6, 기존 3개는 계약에 맞게 갱신).

## 5.5 §7 재빌드 감사 표 (전 항목)

산출물 `challenge/yolo_pose_one_model/datasets/live_gt_contract_v2/_contract_build.json`
(스크립트가 직접 쓴다 — 손으로 넣지 않았다).

```text
항목                              train    val(held-out)
총 frame                            536              297
총 supervised keypoint            4,808            2,673
v=2 개수                          4,808            2,673
v=0 개수                             16                0
원본 xy=None 개수                      0                0
xy=None -> v=2 변환 개수               0                0   <- 필수 0
index convention violation            0                0   <- 필수 0
duplicate stem                        0                0
train/val same source frame overlap            0           <- 필수 0
derived-parent overlap                   N/A (원본 프레임만 씀)
```

필수 3조건 전부 충족 -> 학습 진행.
`derived-parent overlap` 은 이 빌드가 `--crop-dir`/`--aug-dir` 를 쓰지 않아
발생할 수 없다.  빈칸으로 두지 않고 `N/A` 를 명시한다 — 빈칸은 "안 쟀다" 와
"없다" 를 구분하지 못한다.

## 6. 남은 것

- 합성 GT 의 `projected_cuboid` 에 `[-1,-1]`("not projected") 이 얼마나 있는지는
  별도 감사에서 세는 중이다.  있다면 `+PAD` 로 (99,99) 가 되어 **같은 결함이
  R0 의 55,980장에 걸린다**.  숫자를 보기 전에는 fallback 경로를 건드리지 않는다.
- 파생 데이터는 `DERIVED_DATA_AUDIT.md` 참조.  생성기 대부분이
  `keypoint_annotations` 를 떨어뜨려 위반본 `projected_cuboid` 로 되돌아간다.
