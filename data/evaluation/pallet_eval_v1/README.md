# Pallet real evaluation workspace v1

이 directory는 기존 source/raw/GT와 분리된 수정 가능한 evaluation working copy다.
active frame의 `DEV`, `DEV_UNVERIFIED`, `FINAL` role은 서로 섞거나 변경하지 않는다.
`FINAL_EVAL` manifest는 controlled `DEV_EVAL`을 물리 복사 없이 재사용하는 실행용
alias이며 held-out FINAL이 아니다.

## How does the tool know the condition?

- `SESSION.JSON`: object type, day/night lighting, session-wide defaults
- `ANNOTATION JSON`: annotation 완료 여부, occlusion evidence, truncation evidence
- `FRAME TAG UI`: occlusion/truncation, distance NEAR/MID/FAR, elevation override와
  annotated-session batch apply

우선순위는 `FRAME > JSON > SESSION > UNKNOWN`이다. elevation은 frame UI에서
LOW/MID/HIGH로 입력하고 distance는 NEAR/MID/FAR로 입력한다. 현재 contract에는
고정 meter threshold가 없다. 과거 CSV의 `size_bin`은 파일 호환을 위해 보존하지만
active UI, 진행률, 논문 condition 표에서는 사용하지 않는다. 그 외 view는 기존
metadata가 없으면 `unknown`으로 남는다.
조건은 dataset metadata 자체에 저장되므로 사용자가 매 frame마다 ChatGPT나 CLI에
"이 사진은 far다"라고 별도로 설명할 필요가 없다.

## Annotation SESSION selector

annotation UI의 `SESSION` 목록에는 수정 가능한 DEV 9개(플라스틱 7개 + 목재 2개)와
신규 촬영본의 zero-copy `STAGING EDIT` 4개가 함께 표시된다. DAY와 NIGHT capture가
각각 `PLASTIC`, `WOOD` 두 행으로 보인다. 각 행은 같은 raw capture를 복사하지 않고
참조하되 서로 겹치지 않는 실제 frame subset만 표시하며, object별 registry
geometry로 PnP를 푼다. raw session은 수정·이동하지 않는다.

분류 정본은 각 raw frame을 정확히 한 번 기록한
`incoming/sessions/<capture>/manifests/frame_review.csv`다. 실제 픽셀을 프레임
단위로 검수해 `plastic`, `wood`, `exclude`로 나눴으며, `exclude`(파렛트 없음,
카메라 이동, 심한 motion blur)는 두 객체 view에서 모두 숨긴다. 큰 재질 경계는
DAY source ordinal 기준 `WOOD=69..5480`, `PLASTIC=5481..24193`,
`WOOD=24241..29028`이고, NIGHT는 `WOOD=1..4849`,
`PLASTIC=4850..13583`이다. 경계 안의 검수 제외 구간까지 적용한 최종 수는 DAY
`PLASTIC 17,917 / WOOD 9,362 / EXCLUDE 1,749`, NIGHT
`PLASTIC 7,913 / WOOD 4,546 / EXCLUDE 1,124`이다.
partition view 안의 `frame N/M`과 goto는 view-local 번호다. 패널에는 원본 기준
`source ordinal N/raw_total`도 함께 표시하며, 경계 추적은 source ordinal 또는
filename을 사용한다.

PnP GT JSON, 호환 PNG, `frame_tags.csv`, `_overlays/<stem>.png`는 각각
`incoming/annotations/<capture>__plastic/` 또는
`incoming/annotations/<capture>__wood/` 아래에만 저장된다. 제공된 camera
intrinsics는 검증되지 않았으므로 GT의 `intrinsics_quality`는 `UNKNOWN`이고, 원래
`PROVIDED_UNVERIFIED` 품질과 `camera_info.json` 출처는 `intrinsics_source`에 보존한다.

staging save는 top-level `manifests/frames.csv`, DEV/FINAL 평가 manifest,
progress/report MD를 자동 갱신하거나 evaluation member를 만들지 않는다. 맞는 object
frame을 검수한 뒤 DEV/FINAL로 promotion하는 작업은 별도 절차다.

## Evaluation populations

- `DEV_EVAL_POSITIVE.csv`: controlled plastic 128 + wood 45 (173 images)
- `DEV_EVAL_NEGATIVE.csv`: frozen DEV membership 2689 rows
- `FINAL_EVAL_POSITIVE.csv`: registered DEV_EVAL positive 173행의 frozen 실행 alias
- `FINAL_EVAL_NEGATIVE.csv`: registered DEV_EVAL negative 2689행의 frozen 실행 alias
- `FINAL_{POSITIVE,NEGATIVE}.csv`: physical FINAL inventory만 포함하며 alias는 포함하지 않음
- `ALL_AVAILABLE_{POSITIVE,NEGATIVE}.csv`: DEV_EVAL + physical FINAL의 SHA256-deduplicated union
- `DEV_PLASTIC_AUDITED140.csv`: FT-overlap 12장을 포함한 review population

`FINAL_EVAL` alias row는 원래 `population_role=DEV`를 유지하고 notes에
`REUSED_DEV_EVAL_NOT_HELD_OUT; ORIGINAL_ROLE_DEV`를 기록한다. `FINAL_EVAL`과
`ALL_AVAILABLE` 어느 것도 held-out FINAL로 부르지 않는다. 현재 evaluation은
이미 준비되었으며 새 annotation은 필수가 아니다. `DATASET_TARGETS.json`은
DEV_EVAL과 physical FINAL을 함께 세는 대략적 evaluation collection 목표다.
현재 DEV positive와 negative image는 모두 `dev_existing/sessions/` 아래의 독립
복사본이다. 원본 raw/GT와 source SHA provenance는 그대로 보존한다.

## Evaluator binding

현재 paper evaluator는 workspace CSV를 다시 `population_role == FINAL`로 거르지
않는다. 동일한 controlled membership의 기존 frozen DEV manifest를 직접 입력한다.

```bash
python challenge/evaluation_v2/paper_real_eval.py \
  --positive-manifest challenge/real_gt_v2/manifests/COMMON_DEV_MULTISHAPE_POS.json \
  --negative-manifest challenge/real_gt_v2/manifests/DEV_NEG2689.json \
  --population-role DEV \
  --weights <checkpoint.pt> --out <result.json>
```

positive membership은 workspace 실행 alias와 같은 173행이다. negative도 같은
frozen 2,689행이며 그 안의 unique image는 2,688장이다. physical FINAL은 이 실행
alias에 자동으로 섞지 않는다. 이 평가에 `--population-role FINAL`을 사용하지 않는다.
등록된 pair SHA256은
`2cfa7011d8ba3677b11019c103e2ccbaeeac53521c9291ed632f94c8d2c5c887`이다.
AP/AUROC/FPR95 score pipeline도 같은 173/2,689 membership을 사용해야 한다.
현재 그 score pipeline과 workspace condition tag를 pair SHA에 묶은 통합 artifact는
아직 없으므로 해당 ranking/condition metric은 생성 전까지 `—`로 둔다.

## Workflow

```bash
# 선택: destination을 만들지 않는 source audit
python scripts/evaluation/import_existing_evaluation_data.py \
  --root data/evaluation/pallet_eval_v1 --audit-only

# A. 기존 데이터 비파괴 import
python scripts/evaluation/import_existing_evaluation_data.py \
  --root data/evaluation/pallet_eval_v1

# B. 기존 overlay 재생성
python scripts/annotate/rebuild_annotation_overlays.py \
  --dataset-root data/evaluation/pallet_eval_v1 \
  --scope dev_existing --force

# C. 상태/manifest/report 갱신
python scripts/evaluation/eval_dataset_status.py \
  --root data/evaluation/pallet_eval_v1

# D/E. 새 FINAL image 배치 후 session.json 작성 (frame_tags.csv는 UI가 갱신)
# final/positive/sessions/<session>/rgb/*.png

# F. 기존 DEV 또는 plastic FINAL annotation
python scripts/annotate/annotate.py \
  --seq data/evaluation/pallet_eval_v1/dev_existing/sessions/<session> \
  --out_dir data/evaluation/pallet_eval_v1/dev_existing/annotations/<session> \
  --default_split eval \
  --eval-root data/evaluation/pallet_eval_v1

python scripts/annotate/annotate.py \
  --seq data/evaluation/pallet_eval_v1/final/positive/sessions/<session> \
  --out_dir data/evaluation/pallet_eval_v1/final/positive/annotations/<session> \
  --default_split eval \
  --eval-root data/evaluation/pallet_eval_v1

# Wood geometry는 session에서 자동 결정하며, session에 없으면 intrinsics provenance만 명시한다.
python scripts/annotate/annotate.py \
  --seq data/evaluation/pallet_eval_v1/final/positive/sessions/<wood_session> \
  --out_dir data/evaluation/pallet_eval_v1/final/positive/annotations/<wood_session> \
  --default_split eval \
  --intrinsics-quality CALIBRATED --intrinsics-source '<calibration artifact>' \
  --eval-root data/evaluation/pallet_eval_v1
```

G. 선택적으로 physical FINAL을 확장할 때 매 save마다 JSON,
`_overlays/<stem>.png`, `manifests/frames.csv`, report가 갱신된다.
`reports/NEXT_ANNOTATION_PRIORITY.md`는 새 annotation을 요구하지 않고 현재
DEV_EVAL과 physical FINAL을 합친 `ALL_AVAILABLE` 목표 진행률을 보여준다.
`0/300` 같은 수치는 DEV/FINAL로 나누지 않고 이 combined population에서 계산한다.

선택적으로 새 FINAL 촬영을 추가할 때만 `final/positive/sessions/<session>/rgb/` 또는
`final/negative/sessions/<session>/rgb/`에 둔다. 각 session에 `session.json`을
작성한다. frame별 수동 tag는 annotation UI에서 `/`를 눌러 전용 `CONDITIONS`
모드에 들어간 뒤 `1=occlusion ON/OFF`, `2=truncation ON/OFF`,
`3=LOW`, `4=MID`, `5=HIGH` elevation, `n=NEAR`, `m=MID`, `6=FAR`를
지정한다. 누를 때마다 화면 값이 즉시 바뀐다. `u`는 distance 수동 tag를
UNKNOWN/default로 되돌린다.
현재 frame에서 방금 바꾼 항목만 같은 session의 annotation JSON이 있는
frame에 일괄 적용하려면 `a`를 두 번 누른다. 미어노테이션 frame은 제외하고
다른 기존 tag는 보존한다. legacy size와 view는 annotation 화면에서 선택하도록
요구하지 않는다. `s`로 annotation과 tag를 함께 저장하므로
`frame_tags.csv`를 직접 편집할 필요가 없다.
현재 session의 annotated frame 전체에서 distance tag만 지우려면
`u`, `a`, `a` 순서로 누른다.

positive/negative 또는 plastic/wood가 섞인 연속 촬영본은 곧바로 FINAL에 넣지 않는다.
`scripts/evaluation/import_incoming_capture.py`로 `incoming/sessions/`에 먼저
비파괴 import한다. raw capture는 `INCOMING_UNREVIEWED`로 유지하면서 SESSION의
object별 zero-copy `STAGING EDIT` 행에서 annotation한다. staging output은
`incoming/annotations/`에만 쓰며 DEV/FINAL 평가와 combined 목표 수치에 자동으로
포함되지 않는다. 검수한 frame의 promotion과 평가 활성화는 별도로 수행한다.

`far`, `elevation`, `view`는 임의 threshold로 추정하지 않는다. 명시하지
않은 값은 `unknown`으로 남는다. 이 값은 DEV alias의 provenance를 그대로 설명할
뿐 새 annotation이나 metadata 보완을 의무화하지 않는다.
