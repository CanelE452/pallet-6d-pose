# Pallet real evaluation workspace v1

이 directory는 기존 source/raw/GT와 분리된 수정 가능한 evaluation working copy다.
`DEV`, `DEV_UNVERIFIED`, `FINAL`은 절대 합쳐서 FINAL 수치로 계산하지 않는다.

## Evaluation populations

- `DEV_EVAL_POSITIVE.csv`: controlled plastic 128 + wood 45 (173 images)
- `DEV_EVAL_NEGATIVE.csv`: frozen DEV membership 2689 rows
- `FINAL_EVAL_POSITIVE.csv`: annotated, QA-eligible FINAL positives only
- `FINAL_EVAL_NEGATIVE.csv`: FINAL negatives
- `ALL_AVAILABLE_{POSITIVE,NEGATIVE}.csv`: DEV/FINAL SHA256-deduplicated union
- `DEV_PLASTIC_AUDITED140.csv`: FT-overlap 12장을 포함한 review population

`ALL_AVAILABLE`은 편의/보조 evaluation이며 held-out FINAL로 부르지 않는다.

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

# D/E. 새 FINAL image 배치 후 session.json과 필요시 frame_tags.csv 작성
# final/positive/sessions/<session>/rgb/*.png

# F. 기존 DEV 또는 plastic FINAL annotation
python scripts/annotate/annotate.py \
  --seq data/evaluation/pallet_eval_v1/dev_existing/sessions/<session> \
  --out_dir data/evaluation/pallet_eval_v1/dev_existing/annotations/<session> \
  --population-role DEV --default_split eval \
  --object-type plastic_standard_110x130x11 \
  --eval-root data/evaluation/pallet_eval_v1

python scripts/annotate/annotate.py \
  --seq data/evaluation/pallet_eval_v1/final/positive/sessions/<session> \
  --out_dir data/evaluation/pallet_eval_v1/final/positive/annotations/<session> \
  --population-role FINAL --default_split eval \
  --object-type plastic_standard_110x130x11 \
  --eval-root data/evaluation/pallet_eval_v1

# Wood는 geometry와 intrinsics provenance를 명시한다.
python scripts/annotate/annotate.py \
  --seq data/evaluation/pallet_eval_v1/final/positive/sessions/<wood_session> \
  --out_dir data/evaluation/pallet_eval_v1/final/positive/annotations/<wood_session> \
  --population-role FINAL --default_split eval \
  --object-type wood_small_80x59x14 \
  --intrinsics-quality CALIBRATED --intrinsics-source '<calibration artifact>' \
  --eval-root data/evaluation/pallet_eval_v1
```

G. 매 save마다 JSON, `_overlays/<stem>.png`, `manifests/frames.csv`, progress
report가 갱신된다. H. 부족 조건은 `reports/ANNOTATION_PROGRESS.md`와
`reports/NEXT_ANNOTATION_PRIORITY.md`에서 확인한다.

새 FINAL 촬영은 `final/positive/sessions/<session>/rgb/` 또는
`final/negative/sessions/<session>/rgb/`에 둔다. 각 session에 `session.json`을
작성하고 frame별 수동 tag는 `frame_tags.csv`로 override한다.

`far/small`, `elevation`, `view`는 임의 threshold로 추정하지 않는다. 명시하지
않은 값은 `unknown`으로 남아 `NEXT_ANNOTATION_PRIORITY.md`의 metadata queue에
표시된다.
