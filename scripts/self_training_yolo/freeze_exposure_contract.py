"""self-training strength 를 결과 보기 전에 동결한다.

self-training 은 filter 만으로 정의되지 않는다.  다음 네 축이 effective strength 를
결정하므로, filter 를 비교하려면 나머지 셋이 전 arm 에서 같아야 한다.

    1. pseudo-real sampling fraction
    2. pseudo-real loss weight
    3. total optimizer updates
    4. teacher refresh rounds

여기서 1·2·3·4 를 전부 못박는다.  arm 사이에 다른 것은 pseudo-label selection rule
하나뿐이다.

## epoch 이 아니라 optimizer update 를 맞춘다

arm 마다 accepted unique PL 수가 다르므로 "같은 epoch" 은 공정 비교가 아니다.
그래서 **한 epoch 의 물리적 크기 자체를 고정**한다.

    epoch 당   1440 pseudo-real + 1440 synthetic replay = 2880 = 90 updates(batch 32)
    epochs     10
    TOTAL      900 optimizer updates,  14400 pseudo exposures, 14400 synthetic exposures

accepted pool 이 작은 arm 은 with replacement 로 1440 슬롯을 채운다.  따라서 filter
품질과 training step 수가 섞이지 않는다.

## TOTAL_UPDATES 와 LR 을 이렇게 고른 이유 (PAPER_EVAL 을 보지 않았다)

repo 이력에 직접 근거가 있다.  `ADAPT_N0_15EP_SEED42` 는 **negative 0, synthetic
replay 전용 control** 인데도 G38 대비 night 검출을 잃었다 (any-cbox 23 -> 19 frames,
`_docs/history/2026-08-25.md`).  그 레시피는 13,554 장 x 15 epoch / batch 32 =
**6,353 optimizer updates @ lr0 0.002** 였다.

즉 이 과제에서 lr0 0.002 로 6,353 update 를 돌리는 것 자체가 이미 비싸다.
그래서 LR 은 repo 의 확립된 adaptation 값(0.002)을 유지하고 — LR 과 update 수를
동시에 바꾸면 원인을 못 가른다 — **update 예산만 그 14% 수준으로 낮춘다**.

    900 / 6353 = 14.2%

이 선택이 충분히 보수적인지는 가정하지 않는다.  `R0-CONT` 가 그것을 측정한다.

## R0-CONT — source-only continuation control

같은 init·LR·augmentation·batch·TOTAL_UPDATES 를 쓰되 pseudo-real exposure 가 0 이고
그 자리를 synthetic replay 로 채운다.  "추가 최적화 자체의 효과" 와 "real pseudo-label
adaptation 효과" 를 분리하기 위한 필수 control 이다.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
R0_DATASET = REPO_ROOT / "challenge/yolo_pose_one_model/datasets/g38_legacy_v1v2_p0_tex20k"
R0_ARGS = (
    REPO_ROOT / "challenge/yolo_pose_one_model/spatial_concat_scratch/runs"
    / "YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42/args.yaml"
)
OUT_DIR = REPO_ROOT / "data/pallet/results/paper_selftrain_v1"

BATCH = 32
PSEUDO_PER_EPOCH = 1440
SYNTHETIC_PER_EPOCH = 1440
EPOCHS = 10
LR0 = 0.002

# 열화가 확인된 기존 adaptation 레시피의 update 수.  예산의 기준점이다.
REFERENCE_HARMFUL_UPDATES = 13554 * 15 // BATCH


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def freeze_replay_subset() -> tuple[list[str], str]:
    """synthetic replay subset 을 결정적으로 고른다.  전 arm 이 같은 집합을 쓴다."""

    names = sorted(path.name for path in (R0_DATASET / "images" / "train").iterdir())
    if not names:
        raise SystemExit("R0_TRAIN_IMAGES_MISSING")
    # 파일명 sha256 오름차순.  이미지 내용을 다시 해싱하지 않아도 결정적이고,
    # 렌더 순서(G38__G__f0000...)에 쏠리지 않는다.
    ordered = sorted(names, key=lambda name: hashlib.sha256(name.encode()).hexdigest())
    subset = ordered[:SYNTHETIC_PER_EPOCH]
    return subset, sha256_text("\n".join(subset))


def main() -> int:
    replay, replay_sha = freeze_replay_subset()
    total_updates = EPOCHS * (PSEUDO_PER_EPOCH + SYNTHETIC_PER_EPOCH) // BATCH

    lock = {
        "schema_version": "paper_selftrain_exposure_lock_v1",
        "purpose": (
            "Freeze self-training strength before any arm is trained, so the only "
            "difference between MAIN arms is the pseudo-label selection rule."
        ),
        "paper_eval_gt_used_for_selection": False,

        "main_pseudo_fraction": 0.50,
        "main_synthetic_fraction": 0.50,
        "lambda_pseudo": 1.0,
        "lambda_note": (
            "Loss weight is fixed at 1.0 and is never searched.  Sampling fraction is "
            "the single self-training strength knob; fraction and loss weight are "
            "never varied together."
        ),

        "teacher_refresh": False,
        "teacher_rounds": 1,
        "teacher_note": (
            "MAIN is a one-round static teacher.  Every student is initialised from the "
            "same R0 checkpoint and consumes the same frozen R0 prediction cache, so "
            "teacher drift never enters the filter comparison.  Iterative rounds are "
            "Appendix A3 only."
        ),

        "batch_size": BATCH,
        "learning_rate": LR0,
        "total_optimizer_updates": total_updates,
        "epochs": EPOCHS,
        "updates_per_epoch": (PSEUDO_PER_EPOCH + SYNTHETIC_PER_EPOCH) // BATCH,
        "pseudo_exposures_per_epoch": PSEUDO_PER_EPOCH,
        "synthetic_exposures_per_epoch": SYNTHETIC_PER_EPOCH,
        "total_pseudo_exposures": EPOCHS * PSEUDO_PER_EPOCH,
        "total_synthetic_exposures": EPOCHS * SYNTHETIC_PER_EPOCH,
        "small_pool_policy": "sampling with replacement to fill the fixed pseudo slots",

        "budget_justification": {
            "reference_run": "challenge/yolo_pose_one_model/runs_camera_facing_loss/ADAPT_N0_15EP_SEED42",
            "reference_updates": REFERENCE_HARMFUL_UPDATES,
            "reference_lr0": 0.002,
            "reference_finding": (
                "synthetic-replay-only control (negative 0, pseudo 0) still lost night "
                "detection versus its own base: any-cbox 23 -> 19 frames "
                "(_docs/history/2026-08-25.md)"
            ),
            "chosen_fraction_of_reference": round(total_updates / REFERENCE_HARMFUL_UPDATES, 4),
            "lr_rationale": (
                "0.002 is the repo's established adaptation learning rate "
                "(runs_ft/ft_a, ft_b, ADAPT_N0/N1, legacy_v1v2_ft).  Changing LR and "
                "update budget together would confound the cause, so only the budget "
                "is reduced."
            ),
            "falsifiable_by": "R0-CONT",
            "paper_eval_not_consulted": True,
        },

        "source_only_control": True,
        "control_arm": {
            "name": "R0-CONT",
            "pseudo_exposures": 0,
            "substitution": "pseudo slots are filled with synthetic replay",
            "everything_else": "same init, LR, augmentation, batch, total updates, seed",
            "question": (
                "how much of any change is caused by additional optimisation alone, "
                "rather than by real pseudo-label adaptation"
            ),
        },

        "initialisation": {
            "checkpoint": (
                "challenge/yolo_pose_one_model/spatial_concat_scratch/runs/"
                "YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42/weights/best.pt"
            ),
            "sha256": "970a0913b38ed4c9e3662837abccbf9d91b8b0858deafae854c1055e477644f7",
            "identical_for_every_arm": True,
        },
        "checkpoint_selection": {
            "rule": "fixed final checkpoint (last.pt)",
            "note": "PAPER_EVAL results never select the epoch or the checkpoint.",
        },
        "augmentation": {
            "source": str(R0_ARGS.relative_to(REPO_ROOT)),
            "fliplr": 0.0,
            "fliplr_note": (
                "Horizontal flip augmentation stays off.  If it were enabled, "
                "horizontal-flip keypoint consistency would become an identity the "
                "training objective enforces, and the filter would stop being an "
                "independent signal."
            ),
            "seed": 42,
        },

        "synthetic_replay": {
            "source_dataset": str(R0_DATASET.relative_to(REPO_ROOT)),
            "source_pool_size": len(list((R0_DATASET / "images" / "train").iterdir())),
            "subset_size": SYNTHETIC_PER_EPOCH,
            "selection": "sha256(filename) ascending — deterministic, render-order free",
            "membership_sha256": replay_sha,
            "identical_for_every_arm": True,
        },

        "arms": [
            {"name": "R0", "trains": False, "note": "existing synthetic-only baseline"},
            {"name": "R0-CONT", "trains": True, "pseudo_filter": None},
            {"name": "R1", "trains": True, "pseudo_filter": "F0 naive"},
            {"name": "R2", "trains": True, "pseudo_filter": "F1 confidence"},
            {"name": "R3", "trains": True, "pseudo_filter": "F2 confidence + reprojection"},
            {"name": "R4", "trains": True, "pseudo_filter": "F3 confidence + keypoint-removal"},
            {"name": "R5", "trains": True, "pseudo_filter": "F4 proposed"},
        ],

        "matching_semantics": {
            "MAIN": "EXPOSURE-MATCHED — identical updates, pseudo exposures, synthetic exposures",
            "appendix_A2": "UNIQUE-QUANTITY-MATCHED — identical number of unique pseudo-labels",
            "never_conflate": True,
        },
        "sensitivity": {
            "appendix": "A12",
            "pseudo_fractions": [0.25, 0.50, 0.75],
            "manifest": "the Proposed (F4) pseudo-label manifest, unchanged",
            "held_constant": [
                "total updates", "learning rate", "init", "pseudo manifest",
                "augmentation", "seed",
            ],
            "main_row_is_fixed_at": 0.50,
            "rule": (
                "0.25 or 0.75 scoring better does not replace the MAIN row.  This "
                "answers whether the effect depends on one particular mixing ratio, "
                "it is not a hyperparameter search."
            ),
            "conditional_extension": (
                "Naive at 0.25/0.75 is run only if Proposed shows a large sensitivity "
                "spread, to check a filter x strength interaction."
            ),
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "SELFTRAIN_EXPOSURE_LOCK.json").write_text(
        json.dumps(lock, indent=2, ensure_ascii=False) + "\n"
    )
    (OUT_DIR / "SYNTHETIC_REPLAY_SUBSET.txt").write_text("\n".join(replay) + "\n")

    print(f"TOTAL_UPDATES        {total_updates}")
    print(f"  vs reference       {REFERENCE_HARMFUL_UPDATES} "
          f"({total_updates / REFERENCE_HARMFUL_UPDATES:.1%})")
    print(f"lr0                  {LR0}")
    print(f"epochs x updates     {EPOCHS} x {(PSEUDO_PER_EPOCH + SYNTHETIC_PER_EPOCH)//BATCH}")
    print(f"pseudo exposures     {EPOCHS * PSEUDO_PER_EPOCH}")
    print(f"synthetic exposures  {EPOCHS * SYNTHETIC_PER_EPOCH}")
    print(f"replay subset        {len(replay)}  sha {replay_sha[:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
