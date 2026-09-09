"""학습 경로 스모크 — 1 epoch, 데이터 1%.  배선만 본다."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from screen_driver import r0_recipe, R0_CKPT, RUNS, DSROOT, assert_no_custom_loss


def main():
    from ultralytics.models.yolo.pose import PoseTrainer
    assert_no_custom_loss()
    arm = sys.argv[1] if len(sys.argv) > 1 else "D1"
    data = DSROOT / ("low_angle_diversity_v1_swap" if arm == "D1" else "g38_legacy_v1v2_p0_tex20k")
    ov = dict(r0_recipe())
    ov.update(dict(model=str(R0_CKPT), data=str(data / "data.yaml"), epochs=1, fraction=0.01,
                   project=str(RUNS.parent / "runs_smoke"), name=f"SMOKE_{arm}", exist_ok=True,
                   patience=0, val=False, plots=False, save_period=-1))
    PoseTrainer(overrides=ov).train()
    print("SMOKE OK", arm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
