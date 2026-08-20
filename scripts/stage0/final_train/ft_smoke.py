"""PHASE 9 -- structural smoke.  Not a performance check, and not selection.

One question only: did the final checkpoint come out structurally intact.  It
deliberately reports no comparison against E3 or any earlier run, because the
historical MH_DEV is INSIDE this training pool -- every synthetic frame reached
here is in-train, so a "better" number would mean nothing.

The F3 path is the real one: `mh_fusion.solve_arms`, on the same prediction
cache `mh_diagnose cache` builds for every other run.  Reimplementing a smaller
solver here would smoke-test a different thing than the model ships with.
"""
from __future__ import annotations

import json, pathlib, sys
import numpy as np

ROOT = pathlib.Path("/home/minjae/Documents/github/pallet-pose")
sys.path.insert(0, str(ROOT / "scripts/stage0/multihead"))
import mh_data as MD      # noqa: E402
import mh_fusion as FU    # noqa: E402

OUT = MD.OUT
RUN = "FINAL40K"
POPULATIONS = ("D2_MH_DEV512", "D0_MH_SEEN512")
MAX_FRAMES = 64


def main(seeds=(1, 2)):
    weight = json.loads(
        (OUT / "theta_posealigned_d0.json").read_text()
    )["seeds"]["seed1"]["selected_lambda_theta"]

    report = {"note": "structural smoke only. Every frame here is IN-TRAIN "
                      "(MH_DEV was folded into FINAL_SYNTH_TRAIN_V1), so no "
                      "number below is a generalisation measurement.",
              "theta_weight_source": "theta_posealigned_d0.json seed1",
              "max_frames_per_population": MAX_FRAMES, "seeds": {}}
    ok_all = True
    for seed in seeds:
        entry = {"populations": {}}
        seed_ok = True
        for population in POPULATIONS:
            path = OUT / f"mh_predcache_{RUN}_seed{seed}_{population}.npz"
            if not path.exists():
                entry["populations"][population] = {"cache": False}
                seed_ok = False
                continue
            data = np.load(path, allow_pickle=True)
            n = min(MAX_FRAMES, len(data["pred_corner"]))
            finite_corner = bool(np.isfinite(data["pred_corner"][:n]).all())
            finite_theta = bool(np.isfinite(data["pred_theta"][:n]).all())
            peaks = np.asarray(data["corner_peak"][:n])[:, :8]
            score_4kp = np.sort(peaks, axis=1)[:, ::-1][:, 3]
            solved = {arm: 0 for arm in FU.ARMS}
            for i in range(n):
                arms, _, _, _ = FU.solve_arms(data, i, weight)
                for arm, pose in arms.items():
                    if pose is not None:
                        solved[arm] += 1
            block = {"cache": True, "n": int(n),
                     "corner_finite": finite_corner,
                     "theta_finite": finite_theta,
                     "score_4kp_finite": bool(np.isfinite(score_4kp).all()),
                     "score_4kp_min": float(score_4kp.min()),
                     "score_4kp_max": float(score_4kp.max()),
                     "solved": solved}
            block["OK"] = bool(finite_corner and finite_theta
                               and block["score_4kp_finite"]
                               and solved["F0"] > 0 and solved["F3"] > 0)
            seed_ok &= block["OK"]
            entry["populations"][population] = block
            print(f"  seed{seed} {population:<14} n={n} corner {finite_corner} "
                  f"theta {finite_theta} score4kp "
                  f"[{score_4kp.min():.3f},{score_4kp.max():.3f}] "
                  f"F0 {solved['F0']}/{n} F3 {solved['F3']}/{n} "
                  f"OK={block['OK']}", flush=True)
        entry["STRUCTURAL_OK"] = bool(seed_ok)
        ok_all &= seed_ok
        report["seeds"][f"seed{seed}"] = entry
    report["STRUCTURAL_OK_ALL"] = bool(ok_all)
    target = OUT / "final_train"
    target.mkdir(parents=True, exist_ok=True)
    (target / "FINAL_TRAIN_SMOKE.json").write_text(
        json.dumps(report, indent=1, default=str))
    print(f"STRUCTURAL_OK_ALL = {ok_all}")
    if not ok_all:
        raise SystemExit("structural smoke failed")


if __name__ == "__main__":
    main()
