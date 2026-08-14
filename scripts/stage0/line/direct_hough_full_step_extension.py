"""Was the direct-Hough FULL miss a feature limit or an exposure limit?

`9e1ad5e` records `DIRECT_HOUGH_TOKEN_XY_FULL_FAIL` at 8,515 steps and it
stands.  But at that step the cross-entropy was still falling, the geometry was
still improving, and the seen/unseen gap was 5.0%, so the result reads as
FULL-scale underfit rather than as an exhausted representation -- especially
next to `DIRECT_HOUGH_OVERFIT_EXTENDED_PASS`, where the same architecture
reaches 0.338 degree on 32 frames.

```
changed      optimizer exposure, 8,515 -> 25,545 steps (5 -> 15 passes, 3x)
unchanged    the 13,618 frames, frozen A1, token XY, twelve role queries, the
             cross-attention block, DirectHoughHead, the theta/rho lattice, the
             target, the cross-entropy, batch, LR, weight decay, seed, and every
             gate
```

Fresh from step 0.  The recorded FULL checkpoints carry no optimizer state, so
resuming would continue a different AdamW trajectory than the one being
extended.  The dead `self.position` is neither deleted nor called: it consumes
RNG before the encoder and the head, and removing it would change every
initialisation.

`train_long` re-composes `DH.train_network` from the same pieces rather than
importing it, because the slope bookkeeping and the per-role gating need the
loss history the original does not return.  `parity` exists to prove that
re-composition trains the identical trajectory.

Decision at 25,545 on `D2_LINE_DEV512` only.  `D0_SEEN512` is diagnostic and
enters no selection.  Scope and verdict labels are fixed in
`DIRECT_HOUGH_FULL_OPTIMIZATION_SCOPE.md` before this file runs.
"""
from __future__ import annotations
import os as _os, sys as _sys

# --- stage0 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_S0 = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_S0] + [_os.path.join(_S0, _d) for _d in sorted(_os.listdir(_S0))
                         if _os.path.isdir(_os.path.join(_S0, _d)) and not _d.startswith(".")]


import argparse, importlib.util, json, os, pathlib, sys, time
import numpy as np, torch

ROOT = pathlib.Path(__file__).resolve().parents[3]
for _e in ("scripts/stage0", "Deep_Object_Pose/common", "challenge/scripts"):
    if str(ROOT / _e) not in sys.path:
        sys.path.insert(0, str(ROOT / _e))


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


DH = _load("DH_LONG", "scripts/stage0/line/direct_hough_role_heatmap.py")
EXT = _load("DH_EXT_LONG", "scripts/stage0/line/direct_hough_overfit_extension.py")
CAP, V2, SCALE = DH.CAP, DH.V2, DH.SCALE
OUT, DEV = DH.OUT, DH.DEV

PASSES = 15
RECORDED_PASSES = 5
LONG_MARKS = (1703, 5000, 8515, 17030, 25545)
DECISION_STEP = 25545
PER_ROLE_MARKS = (8515, 17030, 25545)
DIAGNOSTIC_MARKS = (5000,)
TAG = "long"
RECORDED_FULL = "direct_hough_full.json"
RECORDED_MARK = "8515"

# Pre-registered in DIRECT_HOUGH_FULL_OPTIMIZATION_SCOPE.md before any run.
# The instruction fixes the labels and leaves the words unquantified; these are
# the quantities and they are not adjustable after a number is read.
WEAK_IMPROVEMENT = 0.20        # both D2 medians, 8,515 -> 25,545
GEOMETRY_PLATEAU = 0.05        # both D2 medians, 17,030 -> 25,545
CE_PLATEAU_DROP = 0.02         # CE(17,030) - CE(25,545)
CE_PLATEAU_SLOPE = -1e-5       # per step over the final pass; >= this is flat
CE_STRONG_DROP = 0.10          # CE(17,030) - CE(25,545)
PARITY_STEPS = 20
PARITY_FRAMES = 64
DETERMINISTIC_WORKSPACE = ":4096:8"


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def slope(values):
    """Least-squares slope of CE against step, in CE units per step."""
    if len(values) < 2:
        return 0.0
    y = np.asarray(values, dtype=np.float64)
    x = np.arange(y.size, dtype=np.float64)
    return float(np.polyfit(x, y, 1)[0])


def recorded_full():
    """The 9e1ad5e FULL at full stored precision -- never a transcription."""
    return json.loads((OUT / RECORDED_FULL).read_text())


def check_exposure(pool):
    per_pass = V2.steps_per_pass(pool, CAP.BATCH)
    want = {"steps_per_pass": 1703,
            "recorded_decision": 1703 * RECORDED_PASSES,
            "long_decision": 1703 * PASSES}
    got = {"steps_per_pass": per_pass,
           "recorded_decision": per_pass * RECORDED_PASSES,
           "long_decision": per_pass * PASSES}
    for key, value in want.items():
        if got[key] != value:
            raise RuntimeError(f"HARD_BLOCK exposure {key}: {got[key]} != {value}")
    if got["long_decision"] != DECISION_STEP:
        raise RuntimeError("HARD_BLOCK: decision step is not 15 passes")
    if max(DH.MARKS) != got["recorded_decision"]:
        raise RuntimeError("HARD_BLOCK: recorded ladder is not 5 passes")
    got["exposure_ratio"] = PASSES / RECORDED_PASSES
    return got


def checkpoint_optimizer_state():
    """PHASE 3: verified by loading the file, not assumed from the code."""
    path = CAP.checkpoint_path("DH_full", f"step_{max(DH.MARKS):05d}")
    if not path.exists():
        return {"path": str(path), "exists": False, "resume_possible": False}
    stored = torch.load(path, map_location="cpu", weights_only=False)
    keys = sorted(stored) if isinstance(stored, dict) else []
    return {"path": str(path), "exists": True, "keys": keys,
            "has_optimizer_state": any("optim" in k.lower() for k in keys),
            "resume_possible": False}


def train_long(pool, marks, edges, a1, populations, tag, per_pass):
    """DH.train_network re-composed: same pieces, extra bookkeeping only."""
    grid_theta, grid_rho, valid = DH.lattice()
    features = DH.hypothesis_features(grid_theta, grid_rho)
    model = DH.DirectHoughModel().to(DEV)
    optimiser = torch.optim.AdamW(model.parameters(), lr=CAP.LR,
                                  weight_decay=CAP.WD)
    history, losses, done = {}, [], 0
    for chunk, _ in V2.step_schedule(pool, max(marks), CAP.BATCH):
        model.train()
        pack = V2.load_pack(chunk)
        theta_c, rho_c, support = DH.batch_rows(pack, edges)
        target = DH.target_distribution(
            theta_c.reshape(-1), rho_c.reshape(-1), grid_theta, grid_rho, valid
        ).reshape(*theta_c.shape, -1)
        f50, _ = DH.encoder_features(pack, a1)
        loss = DH.cross_entropy(model(f50, features), target, support, valid)
        optimiser.zero_grad(set_to_none=True)
        loss.backward(); optimiser.step()
        losses.append(float(loss.detach()))
        done += 1
        if done in marks:
            entry = {"step": done,
                     "train_loss_mean_last250": float(np.mean(losses[-250:])),
                     "train_loss_slope_last250": slope(losses[-250:]),
                     "train_loss_mean_last_pass": float(np.mean(losses[-per_pass:])),
                     "train_loss_slope_last_pass": slope(losses[-per_pass:]),
                     "diagnostic_only": done in DIAGNOSTIC_MARKS}
            for label, indices in populations.items():
                entry[label] = DH.evaluate_network(
                    indices, model, a1, edges, features, grid_theta, grid_rho,
                    valid, per_role=(label == "D2_LINE_DEV512"
                                     and done in PER_ROLE_MARKS))
                log(f"  {tag} @{done:6d} {label:<16} angle med "
                    f"{entry[label]['angle_median']:7.4f} p90 "
                    f"{entry[label]['angle_p90']:7.3f} | offset med "
                    f"{entry[label]['offset_median']:7.4f} p90 "
                    f"{entry[label]['offset_p90']:7.3f}")
            log(f"  {tag} @{done:6d} CE last250 {entry['train_loss_mean_last250']:.6f}"
                f"  slope/step {entry['train_loss_slope_last_pass']:+.3e}")
            torch.save({"tag": tag, "step": done, "model": model.state_dict(),
                        **CAP.provenance()},
                       CAP.checkpoint_path(f"DH_{tag}", f"step_{done:05d}"))
            history[str(done)] = entry
    return history, model


def parameter_distance(left_model, right_model):
    left, right = left_model.state_dict(), right_model.state_dict()
    if sorted(left) != sorted(right):
        raise RuntimeError("HARD_BLOCK parity: different parameter sets")
    deltas = {k: float((left[k].float() - right[k].float()).abs().max())
              for k in left}
    worst = max(deltas, key=deltas.get)
    return {"max_abs_delta": deltas[worst], "worst_tensor": worst,
            "tensors": len(deltas),
            "nonzero_tensors": sum(1 for v in deltas.values() if v > 0.0)}


def run_parity(edges, a1):
    """Does the re-composed loop train the identical trajectory?

    The first version of this check compared two state dicts under the default
    kernels and demanded bit equality.  It failed -- and then the locked runner
    failed it against *itself*, so the assumption it encoded was simply false.
    That measurement is kept here as `default_mode_spread`, because a FULL
    trajectory that is not bit-reproducible is worth recording.

    Structural parity is therefore asked with the nondeterminism removed:
    `torch.use_deterministic_algorithms` makes the locked runner exactly
    reproducible, and under that condition two structurally identical loops must
    agree bit for bit.  `deterministic_control` proves the mode does what it
    claims before `structural_parity` is believed.  The real run never enables
    it, so it stays in the same numerical regime as the recorded FULL.
    """
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != DETERMINISTIC_WORKSPACE:
        raise RuntimeError("parity needs CUBLAS_WORKSPACE_CONFIG="
                           f"{DETERMINISTIC_WORKSPACE} in the environment")
    pool = V2.split_indices()[0][:PARITY_FRAMES]
    per_pass = V2.steps_per_pass(pool, CAP.BATCH)
    populations = {"PARITY": pool[:CAP.BATCH]}

    def locked(tag):
        return DH.train_network(pool, (PARITY_STEPS,), edges, a1, populations, tag)[1]

    def recomposed(tag):
        return train_long(pool, (PARITY_STEPS,), edges, a1, populations, tag,
                          per_pass)[1]

    spread = parameter_distance(locked("parity_default_a"), locked("parity_default_b"))
    torch.use_deterministic_algorithms(True)
    try:
        control = parameter_distance(locked("parity_locked_a"), locked("parity_locked_b"))
        reference = locked("parity_locked_c")
        parity = parameter_distance(reference, recomposed("parity_candidate"))
    finally:
        torch.use_deterministic_algorithms(False)
    report = {"steps": PARITY_STEPS, "frames": PARITY_FRAMES,
              "default_mode_spread": spread,
              "deterministic_control": control,
              "structural_parity": parity,
              "DEFAULT_MODE_BIT_REPRODUCIBLE":
                  bool(spread["max_abs_delta"] == 0.0),
              "DETERMINISTIC_MODE_VERIFIED":
                  bool(control["max_abs_delta"] == 0.0)}
    report["TRAINING_PATH_PARITY"] = bool(
        report["DETERMINISTIC_MODE_VERIFIED"]
        and parity["max_abs_delta"] == 0.0)
    return report


def interval(history, low, high, population="D2_LINE_DEV512"):
    a, b = history[str(low)], history[str(high)]
    out = {"from": low, "to": high}
    for key in ("angle_median", "offset_median", "angle_p90", "offset_p90"):
        start, end = a[population][key], b[population][key]
        out[key] = {"from": start, "to": end,
                    "reduction": 1.0 - end / start if start else 0.0}
    ce_start = a["train_loss_mean_last250"]
    ce_end = b["train_loss_mean_last250"]
    out["train_ce"] = {"from": ce_start, "to": ce_end,
                       "drop": ce_start - ce_end,
                       "reduction": 1.0 - ce_end / ce_start if ce_start else 0.0}
    return out


def reproduction(history):
    """PHASE 7: diagnostic only.  No tolerance, no hard block."""
    recorded = recorded_full()["history"][RECORDED_MARK]
    fresh = history[RECORDED_MARK]
    report = {"mark": int(RECORDED_MARK), "source": RECORDED_FULL,
              "note": "diagnostic; component drift is on record as UNRESOLVED"}
    for population in ("D0_SEEN512", "D2_LINE_DEV512"):
        report[population] = {}
        for key in ("angle_median", "angle_p90", "offset_median", "offset_p90"):
            was, now = recorded[population][key], fresh[population][key]
            report[population][key] = {
                "recorded": was, "fresh": now, "delta": now - was,
                "relative": (now - was) / max(abs(was), 1e-12)}
        report[population]["qualification_class"] = {
            "recorded": [recorded[population]["PASS"], recorded[population]["SAFETY"]],
            "fresh": [fresh[population]["PASS"], fresh[population]["SAFETY"]]}
        report[population]["SAME_QUALIFICATION_CLASS"] = bool(
            report[population]["qualification_class"]["recorded"]
            == report[population]["qualification_class"]["fresh"])
    was, now = (recorded["train_loss_mean_last250"],
                fresh["train_loss_mean_last250"])
    report["train_loss_mean_last250"] = {
        "recorded": was, "fresh": now, "delta": now - was,
        "relative": (now - was) / max(abs(was), 1e-12)}
    return report


def verdict(history, limits):
    final = history[str(DECISION_STEP)]["D2_LINE_DEV512"]
    base = limits["baseline_full_precision"]
    late = interval(history, 17030, DECISION_STEP)
    span = interval(history, 8515, DECISION_STEP)
    out = {"decision_step": DECISION_STEP, "population": "D2_LINE_DEV512",
           "angle_reduction_vs_baseline":
               1.0 - final["angle_median"] / base["angle_median"],
           "offset_reduction_vs_baseline":
               1.0 - final["offset_median"] / base["offset_median"],
           "ABSOLUTE_PASS": bool(final["PASS"] and final["SAFETY"]),
           "REDUCTION_40": bool(
               final["angle_median"] <= limits["reduction_40"]["angle_median"]
               and final["offset_median"] <= limits["reduction_40"]["offset_median"]),
           "span_8515_to_25545": span, "late_17030_to_25545": late}
    ce_drop = late["train_ce"]["drop"]
    last_slope = history[str(DECISION_STEP)]["train_loss_slope_last_pass"]
    out["conditions"] = {
        "WEAK_IMPROVEMENT": bool(
            span["angle_median"]["reduction"] < WEAK_IMPROVEMENT
            and span["offset_median"]["reduction"] < WEAK_IMPROVEMENT),
        "GEOMETRY_PLATEAU": bool(
            late["angle_median"]["reduction"] < GEOMETRY_PLATEAU
            and late["offset_median"]["reduction"] < GEOMETRY_PLATEAU),
        "CE_PLATEAU": bool(ce_drop < CE_PLATEAU_DROP
                           and last_slope >= CE_PLATEAU_SLOPE),
        "CE_STRONG_DROP": bool(ce_drop >= CE_STRONG_DROP),
        "ce_drop_late": ce_drop, "ce_slope_last_pass": last_slope}
    if out["ABSOLUTE_PASS"]:
        out["DECISION"] = "DIRECT_HOUGH_LONG_SCHEDULE_VALID_CANDIDATE"
        out["READING"] = "FULL_OPTIMIZER_EXPOSURE_WAS_PRIMARY_LIMIT"
        out["NEXT"] = ["same_protocol_replicate", "role_shuffle",
                       "whole_LINE_DEV"]
        out["CIGM"] = "BLOCKED"
    elif out["REDUCTION_40"]:
        out["DECISION"] = "DIRECT_HOUGH_LONG_SCHEDULE_SIGNAL"
        if not out["conditions"]["CE_PLATEAU"]:
            out["READING"] = "OPTIMIZATION_EXPOSURE_REMAINS_ACTIVE"
        out["NEXT"] = ["architecture_screen"]
        out["FURTHER_STEP_EXTENSION"] = "FORBIDDEN"
    elif (out["conditions"]["WEAK_IMPROVEMENT"]
          and out["conditions"]["CE_PLATEAU"]
          and out["conditions"]["GEOMETRY_PLATEAU"]):
        out["DECISION"] = "DIRECT_HOUGH_FULL_OPTIMIZATION_PLATEAU"
        out["NEXT"] = ["FROZEN_A1_FEATURE_OR_ENCODER_LIMIT"]
    elif out["conditions"]["CE_STRONG_DROP"]:
        out["DECISION"] = "LONG_SCHEDULE_STILL_OPTIMIZING_BUT_TASK_FAIL"
        out["NEXT"] = ["architecture_screen"]
        out["FURTHER_STEP_EXTENSION"] = "FORBIDDEN"
        out["FEATURE_LIMIT_CONFIRMED"] = False
    else:
        out["DECISION"] = "DIRECT_HOUGH_LONG_SCHEDULE_INCONCLUSIVE"
        out["NEXT"] = ["report_conditions_as_they_fell"]
    return out


def build_plan(pool):
    return {"marks": list(LONG_MARKS), "decision_step": DECISION_STEP,
            "decision_population": "D2_LINE_DEV512",
            "diagnostic_population": "D0_SEEN512",
            "diagnostic_marks": list(DIAGNOSTIC_MARKS),
            "per_role_marks": list(PER_ROLE_MARKS),
            "frames": len(pool), "passes": PASSES,
            "exposure": check_exposure(pool),
            "resume": False, "fresh_init": True,
            "recorded_checkpoint": checkpoint_optimizer_state(),
            "extension_allowance": "once",
            "gate": {"angle_median": CAP.ANGLE_BUDGET_DEG,
                     "offset_median": CAP.OFFSET_BUDGET_CELL,
                     "angle_p90": CAP.SAFETY_ANGLE,
                     "offset_p90": CAP.SAFETY_OFFSET},
            "thresholds": DH.thresholds(),
            "preregistered": {"WEAK_IMPROVEMENT": WEAK_IMPROVEMENT,
                              "GEOMETRY_PLATEAU": GEOMETRY_PLATEAU,
                              "CE_PLATEAU_DROP": CE_PLATEAU_DROP,
                              "CE_PLATEAU_SLOPE": CE_PLATEAU_SLOPE,
                              "CE_STRONG_DROP": CE_STRONG_DROP},
            "recorded_full_verdict": recorded_full()["verdict"]["DECISION"],
            **CAP.provenance()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "parity", "run"])
    arguments = parser.parse_args()
    import instance_edge_topology as IET
    edges = [tuple(e) for e in IET.build_topology()["edges"]]
    if not V2.sha_file(OUT / "line_internal_split.csv").startswith(V2.LINE_SPLIT_SHA):
        raise RuntimeError("HARD_BLOCKED: LINE split changed")
    eligibility = OUT / "direct_hough_extension.json"
    if not eligibility.exists() or not json.loads(
            eligibility.read_text())["EXTENDED_PASS"]:
        raise RuntimeError("DIRECT_HOUGH_NETWORK_FIT_FAIL_CONFIRMED: blocked")
    if not (OUT / RECORDED_FULL).exists():
        raise RuntimeError("HARD_BLOCK: recorded FULL is missing")
    pool = V2.split_indices()[0]

    if arguments.command == "plan":
        plan = build_plan(pool)
        (OUT / "direct_hough_long_plan.json").write_text(json.dumps(plan, indent=2))
        limits = plan["thresholds"]
        log(f"[plan] fresh 0 -> {DECISION_STEP}  ({PASSES} passes, "
            f"{plan['exposure']['exposure_ratio']:.0f}x the recorded "
            f"{plan['exposure']['recorded_decision']})")
        log(f"[plan] marks {LONG_MARKS}  decision {DECISION_STEP} only  "
            f"resume {plan['resume']}  optimizer state in recorded ckpt "
            f"{plan['recorded_checkpoint'].get('has_optimizer_state')}")
        log(f"[plan] gate {CAP.ANGLE_BUDGET_DEG}/{CAP.OFFSET_BUDGET_CELL} "
            f"safety {CAP.SAFETY_ANGLE}/{CAP.SAFETY_OFFSET}  40% -> "
            f"{limits['reduction_40']['angle_median']:.6f} / "
            f"{limits['reduction_40']['offset_median']:.6f}")
        return

    a1 = V2.load_a1()

    if arguments.command == "parity":
        report = run_parity(edges, a1)
        (OUT / "direct_hough_long_parity.json").write_text(
            json.dumps(report, indent=2, default=float))
        log(f"[parity] default-mode spread of the locked runner against itself "
            f"{report['default_mode_spread']['max_abs_delta']:.3e} "
            f"({report['default_mode_spread']['nonzero_tensors']}/"
            f"{report['default_mode_spread']['tensors']} tensors)")
        log(f"[parity] deterministic control "
            f"{report['deterministic_control']['max_abs_delta']:.3e}  "
            f"structural {report['structural_parity']['max_abs_delta']:.3e}  "
            f"PARITY={report['TRAINING_PATH_PARITY']}")
        if not report["TRAINING_PATH_PARITY"]:
            raise RuntimeError("TRAINING_PATH_PARITY_FAIL")
        return

    parity = OUT / "direct_hough_long_parity.json"
    if not parity.exists() or not json.loads(
            parity.read_text())["TRAINING_PATH_PARITY"]:
        raise RuntimeError("TRAINING_PATH_PARITY_FAIL: run parity first")
    plan = build_plan(pool)
    per_pass = plan["exposure"]["steps_per_pass"]
    history, _ = train_long(pool, LONG_MARKS, edges, a1, SCALE.populations(),
                            TAG, per_pass)
    limits = plan["thresholds"]
    report = {"plan": plan, "history": history, "thresholds": limits,
              "verdict": verdict(history, limits),
              "reproduction": reproduction(history),
              "parity": json.loads(parity.read_text()), **CAP.provenance()}
    (OUT / "direct_hough_long.json").write_text(
        json.dumps(report, indent=2, default=float))
    final = history[str(DECISION_STEP)]["D2_LINE_DEV512"]
    log(f"[long] {report['verdict']['DECISION']}  angle {final['angle_median']:.6f}"
        f" offset {final['offset_median']:.6f}  p90 {final['angle_p90']:.6f}/"
        f"{final['offset_p90']:.6f}")
    log(f"[long] conditions {report['verdict']['conditions']}")


if __name__ == "__main__":
    main()
