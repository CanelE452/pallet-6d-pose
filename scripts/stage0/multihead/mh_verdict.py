"""PHASE 8 -- apply the pre-registered decision tree to whatever has been measured.

The gates are copied from the brief and are not editable here: this file reads
the diagnostic JSONs and reports which CASE the evidence lands in.  Writing the
tree as code rather than prose is the point -- a verdict assembled by hand after
seeing the numbers is a verdict that can drift toward whatever the numbers
suggest.

    CASE A  SHARED_GRADIENT_INTERFERENCE      conflict AND stop-grad recovers
    CASE B  LATE_FEATURE_OPTIMUM_MISMATCH     no conflict AND split-late works
    CASE C  WEAK_REPRESENTATION_COMPLEMENTARITY  nothing recovers the line and the
                                              native solver also fails
    CASE D  CIGM_INTERSECTION_BOTTLENECK      CIGM fusion weak BUT native
                                              point-line solver passes
    CASE E  SYSTEMATIC_CORNER_GEOMETRY_BIAS   strong structured residual (additive)
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import mh_data as MD                                            # noqa: E402

OUT = MD.OUT
SEEDS = (1, 2)
ARMS = ("E0_CONTINUE_LINE", "E1_SHARED_CORNER_LINE", "E2_STOPGRAD_CORNER")


def _load(name):
    path = OUT / name
    return json.loads(path.read_text()) if path.exists() else None


def stopgrad_table():
    """Final-mark line metrics for every arm and seed that finished."""
    table = {}
    for seed in SEEDS:
        for arm in ARMS:
            path = OUT / f"stopgrad_{arm}_seed{seed}.json"
            if not path.exists():
                continue
            history = json.loads(path.read_text())
            marks = [k for k in history if k.isdigit()]
            if not marks:
                continue
            last = max(marks, key=int)
            block = history[last]["D2_MH_DEV512"]
            table[(arm, seed)] = {
                "step": int(last),
                "angle": block["line"]["angle_median"],
                "offset": block["line"]["offset_median"],
                "angle_p90": block["line"]["angle_p90"],
                "cigm": block["corner"]["cigm_cell_median"],
                "direct": block["corner"].get("direct_cell_median"),
                "start_direct": history["0"]["D2_MH_DEV512"]["corner"].get(
                    "direct_cell_median"),
            }
    return table


def _rel(base, candidate):
    if base in (None, 0) or candidate is None:
        return None
    return 100.0 * (base - candidate) / abs(base)


def stopgrad_gates(table):
    """E2 vs E1 and E2 vs E0, exactly as the brief words them."""
    out = {}
    for seed in SEEDS:
        if not all((arm, seed) in table for arm in ARMS):
            out[f"seed{seed}"] = {"complete": False, "reason": "arm missing"}
            continue
        e0, e1, e2 = (table[(arm, seed)] for arm in ARMS)
        # An arm still mid-run has a lower last mark than the others.  Comparing
        # across different step counts would read training progress as an arm
        # effect, which is exactly the confound this screen exists to avoid.
        steps = {e0["step"], e1["step"], e2["step"]}
        if len(steps) != 1:
            out[f"seed{seed}"] = {"complete": False,
                                  "reason": f"steps differ {sorted(steps)}"}
            continue
        angle_21 = _rel(e1["angle"], e2["angle"])
        offset_21 = _rel(e1["offset"], e2["offset"])
        angle_20 = _rel(e0["angle"], e2["angle"])
        offset_20 = _rel(e0["offset"], e2["offset"])
        out[f"seed{seed}"] = {
            "complete": True,
            "E1_vs_E0_angle": _rel(e0["angle"], e1["angle"]),
            "E1_vs_E0_offset": _rel(e0["offset"], e1["offset"]),
            "E2_vs_E1_angle": angle_21,
            "E2_vs_E1_offset": offset_21,
            "E2_vs_E0_angle": angle_20,
            "E2_vs_E0_offset": offset_20,
            # brief: E2 >= 3% better than E1 on both axes AND within +-3% of E0
            "CORNER_GRADIENT_HURTS_LINE": bool(
                angle_21 is not None and offset_21 is not None
                and angle_21 >= 3.0 and offset_21 >= 3.0
                and abs(angle_20) <= 3.0 and abs(offset_20) <= 3.0),
            "STOP_GRAD_DOES_NOT_RECOVER_LINE": bool(
                angle_20 is not None and (angle_20 < -3.0 or offset_20 < -3.0)),
        }
    return out


def decide():
    gradient = _load("gradient_conflict.json")
    residual = _load("corner_residual_modes.json")
    solver = _load("point_line_solver.json")
    uncertainty = _load("line_uncertainty.json")
    table = stopgrad_table()
    gates = stopgrad_gates(table)

    conflict = bool(gradient and gradient.get("GRADIENT_CONFLICT_SUPPORTED"))
    solver_pass = bool(solver and solver.get("POINT_LINE_SOLVER_SIGNAL"))
    systematic = bool(residual
                      and residual.get("SYSTEMATIC_CORNER_BIAS_SUPPORTED"))
    line_conf = bool(uncertainty
                     and uncertainty.get("LINE_UNCERTAINTY_SIGNAL"))
    complete = [s for s in SEEDS if gates.get(f"seed{s}", {}).get("complete")]
    hurts = [s for s in complete
             if gates[f"seed{s}"]["CORNER_GRADIENT_HURTS_LINE"]]
    no_recover = [s for s in complete
                  if gates[f"seed{s}"]["STOP_GRAD_DOES_NOT_RECOVER_LINE"]]

    causes = []
    if conflict and hurts:
        causes.append("SHARED_GRADIENT_INTERFERENCE")
    if solver_pass:
        causes.append("CIGM_INTERSECTION_BOTTLENECK")
    if systematic:
        causes.append("SYSTEMATIC_CORNER_GEOMETRY_BIAS")
    if complete and not hurts and not solver_pass:
        causes.append("WEAK_REPRESENTATION_COMPLEMENTARITY")

    return {
        "inputs": {
            "GRADIENT_CONFLICT_SUPPORTED": conflict,
            "POINT_LINE_SOLVER_SIGNAL": solver_pass,
            "SYSTEMATIC_CORNER_BIAS_SUPPORTED": systematic,
            "LINE_UNCERTAINTY_SIGNAL": line_conf,
            "stopgrad_seeds_complete": complete,
            "CORNER_GRADIENT_HURTS_LINE_seeds": hurts,
            "STOP_GRAD_DOES_NOT_RECOVER_LINE_seeds": no_recover,
        },
        "stopgrad": {f"{a}|seed{s}": v for (a, s), v in table.items()},
        "stopgrad_gates": gates,
        "ROOT_CAUSE": causes or ["PENDING_STOPGRAD"],
        "PHASE4_SPLIT_LATE_ELIGIBLE": bool(conflict or hurts),
    }


# --------------------------------------------------------------------------
# PHASE 1 -- close E3 against E0 and E2


E_ARMS = {"E0_CONTINUE_LINE": "stopgrad", "E1_SHARED_CORNER_LINE": "stopgrad",
          "E2_STOPGRAD_CORNER": "stopgrad", "E3_SPLIT_LATE": "splitlate"}


def _e_rows(arm, seed, step="3000"):
    prefix = E_ARMS[arm]
    path = OUT / f"{prefix}_{arm}_seed{seed}.json"
    if not path.exists():
        return None
    history = json.loads(path.read_text())
    if step not in history:
        return None
    return history[step]["D2_MH_DEV512"]


def _pose_c(block):
    """PATH-C pose from the stored per-frame rows."""
    rows = block.get("rows")
    if not rows or "pose_C" not in rows[0]:
        return {}
    solved = [r["pose_C"] for r in rows
              if r.get("pose_C", {}).get("solved")]
    if not solved:
        return {}
    R = np.array([e["R_deg"] for e in solved], float)
    t = np.array([e["t_m"] for e in solved], float)
    return {"n": len(rows), "solve_rate": round(len(solved) / len(rows), 4),
            "R_median": float(np.median(R)), "R_p90": float(np.percentile(R, 90)),
            "t_median": float(np.median(t)), "t_p90": float(np.percentile(t, 90)),
            "success_5cm5deg": round(float(
                ((R <= 5.0) & (t <= 0.05)).sum() / len(rows)), 4)}


def e3_report():
    out = {"note": "front_rear_shift / isotropic scale / non-affine RMS are "
                   "unavailable for the E arms: those continuation runners "
                   "stored metrics but not weights, and a median cannot be "
                   "un-medianed. Checkpoint saving has since been added, so "
                   "seed 2 onward will carry them.",
           "seeds": {}}
    for seed in SEEDS:
        blocks = {arm: _e_rows(arm, seed) for arm in E_ARMS}
        if not all(blocks.values()):
            out["seeds"][f"seed{seed}"] = {
                "complete": False,
                "missing": [a for a, b in blocks.items() if b is None]}
            continue
        entry = {"complete": True, "arms": {}}
        for arm, block in blocks.items():
            entry["arms"][arm] = {
                "angle_median": block["line"]["angle_median"],
                "angle_p90": block["line"]["angle_p90"],
                "offset_median": block["line"]["offset_median"],
                "offset_p90": block["line"]["offset_p90"],
                "cigm_median": block["corner"]["cigm_cell_median"],
                "direct_median": block["corner"].get("direct_cell_median"),
                "pose_C": _pose_c(block),
            }
        a = entry["arms"]
        e0, e2, e3 = a["E0_CONTINUE_LINE"], a["E2_STOPGRAD_CORNER"], a["E3_SPLIT_LATE"]

        def rel(base, cand):
            if base in (None, 0) or cand is None:
                return None
            return round(100.0 * (base - cand) / abs(base), 3)

        # 1A -- is the line preserved?
        entry["line_preservation"] = {
            "E3_vs_E0_angle": rel(e0["angle_median"], e3["angle_median"]),
            "E3_vs_E0_offset": rel(e0["offset_median"], e3["offset_median"]),
            "E3_vs_E0_angle_p90": rel(e0["angle_p90"], e3["angle_p90"]),
            "E3_vs_E0_offset_p90": rel(e0["offset_p90"], e3["offset_p90"]),
            "E3_vs_E0_cigm": rel(e0["cigm_median"], e3["cigm_median"]),
            "E3_vs_E2_angle": rel(e2["angle_median"], e3["angle_median"]),
            "E3_vs_E2_offset": rel(e2["offset_median"], e3["offset_median"]),
            # pre-registered: degradation <= 2% against E0
            "PRESERVES_LINE": bool(
                rel(e0["angle_median"], e3["angle_median"]) >= -2.0
                and rel(e0["offset_median"], e3["offset_median"]) >= -2.0),
        }
        # 1B -- localisation gain and pose/geometry gain are judged separately
        start = 21.7  # A0 never trained the belief stages; both arms start here
        entry["corner"] = {
            "E3_vs_E2_direct": rel(e2["direct_median"], e3["direct_median"]),
            "E3_vs_start_direct": rel(start, e3["direct_median"]),
            "E3_vs_E2_pose_R": rel(e2["pose_C"].get("R_median"),
                                   e3["pose_C"].get("R_median")),
            "E3_vs_E2_pose_R_p90": rel(e2["pose_C"].get("R_p90"),
                                       e3["pose_C"].get("R_p90")),
            "E3_vs_E2_pose_t": rel(e2["pose_C"].get("t_median"),
                                   e3["pose_C"].get("t_median")),
            "E3_vs_E2_success_pp": round(100.0 * (
                e3["pose_C"].get("success_5cm5deg", 0)
                - e2["pose_C"].get("success_5cm5deg", 0)), 3),
        }
        c = entry["corner"]
        entry["E3_LOCALIZATION_GAIN"] = bool(
            c["E3_vs_E2_direct"] is not None and c["E3_vs_E2_direct"] >= 5.0)
        # the pose bottleneck is systematic geometry, so a localisation gain that
        # does not reach R/t is explicitly not a geometry gain
        entry["E3_POSE_GEOMETRY_GAIN"] = bool(
            c["E3_vs_E2_pose_R"] is not None
            and c["E3_vs_E2_pose_R"] >= 5.0
            and c["E3_vs_E2_pose_t"] is not None
            and c["E3_vs_E2_pose_t"] >= 0.0)
        out["seeds"][f"seed{seed}"] = entry

    done = [s for s in SEEDS
            if out["seeds"].get(f"seed{s}", {}).get("complete")]
    out["E3_LOCALIZATION_GAIN"] = bool(
        done and all(out["seeds"][f"seed{s}"]["E3_LOCALIZATION_GAIN"]
                     for s in done))
    out["E3_POSE_GEOMETRY_GAIN"] = bool(
        done and all(out["seeds"][f"seed{s}"]["E3_POSE_GEOMETRY_GAIN"]
                     for s in done))
    out["seeds_complete"] = done
    return out


def print_e3(report):
    print("\n===== PHASE 1: E3 against E0 and E2 (step 3000) =====")
    for seed in SEEDS:
        entry = report["seeds"].get(f"seed{seed}", {})
        if not entry.get("complete"):
            print(f"\nseed {seed}: incomplete {entry.get('missing', '')}")
            continue
        print(f"\nseed {seed}")
        print(f"  {'arm':<24}{'angle':>9}{'ang p90':>9}{'offset':>9}"
              f"{'CIGM':>9}{'direct':>9}{'R med':>8}{'R p90':>8}"
              f"{'t med':>8}{'5cm5':>8}")
        for arm, v in entry["arms"].items():
            pose = v["pose_C"]
            print(f"  {arm:<24}{v['angle_median']:>9.4f}{v['angle_p90']:>9.3f}"
                  f"{v['offset_median']:>9.4f}{v['cigm_median']:>9.4f}"
                  f"{(v['direct_median'] or float('nan')):>9.4f}"
                  f"{pose.get('R_median', float('nan')):>8.3f}"
                  f"{pose.get('R_p90', float('nan')):>8.2f}"
                  f"{pose.get('t_median', float('nan')):>8.4f}"
                  f"{pose.get('success_5cm5deg', float('nan')):>8.4f}")
        print("  line: " + json.dumps(entry["line_preservation"]))
        print("  corner: " + json.dumps(entry["corner"]))
        print(f"  E3_LOCALIZATION_GAIN={entry['E3_LOCALIZATION_GAIN']}  "
              f"E3_POSE_GEOMETRY_GAIN={entry['E3_POSE_GEOMETRY_GAIN']}")
    print(f"\noverall  E3_LOCALIZATION_GAIN={report['E3_LOCALIZATION_GAIN']}  "
          f"E3_POSE_GEOMETRY_GAIN={report['E3_POSE_GEOMETRY_GAIN']}  "
          f"seeds={report['seeds_complete']}")
    print("  " + report["note"])


# --------------------------------------------------------------------------
# PHASE 7 -- pick exactly one 2-head architecture, by the stated priority order


def choose_architecture():
    """1. does it improve the final pose  2. does it preserve line-only
    3. does the corner branch improve pose-critical geometry  4. is the extra
    parameter cost justified.  Written as code so the choice cannot drift."""
    e3 = _load("mh_e3_report.json") or {}
    solver = _load("point_line_solver.json") or {}
    scale = _load("scale_oracle.json") or {}
    signed = _load("line_signed_bias.json") or {}
    gradient = _load("gradient_conflict.json") or {}

    done = e3.get("seeds_complete", [])
    both = len(done) == 2
    e3_line_ok = all(
        e3["seeds"][f"seed{s}"]["line_preservation"]["PRESERVES_LINE"]
        for s in done) if done else False
    e3_pose = all(e3["seeds"][f"seed{s}"]["E3_POSE_GEOMETRY_GAIN"]
                  for s in done) if done else False
    e3_local = all(e3["seeds"][f"seed{s}"]["E3_LOCALIZATION_GAIN"]
                   for s in done) if done else False

    if not both:
        choice, why = "PENDING_E3_SEED2", (
            "one seed decided nothing twice already in this study; the choice "
            "waits for the second seed")
    elif e3_line_ok and e3_pose:
        choice, why = "SPLIT_LATE_2HEAD", (
            "line preserved exactly by construction, and the corner branch "
            "improves PATH-C rotation and translation, not only its own median")
    elif e3_line_ok and e3_local and not e3_pose:
        choice, why = "STOPGRAD_2HEAD", (
            "split-late buys corner localisation that does not reach the pose, "
            "so the duplicated late block is not justified")
    else:
        choice, why = "STOPGRAD_2HEAD", (
            "stop-grad is free on the line side by construction and gives a "
            "working corner head; nothing measured beats it")

    # pose solver, by what the oracles actually showed
    if scale.get("SCALE_EXPLAINS_TRANSLATION_LOSS") and \
            not scale.get("CONSTANT_SCALE_IS_ENOUGH"):
        pose_solver = "PATH_C_PLUS_SCALE_CORRECTION_RESEARCH"
        pose_why = ("the point branch's per-frame isotropic scale error is the "
                    "largest single lever on translation (+33-34%), larger than "
                    "anything the line branch contributes, and a constant "
                    "correction captures almost none of it")
    elif solver.get("POINT_LINE_SOLVER_SIGNAL"):
        pose_solver, pose_why = "NATIVE_POINT_LINE", "solver gate passed"
    else:
        pose_solver, pose_why = "PATH_C_ONLY", "no line formulation beat points"

    return {
        "seeds_complete": done,
        "E3_PRESERVES_LINE": e3_line_ok,
        "E3_LOCALIZATION_GAIN": e3_local,
        "E3_POSE_GEOMETRY_GAIN": e3_pose,
        "GRADIENT_CONFLICT_SUPPORTED": bool(
            gradient.get("GRADIENT_CONFLICT_SUPPORTED")),
        "POINT_LINE_SOLVER_SIGNAL": bool(
            solver.get("POINT_LINE_SOLVER_SIGNAL")),
        "SCALE_EXPLAINS_TRANSLATION_LOSS": bool(
            scale.get("SCALE_EXPLAINS_TRANSLATION_LOSS")),
        "CONSTANT_SCALE_IS_ENOUGH": bool(scale.get("CONSTANT_SCALE_IS_ENOUGH")),
        "UNCERTAINTY_CANNOT_FIX_SYSTEMATIC_RHO_BIAS": bool(
            signed.get("UNCERTAINTY_CANNOT_FIX_SYSTEMATIC_RHO_BIAS")),
        "TWO_HEAD_DECISION": choice,
        "TWO_HEAD_REASON": why,
        "POSE_SOLVER": pose_solver,
        "POSE_SOLVER_REASON": pose_why,
        "LONG_CONFIRM": "RUN" if (both and e3_line_ok and e3_pose)
                        else "DO_NOT_RUN",
        "LONG_CONFIRM_CANDIDATE": ("E3_SPLIT_LATE at 25k x 2 seed"
                                   if (both and e3_line_ok and e3_pose) else None),
    }


def main():
    report = e3_report()
    (OUT / "mh_e3_report.json").write_text(json.dumps(report, indent=1))
    print_e3(report)
    verdict = decide()
    (OUT / "mh_root_cause.json").write_text(json.dumps(verdict, indent=1))

    table = stopgrad_table()
    if table:
        print(f"{'arm':<24}{'seed':>5}{'step':>7}{'angle':>9}{'offset':>9}"
              f"{'ang p90':>9}{'CIGM':>9}{'direct':>9}{'dir@0':>9}")
        for (arm, seed), v in sorted(table.items()):
            print(f"{arm:<24}{seed:>5}{v['step']:>7}{v['angle']:>9.4f}"
                  f"{v['offset']:>9.4f}{v['angle_p90']:>9.3f}{v['cigm']:>9.4f}"
                  f"{(v['direct'] or float('nan')):>9.4f}"
                  f"{(v['start_direct'] or float('nan')):>9.4f}")
    else:
        print("stop-grad screen has produced nothing yet")

    print()
    for key, value in verdict["inputs"].items():
        print(f"  {key:<42} {value}")
    print()
    for seed in SEEDS:
        block = verdict["stopgrad_gates"].get(f"seed{seed}", {})
        if block.get("complete"):
            print(f"  seed {seed}: " + json.dumps(
                {k: (round(v, 2) if isinstance(v, float) else v)
                 for k, v in block.items() if k != "complete"}))
    print()
    print("ROOT_CAUSE =", verdict["ROOT_CAUSE"])
    print("PHASE4_SPLIT_LATE_ELIGIBLE =", verdict["PHASE4_SPLIT_LATE_ELIGIBLE"])
    print("->", OUT / "mh_root_cause.json")

    choice = choose_architecture()
    (OUT / "mh_architecture_choice.json").write_text(json.dumps(choice, indent=1))
    print("\n===== PHASE 7: architecture choice =====")
    for key, value in choice.items():
        print(f"  {key:<44} {value}")
    print("->", OUT / "mh_architecture_choice.json")


if __name__ == "__main__":
    main()
