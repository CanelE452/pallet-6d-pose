"""Was the 3,000-step direct-Hough miss premature?

`af3e8cf` records the miss and it stands: three of four gates failed at the
pre-registered decision, so FULL is blocked.  But every metric halved between
1,500 and 3,000 steps, which is a miss at a chosen step count rather than a
converged failure.  This asks that one question and only that one.

```
changed      max optimizer steps, 3,000 -> 6,000
unchanged    the 32 frames, frozen A1 F50, XY, role queries, DirectHoughHead,
             the Hough lattice, the target distribution, the cross-entropy,
             batch, learning rate, weight decay, seed, and every gate
```

The trajectory is fresh from step 0.  The recorded 3,000-step checkpoint stored
no optimizer state, so resuming from it would continue a different AdamW
trajectory than the one being extended -- which would answer a question nobody
asked.

This extension happens once.  A miss at 6,000 is
`DIRECT_HOUGH_NETWORK_FIT_FAIL_CONFIRMED` and there is no 9,000.
"""
from __future__ import annotations
import os as _os, sys as _sys

# --- stage0 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_S0 = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_S0] + [_os.path.join(_S0, _d) for _d in sorted(_os.listdir(_S0))
                         if _os.path.isdir(_os.path.join(_S0, _d)) and not _d.startswith(".")]


import argparse, importlib.util, json, pathlib, sys, time
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


DH = _load("DH_EXT", "scripts/stage0/line/direct_hough_role_heatmap.py")
CAP, V2, SCALE = DH.CAP, DH.V2, DH.SCALE
OUT, DEV = DH.OUT, DH.DEV
EXTENDED_MARKS = (1500, 3000, 4500, 6000)
DECISION_STEP = 6000
HISTORICAL = "direct_hough_overfit.json"
HISTORICAL_MARKS = (1500, 3000)


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def historical_reference():
    """The recorded run at full stored precision -- never a transcription."""
    recorded = json.loads((OUT / HISTORICAL).read_text())["history"]
    return {mark: {key: recorded[str(mark)]["OVERFIT32"][key]
                   for key in ("angle_median", "angle_p90",
                               "offset_median", "offset_p90")}
            for mark in HISTORICAL_MARKS}


def gates(entry):
    return {"angle_median": entry["angle_median"] <= CAP.ANGLE_BUDGET_DEG,
            "offset_median": entry["offset_median"] <= CAP.OFFSET_BUDGET_CELL,
            "angle_p90": entry["angle_p90"] <= CAP.SAFETY_ANGLE,
            "offset_p90": entry["offset_p90"] <= CAP.SAFETY_OFFSET}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "run", "full"])
    arguments = parser.parse_args()
    import instance_edge_topology as IET
    edges = [tuple(e) for e in IET.build_topology()["edges"]]
    if not V2.sha_file(OUT / "line_internal_split.csv").startswith(V2.LINE_SPLIT_SHA):
        raise RuntimeError("HARD_BLOCKED: LINE split changed")
    pool = V2.split_indices()[0][:DH.OVERFIT_FRAMES]
    plan = {"marks": list(EXTENDED_MARKS), "decision_step": DECISION_STEP,
            "frames": len(pool), "historical": historical_reference(),
            "gate": {"angle_median": CAP.ANGLE_BUDGET_DEG,
                     "offset_median": CAP.OFFSET_BUDGET_CELL,
                     "angle_p90": CAP.SAFETY_ANGLE,
                     "offset_p90": CAP.SAFETY_OFFSET},
            "resume": False, "extension_allowance": "once", **CAP.provenance()}

    if arguments.command == "plan":
        (OUT / "direct_hough_extension_plan.json").write_text(json.dumps(plan, indent=2))
        log(f"[plan] fresh 0 -> {DECISION_STEP}  marks {EXTENDED_MARKS}  "
            f"decision {DECISION_STEP} only  gate unchanged")
        return

    if arguments.command == "run":
        a1 = V2.load_a1()
        history, _ = DH.train_network(pool, EXTENDED_MARKS, edges, a1,
                                      {"OVERFIT32": pool}, "overfit_ext")
        final = history[str(DECISION_STEP)]["OVERFIT32"]
        report = {"plan": plan, "history": history,
                  "decision_step": DECISION_STEP, "gates": gates(final)}
        report["EXTENDED_PASS"] = all(report["gates"].values())
        reference = historical_reference()
        report["reproduction"] = {}
        for mark in HISTORICAL_MARKS:
            entry = history[str(mark)]["OVERFIT32"]
            report["reproduction"][str(mark)] = {
                key: {"new": entry[key], "recorded": reference[mark][key],
                      "delta": entry[key] - reference[mark][key],
                      "relative": (entry[key] - reference[mark][key])
                      / max(abs(reference[mark][key]), 1e-12)}
                for key in reference[mark]}
        report["DECISION"] = ("DIRECT_HOUGH_OVERFIT_EXTENDED_PASS"
                              if report["EXTENDED_PASS"]
                              else "DIRECT_HOUGH_NETWORK_FIT_FAIL_CONFIRMED")
        (OUT / "direct_hough_extension.json").write_text(
            json.dumps(report, indent=2, default=float))
        for mark in EXTENDED_MARKS:
            entry = history[str(mark)]["OVERFIT32"]
            log(f"[ext] @{mark:5d} angle med {entry['angle_median']:.6f} p90 "
                f"{entry['angle_p90']:.6f} | offset med {entry['offset_median']:.6f}"
                f" p90 {entry['offset_p90']:.6f}")
        log(f"[ext] gates {report['gates']}  {report['DECISION']}")
        if not report["EXTENDED_PASS"]:
            raise RuntimeError("DIRECT_HOUGH_NETWORK_FIT_FAIL_CONFIRMED")
        return

    # FULL runs only on the extension's eligibility, and never overwrites the
    # historical overfit record.
    extension = OUT / "direct_hough_extension.json"
    if not extension.exists() or not json.loads(extension.read_text())["EXTENDED_PASS"]:
        raise RuntimeError("DIRECT_HOUGH_NETWORK_FIT_FAIL_CONFIRMED: FULL blocked")
    a1 = V2.load_a1()
    populations = SCALE.populations()
    history, model = DH.train_network(V2.split_indices()[0], DH.MARKS, edges, a1,
                                      populations, "full")
    limits = DH.thresholds()
    final = history[str(max(DH.MARKS))]["D2_LINE_DEV512"]
    base = limits["baseline_full_precision"]
    verdict = {"angle_reduction": 1.0 - final["angle_median"] / base["angle_median"],
               "offset_reduction": 1.0 - final["offset_median"] / base["offset_median"],
               "ABSOLUTE_PASS": bool(final["PASS"] and final["SAFETY"])}
    verdict["REDUCTION_40"] = bool(
        final["angle_median"] <= limits["reduction_40"]["angle_median"]
        and final["offset_median"] <= limits["reduction_40"]["offset_median"])
    verdict["DECISION"] = ("DIRECT_HOUGH_ROLE_HEATMAP_VALID" if verdict["ABSOLUTE_PASS"]
                           else "DIRECT_HOUGH_LINE_NATIVE_SIGNAL" if verdict["REDUCTION_40"]
                           else "DIRECT_HOUGH_ROLE_HEATMAP_FAIL")
    report = {"history": history, "thresholds": limits, "verdict": verdict,
              "eligibility": "direct_hough_extension.json", **CAP.provenance()}
    if verdict["DECISION"] != "DIRECT_HOUGH_ROLE_HEATMAP_FAIL":
        grid_theta, grid_rho, valid = DH.lattice()
        features = DH.hypothesis_features(grid_theta, grid_rho)
        shuffled = DH.evaluate_network(populations["D2_LINE_DEV512"], model, a1,
                                       edges, features, grid_theta, grid_rho,
                                       valid, permute=CAP.DERANGEMENT)
        report["shuffle"] = {
            "normal": final, "shuffled": shuffled,
            "angle_degradation": shuffled["angle_median"] - final["angle_median"],
            "offset_degradation": shuffled["offset_median"] - final["offset_median"]}
        report["shuffle"]["ROLE_IDENTITY_CAUSAL"] = bool(
            report["shuffle"]["angle_degradation"] >= CAP.SHUFFLE_ANGLE_MARGIN
            or report["shuffle"]["offset_degradation"] >= CAP.SHUFFLE_OFFSET_MARGIN)
    (OUT / "direct_hough_full.json").write_text(json.dumps(report, indent=2,
                                                            default=float))
    log(f"[full] {verdict['DECISION']}  angle {final['angle_median']:.6f} "
        f"offset {final['offset_median']:.6f}")


if __name__ == "__main__":
    main()
