"""Is the underfit a shortage of optimizer steps or of data?

`13ca73d` showed the search2k checkpoint fails on frames it trained on, so the
question is no longer generalization.  This separates the two remaining causes
with M0 alone: two trajectories, one on the 2,000-frame pool and one on the full
13,618, each run to the same 8,515 optimizer steps and read at the same points.

The cleanest data contrast is C against D -- identical step budget, one pool
repeated about 34 times against the other repeated 5.  B is *not* a data-scale
condition: 1,250 steps at batch 8 is 10,000 example exposures, so the FULL
trajectory has not finished even one pass at that point.  It is named
B_FULL_PREFIX_SHORT for that reason and read as a short-budget diagnostic.

Architecture, target, loss, decoder, MAP100, sigma, population, batch, optimizer,
learning rate and seed are all as locked.  Nothing is filtered or deleted.
"""
from __future__ import annotations

import argparse, collections, csv, importlib.util, json, pathlib, sys, time
import numpy as np, torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
for _e in ("scripts/stage0", "Deep_Object_Pose/common", "challenge/scripts"):
    if str(ROOT / _e) not in sys.path:
        sys.path.insert(0, str(ROOT / _e))


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CAP = _load("CAP_SCALE", "scripts/stage0/supporting_line_map_capacity.py")
H, V2, OUT, DEV = CAP.H, CAP.V2, CAP.OUT, CAP.DEV
ARM = "M0_F50_SLINE"
EXPECTED = {"K2_steps_per_pass": 250, "FULL_steps_per_pass": 1703,
            "S_SHORT": 1250, "S_LONG": 8515}
MARKS = (1250, 2500, 5000, 8515)
CONDITIONS = {"A_K2_SHORT": ("K2", 1250), "C_K2_LONG": ("K2", 8515),
              "B_FULL_PREFIX_SHORT": ("FULL", 1250), "D_FULL_LONG": ("FULL", 8515)}
REPRODUCTION = {"D0_SEEN512": (6.6040, 2.7023), "D2_LINE_DEV512": (6.8450, 2.7717)}
REPRODUCTION_TOLERANCE = 1e-6
REDUCTION_THRESHOLD = 0.40
LOSS_WINDOW = 250


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def pools():
    train, _ = V2.split_indices()
    return {"K2": V2.manifest("line_search2k"), "FULL": train}


def populations():
    return {"D0_SEEN512": [r["index"] for r in
                           csv.DictReader(open(OUT / "d0_seen512_manifest.csv"))],
            "D2_LINE_DEV512": V2.manifest("line_dev512")}


def check_steps(pool_map):
    got = {"K2_steps_per_pass": V2.steps_per_pass(pool_map["K2"], CAP.BATCH),
           "FULL_steps_per_pass": V2.steps_per_pass(pool_map["FULL"], CAP.BATCH)}
    got["S_SHORT"] = got["K2_steps_per_pass"] * max(CAP.EPOCH_LADDER)
    got["FULL_S_LONG"] = got["FULL_steps_per_pass"] * max(CAP.EPOCH_LADDER)
    got["S_LONG"] = got["FULL_S_LONG"]
    for key, want in EXPECTED.items():
        if got[key] != want:
            raise RuntimeError(f"HARD_BLOCK step count {key}: {got[key]} != {want}")
    return got


def exposure(pool, steps, group):
    """What the trajectory actually saw -- named rather than assumed."""
    visits = collections.Counter()
    for chunk, _ in V2.step_schedule(pool, steps, CAP.BATCH):
        visits.update(chunk)
    counts = np.array(sorted(visits.values())) if visits else np.zeros(1)
    seen = set(visits)
    return {"steps": steps, "example_exposures": int(sum(visits.values())),
            "unique_frames_seen": len(seen), "pool_frames": len(pool),
            "unique_groups_seen": len({group[i] for i in seen}),
            "pool_groups": len({group[i] for i in pool}),
            "frame_visit": {f"p{p}": float(np.percentile(counts, p))
                            for p in (0, 10, 50, 90, 100)},
            "unseen_frames": len(pool) - len(seen),
            "unseen_groups": len({group[i] for i in pool}) - len({group[i] for i in seen})}


def run_trajectory(name, pool, group, edges, coarse, xx, yy, a1):
    head, stem, parameters = CAP.build_arm(ARM)
    optimiser = torch.optim.AdamW(parameters, lr=CAP.LR, weight_decay=CAP.WD)
    history, losses = {}, []
    done = 0
    for chunk, visit in V2.step_schedule(pool, max(MARKS), CAP.BATCH):
        head.train()
        pack = V2.load_pack(chunk)
        _, _, seg, target = CAP.geometry(pack, edges)
        loss = CAP.map_loss(head(CAP.features(pack, a1, stem)), target,
                            torch.tensor(seg["hit"], device=DEV))
        optimiser.zero_grad(set_to_none=True)
        loss.backward(); optimiser.step()
        losses.append(float(loss.detach()))
        done += 1
        if done in MARKS:
            entry = {"step": done,
                     "train_loss_mean_last250": float(np.mean(losses[-LOSS_WINDOW:])),
                     "train_loss_slope_last250": float(
                         np.polyfit(np.arange(len(losses[-LOSS_WINDOW:])),
                                    losses[-LOSS_WINDOW:], 1)[0])}
            for label, indices in populations().items():
                entry[label] = CAP.evaluate(indices, head, stem, a1, edges,
                                            coarse, xx, yy)
                log(f"  {name} @{done:5d} {label:<16} angle med "
                    f"{entry[label]['angle_median']:7.4f} p90 "
                    f"{entry[label]['angle_p90']:7.3f} | offset med "
                    f"{entry[label]['offset_median']:7.4f}  PASS="
                    f"{entry[label]['PASS']}")
            torch.save({"arm": ARM, "trajectory": name, "step": done,
                        "model": head.state_dict(),
                        "stem": None if stem is None else stem.state_dict(),
                        "optimizer": optimiser.state_dict(), **CAP.provenance()},
                       CAP.checkpoint_path(f"{ARM}_{name}", f"step_{done:05d}"))
            history[str(done)] = entry
    return history


def reduction(base, later, key):
    return 1.0 - later[key] / max(base[key], 1e-12)


def decide(report):
    a = report["conditions"]["A_K2_SHORT"]["D0_SEEN512"]
    c = report["conditions"]["C_K2_LONG"]["D0_SEEN512"]
    c2 = report["conditions"]["C_K2_LONG"]["D2_LINE_DEV512"]
    d = report["conditions"]["D_FULL_LONG"]["D0_SEEN512"]
    slope = report["conditions"]["C_K2_LONG"]["train_loss_slope_last250"]
    verdict = {"A_to_C_angle_reduction": reduction(a, c, "angle_median"),
               "A_to_C_offset_reduction": reduction(a, c, "offset_median"),
               "C_to_D_angle_reduction": reduction(c, d, "angle_median"),
               "C_to_D_offset_reduction": reduction(c, d, "offset_median"),
               "C_final_loss_slope": slope}
    step_signal = (verdict["A_to_C_angle_reduction"] >= REDUCTION_THRESHOLD
                   and verdict["A_to_C_offset_reduction"] >= REDUCTION_THRESHOLD)
    data_signal = (verdict["C_to_D_angle_reduction"] >= REDUCTION_THRESHOLD
                   and verdict["C_to_D_offset_reduction"] >= REDUCTION_THRESHOLD)
    if c["PASS"]:
        verdict["CAUSE"] = "OPTIMIZATION_STEPS_RESCUE_MAP_FIT"
        verdict["K2_LONG_MAP_VALID"] = bool(c["PASS"] and c2["PASS"])
    elif step_signal:
        verdict["CAUSE"] = "OPTIMIZATION_SIGNAL_PRESENT_BUT_INSUFFICIENT"
        verdict["OPTIMIZATION_NOT_CONVERGED"] = bool(slope < 0)
    elif d["PASS"] or data_signal:
        verdict["CAUSE"] = "DATA_POOL_SCALE_HELPS"
    elif (c["angle_median"] > CAP.APPROACH_ANGLE
          and d["angle_median"] > CAP.APPROACH_ANGLE
          and c["offset_median"] > CAP.APPROACH_OFFSET
          and d["offset_median"] > CAP.APPROACH_OFFSET and slope >= 0):
        verdict["CAUSE"] = "LOCKED_MAP_TRAINING_RECIPE_FAIL"
    else:
        verdict["CAUSE"] = "DATA_SCALE_NOT_THE_BOTTLENECK"
        verdict["K2_STEP_SCALE_WEAK"] = not step_signal
    return verdict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "run"])
    arguments = parser.parse_args()
    import instance_edge_topology as IET
    edges = [tuple(e) for e in IET.build_topology()["edges"]]
    if not V2.sha_file(OUT / "line_internal_split.csv").startswith(V2.LINE_SPLIT_SHA):
        raise RuntimeError("HARD_BLOCKED: LINE split changed")
    pool_map = pools()
    steps = check_steps(pool_map)
    group = {row["index"]: row["group_id"] for row in
             csv.DictReader(open(OUT / "line_internal_split.csv"))}
    plan = {"steps": steps, "marks": list(MARKS), "arm": ARM,
            "conditions": {k: {"pool": v[0], "step": v[1]}
                           for k, v in CONDITIONS.items()},
            "exposure": {k: exposure(pool_map[v[0]], v[1], group)
                         for k, v in CONDITIONS.items()},
            "reproduction_target": REPRODUCTION,
            "reduction_threshold": REDUCTION_THRESHOLD, **CAP.provenance()}

    if arguments.command == "plan":
        (OUT / "data_vs_step_plan.json").write_text(json.dumps(plan, indent=2))
        for name, entry in plan["exposure"].items():
            log(f"[plan] {name:<20} steps {entry['steps']:5d} exposures "
                f"{entry['example_exposures']:6d} unique {entry['unique_frames_seen']:6d}"
                f"/{entry['pool_frames']}  groups {entry['unique_groups_seen']:3d}"
                f"/{entry['pool_groups']}  visits p50 {entry['frame_visit']['p50']:.0f}"
                f"  unseen {entry['unseen_frames']}")
        return

    (OUT / "data_vs_step_plan.json").write_text(json.dumps(plan, indent=2))
    coarse, (xx, yy) = H.CoarseRadon(), H.pixel_coordinates()
    a1 = V2.load_a1()
    report = {"plan": plan, "trajectories": {}, "conditions": {}}
    for trajectory in ("K2", "FULL"):
        log(f"[run] trajectory {trajectory}")
        report["trajectories"][trajectory] = run_trajectory(
            trajectory, pool_map[trajectory], group, edges, coarse, xx, yy, a1)
        (OUT / "data_vs_step_result.json").write_text(json.dumps(report, indent=2,
                                                                 default=float))
    for name, (trajectory, step) in CONDITIONS.items():
        report["conditions"][name] = report["trajectories"][trajectory][str(step)]
    a = report["conditions"]["A_K2_SHORT"]
    drift = max(max(abs(a[label]["angle_median"] - want[0]),
                    abs(a[label]["offset_median"] - want[1]))
                for label, want in REPRODUCTION.items())
    report["condition_A_drift"] = drift
    if drift > REPRODUCTION_TOLERANCE:
        report["CONDITION_A_NOT_REPRODUCED"] = True
        (OUT / "data_vs_step_result.json").write_text(json.dumps(report, indent=2,
                                                                 default=float))
        raise RuntimeError(f"CONDITION_A_NOT_REPRODUCED: drift {drift:.3e}")
    report["verdict"] = decide(report)
    (OUT / "data_vs_step_result.json").write_text(json.dumps(report, indent=2,
                                                             default=float))
    log(f"[run] condition A drift {drift:.2e}  {report['verdict']['CAUSE']}")


if __name__ == "__main__":
    main()
