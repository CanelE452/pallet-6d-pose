"""Does line supervision have to reach the A1 feature extractor?

Phase A closed the step axis: `LONG_SCHEDULE_STILL_OPTIMIZING_BUT_TASK_FAIL` at
25,545 steps, 0 of 4 gates, with the cross-entropy still falling.  That says
exposure was not the primary limit and deliberately does not say the frozen
feature is.  This asks the feature question once, with one factor moving.

```
F0_FROZEN_A1          DIRECT_HOUGH_TOKEN_XY_V0 exactly as Phase A ran it
F1_LATE_A1_TRAINABLE  the same, with net.vgg[19:27] receiving gradient
                      5,014,912 params = 68.3% of net.vgg, 9.15% of A1
```

Everything else is held: role-query encoder, DirectHoughHead, lattice, target,
cross-entropy, token XY, query count, attention depth, batch, weight decay,
seed, pool, marks, gates.  The dead `self.position` stays dead.

`net.vgg[19:27]` is the block boundary because `net.vgg[18]` is the last
MaxPool -- 19 through 26 is the only stage at F50 resolution and splitting it
would be an invented boundary.  A1 carries no normalisation layer and no dropout
in the trunk, both measured, so there is no running-statistic policy to set and
A1 stays in `.eval()` in both arms.

`AdaptableA1` calls `net.vgg` directly instead of hooking a full `net()` forward.
That is the same tensor by measurement (max abs diff 0.000e+00) and avoids
building a graph through 47M parameters no loss touches.

F0 is reused from Phase A only if the Phase B trainer with unfreeze disabled is
bit-identical to the locked trainer under deterministic kernels.  Otherwise F0 is
re-run.  Scope, learning rate and verdict labels are fixed in
`LATE_A1_ADAPTATION_SCOPE.md` before this file runs.
"""
from __future__ import annotations

import argparse, importlib.util, json, os, pathlib, sys, time
import numpy as np, torch, torch.nn as nn

ROOT = pathlib.Path(__file__).resolve().parents[2]
for _e in ("scripts/stage0", "Deep_Object_Pose/common", "challenge/scripts"):
    if str(ROOT / _e) not in sys.path:
        sys.path.insert(0, str(ROOT / _e))


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


LONG = _load("DH_LONG_B", "scripts/stage0/line/direct_hough_full_step_extension.py")
DH, CAP, V2, SCALE = LONG.DH, LONG.CAP, LONG.V2, LONG.SCALE
OUT, DEV = LONG.OUT, LONG.DEV

FIRST_TRAINABLE_INDEX = 19          # net.vgg[18] is the last MaxPool
MARKS = LONG.LONG_MARKS
DECISION_STEP = LONG.DECISION_STEP
PER_ROLE_MARKS = LONG.PER_ROLE_MARKS
DIAGNOSTIC_MARKS = LONG.DIAGNOSTIC_MARKS
A1_LR_SCALE = 0.1                   # A1 LR = head LR x 0.1, one value, no sweep
PHASE_A_RESULT = "direct_hough_long.json"
SIMILAR_TO_F0 = 0.05                # both D2 medians within 5% relative
PARITY_STEPS = LONG.PARITY_STEPS
PARITY_FRAMES = LONG.PARITY_FRAMES
DETERMINISTIC_WORKSPACE = LONG.DETERMINISTIC_WORKSPACE
ARMS = ("F0_FROZEN_A1", "F1_LATE_A1_TRAINABLE")


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


class AdaptableA1(nn.Module):
    """A1 with the F50 tap taken directly and one block optionally trainable."""

    def __init__(self, trainable_from=None):
        super().__init__()
        self.inner = V2.load_a1()
        self.vgg = self.inner.model.net.vgg
        self.inner.model.eval()               # no norm layers, no dropout: inert
        for parameter in self.inner.model.parameters():
            parameter.requires_grad_(False)
        self.trainable_from = trainable_from
        if trainable_from is not None:
            for index, child in self.vgg.named_children():
                if int(index) >= trainable_from:
                    for parameter in child.parameters():
                        parameter.requires_grad_(True)

    def parameters_to_train(self):
        return [p for p in self.vgg.parameters() if p.requires_grad]

    def report(self):
        total = sum(p.numel() for p in self.inner.model.parameters())
        vgg = sum(p.numel() for p in self.vgg.parameters())
        trainable = sum(p.numel() for p in self.parameters_to_train())
        return {"a1_total_params": total, "vgg_params": vgg,
                "trainable_params": trainable,
                "trainable_fraction_of_vgg": trainable / vgg,
                "trainable_fraction_of_a1": trainable / total,
                "trainable_from_index": self.trainable_from,
                "trainable_tensors": len(self.parameters_to_train()),
                "eval_mode": not self.inner.model.training,
                "normalisation_layers": sum(
                    1 for m in self.inner.model.modules()
                    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm,
                                      nn.LayerNorm, nn.InstanceNorm2d)))}

    def forward(self, images):
        if self.trainable_from is None:
            with torch.no_grad():
                return self.vgg(images).detach(), None, None
        return self.vgg(images), None, None


def encoder_features(pack, a1):
    """DH.encoder_features without its no_grad, so F1's gradient survives.

    DH's version wraps the call in `torch.no_grad` and detaches; F0 reproduces
    that exactly because `AdaptableA1` detaches internally when nothing is
    trainable, and F1 needs the graph.  The upsampled branch DH returns is unused
    by the direct-Hough path and is not computed here.
    """
    f50, _, _ = a1(pack["images"])
    return f50, None


def train_arm(arm, pool, marks, edges, populations, per_pass, tag):
    """LONG.train_long with the A1 parameter group added for F1."""
    grid_theta, grid_rho, valid = DH.lattice()
    features = DH.hypothesis_features(grid_theta, grid_rho)
    a1 = AdaptableA1(FIRST_TRAINABLE_INDEX if arm == ARMS[1] else None).to(DEV)
    model = DH.DirectHoughModel().to(DEV)
    groups = [{"params": list(model.parameters()), "lr": CAP.LR}]
    if a1.parameters_to_train():
        groups.append({"params": a1.parameters_to_train(),
                       "lr": CAP.LR * A1_LR_SCALE})
    optimiser = torch.optim.AdamW(groups, lr=CAP.LR, weight_decay=CAP.WD)
    history, losses, done = {}, [], 0
    for chunk, _ in V2.step_schedule(pool, max(marks), CAP.BATCH):
        model.train()
        pack = V2.load_pack(chunk)
        theta_c, rho_c, support = DH.batch_rows(pack, edges)
        target = DH.target_distribution(
            theta_c.reshape(-1), rho_c.reshape(-1), grid_theta, grid_rho, valid
        ).reshape(*theta_c.shape, -1)
        f50, _ = encoder_features(pack, a1)
        loss = DH.cross_entropy(model(f50, features), target, support, valid)
        optimiser.zero_grad(set_to_none=True)
        loss.backward(); optimiser.step()
        losses.append(float(loss.detach()))
        done += 1
        if done in marks:
            entry = {"step": done,
                     "train_loss_mean_last250": float(np.mean(losses[-250:])),
                     "train_loss_slope_last250": LONG.slope(losses[-250:]),
                     "train_loss_mean_last_pass": float(np.mean(losses[-per_pass:])),
                     "train_loss_slope_last_pass": LONG.slope(losses[-per_pass:]),
                     "diagnostic_only": done in DIAGNOSTIC_MARKS}
            for label, indices in populations.items():
                entry[label] = evaluate(indices, model, a1, edges, features,
                                        grid_theta, grid_rho, valid,
                                        per_role=(label == "D2_LINE_DEV512"
                                                  and done in PER_ROLE_MARKS))
                log(f"  {tag} @{done:6d} {label:<16} angle med "
                    f"{entry[label]['angle_median']:7.4f} p90 "
                    f"{entry[label]['angle_p90']:7.3f} | offset med "
                    f"{entry[label]['offset_median']:7.4f} p90 "
                    f"{entry[label]['offset_p90']:7.3f}")
            log(f"  {tag} @{done:6d} CE last250 {entry['train_loss_mean_last250']:.6f}"
                f"  slope/step {entry['train_loss_slope_last_pass']:+.3e}")
            torch.save({"tag": tag, "step": done, "model": model.state_dict(),
                        "a1_trainable": a1.report(), **CAP.provenance()},
                       CAP.checkpoint_path(f"DH_{tag}", f"step_{done:05d}"))
            history[str(done)] = entry
    return history, model, a1


@torch.no_grad()
def evaluate(indices, model, a1, edges, features, grid_theta, grid_rho, valid,
             per_role=False):
    """DH.evaluate_network with this screen's feature path.

    Identical to DH's otherwise -- same batching, same support mask, same decode,
    same summary -- so the two arms and Phase A are scored the same way.
    """
    model.eval()
    angle, offset, roles = [], [], []
    for start in range(0, len(indices), CAP.BATCH):
        chunk = indices[start:start + CAP.BATCH]
        if len(chunk) < 2:
            continue
        pack = V2.load_pack(chunk)
        theta_c, rho_c, support = DH.batch_rows(pack, edges)
        f50, _ = encoder_features(pack, a1)
        scores = model(f50, features)
        for frame in range(scores.shape[0]):
            live = torch.nonzero(support[frame]).flatten()
            if live.numel() == 0:
                continue
            theta_p, rho_p = DH.decode(scores[frame][live], grid_theta, grid_rho,
                                       valid)
            a, o = DH.measure(theta_p, rho_p, theta_c[frame][live],
                              rho_c[frame][live])
            angle.append(a); offset.append(o); roles.append(live.cpu().numpy())
    angle = np.concatenate(angle) if angle else np.zeros(1)
    offset = np.concatenate(offset) if offset else np.zeros(1)
    report = DH.summarise(angle, offset)
    if per_role:
        index = np.concatenate(roles)
        report["per_role"] = {}
        for r in range(DH.ROLES):
            keep = index == r
            if keep.sum():
                report["per_role"][str(r)] = {
                    "n": int(keep.sum()),
                    "angle_median": float(np.median(angle[keep])),
                    "angle_p90": float(np.percentile(angle[keep], 90)),
                    "offset_median": float(np.median(offset[keep])),
                    "offset_p90": float(np.percentile(offset[keep], 90))}
    return report


def run_parity(edges):
    """Is this trainer, with unfreeze off, the Phase A trainer?

    If yes the recorded 25,545-step run is the F0 arm and does not need two hours
    of GPU to restate.  Asked under deterministic kernels because the default
    path is not bit-reproducible -- the locked runner differs from itself by
    1.4e-03 after twenty steps, measured in `direct_hough_long_parity.json`.
    """
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != DETERMINISTIC_WORKSPACE:
        raise RuntimeError("parity needs CUBLAS_WORKSPACE_CONFIG="
                           f"{DETERMINISTIC_WORKSPACE} in the environment")
    pool = V2.split_indices()[0][:PARITY_FRAMES]
    per_pass = V2.steps_per_pass(pool, CAP.BATCH)
    populations = {"PARITY": pool[:CAP.BATCH]}
    torch.use_deterministic_algorithms(True)
    try:
        locked_a1 = V2.load_a1()
        locked = DH.train_network(pool, (PARITY_STEPS,), edges, locked_a1,
                                  populations, "phaseb_locked")[1]
        control = DH.train_network(pool, (PARITY_STEPS,), edges, locked_a1,
                                   populations, "phaseb_control")[1]
        candidate = train_arm(ARMS[0], pool, (PARITY_STEPS,), edges, populations,
                              per_pass, "phaseb_f0_probe")[1]
    finally:
        torch.use_deterministic_algorithms(False)
    report = {"steps": PARITY_STEPS, "frames": PARITY_FRAMES,
              "deterministic_control": LONG.parameter_distance(locked, control),
              "f0_against_locked": LONG.parameter_distance(locked, candidate)}
    report["DETERMINISTIC_MODE_VERIFIED"] = bool(
        report["deterministic_control"]["max_abs_delta"] == 0.0)
    report["F0_CODE_PATH_PARITY"] = bool(
        report["DETERMINISTIC_MODE_VERIFIED"]
        and report["f0_against_locked"]["max_abs_delta"] == 0.0)
    report["F0_SOURCE"] = ("phase_a_reuse" if report["F0_CODE_PATH_PARITY"]
                           else "fresh_rerun")
    return report


def phase_a_history():
    return json.loads((OUT / PHASE_A_RESULT).read_text())["history"]


def judge(f0, f1, limits):
    """Primary is D2 at the decision step, and nothing else decides."""
    a = f0[str(DECISION_STEP)]["D2_LINE_DEV512"]
    b = f1[str(DECISION_STEP)]["D2_LINE_DEV512"]
    out = {"decision_step": DECISION_STEP, "population": "D2_LINE_DEV512",
           "F0": {k: a[k] for k in ("angle_median", "offset_median",
                                    "angle_p90", "offset_p90", "PASS", "SAFETY")},
           "F1": {k: b[k] for k in ("angle_median", "offset_median",
                                    "angle_p90", "offset_p90", "PASS", "SAFETY")},
           "ABSOLUTE_PASS": bool(b["PASS"] and b["SAFETY"]),
           "REDUCTION_40": bool(
               b["angle_median"] <= limits["reduction_40"]["angle_median"]
               and b["offset_median"] <= limits["reduction_40"]["offset_median"])}
    out["vs_F0"] = {k: 1.0 - b[k] / a[k] for k in
                    ("angle_median", "offset_median", "angle_p90", "offset_p90")}
    out["vs_baseline"] = {
        "angle_median": 1.0 - b["angle_median"]
        / limits["baseline_full_precision"]["angle_median"],
        "offset_median": 1.0 - b["offset_median"]
        / limits["baseline_full_precision"]["offset_median"]}
    ce = {"F0": f0[str(DECISION_STEP)]["train_loss_mean_last250"],
          "F1": f1[str(DECISION_STEP)]["train_loss_mean_last250"]}
    out["train_ce"] = {**ce, "F1_below_F0": bool(ce["F1"] < ce["F0"])}
    out["conditions"] = {
        "SIMILAR_TO_F0": bool(
            abs(out["vs_F0"]["angle_median"]) < SIMILAR_TO_F0
            and abs(out["vs_F0"]["offset_median"]) < SIMILAR_TO_F0),
        "OVERFITS": bool(ce["F1"] < ce["F0"]
                         and (b["angle_median"] > a["angle_median"]
                              or b["offset_median"] > a["offset_median"]))}
    if out["ABSOLUTE_PASS"]:
        out["DECISION"] = "LATE_A1_FEATURE_ADAPTATION_RESCUES_DIRECT_HOUGH"
        out["STATUS"] = "LINE_STAGE_CANDIDATE"
        out["REMAINING_FOR_LOCK"] = ["same_protocol_replicate", "role_shuffle",
                                     "whole_LINE_DEV"]
    elif out["REDUCTION_40"]:
        out["DECISION"] = "LATE_A1_FEATURE_ADAPTATION_SIGNAL"
        out["NEXT"] = "ROLE_ENCODER_CAPACITY_SCREEN"
    elif out["conditions"]["OVERFITS"]:
        out["DECISION"] = "LATE_A1_ADAPTATION_OVERFITS"
        out["WIDEN_UNFREEZE"] = "FORBIDDEN"
    elif out["conditions"]["SIMILAR_TO_F0"]:
        out["DECISION"] = "FROZEN_A1_NOT_PRIMARY_LIMIT"
        out["NEXT"] = "ROLE_ENCODER_CAPACITY_SCREEN"
    else:
        out["DECISION"] = "LATE_A1_ADAPTATION_INCONCLUSIVE"
    out.setdefault("STATUS", "NOT_LOCKED")
    out["CIGM"] = "BLOCKED"
    out["LIMITATION"] = ("one A1 learning rate, CAP.LR x "
                         f"{A1_LR_SCALE}, pre-registered and not swept")
    return out


def build_plan(pool):
    probe = AdaptableA1(FIRST_TRAINABLE_INDEX)
    plan = {"arms": list(ARMS), "marks": list(MARKS),
            "decision_step": DECISION_STEP,
            "decision_population": "D2_LINE_DEV512",
            "diagnostic_population": "D0_SEEN512",
            "per_role_marks": list(PER_ROLE_MARKS),
            "frames": len(pool), "block": "net.vgg[19:27]",
            "first_trainable_index": FIRST_TRAINABLE_INDEX,
            "a1": probe.report(),
            "head_lr": CAP.LR, "a1_lr": CAP.LR * A1_LR_SCALE,
            "a1_lr_scale": A1_LR_SCALE, "lr_sweep": False,
            "similar_to_f0": SIMILAR_TO_F0,
            "gate": {"angle_median": CAP.ANGLE_BUDGET_DEG,
                     "offset_median": CAP.OFFSET_BUDGET_CELL,
                     "angle_p90": CAP.SAFETY_ANGLE,
                     "offset_p90": CAP.SAFETY_OFFSET},
            "thresholds": DH.thresholds(),
            "phase_a_decision": json.loads(
                (OUT / PHASE_A_RESULT).read_text())["verdict"]["DECISION"],
            **CAP.provenance()}
    del probe
    torch.cuda.empty_cache()
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "parity", "run"])
    arguments = parser.parse_args()
    import instance_edge_topology as IET
    edges = [tuple(e) for e in IET.build_topology()["edges"]]
    if not V2.sha_file(OUT / "line_internal_split.csv").startswith(V2.LINE_SPLIT_SHA):
        raise RuntimeError("HARD_BLOCKED: LINE split changed")
    phase_a = OUT / PHASE_A_RESULT
    if not phase_a.exists():
        raise RuntimeError("HARD_BLOCK: Phase A result is missing")
    if json.loads(phase_a.read_text())["verdict"]["DECISION"] == \
            "DIRECT_HOUGH_LONG_SCHEDULE_VALID_CANDIDATE":
        raise RuntimeError("PHASE_B_BLOCKED: Phase A was A1")
    pool = V2.split_indices()[0]

    if arguments.command == "plan":
        plan = build_plan(pool)
        (OUT / "late_a1_plan.json").write_text(json.dumps(plan, indent=2))
        a1 = plan["a1"]
        log(f"[plan] block {plan['block']}  trainable "
            f"{a1['trainable_params']:,} params over {a1['trainable_tensors']} "
            f"tensors  = {a1['trainable_fraction_of_vgg']:.1%} of net.vgg, "
            f"{a1['trainable_fraction_of_a1']:.2%} of A1")
        log(f"[plan] normalisation layers {a1['normalisation_layers']}  "
            f"A1 eval mode {a1['eval_mode']}  head LR {plan['head_lr']}  "
            f"A1 LR {plan['a1_lr']}  sweep {plan['lr_sweep']}")
        log(f"[plan] decision {DECISION_STEP} on D2 only  gate "
            f"{CAP.ANGLE_BUDGET_DEG}/{CAP.OFFSET_BUDGET_CELL} safety "
            f"{CAP.SAFETY_ANGLE}/{CAP.SAFETY_OFFSET}")
        return

    if arguments.command == "parity":
        report = run_parity(edges)
        (OUT / "late_a1_parity.json").write_text(
            json.dumps(report, indent=2, default=float))
        log(f"[parity] deterministic control "
            f"{report['deterministic_control']['max_abs_delta']:.3e}  "
            f"F0 vs locked {report['f0_against_locked']['max_abs_delta']:.3e}  "
            f"F0_SOURCE={report['F0_SOURCE']}")
        if not report["DETERMINISTIC_MODE_VERIFIED"]:
            raise RuntimeError("DETERMINISTIC_MODE_UNVERIFIED")
        return

    parity_path = OUT / "late_a1_parity.json"
    if not parity_path.exists():
        raise RuntimeError("run parity first")
    parity = json.loads(parity_path.read_text())
    plan = build_plan(pool)
    per_pass = V2.steps_per_pass(pool, CAP.BATCH)
    populations = SCALE.populations()
    histories = {}
    if parity["F0_CODE_PATH_PARITY"]:
        histories[ARMS[0]] = phase_a_history()
        log(f"[run] {ARMS[0]} reused from Phase A (code path proven identical)")
    else:
        log(f"[run] {ARMS[0]} re-run fresh (code path parity not proven)")
        histories[ARMS[0]] = train_arm(ARMS[0], pool, MARKS, edges, populations,
                                       per_pass, "f0")[0]
    histories[ARMS[1]], _, trained = train_arm(ARMS[1], pool, MARKS, edges,
                                               populations, per_pass, "f1")
    limits = plan["thresholds"]
    report = {"plan": plan, "parity": parity, "histories": histories,
              "f0_source": parity["F0_SOURCE"], "thresholds": limits,
              "a1_trained": trained.report(),
              "verdict": judge(histories[ARMS[0]], histories[ARMS[1]], limits),
              **CAP.provenance()}
    (OUT / "late_a1_adaptation.json").write_text(
        json.dumps(report, indent=2, default=float))
    v = report["verdict"]
    log(f"[run] {v['DECISION']}  F1 {v['F1']['angle_median']:.6f}/"
        f"{v['F1']['offset_median']:.6f}  F0 {v['F0']['angle_median']:.6f}/"
        f"{v['F0']['offset_median']:.6f}")
    log(f"[run] vs F0 angle {v['vs_F0']['angle_median']:+.2%} offset "
        f"{v['vs_F0']['offset_median']:+.2%}  conditions {v['conditions']}")


if __name__ == "__main__":
    main()
