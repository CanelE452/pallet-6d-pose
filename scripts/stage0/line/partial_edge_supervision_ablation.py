"""Does supervising partially in-frame edges help predict them?

The zero-training diagnostic found signal on `T1_PARTIAL`
(`TRUNCATED_CORNER_VISIBLE_EDGE_SIGNAL_PRESENT`), which says the current
predictor is not blind there.  It does not say that training on those roles is
what produced it, so one factor moves and nothing else.

```
B1_FULL_PLUS_PARTIAL  support = seg["hit"]            the historical P0 semantics
B0_FULL_ONLY          support = seg["in_frame_full"]  T1 contributes exactly zero
T2_OFF_FRAME          unsupported in both arms
```

Everything else is P0's: two photometric views per sample, `L_sup = 0.5 CE(a) +
0.5 CE(b)`, no consistency term, the same architecture, late-A1 policy, lattice,
target, optimizer, learning rates, batch, seed, pool, schedule and marks.

B0 has fewer supervised roles per batch and its loss is therefore a mean over a
smaller set.  That is what masking does; rescaling to equalise it would be a
second factor, so the exposure difference is measured and reported instead.

B1 is historical P0 if and only if it is proven to be, by
`HISTORICAL_P0_REUSE_QUALIFIED`: identical initialisation, identical batches and
views, and a bit-exact twenty-step trajectory against the screen that produced
it, all under deterministic kernels with a control run to show the mode is
honest.  Anything less and both arms are trained fresh -- "the same recipe" is
not an argument, it is a measurement.

Evaluation is unchanged in both arms: `LATE.evaluate` supports every in-frame
role, so T1 is scored whether or not it was trained on.
"""
from __future__ import annotations
import os as _os, sys as _sys

# --- stage0 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_S0 = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_S0] + [_os.path.join(_S0, _d) for _d in sorted(_os.listdir(_S0))
                         if _os.path.isdir(_os.path.join(_S0, _d)) and not _d.startswith(".")]


import argparse, hashlib, importlib.util, json, os, pathlib, sys, time
import numpy as np, torch, torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[3]
for _e in ("scripts/stage0", "Deep_Object_Pose/common", "challenge/scripts"):
    if str(ROOT / _e) not in sys.path:
        sys.path.insert(0, str(ROOT / _e))


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


TRUNC = _load("TRUNCATION_BASE",
              "scripts/stage0/line/partial_edge_truncation_screen.py")
AC, LATE, DH = TRUNC.AC, TRUNC.LATE, TRUNC.DH
CAP, V2, SCALE = TRUNC.CAP, TRUNC.V2, TRUNC.SCALE
OUT, DEV = TRUNC.OUT, TRUNC.DEV

ARMS = ("B0_FULL_ONLY", "B1_FULL_PLUS_PARTIAL")
SUPPORT_MODE = {"B0_FULL_ONLY": "in_frame_full", "B1_FULL_PLUS_PARTIAL": "hit"}
MARKS = AC.MARKS
DECISION_STEP = AC.DECISION_STEP
PER_ROLE_MARKS = AC.PER_ROLE_MARKS
PARITY_STEPS = 20
PARITY_FRAMES = 64
DETERMINISTIC_WORKSPACE = AC.DETERMINISTIC_WORKSPACE
P0_ARM = "P0_AUG_ONLY"
P0_RESULT = "appearance_result.json"
# Fixed before any ablation metric is read.
SPAN_BINS = (0.0, 0.25, 0.50, 0.75, 1.0)
A_MEDIAN_REDUCTION = 0.15
B_GATE_POINTS = 0.10
C_SAFETY_DEGRADATION = 0.10
NOT_EVALUATED = dict(TRUNC.NOT_EVALUATED)


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def support_for(pack, edges, mode):
    """The training support mask.  `hit` is P0's; `in_frame_full` drops T1."""
    rows = TRUNC.categorise(pack, edges)
    key = {"hit": "support", "in_frame_full": "T0_FULL"}[mode]
    return torch.tensor(rows[key], device=DEV), rows


def per_role_cross_entropy(scores, target, valid):
    """DH.cross_entropy's numerator, before any support weighting."""
    logp = F.log_softmax(scores.masked_fill(~valid[None, None], -1e9), -1)
    return -(target * logp).sum(-1)


def supervised_loss(scores_a, scores_b, target, support, valid):
    """P0's objective with the arm's support mask, normalised by its own count.

    A smaller mask means a mean over fewer roles.  Rescaling that back would
    equalise exposure and become a second factor, so it is left alone and the
    counts are reported.
    """
    weight = support.float()
    denominator = weight.sum().clamp_min(1.0)
    terms = []
    for scores in (scores_a, scores_b):
        per_role = per_role_cross_entropy(scores, target, valid)
        terms.append((per_role * weight).sum() / denominator)
    return 0.5 * terms[0] + 0.5 * terms[1]


def train_arm(arm, pool, marks, edges, populations, per_pass, probe=None):
    """P0's loop with the support mask as the only arm-dependent quantity."""
    grid_theta, grid_rho, valid = DH.lattice()
    features = DH.hypothesis_features(grid_theta, grid_rho)
    a1, model = AC.build()
    optimiser = AC.optimiser_for(model, a1)
    mode = SUPPORT_MODE[arm]
    probe = probe if probe is not None else V2.split_indices()[0][:AC.PROBE_FRAMES]
    history, sup_log, done = {}, [], 0
    exposure = {"base_frames": 0, "view_instances": 0,
                "supervised_roles": 0, "t0_roles": 0, "t1_roles": 0}
    for chunk, _ in V2.step_schedule(pool, max(marks), CAP.BATCH):
        model.train()
        pack = V2.load_pack(chunk)
        views = AC.two_views(pack, done)
        theta_c, rho_c, _ = DH.batch_rows(pack, edges)
        target = DH.target_distribution(
            theta_c.reshape(-1), rho_c.reshape(-1), grid_theta, grid_rho, valid
        ).reshape(*theta_c.shape, -1)
        support, rows = support_for(pack, edges, mode)
        scores = []
        for images in views:
            f50, _ = LATE.encoder_features({"images": images}, a1)
            scores.append(model(f50, features))
        loss = supervised_loss(scores[0], scores[1], target, support, valid)
        optimiser.zero_grad(set_to_none=True)
        loss.backward(); optimiser.step()
        sup_log.append(float(loss.detach()))
        exposure["base_frames"] += len(chunk)
        exposure["view_instances"] += 2 * len(chunk)
        exposure["supervised_roles"] += int(support.sum())
        exposure["t0_roles"] += int(rows["T0_FULL"].sum())
        exposure["t1_roles"] += int((rows["T1_PARTIAL"]).sum())
        done += 1
        if done in marks:
            model.eval()
            entry = {"step": done, "arm": arm, "support_mode": mode,
                     "finite": bool(np.isfinite(sup_log[-1])),
                     "sup_mean_last250": float(np.mean(sup_log[-250:])),
                     "exposure": dict(exposure)}
            for label, indices in populations.items():
                entry[label] = LATE.evaluate(
                    indices, model, a1, edges, features, grid_theta, grid_rho,
                    valid, per_role=(label == "D2_LINE_DEV512"
                                     and done in PER_ROLE_MARKS))
            log(f"  {arm[:2].lower()} @{done:6d} D2 angle "
                f"{entry['D2_LINE_DEV512']['angle_median']:7.4f} p90 "
                f"{entry['D2_LINE_DEV512']['angle_p90']:7.3f} | offset "
                f"{entry['D2_LINE_DEV512']['offset_median']:7.4f} | L_sup "
                f"{entry['sup_mean_last250']:.6f} | roles sup "
                f"{exposure['supervised_roles']}")
            torch.save({"tag": arm, "step": done, "model": model.state_dict(),
                        "late_a1": {name: parameter.detach().cpu() for name,
                                    parameter in AC.late_parameters(a1)},
                        "support_mode": mode, **CAP.provenance()},
                       CAP.checkpoint_path(f"DH_{arm}", f"step_{done:05d}"))
            history[str(done)] = entry
    del a1, model
    torch.cuda.empty_cache()
    return history


# ------------------------------------------------------------ qualification
def parameter_gap(left, right):
    a, b = left.state_dict(), right.state_dict()
    if sorted(a) != sorted(b):
        return float("inf")
    return max(float((a[k].float() - b[k].float()).abs().max()) for k in a)


def run_qualification(edges):
    """Is the historical P0 arm reproducible by this runner, exactly?"""
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != DETERMINISTIC_WORKSPACE:
        raise RuntimeError("qualification needs CUBLAS_WORKSPACE_CONFIG="
                           f"{DETERMINISTIC_WORKSPACE}")
    pool = V2.split_indices()[0][:PARITY_FRAMES]
    per_pass = V2.steps_per_pass(pool, CAP.BATCH)
    # AC.train_arm derives its generalization entry from these two names, so the
    # real populations are used rather than a stand-in with a misleading label.
    populations = SCALE.populations()
    grid_theta, grid_rho, valid = DH.lattice()
    features = DH.hypothesis_features(grid_theta, grid_rho)

    # A. initialisation
    a1_left, model_left = AC.build()
    a1_right, model_right = AC.build()
    init = {"decoder_max_abs": parameter_gap(model_left, model_right),
            "late_a1_max_abs": max(
                float((x[1] - y[1]).abs().max()) for x, y in
                zip(AC.late_parameters(a1_left), AC.late_parameters(a1_right)))}
    groups_left = AC.optimiser_for(model_left, a1_left).param_groups
    groups_right = AC.optimiser_for(model_right, a1_right).param_groups
    init["param_groups_match"] = bool(
        [(g["lr"], g["weight_decay"], len(g["params"])) for g in groups_left]
        == [(g["lr"], g["weight_decay"], len(g["params"])) for g in groups_right])

    # B. batches and views
    batch = {"frame_ids_equal": True, "view_a_max_abs": 0.0,
             "view_b_max_abs": 0.0, "geometry_max_abs": 0.0,
             "support_equal": True, "category_equal": True,
             "logit_max_abs": 0.0, "ce_max_abs": 0.0, "sup_max_abs": 0.0}
    probes = [0, 1, 7, 19]
    for step, (chunk, _) in enumerate(V2.step_schedule(pool, 20, CAP.BATCH)):
        if step not in probes:
            continue
        pack = V2.load_pack(chunk)
        left_views = AC.two_views(pack, step)
        right_views = AC.two_views(pack, step)
        batch["view_a_max_abs"] = max(batch["view_a_max_abs"],
                                      float((left_views[0] - right_views[0]).abs().max()))
        batch["view_b_max_abs"] = max(batch["view_b_max_abs"],
                                      float((left_views[1] - right_views[1]).abs().max()))
        theta_l, rho_l, support_l = DH.batch_rows(pack, edges)
        theta_r, rho_r, support_r = DH.batch_rows(pack, edges)
        batch["geometry_max_abs"] = max(
            batch["geometry_max_abs"],
            float((theta_l - theta_r).abs().max()),
            float((rho_l - rho_r).abs().max()))
        batch["support_equal"] &= bool(torch.equal(support_l, support_r))
        hit, rows = support_for(pack, edges, "hit")
        full, _ = support_for(pack, edges, "in_frame_full")
        batch["category_equal"] &= bool(torch.equal(hit, support_l))
        target = DH.target_distribution(
            theta_l.reshape(-1), rho_l.reshape(-1), grid_theta, grid_rho, valid
        ).reshape(*theta_l.shape, -1)
        with torch.no_grad():
            scores = []
            for images in left_views:
                f50, _ = LATE.encoder_features({"images": images}, a1_left)
                scores.append(model_left(f50, features))
            mine = supervised_loss(scores[0], scores[1], target, hit, valid)
            theirs = 0.5 * DH.cross_entropy(scores[0], target, support_l, valid) \
                + 0.5 * DH.cross_entropy(scores[1], target, support_l, valid)
            batch["sup_max_abs"] = max(batch["sup_max_abs"],
                                       abs(float(mine) - float(theirs)))
    del a1_left, a1_right, model_left, model_right
    torch.cuda.empty_cache()

    # C. twenty deterministic steps, with a control
    torch.use_deterministic_algorithms(True)
    try:
        control_a = AC.train_arm(P0_ARM, pool, (PARITY_STEPS,), edges,
                                 populations, per_pass, 0.0)
        control_b = AC.train_arm(P0_ARM, pool, (PARITY_STEPS,), edges,
                                 populations, per_pass, 0.0)
        mine = train_arm("B1_FULL_PLUS_PARTIAL", pool, (PARITY_STEPS,), edges,
                         populations, per_pass, probe=pool[:CAP.BATCH])
    finally:
        torch.use_deterministic_algorithms(False)

    def final(history):
        entry = history[str(PARITY_STEPS)]["D2_LINE_DEV512"]
        return {k: entry[k] for k in ("angle_median", "offset_median",
                                      "angle_p90", "offset_p90")}

    control_gap = {k: abs(final(control_a)[k] - final(control_b)[k])
                   for k in final(control_a)}
    cross_gap = {k: abs(final(control_a)[k] - final(mine)[k])
                 for k in final(control_a)}
    report = {"init": init, "batch": batch,
              "control": final(control_a), "control_repeat": final(control_b),
              "candidate": final(mine),
              "control_gap": control_gap, "cross_gap": cross_gap,
              "steps": PARITY_STEPS, "frames": PARITY_FRAMES,
              "deterministic": True}
    report["DETERMINISTIC_MODE_VERIFIED"] = bool(
        max(control_gap.values()) == 0.0)
    report["HISTORICAL_P0_REUSE_QUALIFIED"] = bool(
        report["DETERMINISTIC_MODE_VERIFIED"]
        and max(cross_gap.values()) == 0.0
        and init["decoder_max_abs"] == 0.0 and init["late_a1_max_abs"] == 0.0
        and init["param_groups_match"]
        and batch["view_a_max_abs"] == 0.0 and batch["view_b_max_abs"] == 0.0
        and batch["geometry_max_abs"] == 0.0 and batch["support_equal"]
        and batch["category_equal"] and batch["sup_max_abs"] == 0.0)
    return report


def run_mask_factor(edges):
    """Is the mask the only thing that differs, and does it isolate T1?"""
    grid_theta, grid_rho, valid = DH.lattice()
    features = DH.hypothesis_features(grid_theta, grid_rho)
    a1, model = AC.build()
    pool = V2.split_indices()[0][:CAP.BATCH]
    pack = V2.load_pack(pool)
    views = AC.two_views(pack, 0)
    theta_c, rho_c, _ = DH.batch_rows(pack, edges)
    target = DH.target_distribution(
        theta_c.reshape(-1), rho_c.reshape(-1), grid_theta, grid_rho, valid
    ).reshape(*theta_c.shape, -1)
    hit, rows = support_for(pack, edges, "hit")
    full, _ = support_for(pack, edges, "in_frame_full")
    partial = torch.tensor(rows["T1_PARTIAL"], device=DEV)
    off = torch.tensor(rows["T2_OFF_FRAME"], device=DEV)
    scores = []
    for images in views:
        f50, _ = LATE.encoder_features({"images": images}, a1)
        scores.append(model(f50, features))
    per_role = 0.5 * (per_role_cross_entropy(scores[0], target, valid)
                      + per_role_cross_entropy(scores[1], target, valid))

    def grad_norm(mask):
        for parameter in list(model.parameters()) + a1.parameters_to_train():
            parameter.grad = None
        weight = mask.float()
        if float(weight.sum()) == 0.0:
            return 0.0
        ((per_role * weight).sum() / weight.sum()).backward(retain_graph=True)
        pieces = [p.grad.detach().reshape(-1) for p in a1.parameters_to_train()
                  if p.grad is not None]
        return float(torch.cat(pieces).norm()) if pieces else 0.0

    report = {
        "counts": {"T0": int(full.sum()), "T1": int(partial.sum()),
                   "T2": int(off.sum()), "hit": int(hit.sum())},
        "t0_loss_equal": float(
            abs(float((per_role * full.float()).sum())
                - float((per_role * (hit & ~partial).float()).sum()))),
        "t1_contribution_B0": float((per_role * (full & partial).float()).sum()),
        "t1_contribution_B1": float((per_role * (hit & partial).float()).sum()),
        "t2_contribution_B0": float((per_role * (full & off).float()).sum()),
        "t2_contribution_B1": float((per_role * (hit & off).float()).sum()),
        "grad_T1_B0": grad_norm(full & partial),
        "grad_T1_B1": grad_norm(hit & partial),
        "grad_T0_B0": grad_norm(full),
        "grad_T0_B1": grad_norm(hit & ~partial)}
    report["grad_T0_relative_gap"] = abs(
        report["grad_T0_B0"] - report["grad_T0_B1"]) / max(
        report["grad_T0_B0"], 1e-12)
    report["PARTIAL_EDGE_MASK_FACTOR_ISOLATED"] = bool(
        report["counts"]["T1"] > 0
        and report["t0_loss_equal"] <= 1e-4
        and report["t1_contribution_B0"] == 0.0
        and report["t1_contribution_B1"] > 0.0
        and report["t2_contribution_B0"] == 0.0
        and report["t2_contribution_B1"] == 0.0
        and report["grad_T1_B0"] == 0.0 and report["grad_T1_B1"] > 0.0
        and report["grad_T0_relative_gap"] <= 1e-6)
    del a1, model
    torch.cuda.empty_cache()
    return report



@torch.no_grad()
def stratified(indices, decoder, backbone, edges):
    """One decode pass, sliced by the locked categories and by chord fraction.

    The span bins exist because T1's chord-to-span ratio runs from 0.0101 to
    0.9954, so a single "partial" number mixes an edge with one percent left in
    frame against one with almost all of it.  They are diagnostic and were fixed
    before any ablation metric was read.
    """
    grid_theta, grid_rho, valid = DH.lattice()
    features = DH.hypothesis_features(grid_theta, grid_rho)
    decoder.eval()
    angle, offset, roles, tags, outs, spans = [], [], [], [], [], []
    for start in range(0, len(indices), CAP.BATCH):
        chunk = indices[start:start + CAP.BATCH]
        if len(chunk) < 2:
            continue
        pack = V2.load_pack(chunk)
        rows = TRUNC.categorise(pack, edges)
        theta_c, rho_c, support = DH.batch_rows(pack, edges)
        f50, _ = LATE.encoder_features(pack, backbone)
        scores = decoder(f50, features)
        chord = np.linalg.norm(rows["q1"] - rows["q0"], axis=-1)
        full_span = np.linalg.norm(rows["p1"] - rows["p0"], axis=-1)
        fraction = chord / np.clip(full_span, 1e-9, None)
        for frame in range(scores.shape[0]):
            live = torch.nonzero(support[frame]).flatten()
            if live.numel() == 0:
                continue
            theta_p, rho_p = DH.decode(scores[frame][live], grid_theta,
                                       grid_rho, valid)
            a, o = DH.measure(theta_p, rho_p, theta_c[frame][live],
                              rho_c[frame][live])
            index = live.cpu().numpy()
            angle.append(a); offset.append(o); roles.append(index)
            tags.append(np.where(rows["T0_FULL"][frame][index], "T0_FULL",
                                 "T1_PARTIAL"))
            outs.append(rows["endpoints_outside"][frame][index])
            spans.append(fraction[frame][index])
    return {"angle": np.concatenate(angle), "offset": np.concatenate(offset),
            "role": np.concatenate(roles), "tag": np.concatenate(tags),
            "endpoints_outside": np.concatenate(outs),
            "chord_fraction": np.concatenate(spans)}


def summarise(sliced):
    angle, offset = sliced["angle"], sliced["offset"]
    tag, outs = sliced["tag"], sliced["endpoints_outside"]
    partial = tag == "T1_PARTIAL"
    report = {"blocks": {
        "OVERALL": TRUNC.block(angle, offset),
        "T0_FULL": TRUNC.block(angle[~partial], offset[~partial]),
        "T1_PARTIAL": TRUNC.block(angle[partial], offset[partial])}}
    report["t1_breakdown"] = {
        "ONE_ENDPOINT_OUT": TRUNC.block(angle[partial & (outs == 1)],
                                        offset[partial & (outs == 1)]),
        "BOTH_ENDPOINTS_OUT": TRUNC.block(angle[partial & (outs == 2)],
                                          offset[partial & (outs == 2)])}
    fraction = sliced["chord_fraction"]
    report["span_bins"] = {}
    for low, high in zip(SPAN_BINS[:-1], SPAN_BINS[1:]):
        keep = partial & (fraction > low) & (fraction <= high)
        label = f"{low:.2f}-{high:.2f}"
        report["span_bins"][label] = (TRUNC.block(angle[keep], offset[keep])
                                      if keep.sum() else {"n": 0})
    report["per_role"] = {}
    for role in range(DH.ROLES):
        entry = {}
        for name, mask in (("T0_FULL", ~partial), ("T1_PARTIAL", partial)):
            keep = mask & (sliced["role"] == role)
            if keep.sum():
                piece = TRUNC.block(angle[keep], offset[keep])
                entry[name] = {k: piece[k] for k in
                               ("n", "angle_median", "offset_median",
                                "frac_both_task_gate")}
        report["per_role"][str(role)] = entry
    return report


def restore(arm, step):
    path = CAP.checkpoint_path(f"DH_{arm}", f"step_{step:05d}")
    stored = torch.load(path, map_location=DEV, weights_only=False)
    backbone, decoder = AC.build()
    decoder.load_state_dict(stored["model"])
    current = dict(AC.late_parameters(backbone))
    with torch.no_grad():
        for name, tensor in stored["late_a1"].items():
            current[name].copy_(tensor.to(current[name].device))
    return backbone, decoder, str(path), hashlib.sha256(path.read_bytes()).hexdigest()


def judge(b0, b1):
    """B1 against B0 on T1_PARTIAL, with T0_FULL as the safety condition."""
    primary = {arm: report["blocks"]["T1_PARTIAL"] for arm, report in
               (("B0", b0), ("B1", b1))}
    safety = {arm: report["blocks"]["T0_FULL"] for arm, report in
              (("B0", b0), ("B1", b1))}
    improvement = {
        k: 1.0 - primary["B1"][k] / primary["B0"][k]
        for k in ("angle_median", "offset_median")}
    gate_points = (primary["B1"]["frac_both_task_gate"]
                   - primary["B0"]["frac_both_task_gate"])
    degradation = {k: primary["B1"][k] / primary["B0"][k] - 1.0
                   for k in ("angle_median", "offset_median")}
    safety_degradation = {k: safety["B1"][k] / safety["B0"][k] - 1.0
                          for k in ("angle_median", "offset_median")}
    conditions = {
        "A_both_medians_15pct_better":
            all(v >= A_MEDIAN_REDUCTION for v in improvement.values()),
        "B_gate_plus_10_points": gate_points >= B_GATE_POINTS,
        "C_T0_safety": all(v <= C_SAFETY_DEGRADATION
                           for v in safety_degradation.values()),
        "both_medians_worse": all(v > 0.0 for v in degradation.values())}
    out = {"primary": primary, "safety": safety, "improvement": improvement,
           "gate_points": gate_points, "safety_degradation": safety_degradation,
           "conditions": conditions}
    if conditions["A_both_medians_15pct_better"] and conditions["B_gate_plus_10_points"]:
        out["DECISION"] = ("PARTIAL_EDGE_SUPERVISION_USEFUL"
                           if conditions["C_T0_safety"]
                           else "PARTIAL_EDGE_SUPERVISION_TRADEOFF")
    elif conditions["both_medians_worse"]:
        out["DECISION"] = "PARTIAL_EDGE_SUPERVISION_HURTS"
    else:
        out["DECISION"] = "PARTIAL_EDGE_SUPERVISION_NOT_USEFUL"
    out["SEMANTICS"] = "PARTIAL_IN_FRAME_STRUCTURAL_EDGE"
    out["CAUSAL_LIMIT"] = (
        "training on partially in-frame structural edges, evaluated on "
        "geometrically truncated edges; occlusion is not evaluated anywhere")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["qualify", "mask", "plan",
                                            "train", "evaluate"])
    arguments = parser.parse_args()
    import instance_edge_topology as IET
    edges = [tuple(e) for e in IET.build_topology()["edges"]]
    if not V2.sha_file(OUT / "line_internal_split.csv").startswith(V2.LINE_SPLIT_SHA):
        raise RuntimeError("HARD_BLOCKED: LINE split changed")

    if arguments.command == "qualify":
        report = run_qualification(edges)
        (OUT / "partial_supervision_qualification.json").write_text(
            json.dumps(report, indent=2, default=float))
        log(f"[qualify] init decoder {report['init']['decoder_max_abs']:.3e} "
            f"late-A1 {report['init']['late_a1_max_abs']:.3e} groups "
            f"{report['init']['param_groups_match']}")
        b = report["batch"]
        log(f"[qualify] batch views {b['view_a_max_abs']:.3e}/"
            f"{b['view_b_max_abs']:.3e} geometry {b['geometry_max_abs']:.3e} "
            f"support {b['support_equal']} category {b['category_equal']} "
            f"L_sup {b['sup_max_abs']:.3e}")
        log(f"[qualify] control gap {max(report['control_gap'].values()):.3e}  "
            f"cross gap {max(report['cross_gap'].values()):.3e}")
        log(f"[qualify] HISTORICAL_P0_REUSE_QUALIFIED="
            f"{report['HISTORICAL_P0_REUSE_QUALIFIED']}")
        return

    if arguments.command == "mask":
        report = run_mask_factor(edges)
        (OUT / "partial_supervision_mask_factor.json").write_text(
            json.dumps(report, indent=2, default=float))
        log(f"[mask] counts {report['counts']}")
        log(f"[mask] T0 loss gap {report['t0_loss_equal']:.3e} | T1 contrib B0 "
            f"{report['t1_contribution_B0']:.3e} B1 "
            f"{report['t1_contribution_B1']:.6f} | T2 B0 "
            f"{report['t2_contribution_B0']:.3e} B1 "
            f"{report['t2_contribution_B1']:.3e}")
        log(f"[mask] grad T1 B0 {report['grad_T1_B0']:.3e} B1 "
            f"{report['grad_T1_B1']:.3e} | grad T0 rel gap "
            f"{report['grad_T0_relative_gap']:.3e}  ISOLATED="
            f"{report['PARTIAL_EDGE_MASK_FACTOR_ISOLATED']}")
        if not report["PARTIAL_EDGE_MASK_FACTOR_ISOLATED"]:
            raise RuntimeError("PARTIAL_EDGE_MASK_FACTOR_NOT_ISOLATED")
        return

    if arguments.command == "train":
        for name, key in (("partial_supervision_qualification.json",
                           "HISTORICAL_P0_REUSE_QUALIFIED"),
                          ("partial_supervision_mask_factor.json",
                           "PARTIAL_EDGE_MASK_FACTOR_ISOLATED")):
            path = OUT / name
            if not path.exists() or not json.loads(path.read_text())[key]:
                raise RuntimeError(f"BLOCKED: {name} must pass first")
        pool = V2.split_indices()[0]
        per_pass = V2.steps_per_pass(pool, CAP.BATCH)
        history = train_arm("B0_FULL_ONLY", pool, MARKS, edges,
                            SCALE.populations(), per_pass)
        (OUT / "partial_supervision_b0.json").write_text(
            json.dumps({"arm": "B0_FULL_ONLY", "history": history,
                        **CAP.provenance()}, indent=2, default=float))
        log("[train] B0_FULL_ONLY complete")
        return

    if arguments.command == "evaluate":
        _, dev = V2.split_indices()
        reports, provenance = {}, {}
        for label, arm in (("B0", "B0_FULL_ONLY"), ("B1", P0_ARM)):
            backbone, decoder, path, sha = restore(arm, DECISION_STEP)
            reports[label] = summarise(stratified(dev, decoder, backbone, edges))
            provenance[label] = {"arm": arm, "checkpoint": path,
                                 "checkpoint_sha256": sha}
            del backbone, decoder
            torch.cuda.empty_cache()
        verdict = judge(reports["B0"], reports["B1"])
        report = {"arms": {"B0": "B0_FULL_ONLY",
                           "B1": f"{P0_ARM} (reused, qualified)"},
                  "reports": reports, "provenance": provenance,
                  "verdict": verdict,
                  "qualification": json.loads(
                      (OUT / "partial_supervision_qualification.json").read_text()),
                  "b0_history": json.loads(
                      (OUT / "partial_supervision_b0.json").read_text())["history"],
                  **NOT_EVALUATED, **CAP.provenance()}
        (OUT / "partial_supervision_result.json").write_text(
            json.dumps(report, indent=2, default=float))
        for label in ("B0", "B1"):
            for name in ("T0_FULL", "T1_PARTIAL"):
                e = reports[label]["blocks"][name]
                log(f"[eval] {label} {name:11s} n {e['n']:6d} angle "
                    f"{e['angle_median']:7.4f} p90 {e['angle_p90']:7.3f} | offset "
                    f"{e['offset_median']:7.4f} | gate {e['frac_both_task_gate']:6.2%}")
        log(f"[eval] improvement {verdict['improvement']} gate_points "
            f"{verdict['gate_points']:+.4f}")
        log(f"[eval] {verdict['DECISION']}  conditions {verdict['conditions']}")
        return

    checkpoint = CAP.checkpoint_path(f"DH_{P0_ARM}", f"step_{DECISION_STEP:05d}")
    plan = {"arms": list(ARMS), "support_mode": SUPPORT_MODE,
            "base_recipe": "P0_AUGMENTATION_ONLY",
            "consistency": None, "lambda_cons": None,
            "marks": list(MARKS), "decision_step": DECISION_STEP,
            "span_bins": list(SPAN_BINS),
            "gates": {"A_median_reduction": A_MEDIAN_REDUCTION,
                      "B_gate_points": B_GATE_POINTS,
                      "C_safety_degradation": C_SAFETY_DEGRADATION},
            "p0_provenance": {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": hashlib.sha256(
                    checkpoint.read_bytes()).hexdigest() if checkpoint.exists() else None,
                "result_json": P0_RESULT,
                "screen_commit": "da457ac"},
            **NOT_EVALUATED, **CAP.provenance()}
    (OUT / "partial_supervision_plan.json").write_text(
        json.dumps(plan, indent=2, default=float))
    log(f"[plan] arms {ARMS} support {SUPPORT_MODE} decision {DECISION_STEP}")
    log(f"[plan] P0 checkpoint sha "
        f"{plan['p0_provenance']['checkpoint_sha256'][:24]}...")


if __name__ == "__main__":
    main()
