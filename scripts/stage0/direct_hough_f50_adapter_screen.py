"""Can a small line adapter keep the late-A1 gain without unfreezing A1?

Phase B bought roughly 45% off both D2 medians by letting gradient reach
`net.vgg[19:27]`, and it arrived with a D0/D2 gap that opened from 1-7% to 42.5%
and a run that peaked before its own decision step.  That looks less like a
capacity shortfall than like more backbone adaptation than the task needed.  So
A1 goes back to fully frozen and one zero-init residual adapter sits on F50.

```
baseline   RGB -> frozen A1 -> F50 -> token XY -> 12 role queries -> head
candidate  RGB -> frozen A1 -> F50 -> zero-init residual adapter -> F50'
                             -> token XY -> the same encoder -> the same head
```

One factor: `CONSTRAINED_F50_LINE_ADAPTER`.  Fixed shape, no sweep --
1x1 128->32, ReLU, 3x3 32->32, ReLU, 1x1 32->128, and `F50' = F50 + alpha *
adapter(F50)` with a learnable scalar alpha initialised to zero.

Zero init makes step 0 a testable claim rather than an intention, so `step0`
checks logits and descriptors against the frozen baseline at 1e-6 and `wiring`
checks that gradient actually reaches both ends of the adapter while A1 receives
none.  Both must pass before training.

Decision at 25,545 on `D2_LINE_DEV512`.  The primary baseline is Phase A's F0
loaded at full precision; Phase B's F1 is context only and selects nothing.
Scope and labels are fixed in `DIRECT_HOUGH_ADAPTATION_SCOPE_ADDENDUM.md`.
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


LONG = _load("DH_LONG_C", "scripts/stage0/direct_hough_full_step_extension.py")
DH, CAP, V2, SCALE = LONG.DH, LONG.CAP, LONG.V2, LONG.SCALE
OUT, DEV = LONG.OUT, LONG.DEV

F50_CHANNELS = DH.F50_CHANNELS
BOTTLENECK = 32                       # 4:1, fixed, not swept
MARKS = LONG.LONG_MARKS
DECISION_STEP = LONG.DECISION_STEP
PER_ROLE_MARKS = LONG.PER_ROLE_MARKS
DIAGNOSTIC_MARKS = LONG.DIAGNOSTIC_MARKS
PHASE_A_RESULT = "direct_hough_long.json"          # F0, the primary baseline
PHASE_B_RESULT = "late_a1_adaptation.json"         # F1, context only
PROBE_FRAMES = 32
STEP0_TOLERANCE = 1e-6
REDUCTION = DH.REDUCTION
TAG = "f2"


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


class F50LineAdapter(nn.Module):
    """Zero-gated residual refinement of F50.  Identity at initialisation."""

    def __init__(self, channels=F50_CHANNELS, bottleneck=BOTTLENECK):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, bottleneck, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck, bottleneck, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck, channels, 1))
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, f50):
        return f50 + self.alpha * self.body(f50)

    def report(self):
        return {"channels": F50_CHANNELS, "bottleneck": BOTTLENECK,
                "params": sum(p.numel() for p in self.parameters()),
                "body_params": sum(p.numel() for p in self.body.parameters()),
                "alpha_params": self.alpha.numel(),
                "alpha": float(self.alpha.detach())}


def frozen_a1():
    """A1 with every parameter frozen, verified rather than assumed."""
    a1 = V2.load_a1()
    for parameter in a1.parameters():
        parameter.requires_grad_(False)
    return a1


def trainable_a1_params(a1):
    return sum(p.numel() for p in a1.parameters() if p.requires_grad)


def build_pair():
    """A baseline and a candidate whose shared modules are bit-identical.

    `DirectHoughModel` seeds itself, so two constructions agree -- but only if
    nothing consumes RNG between them.  The adapter is therefore built *after*
    both base models exist and its state is copied in, so its own initialisation
    cannot shift the encoder or the head.  Step 0 equivalence is then a property
    of the construction, not a hope, and `run_step0` measures it anyway.
    """
    baseline = DH.DirectHoughModel().to(DEV)
    candidate = DH.DirectHoughModel().to(DEV)
    candidate.load_state_dict(baseline.state_dict())
    adapter = F50LineAdapter().to(DEV)
    return baseline, candidate, adapter


def base_checksum(model):
    digest = 0.0
    for name, parameter in sorted(model.state_dict().items()):
        digest += float(parameter.double().abs().sum())
    return digest


def parameter_audit(model, adapter, a1):
    encoder = sum(p.numel() for p in model.encoder.parameters())
    head = sum(p.numel() for p in model.head.parameters())
    return {"adapter": adapter.report(),
            "role_encoder_params": encoder,
            "direct_hough_head_params": head,
            "a1_params": sum(p.numel() for p in a1.parameters()),
            "a1_trainable_params": trainable_a1_params(a1),
            "trainable_total": adapter.report()["params"] + encoder + head}


def probe_pack():
    return V2.load_pack(V2.split_indices()[0][:CAP.BATCH])


@torch.no_grad()
def base_f50(pack, a1):
    f50, _, _ = a1(pack["images"])
    return f50.detach()


def run_step0(edges):
    """Does the candidate reproduce the baseline exactly before any step?"""
    a1 = frozen_a1()
    baseline, candidate, adapter = build_pair()
    grid_theta, grid_rho, valid = DH.lattice()
    features = DH.hypothesis_features(grid_theta, grid_rho)
    indices = V2.split_indices()[0][:PROBE_FRAMES]
    logit_gap = descriptor_gap = 0.0
    with torch.no_grad():
        for start in range(0, len(indices), CAP.BATCH):
            pack = V2.load_pack(indices[start:start + CAP.BATCH])
            f50 = base_f50(pack, a1)
            reference = baseline(f50, features)
            adapted = candidate(adapter(f50), features)
            logit_gap = max(logit_gap, float((reference - adapted).abs().max()))
            descriptor_gap = max(descriptor_gap, float(
                (baseline.descriptors(f50)
                 - candidate.descriptors(adapter(f50))).abs().max()))
    report = {"frames": PROBE_FRAMES, "tolerance": STEP0_TOLERANCE,
              "logit_max_abs": logit_gap, "descriptor_max_abs": descriptor_gap,
              "alpha_at_init": float(adapter.alpha.detach()),
              "base_checksum_equal": bool(
                  base_checksum(baseline) == base_checksum(candidate)),
              "audit": parameter_audit(candidate, adapter, a1)}
    report["F50_ADAPTER_STEP0_EQUIVALENT"] = bool(
        logit_gap <= STEP0_TOLERANCE and descriptor_gap <= STEP0_TOLERANCE
        and report["base_checksum_equal"])
    return report


def run_wiring(edges):
    """Does gradient reach both ends of the adapter, and never reach A1?"""
    a1 = frozen_a1()
    _, model, adapter = build_pair()
    grid_theta, grid_rho, valid = DH.lattice()
    features = DH.hypothesis_features(grid_theta, grid_rho)
    optimiser = torch.optim.AdamW(
        [{"params": list(model.parameters())},
         {"params": list(adapter.parameters())}],
        lr=CAP.LR, weight_decay=CAP.WD)
    pack = probe_pack()
    theta_c, rho_c, support = DH.batch_rows(pack, edges)
    target = DH.target_distribution(
        theta_c.reshape(-1), rho_c.reshape(-1), grid_theta, grid_rho, valid
    ).reshape(*theta_c.shape, -1)
    f50 = base_f50(pack, a1)

    def step():
        loss = DH.cross_entropy(model(adapter(f50), features), target, support,
                                valid)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        return loss

    step()
    alpha_grad = float(adapter.alpha.grad.abs().max())
    optimiser.step()
    alpha_after = float(adapter.alpha.detach().abs().max())
    step()
    first = adapter.body[0].weight
    last = adapter.body[4].weight
    report = {"alpha_grad_at_step0": alpha_grad,
              "alpha_after_one_step": alpha_after,
              "first_conv_grad_norm": float(first.grad.norm()),
              "last_conv_grad_norm": float(last.grad.norm()),
              "a1_params_with_grad": sum(
                  1 for p in a1.parameters()
                  if p.grad is not None and float(p.grad.abs().sum()) != 0.0),
              "a1_trainable_params": trainable_a1_params(a1)}
    report["F50_ADAPTER_GRADIENT_WIRING"] = bool(
        alpha_grad > 0.0 and alpha_after != 0.0
        and report["first_conv_grad_norm"] > 0.0
        and report["last_conv_grad_norm"] > 0.0
        and report["a1_params_with_grad"] == 0
        and report["a1_trainable_params"] == 0)
    return report


def run_memory(edges):
    """One real batch-8 train step, with the peak recorded."""
    torch.cuda.reset_peak_memory_stats(DEV)
    a1 = frozen_a1()
    _, model, adapter = build_pair()
    grid_theta, grid_rho, valid = DH.lattice()
    features = DH.hypothesis_features(grid_theta, grid_rho)
    optimiser = torch.optim.AdamW(
        [{"params": list(model.parameters())},
         {"params": list(adapter.parameters())}],
        lr=CAP.LR, weight_decay=CAP.WD)
    pack = probe_pack()
    theta_c, rho_c, support = DH.batch_rows(pack, edges)
    target = DH.target_distribution(
        theta_c.reshape(-1), rho_c.reshape(-1), grid_theta, grid_rho, valid
    ).reshape(*theta_c.shape, -1)
    f50 = base_f50(pack, a1)
    loss = DH.cross_entropy(model(adapter(f50), features), target, support, valid)
    optimiser.zero_grad(set_to_none=True); loss.backward(); optimiser.step()
    peak = torch.cuda.max_memory_allocated(DEV)
    total = torch.cuda.get_device_properties(DEV).total_memory
    return {"batch": CAP.BATCH, "peak_bytes": int(peak),
            "peak_mib": peak / 2 ** 20, "device_total_mib": total / 2 ** 20,
            "loss": float(loss.detach()),
            "F50_ADAPTER_BATCH8_MEMORY_OK": bool(peak < total)}


@torch.no_grad()
def adapter_use(indices, adapter, a1):
    """Is the adapter doing anything, and how much?"""
    relative, cosine, ratio = [], [], []
    for start in range(0, len(indices), CAP.BATCH):
        pack = V2.load_pack(indices[start:start + CAP.BATCH])
        base = base_f50(pack, a1)
        adapted = adapter(base)
        delta = adapted - base
        flat_base = base.flatten(1)
        flat_adapted = adapted.flatten(1)
        relative.append(float(delta.flatten(1).norm(dim=1).mean()
                              / flat_base.norm(dim=1).mean().clamp_min(1e-12)))
        cosine.append(float(nn.functional.cosine_similarity(
            flat_base, flat_adapted, dim=1).mean()))
        ratio.append(float(adapter.body(base).flatten(1).norm(dim=1).mean()
                           / flat_base.norm(dim=1).mean().clamp_min(1e-12)))
    return {"relative_l2": float(np.mean(relative)),
            "cosine_similarity": float(np.mean(cosine)),
            "adapter_output_norm_ratio": float(np.mean(ratio)),
            "alpha": float(adapter.alpha.detach())}


@torch.no_grad()
def evaluate(indices, model, adapter, a1, edges, features, grid_theta, grid_rho,
             valid, per_role=False):
    """DH.evaluate_network with the adapter in the feature path."""
    model.eval()
    angle, offset, roles = [], [], []
    for start in range(0, len(indices), CAP.BATCH):
        chunk = indices[start:start + CAP.BATCH]
        if len(chunk) < 2:
            continue
        pack = V2.load_pack(chunk)
        theta_c, rho_c, support = DH.batch_rows(pack, edges)
        scores = model(adapter(base_f50(pack, a1)), features)
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


def train_adapter(pool, marks, edges, populations, per_pass, probe):
    grid_theta, grid_rho, valid = DH.lattice()
    features = DH.hypothesis_features(grid_theta, grid_rho)
    a1 = frozen_a1()
    _, model, adapter = build_pair()
    if trainable_a1_params(a1) != 0:
        raise RuntimeError("HARD_BLOCK: A1 is not fully frozen")
    start_state = {k: v.detach().clone() for k, v in adapter.state_dict().items()}
    optimiser = torch.optim.AdamW(
        [{"params": list(model.parameters()), "lr": CAP.LR},
         {"params": list(adapter.parameters()), "lr": CAP.LR}],
        lr=CAP.LR, weight_decay=CAP.WD)
    history, losses, done = {}, [], 0
    for chunk, _ in V2.step_schedule(pool, max(marks), CAP.BATCH):
        model.train(); adapter.train()
        pack = V2.load_pack(chunk)
        theta_c, rho_c, support = DH.batch_rows(pack, edges)
        target = DH.target_distribution(
            theta_c.reshape(-1), rho_c.reshape(-1), grid_theta, grid_rho, valid
        ).reshape(*theta_c.shape, -1)
        loss = DH.cross_entropy(model(adapter(base_f50(pack, a1)), features),
                                target, support, valid)
        optimiser.zero_grad(set_to_none=True)
        loss.backward(); optimiser.step()
        losses.append(float(loss.detach()))
        done += 1
        if done in marks:
            adapter.eval()
            entry = {"step": done,
                     "train_loss_mean_last250": float(np.mean(losses[-250:])),
                     "train_loss_slope_last250": LONG.slope(losses[-250:]),
                     "train_loss_mean_last_pass": float(np.mean(losses[-per_pass:])),
                     "train_loss_slope_last_pass": LONG.slope(losses[-per_pass:]),
                     "diagnostic_only": done in DIAGNOSTIC_MARKS,
                     "adapter_use": adapter_use(probe, adapter, a1),
                     "adapter_delta_norm": float(sum(
                         (adapter.state_dict()[k].detach() - v).norm() ** 2
                         for k, v in start_state.items()) ** 0.5),
                     "finite": bool(np.isfinite(losses[-1]))}
            for label, indices in populations.items():
                entry[label] = evaluate(indices, model, adapter, a1, edges,
                                        features, grid_theta, grid_rho, valid,
                                        per_role=(label == "D2_LINE_DEV512"
                                                  and done in PER_ROLE_MARKS))
                log(f"  {TAG} @{done:6d} {label:<16} angle med "
                    f"{entry[label]['angle_median']:7.4f} p90 "
                    f"{entry[label]['angle_p90']:7.3f} | offset med "
                    f"{entry[label]['offset_median']:7.4f} p90 "
                    f"{entry[label]['offset_p90']:7.3f}")
            d0, d2 = entry["D0_SEEN512"], entry["D2_LINE_DEV512"]
            entry["generalization"] = {
                "angle_ratio": d2["angle_median"] / d0["angle_median"],
                "offset_ratio": d2["offset_median"] / d0["offset_median"]}
            use = entry["adapter_use"]
            log(f"  {TAG} @{done:6d} CE {entry['train_loss_mean_last250']:.6f} "
                f"slope {entry['train_loss_slope_last_pass']:+.3e} | alpha "
                f"{use['alpha']:+.5f} relL2 {use['relative_l2']:.5f} cos "
                f"{use['cosine_similarity']:.6f} | D2/D0 "
                f"{entry['generalization']['angle_ratio']:.3f}/"
                f"{entry['generalization']['offset_ratio']:.3f}")
            torch.save({"tag": TAG, "step": done, "model": model.state_dict(),
                        "adapter": adapter.state_dict(), **CAP.provenance()},
                       CAP.checkpoint_path(f"DH_{TAG}", f"step_{done:05d}"))
            history[str(done)] = entry
    return history, model, adapter


def reference(name, source):
    return json.loads((OUT / name).read_text())[source]


def baselines():
    f0 = json.loads((OUT / PHASE_A_RESULT).read_text())["history"]
    f1 = json.loads((OUT / PHASE_B_RESULT).read_text())["histories"]
    return (f0[str(DECISION_STEP)], f1["F1_LATE_A1_TRAINABLE"][str(DECISION_STEP)])


def judge(history):
    """Primary is D2 at 25,545 against Phase A's F0.  F1 selects nothing."""
    f0_entry, f1_entry = baselines()
    f0 = f0_entry["D2_LINE_DEV512"]
    f1 = f1_entry["D2_LINE_DEV512"]
    f2 = history[str(DECISION_STEP)]["D2_LINE_DEV512"]
    keys = ("angle_median", "offset_median", "angle_p90", "offset_p90")
    out = {"decision_step": DECISION_STEP, "population": "D2_LINE_DEV512",
           "F0": {k: f0[k] for k in keys},
           "F2": {k: f2[k] for k in keys},
           "F1_context_only": {k: f1[k] for k in keys},
           "vs_F0": {k: 1.0 - f2[k] / f0[k] for k in keys},
           "vs_F1_context_only": {k: 1.0 - f2[k] / f1[k] for k in keys},
           "ABSOLUTE_PASS": bool(f2["PASS"] and f2["SAFETY"]),
           "finite": bool(history[str(DECISION_STEP)]["finite"])}
    out["REDUCTION_40"] = bool(
        out["vs_F0"]["angle_median"] >= REDUCTION
        and out["vs_F0"]["offset_median"] >= REDUCTION)
    dominates = all(f2[k] <= f1[k] for k in keys)
    out["F50_ADAPTER_PARETO_BETTER_THAN_LATE_UNFREEZE"] = bool(
        dominates and any(f2[k] < f1[k] for k in keys))
    final = history[str(DECISION_STEP)]
    out["generalization"] = {
        "F2": final["generalization"],
        "F1_context_only": {
            "angle_ratio": f1["angle_median"] / f1_entry["D0_SEEN512"]["angle_median"],
            "offset_ratio": f1["offset_median"] / f1_entry["D0_SEEN512"]["offset_median"]},
        "F0_context_only": {
            "angle_ratio": f0["angle_median"] / f0_entry["D0_SEEN512"]["angle_median"],
            "offset_ratio": f0["offset_median"] / f0_entry["D0_SEEN512"]["offset_median"]}}
    d0 = final["D0_SEEN512"]
    out["SPECIALIZES"] = bool(
        1.0 - d0["angle_median"] / f0_entry["D0_SEEN512"]["angle_median"] >= REDUCTION
        and out["vs_F0"]["angle_median"] < REDUCTION)
    if not out["finite"]:
        out["DECISION"] = "F50_ADAPTER_TRAINING_UNSTABLE"
        out["RETRY_WITH_NEW_LR"] = "FORBIDDEN"
    elif out["ABSOLUTE_PASS"]:
        out["DECISION"] = "F50_ADAPTER_VALID_CANDIDATE"
        out["STATUS"] = "LINE_STAGE_CANDIDATE"
        out["NEXT"] = "same_protocol_execution_replicate"
    elif out["REDUCTION_40"]:
        out["DECISION"] = "F50_ADAPTER_SIGNAL"
        out["NEXT"] = "ROLE_ENCODER_DEPTH_SCREEN"
    else:
        out["DECISION"] = "F50_ADAPTER_INSUFFICIENT"
        out["NEXT"] = "ROLE_ENCODER_DEPTH_SCREEN"
        out["BROADER_UNFREEZE_PROVEN_NECESSARY"] = False
    out.setdefault("STATUS", "NOT_LOCKED")
    out["CIGM"] = "BLOCKED"
    return out


def build_plan(pool):
    a1 = frozen_a1()
    _, model, adapter = build_pair()
    f0_entry, f1_entry = baselines()
    plan = {"factor": "CONSTRAINED_F50_LINE_ADAPTER",
            "marks": list(MARKS), "decision_step": DECISION_STEP,
            "decision_population": "D2_LINE_DEV512",
            "diagnostic_population": "D0_SEEN512",
            "per_role_marks": list(PER_ROLE_MARKS),
            "frames": len(pool), "batch": CAP.BATCH,
            "audit": parameter_audit(model, adapter, a1),
            "adapter_lr": CAP.LR, "encoder_lr": CAP.LR, "head_lr": CAP.LR,
            "weight_decay": CAP.WD, "scheduler": None, "gradient_clipping": None,
            "lr_sweep": False, "bottleneck_sweep": False, "depth_sweep": False,
            "gate": {"angle_median": CAP.ANGLE_BUDGET_DEG,
                     "offset_median": CAP.OFFSET_BUDGET_CELL,
                     "angle_p90": CAP.SAFETY_ANGLE,
                     "offset_p90": CAP.SAFETY_OFFSET},
            "reduction": REDUCTION,
            "baseline_F0_full_precision": f0_entry["D2_LINE_DEV512"],
            "context_F1_full_precision": f1_entry["D2_LINE_DEV512"],
            **CAP.provenance()}
    del a1, model, adapter
    torch.cuda.empty_cache()
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "step0", "wiring", "memory",
                                            "run"])
    arguments = parser.parse_args()
    import instance_edge_topology as IET
    edges = [tuple(e) for e in IET.build_topology()["edges"]]
    if not V2.sha_file(OUT / "line_internal_split.csv").startswith(V2.LINE_SPLIT_SHA):
        raise RuntimeError("HARD_BLOCKED: LINE split changed")
    for name in (PHASE_A_RESULT, PHASE_B_RESULT):
        if not (OUT / name).exists():
            raise RuntimeError(f"HARD_BLOCK: {name} is missing")
    pool = V2.split_indices()[0]

    if arguments.command == "plan":
        plan = build_plan(pool)
        (OUT / "f50_adapter_plan.json").write_text(json.dumps(plan, indent=2))
        audit = plan["audit"]
        log(f"[plan] adapter {audit['adapter']['params']:,} params "
            f"(bottleneck {BOTTLENECK}) | encoder "
            f"{audit['role_encoder_params']:,} | head "
            f"{audit['direct_hough_head_params']:,} | A1 trainable "
            f"{audit['a1_trainable_params']}")
        log(f"[plan] decision {DECISION_STEP} on D2 only  gate "
            f"{CAP.ANGLE_BUDGET_DEG}/{CAP.OFFSET_BUDGET_CELL} safety "
            f"{CAP.SAFETY_ANGLE}/{CAP.SAFETY_OFFSET}  reduction {REDUCTION}")
        return

    if arguments.command == "step0":
        report = run_step0(edges)
        (OUT / "f50_adapter_step0.json").write_text(
            json.dumps(report, indent=2, default=float))
        log(f"[step0] logits {report['logit_max_abs']:.3e}  descriptors "
            f"{report['descriptor_max_abs']:.3e}  tol {STEP0_TOLERANCE}  "
            f"EQUIVALENT={report['F50_ADAPTER_STEP0_EQUIVALENT']}")
        if not report["F50_ADAPTER_STEP0_EQUIVALENT"]:
            raise RuntimeError("F50_ADAPTER_STEP0_MISMATCH")
        return

    if arguments.command == "wiring":
        report = run_wiring(edges)
        (OUT / "f50_adapter_wiring.json").write_text(
            json.dumps(report, indent=2, default=float))
        log(f"[wiring] alpha grad {report['alpha_grad_at_step0']:.3e} -> alpha "
            f"{report['alpha_after_one_step']:.3e} | first conv "
            f"{report['first_conv_grad_norm']:.3e} last conv "
            f"{report['last_conv_grad_norm']:.3e} | A1 with grad "
            f"{report['a1_params_with_grad']}  OK="
            f"{report['F50_ADAPTER_GRADIENT_WIRING']}")
        if not report["F50_ADAPTER_GRADIENT_WIRING"]:
            raise RuntimeError("F50_ADAPTER_GRADIENT_WIRING_FAIL")
        return

    if arguments.command == "memory":
        report = run_memory(edges)
        (OUT / "f50_adapter_memory.json").write_text(
            json.dumps(report, indent=2, default=float))
        log(f"[memory] batch {report['batch']} peak {report['peak_mib']:.1f} MiB "
            f"of {report['device_total_mib']:.0f} MiB  OK="
            f"{report['F50_ADAPTER_BATCH8_MEMORY_OK']}")
        if not report["F50_ADAPTER_BATCH8_MEMORY_OK"]:
            raise RuntimeError("F50_ADAPTER_BATCH8_MEMORY_FAIL")
        return

    for name, key, label in (
            ("f50_adapter_step0.json", "F50_ADAPTER_STEP0_EQUIVALENT",
             "F50_ADAPTER_STEP0_MISMATCH"),
            ("f50_adapter_wiring.json", "F50_ADAPTER_GRADIENT_WIRING",
             "F50_ADAPTER_GRADIENT_WIRING_FAIL"),
            ("f50_adapter_memory.json", "F50_ADAPTER_BATCH8_MEMORY_OK",
             "F50_ADAPTER_BATCH8_MEMORY_FAIL")):
        path = OUT / name
        if not path.exists() or not json.loads(path.read_text())[key]:
            raise RuntimeError(f"{label}: preflight must pass first")
    plan = build_plan(pool)
    per_pass = V2.steps_per_pass(pool, CAP.BATCH)
    probe = V2.split_indices()[0][:PROBE_FRAMES]
    history, _, adapter = train_adapter(pool, MARKS, edges, SCALE.populations(),
                                        per_pass, probe)
    report = {"plan": plan, "history": history, "verdict": judge(history),
              "adapter_final": adapter.report(), **CAP.provenance()}
    (OUT / "f50_adapter.json").write_text(
        json.dumps(report, indent=2, default=float))
    v = report["verdict"]
    log(f"[run] {v['DECISION']}  F2 {v['F2']['angle_median']:.6f}/"
        f"{v['F2']['offset_median']:.6f} p90 {v['F2']['angle_p90']:.6f}/"
        f"{v['F2']['offset_p90']:.6f}")
    log(f"[run] vs F0 angle {v['vs_F0']['angle_median']:+.2%} offset "
        f"{v['vs_F0']['offset_median']:+.2%} | pareto vs F1 "
        f"{v['F50_ADAPTER_PARETO_BETTER_THAN_LATE_UNFREEZE']} | SPECIALIZES "
        f"{v['SPECIALIZES']}")


if __name__ == "__main__":
    main()
