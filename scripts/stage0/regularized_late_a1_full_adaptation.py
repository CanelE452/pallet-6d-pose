"""Does anchoring the late weights to their pretrained values change F1's tradeoff?

Every arm so far constrained *where* adaptation happens or *how many* parameters
it may use.  None constrained how far the weights may move, and the low-rank arm
showed that a structural constraint is not a magnitude constraint: at rank 8 the
effective delta kernels ended larger than the frozen kernels they corrected.  So
the untested factor is a penalty on magnitude.

```
S0   historical F1 exactly: net.vgg[19:27] fully trainable at the F1 A1 rate,
     the original role encoder and DirectHoughHead, no adapter, no extra block,
     no low-rank branch
S1   S0 plus an L2-SP term
```

```
r_l      = ||W_l - W_l0||_F^2 / (||W_l0||_F^2 + eps)
R_SP     = mean over the late tensors of r_l
L_total  = L_task + lambda_sp * R_SP
```

The penalty pulls toward `W0`, not toward zero, and it does not replace the
existing weight decay -- this is F1 plus a term, not a different optimizer.

`lambda_sp` is never chosen by looking at held-out geometry.  It is calibrated
once, deterministically, from the historical F1 state at 1,703 steps so that
`lambda_sp * R_SP_ref == CE_ref` there -- a scale normalisation, not a search.
`LAMBDA_OPTIMALITY_NOT_ESTABLISHED`.

That calibration needs the adapted late weights at that checkpoint.  If they were
not recorded, this HARD_BLOCKs with `SP_CALIBRATION_REFERENCE_MISSING` rather
than estimating or reconstructing them, and a separate coefficient lock is
required before anything runs.

Decision at 25,545 on `D2_LINE_DEV512`; the causal baseline is historical F1 at
full precision, and F0, F2, R1 and L1 are context only.
"""
from __future__ import annotations

import argparse, importlib.util, json, hashlib, pathlib, sys, time
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


LATE = _load("LATE_A1_SP", "scripts/stage0/late_a1_adaptation_screen.py")
LONG, DH = LATE.LONG, LATE.DH
CAP, V2, SCALE = LATE.CAP, LATE.V2, LATE.SCALE
OUT, DEV = LATE.OUT, LATE.DEV

FIRST_TRAINABLE_INDEX = LATE.FIRST_TRAINABLE_INDEX
A1_LR_SCALE = LATE.A1_LR_SCALE
MARKS = LATE.MARKS
DECISION_STEP = LATE.DECISION_STEP
PER_ROLE_MARKS = LATE.PER_ROLE_MARKS
DIAGNOSTIC_MARKS = LATE.DIAGNOSTIC_MARKS
EXPECTED_TRAINABLE = 5014912
PROBE_FRAMES = 32
STEP0_TOLERANCE = 1e-6
SP_EPS = 1e-12
REDUCTION = DH.REDUCTION
F1_RESULT = "late_a1_adaptation.json"
F1_ARM = "F1_LATE_A1_TRAINABLE"
CALIBRATION_STEP = 1703
CALIBRATION_CHECKPOINT = ("DH_f1", f"step_{CALIBRATION_STEP:05d}")
CONTEXT_RESULTS = {"F0_FROZEN": ("direct_hough_long.json", "history", None),
                   "F2_ADAPTER": ("f50_adapter.json", "history", None),
                   "R1_ROLE_DEPTH": ("role_depth.json", "history", None),
                   "L1_LOW_RANK": ("low_rank_a1.json", "history", None)}
ARMS = ("S0_F1_HISTORICAL", "S1_F1_PLUS_L2SP")
# G3 and G5, fixed before any result is read.
G3_MAX_DEGRADATION = 0.10          # both D2 medians within +10% of F1
G3_MIN_GAP_CLOSURE = 0.20          # both ratios 20% closer to 1 than F1's
G5_MEDIAN_BAND = 0.05
G5_GAP_BAND = 0.10
TAG = "s1"


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def late_parameters(a1):
    """The tensors F1 made trainable, in a fixed order."""
    return [(name, parameter)
            for name, parameter in a1.vgg.named_parameters()
            if int(name.split(".")[0]) >= FIRST_TRAINABLE_INDEX]


class SPReference:
    """An immutable clone of the pretrained late weights.

    Cloned straight after the canonical checkpoint is loaded and never written
    to again; `verify` re-checks the digests so a silent mutation during a long
    run cannot pass unnoticed.
    """

    def __init__(self, a1):
        self.entries = []
        for name, parameter in late_parameters(a1):
            reference = parameter.detach().clone()
            reference.requires_grad_(False)
            self.entries.append((name, reference))
        self.audit = [{"name": name,
                       "shape": list(reference.shape),
                       "numel": reference.numel(),
                       "norm": float(reference.norm()),
                       "sha256": hashlib.sha256(
                           reference.cpu().numpy().tobytes()).hexdigest()[:16]}
                      for name, reference in self.entries]

    def verify(self):
        for (name, reference), row in zip(self.entries, self.audit):
            digest = hashlib.sha256(reference.cpu().numpy().tobytes()).hexdigest()[:16]
            if digest != row["sha256"]:
                raise RuntimeError(f"HARD_BLOCK: SP reference mutated at {name}")
        return True

    def penalty(self, a1):
        current = dict(late_parameters(a1))
        terms = []
        for name, reference in self.entries:
            weight = current[name]
            terms.append((weight - reference).pow(2).sum()
                         / (reference.pow(2).sum() + SP_EPS))
        return torch.stack(terms).mean()

    @torch.no_grad()
    def drift(self, a1):
        current = dict(late_parameters(a1))
        rows = {}
        for name, reference in self.entries:
            rows[name] = float((current[name] - reference).norm()
                               / reference.norm().clamp_min(SP_EPS))
        values = list(rows.values())
        return {"per_tensor": rows, "mean": float(np.mean(values)),
                "max": float(np.max(values))}


def build_arm(with_penalty):
    a1 = LATE.AdaptableA1(FIRST_TRAINABLE_INDEX).to(DEV)
    reference = SPReference(a1)
    trainable = sum(p.numel() for p in a1.parameters_to_train())
    if trainable != EXPECTED_TRAINABLE:
        raise RuntimeError(f"L2SP_TRAINABILITY_MISMATCH: {trainable} != "
                           f"{EXPECTED_TRAINABLE}")
    model = DH.DirectHoughModel().to(DEV)
    return a1, model, (reference if with_penalty else None), reference


def optimiser_for(model, a1):
    """F1's groups exactly: head rate for the decoder, 0.1x for late A1."""
    return torch.optim.AdamW(
        [{"params": list(model.parameters()), "lr": CAP.LR},
         {"params": a1.parameters_to_train(), "lr": CAP.LR * A1_LR_SCALE}],
        lr=CAP.LR, weight_decay=CAP.WD)


def f1_reference():
    return json.loads((OUT / F1_RESULT).read_text())["histories"][F1_ARM]


def context_arms():
    out = {}
    for name, (source, key, arm) in CONTEXT_RESULTS.items():
        blob = json.loads((OUT / source).read_text())[key]
        entry = blob[arm][str(DECISION_STEP)] if arm else blob[str(DECISION_STEP)]
        d0, d2 = entry["D0_SEEN512"], entry["D2_LINE_DEV512"]
        out[name] = {k: d2[k] for k in ("angle_median", "offset_median",
                                        "angle_p90", "offset_p90")}
        out[name]["d2_over_d0_angle"] = d2["angle_median"] / d0["angle_median"]
        out[name]["d2_over_d0_offset"] = d2["offset_median"] / d0["offset_median"]
    return out


def calibration_checkpoint():
    return CAP.checkpoint_path(*CALIBRATION_CHECKPOINT)


def adapted_late_weights():
    """The late weights as F1 had them at the calibration step, or None.

    Phase B stored `model.state_dict()` and an audit dict; whether the adapted
    backbone went in with it is a question about the file, so it is answered by
    reading the file.
    """
    path = calibration_checkpoint()
    if not path.exists():
        return None, {"exists": False, "path": str(path)}
    stored = torch.load(path, map_location="cpu", weights_only=False)
    tensors = {}
    for top, value in stored.items():
        if isinstance(value, dict):
            for key, item in value.items():
                if isinstance(item, torch.Tensor) and "vgg" in key.lower():
                    tensors[f"{top}.{key}"] = item
    return (tensors or None), {"exists": True, "path": str(path),
                               "top_level_keys": sorted(stored),
                               "late_weight_tensors_found": len(tensors)}


def run_calibration():
    """One deterministic scale normalisation, from train-side quantities only.

    `CE_ref` is F1's recorded training cross-entropy at the calibration step and
    `R_SP_ref` is the penalty evaluated at F1's weights there.  No held-out
    geometry is read, and no search happens: the coefficient is whatever makes
    the two terms equal at that one state.
    """
    history = f1_reference()
    if str(CALIBRATION_STEP) not in history:
        raise RuntimeError("SP_CALIBRATION_REFERENCE_MISSING: no recorded "
                           f"F1 mark at {CALIBRATION_STEP}")
    ce_ref = history[str(CALIBRATION_STEP)]["train_loss_mean_last250"]
    weights, provenance = adapted_late_weights()
    report = {"calibration_step": CALIBRATION_STEP, "CE_ref": ce_ref,
              "checkpoint": provenance,
              "LAMBDA_OPTIMALITY_NOT_ESTABLISHED": True}
    if weights is None:
        report["R_SP_ref"] = None
        report["lambda_sp"] = None
        report["LAMBDA_LOCKED"] = False
        report["BLOCK"] = "SP_CALIBRATION_REFERENCE_MISSING"
        report["reason"] = (
            "the recorded F1 checkpoint stores the decoder state and an audit "
            "dict, not the adapted net.vgg[19:27] weights, so R_SP_ref cannot "
            "be evaluated at that state; it is not estimated and not "
            "reconstructed by re-running, because the training path is not "
            "bit-reproducible")
        return report
    a1 = LATE.AdaptableA1(FIRST_TRAINABLE_INDEX)
    reference = SPReference(a1)
    with torch.no_grad():
        current = dict(late_parameters(a1))
        for key, tensor in weights.items():
            name = key.split(".", 1)[1]
            if name in current:
                current[name].copy_(tensor.to(current[name].device))
        r_sp_ref = float(reference.penalty(a1))
    report["R_SP_ref"] = r_sp_ref
    report["lambda_sp"] = ce_ref / r_sp_ref if r_sp_ref > 0 else None
    report["LAMBDA_LOCKED"] = bool(report["lambda_sp"])
    del a1
    return report


def locked_lambda():
    path = OUT / "l2sp_calibration.json"
    if not path.exists():
        raise RuntimeError("SP_CALIBRATION_REFERENCE_MISSING: run calibrate first")
    blob = json.loads(path.read_text())
    if not blob.get("LAMBDA_LOCKED"):
        raise RuntimeError(blob.get("BLOCK", "SP_CALIBRATION_REFERENCE_MISSING"))
    return blob["lambda_sp"]


def run_step0(edges):
    """At step 0 the weights are W0, so R_SP is zero and S1 is S0."""
    a1_s0, model_s0, _, _ = build_arm(False)
    a1_s1, model_s1, penalty, reference = build_arm(True)
    model_s1.load_state_dict(model_s0.state_dict())
    grid_theta, grid_rho, valid = DH.lattice()
    features = DH.hypothesis_features(grid_theta, grid_rho)
    indices = V2.split_indices()[0][:PROBE_FRAMES]
    gaps = {"f50": 0.0, "descriptor": 0.0, "logits": 0.0, "task_loss": 0.0}
    with torch.no_grad():
        sp_value = float(penalty.penalty(a1_s1))
        for start in range(0, len(indices), CAP.BATCH):
            pack = V2.load_pack(indices[start:start + CAP.BATCH])
            theta_c, rho_c, support = DH.batch_rows(pack, edges)
            target = DH.target_distribution(
                theta_c.reshape(-1), rho_c.reshape(-1), grid_theta, grid_rho,
                valid).reshape(*theta_c.shape, -1)
            left, _ = LATE.encoder_features(pack, a1_s0)
            right, _ = LATE.encoder_features(pack, a1_s1)
            gaps["f50"] = max(gaps["f50"], float((left - right).abs().max()))
            gaps["descriptor"] = max(gaps["descriptor"], float(
                (model_s0.descriptors(left)
                 - model_s1.descriptors(right)).abs().max()))
            logits_s0 = model_s0(left, features)
            logits_s1 = model_s1(right, features)
            gaps["logits"] = max(gaps["logits"],
                                 float((logits_s0 - logits_s1).abs().max()))
            gaps["task_loss"] = max(gaps["task_loss"], abs(
                float(DH.cross_entropy(logits_s0, target, support, valid))
                - float(DH.cross_entropy(logits_s1, target, support, valid))))
    report = {"frames": PROBE_FRAMES, "tolerance": STEP0_TOLERANCE, "gaps": gaps,
              "R_SP_at_step0": sp_value,
              "reference_audit": reference.audit,
              "reference_verified": reference.verify(),
              "trainable_late_params": sum(
                  p.numel() for p in a1_s1.parameters_to_train())}
    report["L2SP_STEP0_FUNCTION_EQUIVALENT"] = bool(
        all(v <= STEP0_TOLERANCE for v in gaps.values())
        and sp_value <= STEP0_TOLERANCE
        and report["trainable_late_params"] == EXPECTED_TRAINABLE)
    del a1_s0, a1_s1
    return report


def run_gradient_sanity(edges, lambda_sp):
    """SP gradient is zero at step 0 by construction, so it is asked after one.

    A task step moves the weights off `W0`; only then does the penalty have
    anything to push against.  Demanding a non-zero SP gradient at step 0 would
    be demanding a bug.
    """
    a1, model, penalty, _ = build_arm(True)
    grid_theta, grid_rho, valid = DH.lattice()
    features = DH.hypothesis_features(grid_theta, grid_rho)
    optimiser = optimiser_for(model, a1)
    pack = V2.load_pack(V2.split_indices()[0][:CAP.BATCH])
    theta_c, rho_c, support = DH.batch_rows(pack, edges)
    target = DH.target_distribution(
        theta_c.reshape(-1), rho_c.reshape(-1), grid_theta, grid_rho, valid
    ).reshape(*theta_c.shape, -1)

    def task_loss():
        f50, _ = LATE.encoder_features(pack, a1)
        return DH.cross_entropy(model(f50, features), target, support, valid)

    at_zero = float(penalty.penalty(a1))
    optimiser.zero_grad(set_to_none=True)
    task_loss().backward()
    optimiser.step()
    names = [name for name, _ in late_parameters(a1)]
    optimiser.zero_grad(set_to_none=True)
    task_loss().backward()
    task_grads = {n: p.grad.detach().clone()
                  for n, p in late_parameters(a1) if p.grad is not None}
    optimiser.zero_grad(set_to_none=True)
    penalty.penalty(a1).backward()
    sp_grads = {n: p.grad.detach().clone()
                for n, p in late_parameters(a1) if p.grad is not None}
    rows = {}
    for name in names:
        task, sp = task_grads.get(name), sp_grads.get(name)
        if task is None or sp is None:
            continue
        denominator = (task.norm() * sp.norm()).clamp_min(SP_EPS)
        rows[name] = {"task_grad_norm": float(task.norm()),
                      "sp_grad_norm": float(sp.norm()),
                      "cosine": float((task * sp).sum() / denominator)}
    report = {"R_SP_at_step0": at_zero, "lambda_sp": lambda_sp,
              "per_tensor": rows,
              "sp_grad_positive": bool(rows) and all(
                  v["sp_grad_norm"] > 0.0 for v in rows.values()),
              "task_grad_positive": bool(rows) and all(
                  v["task_grad_norm"] > 0.0 for v in rows.values())}
    report["L2SP_GRADIENT_SANITY"] = bool(
        report["sp_grad_positive"] and report["task_grad_positive"]
        and at_zero <= STEP0_TOLERANCE)
    del a1
    return report


def run_memory(edges, lambda_sp):
    torch.cuda.reset_peak_memory_stats(DEV)
    a1, model, penalty, _ = build_arm(True)
    grid_theta, grid_rho, valid = DH.lattice()
    features = DH.hypothesis_features(grid_theta, grid_rho)
    optimiser = optimiser_for(model, a1)
    pack = V2.load_pack(V2.split_indices()[0][:CAP.BATCH])
    theta_c, rho_c, support = DH.batch_rows(pack, edges)
    target = DH.target_distribution(
        theta_c.reshape(-1), rho_c.reshape(-1), grid_theta, grid_rho, valid
    ).reshape(*theta_c.shape, -1)
    f50, _ = LATE.encoder_features(pack, a1)
    task = DH.cross_entropy(model(f50, features), target, support, valid)
    sp = penalty.penalty(a1)
    total = task + lambda_sp * sp
    optimiser.zero_grad(set_to_none=True); total.backward(); optimiser.step()
    peak = torch.cuda.max_memory_allocated(DEV)
    capacity = torch.cuda.get_device_properties(DEV).total_memory
    del a1
    return {"batch": CAP.BATCH, "peak_bytes": int(peak),
            "peak_mib": peak / 2 ** 20, "device_total_mib": capacity / 2 ** 20,
            "task_loss": float(task.detach()), "sp": float(sp.detach()),
            "total_loss": float(total.detach()),
            "L2SP_BATCH8_MEMORY_OK": bool(peak < capacity)}


def judge(history):
    f1_history = f1_reference()
    f1_entry = f1_history[str(DECISION_STEP)]
    f1 = f1_entry["D2_LINE_DEV512"]
    s1 = history[str(DECISION_STEP)]["D2_LINE_DEV512"]
    keys = ("angle_median", "offset_median", "angle_p90", "offset_p90")
    f1_ratio = {"angle": f1["angle_median"] / f1_entry["D0_SEEN512"]["angle_median"],
                "offset": f1["offset_median"] / f1_entry["D0_SEEN512"]["offset_median"]}
    s1_ratio = history[str(DECISION_STEP)]["generalization"]
    closure = {axis: 1.0 - abs(s1_ratio[f"{axis}_ratio"] - 1.0)
               / max(abs(f1_ratio[axis] - 1.0), SP_EPS)
               for axis in ("angle", "offset")}
    degradation = {k: s1[k] / f1[k] - 1.0 for k in ("angle_median",
                                                    "offset_median")}
    out = {"decision_step": DECISION_STEP, "population": "D2_LINE_DEV512",
           "F1": {k: f1[k] for k in keys}, "S1": {k: s1[k] for k in keys},
           "vs_F1": {k: 1.0 - s1[k] / f1[k] for k in keys},
           "degradation_vs_F1": degradation,
           "generalization": {"S1": s1_ratio, "F1": f1_ratio,
                              "gap_closure": closure},
           "context_only": context_arms(),
           "ABSOLUTE_PASS": bool(s1["PASS"] and s1["SAFETY"]),
           "finite": bool(history[str(DECISION_STEP)]["finite"])}
    both_better = all(s1[k] < f1[k] for k in ("angle_median", "offset_median"))
    preserved = all(v <= G3_MAX_DEGRADATION for v in degradation.values())
    closer = all(v >= G3_MIN_GAP_CLOSURE for v in closure.values())
    similar = (all(abs(v) <= G5_MEDIAN_BAND for v in degradation.values())
               and all(abs(v) < G5_GAP_BAND for v in closure.values()))
    out["conditions"] = {"BOTH_MEDIANS_BETTER_THAN_F1": both_better,
                         "ACCURACY_PRESERVED": preserved,
                         "GAP_CLOSED": closer, "SIMILAR_TO_F1": similar}
    if not out["finite"]:
        out["DECISION"] = "REGULARIZED_LATE_A1_UNSTABLE"
        out["RETRY_WITH_NEW_LAMBDA"] = "FORBIDDEN"
    elif out["ABSOLUTE_PASS"]:
        out["DECISION"] = "REGULARIZED_LATE_A1_VALID_CANDIDATE"
        out["STATUS"] = "LINE_STAGE_CANDIDATE"
        out["NEXT"] = "execution_replicate"
    elif both_better:
        out["DECISION"] = "REGULARIZED_LATE_A1_ACCURACY_SIGNAL"
        out["PROMOTION"] = "BLOCKED"
    elif preserved and closer:
        out["DECISION"] = "REGULARIZED_LATE_A1_GENERALIZATION_SIGNAL"
        out["PROMOTION"] = "BLOCKED"
        out["ARCHITECTURE_CANDIDATE"] = False
    elif similar:
        out["DECISION"] = "REGULARIZED_LATE_A1_NO_MATERIAL_EFFECT"
    elif closer:
        out["DECISION"] = "REGULARIZED_LATE_A1_OVERCONSTRAINED"
        out["SCOPE"] = "this fixed calibration only; lambda is not called large"
    else:
        out["DECISION"] = "REGULARIZED_LATE_A1_INCONCLUSIVE"
    out.setdefault("STATUS", "NOT_LOCKED")
    out["CAUSAL_LIMIT"] = ("explicit anchoring changes the accuracy and "
                           "generalization tradeoff; drift is not shown to "
                           "cause specialization")
    out["CIGM"] = "BLOCKED"
    return out


def build_plan(pool, lambda_sp):
    a1, model, _, reference = build_arm(True)
    plan = {"arms": list(ARMS), "factor": "L2_SP_PRETRAINED_WEIGHT_ANCHORING",
            "lambda_sp": lambda_sp, "lambda_sweep": False,
            "LAMBDA_OPTIMALITY_NOT_ESTABLISHED": True,
            "sp_eps": SP_EPS, "penalty_target": "W0_not_zero",
            "weight_decay_replaced": False,
            "marks": list(MARKS), "decision_step": DECISION_STEP,
            "decision_population": "D2_LINE_DEV512",
            "diagnostic_population": "D0_SEEN512",
            "per_role_marks": list(PER_ROLE_MARKS),
            "frames": len(pool), "batch": CAP.BATCH,
            "trainable_late_params": sum(
                p.numel() for p in a1.parameters_to_train()),
            "role_encoder_params": sum(p.numel() for p in model.encoder.parameters()),
            "head_params": sum(p.numel() for p in model.head.parameters()),
            "a1_lr": CAP.LR * A1_LR_SCALE, "decoder_lr": CAP.LR,
            "weight_decay": CAP.WD, "scheduler": None, "gradient_clipping": None,
            "post_f50_adapter": False, "extra_role_block": False,
            "low_rank_branch": False,
            "sp_reference_audit": reference.audit,
            "gate": {"angle_median": CAP.ANGLE_BUDGET_DEG,
                     "offset_median": CAP.OFFSET_BUDGET_CELL,
                     "angle_p90": CAP.SAFETY_ANGLE,
                     "offset_p90": CAP.SAFETY_OFFSET},
            "g3": {"max_degradation": G3_MAX_DEGRADATION,
                   "min_gap_closure": G3_MIN_GAP_CLOSURE},
            "g5": {"median_band": G5_MEDIAN_BAND, "gap_band": G5_GAP_BAND},
            "baseline_F1_full_precision":
                f1_reference()[str(DECISION_STEP)]["D2_LINE_DEV512"],
            "context_only": context_arms(), **CAP.provenance()}
    del a1, model
    torch.cuda.empty_cache()
    return plan


def train_arm(pool, marks, edges, populations, per_pass, lambda_sp):
    grid_theta, grid_rho, valid = DH.lattice()
    features = DH.hypothesis_features(grid_theta, grid_rho)
    a1, model, penalty, reference = build_arm(True)
    frozen = LATE.AdaptableA1(None).to(DEV)
    optimiser = optimiser_for(model, a1)
    history, task_log, total_log, sp_log, done = {}, [], [], [], 0
    for chunk, _ in V2.step_schedule(pool, max(marks), CAP.BATCH):
        model.train()
        pack = V2.load_pack(chunk)
        theta_c, rho_c, support = DH.batch_rows(pack, edges)
        target = DH.target_distribution(
            theta_c.reshape(-1), rho_c.reshape(-1), grid_theta, grid_rho, valid
        ).reshape(*theta_c.shape, -1)
        f50, _ = LATE.encoder_features(pack, a1)
        task = DH.cross_entropy(model(f50, features), target, support, valid)
        sp = penalty.penalty(a1)
        total = task + lambda_sp * sp
        optimiser.zero_grad(set_to_none=True)
        total.backward(); optimiser.step()
        task_log.append(float(task.detach()))
        sp_log.append(float(sp.detach()))
        total_log.append(float(total.detach()))
        done += 1
        if done in marks:
            reference.verify()
            entry = {"step": done, "diagnostic_only": done in DIAGNOSTIC_MARKS,
                     "finite": bool(np.isfinite(total_log[-1])),
                     "weight_drift": reference.drift(a1)}
            for label, series in (("task", task_log), ("total", total_log),
                                  ("sp", sp_log)):
                entry[f"{label}_mean_last250"] = float(np.mean(series[-250:]))
                entry[f"{label}_slope_last_pass"] = LONG.slope(series[-per_pass:])
            for label, indices in populations.items():
                entry[label] = LATE.evaluate(
                    indices, model, a1, edges, features, grid_theta, grid_rho,
                    valid, per_role=(label == "D2_LINE_DEV512"
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
            entry["feature_drift"] = LATE_feature_drift(
                V2.split_indices()[0][:PROBE_FRAMES], a1, frozen, model)
            drift = entry["weight_drift"]
            log(f"  {TAG} @{done:6d} task {entry['task_mean_last250']:.6f} "
                f"total {entry['total_mean_last250']:.6f} R_SP "
                f"{entry['sp_mean_last250']:.6f} | dW/W mean {drift['mean']:.4f} "
                f"max {drift['max']:.4f} | F50 drift "
                f"{entry['feature_drift']['f50_relative_l2']:.4f} cos "
                f"{entry['feature_drift']['f50_cosine']:.4f} | D2/D0 "
                f"{entry['generalization']['angle_ratio']:.3f}/"
                f"{entry['generalization']['offset_ratio']:.3f}")
            torch.save({"tag": TAG, "step": done, "model": model.state_dict(),
                        "late_a1": {name: parameter.detach().cpu()
                                    for name, parameter in late_parameters(a1)},
                        **CAP.provenance()},
                       CAP.checkpoint_path(f"DH_{TAG}", f"step_{done:05d}"))
            history[str(done)] = entry
    return history, model, a1, reference


@torch.no_grad()
def LATE_feature_drift(indices, a1, frozen, model):
    f50_drift, cosine, descriptor = [], [], []
    for start in range(0, len(indices), CAP.BATCH):
        pack = V2.load_pack(indices[start:start + CAP.BATCH])
        adapted, _ = LATE.encoder_features(pack, a1)
        base, _ = LATE.encoder_features(pack, frozen)
        f50_drift.append(float((adapted - base).flatten(1).norm(dim=1).mean()
                               / base.flatten(1).norm(dim=1).mean().clamp_min(SP_EPS)))
        cosine.append(float(nn.functional.cosine_similarity(
            adapted.flatten(1), base.flatten(1), dim=1).mean()))
        left, right = model.descriptors(adapted), model.descriptors(base)
        descriptor.append(float((left - right).flatten(1).norm(dim=1).mean()
                                / right.flatten(1).norm(dim=1).mean().clamp_min(SP_EPS)))
    return {"f50_relative_l2": float(np.mean(f50_drift)),
            "f50_cosine": float(np.mean(cosine)),
            "descriptor_relative_l2": float(np.mean(descriptor))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["calibrate", "plan", "step0",
                                            "gradient", "memory", "run"])
    arguments = parser.parse_args()
    import instance_edge_topology as IET
    edges = [tuple(e) for e in IET.build_topology()["edges"]]
    if not V2.sha_file(OUT / "line_internal_split.csv").startswith(V2.LINE_SPLIT_SHA):
        raise RuntimeError("HARD_BLOCKED: LINE split changed")
    for name in [F1_RESULT] + [v[0] for v in CONTEXT_RESULTS.values()]:
        if not (OUT / name).exists():
            raise RuntimeError(f"HARD_BLOCK: {name} is missing")
    pool = V2.split_indices()[0]

    if arguments.command == "calibrate":
        report = run_calibration()
        (OUT / "l2sp_calibration.json").write_text(
            json.dumps(report, indent=2, default=float))
        log(f"[calibrate] CE_ref {report['CE_ref']:.6f}  R_SP_ref "
            f"{report['R_SP_ref']}  lambda_sp {report['lambda_sp']}  LOCKED="
            f"{report['LAMBDA_LOCKED']}")
        if not report["LAMBDA_LOCKED"]:
            log(f"[calibrate] {report['BLOCK']}: {report['reason']}")
            raise RuntimeError(report["BLOCK"])
        return

    lambda_sp = locked_lambda()

    if arguments.command == "plan":
        plan = build_plan(pool, lambda_sp)
        (OUT / "l2sp_plan.json").write_text(json.dumps(plan, indent=2))
        log(f"[plan] lambda_sp {lambda_sp:.9g} | trainable late "
            f"{plan['trainable_late_params']:,} A1 LR {plan['a1_lr']} decoder "
            f"LR {plan['decoder_lr']} WD {plan['weight_decay']}")
        return

    if arguments.command == "step0":
        report = run_step0(edges)
        (OUT / "l2sp_step0.json").write_text(
            json.dumps(report, indent=2, default=float))
        g = report["gaps"]
        log(f"[step0] F50 {g['f50']:.3e} descriptor {g['descriptor']:.3e} "
            f"logits {g['logits']:.3e} task {g['task_loss']:.3e} | R_SP "
            f"{report['R_SP_at_step0']:.3e}  EQUIVALENT="
            f"{report['L2SP_STEP0_FUNCTION_EQUIVALENT']}")
        if not report["L2SP_STEP0_FUNCTION_EQUIVALENT"]:
            raise RuntimeError("L2SP_STEP0_FUNCTION_MISMATCH")
        return

    if arguments.command == "gradient":
        report = run_gradient_sanity(edges, lambda_sp)
        (OUT / "l2sp_gradient.json").write_text(
            json.dumps(report, indent=2, default=float))
        log(f"[gradient] R_SP at step0 {report['R_SP_at_step0']:.3e} (zero is "
            f"correct) | after one step, sp>0 {report['sp_grad_positive']} "
            f"task>0 {report['task_grad_positive']}  OK="
            f"{report['L2SP_GRADIENT_SANITY']}")
        if not report["L2SP_GRADIENT_SANITY"]:
            raise RuntimeError("L2SP_GRADIENT_SANITY_FAIL")
        return

    if arguments.command == "memory":
        report = run_memory(edges, lambda_sp)
        (OUT / "l2sp_memory.json").write_text(
            json.dumps(report, indent=2, default=float))
        log(f"[memory] batch {report['batch']} peak {report['peak_mib']:.1f} MiB "
            f"of {report['device_total_mib']:.0f} MiB  OK="
            f"{report['L2SP_BATCH8_MEMORY_OK']}")
        if not report["L2SP_BATCH8_MEMORY_OK"]:
            raise RuntimeError("L2SP_BATCH8_MEMORY_FAIL")
        return

    for name, key, label in (
            ("l2sp_step0.json", "L2SP_STEP0_FUNCTION_EQUIVALENT",
             "L2SP_STEP0_FUNCTION_MISMATCH"),
            ("l2sp_gradient.json", "L2SP_GRADIENT_SANITY",
             "L2SP_GRADIENT_SANITY_FAIL"),
            ("l2sp_memory.json", "L2SP_BATCH8_MEMORY_OK",
             "L2SP_BATCH8_MEMORY_FAIL")):
        path = OUT / name
        if not path.exists() or not json.loads(path.read_text())[key]:
            raise RuntimeError(f"{label}: preflight must pass first")
    plan = build_plan(pool, lambda_sp)
    per_pass = V2.steps_per_pass(pool, CAP.BATCH)
    history, _, _, _ = train_arm(pool, MARKS, edges, SCALE.populations(),
                                 per_pass, lambda_sp)
    report = {"plan": plan, "history": history, "verdict": judge(history),
              **CAP.provenance()}
    (OUT / "l2sp_result.json").write_text(
        json.dumps(report, indent=2, default=float))
    v = report["verdict"]
    log(f"[run] {v['DECISION']}  S1 {v['S1']['angle_median']:.6f}/"
        f"{v['S1']['offset_median']:.6f} p90 {v['S1']['angle_p90']:.6f}/"
        f"{v['S1']['offset_p90']:.6f}")
    log(f"[run] vs F1 angle {v['vs_F1']['angle_median']:+.2%} offset "
        f"{v['vs_F1']['offset_median']:+.2%} | conditions {v['conditions']}")


if __name__ == "__main__":
    main()
