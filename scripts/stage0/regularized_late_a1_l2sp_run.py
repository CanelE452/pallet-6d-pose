"""Fix the L2-SP coefficient from a fresh train-only state, then run S1.

`d543529` blocked on `SP_CALIBRATION_REFERENCE_MISSING`: F1's adapted late
weights at 1,703 steps were never written, so the planned
`lambda = CE_ref / R_SP_ref` had no state to evaluate at.  That block stays
recorded and is not amended.

This replaces the rule rather than reconstructing the state.

```
lambda_sp = ||g_task||_2 / ||g_sp||_2
```

evaluated once at a fresh task-only state after exactly one pass.  A term's
influence on the optimizer is its gradient, not its value, so gradients are what
get equalised.  It is a unit choice: `LAMBDA_OPTIMALITY_NOT_ESTABLISHED`.

The calibration state is `SP_CALIBRATION_STATE_1PASS` and is **not** the
historical F1 state -- the training path is not bit-reproducible, so a fresh run
reaches a different `W`.  Calibration runs under deterministic algorithms so the
coefficient is auditable; S1 runs under the default kernels so it stays
comparable with F1.  Those two regimes are deliberately different and serve
different roles.

At `W == W0` both the penalty and its gradient are zero, which is why the
coefficient cannot be taken at step 0.

The calibration path may read LINE_TRAIN, the canonical `W0` and its own fresh
state.  It may not evaluate any dev or sealed population, and a static test
enforces that.
"""
from __future__ import annotations

import argparse, ast, hashlib, importlib.util, json, os, pathlib, sys, time
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


SP = _load("L2SP_BLOCKED", "scripts/stage0/regularized_late_a1_full_adaptation.py")
LATE, LONG, DH = SP.LATE, SP.LONG, SP.DH
CAP, V2, SCALE = SP.CAP, SP.V2, SP.SCALE
OUT, DEV = SP.OUT, SP.DEV

CALIBRATION_STEPS = 1703
CALIBRATION_CHECKPOINT = "sp_lambda_calibration_step_01703"
DETERMINISTIC_WORKSPACE = LONG.DETERMINISTIC_WORKSPACE
REPEAT_TOLERANCE = 1e-8
SP_EPS = SP.SP_EPS
MARKS = SP.MARKS
DECISION_STEP = SP.DECISION_STEP
PER_ROLE_MARKS = SP.PER_ROLE_MARKS
DIAGNOSTIC_MARKS = SP.DIAGNOSTIC_MARKS
PROBE_FRAMES = SP.PROBE_FRAMES
STEP0_TOLERANCE = SP.STEP0_TOLERANCE
EXPECTED_TRAINABLE = SP.EXPECTED_TRAINABLE
FORBIDDEN_IN_CALIBRATION = ("D0_SEEN512", "D2_LINE_DEV512", "validation512",
                            "untouched", "eval56", "wood45", "final_test")
TAG = "s1"


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


class ModuleSPReference:
    """Per-module L2-SP against the canonical pretrained weights.

    Weight and bias share one denominator per convolution, so a small bias norm
    cannot dominate the ratio the way per-tensor normalisation would allow.
    """

    def __init__(self, a1):
        self.modules = []
        for index in sorted({int(name.split(".")[0])
                             for name, _ in SP.late_parameters(a1)}):
            conv = a1.vgg[index]
            entry = {"index": index,
                     "weight": conv.weight.detach().clone(),
                     "bias": (conv.bias.detach().clone()
                              if conv.bias is not None else None)}
            entry["weight"].requires_grad_(False)
            if entry["bias"] is not None:
                entry["bias"].requires_grad_(False)
            self.modules.append(entry)
        self.audit = [{"index": e["index"],
                       "weight_shape": list(e["weight"].shape),
                       "weight_norm": float(e["weight"].norm()),
                       "bias_norm": (float(e["bias"].norm())
                                     if e["bias"] is not None else None),
                       "sha256": self._digest(e)} for e in self.modules]

    @staticmethod
    def _digest(entry):
        digest = hashlib.sha256(entry["weight"].cpu().numpy().tobytes())
        if entry["bias"] is not None:
            digest.update(entry["bias"].cpu().numpy().tobytes())
        return digest.hexdigest()[:16]

    def verify(self):
        for entry, row in zip(self.modules, self.audit):
            if self._digest(entry) != row["sha256"]:
                raise RuntimeError("HARD_BLOCK: SP reference mutated at "
                                   f"index {entry['index']}")
        return True

    def penalty(self, a1):
        terms = []
        for entry in self.modules:
            conv = a1.vgg[entry["index"]]
            numerator = (conv.weight - entry["weight"]).pow(2).sum()
            denominator = entry["weight"].pow(2).sum()
            if entry["bias"] is not None:
                numerator = numerator + (conv.bias - entry["bias"]).pow(2).sum()
                denominator = denominator + entry["bias"].pow(2).sum()
            terms.append(numerator / (denominator + SP_EPS))
        return torch.stack(terms).mean()

    @torch.no_grad()
    def drift(self, a1):
        rows = {}
        for entry in self.modules:
            conv = a1.vgg[entry["index"]]
            numerator = (conv.weight - entry["weight"]).pow(2).sum()
            denominator = entry["weight"].pow(2).sum()
            if entry["bias"] is not None:
                numerator = numerator + (conv.bias - entry["bias"]).pow(2).sum()
                denominator = denominator + entry["bias"].pow(2).sum()
            rows[str(entry["index"])] = float(
                (numerator / denominator.clamp_min(SP_EPS)).sqrt())
        values = list(rows.values())
        return {"per_module": rows, "mean": float(np.mean(values)),
                "max": float(np.max(values))}


def flat_gradient(a1):
    pieces = [p.grad.detach().reshape(-1) for _, p in SP.late_parameters(a1)
              if p.grad is not None]
    return torch.cat(pieces) if pieces else torch.zeros(1, device=DEV)


def build(with_reference=True):
    a1 = LATE.AdaptableA1(SP.FIRST_TRAINABLE_INDEX).to(DEV)
    trainable = sum(p.numel() for p in a1.parameters_to_train())
    if trainable != EXPECTED_TRAINABLE:
        raise RuntimeError(f"L2SP_TRAINABILITY_MISMATCH: {trainable}")
    model = DH.DirectHoughModel().to(DEV)
    return a1, model, (ModuleSPReference(a1) if with_reference else None)


def calibration_state(edges, pool):
    """One pass of pure task training.  No SP term, no evaluation, no dev."""
    a1, model, reference = build()
    optimiser = SP.optimiser_for(model, a1)
    grid_theta, grid_rho, valid = DH.lattice()
    features = DH.hypothesis_features(grid_theta, grid_rho)
    losses = []
    for chunk, _ in V2.step_schedule(pool, CALIBRATION_STEPS, CAP.BATCH):
        model.train()
        pack = V2.load_pack(chunk)
        theta_c, rho_c, support = DH.batch_rows(pack, edges)
        target = DH.target_distribution(
            theta_c.reshape(-1), rho_c.reshape(-1), grid_theta, grid_rho, valid
        ).reshape(*theta_c.shape, -1)
        f50, _ = LATE.encoder_features(pack, a1)
        loss = DH.cross_entropy(model(f50, features), target, support, valid)
        optimiser.zero_grad(set_to_none=True)
        loss.backward(); optimiser.step()
        losses.append(float(loss.detach()))
    return a1, model, reference, optimiser, losses


def accumulate_task_gradient(a1, model, edges, pool):
    """Gradient of the mean task loss over all of LINE_TRAIN.

    Accumulated in batches of eight with no optimizer step, weighted by frames
    so the result is the mean over samples rather than over batches.
    """
    grid_theta, grid_rho, valid = DH.lattice()
    features = DH.hypothesis_features(grid_theta, grid_rho)
    model.eval()
    for _, parameter in SP.late_parameters(a1):
        if parameter.grad is not None:
            parameter.grad = None
    total_frames, weighted_loss, batches = 0, 0.0, 0
    for start in range(0, len(pool), CAP.BATCH):
        chunk = pool[start:start + CAP.BATCH]
        if len(chunk) < 2:
            continue
        total_frames += len(chunk)
    seen = 0
    for start in range(0, len(pool), CAP.BATCH):
        chunk = pool[start:start + CAP.BATCH]
        if len(chunk) < 2:
            continue
        pack = V2.load_pack(chunk)
        theta_c, rho_c, support = DH.batch_rows(pack, edges)
        target = DH.target_distribution(
            theta_c.reshape(-1), rho_c.reshape(-1), grid_theta, grid_rho, valid
        ).reshape(*theta_c.shape, -1)
        f50, _ = LATE.encoder_features(pack, a1)
        loss = DH.cross_entropy(model(f50, features), target, support, valid)
        (loss * (len(chunk) / total_frames)).backward()
        weighted_loss += float(loss.detach()) * len(chunk) / total_frames
        seen += len(chunk); batches += 1
    return flat_gradient(a1).clone(), weighted_loss, seen, batches


def sp_gradient(a1, reference):
    for _, parameter in SP.late_parameters(a1):
        if parameter.grad is not None:
            parameter.grad = None
    value = reference.penalty(a1)
    value.backward()
    return flat_gradient(a1).clone(), float(value.detach())


def checkpoint_path():
    return CAP.checkpoint_path("DH_l2sp_calibration", CALIBRATION_CHECKPOINT)


def run_calibration(edges, pool):
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != DETERMINISTIC_WORKSPACE:
        raise RuntimeError("calibration needs CUBLAS_WORKSPACE_CONFIG="
                           f"{DETERMINISTIC_WORKSPACE} in the environment")
    torch.use_deterministic_algorithms(True)
    try:
        a1, model, reference, optimiser, losses = calibration_state(edges, pool)
        reference.verify()
        task_grad, ce_fulltrain, frames, batches = accumulate_task_gradient(
            a1, model, edges, pool)
        unit_sp_grad, r_sp = sp_gradient(a1, reference)
        task_norm = float(task_grad.norm())
        sp_norm = float(unit_sp_grad.norm())
        cosine = float((task_grad * unit_sp_grad).sum()
                       / max(task_norm * sp_norm, SP_EPS))
        state = {"tag": "sp_lambda_calibration", "step": CALIBRATION_STEPS,
                 "late_a1": {name: parameter.detach().cpu()
                             for name, parameter in SP.late_parameters(a1)},
                 "decoder": model.state_dict(),
                 "optimizer": optimiser.state_dict(),
                 "seed": CAP.SEED, "deterministic": True,
                 "schedule": {"frames": len(pool), "batch": CAP.BATCH,
                              "steps": CALIBRATION_STEPS},
                 "purpose": "coefficient audit only, never an S1 initialisation",
                 **CAP.provenance()}
        path = checkpoint_path()
        torch.save(state, path)
    finally:
        torch.use_deterministic_algorithms(False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    report = {"state_name": "SP_CALIBRATION_STATE_1PASS",
              "is_historical_f1": False,
              "calibration_steps": CALIBRATION_STEPS,
              "task_grad_norm": task_norm, "sp_unit_grad_norm": sp_norm,
              "gradient_cosine": cosine, "R_SP_cal": r_sp,
              "CE_cal_fulltrain": ce_fulltrain,
              "frames_accumulated": frames, "batches_accumulated": batches,
              "train_loss_mean_last250": float(np.mean(losses[-250:])),
              "checkpoint": str(path), "checkpoint_sha256": digest,
              "reference_audit": reference.audit,
              "deterministic": True,
              "LAMBDA_OPTIMALITY_NOT_ESTABLISHED": True,
              "LAMBDA_SELECTED_WITH_DEV": False, "LAMBDA_SWEEP": False,
              "rule": "ONE_PASS_GRADIENT_BALANCED_L2SP"}
    valid = (task_norm > 0.0 and sp_norm > 0.0
             and np.isfinite(task_norm) and np.isfinite(sp_norm))
    report["lambda_sp"] = task_norm / sp_norm if valid else None
    report["SP_GRADIENT_CALIBRATION_VALID"] = bool(
        valid and report["lambda_sp"] is not None
        and report["lambda_sp"] > 0.0 and np.isfinite(report["lambda_sp"]))
    del a1, model
    torch.cuda.empty_cache()
    return report


def leakage_guard():
    """Static check that the calibration path never touches a held-out set."""
    source = pathlib.Path(__file__).read_text("utf-8")
    tree = ast.parse(source)
    watched = {"run_calibration", "calibration_state",
               "accumulate_task_gradient", "sp_gradient"}
    hits = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in watched:
            found = {n.value for n in ast.walk(node)
                     if isinstance(n, ast.Constant) and isinstance(n.value, str)}
            found |= {n.attr for n in ast.walk(node)
                      if isinstance(n, ast.Attribute)}
            found |= {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
            bad = sorted(t for t in FORBIDDEN_IN_CALIBRATION
                         if any(t in str(f) for f in found))
            if bad:
                hits[node.name] = bad
    return {"functions_checked": sorted(watched), "violations": hits,
            "populations_call": "populations" in {
                n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)},
            "CALIBRATION_LEAKAGE_GUARD_CLEAN": not hits}


def locked_coefficient():
    path = OUT / "l2sp_coefficient_lock.json"
    if not path.exists():
        raise RuntimeError("SP_COEFFICIENT_NOT_LOCKED: run calibrate then lock")
    blob = json.loads(path.read_text())
    if not blob.get("SP_COEFFICIENT_LOCKED"):
        raise RuntimeError(blob.get("BLOCK", "SP_COEFFICIENT_NOT_LOCKED"))
    stored = pathlib.Path(blob["checkpoint"])
    if not stored.exists():
        raise RuntimeError("SP_COEFFICIENT_NOT_LOCKED: calibration checkpoint gone")
    digest = hashlib.sha256(stored.read_bytes()).hexdigest()
    if digest != blob["checkpoint_sha256"]:
        raise RuntimeError("SP_COEFFICIENT_NOT_LOCKED: checkpoint sha mismatch")
    return blob["lambda_sp"]


def run_step0(edges, lambda_sp):
    a1_s0, model_s0, _ = build(False)
    a1_s1, model_s1, reference = build(True)
    model_s1.load_state_dict(model_s0.state_dict())
    grid_theta, grid_rho, valid = DH.lattice()
    features = DH.hypothesis_features(grid_theta, grid_rho)
    indices = V2.split_indices()[0][:PROBE_FRAMES]
    gaps = {"f50": 0.0, "descriptor": 0.0, "logits": 0.0, "task_loss": 0.0}
    sp_value = float(reference.penalty(a1_s1).detach())
    with torch.no_grad():
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
            logits_s0, logits_s1 = model_s0(left, features), model_s1(right, features)
            gaps["logits"] = max(gaps["logits"],
                                 float((logits_s0 - logits_s1).abs().max()))
            gaps["task_loss"] = max(gaps["task_loss"], abs(
                float(DH.cross_entropy(logits_s0, target, support, valid))
                - float(DH.cross_entropy(logits_s1, target, support, valid))))
    report = {"frames": PROBE_FRAMES, "tolerance": STEP0_TOLERANCE, "gaps": gaps,
              "R_SP_at_step0": sp_value, "lambda_sp": lambda_sp,
              "reference_audit": reference.audit,
              "reference_verified": reference.verify()}
    report["L2SP_STEP0_FUNCTION_EQUIVALENT"] = bool(
        all(v <= STEP0_TOLERANCE for v in gaps.values())
        and sp_value <= STEP0_TOLERANCE)
    del a1_s0, a1_s1
    return report


def run_gradient_sanity(edges, lambda_sp):
    a1, model, reference = build(True)
    grid_theta, grid_rho, valid = DH.lattice()
    features = DH.hypothesis_features(grid_theta, grid_rho)
    optimiser = SP.optimiser_for(model, a1)
    pack = V2.load_pack(V2.split_indices()[0][:CAP.BATCH])
    theta_c, rho_c, support = DH.batch_rows(pack, edges)
    target = DH.target_distribution(
        theta_c.reshape(-1), rho_c.reshape(-1), grid_theta, grid_rho, valid
    ).reshape(*theta_c.shape, -1)

    def task():
        f50, _ = LATE.encoder_features(pack, a1)
        return DH.cross_entropy(model(f50, features), target, support, valid)

    at_zero = float(reference.penalty(a1).detach())
    optimiser.zero_grad(set_to_none=True); task().backward(); optimiser.step()
    optimiser.zero_grad(set_to_none=True); task().backward()
    task_grad = flat_gradient(a1).clone()
    sp_grad, sp_value = sp_gradient(a1, reference)
    task_norm, sp_norm = float(task_grad.norm()), float(sp_grad.norm())
    report = {"R_SP_at_step0": at_zero, "R_SP_after_one_step": sp_value,
              "task_grad_norm": task_norm, "sp_grad_norm": sp_norm,
              "scaled_sp_grad_norm": lambda_sp * sp_norm,
              "gradient_cosine": float((task_grad * sp_grad).sum()
                                       / max(task_norm * sp_norm, SP_EPS)),
              "lambda_sp": lambda_sp}
    report["L2SP_GRADIENT_SANITY"] = bool(
        at_zero <= STEP0_TOLERANCE and sp_norm > 0.0 and task_norm > 0.0)
    del a1, model
    return report


def train_s1(pool, marks, edges, populations, per_pass, lambda_sp):
    grid_theta, grid_rho, valid = DH.lattice()
    features = DH.hypothesis_features(grid_theta, grid_rho)
    a1, model, reference = build(True)
    frozen = LATE.AdaptableA1(None).to(DEV)
    optimiser = SP.optimiser_for(model, a1)
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
        penalty = reference.penalty(a1)
        total = task + lambda_sp * penalty
        optimiser.zero_grad(set_to_none=True)
        total.backward(); optimiser.step()
        task_log.append(float(task.detach()))
        sp_log.append(float(penalty.detach()))
        total_log.append(float(total.detach()))
        done += 1
        if done in marks:
            reference.verify()
            entry = {"step": done, "diagnostic_only": done in DIAGNOSTIC_MARKS,
                     "finite": bool(np.isfinite(total_log[-1])),
                     "weight_drift": reference.drift(a1),
                     "lambda_sp": lambda_sp,
                     "scaled_sp_mean_last250": lambda_sp * float(
                         np.mean(sp_log[-250:]))}
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
            entry["feature_drift"] = SP.LATE_feature_drift(
                V2.split_indices()[0][:PROBE_FRAMES], a1, frozen, model)
            drift = entry["weight_drift"]
            log(f"  {TAG} @{done:6d} task {entry['task_mean_last250']:.6f} R_SP "
                f"{entry['sp_mean_last250']:.6f} lR_SP "
                f"{entry['scaled_sp_mean_last250']:.6f} total "
                f"{entry['total_mean_last250']:.6f} | dW/W mean {drift['mean']:.4f}"
                f" max {drift['max']:.4f} | F50 "
                f"{entry['feature_drift']['f50_relative_l2']:.4f} cos "
                f"{entry['feature_drift']['f50_cosine']:.4f} | D2/D0 "
                f"{entry['generalization']['angle_ratio']:.3f}/"
                f"{entry['generalization']['offset_ratio']:.3f}")
            torch.save({"tag": TAG, "step": done, "model": model.state_dict(),
                        "late_a1": {name: parameter.detach().cpu()
                                    for name, parameter in SP.late_parameters(a1)},
                        "lambda_sp": lambda_sp, **CAP.provenance()},
                       CAP.checkpoint_path(f"DH_{TAG}", f"step_{done:05d}"))
            history[str(done)] = entry
    return history, model, a1, reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["calibrate", "lock", "plan",
                                            "step0", "gradient", "memory",
                                            "run"])
    arguments = parser.parse_args()
    import instance_edge_topology as IET
    edges = [tuple(e) for e in IET.build_topology()["edges"]]
    if not V2.sha_file(OUT / "line_internal_split.csv").startswith(V2.LINE_SPLIT_SHA):
        raise RuntimeError("HARD_BLOCKED: LINE split changed")
    pool = V2.split_indices()[0]

    if arguments.command == "calibrate":
        guard = leakage_guard()
        if not guard["CALIBRATION_LEAKAGE_GUARD_CLEAN"]:
            raise RuntimeError(f"CALIBRATION_LEAKAGE: {guard['violations']}")
        report = run_calibration(edges, pool)
        report["leakage_guard"] = guard
        existing = OUT / "l2sp_coefficient_calibration.json"
        if existing.exists():
            first = json.loads(existing.read_text())
            relative = abs(report["lambda_sp"] - first["lambda_sp"]) / max(
                abs(first["lambda_sp"]), SP_EPS)
            report["repeat"] = {
                "first_lambda": first["lambda_sp"],
                "second_lambda": report["lambda_sp"],
                "relative_difference": relative,
                "task_grad_equal": first["task_grad_norm"] == report["task_grad_norm"],
                "sp_grad_equal": first["sp_unit_grad_norm"] == report["sp_unit_grad_norm"],
                "checkpoint_sha_equal":
                    first["checkpoint_sha256"] == report["checkpoint_sha256"]}
            report["repeat"]["SP_COEFFICIENT_REPRODUCIBLE"] = bool(
                relative <= REPEAT_TOLERANCE
                and report["repeat"]["task_grad_equal"]
                and report["repeat"]["sp_grad_equal"]
                and report["repeat"]["checkpoint_sha_equal"])
            (OUT / "l2sp_coefficient_repeat.json").write_text(
                json.dumps(report, indent=2, default=float))
            log(f"[calibrate] repeat lambda {report['lambda_sp']:.12g} vs "
                f"{first['lambda_sp']:.12g}  rel {relative:.3e}  "
                f"REPRODUCIBLE={report['repeat']['SP_COEFFICIENT_REPRODUCIBLE']}")
            if not report["repeat"]["SP_COEFFICIENT_REPRODUCIBLE"]:
                raise RuntimeError("SP_COEFFICIENT_NOT_REPRODUCIBLE")
            return
        existing.write_text(json.dumps(report, indent=2, default=float))
        log(f"[calibrate] ||g_task|| {report['task_grad_norm']:.9g}  ||g_sp|| "
            f"{report['sp_unit_grad_norm']:.9g}  cos "
            f"{report['gradient_cosine']:+.6f}  R_SP {report['R_SP_cal']:.9g}")
        log(f"[calibrate] lambda_sp {report['lambda_sp']:.12g}  VALID="
            f"{report['SP_GRADIENT_CALIBRATION_VALID']}  frames "
            f"{report['frames_accumulated']}  guard clean "
            f"{guard['CALIBRATION_LEAKAGE_GUARD_CLEAN']}")
        if not report["SP_GRADIENT_CALIBRATION_VALID"]:
            raise RuntimeError("SP_GRADIENT_CALIBRATION_INVALID")
        return

    if arguments.command == "lock":
        first = OUT / "l2sp_coefficient_calibration.json"
        repeat = OUT / "l2sp_coefficient_repeat.json"
        if not first.exists() or not repeat.exists():
            raise RuntimeError("SP_COEFFICIENT_NOT_LOCKED: calibrate twice first")
        one, two = json.loads(first.read_text()), json.loads(repeat.read_text())
        locked = {"lambda_sp": one["lambda_sp"],
                  "rule": one["rule"], "state_name": one["state_name"],
                  "is_historical_f1": False,
                  "task_grad_norm": one["task_grad_norm"],
                  "sp_unit_grad_norm": one["sp_unit_grad_norm"],
                  "gradient_cosine": one["gradient_cosine"],
                  "R_SP_cal": one["R_SP_cal"],
                  "CE_cal_fulltrain": one["CE_cal_fulltrain"],
                  "checkpoint": one["checkpoint"],
                  "checkpoint_sha256": one["checkpoint_sha256"],
                  "repeat": two["repeat"],
                  "leakage_guard": one["leakage_guard"],
                  "LAMBDA_OPTIMALITY_NOT_ESTABLISHED": True,
                  "LAMBDA_SELECTED_WITH_DEV": False, "LAMBDA_SWEEP": False,
                  "CALIBRATION_NUMERICAL_REGIME": "deterministic",
                  "ACTUAL_TRAINING_NUMERICAL_REGIME": "default",
                  "supersedes": "SP_CALIBRATION_REFERENCE_MISSING",
                  **CAP.provenance()}
        locked["SP_COEFFICIENT_LOCKED"] = bool(
            one["SP_GRADIENT_CALIBRATION_VALID"]
            and two["repeat"]["SP_COEFFICIENT_REPRODUCIBLE"]
            and one["leakage_guard"]["CALIBRATION_LEAKAGE_GUARD_CLEAN"])
        (OUT / "l2sp_coefficient_lock.json").write_text(
            json.dumps(locked, indent=2, default=float))
        log(f"[lock] lambda_sp {locked['lambda_sp']:.12g}  LOCKED="
            f"{locked['SP_COEFFICIENT_LOCKED']}")
        if not locked["SP_COEFFICIENT_LOCKED"]:
            raise RuntimeError("SP_COEFFICIENT_NOT_LOCKED")
        return

    lambda_sp = locked_coefficient()

    if arguments.command == "plan":
        plan = SP.build_plan(pool, lambda_sp)
        plan["coefficient_rule"] = "ONE_PASS_GRADIENT_BALANCED_L2SP"
        plan["sp_normalisation"] = "per_conv_module_weight_and_bias_shared"
        (OUT / "l2sp_plan.json").write_text(json.dumps(plan, indent=2, default=float))
        log(f"[plan] lambda_sp {lambda_sp:.12g} | trainable late "
            f"{plan['trainable_late_params']:,} A1 LR {plan['a1_lr']} decoder LR "
            f"{plan['decoder_lr']} WD {plan['weight_decay']}")
        return

    if arguments.command == "step0":
        report = run_step0(edges, lambda_sp)
        (OUT / "l2sp_step0.json").write_text(json.dumps(report, indent=2, default=float))
        g = report["gaps"]
        log(f"[step0] F50 {g['f50']:.3e} descriptor {g['descriptor']:.3e} logits "
            f"{g['logits']:.3e} task {g['task_loss']:.3e} | R_SP "
            f"{report['R_SP_at_step0']:.3e}  EQUIVALENT="
            f"{report['L2SP_STEP0_FUNCTION_EQUIVALENT']}")
        if not report["L2SP_STEP0_FUNCTION_EQUIVALENT"]:
            raise RuntimeError("L2SP_STEP0_FUNCTION_MISMATCH")
        return

    if arguments.command == "gradient":
        report = run_gradient_sanity(edges, lambda_sp)
        (OUT / "l2sp_gradient.json").write_text(json.dumps(report, indent=2, default=float))
        log(f"[gradient] R_SP step0 {report['R_SP_at_step0']:.3e} (zero is correct)"
            f" | after one step ||g_task|| {report['task_grad_norm']:.6g} "
            f"||l*g_sp|| {report['scaled_sp_grad_norm']:.6g} cos "
            f"{report['gradient_cosine']:+.6f}  OK={report['L2SP_GRADIENT_SANITY']}")
        if not report["L2SP_GRADIENT_SANITY"]:
            raise RuntimeError("L2SP_GRADIENT_SANITY_FAIL")
        return

    if arguments.command == "memory":
        report = SP.run_memory(edges, lambda_sp)
        (OUT / "l2sp_memory.json").write_text(json.dumps(report, indent=2, default=float))
        log(f"[memory] batch {report['batch']} peak {report['peak_mib']:.1f} MiB of "
            f"{report['device_total_mib']:.0f} MiB  OK={report['L2SP_BATCH8_MEMORY_OK']}")
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
    plan = SP.build_plan(pool, lambda_sp)
    per_pass = V2.steps_per_pass(pool, CAP.BATCH)
    history, _, _, _ = train_s1(pool, MARKS, edges, SCALE.populations(),
                                per_pass, lambda_sp)
    report = {"plan": plan, "history": history, "verdict": SP.judge(history),
              "coefficient": json.loads(
                  (OUT / "l2sp_coefficient_lock.json").read_text()),
              **CAP.provenance()}
    (OUT / "l2sp_result.json").write_text(json.dumps(report, indent=2, default=float))
    v = report["verdict"]
    log(f"[run] {v['DECISION']}  S1 {v['S1']['angle_median']:.6f}/"
        f"{v['S1']['offset_median']:.6f} p90 {v['S1']['angle_p90']:.6f}/"
        f"{v['S1']['offset_p90']:.6f}")
    log(f"[run] vs F1 angle {v['vs_F1']['angle_median']:+.2%} offset "
        f"{v['vs_F1']['offset_median']:+.2%} | conditions {v['conditions']}")


if __name__ == "__main__":
    main()
