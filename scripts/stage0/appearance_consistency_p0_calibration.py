"""Fix the consistency coefficient at a state where the term actually exists.

`appearance_lambda_fresh_init_degenerate.json` keeps the first attempt and is
not deleted.  It reproduced to the bit -- 613163.9148076542 twice, relative
difference zero -- and it is still unusable, for a reason that is neither
reproducibility, formulation nor leakage: at a fresh initialisation the
predictor is near-uniform, so over the whole of LINE_TRAIN the consistency term
is 1.3559e-10 and its gradient 1.0315e-09.  The gradient being balanced against
has effectively not arisen yet.  That is a statement about the calibration
state, not about the size of the coefficient.

So the state moves, once, to the regime the screen actually trains in:

```
P0_AUG_ONLY_CALIBRATION_STATE_1PASS
    the F1 architecture, two photometric views per sample from the same fixed
    policy and the same local-RNG mechanism, supervised by
    L_sup = 0.5 CE(a) + 0.5 CE(b), consistency term absent,
    fresh init, 0 -> 1,703 optimizer steps
```

It is not `HISTORICAL_F1_1703`, not an F1 reconstruction, and not a clean
task-only trajectory: it is P0's own objective, so the gradients being compared
come from the regime P0 and P1 are trained in.

The coefficient is then `||g_sup|| / ||g_cons||` over all of LINE_TRAIN at that
frozen state, with no optimizer step and no held-out population read.  A
validity audit runs first and gates only on what must be true for the ratio to
mean anything -- finite, both terms and both gradients strictly positive,
distinct views producing distinct logits, identical views producing exactly
zero.  No entropy or peak threshold is invented, because a "sharp enough"
criterion chosen now would be a criterion chosen after seeing the first attempt
fail.

If this state is also degenerate, the screen stops.  There is no second move to
two passes or five: that would be searching for a state that produces an
agreeable number.
"""
from __future__ import annotations

import argparse, ast, hashlib, importlib.util, json, os, pathlib, sys, time
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


AC = _load("APPEARANCE_BASE", "scripts/stage0/appearance_consistency_f1_screen.py")
LATE, LONG, DH = AC.LATE, AC.LONG, AC.DH
CAP, V2, SCALE = AC.CAP, AC.V2, AC.SCALE
OUT, DEV = AC.OUT, AC.DEV

CALIBRATION_STEPS = 1703
STATE_NAME = "P0_AUG_ONLY_CALIBRATION_STATE_1PASS"
CHECKPOINT = "appearance_lambda_calibration_p0_01703"
DEGENERATE_RECORD = "appearance_lambda_fresh_init_degenerate.json"
FIRST = "appearance_consistency_lambda_p0_1pass.json"
REPEAT = "appearance_consistency_lambda_p0_1pass_repeat.json"
LOCK = "appearance_consistency_lambda_p0_lock.json"
REPEAT_TOLERANCE = 1e-8
EPS = AC.EPS
DETERMINISTIC_WORKSPACE = AC.DETERMINISTIC_WORKSPACE
FORBIDDEN = AC.FORBIDDEN_IN_TRAINING


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def one_pass_state(edges, pool):
    """P0's objective for exactly one pass.  No consistency term, no eval."""
    a1, model = AC.build()
    optimiser = AC.optimiser_for(model, a1)
    grid_theta, grid_rho, valid = DH.lattice()
    features = DH.hypothesis_features(grid_theta, grid_rho)
    losses = []
    for chunk, _ in V2.step_schedule(pool, CALIBRATION_STEPS, CAP.BATCH):
        model.train()
        pack = V2.load_pack(chunk)
        views = AC.two_views(pack, len(losses))
        _, ce, _, _ = AC.forward_views(pack, views, a1, model, edges, features,
                                       grid_theta, grid_rho, valid)
        supervised = 0.5 * ce[0] + 0.5 * ce[1]
        optimiser.zero_grad(set_to_none=True)
        supervised.backward(); optimiser.step()
        losses.append(float(supervised.detach()))
    return a1, model, optimiser, losses


def validity_audit(a1, model, edges, pool):
    """Only what must hold for a gradient ratio to mean anything."""
    grid_theta, grid_rho, valid = DH.lattice()
    features = DH.hypothesis_features(grid_theta, grid_rho)
    pack = V2.load_pack(pool[:CAP.BATCH])
    view_a, view_b = AC.two_views(pack, 0)
    _, _, support = DH.batch_rows(pack, edges)
    with torch.no_grad():
        scores_a = model(a1(view_a)[0], features)
        scores_b = model(a1(view_b)[0], features)
        identical = float(AC.js_divergence(scores_a, scores_a, support, valid))
        distinct = float(AC.js_divergence(scores_a, scores_b, support, valid))
        logit_gap = float((scores_a - scores_b).abs().max())
        masked = scores_a.masked_fill(~valid[None, None], -1e9)
        probability = torch.softmax(masked, -1)
        entropy = float(-(probability.clamp_min(EPS).log()
                          * probability).sum(-1).mean())
        peak = float(probability.max())
    bins = int(valid.sum())
    report = {"valid_bins": bins, "uniform_probability": 1.0 / bins,
              "max_probability": peak, "mean_entropy": entropy,
              "uniform_entropy": float(np.log(bins)),
              "js_identical_views": identical, "js_distinct_views": distinct,
              "logit_max_abs_between_views": logit_gap}
    report["ONE_PASS_STATE_VALID"] = bool(
        np.isfinite(entropy) and np.isfinite(peak)
        and identical == 0.0 and distinct > 0.0 and logit_gap > 0.0)
    return report


def accumulate(a1, model, edges, pool):
    """Both gradients over all of LINE_TRAIN, one forward and two backwards."""
    grid_theta, grid_rho, valid = DH.lattice()
    features = DH.hypothesis_features(grid_theta, grid_rho)
    total = sum(len(pool[s:s + CAP.BATCH])
                for s in range(0, len(pool), CAP.BATCH)
                if len(pool[s:s + CAP.BATCH]) >= 2)
    buffers = {"sup": None, "cons": None}
    sums = {"sup": 0.0, "cons": 0.0}
    seen = 0
    model.eval()
    for start in range(0, len(pool), CAP.BATCH):
        chunk = pool[start:start + CAP.BATCH]
        if len(chunk) < 2:
            continue
        weight = len(chunk) / total
        pack = V2.load_pack(chunk)
        views = AC.two_views(pack, 0)
        scores, ce, support, _ = AC.forward_views(
            pack, views, a1, model, edges, features, grid_theta, grid_rho, valid)
        terms = {"sup": 0.5 * ce[0] + 0.5 * ce[1],
                 "cons": AC.js_divergence(scores[0], scores[1], support, valid)}
        for key in ("sup", "cons"):
            for parameter in a1.parameters_to_train():
                parameter.grad = None
            terms[key].backward(retain_graph=(key == "sup"))
            piece = AC.flat_gradient(a1) * weight
            buffers[key] = piece.clone() if buffers[key] is None else buffers[key] + piece
            sums[key] += float(terms[key].detach()) * weight
        seen += len(chunk)
    norms = {key: float(value.norm()) for key, value in buffers.items()}
    cosine = float((buffers["sup"] * buffers["cons"]).sum()
                   / max(norms["sup"] * norms["cons"], EPS))
    return norms, sums, cosine, seen


def leakage_guard():
    source = pathlib.Path(__file__).read_text("utf-8")
    tree = ast.parse(source)
    watched = {"one_pass_state", "accumulate", "validity_audit", "run"}
    hits = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in watched:
            found = {n.value for n in ast.walk(node)
                     if isinstance(n, ast.Constant) and isinstance(n.value, str)}
            bad = sorted(t for t in FORBIDDEN
                         if any(t in str(f) for f in found))
            if bad:
                hits[node.name] = bad
    inner = AC.leakage_guard()
    return {"functions_checked": sorted(watched), "violations": hits,
            "screen_guard": inner,
            "CALIBRATION_LEAKAGE_GUARD_CLEAN":
                not hits and inner["TRAINING_LEAKAGE_GUARD_CLEAN"]}


def run(edges, pool):
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != DETERMINISTIC_WORKSPACE:
        raise RuntimeError("calibration needs CUBLAS_WORKSPACE_CONFIG="
                           f"{DETERMINISTIC_WORKSPACE}")
    torch.use_deterministic_algorithms(True)
    try:
        a1, model, optimiser, losses = one_pass_state(edges, pool)
        audit = validity_audit(a1, model, edges, pool)
        norms, sums, cosine, seen = accumulate(a1, model, edges, pool)
        state = {"tag": STATE_NAME, "step": CALIBRATION_STEPS,
                 "late_a1": {name: parameter.detach().cpu() for name, parameter
                             in AC.late_parameters(a1)},
                 "decoder": model.state_dict(),
                 "optimizer": optimiser.state_dict(),
                 "augmentation": {"policy": AC.POLICY,
                                  "source": AC.AUG_SOURCE,
                                  "seed_rule": "sha256(seed|frame|step|view)"},
                 "deterministic": True, "seed": CAP.SEED,
                 "purpose": "coefficient audit only, never a P0 or P1 "
                            "initialisation",
                 **CAP.provenance()}
        path = CAP.checkpoint_path("DH_appearance_calibration", CHECKPOINT)
        torch.save(state, path)
    finally:
        torch.use_deterministic_algorithms(False)
    degenerate = json.loads((OUT / DEGENERATE_RECORD).read_text())
    report = {"state": STATE_NAME, "is_historical": False,
              "is_model_selection": False,
              "calibration_steps": CALIBRATION_STEPS,
              "objective": "P0_AUG_ONLY", "consistency_in_trajectory": False,
              "frames_accumulated": seen,
              "sup_grad_norm": norms["sup"], "cons_grad_norm": norms["cons"],
              "gradient_cosine": cosine,
              "L_sup_fulltrain": sums["sup"], "L_cons_fulltrain": sums["cons"],
              "train_loss_mean_last250": float(np.mean(losses[-250:])),
              "validity": audit, "checkpoint": str(path),
              "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "augmentation": {"policy": AC.POLICY, "source": AC.AUG_SOURCE},
              "deterministic": True,
              "LAMBDA_OPTIMALITY_NOT_ESTABLISHED": True,
              "LAMBDA_SELECTED_WITH_DEV": False, "LAMBDA_SWEEP": False}
    ok = (norms["sup"] > 0 and norms["cons"] > 0
          and np.isfinite(norms["sup"]) and np.isfinite(norms["cons"])
          and sums["cons"] > 0 and audit["ONE_PASS_STATE_VALID"])
    report["lambda_cons"] = norms["sup"] / norms["cons"] if ok else None
    report["ONE_PASS_CALIBRATION_VALID"] = bool(
        ok and report["lambda_cons"] and np.isfinite(report["lambda_cons"]))
    report["against_fresh_init"] = {
        "fresh_lambda": degenerate["lambda_cons"],
        "fresh_cons_grad_norm": degenerate["cons_grad_norm"],
        "cons_grad_ratio": (norms["cons"] / degenerate["cons_grad_norm"]
                            if degenerate["cons_grad_norm"] else None),
        "lambda_ratio": (report["lambda_cons"] / degenerate["lambda_cons"]
                         if report["lambda_cons"] else None),
        "note": "a smaller coefficient is not thereby a better one; what "
                "changed is that the state now has a consistency gradient to "
                "balance against"}
    del a1, model
    torch.cuda.empty_cache()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["calibrate", "lock"])
    arguments = parser.parse_args()
    import instance_edge_topology as IET
    edges = [tuple(e) for e in IET.build_topology()["edges"]]
    if not V2.sha_file(OUT / "line_internal_split.csv").startswith(V2.LINE_SPLIT_SHA):
        raise RuntimeError("HARD_BLOCKED: LINE split changed")
    if not (OUT / DEGENERATE_RECORD).exists():
        raise RuntimeError("HARD_BLOCK: the degenerate record must be preserved")
    pool = V2.split_indices()[0]

    if arguments.command == "calibrate":
        guard = leakage_guard()
        if not guard["CALIBRATION_LEAKAGE_GUARD_CLEAN"]:
            raise RuntimeError(f"CALIBRATION_LEAKAGE: {guard['violations']}")
        report = run(edges, pool)
        report["leakage_guard"] = guard
        first = OUT / FIRST
        if first.exists():
            previous = json.loads(first.read_text())
            relative = abs(report["lambda_cons"] - previous["lambda_cons"]) / max(
                abs(previous["lambda_cons"]), EPS)
            report["repeat"] = {
                "first_lambda": previous["lambda_cons"],
                "second_lambda": report["lambda_cons"],
                "relative_difference": relative,
                "sup_equal": previous["sup_grad_norm"] == report["sup_grad_norm"],
                "cons_equal": previous["cons_grad_norm"] == report["cons_grad_norm"],
                "L_sup_equal": previous["L_sup_fulltrain"] == report["L_sup_fulltrain"],
                "checkpoint_sha_equal":
                    previous["checkpoint_sha256"] == report["checkpoint_sha256"]}
            report["repeat"]["ONE_PASS_LAMBDA_REPRODUCIBLE"] = bool(
                relative <= REPEAT_TOLERANCE and report["repeat"]["sup_equal"]
                and report["repeat"]["cons_equal"]
                and report["repeat"]["L_sup_equal"]
                and report["repeat"]["checkpoint_sha_equal"])
            (OUT / REPEAT).write_text(json.dumps(report, indent=2, default=float))
            log(f"[calibrate] repeat lambda {report['lambda_cons']:.12g} vs "
                f"{previous['lambda_cons']:.12g}  rel {relative:.3e}  "
                f"REPRODUCIBLE="
                f"{report['repeat']['ONE_PASS_LAMBDA_REPRODUCIBLE']}")
            if not report["repeat"]["ONE_PASS_LAMBDA_REPRODUCIBLE"]:
                raise RuntimeError("ONE_PASS_LAMBDA_NOT_REPRODUCIBLE")
            return
        first.write_text(json.dumps(report, indent=2, default=float))
        audit = report["validity"]
        log(f"[calibrate] state {STATE_NAME}  L_sup last250 "
            f"{report['train_loss_mean_last250']:.6f}")
        log(f"[calibrate] entropy {audit['mean_entropy']:.4f} (uniform "
            f"{audit['uniform_entropy']:.4f}) max p {audit['max_probability']:.3e}"
            f" (uniform {audit['uniform_probability']:.3e}) | JS(a,a) "
            f"{audit['js_identical_views']:.3e} JS(a,b) "
            f"{audit['js_distinct_views']:.6e}  VALID="
            f"{audit['ONE_PASS_STATE_VALID']}")
        log(f"[calibrate] ||g_sup|| {report['sup_grad_norm']:.9g} ||g_cons|| "
            f"{report['cons_grad_norm']:.9g} cos {report['gradient_cosine']:+.6f}"
            f" | L_sup {report['L_sup_fulltrain']:.6f} L_cons "
            f"{report['L_cons_fulltrain']:.6e}")
        if not report["ONE_PASS_CALIBRATION_VALID"]:
            log(f"[calibrate] lambda_cons None  VALID=False  validity "
                f"{report['validity']['ONE_PASS_STATE_VALID']}  g_cons "
                f"{report['cons_grad_norm']:.9g}  L_cons "
                f"{report['L_cons_fulltrain']:.6e}")
            raise RuntimeError("ONE_PASS_CONSISTENCY_SIGNAL_STILL_DEGENERATE")
        log(f"[calibrate] lambda_cons {report['lambda_cons']:.12g}  VALID="
            f"{report['ONE_PASS_CALIBRATION_VALID']}  vs fresh-init lambda "
            f"ratio {report['against_fresh_init']['lambda_ratio']:.6g}  "
            f"g_cons ratio {report['against_fresh_init']['cons_grad_ratio']:.6g}")
        return

    first, repeat = OUT / FIRST, OUT / REPEAT
    if not first.exists() or not repeat.exists():
        raise RuntimeError("ONE_PASS_LAMBDA_NOT_LOCKED: calibrate twice first")
    one, two = json.loads(first.read_text()), json.loads(repeat.read_text())
    locked = {"state": one["state"], "is_historical": False,
              "is_model_selection": False,
              "lambda_cons": one["lambda_cons"],
              "sup_grad_norm": one["sup_grad_norm"],
              "cons_grad_norm": one["cons_grad_norm"],
              "gradient_cosine": one["gradient_cosine"],
              "L_sup_fulltrain": one["L_sup_fulltrain"],
              "L_cons_fulltrain": one["L_cons_fulltrain"],
              "validity": one["validity"], "repeat": two["repeat"],
              "checkpoint": one["checkpoint"],
              "checkpoint_sha256": one["checkpoint_sha256"],
              "checkpoint_use": "audit only; never a P0 or P1 initialisation",
              "augmentation": one["augmentation"],
              "leakage_guard": one["leakage_guard"],
              "supersedes": "FRESH_INIT_CONSISTENCY_CALIBRATION_DEGENERATE",
              "against_fresh_init": one["against_fresh_init"],
              "CALIBRATION_NUMERICAL_REGIME": "deterministic",
              "ACTUAL_TRAINING_NUMERICAL_REGIME": "default",
              "LAMBDA_OPTIMALITY_NOT_ESTABLISHED": True,
              "LAMBDA_SELECTED_WITH_DEV": False, "LAMBDA_SWEEP": False,
              "FURTHER_STATE_MOVE": "FORBIDDEN", **CAP.provenance()}
    locked["CONSISTENCY_LAMBDA_LOCKED"] = bool(
        one["ONE_PASS_CALIBRATION_VALID"]
        and two["repeat"]["ONE_PASS_LAMBDA_REPRODUCIBLE"]
        and one["leakage_guard"]["CALIBRATION_LEAKAGE_GUARD_CLEAN"])
    (OUT / LOCK).write_text(json.dumps(locked, indent=2, default=float))
    log(f"[lock] lambda_cons {locked['lambda_cons']:.12g}  LOCKED="
        f"{locked['CONSISTENCY_LAMBDA_LOCKED']}")
    if not locked["CONSISTENCY_LAMBDA_LOCKED"]:
        raise RuntimeError("ONE_PASS_LAMBDA_NOT_LOCKED")


if __name__ == "__main__":
    main()
