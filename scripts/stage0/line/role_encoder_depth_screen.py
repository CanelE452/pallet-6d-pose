"""Does one more role-conditioned nonlocal block convert the adapter signal?

`a6c987c` left `F50_ADAPTER_INSUFFICIENT`: the constrained adapter is active --
it changes F50 by 29.7% -- and buys 17.42% and 21.23% off F0's medians against a
40% threshold.  It also does not reproduce the large D0/D2 gap the broad late-A1
arm carried.  So the question moves downstream of the feature, to whether the
role decoder has the capacity to use what it is given.

```
R0   F2 exactly: frozen A1 -> zero-init F50 adapter -> token XY
                 -> one role-query cross-attention block -> DirectHoughHead
R1   R0 plus exactly one role-query refinement block
```

The refinement block reads the twelve descriptors the first block produced and
the same F50+XY tokens.  It creates no new role embeddings, so channel k stays
role k and there is no assignment, no matching, no Hungarian step.  Its output
enters as `z + beta * DeltaBlock(z, tokens)` with `beta` a learnable scalar
initialised to zero, which makes step 0 the F2 function exactly rather than
approximately -- `step0` checks that at 0 on the adapted F50 and the first
descriptor and at 1e-6 on the final descriptor, the logits and the loss.

Decision at 25,545 on `D2_LINE_DEV512` against F2 at full precision.  F1's
late-A1 numbers are context and select nothing.  Scope and forbidden phrasing
are fixed in `F50_ADAPTER_SCOPE_ADDENDUM.md` before this runs.
"""
from __future__ import annotations

import argparse, importlib.util, json, os, pathlib, sys, time
import numpy as np, torch, torch.nn as nn

ROOT = pathlib.Path(__file__).resolve().parents[3]
for _e in ("scripts/stage0", "Deep_Object_Pose/common", "challenge/scripts"):
    if str(ROOT / _e) not in sys.path:
        sys.path.insert(0, str(ROOT / _e))


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ADAPTER = _load("F50_ADAPTER_BASE", "scripts/stage0/line/direct_hough_f50_adapter_screen.py")
LONG, DH = ADAPTER.LONG, ADAPTER.DH
CAP, V2, SCALE = ADAPTER.CAP, ADAPTER.V2, ADAPTER.SCALE
OUT, DEV = ADAPTER.OUT, ADAPTER.DEV
RQ = sys.modules["RQ_HOUGH"]

QUERY_DIM, QUERY_HEADS, ROLES = RQ.QUERY_DIM, RQ.QUERY_HEADS, RQ.ROLES
FFN_HIDDEN = 128
MARKS = ADAPTER.MARKS
DECISION_STEP = ADAPTER.DECISION_STEP
PER_ROLE_MARKS = ADAPTER.PER_ROLE_MARKS
DIAGNOSTIC_MARKS = ADAPTER.DIAGNOSTIC_MARKS
PROBE_FRAMES = ADAPTER.PROBE_FRAMES
STEP0_EXACT = 0.0
STEP0_TOLERANCE = 1e-6
REDUCTION = DH.REDUCTION
ADAPTER_RESULT = "f50_adapter.json"                 # R0, the primary baseline
LATE_A1_RESULT = "late_a1_adaptation.json"          # F1, context only
ARMS = ("R0_F2_ADAPTER", "R1_EXTRA_ROLE_BLOCK")
TAG = "r1"


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


class RoleRefinementBlock(nn.Module):
    """One more nonlocal pass over the descriptors the first block produced.

    No new query embeddings: the twelve descriptors arrive already bound to
    their roles and leave in the same channels, so nothing here can relabel a
    role.  `beta` starts at zero, so the block is inert until trained.
    """

    def __init__(self, dim=QUERY_DIM, heads=QUERY_HEADS, hidden=FFN_HIDDEN):
        super().__init__()
        self.norm_query = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(inplace=True),
                                 nn.Linear(hidden, dim))
        self.beta = nn.Parameter(torch.zeros(()))
        self.dim, self.heads = dim, heads

    def delta(self, descriptor, tokens, need_weights=False):
        attended, weights = self.attention(
            self.norm_query(descriptor), tokens, tokens,
            need_weights=need_weights, average_attn_weights=True)
        refined = descriptor + attended
        return refined + self.ffn(self.norm_ffn(refined)) - descriptor, weights

    def forward(self, descriptor, tokens):
        change, _ = self.delta(descriptor, tokens)
        return descriptor + self.beta * change

    def report(self):
        return {"params": sum(p.numel() for p in self.parameters()),
                "dim": self.dim, "heads": self.heads, "ffn_hidden": FFN_HIDDEN,
                "beta": float(self.beta.detach()),
                "attention_params": sum(p.numel() for p in self.attention.parameters()),
                "ffn_params": sum(p.numel() for p in self.ffn.parameters())}


def tokens_of(model, f50):
    """The token construction the first block already uses, reused verbatim."""
    encoder = model.encoder
    flat = f50.flatten(2).transpose(1, 2)
    coordinates = encoder.coordinates[None].expand(f50.shape[0], -1, -1)
    return encoder.to_token(torch.cat([flat, coordinates], -1))


class DeeperRoleModel(nn.Module):
    """R0's modules untouched, with an optional refinement block after them."""

    def __init__(self, base, block=None):
        super().__init__()
        self.base = base
        self.block = block

    def descriptors(self, f50, keep_first=False):
        first = self.base.descriptors(f50)
        if self.block is None:
            return (first, first) if keep_first else first
        refined = self.block(first, tokens_of(self.base, f50))
        return (first, refined) if keep_first else refined

    def forward(self, f50, features):
        return self.base.head(self.descriptors(f50), features)


def build_pair():
    """R0 and R1 sharing bit-identical base weights.

    `DirectHoughModel` seeds itself, so two constructions agree only if nothing
    consumes RNG in between.  Both bases and both adapters are therefore built
    first and copied across; the refinement block is created last, so its own
    initialisation cannot move anything R0 has.
    """
    base_r0 = DH.DirectHoughModel().to(DEV)
    base_r1 = DH.DirectHoughModel().to(DEV)
    base_r1.load_state_dict(base_r0.state_dict())
    adapter_r0 = ADAPTER.F50LineAdapter().to(DEV)
    adapter_r1 = ADAPTER.F50LineAdapter().to(DEV)
    adapter_r1.load_state_dict(adapter_r0.state_dict())
    block = RoleRefinementBlock().to(DEV)
    return (DeeperRoleModel(base_r0), adapter_r0,
            DeeperRoleModel(base_r1, block), adapter_r1, block)


def shared_checksum(model, adapter):
    total = 0.0
    for state in (model.base.state_dict(), adapter.state_dict()):
        for name, parameter in sorted(state.items()):
            total += float(parameter.double().abs().sum())
    return total


def parameter_audit(model, adapter, block, a1):
    encoder = sum(p.numel() for p in model.base.encoder.parameters())
    head = sum(p.numel() for p in model.base.head.parameters())
    adapter_params = adapter.report()["params"]
    extra = block.report()["params"] if block is not None else 0
    shared = adapter_params + encoder + head
    return {"f50_adapter_params": adapter_params,
            "existing_role_encoder_params": encoder,
            "new_refinement_block_params": extra,
            "direct_hough_head_params": head,
            "a1_params": sum(p.numel() for p in a1.parameters()),
            "a1_trainable_params": ADAPTER.trainable_a1_params(a1),
            "trainable_total_R0": shared,
            "trainable_total_R1": shared + extra,
            "trainable_increase": extra / shared if shared else 0.0}


def run_step0(edges):
    """Is R1 the F2 function exactly, before any step?"""
    a1 = ADAPTER.frozen_a1()
    r0, adapter_r0, r1, adapter_r1, block = build_pair()
    grid_theta, grid_rho, valid = DH.lattice()
    features = DH.hypothesis_features(grid_theta, grid_rho)
    indices = V2.split_indices()[0][:PROBE_FRAMES]
    gaps = {"adapted_f50": 0.0, "first_descriptor": 0.0,
            "final_descriptor": 0.0, "logits": 0.0, "loss": 0.0}
    with torch.no_grad():
        for start in range(0, len(indices), CAP.BATCH):
            chunk = indices[start:start + CAP.BATCH]
            pack = V2.load_pack(chunk)
            theta_c, rho_c, support = DH.batch_rows(pack, edges)
            target = DH.target_distribution(
                theta_c.reshape(-1), rho_c.reshape(-1), grid_theta, grid_rho,
                valid).reshape(*theta_c.shape, -1)
            base = ADAPTER.base_f50(pack, a1)
            left, right = adapter_r0(base), adapter_r1(base)
            gaps["adapted_f50"] = max(gaps["adapted_f50"],
                                      float((left - right).abs().max()))
            first_r0 = r0.descriptors(left, keep_first=True)[0]
            first_r1, final_r1 = r1.descriptors(right, keep_first=True)
            gaps["first_descriptor"] = max(gaps["first_descriptor"],
                                           float((first_r0 - first_r1).abs().max()))
            gaps["final_descriptor"] = max(gaps["final_descriptor"],
                                           float((first_r0 - final_r1).abs().max()))
            logits_r0, logits_r1 = r0(left, features), r1(right, features)
            gaps["logits"] = max(gaps["logits"],
                                 float((logits_r0 - logits_r1).abs().max()))
            gaps["loss"] = max(gaps["loss"], abs(
                float(DH.cross_entropy(logits_r0, target, support, valid))
                - float(DH.cross_entropy(logits_r1, target, support, valid))))
    report = {"frames": PROBE_FRAMES, "exact_required": ["adapted_f50",
                                                         "first_descriptor"],
              "tolerance": STEP0_TOLERANCE, "gaps": gaps,
              "beta_at_init": float(block.beta.detach()),
              "shared_checksum_equal": bool(
                  shared_checksum(r0, adapter_r0) == shared_checksum(r1, adapter_r1)),
              "audit": parameter_audit(r1, adapter_r1, block, a1)}
    report["ROLE_DEPTH_STEP0_EQUIVALENT"] = bool(
        gaps["adapted_f50"] == STEP0_EXACT
        and gaps["first_descriptor"] == STEP0_EXACT
        and gaps["final_descriptor"] <= STEP0_TOLERANCE
        and gaps["logits"] <= STEP0_TOLERANCE
        and gaps["loss"] <= STEP0_TOLERANCE
        and report["shared_checksum_equal"])
    return report


def trainable_groups(model, adapter, block):
    groups = [{"params": list(model.base.parameters()), "lr": CAP.LR},
              {"params": list(adapter.parameters()), "lr": CAP.LR}]
    if block is not None:
        groups.append({"params": list(block.parameters()), "lr": CAP.LR})
    return groups


def run_wiring(edges):
    """Does gradient reach the new block, the old parts, and never A1?"""
    a1 = ADAPTER.frozen_a1()
    _, _, model, adapter, block = build_pair()
    grid_theta, grid_rho, valid = DH.lattice()
    features = DH.hypothesis_features(grid_theta, grid_rho)
    optimiser = torch.optim.AdamW(trainable_groups(model, adapter, block),
                                  lr=CAP.LR, weight_decay=CAP.WD)
    pack = ADAPTER.probe_pack()
    theta_c, rho_c, support = DH.batch_rows(pack, edges)
    target = DH.target_distribution(
        theta_c.reshape(-1), rho_c.reshape(-1), grid_theta, grid_rho, valid
    ).reshape(*theta_c.shape, -1)
    base = ADAPTER.base_f50(pack, a1)

    def step():
        loss = DH.cross_entropy(model(adapter(base), features), target, support,
                                valid)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        return loss

    step()
    beta_grad = float(block.beta.grad.abs().max())
    optimiser.step()
    beta_after = float(block.beta.detach().abs().max())
    step()
    report = {"beta_grad_at_step0": beta_grad, "beta_after_one_step": beta_after,
              "new_attention_grad_norm": float(
                  block.attention.in_proj_weight.grad.norm()),
              "new_ffn_grad_norm": float(block.ffn[0].weight.grad.norm()),
              "adapter_grad_norm": float(adapter.body[0].weight.grad.norm()),
              "first_role_block_grad_norm": float(
                  model.base.encoder.attention.in_proj_weight.grad.norm()),
              "head_grad_norm": float(model.base.head.project.weight.grad.norm()),
              "a1_params_with_grad": sum(
                  1 for p in a1.parameters()
                  if p.grad is not None and float(p.grad.abs().sum()) != 0.0),
              "a1_trainable_params": ADAPTER.trainable_a1_params(a1)}
    report["ROLE_DEPTH_GRADIENT_WIRING"] = bool(
        beta_grad > 0.0 and beta_after != 0.0
        and report["new_attention_grad_norm"] > 0.0
        and report["new_ffn_grad_norm"] > 0.0
        and report["adapter_grad_norm"] > 0.0
        and report["first_role_block_grad_norm"] > 0.0
        and report["head_grad_norm"] > 0.0
        and report["a1_params_with_grad"] == 0
        and report["a1_trainable_params"] == 0)
    return report


def run_memory(edges):
    torch.cuda.reset_peak_memory_stats(DEV)
    a1 = ADAPTER.frozen_a1()
    _, _, model, adapter, block = build_pair()
    grid_theta, grid_rho, valid = DH.lattice()
    features = DH.hypothesis_features(grid_theta, grid_rho)
    optimiser = torch.optim.AdamW(trainable_groups(model, adapter, block),
                                  lr=CAP.LR, weight_decay=CAP.WD)
    pack = ADAPTER.probe_pack()
    theta_c, rho_c, support = DH.batch_rows(pack, edges)
    target = DH.target_distribution(
        theta_c.reshape(-1), rho_c.reshape(-1), grid_theta, grid_rho, valid
    ).reshape(*theta_c.shape, -1)
    loss = DH.cross_entropy(
        model(adapter(ADAPTER.base_f50(pack, a1)), features), target, support,
        valid)
    optimiser.zero_grad(set_to_none=True); loss.backward(); optimiser.step()
    peak = torch.cuda.max_memory_allocated(DEV)
    total = torch.cuda.get_device_properties(DEV).total_memory
    return {"batch": CAP.BATCH, "peak_bytes": int(peak),
            "peak_mib": peak / 2 ** 20, "device_total_mib": total / 2 ** 20,
            "loss": float(loss.detach()),
            "ROLE_DEPTH_BATCH8_MEMORY_OK": bool(peak < total)}


@torch.no_grad()
def block_use(indices, model, adapter, a1):
    """Is the new block doing anything?  Descriptive only, never causal."""
    change, cosine, entropy, ffn_norm = [], [], [], []
    for start in range(0, len(indices), CAP.BATCH):
        pack = V2.load_pack(indices[start:start + CAP.BATCH])
        f50 = adapter(ADAPTER.base_f50(pack, a1))
        first = model.base.descriptors(f50)
        tokens = tokens_of(model.base, f50)
        delta, weights = model.block.delta(first, tokens, need_weights=True)
        refined = first + model.block.beta * delta
        change.append(float((refined - first).flatten(1).norm(dim=1).mean()
                            / first.flatten(1).norm(dim=1).mean().clamp_min(1e-12)))
        cosine.append(float(nn.functional.cosine_similarity(
            first.flatten(1), refined.flatten(1), dim=1).mean()))
        safe = weights.clamp_min(1e-12)
        entropy.append(float((-(safe * safe.log()).sum(-1)).mean()))
        ffn_norm.append(float(model.block.ffn(
            model.block.norm_ffn(first)).flatten(1).norm(dim=1).mean()
            / first.flatten(1).norm(dim=1).mean().clamp_min(1e-12)))
    return {"beta": float(model.block.beta.detach()),
            "relative_descriptor_change": float(np.mean(change)),
            "descriptor_cosine": float(np.mean(cosine)),
            "attention_entropy": float(np.mean(entropy)),
            "ffn_output_norm_ratio": float(np.mean(ffn_norm))}


@torch.no_grad()
def evaluate(indices, model, adapter, a1, edges, features, grid_theta, grid_rho,
             valid, per_role=False):
    model.eval()
    angle, offset, roles = [], [], []
    for start in range(0, len(indices), CAP.BATCH):
        chunk = indices[start:start + CAP.BATCH]
        if len(chunk) < 2:
            continue
        pack = V2.load_pack(chunk)
        theta_c, rho_c, support = DH.batch_rows(pack, edges)
        scores = model(adapter(ADAPTER.base_f50(pack, a1)), features)
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


def train_arm(pool, marks, edges, populations, per_pass, probe):
    grid_theta, grid_rho, valid = DH.lattice()
    features = DH.hypothesis_features(grid_theta, grid_rho)
    a1 = ADAPTER.frozen_a1()
    _, _, model, adapter, block = build_pair()
    if ADAPTER.trainable_a1_params(a1) != 0:
        raise RuntimeError("HARD_BLOCK: A1 is not fully frozen")
    optimiser = torch.optim.AdamW(trainable_groups(model, adapter, block),
                                  lr=CAP.LR, weight_decay=CAP.WD)
    history, losses, done = {}, [], 0
    for chunk, _ in V2.step_schedule(pool, max(marks), CAP.BATCH):
        model.train(); adapter.train()
        pack = V2.load_pack(chunk)
        theta_c, rho_c, support = DH.batch_rows(pack, edges)
        target = DH.target_distribution(
            theta_c.reshape(-1), rho_c.reshape(-1), grid_theta, grid_rho, valid
        ).reshape(*theta_c.shape, -1)
        loss = DH.cross_entropy(
            model(adapter(ADAPTER.base_f50(pack, a1)), features), target,
            support, valid)
        optimiser.zero_grad(set_to_none=True)
        loss.backward(); optimiser.step()
        losses.append(float(loss.detach()))
        done += 1
        if done in marks:
            model.eval(); adapter.eval()
            entry = {"step": done,
                     "train_loss_mean_last250": float(np.mean(losses[-250:])),
                     "train_loss_slope_last250": LONG.slope(losses[-250:]),
                     "train_loss_mean_last_pass": float(np.mean(losses[-per_pass:])),
                     "train_loss_slope_last_pass": LONG.slope(losses[-per_pass:]),
                     "diagnostic_only": done in DIAGNOSTIC_MARKS,
                     "adapter_use": ADAPTER.adapter_use(probe, adapter, a1),
                     "block_use": block_use(probe, model, adapter, a1),
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
            b = entry["block_use"]
            log(f"  {TAG} @{done:6d} CE {entry['train_loss_mean_last250']:.6f} "
                f"slope {entry['train_loss_slope_last_pass']:+.3e} | beta "
                f"{b['beta']:+.5f} dchg {b['relative_descriptor_change']:.5f} "
                f"cos {b['descriptor_cosine']:.6f} ent "
                f"{b['attention_entropy']:.4f} | D2/D0 "
                f"{entry['generalization']['angle_ratio']:.3f}/"
                f"{entry['generalization']['offset_ratio']:.3f}")
            torch.save({"tag": TAG, "step": done, "model": model.state_dict(),
                        "adapter": adapter.state_dict(), **CAP.provenance()},
                       CAP.checkpoint_path(f"DH_{TAG}", f"step_{done:05d}"))
            history[str(done)] = entry
    return history, model, adapter, block


def baselines():
    r0 = json.loads((OUT / ADAPTER_RESULT).read_text())["history"][str(DECISION_STEP)]
    f1 = json.loads((OUT / LATE_A1_RESULT).read_text())["histories"][
        "F1_LATE_A1_TRAINABLE"][str(DECISION_STEP)]
    return r0, f1


def judge(history):
    r0_entry, f1_entry = baselines()
    r0 = r0_entry["D2_LINE_DEV512"]
    f1 = f1_entry["D2_LINE_DEV512"]
    r1 = history[str(DECISION_STEP)]["D2_LINE_DEV512"]
    keys = ("angle_median", "offset_median", "angle_p90", "offset_p90")
    out = {"decision_step": DECISION_STEP, "population": "D2_LINE_DEV512",
           "R0": {k: r0[k] for k in keys}, "R1": {k: r1[k] for k in keys},
           "F1_context_only": {k: f1[k] for k in keys},
           "vs_R0": {k: 1.0 - r1[k] / r0[k] for k in keys},
           "vs_F1_context_only": {k: 1.0 - r1[k] / f1[k] for k in keys},
           "ABSOLUTE_PASS": bool(r1["PASS"] and r1["SAFETY"]),
           "finite": bool(history[str(DECISION_STEP)]["finite"])}
    out["REDUCTION_40"] = bool(
        out["vs_R0"]["angle_median"] >= REDUCTION
        and out["vs_R0"]["offset_median"] >= REDUCTION)
    out["PARETO_BETTER_THAN_LATE_UNFREEZE"] = bool(
        all(r1[k] <= f1[k] for k in keys) and any(r1[k] < f1[k] for k in keys))
    final = history[str(DECISION_STEP)]
    out["generalization"] = {
        "R1": final["generalization"],
        "R0_context_only": {
            "angle_ratio": r0["angle_median"] / r0_entry["D0_SEEN512"]["angle_median"],
            "offset_ratio": r0["offset_median"] / r0_entry["D0_SEEN512"]["offset_median"]},
        "F1_context_only": {
            "angle_ratio": f1["angle_median"] / f1_entry["D0_SEEN512"]["angle_median"],
            "offset_ratio": f1["offset_median"] / f1_entry["D0_SEEN512"]["offset_median"]}}
    d0 = final["D0_SEEN512"]
    out["SPECIALIZES"] = bool(
        1.0 - d0["angle_median"] / r0_entry["D0_SEEN512"]["angle_median"] >= REDUCTION
        and out["vs_R0"]["angle_median"] < REDUCTION)
    if not out["finite"]:
        out["DECISION"] = "ROLE_ENCODER_DEPTH_UNSTABLE"
        out["RETRY_WITH_NEW_LR"] = "FORBIDDEN"
    elif out["ABSOLUTE_PASS"]:
        out["DECISION"] = "ROLE_ENCODER_DEPTH_VALID_CANDIDATE"
        out["STATUS"] = "LINE_STAGE_CANDIDATE"
        out["NEXT"] = "same_protocol_replicate"
    elif out["REDUCTION_40"]:
        out["DECISION"] = "ROLE_ENCODER_DEPTH_SIGNAL"
        out["NEXT"] = "failure_shape_analysis_before_any_further_change"
    else:
        out["DECISION"] = "ROLE_ENCODER_DEPTH_INSUFFICIENT"
        out["SCOPE"] = "one extra block at this width and head count only"
        out["ROLE_ENCODER_CAPACITY_EXONERATED"] = False
        out["NEXT"] = "separate_lock_decides_between_A_B_C"
    out.setdefault("STATUS", "NOT_LOCKED")
    out["CIGM"] = "BLOCKED"
    return out


def build_plan(pool):
    a1 = ADAPTER.frozen_a1()
    _, _, model, adapter, block = build_pair()
    r0_entry, f1_entry = baselines()
    audit = parameter_audit(model, adapter, block, a1)
    plan = {"arms": list(ARMS), "factor": "ONE_EXTRA_ROLE_REFINEMENT_BLOCK",
            "marks": list(MARKS), "decision_step": DECISION_STEP,
            "decision_population": "D2_LINE_DEV512",
            "diagnostic_population": "D0_SEEN512",
            "per_role_marks": list(PER_ROLE_MARKS),
            "frames": len(pool), "batch": CAP.BATCH, "audit": audit,
            "block": block.report(),
            "new_role_embeddings": 0, "role_assignment": None,
            "lr": CAP.LR, "weight_decay": CAP.WD, "scheduler": None,
            "gradient_clipping": None, "lr_sweep": False, "block_count_sweep": False,
            "gate": {"angle_median": CAP.ANGLE_BUDGET_DEG,
                     "offset_median": CAP.OFFSET_BUDGET_CELL,
                     "angle_p90": CAP.SAFETY_ANGLE,
                     "offset_p90": CAP.SAFETY_OFFSET},
            "reduction": REDUCTION,
            "baseline_R0_full_precision": r0_entry["D2_LINE_DEV512"],
            "context_F1_full_precision": f1_entry["D2_LINE_DEV512"],
            **CAP.provenance()}
    del a1, model, adapter, block
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
    for name in (ADAPTER_RESULT, LATE_A1_RESULT):
        if not (OUT / name).exists():
            raise RuntimeError(f"HARD_BLOCK: {name} is missing")
    pool = V2.split_indices()[0]

    if arguments.command == "plan":
        plan = build_plan(pool)
        (OUT / "role_depth_plan.json").write_text(json.dumps(plan, indent=2))
        audit = plan["audit"]
        log(f"[plan] new block {audit['new_refinement_block_params']:,} params "
            f"(dim {plan['block']['dim']} heads {plan['block']['heads']} ffn "
            f"{plan['block']['ffn_hidden']}) | trainable "
            f"{audit['trainable_total_R0']:,} -> {audit['trainable_total_R1']:,} "
            f"(+{audit['trainable_increase']:.1%})")
        log(f"[plan] adapter {audit['f50_adapter_params']:,} | first role block "
            f"{audit['existing_role_encoder_params']:,} | head "
            f"{audit['direct_hough_head_params']:,} | A1 trainable "
            f"{audit['a1_trainable_params']} | new role embeddings "
            f"{plan['new_role_embeddings']}")
        log(f"[plan] decision {DECISION_STEP} on D2 only  gate "
            f"{CAP.ANGLE_BUDGET_DEG}/{CAP.OFFSET_BUDGET_CELL} safety "
            f"{CAP.SAFETY_ANGLE}/{CAP.SAFETY_OFFSET}  reduction {REDUCTION}")
        return

    if arguments.command == "step0":
        report = run_step0(edges)
        (OUT / "role_depth_step0.json").write_text(
            json.dumps(report, indent=2, default=float))
        g = report["gaps"]
        log(f"[step0] f50 {g['adapted_f50']:.3e} first {g['first_descriptor']:.3e}"
            f" (exact required) | final {g['final_descriptor']:.3e} logits "
            f"{g['logits']:.3e} loss {g['loss']:.3e} (tol {STEP0_TOLERANCE})  "
            f"EQUIVALENT={report['ROLE_DEPTH_STEP0_EQUIVALENT']}")
        if not report["ROLE_DEPTH_STEP0_EQUIVALENT"]:
            raise RuntimeError("ROLE_DEPTH_STEP0_MISMATCH")
        return

    if arguments.command == "wiring":
        report = run_wiring(edges)
        (OUT / "role_depth_wiring.json").write_text(
            json.dumps(report, indent=2, default=float))
        log(f"[wiring] beta grad {report['beta_grad_at_step0']:.3e} -> beta "
            f"{report['beta_after_one_step']:.3e} | new attn "
            f"{report['new_attention_grad_norm']:.3e} new ffn "
            f"{report['new_ffn_grad_norm']:.3e} | adapter "
            f"{report['adapter_grad_norm']:.3e} first block "
            f"{report['first_role_block_grad_norm']:.3e} head "
            f"{report['head_grad_norm']:.3e} | A1 with grad "
            f"{report['a1_params_with_grad']}  OK="
            f"{report['ROLE_DEPTH_GRADIENT_WIRING']}")
        if not report["ROLE_DEPTH_GRADIENT_WIRING"]:
            raise RuntimeError("ROLE_DEPTH_GRADIENT_WIRING_FAIL")
        return

    if arguments.command == "memory":
        report = run_memory(edges)
        (OUT / "role_depth_memory.json").write_text(
            json.dumps(report, indent=2, default=float))
        log(f"[memory] batch {report['batch']} peak {report['peak_mib']:.1f} MiB "
            f"of {report['device_total_mib']:.0f} MiB  OK="
            f"{report['ROLE_DEPTH_BATCH8_MEMORY_OK']}")
        if not report["ROLE_DEPTH_BATCH8_MEMORY_OK"]:
            raise RuntimeError("ROLE_DEPTH_BATCH8_MEMORY_FAIL")
        return

    for name, key, label in (
            ("role_depth_step0.json", "ROLE_DEPTH_STEP0_EQUIVALENT",
             "ROLE_DEPTH_STEP0_MISMATCH"),
            ("role_depth_wiring.json", "ROLE_DEPTH_GRADIENT_WIRING",
             "ROLE_DEPTH_GRADIENT_WIRING_FAIL"),
            ("role_depth_memory.json", "ROLE_DEPTH_BATCH8_MEMORY_OK",
             "ROLE_DEPTH_BATCH8_MEMORY_FAIL")):
        path = OUT / name
        if not path.exists() or not json.loads(path.read_text())[key]:
            raise RuntimeError(f"{label}: preflight must pass first")
    plan = build_plan(pool)
    per_pass = V2.steps_per_pass(pool, CAP.BATCH)
    probe = V2.split_indices()[0][:PROBE_FRAMES]
    history, _, _, block = train_arm(pool, MARKS, edges, SCALE.populations(),
                                     per_pass, probe)
    report = {"plan": plan, "history": history, "verdict": judge(history),
              "block_final": block.report(), **CAP.provenance()}
    (OUT / "role_depth.json").write_text(
        json.dumps(report, indent=2, default=float))
    v = report["verdict"]
    log(f"[run] {v['DECISION']}  R1 {v['R1']['angle_median']:.6f}/"
        f"{v['R1']['offset_median']:.6f} p90 {v['R1']['angle_p90']:.6f}/"
        f"{v['R1']['offset_p90']:.6f}")
    log(f"[run] vs R0 angle {v['vs_R0']['angle_median']:+.2%} offset "
        f"{v['vs_R0']['offset_median']:+.2%} | pareto vs F1 "
        f"{v['PARETO_BETTER_THAN_LATE_UNFREEZE']} | SPECIALIZES "
        f"{v['SPECIALIZES']}")


if __name__ == "__main__":
    main()
