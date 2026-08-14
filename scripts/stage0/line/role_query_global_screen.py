"""Can role-conditioned nonlocal access place the supporting lines?

`e49e304` found absolute XY worth 20% and GAP-FiLM worth 3%.  GAP throws the
spatial arrangement away before the global summary is formed, so its failure says
nothing about global reasoning that *keeps* the arrangement.  This screen asks
that question with exactly one factor.

```
Q0_LOCAL_XY            C_G0P1 unchanged: F50 -> MAP100 -> additive XY -> head
Q1_ROLE_QUERY_GLOBAL   the same, plus twelve fixed role queries that
                       cross-attend to the 2,500 F50 tokens (each carrying its
                       own normalised x, y) and emit one descriptor per role,
                       which forms an additive residual on the role's map logit
```

The residual is a spatial compatibility score, not a coordinate: no theta, no
rho, no regression head.  Global reasoning happens on F50's 50x50 grid rather
than on MAP100, and the local path survives untouched, so a dead global branch
leaves Q1 exactly equal to Q0.

Target, map loss, Hough decoder, MAP100, sigma, splits, optimizer, batch,
learning rate, seed and step marks are all as locked.
"""
from __future__ import annotations
import os as _os, sys as _sys

# --- stage0 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_S0 = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_S0] + [_os.path.join(_S0, _d) for _d in sorted(_os.listdir(_S0))
                         if _os.path.isdir(_os.path.join(_S0, _d)) and not _d.startswith(".")]


import argparse, importlib.util, json, math, pathlib, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[3]
for _e in ("scripts/stage0", "Deep_Object_Pose/common", "challenge/scripts"):
    if str(ROOT / _e) not in sys.path:
        sys.path.insert(0, str(ROOT / _e))


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ARCH = _load("ARCH_RQ", "scripts/stage0/architecture_context_position_screen.py")
CAP, H, V2, SCALE = ARCH.CAP, ARCH.H, ARCH.V2, ARCH.SCALE
OUT, DEV, MAP = CAP.OUT, CAP.DEV, CAP.MAP
F50_GRID, F50_CHANNELS = 50, 128
ARMS = ("Q0_LOCAL_XY", "Q1_ROLE_QUERY_GLOBAL")
BASELINE_SOURCE = ("architecture_screen_results.json", "C_G0P1")
MARKS = ARCH.MARKS
REDUCTION = 0.40
QUERY_DIM, QUERY_HEADS, ROLES = 64, 4, 12
INIT_TOLERANCE = 1e-6


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def baseline_reference():
    """C_G0P1 at full stored precision -- never a transcribed literal."""
    recorded = json.loads((OUT / BASELINE_SOURCE[0]).read_text())["arms"][BASELINE_SOURCE[1]]
    return {mark: {key: recorded[str(mark)]["D2_LINE_DEV512"][key]
                   for key in ("angle_median", "offset_median",
                               "angle_p90", "offset_p90")}
            for mark in MARKS}


def thresholds():
    final = baseline_reference()[max(MARKS)]
    return {"absolute": {"angle_median": CAP.ANGLE_BUDGET_DEG,
                         "offset_median": CAP.OFFSET_BUDGET_CELL},
            "safety": {"angle_p90": CAP.SAFETY_ANGLE,
                       "offset_p90": CAP.SAFETY_OFFSET},
            "reduction_40": {"angle_median": final["angle_median"] * (1 - REDUCTION),
                             "offset_median": final["offset_median"] * (1 - REDUCTION)},
            "baseline_full_precision": final}


# --------------------------------------------------------------- the factor
class RoleQueryGlobal(nn.Module):
    """Twelve fixed role queries over F50 tokens that keep their own position.

    Channel k is role k for the whole run; there is no matching step, because an
    assignment would let the block relabel its own outputs.
    """

    def __init__(self, channels=F50_CHANNELS, grid=F50_GRID, dim=QUERY_DIM,
                 heads=QUERY_HEADS, roles=ROLES):
        super().__init__()
        axis = torch.linspace(-1.0, 1.0, grid)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer("coordinates", torch.stack([xx, yy]).reshape(2, -1).T,
                             persistent=False)
        self.to_token = nn.Linear(channels + 2, dim)
        self.queries = nn.Embedding(roles, dim)
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm_query = nn.LayerNorm(dim)
        self.norm_out = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 2 * dim), nn.ReLU(inplace=True),
                                 nn.Linear(2 * dim, dim))
        self.pixel = nn.Conv2d(channels, dim, 1)
        self.role = nn.Linear(dim, dim)
        self.alpha = nn.Parameter(torch.zeros(()))          # zero-init gate
        self.dim = dim

    def forward(self, f50, map_feature):
        batch = f50.shape[0]
        flat = f50.flatten(2).transpose(1, 2)                       # B, 2500, C
        coordinates = self.coordinates[None].expand(batch, -1, -1)
        tokens = self.to_token(torch.cat([flat, coordinates], -1))
        query = self.norm_query(self.queries.weight[None].expand(batch, -1, -1))
        attended, weights = self.attention(query, tokens, tokens,
                                           need_weights=True, average_attn_weights=True)
        descriptor = self.norm_out(query + attended)
        descriptor = descriptor + self.ffn(descriptor)              # B, 12, dim
        pixel = self.pixel(map_feature)                             # B, dim, M, M
        role = self.role(descriptor)                                # B, 12, dim
        residual = torch.einsum("brd,bdhw->brhw", role, pixel) / math.sqrt(self.dim)
        return self.alpha * residual, {"descriptor": descriptor, "attention": weights}


class RoleQueryModel(nn.Module):
    """Q0 plus, optionally, the one factor."""

    def __init__(self, global_role):
        super().__init__()
        torch.manual_seed(CAP.SEED)
        self.head = CAP.SupportingLineHead(F50_CHANNELS)
        self.position = ARCH.AbsoluteXY()
        self.global_role = RoleQueryGlobal() if global_role else None

    def forward(self, f50, feature):
        feature = self.position(feature)
        logit = self.head(feature)
        extra = None
        if self.global_role is not None:
            residual, extra = self.global_role(f50, feature)
            logit = logit + residual
        return logit, extra


def build_arm(name):
    model = RoleQueryModel(name == "Q1_ROLE_QUERY_GLOBAL").to(DEV)
    return model, list(model.parameters())


def features(pack, a1):
    with torch.no_grad():
        f50, _, _ = a1(pack["images"])
    f50 = f50.detach()
    return f50, F.interpolate(f50, size=(MAP, MAP), mode="bilinear",
                              align_corners=False)


# ------------------------------------------------------------ wiring checks
def wiring_report(edges, a1, frames=16):
    """Zero-init must mean 'starts equal', never 'is dead'."""
    indices = V2.split_indices()[0][:frames]
    pack = V2.load_pack(indices[:CAP.BATCH])
    _, _, seg, target = CAP.geometry(pack, edges)
    support = torch.tensor(seg["hit"], device=DEV)
    f50, feature = features(pack, a1)

    q0, _ = build_arm("Q0_LOCAL_XY"); q0.eval()
    q1, parameters = build_arm("Q1_ROLE_QUERY_GLOBAL")
    with torch.no_grad():
        base = q0(f50, feature)[0]
        start = q1(f50, feature)[0]
    report = {"init_max_abs_diff": float((start - base).abs().max()),
              "tolerance": INIT_TOLERANCE}
    report["INIT_EQUIVALENT"] = bool(report["init_max_abs_diff"] <= INIT_TOLERANCE)

    q1.train()
    optimiser = torch.optim.AdamW(parameters, lr=CAP.LR, weight_decay=CAP.WD)
    loss = CAP.map_loss(q1(f50, feature)[0], target, support)
    loss.backward()
    report["alpha_grad_at_step0"] = float(q1.global_role.alpha.grad.abs())
    optimiser.step(); optimiser.zero_grad(set_to_none=True)
    report["alpha_after_one_step"] = float(q1.global_role.alpha.detach())
    loss = CAP.map_loss(q1(f50, feature)[0], target, support)
    loss.backward()
    attention_grad = sum(float(p.grad.norm()) for p in q1.global_role.attention.parameters()
                         if p.grad is not None)
    report["attention_grad_norm_at_step2"] = attention_grad
    report["GRADIENT_ALIVE"] = bool(report["alpha_grad_at_step0"] > 0
                                    and report["alpha_after_one_step"] != 0.0
                                    and attention_grad > 0)
    return report


# ------------------------------------------------------------------ metrics
@torch.no_grad()
def evaluate(indices, model, a1, edges, coarse, xx, yy, permute=None, per_role=False):
    model.eval()
    angle, offset, roles = [], [], []
    margin, entropy = [], []
    for start in range(0, len(indices), CAP.BATCH):
        chunk = indices[start:start + CAP.BATCH]
        if len(chunk) < 2:
            continue
        pack = V2.load_pack(chunk)
        theta, rho, seg, _ = CAP.geometry(pack, edges)
        support = torch.tensor(seg["hit"], device=DEV)
        f50, feature = features(pack, a1)
        logit, _ = model(f50, feature)
        if permute is not None:
            logit = logit[:, list(permute)]
        probability = torch.sigmoid(logit)
        theta_t = torch.tensor(theta, dtype=torch.float32, device=DEV)
        rho_t = torch.tensor(rho, dtype=torch.float32, device=DEV)
        for frame in range(probability.shape[0]):
            live = torch.nonzero(support[frame]).flatten()
            if live.numel() == 0:
                continue
            maps = probability[frame][live].reshape(live.numel(), -1).T.contiguous()
            decoded = H.decode(maps, coarse, xx, yy)[H.PRIMARY]
            a, o = H.measure(decoded["normal"], decoded["rho"],
                             theta_t[frame][live], rho_t[frame][live])
            angle.append(a); offset.append(o)
            roles.append(live.cpu().numpy())
            margin.append(decoded["margin"].cpu().numpy())
            entropy.append(decoded["entropy"].cpu().numpy())
    angle = np.concatenate(angle) if angle else np.zeros(1)
    offset = np.concatenate(offset) if offset else np.zeros(1)
    report = {"angle_median": float(np.median(angle)),
              "angle_p90": float(np.percentile(angle, 90)),
              "offset_median": float(np.median(offset)),
              "offset_p90": float(np.percentile(offset, 90)), "n": int(angle.size),
              "margin_median": float(np.median(np.concatenate(margin))),
              "entropy_median": float(np.median(np.concatenate(entropy))),
              "PASS": bool(np.median(angle) <= CAP.ANGLE_BUDGET_DEG
                           and np.median(offset) <= CAP.OFFSET_BUDGET_CELL),
              "SAFETY": bool(np.percentile(angle, 90) <= CAP.SAFETY_ANGLE
                             and np.percentile(offset, 90) <= CAP.SAFETY_OFFSET)}
    if per_role:
        role_index = np.concatenate(roles)
        report["per_role"] = {}
        for r in range(ROLES):
            keep = role_index == r
            if keep.sum():
                report["per_role"][str(r)] = {
                    "n": int(keep.sum()),
                    "angle_median": float(np.median(angle[keep])),
                    "angle_p90": float(np.percentile(angle[keep], 90)),
                    "offset_median": float(np.median(offset[keep])),
                    "offset_p90": float(np.percentile(offset[keep], 90))}
    return report


def run_arm(name, pool, populations, edges, coarse, xx, yy, a1):
    model, parameters = build_arm(name)
    optimiser = torch.optim.AdamW(parameters, lr=CAP.LR, weight_decay=CAP.WD)
    history, losses, done = {}, [], 0
    for chunk, _ in V2.step_schedule(pool, max(MARKS), CAP.BATCH):
        model.train()
        pack = V2.load_pack(chunk)
        _, _, seg, target = CAP.geometry(pack, edges)
        f50, feature = features(pack, a1)
        loss = CAP.map_loss(model(f50, feature)[0], target,
                            torch.tensor(seg["hit"], device=DEV))
        optimiser.zero_grad(set_to_none=True)
        loss.backward(); optimiser.step()
        losses.append(float(loss.detach()))
        done += 1
        if done in MARKS:
            entry = {"step": done,
                     "train_loss_mean_last250": float(np.mean(losses[-250:])),
                     "train_loss_slope_last250": float(
                         np.polyfit(np.arange(len(losses[-250:])), losses[-250:], 1)[0])}
            for label, indices in populations.items():
                entry[label] = evaluate(indices, model, a1, edges, coarse, xx, yy,
                                        per_role=(label == "D2_LINE_DEV512"))
                log(f"  {name} @{done:5d} {label:<16} angle med "
                    f"{entry[label]['angle_median']:7.4f} p90 "
                    f"{entry[label]['angle_p90']:7.3f} | offset med "
                    f"{entry[label]['offset_median']:7.4f}  PASS={entry[label]['PASS']}")
            if model.global_role is not None:
                with torch.no_grad():
                    pack = V2.load_pack(populations["D2_LINE_DEV512"][:CAP.BATCH])
                    f50, feature = features(pack, a1)
                    _, extra = model(f50, feature)
                    attention = extra["attention"]
                    entry["global_branch"] = {
                        "alpha": float(model.global_role.alpha.detach()),
                        "descriptor_norm": float(extra["descriptor"].norm(dim=-1).mean()),
                        "attention_entropy": float(
                            -(attention * (attention + 1e-12).log()).sum(-1).mean())}
            torch.save({"arm": name, "step": done, "model": model.state_dict(),
                        "optimizer": optimiser.state_dict(), **CAP.provenance()},
                       CAP.checkpoint_path(f"RQ_{name}", f"step_{done:05d}"))
            history[str(done)] = entry
            (OUT / f"role_query_trajectory_{name}.json").write_text(
                json.dumps(history, indent=2, default=float))
    return history, model


def decide(report):
    limits = thresholds()
    final = report["arms"]["Q1_ROLE_QUERY_GLOBAL"][str(max(MARKS))]["D2_LINE_DEV512"]
    base = limits["baseline_full_precision"]
    verdict = {
        "angle_reduction": 1.0 - final["angle_median"] / base["angle_median"],
        "offset_reduction": 1.0 - final["offset_median"] / base["offset_median"],
        "ABSOLUTE_PASS": bool(final["PASS"] and final["SAFETY"]),
        "REDUCTION_40": bool(
            final["angle_median"] <= limits["reduction_40"]["angle_median"]
            and final["offset_median"] <= limits["reduction_40"]["offset_median"])}
    slope = report["arms"]["Q1_ROLE_QUERY_GLOBAL"][str(max(MARKS))]["train_loss_slope_last250"]
    if verdict["ABSOLUTE_PASS"]:
        verdict["DECISION"] = "ROLE_CONDITIONED_GLOBAL_MAP_VALID"
    elif verdict["REDUCTION_40"]:
        verdict["DECISION"] = "ROLE_CONDITIONED_GLOBAL_SIGNAL"
        verdict["LONGER_QUERY_DECODER_SCHEDULE_ALLOWED"] = bool(slope < 0)
    else:
        verdict["DECISION"] = "ROLE_CONDITIONED_GLOBAL_MAP_FAIL"
        verdict["NEXT"] = "DIRECT_HOUGH_SPACE_ROLE_HEATMAP"
    verdict["thresholds"] = limits
    return verdict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "wiring", "run", "promote"])
    arguments = parser.parse_args()
    import instance_edge_topology as IET
    edges = [tuple(e) for e in IET.build_topology()["edges"]]
    if not V2.sha_file(OUT / "line_internal_split.csv").startswith(V2.LINE_SPLIT_SHA):
        raise RuntimeError("HARD_BLOCKED: LINE split changed")
    pool = V2.split_indices()[0]
    populations = SCALE.populations()
    plan = {"arms": list(ARMS), "pool": "FULL", "pool_frames": len(pool),
            "marks": list(MARKS), "primary_population": "D2_LINE_DEV512",
            "baseline_source": list(BASELINE_SOURCE),
            "baseline_trajectory": baseline_reference(),
            "thresholds": thresholds(), "query_dim": QUERY_DIM,
            "query_heads": QUERY_HEADS, "roles": ROLES, **CAP.provenance()}

    if arguments.command == "plan":
        (OUT / "role_query_plan.json").write_text(json.dumps(plan, indent=2))
        limits = plan["thresholds"]
        log(f"[plan] baseline {limits['baseline_full_precision']['angle_median']:.6f} / "
            f"{limits['baseline_full_precision']['offset_median']:.6f}  40% -> "
            f"{limits['reduction_40']['angle_median']:.6f} / "
            f"{limits['reduction_40']['offset_median']:.6f}")
        return

    a1 = V2.load_a1()
    if arguments.command == "wiring":
        report = wiring_report(edges, a1)
        (OUT / "role_query_wiring.json").write_text(json.dumps(report, indent=2))
        log(f"[wiring] init diff {report['init_max_abs_diff']:.3e}  "
            f"alpha grad {report['alpha_grad_at_step0']:.3e}  alpha after 1 step "
            f"{report['alpha_after_one_step']:.3e}  attn grad "
            f"{report['attention_grad_norm_at_step2']:.3e}")
        if not report["INIT_EQUIVALENT"]:
            raise RuntimeError("INIT_NOT_EQUIVALENT")
        if not report["GRADIENT_ALIVE"]:
            raise RuntimeError("GLOBAL_BRANCH_GRADIENT_WIRING_FAIL")
        return

    wiring = OUT / "role_query_wiring.json"
    if not wiring.exists():
        raise RuntimeError("run wiring first")
    checked = json.loads(wiring.read_text())
    if not (checked["INIT_EQUIVALENT"] and checked["GRADIENT_ALIVE"]):
        raise RuntimeError("GLOBAL_BRANCH_GRADIENT_WIRING_FAIL")
    coarse, (xx, yy) = H.CoarseRadon(), H.pixel_coordinates()

    if arguments.command == "run":
        results = {}
        for name in ARMS:
            log(f"[run] arm {name}")
            results[name], _ = run_arm(name, pool, populations, edges, coarse,
                                       xx, yy, a1)
            (OUT / "role_query_results.json").write_text(
                json.dumps({"plan": plan, "arms": results}, indent=2, default=float))
        report = {"plan": plan, "arms": results}
        reference = baseline_reference()
        drift = max(abs(results["Q0_LOCAL_XY"][str(mark)]["D2_LINE_DEV512"][key]
                        - reference[mark][key])
                    for mark in MARKS for key in reference[mark])
        report["Q0_baseline_drift"] = drift
        if drift > INIT_TOLERANCE:
            report["Q0_BASELINE_NOT_REPRODUCED"] = True
            (OUT / "role_query_results.json").write_text(
                json.dumps(report, indent=2, default=float))
            raise RuntimeError(f"Q0_BASELINE_NOT_REPRODUCED: drift {drift:.3e}")
        report["verdict"] = decide(report)
        (OUT / "role_query_results.json").write_text(json.dumps(report, indent=2,
                                                                default=float))
        log(f"[run] Q0 drift {drift:.2e}  {report['verdict']['DECISION']}")
        return

    report = json.loads((OUT / "role_query_results.json").read_text())
    if report["verdict"]["DECISION"] == "ROLE_CONDITIONED_GLOBAL_MAP_FAIL":
        raise RuntimeError("NOT_QUALIFIED: full LINE_DEV promotion is blocked")
    state = torch.load(CAP.checkpoint_path("RQ_Q1_ROLE_QUERY_GLOBAL",
                                           f"step_{max(MARKS):05d}"),
                       map_location=DEV, weights_only=False)
    model, _ = build_arm("Q1_ROLE_QUERY_GLOBAL")
    model.load_state_dict(state["model"])
    full_dev = V2.split_indices()[1]
    normal = evaluate(full_dev, model, a1, edges, coarse, xx, yy, per_role=True)
    shuffled = evaluate(full_dev, model, a1, edges, coarse, xx, yy,
                        permute=CAP.DERANGEMENT)
    promotion = {"full_line_dev": normal, "shuffled": shuffled,
                 "angle_degradation": shuffled["angle_median"] - normal["angle_median"],
                 "offset_degradation": shuffled["offset_median"] - normal["offset_median"]}
    promotion["ROLE_QUERY_SEMANTICS_CAUSAL"] = bool(
        promotion["angle_degradation"] >= CAP.SHUFFLE_ANGLE_MARGIN
        or promotion["offset_degradation"] >= CAP.SHUFFLE_OFFSET_MARGIN)
    promotion["DECISION"] = ("SUPPORTING_LINE_MAP_GO"
                             if (normal["PASS"] and normal["SAFETY"]
                                 and promotion["ROLE_QUERY_SEMANTICS_CAUSAL"])
                             else "ROLE_QUERY_SEMANTICS_NOT_CAUSAL")
    (OUT / "role_query_promotion.json").write_text(json.dumps(promotion, indent=2,
                                                              default=float))
    log(f"[promote] {promotion['DECISION']}  n={normal['n']}")


if __name__ == "__main__":
    main()
