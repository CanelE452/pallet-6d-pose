"""Global content context x absolute XY position, as a 2x2 on M0.

`72f080b` left the recipe plateaued at 5.5966 degree on the holdout with the
training loss no longer moving, which is the pre-registered condition for asking
an architecture question.  Two factors, and only two:

```
P   absolute XY   the F50 grid's own x and y in [-1, 1], two channels, 1x1
                  projected and added.  No sin/cos, no learned embedding, no
                  squares, no intrinsics -- the factor is "does it know where it
                  is?" and nothing else.
G   global        GAP over F50 -> MLP -> FiLM gamma/beta.  A spatial average
                  carries no position, so G cannot smuggle P in.  No spatial
                  pyramid, no positional transformer.
```

Both added branches are zero-initialised, so at step 0 all four arms compute the
identical function and any difference later is the factor rather than the draw.

Target, MAP100, sigma, decoder, loss, roles, splits, optimizer, batch, learning
rate and step marks are all as locked.  No MAP200, no PnP, no CIGM, no
dimensions.
"""
from __future__ import annotations

import argparse, csv, importlib.util, json, pathlib, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[2]
for _e in ("scripts/stage0", "Deep_Object_Pose/common", "challenge/scripts"):
    if str(ROOT / _e) not in sys.path:
        sys.path.insert(0, str(ROOT / _e))


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SCALE = _load("SCALE_ARCH", "scripts/stage0/line/supporting_line_data_vs_step.py")
CAP, H, V2 = SCALE.CAP, SCALE.H, SCALE.V2
OUT, DEV, MAP = CAP.OUT, CAP.DEV, CAP.MAP
ARMS = {"A_G0P0": (False, False), "B_G1P0": (True, False),
        "C_G0P1": (False, True), "D_G1P1": (True, True)}
MARKS = SCALE.MARKS                                  # 1250 2500 5000 8515
BASELINE = {"angle_median": 5.5966, "offset_median": 2.2597}
REDUCTION = 0.40
THRESHOLD_40 = {"angle_median": BASELINE["angle_median"] * (1 - REDUCTION),
                "offset_median": BASELINE["offset_median"] * (1 - REDUCTION)}
INIT_TOLERANCE = 1e-6
PARITY_TOLERANCE = 1e-6
F50_CHANNELS = 128


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# ------------------------------------------------------------------ factors
class AbsoluteXY(nn.Module):
    """Two channels of normalised grid coordinate, projected and added.

    Zero-initialised, so P1 is exactly P0 at step 0.
    """

    def __init__(self, channels=F50_CHANNELS, size=MAP):
        super().__init__()
        axis = torch.linspace(-1.0, 1.0, size)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer("grid", torch.stack([xx, yy])[None], persistent=False)
        self.project = nn.Conv2d(2, channels, 1)
        nn.init.zeros_(self.project.weight); nn.init.zeros_(self.project.bias)

    def forward(self, feature):
        return feature + self.project(self.grid.expand(feature.shape[0], -1, -1, -1))


class GlobalContentContext(nn.Module):
    """GAP -> MLP -> FiLM.  A spatial average carries no position, which is the
    point: this factor must not be able to stand in for absolute XY."""

    def __init__(self, channels=F50_CHANNELS, hidden=64):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(channels, hidden), nn.ReLU(inplace=True),
                                  nn.Linear(hidden, 2 * channels))
        nn.init.zeros_(self.body[-1].weight); nn.init.zeros_(self.body[-1].bias)
        self.channels = channels

    def forward(self, feature):
        gamma, beta = self.body(feature.mean((-2, -1))).chunk(2, dim=-1)
        return feature * (1.0 + gamma[..., None, None]) + beta[..., None, None]


class ArmModel(nn.Module):
    """Frozen F50 upsampled, optional factors, then the locked map head."""

    def __init__(self, global_context, position):
        super().__init__()
        torch.manual_seed(CAP.SEED)
        self.head = CAP.SupportingLineHead(F50_CHANNELS)
        self.context = GlobalContentContext() if global_context else None
        self.position = AbsoluteXY() if position else None

    def forward(self, feature):
        if self.context is not None:
            feature = self.context(feature)
        if self.position is not None:
            feature = self.position(feature)
        return self.head(feature)


def build_arm(name):
    global_context, position = ARMS[name]
    model = ArmModel(global_context, position).to(DEV)
    return model, list(model.parameters())


def base_feature(pack, a1):
    with torch.no_grad():
        f50, _, _ = a1(pack["images"])
    return F.interpolate(f50.detach(), size=(MAP, MAP), mode="bilinear",
                         align_corners=False)


# ------------------------------------------------------------ equivalence
def init_equivalence(edges, a1, frames=32):
    """Every arm must be the same function at step 0, and A must be the recorded
    M0 -- otherwise a difference later could be the initialisation."""
    indices = V2.split_indices()[0][:frames]
    logits, losses = {}, {}
    for name in ARMS:
        model, _ = build_arm(name)
        model.eval()
        collected, loss_total = [], 0.0
        with torch.no_grad():
            for start in range(0, frames, CAP.BATCH):
                pack = V2.load_pack(indices[start:start + CAP.BATCH])
                _, _, seg, target = CAP.geometry(pack, edges)
                logit = model(base_feature(pack, a1))
                collected.append(logit.cpu())
                loss_total += float(CAP.map_loss(
                    logit, target, torch.tensor(seg["hit"], device=DEV)))
        logits[name] = torch.cat(collected)
        losses[name] = loss_total
    report = {"frames": frames, "tolerance": INIT_TOLERANCE, "pairs": {}}
    reference = logits["A_G0P0"]
    for name in ARMS:
        delta = float((logits[name] - reference).abs().max())
        report["pairs"][f"A_G0P0 vs {name}"] = delta
    report["max_pair_delta"] = max(report["pairs"].values())
    report["loss_at_init"] = losses
    report["INIT_EQUIVALENT"] = bool(report["max_pair_delta"] <= INIT_TOLERANCE)

    # legacy parity: does the new G0P0 reproduce the locked M0 head exactly?
    legacy_head, legacy_stem, _ = CAP.build_arm("M0_F50_SLINE")
    legacy_head.eval()
    new_model, _ = build_arm("A_G0P0")
    new_model.eval()
    drift = 0.0
    with torch.no_grad():
        for start in range(0, frames, CAP.BATCH):
            pack = V2.load_pack(indices[start:start + CAP.BATCH])
            feature = base_feature(pack, a1)
            drift = max(drift, float((new_model(feature)
                                      - legacy_head(feature)).abs().max()))
    report["legacy_head_drift"] = drift
    report["LEGACY_PARITY"] = bool(drift <= PARITY_TOLERANCE)
    return report


# ------------------------------------------------------------------ training
@torch.no_grad()
def evaluate(indices, model, a1, edges, coarse, xx, yy, permute=None):
    model.eval()
    angle, offset = [], []
    for start in range(0, len(indices), CAP.BATCH):
        chunk = indices[start:start + CAP.BATCH]
        if len(chunk) < 2:
            continue
        pack = V2.load_pack(chunk)
        theta, rho, seg, _ = CAP.geometry(pack, edges)
        support = torch.tensor(seg["hit"], device=DEV)
        logit = model(base_feature(pack, a1))
        if permute is not None:
            logit = logit[:, list(permute)]
        probability = torch.sigmoid(logit)
        theta_t = torch.tensor(theta, dtype=torch.float32, device=DEV)
        rho_t = torch.tensor(rho, dtype=torch.float32, device=DEV)
        a, o = CAP.decode_maps(probability, support, theta_t, rho_t, coarse, xx, yy)
        angle.append(a); offset.append(o)
    angle = np.concatenate(angle) if angle else np.zeros(1)
    offset = np.concatenate(offset) if offset else np.zeros(1)
    return {"angle_median": float(np.median(angle)),
            "angle_p90": float(np.percentile(angle, 90)),
            "offset_median": float(np.median(offset)),
            "offset_p90": float(np.percentile(offset, 90)), "n": int(angle.size),
            "PASS": bool(np.median(angle) <= CAP.ANGLE_BUDGET_DEG
                         and np.median(offset) <= CAP.OFFSET_BUDGET_CELL),
            "SAFETY": bool(np.percentile(angle, 90) <= CAP.SAFETY_ANGLE
                           and np.percentile(offset, 90) <= CAP.SAFETY_OFFSET)}


def run_arm(name, pool, populations, edges, coarse, xx, yy, a1):
    model, parameters = build_arm(name)
    optimiser = torch.optim.AdamW(parameters, lr=CAP.LR, weight_decay=CAP.WD)
    history, losses, done = {}, [], 0
    for chunk, _ in V2.step_schedule(pool, max(MARKS), CAP.BATCH):
        model.train()
        pack = V2.load_pack(chunk)
        _, _, seg, target = CAP.geometry(pack, edges)
        loss = CAP.map_loss(model(base_feature(pack, a1)), target,
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
                entry[label] = evaluate(indices, model, a1, edges, coarse, xx, yy)
                log(f"  {name} @{done:5d} {label:<16} angle med "
                    f"{entry[label]['angle_median']:7.4f} p90 "
                    f"{entry[label]['angle_p90']:7.3f} | offset med "
                    f"{entry[label]['offset_median']:7.4f}  PASS={entry[label]['PASS']}")
            torch.save({"arm": name, "step": done, "model": model.state_dict(),
                        "optimizer": optimiser.state_dict(), **CAP.provenance()},
                       CAP.checkpoint_path(f"ARCH_{name}", f"step_{done:05d}"))
            history[str(done)] = entry
            (OUT / f"trajectory_{name}.json").write_text(
                json.dumps(history, indent=2, default=float))
    return history, model


def qualify(entry):
    """Pre-registered, on D2 medians only.  Both metrics or it does not count."""
    absolute = entry["PASS"] and entry["SAFETY"]
    reduced = (entry["angle_median"] <= THRESHOLD_40["angle_median"]
               and entry["offset_median"] <= THRESHOLD_40["offset_median"])
    partial = ((entry["angle_median"] <= THRESHOLD_40["angle_median"])
               != (entry["offset_median"] <= THRESHOLD_40["offset_median"]))
    return {"ABSOLUTE_PASS": bool(absolute), "REDUCTION_40": bool(reduced),
            "PARTIAL": bool(partial and not reduced),
            "QUALIFIES": bool(absolute or reduced)}


def interpret(results):
    last = str(max(MARKS))
    final = {name: results[name][last]["D2_LINE_DEV512"] for name in ARMS}
    marks = {name: qualify(final[name]) for name in ARMS}
    qualifying = [n for n in ARMS if marks[n]["QUALIFIES"]]
    if not qualifying:
        label = ("SIMPLE_CONTEXT_POSITION_SCREEN_FAIL"
                 if not any(marks[n]["PARTIAL"] for n in ARMS)
                 else "PARTIAL_ARCHITECTURE_SIGNAL")
    elif qualifying == ["D_G1P1"]:
        label = "GLOBAL_X_POSITION_INTERACTION"
    elif set(qualifying) >= {"B_G1P0", "C_G0P1"}:
        label = "BOTH_FACTORS_CONTRIBUTE"
    elif "B_G1P0" in qualifying:
        label = "GLOBAL_CONTENT_CONDITIONING_HELPS"
    elif "C_G0P1" in qualifying:
        label = "ABSOLUTE_POSITION_MISSING"
    else:
        label = "PARTIAL_ARCHITECTURE_SIGNAL"
    pareto = []
    for name in qualifying:
        dominated = any(final[o]["angle_median"] <= final[name]["angle_median"]
                        and final[o]["offset_median"] <= final[name]["offset_median"]
                        and o != name for o in qualifying)
        if not dominated:
            pareto.append(name)
    return {"final_D2": final, "qualification": marks, "qualifying": qualifying,
            "pareto_front": pareto, "LABEL": label,
            "winner": pareto[0] if len(pareto) == 1 else None,
            "baseline": BASELINE, "threshold_40": THRESHOLD_40}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "init", "run", "shuffle"])
    arguments = parser.parse_args()
    import instance_edge_topology as IET
    edges = [tuple(e) for e in IET.build_topology()["edges"]]
    if not V2.sha_file(OUT / "line_internal_split.csv").startswith(V2.LINE_SPLIT_SHA):
        raise RuntimeError("HARD_BLOCKED: LINE split changed")
    pool = V2.split_indices()[0]
    populations = SCALE.populations()
    plan = {"arms": {k: {"global_context": v[0], "position": v[1]}
                     for k, v in ARMS.items()},
            "pool": "FULL", "pool_frames": len(pool), "marks": list(MARKS),
            "primary_population": "D2_LINE_DEV512", "baseline": BASELINE,
            "threshold_40": THRESHOLD_40, "reduction": REDUCTION,
            "absolute_budget": {"angle_median": CAP.ANGLE_BUDGET_DEG,
                                "offset_median": CAP.OFFSET_BUDGET_CELL},
            "safety": {"angle_p90": CAP.SAFETY_ANGLE,
                       "offset_p90": CAP.SAFETY_OFFSET},
            "init_tolerance": INIT_TOLERANCE, **CAP.provenance()}

    if arguments.command == "plan":
        (OUT / "architecture_screen_plan.json").write_text(json.dumps(plan, indent=2))
        log(f"[plan] pool FULL {len(pool)}  arms {list(ARMS)}  "
            f"40% -> {THRESHOLD_40['angle_median']:.5f} deg / "
            f"{THRESHOLD_40['offset_median']:.5f} cell")
        return

    a1 = V2.load_a1()
    if arguments.command == "init":
        report = init_equivalence(edges, a1)
        (OUT / "init_equivalence.json").write_text(json.dumps(report, indent=2))
        log(f"[init] max pair delta {report['max_pair_delta']:.3e}  "
            f"INIT_EQUIVALENT={report['INIT_EQUIVALENT']}")
        log(f"[init] legacy head drift {report['legacy_head_drift']:.3e}  "
            f"LEGACY_PARITY={report['LEGACY_PARITY']}")
        if not report["INIT_EQUIVALENT"]:
            raise RuntimeError("INIT_NOT_EQUIVALENT")
        return

    equivalence = OUT / "init_equivalence.json"
    if not equivalence.exists() or not json.loads(equivalence.read_text())["INIT_EQUIVALENT"]:
        raise RuntimeError("INIT_NOT_EQUIVALENT: run init first")
    coarse, (xx, yy) = H.CoarseRadon(), H.pixel_coordinates()

    if arguments.command == "run":
        results, models = {}, {}
        for name in ARMS:
            log(f"[run] arm {name}")
            results[name], models[name] = run_arm(name, pool, populations, edges,
                                                  coarse, xx, yy, a1)
            (OUT / "architecture_screen_results.json").write_text(
                json.dumps({"plan": plan, "arms": results}, indent=2, default=float))
        report = {"plan": plan, "arms": results, "interpretation": interpret(results)}
        (OUT / "architecture_screen_results.json").write_text(
            json.dumps(report, indent=2, default=float))
        log(f"[run] {report['interpretation']['LABEL']}  "
            f"qualifying {report['interpretation']['qualifying']}")
        return

    report = json.loads((OUT / "architecture_screen_results.json").read_text())
    qualifying = report["interpretation"]["qualifying"]
    if not qualifying:
        raise RuntimeError("NO_QUALIFYING_ARM: role shuffle is not run")
    shuffle = {}
    for name in qualifying:
        state = torch.load(CAP.checkpoint_path(f"ARCH_{name}", f"step_{max(MARKS):05d}"),
                           map_location=DEV, weights_only=False)
        model, _ = build_arm(name)
        model.load_state_dict(state["model"])
        normal = report["arms"][name][str(max(MARKS))]["D2_LINE_DEV512"]
        shuffled = evaluate(populations["D2_LINE_DEV512"], model, a1, edges,
                            coarse, xx, yy, permute=CAP.DERANGEMENT)
        shuffle[name] = {
            "normal": normal, "shuffled": shuffled,
            "angle_degradation": shuffled["angle_median"] - normal["angle_median"],
            "offset_degradation": shuffled["offset_median"] - normal["offset_median"]}
        shuffle[name]["ROLE_SEMANTICS_USED"] = bool(
            shuffle[name]["angle_degradation"] >= CAP.SHUFFLE_ANGLE_MARGIN
            or shuffle[name]["offset_degradation"] >= CAP.SHUFFLE_OFFSET_MARGIN)
        log(f"[shuffle] {name} angle {normal['angle_median']:.4f} -> "
            f"{shuffled['angle_median']:.4f}  used="
            f"{shuffle[name]['ROLE_SEMANTICS_USED']}")
    (OUT / "role_shuffle.json").write_text(json.dumps(shuffle, indent=2, default=float))


if __name__ == "__main__":
    main()
