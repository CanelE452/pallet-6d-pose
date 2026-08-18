"""E4 -- the capacity-matched control that E3's result needs.

E3 beats E2 by giving the corner branch its own late block.  That bundles two
changes, and only one of them is the hypothesis:

    (a) +5,014,912 trainable parameters for the corner branch
    (b) corner features recomputed from the frozen early trunk (256 ch) instead
        of read off the line branch's late output (128 ch)

E4 grants (a) and withholds (b):

    frozen early -> line_late -> F50 -+-> line head          (identical to E0/E2)
                                      |
                                      +-> detach -> capacity block -> corner head

So E4 has E3's parameter budget and E2's feature source.  If E4 matches E3 the
gain was capacity; if E3 still wins, the private late path is doing the work.

What this control does NOT separate: "own late path" is itself two things -- a
task-specific representation, and access to 256 early channels rather than the
line branch's 128-channel output.  E4 cannot tell those apart, and the write-up
says so rather than claiming a cleaner isolation than exists.

Two design choices, both to keep the comparison honest:

  * parameter match to 0.005%.  The block is 128->512->512->256->256->128 with
    3x3 convs, 5,015,168 parameters against E3's 5,014,912.  It needs five convs
    where E3's late block has four, because it starts from 128 channels rather
    than 256; the receptive field is therefore 11 cells against E3's 9, which is
    the one deviation from an exact operator match.
  * zero-initialised residual.  `corner_features = F50 + block(F50)` with the
    last conv at zero, so at step 0 the corner head sees exactly what it sees in
    E2 -- no randomly initialised block disturbing a pretrained head, and E4's
    step-0 corner output equals E2's by construction.  This is the same idiom
    `maskBeliefFusion` and `F50LineAdapter` already use in this repository.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import mh_arms as MH                                            # noqa: E402
import mh_cigm as CG                                            # noqa: E402
import mh_data as MD                                            # noqa: E402
import mh_screen as MS                                          # noqa: E402
import mh_stopgrad as SG                                        # noqa: E402
from mh_arms import CAP, DH, V2                                 # noqa: E402

OUT = MD.OUT
ARM = "E4_CAPACITY_MATCHED_CORNER"
WIDTHS = (128, 512, 512, 256, 256, 128)


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


class CornerCapacity(nn.Module):
    """Extra corner-only capacity on the shared F50, as a zero-init residual."""

    def __init__(self, widths=WIDTHS):
        super().__init__()
        layers = []
        for index in range(len(widths) - 1):
            layers.append(nn.Conv2d(widths[index], widths[index + 1], 3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        self.body = nn.Sequential(*layers[:-1])     # no ReLU on the residual
        final = self.body[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, features):
        return features + self.body(features)


class CapacityMatched(MH.MultiHeadModel):
    """E2's graph plus a corner-only capacity block of E3's size."""

    def __init__(self, arm="A1_CORNER_LINE"):
        super().__init__(arm)
        self.corner_capacity = CornerCapacity().to(MH.DEV)

    def corner_parameters(self):
        return list(self.corner_capacity.parameters()) + super().corner_parameters()

    def forward(self, images, hypothesis_features):
        f50, _, _ = self.a1(images)
        out = {"f50": f50,
               "line_scores": self.line(f50, hypothesis_features)}
        corner_features = self.corner_capacity(f50.detach())
        beliefs, affinities, _ = MH.heads_from_f50(self.net, corner_features)
        out["beliefs"] = beliefs
        out["affinities"] = affinities
        return out


def build(seed):
    source = (SG.CKPT / f"screen_A0_LINE_ONLY_{SG.LABEL}_seed{seed}"
              / f"step_{SG.SOURCE_STEP:05d}.pth")
    if not source.exists():
        raise SystemExit(f"source checkpoint missing: {source}")
    state = torch.load(source, map_location=MH.DEV, weights_only=False)
    torch.manual_seed(CAP.SEED)
    np.random.seed(CAP.SEED)
    model = CapacityMatched("A1_CORNER_LINE")
    model.load_state_dict(state["model"], strict=False)   # capacity block is new
    return model, str(source)


# --------------------------------------------------------------------------
# PHASE 3 wiring


def run_wiring(arguments):
    import mh_splitlate as SL
    MS.deterministic()
    grid_theta, grid_rho, valid, features = MS.lattice()
    train, populations = MD.pools()
    pack = MD.load_pack(train[:MS.BATCH])
    result = {}

    def fresh(name):
        """Each arm has its own constructor; E3 re-wraps modules after loading,
        so it cannot be built by loading a state dict into a bare class."""
        if name == "E3":
            return SL.build(1)[0]
        if name == "E4":
            return build(1)[0]
        torch.manual_seed(CAP.SEED)
        np.random.seed(CAP.SEED)
        state = torch.load(
            SG.CKPT / f"screen_A0_LINE_ONLY_{SG.LABEL}_seed1"
            / f"step_{SG.SOURCE_STEP:05d}.pth",
            map_location=MH.DEV, weights_only=False)
        cls = MH.MultiHeadModel if name == "E0" else SG.StopGradCorner
        arm = "A0_LINE_ONLY" if name == "E0" else "A1_CORNER_LINE"
        model = cls(arm)
        model.load_state_dict(state["model"])
        return model

    # T0 -- line parity against E0/E2/E3
    logits = {}
    for name in ("E0", "E2", "E3", "E4"):
        model = fresh(name)
        model.eval()
        with torch.no_grad():
            logits[name] = model(pack["images"], features)["line_scores"].clone()
        del model
    result["T0_line_parity"] = {
        f"{k}_vs_E0_max_abs": float((v - logits["E0"]).abs().max())
        for k, v in logits.items()}
    result["T0_line_parity"]["PASS"] = bool(
        result["T0_line_parity"]["E4_vs_E0_max_abs"] == 0.0)

    # T1 / T2 -- finiteness and gradient routing
    model = fresh("E4")
    model.train()
    out = model(pack["images"], features)
    theta_c, rho_c, support = DH.batch_rows(pack, CG.EDGES)
    target = DH.target_distribution(theta_c.reshape(-1), rho_c.reshape(-1),
                                    grid_theta, grid_rho,
                                    valid).reshape(*theta_c.shape, -1)
    line_loss = DH.cross_entropy(out["line_scores"], target, support, valid)
    corner_loss = MH.corner_loss(out["beliefs"], pack["belief"],
                                 pack["belief_valid"])
    groups = {
        "line_late": model.shared_parameters(),
        "line_head": model.line_parameters(),
        "corner_capacity": list(model.corner_capacity.parameters()),
        "corner_head": [p for stage in MH.TRAINABLE_BELIEF_STAGES
                        for p in getattr(model.net, f"m{stage}_2").parameters()],
        "early_frozen": [p for name, p in model.a1.vgg.named_parameters()
                         if int(name.split(".")[0]) < MH.FIRST_TRAINABLE_VGG],
    }
    result["T1_finite"] = {
        "corner_logits_finite": bool(torch.isfinite(out["beliefs"][-1]).all()),
        "line_loss_finite": bool(torch.isfinite(line_loss)),
        "corner_loss_finite": bool(torch.isfinite(corner_loss)),
        "corner_capacity_is_identity_at_step0": float(
            (model.corner_capacity(out["f50"].detach())
             - out["f50"].detach()).abs().max()),
    }
    result["T1_finite"]["PASS"] = bool(
        result["T1_finite"]["corner_logits_finite"]
        and result["T1_finite"]["line_loss_finite"]
        and result["T1_finite"]["corner_loss_finite"]
        and result["T1_finite"]["corner_capacity_is_identity_at_step0"] == 0.0)

    routing = {}
    for loss_name, loss in (("L_line", line_loss), ("L_corner", corner_loss)):
        row = {}
        for group, params in groups.items():
            live = [p for p in params if p.requires_grad]
            if not live:
                # a frozen group cannot receive gradient at all; recording it as
                # 0.0 would look like a measurement, so it is named instead
                row[group] = "frozen"
                continue
            row[group] = MH.gradient_norm(loss, live)
        routing[loss_name] = row
    result["T2_routing"] = routing
    result["T2_routing"]["PASS"] = bool(
        routing["L_line"]["line_late"] > 0
        and routing["L_line"]["line_head"] > 0
        and routing["L_line"]["corner_capacity"] == 0.0
        and routing["L_line"]["corner_head"] == 0.0
        and routing["L_corner"]["corner_capacity"] > 0
        and routing["L_corner"]["corner_head"] > 0
        and routing["L_corner"]["line_late"] == 0.0
        and routing["L_corner"]["line_head"] == 0.0
        and routing["L_line"]["early_frozen"] == "frozen"
        and routing["L_corner"]["early_frozen"] == "frozen")

    # T3 -- parameter audit
    def total(m):
        return int(sum(p.numel() for p in m.parameters() if p.requires_grad))
    e2 = fresh("E2")
    e3 = fresh("E3")
    e2_n = int(sum(p.numel() for p in e2.trainable_parameters()))
    e3_n, e4_n = total(e3), total(model)
    capacity_n = int(sum(p.numel() for p in model.corner_capacity.parameters()))
    extra_e3 = int(sum(p.numel() for p in e3.corner_late.parameters()))
    result["T3_params"] = {
        "E2_trainable": e2_n, "E3_trainable": e3_n, "E4_trainable": e4_n,
        "E3_extra_block": extra_e3, "E4_extra_block": capacity_n,
        "extra_block_diff_pct": round(
            100.0 * (capacity_n - extra_e3) / extra_e3, 4),
        "total_diff_pct": round(100.0 * (e4_n - e3_n) / e3_n, 4)}
    result["T3_params"]["PASS"] = bool(
        abs(result["T3_params"]["extra_block_diff_pct"]) <= 2.0)
    del e2, e3

    # T4 -- deterministic replay
    def replay(steps=20):
        m = fresh("E4")
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],
                                lr=CAP.LR, weight_decay=CAP.WD)
        for _ in range(steps):
            o = m(pack["images"], features)
            l = DH.cross_entropy(o["line_scores"], target, support, valid)
            c = MH.corner_loss(o["beliefs"], pack["belief"], pack["belief_valid"])
            total_loss = l + MS.lambdas()["corner"] * c
            opt.zero_grad(set_to_none=True)
            total_loss.backward()
            opt.step()
        with torch.no_grad():
            return m(pack["images"], features)["line_scores"].clone(), float(l)
    a_logits, a_loss = replay()
    b_logits, b_loss = replay()
    result["T4_replay"] = {
        "line_logits_max_abs_diff": float((a_logits - b_logits).abs().max()),
        "line_loss_diff": abs(a_loss - b_loss)}
    result["T4_replay"]["PASS"] = bool(
        result["T4_replay"]["line_logits_max_abs_diff"] == 0.0)

    result["ALL_PASS"] = all(v["PASS"] for v in result.values()
                             if isinstance(v, dict) and "PASS" in v)
    MS.write_json(OUT / "capacity_control_wiring.json", result)

    print("T0 line parity vs E0 (max abs diff)")
    for k, v in result["T0_line_parity"].items():
        if k != "PASS":
            print(f"   {k:<22} {v}")
    print(f"T1 finite / identity-at-step0  "
          f"{result['T1_finite']['corner_capacity_is_identity_at_step0']}")
    print("T2 gradient routing")
    for loss_name in ("L_line", "L_corner"):
        row = routing[loss_name]
        print(f"   {loss_name:<9} " + "  ".join(
            f"{g}={row[g]:.3e}" if isinstance(row[g], float) else f"{g}={row[g]}"
            for g in groups))
    print("T3 parameters")
    for k, v in result["T3_params"].items():
        if k != "PASS":
            print(f"   {k:<22} {v:,}" if isinstance(v, int) else
                  f"   {k:<22} {v}")
    print(f"T4 replay line logits diff  "
          f"{result['T4_replay']['line_logits_max_abs_diff']}")
    print()
    for key, value in result.items():
        if isinstance(value, dict) and "PASS" in value:
            print(f"  {key:<18} PASS={value['PASS']}")
    print("ALL_PASS =", result["ALL_PASS"])
    log(f"-> {OUT / 'capacity_control_wiring.json'}")


# --------------------------------------------------------------------------


def run_screen(arguments):
    MS.deterministic()
    weights = MS.lambdas()
    pool, populations = MD.pools()
    seed = arguments.seed
    if seed != 1:
        import random
        pool = list(pool)
        random.Random(seed).shuffle(pool)
    grid_theta, grid_rho, valid, features = MS.lattice()
    model, source = build(seed)
    optimiser = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=CAP.LR, weight_decay=CAP.WD)
    history = {"source_checkpoint": source, "arm": ARM, "seed": seed,
               "steps": SG.STEPS, "marks": list(SG.MARKS),
               "CONTINUATION_OPTIMIZER": "FRESH",
               "lambda_corner": weights["corner"],
               "trainable_params": int(sum(p.numel() for p in model.parameters()
                                           if p.requires_grad))}
    path = OUT / f"capacity_control_seed{seed}.json"
    log(f"{ARM} seed {seed}  trainable {history['trainable_params']:,}")

    def evaluate_marks(step):
        entry = {"step": step}
        for label, stems in populations.items():
            entry[label] = MS.evaluate(model, stems, features, grid_theta,
                                       grid_rho, valid)
            line = entry[label]["line"]
            corner = entry[label]["corner"]
            log(f"  {ARM} s{seed} @{step:5d} {label:<14} angle "
                f"{line['angle_median']:7.4f} offset {line['offset_median']:7.4f}"
                f" | cornerC {corner.get('direct_cell_median', '-')}"
                f" cornerL {corner['cigm_cell_median']}")
        history[str(step)] = entry
        MS.write_json(path, history)
        directory = MS.CKPT / f"capacity_{ARM}_seed{seed}"
        directory.mkdir(parents=True, exist_ok=True)
        torch.save({"arm": ARM, "seed": seed, "step": step, "source": source,
                    "model": model.state_dict()},
                   directory / f"step_{step:05d}.pth")

    evaluate_marks(0)
    done = 0
    for chunk, _ in V2.step_schedule(pool, SG.STEPS, MS.BATCH):
        model.train()
        pack = MD.load_pack(chunk)
        out = model(pack["images"], features)
        theta_c, rho_c, support = DH.batch_rows(pack, CG.EDGES)
        target = DH.target_distribution(theta_c.reshape(-1), rho_c.reshape(-1),
                                        grid_theta, grid_rho,
                                        valid).reshape(*theta_c.shape, -1)
        loss = DH.cross_entropy(out["line_scores"], target, support, valid)
        corner = MH.corner_loss(out["beliefs"], pack["belief"],
                                pack["belief_valid"])
        loss = loss + weights["corner"] * corner
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        done += 1
        if done in SG.MARKS:
            evaluate_marks(done)
    log(f"-> {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["wiring", "screen"])
    parser.add_argument("--seed", type=int, default=1)
    arguments = parser.parse_args()
    {"wiring": run_wiring, "screen": run_screen}[arguments.command](arguments)


if __name__ == "__main__":
    main()
