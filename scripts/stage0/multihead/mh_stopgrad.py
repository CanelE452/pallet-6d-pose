"""PHASE 3 -- the cheapest causal test of "does the corner gradient hurt the line?"

Three continuations from the same mature A0 checkpoint, 3,000 steps each, two
seeds.  They differ in one thing:

    E0_CONTINUE_LINE     L = L_line                       (A0 carried on)
    E1_SHARED_CORNER_LINE L = L_line + λ L_corner          (what A1 does)
    E2_STOPGRAD_CORNER   same, but the corner head reads F50.detach()

E2 is the arm the whole phase exists for.  Both heads are present and both are
trained; the only thing removed is the corner loss's path into the shared late
block.  If E2 recovers the line accuracy that E1 gives up, the problem was never
"two heads on one trunk" -- it was the corner gradient rewriting shared features.

A0 @18,000 is the source rather than @25,000: far enough in that the random
line-head transient is long gone, not so far that the run has settled into a
basin nothing can move it out of within 3,000 steps.  Chosen before running and
not revisited.

Nothing here touches the historical runners, and the 25k A0/A1 results are read
as controls, never recomputed.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import mh_arms as MH                                            # noqa: E402
import mh_cigm as CG                                            # noqa: E402
import mh_data as MD                                            # noqa: E402
import mh_screen as MS                                          # noqa: E402
from mh_arms import CAP, DH, V2                                 # noqa: E402

OUT = MD.OUT
CKPT = MD.ROOT / "weights/paper_s2/paper_s2_multihead"
LABEL = "long25k"
SOURCE_STEP = 18000                     # locked before running
STEPS = 3000
MARKS = (250, 500, 1000, 2000, 3000)
ARMS = ("E0_CONTINUE_LINE", "E1_SHARED_CORNER_LINE", "E2_STOPGRAD_CORNER")


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


class StopGradCorner(MH.MultiHeadModel):
    """A1 with one edge cut: the corner head sees F50 with the graph detached.

    Subclassed rather than flagged inside `MultiHeadModel`, so `mh_arms.ARMS`
    and the tests that pin it stay exactly as the screen left them.
    """

    def forward(self, images, hypothesis_features):
        f50, _, _ = self.a1(images)
        out = {"f50": f50,
               "line_scores": self.line(f50, hypothesis_features)}
        beliefs, affinities, _ = MH.heads_from_f50(self.net, f50.detach())
        out["beliefs"] = beliefs
        out["affinities"] = affinities
        return out


def build(arm, seed):
    """All three arms are the same weights; only the graph and the loss differ."""
    source = (CKPT / f"screen_A0_LINE_ONLY_{LABEL}_seed{seed}"
              / f"step_{SOURCE_STEP:05d}.pth")
    if not source.exists():
        raise SystemExit(f"source checkpoint missing: {source}")
    state = torch.load(source, map_location=MH.DEV, weights_only=False)
    torch.manual_seed(CAP.SEED)
    np.random.seed(CAP.SEED)
    cls = StopGradCorner if arm == "E2_STOPGRAD_CORNER" else MH.MultiHeadModel
    # A0's checkpoint has every head in it -- it was saved from the same module --
    # so the corner head arrives at the same weights in all three arms.
    model = cls("A0_LINE_ONLY" if arm == "E0_CONTINUE_LINE" else "A1_CORNER_LINE")
    model.load_state_dict(state["model"])
    return model, str(source)


def run(arm, seed, weights, populations, pool):
    grid_theta, grid_rho, valid, features = MS.lattice()
    model, source = build(arm, seed)
    optimiser = torch.optim.AdamW(model.trainable_parameters(), lr=CAP.LR,
                                  weight_decay=CAP.WD)
    history = {"source_checkpoint": source, "arm": arm, "seed": seed,
               "steps": STEPS, "marks": list(MARKS),
               "CONTINUATION_OPTIMIZER": "FRESH",
               "lambda_corner": weights["corner"]}
    path = OUT / f"stopgrad_{arm}_seed{seed}.json"
    tag_for_checkpoints = f"stopgrad_{arm}_seed{seed}"

    def evaluate_marks(step, ramp):
        entry = {"step": step, "ramp": ramp}
        for label, stems in populations.items():
            entry[label] = MS.evaluate(model, stems, features, grid_theta,
                                       grid_rho, valid)
            entry[label].pop("rows", None) if step == 0 else None
            line = entry[label]["line"]
            corner = entry[label]["corner"]
            log(f"  {arm} s{seed} @{step:5d} {label:<14} angle "
                f"{line['angle_median']:7.4f} offset {line['offset_median']:7.4f}"
                f" | cornerC {corner.get('direct_cell_median', '-')}"
                f" cornerL {corner['cigm_cell_median']}")
        history[str(step)] = entry
        MS.write_json(path, history)
        # Weights as well as metrics: the geometry audit that decides between
        # these arms needs the predictions, and a median cannot be un-medianed.
        directory = MS.CKPT / f"{tag_for_checkpoints}"
        directory.mkdir(parents=True, exist_ok=True)
        torch.save({"arm": arm, "seed": seed, "step": step,
                    "source": source, "model": model.state_dict()},
                   directory / f"step_{step:05d}.pth")

    evaluate_marks(0, 0.0)
    running = {"line": [], "corner": []}
    done = 0
    for chunk, _ in V2.step_schedule(pool, STEPS, MS.BATCH):
        model.train()
        pack = MD.load_pack(chunk)
        out = model(pack["images"], features)
        theta_c, rho_c, support = DH.batch_rows(pack, CG.EDGES)
        target = DH.target_distribution(theta_c.reshape(-1), rho_c.reshape(-1),
                                        grid_theta, grid_rho,
                                        valid).reshape(*theta_c.shape, -1)
        loss = DH.cross_entropy(out["line_scores"], target, support, valid)
        running["line"].append(float(loss.detach()))
        if arm != "E0_CONTINUE_LINE":
            corner = MH.corner_loss(out["beliefs"], pack["belief"],
                                    pack["belief_valid"])
            running["corner"].append(float(corner.detach()))
            loss = loss + weights["corner"] * corner
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        done += 1
        if done in MARKS:
            evaluate_marks(done, 1.0)
            for key, values in running.items():
                if values:
                    history[str(done)][f"train_{key}_last250"] = float(
                        np.mean(values[-250:]))
            MS.write_json(path, history)
    log(f"-> {path}")
    del model
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", default="all")
    parser.add_argument("--seed", type=int, default=1)
    arguments = parser.parse_args()

    MS.deterministic()
    weights = MS.lambdas()
    pool, populations = MD.pools()
    if arguments.seed != 1:
        import random
        pool = list(pool)
        random.Random(arguments.seed).shuffle(pool)
    arms = ARMS if arguments.arm == "all" else (arguments.arm,)
    for arm in arms:
        log(f"{arm} seed {arguments.seed}  source A0 @{SOURCE_STEP}")
        run(arm, arguments.seed, weights, populations, pool)


if __name__ == "__main__":
    main()
