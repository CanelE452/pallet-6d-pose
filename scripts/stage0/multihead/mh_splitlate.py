"""PHASE 4 -- E3_SPLIT_LATE: give each task its own copy of the late block.

Eligible because `CORNER_GRADIENT_HURTS_LINE` fired on seed 2 in PHASE 3.  The
line-side question is already closed analytically -- E2's stop-grad makes the
line branch bit-identical to E0 -- so what remains for E3 to answer is whether a
*corner-specific* late block buys corner accuracy that the shared one cannot,
and whether that costs the line anything.

    shared      vgg[:FIRST_TRAINABLE_INDEX]        frozen, as in every arm
    line late   vgg[FIRST_TRAINABLE_INDEX:] copy L trainable
    corner late vgg[FIRST_TRAINABLE_INDEX:] copy C trainable
    heads       line query + Direct Hough on L, belief stages on C

Both copies start from the same A0 @18,000 weights, so at step 0 the two branches
are bit-identical and E3's line output equals E0's.  No head is randomly
initialised anywhere.

The split index is resolved from `mh_arms.FIRST_TRAINABLE_VGG` and the real child
list, never from a copied `[19:27]` literal.
"""
from __future__ import annotations

import argparse
import copy
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
ARM = "E3_SPLIT_LATE"


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


class SplitLate(MH.MultiHeadModel):
    """Two late blocks over one frozen early trunk.

    `self.a1.vgg` keeps the line copy so every inherited helper -- the parameter
    groups, the report, the trainable set -- keeps meaning what it meant.  The
    corner copy is a separate module that only the belief stages read.
    """

    def __init__(self, arm="A1_CORNER_LINE"):
        super().__init__(arm)
        children = list(self.a1.vgg.named_children())
        split = MH.FIRST_TRAINABLE_VGG
        self.early = nn.Sequential(*[m for name, m in children
                                     if int(name) < split])
        line_late = [m for name, m in children if int(name) >= split]
        self.line_late = nn.Sequential(*line_late)
        # deepcopy, not a re-load: the two branches must be bit-identical at
        # step 0 or E3's line output would not start where E0's does.
        self.corner_late = copy.deepcopy(self.line_late)
        for parameter in self.corner_late.parameters():
            parameter.requires_grad_(True)
        self.to(MH.DEV)

    def corner_parameters(self):
        return (list(self.corner_late.parameters())
                + super().corner_parameters())

    def forward(self, images, hypothesis_features):
        with torch.no_grad():
            stem = self.early(images).detach()
        f50_line = self.line_late(stem)
        f50_corner = self.corner_late(stem)
        beliefs, affinities, _ = MH.heads_from_f50(self.net, f50_corner)
        return {"f50": f50_line,
                "line_scores": self.line(f50_line, hypothesis_features),
                "beliefs": beliefs, "affinities": affinities}


def build(seed):
    source = (SG.CKPT / f"screen_A0_LINE_ONLY_{SG.LABEL}_seed{seed}"
              / f"step_{SG.SOURCE_STEP:05d}.pth")
    if not source.exists():
        raise SystemExit(f"source checkpoint missing: {source}")
    state = torch.load(source, map_location=MH.DEV, weights_only=False)
    torch.manual_seed(CAP.SEED)
    np.random.seed(CAP.SEED)
    model = MH.MultiHeadModel("A1_CORNER_LINE")
    model.load_state_dict(state["model"])
    split = SplitLate.__new__(SplitLate)
    nn.Module.__init__(split)
    split.__dict__.update({k: v for k, v in model.__dict__.items()
                           if k not in ("_modules", "_parameters", "_buffers")})
    split._modules = model._modules
    split._parameters = model._parameters
    split._buffers = model._buffers
    children = list(split.a1.vgg.named_children())
    boundary = MH.FIRST_TRAINABLE_VGG
    split.early = nn.Sequential(*[m for n, m in children if int(n) < boundary])
    split.line_late = nn.Sequential(*[m for n, m in children
                                      if int(n) >= boundary])
    split.corner_late = copy.deepcopy(split.line_late)
    for parameter in split.corner_late.parameters():
        parameter.requires_grad_(True)
    split.to(MH.DEV)
    return split, str(source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--label", default="",
                        help="suffix for outputs; use for recovery runs so the "
                             "original result is never written over")
    arguments = parser.parse_args()

    MS.deterministic()
    weights = MS.lambdas()
    pool, populations = MD.pools()
    if arguments.seed != 1:
        import random
        pool = list(pool)
        random.Random(arguments.seed).shuffle(pool)

    grid_theta, grid_rho, valid, features = MS.lattice()
    model, source = build(arguments.seed)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(trainable, lr=CAP.LR, weight_decay=CAP.WD)
    history = {"source_checkpoint": source, "arm": ARM, "seed": arguments.seed,
               "steps": SG.STEPS, "marks": list(SG.MARKS),
               "CONTINUATION_OPTIMIZER": "FRESH",
               "lambda_corner": weights["corner"],
               "split_index": MH.FIRST_TRAINABLE_VGG,
               "trainable_params": int(sum(p.numel() for p in trainable))}
    # A recovery run must never land on the original path.  Re-running E3 to
    # regenerate its checkpoints once overwrote the real seed-1 result and lost
    # five of its six marks, because the runner writes results as it goes.
    suffix = f"_{arguments.label}" if arguments.label else ""
    path = OUT / f"splitlate_{ARM}_seed{arguments.seed}{suffix}.json"
    log(f"{ARM} seed {arguments.seed}  split at vgg[{MH.FIRST_TRAINABLE_VGG}]  "
        f"trainable {history['trainable_params']:,}")

    def evaluate_marks(step):
        entry = {"step": step}
        for label, stems in populations.items():
            entry[label] = MS.evaluate(model, stems, features, grid_theta,
                                       grid_rho, valid)
            line = entry[label]["line"]
            corner = entry[label]["corner"]
            log(f"  {ARM} s{arguments.seed} @{step:5d} {label:<14} angle "
                f"{line['angle_median']:7.4f} offset {line['offset_median']:7.4f}"
                f" | cornerC {corner.get('direct_cell_median', '-')}"
                f" cornerL {corner['cigm_cell_median']}")
        history[str(step)] = entry
        MS.write_json(path, history)
        # Save the weights too.  The first continuation runs stored only metrics,
        # and the geometry audit that decides this arm -- front_rear_shift,
        # isotropic scale, the non-affine residual -- needs the predictions
        # themselves, which cannot be recovered from a median.
        directory = MS.CKPT / f"splitlate_{ARM}_seed{arguments.seed}{suffix}"
        directory.mkdir(parents=True, exist_ok=True)
        torch.save({"arm": ARM, "seed": arguments.seed, "step": step,
                    "source": source, "model": model.state_dict()},
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


if __name__ == "__main__":
    main()
