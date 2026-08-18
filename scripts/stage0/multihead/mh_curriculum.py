"""PHASE 2-6 -- branch-specific curriculum on the locked SPLIT_LATE_2HEAD.

No architecture change.  The only new thing is that the two branches are allowed
to read *different* batches in the same step, which `SplitLate` already permits:
the early trunk is frozen and detached, and `line_late` / `corner_late` are
disjoint parameter sets, so

    grad_{line}(L_corner) = 0        grad_{corner}(L_line) = 0
    grad_{early}(anything) = 0

hold by construction rather than by tuning.  That makes the line branch a
*wiring invariant* between arms, not something to guard with a percentage.

    C0_BRANCH_CONTROL   line: BROAD        corner: BROAD
    C1_LA_FULL          line: BROAD (same) corner: BROAD + CORNER_LA_OBLIQUE_V1

Yaw convention: the dataset's own, `abs_frontal_yaw = 45 - facing_margin`.  This
is not the yaw the earlier diagnostic derived -- the two agree on only 51% of
frames -- and the dataset was targeted with this one.  Recomputing BROAD's target
cells with it reproduces the release note's counts exactly (1120 / 1116, diff 0),
which is what settles it.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import mh_arms as MH                                             # noqa: E402
import mh_cigm as CG                                             # noqa: E402
import mh_data as MD                                             # noqa: E402
import mh_diagnose as DG                                         # noqa: E402
import mh_poseaware as PA                                        # noqa: E402
import mh_screen as MS                                           # noqa: E402
import mh_splitlate as SL                                        # noqa: E402
from mh_arms import CAP, DH, V2                                  # noqa: E402

OUT = MD.OUT
CKPT = MS.CKPT
BROAD_ROOT = MD.DATA
LA_ROOT = (MD.ROOT / "data/pallet/training_data/paper_release/oblique/extracted")
LA_BUCKETS = ("corner_la_oblique_v1_y15_30", "corner_la_oblique_v1_y30_plus")

SOURCE_RUN = PA.SOURCE_RUN
SOURCE_STEP = PA.SOURCE_STEP
STEPS = PA.STEPS
MARKS = PA.MARKS
ARMS = ("C0", "C1", "C1_VCTRL")

# Locked before training.  7 BROAD + 1 LA per corner batch of 8 = 12.5%, close
# to the natural 5/(40+5) = 11.1% of simply adding the set.
LA_PER_BATCH = 1
LINE_STREAM_SEED = 20260822
CORNER_STREAM_SEED = 20260823

TARGETS = ("T1 lo|y15-30", "T2 lo|y30+", "T3 lo|y<15", "T4 other")


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# ------------------------------------------------------------------ loading

def read_label_from(root, stem):
    return json.loads((root / "labels" / f"{stem}_label.json").read_text("utf-8"))


def load_frame_from(root, stem):
    """Byte-for-byte the preprocessing `mh_data.load_frame` does, with the
    dataset root injected.  The new set uses the identical directory layout."""
    payload = read_label_from(root, stem)
    camera = payload["camera_data"]
    width, height = float(camera["width"]), float(camera["height"])
    image = cv2.imread(str(root / "rgb" / f"{stem}_rgb.png"))
    if image is None:
        raise FileNotFoundError(f"{root.name}/{stem}: no rgb")
    rgb = cv2.cvtColor(cv2.resize(image, (MD.IMAGE, MD.IMAGE)),
                       cv2.COLOR_BGR2RGB)
    normalised = (rgb.astype(np.float32) / 255.0 - MD.MEAN) / MD.STD
    obj = payload["objects"][0]
    points = np.concatenate(
        [np.asarray(obj["projected_cuboid"], float),
         np.asarray(obj["projected_cuboid_centroid"], float).reshape(1, 2)], 0)
    grid = np.stack([points[:, 0] * MD.GRID / width,
                     points[:, 1] * MD.GRID / height], 1)
    raw = cv2.imread(str(root / "mask_visible" / f"{stem}.png"),
                     cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise FileNotFoundError(f"{root.name}/{stem}: no mask_visible")
    mask = cv2.resize((raw > MD.MASK_THRESHOLD).astype(np.float32),
                      (MD.GRID, MD.GRID), interpolation=cv2.INTER_AREA)
    return normalised.transpose(2, 0, 1), rgb, grid, mask, (width, height)


def load_pack_items(items):
    """`items` is [(root, stem), ...] so the two streams can mix sources."""
    frames = [load_frame_from(root, stem) for root, stem in items]
    targets = [MD.belief_target(f[2]) for f in frames]
    return {
        "chunk": [s for _, s in items],
        "images": torch.from_numpy(np.stack([f[0] for f in frames])).to(MD.DEV),
        "grid": np.stack([f[2] for f in frames]),
        "mask": torch.from_numpy(
            np.stack([f[3] for f in frames])[:, None]).to(MD.DEV),
        "belief": torch.from_numpy(
            np.stack([t[0] for t in targets])).to(MD.DEV),
        "belief_valid": torch.from_numpy(
            np.stack([t[1] for t in targets])).to(MD.DEV),
        "resolution": [f[4] for f in frames],
    }


# ------------------------------------------------------------------ streams

def broad_pool(seed):
    rows = [r["stem"] for r in MD.load_split() if r["split"] == "MH_TRAIN"]
    pool = list(rows)
    random.Random(seed).shuffle(pool)
    return [(BROAD_ROOT, s) for s in pool]


def la_pool(seed, weights=None):
    """LA exposures, 50:50 between the two yaw buckets.

    `weights` optionally reweights within a bucket by V_vis so the LA stream can
    be matched to BROAD's target-cell visibility mix (PHASE 11).  It never
    invents a bucket that has no support.
    """
    rng = random.Random(seed)
    per_bucket = []
    for name in LA_BUCKETS:
        root = LA_ROOT / name
        stems = sorted(p.name.replace("_label.json", "")
                       for p in (root / "labels").glob("*_label.json"))
        if weights is None:
            order = list(stems)
            rng.shuffle(order)
        else:
            vvis = []
            for stem in stems:
                labels = read_label_from(root, stem)["objects"][0]
                vvis.append(int((labels.get("v2_labels") or {})
                                .get("V_vis_actual", -1)))
            weight = np.array([weights.get(v, 0.0) for v in vvis], float)
            if weight.sum() <= 0:
                raise SystemExit("V_vis reweighting has no support")
            weight = weight / weight.sum()
            draw = np.random.default_rng(seed).choice(
                len(stems), size=len(stems), replace=True, p=weight)
            order = [stems[i] for i in draw]
        per_bucket.append([(root, s) for s in order])
    mixed = []
    for a, b in zip(*per_bucket):
        mixed.extend([a, b])
    return mixed


class Stream:
    """Deterministic cycling stream with its own state, isolated per branch."""

    def __init__(self, items):
        self.items = items
        self.position = 0

    def take(self, count):
        out = []
        while len(out) < count:
            if self.position >= len(self.items):
                self.position = 0
            out.append(self.items[self.position])
            self.position += 1
        return out


def corner_stream(arm, seed, weights=None):
    """C0 draws BROAD only; C1 interleaves LA at a fixed slot in every batch."""
    broad = Stream(broad_pool(CORNER_STREAM_SEED + seed))
    if arm == "C0":
        return broad, None
    return broad, Stream(la_pool(CORNER_STREAM_SEED + 500 + seed, weights))


# ------------------------------------------------------------------ forward

def line_forward(model, images, features):
    with torch.no_grad():
        stem = model.early(images).detach()
    f50 = model.line_late(stem)
    return model.line(f50, features)


def corner_forward(model, images):
    with torch.no_grad():
        stem = model.early(images).detach()
    f50 = model.corner_late(stem)
    beliefs, _, _ = MH.heads_from_f50(model.net, f50)
    return beliefs


def build_batches(arm, corner_broad, corner_la, batch):
    take_la = 0 if arm == "C0" else LA_PER_BATCH
    items = corner_broad.take(batch - take_la)
    if take_la:
        items = items + corner_la.take(take_la)
    return items


# ------------------------------------------------------------------ parity

def run_parity(arguments):
    """PHASE 3B: 20 deterministic steps, line branch must stay bit-identical."""
    MS.deterministic()
    weights = MS.lambdas()
    grid_theta, grid_rho, valid, features = MS.lattice()
    seed = arguments.seed
    report = {"steps": 20, "seed": seed}
    traces = {}
    for arm in ("C0", "C1"):
        model, _ = PA.build_model(seed)
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimiser = torch.optim.AdamW(trainable, lr=CAP.LR,
                                      weight_decay=CAP.WD)
        line = Stream(broad_pool(LINE_STREAM_SEED + seed))
        broad, la = corner_stream(arm, seed)
        logits, losses = [], []
        for _ in range(20):
            model.train()
            line_items = line.take(MS.BATCH)
            pack_line = load_pack_items(line_items)
            scores = line_forward(model, pack_line["images"], features)
            theta_c, rho_c, support = DH.batch_rows(pack_line, CG.EDGES)
            target = DH.target_distribution(
                theta_c.reshape(-1), rho_c.reshape(-1), grid_theta, grid_rho,
                valid).reshape(*theta_c.shape, -1)
            loss_line = DH.cross_entropy(scores, target, support, valid)

            corner_items = build_batches(arm, broad, la, MS.BATCH)
            pack_corner = load_pack_items(corner_items)
            beliefs = corner_forward(model, pack_corner["images"])
            loss_corner = MH.corner_loss(beliefs, pack_corner["belief"],
                                         pack_corner["belief_valid"])

            loss = loss_line + weights["corner"] * loss_corner
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            logits.append(scores.detach().float().cpu().numpy().copy())
            losses.append(float(loss_line))
        traces[arm] = {
            "logits": np.stack(logits), "losses": np.array(losses),
            "params": torch.cat([p.detach().reshape(-1)
                                 for p in model.line_late.parameters()]
                                ).float().cpu().numpy()}
    report["max_abs_logit_diff"] = float(np.abs(
        traces["C0"]["logits"] - traces["C1"]["logits"]).max())
    report["max_abs_line_loss_diff"] = float(np.abs(
        traces["C0"]["losses"] - traces["C1"]["losses"]).max())
    report["max_abs_line_param_diff"] = float(np.abs(
        traces["C0"]["params"] - traces["C1"]["params"]).max())
    report["LINE_ISOLATION_EXACT"] = bool(
        report["max_abs_logit_diff"] == 0.0
        and report["max_abs_line_loss_diff"] == 0.0
        and report["max_abs_line_param_diff"] == 0.0)
    path = OUT / f"branch_curriculum_parity_seed{seed}.json"
    path.write_text(json.dumps(report, indent=1))
    log(json.dumps(report, indent=1))
    if not report["LINE_ISOLATION_EXACT"]:
        raise SystemExit("line parity broken -- fix wiring before training")


# ------------------------------------------------------------------ train

def run_train(arguments):
    MS.deterministic()
    lambdas = MS.lambdas()
    _, populations = MD.pools()
    grid_theta, grid_rho, valid, features = MS.lattice()
    arm, seed = arguments.arm, arguments.seed
    weights = None
    if arm == "C1_VCTRL":
        weights = json.loads(
            (OUT / "vvis_control_weights.json").read_text())["weights"]
        weights = {int(k): float(v) for k, v in weights.items()}

    model, source = PA.build_model(seed)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(trainable, lr=CAP.LR, weight_decay=CAP.WD)
    line = Stream(broad_pool(LINE_STREAM_SEED + seed))
    broad, la = corner_stream("C0" if arm == "C0" else "C1", seed, weights)

    history = {"arm": arm, "seed": seed, "source_checkpoint": source,
               "source_step": SOURCE_STEP, "steps": STEPS,
               "marks": list(MARKS), "CONTINUATION_OPTIMIZER": "FRESH",
               "la_per_batch": 0 if arm == "C0" else LA_PER_BATCH,
               "batch": MS.BATCH, "lambda_corner": lambdas["corner"],
               "yaw_convention": "45 - facing_margin (dataset definition)"}
    path = OUT / f"branch_curriculum_{arm}_seed{seed}.json"
    directory = CKPT / f"curriculum_{arm}_seed{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    log(f"{arm} seed{seed}  LA/batch {history['la_per_batch']}/{MS.BATCH}")

    def mark(step):
        entry = {"step": step}
        for label, stems in populations.items():
            entry[label] = MS.evaluate(model, stems, features, grid_theta,
                                       grid_rho, valid)
        history[str(step)] = entry
        MS.write_json(path, history)
        torch.save({"arm": arm, "seed": seed, "step": step, "source": source,
                    "model": model.state_dict()},
                   directory / f"step_{step:05d}.pth")
        line_entry = entry["D2_MH_DEV512"]["line"]
        corner_entry = entry["D2_MH_DEV512"]["corner"]
        log(f"  {arm} s{seed} @{step:5d} angle {line_entry['angle_median']:7.4f}"
            f" offset {line_entry['offset_median']:7.4f}"
            f" | cornerC {corner_entry.get('direct_cell_median', '-')}")

    mark(0)
    for step in range(1, STEPS + 1):
        model.train()
        pack_line = load_pack_items(line.take(MS.BATCH))
        scores = line_forward(model, pack_line["images"], features)
        theta_c, rho_c, support = DH.batch_rows(pack_line, CG.EDGES)
        target = DH.target_distribution(
            theta_c.reshape(-1), rho_c.reshape(-1), grid_theta, grid_rho,
            valid).reshape(*theta_c.shape, -1)
        loss = DH.cross_entropy(scores, target, support, valid)

        pack_corner = load_pack_items(
            build_batches("C0" if arm == "C0" else "C1", broad, la, MS.BATCH))
        beliefs = corner_forward(model, pack_corner["images"])
        loss = loss + lambdas["corner"] * MH.corner_loss(
            beliefs, pack_corner["belief"], pack_corner["belief_valid"])

        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        if step in MARKS:
            mark(step)
    log(f"-> {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["parity", "train"])
    parser.add_argument("--arm", default="C1", choices=list(ARMS))
    parser.add_argument("--seed", type=int, default=1)
    arguments = parser.parse_args()
    {"parity": run_parity, "train": run_train}[arguments.command](arguments)


if __name__ == "__main__":
    main()
