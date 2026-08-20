"""Detached presence gate.  The pose network is frozen; only 9 numbers are learned.

The dense zero-suppression arm reduced synthetic FP sharply (-40.7% / -88.1%) but
damaged seed 2's pose (R -20.4%, t -33.4%, ADD-S -30.1%), so
`NEGATIVE_HANDLING_SUPPORTED` came out False.  That arm is kept as the historical
baseline and its lambda is not retuned.

This candidate removes the mechanism that caused the damage: the negative signal
never reaches the pose network at all.  Corner peaks are read, detached, and fed
to a linear layer.

    P0  no gate                 E3 + F3, nothing rejected
    P1  score_4kp threshold     the 4th highest corner peak, no training
    P2  detached linear         z = sorted 8 peaks -> Linear(8,1) -> sigmoid

`z` is the only feature: the eight cuboid-corner spatial maxima, sorted
descending, centroid excluded.  No new CNN, no engineered features.
"""
from __future__ import annotations

import argparse, json, pathlib, sys
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import mh_curriculum as CU  # noqa: E402
import mh_data as MD        # noqa: E402
import mh_negative as NG    # noqa: E402
import mh_poseaware as PA   # noqa: E402
import mh_screen as MS      # noqa: E402

OUT = MD.OUT
SEEDS = (1, 2)
CORNER_CHANNELS = 8
POS_TRAIN_SAMPLE = 9000        # matched to the 9,000 negatives, drawn once
SAMPLE_SEED = 20260902
TRAIN_STEPS = 2000
BATCH_PER_CLASS = 64           # balanced 1:1 minibatch
LR = 1e-2
RECALL_TARGET = 0.95


def log(m):
    print(m, flush=True)


def peaks_from(model, items, is_negative):
    """z = the 8 corner peaks, sorted descending.  Detached by construction --
    this runs under no_grad and the tensors leave the graph entirely."""
    out = []
    with torch.no_grad():
        for start in range(0, len(items), MS.BATCH):
            chunk = items[start:start + MS.BATCH]
            pack = (NG.load_negative_pack(chunk) if is_negative
                    else CU.load_pack_items(chunk))
            beliefs = CU.corner_forward(model, pack["images"])
            peaks = beliefs[-1][:, :CORNER_CHANNELS].amax(dim=(2, 3))
            ordered, _ = peaks.sort(dim=1, descending=True)
            out.append(ordered.detach().float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, CORNER_CHANNELS))


def build_cache(seed, model):
    """One forward pass over everything; every arm then reads these numbers."""
    rows = MD.load_split()
    train = [r["stem"] for r in rows if r["split"] == "MH_TRAIN"]
    dev = [r["stem"] for r in rows if r["split"] == "MH_DEV"]
    rng = np.random.default_rng(SAMPLE_SEED)
    train = sorted(rng.choice(train, size=POS_TRAIN_SAMPLE,
                              replace=False).tolist())
    cache = {}
    cache["pos_train"] = peaks_from(model, [(MD.DATA, s) for s in train], False)
    log(f"  seed{seed} pos_train {cache['pos_train'].shape}")
    cache["pos_dev"] = peaks_from(model, [(MD.DATA, s) for s in sorted(dev)], False)
    log(f"  seed{seed} pos_dev   {cache['pos_dev'].shape}")
    neg_train = NG.negative_pool("train")
    cache["neg_train"] = peaks_from(model, neg_train, True)
    log(f"  seed{seed} neg_train {cache['neg_train'].shape}")
    neg_dev = NG.negative_pool("dev")
    cache["neg_dev"] = peaks_from(model, neg_dev, True)
    log(f"  seed{seed} neg_dev   {cache['neg_dev'].shape}")
    meta = {r["stem"]: r["negative_type"] for r in json.loads(
        (OUT / "negative_filtered_manifest_dev.json").read_text())["items"]}
    cache["neg_dev_type"] = np.array([meta[s] for _, s in neg_dev])
    cache["pos_dev_stem"] = np.array(sorted(dev))
    np.savez_compressed(OUT / f"presence_z_cache_seed{seed}.npz", **cache)
    return cache


def run_cache(_a):
    MS.deterministic()
    for seed in SEEDS:
        model, _ = PA.build_model(seed)
        for p in model.parameters():
            p.requires_grad_(False)          # PHASE 0: pose network frozen
        model.eval()
        build_cache(seed, model)
    log("-> presence_z_cache_seed{1,2}.npz")




# ------------------------------------------------------------------ P1 / P2

def operating_point(pos, neg, recall_target=RECALL_TARGET):
    order = np.sort(pos)
    index = min(max(int(np.floor((1.0 - recall_target) * len(pos))), 0),
                len(pos) - 1)
    threshold = float(order[index])
    return {"threshold": threshold,
            "recall": round(float((pos >= threshold).mean()), 5),
            "fp_per_image": round(float((neg >= threshold).mean()), 5)}


def curves(pos, neg):
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    s = np.concatenate([pos, neg])
    order = np.argsort(-s)
    y = y[order]
    tp, fp = np.cumsum(y), np.cumsum(1 - y)
    recall = tp / max(y.sum(), 1)
    precision = tp / np.maximum(tp + fp, 1)
    auprc = float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))
    r = np.concatenate([pos, neg]).argsort().argsort().astype(float) + 1
    auroc = float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                  / (len(pos) * len(neg)))
    return {"AUROC": round(auroc, 5), "AUPRC": round(auprc, 5)}


def fit_linear(pos, neg, seed):
    """Linear(8,1) on the detached peaks.  Nine parameters, balanced batches."""
    torch.manual_seed(seed)
    layer = torch.nn.Linear(CORNER_CHANNELS, 1)
    optimiser = torch.optim.Adam(layer.parameters(), lr=LR)
    criterion = torch.nn.BCEWithLogitsLoss()
    P = torch.from_numpy(pos.astype(np.float32))
    N = torch.from_numpy(neg.astype(np.float32))
    rng = np.random.default_rng(seed)
    for _ in range(TRAIN_STEPS):
        pi = rng.integers(0, len(P), BATCH_PER_CLASS)
        ni = rng.integers(0, len(N), BATCH_PER_CLASS)
        x = torch.cat([P[pi], N[ni]])
        t = torch.cat([torch.ones(BATCH_PER_CLASS, 1),
                       torch.zeros(BATCH_PER_CLASS, 1)])
        loss = criterion(layer(x), t)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
    return layer, float(loss)


def apply_linear(layer, z):
    with torch.no_grad():
        return torch.sigmoid(
            layer(torch.from_numpy(z.astype(np.float32)))).squeeze(1).numpy()


def run_fit(_a):
    MS.deterministic()
    result = {"arms": ["P0_NO_GATE", "P1_SCORE4KP", "P2_DETACHED_LINEAR"],
              "feature": "z = sorted desc of 8 corner-channel spatial maxima "
                         "(centroid excluded), detached",
              "trainable_parameters": CORNER_CHANNELS + 1,
              "recall_target": RECALL_TARGET,
              "train_steps": TRAIN_STEPS, "batch_per_class": BATCH_PER_CLASS,
              "seeds": {}}
    for seed in SEEDS:
        c = np.load(OUT / f"presence_z_cache_seed{seed}.npz", allow_pickle=True)
        block = {"n_pos_dev": int(len(c["pos_dev"])),
                 "n_neg_dev": int(len(c["neg_dev"]))}
        # P0: no gate -- everything accepted
        block["P0_NO_GATE"] = {"recall": 1.0, "fp_per_image": 1.0,
                               "note": "reject 없음. FP/image = 1.0 by definition"}
        # P1: score_4kp, no training
        p1_pos = c["pos_dev"][:, 3]
        p1_neg = c["neg_dev"][:, 3]
        block["P1_SCORE4KP"] = {**curves(p1_pos, p1_neg),
                                "operating_point": operating_point(p1_pos, p1_neg)}
        # P2: detached linear
        layer, final_loss = fit_linear(c["pos_train"], c["neg_train"], seed)
        p2_pos = apply_linear(layer, c["pos_dev"])
        p2_neg = apply_linear(layer, c["neg_dev"])
        block["P2_DETACHED_LINEAR"] = {
            **curves(p2_pos, p2_neg),
            "operating_point": operating_point(p2_pos, p2_neg),
            "final_train_loss": round(final_loss, 6),
            "weights": [round(float(x), 5) for x in layer.weight.detach()[0]],
            "bias": round(float(layer.bias.detach()[0]), 5)}
        for arm, pos, neg in (("P1_SCORE4KP", p1_pos, p1_neg),
                              ("P2_DETACHED_LINEAR", p2_pos, p2_neg)):
            thr = block[arm]["operating_point"]["threshold"]
            by = {}
            for category in sorted(set(c["neg_dev_type"].tolist())):
                m = c["neg_dev_type"] == category
                by[str(category)] = {
                    "n": int(m.sum()),
                    "fp_per_image": round(float((neg[m] >= thr).mean()), 5),
                    "median": round(float(np.median(neg[m])), 5),
                    "p90": round(float(np.percentile(neg[m], 90)), 5)}
            block[arm]["by_category"] = by
        np.savez_compressed(OUT / f"presence_scores_seed{seed}.npz",
                            p1_pos=p1_pos, p1_neg=p1_neg,
                            p2_pos=p2_pos, p2_neg=p2_neg,
                            neg_type=c["neg_dev_type"])
        result["seeds"][f"seed{seed}"] = block
        for arm in ("P1_SCORE4KP", "P2_DETACHED_LINEAR"):
            e = block[arm]; o = e["operating_point"]
            log(f"seed{seed} {arm:<20} AUROC {e['AUROC']:.4f} AUPRC {e['AUPRC']:.4f} "
                f"thr {o['threshold']:.4f} recall {o['recall']:.4f} "
                f"FP/img {o['fp_per_image']:.4f}")
    (OUT / "presence_gate_result.json").write_text(json.dumps(result, indent=1))
    log("-> presence_gate_result.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["cache", "fit"])
    arguments = parser.parse_args()
    {"cache": run_cache, "fit": run_fit}[arguments.command](arguments)
