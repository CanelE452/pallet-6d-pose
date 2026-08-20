"""PHASE 6 -- q screen at 500 steps, seed 1 only, and the presence evaluation.

`q` is the contract; the raw weight is derived per seed as `lambda = q / r_s`
from the gradient calibration, so every arm receives the same intervention
strength rather than the same number.

The presence score is `score_4kp`: the 4th highest corner-belief peak, centroid
excluded.  Four correspondences is the minimum a pose needs, so that scalar says
"there is enough corner evidence here to attempt a pose".  No objectness head is
added.
"""
from __future__ import annotations

import argparse, json, pathlib, sys
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import mh_arms as MH        # noqa: E402
import mh_cigm as CG        # noqa: E402
import mh_curriculum as CU  # noqa: E402
import mh_data as MD        # noqa: E402
import mh_negative as NG    # noqa: E402
import mh_poseaware as PA   # noqa: E402
import mh_screen as MS      # noqa: E402
import mh_splitlate as SL   # noqa: E402
from mh_arms import CAP, DH  # noqa: E402

OUT = MD.OUT
Q_CANDIDATES = (0.05, 0.10, 0.20)
SCREEN_STEPS = 500
POS_EVAL = "D2_MH_DEV512"          # frozen positive subset for the screen
RECALL_TARGET = 0.95               # operating point constraint, fixed before results


def log(m):
    print(m, flush=True)


def lambda_for(q, seed):
    cal = json.loads((OUT / "NEG_GRAD_CALIBRATION.json").read_text())
    return q / cal["seeds"][f"seed{seed}"]["ratio"]["median"]


def train_steps(seed, q, steps):
    """One backward, one optimizer step -- identical schedule whether q is 0."""
    MS.deterministic()
    weights = MS.lambdas()
    grid_theta, grid_rho, valid, features = MS.lattice()
    lam = 0.0 if q is None else lambda_for(q, seed)
    model, _ = PA.build_model(seed)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(trainable, lr=CAP.LR, weight_decay=CAP.WD)
    line = CU.Stream(CU.broad_pool(CU.LINE_STREAM_SEED + seed))
    broad, _ = CU.corner_stream("C0", seed)
    negatives = CU.negative_stream(seed) if lam else None
    for _ in range(steps):
        model.train()
        pack_line = CU.load_pack_items(line.take(MS.BATCH))
        scores = CU.line_forward(model, pack_line["images"], features)
        theta_c, rho_c, support = DH.batch_rows(pack_line, CG.EDGES)
        target = DH.target_distribution(
            theta_c.reshape(-1), rho_c.reshape(-1), grid_theta, grid_rho,
            valid).reshape(*theta_c.shape, -1)
        loss = DH.cross_entropy(scores, target, support, valid)
        pack_corner = CU.load_pack_items(
            CU.build_batches("C0", broad, None, MS.BATCH))
        loss = loss + weights["corner"] * MH.corner_loss(
            CU.corner_forward(model, pack_corner["images"]),
            pack_corner["belief"], pack_corner["belief_valid"])
        if negatives is not None:
            pack_neg = NG.load_negative_pack(negatives.take(CU.NEG_PER_BATCH))
            loss = loss + lam * NG.negative_belief_loss(
                CU.corner_forward(model, pack_neg["images"]))
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
    return model, lam


def scores_positive(model, stems):
    out = []
    with torch.no_grad():
        for start in range(0, len(stems), MS.BATCH):
            pack = CU.load_pack_items(
                [(MD.DATA, s) for s in stems[start:start + MS.BATCH]])
            beliefs = CU.corner_forward(model, pack["images"])
            out += NG.presence_score(beliefs).cpu().numpy().tolist()
    return np.asarray(out)


def scores_negative(model, items):
    out = []
    with torch.no_grad():
        for start in range(0, len(items), MS.BATCH):
            pack = NG.load_negative_pack(items[start:start + MS.BATCH])
            beliefs = CU.corner_forward(model, pack["images"])
            out += NG.presence_score(beliefs).cpu().numpy().tolist()
    return np.asarray(out)


def presence_metrics(pos, neg):
    """Threshold-free first; the operating point is chosen afterwards."""
    y = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
    s = np.concatenate([pos, neg])
    order = np.argsort(-s)
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    recall = tp / max(y.sum(), 1)
    precision = tp / np.maximum(tp + fp, 1)
    auprc = float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))
    ranks = np.empty(len(s))
    ranks[order] = np.arange(len(s))
    pos_rank = ranks[:0]
    # AUROC via the Mann-Whitney statistic on the original arrays
    allv = np.concatenate([pos, neg])
    r = allv.argsort().argsort().astype(float) + 1
    rpos = r[:len(pos)].sum()
    auroc = float((rpos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
    return {"AUROC": round(auroc, 5), "AUPRC": round(auprc, 5)}


def operating_point(pos, neg, recall_target=RECALL_TARGET):
    """Lowest FP/image among thresholds that keep positive recall at target."""
    order = np.sort(pos)
    index = int(np.floor((1.0 - recall_target) * len(pos)))
    index = min(max(index, 0), len(pos) - 1)
    threshold = float(order[index])
    recall = float((pos >= threshold).mean())
    fp_per_image = float((neg >= threshold).mean())
    return {"threshold": threshold, "recall": round(recall, 5),
            "fp_per_image": round(fp_per_image, 5),
            "recall_target": recall_target}


def run_screen(_a):
    seed = 1
    pos_stems = json.loads(
        (OUT / f"{POS_EVAL.lower()}_manifest.json").read_text())["stems"]
    neg_items = NG.negative_pool("dev")
    neg_meta = {row["stem"]: row["negative_type"] for row in json.loads(
        (OUT / "negative_filtered_manifest_dev.json").read_text())["items"]}
    result = {"steps": SCREEN_STEPS, "seed": seed, "q_candidates": list(Q_CANDIDATES),
              "positive_eval": POS_EVAL, "n_pos": len(pos_stems),
              "negative_eval": "negative_synth_v1_dev", "n_neg": len(neg_items),
              "recall_target": RECALL_TARGET, "arms": {}}
    baseline = None
    for q in (None,) + Q_CANDIDATES:
        name = "N0" if q is None else f"q={q:g}"
        model, lam = train_steps(seed, q, SCREEN_STEPS)
        model.eval()
        pos = scores_positive(model, pos_stems)
        neg = scores_negative(model, neg_items)
        entry = {"lambda": lam, **presence_metrics(pos, neg)}
        entry["operating_point"] = operating_point(pos, neg)
        entry["pos_score"] = {"median": float(np.median(pos)),
                              "p10": float(np.percentile(pos, 10))}
        entry["neg_score"] = {"median": float(np.median(neg)),
                              "p90": float(np.percentile(neg, 90))}
        by = {}
        for stem, category in neg_meta.items():
            by.setdefault(category, [])
        for (root, stem), score in zip(neg_items, neg):
            by[neg_meta[stem]].append(float(score))
        entry["neg_by_category"] = {
            k: {"n": len(v), "median": round(float(np.median(v)), 5),
                "p90": round(float(np.percentile(v, 90)), 5)}
            for k, v in sorted(by.items())}
        np.savez_compressed(OUT / f"neg_screen_scores_{name.replace('=','')}.npz",
                            pos=pos, neg=neg,
                            neg_type=np.array([neg_meta[s] for _, s in neg_items]))
        if q is None:
            baseline = entry
        else:
            entry["recall_drop_pp"] = round(
                100.0 * (baseline["operating_point"]["recall"]
                         - entry["operating_point"]["recall"]), 3)
            entry["fp_reduction_pct"] = round(100.0 * (
                baseline["operating_point"]["fp_per_image"]
                - entry["operating_point"]["fp_per_image"])
                / max(baseline["operating_point"]["fp_per_image"], 1e-9), 2)
        result["arms"][name] = entry
        op = entry["operating_point"]
        log(f"  {name:<8} lam {lam:.6f}  AUROC {entry['AUROC']:.4f} "
            f"AUPRC {entry['AUPRC']:.4f}  thr {op['threshold']:.4f} "
            f"recall {op['recall']:.3f}  FP/img {op['fp_per_image']:.4f}")
    (OUT / "neg_lambda_screen.json").write_text(json.dumps(result, indent=1))
    log("-> neg_lambda_screen.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["screen"])
    run_screen(p.parse_args())
