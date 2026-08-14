"""Does pose degeneracy damage supporting-line training, or only pose recovery?

A cuboid seen nearly edge-on projects its top and bottom faces onto almost the
same pixels.  That is a real problem for PnP -- the configuration approaches
coplanar and the pose stops being uniquely recoverable -- but it is not obviously
a problem for a *line* label, which is defined by the projected geometry whether
or not that geometry determines a unique pose.

Nothing is trained or deleted here.  The existing M0 epoch-5 checkpoint is
evaluated once and its errors and per-role map loss are stratified by degeneracy.

The degeneracy criterion is derived from the data with a threshold taken from an
already-locked constant, not chosen for this screen: the label's own tube sigma,
1.5 MAP100 pixel = 0.75 canonical50 cell.  Two projected faces closer than one
sigma are inside the blur of the supervision itself.
"""
from __future__ import annotations

import argparse, csv, importlib.util, json, pathlib, sys, time
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


CAP = _load("CAP_DEG", "scripts/stage0/line/supporting_line_map_capacity.py")
SEM, H, V2 = CAP.SEM, CAP.H, CAP.V2
CANON, OUT, DEV = CAP.CANON, CAP.OUT, CAP.DEV
SIGMA_CANONICAL = CAP.SIGMA_CELLS * CANON / CAP.MAP        # 0.75 canonical50 cell
LINE_EPS = 1e-4
TOP_BOTTOM = ((0, 3), (1, 2), (4, 7), (5, 6))
GROUPS = ("G0_nondegenerate", "G1_pose_coplanar", "G2_pose_collinear",
          "G3_line_invalid")


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def frame_degeneracy(corners, theta, rho, length):
    """corners (8,2) canonical50 -> group label plus the two measures.

    ``thickness``   median projected separation of the top and bottom faces.
                    As it goes to zero the eight corners collapse onto four and
                    the PnP configuration approaches coplanar.
    ``spread``      smaller singular value of the centred projected corners.
                    As it goes to zero the whole projection approaches a line.
    """
    thickness = float(np.median([np.linalg.norm(corners[a] - corners[b])
                                 for a, b in TOP_BOTTOM]))
    centred = corners - corners.mean(0)
    spread = float(np.linalg.svd(centred, compute_uv=False)[-1] / np.sqrt(len(corners)))
    line_invalid = bool((length < LINE_EPS).any()
                        or not np.isfinite(theta).all() or not np.isfinite(rho).all())
    if line_invalid:
        group = "G3_line_invalid"
    elif spread < SIGMA_CANONICAL:
        group = "G2_pose_collinear"
    elif thickness < SIGMA_CANONICAL:
        group = "G1_pose_coplanar"
    else:
        group = "G0_nondegenerate"
    return group, thickness, spread, line_invalid


def classify(indices, edges):
    import instance_edge_topology  # noqa: F401  (edges already built by caller)
    record = {}
    for index in indices:
        corners = V2.load_geometry(index)
        theta, rho, p0, p1, length = V2.gt_lines(corners[None], edges)
        group, thickness, spread, invalid = frame_degeneracy(
            corners, theta[0], rho[0], length[0])
        record[index] = {"group": group, "thickness": thickness,
                         "spread": spread, "line_invalid": invalid}
    return record


def per_role_terms(logit, target, supported):
    """Balanced positive/negative MSE per role, the same split map_loss uses."""
    probability = torch.sigmoid(logit)
    error = (probability - target) ** 2
    positive = (target > CAP.POSITIVE_THRESHOLD).float()
    negative = 1.0 - positive
    pos = (error * positive).sum((-2, -1)) / positive.sum((-2, -1)).clamp_min(1.0)
    neg = (error * negative).sum((-2, -1)) / negative.sum((-2, -1)).clamp_min(1.0)
    loss = 0.5 * pos + 0.5 * neg
    return pos, neg, loss


@torch.no_grad()
def evaluate_stratified(indices, record, edges, coarse, xx, yy, arm="M0_F50_SLINE",
                        epoch=5):
    state = torch.load(CAP.checkpoint_path(arm, f"search2k_epoch{epoch}"),
                       map_location=DEV, weights_only=False)
    head, stem, _ = CAP.build_arm(arm)
    head.load_state_dict(state["model"])
    if stem is not None:
        stem.load_state_dict(state["stem"])
    head.eval()
    a1 = V2.load_a1()
    rows = []
    for start in range(0, len(indices), CAP.BATCH):
        chunk = indices[start:start + CAP.BATCH]
        pack = V2.load_pack(chunk)
        theta, rho, seg, target = CAP.geometry(pack, edges)
        support = torch.tensor(seg["hit"], device=DEV)
        logit = head(CAP.features(pack, a1, stem))
        probability = torch.sigmoid(logit)
        pos, neg, loss = per_role_terms(logit, target, support)
        theta_t = torch.tensor(theta, dtype=torch.float32, device=DEV)
        rho_t = torch.tensor(rho, dtype=torch.float32, device=DEV)
        for frame, index in enumerate(chunk):
            live = torch.nonzero(support[frame]).flatten()
            if live.numel() == 0:
                continue
            maps = probability[frame][live].reshape(live.numel(), -1).T.contiguous()
            decoded = H.decode(maps, coarse, xx, yy)[H.PRIMARY]
            angle, offset = H.measure(decoded["normal"], decoded["rho"],
                                      theta_t[frame][live], rho_t[frame][live])
            for slot, role in enumerate(live.cpu().numpy()):
                rows.append({"index": index, "role": int(role),
                             "group": record[index]["group"],
                             "angle": float(angle[slot]),
                             "offset": float(offset[slot]),
                             "pos_mse": float(pos[frame, role]),
                             "neg_mse": float(neg[frame, role]),
                             "loss": float(loss[frame, role])})
    return rows


def summarise(rows, record, indices):
    total_loss = sum(r["loss"] for r in rows)
    report = {"n_roles": len(rows), "n_frames": len(indices),
              "sigma_canonical50_cell": SIGMA_CANONICAL, "groups": {}}
    for group in GROUPS:
        subset = [r for r in rows if r["group"] == group]
        frames = [i for i in indices if record[i]["group"] == group]
        entry = {"n_frames": len(frames), "frame_share": len(frames) / len(indices),
                 "n_roles": len(subset),
                 "role_share": len(subset) / max(len(rows), 1)}
        if subset:
            angle = np.array([r["angle"] for r in subset])
            offset = np.array([r["offset"] for r in subset])
            loss = np.array([r["loss"] for r in subset])
            entry.update({
                "angle_median": float(np.median(angle)),
                "angle_p90": float(np.percentile(angle, 90)),
                "offset_median": float(np.median(offset)),
                "offset_p90": float(np.percentile(offset, 90)),
                "map_pos_mse": float(np.mean([r["pos_mse"] for r in subset])),
                "map_neg_mse": float(np.mean([r["neg_mse"] for r in subset])),
                "loss_mean": float(loss.mean()), "loss_p90": float(np.percentile(loss, 90)),
                "loss_share": float(loss.sum() / max(total_loss, 1e-12))})
            entry["leverage"] = entry["loss_share"] / max(entry["frame_share"], 1e-12)
        report["groups"][group] = entry
    return report


def decide(report):
    groups = report["groups"]
    if groups["G3_line_invalid"]["n_frames"]:
        return "STRUCTURAL_LINE_LABEL_DEGENERACY", "ROLE_MASK"
    degenerate = [g for g in ("G1_pose_coplanar", "G2_pose_collinear")
                  if groups[g]["n_frames"]]
    if not degenerate:
        return "POSE_ONLY_DEGENERACY", "KEEP"
    if any(groups[g].get("leverage", 0.0) >= 3.0 for g in degenerate):
        return "POSE_DEGENERACY_POISONS_LINE_TRAINING", "ROLE_MASK"
    return "POSE_ONLY_DEGENERACY", "KEEP"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["classify", "stratify"])
    arguments = parser.parse_args()
    import instance_edge_topology as IET
    edges = [tuple(e) for e in IET.build_topology()["edges"]]
    if not V2.sha_file(OUT / "line_internal_split.csv").startswith(V2.LINE_SPLIT_SHA):
        raise RuntimeError("HARD_BLOCKED: LINE split changed")
    train, _ = V2.split_indices()
    pools = {"LINE_TRAIN": train, "line_search2k": V2.manifest("line_search2k"),
             "D0_SEEN512": [r["index"] for r in
                            csv.DictReader(open(OUT / "d0_seen512_manifest.csv"))]}

    if arguments.command == "classify":
        summary = {"sigma_canonical50_cell": SIGMA_CANONICAL, "pools": {}}
        for name, indices in pools.items():
            record = classify(indices, edges)
            counts = {g: sum(1 for v in record.values() if v["group"] == g)
                      for g in GROUPS}
            thickness = np.array([v["thickness"] for v in record.values()])
            spread = np.array([v["spread"] for v in record.values()])
            summary["pools"][name] = {
                "frames": len(indices), "counts": counts,
                "shares": {g: counts[g] / len(indices) for g in GROUPS},
                "thickness": {f"p{p}": float(np.percentile(thickness, p))
                              for p in (0, 1, 5, 50, 95, 100)},
                "spread": {f"p{p}": float(np.percentile(spread, p))
                           for p in (0, 1, 5, 50, 95, 100)}}
            log(f"[classify] {name:<14} frames {len(indices):6d}  " +
                "  ".join(f"{g.split('_')[0]} {counts[g]}" for g in GROUPS) +
                f"  thickness p1 {np.percentile(thickness, 1):.3f}")
        (OUT / "degeneracy_classification.json").write_text(json.dumps(summary, indent=2))
        return

    coarse, (xx, yy) = H.CoarseRadon(), H.pixel_coordinates()
    indices = pools["D0_SEEN512"]
    record = classify(indices, edges)
    rows = evaluate_stratified(indices, record, edges, coarse, xx, yy)
    report = summarise(rows, record, indices)
    report["CAUSE"], report["TRAINING_FILTER"] = decide(report)
    report.update(CAP.provenance())
    (OUT / "degeneracy_stratified.json").write_text(json.dumps(report, indent=2))
    for group in GROUPS:
        entry = report["groups"][group]
        if not entry["n_frames"]:
            log(f"[stratify] {group:<20} frames 0")
            continue
        log(f"[stratify] {group:<20} frames {entry['n_frames']:4d} "
            f"({100 * entry['frame_share']:5.2f}%) roles {entry['n_roles']:5d}  "
            f"angle med {entry['angle_median']:7.4f} p90 {entry['angle_p90']:7.3f}  "
            f"loss share {100 * entry['loss_share']:5.2f}%  leverage {entry['leverage']:.2f}")
    log(f"[stratify] {report['CAUSE']}  filter={report['TRAINING_FILTER']}")


if __name__ == "__main__":
    main()
