"""Do any two of the twelve supporting-line targets collide in supervision?

`d799101` measured whether every role has a finite line.  It does, everywhere.
That is not the same question as whether the twelve targets are separated enough
to be told apart: two roles whose lines nearly coincide produce nearly identical
maps, so the supervision asks the network for two things it cannot distinguish.

Geometry only.  Nothing trains, nothing is filtered, and both thresholds are
reused rather than invented -- 1.0 degree is the task angle budget and 0.75
canonical50 cell is the label's own tube sigma.
"""
from __future__ import annotations

import argparse, collections, csv, importlib.util, json, math, pathlib, sys, time
import numpy as np, torch

ROOT = pathlib.Path(__file__).resolve().parents[3]
for _e in ("scripts/stage0", "Deep_Object_Pose/common", "challenge/scripts"):
    if str(ROOT / _e) not in sys.path:
        sys.path.insert(0, str(ROOT / _e))


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


DEG = _load("DEG_ALIAS", "scripts/stage0/degeneracy_line_audit.py")
CAP, H, V2 = DEG.CAP, DEG.H, DEG.V2
CANON, OUT, DEV = DEG.CANON, DEG.OUT, DEG.DEV
ANGLE_THRESHOLD = CAP.ANGLE_BUDGET_DEG          # 1.0 degree, the task budget
RHO_THRESHOLD = DEG.SIGMA_CANONICAL             # 0.75 canonical50 cell, label sigma
PAIRS = [(a, b) for a in range(12) for b in range(a + 1, 12)]   # 66


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def pair_separation(theta, rho):
    """Undirected angle difference (deg) and rho separation for all 66 pairs.

    A line equals its own negation, so rho is compared after aligning the sign
    of the normals -- otherwise two identical lines written with opposite
    normals would look maximally separated.
    """
    normal = np.stack([np.cos(theta), np.sin(theta)], -1)
    angle, offset = np.empty(len(PAIRS)), np.empty(len(PAIRS))
    for k, (a, b) in enumerate(PAIRS):
        sign = 1.0 if float(normal[a] @ normal[b]) >= 0 else -1.0
        delta = float(V2.wrap_half_pi(theta[a] - theta[b] if sign > 0
                                      else theta[a] - theta[b] + math.pi))
        angle[k] = abs(math.degrees(delta))
        offset[k] = abs(rho[a] - sign * rho[b])
    return angle, offset


def audit(indices, edges, record):
    counts = collections.Counter()
    per_group = {g: {"frames": 0, "frames_with_alias": 0, "alias_pairs": 0,
                     "per_frame": [], "min_angle": [], "min_rho": []}
                 for g in DEG.GROUPS}
    aliased_frames = set()
    for index in indices:
        corners = V2.load_geometry(index)
        theta, rho, p0, p1, length = V2.gt_lines(corners[None], edges)
        angle, offset = pair_separation(theta[0], rho[0])
        aliased = (angle <= ANGLE_THRESHOLD) & (offset <= RHO_THRESHOLD)
        group = record[index]["group"]
        bucket = per_group[group]
        bucket["frames"] += 1
        bucket["alias_pairs"] += int(aliased.sum())
        bucket["per_frame"].append(int(aliased.sum()))
        bucket["min_angle"].append(float(angle.min()))
        bucket["min_rho"].append(float(offset.min()))
        if aliased.any():
            bucket["frames_with_alias"] += 1
            aliased_frames.add(index)
            for k in np.flatnonzero(aliased):
                counts[PAIRS[k]] += 1
    report = {"angle_threshold_deg": ANGLE_THRESHOLD,
              "rho_threshold_canonical50_cell": RHO_THRESHOLD,
              "pairs_per_frame": len(PAIRS), "frames": len(indices),
              "frames_with_alias": len(aliased_frames), "groups": {}}
    for group, bucket in per_group.items():
        if not bucket["frames"]:
            report["groups"][group] = {"frames": 0}
            continue
        per_frame = np.array(bucket["per_frame"])
        report["groups"][group] = {
            "frames": bucket["frames"],
            "frames_with_alias": bucket["frames_with_alias"],
            "frame_share_with_alias": bucket["frames_with_alias"] / bucket["frames"],
            "alias_pairs": bucket["alias_pairs"],
            "alias_per_frame_p50": float(np.percentile(per_frame, 50)),
            "alias_per_frame_p90": float(np.percentile(per_frame, 90)),
            "min_angle_separation_p1": float(np.percentile(bucket["min_angle"], 1)),
            "min_angle_separation_p50": float(np.percentile(bucket["min_angle"], 50)),
            "min_rho_separation_p1": float(np.percentile(bucket["min_rho"], 1)),
            "min_rho_separation_p50": float(np.percentile(bucket["min_rho"], 50))}
    report["top_pairs"] = [{"pair": list(pair), "frames": n}
                           for pair, n in counts.most_common(10)]
    report["ALIASING_PRESENT"] = bool(aliased_frames)
    return report, aliased_frames


@torch.no_grad()
def link_to_checkpoint(indices, aliased_frames, edges, coarse, xx, yy):
    """Existing M0 epoch-5 only.  Diagnostic; never used to select or filter."""
    state = torch.load(CAP.checkpoint_path("M0_F50_SLINE", "search2k_epoch5"),
                       map_location=DEV, weights_only=False)
    head, stem, _ = CAP.build_arm("M0_F50_SLINE")
    head.load_state_dict(state["model"])
    head.eval()
    a1 = V2.load_a1()
    buckets = {"with_alias": {"angle": [], "offset": []},
               "no_alias": {"angle": [], "offset": []}}
    for start in range(0, len(indices), CAP.BATCH):
        chunk = indices[start:start + CAP.BATCH]
        pack = V2.load_pack(chunk)
        theta, rho, seg, _ = CAP.geometry(pack, edges)
        support = torch.tensor(seg["hit"], device=DEV)
        probability = torch.sigmoid(head(CAP.features(pack, a1, stem)))
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
            key = "with_alias" if index in aliased_frames else "no_alias"
            buckets[key]["angle"].append(angle)
            buckets[key]["offset"].append(offset)
    out = {}
    for key, value in buckets.items():
        if not value["angle"]:
            out[key] = {"n_roles": 0}
            continue
        angle = np.concatenate(value["angle"]); offset = np.concatenate(value["offset"])
        out[key] = {"n_roles": int(angle.size),
                    "angle_median": float(np.median(angle)),
                    "angle_p90": float(np.percentile(angle, 90)),
                    "offset_median": float(np.median(offset)),
                    "offset_p90": float(np.percentile(offset, 90))}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["audit", "link"])
    arguments = parser.parse_args()
    import instance_edge_topology as IET
    edges = [tuple(e) for e in IET.build_topology()["edges"]]
    if not V2.sha_file(OUT / "line_internal_split.csv").startswith(V2.LINE_SPLIT_SHA):
        raise RuntimeError("HARD_BLOCKED: LINE split changed")
    train, _ = V2.split_indices()

    if arguments.command == "audit":
        record = DEG.classify(train, edges)
        report, aliased = audit(train, edges, record)
        (OUT / "role_target_aliasing.json").write_text(json.dumps(report, indent=2))
        log(f"[audit] LINE_TRAIN {report['frames']} frames  "
            f"frames_with_alias {report['frames_with_alias']}  "
            f"thresholds {ANGLE_THRESHOLD} deg / {RHO_THRESHOLD} cell")
        for group, entry in report["groups"].items():
            if not entry["frames"]:
                continue
            log(f"          {group:<20} frames {entry['frames']:6d} "
                f"with_alias {entry['frames_with_alias']:5d} "
                f"({100 * entry['frame_share_with_alias']:5.2f}%)  "
                f"pairs {entry['alias_pairs']:6d}  min-angle p1 "
                f"{entry['min_angle_separation_p1']:6.3f}  min-rho p1 "
                f"{entry['min_rho_separation_p1']:6.3f}")
        for entry in report["top_pairs"][:5]:
            log(f"          top pair {entry['pair']} in {entry['frames']} frames")
        return

    d0 = [row["index"] for row in
          csv.DictReader(open(OUT / "d0_seen512_manifest.csv"))]
    record = DEG.classify(d0, edges)
    report, aliased = audit(d0, edges, record)
    coarse, (xx, yy) = H.CoarseRadon(), H.pixel_coordinates()
    report["checkpoint_link"] = link_to_checkpoint(d0, aliased, edges, coarse, xx, yy)
    report["pool"] = "D0_SEEN512"
    (OUT / "role_target_aliasing_d0.json").write_text(json.dumps(report, indent=2))
    for key, entry in report["checkpoint_link"].items():
        if not entry["n_roles"]:
            log(f"[link] {key:<12} n_roles 0")
            continue
        log(f"[link] {key:<12} roles {entry['n_roles']:5d}  angle med "
            f"{entry['angle_median']:7.4f} p90 {entry['angle_p90']:7.3f} | offset med "
            f"{entry['offset_median']:7.4f} p90 {entry['offset_p90']:7.3f}")


if __name__ == "__main__":
    main()
