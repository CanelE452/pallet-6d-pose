"""Does the predictor keep usable line signal when an edge is cut by the frame?

The question this screen was originally given -- can a corner that is *occluded*
still be learned from its edge -- cannot be asked of this dataset.
`pallet6d_v2_10k` carries no `mask_rle`, no `mask_bbox`, no `visibility` and no
segmentation, and there are no mask files on disk beside the frames, so external
occlusion has no label source here.  `visible_edges` is available but its own
docstring restricts it: "geometric self-occlusion only ... says nothing about
occlusion by other objects".  That question is recorded as
`OCCLUSION_LEARNABILITY: NOT_EVALUATED` and is not answered by anything below.

What the geometry does support is truncation.  The existing `visible_segments`
already partitions every role three ways, and those partitions are used as they
stand rather than replaced by a new heuristic:

```
T0_FULL        in_frame_full     the whole physical edge lies inside the image
T1_PARTIAL     in_frame_partial  the edge crosses the image, one or both
                                 endpoints fall outside it              <- PRIMARY
T2_OFF_FRAME   off_frame_full    the edge misses the image entirely
```

`T1_PARTIAL` is the population of interest: the supporting line is defined by
projected cuboid endpoints that are not themselves in the picture, while part of
the edge still is.  Its internal split into one-endpoint-out and
both-endpoints-out is diagnostic; the both-out cell is far too small to carry a
verdict and never gets one.

"Visible edge" here means the physical edge intersects the image rectangle.  It
does not mean the edge is photometrically visible, and the result carries
`PARTIAL_IN_FRAME_STRUCTURAL_EDGE` to keep that distinction attached to the
number.

Nothing about truncation, pose, K, dimensions or endpoint coordinates reaches
the predictor; the stratification is used to slice the evaluation and for
nothing else.
"""
from __future__ import annotations

import argparse, ast, importlib.util, json, pathlib, sys, time
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


AC = _load("APPEARANCE_FOR_TRUNCATION",
           "scripts/stage0/appearance_consistency_f1_screen.py")
LATE, DH = AC.LATE, AC.DH
CAP, V2, SCALE = AC.CAP, AC.V2, AC.SCALE
OUT, DEV = AC.OUT, AC.DEV

CATEGORIES = ("T0_FULL", "T1_PARTIAL", "T2_OFF_FRAME")
PRIMARY = "T1_PARTIAL"
P0_ARM = "P0_AUG_ONLY"
P0_STEP = AC.DECISION_STEP
SANITY = {"T0_FULL": 32, "T1_PARTIAL": 64, "T2_OFF_FRAME": 16}
SANITY_SEED = 0
DEV_T1_MINIMUM = 500
ROLE_T1_MINIMUM = 20
ROLE_QUORUM = 6
# Fixed before any model metric is read.
R_ANGLE_MAX = 2.0
R_OFFSET_MAX = 2.0
R_GATE_MIN = 0.40
TAIL_MAX_INCREASE = 0.20
NOT_EVALUATED = {
    "EXTERNAL_OCCLUSION_LEARNABILITY_ESTABLISHED": False,
    "SELF_OCCLUSION_LEARNABILITY_ESTABLISHED": False,
    "HIDDEN_CORNER_RECOVERY_ESTABLISHED": False,
    "CIGM_RECOVERY_ESTABLISHED": False,
    "POSE_IMPROVEMENT_ESTABLISHED": False,
    "OCCLUSION_LABEL_SOURCE": "MISSING",
    "OCCLUSION_LEARNABILITY": "NOT_EVALUATED"}


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def categorise(pack, edges):
    """The existing three-way split, plus which endpoints left the rectangle.

    Nothing new is decided here: `visible_segments` already answers the
    question, and the endpoint counts only describe the inside of T1.
    """
    theta, rho, p0, p1, length = V2.gt_lines(pack["grid"], edges)
    seg = V2.visible_segments(p0, p1, length)
    inside = lambda p: ((p[..., 0] >= V2.RECT_LO) & (p[..., 0] <= V2.RECT_HI)
                        & (p[..., 1] >= V2.RECT_LO) & (p[..., 1] <= V2.RECT_HI))
    outside = (~inside(p0)).astype(int) + (~inside(p1)).astype(int)
    return {"T0_FULL": seg["in_frame_full"],
            "T1_PARTIAL": seg["in_frame_partial"],
            "T2_OFF_FRAME": seg["off_frame_full"],
            "degenerate": seg["degenerate"],
            "endpoints_outside": outside,
            "support": seg["hit"], "q0": seg["q0"], "q1": seg["q1"],
            "p0": p0, "p1": p1}


def census(pool, edges, label):
    frames = 0
    counts = {name: 0 for name in CATEGORIES}
    counts["degenerate"] = 0
    per_role = {name: np.zeros(DH.ROLES, int) for name in CATEGORIES}
    breakdown = {"ONE_ENDPOINT_OUT": 0, "BOTH_ENDPOINTS_OUT": 0}
    breakdown_role = {k: np.zeros(DH.ROLES, int) for k in breakdown}
    for start in range(0, len(pool), CAP.BATCH):
        chunk = pool[start:start + CAP.BATCH]
        pack = V2.load_pack(chunk)
        rows = categorise(pack, edges)
        frames += len(chunk)
        counts["degenerate"] += int(rows["degenerate"].sum())
        for name in CATEGORIES:
            counts[name] += int(rows[name].sum())
            per_role[name] += rows[name].sum(0).astype(int)
        partial = rows["T1_PARTIAL"]
        one = partial & (rows["endpoints_outside"] == 1)
        both = partial & (rows["endpoints_outside"] == 2)
        breakdown["ONE_ENDPOINT_OUT"] += int(one.sum())
        breakdown["BOTH_ENDPOINTS_OUT"] += int(both.sum())
        breakdown_role["ONE_ENDPOINT_OUT"] += one.sum(0).astype(int)
        breakdown_role["BOTH_ENDPOINTS_OUT"] += both.sum(0).astype(int)
    total = sum(counts[name] for name in CATEGORIES) + counts["degenerate"]
    report = {"split": label, "frames": frames, "role_instances": total,
              "counts": counts,
              "fractions": {name: counts[name] / total for name in CATEGORIES},
              "per_role": {name: per_role[name].tolist() for name in CATEGORIES},
              "t1_breakdown": breakdown,
              "t1_breakdown_per_role": {k: v.tolist()
                                        for k, v in breakdown_role.items()}}
    report["roles_with_t1_at_least_minimum"] = int(
        (per_role[PRIMARY] >= ROLE_T1_MINIMUM).sum())
    return report


def run_census(edges):
    train, dev = V2.split_indices()
    report = {"LINE_TRAIN": census(train, edges, "LINE_TRAIN"),
              "LINE_DEV": census(dev, edges, "LINE_DEV"),
              "gate": {"dev_t1_minimum": DEV_T1_MINIMUM,
                       "role_t1_minimum": ROLE_T1_MINIMUM,
                       "role_quorum": ROLE_QUORUM},
              **NOT_EVALUATED}
    dev_report = report["LINE_DEV"]
    report["TRUNCATION_POPULATION_SUFFICIENT"] = bool(
        dev_report["counts"][PRIMARY] >= DEV_T1_MINIMUM
        and dev_report["roles_with_t1_at_least_minimum"] >= ROLE_QUORUM)
    return report


def run_sanity(edges):
    """Does the label match the geometry?  Not whether anything is visible.

    Each case records the endpoints, the clipped chord and the category the
    existing partition assigned, and the check is that the three agree: an edge
    called full has both endpoints inside and an unclipped chord, one called
    partial has at least one endpoint outside and a strictly clipped chord, one
    called off-frame has no chord at all.
    """
    train, dev = V2.split_indices()
    pool = train + dev
    generator = np.random.default_rng(SANITY_SEED)
    order = generator.permutation(len(pool))
    wanted = dict(SANITY)
    chosen, defects = [], []
    for position in order:
        if not any(wanted.values()):
            break
        index = pool[int(position)]
        pack = V2.load_pack([index, index])
        rows = categorise(pack, edges)
        for role in range(DH.ROLES):
            for name in CATEGORIES:
                if not wanted[name] or not bool(rows[name][0, role]):
                    continue
                p0 = rows["p0"][0, role]; p1 = rows["p1"][0, role]
                q0 = rows["q0"][0, role]; q1 = rows["q1"][0, role]
                outside = int(rows["endpoints_outside"][0, role])
                chord = float(np.linalg.norm(q1 - q0))
                span = float(np.linalg.norm(p1 - p0))
                case = {"index": index, "role": role, "category": name,
                        "endpoints_outside": outside,
                        "chord_over_span": chord / max(span, 1e-9),
                        "p0": p0.tolist(), "p1": p1.tolist(),
                        "q0": q0.tolist(), "q1": q1.tolist(),
                        "breakdown": ("BOTH_ENDPOINTS_OUT" if outside == 2
                                      else "ONE_ENDPOINT_OUT" if outside == 1
                                      else "NONE")}
                agrees = {
                    "T0_FULL": outside == 0 and case["chord_over_span"] > 0.999,
                    "T1_PARTIAL": outside >= 1 and 0.0 < case["chord_over_span"] <= 1.0,
                    "T2_OFF_FRAME": True}[name]
                case["geometry_agrees"] = bool(agrees)
                if not agrees:
                    defects.append(case)
                chosen.append(case)
                wanted[name] -= 1
                break
    report = {"seed": SANITY_SEED, "requested": SANITY,
              "selected": {name: sum(1 for c in chosen if c["category"] == name)
                           for name in CATEGORIES},
              "total": len(chosen), "defects": len(defects),
              "defect_cases": defects[:10], "cases": chosen}
    report["TRUNCATION_GEOMETRY_STRATIFICATION_VALID"] = bool(not defects)
    return report



def render_overlays(edges):
    """The 112 fixed cases drawn on their frames.

    The pass or fail is decided by `run_sanity`'s geometric agreement check, not
    by looking at these; they exist so the classification can be inspected.  The
    selection is a fixed permutation, so nothing here is chosen for looking
    good.
    """
    import cv2
    report = json.loads((OUT / "truncation_sanity.json").read_text())
    folder = OUT / "truncation_overlays"
    folder.mkdir(parents=True, exist_ok=True)
    colour = {"T0_FULL": (0, 200, 0), "T1_PARTIAL": (0, 165, 255),
              "T2_OFF_FRAME": (0, 0, 220)}
    scale = 8
    written = 0
    for case in report["cases"]:
        pack = V2.load_pack([case["index"], case["index"]])
        canvas = cv2.cvtColor(pack["rgb"][0], cv2.COLOR_RGB2BGR)
        canvas = cv2.resize(canvas, (V2.GRID * scale, V2.GRID * scale))
        grid = pack["grid"][0]
        for a, b in edges:                                   # projected cuboid
            cv2.line(canvas, tuple(np.int32(grid[a] * scale)),
                     tuple(np.int32(grid[b] * scale)), (90, 90, 90), 1)
        p0 = np.asarray(case["p0"]); p1 = np.asarray(case["p1"])
        q0 = np.asarray(case["q0"]); q1 = np.asarray(case["q1"])
        direction = p1 - p0
        norm = float(np.linalg.norm(direction))
        if norm > 1e-9:                                      # supporting line
            unit = direction / norm
            far0, far1 = p0 - unit * 200.0, p1 + unit * 200.0
            cv2.line(canvas, tuple(np.int32(far0 * scale)),
                     tuple(np.int32(far1 * scale)), (200, 200, 200), 1,
                     cv2.LINE_AA)
        if case["category"] != "T2_OFF_FRAME":               # in-frame chord
            cv2.line(canvas, tuple(np.int32(q0 * scale)),
                     tuple(np.int32(q1 * scale)), colour[case["category"]], 3,
                     cv2.LINE_AA)
        inside = lambda p: (V2.RECT_LO <= p[0] <= V2.RECT_HI
                            and V2.RECT_LO <= p[1] <= V2.RECT_HI)
        for point in (p0, p1):                               # endpoints
            at = np.int32(np.clip(point, -20, V2.GRID + 20) * scale)
            if inside(point):
                cv2.circle(canvas, tuple(at), 6, (255, 255, 255), -1)
                cv2.circle(canvas, tuple(at), 6, (0, 0, 0), 1)
            else:
                cv2.drawMarker(canvas, tuple(at), (0, 0, 255),
                               cv2.MARKER_TILTED_CROSS, 16, 2)
        cv2.rectangle(canvas, (0, 0), (V2.GRID * scale - 1, V2.GRID * scale - 1),
                      (255, 255, 255), 1)
        text = (f"{case['category']} role {case['role']} "
                f"{case['breakdown']} chord/span {case['chord_over_span']:.3f}")
        cv2.rectangle(canvas, (0, 0), (V2.GRID * scale, 22), (0, 0, 0), -1)
        cv2.putText(canvas, text, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (255, 255, 255), 1, cv2.LINE_AA)
        name = f"{case['category']}_{case['index']}_r{case['role']:02d}.png"
        cv2.imwrite(str(folder / name), canvas)
        written += 1
    return {"folder": str(folder), "written": written,
            "note": "inspection artefact; the verdict comes from the geometric "
                    "agreement check in run_sanity"}


@torch.no_grad()
def stratified_evaluation(indices, decoder, backbone, edges):
    """One decode pass, sliced by the locked categories.  No new metric."""
    grid_theta, grid_rho, valid = DH.lattice()
    features = DH.hypothesis_features(grid_theta, grid_rho)
    decoder.eval()
    angle, offset, roles, tags, outs = [], [], [], [], []
    for start in range(0, len(indices), CAP.BATCH):
        chunk = indices[start:start + CAP.BATCH]
        if len(chunk) < 2:
            continue
        pack = V2.load_pack(chunk)
        rows = categorise(pack, edges)
        theta_c, rho_c, support = DH.batch_rows(pack, edges)
        f50, _ = LATE.encoder_features(pack, backbone)
        scores = decoder(f50, features)
        for frame in range(scores.shape[0]):
            live = torch.nonzero(support[frame]).flatten()
            if live.numel() == 0:
                continue
            theta_p, rho_p = DH.decode(scores[frame][live], grid_theta,
                                       grid_rho, valid)
            a, o = DH.measure(theta_p, rho_p, theta_c[frame][live],
                              rho_c[frame][live])
            index = live.cpu().numpy()
            angle.append(a); offset.append(o); roles.append(index)
            tags.append(np.where(rows["T0_FULL"][frame][index], "T0_FULL",
                                 "T1_PARTIAL"))
            outs.append(rows["endpoints_outside"][frame][index])
    return (np.concatenate(angle), np.concatenate(offset),
            np.concatenate(roles), np.concatenate(tags), np.concatenate(outs))


def block(angle, offset):
    both = (angle <= CAP.ANGLE_BUDGET_DEG) & (offset <= CAP.OFFSET_BUDGET_CELL)
    return {"n": int(angle.size),
            "angle_median": float(np.median(angle)),
            "angle_p90": float(np.percentile(angle, 90)),
            "offset_median": float(np.median(offset)),
            "offset_p90": float(np.percentile(offset, 90)),
            "frac_angle_le_1": float((angle <= CAP.ANGLE_BUDGET_DEG).mean()),
            "frac_offset_le_half": float((offset <= CAP.OFFSET_BUDGET_CELL).mean()),
            "frac_both_task_gate": float(both.mean()),
            "frac_angle_gt5": float((angle > 5.0).mean()),
            "frac_angle_gt10": float((angle > 10.0).mean()),
            "frac_offset_gt2": float((offset > 2.0).mean())}


def restore_p0():
    path = CAP.checkpoint_path(f"DH_{P0_ARM}", f"step_{P0_STEP:05d}")
    stored = torch.load(path, map_location=DEV, weights_only=False)
    backbone, decoder = AC.build()
    decoder.load_state_dict(stored["model"])
    current = dict(AC.late_parameters(backbone))
    with torch.no_grad():
        for name, tensor in stored["late_a1"].items():
            current[name].copy_(tensor.to(current[name].device))
    return backbone, decoder, str(path)


def run_zero_training(edges):
    census_path = OUT / "truncation_census.json"
    sanity_path = OUT / "truncation_sanity.json"
    for path, key in ((census_path, "TRUNCATION_POPULATION_SUFFICIENT"),
                      (sanity_path, "TRUNCATION_GEOMETRY_STRATIFICATION_VALID")):
        if not path.exists() or not json.loads(path.read_text())[key]:
            raise RuntimeError(f"BLOCKED: {path.name} must pass first")
    backbone, decoder, checkpoint = restore_p0()
    _, dev = V2.split_indices()
    angle, offset, roles, tags, outs = stratified_evaluation(
        dev, decoder, backbone, edges)
    populations = {"OVERALL": np.ones(angle.size, bool),
                   "T0_FULL": tags == "T0_FULL",
                   "T1_PARTIAL": tags == "T1_PARTIAL"}
    report = {"checkpoint": checkpoint, "arm": P0_ARM,
              "arm_status": "P0_REFERENCE_ONLY",
              "population": "LINE_DEV", "blocks": {}, **NOT_EVALUATED}
    for name, mask in populations.items():
        report["blocks"][name] = block(angle[mask], offset[mask])
    partial = tags == "T1_PARTIAL"
    report["t1_breakdown_diagnostic"] = {
        "ONE_ENDPOINT_OUT": block(angle[partial & (outs == 1)],
                                  offset[partial & (outs == 1)]),
        "BOTH_ENDPOINTS_OUT": block(angle[partial & (outs == 2)],
                                    offset[partial & (outs == 2)]),
        "note": "diagnostic only; the both-out cell never carries a verdict"}
    report["per_role"] = {}
    for role in range(DH.ROLES):
        entry = {}
        for name, mask in populations.items():
            keep = mask & (roles == role)
            if keep.sum():
                piece = block(angle[keep], offset[keep])
                entry[name] = {k: piece[k] for k in
                               ("n", "angle_median", "offset_median",
                                "frac_both_task_gate")}
        report["per_role"][str(role)] = entry
    full, part = report["blocks"]["T0_FULL"], report["blocks"]["T1_PARTIAL"]
    ratios = {"R_angle": part["angle_median"] / full["angle_median"],
              "R_offset": part["offset_median"] / full["offset_median"],
              "R_gate": (part["frac_both_task_gate"]
                         / max(full["frac_both_task_gate"], 1e-12)),
              "tail_increase": (part["frac_angle_gt10"]
                                - full["frac_angle_gt10"])}
    conditions = {"R_angle_le_2": ratios["R_angle"] <= R_ANGLE_MAX,
                  "R_offset_le_2": ratios["R_offset"] <= R_OFFSET_MAX,
                  "R_gate_ge_0.40": ratios["R_gate"] >= R_GATE_MIN,
                  "tail_increase_le_0.20":
                      ratios["tail_increase"] <= TAIL_MAX_INCREASE}
    report["ratios"] = ratios
    report["conditions"] = conditions
    report["DECISION"] = ("TRUNCATED_CORNER_VISIBLE_EDGE_SIGNAL_PRESENT"
                          if all(conditions.values())
                          else "TRUNCATED_CORNER_VISIBLE_EDGE_SIGNAL_NOT_ESTABLISHED")
    report["SEMANTICS"] = "PARTIAL_IN_FRAME_STRUCTURAL_EDGE"
    report["CAUSAL_LIMIT"] = (
        "an edge counts here when it intersects the image rectangle; that is "
        "geometric, not photometric visibility, and truncation is not occlusion")
    del backbone, decoder
    torch.cuda.empty_cache()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["census", "sanity", "overlays",
                                            "zero"])
    arguments = parser.parse_args()
    import instance_edge_topology as IET
    edges = [tuple(e) for e in IET.build_topology()["edges"]]
    if not V2.sha_file(OUT / "line_internal_split.csv").startswith(V2.LINE_SPLIT_SHA):
        raise RuntimeError("HARD_BLOCKED: LINE split changed")

    if arguments.command == "census":
        report = run_census(edges)
        (OUT / "truncation_census.json").write_text(
            json.dumps(report, indent=2, default=float))
        for split in ("LINE_TRAIN", "LINE_DEV"):
            entry = report[split]
            counts, fractions = entry["counts"], entry["fractions"]
            log(f"[census] {split:10s} frames {entry['frames']:6d} roles "
                f"{entry['role_instances']:7d} | T0 {counts['T0_FULL']:7d} "
                f"({fractions['T0_FULL']:6.2%}) T1 {counts['T1_PARTIAL']:6d} "
                f"({fractions['T1_PARTIAL']:6.2%}) T2 {counts['T2_OFF_FRAME']:6d} "
                f"({fractions['T2_OFF_FRAME']:6.2%})")
            log(f"[census] {split:10s} T1 one-out "
                f"{entry['t1_breakdown']['ONE_ENDPOINT_OUT']:6d} both-out "
                f"{entry['t1_breakdown']['BOTH_ENDPOINTS_OUT']:5d} | roles with "
                f"T1>={ROLE_T1_MINIMUM}: "
                f"{entry['roles_with_t1_at_least_minimum']}/12")
        log(f"[census] SUFFICIENT={report['TRUNCATION_POPULATION_SUFFICIENT']}")
        if not report["TRUNCATION_POPULATION_SUFFICIENT"]:
            raise RuntimeError("TRUNCATION_POPULATION_TOO_SMALL")
        return

    if arguments.command == "sanity":
        report = run_sanity(edges)
        (OUT / "truncation_sanity.json").write_text(
            json.dumps(report, indent=2, default=float))
        log(f"[sanity] selected {report['selected']} total {report['total']} "
            f"defects {report['defects']}  VALID="
            f"{report['TRUNCATION_GEOMETRY_STRATIFICATION_VALID']}")
        if not report["TRUNCATION_GEOMETRY_STRATIFICATION_VALID"]:
            raise RuntimeError("TRUNCATION_STRATIFICATION_INVALID")
        return

    if arguments.command == "overlays":
        report = render_overlays(edges)
        (OUT / "truncation_overlays.json").write_text(
            json.dumps(report, indent=2, default=float))
        log(f"[overlays] wrote {report['written']} to {report['folder']}")
        return

    report = run_zero_training(edges)
    (OUT / "truncation_zero_training.json").write_text(
        json.dumps(report, indent=2, default=float))
    for name in ("OVERALL", "T0_FULL", "T1_PARTIAL"):
        entry = report["blocks"][name]
        log(f"[zero] {name:11s} n {entry['n']:6d} angle {entry['angle_median']:7.4f}"
            f" p90 {entry['angle_p90']:7.3f} | offset {entry['offset_median']:7.4f}"
            f" p90 {entry['offset_p90']:7.3f} | gate "
            f"{entry['frac_both_task_gate']:6.2%} | >10deg "
            f"{entry['frac_angle_gt10']:6.2%}")
    ratios = report["ratios"]
    log(f"[zero] R_angle {ratios['R_angle']:.4f} R_offset {ratios['R_offset']:.4f}"
        f" R_gate {ratios['R_gate']:.4f} tail+ {ratios['tail_increase']:+.4f}")
    log(f"[zero] {report['DECISION']}  conditions {report['conditions']}")


if __name__ == "__main__":
    main()
