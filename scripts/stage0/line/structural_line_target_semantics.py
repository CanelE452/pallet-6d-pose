"""Is the target a finite segment or the supporting line it lies on?

Every structural-line screen so far has supervised a tube around the *clipped
visible segment* and then decoded with a template for the *infinite supporting
line*.  Those are different geometric objects, and the O_NUM strata point
straight at it: `border`, whose segment spans the whole chord and so matches the
template's extent, scored three times better than `interior_long`, whose segment
is shorter than the chord.

This screen changes one thing.  The decoder, the lattice, the coarse and fine
search, sigma, the gates and the population are all reused unchanged from
`structural_line_hough_decoder.py`; only the target is rebuilt from the infinite
supporting line.  No segment-aware template, no extent predictor, no half-length
-- those would put the nuisance back rather than remove it.

No model forward, no optimizer, no pose, no dimensions.
"""
from __future__ import annotations

import argparse, hashlib, importlib.util, json, math, pathlib, sys, time
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


H = _load("HOUGH_TARGET", "scripts/stage0/line/structural_line_hough_decoder.py")
SLM, V2 = H.SLM, H.V2
CANON, MAP, SIGMA_CELLS = H.CANON, H.MAP, H.SIGMA_CELLS
DEV, OUT, PRIMARY = H.DEV, H.OUT, H.PRIMARY
ONUM_GATE, OMAP_GATE = H.ONUM_GATE, H.OMAP_GATE
TARGETS = ("S0_FINITE_SEGMENT", "S1_SUPPORTING_LINE")
PRIMARY_TARGET = "S1_SUPPORTING_LINE"
THETA_BIN_DEG = 10.0


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def raster_supporting_line(theta, rho, hit, device=DEV):
    """(B, R, MAP, MAP) tube around the *infinite* line, not around a segment.

    Physical segment length never enters, so this target and the Hough template
    describe the same geometric object.
    """
    xx, yy = H.pixel_coordinates()
    theta_t = torch.as_tensor(theta, dtype=torch.float32, device=device)
    rho_t = torch.as_tensor(rho, dtype=torch.float32, device=device) * (MAP / CANON)
    normal = torch.stack([theta_t.cos(), theta_t.sin()], -1)
    distance = (normal[..., 0][..., None] * xx[None, None]
                + normal[..., 1][..., None] * yy[None, None]
                - rho_t[..., None])
    tube = torch.exp(-(distance ** 2) / (2.0 * SIGMA_CELLS ** 2))
    tube = tube.reshape(*theta_t.shape, MAP, MAP)
    return tube * torch.as_tensor(hit, device=device).float()[..., None, None]


def decode_both(q0, q1, hit, theta, rho, coarse, xx, yy):
    """Same decoder, same lattice, same gates -- only the target differs."""
    finite = SLM.raster_targets(q0, q1, hit, DEV)[0]
    infinite = raster_supporting_line(theta, rho, hit)[0]
    theta_t = torch.as_tensor(theta[0], dtype=torch.float32, device=DEV)
    rho_t = torch.as_tensor(rho[0], dtype=torch.float32, device=DEV)
    out = {}
    for name, target in (("S0_FINITE_SEGMENT", finite),
                         ("S1_SUPPORTING_LINE", infinite)):
        maps = target.reshape(target.shape[0], -1).T.contiguous()
        decoded = H.decode(maps, coarse, xx, yy)[PRIMARY]
        angle, offset = H.measure(decoded["normal"], decoded["rho"], theta_t, rho_t)
        out[name] = {"angle": angle, "offset": offset}
    return out


def summarise(angle, offset, percentile):
    return {"angle_median": float(np.median(angle)),
            f"angle_p{percentile}": float(np.percentile(angle, percentile)),
            "offset_median": float(np.median(offset)),
            f"offset_p{percentile}": float(np.percentile(offset, percentile)),
            "n": int(angle.size)}


def run_onum_target():
    """The same 10,000 synthetic lines as 1b9685c, decoded from both targets.

    short_chord is kept.  Its segment length simply stops entering the S1 map,
    which is the causal test: was its collapse a property of the line or of the
    finite-segment representation?
    """
    coarse = H.CoarseRadon()
    xx, yy = H.pixel_coordinates()
    q0, q1, theta, rho, label = H.synthetic_segments()
    gathered = {name: {"angle": [], "offset": []} for name in TARGETS}
    for start in range(0, q0.shape[1], 12):
        piece = slice(start, start + 12)
        size = q0[:, piece].shape[1]
        result = decode_both(q0[:, piece], q1[:, piece], np.ones((1, size), bool),
                             theta[:, piece], rho[:, piece], coarse, xx, yy)
        for name in TARGETS:
            for key in ("angle", "offset"):
                gathered[name][key].append(result[name][key])
    report = {"lines": int(q0.shape[1]), "targets": {}}
    degrees = np.degrees(theta[0]) % 180.0
    for name in TARGETS:
        angle = np.concatenate(gathered[name]["angle"])
        offset = np.concatenate(gathered[name]["offset"])
        entry = summarise(angle, offset, 99)
        entry["strata"] = {k: summarise(angle[label == k], offset[label == k], 99)
                           for k in np.unique(label)}
        if name == PRIMARY_TARGET:
            entry["gates"] = {k: bool(entry[k] <= v) for k, v in ONUM_GATE.items()}
            entry["ONUM_PASS"] = all(entry["gates"].values())
            entry["theta_bins"] = {}
            for low in np.arange(0.0, 180.0, THETA_BIN_DEG):
                keep = (degrees >= low) & (degrees < low + THETA_BIN_DEG)
                if keep.sum():
                    entry["theta_bins"][f"{low:.0f}-{low + THETA_BIN_DEG:.0f}"] = \
                        summarise(angle[keep], offset[keep], 99)
        report["targets"][name] = entry
    primary = report["targets"][PRIMARY_TARGET]
    report["ONUM_PASS"] = primary["ONUM_PASS"]
    report["DECISION"] = ("SUPPORTING_LINE_TARGET_NUMERICAL_VALID"
                          if report["ONUM_PASS"]
                          else "SUPPORTING_LINE_TARGET_DECODER_FAIL")
    report["Q1_interior_long_p99_le_gate"] = bool(
        primary["strata"]["interior_long"]["angle_p99"] <= ONUM_GATE["angle_p99"])
    report["Q2_short_chord_p99_le_gate"] = bool(
        primary["strata"]["short_chord"]["angle_p99"] <= ONUM_GATE["angle_p99"])
    report["FINITE_SEGMENT_EXTENT_MISMATCH_CONFIRMED"] = bool(
        report["Q1_interior_long_p99_le_gate"] and report["Q2_short_chord_p99_le_gate"])
    return report


def run_linedev(indices, edges):
    """Both targets on the whole LINE_DEV, paired per role."""
    coarse = H.CoarseRadon()
    xx, yy = H.pixel_coordinates()
    gathered = {name: {"angle": [], "offset": []} for name in TARGETS}
    border, visible, full, population = [], [], [], []
    for start in range(0, len(indices), 12):
        chunk = indices[start:start + 12]
        corners = np.stack([V2.load_geometry(i) for i in chunk])
        theta, rho, p0, p1, length = V2.gt_lines(corners, edges)
        seg = V2.visible_segments(p0, p1, length)
        for frame, index in enumerate(chunk):
            live = np.flatnonzero(seg["hit"][frame])
            if live.size == 0:
                continue
            result = decode_both(
                seg["q0"][frame][None, live], seg["q1"][frame][None, live],
                np.ones((1, live.size), bool), theta[frame][None, live],
                rho[frame][None, live], coarse, xx, yy)
            for name in TARGETS:
                for key in ("angle", "offset"):
                    gathered[name][key].append(result[name][key])
            near = np.minimum(
                np.minimum(seg["q0"][frame].min(-1), seg["q1"][frame].min(-1)),
                (CANON - 1) - np.maximum(seg["q0"][frame].max(-1),
                                         seg["q1"][frame].max(-1)))
            border.append(near[live])
            visible.append(np.linalg.norm(
                seg["q1"][frame] - seg["q0"][frame], axis=-1)[live])
            full.append(seg["in_frame_full"][frame][live])
            population.extend((index, int(r)) for r in live)
    border = np.concatenate(border); visible = np.concatenate(visible)
    full = np.concatenate(full)
    report = {"frames": len(indices), "targets": {},
              "population_sha": hashlib.sha256(repr(population).encode()).hexdigest()[:16]}
    stacked = {}
    for name in TARGETS:
        angle = np.concatenate(gathered[name]["angle"])
        offset = np.concatenate(gathered[name]["offset"])
        stacked[name] = (angle, offset)
        entry = summarise(angle, offset, 90)
        entry["gates"] = {k: bool(entry[k] <= v) for k, v in OMAP_GATE.items()}
        entry["PASS"] = all(entry["gates"].values())
        entry["cross_tab"] = {}
        for label, keep in (("A_border_ge_vis_ge", (border >= 1.5) & (visible >= 2.0)),
                            ("B_border_ge_vis_lt", (border >= 1.5) & (visible < 2.0)),
                            ("C_border_lt_vis_ge", (border < 1.5) & (visible >= 2.0)),
                            ("D_border_lt_vis_lt", (border < 1.5) & (visible < 2.0))):
            if keep.sum():
                entry["cross_tab"][label] = summarise(angle[keep], offset[keep], 90)
        for label, keep in (("in_frame_full", full), ("in_frame_partial", ~full)):
            if keep.sum():
                entry[label] = summarise(angle[keep], offset[keep], 90)
        report["targets"][name] = entry
    seg_angle, seg_offset = stacked["S0_FINITE_SEGMENT"]
    line_angle, line_offset = stacked["S1_SUPPORTING_LINE"]
    report["paired"] = {
        "angle_delta_median": float(np.median(seg_angle - line_angle)),
        "angle_delta_p90": float(np.percentile(seg_angle - line_angle, 90)),
        "offset_delta_median": float(np.median(seg_offset - line_offset)),
        "offset_delta_p90": float(np.percentile(seg_offset - line_offset, 90)),
        "line_better_fraction_angle": float((line_angle < seg_angle).mean()),
        "line_better_fraction_offset": float((line_offset < seg_offset).mean())}
    report["PRIMARY"] = PRIMARY_TARGET
    report["DECISION"] = ("SUPPORTING_LINE_HOUGH_DECODER_VALID"
                          if report["targets"][PRIMARY_TARGET]["PASS"]
                          else "SUPPORTING_LINE_HOUGH_DECODER_FAIL")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["onum-target", "linedev"])
    arguments = parser.parse_args()
    import instance_edge_topology as IET
    edges = [tuple(e) for e in IET.build_topology()["edges"]]
    if not V2.sha_file(OUT / "line_internal_split.csv").startswith(V2.LINE_SPLIT_SHA):
        raise RuntimeError("HARD_BLOCKED: LINE split changed")
    _, full_dev = V2.split_indices()

    if arguments.command == "onum-target":
        report = run_onum_target()
        (OUT / "target_semantics_onum.json").write_text(json.dumps(report, indent=2))
        for name in TARGETS:
            entry = report["targets"][name]
            log(f"[O_NUM-T] {name:<20} angle med {entry['angle_median']:.4f} p99 "
                f"{entry['angle_p99']:8.4f} | offset med {entry['offset_median']:.4f}"
                f" p99 {entry['offset_p99']:8.4f}")
        for stratum in sorted(report["targets"][PRIMARY_TARGET]["strata"]):
            s0 = report["targets"]["S0_FINITE_SEGMENT"]["strata"][stratum]
            s1 = report["targets"][PRIMARY_TARGET]["strata"][stratum]
            log(f"          {stratum:<14} S0 med {s0['angle_median']:8.4f} p99 "
                f"{s0['angle_p99']:9.4f}  ->  S1 med {s1['angle_median']:.4f} p99 "
                f"{s1['angle_p99']:.4f}")
        log(f"[O_NUM-T] Q1 interior_long {report['Q1_interior_long_p99_le_gate']}  "
            f"Q2 short_chord {report['Q2_short_chord_p99_le_gate']}  "
            f"{report['DECISION']}")
        if not report["ONUM_PASS"]:
            raise RuntimeError("SUPPORTING_LINE_TARGET_DECODER_FAIL")
        return

    onum = OUT / "target_semantics_onum.json"
    if not onum.exists() or not json.loads(onum.read_text())["ONUM_PASS"]:
        raise RuntimeError("SUPPORTING_LINE_TARGET_DECODER_FAIL: LINE_DEV is "
                           "blocked until the supporting-line O_NUM passes")
    report = run_linedev(full_dev, edges)
    (OUT / "target_semantics_linedev.json").write_text(json.dumps(report, indent=2))
    for name in TARGETS:
        entry = report["targets"][name]
        log(f"[LINE_DEV] {name:<20} angle med {entry['angle_median']:.4f} p90 "
            f"{entry['angle_p90']:.4f} | offset med {entry['offset_median']:.4f} "
            f"p90 {entry['offset_p90']:.4f}  n={entry['n']}  PASS={entry['PASS']}")
    paired = report["paired"]
    log(f"[LINE_DEV] paired angle delta med {paired['angle_delta_median']:+.4f} "
        f"line better {100 * paired['line_better_fraction_angle']:.1f}%")
    log(f"[LINE_DEV] {report['DECISION']}")


if __name__ == "__main__":
    main()
