"""PHASE B4/B5 -- paired bootstrap over the perturbation rows.

Everything here was fixed in `complementarity/PURPOSE.md` before any number was
seen.  Relative effect is taken against the *unperturbed* median of the same
metric, which is the only denominator that does not move between conditions.

Random controls are area-matched by construction:
    IR0, IR2  <- matched to IC's soft area
    IR1, IR3  <- matched to IE's soft area
(`masks_for` alternates the target on k % 2.)
"""
from __future__ import annotations
import json, pathlib
import numpy as np

OUT = pathlib.Path(__file__).resolve().parents[3] / "data/pallet/results/paper_s2_multihead"
C = OUT / "complementarity"
SEEDS = (1, 2)
IC_MATCHED = ("IR0", "IR2")
IE_MATCHED = ("IR1", "IR3")
REL_GATE = 0.05


def _col(rows, key):
    return np.array([r[key] for r in rows], float)


def run(resamples=10_000):
    report = {"resamples": resamples, "relative_denominator": "median(I0) of the same metric",
              "random_control_pairing": {"IC": list(IC_MATCHED), "IE": list(IE_MATCHED)},
              "gate": {"both_seeds_positive": True, "relative_effect_min": REL_GATE,
                       "bootstrap_CI_lower_gt_zero": True},
              "seeds": {}}
    for seed in SEEDS:
        d = json.load(open(C / f"perturb_seed{seed}.json"))
        rows = d["rows"]
        rng = np.random.default_rng(20260821 + seed)
        blk = {"n_frames": d["n_frames"]}
        # per-frame deltas
        delta = {}
        for metric in ("corner_rms", "line_angle", "line_offset"):
            base = _col(rows["I0"], metric)
            delta[metric] = {c: _col(rows[c], metric) - base
                             for c in rows if c != "I0"}
            blk[f"{metric}_I0_median"] = float(np.nanmedian(base))
        n = d["n_frames"]
        idx = rng.integers(0, n, (resamples, n))

        def boot(vec):
            v = np.asarray(vec, float)
            return np.nanmedian(v[idx], 1)

        def entry(vec, denom):
            b = boot(vec)
            obs = float(np.nanmedian(vec))
            return {"observed": obs, "relative": obs / denom if denom else None,
                    "CI95": [float(np.percentile(b, 2.5)),
                             float(np.percentile(b, 97.5))],
                    "CI_lower_gt_zero": bool(np.percentile(b, 2.5) > 0)}

        # main effects
        blk["main_effects"] = {}
        for metric in ("corner_rms", "line_angle", "line_offset"):
            den = blk[f"{metric}_I0_median"]
            blk["main_effects"][metric] = {
                c: entry(delta[metric][c], den) for c in delta[metric]}
        # pre-registered specificity scores
        dc, da, do = delta["corner_rms"], delta["line_angle"], delta["line_offset"]
        blk["S_corner"] = entry(dc["IC"] - dc["IE"], blk["corner_rms_I0_median"])
        blk["S_line_angle"] = entry(da["IE"] - da["IC"], blk["line_angle_I0_median"])
        blk["S_line_offset"] = entry(do["IE"] - do["IC"], blk["line_offset_I0_median"])
        # area-matched contrasts (해석용, 사전등록 gate 아님)
        blk["area_controlled"] = {
            "corner_IC_vs_IRmatched": entry(
                dc["IC"] - np.nanmean([dc[c] for c in IC_MATCHED], 0),
                blk["corner_rms_I0_median"]),
            "line_angle_IE_vs_IRmatched": entry(
                da["IE"] - np.nanmean([da[c] for c in IE_MATCHED], 0),
                blk["line_angle_I0_median"]),
            "line_offset_IE_vs_IRmatched": entry(
                do["IE"] - np.nanmean([do[c] for c in IE_MATCHED], 0),
                blk["line_offset_I0_median"]),
            "line_angle_IC_vs_IRmatched": entry(
                da["IC"] - np.nanmean([da[c] for c in IC_MATCHED], 0),
                blk["line_angle_I0_median"])}
        # mask geometry actually realised
        areas = d["areas"]
        blk["mask_geometry"] = {
            "area_IC_mean": float(np.mean([a["area_IC"] for a in areas])),
            "area_IE_mean": float(np.mean([a["area_IE"] for a in areas])),
            "IR_over_IC_median": float(np.median(
                [a["area_IR0"] / max(a["area_IC"], 1) for a in areas])),
            "IR_over_IE_median": float(np.median(
                [a["area_IR1"] / max(a["area_IE"], 1) for a in areas])),
            "soft_overlap_IC_IE_over_IC_median": float(np.median(
                [a["soft_overlap_IC_IE"] / max(a["area_IC"], 1) for a in areas]))}
        report["seeds"][f"seed{seed}"] = blk

    def passes(key):
        return all(
            report["seeds"][f"seed{s}"][key]["observed"] > 0
            and report["seeds"][f"seed{s}"][key]["relative"] >= REL_GATE
            and report["seeds"][f"seed{s}"][key]["CI_lower_gt_zero"]
            for s in SEEDS)

    corner_ok = passes("S_corner")
    line_ok = passes("S_line_angle") or passes("S_line_offset")
    # random control: IC / IE 의 주효과가 면적정합 IR 보다 커야 한다
    ctrl = all(
        report["seeds"][f"seed{s}"]["area_controlled"]["corner_IC_vs_IRmatched"]["CI_lower_gt_zero"]
        and report["seeds"][f"seed{s}"]["area_controlled"]["line_angle_IE_vs_IRmatched"]["CI_lower_gt_zero"]
        for s in SEEDS)
    report["CORNER_SPECIALIZATION"] = bool(corner_ok)
    report["LINE_SPECIALIZATION"] = bool(line_ok)
    report["RANDOM_CONTROL"] = bool(ctrl)
    report["COMPLEMENTARY_EVIDENCE_SUPPORTED"] = bool(corner_ok and line_ok and ctrl)
    json.dump(report, open(C / "perturb_bootstrap.json", "w"), indent=1,
              ensure_ascii=False)

    for s in SEEDS:
        b = report["seeds"][f"seed{s}"]
        print(f"\n=== seed{s} (n={b['n_frames']}) ===")
        print(f"{'metric':13}{'I0':>8}" + "".join(f"{c:>9}" for c in
                                                  ("IC", "IE", "IR0", "IR1", "IR2", "IR3")))
        print("-" * 68)
        for metric in ("corner_rms", "line_angle", "line_offset"):
            me = b["main_effects"][metric]
            print(f"{metric:13}{b[f'{metric}_I0_median']:>8.4f}" +
                  "".join(f"{me[c]['observed']:>+9.4f}" for c in
                          ("IC", "IE", "IR0", "IR1", "IR2", "IR3")))
        print(f"  {'score':22}{'observed':>11}{'relative':>10}{'CI low':>11}{'CI high':>11}{'>0':>5}")
        for k in ("S_corner", "S_line_angle", "S_line_offset"):
            e = b[k]
            print(f"  {k:22}{e['observed']:>+11.4f}{e['relative']:>+10.3f}"
                  f"{e['CI95'][0]:>+11.4f}{e['CI95'][1]:>+11.4f}"
                  f"{str(e['CI_lower_gt_zero']):>5}")
        for k, e in b["area_controlled"].items():
            print(f"  [면적통제] {k:30}{e['observed']:>+9.4f}  CI[{e['CI95'][0]:+.4f},"
                  f"{e['CI95'][1]:+.4f}] >0={e['CI_lower_gt_zero']}")
        g = b["mask_geometry"]
        print(f"  마스크: IC {g['area_IC_mean']:.0f}px  IE {g['area_IE_mean']:.0f}px  "
              f"IR/IC {g['IR_over_IC_median']:.3f}  IR/IE {g['IR_over_IE_median']:.3f}  "
              f"IC∩IE/IC {g['soft_overlap_IC_IE_over_IC_median']:.4f}")
    print(f"\nCORNER_SPECIALIZATION           = {report['CORNER_SPECIALIZATION']}")
    print(f"LINE_SPECIALIZATION             = {report['LINE_SPECIALIZATION']}")
    print(f"RANDOM_CONTROL                  = {report['RANDOM_CONTROL']}")
    print(f"COMPLEMENTARY_EVIDENCE_SUPPORTED = {report['COMPLEMENTARY_EVIDENCE_SUPPORTED']}")


if __name__ == "__main__":
    run()
