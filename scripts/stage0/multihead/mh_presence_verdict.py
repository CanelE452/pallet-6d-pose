"""PHASE 7-8 -- the gate, hardcoded so results cannot move it."""
from __future__ import annotations

import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import mh_data as MD  # noqa: E402

OUT = MD.OUT
GATE = {"recall_drop_pp": 2.0, "fp_reduction_pct": 30.0}
# P0 accepts everything, so its FP/image is 1.0 by definition and its recall 1.0.
P0_FP, P0_RECALL = 1.0, 1.0


def main():
    r = json.loads((OUT / "presence_gate_result.json").read_text())
    inv = json.loads((OUT / "presence_pose_invariance.json").read_text())
    lines = [f"pose invariance (param/output diff = 0): {inv['POSE_INVARIANT_ALL']}"]
    verdict = {"gate": GATE, "pose_invariant": inv["POSE_INVARIANT_ALL"],
               "per_seed": {}}
    passes = {"P1_SCORE4KP": [], "P2_DETACHED_LINEAR": []}
    for seed in (1, 2):
        b = r["seeds"][f"seed{seed}"]
        block = {}
        for arm in ("P1_SCORE4KP", "P2_DETACHED_LINEAR"):
            e = b[arm]; o = e["operating_point"]
            drop = 100.0 * (P0_RECALL - o["recall"])
            fp_red = 100.0 * (P0_FP - o["fp_per_image"]) / P0_FP
            ok = (drop <= GATE["recall_drop_pp"]
                  and fp_red >= GATE["fp_reduction_pct"]
                  and inv["POSE_INVARIANT_ALL"])
            block[arm] = {"recall": o["recall"], "recall_drop_pp": round(drop, 3),
                          "fp_per_image": o["fp_per_image"],
                          "fp_reduction_pct": round(fp_red, 2),
                          "AUPRC": e["AUPRC"], "AUROC": e["AUROC"], "PASS": ok}
            passes[arm].append(ok)
            lines.append(f"seed{seed} {arm:<20} recall {o['recall']:.4f} "
                         f"(drop {drop:+.2f}pp)  FP/img {o['fp_per_image']:.4f} "
                         f"({fp_red:+.1f}%)  AUPRC {e['AUPRC']:.4f}  "
                         f"{'PASS' if ok else 'FAIL'}")
        better = (b["P2_DETACHED_LINEAR"]["operating_point"]["fp_per_image"]
                  < b["P1_SCORE4KP"]["operating_point"]["fp_per_image"]
                  or b["P2_DETACHED_LINEAR"]["AUPRC"] > b["P1_SCORE4KP"]["AUPRC"])
        block["P2_beats_P1"] = bool(better)
        verdict["per_seed"][f"seed{seed}"] = block
    p2 = all(passes["P2_DETACHED_LINEAR"])
    p1 = all(passes["P1_SCORE4KP"])
    verdict["P2_PASS"], verdict["P1_PASS"] = p2, p1
    verdict["FINAL_NEGATIVE_HANDLING"] = (
        "DETACHED_LINEAR_PRESENCE" if p2 else
        ("SCORE4KP_THRESHOLD" if p1 else "NEGATIVE_REJECTION_NOT_ESTABLISHED"))
    lines.append("")
    lines.append(f"P2_PASS={p2}  P1_PASS={p1}")
    lines.append(f"FINAL_NEGATIVE_HANDLING = {verdict['FINAL_NEGATIVE_HANDLING']}")
    (OUT / "presence_verdict.json").write_text(json.dumps(verdict, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
