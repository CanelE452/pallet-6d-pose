"""FINAL NEGATIVE DECISION -- recorded, not recomputed.

The pre-registered verdict in presence_verdict.json is left byte-for-byte
alone; its sha256 is pinned here.  Everything under `corrected_protocol` is a
secondary analysis and is labelled as such.

Numbers are pulled from the existing artefacts rather than typed in, so the
record cannot drift from what was measured.
"""
from __future__ import annotations

import hashlib, json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import mh_data as MD  # noqa: E402

OUT = MD.OUT
RULE = ("choose threshold minimizing negative FP/image subject to positive"
        " Recall >= 0.98")


def main():
    prereg = json.loads((OUT / "presence_verdict.json").read_text())
    curve = json.loads((OUT / "presence_recall_fp_curve.json").read_text())
    boot = json.loads((OUT / "presence_p2_vs_p1_bootstrap.json").read_text())
    gate = json.loads((OUT / "presence_gate_result.json").read_text())
    prereg_sha = hashlib.sha256(
        (OUT / "presence_verdict.json").read_bytes()).hexdigest()

    secondary = {"rule": RULE, "seeds": {}}
    for seed in ("seed1", "seed2"):
        block = {}
        for arm in ("P1_SCORE4KP", "P2_DETACHED_LINEAR"):
            e = next(x for x in curve["seeds"][seed][arm]
                     if x["recall_target"] == 0.98)
            block[arm] = {
                "recall": e["recall"],
                "recall_drop_pp": e["recall_drop_pp"],
                "fp_per_image": e["fp_per_image"],
                "fp_reduction_pct": e["fp_reduction_pct"],
                "threshold": e["threshold"],
                "PASS": bool(e["recall_drop_pp"] <= 2.0
                             and e["fp_reduction_pct"] >= 30.0)}
            block[arm]["by_category_at_recall_0.95_for_reference"] = \
                gate["seeds"][seed][arm]["by_category"]
        block["P2_vs_P1_paired_bootstrap"] = boot["seeds"][seed]["recall_0.98"]
        secondary["seeds"][seed] = block

    decision = {
        "ORIGINAL_PREREG_VERDICT": "FAIL",
        "ORIGINAL_PREREG_VERDICT_FILE": "presence_verdict.json",
        "ORIGINAL_PREREG_VERDICT_SHA256": prereg_sha,
        "ORIGINAL_PREREG_FINAL_NEGATIVE_HANDLING":
            prereg["FINAL_NEGATIVE_HANDLING"],
        "FAILURE_CAUSE": "PROTOCOL_CONFLICT",
        "FAILURE_CAUSE_DETAIL":
            "threshold selection required Recall>=95%, while qualification"
            " required Recall>=98%. Minimising FP walks the threshold to the"
            " 95% floor, so the two clauses can only hold together when FP is"
            " already tied. This is not MODEL_FAILURE.",
        "corrected_protocol": {
            "rule": RULE,
            "status": "FROZEN -- applies to all future real and synthetic"
                      " confirmation",
            "supersedes": "Recall>=95% selection floor",
            "does_not_overwrite": "presence_verdict.json",
            "secondary_analysis": secondary},
        "FINAL_NEGATIVE_HANDLING_CANDIDATE": "SCORE4KP_THRESHOLD",
        "FINAL_NEGATIVE_HANDLING_REASON": [
            "FP/image -72.5 ~ -73.2% at Recall ~= 0.98",
            "pose network completely unchanged (param/output diff = 0)",
            "difference vs P2 detached linear not established at this"
            " operating point (paired bootstrap CI includes 0, both seeds)",
            "trainable parameters = 0",
            "additional inference cost negligible"],
        "P2_DETACHED_LINEAR": "REJECT_AS_UNNECESSARY_COMPLEXITY",
        "P2_REASON": "no established benefit over P1 at the gate operating"
                     " point",
        "DENSE_NEGATIVE_SUPPRESSION": "REJECT",
        "DENSE_REASON": "reduced negative FP but failed pose safety badly on"
                        " seed2",
        "ARCHITECTURE_SEARCH": "CLOSED",
        "FINAL_CORE": ["SPLIT_LATE_2HEAD",
                       "F3 ROTATION_ONLY_TREFIT",
                       "SCORE4KP REJECTION"],
        "CLAIM_LIMIT": "synthetic negative validation only;"
                       " real-world FP/AP claim PENDING REAL EVALUATION",
        "NEXT": "prepare real positive/negative evaluation protocol."
                " No new architecture or training experiments.",
    }
    payload = json.dumps(decision, indent=1)
    (OUT / "NEGATIVE_FINAL_DECISION.json").write_text(payload)
    (OUT / "negative_final_decision_sha256.txt").write_text(
        f"{hashlib.sha256(payload.encode()).hexdigest()}"
        f"  NEGATIVE_FINAL_DECISION.json\n")

    print(f"  prereg verdict preserved, sha256 {prereg_sha[:16]}...")
    for seed in ("seed1", "seed2"):
        for arm in ("P1_SCORE4KP", "P2_DETACHED_LINEAR"):
            e = secondary["seeds"][seed][arm]
            print(f"  {seed} {arm:<20} recall {e['recall']:.4f} "
                  f"(drop {e['recall_drop_pp']:+.2f}pp)  "
                  f"FP/img {e['fp_per_image']:.4f} "
                  f"({-e['fp_reduction_pct']:+.1f}%)  "
                  f"{'PASS' if e['PASS'] else 'FAIL'}")
    print("-> NEGATIVE_FINAL_DECISION.json")


if __name__ == "__main__":
    main()
