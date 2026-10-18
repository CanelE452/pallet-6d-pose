"""Post-scoring cross-tabs and tail changes; never imported by prediction."""
from __future__ import annotations
import itertools
from .registry import SEEDS, contrasts
from scripts.research.pallet_risk_information_v1.common import csv_write, save_rows


def export_details(output, subsets, registered, ledger, augmentation):
    tails, cross, decisions = [], [], []
    for population, records in subsets.items():
        for before, after in contrasts(augmentation):
            for axis, threshold in (("translation_cm", 50), ("translation_cm", 100),
                                    ("rotation_deg", 45)):
                for row in records:
                    b, a = row["metrics"][before], row["metrics"][after]
                    bv, av = b[axis], a[axis]
                    bt, at = bv is not None and bv > threshold, av is not None and av > threshold
                    tails.append(dict(population=population, before=before, after=after,
                        id=row["id"], axis=axis, threshold=threshold,
                        before_available=bv is not None, after_available=av is not None,
                        before_large_error=bt, after_large_error=at,
                        event="NEW_LARGE_ERROR" if at and not bt else "REMOVED_LARGE_ERROR" if bt and not at
                              else "BOTH_LARGE" if bt and at else "NEITHER_LARGE",
                        both_joint_failure=not b["success"] and not a["success"],
                        delta=av-bv if av is not None and bv is not None else None))
        for arm in registered:
            seed = next((s for s in SEEDS if arm.endswith("_s" + str(s))), SEEDS[0])
            h1 = "D1_H1_MLP_s" + str(seed)
            for base in ("N0", "E0"):
                for b, h, a in itertools.product((False, True), repeat=3):
                    ids = [r["id"] for r in records if
                           (r["metrics"][base]["success"], r["metrics"][h1]["success"],
                            r["metrics"][arm]["success"]) == (b, h, a)]
                    cross.append(dict(population=population, arm=arm, base=base, H1=h1,
                        base_success=b, H1_success=h, new_success=a, n=len(ids),
                        denominator=len(records), ids="|".join(ids)))
    for row in ledger:
        for arm in registered:
            prediction = row["predictions"][arm]
            if "guard_decision" not in prediction:
                continue
            d = prediction["guard_decision"]
            decisions.append(dict(id=row["id"], partition=row["partition"], arm=arm,
                diagnostic_trial=prediction["diagnostic_trial"], policy_status=prediction["policy_status"],
                selected_candidate_id=prediction.get("selected_candidate_id"),
                selected_source=prediction["action_role"], **d))
    csv_write(output / "TAIL_TRANSITIONS.csv.gz", tails)
    csv_write(output / "H1_THREE_WAY_SUCCESS.csv", cross)
    csv_write(output / "POLICY_DECISIONS.csv.gz", decisions)
    save_rows(output / "SELECTED_POSES.jsonl.gz", (dict(id=r["id"], partition=r["partition"],
        predictions={a: dict(actual_pose=r["predictions"][a]["actual_pose"]) for a in registered})
        for r in ledger))
