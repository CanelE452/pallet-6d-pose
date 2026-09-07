"""utility_selftrain_v1 section 2 — reproduce the current filter's failure structure.

Reads the existing M4 frame records (plastic 194, teacher predictions + GT + F0..F5
verdicts).  Restricted to POLICY_DEV and POLICY_VAL: STUDENT_EVAL is not opened here,
so the section 11 freeze stays intact.

GT-derived fields are used for ANALYSIS ONLY.  Nothing here feeds a selection rule.
"""
import json, sys, collections
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "_docs/experiments/utility_selftrain_v1"

RECORDS = ROOT / "data/pallet/results/paper_selftrain_v1/M4_FRAME_RECORDS.json"
SPLIT = OUT / "SPLIT_CONTRACT.json"

ARMS = ["F0_NAIVE", "F1_CONF", "F2_CONF_REPROJ", "F3_CONF_REMOVE",
        "F5_CONF_FLIP", "F4_PROPOSED"]
GROSS_PX, CATASTROPHIC_PX = 20.0, 40.0

# "accurate" is deliberately reported under several definitions (section 2).
ACCURACY_DEFS = {
    "max_corner_le_10px": lambda r: r["corner_max_px"] is not None and r["corner_max_px"] <= 10.0,
    "max_corner_le_20px": lambda r: r["corner_max_px"] is not None and r["corner_max_px"] <= 20.0,
    "no_gross_gt20px": lambda r: r["gross_keypoints"] == 0,
    "no_catastrophic_gt40px": lambda r: r["catastrophic_keypoints"] == 0,
}


def session_of(record):
    return record["frame_id"].split(":")[0]


def confusion(records, arm, acc_fn):
    t = collections.Counter()
    for r in records:
        accepted = bool(r["verdict"][arm])
        accurate = acc_fn(r)
        t[("accept" if accepted else "reject", "accurate" if accurate else "wrong")] += 1
    acc_acc = t[("accept", "accurate")]
    acc_wrong = t[("accept", "wrong")]
    rej_acc = t[("reject", "accurate")]
    rej_wrong = t[("reject", "wrong")]
    n_acc = acc_acc + acc_wrong
    return {
        "accept_accurate": acc_acc, "accept_wrong": acc_wrong,
        "reject_accurate": rej_acc, "reject_wrong": rej_wrong,
        "coverage": n_acc / len(records) if records else None,
        "precision": acc_acc / n_acc if n_acc else None,
        "recall": acc_acc / (acc_acc + rej_acc) if (acc_acc + rej_acc) else None,
    }


def pooled_error_stats(records, arm, accepted):
    errs = []
    for r in records:
        if bool(r["verdict"][arm]) == accepted:
            errs.extend(r["errors_px"])
    if not errs:
        return {"n_frames": 0, "n_keypoints": 0, "median_px": None, "p90_px": None,
                "gross_rate": None, "catastrophic_rate": None}
    errs.sort()
    n = len(errs)
    q = lambda p: errs[min(n - 1, int(p * n))]
    return {
        "n_frames": sum(1 for r in records if bool(r["verdict"][arm]) == accepted),
        "n_keypoints": n,
        "median_px": q(0.50), "p90_px": q(0.90),
        "gross_rate": sum(e > GROSS_PX for e in errs) / n,
        "catastrophic_rate": sum(e > CATASTROPHIC_PX for e in errs) / n,
    }


def qualitative_cells(records, tau_box, tau_geom):
    """The four cases the instruction names in section 2."""
    cells = collections.defaultdict(list)
    for r in records:
        conf_hi = r["box_conf"] >= tau_box
        g4 = max(v for v in (r["s_remove"], r["s_flip"]) if v is not None) \
            if r["s_remove"] is not None and r["s_flip"] is not None else float("inf")
        geom_ok = g4 <= tau_geom
        accurate = ACCURACY_DEFS["max_corner_le_20px"](r)
        accepted = bool(r["verdict"]["F4_PROPOSED"])
        if conf_hi and not accurate:
            cells["high_confidence_but_wrong"].append(r)
        if (not conf_hi) and geom_ok and accurate:
            cells["low_confidence_but_geometry_consistent_and_accurate"].append(r)
        if accurate and not accepted:
            cells["correct_but_rejected"].append(r)
        if (not accurate) and accepted:
            cells["wrong_but_accepted"].append(r)
    return {k: {"n": len(v),
                "sessions": dict(collections.Counter(session_of(r) for r in v)),
                "median_box_conf": sorted(x["box_conf"] for x in v)[len(v) // 2] if v else None,
                "median_corner_max_px": sorted(
                    x["corner_max_px"] for x in v)[len(v) // 2] if v else None}
            for k, v in cells.items()}


def main() -> int:
    recs = json.loads(RECORDS.read_text())
    split = json.loads(SPLIT.read_text())
    lock = json.loads((ROOT / "data/evaluation/pallet_eval_v1/adaptation/"
                       "PSEUDOLABEL_FILTER_LOCK.json").read_text())
    tau_box = lock["TAU_BOX"]
    tau_geom = lock["geometry_thresholds"]["tau_remove"]

    sess_role = {}
    for role, v in split["roles"].items():
        for s in v["sessions"]:
            sess_role[s] = role

    by_role = collections.defaultdict(list)
    unmapped = collections.Counter()
    for r in recs["frames"]:
        role = sess_role.get(session_of(r))
        if role is None:
            unmapped[session_of(r)] += 1
        else:
            by_role[role].append(r)

    report = {
        "schema_version": "utility_selftrain_v1_baseline_filter_audit",
        "source_records": str(RECORDS.relative_to(ROOT)),
        "source_population": recs["population"],
        "teacher_sha256": recs["teacher_sha256"],
        "filter_lock_sha256": recs["filter_lock_sha256"],
        "gt_used_for": "analysis only; no GT-derived field feeds any selection rule",
        "student_eval_untouched": True,
        "roles_audited": ["POLICY_DEV", "POLICY_VAL"],
        "unmapped_sessions": dict(unmapped),
        "note_wood": "M4 records cover PAPER_EVAL_PLASTIC_POS only, so the wood sessions "
                     "in each role have no teacher record here and are absent from these counts.",
        "roles": {},
    }

    for role in ("POLICY_DEV", "POLICY_VAL"):
        rs = by_role[role]
        report["roles"][role] = {
            "n_frames_with_records": len(rs),
            "n_frames_in_split": split["roles"][role]["n_frames"],
            "sessions": dict(collections.Counter(session_of(r) for r in rs)),
            "confusion": {
                arm: {name: confusion(rs, arm, fn) for name, fn in ACCURACY_DEFS.items()}
                for arm in ARMS
            },
            "pooled_error": {
                arm: {"pass": pooled_error_stats(rs, arm, True),
                      "reject": pooled_error_stats(rs, arm, False)}
                for arm in ARMS
            },
            "qualitative": qualitative_cells(rs, tau_box, tau_geom),
        }

    (OUT / "BASELINE_FILTER_AUDIT.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False))

    for role in ("POLICY_DEV", "POLICY_VAL"):
        d = report["roles"][role]
        print(f"\n=== {role}  records={d['n_frames_with_records']} "
              f"(split N={d['n_frames_in_split']}, wood excluded) ===")
        print(f"{'arm':16s}{'cover':>7s}{'acc_acc':>9s}{'acc_wrong':>10s}"
              f"{'rej_acc':>9s}{'rej_wrong':>10s}{'passMed':>9s}{'pass>20':>9s}{'pass>40':>9s}")
        for arm in ARMS:
            c = d["confusion"][arm]["max_corner_le_20px"]
            p = d["pooled_error"][arm]["pass"]
            fmt = lambda v, f="{:.3f}": ("--" if v is None else f.format(v))
            print(f"{arm:16s}{fmt(c['coverage']):>7s}{c['accept_accurate']:>9d}"
                  f"{c['accept_wrong']:>10d}{c['reject_accurate']:>9d}{c['reject_wrong']:>10d}"
                  f"{fmt(p['median_px'],'{:.2f}'):>9s}{fmt(p['gross_rate']):>9s}"
                  f"{fmt(p['catastrophic_rate']):>9s}")
        print("  qualitative:", {k: v["n"] for k, v in d["qualitative"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
