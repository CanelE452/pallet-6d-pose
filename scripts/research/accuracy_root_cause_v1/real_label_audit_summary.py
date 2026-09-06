"""REAL_LABEL_AUDIT.csv 를 집계한다.  읽기 전용, 새 추론 0 회."""
from __future__ import annotations

import collections
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "data/pallet/results/accuracy_root_cause_v1"
CSV = RES / "REAL_LABEL_AUDIT.csv"
OUT = RES / "REAL_LABEL_AUDIT_SUMMARY.json"

# migrated_gt / migrated_gt_wood 는 정본의 gt_v2 사본이다
# (challenge/data/01_real/gt_v2_canonical -> ../../real_gt_v2/migrated_gt 심링크).
# 감사는 하되 전체 합계에서는 뺀다 — 안 그러면 같은 프레임을 두 번 센다.
DUP_GROUPS = ("real_gt_v2/migrated_gt", "real_gt_v2/migrated_gt_wood")
VERDICTS = ("LABEL_OK", "LR_ORDER_VIOLATION", "YAW90_STALE", "OTHER_DEFECT", "AMBIGUOUS")
BINS = ("<8", "8-15", ">=15")


def ebin(v):
    if v == "":
        return "unknown"
    e = float(v)
    return "<8" if e < 8 else "8-15" if e < 15 else ">=15"


def tally(rows, key=lambda r: r["verdict"]):
    return dict(collections.Counter(key(r) for r in rows))


def main():
    rows = list(csv.DictReader(CSV.open(encoding="utf-8")))
    prim = [r for r in rows if r["group"] not in DUP_GROUPS]
    dup = [r for r in rows if r["group"] in DUP_GROUPS]

    # 학습에 쓸 수 있는 모집단: eval split 제외, apriltag GT(pallet11) 제외, 사본 제외.
    train_pool = [r for r in prim
                  if r["split"] != "eval" and r["gt_source"] != "apriltag"]
    # `_night_eval_manual_gt` 는 night05/06/07 의, `_outside_eval_manual_gt` 는
    # pallet02/03/04/05/08 의 큐레이션 사본이다 — stem 이 원본 세션과 겹친다(실측 133건).
    # 세션 단위 held-out 을 짜려면 빼야 한다.
    CURATED_DUP = ("_night_eval_manual_gt", "_outside_eval_manual_gt")
    train_pool = [r for r in train_pool if r["folder"] not in CURATED_DUP]
    ok = [r for r in train_pool if r["verdict"] == "LABEL_OK"]
    ok_or_fix = [r for r in train_pool if r["usable_if_kpann_canonical"] == "True"]

    def strat(rs):
        c = collections.Counter(ebin(r["elevation_deg"]) for r in rs)
        return {b: c.get(b, 0) for b in BINS} | ({"unknown": c["unknown"]} if c["unknown"] else {})

    s_ok, s_fix = strat(ok), strat(ok_or_fix)
    sessions_ok = sorted({r["session"] for r in ok})
    sessions_fix = sorted({r["session"] for r in ok_or_fix})

    out = {
        "schema": "real_label_audit_summary_v1",
        "generated_by": "scripts/research/accuracy_root_cause_v1/real_label_audit_summary.py",
        "source_csv": str(CSV.relative_to(ROOT)),
        "gt_json_modified": 0,
        "population": {
            "TOTAL_ROWS_AUDITED": len(rows),
            "TOTAL_REAL_LABELED_FRAMES": len(prim),
            "duplicate_v2_copies_excluded": len(dup),
            "duplicate_groups": list(DUP_GROUPS),
            "train_eligible_pool": len(train_pool),
            "excluded_from_train_pool": {
                "split_eval": sum(1 for r in prim if r["split"] == "eval"),
                "apriltag_gt_pallet11": sum(1 for r in prim if r["gt_source"] == "apriltag"),
            },
        },
        "verdicts_all": {v: sum(1 for r in prim if r["verdict"] == v) for v in VERDICTS},
        "verdicts_duplicate_copies": {v: sum(1 for r in dup if r["verdict"] == v) for v in VERDICTS},
        "verdicts_train_pool": {v: sum(1 for r in train_pool if r["verdict"] == v) for v in VERDICTS},
        "curated_duplicate_folders_removed_from_train_pool": {
            "folders": ["_night_eval_manual_gt", "_outside_eval_manual_gt"],
            "n": sum(1 for r in prim
                     if r["folder"] in ("_night_eval_manual_gt", "_outside_eval_manual_gt")
                     and r["split"] != "eval"),
            "why": "stem 이 원본 세션과 중복 — 세션 held-out 이 새어 나간다",
        },
        "elevation_x_object_of_usable": {
            b: dict(collections.Counter(
                (r["object_type"] or r["dims_wd"]) for r in ok if ebin(r["elevation_deg"]) == b))
            for b in BINS
        },
        "elevation_x_object_if_kpann_canonical": {
            b: dict(collections.Counter(
                (r["object_type"] or r["dims_wd"]) for r in ok_or_fix
                if ebin(r["elevation_deg"]) == b))
            for b in BINS
        },
        "sessions_per_elevation_bin": {
            b: len({r["session"] for r in ok if ebin(r["elevation_deg"]) == b}) for b in BINS
        },
        "by_group": {
            g: {"n": sum(1 for r in rows if r["group"] == g),
                "verdicts": tally([r for r in rows if r["group"] == g])}
            for g in sorted({r["group"] for r in rows})
        },
        "by_folder": {
            f"{g}/{f}": {"n": n,
                         "verdicts": tally([r for r in rows if r["group"] == g and r["folder"] == f]),
                         "elevation": strat([r for r in rows if r["group"] == g and r["folder"] == f])}
            for (g, f), n in sorted(collections.Counter(
                (r["group"], r["folder"]) for r in rows).items())
        },
        "USABLE_BY_ELEVATION": s_ok,
        "USABLE_BY_ELEVATION_IF_KEYPOINT_ANNOTATIONS_CANONICAL": s_fix,
        "MAX_BALANCED_ARM_SIZE": min(s_ok[b] for b in BINS),
        "MAX_BALANCED_ARM_SIZE_IF_KEYPOINT_ANNOTATIONS_CANONICAL": min(s_fix[b] for b in BINS),
        "SESSION_COUNT_FOR_HELD_OUT_SPLIT": len(sessions_ok),
        "SESSION_COUNT_IF_KEYPOINT_ANNOTATIONS_CANONICAL": len(sessions_fix),
        "sessions_with_usable_frames": sessions_ok,
        "elevation_x_session_usable": {
            s: strat([r for r in ok if r["session"] == s]) for s in sessions_ok
        },
        "diagnostics": {
            "cross_field_test_available": sum(1 for r in rows if r["test_power"] == "CROSS_FIELD"),
            "internal_only_test": sum(1 for r in rows if r["test_power"] == "INTERNAL_ONLY"),
            "lr_screen_flip_true": sum(1 for r in rows if r["lr_screen_flip"] == "True"),
            "face_edge_on_gt60deg": sum(1 for r in rows if r["face_obliquity_deg"]
                                        and float(r["face_obliquity_deg"]) > 60.0),
            "near_invariant_violation": sum(1 for r in rows if r["near_ok"] == "False"),
            "top_invariant_violation": sum(1 for r in rows if r["top_ok"] == "False"),
            "neighbour_phase_flip_frames": sum(1 for r in rows if r["neighbour_phase_flip"] == "True"),
            "frames_with_null_keypoint": sum(1 for r in rows if r["n_none"] not in ("", "0")),
            "frames_with_sentinel": sum(1 for r in rows if r["n_sentinel"] not in ("", "0")),
            "image_missing_all_rows": sum(1 for r in rows if r["image_exists"] == "False"),
            "image_missing_in_train_pool": sum(1 for r in train_pool if r["image_exists"] == "False"),
            "pose_status_unconfirmed": sum(1 for r in rows
                                           if r["pose_status"] == "UNCONFIRMED_SIGNED_AXIS"),
            "n_click_median_of_usable": sorted(int(r["n_click"]) for r in ok)[len(ok) // 2] if ok else 0,
        },
        "thresholds_estimated_unverified": {
            "CLICK_MATCH_PX": 5.0, "OK_PX": 15.0, "FACE_EDGEON_DEG": 60.0,
            "NEIGHBOUR_TRANS_M": 0.15, "NEIGHBOUR_FLIP_DEG": 45.0,
            "NEIGHBOUR_C4_DEG": 15.0, "MIN_CLICKS": 4,
        },
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: out[k] for k in
                      ("population", "verdicts_all", "verdicts_train_pool",
                       "USABLE_BY_ELEVATION",
                       "USABLE_BY_ELEVATION_IF_KEYPOINT_ANNOTATIONS_CANONICAL",
                       "MAX_BALANCED_ARM_SIZE",
                       "MAX_BALANCED_ARM_SIZE_IF_KEYPOINT_ANNOTATIONS_CANONICAL",
                       "SESSION_COUNT_FOR_HELD_OUT_SPLIT",
                       "SESSION_COUNT_IF_KEYPOINT_ANNOTATIONS_CANONICAL",
                       "diagnostics")},
                     indent=2, ensure_ascii=False))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
