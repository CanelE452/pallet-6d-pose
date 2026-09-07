"""utility_selftrain_v1 section 4 — session-level 3-way split of PAPER_EVAL_POSITIVE.

Roles: POLICY_DEV (feature/purity analysis) / POLICY_VAL (arm screening)
       / STUDENT_EVAL (frozen final comparison).
No session appears in two roles.  Written before any utility result is observed.
"""
import json, sys, collections
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/evaluation"))
from eval_workspace import load_frames, evaluation_population_views  # noqa: E402

OUT = ROOT / "_docs/experiments/utility_selftrain_v1"

# Session -> role.  Rationale recorded in SPLIT_CONTRACT.md.
ROLE = {
    # frozen metric_split_lock.md 1.6 final_test -> STUDENT_EVAL (never tuned on)
    "eval_pallet07": "STUDENT_EVAL", "eval_pallet09": "STUDENT_EVAL",
    "eval_night08": "STUDENT_EVAL",  "eval_night09": "STUDENT_EVAL",
    "wood_day_01": "STUDENT_EVAL",   "wood_184309": "STUDENT_EVAL",
    # policy development
    "plastic_day_01": "POLICY_DEV", "plastic_night_01": "POLICY_DEV",
    "wood_183705": "POLICY_DEV",
    # policy screening
    "eval_cad": "POLICY_VAL", "eval_noapril": "POLICY_VAL",
    "eval_outside": "POLICY_VAL", "wood_night_01": "POLICY_VAL",
}

# Sessions that share raw filenames with the adaptation pool.  Never promote.
FORBIDDEN = ["pallet11_gt", "capturenight01_manual_gt", "capturenight02_manual_gt",
             "capturenight03_manual_gt", "capturenight04_manual_gt"]


def main() -> int:
    frames = load_frames(ROOT / "data/evaluation/pallet_eval_v1")
    pos = evaluation_population_views(frames)["PAPER_EVAL_POSITIVE"]

    by_role = collections.defaultdict(list)
    unmapped = []
    for f in pos:
        r = ROLE.get(f["session_id"])
        (by_role[r] if r else unmapped).append(f)
    if unmapped:
        print(f"UNMAPPED_SESSIONS: {sorted({f['session_id'] for f in unmapped})}")
        return 1

    def summarise(fs):
        return {
            "n_frames": len(fs),
            "n_sessions": len({f["session_id"] for f in fs}),
            "sessions": sorted({f["session_id"] for f in fs}),
            "material": dict(collections.Counter(
                "wood" if "wood" in f["object_type"] else "plastic" for f in fs)),
            "lighting": dict(collections.Counter(f["lighting"] or "unknown" for f in fs)),
            "occlusion": dict(collections.Counter(f["occlusion"] or "unknown" for f in fs)),
            "distance_bin": dict(collections.Counter(f["distance_bin"] or "unknown" for f in fs)),
            "elevation_bin": dict(collections.Counter(f["elevation_bin"] or "unknown" for f in fs)),
            "frame_sha256": sorted(f["image_sha256"] for f in fs),
        }

    roles = {r: summarise(by_role[r]) for r in ("POLICY_DEV", "POLICY_VAL", "STUDENT_EVAL")}

    sets = {r: set(v["frame_sha256"]) for r, v in roles.items()}
    ss = {r: set(v["sessions"]) for r, v in roles.items()}
    pairs = [("POLICY_DEV", "POLICY_VAL"), ("POLICY_DEV", "STUDENT_EVAL"),
             ("POLICY_VAL", "STUDENT_EVAL")]
    inv = {
        "total_frames": sum(v["n_frames"] for v in roles.values()),
        "expected_total": len(pos),
        "frame_overlap": {f"{a}&{b}": len(sets[a] & sets[b]) for a, b in pairs},
        "session_overlap": {f"{a}&{b}": sorted(ss[a] & ss[b]) for a, b in pairs},
    }
    inv["passed"] = (inv["total_frames"] == inv["expected_total"]
                     and all(v == 0 for v in inv["frame_overlap"].values())
                     and all(v == [] for v in inv["session_overlap"].values()))

    contract = {
        "schema_version": "utility_selftrain_v1_split_contract",
        "population": "PAPER_EVAL_POSITIVE",
        "population_role": "DEV",
        "RESULT_ROLE": "DEVELOPMENT_ONLY",
        "result_role_reason":
            "PAPER_EVAL has been consumed as a development population by five prior "
            "selection tracks (PAPER_CLAIM_LOCK.json held_out_final=false).  No confirmatory "
            "population exists in this repository, so instruction section 4 forces "
            "DEVELOPMENT_ONLY and forbids a paper final claim.",
        "roles": roles,
        "invariants": inv,
        "forbidden_promotions": {
            "sessions": FORBIDDEN,
            "reason": "these share raw filenames with the adaptation pool "
                      "(pallet11_gt 243 + capturenight0*_manual_gt 45); promoting them "
                      "into any evaluation role creates pool/eval leakage.",
        },
        "adaptation_pool_disjoint": {
            "verified": True,
            "evidence": ["sha256 census raw 8031 vs eval 3007 -> 0 hits",
                         "ns-timestamp basename census -> 0 collisions",
                         "camera intrinsics separate three capture campaigns"],
        },
    }
    (OUT / "SPLIT_CONTRACT.json").write_text(json.dumps(contract, indent=2, ensure_ascii=False))

    for r, v in roles.items():
        print(f"{r:14s} N={v['n_frames']:3d}  sess={v['n_sessions']}  "
              f"{v['material']}  {v['lighting']}")
    print("invariants passed:", inv["passed"], inv["frame_overlap"], inv["session_overlap"])
    return 0 if inv["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
