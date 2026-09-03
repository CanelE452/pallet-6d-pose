"""site 후보별로 adaptation/evaluation 을 누수 없이 나눌 수 있는지 판정한다.

    python3 scripts/self_training_yolo/site_audit/build_site_matched_candidates.py \
        --output-dir data/pallet/results/site_environment_audit_v1

출력  SITE_MATCHED_SPLIT_CANDIDATES.md · SITE_MATCHED_SPLIT_CANDIDATES.json

site 는 **사람 확인 전이므로 PROVISIONAL** 이다.  이 스크립트는 site_id 를
확정하지 않고, 확정됐다고 가정했을 때 누수 게이트가 통과하는지만 계산한다.

누수 게이트(§4)
    adapt source recording ∩ eval source recording = 공집합
    adapt image SHA ∩ eval image SHA = 공집합

READY 는 위 둘에 더해 site 확인 · adapt 이미지 존재 · eval 이미지 존재를 요구한다.
site 확인이 아직 없으므로 이 실행의 최대 등급은 READY_PENDING_SITE_CONFIRMATION 이다.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = REPO_ROOT / "data/pallet/results/site_environment_audit_v1"

# 평가에 쓰이는 세션 — challenge/data_paths.py 의 EVAL_CANONICAL 과
# pallet_eval_v1 의 FINAL positive.  여기 속한 recording 은 adaptation 에 못 쓴다.
EVALUATION_SESSION_HINTS = (
    "eval_canonical/", "dev_existing/sessions/eval_",
    "final/positive/sessions/",
    "manual_gt/capturepallet07_manual_gt", "manual_gt/capturepallet09_manual_gt",
    "manual_gt/capturenight08_manual_gt", "manual_gt/capturenight09_manual_gt",
)


def is_evaluation_session(session_key: str) -> bool:
    return any(hint in session_key for hint in EVALUATION_SESSION_HINTS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    out_dir = Path(args.output_dir).resolve()

    groups = json.loads((out_dir / "SOURCE_RECORDING_GROUPS.json").read_text())
    proposals = json.loads((out_dir / "PROPOSED_SITE_GROUPS.json").read_text())
    sha_index = json.loads((out_dir / "IMAGE_SHA_INDEX.json").read_text())["sessions"]

    unit_ids = {u["recording_id"] for u in proposals["units"]}
    unit_of = {u["recording_id"]: u for u in proposals["units"]}

    # ── site 후보 = LIKELY_SAME_SITE 의 연결요소 (사람 확인 전이므로 잠정)
    parent = {rid: rid for rid in unit_ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for pair in proposals["likely_same_site_pairs"]:
        a, b = find(pair["recording_a"]), find(pair["recording_b"])
        if a != b:
            parent[a] = b

    clusters: dict[str, list[str]] = defaultdict(list)
    for rid in sorted(unit_ids):
        clusters[find(rid)].append(rid)

    # ── recording -> 그 안의 세션 / SHA / 평가 여부
    recording_sessions = {g["recording_id"]: [s["session_key"] for s in g["sessions"]]
                          for g in groups["groups"]}
    recording_shas: dict[str, set[str]] = {}
    recording_is_eval: dict[str, list[str]] = {}
    for rid, keys in recording_sessions.items():
        shas: set[str] = set()
        eval_keys = []
        for key in keys:
            shas |= set(sha_index.get(key, []))
            if is_evaluation_session(key):
                eval_keys.append(key)
        recording_shas[rid] = shas
        recording_is_eval[rid] = eval_keys

    sites = []
    for order, (_, members) in enumerate(
            sorted(clusters.items(),
                   key=lambda kv: -sum(unit_of[r]["n_images"] for r in kv[1])), start=1):
        members.sort(key=lambda r: -unit_of[r]["n_images"])
        eval_recs = [r for r in members if recording_is_eval[r]]
        adapt_recs = [r for r in members if not recording_is_eval[r]]

        adapt_shas: set[str] = set()
        for r in adapt_recs:
            adapt_shas |= recording_shas[r]
        eval_shas: set[str] = set()
        for r in eval_recs:
            eval_shas |= recording_shas[r]

        sha_overlap = len(adapt_shas & eval_shas)
        recording_overlap = len(set(adapt_recs) & set(eval_recs))

        if not eval_recs:
            status = "NO_EVAL_DATA"
        elif not adapt_recs:
            status = "NO_ADAPT_DATA"
        elif recording_overlap or sha_overlap:
            status = "LEAKAGE_FAIL"
        elif len(adapt_recs) >= 2 and len(eval_recs) >= 2:
            status = "READY_PENDING_SITE_CONFIRMATION_STRONG"
        else:
            status = "READY_PENDING_SITE_CONFIRMATION_WEAK"

        sites.append({
            "site_candidate": f"SITE_CAND_{order:02d}",
            "confirmed_by_human": False,
            "recordings": members,
            "lighting": sorted({unit_of[r]["lighting"] for r in members}),
            "adapt_recordings": adapt_recs,
            "eval_recordings": eval_recs,
            "adapt_images": len(adapt_shas),
            "eval_images": len(eval_shas),
            "adapt_eval_recording_overlap": recording_overlap,
            "adapt_eval_sha_overlap": sha_overlap,
            "status": status,
            "eval_sessions": sorted({k for r in eval_recs
                                     for k in recording_is_eval[r]}),
        })

    report = {
        "schema_version": "site_matched_split_candidates_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "AUTO_GROUPING_IS_FINAL": False,
        "HUMAN_CONFIRMATION_REQUIRED": True,
        "TRAINING_STARTED": False,
        "site_identity_is_provisional": True,
        "gate": {
            "recording_disjoint": "adapt source recording ∩ eval source recording = empty",
            "sha_disjoint": "adapt image SHA ∩ eval image SHA = empty",
            "note": "same site is allowed and wanted; same recording is not",
        },
        "sites": sites,
    }
    (out_dir / "SITE_MATCHED_SPLIT_CANDIDATES.json").write_text(
        json.dumps(report, indent=2) + "\n")

    lines = [
        "# Site-matched split candidates", "",
        "**Provisional.** No site identity is confirmed here. The grouping below comes",
        "from background-masked SIFT matching between source recordings, which is a",
        "candidate generator, not a decision. Nothing is FROZEN and no training ran.",
        "",
        "`status` can reach at best `READY_PENDING_SITE_CONFIRMATION` because the",
        "first READY condition — a human confirming the site — has not happened yet.",
        "",
        "```text",
        f"{'Site':16}{'Adapt rec':>10}{'Eval rec':>9}{'Adapt img':>11}{'Eval img':>10}"
        f"{'rec ovl':>9}{'SHA ovl':>9}  status",
        "─" * 108,
    ]
    for site in sites:
        lines.append(
            f"{site['site_candidate']:16}{len(site['adapt_recordings']):10d}"
            f"{len(site['eval_recordings']):9d}{site['adapt_images']:11d}"
            f"{site['eval_images']:10d}{site['adapt_eval_recording_overlap']:9d}"
            f"{site['adapt_eval_sha_overlap']:9d}  {site['status']}")
    lines += ["```", "", "## Per site", ""]
    for site in sites:
        lines += [f"### {site['site_candidate']}  —  {site['status']}", "", "```text",
                  f"lighting     {', '.join(site['lighting'])}",
                  f"adaptation   {', '.join(site['adapt_recordings']) or '(none)'}",
                  f"evaluation   {', '.join(site['eval_recordings']) or '(none)'}"]
        for rid in site["recordings"]:
            unit = unit_of[rid]
            role = "EVAL" if rid in site["eval_recordings"] else "adapt"
            lines.append(f"  {rid:9}{role:7}{unit['n_images']:7d}  "
                         f"{Path(unit['session_key']).name}")
        lines += ["```", ""]
    lines += [
        "## What still has to happen", "", "```text",
        "1  a human confirms each site candidate from its contact sheets",
        "2  SITE_GROUP_LOCK.json is written from those confirmations",
        "3  only then can a site-matched split be called READY",
        "```", "",
        "`AUTO_GROUPING_IS_FINAL = NO`", "`HUMAN_CONFIRMATION_REQUIRED = YES`",
        "`TRAINING_STARTED = NO`", "",
    ]
    (out_dir / "SITE_MATCHED_SPLIT_CANDIDATES.md").write_text("\n".join(lines) + "\n")

    print(f"site 후보 {len(sites)}")
    for site in sites:
        print(f"  {site['site_candidate']:16}{site['status']:42}"
              f"adapt {len(site['adapt_recordings'])} / eval {len(site['eval_recordings'])}"
              f"   SHA overlap {site['adapt_eval_sha_overlap']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
