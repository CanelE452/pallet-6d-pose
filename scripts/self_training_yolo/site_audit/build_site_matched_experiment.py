"""사람이 확정한 outdoor site 로 recording 단위 split 과 실험 lock 을 만든다.

    python3 scripts/self_training_yolo/site_audit/build_site_matched_experiment.py \
        --output-dir data/pallet/results/site_environment_audit_v1

입력  OUTDOOR_SITE_REVIEW.json   (review_outdoor_site.py 로 사람이 작성)
출력  OUTDOOR_SITE_GROUP_LOCK.json
      <SITE>_EVAL_POSITIVE.csv
      <SITE>_UNLABELED_ADAPT.csv
      SITE_MATCHED_EXPERIMENT_LOCK.json

핵심 규칙
    평가 프레임이 한 장이라도 나온 raw recording 은 recording 전체가 adaptation
    에서 빠진다.  한 촬영을 앞뒤로 갈라 adapt/eval 로 쓰지 않는다.

    site 는 자동 유사도가 아니라 **사람 판단** 그대로 저장한다.

    성능 결과는 읽지 않는다.  이 스크립트는 어떤 모델 산출물도 열지 않는다.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = REPO_ROOT / "data/pallet/results/site_environment_audit_v1"
OUTDOOR_ROOT = "data/pallet/raw_data/outside"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}

MIN_EVAL_FRAMES_FOR_STRONG = 20   # 이보다 적으면 WEAK.  결과 보기 전에 고정.


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    out_dir = Path(args.output_dir).resolve()

    review_path = out_dir / "OUTDOOR_SITE_REVIEW.json"
    if not review_path.exists():
        print("OUTDOOR_SITE_REVIEW.json 이 없다 — 먼저 review_outdoor_site.py 를 돌려라")
        return 1
    review = json.loads(review_path.read_text())
    labels = review["labels"]

    eval_report = json.loads((out_dir / "OUTDOOR_EVAL_SOURCE_RECORDINGS.json").read_text())
    sha_index = json.loads((out_dir / "IMAGE_SHA_INDEX.json").read_text())["sessions"]
    groups = json.loads((out_dir / "SOURCE_RECORDING_GROUPS.json").read_text())["groups"]
    similarity_path = out_dir / "SESSION_VISUAL_SIMILARITY.csv"

    recording_of_session = {}
    sessions_of_recording = defaultdict(list)
    for group in groups:
        for session in group["sessions"]:
            recording_of_session[session["session_key"]] = group["recording_id"]
            sessions_of_recording[group["recording_id"]].append(session["session_key"])

    eval_by_raw = {e["raw_recording"]: e for e in eval_report["eval_source_recordings"]}

    # ── site -> recordings (사람 판단 그대로)
    sites: dict[str, list[str]] = defaultdict(list)
    for name, entry in labels.items():
        sites[entry["site"]].append(name)
    for members in sites.values():
        members.sort()

    # ── viewpoint 잠정 묶음 (§9) — 확정이 아니라 기록용
    provisional_viewpoint: dict[str, str] = {}
    if similarity_path.exists():
        strong = defaultdict(set)
        rec_to_raw = {}
        for name in labels:
            rid = recording_of_session.get(f"{OUTDOOR_ROOT}/{name}")
            if rid:
                rec_to_raw[rid] = name
        for row in csv.DictReader(similarity_path.open()):
            a = rec_to_raw.get(row["recording_a"])
            b = rec_to_raw.get(row["recording_b"])
            if a and b and int(row["geometric_match_inliers"]) >= 200:
                strong[a].add(b)
                strong[b].add(a)
        seen, order = set(), 0
        for name in sorted(labels):
            if name in seen:
                continue
            order += 1
            stack, cluster = [name], []
            while stack:
                current = stack.pop()
                if current in seen:
                    continue
                seen.add(current)
                cluster.append(current)
                stack.extend(strong.get(current, ()))
            for member in cluster:
                provisional_viewpoint[member] = f"V{order}"

    lock = {
        "schema_version": "outdoor_site_group_lock_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": "human review, review_outdoor_site.py",
        "review_file": review_path.name,
        "n_labelled": len(labels),
        "automatic_similarity_did_not_decide": True,
        "viewpoint_note": ("viewpoint_group is provisional and comes from strong "
                           "background feature matches; site identity is the human's, "
                           "viewpoint is not confirmed and is recorded only"),
        "sites": {
            site: [{
                "raw_recording": name,
                "recording_id": recording_of_session.get(f"{OUTDOOR_ROOT}/{name}"),
                "site_id": site,
                "viewpoint_group": provisional_viewpoint.get(name, "UNASSIGNED"),
                "lighting": "day",
                "material": "plastic",
                "evaluation_frames": eval_by_raw.get(name, {}).get("evaluation_frames", 0),
                "raw_images": len(sha_index.get(f"{OUTDOOR_ROOT}/{name}", [])),
                "current_role": ("evaluation_source" if name in eval_by_raw
                                 else "unlabelled_only"),
            } for name in members]
            for site, members in sorted(sites.items())
        },
    }
    (out_dir / "OUTDOOR_SITE_GROUP_LOCK.json").write_text(json.dumps(lock, indent=2) + "\n")

    # ── site 별 split · manifest · 게이트
    experiments = []
    for site, members in sorted(sites.items()):
        if site == "UNCLEAR":
            continue
        eval_recs = [n for n in members if n in eval_by_raw]
        adapt_recs = [n for n in members if n not in eval_by_raw]

        eval_shas, eval_rows = set(), []
        for name in eval_recs:
            entry = eval_by_raw[name]
            eval_shas |= set(entry["evaluation_frame_sha256"])
        # 평가 population 은 이 site 의 held-out recording 에서 나온 GT 프레임만
        for row in csv.DictReader(
                (REPO_ROOT / eval_report["provenance_source"]).open()):
            if row["source_image_sha256"] not in eval_shas:
                continue
            # manifest 에는 active_frame_id 가 빈 보조 행이 있다.  프레임이 아니다.
            if not row["active_frame_id"].strip():
                continue
            session_name = row["active_frame_id"].split("__", 1)[0]
            raw = None
            for name in eval_recs:
                if row["source_image_sha256"] in set(
                        eval_by_raw[name]["evaluation_frame_sha256"]):
                    raw = name
                    break
            eval_rows.append({
                "frame_id": row["active_frame_id"],
                "source_recording": raw,
                "eval_session": session_name,
                "image": row["destination_image_path"],
                "annotation": row["destination_annotation_path"],
                "image_sha256": row["source_image_sha256"],
                "material": "plastic",
                "lighting": "day",
            })
        # 같은 프레임이 여러 행으로 올 수 있다 — frame_id 로 중복 제거
        unique_eval = {r["frame_id"]: r for r in eval_rows}
        eval_rows = [unique_eval[k] for k in sorted(unique_eval)]

        adapt_rows, adapt_shas = [], set()
        for name in adapt_recs:
            key = f"{OUTDOOR_ROOT}/{name}"
            image_dir = REPO_ROOT / key / "rgb"
            digests = sha_index.get(key, [])
            adapt_shas |= set(digests)
            paths = sorted(p for p in image_dir.iterdir()
                           if p.suffix.lower() in IMAGE_SUFFIXES) if image_dir.is_dir() else []
            for path in paths:
                adapt_rows.append({
                    "image_path": str(path.relative_to(REPO_ROOT)),
                    "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "source_recording": name,
                    "lighting": "day",
                })

        # ── 세 게이트 (§8)
        gate_a = len(adapt_shas & eval_shas)
        gate_b = len(set(adapt_recs) & set(eval_recs))
        adapt_underlying = {recording_of_session.get(f"{OUTDOOR_ROOT}/{n}") for n in adapt_recs}
        eval_underlying = {recording_of_session.get(f"{OUTDOOR_ROOT}/{n}") for n in eval_recs}
        gate_c = len(adapt_underlying & eval_underlying)

        if gate_a or gate_b or gate_c:
            readiness = "NOT_READY"
        elif not adapt_recs or not eval_recs:
            readiness = "NOT_READY"
        elif len(eval_rows) < MIN_EVAL_FRAMES_FOR_STRONG:
            readiness = "WEAK"
        elif len(adapt_recs) >= 2 and len(eval_recs) >= 2:
            readiness = "READY_STRONG"
        else:
            readiness = "READY_MINIMAL"

        if eval_rows:
            with (out_dir / f"{site}_EVAL_POSITIVE.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(eval_rows[0]))
                writer.writeheader()
                writer.writerows(eval_rows)
        if adapt_rows:
            with (out_dir / f"{site}_UNLABELED_ADAPT.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(adapt_rows[0]))
                writer.writeheader()
                writer.writerows(adapt_rows)

        experiments.append({
            "site": site,
            "site_confirmed_by_human": True,
            "adapt_recordings": adapt_recs,
            "eval_recordings": eval_recs,
            "adapt_images": len(adapt_rows),
            "eval_gt_frames": len(eval_rows),
            "adapt_underlying_recording_ids": sorted(x for x in adapt_underlying if x),
            "eval_underlying_recording_ids": sorted(x for x in eval_underlying if x),
            "lighting_composition": {"day": len(adapt_rows)},
            "material": "plastic",
            "viewpoint_groups": {n: provisional_viewpoint.get(n, "UNASSIGNED")
                                 for n in members},
            "gate_a_image_sha_overlap": gate_a,
            "gate_b_source_recording_overlap": gate_b,
            "gate_c_underlying_recording_overlap": gate_c,
            "readiness": readiness,
            "eval_manifest": f"{site}_EVAL_POSITIVE.csv",
            "adapt_manifest": f"{site}_UNLABELED_ADAPT.csv",
        })

    experiment_lock = {
        "schema_version": "site_matched_experiment_lock_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_before_any_result": True,
        "TRAINING_STARTED": False,
        "model_results_read": False,
        "headline_evaluation_population": ("the site's own held-out recordings only, "
                                           "not PAPER_EVAL 319"),
        "recording_level_rule": ("a recording that supplies any evaluation frame is "
                                 "excluded from adaptation in full"),
        "strong_threshold_eval_frames": MIN_EVAL_FRAMES_FOR_STRONG,
        "experiments": experiments,
    }
    (out_dir / "SITE_MATCHED_EXPERIMENT_LOCK.json").write_text(
        json.dumps(experiment_lock, indent=2) + "\n")

    print(f"site {len(sites)}  (UNCLEAR 제외 {len(experiments)})")
    for entry in experiments:
        print(f"\n{entry['site']}   {entry['readiness']}")
        print(f"  eval  recordings {len(entry['eval_recordings'])}  "
              f"GT frames {entry['eval_gt_frames']}   {', '.join(entry['eval_recordings'])}")
        print(f"  adapt recordings {len(entry['adapt_recordings'])}  "
              f"images {entry['adapt_images']}   {', '.join(entry['adapt_recordings'])}")
        print(f"  gates  A(sha) {entry['gate_a_image_sha_overlap']}  "
              f"B(recording) {entry['gate_b_source_recording_overlap']}  "
              f"C(underlying) {entry['gate_c_underlying_recording_overlap']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
