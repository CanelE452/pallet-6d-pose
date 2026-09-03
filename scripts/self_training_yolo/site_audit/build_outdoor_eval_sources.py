"""outdoor 평가 GT 가 실제로 어떤 raw recording 에서 왔는지 frame 단위로 확정한다.

    python3 scripts/self_training_yolo/site_audit/build_outdoor_eval_sources.py \
        --output-dir data/pallet/results/site_environment_audit_v1

출력  OUTDOOR_EVAL_SOURCE_RECORDINGS.json

`import_provenance.csv` 의 `source_image_path` 를 프레임마다 읽는다 — 세션 이름이나
문서 요약을 믿지 않는다.  `ACQUISITION_DOMAIN_MAP.json` 이 capturepallet04 를
빠뜨린 전례가 있어서, 여기서는 manifest 원본만 근거로 쓴다.

**recording 단위 규칙**: 평가 프레임이 한 장이라도 나온 raw recording 은 그
recording 전체가 adaptation 후보에서 빠진다.  같은 촬영을 프레임으로 쪼개
앞은 adapt, 뒤는 eval 로 쓰는 것을 금지하기 때문이다.

모델 예측·성능 지표는 읽지 않는다.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = REPO_ROOT / "data/pallet/results/site_environment_audit_v1"
PROVENANCE = REPO_ROOT / "data/evaluation/pallet_eval_v1/manifests/import_provenance.csv"

# challenge/data_paths.py EVAL_CANONICAL 이 선언한 정본 평가 세션의 활성 이름.
# 문자열을 새로 만들지 않고 data_paths 에서 읽는다.
OUTDOOR_PREFIX = "data/pallet/raw_data/outside/"


def raw_recording_of(source_image_path: str) -> str | None:
    """`.../outside/capturepallet02/rgb/x.png` -> `capturepallet02`."""

    if not source_image_path.startswith(OUTDOOR_PREFIX):
        return None
    return source_image_path[len(OUTDOOR_PREFIX):].split("/", 1)[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    out_dir = Path(args.output_dir).resolve()

    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from challenge.data_paths import EVAL_CANONICAL, FINAL_TEST

    eval_session_names = set(EVAL_CANONICAL)
    inventory = {s["session_key"]: s for s in
                 json.loads((out_dir / "SESSION_INVENTORY.json").read_text())["sessions"]}
    sha_index = json.loads((out_dir / "IMAGE_SHA_INDEX.json").read_text())["sessions"]
    groups = json.loads((out_dir / "SOURCE_RECORDING_GROUPS.json").read_text())["groups"]

    recording_of_session = {}
    for group in groups:
        for session in group["sessions"]:
            recording_of_session[session["session_key"]] = group["recording_id"]

    # ── 평가 프레임 -> raw recording, frame 단위
    per_recording: dict[str, list[dict]] = defaultdict(list)
    unmatched: list[dict] = []
    seen_eval_frames: set[str] = set()
    for row in csv.DictReader(PROVENANCE.open()):
        active = row["active_frame_id"]
        session_name = active.split("__", 1)[0]
        if session_name not in eval_session_names:
            continue
        if row["disposition"].startswith("DUPLICATE"):
            continue
        seen_eval_frames.add(active)
        raw = raw_recording_of(row["source_image_path"])
        record = {
            "frame_id": active,
            "eval_session": session_name,
            "source_image_path": row["source_image_path"],
            "source_image_sha256": row["source_image_sha256"],
            "source_annotation_path": row["source_annotation_path"],
            "disposition": row["disposition"],
            "is_final_test_session": session_name in FINAL_TEST,
        }
        if raw is None:
            unmatched.append(record)
        else:
            record["resolved_by"] = "provenance_path"
            per_recording[raw].append(record)

    # ── 2번째 hop: manifest 가 GT 폴더를 source 로 적은 프레임은 경로로 raw 에
    # 닿지 않는다.  그 경우 **이미지 내용 SHA** 로 raw recording 을 찾는다.
    # 경로 문자열보다 내용이 강한 근거다.
    raw_sha_lookup: dict[str, str] = {}
    for key, digests in sha_index.items():
        if key.startswith(OUTDOOR_PREFIX) and key.count("/") == 4:
            name = key[len(OUTDOOR_PREFIX):]
            for digest in digests:
                raw_sha_lookup[digest] = name

    still_unmatched = []
    for record in unmatched:
        name = raw_sha_lookup.get(record["source_image_sha256"])
        if name is None:
            still_unmatched.append(record)
            continue
        record["resolved_by"] = "image_content_sha256"
        per_recording[name].append(record)
    unmatched = still_unmatched

    recordings = []
    for name in sorted(per_recording):
        session_key = f"data/pallet/raw_data/outside/{name}"
        session = inventory.get(session_key)
        frames = per_recording[name]
        recordings.append({
            "raw_recording": name,
            "raw_session_key": session_key,
            "recording_id": recording_of_session.get(session_key),
            "evaluation_frames": len(frames),
            "raw_rgb_total": session["frame_count"] if session else 0,
            "raw_unique_images": len(sha_index.get(session_key, [])),
            "eval_sessions": sorted({f["eval_session"] for f in frames}),
            "resolved_by": sorted({f["resolved_by"] for f in frames}),
            "contains_final_test_frames": any(f["is_final_test_session"] for f in frames),
            "evaluation_frame_sha256": sorted({f["source_image_sha256"] for f in frames}),
            "adaptation_eligible": False,
            "exclusion_reason": ("this recording supplies evaluation GT frames; "
                                 "recording-level exclusion, not frame-level"),
        })

    # ── 같은 site 후보인데 평가 프레임이 없는 outdoor recording
    outdoor_all = sorted(
        key for key in inventory
        if key.startswith(OUTDOOR_PREFIX) and key.count("/") == 4)
    eval_names = set(per_recording)
    adapt_candidates = []
    for key in outdoor_all:
        name = key[len(OUTDOOR_PREFIX):]
        if name in eval_names:
            continue
        adapt_candidates.append({
            "raw_recording": name,
            "raw_session_key": key,
            "recording_id": recording_of_session.get(key),
            "raw_rgb_total": inventory[key]["frame_count"],
            "raw_unique_images": len(sha_index.get(key, [])),
            "evaluation_frames": 0,
            "adaptation_eligible": True,
        })

    report = {
        "schema_version": "outdoor_eval_source_recordings_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "provenance_source": str(PROVENANCE.relative_to(REPO_ROOT)),
        "eval_sessions_considered": sorted(eval_session_names),
        "evaluation_frames_traced": len(seen_eval_frames),
        "rule": ("a raw recording that supplies even one evaluation frame is removed "
                 "from adaptation entirely; splitting one recording into an adapt half "
                 "and an eval half is forbidden"),
        "model_results_read": False,
        "eval_source_recordings": recordings,
        "outdoor_adaptation_candidates": adapt_candidates,
        "eval_frames_not_traceable_to_outdoor_raw": len(unmatched),
        "second_hop_note": ("import_provenance.csv records the GT folder as the source "
                            "for most frames, so the path alone does not reach the raw "
                            "recording; those frames are linked by image content SHA256"),
        "non_outdoor_source_examples": sorted(
            {u["source_image_path"].rsplit("/", 2)[0] for u in unmatched}),
    }
    (out_dir / "OUTDOOR_EVAL_SOURCE_RECORDINGS.json").write_text(
        json.dumps(report, indent=2) + "\n")

    print(f"평가 프레임 추적 {len(seen_eval_frames)}  "
          f"(그중 outdoor 원본 {sum(len(v) for v in per_recording.values())})")
    print(f"\n{'raw recording':22}{'REC':9}{'eval frames':>12}{'raw RGB':>9}  eval sessions")
    print("-" * 84)
    for entry in recordings:
        print(f"{entry['raw_recording']:22}{str(entry['recording_id']):9}"
              f"{entry['evaluation_frames']:12d}{entry['raw_rgb_total']:9d}  "
              f"{', '.join(entry['eval_sessions'])}")
    print(f"\nadaptation 후보 (평가 프레임 0):")
    for entry in adapt_candidates:
        print(f"  {entry['raw_recording']:22}{str(entry['recording_id']):9}"
              f"{entry['raw_rgb_total']:9d}")
    print(f"\noutdoor raw 로 못 잇는 평가 프레임 {len(unmatched)}")
    for path in report["non_outdoor_source_examples"]:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
