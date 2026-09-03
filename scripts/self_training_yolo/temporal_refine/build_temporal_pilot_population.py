"""TEMPORAL PILOT — center frame 과 7-frame tracklet 을 구성한다.

    python3 scripts/self_training_yolo/temporal_refine/build_temporal_pilot_population.py \
        --output-dir data/pallet/results/paper_temporal_selftrain_v1/pilot

출력  TEMPORAL_PILOT_POPULATION.csv

center 는 수동 어노가 있고 raw 촬영본으로 SHA 역추적되는 plastic 프레임이다.
PAPER_EVAL 319 에 프레임을 대는 recording 은 **통째로** 제외한다 — 이 method 의
확인용 population 을 미리 소모하지 않기 위해서다.  깨진 것으로 이미 기록된
`pallet11_gt` 와 평가 부적격(FT_OVERLAP) 프레임도 center 로 쓰지 않는다.

이웃 프레임은 같은 촬영본 안에서 **타임스탬프 순서**로 잡는다.  빠진 프레임을
보간하지 않는다.  경계를 넘지 않는다.

깊이를 읽지 않는다.  GT 좌표를 읽지 않는다(어노 경로만 기록한다).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE = REPO_ROOT / "data/evaluation/pallet_eval_v1"
RAW = REPO_ROOT / "data/pallet/raw_data"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def raw_sha_index() -> dict[str, Path]:
    index: dict[str, Path] = {}
    for root in (RAW / "outside", RAW / "night"):
        if not root.is_dir():
            continue
        for session in sorted(root.iterdir()):
            rgb = session / "rgb"
            if not rgb.is_dir():
                continue
            for image in rgb.iterdir():
                if image.suffix.lower() in IMAGE_SUFFIXES:
                    index[hashlib.sha256(image.read_bytes()).hexdigest()] = image
    return index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    lock = json.loads((out_dir.parent / "TEMPORAL_METHOD_LOCK.json").read_text())
    offsets = lock["tracklet"]["offsets"]
    excluded_recordings = set(
        lock["population"]["excluded"]["recordings_that_feed_PAPER_EVAL_319"])
    excluded_recordings.add("capturepallet11")
    object_type = lock["object_geometry"]["object_type"]
    print(f"tracklet offsets {offsets}")
    print(f"excluded recordings {sorted(excluded_recordings)}")

    sys.path.insert(0, str(REPO_ROOT / "scripts/evaluation"))
    from eval_workspace import load_frames, evaluation_population_views

    frames = load_frames(WORKSPACE)
    eligible = {r["frame_id"] for r in
                evaluation_population_views(frames)["PAPER_EVAL_POSITIVE"]}
    print("hashing raw recordings ...", flush=True)
    index = raw_sha_index()
    print(f"  raw images {len(index)}")

    # 촬영본별 타임스탬프 정렬 목록 (실제 취득 순서)
    ordered: dict[str, list[Path]] = {}

    rows = []
    for frame in frames:
        record = {
            "center_frame_id": frame.get("frame_id"),
            "source_recording": None, "center_rgb": None,
            "neighbor_rgb_paths": None,
            "gt_annotation_path": frame.get("annotation_path"),
            "lighting": frame.get("lighting"),
            "manual_gt_status": frame.get("source_dataset", ""),
            "paper_eval_recording_overlap": None,
            "eligible": False, "exclusion_reason": "",
        }
        if frame.get("is_annotated") != "true" or frame.get("is_positive") != "true":
            continue
        if frame.get("object_type") != "plastic":
            record["exclusion_reason"] = "not plastic"
            rows.append(record)
            continue
        # FT_OVERLAP 은 fine-tuning 과 겹쳐 평가에 쓸 수 없는 프레임이므로 center 금지.
        # UNVERIFIED_LEGACY 는 "깨졌다" 가 아니라 "v2 마이그레이션에서 재검증되지
        # 않았다" 는 뜻이라 배제 사유가 아니다 — lock 의 사전 계수 88 장도 이 기준이다.
        if frame.get("exclusion_reason") == "FT_OVERLAP":
            record["exclusion_reason"] = "workspace: FT_OVERLAP"
            rows.append(record)
            continue
        # PAPER_EVAL_POSITIVE 소속을 요구하면 안 된다.  이 pilot 은 PAPER_EVAL 을
        # 대는 촬영본을 통째로 빼는 것이 목적이므로, 두 조건은 서로 배타적이다.
        image = index.get(frame.get("source_image_sha256") or "")
        if image is None:
            record["exclusion_reason"] = "not resolvable to a raw recording by sha256"
            rows.append(record)
            continue

        session = image.parent.parent.name
        record["source_recording"] = session
        record["center_rgb"] = str(image.relative_to(REPO_ROOT))
        record["paper_eval_recording_overlap"] = session in excluded_recordings
        if session in excluded_recordings:
            record["exclusion_reason"] = (
                "recording feeds PAPER_EVAL 319" if session != "capturepallet11"
                else "known-broken annotation set")
            rows.append(record)
            continue

        if session not in ordered:
            ordered[session] = sorted(
                (p for p in image.parent.iterdir()
                 if p.suffix.lower() in IMAGE_SUFFIXES),
                key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
        sequence = ordered[session]
        position = sequence.index(image)
        neighbours, missing = [], False
        for offset in offsets:
            if offset == 0:
                continue
            j = position + offset
            if j < 0 or j >= len(sequence):
                missing = True
                break
            neighbours.append(str(sequence[j].relative_to(REPO_ROOT)))
        if missing:
            record["exclusion_reason"] = "tracklet would cross the recording boundary"
            rows.append(record)
            continue

        record["neighbor_rgb_paths"] = "|".join(neighbours)
        record["eligible"] = True
        record["object_type"] = object_type
        rows.append(record)

    fields = ["center_frame_id", "source_recording", "center_rgb",
              "neighbor_rgb_paths", "gt_annotation_path", "lighting",
              "manual_gt_status", "paper_eval_recording_overlap",
              "eligible", "exclusion_reason"]
    with (out_dir / "TEMPORAL_PILOT_POPULATION.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})

    keep = [r for r in rows if r["eligible"]]
    import collections
    print(f"\neligible centres {len(keep)}")
    print(f"  recordings {dict(sorted(collections.Counter(r['source_recording'] for r in keep).items()))}")
    print(f"  lighting   {dict(collections.Counter(r['lighting'] for r in keep))}")
    print(f"  paper_eval frame overlap {sum(1 for r in keep if r['center_frame_id'] in eligible and r['paper_eval_recording_overlap'])}")
    reasons = collections.Counter(r["exclusion_reason"] for r in rows if not r["eligible"])
    print("  excluded:")
    for reason, count in reasons.most_common():
        print(f"    {count:5d}  {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
