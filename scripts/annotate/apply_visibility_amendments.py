"""사람이 확인한 visibility 를 workspace GT 에 반영한다.  허용 필드 외엔 롤백한다.

review 도구는 GT 를 건드리지 않고 amendment layer 에만 쌓는다.  실제 반영은 여기서
하고, 여기서 **저장 전후 diff 를 검사**한다.

    허용   keypoint_annotations[].visibility / source / reason
    금지   xy · bbox · projected_cuboid · pose_transform · camera_facing_pnp ·
           object_type · camera_data · 그 밖의 모든 필드

금지 필드가 한 글자라도 바뀌면 그 프레임을 되돌리고 실패로 센다.  좌표가 조용히
바뀌는 사고가 이 저장소에 있었기 때문에, 선언이 아니라 검사로 막는다.

legacy source GT(`challenge/real_gt_v2/migrated_gt*`)는 건드리지 않는다.  workspace
사본만 갱신한다.

사용:
    python scripts/annotate/apply_visibility_amendments.py --dry-run
    python scripts/annotate/apply_visibility_amendments.py --apply
"""

from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = REPO_ROOT / "data" / "evaluation" / "pallet_eval_v1"
AMENDMENTS = WORKSPACE / "review" / "DAYTIME_VISIBILITY_AMENDMENTS.json"
QUEUE = WORKSPACE / "review" / "DAYTIME_VISIBILITY_REVIEW_QUEUE.csv"
LOCK = REPO_ROOT / "_docs" / "paper" / "DAYTIME_VISIBILITY_REVIEW_LOCK.json"
COORD_QA = WORKSPACE / "review" / "DAYTIME_COORDINATE_QA_QUEUE.csv"

STATES = {"v": 2, "o": 1, "t": 0, "u": None}
ALLOWED_KEYPOINT_FIELDS = {"visibility", "source", "reason"}
REVIEW_SOURCE = "human_visibility_review"
REVIEW_REASON = "prediction_blinded_visibility_review_2026_09"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def forbidden_diff(before: dict, after: dict) -> list[str]:
    """허용 필드만 빼고 완전히 같은지 본다.  같지 않으면 경로를 돌려준다."""

    trimmed_before = copy.deepcopy(before)
    trimmed_after = copy.deepcopy(after)
    for payload in (trimmed_before, trimmed_after):
        for point in payload["objects"][0]["keypoint_annotations"]:
            for field in ALLOWED_KEYPOINT_FIELDS:
                point.pop(field, None)
    if trimmed_before == trimmed_after:
        return []

    problems: list[str] = []

    def walk(left, right, path: str) -> None:
        if type(left) is not type(right):
            problems.append(f"{path}: type {type(left).__name__} -> {type(right).__name__}")
            return
        if isinstance(left, dict):
            for key in set(left) | set(right):
                if key not in left:
                    problems.append(f"{path}.{key}: added")
                elif key not in right:
                    problems.append(f"{path}.{key}: removed")
                else:
                    walk(left[key], right[key], f"{path}.{key}")
        elif isinstance(left, list):
            if len(left) != len(right):
                problems.append(f"{path}: length {len(left)} -> {len(right)}")
                return
            for index, (a, b) in enumerate(zip(left, right)):
                walk(a, b, f"{path}[{index}]")
        elif left != right:
            problems.append(f"{path}: {left!r} -> {right!r}")

    walk(trimmed_before, trimmed_after, "")
    return problems or ["unspecified difference"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="실제로 쓴다")
    parser.add_argument("--dry-run", action="store_true", help="검사만 한다 (기본)")
    args = parser.parse_args()
    write = bool(args.apply) and not args.dry_run

    if not AMENDMENTS.exists():
        raise SystemExit(f"AMENDMENTS_NOT_FOUND: {AMENDMENTS}")
    amendments = json.loads(AMENDMENTS.read_text())
    queue = {row["frame_id"]: row for row in __import__("csv").DictReader(
        QUEUE.open(encoding="utf-8"))}
    lock = json.loads(LOCK.read_text()) if LOCK.exists() else {}
    locked_sha = lock.get("annotation_sha256", {})

    tally = collections.Counter()
    coordinate_flags: list[dict] = []
    problems_seen: list[str] = []

    for frame_id, amendment in amendments.get("frames", {}).items():
        decisions = {k: v for k, v in amendment.items() if k.isdigit()}
        if amendment.get("coordinate_review_flag"):
            coordinate_flags.append({
                "frame_id": frame_id,
                "session_id": queue.get(frame_id, {}).get("session_id", ""),
                "image_path": queue.get(frame_id, {}).get("image_path", ""),
                "reason": "flagged during visibility review; coordinates NOT modified",
            })
        if not decisions:
            tally["frames_without_decision"] += 1
            continue
        row = queue.get(frame_id)
        if row is None:
            tally["frame_not_in_queue"] += 1
            problems_seen.append(f"{frame_id}: queue 에 없다")
            continue

        path = WORKSPACE / row["annotation_path"]
        current_sha = sha256_file(path)
        expected_sha = locked_sha.get(frame_id)
        if expected_sha and current_sha != expected_sha:
            # lock 이후 GT 가 바뀐 상태다.  덮어쓰지 않고 멈춘다.
            tally["frame_changed_since_lock"] += 1
            problems_seen.append(f"{frame_id}: lock 이후 GT 가 바뀌었다")
            continue

        before = json.loads(path.read_text())
        after = copy.deepcopy(before)
        points = after["objects"][0]["keypoint_annotations"]
        changed = 0
        for key, state in decisions.items():
            index = int(key)
            if state not in STATES or not (0 <= index < len(points)):
                tally["invalid_decision"] += 1
                continue
            value = STATES[state]
            if value is None:  # unknown = 보류, 쓰지 않는다
                tally["held_unknown"] += 1
                continue
            points[index]["visibility"] = value
            points[index]["source"] = REVIEW_SOURCE
            points[index]["reason"] = REVIEW_REASON
            changed += 1

        problems = forbidden_diff(before, after)
        if problems:
            tally["ROLLBACK_FORBIDDEN_CHANGE"] += 1
            problems_seen.extend(f"{frame_id}: {p}" for p in problems[:4])
            continue

        tally["frames_ok"] += 1
        tally["keypoints_written"] += changed
        if write and changed:
            path.write_text(json.dumps(after, indent=2, ensure_ascii=False) + "\n")

    print("apply" if write else "dry-run")
    for key, value in sorted(tally.items()):
        print(f"  {key:34} {value}")
    if problems_seen:
        print("\n  문제:")
        for line in problems_seen[:10]:
            print(f"    {line}")

    if coordinate_flags:
        COORD_QA.parent.mkdir(parents=True, exist_ok=True)
        with COORD_QA.open("w", encoding="utf-8", newline="") as handle:
            writer = __import__("csv").DictWriter(
                handle, fieldnames=list(coordinate_flags[0]))
            writer.writeheader()
            writer.writerows(coordinate_flags)
        print(f"\n  좌표 의심 {len(coordinate_flags)} 프레임 -> "
              f"{COORD_QA.relative_to(REPO_ROOT)}")
        print("  좌표는 자동으로 고치지 않는다.  별도 QA 대상이다.")

    if tally["ROLLBACK_FORBIDDEN_CHANGE"] or problems_seen:
        return 2
    if not write:
        print("\n  --apply 를 주면 실제로 쓴다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
