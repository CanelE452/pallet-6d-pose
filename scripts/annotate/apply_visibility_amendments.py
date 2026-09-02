"""사람이 확인한 visibility 를 workspace GT 에 반영한다.  허용 필드 외엔 롤백한다.

review 도구는 GT 를 건드리지 않고 amendment layer 에만 쌓는다.  실제 반영은 여기서
하고, 여기서 **저장 전후 diff 를 검사**한다.

    허용   keypoint_annotations[].visibility / reason
    금지   xy · source · bbox · projected_cuboid · pose_transform ·
           camera_facing_pnp · object_type · camera_data · 그 밖의 모든 필드

`source` 는 **좌표의 출처**(manual_click / pnp_projected / ...)다.  가시성만 본
리뷰는 좌표를 만들지 않았으므로 출처를 바꿀 자격이 없다.  `reason` 만 갱신한다.

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
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "annotate"))
from real_gt_v2_schema import validate_gt_v2  # noqa: E402
WORKSPACE = REPO_ROOT / "data" / "evaluation" / "pallet_eval_v1"
AMENDMENTS = WORKSPACE / "review" / "DAYTIME_VISIBILITY_AMENDMENTS.json"
QUEUE = WORKSPACE / "review" / "DAYTIME_VISIBILITY_REVIEW_QUEUE.csv"
LOCK = REPO_ROOT / "_docs" / "paper" / "DAYTIME_VISIBILITY_REVIEW_LOCK.json"
COORD_QA = WORKSPACE / "review" / "DAYTIME_COORDINATE_QA_QUEUE.csv"
AUTO_QUEUE = WORKSPACE / "review" / "DAYTIME_OCCLUSION_REVIEW_QUEUE.csv"

# 사전 고정된 자동 분류 -> (visibility, reason).
# 규정은 `_docs/paper/DAYTIME_OCCLUSION_AUTO_CLASSIFICATION.md` 에 있고 모델 결과를
# 보기 전에 정해졌다.  여기서는 그 판정을 옮겨 적을 뿐, 새로 정하지 않는다.
AUTO_STATES = {
    "AUTO_TRUNCATED": (0, "truncated"),
    "AUTO_SELF_OCCLUDED": (1, "occluded"),
    "AUTO_CENTROID_OCCLUDED": (1, "occluded"),
    "SELF_VISIBLE_CANDIDATE": (2, "visible"),
}

# 사람 판정 -> (visibility, reason).  둘 다 GT-v2 enum 안의 값이어야 한다.
STATES = {"v": (2, "visible"), "o": (1, "occluded"), "t": (0, "truncated"),
          "u": None}
ALLOWED_KEYPOINT_FIELDS = {"visibility", "reason"}


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


def read_auto_decisions() -> dict[str, dict[str, str]]:
    """자동 분류 큐에서 사람이 안 봐도 되는 판정만 모은다."""

    decisions: dict[str, dict[str, str]] = collections.defaultdict(dict)
    if not AUTO_QUEUE.exists():
        return {}
    for row in __import__("csv").DictReader(AUTO_QUEUE.open(encoding="utf-8")):
        if row.get("requires_human") == "true":
            continue  # 사람 판정은 amendment layer 로 온다
        status = row.get("final_auto_status", "")
        if status in AUTO_STATES:
            decisions[row["frame_id"]][row["kp_index"]] = status
    return dict(decisions)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="실제로 쓴다")
    parser.add_argument("--dry-run", action="store_true", help="검사만 한다 (기본)")
    parser.add_argument("--auto", action="store_true",
                        help="사전 고정된 자동 분류도 함께 반영한다")
    parser.add_argument("--amendments", default=None)
    parser.add_argument("--queue", default=None)
    parser.add_argument("--auto-queue", default=None)
    parser.add_argument("--lock", default=None)
    args = parser.parse_args()

    global AMENDMENTS, QUEUE, AUTO_QUEUE, LOCK
    if args.amendments: AMENDMENTS = Path(args.amendments).resolve()
    if args.queue:      QUEUE = Path(args.queue).resolve()
    if args.auto_queue: AUTO_QUEUE = Path(args.auto_queue).resolve()
    if args.lock:       LOCK = Path(args.lock).resolve()
    write = bool(args.apply) and not args.dry_run

    amendments = (json.loads(AMENDMENTS.read_text())
                  if AMENDMENTS.exists() else {"frames": {}})
    queue = {row["frame_id"]: row for row in __import__("csv").DictReader(
        QUEUE.open(encoding="utf-8"))}
    lock = json.loads(LOCK.read_text()) if LOCK.exists() else {}
    locked_sha = lock.get("annotation_sha256", {})

    auto = read_auto_decisions() if args.auto else {}

    tally = collections.Counter()
    coordinate_flags: list[dict] = []
    problems_seen: list[str] = []

    frames = dict(amendments.get("frames", {}))
    for frame_id in auto:
        frames.setdefault(frame_id, {})

    for frame_id, amendment in frames.items():
        decisions = {k: v for k, v in amendment.items() if k.isdigit()}
        if amendment.get("coordinate_review_flag"):
            coordinate_flags.append({
                "frame_id": frame_id,
                "session_id": queue.get(frame_id, {}).get("session_id", ""),
                "image_path": queue.get(frame_id, {}).get("image_path", ""),
                "reason": "flagged during visibility review; coordinates NOT modified",
            })
        if not decisions and not auto.get(frame_id):
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
            decision = STATES[state]
            if decision is None:  # unknown = 보류, 쓰지 않는다
                tally["held_unknown"] += 1
                continue
            visibility, reason = decision
            points[index]["visibility"] = visibility
            points[index]["reason"] = reason
            changed += 1

        # 자동 분류는 **아직 미상(visibility 0)인 점에만** 쓴다.  이미 판정이 있는
        # 점을 기하 추정으로 덮지 않는다.
        for key, status in auto.get(frame_id, {}).items():
            index = int(key)
            if not (0 <= index < len(points)):
                tally["invalid_decision"] += 1
                continue
            if key in decisions:
                tally["auto_superseded_by_human"] += 1
                continue
            if points[index].get("visibility") != 0:
                tally["auto_skipped_already_decided"] += 1
                continue
            visibility, reason = AUTO_STATES[status]
            points[index]["visibility"] = visibility
            points[index]["reason"] = reason
            changed += 1
            tally["auto_written"] += 1

        problems = forbidden_diff(before, after)
        try:
            # 쓰기 전에 스키마를 통과하는지 본다.  enum 밖의 값을 넣어 평가기를
            # 통째로 못 돌게 만든 사고가 있었다.
            validate_gt_v2(after)
        except Exception as exc:  # noqa: BLE001 - 스키마 오류 문구를 그대로 보고한다
            problems = problems + [f"schema: {exc}"]
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
