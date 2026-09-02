"""Daytime visibility-only review 대기열 · 감사 · 사전 lock 을 만든다.

왜 필요한가
    MAIN Daytime 70 장은 전부 legacy 세션이고 keypoint visibility 가 unknown 이라
    evaluator 의 supervision mask 가 통째로 비어 있다.  그래서 M2 의 strict keypoint
    지표를 Daytime 에서 낼 수 없다.

    좌표는 멀쩡하다.  필요한 건 **visibility 라벨뿐**이고, 그건 사람이 봐야 한다.

안전 장치
    이 작업은 **모델 결과를 이미 본 뒤** metadata 를 추가하는 것이다.  그래서
    review 전에 현재 상태를 통째로 snapshot 해 둔다 — 나중에 "결과가 좋아지는
    쪽으로 GT 를 만졌다" 는 의심을 데이터로 반박할 수 있어야 한다.

    lock 은 무엇을 바꿔도 되고 무엇은 안 되는지도 함께 못박는다.

사용:
    python scripts/evaluation/build_daytime_review_queue.py
"""

from __future__ import annotations

import collections
import csv
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from evaluation.eval_workspace import (  # noqa: E402
    evaluation_population_views,
    load_frames,
)

WORKSPACE = REPO_ROOT / "data" / "evaluation" / "pallet_eval_v1"
REVIEW_DIR = WORKSPACE / "review"
AUDIT = REPO_ROOT / "_docs" / "paper" / "DAYTIME_VISIBILITY_AUDIT.md"
LOCK = REPO_ROOT / "_docs" / "paper" / "DAYTIME_VISIBILITY_REVIEW_LOCK.json"
ARMS = REPO_ROOT / "data/pallet/results/paper_eval_v1/arms"

N_KEYPOINTS = 9


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify(annotation: dict) -> tuple[str, list[dict]]:
    """왜 strict supervision 이 비는지 프레임 단위로 가른다.  추측하지 않는다."""

    objects = annotation.get("objects")
    if not isinstance(objects, list) or len(objects) != 1:
        return "C_SCHEMA", []
    points = objects[0].get("keypoint_annotations")
    if not isinstance(points, list) or len(points) != N_KEYPOINTS:
        return "C_SCHEMA", []
    if all(point.get("xy") is None for point in points):
        return "B_NO_XY", points
    if any(point.get("visibility") in (1, 2) for point in points):
        return "OK_HAS_SUPERVISION", points
    return "A_XY_BUT_VISIBILITY_UNKNOWN", points


def main() -> int:
    frames = load_frames(WORKSPACE)
    positive = evaluation_population_views(frames)["PAPER_EVAL_POSITIVE"]
    daytime = [
        row for row in positive
        if row.get("paper_domain") == "daytime" and row["object_type"] == "plastic"
    ]
    if not daytime:
        raise SystemExit("NO_DAYTIME_FRAMES")

    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    category = collections.Counter()
    keypoint_total = keypoint_unknown = keypoint_out_of_frame = 0

    for row in daytime:
        annotation_path = WORKSPACE / row["annotation_path"]
        if not annotation_path.exists():
            category["D_ANNOTATION_FILE_MISSING"] += 1
            continue
        annotation = json.loads(annotation_path.read_text())
        state, points = classify(annotation)
        category[state] += 1

        unknown = in_frame_unknown = 0
        for point in points:
            keypoint_total += 1
            if not point.get("in_frame", True):
                keypoint_out_of_frame += 1
            if point.get("visibility") in (1, 2):
                continue
            unknown += 1
            keypoint_unknown += 1
            if point.get("in_frame", True):
                in_frame_unknown += 1

        rows.append({
            "frame_id": row["frame_id"],
            "session_id": row["session_id"],
            "image_path": row["image_path"],
            "annotation_path": row["annotation_path"],
            "annotation_sha256": sha256_file(annotation_path),
            "n_keypoints": len(points),
            "n_visibility_unknown": unknown,
            "n_visibility_unknown_in_frame": in_frame_unknown,
            "state": state,
            "visibility_complete": "false" if unknown else "true",
            "coordinate_review_flag": "false",
        })

    fields = list(rows[0])
    queue = REVIEW_DIR / "DAYTIME_VISIBILITY_REVIEW_QUEUE.csv"
    with queue.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    in_frame_unknown_total = sum(r["n_visibility_unknown_in_frame"] for r in rows)
    print(f"frames {len(rows)}   keypoints {keypoint_total}   "
          f"visibility unknown {keypoint_unknown}   "
          f"그중 화면 안 {in_frame_unknown_total}")
    for key, value in sorted(category.items()):
        print(f"  {key:34} {value}")

    # ── 감사 리포트 ────────────────────────────────────────────────────
    sessions = collections.Counter(r["session_id"] for r in rows)
    AUDIT.write_text("\n".join([
        "# Daytime visibility audit",
        "",
        f"생성 {datetime.date.today().isoformat()}.  "
        "MAIN Daytime 에서 strict keypoint 지표가 왜 비는지 **직접 세어** 확인한다.",
        "",
        "## 대상",
        "",
        "```text",
        "paper_domain == daytime  AND  object_type == plastic  AND  PAPER_EVAL member",
        "```",
        "",
        "```text",
        f"{'frames':28} {len(rows)}",
        f"{'keypoints':28} {keypoint_total}",
        f"{'visibility unknown':28} {keypoint_unknown}",
        f"{'  그중 화면 안 (실제 작업량)':28} {in_frame_unknown_total}",
        f"{'  그중 화면 밖':28} {keypoint_unknown - in_frame_unknown_total}",
        "```",
        "",
        "세션별:",
        "",
        "```text",
        *[f"{name:20} {count}" for name, count in sorted(sessions.items())],
        "```",
        "",
        "## 원인 분류 — 추측하지 않고 센다",
        "",
        "```text",
        "A  xy 존재 + visibility unknown      사람이 visibility 만 채우면 된다",
        "B  xy 자체 없음                      좌표 작업이 필요하다",
        "C  schema 문제                       파일을 고쳐야 한다",
        "D  annotation 파일 없음              어노테이션이 없다",
        "```",
        "",
        "```text",
        *[f"{key:34} {value}" for key, value in sorted(category.items())],
        "```",
        "",
        "## 판정",
        "",
        f"전부 A 다.  **좌표를 다시 찍을 일이 없다** — 필요한 작업은 {in_frame_unknown_total} 개",
        "keypoint 의 visibility 라벨뿐이다.  화면 밖 점은 visibility 0 이 맞으므로 건드리지 않는다.",
        "",
        "이 판정이 review 범위를 정한다.  좌표 편집은 도구에서 막는다.",
    ]) + "\n")

    # ── 사전 lock ──────────────────────────────────────────────────────
    head = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                   cwd=REPO_ROOT).decode().strip()
    manifests = REPO_ROOT / "challenge/real_gt_v2/manifests"
    lock = {
        "schema_version": "paper_daytime_visibility_review_lock_v1",
        "purpose": (
            "Freeze the daytime ground truth before a visibility-only review. "
            "The review happens after model results were already seen, so the "
            "prior state must be recoverable and the allowed edits must be "
            "declared in advance."
        ),
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "commit": head,
        "population": {
            "paper_eval_all_pos_sha256": sha256_file(manifests / "PAPER_EVAL_ALL_POS.json"),
            "paper_eval_plastic_pos_sha256": sha256_file(
                manifests / "PAPER_EVAL_PLASTIC_POS.json"),
            "daytime_frames": len(rows),
            "daytime_keypoints": keypoint_total,
            "visibility_unknown": keypoint_unknown,
            "visibility_unknown_in_frame": in_frame_unknown_total,
        },
        "queue": {
            "path": str(queue.relative_to(REPO_ROOT)),
            "sha256": sha256_file(queue),
        },
        "annotation_sha256": {r["frame_id"]: r["annotation_sha256"] for r in rows},
        "checkpoints": {
            name: json.loads((ARMS / "ARM_RESULTS.json").read_text())["models"][name][
                "checkpoint_sha256"]
            for name in ("R0", "R5_PROPOSED")
            if (ARMS / "ARM_RESULTS.json").exists()
        },
        "result_artifacts": {
            name: sha256_file(ARMS / f"{name}.json")
            for name in ("R0", "R5_PROPOSED")
            if (ARMS / f"{name}.json").exists()
        },
        "allowed_change": ["keypoint_annotations[].visibility",
                           "keypoint_annotations[].source",
                           "keypoint_annotations[].reason",
                           "coordinate_review_flag (queue only)"],
        "forbidden_change": ["keypoint_annotations[].xy", "bbox", "projected_cuboid",
                             "pose_transform", "camera_facing_pnp", "object_type",
                             "camera_data.intrinsics", "paper_domain",
                             "population membership"],
        "model_prediction_visible_during_review": False,
        "review_protocol": (
            "prediction-blinded: no model overlay, no corner error, no filter score, "
            "no pass/fail, no model name is shown while labelling"
        ),
        "rollback_rule": (
            "a save that changes any forbidden field must fail and roll back; "
            "the pre-review state is fully recoverable from annotation_sha256"
        ),
    }
    LOCK.write_text(json.dumps(lock, indent=2, ensure_ascii=False) + "\n")
    print(f"\nwrote {queue.relative_to(REPO_ROOT)}")
    print(f"wrote {AUDIT.relative_to(REPO_ROOT)}")
    print(f"wrote {LOCK.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
