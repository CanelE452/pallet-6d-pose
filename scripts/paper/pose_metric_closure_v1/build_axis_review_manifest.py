"""사람이 물리 긴축을 확인할 319 프레임 manifest 를 만든다.

원본 annotation 을 읽기만 한다.  사람 판정은 별도 sidecar 에 저장되고, 이 스크립트는
그 sidecar 를 건드리지 않는다.

    python3 scripts/paper/pose_metric_closure_v1/build_axis_review_manifest.py

출력: data/pallet/results/paper_pose_metric_closure_v1/
        AXIS_REVIEW_MANIFEST.json
        AXIS_REVIEW_PROGRESS.json

축 정의 (camera-facing 0123 convention)
    Axis A = camera-facing WIDTH   edges (0,1) (2,3) (4,5) (6,7)
    Axis B = camera-facing DEPTH   edges (0,4) (1,5) (2,6) (3,7)

사람에게 묻는 것은 하나뿐이다 — **물리적으로 긴 변이 A 인가 B 인가.**
180 도 부호는 묻지 않는다 (평가를 180 도 등가류로 정의하므로).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "evaluation"))

WORKSPACE = REPO_ROOT / "data/evaluation/pallet_eval_v1"
REGISTRY = REPO_ROOT / "challenge/real_gt_v2/OBJECT_GEOMETRY_REGISTRY.json"
OUT_DIR = REPO_ROOT / "data/pallet/results/paper_pose_metric_closure_v1"
MANIFEST = OUT_DIR / "AXIS_REVIEW_MANIFEST.json"
PROGRESS = OUT_DIR / "AXIS_REVIEW_PROGRESS.json"

AXIS_A_EDGES = [[0, 1], [2, 3], [4, 5], [6, 7]]
AXIS_B_EDGES = [[0, 4], [1, 5], [2, 6], [3, 7]]

# camera_dynamic 규약에서는 카메라 기준 축이 프레임마다 다시 유도된다.
# 따라서 같은 세션이라도 parity 가 프레임마다 바뀔 수 있어 session propagation 은
# 기본적으로 금지한다 (§9 의 "조금이라도 바뀔 수 있으면 disable").
SESSION_PROPAGATION_ALLOWED = False
CAMERA_DYNAMIC_FRAME = "camera_dynamic_0123_v4"


def load_registry() -> dict[str, dict]:
    payload = json.loads(REGISTRY.read_text())
    table: dict[str, dict] = {}
    for entry in payload["objects"]:
        dims = entry["physical_dimensions_m"]
        footprint = sorted((float(dims["x"]), float(dims["z"])))
        table[entry["object_type"]] = {
            "object_type": entry["object_type"],
            "display_name": entry["display_name"],
            "physical_short_m": footprint[0],
            "physical_long_m": footprint[1],
            "height_m": float(dims["y"]),
        }
    return table


def resolve_object_type(payload: dict, obj: dict, registry: dict[str, dict]) -> str | None:
    """명시 필드가 있으면 그것, 없으면 저장된 치수로 판정한다.

    PAPER_EVAL 319 중 129 장은 `object_type` 이 비어 있는 legacy 프레임이다.
    """

    explicit = obj.get("object_type") or payload.get("object_type")
    if explicit in registry:
        return explicit
    dims = obj.get("physical_dimensions_m") or obj.get("dimensions_m") or {}
    values = [float(v) for v in dims.values()] if dims else []
    if len(values) != 3:
        return None
    footprint = sorted(v for v in values if v > 0.2)  # 높이(0.11 / 0.14)를 뺀다
    if len(footprint) != 2:
        return None
    for name, spec in registry.items():
        if (abs(footprint[0] - spec["physical_short_m"]) < 0.02
                and abs(footprint[1] - spec["physical_long_m"]) < 0.02):
            return name
    return None


def stored_long_axis(obj: dict, spec: dict) -> str | None:
    """어노에 이미 들어 있는 camera-facing parity.

    ⚠️ 이 값은 **사람에게 보여주지 않는다.**  사람 판정을 편향시키기 때문이다.
    검수가 끝난 뒤 '기존 저장값과 사람이 얼마나 다른가' 를 감사하는 용도로만 쓴다.
    `axis_assignment_confirmed` 가 319/319 False 이므로 이 값은 미확인 상태다.
    """

    dims = obj.get("dimensions_m") or {}
    width, depth = dims.get("width"), dims.get("depth")
    if width is None or depth is None:
        return None
    long_m = spec["physical_long_m"]
    if abs(float(width) - long_m) < 0.02:
        return "CF_WIDTH"
    if abs(float(depth) - long_m) < 0.02:
        return "CF_DEPTH"
    return None


def main() -> int:
    from eval_workspace import load_frames, evaluation_population_views

    registry = load_registry()
    rows = evaluation_population_views(load_frames(WORKSPACE))["PAPER_EVAL_POSITIVE"]

    frames: list[dict] = []
    unresolved: list[str] = []
    non_dynamic: list[str] = []
    for row in rows:
        annotation_path = WORKSPACE / row["annotation_path"]
        payload = json.loads(annotation_path.read_text())
        obj = payload["objects"][0]
        object_type = resolve_object_type(payload, obj, registry)
        if object_type is None:
            unresolved.append(row["frame_id"])
            continue
        spec = registry[object_type]
        if obj.get("keypoint_frame") != CAMERA_DYNAMIC_FRAME:
            non_dynamic.append(row["frame_id"])

        points = obj.get("keypoint_annotations") or []
        keypoints = [p.get("xy") for p in points[:9]]

        frames.append({
            "frame_id": row["frame_id"],
            "session_id": row.get("session_id"),
            "paper_domain": row.get("paper_domain"),
            "object_type": object_type,
            "display_name": spec["display_name"],
            "image": str((WORKSPACE / row["image_path"]).relative_to(REPO_ROOT)),
            "annotation": str(annotation_path.relative_to(REPO_ROOT)),
            "physical_long_m": spec["physical_long_m"],
            "physical_short_m": spec["physical_short_m"],
            "physical_long_cm": round(spec["physical_long_m"] * 100),
            "physical_short_cm": round(spec["physical_short_m"] * 100),
            "camera_facing_axis_A": "WIDTH",
            "camera_facing_axis_A_edges": AXIS_A_EDGES,
            "camera_facing_axis_B": "DEPTH",
            "camera_facing_axis_B_edges": AXIS_B_EDGES,
            "keypoints_xy": keypoints,
            "keypoint_frame": obj.get("keypoint_frame"),
            "_hidden_stored_long_axis": stored_long_axis(obj, spec),
            "review": None,
        })

    by_object: dict[str, int] = {}
    by_domain: dict[str, int] = {}
    by_session: dict[str, int] = {}
    for frame in frames:
        by_object[frame["object_type"]] = by_object.get(frame["object_type"], 0) + 1
        key = str(frame["paper_domain"])
        by_domain[key] = by_domain.get(key, 0) + 1
        by_session[str(frame["session_id"])] = by_session.get(str(frame["session_id"]), 0) + 1

    manifest = {
        "schema_version": "physical_axis_review_manifest_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "population": "PAPER_EVAL_POSITIVE",
        "expected_total": len(rows),
        "total": len(frames),
        "unresolved_object_type": unresolved,
        "frames_not_camera_dynamic": non_dynamic,
        "counts": {"by_object": by_object, "by_domain": by_domain, "by_session": by_session},
        "axis_definition": {
            "A": {"name": "camera-facing WIDTH", "edges": AXIS_A_EDGES},
            "B": {"name": "camera-facing DEPTH", "edges": AXIS_B_EDGES},
            "convention": "camera-facing 0123: 0-3 near face, 4-7 far face, {0,1,4,5} up, 8 centroid",
        },
        "question_asked_of_the_reviewer":
            "which axis is the physical LONG side — A or B",
        "not_asked": "the 180-degree sign; the evaluation defines orientation as a 180-degree equivalence class",
        "session_propagation_allowed": SESSION_PROPAGATION_ALLOWED,
        "session_propagation_reason":
            "every frame uses keypoint_frame camera_dynamic_0123_v4, so the camera-facing axes are "
            "re-derived per frame and the parity can change within a session. Propagating one answer "
            "across a session could silently mislabel frames.",
        "hidden_field_policy":
            "_hidden_stored_long_axis carries the parity already stored in the annotation. It is NEVER "
            "shown in the review GUI, because seeing it would bias the reviewer. It exists so that after "
            "the review we can measure how often the unverified stored value disagrees with a human. "
            "axis_assignment_confirmed is False on all 319 frames, so the stored value is unverified.",
        "source_annotations_modified": False,
        "frames_list": frames,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")

    PROGRESS.write_text(json.dumps({
        "schema_version": "physical_axis_review_progress_v1",
        "manifest": str(MANIFEST.relative_to(REPO_ROOT)),
        "total": len(frames),
        "reviewed": 0,
        "confirmed": 0,
        "unclear": 0,
        "last_index": 0,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n")

    print(f"frames            {len(frames)} / {len(rows)}")
    for name, count in sorted(by_object.items()):
        spec = registry[name]
        print(f"  {name:32} {count:4d}   long {spec['physical_long_m']:.2f} m"
              f" / short {spec['physical_short_m']:.2f} m")
    print(f"domains           {by_domain}")
    print(f"unresolved type   {len(unresolved)}")
    print(f"non camera-dynamic{len(non_dynamic)}")
    print(f"session propagation allowed: {SESSION_PROPAGATION_ALLOWED}")
    print(f"wrote {MANIFEST.relative_to(REPO_ROOT)}")
    return 0 if len(frames) == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
