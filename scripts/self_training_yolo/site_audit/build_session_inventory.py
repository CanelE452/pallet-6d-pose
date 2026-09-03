"""저장소의 real RGB 세션을 전수 열거하고 provenance 를 복원한다.  읽기 전용.

    python3 scripts/self_training_yolo/site_audit/build_session_inventory.py \
        --output-dir data/pallet/results/site_environment_audit_v1

출력  SESSION_INVENTORY.csv · SESSION_INVENTORY.json

폴더 이름을 장소로 간주하지 않는다.  이름은 약한 메타데이터일 뿐이라
session.json · camera_info · capture metadata · source path · promoted_from 을
따로 읽어 기록하고, 장소 판정은 이 스크립트가 하지 않는다.

모델 예측 · GT 정확도 · 성능 산출물은 읽지 않는다.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = REPO_ROOT / "data/pallet/results/site_environment_audit_v1"

# real RGB 가 존재할 수 있는 root.  여기 없는 곳은 조사 대상이 아니다.
SEARCH_ROOTS = [
    "data/pallet/raw_data",
    "data/evaluation/pallet_eval_v1",
    "challenge/data/01_real",
]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
# 합성 · 파생 · 결과물 · 오버레이는 real 촬영본이 아니다
EXCLUDE_PARTS = {"02_synthetic", "03_derived", "04_results", "results",
                 "overlays", "overlay", "visual_audit", "pl_review",
                 "contact_sheets", "__pycache__", "augmented",
                 # depth 는 촬영본이지만 RGB 가 아니다 — 이번 감사 대상 밖
                 "depth", "depth_raw", "aligned_depth"}
# provenance 를 담고 있을 수 있는 키
PROVENANCE_KEYS = ("promoted_from_sessions", "promoted_from", "source_session",
                   "source_sessions", "source_session_id", "original_recording",
                   "original_recording_id", "source_path", "provenance",
                   "source_archive", "imported_from", "source_set")

TIMESTAMP_RE = re.compile(r"^\d{16,19}$")


def is_excluded(path: Path) -> bool:
    """이름 앞의 밑줄은 무시하고 비교한다 — `_overlays` 도 `overlays` 다."""

    for part in path.parts:
        normalised = part.lstrip("_")
        if normalised in EXCLUDE_PARTS:
            return True
        # 추론 결과를 원본 옆에 복사해 둔 폴더 (infer_flipcompare 등)
        if normalised.startswith("infer_"):
            return True
    return False


def image_dirs(root: Path) -> dict[Path, list[Path]]:
    """이미지를 직접 담고 있는 디렉토리만 세션 후보로 본다."""

    seen: dict[Path, list[Path]] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if is_excluded(path):
            continue
        seen.setdefault(path.parent, []).append(path)
    return seen


def session_root_for(image_dir: Path) -> Path:
    """`.../<session>/rgb` 면 세션은 그 부모다."""

    return image_dir.parent if image_dir.name in {"rgb", "images", "image"} else image_dir


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def relative(path: Path) -> str | None:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return None


def collect_metadata(session_dir: Path) -> dict:
    """세션 폴더와 그 상위에서 메타데이터와 provenance 를 긁는다."""

    found: dict = {"files": [], "provenance": {}, "intrinsics": None}
    candidates = [p for p in session_dir.glob("*") if p.is_file()]
    for parent in (session_dir.parent, session_dir.parent.parent):
        if parent.is_dir() and parent != REPO_ROOT and REPO_ROOT in parent.parents:
            candidates += [p for p in parent.glob("*")
                           if p.is_file()
                           and p.suffix in {".json", ".csv", ".txt", ".yaml", ".md"}]

    for path in candidates:
        if path.suffix.lower() in IMAGE_SUFFIXES:
            continue
        rel = relative(path)
        if rel is None:
            continue
        if path.suffix in {".json", ".txt", ".yaml", ".csv", ".md"}:
            found["files"].append(rel)
        if path.suffix == ".json":
            payload = read_json(path)
            if isinstance(payload, dict):
                for key in PROVENANCE_KEYS:
                    if key in payload:
                        found["provenance"].setdefault(key, []).append(
                            {"file": rel, "value": payload[key]})
                camera = payload.get("camera_data")
                intrinsics = (camera or {}).get("intrinsics") if isinstance(camera, dict) else None
                intrinsics = intrinsics or payload.get("intrinsics")
                if intrinsics and found["intrinsics"] is None:
                    found["intrinsics"] = intrinsics
        if path.name == "cam_K.txt" and found["intrinsics"] is None:
            found["intrinsics"] = {"cam_K_txt": rel, "text": path.read_text().strip()[:200]}
    return found


def resolution_of(path: Path) -> str:
    import cv2

    image = cv2.imread(str(path))
    return "unreadable" if image is None else f"{image.shape[1]}x{image.shape[0]}"


def intrinsics_hash(intrinsics) -> str | None:
    if not intrinsics:
        return None
    return hashlib.sha256(
        json.dumps(intrinsics, sort_keys=True).encode()).hexdigest()[:16]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sessions: list[dict] = []
    for root_name in SEARCH_ROOTS:
        root = REPO_ROOT / root_name
        if not root.exists():
            print(f"  root 없음: {root_name}")
            continue
        for image_dir, images in sorted(image_dirs(root).items()):
            session_dir = session_root_for(image_dir)
            names = sorted(p.name for p in images)
            stems = [Path(n).stem for n in names]
            meta = collect_metadata(session_dir)
            numeric = [s for s in stems if TIMESTAMP_RE.match(s)]

            sessions.append({
                "session_key": relative(session_dir),
                "session_name": session_dir.name,
                "image_dir": relative(image_dir),
                "root": root_name,
                "frame_count": len(images),
                "first_frame": names[0],
                "last_frame": names[-1],
                "stem_is_timestamp": bool(numeric) and len(numeric) == len(stems),
                "timestamp_min": min(numeric) if numeric else None,
                "timestamp_max": max(numeric) if numeric else None,
                "resolution": resolution_of(images[0]),
                "metadata_files": meta["files"][:40],
                "provenance": meta["provenance"],
                "intrinsics": meta["intrinsics"],
                "intrinsics_hash": intrinsics_hash(meta["intrinsics"]),
                "filename_list_sha256": hashlib.sha256(
                    "\n".join(names).encode()).hexdigest(),
            })

    report = {
        "schema_version": "session_inventory_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "search_roots": SEARCH_ROOTS,
        "excluded_path_parts": sorted(EXCLUDE_PARTS),
        "model_outputs_read": False,
        "gt_accuracy_read": False,
        "note": "folder names are weak metadata; this file assigns no site identity",
        "total_sessions": len(sessions),
        "total_frames": sum(s["frame_count"] for s in sessions),
        "sessions": sessions,
    }
    (out_dir / "SESSION_INVENTORY.json").write_text(json.dumps(report, indent=2) + "\n")

    fields = ["session_key", "session_name", "root", "frame_count", "resolution",
              "first_frame", "last_frame", "stem_is_timestamp", "timestamp_min",
              "timestamp_max", "intrinsics_hash", "has_provenance",
              "n_metadata_files", "filename_list_sha256"]
    with (out_dir / "SESSION_INVENTORY.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for session in sessions:
            row = {key: session.get(key) for key in fields}
            row["has_provenance"] = bool(session["provenance"])
            row["n_metadata_files"] = len(session["metadata_files"])
            writer.writerow(row)

    print(f"sessions {len(sessions)}   frames {report['total_frames']}")
    by_root: dict[str, int] = {}
    for session in sessions:
        by_root[session["root"]] = by_root.get(session["root"], 0) + 1
    for root, count in sorted(by_root.items()):
        print(f"  {root:44}{count:5d}")
    print(f"  provenance 필드 보유 세션 {sum(1 for s in sessions if s['provenance'])}")
    print(f"wrote {(out_dir / 'SESSION_INVENTORY.json').relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
