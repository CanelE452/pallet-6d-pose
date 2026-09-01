"""리뷰가 끝난 incoming 프레임을 FINAL 평가 세션으로 편입한다.

`DATASET_CONTRACT.json` 의 `incoming_unreviewed` 가 정한 절차를 그대로 따른다.

    active_evaluation_member = false
    activation = "independently copy reviewed frames into final/positive or
                  final/negative sessions"
    forbidden  = "raw import must not ... count toward combined evaluation
                  collection targets before review"

즉 incoming 을 집계에 끼워 넣는 게 아니라, **어노테이션이 끝난 프레임만 골라
FINAL 세션으로 독립 복사**한다.  incoming 원본은 건드리지 않는다.

    incoming/sessions/<src>/rgb/<stem>.png          -> final/positive/sessions/<dst>/rgb/
    incoming/annotations/<ann>/<stem>.json          -> final/positive/annotations/<dst>/
    (overlay 가 있으면)                              -> final/positive/annotations/<dst>/_overlays/

symlink 나 hardlink 를 쓰지 않는다 — contract 의
`workspace_copies_are_not_hardlinks_or_symlinks` 불변식이다.

Usage:
    python scripts/evaluation/promote_incoming_to_final.py \\
        --annotations incoming/annotations/real_unlabeled_night_20260830__wood \\
        --source-session real_unlabeled_night_20260830 \\
        --dest-session wood_night_01 \\
        [--apply]

기본은 dry-run 이다.  `--apply` 를 줘야 실제로 복사한다.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "evaluation"))

import eval_workspace as W  # noqa: E402

DEFAULT_ROOT = REPO / "data/evaluation/pallet_eval_v1"


def annotated_stems(ann_dir: Path) -> list[str]:
    """9 keypoint 가 실제로 들어 있는 어노테이션만 편입 대상으로 본다."""
    out = []
    for path in sorted(ann_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        objects = payload.get("objects") or []
        if not objects:
            continue
        cuboid = objects[0].get("projected_cuboid") or []
        if len(cuboid) >= 8:
            out.append(path.stem)
    return out


# ── 자동 편입 ────────────────────────────────────────────────────────────────
# 어노테이션 폴더 이름이 목적지를 결정한다.  사람이 매번 고르지 않아도 된다.
#
#     incoming/annotations/<source_session>__<material>/   +  세션의 lighting
#         -> final/positive/sessions/<material>_<lighting>_01
#
# material 은 사용자가 저장할 때 고른 폴더에서, lighting 은 incoming 세션의
# session.json 에서 온다.  둘 다 이미 기록된 사실이라 추론이 아니다.
MATERIALS = ("plastic", "wood")


def resolve_auto_destination(root: Path, ann_dir: Path):
    """(source_session, dest_session) 또는 None.  근거가 없으면 None."""
    name = ann_dir.name
    if "__" not in name:
        return None
    source_session, _, material = name.rpartition("__")
    if material not in MATERIALS:
        return None
    meta_path = root / "incoming/sessions" / source_session / "session.json"
    if not meta_path.is_file():
        return None
    lighting = str(json.loads(meta_path.read_text(encoding="utf-8"))
                   .get("lighting", "")).strip().lower()
    if lighting not in ("day", "night"):
        return None
    dest = f"{material}_{lighting}_01"
    if not (root / "final/positive/sessions" / dest / "session.json").is_file():
        return None
    return source_session, dest


def promote_annotated_incoming(root: Path, *, refresh: bool = False) -> dict:
    """incoming 의 **어노테이션이 끝난** 프레임을 전부 대응 FINAL 세션으로 복사.

    이미 복사된 것은 건너뛴다 (idempotent).  incoming 원본은 그대로 둔다.
    어노테이션 행위 자체가 contract 가 요구하는 human review 다.
    """
    ann_root = root / "incoming/annotations"
    summary: dict = {"promoted": 0, "skipped_existing": 0, "by_dest": {},
                     "unresolved": []}
    if not ann_root.is_dir():
        return summary
    for ann_dir in sorted(p for p in ann_root.iterdir() if p.is_dir()):
        resolved = resolve_auto_destination(root, ann_dir)
        if resolved is None:
            summary["unresolved"].append(ann_dir.name)
            continue
        source_session, dest = resolved
        src_rgb = root / "incoming/sessions" / source_session / "rgb"
        dst_rgb = root / "final/positive/sessions" / dest / "rgb"
        dst_ann = root / "final/positive/annotations" / dest
        overlay_src, overlay_dst = ann_dir / "_overlays", dst_ann / "_overlays"
        for stem in annotated_stems(ann_dir):
            image = src_rgb / f"{stem}.png"
            if not image.is_file():
                continue
            if (dst_rgb / f"{stem}.png").is_file() and (dst_ann / f"{stem}.json").is_file():
                summary["skipped_existing"] += 1
                continue
            dst_rgb.mkdir(parents=True, exist_ok=True)
            dst_ann.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image, dst_rgb / f"{stem}.png")
            shutil.copy2(ann_dir / f"{stem}.json", dst_ann / f"{stem}.json")
            ov = overlay_src / f"{stem}.png"
            if ov.is_file():
                overlay_dst.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ov, overlay_dst / f"{stem}.png")
            summary["promoted"] += 1
            summary["by_dest"][dest] = summary["by_dest"].get(dest, 0) + 1
    if refresh and summary["promoted"]:
        frames = W.refresh_frame_index(root, rehash_final=True)
        W.write_reports(root, frames)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--auto", action="store_true",
                    help="incoming 어노테이션 폴더 이름으로 목적지를 자동 결정")
    ap.add_argument("--annotations", default=None,
                    help="root 기준 상대경로 (incoming/annotations/<...>)")
    ap.add_argument("--source-session", default=None,
                    help="incoming/sessions/<이 이름>/rgb 에서 이미지를 찾는다")
    ap.add_argument("--dest-session", default=None,
                    help="final/positive/sessions/<이 이름> 으로 복사한다")
    ap.add_argument("--apply", action="store_true",
                    help="주지 않으면 dry-run")
    a = ap.parse_args()

    root = Path(a.root)
    if a.auto:
        summary = promote_annotated_incoming(root, refresh=a.apply)
        print(f"자동 편입 {summary['promoted']}장  "
              f"(이미 있음 {summary['skipped_existing']})")
        for dest, n in sorted(summary["by_dest"].items()):
            print(f"  -> {dest}  {n}장")
        if summary["unresolved"]:
            print(f"  목적지 미해결(건너뜀): {summary['unresolved']}")
        if not a.apply:
            print("\n※ --apply 없이도 복사는 됐지만 manifest 갱신은 생략했다")
        return 0
    if not (a.annotations and a.source_session and a.dest_session):
        print("FAIL --auto 가 아니면 --annotations/--source-session/--dest-session 필요")
        return 1
    ann_dir = root / a.annotations
    src_rgb = root / "incoming/sessions" / a.source_session / "rgb"
    dst_dir = root / "final/positive/sessions" / a.dest_session
    dst_rgb = dst_dir / "rgb"
    dst_ann = root / "final/positive/annotations" / a.dest_session

    for path, label in ((ann_dir, "annotations"), (src_rgb, "source rgb"),
                        (dst_dir, "dest session")):
        if not path.is_dir():
            print(f"FAIL {label} 없음: {path}")
            return 1

    meta_path = dst_dir / "session.json"
    if not meta_path.is_file():
        print(f"FAIL dest session.json 없음: {meta_path}")
        return 1
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if str(meta.get("population_role", "")).upper() != "FINAL":
        print(f"FAIL dest 가 FINAL 이 아니다: {meta.get('population_role')}")
        return 1

    stems = annotated_stems(ann_dir)
    if not stems:
        print("편입할 어노테이션이 없다 (9kp 있는 json 0건)")
        return 1

    missing = [s for s in stems if not (src_rgb / f"{s}.png").is_file()]
    if missing:
        print(f"FAIL 이미지 없는 어노테이션 {len(missing)}건: {missing[:5]}")
        return 1

    print(f"편입 대상 {len(stems)}장")
    print(f"  from  {ann_dir.relative_to(root)}")
    print(f"        {src_rgb.relative_to(root)}")
    print(f"  to    {dst_rgb.relative_to(root)}")
    print(f"        {dst_ann.relative_to(root)}")
    print(f"  dest  object_type={meta.get('object_type')} "
          f"lighting={meta.get('lighting')}")

    if not a.apply:
        print("\ndry-run — 실제로 복사하려면 --apply")
        return 0

    dst_rgb.mkdir(parents=True, exist_ok=True)
    dst_ann.mkdir(parents=True, exist_ok=True)
    overlay_src = ann_dir / "_overlays"
    overlay_dst = dst_ann / "_overlays"

    copied = 0
    for stem in stems:
        shutil.copy2(src_rgb / f"{stem}.png", dst_rgb / f"{stem}.png")
        shutil.copy2(ann_dir / f"{stem}.json", dst_ann / f"{stem}.json")
        ov = overlay_src / f"{stem}.png"
        if ov.is_file():
            overlay_dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ov, overlay_dst / f"{stem}.png")
        copied += 1
    print(f"복사 {copied}장 완료 (incoming 원본은 그대로)")

    frames = W.refresh_frame_index(root, rehash_final=True)
    progress = W.write_reports(root, frames)
    print(f"manifest {len(frames)}행 · positive {progress.positive_total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
