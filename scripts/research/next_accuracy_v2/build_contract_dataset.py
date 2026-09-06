"""정정 라벨 + 촬영단위 split 으로 real YOLO-pose 데이터셋을 만든다 (§7 · §11).

변환 규약은 새로 쓰지 않고 ``prepare_yolo_pose.one`` 을 그대로 import 한다 —
PAD / BORDER_REFLECT_101 / visibility 판정 / bbox 가 한 글자도 달라지면 안 된다.
이 파일이 하는 일은 **어떤 프레임이 train 이고 어떤 프레임이 held-out 인가**를
봉인 파일에서 읽어 오는 것뿐이다.

split 정본:
  data/pallet/results/accuracy_root_cause_v1/next_experiment/HOLDOUT_{TRAIN,HELD_OUT}_FRAMES.txt
  (폴더 = 물리 촬영 단위. interleave 금지 — 일반화를 재는 실험이다.)

기존 데이터셋을 덮어쓰지 않는다.  기본 출력은 새 이름이다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "challenge/yolo_pose_one_model/scripts"))
import prepare_yolo_pose as pyp  # noqa: E402
from prepare_yolo_pose import PAD, one  # noqa: E402


def _legacy_load_kps(ann_path):
    """`f2b2739` 이전의 로더 — `projected_cuboid` 만 읽는다.

    대조군용이다.  이 필드는 live_capture_gt 851장에서 camera-facing 0123 규약을
    198장(23.3%) 어긴다.  상태 분리(3-튜플)는 유지해 다른 축이 섞이지 않게 한다.
    """
    import json as _json
    try:
        obj = _json.load(open(ann_path, encoding="utf-8"))["objects"][0]
    except Exception:
        return None
    proj = obj.get("projected_cuboid")
    if not proj or len(proj) < 8:
        return None
    cen = obj.get("projected_cuboid_centroid")
    kps = [(float(p[0]), float(p[1]), True) for p in proj[:8]]
    kps.append((float(cen[0]), float(cen[1]), True) if cen else (0.0, 0.0, False))
    return kps


def _use_legacy_field():
    pyp.load_kps = _legacy_load_kps

GT_ROOT = REPO / "challenge/data/01_real/live_capture_gt"
CAPTURE_ROOT = REPO / "challenge/data/01_real/_live_captures"
SEAL_DIR = REPO / "data/pallet/results/accuracy_root_cause_v1/next_experiment"
OUT_ROOT = REPO / "challenge/yolo_pose_one_model"


def rgb_dir(session: str):
    hits = list(CAPTURE_ROOT.glob(f"*/sessions/{session}/rgb"))
    return hits[0] if len(hits) == 1 else None


def jobs_for(ids, out_root: Path, split: str, subset=None):
    jobs, missing = [], []
    for fid in ids:
        folder, stem = fid.split("/", 1)
        session = folder[: -len("_manual_gt")]
        if subset is not None and fid not in subset:
            continue
        ann = GT_ROOT / folder / f"{stem}.json"
        rd = rgb_dir(session)
        img = None if rd is None else rd / f"{stem}.png"
        if not ann.is_file() or img is None or not img.is_file():
            missing.append(fid)
            continue
        out_stem = f"{folder}__{stem}"
        jobs.append((out_stem, os.path.relpath(img, REPO), os.path.relpath(ann, REPO),
                     str(out_root / "images" / split / f"{out_stem}.png"),
                     str(out_root / "labels" / split / f"{out_stem}.txt")))
    return jobs, missing


def audit(out_root: Path, split: str, jobs):
    """§7 이 요구하는 감사 표."""
    a = Counter()
    seen = set()
    for out_stem, _, ann_rel, _, _ in jobs:
        lp = out_root / "labels" / split / f"{out_stem}.txt"
        if not lp.is_file():
            a["label_missing"] += 1
            continue
        if out_stem in seen:
            a["duplicate_stem"] += 1
        seen.add(out_stem)
        a["frames"] += 1
        f = lp.read_text(encoding="utf-8").strip().split()
        for i in range(9):
            v = int(f[7 + 3 * i])
            a[f"v{v}"] += 1
        obj = json.load(open(REPO / ann_rel, encoding="utf-8"))["objects"][0]
        ann = obj.get("keypoint_annotations") or []
        for i, e in enumerate(ann[:9]):
            unknown = e.get("xy") is None or int(e.get("visibility", 0)) == 0
            if e.get("xy") is None:
                a["src_xy_none"] += 1
            if unknown and int(f[7 + 3 * i]) == 2:
                a["unknown_became_v2"] += 1
        pts = [e.get("xy") for e in ann[:4]]
        if all(p is not None for p in pts):
            if not (pts[0][0] < pts[1][0] and pts[3][0] < pts[2][0]):
                a["index_convention_violation"] += 1
    return a


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="datasets/live_gt_contract_v2")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--label-field", choices=["contract", "legacy"], default="contract",
                    help="contract = keypoint_annotations 우선(정본). "
                         "legacy = projected_cuboid 만(f2b2739 이전 동작, 대조군)")
    ap.add_argument("--train-subset", default=None,
                    help="train 을 더 좁힐 프레임 id 목록 파일 (arm 실험용)")
    args = ap.parse_args()

    if args.label_field == "legacy":
        _use_legacy_field()
        print("★ legacy 라벨 필드(projected_cuboid) 로 빌드한다 — 대조군")

    out_root = OUT_ROOT / args.out
    for s in ("train", "val"):
        (out_root / "images" / s).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / s).mkdir(parents=True, exist_ok=True)

    train_ids = [l.strip() for l in
                 open(SEAL_DIR / "HOLDOUT_TRAIN_FRAMES.txt", encoding="utf-8") if l.strip()]
    held_ids = [l.strip() for l in
                open(SEAL_DIR / "HOLDOUT_HELD_OUT_FRAMES.txt", encoding="utf-8") if l.strip()]
    overlap = set(train_ids) & set(held_ids)

    subset = None
    if args.train_subset:
        subset = {l.strip() for l in open(args.train_subset, encoding="utf-8") if l.strip()}

    tj, tmiss = jobs_for(train_ids, out_root, "train", subset)
    hj, hmiss = jobs_for(held_ids, out_root, "val")
    print(f"train {len(tj)} (누락 {len(tmiss)})   held-out {len(hj)} (누락 {len(hmiss)})")

    counts = {}
    for split, jobs in (("train", tj), ("val", hj)):
        c = Counter()
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for r in ex.map(one, jobs, chunksize=16):
                c[r] += 1
        counts[split] = dict(c)
        print(f"  {split}: {dict(c)}")

    (out_root / "data.yaml").write_text(
        f"path: {out_root}\ntrain: images/train\nval: images/val\n"
        "kpt_shape: [9, 3]\nflip_idx: [1, 0, 3, 2, 5, 4, 7, 6, 8]\nnames:\n  0: pallet\n",
        encoding="utf-8")

    audits = {s: dict(audit(out_root, s, j))
              for s, j in (("train", tj), ("val", hj))}
    rep = {
        "split_source": str(SEAL_DIR.relative_to(REPO)),
        "split_unit": "folder (물리 촬영 단위)", "split_mode": "session_level",
        "pad": PAD, "border": "BORDER_REFLECT_101",
        "keypoint_field": ("keypoint_annotations (visibility 0 / xy None 은 미감독)"
                           if args.label_field == "contract"
                           else "projected_cuboid (legacy, f2b2739 이전)"),
        "label_field": args.label_field,
        "train_subset_file": args.train_subset,
        "counts": counts,
        "train_missing": tmiss, "held_out_missing": hmiss,
        "train_heldout_frame_overlap": sorted(overlap),

        "audit": audits,
        # §7 필수 항목.  이 빌드는 원본 프레임만 쓰므로 부모-파생 누수가 발생할 수
        # 없다.  "해당 없음" 을 명시적으로 남긴다 — 빈칸은 '안 쟀다' 와 '없다' 를
        # 구분하지 못한다.
        "derived_parent_overlap": {
            "n": 0,
            "note": ("N/A — 원본 프레임만 쓴다"
                     "(--crop-dir/--aug-dir 미사용).")},
        "total_supervised_keypoints": {s: audits[s].get("v2", 0)
                                       for s in ("train", "val")},
    }
    dst = out_root / "_contract_build.json"
    dst.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== §7 감사 ===")
    for s in ("train", "val"):
        a = rep["audit"][s]
        print(f"  {s:<6} frames {a.get('frames',0):>5}  v2 {a.get('v2',0):>6}  "
              f"v0 {a.get('v0',0):>5}  원본 xy=None {a.get('src_xy_none',0):>3}  "
              f"unknown->v2 {a.get('unknown_became_v2',0):>3}  "
              f"규약위반 {a.get('index_convention_violation',0):>3}  "
              f"중복stem {a.get('duplicate_stem',0):>3}")
    print(f"  train/held-out 프레임 중복: {len(overlap)}")
    # 규약 위반 카운터는 원본 keypoint_annotations 를 보므로 legacy 빌드에서도 0 이다.
    # legacy 의 위반은 별도로 REAL_LABEL_AUDIT 이 잰 198/851 이다.
    ok = all(rep["audit"][s].get("unknown_became_v2", 0) == 0
             and rep["audit"][s].get("index_convention_violation", 0) == 0
             and rep["audit"][s].get("duplicate_stem", 0) == 0
             for s in ("train", "val")) and not overlap
    print(f"\n  필수 조건: {'PASS' if ok else 'FAIL — 학습 STOP'}")
    print(f"wrote {dst.relative_to(REPO)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
