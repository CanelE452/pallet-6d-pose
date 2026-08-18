"""Discover G/T/R roots, build manifests/all_samples.csv, and write reports/01_data_inventory.md.

Every path is resolved through challenge/data_paths.py or an explicit constant below;
nothing is guessed. All paths are stored repo-relative.

Domains
  generic_synth  paper_release v2_prod40k_clean_merged   (many pallet assets)
  target_synth   02_synthetic/training/v1 and v2         (palletobj = task pallet)
  real           01_real/manual_gt/* and eval_canonical/*

Usage:
  python challenge/yolo_pose_one_model/scripts/discover_and_audit.py
  python challenge/yolo_pose_one_model/scripts/discover_and_audit.py --probe 400
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import hashlib
import json
import os
import random
import struct
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from challenge.data_paths import get as dp  # noqa: E402

OUT_DIR = REPO / "challenge/yolo_pose_one_model"
G_ROOT = "data/pallet/training_data/paper_release/v2_prod40k_clean_merged"
R_GLOBS = ["challenge/data/01_real/manual_gt/*", "challenge/data/01_real/eval_canonical/*"]

# Task pallet identified by GT height; other heights are different physical pallets.
TASK_PALLET_HEIGHT = 0.11


def rel(p) -> str:
    return os.path.relpath(str(p), str(REPO)).replace(os.sep, "/")


def png_size(path):
    """Read width/height from the PNG/JPEG header without decoding."""
    try:
        with open(path, "rb") as f:
            head = f.read(32)
            if head[:8] == b"\x89PNG\r\n\x1a\n":
                w, h = struct.unpack(">II", head[16:24])
                return int(w), int(h)
            if head[:2] == b"\xff\xd8":
                f.seek(2)
                while True:
                    b = f.read(1)
                    while b and b != b"\xff":
                        b = f.read(1)
                    m = f.read(1)
                    if not m:
                        return None
                    if m[0] in (0xC0, 0xC1, 0xC2, 0xC3):
                        f.read(3)
                        h, w = struct.unpack(">HH", f.read(4))
                        return int(w), int(h)
                    ln = struct.unpack(">H", f.read(2))[0]
                    f.seek(ln - 2, 1)
    except Exception:
        return None
    return None


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- collectors

def collect_generic():
    root = REPO / G_ROOT
    rows = []
    for lab in sorted((root / "labels").glob("*.json")):
        stem = lab.name.replace("_label.json", "")
        img = root / "rgb" / f"{stem}_rgb.png"
        rows.append({"sample_id": f"G/{stem}", "domain": "generic_synth",
                     "image_path": rel(img), "annotation_path": rel(lab),
                     "session_id": "", "scene_id": "", "asset_id": "",
                     "frame_index": stem.lstrip("f")})
    return rows


def collect_target():
    rows = []
    for key, tag in (("synth.v1", "v1"), ("synth.v2", "v2")):
        root = Path(dp(key, absolute=True))
        for js in sorted(root.glob("part_*/*/*.json")):
            if js.name.endswith(".orig"):
                continue
            img = js.with_suffix(".png")
            part = js.parent.parent.name          # part_000
            rows.append({"sample_id": f"T/{tag}/{part}/{js.stem}", "domain": "target_synth",
                         "image_path": rel(img), "annotation_path": rel(js),
                         "session_id": f"{tag}_{part}", "scene_id": f"{tag}_{part}",
                         "asset_id": f"palletobj_{tag}", "frame_index": js.stem})
    return rows


def collect_real():
    rows = []
    for g in R_GLOBS:
        for folder in sorted((REPO).glob(g)):
            if not folder.is_dir():
                continue
            for js in sorted(folder.glob("*.json")):
                img = js.with_suffix(".png")
                if not img.exists():
                    alt = js.with_suffix(".jpg")
                    img = alt if alt.exists() else img
                rows.append({"sample_id": f"R/{folder.name}/{js.stem}", "domain": "real",
                             "image_path": rel(img), "annotation_path": rel(js),
                             "session_id": folder.name, "scene_id": folder.name,
                             "asset_id": "", "frame_index": js.stem})
    return rows


# ---------------------------------------------------------------- enrichment

def enrich(row, do_hash):
    ann = REPO / row["annotation_path"]
    img = REPO / row["image_path"]
    out = dict(row)
    out.update({"width": "", "height": "", "is_padded": "false", "has_pallet": "0",
                "annotation_source": "", "annotation_quality": "", "num_manual_keypoints": "",
                "num_auto_filled_keypoints": "", "num_visible_keypoints": "",
                "pallet_width_m": "", "pallet_depth_m": "", "pallet_height_m": "",
                "camera_intrinsic_source": "", "sha256": "", "split": "",
                "image_exists": str(img.exists()).lower()})
    try:
        data = json.load(open(ann, encoding="utf-8"))
    except Exception:
        out["annotation_quality"] = "unreadable"
        return out
    cd = data.get("camera_data", {}) or {}
    objs = data.get("objects") or []
    out["width"] = cd.get("width", "")
    out["height"] = cd.get("height", "")
    it = cd.get("intrinsics") or {}
    if it:
        out["camera_intrinsic_source"] = f"json:fx{round(float(it.get('fx', 0)), 1)}"
    if not objs:
        out["annotation_quality"] = "negative"
        return out
    o = objs[0]
    out["has_pallet"] = "1"
    out["split"] = o.get("split", "")

    dm = o.get("dimensions_m")
    if isinstance(dm, dict):
        out["pallet_width_m"], out["pallet_height_m"], out["pallet_depth_m"] = (
            dm.get("width"), dm.get("height"), dm.get("depth"))
    elif o.get("cuboid_dimensions_m"):        # v1/v2 store [depth, width, height]
        d_, w_, h_ = o["cuboid_dimensions_m"]
        out["pallet_width_m"], out["pallet_height_m"], out["pallet_depth_m"] = w_, h_, d_

    if row["domain"] == "real":
        mk = o.get("manual_kps")
        if mk is None:
            out["annotation_source"] = o.get("gt_source", "")
            out["annotation_quality"] = "legacy_mixed"
        else:
            n_man = sum(1 for p in mk if p is not None)
            out["num_manual_keypoints"] = n_man
            out["num_auto_filled_keypoints"] = len(mk) - n_man
            out["num_visible_keypoints"] = n_man
            out["annotation_source"] = o.get("gt_source", "manual")
            # A clicked point cannot be distinguished from a 'x'-key parallelogram
            # extrapolation after the fact -> manual_direct is not provable per point.
            out["annotation_quality"] = "manual_direct" if n_man == 9 else "manual_inferred"
    else:
        proj = o.get("projected_cuboid") or []
        out["num_visible_keypoints"] = len(proj) + (1 if o.get("projected_cuboid_centroid") else 0)
        out["annotation_source"] = "renderer"
        out["annotation_quality"] = "exact_synthetic"

    if do_hash and img.exists():
        out["sha256"] = sha256(img)
    return out


COLUMNS = ["sample_id", "domain", "image_path", "annotation_path", "session_id", "scene_id",
           "asset_id", "frame_index", "width", "height", "is_padded", "has_pallet",
           "annotation_source", "annotation_quality", "num_manual_keypoints",
           "num_auto_filled_keypoints", "num_visible_keypoints", "pallet_width_m",
           "pallet_depth_m", "pallet_height_m", "camera_intrinsic_source", "sha256",
           "split", "image_exists"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=int, default=400, help="frames per domain for resolution probe")
    ap.add_argument("--hash-real", action="store_true", default=True,
                    help="sha256 every real image (cheap, few hundred files)")
    args = ap.parse_args()

    print("collecting ...")
    rows = collect_generic() + collect_target() + collect_real()
    print(f"  {len(rows)} samples")

    print("enriching ...")
    out = []
    for i, r in enumerate(rows):
        out.append(enrich(r, do_hash=(args.hash_real and r["domain"] == "real")))
        if (i + 1) % 10000 == 0:
            print(f"  {i + 1}/{len(rows)}")

    (OUT_DIR / "manifests").mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "manifests/all_samples.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in out:
            w.writerow({k: r.get(k, "") for k in COLUMNS})
    print("wrote", rel(csv_path))

    # ---- resolution probe (header read only)
    res = collections.defaultdict(collections.Counter)
    for dom in ("generic_synth", "target_synth", "real"):
        pool = [r for r in out if r["domain"] == dom and r["image_exists"] == "true"]
        for r in random.Random(42).sample(pool, min(args.probe, len(pool))):
            res[dom][png_size(REPO / r["image_path"]) or "unreadable"] += 1

    # ---- summary
    by_dom = collections.Counter(r["domain"] for r in out)
    missing_img = collections.Counter(r["domain"] for r in out if r["image_exists"] != "true")
    qual = collections.defaultdict(collections.Counter)
    for r in out:
        qual[r["domain"]][r["annotation_quality"]] += 1

    lines = []
    A = lines.append
    A("# 01 — Data inventory\n")
    A("생성: `python challenge/yolo_pose_one_model/scripts/discover_and_audit.py`  ")
    A(f"registry: `{rel(csv_path)}` ({len(out)} rows)\n")
    A("모든 경로는 repo-relative. G 는 상수, T 는 `challenge/data_paths.py` 의 `synth.v1/v2`,")
    A("R 은 `01_real/manual_gt/*` + `01_real/eval_canonical/*` 글롭으로 찾았다.\n")

    A("## 도메인별 개수\n```")
    A(f"{'domain':<16}{'samples':>9}{'image 없음':>11}")
    for d in ("generic_synth", "target_synth", "real"):
        A(f"{d:<16}{by_dom[d]:>9}{missing_img[d]:>11}")
    A("```\n")

    A("## 해상도 (헤더 판독, 도메인당 표본 %d)\n```" % args.probe)
    for d in ("generic_synth", "target_synth", "real"):
        A(f"{d}: " + ", ".join(f"{k}×{v}" if not isinstance(k, str) else str(k)
                               for k, v in [(f"{a[0]}x{a[1]}" if isinstance(a, tuple) else a, b)
                                            for a, b in res[d].most_common()]))
    A("```\n")

    A("## annotation_quality\n```")
    for d in ("generic_synth", "target_synth", "real"):
        A(f"{d}: {dict(qual[d])}")
    A("```\n")

    # real detail
    A("## real 세션 상세\n```")
    A(f"{'session_id':<40}{'n':>5}{'img':>5}{'kp9':>5}{'dims(w,h,d)':>26}  split")
    sess = collections.defaultdict(list)
    for r in out:
        if r["domain"] == "real":
            sess[r["session_id"]].append(r)
    for s in sorted(sess):
        rs = sess[s]
        n_img = sum(1 for r in rs if r["image_exists"] == "true")
        n9 = sum(1 for r in rs if str(r["num_manual_keypoints"]) == "9")
        dims = collections.Counter((r["pallet_width_m"], r["pallet_height_m"], r["pallet_depth_m"])
                                   for r in rs)
        sp = collections.Counter(r["split"] or "-" for r in rs)
        A(f"{s:<40}{len(rs):>5}{n_img:>5}{n9:>5}{str(dims.most_common(1)[0][0]):>26}  {dict(sp)}")
    A("```\n")
    A(f"과제 팔레트 판별 기준: GT height == {TASK_PALLET_HEIGHT} m "
      "(0.15 = pallet11 정사각, 0.14 = wood 소형은 다른 물체).\n")

    rep = OUT_DIR / "reports/01_data_inventory.md"
    rep.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", rel(rep))


if __name__ == "__main__":
    main()
