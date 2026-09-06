"""빌드된 base 라벨이 두 필드 중 어느 것에서 나왔는지 프레임 단위로 판정한다.

`load_kps` 가 2026-09-06 `f2b2739` 에서 `keypoint_annotations` 우선으로 바뀌었다.
그 이전에 만들어진 데이터셋은 `projected_cuboid` 를 읽었고, 그 필드는
`live_capture_gt` 851장에서 규약을 198장(23.3%) 어긴다.

각 base 라벨의 9 keypoint 를 두 필드의 padded 좌표와 대조해 어느 쪽과 맞는지 센다.
읽기 전용.
"""
from __future__ import annotations

import json
import struct
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DATASETS = REPO / "challenge/yolo_pose_one_model/datasets"
LIVE = REPO / "challenge/data/01_real/live_capture_gt"
PAD = 100


def png_size(p):
    with open(p, "rb") as f:
        h = f.read(26)
    return struct.unpack(">II", h[16:24])


def read_label(p):
    f = p.read_text(encoding="utf-8").strip().split()
    return [(float(f[5 + 3 * i]), float(f[6 + 3 * i]), int(f[7 + 3 * i]))
            for i in range(9)]


def fields(jp):
    obj = json.load(open(jp, encoding="utf-8"))["objects"][0]
    ka = obj.get("keypoint_annotations")
    ka_pts = None
    if isinstance(ka, list) and len(ka) >= 9:
        ka_pts = [None if e.get("xy") is None else
                  (float(e["xy"][0]), float(e["xy"][1])) for e in ka[:9]]
    pc = obj.get("projected_cuboid")
    pc_pts = None
    if pc and len(pc) >= 8:
        cen = obj.get("projected_cuboid_centroid")
        pc_pts = [(float(p[0]), float(p[1])) for p in pc[:8]]
        pc_pts.append((float(cen[0]), float(cen[1])) if cen else None)
    return ka_pts, pc_pts


def err(label, pts, w, h):
    if pts is None:
        return None
    e = 0.0
    n = 0
    for i, (lx, ly, v) in enumerate(label):
        if v == 0 or pts[i] is None:
            continue
        e = max(e, abs(lx * w - (pts[i][0] + PAD)), abs(ly * h - (pts[i][1] + PAD)))
        n += 1
    return e if n else None


def main() -> int:
    index = {}
    for jp in LIVE.rglob("*.json"):
        if jp.stem.isdigit():
            index.setdefault((jp.parent.name, jp.stem), jp)

    out = []
    for ds in ("live_gt_v4", "live_gt_v5_nocrop", "live_gt_v6_clean", "live_gt_v7_nopad"):
        lbl = DATASETS / ds / "labels/train"
        img = DATASETS / ds / "images/train"
        if not lbl.exists():
            continue
        c = Counter()
        for lp in sorted(lbl.glob("*.txt")):
            s = lp.stem
            if "__" not in s or s.endswith(("_f", "_n")):
                continue
            sess, frame = s.split("__", 1)
            jp = index.get((sess, frame))
            if jp is None:
                c["no_source_json"] += 1
                continue
            w, h = png_size(img / f"{s}.png")
            label = read_label(lp)
            ka_pts, pc_pts = fields(jp)
            eka, epc = err(label, ka_pts, w, h), err(label, pc_pts, w, h)
            c["matched"] += 1
            ok_ka = eka is not None and eka <= 1.5
            ok_pc = epc is not None and epc <= 1.5
            if ok_ka and ok_pc:
                c["both_fields_agree"] += 1
            elif ok_ka:
                c["from_keypoint_annotations"] += 1
            elif ok_pc:
                c["from_projected_cuboid"] += 1
            else:
                c["neither"] += 1
        out.append({"dataset": ds, **dict(c)})

    dst = REPO / "data/pallet/results/next_accuracy_v2/LABEL_PROVENANCE.json"
    dst.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{'dataset':<22}{'base':>7}{'둘다일치':>10}{'kp_ann':>9}"
          f"{'proj_cub':>10}{'neither':>9}")
    for r in out:
        print(f"{r['dataset']:<22}{r.get('matched',0):>7}"
              f"{r.get('both_fields_agree',0):>10}{r.get('from_keypoint_annotations',0):>9}"
              f"{r.get('from_projected_cuboid',0):>10}{r.get('neither',0):>9}")
    print(f"\nwrote {dst.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
