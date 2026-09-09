"""§2 · §6 — R0 train/val manifest 와 프레임 속성을 **source label 에서 재생성**한다.

기존 audit(PROBE_METADATA_60K 등)을 복사하지 않는다.  다만 이미 검증된 그 파일과
교차대조해 두 경로가 일치하는지 확인한다(일치하지 않으면 멈춘다).
"""
from __future__ import annotations
import json, collections
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]
DS = REPO / "challenge/yolo_pose_one_model/datasets/g38_legacy_v1v2_p0_tex20k"
PROBE = REPO / "challenge/yolo_pose_one_model/spatial_concat_scratch/PROBE_METADATA_60K.jsonl"
P0_RAW = REPO / "challenge/yolo_pose_one_model/datasets/_raw_legacy_v1v2_p0_10k"
TEX_RAW = REPO / "challenge/yolo_pose_one_model/datasets/_raw_legacy_v1v2_p0_tex10k"
OUT = REPO / "data/pallet/results/low_angle_diversity_v1"


def elevation_deg(R, t):
    up = R[:, 1]
    return float(np.degrees(np.arcsin(np.clip(abs(float(up @ (t / np.linalg.norm(t)))), 0, 1))))


def label_path(row):
    if row["source"] == "G38":
        return REPO / row["renderer_label_locator"]
    raw = P0_RAW if row["source"] == "P0" else TEX_RAW
    return raw / row["label_member"]


def main():
    # merged_stem -> split 은 실제 dataset 폴더에서 읽는다 (manifest 를 믿지 않는다)
    split_of = {}
    for sp in ("train", "val"):
        for f in (DS / "labels" / sp).glob("*.txt"):
            split_of[f.stem] = sp
    print(f"dataset stems: train {sum(v=='train' for v in split_of.values())} "
          f"val {sum(v=='val' for v in split_of.values())}")

    rows = []
    for i, line in enumerate(PROBE.open(encoding="utf-8")):
        if i % 10000 == 0:
            print(f"  {i}", flush=True)
        r = json.loads(line)
        stem = r["merged_stem"]
        sp = split_of.get(stem)
        if sp is None:
            raise RuntimeError(f"{stem}: dataset 폴더에 없음")
        d = json.load(open(label_path(r)))
        cam = d["camera_data"]
        o = d["objects"][0]
        M = np.asarray(o["pose_transform"], float)
        R, t = M[:3, :3], M[:3, 3]
        pc = np.asarray(o["projected_cuboid"], float)[:8]
        diag = float(np.linalg.norm(pc.max(0) - pc.min(0)))
        dm = o["dimensions_m"]
        w, h, dp = dm["width"], dm["height"], dm["depth"]
        rows.append({
            "frame_id": stem, "split": sp, "stream": r["source"],
            "source_asset": o.get("source_asset") or r.get("pallet_type") or "unknown",
            "elevation_deg": elevation_deg(R, t),
            "distance_m": float(np.linalg.norm(t)),
            "diag_ratio": diag / float(np.hypot(cam["width"], cam["height"])),
            "w": w, "h": h, "d": dp, "aspect": max(w, dp) / min(w, dp),
        })

    # 교차대조: 이미 검증된 side table 의 치수와 일치하는가
    side = np.load(REPO / "challenge/yolo_pose_one_model/pallet_translation_loss_v1/GEOMETRY_SIDETABLE.npz",
                   allow_pickle=True)
    sidx = {s: i for i, s in enumerate(side["stems"].tolist())}
    # ★ 라벨의 dimensions_m 은 **camera-facing WHD** 이고 side table 은 **fixed renderer
    # XYZ** 다.  둘은 W/D 순열만큼 다르므로 순열 불변 비교(정렬된 삼중항)로 대조한다.
    diffs = []
    for r in rows[:5000]:
        xyz = np.sort(np.asarray(side["dims"][sidx[r["frame_id"]]], float))
        mine = np.sort(np.array([r["w"], r["h"], r["d"]], float))
        diffs.append(float(np.abs(xyz - mine).max()))
    md = float(np.max(diffs))
    print(f"교차대조 (독립 재생성 vs 기존 side table, 5,000 표본, 순열불변) "
          f"최대 치수 차: {md:.3e} m")
    if md > 1e-9:
        raise RuntimeError("두 경로가 어긋난다 — manifest 를 신뢰할 수 없다")

    tr = [r for r in rows if r["split"] == "train"]
    va = [r for r in rows if r["split"] == "val"]
    (OUT / "R0_TRAIN_MANIFEST.txt").write_text("\n".join(r["frame_id"] for r in tr) + "\n")
    (OUT / "R0_VAL_MANIFEST.txt").write_text("\n".join(r["frame_id"] for r in va) + "\n")
    keys = list(rows[0].keys())
    with (OUT / "R0_FRAME_TABLE.csv").open("w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")

    # §6 — 저앙각 편중 재계산
    def eff(counter, n):
        p = np.array(list(counter.values()), float) / n
        return float(np.exp(-(p * np.log(p)).sum()))

    low = [r for r in tr if r["elevation_deg"] < 8]
    ca, cl = collections.Counter(r["source_asset"] for r in tr), collections.Counter(r["source_asset"] for r in low)
    rep = {
        "schema_version": "low_angle_diversity_v1_r0_manifest_v1",
        "regenerated_from": "source renderer labels (not copied from prior audits)",
        "crosscheck_max_dimension_diff_m": md,
        "train_n": len(tr), "val_n": len(va),
        "train_all": {"assets": dict(ca), "effective_asset_count": eff(ca, len(tr))},
        "train_low_angle_lt8": {
            "n": len(low), "frac_of_train": len(low) / len(tr),
            "assets": dict(cl), "effective_asset_count": eff(cl, len(low)),
            "dominant_asset": cl.most_common(1)[0][0],
            "dominant_share": cl.most_common(1)[0][1] / len(low),
            "elevation_p5_p50_p95": [float(np.percentile([r["elevation_deg"] for r in low], q)) for q in (5, 50, 95)],
            "distance_p5_p50_p95": [float(np.percentile([r["distance_m"] for r in low], q)) for q in (5, 50, 95)],
            "diag_ratio_p5_p50_p95": [float(np.percentile([r["diag_ratio"] for r in low], q)) for q in (5, 50, 95)],
        },
        "train_low_angle_by_stream": dict(collections.Counter(r["stream"] for r in low)),
    }
    (OUT / "R0_MANIFEST_AUDIT.json").write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n")
    print(json.dumps(rep, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
