"""A2b 가 loss 안에서 조회할 per-frame geometry 부수 테이블.

YOLO label(.txt)에는 K/치수/pose 가 없다.  merged_stem 을 키로 renderer label 에서
끌어와 하나의 npz 로 굳힌다.  학습 데이터는 건드리지 않는다 (읽기만).

저장 좌표계 주의: 이미지가 raw 에서 상하좌우 100px 로 padding 된 `prepared` 본이라
cx, cy 에 +100 을 더해 둔다 (`prepared_pad_px`).  letterbox 로 인한 추가 배율은
학습 중 batch GT 로 맞춘다 -- 여기서는 prepared 좌표계까지만 책임진다.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
MANIFEST = REPO / "challenge/yolo_pose_one_model/spatial_concat_scratch/PROBE_METADATA_60K.jsonl"
P0_RAW = REPO / "challenge/yolo_pose_one_model/datasets/_raw_legacy_v1v2_p0_10k"
TEX_RAW = REPO / "challenge/yolo_pose_one_model/datasets/_raw_legacy_v1v2_p0_tex10k"
OUT = REPO / "challenge/yolo_pose_one_model/pallet_translation_loss_v1/GEOMETRY_SIDETABLE.npz"


def label_path(row):
    if row["source"] == "G38":
        return REPO / row["renderer_label_locator"]
    raw = P0_RAW if row["source"] == "P0" else TEX_RAW
    return raw / row["label_member"]


def main() -> int:
    rows = [json.loads(l) for l in MANIFEST.open(encoding="utf-8")]
    stems, K, dims, R, t, pad = [], [], [], [], [], []
    Xcf, match_err = [], []
    for i, row in enumerate(rows):
        if i % 10000 == 0:
            print(f"  {i}/{len(rows)}", flush=True)
        payload = json.loads(label_path(row).read_text(encoding="utf-8"))
        intr = payload["camera_data"]["intrinsics"]
        obj = payload["objects"][0]
        M = np.asarray(obj["pose_transform"], np.float64)
        p = float(row["prepared_pad_px"])
        stems.append(row["merged_stem"])
        K.append([intr["fx"], intr["fy"], intr["cx"] + p, intr["cy"] + p])
        # 모델 입력으로 허용된 것은 fixed renderer XYZ 뿐이다 (camera-facing WHD 는 금지)
        dims.append(row["fixed_renderer_dimensions_m_xyz_model_input"])
        R.append(M[:3, :3])
        t.append(M[:3, 3])
        pad.append(p)

        # 3D 코너를 **camera-facing 0..7 순서로** 확정해 둔다.  perm_v4 를 믿고
        # 재구성하지 않고, 저장된 projected_cuboid 와 기하로 직접 대응시킨다 --
        # 잔차를 같이 저장하므로 틀리면 드러난다.
        w, hgt, dep = row["fixed_renderer_dimensions_m_xyz_model_input"]
        cand = np.array(list(itertools.product([-w / 2, w / 2], [-hgt / 2, hgt / 2],
                                               [-dep / 2, dep / 2])))
        Pc = cand @ M[:3, :3].T + M[:3, 3]
        u = np.stack([intr["fx"] * Pc[:, 0] / Pc[:, 2] + intr["cx"] + p,
                      intr["fy"] * Pc[:, 1] / Pc[:, 2] + intr["cy"] + p], axis=1)
        pc = np.asarray(obj["projected_cuboid"], np.float64)[:8] + p
        D = np.linalg.norm(pc[:, None, :] - u[None, :, :], axis=2)
        assign = D.argmin(1)
        if len(set(assign.tolist())) != 8:
            raise RuntimeError(f"{row['merged_stem']}: corner assignment is not a bijection")
        Xcf.append(cand[assign])
        match_err.append(float(D[np.arange(8), assign].max()))

    np.savez_compressed(
        OUT,
        stems=np.array(stems),
        K=np.asarray(K, np.float64),
        dims=np.asarray(dims, np.float64),
        R=np.asarray(R, np.float64),
        t=np.asarray(t, np.float64),
        pad=np.asarray(pad, np.float64),
        Xcf=np.asarray(Xcf, np.float64),
        match_err=np.asarray(match_err, np.float64),
    )
    me = np.asarray(match_err)
    print(f"corner assignment residual px: median {np.median(me):.5f} "
          f"p99 {np.percentile(me, 99):.5f} max {me.max():.5f}")
    if me.max() > 0.05:
        raise RuntimeError("corner assignment residual too large")
    print(f"wrote {OUT.relative_to(REPO)}  n={len(stems)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
