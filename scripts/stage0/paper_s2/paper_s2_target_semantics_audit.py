"""paper_s2_target_semantics_audit.py — PAPER_S2 ep57 target-semantics audit.

목적(진단 전용, 성능 개선 아님):
  H1 = 화면(=belief map) 안에 중심이 있으나 full Gaussian support 가 border 를
       넘는 keypoint 가 all-zero belief target + valid channel mask 를 동시에
       받는가?  (= border-positive 가 background-negative 로 supervise 되는가)

핵심 원칙:
  * 수식만 재구현하지 않는다.  ep57 학습과 **동일한 loader/CreateBeliefMap 경로**로
    실제 target tensor 를 생성하고, 그 tensor 의 sum/max 로 nonzero 를 판정한다.
    별도로 계산한 full-support 수식과의 불일치 건수를 sanity check 로 보고한다.
  * 좌표는 세 공간을 섞지 않고 각각 기록한다:
        original image / transformed 400x400 / belief 50x50
    loader 가 직접 내보낸 refine_keypoints(=belief grid) 를 기준으로 삼아
    좌표계 재유도 위험을 제거한다.
  * 기존 데이터/체크포인트는 읽기 전용.  아무 것도 덮어쓰지 않는다.

ep57 (weights/paper_s2_stageB/net_epoch_0057.pth) 학습 인자 재현:
  sigma=2.0  output_size=50  imagesize=400  truncation_aug_prob=0.0
  mask_aux=True  diffpnp=True(-> aspect_resize=True)  clip_belief_border=False
  (header.txt 참조)

refinement_targets 는 출력 필드만 추가할 뿐 beliefs 를 바꾸지 않으므로
(utils_dataset.__getitem__ 참조) 감사에서 켜도 target semantics 는 불변이다.

randomness 주의: albumentations 증강은 확률적이므로 frame 당 1 draw =
"epoch 당 기대 rate" 의 불편추정이다 (ep57 은 57 epoch = frame 당 57 draw).

Usage:
  python scripts/stage0/paper_s2/paper_s2_target_semantics_audit.py            # 전체
  python scripts/stage0/paper_s2/paper_s2_target_semantics_audit.py --limit 300  # 스모크
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "common"))

from utils_dataset import CleanVisiiDopeLoader  # noqa: E402
from utils_belief import CreateBeliefMap  # noqa: E402

# ---- ep57 training constants (weights/paper_s2_stageB/header.txt) -----------
SIGMA = 2.0
OUTPUT_SIZE = 50          # train.py:987
IMAGESIZE = 400           # transformed space before the belief grid
TRUNC_AUG_PROB = 0.0      # ep57 used the *pre-generated* aug_trunc_v2 only
MASK_AUX = True
ASPECT_RESIZE = True      # diffpnp=True => aspect_resize on
CLIP_BELIEF_BORDER = False

DATASETS = [
    "mixed_v8_train",
    "v4_split_base",
    "aug_squash_v2",
    "aug_trunc_v2",
    "aug_scale_v2",
    "paper_4pallet_mask_v1",
]
TRAIN_ROOT = os.path.join(ROOT, "data", "pallet", "training_data")
INDEX_DIR = os.path.join(ROOT, "data", "pallet", "results",
                         "paper_s2_scratch_diffpnp", "pnp_valid_3d_index")
OUT_DIR = os.path.join(ROOT, "data", "pallet", "results",
                       "paper_s2_target_semantics_audit")

NEAR_KP = {0, 1, 2, 3}
FAR_KP = {4, 5, 6, 7}
TOP_KP = {0, 1, 4, 5}
BOTTOM_KP = {2, 3, 6, 7}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_diffpnp_index() -> dict:
    """abs json path -> entry (pnp_valid_3d / V / V8 / aug_kind / dims)."""
    idx = {}
    for jf in glob.glob(os.path.join(INDEX_DIR, "*.json")):
        for rel, entry in json.load(open(jf)).items():
            idx[os.path.abspath(os.path.join(TRAIN_ROOT, rel))] = entry
    return idx


def full_support_inside(y: float, x: float, size: int, sigma: float) -> bool:
    """CreateBeliefMap 의 조건을 그대로 옮긴 것 (sanity check 전용).

    utils_belief.CreateBeliefMap 은 p = [point[1], point[0]] 로 (y,x) 를 만든 뒤
    p[0]-w>=0 and p[0]+w<size and p[1]-w>=0 and p[1]+w<size 를 본다.
    """
    w = int(sigma * 2)
    return (y - w >= 0 and y + w < size and x - w >= 0 and x + w < size)


def centre_inside(y: float, x: float, size: int) -> bool:
    return 0 <= y < size and 0 <= x < size


def audit_dataset(ds: str, dp_index: dict, limit: int | None, seed: int):
    """ep57 loader 를 그대로 돌려 frame 당 9 keypoint 의 target semantics 기록."""
    ds_path = os.path.join(TRAIN_ROOT, ds)
    loader = CleanVisiiDopeLoader(
        [ds_path],
        objects=["pallet"],
        sigma=SIGMA,
        output_size=OUTPUT_SIZE,
        truncation_aug_prob=TRUNC_AUG_PROB,
        mask_aux=MASK_AUX,
        clip_belief_border=CLIP_BELIEF_BORDER,
        refinement_targets=True,          # 출력 필드만 추가 (beliefs 불변)
        aspect_resize=ASPECT_RESIZE,
        diffpnp_index=dp_index,
    )
    n_total = len(loader)
    order = list(range(n_total))
    if limit is not None and limit < n_total:
        rng = random.Random(seed)
        order = rng.sample(order, limit)
    order.sort()

    rows = []
    mismatch = 0
    for n_done, i in enumerate(order):
        # frame 단위 결정적 시드 (albumentations 는 전역 random/np.random 사용)
        random.seed(seed * 1_000_003 + i)
        np.random.seed((seed * 1_000_003 + i) % (2**31 - 1))

        sample = loader[i]
        beliefs = sample["beliefs"].numpy()             # (9,50,50) 실제 target
        mask = sample["belief_channel_mask"].numpy()    # (9,)
        kp_belief = sample["refine_keypoints"].numpy()  # (9,2) belief grid (x,y)
        kp_valid = sample["refine_keypoints_valid"].numpy()
        vis_raw = sample.get("visibility", 0.0)
        vis_arr = np.atleast_1d(np.asarray(vis_raw, dtype=np.float64)).ravel()
        vis = float(vis_arr.mean()) if vis_arr.size else float("nan")
        fname = sample["file_name"]

        jp = os.path.join(ds_path, os.path.splitext(fname)[0] + ".json")
        entry = dp_index.get(os.path.abspath(jp), {})
        raw = {}
        if os.path.exists(jp):
            try:
                raw = json.load(open(jp))
            except Exception:
                raw = {}
        objs = raw.get("objects") or [{}]
        orig_kp = objs[0].get("projected_cuboid")
        orig_ctr = objs[0].get("projected_cuboid_centroid")
        if orig_kp is not None and orig_ctr is not None and len(orig_kp) == 8:
            orig9 = list(orig_kp) + [orig_ctr]
        elif orig_kp is not None and len(orig_kp) == 9:
            orig9 = list(orig_kp)
        else:
            orig9 = [[float("nan")] * 2] * 9
        cam = raw.get("camera_data", {})
        img_w = float(cam.get("width", 640))
        img_h = float(cam.get("height", 480))

        pnp_ok = bool(entry.get("pnp_valid_3d", False))
        V8 = bool(entry.get("V8", False))
        diffpnp_valid = pnp_ok and V8

        for k in range(9):
            bx, by = float(kp_belief[k][0]), float(kp_belief[k][1])
            ch = beliefs[k]
            tsum = float(ch.sum())
            tmax = float(ch.max())
            nonzero = tmax > 0.0

            ci = centre_inside(by, bx, OUTPUT_SIZE)
            fs = full_support_inside(by, bx, OUTPUT_SIZE, SIGMA)
            # 수식 예측 (clip off): full support 있어야만 nonzero
            predicted_nonzero = fs
            if predicted_nonzero != nonzero:
                mismatch += 1

            ox, oy = float(orig9[k][0]), float(orig9[k][1])
            sentinel = (ox <= -90.0 and oy <= -90.0)

            rows.append(dict(
                dataset=ds,
                frame_id=os.path.splitext(fname)[0],
                json_path=os.path.relpath(jp, ROOT),
                aug_kind=entry.get("aug_kind", "unknown"),
                keypoint_id=k,
                # --- original image space ---
                x_original=ox, y_original=oy,
                image_width=img_w, image_height=img_h,
                # --- belief (50x50) space, loader 가 직접 낸 좌표 ---
                x_belief=bx, y_belief=by,
                # --- transformed 400x400 space (belief * 400/50) ---
                x_transformed=bx * (IMAGESIZE / OUTPUT_SIZE),
                y_transformed=by * (IMAGESIZE / OUTPUT_SIZE),
                belief_size=OUTPUT_SIZE, sigma=SIGMA,
                distance_left=bx, distance_right=OUTPUT_SIZE - 1 - bx,
                distance_top=by, distance_bottom=OUTPUT_SIZE - 1 - by,
                dist_to_border=min(bx, by, OUTPUT_SIZE - 1 - bx,
                                   OUTPUT_SIZE - 1 - by),
                center_inside_belief=ci,
                full_gaussian_support_inside=fs,
                belief_target_sum=tsum,
                belief_target_max=tmax,
                belief_target_nonzero=nonzero,
                formula_actual_mismatch=(predicted_nonzero != nonzero),
                belief_channel_mask=float(mask[k]),
                refine_keypoint_valid=float(kp_valid[k]),
                is_exact_sentinel=sentinel,
                visibility=vis,
                pnp_valid_3d=pnp_ok, V=entry.get("V"), V8=V8,
                diffpnp_valid=diffpnp_valid,
                is_trunc_dataset=(ds == "aug_trunc_v2"),
                is_squash_dataset=(ds == "aug_squash_v2"),
                is_scale_dataset=(ds == "aug_scale_v2"),
                kp_group_near=(k in NEAR_KP), kp_group_far=(k in FAR_KP),
                kp_group_top=(k in TOP_KP), kp_group_bottom=(k in BOTTOM_KP),
            ))
        if (n_done + 1) % 500 == 0:
            print(f"  [{ds}] {n_done + 1}/{len(order)}", flush=True)
    return rows, mismatch, n_total, len(order)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="dataset 당 최대 frame (None=전체)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    dp_index = load_diffpnp_index()
    print(f"[index] {len(dp_index)} frames")

    all_rows, meta, total_mismatch = [], [], 0
    for ds in args.datasets:
        print(f"[audit] {ds} ...", flush=True)
        rows, mism, n_tot, n_used = audit_dataset(ds, dp_index, args.limit,
                                                  args.seed)
        all_rows.extend(rows)
        total_mismatch += mism
        meta.append(dict(dataset=ds, n_frames_total=n_tot,
                         n_frames_audited=n_used,
                         formula_actual_mismatch=mism))
        print(f"  -> {n_used}/{n_tot} frames, mismatch={mism}")

    df = pd.DataFrame(all_rows)
    pq = os.path.join(OUT_DIR, "target_semantics_keypoints.parquet")
    df.to_parquet(pq, index=False)
    pd.DataFrame(meta).to_csv(
        os.path.join(OUT_DIR, "audit_meta.csv"), index=False)
    print(f"\n[saved] {pq}  rows={len(df)}")
    print(f"[sanity] formula-vs-actual mismatch = {total_mismatch} "
          f"(0 이어야 정상)")


if __name__ == "__main__":
    main()
