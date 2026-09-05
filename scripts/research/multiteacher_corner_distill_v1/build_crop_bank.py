"""합성 crop 은행을 **한 번만** 만든다 — Gate C 학습의 IO 병목 제거.

원래 설계는 step 마다 840x680 PNG 를 6장 디코딩해서 64x64 crop 8개를 뽑았다.
55,980 장을 섞어 콜드로 읽으니 3.1 초/step 이 나왔고 5,000 step 에 4시간이 넘었다.
crop 만 미리 뽑아 두면 학습은 순수 GPU 작업이 된다.

계약은 그대로다 — jitter 는 SOURCE_DEV 에서 측정한 R0 coarse residual 벡터를
복원추출하고(METHOD_LOCK gate_c.crop_jitter), crop 크기·가시성 정의도 같다.

    출력  audit/CROP_BANK_train.npz   (uint8 crop + 라벨)
          audit/CROP_BANK_val.npz
"""
from __future__ import annotations

import os

# ★ numpy/OpenCV 를 import 하기 **전에** 스레드를 1 로 묶는다.
#   부모가 OpenMP 런타임을 띄운 뒤 fork 하면 자식이 100% CPU 로 무한 스핀한다
#   (2026-09-05 실측: 워커 8개가 99.8% 로 돌면서 부모에 결과를 하나도 못 보냄).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import multiprocessing as mp
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mtcd_common as M
from mtcd_common import CROP_PX as CROP, extract_crop   # torch 를 부르지 않는 경로

SYN = M.REPO_ROOT / "challenge/yolo_pose_one_model/datasets/g38_legacy_v1v2_p0_tex20k"
cv2.setNumThreads(1)          # 워커마다 1스레드 — 프로세스 병렬로 쓴다


def _one(task):
    stem, split, seed = task
    rng = np.random.default_rng(seed)
    residuals = _one.residuals
    img = cv2.imread(str(SYN / f"images/{split}" / f"{stem}.png"))
    if img is None:
        return []
    h, w = img.shape[:2]
    line = (SYN / f"labels/{split}" / f"{stem}.txt").read_text().split("\n")[0].split()
    if len(line) < 32:
        return []
    v = list(map(float, line[5:]))
    xy = np.array([[v[3 * i] * w, v[3 * i + 1] * h] for i in range(9)])
    vis = np.array([v[3 * i + 2] for i in range(9)])
    out = []
    for k in range(8):
        if vis[k] <= 0:
            continue
        pool = residuals[k]
        centre = xy[k] + pool[rng.integers(len(pool))]
        patch, origin = extract_crop(img, centre)
        local = (xy[k] - origin).astype(np.float32)
        inside = bool(0 <= local[0] < CROP and 0 <= local[1] < CROP)
        dirs = []
        for a, b in M.INCIDENT_EDGES[k]:
            other = b if a == k else a
            d = xy[other] - xy[k]
            n = float(np.linalg.norm(d))
            if n > 1e-6:
                dirs.append((n, d / n))
        dirs.sort(key=lambda t: -t[0])
        edge = np.zeros((2, 2), np.float32)
        for i in range(min(2, len(dirs))):
            edge[i] = dirs[i][1]
        out.append((patch, k, local, inside, edge, float(len(dirs) >= 2)))
    return out


def _init(residuals):
    _one.residuals = residuals


def build(split, n_images, seed, workers):
    a = np.load(M.AUDIT / "R0_SOURCE_COARSE_RESIDUALS.npy")
    residuals = {k: a[a[:, 0] == k][:, 1:] for k in range(8)}
    stems = sorted(p.stem for p in (SYN / f"images/{split}").glob("*.png"))
    random.Random(seed).shuffle(stems)
    stems = stems[:n_images]
    tasks = [(s, split, seed + i) for i, s in enumerate(stems)]

    patches, kps, locals_, insides, edges, has_edges = [], [], [], [], [], []
    started = time.time()
    # spawn — fork 는 부모의 OpenMP/BLAS 상태를 물려받아 자식이 스핀한다.
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                             initializer=_init, initargs=(residuals,)) as ex:
        for i, res in enumerate(ex.map(_one, tasks, chunksize=16)):
            for patch, k, local, inside, edge, has_edge in res:
                patches.append(patch)
                kps.append(k)
                locals_.append(local)
                insides.append(inside)
                edges.append(edge)
                has_edges.append(has_edge)
            if (i + 1) % 2000 == 0:
                rate = (i + 1) / (time.time() - started)
                print(f"  {split} {i+1}/{len(tasks)} images  {len(patches)} crops  "
                      f"{rate:.0f} img/s  eta {(len(tasks)-i-1)/rate/60:.1f} min", flush=True)

    out = M.AUDIT / f"CROP_BANK_{split}.npz"
    np.savez(out,
             patches=np.stack(patches).astype(np.uint8),
             kp=np.asarray(kps, np.int64),
             local=np.stack(locals_).astype(np.float32),
             inside=np.asarray(insides, bool),
             edge=np.stack(edges).astype(np.float32),
             has_edge=np.asarray(has_edges, np.float32))
    size_mb = out.stat().st_size / 1e6
    print(f"{split}: {len(patches)} crops from {len(stems)} images  "
          f"{size_mb:.0f} MB  {(time.time()-started)/60:.1f} min  -> {out.name}", flush=True)
    return {"split": split, "n_images": len(stems), "n_crops": len(patches),
            "size_mb": round(size_mb, 1), "seed": seed,
            "minutes": round((time.time() - started) / 60.0, 2)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-images", type=int, default=14000)
    parser.add_argument("--val-images", type=int, default=1200)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    info = {"jitter_source": "audit/R0_SOURCE_COARSE_RESIDUALS.npy",
            "jitter_sha256": M.sha256_file(M.AUDIT / "R0_SOURCE_COARSE_RESIDUALS.npy"),
            "crop_px": CROP, "workers": args.workers,
            "why": "step 마다 PNG 를 디코딩하던 것을 한 번의 사전추출로 바꿨다. "
                   "계약(jitter 분포·crop 크기·가시성 정의)은 그대로다.",
            "splits": []}
    info["splits"].append(build("train", args.train_images, 20260905, args.workers))
    info["splits"].append(build("val", args.val_images, 20260907, args.workers))
    p = M.AUDIT / "CROP_BANK.json"
    p.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")
    print(f"-> {p.relative_to(M.REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
