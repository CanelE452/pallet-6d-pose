#!/usr/bin/env python3
"""포크리프트 실주행 녹화(rec)를 annotate.py 가 읽는 시퀀스로 푼다.

녹화 한 세션은 여섯 파일이다.  이 중 오버레이가 없는 ``_raw.mp4`` 와 intrinsics 가
들어 있는 ``_meta.json`` 만 쓴다::

    forklift_v4_recording_<날짜>_<시각>_raw.mp4     원본 640x480
    forklift_v4_recording_<날짜>_<시각>_meta.json   RealSense intrinsics 등

출력은 annotate.py 의 세션 규약을 따른다::

    <out>/<prefix><시각>/rgb/000000.png ...
    <out>/<prefix><시각>/cam_K.txt

``cam_K.txt`` 를 빼먹으면 안 된다.  ``_resolve_session_intrinsics`` 가 이 파일이
없을 때 **경고 없이** legacy default K 로 떨어져, 잘못된 intrinsics 로 PnP 를 푼
GT 가 조용히 쌓인다.

녹화가 비정상 종료된 mp4 는 ``moov`` atom 이 없어 프레임 수가 0 으로 읽힌다.
그런 세션은 건너뛰고 마지막에 이름을 모아 보고한다.

사용 예::

    python scripts/data_prep/extract_forklift_recording.py \\
        --rec-dir challenge/data/01_real/_live_captures/forklift_v4_20260901 \\
        --out-dir challenge/data/01_real/_live_captures/forklift_v4_20260901/sessions
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import cv2
import numpy as np


def _session_stems(rec_dir):
    """``_raw.mp4`` 를 가진 세션 stem 을 시각 순으로 돌려준다."""
    stems = []
    for raw in sorted(glob.glob(os.path.join(rec_dir, "*_raw.mp4"))):
        stem = raw[: -len("_raw.mp4")]
        if not os.path.isfile(stem + "_meta.json"):
            print(f"[SKIP] meta.json 없음: {os.path.basename(stem)}")
            continue
        stems.append(stem)
    return stems


def _write_cam_k(meta_path, dst_dir):
    """meta.json 의 intrinsics 를 3x3 으로 적고 (fx, fy, cx, cy) 를 돌려준다."""
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    intr = meta["intrinsics"]
    k = np.array([
        [intr["fx"], 0.0, intr["ppx"]],
        [0.0, intr["fy"], intr["ppy"]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    np.savetxt(os.path.join(dst_dir, "cam_K.txt"), k, fmt="%.10f")
    return intr["fx"], intr["fy"], intr["ppx"], intr["ppy"]


def _extract_one(stem, out_dir, prefix, overwrite):
    """세션 하나를 풀고 (세션명, 프레임수) 를 돌려준다.  손상이면 프레임수 0."""
    raw = stem + "_raw.mp4"
    name = prefix + os.path.basename(stem).split("_")[-1]
    dst = os.path.join(out_dir, name)
    rgb = os.path.join(dst, "rgb")

    cap = cv2.VideoCapture(raw)
    declared = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if declared <= 0:
        cap.release()
        print(f"[CORRUPT] {name}: moov atom 이 없다 (녹화 비정상 종료) — 건너뛴다")
        return name, 0

    if os.path.isdir(rgb) and glob.glob(os.path.join(rgb, "*.png")) and not overwrite:
        cap.release()
        existing = len(glob.glob(os.path.join(rgb, "*.png")))
        print(f"[KEEP] {name}: 이미 {existing} 장 있다 (--overwrite 로 다시 뽑는다)")
        return name, existing

    os.makedirs(rgb, exist_ok=True)
    fx, fy, cx, cy = _write_cam_k(stem + "_meta.json", dst)

    written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(os.path.join(rgb, f"{written:06d}.png"), frame)
        written += 1
    cap.release()

    print(f"[OK]   {name}: {written} 장  K = fx={fx:.1f} fy={fy:.1f} "
          f"cx={cx:.1f} cy={cy:.1f}")
    return name, written


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="rec 녹화를 annotate.py 세션(rgb/ + cam_K.txt)으로 푼다")
    ap.add_argument("--rec-dir", required=True,
                    help="*_raw.mp4 와 *_meta.json 이 있는 녹화 폴더")
    ap.add_argument("--out-dir", default=None,
                    help="세션 폴더를 만들 위치 (기본: <rec-dir>/sessions)")
    ap.add_argument("--prefix", default="forklift_v4_",
                    help="세션 폴더 이름 접두어 (기본: forklift_v4_)")
    ap.add_argument("--overwrite", action="store_true",
                    help="이미 프레임이 있어도 다시 뽑는다")
    args = ap.parse_args(argv)

    rec_dir = os.path.abspath(args.rec_dir)
    out_dir = os.path.abspath(args.out_dir or os.path.join(rec_dir, "sessions"))
    if not os.path.isdir(rec_dir):
        ap.error(f"녹화 폴더가 없다: {rec_dir}")

    stems = _session_stems(rec_dir)
    if not stems:
        ap.error(f"*_raw.mp4 가 없다: {rec_dir}")

    os.makedirs(out_dir, exist_ok=True)
    print(f"녹화 {len(stems)} 세션 -> {out_dir}\n")

    results = [_extract_one(s, out_dir, args.prefix, args.overwrite) for s in stems]
    good = [(n, c) for n, c in results if c > 0]
    corrupt = [n for n, c in results if c == 0]

    print(f"\n총 {sum(c for _, c in good)} 장 / {len(good)} 세션")
    if corrupt:
        print(f"손상되어 건너뛴 세션 {len(corrupt)} 개: {', '.join(corrupt)}")
    print("\nannotate.py 예시:")
    print(f"  python scripts/annotate/annotate.py \\\n"
          f"    --seq {os.path.join(out_dir, good[0][0]) if good else '<세션>'} \\\n"
          f"    --pool {out_dir} \\\n"
          f"    --out-root challenge/data/01_real/live_capture_gt \\\n"
          f"    --population-role DEV \\\n"
          f"    --geometry-registry challenge/config/CHALLENGE_OBJECT_GEOMETRY_REGISTRY.json \\\n"
          f"    --object-type plastic_standard_110x110x15 \\\n"
          f"    --intrinsics-quality CALIBRATED \\\n"
          f"    --intrinsics-source realsense_factory_intrinsics_meta_json \\\n"
          f"    --stride 30")
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
