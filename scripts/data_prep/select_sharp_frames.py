"""영상에서 선명한(안 흔들린) 프레임을 골라 저장.

선명도 = Laplacian variance. 임계값 이상 프레임을 추출한다.
파렛트 가시성 판정은 하지 않음 (선명도만).

임계값은 두 방식 중 하나:
  --thr <값>   고정 절대 임계 (모든 영상 동일)
  --pct <백분위> 영상별 적응형 — 각 영상 lapvar 분포의 하위 <pct>%를 컷.
                영상마다 해상도/배경 텍스처로 절대 스케일이 달라질 때 사용.

--stride 로 인접 중복(거의 동일한 연속 프레임)을 솎는다: 직전 저장 프레임과
원본 인덱스 차이가 stride 이상일 때만 저장.

예:
  # 영상별 적응형(하위 20% 컷) + 솎기(원본 12프레임 간격)
  python scripts/data_prep/select_sharp_frames.py \
    --videos data/pallet/raw_data/wood/*.mp4 \
    --out data/pallet/raw_data/wood/selected --pct 20 --stride 12
"""
import argparse
import glob
from pathlib import Path

import cv2
import numpy as np


def lap_var(frame):
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(g, cv2.CV_64F).var()


def collect_lapvars(vp):
    """영상 전체 프레임의 lapvar 리스트 (1-pass, 프레임 보관 안 함)."""
    cap = cv2.VideoCapture(vp)
    vals = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        vals.append(lap_var(fr))
    cap.release()
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", required=True, help="영상 경로 (glob 가능)")
    ap.add_argument("--out", required=True, help="출력 폴더")
    ap.add_argument("--thr", type=float, default=None, help="고정 절대 임계 (모든 영상 동일)")
    ap.add_argument("--pct", type=float, default=None,
                    help="영상별 적응형: 각 영상 lapvar 하위 pct%%를 컷")
    ap.add_argument("--stride", type=int, default=1,
                    help="직전 저장 프레임과 원본 idx 차이가 이 값 이상일 때만 저장 (중복 솎기)")
    args = ap.parse_args()

    if (args.thr is None) == (args.pct is None):
        raise SystemExit("--thr 와 --pct 중 정확히 하나를 지정하세요")

    paths = []
    for v in args.videos:
        paths.extend(sorted(glob.glob(v)))
    if not paths:
        raise SystemExit("영상을 찾지 못함")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    total = 0
    for vp in paths:
        stem = Path(vp).stem
        vout = out / stem  # 영상별 하위 폴더
        vout.mkdir(parents=True, exist_ok=True)

        if args.pct is not None:
            vals = collect_lapvars(vp)
            thr = float(np.percentile(vals, args.pct))
            n_total = len(vals)
        else:
            thr = args.thr
            n_total = None

        cap = cv2.VideoCapture(vp)
        idx = saved = 0
        last_saved = -10**9
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if lap_var(fr) > thr and (idx - last_saved) >= args.stride:
                cv2.imwrite(str(vout / f"{idx:06d}.jpg"), fr,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved += 1
                last_saved = idx
            idx += 1
        cap.release()
        denom = n_total if n_total is not None else idx
        print(f"{vp}: {saved}/{denom} 저장 "
              f"(thr={thr:.1f}{'=p'+str(args.pct) if args.pct is not None else ''}, "
              f"stride={args.stride}) → {vout}")
        total += saved
    print(f"\n총 {total}장 → {out}/<영상명>/")


if __name__ == "__main__":
    main()
