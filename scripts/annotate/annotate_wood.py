"""challenge/scripts/annotate/annotate_wood.py — 나무 파렛트(wood) 전용 어노테이션 런처.

기존 annotate.py 를 **수정하지 않고** wood 파렛트(실측 0.8 × 0.59 × 0.14 m)로 PnP 되게
한다. 공유 모듈 annotate_pnp.PALLET_DIMS(표준 1.1×1.3, eval 파이프라인 20+곳이 import)는
소스 그대로 두고, 이 런처 **프로세스 안에서만** 런타임 override 한다 (eval 스크립트는 별도
프로세스라 영향 없음).

하는 일:
  1. annotate_pnp.PALLET_DIMS = (0.8, 0.59, 0.14)  ← annotate import 전에 세팅
  2. wood jpg(평면) → annotate 가 기대하는 <seq>/rgb/*.png 레이아웃으로 심링크 준비
  3. cam_K.txt 생성 (★wood 카메라 K 미지 = 폰 영상 → HFOV 추정값. --K 로 실제값 교체 가능)
  4. annotate.main() 호출

사용:
  conda activate pallet-pose
  python challenge/scripts/annotate/annotate_wood.py --video pallet_20260618_183705 \
      --stride 5 --population-role DEV
  # 실제 K 있으면:  --K path/to/cam_K.txt   (3x3, 프레임 해상도 기준)
  # 또는:           --hfov 63   (수평 화각 deg 로 fx 추정)

주의:
  - K 는 추정값이므로 PnP pose / 자동채움(f,g) 코너는 근사. **직접 클릭한 2D 점은 정확**.
  - 정확 GT 원하면 촬영 카메라 intrinsics 를 --K 로 넣을 것.
  - dims W/D 순서는 solve_pose 가 (W,D,H)/(D,W,H) 둘 다 시도하므로 자동 처리. H=0.14 고정.
"""
from __future__ import annotations
import os as _os, sys as _sys

# --- challenge/scripts 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_CS = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_CS] + [_os.path.join(_CS, _d) for _d in sorted(_os.listdir(_CS))
                         if _os.path.isdir(_os.path.join(_CS, _d)) and not _d.startswith(".")]

import argparse
import glob
import math
import os
import sys

import numpy as np
from PIL import Image

WOOD_DIMS = (0.8, 0.59, 0.14)   # width, depth, height (m) — 실측 80×59×14cm

# wood 는 RealSense D435I 로 촬영 (2026-07-09 라이브 쿼리 실측). intrinsic 은 해상도별
# 로 다르다. (fx, fy, cx, cy):
RS_INTRINSICS = {
    (640, 480):  (605.9065, 605.9698, 317.5962, 256.2923),   # 4:3
    (1920, 1080): (1363.2896, 1363.4321, 954.5915, 576.6577),  # 16:9
}
# 1280×720 은 미쿼리이나 1920×1080 과 동일 16:9 FOV → 선형 ×(w/1920) 유도 (기하적 정합).


def K_for_resolution(w, h):
    """해상도에 맞는 RealSense K. 표에 있으면 그대로, 없으면 16:9 는 1080p 에서 유도."""
    if (w, h) in RS_INTRINSICS:
        fx, fy, cx, cy = RS_INTRINSICS[(w, h)]
        src = f"RealSense 실측 {w}x{h}"
    elif abs(w / h - 16 / 9) < 0.02:      # 16:9 → 1920x1080 에서 선형 유도
        fx0, fy0, cx0, cy0 = RS_INTRINSICS[(1920, 1080)]
        sx, sy = w / 1920.0, h / 1080.0
        fx, fy, cx, cy = fx0 * sx, fy0 * sy, cx0 * sx, cy0 * sy
        src = f"1920x1080 에서 16:9 선형유도 → {w}x{h}"
    else:
        return None, f"{w}x{h} = RealSense 표에 없고 16:9 도 아님 (--K/--hfov 필요)"
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64), src

_HERE = os.path.dirname(os.path.abspath(__file__))
# 2026-08-15 재편: 이 파일이 challenge/scripts/<계열>/ 로 한 단계 내려갔다.
# dirname 을 하나 더 감싸야 저장소 루트다 (계열 -> scripts -> challenge -> repo).
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)

WOOD_ROOT = os.path.join(_REPO, "data", "pallet", "raw_data", "wood", "selected")


def estimate_K(w, h, hfov_deg):
    """해상도 + 수평 HFOV 로 K 추정 (폰 카메라, calibration 없을 때)."""
    fx = (w / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    fy = fx  # square pixel 가정
    return np.array([[fx, 0, w / 2.0], [0, fy, h / 2.0], [0, 0, 1]], dtype=np.float64)


def prep_seq(video, K):
    """wood/selected/<video>/*.jpg → <seq>/rgb/NNNNNN.png 심링크 + cam_K.txt.
    cv2.imread 는 내용(magic)으로 디코드하므로 .png 심링크가 jpg 를 가리켜도 정상."""
    src = os.path.join(WOOD_ROOT, video)
    jpgs = sorted(glob.glob(os.path.join(src, "*.jpg")))
    if not jpgs:
        sys.exit(f"[ERROR] no jpg in {src}")
    seq = os.path.join(_REPO, "data", "pallet", "raw_data", "wood", f"_annotate_{video}")
    rgb = os.path.join(seq, "rgb")
    os.makedirs(rgb, exist_ok=True)
    for j in jpgs:
        stem = os.path.splitext(os.path.basename(j))[0]
        link = os.path.join(rgb, f"{stem}.png")
        if not os.path.islink(link) and not os.path.exists(link):
            os.symlink(os.path.abspath(j), link)
    np.savetxt(os.path.join(seq, "cam_K.txt"), K)
    return seq, len(jpgs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="pallet_20260618_183705",
                    help="wood/selected/ 하위 폴더명 (pallet_20260618_183705 / "
                         "pallet_20260618_184309 / 20260618_132917)")
    ap.add_argument("--stride", type=int, default=5, help="N frame 마다 1개")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--K", default=None, help="실제 cam_K.txt (3x3). 최우선")
    ap.add_argument("--hfov", type=float, default=None,
                    help="주면 이 수평 화각(deg)로 K 추정. 안 주면 이전 어노와 동일 K(PREV_K)")
    ap.add_argument("--out_dir", default=None,
                    help="기본: challenge/data/wood_<video>_manual_gt")
    ap.add_argument("--population-role", "--population_role", required=True,
                    choices=["DEV", "FINAL"],
                    help="annotate.py와 동일한 명시적 population 역할; 자동 추론 없음")
    args = ap.parse_args()

    src = os.path.join(WOOD_ROOT, args.video)
    jpgs = sorted(glob.glob(os.path.join(src, "*.jpg")))
    if not jpgs:
        sys.exit(f"[ERROR] no jpg in {src}\n  가능: {os.listdir(WOOD_ROOT)}")
    w, h = Image.open(jpgs[0]).size

    if args.K:
        K = np.loadtxt(args.K).reshape(3, 3)
        k_src = f"file:{args.K}"
    elif args.hfov is not None:
        K = estimate_K(w, h, args.hfov)
        k_src = f"ESTIMATED (hfov={args.hfov}deg)"
    else:
        K, k_src = K_for_resolution(w, h)   # RealSense 해상도별 실측/유도
        if K is None:
            sys.exit(f"[ERROR] K 자동선택 실패: {k_src}. --K 또는 --hfov 로 지정.")

    seq, n = prep_seq(args.video, K)
    out_dir = args.out_dir or os.path.join(_REPO, "challenge", "data",
                                           f"wood_{args.video}_manual_gt")

    print("=" * 64)
    print(f"[annotate_wood] video={args.video}  frames={n}  res={w}x{h}")
    print(f"  PALLET_DIMS = {WOOD_DIMS} m  (wood 실측 80x59x14cm)")
    print(f"  K = fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}")
    print(f"  K source: {k_src}")
    print(f"  out_dir = {out_dir}")
    print("=" * 64)

    # ── 공유 코드 건드리지 않고 이 프로세스에서만 dims override ──
    import annotate_pnp
    annotate_pnp.PALLET_DIMS = WOOD_DIMS          # solve_pose 가 call-time 에 읽음
    import annotate                                # 이제 annotate 의 copy 도 wood

    sys.argv = ["annotate_wood", "--seq", seq, "--stride", str(args.stride),
                "--start", str(args.start), "--out_dir", out_dir,
                "--population-role", args.population_role]
    annotate.main()


if __name__ == "__main__":
    main()
