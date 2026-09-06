#!/usr/bin/env python3
"""수동 어노 GT 에서 좌우 flip / 센서 노이즈 증강본을 만든다.

``gen_truncation_crops.py`` 와 같은 자리에 두고 같은 규약으로 쓴다 — 산출물은
``projected_cuboid``(코너 8) + ``projected_cuboid_centroid`` 를 가진 NDDS json 과
640x480 png 쌍이다.  ``prepare_yolo_pose.load_kps`` 가 읽는 건 이 둘뿐이다.

**flip 은 학습 옵션으로 켜면 안 된다.**  ``YOLO26_TRAINING_SPEC`` 이 ``fliplr`` 를
0.0 으로 못박았고 학습 전 가드가 이를 검사한다.  그래서 파일로 만든다.

좌우 flip 은 픽셀만 뒤집는 게 아니다.  camera-facing 0123 규약에서 좌우가 바뀌면
코너 이름도 바뀐다::

    x  ->  W - 1 - x
    코너 순서 0<->1, 2<->3, 4<->5, 6<->7      (centroid 는 불변)

``pose_transform`` 은 flip 본에서 ``null`` 로 둔다.  거울 반사는 회전이 아니라
det=-1 이라 그대로 두면 틀린 6D pose 가 남는다.  학습 경로가 이 필드를 읽지 않으므로
지우는 편이 안전하다(나중에 PnP 용으로 잘못 집어가는 것을 막는다).

노이즈는 라벨을 바꾸지 않는다.  센서 노이즈(가우시안) + 약한 밝기/대비 지터를 준다.
색상 지터(hsv)는 학습 로더가 이미 하므로 여기서는 겹치지 않게 세기만 낮게 둔다.

사용 예::

    python challenge/scripts/dataset/gen_flip_noise_aug.py \\
        --out challenge/data/03_derived/flip_noise_aug_livegt --modes flip noise
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
GT_ROOT = REPO / "challenge/data/01_real/live_capture_gt"
CAPTURE_ROOT = REPO / "challenge/data/01_real/_live_captures"

# 좌우가 바뀌면 코너 이름도 바뀐다.  9 점짜리(centroid 포함)는 마지막이 불변.
import keypoint_annotations_transform as kat  # noqa: E402

FLIP_PERM_8 = kat.FLIP_PERM_8


def sessions() -> dict[str, Path]:
    """GT 폴더 이름에서 세션을 찾는다.

    하드코딩 목록을 두면 세션이 늘 때마다 조용히 빠진다 — 실제로 6 개짜리 목록이
    24 개가 된 뒤에도 그대로였다.  이름 규약(``<세션>_manual_gt``)으로 찾는다.
    """
    found = {}
    for gt_dir in sorted(GT_ROOT.glob("*_manual_gt")):
        name = gt_dir.name[: -len("_manual_gt")]
        hits = list(CAPTURE_ROOT.glob(f"*/sessions/{name}/rgb"))
        if len(hits) == 1:
            found[name] = hits[0]
        elif hits:
            print(f"   ⚠️ {name}: rgb 후보 {len(hits)}개 — 건너뜀")
    return found


def flip(img: np.ndarray, obj: dict, parent_frame: str | None = None
         ) -> tuple[np.ndarray, dict]:
    w = img.shape[1]
    out = dict(obj)
    proj = obj["projected_cuboid"]
    moved = [[w - 1.0 - float(p[0]), float(p[1])] for p in proj[:8]]
    out["projected_cuboid"] = [moved[i] for i in FLIP_PERM_8]
    cen = obj.get("projected_cuboid_centroid")
    if cen:
        out["projected_cuboid_centroid"] = [w - 1.0 - float(cen[0]), float(cen[1])]
    # ``keypoint_annotations`` 도 같이 뒤집는다.  ``dict(obj)`` 는 얕은 복사라
    # 이걸 안 하면 이미지와 projected_cuboid 만 뒤집히고 keypoint_annotations 는
    # 원본 그대로 남는다.  학습 변환기(prepare_yolo_pose.load_kps)가 그 필드를
    # 우선하므로, 그대로 두면 **뒤집힌 이미지에 안 뒤집힌 라벨**이 붙는다.
    ann = obj.get("keypoint_annotations")
    if isinstance(ann, list) and len(ann) >= 9:
        flipped = []
        for entry in ann[:9]:
            e = dict(entry)
            xy = e.get("xy")
            if xy is not None:
                e["xy"] = [w - 1.0 - float(xy[0]), float(xy[1])]
            flipped.append(e)
        # 코너 8개는 좌우 짝을 맞바꾸고, centroid(8)는 제자리다.
        out["keypoint_annotations"] = [flipped[i] for i in FLIP_PERM_8] + [flipped[8]]
    out["pose_transform"] = None          # 반사는 유효한 회전이 아니다
    out["aug"] = "fliplr"
    out.update(kat.provenance(parent_frame,
                              {"kind": "hflip", "width": w,
                               "perm": list(FLIP_PERM_8)}))
    return cv2.flip(img, 1), out


def noise(img: np.ndarray, obj: dict, rng: random.Random,
          parent_frame: str | None = None) -> tuple[np.ndarray, dict]:
    sigma = rng.uniform(4.0, 12.0)          # 8bit 기준 센서 노이즈
    gain = rng.uniform(0.92, 1.08)          # 약한 대비
    bias = rng.uniform(-10.0, 10.0)         # 약한 밝기
    f = img.astype(np.float32) * gain + bias
    f += np.random.normal(0.0, sigma, img.shape).astype(np.float32)
    out = dict(obj)
    out["aug"] = f"noise(sigma={sigma:.1f},gain={gain:.2f},bias={bias:+.1f})"
    # 노이즈는 좌표를 안 바꾸므로 keypoint_annotations 는 그대로가 맞다.
    out.update(kat.provenance(parent_frame,
                              {"kind": "noise", "sigma": round(sigma, 2),
                               "gain": round(gain, 3), "bias": round(bias, 2)}))
    return np.clip(f, 0, 255).astype(np.uint8), out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="challenge/data/03_derived/flip_noise_aug_livegt")
    ap.add_argument("--modes", nargs="+", default=["flip", "noise"],
                    choices=["flip", "noise"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="세션당 최대 장수 (스모크용)")
    args = ap.parse_args(argv)

    out_dir = REPO / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    sess = sessions()
    print(f"세션 {len(sess)}개   modes={args.modes}   -> {out_dir}")

    made = {m: 0 for m in args.modes}
    skipped = {"no_image": 0, "bad_gt": 0}
    for name, rgb_dir in sess.items():
        anns = sorted((GT_ROOT / f"{name}_manual_gt").glob("*.json"))
        if args.limit:
            anns = anns[: args.limit]
        for ann in anns:
            img_path = rgb_dir / f"{ann.stem}.png"
            if not img_path.is_file():
                skipped["no_image"] += 1
                continue
            doc = json.loads(ann.read_text(encoding="utf-8"))
            objs = doc.get("objects") or []
            if not objs or len(objs[0].get("projected_cuboid") or []) < 8:
                skipped["bad_gt"] += 1
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                skipped["no_image"] += 1
                continue

            for mode in args.modes:
                if mode == "flip":
                    new_img, new_obj = flip(img, objs[0], str(ann))
                else:
                    new_img, new_obj = noise(img, objs[0], rng, str(ann))
                stem = f"{name}_{ann.stem}_{'f' if mode == 'flip' else 'n'}"
                cv2.imwrite(str(out_dir / f"{stem}.png"), new_img)
                (out_dir / f"{stem}.json").write_text(json.dumps({
                    "camera_data": doc.get("camera_data", {}),
                    "objects": [new_obj],
                }, ensure_ascii=False), encoding="utf-8")
                made[mode] += 1

    total = sum(made.values())
    print(f"\n생성 {total}장   {made}")
    if any(skipped.values()):
        print(f"건너뜀 {skipped}")

    n_json = len(list(out_dir.glob("*.json")))
    n_png = len(list(out_dir.glob("*.png")))
    print(f"디스크 확인: json {n_json}  png {n_png}")
    return 0 if n_json == n_png else 1


if __name__ == "__main__":
    raise SystemExit(main())
