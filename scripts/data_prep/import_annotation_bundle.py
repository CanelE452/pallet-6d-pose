#!/usr/bin/env python3
"""다른 사람이 어노테이션해 보낸 묶음(zip 해제본)을 프로젝트 규약으로 반입한다.

``import_frame_bundle.py`` 는 라벨 없는 프레임을 받고, 이쪽은 **GT 가 붙은 것**을 받는다.
입력은 한 폴더에 json/png 가 쌍으로 들어 있는 평평한 구조다::

    <bundle>/forklift_v4_recording_<날짜>_<시각>_frame_<i>.json
    <bundle>/forklift_v4_recording_<날짜>_<시각>_frame_<i>.png

출력은 프로젝트가 이미 쓰는 두 갈래로 나눈다 — 이미지와 GT 를 한 폴더에 두면
``prepare_yolo_pose_from_live_gt.py`` 가 이미지를 못 찾는다::

    _live_captures/<group>/sessions/<prefix>_<시각>/rgb/<i:06d>.png
    live_capture_gt/<prefix>_<시각>_manual_gt/<i:06d>.json

이미 있는 세션에는 **덧붙인다**.  같은 프레임 번호가 이미 있으면 건너뛰고 마지막에
센다 — 남의 라벨로 내 라벨을 조용히 덮어쓰지 않기 위해서다.

``cam_K.txt`` 는 묶음에 없으므로 ``--cam-k`` 로 받아 새 세션에만 복사한다.
빠뜨리면 ``_resolve_session_intrinsics`` 가 경고 없이 legacy default K 로 떨어진다.

반입 뒤에는 **4-fold 정규화를 따로 돌려야 한다** — 사람마다 45° 부근 앞면 선택이
갈리므로(`canonicalize_fourfold_yaw.py`), 이 스크립트는 라벨 값을 건드리지 않는다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil

FRAME_RE = re.compile(r"_(\d{8})_(\d{6})_frame_(\d+)$")


def plan(bundle: Path, prefix_fmt: str) -> tuple[dict, list]:
    """묶음 파일을 (세션 -> [(json, png, 프레임번호)]) 로 가른다."""
    sessions: dict[str, list] = {}
    orphans = []
    for js in sorted(bundle.glob("*.json")):
        m = FRAME_RE.search(js.stem)
        if not m:
            orphans.append(js.name)
            continue
        date, hhmmss, idx = m.groups()
        png = js.with_suffix(".png")
        if not png.is_file():
            orphans.append(js.name)
            continue
        sessions.setdefault(prefix_fmt.format(date=date, time=hhmmss), []).append(
            (js, png, int(idx)))
    return sessions, orphans


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("bundle", help="json/png 가 평평하게 든 폴더")
    ap.add_argument("--captures-root", required=True,
                    help="이미지가 들어갈 <group>/sessions 폴더")
    ap.add_argument("--gt-root", required=True, help="GT json 이 들어갈 폴더")
    ap.add_argument("--prefix", default="forklift_v4_{date}",
                    help="세션 이름 접두어. {date} 를 쓸 수 있다")
    ap.add_argument("--cam-k", required=True, help="새 세션에 복사할 cam_K.txt")
    ap.add_argument("--apply", action="store_true", help="실제로 쓴다 (기본 dry-run)")
    args = ap.parse_args(argv)

    bundle = Path(args.bundle).expanduser()
    cam_k = Path(args.cam_k)
    if not bundle.is_dir():
        print(f"[FAIL] 묶음 폴더가 없다: {bundle}")
        return 1
    if not cam_k.is_file():
        print(f"[FAIL] cam_K 가 없다: {cam_k}")
        return 1

    sessions, orphans = plan(bundle, args.prefix + "_{time}")
    cap_root, gt_root = Path(args.captures_root), Path(args.gt_root)

    print(f"묶음 : {bundle.name}   세션 {len(sessions)}개, 프레임 "
          f"{sum(len(v) for v in sessions.values())}장")
    if orphans:
        print(f"   ⚠️ 짝 없는/이름 다른 파일 {len(orphans)}개: {orphans[:3]}")

    written = skipped = new_sessions = 0
    for name in sorted(sessions):
        rgb = cap_root / name / "rgb"
        gt = gt_root / f"{name}_manual_gt"
        fresh = not rgb.is_dir()
        exist = {p.stem for p in gt.glob("*.json")} if gt.is_dir() else set()
        jobs = [(j, p, i) for j, p, i in sessions[name] if f"{i:06d}" not in exist]
        dup = len(sessions[name]) - len(jobs)
        lo = min(i for _, _, i in sessions[name])
        hi = max(i for _, _, i in sessions[name])
        print(f"   {name:34} {len(jobs):4d}장 (중복 {dup})  frame {lo}~{hi}"
              + ("  ★새 세션" if fresh else ""))
        skipped += dup
        if not args.apply:
            continue
        rgb.mkdir(parents=True, exist_ok=True)
        gt.mkdir(parents=True, exist_ok=True)
        if fresh:
            new_sessions += 1
            shutil.copyfile(cam_k, cap_root / name / "cam_K.txt")
        for js, png, idx in jobs:
            shutil.copyfile(png, rgb / f"{idx:06d}.png")
            shutil.copyfile(js, gt / f"{idx:06d}.json")
            written += 1

    if not args.apply:
        print("\n[DRY-RUN] --apply 를 줘야 실제로 쓴다")
        return 0

    # 선언이 아니라 디스크로 확인한다.
    ok = True
    for name in sorted(sessions):
        n_gt = len(list((gt_root / f"{name}_manual_gt").glob("*.json")))
        n_im = len(list((cap_root / name / "rgb").glob("*.png")))
        if n_gt > n_im:
            print(f"[FAIL] {name}: GT {n_gt} > 이미지 {n_im}")
            ok = False
    print(f"\n기록 {written}장 · 중복 건너뜀 {skipped} · 새 세션 {new_sessions}")
    print("⚠️ 다음: canonicalize_fourfold_yaw.py 로 4-fold 정규화를 돌릴 것")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
