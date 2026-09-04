#!/usr/bin/env python3
"""담당자가 보낸 프레임 배포본(zip)을 annotate.py 가 읽는 세션으로 푼다.

``extract_forklift_recording.py`` 와 목적은 같지만 입력이 다르다.  저쪽은 ``raw.mp4``
를 직접 디코딩하고, 이쪽은 **이미 PNG 로 추출되어 온 zip** 을 배치만 한다::

    video_frames/forklift_v4_recording_<날짜>_<시각>_frame_<i>.png   녹화 추출본
    captured_images/<글로벌인덱스>.png                                직접 촬영본
    MANIFEST.csv    source_type,archive_path,source_name,source_frame_index,...
    README.txt

출력은 annotate.py 의 세션 규약을 따른다::

    <out>/<세션>/rgb/000000.png ...      video_frame  (세션 내 프레임 인덱스)
    <out>/<세션>/rgb/014578.png ...      captured     (원본 이름 유지)
    <out>/<세션>/cam_K.txt

``cam_K.txt`` 는 zip 에 없다.  배포본에 intrinsics 가 들어 있지 않기 때문이다.
그래서 ``--cam-k`` 로 기존 세션의 것을 받아 복사하고, 어디서 왔는지를
``_import.json`` 에 남긴다.  이 파일을 빼먹으면 ``_resolve_session_intrinsics`` 가
**경고 없이** legacy default K 로 떨어져 잘못된 intrinsics 로 푼 GT 가 조용히 쌓인다.

사용 예::

    python scripts/data_prep/import_frame_bundle.py \\
        --zip ~/Downloads/김민재_9-4.zip \\
        --out challenge/data/01_real/_live_captures/forklift_v4_20260904/sessions \\
        --prefix forklift_v4_20260904 \\
        --cam-k challenge/data/01_real/_live_captures/forklift_v4_20260901/sessions/forklift_v4_173507/cam_K.txt
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import re
import shutil
import sys
import time
import zipfile

VIDEO_RE = re.compile(r"^video_frames/.*_(\d{8})_(\d{6})_frame_(\d+)\.png$")
CAPTURED_RE = re.compile(r"^captured_images/(.+)\.png$")


def plan(zf: zipfile.ZipFile, prefix: str) -> tuple[dict, list]:
    """zip 항목을 (세션 → [(zip 경로, 출력 파일명)]) 으로 가른다.

    녹화 추출본은 세션(시각)마다 프레임 인덱스를 0 부터 다시 세므로 그대로 쓰고,
    직접 촬영본은 원본 이름이 이미 전역 인덱스라 유지한다.  ``captured`` 세션을
    따로 두는 이유는 촬영 방식이 달라 split 단위가 되어야 하기 때문이다.
    """
    sessions: dict[str, list] = {}
    skipped = []
    for name in zf.namelist():
        if name.endswith("/"):
            continue
        m = VIDEO_RE.match(name)
        if m:
            _, hhmmss, idx = m.groups()
            sessions.setdefault(f"{prefix}_{hhmmss}", []).append(
                (name, f"{int(idx):06d}.png"))
            continue
        m = CAPTURED_RE.match(name)
        if m:
            sessions.setdefault(f"{prefix}_captured", []).append(
                (name, f"{m.group(1)}.png"))
            continue
        if not name.endswith((".csv", ".txt")):
            skipped.append(name)
    for jobs in sessions.values():
        jobs.sort(key=lambda j: j[1])
    return sessions, skipped


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", required=True, help="세션들이 들어갈 sessions/ 폴더")
    ap.add_argument("--prefix", required=True, help="세션 이름 접두어")
    ap.add_argument("--cam-k", required=True,
                    help="각 세션에 복사할 cam_K.txt (같은 장비·같은 해상도여야 한다)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    zip_path = Path(args.zip).expanduser()
    out_root = Path(args.out)
    cam_k = Path(args.cam_k)
    for p, what in ((zip_path, "zip"), (cam_k, "cam_K")):
        if not p.is_file():
            print(f"[FAIL] {what} 가 없다: {p}")
            return 1

    with zipfile.ZipFile(zip_path) as zf:
        sessions, skipped = plan(zf, args.prefix)
        total = sum(len(v) for v in sessions.values())
        print(f"zip  : {zip_path}  ({len(zf.namelist())} entries)")
        print(f"세션 : {len(sessions)}개, 이미지 {total}장")
        for name in sorted(sessions):
            print(f"   {name:36} {len(sessions[name]):6d}")
        if skipped:
            print(f"   ⚠️ 분류 못한 항목 {len(skipped)}개: {skipped[:3]}")
        if args.dry_run:
            return 0

        started = time.time()
        written = {}
        for name in sorted(sessions):
            rgb = out_root / name / "rgb"
            rgb.mkdir(parents=True, exist_ok=True)
            for src, dst in sessions[name]:
                with zf.open(src) as fin, open(rgb / dst, "wb") as fout:
                    shutil.copyfileobj(fin, fout, 1 << 20)
            shutil.copyfile(cam_k, out_root / name / "cam_K.txt")
            written[name] = len(list(rgb.glob("*.png")))
            print(f"   {name:36} {written[name]:6d} 장 기록", flush=True)
        elapsed = time.time() - started

        for meta in ("MANIFEST.csv", "README.txt"):
            if meta in zf.namelist():
                (out_root.parent / meta).write_bytes(zf.read(meta))

    # 선언이 아니라 디스크로 검증한다 — 개수와 해상도를 실제로 읽는다.
    bad = {n: (len(sessions[n]), written[n])
           for n in sessions if written[n] != len(sessions[n])}
    sizes = set()
    try:
        from PIL import Image
        for name in sorted(sessions):
            first = sorted((out_root / name / "rgb").glob("*.png"))[0]
            sizes.add(Image.open(first).size)
    except ImportError:
        sizes = {"(PIL 없음 — 해상도 미확인)"}

    (out_root.parent / "_import.json").write_text(json.dumps({
        "zip": str(zip_path), "prefix": args.prefix,
        "cam_k_source": str(cam_k),
        "cam_k_note": "zip 에 intrinsics 가 없어 같은 장비의 기존 세션에서 복사했다",
        "sessions": {n: written[n] for n in sorted(written)},
        "total_images": sum(written.values()),
        "image_sizes": sorted(map(str, sizes)),
        "elapsed_seconds": round(elapsed, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n합계 {sum(written.values())}장  해상도 {sorted(map(str, sizes))}"
          f"  {elapsed/60:.1f}분")
    if bad:
        print(f"[FAIL] 개수 불일치: {bad}")
        return 1
    print(f"기록: {out_root.parent / '_import.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
