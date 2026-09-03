#!/usr/bin/env python3
"""`release/pallet-pose-yolo26n-livegt` 폴더를 HuggingFace 로 올린다.

메모리에 남은 기존 절차를 따른다 — 로컬 release 폴더를 정본으로 두고
``upload_folder`` 로 통째로 올린 뒤, 원격 sha256 을 로컬과 대조한다.

토큰은 인자로 받지 않는다.  대화나 명령 이력에 평문으로 남기지 않기 위해서다.
``huggingface-cli login`` 으로 저장된 토큰이나 ``HF_TOKEN`` 환경변수를 쓴다.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

from huggingface_hub import HfApi

REPO = Path(__file__).resolve().parents[1]
LOCAL = REPO / "challenge/yolo_pose_one_model/release/pallet-pose-yolo26n-livegt"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo-id", default="CanelE452/pallet-pose-yolo26n-livegt")
    ap.add_argument("--private", action="store_true",
                    help="비공개로 만든다. 성능 근거가 불충분한 동안 권장")
    ap.add_argument("--dry-run", action="store_true", help="올리지 않고 점검만")
    args = ap.parse_args(argv)

    if not LOCAL.is_dir():
        print(f"[FAIL] 로컬 패키지가 없다: {LOCAL}")
        return 1

    files = sorted(p for p in LOCAL.iterdir() if p.is_file())
    total = sum(p.stat().st_size for p in files)
    print(f"업로드 대상 {len(files)}개, {total/1e6:.1f} MB → {args.repo_id}"
          f"{' (private)' if args.private else ' (public)'}")
    for p in files:
        print(f"   {p.name:36} {p.stat().st_size/1e6:8.2f} MB")

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    try:
        who = api.whoami()
        print(f"\n로그인: {who.get('name')}")
    except Exception as exc:                       # 토큰이 없으면 여기서 멈춘다
        print(f"\n[FAIL] HuggingFace 인증이 없다: {type(exc).__name__}")
        print("  별도 터미널에서 `huggingface-cli login` 을 실행하거나")
        print("  HF_TOKEN 환경변수를 설정한 뒤 다시 실행하라.")
        return 2

    if args.dry_run:
        print("\n[DRY-RUN] 여기까지. 실제 업로드는 --dry-run 없이.")
        return 0

    api.create_repo(repo_id=args.repo_id, repo_type="model",
                    private=args.private, exist_ok=True)
    api.upload_folder(
        folder_path=str(LOCAL), repo_id=args.repo_id, repo_type="model",
        commit_message="Add YOLO26n-pose fine-tuned on 402 hand-labelled field frames")
    print("업로드 완료")

    # 올라간 것이 로컬과 같은지 확인한다.  "올렸다" 와 "제대로 올라갔다" 는 다르다.
    print("\n원격 sha256 대조:")
    mismatches = 0
    for info in api.list_repo_files(repo_id=args.repo_id, repo_type="model"):
        local = LOCAL / info
        if not local.is_file():
            continue
        meta = api.get_paths_info(repo_id=args.repo_id, paths=[info],
                                  repo_type="model")[0]
        remote = getattr(getattr(meta, "lfs", None), "sha256", None)
        if remote is None:
            print(f"   {info:36} (LFS 아님 — 해시 비교 생략)")
            continue
        ok = remote == sha256_of(local)
        mismatches += 0 if ok else 1
        print(f"   {info:36} {'OK' if ok else '불일치'}")
    print(f"\nhttps://huggingface.co/{args.repo_id}")
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
