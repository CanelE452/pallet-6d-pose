"""Daytime / Nighttime unlabeled adaptation pool 을 leakage-free 로 고정한다.

MAIN self-training 은 모델을 **하나** 만든다.  주간 모델과 야간 모델을 따로 MAIN 으로
만들지 않는다.  배포 환경 하나가 주간과 야간을 모두 보기 때문이다.  그래서 두 조건을
같은 수로 섞은 balanced union 을 쓴다.

    U_MAIN = Daytime N_COMMON + Nighttime N_COMMON

pool 후보는 새로 정하지 않는다.  `metric_split_lock.md` §1.6 이 이미 세션 단위로
동결한 `pl_pool` 을 그대로 쓴다 (`data/pallet/eval_results/split_lock/`).
그 lock 은 `pool∩filter-val=∅`, `pool∩final-test=∅` 를 이미 통과했다.

    OUTSIDE(daytime)   capturepallet01, capturepallet10, capturepallet11
    NIGHT(nighttime)   capturenight01..04, capturenight10

여기서 다시 검증하는 것은 lock 이후에 생긴 위험, 즉 **새로 어노테이션된 PAPER_EVAL
프레임과의 중복**이다.  세션 이름이 아니라 이미지 SHA256 으로 판정한다.

    adapt_session ∩ eval_session = 0
    adapt_sha256  ∩ eval_sha256  = 0     (positive + negative 모두)

sampling 은 sha256 오름차순 앞에서 N 개다.  sha256 은 내용에 대해 균일하므로
결정적이면서 편향이 없고, 세션·시간 순서에 쏠리지 않는다.

사용:
    python scripts/self_training_yolo/build_adaptation_pool.py [--target 500]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from evaluation.eval_workspace import (  # noqa: E402
    evaluation_population_views,
    load_frames,
)

WORKSPACE = REPO_ROOT / "data" / "evaluation" / "pallet_eval_v1"
SPLIT_LOCK = REPO_ROOT / "data" / "pallet" / "eval_results" / "split_lock"
OUT_DIR = WORKSPACE / "adaptation"

# metric_split_lock.md §1.6 에서 그대로 옮긴 세션 목록.  여기서 새로 고르지 않는다.
POOL_SESSIONS = {
    "daytime": (
        REPO_ROOT / "data" / "pallet" / "raw_data" / "outside",
        ("capturepallet01", "capturepallet10", "capturepallet11"),
    ),
    "nighttime": (
        REPO_ROOT / "data" / "pallet" / "raw_data" / "night",
        ("capturenight01", "capturenight02", "capturenight03", "capturenight04",
         "capturenight10"),
    ),
}

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
# capture 세션은 rgb/ 와 depth/ 를 나란히 담는다.  depth map 은 pseudo-label 대상이
# 아니므로 rgb/ 만 센다.  이 구분을 빼면 pool 의 절반이 depth 로 채워진다.
RGB_SUBDIR = "rgb"
PREFERRED_TARGET = 500
MAIN_POOL_MINIMUM = 150


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _eval_identity() -> tuple[set[str], set[str]]:
    """PAPER_EVAL 이 실제로 쓰는 이미지의 sha256 과 세션 이름."""

    frames = load_frames(WORKSPACE)
    views = evaluation_population_views(frames)
    shas: set[str] = set()
    sessions: set[str] = set()
    for key in ("PAPER_EVAL_POSITIVE", "PAPER_EVAL_NEGATIVE"):
        for row in views[key]:
            sessions.add(row.get("session_id", ""))
            for field in ("image_sha256", "source_image_sha256"):
                value = row.get(field)
                if value:
                    shas.add(value)
    # 원본 촬영 세션 이름도 함께 막는다.  workspace 세션명은 재명명된 것이라
    # 원본 capture id 와 문자열이 다를 수 있다.
    for row in frames:
        source = row.get("source_image_path") or ""
        for part in Path(source).parts:
            if part.startswith(("capturepallet", "capturenight")):
                sessions.add(part)
    sessions.discard("")
    return shas, sessions


def _pool_rows(condition: str) -> list[dict[str, str]]:
    root, sessions = POOL_SESSIONS[condition]
    rows: list[dict[str, str]] = []
    for session in sessions:
        session_dir = root / session
        if not session_dir.is_dir():
            raise SystemExit(f"MISSING_POOL_SESSION: {session_dir}")
        rgb_dir = session_dir / RGB_SUBDIR
        if not rgb_dir.is_dir():
            raise SystemExit(f"MISSING_RGB_DIR: {rgb_dir}")
        for path in sorted(rgb_dir.iterdir()):
            if path.suffix.lower() not in IMAGE_SUFFIXES or not path.is_file():
                continue
            rows.append(
                {
                    "image_path": str(path.relative_to(REPO_ROOT)),
                    "image_sha256": sha256_file(path),
                    "capture_session": session,
                    "paper_condition": condition,
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, str]]) -> str:
    fields = ("image_path", "image_sha256", "capture_session", "paper_condition")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return sha256_file(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=PREFERRED_TARGET)
    args = parser.parse_args()

    eval_shas, eval_sessions = _eval_identity()
    print(f"PAPER_EVAL identity: {len(eval_shas)} image sha256, "
          f"{len(eval_sessions)} session names")

    split_assignment = json.loads(
        (SPLIT_LOCK / "split_assignment.json").read_text()
    )
    expected_frames = {
        "daytime": split_assignment["outside"]["pl_pool"]["n_frames"],
        "nighttime": split_assignment["night"]["pl_pool"]["n_frames"],
    }

    audit: dict[str, dict] = {}
    selected: dict[str, list[dict[str, str]]] = {}
    for condition in ("daytime", "nighttime"):
        rows = _pool_rows(condition)
        # 세션 목록을 옮겨 적다가 틀리거나 depth 를 섞으면 여기서 걸린다.
        if len(rows) != expected_frames[condition]:
            raise SystemExit(
                f"POOL_SIZE_DISAGREES_WITH_SPLIT_LOCK: {condition} "
                f"found={len(rows)} split_lock={expected_frames[condition]}"
            )
        pool_sessions = sorted({row["capture_session"] for row in rows})
        session_overlap = sorted(set(pool_sessions) & eval_sessions)
        leaked = [row for row in rows if row["image_sha256"] in eval_shas]
        eligible = [row for row in rows if row["image_sha256"] not in eval_shas]
        # 같은 내용의 프레임이 pool 안에서 중복되면 학습 노출이 왜곡된다.
        unique: dict[str, dict[str, str]] = {}
        for row in eligible:
            unique.setdefault(row["image_sha256"], row)
        eligible_unique = sorted(unique.values(), key=lambda row: row["image_sha256"])
        audit[condition] = {
            "sessions": pool_sessions,
            "images_found": len(rows),
            "duplicate_sha_dropped": len(eligible) - len(eligible_unique),
            "eval_sha_overlap": len(leaked),
            "eval_session_overlap": session_overlap,
            "eligible_unique": len(eligible_unique),
        }
        selected[condition] = eligible_unique
        print(f"{condition:10} found={len(rows):<6} eligible_unique={len(eligible_unique):<6} "
              f"sha_overlap={len(leaked)} session_overlap={session_overlap}")

    n_common = min(args.target,
                   audit["daytime"]["eligible_unique"],
                   audit["nighttime"]["eligible_unique"])
    print(f"\nN_COMMON = {n_common}")

    manifests: dict[str, str] = {}
    balanced: list[dict[str, str]] = []
    for condition, filename in (("daytime", "DAYTIME_UNLABELED.csv"),
                                ("nighttime", "NIGHTTIME_UNLABELED.csv")):
        chosen = selected[condition][:n_common]
        manifests[filename] = _write_csv(OUT_DIR / filename, chosen)
        balanced.extend(chosen)
        print(f"  wrote {filename}  N={len(chosen)}")
    manifests["MAIN_UNLABELED_BALANCED.csv"] = _write_csv(
        OUT_DIR / "MAIN_UNLABELED_BALANCED.csv", balanced
    )
    print(f"  wrote MAIN_UNLABELED_BALANCED.csv  N={len(balanced)}")

    lock = {
        "schema_version": "paper_adaptation_pool_lock_v1",
        "purpose": (
            "Leakage-free unlabeled adaptation pool for the MAIN one-model "
            "self-training track.  Frozen before any pseudo-label is produced."
        ),
        "source_of_truth": {
            "session_split": "metric_split_lock.md §1.6 pl_pool",
            "split_assignment": "data/pallet/eval_results/split_lock/split_assignment.json",
            "frame_count_invariant": expected_frames,
            "note": (
                "pl_pool already satisfies pool∩filter-val=0 and pool∩final-test=0. "
                "This lock re-verifies against the *current* PAPER_EVAL membership, "
                "which grew after that split was frozen."
            ),
        },
        "leakage_gate": {
            "adapt_session_intersect_eval_session": sorted(
                set(audit["daytime"]["eval_session_overlap"])
                | set(audit["nighttime"]["eval_session_overlap"])
            ),
            "adapt_sha_intersect_eval_sha": (
                audit["daytime"]["eval_sha_overlap"]
                + audit["nighttime"]["eval_sha_overlap"]
            ),
            "passed": (
                not audit["daytime"]["eval_session_overlap"]
                and not audit["nighttime"]["eval_session_overlap"]
                and audit["daytime"]["eval_sha_overlap"] == 0
                and audit["nighttime"]["eval_sha_overlap"] == 0
            ),
        },
        "sampling": {
            "rule": "sha256 ascending, first N_COMMON",
            "deterministic": True,
            "balanced": True,
        },
        "N_COMMON": n_common,
        "preferred_target": args.target,
        "POOL_BELOW_PREFERRED": n_common < args.target,
        "M2_MAIN_POOL_READY": n_common >= MAIN_POOL_MINIMUM,
        "condition_audit": audit,
        "manifest_sha256": manifests,
        "U_MAIN": len(balanced),
    }
    (OUT_DIR / "ADAPTATION_POOL_LOCK.json").write_text(
        json.dumps(lock, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"\nleakage gate passed = {lock['leakage_gate']['passed']}")
    print(f"U_MAIN = {lock['U_MAIN']}  M2_MAIN_POOL_READY = {lock['M2_MAIN_POOL_READY']}")
    return 0 if lock["leakage_gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
