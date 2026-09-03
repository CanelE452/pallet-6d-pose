"""이름이 다른 세션이 실제로는 같은 원본 recording 인지 복원한다.  읽기 전용.

    python3 scripts/self_training_yolo/site_audit/build_source_recording_groups.py \
        --output-dir data/pallet/results/site_environment_audit_v1

출력  SOURCE_RECORDING_GROUPS.json · IMAGE_SHA_INDEX.json

세션 이름이 다르다는 이유만으로 독립 recording 으로 취급하지 않는다(§3).
이미지 **내용 SHA256** 을 기준으로 세션 간 포함·중첩 관계를 계산하고,
provenance 필드(promoted_from_sessions 등)를 별도 근거로 함께 기록한다.

동일성 판정은 파일 이름이 아니라 내용이다 — 같은 원본을 다른 이름으로 복사해 둔
경우가 이 저장소에 실제로 있다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = REPO_ROOT / "data/pallet/results/site_environment_audit_v1"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}

# 포함 관계로 볼 최소 비율.  아래 두 값은 결과를 보기 전에 정한다.
CONTAINMENT_STRONG = 0.99   # 사실상 부분집합
CONTAINMENT_PARTIAL = 0.10  # 무시할 수 없는 중첩 (보고용, 병합에는 쓰지 않는다)
# 수집 폴더 판정: 품고 있는 세션들이 자기 자신을 이만큼 덮으면 촬영이 아니라 모음이다
COLLECTION_COVERAGE = 0.90


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_one(argument):
    path_str, key = argument
    try:
        return key, sha256_file(Path(path_str))
    except OSError:
        return key, None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--rehash", action="store_true",
                        help="기존 SHA 인덱스를 무시하고 다시 해싱")
    args = parser.parse_args()
    out_dir = Path(args.output_dir).resolve()

    inventory = json.loads((out_dir / "SESSION_INVENTORY.json").read_text())
    sessions = inventory["sessions"]

    # ── 1. 모든 세션 이미지의 내용 SHA (이미 만들어 둔 인덱스가 있으면 재사용)
    keys = [s["session_key"] for s in sessions]
    cache_path = out_dir / "IMAGE_SHA_INDEX.json"
    cached = None
    if cache_path.exists() and not args.rehash:
        payload = json.loads(cache_path.read_text())["sessions"]
        if set(payload) == set(keys):
            cached = payload
            print(f"reusing {cache_path.name}")

    if cached is not None:
        session_shas = [set(cached[key]) for key in keys]
    else:
        jobs, owner = [], []
        for index, session in enumerate(sessions):
            image_dir = REPO_ROOT / session["image_dir"]
            for path in sorted(image_dir.iterdir()):
                if path.suffix.lower() in IMAGE_SUFFIXES:
                    jobs.append((str(path), len(owner)))
                    owner.append(index)

        print(f"hashing {len(jobs)} images from {len(sessions)} sessions "
              f"({args.workers} workers)")
        digests: list[str | None] = [None] * len(jobs)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for done, (key, digest) in enumerate(
                    pool.map(hash_one, jobs, chunksize=64), start=1):
                digests[key] = digest
                if done % 10000 == 0:
                    print(f"  {done}/{len(jobs)}", flush=True)

        session_shas = [set() for _ in sessions]
        for key, digest in enumerate(digests):
            if digest:
                session_shas[owner[key]].add(digest)

    # ── 1b. 수집(collection) 폴더 탐지
    # `_outside_all` 처럼 서로 다른 촬영을 한데 모아 둔 폴더는 recording 이 아니다.
    # 이런 폴더를 그대로 두면 union-find 가 그것을 다리 삼아 별개 촬영을 합쳐 버린다.
    # 판정 규칙(결과 보기 전 고정): 어떤 세션 X 안에 서로 겹치지 않는 다른 세션이
    # 두 개 이상 들어 있으면 X 는 collection 이다.
    contained_in: dict[int, list[int]] = defaultdict(list)
    for a in range(len(sessions)):
        for b in range(len(sessions)):
            if a == b or not session_shas[a]:
                continue
            if session_shas[a] <= session_shas[b]:
                contained_in[b].append(a)

    # "서로 겹치지 않는 부분집합 두 개를 품었다" 만으로는 부족하다 — 하나의 연속
    # 촬영에서 서로 다른 두 묶음을 골라 어노테이션했을 때도 그 조건이 성립한다.
    # 수집 폴더의 진짜 특징은 **품고 있는 세션들이 자기 자신을 거의 다 덮는다** 는
    # 것이다.  `_outside_all` 은 하위 촬영들의 합집합이 100% 를 덮지만,
    # 29k 짜리 미라벨 pool 은 거기서 뽑은 어노 68 장이 0.2% 밖에 못 덮는다.
    is_collection = [False] * len(sessions)
    collection_detail = {}
    for outer, inners in contained_in.items():
        covered: set[str] = set()
        for inner in inners:
            covered |= session_shas[inner]
        coverage = len(covered) / len(session_shas[outer]) if session_shas[outer] else 0.0
        disjoint_pair = any(
            not (session_shas[x] & session_shas[y])
            for i, x in enumerate(inners) for y in inners[i + 1:])
        if disjoint_pair and coverage >= COLLECTION_COVERAGE:
            is_collection[outer] = True
            collection_detail[sessions[outer]["session_key"]] = {
                "coverage_by_contained_sessions": round(coverage, 4),
                "n_contained": len(inners),
            }
    print(f"collection 폴더 {sum(is_collection)} 개 — recording 병합에서 제외")

    # ── 2. 내용 기준 포함 · 중첩 관계
    relations = []
    for a in range(len(sessions)):
        for b in range(a + 1, len(sessions)):
            shared = session_shas[a] & session_shas[b]
            if not shared:
                continue
            small, large = sorted((len(session_shas[a]), len(session_shas[b])))
            relations.append({
                "session_a": sessions[a]["session_key"],
                "session_b": sessions[b]["session_key"],
                "n_a": len(session_shas[a]),
                "n_b": len(session_shas[b]),
                "shared_images": len(shared),
                "containment_of_smaller": len(shared) / small if small else 0.0,
                "jaccard": len(shared) / len(session_shas[a] | session_shas[b]),
            })
    relations.sort(key=lambda r: -r["shared_images"])

    # ── 3. 겹치는 세션을 하나의 원본 recording 으로 묶는다 (union-find)
    parent = list(range(len(sessions)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[max(rx, ry)] = min(rx, ry)

    index_of = {s["session_key"]: i for i, s in enumerate(sessions)}
    for relation in relations:
        a, b = index_of[relation["session_a"]], index_of[relation["session_b"]]
        if is_collection[a] or is_collection[b]:
            continue          # 수집 폴더를 다리 삼아 별개 촬영을 합치지 않는다
        # 부분 중첩으로는 합치지 않는다.  여러 촬영에서 골라 만든 eval 셋이
        # 서로 다른 recording 을 이어붙이는 것을 막기 위해서다 — 그런 중첩은
        # partial_overlap_pairs 에 그대로 보고되고 누수 게이트가 따로 본다.
        if relation["containment_of_smaller"] >= CONTAINMENT_STRONG:
            union(a, b)

    grouped: dict[int, list[int]] = defaultdict(list)
    for index in range(len(sessions)):
        grouped[find(index)].append(index)

    groups = []
    for order, (root, members) in enumerate(
            sorted(grouped.items(),
                   key=lambda kv: -sum(len(session_shas[i]) for i in kv[1])), start=1):
        union_shas: set[str] = set()
        for index in members:
            union_shas |= session_shas[index]
        # 그룹 대표는 가장 프레임이 많은 세션
        members.sort(key=lambda i: -len(session_shas[i]))
        groups.append({
            "recording_id": f"REC_{order:03d}",
            "is_collection": all(is_collection[i] for i in members),
            "representative": sessions[members[0]]["session_key"],
            "n_sessions": len(members),
            "n_unique_images": len(union_shas),
            "union_sha256": hashlib.sha256(
                "\n".join(sorted(union_shas)).encode()).hexdigest(),
            "sessions": [{
                "session_key": sessions[i]["session_key"],
                "root": sessions[i]["root"],
                "frame_count": sessions[i]["frame_count"],
                "unique_images": len(session_shas[i]),
                "resolution": sessions[i]["resolution"],
                "timestamp_min": sessions[i]["timestamp_min"],
                "timestamp_max": sessions[i]["timestamp_max"],
                "provenance": sessions[i]["provenance"],
                "is_collection": is_collection[i],
            } for i in members],
        })

    report = {
        "schema_version": "source_recording_groups_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "method": "image content SHA256 overlap, then union-find",
        "identity_is_content_not_filename": True,
        "containment_thresholds": {
            "strong_subset": CONTAINMENT_STRONG,
            "partial_overlap_merges": CONTAINMENT_PARTIAL,
            "declared_before_results": True,
        },
        "collection_folders": [sessions[i]["session_key"]
                               for i in range(len(sessions)) if is_collection[i]],
        "collection_rule": ("a session is an aggregate, not a recording, when the "
                            "sessions it contains are mutually disjoint AND together "
                            "cover at least 90% of it; an aggregate never bridges a "
                            "merge. A long unlabelled pool that merely had two small "
                            "disjoint subsets annotated out of it is NOT an aggregate."),
        "collection_detail": collection_detail,
        "total_sessions": len(sessions),
        "total_source_recordings": len(groups),
        "multi_session_recordings": sum(1 for g in groups if g["n_sessions"] > 1),
        "overlapping_pairs": len(relations),
        "strong_subset_pairs": [
            r for r in relations
            if r["containment_of_smaller"] >= CONTAINMENT_STRONG],
        "partial_overlap_pairs": [
            r for r in relations
            if CONTAINMENT_PARTIAL <= r["containment_of_smaller"] < CONTAINMENT_STRONG],
        "groups": groups,
    }
    (out_dir / "SOURCE_RECORDING_GROUPS.json").write_text(
        json.dumps(report, indent=2) + "\n")

    # 누수 게이트가 쓸 세션별 SHA 집합 (재사용했으면 다시 쓰지 않는다)
    if cached is None:
        cache_path.write_text(json.dumps({
            "schema_version": "image_sha_index_v1",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "sessions": {sessions[i]["session_key"]: sorted(session_shas[i])
                         for i in range(len(sessions))},
        }, indent=2) + "\n")

    print(f"\nsessions {len(sessions)}  ->  source recordings {len(groups)}")
    print(f"여러 세션이 한 recording 인 경우 {report['multi_session_recordings']}")
    print(f"내용이 겹치는 세션 쌍 {len(relations)}  "
          f"(그중 사실상 부분집합 {len(report['strong_subset_pairs'])})")
    print(f"\n{'recording':11}{'sessions':>9}{'images':>9}  representative")
    print("-" * 88)
    for group in groups[:20]:
        print(f"{group['recording_id']:11}{group['n_sessions']:9d}"
              f"{group['n_unique_images']:9d}  {group['representative'][:52]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
