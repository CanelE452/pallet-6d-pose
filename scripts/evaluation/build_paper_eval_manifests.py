"""PAPER_EVAL population manifest 를 evaluation workspace 에서 재계산해 쓴다.

paper-facing source of truth 는 `PAPER_EVAL` 이다.  숫자를 문서에서 복붙하지 않고
`eval_workspace.evaluation_population_views` 가 계산한 membership 을 그대로
`pallet_pose_population_manifest_v1` 형식으로 내보낸다.  그래야 M1/M2/M3/M5 가
전부 같은 population 을 본다.

기존 `FINAL_EVAL` 계열은 frozen DEV alias(173행)이므로 논문 표에 쓰지 않는다.
여기서 만드는 세 population 이 논문 표의 분모다.

    PAPER_EVAL_PLASTIC_POS   plastic  DEV
    PAPER_EVAL_WOOD_POS      wood     CROSS_SHAPE_DEV
    PAPER_EVAL_ALL_POS       union    DEV   (plastic 먼저, 그 다음 wood)

negative 는 이미 등록된 `DEV_NEG2689` 를 그대로 쓴다 — 새로 만들지 않는다.

사용:
    python scripts/evaluation/build_paper_eval_manifests.py [--check]

`--check` 는 파일을 쓰지 않고 현재 manifest 가 workspace 와 일치하는지만 본다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "challenge"))

from evaluation.eval_workspace import (  # noqa: E402
    atomic_write_json,
    evaluation_population_views,
    load_frames,
)
from evaluation_v2.real_dataset_contract import (  # noqa: E402
    MANIFEST_DIR,
    MANIFEST_SCHEMA_VERSION,
    PLASTIC_OBJECT_TYPE,
    WOOD_OBJECT_TYPE,
    ManifestItem,
    membership_sha256,
)

WORKSPACE = REPO_ROOT / "data" / "evaluation" / "pallet_eval_v1"
WORKSPACE_REL = "data/evaluation/pallet_eval_v1"

PLASTIC_POPULATION = "PAPER_EVAL_PLASTIC_POS"
WOOD_POPULATION = "PAPER_EVAL_WOOD_POS"
ALL_POPULATION = "PAPER_EVAL_ALL_POS"

# workspace 의 object_type 은 registry alias 다.  dispatch 는 registry 정식 이름으로만
# 한다 — 파일명/세션명에서 유추하지 않는다.
OBJECT_TYPE_BY_ALIAS = {
    "plastic": PLASTIC_OBJECT_TYPE,
    "wood": WOOD_OBJECT_TYPE,
}

# legacy manifest 가 쓰는 domain 표기와 같은 어휘를 유지한다.
DOMAIN_BY_LIGHTING = {"day": "DAY", "night": "NIGHT"}


def _repo_relative(workspace_path: str) -> str:
    return f"{WORKSPACE_REL}/{workspace_path}"


def _item(row: dict[str, str], source_population: str, role: str) -> ManifestItem:
    object_type = OBJECT_TYPE_BY_ALIAS[row["object_type"]]
    session_id = row["session_id"]
    stem = Path(row["annotation_path"]).stem
    # wood 는 contract 가 session 한정 frame_id 를 요구한다.  plastic 도 같은 형식으로
    # 맞춰 두면 세션 간 stem 충돌이 구조적으로 불가능해진다.
    frame_id = f"{session_id}:{stem}"
    domain = DOMAIN_BY_LIGHTING.get(row.get("lighting", ""))
    return ManifestItem(
        frame_id=frame_id,
        image=_repo_relative(row["image_path"]),
        label=_repo_relative(row["annotation_path"]),
        source_set=session_id,
        domain=domain,
        object_type=object_type,
        session_id=session_id,
        population_role=role,
        source_population=source_population,
    )


def _session_counts(items: list[ManifestItem]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        counts[item.session_id or ""] = counts.get(item.session_id or "", 0) + 1
    return dict(sorted(counts.items()))


def _domain_counts(items: list[ManifestItem]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        counts[item.domain or "UNTAGGED"] = counts.get(item.domain or "UNTAGGED", 0) + 1
    return dict(sorted(counts.items()))


def _manifest(
    population_id: str,
    items: list[ManifestItem],
    *,
    role: str,
    object_types: list[str],
    provenance: dict,
) -> dict:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "population_id": population_id,
        "object_types": object_types,
        "kind": "POSITIVE",
        "role": role,
        "membership_status": "AVAILABLE",
        "frozen": True,
        "expected_count": len(items),
        "membership_sha256": membership_sha256(items),
        "provenance": provenance,
        "items": [item.canonical_record() for item in items],
    }


def build() -> dict[str, dict]:
    frames = load_frames(WORKSPACE)
    positive = evaluation_population_views(frames)["PAPER_EVAL_POSITIVE"]

    plastic = [
        _item(row, PLASTIC_POPULATION, "DEV")
        for row in positive
        if row["object_type"] == "plastic"
    ]
    wood = [
        _item(row, WOOD_POPULATION, "CROSS_SHAPE_DEV")
        for row in positive
        if row["object_type"] == "wood"
    ]
    unexpected = {row["object_type"] for row in positive} - set(OBJECT_TYPE_BY_ALIAS)
    if unexpected:
        raise SystemExit(f"UNKNOWN_OBJECT_TYPE_IN_PAPER_EVAL: {sorted(unexpected)}")
    if len(plastic) + len(wood) != len(positive):
        raise SystemExit("PAPER_EVAL_PARTITION_LOST_ROWS")

    shared = {
        "derived_from": "PAPER_EVAL_POSITIVE",
        "derivation": (
            "scripts/evaluation/eval_workspace.py::evaluation_population_views; "
            "SHA256-deduplicated union(DEV_EVAL, NEW_EVAL)"
        ),
        "workspace": WORKSPACE_REL,
        "held_out_final": False,
        "builder": "scripts/evaluation/build_paper_eval_manifests.py",
    }

    return {
        PLASTIC_POPULATION: _manifest(
            PLASTIC_POPULATION,
            plastic,
            role="DEV",
            object_types=[PLASTIC_OBJECT_TYPE],
            provenance={
                **shared,
                "session_counts": _session_counts(plastic),
                "domain_counts": _domain_counts(plastic),
            },
        ),
        WOOD_POPULATION: _manifest(
            WOOD_POPULATION,
            wood,
            role="CROSS_SHAPE_DEV",
            object_types=[WOOD_OBJECT_TYPE],
            provenance={
                **shared,
                "session_counts": _session_counts(wood),
                "domain_counts": _domain_counts(wood),
                "symmetry_status": "UNREVIEWED",
                "selector_status": "NOT_RUN",
            },
        ),
        ALL_POPULATION: _manifest(
            ALL_POPULATION,
            plastic + wood,
            role="DEV",
            object_types=[PLASTIC_OBJECT_TYPE, WOOD_OBJECT_TYPE],
            provenance={
                **shared,
                "ordered_union_of": [PLASTIC_POPULATION, WOOD_POPULATION],
                "component_counts": {
                    PLASTIC_POPULATION: len(plastic),
                    WOOD_POPULATION: len(wood),
                },
                "order_contract": "plastic_then_wood",
                "session_counts": _session_counts(plastic + wood),
                "domain_counts": _domain_counts(plastic + wood),
            },
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="쓰지 않고 기존 manifest 와 일치하는지만 확인한다",
    )
    args = parser.parse_args()

    manifests = build()
    stale = []
    for population_id, payload in manifests.items():
        target = MANIFEST_DIR / f"{population_id}.json"
        current = json.loads(target.read_text()) if target.exists() else None
        same = current == payload
        counts = payload["provenance"].get("domain_counts", {})
        print(
            f"{population_id:24} N={payload['expected_count']:<5} "
            f"sha={payload['membership_sha256'][:12]} "
            f"domain={counts} "
            f"{'UNCHANGED' if same else ('STALE' if current else 'NEW')}"
        )
        if same:
            continue
        stale.append(population_id)
        if not args.check:
            atomic_write_json(target, payload)
            print(f"  -> wrote {target.relative_to(REPO_ROOT)}")

    if args.check and stale:
        print(f"\nMANIFEST_STALE: {', '.join(stale)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
