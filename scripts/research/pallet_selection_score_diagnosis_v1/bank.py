"""Read-only views of sealed original candidate records, without pose solving.

Every retained sample keeps its original stage, family, index and role.  Source
roles are never collapsed merely because their numerical poses agree.
"""
from __future__ import annotations

from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path

FAMILIES = ("ORIGINAL_WD", "SWAPPED_WD")
STAGES = ("G10", "G13")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def save_rows(path, values):
    with gzip.open(path, "xt", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def group_for_oracle(oracle):
    """Explicit mapping to the old field, with the old fixed coverage scope."""
    if oracle["final_joint_success"]:
        require(oracle["earliest_observed_failure_stage"] is None, "Successful oracle has failure stage")
        return "FINAL_SUCCESS_CONTROL"
    earliest = oracle["earliest_observed_failure_stage"]
    mapping = {
        "family_selection": "FAMILY_SELECTION_LOSS",
        "within_family_final_selection": "WITHIN_FAMILY_SELECTION_LOSS",
        "confirmed_original_consumer_observation_or_proposal_limit": "NO_SUCCESS_CANDIDATE",
    }
    require(earliest in mapping, "Unresolved original oracle classification: " + str(earliest))
    require(oracle["original_consumer_generated_bank_complete"] is True, "Original consumer coverage incomplete")
    return mapping[earliest]


def input_rows(old_root):
    """Hash gates precede opening the original numeric records."""
    old_root = Path(old_root)
    original = old_root / "original_candidate_archive_v1/ORIGINAL_E_CANDIDATE_BANK.jsonl.gz"
    manifest_path = original.parent / "CANDIDATE_MANIFEST.json"
    manifest = read(manifest_path)
    require(sha(original) == manifest["artifacts"][original.name], "Original candidate bank changed")
    endpoint = old_root / "GENERATED_CANDIDATES.jsonl.gz"
    candidate_seal = read(old_root / "CANDIDATE_SEAL.json")
    require(sha(endpoint) == candidate_seal["generated_candidates_sha256"], "Sealed generated endpoints changed")
    oracle_path = old_root / "oracle_v1/ORACLE_ROWS.jsonl.gz"
    oracle_seal = read(oracle_path.parent / "ORACLE_SEAL.json")
    # The seal's artifact mapping is intentionally read according to its schema.
    artifact_hashes = oracle_seal.get("artifacts", oracle_seal.get("files", {}))
    entry = artifact_hashes.get(oracle_path.name)
    expected = entry.get("sha256") if isinstance(entry, dict) else entry
    require(expected is not None and sha(oracle_path) == expected, "Sealed oracle rows changed")
    ids = manifest["ordered_ids"]
    originals, endpoints, oracles = rows(original), rows(endpoint), rows(oracle_path)
    for index, (a, b, c) in enumerate(zip(originals, endpoints, oracles, strict=True)):
        require(index < len(ids) and a["id"] == b["id"] == c["id"] == ids[index], "Exact ordered238 binding")
        require(a["partition"] == c["partition"] and a["session"] == c["session"], "Oracle partition/session differs")
        yield index, a, b["candidates"], c
    require(index + 1 == len(ids) == 238, "Exactly238 complete original rows required")


def retained_record(index, original, stage, family, sample, endpoint_record, oracle):
    order = sample["original_index"]
    cid = f'ORIGINAL_{stage}/{family["name"]}/feasible/{order}'
    successful = oracle["original_stage_oracles"][stage]["feasible"]["joint_success_candidate_ids"]
    record = sample["original_record"]
    return dict(candidate_id=cid, id=original["id"], index=index,
        partition=original["partition"], session=original["session"], stage=stage, family=family["name"],
        source_order=order, role="original_retained_feasible_sample", retained=True,
        original_record=deepcopy(record), original_endpoint_record=deepcopy(endpoint_record),
        R=deepcopy(record["R"]), t=deepcopy(record["t"]), pose=deepcopy(sample["pose"]),
        native_xyz=deepcopy(family["native_xyz"]), physical_registration_Q=deepcopy(family["physical_registration_Q"]),
        cf_extents=deepcopy(family["cf_extents"]), solver_schema=family["solver_schema"],
        accepted_family=family["accepted"], reference_success=cid in successful,
        reference_semantics="Existing sealed geometric-reference joint T<=5cm AND physical/C2 R<=5deg; not semantic-owner truth",
        source_path="original_candidate_archive_v1/ORIGINAL_E_CANDIDATE_BANK.jsonl.gz",
        source_key=f'stages.{stage}.families.{FAMILIES.index(family["name"])}.feasible_samples.{order}')
