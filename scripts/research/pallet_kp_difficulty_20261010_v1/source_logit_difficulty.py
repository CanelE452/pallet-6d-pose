#!/usr/bin/env python3
"""Pure public logit analysis by supplied source target class and predicted role.

Recomputes probabilities from saved float32 logits. Conditional entropy excludes
none and normalizes the65 spatial candidates. Source labels are preserved as
supplied; this tool does not establish that their physical supervision is valid.
No model/ray/image/pose evaluation or threshold sweep is executed.
"""
import argparse
import base64
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct
import time
import zlib


EXPERIMENT = "pallet_kp_difficulty_20261010_v1"
ARMS = ("GEOMETRY_ONLY", "IMAGE_NO_ROLE", "IMAGE_ROLE")
ROLES = ("BOUNDARY", "INTERNAL", "UNAVAILABLE")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def distribution(values):
    a = sorted(values)
    n = len(a)
    def quantile(p):
        x = (n - 1) * p
        i = math.floor(x)
        return a[i] + (a[min(i + 1, n - 1)] - a[i]) * (x - i)
    mean = math.fsum(a) / n if n else None
    var = math.fsum((v - mean) ** 2 for v in a) / (n - 1) if n > 1 else None
    return dict(n=n, mean=mean, sample_variance=var, sample_std=math.sqrt(var) if var is not None else None,
                min=a[0] if n else None, max=a[-1] if n else None,
                **{"P" + str(int(p * 100)).zfill(2): quantile(p) if n else None
                   for p in (.01, .1, .25, .5, .75, .9, .99)})


def probability_fields(scores, lo, hi, weight, label):
    peak = max(scores)
    exp = [math.exp(v - peak) for v in scores]
    total = math.fsum(exp)
    spatial_total = math.fsum(exp[:65])
    joint = [v / total for v in exp]
    conditional = [v / spatial_total for v in exp[:65]]
    pmatch = spatial_total / total
    entropy = -math.fsum(p * math.log(p) for p in conditional if p > 0)
    result = dict(Pmatch=pmatch, conditional_candidate_entropy_nats=entropy,
                  conditional_candidate_entropy_bits=entropy / math.log(2),
                  conditional_effective_candidates=math.exp(entropy),
                  maximum_conditional_candidate_probability=max(conditional),
                  none_probability=joint[65])
    if label == "POSITIVE":
        bins = {lo, hi}
        result.update(two_target_bin_joint_probability=math.fsum(joint[b] for b in bins),
                      two_target_bin_conditional_match_probability=math.fsum(conditional[b] for b in bins),
                      supplied_weighted_target_joint_probability=(1 - weight) * joint[lo] + weight * joint[hi],
                      supplied_weighted_target_conditional_match_probability=(1 - weight) * conditional[lo] + weight * conditional[hi])
    return result


def compute(doc):
    ceiling_path = doc / "SOURCE_CEILING_ROWS.jsonl.gz"
    logits_path = doc / "SOURCE_MATCH_MASS_LOGITS.jsonl.gz"
    ceiling = {r["index"]: r for r in rows(ceiling_path)}
    assert len(ceiling) == 1024
    grouped = defaultdict(lambda: defaultdict(list))
    decisions, identities = defaultdict(Counter), set()
    pmatch_difference = 0.
    replay_queries = replay_logits = positive_queries = 0
    for row in rows(logits_path):
        arm, partition, index = row["arm"], row["partition"], row["index"]
        key = (arm, partition, index)
        assert arm in ARMS and partition in ("calibration", "source_test") and key not in identities
        identities.add(key)
        truth = ceiling[index]
        assert row["id"] == truth["id"] and row["family"] == truth["family"] and partition == truth["partition"]
        target_semantic_sha = hashlib.sha256(json.dumps(truth, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        assert row["source_ceiling_row"] == dict(file=ceiling_path.name, line=index + 1, sha256=target_semantic_sha)
        raw = zlib.decompress(base64.b64decode(row["logits_f32_zlib_base64"], validate=True))
        assert row["logits_shape"] == [84, 66] and len(raw) == 84 * 66 * 4
        assert hashlib.sha256(raw).hexdigest() == row["logits_uncompressed_sha256"]
        logits = struct.unpack("<" + "f" * (84 * 66), raw)
        assert all(math.isfinite(v) for v in logits)
        target = truth["targets"]
        assert all(len(target[k]) == 84 for k in ("lo", "hi", "weight", "valid"))
        assert len(truth["predicted_role_query_ids"]) == 84
        for i in range(84):
            lo, hi, weight, valid = (target[k][i] for k in ("lo", "hi", "weight", "valid"))
            label = "IGNORE" if not valid else "NONE" if lo == 65 else "POSITIVE"
            if label == "POSITIVE":
                assert 0 <= lo <= hi < 65 and 0 <= weight <= 1
                positive_queries += 1
            else:
                assert lo == hi == 65 and weight == 0
            role = ROLES[truth["predicted_role_query_ids"][i]]
            scores = logits[i * 66:(i + 1) * 66]
            value = probability_fields(scores, lo, hi, weight, label)
            difference = abs(value["Pmatch"] - row["Pmatch"][i])
            pmatch_difference = max(pmatch_difference, difference)
            assert difference <= 1e-12
            group = (partition, arm, label, role)
            for name, v in value.items():
                grouped[group][name].append(v)
            old_choice = max(range(66), key=scores.__getitem__)
            mass_choice = max(range(65), key=scores.__getitem__) if value["Pmatch"] > .5 else 65
            assert old_choice == row["decoders"]["ORIGINAL_66_CLASS_MAP"]["choices"][i]
            assert mass_choice == row["decoders"]["FIXED_BINARY_EXISTENCE_MAP"]["choices"][i]
            decisions[group].update(dict(queries=1, old_66way_match=int(old_choice < 65),
                                         fixed_mass_match=int(mass_choice < 65),
                                         old_none_mass_match=int(old_choice == 65 and mass_choice < 65)))
            replay_queries += 1
            replay_logits += 66
    expected = {(arm, partition, index) for arm in ARMS for index, r in ceiling.items()
                if (partition := r["partition"]) in ("calibration", "source_test")}
    assert identities == expected and len(identities) == 768
    groups = [dict(partition=partition, arm=arm, supplied_label=label, predicted_role=role,
                   counts=dict(decisions[key]), distributions={metric: distribution(values) for metric, values in measures.items()})
              for key, measures in sorted(grouped.items()) for partition, arm, label, role in (key,)]
    return dict(groups=groups, source_logit_rows=len(identities), source_target_rows=len(ceiling),
                probability_query_replays=replay_queries, float32_logit_values=replay_logits,
                positive_query_target_probability_replays=positive_queries,
                stored_Pmatch_max_difference=pmatch_difference)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3], help="repository root")
    parser.add_argument("--output", type=Path, help="new JSON path; completed files cannot be overwritten")
    args = parser.parse_args()
    repo = args.root.resolve()
    doc = repo / "_docs" / "experiments" / EXPERIMENT
    output = (args.output or doc / "SOURCE_LOGIT_DIFFICULTY.json").resolve()
    pending = output.with_suffix(output.suffix + ".pending")
    if output.exists() or pending.exists():
        raise FileExistsError("Preserve completed/pending source logit analysis")
    start = time.monotonic()
    paths = [doc / name for name in ("SOURCE_CEILING_ROWS.jsonl.gz", "SOURCE_MATCH_MASS_LOGITS.jsonl.gz")]
    bindings = [dict(path=str(p.relative_to(repo)), bytes=p.stat().st_size, sha256=sha(p)) for p in paths]
    analysis = compute(doc)
    assert all(sha(p) == binding["sha256"] for p, binding in zip(paths, bindings)), "Input changed during analysis"
    result = dict(schema="source_logit_difficulty_v1", complete=True, passed=True,
                  generated_utc=datetime.now(timezone.utc).isoformat(), **analysis,
                  definitions=dict(Pmatch="sum65 softmax probabilities; none is the66th category",
                                   candidate_entropy="Shannon entropy over65 spatial probabilities conditional on match; natural log and bits",
                                   effective_candidates="exp(conditional entropy in nats)",
                                   two_target_bin_probability="sum probability at distinct supplied lo/hi target bins; sum once if lo==hi",
                                   supplied_weighted_target_probability="(1-weight)*P(lo)+weight*P(hi), not the unweighted interval mass",
                                   joint="66-category softmax", conditional="65-category softmax conditioned on match",
                                   quantiles="linear interpolation at (n-1)*p; all supplied queries in each group; no acceptance conditioning",
                                   decisions="original66-way MAP and fixed summed65-mass>none; no threshold sweep"),
                  input_bindings=bindings,
                  implementation=dict(path=str(Path(__file__).resolve().relative_to(repo)), sha256=sha(Path(__file__).resolve()), dependencies="Python>=3.9 standard library"),
                  execution=dict(new_detector_forwards=0, new_head_forwards=0, new_training_updates=0,
                                 new_rays=0, new_pose_solves=0, image_reads=0, private_GT_reads=0,
                                 mathematical_probability_query_replays=analysis["probability_query_replays"],
                                 supplied_positive_two_bin_replays=analysis["positive_query_target_probability_replays"],
                                 old_and_fixed_decision_replays=2 * analysis["probability_query_replays"],
                                 wall_seconds_through_analysis=time.monotonic() - start),
                  limits=["POSITIVE/NONE/IGNORE are the existing supplied source labels, not a new physical label validation.",
                          "IGNORE probabilities are descriptive only; their acceptance is not counted as a valid correspondence.",
                          "Candidate entropy measures spread, not correctness, calibration or independent modes.",
                          "No locations, targets, logits, checkpoints or solver outputs are changed.",
                          "Partitions are reported separately; no model reselection or confidence threshold tuning occurs."])
    output.parent.mkdir(parents=True, exist_ok=True)
    pending.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    pending.replace(output)
    print(json.dumps(dict(passed=True, rows=analysis["source_logit_rows"], query_replays=analysis["probability_query_replays"],
                          groups=len(analysis["groups"]), pmatch_max_difference=analysis["stored_Pmatch_max_difference"])))


if __name__ == "__main__":
    main()
