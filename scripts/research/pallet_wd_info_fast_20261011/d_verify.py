"""Independent arithmetic, preserved-source and matched-arm verification (no PnP)."""
import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with gzip.open(path, "rt") as handle:
        return [json.loads(line) for line in handle]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quantile(values, fraction):
    ordered = sorted(values); position = (len(ordered)-1)*fraction
    lower = math.floor(position); upper = math.ceil(position)
    return ordered[lower]*(upper-position)+ordered[upper]*(position-lower) if upper!=lower else ordered[lower]


def moments(values):
    mean = math.fsum(values)/len(values)
    variance = math.fsum((value-mean)**2 for value in values)/(len(values)-1)
    return dict(mean=mean, SD=math.sqrt(variance), median=quantile(values, .5),
                P90=quantile(values, .9), P95=quantile(values, .95))


def run(args):
    source = Path(args.source_root); public = Path(args.public_dir); private = Path(args.private_dir)
    contract = read(public/"D_CONTRACT.json"); training = read(public/"D_TRAINING_RECEIPT.json")
    assert contract["trainable_parameters"] == {"D_CONTROL":2305, "D_JOINT_DIM":2305}
    assert training["updates_per_arm"] == {"D_CONTROL":1000, "D_JOINT_DIM":1000}
    assert training["initial_scorer_sha256"] == contract["shared_initial_sha256"]
    assert training["frozen_state_unchanged"] and training["frozen_gradients_absent"]
    for binding in contract["source_bindings"]:
        assert sha(source/binding["path"]) == binding["sha256"]
    for value in training["checkpoints"].values():
        assert sha(private/value["private_namespace"]) == value["sha256"]
    seal = read(public/"D_REPAIRED_POSE_SEAL.json")
    assert seal["cumulative_D_cv_calls"] == 8495
    assert all(check["PASS"] for check in seal["fresh_complete_original_F_checks"])
    allowed = read(public/"INPUT_CONTRACT.json"); ids = set(allowed["allowed_ids"])
    first = {(r["id"], r["method"]):r for r in rows(public/"D_ROWS.jsonl.gz")}
    corrected = rows(public/"D_REPAIRED_ROWS.jsonl.gz")
    assert len(corrected)==1730 and len({(r["id"],r["method"]) for r in corrected})==1730
    for row in corrected:
        assert row["id"] in ids and row["pose"]["available"]
        assert row["corner"] == first[row["id"],row["method"]]["corner"]
    result = read(public/"D_REPAIRED_RESULTS.json")
    comparisons = 0; max_delta = 0
    for population, subset in [("ALLOWED173", ids), ("RECT245_INTERSECTION110", set(allowed["rect245_intersection_ids"]))]:
        by = defaultdict(list)
        for row in corrected:
            if row["id"] in subset:
                by[row["method"]].append(row)
        for name, records in by.items():
            actual = result["aggregates"][population][name]
            assert actual["frames"]==len(records)
            assert actual["success_count"]==sum(r["pose"]["translation_cm"]<5 and r["pose"]["rotation_deg"]<5 for r in records)
            assert actual["wrong_axis"]==sum(r["axis_wrong"] for r in records if r["axis_wrong"] is not None)
            for metric in ("translation_cm", "rotation_deg", "ADDsym_m", "IoU3D"):
                independent = moments([r["pose"][metric] for r in records])
                for statistic, value in independent.items():
                    delta = abs(value-actual[metric][statistic]); max_delta=max(max_delta,delta); comparisons+=1
                    assert delta<=1e-8, (population,name,metric,statistic,delta)
    by = {(r["id"],r["method"]):r for r in corrected}; changes=[]
    for suffix in ("", "_SUBPIX"):
        for fid in sorted(ids):
            a,b=by[fid,"D_CONTROL"+suffix],by[fid,"D_JOINT_DIM"+suffix]
            for corner, valid in enumerate(a["corner"]["canonical_valid"]):
                if not valid: continue
                ae=a["corner"]["canonical_errors"][corner];be=b["corner"]["canonical_errors"][corner]
                if ae<5 and be>10:
                    changes.append(dict(id=fid,route="Stage1" if not suffix else "SubPix",corner=corner,
                                        control_px=ae,joint_px=be,event="good5_to_bad10"))
                if ae>20 and be<10:
                    changes.append(dict(id=fid,route="Stage1" if not suffix else "SubPix",corner=corner,
                                        control_px=ae,joint_px=be,event="bad20_to_good10"))
    with (public/"D_EXTREME_CORNER_TRANSITIONS.csv").open("x", newline="") as handle:
        fields=["id","route","corner","control_px","joint_px","event"]
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(changes)
    receipt=dict(status="PASS", independent_statistic_comparisons=comparisons,max_absolute_moment_delta=max_delta,
                 same_corner_rows_before_after_adapter_repair=True, allowed_frames=173, methods=10,
                 protected_frame_scoring=0, source_hashes_unchanged=True, matched_trainable_parameters=2305,
                 matched_updates=1000, cumulative_cv_calls=8495, verification_new_PnP_calls=0,
                 first_attempt_invalid_adapter_not_method_failure=True,
                 fresh_complete_F_verification_predictions=3, fresh_complete_F_verification_distinct_frames=1,
                 correction_to_repair_lock_wording="Three predictions from three arms on one fixed first frame; not three distinct frames",
                 extreme_corner_transitions=changes)
    with (public/"D_VERIFICATION.json").open("x") as handle:
        json.dump(receipt,handle,indent=2);handle.write("\n")
    print(json.dumps(receipt),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for name in ("source-root","public-dir","private-dir"):
        parser.add_argument("--"+name,required=True)
    run(parser.parse_args())
