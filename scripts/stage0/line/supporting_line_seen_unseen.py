"""Is 6.845 degree a generalization collapse, or did the search2k model underfit?

`5dd6036` measured the epoch-5 checkpoint on held-out frames only, so the number
it produced cannot distinguish the two.  I then divided it by an overfit32 result
and called the ratio a generalization gap -- but those are different training
runs, and the ratio has no meaning until the same checkpoint is measured on
frames it actually trained on.

This screen trains nothing and changes nothing.  It loads the existing
checkpoints and evaluates each on three populations:

```
D0_SEEN512          512 frames the model trained on
D1_TRAIN_UNSEEN512  512 frames from LINE_TRAIN it did not, group-matched to D0
D2_LINE_DEV512      the existing group-disjoint holdout
```

If D0 fails, the model never fit its own training data and nothing about
generalization has been observed.
"""
from __future__ import annotations

import argparse, collections, csv, hashlib, importlib.util, json, pathlib, sys, time
import numpy as np, torch

ROOT = pathlib.Path(__file__).resolve().parents[3]
for _e in ("scripts/stage0", "Deep_Object_Pose/common", "challenge/scripts"):
    if str(ROOT / _e) not in sys.path:
        sys.path.insert(0, str(ROOT / _e))


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CAP = _load("CAP_DIAG", "scripts/stage0/line/supporting_line_map_capacity.py")
H, V2, OUT, DEV = CAP.H, CAP.V2, CAP.OUT, CAP.DEV
SIZE = 512
DATASETS = ("D0_SEEN512", "D1_TRAIN_UNSEEN512", "D2_LINE_DEV512")
PRIMARY_ARM = "M0_F50_SLINE"


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def groups():
    rows = list(csv.DictReader(open(OUT / "line_internal_split.csv")))
    return {row["index"]: row["group_id"] for row in rows}


def proportional_quota(counts, total, size):
    """Deterministic largest-remainder allocation; no RNG, no tie-breaking by
    dict order."""
    ordered = sorted(counts)
    exact = {g: size * counts[g] / total for g in ordered}
    quota = {g: int(exact[g]) for g in ordered}
    short = size - sum(quota.values())
    for g in sorted(ordered, key=lambda k: (-(exact[k] - quota[k]), k))[:short]:
        quota[g] += 1
    return quota


def build_manifests():
    """D0 from the trained frames, D1 from the rest of LINE_TRAIN with the same
    group histogram, so the only thing that changes is whether the model saw the
    frame."""
    group = groups()
    train, _ = V2.split_indices()
    seen = V2.manifest("line_search2k")
    seen_set = set(seen)
    rest = [i for i in train if i not in seen_set]
    counts = collections.Counter(group[i] for i in seen)
    quota = proportional_quota(counts, len(seen), SIZE)

    by_group_seen = collections.defaultdict(list)
    for index in sorted(seen):
        by_group_seen[group[index]].append(index)
    by_group_rest = collections.defaultdict(list)
    for index in sorted(rest):
        by_group_rest[group[index]].append(index)

    d0, d1, shortfall = [], [], {}
    for g in sorted(quota):
        want = quota[g]
        d0.extend(by_group_seen[g][:want])
        available = by_group_rest[g][:want]
        d1.extend(available)
        if len(available) < want:
            shortfall[g] = want - len(available)
    return sorted(d0), sorted(d1), quota, shortfall, group


def manifest_record(name, indices, group):
    histogram = collections.Counter(group[i] for i in indices)
    return {"name": name, "frames": len(indices), "groups": len(histogram),
            "group_histogram": dict(sorted(histogram.items())),
            "sha": hashlib.sha256(repr(indices).encode()).hexdigest()[:16]}


def write_manifests():
    d0, d1, quota, shortfall, group = build_manifests()
    dev = V2.manifest("line_dev512")
    record = {"D0_SEEN512": manifest_record("D0_SEEN512", d0, group),
              "D1_TRAIN_UNSEEN512": manifest_record("D1_TRAIN_UNSEEN512", d1, group),
              "D2_LINE_DEV512": manifest_record("D2_LINE_DEV512", sorted(dev), group),
              "group_quota": quota, "shortfall": shortfall,
              "overlap": {"D0_D1": len(set(d0) & set(d1)),
                          "D0_DEV": len(set(d0) & set(dev)),
                          "D1_DEV": len(set(d1) & set(dev))},
              **CAP.provenance()}
    for key in ("D0_D1", "D0_DEV", "D1_DEV"):
        if record["overlap"][key]:
            raise RuntimeError(f"OVERLAP_NOT_ZERO: {key}")
    if record["D0_SEEN512"]["frames"] != SIZE:
        raise RuntimeError(f"D0 size {record['D0_SEEN512']['frames']}")
    for name in ("D0_SEEN512", "D1_TRAIN_UNSEEN512"):
        path = OUT / f"{name.lower()}_manifest.csv"
        with open(path, "w", newline="") as handle:
            writer = csv.writer(handle); writer.writerow(["index"])
            writer.writerows([[i] for i in (d0 if name.startswith("D0") else d1)])
    (OUT / "seen_unseen_manifests.json").write_text(json.dumps(record, indent=2))
    return record, d0, d1, sorted(dev)


def read_manifest(name):
    path = OUT / f"{name.lower()}_manifest.csv"
    return [row["index"] for row in csv.DictReader(open(path))]


def evaluate_checkpoints(edges, coarse, xx, yy):
    a1 = V2.load_a1()
    populations = {"D0_SEEN512": read_manifest("D0_SEEN512"),
                   "D1_TRAIN_UNSEEN512": read_manifest("D1_TRAIN_UNSEEN512"),
                   "D2_LINE_DEV512": V2.manifest("line_dev512")}
    report = {"arms": {}, **CAP.provenance()}
    for name in CAP.ARMS:
        entry = {}
        for epoch in CAP.EPOCH_LADDER:
            path = CAP.checkpoint_path(name, f"search2k_epoch{epoch}")
            if not path.exists():
                raise RuntimeError(f"MISSING_CHECKPOINT: {path}")
            state = torch.load(path, map_location=DEV, weights_only=False)
            head, stem, _ = CAP.build_arm(name)
            head.load_state_dict(state["model"])
            if stem is not None:
                stem.load_state_dict(state["stem"])
            for dataset, indices in populations.items():
                result = CAP.evaluate(indices, head, stem, a1, edges, coarse, xx, yy)
                entry[f"epoch{epoch}_{dataset}"] = result
                log(f"  {name} epoch{epoch} {dataset:<18} angle med "
                    f"{result['angle_median']:7.4f} p90 {result['angle_p90']:8.4f} | "
                    f"offset med {result['offset_median']:7.4f} p90 "
                    f"{result['offset_p90']:8.4f}  n={result['n']}  PASS={result['PASS']}")
        for epoch in CAP.EPOCH_LADDER:
            base = entry[f"epoch{epoch}_D0_SEEN512"]
            for dataset in DATASETS[1:]:
                other = entry[f"epoch{epoch}_{dataset}"]
                entry[f"epoch{epoch}_{dataset}_ratio"] = {
                    "angle": other["angle_median"] / max(base["angle_median"], 1e-9),
                    "offset": other["offset_median"] / max(base["offset_median"], 1e-9)}
        report["arms"][name] = entry
    return report


def diagnose(report):
    """Cause taxonomy, decided by the primary arm at epoch 5."""
    entry = report["arms"][PRIMARY_ARM]
    last = max(CAP.EPOCH_LADDER)
    passes = {d: bool(entry[f"epoch{last}_{d}"]["PASS"]) for d in DATASETS}
    if not passes["D0_SEEN512"]:
        cause = "SEARCH2K_MODEL_UNDERFIT_CONFIRMED"
    elif not passes["D1_TRAIN_UNSEEN512"]:
        cause = "WITHIN_LINE_TRAIN_GENERALIZATION_GAP"
    elif not passes["D2_LINE_DEV512"]:
        cause = "APPEARANCE_COMBINATION_GENERALIZATION_GAP"
    else:
        cause = "HARD_BLOCKED_DIAGNOSTIC_INCONSISTENCY"
    return {"primary_arm": PRIMARY_ARM, "epoch": last, "passes": passes,
            "CAUSE": cause}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["manifests", "evaluate"])
    arguments = parser.parse_args()
    import instance_edge_topology as IET
    edges = [tuple(e) for e in IET.build_topology()["edges"]]
    if not V2.sha_file(OUT / "line_internal_split.csv").startswith(V2.LINE_SPLIT_SHA):
        raise RuntimeError("HARD_BLOCKED: LINE split changed")

    if arguments.command == "manifests":
        record, d0, d1, dev = write_manifests()
        for name in DATASETS:
            entry = record[name]
            log(f"[manifests] {name:<18} frames {entry['frames']} "
                f"groups {entry['groups']} sha {entry['sha']}")
        log(f"[manifests] overlap {record['overlap']}  shortfall groups "
            f"{len(record['shortfall'])}")
        return

    coarse, (xx, yy) = H.CoarseRadon(), H.pixel_coordinates()
    report = evaluate_checkpoints(edges, coarse, xx, yy)
    report["diagnosis"] = diagnose(report)
    (OUT / "seen_unseen_diagnostic.json").write_text(json.dumps(report, indent=2,
                                                                default=float))
    log(f"[diagnose] {report['diagnosis']['CAUSE']}  passes "
        f"{report['diagnosis']['passes']}")


if __name__ == "__main__":
    main()
