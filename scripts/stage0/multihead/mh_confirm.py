"""PHASE 2 -- population audit, and a confirmation set that nothing has read yet.

The dev split holds 6,242 frames.  Every solver decision in this study so far has
been read off `D2_MH_DEV512`, which is 512 of them, so 5,730 dev frames have never
been looked at.  That is a real untouched population and it costs nothing to
designate one, so the theta-only method gets a confirmation set rather than a
second look at the set that every previous phase already used.

    D0_MH_SEEN512     512 train frames     calibration, read many times
    D2_MH_DEV512      512 dev frames       evaluation, read by PHASE 4/6/8/9
    D3_MH_CONF512     512 dev frames       new, disjoint from D2, read once

What this does and does not buy:

    it does buy   a population that never influenced the method, so a second
                  reading is not a second chance at the same 512 frames
    it does not   scene independence -- dev is 17 groups and D3 draws from the
      buy         same 17, so D3 is frame-disjoint from D2, not group-disjoint

Sealed/final real test is not touched and is not what this is.
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import mh_data as MD                                             # noqa: E402

OUT = MD.OUT
NAME = "D3_MH_CONF512"
COUNT = 512
# A different draw from the same locked split.  Not SPLIT_SEED and not
# SPLIT_SEED+1, which D2 and D0 already used.
CONFIRM_SEED = MD.SPLIT_SEED + 2


def log(message):
    print(message, flush=True)


def build():
    rows = MD.load_split()
    dev = [r for r in rows if r["split"] == "MH_DEV"]
    train = [r for r in rows if r["split"] == "MH_TRAIN"]
    d2 = set(json.loads((OUT / "d2_mh_dev512_manifest.json").read_text())["stems"])
    d0 = set(json.loads((OUT / "d0_mh_seen512_manifest.json").read_text())["stems"])

    remaining = [r for r in dev if r["stem"] not in d2]
    stems = MD._stratified_sample(remaining, COUNT, CONFIRM_SEED)
    chosen = set(stems)

    audit = {
        "name": NAME, "n": len(stems),
        "source": "MH_DEV minus D2_MH_DEV512",
        "dev_total": len(dev), "train_total": len(train),
        "dev_unused_before": len(remaining),
        "dev_unused_after": len(remaining) - len(stems),
        "seed": CONFIRM_SEED,
        "overlap_with_D2": len(chosen & d2),
        "overlap_with_D0": len(chosen & d0),
        "group_disjoint_from_D2": False,
        "group_note": "dev is 17 groups; D3 draws from the same groups, so it is "
                      "frame-disjoint from D2 but not scene-independent",
        "sealed_real_test_touched": False,
    }

    def shares(subset):
        counts = {}
        for row in subset:
            counts[row["stratum"]] = counts.get(row["stratum"], 0) + 1
        total = max(len(subset), 1)
        return {k: round(v / total, 4) for k, v in sorted(counts.items())}

    by_stem = {r["stem"]: r for r in rows}
    audit["strata_dev"] = shares(dev)
    audit["strata_D2"] = shares([by_stem[s] for s in sorted(d2)])
    audit["strata_D3"] = shares([by_stem[s] for s in stems])
    audit["max_stratum_share_gap_vs_dev"] = round(max(
        abs(audit["strata_D3"].get(k, 0.0) - v)
        for k, v in audit["strata_dev"].items()), 4)

    (OUT / f"{NAME.lower()}_manifest.json").write_text(
        json.dumps({"n": len(stems), "stems": stems}, indent=1))
    (OUT / "population_audit.json").write_text(json.dumps(audit, indent=1))

    log(f"{NAME}: {len(stems)} frames from {len(remaining)} unused dev frames")
    log(f"  overlap with D2 = {audit['overlap_with_D2']}, "
        f"with D0 = {audit['overlap_with_D0']}")
    log(f"  max stratum share gap vs dev = "
        f"{audit['max_stratum_share_gap_vs_dev']}")
    for key in ("strata_dev", "strata_D2", "strata_D3"):
        log(f"  {key:<12} {audit[key]}")
    log(f"-> {OUT / f'{NAME.lower()}_manifest.json'}")
    return audit


if __name__ == "__main__":
    build()
