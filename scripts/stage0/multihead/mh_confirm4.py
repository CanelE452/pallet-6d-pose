"""PHASE 3 -- contamination audit and D4, a population nothing has selected on.

D3 was built to be untouched and then was read as an evaluation population, so it
can no longer confirm a rule that was written after seeing it.  This asks whether
anything is left, counts it from the manifests rather than from the earlier
notes, and locks a new draw with a SHA before any number from it is looked at.

Contaminated populations, from the manifests and the runners that read them:

    D0_MH_SEEN512    train side, every lambda in the study was chosen on it
    D2_MH_DEV512     dev, read by the scale, theta-only and pose-aware phases
    D3_MH_CONF512    dev, read by the theta-only and pose-aware evaluations
    MH_TRAIN pool    every training run streams from it

D4 is drawn from `MH_DEV` minus D2 minus D3.  Since dev is 17 groups and D4 draws
from the same ones, it is frame-disjoint and not scene-independent, and this says
so rather than implying otherwise.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import mh_data as MD                                             # noqa: E402

OUT = MD.OUT
NAME = "D4_THETA_CONFIRM512"
COUNT = 512
CONFIRM_SEED = MD.SPLIT_SEED + 3          # D2 used +0, D0 +1, D3 +2


def log(message):
    print(message, flush=True)


def stems_of(name):
    path = OUT / f"{name.lower()}_manifest.json"
    if not path.exists():
        return set()
    return set(json.loads(path.read_text())["stems"])


def build():
    rows = MD.load_split()
    dev = [r for r in rows if r["split"] == "MH_DEV"]
    train = [r for r in rows if r["split"] == "MH_TRAIN"]
    used = {name: stems_of(name) for name in
            ("D0_MH_SEEN512", "D2_MH_DEV512", "D3_MH_CONF512")}
    contaminated = set().union(*used.values())

    remaining = [r for r in dev if r["stem"] not in contaminated]
    audit = {
        "dev_total": len(dev), "train_total": len(train),
        "used": {k: len(v) for k, v in used.items()},
        "dev_contaminated": len([r for r in dev
                                 if r["stem"] in contaminated]),
        "dev_untouched": len(remaining),
        "train_note": "every training run streams the whole MH_TRAIN pool, so "
                      "no train frame can serve as a confirmation population",
        "sealed_real_test_touched": False,
    }
    log(f"dev {len(dev)}  contaminated {audit['dev_contaminated']}  "
        f"untouched {len(remaining)}")

    if len(remaining) < COUNT:
        audit["THETA_CONFIRM_POPULATION_UNAVAILABLE"] = True
        (OUT / "population_audit_d4.json").write_text(json.dumps(audit, indent=1))
        log("not enough untouched frames -- D4 not created")
        return audit

    stems = MD._stratified_sample(remaining, COUNT, CONFIRM_SEED)
    chosen = set(stems)
    payload = json.dumps({"n": len(stems), "stems": stems}, indent=1)
    digest = hashlib.sha256(payload.encode()).hexdigest()

    by_stem = {r["stem"]: r for r in rows}

    def shares(subset):
        counts = {}
        for row in subset:
            counts[row["stratum"]] = counts.get(row["stratum"], 0) + 1
        total = max(len(subset), 1)
        return {k: round(v / total, 4) for k, v in sorted(counts.items())}

    audit.update({
        "name": NAME, "n": len(stems), "seed": CONFIRM_SEED,
        "manifest_sha256": digest,
        "overlap": {k: len(chosen & v) for k, v in used.items()},
        "THETA_CONFIRM_POPULATION_UNAVAILABLE": False,
        "FRAME_DISJOINT_ONLY": True,
        "SCENE_INDEPENDENT": False,
        "scene_note": "dev is 17 groups and D4 draws from the same ones",
        "strata_dev": shares(dev),
        "strata_D4": shares([by_stem[s] for s in stems]),
    })
    audit["max_stratum_share_gap_vs_dev"] = round(max(
        abs(audit["strata_D4"].get(k, 0.0) - v)
        for k, v in audit["strata_dev"].items()), 4)

    (OUT / f"{NAME.lower()}_manifest.json").write_text(payload)
    (OUT / "population_audit_d4.json").write_text(json.dumps(audit, indent=1))
    log(f"{NAME}: {len(stems)} frames   overlap {audit['overlap']}")
    log(f"  sha256 {digest}")
    log(f"  max stratum share gap vs dev = "
        f"{audit['max_stratum_share_gap_vs_dev']}")
    log(f"-> {OUT / f'{NAME.lower()}_manifest.json'}")
    return audit


if __name__ == "__main__":
    build()
