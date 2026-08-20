"""L5 chance-level normalisation and the L4 pair finding.

A raw "6,037 near-duplicate pairs" is unreadable without the rate the hash
produces on unrelated images.  Measured here: random BROAD pairs land at median
Hamming 31/64 and only 0.025% fall within radius 4, so the threshold is tight --
but 4,000 x 2,500 comparisons still yield ~2,500 collisions by chance alone.
Every pair count is therefore reported against its own expectation.
"""
from __future__ import annotations

import json, pathlib, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import da_common as DA     # noqa: E402
import da_leakage as DL    # noqa: E402

CHANCE_PROBE = 1500
CHANCE_DRAWS = 20000
PROBE_SEED = 20260904


def chance_rate():
    src = next(s for s in DA.POSITIVE_SOURCES if s.dataset_id == "BROAD_40K")
    table = DL.sampled_phashes(src, np.random.default_rng(1))
    values = list(table.values())[:CHANCE_PROBE]
    bits = np.unpackbits(
        np.frombuffer(b"".join(values), np.uint8).reshape(len(values), -1), axis=1)
    rng = np.random.default_rng(PROBE_SEED)
    i = rng.integers(0, len(values), CHANCE_DRAWS)
    j = rng.integers(0, len(values), CHANCE_DRAWS)
    keep = i != j
    dist = (bits[i[keep]] != bits[j[keep]]).sum(1)
    return {"median_distance": float(np.median(dist)),
            "p01_distance": float(np.percentile(dist, 1)),
            "rate_within_radius": float(np.mean(dist <= DL.NEAR_DUP_RADIUS))}


def main():
    path = DA.AUDIT / "DUPLICATE_LEAKAGE_AUDIT.json"
    report = json.loads(path.read_text())
    counts = {s.dataset_id: len(s.stems())
              for s in DA.POSITIVE_SOURCES + DA.NEGATIVE_SOURCES if s.exists}
    sample = {k: min(v, DL.SAMPLE_FOR_PHASH) for k, v in counts.items()}

    chance = chance_rate()
    rate = chance["rate_within_radius"]
    observed = report["levels"]["L5_near_duplicate"]["pairs_within_radius"]
    normalised = {}
    for key, value in observed.items():
        a, b = key.split(" vs ")
        na, nb = sample.get(a, 0), sample.get(b, 0)
        pairs = na * (na - 1) / 2 if a == b else na * nb
        expected = pairs * rate
        normalised[key] = {
            "observed": int(value),
            "expected_by_chance": round(expected, 1),
            "excess_ratio": round(value / expected, 2) if expected else None,
            "observed_rate": round(value / pairs, 6) if pairs else None}
    report["levels"]["L5_near_duplicate"]["chance_calibration"] = chance
    report["levels"]["L5_near_duplicate"]["normalised"] = normalised
    report["levels"]["L5_near_duplicate"]["reading"] = (
        "counts are meaningless without the chance rate. BROAD-vs-BROAD sits at"
        " its own expectation, so the hash is not flagging copies inside the"
        " main pool. Where an excess exists it is small and lands on the"
        " viewpoint-targeted sets, which look alike by construction -- and"
        " exact RGB (L2) and label (L3) collisions are both 0.")

    report["levels"]["L4_generator_seed"]["finding"] = (
        "the 1,000 shared seed tuples are exactly"
        " EDGE_HARD_CLEAN_UNTOUCHED x EDGE_HARD_TRUNC_UNTOUCHED. That is the"
        " dataset's own pair contract -- one scene, two camera aims -- not"
        " leakage. Consequence: the two untouched zips are the SAME 1,000"
        " scenes, so they are not two independent holdouts and must not be"
        " counted as 2,000.")
    report["levels"]["L6_source_concentration"]["finding"] = (
        "every pool has exactly 2 background assets (industrial, parking_lot),"
        " so a >50% top share is structural, not a sampling artefact. Background"
        " diversity is a property of the generator, and it is low across the"
        " board.")
    path.write_text(json.dumps(report, indent=1, default=str))

    print(f"  chance: median {chance['median_distance']:.0f}/64, "
          f"within radius {rate:.4%}")
    for key, entry in sorted(normalised.items(),
                             key=lambda kv: -(kv[1]["excess_ratio"] or 0))[:8]:
        print(f"  {key:52} obs {entry['observed']:6d}  exp "
              f"{entry['expected_by_chance']:8.0f}  x{entry['excess_ratio']}")
    print("-> DUPLICATE_LEAKAGE_AUDIT.json (addendum)")


if __name__ == "__main__":
    main()
