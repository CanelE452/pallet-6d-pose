"""PHASE 2 -- duplicate and leakage audit, six levels kept apart.

Each level answers a different question, so they are never merged into one
"duplicates: 0" line:

  L1 frame_id      the same stem in two pools (cheap, and usually a false alarm:
                   every pool numbers from f00000, so names collide by design)
  L2 rgb checksum  the same pixels (sha256 over the whole file)
  L3 label digest  the same geometry, even if the pixels were re-encoded
  L4 seed          the same generator draw (stage_seeds tuple)
  L5 near-dup      perceptual hash within a Hamming radius
  L6 concentration one source/background dominating a pool

An eval frame appearing in any future training manifest is a HARD BLOCK.
"""
from __future__ import annotations

import hashlib
import io
import json
import pathlib
import sys
import zipfile
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import da_common as DA  # noqa: E402

PHASH_SIZE = 8
NEAR_DUP_RADIUS = 4
SAMPLE_FOR_PHASH = 4000       # per dataset; full-cross phash on 66k is quadratic
PHASH_SEED = 20260903


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rgb_digests(src: DA.Source):
    """(stem -> sha256) over the encoded bytes.  Zips are streamed."""
    out = {}
    if not src.exists:
        return out
    if src.kind == "dir":
        rgb = src.path / "rgb"
        if not rgb.exists():
            return out
        for path in sorted(rgb.iterdir()):
            out[path.stem] = _sha256(path.read_bytes())
        return out
    with zipfile.ZipFile(src.path) as zf:
        for name in sorted(zf.namelist()):
            if "/rgb/" not in name and not name.startswith("rgb/"):
                continue
            if name.endswith("/"):
                continue
            stem = pathlib.PurePosixPath(name).stem
            with zf.open(name) as fh:
                out[stem] = _sha256(fh.read())
    return out


def phash_of(data: bytes):
    """8x8 DCT-free average hash -- enough to catch re-renders of one scene."""
    import cv2
    buf = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    small = cv2.resize(image, (PHASH_SIZE, PHASH_SIZE),
                       interpolation=cv2.INTER_AREA)
    bits = (small > small.mean()).astype(np.uint8).reshape(-1)
    return np.packbits(bits).tobytes()


def sampled_phashes(src: DA.Source, rng):
    if not src.exists:
        return {}
    names = []
    if src.kind == "dir":
        rgb = src.path / "rgb"
        names = sorted(p.name for p in rgb.iterdir()) if rgb.exists() else []
        reader = lambda n: (src.path / "rgb" / n).read_bytes()
        zf = None
    else:
        zf = zipfile.ZipFile(src.path)
        names = sorted(n for n in zf.namelist()
                       if ("/rgb/" in n or n.startswith("rgb/"))
                       and not n.endswith("/"))
        reader = lambda n: zf.open(n).read()
    if len(names) > SAMPLE_FOR_PHASH:
        pick = rng.choice(len(names), SAMPLE_FOR_PHASH, replace=False)
        names = [names[i] for i in sorted(pick)]
    out = {}
    for name in names:
        stem = pathlib.PurePosixPath(name).stem
        value = phash_of(reader(name))
        if value is not None:
            out[stem] = value
    if zf is not None:
        zf.close()
    return out


def hamming(a: bytes, b: bytes) -> int:
    return int(np.unpackbits(np.frombuffer(a, np.uint8)
                             ^ np.frombuffer(b, np.uint8)).sum())


def main():
    DA.AUDIT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(PHASH_SEED)
    sources = [s for s in DA.POSITIVE_SOURCES + DA.NEGATIVE_SOURCES if s.exists]

    report = {"levels": {}, "HARD_BLOCK": []}

    # ---- L1 frame_id ------------------------------------------------------
    stems = {s.dataset_id: set(s.stems()) for s in sources}
    l1 = {}
    ids = sorted(stems)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            shared = stems[a] & stems[b]
            if shared:
                l1[f"{a} vs {b}"] = len(shared)
    report["levels"]["L1_frame_id"] = {
        "collisions": l1,
        "note": "every pool numbers from f00000, so a shared stem is a naming"
                " coincidence until L2/L3 agree. Loaders must namespace by"
                " dataset_id."}
    print(f"  L1 frame_id      pairs with shared stems: {len(l1)}", flush=True)

    # ---- L2 rgb checksum --------------------------------------------------
    digests = {}
    for src in sources:
        digests[src.dataset_id] = rgb_digests(src)
        print(f"     rgb digested  {src.dataset_id:28} "
              f"{len(digests[src.dataset_id]):6d}", flush=True)
    owner = defaultdict(list)
    for dataset_id, table in digests.items():
        for stem, digest in table.items():
            owner[digest].append((dataset_id, stem))
    cross = defaultdict(int)
    within = Counter()
    for digest, holders in owner.items():
        if len(holders) < 2:
            continue
        pools = sorted({d for d, _ in holders})
        if len(pools) == 1:
            within[pools[0]] += len(holders) - 1
        else:
            for i, a in enumerate(pools):
                for b in pools[i + 1:]:
                    cross[f"{a} vs {b}"] += 1
    report["levels"]["L2_rgb_sha256"] = {
        "cross_dataset": dict(cross), "within_dataset": dict(within),
        "total_images": sum(len(t) for t in digests.values())}
    print(f"  L2 rgb sha256    cross {sum(cross.values())}  "
          f"within {sum(within.values())}", flush=True)

    # ---- L3 label digest --------------------------------------------------
    label_owner = defaultdict(list)
    for src in sources:
        for stem, payload in src.labels():
            objects = payload.get("objects") or []
            if not objects:
                key = json.dumps(payload.get("camera_data"), sort_keys=True)
            else:
                obj = objects[0]
                key = json.dumps([obj.get("projected_cuboid"),
                                  obj.get("quaternion_xyzw"),
                                  obj.get("location")], sort_keys=True)
            label_owner[_sha256(key.encode())].append((src.dataset_id, stem))
    l3_cross, l3_within = defaultdict(int), Counter()
    for _, holders in label_owner.items():
        if len(holders) < 2:
            continue
        pools = sorted({d for d, _ in holders})
        if len(pools) == 1:
            l3_within[pools[0]] += len(holders) - 1
        else:
            for i, a in enumerate(pools):
                for b in pools[i + 1:]:
                    l3_cross[f"{a} vs {b}"] += 1
    report["levels"]["L3_label_digest"] = {
        "cross_dataset": dict(l3_cross), "within_dataset": dict(l3_within)}
    print(f"  L3 label digest  cross {sum(l3_cross.values())}  "
          f"within {sum(l3_within.values())}", flush=True)

    # ---- L4 generator seed ------------------------------------------------
    features = DA.AUDIT / "positive_frame_features.parquet"
    l4 = {}
    if features.exists():
        frame = pd.read_parquet(features)
        cols = [c for c in frame.columns if c.startswith("seed_")]
        have = frame.dropna(subset=cols, how="all")
        if len(have):
            key = have[cols].astype("string").agg("|".join, axis=1)
            grouped = pd.DataFrame({"dataset_id": have["dataset_id"],
                                    "key": key})
            per_key = grouped.groupby("key")["dataset_id"].nunique()
            dup = grouped.groupby("key").size()
            l4 = {"frames_with_seeds": int(len(have)),
                  "unique_seed_tuples": int(grouped["key"].nunique()),
                  "seed_tuples_in_more_than_one_dataset":
                      int((per_key > 1).sum()),
                  "seed_tuples_repeated": int((dup > 1).sum())}
    report["levels"]["L4_generator_seed"] = l4
    print(f"  L4 seed collision: {l4}", flush=True)

    # ---- L5 near duplicate -------------------------------------------------
    hashes = {s.dataset_id: sampled_phashes(s, rng) for s in sources}
    pairs = defaultdict(int)
    ids = sorted(hashes)
    for i, a in enumerate(ids):
        va = list(hashes[a].values())
        for b in ids[i:]:
            vb = list(hashes[b].values())
            if not va or not vb:
                continue
            arr_a = np.unpackbits(
                np.frombuffer(b"".join(va), np.uint8).reshape(len(va), -1),
                axis=1)
            arr_b = np.unpackbits(
                np.frombuffer(b"".join(vb), np.uint8).reshape(len(vb), -1),
                axis=1)
            dist = (arr_a[:, None, :] != arr_b[None, :, :]).sum(-1)
            if a == b:
                dist[np.triu_indices(len(va), 0)] = 999
            pairs[f"{a} vs {b}"] = int((dist <= NEAR_DUP_RADIUS).sum())
    report["levels"]["L5_near_duplicate"] = {
        "radius": NEAR_DUP_RADIUS, "sample_per_dataset": SAMPLE_FOR_PHASH,
        "pairs_within_radius": {k: v for k, v in pairs.items() if v}}
    print(f"  L5 near-dup      non-zero pairs: "
          f"{sum(1 for v in pairs.values() if v)}", flush=True)

    # ---- L6 source concentration ------------------------------------------
    l6 = {}
    if features.exists():
        frame = pd.read_parquet(features)
        for dataset_id, block in frame.groupby("dataset_id"):
            entry = {}
            for axis in ("background_asset", "scene_preset", "pallet_type",
                         "source_asset", "resolution"):
                if axis not in block or block[axis].isna().all():
                    continue
                counts = block[axis].value_counts(normalize=True)
                entry[axis] = {"top": str(counts.index[0]),
                               "share": round(float(counts.iloc[0]), 4),
                               "n_unique": int(block[axis].nunique())}
            l6[dataset_id] = entry
    report["levels"]["L6_source_concentration"] = l6

    # ---- HARD BLOCK -------------------------------------------------------
    eval_pools = {"EDGE_HARD_TRUNC_DEV", "EDGE_HARD_TRUNC_UNTOUCHED",
                  "EDGE_HARD_CLEAN_UNTOUCHED", "NEGATIVE_SYNTH_V1_DEV"}
    for key, count in cross.items():
        a, b = key.split(" vs ")
        if count and ({a, b} & eval_pools):
            report["HARD_BLOCK"].append(
                f"L2 rgb identity between {key}: {count}")
    report["MH_DEV_note"] = (
        "BROAD_40K carries mh_split; MH_DEV frames must never enter a training"
        " manifest. Enforced in da_manifest by filtering on mh_split.")
    report["LEAKAGE_CLEAN"] = not report["HARD_BLOCK"]

    (DA.AUDIT / "DUPLICATE_LEAKAGE_AUDIT.json").write_text(
        json.dumps(report, indent=1, default=str))
    print(f"  HARD_BLOCK: {len(report['HARD_BLOCK'])}  "
          f"LEAKAGE_CLEAN={report['LEAKAGE_CLEAN']}", flush=True)
    print("-> DUPLICATE_LEAKAGE_AUDIT.json")


if __name__ == "__main__":
    main()
