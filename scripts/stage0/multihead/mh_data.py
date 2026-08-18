"""Data contract for the corner+line+mask screen, on v2_prod40k_clean_merged.

The line stage's own dataset (`pallet6d_v2_10k`) was deleted on 2026-08-14 and a
full-disk search found no copy, so `line_internal_split.csv` and every manifest
beside it point at frames that do not exist.  The 40k release is a re-render of
the same scene parameters with different pixels and different label resolutions,
which means the historical numbers are not a baseline for anything trained here.
A0 is therefore re-established on this data rather than reproduced, and all three
arms share this one contract.

What is carried over unchanged from `line_feature_capacity_v2.load_frame`:
the 400x400 squash, ImageNet normalisation, and the canonical 50-grid mapping
`x * 50 / width`.  Only the file layout differs, so the geometry the line stage
was tuned against is the same geometry here.

What is new: masks (the old set had none) and a group-aware split built from
appearance keys.  Grouping on `background_asset | scene_preset | floor` keeps the
dev set appearance-disjoint from train, which is what the D0/D2 gap in the line
stage was measuring; object identity is deliberately not in the key, because
holding out one of four pallet meshes would cost a quarter of the data and
change the question from "did it memorise this backdrop" to "does it generalise
across meshes".
"""
from __future__ import annotations

import csv
import hashlib
import json
import pathlib
import random
from typing import Iterable

import cv2
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[3]
DATA = ROOT / "data/pallet/training_data/paper_release/v2_prod40k_clean_merged"
OUT = ROOT / "data/pallet/results/paper_s2_multihead"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE = 400                     # the resolution A1 was trained at
GRID = 50                       # canonical belief grid; every error lives here
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

CORNER_SIGMA = 2.0              # cells.  The A1 checkpoint's own blobs measure
                                # sigma ~2.1, so a wider target would fight the
                                # head we warm-start from.  Locked, no sweep.
MASK_THRESHOLD = 127            # mask PNGs are antialiased: background 0..2,
                                # foreground 254..255
DEV_GROUP_FRACTION = 0.15
DEV_SAMPLE = 512
SEEN_SAMPLE = 512
SPLIT_SEED = 20260816

SPLIT_CSV = OUT / "mh_split.csv"
CONTRACT_JSON = OUT / "mh_data_contract.json"


# --------------------------------------------------------------------------
# frames and labels


def frame_stems() -> list[str]:
    suffix = "_label.json"
    return sorted(p.name[: -len(suffix)] for p in (DATA / "labels").iterdir()
                  if p.name.endswith(suffix))


def read_label(stem: str) -> dict:
    return json.loads((DATA / "labels" / f"{stem}_label.json").read_text("utf-8"))


def group_key(payload: dict) -> str:
    camera = payload["camera_data"]
    floor = camera.get("floor")
    texture = floor.get("floor_texture") if isinstance(floor, dict) else None
    return "|".join(str(v) for v in (camera.get("background_asset"),
                                     camera.get("scene_preset"),
                                     camera.get("floor_mode"), texture))


def frame_row(stem: str) -> dict:
    """Everything the split and the strata need, from JSON only -- no PNG decode."""
    payload = read_label(stem)
    camera = payload["camera_data"]
    obj = payload["objects"][0]
    width, height = float(camera["width"]), float(camera["height"])
    cuboid = np.asarray(obj["projected_cuboid"], float)
    inside = [(0 <= x < width and 0 <= y < height) for x, y in cuboid]
    span = max(cuboid[:, 0].ptp(), cuboid[:, 1].ptp()) / max(width, height)
    labels = obj.get("v2_labels", {})
    return {"stem": stem, "group": group_key(payload),
            "v": int(sum(inside)),
            "elev": float(labels.get("elevation_deg_actual", float("nan"))),
            "size": float(span),
            "pallet_type": str(labels.get("pallet_type")),
            "width": width, "height": height}


# --------------------------------------------------------------------------
# split


def _stratum(row: dict) -> str:
    """The axes the screen reports on, so dev covers all of them by construction."""
    v = "V8" if row["v"] == 8 else "Vlt8"
    elev = "lo" if row["elev"] < 15 else "hi"
    size = "near" if row["size"] >= 0.40 else "far"
    return f"{v}-{elev}-{size}"


BALANCE_AXES = ("background", "preset", "floor_mode", "pallet_type", "stratum")


def _group_profile(rows: list[dict]) -> tuple[dict, dict, dict]:
    """Per-group frame count and per-axis value counts, plus the global counts."""
    sizes: dict[str, int] = {}
    profile: dict[str, dict[tuple[str, str], int]] = {}
    overall: dict[tuple[str, str], int] = {}
    for row in rows:
        group = row["group"]
        sizes[group] = sizes.get(group, 0) + 1
        bucket = profile.setdefault(group, {})
        parts = row["group"].split("|")
        values = {"background": parts[0], "preset": parts[1],
                  "floor_mode": parts[2], "pallet_type": row["pallet_type"],
                  "stratum": row["stratum"]}
        for axis, value in values.items():
            bucket[(axis, value)] = bucket.get((axis, value), 0) + 1
            overall[(axis, value)] = overall.get((axis, value), 0) + 1
    return sizes, profile, overall


def build_split(rows: list[dict]) -> list[dict]:
    """Groups go whole to train or dev; frames never straddle the boundary.

    Which groups is chosen by minimising marginal imbalance directly rather than
    by any sampling rule.  Every rule tried first failed on the same asymmetry:
    `native` floors exist as exactly eight large groups (one per background x
    preset) while `plane` floors exist as eighty-eight small ones, so
    largest-first put all natives in dev, a per-cell quota put none of them
    there, and a per-cell lottery swung the background and preset shares by
    twenty points.  Picking greedily on the imbalance itself is both simpler and
    the thing actually wanted: dev ends up looking like the dataset on every
    reported axis, while still sharing no appearance group with train.

    Cost is the total absolute deviation between dev's share of each axis value
    and the dataset's.  `stratum` is included because V<8 and low-angle frames
    are the populations the screen reports on, and a dev set short of them
    cannot answer the truncation question.
    """
    for row in rows:
        row["stratum"] = _stratum(row)
    sizes, profile, overall = _group_profile(rows)
    total = len(rows)
    target = DEV_GROUP_FRACTION * total

    def imbalance(counts: dict, n: int) -> float:
        if n == 0:
            return float("inf")
        return sum(abs(counts.get(key, 0) / n - value / total)
                   for key, value in overall.items())

    order = sorted(sizes)
    random.Random(SPLIT_SEED).shuffle(order)
    dev_groups: set[str] = set()
    counts: dict[tuple[str, str], int] = {}
    taken = 0
    while taken < target:
        best, best_cost = None, float("inf")
        for group in order:
            if group in dev_groups or taken + sizes[group] > target * 1.05:
                continue
            trial = dict(counts)
            for key, value in profile[group].items():
                trial[key] = trial.get(key, 0) + value
            cost = imbalance(trial, taken + sizes[group])
            if cost < best_cost:
                best, best_cost = group, cost
        if best is None:
            break
        dev_groups.add(best)
        for key, value in profile[best].items():
            counts[key] = counts.get(key, 0) + value
        taken += sizes[best]
    for row in rows:
        row["split"] = "MH_DEV" if row["group"] in dev_groups else "MH_TRAIN"
    return rows


def _stratified_sample(rows: list[dict], count: int, seed: int) -> list[str]:
    """Proportional over strata, deterministic, so V<8 cannot vanish from dev."""
    buckets: dict[str, list[str]] = {}
    for row in rows:
        buckets.setdefault(row["stratum"], []).append(row["stem"])
    rng = random.Random(seed)
    for stems in buckets.values():
        stems.sort()
        rng.shuffle(stems)
    total = len(rows)
    picked: list[str] = []
    for stratum in sorted(buckets):
        share = round(count * len(buckets[stratum]) / total)
        picked.extend(buckets[stratum][:share])
    # Rounding can miss by a few either way; fix it deterministically.
    pool = [s for stratum in sorted(buckets) for s in buckets[stratum]
            if s not in set(picked)]
    picked.extend(pool[: max(0, count - len(picked))])
    return sorted(picked[:count])


def write_split() -> dict:
    rows = [frame_row(stem) for stem in frame_stems()]
    rows = build_split(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    with SPLIT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "stem", "group", "split", "stratum", "v", "elev", "size",
            "pallet_type", "width", "height"])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in writer.fieldnames})
    train = [r for r in rows if r["split"] == "MH_TRAIN"]
    dev = [r for r in rows if r["split"] == "MH_DEV"]
    manifests = {
        "D2_MH_DEV512": _stratified_sample(dev, DEV_SAMPLE, SPLIT_SEED),
        "D0_MH_SEEN512": _stratified_sample(train, SEEN_SAMPLE, SPLIT_SEED + 1),
    }
    for name, stems in manifests.items():
        (OUT / f"{name.lower()}_manifest.json").write_text(
            json.dumps({"n": len(stems), "stems": stems}, indent=1))
    contract = {
        "dataset": str(DATA.relative_to(ROOT)),
        "frames": len(rows),
        "groups": len({r["group"] for r in rows}),
        "train": len(train), "dev": len(dev),
        "dev_groups": sorted({r["group"] for r in dev}),
        "train_group_overlap": sorted({r["group"] for r in train}
                                      & {r["group"] for r in dev}),
        "split_sha256": hashlib.sha256(SPLIT_CSV.read_bytes()).hexdigest(),
        "manifest_sha256": {n: hashlib.sha256(
            json.dumps(s).encode()).hexdigest() for n, s in manifests.items()},
        "strata_train": _counts(train), "strata_dev": _counts(dev),
        "marginals": {axis: {"train": _marginal(train, axis),
                             "dev": _marginal(dev, axis)}
                      for axis in ("background", "preset", "floor_mode",
                                   "pallet_type")},
        "image": IMAGE, "grid": GRID, "corner_sigma": CORNER_SIGMA,
        "seed": SPLIT_SEED,
    }
    CONTRACT_JSON.write_text(json.dumps(contract, indent=1))
    return contract


def _counts(rows: list[dict]) -> dict:
    out: dict[str, int] = {}
    for row in rows:
        out[row["stratum"]] = out.get(row["stratum"], 0) + 1
    return dict(sorted(out.items()))


_AXIS_FIELD = {"background": 0, "preset": 1, "floor_mode": 2}


def _marginal(rows: list[dict], axis: str) -> dict:
    """Share of frames per value, so train/dev skew is visible not assumed."""
    out: dict[str, float] = {}
    for row in rows:
        value = (row["pallet_type"] if axis == "pallet_type"
                 else row["group"].split("|")[_AXIS_FIELD[axis]])
        out[value] = out.get(value, 0) + 1
    return {k: round(v / len(rows), 4) for k, v in sorted(out.items())}


def load_split() -> list[dict]:
    with SPLIT_CSV.open() as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["v"] = int(row["v"])
        row["elev"] = float(row["elev"])
        row["size"] = float(row["size"])
    return rows


def pools() -> tuple[list[str], dict[str, list[str]]]:
    rows = load_split()
    train = [r["stem"] for r in rows if r["split"] == "MH_TRAIN"]
    populations = {}
    for name in ("D2_MH_DEV512", "D0_MH_SEEN512"):
        populations[name] = json.loads(
            (OUT / f"{name.lower()}_manifest.json").read_text())["stems"]
    return train, populations


# --------------------------------------------------------------------------
# frame loading


def load_frame(stem: str):
    """Image, 9 keypoints in 50-grid units, visible mask on the same grid.

    Identical preprocessing to `line_feature_capacity_v2.load_frame` -- the same
    squash to 400x400, the same ImageNet statistics, the same `x * GRID / width`
    mapping -- so the line targets mean here what they meant there.
    """
    payload = read_label(stem)
    camera = payload["camera_data"]
    width, height = float(camera["width"]), float(camera["height"])
    image = cv2.imread(str(DATA / "rgb" / f"{stem}_rgb.png"))
    if image is None:
        raise FileNotFoundError(f"{stem}: no rgb")
    if image.shape[:2] != (int(height), int(width)):
        raise RuntimeError(f"{stem}: JSON says {width}x{height}, image is "
                           f"{image.shape[1]}x{image.shape[0]}")
    rgb = cv2.cvtColor(cv2.resize(image, (IMAGE, IMAGE)), cv2.COLOR_BGR2RGB)
    normalised = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
    obj = payload["objects"][0]
    points = np.concatenate([np.asarray(obj["projected_cuboid"], float),
                             np.asarray(obj["projected_cuboid_centroid"],
                                        float).reshape(1, 2)], 0)
    grid = np.stack([points[:, 0] * GRID / width,
                     points[:, 1] * GRID / height], 1)
    raw = cv2.imread(str(DATA / "mask_visible" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise FileNotFoundError(f"{stem}: no mask_visible")
    mask = cv2.resize((raw > MASK_THRESHOLD).astype(np.float32), (GRID, GRID),
                      interpolation=cv2.INTER_AREA)
    return normalised.transpose(2, 0, 1), rgb, grid, mask, (width, height)


def belief_target(grid: np.ndarray, sigma: float = CORNER_SIGMA):
    """(9, GRID, GRID) Gaussians and a per-channel validity flag.

    The map itself is `utils_belief.CreateBeliefMap(clip_at_border=True)` to the
    bit -- same 4-sigma window, same `int()` truncation of the window centre,
    same `0 <= p < GRID` inside test -- only vectorised.  `mh_wiring.T0B` pins
    the two against each other at 0.0, so a change upstream fails the test rather
    than silently splitting the target from the one the warm-started head knows.

    The validity flag is the new part, and it is the whole point of section 5.
    A corner whose centre falls outside the grid gets `valid=False` and is
    dropped from the loss mean, instead of being supervised as an empty channel.
    The legacy all-ones mask teaches a truncated corner that there is no corner
    anywhere in the image, which is a different and wrong statement; neither path
    clamps a coordinate to the border.
    """
    channels = grid.shape[0]
    window = int(sigma * 2)
    maps = np.zeros((channels, GRID, GRID), np.float32)
    valid = np.zeros(channels, bool)
    for c in range(channels):
        x, y = float(grid[c][0]), float(grid[c][1])
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        if not (0 <= y < GRID and 0 <= x < GRID):
            continue
        valid[c] = True
        i0, i1 = max(0, int(y) - window), min(GRID - 1, int(y) + window)
        j0, j1 = max(0, int(x) - window), min(GRID - 1, int(x) + window)
        rows = np.arange(i0, i1 + 1)[:, None]
        cols = np.arange(j0, j1 + 1)[None, :]
        maps[c, i0:i1 + 1, j0:j1 + 1] = np.exp(
            -(((rows - y) ** 2 + (cols - x) ** 2) / (2.0 * sigma ** 2)))
    return maps, valid


def load_pack(chunk: Iterable[str]) -> dict:
    frames = [load_frame(stem) for stem in chunk]
    targets = [belief_target(f[2]) for f in frames]
    return {
        "chunk": list(chunk),
        "images": torch.from_numpy(np.stack([f[0] for f in frames])).to(DEV),
        "rgb": [f[1] for f in frames],
        "grid": np.stack([f[2] for f in frames]),
        "mask": torch.from_numpy(
            np.stack([f[3] for f in frames])[:, None]).to(DEV),
        "belief": torch.from_numpy(np.stack([t[0] for t in targets])).to(DEV),
        "belief_valid": torch.from_numpy(
            np.stack([t[1] for t in targets])).to(DEV),
        "resolution": [f[4] for f in frames],
    }


if __name__ == "__main__":
    import pprint
    pprint.pprint(write_split())
