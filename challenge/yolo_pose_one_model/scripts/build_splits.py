"""Group-aware train/val split for the synthetic domains (G and T).

Plain image-level random splitting is forbidden. Grouping keys were chosen from what the
data actually records:

  G (paper_release 40k)
      Every frame has a unique seed (40000/40000) and diagnostic_mode "clean-static",
      i.e. frames are independently rendered - there is no camera trajectory to leak.
      Split is still deterministic and stratified over
      (pallet_type, scene_preset, background_asset) so every condition appears on both
      sides. Within a stratum, frames are ordered by seed and the tail 5% becomes val.

  T (v1 + v2 palletobj)
      frame_meta.scenario groups several camera frames of the SAME scene
      (v1: 2782 scenarios / 9997 frames, v2: 1305 / 9994). Splitting by frame would leak.
      Whole scenarios go to one side only; strata are (set, background_3d).

Outputs
  manifests/generic_train.txt, generic_val.txt, target_train.txt, target_val.txt
  manifests/split_manifest.json
  reports/03_split_report.md

Usage:
  python challenge/yolo_pose_one_model/scripts/build_splits.py
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

OUT = REPO / "challenge/yolo_pose_one_model"
G_RECORDS = REPO / ("data/pallet/training_data/paper_release/"
                    "v2_prod40k_clean_merged/records.jsonl")
SEED = 42


def stable_rank(key: str) -> int:
    """Deterministic pseudo-random rank, stable across runs and machines."""
    return int(hashlib.sha256(f"{SEED}:{key}".encode()).hexdigest()[:16], 16)


def load_registry():
    rows = list(csv.DictReader(open(OUT / "manifests/all_samples.csv", encoding="utf-8")))
    return [r for r in rows if r["image_exists"] == "true" and r["has_pallet"] == "1"]


def group_generic(rows, cache_path):
    """sample_id -> (pallet_type, scene_preset, background_asset).

    Read from each frame's own JSON, not from records.jsonl. records.jsonl cannot be
    joined reliably: its `idx` has only 38955 unique values over 40000 lines, and the
    file stems are zero-padded inconsistently (10000 four-digit + 30000 five-digit), so
    an idx->filename join silently mismatches. The three fields we need are stored in
    the per-frame JSON anyway (camera_data.scene_preset / .background_asset,
    objects[0].name), so we take them from there and cache the result.
    """
    cache = json.load(open(cache_path, encoding="utf-8")) if cache_path.exists() else {}
    dirty = False
    out = {}
    for r in rows:
        sid = r["sample_id"]
        if sid not in cache:
            try:
                d = json.load(open(REPO / r["annotation_path"], encoding="utf-8"))
                cd = d.get("camera_data", {}) or {}
                ob = (d.get("objects") or [{}])[0]
                cache[sid] = [ob.get("name") or "unknown",
                              cd.get("scene_preset") or "unknown",
                              cd.get("background_asset") or "unknown"]
            except Exception:
                cache[sid] = ["unknown", "unknown", "unknown"]
            dirty = True
        out[sid] = tuple(cache[sid])
    if dirty:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        json.dump(cache, open(cache_path, "w", encoding="utf-8"))
    return out


def group_target(rows, cache_path):
    """sample_id -> (stratum, scenario). scenario read from frame_meta, cached."""
    if cache_path.exists():
        cache = json.load(open(cache_path, encoding="utf-8"))
    else:
        cache = {}
    dirty = False
    out = {}
    for r in rows:
        sid = r["sample_id"]
        if sid not in cache:
            try:
                m = json.load(open(REPO / r["annotation_path"], encoding="utf-8")).get(
                    "frame_meta") or {}
            except Exception:
                m = {}
            cache[sid] = [m.get("scenario") or f"__solo__{sid}", m.get("background_3d") or "none"]
            dirty = True
        sc, bg = cache[sid]
        tag = sid.split("/")[1]                      # v1 / v2
        # Scenario ids repeat across v1 and v2 (both "R000xxx"), so they must be
        # namespaced or the two sets' scenarios collide and leak across the split.
        out[sid] = ((tag, bg), f"{tag}:{sc}")
    if dirty:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        json.dump(cache, open(cache_path, "w", encoding="utf-8"))
    return out


def split_generic(rows, strata, val_frac):
    by = collections.defaultdict(list)
    for r in rows:
        by[strata[r["sample_id"]]].append(r["sample_id"])
    train, val = [], []
    per_stratum = {}
    for st, sids in by.items():
        sids = sorted(sids, key=stable_rank)
        n_val = max(1, round(len(sids) * val_frac)) if len(sids) >= 20 else 0
        val += sids[:n_val]
        train += sids[n_val:]
        per_stratum["|".join(map(str, st))] = {"n": len(sids), "val": n_val}
    return sorted(train), sorted(val), per_stratum


def split_target(rows, groups, val_frac):
    """Whole scenarios move together.

    A scenario may contain frames tagged with different background_3d values. Assigning
    it per-frame would tear the scenario across strata and leak, so each scenario is
    placed in exactly one stratum: its most common background.
    """
    scen_frames = collections.defaultdict(list)
    scen_bg = collections.defaultdict(collections.Counter)
    for r in rows:
        st, sc = groups[r["sample_id"]]
        scen_frames[sc].append(r["sample_id"])
        scen_bg[sc][st] += 1

    by_stratum = collections.defaultdict(lambda: collections.defaultdict(list))
    for sc, sids in scen_frames.items():
        st = scen_bg[sc].most_common(1)[0][0]
        by_stratum[st][sc] = sids
    train, val = [], []
    per_stratum = {}
    for st, scen in by_stratum.items():
        names = sorted(scen, key=stable_rank)
        total = sum(len(scen[s]) for s in names)
        target = total * val_frac
        got, val_sc = 0, []
        for s in names:
            if got >= target:
                break
            val_sc.append(s)
            got += len(scen[s])
        for s in names:
            (val if s in set(val_sc) else train).extend(scen[s])
        per_stratum["|".join(map(str, st))] = {
            "frames": total, "scenarios": len(names),
            "val_frames": got, "val_scenarios": len(val_sc)}
    return sorted(train), sorted(val), per_stratum


def write_list(path, sids, index):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in sids:
            f.write(index[s]["image_path"] + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generic-val-frac", type=float, default=0.05)
    ap.add_argument("--target-val-frac", type=float, default=0.10)
    args = ap.parse_args()

    rows = load_registry()
    index = {r["sample_id"]: r for r in rows}
    G = [r for r in rows if r["domain"] == "generic_synth"]
    T = [r for r in rows if r["domain"] == "target_synth"]
    print(f"usable: G={len(G)}  T={len(T)}")

    print("grouping G (reads each JSON once, cached) ...")
    gstrata = group_generic(G, OUT / "manifests/_generic_stratum_cache.json")
    gtr, gva, gst = split_generic(G, gstrata, args.generic_val_frac)

    print("grouping T (reads frame_meta once, cached) ...")
    tgroups = group_target(T, OUT / "manifests/_target_scenario_cache.json")
    ttr, tva, tst = split_target(T, tgroups, args.target_val_frac)

    # leakage checks
    assert not (set(gtr) & set(gva)), "G train/val overlap"
    assert not (set(ttr) & set(tva)), "T train/val overlap"
    tr_sc = {tgroups[s][1] for s in ttr}
    va_sc = {tgroups[s][1] for s in tva}
    shared = tr_sc & va_sc
    assert not shared, f"T scenario leak: {list(shared)[:5]}"

    write_list(OUT / "manifests/generic_train.txt", gtr, index)
    write_list(OUT / "manifests/generic_val.txt", gva, index)
    write_list(OUT / "manifests/target_train.txt", ttr, index)
    write_list(OUT / "manifests/target_val.txt", tva, index)

    manifest = {
        "seed": SEED,
        "scope": "synthetic only (G+T); real deferred to a later finetune stage",
        "generic": {"train": len(gtr), "val": len(gva),
                    "group_key": "stratify(pallet_type, scene_preset, background_asset); "
                                 "unique seed per frame so no sequence leak exists",
                    "strata": gst},
        "target": {"train": len(ttr), "val": len(tva),
                   "group_key": "frame_meta.scenario (whole scenario to one side); "
                                "stratify(set, background_3d)",
                   "train_scenarios": len(tr_sc), "val_scenarios": len(va_sc),
                   "strata": tst},
        "real": {"status": "not built this round"},
    }
    json.dump(manifest, open(OUT / "manifests/split_manifest.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    lines = ["# 03 — Split report\n",
             "생성: `python challenge/yolo_pose_one_model/scripts/build_splits.py`  ",
             f"seed={SEED}, 결정적(해시 순위) — 재실행하면 같은 분할이 나온다.\n",
             "범위: 합성 G/T 만. real 은 이번 라운드에서 만들지 않았다(사용자 지시).\n",
             "## 결과\n```",
             f"{'domain':<16}{'train':>8}{'val':>8}{'val%':>7}",
             f"{'generic_synth':<16}{len(gtr):>8}{len(gva):>8}{100*len(gva)/(len(gtr)+len(gva)):>6.1f}%",
             f"{'target_synth':<16}{len(ttr):>8}{len(tva):>8}{100*len(tva)/(len(ttr)+len(tva)):>6.1f}%",
             "```\n",
             "## 누수 검사 [확인]\n```",
             "G train∩val sample_id      0",
             "T train∩val sample_id      0",
             f"T train∩val scenario       0   (train {len(tr_sc)} / val {len(va_sc)} scenario)",
             "```\n",
             "## G 그룹 근거\n",
             "`records.jsonl` 의 seed 가 40000/40000 전부 고유하고 `diagnostic_mode` 가",
             "`clean-static` 이다 = 프레임마다 독립 렌더, 카메라 궤적 없음. 시퀀스 누수가",
             "원천적으로 없다. 그래도 image-level 무작위를 피하려고",
             "(pallet_type, scene_preset, background_asset) 층화 + seed 해시 순위로 결정적 분할했다.\n",
             "```", f"{'stratum':<44}{'n':>7}{'val':>6}"]
    for k in sorted(gst):
        lines.append(f"{k:<44}{gst[k]['n']:>7}{gst[k]['val']:>6}")
    lines += ["```\n", "## T 그룹 근거\n",
              "`frame_meta.scenario` 가 같은 장면의 여러 카메라 프레임을 묶는다",
              "(v1 2782 scenario/9997 frame, v2 1305/9994). scenario 를 통째로 한쪽에만 넣었다.\n",
              "```", f"{'stratum(set|background_3d)':<34}{'frames':>8}{'scen':>7}{'val_fr':>8}{'val_sc':>8}"]
    for k in sorted(tst):
        s = tst[k]
        lines.append(f"{k:<34}{s['frames']:>8}{s['scenarios']:>7}{s['val_frames']:>8}{s['val_scenarios']:>8}")
    lines += ["```\n",
              "## 산출\n```",
              "manifests/generic_train.txt  generic_val.txt",
              "manifests/target_train.txt   target_val.txt",
              "manifests/split_manifest.json",
              "```"]
    (OUT / "reports/03_split_report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"G train={len(gtr)} val={len(gva)} | T train={len(ttr)} val={len(tva)}")
    print("wrote manifests/ and reports/03_split_report.md")


if __name__ == "__main__":
    main()
