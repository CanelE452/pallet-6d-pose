"""§8~§11 — DIVERSE_SWAP 구성 (추가가 아니라 **교체**) + 분포 게이트.

총 N 을 R0 train 과 정확히 같게 유지한다.  저앙각 dominant-asset(scene.usd) 프레임을
층(elevation x projected-size)별로 대응시켜 diverse pool 프레임과 1:1 로 바꾼다.
"""
from __future__ import annotations
import csv, json, collections
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "data/pallet/results/low_angle_diversity_v1"
SWAP_CAP = 5000


def load_csv(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def smd(a, b):
    """standardized mean difference (pooled sd)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    s = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((b.mean() - a.mean()) / s) if s > 0 else 0.0


def eff(counter, n):
    p = np.array(list(counter.values()), float) / n
    return float(np.exp(-(p * np.log(p)).sum()))


def main():
    r0 = load_csv(OUT / "R0_FRAME_TABLE.csv")
    pool = load_csv(OUT / "POOL_FRAMES.csv")
    stem_of = {}
    for line in (OUT / "POOL_STEMS.txt").read_text().split():
        # OBL__<pkgtail>__f0000  ->  (pkg, fid)
        _, tail, fid = line.split("__")
        stem_of[(f"corner_la_oblique_v1_{tail}", fid)] = line
    for r in pool:
        r["stem"] = stem_of[(r["pkg"], r["stem"])]

    tr = [r for r in r0 if r["split"] == "train"]
    for r in tr + pool:
        for k in ("elevation_deg", "distance_m", "diag_ratio", "aspect"):
            r[k] = float(r[k])

    donor = [r for r in tr if r["elevation_deg"] < 8 and r["source_asset"] == "scene.usd"]
    print(f"train {len(tr)} · 저앙각 dominant(scene.usd) donor {len(donor)} · pool {len(pool)}")

    # ---- 층: elevation 3구간 x projected size 3분위 (donor 기준 경계)
    def ebin(v):
        return 0 if v < 3 else (1 if v < 5 else 2)
    q = np.quantile([r["diag_ratio"] for r in donor], [1 / 3, 2 / 3])
    def sbin(v):
        return int(np.digitize(v, q))

    # §9 — distance 를 세 번째 층으로 넣을지.  넣으면 매칭은 좋아지지만 swap_N 이 준다.
    import os
    USE_DIST = os.environ.get("SWAP_STRATIFY_DISTANCE", "0") == "1"
    dq = np.quantile([r["distance_m"] for r in donor], [1 / 3, 2 / 3])
    def dbin(v):
        return int(np.digitize(v, dq)) if USE_DIST else 0

    key = lambda r: (ebin(r["elevation_deg"]), sbin(r["diag_ratio"]), dbin(r["distance_m"]))
    dcell = collections.defaultdict(list)
    for r in donor:
        dcell[key(r)].append(r)
    pcell = collections.defaultdict(list)
    for r in pool:
        pcell[key(r)].append(r)

    rng = np.random.default_rng(20260907)
    add, remove, per_cell = [], [], {}
    for c in sorted(set(dcell) | set(pcell)):
        n = min(len(dcell.get(c, [])), len(pcell.get(c, [])))
        per_cell[str(c)] = {"donor": len(dcell.get(c, [])), "pool": len(pcell.get(c, [])), "swap": n}
        if n == 0:
            continue
        add += [pcell[c][i] for i in rng.choice(len(pcell[c]), n, replace=False)]
        remove += [dcell[c][i] for i in rng.choice(len(dcell[c]), n, replace=False)]
    # 총 상한
    if len(add) > SWAP_CAP:
        idx = rng.choice(len(add), SWAP_CAP, replace=False)
        add = [add[i] for i in idx]
        remove = [remove[i] for i in idx]
    swap_n = len(add)
    print(f"swap_N = {swap_n}")

    rm = {r["frame_id"] for r in remove}
    ctrl = tr
    treat = [r for r in tr if r["frame_id"] not in rm] + add
    assert len(treat) == len(ctrl), (len(treat), len(ctrl))

    def dist(rows, key):
        return [r[key] for r in rows]

    gates, smds = {}, {}
    for key in ("elevation_deg", "distance_m", "diag_ratio", "aspect"):
        s = smd(dist(ctrl, key), dist(treat, key))
        smds[key] = s
        gates[key] = abs(s) <= 0.10
    # 저앙각 부분모집단에서도 본다 (전체 평균은 저앙각 5,000 을 희석한다)
    low_c = [r for r in ctrl if r["elevation_deg"] < 8]
    low_t = [r for r in treat if r["elevation_deg"] < 8]
    smds_low = {k: smd(dist(low_c, k), dist(low_t, k))
                for k in ("elevation_deg", "distance_m", "diag_ratio", "aspect")}

    ca = collections.Counter(r["source_asset"] for r in ctrl)
    ta = collections.Counter(r["source_asset"] for r in treat)
    cl = collections.Counter(r["source_asset"] for r in low_c)
    tl = collections.Counter(r["source_asset"] for r in low_t)

    rep = {
        "schema_version": "low_angle_diversity_v1_swap_v1",
        "stratify_distance": USE_DIST,
        "design": "matched replacement (총 N 불변). 추가가 아니다.",
        "swap_N": swap_n, "swap_cap": SWAP_CAP,
        "donor_pool_n": len(donor), "diverse_pool_n": len(pool),
        "total_N": {"control": len(ctrl), "treatment": len(treat), "delta": len(treat) - len(ctrl)},
        "per_stratum": per_cell,
        "SMD_full_train": smds, "SMD_gate_pass": gates,
        "SMD_low_angle_subpopulation": smds_low,
        "assets_full": {"control": dict(ca), "treatment": dict(ta),
                        "eff_control": eff(ca, len(ctrl)), "eff_treatment": eff(ta, len(treat))},
        "assets_low_angle": {"control": dict(cl), "treatment": dict(tl),
                             "n_control": len(low_c), "n_treatment": len(low_t),
                             "eff_control": eff(cl, len(low_c)), "eff_treatment": eff(tl, len(low_t))},
        "distributions_low_angle": {
            k: {"control_p5_p50_p95": [float(np.percentile(dist(low_c, k), p)) for p in (5, 50, 95)],
                "treatment_p5_p50_p95": [float(np.percentile(dist(low_t, k), p)) for p in (5, 50, 95)]}
            for k in ("elevation_deg", "distance_m", "diag_ratio", "aspect")},
    }
    worst = max(abs(v) for v in smds_low.values())
    rep["identification"] = ("MATCHED_SWAP_IDENTIFIED" if all(gates.values()) and worst <= 0.10
                             else "MATCHED_SWAP_NOT_IDENTIFIED")
    rep["interpretation_scope"] = (
        "asset diversity causal effect" if rep["identification"] == "MATCHED_SWAP_IDENTIFIED"
        else "low-angle diverse-pool package effect (인과 해석 낮춤)")
    (OUT / "SWAP_DESIGN.json").write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n")
    (OUT / "SWAP_REMOVED.txt").write_text("\n".join(sorted(rm)) + "\n")
    (OUT / "SWAP_ADDED.txt").write_text("\n".join(sorted(r["stem"] for r in add)) + "\n")
    # pool row 는 "stem", R0 row 는 "frame_id" 를 키로 쓴다
    (OUT / "DIVERSE_SWAP_MANIFEST.txt").write_text(
        "\n".join(sorted(r.get("frame_id") or r["stem"] for r in treat)) + "\n")
    print(json.dumps({k: rep[k] for k in
                      ("swap_N", "total_N", "SMD_full_train", "SMD_low_angle_subpopulation",
                       "identification", "interpretation_scope")}, indent=2, sort_keys=True))
    print("\nassets low-angle  control:", dict(cl), f"eff {rep['assets_low_angle']['eff_control']:.3f}")
    print("assets low-angle treatment:", dict(tl), f"eff {rep['assets_low_angle']['eff_treatment']:.3f}")


if __name__ == "__main__":
    main()
