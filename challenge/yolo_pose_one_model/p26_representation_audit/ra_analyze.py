"""separability + paired margin + domain matched + kp secondary -> CASE 판정."""
from __future__ import annotations
import csv, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ra_core as RC                                                # noqa: E402
NS = RC.NS
Z = np.load(f"{NS}/FEATURE_VECTORS.npz")
ROWS = list(csv.DictReader(open(f"{NS}/CANDIDATE_PROVENANCE.csv")))
for r in ROWS:
    for k in ("class_score", "iou", "kp_median", "kp_p90", "kp8_median"):
        r[k] = float(r[k]) if r.get(k) not in (None, "", "None") else None
TAPS = ("neck_in", "cls1", "cls_pen", "logit")
LEVELS = ("P3", "P4", "P5")
RNG = np.random.default_rng(42)
BOOT, MIN_N = 100, 6


def get(group, level, tap):
    k = f"{group}|{level}|{tap}"
    if k not in Z.files:
        return None, None
    v = Z[k].astype(np.float32)
    rid = Z[f"{group}|{level}|row_id"]
    return v, rid


def l2(v):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def knn_auroc(A, B, k=5):
    X = np.vstack([A, B]); y = np.r_[np.zeros(len(A)), np.ones(len(B))]
    from sklearn.neighbors import NearestNeighbors
    kk = min(k, len(X) - 1)
    nn = NearestNeighbors(n_neighbors=kk + 1).fit(X)
    _, idx = nn.kneighbors(X)
    s = y[idx[:, 1:]].mean(1)
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def centroid_bacc(A, B):
    ca, cb = A.mean(0), B.mean(0)
    da = np.linalg.norm(A - ca, axis=1) - np.linalg.norm(A - cb, axis=1)
    db = np.linalg.norm(B - ca, axis=1) - np.linalg.norm(B - cb, axis=1)
    return float(0.5 * ((da < 0).mean() + (db > 0).mean()))


def fisher(A, B):
    ca, cb = A.mean(0), B.mean(0)
    num = float(np.sum((ca - cb) ** 2))
    den = float(A.var(0).sum() + B.var(0).sum())
    return num / max(den, 1e-12)


def cos_dist(A, B):
    ca, cb = A.mean(0), B.mean(0)
    return float(1 - np.dot(ca, cb) / max(np.linalg.norm(ca) * np.linalg.norm(cb), 1e-12))


def dispersion(A):
    c = A.mean(0)
    return float(np.mean(np.linalg.norm(A - c, axis=1)))


def sep(A, B, boot=BOOT):
    if A is None or B is None or len(A) < MIN_N or len(B) < MIN_N:
        return {"n_a": (0 if A is None else len(A)), "n_b": (0 if B is None else len(B)),
                "SKIPPED": True, "why": f"표본 부족 (<{MIN_N})"}
    A, B = l2(A), l2(B)
    n = min(len(A), len(B))
    out = {"n_a": len(A), "n_b": len(B), "n_balanced": n, "SKIPPED": False,
           "knn5_auroc_full": knn_auroc(A, B),
           "centroid_bacc_full": centroid_bacc(A, B),
           "fisher_ratio": fisher(A, B), "centroid_cos_dist": cos_dist(A, B),
           "dispersion_a": dispersion(A), "dispersion_b": dispersion(B)}
    au, ba = [], []
    for _ in range(boot):
        ia = RNG.choice(len(A), n, replace=False)
        ib = RNG.choice(len(B), n, replace=False)
        try:
            au.append(knn_auroc(A[ia], B[ib]))
            ba.append(centroid_bacc(A[ia], B[ib]))
        except Exception:
            pass
    if au:
        out["knn5_auroc_balanced"] = {"median": float(np.median(au)),
                                      "ci95": [float(np.percentile(au, 2.5)),
                                               float(np.percentile(au, 97.5))]}
        out["centroid_bacc_balanced"] = {"median": float(np.median(ba)),
                                         "ci95": [float(np.percentile(ba, 2.5)),
                                                  float(np.percentile(ba, 97.5))]}
    return out


def stack_all(group, tap):
    """ALL scope — neck_in 은 level 마다 차원이 달라 합칠 수 없다."""
    if tap == "neck_in":
        return None, None
    vs, rs = [], []
    for lv in LEVELS:
        v, r = get(group, lv, tap)
        if v is not None and len(v):
            vs.append(v); rs.append(r)
    if not vs:
        return None, None
    return np.vstack(vs), np.concatenate(rs)


PAIRS = [("A_Rplus_vs_RW", "R+", "RW"), ("B_Rplus_vs_RANKFAIL", "R+", "RW_RANKFAIL"),
         ("C_Rplus_vs_RN", "R+", "RN"), ("D_Splus_vs_Rplus", "S+", "R+")]
SEP_ALL, SEP_LV = {}, {}
for name, ga, gb in PAIRS:
    SEP_ALL[name] = {}
    for tap in TAPS:
        if tap == "neck_in":
            SEP_ALL[name][tap] = {"SKIPPED": True,
                                  "why": "neck_in 은 level 마다 채널 수가 달라 ALL 로 합칠 수 없다"}
            continue
        A, _ = stack_all(ga, tap); B, _ = stack_all(gb, tap)
        SEP_ALL[name][tap] = sep(A, B)
    SEP_LV[name] = {}
    for lv in LEVELS:
        SEP_LV[name][lv] = {}
        for tap in TAPS:
            A, _ = get(ga, lv, tap); B, _ = get(gb, lv, tap)
            SEP_LV[name][lv][tap] = sep(A, B)
json.dump({"pairs": SEP_ALL, "note": "neck_in 은 level 별로만 유효 (채널 차원 상이)",
           "metrics": ["5NN AUROC", "nearest-centroid balanced acc", "Fisher ratio",
                       "centroid cosine distance", "within-class dispersion"],
           "balanced_subsample": "min(nA,nB), seed 42, 100 bootstrap 95% CI",
           "PCA_UMAP": "visualization only — gate evidence 아님"},
          open(f"{NS}/SEPARABILITY_ALL.json", "w"), indent=2, ensure_ascii=False)
json.dump(SEP_LV, open(f"{NS}/SEPARABILITY_BY_LEVEL.json", "w"), indent=2, ensure_ascii=False)

# ---------------------------------------------------------------- paired margin
byimg = {}
for r in ROWS:
    byimg.setdefault(r["image_id"], {})[r["group"]] = r
pairs = [(v["R+"], v["RW"]) for v in byimg.values() if "R+" in v and "RW" in v]


def mstat(ps):
    if not ps:
        return {"n": 0}
    m = np.array([a["class_score"] - b["class_score"] for a, b in ps])
    return {"n": len(m), "median": float(np.median(m)),
            "p10": float(np.percentile(m, 10)), "p90": float(np.percentile(m, 90)),
            "frac_correct_gt_wrong": float((m > 0).mean())}


PM = {"ALL": mstat(pairs),
      "DAY": mstat([p for p in pairs if p[0]["domain"] == "DAY"]),
      "NIGHT": mstat([p for p in pairs if p[0]["domain"] == "NIGHT"]),
      "definition": "score_margin = class_score(R+) - class_score(RW), 같은 프레임",
      "note": "stage 별 직접 score 가 없으므로 separability 와 최종 logit margin 을 연결해 읽는다"}
json.dump(PM, open(f"{NS}/PAIRED_RPLUS_RW.json", "w"), indent=2, ensure_ascii=False)

# ---------------------------------------------------------------- domain matched
def area(r):
    b = json.loads(r["box"]); return (b[2]-b[0]) * (b[3]-b[1])


rows_by_id = {i: r for i, r in enumerate(ROWS)}
DM = {"raw": {}, "level_matched": {}, "size_matched": {}, "DAY": {}, "NIGHT": {}}
for tap in TAPS:
    if tap != "neck_in":
        A, _ = stack_all("S+", tap); B, _ = stack_all("R+", tap)
        DM["raw"][tap] = sep(A, B)
    DM["level_matched"][tap] = {lv: sep(*[get(g, lv, tap)[0] for g in ("S+", "R+")])
                                for lv in LEVELS}
    for dom in ("DAY", "NIGHT"):
        v, rid = get("R+", "P5", tap)
        if v is None:
            DM[dom][tap] = {"SKIPPED": True}; continue
        keep = [i for i, r in enumerate(rid) if ROWS[int(r)]["domain"] == dom]
        s, _ = get("S+", "P5", tap)
        DM[dom][tap] = sep(s, v[keep] if keep else None)
# size matched — P5 안에서 bbox area 사분위 맞춤
sp_v, sp_r = get("S+", "P5", "cls_pen"); rp_v, rp_r = get("R+", "P5", "cls_pen")
if sp_v is not None and rp_v is not None:
    ra_ = np.array([area(ROWS[int(i)]) for i in rp_r])
    sa_ = np.array([area(ROWS[int(i)]) for i in sp_r])
    lo, hi = np.percentile(ra_, [10, 90])
    keep_s = np.where((sa_ >= lo) & (sa_ <= hi))[0]
    for tap in TAPS:
        s, _ = get("S+", "P5", tap); r_, _ = get("R+", "P5", tap)
        DM["size_matched"][tap] = sep(s[keep_s] if len(keep_s) else None, r_)
    DM["size_matched"]["_window"] = {"real_area_p10": float(lo), "real_area_p90": float(hi),
                                     "n_synth_kept": int(len(keep_s))}
DM["confound_note"] = ("S+ 와 R+ 는 source level 분포가 다르다 (S+ P5 73.7% vs R+ P5 88.1%). "
                       "level_matched(P5) 를 primary 로 읽는다.")
json.dump(DM, open(f"{NS}/DOMAIN_MATCHED_ANALYSIS.json", "w"), indent=2, ensure_ascii=False)

# ---------------------------------------------------------------- kp secondary
KP = {"threshold_px": 20, "levels": {}}
for lv in LEVELS:
    v, rid = get("R+", lv, "pose_pen")
    if v is None or not len(v):
        KP["levels"][lv] = {"SKIPPED": True}; continue
    e = np.array([ROWS[int(i)]["kp_median"] or np.nan for i in rid])
    good, bad = v[e <= 20], v[e > 20]
    KP["levels"][lv] = {"n_good": int(len(good)), "n_bad": int(len(bad)),
                        "pose_pen": sep(good, bad), "cls_pen": sep(
                            *[get("R+", lv, "cls_pen")[0][e <= 20],
                              get("R+", lv, "cls_pen")[0][e > 20]])}
kpall = np.array([r["kp_median"] for r in ROWS if r["group"] == "R+" and r["kp_median"] is not None])
KP["R_plus_kp"] = {"n": int(kpall.size), "median": float(np.median(kpall)),
                   "p90": float(np.percentile(kpall, 90)),
                   "frac_bad_gt20": float((kpall > 20).mean())}
kp8 = np.array([r["kp8_median"] for r in ROWS if r["group"] == "R+" and r["kp8_median"] is not None])
KP["R_plus_kp8"] = {"median": float(np.median(kp8)), "p90": float(np.percentile(kp8, 90))}
KP["★secondary"] = "primary architecture gate 와 분리해 읽는다"
json.dump(KP, open(f"{NS}/KP_SECONDARY_ANALYSIS.json", "w"), indent=2, ensure_ascii=False)
print("analysis 완료")
