"""이 감사가 만든 수치에 불확실성을 붙인다 (지시문 §29).

목적 : 점추정만으로 다음 실험의 성공 임계를 정한 것을 바로잡는다.
지표 : (1) 앙각별 실패율의 세션 클러스터 부트스트랩 CI
       (2) FT vs synth 의 **짝지은(paired)** 차이 — 프레임 부트스트랩과 세션 클러스터 둘 다
       (3) 세션별 개별 결과
       클러스터 수가 적으면 그 사실 자체를 결과로 보고한다. 넓은 CI 를 좁혀 쓰지 않는다.

읽기 전용, 새 추론 0 회.
"""
import csv, json, pathlib, sys, collections
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from challenge.data_paths import EVAL_CANONICAL, FINAL_TEST
OUT = ROOT / "data/pallet/results/accuracy_root_cause_v1"
RNG = np.random.default_rng(20260906)
B = 10000


def cluster_boot(units, stat, n=B):
    """units = {cluster_id: [값...]}. 클러스터를 복원추출해 stat 을 다시 계산한다."""
    keys = list(units)
    if len(keys) < 2:
        return (np.nan, np.nan)
    out = []
    for _ in range(n):
        pick = RNG.choice(len(keys), size=len(keys), replace=True)
        pool = [v for i in pick for v in units[keys[i]]]
        if pool:
            out.append(stat(pool))
    if not out:
        return (np.nan, np.nan)
    return tuple(np.percentile(out, [2.5, 97.5]))


def frame_boot(vals, stat, n=B):
    v = np.asarray(vals, float)
    if len(v) < 2:
        return (np.nan, np.nan)
    idx = RNG.integers(0, len(v), size=(n, len(v)))
    return tuple(np.percentile([stat(v[i]) for i in idx], [2.5, 97.5]))


rate = lambda a: float(np.mean(a))
med = lambda a: float(np.median(a))

# ── (1) 앙각별 실패율 ────────────────────────────────────────────────
rec = list(csv.DictReader(open(OUT / "R0_ELEVATION_FAILURE.csv")))
for r in rec:
    r["elev"] = float(r["elev"]); r["nme"] = float(r["nme"]); r["mx"] = float(r["mx"])
NME_T = 0.0747
EB = ["<3", "3-8", "8-15", "15-30", ">=30"]
ebin = lambda e: "<3" if e < 3 else "3-8" if e < 8 else "8-15" if e < 15 else "15-30" if e < 30 else ">=30"

print("=" * 78)
print("(1) 앙각별 NME 실패율 — 세션 클러스터 부트스트랩 95% CI")
print(f"    모집단 PAPER_EVAL 319, 세션 {len(set(r['sess'] for r in rec))}개, B={B}")
print("=" * 78)
print(f"{'앙각':8s} {'N':>4s} {'세션':>5s} {'실패율':>8s} {'세션클러스터 95%CI':>24s} {'프레임 95%CI':>22s}")
elev_rows = []
for k in EB:
    sub = [r for r in rec if ebin(r["elev"]) == k]
    if not sub: continue
    units = collections.defaultdict(list)
    for r in sub:
        units[r["sess"]].append(r["nme"] > NME_T)
    p = np.mean([r["nme"] > NME_T for r in sub])
    lo, hi = cluster_boot(units, rate)
    flo, fhi = frame_boot([r["nme"] > NME_T for r in sub], rate)
    print(f"{k:8s} {len(sub):4d} {len(units):5d} {p*100:7.1f}% "
          f"[{lo*100:7.1f}, {hi*100:7.1f}] {'':2s}[{flo*100:6.1f}, {fhi*100:6.1f}]")
    elev_rows.append(dict(bin=k, n=len(sub), n_sessions=len(units), rate=p,
                          cluster_ci=[lo, hi], frame_ci=[flo, fhi]))

# ── (2) FT vs synth, 짝지은 차이 ─────────────────────────────────────
def load_kps(tag):
    d = json.load(open(ROOT / f"data/pallet/results/model_compare/kps_{tag}.json"))
    return {f["fid"]: np.asarray(f["kps"], float) for f in d["frames"] if f.get("kps")}

gt = {}
for name, rel in EVAL_CANONICAL.items():
    for jp in sorted((ROOT / rel).glob("*.json")):
        o = json.loads(jp.read_text()); objs = o.get("objects") or []
        if not objs or objs[0].get("split") != "eval": continue
        ob = objs[0]; mk = ob.get("manual_kps")
        if mk is None: continue
        pts = np.asarray([[-1., -1.] if v is None else [float(v[0]), float(v[1])] for v in mk], float)
        T = np.asarray(ob["pose_transform"], float); R, t = T[:3, :3], T[:3, 3]
        n = R @ np.array([0., -1., 0.]); u = -t / max(np.linalg.norm(t), 1e-9)
        gt[jp.stem] = (pts, float(np.degrees(np.arcsin(np.clip(abs(float(n @ u)), 0, 1)))), name)

M = {t: load_kps(t) for t in ("yolo26n_synth", "yolo26n_ft", "yolo26m_ft")}
common = set(gt) & set.intersection(*[set(m) for m in M.values()])
sealed = {f for f in common if gt[f][2] in set(FINAL_TEST)}

def merr(p, g):
    ok = ~((g[:, 0] == -1) & (g[:, 1] == -1))
    return float(np.median(np.linalg.norm(p[:len(g)][ok] - g[ok], axis=1))) if ok.sum() else np.nan

print()
print("=" * 78)
print("(2) real FT 이득 — 같은 프레임 짝지은 차이 (synth - n_ft), 양수 = FT 가 좋음")
print(f"    SEALED(unseen 세션) {len(sealed)} 프레임, 세션 {len(set(gt[f][2] for f in sealed))}개")
print("=" * 78)
print(f"{'앙각':8s} {'N':>4s} {'세션':>5s} {'중앙차 px':>10s} {'프레임 95%CI':>20s} {'세션클러스터 95%CI':>22s} {'>0 비율':>8s}")
ft_rows = []
for k, sel in (("<8", lambda e: e < 8), ("8-15", lambda e: 8 <= e < 15), ("전체", lambda e: True)):
    fs = [f for f in sealed if sel(gt[f][1])]
    if not fs: continue
    d = {}; units = collections.defaultdict(list)
    for f in fs:
        pts, _, sess = gt[f]
        diff = merr(M["yolo26n_synth"][f], pts) - merr(M["yolo26n_ft"][f], pts)
        d[f] = diff; units[sess].append(diff)
    vals = list(d.values())
    lo, hi = frame_boot(vals, med)
    clo, chi = cluster_boot(units, med)
    print(f"{k:8s} {len(fs):4d} {len(units):5d} {np.median(vals):9.2f} "
          f"[{lo:8.2f}, {hi:7.2f}] [{clo:9.2f}, {chi:8.2f}] {np.mean(np.array(vals)>0)*100:7.1f}%")
    ft_rows.append(dict(bin=k, n=len(fs), n_sessions=len(units), median_diff=float(np.median(vals)),
                        frame_ci=[lo, hi], cluster_ci=[clo, chi],
                        frac_better=float(np.mean(np.array(vals) > 0))))

# ── (3) 세션별 개별 결과 ────────────────────────────────────────────
print()
print("=" * 78)
print("(3) 세션별 개별 결과 — 클러스터가 4개뿐이라 CI 를 믿기 전에 이걸 본다")
print("=" * 78)
print(f"{'세션':16s} {'N':>4s} {'앙각 p50':>9s} {'synth med':>10s} {'n_ft med':>9s} {'차이':>8s}")
sess_rows = []
for s in sorted(set(gt[f][2] for f in sealed)):
    fs = [f for f in sealed if gt[f][2] == s]
    a = np.median([merr(M["yolo26n_synth"][f], gt[f][0]) for f in fs])
    b = np.median([merr(M["yolo26n_ft"][f], gt[f][0]) for f in fs])
    e = np.median([gt[f][1] for f in fs])
    print(f"{s:16s} {len(fs):4d} {e:8.1f}° {a:10.2f} {b:9.2f} {a-b:7.2f}")
    sess_rows.append(dict(session=s, n=len(fs), elev_p50=float(e),
                          synth_med=float(a), ft_med=float(b), diff=float(a - b)))

# ── (4) 고앙각 악화 주장의 표본 ─────────────────────────────────────
print()
print("=" * 78)
print("(4) '>=30도에서 FT 가 악화' 주장의 표본 — 정본 140 전체(SEALED 아님)")
print("=" * 78)
hi30 = [f for f in common if gt[f][1] >= 30]
units = collections.defaultdict(list)
for f in hi30:
    units[gt[f][2]].append(merr(M["yolo26n_synth"][f], gt[f][0]) - merr(M["yolo26n_ft"][f], gt[f][0]))
vals = [v for u in units.values() for v in u]
lo, hi = frame_boot(vals, med); clo, chi = cluster_boot(units, med)
print(f"  N={len(hi30)}  세션={len(units)}  중앙차 {np.median(vals):+.2f} px  "
      f"프레임 95%CI [{lo:+.2f}, {hi:+.2f}]  세션CI [{clo:+.2f}, {chi:+.2f}]")
print(f"  세션 분포: {dict(collections.Counter(gt[f][2] for f in hi30))}")

json.dump(dict(bootstrap_B=B, seed=20260906, nme_threshold=NME_T,
               elevation_failure=elev_rows, ft_paired_gain=ft_rows, per_session=sess_rows,
               high_elev_regression=dict(n=len(hi30), n_sessions=len(units),
                                         median_diff=float(np.median(vals)),
                                         frame_ci=[lo, hi], cluster_ci=[clo, chi])),
          open(OUT / "UNCERTAINTY.json", "w"), indent=1, ensure_ascii=False)
print(f"\nwrote {OUT/'UNCERTAINTY.json'}")
