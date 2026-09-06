"""real fine-tune 의 이득이 저앙각(edge-on) 레짐에서 나오는가.

목적 : 다음 실험을 (H) real supervision 에 걸지 (F) 저앙각 자산 다양성 재렌더에 걸지 가른다.
지표 : 앙각 구간별로 synth 대비 FT 의 코너 오차 개선. 저앙각에서 이득이 크면
       "저앙각 실패는 데이터로 고쳐진다" 가 실증되고, 저앙각에서만 이득이 없으면
       거기가 real supervision 으로도 안 되는 구간이라는 뜻이다.
읽기 전용, 새 추론 0 회. 정본 140장(`objects[0].split=="eval"`) 위에서만 비교한다.
"""
import json, pathlib, collections, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from challenge.data_paths import EVAL_CANONICAL
MC = ROOT / "data/pallet/results/model_compare"

# 정본 GT: fid -> (kp GT 9x2, 앙각, in_frame mask)
gt = {}
for name, rel in EVAL_CANONICAL.items():
    for jp in sorted((ROOT / rel).glob("*.json")):
        o = json.loads(jp.read_text()); objs = o.get("objects") or []
        if not objs or objs[0].get("split") != "eval":
            continue
        ob = objs[0]
        mk = ob.get("manual_kps")
        if mk is None:
            continue
        pts = np.asarray([[-1.0, -1.0] if v is None else [float(v[0]), float(v[1])] for v in mk], float)
        T = np.asarray(ob["pose_transform"], float); R, t = T[:3, :3], T[:3, 3]
        n = R @ np.array([0.0, -1.0, 0.0]); u = -t / max(np.linalg.norm(t), 1e-9)
        elev = float(np.degrees(np.arcsin(np.clip(abs(float(n @ u)), 0, 1))))
        gt[jp.stem] = (pts, elev, float(np.linalg.norm(t)))
print(f"정본 GT 프레임 {len(gt)}")


def load(tag):
    d = json.load(open(MC / f"kps_{tag}.json"))
    out = {}
    for f in d["frames"]:
        k = f.get("kps")
        if k:
            out[f["fid"]] = np.asarray(k, float)
    return out


models = {t: load(t) for t in ("yolo26n_synth", "yolo26n_ft", "yolo26m_ft")}
for t, m in models.items():
    print(f"  {t:16s} 검출 프레임 {len(m)}")

common = set(gt) & set.intersection(*[set(m) for m in models.values()])
print(f"세 모델 모두 검출 + 정본 GT = {len(common)} 프레임\n")


def med_err(pred, g):
    """GT sentinel(-1,-1) 을 뺀 코너의 오차 중앙값·최댓값."""
    ok = ~((g[:, 0] == -1) & (g[:, 1] == -1))
    if ok.sum() == 0:
        return np.nan, np.nan
    e = np.linalg.norm(pred[:len(g)][ok] - g[ok], axis=1)
    return float(np.median(e)), float(np.max(e))


EB = ["<3", "3-8", "8-15", "15-30", ">=30"]
def ebin(e):
    return "<3" if e < 3 else "3-8" if e < 8 else "8-15" if e < 15 else "15-30" if e < 30 else ">=30"

g = collections.defaultdict(list)
for fid in common:
    pts, elev, dist = gt[fid]
    row = {t: med_err(models[t][fid], pts) for t in models}
    g[ebin(elev)].append(row)

print("앙각 구간별 코너 오차 중앙값 (px) — 정본 140장 중 세 모델 공통 검출분")
print(f"{'앙각':8s} {'N':>4s} {'synth':>9s} {'n_ft':>9s} {'m_ft':>9s} {'n_ft 개선':>10s} {'m_ft 개선':>10s}")
for k in EB:
    v = g.get(k)
    if not v: continue
    s = np.median([x["yolo26n_synth"][0] for x in v])
    n = np.median([x["yolo26n_ft"][0] for x in v])
    m = np.median([x["yolo26m_ft"][0] for x in v])
    print(f"{k:8s} {len(v):4d} {s:9.2f} {n:9.2f} {m:9.2f} "
          f"{(s-n)/s*100:9.1f}% {(s-m)/s*100:9.1f}%")

allv = [x for v in g.values() for x in v]
s = np.median([x["yolo26n_synth"][0] for x in allv])
n = np.median([x["yolo26n_ft"][0] for x in allv])
m = np.median([x["yolo26m_ft"][0] for x in allv])
print(f"{'전체':8s} {len(allv):4d} {s:9.2f} {n:9.2f} {m:9.2f} {(s-n)/s*100:9.1f}% {(s-m)/s*100:9.1f}%")

print("\n같은 표, 프레임 최대 코너 오차 기준 (gross 를 보는 눈)")
print(f"{'앙각':8s} {'N':>4s} {'synth':>9s} {'n_ft':>9s} {'m_ft':>9s} {'synth>25':>9s} {'n_ft>25':>9s} {'m_ft>25':>9s}")
for k in EB:
    v = g.get(k)
    if not v: continue
    a = [np.median([x[t][1] for x in v]) for t in ("yolo26n_synth", "yolo26n_ft", "yolo26m_ft")]
    b = [np.mean([x[t][1] > 25 for x in v]) * 100 for t in ("yolo26n_synth", "yolo26n_ft", "yolo26m_ft")]
    print(f"{k:8s} {len(v):4d} {a[0]:9.2f} {a[1]:9.2f} {a[2]:9.2f} {b[0]:8.1f}% {b[1]:8.1f}% {b[2]:8.1f}%")
