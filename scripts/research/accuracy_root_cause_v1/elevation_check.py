"""실패가 저앙각(edge-on)에 몰리는지 메인 세션에서 독립 재현한다.

목적 : 다음 실험을 "저앙각 조건" 에 걸어도 되는지 확인.
지표 : GT pose 로 계산한 카메라 앙각 대 실패율(NME 기준). 투영 크기와의 교란을 분리.
정의 : 팔레트 상판 법선 n = R @ (0,-1,0) (object frame 은 Y=아래 +).
       팔레트→카메라 방향 u = -t/|t| (카메라가 원점).
       앙각 = asin(|n·u|)  — 상판 평면 위로 카메라가 몇 도 떠 있는가. 0도 = 완전 edge-on.
읽기 전용, 새 추론 0 회.
"""
import csv, json, pathlib, collections
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
WS = ROOT / "data/evaluation/pallet_eval_v1"
ax = json.load(open(ROOT / "data/pallet/results/paper_eval_v1/AXIS_FAILURES.json"))["models"]["R0"]
rows = {r["frame_id"]: r for r in csv.DictReader(
    open(ROOT / "data/pallet/results/paper_eval_v1/arms/R0_per_frame.csv")) if r["kind"] == "POSITIVE"}
ann = {(p.parent.name, p.stem): p for p in WS.rglob("annotations/*/*.json")}

rec = []
for fid, e in ax.items():
    sess, stem = fid.split(":", 1)
    p = ann.get((sess, stem))
    if p is None or fid not in rows:
        continue
    ob = json.loads(p.read_text())["objects"][0]
    T = np.asarray(ob["pose_transform"], float)
    R, t = T[:3, :3], T[:3, 3]
    if np.linalg.norm(t) < 1e-9:
        continue
    n = R @ np.array([0.0, -1.0, 0.0])
    u = -t / np.linalg.norm(t)
    elev = float(np.degrees(np.arcsin(np.clip(abs(float(n @ u)), 0, 1))))
    r = rows[fid]
    w = float(r["top_box_x2"]) - float(r["top_box_x1"]); h = float(r["top_box_y2"]) - float(r["top_box_y1"])
    diag = float(np.hypot(w, h))
    rec.append(dict(fid=fid, sess=sess, elev=elev, diag=diag, domain=r["domain"],
                    obj=r["object_type"].split("_")[0],
                    mx=e["identity_max_px"], nme=e["identity_max_px"] / diag,
                    dist=float(np.linalg.norm(t))))

print(f"N = {len(rec)}")
NME_T = 0.0747   # FAILURE_DECOMPOSITION.md 와 같은 임계 [추정][미검증]
y = np.array([x["nme"] > NME_T for x in rec])
print(f"NME 실패 기저율 {y.mean()*100:.1f}%  (임계 {NME_T})\n")

def tab(title, keyfn, order=None):
    g = collections.defaultdict(list)
    for x in rec:
        g[keyfn(x)].append(x)
    ks = order or sorted(g, key=str)
    print(f"--- {title} ---")
    print(f"{'bin':16s} {'N':>4s} {'NME 실패':>9s} {'px>25 실패':>11s} {'bbox대각 p50':>12s} {'거리m p50':>10s}")
    for k in ks:
        if k not in g: continue
        v = g[k]
        nf = np.mean([x["nme"] > NME_T for x in v]) * 100
        pf = np.mean([x["mx"] > 25 for x in v]) * 100
        print(f"{str(k):16s} {len(v):4d} {nf:8.1f}% {pf:10.1f}% "
              f"{np.median([x['diag'] for x in v]):12.1f} {np.median([x['dist'] for x in v]):10.2f}")
    print()

EB = ["<3", "3-8", "8-15", "15-30", ">=30"]
def ebin(x):
    e = x["elev"]
    return "<3" if e < 3 else "3-8" if e < 8 else "8-15" if e < 15 else "15-30" if e < 30 else ">=30"
tab("카메라 앙각(도)", ebin, EB)

med = np.median([x["diag"] for x in rec])
tab("앙각 x 투영크기 (교란 분리)",
    lambda x: f"{ebin(x)} / {'대' if x['diag']>=med else '소'}",
    [f"{e} / {s}" for e in EB for s in ("소", "대")])
tab("조명 x 앙각", lambda x: f"{x['domain'] or 'none'} / {'<15' if x['elev']<15 else '>=15'}")

e = np.array([x["elev"] for x in rec])
print(f"real 앙각 분포: p10 {np.percentile(e,10):.1f} p50 {np.median(e):.1f} p90 {np.percentile(e,90):.1f} "
      f"· <8도 비율 {(e<8).mean()*100:.1f}% · <15도 비율 {(e<15).mean()*100:.1f}%")
d = np.array([x["dist"] for x in rec])
print(f"real 거리 분포: p10 {np.percentile(d,10):.2f} p50 {np.median(d):.2f} p90 {np.percentile(d,90):.2f} m "
      f"· <1.5m 비율 {(d<1.5).mean()*100:.1f}%")

with open(ROOT / "data/pallet/results/accuracy_root_cause_v1/R0_ELEVATION_FAILURE.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rec[0].keys())); w.writeheader(); w.writerows(rec)
print("\nwrote data/pallet/results/accuracy_root_cause_v1/R0_ELEVATION_FAILURE.csv")
