"""추론 시점에만 쓸 수 있는 신호로 R0 의 실패를 거부할 수 있는가.

목적 : 선택지 C(기존 point model 유지 + inference/selection 개선)의 헤드룸을 잰다.
지표 : 실패(= identity 최대 코너 오차 > 25 px, 기저율 30.7%)에 대한 AUROC 와
       selective-risk curve. GT 를 쓰는 신호는 오라클로 따로 표기하고 판정에서 뺀다.

입력은 전부 기존 artifact — 새 추론 0 회.
  multiteacher_corner_distill_v1/predictions/T0_R0_YOLO26N_G38LEGACY.json  (R0 예측 9kp + conf)
  paper_eval_v1/AXIS_FAILURES.json                                          (실패 라벨)
"""
import json, pathlib, numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
P = ROOT / "data/pallet/results/multiteacher_corner_distill_v1/predictions/T0_R0_YOLO26N_G38LEGACY.json"
A = ROOT / "data/pallet/results/paper_eval_v1/AXIS_FAILURES.json"
OUT = ROOT / "data/pallet/results/accuracy_root_cause_v1"

pred = json.load(open(P))["frames"]
ax = json.load(open(A))["models"]["R0"]
print(f"R0 예측 프레임 {len(pred)} · AXIS_FAILURES 프레임 {len(ax)}")


def seg_int(p1, p2, p3, p4):
    """선분 p1p2 와 p3p4 를 담은 두 직선의 교점. 평행이면 None."""
    d1, d2 = p2 - p1, p4 - p3
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-9:
        return None
    t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / den
    return p1 + t * d1


rows = []
for fid, e in ax.items():
    key = fid.replace(":", "__")
    q = pred.get(key)
    if q is None or q.get("status") != "OK":
        continue
    k = np.asarray(q["keypoints_xy"], float)
    c = np.asarray(q["keypoints_conf"], float)
    b = np.asarray(q["box_xyxy"], float)
    diag = float(np.hypot(b[2] - b[0], b[3] - b[1]))
    if diag <= 0:
        continue
    # 1) centroid 자기일관성 — 예측 centroid keypoint 와 예측 코너 8개 평균의 어긋남
    cen = float(np.linalg.norm(k[8] - k[:8].mean(axis=0))) / diag
    # 2) 공간 대각 교점 — 0-6, 1-7, 2-4, 3-5 는 centroid 에서 만나야 한다
    dsp = []
    for (a1, a2), (b1, b2) in (((0, 6), (1, 7)), ((0, 6), (2, 4)), ((0, 6), (3, 5)),
                               ((1, 7), (2, 4)), ((1, 7), (3, 5)), ((2, 4), (3, 5))):
        x = seg_int(k[a1], k[a2], k[b1], k[b2])
        if x is not None:
            dsp.append(np.linalg.norm(x - k[8]))
    diagc = float(np.median(dsp)) / diag if dsp else np.nan
    # 3) 앞면(0123)·뒷면(4567) 대각 교점이 각 면 중심에 오는가
    face = []
    for f in ((0, 1, 2, 3), (4, 5, 6, 7)):
        x = seg_int(k[f[0]], k[f[2]], k[f[1]], k[f[3]])
        if x is not None:
            face.append(np.linalg.norm(x - k[list(f)].mean(axis=0)))
    facec = float(np.median(face)) / diag if face else np.nan
    # 4) 연결선(0-4,1-5,2-6,3-7) 길이의 산포 — 직육면체면 투영에서도 비슷해야 한다
    con = np.array([np.linalg.norm(k[i] - k[i + 4]) for i in range(4)])
    conv = float(con.std() / max(con.mean(), 1e-6))
    rows.append(dict(fid=fid, fail=e["identity_max_px"] > 25.0,
                     box_conf=float(q["box_conf"]),
                     kp_conf_min=float(c.min()), kp_conf_mean=float(c.mean()),
                     kp_conf_corner_min=float(c[:8].min()),
                     centroid_selfconsist=cen, space_diag=diagc,
                     face_diag=facec, connector_cv=conv))

print(f"매칭·유효 프레임 {len(rows)}")
y = np.array([r["fail"] for r in rows])
print(f"실패 기저율 {y.mean()*100:.1f}%\n")


def auc(s, y):
    s = np.asarray(s, float)
    ok = np.isfinite(s)
    s, yy = s[ok], np.asarray(y)[ok]
    o = np.argsort(s); yy = yy[o]
    r = np.arange(1, len(yy) + 1); pos = yy.sum(); neg = len(yy) - pos
    if pos == 0 or neg == 0:
        return np.nan, len(yy)
    return (r[yy == 1].sum() - pos * (pos + 1) / 2) / (pos * neg), len(yy)


print("=== 배포 가능 신호의 실패 예측력 (높을수록 실패를 잘 가리킴) ===")
sig = [("-box_conf", [-r["box_conf"] for r in rows]),
       ("-kp_conf_min", [-r["kp_conf_min"] for r in rows]),
       ("-kp_conf_corner_min", [-r["kp_conf_corner_min"] for r in rows]),
       ("-kp_conf_mean", [-r["kp_conf_mean"] for r in rows]),
       ("centroid 자기일관성", [r["centroid_selfconsist"] for r in rows]),
       ("공간대각 교점 산포", [r["space_diag"] for r in rows]),
       ("면대각 교점 산포", [r["face_diag"] for r in rows]),
       ("연결선 길이 변동계수", [r["connector_cv"] for r in rows])]
for n, s in sig:
    a, m = auc(s, y)
    print(f"  {n:22s} AUROC {a:.3f}   (n={m})")

best = max(sig, key=lambda kv: (auc(kv[1], y)[0] if np.isfinite(auc(kv[1], y)[0]) else 0))
print(f"\n=== selective-risk curve — 최고 신호 '{best[0]}' 로 거부 ===")
s = np.asarray(best[1], float)
s = np.where(np.isfinite(s), s, np.nanmax(s[np.isfinite(s)]))
order = np.argsort(s)          # 낮은 위험부터 채택
print(f"{'coverage':>9s} {'accepted gross%':>16s} {'거부분 실패%':>13s}")
for cov in (1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5):
    k = int(round(cov * len(rows)))
    acc, rej = order[:k], order[k:]
    print(f"{cov:9.2f} {y[acc].mean()*100:15.1f}% "
          f"{(y[rej].mean()*100 if len(rej) else float('nan')):12.1f}%")

import csv
OUT.mkdir(parents=True, exist_ok=True)
with open(OUT / "R0_SELECTIVE_SIGNALS.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"\nwrote {OUT/'R0_SELECTIVE_SIGNALS.csv'} ({len(rows)} rows)")
