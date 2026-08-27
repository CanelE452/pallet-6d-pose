"""REAL SYMMETRY-EQUIVALENT EVALUATION — A0 vs 후보, camera-facing GT 기준.

★ canonical semantic accuracy 가 아니다.  real manual GT 에는 fixed-object perm_v4 가
  없으므로, 허용 순열군 G 위의 **최소값**으로 기하 정확도만 본다.
  G 는 synthetic 40,000 프레임 perm_v4 census 에서 유도했고 (PERMUTATION_GROUP.json)
  위수 4의 군임을 검증했다.  real GT 를 보고 고른 것이 아니다.
★ |G|=4 이므로 프레임마다 4개 중 최선을 고른다 -> 낙관적 하한.  A0 와 후보에 동일
  적용하므로 비교는 공정하다.
"""
import argparse, glob, hashlib, json, os, sys
import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, ROOT)
sys.path.insert(0, f"{ROOT}/challenge")
OUT = f"{ROOT}/challenge/yolo_pose_one_model/runs_pallet_loss/overnight_20260823"
MANI = ("/home/minjae/pallet_worker_transfer_20260821T105141Z/REAL_GT_QA_20260821T133405Z/"
        "REVIEWED_CLEAN_REALDEV_V2_MANIFEST.json")

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True)          # 예: ASC
ap.add_argument("--weights", required=True)
ap.add_argument("--baseline", required=True)     # A0 last.pt
ap.add_argument("--conf", type=float, default=0.4)
ap.add_argument("--pad", type=int, default=100)
A = ap.parse_args()

import cv2
import torch
from ultralytics import YOLO
import data_paths as DP

# ★ 검증된 real 추론 recipe (analysis_pre_v2/_cc_raw_dump.json).
#   reflect padding 없이 돌리면 truncation·근접에서 체계적 과소검출이 생겨
#   모델/도메인을 오진한다 (과거 2회 교정된 함정).
PAD = 100
BORDER = "BORDER_REFLECT_101"

G = [tuple(p) for p in json.load(open(f"{OUT}/PERMUTATION_GROUP.json"))["G"]]

# ---- membership -------------------------------------------------------------
# ★ data_paths 는 **repo 상대경로**를 준다.  큐가 다른 cwd 에서 돌면 glob 이 0 개가
#   된다(2026-08-23 실제 발생).  manifest 가 image/label 경로를 직접 갖고 있으므로
#   ROOT 와 join 해서 쓴다 — cwd 무관.
man = json.load(open(MANI))
rows = []
for it in man["items"]:
    jp = os.path.join(ROOT, it["label"])
    ip = os.path.join(ROOT, it["image"])
    if os.path.exists(jp) and os.path.exists(ip):
        rows.append((it.get("set", "?"), jp, ip))
if not rows:
    raise SystemExit(f"membership 매칭 0 — manifest {len(man['items'])}개 중 파일 0")


def sha(paths):
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(os.path.basename(p).encode())
        with open(p, "rb") as f:
            h.update(hashlib.sha256(f.read()).hexdigest().encode())
    return h.hexdigest()[:16]


def img_of(j):
    for ext in (".png", ".jpg", ".jpeg"):
        p = os.path.splitext(j)[0] + ext
        if os.path.exists(p):
            return p
    return None


lock = {"manifest": MANI, "manifest_n": man.get("n_total"), "matched": len(rows),
        "role": "EXPLORATORY_REAL_EVAL — FINAL TEST 아님",
        "gt_sha": sha([r[1] for r in rows]), "rgb_sha": sha([r[2] for r in rows]),
        "permutation_group_size": len(G), "conf": A.conf,
        "recipe": {"pad": A.pad, "border": BORDER, "imgsz": 640,
                   "selection": "top-1 by box conf",
                   "source": "analysis_pre_v2/_cc_raw_dump.json 검증본"}}
json.dump(lock, open(f"{OUT}/REAL_SOURCE_LOCK.json", "w"), indent=2, ensure_ascii=False)


def predict(w):
    m = YOLO(w, task="pose")
    out = {}
    for sess, jp, ip in rows:
        im = cv2.imread(ip)
        pim = cv2.copyMakeBorder(im, A.pad, A.pad, A.pad, A.pad, cv2.BORDER_REFLECT_101)
        r = m.predict(pim, conf=A.conf, imgsz=640, device=0, verbose=False)[0]
        stem = os.path.splitext(os.path.basename(jp))[0]
        if r.keypoints is None or len(r.boxes) == 0:
            out[stem] = None
            continue
        i = int(np.argmax(r.boxes.conf.cpu().numpy()))
        out[stem] = r.keypoints.xy.cpu().numpy()[i] - A.pad   # 원본 좌표계로 복귀
    del m
    torch.cuda.empty_cache()
    return out


def gt_of(jp):
    o = json.load(open(jp))["objects"][0]
    pc = np.array(o["projected_cuboid"], dtype=float)
    c = o.get("projected_cuboid_centroid")
    return pc[:8], (np.array(c, dtype=float) if c is not None else None)


def stats(pred, base_pred):
    """paired detected intersection 에서 지표."""
    rec = []
    for sess, jp, _ in rows:
        s = os.path.splitext(os.path.basename(jp))[0]
        gt8, gc = gt_of(jp)
        vis = np.isfinite(gt8).all(1)
        r = {"stem": s, "session": sess, "det_a": pred.get(s) is not None,
             "det_b": base_pred.get(s) is not None}
        for tag, P in (("a", pred.get(s)), ("b", base_pred.get(s))):
            if P is None or len(P) < 9:
                continue
            best, bc = None, None
            for perm in G:
                q = P[list(perm)]
                e = np.linalg.norm(q[:8][vis] - gt8[vis], axis=1)
                mv = float(np.mean(e))
                if best is None or mv < best[0]:
                    best = (mv, e)
            r[f"{tag}_mean"] = best[0]
            r[f"{tag}_med"] = float(np.median(best[1]))
            r[f"{tag}_e"] = best[1].tolist()
            if gc is not None:
                r[f"{tag}_cen"] = float(np.linalg.norm(P[8] - gc))
        rec.append(r)
    return rec


pa, pb = predict(A.weights), predict(A.baseline)
rec = stats(pa, pb)
both = [r for r in rec if "a_e" in r and "b_e" in r]


def agg(rs, t):
    e = np.concatenate([r[f"{t}_e"] for r in rs])
    cen = [r[f"{t}_cen"] for r in rs if f"{t}_cen" in r]
    return {"n_frames": len(rs), "n_corners": int(e.size),
            "corner_median": float(np.median(e)), "corner_p90": float(np.percentile(e, 90)),
            "gross20": float((e > 20).mean()), "gross40": float((e > 40).mean()),
            "centroid_median": float(np.median(cen)) if cen else None,
            "centroid_p90": float(np.percentile(cen, 90)) if cen else None}


det_a = float(np.mean([r["det_a"] for r in rec]))
det_b = float(np.mean([r["det_b"] for r in rec]))
SA, SB = agg(both, "a"), agg(both, "b")


def boot(key, fn):
    da = np.array([fn(np.array(r["a_e"])) for r in both])
    db = np.array([fn(np.array(r["b_e"])) for r in both])
    d = db - da                                   # >0 이면 후보(a)가 우세
    rng = np.random.default_rng(0)
    bs = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
    return {"delta_mean": float(d.mean()),
            "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}


B = {"corner_median": boot("m", lambda e: float(np.median(e))),
     "corner_p90": boot("p", lambda e: float(np.percentile(e, 90))),
     "gross20": boot("g", lambda e: float((e > 20).mean()))}


def rel(b, a):
    return (b - a) / max(abs(b), 1e-12)


imp_p90 = rel(SB["corner_p90"], SA["corner_p90"])
red_g20 = rel(SB["gross20"], SA["gross20"]) if SB["gross20"] > 0 else 0.0
med_reg = rel(SB["corner_median"], SA["corner_median"])       # 음수면 악화
det_reg = det_a - det_b
gate_hit = (imp_p90 >= 0.05) or (red_g20 >= 0.10)
guard = (med_reg >= -0.02) and (det_reg >= -0.02)
if gate_hit and guard:
    ci_ok = (B["corner_p90"]["ci95"][0] > 0) or (B["gross20"]["ci95"][0] > 0)
    verdict = "POSITIVE" if ci_ok else "POINT_ESTIMATE_POSITIVE_UNCERTAIN"
else:
    verdict = "NO_REAL_SIGNAL"

res = {"arm": A.arm, "verdict": verdict, "role": "EXPLORATORY_REAL_EVAL",
       "n_paired": len(both), "detection_rate": {"A0": det_b, A.arm: det_a},
       "A0": SB, A.arm: SA, "bootstrap_delta_A0_minus_arm": B,
       "relative": {"p90_improvement": imp_p90, "gross20_reduction": red_g20,
                    "median_regression": med_reg, "detection_regression_pp": det_reg},
       "gate": ("(p90 >=5% 개선 OR gross20 >=10% 감소) AND median 악화 <=2% "
                "AND detection 악화 <=2pp"),
       "permutation_group_size": len(G),
       "★caveat": ("허용 순열군 위 최소값이라 낙관적 하한. canonical semantic accuracy "
                   "가 아니다. A0 와 동일 적용."),
       "source_lock": lock}
json.dump(res, open(f"{OUT}/REAL_{A.arm}_VERDICT.json", "w"), indent=2, ensure_ascii=False)
json.dump({"per_frame": rec}, open(f"{OUT}/A0_{A.arm}_REAL_EQUIV.json", "w"))
md = ["```", f"{'':16} {'A0':>12} {A.arm:>12}", "-" * 44,
      f"{'detection':16} {det_b:12.4f} {det_a:12.4f}"]
for k in ("corner_median", "corner_p90", "gross20", "gross40", "centroid_median"):
    vb, va = SB.get(k), SA.get(k)
    md.append(f"{k:16} {('-' if vb is None else f'{vb:12.4f}')} "
              f"{('-' if va is None else f'{va:12.4f}')}")
md += ["```", "", f"paired {len(both)} 프레임  bootstrap 10,000  (Δ = A0 − {A.arm}, >0 이면 {A.arm} 우세)"]
for k, v in B.items():
    md.append(f"- {k}: Δ {v['delta_mean']:+.4f}  95%CI [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]")
md += ["", f"**REAL_{A.arm}_SIGNAL = {verdict}**", "",
       f"|G| = {len(G)} 순열군 위 최소값 — canonical semantic accuracy 아님. EXPLORATORY."]
open(f"{OUT}/A0_{A.arm}_REAL_EQUIV.md", "w").write("\n".join(md))
print("\n".join(md))
