#!/usr/bin/env python3
"""STAGE22 PART A: "윗면 보임" 앙각 실측 (학습·재추론 X).

목적: "윗면이 잘 보이는데 왜 rear(4-7) 가 안 되나" 의 데이터 답.
  real 프레임별 실측 elevation(GT pose) + 윗면 밴드두께(px) + front/rear median 오차.

재사용:
  - stage17 corner01_diagnosis.json : e_corner, gt8, pred8 (real 98)
  - stage18 elev_from_pose / build_path_lut : GT pose -> elevation

윗면 밴드두께 px = |mean_y(kp4,5) - mean_y(kp0,1)|  (GT 2D)
  top-front edge(0,1) <-> top-rear edge(4,5) 의 화면상 수직간격 근사.
  ★수평 팔레트 근사 — 크게 기운 프레임은 예외표기.
Convention: camera-facing 0123 v4. 0-3=front, 4-7=rear, {0,1,4,5}=top.
"""
import os as _os, sys as _sys

# --- stage0 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_S0 = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_S0] + [_os.path.join(_S0, _d) for _d in sorted(_os.listdir(_S0))
                         if _os.path.isdir(_os.path.join(_S0, _d)) and not _d.startswith(".")]

import os, sys, json, glob
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "stage0"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "data_prep", "eval"))

sys.path[:0] = [os.path.join(ROOT, "scripts", "data_prep", _s)
                for _s in ("plots", "filters")]
from stage18_elevation_threshold import build_path_lut, gt_pose, elev_from_pose  # noqa
from tau_calibrate import collect_val_frames  # noqa
from eval_pvnet_heads import collect_manual, collect_syn  # noqa

STAGE17 = os.path.join(ROOT, "data", "pallet", "eval_results",
                       "stage17_corner01_diagnosis", "corner01_diagnosis.json")
CAD_DIR = os.path.join(ROOT, "challenge", "data", "capturepalletcad_manual_gt")
OUT = os.path.join(ROOT, "data", "pallet", "eval_results", "stage22_coord_diag", "partA")

FRONT = [0, 1, 2, 3]
REAR = [4, 5, 6, 7]
TOP_FRONT = [0, 1]
TOP_REAR = [4, 5]
SPIKE_PX = 15.0
BINS = [(-90, 3), (3, 8), (8, 15), (15, 25), (25, 90)]
BIN_LBL = ["<3", "3-8", "8-15", "15-25", "25+"]


def build_img_lut():
    """(dom,fid) -> image path. mirror stage18 collectors + CAD."""
    lut = {}
    for dom, fid, jp, ip in collect_val_frames():
        lut[(dom, str(fid))] = ip
    for _, fid, jp, ip in collect_manual():
        lut[("manual", str(fid))] = ip
    for jp in sorted(glob.glob(os.path.join(CAD_DIR, "*.json"))):
        fid = os.path.splitext(os.path.basename(jp))[0]
        for ext in (".png", ".jpg", ".jpeg"):
            ip = os.path.join(CAD_DIR, fid + ext)
            if os.path.exists(ip):
                lut[("cad", str(fid))] = ip
                break
    return lut


def band_thickness(gt8):
    g = np.array(gt8, float)
    y_tf = np.mean(g[TOP_FRONT, 1])
    y_tr = np.mean(g[TOP_REAR, 1])
    return float(abs(y_tf - y_tr))


def tilt_flag(gt8):
    """수평 팔레트 근사 위반 감지: top-front edge(0-1) 의 화면상 기울기(도)."""
    g = np.array(gt8, float)
    dx = g[1, 0] - g[0, 0]
    dy = g[1, 1] - g[0, 1]
    return float(abs(np.degrees(np.arctan2(dy, dx))))


def bin_of(e):
    for i, (lo, hi) in enumerate(BINS):
        if lo <= e < hi:
            return i
    return len(BINS) - 1


def agg(vals):
    a = np.array([v for v in vals if v is not None], float)
    if a.size == 0:
        return {"n": 0, "median": None, "p90": None}
    return {"n": int(a.size), "median": round(float(np.median(a)), 2),
            "p90": round(float(np.percentile(a, 90)), 2)}


def main():
    os.makedirs(OUT, exist_ok=True)
    recs = json.load(open(STAGE17))["records"]
    real = [r for r in recs if r["is_real"]]
    lut = build_path_lut()
    img_lut = build_img_lut()

    enriched = []
    for r in real:
        jp = lut.get((r["dom"], str(r["fid"])))
        if not jp:
            continue
        T = gt_pose(jp)
        if T is None:
            continue
        elev = elev_from_pose(T)
        if elev is None:
            continue
        ec = r["e_corner"]
        ef = [ec[i] for i in FRONT if ec[i] is not None]
        er = [ec[i] for i in REAR if ec[i] is not None]
        if not ef or not er:
            continue
        e2 = {
            "dom": r["dom"], "fid": r["fid"],
            "elev": round(float(elev), 2),
            "band_px": round(band_thickness(r["gt8"]), 1),
            "tilt_deg": round(tilt_flag(r["gt8"]), 1),
            "e_front": round(float(np.mean(ef)), 2),
            "e_rear": round(float(np.mean(er)), 2),
            "rear_spike": bool(np.mean(er) > SPIKE_PX),
            "n_det": r["n_det"],
            "img": img_lut.get((r["dom"], str(r["fid"]))),
            "gt8": r["gt8"], "pred8": r["pred8"], "e_corner": ec,
        }
        enriched.append(e2)

    print(f"[partA] enriched real {len(enriched)}/{len(real)}")

    # ---- table by elevation bin ----
    table = {}
    for i, lbl in enumerate(BIN_LBL):
        g = [r for r in enriched if bin_of(r["elev"]) == i]
        rspk = sum(1 for r in g if r["rear_spike"])
        table[lbl] = {
            "n": len(g),
            "band_px_med": agg([r["band_px"] for r in g])["median"],
            "band_px_p90": agg([r["band_px"] for r in g])["p90"],
            "front_med": agg([r["e_front"] for r in g])["median"],
            "rear_med": agg([r["e_rear"] for r in g])["median"],
            "rear_spike_rate": round(rspk / len(g), 3) if g else None,
            "elev_span": [round(min(r["elev"] for r in g), 1),
                          round(max(r["elev"] for r in g), 1)] if g else None,
        }

    # correlation elev<->band, band<->rear_err
    def sp(x, y):
        try:
            from scipy.stats import spearmanr
            m = [i for i in range(len(x)) if x[i] is not None and y[i] is not None]
            return round(float(spearmanr([x[i] for i in m], [y[i] for i in m])[0]), 3)
        except Exception:
            return None
    els = [r["elev"] for r in enriched]
    bnd = [r["band_px"] for r in enriched]
    rer = [r["e_rear"] for r in enriched]
    corr = {"spearman_elev_band": sp(els, bnd),
            "spearman_band_rearerr": sp(bnd, rer),
            "spearman_elev_rearerr": sp(els, rer)}

    result = {"n": len(enriched), "spike_px": SPIKE_PX,
              "bins": BIN_LBL, "table": table, "corr": corr,
              "band_def": "|mean_y(kp4,5)-mean_y(kp0,1)| GT2D; top-front<->top-rear vertical screen gap. horizontal-pallet approx.",
              "records": enriched}
    json.dump(result, open(os.path.join(OUT, "partA.json"), "w"), indent=1)

    # ---- overlays: ~10 frames spanning elevation ----
    make_overlays(enriched)

    # ---- summary.md ----
    write_summary(result, enriched)
    print(f"[partA] done -> {OUT}")
    return result


def make_overlays(enriched, n_want=10):
    try:
        import cv2
    except Exception as e:
        print("[partA] no cv2, skip overlays:", e)
        return
    ov_dir = os.path.join(OUT, "overlays")
    os.makedirs(ov_dir, exist_ok=True)
    # pick spanning elevation, prefer frames with image
    with_img = [r for r in enriched if r["img"] and os.path.exists(r["img"])]
    with_img.sort(key=lambda r: r["elev"])
    if not with_img:
        print("[partA] no images for overlay")
        return
    idx = np.linspace(0, len(with_img) - 1, min(n_want, len(with_img))).astype(int)
    picks = [with_img[i] for i in dict.fromkeys(idx)]
    EDGES_TOP = [(0, 1), (1, 5), (5, 4), (4, 0)]      # top face
    EDGES_BOT = [(3, 2), (2, 6), (6, 7), (7, 3)]      # bottom face
    EDGES_VERT = [(0, 3), (1, 2), (5, 6), (4, 7)]
    for r in picks:
        img = cv2.imread(r["img"])
        if img is None:
            continue
        gt = np.array(r["gt8"], float)
        pr = np.array(r["pred8"], float)
        def draw(pts, col):
            for a, b in EDGES_TOP + EDGES_BOT + EDGES_VERT:
                cv2.line(img, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)), col, 1)
            for k in range(8):
                c = (0, 165, 255) if k in REAR else col
                cv2.circle(img, tuple(pts[k].astype(int)), 3, c, -1)
                cv2.putText(img, str(k), tuple((pts[k] + 3).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)
        draw(gt, (0, 255, 0))       # GT green (rear=orange dots)
        # band visualization: draw top-front and top-rear mean-y lines
        y_tf = int(np.mean(gt[TOP_FRONT, 1]))
        y_tr = int(np.mean(gt[TOP_REAR, 1]))
        cv2.line(img, (0, y_tf), (img.shape[1], y_tf), (255, 255, 0), 1)
        cv2.line(img, (0, y_tr), (img.shape[1], y_tr), (255, 0, 255), 1)
        txt = f"elev={r['elev']}deg band={r['band_px']}px tilt={r['tilt_deg']}deg rear_err={r['e_rear']} front_err={r['e_front']} {r['dom']}"
        cv2.rectangle(img, (0, 0), (img.shape[1], 18), (0, 0, 0), -1)
        cv2.putText(img, txt, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        fn = f"elev{r['elev']:+06.1f}_{r['dom']}_{r['fid']}.jpg"
        cv2.imwrite(os.path.join(ov_dir, fn), img)
    print(f"[partA] overlays -> {ov_dir} ({len(picks)})")


def write_summary(res, enriched):
    L = []
    L.append("# STAGE22 PART A — \"윗면 보임\" 앙각 실측 (real 98, 학습·재추론 X)\n")
    L.append(f"real enriched N = {res['n']}. elevation = GT pose 시야각(edge-on~0, top-down~90).")
    L.append(f"band_px = |mean_y(kp4,5)-mean_y(kp0,1)| (top-front↔top-rear 화면 수직간격, 수평 팔레트 근사).\n")
    L.append("## elevation bin × 측정")
    L.append("```")
    L.append(f"{'elev':>7} {'n':>4} {'band_med':>9} {'band_p90':>9} {'front_med':>10} {'rear_med':>9} {'rear_spk':>9} {'elev_span':>14}")
    L.append("-" * 80)
    for lbl in res["bins"]:
        b = res["table"][lbl]
        span = f"[{b['elev_span'][0]},{b['elev_span'][1]}]" if b["elev_span"] else "-"
        L.append(f"{lbl:>7} {b['n']:>4} {str(b['band_px_med']):>9} {str(b['band_px_p90']):>9} "
                 f"{str(b['front_med']):>10} {str(b['rear_med']):>9} {str(b['rear_spike_rate']):>9} {span:>14}")
    L.append("```\n")
    c = res["corr"]
    L.append("## 상관 (Spearman)")
    L.append(f"- elev↔band = {c['spearman_elev_band']} (앙각↑ → 윗면 밴드↑ 기대)")
    L.append(f"- band↔rear_err = {c['spearman_band_rearerr']} (밴드 두꺼울수록 rear 정확?)")
    L.append(f"- elev↔rear_err = {c['spearman_elev_rearerr']}\n")

    # tilt caveat
    tilts = [r["tilt_deg"] for r in enriched]
    n_tilt = sum(1 for t in tilts if t > 15)
    L.append(f"## 기운 프레임 caveat: tilt>15deg = {n_tilt}/{len(enriched)} (band 수평근사 부정확 프레임)\n")

    # low-elev but thick-band frames: the "윗면 보이는데 안 되는" 케이스
    low = [r for r in enriched if r["elev"] < 8]
    if low:
        band_med_low = float(np.median([r["band_px"] for r in low]))
        L.append("## 판정: \"윗면 보임 인상\" vs 실측")
        L.append(f"- real 저앙각(<8도) N={len(low)}, band_med={band_med_low:.1f}px, "
                 f"rear_med={np.median([r['e_rear'] for r in low]):.1f}px")
        hi = [r for r in enriched if r["elev"] >= 8]
        if hi:
            L.append(f"- real 고앙각(≥8도) N={len(hi)}, band_med={np.median([r['band_px'] for r in hi]):.1f}px, "
                     f"rear_med={np.median([r['e_rear'] for r in hi]):.1f}px")
    open(os.path.join(OUT, "summary.md"), "w").write("\n".join(L))


if __name__ == "__main__":
    main()
