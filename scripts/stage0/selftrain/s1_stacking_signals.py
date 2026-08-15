"""s1_stacking_signals.py — Paper-S1 over ALL dev real GT sets (outside/night/cad/
noapril), consolidate 9 PL-accept signals per-frame, then evaluate whether STACKING
(consensus count / per-signal discriminability / weighted combo) selects clean PL
(corner_med < 10px).

핵심 질문: 개별 필터(cad 에서 precision 0)를 넘어, 9 신호를 쌓으면(stacking) clean PL
           을 골라낼 수 있나. 그리고 good PL 이 실제 어느 도메인에 있나.

재사용: s1_cad_9filters.py 의 모든 helper (infer/belief/9 signal/apply_filter/overlay).
        여기선 (1) 다중 도메인 루프 (2) signals.csv consolidate (3) stacking 평가만 추가.

★ SEAL: capturepallet07/09, capturenight08/09 (final-test) 절대 미포함.
★ pad100 = near-field PL 후보 생성용, official eval 아님 (명시).
★ 의심 §5: 필터 천장 = base 코너 정확도. good PL 없는 도메인선 stacking 무의미.
           GT 10px 취약(near-miss) → 12px 병기. N 도메인별 작음 → 예비.
           dims-free 2D기하 PL필터 원리한계(memory) 정합 여부 언급. 과결론 금지.
"""
from __future__ import annotations
import os as _os, sys as _sys

# --- stage0 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_S0 = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_S0] + [_os.path.join(_S0, _d) for _d in sorted(_os.listdir(_S0))
                         if _os.path.isdir(_os.path.join(_S0, _d)) and not _d.startswith(".")]

import glob
import importlib.util
import json
import os
import sys

import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
STAGE0 = os.path.join(ROOT, "scripts", "stage0")
sys.path.insert(0, STAGE0)


def _load(mod_name, path):
    spec = importlib.util.spec_from_file_location(mod_name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# base module: reuse ALL helpers (process, apply_filter, TAU, FILTER_ORDER, overlay ...)
B = _load("s1base", os.path.join(STAGE0, "s1_cad_9filters.py"))
import torch  # noqa: E402  (base already imported torch/cv2)
import cv2  # noqa: E402

OUT_DIR = os.path.join(ROOT, "data", "pallet", "eval_results", "s1_stacking")
GOOD_PX = 10.0
NEAR_PX = 12.0          # near-miss 병기 (GT 임계 취약 대비)
N_DET_MIN = B.N_DET_MIN  # 6

# dev sets (SEAL 제외). challenge/data/{name}_manual_gt
DEV_SETS = {
    "outside": ["capturepallet02", "capturepallet03", "capturepallet04",
                "capturepallet05", "capturepallet08"],
    "night":   ["capturenight04", "capturenight05", "capturenight06", "capturenight07"],
    "cad":     ["capturepalletcad"],
    "noapril": ["capture0403noapril"],
}
SEAL = {"capturepallet07", "capturepallet09", "capturenight08", "capturenight09"}

# oriented continuous signals: name -> (raw key, sign) ; sign=+1 higher=good
SIG = B.FILTER_ORDER  # f1..f9 order


def frames_of(name):
    d = os.path.join(ROOT, "challenge", "data", name + "_manual_gt")
    out = []
    for jp in sorted(glob.glob(os.path.join(d, "*.json"))):
        fid = os.path.splitext(os.path.basename(jp))[0]
        for ext in (".png", ".jpg", ".jpeg"):
            ip = os.path.join(d, fid + ext)
            if os.path.exists(ip):
                out.append((jp, ip, f"{name}/{fid}"))
                break
    return out


def oriented(rec):
    """9 oriented scores (higher=better). None if not computable. f2 inf capped."""
    s = rec["scores"]
    f2 = s["f2_peak_ratio"]
    if f2 is not None and not np.isfinite(f2):
        f2 = 10.0  # single-peak = very confident, cap
    # f8 excursion (0=inside envelope) computed later w/ global env; placeholder here
    return {
        "f1_peak": s["f1_peak"],
        "f2_peak_ratio": f2,
        "f3_flip": None if s["f3_flip"] is None else -s["f3_flip"],
        "f4_tta_stab": None if s["f4_tta_stab"] is None else -s["f4_tta_stab"],
        "f5_rear_conf": s["f5_rear_conf"],
        "f6_frsep": s["f6_frsep"],
        "f7_posdepth": 1.0 if rec["f7_posdepth"] else 0.0,
        "f8_size_env": None,  # filled after env
        "f9_bbox_iou": s["f9_bbox_iou"],
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = B.E.load_model(B.WEIGHTS, device)

    # ── inference over all dev frames ────────────────────────────────────
    recs = []  # each: base process() dict + domain
    for dom, names in DEV_SETS.items():
        for name in names:
            assert name not in SEAL, f"SEAL leak: {name}"
            fl = frames_of(name)
            print(f"[set] {dom}/{name} N={len(fl)}")
            for jp, ip, tag in fl:
                r = B.process(model, jp, ip, device)
                if r is None:
                    print(f"   FAIL {tag}")
                    continue
                r["domain"] = dom
                r["tag"] = tag
                recs.append(r)
                print(f"   {tag} det={r['n_det']} cm="
                      f"{r['corner_med'] if r['corner_med'] is not None else 'na'}")

    # ── global GT envelope for f8 (model-independent) ────────────────────
    gsr = np.array([r["gt_sr"] for r in recs if r["gt_sr"] is not None])
    gasp = np.array([r["gt_asp"] for r in recs if r["gt_asp"] is not None])
    env = {"size_lo": float(np.percentile(gsr, 2.5)),
           "size_hi": float(np.percentile(gsr, 97.5)),
           "asp_lo": float(np.percentile(gasp, 2.5)),
           "asp_hi": float(np.percentile(gasp, 97.5))}

    # ── consolidate: rows = all frames (det<6 -> signals NA) ─────────────
    import csv
    N = len(recs)
    detected = [r for r in recs if r["n_det"] >= N_DET_MIN]
    for r in recs:
        r["good"] = bool(r["corner_med"] is not None and r["corner_med"] < GOOD_PX)
        r["good12"] = bool(r["corner_med"] is not None and r["corner_med"] < NEAR_PX)
        # f8 excursion (lower=better) + oriented
        o = oriented(r)
        if r["pred_sr"] is not None and r["pred_asp"] is not None:
            es = B.excursion(r["pred_sr"], env["size_lo"], env["size_hi"])
            ea = B.excursion(r["pred_asp"], env["asp_lo"], env["asp_hi"])
            r["f8_excursion"] = float(max(es, ea))
            o["f8_size_env"] = -r["f8_excursion"]
        else:
            r["f8_excursion"] = None
        r["oriented"] = o

    csv_path = os.path.join(OUT_DIR, "signals.csv")
    cols = (["domain", "fid", "det", "gt_corner_med", "is_good", "is_good12"]
            + SIG)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in recs:
            det = r["n_det"] >= N_DET_MIN
            s = r["scores"]
            row = [r["domain"], r["tag"].split("/")[1], r["n_det"],
                   ("" if r["corner_med"] is None else round(r["corner_med"], 2)),
                   int(r["good"]), int(r["good12"])]
            if not det:
                row += [""] * len(SIG)  # signals NA
            else:
                vals = {
                    "f1_peak": s["f1_peak"], "f2_peak_ratio": s["f2_peak_ratio"],
                    "f3_flip": s["f3_flip"], "f4_tta_stab": s["f4_tta_stab"],
                    "f5_rear_conf": s["f5_rear_conf"], "f6_frsep": s["f6_frsep"],
                    "f7_posdepth": int(r["f7_posdepth"]),
                    "f8_size_env": r["f8_excursion"], "f9_bbox_iou": s["f9_bbox_iou"],
                }
                row += [("" if vals[k] is None else
                         (round(vals[k], 4) if isinstance(vals[k], float) else vals[k]))
                        for k in SIG]
            w.writerow(row)
    print(f"\n[save] {csv_path}  rows={N} (detected>=6: {len(detected)})")

    # ── (a) consensus curve: pass matrix at canonical TAU ────────────────
    passmat = {}   # tag -> {fname: bool}
    for r in detected:
        passmat[r["tag"]] = {}
        for fname in SIG:
            tau = B.TAU.get(fname)
            passmat[r["tag"]][fname] = bool(B.apply_filter(fname, r, tau, env))
    for r in detected:
        r["k_pass"] = sum(passmat[r["tag"]].values())

    consensus = []
    for k in range(1, 10):
        sel = [r for r in detected if r["k_pass"] >= k]
        ng = sum(1 for r in sel if r["good"])
        ng12 = sum(1 for r in sel if r["good12"])
        consensus.append({
            "k": k, "n_pass": len(sel),
            "n_good": ng, "n_good12": ng12,
            "precision": (round(ng / len(sel), 3) if sel else None),
            "precision12": (round(ng12 / len(sel), 3) if sel else None),
            "recall": (round(ng / max(1, sum(1 for r in detected if r["good"])), 3)),
        })

    # ── (b) per-signal discriminability: AUC + Spearman vs is_good ───────
    from sklearn.metrics import roc_auc_score
    from scipy.stats import spearmanr
    y = np.array([int(r["good"]) for r in detected])
    y12 = np.array([int(r["good12"]) for r in detected])
    sig_stats = {}
    for fname in SIG:
        xs = np.array([r["oriented"][fname] if r["oriented"][fname] is not None
                       else np.nan for r in detected], float)
        ok = ~np.isnan(xs)
        auc = auc12 = rho = None
        if ok.sum() >= 4 and 0 < y[ok].sum() < ok.sum():
            try:
                auc = round(float(roc_auc_score(y[ok], xs[ok])), 3)
            except Exception:
                pass
        if ok.sum() >= 4 and 0 < y12[ok].sum() < ok.sum():
            try:
                auc12 = round(float(roc_auc_score(y12[ok], xs[ok])), 3)
            except Exception:
                pass
        if ok.sum() >= 4:
            rr = spearmanr(xs[ok], y[ok])
            rho = round(float(rr.correlation), 3) if np.isfinite(rr.correlation) else None
        sig_stats[fname] = {"auc_good": auc, "auc_good12": auc12,
                            "spearman_good": rho, "n_valid": int(ok.sum())}

    # ── (c) weighted combo (logistic, LOO-CV) vs best single ─────────────
    combo = _combo_lr(detected, SIG, y)

    # ── domain-level good distribution ───────────────────────────────────
    dom_stats = {}
    for dom in DEV_SETS:
        dr = [r for r in recs if r["domain"] == dom]
        dd = [r for r in dr if r["n_det"] >= N_DET_MIN]
        cms = [r["corner_med"] for r in dd if r["corner_med"] is not None]
        dom_stats[dom] = {
            "n_frames": len(dr), "n_detected": len(dd),
            "n_good10": sum(1 for r in dr if r["good"]),
            "n_good12": sum(1 for r in dr if r["good12"]),
            "best_corner_med": (round(min(cms), 1) if cms else None),
            "median_corner_med": (round(float(np.median(cms)), 1) if cms else None),
        }

    # ── save report + png + overlays ─────────────────────────────────────
    _write_report(recs, detected, consensus, sig_stats, combo, dom_stats, env)
    _consensus_png(consensus)
    _overlays(detected, passmat, env)

    # console
    print("\n=== consensus (k>= passes) ===")
    for c in consensus:
        print(f"  k>={c['k']}: N={c['n_pass']:>3} good10={c['n_good']:>2} "
              f"prec10={c['precision']} good12={c['n_good12']:>2} "
              f"prec12={c['precision12']} recall10={c['recall']}")
    print("\n=== signal AUC (vs is_good<10px) ===")
    for f in SIG:
        st = sig_stats[f]
        print(f"  {f:<15} AUC10={st['auc_good']} AUC12={st['auc_good12']} "
              f"rho={st['spearman_good']} n={st['n_valid']}")
    print(f"\n=== combo LR (LOO-CV) AUC={combo['loo_auc']} vs best single "
          f"{combo['best_single']} AUC={combo['best_single_auc']} ===")
    print("\n=== domain good distribution ===")
    for dom, st in dom_stats.items():
        print(f"  {dom:<8} N={st['n_frames']:>3} det={st['n_detected']:>3} "
              f"good10={st['n_good10']} good12={st['n_good12']} "
              f"best_cm={st['best_corner_med']} med_cm={st['median_corner_med']}")
    print(f"\n[save] {OUT_DIR}/")


def _combo_lr(detected, SIG, y):
    """Standardized oriented signals -> LOO-CV logistic AUC. Impute None w/ median."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    X = np.zeros((len(detected), len(SIG)))
    for j, f in enumerate(SIG):
        col = np.array([r["oriented"][f] if r["oriented"][f] is not None else np.nan
                        for r in detected], float)
        med = np.nanmedian(col) if np.isfinite(np.nanmedian(col)) else 0.0
        col = np.where(np.isnan(col), med, col)
        sd = col.std() if col.std() > 1e-9 else 1.0
        X[:, j] = (col - col.mean()) / sd
    npos = int(y.sum())
    if npos < 3 or npos > len(y) - 3:
        # LR CV meaningless; report best single only
        aucs = {}
        for j, f in enumerate(SIG):
            try:
                aucs[f] = roc_auc_score(y, X[:, j])
            except Exception:
                aucs[f] = 0.5
        bf = max(aucs, key=aucs.get)
        return {"loo_auc": None, "note": f"npos={npos} too small for CV",
                "best_single": bf, "best_single_auc": round(float(aucs[bf]), 3)}
    # LOO-CV
    preds = np.zeros(len(y))
    for i in range(len(y)):
        tr = np.ones(len(y), bool); tr[i] = False
        if len(np.unique(y[tr])) < 2:
            preds[i] = y[tr].mean(); continue
        clf = LogisticRegression(C=0.5, max_iter=500)
        clf.fit(X[tr], y[tr])
        preds[i] = clf.predict_proba(X[i:i+1])[0, 1]
    loo = round(float(roc_auc_score(y, preds)), 3)
    aucs = {f: roc_auc_score(y, X[:, j]) for j, f in enumerate(SIG)}
    bf = max(aucs, key=aucs.get)
    return {"loo_auc": loo, "best_single": bf,
            "best_single_auc": round(float(aucs[bf]), 3)}


def _consensus_png(consensus):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        with open(os.path.join(OUT_DIR, "consensus_pr.txt"), "w") as f:
            for c in consensus:
                f.write(f"k>={c['k']} N={c['n_pass']} prec10={c['precision']} "
                        f"prec12={c['precision12']} recall={c['recall']}\n")
        return
    ks = [c["k"] for c in consensus]
    p10 = [c["precision"] if c["precision"] is not None else np.nan for c in consensus]
    p12 = [c["precision12"] if c["precision12"] is not None else np.nan for c in consensus]
    rc = [c["recall"] for c in consensus]
    npass = [c["n_pass"] for c in consensus]
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(ks, p10, "o-", color="tab:red", label="precision (good<10px)")
    ax1.plot(ks, p12, "s--", color="tab:orange", label="precision (good<12px)")
    ax1.plot(ks, rc, "^-", color="tab:green", label="recall (good<10px)")
    ax1.set_xlabel("consensus k (# signals passed, >=k)")
    ax1.set_ylabel("precision / recall")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(loc="upper left")
    ax2 = ax1.twinx()
    ax2.bar(ks, npass, alpha=0.15, color="tab:blue")
    ax2.set_ylabel("N frames passing (>=k)", color="tab:blue")
    ax1.set_title("Paper-S1 stacking: consensus count vs clean-PL precision (dev real)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "consensus_pr.png"), dpi=110)
    plt.close(fig)


def _overlays(detected, passmat, env):
    """stacking-best rule (highest k with precision>0, else max k observed) 통과분 오버레이."""
    odir = os.path.join(OUT_DIR, "overlays")
    os.makedirs(odir, exist_ok=True)
    for old in glob.glob(os.path.join(odir, "*.jpg")):
        os.remove(old)
    # choose rule: max k that still yields >=1 frame; prefer showing good & bad mix
    ranked = sorted(detected, key=lambda r: -r["k_pass"])
    top = ranked[:10]
    for r in top:
        cm = int(round(r["corner_med"])) if r["corner_med"] is not None else -1
        g = "good" if r["good"] else "bad"
        fn = f"k{r['k_pass']}_{g}_cm{cm}_{r['tag'].replace('/', '_')}.jpg"
        _save_ov(r, os.path.join(odir, fn))


def _save_ov(rec, out_path):
    img = cv2.imread(rec["ip"])
    if img is None:
        return
    B.draw_cuboid(img, rec["gt8"], rec["gtc"], B.GT_COL, 6, 2, 7, num=False)
    B.draw_cuboid(img, np.array(rec["pred8"], float), rec["pred_c"], B.PRED_COL, 4, -1, 6)
    cm = f"{rec['corner_med']:.1f}" if rec["corner_med"] is not None else "n/a"
    good = "GOOD" if rec["good"] else "BAD"
    h1 = f"{rec['tag']}  k_pass={rec['k_pass']}/9  det={rec['n_det']}/8"
    h2 = f"corner_med={cm}px [{good}]  GT=green S1=red  (pad100, NOT official)"
    cv2.rectangle(img, (0, 0), (img.shape[1], 44), (0, 0, 0), -1)
    cv2.putText(img, h1, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, h2, (6, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, img)


def _write_report(recs, detected, consensus, sig_stats, combo, dom_stats, env):
    L = []
    A = L.append
    A("# Paper-S1 stacking of 9 PL-accept signals over dev real GT sets")
    A("")
    A(f"- weights: `weights/paper_s1/paper_s1_maskaux/net_epoch_0065.pth`")
    A(f"- inference: reflect-pad100 (near-field PL 후보 확보용; **NOT official eval**)")
    A(f"- domains(dev): outside/night/cad/noapril | SEAL 제외(pallet07/09,night08/09)")
    A(f"- total frames={len(recs)} | detected(>=6 corner)={len(detected)}")
    A(f"- good = order-free Hungarian corner_med < 10px (12px 병기: GT 임계 취약)")
    A(f"- ★ N 도메인별 작음 → 예비. 필터 천장 = base 코너 정확도 (memory).")
    A("")
    A("## (Domain) good PL distribution — good PL 이 어디에 있나")
    A("```")
    A(f"{'domain':<9}{'N':>4}{'det':>5}{'good10':>8}{'good12':>8}"
      f"{'best_cm':>9}{'med_cm':>8}")
    A("-" * 51)
    for dom, st in dom_stats.items():
        A(f"{dom:<9}{st['n_frames']:>4}{st['n_detected']:>5}{st['n_good10']:>8}"
          f"{st['n_good12']:>8}{str(st['best_corner_med']):>9}{str(st['median_corner_med']):>8}")
    A("```")
    A("")
    A("## (a) Consensus curve — k신호 이상 통과 시 PL precision")
    A("```")
    A(f"{'k>=':>4}{'N_pass':>8}{'good10':>8}{'prec10':>8}{'good12':>8}"
      f"{'prec12':>8}{'recall10':>10}")
    A("-" * 54)
    for c in consensus:
        A(f"{c['k']:>4}{c['n_pass']:>8}{c['n_good']:>8}{str(c['precision']):>8}"
          f"{c['n_good12']:>8}{str(c['precision12']):>8}{str(c['recall']):>10}")
    A("```")
    A("")
    A("## (b) Per-signal discriminability (oriented so higher=better)")
    A("```")
    A(f"{'signal':<15}{'AUC(good10)':>12}{'AUC(good12)':>12}{'Spearman':>10}{'n':>5}")
    A("-" * 54)
    for f in B.FILTER_ORDER:
        st = sig_stats[f]
        A(f"{f:<15}{str(st['auc_good']):>12}{str(st['auc_good12']):>12}"
          f"{str(st['spearman_good']):>10}{st['n_valid']:>5}")
    A("```")
    A("- AUC>0.5 = good 을 양의 방향으로 가름 / ~0.5 = 무관 / <0.5 = 역방향.")
    A("")
    A("## (c) Weighted combo (logistic, LOO-CV) vs best single signal")
    A("```")
    if combo.get("loo_auc") is not None:
        A(f"combo LR LOO-CV AUC = {combo['loo_auc']}")
    else:
        A(f"combo LR: {combo.get('note','n/a')} (CV 불가)")
    A(f"best single signal = {combo['best_single']} (AUC {combo['best_single_auc']})")
    A("```")
    A("")
    A("## 판정 (stacking 이 clean PL 을 만드나)")
    n_good10 = sum(1 for r in recs if r["good"])
    n_good12 = sum(1 for r in recs if r["good12"])
    best_k = max(consensus, key=lambda c: (c["precision"] or 0, c["k"]))
    A(f"- 전체 good10={n_good10}, good12={n_good12} (detected={len(detected)}).")
    if n_good10 == 0:
        A("- ★ dev 전체에 good10 PL 이 0 → 어떤 stacking 규칙도 precision>0 불가. "
          "필터/stacking 실패가 아니라 **base(S1) 코너 정확도 천장** 문제. "
          "(memory: 필터 천장=base / dims-free 2D기하 PL필터 원리한계 정합).")
    else:
        A(f"- 최고 precision consensus: k>={best_k['k']} → "
          f"prec10={best_k['precision']} (N={best_k['n_pass']}, good={best_k['n_good']}).")
        A(f"- combo LR LOO-CV AUC={combo.get('loo_auc')} vs best single "
          f"{combo['best_single']}={combo['best_single_auc']} → "
          "stacking 이 단일 대비 이득 있으면 combo AUC 가 유의미하게 높아야 함.")
    A("- ★ good PL 존재 도메인에서만 stacking 유효 (도메인 분포 표 참조). "
      "cad/noapril=unseen·near-field 라 good≈0 예상, outside 저앙각이 주력.")
    A("- ★ 과결론 금지: N 작고 pad(비공식)·GT 10px 취약. 데이터로만 판단.")
    with open(os.path.join(OUT_DIR, "stacking_report.md"), "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
