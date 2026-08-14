"""orthogonal_filters_exp.py — DiffPnP 와 순환하지 않는 "직교(orthogonal) 신호"
필터 실험. Stage B(paper_s2 net_epoch_0057) 예측을 REAL filterval(outside44 +
night43 + manual36 = 123, low-angle)에 추론하고, 세 계열의 비-기하 신호를 계산해
GT 로 분리력(selectivity)을 검증한다.

배경 (memory: filter-circular-with-diffpnp-need-orthogonal):
  기존 기하 필터(f6 frsep / f7 posdepth)는 DiffPnP 학습이 강제한 기하와 순환이라
  depth-collapse / half=whole 같은 "기하적으로 valid 한데 틀린" 오류를 blind.
  confidence 필터(f1/f2/f5)도 rear 는 confidently-wrong(sharp 단봉)이라 무력.
  → 순환 밖 신호가 필요. 사용자 제안 4계열 중 3개를 실험:

  (1) PHOTOMETRIC TTA  ★#1 recommended
      기하를 안 바꾸고 외형만 교란(gamma/WB/noise/JPEG/blur) 후 재추론.
      raw-heatmap keypoint 위치가 흔들리면 = 이미지 evidence 불안정.
      ★핵심 = FRONT/REAR 분리. f4(tta_stab)는 8코너 평균 1개 스칼라라 front(항상
      안정)에 희석돼 무력했음. rear 만 흔들리는 프레임을 잡는 게 미검증 가설.

  (2) HEATMAP EVIDENCE SPREAD  (#4)
      DiffPnP 이전 belief 자체의 국소 퍼짐(covariance)·entropy. 넓게 퍼지면 약한
      evidence. f1/f2(peak/2차봉)와 부분 겹침이나 spatial spread 는 새 축.

  (3) ENSEMBLE DISAGREEMENT  (#3)
      다른 recipe 모델(paper_s1_maskaux)로도 추론해 raw keypoint 불일치 측정.
      ★caveat: s1 은 squash-parity 입력에 OOD 가능(전처리 계보 다름) → 불일치가
      epistemic 이 아니라 domain-shift 일 수 있음. SECONDARY 로만 해석.

검증 (핵심 = "조금만 더 고치면"):
  - Spearman(signal, GT 오차) — overall 과 ★REAR(back, index-aligned) 각각.
  - "배포필터(f1&f2&f4&f5&f6&f7) 통과했지만 GT-나쁨" 프레임을 각 rear-signal 이
    threshold 로 몇 개 잡는가 = pass=truly-good 만들기 직접 측정.
  - 도메인별(outside/manual = base 품질 有 / night = base 붕괴 → 참고만).

재사용 (필터·decode 재구현 없음):
  T = paper_s2_testset17_9filters : infer_squash / eval_frame_squash / M(filters) /
      TAU / FILTER_ORDER / preprocess_squash / E / WEIGHTS / THRESH / N_DET_MIN.
  S = stage25_paperbase_eval : frames_filterval() -> (dom, fid, jp, ip) + GT.

산출: data/pallet/eval_results/paper_s2_scratch_diffpnp/orthogonal_filters_exp.md
      + orthogonal_filters_exp.json (per-frame 원자료).

Usage: conda activate pallet-pose; python scripts/stage0/filter_pl/orthogonal_filters_exp.py
"""
from __future__ import annotations
import os as _os, sys as _sys

# --- stage0 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_S0 = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_S0] + [_os.path.join(_S0, _d) for _d in sorted(_os.listdir(_S0))
                         if _os.path.isdir(_os.path.join(_S0, _d)) and not _d.startswith(".")]

import json
import os
import sys

import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, os.path.join(ROOT, "scripts", "stage0"))

import paper_s2_testset17_9filters as T  # noqa: E402
import stage25_paperbase_eval as S       # noqa: E402
from eval_pvnet_heads import split_metrics  # noqa: E402
import annotate_pnp as APNP  # noqa: E402
import cv2      # noqa: E402
import torch    # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

M = T.M
TAU = T.TAU
E = T.E
THRESH = T.THRESH
N_DET_MIN = T.N_DET_MIN
GOOD_PX = T.GOOD_PX
GROSS_PX = 20.0
S1_WEIGHTS = os.path.join(ROOT, "weights", "paper_s1_maskaux", "net_epoch_0065.pth")
OUT_DIR = os.path.join(ROOT, "data", "pallet", "eval_results", "paper_s2_scratch_diffpnp")
OUT_MD = os.path.join(OUT_DIR, "orthogonal_filters_exp.md")
OUT_JSON = os.path.join(OUT_DIR, "orthogonal_filters_exp.json")

# 배포 필터 (history 2026-07-10): f3 제외(L-R broken), f8/f9 제외(GT 필요)
DEPLOY = ["f1_peak", "f2_peak_ratio", "f4_tta_stab", "f5_rear_conf",
          "f6_frsep", "f7_posdepth"]
FRONT_IDX, REAR_IDX = (0, 1, 2, 3), (4, 5, 6, 7)


# ── (1) photometric variants: 기하 불변, 외형만 교란 ─────────────────────────
def _gamma(img, g):
    lut = (np.linspace(0, 1, 256) ** g * 255).astype(np.uint8)
    return cv2.LUT(img, lut)


def _wb(img, rs, bs):
    o = img.astype(np.float32).copy()
    o[:, :, 2] *= rs   # BGR: R=2
    o[:, :, 0] *= bs   # B=0
    return np.clip(o, 0, 255).astype(np.uint8)


def _jpeg(img, q):
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else img


def photometric_variants(img):
    """appearance-only (geometry preserved) → 각 변형 이미지 list."""
    n = np.random.default_rng(0).normal(0, 12, img.shape).astype(np.float32)  # fixed noise
    return {
        "gamma0.6": _gamma(img, 0.6),
        "gamma1.6": _gamma(img, 1.6),
        "bright0.7": np.clip(img.astype(np.float32) * 0.7, 0, 255).astype(np.uint8),
        "bright1.3": np.clip(img.astype(np.float32) * 1.3, 0, 255).astype(np.uint8),
        "wb_warm": _wb(img, 1.15, 0.85),
        "wb_cool": _wb(img, 0.85, 1.15),
        "noise": np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8),
        "jpeg25": _jpeg(img, 25),
        "blur": cv2.GaussianBlur(img, (5, 5), 0),
    }


def photo_tta(model, img, device, base_pred8):
    """base_pred8 대비 photometric 변형들의 per-corner 변위 + dropout.
    반환: per-corner disp(px, nan if base 미검출), per-corner dropout frac."""
    variants = photometric_variants(img)
    preds = []
    for v in variants.values():
        _, p8, _, _, _ = T.infer_squash(model, v, device)
        preds.append(p8)
    preds = np.stack(preds, 0)  # (V,8,2)
    V = preds.shape[0]
    disp = np.full(8, np.nan)
    drop = np.full(8, np.nan)
    for i in range(8):
        if np.isnan(base_pred8[i, 0]):
            continue
        col = preds[:, i, :]
        ok = ~np.isnan(col[:, 0])
        drop[i] = float((~ok).sum()) / V
        if ok.sum() >= 1:
            d = np.linalg.norm(col[ok] - base_pred8[i], axis=1)
            disp[i] = float(np.mean(d))
        else:
            disp[i] = np.nan  # 전부 사라짐 → dropout 으로 포착
    return disp, drop


# ── (2) heatmap evidence spread (belief 국소 2차 모멘트 + entropy) ────────────
def heatmap_spread(belief, W, H, win=4):
    """corner i belief 의 peak 주변 window 확률 covariance trace(px) + entropy.
    반환: per-corner spread_px(8,), entropy(8,)."""
    sx, sy = W / 50.0, H / 50.0
    spread = np.full(8, np.nan)
    ent = np.full(8, np.nan)
    for i in range(8):
        b = belief[i]
        if b.max() <= 1e-6:
            continue
        py, px = np.unravel_index(int(np.argmax(b)), b.shape)
        y0, y1 = max(0, py - win), min(b.shape[0], py + win + 1)
        x0, x1 = max(0, px - win), min(b.shape[1], px + win + 1)
        w = np.clip(b[y0:y1, x0:x1], 0, None).astype(np.float64)
        s = w.sum()
        if s <= 1e-9:
            continue
        p = w / s
        ys, xs = np.mgrid[y0:y1, x0:x1]
        mx = (p * xs).sum()
        my = (p * ys).sum()
        vx = (p * (xs - mx) ** 2).sum()
        vy = (p * (ys - my) ** 2).sum()
        spread[i] = float(np.sqrt(vx * sx * sx + vy * sy * sy))
        pf = p[p > 1e-12]
        ent[i] = float(-(pf * np.log(pf)).sum())
    return spread, ent


def _agg(vals, idx):
    v = [vals[i] for i in idx if not np.isnan(vals[i])]
    return float(np.mean(v)) if v else None


def _worst(vals):
    v = [x for x in vals if not np.isnan(x)]
    return float(np.max(v)) if v else None


# ── (3) ensemble disagreement (paper_s1, OOD-caveat) ─────────────────────────
def ensemble_disagree(model_s1, img, device, base_pred8):
    _, p8, _, _, _ = T.infer_squash(model_s1, img, device)
    disp = np.full(8, np.nan)
    for i in range(8):
        if np.isnan(base_pred8[i, 0]) or np.isnan(p8[i, 0]):
            continue
        disp[i] = float(np.linalg.norm(p8[i] - base_pred8[i]))
    return disp


# ── per-frame ────────────────────────────────────────────────────────────────
def process(model, model_s1, dom, fid, jp, ip, device, env):
    row = T.eval_frame_squash(model, jp, ip, device)
    if row is None:
        return None
    img = cv2.imread(ip)
    H, W = img.shape[:2]
    d = json.load(open(jp))
    K = E.K_from_json(d)
    img_diag = float(np.hypot(W, H))

    gt8 = np.array(row["gt8"], float)
    pred8 = np.array(row["pred8"], float)
    pred_c = row["pred_c"]
    n_det = int(row["n_det"])
    corner = row["corner"] if np.isfinite(row["corner"]) else None
    front_e = row["front"] if np.isfinite(row["front"]) else None
    rear_e = row["back"] if np.isfinite(row["back"]) else None

    rec = {
        "dom": dom, "fid": str(fid), "ip": ip,
        "v_geom": int(row["v_geom"]), "n_det": n_det,
        "corner_med": corner, "front_err": front_e, "rear_err": rear_e,
        "good": bool(corner is not None and corner < GOOD_PX),
        "gross": bool(corner is None or corner > GROSS_PX),
    }
    if n_det < N_DET_MIN:
        rec["deploy_pass"] = False
        return rec

    # base belief + filter scores (build_rec 본체 재사용)
    belief, pred8b, pred_cb, peaks, ratios = T.infer_squash(model, img, device)
    det8 = [i for i in range(8) if not np.isnan(pred8[i, 0])]
    scores = {
        "f1_peak": float(min(peaks[i] for i in det8)),
        "f2_peak_ratio": float(min(ratios[i] for i in det8)),
        "f4_tta_stab": T.tta_stab_squash(model, img, device),
        "f6_frsep": M.frsep_frac(pred8),
    }
    rear_det = [i for i in REAR_IDX if not np.isnan(pred8[i, 0])]
    scores["f5_rear_conf"] = (float(min(peaks[i] for i in rear_det))
                              if len(rear_det) == 4 else None)
    f7, _ = M.posdepth_ok(pred8, pred_c, K, img.shape)
    pred_sr, pred_asp = M.size_aspect(pred8, img_diag)
    row_flt = {"n_det": n_det, "scores": scores, "f7_posdepth": bool(f7),
               "pred_sr": pred_sr, "pred_asp": pred_asp}
    deploy_pass = all(M.apply_filter(f, row_flt, TAU.get(f), env) for f in DEPLOY)

    # ── orthogonal signals ──
    disp, drop = photo_tta(model, img, device, pred8)
    spread, ent = heatmap_spread(belief, W, H)
    ens = ensemble_disagree(model_s1, img, device, pred8)

    rec.update({
        "deploy_pass": bool(deploy_pass),
        "f_scores": scores, "f7": bool(f7),
        # (1) photometric TTA
        "photo_front": _agg(disp, FRONT_IDX), "photo_rear": _agg(disp, REAR_IDX),
        "photo_worst": _worst(disp),
        "drop_front": _agg(drop, FRONT_IDX), "drop_rear": _agg(drop, REAR_IDX),
        # (2) heatmap spread
        "spread_front": _agg(spread, FRONT_IDX), "spread_rear": _agg(spread, REAR_IDX),
        "spread_worst": _worst(spread),
        "ent_front": _agg(ent, FRONT_IDX), "ent_rear": _agg(ent, REAR_IDX),
        # (3) ensemble
        "ens_front": _agg(ens, FRONT_IDX), "ens_rear": _agg(ens, REAR_IDX),
        "ens_worst": _worst(ens),
    })
    return rec


# ── validation helpers ───────────────────────────────────────────────────────
SIGNALS = ["photo_front", "photo_rear", "photo_worst", "drop_rear",
           "spread_front", "spread_rear", "spread_worst", "ent_rear",
           "ens_front", "ens_rear", "ens_worst",
           "f4_tta_stab_ref"]  # f4 = 기존 baseline 대조군


def _sig(rec, name):
    if name == "f4_tta_stab_ref":
        return rec.get("f_scores", {}).get("f4_tta_stab")
    return rec.get(name)


def spear(recs, sig, target):
    xs, ys = [], []
    for r in recs:
        v = _sig(r, sig)
        t = r.get(target)
        if v is not None and t is not None and np.isfinite(v) and np.isfinite(t):
            xs.append(v)
            ys.append(t)
    if len(xs) < 6:
        return None, len(xs)
    rho, _ = spearmanr(xs, ys)
    return (float(rho) if np.isfinite(rho) else None), len(xs)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    APNP.PALLET_DIMS = (1.1, 1.3, 0.12)
    frames = S.frames_filterval()
    print(f"[filterval] {len(frames)} frames; Stage B vs signals "
          f"(photometric/spread/ensemble-s1); dims={APNP.PALLET_DIMS}")

    model = E.load_model(T.WEIGHTS, device)
    model_s1 = E.load_model(S1_WEIGHTS, device)

    # DEPLOY = f1,f2,f4,f5,f6,f7 — 모두 env(f8 전용) 미사용 → dummy 로 충분.
    env = {"size_lo": 0.0, "size_hi": 1.0, "asp_lo": 0.0, "asp_hi": 10.0}

    recs = []
    for k, (dom, fid, jp, ip) in enumerate(frames):
        r = process(model, model_s1, dom, fid, jp, ip, device, env)
        if r is not None:
            recs.append(r)
        cm = f"{r['corner_med']:.1f}" if (r and r.get("corner_med") is not None) else "-"
        pr = f"{r.get('photo_rear'):.1f}" if (r and r.get("photo_rear") is not None) else "-"
        print(f"  [{k+1}/{len(frames)}] {dom:<8}{str(fid)[:12]:<13} det={r['n_det'] if r else '-'} "
              f"cm={cm} photo_rear={pr}")

    with open(OUT_JSON, "w") as f:
        json.dump(recs, f, indent=1, default=lambda x: None
                  if isinstance(x, float) and not np.isfinite(x) else x)

    det = [r for r in recs if r["n_det"] >= N_DET_MIN and "photo_rear" in r]
    report(recs, det, env)


def report(recs, det, env):
    L = []
    L.append("# 직교(orthogonal) 신호 필터 실험 — Stage B on filterval(123)")
    L.append("")
    L.append("- weights: `weights/paper_s2_stageB/net_epoch_0057.pth` (squash-parity)")
    L.append("- set: filterval = outside44 + night43 + manual36 = 123 (real, low-angle)")
    L.append("- 목적: DiffPnP 와 순환 안 하는 비-기하 신호가 rear-collapse / half=whole 를 잡는가.")
    L.append(f"- signals: (1)photometric TTA front/rear/worst + rear-dropout  "
             f"(2)heatmap spread/entropy  (3)ensemble(s1) disagreement")
    L.append(f"- GT-good=corner_med<{GOOD_PX:.0f}px, GT-gross=>{GROSS_PX:.0f}px. "
             f"rear_err=back(index-aligned Hungarian, GT rear index).")
    L.append(f"- ★f4_tta_stab_ref = 기존 f4(8코너 평균 scalar) 대조군.")
    L.append("")

    def dom_stat(rs):
        d = [r for r in rs if r["n_det"] >= N_DET_MIN and "photo_rear" in r]
        good = sum(1 for r in d if r["good"])
        return len(rs), len(d), good

    L.append("## 집합 요약")
    L.append("```")
    L.append(f"{'set':<10}{'N':>5}{'det':>5}{'GT-good':>9}")
    for nm, rs in [("overall", recs)] + [(dm, [r for r in recs if r["dom"] == dm])
                                         for dm in ("outside", "night", "manual")]:
        n, nd, ng = dom_stat(rs)
        L.append(f"{nm:<10}{n:>5}{nd:>5}{ng:>9}")
    L.append("```")
    L.append("")

    # ── Spearman: signal vs GT error (overall & rear) ──
    L.append("## Spearman(signal, GT오차) — 양수=신호↑일수록 오차↑ (좋은 필터=강한 양상관)")
    L.append("낮은 신호=신뢰. reject 규칙 = signal > tau. |rho| 클수록 분리력.")
    for setname, rs in [("outside+manual (base 품질 有)",
                         [r for r in det if r["dom"] in ("outside", "manual")]),
                        ("outside", [r for r in det if r["dom"] == "outside"]),
                        ("manual", [r for r in det if r["dom"] == "manual"]),
                        ("night (base 붕괴, 참고만)", [r for r in det if r["dom"] == "night"]),
                        ("overall", det)]:
        L.append("")
        L.append(f"### {setname}  (n_det={len(rs)})")
        L.append("```")
        L.append(f"{'signal':<18}{'rho vs overall':>16}{'(n)':>6}{'rho vs REAR':>14}{'(n)':>6}")
        L.append("-" * 60)
        for sig in SIGNALS:
            ro, no = spear(rs, sig, "corner_med")
            rr, nr = spear(rs, sig, "rear_err")
            ros = f"{ro:+.2f}" if ro is not None else "  -  "
            rrs = f"{rr:+.2f}" if rr is not None else "  -  "
            L.append(f"{sig:<18}{ros:>16}{no:>6}{rrs:>14}{nr:>6}")
        L.append("```")

    # ── 핵심: 배포필터 통과했지만 나쁜 프레임을 rear-signal 이 잡는가 ──
    L.append("")
    L.append("## ★ 핵심: 배포필터 통과했지만 GT-나쁨(=confidently-wrong) 프레임 잡기")
    L.append("배포필터 = f1&f2&f4&f5&f6&f7. 통과 중 GT-bad(corner_med>=10px) 를 "
             "각 rear-signal 이 상위 threshold(통과군 75-percentile)로 몇 개 flag 하는가.")
    L.append("이상적 = accepted-bad 는 많이 잡고, accepted-good 는 적게 버림.")
    accepted = [r for r in det if r["deploy_pass"]]
    acc_good = [r for r in accepted if r["good"]]
    acc_bad = [r for r in accepted if not r["good"]]
    L.append("```")
    L.append(f"배포필터 통과: {len(accepted)}  (good={len(acc_good)}, bad={len(acc_bad)})")
    L.append(f"{'rear-signal':<16}{'tau(good p75)':>14}{'bad잡음':>9}{'good버림':>10}"
             f"{'net(잡-버)':>11}")
    L.append("-" * 62)
    for sig in ["photo_rear", "photo_worst", "spread_rear", "spread_worst",
                "ens_rear", "ens_worst", "drop_rear", "f4_tta_stab_ref"]:
        gv = [v for r in acc_good if (v := _sig(r, sig)) is not None and np.isfinite(v)]
        if len(gv) < 4:
            L.append(f"{sig:<16}{'n/a (good<4)':>14}")
            continue
        tau = float(np.percentile(gv, 75))
        caught = sum(1 for r in acc_bad
                     if (v := _sig(r, sig)) is not None and np.isfinite(v) and v > tau)
        lost = sum(1 for r in acc_good
                   if (v := _sig(r, sig)) is not None and np.isfinite(v) and v > tau)
        L.append(f"{sig:<16}{tau:>14.2f}{caught:>6}/{len(acc_bad):<2}"
                 f"{lost:>7}/{len(acc_good):<2}{caught - lost:>+11}")
    L.append("```")
    L.append(f"(accepted-bad fids: " +
             ", ".join(f"{r['dom']}/{r['fid']}(cm{r['corner_med']:.0f},rear{r['rear_err']:.0f})"
                       for r in acc_bad) + ")")

    L.append("")
    L.append("## per-frame (detected, outside+manual, corner_med 순)")
    L.append("```")
    hdr = (f"{'dom':<8}{'fid':<13}{'cm':>5}{'rear':>5}{'ph_f':>6}{'ph_r':>6}"
           f"{'sp_r':>6}{'ens_r':>6}{'f4':>5}{'acc':>4}{'good':>5}")
    L.append(hdr)
    L.append("-" * len(hdr))
    for r in sorted([r for r in det if r["dom"] in ("outside", "manual")],
                    key=lambda r: r["corner_med"] or 1e9):
        def c(v, f="{:.1f}"):
            return f.format(v) if v is not None and np.isfinite(v) else " - "
        L.append(f"{r['dom']:<8}{r['fid'][:12]:<13}{c(r['corner_med']):>5}"
                 f"{c(r['rear_err']):>5}{c(r['photo_front']):>6}{c(r['photo_rear']):>6}"
                 f"{c(r['spread_rear']):>6}{c(r['ens_rear']):>6}"
                 f"{c(r['f_scores']['f4_tta_stab']):>5}"
                 f"{'  Y' if r['deploy_pass'] else '  n':>4}{'  Y' if r['good'] else '  n':>5}")
    L.append("```")
    L.append("")
    L.append("★ caveat: real low-angle domain-mixed, heuristic tau, 소표본(도메인별 accepted-bad "
             "특히 적음). ensemble-s1 = squash OOD 가능(domain-shift confound). night = base "
             "GT-good 붕괴라 어떤 신호도 순도 못 냄(참고만).")
    with open(OUT_MD, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n[save] {OUT_MD}\n[save] {OUT_JSON}")


if __name__ == "__main__":
    main()
