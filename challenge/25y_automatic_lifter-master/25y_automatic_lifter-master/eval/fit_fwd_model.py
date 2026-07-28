"""전진 t(d) piecewise 모델 refit.

스펙 ★보정3: 논문 식(13-14) 형태 고정.
스펙 ★보정5: d를 노이즈 변수로 두고 d=f(T) 적합 후 t(d)로 역산.

사용:
  python eval/fit_fwd_model.py eval/results/calib_fwd.csv
  python eval/fit_fwd_model.py eval/results/calib_fwd.csv --dir fwd
"""
from __future__ import annotations
import argparse
import csv
import math
import numpy as np
from scipy.optimize import curve_fit


def d_of_T(T, t0, t1, a):
    """명령시간 T(s) → 이동거리 d(m). 가속→정속 piecewise."""
    T = np.asarray(T, dtype=float)
    a = max(a, 1e-9)
    d_acc = 0.5 * a * t1 * t1
    vmax = a * t1
    tau = T - t0
    d = np.where(
        tau <= 0.0, 0.0,
        np.where(tau <= t1, 0.5 * a * np.square(np.clip(tau, 0.0, None)),
                 d_acc + vmax * (tau - t1)),
    )
    return d if d.ndim else float(d)


def t_of_d(d, t0, t1, a):
    """이동거리 d(m) → 명령시간 t(s). d_of_T의 해석적 역함수."""
    a = max(a, 1e-9)
    d_acc = 0.5 * a * t1 * t1
    vmax = a * t1
    if d <= d_acc:
        return t0 + math.sqrt(max(0.0, 2.0 * d / a))
    return t0 + t1 + (d - d_acc) / vmax


def fit_fwd(T, d):
    """(T, d) 측정점에 d=f(T) 적합. 반환: {t0,t1,a,vmax,d_acc,r2,rmse,n}."""
    T = np.asarray(T, dtype=float)
    d = np.asarray(d, dtype=float)
    p0 = [0.0, 4.0, 0.075]                       # t0, t1, a 초기추정
    bounds = ([-2.0, 0.1, 1e-3], [2.0, 30.0, 5.0])
    popt, _ = curve_fit(d_of_T, T, d, p0=p0, bounds=bounds, maxfev=20000)
    t0, t1, a = (float(x) for x in popt)
    pred = d_of_T(T, t0, t1, a)
    ss_res = float(np.sum((d - pred) ** 2))
    ss_tot = float(np.sum((d - np.mean(d)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(ss_res / len(d)))
    return {"t0": t0, "t1": t1, "a": a, "vmax": a * t1,
            "d_acc": 0.5 * a * t1 * t1, "r2": r2, "rmse": rmse, "n": len(d)}


def _load_calib(path, direction=None):
    T, d = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if direction and row.get("direction") != direction:
                continue
            T.append(float(row["T_sec"]))
            d.append(float(row["d_measured"]))
    return T, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="calib_fwd.csv 경로")
    ap.add_argument("--dir", default=None, help="fwd|back (생략 시 전체)")
    args = ap.parse_args()
    T, d = _load_calib(args.csv, args.dir)
    if len(T) < 3:
        raise SystemExit(f"적합에 최소 3점 필요 (현재 {len(T)}점). calib run 더 수집하세요.")
    r = fit_fwd(T, d)
    print(f"[fit dir={args.dir or 'all'}] n={r['n']}  R²={r['r2']:.4f}  RMSE={r['rmse']*1000:.1f}mm")
    print(f"  FWD_T0 = {r['t0']:.4f}")
    print(f"  FWD_T1 = {r['t1']:.4f}")
    print(f"  FWD_A  = {r['a']:.6f}   (vmax={r['vmax']:.4f} m/s, d_acc={r['d_acc']:.4f} m)")
    print("→ 위 값을 depth_cam/calib/config.py 의 FWD_T0/FWD_T1/FWD_A 에 반영 후 drive --eval 실행 (워크플로 c→d)")


if __name__ == "__main__":
    main()
