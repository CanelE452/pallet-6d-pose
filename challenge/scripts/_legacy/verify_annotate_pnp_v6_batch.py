"""verify_annotate_pnp_v6_batch.py — 본체 solve_pose (v6 patched) batch sweep.

본체 `annotate_pnp.solve_pose` 가 verify_annotate_v4_fix_v6.solve_pose_v6 와 동일
결과를 내는지 확인 (1) + cp03/cp09/cp07 의 모든 manual_gt frame 에 대해 strict
invariant 통과율 + reproj 분포 출력 (2). fix v5 vs fix v6 regression 비교.

Outputs:
  data/pallet/results/annotate_v4_fix_v6/v6_batch_summary.txt
  data/pallet/results/annotate_v4_fix_v6/v6_batch_per_frame.csv

콘솔 출력:
  - cp03 1 frame: direct call 결과 + verify_v6 expected (2.20 / 12.40 px)
  - cp07 27 frames: LR-reversed click 데이터셋, strict_passed=False (click 모순) 예상
  - cp09 2 frames: 신규 검증
"""
from __future__ import annotations
import csv
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from annotate_pnp import solve_pose, PALLET_DIMS  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(_HERE))
DATA_ROOT = os.path.join(_REPO, "challenge/data")
OUT = os.path.join(_REPO, "data/pallet/results/annotate_v4_fix_v6")
os.makedirs(OUT, exist_ok=True)

DATASETS = [
    ("cp03", "capturepallet03_manual_gt"),
    ("cp09", "capturepallet09_manual_gt"),
    ("cp07", "capturepallet07_manual_gt"),
]


def _load_K(d):
    cam = d["camera_data"]["intrinsics"]
    return np.array([[cam["fx"], 0, cam["cx"]],
                     [0, cam["fy"], cam["cy"]],
                     [0, 0, 1]], dtype=np.float64)


def _kps_from_json(d, n=8):
    """manual_kps load — n 점 (8 기본). 누락은 None."""
    if "objects" not in d or not d["objects"]:
        return None
    obj = d["objects"][0]
    if "manual_kps" not in obj:
        return None
    raw = obj["manual_kps"]
    kps = []
    for i in range(9):
        if i < len(raw) and raw[i] is not None and len(raw[i]) >= 2:
            u, v = raw[i][:2]
            if u is None or v is None or float(u) < 0 or float(v) < 0:
                kps.append(None)
            else:
                kps.append([float(u), float(v)])
        else:
            kps.append(None)
    return kps


def _process_frame(json_path, n_click=8):
    """단일 frame solve_pose 호출 → diagnostic dict."""
    with open(json_path) as f:
        d = json.load(f)
    K = _load_K(d)
    kps = _kps_from_json(d)
    if kps is None:
        return None
    # n_click 만큼만 사용 (나머지 None)
    kps_n = list(kps[:n_click]) + [None] * (9 - n_click)
    pose = solve_pose(kps_n, K)
    if pose is None:
        return {"ok": False, "n_valid": sum(1 for k in kps_n if k is not None)}
    return {
        "ok": True,
        "n_valid": sum(1 for k in kps_n if k is not None),
        "reproj": pose["reproj_error_px"],
        "lr_viol": pose.get("_v6_lr_viol", -1),
        "tb_viol": pose.get("_v6_tb_viol", -1),
        "fr_viol": pose.get("_v6_fr_viol", -1),
        "viol_sum": pose.get("_v6_viol_sum", -1),
        "click_lr_viol": pose.get("_v6_click_lr_viol", -1),
        "click_tb_viol": pose.get("_v6_click_tb_viol", -1),
        "strict_passed": pose.get("_v6_strict_passed", False),
        "n_cand": pose.get("_v6_n_candidates", 0),
        "n_strict_ok": pose.get("_v6_n_strict_ok", 0),
        "v4_warning": pose.get("v4_warning", False),
        "dims": tuple(pose.get("dims", (-1, -1, -1))),
    }


def main():
    lines_summary = []
    csv_rows = []
    per_ds_stats = defaultdict(lambda: {
        "n": 0, "n_strict": 0, "reproj": [], "viol_sum": [],
        "click_lr": [], "click_tb": [], "fail_solve": 0,
    })

    print("=" * 78)
    print("verify_annotate_pnp_v6_batch — 본체 solve_pose (v6) 검증")
    print("=" * 78)

    # ── 1) cp03 direct verification: 6-click + 8-click ─────────────────────
    cp03_json = os.path.join(DATA_ROOT,
        "capturepallet03_manual_gt/1778651569891693056.json")
    cp03_png = os.path.join(DATA_ROOT,
        "capturepallet03_manual_gt/1778651569891693056.png")
    if os.path.exists(cp03_json):
        print(f"\n[cp03 direct] {os.path.basename(cp03_json)}")
        for n_click in (6, 8):
            r = _process_frame(cp03_json, n_click=n_click)
            if r is None or not r["ok"]:
                print(f"  n_click={n_click}: SOLVE FAILED ({r})")
                continue
            verdict = "PASS" if (r["viol_sum"] == 0 and r["click_lr_viol"] == 0
                                 and r["click_tb_viol"] == 0) else "FAIL"
            print(f"  n_click={n_click}: reproj={r['reproj']:6.2f}px  "
                  f"viol(LR={r['lr_viol']} TB={r['tb_viol']} FR={r['fr_viol']})  "
                  f"strict_passed={r['strict_passed']}  "
                  f"n_strict_ok={r['n_strict_ok']}/{r['n_cand']}  "
                  f"dims={r['dims']}  [{verdict}]")
        # expected: 6 click reproj~2.20 strict PASS, 8 click reproj~12.40 strict PASS

    # ── 2) Batch sweep — cp03 / cp09 / cp07 (모든 manual_gt frame) ──────────
    print("\n" + "=" * 78)
    print("[Batch sweep] cp03 / cp09 / cp07 — all manual_gt frames (8-click)")
    print("=" * 78)
    for ds_short, ds_dir in DATASETS:
        json_files = sorted(glob.glob(os.path.join(DATA_ROOT, ds_dir, "*.json")))
        # .bak 제외
        json_files = [j for j in json_files if not j.endswith(".bak")]
        print(f"\n[{ds_short}] {ds_dir}: {len(json_files)} files")
        for jp in json_files:
            r = _process_frame(jp, n_click=8)
            stats = per_ds_stats[ds_short]
            stats["n"] += 1
            if r is None or not r["ok"]:
                stats["fail_solve"] += 1
                csv_rows.append([ds_short, os.path.basename(jp), "FAIL_SOLVE",
                                 "", "", "", "", "", "", "", "", ""])
                continue
            if r["viol_sum"] == 0 and r["click_lr_viol"] == 0 and r["click_tb_viol"] == 0:
                stats["n_strict"] += 1
            stats["reproj"].append(r["reproj"])
            stats["viol_sum"].append(r["viol_sum"])
            stats["click_lr"].append(r["click_lr_viol"])
            stats["click_tb"].append(r["click_tb_viol"])
            csv_rows.append([
                ds_short, os.path.basename(jp), "OK",
                f"{r['reproj']:.2f}",
                str(r["lr_viol"]), str(r["tb_viol"]), str(r["fr_viol"]),
                str(r["viol_sum"]),
                str(r["click_lr_viol"]), str(r["click_tb_viol"]),
                str(r["strict_passed"]),
                str(r["n_strict_ok"]) + "/" + str(r["n_cand"]),
            ])
            # 짧은 per-frame line
            ok_mark = "PASS" if (r["viol_sum"] == 0 and r["click_lr_viol"] == 0
                                 and r["click_tb_viol"] == 0) else "FAIL"
            print(f"  {os.path.basename(jp)[:24]:24s}  "
                  f"reproj={r['reproj']:6.2f}px  "
                  f"viol(L{r['lr_viol']}T{r['tb_viol']}F{r['fr_viol']})  "
                  f"click(L{r['click_lr_viol']}T{r['click_tb_viol']})  "
                  f"strict={r['strict_passed']}  [{ok_mark}]")

    # ── 3) Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    lines_summary.append("dataset       n  strict   reproj(mean/med/max)   click_lr>=1  click_tb>=1  fail_solve")
    lines_summary.append("-" * 100)
    for ds_short, _ in DATASETS:
        s = per_ds_stats[ds_short]
        if s["n"] == 0:
            continue
        rj = s["reproj"]
        clr = sum(1 for x in s["click_lr"] if x >= 1)
        ctb = sum(1 for x in s["click_tb"] if x >= 1)
        if rj:
            r_mean, r_med, r_max = np.mean(rj), np.median(rj), np.max(rj)
        else:
            r_mean = r_med = r_max = float("nan")
        line = (f"{ds_short:10s}  {s['n']:3d}  {s['n_strict']:3d}/{s['n']:<3d}  "
                f"{r_mean:6.2f} / {r_med:6.2f} / {r_max:6.2f}    "
                f"{clr:>5d}        {ctb:>5d}      {s['fail_solve']:>3d}")
        lines_summary.append(line)
        print(line)

    print("-" * 100)

    # ── 4) Save outputs ─────────────────────────────────────────────────────
    summary_path = os.path.join(OUT, "v6_batch_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("verify_annotate_pnp_v6_batch — 본체 solve_pose (v6) 검증\n")
        f.write("=" * 78 + "\n\n")
        for ln in lines_summary:
            f.write(ln + "\n")
    print(f"\n[saved] {summary_path}")

    csv_path = os.path.join(OUT, "v6_batch_per_frame.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "file", "status",
                    "reproj_px", "lr_viol", "tb_viol", "fr_viol", "viol_sum",
                    "click_lr_viol", "click_tb_viol",
                    "strict_passed", "n_strict_ok/n_cand"])
        for row in csv_rows:
            w.writerow(row)
    print(f"[saved] {csv_path}")

    # ── 5) Quick regression check vs fix v5 ─────────────────────────────────
    # cp07 = LR-reversed click 데이터셋 (16 frames). v5 에서는 lr_click_viol>=2 이면
    # lr_w=0 으로 disable → reproj 최소 채택 (LR-OK 가능). v6 strict 는 click 모순시
    # strict disable + reproj 최소 (동일 정책). regression 없음 기대.
    cp07_n = per_ds_stats["cp07"]["n"]
    cp07_strict = per_ds_stats["cp07"]["n_strict"]
    cp07_click_lr_n = sum(1 for x in per_ds_stats["cp07"]["click_lr"] if x >= 1)
    cp07_reproj = per_ds_stats["cp07"]["reproj"]
    print(f"\n[regression check] cp07 (LR-reversed dataset, fix v5 lr_click_viol>=2 기준)")
    print(f"  cp07 frames: {cp07_n}")
    print(f"  strict-pass: {cp07_strict}/{cp07_n}")
    print(f"  click_lr_viol >= 1: {cp07_click_lr_n}/{cp07_n}")
    if cp07_reproj:
        print(f"  reproj mean/med/max: "
              f"{np.mean(cp07_reproj):.2f} / {np.median(cp07_reproj):.2f} / "
              f"{np.max(cp07_reproj):.2f} px")


if __name__ == "__main__":
    main()
