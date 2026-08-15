"""s1_ckpt_select.py — Paper-S1 (mask-aux) ckpt selection on synthetic val.

과거 교훈(s16): real 로 ckpt 고르면 과적합. synthetic held-out val 로 고른 뒤 real 판정.
  - Val-Old = data/pallet/training_data/val (held-out; paper_s1 학습셋에 없음)
  - paper_4pallet_mask_v1 은 학습에 100% 사용 -> held-out 없음(선택 기준 아님).

선택 기준: Val-Old V=8 corner_med 최소 + det 유지(붕괴 없음). ep64~70.
지표 order-free: Hungarian corner median/good%, det%(n_det>=6), honest full-8 reproj.
전처리: aspect-only (synthetic clean full-frame -> pad 불필요). eval_frame/agg/collect 는
  s16_ckpt_select 그대로 재사용(동일 인프라).

출력: data/pallet/eval_results/paper_s1/ckpt_select.{txt,json}
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

import torch  # noqa: E402
from s16_ckpt_select import collect, eval_frame, agg, VAL_OLD  # noqa: E402
from eval_stage11 import load_model  # noqa: E402

OUT_DIR = os.path.join(ROOT, "data/pallet/eval_results/paper_s1")
CKPT_DIR = os.path.join(ROOT, "weights/paper_s1/paper_s1_maskaux")
N_VAL = 300


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    frames = collect(VAL_OLD, N_VAL)
    print(f"[set] Val-Old: {len(frames)} frames")

    ckpts = {}
    for ep in range(64, 71):
        p = os.path.join(CKPT_DIR, f"net_epoch_{ep:04d}.pth")
        if ep == 70:
            # final_net_epoch_0070 == net_epoch_0070 (same ep); use final as canonical
            p = os.path.join(CKPT_DIR, "final_net_epoch_0070.pth")
        if os.path.exists(p):
            ckpts[f"ep{ep}"] = p

    out = {}
    txt = ["# Paper-S1 (mask-aux) ckpt selection — synthetic Val-Old (held-out); real is judgment-only",
           "# metric order-free: corner_med(px) good%(<10px) det%(n>=6) honest8(full-8 reproj px)",
           f"# Val-Old = {VAL_OLD} (N={len(frames)}); selection = V=8 corner_med min + det held",
           ""]
    for name, path in ckpts.items():
        model = load_model(path, device)
        rows = [eval_frame(model, jp, ip, device) for jp, ip in frames]
        rows = [r for r in rows if r is not None]
        a_all = agg(rows)
        a_v8 = agg([r for r in rows if r["occ"] == 8])
        out[name] = {"all": a_all, "V8": a_v8, "path": path}
        line = (f"{name:<8} ALL cm={str(a_all['corner_med']):>5} "
                f"g%={str(a_all['good_pct']):>5} d%={str(a_all['det_pct']):>5} "
                f"h8={str(a_all['honest8_med']):>5}  ||  "
                f"V8 cm={str(a_v8.get('corner_med')):>5} "
                f"g%={str(a_v8.get('good_pct')):>5} d%={str(a_v8.get('det_pct')):>5} "
                f"h8={str(a_v8.get('honest8_med')):>5} n={a_v8.get('n')}")
        txt.append(line)
        print(line)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # pick best: V=8 corner_med min, tie-break det% max then good% max
    def key(nm):
        v8 = out[nm]["V8"]
        cm = v8.get("corner_med")
        cm = 1e9 if cm is None else cm
        d = v8.get("det_pct") or 0
        g = v8.get("good_pct") or 0
        return (cm, -d, -g)
    best = min(out, key=key)
    txt.append("")
    txt.append(f"# BEST (V=8 corner_med min, det tie-break) = {best}  -> {out[best]['path']}")
    print(txt[-1])

    with open(os.path.join(OUT_DIR, "ckpt_select.txt"), "w") as f:
        f.write("\n".join(txt) + "\n")
    with open(os.path.join(OUT_DIR, "ckpt_select.json"), "w") as f:
        json.dump({"best": best, "ckpts": out}, f, indent=2,
                  default=lambda x: None
                  if isinstance(x, float) and not np.isfinite(x) else x)
    print(f"\nsaved: {OUT_DIR}/ckpt_select.txt , ckpt_select.json")


if __name__ == "__main__":
    main()
