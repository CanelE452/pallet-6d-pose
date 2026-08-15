"""challenge0123_ft_manual 의 loss 곡선 PNG. challenge0123 baseline 과 비교."""
import os as _os, sys as _sys

# --- challenge/scripts 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_CS = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_CS] + [_os.path.join(_CS, _d) for _d in sorted(_os.listdir(_CS))
                         if _os.path.isdir(_os.path.join(_CS, _d)) and not _d.startswith(".")]

import sys, os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from glob import glob

def load_scalars(logdir):
    ef = sorted(glob(os.path.join(logdir, "**", "events.out.tfevents*"), recursive=True))
    if not ef:
        return {}
    ea = EventAccumulator(ef[-1]); ea.Reload()
    out = {}
    for tag in ea.Tags().get("scalars", []):
        ev = ea.Scalars(tag)
        out[tag] = ([e.step for e in ev], [e.value for e in ev])
    return out

base = load_scalars("weights/challenge_track/challenge0123/runs")
ft = load_scalars("weights/challenge_track/challenge0123_ft_manual/runs")

tags = ["loss/train_loss", "loss/train_bel", "loss/train_aff", "health/belief_peak_mean"]
fig, axs = plt.subplots(2, 2, figsize=(14, 8))
for ax, tag in zip(axs.flat, tags):
    if tag in base:
        ax.plot(base[tag][0], base[tag][1], 'b-', alpha=0.6, label=f"challenge0123 (60ep scratch)")
    if tag in ft:
        ax.plot(ft[tag][0], ft[tag][1], 'r-', lw=2, label=f"ft_manual (61-80, 104 frames)")
        ax.axvline(60, color='gray', ls='--', alpha=0.5, label='fine-tune start (ep 61)')
    ax.set_title(tag); ax.set_xlabel("epoch"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    if "loss" in tag: ax.set_yscale("log")

plt.tight_layout()
out = "data/pallet/results/ft_loss_curves.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=100, bbox_inches='tight')
print(f"saved {out}")

# 안정화 분석
if "loss/train_loss" in ft:
    steps, vals = ft["loss/train_loss"]
    ft_vals = np.array(vals)
    print(f"\nft_manual train_loss:")
    print(f"  ep 61: {ft_vals[0]:.4f}")
    print(f"  ep 70: {ft_vals[min(9, len(ft_vals)-1)]:.4f}")
    print(f"  ep 80: {ft_vals[-1]:.4f}")
    print(f"  마지막 5ep mean: {ft_vals[-5:].mean():.4f}, std: {ft_vals[-5:].std():.4f}")
