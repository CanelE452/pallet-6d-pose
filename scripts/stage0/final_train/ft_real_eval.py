"""Real evaluation of the FINAL40K checkpoints -- the first number that means
anything for this model.

Why this is valid while every synthetic number is not: FINAL_SYNTH_TRAIN_V1 is
BROAD synthetic only.  These frames are real photographs with manual GT and were
never in any training pool.

SEALED: the four final-test sessions (pallet07 / pallet09 / night08 / night09,
105 frames) are NOT touched here.  Spending that seal is the user's call, not a
side effect of answering "how good is it".  Only the 56 non-sealed canonical
eval frames are read.

Parity: FINAL40K trained on the anisotropic 400x400 squash (mh_data.load_frame),
so the squash decode path is the matching one -- `eval_frame_squash`, the same
function every other squash-trained model in this repo is measured with.  The
downstream metric path (order-free Hungarian corners, solve_pose with
PALLET_DIMS, honest full-8 reprojection) is shared, so these numbers sit on the
same axis as the historical ones.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, os.path.join(ROOT, "scripts", "stage0"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "stage0", "paper_s2"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "stage0", "multihead"))
sys.path.insert(0, os.path.join(ROOT, "challenge"))

import paper_s2_real_eval as PRE   # noqa: E402  eval_frame_squash + parity
import stage25_paperbase_eval as S  # noqa: E402  summarize / elev_of
import data_paths as DP             # noqa: E402
import mh_data as MD                # noqa: E402
import mh_splitlate as SL           # noqa: E402
import mh_arms as MA                # noqa: E402

OUT = os.path.join(ROOT, "data/pallet/results/paper_s2_multihead/final_train")
CKPT = os.path.join(ROOT, "weights/paper_s2/paper_s2_multihead")
STEP = 25000
# non-sealed canonical eval only.  FINAL_TEST stays untouched.
OPEN_SETS = [k for k in DP.EVAL_CANONICAL if k not in DP.FINAL_TEST]


class DopeLike(torch.nn.Module):
    """Expose SplitLate through the interface eval_frame_squash expects.

    It calls `model(x) -> (beliefs, affinities)`.  Routing through the corner
    branch is not a convenience: the corner head is what produces the belief
    maps the whole real-eval metric path decodes.
    """

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        stem = self.inner.early(x).detach()
        f50 = self.inner.corner_late(stem)
        beliefs, affinities, _ = MA.heads_from_f50(self.inner.net, f50)
        return beliefs, affinities


def load(seed):
    path = os.path.join(CKPT, f"screen_A1_CORNER_LINE_FINAL40K_seed{seed}",
                        f"step_{STEP:05d}.pth")
    state = torch.load(path, map_location=MD.DEV, weights_only=False)
    inner = SL.SplitLate(state["arm"])
    inner.load_state_dict(state["model"])
    model = DopeLike(inner).to(MD.DEV)
    model.eval()
    return model, path


def frames(set_key):
    folder = os.path.join(ROOT, DP.EVAL_CANONICAL[set_key])
    out = []
    for name in sorted(os.listdir(folder)):
        if not name.endswith(".json"):
            continue
        jp = os.path.join(folder, name)
        try:
            payload = json.load(open(jp))
        except Exception:
            continue
        objects = payload.get("objects") or []
        if not objects or objects[0].get("split") != "eval":
            continue
        for ext in (".png", ".jpg", ".jpeg"):
            ip = jp[:-5] + ext
            if os.path.exists(ip):
                out.append((jp, ip))
                break
    return out


def main():
    report = {
        "scope": "REAL manual GT, non-sealed canonical eval only",
        "sealed_not_touched": list(DP.FINAL_TEST),
        "why_valid": "training pool was BROAD synthetic only; these are real "
                     "photographs never seen in training",
        "parity": "squash decode (model trained on 400x400 anisotropic squash); "
                  "shared order-free metric path via eval_frame_squash",
        "threshold_note": "no threshold is chosen here. score_4kp stays "
                          "UNSET_PENDING_REAL_DEV.",
        "sets": {}, "seeds": {},
    }
    counts = {}
    for key in OPEN_SETS:
        counts[key] = len(frames(key))
    report["frame_counts"] = counts
    total = sum(counts.values())
    print(f"  non-sealed eval frames: {counts} (총 {total})", flush=True)

    for seed in (1, 2):
        model, path = load(seed)
        per_set, all_rows = {}, []
        for key in OPEN_SETS:
            rows = []
            for jp, ip in frames(key):
                row = PRE.eval_frame_squash(model, jp, ip, MD.DEV)
                if row is not None:
                    rows.append(row)
            if rows:
                per_set[key] = S.summarize(rows)
                all_rows.extend(rows)
        block = {"checkpoint": os.path.relpath(path, ROOT),
                 "per_set": per_set,
                 "ALL": S.summarize(all_rows) if all_rows else None}
        report["seeds"][f"seed{seed}"] = block
        a = block["ALL"]
        print(f"  seed{seed} ALL n={a['n']} det {a['det_pct']}% "
              f"corner {a['corner_med']} front {a['front_med']} "
              f"rear {a['rear_med']} honest8 {a['honest8_med']} "
              f"pnp {a['pnp_pct']}% good {a['good_pct']}% "
              f"gross {a['gross_pct']}%", flush=True)
        del model
        torch.cuda.empty_cache()

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "FINAL_REAL_EVAL.json"), "w") as fh:
        json.dump(report, fh, indent=1, default=str)
    print("-> FINAL_REAL_EVAL.json")


if __name__ == "__main__":
    main()
