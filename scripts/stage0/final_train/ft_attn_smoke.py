"""attention 이 실제로 공간을 가려 보는지 먼저 확인한다 -- 그림을 그리기 전에.

line branch 의 role query 는 12 개 엣지 각각에 대해 f50 의 2500 개 위치를
가중합한다.  그 가중치가 균일(1/2500)에 가까우면 "어디를 본다"는 그림은
의미가 없고, 집중돼 있어야 볼 곳이 있다.  그래서 학습 코드는 손대지 않고
`DirectHoughModel.descriptors()` 와 **같은 경로**를 재현하되 need_weights 만
켜서 가중치를 꺼내 통계를 낸다.

판정
    entropy  균일 = ln(2500) = 7.824 nats.  이보다 확실히 낮아야 집중.
    max      균일 = 4.0e-4.  이보다 훨씬 커야 peak 가 있는 것.
    role 간 상관  12 role 이 전부 같은 곳을 보면 role 별 그림은 무의미.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = "/home/minjae/Documents/github/pallet-pose"
for sub in ("scripts/stage0", "scripts/stage0/paper_s2", "scripts/stage0/multihead",
            "scripts/stage0/line", "scripts/stage0/real_eval", "challenge",
            "scripts/annotate", "scripts/stage0/final_train"):
    sys.path.insert(0, os.path.join(ROOT, sub))

import cv2                                        # noqa: E402
import paper_s2_real_eval as PRE                  # noqa: E402
import mh_data as MD                              # noqa: E402
import mh_screen as MS                            # noqa: E402
import ft_f0f3_eval as EV                         # noqa: E402
from mh_arms import DH                            # noqa: E402

GRID, ROLES, TOKENS = 50, 12, 2500


@torch.no_grad()
def attention_weights(model, f50):
    """`DirectHoughModel.descriptors()` 와 동일한 연산, need_weights 만 True.

    모델 파일을 고치지 않는 이유: 학습에 쓰인 경로가 그대로여야 그림이
    말하는 것과 학습된 것이 같다.
    """
    encoder = model.line.encoder
    batch = f50.shape[0]
    flat = f50.flatten(2).transpose(1, 2)
    coordinates = encoder.coordinates[None].expand(batch, -1, -1)
    tokens = encoder.to_token(torch.cat([flat, coordinates], -1))
    query = encoder.norm_query(encoder.queries.weight[None].expand(batch, -1, -1))
    _, weights = encoder.attention(query, tokens, tokens, need_weights=True,
                                   average_attn_weights=True)
    return weights.reshape(batch, ROLES, GRID, GRID)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=os.path.join(
        ROOT, "data/pallet/results/paper_s2_multihead/final_train/attention"))
    arguments = parser.parse_args()

    MS.deterministic()
    _, _, _, features = MS.lattice()
    model, ckpt = EV.load(1)
    print(f"checkpoint {ckpt}")

    key = EV.OPEN_SETS[0]
    jp, ip, label = EV.frames(key)[0]
    image = cv2.imread(ip)
    print(f"frame {os.path.basename(ip)}  {key}  {image.shape}")

    out = model(PRE.preprocess_squash(image).to(MD.DEV), features)
    print("forward keys:", {k: tuple(v.shape) if torch.is_tensor(v) else type(v).__name__
                            for k, v in out.items() if k != "beliefs"})
    print("beliefs:", len(out["beliefs"]), tuple(out["beliefs"][-1].shape))

    weights = attention_weights(model, out["f50"])
    print("attention:", tuple(weights.shape),
          "row sum", float(weights[0].sum(dim=(1, 2)).mean()))

    flat = weights[0].reshape(ROLES, TOKENS).double().cpu().numpy()
    entropy = -(flat * np.log(np.clip(flat, 1e-12, None))).sum(1)
    uniform_h, uniform_max = np.log(TOKENS), 1.0 / TOKENS

    print(f"\n균일 기준: entropy {uniform_h:.3f} nats, max {uniform_max:.2e}")
    print("role  entropy   max        max/uniform  top1%mass")
    print("─" * 56)
    for r in range(ROLES):
        order = np.sort(flat[r])[::-1]
        print(f" {r:2d}   {entropy[r]:7.3f}  {flat[r].max():.3e}  "
              f"{flat[r].max() / uniform_max:9.1f}x   "
              f"{order[:TOKENS // 100].sum():.3f}")

    correlation = np.corrcoef(flat)
    off = correlation[~np.eye(ROLES, dtype=bool)]
    print(f"\nrole 간 상관: mean {off.mean():.3f}  max {off.max():.3f}  "
          f"min {off.min():.3f}")

    belief = out["beliefs"][-1][0, :9].detach().cpu().numpy()
    print(f"\nbelief 9ch: max/ch " +
          " ".join(f"{belief[c].max():.2f}" for c in range(9)))

    concentrated = bool(entropy.mean() < uniform_h - 0.5)
    distinct = bool(off.mean() < 0.9)
    print(f"\n[판정] 집중 {concentrated}  role별상이 {distinct}  "
          f"-> 시각화 {'의미있음' if concentrated and distinct else '재검토 필요'}")

    os.makedirs(arguments.output, exist_ok=True)
    report = {"checkpoint": ckpt, "frame": os.path.relpath(ip, ROOT),
              "uniform_entropy": uniform_h, "uniform_max": uniform_max,
              "entropy_per_role": entropy.tolist(),
              "max_per_role": flat.max(1).tolist(),
              "role_correlation_mean": float(off.mean()),
              "role_correlation_max": float(off.max()),
              "belief_max_per_channel": belief.max(axis=(1, 2)).tolist(),
              "CONCENTRATED": concentrated, "ROLES_DISTINCT": distinct}
    path = os.path.join(arguments.output, "ATTENTION_SMOKE.json")
    open(path, "w").write(json.dumps(report, indent=2))
    print(f"-> {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()
