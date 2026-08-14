"""eval_edgevec_refine.py — STEP 7 sparse edge-vector mask-free refine 평가.

★ 핵심 판별 arm = sparse-refine: belief peak 9 키포인트(초기)를 예측 edge 단위방향
으로 **위치 least-squares 보정**(마스크 불필요). HybridPose 의 keypoint+edge 결합
최적화 단순판. dense voting(STEP4/5)이 진 게 마스크 문제였다면, 마스크 없는 refine
가 control-heatmap 보다 corner(특히 real back)를 낮춰야 한다.

3 arm (same-frame 페어):
  control-heatmap : off ckpt 의 belief peak (벡터 효과 0, baseline)
  sparse-heatmap  : edge ckpt 의 belief peak (multitask 부수효과만)
  sparse-refine   : edge ckpt belief peak + edge 방향 least-squares 보정 (★ 벡터 직접효과)

── refine 정식화 (linear least squares, closed-form) ──
  변수: 키포인트 위치 p_k ∈ R² (k=0..8), belief 격자 좌표.
  최소화:
     E(p) = λ_data Σ_k ||p_k - p_k^0||²                          (heatmap anchor)
          + λ_edge Σ_(i,j)∈E  w_e || (I - e_ij e_ijᵀ)(p_j - p_i) ||²   (방향 제약)
  - p_k^0 = belief peak 초기값. (I - e e^T) = e 에 수직인 성분만 추출 →
    edge 의 *방향*만 제약하고 *길이(스케일)* 는 자유 = heatmap 이 스케일 고정.
  - 무효(검출 안 된/peak 없는) 키포인트는 강한 anchor 로 고정(이동 안 함).
  - 각 항이 p 에 선형 → normal equation A p = b 한 번에 풀이(수렴 가드 불필요,
    안 풀리면 pinv fallback). edge 단위벡터는 ckpt 예측(=학습된 방향).

좌표/전처리/역매핑 = eval_pvnet_heads.py 재사용(import). 평가셋 = filter-val(real)/
manual/synthetic. split_metrics(overall/front/back) + good/gross + win count.
출력 → data/pallet/eval_results/stage7_sparse/.
"""

from __future__ import annotations
import os as _os, sys as _sys

# --- stage0 형제 탐색: 계열 폴더로 나뉘어 있어도 서로를 찾게 한다.
#     형제를 import 하는 줄보다 반드시 먼저 실행돼야 하므로 최상단에 둔다.
_S0 = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_S0] + [_os.path.join(_S0, _d) for _d in sorted(_os.listdir(_S0))
                         if _os.path.isdir(_os.path.join(_S0, _d)) and not _d.startswith(".")]


import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts", "data_prep", "eval"))
sys.path[:0] = [os.path.join(ROOT, "scripts", "data_prep", _s)
                for _s in ("plots", "filters")]
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "common"))
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "train"))

import cv2  # noqa: E402
import torch  # noqa: E402
from models import DopeNetwork  # noqa: E402
from geo_loss_struct import CUBOID_EDGES, CUBOID_DIAGONALS, EDGE_WEIGHTS  # noqa: E402
# eval_pvnet_heads 의 전처리/decode/평가 헬퍼 재사용(재정의 금지)
from eval_pvnet_heads import (  # noqa: E402
    preprocess, belief_to_orig, kp_to_pred8, heatmap_pred8, split_metrics,
    collect_manual, collect_syn, load_gt8, bucket, fmt_table,
    GOOD_PX, GROSS_PX, FRONT, BACK,
)
from filter_pr_camfacing import extract_keypoints_from_belief  # noqa: E402
from tau_calibrate import collect_val_frames  # noqa: E402

OUT_DIR = os.path.join(ROOT, "data", "pallet", "eval_results", "stage7_sparse")

STRUCT_EDGES = list(CUBOID_EDGES) + list(CUBOID_DIAGONALS)
# EDGE_WEIGHTS = 16개(12 edge + 4 diag) — STRUCT_EDGES 와 1:1 정렬.
EDGE_W = list(EDGE_WEIGHTS)


# ── 모델 로드 (numEdgeVec 자동 추론, strict=False) ─────────────────────────
def _edge_out_dim(state):
    """state_dict 의 edge FC 마지막 Linear out_features. 없으면 0."""
    ks = [k for k in state if k.startswith("m_edge_fc")
          and k.endswith(".weight")]
    if not ks:
        return 0
    # 마지막 Linear (가장 큰 인덱스의 .weight) out_features
    return state[sorted(ks)[-1]].shape[0]


def load_edge_model(weights_path, device):
    state = torch.load(weights_path, map_location=device)
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    numEdgeVec = _edge_out_dim(state)
    model = DopeNetwork(numEdgeVec=numEdgeVec)
    model.load_state_dict(state, strict=False)
    return model.to(device).eval(), numEdgeVec


# ── mask-free edge refine (closed-form linear least squares) ───────────────
def refine_keypoints(kp0, valid, edge_dirs, edges, edge_w,
                     lam_data=1.0, lam_edge=0.5, anchor_invalid=1e3):
    """kp0 (9,2) belief 초기 좌표, valid (9,) bool(검출됨), edge_dirs (E,2) 예측
    단위방향, edges list[(i,j)], edge_w list[float]. -> kp_ref (9,2).

    E(p)=λd Σ||p_k-p_k^0||² + λe Σ w_e ||(I-e e^T)(p_j-p_i)||²
    각 항 선형 → normal eq. (A p = b). 무효 kp 는 anchor 강하게(이동 X)."""
    nK = kp0.shape[0]
    # x, y 분리 풀이 (edge 방향 항이 x,y 결합이라 분리 불가) → 2D 동시 풀이.
    # 변수 벡터 z = [p0x,p0y, p1x,p1y, ...] (2K,)
    n = 2 * nK
    A = np.zeros((n, n), dtype=np.float64)
    b = np.zeros(n, dtype=np.float64)

    # data(anchor) term: λd (또는 무효 kp 는 anchor_invalid) * ||p_k - p0_k||²
    for k in range(nK):
        w = lam_data if valid[k] else anchor_invalid
        if not np.isfinite(kp0[k, 0]):
            # 초기값 없으면 0 으로 약 anchor (refine 이 채우게)
            p0 = np.array([0.0, 0.0])
            w = lam_data * 1e-3
        else:
            p0 = kp0[k]
        for c in range(2):
            idx = 2 * k + c
            A[idx, idx] += w
            b[idx] += w * p0[c]

    # edge direction term: λe w_e ||(I - e e^T)(p_j - p_i)||²
    for e, (i, j) in enumerate(edges):
        d = edge_dirs[e]
        if not np.isfinite(d).all() or np.linalg.norm(d) < 1e-6:
            continue
        d = d / (np.linalg.norm(d) + 1e-9)
        P = np.eye(2) - np.outer(d, d)         # (2,2) 수직 투영자
        M = P.T @ P                            # = P (idempotent) but keep general
        w = lam_edge * edge_w[e]
        # (p_j - p_i)^T M (p_j - p_i): block 기여
        # ∂/∂p contributes: for variables p_i,p_j with sign s_i=-1,s_j=+1
        for (a_node, sa) in ((i, -1.0), (j, +1.0)):
            for (b_node, sb) in ((i, -1.0), (j, +1.0)):
                for ca in range(2):
                    for cb in range(2):
                        A[2 * a_node + ca, 2 * b_node + cb] += \
                            w * sa * sb * M[ca, cb]
    try:
        z = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        z = np.linalg.pinv(A) @ b
    kp_ref = z.reshape(nK, 2)
    # 무효(미검출) kp 는 refine 결과 신뢰 안 함 → NaN 유지(원래 미검출)
    for k in range(nK):
        if not np.isfinite(kp0[k, 0]):
            kp_ref[k] = np.nan
    return kp_ref


def kps_bel_to_array(kps_bel):
    """extract_keypoints_from_belief 결과(9개 [x,y,..], x<0=미검출)->(9,2)+valid(9,)."""
    kp = np.full((9, 2), np.nan)
    valid = np.zeros(9, dtype=bool)
    for k in range(min(9, len(kps_bel))):
        if kps_bel[k][0] >= 0:
            kp[k] = (kps_bel[k][0], kps_bel[k][1])
            valid[k] = True
    return kp, valid


def predict_edge_dirs(edge_raw, n_edges):
    """edge_raw (2E,) raw FC -> (E,2) 단위방향."""
    pe = edge_raw.reshape(n_edges, 2)
    nrm = np.linalg.norm(pe, axis=1, keepdims=True)
    nrm[nrm < 1e-6] = 1.0
    return pe / nrm


def run_arm_heatmap(belief, threshold, bw, bh, nw, nh, sc, gt8):
    kps_bel = extract_keypoints_from_belief(belief, threshold)
    ph8 = heatmap_pred8(kps_bel, bw, bh, nw, nh, sc)
    return split_metrics(ph8, gt8), kps_bel


def run_evalset(control_model, edge_model, edge_n, frames, threshold, device,
                lam_data, lam_edge):
    arms = {"control-heatmap": [], "sparse-heatmap": [], "sparse-refine": []}
    wins = {"refine_vs_control": 0, "refine_vs_sparseHM": 0, "n_pair": 0}
    sanity = []  # synthetic refine sanity 기록용
    for dom, fid, jp, ip in frames:
        img = cv2.imread(ip)
        if img is None:
            continue
        gt8 = load_gt8(jp)
        tensor, nw, nh, sc = preprocess(img)
        tensor = tensor.to(device)

        # control-heatmap
        with torch.no_grad():
            out_c = control_model(tensor)
        bel_c = out_c[0][-1][0].cpu().numpy()
        bh, bw = bel_c.shape[1], bel_c.shape[2]
        m_ctrl, _ = run_arm_heatmap(bel_c, threshold, bw, bh, nw, nh, sc, gt8)
        arms["control-heatmap"].append(m_ctrl)

        # sparse model forward (belief + edge)
        with torch.no_grad():
            out_e = edge_model(tensor)
        beliefs_e, _, edge_raw_t = out_e
        bel_e = beliefs_e[-1][0].cpu().numpy()
        edge_raw = edge_raw_t[0].cpu().numpy()

        # sparse-heatmap
        m_shm, kps_bel = run_arm_heatmap(bel_e, threshold, bw, bh, nw, nh,
                                         sc, gt8)
        arms["sparse-heatmap"].append(m_shm)

        # sparse-refine
        kp0, valid = kps_bel_to_array(kps_bel)
        edge_dirs = predict_edge_dirs(edge_raw, edge_n)
        kp_ref = refine_keypoints(kp0, valid, edge_dirs, STRUCT_EDGES, EDGE_W,
                                  lam_data=lam_data, lam_edge=lam_edge)
        pr8 = kp_to_pred8(kp_ref, bw, bh, nw, nh, sc)
        m_ref = split_metrics(pr8, gt8)
        arms["sparse-refine"].append(m_ref)

        # win count (overall median 유한 비교)
        if (np.isfinite(m_ref["overall"]) and np.isfinite(m_ctrl["overall"])
                and np.isfinite(m_shm["overall"])):
            wins["n_pair"] += 1
            if m_ref["overall"] < m_ctrl["overall"]:
                wins["refine_vs_control"] += 1
            if m_ref["overall"] < m_shm["overall"]:
                wins["refine_vs_sparseHM"] += 1

        if dom == "synthetic" and len(sanity) < 5:
            sanity.append({
                "fid": fid, "n_det": int(valid.sum()),
                "refine_overall": (None if not np.isfinite(m_ref["overall"])
                                   else round(m_ref["overall"], 2)),
                "hm_overall": (None if not np.isfinite(m_shm["overall"])
                               else round(m_shm["overall"], 2)),
            })
    return arms, wins, sanity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control_weights", required=True,
                    help="edge_head 없이 학습한 control ckpt")
    ap.add_argument("--sparse_weights", required=True,
                    help="edge_head 로 학습한 sparse ckpt")
    ap.add_argument("--evalset", choices=["manual", "filter-val", "synthetic"],
                    default="manual")
    ap.add_argument("--n_frames", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--lam_data", type=float, default=1.0)
    ap.add_argument("--lam_edge", type=float, default=0.5)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.evalset == "manual":
        frames = collect_manual(args.n_frames)
    elif args.evalset == "synthetic":
        frames = collect_syn(args.n_frames)
    else:
        frames = collect_val_frames()
        frames = [f for f in frames if f[0] == "outside"]
        if args.n_frames > 0:
            frames = frames[:args.n_frames]
    print(f"[eval] evalset={args.evalset} frames={len(frames)}")

    control_model, ctrl_edge_n = load_edge_model(args.control_weights, device)
    edge_model, edge_n = load_edge_model(args.sparse_weights, device)
    print(f"[model] control numEdgeVec={ctrl_edge_n} "
          f"(0 기대) | sparse numEdgeVec={edge_n} (={2 * len(STRUCT_EDGES)} 기대)")
    if edge_n == 0:
        print("[WARN] sparse ckpt 에 edge 헤드 없음 → refine=heatmap 과 동일")
    edge_n_use = edge_n if edge_n > 0 else 2 * len(STRUCT_EDGES)

    arms, wins, sanity = run_evalset(
        control_model, edge_model, edge_n_use // 2, frames,
        args.threshold, device, args.lam_data, args.lam_edge)

    title = (f"STEP7 sparse edge-vector — evalset={args.evalset} "
             f"frames={len(frames)} (good<{GOOD_PX} gross>{GROSS_PX})")
    table = fmt_table(title, arms)
    win_line = (f"# win(overall<) refine_vs_control={wins['refine_vs_control']}"
                f"/{wins['n_pair']} refine_vs_sparseHM="
                f"{wins['refine_vs_sparseHM']}/{wins['n_pair']}")
    note = ("# refine=mask-free edge least-squares (방향제약, scale=heatmap 고정). "
            "낮추면 (나)마스크 문제 / 안낮추면 (가)벡터 불필요.")
    print("\n" + table + "\n" + win_line + "\n" + note + "\n")
    if sanity:
        print("# synthetic refine sanity (n_det / refine_ov / hm_ov):")
        for s in sanity:
            print(f"#   {s['fid']} n_det={s['n_det']} "
                  f"refine={s['refine_overall']} hm={s['hm_overall']}")

    suf = f"_{args.tag}" if args.tag else ""
    out_txt = os.path.join(OUT_DIR, f"eval_{args.evalset}{suf}.txt")
    with open(out_txt, "w") as f:
        f.write(table + "\n" + win_line + "\n" + note + "\n")
        for s in sanity:
            f.write(f"# sanity {s}\n")
    out_json = os.path.join(OUT_DIR, f"eval_{args.evalset}{suf}.json")
    json.dump({"evalset": args.evalset, "n_frames": len(frames),
               "lam_data": args.lam_data, "lam_edge": args.lam_edge,
               "wins": wins, "sanity": sanity,
               "arms": {arm: {"overall": bucket(m, "overall"),
                              "front": bucket(m, "front"),
                              "back": bucket(m, "back")}
                        for arm, m in arms.items()}},
              open(out_json, "w"), indent=2)
    print(f"[save] {out_txt}\n[save] {out_json}")


if __name__ == "__main__":
    main()
