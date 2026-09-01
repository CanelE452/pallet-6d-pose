"""YOLO26-Pose 가 어디를 보고 어떻게 틀리는지 — DOPE belief map 에 대응하는 시각화.

DOPE 는 keypoint 마다 belief map 을 내놓아서 "어디를 보고 있나" 를 그냥 그리면 된다.
YOLO26 은 좌표를 직접 회귀하므로 그런 map 이 없다 — 그래서 "시각화가 안 된다" 고
넘기기 쉬운데, 실제 head 출력을 보면 재구성할 수 있다.

    scores  (1, 1, 8400)    anchor 마다 objectness   -> 80/40/20 격자로 되돌리면 heatmap
    kpts    (1, 27, 8400)   ★anchor 마다 9-keypoint 전체를 예측한다

즉 8,400 개 anchor 가 각자 팔레트 전체 답안을 낸다.  그래서 두 가지를 그릴 수 있다.

    (1) objectness heatmap      DOPE 의 "물체가 여기 있다" 에 대응
    (2) keypoint vote map       anchor 별 keypoint 예측을 score 로 가중해 쌓은 것.
                                DOPE 의 keypoint belief map 에 직접 대응한다.
    (3) vote 산포                같은 keypoint 를 anchor 들이 얼마나 다르게 보는가
                                = 불확실성.  belief map 의 봉우리 폭에 해당한다.

(2)(3) 은 DOPE 처럼 학습된 map 이 아니라 예측을 모아 만든 것이다.  "모델이 그렇게
표현한다" 가 아니라 "모델의 예측을 이렇게 읽으면 보인다" 는 뜻이므로, 논문에 쓸 때
belief map 과 같은 것처럼 적지 말 것.
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import torch

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, ROOT)

PAD = 100
OUT = f"{ROOT}/data/pallet/results/yolo26_explain"
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]


def raw_forward(net, img_bgr, imgsz=640):
    """전처리는 추론 경로와 같게 — reflect-pad 100 후 letterbox 없이 resize."""
    padded = cv2.copyMakeBorder(img_bgr, PAD, PAD, PAD, PAD,
                                cv2.BORDER_REFLECT_101)
    h0, w0 = padded.shape[:2]
    x = cv2.resize(padded, (imgsz, imgsz))
    t = torch.from_numpy(x[:, :, ::-1].copy()).permute(2, 0, 1)[None].float() / 255
    with torch.no_grad():
        out = net(t)
    raw = out[1] if isinstance(out, (list, tuple)) else out
    head = net.model[-1]
    # ★one2many/one2one 의 kpts 는 **디코딩 전 raw offset** 이다 (격자 상대, +-13 범위).
    #   Pose26.kpts_decode 와 같은 식으로 (offset + anchor) * stride 를 해야 픽셀이 된다.
    #   이걸 빼먹으면 좌표가 전부 화면 밖으로 나가 vote map 이 텅 빈다.
    anchors = head.anchors.detach().numpy()          # (2, 8400)
    strides = head.strides.detach().numpy().reshape(-1)   # (8400,)
    return raw, (h0, w0), padded, anchors, strides


def split_scales(v, sizes=(80, 40, 20)):
    """(N, 8400) -> [(N,80,80), (N,40,40), (N,20,20)]"""
    out, i = [], 0
    for s in sizes:
        n = s * s
        out.append(v[:, i:i + n].reshape(-1, s, s))
        i += n
    return out


def heat(m, shape, cmap=cv2.COLORMAP_JET):
    m = np.asarray(m, np.float32)
    if m.max() > m.min():
        m = (m - m.min()) / (m.max() - m.min())
    m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
    return cv2.applyColorMap((m * 255).astype(np.uint8), cmap)


def blend(img, hm, a=0.55):
    return cv2.addWeighted(img, 1 - a, hm, a, 0)


def label(im, text, colour=(255, 255, 255)):
    bar = np.zeros((26, im.shape[1], 3), np.uint8)
    cv2.putText(bar, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1,
                cv2.LINE_AA)
    return np.vstack([bar, im])


def explain(weights, ip, tag):
    from ultralytics import YOLO
    net = YOLO(weights, task="pose").model.float().eval()
    img = cv2.imread(ip)
    if img is None:
        raise FileNotFoundError(ip)
    raw, (h0, w0), padded, anchors, strides = raw_forward(net, img)
    # ★one2one 은 anchor 를 딱 1개만 발화시킨다(NMS-free 설계).  vote 를 보려면
    #   밀집 감독을 받은 one2many 를 써야 한다 — 그래도 8,400 중 10개 남짓이다.
    d = raw["one2many"]
    # scores 는 logit 이다 — sigmoid 를 거쳐야 확률이 된다.
    scores = 1.0 / (1.0 + np.exp(-d["scores"][0, 0].numpy()))
    kpts = d["kpts"][0].numpy()                         # (27, 8400)
    sc_scales = split_scales(scores[None])              # 각 (1,s,s)
    feats = d["feats"]                                  # 백본 3스케일 특징

    # 원본 좌표계로 되돌리는 배율 (pad 포함 -> pad 제거)
    sx, sy = w0 / 640.0, h0 / 640.0
    kx = (kpts[0::3] + anchors[0]) * strides * sx - PAD   # (9, 8400)
    ky = (kpts[1::3] + anchors[1]) * strides * sy - PAD
    kc = 1.0 / (1.0 + np.exp(-kpts[2::3]))              # sigmoid

    H, W = img.shape[:2]
    tiles = []

    # ── (1) objectness heatmap, 스케일별
    for s, m in zip((80, 40, 20), sc_scales):
        tiles.append(label(blend(img, heat(m[0], (H, W))),
                           f"{tag}  objectness  {s}x{s} grid"))

    # ── (1b) feature activation — "어디를 보고 있나" 에 가장 가까운 것.
    # score map 은 anchor 10개만 살아 면도날처럼 뾰족해서 주의 분포를 못 보여준다.
    # 백본 특징의 채널 L2 노름을 쓰면 공간적으로 어디가 반응했는지 보인다.
    for f in feats:
        a_ = f[0].numpy()
        act = np.sqrt((a_ ** 2).sum(0))
        tiles.append(label(blend(img, heat(act, (H, W))),
                           f"{tag}  feature activation  {act.shape[0]}x{act.shape[1]}"))

    # ── (2) keypoint vote map — anchor 예측을 score 로 가중해 누적
    w = scores.copy()
    keep = w > 0.05                       # 확률 5% 이상인 anchor 만
    if keep.sum() < 8:                    # 너무 적으면 상위 32개로 채운다
        keep = np.zeros_like(keep)
        keep[np.argsort(-w)[:32]] = True
    vote = np.zeros((H, W), np.float32)
    per_kp = []
    for i in range(9):
        acc = np.zeros((H, W), np.float32)
        xs, ys, ws = kx[i][keep], ky[i][keep], (w * kc[i])[keep]
        ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
        for x_, y_, w_ in zip(xs[ok].astype(int), ys[ok].astype(int), ws[ok]):
            acc[y_, x_] += w_
        acc = cv2.GaussianBlur(acc, (0, 0), 6)
        per_kp.append(acc)
        if i < 8:
            vote += acc
    tiles.append(label(blend(img, heat(vote, (H, W))),
                       f"{tag}  keypoint vote map (8 corners, score-weighted)"))

    # ── (3) vote 산포 — anchor 들이 같은 코너를 얼마나 다르게 보나
    spread = np.zeros(9)
    for i in range(9):
        xs, ys, ws = kx[i][keep], ky[i][keep], w[keep]
        if ws.sum() > 0:
            mx = (xs * ws).sum() / ws.sum()
            my = (ys * ws).sum() / ws.sum()
            spread[i] = float(np.sqrt((((xs - mx) ** 2 + (ys - my) ** 2)
                                       * ws).sum() / ws.sum()))
    dis = img.copy()
    top = np.argsort(-scores)[:32]
    for a in top:
        for i in range(8):
            x_, y_ = int(kx[i, a]), int(ky[i, a])
            if 0 <= x_ < W and 0 <= y_ < H:
                cv2.circle(dis, (x_, y_), 1, (0, 200, 255), -1)
    best = int(np.argmax(scores))
    pts = [(int(kx[i, best]), int(ky[i, best])) for i in range(8)]
    for a, b in EDGES:
        cv2.line(dis, pts[a], pts[b], (0, 0, 255), 2, cv2.LINE_AA)
    for i, p in enumerate(pts):
        cv2.circle(dis, p, 4, (255, 0, 0), -1, cv2.LINE_AA)
    tiles.append(label(dis, f"{tag}  top-32 anchor votes (orange) + best anchor "
                            f"(red)   spread px: "
                            + " ".join(f"{s:.0f}" for s in spread[:8])))

    # ── (4) anchor 선택 — "왜 이 anchor 가 뽑혔나"
    # 상위 후보들이 각자 낸 답(9kp)을 겹쳐 그린다.  답이 서로 크게 다르면 선택이
    # 성능의 레버이고, 다들 비슷하면 선택은 무의미하다 — 그 판단을 눈으로 하게 한다.
    order = np.argsort(-scores)[:8]
    sel = img.copy()
    palette = [(0, 0, 255), (0, 140, 255), (0, 255, 255), (0, 255, 0),
               (255, 200, 0), (255, 0, 0), (255, 0, 200), (160, 160, 160)]
    rows = []
    bx = raw["one2many"]["boxes"][0].numpy()          # (4, 8400) ltrb, 격자 단위
    for rank, a in enumerate(order):
        col = palette[rank % len(palette)]
        pts = [(int(kx[i, a]), int(ky[i, a])) for i in range(8)]
        th = 3 if rank == 0 else 1
        for u, v in EDGES:
            cv2.line(sel, pts[u], pts[v], col, th, cv2.LINE_AA)
        # 이 anchor 가 격자 어디에 앉아 있는지
        ax = (anchors[0][a] * strides[a]) * sx - PAD
        ay = (anchors[1][a] * strides[a]) * sy - PAD
        cv2.circle(sel, (int(ax), int(ay)), 6 if rank == 0 else 4, col, -1,
                   cv2.LINE_AA)
        cv2.putText(sel, str(rank), (int(ax) + 6, int(ay) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2, cv2.LINE_AA)
        # rank0 대비 keypoint 가 얼마나 다른가
        d0 = float(np.median(np.hypot(kx[:8, a] - kx[:8, order[0]],
                                      ky[:8, a] - ky[:8, order[0]])))
        rows.append({"rank": rank, "anchor": int(a),
                     "score": float(scores[a]), "stride": float(strides[a]),
                     "grid_xy": [float(ax), float(ay)],
                     "kp_dist_from_top1_px": d0})
    txt = "  ".join(f"#{r['rank']}:{r['score']:.2f}/{r['kp_dist_from_top1_px']:.0f}px"
                    for r in rows)
    tiles.append(label(sel, f"{tag}  anchor selection — rank:score/kp-dist-from-top1"))
    tiles.append(label(np.zeros_like(sel), txt[:150]))

    # ── 개별 코너 belief 대응물 (0~7)
    kp_tiles = [label(blend(img, heat(per_kp[i], (H, W))), f"kp{i} vote")
                for i in range(8)]

    return tiles, kp_tiles, {"spread": spread.tolist(),
                             "max_score": float(scores.max()),
                             "n_anchor_kept": int(keep.sum()),
                             "candidates": rows}


def grid(tiles, cols, w=560):
    out = []
    for t in tiles:
        s = w / t.shape[1]
        out.append(cv2.resize(t, (w, int(t.shape[0] * s)),
                              interpolation=cv2.INTER_AREA))
    h = max(t.shape[0] for t in out)
    out = [np.vstack([t, np.zeros((h - t.shape[0], t.shape[1], 3), np.uint8)])
           for t in out]
    rows = []
    for i in range(0, len(out), cols):
        r = out[i:i + cols]
        while len(r) < cols:
            r.append(np.zeros_like(out[0]))
        rows.append(np.hstack(r))
    return np.vstack(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--weights", default=(
        "challenge/yolo_pose_one_model/spatial_concat_scratch/runs/"
        "YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42/weights/best.pt"))
    ap.add_argument("--tag", default="paper_yolo26n_joint")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    tiles, kp_tiles, info = explain(os.path.join(ROOT, a.weights),
                                    a.image, a.tag)
    stem = os.path.splitext(os.path.basename(a.image))[0][:40]
    cv2.imwrite(f"{a.out}/{stem}__overview.png", grid(tiles, 3))
    cv2.imwrite(f"{a.out}/{stem}__per_corner.png", grid(kp_tiles, 4))
    print(f"  max score {info['max_score']:.3f}   "
          f"쓴 anchor {info['n_anchor_kept']}/8400", flush=True)
    print("  코너별 vote 산포(px): "
          + " ".join(f"{s:.0f}" for s in info["spread"][:8]), flush=True)
    print("  후보 anchor (rank / score / stride / top1 과의 kp 거리):", flush=True)
    for r in info["candidates"]:
        print(f"    #{r['rank']}  score {r['score']:.3f}  stride {r['stride']:.0f}"
              f"  kp_dist {r['kp_dist_from_top1_px']:6.1f}px", flush=True)
    print(f"-> {a.out}/{stem}__overview.png", flush=True)
    print(f"-> {a.out}/{stem}__per_corner.png", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
