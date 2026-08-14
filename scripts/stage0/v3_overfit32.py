"""v3_overfit32.py — STAGE10 M2 PVNet (unit-vec + seg, ★real mask) 구현 게이트.

목적: 새 v3 합성셋의 ★진짜 per-pixel mask(mask_rle 디코드)로 PVNet seg + unit-vec
voting 구현(RLE decode·좌표·loss·voting)이 옳은지 32장 암기로 검증. STEP5 voting
실패의 유일 confound 였던 "박스투영(cuboid hull) 마스크 과포함"을 real mask 로 제거.

offset_overfit32.py 변형:
  - GT mask = mask_rle 디코드(real per-pixel), NOT cuboid hull.
  - vec = unit 벡터(각 픽셀 → 9 키포인트 방향), real mask 내부만 supervise (mask 밖
    0-vector 정답 금지 = loss 미계산).
  - 학습 헤드 = m_vec1/m_vec2(numVec=18) + m_seg1/m_seg2(numSeg=1). base frozen.
  - loss = seg BCE(2스테이지) + vec_loss(mask 내부, 2스테이지). NaN 가드.

전처리(offset_overfit32 와 동일): 고정 400×400 resize(aspect 왜곡 허용 — overfit
단순화. 키포인트/마스크도 동일 스케일 변환하므로 일관). grid=50, stride=8.

통과 판정:
  (a) seg sigmoid>0.5 vs GT mask(50px) IoU → ~1
  (b) GT-mask 로 vote_unit_ransac decode 한 9 키포인트 corner err(GT 대비, output cell)
      median < 1.  (pred-mask voting 도 참고 출력.)

데이터: challenge/data/02_synthetic/training/v3/batch_000 앞 N장(occlusion 적은 것 우선), aug 없음.
출력: data/pallet/eval_results/stage10_v3mask/overfit32/
★ 전체학습/2-4k quick screen 금지 — 32 overfit 게이트까지만.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "Deep_Object_Pose", "common"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from models import DopeNetwork  # noqa: E402
from utils_pvnet import make_vector_field  # noqa: E402

# 재정의 금지 — eval_pvnet_heads 의 voting / seg 디코드, vec_loss 재사용
from eval_pvnet_heads import vote_unit_ransac, seg_pred_mask  # noqa: E402
from measure_lambda_gradients import vec_loss  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
INPUT = 400
GRID = 50
STRIDE = INPUT / GRID  # 8.0
OUT_DIR = os.path.join(ROOT, "data", "pallet", "eval_results",
                       "stage10_v3mask", "overfit32")
V3_DIR = os.path.join(ROOT, "challenge", "data", "training", "v3", "batch_000")
DEFAULT_WEIGHTS = os.path.join(ROOT, "weights", "challenge0123",
                               "final_net_epoch_0060.pth")


# ── mask_rle 디코드 (검증된 uncompressed COCO RLE, column-major) ──────────────
def decode_mask_rle(rle):
    """uncompressed COCO RLE -> (H, W) uint8 {0,1}. 시작값 0 토글, column-major."""
    sz = rle["size"]                     # [H, W]
    flat = np.zeros(sz[0] * sz[1], np.uint8)
    idx = 0
    val = 0
    for c in rle["counts"]:
        flat[idx:idx + c] = val
        idx += c
        val ^= 1
    mask = flat.reshape((sz[1], sz[0])).T  # (H, W)
    return mask


# ── 모델 ──────────────────────────────────────────────────────────────────
def load_frozen_base(weights, device):
    """base ckpt(belief+affinity) 를 numVec=18+numSeg=1 모델에 strict=False 로 로드.
    vec/seg 헤드만 새 파라미터(missing). base 파라미터 전부 freeze, eval."""
    model = DopeNetwork(numVec=18, numSeg=1)
    state = torch.load(weights, map_location=device)
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    vec_seg_missing = [m for m in missing
                       if m.startswith("m_vec") or m.startswith("m_seg")]
    other_missing = [m for m in missing
                     if not (m.startswith("m_vec") or m.startswith("m_seg"))]
    assert not other_missing, f"unexpected base missing keys: {other_missing}"
    assert not unexpected, f"unexpected keys in ckpt: {unexpected}"
    print(f"[load] strict=False missing(vec/seg head)={len(vec_seg_missing)} "
          f"other_missing={len(other_missing)} unexpected={len(unexpected)}")
    model.to(device)
    # base freeze: vec/seg 헤드만 학습
    for name, p in model.named_parameters():
        train = name.startswith("m_vec") or name.startswith("m_seg")
        p.requires_grad_(train)
    return model


def preprocess_fixed(img):
    """고정 400×400 (aspect 왜곡 허용 — overfit 단순화). 반환 tensor, sx, sy."""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h0, w0 = rgb.shape[:2]
    r = cv2.resize(rgb, (INPUT, INPUT))
    t = ((r.astype(np.float32) / 255.0 - MEAN) / STD)
    tensor = torch.from_numpy(t.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor, INPUT / w0, INPUT / h0


def mask_orig_to_grid(mask_orig, sx, sy):
    """real mask (H0,W0) -> 400×400 (동일 aspect 왜곡 resize) -> 50×50 area-preserving.
    INTER_AREA 후 >0.5 (downsample 시 면적 보존). 반환 (50,50) uint8 {0,1}."""
    m400 = cv2.resize(mask_orig.astype(np.float32), (INPUT, INPUT),
                      interpolation=cv2.INTER_NEAREST)
    m50 = cv2.resize(m400, (GRID, GRID), interpolation=cv2.INTER_AREA)
    return (m50 > 0.5).astype(np.uint8)


def load_sample(jp, ip):
    """반환: tensor, kps9_grid(9,2 output px), mask50(50,50), gt8_grid(8,2 output px),
    occ(num_corners_unoccluded). real mask = mask_rle 디코드."""
    d = json.load(open(jp))
    o = d["objects"][0]
    gt8_orig = np.array(o["projected_cuboid"], float)[:8]
    ctr_orig = np.array(o["projected_cuboid_centroid"], float)
    occ = int(o.get("num_corners_unoccluded", 0))
    mask_orig = decode_mask_rle(o["mask_rle"])  # (480,640)
    area_json = int(o.get("mask_area_px", -1))

    img = cv2.imread(ip)
    tensor, sx, sy = preprocess_fixed(img)
    # orig px -> input px -> output px (/STRIDE)
    kps9_orig = np.vstack([gt8_orig, ctr_orig[None, :]])  # (9,2)
    kps9_grid = np.stack([kps9_orig[:, 0] * sx, kps9_orig[:, 1] * sy], 1) / STRIDE
    gt8_grid = kps9_grid[:8]
    mask50 = mask_orig_to_grid(mask_orig, sx, sy)
    return (tensor, kps9_grid, mask50, gt8_grid, occ,
            mask_orig, area_json, sx, sy)


def iou(a, b):
    a = a.astype(bool)
    b = b.astype(bool)
    inter = (a & b).sum()
    uni = (a | b).sum()
    return float(inter) / float(uni) if uni > 0 else 1.0


# ── sanity: mask_rle 정합 + vec GT assert + overlay ──────────────────────────
def run_sanity(samples, out_dir):
    s = samples[0]
    fid = s["fid"]
    mask_orig = s["mask_orig"]
    area_json = s["area_json"]
    area_dec = int(mask_orig.sum())

    # (1) mask_rle area vs json mask_area_px, 그리고 cuboid bbox 와 겹침
    gt8_orig = s["gt8_orig"]
    x0, y0 = gt8_orig[:, 0].min(), gt8_orig[:, 1].min()
    x1, y1 = gt8_orig[:, 0].max(), gt8_orig[:, 1].max()
    H, W = mask_orig.shape
    bx0, by0 = int(max(0, np.floor(x0))), int(max(0, np.floor(y0)))
    bx1, by1 = int(min(W, np.ceil(x1))), int(min(H, np.ceil(y1)))
    bbox_mask = np.zeros_like(mask_orig)
    bbox_mask[by0:by1, bx0:bx1] = 1
    inside = int((mask_orig & bbox_mask).sum())
    frac_in_bbox = inside / max(1, area_dec)

    # (2) vec GT assert: 마스크 픽셀 + unit_vec * dist == kp 회복 (수치)
    mask50 = s["mask50"]
    kps9 = s["kps9_grid"]
    vec, m = make_vector_field(kps9, GRID, mask=mask50, unit=True)
    ys, xs = np.nonzero(m)
    # unit 벡터라 크기 복원 불가 → 방향 일치로 assert (각 픽셀→kp 방향과 cos≈1)
    max_cos_err = 0.0
    for k in range(9):
        dx = vec[2 * k][ys, xs]
        dy = vec[2 * k + 1][ys, xs]
        # GT 방향
        tx = kps9[k, 0] - xs
        ty = kps9[k, 1] - ys
        tn = np.sqrt(tx * tx + ty * ty)
        ok = tn > 1e-6
        # 예측 unit (vec) 과 GT 단위방향 cos
        cos = (dx[ok] * (tx[ok] / tn[ok]) + dy[ok] * (ty[ok] / tn[ok]))
        if ok.sum() > 0:
            max_cos_err = max(max_cos_err, float(np.max(np.abs(cos - 1.0))))
    # offset(non-unit) 으로도 정확 회복 검증 (픽셀 + offset == kp)
    vec_off, _ = make_vector_field(kps9, GRID, mask=mask50, unit=False)
    max_recover_err = 0.0
    for k in range(9):
        dx = vec_off[2 * k][ys, xs]
        dy = vec_off[2 * k + 1][ys, xs]
        rec_x = xs + dx
        rec_y = ys + dy
        e = np.max(np.sqrt((rec_x - kps9[k, 0]) ** 2 + (rec_y - kps9[k, 1]) ** 2))
        max_recover_err = max(max_recover_err, float(e))

    # overlay: 원본 이미지 + mask(반투명) + unit-vec quiver (subsample)
    img = cv2.imread(s["ip"])
    overlay = img.copy()
    # mask_orig 를 빨강으로
    red = np.zeros_like(img)
    red[..., 2] = 255
    mo = mask_orig.astype(bool)
    overlay[mo] = (0.5 * overlay[mo] + 0.5 * red[mo]).astype(np.uint8)
    # quiver: 50px grid 마스크 픽셀에서 centroid(kp8) 방향 unit-vec, orig 좌표로
    sx, sy = s["sx"], s["sy"]
    step = 3
    for i in range(0, len(xs), step):
        gx, gy = xs[i], ys[i]
        # grid -> orig px
        ox = gx * STRIDE / sx
        oy = gy * STRIDE / sy
        dx = vec[2 * 8][gy, gx]  # centroid(k=8) unit dir
        dy = vec[2 * 8 + 1][gy, gx]
        ex = ox + dx * 12
        ey = oy + dy * 12
        cv2.arrowedLine(overlay, (int(ox), int(oy)), (int(ex), int(ey)),
                        (0, 255, 0), 1, tipLength=0.3)
    # 키포인트 표시
    for k in range(9):
        ox = kps9[k, 0] * STRIDE / sx
        oy = kps9[k, 1] * STRIDE / sy
        c = (0, 255, 255) if k < 8 else (255, 0, 255)
        cv2.circle(overlay, (int(ox), int(oy)), 4, c, -1)
        cv2.putText(overlay, str(k), (int(ox) + 4, int(oy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)
    ov_path = os.path.join(out_dir, f"sanity_overlay_{fid}.jpg")
    cv2.imwrite(ov_path, overlay)

    sanity = {
        "fid": fid,
        "mask_area_decoded": area_dec,
        "mask_area_json": area_json,
        "area_match": bool(area_dec == area_json),
        "frac_mask_in_cuboid_bbox": round(frac_in_bbox, 4),
        "vec_unit_max_cos_err": round(max_cos_err, 6),
        "vec_offset_max_recover_err_cell": round(max_recover_err, 6),
        "overlay": ov_path,
        "n_mask_px_50grid": int(mask50.sum()),
    }
    return sanity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam_seg", type=float, default=1.0)
    ap.add_argument("--lam_vec", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ransac_hyp", type=int, default=256)
    ap.add_argument("--inlier_deg", type=float, default=2.0)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 데이터: occlusion 적은 것(unoccluded 많은 순) 우선 N장 ──
    files = sorted(glob.glob(os.path.join(V3_DIR, "*.json")))
    cand = []
    for jp in files:
        fid = os.path.splitext(os.path.basename(jp))[0]
        ip = os.path.join(V3_DIR, fid + ".png")
        if not os.path.exists(ip):
            continue
        d = json.load(open(jp))
        o = d["objects"][0]
        occ = int(o.get("num_corners_unoccluded", 0))
        cand.append((occ, fid, jp, ip))
    # unoccluded 많은 것 우선, 동률은 fid 순
    cand.sort(key=lambda x: (-x[0], x[1]))
    cand = cand[:args.n]

    samples = []
    for occ, fid, jp, ip in cand:
        (tensor, kps9_grid, mask50, gt8_grid, occ2,
         mask_orig, area_json, sx, sy) = load_sample(jp, ip)
        # centroid 가 grid 밖이면 skip (안전)
        if mask50.sum() < 4:
            continue
        vec_gt, _ = make_vector_field(kps9_grid, GRID, mask=mask50, unit=True)
        d = json.load(open(jp))
        gt8_orig = np.array(d["objects"][0]["projected_cuboid"], float)[:8]
        samples.append({
            "fid": fid, "ip": ip,
            "tensor": tensor.to(device),
            "kps9_grid": kps9_grid,
            "gt8_grid": gt8_grid,
            "gt8_orig": gt8_orig,
            "mask50": mask50,
            "mask_orig": mask_orig,
            "area_json": area_json,
            "occ": occ2, "sx": sx, "sy": sy,
            "vec_gt": torch.from_numpy(vec_gt).unsqueeze(0).to(device),
            "mask_gt": torch.from_numpy(
                mask50[None, None].astype(np.float32)).to(device),
        })
    print(f"[data] {len(samples)} samples (requested {args.n})")

    base = load_frozen_base(args.weights, device)

    # cache frozen VGG features (base trunk never changes)
    base.eval()
    with torch.no_grad():
        for s in samples:
            s["feat"] = base.vgg(s["tensor"])   # (1,128,50,50)

    # ── sanity (학습 전) ──
    sanity = run_sanity(samples, OUT_DIR)
    print("\n[sanity] frame", sanity["fid"])
    print(f"  mask area decoded={sanity['mask_area_decoded']} "
          f"json={sanity['mask_area_json']} match={sanity['area_match']}")
    print(f"  frac mask in cuboid bbox = {sanity['frac_mask_in_cuboid_bbox']}")
    print(f"  vec unit max|cos-1| = {sanity['vec_unit_max_cos_err']} "
          f"(0 이면 방향 정확)")
    print(f"  vec offset max recover err (cell) = "
          f"{sanity['vec_offset_max_recover_err_cell']} (0 이면 px+offset==kp)")
    print(f"  overlay: {sanity['overlay']}")
    json.dump(sanity, open(os.path.join(OUT_DIR, "sanity.json"), "w"), indent=2)

    # ── 학습: vec/seg 헤드만 ──
    head_params = [p for n, p in base.named_parameters()
                   if (n.startswith("m_vec") or n.startswith("m_seg"))
                   and p.requires_grad]
    opt = torch.optim.Adam(head_params, lr=args.lr)

    log = []
    for it in range(1, args.iters + 1):
        base.m_vec1.train(); base.m_vec2.train()
        base.m_seg1.train(); base.m_seg2.train()
        opt.zero_grad(set_to_none=True)
        tot = tot_v = tot_s = 0.0
        for s in samples:
            feat = s["feat"]
            # vec head (2-stage)
            vec1 = base.m_vec1(feat)
            vec2 = base.m_vec2(torch.cat([vec1, feat], 1))
            # seg head (2-stage)
            seg1 = base.m_seg1(feat)
            seg2 = base.m_seg2(torch.cat([seg1, feat], 1))

            L_vec = vec_loss([vec1, vec2], s["vec_gt"], s["mask_gt"])
            tgt = (s["mask_gt"] > 0.5).float()
            L_seg = (F.binary_cross_entropy_with_logits(seg1, tgt)
                     + F.binary_cross_entropy_with_logits(seg2, tgt))
            loss = args.lam_vec * L_vec + args.lam_seg * L_seg
            if not torch.isfinite(loss):
                print(f"[FATAL] non-finite loss at it={it} "
                      f"L_vec={float(L_vec)} L_seg={float(L_seg)} — abort")
                sys.exit(2)
            loss.backward()
            tot += float(loss); tot_v += float(L_vec); tot_s += float(L_seg)
        opt.step()
        if it % 250 == 0 or it == 1:
            n = len(samples)
            print(f"[it {it}] loss={tot/n:.5f} L_vec={tot_v/n:.5f} "
                  f"L_seg={tot_s/n:.5f}")
            log.append({"iter": it, "loss": tot / n,
                        "L_vec": tot_v / n, "L_seg": tot_s / n})

    # ── eval: IoU + voting corner err (GT-mask & pred-mask) ──
    base.eval()
    ious = []
    err_gtmask = []     # per-corner err (cell) GT-mask voting
    err_predmask = []
    per_frame = []
    with torch.no_grad():
        for s in samples:
            feat = s["feat"]
            vec1 = base.m_vec1(feat)
            vec2 = base.m_vec2(torch.cat([vec1, feat], 1))
            seg1 = base.m_seg1(feat)
            seg2 = base.m_seg2(torch.cat([seg1, feat], 1))
            vec = vec2[0].cpu().numpy()             # (18,50,50)
            seg_logit = seg2[0].cpu().numpy()       # (1,50,50)

            pred_mask = seg_pred_mask(seg_logit, GRID, thr=0.5)
            gt_mask = s["mask50"]
            iou_v = iou(pred_mask, gt_mask)
            ious.append(iou_v)

            gt8 = s["gt8_grid"]
            # GT-mask voting
            kp_gt = vote_unit_ransac(vec, gt_mask, n_hyp=args.ransac_hyp,
                                     inlier_thr_deg=args.inlier_deg,
                                     seed=args.seed)
            e_gt = np.linalg.norm(kp_gt[:8] - gt8, axis=1)
            err_gtmask.extend(e_gt[np.isfinite(e_gt)].tolist())
            # pred-mask voting
            if pred_mask.sum() >= 8:
                kp_pr = vote_unit_ransac(vec, pred_mask, n_hyp=args.ransac_hyp,
                                         inlier_thr_deg=args.inlier_deg,
                                         seed=args.seed)
                e_pr = np.linalg.norm(kp_pr[:8] - gt8, axis=1)
                e_pr_f = e_pr[np.isfinite(e_pr)]
            else:
                e_pr_f = np.array([])
            err_predmask.extend(e_pr_f.tolist())
            per_frame.append({
                "fid": s["fid"], "occ": s["occ"], "iou": round(iou_v, 4),
                "gtmask_med_cell": round(float(np.median(e_gt)), 3),
                "predmask_med_cell": (round(float(np.median(e_pr_f)), 3)
                                      if len(e_pr_f) else None),
            })

    iou_med = float(np.median(ious))
    gtm_med = float(np.median(err_gtmask)) if err_gtmask else float("inf")
    gtm_p95 = (float(np.percentile(err_gtmask, 95))
               if err_gtmask else float("inf"))
    prm_med = float(np.median(err_predmask)) if err_predmask else float("inf")
    pass_iou = iou_med > 0.9
    pass_vote = gtm_med < 1.0
    passed = pass_iou and pass_vote

    lines = [
        "# STAGE10 M2 v3 real-mask overfit32 — PVNet unit-vec + seg",
        f"samples={len(samples)} iters={args.iters} lr={args.lr} "
        f"lam_vec={args.lam_vec} lam_seg={args.lam_seg}",
        f"seg IoU median={iou_med:.4f}  (pass: >0.9 -> {pass_iou})",
        f"GT-mask voting corner err (cell): median={gtm_med:.4f} "
        f"p95={gtm_p95:.4f}  (pass: median<1.0 -> {pass_vote})",
        f"pred-mask voting corner err (cell): median={prm_med:.4f} (참고)",
        f"GATE: IoU>0.9 AND GTmask-vote-med<1.0  =>  "
        f"{'PASS' if passed else 'FAIL'}",
        "",
        f"{'frame':<10}{'occ':>5}{'iou':>8}{'gtm_med':>10}{'prm_med':>10}",
        "-" * 43,
    ]
    for pf in sorted(per_frame, key=lambda x: -x["gtmask_med_cell"]):
        prm = "-" if pf["predmask_med_cell"] is None \
            else f"{pf['predmask_med_cell']:.3f}"
        lines.append(f"{pf['fid']:<10}{pf['occ']:>5}{pf['iou']:>8.3f}"
                     f"{pf['gtmask_med_cell']:>10.3f}{prm:>10}")
    table = "\n".join(lines)
    print("\n" + table)

    with open(os.path.join(OUT_DIR, "overfit32.txt"), "w") as f:
        f.write(table + "\n")
    json.dump({"n_samples": len(samples), "iters": args.iters, "lr": args.lr,
               "iou_median": iou_med,
               "gtmask_vote_median_cell": gtm_med,
               "gtmask_vote_p95_cell": gtm_p95,
               "predmask_vote_median_cell": prm_med,
               "pass_iou": bool(pass_iou), "pass_vote": bool(pass_vote),
               "passed": bool(passed), "loss_log": log,
               "per_frame": per_frame, "sanity": sanity},
              open(os.path.join(OUT_DIR, "overfit32.json"), "w"), indent=2)
    print(f"\n[save] {OUT_DIR}/overfit32.json  passed={passed}")


if __name__ == "__main__":
    main()
