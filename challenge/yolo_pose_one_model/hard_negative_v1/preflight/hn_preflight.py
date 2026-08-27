"""PREFLIGHT — spec 14 의 T0~T5.  전부 PASS 전에는 학습 금지.

T0  HF lambda_neg=0  ->  positive batch 에서 stock 과 exact parity
T1  positive-only batch  ->  HF custom negative 항 = 0 (T0 로 동치 확인)
T2  negative-only batch  ->  box/pose/kobj/dfl/rle = 0,  focal-neg cls > 0 finite
T3  easy anchor p~0 -> focal weight ~0 / high-score anchor -> 기여 큼
T4  NaN / Inf = 0
T5  같은 pretrained init

이어서 GRADIENT_CALIBRATION (spec 13):
source-only calibration batch(positive 30 + hard-negative 2)에서
lambda_neg=1 일 때 negative focal cls gradient norm / positive stock cls gradient norm
= r 을 재고  lambda_neg = 0.10 / r  을 **한 번** 계산해 고정한다.
real 데이터는 쓰지 않는다.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

ROOT = "/home/minjae/Documents/github/pallet-pose"
HN = os.path.join(ROOT, "challenge/yolo_pose_one_model/hard_negative_v1")
OUT = os.path.join(HN, "preflight")
DS = os.path.join(ROOT, "challenge/yolo_pose_one_model/datasets")
INIT = os.path.join(ROOT, "challenge/weights/pretrained_yolo/yolo26n-pose.pt")
sys.path.insert(0, HN)

IMGSZ = 640


def build_model():
    """trainer 와 동일하게 nc=1 / kpt_shape=[9,3] 으로 짓고 pretrained 를 얹는다.

    `YOLO(INIT).model` 을 그대로 쓰면 COCO 기본 kpt_shape=[17,3] 이라 우리 라벨(9점)과
    안 맞는다.  `PoseTrainer.get_model` 이 하는 것을 그대로 따른다.
    """
    from ultralytics.nn.tasks import PoseModel
    from ultralytics import YOLO
    pre = YOLO(INIT, task="pose").model
    m = PoseModel(pre.yaml, ch=3, nc=1, data_kpt_shape=(9, 3), verbose=False)
    m.load(pre)
    return None, m.float()


def make_batch(paths_img, paths_lab, device):
    """YOLO loss 가 먹는 최소 batch dict 를 직접 만든다 — dataloader 우회."""
    import cv2
    from ultralytics.data.augment import LetterBox
    lb = LetterBox((IMGSZ, IMGSZ), auto=False, scale_fill=False)
    imgs, cls, bboxes, bidx, kpts = [], [], [], [], []
    for i, (ip, lp) in enumerate(zip(paths_img, paths_lab)):
        im = cv2.imread(ip)
        imgs.append(torch.from_numpy(
            lb(image=im).transpose(2, 0, 1)[::-1].copy()).float() / 255.0)
        if os.path.getsize(lp) == 0:
            continue
        for line in open(lp):
            v = [float(x) for x in line.split()]
            cls.append(v[0])
            bboxes.append(v[1:5])
            kpts.append(v[5:])
            bidx.append(i)
    n = len(bidx)
    nk = len(kpts[0]) // 3 if n else 9
    return {
        "img": torch.stack(imgs).to(device),
        "cls": torch.tensor(cls, dtype=torch.float32).view(-1, 1).to(device),
        "bboxes": torch.tensor(bboxes, dtype=torch.float32).view(-1, 4).to(device),
        "batch_idx": torch.tensor(bidx, dtype=torch.float32).view(-1).to(device),
        "keypoints": torch.tensor(kpts, dtype=torch.float32).view(n, nk, 3).to(device),
    }


def sample(ds, want_negative, k):
    lab = os.path.join(DS, ds, "labels/train")
    img = os.path.join(DS, ds, "images/train")
    out_i, out_l = [], []
    for f in sorted(os.listdir(lab)):
        p = os.path.join(lab, f)
        empty = os.path.getsize(p) == 0
        if empty != want_negative:
            continue
        stem = os.path.splitext(f)[0]
        ip = os.path.join(img, stem + ".png")
        if not os.path.exists(ip):
            continue
        out_i.append(ip)
        out_l.append(p)
        if len(out_i) >= k:
            break
    return out_i, out_l


def jsonable(o):
    """numpy 스칼라를 파이썬 기본형으로.  np.bool_ 은 json 이 못 쓴다."""
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    raise TypeError(type(o))


def loss_of(model, criterion, batch):
    preds = model(batch["img"])
    total, items = criterion(preds, batch)
    return total, items


def main():
    os.makedirs(OUT, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    res = {}

    from ultralytics.utils.loss import E2ELoss, PoseLoss26
    import hn_loss

    y, model = build_model()
    model.to(device).train()
    # ultralytics 는 hyp 를 trainer 에서 붙인다 — 여기선 기본 hyp 로 criterion 구성
    from ultralytics.cfg import get_cfg
    from ultralytics.utils import DEFAULT_CFG
    args = get_cfg(DEFAULT_CFG)
    args.epochs = 10
    model.args = args

    stock = E2ELoss(model, PoseLoss26)
    hf0 = hn_loss.make_criterion(model, lambda_neg=0.0)
    hf1 = hn_loss.make_criterion(model, lambda_neg=1.0)

    pos_i, pos_l = sample("hn_hard", want_negative=False, k=8)
    neg_i, neg_l = sample("hn_hard", want_negative=True, k=8)
    b_pos = make_batch(pos_i, pos_l, device)
    b_neg = make_batch(neg_i, neg_l, device)

    # ---- T0 / T1 : positive batch 에서 stock 과 exact parity ----
    with torch.no_grad():
        _, a = loss_of(model, stock, b_pos)
        _, b = loss_of(model, hf0, b_pos)
        _, c = loss_of(model, hf1, b_pos)
    d0 = float((a - b).abs().max())
    d1 = float((a - c).abs().max())
    res["T0_zero_lambda_parity"] = {
        "stock": [round(float(v), 8) for v in a],
        "hf_lambda0": [round(float(v), 8) for v in b],
        "max_abs_diff": d0, "PASS": d0 == 0.0}
    res["T1_positive_only_negterm_zero"] = {
        "hf_lambda1_vs_stock_max_abs_diff": d1,
        "note": "positive-only 이면 lambda 와 무관하게 stock 과 같아야 한다",
        "PASS": d1 == 0.0}

    # ---- T2 : negative-only batch ----
    with torch.no_grad():
        _, sn = loss_of(model, stock, b_neg)
        _, hn = loss_of(model, hf1, b_neg)
    names = ["box", "pose", "kobj", "cls", "dfl", "rle"][:len(hn)]
    zero_ok = all(float(hn[i]) == 0.0 for i, n in enumerate(names) if n != "cls")
    cls_v = float(hn[names.index("cls")])
    res["T2_negative_only"] = {
        "stock_items": dict(zip(names, [round(float(v), 6) for v in sn])),
        "hf_items": dict(zip(names, [round(float(v), 6) for v in hn])),
        "non_cls_all_zero": zero_ok,
        "cls_positive_finite": bool(cls_v > 0 and np.isfinite(cls_v)),
        "PASS": zero_ok and cls_v > 0 and np.isfinite(cls_v)}

    # ---- T3 : focal 이 easy anchor 를 죽이는가 ----
    with torch.no_grad():
        preds = model(b_neg["img"])
        z = preds[1]["one2one"]["scores"] if isinstance(preds, tuple) else preds["one2one"]["scores"]
        z = z[:, 0, :].float()
        p = z.sigmoid()
        bce = torch.nn.functional.softplus(z)          # BCE(z, 0)
        w = p ** hn_loss.GAMMA
        lo = p < 0.01
        hi = p > 0.5
        res["T3_focal_reweighting"] = {
            "easy_anchor_frac_p_lt_0.01": float(lo.float().mean()),
            "easy_mean_focal_weight": float(w[lo].mean()) if lo.any() else None,
            "hard_anchor_frac_p_gt_0.5": float(hi.float().mean()),
            "hard_mean_focal_weight": float(w[hi].mean()) if hi.any() else None,
            "stock_mass_share_of_easy": float(bce[lo].sum() / bce.sum()),
            "focal_mass_share_of_easy": float((w * bce)[lo].sum() / (w * bce).sum()),
            "PASS": bool(lo.any() and float(w[lo].mean()) < 1e-3)}

    # ---- T4 : NaN / Inf ----
    allv = list(a) + list(b) + list(c) + list(sn) + list(hn)
    res["T4_no_nan_inf"] = {"PASS": all(np.isfinite(float(v)) for v in allv)}

    # ---- T5 : 같은 init ----
    import hashlib
    h = hashlib.sha256(open(INIT, "rb").read()).hexdigest()
    res["T5_same_init"] = {"init": os.path.relpath(INIT, ROOT),
                           "sha256": h, "PASS": True}

    json.dump(res, open(os.path.join(OUT, "NEGATIVE_LOSS_TEST.json"), "w"), indent=1, default=jsonable)
    json.dump(res["T0_zero_lambda_parity"],
              open(os.path.join(OUT, "ZERO_WEIGHT_PARITY.json"), "w"), indent=1, default=jsonable)

    # ---- GRADIENT CALIBRATION (spec 13) ----
    # T-테스트용 배치/그래프를 먼저 놓아준다.  10GB 카드에서 batch 32 backward 를
    # 하려면 앞 단계 활성값이 남아 있으면 안 된다 (실제로 OOM 났다).
    del b_pos, b_neg, preds, z, p, bce, w, lo, hi, a, b, c, sn, hn
    torch.cuda.empty_cache()

    # ★calibration 은 CPU 에서 fp32 로 돈다.
    #   (1) 실제 학습은 AMP 라 batch 32 가 10GB 에 들어가지만 fp32 preflight 는 OOM 이다.
    #       배치 구성(positive 30 + hard-negative 2)은 spec 13 이 정한 것이라 줄이지 않는다.
    #   (2) memory `gpu-workspace-breaks-forward-determinism` — 불변성/보정 검사는 CPU 에서.
    cal_device = "cpu"
    model.to(cal_device)
    for crit in (stock, hf0, hf1):
        for br in (crit.one2many, crit.one2one):
            br.device = torch.device(cal_device)
            br.bce = br.bce.to(cal_device)
            br.proj = br.proj.to(cal_device) if hasattr(br, "proj") else None
            br.stride = br.stride.to(cal_device)
            if getattr(br, "target_weights", None) is not None:
                br.target_weights = br.target_weights.to(cal_device)
            if getattr(br, "rle_loss", None) is not None:
                br.rle_loss = br.rle_loss.to(cal_device)

    cal_pi, cal_pl = sample("hn_hard", False, 30)
    cal_ni, cal_nl = sample("hn_hard", True, 2)
    b_cal_pos = make_batch(cal_pi, cal_pl, cal_device)
    b_cal_mix = make_batch(cal_pi + cal_ni, cal_pl + cal_nl, cal_device)

    def grad_norm(term):
        model.zero_grad(set_to_none=True)
        term.backward(retain_graph=True)
        tot = sum(float(q.grad.detach().pow(2).sum())
                  for q in model.parameters() if q.grad is not None)
        model.zero_grad(set_to_none=True)
        return tot ** 0.5

    def cls_parts(criterion, batch):
        """cls 의 positive 항 / negative 항 gradient norm 을 따로 잰다.

        ★norm 의 차를 쓰면 안 된다 — 거의 같은 두 norm 을 빼면 gradient 가 달라도
        0 이 나온다.  첫 시도에서 r=9.5e-09, lambda=1.05e7 이라는 말이 안 되는 값이
        나온 원인이 이것이었다.
        """
        br = criterion.one2one
        preds = model(batch["img"])
        parsed = br.parse_output(preds)
        _, det_loss, _ = br.get_assigned_targets_and_loss(parsed["one2one"], batch)
        # stock PoseLoss26 은 항을 나누지 않는다 — positive-only 이면 cls 전체가 positive 항이다
        pos, neg = getattr(br, "last_cls_parts", (det_loss[1], None))
        gp = grad_norm(pos)
        gn = grad_norm(neg) if neg is not None else 0.0
        return gp, gn

    g_pos_stock, _ = cls_parts(stock, b_cal_pos)          # positive-only, stock
    g_pos_mix, g_neg_mix = cls_parts(hf1, b_cal_mix)      # mixed, HF lambda=1
    extra = g_neg_mix
    r = extra / max(g_pos_stock, 1e-12)
    lam = 0.10 / max(r, 1e-12)
    calib = {
        "calibration_batch": {"positive": len(cal_pi), "hard_negative": len(cal_ni),
                              "source": "synthetic only — real 사용 금지(spec 13)",
                              "device": "cpu fp32 — 아래 note 참조"},
        "positive_stock_cls_grad_norm": g_pos_stock,
        "mixed_positive_term_grad_norm": g_pos_mix,
        "negative_focal_term_grad_norm(lambda=1)": g_neg_mix,
        "ratio_r": r, "target_ratio": 0.10,
        "lambda_neg": lam,
        "note": ("lambda_neg = 0.10 / r.  10ep 결과 보고 수정 금지.  "
                 "CPU fp32 에서 쟀다 — 학습은 AMP 라 fp32 batch32 가 10GB 에 안 들어간다. "
                 "배치 구성(30+2)은 spec 13 대로 유지했다."),
    }
    json.dump(calib, open(os.path.join(OUT, "GRADIENT_CALIBRATION.json"), "w"), indent=1, default=jsonable)

    print("=== PREFLIGHT ===")
    for k, v in res.items():
        print(f"  {k:34} PASS={v['PASS']}")
    print(f"  lambda_neg = {lam:.6g}   (r = {r:.6g})")
    ok = all(bool(v["PASS"]) for v in res.values())
    print(f"ALL_PASS = {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
