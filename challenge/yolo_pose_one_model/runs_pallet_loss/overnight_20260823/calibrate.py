"""TAKL / NRL 사전 캘리브레이션 — train-only, 후보 결과를 보기 전에 고정한다.

tau        = A0 train residual q75
z_clip     = train q95 of z  (상위 5% 가 L_tail 의 80% 초과를 독점할 때만 사용)
lambda     = 0.10 * median(g_base) / median(g_tail)      (TAKL)
             median(g_base) / median(g_nrl)              (NRL)

gradient norm 은 고정된 8 batch 에서 잰다.  결과를 보고 바꾸지 않는다.
"""
import json, os, sys
import numpy as np, torch

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, ROOT)
R = f"{ROOT}/challenge/yolo_pose_one_model/runs_pallet_loss"
OUT = f"{R}/overnight_20260823"
DATA = f"{ROOT}/challenge/yolo_pose_one_model/datasets/v1_fixed_matched10k/data.yaml"
A0 = f"{ROOT}/challenge/yolo_pose_one_model/runs_fixed/V1_FIXED_MATCHED10K_60EP_SEED42_UBUNTU/weights/last.pt"
N_CAL_IMG = 1024          # train-only 고정 부분집합
N_GRAD_BATCH = 8

from ultralytics import YOLO
from ultralytics.utils.loss import E2ELoss
from ultralytics.utils.ops import xyxy2xywh
from ultralytics.data.build import build_yolo_dataset, build_dataloader
from ultralytics.data.utils import check_det_dataset
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG
import pallet_yolo_loss.symmetry as SY
from pallet_yolo_loss.symmetry import (A1SymmetryPoseLoss, normalized_residual,
                                       takl_tail, nrl_coord)

DEV = "cuda"
RES, GRADS = [], []
_orig = A1SymmetryPoseLoss.calculate_keypoints_loss


def probe(self, masks, tgi, kpts, bidx, st, tb, pk):
    if masks.any() and getattr(self, "_probe", False):
        sel = self._select_target_keypoints(kpts, bidx, tgi, masks)
        sel[..., :2] /= st.view(1, -1, 1, 1)
        tbb = tb / st
        gt = sel[masks]
        pr = pk[masks]
        area = xyxy2xywh(tbb[masks])[:, 2:].prod(1, keepdim=True)
        m = gt[..., 2] != 0
        with torch.no_grad():
            r = normalized_residual(pr, gt, area)
            RES.append(r[m].detach().float().cpu().numpy())
        self._cache = (pr, gt, m, area)
    return _orig(self, masks, tgi, kpts, bidx, st, tb, pk)


A1SymmetryPoseLoss.calculate_keypoints_loss = probe

y = YOLO(A0, task="pose")
model = y.model.to(DEV).float().train()
data = check_det_dataset(DATA)
args = get_cfg(DEFAULT_CFG, overrides=dict(
    task="pose", mode="train", data=DATA, imgsz=640, batch=16, epochs=1, workers=4,
    device=0, seed=0, single_cls=True, mosaic=0.0, scale=0.0, hsv_s=0.0, hsv_v=0.0,
    fliplr=0.0, flipud=0.0, erasing=0.0))
model.args = args
model.nc = data["nc"]
model.names = data["names"]
ds = build_yolo_dataset(args, data["train"], 16, data, mode="train", rect=False, stride=32)
dl = build_dataloader(ds, 16, 4, shuffle=False, rank=-1)

os.environ["A1_CONFIG"] = ""
os.environ["PSPC_CONFIG"] = ""
crit = E2ELoss(model, A1SymmetryPoseLoss) if getattr(model, "end2end", False) \
    else A1SymmetryPoseLoss(model)
inner = getattr(crit, "one2many", crit)
inner._probe = True

# ---- 1) residual 분포 (grad 불필요) ---------------------------------------
for p in model.parameters():
    p.requires_grad_(False)
seen = 0
batches = []
for b in dl:
    if seen >= N_CAL_IMG:
        break
    b["img"] = b["img"].to(DEV).float() / 255
    for k in ("keypoints", "bboxes", "cls", "batch_idx"):
        b[k] = b[k].to(DEV)
    if len(batches) < N_GRAD_BATCH:
        batches.append(b)
    with torch.no_grad():
        crit(model(b["img"]), b)
    seen += b["img"].shape[0]

r = np.concatenate(RES)
tau = float(np.percentile(r, 75))
z = np.maximum(r - tau, 0.0)
zpos = z[z > 0]
q95 = float(np.percentile(z, 95)) if len(zpos) else 0.0
# top 5% 가 L_tail 을 독점하는가
sl = np.where(z < tau, 0.5 * z ** 2 / max(tau, 1e-9), z - 0.5 * tau)
thr = np.percentile(z, 95)
share = float(sl[z >= thr].sum() / max(sl.sum(), 1e-12))
use_clip = share > 0.80
diag = {"n_visible_kp": int(r.size), "n_images": seen,
        "residual_q": {f"q{q}": float(np.percentile(r, q)) for q in (10, 25, 50, 75, 90, 95, 99)},
        "tau_q75": tau, "z_q95": q95,
        "top5pct_share_of_L_tail": share,
        "winsorize": use_clip,
        "winsorize_rule": "top 5% 가 L_tail 의 80% 초과를 차지하면 z 를 train q95 로 clip",
        "nrl_beta_q75_of_abs_axis": None}

# NRL beta = coordinate absolute residual q75 (축별)
AX = []
for b in batches:
    with torch.no_grad():
        crit(model(b["img"]), b)
    pr, gt, m, area = inner._cache
    sx = area.clamp_min(1e-9).sqrt()
    dx = ((pr[..., 0] - gt[..., 0]) / sx)[m].abs()
    dy = ((pr[..., 1] - gt[..., 1]) / sx)[m].abs()
    AX.append(torch.cat([dx, dy]).float().cpu().numpy())
ax = np.concatenate(AX)
nrl_beta = float(np.percentile(ax, 75))
diag["nrl_beta_q75_of_abs_axis"] = nrl_beta

# ---- 2) gradient norm 매칭 -------------------------------------------------
for p in model.parameters():
    p.requires_grad_(True)
head = [p for p in model.parameters() if p.requires_grad]


def gnorm(fn):
    out = []
    for b in batches:
        model.zero_grad(set_to_none=True)
        crit(model(b["img"]), b)
        pr, gt, m, area = inner._cache
        v = fn(pr, gt, m, area)
        g = torch.autograd.grad(v, head, retain_graph=False, allow_unused=True)
        out.append(float(torch.sqrt(sum((x.pow(2).sum() for x in g if x is not None)))))
    return out


g_base = gnorm(lambda pr, gt, m, area: inner.keypoint_loss(pr, gt, m, area))
g_tail = gnorm(lambda pr, gt, m, area: takl_tail(pr, gt, m, area, tau,
                                                 q95 if use_clip else 0.0))
g_nrl = gnorm(lambda pr, gt, m, area: nrl_coord(pr, gt, m, area, nrl_beta))
mb, mt, mn = np.median(g_base), np.median(g_tail), np.median(g_nrl)
takl_lambda = float(0.10 * mb / (mt + 1e-12))
nrl_lambda = float(mb / (mn + 1e-12))
diag["grad_norms"] = {"median_base": float(mb), "median_tail": float(mt),
                      "median_nrl": float(mn), "n_batches": len(batches)}
diag["takl_lambda"] = takl_lambda
diag["nrl_lambda"] = nrl_lambda
diag["frozen_before_seeing_candidate_results"] = True
json.dump(diag, open(f"{OUT}/TRAIN_DIAGNOSTIC.json", "w"), indent=2)

base = dict(enabled=True, mode="exact_min", lambda_role=0.0, margin=0.0,
            p180=[5, 4, 7, 6, 1, 0, 3, 2], centroid_index=8,
            sym_assets=[], asym_assets=[],
            stem_asset_map=f"{R}/STEM_ASSET_MAP.json", role_ramp=[5, 20])
json.dump({**base, "takl_enabled": True, "takl_tau": tau,
           "takl_lambda": takl_lambda, "takl_z_clip": q95 if use_clip else 0.0},
          open(f"{OUT}/TAKL_LOSS_CONFIG.json", "w"), indent=2)
json.dump({**base, "nrl_enabled": True, "nrl_beta": nrl_beta, "nrl_lambda": nrl_lambda},
          open(f"{OUT}/NRL_LOSS_CONFIG.json", "w"), indent=2)
asc = json.load(open(f"{R}/ASC_LOSS_CONFIG.json"))
json.dump({**asc, "takl_enabled": True, "takl_tau": tau,
           "takl_lambda": takl_lambda, "takl_z_clip": q95 if use_clip else 0.0},
          open(f"{OUT}/ASC_TAKL_LOSS_CONFIG.json", "w"), indent=2)

print(f"  visible kp {r.size}  이미지 {seen}")
print(f"  residual  q50 {np.percentile(r,50):.4f}  q75(tau) {tau:.4f}  q95 {np.percentile(r,95):.4f}")
print(f"  top5% 가 L_tail 의 {100*share:.1f}% 차지  -> winsorize {use_clip} (z_clip {q95:.4f})")
print(f"  grad median  base {mb:.4f}  tail {mt:.4f}  nrl {mn:.4f}")
print(f"  lambda_tail {takl_lambda:.5f}   lambda_nrl {nrl_lambda:.5f}   nrl_beta {nrl_beta:.4f}")
