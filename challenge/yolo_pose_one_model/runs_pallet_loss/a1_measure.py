"""A1 진단 측정 — 임의 checkpoint 를 v1 val 133 에서 잰다.

A0 와 A1 을 **같은 식**으로 잰다.  min() 은 loss 를 구조적으로 낮추므로 학습 loss
값 비교는 무의미하다 — 그래서 두 모델 모두 여기서 다시 잰다.

★ 진단셋은 학습과 **같은 이미지 도메인**이어야 한다.  broad40k 원본(920x680)은
handoff(720x480)와 다른 렌더라, 거기서 재면 대칭이 아니라 도메인 갭을 잰다.
"""
import argparse, json, os, sys, collections
import numpy as np, torch

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, ROOT)
R = f"{ROOT}/challenge/yolo_pose_one_model/runs_pallet_loss"
VAL = f"{ROOT}/challenge/yolo_pose_one_model/datasets/v1_fixed_matched10k/data.yaml"

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--data", default=VAL)
A = ap.parse_args()

from ultralytics import YOLO
from ultralytics.utils.loss import E2ELoss
from ultralytics.utils.ops import xyxy2xywh
from ultralytics.data.build import build_yolo_dataset, build_dataloader
from ultralytics.data.utils import check_det_dataset
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG
from pallet_yolo_loss.symmetry import A1SymmetryPoseLoss, per_instance_kpt_loss

DEV = "cuda"
P180 = tuple(json.load(open(f"{R}/FIXED_P180_PERMUTATIONS.json"))["P180"])
perm = list(P180) + [8]
smap = json.load(open(f"{R}/STEM_ASSET_MAP.json"))
ROWS = []

_orig = A1SymmetryPoseLoss.calculate_keypoints_loss


def probe(self, masks, tgi, kpts, bidx, st, tb, pk):
    if masks.any() and getattr(self, "_probe", False):
        sel = self._select_target_keypoints(kpts, bidx, tgi, masks).clone()
        sel[..., :2] /= st.view(1, -1, 1, 1)
        gt = sel[masks]
        pr = pk[masks]
        area = xyxy2xywh((tb / st)[masks])[:, 2:].prod(1, keepdim=True)
        m = gt[..., 2] != 0
        did = per_instance_kpt_loss(self.keypoint_loss, pr, gt, m, area)
        d180 = per_instance_kpt_loss(self.keypoint_loss, pr, gt[:, perm, :], m[:, perm], area)
        # 입력 640 픽셀 공간의 corner 오차
        sd = st.view(1, -1, 1).expand(masks.shape[0], -1, 1)[masks]
        def px(g):
            mv = (g[..., 2] != 0).float()
            e = ((pr[..., :2] - g[..., :2]).pow(2).sum(-1).sqrt()) * sd
            return (e * mv).sum(1) / mv.sum(1).clamp_min(1)
        pid = px(gt)
        p180 = px(gt[:, perm, :])
        # per-corner 오차 (gross20 / bottom-corner 게이트용)
        mvv = (gt[..., 2] != 0).float()
        pc_err = ((pr[..., :2] - gt[..., :2]).pow(2).sum(-1).sqrt()) * sd
        pc_err = (pc_err * mvv + (-1.0) * (1 - mvv))
        img = torch.arange(masks.shape[0], device=masks.device)[:, None].expand_as(masks)[masks]
        files = (self._batch or {}).get("im_file") or []
        for n, i in enumerate(img.tolist()):
            s = os.path.splitext(os.path.basename(files[i]))[0] if i < len(files) else "?"
            ROWS.append((s, float(did[n]), float(d180[n]), float(pid[n]), float(p180[n]),
                         pc_err[n].detach().cpu().numpy().tolist()))
    return _orig(self, masks, tgi, kpts, bidx, st, tb, pk)


A1SymmetryPoseLoss.calculate_keypoints_loss = probe

y = YOLO(A.weights, task="pose")
model = y.model.to(DEV).float().train()
for q_ in model.parameters():
    q_.requires_grad_(False)
data = check_det_dataset(A.data)
args = get_cfg(DEFAULT_CFG, overrides=dict(
    task="pose", mode="val", data=A.data, imgsz=640, batch=16, workers=4,
    device=0, seed=0, single_cls=True, rect=False))
model.args = args
model.nc = data["nc"]
model.names = data["names"]
ds = build_yolo_dataset(args, data["val"], 16, data, mode="val", rect=False, stride=32)
dl = build_dataloader(ds, 16, 4, shuffle=False, rank=-1)
os.environ["A1_CONFIG"] = ""
os.environ["PSPC_CONFIG"] = ""
crit = E2ELoss(model, A1SymmetryPoseLoss) if getattr(model, "end2end", False) \
    else A1SymmetryPoseLoss(model)
getattr(crit, "one2many", crit)._probe = True
for b in dl:
    b["img"] = b["img"].to(DEV).float() / 255
    for k in ("keypoints", "bboxes", "cls", "batch_idx"):
        b[k] = b[k].to(DEV)
    with torch.no_grad():
        crit(model(b["img"]), b)

per = collections.defaultdict(list)
PCE = collections.defaultdict(list)
for s, x, z, pi, pz, ce in ROWS:
    per[s].append((x, z, pi, pz))
    PCE[s].append(ce)
rec = []
for s in sorted(per):
    v = np.array(per[s])
    rec.append({"stem": s, "asset": smap.get(s, "?"), "n_anchor": len(v),
                "d_id": float(v[:, 0].mean()), "d_180": float(v[:, 1].mean()),
                "e_sym": float(np.minimum(v[:, 0], v[:, 1]).mean()),
                "px_id": float(v[:, 2].mean()), "px_180": float(v[:, 3].mean()),
                "px_best": float(np.minimum(v[:, 2], v[:, 3]).mean()),
                "flipped": bool((v[:, 1] < v[:, 0]).mean() > 0.5),
                "corner_err": np.array(PCE[s]).mean(0).tolist()})


def qt(a, p):
    return float(np.percentile(np.array(a), p))


pid_l = [r["px_id"] for r in rec]
pbe_l = [r["px_best"] for r in rec]
out = {"tag": A.tag, "weights": A.weights, "data": A.data, "n_frames": len(rec),
       "identity_d_id_median": float(np.median([r["d_id"] for r in rec])),
       "yaw180best_e_sym_median": float(np.median([r["e_sym"] for r in rec])),
       "corner_px_identity_median": qt(pid_l, 50), "corner_px_identity_p90": qt(pid_l, 90),
       "corner_px_best_median": qt(pbe_l, 50), "corner_px_best_p90": qt(pbe_l, 90),
       "flip_rate": float(np.mean([r["flipped"] for r in rec])),
       "per_frame": rec}
_ce = np.array([r["corner_err"] for r in rec])          # (F, 9)
_v = _ce[:, :8] >= 0
_e = _ce[:, :8][_v]
out["gross20"] = float((_e > 20).mean())
out["gross40"] = float((_e > 40).mean())
_bot = _ce[:, [2, 3, 6, 7]]
_bv = _bot >= 0
out["bottom_p90"] = float(np.percentile(_bot[_bv], 90)) if _bv.any() else None
out["bottom_median"] = float(np.median(_bot[_bv])) if _bv.any() else None
try:
    mv = YOLO(A.weights, task="pose").val(data=A.data, imgsz=640, batch=16, device=0,
                                          verbose=False, plots=False, save_json=False)
    out["pose_map50_95"] = float(mv.pose.map)
    out["pose_map50"] = float(mv.pose.map50)
    out["box_map50_95"] = float(mv.box.map)
except Exception as e:
    out["pose_map50_95"] = None
    out["val_error"] = str(e)
json.dump(out, open(f"{R}/A1_MEASURE_{A.tag}.json", "w"))
print(f"{A.tag:18} n{len(rec):4d}  ident {out['identity_d_id_median']:.5f}  "
      f"yaw180best {out['yaw180best_e_sym_median']:.5f}  "
      f"px_id med/p90 {out['corner_px_identity_median']:.2f}/{out['corner_px_identity_p90']:.2f}  "
      f"flip {100*out['flip_rate']:.2f}%  mAP {out['pose_map50_95']}")
