"""A1 사전 캘리브레이션 — asset 별 d_id / d_180 분포.

forward 를 재구현하지 않는다.  **실제 loss 경로**(E2ELoss -> A1SymmetryPoseLoss)
안에서 d_id/d_180 을 그대로 받아 적는다.  contract 가 어느 asset 을 SYM 이라 부르든
이 통계는 바뀌지 않는다.  산출은 분포뿐 — 판정하지 않는다.
"""
import json, os, sys, collections
import numpy as np, torch

ROOT="/home/minjae/Documents/github/pallet-pose"; sys.path.insert(0, ROOT)
R=f"{ROOT}/challenge/yolo_pose_one_model/runs_pallet_loss"
DATA=f"{ROOT}/challenge/yolo_pose_one_model/datasets/v1_fixed_matched10k/data.yaml"
INIT=f"{ROOT}/challenge/yolo_pose_one_model/runs_fixed/V1_FIXED_MATCHED10K_60EP_SEED42_UBUNTU/weights/last.pt"
N_IMG=int(os.environ.get("N_IMG","2048"))

from ultralytics import YOLO
from ultralytics.utils.loss import E2ELoss
from ultralytics.utils.ops import xyxy2xywh
from ultralytics.data.build import build_yolo_dataset, build_dataloader
from ultralytics.data.utils import check_det_dataset
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG
from pallet_yolo_loss.symmetry import A1SymmetryPoseLoss, per_instance_kpt_loss

DEV="cuda"; P180=tuple(json.load(open(f"{R}/FIXED_P180_PERMUTATIONS.json"))["P180"])
perm=list(P180)+[8]
smap=json.load(open(f"{R}/STEM_ASSET_MAP.json"))
ROWS=[]

_orig=A1SymmetryPoseLoss.calculate_keypoints_loss
def probe(self, masks, target_gt_idx, keypoints, batch_idx, stride_tensor,
          target_bboxes, pred_kpts):
    if masks.any() and getattr(self,"_probe",False):
        sel=self._select_target_keypoints(keypoints, batch_idx, target_gt_idx, masks).clone()
        sel[...,:2]/=stride_tensor.view(1,-1,1,1)
        gt=sel[masks]; pr=pred_kpts[masks]
        area=xyxy2xywh((target_bboxes/stride_tensor)[masks])[:,2:].prod(1,keepdim=True)
        m=gt[...,2]!=0
        did=per_instance_kpt_loss(self.keypoint_loss, pr, gt, m, area)
        d180=per_instance_kpt_loss(self.keypoint_loss, pr, gt[:,perm,:], m[:,perm], area)
        bs=masks.shape[0]
        img=torch.arange(bs,device=masks.device)[:,None].expand_as(masks)[masks]
        files=(self._batch or {}).get("im_file") or []
        for n,i in enumerate(img.tolist()):
            st=os.path.splitext(os.path.basename(files[i]))[0] if i<len(files) else "?"
            ROWS.append((smap.get(st,"?"), float(did[n]), float(d180[n]), st))
    return _orig(self, masks, target_gt_idx, keypoints, batch_idx, stride_tensor,
                 target_bboxes, pred_kpts)
A1SymmetryPoseLoss.calculate_keypoints_loss=probe

y=YOLO(INIT, task="pose"); model=y.model.to(DEV).float().train()
for q in model.parameters(): q.requires_grad_(False)
data=check_det_dataset(DATA)
args=get_cfg(DEFAULT_CFG, overrides=dict(task="pose", mode="train", data=DATA, imgsz=640,
    batch=16, epochs=1, workers=4, device=0, seed=0, single_cls=True,
    mosaic=0.0, scale=0.0, hsv_s=0.0, hsv_v=0.0, fliplr=0.0, flipud=0.0, erasing=0.0))
model.args=args; model.nc=data["nc"]; model.names=data["names"]
ds=build_yolo_dataset(args, data["train"], 16, data, mode="train", rect=False, stride=32)
dl=build_dataloader(ds, 16, 4, shuffle=False, rank=-1)

os.environ["A1_CONFIG"]=""; os.environ["PSPC_CONFIG"]=""
crit=E2ELoss(model, A1SymmetryPoseLoss) if getattr(model,"end2end",False) \
     else A1SymmetryPoseLoss(model)
# one2many 경로에서만 기록한다 (o2o 는 anchor 1개라 분포가 다르다)
getattr(crit,"one2many",crit)._probe=True

seen=0
for batch in dl:
    if seen>=N_IMG: break
    batch["img"]=batch["img"].to(DEV).float()/255
    for k in ("keypoints","bboxes","cls","batch_idx"): batch[k]=batch[k].to(DEV)
    with torch.no_grad():
        crit(model(batch["img"]), batch)
    seen+=batch["img"].shape[0]

out={"n_images":seen,"n_instances":len(ROWS),"checkpoint":INIT,"P180":list(P180),
     "probe":"one2many branch, anchor-level, per-instance OKS-form keypoint loss",
     "per_asset":{}}
by=collections.defaultdict(list)
for a,x,z,_ in ROWS: by[a].append((x,z))
for a,v in sorted(by.items()):
    x=np.array([q[0] for q in v]); z=np.array([q[1] for q in v]); s=z-x
    out["per_asset"][a]={
        "n":len(v),
        "d_id":{f"p{p}":float(np.percentile(x,p)) for p in (10,50,90)},
        "d_180":{f"p{p}":float(np.percentile(z,p)) for p in (10,50,90)},
        "sep_d180_minus_did":{f"p{p}":float(np.percentile(s,p)) for p in (5,25,50,75,95)},
        "frac_flipped_d180_lt_did": float((s<0).mean()),
        "frac_near_tie_abs_sep_lt_002": float((np.abs(s)<0.02).mean())}
# ---- frame 단위 (yaw180 트랩은 프레임 현상이다) --------------------------
fr=collections.defaultdict(lambda: [0,0,None])
for a,x,z,st in ROWS:
    fr[st][0]+=1; fr[st][1]+= (z<x); fr[st][2]=a
fa=collections.defaultdict(lambda:[0,0])
for st,(n,k,a) in fr.items():
    fa[a][0]+=1; fa[a][1]+= (k> n/2)      # 다수 anchor 가 뒤집힘 = 프레임 플립
out["frame_level"]={a:{"n_frames":v[0],"n_flipped":v[1],
                       "frac_flipped":v[1]/max(v[0],1)} for a,v in sorted(fa.items())}
out["frame_level"]["ALL"]={"n_frames":sum(v[0] for v in fa.values()),
    "n_flipped":sum(v[1] for v in fa.values()),
    "frac_flipped":sum(v[1] for v in fa.values())/max(sum(v[0] for v in fa.values()),1)}
# ---- margin 활성 곡선 (role term 이 실제로 작동할 수 있는가) --------------
allsep=np.array([z-x for _,x,z,_ in ROWS])
out["role_hinge_activation_by_margin"]={
    f"m={m}": float((allsep < m).mean()) for m in
    (0.0,0.01,0.02,0.05,0.10,0.20,0.40,0.60,0.85,1.00)}
out["caveat"]=("수렴한 A0 체크포인트에서 측정했다. 학습 도중 flip 률은 이보다 "
               "높으므로 이 값은 min() 발화 빈도의 **하한**이다. 상한은 측정 불가"
               "(학습을 돌려야 하는데 contract 전 학습 금지).")
json.dump(out, open(f"{R}/A1_PRECALIB.json","w"), indent=2)
print(f"images {seen}  instances {len(ROWS)}")
print(f"{'asset':38} {'n':>7} {'d_id p50':>9} {'d180 p50':>9} {'sep p50':>9} {'flip%':>7} {'tie%':>6}")
print("─"*92)
for a,v in out["per_asset"].items():
    print(f"{a:38} {v['n']:7d} {v['d_id']['p50']:9.4f} {v['d_180']['p50']:9.4f} "
          f"{v['sep_d180_minus_did']['p50']:9.4f} {100*v['frac_flipped_d180_lt_did']:6.2f}% "
          f"{100*v['frac_near_tie_abs_sep_lt_002']:5.2f}%")
print()
print(f"{'frame-level (다수 anchor 뒤집힘)':44} {'frames':>8} {'flipped':>8} {'%':>7}")
print("─"*72)
for a,v in out["frame_level"].items():
    print(f"{a:44} {v['n_frames']:8d} {v['n_flipped']:8d} {100*v['frac_flipped']:6.2f}%")
print()
print("role hinge 활성 비율 (ASYM 에서 margin m 일 때 relu 가 켜지는 instance 비율)")
print("─"*72)
print("  " + "  ".join(f"{k}:{100*v:5.1f}%" for k,v in out["role_hinge_activation_by_margin"].items()))
