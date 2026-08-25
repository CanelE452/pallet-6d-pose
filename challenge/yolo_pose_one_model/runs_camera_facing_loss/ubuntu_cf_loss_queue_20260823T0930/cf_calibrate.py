"""CF-NRL 캘리브레이션 — CF-A0 train residual 로 beta/lambda 를 후보 결과 전에 고정."""
import json, os, sys, hashlib
import numpy as np, torch
ROOT="/home/minjae/Documents/github/pallet-pose"; sys.path.insert(0,ROOT)
Q=os.path.dirname(os.path.abspath(__file__))
CFR=f"{ROOT}/challenge/yolo_pose_one_model/runs_camera_facing_loss"
A0=f"{CFR}/CF_A0_V1MATCHED10K_60EP_SEED42_UBUNTU/weights/last.pt"
DATA=f"{ROOT}/challenge/yolo_pose_one_model/datasets/v1_cf_matched10k/data.yaml"
NB=8
from ultralytics import YOLO
from ultralytics.utils.loss import E2ELoss
from ultralytics.utils.ops import xyxy2xywh
from ultralytics.data.build import build_yolo_dataset, build_dataloader
from ultralytics.data.utils import check_det_dataset
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG
from pallet_yolo_loss.symmetry import A1SymmetryPoseLoss, nrl_coord
DEV="cuda"
_o=A1SymmetryPoseLoss.calculate_keypoints_loss
def probe(self, masks, tgi, kpts, bidx, st, tb, pk):
    if masks.any() and getattr(self,"_probe",False):
        sel=self._select_target_keypoints(kpts,bidx,tgi,masks)
        sel=sel.clone(); sel[...,:2]/=st.view(1,-1,1,1)
        gt=sel[masks]; pr=pk[masks]
        area=xyxy2xywh((tb/st)[masks])[:,2:].prod(1,keepdim=True)
        self._cache=(pr,gt,gt[...,2]!=0,area)
    return _o(self, masks, tgi, kpts, bidx, st, tb, pk)
A1SymmetryPoseLoss.calculate_keypoints_loss=probe
y=YOLO(A0,task="pose"); model=y.model.to(DEV).float().train()
data=check_det_dataset(DATA)
args=get_cfg(DEFAULT_CFG,overrides=dict(task="pose",mode="train",data=DATA,imgsz=640,batch=16,
  epochs=1,workers=4,device=0,seed=0,single_cls=True,mosaic=0.0,scale=0.0,hsv_s=0.0,
  hsv_v=0.0,fliplr=0.0,flipud=0.0,erasing=0.0))
model.args=args; model.nc=data["nc"]; model.names=data["names"]
ds=build_yolo_dataset(args,data["train"],16,data,mode="train",rect=False,stride=32)
dl=build_dataloader(ds,16,4,shuffle=False,rank=-1)
os.environ["A1_CONFIG"]=""; os.environ["PSPC_CONFIG"]=""
crit=E2ELoss(model,A1SymmetryPoseLoss) if getattr(model,"end2end",False) else A1SymmetryPoseLoss(model)
inner=getattr(crit,"one2many",crit); inner._probe=True
for p in model.parameters(): p.requires_grad_(False)
AX=[]; batches=[]; seen=0; ids=[]
for b in dl:
    if seen>=1024: break
    b["img"]=b["img"].to(DEV).float()/255
    for k in ("keypoints","bboxes","cls","batch_idx"): b[k]=b[k].to(DEV)
    if len(batches)<NB:
        batches.append(b); ids += [os.path.basename(f) for f in b["im_file"]]
    with torch.no_grad(): crit(model(b["img"]), b)
    pr,gt,m,area=inner._cache
    sx=area.clamp_min(1e-9).sqrt()
    AX.append(torch.cat([((pr[...,0]-gt[...,0])/sx)[m].abs(),
                         ((pr[...,1]-gt[...,1])/sx)[m].abs()]).float().cpu().numpy())
    seen+=b["img"].shape[0]
ax=np.concatenate(AX); beta=float(np.percentile(ax,75))
for p in model.parameters(): p.requires_grad_(True)
head=[p for p in model.parameters() if p.requires_grad]
def gnorm(fn):
    out=[]
    for b in batches:
        model.zero_grad(set_to_none=True); crit(model(b["img"]), b)
        pr,gt,m,area=inner._cache
        g=torch.autograd.grad(fn(pr,gt,m,area), head, allow_unused=True)
        out.append(float(torch.sqrt(sum(x.pow(2).sum() for x in g if x is not None))))
    return out
gb=gnorm(lambda pr,gt,m,area: inner.keypoint_loss(pr,gt,m,area))
gn=gnorm(lambda pr,gt,m,area: nrl_coord(pr,gt,m,area,beta))
mb,mn=float(np.median(gb)),float(np.median(gn))
lam=mb/(mn+1e-12)
cal={"source_checkpoint":A0,"n_images":seen,"n_axis_residual":int(ax.size),
     "axis_abs_q":{f"q{q}":float(np.percentile(ax,q)) for q in (50,75,90,95)},
     "nrl_beta":beta,"grad_median_base":mb,"grad_median_nrl":mn,"nrl_lambda":lam,
     "calib_batches":NB,"batch_ids_sha16":hashlib.sha256("".join(sorted(ids)).encode()).hexdigest()[:16],
     "stable":bool(1e-3<=lam<=10),
     "frozen_before_candidate_results":True}
json.dump(cal, open(f"{Q}/CF_CALIBRATION.json","w"), indent=2)
if cal["stable"]:
    json.dump({"enabled":True,"mode":"exact_min","lambda_role":0.0,"margin":0.0,
      "p180":[5,4,7,6,1,0,3,2],"centroid_index":8,"sym_assets":[],"asym_assets":[],
      "stem_asset_map":"","role_ramp":[5,20],"asc_enabled":False,"takl_enabled":False,
      "nrl_enabled":True,"nrl_beta":beta,"nrl_lambda":lam},
      open(f"{Q}/CF_NRL_CONFIG.json","w"), indent=2)
print(f"  beta {beta:.5f}  grad base {mb:.4f} nrl {mn:.4f}  lambda {lam:.5f}  stable {cal['stable']}")
