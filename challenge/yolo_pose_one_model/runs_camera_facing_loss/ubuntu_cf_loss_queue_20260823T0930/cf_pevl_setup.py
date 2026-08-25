"""CF-PEVL 준비 — camera-facing contract 에서 12 edge 를 유도하고 lambda/alpha 를 고정.

★ fixed-object edge 매핑 하드코딩을 쓰지 않는다.  현재 CF 라벨의 3D cuboid 좌표
  부호에서 직접 유도한다.
"""
import json, os, sys, glob, itertools
import numpy as np, torch
ROOT="/home/minjae/Documents/github/pallet-pose"; sys.path.insert(0,ROOT)
Q=os.path.dirname(os.path.abspath(__file__))
CFR=f"{ROOT}/challenge/yolo_pose_one_model/runs_camera_facing_loss"
A0=f"{CFR}/CF_A0_V1MATCHED10K_60EP_SEED42_UBUNTU/weights/last.pt"
DATA=f"{ROOT}/challenge/yolo_pose_one_model/datasets/v1_cf_matched10k/data.yaml"
S=f"{ROOT}/data/pallet/training_data/paper_release/v2_prod40k_clean_merged/labels"

# ---- 1) camera-facing 3D cuboid 에서 edge 유도 ------------------------------
stems=[f[:-4] for sp in ("train","val")
       for f in sorted(os.listdir(f"{ROOT}/challenge/yolo_pose_one_model/datasets/v1_cf_matched10k/labels/{sp}"))]
votes={}
for s in stems[:200]:
    j=json.load(open(f"{S}/{s}_label.json"))["objects"][0]
    cub=j.get("cuboid")
    if not cub or len(cub)<8: continue
    P=np.array(cub[:8],dtype=float)
    perm=j.get("perm_v4")
    if perm is None: continue
    # fixed[perm[i]] = cf[i]  ->  cf 순서의 3D = fixed 3D 를 역적용
    cf3=np.empty_like(P); cf3[list(range(8))]=P[list(perm)]
    C=cf3.mean(0); R=cf3-C
    sg=np.sign(np.round(R/ (np.abs(R).max(0)+1e-12), 3))
    for a,b in itertools.combinations(range(8),2):
        if int((sg[a]!=sg[b]).sum())==1:      # 부호가 한 축만 다르면 물리적 edge
            votes[(a,b)]=votes.get((a,b),0)+1
edges=sorted([e for e,v in votes.items() if v>=0.9*max(votes.values())])
deg={i:0 for i in range(8)}
for a,b in edges: deg[a]+=1; deg[b]+=1
ok = len(edges)==12 and all(d==3 for d in deg.values())
if not ok:
    json.dump({"edges":[list(e) for e in edges],"degree":deg,"valid":False,
               "reason":"12 edge / degree 3 조건 불충족"},
              open(f"{Q}/CF_PEVL_EDGES.json","w"), indent=2)
    raise SystemExit(f"edge 유도 실패: {len(edges)} edges, degree {deg}")
json.dump({"edges":[list(e) for e in edges],"degree":deg,"valid":True,
           "n_frames_voted":min(200,len(stems)),
           "derivation":"CF 3D cuboid 의 축 부호가 정확히 하나만 다른 코너쌍",
           "centroid_excluded":True},
          open(f"{Q}/CF_PEVL_EDGES.json","w"), indent=2)

# ---- 2) alpha_len / lambda / q95 ------------------------------------------
from ultralytics import YOLO
from ultralytics.utils.loss import E2ELoss
from ultralytics.utils.ops import xyxy2xywh
from ultralytics.data.build import build_yolo_dataset, build_dataloader
from ultralytics.data.utils import check_det_dataset
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG
from pallet_yolo_loss.symmetry import (A1SymmetryPoseLoss, pevl_loss,
                                       normalized_residual, smooth_l1_to_zero, EPS)
DEV="cuda"
_o=A1SymmetryPoseLoss.calculate_keypoints_loss
def probe(self,masks,tgi,kpts,bidx,st,tb,pk):
    if masks.any() and getattr(self,"_probe",False):
        sel=self._select_target_keypoints(kpts,bidx,tgi,masks).clone()
        sel[...,:2]/=st.view(1,-1,1,1)
        gt=sel[masks]; pr=pk[masks]
        self._cache=(pr,gt,gt[...,2]!=0,xyxy2xywh((tb/st)[masks])[:,2:].prod(1,keepdim=True))
    return _o(self,masks,tgi,kpts,bidx,st,tb,pk)
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
R=[]; D=[]; L=[]; batches=[]; seen=0
for b in dl:
    if seen>=512: break
    b["img"]=b["img"].to(DEV).float()/255
    for k in ("keypoints","bboxes","cls","batch_idx"): b[k]=b[k].to(DEV)
    if len(batches)<8: batches.append(b)
    with torch.no_grad(): crit(model(b["img"]), b)
    pr,gt,m,area=inner._cache
    R.append(normalized_residual(pr,gt,area)[m].float().cpu().numpy())
    for i,j in edges:
        v=m[:,i]&m[:,j]
        if not bool(v.any()): continue
        pe=pr[v][:,j,:2]-pr[v][:,i,:2]; ge=gt[v][:,j,:2]-gt[v][:,i,:2]
        pn=pe.norm(dim=1).clamp_min(EPS); gn=ge.norm(dim=1).clamp_min(EPS)
        D.append(float((1-(pe*ge).sum(1)/(pn*gn)).mean()))
        L.append(float(smooth_l1_to_zero(torch.log((pn+EPS)/(gn+EPS)),0.1).mean()))
    seen+=b["img"].shape[0]
r=np.concatenate(R); q95=float(np.percentile(r,95))
alpha=float(np.median(D)/(np.median(L)+1e-12))
for p in model.parameters(): p.requires_grad_(True)
head=[p for p in model.parameters() if p.requires_grad]
def gnorm(fn):
    out=[]
    for b in batches:
        model.zero_grad(set_to_none=True); crit(model(b["img"]), b)
        pr,gt,m,area=inner._cache
        g=torch.autograd.grad(fn(pr,gt,m,area),head,allow_unused=True)
        out.append(float(torch.sqrt(sum(x.pow(2).sum() for x in g if x is not None))))
    return out
gb=gnorm(lambda pr,gt,m,area: inner.keypoint_loss(pr,gt,m,area))
gp=gnorm(lambda pr,gt,m,area: pevl_loss(pr,gt,m,area,edges,alpha,q95))
mb,mp=float(np.median(gb)),float(np.median(gp))
lam=float(0.10*mb/(mp+1e-12))
json.dump({"edges":[list(e) for e in edges],"alpha_len":alpha,"pevl_lambda":lam,
           "resid_q95":q95,"grad_median_base":mb,"grad_median_pevl":mp,
           "median_dir":float(np.median(D)),"median_len":float(np.median(L)),
           "ramp":[10,20],"frozen_before_candidate_results":True},
          open(f"{Q}/CF_PEVL_CALIBRATION.json","w"), indent=2)
json.dump({"enabled":True,"mode":"exact_min","lambda_role":0.0,"margin":0.0,
  "p180":[5,4,7,6,1,0,3,2],"centroid_index":8,"sym_assets":[],"asym_assets":[],
  "stem_asset_map":"","role_ramp":[5,20],"asc_enabled":False,"takl_enabled":False,
  "nrl_enabled":False,"pevl_enabled":True,"pevl_edges":[list(e) for e in edges],
  "pevl_alpha_len":alpha,"pevl_lambda":lam,"pevl_resid_q95":q95,"pevl_ramp":[10,20]},
  open(f"{Q}/CF_PEVL_CONFIG.json","w"), indent=2)
print(f"  edges {len(edges)} degree3 OK  alpha {alpha:.4f}  lambda {lam:.5f}  q95 {q95:.4f}")
