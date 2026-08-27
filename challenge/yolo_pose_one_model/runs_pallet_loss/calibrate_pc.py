"""PHASE G — lambda_pc 캘리브레이션. val/real 성능을 보고 정하지 않는다.

train 전용 고정 batch 에서 decoded predicted keypoints 에 대한 gradient norm 을
재고, auxiliary 가 base pose gradient 의 10% 가 되게 맞춘다.
"""
from __future__ import annotations
import hashlib, json, os, sys
import numpy as np, torch
ROOT="/home/minjae/Documents/github/pallet-pose"; sys.path.insert(0,ROOT)
R=os.path.join(ROOT,"challenge/yolo_pose_one_model/runs_pallet_loss")
DATA=os.path.join(ROOT,"challenge/yolo_pose_one_model/datasets/v1_fixed_matched10k/data.yaml")
INIT=os.path.join(ROOT,"challenge/weights/pretrained_yolo/yolo26n-pose.pt")
N_BATCH=8; SEED=20260822; TARGET=0.10
from ultralytics import YOLO
from ultralytics.data.build import build_yolo_dataset, build_dataloader
from ultralytics.data.utils import check_det_dataset
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.loss import PoseLoss26
from ultralytics.utils.ops import xyxy2xywh
from pallet_yolo_loss.loss import PSPCConfig
DEV="cuda"
torch.manual_seed(SEED); np.random.seed(SEED)
pairs=[tuple(x) for x in json.load(open(os.path.join(R,"CUBOID_DIAGONAL_PAIRS.json")))["pairs"]]

# ★ 캘리브레이션은 **학습 시작 시점** 에서 한다.
# 사전학습본은 17 kp head 라 그대로 못 쓴다 — trainer 와 동일하게
# 9 kp head 를 만들고 pretrained 가중치를 load 한다.
from ultralytics.nn.tasks import PoseModel
data=check_det_dataset(DATA)
args=get_cfg(DEFAULT_CFG,overrides=dict(task="pose",mode="train",data=DATA,imgsz=640,
    batch=8,epochs=60,workers=2,device=0,seed=42,single_cls=True,mosaic=0.3,scale=0.25,
    hsv_s=0.5,hsv_v=0.35,fliplr=0.0,flipud=0.0,cos_lr=True,close_mosaic=10))
model=PoseModel("yolo26n-pose.yaml", nc=data["nc"], ch=data["channels"],
                data_kpt_shape=data["kpt_shape"], verbose=False)
model.load(YOLO(INIT, task="pose").model)
model=model.to(DEV).float().train()
model.args=args; model.nc=data["nc"]; model.names=data["names"]
for q in model.parameters(): q.requires_grad_(True)
ds=build_yolo_dataset(args,data["train"],8,data,mode="train",rect=False,stride=32)
dl=build_dataloader(ds,8,2,shuffle=False,rank=-1)

crit=PoseLoss26(model)
gb,gp,ids=[],[],[]
it=iter(dl)
for bi in range(N_BATCH):
    b=next(it)
    ids += [os.path.basename(x) for x in b["im_file"]]
    b["img"]=b["img"].to(DEV).float()/255
    for k in ("keypoints","bboxes","cls","batch_idx"):
        if k in b: b[k]=b[k].to(DEV)
    model.zero_grad(set_to_none=True)
    preds=model(b["img"])
    p=crit.parse_output(preds) if hasattr(crit,"parse_output") else preds
    o2m=p["one2many"] if isinstance(p,dict) and "one2many" in p else p
    pk=o2m["kpts"].permute(0,2,1).contiguous()
    bs=pk.shape[0]
    (fg,tgi,tb,ap,st),_,_=crit.get_assigned_targets_and_loss(o2m,b)
    pk=pk.view(bs,-1,*crit.kpt_shape)
    pk=crit.kpts_decode(ap,pk)
    pk=pk.detach().requires_grad_(True)      # decoded predicted keypoints
    if not fg.any(): continue
    kp=b["keypoints"].to(DEV).float().clone()
    imgsz=torch.tensor(o2m["feats"][0].shape[2:],device=DEV,dtype=pk.dtype)*crit.stride[0]
    kp[...,0]*=imgsz[1]; kp[...,1]*=imgsz[0]
    sel=crit._select_target_keypoints(kp,b["batch_idx"].to(DEV),tgi,fg).clone()
    sel[...,:2]/=st.view(1,-1,1,1)
    gt=sel[fg]; pred=pk[fg]
    tbs=tb/st; area=xyxy2xywh(tbs[fg])[:,2:].prod(1,keepdim=True)
    m=gt[...,2]!=0
    # base
    pk.grad=None
    crit.keypoint_loss(pred,gt,m,area).backward(retain_graph=True)
    gb.append(float(pk.grad.norm()))
    # pc
    pk.grad=None
    c=gt[:,8,:2]; sc=area.squeeze(1).clamp_min(1e-9).sqrt(); tot=0; n=0
    for i,j in pairs:
        a=pred[:,i,:2]; bb=pred[:,j,:2]; ab=bb-a
        cr=(ab[:,0]*(c[:,1]-a[:,1])-ab[:,1]*(c[:,0]-a[:,0])).abs()
        ok=(gt[:,8,2]!=0)&(gt[:,i,2]!=0)&(gt[:,j,2]!=0)&(ab.norm(dim=1)>1e-3)
        d=cr/(ab.norm(dim=1)+1e-9)/(sc+1e-9)
        tot=tot+torch.where(ok,d,torch.zeros_like(d)); n=n+ok.float()
    (tot/n.clamp_min(1.0))[n>0].mean().backward()
    gp.append(float(pk.grad.norm()))
    print(f"  batch {bi}: g_base {gb[-1]:.5f}  g_pc {gp[-1]:.5f}")

gbm,gpm=float(np.median(gb)),float(np.median(gp))
lam=TARGET*gbm/(gpm+1e-12)
sha=hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()
unstable = not (1e-3 <= lam <= 10)
out={"target_ratio":TARGET,"n_batches":len(gb),"seed":SEED,
 "g_base_median":gbm,"g_pc_median":gpm,"g_base_all":gb,"g_pc_all":gp,
 "lambda_pc":lam,"bounds":[1e-3,10],"CALIBRATION_UNSTABLE":unstable,
 "calibration_batch_sha256":sha,"n_images":len(ids),
 "★rule":"train 전용 고정 batch. val/real 결과를 보고 바꾸지 않는다.",
 "init":"pretrained yolo26n-pose.pt (학습 시작 시점 비율)"}
json.dump(out,open(os.path.join(R,"LOSS_CALIBRATION.json"),"w"),indent=1,ensure_ascii=False)
open(os.path.join(R,"CALIBRATION_BATCHES.sha256"),"w").write(f"{sha}  {len(ids)} images\n")
print(f"\n  g_base median {gbm:.5f}   g_pc median {gpm:.5f}")
print(f"  lambda_pc = {lam:.6f}   UNSTABLE={unstable}")
