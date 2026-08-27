"""PHASE E/F/H — 13개 테스트. 전부 PASS 전에는 학습 금지."""
from __future__ import annotations
import json, os, sys, copy
import numpy as np, torch

ROOT="/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, ROOT)
R=os.path.join(ROOT,"challenge/yolo_pose_one_model/runs_pallet_loss")
DATA=os.path.join(ROOT,"challenge/yolo_pose_one_model/datasets/v1_fixed_matched10k/data.yaml")
# ★ parity 테스트는 **학습 경로와 같은 kpt_shape(9)** 이어야 한다.
# COCO 사전학습본은 17 kp 라 loss 의 sigmas 가 어긋난다.
INIT=os.path.join(ROOT,"challenge/yolo_pose_one_model/runs_fixed/"
                       "V1_FIXED_MATCHED10K_60EP_SEED42_UBUNTU/weights/last.pt")
res={}; ok_all=True
def rec(name, ok, detail=""):
    global ok_all
    res[name]={"pass":bool(ok),"detail":detail}; ok_all &= bool(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {name:44} {detail}")

from ultralytics import YOLO
from ultralytics.models.yolo.pose import PoseTrainer
from ultralytics.utils.loss import E2ELoss, PoseLoss26
from pallet_yolo_loss.loss import PSPCPoseLoss26, PSPCConfig
from pallet_yolo_loss.model import PSPCPoseModel

torch.manual_seed(0)
DEV="cuda" if torch.cuda.is_available() else "cpu"

# ---- 배치 하나 확보 -------------------------------------------------------
from ultralytics.data.build import build_yolo_dataset, build_dataloader
from ultralytics.data.utils import check_det_dataset
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG

y=YOLO(INIT, task="pose")
model=y.model.to(DEV).float().train()
for q in model.parameters(): q.requires_grad_(True)   # .pt 로드본은 grad 가 꺼져 있다
data=check_det_dataset(DATA)
args=get_cfg(DEFAULT_CFG, overrides=dict(
    task="pose", mode="train", data=DATA, imgsz=640, batch=8, epochs=1,
    workers=2, device=0, seed=0, single_cls=True, mosaic=0.3, scale=0.25,
    hsv_s=0.5, hsv_v=0.35, fliplr=0.0, flipud=0.0, cos_lr=True, close_mosaic=10))
model.args = args
model.nc = data["nc"]; model.names = data["names"]
ds=build_yolo_dataset(args, data["train"], 8, data, mode="train", rect=False, stride=32)
dl=build_dataloader(ds, 8, 2, shuffle=False, rank=-1)
batch=next(iter(dl))
batch["img"]=batch["img"].to(DEV).float()/255
for k in ("keypoints","bboxes","cls","batch_idx"):
    if k in batch: batch[k]=batch[k].to(DEV)

def fresh_preds():
    model.zero_grad(set_to_none=True)
    return model(batch["img"])

# ---- T3 diagonal derivation ----------------------------------------------
dp=json.load(open(os.path.join(R,"CUBOID_DIAGONAL_PAIRS.json")))
pairs=[tuple(x) for x in dp["pairs"]]
used=sorted([x for p in pairs for x in p])
rec("T3 diagonal derivation", len(pairs)==4 and used==list(range(8)) and dp["centroid_index"]==8,
    f"{pairs}")

# ---- criterion 생성 -------------------------------------------------------
cfg0=os.path.join(R,"_cfg_lambda0.json")
json.dump({"enabled":True,"lambda_pc":0.0,"diagonal_pairs":[list(p) for p in pairs],
           "centroid_index":8},open(cfg0,"w"))
cfg1=os.path.join(R,"_cfg_lambda1.json")
json.dump({"enabled":True,"lambda_pc":1.0,"diagonal_pairs":[list(p) for p in pairs],
           "centroid_index":8},open(cfg1,"w"))

def make(kind, cfgpath):
    os.environ["PSPC_CONFIG"]=cfgpath
    torch.manual_seed(0)
    return E2ELoss(model, kind) if getattr(model,"end2end",False) else kind(model)

std=make(PoseLoss26,cfg0)
os.environ["PSPC_CONFIG"]=cfg0
cus=make(PSPCPoseLoss26,cfg0)

# ---- T10/T11 두 경로 모두 subclass 인가 -----------------------------------
rec("T10 one2many uses PSPC", isinstance(getattr(cus,"one2many",cus),PSPCPoseLoss26),
    type(getattr(cus,"one2many",cus)).__name__)
rec("T11 one2one uses PSPC", isinstance(getattr(cus,"one2one",cus),PSPCPoseLoss26),
    type(getattr(cus,"one2one",cus)).__name__)

# ---- T1/T2 parity (lambda=0) ---------------------------------------------
def run(crit, need_grad=False):
    torch.manual_seed(0)
    p=fresh_preds()
    b=copy.deepcopy(batch)
    l,items=crit(p,b)
    if need_grad:
        l.sum().backward()
        g=[q.grad.detach().clone() for q in model.parameters() if q.grad is not None]
        return float(l.sum()), items.detach().cpu().numpy(), g
    return float(l.sum()), items.detach().cpu().numpy(), None

ls,is_,gs=run(std,True)
lc,ic,gc=run(cus,True)
dl_=abs(ls-lc); di=float(np.abs(is_-ic).max())
rec("T1 baseline loss parity", dl_<1e-6 and di<1e-6, f"total {dl_:.3e} items {di:.3e}")
dg=max((a-b).abs().max().item() for a,b in zip(gs,gc)) if len(gs)==len(gc) else 9e9
# ★ backward 는 cuDNN algo 선택 때문에 비결정적이다. 절대 1e-6 로 재면
#   구현 차이가 아니라 잡음을 재게 된다. 같은 criterion 두 번의 잡음 바닥과 비교한다.
_,_,gs2=run(std,True)
noise=max((a-b).abs().max().item() for a,b in zip(gs,gs2))
rec("T2 baseline gradient parity", dg<=max(noise,1e-6)*1.5,
    f"max|dgrad| {dg:.3e}  noise floor(std vs std) {noise:.3e}")

# ---- PC 단독 동작 확인용 헬퍼 ---------------------------------------------
os.environ["PSPC_CONFIG"]=cfg1
pc_crit=PSPCPoseLoss26(model)
pc_crit.pspc=PSPCConfig(enabled=True,lambda_pc=1.0,diagonal_pairs=tuple(pairs),centroid_index=8)

def pc_from(pred_kpts, gt_kpts, area_scale=1.0):
    """테스트용 직접 계산 — loss.py 와 같은 식."""
    c=gt_kpts[:,8,:2]; tot=0.0; n=0
    for i,j in pairs:
        a=pred_kpts[:,i,:2]; b=pred_kpts[:,j,:2]; ab=b-a
        cr=(ab[:,0]*(c[:,1]-a[:,1])-ab[:,1]*(c[:,0]-a[:,0])).abs()
        tot=tot+cr/(ab.norm(dim=1)+1e-9)/area_scale; n+=1
    return tot/n

# ---- T4 GT residual ~ 0 ---------------------------------------------------
gtk=batch["keypoints"].clone().float()
gtk[...,0]*=640; gtk[...,1]*=640
# ★ loss 와 같은 유효성 마스크를 건다. 화면 밖 점은 (0,0,v=0) 으로 저장돼 있어
#   마스크 없이 재면 직선성이 깨진 것처럼 보인다 (첫 판에서 그랬다).
vis=gtk[...,2]!=0
allv=vis[:,8]
for i,j in pairs: allv=allv & vis[:,i] & vis[:,j]
rr=pc_from(gtk[:,:,:2].clone(), gtk)[allv]
rec("T4 GT projective residual ~0", float(rr.max())<1e-2 if rr.numel() else False,
    f"n={int(allv.sum())}/{len(allv)}  median {float(rr.median()):.3e} "
    f"p99 {float(np.percentile(rr.cpu().numpy(),99)):.3e} max {float(rr.max()):.3e}")

# ---- T5 perturbation monotonic -------------------------------------------
vals=[]
for px in (0,1,5,20):
    p=gtk[:,:,:2].clone(); p[:,0,0]+=px
    vals.append(float(pc_from(p,gtk).mean()))
rec("T5 +1/+5/+20px monotonic", all(vals[k]<vals[k+1] for k in range(3)),
    " -> ".join(f"{v:.4f}" for v in vals))

# ---- T6/T7 gradient 방향 --------------------------------------------------
p=gtk[:,:,:2].clone().requires_grad_(True)
pk=torch.cat([p, torch.zeros_like(p[...,:1])],-1)
pk=pk.clone(); pk[:,0,0]=pk[:,0,0]+3.0
p2=gtk[:,:,:2].clone(); p2[:,0,0]+=3.0; p2.requires_grad_(True)
pc_from(p2,gtk).mean().backward()
g=p2.grad
i0,j0=pairs[0]
rec("T6 endpoint gradients nonzero",
    g[:,i0].abs().sum()>0 and g[:,j0].abs().sum()>0,
    f"|g[{i0}]| {float(g[:,i0].abs().sum()):.4f}  |g[{j0}]| {float(g[:,j0].abs().sum()):.4f}")
rec("T7 centroid PC gradient == 0", float(g[:,8].abs().max())==0.0,
    f"max|g[8]| {float(g[:,8].abs().max()):.3e}")

# ---- T8/T9 안전성 ---------------------------------------------------------
gz=gtk.clone(); gz[...,2]=0
try:
    v=pc_crit.projective_loss(torch.zeros(1,1,dtype=torch.bool,device=DEV),
        torch.zeros(1,1,dtype=torch.long,device=DEV), batch["keypoints"].to(DEV),
        batch["batch_idx"].to(DEV), torch.ones(1,1,device=DEV),
        torch.ones(1,1,4,device=DEV), torch.zeros(1,1,9,3,device=DEV))
    rec("T8 masked/empty safe", torch.isfinite(v).all() and float(v)==0.0, f"pc={float(v):.3e}")
except Exception as e:
    rec("T8 masked/empty safe", False, str(e)[:60])
pdeg=gtk[:,:,:2].clone(); pdeg[:,pairs[0][1]]=pdeg[:,pairs[0][0]]
vd=pc_from(pdeg,gtk)
rec("T9 degenerate segment safe", bool(torch.isfinite(vd).all()), f"finite={bool(torch.isfinite(vd).all())}")

# ---- T12 forward/backward finite (lambda=1) -------------------------------
os.environ["PSPC_CONFIG"]=cfg1
cus1=make(PSPCPoseLoss26,cfg1)
l1,i1,g1=run(cus1,True)
rec("T12 forward/backward finite", np.isfinite(l1) and all(torch.isfinite(x).all() for x in g1),
    f"loss {l1:.4f}")
rec("T13 lambda=1 changes loss", abs(l1-ls)>1e-9, f"std {ls:.5f} -> pc {l1:.5f}")

json.dump({"passed":sum(1 for v in res.values() if v["pass"]),"total":len(res),
           "all_pass":ok_all,"tests":res},
          open(os.path.join(R,"TEST_RESULTS.json"),"w"),indent=1,ensure_ascii=False)
print(f"\n{sum(1 for v in res.values() if v['pass'])}/{len(res)} PASS   ALL={ok_all}")
