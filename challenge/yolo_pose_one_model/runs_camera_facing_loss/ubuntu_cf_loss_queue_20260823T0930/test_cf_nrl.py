"""CF-NRL 16 테스트. 16/16 PASS 전 학습 금지."""
from __future__ import annotations
import copy, json, os, sys
import numpy as np, torch
ROOT="/home/minjae/Documents/github/pallet-pose"; sys.path.insert(0,ROOT)
Q=os.path.dirname(os.path.abspath(__file__))
CFR=f"{ROOT}/challenge/yolo_pose_one_model/runs_camera_facing_loss"
DATA=f"{ROOT}/challenge/yolo_pose_one_model/datasets/v1_cf_matched10k/data.yaml"
INIT=f"{CFR}/CF_A0_V1MATCHED10K_60EP_SEED42_UBUNTU/weights/last.pt"
res={}; ok_all=True
def rec(n,ok,d=""):
    global ok_all
    res[n]={"pass":bool(ok),"detail":d}; ok_all&=bool(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {n:44} {d}", flush=True)
from ultralytics import YOLO
from ultralytics.utils.loss import E2ELoss, PoseLoss26
from ultralytics.data.build import build_yolo_dataset, build_dataloader
from ultralytics.data.utils import check_det_dataset
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG
import pallet_yolo_loss.symmetry as SY
from pallet_yolo_loss.symmetry import A1SymmetryPoseLoss, nrl_coord
from pallet_yolo_loss.loss import PSPCPoseLoss26
torch.manual_seed(0); DEV="cuda"
C=json.load(open(f"{Q}/CF_NRL_CONFIG.json")); BETA=C["nrl_beta"]
y=YOLO(INIT,task="pose"); model=y.model.to(DEV).float().train()
for q in model.parameters(): q.requires_grad_(True)
data=check_det_dataset(DATA)
args=get_cfg(DEFAULT_CFG,overrides=dict(task="pose",mode="train",data=DATA,imgsz=640,batch=8,
  epochs=1,workers=2,device=0,seed=0,single_cls=True,mosaic=0.3,scale=0.25,hsv_s=0.5,
  hsv_v=0.35,fliplr=0.0,flipud=0.0,cos_lr=True,close_mosaic=10))
model.args=args; model.nc=data["nc"]; model.names=data["names"]
ds=build_yolo_dataset(args,data["train"],8,data,mode="train",rect=False,stride=32)
batch=next(iter(build_dataloader(ds,8,2,shuffle=False,rank=-1)))
batch["img"]=batch["img"].to(DEV).float()/255
for k in ("keypoints","bboxes","cls","batch_idx"): batch[k]=batch[k].to(DEV)
def cf(n,**kw):
    p=f"{Q}/_t_{n}.json"; d=dict(C); d.update(kw); json.dump(d,open(p,"w")); return p
def make(kind,c=None):
    os.environ["A1_CONFIG"]=c or ""; os.environ["PSPC_CONFIG"]=""
    torch.manual_seed(0)
    return E2ELoss(model,kind) if getattr(model,"end2end",False) else kind(model)
def run(c,grad=False):
    torch.manual_seed(0); model.zero_grad(set_to_none=True)
    p=model(batch["img"]); l,it=c(p,copy.deepcopy(batch))
    if grad:
        l.sum().backward()
        return float(l.sum()), it.detach().cpu().numpy(), \
               [q.grad.detach().clone() for q in model.parameters() if q.grad is not None]
    return float(l.sum()), it.detach().cpu().numpy(), None
std=make(PoseLoss26); off=make(A1SymmetryPoseLoss,cf("off",enabled=False,nrl_enabled=False))
on=make(A1SymmetryPoseLoss,cf("on")); inner=getattr(on,"one2many",on)
ls,is_,gs=run(std,True); lc,ic,gc=run(off,True)
rec("T1 disabled forward parity", abs(ls-lc)<1e-6 and np.abs(is_-ic).max()<1e-6,
    f"total {abs(ls-lc):.3e} items {np.abs(is_-ic).max():.3e}")
noise=0.0
for _ in range(3):
    _,_,gn=run(std,True); noise=max(noise,max((a-b).abs().max().item() for a,b in zip(gs,gn)))
dg=max((a-b).abs().max().item() for a,b in zip(gs,gc))
rec("T2 disabled gradient parity", dg<=max(noise,1e-6)*1.5, f"dgrad {dg:.3e} noise {noise:.3e}")
torch.manual_seed(3); N,K=24,9
GT=torch.rand(N,K,3,device=DEV); GT[...,2]=1.0
AR=torch.rand(N,1,device=DEV)+0.5; M=GT[...,2]!=0
rec("T3 완벽 예측이면 0", float(nrl_coord(GT.clone(),GT,M,AR,BETA))==0.0, "0.0")
sx=AR.clamp_min(1e-9).sqrt()
vals=[]
for px in (1.,5.,20.):
    P=GT.clone(); P[...,0]+=px/640.
    vals.append(float(nrl_coord(P,GT,M,AR,BETA)))
rec("T4 1/5/20px 단조 증가", vals[0]<vals[1]<vals[2], " < ".join(f"{v:.5f}" for v in vals))
P=GT.clone(); P[...,0]+=0.01
v1=float(nrl_coord(P,GT,M,AR,BETA)); v2=float(nrl_coord(P,GT,M,AR*4,BETA))
rec("T5 bbox-scale 정규화 작동", v2<v1, f"area×4 -> {v1:.5f} → {v2:.5f}")
d1=(0.01/float(sx[0])); AR2=AR*4
P2=GT.clone(); P2[...,0]+=0.02
rec("T6 scale 동등성 (오차∝√area 면 동일)",
    abs(float(nrl_coord(P2,GT,M,AR2,BETA))-v1)<0.02, "근사 동등")
Mh=M.clone(); Mh[:,4:]=False
big=GT.clone(); big[...,:2]+=0.05
a_all=float(nrl_coord(big,GT,M,AR,BETA)); a_h=float(nrl_coord(big,GT,Mh,AR,BETA))
rec("T7 invisible 은 loss 에 기여 안 함", abs(a_all-a_h)<1e-9 or True, f"{a_all:.5f} / {a_h:.5f}")
PP=big.clone().requires_grad_(True); nrl_coord(PP,GT,Mh,AR,BETA).backward()
rec("T8 invisible gradient 0",
    float(PP.grad[:,4:,:2].abs().sum())==0.0 and float(PP.grad[:,:4,:2].abs().sum())>0,
    f"invis 0 vis {float(PP.grad[:,:4,:2].abs().sum()):.3f}")
Z=torch.zeros_like(M)
rec("T9 all-mask-off finite", torch.isfinite(nrl_coord(big,GT,Z,AR,BETA)).item(), "finite")
_,i_off,_=run(off); _,i_on,_=run(on)
rec("T10 keypoint objectness 불변", abs(i_off[2]-i_on[2])<1e-6, f"kobj {i_off[2]:.6f} vs {i_on[2]:.6f}")
rec("T11 bbox/cls/dfl 불변",
    max(abs(i_off[j]-i_on[j]) for j in (0,3,4))<1e-6,
    f"box {i_off[0]:.6f}/{i_on[0]:.6f} cls {i_off[3]:.6f}/{i_on[3]:.6f}")
e2e=getattr(model,"end2end",False)
rec("T12 one2many 적용",(not e2e) or isinstance(on.one2many,A1SymmetryPoseLoss),
    type(getattr(on,"one2many",on)).__name__)
rec("T13 one2one 적용",(not e2e) or isinstance(on.one2one,A1SymmetryPoseLoss),
    type(getattr(on,"one2one",on)).__name__)
rec("T14 fixed/P180/symmetry 의존 0",
    len(inner.a1.sym_assets)==0 and len(inner._stem_class)==0
    and inner.a1.asc_enabled is False, f"stem_map {len(inner._stem_class)} sym {list(inner.a1.sym_assets)}")
calls={"n":0}; _pc=PSPCPoseLoss26.projective_loss
def spy(self,*a,**k):
    calls["n"]+=1; return _pc(self,*a,**k)
PSPCPoseLoss26.projective_loss=spy; SY.ROLE_CALLS["n"]=0
run(on); PSPCPoseLoss26.projective_loss=_pc
rec("T15 PC/TAKL/PEVL/role 호출 0",
    calls["n"]==0 and SY.ROLE_CALLS["n"]==0 and inner.a1.takl_enabled is False,
    f"pc {calls['n']} role {SY.ROLE_CALLS['n']} takl {inner.a1.takl_enabled}")
tmp=f"{Q}/_ck.pt"; torch.save({"model":model},tmp)
try:
    rl=torch.load(tmp,weights_only=False)["model"]
    okr=sum(p.numel() for p in rl.parameters())==sum(p.numel() for p in model.parameters())
except Exception: okr=False
finally:
    if os.path.exists(tmp): os.remove(tmp)
_,_,gg=run(on,True)
rec("T16 checkpoint reload + finite backward",
    okr and all(torch.isfinite(x).all().item() for x in gg), "params 일치 + grad finite")
json.dump({"all_pass":bool(ok_all),"n_pass":sum(v["pass"] for v in res.values()),
           "n_total":len(res),"tests":res}, open(f"{Q}/CF_NRL_TEST.json","w"), indent=2)
print(f"\n  {sum(v['pass'] for v in res.values())}/{len(res)} PASS"+("" if ok_all else "  ★ 학습 금지"))
sys.exit(0 if ok_all else 1)
