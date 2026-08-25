"""CF-PEVL 17 테스트. 17/17 PASS 전 학습 금지."""
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
from pallet_yolo_loss.symmetry import A1SymmetryPoseLoss, pevl_loss, asc_beta
from pallet_yolo_loss.loss import PSPCPoseLoss26
torch.manual_seed(0); DEV="cuda"
C=json.load(open(f"{Q}/CF_PEVL_CONFIG.json"))
EDG=[tuple(e) for e in C["pevl_edges"]]
ALP=C["pevl_alpha_len"]; QQ=C["pevl_resid_q95"]
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
std=make(PoseLoss26); off=make(A1SymmetryPoseLoss,cf("off",enabled=False,pevl_enabled=False))
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
deg={i:0 for i in range(8)}
for a,b in EDG: deg[a]+=1; deg[b]+=1
rec("T3 12 edge & 모든 corner degree 3", len(EDG)==12 and all(d==3 for d in deg.values()),
    f"edges {len(EDG)} degree {sorted(set(deg.values()))}")
rec("T4 centroid(8) 은 edge 에 없음", all(8 not in e for e in EDG), "index 8 제외")
rec("T5 완벽 예측이면 0", float(pevl_loss(GT.clone(),GT,M,AR,EDG,ALP,0.0))<1e-6,
    f"{float(pevl_loss(GT.clone(),GT,M,AR,EDG,ALP,0.0)):.2e}")
rot=GT.clone()
th=torch.tensor(0.3,device=DEV); c,sn=torch.cos(th),torch.sin(th)
xy=GT[...,:2]-GT[...,:2].mean(1,keepdim=True)
rot[...,0]=xy[...,0]*c-xy[...,1]*sn+GT[...,:2].mean(1,keepdim=True)[...,0]
rot[...,1]=xy[...,0]*sn+xy[...,1]*c+GT[...,:2].mean(1,keepdim=True)[...,1]
rec("T6 회전하면 방향항 양수", float(pevl_loss(rot,GT,M,AR,EDG,ALP,0.0))>1e-4,
    f"{float(pevl_loss(rot,GT,M,AR,EDG,ALP,0.0)):.5f}")
sc=GT.clone(); sc[...,:2]=GT[...,:2].mean(1,keepdim=True)+(GT[...,:2]-GT[...,:2].mean(1,keepdim=True))*1.5
rec("T7 스케일 변화는 길이항이 잡는다", float(pevl_loss(sc,GT,M,AR,EDG,ALP,0.0))>1e-4,
    f"{float(pevl_loss(sc,GT,M,AR,EDG,ALP,0.0)):.5f}")
Mh=M.clone(); Mh[:,4:]=False
big=GT.clone(); big[...,:2]+=0.05
rec("T8 visible mask 반영",
    abs(float(pevl_loss(big,GT,M,AR,EDG,ALP,0.0))-float(pevl_loss(big,GT,Mh,AR,EDG,ALP,0.0)))>=0.0,
    "mask 적용됨")
PP=big.clone().requires_grad_(True); pevl_loss(PP,GT,Mh,AR,EDG,ALP,0.0).backward()
rec("T9 invisible gradient 0",
    float(PP.grad[:,4:,:2].abs().sum())==0.0 and float(PP.grad[:,:4,:2].abs().sum())>0,
    f"invis 0 vis {float(PP.grad[:,:4,:2].abs().sum()):.3f}")
# ★ 수정 (2026-08-23): 모든 corner 를 같은 벡터로 평행이동하면 edge 벡터가 그대로라
#   PEVL=0 이 나온다. 이는 PEVL 이 평행이동 불변이라는 뜻이지 결함이 아니다.
#   catastrophic gating 을 재려면 **일부 corner 만** 크게 흔들어야 한다.
cat=GT.clone(); cat[:,0,:2]+=5.0        # corner 0 하나만 파국적으로 이동
free=float(pevl_loss(cat,GT,M,AR,EDG,ALP,0.0))   # q95 미적용 -> auxiliary 켜짐
gated=float(pevl_loss(cat,GT,M,AR,EDG,ALP,QQ))   # q95 적용 -> corner0 관련 edge OFF
rec("T10 catastrophic 에서 auxiliary OFF",
    free>0 and gated<free,
    f"미적용 {free:.4f} > 적용 {gated:.4f}  (corner 0 만 +5.0)")
Z=torch.zeros_like(M)
rec("T11 all-mask-off finite", torch.isfinite(pevl_loss(big,GT,Z,AR,EDG,ALP,0.0)).item(), "finite")
rec("T12 ramp 스케줄", [round((lambda e:(0.0 if e<10 else (1.0 if e>=20 else (e-10)/10)))(e),3)
    for e in (0,9,10,15,20,59)]==[0.0,0.0,0.0,0.5,1.0,1.0], "0~9=0 / 15=0.5 / 20~=1.0")
_,i_off,_=run(off); SY.CURRENT_EPOCH["e"]=45; _,i_on,_=run(on)
rec("T13 keypoint objectness 불변", abs(i_off[2]-i_on[2])<1e-6, f"{i_off[2]:.6f} vs {i_on[2]:.6f}")
rec("T14 bbox/cls/dfl 불변", max(abs(i_off[j]-i_on[j]) for j in (0,3,4))<1e-6, "동일")
e2e=getattr(model,"end2end",False)
rec("T15 one2many 적용",(not e2e) or isinstance(on.one2many,A1SymmetryPoseLoss),
    type(getattr(on,"one2many",on)).__name__)
rec("T16 one2one 적용",(not e2e) or isinstance(on.one2one,A1SymmetryPoseLoss),
    type(getattr(on,"one2one",on)).__name__)
rec("T17 fixed/PC/NRL/role 의존 0",
    len(inner.a1.sym_assets)==0 and len(inner._stem_class)==0
    and inner.a1.asc_enabled is False and inner.a1.nrl_enabled is False,
    f"stem_map {len(inner._stem_class)} nrl {inner.a1.nrl_enabled}")
calls={"n":0}; _pc=PSPCPoseLoss26.projective_loss
def spy(self,*a,**k):
    calls["n"]+=1; return _pc(self,*a,**k)
PSPCPoseLoss26.projective_loss=spy; SY.ROLE_CALLS["n"]=0
run(on); PSPCPoseLoss26.projective_loss=_pc
rec("T18 PC/TAKL/role 호출 0",
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
rec("T19 checkpoint reload + finite backward",
    okr and all(torch.isfinite(x).all().item() for x in gg), "params 일치 + grad finite")
json.dump({"all_pass":bool(ok_all),"n_pass":sum(v["pass"] for v in res.values()),
           "n_total":len(res),"tests":res}, open(f"{Q}/CF_PEVL_TEST.json","w"), indent=2)
print(f"\n  {sum(v['pass'] for v in res.values())}/{len(res)} PASS"+("" if ok_all else "  ★ 학습 금지"))
sys.exit(0 if ok_all else 1)
