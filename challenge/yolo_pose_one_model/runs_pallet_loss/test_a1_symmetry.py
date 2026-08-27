"""A1 — symmetry-aware keypoint objective. 13 테스트. 전부 PASS 전에는 학습 금지.

★ 이 테스트는 CASE(전부 SYM / 혼재)를 가정하지 않는다. 두 분기를 각각 검사한다.
★ A2 projective term 이 A1 경로에 새어들지 않음을 실증한다(T5).
"""
from __future__ import annotations
import json, os, sys, copy
import numpy as np, torch

ROOT="/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, ROOT)
R=os.path.join(ROOT,"challenge/yolo_pose_one_model/runs_pallet_loss")
DATA=os.path.join(ROOT,"challenge/yolo_pose_one_model/datasets/v1_fixed_matched10k/data.yaml")
INIT=os.path.join(ROOT,"challenge/yolo_pose_one_model/runs_fixed/"
                       "V1_FIXED_MATCHED10K_60EP_SEED42_UBUNTU/weights/last.pt")
res={}; ok_all=True
def rec(name, ok, detail=""):
    global ok_all
    res[name]={"pass":bool(ok),"detail":detail}; ok_all &= bool(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {name:46} {detail}", flush=True)

from ultralytics import YOLO
from ultralytics.utils.loss import E2ELoss, PoseLoss26, KeypointLoss
from ultralytics.data.build import build_yolo_dataset, build_dataloader
from ultralytics.data.utils import check_det_dataset
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG
from pallet_yolo_loss.symmetry import A1SymmetryPoseLoss, A1Config, per_instance_kpt_loss
from pallet_yolo_loss.loss import PSPCPoseLoss26

torch.manual_seed(0)
DEV="cuda" if torch.cuda.is_available() else "cpu"
P180=tuple(json.load(open(os.path.join(R,"FIXED_P180_PERMUTATIONS.json")))["P180"])
SMAP=os.path.join(R,"STEM_ASSET_MAP.json")
ASSETS=json.load(open(os.path.join(R,"SYMMETRY_MANIFEST.json")))["assets"]

# ---- T1/T2 순열 성질 (contract 무관) --------------------------------------
rec("T1 P180 involution+bijection",
    sorted(P180)==list(range(8)) and all(P180[P180[i]]==i for i in range(8)),
    f"{P180}")
g=torch.arange(27.).view(1,9,3)
perm=list(P180)+[8]
rec("T2 GT permutation is self-inverse",
    torch.equal(g[:,perm,:][:,perm,:], g), "P180(P180(GT))==GT")

# ---- 배치 확보 -------------------------------------------------------------
y=YOLO(INIT, task="pose")
model=y.model.to(DEV).float().train()
for q in model.parameters(): q.requires_grad_(True)
data=check_det_dataset(DATA)
args=get_cfg(DEFAULT_CFG, overrides=dict(
    task="pose", mode="train", data=DATA, imgsz=640, batch=8, epochs=1,
    workers=2, device=0, seed=0, single_cls=True, mosaic=0.3, scale=0.25,
    hsv_s=0.5, hsv_v=0.35, fliplr=0.0, flipud=0.0, cos_lr=True, close_mosaic=10))
model.args=args; model.nc=data["nc"]; model.names=data["names"]
ds=build_yolo_dataset(args, data["train"], 8, data, mode="train", rect=False, stride=32)
dl=build_dataloader(ds, 8, 2, shuffle=False, rank=-1)
batch=next(iter(dl))
batch["img"]=batch["img"].to(DEV).float()/255
for k in ("keypoints","bboxes","cls","batch_idx"):
    if k in batch: batch[k]=batch[k].to(DEV)
def fresh_preds():
    model.zero_grad(set_to_none=True); return model(batch["img"])

# ---- T3 per-instance 수식이 원본과 같은가 ----------------------------------
kl=KeypointLoss(sigmas=torch.full((9,), 1/9., device=DEV))
pp=torch.rand(11,9,3,device=DEV); gg=torch.rand(11,9,3,device=DEV)
mm=(torch.rand(11,9,device=DEV)>0.3).float(); mm[:,0]=1
ar=torch.rand(11,1,device=DEV)+0.5
d=abs(float(per_instance_kpt_loss(kl,pp,gg,mm,ar).mean())-float(kl(pp,gg,mm,ar)))
rec("T3 per-instance mean == KeypointLoss", d<1e-12, f"|diff| {d:.3e}")

# ---- criterion 만들기 -------------------------------------------------------
def cfgfile(name, **kw):
    p=os.path.join(R,f"_a1_{name}.json")
    base=dict(enabled=True, mode="exact_min", lambda_role=0.0, margin=0.0,
              p180=list(P180), centroid_index=8, sym_assets=[], asym_assets=[],
              stem_asset_map=SMAP, role_ramp=[5,20])
    base.update(kw); json.dump(base, open(p,"w")); return p
def make(kind, a1cfg=None):
    os.environ["A1_CONFIG"]=a1cfg or ""
    os.environ["PSPC_CONFIG"]=""
    torch.manual_seed(0)
    return E2ELoss(model, kind) if getattr(model,"end2end",False) else kind(model)
def run(crit, need_grad=False):
    torch.manual_seed(0); p=fresh_preds(); b=copy.deepcopy(batch)
    l,items=crit(p,b)
    if need_grad:
        l.sum().backward()
        return float(l.sum()), items.detach().cpu().numpy(), \
               [q.grad.detach().clone() for q in model.parameters() if q.grad is not None]
    return float(l.sum()), items.detach().cpu().numpy(), None

OFF=cfgfile("off", enabled=False)
std=make(PoseLoss26); a1off=make(A1SymmetryPoseLoss, OFF)
ls,is_,gs=run(std,True); lc,ic,gc=run(a1off,True)
dl_=abs(ls-lc); di=float(np.abs(is_-ic).max())
rec("T4 disabled == PoseLoss26 (parity)", dl_<1e-6 and di<1e-6,
    f"total {dl_:.3e} items {di:.3e}")
_,_,gs2=run(std,True)
noise=max((a-b).abs().max().item() for a,b in zip(gs,gs2))
dg=max((a-b).abs().max().item() for a,b in zip(gs,gc))
rec("T4b gradient parity", dg<=max(noise,1e-6)*1.5, f"dgrad {dg:.3e} noise {noise:.3e}")

# ---- T5 A2 projective term 이 A1 에 새어드는가 -----------------------------
ALL=cfgfile("all_sym", sym_assets=list(ASSETS.keys()))
a1=make(A1SymmetryPoseLoss, ALL)
inner=getattr(a1,"one2many",a1)
calls={"n":0}
orig=PSPCPoseLoss26.projective_loss
def spy(self,*a,**k):
    calls["n"]+=1; return orig(self,*a,**k)
PSPCPoseLoss26.projective_loss=spy
run(a1)
PSPCPoseLoss26.projective_loss=orig
rec("T5 PC term isolated from A1",
    calls["n"]==0 and inner.pspc.enabled is False and inner.pspc.lambda_pc==0.0,
    f"projective_loss calls={calls['n']} enabled={inner.pspc.enabled} lpc={inner.pspc.lambda_pc}")

# ---- 합성 텐서로 분기 검증 --------------------------------------------------
torch.manual_seed(1)
G=torch.rand(6,9,3,device=DEV); G[...,2]=1.0
A=torch.ones(6,1,device=DEV)
def dpair(pred, gt):
    m=gt[...,2]!=0
    g2=gt[:,perm,:]; m2=m[:,perm]
    return (per_instance_kpt_loss(kl,pred,gt,m,A),
            per_instance_kpt_loss(kl,pred,g2,m2,A))
did,d180=dpair(G.clone(), G)
rec("T6 pred==GT -> d_id 0, d_180 separated",
    float(did.max())<1e-9 and float(d180.min())>1e-3,
    f"d_id {float(did.max()):.2e}  d_180 {float(d180.mean()):.4f}")
did2,d1802=dpair(G[:,perm,:].clone(), G)
rec("T7 pred==P180(GT) -> d_180 0",
    float(d1802.max())<1e-9 and float(did2.min())>1e-3,
    f"d_180 {float(d1802.max()):.2e}  d_id {float(did2.mean()):.4f}")
rec("T8 SYM min() invariant to GT ordering",
    abs(float(torch.minimum(did,d180).mean())-float(torch.minimum(did2,d1802).mean()))<1e-9,
    "min(d_id,d_180) 동일")

# ---- T9 role term ------------------------------------------------------------
m_=0.05
r=torch.relu(m_+did2-d1802)          # d_id 큼 -> 활성
r0=torch.relu(0.0+did-d180)          # d_id 0, d_180 큼 -> 0
rec("T9 role margin hinge", float(r.min())>0 and float(r0.max())<1e-9,
    f"active {float(r.mean()):.4f}  inactive {float(r0.max()):.2e}")

# ---- T10 ramp ---------------------------------------------------------------
c=A1SymmetryPoseLoss.__new__(A1SymmetryPoseLoss); c.a1=A1Config(role_ramp=(5,20))
vals=[]
for e in (0,4,5,12,12.5,20,59):
    c._epoch=e; vals.append(round(c._ramp(),4))
rec("T10 role ramp schedule", vals[0]==0 and vals[1]==0 and vals[2]==0
    and abs(vals[4]-0.5)<1e-9 and vals[5]==1.0 and vals[6]==1.0, f"{vals}")

# ---- T11 validity mask 가 순열과 함께 움직이는가 -----------------------------
G2=G.clone(); G2[:,2,2]=0.0; G2[:,5,2]=0.0
mA=(G2[...,2]!=0); mB=mA[:,perm]
exp=[perm.index(2), perm.index(5)]
rec("T11 mask permutes with points",
    (~mB[0]).nonzero().flatten().tolist()==sorted(exp), f"invalid@{sorted(exp)}")

# ---- T12 class 배정이 im_file 에서 정확한가 ----------------------------------
smap=json.load(open(SMAP))
MIX=cfgfile("mixed", sym_assets=["scene.usd","scene_1.usd"],
            asym_assets=["woodpallet_block_jtoastie_ccby.glb","eur_pallet_bk_cc0.glb"])
a1m=make(A1SymmetryPoseLoss, MIX); im=getattr(a1m,"one2many",a1m)
im._batch=batch
stems=[os.path.splitext(os.path.basename(f))[0] for f in batch["im_file"]]
want=[1 if smap[s] in ("scene.usd","scene_1.usd") else 2 for s in stems]
mk=torch.zeros(len(stems), 3, dtype=torch.bool, device=DEV); mk[:,0]=True
got=im._instance_class(mk).tolist()
rec("T12 class from im_file", got==want, f"got={got} want={want}")

# ---- T13 E2E 두 경로 + 실제 경로에서 GT 는 미분 대상이 아니다 ----------------
# 합성 텐서로 GT.requires_grad 를 켜서 재면 "GT 도 grad 를 받는다" 는 자명한 사실만
# 재게 된다.  주장은 **실제 학습 경로**에서 성립해야 하므로 거기서 잰다.
both = (not getattr(model,"end2end",False)) or (
    isinstance(a1.one2many, A1SymmetryPoseLoss) and isinstance(a1.one2one, A1SymmetryPoseLoss))
torch.manual_seed(0); pr=fresh_preds(); bb=copy.deepcopy(batch)
gt_req = bool(bb["keypoints"].requires_grad)
lo,_ = a1(pr, bb); lo.sum().backward()
npar = sum(1 for q in model.parameters() if q.grad is not None and q.grad.abs().sum()>0)
gt_grad = bb["keypoints"].grad
rec("T13 E2E both paths + GT not differentiable",
    both and npar>0 and gt_req is False and gt_grad is None,
    f"e2e={getattr(model,'end2end',False)} both={both} params_with_grad={npar} "
    f"GT.requires_grad={gt_req} GT.grad={'None' if gt_grad is None else 'SET'}")

json.dump({"all_pass":bool(ok_all),"tests":res}, open(os.path.join(R,"A1_TEST_RESULTS.json"),"w"), indent=2)
print("\n  ALL PASS" if ok_all else "\n  ★ FAIL 있음 — 학습 금지")
sys.exit(0 if ok_all else 1)
