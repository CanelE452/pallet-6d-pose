"""camera-facing synthetic 평가 — CF val 133, 모든 arm 동일 경로."""
import argparse, json, os, sys, collections
import numpy as np, torch
ROOT="/home/minjae/Documents/github/pallet-pose"; sys.path.insert(0, ROOT)
Q=os.path.dirname(os.path.abspath(__file__))
DATA=f"{ROOT}/challenge/yolo_pose_one_model/datasets/v1_cf_matched10k/data.yaml"
ap=argparse.ArgumentParser(); ap.add_argument("--weights",required=True)
ap.add_argument("--tag",required=True); ap.add_argument("--data",default=DATA)
A=ap.parse_args()
from ultralytics import YOLO
from ultralytics.utils.loss import E2ELoss
from ultralytics.utils.ops import xyxy2xywh
from ultralytics.data.build import build_yolo_dataset, build_dataloader
from ultralytics.data.utils import check_det_dataset
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG
from pallet_yolo_loss.symmetry import A1SymmetryPoseLoss
DEV="cuda"; ROWS=[]
_o=A1SymmetryPoseLoss.calculate_keypoints_loss
def probe(self, masks, tgi, kpts, bidx, st, tb, pk):
    if masks.any() and getattr(self,"_probe",False):
        sel=self._select_target_keypoints(kpts,bidx,tgi,masks).clone()
        sel[...,:2]/=st.view(1,-1,1,1)
        gt=sel[masks]; pr=pk[masks]
        area=xyxy2xywh((tb/st)[masks])[:,2:].prod(1,keepdim=True)
        sd=st.view(1,-1,1).expand(masks.shape[0],-1,1)[masks]
        mv=(gt[...,2]!=0).float()
        e=((pr[...,:2]-gt[...,:2]).pow(2).sum(-1).sqrt())*sd            # 픽셀
        en=e/area.clamp_min(1e-9).sqrt()                                # 정규화
        img=torch.arange(masks.shape[0],device=masks.device)[:,None].expand_as(masks)[masks]
        files=(self._batch or {}).get("im_file") or []
        for n,i in enumerate(img.tolist()):
            s=os.path.splitext(os.path.basename(files[i]))[0] if i<len(files) else "?"
            ROWS.append((s, (e[n]*mv[n]+(-1)*(1-mv[n])).cpu().numpy().tolist(),
                            (en[n]*mv[n]+(-1)*(1-mv[n])).cpu().numpy().tolist()))
    return _o(self, masks, tgi, kpts, bidx, st, tb, pk)
A1SymmetryPoseLoss.calculate_keypoints_loss=probe
y=YOLO(A.weights,task="pose"); model=y.model.to(DEV).float().train()
for q in model.parameters(): q.requires_grad_(False)
data=check_det_dataset(A.data)
args=get_cfg(DEFAULT_CFG,overrides=dict(task="pose",mode="val",data=A.data,imgsz=640,
  batch=16,workers=4,device=0,seed=0,single_cls=True,rect=False))
model.args=args; model.nc=data["nc"]; model.names=data["names"]
ds=build_yolo_dataset(args,data["val"],16,data,mode="val",rect=False,stride=32)
os.environ["A1_CONFIG"]=""; os.environ["PSPC_CONFIG"]=""
crit=E2ELoss(model,A1SymmetryPoseLoss) if getattr(model,"end2end",False) else A1SymmetryPoseLoss(model)
getattr(crit,"one2many",crit)._probe=True
for b in build_dataloader(ds,16,4,shuffle=False,rank=-1):
    b["img"]=b["img"].to(DEV).float()/255
    for k in ("keypoints","bboxes","cls","batch_idx"): b[k]=b[k].to(DEV)
    with torch.no_grad(): crit(model(b["img"]), b)
per=collections.defaultdict(list); pern=collections.defaultdict(list)
for s,e,en in ROWS: per[s].append(e); pern[s].append(en)
PX=np.array([np.mean(per[s],0) for s in sorted(per)])      # (F,9)
NM=np.array([np.mean(pern[s],0) for s in sorted(pern)])
v=PX[:,:8]>=0; e=PX[:,:8][v]; ne=NM[:,:8][NM[:,:8]>=0]
def q(a,p): return float(np.percentile(a,p)) if len(a) else None
def sub(idx):
    z=PX[:,idx]; m=z>=0
    return (q(z[m],50), q(z[m],90))
out={"tag":A.tag,"weights":A.weights,"n_frames":int(PX.shape[0]),
     "corner_median":q(e,50),"corner_p90":q(e,90),
     "corner_norm_median":q(ne,50),"corner_norm_p90":q(ne,90),
     "gross20":float((e>20).mean()),"gross40":float((e>40).mean()),
     "bottom_median":sub([2,3,6,7])[0],"bottom_p90":sub([2,3,6,7])[1],
     "front_p90":sub([0,1,2,3])[1],"rear_p90":sub([4,5,6,7])[1],
     "per_keypoint":{str(k):{"median":sub([k])[0],"p90":sub([k])[1]} for k in range(9)},
     "channel_collapse":int((PX[:,:8].max(1)>300).sum()),
     "per_frame_corner_px":PX.tolist(),"stems":sorted(per)}
try:
    mv=YOLO(A.weights,task="pose").val(data=A.data,imgsz=640,batch=16,device=0,
                                       verbose=False,plots=False,save_json=False)
    out["pose_map50_95"]=float(mv.pose.map); out["pose_map50"]=float(mv.pose.map50)
except Exception as ex:
    out["pose_map50_95"]=None; out["val_error"]=str(ex)
json.dump(out, open(f"{Q}/SYNTH_{A.tag}.json","w"))
print(f"{A.tag:10} mAP {out['pose_map50_95']}  med {out['corner_median']:.2f} "
      f"p90 {out['corner_p90']:.2f} gross20 {out['gross20']:.4f} bottom_p90 {out['bottom_p90']:.2f}")
