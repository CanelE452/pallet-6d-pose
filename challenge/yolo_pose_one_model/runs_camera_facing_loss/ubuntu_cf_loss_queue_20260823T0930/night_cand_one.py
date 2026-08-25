"""임의 checkpoint 의 NIGHT candidate 거동. 기존 계약 그대로 (IoU>=0.5, conf=0.001, pad100)."""
import argparse, json, os, sys, collections
import numpy as np, cv2
ROOT="/home/minjae/Documents/github/pallet-pose"; sys.path.insert(0,ROOT)
Q=os.path.dirname(os.path.abspath(__file__))
ap=argparse.ArgumentParser(); ap.add_argument("--weights",required=True); ap.add_argument("--tag",required=True)
A=ap.parse_args()
from ultralytics import YOLO
MANI=("/home/minjae/pallet_worker_transfer_20260821T105141Z/REAL_GT_QA_20260821T133405Z/"
      "REVIEWED_CLEAN_REALDEV_V2_MANIFEST.json")
PAD,IOU_T,CONF=100,0.5,0.001; NIGHT={"eval_night08","eval_night09"}
man=json.load(open(MANI))
night=[(it["frame_id"],os.path.join(ROOT,it["label"]),os.path.join(ROOT,it["image"]))
       for it in man["items"] if it.get("set") in NIGHT]
def gtb(jp):
    g=np.array(json.load(open(jp))["objects"][0]["projected_cuboid"],dtype=float)[:8]
    return [g[:,0].min(),g[:,1].min(),g[:,0].max(),g[:,1].max()]
def iou(b,g):
    xx=max(0,min(b[2],g[2])-max(b[0],g[0])); yy=max(0,min(b[3],g[3])-max(b[1],g[1])); i=xx*yy
    return i/max((b[2]-b[0])*(b[3]-b[1])+(g[2]-g[0])*(g[3]-g[1])-i,1e-9)
m=YOLO(A.weights,task="pose"); rows=[]
for fid,jp,ip in night:
    im=cv2.imread(ip); p=cv2.copyMakeBorder(im,PAD,PAD,PAD,PAD,cv2.BORDER_REFLECT_101)
    r=m.predict(p,conf=CONF,imgsz=640,device=0,verbose=False)[0]; g=gtb(jp)
    if r.boxes is None or not len(r.boxes): rows.append({"n":0}); continue
    cf=r.boxes.conf.cpu().numpy(); bx=r.boxes.xyxy.cpu().numpy()-PAD
    o=np.argsort(-cf); iv=np.array([iou(bx[i],g) for i in o]); ok=iv>=IOU_T
    d={"n":len(cf),"top1":bool(ok[0]),"any":bool(ok.any()),
       "rank":(int(np.argmax(ok))+1 if ok.any() else None),"nwrong":int((~ok).sum())}
    if ok.any():
        d["margin"]=float(cf[o][ok].max()-(cf[o][~ok].max() if (~ok).any() else 0.0))
    rows.append(d)
n=len(rows); mg=[r["margin"] for r in rows if "margin" in r]
rk=collections.Counter("rank1" if r.get("rank")==1 else "rank2" if r.get("rank")==2
                       else "rank3+" if r.get("rank") else "absent" for r in rows)
out={"model":A.tag,"n":n,"any_det":sum(r['n']>0 for r in rows)/n,
     "any_cbox":sum(r.get('any',False) for r in rows)/n,
     "top1_cbox":sum(r.get('top1',False) for r in rows)/n,
     "cand_per_frame":float(np.mean([r['n'] for r in rows])),
     "wrong_per_frame":float(np.mean([r.get('nwrong',0) for r in rows])),
     "wrong_present_frac":float(np.mean([r.get('nwrong',0)>0 for r in rows])),
     "margin_median":float(np.median(mg)) if mg else None,
     "margin_positive_frac":float(np.mean(np.array(mg)>0)) if mg else None,
     "rank_hist":dict(rk)}
json.dump(out, open(f"{Q}/NIGHT_CAND_{A.tag}.json","w"), indent=2)
print(f"{A.tag}: any-cbox {out['any_cbox']:.3f} top1 {out['top1_cbox']:.3f} "
      f"cand/frame {out['cand_per_frame']:.2f} margin {out['margin_median']}")
