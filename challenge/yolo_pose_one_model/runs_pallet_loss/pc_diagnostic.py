"""PC loss 가 **의도한 일**을 했는지 — projective residual 을 직접 잰다.

pose 붕괴(트랩 미탈출)와 projective 효과를 분리한다.
"""
from __future__ import annotations
import csv, glob, json, os, sys
import numpy as np, cv2
ROOT="/home/minjae/Documents/github/pallet-pose"
# ★ A2 checkpoint 는 PSPCPoseModel 을 pickle 하고 있어 프로젝트 패키지가
#   import path 에 없으면 로드되지 않는다 (체크포인트 이식성 제약).
sys.path.insert(0, ROOT)
import pallet_yolo_loss  # noqa: F401
from ultralytics import YOLO
D=os.path.join(ROOT,"challenge/yolo_pose_one_model"); R=os.path.join(D,"runs_pallet_loss")
PAIRS=[tuple(x) for x in json.load(open(os.path.join(R,"CUBOID_DIAGONAL_PAIRS.json")))["pairs"]]
RUNS={"A0":os.path.join(D,"runs_fixed/V1_FIXED_MATCHED10K_60EP_SEED42_UBUNTU/weights/last.pt"),
      "A2":os.path.join(R,"PSPC_A2_PC_ONLY_V1MATCHED10K_60EP_SEED42/weights/last.pt")}
imgs=sorted(glob.glob(os.path.join(D,"datasets/v1_fixed_matched10k/images/val/*.png")))
rows=[]
for name,ck in RUNS.items():
    m=YOLO(ck,task="pose")
    proj_d=[]; corn=[]; ident=0; n=0
    for p in imgs:
        lp=p.replace("/images/","/labels/").replace(".png",".txt")
        if not os.path.exists(lp): continue
        f=open(lp).read().split()
        kp=np.array([[float(f[5+3*i]),float(f[6+3*i]),float(f[7+3*i])] for i in range(9)])
        r=m.predict(p,imgsz=640,conf=0.25,verbose=False)[0]
        if r.keypoints is None or not len(r.boxes): continue
        H,W=cv2.imread(p).shape[:2]
        pr=r.keypoints.xy.cpu().numpy()[0]
        gt=np.stack([kp[:,0]*W,kp[:,1]*H],1); vis=kp[:,2]>0
        if vis[:8].sum()<4 or not vis[8]: continue
        n+=1
        diag=np.hypot(gt[vis][:,0].ptp(),gt[vis][:,1].ptp()) or 1.0
        # ★ 학습에 쓴 것과 같은 정의: GT centroid 를 pred 대각선에 대어 잰다
        ds=[]
        for i,j in PAIRS:
            if not (vis[i] and vis[j]): continue
            a,b=pr[i],pr[j]; ab=b-a; L=np.linalg.norm(ab)
            if L<1e-3: continue
            c=gt[8]
            ds.append(abs(ab[0]*(c[1]-a[1])-ab[1]*(c[0]-a[0]))/L/diag)
        if ds: proj_d.append(float(np.mean(ds)))
        corn.append(float(np.median(np.linalg.norm(pr[:8][vis[:8]]-gt[:8][vis[:8]],axis=1)/diag)))
    proj_d=np.array(proj_d); corn=np.array(corn)
    rows.append({"run":name,"n_val":n,
      "projective_median":float(np.median(proj_d)),"projective_p90":float(np.percentile(proj_d,90)),
      "projective_p99":float(np.percentile(proj_d,99)),
      "corner_median":float(np.median(corn)),"corner_p90":float(np.percentile(corn,90))})
    print(f"  {name}: n={n}  proj med {rows[-1]['projective_median']:.5f} p90 {rows[-1]['projective_p90']:.5f}"
          f"  corner med {rows[-1]['corner_median']:.5f} p90 {rows[-1]['corner_p90']:.5f}")
with open(os.path.join(R,"PC_DIAGNOSTIC.csv"),"w",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
a0,a2=rows[0],rows[1]
def rel(n,o): return (n-o)/o if o else float("nan")
print(f"\n  projective p90 상대변화 {rel(a2['projective_p90'],a0['projective_p90']):+.1%}")
print(f"  corner     p90 상대변화 {rel(a2['corner_p90'],a0['corner_p90']):+.1%}")
