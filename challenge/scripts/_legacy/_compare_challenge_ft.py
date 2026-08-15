"""crop-aug DOPE truncation 강건성 정성 비교 (baseline vs ft_s1 vs ft_s2).

forklift mp4 프레임에 3개 모델을 추론, keypoint + cuboid wireframe overlay 를
side-by-side 로 비교. cuboid pose 는 annotate_pnp.solve_pose (auto order/dims,
24 cube symmetry + strict invariants) 로 복원해 keypoint 순서 convention 차이에
영향받지 않게 한다.

usage:
  python challenge/scripts/compare_cropaug_truncation.py \
      --rgb_dir data/outside/forklift_raw_20260528_163408/rgb \
      --gt_dir  data/outside/forklift_raw_20260528_163408/gt_manual \
      --out_dir challenge/data/04_results/cropaug_truncation_eval
"""
import argparse, glob, json, os, sys
import cv2, numpy as np, torch
from scipy.ndimage import gaussian_filter

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "Deep_Object_Pose", "common"))
sys.path.insert(0, os.path.join(_REPO, "Deep_Object_Pose", "train"))
sys.path.insert(0, os.path.join(_REPO, "challenge", "scripts"))
from models import DopeNetwork
import annotate_pnp as ap

KP_COLORS = [
    (0,0,255),(0,128,255),(0,255,255),(0,255,0),
    (255,255,0),(255,0,0),(255,0,128),(128,0,255),(255,255,255),
]
EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
MEAN = np.array([0.485,0.456,0.406]); STD = np.array([0.229,0.224,0.225])


def load_model(w, dev):
    m = DopeNetwork(); s = torch.load(w, map_location=dev)
    if any(k.startswith("module.") for k in s):
        s = {k.replace("module.",""): v for k,v in s.items()}
    m.load_state_dict(s); m.to(dev); m.eval(); return m


def infer_belief(m, img_bgr, dev):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    r = cv2.resize(rgb, (448,448)); n = (r.astype(np.float32)/255 - MEAN)/STD
    t = torch.from_numpy(n.transpose(2,0,1)).float().unsqueeze(0).to(dev)
    with torch.no_grad():
        ob,_ = m(t)
    return ob[-1][0].cpu().numpy()


def extract_kps(bel, thr=0.3):
    OFF=0.4395; RAN=5; out=[]
    for i in range(bel.shape[0]):
        b=bel[i]
        if b.max()<thr: out.append(None); continue
        sm=gaussian_filter(b,sigma=2); p=1
        pl=np.zeros_like(sm); pl[p:,:]=sm[:-p,:]
        pr=np.zeros_like(sm); pr[:-p,:]=sm[p:,:]
        pu=np.zeros_like(sm); pu[:,p:]=sm[:,:-p]
        pd=np.zeros_like(sm); pd[:,:-p]=sm[:,p:]
        pk=(sm>=pl)&(sm>=pr)&(sm>=pu)&(sm>=pd)&(sm>thr)
        ys,xs=np.nonzero(pk)
        if len(xs)==0: out.append(None); continue
        vals=[b[y,x] for y,x in zip(ys,xs)]; bi=np.argmax(vals)
        px,py=int(xs[bi]),int(ys[bi])
        y0=max(0,py-RAN);y1=min(b.shape[0],py+RAN+1);x0=max(0,px-RAN);x1=min(b.shape[1],px+RAN+1)
        patch=b[y0:y1,x0:x1]
        if patch.sum()>0:
            xg,yg=np.meshgrid(np.arange(x0,x1),np.arange(y0,y1))
            wx=np.average(xg,weights=patch)+OFF; wy=np.average(yg,weights=patch)+OFF
        else: wx,wy=float(px),float(py)
        out.append((wx,wy,float(b.max())))
    return out


def kps_to_orig(kps, bel, W, H):
    sx,sy = W/bel.shape[2], H/bel.shape[1]
    res=[]
    for kp in kps:
        if kp is None: res.append(None)
        else: res.append([kp[0]*sx, kp[1]*sy])
    return res


def draw_panel(img, kps_orig, K, label):
    vis = img.copy()
    ndet = sum(1 for k in kps_orig if k is not None)
    # cuboid via auto-order PnP
    pose = None
    if ndet >= 4:
        try:
            pose = ap.solve_pose(kps_orig, K, img_shape=img.shape)
        except Exception:
            pose = None
    if pose is not None:
        proj = pose["projected_all"]
        for a,b in EDGES:
            pa,pb = proj[a],proj[b]
            if pa[0]==-1 or pb[0]==-1: continue
            cv2.line(vis,(int(pa[0]),int(pa[1])),(int(pb[0]),int(pb[1])),(0,255,255),2)
    # keypoints
    for i,k in enumerate(kps_orig):
        if k is None: continue
        pt=(int(k[0]),int(k[1]))
        cv2.circle(vis,pt,5,KP_COLORS[i],-1); cv2.circle(vis,pt,6,(0,0,0),1)
    reproj = pose["reproj_error_px"] if pose else -1
    pnp_ok = pose is not None
    txt = f"{label}: {ndet}/9 kp  PnP {'OK' if pnp_ok else 'X'}"
    if pnp_ok: txt += f" rp{reproj:.0f}"
    cv2.rectangle(vis,(0,0),(vis.shape[1],22),(0,0,0),-1)
    cv2.putText(vis,txt,(6,16),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
    return vis, ndet, pnp_ok, reproj


def gt_trunc_count(gt_json, W, H):
    """GT projected_cuboid 중 화면 밖 corner 수 (truncation 심도)."""
    d=json.load(open(gt_json)); pc=d["objects"][0]["projected_cuboid"]
    n=0
    for u,v in pc[:8]:
        if u<0 or v<0 or u>=W or v>=H: n+=1
    return n


def main():
    ap_=argparse.ArgumentParser()
    ap_.add_argument("--rgb_dir",required=True)
    ap_.add_argument("--gt_dir",default=None)
    ap_.add_argument("--out_dir",required=True)
    ap_.add_argument("--fx",type=float,default=614.18)
    ap_.add_argument("--fy",type=float,default=614.31)
    ap_.add_argument("--cx",type=float,default=329.28)
    ap_.add_argument("--cy",type=float,default=234.53)
    ap_.add_argument("--threshold",type=float,default=0.3)
    ap_.add_argument("--stride",type=int,default=1,help="frame subsample for full-mp4 stats")
    a=ap_.parse_args()
    os.makedirs(a.out_dir,exist_ok=True)
    panels_dir=os.path.join(a.out_dir,"panels"); os.makedirs(panels_dir,exist_ok=True)
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    K=np.array([[a.fx,0,a.cx],[0,a.fy,a.cy],[0,0,1.]])

    models={
        "ft_s2_OLD": os.path.join(_REPO,"weights/dope/dope_cropaug_ft_s2/final_net_epoch_0180.pth"),
        "ft_otftrunc_NEW": os.path.join(_REPO,"weights/challenge_track/challenge_ft_otftrunc/final_net_epoch_0150.pth"),
    }
    M={k:load_model(v,dev) for k,v in models.items()}
    order=["ft_s2_OLD","ft_otftrunc_NEW"]

    pngs=sorted(glob.glob(os.path.join(a.rgb_dir,"*.png")))
    # truncation level per frame from gt_manual (if available)
    stats={k:{"det":[],"pnp":[],"reproj":[]} for k in order}
    # split by trunc level using gt frames
    trunc_rows=[]
    for pf in pngs[::a.stride]:
        base=os.path.splitext(os.path.basename(pf))[0]
        img=cv2.imread(pf); H,W=img.shape[:2]
        gtf=os.path.join(a.gt_dir,base+".json") if a.gt_dir else None
        tl = gt_trunc_count(gtf,W,H) if (gtf and os.path.exists(gtf)) else None
        panels=[]; rowrec={"frame":base,"trunc":tl}
        for k in order:
            bel=infer_belief(M[k],img,dev)
            kps=extract_kps(bel,a.threshold)
            ko=kps_to_orig(kps,bel,W,H)
            vis,ndet,pnp_ok,rp=draw_panel(img,ko,K,k)
            panels.append(vis)
            stats[k]["det"].append(ndet); stats[k]["pnp"].append(int(pnp_ok))
            stats[k]["reproj"].append(rp if pnp_ok else np.nan)
            rowrec[f"{k}_det"]=ndet; rowrec[f"{k}_pnp"]=int(pnp_ok); rowrec[f"{k}_rp"]=round(rp,1) if pnp_ok else -1
        # save side-by-side only for gt frames (representative)
        if tl is not None:
            combo=np.hstack(panels)
            cv2.imwrite(os.path.join(panels_dir,f"t{tl}_{base}.jpg"),combo)
        trunc_rows.append(rowrec)

    # summary
    print("\n==== Per-model full-sequence stats (stride=%d, n=%d) ===="%(a.stride,len(pngs[::a.stride])))
    for k in order:
        det=np.array(stats[k]["det"]); pnp=np.array(stats[k]["pnp"])
        rp=np.array(stats[k]["reproj"]); rpv=rp[~np.isnan(rp)]
        print(f"  {k:9s} det>=8: {100*np.mean(det>=8):5.1f}%  det>=6: {100*np.mean(det>=6):5.1f}%  "
              f"PnP: {100*pnp.mean():5.1f}%  reproj med: {np.median(rpv) if len(rpv) else -1:6.1f}")

    # truncation-level breakdown (gt frames)
    gtrows=[r for r in trunc_rows if r["trunc"] is not None]
    print("\n==== GT-frame truncation breakdown (n=%d) ===="%len(gtrows))
    levels={"clean(0)":lambda t:t==0,"mild(1-2)":lambda t:1<=t<=2,"severe(3+)":lambda t:t>=3}
    hdr="  %-10s %-5s | "%("level","n") + " | ".join(f"{k}(det>=6 / PnP%% / rpMed)" for k in order)
    print(hdr)
    for lname,fn in levels.items():
        rows=[r for r in gtrows if fn(r["trunc"])]
        if not rows: continue
        cells=[]
        for k in order:
            d6=100*np.mean([r[f"{k}_det"]>=6 for r in rows])
            pn=100*np.mean([r[f"{k}_pnp"] for r in rows])
            rps=[r[f"{k}_rp"] for r in rows if r[f"{k}_rp"]>=0]
            rpm=np.median(rps) if rps else -1
            cells.append(f"{d6:4.0f}/{pn:4.0f}/{rpm:5.1f}")
        print(f"  {lname:10s} {len(rows):<5d}| "+" | ".join(cells))

    with open(os.path.join(a.out_dir,"per_frame.json"),"w") as f:
        json.dump(trunc_rows,f,indent=1)
    print(f"\nPanels: {panels_dir}")
    print(f"Per-frame: {os.path.join(a.out_dir,'per_frame.json')}")


if __name__=="__main__":
    main()
