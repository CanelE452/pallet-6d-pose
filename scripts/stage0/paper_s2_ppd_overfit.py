"""PPD 32-frame overfit: L0 / M0 / M1 (matched control)."""
import sys, json, math, pathlib, hashlib, time
import numpy as np, pandas as pd, cv2, torch, torch.nn as nn, torch.nn.functional as F
ROOT=pathlib.Path("/home/minjae/Documents/github/pallet-pose")
for p in ("Deep_Object_Pose/common","challenge/scripts","scripts/data_prep/eval"):
    sys.path.insert(0,str(ROOT/p))
import pallet_graph_geometry as PG, pallet_polarity_disambiguation as PPD
import semantic_axis_initialization as SAI, dimension_guided_graph_pose as DGP
import polarity_aware_line_head as PLH
from models import DopeNetwork

D=ROOT/"data/pallet/results/paper_s2_palletgraph_line_screen"
DATA=ROOT/"data/pallet/training_data/paper_4pallet_mask_v1"
WOUT=ROOT/"weights/paper_s2_ppd_t2_screen"
ALLOWED_ROOT="paper_4pallet_mask_v1"
SEED=1; STEPS=800; BATCH=8; LR=1e-3; WD=1e-4; GRID=PLH.TARGET_GRID
dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN=np.array([0.485,0.456,0.406],np.float32); STD=np.array([0.229,0.224,0.225],np.float32)

def seed_all(s):
    import random; random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def support_maps(R,t,K,dims,size,keep):
    edges=PPD.polarity_edge_classes(dims); vis=PG.visible_edges(R,t,dims)
    proj,dep=PG.project_points(PG.make_corners(*dims)[:8],R,t,K)
    out={c:np.zeros((size[1],size[0]),np.uint8) for c in PLH.CLASS_ORDER}
    for (i,j),cls in edges:
        if dep[i]<=1e-6 or dep[j]<=1e-6 or not vis[(i,j)]: continue
        cl=PG.clip_segment_to_image(proj[i],proj[j],size[0],size[1])
        if cl is None: continue
        for q in PG.sample_along(cl[0],cl[1],pixels_per_sample=1.0):
            x,y=int(round(q[0])),int(round(q[1]))
            if 0<=x<size[0] and 0<=y<size[1] and keep[y,x]: out[cls][y,x]=1
    return out

def load_frames(files):
    fr=[]
    for fn in files:
        assert ALLOWED_ROOT in str(DATA), "training root violation"
        d=json.load(open(DATA/fn)); o=d["objects"][0]; c=d["camera_data"]
        K=np.array([[c["intrinsics"]["fx"],0,c["intrinsics"]["cx"]],
                    [0,c["intrinsics"]["fy"],c["intrinsics"]["cy"]],[0,0,1.]])
        dims=(o["dimensions_m"]["width"],o["dimensions_m"]["depth"],o["dimensions_m"]["height"])
        T=np.asarray(o["pose_transform"],float); R,t=T[:3,:3],T[:3,3]
        img=cv2.imread(str(DATA/fn.replace(".json",".png"))); size=(c["width"],c["height"])
        mask=PLH.decode_mask_rle(o["mask_rle"],(c["height"],c["width"]))
        tg,_=PLH.build_polarity_targets_v2(R,t,K,dims,size,"observed_fragment",image_bgr=img)
        rgb=cv2.cvtColor(cv2.resize(img,(400,400)),cv2.COLOR_BGR2RGB).astype(np.float32)/255.
        x=torch.from_numpy(((rgb-MEAN)/STD).transpose(2,0,1))
        mt=torch.from_numpy(cv2.resize(mask,(GRID,GRID),interpolation=cv2.INTER_NEAREST).astype(np.float32))[None]
        keep=PLH.gradient_association_mask(img)
        sup=support_maps(R,t,K,dims,size,keep)
        uns={"width":sup["top_width"]|sup["base_width"],"depth":sup["top_depth"]|sup["base_depth"],"vertical":sup["vertical"]}
        cands=SAI.semantic_axis_initialization(uns,K,dims,size)["candidates"]
        fr.append({"file":fn,"x":x,"target":torch.from_numpy(tg),"mask":mt,
                   "K":K,"dims":dims,"R":R,"t":t,"size":size,
                   "cands":[c_["R"] for c_ in cands]})
    return fr

class Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=DopeNetwork(numVec=0,numSeg=1)
        st=torch.load(str(ROOT/"weights/paper_s2_stageB/net_epoch_0057.pth"),map_location="cpu",weights_only=True)
        st={k.removeprefix("module."):v for k,v in st.items()}
        self.net.load_state_dict(st,strict=True)
        for p_ in self.net.parameters(): p_.requires_grad_(False)
        self.net.eval(); self.feat=None; self.idx=None; self.ch=None
    def discover(self,sample):
        self.idx,self.ch=PLH.find_high_resolution_feature(self.net.vgg,sample,GRID)
        self.net.vgg[self.idx].register_forward_hook(
            lambda m,i,o: setattr(self,"feat",o))
        return self.idx,self.ch
    @torch.no_grad()
    def forward(self,x):
        self.net(x); assert self.feat.shape[-2:]==(GRID,GRID), self.feat.shape
        return self.feat.detach()

def line_metrics(logits,target,tol=2):
    p=(torch.sigmoid(logits)>=0.5).float(); g=(target>=0.5).float()
    k=2*tol+1
    gd=F.max_pool2d(g,k,1,tol); pd_=F.max_pool2d(p,k,1,tol)
    # recall@tol : GT cells that have a prediction within tol
    # precision@tol: predicted cells that have a GT within tol
    rec=((g*pd_).sum(dim=(0,2,3))/g.sum(dim=(0,2,3)).clamp_min(1))
    pre=((p*gd).sum(dim=(0,2,3))/p.sum(dim=(0,2,3)).clamp_min(1))
    f1=2*pre*rec/(pre+rec).clamp_min(1e-9)
    return rec.cpu().numpy(),pre.cpu().numpy(),f1.cpu().numpy()

def polarity_from_maps(logits,fr):
    """native 100x100 map 에서 bilinear sampling (resize 금지)."""
    prob=torch.sigmoid(logits).detach().cpu().numpy()
    W,H=fr["size"]; dims=fr["dims"]; K=fr["K"]; t=fr["t"]
    es=PPD.polarity_edge_classes(dims)
    best=None
    for Rc in fr["cands"]:
        proj,dep=PG.project_points(PG.make_corners(*dims)[:8],Rc,t,K)
        num=den=0.0
        for (i,j),cls in es:
            if dep[i]<=1e-6 or dep[j]<=1e-6: continue
            cl=PG.clip_segment_to_image(proj[i],proj[j],W,H)
            if cl is None: continue
            s=PG.sample_along(cl[0],cl[1],pixels_per_sample=4.0)
            g=np.stack([s[:,0]*(GRID-1)/max(W-1,1), s[:,1]*(GRID-1)/max(H-1,1)],1)
            ci=PLH.CLASS_ORDER.index(cls)
            v,ins=DGP.bilinear_sample(prob[ci],g)
            if not bool(ins.any()): continue
            num+=float(np.sum(-np.log(np.clip(v[ins],1e-6,1.)))); den+=float(ins.sum())
        if den<=0: continue
        e=num/den
        if best is None or e<best[0]: best=(e,Rc)
    if best is None: return None
    return PPD.polarity_correct(best[1],fr["R"],fr["dims"])

def run(arm,frames,base,ch,init_line,init_mask,cal,init_line0=None):
    init_line0=init_line0 if init_line0 is not None else init_line
    seed_all(SEED)
    line=PLH.PolarityLineHead(ch).to(dev); line.load_state_dict(init_line)
    mask=PLH.FreshMaskHead(ch).to(dev); mask.load_state_dict(init_mask)
    params=list(line.parameters())+(list(mask.parameters()) if arm in("M0","M1") else [])
    opt=torch.optim.AdamW(params,lr=LR,weight_decay=WD)
    X=torch.stack([f["x"] for f in frames]).to(dev)
    Y=torch.stack([f["target"] for f in frames]).to(dev)
    M=torch.stack([f["mask"] for f in frames]).to(dev)
    with torch.no_grad(): FEAT=torch.cat([base(X[i:i+8]) for i in range(0,len(X),8)])
    pw=torch.tensor(cal["pos_weight"],dtype=torch.float32)
    g=torch.Generator().manual_seed(SEED); hist=[]
    for step in range(STEPS):
        idx=torch.randperm(len(frames),generator=g)[:BATCH]
        f_,y_,m_=FEAT[idx],Y[idx],M[idx]
        ml=mask(f_)
        gate=PLH.soft_gate(ml) if arm=="M1" else torch.ones_like(ml)
        lo=line(f_,gate)
        loss=PLH.line_map_loss(lo,y_,pw)+cal["lambda_pol"]*PLH.polarity_contrast_loss(lo,y_)
        if arm in("M0","M1"):
            loss=loss+cal["lambda_mask"]*PLH.mask_loss(ml,m_)+cal["lambda_out"]*PLH.outside_mask_penalty(lo,PLH.soft_gate(ml))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if (step+1)%200==0: hist.append({"step":step+1,"loss":float(loss)})
    line.eval(); mask.eval()
    with torch.no_grad():
        ml=mask(FEAT); gate=PLH.soft_gate(ml) if arm=="M1" else torch.ones_like(ml)
        lo=line(FEAT,gate)
        rec,pre,f1=line_metrics(lo,Y)
        with torch.no_grad():
            ln0=PLH.PolarityLineHead(FEAT.shape[1]).to(dev); ln0.load_state_dict(init_line0)
            lo0=ln0(FEAT,torch.ones(len(FEAT),1,GRID,GRID,device=dev))
            d_init=float((torch.sigmoid(lo0)-Y).abs().mean()); d_fin=float((torch.sigmoid(lo)-Y).abs().mean())
        mp=(torch.sigmoid(ml)>=0.5).float()
        inter=(mp*M).sum(); iou=float(inter/((mp+M).clamp(0,1).sum().clamp_min(1)))
        dice=float(2*inter/(mp.sum()+M.sum()).clamp_min(1))
        # top/base semantic pixel accuracy
        idxs={n:i for i,n in enumerate(PLH.CLASS_ORDER)}
        corr=tot=0
        for tn,bn in PLH.POLARITY_PAIRS:
            ti,bi=idxs[tn],idxs[bn]
            pos=(Y[:,ti]>=0.5)&(Y[:,bi]<=0.1)
            if pos.any(): corr+=int(((lo[:,ti]>lo[:,bi])&pos).sum()); tot+=int(pos.sum())
            pos=(Y[:,bi]>=0.5)&(Y[:,ti]<=0.1)
            if pos.any(): corr+=int(((lo[:,bi]>lo[:,ti])&pos).sum()); tot+=int(pos.sum())
        sem=corr/max(tot,1)
        ok=n=0
        for i,fr in enumerate(frames):
            r=polarity_from_maps(lo[i],fr)
            if r is None: continue
            n+=1; ok+=int(r)
    return {"arm":arm,"loss_hist":hist,"dist_init":d_init,"dist_final":d_fin,
            "dist_reduction":1.0-d_fin/max(d_init,1e-9),"mask_iou":iou,"mask_dice":dice,
            "line_recall":{c:float(rec[i]) for i,c in enumerate(PLH.CLASS_ORDER)},
            "line_precision":{c:float(pre[i]) for i,c in enumerate(PLH.CLASS_ORDER)},
            "macro_f1":float(np.mean(f1)),"topbase_semantic_acc":sem,
            "candidate_polarity_acc":ok/max(n,1),"n_scored":n,"inversion":n-ok}

def main():
    man=json.load(open(D/"ppd_overfit32_pair_manifest.json"))
    print(f"[overfit] {man['n']} frames  sha={man['sha256'][:16]}")
    frames=load_frames(man["files"]); print(f"  loaded, cands median={np.median([len(f['cands']) for f in frames]):.0f}")
    base=Base().to(dev)
    idx,ch=base.discover(frames[0]["x"][None].to(dev))
    print(f"  high-res feature: vgg[{idx}]  channels={ch}  shape asserted {GRID}x{GRID}")
    seed_all(SEED)
    init_line=PLH.PolarityLineHead(ch).state_dict(); init_mask=PLH.FreshMaskHead(ch).state_dict()
    # loss calibration (train frames only, no update)
    X=torch.stack([f["x"] for f in frames]).to(dev)
    with torch.no_grad(): FEAT=torch.cat([base(X[i:i+8]) for i in range(0,len(X),8)])
    Y=torch.stack([f["target"] for f in frames]).to(dev); M=torch.stack([f["mask"] for f in frames]).to(dev)
    pos=(Y>=0.5).float().mean(dim=(0,2,3)).clamp_min(1e-6)
    pw=((1-pos)/pos).clamp(1,200).cpu().numpy().tolist()
    ln=PLH.PolarityLineHead(ch).to(dev); ln.load_state_dict(init_line)
    mk=PLH.FreshMaskHead(ch).to(dev); mk.load_state_dict(init_mask)
    L=[];P=[];MK=[];O=[]
    with torch.no_grad():
        g=torch.Generator().manual_seed(SEED)
        for _ in range(20):
            i=torch.randperm(len(frames),generator=g)[:BATCH]
            ml=mk(FEAT[i]); lo=ln(FEAT[i],torch.ones_like(ml))
            L.append(float(PLH.line_map_loss(lo,Y[i],torch.tensor(pw))))
            P.append(float(PLH.polarity_contrast_loss(lo,Y[i])))
            MK.append(float(PLH.mask_loss(ml,M[i]))); O.append(float(PLH.outside_mask_penalty(lo,PLH.soft_gate(ml))))
    lm=float(np.median(L))
    cal={"pos_weight":pw,"L_line_median":lm,"L_pol_median":float(np.median(P)),
         "L_mask_median":float(np.median(MK)),"L_out_median":float(np.median(O)),
         "lambda_pol":0.10*lm/max(np.median(P),1e-9),
         "lambda_mask":0.50*lm/max(np.median(MK),1e-9),
         "lambda_out":0.05*lm/max(np.median(O),1e-9),
         "source":"train split only, 20 batches, no update"}
    json.dump(cal,open(D/"ppd_t2_loss_calibration.json","w"),indent=1)
    print(f"  calibration: L_line={lm:.4f} lam_pol={cal['lambda_pol']:.4g} lam_mask={cal['lambda_mask']:.4g} lam_out={cal['lambda_out']:.4g}")
    res=[]
    for arm in ("L0","M0","M1"):
        s=time.time(); r=run(arm,frames,base,ch,init_line,init_mask,cal); r["seconds"]=time.time()-s
        res.append(r)
        print(f"\n[{arm}] maskIoU={r['mask_iou']:.3f} dice={r['mask_dice']:.3f} macroF1={r['macro_f1']:.3f} "
              f"semAcc={r['topbase_semantic_acc']:.3f} polAcc={r['candidate_polarity_acc']:.3f} "
              f"inv={r['inversion']}/{r['n_scored']}  ({r['seconds']:.0f}s)")
        print(f"      recall {[round(v,3) for v in r['line_recall'].values()]}")
        print(f"      prec   {[round(v,3) for v in r['line_precision'].values()]}")
    json.dump(res,open(D/"ppd_t2_gate_overfit.json","w"),indent=1)
    WOUT.mkdir(parents=True,exist_ok=True)
main()
