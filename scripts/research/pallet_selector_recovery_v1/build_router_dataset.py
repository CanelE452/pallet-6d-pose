"""Fixed synthetic clean/occluded pairs; exact labels only after predictions freeze."""
import argparse
import ast
import time
from collections import Counter
import cv2
import numpy as np
import torch
from . import common as C
from . import features as F
from . import router_features as RF
from .extract_features import Expert

def base_lock():
    d=C.read(C.sdoc(3)/'STAGE3_DECISION.json')
    base='SYNTH_SCORER' if d['primary']=='SELECTOR_RECOVERY_WITHOUT_COLLATERAL' and d['S0_valid_for_router'] else 'PRODUCTION_D9'
    C.freeze(C.sdoc(4)/'BASE_SELECTOR_FOR_ROUTER.json',dict(created_at=C.now(),base=base,both_experts_same=True,decision=C.bind(C.sdoc(3)/'STAGE3_DECISION.json')))

def policy():
    path=C.ROOT/'scripts/research/pallet_clean19_structured_easyhard_v1/augmentation.py'
    tree=ast.parse(path.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
    edges=[(0,1),(2,3),(4,5),(6,7),(0,4),(1,5),(2,6),(3,7),(0,3),(1,2),(4,7),(5,6)]
    env=dict(np=np,cv2=cv2,EDGES=edges);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)
    return env,path

def plans():
    source={r['id']:r for r in C.read(C.ROOT/'data/pallet/results/pallet_line_pose_v1/SOURCE_MANIFEST.json')['records']}
    rows=C.read(C.sraw(2)/'SYNTH_RECORDS.json');a,path=policy();out={}
    for r in rows:
        src=source[r['id']];h,w=r['hw'];scale=640/max(h,w);offset=np.array([(640-w*scale)/2,(640-h*scale)/2]);pad=r['pad'] if 'pad' in r else src['reflect_pad_px']
        kp=np.array(src['targets'][0]['keypoints_normalized']);points=kp[:,:2]*[w,h]*scale+offset
        xywh=np.array(src['targets'][0]['box_xywh_normalized'])*[w,h,w,h];box=np.r_[xywh[:2]-xywh[2:]/2,xywh[:2]+xywh[2:]/2]*scale+np.tile(offset,2)
        canvas=np.array([pad,pad,w-pad,h-pad])*scale+np.tile(offset,2)
        seed=int(C.key(r['id'],20260926)[:8],16)
        # Existing policy's supervised flag is v==2, not every nonzero point.
        p=a['plan'](points,kp[:,2]==2,box,canvas,seed)
        p.update(scale=scale,offset=offset.tolist(),native_canvas=canvas.tolist(),bbox=box.tolist())
        if p['applied']:
            l,t,rw,rh=p['S1'];lo=np.rint((np.array([l,t])-offset)/scale).astype(int);hi=np.rint((np.array([l+rw,t+rh])-offset)/scale).astype(int)
            lo=np.maximum(lo,[pad,pad]);hi=np.minimum(hi,[w-pad,h-pad]);assert np.all(hi>lo)
            p['prepared_rectangle']=[int(lo[0]),int(lo[1]),int(hi[0]),int(hi[1])]
        out[r['id']]=p
    C.freeze(C.sraw(4)/'OCCLUSION_PLANS.json',out)
    C.freeze(C.sdoc(4)/'OCCLUSION_PLAN_LOCK.json',dict(created_at=C.now(),plans=C.bind(C.sraw(4)/'OCCLUSION_PLANS.json'),original_policy=C.bind(path),
        source_geometry_used_for_original_coverage_only=True,GT_error_used=False,model_outcomes_used=False,seed=20260926,
        reasons=dict(Counter(p['reason'] for p in out.values())),applied=sum(p['applied'] for p in out.values()),frames=len(out),
        adaptation='Same original size/fill/coverage/paired-feasibility policy on 640 letterbox, inverse mapped with rounding to native prepared canvas; no original training random-affine augmentation.',
        difference='Original S1 applied this policy to real training images; this pilot applies it to synthetic frames as requested. v==2 supervised mask retained.'))

def extract(batch=16):
    C.setup();start=time.perf_counter();gpu=C.gpu();rr=C.read(C.sraw(2)/'SYNTH_INPUTS.json');pl=C.read(C.sraw(4)/'OCCLUSION_PLANS.json');a,_=policy();arrays={};allpred={};audits=[]
    assert not (C.sraw(4)/'FEATURES_OCCLUDED.npz').exists()
    for arm in C.ARMS:
        model=Expert(arm);audits.append(model.capture_test(cv2.imread(str(C.ROOT/rr[0]['image']['path']))));geo=[];ctx=[];valid=[];cur=[];allpred[arm]={}
        for off in range(0,len(rr),batch):
            if off%256==0:C.gpu();print('ROUTER_OCC',arm,off,'/',len(rr),flush=True)
            items=rr[off:off+batch];images=[]
            for r in items:
                im=cv2.imread(str(C.ROOT/r['image']['path']));p=pl[r['id']]
                if p['applied']:
                    x0,y0,x1,y1=p['prepared_rectangle'];rgbfill=a['fill'](p).transpose(1,2,0)
                    im[y0:y1,x0:x1]=cv2.resize(rgbfill,(x1-x0,y1-y0),interpolation=cv2.INTER_LINEAR)[:,:,::-1]
                images.append(im)
            pp,cc=model.run(images)
            for j,(r,p) in enumerate(zip(items,pp)):
                hw=images[j].shape[:2];g=F.extract(p,r['K'],r['dims'],hw)
                geo.append(g['features'] if g['valid'] else np.zeros((2,len(F.names()))));ctx.append(cc[j] if cc is not None else np.zeros(0));valid.append(g['valid']);cur.append(C.HYP.index(g['selection']) if g['selection'] in C.HYP else -1)
                allpred[arm][r['id']]=dict(prediction=p,geometry=g,hw=hw)
        model.integrity();arrays[arm+'_geo']=np.array(geo,np.float32);arrays[arm+'_ctx']=np.array(ctx,np.float32);arrays[arm+'_valid']=np.array(valid);arrays[arm+'_current']=np.array(cur)
        del model;torch.cuda.empty_cache()
    np.savez_compressed(C.sraw(4)/'FEATURES_OCCLUDED.npz',ids=np.array([r['id'] for r in rr]),**arrays)
    C.freeze(C.sraw(4)/'PREDICTIONS_OCCLUDED.json',allpred)
    C.freeze(C.sdoc(4)/'ROUTER_SYNTH_PREDICTION_LOCK.json',dict(created_at=C.now(),features=C.bind(C.sraw(4)/'FEATURES_OCCLUDED.npz'),predictions=C.bind(C.sraw(4)/'PREDICTIONS_OCCLUDED.json'),
        plans=C.bind(C.sdoc(4)/'OCCLUSION_PLAN_LOCK.json'),captures=audits,seconds=time.perf_counter()-start,batch=batch,gpu_start=gpu,gpu_end=C.gpu(),GT_error_input=False))

def assemble():
    from .synth_labels import exact,addnorm
    lock=C.read(C.sdoc(4)/'ROUTER_SYNTH_PREDICTION_LOCK.json');C.verify(lock['features']);C.verify(lock['predictions']);refstart=C.now();assert refstart>lock['created_at']
    rows=C.read(C.sraw(2)/'SYNTH_RECORDS.json');gt=exact(rows);rows={r['id']:r for r in rows};pl=C.read(C.sraw(4)/'OCCLUSION_PLANS.json');base=C.read(C.sdoc(4)/'BASE_SELECTOR_FOR_ROUTER.json')['base'];ck=None
    if base=='SYNTH_SCORER':
        b=C.read(C.sdoc(2)/'SCORER_SELECTION_LOCK.json')['checkpoint'];C.verify(b);ck=torch.load(C.ROOT/b['path'],map_location='cpu',weights_only=False)
    X=[];yy=[];ids=[];parts=[];conditions=[];errs=[];applied=[]
    for cond,root,tag in [('CLEAN_SYNTH',C.sraw(2),'CLEAN'),('OCCLUDED_SYNTH',C.sraw(4),'OCCLUDED')]:
        z=dict(np.load(root/f'FEATURES_{tag}.npz'));pred=C.read(root/f'PREDICTIONS_{tag}.json');choices=RF.choices(z,pred,base,ck)
        for i,fid in enumerate(z['ids']):
            pp=[pred[a][fid] for a in C.ARMS];idx=[int(choices[a]['selected'][i]) for a in C.ARMS];names=[C.HYP[k] if k>=0 else None for k in idx];g=gt[fid]
            X.append(RF.make(pp[0]['prediction'],pp[1]['prediction'],pp[0]['geometry'],pp[1]['geometry'],z['S1_ctx'][i],*names,choices['S0']['margin'][i],choices['S1']['margin'][i],g['dims']))
            e=[addnorm(p['geometry']['hypotheses'][k],g) if k>=0 and len(p['geometry']['hypotheses'])==2 else float('inf') for p,k in zip(pp,idx)]
            # Both unavailable => uninformative tie S0; one unavailable => finite expert.
            label=int(e[1]<e[0] and (not np.isfinite(e[0]) or abs(e[1]-e[0])>1e-9))
            yy.append(label);errs.append(e);ids.append(fid);parts.append(rows[fid]['split']);conditions.append(cond);applied.append(cond=='OCCLUDED_SYNTH' and pl[fid]['applied'])
    path=C.sraw(4)/'ROUTER_DATASET.npz';assert not path.exists();np.savez_compressed(path,X=np.array(X),y=np.array(yy),errors=np.array(errs),ids=np.array(ids),split=np.array(parts),condition=np.array(conditions),applied=np.array(applied))
    stats={}
    for part in ('TRAIN','VAL','TEST'):
        mask=np.array(parts)==part;stats[part]=dict(samples=int(mask.sum()),choose_S1=int(np.array(yy)[mask].sum()),occlusion_applied=int(np.array(applied)[mask].sum()),both_unavailable=int(np.isinf(np.array(errs)[mask]).all(1).sum()))
    C.freeze(C.sdoc(4)/'ROUTER_DATASET_LOCK.json',dict(created_at=C.now(),dataset=C.bind(path),reference_read_time=refstart,prediction_lock_time=lock['created_at'],stats=stats,base_selector=base,dimensions=np.array(X).shape[1],same_frame_same_split=True,labels='lower exact synthetic C2 ADDnorm; tie1e-9 -> S0'))
    print('ROUTER_DATASET_READY',stats,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('step',choices=['base_lock','plans','extract','assemble']);p.add_argument('--batch',type=int,default=16);a=p.parse_args()
    extract(a.batch) if a.step=='extract' else globals()[a.step]()
