"""Add training-only corner occlusion, preserving all existing experiments."""
import cv2
from context12 import *

def main():
    setup();cv2.setNumThreads(1);RUN.mkdir(parents=True,exist_ok=True)
    if (RUN/'PREPARATION_COMPLETE.json').exists():raise SystemExit('V12 preparation already complete')
    pop=read(V11/'POPULATIONS.json')['train'];old,clean=observations('train','cpu');lookup={r['id']:i for i,r in enumerate(old.records)}
    p=import_file('v12_extract_compat',KIT.parent/'pallet_optional_lines_v11'/'prepare.py');extractor=p.Extractor()
    directory=RUN/'cache'/'train_corner20';directory.mkdir(parents=True,exist_ok=True)
    write(RUN/'PROTOCOL.json',dict(arms=ARMS,seeds=SEEDS,steps=2000,batch=16,lr=3e-4,
        question='Can learned suppression of the ORIGINAL local correction protect occluded points, and separate global recovery restore outside points?',
        train='Frozen synthetic1775 clean + existing V11 four stress variants + newly generated predicted-corner occlusion of same384 training images. No real training.',
        inference='Frozen YOLO and V10 local. Per-point image tokens attend to global ROI grid and other predicted point tokens. Three-way soft routing: intact / artificially occluded / outside. No GT, corruption mask, condition ID or annotation visibility at inference.',
        architecture='stock + q_intact*local_delta + q_occluded*recovery_occluded + q_outside*recovery_outside. Separate global residual heads bound each coordinate to .25*observed_box_diagonal.',
        controls='scalar_gate: one input-independent learned shrink factor. learned_gate: image-based routing without recovery. gated_amodal: routed recovery. always_amodal: original local always applied plus recovery, same full module. Inactive capacity differs for gate-only arms.',
        loss='Point SmoothL1 + .25 nonworsening relative to stock; plus .5 class-balanced CE for artificial observability classes except scalar_gate. Valid synthetic points only. No GT filtering/selftraining.',
        evaluation='All six V11 conditions on synthetic505 and real297/9sessions. Same existing observations and missing detections. Real manual clicks only. Reused DEV, no real tuning or new holdout.',
        acceptance='Suppression: all3seed hidden-corner mean better than frozen local and scalar shrink; session-cluster CI below0; real clean degradation<=1% against frozen local. Recovery: all3seed outside mean >=3% better than learned_gate on both crops; cluster CI below0; real clean degradation<=1%. Also compare original YOLO, all-frame failure-inclusive score, visible points and catastrophes.',
        limits='Artificial visibility targets identify intervention exposure, not true natural visibility. No Hough added in this mechanism-isolation test. No production replacement.',
        parent_summary_sha256=sha(V11/'SUMMARY.json'),parent_targeted_summary_sha256=sha(V11/'TARGETED_SUMMARY.json'),
        population_sha256=sha(V11/'POPULATIONS.json')))
    chunks=[];rows=[];pending=[];n=0
    for rec in pop:
        i=lookup[rec['id']];im=p.raw_image(rec);h,w=im.shape[:2];box=clean['box'][i].numpy()
        corner=[0,1,4,5][int(digest('v12-train-corner:'+rec['id'])[:8],16)%4];center=clean['points'][i,corner].numpy()
        side=.2*np.linalg.norm(box[2:]-box[:2]);a=max(0,min(w-1,int(round(center[0]-side/2))));b=max(a+1,min(w,int(round(center[0]+side/2))))
        u=max(0,min(h-1,int(round(center[1]-side/2))));v=max(u+1,min(h,int(round(center[1]+side/2))))
        rng=np.random.default_rng(int(digest('v12-train-color:'+rec['id'])[:8],16));color=rng.integers(40,216,3).astype(np.uint8)
        canvas=im.copy();canvas[u:v,a:b]=color;O,info=extractor(canvas,rec['K'],rec['dims'])
        rows.append(dict(**rec,**info,origin=[0,0],image_hw=[h,w],original_hw=[h,w],mask=[a,u,b,v],fill_bgr=color.tolist(),
            observation_index=n if O is not None else None,image_sha256=sha(ROOT/rec['image'])))
        if O is not None:pending.append(O);n+=1
        if len(pending)>=64 or (len(rows)==len(pop) and pending):
            name=f'obs_{len(chunks):03d}.pt';torch.save({k:torch.cat([o[k] for o in pending]) for k in KEYS},directory/name);chunks.append(name);pending=[]
    write(directory/'manifest.json',dict(rows=rows,chunks=chunks,chunk_sha256={name:sha(directory/name) for name in chunks}))
    # GT point exposure is calculated only after image observations are saved.
    source={r['id']:r for r in read(V10/'targets_train.json')['rows']};targets=[]
    for rec in rows:
        r=dict(source[rec['id']]);g=np.array(r['points']);a,u,b,v=rec['mask'];h,w=rec['image_hw']
        r['hidden']=((g[:,0]>=a)&(g[:,0]<b)&(g[:,1]>=u)&(g[:,1]<v)).tolist()
        r['outside']=((g[:,0]<0)|(g[:,0]>=w)|(g[:,1]<0)|(g[:,1]>=h)).tolist();targets.append(r)
    write(RUN/'targets_train_corner20.json',dict(rows=targets))
    write(RUN/'PREPARATION_COMPLETE.json',dict(complete=True,frames=len(rows),detected=n,
        targeted_hidden_valid_points=int(sum((np.array(r['hidden'])&np.array(r['valid'])).sum() for r in targets)),
        no_real_training=True,source_sha256=sha(Path(__file__))))
    print('PREPARATION COMPLETE',len(rows),n,flush=True)

if __name__=='__main__':main()
