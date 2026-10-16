"""Continuous-vector predictions; same original populations and point metrics."""
import argparse
from context18 import *
from model18 import VectorRefiner

METHODS=('stock','base','amodal','balanced_old','strip','high_none')+ARMS+('line_vector_off','line_vector_donor','line_risk_off','line_risk_donor')
PAIRS=[(a,b) for a in ARMS for b in ('base','amodal','balanced_old','strip','high_none')]+[('line_vector','point_vector'),('line_risk','point_risk'),('point_risk','point_vector'),('line_risk','line_vector')]+[(a,a+'_'+v) for a in ('line_vector','line_risk') for v in ('off','donor')]

def prediction_path(split,condition,name):
    a=name.rsplit('_seed',1)[0]
    root=V.RUN if a in ('stock','base','amodal','balanced_old') else P.RUN if a=='strip' else ROOT/'data/pallet/results/pallet_detail_v17_fixed' if a=='high_none' else RUN
    return root/'predictions'/split/condition/f'{name}.npz'

def inference():
    setup();assert read(RUN/'TRAINING_COMPLETE.json')['complete']
    if (RUN/'EVAL_PROTOCOL.json').exists():raise SystemExit('Refusing overwrite')
    write(RUN/'EVAL_PROTOCOL.json',dict(training_sha256=sha(RUN/'TRAIN_PROTOCOL.json'),source_sha256=sha(Path(__file__)),gt_used=False,
        controls='Line off zeros96line token in the same checkpoint. Donor rolls only source feature maps for line-strip sampling by1 in detected population, retaining true-image local patches/base/context/line geometry. OOD causal diagnostic, not selection.',
        baselines='Immutable V15 balanced_old, V16 strip and corrected V17 high_none candidate selectors plus V12 base/amodal and YOLO; same observation caches/detections/GT.'))
    receipts=[]
    for split in ('synth_val','real'):
        for condition in CONDITIONS:
            m,O=load(split,condition,'cuda:0');donor=O['features'].roll(1,0)
            for seed in SEEDS:
                rep=repcache(split,condition,seed,device='cuda:0')
                for arm in ARMS:
                    ck=RUN/'models'/f'{arm}_seed{seed}'/'final.pt';model=VectorRefiner(seed).cuda().eval();model.load_state_dict(torch.load(ck,weights_only=True)['state_dict'])
                    for variant in (('actual','off','donor') if arm.startswith('line_') else ('actual',)):
                        parts=[]
                        with torch.inference_mode():
                            for first in range(0,len(O['points']),32):
                                sl=slice(first,first+32);out=model(subset(O,sl),subset(rep,sl),arm,line_off=variant=='off',line_features=donor[sl] if variant=='donor' else None)
                                assert torch.isfinite(out['points']).all();parts.append({k:v.cpu() for k,v in out.items() if k!='sample_center'})
                        out={k:torch.cat([p[k] for p in parts]) for k in parts[0]};points=out.pop('points');name=arm if variant=='actual' else arm+'_'+variant;save(split,condition,f'{name}_seed{seed}',m,O,points,out)
                        receipts.append(dict(split=split,condition=condition,arm=name,seed=seed,frames=len(points),checkpoint_sha256=sha(ck)))
                    del model
            print('INFERRED',split,condition,len(O['points']),flush=True)
    write(RUN/'INFERENCE_COMPLETE.json',dict(complete=True,rows=receipts,gt_used=False))

def scoring():
    assert read(RUN/'INFERENCE_COMPLETE.json')['complete'];results={};comparisons={};diagnostics={};bindings={}
    for split in ('synth_val','real'):
        for condition in CONDITIONS:
            m=manifest(split,condition);t=targets(split,condition);gt=np.array([r['points'] for r in t]);valid=np.array([r['valid'] for r in t]);kind=C.P.target_type(t).numpy();rows={};arrays={};diag={}
            for arm in METHODS:
                for seed in ([1] if arm=='stock' else SEEDS):
                    name='stock' if arm=='stock' else f'{arm}_seed{seed}';path=prediction_path(split,condition,name);rows[name],arrays[name]=E.score_file(path,m,t);bindings[str(path.relative_to(ROOT))]=sha(path)
                    pred=np.load(path);det=pred['detected'];assert np.array_equal(det,arrays['stock']['detected'])
                    if arm not in ARMS and arm not in METHODS[-4:]:continue
                    base=pred['base'].astype(float);new=pred['points'].astype(float);step=new-base;target=gt-base;before=np.linalg.norm(target,axis=-1);after=arrays[name]['error'];movement=np.linalg.norm(step,axis=-1);dot=(step*target).sum(-1);radius=pred['radius'];d={}
                    for typ,k in [('all',None),('hidden',1),('outside',2)]:
                        mask=valid&det[:,None]
                        if k is not None:mask&=kind==k
                        if not mask.any():continue
                        moved=mask&(movement>.1);bad=moved&(after>before+1e-4);good=moved&(after<before-1e-4);direction=moved&(before>.01)
                        d[typ]=dict(points=int(mask.sum()),mean_movement_px=float(movement[mask].mean()),moved_points=int(moved.sum()),helpful_points=int(good.sum()),harmful_points=int(bad.sum()),
                            wrong_direction_harms=int((bad&(dot<=0)).sum()),overshoot_harms=int((bad&(dot>0)).sum()),
                            mean_direction_cosine=float((dot/(movement*before).clip(1e-9))[direction].mean()) if direction.any() else None,
                            base_error_px=float(before[mask].mean()),actual_error_px=float(after[mask].mean()),vector_infimum_px=float(np.maximum(before-radius[:,None],0)[mask].mean()),
                            catastrophes_base5_to20=int((mask&(before<=5)&(after>20)).sum()),near_bound_points=int((mask&(movement>.9*radius[:,None])).sum()))
                    diag[name]=d
            comp={};sessions=[r['session'] for r in t]
            for a,b in PAIRS:
                v=COMPARE(arrays,rows,a,b,sessions,split);v['subgroups']={typ:COMPARE(arrays,rows,a,b,sessions,split,mask=valid&(kind==k)) for typ,k in [('hidden',1),('outside',2)]};comp[a+'_vs_'+b]=v
            key=split+'/'+condition;results[key]=rows;comparisons[key]=comp;diagnostics[key]=diag;print('SCORED',key,{a:round(float(np.mean([rows[f'{a}_seed{i}']['frame_mean_px'] for i in SEEDS])),3) for a in ARMS},flush=True)
    gates={}
    for arm in ARMS:
        per={}
        for seed in SEEDS:
            n=f'{arm}_seed{seed}';b=f'base_seed{seed}';a=f'amodal_seed{seed}';r=results['real/clean'];g=dict(clean_within1=r[n]['frame_mean_px']<=1.01*r[b]['frame_mean_px'])
            for c in ('crop20','crop10','crop30'):
                r=results['real/'+c];g[c+'_outside_within1']=r[n]['strata']['outside']['mean_px']<=1.01*r[b]['strata']['outside']['mean_px']
            for c,typ in [('corner20','hidden'),('crop40','outside')]:
                r=results['real/'+c];before=r[b]['strata'][typ]['mean_px'];full=r[a]['strata'][typ]['mean_px'];new=r[n]['strata'][typ]['mean_px'];g[c+'_retain80']=before-new>=.8*(before-full)
            per[str(seed)]=g
        gates[arm]=dict(per_seed=per,all_conditions_pass=all(all(g.values()) for g in per.values()))
    line_gates={}
    for a,b in [('line_vector','point_vector'),('line_risk','point_risk')]:
        key=a+'_vs_'+b;line_gates[key]={}
        for c,typ in [('clean',None),('corner20','hidden'),('crop20','outside'),('crop40','outside'),('crop10','outside'),('crop30','outside')]:
            comp=comparisons['real/'+c][key];gain=comp['relative_gain_percent'] if typ is None else comp['subgroups'][typ]['per_seed_gain_percent'];ci=comp['paired_px']['ci95'] if typ is None else comp['subgroups'][typ]['ci95'];line_gates[key][c]=dict(every_seed_gain1=all(v>=1 for v in gain),ci_below0=ci[1]<0,pass_gate=all(v>=1 for v in gain) and ci[1]<0)
    write(RUN/'SUMMARY.json',dict(complete=True,results=results,comparisons=comparisons,diagnostics=diagnostics,gates=gates,line_gates=line_gates,stable_arms=[a for a,g in gates.items() if g['all_conditions_pass']],scope='Reused DEV; continuous point-vector supervision; no mask or line labels. Movement threshold.1px diagnostic only, no filtering at inference.'))
    write(RUN/'EVALUATION_BINDINGS.json',dict(files=bindings))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['inference','score'],default='inference');a=ap.parse_args();inference() if a.phase=='inference' else scoring()
