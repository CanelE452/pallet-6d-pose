"""Fixed population and unseen-ratio evaluation. No real labels in inference."""
import argparse
from context16 import *
from model16 import EvidenceEvaluator


METHODS=('stock','base','amodal','v14','balanced_old','risk_old','balanced_rich')+ARMS+('role_off','role_donor')

def prediction_path(split,condition,name):
    arm=name.rsplit('_seed',1)[0]
    parent=V.RUN if arm in ('stock','base','amodal','v14','balanced_old','risk_old','balanced_rich') else RUN
    return parent/'predictions'/split/condition/f'{name}.npz'

def inference():
    setup();assert read(RUN/'TRAINING_COMPLETE.json')['complete']
    if (RUN/'EVAL_PROTOCOL.json').exists():raise SystemExit('Refusing to overwrite inference')
    write(RUN/'EVAL_PROTOCOL.json',dict(training_sha256=sha(RUN/'TRAIN_PROTOCOL.json'),source_sha256=sha(Path(__file__)),
        policy='No GT/condition/corruption mask input. role_off zero evidence embedding; role_donor rolls line-sampled feature source by one detected image in fixed population, preserving candidates, point patches and target-image coordinate grid/content mask. An OOD intervention, not training-matched replacement.',
        baseline='Immutable V15 prediction artifacts, same population/caches.'))
    receipts=[]
    for split in ('synth_val','real'):
        for condition in CONDITIONS:
            m,O=load(split,condition,'cuda:0')
            donor=O['features'].roll(1,0)
            for seed in SEEDS:
                rep=repcache(split,condition,seed,device='cuda:0')
                for arm in ARMS:
                    ck=RUN/'models'/f'{arm}_seed{seed}'/'final.pt';model=EvidenceEvaluator(seed).cuda().eval();model.load_state_dict(torch.load(ck,map_location='cpu',weights_only=False)['state_dict'])
                    for variant in (('actual','off','donor') if arm=='role' else ('actual',)):
                        parts=[]
                        with torch.inference_mode():
                            for first in range(0,len(O['points']),32):
                                sl=slice(first,first+32);p=model(subset(O,sl),subset(rep,sl),arm,evidence_off=variant=='off',evidence_features=donor[sl] if variant=='donor' else None)
                                assert torch.isfinite(p['points']).all();parts.append({k:v.cpu() for k,v in p.items() if k not in ('role_logits','sample_center')})
                        out={k:torch.cat([p[k] for p in parts]) for k in parts[0]};points=out.pop('points');name=arm if variant=='actual' else 'role_'+variant
                        save(split,condition,f'{name}_seed{seed}',m,O,points,out)
                        receipts.append(dict(split=split,condition=condition,arm=name,seed=seed,frames=len(points),checkpoint_sha256=sha(ck)))
                    del model
            print('INFERRED',split,condition,len(O['points']),flush=True)
    write(RUN/'INFERENCE_COMPLETE.json',dict(complete=True,rows=receipts,gt_used=False))

def scoring():
    assert read(RUN/'INFERENCE_COMPLETE.json')['complete'];results={};comparisons={};diagnostics={};bindings={}
    for split in ('synth_val','real'):
        for condition in CONDITIONS:
            m=manifest(split,condition);t=targets(split,condition);gt=np.array([r['points'] for r in t]);valid=np.array([r['valid'] for r in t]);kind=C.P.target_type(t).numpy()
            rows={};arrays={};diag={};methods=METHODS
            for arm in methods:
                for seed in ([1] if arm=='stock' else SEEDS):
                    name='stock' if arm=='stock' else f'{arm}_seed{seed}';path=prediction_path(split,condition,name)
                    rows[name],arrays[name]=E.score_file(path,m,t);bindings[str(path.relative_to(ROOT))]=sha(path)
                    p=np.load(path);det=p['detected'];assert np.array_equal(det,arrays['stock']['detected'])
                    if arm not in ARMS and not arm.startswith('role_'):continue
                    error=np.linalg.norm(p['candidates']-gt[:,:,None],axis=-1);allowed=p['eligible']>.5;oracle=np.where(allowed,error,np.inf).min(-1);actual=arrays[name]['error'];d={}
                    for typ,k in [('all',None),('hidden',1),('outside',2)]:
                        mask=valid&det[:,None]
                        if k is not None:mask=mask&(kind==k)
                        if not mask.any():continue
                        applied=mask&(p['choice']>0);bad=applied&(actual>error[...,0]);truth=error[...,0]-error[...,2]
                        d[typ]=dict(points=int(mask.sum()),choice_counts=[int((mask&(p['choice']==j)).sum()) for j in range(8)],
                            oracle_px=float(oracle[mask].mean()),actual_px=float(actual[mask].mean()),regret_px=float((actual-oracle)[mask].mean()),
                            applied_points=int(applied.sum()),harmful_applied_points=int(bad.sum()),harmful_applied_fraction=float(bad.sum()/applied.sum()) if applied.any() else None,
                            true_amodal_gain_mean_px=float(truth[mask].mean()),predicted_amodal_lower_gain_mean_px=float(p['gain'][...,2][mask].mean()),
                            catastrophes_base5_to20=int((mask&(error[...,0]<=5)&(actual>20)).sum()))
                    diag[name]=d
            pairs=[(a,b) for a in ARMS for b in ('base','balanced_old','control') if a!=b]+[('role','strip'),('role','role_off'),('role','role_donor')]
            comp={};sessions=[r['session'] for r in t]
            for a,b in pairs:
                c=COMPARE(arrays,rows,a,b,sessions,split);c['subgroups']={typ:COMPARE(arrays,rows,a,b,sessions,split,mask=valid&(kind==k)) for typ,k in [('hidden',1),('outside',2)]};comp[f'{a}_vs_{b}']=c
            key=split+'/'+condition;results[key]=rows;comparisons[key]=comp;diagnostics[key]=diag
            print('SCORED',key,{a:[round(rows[f'{a}_seed{s}']['frame_mean_px'],3) for s in SEEDS] for a in ('base','amodal')+ARMS},flush=True)
    gates={};line_gates={}
    for arm in ARMS:
        per={}
        for seed in SEEDS:
            name=f'{arm}_seed{seed}';b=f'base_seed{seed}';a=f'amodal_seed{seed}';r=results['real/clean']
            g=dict(clean_within1=r[name]['frame_mean_px']<=1.01*r[b]['frame_mean_px'])
            for c in ('crop20',)+NEW_CONDITIONS:
                r=results['real/'+c];g[c+'_outside_within1']=r[name]['strata']['outside']['mean_px']<=1.01*r[b]['strata']['outside']['mean_px']
            for c,typ in [('corner20','hidden'),('crop40','outside')]:
                r=results['real/'+c];before=r[b]['strata'][typ]['mean_px'];full=r[a]['strata'][typ]['mean_px'];new=r[name]['strata'][typ]['mean_px'];g[c+'_retain80']=before-new>=.8*(before-full)
            per[str(seed)]=g
        gates[arm]=dict(per_seed=per,core_pass=all(all(v for k,v in g.items() if not any(k.startswith(c) for c in NEW_CONDITIONS)) for g in per.values()),all_conditions_pass=all(all(g.values()) for g in per.values()))
    for arm in ('strip','role','balanced_role','median_role'):
        line_gates[arm]={}
        for c,typ in [('clean',None),('corner20','hidden'),('crop20','outside'),('crop40','outside'),('crop10','outside'),('crop30','outside')]:
            comp=comparisons['real/'+c][arm+'_vs_control'];gain=comp['relative_gain_percent'] if typ is None else comp['subgroups'][typ]['per_seed_gain_percent'];ci=comp['paired_px']['ci95'] if typ is None else comp['subgroups'][typ]['ci95']
            line_gates[arm][c]=dict(every_seed_gain1=all(v>=1 for v in gain),ci_below0=ci[1]<0,pass_gate=all(v>=1 for v in gain) and ci[1]<0)
    write(RUN/'SUMMARY.json',dict(complete=True,results=results,comparisons=comparisons,diagnostics=diagnostics,gates=gates,line_gates=line_gates,
        stable_arms=[a for a,g in gates.items() if g['all_conditions_pass']],scope='Reused DEV, unchanged V15 populations/severities. Auxiliary boundary labels are synthetic geometric proxies. No real tuning. Comparison of strip vs control isolates evidence; role vs strip isolates auxiliary supervision; balanced/median also change loss.'))
    write(RUN/'EVALUATION_BINDINGS.json',dict(files=bindings))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['inference','score'],default='inference');a=ap.parse_args();inference() if a.phase=='inference' else scoring()
