"""Identical original populations; separate inference, scoring and predeclared gates."""
import argparse
from context19 import *
from model19 import RetentionGate
from eval18 import prediction_path as old_path
METHODS=('stock','base','amodal','strip','point_vector','line_vector','line_risk','fixed_half')+ARMS+('area_gate_off','area_gate_donor','line_gate_off','line_gate_donor')
PAIRS=[(a,b) for a in ARMS for b in ('base','line_vector','fixed_half')]+[('area_gate','local_gate'),('line_gate','local_gate'),('line_gate','area_gate')]+[(a,a+'_'+mode) for a in ('area_gate','line_gate') for mode in ('off','donor')]

def prediction_path(split,condition,name):
    arm=name.rsplit('_seed',1)[0]
    return old_path(split,condition,name) if arm in METHODS[:7] else RUN/'predictions'/split/condition/f'{name}.npz'

def inference():
    setup();assert read(RUN/'TRAINING_COMPLETE.json')['complete']
    if (RUN/'EVAL_PROTOCOL.json').exists():raise SystemExit('Refusing overwrite')
    write(RUN/'EVAL_PROTOCOL.json',dict(source_sha256=sha(Path(__file__)),training_sha256=sha(RUN/'TRAIN_PROTOCOL.json'),gt_used=False,interventions=read(RUN/'PROTOCOL.json')['interventions']));receipts=[]
    for split in ('synth_val','real'):
        for condition in CONDITIONS:
            for seed in SEEDS:
                m,O,rep=eval_inputs(split,condition,seed,'cuda:0');donor=O['features'].roll(1,0);base=rep['base'];delta=rep['proposal_delta'];half=dict(points=base+.5*delta,base=base,proposal_delta=delta,alpha=base.new_full(base.shape[:2],.5),coverage=base.new_zeros(base.shape[:2]));save(split,condition,f'fixed_half_seed{seed}',m,O,half)
                for arm in ARMS:
                    ck=RUN/'models'/f'{arm}_seed{seed}'/'final.pt';model=RetentionGate(seed).cuda().eval();model.load_state_dict(torch.load(ck,weights_only=True)['state_dict'])
                    for variant in (('actual','off','donor') if arm!='local_gate' else ('actual',)):
                        parts=[]
                        with torch.inference_mode():
                            for first in range(0,len(base),32):
                                sl=slice(first,first+32);out=model(subset(O,sl),subset(rep,sl),arm,evidence_off=variant=='off',evidence_features=donor[sl] if variant=='donor' else None);assert torch.isfinite(out['points']).all();parts.append({k:v.cpu() for k,v in out.items() if k in ('points','base','proposal_delta','alpha','coverage')})
                        out={k:torch.cat([v[k] for v in parts]) for k in parts[0]};name=arm if variant=='actual' else arm+'_'+variant;save(split,condition,f'{name}_seed{seed}',m,O,out);receipts.append(dict(split=split,condition=condition,arm=name,seed=seed,frames=len(base),checkpoint_sha256=sha(ck)))
                    del model
            print('INFERRED',split,condition,flush=True)
    write(RUN/'INFERENCE_COMPLETE.json',dict(complete=True,rows=receipts,gt_used=False))

def scoring():
    assert read(RUN/'INFERENCE_COMPLETE.json')['complete'];results={};comparisons={};diagnostics={};bindings={}
    for split in ('synth_val','real'):
        for condition in CONDITIONS:
            m=manifest(split,condition);ts=targets(split,condition);gt=np.array([t['points'] for t in ts]);valid=np.array([t['valid'] for t in ts]);kind=C.P.target_type(ts).numpy();rows={};arrays={};ds={}
            for arm in METHODS:
                for seed in ([1] if arm=='stock' else SEEDS):
                    name='stock' if arm=='stock' else f'{arm}_seed{seed}';path=prediction_path(split,condition,name);rows[name],arrays[name]=E.score_file(path,m,ts);bindings[str(path.relative_to(ROOT))]=sha(path);pred=np.load(path);det=pred['detected'];assert np.array_equal(det,arrays['stock']['detected'])
                    if arm not in METHODS[7:]:continue
                    base=pred['base'].astype(float);delta=pred['proposal_delta'].astype(float);alpha=pred['alpha'].astype(float);res=gt-base;movement=np.linalg.norm(delta,axis=-1);oracle=np.clip((res*delta).sum(-1)/np.maximum((delta*delta).sum(-1),1e-12),0,1);before=np.linalg.norm(res,axis=-1);full=np.linalg.norm(base+delta-gt,axis=-1);after=arrays[name]['error'];d={}
                    for typ,k in [('all',None),('hidden',1),('outside',2)]:
                        mask=valid&det[:,None]
                        if k is not None:mask&=kind==k
                        if not mask.any():continue
                        bad=mask&(full>before+.1);good=mask&(full<before-.1);wrong=mask&((res*delta).sum(-1)<=0)&(movement>.1)
                        d[typ]=dict(points=int(mask.sum()),mean_alpha=float(alpha[mask].mean()),oracle_alpha=float(oracle[mask].mean()),alpha_mae=float(np.abs(alpha-oracle)[mask].mean()),movement_weighted_alpha_mae_px=float((movement*np.abs(alpha-oracle))[mask].mean()),base_error_px=float(before[mask].mean()),full_error_px=float(full[mask].mean()),actual_error_px=float(after[mask].mean()),oracle_scale_error_px=float(np.linalg.norm(base+oracle[...,None]*delta-gt,axis=-1)[mask].mean()),original_harmful_points=int(bad.sum()),alpha_on_original_harmful=float(alpha[bad].mean()) if bad.any() else None,original_helpful_points=int(good.sum()),alpha_on_original_helpful=float(alpha[good].mean()) if good.any() else None,remaining_harmful_points=int((mask&(after>before+.1)).sum()),wrong_direction_points=int(wrong.sum()),alpha_on_wrong_direction=float(alpha[wrong].mean()) if wrong.any() else None,mean_coverage=float(pred['coverage'][mask].mean()),catastrophes_base5_to20=int((mask&(before<=5)&(after>20)).sum()))
                    ds[name]=d
            comp={};sessions=[t['session'] for t in ts]
            for a,b in PAIRS:
                v=COMPARE(arrays,rows,a,b,sessions,split);v['subgroups']={typ:COMPARE(arrays,rows,a,b,sessions,split,mask=valid&(kind==k)) for typ,k in [('hidden',1),('outside',2)]};comp[a+'_vs_'+b]=v
            key=split+'/'+condition;results[key]=rows;comparisons[key]=comp;diagnostics[key]=ds;print('SCORED',key,{a:round(float(np.mean([rows[f'{a}_seed{i}']['frame_mean_px'] for i in SEEDS])),3) for a in ARMS},flush=True)
    gates={}
    for arm in ARMS:
        per={}
        for seed in SEEDS:
            n=f'{arm}_seed{seed}';b=f'base_seed{seed}';full=f'line_vector_seed{seed}';r=results['real/clean'];g=dict(clean_within1=r[n]['frame_mean_px']<=1.01*r[b]['frame_mean_px'])
            for c in ('crop20','crop10','crop30'):
                r=results['real/'+c];g[c+'_outside_within1']=r[n]['strata']['outside']['mean_px']<=1.01*r[b]['strata']['outside']['mean_px']
            for c,t in [('corner20','hidden'),('crop40','outside')]:
                r=results['real/'+c];before=r[b]['strata'][t]['mean_px'];original=r[full]['strata'][t]['mean_px'];after=r[n]['strata'][t]['mean_px'];g[c+'_retain90']=before-after>=.9*(before-original)
            per[str(seed)]=g
        gates[arm]=dict(per_seed=per,all_conditions_pass=all(all(g.values()) for g in per.values()))
    evidence_gates={}
    for a,b in [('area_gate','local_gate'),('line_gate','local_gate'),('line_gate','area_gate')]:
        pair=a+'_vs_'+b;by={}
        for c,t in [('clean',None),('corner20','hidden'),('crop20','outside'),('crop40','outside'),('crop10','outside'),('crop30','outside')]:
            v=comparisons['real/'+c][pair];gain=v['relative_gain_percent'] if t is None else v['subgroups'][t]['per_seed_gain_percent'];ci=v['paired_px']['ci95'] if t is None else v['subgroups'][t]['ci95'];by[c]=dict(every_seed_gain1=all(x>=1 for x in gain),ci_below0=ci[1]<0,pass_gate=all(x>=1 for x in gain) and ci[1]<0)
        evidence_gates[pair]=by
    write(RUN/'SUMMARY.json',dict(complete=True,results=results,comparisons=comparisons,diagnostics=diagnostics,gates=gates,evidence_gates=evidence_gates,stable_arms=[a for a,v in gates.items() if v['all_conditions_pass']],scope='Reused DEV; frozen V18 proposal, supervised continuous retention. No real checkpoint selection. Oracle scales are privileged diagnostics.'))
    write(RUN/'EVALUATION_BINDINGS.json',dict(files=bindings))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['inference','score'],default='inference');a=ap.parse_args();inference() if a.phase=='inference' else scoring()
