"""Immutable V21/V20 baselines and exact original populations."""
import argparse
from context23 import *
from model20 import CauseGate
sys.path.insert(0,str(KIT.parent/'pallet_objective_v22'))
from eval22 import population as prior_population,prediction_path as previous_path
ARMS=ARMS+('unit_mass','unit_mass_local')
METHODS=('stock','base','line_vector','unit_weighted_quality','unit_old_pcgrad','unit_bce')+ARMS
FACTORS=[('unit_quality','unit_expand'),('unit_quality_local','unit_expand_local'),('unit_quality','unit_quality_local'),('unit_expand','unit_expand_local'),('unit_expand','unit_bce'),('unit_mass','unit_expand'),('unit_mass','unit_quality'),('unit_mass','unit_mass_local')]
PAIRS=[(a,b) for a in ARMS for b in ('base','line_vector','unit_bce')]+FACTORS
SPLITS=('prospective','new_check','synth_val','real')
def population(split,c):
    return (read(RUN/'cache/new_check'/c/'manifest.json'),new_targets('new_check',c)) if split=='prospective' else prior_population(split,c)
def prediction_path(split,c,name):
    return RUN/'predictions'/split/c/f'{name}.npz' if split=='prospective' or name.rsplit('_seed',1)[0] in ARMS else previous_path(split,c,name)
def inputs(split,c,seed):
    if split=='prospective':return (*new_load('new_check',c),new_rep('new_check',c,seed))
    if split=='new_check':
        import context21
        return (*context21.new_load(split,c),context21.new_rep(split,c,seed))
    return eval_inputs(split,c,seed)

def inference():
    setup();assert read(RUN/'TRAINING_COMPLETE.json')['complete'];assert read(RUN/'MASS_TRAINING_COMPLETE.json')['complete'];assert not (RUN/'EVAL_PROTOCOL.json').exists();write(RUN/'EVAL_PROTOCOL.json',dict(source_sha256=sha(Path(__file__)),training_sha256=sha(RUN/'TRAINING_COMPLETE.json'),no_GT_forward=True));receipts=[]
    for split in SPLITS:
        for c in (TRAIN_CONDITIONS if split in ('new_check','prospective') else CONDITIONS):
            for seed in SEEDS:
                m,O,r=inputs(split,c,seed)
                O={k:v.cuda() for k,v in O.items()};r={k:v.cuda() for k,v in r.items()}
                if split=='prospective':
                    for a in ('stock','base','line_vector'):
                        if a=='stock' and seed!=1:continue
                        base=O['points'] if a=='stock' else r['base'];d=base*0 if a=='stock' else r['proposal_delta'];alpha=torch.ones_like(base[...,0]) if a=='line_vector' else torch.zeros_like(base[...,0]);save23(split,c,'stock' if a=='stock' else f'{a}_seed{seed}',m,O,dict(points=base+alpha[...,None]*d,base=base,proposal_delta=d,alpha=alpha,coverage=alpha*0))
                for arm in (ARMS+('unit_weighted_quality','unit_old_pcgrad','unit_bce') if split=='prospective' else ARMS):
                    ck=(RUN if arm in ARMS else OLD.RUN if arm=='unit_weighted_quality' else ROOT/'data/pallet/results/pallet_residual_v21' if arm=='unit_old_pcgrad' else ROOT/'data/pallet/results/pallet_objective_v22')/'models'/f'{arm}_seed{seed}'/'final.pt';model=CauseGate(seed).cuda().eval();model.load_state_dict(torch.load(ck,weights_only=True)['state_dict']);parts=[]
                    with torch.inference_mode():
                        for first in range(0,len(O['points']),32):
                            sl=slice(first,first+32);out=model(subset(O,sl),subset(r,sl),arm);parts.append({k:out[k].cpu() for k in ('points','base','proposal_delta','alpha','coverage')})
                    save23(split,c,f'{arm}_seed{seed}',m,O,{k:torch.cat([x[k] for x in parts]) for k in parts[0]});receipts.append(dict(arm=arm,seed=seed,split=split,condition=c,sha256=sha(ck)));del model
            print('INFERRED',split,c,flush=True)
    write(RUN/'INFERENCE_COMPLETE.json',dict(complete=True,rows=receipts,no_GT_forward=True))

def scoring():
    assert read(RUN/'INFERENCE_COMPLETE.json')['complete'];results={};comparisons={};diagnostics={};bindings={}
    for split in SPLITS:
        for condition in (TRAIN_CONDITIONS if split in ('new_check','prospective') else CONDITIONS):
            m,ts=population(split,condition);gt=np.array([t['points'] for t in ts]);valid=np.array([t['valid'] for t in ts]);kind=C.P.target_type(ts).numpy();rows={};arrays={};ds={}
            for arm in METHODS:
                for seed in ([1] if arm=='stock' else SEEDS):
                    name='stock' if arm=='stock' else f'{arm}_seed{seed}';path=prediction_path(split,condition,name);rows[name],arrays[name]=E.score_file(path,m,ts);bindings[str(path.relative_to(ROOT))]=sha(path);pred=np.load(path);det=pred['detected'];assert np.array_equal(det,arrays['stock']['detected'])
                    if arm not in ARMS:continue
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
    for a,b in FACTORS:
        pair=a+'_vs_'+b;by={}
        for c,t in [('clean',None),('corner20','hidden'),('crop20','outside'),('crop40','outside'),('crop10','outside'),('crop30','outside')]:
            v=comparisons['real/'+c][pair];gain=v['relative_gain_percent'] if t is None else v['subgroups'][t]['per_seed_gain_percent'];ci=v['paired_px']['ci95'] if t is None else v['subgroups'][t]['ci95'];by[c]=dict(every_seed_gain1=all(x>=1 for x in gain),ci_below0=ci[1]<0,pass_gate=all(x>=1 for x in gain) and ci[1]<0)
        evidence_gates[pair]=by
    write(RUN/'SUMMARY.json',dict(complete=True,results=results,comparisons=comparisons,diagnostics=diagnostics,gates=gates,evidence_gates=evidence_gates,stable_arms=[a for a,v in gates.items() if v['all_conditions_pass']],scope='Real/synth_val/V21 new_check reused DEV; prospective128 is new to refinement but historical YOLO validation. Intact observations, training-only synthetic quality weighting, frozen V18 proposal. No real checkpoint selection. Oracle scales privileged diagnostics.'))
    write(RUN/'EVALUATION_BINDINGS.json',dict(files=bindings))



if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['inference','score'],required=True);a=ap.parse_args();inference() if a.phase=='inference' else scoring()
