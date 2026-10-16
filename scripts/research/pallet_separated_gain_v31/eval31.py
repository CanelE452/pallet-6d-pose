"""GT-free predictions first, then unchanged-population metrics and diagnostics."""
from context31 import *
from data31 import training,fresh
import data30
from prepare31 import load_fresh
import features25
from train31 import load_model
from sklearn.metrics import roc_auc_score
SPLITS=('real','prospective','synth_val','fresh_calibration','fresh_check')
TRAINED=ARMS+('selector_old','selector_new')
INTERVENTIONS=('split_line_off','split_donor','split_rank_only','split_mean_only')

def conditions(split):return TRAIN_CONDITIONS if split.startswith('fresh') else data30.conditions(split)
def population(split,c):
    if not split.startswith('fresh'):return data30.population(split,c)
    m=read(DATA/'cache/new_check'/c/'manifest.json');ts=read(DATA/'targets'/f'new_check_{c}.json')['rows']
    ids={r['id'] for r in read(DATA/'POPULATIONS.json')['new_check'][slice(0,64) if split=='fresh_calibration' else slice(64,None)]}
    return dict(rows=[r for r in m['rows'] if r['id'] in ids]),[t for t in ts if t['id'] in ids]
def inputs(split,c,seed):
    if not split.startswith('fresh'):
        previous=features25.RUN
        try:
            features25.RUN=PREV25.RUN
            return data30.evaluation_data(split,c,seed)[:3]
        finally:features25.RUN=previous
    m,O=load_fresh('new_check',c);features25.RUN=RUN;O['edge']=features25.edges_for_manifest(m,'new_check_'+c)
    S=torch.load(DATA/'sources/new_check'/c/f'seed{seed}.pt',weights_only=True,map_location='cpu');pop,_=population(split,c);ids={r['id'] for r in pop['rows']}
    ix=torch.tensor([i for i,r in enumerate([r for r in m['rows'] if r['detected']]) if r['id'] in ids]);return pop,subset(O,ix),subset(S,ix)

def forward(model,arm,O,S):
    if arm.startswith('selector'):return model(O,S)
    if arm.startswith('joint'):return model(O,S,'normal_gain')
    out=model(O,S,'generic' if arm.startswith('generic') else 'split',line_off=arm=='split_line_off')
    if arm in ('split_rank_only','split_mean_only'):
        score=out['rank'] if arm=='split_rank_only' else out['gain']
        score=score.masked_fill(~out['allowed'],-1e9);ix=score.argmax(-1);apply=(score.max(-1).values>0)
        points=out['candidate_points'].gather(2,ix[...,None,None].expand(-1,-1,1,2))[:,:,0]
        out['points']=torch.where(apply[...,None],points,out['parent_points']);out['chosen']=ix;out['applied']=apply
    return out

def save(split,c,name,m,out):
    folder=RUN/'predictions'/split/c;folder.mkdir(parents=True,exist_ok=True);det=np.array([r['detected'] for r in m['rows']]);d=dict(ids=np.array([r['id'] for r in m['rows']]),detected=det)
    for k,x in out.items():
        x=x.detach().cpu().numpy();a=np.full((len(det),)+x.shape[1:],np.nan,np.float32);a[det]=x;d[k]=a
    np.savez_compressed(folder/f'{name}.npz',**d)

def infer():
    setup();assert read(RUN/'TRAINING_COMPLETE.json')['updates']==27000
    for seed in SEEDS:
        models={a:load_model(a,seed) for a in TRAINED}
        for split in SPLITS:
            for c in conditions(split):
                marker=RUN/'predictions'/split/c/f'DONE_seed{seed}.json'
                if marker.exists():continue
                m,O,S=inputs(split,c,seed);O={k:v.cuda() for k,v in O.items()};S={k:v.cuda() for k,v in S.items()}
                for a,pts in [('base',S['base']),('strong',S['strong']),('stock',O['points'])]:save(split,c,f'{a}_seed{seed}',m,dict(points=pts,original_points=O['points']))
                for arm in TRAINED+(INTERVENTIONS if split=='real' else ()):
                    model=models[arm if arm in models else 'split_new'];parts=[]
                    SS=dict(S)
                    if arm=='split_donor':
                        for k in ('lines','mass','expert_points'):SS[k]=SS[k].roll(1,0)
                    with torch.no_grad():
                        for i in range(0,len(O['box']),32):
                            sl=slice(i,i+32);out=forward(model,arm,subset(O,sl),subset(SS,sl))
                            assert torch.isfinite(out['points']).all();parts.append({k:v.cpu() for k,v in out.items()})
                    save(split,c,f'{arm}_seed{seed}',m,dict(original_points=O['points'],**{k:torch.cat([p[k] for p in parts]) for k in parts[0]}))
                write(marker,dict(complete=True,GT_forward=False,state_forward=False));print('INFER31',seed,split,c,flush=True)
                del O,S
        del models
    write(RUN/'INFERENCE_COMPLETE.json',dict(complete=True,GT_forward=False))

def diagnostic(p,gt,valid,kind):
    finite=np.isfinite(p['points']).all(-1);rows={}
    for typ,k in [('all',None),('hidden',1),('outside',2)]:
        mask=valid&finite
        if k is not None:mask &= kind==k
        if not mask.any():continue
        r=dict(points=int(mask.sum()),mean_px=float(np.linalg.norm(p['points']-gt,axis=-1)[mask].mean()))
        if 'parent_points' in p:
            anchor=p['parent_points'];ea=np.linalg.norm(anchor-gt,axis=-1);ef=np.linalg.norm(p['points']-gt,axis=-1)
            eb=np.linalg.norm(p['anchors'][:,:,0]-gt,axis=-1);es=np.linalg.norm(p['anchors'][:,:,1]-gt,axis=-1)
            actual=np.linalg.norm(p['anchors']-gt[:,:,None],axis=-1)-np.linalg.norm(p['candidate_points']-gt[:,:,None],axis=-1)
            alt=mask[:,:,None]&(p['allowed']==1)&(p['move']>.1);y=actual[alt]>0
            idx=np.nan_to_num(p['chosen']).astype(int);pred=np.take_along_axis(p['gain'],idx[...,None],-1)[...,0];applied=mask&(p['applied']==1)
            oracle=np.minimum(ea,np.where(p['allowed']==1,np.linalg.norm(p['candidate_points']-gt[:,:,None],axis=-1),np.inf).min(-1))
            r.update(parent_px=float(ea[mask].mean()),best_parent_GT_px=float(np.minimum(eb,es)[mask].mean()),parent_regret_px=float((ea-np.minimum(eb,es))[mask].mean()),parent_help_auc=float(roc_auc_score((eb-es)[mask]>0,p['parent_gain'][mask])) if len(set(((eb-es)[mask]>0).tolist()))>1 else None,applied_fraction=float(applied.sum()/mask.sum()),applied_points=int(applied.sum()),actual_applied_gain_px=float((ea-ef)[applied].mean()) if applied.any() else None,predicted_applied_gain_px=float(pred[applied].mean()) if applied.any() else None,harm_over1px=int((applied&(ef>ea+1)).sum()),candidate_gain_auc=float(roc_auc_score(y,p['gain'][alt])) if len(set(y.tolist()))>1 else None,lower_coverage=float((actual[alt]>=p['lower'][alt]-1e-4).mean()) if alt.any() else None,oracle_candidates_GT_px=float(oracle[mask].mean()),gain_bound_violations=int((alt&(np.abs(p['gain'])>p['move']+1e-3)).sum()))
        rows[typ]=r
    return rows

def score():
    results={};comparisons={};diagnostics={};bindings={}
    for split in SPLITS:
        for c in conditions(split):
            m,ts=population(split,c);gt=np.array([t['points'] for t in ts]);valid=np.array([t['valid'] for t in ts]);kind=C.P.target_type(ts).numpy();rows={};arrays={};ds={}
            for seed in SEEDS:
                for a in ('base','strong','stock')+TRAINED+(INTERVENTIONS if split=='real' else ()):
                    name=f'{a}_seed{seed}';path=RUN/'predictions'/split/c/f'{name}.npz';rows[name],arrays[name]=E.score_file(path,m,ts);bindings[str(path.relative_to(ROOT))]=sha(path)
                    if a in TRAINED or a in INTERVENTIONS:ds[name]=diagnostic(np.load(path),gt,valid,kind)
                if not split.startswith('fresh'):
                    for a,old in [('v30','normal_gain'),('v18','line_vector')]:
                        path=(OLD30.RUN if a=='v30' else OLD27.RUN)/'predictions'/split/c/f'{old}_seed{seed}.npz';rows[f'{a}_seed{seed}'],arrays[f'{a}_seed{seed}']=E.score_file(path,m,ts)
            comp={}
            if split=='real':
                arrays['stock']=arrays['stock_seed1'];rows['stock']=rows['stock_seed1']
                pairs=[('split_new','selector_new'),('split_new','generic_new'),('split_new','split_old'),('joint_new','joint_old'),('split_new','joint_new'),('split_old','joint_old'),('split_new','split_donor'),('split_new','split_rank_only'),('split_new','split_mean_only')]
                for a,b in pairs:
                    v=COMPARE(arrays,rows,a,b,[t['session'] for t in ts],split);v['subgroups']={typ:COMPARE(arrays,rows,a,b,[t['session'] for t in ts],split,mask=valid&(kind==k)) for typ,k in [('hidden',1),('outside',2)]};comp[a+'_vs_'+b]=v
            results[split+'/'+c]=rows;diagnostics[split+'/'+c]=ds;comparisons[split+'/'+c]=comp;print('SCORE31',split,c,flush=True)
    gates={}
    for arm in TRAINED:
        per={}
        for seed in SEEDS:
            a=f'{arm}_seed{seed}';b=f'base_seed{seed}';old=f'v18_seed{seed}';rr=results['real/clean'];g=dict(clean_within1=rr[a]['frame_mean_px']<=1.01*rr[b]['frame_mean_px'])
            for c in ('crop10','crop20','crop30'):
                rr=results['real/'+c];g[c+'_within1']=rr[a]['strata']['outside']['mean_px']<=1.01*rr[b]['strata']['outside']['mean_px']
            for c,t in [('corner20','hidden'),('crop40','outside')]:
                rr=results['real/'+c];g[c+'_retain90']=rr[b]['strata'][t]['mean_px']-rr[a]['strata'][t]['mean_px']>=.9*(rr[b]['strata'][t]['mean_px']-rr[old]['strata'][t]['mean_px'])
            per[str(seed)]=g
        gates[arm]=dict(per_seed=per,pass_gate=all(all(x.values()) for x in per.values()))
    evidence={}
    for b in ('selector_new','generic_new'):
        r=comparisons['real/crop20']['split_new_vs_'+b]['subgroups']['outside'];evidence[b]=dict(per_seed_gain_percent=r['per_seed_gain_percent'],ci95=r['ci95'],pass_gate=min(r['per_seed_gain_percent'])>=1 and r['ci95'][1]<0)
    accepted=gates['split_new']['pass_gate'] and all(x['pass_gate'] for x in evidence.values())
    write(RUN/'SUMMARY.json',dict(complete=True,accepted=accepted,results=results,comparisons=comparisons,diagnostics=diagnostics,gates=gates,line_evidence=evidence));write(RUN/'EVALUATION_BINDINGS.json',dict(files=bindings));print('VERDICT31',accepted,flush=True)

def train_diagnosis():
    setup();rows=[]
    for seed in SEEDS:
        for dataset in ('old','new'):
            O,S,g,v,ids,kind=training(seed,dataset);O={k:x.cuda() for k,x in O.items()};S={k:x.cuda() for k,x in S.items()}
            for arm in ('split_'+dataset,):
                model=load_model(arm,seed);parts=[]
                with torch.no_grad():
                    for i in range(0,len(ids),32):
                        sl=slice(i,i+32);out=forward(model,arm,subset(O,sl),subset(S,sl));parts.append({k:x.cpu().numpy() for k,x in out.items()})
                out={k:np.concatenate([p[k] for p in parts]) for k in parts[0]};rows.append(dict(dataset=dataset,seed=seed,stats=diagnostic(out,g.numpy(),v.numpy(),kind.numpy())));del model
            del O,S
    write(RUN/'TRAIN_DIAGNOSIS.json',dict(complete=True,rows=rows,no_retraining=True))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=('infer','score','diagnose'),required=True);a=p.parse_args();{'infer':infer,'score':score,'diagnose':train_diagnosis}[a.phase]()
