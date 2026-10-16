from context34 import *
from data32 import inputs,population,conditions,render_data
from eval32 import SPLITS
from train_dense34 import load_model

def infer():
    deterministic();assert read(RUN/'DENSE_COMPLETE.json')['complete']
    for seed in SEEDS:
        models={f'{a}_r{r}':load_model(a,seed,r) for a in ARMS for r in REPEATS}
        for split in SPLITS:
            for c in conditions(split):
                folder=RUN/'predictions'/split/c;folder.mkdir(parents=True,exist_ok=True)
                if all((folder/f'{name}_seed{seed}.npz').exists() for name in models):continue
                m,O,S=inputs(split,c,seed);original=O.get('points')
                if original is None:original=render_data(seed,'check',c)[1]['points']
                det=np.array([r['detected'] for r in m['rows']]);O={k:v.cuda() for k,v in O.items()};S={k:v.cuda() for k,v in S.items()}
                for name,model in models.items():
                    arm=name.rsplit('_r',1)[0];parts=[]
                    with torch.no_grad():
                        for i in range(0,len(O['box']),32):
                            out=model(subset(O,slice(i,i+32)),subset(S,slice(i,i+32)),arm);parts.append({k:out[k].cpu() for k in ('points','base','strong','mix','residual')})
                    arrays=dict(original_points=original,**{k:torch.cat([p[k] for p in parts]) for k in parts[0]});z=dict(ids=np.array([r['id'] for r in m['rows']]),detected=det)
                    for k,v in arrays.items():
                        v=v.numpy();full=np.full((len(det),)+v.shape[1:],np.nan,np.float32);full[det]=v;z[k]=full
                    np.savez_compressed(folder/f'{name}_seed{seed}.npz',**z)
                print('INFER34',seed,split,c,flush=True)
        del models
    write(RUN/'INFERENCE_COMPLETE.json',dict(complete=True,GT_forward=False))

def score():
    old=read(V33.RUN/'SUMMARY.json');results={};comparisons={};bindings={};repeated=[]
    for split in SPLITS:
        for c in conditions(split):
            m,ts=population(split,c);rows={};arrays={};key=split+'/'+c
            for seed in SEEDS:
                for arm in ARMS:
                    zs=[]
                    for repeat in REPEATS:
                        name=f'{arm}_r{repeat}_seed{seed}';p=RUN/'predictions'/split/c/f'{name}.npz';rows[name],arrays[name]=E.score_file(p,m,ts);bindings[str(p.relative_to(ROOT))]=sha(p);zs.append(np.load(p)['points'])
                    repeated.append(dict(split=split,condition=c,arm=arm,seed=seed,equal=bool(np.array_equal(zs[0],zs[1],equal_nan=True)),max_difference_px=float(np.nanmax(np.abs(zs[0]-zs[1])))))
                for a in ('base','strong','v18','v30','visible_hough'):
                    name=f'{a}_seed{seed}'
                    if name in old['results'][key]:rows[name]=old['results'][key][name]
                for a in ('line_only','pooled'):rows[f'v33_{a}_seed{seed}']=old['results'][key][f'{a}_seed{seed}']
            results[key]=rows
            if split=='real':
                arrays['stock']=arrays['pooled_r0_seed1'];rows['stock']=rows['pooled_r0_seed1'];valid=np.array([t['valid'] for t in ts]);v=COMPARE(arrays,rows,'line_only_r0','pooled_r0',[t['session'] for t in ts],split);v['subgroups']={typ:COMPARE(arrays,rows,'line_only_r0','pooled_r0',[t['session'] for t in ts],split,mask=valid&np.array([t[typ] for t in ts])) for typ in ('hidden','outside')};comparisons[key]=v
    gates={}
    for arm in ARMS:
        per={}
        for seed in SEEDS:
            a=f'{arm}_r0_seed{seed}';b=f'base_seed{seed}';rr=results['real/clean'];g=dict(clean_within1=rr[a]['frame_mean_px']<=1.01*rr[b]['frame_mean_px'])
            for c in ('crop10','crop20','crop30'):
                rr=results['real/'+c];g[c+'_within1']=rr[a]['strata']['outside']['mean_px']<=1.01*rr[b]['strata']['outside']['mean_px']
            for c,t in [('corner20','hidden'),('crop40','outside')]:
                rr=results['real/'+c];base=rr[b]['strata'][t]['mean_px'];g[c+'_retain90']=base-rr[a]['strata'][t]['mean_px']>=.9*(base-rr[f'v18_seed{seed}']['strata'][t]['mean_px'])
            per[str(seed)]=g
        gates[arm]=dict(per_seed=per,pass_gate=all(all(x.values()) for x in per.values()))
    r=comparisons['real/crop20']['subgroups']['outside'];added=min(r['per_seed_gain_percent'])>=1 and r['ci95'][1]<0;rep=read(RUN/'DENSE_REPLICATION.json');rep['prediction_checks']=repeated;rep['all_predictions_equal']=all(r['equal'] for r in repeated);write(RUN/'DENSE_REPLICATION.json',rep);write(RUN/'SUMMARY.json',dict(complete=True,accepted=bool(gates['line_only']['pass_gate'] and added),results=results,comparisons=comparisons,gates=gates,added_line_effect=dict(pass_gate=bool(added),**r)));write(RUN/'EVALUATION_BINDINGS.json',dict(files=bindings));print('SCORE34 COMPLETE',gates['line_only']['pass_gate'],added,rep['all_equal'],rep['all_predictions_equal'],flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['infer','score'],required=True);a=p.parse_args();dict(infer=infer,score=score)[a.phase]()
