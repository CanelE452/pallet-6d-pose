from context36 import *
from train36 import load_model
from data32 import population,conditions
from eval32 import SPLITS

def predict(model,x,arm,intervention=None,donor=None):
    parts=[]
    with torch.no_grad():
        for i in range(0,len(x['box']),32):
            sl=slice(i,i+32);out=model(subset(x,sl),arm,intervention,subset(donor,sl) if donor else None);parts.append({k:out[k].cpu() for k in ('points','normal','offset','loc','feature','score','attention')})
    return {k:torch.cat([p[k] for p in parts]) for k in parts[0]}

def save(out,path,m,old):
    det=np.array([r['detected'] for r in m['rows']]);z=dict(ids=old['ids'],detected=det,original_points=old['original_points']);assert old['ids'].tolist()==[r['id'] for r in m['rows']]
    for k in ('points','normal','offset','score','attention'):
        v=out[k].numpy();a=np.full((len(det),)+v.shape[1:],np.nan,np.float32);a[det]=v;z[k]=a
    path.parent.mkdir(exist_ok=True,parents=True);np.savez_compressed(path,**z)

def infer():
    setup36();assert read(RUN/'TRAINING_COMPLETE.json')['complete']
    for seed in SEEDS:
        models={a:load_model(a,seed) for a in ARMS}
        for split in SPLITS:
            for c in conditions(split):
                cache=torch.load(RUN/'cache'/split/c/f'seed{seed}.pt',weights_only=True,map_location='cpu');x={k:v.cuda() for k,v in cache['x'].items()};m,_=population(split,c);old=np.load(V34.RUN/'predictions'/split/c/f'line_only_r0_seed{seed}.npz')
                for arm,model in models.items():
                    path=RUN/'predictions'/split/c/f'{arm}_seed{seed}.npz';out=predict(model,x,arm);save(out,path,m,old)
                    if split=='real' and c=='crop20' and arm=='both':
                        rows=[r for r in m['rows'] if r['detected']];n=len(rows);donors=[next((i+n//2+j)%n for j in range(n) if rows[(i+n//2+j)%n]['session']!=r['session']) for i,r in enumerate(rows)];donor={k:out[k][donors].cuda() for k in ('normal','offset','loc','feature')}
                        for mode in ('geometry_donor','line_off'):
                            result=predict(model,x,arm,mode,donor if mode=='geometry_donor' else None);save(result,RUN/'predictions'/split/c/f'both_{mode}_seed{seed}.npz',m,old)
                print('INFER36',seed,split,c,flush=True)
        cache=torch.load(RUN/'cache'/f'training_seed{seed}.pt',weights_only=True,map_location='cpu');ix=cache['x']['supervised'];x={k:v[ix].cuda() for k,v in cache['x'].items()};ids=np.array(cache['ids'])[ix.numpy()]
        for arm,model in models.items():
            out=predict(model,x,arm);path=RUN/'training_predictions'/f'{arm}_seed{seed}.npz';path.parent.mkdir(exist_ok=True);np.savez_compressed(path,ids=ids,**{k:out[k].numpy() for k in ('points','normal','offset','score','attention')})
        del models
    files=list((RUN/'predictions').rglob('*.npz'))+list((RUN/'training_predictions').glob('*.npz'));write(RUN/'INFERENCE_COMPLETE.json',dict(complete=True,files={str(p.relative_to(ROOT)):sha(p) for p in files},GT_forward=False))

def score():
    old=read(V34.RUN/'SUMMARY.json');results={};comparisons={}
    for split in SPLITS:
        for c in conditions(split):
            m,ts=population(split,c);rows={};arrays={};key=split+'/'+c
            for seed in SEEDS:
                for arm in list(ARMS)+(['both_geometry_donor','both_line_off'] if split=='real' and c=='crop20' else []):
                    name=f'{arm}_seed{seed}';rows[name],arrays[name]=E.score_file(RUN/'predictions'/split/c/(name+'.npz'),m,ts)
                for a in ('base','strong','v18','v30'):
                    name=f'{a}_seed{seed}'
                    if name in old['results'][key]:rows[name]=old['results'][key][name]
                rows[f'v34_seed{seed}']=old['results'][key][f'line_only_r0_seed{seed}']
            results[key]=rows
            if split=='real':
                arrays['stock']=arrays['grid_seed1'];rows['stock']=rows['grid_seed1'];valid=np.array([t['valid'] for t in ts]);comp={}
                for arm in ARMS[1:]:comp[arm]=COMPARE(arrays,rows,arm,'grid',[t['session'] for t in ts],split,mask=valid&np.array([t['outside'] for t in ts]))
                if c=='crop20':
                    for a in ('both_geometry_donor','both_line_off'):comp[a]=COMPARE(arrays,rows,a,'both',[t['session'] for t in ts],split,mask=valid&np.array([t['outside'] for t in ts]))
                comparisons[key]=comp
    gates={}
    for arm in ARMS:
        per={}
        for seed in SEEDS:
            a=f'{arm}_seed{seed}';b=f'base_seed{seed}';rr=results['real/clean'];g=dict(clean_within1=rr[a]['frame_mean_px']<=1.01*rr[b]['frame_mean_px'])
            for c in ('crop10','crop20','crop30'):
                rr=results['real/'+c];g[c+'_within1']=rr[a]['strata']['outside']['mean_px']<=1.01*rr[b]['strata']['outside']['mean_px']
            for c,t in [('corner20','hidden'),('crop40','outside')]:
                rr=results['real/'+c];base=rr[b]['strata'][t]['mean_px'];g[c+'_retain90']=base-rr[a]['strata'][t]['mean_px']>=.9*(base-rr[f'v18_seed{seed}']['strata'][t]['mean_px'])
            per[str(seed)]=g
        gates[arm]=dict(per_seed=per,pass_gate=all(all(x.values()) for x in per.values()))
    r=comparisons['real/crop20']['both'];added=min(r['per_seed_gain_percent'])>=1 and r['ci95'][1]<0;write(RUN/'POINT_SUMMARY.json',dict(complete=True,results=results,comparisons=comparisons,gates=gates,added_point_effect=bool(added),point_stable=bool(gates['both']['pass_gate'] and added)));print('POINT36',gates['both']['pass_gate'],added,flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['infer','score'],required=True);a=p.parse_args();dict(infer=infer,score=score)[a.phase]()
