from context32 import *
from data32 import inputs,population,conditions,render_data,training
from train32 import load_model
SPLITS=('real','prospective','fresh_check','render_check')
INTERVENTIONS=('tokens_off','donor','hough_off','endpoints_off')

def raw_lines(hough,box):
    B=len(box);ix=hough.flatten(-2).argmax(-1);theta=(ix//48).float()*torch.pi/24;rho=(ix%48).float()*2*2**.5/47-2**.5;center=(box[:,:2]+box[:,2:])/2;half=(box[:,2:]-box[:,:2]).clamp_min(1)/2
    a=theta.cos()/half[:,0,None];b=theta.sin()/half[:,1,None];c=-rho-a*center[:,0,None]-b*center[:,1,None];lines=torch.stack((a,b,c),-1);return lines/lines[...,:2].norm(dim=-1,keepdim=True).clamp_min(1e-9)

def save(split,c,name,m,out):
    folder=RUN/'predictions'/split/c;folder.mkdir(parents=True,exist_ok=True);det=np.array([r['detected'] for r in m['rows']]);d=dict(ids=np.array([r['id'] for r in m['rows']]),detected=det)
    for k,x in out.items():
        x=x.detach().cpu().numpy();dtype=np.float16 if k=='role_maps' else np.float32;a=np.full((len(det),)+x.shape[1:],np.nan,dtype);a[det]=x;d[k]=a
    np.savez_compressed(folder/f'{name}.npz',**d)

def infer():
    setup();assert read(RUN/'TRAINING_COMPLETE.json')['updates']==24000
    for seed in SEEDS:
        models={a:load_model(a,seed) for a in ARMS}
        for split in SPLITS:
            for c in conditions(split):
                marker=RUN/'predictions'/split/c/f'DONE_seed{seed}.json'
                if marker.exists():continue
                m,O,S=inputs(split,c,seed)
                original=O['points'] if 'points' in O else render_data(seed,'check',c)[1]['points']
                O={k:x.cuda() for k,x in O.items()};S={k:x.cuda() for k,x in S.items()}
                for a,pts in [('base',S['base']),('strong',S['strong']),('stock',O.get('points',S['base']))]:
                    if a=='stock' and 'points' not in O:continue
                    save(split,c,f'{a}_seed{seed}',m,dict(points=pts,original_points=original))
                for arm in ARMS+(INTERVENTIONS if split=='real' else ()):
                    actual=arm if arm in ARMS else 'visible_hough';model=models[actual];parts=[]
                    with torch.no_grad():
                        for i in range(0,len(O['box']),32):
                            sl=slice(i,i+32);out=model(subset(O,sl),subset(S,sl),actual,intervention=arm if arm in INTERVENTIONS else None);assert torch.isfinite(out['points']).all();out['lines']=raw_lines(out['hough'],O['box'][sl]);keys=['points','base','strong','mix','residual','lines']
                            if split=='render_check' or (seed==1 and arm=='visible_hough' and split=='real' and c in ('clean','corner20','crop20')):keys+=['role_maps']
                            parts.append({k:out[k].cpu() for k in keys})
                    save(split,c,f'{arm}_seed{seed}',m,dict(original_points=original,**{k:torch.cat([p[k] for p in parts]) for k in parts[0]}))
                write(marker,dict(complete=True,GT_forward=False));print('INFER32',seed,split,c,flush=True)
                del O,S
        del models
    write(RUN/'INFERENCE_COMPLETE.json',dict(complete=True,GT_forward=False))

def role_metrics(prob,gt):
    out={}
    for name,sl in [('edges',slice(0,12)),('endpoints',slice(12,24))]:
        pred=prob[:,sl]>.5;target=gt[:,sl]>0;tp=int((pred&target).sum());fp=int((pred&~target).sum());fn=int((~pred&target).sum());precision=tp/max(tp+fp,1);recall=tp/max(tp+fn,1)
        out[name]=dict(precision=precision,recall=recall,f1=2*tp/max(2*tp+fp+fn,1),tp=tp,fp=fp,fn=fn,threshold=.5)
    return out

def score():
    results={};comparisons={};diagnostics={};role={};bindings={}
    for split in SPLITS:
        for c in conditions(split):
            m,ts=population(split,c);gt=np.array([t['points'] for t in ts]);valid=np.array([t['valid'] for t in ts]);kind=C.P.target_type(ts).numpy();rows={};arrays={};ds={}
            for seed in SEEDS:
                for arm in ('base','strong')+ARMS+(INTERVENTIONS if split=='real' else ()):
                    name=f'{arm}_seed{seed}';path=RUN/'predictions'/split/c/f'{name}.npz';rows[name],arrays[name]=E.score_file(path,m,ts);bindings[str(path.relative_to(ROOT))]=sha(path)
                    if arm not in ARMS and arm not in INTERVENTIONS:continue
                    p=np.load(path);eb=np.linalg.norm(p['base']-gt,axis=-1);ef=np.linalg.norm(p['points']-gt,axis=-1);dd={}
                    for typ,k in [('all',None),('hidden',1),('outside',2)]:
                        mask=valid&p['detected'][:,None]
                        if k is not None:mask &= kind==k
                        if not mask.any():continue
                        dd[typ]=dict(points=int(mask.sum()),mean_px=float(ef[mask].mean()),harm_over1px=int((mask&(ef>eb+1)).sum()),base5_to20=int((mask&(eb<=5)&(ef>20)).sum()),mean_movement_px=float(np.linalg.norm(p['points']-p['base'],axis=-1)[mask].mean()),mean_mix=float(p['mix'][mask].mean()))
                    ds[name]=dd
                    if split=='render_check':
                        _,obs,_,_=render_data(seed,'check',c);role[f'{split}/{c}/{name}']=role_metrics(p['role_maps'][p['detected']].astype(float),obs['physical_target'].numpy())
                if split!='render_check':
                    for name,run,arm in [('v30',OLD30.RUN,'normal_gain'),('v31',OLD31.RUN,'split_new'),('v18',OLD27.RUN,'line_vector')]:
                        path=run/'predictions'/split/c/f'{arm}_seed{seed}.npz'
                        if path.exists():rows[f'{name}_seed{seed}'],arrays[f'{name}_seed{seed}']=E.score_file(path,m,ts)
                else:
                    # V18 rendered baseline is read from frozen-parent cache, never fitted here.
                    mm,obs,source,_=render_data(seed,'check',c);save(split,c,f'v18_seed{seed}',mm,dict(points=source['v18'],original_points=obs['points']));path=RUN/'predictions'/split/c/f'v18_seed{seed}.npz';rows[f'v18_seed{seed}'],arrays[f'v18_seed{seed}']=E.score_file(path,m,ts);bindings[str(path.relative_to(ROOT))]=sha(path)
            comp={}
            if split=='real':
                arrays['stock']=arrays['base_seed1'];rows['stock']=rows['base_seed1']
                for a,b in [('visible_hough','point'),('visible_hough','virtual_hough'),('visible_hough','latent_hough')]+[('visible_hough',b) for b in INTERVENTIONS]:
                    value=COMPARE(arrays,rows,a,b,[t['session'] for t in ts],split);value['subgroups']={typ:COMPARE(arrays,rows,a,b,[t['session'] for t in ts],split,mask=valid&(kind==k)) for typ,k in [('hidden',1),('outside',2)]};comp[a+'_vs_'+b]=value
            results[split+'/'+c]=rows;comparisons[split+'/'+c]=comp;diagnostics[split+'/'+c]=ds;print('SCORE32',split,c,flush=True)
    gates={}
    for arm in ARMS:
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
    for b in ('point','virtual_hough'):
        v=comparisons['real/crop20']['visible_hough_vs_'+b]['subgroups']['outside'];evidence[b]=dict(per_seed_gain_percent=v['per_seed_gain_percent'],ci95=v['ci95'],pass_gate=min(v['per_seed_gain_percent'])>=1 and v['ci95'][1]<0)
    accepted=gates['visible_hough']['pass_gate'] and all(v['pass_gate'] for v in evidence.values())
    write(RUN/'SUMMARY.json',dict(complete=True,accepted=accepted,results=results,comparisons=comparisons,diagnostics=diagnostics,gates=gates,line_evidence=evidence));write(RUN/'ROLE_METRICS.json',dict(rows=role,GT='Physical rendered48x48role-specific raster masks; threshold0.5 fixed; pixel correlations, partial role definitions. No real line accuracy labels.'));write(RUN/'EVALUATION_BINDINGS.json',dict(files=bindings));print('VERDICT32',accepted,flush=True)

def diagnose():
    setup();rows=[]
    for seed in SEEDS:
        O,S,g,v,physical,virtual,sup,ids,n=training(seed);O={k:x.cuda() for k,x in O.items()};S={k:x.cuda() for k,x in S.items()}
        for arm in ARMS:
            model=load_model(arm,seed);parts=[];prob=[]
            with torch.no_grad():
                for i in range(0,len(ids),32):
                    sl=slice(i,i+32);out=model(subset(O,sl),subset(S,sl),arm);parts.append(out['points'].cpu());prob.append(out['role_maps'].cpu())
            points=torch.cat(parts);maps=torch.cat(prob);e=(points-g).norm(dim=-1)
            rows.append(dict(seed=seed,arm=arm,legacy_mean_px=float(e[:n][v[:n]].mean()),render_mean_px=float(e[n:][v[n:]].mean()),physical_role=role_metrics(maps[n:].numpy(),physical[n:].numpy())));del model
        del O,S
    write(RUN/'TRAIN_DIAGNOSIS.json',dict(complete=True,rows=rows));print('DIAGNOSE32 COMPLETE',flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['infer','score','diagnose'],required=True);a=p.parse_args();{'infer':infer,'score':score,'diagnose':diagnose}[a.phase]()
