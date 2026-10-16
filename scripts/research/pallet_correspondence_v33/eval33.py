from context33 import *
from data32 import inputs,population,conditions,render_data,training
from train33 import load_model
from eval32 import raw_lines,role_metrics,SPLITS
INTERVENTIONS=('line_off','end_off','both_off','cross_donor')

def save(split,c,name,m,out):
    folder=RUN/'predictions'/split/c;folder.mkdir(parents=True,exist_ok=True);det=np.array([r['detected'] for r in m['rows']]);d=dict(ids=np.array([r['id'] for r in m['rows']]),detected=det)
    for k,x in out.items():
        x=x.detach().cpu().numpy();dtype=np.float16 if k in ('role_maps','line_attention','end_attention') else np.float32;a=np.full((len(det),)+x.shape[1:],np.nan,dtype);a[det]=x;d[k]=a
    np.savez_compressed(folder/f'{name}.npz',**d)

def infer():
    setup();assert read(RUN/'TRAINING_COMPLETE.json')['updates']==36000;donors=[]
    for seed in SEEDS:
        models={a:load_model(a,seed) for a in ARMS}
        for split in SPLITS:
            for c in conditions(split):
                marker=RUN/'predictions'/split/c/f'DONE_seed{seed}.json'
                if marker.exists():continue
                m,O,S=inputs(split,c,seed);original=O.get('points')
                if original is None:original=render_data(seed,'check',c)[1]['points']
                O={k:x.cuda() for k,x in O.items()};S={k:x.cuda() for k,x in S.items()};N=len(O['box']);maps=[]
                if split=='real':
                    sessions=[r['session'] for r in m['rows'] if r['detected']];donor=[next((i+N//2+k)%N for k in range(N) if sessions[(i+N//2+k)%N]!=sessions[i]) for i in range(N)];donors.append(dict(seed=seed,condition=c,indices=donor,all_different_session=all(sessions[i]!=sessions[j] for i,j in enumerate(donor))))
                    with torch.no_grad():
                        for i in range(0,N,32):maps.append(models['query'](subset(O,slice(i,i+32)),subset(S,slice(i,i+32)))['role_maps'])
                    maps=torch.cat(maps)
                for arm in ARMS+(INTERVENTIONS if split=='real' else ()):
                    model=models[arm if arm in ARMS else 'query'];parts=[]
                    with torch.no_grad():
                        for i in range(0,N,32):
                            sl=slice(i,i+32);out=model(subset(O,sl),subset(S,sl),arm if arm in ARMS else 'query',intervention=arm if arm in INTERVENTIONS else None,donor_maps=maps[donor[i:i+32]] if arm=='cross_donor' else None);out['lines']=raw_lines(out['hough'],O['box'][sl]);keys=['points','base','strong','mix','residual','lines']
                            if 'line_attention' in out:
                                out['line_use']=(1-out['line_attention'][...,-1]).reshape(-1,8,2);out['end_use']=(1-out['end_attention'][...,-1]).reshape(-1,8,2);keys+=['line_use','end_use']
                            if split=='render_check':keys+=['role_maps']
                            if seed==1 and arm=='query' and split=='real' and c in ('clean','corner20','crop20'):
                                keys+=['role_maps','line_attention','end_attention','line_normal','line_offset','line_support','line_spread','endpoint_locations']
                            parts.append({k:out[k].cpu() for k in keys})
                    save(split,c,f'{arm}_seed{seed}',m,dict(original_points=original,**{k:torch.cat([p[k] for p in parts]) for k in parts[0]}))
                write(marker,dict(complete=True,GT_forward=False));print('INFER33',seed,split,c,flush=True)
        del models
    write(RUN/'INFERENCE_COMPLETE.json',dict(complete=True,GT_forward=False,cross_session_donors=donors))

def score():
    s32=read(V32.RUN/'SUMMARY.json');results={};comp={};roles={};bindings={}
    for split in SPLITS:
        for c in conditions(split):
            m,ts=population(split,c);rr={};arrays={};key=split+'/'+c
            for seed in SEEDS:
                for a in ARMS+(INTERVENTIONS if split=='real' else ()):
                    name=f'{a}_seed{seed}';p=RUN/'predictions'/split/c/f'{name}.npz';rr[name],arrays[name]=E.score_file(p,m,ts);bindings[str(p.relative_to(ROOT))]=sha(p)
                    if split=='render_check':
                        z=np.load(p);_,O,_,_=render_data(seed,'check',c);roles[key+'/'+name]=role_metrics(z['role_maps'][z['detected']].astype(float),O['physical_target'].numpy())
                for a in ('base','strong','v18','v30','v31','visible_hough'):
                    name=f'{a}_seed{seed}'
                    if name in s32['results'][key]:rr[name]=s32['results'][key][name]
                p=V32.RUN/'predictions'/split/c/f'base_seed{seed}.npz';_,arrays[f'base_seed{seed}']=E.score_file(p,m,ts)
            results[key]=rr
            if split=='real':
                arrays['stock']=arrays['base_seed1'];rr['stock']=rr['base_seed1'];valid=np.array([t['valid'] for t in ts]);cc={}
                for b in ('pooled','end_only','line_only','no_geometry','point_aux')+INTERVENTIONS:
                    v=COMPARE(arrays,rr,'query',b,[t['session'] for t in ts],split);v['subgroups']={typ:COMPARE(arrays,rr,'query',b,[t['session'] for t in ts],split,mask=valid&np.array([t[typ] for t in ts])) for typ in ('hidden','outside')};cc['query_vs_'+b]=v
                comp[key]=cc
            print('SCORE33',key,flush=True)
    gates={}
    for a in ARMS:
        per={}
        for seed in SEEDS:
            name=f'{a}_seed{seed}';b=f'base_seed{seed}';rr=results['real/clean'];g=dict(clean_within1=rr[name]['frame_mean_px']<=1.01*rr[b]['frame_mean_px'])
            for c in ('crop10','crop20','crop30'):
                rr=results['real/'+c];g[c+'_within1']=rr[name]['strata']['outside']['mean_px']<=1.01*rr[b]['strata']['outside']['mean_px']
            for c,t in [('corner20','hidden'),('crop40','outside')]:
                rr=results['real/'+c];base=rr[b]['strata'][t]['mean_px'];g[c+'_retain90']=base-rr[name]['strata'][t]['mean_px']>=.9*(base-rr[f'v18_seed{seed}']['strata'][t]['mean_px'])
            per[str(seed)]=g
        gates[a]=dict(per_seed=per,pass_gate=all(all(v.values()) for v in per.values()))
    evidence={}
    for b in ('pooled','end_only'):
        r=comp['real/crop20']['query_vs_'+b]['subgroups']['outside'];evidence[b]=dict(per_seed_gain_percent=r['per_seed_gain_percent'],ci95=r['ci95'],pass_gate=min(r['per_seed_gain_percent'])>=1 and r['ci95'][1]<0)
    accepted=gates['query']['pass_gate'] and all(v['pass_gate'] for v in evidence.values());write(RUN/'SUMMARY.json',dict(complete=True,accepted=accepted,results=results,comparisons=comp,gates=gates,line_evidence=evidence));write(RUN/'ROLE_METRICS.json',dict(rows=roles));write(RUN/'EVALUATION_BINDINGS.json',dict(files=bindings));print('VERDICT33',accepted,flush=True)

def diagnose():
    setup();rows=[]
    for seed in SEEDS:
        O,S,g,v,phys,virt,sup,ids,n=training(seed);O={k:x.cuda() for k,x in O.items()};S={k:x.cuda() for k,x in S.items()}
        for a in ARMS:
            model=load_model(a,seed);points=[];lu=[];eu=[]
            with torch.no_grad():
                for i in range(0,len(ids),32):
                    out=model(subset(O,slice(i,i+32)),subset(S,slice(i,i+32)),a);points.append(out['points'].cpu())
                    if 'line_attention' in out:lu.append((1-out['line_attention'][...,-1]).cpu());eu.append((1-out['end_attention'][...,-1]).cpu())
            error=(torch.cat(points)-g).norm(dim=-1);row=dict(seed=seed,arm=a,legacy_mean_px=float(error[:n][v[:n]].mean()),render_mean_px=float(error[n:][v[n:]].mean()))
            if lu:row.update(line_nonnull=float(torch.cat(lu).mean()),end_nonnull=float(torch.cat(eu).mean()))
            # Fixed first renderer-containing batch; shared-trunk gradient alignment diagnostic.
            ix=read(V32.RUN/f'PLAN_seed{seed}.json')['plan'][0];ii=torch.tensor(ix,device='cuda');out=model(subset(O,ii),subset(S,ii),a);_,terms=__import__('model33').objective(out,g[ii.cpu()].cuda(),v[ii.cpu()].cuda(),phys[ii.cpu()].cuda(),sup[ii.cpu()].cuda(),'visible_hough');params=list(model.trunk.parameters());pg=torch.autograd.grad(terms['point'],params,retain_graph=True);ag=torch.autograd.grad(terms['aux'],params);p=torch.cat([x.flatten() for x in pg]);q=torch.cat([x.flatten() for x in ag]);row['one_batch_gradient_cosine']=float((p@q)/(p.norm()*q.norm()).clamp_min(1e-12));rows.append(row);del model
        del O,S
    write(RUN/'TRAIN_DIAGNOSIS.json',dict(complete=True,rows=rows,gradient_limits='One preselected mixed batch per final model; diagnostic only, not representative proof of persistent conflict. PCGrad not run.'));print('DIAGNOSE33 COMPLETE',flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['infer','score','diagnose'],required=True);a=p.parse_args();dict(infer=infer,score=score,diagnose=diagnose)[a.phase]()
