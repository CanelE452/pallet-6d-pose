"""Paired real/synthetic stress evaluation with detection failures retained."""
import time
import numpy as np
import torch
from common import *
from optional_model import OptionalRefiner,predict_lines,donor_lines


def save_predictions(split,condition,name,manifest,O,points,extras=None):
    directory=RUN/'predictions'/split/condition;directory.mkdir(parents=True,exist_ok=True)
    detected=np.array([r['detected'] for r in manifest['rows']]);n=len(detected)
    arr=np.full((n,8,2),np.nan,dtype=np.float32);arr[detected]=points.cpu().numpy()
    original=np.full_like(arr,np.nan);original[detected]=O['points'].cpu().numpy()
    payload=dict(points=arr,original_points=original,detected=detected,ids=np.array([r['id'] for r in manifest['rows']]))
    if extras:
        for k,v in extras.items():
            dest=np.full((n,)+tuple(v.shape[1:]),np.nan,dtype=np.float32);dest[detected]=v.cpu().numpy();payload[k]=dest
    np.savez_compressed(directory/f'{name}.npz',**payload)


def inference():
    setup();assert read(RUN/'TRAINING_COMPLETE.json')['complete']
    proto=RUN/'EVAL_PROTOCOL.json'
    if proto.exists():raise SystemExit('Evaluation already registered; refusing overwrite')
    write(proto,dict(training_protocol_sha256=sha(RUN/'TRAIN_PROTOCOL.json'),source_sha256=sha(Path(__file__)),
        same_input='Every model uses identical current-corruption YOLO boxes, scores, points, ROI features and missing-frame mask.',
        targets='No targets JSON is opened during this function. K/dimensions are deployment metadata stripped into observation cache.',
        interventions='Wrong-image lines mapped by observed ROI affine. Extra-branch off exactly returns preserved V10 local model. All models evaluated; no real checkpoint selection.',
        normalization='Use original image diagonal even after crop, to avoid changing the denominator under truncation. Missing selected detection >=0.25 contributes normalized error1. Pixel errors conditional on selected detection, no GT matching.'))
    audit=[];start=time.monotonic()
    for split in ('synth_val','real'):
        for condition in CONDITIONS:
            manifest,O=load_observations(split,condition,'cuda:0');n=len(O['points'])
            save_predictions(split,condition,'stock',manifest,O,O['points'])
            for seed in SEEDS:
                frozen=frozen_lines(split,condition,seed,'cuda:0');ll,mm=frozen['lines'],frozen['mass']
                base=local(seed);base.requires_grad_(False);bp=[];old=local(seed,arm='directional');op=[]
                with torch.inference_mode():
                    for first in range(0,n,32):
                        sl=slice(first,first+32);obs=subset(O,sl)
                        bp.append(base(obs,ll[sl],mm[sl],'local')['points']);op.append(old(obs,ll[sl],mm[sl],'directional')['points'])
                bp=torch.cat(bp);op=torch.cat(op)
                save_predictions(split,condition,f'v10_local_seed{seed}',manifest,O,bp)
                save_predictions(split,condition,f'v10_directional_seed{seed}',manifest,O,op)
                del base,old,op
                for arm in ARMS:
                    ck=RUN/'models'/f'{arm}_seed{seed}'/'final.pt';state=torch.load(ck,map_location='cpu',weights_only=False)
                    model=OptionalRefiner(seed).cuda().eval();model.load_state_dict(state['state_dict'])
                    l,m=ll,mm
                    if arm=='lines_joint':
                        line=dht(seed);line.load_state_dict(state['dht_state_dict']);ls=[];ms=[]
                        with torch.inference_mode():
                            for first in range(0,n,32):
                                a,b=predict_lines(line,subset(O,slice(first,first+32)));ls.append(a);ms.append(b)
                        l,m=torch.cat(ls),torch.cat(ms);del line
                    variants=('actual',) if arm=='local_extra' else ('actual','donor')
                    detected_ids=[r['id'] for r in manifest['rows'] if r['detected']]
                    order=sorted(range(n),key=lambda i:digest('v11-donor:'+detected_ids[i]))
                    indices=np.zeros(n,dtype=np.int64)
                    for j,i in enumerate(order):indices[i]=order[(j+1)%n]
                    dl,dm=donor_lines(l,m,O['box'],torch.tensor(indices,device='cuda:0'))
                    for variant in variants:
                        ls,ms=(l,m) if variant=='actual' else (dl,dm)
                        pieces=[];max_off=0.
                        with torch.inference_mode():
                            for first in range(0,n,32):
                                sl=slice(first,first+32);obs=subset(O,sl)
                                p=model(obs,bp[sl],ls[sl],ms[sl],arm)
                                off=model(obs,bp[sl],ls[sl],ms[sl],arm,off=True)
                                assert torch.equal(off['points'],bp[sl])
                                assert torch.isfinite(p['points']).all()
                                pieces.append({k:v.cpu() for k,v in p.items()})
                        merged={k:torch.cat([p[k] for p in pieces]) for k in pieces[0]}
                        # Replay the additive directional path independently in report.py.
                        merged['lines']=ls.cpu()
                        name=f'{arm}_seed{seed}'+('_donor' if variant=='donor' else '')
                        save_predictions(split,condition,name,manifest,O,merged.pop('points'),merged)
                        audit.append(dict(split=split,condition=condition,name=name,frames=n,off_base_max_error_px=0.,checkpoint_sha256=sha(ck)))
                    del model
                print('INFERRED',split,condition,'seed',seed,'frames',n,'elapsed',round(time.monotonic()-start),flush=True)
            del O
    write(RUN/'INFERENCE_COMPLETE.json',dict(complete=True,rows=audit,targets_used_in_inference=False,elapsed_seconds=time.monotonic()-start))


def score_file(path,manifest,targets):
    p=np.load(path);gt=np.array([r['points'] for r in targets]);valid=np.array([r['valid'] for r in targets])
    ids=[r['id'] for r in targets];assert ids==p['ids'].tolist()==[r['id'] for r in manifest['rows']]
    det=p['detected'];e=np.linalg.norm(p['points']-gt,axis=-1);old=np.linalg.norm(p['original_points']-gt,axis=-1)
    diag=np.linalg.norm(np.array([r['original_hw'] for r in manifest['rows']]),axis=-1)
    hidden=np.array([r['hidden'] for r in targets]);outside=np.array([r['outside'] for r in targets])
    frame=np.full(len(det),np.nan);frame[det]=np.sum(np.where(valid[det],e[det],0),axis=-1)/valid[det].sum(-1)
    norm=np.ones(len(det));norm[det]=np.sum(np.where(valid[det],np.minimum(e[det]/diag[det,None],1),0),axis=-1)/valid[det].sum(-1)
    strata={}
    for name,mask in dict(all=valid,hidden=valid&hidden,outside=valid&outside,visible=valid&~hidden&~outside).items():
        eligible=mask&det[:,None];values=e[eligible]
        strata[name]=dict(total_points=int(mask.sum()),detected_points=int(eligible.sum()),
            mean_px=float(values.mean()) if len(values) else None,median_px=float(np.median(values)) if len(values) else None,
            p90_px=float(np.quantile(values,.9)) if len(values) else None,pck5=float((values<=5).mean()) if len(values) else None,
            pck10=float((values<=10).mean()) if len(values) else None)
    summary=dict(frames=len(det),detected_frames=int(det.sum()),detection_rate=float(det.mean()),
        frame_mean_px=float(np.nanmean(frame)),frame_median_px=float(np.nanmedian(frame)),frame_p90_px=float(np.nanquantile(frame,.9)),
        full_normalized_mean=float(norm.mean()),strata=strata,
        new_corner_catastrophes=int(((old<=5)&(e>20)&valid&det[:,None]).sum()))
    return summary,dict(frame=frame,normalized=norm,error=e,valid=valid,detected=det,hidden=hidden,outside=outside)


def bootstrap_difference(a,b,sessions,split):
    # a/b contain [seed,frame]; average independent trained seeds per frame first.
    diff=np.mean(a-b,axis=0)
    groups=np.unique(sessions) if split=='real' else np.arange(len(diff))
    groupsums=np.array([diff[np.array(sessions)==g].sum() for g in groups]) if split=='real' else diff
    counts=np.array([sum(np.array(sessions)==g) for g in groups]) if split=='real' else np.ones(len(diff))
    rng=np.random.default_rng(11092026);ix=rng.integers(0,len(groups),(5000,len(groups)))
    vals=groupsums[ix].sum(-1)/counts[ix].sum(-1)
    return dict(mean_difference=float(diff.mean()),ci95=np.quantile(vals,[.025,.975]).tolist(),
        unit='session cluster' if split=='real' else 'original image',clusters=len(groups),
        per_seed_difference=(a-b).mean(-1).tolist())


def scoring():
    assert read(RUN/'INFERENCE_COMPLETE.json')['complete']
    results={};comparisons={};gates={}
    for split in ('synth_val','real'):
        for condition in CONDITIONS:
            manifest=read(RUN/'cache'/split/condition/'manifest.json');targets=read(RUN/'targets'/f'{split}_{condition}.json')['rows']
            key=f'{split}/{condition}';rows={};arrays={}
            for path in sorted((RUN/'predictions'/split/condition).glob('*.npz')):
                rows[path.stem],arrays[path.stem]=score_file(path,manifest,targets)
            results[key]=rows;comp={};sessions=[r['session'] for r in targets]
            pairs=[('v10_local','stock'),('v10_directional','v10_local'),('local_extra','v10_local'),
                ('lines_frozen','local_extra'),('lines_joint','local_extra'),('lines_joint','lines_frozen'),
                ('lines_frozen','lines_frozen_donor'),('lines_joint','lines_joint_donor')]
            def names(arm):
                if arm=='stock':return ['stock']*3
                return [f'{arm[:-6]}_seed{s}_donor' if arm.endswith('_donor') else f'{arm}_seed{s}' for s in SEEDS]
            for arm,reference in pairs:
                an,bn=names(arm),names(reference)
                a=np.stack([arrays[n]['normalized'] for n in an]);b=np.stack([arrays[n]['normalized'] for n in bn])
                c=bootstrap_difference(a,b,sessions,split)
                c['conditional_frame_difference_px']=[rows[x]['frame_mean_px']-rows[y]['frame_mean_px'] for x,y in zip(an,bn)]
                c['conditional_relative_gain_percent']=[100*(1-rows[x]['frame_mean_px']/rows[y]['frame_mean_px']) for x,y in zip(an,bn)]
                # Raw-pixel bootstrap restricted to identical detection mask.
                det=arrays['stock']['detected'];aa=np.stack([arrays[n]['frame'][det] for n in an]);bb=np.stack([arrays[n]['frame'][det] for n in bn])
                c['paired_conditional_px']=bootstrap_difference(aa,bb,np.array(sessions)[det].tolist(),split)
                comp[f'{arm}_vs_{reference}']=c
            comparisons[key]=comp
            gates[key]={arm:dict(all_seeds_full_better=all(x<0 for x in comp[f'{arm}_vs_local_extra']['per_seed_difference']),
                cluster_ci_below_zero=comp[f'{arm}_vs_local_extra']['ci95'][1]<0,
                every_seed_gain1=all(x>=1 for x in comp[f'{arm}_vs_local_extra']['conditional_relative_gain_percent'])) for arm in ('lines_frozen','lines_joint')}
            print('SCORED',key,'stock',round(rows['stock']['frame_mean_px'],3),
                {arm:[round(rows[f'{arm}_seed{s}']['frame_mean_px'],3) for s in SEEDS] for arm in ('v10_local',)+ARMS},flush=True)
    for key in gates:
        for arm,gate in gates[key].items():
            clean=comparisons['real/clean'][f'{arm}_vs_local_extra']['conditional_relative_gain_percent']
            gate['real_clean_noninferior1']=all(x>=-1 for x in clean)
            gate['pass']=all(gate.values())
    write(RUN/'SUMMARY.json',dict(complete=True,results=results,comparisons=comparisons,gates=gates,
        limits='Reused synthetic and DEV real; no real training/tuning. Controlled image occlusion/crop does not establish natural occlusion performance. Same upstream detection failures for all refiners.'))


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['inference','score'],default='inference');args=ap.parse_args()
    inference() if args.phase=='inference' else scoring()
