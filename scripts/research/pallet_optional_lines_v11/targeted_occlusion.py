"""Posthoc exposure diagnostic: occlude a predicted upper corner, no new training."""
import importlib.util
import numpy as np
import torch
import cv2
from common import *
from optional_model import OptionalRefiner,predict_lines,donor_lines

def module(name,filename):
    spec=importlib.util.spec_from_file_location(name,KIT/filename);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

P=module('v11_image_preparation','prepare.py')
E=module('v11_primary_evaluation','evaluate.py')
CONDITION='corner20'


def extract():
    extractor=P.Extractor();population=read(RUN/'POPULATIONS.json')
    for split in ('synth_val','real'):
        directory=RUN/'cache'/split/CONDITION;directory.mkdir(parents=True,exist_ok=True)
        if (directory/'manifest.json').exists():continue
        clean,O=load_observations(split,'clean');lookup={r['id']:r for r in clean['rows']};rows=[];chunks=[];pending=[];count=0
        for rec in population[split]:
            c=lookup[rec['id']];im=P.raw_image(rec);h,w=im.shape[:2]
            box=np.array(c['box'] if c['box'] is not None else [.2*w,.2*h,.8*w,.8*h])
            corner=[0,1,4,5][int(digest('v11-corner:'+rec['id'])[:8],16)%4]
            center=O['points'][c['observation_index'],corner].numpy() if c['detected'] else (box[:2]+box[2:])/2
            side=.2*np.linalg.norm(box[2:]-box[:2]);a=max(0,min(w-1,int(round(center[0]-side/2))))
            b=max(a+1,min(w,int(round(center[0]+side/2))));u=max(0,min(h-1,int(round(center[1]-side/2))))
            v=max(u+1,min(h,int(round(center[1]+side/2))));canvas=im.copy()
            rng=np.random.default_rng(int(digest('v11-corner-color:'+rec['id'])[:8],16));color=rng.integers(40,216,3).astype(np.uint8)
            canvas[u:v,a:b]=color;obs,info=extractor(canvas,rec['K'],rec['dims'])
            rows.append(dict(**rec,condition=CONDITION,origin=[0,0],original_hw=[h,w],image_hw=[h,w],
                mask=[a,u,b,v],fill_bgr=color.tolist(),targeted_predicted_corner=corner,**info,
                image_sha256=sha(ROOT/rec['image']),observation_index=count if obs is not None else None))
            if obs is not None:pending.append(obs);count+=1
            if len(pending)>=64 or (len(rows)==len(population[split]) and pending):
                name=f'obs_{len(chunks):03d}.pt';torch.save({k:torch.cat([p[k] for p in pending]) for k in pending[0]},directory/name);chunks.append(name);pending=[]
        write(directory/'manifest.json',dict(rows=rows,chunks=chunks,chunk_sha256={p:sha(directory/p) for p in chunks},
            policy='No GT placement. SHA-selected upper corner ID from clean YOLO; square patch side20% observed box diagonal. No retraining.'))
        print('CORNER EXTRACTION',split,count,len(rows),flush=True)
    extractor.cap.handle.remove()


def inference():
    for split in ('synth_val','real'):
        manifest,O=load_observations(split,CONDITION,'cuda:0');n=len(O['points'])
        E.save_predictions(split,CONDITION,'stock',manifest,O,O['points'])
        for seed in SEEDS:
            line=dht(seed);ls=[];ms=[]
            with torch.inference_mode():
                for start in range(0,n,32):
                    a,b=predict_lines(line,subset(O,slice(start,start+32)));ls.append(a);ms.append(b)
            ll,mm=torch.cat(ls),torch.cat(ms);del line
            torch.save(dict(lines=ll.cpu(),mass=mm.cpu()),RUN/'cache'/split/CONDITION/f'lines_seed{seed}.pt')
            base=local(seed);old=local(seed,arm='directional');bp=[];op=[]
            with torch.inference_mode():
                for start in range(0,n,32):
                    sl=slice(start,start+32);obs=subset(O,sl)
                    bp.append(base(obs,ll[sl],mm[sl],'local')['points']);op.append(old(obs,ll[sl],mm[sl],'directional')['points'])
            bp=torch.cat(bp);op=torch.cat(op);del base,old
            E.save_predictions(split,CONDITION,f'v10_local_seed{seed}',manifest,O,bp)
            E.save_predictions(split,CONDITION,f'v10_directional_seed{seed}',manifest,O,op)
            for arm in ARMS:
                state=torch.load(RUN/'models'/f'{arm}_seed{seed}'/'final.pt',map_location='cpu',weights_only=False)
                model=OptionalRefiner(seed).cuda().eval();model.load_state_dict(state['state_dict']);l,m=ll,mm
                if arm=='lines_joint':
                    line=dht(seed);line.load_state_dict(state['dht_state_dict']);ls=[];ms=[]
                    with torch.inference_mode():
                        for start in range(0,n,32):
                            a,b=predict_lines(line,subset(O,slice(start,start+32)));ls.append(a);ms.append(b)
                    l,m=torch.cat(ls),torch.cat(ms);del line
                ids=[r['id'] for r in manifest['rows'] if r['detected']];order=sorted(range(n),key=lambda i:digest('v11-donor:'+ids[i]));ix=np.zeros(n,dtype=np.int64)
                for j,i in enumerate(order):ix[i]=order[(j+1)%n]
                dl,dm=donor_lines(l,m,O['box'],torch.tensor(ix,device='cuda:0'))
                for variant in (('actual',) if arm=='local_extra' else ('actual','donor')):
                    a,b=(l,m) if variant=='actual' else (dl,dm);parts=[]
                    with torch.inference_mode():
                        for start in range(0,n,32):
                            sl=slice(start,start+32);obs=subset(O,sl);p=model(obs,bp[sl],a[sl],b[sl],arm)
                            assert torch.equal(model(obs,bp[sl],a[sl],b[sl],arm,off=True)['points'],bp[sl])
                            parts.append({k:v.cpu() for k,v in p.items()})
                    merged={k:torch.cat([p[k] for p in parts]) for k in parts[0]};merged['lines']=a.cpu()
                    name=f'{arm}_seed{seed}'+('_donor' if variant=='donor' else '')
                    E.save_predictions(split,CONDITION,name,manifest,O,merged.pop('points'),merged)
                del model
            print('CORNER INFERRED',split,seed,flush=True)


def targets_and_score():
    results={};comparisons={};audits={}
    for split in ('synth_val','real'):
        manifest=read(RUN/'cache'/split/CONDITION/'manifest.json')
        original={r['id']:r for r in read(RUN/'targets'/f'{split}_clean.json')['rows']};targets=[]
        for rec in manifest['rows']:
            t=dict(original[rec['id']]);gt=np.array(t['points']);a,u,b,v=rec['mask']
            t['hidden']=((gt[:,0]>=a)&(gt[:,0]<b)&(gt[:,1]>=u)&(gt[:,1]<v)).tolist();targets.append(t)
        write(RUN/'targets'/f'{split}_{CONDITION}.json',dict(rows=targets,scope='Original clean targets unchanged; patch membership computed after inference.'))
        key=f'{split}/{CONDITION}';rows={};arrays={};maximum=0.
        for path in (RUN/'predictions'/split/CONDITION).glob('*.npz'):
            rows[path.stem],arrays[path.stem]=E.score_file(path,manifest,targets)
            p=np.load(path);det=p['detected'];assert np.isfinite(p['points'][det]).all()
            if 'base_points' in p:
                seed=int(path.stem.split('_seed')[1][0]);base=np.load(path.with_name(f'v10_local_seed{seed}.npz'))['points'][det]
                assert np.array_equal(base,p['base_points'][det])
                replay=base.astype(float)+p['gate'][det]*(p['A'][det].astype(float)@p['proposal'][det,...,None].astype(float))[...,0]
                maximum=max(maximum,float(np.max(np.abs(replay-p['points'][det]))));assert maximum<.001
        results[key]=rows;comp={};sessions=[r['session'] for r in targets]
        for arm,reference in [('v10_local','stock'),('lines_joint','local_extra'),('lines_frozen','local_extra'),('lines_joint','lines_joint_donor')]:
            aa=[f'{arm}_seed{s}' for s in SEEDS]
            bb=['stock']*3 if reference=='stock' else [f'{reference[:-6]}_seed{s}_donor' if reference.endswith('_donor') else f'{reference}_seed{s}' for s in SEEDS]
            c=E.bootstrap_difference(np.stack([arrays[n]['normalized'] for n in aa]),np.stack([arrays[n]['normalized'] for n in bb]),sessions,split)
            c['conditional_frame_difference_px']=[rows[a]['frame_mean_px']-rows[b]['frame_mean_px'] for a,b in zip(aa,bb)]
            c['conditional_relative_gain_percent']=[100*(1-rows[a]['frame_mean_px']/rows[b]['frame_mean_px']) for a,b in zip(aa,bb)]
            det=arrays['stock']['detected'];c['paired_conditional_px']=E.bootstrap_difference(np.stack([arrays[n]['frame'][det] for n in aa]),np.stack([arrays[n]['frame'][det] for n in bb]),np.array(sessions)[det].tolist(),split)
            comp[f'{arm}_vs_{reference}']=c
        comparisons[key]=comp;audits[key]=dict(additive_replay_max_px=maximum,original_gt_unchanged=True,exact_off_parity=True)
        print('CORNER SCORED',key,{a:[rows[f'{a}_seed{s}']['frame_mean_px'] for s in SEEDS] for a in ('v10_local',)+ARMS},flush=True)
    write(RUN/'TARGETED_SUMMARY.json',dict(complete=True,posthoc=True,results=results,comparisons=comparisons,audits=audits,
        scope='Supplement after observing low hidden-click exposure under central strip. All original trained models unchanged. No training/threshold/model selection.'))


def main():
    setup();cv2.setNumThreads(1)
    lock=RUN/'TARGETED_PROTOCOL.json'
    if lock.exists():raise SystemExit('Targeted diagnostic already registered')
    write(lock,dict(reason='Central strips hide only10/63 manual clicks at20/40%; targeted predicted-corner stress improves hidden-click exposure.',
        posthoc=True,condition=CONDITION,no_retraining=True,placement='SHA-selected point ID among0,1,4,5 from frozen clean YOLO; square side20% clean predicted box diagonal, solid deterministic color. No GT placement.',
        primary_summary_sha256=sha(RUN/'SUMMARY.json'),training_complete_sha256=sha(RUN/'TRAINING_COMPLETE.json'),
        source_sha256=sha(Path(__file__)),evaluation_source_sha256=sha(KIT/'evaluate.py')))
    extract();inference();targets_and_score()


if __name__=='__main__':main()
