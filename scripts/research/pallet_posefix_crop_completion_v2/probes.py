"""Post-fit fixed source evaluation and preselected TRAIN-fit diagnostics."""
import argparse
import gc
import numpy as np
import torch
from . import common as C
from . import data as D
from .inference import load_model
from scripts.research.pallet_sensors_submission_v1.prior_model import expectation
from scripts.research.pallet_posefix_limited_adaptation_pilot_v1.train import batch,parts
B=C.B

def summarize(rows,field):
    err=np.array([e for r in rows for e in r[field]['errors']]);inp=np.array([e for r in rows for e in r[field]['input_errors']]);hard=inp>20
    return dict(n=len(err),correct10=int((err<=10).sum()),correct20=int((err<=20).sum()),PCK10=float((err<=10).mean()) if len(err) else None,
        PCK20=float((err<=20).mean()) if len(err) else None,median=float(np.median(err)) if len(err) else None,P90=float(np.quantile(err,.9)) if len(err) else None,
        hard20=int(hard.sum()),recovered10=int((hard&(err<=10)).sum()),B3_input=int(((inp>20)&(inp<=40)).sum()),B3_recovered=int(((inp>20)&(inp<=40)&(err<=10)).sum()))

@torch.no_grad()
def forward_rows(model,items,masks):
    result=[]
    for i in range(0,len(items),2):
        xs=items[i:i+2];b=batch(xs);z=model(b['rgb'],b['points'],b['valid']);q=expectation(z).cpu().numpy();loss=parts(z,b['target'],b['target_valid'])
        for j,(x,p) in enumerate(zip(xs,q)):
            scale=x['matrix'][0,0];err=np.linalg.norm(p-x['target'],axis=1)/scale;initial=np.linalg.norm(x['points']-x['target'],axis=1)/scale
            rr=dict(id=x['id'],task_loss_heatmap=float(loss['heatmap']),task_loss_coordinate=float(loss['coordinate']))
            for name,mask in masks[i+j].items():rr[name]=dict(indices=np.flatnonzero(mask).tolist(),errors=err[mask].tolist(),input_errors=initial[mask].tolist())
            result.append(rr)
        if i%80==0:B.L.gpu_guard()
    return result

def source():
    assert C.read(C.DOC/'PREDICTION_LOCK.json')['all_frozen_before_scoring'];B.L.setup('cuda');audit=C.read(C.DOC/'TRAIN_INPUT_AUDIT.json')
    orders=np.load(C.ROOT/audit['source_orders']['path']);data=D.Source();pairs=[data.pair(int(i)) for i in orders['held_rows']]
    assert all(p[0]['partition']=='heldout' for p in pairs)
    rng=np.random.default_rng(7104);stress=[]
    for old,new in pairs:
        perturbed,original=D.original_corrupted(old,rng,True);stress.append((perturbed,D.with_original_points(new,original,perturbed['valid'])))
    masks=[dict(COMMON_SUPPORT=a['target_valid'],NEW_SUPPORT=b['target_valid']&~a['target_valid']) for a,b in pairs]
    results={};raw={}
    for arm,(weight,exp) in C.ARMS.items():
        model=load_model(weight);index=0 if exp==C.BASE_EXPANSION else 1;results[arm]={};raw[arm]={}
        for mode,inputs in [('clean',pairs),('stress',stress)]:
            rr=forward_rows(model,[x[index] for x in inputs],masks);raw[arm][mode]=rr
            results[arm][mode]={key:summarize(rr,key) for key in ('COMMON_SUPPORT','NEW_SUPPORT')}
            if arm=='A':
                legacy=C.read(B.DOC/'SOURCE_RESULTS.json')['models']['FULL'][mode]
                assert results[arm][mode]['COMMON_SUPPORT']['n']==legacy['corners'];assert results[arm][mode]['COMMON_SUPPORT']['PCK10']==legacy['PCK10']
                assert abs(results[arm][mode]['COMMON_SUPPORT']['P90']-legacy['P90_px'])<.001
        del model;gc.collect();torch.cuda.empty_cache();print('SOURCE_PROBE',arm,flush=True)
    C.save(C.RAW/'SOURCE_ROWS.json',raw);C.save(C.DOC/'SOURCE_RESULTS.json',dict(results=results,held_images=len(pairs),common_denominator='old1.25 support fixed, same original normal/stress points all4arms',
        new_support='separate supplemental; never mixed into main denominator',same_source_input_corruption=True))

def train_fit():
    assert C.read(C.DOC/'PREDICTION_LOCK.json')['all_frozen_before_scoring'];B.L.setup('cuda');audit=C.read(C.DOC/'TRAIN_INPUT_AUDIT.json');data=D.Source()
    pairs={'real_clean':[],'real_OCC':[],'source_clean':[],'source_stress':[]}
    for i in audit['real_probe_indices']:
        record=audit['real'][int(i)];old=torch.load(C.ROOT/record['old']['path'],map_location='cpu',weights_only=False)['pair'];new=torch.load(C.ROOT/record['pair']['path'],map_location='cpu',weights_only=False)['pair']
        for mode in ('CLEAN','OCC'):pairs['real_clean' if mode=='CLEAN' else 'real_OCC'].append((old[mode],new[mode]))
    orders=np.load(C.ROOT/audit['source_orders']['path']);corruption=np.load(C.ROOT/audit['corruption']['path'])
    for row in audit['source_probe_rows']:
        old,new=data.pair(int(row));pairs['source_clean'].append((old,new))
        step,j=np.argwhere(orders['source_rows']==row)[0];xy=corruption['points'][step,j];v=corruption['valid'][step,j]
        pairs['source_stress'].append((D.with_original_points(old,xy,v),D.with_original_points(new,xy,v)))
    arms={'PRIOR1_125':('PRIOR1',0),'PRIOR1_150':('PRIOR1',1),'A':('FULL',0),'C':('C',1)};results={};raw={}
    for arm,(weight,index) in arms.items():
        m=load_model(weight);results[arm]={};raw[arm]={}
        for mode,pp in pairs.items():
            masks=[dict(COMMON_SUPPORT=x['target_valid'],NEW_SUPPORT=y['target_valid']&~x['target_valid']) for x,y in pp]
            rr=forward_rows(m,[x[index] for x in pp],masks);raw[arm][mode]=rr
            results[arm][mode]=dict(**{key:summarize(rr,key) for key in ('COMMON_SUPPORT','NEW_SUPPORT')},
                task_heatmap=float(np.mean([r['task_loss_heatmap'] for r in rr])),task_coordinate=float(np.mean([r['task_loss_coordinate'] for r in rr])))
        del m;gc.collect();torch.cuda.empty_cache();print('TRAIN_FIT_PROBE',arm,flush=True)
    C.save(C.RAW/'TRAIN_FIT_ROWS.json',raw);C.save(C.DOC/'TRAIN_FIT_RESULTS.json',dict(results=results,selected_before_inference=True,source_stress='first original corrupted occurrence from locked training trace (no new RNG)',
        not_validation=True,real_probe_indices=audit['real_probe_indices'],source_probe_rows=audit['source_probe_rows']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['source','train_fit']);a=p.parse_args();globals()[a.stage]()
