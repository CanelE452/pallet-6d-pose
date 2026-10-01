"""One frozen synthetic calibration/selection grid; heldout opened afterwards."""
import math
import numpy as np
import torch
from common import *
from data import Dataset
from train import CLASSES,forward
from refiner import GenericPointRefiner,targets

def load_heads(device):
    protocol=verify_lock()
    heads={}
    for seed in SEEDS:
        path=RAW/'runs'/f'seed{seed}'/'paired_last.pt'
        assert sha(path)==read(DOC/f'TRAIN_SEED{seed}.json')['checkpoint']['sha256']
        ck=torch.load(path,map_location='cpu',weights_only=False)
        assert ck['complete'] and ck['step']==6000 and ck['config']==CONFIG
        assert ck['protocol_sha256']==sha(DOC/'PROTOCOL.json')
        assert ck['baseline_sha256']==protocol['baseline']['sha256']
        for arm in ('P','D'):
            model=CLASSES[arm](**CONFIG).to(device).eval();model.load_state_dict(ck['models'][arm])
            heads[f'{arm}{seed}']=model
    return heads

@torch.no_grad()
def validation(adapter):
    verify_lock();assert read(DOC/'TRAINING_COMPLETE.json')['complete']
    complete=DOC/'VALIDATION_OUTPUTS.json'
    if complete.exists():
        for b in read(complete)['outputs']:assert sha(ROOT/b['path'])==b['sha256']
        return
    data=Dataset();heads=load_heads(adapter.device);rows=data.validation_rows
    collected={a:dict(support=[],value=[]) for a in heads}
    for begin in range(0,len(rows),8):
        batch=data.batch_features(rows[begin:begin+8],adapter)
        for arm,head in heads.items():
            out=forward(head,batch)
            collected[arm]['support'].append(out['point_support'].cpu().numpy())
            field='logits' if arm[0]=='P' else 'delta_normalized'
            value=out[field].cpu().numpy();assert np.isfinite(value).all()
            collected[arm]['value'].append(value)
        if begin%128==0:print('RESNET18_VALIDATION',begin,len(rows),flush=True)
    bindings=[]
    for arm,a in collected.items():
        path=RAW/f'validation_{arm}.npz'
        with path.open('xb') as f:np.savez(f,rows=rows,support=np.concatenate(a['support']),value=np.concatenate(a['value']))
        bindings.append(bound(path))
    write(complete,dict(complete=True,outputs=bindings,rows=len(rows),accuracy_read=False,
       training=bound(DOC/'TRAINING_COMPLETE.json'),protocol=bound(DOC/'PROTOCOL.json')))
    data.close()

def source_arrays():
    data=Dataset();rows=data.validation_rows
    a={k:np.array(v[rows],copy=True) for k,v in data.arrays.items() if k!='done'}
    a['raw_diagonal']=np.array([math.hypot(*data.records[i]['raw_shape_hw']) for i in rows])
    a['partition']=data.partitions[rows];a['record_indices']=rows
    return data,a

def displacement(logits,temperature,bank,diag):
    x=torch.from_numpy(logits)/temperature
    p=x.softmax(-1).numpy()
    return (p[...,None]*bank[None,None]).sum(-2)*diag[:,None,None]

def refined_delta(values,arm,T,a,lam,cap_fraction):
    boxes=a['boxes'];valid=np.isfinite(boxes).all(-1)&(boxes[:,2:]>boxes[:,:2]).all(-1)
    safe=np.where(valid[:,None],boxes,np.array([0,0,1,1]));diag=np.maximum(np.linalg.norm(safe[:,2:]-safe[:,:2],axis=-1),1.)
    if arm.startswith('P'):
        bank=GenericPointRefiner(**CONFIG).displacements.numpy()
        delta=displacement(values['value'],T,bank,diag)
    else:delta=values['value']*diag[:,None,None]
    delta=delta*float(lam)/a['scale_xy'][:,None]
    if cap_fraction is not None:
        cap=a['raw_diagonal']*cap_fraction
        delta*=np.minimum(1.,cap[:,None]/np.maximum(np.linalg.norm(delta,axis=-1),1e-12))[...,None]
    return np.where(values['support'][...,None],delta,0.)

def scores(delta,a,subset):
    # All eight source GT corners form the fixed selection denominator.
    # Restrict rows BEFORE any reference comparison, including diagnostics.
    delta=delta[subset];a={k:v[subset] for k,v in a.items()}
    base=(a['points']-a['shift_xy'][:,None])/a['scale_xy'][:,None]
    gt=(a['gt_points']-a['shift_xy'][:,None])/a['scale_xy'][:,None]
    q=base.copy();q[:,:8]+=delta
    gv=a['gt_valid'];observed=gv&a['point_valid']&a['matched'][:,None]&np.isfinite(q).all(-1)
    e=np.linalg.norm(q-gt,axis=-1)
    norm=np.where(observed[:,:8],np.minimum(e[:,:8]/a['raw_diagonal'][:,None],1.),1.)
    counts=gv[:,:8].sum(-1)
    frame=np.where(counts>0,(np.where(gv[:,:8],norm,0.)).sum(-1)/np.maximum(counts,1),1.)
    errors=e[observed];denom=int(gv.sum())
    return dict(score=float(frame.mean()),frames=len(frame),gt9=denom,
        observed9=len(errors),missing9=denom-len(errors),
        median_px=float(np.median(errors)) if len(errors) else None,
        p90_px=float(np.quantile(errors,.9)) if len(errors) else None,
        pck10_all_gt=float((errors<=10).sum()/denom),
        matched_frames=int(a['matched'].sum()))

def select():
    protocol=verify_lock();manifest=read(DOC/'VALIDATION_OUTPUTS.json')
    for b in manifest['outputs']:assert sha(ROOT/b['path'])==b['sha256']
    data,a=source_arrays();values={f'{arm}{s}':dict(np.load(RAW/f'validation_{arm}{s}.npz')) for s in SEEDS for arm in ('P','D')}
    for v in values.values():assert np.array_equal(v['rows'],a['record_indices'])
    selection_file=DOC/'SELECTION.json'
    if not selection_file.exists():
        subset=a['partition']=='calibration';temperatures={};calibration={}
        for seed in SEEDS:
            key=f'P{seed}';v=values[key];boxes=a['boxes'][subset]
            bv=np.isfinite(boxes).all(-1)&(boxes[:,2:]>boxes[:,:2]).all(-1)
            diag=np.maximum(np.linalg.norm(np.where(bv[:,None],boxes,np.array([0,0,1,1]))[:,2:]-np.where(bv[:,None],boxes,np.array([0,0,1,1]))[:,:2],axis=-1),1.)
            bank=GenericPointRefiner(**CONFIG).displacements.detach()
            o=dict(points_raw=torch.from_numpy(a['points'][subset]),point_support=torch.from_numpy(v['support'][subset]),
                   candidate_displacements=torch.from_numpy(diag).float()[:,None,None]*bank,
                   box_diagonal=torch.from_numpy(diag).float())
            gt_valid=a['gt_valid'][subset]&a['matched'][subset,None]
            target=targets(o,torch.from_numpy(a['gt_points'][subset]),torch.from_numpy(gt_valid))
            mask=target['support'];count=mask.sum(-1);used=count>0;assert used.any()
            candidates=[]
            for T in protocol['calibration']['temperature_grid']:
                ce=-(target['distribution'].double()*(torch.from_numpy(v['value'][subset]).double()/T).log_softmax(-1)).sum(-1)
                score=float(((ce*mask).sum(-1)/count.clamp_min(1))[used].mean())
                candidates.append(dict(temperature=T,score=score))
            chosen=min(candidates,key=lambda r:(r['score'],abs(math.log(r['temperature'])),r['temperature']))
            temperatures[key]=chosen['temperature'];calibration[key]=dict(selected=chosen,candidates=candidates,supported_frames=int(used.sum()))
        for seed in SEEDS:temperatures[f'D{seed}']=1.
        subset=a['partition']=='selection';rules={};grids={}
        for arm in ('P','D'):
            candidates=[]
            for lam in protocol['selection']['lambda_grid']:
                for cap in protocol['selection']['max_move_image_diagonal_fractions']:
                    byseed={str(s):scores(refined_delta(values[f'{arm}{s}'],arm,temperatures[f'{arm}{s}'],a,lam,cap),a,subset) for s in SEEDS}
                    candidates.append(dict(lam=lam,max_move_image_diagonal_fraction=cap,score=float(np.mean([r['score'] for r in byseed.values()])),seeds=byseed))
            rules[arm]=min(candidates,key=lambda r:(r['score'],r['lam'],math.inf if r['max_move_image_diagonal_fraction'] is None else r['max_move_image_diagonal_fraction']))
            grids[arm]=candidates
        write(selection_file,dict(complete=True,created_at=now(),protocol=bound(DOC/'PROTOCOL.json'),validation=bound(DOC/'VALIDATION_OUTPUTS.json'),
             temperatures=temperatures,calibration=calibration,rules=rules,candidates=grids,real_selection=False,heldout_accuracy_opened=False))
    selection=read(selection_file);subset=a['partition']=='heldout'
    result={'RESNET18':scores(np.zeros((len(subset),8,2)),a,subset)}
    for key,v in values.items():
        rule=selection['rules'][key[0]]
        result[key]=scores(refined_delta(v,key,selection['temperatures'][key],a,rule['lam'],rule['max_move_image_diagonal_fraction']),a,subset)
    write(DOC/'SYNTHETIC_HELDOUT.json',dict(complete=True,selection=bound(selection_file),results=result,
        scope='Historical source validation heldout partition, not independent of backbone development'))
    data.close();print('RESNET18_SELECTION_FROZEN',selection['rules'],flush=True)
