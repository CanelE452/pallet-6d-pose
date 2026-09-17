"""Separate square/mixed tracks, frozen main-paper decode; no DEV selection."""
import argparse,math
import numpy as np
import torch
import dcp_env as E
from square_data import SquareData
from data import PaperData
from inference import load_head,preservation,serial
from refiner import forward,decode,targets
from eval_math import measure,summary,contrast,damage
from point_inference import replace_selected

@torch.no_grad()
def square(arms):
    data=SquareData();rule=E.read(E.DOC/'CALIBRATION_AND_SELECTION.json')['rule'];allrows={};summaries={};files={}
    mixed=arms==E.MIXED_ARMS;temperatures=E.read(E.DOC/'MIXED_CALIBRATION.json')['temperatures'] if mixed else None
    for arm in arms:
        for seed in [1,2,3]:
            name=f'{arm}_seed{seed}';dst=E.RAW/f'predictions/SQUARE_DEV/{name}.json'
            if not dst.exists():
                head,_=load_head(arm,seed);preds=[];T=temperatures[name]['temperature'] if mixed else 1.
                for start in range(0,len(data.validation_rows),16):
                    rows=data.validation_rows[start:start+16];o=forward(head,data.batch(rows,arm,supervision=False));cap=rule['max_move_image_diagonal_fraction']*np.array([math.hypot(*data.rows[i]['raw_hw']) for i in rows])*data.arrays['gain'][rows] if rule['max_move_image_diagonal_fraction'] is not None else None
                    q=decode(o,T,rule['lam'],cap).cpu().numpy()
                    for j,i in enumerate(rows):
                        base=E.read(E.RAW/f'square_cache/{i:04d}.json');idx=base['selected_index'];c=replace_selected(base['candidates'],idx,data.arrays['points'][i],q[j],data.arrays['gain'][i],rule['lam']);preservation(base['candidates'],c,idx)
                        preds.append(dict(id=base['id'],selected_index=idx,candidates=serial(c)))
                E.write(dst,dict(complete=True,records=preds,GT_input=False,temperature=T));del head;torch.cuda.empty_cache()
            preds=E.read(dst)['records'];files[name]=E.bound(dst);scored=[]
            for p,i in zip(preds,data.validation_rows):
                r=data.rows[i];assert r['id']==p['id'];a=np.array(r['target'])[0,5:].reshape(9,3);gt=a[:,:2]*(np.array(r['raw_hw'])[::-1]+200)-100
                idx=p['selected_index'];points=np.full((9,2),np.nan) if idx is None else p['candidates'][idx]['keypoints_xy']
                m=measure(points,gt,a[:,2]>0,data.perms,r['raw_hw'],bool(data.arrays['matched'][i]),idx is not None);m.update(id=r['id'],session=r['id'].split('__')[0],group='C4',object='plastic_standard_110x110x15',ratio_bin='<=1.05');scored.append(m)
            allrows[name]=scored;summaries[name]=summary(scored);print('SQUARE_RESULT',name,summaries[name],flush=True)
    prefix='MIXED_SQUARE' if mixed else 'SQUARE';E.write(E.RAW/f'{prefix}_METRICS.json',allrows)
    E.write(E.DOC/f'{prefix}_RESULTS.json',dict(complete=True,summary=summaries,predictions=files,dimension_effect_identifiable=False,constant_dimensions=True,real_supervised_secondary=True,DEV_reused=True))
    return allrows

@torch.no_grad()
def mixed_paper():
    # Reuse exactly the paper evaluation functions with explicit arm arguments;
    # never overwrite the completed paper-track artifacts.
    from paper_evaluate import light_batch
    data=PaperData();lock=E.read(E.DOC/'TRAIN_PROTOCOL_LOCK.json');temperatures={};allrows={};summaries={}
    for arm in E.MIXED_ARMS:
        for seed in [1,2,3]:
            name=f'{arm}_seed{seed}';dst=E.RAW/f'logits/{name}.npz'
            if not dst.exists():
                head,_=load_head(arm,seed);logs=[];sup=[]
                for start in range(0,len(data.validation_rows),16):
                    o=forward(head,data.batch(data.validation_rows[start:start+16],arm,supervision=False));logs.append(o['logits'].cpu().numpy());sup.append(o['point_support'].cpu().numpy())
                with dst.open('xb') as f:np.savez(f,rows=data.validation_rows,logits=np.concatenate(logs),support=np.concatenate(sup))
                del head;torch.cuda.empty_cache()
            logs=np.load(dst);ix=np.flatnonzero(data.partitions[logs['rows']]=='calibration');tot=np.zeros(len(lock['temperature_grid']));n=0
            for start in range(0,len(ix),64):
                rows,t,o=light_batch(data,logs,ix[start:start+64],supervision=True);target=targets(o,t['gt_points'],t['gt_valid']);mask=target['support'];count=mask.sum(-1);used=count>0;n+=int(used.sum())
                for j,T in enumerate(lock['temperature_grid']):
                    ce=-(target['distribution'].double()*(o['logits'].double()/T).log_softmax(-1)).sum(-1);tot[j]+=float(((ce*mask).sum(-1)/count.clamp_min(1))[used].sum())
            candidates=[dict(temperature=T,score=float(s/n)) for T,s in zip(lock['temperature_grid'],tot)];temperatures[name]=dict(**E.old('select_synthetic').choose_temperature(candidates),candidates=candidates)
    E.freeze(E.DOC/'MIXED_CALIBRATION.json',dict(complete=True,temperatures=temperatures,real_selection=False,rule=lock['selected_rule']))
    for arm in E.MIXED_ARMS:
        for seed in [1,2,3]:
            name=f'{arm}_seed{seed}';logs=np.load(E.RAW/f'logits/{name}.npz');ix=np.flatnonzero(data.partitions[logs['rows']]=='heldout');preds=[];rule=lock['selected_rule'];T=temperatures[name]['temperature']
            for start in range(0,len(ix),64):
                rows,t,o=light_batch(data,logs,ix[start:start+64]);records=[data.source['records'][data.indices[i]] for i in rows];gain=data.arrays['gain'][rows]
                cap=None if rule['max_move_image_diagonal_fraction'] is None else rule['max_move_image_diagonal_fraction']*np.array([math.hypot(*r['raw_shape_hw']) for r in records])*gain
                q=decode(o,T,rule['lam'],cap).numpy()
                for j,row in enumerate(rows):
                    base=E.read(E.RAW/f'source_baseline/{row:05d}.json');idx=base['selected_index'];c=replace_selected(base['candidates'],idx,data.arrays['points'][row],q[j],gain[j],rule['lam']);preservation(base['candidates'],c,idx);preds.append(dict(id=base['id'],selected_index=idx,candidates=serial(c)))
            E.write(E.RAW/f'predictions/SYNTH_HELDOUT/{name}.json',dict(complete=True,records=preds,GT_input=False));scored=[]
            for row,p in zip(logs['rows'][ix],preds):
                r=data.source['records'][data.indices[row]];a=np.array(r['targets'][0]['keypoints_normalized']);gt=a[:,:2]*np.array(r['prepared_shape_hw'])[::-1]-100;idx=p['selected_index'];g=int(data.side['order'][row]);point=np.full((9,2),np.nan) if idx is None else p['candidates'][idx]['keypoints_xy']
                m=measure(point,gt,a[:,2]>0,data.side['permutations'][row,:g],r['raw_shape_hw'],bool(data.arrays['matched'][row]),idx is not None);m.update(id=r['id'],session=r['scenario_id'],group='C'+str(g),object=r['source'],ratio_bin='diagnostic');scored.append(m)
            allrows[name]=scored;summaries[name]=summary(scored)
    E.write(E.RAW/'MIXED_SYNTH_METRICS.json',allrows);E.write(E.DOC/'MIXED_SYNTH_RESULTS.json',dict(complete=True,summary=summaries,diagnostic_only=True))
    from dev_evaluate import infer,evaluate
    infer(E.MIXED_ARMS,temperatures);evaluate(E.MIXED_ARMS);square(E.MIXED_ARMS)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('track',choices=['square','mixed']);args=p.parse_args();torch.set_num_threads(4);E.gpu()
    if args.track=='square':assert E.read(E.DOC/'SQUARE_TRAINING_COMPLETE.json')['complete'];square(E.SQUARE_ARMS)
    else:assert E.read(E.DOC/'MIXED_TRAINING_COMPLETE.json')['complete'];mixed_paper()
