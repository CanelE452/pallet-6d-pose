"""Actual camera-order point-coordinate evaluation, no oracle correspondence."""
from pathlib import Path
import time
import numpy as np
import torch
from data import RUN,PARENT,ROOT,EXPORT,observations,line_inputs,read,write,sha
from refiner import Refiner
from experiment import ARMS


def score(rows,targets,full_count):
    assert len(rows)==len(targets)
    frames=[];before_all=[];after_all=[];catastrophes=0
    for row,gt in zip(rows,targets):
        assert row['id']==gt['id']
        valid=np.array(gt['valid']);truth=np.array(gt['points']);p=np.array(row['points']);old=np.array(row['original_points'])
        e=np.linalg.norm(p-truth,axis=-1);b=np.linalg.norm(old-truth,axis=-1)
        assert np.isfinite(e).all()
        catastrophe=(b<=5)&(e>20)&valid;catastrophes+=int(catastrophe.sum())
        before_all.extend(b[valid]);after_all.extend(e[valid])
        frames.append(dict(id=row['id'],mean_px=float(e[valid].mean()),baseline_mean_px=float(b[valid].mean()),
            normalized=float(np.minimum(e[valid]/row['image_diagonal'],1.).mean()),
            baseline_normalized=float(np.minimum(b[valid]/row['image_diagonal'],1.).mean()),
            errors=e.tolist(),baseline_errors=b.tolist(),valid=valid.tolist(),new_catastrophes=int(catastrophe.sum()),
            mean_movement_px=float(np.linalg.norm(p-old,axis=-1)[valid].mean())))
    a=np.array(after_all);means=np.array([r['mean_px'] for r in frames]);miss=full_count-len(frames)
    return dict(rows=frames,summary=dict(frames=len(frames),full_population=full_count,missing=miss,valid_corners=len(a),
        raw_frame_mean=float(means.mean()),raw_frame_median=float(np.median(means)),raw_frame_p90=float(np.quantile(means,.9)),
        corner_mean=float(a.mean()),pck5=float((a<=5).mean()),pck10=float((a<=10).mean()),
        full_normalized_mean=float((sum(r['normalized'] for r in frames)+miss)/full_count),
        new_corner_catastrophes=catastrophes,mean_movement_px=float(np.mean([r['mean_movement_px'] for r in frames]))))


def main():
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert read(RUN/'TRAINING_COMPLETE.json')['complete']
    proto=RUN/'EVALUATION_PROTOCOL.json'
    if proto.exists():raise SystemExit('evaluation protocol exists')
    write(proto,dict(source_sha256=sha(Path(__file__)),training_protocol_sha256=sha(RUN/'PROTOCOL.json'),
        scoring='Original camera-facing point IDs exactly as source labels; no Hungarian matching, symmetry flip or GT pose. '
            'Visibility>0, frame-average pixels plus micro corner metrics. Missing matches counted as normalized error1.',
        causal='Same directional model: actual lines, wrong-image lines, same attention/proposal with isotropic projection. '
            'Actual line input and isotropic projection must have identical proposal and attention; only matrix A changes.',
        scope='Reused synthetic matched val505/512 and calibration253/256. No real or 6D-pose claim; no checkpoint selection.'))
    (RUN/'predictions').mkdir(exist_ok=True);(RUN/'metrics').mkdir(exist_ok=True)
    allnames=[];started=time.monotonic()
    for split in ('synth_val','calibration'):
        # Targets remain unopened throughout this prediction phase.
        calls=[];original_load=torch.load
        def traced(p,*a,**k):calls.append(str(p));return original_load(p,*a,**k)
        torch.load=traced
        try:
            data,O=observations(split,'cuda:0')
            original=[]
            for i,rec in enumerate(data.records):
                pp=O['points'][i].cpu().tolist()
                original.append(dict(id=rec['id'],points=pp,original_points=pp,image_diagonal=float(torch.linalg.vector_norm(O['image_hw'][i]))))
            name=f'stock_{split}';write(RUN/'predictions'/f'{name}.json',dict(rows=original,split=split,arm='stock'));allnames.append(name)
            for seed in (1,2,3):
                l,m=line_inputs(split,seed,O);dl,dm=line_inputs(split,seed,O,donor=True)
                for arm in ARMS:
                    if split=='calibration' and arm!='directional':continue
                    ck=RUN/'models'/f'{arm}_seed{seed}'/'final.pt';model=Refiner(seed).to('cuda:0')
                    model.load_state_dict(torch.load(ck,map_location='cuda:0',weights_only=False)['state_dict']);model.eval()
                    variants=['actual','isotropic_projection','donor_input'] if split=='calibration' else ['actual']
                    for variant in variants:
                        ll,mm=(dl,dm) if arm=='donor' or variant=='donor_input' else (l,m)
                        rows=[]
                        with torch.inference_mode():
                            for first in range(0,len(data.records),32):
                                ii=list(range(first,min(first+32,len(data.records))));obs={k:v[ii] for k,v in O.items()}
                                out=model(obs,ll[ii],mm[ii],arm,projection='isotropic' if variant=='isotropic_projection' else None)
                                for j,i in enumerate(ii):
                                    row=dict(id=data.records[i]['id'],points=out['points'][j].cpu().tolist(),original_points=O['points'][i].cpu().tolist(),
                                        image_diagonal=float(torch.linalg.vector_norm(O['image_hw'][i])),proposal=out['proposal'][j].cpu().tolist(),
                                        A=out['A'][j].cpu().tolist(),null_weight=out['weights'][j,:,-1].cpu().tolist())
                                    if arm=='directional':row.update(weights=out['weights'][j].cpu().tolist(),lines=ll[i].cpu().tolist())
                                    rows.append(row)
                        name=f'{arm}_seed{seed}_{split}_{variant}';allnames.append(name)
                        write(RUN/'predictions'/f'{name}.json',dict(rows=rows,arm=arm,seed=seed,split=split,variant=variant,
                             checkpoint_sha256=sha(ck),scope='GT-free point refinement; null attention is learned trust, not calibrated probability.'))
                        print('predicted',name,len(rows),flush=True)
                    del model
        finally:torch.load=original_load
        assert not any('/supervision/' in p for p in calls)
        write(RUN/f'INFERENCE_AUDIT_{split}.json',dict(supervision_tensor_loads=0,observed_tensor_loads=len(calls),targets_json_opened=False))
        del O
    summaries={}
    # Scoring starts only after every prediction file is on disk.
    for name in allnames:
        p=read(RUN/'predictions'/f'{name}.json');targets=read(RUN/f'targets_{p["split"]}.json')['rows']
        result=score(p['rows'],targets,512 if p['split']=='synth_val' else 256)
        result['prediction_sha256']=sha(RUN/'predictions'/f'{name}.json')
        write(RUN/'metrics'/f'{name}.json',result);summaries[name]=result['summary']
        print('scored',name,round(result['summary']['raw_frame_mean'],4),'catastrophes',result['summary']['new_corner_catastrophes'],flush=True)
    gates={};stock=summaries['stock_synth_val']
    for seed in (1,2,3):
        cells={a:summaries[f'{a}_seed{seed}_synth_val_actual'] for a in ARMS};d=cells['directional']
        gates[str(seed)]=dict(stock_gain3=d['raw_frame_mean']<=.97*stock['raw_frame_mean'],
            stock_median_nonworse=d['raw_frame_median']<=stock['raw_frame_median'],stock_p90_nonworse=d['raw_frame_p90']<=stock['raw_frame_p90'],
            full_mean_better=d['full_normalized_mean']<stock['full_normalized_mean'],
            local_nonworse=d['raw_frame_mean']<=cells['local']['raw_frame_mean'],
            isotropic_nonworse=d['raw_frame_mean']<=cells['isotropic']['raw_frame_mean'],
            beats_donor=d['raw_frame_mean']<cells['donor']['raw_frame_mean'],no_new_catastrophes=d['new_corner_catastrophes']==0)
    write(RUN/'SUMMARY.json',dict(complete=True,summaries=summaries,gates=gates,pilot_gate=all(all(g.values()) for g in gates.values()),
                                training_updates=24000,evaluated_cells=len(allnames),elapsed_seconds=time.monotonic()-started,real_or_6d_pose_established=False))
    print('EVALUATION COMPLETE',gates,flush=True)


if __name__=='__main__':main()
