"""Calibrate a global residual scale on synthetic data before evaluating."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from scripts.research.pallet_dht_structured_v2.cache import read,write,sha
from scripts.research.pallet_dht_structured_v2.evaluation import corner_metrics
from .cache import LocalData
from .train import load_trained,verify


def predict_population(run,arm,seed,population,device='cuda:0',batch_size=16):
    run=Path(run).resolve();protocol,bindings=verify(run)
    if population not in ('calibration','synth_val'):raise ValueError('Synthetic evaluation only')
    directory=run/'evaluation'/f'{arm}_seed{seed}';path=directory/f'{population}.npz';receipt=directory/f'{population}_COMPLETE.json'
    checkpoint_sha=sha(run/'runs'/f'{arm}_seed{seed}'/'checkpoint_final.pth')
    if receipt.exists():
        done=read(receipt)
        if not (done['complete'] and done['PASS'] and done['arm']==arm and done['seed']==seed and done['population']==population and done['bindings']==bindings and done['checkpoint_sha256']==checkpoint_sha and sha(path)==done['archive_sha256']):raise ValueError('Completed evaluation differs')
        with np.load(path,allow_pickle=False) as archive:return {k:np.array(archive[k]) for k in archive.files}
    if path.exists():raise ValueError('Interrupted population archive exists; preserve explicitly before restart')
    data=LocalData(run);ix=data.populations[population];model=load_trained(run,arm,seed,device)
    records=[];out=[];base=[];target=[];mask=[];diagonals=[]
    with torch.inference_mode():
        for start in range(0,len(ix),batch_size):
            ids=ix[start:start+batch_size];inputs,targets=data.batch(ids,device)
            predicted,diagnostics=model(inputs)
            if not torch.isfinite(predicted).all():raise ValueError('Nonfinite local evaluated output')
            if not torch.equal(predicted[:,8],inputs['baseline_points'][:,8]):raise ValueError('Centroid changed')
            out.append(predicted.cpu().numpy());base.append(inputs['baseline_points'].cpu().numpy());target.append(targets['points'].cpu().numpy())
            mask.append(targets['loss_valid'].cpu().numpy());diagonals.append(inputs['diagonal'].cpu().numpy())
    values=dict(indices=np.array(ix,np.int64),predicted=np.concatenate(out),baseline=np.concatenate(base),
        target=np.concatenate(target),loss_valid=np.concatenate(mask),diagonal=np.concatenate(diagonals))
    directory.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.pending.npz');np.savez_compressed(temporary,**values);temporary.replace(path)
    write(receipt,dict(complete=True,PASS=True,arm=arm,seed=seed,population=population,
        bindings=bindings,n_frames=len(ix),archive_sha256=sha(path),archive=str(path),real_GT_used=False,
        model_targets_separate=True,checkpoint_sha256=checkpoint_sha))
    return values


def scaled_metrics(values,scale):
    base=np.asarray(values['baseline'],float);pred=np.asarray(values['predicted'],float);gt=np.asarray(values['target'],float)
    q=base+scale*(pred-base)
    errors=np.linalg.norm(q[:,:8]-gt[:,:8],axis=-1);before=np.linalg.norm(base[:,:8]-gt[:,:8],axis=-1)
    return corner_metrics(errors,values['loss_valid'][:,:8],values['diagonal'],before)


def constraints(metric,baseline,tolerance=1e-12):
    return {**{f'{k}_no_worse':metric[k]<=baseline[k]+tolerance for k in ['median_px','p90_px','mean_px']},
            'good_crossing_le_one_percent':metric['good_to_bad_fraction']<=.01+tolerance}


def choose_scale(values,protocol):
    base=scaled_metrics(values,0.);rows=[];tolerance=protocol['calibration']['tie_tolerance']
    for scale in protocol['calibration']['output_scale_grid']:
        metric=scaled_metrics(values,scale);checks=constraints(metric,base,tolerance)
        rows.append(dict(scale=scale,metrics=metric,constraints=checks,eligible=all(checks.values())))
    eligible=[r for r in rows if r['eligible']];best=min(r['metrics']['mean_diagonal_normalized'] for r in eligible)
    chosen=min([r for r in eligible if r['metrics']['mean_diagonal_normalized']<=best+tolerance],key=lambda r:r['scale'])
    return dict(scale=chosen['scale'],grid=rows,baseline=base,selected=chosen)


def evaluate_synthetic(run,arm,seed=1,device='cuda:0'):
    run=Path(run).resolve();protocol,bindings=verify(run)
    cal=predict_population(run,arm,seed,'calibration',device);selected=choose_scale(cal,protocol)
    choicepath=run/f'SELECTION_{arm}_seed{seed}.json'
    choice=dict(complete=True,PASS=True,arm=arm,seed=seed,bindings=bindings,scale=selected['scale'],calibration=selected,
        calibration_archive_sha256=sha(run/'evaluation'/f'{arm}_seed{seed}'/'calibration.npz'),real_GT_used=False,
        validation_used_for_selection=False,checkpoint_sha256=sha(run/'runs'/f'{arm}_seed{seed}'/'checkpoint_final.pth'))
    if choicepath.exists() and read(choicepath)!=choice:raise ValueError('Cannot overwrite different calibrated scale')
    write(choicepath,choice)
    val=predict_population(run,arm,seed,'synth_val',device);metric=scaled_metrics(val,selected['scale']);base=scaled_metrics(val,0.)
    reduction=(base['mean_px']-metric['mean_px'])/base['mean_px']
    checks={**constraints(metric,base), 'nonzero_scale':selected['scale']>0.,
        'mean_reduction_at_least_one_percent':reduction>=protocol['synthetic_advancement']['native_mean_reduction_min_fraction']}
    result=dict(complete=True,PASS=True,arm=arm,seed=seed,bindings=bindings,scale=selected['scale'],
        calibration=selected,validation=metric,baseline=base,advancement=dict(advance=all(checks.values()),checks=checks,mean_reduction_fraction=reduction),
        selection_sha256=sha(choicepath),validation_archive_sha256=sha(run/'evaluation'/f'{arm}_seed{seed}'/'synth_val.npz'),
        real_evaluation_done=False,active_goal_complete=False)
    write(run/'evaluation'/f'{arm}_seed{seed}'/'SYNTHETIC_RESULTS.json',result)
    print(dict(arm=arm,scale=selected['scale'],advancement=result['advancement'],mean_px=metric['mean_px']),flush=True)
    return result
