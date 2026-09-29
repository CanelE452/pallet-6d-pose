"""Post-fit integrity audit; no fitting, selection change, or historical write."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from . import common as C
from . import calibration as K
from . import calibration_eval as E


def validate_curve(result,updates_per_epoch):
    curve=result['curve']
    assert 1<=len(curve)<=30 and result['epochs']==len(curve)
    assert [row['epoch'] for row in curve]==list(range(1,len(curve)+1))
    best=max(row['val_accuracy'] for row in curve)
    first=next(row['epoch'] for row in curve if row['val_accuracy']==best)
    assert result['best_epoch']==first and result['best_val_accuracy']==best
    assert len(curve)==30 or len(curve)-first==5
    return len(curve)*updates_per_epoch


def main():
    import torch
    from scripts.research.pallet_selector_recovery_v1 import models as M
    protocol=K.verify_protocol(); features=K.verify_features(); fit=C.read(K.FIT)
    lock=E.verify(); results=C.read(E.RESULT)
    for binding in fit['sources']+[fit['checkpoint']]+results['sources']+results['private_artifacts']:C.verify(binding)
    labels=C.read(K.LABEL_LOCK); C.verify(labels['labels']); C.verify(labels['source_labels'])
    with np.load(C.ROOT/features['features']['path']) as data:z=dict(data)
    with np.load(C.ROOT/labels['labels']['path']) as data:y=dict(data)
    assert np.array_equal(z['ids'],y['ids']) and np.array_equal(z['split'],y['split'])
    assert set(z['split'])=={'TRAIN','VAL'}
    valid=np.concatenate([z[arm+'_valid'] for arm in K.ARMS])
    parts=np.tile(z['split'],2); train=(parts=='TRAIN')&valid; val=(parts=='VAL')&valid
    x=np.concatenate([z[arm+'_geo'] for arm in K.ARMS]); _,mean,std=M.normalize(x,train)
    ck=torch.load(C.ROOT/fit['checkpoint']['path'],map_location='cpu',weights_only=False)
    assert ck['variant']=='GEO_LINEAR' and ck['d']==94
    assert set(ck['state'])=={'net.weight','net.bias'} and ck['state']['net.weight'].shape==(1,94)
    assert ck['state']['net.bias'].shape==(1,)
    assert np.array_equal(ck['mean'],mean) and np.array_equal(ck['std'],std)
    with np.load(C.ROOT/labels['source_labels']['path']) as original:
        indexes={fid:i for i,fid in enumerate(original['ids'])}
        take=[indexes[fid] for fid in y['ids']]
        assert np.array_equal(y['parity'],original['parity'][take])
        assert np.array_equal(y['split'],original['split'][take])
    steps_per_epoch=int(np.ceil(train.sum()/256)); steps=validate_curve(fit['result'],steps_per_epoch)
    assert steps==fit['optimizer_steps'] and steps_per_epoch==32
    assert train.sum()==fit['preflight']['valid_train']==8190 and val.sum()==fit['preflight']['valid_val']==2048
    assert fit['selector_fits']==1 and fit['student_fits']==0 and fit['no_TEST_evaluation']
    attempts=list(C.DOC.glob('SELECTOR_CALIBRATION_ATTEMPT*.json'));assert len(attempts)==1
    read_bad=('GEOMETRY_RESOLVED','TRUTH_FOR_DISPLAY','VERIFIED_LABELS','AXIS_REVIEW',
        '/data/evaluation/','/evaluation/S','ORACLE_METRICS','POSE_METRICS','FRAME_METRICS',
        'CONTROL_RESULTS','CONTROL_FRAME','EVAL_RESULTS','SELECTOR_PAIR_RESULTS','ZERO_FIT_SELECTOR_RESULTS')
    for stage in (protocol,features,labels,fit):
        assert not [p for p in stage.get('read_paths',[]) if any(token in p for token in read_bad)]
    old_metrics=C.read(C.RAW/'CONTROL_FRAME_METRICS_PRIVATE.json')
    new_metrics=C.read(C.RAW/'CURRENT_GEO_FRAME_METRICS_PRIVATE.json')
    old_poses=C.read(C.RAW/'CONTROL_POSES_PRIVATE.json')
    new_poses=C.read(C.RAW/'CURRENT_GEO_POSES_PRIVATE.json')
    assert len(old_metrics)==16 and len(new_metrics)==24
    assert all(new_metrics[arm]==value for arm,value in old_metrics.items())
    assert all(new_poses[arm]==value for arm,value in old_poses.items())
    associations=C.read(C.RAW/'CONTROL_CASE_BINDINGS_PRIVATE.json')
    decisions=C.read(E.PRIVATE/'DECISIONS.json'); generated_poses=C.read(E.PRIVATE/'POSES.json')
    checked=0; sources=[]; cache={}
    for model in E.MODELS:
        binding=associations[model+'_GEO']['candidates']; C.verify(binding)
        if binding['path'] not in cache:cache[binding['path']]=C.read(C.ROOT/binding['path']);sources.append(binding)
        candidates=cache[binding['path']][associations[model+'_GEO']['candidate_arm']]
        arm=model+'_NEWGEO';assert len(candidates)==len(decisions[arm])==128
        for fid,record in candidates.items():
            name=decisions[arm][fid]['selected']; chosen=next((h['pose'] for h in record['hypotheses'] if h['name']==name),record['current'])
            assert chosen==generated_poses[arm][fid]==new_poses[arm][fid]
            checked+=1
    assert checked==1024 and lock['checkpoint']==fit['checkpoint']==results['checkpoint']
    record=dict(created_at=C.now(),passed=True,selector_fits=1,student_fits=0,one_preserved_attempt=True,
        original_16_metrics_and_poses_exact=True,total_arms=24,same_checkpoint_all_8_models=True,
        final_cached_candidate_poses_exact=checked,selected_names_locked_before_real_scoring=True,
        normalization_TRAIN_only_bit_exact=True,normalization_train_pairs=int(train.sum()),
        validation_pairs=int(val.sum()),same_old_exact_renderer_parity=True,
        linear_parameters=95,optimizer_updates_per_epoch=steps_per_epoch,optimizer_updates=steps,
        optimizer_update_evidence='32 batches × executed epoch count, verified unchanged M.fit source loop; not an independent optimizer-hook trace.',
        best_epoch=fit['result']['best_epoch'],best_VAL_accuracy=fit['result']['best_val_accuracy'],
        early_best_synthetic_VAL_contract=True,no_TEST_scoring=True,
        synthetic_container_limitation=labels['physical_container_disclosure'],real_reference_training_reads=0,
        runtime_guard_read_paths_checked=True,new_path_inherited_manual_images=9,new_path_inherited_manual_corners=38,
        cumulative_research_manual_images=19,cumulative_research_manual_corners=86,
        deployment_limit='R0+newGEO removes the student at deployment only; calibration still uses student outputs.',
        evidence_limit='Repeated DEV and geometry-derived reference; no independent physical6D confirmation.',
        sources=[C.bind(K.PROTOCOL),C.bind(K.FEATURE_LOCK),C.bind(K.LABEL_LOCK),C.bind(K.FIT),
            C.bind(E.LOCK),C.bind(E.RESULT),C.bind(Path(__file__)),C.bind(Path(M.__file__)),*sources])
    C.save(C.DOC/'CALIBRATION_INTEGRITY_AUDIT.json',record,True)
    print('CALIBRATION_INTEGRITY_PASS',steps,'optimizer updates;',checked,'final candidate poses exact',flush=True)


if __name__=='__main__':main()
