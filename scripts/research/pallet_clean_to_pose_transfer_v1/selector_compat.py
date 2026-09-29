"""CPU-only frozen old-GEO compatibility; cached final candidates never change.

Use two separate processes: ``freeze`` denies metric/reference file reads;
``score`` connects already frozen names to pre-existing candidate metrics.
No selector fit, alternative feature recipe, or keypoint inference is exposed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import os
from pathlib import Path
import sys
import time

import numpy as np

from . import common as C

ARM = 'CLEAN_REF_OCC'
SEED = 42
PRIVATE = C.RAW/'selector_compat_S42'
LOCK = C.DOC/'ZERO_FIT_SELECTOR_LOCK.json'
RESULT = C.DOC/'ZERO_FIT_SELECTOR_RESULTS.json'
OLD = C.ROOT/'_docs/experiments/pallet_selector_recovery_v1'


def guard_reference_reads():
    reads = []
    forbidden = ('ORACLE_METRICS', 'POSE_METRICS.json', 'FRAME_METRICS.json',
        'GEOMETRY_RESOLVED', 'TRUTH_FOR_DISPLAY', 'VERIFIED_LABELS',
        'EVAL_RESULTS', 'CANDIDATE_ORACLE', '/data/evaluation/', 'AXIS_REVIEW')
    def hook(event,args):
        if event != 'open' or not isinstance(args[0],(str,bytes,os.PathLike)):
            return
        path=os.path.abspath(os.fsdecode(args[0]))
        writing=isinstance(args[1],str) and any(c in args[1] for c in 'wax+')
        if not writing:
            assert not any(token in path for token in forbidden), 'REFERENCE_READ_DENIED: '+path
            if path.startswith(str(C.ROOT)):
                reads.append(path)
    sys.addaudithook(hook)
    return reads


def verify_lock():
    lock=C.read(LOCK)
    assert lock['reference_or_metrics_read_before_decisions'] is False
    assert lock['new_selector_fits']==0 and lock['final_poses_from_cached_candidates']
    for binding in lock['files']+lock['sources']:
        C.verify(binding)
    return lock


def source_feature_shift(features,rows):
    """Descriptive shift only; never alter old scorer normalization or features."""
    from scripts.research.pallet_selector_recovery_v1.feature_contract import names
    feature_lock=OLD/'stage2_synth_scorer/SYNTH_PREDICTION_LOCK.json'
    split_lock=OLD/'stage2_synth_scorer/SYNTHETIC_SPLIT_LOCK.json'
    fl,sl=C.read(feature_lock),C.read(split_lock)
    for binding in (fl['features'],sl['inputs']):
        C.verify(binding)
    source=C.read(C.ROOT/sl['inputs']['path'])
    with np.load(C.ROOT/fl['features']['path']) as arrays:
        assert arrays['ids'].tolist()==[r['id'] for r in source]
        mask=np.asarray([r['split']=='TRAIN' for r in source]) & arrays['S1_valid']
        baseline=arrays['S1_geo'][mask].reshape(-1,94)
    mean=baseline.mean(0); std=np.maximum(baseline.std(0),1e-6)
    valid_ids=[r['id'] for r in rows if features[r['id']]['valid']]
    output={}
    for group in ('FULL128','NATURAL99','CLEAN29'):
        ids=[r['id'] for r in rows if r['id'] in valid_ids and
            (group=='FULL128' or (r['severity']=='CLEAN')==(group=='CLEAN29'))]
        target=np.asarray([features[fid]['features'] for fid in ids],np.float32).reshape(-1,94)
        standardized=(target-mean)/std
        shift=(target.mean(0)-mean)/std
        order=np.argsort(-np.abs(shift))[:10]
        output[group]=dict(frames=len(ids),candidate_rows=len(target),
            fraction_abs_z_gt3=float((np.abs(standardized)>3).mean()),
            fraction_abs_z_gt5=float((np.abs(standardized)>5).mean()),
            top10_absolute_standardized_mean_shifts=[dict(feature=names()[j],signed_shift=float(shift[j])) for j in order])
    return dict(reference='Old S1 synthetic TRAIN valid candidates only',
        reference_frames=int(mask.sum()),groups=output,
        real_GT_used=False,normalization_changed=False,
        limitation='Synthetic-to-real feature distribution shift is descriptive, not proof of selector error or an input for feature selection.'),[
            C.bind(feature_lock),C.bind(split_lock),fl['features'],sl['inputs']]


def freeze():
    if LOCK.exists():
        verify_lock(); print('ZERO_FIT_SELECTOR_ALREADY_LOCKED',flush=True); return
    start=time.monotonic(); reads=guard_reference_reads()
    import torch
    from scripts.research.pallet_selector_recovery_v1 import features as F,models as M,common as U
    from . import eval_student as E
    torch.set_num_threads(2)
    p=E.paths(SEED)
    candidate_lock=C.read(p['candidate_lock'])
    assert candidate_lock['no_reference_coordinates_read']
    for binding in candidate_lock['files']+candidate_lock['sources']:
        C.verify(binding)
    prediction_lock=C.read(p['lock'])
    assert prediction_lock['no_evaluation_reference_coordinates_read']
    for path in (p['metadata'],p['predictions']):
        binding=next(b for b in prediction_lock['files'] if b['path']==str(path.relative_to(C.ROOT)))
        C.verify(binding)
    scorer_lock_path=OLD/'stage2_synth_scorer/SCORER_SELECTION_LOCK.json'
    scorer_lock=C.read(scorer_lock_path)
    assert scorer_lock['winner']=='GEO_LINEAR'
    C.verify(scorer_lock['checkpoint'])
    ck=torch.load(C.ROOT/scorer_lock['checkpoint']['path'],map_location='cpu',weights_only=False)
    assert ck['d']==94 and ck['variant']=='GEO_LINEAR'
    rows=C.read(p['metadata']); E.validate_membership(rows)
    predictions=C.read(p['predictions'])[ARM]
    candidates=C.read(p['candidates'])[ARM]
    features,decisions,poses={},{},{}
    counts=Counter()
    for row in rows:
        fid=row['id'];record=candidates[fid]
        f=F.extract(predictions[fid],row['K'],row['xyz'],row['hw'])
        assert f['selection']==record['selected_name'], (fid,'D9 selection mismatch')
        assert [h['name'] for h in f['hypotheses']]==sorted(h['name'] for h in record['hypotheses'])
        selected=f['selection']; scores=None; fallback=None
        if f['valid']:
            scores=M.scores(ck,np.asarray(f['features'],np.float32)[None])[0]
            index=int(M.selection(scores[None],U.HYP)[0])
            assert index==1-int(M.selection(scores[None,::-1],U.HYP[::-1])[0])
            selected=U.HYP[index]
        else:
            fallback='Invalid feature pair: unchanged production D9'
        found=next((h for h in record['hypotheses'] if h['name']==selected),None)
        pose=found['pose'] if found is not None else record['current']
        # NEVER features.production_pose: current production refines the
        # selected extents with its own final corner0..7 SQPnP/LM solve.
        assert selected is None or pose.get('selected_hypothesis')==selected or not pose['available']
        features[fid]=f
        decisions[fid]=dict(selected=selected,D9_selected=record['selected_name'],
            scores=scores.tolist() if scores is not None else None,
            changed=selected!=record['selected_name'],valid_pair=bool(f['valid']),fallback=fallback)
        poses[fid]=pose
        counts['frames']+=1;counts['valid_pair']+=bool(f['valid'])
        counts['changed']+=selected!=record['selected_name'];counts['fallback']+=fallback is not None
    shift,shift_bindings=source_feature_shift(features,rows)
    files=[]
    for name,value in [('FEATURES',features),('DECISIONS',decisions),('POSES',poses),('FEATURE_SHIFT',shift)]:
        path=PRIVATE/(name+'.json');C.save(path,value,True);files.append(C.bind(path))
    sources=[C.bind(p['candidate_lock']),C.bind(p['candidates']),C.bind(p['lock']),
        C.bind(p['metadata']),C.bind(p['predictions']),C.bind(scorer_lock_path),
        scorer_lock['checkpoint'],C.bind(OLD/'SELECTOR_FEATURE_CONTRACT.json'),
        C.bind(Path(__file__)),C.bind(Path(F.__file__)),C.bind(Path(M.__file__)),*shift_bindings]
    C.save(LOCK,dict(created_at=C.now(),seed=SEED,arm=ARM,files=files,sources=sources,
        reference_or_metrics_read_before_decisions=False,read_guard_active=True,
        read_paths=sorted(set(reads)),old_scorer=scorer_lock['checkpoint'],counts=dict(counts),
        final_poses_from_cached_candidates=True,candidate_order_swap_test=True,
        selection='Frozen old shared Linear94 lower-score/name tie. Invalid pair keeps D9.',
        feature_shift=shift,new_selector_fits=0,student_fits=0,optimizer_updates=0,GPU_seconds=0,
        seconds=time.monotonic()-start),True)
    print('ZERO_FIT_SELECTOR_LOCKED',dict(counts),flush=True)


def score():
    lock=verify_lock()
    if RESULT.exists():
        result=C.read(RESULT)
        for binding in result['sources']+result['private_artifacts']:
            C.verify(binding)
        print('ZERO_FIT_SELECTOR_ALREADY_SCORED',flush=True);return
    from . import eval_student as E
    from scripts.research.pallet_pose_objective_followup_v2 import metric_baseline as M
    start=time.monotonic();p=E.paths(SEED)
    oracle=C.read(p['oracle_result'])
    for binding in oracle['private_artifacts']:
        C.verify(binding)
    choices=C.read(p['raw']/'ORACLE_METRICS_SELECTIONS_PRIVATE.json')['candidate_metrics'][ARM]
    decisions=C.read(PRIVATE/'DECISIONS.json');rows=C.read(p['metadata'])
    primary=C.read(p['results'])
    metric_path=p['raw']/'POSE_METRICS.json'
    C.verify(next(b for b in primary['private_artifacts'] if b['path']==str(metric_path.relative_to(C.ROOT))))
    existing=C.read(metric_path)
    metrics=dict(D9=existing[ARM],R0=existing['R0'],OLD_REF=existing['OLD_REF'],OLD_GEO={})
    for row in rows:
        fid=row['id'];selected=decisions[fid]['selected']
        # Only the prelocked deployment name indexes cached metric rows.
        # No metric/oracle criterion enters selection.
        chosen=next((h['metric'] for h in choices[fid] if h['name']==selected),None)
        metrics['OLD_GEO'][fid]=chosen if chosen is not None else existing[ARM][fid]
        if not decisions[fid]['changed']:
            assert metrics['OLD_GEO'][fid]==metrics['D9'][fid]
    groups=E.group_ids(rows)
    summaries={g:{arm:M.summarize(values[fid] for fid in ids) for arm,values in metrics.items()} for g,ids in groups.items()}
    paired={g:{'OLD_GEO-minus-'+base:M.paired(metrics[base],metrics['OLD_GEO'],ids)
        for base in ('D9','R0','OLD_REF')} for g,ids in groups.items()}
    primary_decision={base:M.classify_candidate(summaries['NATURAL99']['OLD_GEO'],summaries['NATURAL99'][base]) for base in ('D9','R0','OLD_REF')}
    private_path=PRIVATE/'SELECTED_METRICS_PRIVATE.json';C.save(private_path,metrics,True)
    result=dict(created_at=C.now(),seed=SEED,arm=ARM,groups=summaries,paired=paired,
        classification_NATURAL99=primary_decision,changed_by_group={g:sum(decisions[fid]['changed'] for fid in ids) for g,ids in groups.items()},
        prediction_lock=C.bind(LOCK),selection_was_frozen_before_metric_access=True,
        fixed_candidates_preserved=True,feature_shift=lock['feature_shift'],
        source_metrics='Existing current D9 candidate metrics indexed by prelocked selector name; not oracle selection.',
        unchanged_2D_coordinates=True,new_selector_fits=0,student_fits=0,optimizer_updates=0,GPU_seconds=0,
        private_artifacts=[C.bind(private_path)],sources=[C.bind(LOCK),C.bind(p['oracle_result']),*oracle['private_artifacts'],
            C.bind(p['results']),C.bind(metric_path),C.bind(Path(__file__)),C.bind(Path(M.__file__))],
        limitation='Reused DEV; two hypotheses only. Feature shifts are descriptive. No claim that old-GEO zero-fit is optimal or model-independent.',
        seconds=time.monotonic()-start)
    C.save(RESULT,M.clean(result),True)
    print('ZERO_FIT_SELECTOR_SCORED',primary_decision,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('freeze','score'))
    args=parser.parse_args();{'freeze':freeze,'score':score}[args.phase]()
