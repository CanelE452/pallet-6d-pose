"""Canonical scorer metadata compatibility; frozen selection/predictions stay intact."""
from env import *

def run():
    assert complete('TRAIN_COMPLETE')
    if complete('EVALUATE_COMPLETE'):
        verify(); return
    start=now(); verify()
    import evaluate_prior as original
    if not all(complete(f'PRIOR{s}_INFERENCE') for s in (1,2,3)):
        assert not gpu()['foreign_compute']
        original.selection(); original.infer()
    source=DOC/'PRIOR_SELECTION.json'; selected=read(source)
    assert selected['no_real_selection'] is True
    assert selected['source_partitions']==dict(calibration=1004,selection=1031,heldout=1985)
    compatibility=DOC/'PRIOR_SELECTION_CANONICAL_METADATA.json'
    freeze(compatibility,dict(selected,selection_population='synth_val',
        metadata_compatibility_source=bound(source),
        amendment='Add required canonical scorer population label only; original selection, rules, predictions and all weights are unchanged.'))
    pe=old('paper_evaluation')
    derived=[]
    for seed in (1,2,3):
        for raw in (False,True):
            name=f'PRIOR{seed}'+('_raw' if raw else '')
            path=RAW/f'evaluation/PRIOR{seed}'
            original_rep=path/('RAW_POINT_REPLACEMENTS.json' if raw else 'POINT_REPLACEMENTS.json')
            rep=path/('RAW_POINT_CANONICAL_METADATA.json' if raw else 'POINT_CANONICAL_METADATA.json')
            incoming=read(original_rep)
            assert incoming['selection_artifact']['sha256']==sha(source)
            adapted=dict(incoming,selection_artifact=dict(path=str(compatibility),sha256=sha(compatibility)))
            assert adapted['frames']==incoming['frames']
            freeze(rep,adapted); derived.append(dict(original=bound(original_rep),adapted=bound(rep),frames_exact=True))
            if complete(name+'_SCORED'): continue
            t=now(); dst,frames,altered=pe.replace_points(RAW,LINE/'baseline/FULL_CANDIDATES.json',rep,name)
            pe.paper_2d(dst,dst/'PREDICTIONS.json',str(R0)); pe.paper_pose(dst,frames,name)
            receipt(name+'_SCORED',[original_rep,rep,compatibility,HERE/'evaluate_compatible.py'],
                [dst/f for f in ('PREDICTIONS.json','PAPER_2D.json','PAPER_2D_per_frame.csv','POSE_PER_FRAME_BY_ARM.json')],t,altered=altered)
            print('PRIOR_CANONICAL_SCORED',name,flush=True)
    write(DOC/'SCORER_METADATA_AMENDMENT.json',dict(complete=True,time=now(),
        cause='Original frozen selection lacked selection_population field required by canonical scorer.',
        original_selection=bound(source),canonical_selection=bound(compatibility),derived=derived,
        no_model_or_prediction_or_rule_change=True,additional_training_updates=0,
        original_failure_time='2026-09-15T02:19:25.323096+00:00',
        original_failure_message='Only frozen synthetic selection may precede paper DEV evaluation'))
    from prior_analysis import run as analysis
    analysis()
    from submission_runtime import run as runtime
    runtime()
    receipt('EVALUATE_COMPLETE',[DOC/'PRIOR_DEV_LOCK.json',HERE/'evaluate_compatible.py',DOC/'SCORER_METADATA_AMENDMENT.json'],
        [DOC/'UNIFIED_DEV_RESULTS.json',DOC/'P_VS_PRIOR_PAIRED.json',DOC/'RUNTIME_PANEL.json'],start)
