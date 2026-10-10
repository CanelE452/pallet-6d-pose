"""Seal method/input masks first, then actual ALL-F parity and evaluation-only parity."""
from collections import Counter
from pathlib import Path
import time
import cv2
import numpy as np
from . import common as C

def main():
    from .visibility import visibility
    from .adapter import correspondence_mask
    assert not C.DOC.exists(), 'Existing namespace must never be replaced'
    C.DOC.mkdir(parents=True)
    started=time.monotonic()
    binding_paths=[C.OLD_DOC/'PREDICTIONS.jsonl.gz',C.OLD_DOC/'VISIBILITY_LABELS.json',
       C.OLD_FINAL/'INPUT_AUDIT.json',C.OLD_FINAL/'INPUT_AND_METHOD_LOCK.json']
    source_paths=['scripts/research/pallet_dim_conditioned_p_v1/pose.py','scripts/research/pallet_dim_conditioned_p_v1/eval_math.py',
        'challenge/evaluation_v2/pnp_selector.py','scripts/paper/pose_metric_closure_v1/run_pose_evaluation.py',
        'data/pallet/results/paper_pose_metric_closure_v1/GEOMETRY_RESOLVED_POSE_GT.json']
    old=C.historical_rows()
    assert len(old)==319*4*3 and len({r['id'] for r in old})==319
    required=['q0','qN','qS','qFinal','prediction_support','canonical_observed','grade','session','seed','method','final_hypothesis','fixed_metadata','pose']
    for r in old:
        assert all(k in r for k in required)
        assert all(k in r['pose'] for k in ['translation_cm','rotation_deg','yaw_deg','IoU3D','ADDsym_m','ADDsym_normalized'])
        assert 'dimensions_pnp_WH_D_m' in r['fixed_metadata']
    lock=dict(status='LOCKED_BEFORE_ANY_NEW_RESULT',namespace='20261011',execution_client_date='2026-10-10',
        seeds=[1,2,3],methods=C.METHODS,primary='N3_THEN_SUBPIX_VIS minus N3_THEN_SUBPIX',
        primary_definitions=dict(confusion='rotation_deg > 45 and abs(yaw_deg) >= 60',success='translation_cm < 5 and rotation_deg < 5',
            seed_aggregation='threshold each seed, then average binary outcomes per original ID',no_pose='success false; observed confusion/full denominator descriptive only; any primary missing pose makes verdict NOT_ESTIMABLE'),
        verdict_rules=dict(WORSENED='confusion CI lower > 0 or success CI upper < 0',
            SUPPORTED='one improvement CI excludes zero, neither harmful CI excludes zero, >=2 seeds improve that metric',UNRESOLVED='otherwise'),
        A1_gate='stop before REAL if either primary frame-bootstrap CI excludes zero in harmful direction',
        bootstrap=dict(draws=10000,seed=20260917,real='13-session cluster, frame-pooled mean; retain 319 IDs',
            synthetic_primary='1985-frame paired',synthetic_secondary='1505-scenario cluster'),
        settings=dict(subpix_win=[5,5],subpix_zeroZone=[-1,-1],criteria_count=40,epsilon=.001,final_cap=.01,
            cap_reference='original BASE once after final combination',selector_score='original full nine-point score unchanged; mask only solver correspondences'),
        case_selection='seed1; biggest primary success recovery, biggest damage, biggest T gain and loss; ties ID ascending; no RGB publication',
        no_multiple_comparison_correction=True,no_training=True,no_tuning=True,no_new_annotations=True,
        square_gate='B4 explicit human axis approval required; B2/B3 procedural mismatch stops square work',
        new_core_pre_A0=C.core_bindings(),
        historical_bindings=[dict(path=str(p.relative_to(C.ROOT)),sha256=C.sha(p)) for p in binding_paths],
        source_bindings=[dict(path=p,sha256=C.sha(C.SOURCE/p)) for p in source_paths])
    C.write(C.DOC/'METHOD_LOCK.json',lock)
    inputs=[]
    for r in old:
        v=visibility(r['qFinal'],r['prediction_support'][:8])
        inputs.append(dict(id=r['id'],seed=r['seed'],method=r['method'],session=r['session'],grade=r['grade'],
            qFinal=r['qFinal'],prediction_support=r['prediction_support'],raw_hw=r['raw_hw'],
            fixed_metadata=r['fixed_metadata'],visibility=v))
    C.write_rows(C.DOC/'REAL_COORDINATES_MASKS.jsonl.gz',inputs)
    C.write(C.DOC/'COORDINATES_SEAL.json',dict(status='PASS',rows=len(inputs),
        sha256=C.sha(C.DOC/'REAL_COORDINATES_MASKS.jsonl.gz'),source_sha256=C.sha(C.OLD_DOC/'PREDICTIONS.jsonl.gz'),
        inference_fields=['qFinal','prediction_support','K','dimensions'],human_visibility_used=False,
        coordinates_changed=False,center_changed=False,mask_created_before_reference_loading_in_this_execution=True))
    # All geometry/model hashes are recorded before invoking the old scorer.
    input_audit=C.read(C.OLD_FINAL/'INPUT_AUDIT.json')
    assert input_audit['status']=='PASS'
    for model in input_audit['models']:
        for key in ['checkpoint','prediction']:
            entry=model[key];assert C.sha(C.SOURCE/entry['path'])==entry['sha256'],(model['seed'],key)
    for b in input_audit['inputs']:
        for key in ['image','detector_feature_cache','annotation']:
            entry=b[key]; assert C.sha(C.SOURCE/entry['path'])==entry['sha256'],(b['id'],key)
    C.write(C.DOC/'INPUT_AUDIT.json',dict(status='PASS',rows=len(old),images=319,all_required_fields_present=True,
        original_input_images_caches_annotations_authenticated=319,source_bindings=lock['source_bindings'],
        checkpoint_bindings=input_audit['models'],grades=dict(Counter(r['grade'] for r in old if r['seed']==1 and r['method']=='BASE'))))
    cv2.setNumThreads(1)
    from scripts.research.pallet_n3_subpix_final_20261010.evaluate import score, same
    load_real,_,_=C.existing()
    E,frames,targets,_,_=load_real()
    frame_by_id={f['id']:f for f in frames}
    errors=[];counts=Counter()
    for i,r in enumerate(x for x in old if x['seed']==1):
        f=frame_by_id[r['id']];points=np.array(r['qFinal'],float)
        with correspondence_mask(points,f['K'],np.ones(8,bool)) as audit:
            replay=score(E,f,points,targets[r['id']],r['method'],1)
        same(r['pose'],C.finite(replay['pose']),r['id']+'/'+r['method']+'/pose')
        same(r['actual_pose'],C.finite(replay['actual_pose']),r['id']+'/'+r['method']+'/R_t')
        assert r['final_hypothesis']==replay['final_hypothesis']
        same(r['corner'],C.finite(replay['corner']),r['id']+'/corner')
        errors.extend(abs(r['pose'][k]-replay['pose'][k]) for k in ['translation_cm','rotation_deg','yaw_deg','IoU3D','ADDsym_m','ADDsym_normalized'])
        counts.update(replay['PnP_counts'])
        if i%160==0:print('A0_ALL_PARITY',i+1,1276,flush=True)
    # Human states are read only after the GT-free visibility masks are sealed.
    labels={(r['id'],r['corner']):r['category'] for r in C.read(C.OLD_DOC/'VISIBILITY_LABELS.json')['labels']}
    parity={}
    for method in ['N3_DIM_SYM','BASE']:
        selected=[r for r in inputs if r['seed']==1 and r['method']==method]
        hidden=Counter();table={}
        for r in selected:
            v=r['visibility'];hidden[v['hidden_count']]+=1
            for k,show in enumerate(v['visible_mask']):
                category=labels.get((r['id'],k))
                if category:
                    table.setdefault(category,dict(hidden=0,visible=0))['visible' if show else 'hidden']+=1
        expected=({1:196,2:123},[(411,51),(26,1750),(3,215),(2,41)]) if method=='N3_DIM_SYM' else ({1:197,2:122},[(408,54),(27,1749),(4,214),(2,41)])
        assert dict(hidden)==expected[0]
        for category,(h,v) in zip(['SELF_OCCLUDED','DIRECT_VISIBLE','EXTERNAL_OCCLUDED','OUT_OF_FRAME'],expected[1]):assert table[category]==dict(hidden=h,visible=v)
        assert sum(r['visibility']['facing']['front'] for r in selected)==319
        parity[method]=dict(front_facing=319,hidden_counts=dict(hidden),direct_index_diagnostic=table)
    C.write(C.DOC/'A0.json',dict(status='PASS',actual_F_calls=1276,images=319,methods=4,seed=1,
        absolute_tolerance=1e-7,max_pose_metric_absolute_delta=max(errors),all_R_t_and_corner_fields_PASS=True,
        PnP_counts=dict(counts),visibility_parity=parity,elapsed_seconds=time.monotonic()-started,
        geometry_note='Specified face winding is inward in cuboid 3D basis; exact 2D positive-area rule retained and parity passes',
        threshold_note='Legacy evaluate_real uses non-symmetry single-R threshold; preregistered proper-group metric is used here'))
    print('A0_PASS',max(errors),flush=True)

if __name__=='__main__':main()
