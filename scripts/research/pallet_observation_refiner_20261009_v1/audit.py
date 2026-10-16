"""Seal existing prediction inputs and the source-only protocol before new poses."""
from pathlib import Path
import json
import os
import numpy as np
from . import common as C

OLD=C.ROOT/'_docs/experiments/pallet_n3_subpix_20261008_v1'
BOUNDARY=C.ROOT/'_docs/experiments/pallet_visible_boundary_20261009_v1'
VIS=C.ROOT/'_docs/experiments/pallet_combined_closeout_20261003_v1/closeout_20261006_v1/visibility_square/STATIC_VISIBILITY_MERGE_AUDIT.json'

def run():
    assert not (C.DOC/'PROTOCOL.json').exists(), 'Protocol already sealed'
    C.DOC.mkdir(parents=True,exist_ok=True);C.SCRATCH.mkdir(parents=True,exist_ok=True)
    C.write(C.SCRATCH/'INITIAL_STATE.json',C.source_state())
    prior=C.read(OLD/'PROTOCOL.json');inputs=C.read(BOUNDARY/'INPUTS.json')['frames']
    old={m:{} for m in C.CONTROLS}
    for r in C.iter_rows(OLD/'PREDICTIONS.jsonl.gz'):old[r['method']][r['id']]=r
    assert all(len(d)==319 for d in old.values())
    sealed=[]
    for f in inputs:
        fid=f['id'];q=np.asarray(f['q0'],float)
        assert C.digest(q)==C.digest(old['BASE'][fid]['native_points'])
        assert C.sha(C.ROOT/f['image'])==f['image_sha256']
        sealed.append(dict(id=fid,session=f['session'],image=f['image'],raw_hw=f['raw_hw'],K=f['K'],
            xyz=f['dimensions_whd_m'],points={'BASE':f['q0'],'N3_SUBPIX':old['N3_SUBPIX'][fid]['native_points']},
            selected_index=f['selected_index'],candidate_metadata=f['candidate_metadata'],
            prediction_support=f['prediction_support'],image_sha256=f['image_sha256']))
    ids=[r['id'] for r in sealed];assert len(ids)==len(set(ids))==319
    assert len({r['session'] for r in sealed})==13
    C.write(C.DOC/'INPUTS.json',dict(schema='prediction_only_inputs_v1',frames=sealed,GT_input=False))
    paths=[OLD/'PROTOCOL.json',OLD/'PREDICTIONS.jsonl.gz',BOUNDARY/'INPUTS.json',BOUNDARY/'COORDINATES.jsonl.gz',
           BOUNDARY/'COORDINATES_LOCK.json',VIS,C.ROOT/prior['N3_checkpoint']['path'],
           C.ROOT/'data/pallet/results/pallet_posefix_replay_diagnosis_v1/TARGETS.json',
           C.ROOT/'data/pallet/results/paper_pose_metric_closure_v1/GEOMETRY_RESOLVED_POSE_GT.json',
           C.ROOT/'data/pallet/results/pallet_n3_completion_v3/runtime/dope_seed1.json']
    C.write(C.DOC/'SOURCE_BINDINGS.json',dict(inputs=[C.binding(p) for p in paths],
        raw_images=[dict(id=r['id'],path=r['image'],sha256=r['image_sha256']) for r in sealed],
        permission='Existing local research inputs read-only; no new real capture/annotation',
        attachment=C.binding(Path(os.environ['PALLET_INSTRUCTION_PATH']))))
    C.write(C.DOC/'PROTOCOL.json',dict(schema='observation_refiner_robust_pnp_v1',date='2026-10-10',
        attachment_date='2026-10-09',base_commit=C.source_state()['head'],branch='research/observation-refiner-robust-pnp-20261009',
        population=dict(frames=319,sessions=13,ids=ids),inputs=C.binding(C.DOC/'INPUTS.json'),
        methods=[f'{a}_{b}' for a in ('BASE','N3_SUBPIX') for b in C.SUFFIXES],
        controls=list(C.CONTROLS),shared_boundary='BASE_BOUNDARY_NATIVE existing actual intersected corners only',
        solver=dict(algorithm='exhaustive four-subset SQPnP/IPPE consensus; up to3 inlier refits',threshold_px=8.,
                    max_subsets_per_dimension=70,max_dimensions=2,min_corners=4,GT_selection=False,
                    tie_break='fixed hypothesis enumeration; no initial pose prior'),
        mask=dict(initial='existing unchanged full-coordinate legacy pose',margin_deg=2,
                  final_mask_changes_are_diagnostic=True,confidence_gate=False,
                  human='ORACLE_MASK_AND_PHASE; baseline fixed symmetry branch; unannotated not inferred'),
        fallback='same-coordinate historical pose; record separately; never refit reprojections',
        bootstrap=dict(seed=20260917,resamples=10000,sessions=13,shared_existing_draws=True),
        budget=dict(A2=3828,A3_geometry=2816,A3_real=638,C_new=2552,CPU_seconds=3600,
                    cache_bytes=20*1024**3,render_seconds=2700,train_seconds=3600,feature_seconds=1200),
        learning=dict(conditional_on_valid_exact_physical_boundary_supervision=True,models=['GEOMETRY_ONLY','IMAGE_NO_ROLE','IMAGE_ROLE'],
                      updates_each=3000,batch16=True,seed=1,lr=.001,weight_decay=.0001,warmup=100,checkpoint='last'),
        prohibited=dict(real_capture=0,new_manual_annotation=0,N3_training=0,extra_seeds=0,pose_loss=0,
                        PnP_backprop=0,large_models=0,paper_changes=0,outcome_tuning=0,main_merge=0,force_push=0),
        evaluation_reference='Existing geometry-reconstructed reference; not independent measured physical truth',
        timing=dict(arms=['BASE','N3_SUBPIX','N3_SUBPIX_GEOM_NOSELF_ROBUST','IMAGE_ROLE conditional'],
                    warmup_each=20,panel=26,repeats=5,actual_detector_each_call=True,concurrent_benchmarks=False)))
    print('AUDIT SEALED',C.sha(C.DOC/'PROTOCOL.json'),flush=True)

if __name__=='__main__':run()
