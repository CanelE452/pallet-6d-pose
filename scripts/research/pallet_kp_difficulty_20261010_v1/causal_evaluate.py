"""Solve new causal ablations without GT, seal them, then score all319 IDs."""
import argparse
from collections import Counter
from pathlib import Path
import time
import cv2
import numpy as np
from ..pallet_observation_refiner_20261009_v1 import common as C
from ..pallet_observation_refiner_20261009_v1.solver import HypothesisBank
from ..pallet_observation_refiner_20261009_v1.inference import finish, hidden_mask
from ..pallet_observation_refiner_20261009_v1.evaluate import packet, score
from .dimension_prior import DimensionPriorBank

DOC=C.WORKTREE/'_docs/experiments/pallet_kp_difficulty_20261010_v1'
PRIVATE=Path(__import__('os').environ.get('PALLET_KP_DIFFICULTY_SCRATCH','/dev/shm/pallet-kp-difficulty-private-20261010'))


def parity(a,b):
    for field in ('new_pose_estimated','fallback_used','no_pose','hidden_initial','reprojected_ids','excluded'):
        assert a[field]==b[field],(a.get('id'),field,a[field],b[field])
    maximum=0.
    for field in ('native_points',):
        x,y=np.asarray(a[field],float),np.asarray(b[field],float)
        assert x.shape==y.shape and np.array_equal(np.isfinite(x),np.isfinite(y))
        if np.isfinite(x).any():maximum=max(maximum,float(np.max(np.abs(x[np.isfinite(x)]-y[np.isfinite(y)]))))
    x,y=a['actual_pose'],b['actual_pose']
    assert x.get('available')==y.get('available')
    if x.get('available'):
        for field in ('R_cf','R_physical','centroid','cf_extents'):
            maximum=max(maximum,float(np.max(np.abs(np.asarray(x[field])-np.asarray(y[field])))))
    assert maximum<=1e-7,(a.get('id'),maximum)
    return maximum


def dimension_profiles(bank,hidden):
    U=tuple(i for i in bank.eligible if i not in hidden)
    output=[]
    for index,dim in enumerate(bank.dims):
        candidates=[c for c in (bank.hypotheses or []) if c.dim==index and set(c.ids)<=set(U)]
        if not candidates:
            output.append(dict(dimension_index=index,cf_extents=dim.tolist(),candidate_count=0));continue
        c=min(candidates,key=lambda x:bank._score(x,U,True)[0]);key,ids,residual=bank._score(c,U,True)
        output.append(dict(dimension_index=index,cf_extents=dim.tolist(),candidate_count=len(candidates),
            inlier_count=len(ids),inliers=list(ids),truncated_sse_px2=key[1],generator_ids=list(c.ids),
            R_cf=cv2.Rodrigues(c.rvec)[0].tolist(),centroid=c.tvec.reshape(3).tolist(),
            pre_refit_only=True))
    return output


def run(pilot):
    begin=time.monotonic();cv2.setNumThreads(1)
    checks=C.read(DOC/'DIMENSION_PRIOR_CHECKS.json');assert checks['passed']
    protocol=C.read(DOC/'PROTOCOL.json');frames=C.read(C.DOC/'INPUTS.json')['frames']
    prior_rows={}
    for row in C.iter_rows(C.DOC/'OBSERVATIONS.jsonl.gz'):
        if row['method'] in [a+'_'+s for a in ('BASE','N3_SUBPIX') for s in ('NO_MASK_STANDARD','NO_MASK_ROBUST','GEOM_NOSELF_ROBUST')]:
            prior_rows[(row['method'],row['id'])]=row
    if pilot:
        panel={r['id'] for r in C.read(C.DOC/'RUNTIME.json')['panel']}
        frames=[f for f in frames if f['id'] in panel]
        assert len(frames)==26
        raw_path=PRIVATE/'DIMENSION_PILOT_ROWS.jsonl.gz';receipt_path=DOC/'CPU_PILOT.json'
    else:
        raw_path=DOC/'CAUSAL_GEOMETRY_SEALED.jsonl.gz';receipt_path=DOC/'CAUSAL_POSE_EXECUTION.json'
        assert (DOC/'MATCH_MASS_OBSERVATIONS.jsonl.gz').is_file()
    assert not raw_path.exists() and not receipt_path.exists(), 'Preserve completed follow-up results'
    new=[];replays=[];profiles=[];counts=Counter();bankrows=[];max_parity=0.
    code=[C.binding(Path(__file__)),C.binding(Path(__file__).with_name('dimension_prior.py')),
          C.binding(Path(__file__).parents[1]/'pallet_observation_refiner_20261009_v1/solver.py')]
    with C.no_truth_reads():
        for i,f in enumerate(frames):
            for arm in ('BASE','N3_SUBPIX'):
                q=np.asarray(f['points'][arm],float)
                initial=prior_rows[(arm+'_NO_MASK_STANDARD',f['id'])]['initial_pose']
                H,diagnostic=hidden_mask(initial)
                bank=HypothesisBank(q,f['K'],f['xyz'],image_size=(f['raw_hw'][1],f['raw_hw'][0]))
                prior=DimensionPriorBank(bank,initial)
                for label,hidden in [('NO_MASK',[]),('GEOM_NOSELF',H)]:
                    unlocked=finish(bank,q,initial,excluded=hidden,hidden=hidden)
                    unlocked.update(input_points=q,initial_pose=initial,mask_diagnostic=diagnostic,oracle=False)
                    unlocked_row=packet(f,arm+'_'+label+'_ROBUST',unlocked)
                    maximum=parity(unlocked_row,prior_rows[(unlocked_row['method'],f['id'])]);max_parity=max(max_parity,maximum)
                    unlocked_row['unlocked_replay_only']=True
                    replays.append(unlocked_row)
                    locked=finish(prior,q,initial,excluded=hidden,hidden=hidden)
                    locked.update(input_points=q,initial_pose=initial,mask_diagnostic=diagnostic,oracle=False,
                                  inference_GT_input=False,coordinate_branch=arm,causal_change='initial dimension prior only',
                                  no_match_points_filled_from_Base=False)
                    new.append(packet(f,arm+'_INITIAL_DIMENSION_PRIOR_'+label,locked))
                    profiles.append(dict(id=f['id'],coordinate_branch=arm,mask=label,used_ids=unlocked['solver']['used'],
                        initial_cf_extents=initial.get('cf_extents'),initial_dimension_index=prior.index,
                        pre_refit_best_per_dimension=dimension_profiles(bank,hidden),
                        unlocked_final_cf_extents=unlocked['actual_pose'].get('cf_extents'),
                        locked_final_cf_extents=locked['actual_pose'].get('cf_extents'),
                        unlocked_final_truncated_sse_px2=unlocked['solver'].get('truncated_sse_px2'),
                        locked_final_truncated_sse_px2=locked['solver'].get('truncated_sse_px2')))
                counts.update(bank.ledger)
                bankrows.append(dict(id=f['id'],branch=arm,ledger=bank.ledger.copy()))
            if i%26==0:print('CAUSAL_DIMENSION',i+1,len(frames),round(time.monotonic()-begin,2),flush=True)
        if not pilot:
            by_id={f['id']:f for f in frames}
            for observation in C.iter_rows(DOC/'MATCH_MASS_OBSERVATIONS.jsonl.gz'):
                original_arm=observation.get('original_method',observation.get('original_arm',observation.get('arm')))
                method=observation['method'];fid=observation['id'];f=by_id[fid]
                assert method in protocol['new_methods'],method
                original=np.asarray(f['points']['BASE'],float)
                q=np.full((9,2),np.nan);q[8]=original[8]
                for corner in observation['corners']:q[corner['id']]=corner['xy']
                initial=prior_rows[('BASE_NO_MASK_STANDARD',fid)]['initial_pose'];H,diagnostic=hidden_mask(initial)
                bank=HypothesisBank(q,f['K'],f['xyz'],image_size=(f['raw_hw'][1],f['raw_hw'][0]))
                settings=[(method,H)]
                if method=='IMAGE_ROLE_MATCH_MASS':settings.append(('IMAGE_ROLE_MATCH_MASS_NO_MASK',[]))
                for name,hidden in settings:
                    answer=finish(bank,original,initial,excluded=hidden,hidden=hidden)
                    if answer['new_pose_estimated']:
                        for corner in observation['corners']:
                            if corner['id'] not in hidden:answer['native_points'][corner['id']]=corner['xy']
                    answer.update(input_points=q,initial_pose=initial,mask_diagnostic=diagnostic,oracle=False,
                        inference_GT_input=False,coordinate_branch='BASE',causal_change='summed match existence only',
                        selected_corner_ids=[c['id'] for c in observation['corners']],
                        observation_raw_logits_sha256=observation['raw_logits_sha256'],
                        no_match_points_filled_from_Base=False,reprojections_reused_as_observations=False)
                    new.append(packet(f,name,answer))
                counts.update(bank.ledger)
                bankrows.append(dict(id=fid,branch=method,ledger=bank.ledger.copy()))
    expected=(4 if pilot else 8)*len(frames);assert len(new)==expected
    C.save_rows(raw_path,new)
    C.save_rows(PRIVATE/('PILOT_UNLOCKED_REPLAY.jsonl.gz' if pilot else 'UNLOCKED_REPLAY.jsonl.gz'),replays)
    if not pilot:
        C.save_rows(DOC/'DIMENSION_PROFILE_ROWS.jsonl.gz',profiles)
        C.save_rows(DOC/'UNLOCKED_REPLAY_ROWS.jsonl.gz',replays)
    receipt=dict(schema='kp_difficulty_actual_numeric_execution_v1',pilot_only=pilot,
                 complete=True,frames=len(frames),new_pose_paths=len(new),original_unlocked_replay_paths=len(replays),
                 unlocked_numeric_max_difference=max_parity,all_unlocked_replays_passed=True,
                 counts=dict(counts),banks=bankrows,GT_reads_blocked_before_seal=True,
                 initial_poses_reused_from_unchanged_same_coordinate_inputs=True,
                 initial_pose_actual_recomputed=0,new_detector_forwards=0,new_head_forwards_real=0,
                 new_training_updates=0,new_RGB=0,raw_geometry=C.binding(raw_path),code=code,
                 solver_wall_seconds=time.monotonic()-begin,not_deployment_latency=True)
    if pilot:
        C.write(receipt_path,receipt)
        print('CPU_PILOT_COMPLETE',len(new),len(replays),round(time.monotonic()-begin,2),flush=True)
        return
    # Pose and corner reference objects are first opened after all new outputs are sealed.
    C.source_modules()
    from scripts.research.pallet_training_free_compare_20261007_v1.common import load_real
    E,real,targets,_,_=load_real();real={f['id']:f for f in real}
    scored=[score(E,real[row['id']],targets[row['id']],row) for row in new]
    replay_scored=[score(E,real[row['id']],targets[row['id']],row) for row in replays]
    originals={(r['method'],r['id']):r for r in C.iter_rows(C.DOC/'PREDICTIONS.jsonl.gz')}
    max_metric=0.
    for row in replay_scored:
        old=originals[(row['method'],row['id'])]
        assert row['pose']['available']==old['pose']['available']
        for field in ('translation_cm','rotation_deg','ADDsym_m'):
            if row['pose']['available']:max_metric=max(max_metric,abs(row['pose'][field]-old['pose'][field]))
    assert max_metric<=1e-7,max_metric
    C.save_rows(DOC/'PREDICTIONS.jsonl.gz',scored)
    receipt.update(scored_rows=len(scored),score_phase_after_geometry_seal=True,unlocked_metric_max_difference=max_metric,
                   total_wall_seconds=time.monotonic()-begin,scored=C.binding(DOC/'PREDICTIONS.jsonl.gz'),
                   metric_reference='unchanged geometric reconstruction, not independent physical ground truth')
    C.write(receipt_path,receipt)
    print('CAUSAL_EVALUATE_COMPLETE',len(scored),round(time.monotonic()-begin,2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true')
    run(p.parse_args().pilot)
