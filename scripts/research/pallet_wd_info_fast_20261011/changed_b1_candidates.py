"""Recover runtime-only B1 candidate values for every sealed changed case.

No reference data, fitting, decoding, image reads, or configuration changes.
Original outcome rows remain immutable. Only the fourteen already identified
method/frame pairs are replayed through the unchanged F, with both initial
selector fits captured and the selected final8 pose retained separately.
"""
from collections import Counter
from contextlib import contextmanager
import csv
import fcntl
import hashlib
import json
from pathlib import Path
import re
import time

import cv2
import numpy as np

from . import common as C
from . import geometry_analysis as A

PRIOR = 49315
LIMIT = 84
JOB = 'CHANGED_B1_CANDIDATES'
ID = re.compile(r'"id"\s*:\s*"([^"\\]+)"')


@contextmanager
def count_calls():
    ledger = C.PRIVATE / 'PNP_RECEIPTS.jsonl'
    with ledger.open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        before = [json.loads(s) for s in stream if s.strip()]
        assert sum(r['total'] for r in before) == PRIOR
        assert not any(r['job'] == JOB for r in before)
        originals, calls = {}, {}
        started = time.monotonic()
        for name in ('solvePnP', 'solvePnPGeneric', 'solvePnPRansac', 'solvePnPRefineLM', 'solvePnPRefineVVS'):
            if not hasattr(cv2, name):
                continue
            originals[name] = getattr(cv2, name)
            calls[name] = 0
            def counted(*args, _name=name, _fn=originals[name], **kwargs):
                assert sum(calls.values()) < LIMIT and PRIOR + sum(calls.values()) < 50000
                calls[_name] += 1
                return _fn(*args, **kwargs)
            setattr(cv2, name, counted)
        receipt = {}
        try:
            yield calls, receipt
        finally:
            for name, fn in originals.items():
                setattr(cv2, name, fn)
            receipt.update(job=JOB, limit=LIMIT, calls=calls, total=sum(calls.values()),
                prior_global_actual=PRIOR, global_actual_after=PRIOR+sum(calls.values()),
                wall_seconds=time.monotonic()-started)
            stream.seek(0, 2)
            stream.write(json.dumps(receipt)+'\n')
            stream.flush()
            fcntl.flock(stream, fcntl.LOCK_UN)


def main():
    started = time.monotonic()
    cv2.setNumThreads(1)
    plan = C.lock_ready()
    allowed = set(C.read(C.DOC/'INPUT_CONTRACT.json')['allowed_ids'])
    files = [C.DOC/name for name in ('B1_FRAME_ROWS.jsonl.gz', 'B1_PERMUTATION_REPAIR_FRAME_ROWS.jsonl.gz',
        'B1_RESULTS.json', 'B1_PERMUTATION_REPAIR_RESULTS.json', 'PLAN_LOCK.json',
        'AE_CANDIDATE_METRICS.jsonl.gz', 'B1_PREDICTION_SEAL.json', 'B1_PERMUTATION_REPAIR_PREDICTION_SEAL.json')]
    files += [C.PRIVATE/'B1_PREDICTIONS_SEALED.json', C.PRIVATE/'B1_PERMUTATION_REPAIR_PREDICTIONS_SEALED.json',
        C.PUBLIC/A.WD_REL/'STAGE1_ROWS_REAL.jsonl.gz',
        C.SOURCE/'scripts/research/pallet_dim_conditioned_p_v1/pose.py',
        C.SOURCE/'scripts/paper/pose_metric_closure_v1/run_pose_evaluation.py',
        C.SOURCE/'challenge/evaluation_v2/pnp_selector.py']
    bindings = [C.binding(p) for p in files]
    old_rows, perm_rows = [list(C.rows(p)) for p in files[:2]]
    baseline = {r['id']:r for r in old_rows if r['method']=='B1_CORRECT'}
    assert set(baseline)==allowed and len(baseline)==173
    outcomes = {(r['method'], r['id']):r for r in old_rows+perm_rows}
    events = []
    for r in old_rows+perm_rows:
        b = baseline[r['id']]
        values = [('AXIS_SELECTION', b['selected_hypothesis'], r['selected_hypothesis']),
                  ('STRICT_SUCCESS', b['success'], r['success']),
                  ('POSE_AVAILABLE', int(b['pose'].get('available',False)), int(r['pose'].get('available',False)))]
        for event, old, new in values:
            if old!=new:
                events.append(dict(method=r['method'], id=r['id'], event=event, before=old, after=new))
    pairs = sorted({(r['method'],r['id']) for r in events},
        key=lambda x:(hashlib.sha256(json.dumps(x,separators=(',',':')).encode()).hexdigest(),x))
    assert len(pairs)==14 and len(events)==14 and len(pairs)*6==LIMIT
    reservations=sum(plan['pnp_reservations'].values())+1038+LIMIT
    assert reservations==49838 and reservations<=50000 and PRIOR+LIMIT<=50000
    pair_digest=A.digest(pairs)
    code=C.binding(Path(__file__).resolve())
    C.write(C.DOC/'CHANGED_B1_CANDIDATES_LOCK.json',dict(
        status='LOCKED_BEFORE_F_REPLAY', code=code, inputs=bindings,
        fixed_hash_sorted_method_id_pairs=pairs, pair_sha256=pair_digest,
        events=events, new_call_limit=LIMIT, prior_global_actual=PRIOR,
        reservation_total=reservations, global_limit=50000,
        reference_inputs=False, new_fits=0, new_CNN_forwards=0, new_configuration=False,
        replay='Unchanged F.infer; two initial selector fits and selected final8 pose only',
        baseline='Cached AE N3 seed1 final8 metrics reused; semantics differ from initial selector fits'))
    predictions={}
    for path in files[8:10]:
        for r in C.read(path):
            name=r['method'] if r['method'].startswith('B1_') else 'B1_'+r['method']
            predictions[(name,r['id'])]=r
    assert all(key in predictions for key in pairs)
    metadata={}
    with __import__('gzip').open(files[10],'rt') as stream:
        for line in stream:
            m=ID.search(line)
            if m and m.group(1) in {fid for _,fid in pairs}:
                r=json.loads(line)
                if r['seed']==1 and r['method']=='N3_DIM_SYM':
                    metadata[r['id']]=r['fixed_metadata']
    assert set(metadata)=={fid for _,fid in pairs}
    cached={r['id']:r for r in C.rows(files[5])
            if r['backbone']=='YOLO' and r['method']=='N3_DIM_SYM' and r['seed']==1}
    assert set(cached)==allowed
    pose=A.pose_api(C.SOURCE)
    captured={}
    with count_calls() as (calls,counter_receipt):
        for key in pairs:
            method,fid=key; q=np.asarray(predictions[key]['q'],float);meta=metadata[fid]
            assert q.shape==(9,2) and np.isfinite(q).all()
            trace=[];original=pose.select_pnp_hypotheses
            def capture(*args,**kwargs):
                result=original(*args,**kwargs);trace.append(result);return result
            pose.select_pnp_hypotheses=capture
            try:
                actual=pose.infer(q,np.asarray(meta['K'],float),np.asarray(meta['dimensions_pnp_WH_D_m'],float),False)
            finally:
                pose.select_pnp_hypotheses=original
            assert len(trace)==1 and actual['available']==outcomes[key]['pose']['available']
            assert actual.get('selected_hypothesis')==outcomes[key]['selected_hypothesis']
            candidate={h.name:dict(success=h.success, score=h.score, score_components=dict(h.score_components),
                R_cf=h.rotation_camera_facing, centroid_m=h.translation_camera_facing,
                cf_extents_m=h.camera_facing_dimensions.as_dict(), failure=h.failure_reason,
                semantics='Original F initial selector fit; not final8 alternative pose') for h in trace[0].hypotheses}
            assert set(candidate)==set(A.NAMES)
            captured[key]=dict(candidate_selector_values=candidate, selected_final8_pose=actual)
    assert sum(calls.values())==LIMIT
    destination=C.DOC/'CHANGED_B1_CANDIDATES_COMPLETE.csv'
    fields=['method','id','seed','session','event','before','after','old_axis_error','new_axis_error',
        'old_success','new_success','old_pose_metrics_json','new_pose_metrics_json',
        'old_candidates_json','new_candidates_json','new_selected_final8_pose_json',
        'old_candidate_semantics','new_candidate_semantics']
    with destination.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for event in events:
            fid=event['id'];key=(event['method'],fid);r=outcomes[key];b=baseline[fid]
            serial=lambda x:json.dumps(C.clean(x),sort_keys=True,separators=(',',':'),allow_nan=False)
            writer.writerow(dict(event,seed=1,session=r['session'],old_axis_error=b['axis_error'],new_axis_error=r['axis_error'],
                old_success=b['success'],new_success=r['success'],old_pose_metrics_json=serial(b['pose']),
                new_pose_metrics_json=serial(r['pose']),old_candidates_json=serial(dict(
                    candidate_metrics=cached[fid]['candidate_metrics'],candidate_scores=cached[fid]['candidate_scores'])),
                new_candidates_json=serial(captured[key]['candidate_selector_values']),
                new_selected_final8_pose_json=serial(captured[key]['selected_final8_pose']),
                old_candidate_semantics='Reused AE N3 seed1 final8 metrics and original selector scores',
                new_candidate_semantics='Captured original F initial selector fits, plus selected final8 pose separately'))
    assert [C.binding(p) for p in files]==bindings
    assert C.binding(Path(__file__).resolve())==code
    C.write(C.DOC/'CHANGED_B1_CANDIDATES_RECEIPT.json',dict(
        status='COMPLETE_ALL_SEALED_B1_CHANGED_EVENTS',events=len(events),method_id_pairs=len(pairs),
        event_counts=dict(Counter(x['event'] for x in events)),pair_sha256=pair_digest,
        original_outcomes_unchanged=True,source_bindings_after=bindings,code=code,
        output=C.binding(destination),counter_receipt=counter_receipt,
        global_actual_after=PRIOR+sum(calls.values()),reservation_total=reservations,
        baseline_candidate_reuse_new_PnP=0,new_fits=0,new_CNN_forwards=0,RGB_reads=0,
        references_read=0,raw_reference_coordinates_public=False,
        semantics='Initial selector fits and selected final8 pose are distinct; no new alternative final8 pose fitted.',
        active_wall_seconds=time.monotonic()-started))
    print('CHANGED_B1_CANDIDATES_COMPLETE',len(events),'newcalls',sum(calls.values()),'global',PRIOR+sum(calls.values()),flush=True)


if __name__=='__main__':
    main()
