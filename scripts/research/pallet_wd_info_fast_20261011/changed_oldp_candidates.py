"""GT-free unchanged-F replay for twelve OLD_P changed-case baselines."""
from contextlib import contextmanager
import csv
import fcntl
import gzip
import hashlib
import json
from pathlib import Path
import re
import time

import cv2
import numpy as np

from . import common as C
from . import geometry_analysis as A

PRIOR, LIMIT, JOB = 49399, 72, 'OLD_P_CHANGED_CANDIDATES'
DECODER = json.JSONDecoder()


def field(line, name):
    match=re.search(r'"'+name+r'"\s*:\s*',line)
    assert match, ('Missing actual field',name)
    return DECODER.raw_decode(line[match.end():])[0]


@contextmanager
def count_calls():
    with (C.PRIVATE/'PNP_RECEIPTS.jsonl').open('a+') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX);stream.seek(0)
        before=[json.loads(s) for s in stream if s.strip()]
        assert sum(r['total'] for r in before)==PRIOR and not any(r['job']==JOB for r in before)
        originals,calls={},{};started=time.monotonic();receipt={}
        for name in ('solvePnP','solvePnPGeneric','solvePnPRansac','solvePnPRefineLM','solvePnPRefineVVS'):
            if not hasattr(cv2,name):continue
            originals[name]=getattr(cv2,name);calls[name]=0
            def counted(*args,_name=name,_fn=originals[name],**kwargs):
                assert sum(calls.values())<LIMIT and PRIOR+sum(calls.values())<50000
                calls[_name]+=1;return _fn(*args,**kwargs)
            setattr(cv2,name,counted)
        try:yield calls,receipt
        finally:
            for name,fn in originals.items():setattr(cv2,name,fn)
            receipt.update(job=JOB,limit=LIMIT,calls=calls,total=sum(calls.values()),
                prior_global_actual=PRIOR,global_actual_after=PRIOR+sum(calls.values()),wall_seconds=time.monotonic()-started)
            stream.seek(0,2);stream.write(json.dumps(receipt)+'\n');stream.flush();fcntl.flock(stream,fcntl.LOCK_UN)


def main():
    started=time.monotonic();cv2.setNumThreads(1);plan=C.lock_ready()
    paths=[C.DOC/'COMPARISON_TRANSITIONS_COMPLETE.csv', C.DOC/'PLAN_LOCK.json',
        C.SOURCE/'data/pallet/results/pallet_dim_conditioned_p_v1/predictions/REAL_DEV/OLD_P_seed1.json',
        C.PUBLIC/'_docs/experiments/pallet_paper_tables_245_20261011/RECT_ROWS.jsonl.gz',
        C.PUBLIC/A.WD_REL/'STAGE1_ROWS_REAL.jsonl.gz',
        C.SOURCE/'scripts/research/pallet_dim_conditioned_p_v1/pose.py',
        C.SOURCE/'scripts/paper/pose_metric_closure_v1/run_pose_evaluation.py',
        C.SOURCE/'challenge/evaluation_v2/pnp_selector.py']
    bindings=[C.binding(p) for p in paths]
    with paths[0].open() as stream:
        reader=csv.DictReader(stream);columns=reader.fieldnames
        changed=[r for r in reader if r['source']=='D' and r['before']=='OLD_P' and
            (r['selection_changed']=='1' or any(r[k] in ('RECOVERY','DAMAGE')
                for k in ('axis_transition','strict_transition','availability_transition')))]
    ids=sorted({r['id'] for r in changed},key=lambda s:(hashlib.sha256(s.encode()).hexdigest(),s))
    assert len(changed)==15 and len(ids)==12 and LIMIT==len(ids)*6
    assert set(ids)<=set(C.read(C.DOC/'INPUT_CONTRACT.json')['allowed_ids'])
    reservation=sum(plan['pnp_reservations'].values())+1038+84+LIMIT
    assert reservation==49910 and reservation<=50000
    code=C.binding(Path(__file__).resolve())
    C.write(C.DOC/'OLD_P_CHANGED_CANDIDATES_LOCK.json',dict(
        status='LOCKED_BEFORE_GT_FREE_BASELINE_F_REPLAY',code=code,inputs=bindings,
        hash_sorted_ids=ids,sample_sha256=A.digest(ids),comparison_rows=15,
        fixed_pairs=[dict(after=r['after'],id=r['id']) for r in changed],
        prior_global_actual=PRIOR,new_call_limit=LIMIT,reservation_total=reservation,
        replay='Original F.infer with two initial selector fits and selected final8 pose',
        parity_gate='Match cached P actual_pose R_cf, centroid and selected_hypothesis without GT',
        new_fits=0,new_CNN_forwards=0,raw_RGB_reads=0,raw_reference_poses_annotations_read=0))
    # Decode twelve prediction-only records; protected records remain text.
    predictions={};buffer=[];gt_input_false=False
    with paths[2].open() as stream:
        for line in stream:
            if re.fullmatch(r'\s*"GT_input": false,?\s*',line):gt_input_false=True
            if line.rstrip()=='    {':buffer=[line]
            elif buffer:
                buffer.append(line)
                if line.rstrip() in ('    },','    }'):
                    text=''.join(buffer).rstrip().removesuffix(',')
                    if field(text,'id') in set(ids):
                        row=json.loads(text);selected=row['selected_index'];assert selected is not None
                        predictions[row['id']]=np.asarray(row['candidates'][selected]['keypoints_xy'],float)
                    buffer=[]
    assert gt_input_false and set(predictions)==set(ids)
    cached={};metadata={};point_parity_max=0.
    # Extract only prediction fields; do not decode published reference arrays.
    with gzip.open(paths[3],'rt') as stream:
        for line in stream:
            if field(line,'id') not in set(ids):continue
            if field(line,'backbone')!='YOLO' or field(line,'seed')!=1 or field(line,'method')!='P':continue
            fid=field(line,'id');cached[fid]=field(line,'actual_pose')
            difference=float(np.max(np.abs(predictions[fid]-np.asarray(field(line,'qFinal'),float))))
            assert difference==0,'Historical OLD_P predicted coordinates differ'
            point_parity_max=max(point_parity_max,difference)
    with gzip.open(paths[4],'rt') as stream:
        for line in stream:
            if field(line,'id') in set(ids) and field(line,'seed')==1 and field(line,'method')=='BASE':
                metadata[field(line,'id')]=field(line,'fixed_metadata')
    assert set(cached)==set(metadata)==set(ids)
    pose=A.pose_api(C.SOURCE);values={};checks=[]
    with count_calls() as (calls,counter_receipt):
        for fid in ids:
            q=predictions[fid];meta=metadata[fid];assert q.shape==(9,2) and np.isfinite(q).all()
            trace=[];original=pose.select_pnp_hypotheses
            def capture(*args,**kwargs):
                result=original(*args,**kwargs);trace.append(result);return result
            pose.select_pnp_hypotheses=capture
            try:actual=pose.infer(q,np.asarray(meta['K'],float),np.asarray(meta['dimensions_pnp_WH_D_m'],float),False)
            finally:pose.select_pnp_hypotheses=original
            assert len(trace)==1 and actual['available']==cached[fid]['available']
            assert actual.get('selected_hypothesis')==cached[fid].get('selected_hypothesis')
            differences={key:float(np.max(np.abs(np.asarray(actual[key])-np.asarray(cached[fid][key]))))
                         for key in ('R_cf','centroid')}
            assert max(differences.values())<=1e-10,('Cached OLD_P production pose parity failed',fid,differences)
            checks.append(dict(id=fid,**differences,selected_hypothesis_equal=True))
            candidate={h.name:dict(success=h.success,score=h.score,score_components=dict(h.score_components),
                R_cf=h.rotation_camera_facing,centroid_m=h.translation_camera_facing,
                cf_extents_m=h.camera_facing_dimensions.as_dict(),failure=h.failure_reason,
                semantics='Original F initial selector fit; selected final8 pose stored separately') for h in trace[0].hypotheses}
            assert set(candidate)==set(A.NAMES)
            values[fid]=dict(candidates=candidate,selected_final8_pose=actual)
    assert sum(calls.values())==LIMIT
    destination=C.DOC/'OLD_P_CHANGED_CANDIDATES_COMPLETE.csv'
    with destination.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns+['old_selected_final8_pose_json','old_candidate_semantics'])
        writer.writeheader()
        for row in changed:
            r=dict(row);item=values[r['id']]
            r['old_candidates_json']=json.dumps(C.clean(item['candidates']),sort_keys=True,separators=(',',':'),allow_nan=False)
            r['old_selected_final8_pose_json']=json.dumps(C.clean(item['selected_final8_pose']),sort_keys=True,separators=(',',':'),allow_nan=False)
            r['old_candidate_semantics']='GT-free replay original F initial selector fits; selected final8 separately, equality checked against historical P'
            assert 'NA_NOT_RETAINED' not in r['old_candidates_json']+r['new_candidates_json']
            writer.writerow(r)
    assert [C.binding(p) for p in paths]==bindings and C.binding(Path(__file__).resolve())==code
    C.write(C.DOC/'OLD_P_CHANGED_CANDIDATES_RECEIPT.json',dict(
        status='COMPLETE_ALL_15_CHANGED_STAGE1_OLD_P_COMPARISONS',code=code,ids=ids,
        comparison_rows=15,unique_baseline_frames=12,cached_prediction_max_delta_px=point_parity_max,
        cached_pose_parity=checks,output=C.binding(destination),counter_receipt=counter_receipt,
        global_actual_after=PRIOR+sum(calls.values()),reservation_total=reservation,
        original_results_transition_NA_and_sources_unchanged=True,source_bindings_after=bindings,
        new_fits=0,new_CNN_forwards=0,raw_RGB_reads=0,raw_reference_poses_annotations_read=0,
        semantics='New file fills all changed Stage1 OLD_P baseline candidate values. Unchanged historical NA rows preserved.',
        active_wall_seconds=time.monotonic()-started))
    print('OLD_P_CHANGED_CANDIDATES_COMPLETE',len(changed),'frames',len(ids),'calls',sum(calls.values()),'global',PRIOR+sum(calls.values()),flush=True)


if __name__=='__main__':main()
