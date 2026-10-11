"""Independent saved-coordinate/seal/count check; no models, F or training."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
DOC=Path(os.environ.get('PALLET_PAPER_OUTPUT',ROOT/'_docs/experiments/pallet_paper_tables_245_20261011')).resolve()
SOURCE=Path(os.environ.get('PALLET_SOURCE_ROOT',ROOT)).resolve()

def read(name):return json.loads((DOC/name).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def rows(path):
    with gzip.open(path,'rt') as f:return [json.loads(line) for line in f]

def main():
    checks=[]
    for name in ('STAGE0_COORDINATES','STAGE0_POSES','POSEFIX_COORDINATES','POSEFIX_POSES','RECT_COORDINATES','RECT_POSES'):
        seal=read(name+'_SEAL.json');assert sha(DOC/seal['path'])==seal['sha256']
        values=rows(DOC/seal['path']);assert len(values)==seal['rows']
        assert seal['correction_reference_inputs'] is False and seal['F_reference_inputs'] is False
        checks.append(dict(check='stage_seal',stage=name,rows=len(values),sha256=seal['sha256']))
    rc=rows(DOC/'RECT_COORDINATES.jsonl.gz');rp=rows(DOC/'RECT_POSES.jsonl.gz');rr=rows(DOC/'RECT_ROWS.jsonl.gz')
    assert len(rc)==len(rp)==len(rr)==21054
    key=lambda r:(r['backbone'],r['method'],r['seed'],r['id'])
    assert len({key(r) for r in rc})==21054
    assert {key(r) for r in rc}=={key(r) for r in rp}=={key(r) for r in rr}
    coords={key(r):r for r in rc};poses={key(r):r for r in rp};corner_damage={}
    for r in rr:
        k=key(r);a=coords[k];p=poses[k]
        assert np.array_equal(np.asarray(a['qFinal'],float),np.asarray(r['qFinal'],float),equal_nan=True)
        assert r['actual_pose']==p['actual_pose']
        assert r['inference_reference_inputs'] is False and r['reference_inputs_in_F'] is False
        if 'correction' in a:
            q=np.asarray(a['qFinal'],float);q0=np.asarray(a['q0'],float);support=np.asarray(a['prediction_support'],bool)
            assert np.array_equal(q[8],q0[8],equal_nan=True)
            assert np.array_equal(q[~support],q0[~support],equal_nan=True)
            valid=support[:8]&np.isfinite(q0[:8]).all(1)&~(q0[:8]==-1).all(1)
            cap=.01*np.hypot(*a['raw_hw'])
            assert np.max(np.linalg.norm(q[:8][valid]-q0[:8][valid],axis=1),initial=0)<=cap+1e-10
    assert read('STAGE0_F_PARITY.json')['status']=='PASS'
    assert read('POSEFIX_319_PARITY.json')['status']=='PASS'
    execution=read('EVALUATION_EXECUTION.json')
    assert execution['actual_F_calls_total']==21054 and execution['network_forwards']==0 and execution['new_training_updates']==0
    lat=read('LATENCY.json');assert lat['status']=='COMPLETE'
    assert lat['protocol']['actual_pipeline_calls']==900 and lat['protocol']['explicit_detector_calls']==900 and lat['protocol']['detector_forwards']==901 and lat['protocol']['actual_F_calls']==900
    raw=[json.loads(line) for line in (DOC/lat['raw_receipts_path']).read_text().splitlines()]
    assert len(raw)==900 and sha(DOC/lat['raw_receipts_path'])==lat['raw_receipts_sha256']
    for arm,value in lat['arms'].items():
        calls=[r for r in raw if r['arm']==arm];assert len(calls)==150
        measured=np.asarray([r['wall_ms'] for r in calls if r['phase']=='measured'])
        assert len(measured)==130 and sum(r['phase']=='warmup' for r in calls)==20
        assert abs(float(measured.mean())-value['mean_ms'])<1e-12
        assert abs(float(measured.std(ddof=1))-value['sd_ms'])<1e-12
        assert all(r['explicit_detector_calls']==1 and r['actual_F_calls']==1 for r in calls)
        assert all(r['detector_forwards']==1 for r in calls if r['phase']=='measured')
    assert sum(r['internal_initialization_forwards'] for r in raw)==1
    verified=0
    for binding in read('INPUT_AUDIT.json')['bindings']:
        p=(ROOT if binding['owner']=='published' else SOURCE)/binding['path']
        assert sha(p)==binding['sha256'],binding['path'];verified+=1
    for filename,field in [('ROOT_EVALUATION_REVIEW_LOCK.json','source'),('STATS_SOURCE_LOCK.json','bindings')]:
        for b in read(filename)[field]:assert sha(ROOT/b['path'])==b['sha256']
    stats=read('STATISTICS_VERIFICATION.json');assert stats['status']=='PASS'
    result=dict(status='PASS',checks=checks,unique_actual_accuracy_F_calls=21054,
        deployment_actual_F_calls=900,diagnostic_failed_warmup_F_calls=1,all_actual_F_calls=21955,accuracy_model_forwards=0,
        latency_detector_forwards=901,diagnostic_detector_forwards=2,training_updates=0,source_and_input_bindings_unchanged=verified,
        generated_coordinate_center_support_and_one_final_cap_checks='PASS',
        saved_coordinate_pose_scoring_isolation_checks='PASS',six_arm_latency_moments_checks='PASS',
        independent_statistics_receipt_sha256=sha(DOC/'STATISTICS_VERIFICATION.json'),
        source_sha256=sha(Path(__file__)))
    with (DOC/'VERIFICATION.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print('INDEPENDENT_EXECUTION_SEAL_AND_SOURCE_VERIFICATION_PASS',verified,flush=True)

if __name__=='__main__':main()
