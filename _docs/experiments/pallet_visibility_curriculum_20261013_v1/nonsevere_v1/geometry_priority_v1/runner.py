"""Four preregistered fixed-family ordering policies on sealed CAL46 evidence."""
import argparse
import builtins
from collections import Counter,defaultdict
from contextlib import contextmanager
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve().parent;COHORT=HERE.parent;OLD=COHORT/'OOF_pose_v1'
ROOT=Path('/home/minjae/Documents/github/pallet-pose');sys.path.insert(0,str(ROOT))
from study_curriculum import solve_curriculum


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rows(path):return [json.loads(l) for l in gzip.open(path,'rt') if l.strip()]
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def cube(d):
    a,b,c=np.asarray(d)/2
    return np.array([[-a,-b,-c],[a,-b,-c],[a,b,-c],[-a,b,-c],[-a,-b,c],[a,-b,c],[a,b,c],[-a,b,c]])


@contextmanager
def no_labels():
    old=builtins.open,io.open,Path.open
    blocked=('ORACLE','SCORING','SCORED_ROWS','TARGETS','TRAINING_LABELS','LABEL_AUDIT','RESULTS.json','/annotations/')
    def wrap(fn):
        def guarded(path,*args,**kwargs):
            if any(v in str(path) for v in blocked):raise AssertionError('Labelread in variantinference '+str(path))
            return fn(path,*args,**kwargs)
        return guarded
    builtins.open,io.open,Path.open=map(wrap,old)
    try:yield
    finally:builtins.open,io.open,Path.open=old


def infer():
    p=read(HERE/'PROTOCOL.json');assert not(HERE/'PREDICTIONS.jsonl.gz').exists()
    for path,digest in p['data_bindings'].items():assert sha(path)==digest
    assert sha(HERE/'BASE_CURRICULUM.py')==p['base_curriculum_sha256']
    frozen=dict(study_curriculum_sha256=sha(HERE/'study_curriculum.py'),runner_sha256=sha(__file__),protocol_sha256=sha(HERE/'PROTOCOL.json'))
    write(HERE/'INFERENCE_CODE_BINDING.json',dict(**frozen,before_variant_inference=True))
    output=[];parity=0
    with no_labels():
        for path in (OLD/'ISOLATED_CAL_SCORING.json',COHORT/'CAL_SCORING_METADATA.json'):
            try:read(path)
            except AssertionError:pass
            else:raise AssertionError('canary failed')
        frames=read(COHORT/'CAL_INPUTS.json')['frames'];base={r['id']:r for r in rows(OLD/'PREDICTIONS.jsonl.gz') if r['method']=='JOINT_OOF_STAGED'}
        assert len(frames)==len(base)==46
        for frame in frames:
            assert frame['partition']=='CAL';row=base[frame['id']];initial=frame['initial_poses']['N3_SUBPIX']
            q=np.asarray(row['corrected_measurements'],float)[:8];H=np.asarray(row['self_hidden'],bool);app=row['appearance']
            R,t=np.asarray(initial['R_cf']),np.asarray(initial['centroid']);Q=R.T@np.asarray(initial['R_physical']);xyz=cube(initial['cf_extents']);K=np.asarray(frame['K'])
            expected=row['branch_result']['candidates'][0]['curriculum'];original=solve_curriculum(q,xyz,K,R,t,H,app,
                dict(visible_threshold=.45,easy_visibility=.45,analytic_jacobian=False,research_variant='BASELINE'))
            assert np.allclose(original['R'],expected['R'],rtol=0,atol=1e-10)
            assert np.allclose(original['t'],expected['t'],rtol=0,atol=1e-10)
            assert original['used_ids']==expected['used_ids'] and original['accepted']==expected['accepted'];parity+=1
            for variant in p['variants']:
                before=q.copy();solution=solve_curriculum(q,xyz,K,R,t,H,app,
                    dict(visible_threshold=.45,easy_visibility=.45,analytic_jacobian=False,research_variant=variant))
                assert np.array_equal(q,before)
                assert all(not H[k] for s in solution['stages'] for k in s['used_ids'])
                assert solution['visible_ids']==expected['visible_ids']
                actual=dict(initial,R_cf=solution['R'],R_physical=(np.asarray(solution['R'])@Q).tolist(),centroid=solution['t'])
                if not solution['accepted']:
                    assert np.array_equal(actual['R_cf'],initial['R_cf']) and np.array_equal(actual['centroid'],initial['centroid'])
                for stage in solution['stages']:
                    if stage.get('diagnostic_protected_history_reset') and stage['accepted']:
                        assert stage['consolidated_all_visible_score']<=stage['staged_all_visible_score']
                output.append(dict(id=frame['id'],session=frame['session'],variant=variant,actual_pose=actual,
                    solution=solution,initial_pose=initial,corrected_measurements=q.tolist(),fixed_H=H.tolist(),
                    original_family=True,coordinates_and_eligibility_unchanged=True,GT_input=False))
    assert len(output)==184 and parity==46
    assert sha(HERE/'study_curriculum.py')==frozen['study_curriculum_sha256'] and sha(__file__)==frozen['runner_sha256']
    for path,digest in p['data_bindings'].items():assert sha(path)==digest
    with gzip.open(HERE/'PREDICTIONS.jsonl.gz','wt') as stream:
        for row in output:stream.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
    write(HERE/'PREDICTION_SEAL.json',dict(complete=True,before_scoring=True,GT_read_canary='PASS',rows=184,
        sha256=sha(HERE/'PREDICTIONS.jsonl.gz'),base_solver_parity_frames=parity,source_bindings=frozen,
        maincode_edits=0,model_retuning=0,TEST_rows_read=0,FINAL_rows_read=0,severe_rows=0))
    print('PRIORITY_SEALED',184,flush=True)


def distribution(x):
    x=np.asarray(x,float);return dict(mean=float(x.mean()),median=float(np.median(x)),P90=float(np.quantile(x,.9)))


def score():
    from scripts.research.pallet_visibility_curriculum_20261013_v1.scoring import pose_metric
    assert not(HERE/'RESULTS.json').exists();p=read(HERE/'PROTOCOL.json');seal=read(HERE/'PREDICTION_SEAL.json')
    assert seal['before_scoring'] and seal['sha256']==sha(HERE/'PREDICTIONS.jsonl.gz')
    truth=read(OLD/'ISOLATED_CAL_SCORING.json')['rows'];assert len(truth)==46
    baseline={(r['id'],r['method']):r for r in rows(OLD/'SCORED_ROWS.jsonl.gz')}
    scored=rows(HERE/'PREDICTIONS.jsonl.gz');groups=defaultdict(list)
    for row in scored:
        row['metric']=pose_metric(row['actual_pose'],truth[row['id']]['target']['truth'])
        row['staged_metric']=baseline[row['id'],'JOINT_OOF_STAGED']['original_family_metric']
        row['together_metric']=baseline[row['id'],'JOINT_OOF_VISIBLE_TOGETHER']['original_family_metric'];groups[row['variant']].append(row)
    sessions=sorted({r['session'] for r in scored});rng=np.random.default_rng(20261013);draw=rng.integers(0,len(sessions),(10000,len(sessions)))
    summary={}
    for variant,value in groups.items():
        report=dict(frames=len(value),accepted=sum(r['solution']['accepted'] for r in value),
            success_count=sum(r['metric']['success_5cm_5deg'] for r in value),
            metrics={k:distribution([r['metric'][k] for r in value]) for k in ('translation_cm','rotation_deg','ADDsym_cm')},
            stage_rejections=dict(Counter(s['reason'] for r in value for s in r['solution']['stages'][1:] if not s['accepted'])),paired={})
        for ref in ('staged_metric','together_metric'):
            report['paired'][ref]={}
            for k in ('translation_cm','rotation_deg','ADDsym_cm','success_5cm_5deg'):
                delta=np.asarray([float(r['metric'][k])-float(r[ref][k]) for r in value])
                sums=np.asarray([sum(d for d,r in zip(delta,value) if r['session']==s) for s in sessions]);counts=np.asarray([sum(r['session']==s for r in value) for s in sessions]);boots=sums[draw].sum(1)/counts[draw].sum(1)
                report['paired'][ref][k]=dict(mean_difference=float(delta.mean()),CI95=np.quantile(boots,[.025,.975]).tolist())
        summary[variant]=report;print(variant,report,flush=True)
    with gzip.open(HERE/'SCORED_ROWS.jsonl.gz','wt') as stream:
        for row in scored:stream.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
    write(HERE/'RESULTS.json',dict(schema=p['schema'],summary=summary,original_family_only=True,CAL46_only=True,
        maincode_edits=0,head_or_model_retuning=0,TEST_rows_read=0,FINAL_rows_read=0,severe_rows=0,limits=p['limits']))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('infer','score'));args=parser.parse_args()
    infer() if args.phase=='infer' else score()
