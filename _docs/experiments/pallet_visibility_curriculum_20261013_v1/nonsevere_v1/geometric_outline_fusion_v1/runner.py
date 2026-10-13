"""Fixed expected-outline difficulty and frozen SAM-coordinate fusion controls."""
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
from outline_curriculum import solve_curriculum

HERE=Path(__file__).resolve().parent;COHORT=HERE.parent;SOURCE=COHORT/'SAM_boundary_uncertainty_v1';OLD=COHORT/'OOF_pose_v1'
ROOT=Path('/home/minjae/Documents/github/pallet-pose');sys.path.insert(0,str(ROOT))


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
    blocked=('ORACLE','SCORING','SCORED_ROWS','TARGETS','TRAINING_LABELS','LABEL_AUDIT','RESULTS.json','COORDINATE_AUDIT','/annotations/','test_reused','FINAL')
    def wrap(fn):
        def guarded(path,*args,**kwargs):
            if any(v in str(path) for v in blocked):raise AssertionError('Forbidden inference label/eval read '+str(path))
            return fn(path,*args,**kwargs)
        return guarded
    builtins.open,io.open,Path.open=map(wrap,old)
    try:yield
    finally:builtins.open,io.open,Path.open=old


def infer():
    p=read(HERE/'PROTOCOL.json');assert not(HERE/'PREDICTIONS.jsonl.gz').exists()
    for path,digest in p['source_bindings'].items():assert sha(path)==digest
    assert sha(HERE/'BASE_CURRICULUM.py')==p['baseline_curriculum_sha256']
    frozen=dict(outline_curriculum_sha256=sha(HERE/'outline_curriculum.py'),runner_sha256=sha(__file__),protocol_sha256=sha(HERE/'PROTOCOL.json'))
    write(HERE/'INFERENCE_CODE_BINDING.json',dict(**frozen,before_variant_inference=True))
    output=[];parity=0
    with no_labels():
        for path in (OLD/'ISOLATED_CAL_SCORING.json',COHORT/'CAL_SCORING_METADATA.json'):
            try:read(path)
            except AssertionError:pass
            else:raise AssertionError('canary failed')
        seal=read(SOURCE/'PREDICTION_SEAL.json');assert seal['before_scoring'] and seal['predictions_sha256']==sha(SOURCE/'PREDICTIONS.jsonl.gz')
        source={(r['id'],r['method']):r for r in rows(SOURCE/'PREDICTIONS.jsonl.gz')}
        frames=read(COHORT/'CAL_INPUTS.json')['frames'];assert len(frames)==46
        for frame in frames:
            assert frame['partition']=='CAL';initial=frame['initial_poses']['N3_SUBPIX'];R,t=np.asarray(initial['R_cf']),np.asarray(initial['centroid']);Q=R.T@np.asarray(initial['R_physical'])
            raw=source[frame['id'],'RAW_STAGED'];fused=source[frame['id'],'FUSED_SIGMA2_STAGED']
            assert raw['appearance']==fused['appearance'] and raw['self_hidden']==fused['self_hidden']
            app=raw['appearance'];H=np.asarray(raw['self_hidden'],bool);xyz=cube(initial['cf_extents']);K=np.asarray(frame['K'])
            flags=[]
            for arm in p['arms']:
                reference=raw if arm['coordinates']=='RAW' else fused
                observed=np.asarray(reference['observed_points'],float);q=observed[:8];before=q.copy()
                solution=solve_curriculum(q,xyz,K,R,t,H,app,dict(visible_threshold=.2,staged_visible=arm['staged'],geometric_outline_priority=arm['geometric_outline_priority']))
                assert np.array_equal(q,before) and np.array_equal(observed[:8][H],np.asarray(reference['original_points'])[:8][H])
                assert np.array_equal(observed[8:],np.asarray(reference['original_points'])[8:])
                assert all(not H[k] for s in solution['stages'] for k in s['used_ids'])
                assert solution['visible_ids']==reference['solution']['visible_ids']
                if arm['geometric_outline_priority']:flags.append(solution['stages'][0]['initial_geometric_outline'])
                else:
                    name=arm['coordinates']+('_STAGED' if arm['staged'] else '_TOGETHER')
                    expected=source[frame['id'],name]['solution']
                    assert np.allclose(solution['R'],expected['R'],rtol=0,atol=1e-10)
                    assert np.allclose(solution['t'],expected['t'],rtol=0,atol=1e-10)
                    assert solution['used_ids']==expected['used_ids'] and solution['accepted']==expected['accepted'];parity+=1
                actual=dict(initial,R_cf=solution['R'],R_physical=(np.asarray(solution['R'])@Q).tolist(),centroid=solution['t'])
                if not solution['accepted']:assert np.array_equal(actual['R_cf'],initial['R_cf']) and np.array_equal(actual['centroid'],initial['centroid'])
                output.append(dict(id=frame['id'],session=frame['session'],arm=arm['name'],actual_pose=actual,solution=solution,
                    observed_points=observed.tolist(),raw_points=reference['original_points'],appearance=app,fixed_H=H.tolist(),
                    source_confidence='fresh_FULLCAL46_P_visible_not_joint_P_usable',model_or_axis_family_changed=False,
                    geometric_flags_do_not_change_RGB_states_or_weights=True,GT_input=False))
            assert flags[0]==flags[1]
    assert len(output)==230 and parity==138
    for path,digest in p['source_bindings'].items():assert sha(path)==digest
    assert sha(HERE/'outline_curriculum.py')==frozen['outline_curriculum_sha256'] and sha(__file__)==frozen['runner_sha256']
    with gzip.open(HERE/'PREDICTIONS.jsonl.gz','wt') as stream:
        for row in output:stream.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
    write(HERE/'PREDICTION_SEAL.json',dict(complete=True,before_scoring=True,GT_read_canary='PASS',rows=230,
        sha256=sha(HERE/'PREDICTIONS.jsonl.gz'),source_bindings=frozen,baseline_exact_parity_cases=parity,
        RAW_FUSED_initial_outline_flags_identical=True,H_Q_model_phase_center_fixed=True,
        masks_or_heads_trained_or_inferred=0,severe_rows=0,TEST_rows_read=0,FINAL_rows_read=0))
    print('OUTLINE_FUSION_SEALED',230,flush=True)


def distribution(x):
    x=np.asarray(x,float);return dict(mean=float(x.mean()),median=float(np.median(x)),P90=float(np.quantile(x,.9)))


def score():
    from scripts.research.pallet_visibility_curriculum_20261013_v1.scoring import pose_metric
    assert not(HERE/'RESULTS.json').exists();p=read(HERE/'PROTOCOL.json');seal=read(HERE/'PREDICTION_SEAL.json')
    assert seal['before_scoring'] and seal['sha256']==sha(HERE/'PREDICTIONS.jsonl.gz')
    truth=read(OLD/'ISOLATED_CAL_SCORING.json')['rows'];assert len(truth)==46
    scored=rows(HERE/'PREDICTIONS.jsonl.gz');groups=defaultdict(list)
    for row in scored:
        row['metric']=pose_metric(row['actual_pose'],truth[row['id']]['target']['truth']);groups[row['arm']].append(row)
    summary={};lookup={(r['id'],r['arm']):r for r in scored}
    for arm,value in groups.items():
        summary[arm]=dict(frames=len(value),accepted=sum(r['solution']['accepted'] for r in value),success_count=sum(r['metric']['success_5cm_5deg'] for r in value),
            metrics={k:distribution([r['metric'][k] for r in value]) for k in ('translation_cm','rotation_deg','ADDsym_cm')},
            stage_rejections=dict(Counter(s['reason'] for r in value for s in r['solution']['stages'][1:] if not s['accepted'])))
        print(arm,summary[arm],flush=True)
    sessions=sorted({r['session'] for r in scored});rng=np.random.default_rng(20261013);draw=rng.integers(0,len(sessions),(10000,len(sessions)));paired={}
    contrasts=(('RAW_GEOMETRY_INFO_STAGED','RAW_VISIBLE_TOGETHER'),('FUSED_SIGMA2_GEOMETRY_INFO_STAGED','FUSED_SIGMA2_VISIBLE_TOGETHER'),
        ('FUSED_SIGMA2_GEOMETRY_INFO_STAGED','RAW_GEOMETRY_INFO_STAGED'),('FUSED_SIGMA2_VISIBLE_TOGETHER','RAW_VISIBLE_TOGETHER'),
        ('FUSED_SIGMA2_GEOMETRY_INFO_STAGED','FUSED_SIGMA2_LEGACY_STAGED'))
    for a,b in contrasts:
        value=groups[a];report={}
        for metric in ('translation_cm','rotation_deg','ADDsym_cm','success_5cm_5deg'):
            delta=np.asarray([float(r['metric'][metric])-float(lookup[r['id'],b]['metric'][metric]) for r in value]);sums=np.asarray([sum(d for d,r in zip(delta,value) if r['session']==s) for s in sessions]);counts=np.asarray([sum(r['session']==s for r in value) for s in sessions]);boots=sums[draw].sum(1)/counts[draw].sum(1)
            report[metric]=dict(mean_difference=float(delta.mean()),CI95=np.quantile(boots,[.025,.975]).tolist())
        paired[a+'-'+b]=report
    with gzip.open(HERE/'SCORED_ROWS.jsonl.gz','wt') as stream:
        for row in scored:stream.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
    write(HERE/'RESULTS.json',dict(schema=p['schema'],summary=summary,paired=paired,original_family_only=True,CAL46_only=True,
        severe_rows=0,TEST_rows_read=0,FINAL_rows_read=0,model_or_policy_retuning=0,limits=p['limitations']))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('infer','score'));args=parser.parse_args()
    infer() if args.phase=='infer' else score()
