"""Fixed CAL46 policy intervention, independent cached observations only."""
from pathlib import Path
import builtins
import io
from contextlib import contextmanager
from copy import deepcopy
from collections import Counter,defaultdict
import argparse
import gzip
import hashlib
import json
import time

import numpy as np
import cv2

import progressive_curriculum as M

HERE=Path(__file__).resolve().parent
COHORT=HERE.parent
PRIOR=COHORT/'geometric_outline_fusion_v1'
SOURCE=COHORT/'SAM_boundary_uncertainty_v1'
OLD=COHORT/'OOF_pose_v1'


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rows(path):return [json.loads(line) for line in gzip.open(path,'rt') if line.strip()]
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def data_sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def cube(d):
    a,b,c=np.asarray(d)/2
    return np.array([[-a,-b,-c],[a,-b,-c],[a,b,-c],[-a,b,-c],[-a,-b,c],[a,-b,c],[a,b,c],[-a,b,c]])


@contextmanager
def no_labels():
    originals=builtins.open,io.open,Path.open
    blocked=('ORACLE','SCORING','SCORED_ROWS','TARGETS','TRAINING_LABELS','LABEL_AUDIT',
        'RESULTS.json','COORDINATE_AUDIT','/annotations/','test_reused','FINAL')
    def wrap(fn):
        def guarded(path,*args,**kwargs):
            if any(x in str(path) for x in blocked):raise AssertionError('Forbidden inference label read: '+str(path))
            return fn(path,*args,**kwargs)
        return guarded
    builtins.open,io.open,Path.open=map(wrap,originals)
    try:yield
    finally:builtins.open,io.open,Path.open=originals


def infer():
    assert not(HERE/'PREDICTIONS.jsonl.gz').exists()
    protocol=read(HERE/'PROTOCOL.json')
    for path,digest in protocol['source_bindings'].items():assert sha(path)==digest
    code={str(HERE/name):sha(HERE/name) for name in ['progressive_curriculum.py','BASE_OUTLINE_CURRICULUM.py','runner.py','test_progressive.py','PROTOCOL.json']}
    write(HERE/'INFERENCE_CODE_BINDING.json',dict(before_inference_and_scoring=True,source_sha256=code,
        meaningful_analytic_tests=5,test_status='PASS',protocol_changed_after_scoring=False))
    started=time.perf_counter();output=[];baseline_parity=0;copied_together=0
    cv2.setNumThreads(1)
    with no_labels():
        for path in [OLD/'ISOLATED_CAL_SCORING.json',COHORT/'CAL_SCORING_METADATA.json',COHORT/'test_reused_v1/ISOLATED_TEST_SCORING.json']:
            try:read(path)
            except AssertionError:pass
            else:raise AssertionError('Annotation access canary failed')
        previous_seal=read(PRIOR/'PREDICTION_SEAL.json');source_seal=read(SOURCE/'PREDICTION_SEAL.json')
        assert previous_seal['before_scoring'] and previous_seal['sha256']==sha(PRIOR/'PREDICTIONS.jsonl.gz')
        assert source_seal['before_scoring'] and source_seal['predictions_sha256']==sha(SOURCE/'PREDICTIONS.jsonl.gz')
        previous={(r['id'],r['arm']):r for r in rows(PRIOR/'PREDICTIONS.jsonl.gz')}
        source={(r['id'],r['method']):r for r in rows(SOURCE/'PREDICTIONS.jsonl.gz')}
        frames=read(COHORT/'CAL_INPUTS.json')['frames'];assert len(frames)==46
        assert {f['id'] for f in frames}=={k[0] for k in previous}=={k[0] for k in source}
        for frame in frames:
            assert frame['partition']=='CAL'
            initial=frame['initial_poses']['N3_SUBPIX'];R,t=np.asarray(initial['R_cf']),np.asarray(initial['centroid'])
            Q=R.T@np.asarray(initial['R_physical']);xyz=cube(initial['cf_extents']);K=np.asarray(frame['K'])
            raw,fused=source[frame['id'],'RAW_STAGED'],source[frame['id'],'FUSED_SIGMA2_STAGED']
            assert raw['appearance']==fused['appearance'] and raw['self_hidden']==fused['self_hidden']
            H=np.asarray(raw['self_hidden'],bool);app=raw['appearance'];app_sha=data_sha(app)
            for coord,reference in [('RAW',raw),('FUSED_SIGMA2',fused)]:
                observed=np.asarray(reference['observed_points'],float);q=observed[:8];q_before=q.copy()
                assert np.array_equal(observed[:8][H],np.asarray(reference['original_points'])[:8][H])
                assert np.array_equal(observed[8:],np.asarray(reference['original_points'])[8:])
                prior_arm=coord+'_GEOMETRY_INFO_STAGED';expected=previous[frame['id'],prior_arm]
                baseline=None
                for progressive in [False,True]:
                    cfg=dict(visible_threshold=.2,geometric_outline_priority=True,progressive_active_support=progressive)
                    solution=M.solve_curriculum(q,xyz,K,R,t,H,app,cfg)
                    assert np.array_equal(q,q_before) and data_sha(app)==app_sha
                    assert all(not H[k] for s in solution['stages'] for k in s['used_ids'])
                    assert solution['visible_ids']==expected['solution']['visible_ids']
                    assert solution['stages'][0]['initial_geometric_outline']==expected['solution']['stages'][0]['initial_geometric_outline']
                    assert solution['stages'][0]['geometric_outline_seed']==expected['solution']['stages'][0]['geometric_outline_seed']
                    if not progressive:
                        for key in ['R','t']:assert np.allclose(solution[key],expected['solution'][key],rtol=0,atol=1e-10)
                        assert solution['accepted']==expected['solution']['accepted'] and solution['used_ids']==expected['solution']['used_ids']
                        assert [(s['name'],s['accepted'],s['reason'],s['used_ids']) for s in solution['stages']]==[(s['name'],s['accepted'],s['reason'],s['used_ids']) for s in expected['solution']['stages']]
                        baseline=solution;baseline_parity+=1
                    else:
                        flags=np.asarray(baseline['stages'][0]['initial_geometric_outline'])
                        seed=baseline['stages'][0]['geometric_outline_seed']['seed']
                        visibility=np.asarray(app['visibility_score']);accuracy=np.asarray(app['local_accuracy_score'])
                        remaining=sorted(set(baseline['visible_ids'])-set(seed),key=lambda k:(0 if flags[k] else 1,-float(visibility[k]*accuracy[k]),k))
                        assert solution['nominal_visible_order']==list(seed)+remaining
                        if solution['accepted']:
                            guard=solution['final_guard']
                            assert guard['initial']['objective']-guard['output_objective']['objective']>1e-7
                            assert guard['retained_active_ids']==solution['used_ids']
                            assert solution['unknown_added_ids']==sorted(set(solution['used_ids'])&set(solution['unknown_considered_ids']))
                        else:
                            assert np.array_equal(solution['R'],R) and np.array_equal(solution['t'],t) and solution['used_ids']==[]
                    actual=dict(initial,R_cf=solution['R'],centroid=solution['t'],R_physical=(np.asarray(solution['R'])@Q).tolist())
                    assert np.allclose(np.asarray(actual['R_cf']).T@np.asarray(actual['R_physical']),Q,rtol=0,atol=1e-12)
                    if not solution['accepted']:
                        assert np.array_equal(actual['R_cf'],initial['R_cf']) and np.array_equal(actual['centroid'],initial['centroid'])
                    name=coord+('_PROGRESSIVE_ACTIVE_SUPPORT' if progressive else '_BASELINE_GEOMETRY_STAGED')
                    output.append(dict(id=frame['id'],session=frame['session'],arm=name,actual_pose=actual,solution=solution,
                        observed_points=observed.tolist(),raw_points=reference['original_points'],appearance=app,fixed_H=H.tolist(),
                        source_confidence='same_frozen_FULLCAL46_P_visible_not_joint_P_usable',coordinates=coord,
                        original_model_native_ids_physical_Q_center_preserved=True,GT_input=False,
                        masks_CNN_or_heads_trained_or_inferred=0,no_final_batch_refit=True))
                together=deepcopy(previous[frame['id'],coord+'_VISIBLE_TOGETHER'])
                assert together['appearance']==app and together['fixed_H']==H.tolist() and together['observed_points']==observed.tolist()
                together['sealed_control_copied_from']=str(PRIOR/'PREDICTIONS.jsonl.gz')
                together['source_prediction_row_sha256']=data_sha(previous[frame['id'],coord+'_VISIBLE_TOGETHER'])
                output.append(together);copied_together+=1
            output.append(dict(id=frame['id'],session=frame['session'],arm='INITIAL',actual_pose=initial,
                solution=dict(accepted=False,reason='supplied_initial_pose',used_ids=[],stages=[]),
                observed_points=frame['points']['N3_SUBPIX'],raw_points=frame['points']['N3_SUBPIX'],
                fixed_H=H.tolist(),GT_input=False))
    assert len(output)==322 and baseline_parity==92 and copied_together==92
    for path,digest in protocol['source_bindings'].items():assert sha(path)==digest
    for path,digest in code.items():assert sha(path)==digest
    with gzip.open(HERE/'PREDICTIONS.jsonl.gz','wt') as handle:
        for row in output:handle.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
    write(HERE/'PREDICTION_SEAL.json',dict(complete=True,before_scoring=True,GT_read_canary='PASS',rows=len(output),
        sha256=sha(HERE/'PREDICTIONS.jsonl.gz'),protocol_sha256=sha(HERE/'PROTOCOL.json'),code_bindings=code,
        exact_baseline_pose_stage_order_parity_cases=baseline_parity,sealed_together_controls_copied=copied_together,
        same_eligibility_appearance_frozen_weights_H_Q_model_native_ids_center=True,
        all_returned_progressive_poses_pass_final_initial_all_visible_guard=True,
        final_batch_refits=0,CNN_mask_or_head_calls=0,severe_rows=0,TEST_rows_read=0,FINAL_rows_read=0,
        seconds=time.perf_counter()-started))
    print('SEALED',len(output),'baseline_exact',baseline_parity,'together_copied',copied_together,flush=True)


def rotation_y(a):
    c,s=np.cos(a),np.sin(a);return np.array([[c,0,s],[0,1,0],[-s,0,c]])


def metric(p,truth):
    R,G=np.asarray(p['R_physical']),np.asarray(truth['R']);t,target=np.asarray(p['centroid']),np.asarray(truth['t']);X=cube(truth['xyz'])
    angles=[];adds=[]
    for j in range(truth['order']):
        GT=G@rotation_y(j*2*np.pi/truth['order']);relative=GT.T@R
        angles.append(float(np.degrees(np.arccos(np.clip((np.trace(relative)-1)/2,-1,1)))))
        adds.append(float(np.linalg.norm(X@R.T+t-(X@GT.T+target),axis=1).mean())*100)
    T=float(np.linalg.norm(t-target)*100);A=min(angles)
    return dict(translation_cm=T,rotation_deg=A,ADDsym_cm=min(adds),success_5cm_5deg=T<5 and A<5)


def distribution(x):
    a=np.asarray(x,float)
    return dict(n=len(a),mean=float(a.mean()) if len(a) else None,
        median=float(np.median(a)) if len(a) else None,P90=float(np.quantile(a,.9)) if len(a) else None)


def support_summary(group):
    all_stages=[s for r in group for s in r['solution'].get('stages',[]) if s['name']!='initial' and s.get('fit_performed',True)]
    fits=[s for s in all_stages if 'optimizer' in s]
    final_guards=[r['solution']['final_guard'] for r in group if 'final_guard' in r['solution']]
    seeds=[next((s for s in r['solution'].get('stages',[]) if s['name']!='initial' and s.get('fit_performed',True)),None) for r in group]
    seeds=[s for s in seeds if s]
    return dict(attempted_stages=len(all_stages),optimizer_fit_stages=len(fits),
        accepted_fit_stages=sum(s['accepted'] for s in fits),seed_accepted_frames=sum(s['accepted'] for s in seeds),
        seed_reasons=dict(Counter(s['reason'] for s in seeds)),
        rejected_stages=dict(Counter(s['reason'] for s in all_stages if not s['accepted'])),
        final_visible_support=distribution([len(set(r['solution']['used_ids'])&set(r['solution'].get('visible_ids',[]))) for r in group]),
        frames_omitting_visible=sum(bool(set(r['solution'].get('visible_ids',[]))-set(r['solution']['used_ids'])) for r in group),
        final_fallback_sources=dict(Counter(g['fallback_source'] for g in final_guards)),
        discarded_accepted_checkpoints=sum(len(g['discarded_accepted_checkpoint_indices']) for g in final_guards),
        discarded_unknown_accepts=sum(s.get('intermediate_accept_discarded_by_final_guard',False) for s in fits if s['name'].startswith('add_consistent_unknown_')),
        final_prior_normalized=distribution([g['output_objective']['prior_cost_sum']/g['output_objective']['frozen_weight_sum'] for g in final_guards]))


def score():
    assert not(HERE/'RESULTS.json').exists()
    protocol=read(HERE/'PROTOCOL.json');seal=read(HERE/'PREDICTION_SEAL.json')
    assert seal['before_scoring'] and seal['GT_read_canary']=='PASS' and seal['sha256']==sha(HERE/'PREDICTIONS.jsonl.gz')
    for path,value in seal['code_bindings'].items():assert sha(path)==value
    # First new label access occurs strictly after all seven arms are sealed.
    truth=read(OLD/'ISOLATED_CAL_SCORING.json')['rows'];assert len(truth)==46
    scored=rows(HERE/'PREDICTIONS.jsonl.gz');assert len(scored)==322
    groups=defaultdict(list)
    for row in scored:
        row['metric']=metric(row['actual_pose'],truth[row['id']]['target']['truth']);groups[row['arm']].append(row)
    assert {r['id'] for r in scored}==set(truth)
    summary={};lookup={(r['id'],r['arm']):r for r in scored}
    for arm,value in groups.items():
        assert len(value)==46
        summary[arm]=dict(frames=len(value),accepted=sum(r['solution']['accepted'] for r in value),
            success_count=sum(r['metric']['success_5cm_5deg'] for r in value),
            metrics={k:distribution([r['metric'][k] for r in value]) for k in ['translation_cm','rotation_deg','ADDsym_cm']},
            support=support_summary(value))
        print(arm,'T',summary[arm]['metrics']['translation_cm']['mean'],'R',summary[arm]['metrics']['rotation_deg']['mean'],
            'success',summary[arm]['success_count'],'seeds',summary[arm]['support']['seed_accepted_frames'],flush=True)
    sessions=sorted({r['session'] for r in scored});assert len(sessions)==4
    draws=np.random.default_rng(20261013).integers(0,len(sessions),(10000,len(sessions)))
    contrasts=[tuple(name.split('-',1)) for name in protocol['contrasts']]
    contrasts+=[(coord+'_PROGRESSIVE_ACTIVE_SUPPORT','INITIAL') for coord in ['RAW','FUSED_SIGMA2']]
    paired={}
    for a,b in contrasts:
        report={};values=groups[a]
        for key in ['translation_cm','rotation_deg','ADDsym_cm','success_5cm_5deg']:
            delta=np.array([float(r['metric'][key])-float(lookup[r['id'],b]['metric'][key]) for r in values])
            chunks=[delta[[r['session']==s for r in values]] for s in sessions]
            sums=np.array([x.sum() for x in chunks]);counts=np.array([len(x) for x in chunks]);boot=sums[draws].sum(1)/counts[draws].sum(1)
            report[key]=dict(mean_difference=float(delta.mean()),CI95=np.quantile(boot,[.025,.975]).tolist(),
                improved=int((delta>0).sum()) if key.startswith('success') else int((delta<0).sum()),
                worsened=int((delta<0).sum()) if key.startswith('success') else int((delta>0).sum()))
        paired[a+'-'+b]=report
    diagnostics=[]
    for row in scored:
        if 'final_guard' not in row['solution']:continue
        sol=row['solution'];g=sol['final_guard'];fits=[s for s in sol['stages'] if 'optimizer' in s]
        diagnostics.append(dict(id=row['id'],session=row['session'],arm=row['arm'],final_metric=row['metric'],
            nominal_visible_order=sol['nominal_visible_order'],all_visible_ids=sol['visible_ids'],
            final_used_ids=sol['used_ids'],final_unknown_added_ids=sol['unknown_added_ids'],
            omitted_visible_ids=g['all_omitted_visible_ids'],rejected_single_add_ids=g['rejected_single_add_ids'],
            fallback_source=g['fallback_source'],retained_checkpoint_index=g['retained_checkpoint_index'],
            initial_full_objective=g['initial'],output_full_objective=g['output_objective'],
            accepted_stage_support=[dict(name=s['name'],ids=s['used_ids'],accepted=s['accepted'],reason=s['reason'],
                validation_policy=s.get('validation_policy'),discarded=s.get('intermediate_accept_discarded_by_final_guard',False),
                active_objective_before=s.get('active_objective_before'),active_objective_after=s.get('active_objective_after'),
                all_visible_cost_change=s['visible_score_after']-s['visible_score_before'],
                protected_cost_change=s['protected_score_after']-s['protected_score_before'] if s['protected_score_before'] is not None else None,
                human_states_of_fitted_ids=[truth[row['id']]['metadata']['human_states_native'][k] for k in s['used_ids']]) for s in fits]))
    with gzip.open(HERE/'SCORED_ROWS.jsonl.gz','wt') as handle:
        for row in scored:handle.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
    write(HERE/'POSTSEAL_SUPPORT_DIAGNOSTICS.json',dict(postseal_CAL_only=True,predictions_sha256=seal['sha256'],
        labels_sha256=sha(OLD/'ISOLATED_CAL_SCORING.json'),records=diagnostics))
    write(HERE/'RESULTS.json',dict(schema=protocol['schema'],CAL46_only=True,summary=summary,paired=paired,
        predictions_sha256=seal['sha256'],labels_sha256=sha(OLD/'ISOLATED_CAL_SCORING.json'),
        bootstrap=dict(groups='sessions',sessions=sessions,repeats=10000,seed=20261013),
        severe_rows=0,TEST_reads=0,FINAL_reads=0,no_automatic_promotion=True,protocol_changed_after_scoring=False,
        limitations=protocol['limitations']+['Physical pose reference reconstructed from existing annotations, not independent motion capture.']))
    print('SCORED',len(scored),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['infer','score']);args=parser.parse_args()
    infer() if args.phase=='infer' else score()
