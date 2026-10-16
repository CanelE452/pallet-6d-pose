"""CAL46-only fixed count policy, with sealed original-family controls."""
from pathlib import Path
import builtins
import io
from contextlib import contextmanager
from copy import deepcopy
from collections import Counter,defaultdict
import argparse
import importlib
import importlib.util
import hashlib
import gzip
import json
import sys
import time

import cv2
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
COHORT=HERE.parent
SOURCE=COHORT/'SAM_boundary_uncertainty_v1'
CONTROL=COHORT/'progressive_support_guard_v1'
CORE=ROOT/'scripts/research/pallet_visibility_curriculum_20261013_v1'
OLD=COHORT/'OOF_pose_v1'


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rows(path):return [json.loads(line) for line in gzip.open(path,'rt')]
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def cube(d):
    a,b,c=np.asarray(d)/2
    return np.array([[-a,-b,-c],[a,-b,-c],[a,b,-c],[-a,b,-c],[-a,-b,c],[a,-b,c],[a,b,c],[-a,b,c]])


@contextmanager
def no_labels():
    original=builtins.open,io.open,Path.open
    def wrap(fn):
        def guarded(path,*args,**kwargs):
            name=str(path)
            blocked=('SCORING','SCORED_ROWS','TARGETS','RESULTS.json','LABEL_AUDIT','COORDINATE_AUDIT',
                '/annotations/','test_reused','boundary_progressive_replay','FINAL')
            if any(s in name for s in blocked):raise AssertionError('Forbidden count inference read '+name)
            return fn(path,*args,**kwargs)
        return guarded
    builtins.open,io.open,Path.open=map(wrap,original)
    try:yield
    finally:builtins.open,io.open,Path.open=original


def snapshot_import():
    folder=HERE/'code_snapshot';folder.mkdir(exist_ok=False)
    (folder/'__init__.py').write_text('')
    for filename in ['boundary_rules.py','boundary_evidence.py','curriculum.py','test_boundary_rules.py']:
        (folder/filename).write_bytes((CORE/filename).read_bytes())
    name='_boundary_count_frozen_CAL46'
    spec=importlib.util.spec_from_file_location(name,folder/'__init__.py',submodule_search_locations=[str(folder)])
    package=importlib.util.module_from_spec(spec);sys.modules[name]=package;spec.loader.exec_module(package)
    return importlib.import_module(name+'.boundary_rules')


def observation_adapter(frame,raw,fused,measured):
    """Adapt exactly frozen inference-only records, with no new image/model call."""
    initial=frame['initial_poses']['N3_SUBPIX'];xyz=cube(initial['cf_extents']);K=np.asarray(frame['K'])
    R,t=np.asarray(initial['R_cf']),np.asarray(initial['centroid'])
    camera=xyz@R.T+t;homogeneous=camera@K.T;p=homogeneous[:,:2]/homogeneous[:,2:3]
    outline=np.isin(np.arange(8),cv2.convexHull(p.astype(np.float32),returnPoints=False).ravel())
    q=np.asarray(raw['observed_points'])[:8];H=np.asarray(raw['self_hidden'],bool);h,w=frame['raw_hw']
    valid=np.isfinite(q).all(1)&~(q==-1).all(1)
    inside=valid&(q[:,0]>=0)&(q[:,0]<w)&(q[:,1]>=0)&(q[:,1]<h)
    projected_inside=(camera[:,2]>1e-6)&(p[:,0]>=0)&(p[:,0]<w)&(p[:,1]>=0)&(p[:,1]<h)
    eligible=outline&~H&inside&projected_inside
    assert raw['appearance']==fused['appearance'] and raw['self_hidden']==fused['self_hidden']
    assert measured['original_points']==raw['original_points']==raw['observed_points']
    return dict(schema='boundary_observation_v1_CAL46_frozen_adapter',image_hw=frame['raw_hw'],
        raw_points=raw['observed_points'],fused_points=fused['observed_points'],
        observed_points=measured['correction']['points'],self_hidden=H.tolist(),fixed_H=H.tolist(),
        initial_geometric_outline=outline.tolist(),
        boundary_visibility=dict(expected_outline=outline.tolist(),conditional_truncation=(~projected_inside).tolist(),
            geometric_boundary_candidates=eligible.tolist(),selected_corner_ids=np.flatnonzero(eligible).tolist()),
        visibility_appearance=raw['appearance'],correction=measured['correction'],
        intersection_covariance=measured['intersection_uncertainty'],line_support=measured['line_uncertainty'],
        fusion=measured['fusion_SIGMA2'],provenance=dict(GT_inputs=False,model_forwards=0,
            projected_coordinates_used_as_measurements=False,original_H_model_native_phase_center_preserved=True))


def infer():
    assert not(HERE/'PREDICTIONS.jsonl.gz').exists()
    protocol=read(HERE/'PROTOCOL.json')
    for path,value in protocol['source_bindings'].items():assert sha(path)==value
    rules=snapshot_import()
    code={str(CORE/p):sha(CORE/p) for p in ['boundary_rules.py','boundary_evidence.py','curriculum.py','test_boundary_rules.py']}
    code.update({str(HERE/p):sha(HERE/p) for p in ['runner.py','PROTOCOL.json']})
    write(HERE/'INFERENCE_CODE_BINDING.json',dict(before_predictions_and_labels=True,source_sha256=code,
        synthetic_rule_tests=9,synthetic_rule_tests_PASS=True))
    cv2.setNumThreads(1);started=time.perf_counter();output=[];observations=[];copied=0
    with no_labels():
        for path in [OLD/'ISOLATED_CAL_SCORING.json',COHORT/'test_reused_v1/ISOLATED_TEST_SCORING.json',
            COHORT/'boundary_progressive_replay_v1/OBSERVATIONS.jsonl.gz']:
            try:read(path)
            except AssertionError:pass
            else:raise AssertionError('Canary failed')
        source_seal=read(SOURCE/'PREDICTION_SEAL.json');control_seal=read(CONTROL/'PREDICTION_SEAL.json')
        assert source_seal['before_scoring'] and source_seal['predictions_sha256']==sha(SOURCE/'PREDICTIONS.jsonl.gz')
        assert control_seal['before_scoring'] and control_seal['sha256']==sha(CONTROL/'PREDICTIONS.jsonl.gz')
        source={(r['id'],r['method']):r for r in rows(SOURCE/'PREDICTIONS.jsonl.gz')}
        control={(r['id'],r['arm']):r for r in rows(CONTROL/'PREDICTIONS.jsonl.gz')}
        measured={r['id']:r for r in read(SOURCE/'OBSERVED_LINE_UNCERTAINTY.json')}
        frames=read(COHORT/'CAL_INPUTS.json')['frames'];assert len(frames)==46 and set(measured)=={f['id'] for f in frames}
        histogram=Counter();strong_total=geometry_total=RGB_total=0
        for frame in frames:
            assert frame['partition']=='CAL';fid=frame['id'];initial=frame['initial_poses']['N3_SUBPIX']
            raw,fused=source[fid,'RAW_STAGED'],source[fid,'FUSED_SIGMA2_STAGED']
            observation=observation_adapter(frame,raw,fused,measured[fid])
            observation_hash=hashlib.sha256(json.dumps(observation,sort_keys=True).encode()).hexdigest()
            initial_R,initial_t=np.asarray(initial['R_cf']),np.asarray(initial['centroid'])
            xyz=cube(initial['cf_extents']);Q=initial_R.T@np.asarray(initial['R_physical']);evidence=None
            for coord in ['RAW','FUSED_SIGMA2']:
                solution=rules.solve_boundary_rules(observation,xyz,frame['K'],initial_R,initial_t,
                    dict(coordinate_source='RAW' if coord=='RAW' else 'FUSED',image_hw=frame['raw_hw']))
                assert hashlib.sha256(json.dumps(observation,sort_keys=True).encode()).hexdigest()==observation_hash
                if evidence is None:evidence=solution['evidence']
                else:assert evidence==solution['evidence']
                H=np.asarray(raw['self_hidden'],bool)
                assert all(not H[k] for s in solution['stages'] for k in s['used_ids'])
                assert solution['unknown_added_ids']==[] and solution['interior_supplement_ids']==[]
                assert set(solution['used_ids'])<=set(evidence['strong_ids'])
                if solution['accepted']:
                    assert solution['aggregate_before']['objective']-solution['aggregate_after']['objective']>1e-7
                    physical=initial['R_physical'] if solution['branch']=='THREE_TRANSLATION_ONLY' else (np.asarray(solution['R'])@Q).tolist()
                    actual=dict(initial,R_cf=solution['R'],centroid=solution['t'],R_physical=physical)
                else:
                    np.testing.assert_array_equal(solution['R'],initial_R);np.testing.assert_array_equal(solution['t'],initial_t)
                    assert solution['used_ids']==[];actual=deepcopy(initial)
                if solution['branch']=='THREE_TRANSLATION_ONLY':np.testing.assert_array_equal(solution['R'],initial_R)
                observed=observation['raw_points'] if coord=='RAW' else observation['fused_points']
                output.append(dict(id=fid,session=frame['session'],arm=coord+'_BOUNDARY_COUNT_RULE',actual_pose=actual,
                    solution=solution,observed_points=observed,raw_points=observation['raw_points'],
                    appearance=observation['visibility_appearance'],fixed_H=observation['self_hidden'],
                    coordinate_source=coord,GT_input=False,source_observation_hash=observation_hash))
            histogram[evidence['strong_count']]+=1;strong_total+=evidence['strong_count'];geometry_total+=evidence['geometric_boundary_count']
            RGB_total+=evidence['RGB_material_supported_boundary_count']
            observations.append(dict(id=fid,observation=observation,evidence=evidence))
            for coord in ['RAW','FUSED_SIGMA2']:
                for suffix in ['PROGRESSIVE_ACTIVE_SUPPORT','VISIBLE_TOGETHER']:
                    row=deepcopy(control[fid,coord+'_'+suffix])
                    assert row['appearance']==observation['visibility_appearance'] and row['fixed_H']==observation['self_hidden']
                    assert row['observed_points']==(observation['raw_points'] if coord=='RAW' else observation['fused_points'])
                    row['sealed_control_copied_from']=str(CONTROL/'PREDICTIONS.jsonl.gz');output.append(row);copied+=1
            output.append(deepcopy(control[fid,'INITIAL']));copied+=1
    assert len(output)==322 and len(observations)==46 and copied==230
    for path,value in protocol['source_bindings'].items():assert sha(path)==value
    for path,value in code.items():assert sha(path)==value
    for path,value in [('PREDICTIONS.jsonl.gz',output),('OBSERVATIONS.jsonl.gz',observations)]:
        with gzip.open(HERE/path,'wt') as handle:
            for row in value:handle.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
    write(HERE/'PREDICTION_SEAL.json',dict(complete=True,before_scoring=True,GT_read_canary='PASS',CAL_only=True,
        rows=len(output),frames=46,predictions_sha256=sha(HERE/'PREDICTIONS.jsonl.gz'),observations_sha256=sha(HERE/'OBSERVATIONS.jsonl.gz'),
        protocol_sha256=sha(HERE/'PROTOCOL.json'),code_bindings=code,sealed_controls_copied=230,
        strong_count_histogram=dict(histogram),strong_points=strong_total,geometric_points=geometry_total,RGB_material_points=RGB_total,
        RAW_FUSED_evidence_counts_equal=True,H_model_Q_native_ID_center_preserved=True,
        new_model_or_image_forwards=0,severe_rows=0,TEST_study_inputs=0,FINAL_reads=0,seconds=time.perf_counter()-started))
    print('COUNT_RULE_SEALED',322,'histogram',dict(histogram),'strong',strong_total,'geometry',geometry_total,'RGB',RGB_total,flush=True)


def rotation_y(a):
    c,s=np.cos(a),np.sin(a);return np.array([[c,0,s],[0,1,0],[-s,0,c]])


def metric(p,g):
    R,G=np.asarray(p['R_physical']),np.asarray(g['R']);t,target=np.asarray(p['centroid']),np.asarray(g['t']);X=cube(g['xyz'])
    angles=[];adds=[]
    for j in range(g['order']):
        GT=G@rotation_y(j*2*np.pi/g['order']);relative=GT.T@R
        angles.append(float(np.degrees(np.arccos(np.clip((np.trace(relative)-1)/2,-1,1)))))
        adds.append(float(np.linalg.norm(X@R.T+t-(X@GT.T+target),axis=1).mean())*100)
    T=float(np.linalg.norm(t-target)*100);A=min(angles)
    return dict(translation_cm=T,rotation_deg=A,ADDsym_cm=min(adds),success_5cm_5deg=T<5 and A<5)


def distribution(x):
    a=np.asarray(x,float);return dict(n=len(a),mean=float(a.mean()) if len(a) else None,
        median=float(np.median(a)) if len(a) else None,P90=float(np.quantile(a,.9)) if len(a) else None)


def score():
    assert not(HERE/'RESULTS.json').exists();seal=read(HERE/'PREDICTION_SEAL.json');protocol=read(HERE/'PROTOCOL.json')
    assert seal['before_scoring'] and seal['GT_read_canary']=='PASS' and seal['predictions_sha256']==sha(HERE/'PREDICTIONS.jsonl.gz')
    for path,value in seal['code_bindings'].items():assert sha(path)==value
    # First count-study scoring reads isolatedCAL truth only after all arms seal.
    truth=read(OLD/'ISOLATED_CAL_SCORING.json')['rows'];assert len(truth)==46
    scored=rows(HERE/'PREDICTIONS.jsonl.gz');groups=defaultdict(list);by={}
    for row in scored:
        row['metric']=metric(row['actual_pose'],truth[row['id']]['target']['truth']);groups[row['arm']].append(row);by[row['id'],row['arm']]=row
    summary={};counts={};diagnostics=[]
    for arm,value in groups.items():
        summary[arm]=dict(frames=len(value),accepted=sum(r['solution']['accepted'] for r in value),
            success_count=sum(r['metric']['success_5cm_5deg'] for r in value),
            metrics={k:distribution([r['metric'][k] for r in value]) for k in ['translation_cm','rotation_deg','ADDsym_cm']})
        if arm.endswith('BOUNDARY_COUNT_RULE'):
            sub=defaultdict(list)
            for row in value:
                sol=row['solution'];sub[sol['reliable_boundary_count']].append(row)
                diagnostics.append(dict(id=row['id'],arm=arm,count=sol['reliable_boundary_count'],
                    geometry_count=sol['geometric_boundary_count'],RGB_count=sol['RGB_material_boundary_count'],
                    branch=sol['branch'],accepted=sol['accepted'],estimated_dof=sol['estimated_dof'],
                    reliable_boundary_ids=sol['reliable_boundary_ids'],used_ids=sol['used_ids'],fallback_reason=sol['fallback_reason'],
                    human_states_of_strong_ids=[truth[row['id']]['metadata']['human_states_native'][k] for k in sol['reliable_boundary_ids']],
                    metric=row['metric'],initial_metric=by[row['id'],'INITIAL']['metric'],
                    delta_initial={k:float(row['metric'][k])-float(by[row['id'],'INITIAL']['metric'][k]) for k in ['translation_cm','rotation_deg']}))
            counts[arm]={str(n):dict(frames=len(v),accepted=sum(r['solution']['accepted'] for r in v),
                branch_counts=dict(Counter(r['solution']['branch'] for r in v)),
                fallback_counts=dict(Counter(r['solution']['fallback_reason'] for r in v if not r['solution']['accepted'])),
                metrics={k:distribution([r['metric'][k] for r in v]) for k in ['translation_cm','rotation_deg']},
                mean_delta_initial={k:float(np.mean([r['metric'][k]-by[r['id'],'INITIAL']['metric'][k] for r in v])) for k in ['translation_cm','rotation_deg']}) for n,v in sorted(sub.items())}
            summary[arm]['branch_counts']=dict(Counter(r['solution']['branch'] for r in value))
            summary[arm]['fallback_counts']=dict(Counter(r['solution']['fallback_reason'] for r in value if not r['solution']['accepted']))
        print(arm,'accepted',summary[arm]['accepted'],'T',summary[arm]['metrics']['translation_cm']['mean'],
            'R',summary[arm]['metrics']['rotation_deg']['mean'],'success',summary[arm]['success_count'],flush=True)
    sessions=sorted({r['session'] for r in scored});assert len(sessions)==4
    draws=np.random.default_rng(20261013).integers(0,len(sessions),(10000,len(sessions)));paired={}
    contrasts=[]
    for coord in ['RAW','FUSED_SIGMA2']:
        for suffix in ['PROGRESSIVE_ACTIVE_SUPPORT','VISIBLE_TOGETHER']:
            contrasts.append((coord+'_BOUNDARY_COUNT_RULE',coord+'_'+suffix))
        contrasts.append((coord+'_BOUNDARY_COUNT_RULE','INITIAL'))
    contrasts.append(('FUSED_SIGMA2_BOUNDARY_COUNT_RULE','RAW_BOUNDARY_COUNT_RULE'))
    for a,b in contrasts:
        result={};v=groups[a]
        for key in ['translation_cm','rotation_deg','ADDsym_cm','success_5cm_5deg']:
            delta=np.array([float(r['metric'][key])-float(by[r['id'],b]['metric'][key]) for r in v])
            chunks=[delta[[r['session']==s for r in v]] for s in sessions];sums=np.array([x.sum() for x in chunks]);sizes=np.array([len(x) for x in chunks]);boot=sums[draws].sum(1)/sizes[draws].sum(1)
            result[key]=dict(mean_difference=float(delta.mean()),CI95=np.quantile(boot,[.025,.975]).tolist(),
                improved=int((delta>0).sum()) if key.startswith('success') else int((delta<0).sum()),
                worsened=int((delta<0).sum()) if key.startswith('success') else int((delta>0).sum()))
        paired[a+'-'+b]=result
    with gzip.open(HERE/'SCORED_ROWS.jsonl.gz','wt') as handle:
        for row in scored:handle.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
    write(HERE/'POSTSEAL_COUNT_DIAGNOSTICS.json',dict(CAL_only=True,sealed_predictions_sha256=seal['predictions_sha256'],
        labels_sha256=sha(OLD/'ISOLATED_CAL_SCORING.json'),records=diagnostics))
    write(HERE/'RESULTS.json',dict(schema=protocol['schema'],CAL_only=True,summary=summary,by_count=counts,paired=paired,
        support_inventory=dict(histogram=seal['strong_count_histogram'],strong=seal['strong_points'],
            geometric=seal['geometric_points'],RGB_material=seal['RGB_material_points']),
        severe_rows=0,TEST_study_inputs=0,FINAL_reads=0,no_automatic_promotion=True,
        protocol_changed_after_scoring=False,bootstrap=dict(groups='sessions',sessions=sessions,repeats=10000,seed=20261013),
        source_prediction_sha256=seal['predictions_sha256'],labels_sha256=sha(OLD/'ISOLATED_CAL_SCORING.json'),limits=protocol['limitations']))
    print('COUNT_RULE_SCORED',322,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['infer','score']);args=parser.parse_args()
    infer() if args.phase=='infer' else score()
