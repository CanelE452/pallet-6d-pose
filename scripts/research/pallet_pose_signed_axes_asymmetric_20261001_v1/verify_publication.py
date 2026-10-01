"""Independently verify asymmetric-Huber271 publication against frozen artifacts; no fitting or GT metrics."""
import argparse
import ast
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import sys
from urllib.parse import unquote
import numpy as np
from PIL import Image
from . import common as C

HASH_KEYS=('signed_target_sha','input_difference_sha','base_context_sha','errors_sha',
    'scaled_excess_sha','original_valid_sha','direction_raw_sha','direction_difference_sha','extended_input_sha')
REAL_MODELS=C.MODEL_NAMES+('R0','PRIOR1','FULL125','DIVERSE251_s1','DIVERSE251_s2','DIVERSE251_s3',
    'SINGLE251_s1','SINGLE251_s2','SINGLE251_s3')


def install_guard():
    """Publication checks cached artifacts; raw reference labels are unnecessary."""
    allowed={C.DOC/'PUBLIC_REVIEW.json',C.DOC/'PUBLIC_REVIEW_KO.md'}
    def hook(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        path=Path(os.fsdecode(args[0])).resolve();mode=args[1];flags=args[2] if len(args)>2 and isinstance(args[2],int) else 0
        if not path.is_relative_to(C.ROOT):return
        if (isinstance(mode,str) and any(c in mode for c in 'wax+')) or flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):
            assert path in allowed,('PUBLIC_REVIEW_WRITE_SCOPE',str(path));return
        assert not any(token in str(path) for token in ('GEOMETRY_SIDETABLE.npz','SOURCE_TRAIN_LABELS.npz','/SYNTH_RECORDS.json',
            '/SYNTH_LABELS.npz','/SOURCE_MANIFEST.json','/DIMENSION_SIDECAR.json','/real_gt_v2/annotations/',
            '/GEOMETRY_RESOLVED_POSE_GT','TRUTH_FOR_DISPLAY','AXIS_REVIEW_MANIFEST')),('PUBLICATION_RAW_REFERENCE_DENIED',str(path))
    sys.addaudithook(hook)


def array_sha(value):
    value=np.ascontiguousarray(value)
    digest=hashlib.sha256(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape)).encode());digest.update(value.tobytes())
    return digest.hexdigest()


def publication_protocol():
    """Authenticate the seal; leave TRAIN label-value checks to TRAIN audit."""
    seal=C.read(C.DOC/'TRAIN_PROTOCOL_SHA.json')
    assert seal['path']==str((C.DOC/'TRAIN_PROTOCOL.json').relative_to(C.ROOT))
    C.verify(seal)
    value=C.read(C.DOC/'TRAIN_PROTOCOL.json')
    for binding in value['codes']:C.verify(binding)
    return value


def all_bindings(value):
    if isinstance(value,dict):
        if {'path','sha256'}<=value.keys():C.verify(value)
        for item in value.values():all_bindings(item)
    elif isinstance(value,list):
        for item in value:all_bindings(item)


def csv_rows(path):
    assert b'\r' not in path.read_bytes()
    with path.open(newline='') as handle:return list(csv.DictReader(handle))


def scalar_equal(text,value):
    if value is None:return text==''
    if isinstance(value,(int,float)) and not isinstance(value,bool):return float(text)==value
    return text==str(value)


def check_metadata(report,training,source,complete,fits):
    protocol=publication_protocol()
    assert report['complete'] and report['train_protocol']==C.bind(C.DOC/'TRAIN_PROTOCOL.json')
    for key,value in dict(feature_dim=271,base_feature_dim=253,direction_dim=18,output_dim=2).items():
        assert report[key]==training[key]==complete[key]==protocol[key]==value
    assert report['training_verification']==C.bind(C.DOC/'TRAIN_CONVERGENCE.json')
    assert report['source_verification']==C.bind(C.DOC/'SOURCE_VAL_VERIFICATION.json')
    assert report['source_gate']==C.bind(C.DOC/'SOURCE_VAL_GATE.json')
    for key in ('direction_receipt','direction_representation','direction_verification'):
        assert report[key]==protocol['inputs'][key]
    assert report['direction_normalization_sha']==protocol['direction_normalization_sha']
    for key,value in dict(loss_rule='FULL_FRAME_VALID_CANDIDATE_TWO_AXIS_ASYMMETRIC_HUBER_PLUS_SIGN_LOGISTIC',underprediction_coefficient=1.,underprediction_cost=2.,overprediction_cost=1.,underprediction_zero_curvature=0.).items():
        assert report[key]==training[key]==complete[key]==protocol[key]==value
    assert report['new_metric_calls']==report['new_image_forwards']==report['new_PnP_solves']==0
    assert report['goal_complete'] is False
    assert training['protocol']==complete['protocol']==source['protocol']==report['train_protocol']
    independent_source=C.read(C.DOC/'SOURCE_VAL_VERIFICATION.json')
    assert independent_source['source_gate']==report['source_gate']
    assert independent_source['source_gate_PASS']==source['PASS']
    assert independent_source['checks_total']==45 and independent_source['checks_passed']==source['checks_passed']
    assert complete['complete'] and complete['all_certified'] and complete['fit_count']==4
    assert sum(f['objective_calls'] for f in fits.values())==complete['total_objective_calls']
    assert sum(f['iterations'] for f in fits.values())==complete['total_iterations']
    assert source['complete'] and source['checks_total']==45
    assert source['checks_passed']==sum(bool(v) for a in source['comparisons'].values() for b in a.values() for v in b['checks'].values())
    assert source['PASS']==source['real_routing_authorized']==(source['checks_passed']==45)
    assert report['actual_new_fits']==report['actual_training_attempts']==report['certified_models']==4
    assert report['source_VAL_evaluated'] and report['source_VAL_checks_passed']==source['checks_passed']
    assert report['source_VAL_checks_total']==45
    assert report['objective_calls']==complete['total_objective_calls'] and report['iterations']==complete['total_iterations']
    assert report['zero_initialization_verified']
    assert report['previous_weight_warmstart'] is False
    assert report['previous_source_gate']==C.bind(C.PREVIOUS_DOC/'SOURCE_VAL_GATE.json')
    assert report['previous_training_verification']==C.bind(C.PREVIOUS_DOC/'TRAIN_CONVERGENCE.json')
    comparisons={}
    for model in C.MODEL_NAMES:
        r=training['models'][model];q=r['independent_recompute'];old=r['previous_fixed_direction']
        assert old['original_choice_replay_PASS']
        assert old['checkpoint']['path'].startswith(str(C.PREVIOUS_RAW.relative_to(C.ROOT)))
        assert old['original_symmetric_certificate']['loss_rule']=='FULL_FRAME_VALID_CANDIDATE_TWO_AXIS_HUBER_PLUS_SIGN_LOGISTIC'
        assert old['original_objective_and_original_loss_gradient_parity'] and old['same271_input_prediction_parity']
        assert old['new_loss_gradient_not_compared_to_original_certificate']
        np.testing.assert_allclose(old['same_new_objective']['Huber_symmetric'],old['original_symmetric_objective']['Huber'],rtol=0,atol=1e-10)
        np.testing.assert_allclose(old['same_new_objective']['objective'],old['original_symmetric_objective']['objective']+old['same_new_objective']['Huber_underprediction'],rtol=0,atol=1e-10)
        assert 'Sign_logistic' in old['same_new_objective']
        assert q['Sign_logistic']>=0 and q['Huber']>=0 and q['L2_penalty']>=0
        assert q['Huber_symmetric']>=0 and q['Huber_underprediction']>=0
        np.testing.assert_allclose(q['Huber'],q['Huber_symmetric']+q['Huber_underprediction'],rtol=1e-14,atol=1e-14)
        np.testing.assert_allclose(q['objective'],q['Huber']+q['Sign_logistic']+q['L2_penalty'],rtol=1e-14,atol=1e-14)
        assert q['gradient_linf']<=1e-8 and q['certified_gap_upper_bound']<=1e-6
        difference=old['same_new_objective']['objective']-q['objective']
        assert r['same_new_objective_decrease']==difference
        comparisons[model]=dict(old_weight=old['same_new_objective']['objective'],new_weight=q['objective'],decrease=difference)
    assert report['same_new_objective_by_model']==comparisons
    assert report['csv_rows']=={'SOURCE_VAL_FRAME_RESULTS.csv':8192,'SOURCE_VAL_CHECKS.csv':45,
        'TRAINING_OBJECTIVE_LOG.csv':complete['total_objective_calls'],
        'TRAINING_ITERATION_LOG.csv':complete['total_iterations']}
    return comparisons


def verify_not_run(report,source):
    assert not source['PASS'] and not source['real_routing_authorized']
    stopped=C.read(C.DOC/'REAL_EVALUATION_NOT_RUN.json');all_bindings(stopped)
    assert stopped['complete'] and stopped['status']=='NOT_RUN_SOURCE_GATE_FAILED'
    assert stopped['source_gate']==C.bind(C.DOC/'SOURCE_VAL_GATE.json')
    assert stopped['checks_passed']==source['checks_passed'] and stopped['checks_total']==45
    assert stopped['learned_real_routes']==stopped['real_metric_calls']==stopped['raw_real_reference_reads']==0
    paths=[C.DOC/name for name in ('REAL_PROTOCOL.json','REAL_PROTOCOL_SHA.json','REAL_ROUTING_LOCK.json','REAL_RESULTS.json',
        'REAL_FRAME_RESULTS.csv','REAL_REFERENCE_BINDINGS.json','REAL_DETAILED_COMPARISONS.json','REAL_DIRECTION_INPUTS.json','REAL_DIRECTION_INPUTS_FAILED.json')]
    paths += [C.RAW/name for name in ('REAL_CHOICES.json','POSE_METRICS.json','REAL_DIRECTIONS.npz')]
    expected={str(path.relative_to(C.ROOT)):True for path in paths}
    assert stopped['absence_checks']==expected and len(expected)==12
    for path,absent in stopped['absence_checks'].items():assert absent and not (C.ROOT/path).exists(),path
    assert report['status']=='CERTIFIED_TRAIN_SOURCE_GATE_FAILED'
    assert report['real_not_run']==C.bind(C.DOC/'REAL_EVALUATION_NOT_RUN.json')
    assert report['real_results'] is None and report['real_verification'] is None
    assert not report['learned_real_evaluated'] and not report['current_real_TR_effect_measured']
    assert not report['stable_joint_improvement_achieved'] and not report['goal_complete'] and not report['method_success']
    assert report['baseline_panels']==6 and report['current_learned_real_panels']==0
    assert report['new_metric_calls']==report['new_image_forwards']==report['new_PnP_solves']==0
    return stopped


def check_links_and_files():
    pending={'PUBLIC_REVIEW.json','PUBLIC_REVIEW_KO.md','PUBLICATION_MANIFEST.json'}
    checkout=Path('/tmp/pallet-pose-github-review-20260930');links=0
    for p in C.DOC.rglob('*.md'):
        for link in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if link.startswith(('https://','http://','#')):continue
            target=(p.parent/unquote(link.split('#')[0])).resolve()
            assert target.is_relative_to(C.ROOT),(p,link)
            if target.parent==C.DOC and target.name in pending:pass
            elif target.is_relative_to(C.DOC) or target.is_relative_to(C.HERE):assert target.exists(),(p,link)
            else:assert (checkout/target.relative_to(C.ROOT)).exists(),(p,link)
            links+=1
    files=[C.ROOT/'.gitignore',C.ROOT/'readme.md']
    for folder in (C.DOC,C.HERE):
        files += [p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in pending]
    for p in files:
        if p.suffix=='.py':ast.parse(p.read_text())
    return links,files


def verify_source_exports(source):
    C.verify(source['metrics'])
    lock=C.read(C.DOC/'SOURCE_VAL_ROUTING_LOCK.json');C.verify(lock['choices'])
    choices=C.read(C.ROOT/lock['choices']['path'])
    rows=csv_rows(C.DOC/'SOURCE_VAL_FRAME_RESULTS.csv')
    columns={'model','id','split','available','T_cm','R_deg','candidate','fallback'}
    assert all(set(row)==columns for row in rows)
    index={(r['model'],r['id']):r for r in rows}
    assert len(rows)==len(index)==8192
    with np.load(C.ROOT/source['metrics']['path'],allow_pickle=False) as z:
        for model in z['models'].tolist():
            for i,fid in enumerate(z['ids'].tolist()):
                row=index[model,fid]
                assert float(row['T_cm'])==z[model][i,0] and float(row['R_deg'])==z[model][i,1]
                assert row['available']==str(bool(np.isfinite(z[model][i]).all())) and row['split']=='VAL'
                pick=choices['records'].get(model,{}).get(fid,{})
                assert scalar_equal(row['candidate'],pick.get('candidate_name','fixed_GEO'))
                assert row['fallback']==str(pick.get('fallback',False))
    checks=csv_rows(C.DOC/'SOURCE_VAL_CHECKS.csv')
    expected=[dict(model=m,baseline=b,criterion=k,PASS=v) for m,g in source['comparisons'].items()
        for b,comp in g.items() for k,v in comp['checks'].items()]
    assert len(checks)==len(expected)==45
    for row,raw in zip(checks,expected):assert row=={k:str(v) for k,v in raw.items()}
    return dict(frame_rows=8192,checks=45,passed_checks=source['checks_passed'],failed_checks=45-source['checks_passed'])


def verify_training_exports(report,complete):
    protocol=publication_protocol();prefit=bound_read(protocol['inputs']['prefit_review'])
    training=C.read(C.DOC/'TRAIN_CONVERGENCE.json')
    basis=bound_read(protocol['inputs']['rbf_basis'])
    directions=bound_read(protocol['inputs']['direction_receipt'])
    mean18=np.asarray(directions['normalization']['mean18'],np.float32)
    std18=np.asarray(directions['normalization']['std18'],np.float32)
    assert mean18.shape==std18.shape==(18,) and (std18>=np.float32(1e-6)).all()
    assert array_sha(np.stack([mean18,std18]))==protocol['direction_normalization_sha']
    fits={};logs=[];exports={}
    for model in C.MODEL_NAMES:
        fit=C.read(C.DOC/f'FIT_{model}.json');all_bindings(fit)
        ck=C.read(C.ROOT/fit['checkpoint']['path']);assert ck['certificate']['PASS']
        assert ck['schema']=='pallet_pose_signed_axes_asymmetric_linear271x2_v1'
        assert np.asarray(ck['weight']).shape==(271,2) and ck['solver_rule']=='BLOCK_GENERALIZED_NEWTON_ARMIJO'
        start=bound_read(fit['START'])
        assert start['initialization']=='all_zero_float64'
        assert start['initial_weight_sha']==array_sha(np.zeros((271,2),np.float64))
        assert array_sha(np.asarray(ck['weight'],np.float64))==fit['final_weight_sha']
        assert ck['rbf_basis']==basis['basis'] and ck['basis_SHA_bind']==ck['rbf_basis_binding']==protocol['inputs']['rbf_basis']
        mean94,std94=np.asarray(ck['mean'],np.float32),np.asarray(ck['std'],np.float32)
        assert mean94.shape==std94.shape==(94,) and (std94>=np.float32(1e-6)).all()
        assert array_sha(np.stack([mean94,std94]))==ck['normalization_sha']==fit['normalization_sha']
        np.testing.assert_array_equal(np.asarray(ck['direction_mean'],np.float32),mean18)
        np.testing.assert_array_equal(np.asarray(ck['direction_std'],np.float32),std18)
        for artifact in (ck,fit,start,complete,prefit,training):
            assert artifact['direction_receipt_binding']==protocol['inputs']['direction_receipt']
            assert artifact['direction_normalization_sha']==protocol['direction_normalization_sha']
        for key in HASH_KEYS:
            assert ck[key]==fit[key]==start[key]==complete[key+'_by_model'][model]==prefit['models'][model][key]==training['models'][model]['hashes'][key]
        for key,value in dict(loss_rule='FULL_FRAME_VALID_CANDIDATE_TWO_AXIS_ASYMMETRIC_HUBER_PLUS_SIGN_LOGISTIC',
                sign_rule='NONZERO_SIGNED_TARGET_AXES',sign_coefficient=1.,underprediction_coefficient=1.,underprediction_cost=2.,overprediction_cost=1.,underprediction_zero_curvature=0.).items():
            assert ck[key]==fit[key]==start[key]==complete[key]==protocol[key]==training[key]==prefit[key]==value
            assert ck['certificate'][key]==value
        assert ck['certificate']==fit['certificate']
        cert=ck['certificate']
        assert cert['gradient_linf']<=1e-8 and cert['gradient_l2_squared_over_2lambda']<=1e-6
        np.testing.assert_allclose(cert['objective_value'],cert['Huber']+cert['Sign_logistic']+cert['L2_penalty'],rtol=1e-14,atol=1e-14)
        np.testing.assert_allclose(cert['Huber'],cert['Huber_symmetric']+cert['Huber_underprediction'],rtol=1e-14,atol=1e-14)
        assert cert['Huber_symmetric']>=0 and cert['Huber_underprediction']>=0
        assert cert['Huber']>=0 and cert['Sign_logistic']>=0 and cert['L2_penalty']>=0
        public=C.DOC/'model_parameters'/f'{model}.json'
        assert public.read_bytes()==(C.ROOT/fit['checkpoint']['path']).read_bytes()
        exports[model]=dict(local=fit['checkpoint'],published=C.bind(public))
        raw=[json.loads(line) for line in (C.ROOT/fit['trace']['path']).read_text().splitlines()]
        calls=[r for r in raw if r['event']=='objective'];steps=[r for r in raw if r['event']=='iteration']
        assert len(calls)==fit['objective_calls']<=2000 and len(steps)==fit['iterations']<=1000
        assert [r['call'] for r in calls]==list(range(1,len(calls)+1))
        assert [r['iteration'] for r in steps]==list(range(1,len(steps)+1))
        assert calls[-1]['armijo_accepted'] and calls[-1]['call']==fit['final_accepted_call']==ck['final_accepted_call']
        assert calls[-1]['weight_sha']==fit['final_weight_sha']
        assert calls[-1]['objective']==cert['objective_value'] and calls[-1]['Sign_logistic']==cert['Sign_logistic']
        for key in ('Huber_symmetric','Huber_underprediction'):assert calls[-1][key]==cert[key]
        assert complete['final_objective_components_by_model'][model]=={key:cert[key] for key in ('objective_value','Huber','Huber_symmetric','Huber_underprediction','Sign_logistic','L2_penalty')}
        callmap={r['call']:r for r in calls}
        for row in steps:
            for key in ('Huber_symmetric','Huber_underprediction'):assert row[key]==callmap[row['accepted_call']][key]
        for row in raw:
            for key in ('underprediction_coefficient','underprediction_cost','overprediction_cost','underprediction_zero_curvature'):assert row[key]==ck[key]
            for key in HASH_KEYS:assert row[key]==ck[key]
            for key in ('direction_receipt_binding','direction_normalization_sha','basis_SHA_bind'):assert row[key]==ck[key]
            for key in ('feature_dim','base_feature_dim','direction_dim','output_dim','direction_rule','direction_normalization','input_rule','feature_map'):
                assert row[key]==ck[key]==fit[key]==start[key]==complete[key]==protocol[key]
            assert row['sign_rule']=='NONZERO_SIGNED_TARGET_AXES' and row['sign_coefficient']==1.
            assert row['loss_rule']=='FULL_FRAME_VALID_CANDIDATE_TWO_AXIS_ASYMMETRIC_HUBER_PLUS_SIGN_LOGISTIC'
            if row.get('evaluated',True):
                np.testing.assert_allclose(row['Huber'],row['Huber_symmetric']+row['Huber_underprediction'],rtol=1e-14,atol=1e-14)
                np.testing.assert_allclose(row['objective'],row['Huber']+row['Sign_logistic']+row['L2_penalty'],rtol=1e-14,atol=1e-14)
        logs.extend(dict(model=model,**row) for row in raw)
        fits[model]=fit
    assert report['exports']==exports
    assert {p.name for p in (C.DOC/'model_parameters').glob('*.json')}=={f'{m}.json' for m in C.MODEL_NAMES}
    counts={}
    for event,filename,total in [('objective','TRAINING_OBJECTIVE_LOG.csv',complete['total_objective_calls']),('iteration','TRAINING_ITERATION_LOG.csv',complete['total_iterations'])]:
        expected=[row for row in logs if row['event']==event];actual=csv_rows(C.DOC/filename)
        assert len(expected)==len(actual)==total
        # Initial and trial rows have optional fields. CSV uses their union.
        keys=set().union(*(row.keys() for row in expected))
        for row,raw in zip(actual,expected):
            assert set(row)==keys,(event,set(row)^keys)
            for key in keys:assert scalar_equal(row[key],raw.get(key)),(event,key)
        counts[event]=total
    return fits,logs,counts


def verify_figure_values(report,source,training,logs,fits,real=None):
    prior=C.read(C.PREVIOUS_DOC/'SOURCE_VAL_GATE.json');values={}
    for model in C.MODEL_NAMES:
        rows=[r for r in logs if r['model']==model and r['event']=='objective']
        values[f'train_{model}']=dict(calls=[r['call'] for r in rows],objective=[r['objective'] for r in rows],
            Huber=[r['Huber'] for r in rows],Huber_symmetric=[r['Huber_symmetric'] for r in rows],Huber_underprediction=[r['Huber_underprediction'] for r in rows],Sign_logistic=[r['Sign_logistic'] for r in rows],
            L2_penalty=[r['L2_penalty'] for r in rows],
            previous_weight_same_new_J=training['models'][model]['previous_fixed_direction']['same_new_objective']['objective'],
            gap_ratio_to_limit=[r['gradient_gap_upper_bound']/1e-6 for r in rows],
            linf_ratio_to_limit=[r['gradient_linf']/1e-8 for r in rows],armijo_accepted=[r['armijo_accepted'] for r in rows],
            final_accepted_call=fits[model]['final_accepted_call'])
    for axis in ('translation_cm','rotation_deg'):
        for q in ('median','P90'):
            values[f'source_{axis}_{q}']=dict(
                previous_direction271=[prior['summaries'][m]['full_population'][axis][q] for m in C.MODEL_NAMES],
                current_asymmetric271=[source['summaries'][m]['full_population'][axis][q] for m in C.MODEL_NAMES],
                fixed_R0_GEO=source['summaries']['R0_GEO']['full_population'][axis][q])
    for axis in ('T','R'):
        values[f'train_nonanchor_{axis}']={metric:[training['models'][m]['regression_statistics']['nonanchor_valid_candidates']['axes'][axis][metric] for m in C.MODEL_NAMES] for metric in ('MAE','RMSE')}
    values['train_unsafe_count']=dict(
        previous_direction271=[training['models'][m]['previous_fixed_direction']['statistics']['anchor_violations']['either']['count'] for m in C.MODEL_NAMES],
        current_asymmetric271=[training['models'][m]['statistics']['anchor_violations']['either']['count'] for m in C.MODEL_NAMES])
    if real is not None:
        for pop in ('NATURAL99','CLEAN29','WOOD45'):
            for axis in ('translation_cm','rotation_deg'):
                for quantile in ('median','P90'):
                    values[f'real_{pop}_{axis}_{quantile}']=dict(models=real['models'],
                        values=[real['summaries'][pop][m]['full_population'][axis][quantile] for m in real['models']])
        values['real_gate_matrix']=dict(rows=list(real['stability']['gates']),
            columns=['matched','original_SINGLE251','combined_AND'],
            values=[[int(real['matched_intervention_stability']['gates'][key]['PASS']),
                int(real['original_goal_stability']['gates'][key]['PASS']),int(value['PASS'])]
                for key,value in real['stability']['gates'].items()])
    assert report['figure_values']==values
    return len(values)


MODELS = ('R0',)
SIGNS = tuple(itertools.product((-1., 1.), repeat=3))
EDGES = tuple((i,j) for i in range(8) for j in range(i+1,8)
              if sum(x != y for x,y in zip(SIGNS[i],SIGNS[j])) == 1)


def bound_read(binding):
    C.verify(binding)
    return C.read(C.ROOT / binding['path'])


def scalar_projection(pose, K):
    """Scalar pinhole projection independent of report's batch matrix code."""
    assert pose['available']
    rotation=np.asarray(pose['R_cf'],np.float64)
    translation=np.asarray(pose['centroid'],np.float64)
    dimensions=np.asarray(pose['cf_extents'],np.float64)
    matrix=np.asarray(K,np.float64)
    assert rotation.shape == matrix.shape == (3,3)
    assert translation.shape == dimensions.shape == (3,) and (dimensions>0).all()
    corners=[];camera=[];uv=[]
    for sign in SIGNS:
        local=[sign[j]*float(dimensions[j])/2 for j in range(3)]
        point=[sum(float(rotation[j,k])*local[k] for k in range(3))+float(translation[j]) for j in range(3)]
        homogeneous=[sum(float(matrix[j,k])*point[k] for k in range(3)) for j in range(3)]
        assert np.isfinite(homogeneous).all() and homogeneous[2] != 0
        corners.append(local);camera.append(point)
        uv.append([homogeneous[0]/homogeneous[2],homogeneous[1]/homogeneous[2]])
    return np.asarray(corners),np.asarray(camera),np.asarray(uv)


def gallery_selfcheck():
    pose={'available':True,'R_cf':np.eye(3).tolist(),'centroid':[0.,0.,2.], 'cf_extents':[2.,2.,2.]}
    local,camera,uv=scalar_projection(pose,[[100,0,50],[0,100,40],[0,0,1]])
    np.testing.assert_array_equal(local[0],[-1,-1,-1])
    np.testing.assert_array_equal(camera[0],[-1,-1,1])
    np.testing.assert_array_equal(uv[0],[-50,-60])
    assert len(EDGES)==12
    print('GALLERY_PROJECTION_SELFCHECK_PASS artifact_reads=0')


def published_binding(doc, path):
    manifest=C.read(doc/'PUBLICATION_MANIFEST.json')
    assert manifest['complete']
    matches=[b for b in manifest['files'] if b['path']==str(path.relative_to(C.ROOT))]
    assert len(matches)==1
    C.verify(matches[0])
    published=Path('/tmp/pallet-pose-github-review-20260930')/path.relative_to(C.ROOT)
    assert published.is_file() and path.read_bytes()==published.read_bytes()
    return matches[0]


def verify_gallery(report=None):
    """Historical baseline only; return evidence for root's PUBLIC_REVIEW."""
    if report is None:report=C.read(C.DOC/'REPORT_DATA.json')
    not_run=bound_read(C.bind(C.DOC/'REAL_EVALUATION_NOT_RUN.json'))
    assert not_run['complete'] and not_run['status']=='NOT_RUN_SOURCE_GATE_FAILED'
    assert not_run['failed_stage']=='SOURCE_VAL'
    assert not_run['learned_real_routes']==not_run['real_metric_calls']==not_run['raw_real_reference_reads']==0
    assert not_run['method_training_fits_completed']==4
    assert all(not_run['absence_checks'].values())
    for path in not_run['absence_checks']:assert not (C.ROOT/path).exists()
    gate=bound_read(not_run['source_gate'])
    assert gate['complete'] and not gate['PASS'] and not gate['real_routing_authorized']
    assert gate['checks_total']==not_run['checks_total']==45
    assert gate['checks_passed']==not_run['checks_passed']
    assert gate['failed_checks']==not_run['failed_checks']
    complete=bound_read(not_run['method_training_complete'])
    assert complete['complete'] and complete['all_certified'] and complete['fit_count']==4
    assert not not_run['method_success'] and not not_run['goal_complete']

    for path in (C.DOC/'REAL_PROTOCOL.json',C.DOC/'REAL_ROUTING_LOCK.json',
                 C.RAW/'REAL_CHOICES.json',C.DOC/'REAL_RESULTS.json',
                 C.DOC/'REAL_FRAME_RESULTS.csv',C.RAW/'POSE_METRICS.json'):
        assert not path.exists()
    prior_path=C.PREVIOUS_DOC/'REPORT_DATA.json'
    prior_binding=published_binding(C.PREVIOUS_DOC,prior_path)
    prior=bound_read(prior_binding)
    selected_path=C.ANCHOR_DOC/'REPORT_DATA.json'
    selection_binding=published_binding(C.ANCHOR_DOC,selected_path)
    original_selection=bound_read(selection_binding)
    historical=prior['illustrations']; original=original_selection['illustrations']
    assert len(historical)==len(original)==6
    fixed_ids=[r['id'] for r in original]
    assert [r['id'] for r in historical]==fixed_ids
    prior_protocol_binding=published_binding(C.RBF_DOC,C.RBF_DOC/'REAL_PROTOCOL.json')
    historical_rbf=bound_read(published_binding(C.RBF_DOC,C.RBF_DOC/'REPORT_DATA.json'))
    assert historical_rbf['real_protocol']==prior_protocol_binding
    protocol=bound_read(prior_protocol_binding)
    metadata=bound_read(protocol['inputs']['metadata'])
    groups=bound_read(protocol['inputs']['groups'])
    poses=bound_read(protocol['inputs']['poses'])
    rows={r['id']:r for r in metadata};assert len(rows)==173
    csv_binding=published_binding(C.RBF_DOC,C.RBF_DOC/'REAL_FRAME_RESULTS.csv')
    with (C.ROOT/csv_binding['path']).open(newline='') as handle:
        table=list(csv.DictReader(handle))
    assert len(table)==2249 and len({(r['model'],r['id']) for r in table})==2249
    lookup={r['id']:r for r in table if r['model']=='R0'}
    assert set(lookup)==set(rows)
    natural=groups['NATURAL99']; assert len(natural)==99
    recordings=sorted({rows[fid]['recording'] for fid in natural});assert len(recordings)==6
    chosen=[min((fid for fid in natural if rows[fid]['recording']==rec),
        key=lambda fid:(-float(lookup[fid]['translation_cm']),fid)) for rec in recordings]
    assert chosen==fixed_ids
    details=report['illustrations'];assert len(details)==6
    assert [r['id'] for r in details]==fixed_ids and len(set(fixed_ids))==6
    assert report['actual_RGB_images']==6
    assert report['learned_real_evaluated'] is False
    assert report['method_success'] is False and report['goal_complete'] is False
    assert report['baseline_panels']==6 and report['current_learned_real_panels']==0
    selection=bound_read(report['gallery'])
    assert selection['complete'] and selection['status']=='HISTORICAL_R0_INPUT_ILLUSTRATIONS_ONLY'
    assert not selection['current_learned_real_evaluated']
    assert selection['illustrations']==details
    assert selection['source_selection_report']==selection_binding
    assert selection['historical_baseline_report']==prior_binding
    assert selection['not_run']==C.bind(C.DOC/'REAL_EVALUATION_NOT_RUN.json')
    assert selection['actual_RGB_images']==6 and selection['panels']==6
    assert selection['new_metric_calls']==selection['new_image_forwards']==selection['new_PnP_solves']==0
    maximum_uv_difference=0.;maximum_camera_difference=0.;results=[];panel_count=0
    for detail,prior_detail,original_detail in zip(details,historical,original):
        fid=detail['id'];row=rows[fid]
        assert detail['recording']==row['recording']==prior_detail['recording']==original_detail['recording']
        assert detail['dimensions_m']==row['xyz']==prior_detail['dimensions_m']==original_detail['dimensions_m']
        assert detail['image']==row['image']==prior_detail['image']==original_detail['image']
        assert detail['K']==row['K']==prior_detail['K']
        assert detail['current_learned_real_evaluated'] is False
        assert detail['corner_signs']==[list(s) for s in SIGNS] and detail['edges']==[list(e) for e in EDGES]
        C.verify(detail['image'])
        with Image.open(C.ROOT/detail['image']['path']) as image:
            width,height=image.size;rgb=np.asarray(image.convert('RGB'))
        assert [height,width]==row['hw']==detail['image_hw']==prior_detail['image_hw']
        assert rgb.shape==(height,width,3) and rgb.dtype==np.uint8
        assert [p['model'] for p in detail['panels']]==['R0']
        panel=detail['panels'][0]
        historical_panel=next(p for p in prior_detail['panels'] if p['model']=='R0')
        pose=poses['R0'][fid]['GEO_pose'];hypothesis=poses['R0'][fid]['GEO_name']
        assert panel['model']==panel['parent']=='R0' and panel['hypothesis']==hypothesis
        assert panel['pose']==historical_panel['pose']==pose and pose['available']
        assert 'Historical operational R0 input illustration only' in panel['pose_source'] and 'not the current' in panel['pose_source']
        dimensions=np.asarray(row['xyz']);cf=np.asarray(pose['cf_extents'])
        assert np.allclose(cf,dimensions,rtol=0,atol=1e-9) or np.allclose(cf,dimensions[[2,1,0]],rtol=0,atol=1e-9)
        local,camera,uv=scalar_projection(pose,row['K'])
        for key,expected,tolerance in (('corners_cf',local,1e-12),('corners_camera',camera,1e-12),('projected_uv',uv,1e-8)):
            assert panel[key]==historical_panel[key]
            np.testing.assert_allclose(panel[key],expected,rtol=0,atol=tolerance)
        uv_difference=float(np.max(np.abs(uv-np.asarray(panel['projected_uv']))))
        camera_difference=float(np.max(np.abs(camera-np.asarray(panel['corners_camera']))))
        maximum_uv_difference=max(maximum_uv_difference,uv_difference)
        maximum_camera_difference=max(maximum_camera_difference,camera_difference)
        edges=[(i,j) for i,j in EDGES if min(camera[i,2],camera[j,2])>0]
        assert panel['drawn_edges']==historical_panel['drawn_edges']==[list(e) for e in edges]
        values=lookup[fid]
        assert panel['T_cm']==historical_panel['T_cm']==float(values['translation_cm'])
        assert panel['R_deg']==historical_panel['R_deg']==float(values['rotation_deg'])
        assert values['pose_available']=='True' and values['parent']=='R0' and values['hypothesis']==hypothesis
        results.append(dict(id=fid,recording=row['recording'],image=detail['image'],image_hw=[height,width],
            decoded_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),dimensions_m=row['xyz'],
            calibration_exact=True,historical_R0_full_pose_exact=True,hypothesis=hypothesis,
            drawn_edges=len(edges),UV_max_difference_px=uv_difference,
            historical_T_cm=panel['T_cm'],historical_R_deg=panel['R_deg'],current_learned_prediction=False))
        panel_count+=1
    assert panel_count==6
    jpgs=[b for b in report['artifacts'] if b['path'].endswith('.jpg')]
    assert len(jpgs)==3
    assert {Path(b['path']).name for b in jpgs}=={f'baseline_input_rgb_dimensions_{i}.jpg' for i in (1,2,3)}
    montages=[]
    for binding in jpgs:
        C.verify(binding)
        with Image.open(C.ROOT/binding['path']) as image:
            dimensions=image.size;image.verify()
        montages.append(dict(binding=binding,width=dimensions[0],height=dimensions[1]))
    return dict(complete=True,PASS=True,code=C.bind(Path(__file__)),actual_RGB_images=6,
        panels=6,historical_R0_panels=6,current_learned_prediction_panels=0,
        corner_projections=48,projected_coordinate_values=96,
        input_pixels_and_dimensions_verified=True,historical_frozen_pose_parity=True,
        maximum_UV_difference_px=maximum_uv_difference,UV_tolerance_px=1e-8,
        maximum_camera_difference_m=maximum_camera_difference,
        historical_panel_T_R_values_equal_to_previous_public_CSV=12,
        original_case_selection_equal=True,
        selection_rule='Same prior six IDs: maximum historical operational R0 T per natural recording; lexical ID tie. No current model output or performance exists.',
        prior_report=prior_binding,original_selection_report=selection_binding,gallery_selection=report['gallery'],
        historical_CSV=csv_binding,metadata=protocol['inputs']['metadata'],poses=protocol['inputs']['poses'],
        evaluation_not_run=C.bind(C.DOC/'REAL_EVALUATION_NOT_RUN.json'),frames=results,montages=montages,
        real_reference_reads=0,new_reference_metric_calculations=0,new_PnP_solves=0,image_forwards=0,
        new_learned_real_routes=0,method_success=False,goal_complete=False,
        limitation='Historical baseline/input illustrations only. Source RGB and projection metadata verified; final rendered legibility separately reviewed by root.')


def verify_source_choice_comparison(report,source):
    verified=bound_read(report['source_verification'])
    comparison=report['source_selection_comparison'];assert comparison==verified['source_selection_comparison']
    assert comparison['frames']==1024 and comparison['current_gate']==report['source_gate']
    assert comparison['previous_gate']==report['previous_source_gate']
    for key in ('new_policy_evaluations','new_argmin_computations','new_reference_error_computations'):assert comparison[key]==0
    oldgate=bound_read(comparison['previous_gate']);oldlock=bound_read(comparison['previous_routing_lock'])
    old=bound_read(comparison['previous_choices']);lock=C.read(C.DOC/'SOURCE_VAL_ROUTING_LOCK.json');now=bound_read(lock['choices'])
    assert oldlock['choices']==comparison['previous_choices'] and comparison['previous_metrics']==oldgate['metrics']
    for key in ('poses','metadata','feature_lock','source_predictions_lock','source_contract'):assert lock[key]==oldlock[key]
    assert now['ids']==old['ids'] and len(now['ids'])==1024
    with np.load(C.ROOT/source['metrics']['path'],allow_pickle=False) as z:current={m:z[m].copy() for m in z['models']}
    with np.load(C.ROOT/comparison['previous_metrics']['path'],allow_pickle=False) as z:previous={m:z[m].copy() for m in z['models']}
    for m in set(current)-set(C.MODEL_NAMES):assert current[m].tobytes()==previous[m].tobytes()
    anchor=current['R0_GEO'];scale=np.asarray(comparison['scale'],np.float64)
    prefit=bound_read(report['prefit_review']);np.testing.assert_array_equal(scale,prefit['inputs']['scale'])
    names=('anchor','safe_improvement','safe_equal','unsafe','failure');review={}
    for m in C.MODEL_NAMES:
        labels=[];records=[]
        for choice,values in ((old,previous[m]),(now,current[m])):
            ll=[];identities=[]
            for i,fid in enumerate(now['ids']):
                row=choice['records'][m][fid];identity=row['anchor_name'] if row['fallback'] else row['candidate_name'];identities.append(identity)
                if not np.isfinite(values[i]).all():label='failure'
                elif identity==row['anchor_name']:label='anchor';np.testing.assert_array_equal(values[i],anchor[i])
                elif any(values[i,a]>anchor[i,a] for a in (0,1)):label='unsafe'
                elif any(values[i,a]<anchor[i,a] for a in (0,1)):label='safe_improvement'
                else:label='safe_equal'
                ll.append(label)
            labels.append(np.array(ll));records.append(identities)
        cell=comparison['models'][m]
        for key,ll,values in (('previous',labels[0],previous[m]),('current',labels[1],current[m])):
            expected={name:int((ll==name).sum()) for name in names}
            assert cell[key]['classes']==expected and cell[key]['frames']==1024
            finite=np.isfinite(values).all(1);delta=values-anchor
            violation=dict(T=int((finite&(delta[:,0]>0)).sum()),R=int((finite&(delta[:,1]>0)).sum()),both=int((finite&(delta>0).all(1)).sum()))
            assert cell[key]['violations']==violation
            assert cell[key]['nonanchor_count']==int(sum(name not in ('anchor','failure') for name in ll))
            maximum=float(np.maximum(np.max(delta[finite]/scale,axis=1),0.).max()) if finite.any() else None
            assert cell[key]['max_scaled_positive_excess']==maximum
        changed=np.array(records[0])!=np.array(records[1]);assert cell['changed_identity_count']==int(changed.sum())
        np.testing.assert_array_equal(current[m][~changed],previous[m][~changed])
        expected={a+' -> '+b:int(((labels[0]==a)&(labels[1]==b)).sum()) for a in names for b in names}
        assert cell['class_transitions']==expected
        review[m]=dict(changed=int(changed.sum()),previous=cell['previous']['classes'],current=cell['current']['classes'],transition_cells=25)
    return dict(PASS=True,source_verification=report['source_verification'],models=review,
        frozen_choices_checked=4096,new_argmin_computations=0,new_reference_metric_calls=0)


def verify_tables(md,source,training,fits,report,real=None):
    expected=[]
    def add(header,rows):expected.append((header,rows))
    rows=[]
    for model in C.MODEL_NAMES:
        q=training['models'][model]['independent_recompute'];fit=fits[model]
        rows.append(f"| {model} | {fit['objective_calls']} | {fit['iterations']} | {q['objective']:.12f} | {q['Huber']:.12f} | {q['Huber_symmetric']:.12f} | {q['Huber_underprediction']:.12f} | {q['Sign_logistic']:.12f} | {q['L2_penalty']:.12f} | {q['gradient_linf']:.9g} | {q['certified_gap_upper_bound']:.9g} |")
    add('| 모델 | objective 호출 | 승인 반복 | 최종 J | Huber 합계 | 대칭 Huber | 과소예측 추가 Huber | Sign logistic | L2 penalty | gradient Linf | gap 상계 |',rows)
    rows=[]
    for model in C.MODEL_NAMES:
        q=report['same_new_objective_by_model'][model]
        rows.append(f"| {model} | {q['old_weight']:.12f} | {q['new_weight']:.12f} | {q['decrease']:.12f} |")
    add('| 모델 | 직전271 가중치의 새 비대칭 J | 현재271 가중치의 같은 J | 이전−현재 |',rows)
    rows=[]
    for model in C.MODEL_NAMES:
        q=training['models'][model]['regression_statistics']['nonanchor_valid_candidates'];t,r=q['axes']['T'],q['axes']['R']
        rows.append(f"| {model} | {t['MAE']:.6f} | {t['RMSE']:.6f} | {t['sign_accuracy']:.6f} | {r['MAE']:.6f} | {r['RMSE']:.6f} | {r['sign_accuracy']:.6f} | {q['candidate_pairs']} |")
    add('| 모델 | nonanchor T MAE | T RMSE | T 부호 정확도 | nonanchor R MAE | R RMSE | R 부호 정확도 | nonanchor 후보 수 |',rows)
    rows=[]
    for model in C.MODEL_NAMES:
        q=training['models'][model];old=q['previous_fixed_direction'];parts=[]
        for axis in ('T','R','either'):parts.append(f"{old['statistics']['anchor_violations'][axis]['count']}→{q['statistics']['anchor_violations'][axis]['count']}")
        for key in ('safe_improvement','anchor'):parts.append(f"{old['risk_statistics']['classes'][key]['count']}→{q['risk_statistics']['classes'][key]['count']}")
        parts.append(f"{old['risk_statistics']['normalized_max_excess_all_available']['maximum']:.6f}→{q['risk_statistics']['normalized_max_excess_all_available']['maximum']:.6f}")
        rows.append('| '+model+' | '+' | '.join(parts)+' |')
    add('| 모델 | T 위반 이전→현재 | R 위반 이전→현재 | unsafe 이전→현재 | 안전 개선 이전→현재 | anchor 이전→현재 | 최대 정규화 초과 이전→현재 |',rows)
    metric_header='| 모델 | T 중앙값 cm | R 중앙값 ° | T P90 cm | R P90 ° | 실패 |'
    def metric_table(summaries,models):
        rows=[]
        for model in models:
            summary=summaries[model];full=summary['full_population'];parts=[]
            for axis,q in [('translation_cm','median'),('rotation_deg','median'),('translation_cm','P90'),('rotation_deg','P90')]:
                value=full[axis][q];parts.append('inf (failure retained)' if value is None or np.isposinf(value) else f'{value:.6f}')
            rows.append('| '+model+' | '+' | '.join(parts)+f" | {summary['failed_pose']} |")
        add(metric_header,rows)
    metric_table(source['summaries'],source['summaries'])
    rows=[]
    for model in C.MODEL_NAMES:
        q=report['source_selection_comparison']['models'][model];old,new=q['previous'],q['current']
        cells=[str(q['changed_identity_count'])]+[f"{old['classes'][k]}→{new['classes'][k]}" for k in ('safe_improvement','unsafe','anchor')]
        cells.append(f"{old['max_scaled_positive_excess']:.6f}→{new['max_scaled_positive_excess']:.6f}")
        rows.append('| '+model+' | '+' | '.join(cells)+' |')
    add('| 모델 | 선택 변경 | safe 이전→현재 | unsafe 이전→현재 | anchor 이전→현재 | 최대 정규화 초과 이전→현재 |',rows)
    if real is not None:
        rows=[]
        for key,gate in real['stability']['gates'].items():
            flags=[real['matched_intervention_stability']['gates'][key]['PASS'],real['original_goal_stability']['gates'][key]['PASS'],gate['PASS']]
            rows.append('| '+key+' | '+' | '.join('PASS' if flag else 'FAIL' for flag in flags)+' |')
        add('| 안정성 범주 | matched | 원래 goal | 필수 AND |',rows)
        for pop in ('NATURAL99','CLEAN29','WOOD45'):metric_table(real['summaries'][pop],real['models'])
    actual=[];lines=md.splitlines()
    for i in range(len(lines)-1):
        if lines[i].startswith('|') and lines[i+1].startswith('|---'):
            rows=[];j=i+2
            while j<len(lines) and lines[j].startswith('|'):rows.append(lines[j]);j+=1
            actual.append((lines[i],rows))
    assert actual==expected,('MARKDOWN_TABLE_MISMATCH',actual,expected)
    return [dict(header=h,rows=len(rows)) for h,rows in expected]


def selfcheck():
    gallery_selfcheck()
    assert scalar_equal('',None) and scalar_equal('False',False)
    assert scalar_equal('1.25',1.25) and not scalar_equal('1.250000001',1.25)
    assert scalar_equal(str({'a':1}),{'a':1})
    h,s,p=.2,.3,.004
    np.testing.assert_allclose(h+s+p,sum((h,s,p)),rtol=0,atol=0)
    assert not np.isclose(h+s+p,h+p,rtol=1e-14,atol=1e-14)
    assert len(HASH_KEYS)==9 and len(REAL_MODELS)==13
    assert scalar_equal('inf',np.inf) and scalar_equal('',None)
    assert not scalar_equal('1',1.00001)
    # Exercise every publication table in both branches with invented inputs.
    from . import report as R
    transformed=dict(MAE=.1,RMSE=.2,sign_accuracy=.7)
    objective=dict(objective=.6,Huber=.2,Huber_symmetric=.15,Huber_underprediction=.05,Sign_logistic=.3,L2_penalty=.1,gradient_linf=1e-10,certified_gap_upper_bound=1e-8)
    stats=dict(anchor_violations={k:dict(count=j) for j,k in enumerate(('T','R','either'))})
    risk=dict(classes={k:dict(count=4) for k in ('safe_improvement','anchor')},normalized_max_excess_all_available=dict(maximum=2.))
    training=dict(models={m:dict(independent_recompute=objective,
        regression_statistics=dict(nonanchor_valid_candidates=dict(axes=dict(T=transformed,R=transformed),candidate_pairs=5)),
        statistics=stats,risk_statistics=risk,previous_fixed_direction=dict(statistics=stats,risk_statistics=risk)) for m in C.MODEL_NAMES})
    fits={m:dict(objective_calls=3,iterations=1) for m in C.MODEL_NAMES}
    summary=dict(failed_pose=0,full_population={axis:dict(median=1.,P90=2.) for axis in ('translation_cm','rotation_deg')})
    source=dict(checks_passed=44,checks_total=45,summaries={m:summary for m in (*C.MODEL_NAMES,'R0_GEO','DIVERSE251_s1_GEO','DIVERSE251_s2_GEO','DIVERSE251_s3_GEO')},failed_checks=['UNION_s3/R0_ONLY/translation_cm_median_strict'])
    previous=dict(checks_passed=43,checks_total=45)
    report=dict(stable_joint_improvement_achieved=False,same_new_objective_by_model={m:dict(old_weight=.7,new_weight=.6,decrease=.1) for m in C.MODEL_NAMES})
    selection=dict(classes={k:3 for k in ('safe_improvement','unsafe','anchor')},max_scaled_positive_excess=1.)
    report['source_selection_comparison']=dict(models={m:dict(previous=selection,current=selection,changed_identity_count=0) for m in C.MODEL_NAMES})
    gates={f'g{k}':dict(PASS=False) for k in range(5)}
    real=dict(stability=dict(gates=gates),matched_intervention_stability=dict(gates=gates),original_goal_stability=dict(gates=gates),
        models=list(REAL_MODELS),summaries={pop:{m:summary for m in REAL_MODELS} for pop in ('NATURAL99','CLEAN29','WOOD45')})
    for result,total in ((None,28),(real,72)):
        body=R.render_report(source,previous,training,fits,[0]*12,[0]*4,result,report)
        verified=verify_tables(body,source,training,fits,report,result)
        assert sum(item['rows'] for item in verified)==total
        broken=body.replace('| R0_ONLY | 3 | 1 |','| R0_ONLY | 4 | 1 |',1)
        try:verify_tables(broken,source,training,fits,report,result)
        except AssertionError:pass
        else:raise AssertionError('ALTERED_PUBLIC_NUMBER_MUST_FAIL')
    print('PUBLICATION_VERIFIER_SELFCHECK_PASS invented_only=1 artifact_reads=0')


def verify_real_exports(report,source):
    """Verify cached metric/CSV/gate provenance only; never read raw GT."""
    assert source['PASS'] and source['checks_passed']==45 and source['real_routing_authorized']
    assert not (C.DOC/'REAL_EVALUATION_NOT_RUN.json').exists()
    result=bound_read(report['real_results']);audit=bound_read(report['real_verification'])
    assert report['real_results']==C.bind(C.DOC/'REAL_RESULTS.json')
    assert report['real_verification']==C.bind(C.DOC/'REAL_VERIFICATION.json')
    assert result['complete'] and audit['complete'] and audit['PASS']
    all_bindings(result)
    C.verify(C.read(C.DOC/'REAL_PROTOCOL_SHA.json'))
    protocol=C.read(C.DOC/'REAL_PROTOCOL.json');all_bindings(protocol)
    lock=C.read(C.DOC/'REAL_ROUTING_LOCK.json');all_bindings(lock)
    assert result['protocol']==C.bind(C.DOC/'REAL_PROTOCOL.json')==lock['protocol']
    assert result['routing_lock']==C.bind(C.DOC/'REAL_ROUTING_LOCK.json')
    assert lock['source_val_gate']==C.bind(C.DOC/'SOURCE_VAL_GATE.json')
    assert lock['whole_pose_selection'] and not lock['real_reference_values_read']
    assert not lock['runtime_uses_margin'] and not lock['runtime_safe_mask']
    assert lock['new_PnP_solves']==lock['image_forwards']==lock['fits_executed']==0
    assert result['models']==list(REAL_MODELS) and result['full_frame_rows']==2249
    assert result['baseline_metric_parity_checks']==1557
    assert result['real_stability_contract']=='matched_intervention_AND_original_SINGLE251_stability'
    assert report['learned_real_evaluated'] and report['current_real_TR_effect_measured']
    assert report['real_not_run'] is None and report['baseline_panels']==6 and report['current_learned_real_panels']==18
    assert report['method_success']==report['stable_joint_improvement_achieved']==result['stability']['PASS']
    assert report['status']==('REAL_STABILITY_PASS' if result['stability']['PASS'] else 'REAL_STABILITY_FAIL')
    for family in ('matched_intervention_stability','original_goal_stability','stability'):
        value=result[family]
        assert len(value['gates'])==5
        assert value['PASS']==all(g['PASS'] for g in value['gates'].values())
    matched=result['matched_intervention_stability'];original=result['original_goal_stability']
    assert set(matched['gates'])==set(original['gates'])==set(result['stability']['gates'])
    for key,gate in result['stability']['gates'].items():
        assert gate['PASS']==bool(matched['gates'][key]['PASS'] and original['gates'][key]['PASS'])
        assert gate['matched_intervention']==matched['gates'][key] and gate['original_goal']==original['gates'][key]
    assert original['public_baseline_metrics_preserved']
    assert original['evaluation_slot_aliases']=={f'DIVERSE251_s{s}':f'UNION_s{s}' for s in (1,2,3)}
    rows=bound_read(protocol['inputs']['metadata']);groups=bound_read(protocol['inputs']['groups'])
    poses=bound_read(protocol['inputs']['poses']);choices=bound_read(lock['choices'])
    ids=[r['id'] for r in rows];metadata={r['id']:r for r in rows}
    assert len(ids)==len(set(ids))==173 and choices['ids']==ids
    assert choices['models']==list(C.MODEL_NAMES)
    assert sum(len(v) for v in choices['records'].values())==692
    assert {k:len(groups[k]) for k in ('NATURAL99','CLEAN29','WOOD45')}==dict(NATURAL99=99,CLEAN29=29,WOOD45=45)
    mb=next(b for b in result['artifacts'] if b['path']==str((C.RAW/'POSE_METRICS.json').relative_to(C.ROOT)))
    metrics=bound_read(mb);assert set(metrics)==set(REAL_MODELS)
    actual=csv_rows(C.DOC/'REAL_FRAME_RESULTS.csv');lookup={(r['model'],r['id']):r for r in actual}
    assert len(actual)==len(lookup)==2249
    columns={'model','id','recording','severity','material','parent','hypothesis','fallback','pose_available',
        'translation_cm','rotation_deg','full_population_error_status'}
    for model in REAL_MODELS:
        assert set(metrics[model])==set(ids)
        for fid in ids:
            row=lookup[model,fid];meta=metadata[fid];metric=metrics[model][fid]
            choice=choices['records'][model][fid] if model in C.MODEL_NAMES else None
            expected=dict(model=model,id=fid,recording=meta['recording'],severity=meta['severity'],
                material='PLASTIC' if fid in groups['FULL128'] else 'WOOD',
                parent=choice['parent'] if choice else model,
                hypothesis=choice['hypothesis'] if choice else poses[model][fid]['GEO_name'],
                fallback=choice['fallback'] if choice else poses[model][fid]['GEO_fallback'],
                pose_available=metric['available'],translation_cm=metric.get('translation_cm'),rotation_deg=metric.get('rotation_deg'),
                full_population_error_status='FINITE' if metric['available'] else 'POSITIVE_INFINITY')
            assert set(row)==columns
            for key,value in expected.items():assert scalar_equal(row[key],value),(model,fid,key)
    return result,dict(complete=True,PASS=True,models=13,frames=173,full_CSV_rows=2249,
        all_CSV_cells_checked=True,learned_routes=692,baseline_records=1557,
        combined_gate_categories=5,original_AND_matched_exact=True,
        current_real_reference_metrics_independently_verified_by=report['real_verification'],
        physical_metric_recomputation_by_publication_audit=0)


def verify_learned_gallery(report,result):
    """Direct projection of frozen whole poses, never new reference metrics."""
    protocol=C.read(C.DOC/'REAL_PROTOCOL.json');lock=C.read(C.DOC/'REAL_ROUTING_LOCK.json')
    metadata=bound_read(protocol['inputs']['metadata']);groups=bound_read(protocol['inputs']['groups'])
    poses=bound_read(protocol['inputs']['poses']);choices=bound_read(lock['choices'])
    rows={r['id']:r for r in metadata};assert len(rows)==173
    table=csv_rows(C.DOC/'REAL_FRAME_RESULTS.csv');lookup={(r['model'],r['id']):r for r in table}
    prior_binding=published_binding(C.ANCHOR_DOC,C.ANCHOR_DOC/'REPORT_DATA.json')
    prior=bound_read(prior_binding);historical=prior['illustrations'];assert len(historical)==6
    fixed=[r['id'] for r in historical];details=report['illustrations']
    assert len(details)==len(set(fixed))==6 and [r['id'] for r in details]==fixed
    natural=groups['NATURAL99'];recordings=sorted({rows[f]['recording'] for f in natural})
    expected=[min((f for f in natural if rows[f]['recording']==rec),key=lambda f:(-float(lookup['R0',f]['translation_cm']),f)) for rec in recordings]
    assert expected==fixed
    assert report['actual_RGB_images']==6 and report['current_learned_real_panels']==18
    selection=bound_read(report['gallery'])
    assert selection['complete'] and selection['current_learned_real_evaluated']
    assert selection['illustrations']==details and selection['source_selection_report']==prior_binding
    assert selection['routing']==C.bind(C.DOC/'REAL_ROUTING_LOCK.json')
    assert selection['metadata']==protocol['inputs']['metadata'] and selection['poses']==protocol['inputs']['poses']
    assert selection['metrics'] in result['artifacts'] and selection['panels']==24
    assert selection['new_metric_calls']==selection['new_image_forwards']==selection['new_PnP_solves']==0
    results=[];maximum=0.;count=0
    for item,old in zip(details,historical):
        fid=item['id'];row=rows[fid]
        for key,expected in [('recording',row['recording']),('image',row['image']),('K',row['K']),('dimensions_m',row['xyz']),('image_hw',row['hw'])]:assert item[key]==expected
        assert item['image']==old['image'] and item['dimensions_m']==old['dimensions_m']
        assert item['corner_signs']==[list(s) for s in SIGNS] and item['edges']==[list(e) for e in EDGES]
        C.verify(item['image'])
        with Image.open(C.ROOT/item['image']['path']) as image:rgb=np.asarray(image.convert('RGB'))
        assert list(rgb.shape[:2])==row['hw'] and rgb.shape[2]==3 and rgb.dtype==np.uint8
        assert [p['model'] for p in item['panels']]==['R0','UNION_s1','UNION_s2','UNION_s3']
        checked=[]
        for panel in item['panels']:
            model=panel['model'];choice=choices['records'][model][fid] if model!='R0' else None
            pose=choice['pose'] if choice else poses['R0'][fid]['GEO_pose']
            parent=choice['parent'] if choice else 'R0';hyp=choice['hypothesis'] if choice else poses['R0'][fid]['GEO_name']
            assert panel['pose']==pose and panel['parent']==parent and panel['hypothesis']==hyp and pose['available']
            if choice:
                original=poses['R0'][fid]['GEO_pose'] if choice['fallback'] else next(h['pose'] for h in poses[parent][fid]['hypotheses'] if h['name']==hyp)
                assert original==pose
            dims=np.asarray(row['xyz']);cf=np.asarray(pose['cf_extents'])
            assert np.allclose(cf,dims,rtol=0,atol=1e-9) or np.allclose(cf,dims[[2,1,0]],rtol=0,atol=1e-9)
            local,camera,uv=scalar_projection(pose,row['K'])
            for key,value,tolerance in [('corners_cf',local,1e-12),('corners_camera',camera,1e-12),('projected_uv',uv,1e-8)]:
                np.testing.assert_allclose(panel[key],value,rtol=0,atol=tolerance)
            gap=float(np.abs(uv-np.asarray(panel['projected_uv'])).max());maximum=max(maximum,gap)
            edges=[list(e) for e in EDGES if min(camera[e[0],2],camera[e[1],2])>0]
            assert panel['drawn_edges']==edges
            raw=lookup[model,fid]
            assert raw['pose_available']=='True' and raw['parent']==parent and raw['hypothesis']==hyp
            assert panel['T_cm']==float(raw['translation_cm']) and panel['R_deg']==float(raw['rotation_deg'])
            checked.append(dict(model=model,parent=parent,hypothesis=hyp,full_pose_exact=True,UV_max_difference_px=gap))
            count+=1
        results.append(dict(id=fid,image=item['image'],dimensions_m=row['xyz'],image_hw=row['hw'],
            decoded_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),panels=checked))
    assert count==24 and any(np.array_equal(np.asarray(r['dimensions_m']),[1.1,.11,1.3]) for r in results)
    jpgs=[b for b in report['artifacts'] if b['path'].endswith('.jpg')]
    assert {Path(b['path']).name for b in jpgs}=={f'learned_rgb_dimensions_{j}.jpg' for j in (1,2,3)}
    return dict(complete=True,PASS=True,actual_RGB_images=6,panels=24,historical_R0_panels=6,
        current_learned_prediction_panels=18,projected_coordinate_values=384,panel_metric_values_checked=48,
        fixed_previous_cases=True,original_case_selection_equal=True,all_three_seeds_no_selection=True,
        maximum_UV_difference_px=maximum,UV_tolerance_px=1e-8,
        dimensions_110_11_130_cm_exact=True,frames=results,new_reference_metric_calculations=0,
        new_PnP_solves=0,image_forwards=0)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',nargs='?',default='verify',choices=('selfcheck','verify'))
    parser.add_argument('--visual-reviewed',action='store_true');parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    if args.action=='selfcheck':selfcheck();return
    assert args.visual_reviewed,'Root must inspect actual images before publication attestation.'
    sys.dont_write_bytecode=True;install_guard()
    assert not (C.DOC/'PUBLIC_REVIEW.json').exists() and not (C.DOC/'PUBLIC_REVIEW_KO.md').exists(),'Final receipt is immutable.'
    report=C.read(C.DOC/'REPORT_DATA.json');all_bindings(report)
    md=(C.DOC/'REPORT_KO.md').read_text()
    assert (C.DOC/'PUBLIC_REPORT_KO.md').read_bytes()==(C.DOC/'REPORT_KO.md').read_bytes()
    for name in ('PREFIT_REVIEW.json','TRAIN_CONVERGENCE.json','SOURCE_VAL_VERIFICATION.json'):
        audit=C.read(C.DOC/name);assert audit['complete'] and audit['PASS']
    training=C.read(C.DOC/'TRAIN_CONVERGENCE.json');complete=C.read(C.DOC/'TRAINING_COMPLETE.json');source=C.read(C.DOC/'SOURCE_VAL_GATE.json')
    source_review=verify_source_exports(source)
    fits,logs,trace_counts=verify_training_exports(report,complete)
    comparisons=check_metadata(report,training,source,complete,fits)
    if source['PASS']:
        real,real_review=verify_real_exports(report,source);not_run=None
        gallery=verify_learned_gallery(report,real)
    else:
        not_run=verify_not_run(report,source);real=real_review=None
        assert '미측정' in md and ('미실행' in md or '미평가' in md)
        gallery=verify_gallery(report)
        assert any(np.array_equal(np.asarray(r['dimensions_m']),[1.1,.11,1.3]) for r in gallery['frames'])
        gallery['dimensions_110_11_130_cm_exact']=True
    assert f"{source['checks_passed']}/{source['checks_total']}" in md
    for key in source['failed_checks']:assert '- `'+key+'`' in md
    assert '110 × 11 × 130 cm' in md or real is not None
    assert '비교군으로 바뀌므로' in md and '실제 T/R 비악화 보장이 아닙니다' in md
    rejected=sum(r['event']=='objective' and r['phase']=='trial' and not r['armijo_accepted'] for r in logs)
    assert report['rejected_Armijo_trials']==rejected==complete['total_objective_calls']-complete['total_iterations']-4
    figure_count=verify_figure_values(report,source,training,logs,fits,real)
    tables=verify_tables(md,source,training,fits,report,real)
    diagnostic=verify_source_choice_comparison(report,source)
    expected_images={'training_huber_and_certificate.png','source_val_method_comparison.png','train_axis_regression_and_violations.png'}
    prefix='learned_rgb_dimensions' if real is not None else 'baseline_input_rgb_dimensions'
    expected_images|={f'{prefix}_{j}.jpg' for j in (1,2,3)}
    if real is not None:expected_images|={'real_all_models.png','real_stability_gates.png'}
    paths=sorted((C.DOC/'figures').iterdir());assert {p.name for p in paths}==expected_images
    assert {b['path'] for b in report['artifacts']}=={str(p.relative_to(C.ROOT)) for p in paths}
    images=[]
    for path in paths:
        with Image.open(path) as image:width,height=image.size;image.verify()
        assert width>500 and height>300 and 'figures/'+path.name in md
        images.append(dict(binding=C.bind(path),width=width,height=height))
    links,files=check_links_and_files()
    stable=bool(real is not None and real['stability']['PASS'])
    review=dict(complete=True,PASS=True,created_at=C.now(),code=C.bind(__file__),report_code=report['code'],
        report=C.bind(C.DOC/'REPORT_KO.md'),report_data=C.bind(C.DOC/'REPORT_DATA.json'),
        source_gate_PASS=source['PASS'],source_checks_passed=source['checks_passed'],source_checks_total=45,
        source_export=source_review,trace_counts=trace_counts,rejected_Armijo_trials=rejected,
        certified_models=4,parameter_count_per_model=542,parameter_count=2168,feature_dim=271,base_feature_dim=253,direction_dim=18,
        nine_TRAIN_hashes_verified=list(HASH_KEYS),direction_normalization_sha=report['direction_normalization_sha'],
        direction_receipt=report['direction_receipt'],basis=report['basis'],sign_coefficient=1.,
        objective_components_checked=['Huber_symmetric','Huber_underprediction','Huber','Sign_logistic','L2_penalty'],
        underprediction_coefficient=1.,underprediction_cost=2.,overprediction_cost=1.,underprediction_zero_curvature=0.,
        previous_P271_weights_under_same_new_asymmetric_objective=comparisons,original_symmetric_certificate_kept_separate=True,
        exact_figure_value_groups=figure_count,verified_tables=tables,verified_numeric_rows=sum(t['rows'] for t in tables),
        figures=images,gallery=gallery,visual_review_completed=True,markdown_links_checked=links,
        learned_real_evaluated=real is not None,current_method_real_T_R_effect_measured=real is not None,
        method_success=stable,stable_joint_improvement_achieved=stable,goal_complete=False,
        real_export=real_review,fixed_source_diagnostic=diagnostic,
        evaluation_not_run=C.bind(C.DOC/'REAL_EVALUATION_NOT_RUN.json') if not_run is not None else None,
        reviewed_artifacts=[C.bind(p) for p in sorted(set(files))],new_fits_by_audit=0,new_reference_metric_calls=0,
        pending_publication_manifest='PUBLICATION_MANIFEST.json alone is intentionally generated after this review; PUBLIC_REVIEW files are this receipt.')
    if args.dry_run:
        print('PUBLIC_REVIEW_DRY_RUN_PASS',json.dumps(dict(files=len(set(files)),figures=len(images),links=links,trace=trace_counts,numeric_rows=review['verified_numeric_rows']),ensure_ascii=False));return
    C.save(C.DOC/'PUBLIC_REVIEW.json',review)
    state='실사 원래·matched 조건의 AND를 확인했습니다.' if real is not None else 'source 실패에 따른 실사12개 산출물 부재를 확인했습니다.'
    C.save(C.DOC/'PUBLIC_REVIEW_KO.md','# 공개 결과 검산\n\n공개 기록 검산 PASS입니다. 수렴 인증·source '+str(source['checks_passed'])+'/45·실사 상태를 구분했습니다. '+state+'\n\n'
        '- 공개 최종 파라미터4개는 원본과 byte 동일하며, 각542계수·기존253/추가18 입력·9개 TRAIN 해시와 고정 정규화/basis 연결을 검증했습니다.\n'
        f"- objective {trace_counts['objective']}행·승인 반복 {trace_counts['iteration']}행·source8192행과45조건의 모든 CSV셀을 원본에 대조했습니다.\n"
        '- 이전 P271과 현재 Q271을 같은 새 비대칭 목적식에서 비교했고, 이전 대칭 손실의 자체 gradient 인증은 별도 검산했습니다. 두 Huber 성분과 고정2:1 비용의 provenance를 확인했습니다.\n'
        f"- 숫자표 {len(tables)}개/{review['verified_numeric_rows']}행, 그래프 수치 {figure_count}묶음, 실제 RGB6장의 SHA·크기·치수·K·고정pose·투영과 표시 T/R를 검증했습니다. 시각 검토는 root가 완료했습니다.\n"
        '- 상대 링크와 공개 파일을 검증했습니다. PUBLICATION_MANIFEST는 이 기록 뒤 생성됩니다. 이 공개 검산은 방법의 목표 달성 판정이 아닙니다.\n\n[검산 JSON](PUBLIC_REVIEW.json) · [상세 보고서](REPORT_KO.md)\n')
    print('PUBLIC_REVIEW_PASS',len(set(files)),len(images),links,C.bind(C.DOC/'PUBLIC_REVIEW.json'))


if __name__=='__main__':main()
