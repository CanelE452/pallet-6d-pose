"""Independently verify sign-loss publication against frozen artifacts; no fitting or GT metrics."""
import argparse
import ast
import csv
import hashlib
import itertools
import json
from pathlib import Path
import re
from urllib.parse import unquote
import numpy as np
from PIL import Image
from . import common as C


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
    assert report['previous_source_gate']==C.bind(C.NEWTON_DOC/'SOURCE_VAL_GATE.json')
    assert report['previous_training_verification']==C.bind(C.NEWTON_DOC/'TRAIN_CONVERGENCE.json')
    comparisons={}
    for model in C.MODEL_NAMES:
        r=training['models'][model];q=r['independent_recompute'];old=r['previous_fixed_newton']
        assert old['original_choice_replay_PASS']
        assert old['checkpoint']['path'].startswith(str(C.NEWTON_RAW.relative_to(C.ROOT)))
        assert old['original_Huber_only_certificate']['loss_rule']=='FULL_FRAME_VALID_CANDIDATE_TWO_AXIS_HUBER'
        assert 'Sign_logistic' in old['same_new_objective']
        assert q['Sign_logistic']>=0 and q['Huber']>=0 and q['L2_penalty']>=0
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
    for path,absent in stopped['absence_checks'].items():assert absent and not (C.ROOT/path).exists(),path
    assert report['status']=='CERTIFIED_TRAIN_SOURCE_GATE_FAILED'
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
    index={(r['model'],r['id']):r for r in rows}
    assert len(rows)==len(index)==8192
    with np.load(C.ROOT/source['metrics']['path'],allow_pickle=False) as z:
        for model in z['models'].tolist():
            for i,fid in enumerate(z['ids'].tolist()):
                row=index[model,fid]
                assert float(row['T_cm'])==z[model][i,0] and float(row['R_deg'])==z[model][i,1]
                assert row['available']==str(bool(np.isfinite(z[model][i]).all())) and row['split']=='VAL'
                pick=choices['records'].get(model,{}).get(fid,{})
                assert row['candidate']==pick.get('candidate_name','fixed_GEO')
                assert row['fallback']==str(pick.get('fallback',False))
    checks=csv_rows(C.DOC/'SOURCE_VAL_CHECKS.csv')
    expected=[dict(model=m,baseline=b,criterion=k,PASS=v) for m,g in source['comparisons'].items()
        for b,comp in g.items() for k,v in comp['checks'].items()]
    assert len(checks)==len(expected)==45
    for row,raw in zip(checks,expected):assert row=={k:str(v) for k,v in raw.items()}
    return dict(frame_rows=8192,checks=45,passed_checks=source['checks_passed'],failed_checks=45-source['checks_passed'])


def verify_training_exports(report,complete):
    fits={};logs=[];exports={}
    for model in C.MODEL_NAMES:
        fit=C.read(C.DOC/f'FIT_{model}.json');all_bindings(fit)
        ck=C.read(C.ROOT/fit['checkpoint']['path']);assert ck['certificate']['PASS']
        assert ck['schema']=='pallet_pose_signed_axes_sign_linear253x2_v1'
        assert np.asarray(ck['weight']).shape==(253,2) and ck['solver_rule']=='BLOCK_GENERALIZED_NEWTON_ARMIJO'
        for key,value in dict(loss_rule='FULL_FRAME_VALID_CANDIDATE_TWO_AXIS_HUBER_PLUS_SIGN_LOGISTIC',
                sign_rule='NONZERO_SIGNED_TARGET_AXES',sign_coefficient=1.).items():
            assert ck[key]==fit[key]==complete[key]==value
            assert ck['certificate'][key]==value
        assert ck['certificate']==fit['certificate']
        cert=ck['certificate']
        assert cert['gradient_linf']<=1e-8 and cert['gradient_l2_squared_over_2lambda']<=1e-6
        np.testing.assert_allclose(cert['objective_value'],cert['Huber']+cert['Sign_logistic']+cert['L2_penalty'],rtol=1e-14,atol=1e-14)
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
        for row in raw:
            assert row['sign_rule']=='NONZERO_SIGNED_TARGET_AXES' and row['sign_coefficient']==1.
            assert row['loss_rule']=='FULL_FRAME_VALID_CANDIDATE_TWO_AXIS_HUBER_PLUS_SIGN_LOGISTIC'
            if row.get('evaluated',True):
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
    prior=C.read(C.NEWTON_DOC/'SOURCE_VAL_GATE.json');values={}
    for model in C.MODEL_NAMES:
        rows=[r for r in logs if r['model']==model and r['event']=='objective']
        values[f'train_{model}']=dict(calls=[r['call'] for r in rows],objective=[r['objective'] for r in rows],
            Huber=[r['Huber'] for r in rows],Sign_logistic=[r['Sign_logistic'] for r in rows],
            L2_penalty=[r['L2_penalty'] for r in rows],
            previous_weight_same_new_J=training['models'][model]['previous_fixed_newton']['same_new_objective']['objective'],
            gap_ratio_to_limit=[r['gradient_gap_upper_bound']/1e-6 for r in rows],
            linf_ratio_to_limit=[r['gradient_linf']/1e-8 for r in rows],armijo_accepted=[r['armijo_accepted'] for r in rows],
            final_accepted_call=fits[model]['final_accepted_call'])
    for axis in ('translation_cm','rotation_deg'):
        for q in ('median','P90'):
            values[f'source_{axis}_{q}']=dict(
                previous_newton_huber=[prior['summaries'][m]['full_population'][axis][q] for m in C.MODEL_NAMES],
                current_huber_sign=[source['summaries'][m]['full_population'][axis][q] for m in C.MODEL_NAMES],
                fixed_R0_GEO=source['summaries']['R0_GEO']['full_population'][axis][q])
    for axis in ('T','R'):
        values[f'train_nonanchor_{axis}']={metric:[training['models'][m]['regression_statistics']['nonanchor_valid_candidates']['axes'][axis][metric] for m in C.MODEL_NAMES] for metric in ('MAE','RMSE')}
    values['train_unsafe_count']=dict(
        previous_newton_huber=[training['models'][m]['previous_fixed_newton']['statistics']['anchor_violations']['either']['count'] for m in C.MODEL_NAMES],
        current_huber_sign=[training['models'][m]['statistics']['anchor_violations']['either']['count'] for m in C.MODEL_NAMES])
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
    prior_path=C.RBF_DOC/'REPORT_DATA.json'
    prior_binding=published_binding(C.RBF_DOC,prior_path)
    prior=bound_read(prior_binding)
    selected_path=C.ANCHOR_DOC/'REPORT_DATA.json'
    selection_binding=published_binding(C.ANCHOR_DOC,selected_path)
    original_selection=bound_read(selection_binding)
    historical=prior['illustrations']; original=original_selection['illustrations']
    assert len(historical)==len(original)==6
    fixed_ids=[r['id'] for r in original]
    assert [r['id'] for r in historical]==fixed_ids
    prior_protocol_binding=published_binding(C.RBF_DOC,C.RBF_DOC/'REAL_PROTOCOL.json')
    assert prior['real_protocol']==prior_protocol_binding
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


def verify_optional_diagnostic(report,source):
    assert 'source_transfer_diagnostic' in report
    diagnostic=bound_read(report['source_transfer_diagnostic']);all_bindings(diagnostic)
    assert diagnostic['complete'] and diagnostic['PASS']
    assert not diagnostic['method_success'] and not diagnostic['goal_complete']
    assert diagnostic['known_candidate_errors_are_partial'] and diagnostic['no_new_selector_policy_evaluated']
    assert diagnostic['new_intervention_proposed'] is False
    assert diagnostic['source_labels_recomputed_from_raw'] is False
    for key in ('new_argmin_computations','new_fits','raw_GT_reads','real_reference_reads','new_real_routes','threshold_sweeps'):
        assert diagnostic[key]==0,key
    counts={}
    with np.load(C.ROOT/source['metrics']['path'],allow_pickle=False) as z:
        anchor=z['R0_GEO']
        assert np.isfinite(anchor).all()
        for model in C.MODEL_NAMES:
            assert np.isfinite(z[model]).all()
            delta=z[model]-anchor
            safe=int(((delta<=0).all(1)&(delta<0).any(1)).sum())
            unsafe=int((delta>0).any(1).sum())
            actual=diagnostic['models'][model]['actual_source_selection']['classes']
            assert actual['safe_improvement']==safe and actual['unsafe']==unsafe,model
            counts[model]=dict(safe_improvement=safe,unsafe=unsafe)
    return dict(binding=report['source_transfer_diagnostic'],independently_checked_source_selection_counts=counts)


def verify_tables(md,source,training,fits,report):
    lines=md.splitlines();counts={}
    def table(header,expected):
        assert lines.count(header)==1,header
        i=lines.index(header)+2;actual=[]
        while i<len(lines) and lines[i].startswith('|'):
            actual.append(lines[i]);i+=1
        assert actual==expected,(header,actual,expected)
        counts[header]=len(expected)
    rows=[]
    for model in C.MODEL_NAMES:
        q=training['models'][model]['independent_recompute'];fit=fits[model]
        rows.append(f"| {model} | {fit['objective_calls']} | {fit['iterations']} | {q['Huber']:.9f} | {q['Sign_logistic']:.9f} | {q['L2_penalty']:.9f} | {q['objective']:.9f} | {q['gradient_linf']:.3e} | {q['certified_gap_upper_bound']:.3e} |")
    table('| 모델 | 호출 | 승인 반복 | Huber | Sign logistic | L2 | 새 J | gradient Linf | gap 상한 |',rows)
    rows=[]
    for model in C.MODEL_NAMES:
        r=training['models'][model];old=r['previous_fixed_newton']['same_new_objective']['objective'];current=r['independent_recompute']['objective']
        rows.append(f"| {model} | {old:.9f} | {current:.9f} | {old-current:.9f} |")
    table('| 모델 | 이전 Newton 가중치의 같은 새 J | 현재 가중치의 새 J | 감소 |',rows)
    rows=[]
    for model in C.MODEL_NAMES:
        r=training['models'][model]['regression_statistics']['nonanchor_valid_candidates'];t,rot=r['axes']['T'],r['axes']['R']
        rows.append(f"| {model} | {r['candidate_pairs']} | {t['MAE']:.6f} | {t['RMSE']:.6f} | {100*t['sign_accuracy']:.3f}% | {rot['MAE']:.6f} | {rot['RMSE']:.6f} | {100*rot['sign_accuracy']:.3f}% |")
    table('| 모델 | 비-anchor 후보 수 | T MAE | T RMSE | T 부호 정확도 | R MAE | R RMSE | R 부호 정확도 |',rows)
    rows=[]
    for model in C.MODEL_NAMES:
        current=training['models'][model];old=current['previous_fixed_newton'];parts=[]
        for axis in ('T','R','either'):parts.append(f"{old['statistics']['anchor_violations'][axis]['count']} → {current['statistics']['anchor_violations'][axis]['count']}")
        for key in ('safe_improvement','anchor'):parts.append(f"{old['risk_statistics']['classes'][key]['count']} → {current['risk_statistics']['classes'][key]['count']}")
        rows.append('| '+model+' | '+' | '.join(parts)+' |')
    table('| 모델 | T 위반 이전→현재 | R 위반 이전→현재 | 한 축 이상 위반 이전→현재 | 안전 개선 이전→현재 | anchor 선택 이전→현재 |',rows)
    rows=[]
    for model,summary in source['summaries'].items():
        full=summary['full_population'];parts=[]
        for axis,q in [('translation_cm','median'),('rotation_deg','median'),('translation_cm','P90'),('rotation_deg','P90')]:
            value=full[axis][q]
            parts.append('inf (failure retained)' if value is None or np.isposinf(value) else f'{value:.6f}')
        rows.append('| '+model+' | '+' | '.join(parts)+f" | {summary['failed_pose']} |")
    table('| 모델 | T 중앙값 cm | R 중앙값 ° | T P90 cm | R P90 ° | 실패 |',rows)
    diagnostic=bound_read(report['source_transfer_diagnostic']);rows=[]
    for model in ('UNION_s1','UNION_s2','UNION_s3'):
        r=diagnostic['models'][model];old=r['previous_Newton_actual_selection']['classes'];current=r['actual_source_selection']['classes'];known=r['known_safe_opportunities']
        rows.append(f"| {model} | {old['safe_improvement']} → {current['safe_improvement']} | {old['unsafe']} → {current['unsafe']} | {current['anchor']} | {known['frames_with_demonstrated_opportunity']} | {known['miss_lower_bound']} |")
    table('| 모델 | safe 개선 이전 Newton→현재 | unsafe 이전 Newton→현재 | 현재 anchor | 알려진 기회 | 놓친 기회 하한 |',rows)
    prior=C.read(C.NEWTON_DOC/'SOURCE_VAL_GATE.json')
    if source['failed_checks']==prior['failed_checks'] and source['failed_checks']:
        old=prior['summaries']['UNION_s3']['full_population']['translation_cm']['median']
        current=source['summaries']['UNION_s3']['full_population']['translation_cm']['median']
        assert f'{old:.9f}→{current:.9f}' in md
        assert '향상으로 표현하지' in md
    checked_starts=set(counts)
    actual_starts={lines[i] for i in range(len(lines)-1) if lines[i].startswith('|') and lines[i+1].startswith('|---')}
    assert checked_starts==actual_starts,('UNVERIFIED_MARKDOWN_TABLE',actual_starts-checked_starts)
    return counts


def selfcheck():
    gallery_selfcheck()
    assert scalar_equal('',None) and scalar_equal('False',False)
    assert scalar_equal('1.25',1.25) and not scalar_equal('1.250000001',1.25)
    assert scalar_equal(str({'a':1}),{'a':1})
    h,s,p=.2,.3,.004
    np.testing.assert_allclose(h+s+p,sum((h,s,p)),rtol=0,atol=0)
    assert not np.isclose(h+s+p,h+p,rtol=1e-14,atol=1e-14)
    print('PUBLICATION_VERIFIER_SELFCHECK_PASS invented_only=1 artifact_reads=0')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',nargs='?',default='verify',choices=('selfcheck','verify'))
    parser.add_argument('--visual-reviewed',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    if args.action=='selfcheck':selfcheck();return
    assert args.visual_reviewed,'The figures must be inspected before attesting visual review.'
    report=C.read(C.DOC/'REPORT_DATA.json');all_bindings(report)
    md=(C.DOC/'REPORT_KO.md').read_text()
    assert (C.DOC/'PUBLIC_REPORT_KO.md').read_bytes()==(C.DOC/'REPORT_KO.md').read_bytes()
    for name in ('PREFIT_REVIEW.json','TRAIN_CONVERGENCE.json','SOURCE_VAL_VERIFICATION.json'):
        audit=C.read(C.DOC/name);assert audit['complete'] and audit['PASS']
    training=C.read(C.DOC/'TRAIN_CONVERGENCE.json');complete=C.read(C.DOC/'TRAINING_COMPLETE.json')
    source=C.read(C.DOC/'SOURCE_VAL_GATE.json')
    source_review=verify_source_exports(source)
    fits,logs,trace_counts=verify_training_exports(report,complete)
    comparisons=check_metadata(report,training,source,complete,fits)
    # Current phase is explicitly frozen at an actual failed source gate. This
    # verifier must fail closed if someone later inserts learned real outputs.
    not_run=verify_not_run(report,source)
    assert '실사' in md and ('미평가' in md or '미실행' in md)
    assert f"{source['checks_passed']}/{source['checks_total']}" in md
    for item in source['failed_checks']:assert '- `'+item+'`' in md
    rejected=sum(r['event']=='objective' and r['phase']=='trial' and not r['armijo_accepted'] for r in logs)
    assert report['rejected_Armijo_trials']==rejected==complete['total_objective_calls']-complete['total_iterations']-4
    figure_count=verify_figure_values(report,source,training,logs,fits)
    tables=verify_tables(md,source,training,fits,report)
    diagnostic=verify_optional_diagnostic(report,source)
    gallery=verify_gallery(report)
    images=[]
    for path in sorted((C.DOC/'figures').iterdir()):
        assert path.suffix in ('.png','.jpg')
        with Image.open(path) as image:width,height=image.size;image.verify()
        assert width>500 and height>300 and 'figures/'+path.name in md
        images.append(dict(binding=C.bind(path),width=width,height=height))
    assert len(images)==6 and sum(i['binding']['path'].endswith('.jpg') for i in images)==3
    links,files=check_links_and_files()
    review=dict(complete=True,PASS=True,created_at=C.now(),code=C.bind(__file__),
        report_code=report['code'],report=C.bind(C.DOC/'REPORT_KO.md'),report_data=C.bind(C.DOC/'REPORT_DATA.json'),
        source_gate_PASS=source['PASS'],source_checks_passed=source['checks_passed'],source_checks_total=45,
        source_export=source_review,trace_counts=trace_counts,rejected_Armijo_trials=rejected,
        certified_models=4,parameter_count=2024,sign_coefficient=1.,
        objective_components_checked=['Huber','Sign_logistic','L2_penalty'],
        previous_Newton_weights_under_same_new_objective=comparisons,
        old_native_Huber_objective_kept_separate=True,
        exact_figure_value_groups=figure_count,verified_tables=tables,figures=images,gallery=gallery,
        visual_review_completed=True,markdown_links_checked=links,learned_real_evaluated=False,
        current_method_real_T_R_effect_measured=False,method_success=False,goal_complete=False,
        fixed_source_diagnostic=diagnostic,evaluation_not_run=C.bind(C.DOC/'REAL_EVALUATION_NOT_RUN.json'),
        reviewed_artifacts=[C.bind(p) for p in sorted(set(files))],
        new_fits_by_audit=0,new_reference_metric_calls=0,
        pending_publication_manifest='PUBLICATION_MANIFEST.json is the only publication artifact intentionally generated after this review; PUBLIC_REVIEW files are this receipt.')
    if args.dry_run:
        print('PUBLIC_REVIEW_DRY_RUN_PASS',json.dumps(dict(files=len(files),figures=len(images),links=links,trace=trace_counts,tables=tables),ensure_ascii=False))
        return
    C.save(C.DOC/'PUBLIC_REVIEW.json',review)
    C.save(C.DOC/'PUBLIC_REVIEW_KO.md','# 공개 결과 검산\n\n공개 기록 검산 PASS입니다. 네 모델의 수렴 인증과 source '+str(source['checks_passed'])+'/45·전체 gate FAIL, 실사 미실행 판정을 구분했습니다.\n\n'
        '- 실제 최종 모델4개의 공개 파라미터가 원본 checkpoint와 byte 단위로 일치합니다.\n'
        f"- objective {trace_counts['objective']}행·승인 반복 {trace_counts['iteration']}행과 source 8192행·45조건을 원본에 대조했습니다.\n"
        '- Huber·부호 logistic·L2의 합계, 이전 Newton 가중치를 같은 새 목적함수로 평가한 비교, 원래 Huber 인증의 구분을 확인했습니다.\n'
        f'- 그래프 {figure_count}개 수치 묶음과 상세 표 {len(tables)}개를 독립 대조했습니다.\n'
        '- 그래프3개와 과거 R0 사진6장·치수의 그림3개를 직접 검토했습니다. 사진은 현재 모델의 실사 결과가 아닙니다.\n'
        '- 실사 routing·metric 파일의 부재와 모든 상대 문서 링크를 확인했습니다. PUBLICATION_MANIFEST는 이 검산 뒤 생성될 예정입니다.\n\n'
        '[검산 JSON](PUBLIC_REVIEW.json) · [상세 보고서](REPORT_KO.md)\n')
    print('PUBLIC_REVIEW_PASS',len(files),len(images),links)


if __name__=='__main__':main()
