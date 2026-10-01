"""Independent review of the two public bundles; saved results only.

No fit, prediction, pose solver, raw reference or policy selection is invoked.
The report generator is never imported or rerun. Root separately views figures.
"""
import argparse
import ast
import csv
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote
import numpy as np
from PIL import Image
from . import common as C

CHECKOUT=Path('/tmp/pallet-pose-github-review-20260930')
PENDING={'PUBLIC_REVIEW.json','PUBLIC_REVIEW_KO.md','PUBLICATION_MANIFEST.json'}
AXES=('translation_cm','rotation_deg')
SPECS=[('translation_cm','median'),('rotation_deg','median'),('translation_cm','P90'),('rotation_deg','P90')]
MODELS=list(C.MODEL_NAMES)
HASH_KEYS=('signed_target_sha','input_difference_sha','base_context_sha','errors_sha','scaled_excess_sha','original_valid_sha',
           'direction_raw_sha','direction_difference_sha','extended_input_sha','visual_raw_sha','visual_difference_sha','visual_extended_input_sha')
VISUAL_BINDINGS=('visual_normalization_sha','visual_receipt_binding','visual_protocol_binding','visual_verification_binding')
READS=set()


def binding(path):
    path=Path(path);digest=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):digest.update(chunk)
    return dict(path=str(path.relative_to(C.ROOT)),sha256=digest.hexdigest(),bytes=path.stat().st_size)


def verify(b):
    actual=binding(C.ROOT/b['path'])
    assert actual['sha256']==b['sha256'],b['path']
    if 'bytes' in b:assert actual['bytes']==b['bytes'],b['path']


def read(path):return json.loads(Path(path).read_text())


def read_bound(b):
    verify(b);return read(C.ROOT/b['path'])


def install_guard():
    allowed_npz={C.RAW/'SOURCE_VAL_METRICS.npz'}
    outputs={C.DOC/'PUBLIC_REVIEW.json',C.DOC/'PUBLIC_REVIEW_KO.md'}
    def hook(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        path=Path(os.fsdecode(args[0])).resolve();name=str(path);mode,flags=args[1:3]
        writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or bool(
            isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
        if writing and path.is_relative_to(C.ROOT):assert path in outputs,('PUBLIC_REVIEW_OUTPUT_ONLY',name);return
        assert not any(t in name for t in ('/annotations/','/real_gt_v2/','GEOMETRY_RESOLVED_POSE_GT','GEOMETRY_SIDETABLE',
            'SYNTH_RECORDS','SYNTH_LABELS','SOURCE_TRAIN_LABELS','TRUTH_FOR_DISPLAY','TRAIN_TOKENS','SOURCE_VAL_TOKENS','REAL_TOKENS')),('NO_RAW_REFERENCE_OR_TOKEN',name)
        assert path.suffix.lower() not in ('.pt','.pth','.onnx'),name
        if path.suffix=='.npz':assert path in allowed_npz,('ONLY_SAVED_VAL_ERRORS',name)
        if path.is_relative_to(C.ROOT):READS.add(str(path.relative_to(C.ROOT)))
    sys.addaudithook(hook)


def inventory():
    paths=[C.ROOT/'readme.md',C.ROOT/'.gitignore']
    for doc,code in ((C.NATIVE_DOC,C.NATIVE_HERE),(C.DOC,C.HERE)):
        paths+=list(code.glob('*.py'))
        for suffix in ('*.md','*.json','*.csv'):paths+=list(doc.glob(suffix))
        paths+=list((doc/'figures').glob('*.png'))+list((doc/'figures').glob('*.jpg'))+list((doc/'model_parameters').glob('*.json'))
    paths=[p for p in paths if p.name not in PENDING]
    assert len(paths)==len(set(paths))
    for p in paths:
        assert p.is_file() and p.stat().st_size<20_000_000,p
        if p.suffix=='.py':ast.parse(p.read_text())
    return sorted(paths)


def links(paths):
    planned={p.relative_to(C.ROOT).as_posix() for p in paths}|{
        (C.DOC/name).relative_to(C.ROOT).as_posix() for name in PENDING}
    tracked=set(subprocess.check_output(['git','ls-tree','-r','--name-only','HEAD'],cwd=CHECKOUT,text=True).splitlines())
    available=planned|tracked;count=0
    for path in paths:
        if path.suffix!='.md':continue
        for raw in re.findall(r'\]\(([^)]+)\)',path.read_text()):
            link=raw.strip().strip('<>')
            if link.startswith(('https://','http://','mailto:','#')):continue
            target=(path.parent/unquote(link.split('#')[0])).resolve()
            assert target.is_relative_to(C.ROOT),(path,link)
            rel=target.relative_to(C.ROOT).as_posix()
            assert rel in available or any(p.startswith(rel.rstrip('/')+'/') for p in available),(path,link)
            count+=1
    return count


def csv_expected(rows):return [{k:'' if v is None else str(v) for k,v in row.items()} for row in rows]


def check_csv(name,expected):
    path=C.DOC/name;payload=path.read_bytes();assert b'\r\n' not in payload
    rows=list(csv.DictReader(io.StringIO(payload.decode())))
    assert rows==csv_expected(expected),('CSV_DIFF',name)
    return len(rows)


def collect_tables(md):
    tables=[];current=[]
    for line in md.splitlines()+['']:
        if line.startswith('|'):
            cells=[s.strip() for s in line.strip().strip('|').split('|')]
            if not all(re.fullmatch(r'[-:]+',s) for s in cells):current.append(cells)
        elif current:tables.append(current);current=[]
    return tables


def projection(pose,K):
    assert pose['available'];R=np.asarray(pose['R_cf']);t=np.asarray(pose['centroid']);d=np.asarray(pose['cf_extents']);K=np.asarray(K)
    assert R.shape==K.shape==(3,3) and d.shape==t.shape==(3,) and (d>0).all()
    result=[]
    for sign in itertools.product((-1.,1.),repeat=3):
        local=[sign[j]*d[j]/2 for j in range(3)]
        xyz=[sum(R[i,j]*local[j] for j in range(3))+t[i] for i in range(3)]
        uvw=[sum(K[i,j]*xyz[j] for j in range(3)) for i in range(3)]
        assert np.isfinite(uvw).all() and uvw[2]!=0
        result.append([uvw[0]/uvw[2],uvw[1]/uvw[2]])
    return np.array(result,np.float64)


def check_native():
    doc=C.NATIVE_DOC;report=read(doc/'REPORT_DATA.json');receipt=read(doc/'TRAIN_APPEARANCE_INPUTS.json');v=read(doc/'INPUT_VERIFICATION.json')
    assert report['complete'] and v['complete'] and v['PASS'] and v['verification_PASS']
    assert report['code']==binding(C.NATIVE_HERE/'report.py')
    assert report['verification']==binding(doc/'INPUT_VERIFICATION.json') and report['input_receipt']==v['public_input_receipt']==binding(doc/'TRAIN_APPEARANCE_INPUTS.json')
    assert receipt['protocol']==v['protocol'] and receipt['descriptors']==v['descriptors']
    assert v['frames']==2598 and v['all_invalid_rows']==1 and v['valid_candidate_descriptors']==20776
    assert v['normalization_exact'] and v['normalization']['count']==5194 and v['exact_removed_support_matches_parent_padding_audit']
    assert v['token_bytes_unchanged'] and v['token_frame_hashes_checked']==2597 and v['full_token_file_sha_checked']
    assert v['main_array_keys']==18 and v['support_detail_array_keys']==17
    assert report['new_fits']==report['new_image_forwards']==report['new_metric_calls']==0 and not report['performance_improvement_measured']
    assert report['models']==receipt['models'];expected=[]
    for model,s in receipt['models'].items():
        for key,value in s.items():assert v['models'][model][key]==value,(model,key)
        assert s['previous_supported_points']-s['native_supported_points']==s['removed_prepared_padding_points']
        assert s['valid_candidates']==5194 and s['valid_zero_native_support_candidates']==0
        expected.append([model]+[str(s[k]) for k in ('previous_supported_points','native_supported_points','removed_prepared_padding_points','valid_descriptor_changed_candidates')])
    md=(doc/'REPORT_KO.md').read_text();tables=collect_tables(md)
    assert len(tables)==1 and tables[0][1:]==expected
    assert 'T/R 성능 평가는 하지 않았다' in md and '반사 패딩의 영향을 받을 수' in md and '실사 성능 예시가 아니다' in md
    assert f"{max(s['comparison']['max_absolute'] for s in v['models'].values()):.9g}" in md
    assert len(report['figures'])==1 and len(report['inherited_figures'])==3
    for b in report['figures']+report['inherited_figures']:verify(b)
    old=read(C.APPEARANCE_DOC/'REPORT_DATA.json');oldfig={b['path']:b for b in old['figures']}
    for b in report['inherited_figures']:assert oldfig[b['path']]==b
    return dict(PASS=True,rows=2598,valid_candidates=20776,table_rows=4,figures=1,
        native_support_counts={m:s['native_supported_points'] for m,s in receipt['models'].items()},
        excluded_padding_counts={m:s['removed_prepared_padding_points'] for m,s in receipt['models'].items()},
        verification=binding(doc/'INPUT_VERIFICATION.json'),report=binding(doc/'REPORT_DATA.json'))


def check_training(report,protocol,train,complete):
    assert complete['complete'] and complete['all_certified'] and complete['fit_count']==4 and complete['models']==MODELS
    assert train['complete'] and train['PASS'] and train['source_TRAIN_only'] and train['frames']==2598
    assert complete['protocol']==train['protocol']==binding(C.DOC/'TRAIN_PROTOCOL.json')
    assert train['new_fits']==train['optimizer_steps']==0 and not train['VAL_quality_read'] and not train['real_targets_read']
    assert protocol['feature_dim']==656 and protocol['previous_feature_dim']==271 and protocol['visual_dim']==385
    assert protocol['bias']==0 and protocol['lambda_l2']==1e-4 and protocol['max_fits']==4
    assert protocol['solver']['maxiter']==1000 and protocol['solver']['maxfun']==2000 and protocol['solver']['initialization']=='zeros'
    prefit=read_bound(protocol['inputs']['prefit_review']);assert prefit['complete'] and prefit['PASS']
    calls=[];iterations=[];certs=[];fits={}
    for model in MODELS:
        fit=read(C.DOC/f'FIT_{model}.json');assert binding(C.DOC/f'FIT_{model}.json') in complete['fits']
        ck=read_bound(fit['checkpoint']);start=read_bound(fit['START']);verify(fit['trace'])
        assert ck['schema']=='pallet_pose_signed_axes_visual_linear656x2_v1' and np.asarray(ck['weight']).shape==(656,2)
        assert ck['certificate']==fit['certificate'] and ck['certificate']['PASS'] and fit['fits_executed']==1
        assert (C.DOC/'model_parameters'/f'{model}.json').read_bytes()==(C.ROOT/fit['checkpoint']['path']).read_bytes()
        for key in HASH_KEYS:assert ck[key]==start[key]==fit[key]==complete[key+'_by_model'][model]==prefit['models'][model][key]==train['models'][model]['hashes'][key]
        for key in VISUAL_BINDINGS:
            for item in (start,fit,ck,complete,train,prefit):assert item[key]==protocol[key]
        trace=[json.loads(line) for line in (C.ROOT/fit['trace']['path']).read_text().splitlines()]
        cc=[r for r in trace if r['event']=='objective'];ii=[r for r in trace if r['event']=='iteration']
        assert len(cc)==fit['objective_calls']==train['models'][model]['objective_calls'] and len(ii)==fit['iterations']==train['models'][model]['iterations']
        assert cc[-1]['armijo_accepted'] and cc[-1]['call']==ck['final_accepted_call']
        for t in cc:
            assert abs(t['objective']-t['Huber_symmetric']-t['Huber_underprediction']-t['Sign_logistic']-t['L2_penalty'])<1e-12
            calls.append(dict(model=model,**{k:t[k] for k in ('call','objective','Huber_symmetric','Huber_underprediction','Sign_logistic','L2_penalty','gradient_linf','gradient_gap_upper_bound','armijo_accepted')}))
        iterations.extend(dict(model=model,**t) for t in ii)
        cert=fit['certificate'];a=train['models'][model];old=a['previous_fixed_asymmetric']
        assert abs(a['independent_recompute']['objective']-cert['objective_value'])<1e-10
        assert cert['gradient_linf']<=1e-8 and cert['gradient_l2_squared_over_2lambda']<=1e-6
        certs.append(dict(model=model,objective=cert['objective_value'],gradient_linf=cert['gradient_linf'],gap_bound=cert['gradient_l2_squared_over_2lambda'],
            calls=fit['objective_calls'],iterations=fit['iterations'],embedded_Q_objective=old['same_new_objective']['objective'],
            previous_safe=old['risk_statistics']['classes']['safe_improvement']['count'],current_safe=a['risk_statistics']['classes']['safe_improvement']['count'],
            previous_unsafe=old['risk_statistics']['classes']['unsafe']['count'],current_unsafe=a['risk_statistics']['classes']['unsafe']['count']))
        assert old['original_choice_replay_PASS'] and old['expanded656_gradient_not_compared_to_original271_certificate']
        fits[model]=fit
    assert len(calls)==complete['total_objective_calls']==312 and len(iterations)==complete['total_iterations']==84
    assert report['training_certificates']==certs
    check_csv('TRAIN_TRACE.csv',calls);check_csv('TRAIN_ITERATIONS.csv',iterations);check_csv('TRAIN_CERTIFICATES.csv',certs)
    return certs,dict(fits=4,parameters=4*1312,objective_rows=len(calls),iteration_rows=len(iterations),certificate_rows=4,
        rejected_trial_rows=sum(not c['armijo_accepted'] for c in calls),parameter_files_byte_exact=True)


def check_source(report,protocol,gate):
    checked=read(C.DOC/'SOURCE_VAL_VERIFICATION.json');av=read(C.DOC/'SOURCE_VAL_APPEARANCE_VERIFICATION.json')
    lock=read(C.DOC/'SOURCE_VAL_ROUTING_LOCK.json');choices=read_bound(lock['choices'])
    assert gate['complete'] and checked['complete'] and checked['PASS'] and av['complete'] and av['PASS']
    assert checked['source_gate']==binding(C.DOC/'SOURCE_VAL_GATE.json') and checked['routing_lock']==binding(C.DOC/'SOURCE_VAL_ROUTING_LOCK.json')
    assert checked['choices']==lock['choices'] and checked['all_gate_booleans_exact'] and checked['recomputed_error_cells_exact']==16384
    assert gate['protocol']==checked['protocol']==binding(C.DOC/'TRAIN_PROTOCOL.json')
    assert gate['visual_input_verification']==binding(C.DOC/'SOURCE_VAL_APPEARANCE_VERIFICATION.json')
    assert checked['checks_total']==gate['checks_total']==45 and checked['checks_passed']==gate['checks_passed']
    assert checked['source_gate_PASS']==gate['PASS'] and checked['failed_checks']==gate['failed_checks']
    selection=checked['source_selection_comparison']
    assert report['source_selection_comparison']==selection
    assert selection['current_gate']==binding(C.DOC/'SOURCE_VAL_GATE.json') and selection['frames']==1024
    for key in ('new_policy_evaluations','new_argmin_computations','new_reference_error_computations'):assert selection[key]==0
    assert set(selection['models'])==set(MODELS)
    for m,s in selection['models'].items():
        for side in ('previous','current'):
            assert s[side]['frames']==1024 and sum(s[side]['classes'].values())==1024
        assert 0<=s['changed_identity_count']<=1024
    metadata=read_bound(lock['metadata']);poses=read_bound(lock['poses'])
    rows=[r for r in metadata if r['split']=='VAL'];assert len(rows)==1024 and choices['ids']==[r['id'] for r in rows]
    verify(gate['metrics'])
    with np.load(C.ROOT/gate['metrics']['path'],allow_pickle=False) as z:
        assert z['ids'].tolist()==choices['ids'] and len(z['models'])==8
        errors={m:z[m].copy() for m in z['models'].tolist()}
    assert report['source_summaries']==gate['summaries'] and set(errors)==set(gate['summaries'])
    expected=[]
    for i,row in enumerate(rows):
        assert len(row['dims'])==3 and all(d>0 for d in row['dims'])
        for model in gate['summaries']:
            e=errors[model][i];choice=choices['records'].get(model,{}).get(row['id'],{})
            expected.append(dict(id=row['id'],model=model,T_cm=e[0],R_deg=e[1],W_cm=100*row['dims'][0],H_cm=100*row['dims'][1],D_cm=100*row['dims'][2],
                available=bool(np.isfinite(e).all()),candidate=choice.get('candidate_name','fixed_GEO'),fallback=choice.get('fallback',False)))
    assert len(expected)==8192;check_csv('SOURCE_VAL_METRICS.csv',expected)
    checks=[dict(model=m,baseline=b,criterion=k,PASS=value) for m,g in gate['comparisons'].items() for b,c in g.items() for k,value in c['checks'].items()]
    assert len(checks)==45 and sum(c['PASS'] for c in checks)==gate['checks_passed'];check_csv('SOURCE_VAL_CHECKS.csv',checks)
    previous=read_bound(report['inputs']['previous_Q_source'])
    assert report['inputs']['previous_Q_source']==protocol['evidence']['previous_asymmetric_source_gate']
    assert selection['previous_gate']==report['inputs']['previous_Q_source']
    for model in gate['baselines']:
        assert gate['summaries'][model]==previous['summaries'][model] and checked['fixed_GEO_baseline_bit_parity'][model]['bit_exact']
    for axis,stat in SPECS:
        assert report['source_comparison'][axis+'_'+stat]==dict(previous_Q271=[previous['summaries'][m]['full_population'][axis][stat] for m in MODELS],
            current656=[gate['summaries'][m]['full_population'][axis][stat] for m in MODELS])
    assert report['source_checks_passed']==gate['checks_passed'] and report['source_checks_total']==45
    return rows,poses,choices,errors,dict(rows=8192,physical_error_cells=16384,dimension_cells=24576,checks=45,checks_passed=gate['checks_passed'],
        all_metric_cells_exported_exact=True,all_dimensions_WHD_cm_exact=True,summary_and_Q_comparison_exact=True,physical_metrics_recomputed_by_public_review=False)


def check_gallery(report,rows,poses,choices,errors):
    gallery=report['illustrations'];assert len(gallery)==6
    assert report['illustration_selection']=='First6 fixed-order SOURCE VAL rows; no outcome-based selection'
    maximum=0.
    for i,item in enumerate(gallery):
        row=rows[i];fid=row['id']
        assert item['source_VAL_index']==i and item['id']==fid and item['image']==row['image']
        assert item['dimensions_m']==row['dims'] and item['K']==row['K'] and item['image_hw']==row['hw']
        verify(row['image'])
        with Image.open(C.ROOT/row['image']['path']) as im:assert [im.height,im.width]==row['hw'];im.verify()
        assert [p['model'] for p in item['panels']]==['R0_GEO','UNION_s1','UNION_s2','UNION_s3']
        for p in item['panels']:
            m=p['model']
            if m=='R0_GEO':parent='R0';hyp=poses['records']['R0'][fid]['GEO_name'];pose=poses['records']['R0'][fid]['GEO_pose']
            else:
                choice=choices['records'][m][fid];parent,hyp=choice['parent'],choice['hypothesis']
                if choice['fallback']:pose=poses['records']['R0'][fid]['GEO_pose']
                else:pose=next(h['pose'] for h in poses['records'][parent][fid]['hypotheses'] if h['name']==hyp)
            assert p['parent']==parent and p['hypothesis']==hyp and p['pose']==pose
            for key,axis in (('T_cm',0),('R_deg',1)):
                expected=errors[m][i,axis]
                assert p[key]==float(expected) if np.isfinite(expected) else p[key] is None
            if pose['available']:
                reference=projection(pose,row['K']);uv=np.asarray(p['projected_uv'],np.float64)
                np.testing.assert_allclose(uv,reference,rtol=1e-12,atol=1e-7)
                maximum=max(maximum,float(np.max(abs(uv-reference))))
            else:assert p['projected_uv'] is None
    return dict(images=6,panels=24,selection='first6 fixed-order VAL',image_SHA_and_dimensions_exact=True,pose_and_saved_TR_exact=True,
        independent_scalar_projection_max_absolute_difference=maximum,new_pose_estimates=0,new_GT_overlay=False,new_metric_calls=0)


def check_real(report,source):
    if not source['PASS']:
        r=read_bound(report['inputs']['real_not_run'])
        assert r['complete'] and r['status']=='NOT_RUN_SOURCE_GATE_FAILED' and r['source_gate']==binding(C.DOC/'SOURCE_VAL_GATE.json')
        assert r['checks_total']==45 and r['checks_passed']==source['checks_passed'] and r['failed_checks']==source['failed_checks']
        assert len(r['absence_checks'])==17 and all(r['absence_checks'].values())
        for path,absent in r['absence_checks'].items():assert absent and not (C.ROOT/path).exists(),path
        for key in ('learned_real_routes','real_metric_calls','raw_real_reference_reads','image_forwards','new_fits_by_real_evaluator'):assert r[key]==0
        assert not report['current_real_evaluated'] and not report['stable_joint_improvement_achieved']
        return None,dict(evaluated=False,absence_paths=17,not_run=binding(C.DOC/'REAL_EVALUATION_NOT_RUN.json'))
    real=read_bound(report['inputs']['real']);checked=read_bound(report['inputs']['real_verification'])
    assert real['complete'] and checked['complete'] and checked['PASS']
    # The actual independent real verifier schema is separately finalized if
    # this branch is authorized; a hash binding to the exact result is mandatory.
    def nested_bindings(x):
        if isinstance(x,dict):
            if 'path' in x and 'sha256' in x:yield x
            else:
                for v in x.values():yield from nested_bindings(v)
        elif isinstance(x,list):
            for v in x:yield from nested_bindings(v)
    assert report['inputs']['real'] in list(nested_bindings(checked)),'REAL_INDEPENDENT_RESULT_BINDING_REQUIRED'
    assert report['current_real_evaluated'] and report['stable_joint_improvement_achieved']==real['stability']['PASS']
    assert real['real_stability_contract']=='matched_intervention_AND_original_SINGLE251_stability'
    match,original=real['matched_intervention_stability'],real['original_goal_stability']
    assert real['stability']['PASS']==(match['PASS'] and original['PASS'])
    assert len(real['stability']['gates'])==5
    for key,gate in real['stability']['gates'].items():assert gate['PASS']==(match['gates'][key]['PASS'] and original['gates'][key]['PASS'])
    return real,dict(evaluated=True,independently_verified=True,combined_stability_PASS=real['stability']['PASS'])


def check_tables(report,certs,source,real):
    md=(C.DOC/'REPORT_KO.md').read_text();tables=collect_tables(md)
    expected=[]
    expected.append([[c['model'],f"{c['objective']:.9f}",f"{c['embedded_Q_objective']:.9f}",f"{c['gradient_linf']:.3g}",f"{c['gap_bound']:.3g}",str(c['calls']),str(c['iterations'])] for c in certs])
    expected.append([[c['model'],f"{c['previous_safe']} → {c['current_safe']}",f"{c['previous_unsafe']} → {c['current_unsafe']}"] for c in certs])
    def fmt(x):return 'inf (failure retained)' if x is None or np.isposinf(x) else f'{x:.6f}'
    def summary_rows(s,names):return [[m]+[fmt(s[m]['full_population'][a][q]) for a,q in SPECS]+[str(s[m]['failed_pose'])] for m in names]
    expected.append(summary_rows(source['summaries'],source['summaries']))
    previous=read_bound(report['inputs']['previous_Q_source'])
    expected.append([[m]+[f"{previous['summaries'][m]['full_population'][a]['median']:.6f} → {source['summaries'][m]['full_population'][a]['median']:.6f}" for a in AXES] for m in MODELS[1:]])
    expected.append([[m,f"{s['previous']['classes']['safe_improvement']} → {s['current']['classes']['safe_improvement']}",
        f"{s['previous']['classes']['unsafe']} → {s['current']['classes']['unsafe']}",str(s['changed_identity_count'])]
        for m,s in report['source_selection_comparison']['models'].items()])
    if real:
        for pop in ('NATURAL99','CLEAN29','WOOD45'):expected.append(summary_rows(real['summaries'][pop],real['models']))
    assert len(tables)==len(expected),(len(tables),len(expected))
    for table,values in zip(tables,expected):assert table[1:]==values,('MD_NUMERIC_TABLE',table[0])
    blocks=re.findall(r'```json\s*\n(.*?)\n```',md,re.S);assert any(json.loads(b)==source['failed_checks'] for b in blocks)
    assert '합성 VAL이며 실사 성능 예시가 아니다' in md and '수렴이며 T/R 일반화 성능 보장이 아니다' in md
    assert 'source VAL과 실사 DEV는 기존 시도에서 반복 사용했다' in md
    if real is None:
        assert '실사 입력 추출·선택·T/R 평가는 실행하지 않았다' in md and '공동 개선은 아직 입증되지 않았다' in md
    if '세 UNION 모두 T 중앙값은 기준 R0보다 낮다' in md:
        assert all(source['summaries'][m]['full_population']['translation_cm']['median']<source['summaries']['R0_GEO']['full_population']['translation_cm']['median'] for m in MODELS[1:])
        assert source['summaries']['UNION_s1']['full_population']['rotation_deg']['median']==source['summaries']['R0_GEO']['full_population']['rotation_deg']['median']
        assert source['checks_passed']==previous['checks_passed']==43
        assert all(f.startswith('UNION_s1/') and 'rotation_deg_median' in f for f in source['failed_checks'])
        assert all(f.startswith('UNION_s3/') and 'translation_cm_median' in f for f in previous['failed_checks'])
    return dict(tables=len(tables),numeric_rows=sum(map(len,expected)),failed_check_list_exact=True)


def main(write,visual_reviewed):
    assert visual_reviewed,'Root must finish direct visual review before actual publication audit'
    assert not (C.DOC/'PUBLIC_REVIEW.json').exists() and not (C.DOC/'PUBLIC_REVIEW_KO.md').exists()
    install_guard()
    report=read(C.DOC/'REPORT_DATA.json');assert report['complete'] and report['code']==binding(C.HERE/'report.py')
    for b in report['inputs'].values():verify(b)
    protocol=read(C.DOC/'TRAIN_PROTOCOL.json');train=read(C.DOC/'TRAIN_CONVERGENCE.json');complete=read(C.DOC/'TRAINING_COMPLETE.json');source=read(C.DOC/'SOURCE_VAL_GATE.json')
    assert read(C.DOC/'TRAIN_PROTOCOL_SHA.json')==binding(C.DOC/'TRAIN_PROTOCOL.json')
    for key in ('TRAIN_PROTOCOL','TRAIN_CONVERGENCE','TRAINING_COMPLETE','SOURCE_VAL_GATE','SOURCE_VAL_VERIFICATION','SOURCE_VAL_ROUTING_LOCK','SOURCE_VAL_APPEARANCE_VERIFICATION'):
        assert report['inputs'][key]==binding(C.DOC/(key+'.json'))
    for key in ('new_fits','new_metric_calls','new_PnP_solves'):assert report[key]==0
    native=check_native();certs,training=check_training(report,protocol,train,complete)
    rows,poses,choices,errors,source_review=check_source(report,protocol,source)
    gallery=check_gallery(report,rows,poses,choices,errors);real,real_review=check_real(report,source)
    tables=check_tables(report,certs,source,real)
    expected_exports={f'model_parameters/{m}.json' for m in MODELS}|{'TRAIN_TRACE.csv','TRAIN_ITERATIONS.csv','TRAIN_CERTIFICATES.csv','SOURCE_VAL_METRICS.csv','SOURCE_VAL_CHECKS.csv'}
    assert {str(Path(b['path']).relative_to(C.DOC.relative_to(C.ROOT))) for b in report['exports']}==expected_exports
    expected_figures={'training_and_certificate.png','source_val_comparison.png'}|{f'source_val_rgb_dimensions_{i}.jpg' for i in (1,2,3)}
    assert {Path(b['path']).name for b in report['figures']}==expected_figures
    for b in report['exports']+report['figures']:verify(b)
    files=inventory();imagefiles=[p for p in files if p.suffix in ('.png','.jpg')]
    assert len(imagefiles)==6
    for p in imagefiles:
        with Image.open(p) as im:assert min(im.size)>300;im.verify()
    link_count=links(files)
    from . import prepare_publication as producer
    assert set(files)=={p for p in producer.inventory() if p.name not in PENDING}
    result=dict(complete=True,PASS=True,created_at=C.now(),code=binding(__file__),
        native_inputs=native,training=training,source=source_review,gallery=gallery,real=real_review,markdown=tables,
        figures=len(imagefiles),relative_links=link_count,root_visual_review_confirmed=True,
        report=binding(C.DOC/'REPORT_DATA.json'),source_gate_PASS=source['PASS'],source_checks_passed=source['checks_passed'],source_checks_total=45,
        current_real_evaluated=report['current_real_evaluated'],stable_joint_improvement_achieved=report['stable_joint_improvement_achieved'],
        reviewed_artifacts=[binding(p) for p in files],new_fits_by_audit=0,new_image_forwards=0,new_PnP_solves=0,new_policy_selections=0,
        new_reference_metric_calls=0,raw_GT_reads=0,goal_completion_asserted_by_audit=False,
        claim_scope='Verifies saved results, exports,24 saved-pose projections, dimensions, input correction and publication bytes; does not repeat metrics/backbone or infer generalization from convergence.',
        excluded_pending_publication_files=sorted(PENDING),read_paths=sorted(READS))
    if not write:
        print('VISUAL_PUBLIC_REVIEW_DRY_RUN_PASS',len(files),training,source_review,tables,flush=True);return result
    C.save(C.DOC/'PUBLIC_REVIEW.json',result)
    C.save(C.DOC/'PUBLIC_REVIEW_KO.md',f'''# 공개 결과 독립 검산

공개 기록 검산 **PASS**다. 이는 방법의 T/R 성공 판정과 별개다. source는 {source['checks_passed']}/45이며 현재 실사 평가 실행 여부는 {report['current_real_evaluated']}다.

- native 입력2598행·20776후보·padding 제외 수치와 표4행을 독립 입력 검산에 대조했다.
- 실제4fits의312objective·84iteration CSV,4인증행·656×2 파라미터 원본 bytes와12입력해시를 확인했다.
- source8192행의T/R16384값과W/H/D24576값,45판정·이전Q 비교를 저장 결과와 대조했다. 새 물리오차 계산은 없다.
- 순서 고정 첫6합성VAL RGB의 SHA/크기·실제치수·고정선택pose·24패널UV를 독립 scalar 투영으로 확인했다. 실사 예시로 표시하지 않았다.
- 새 그림6개와 숫자표{tables['numeric_rows']+4}행, 상대 링크{link_count}개 및 공개 파일{len(files)}개의 해시를 확인했다. root는 그림을 직접 시각 검토했다.

실사 미실행 또는 별도 독립 실사 검산 필수 분기를 확인했다. 새학습·추론·PnP·후보선택·rawGT조회0이다. GitHub 공개 파일만으로 비공개 원본RGB/token 없이 전체 재실행할 수 있다는 주장은 하지 않는다.

[검산 JSON](PUBLIC_REVIEW.json) · [상세 결과](REPORT_KO.md)
''')
    print('VISUAL_PUBLIC_REVIEW_PASS',binding(C.DOC/'PUBLIC_REVIEW.json'),flush=True)
    return result


def selfcheck():
    pose=dict(available=True,R_cf=np.eye(3).tolist(),centroid=[0.,0.,2.],cf_extents=[2.,2.,2.])
    uv=projection(pose,[[100,0,50],[0,100,40],[0,0,1]])
    np.testing.assert_array_equal(uv[0],[-50.,-60.]);np.testing.assert_array_equal(uv[-1],[250/3,220/3])
    rows=[dict(a=None,b=True,c=1.25,d={'nested':[1,2]})];buf=io.StringIO();w=csv.DictWriter(buf,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    assert list(csv.DictReader(io.StringIO(buf.getvalue())))==csv_expected(rows)
    assert collect_tables('| a | b |\n|---|---:|\n| 1 | 2 |\n')==[[['a','b'],['1','2']]]
    assert not READS
    print('VISUAL_PUBLIC_REVIEW_SELFCHECK_PASS_NO_ARTIFACT_READS',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selfcheck',action='store_true');p.add_argument('--write',action='store_true');p.add_argument('--visual-reviewed',action='store_true')
    a=p.parse_args()
    if a.selfcheck:assert not a.write;selfcheck()
    else:main(a.write,a.visual_reviewed)
