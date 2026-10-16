"""ROOT-only V21 reader report from completed, independently checked evidence.

No error, moment, CI, fit, model, RGB, projection, Jacobian or SVD is computed.
Only saved JSON scalars and saved POSTHOC memberships are transcribed. A short
main report and a complete appendix are exclusive outputs; old reports remain.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import traceback

PRIVATE = Path('/tmp/pallet-whole-u-closure-private-20261011-v21')
HISTORY = Path('/tmp/pallet-finite-support-orthogonal-private-20261011-v19')
ROUTING = Path('/tmp/pallet-consensus-routing-private-20261011-v22')
PRIMARY = 'HYBRID_POINT_CONSENSUS_WHOLE_U_CLOSURE'
METHODS = ('BASE', 'N3_SUBPIX', 'HYBRID_STANDARD_PNP', 'NATIVE_STANDARD_PNP',
    'HYBRID_POINT_CONSENSUS_RESTORED', 'HYBRID_OBSERVED_FRAME_FACTOR_SUBSET_CONSENSUS',
    'HYBRID_OBSERVED_FRAME_POINT_ONLY', PRIMARY)
LABELS = dict(zip(METHODS, ('Base', 'N3→cornerSubPix', 'Hybrid ordinary', 'Native ordinary',
    'V18 결정 복원 점', 'V19 관측선', 'V19 같은 q 점', 'V21 whole-U 점')))
FIELDS = ('translation_cm', 'rotation_deg', 'ADDsym_cm')
MOMENTS = ('n', 'mean', 'population_variance', 'population_std', 'sample_variance', 'sample_std', 'median', 'P90')
TIMES = ('full', 'detector', 'N3_correction', 'initial_pose', 'boundary_observation',
    'hybrid_selection_and_point_consensus', 'final_pose_reprojection_and_metadata', 'return')
FILES = ('REPORT_PROTOCOL.json', 'REPORT_STARTED.json', 'REPORT_CHECKS.json', 'RESULT_KO.md.ready', 'FULL_AUDIT_KO.md.ready')
LIMITS = dict(saved_JSON_and_POSTHOC_membership_transcription_only=True,
    new_error_moment_CI_model_training_GT_RGB_fit_projection_J_SVD_calls=0,
    known_DEV_GEOMETRIC_PROXY_not_independent_physical_GT=True,
    full245_preserved=True, historical_full_V16_564_FAIL_remains_FAIL=True,
    prior_V19_runtime_cleanup_failure_preserved=True, old_reports_not_modified=True,
    report_generation_is_not_publication_or_accuracy_success=True)

def require(value, message):
    if not value: raise ValueError(message)

def read(path): return json.loads(Path(path).read_bytes())

def binding(path):
    path = Path(path).absolute()
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), 'missing/symlink ' + str(path))
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''): h.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=h.hexdigest())

def same(a, b):
    return isinstance(a, dict) and isinstance(b, dict) and all(a.get(k) == b.get(k) for k in ('bytes', 'sha256'))

def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())

def paths(a):
    f = dict(reporter=Path(__file__).resolve(), evaluator=a.private/'evaluate.py', solver=a.private/'solver.py', pipeline=a.private/'pipeline.py', statistics_checker=a.private/'stats_check.py',
        geometry_checker=a.private/'validation.py', runtime_checker=a.private/'runtime_check.py', renderer=a.private/'render_results.py')
    groups = {'g': (a.geometry_input, ('GEOMETRY_PROTOCOL.json', 'GEOMETRY_STARTED.json', 'GEOMETRY_RECEIPT.json')),
        'v': (a.validation_input, ('VALIDATION_PROTOCOL.json', 'VALIDATION_STARTED.json', 'VALIDATION_CHECKS.json')),
        's': (a.statistics_input, ('STATISTICS_CHECK_PROTOCOL.json', 'STATISTICS_CHECK_STARTED.json', 'STATISTICS_CHECKS.json')),
        'e': (a.evaluation_input, ('EVALUATION_PROTOCOL.json', 'SCORING_AUTHORITY.json', 'SCORING_STARTED.json', 'SCORING_RECEIPT.json', 'STATISTICS_STARTED.json', 'METRICS.json', 'POSTHOC_ROWS.jsonl.gz', 'SAME_OBSERVATION_JOIN.json')),
        't': (a.runtime_input, ('RUNTIME_PROTOCOL.json', 'RUNTIME_STARTED.json', 'RUNTIME_RECEIPT.json', 'RUNTIME_RESOURCES.json', 'RUNTIME_PREFLIGHT.json')),
        'c': (a.runtime_checks_input, ('RUNTIME_CHECK_PROTOCOL.json', 'RUNTIME_CHECK_STARTED.json', 'RUNTIME_CHECKS.json')),
        'f': (a.render_input, ('FIGURE_PROTOCOL.json', 'FIGURES_STARTED.json', 'FIGURE_BINDINGS.json', 'FIGURE_DIAGNOSTIC_COUNTS.json', 'ROOT_VISUAL_REVIEW.json')),
        'r': (a.routing_input, ('PROTOCOL.json', 'STARTED.json', 'SUMMARY.json', 'CHECKS.json')),
        'stress': (a.stress_input, ('STRESS_PROTOCOL.json', 'STRESS_STARTED.json', 'STRESS_CHECKS.json')),
        'cpu': (a.cpu_input, ('CPU_PROTOCOL.json', 'CPU_STARTED.json', 'CPU_CHECKS.json')),
        'history_main': (a.history/'report_summary', ('SUMMARY_PROTOCOL.json', 'SUMMARY_STARTED.json', 'SUMMARY_CHECKS.json', 'RESULT_KO.md.ready')),
        'history_full': (a.history/'report_ready', ('REPORT_RESULTS_PROTOCOL.json', 'REPORT_RESULTS_STARTED.json', 'REPORT_RESULTS_CHECKS.json', 'RESULT_KO.md.ready'))}
    for prefix, (folder, names) in groups.items(): f.update({prefix+':'+n: folder/n for n in names})
    routing_protocol=read(f['r:PROTOCOL.json']);f['routing_code']=Path(routing_protocol['inputs']['code']['path'])
    for key in ('original_diagnostic_source','original_attempt_protocol','original_attempt_started','original_attempt_exit_note'):
        f['routing_history:'+key]=Path(routing_protocol['inputs'][key]['path'])
    runtime_protocol=read(f['t:RUNTIME_PROTOCOL.json'])
    for key,value in runtime_protocol['inputs'].items():
        if key.startswith('runtime_output_review:'):f[key]=Path(value['path'])
    fig = read(f['f:FIGURE_BINDINGS.json'])
    require(len(fig['figures']) == 9, 'nine completed figure bindings')
    f.update({'PNG:'+str(i): Path(v['path']) for i,v in enumerate(fig['figures'])})
    receipt = read(f['t:RUNTIME_RECEIPT.json']); require(isinstance(receipt.get('raw_rows'),dict), 'actual runtime raw receipt binding')
    f['runtime_rows'] = Path(receipt['raw_rows']['path'])
    doc = a.repo/'_docs/experiments'
    f['old_full_scalar_failure'] = doc/'pallet_hybrid_factor_subset_consensus_20261011_v16/VALIDATION_READY_CHECKS.json'
    f['supervision_repair'] = doc/'pallet_kp_corrected_supervision_20261010_v1/RESULT_KO.md'
    f['publication_checkpoint'] = doc/'pallet_observation_refiner_status_20261011/PUSH_V16_V18_RECEIPT.json'
    f['historical_no_mask_v3'] = doc/'pallet_cornerwise_refiner_20261010_v3/RESULT_KO.md'
    f['historical_no_mask_v7'] = doc/'pallet_three_head_observation_20261010_v7/RESULT_KO.md'
    f['corner_definition'] = a.repo/'scripts/research/pallet_boundary_corner_refiner_20261010_v2/observations.py'
    f['corner_LOO'] = a.repo/'scripts/research/pallet_cornerwise_independent_20261010_v4/selection.py'
    return f

def checked(f, b, prefix, protocol_name, started_name, checks_name, protocol_schema, checks_schema, error_key='exception'):
    p,s,c = [read(f[prefix+':'+n]) for n in (protocol_name,started_name,checks_name)]
    require(p['schema']==protocol_schema and c['schema']==checks_schema and c['complete'] is c['passed'] is True
        and c.get(error_key) is None and c.get('new_production_calls',0)==s.get('new_production_calls',0)==0,
        'actual completed authority '+prefix)
    require(c['inputs']==p['inputs']==s['inputs'], 'same frozen authority inputs '+prefix)
    cb=c.get('checker',c.get('code')); own=c.get('own_protocol',c.get('protocol'))
    sb=s.get('own_protocol',s.get('protocol'))
    require(same(own,b[prefix+':'+protocol_name]) and same(sb,own), 'own protocol byte join '+prefix)
    require(isinstance(cb,dict), 'source binding '+prefix)
    return p,s,c

def gate(a,f,b):
    vp,vs,v=checked(f,b,'v','VALIDATION_PROTOCOL.json','VALIDATION_STARTED.json','VALIDATION_CHECKS.json',
        'whole_U_closure_point_scalar_protocol_v21','whole_U_closure_point_scalar_checks_v21')
    sp,ss,s=checked(f,b,'s','STATISTICS_CHECK_PROTOCOL.json','STATISTICS_CHECK_STARTED.json','STATISTICS_CHECKS.json',
        'independent_whole_U_closure_statistics_protocol_v21','independent_whole_U_closure_statistics_checks_v21','error')
    rp,rs,r=checked(f,b,'r','PROTOCOL.json','STARTED.json','CHECKS.json',
        'saved_point_closure_routing_protocol_v22','saved_point_closure_routing_checks_v22')
    m,g,route=[read(f[k]) for k in ('e:METRICS.json','g:GEOMETRY_RECEIPT.json','r:SUMMARY.json')]
    require(g['schema']=='whole_U_closure_geometry_receipt_v21' and g['complete'] is True and g.get('error') is None,
        'new full245 completed geometry')
    require(g['actual_execution']==dict(frames_started=245,geometry_rows=245,ledger_rows=245,frames_complete=245), 'complete new245 geometry execution')
    require(same(v['checker'],b['geometry_checker']) and v['failure_count']==0 and v['old_full_V16_validation_passed'] is False
        and v['limits']['old_full_V16_564_FAIL_remains_FAIL'] is True
        and all(v['actual_counts'][k]==245 for k in ('frames','geometry_rows','ledger_rows')), 'new own scalar scope')
    for key,name in (('geometry_protocol','GEOMETRY_PROTOCOL.json'),('geometry_started','GEOMETRY_STARTED.json'),('geometry_receipt','GEOMETRY_RECEIPT.json')):
        require(same(v['inputs'][key], b['g:'+name]), 'same new geometry evidence '+key)
    require(same(s['checker'],b['statistics_checker']) and same(s['inputs']['METRICS'],b['e:METRICS.json']), 'same independently checked statistics source and JSON')
    expected=dict(saved_error_rows_streamed=2205,saved_error_rows_selected=1960,frames=245,methods=8,
        moment_distributions_checked=783,moment_scalars_checked=6264,paired_groups_checked=63,metric_CI_slots_checked=189)
    require(all(s['actual_counts'][k]==n for k,n in expected.items()), 'independent eight-method complete arithmetic')
    require(m['schema']=='whole_U_closure_raw_row_metrics_v21' and m['complete'] is True and m['methods']==list(METHODS)
        and m['V16_full_scalar_validation_passed'] is False and m['all245_preserved_no_performance_filter'] is True
        and m['population_variance_ddof']==m['population_std_ddof']==0 and m['unchanged_sample_variance_ddof']==1,
        'whole known-DEV proxy evaluation')
    require(same(m['posthoc_rows'],b['e:POSTHOC_ROWS.jsonl.gz']) and same(m['same_observation_join'],b['e:SAME_OBSERVATION_JOIN.json']), 'saved diagnostic/observation joins')
    join=read(f['e:SAME_OBSERVATION_JOIN.json']);require(join['complete'] is join['passed'] is True and join['frames']==245
        and join['native_ordinary_has_original_native_q'] is True and join['fixed_baseline_H_and_fit_metadata_unrecorded'] is True,'limited hybrid coordinate parity scope')
    require(r['failure_count']==0 and r['old_full_V16_validation_passed'] is False and r['new_production_calls']==0
        and r['limits']['physical_pose_or_ownership_certified'] is False and same(r['code'],b['routing_code'])
        and same(r['inputs']['statistics:STATISTICS_CHECKS.json'],b['s:STATISTICS_CHECKS.json']), 'actual separate V22 routing authority')
    require(route['schema']=='saved_point_closure_routing_summary_v22' and route['complete'] is True and route['frames']==245
        and same(route['rows'],r['rows']) and route['limits']==r['limits'], 'same full245 routing summary')
    memory=r['memory_only_review']
    require(memory==rp['memory_only_review']==rs['memory_only_review']==route['memory_only_review']
        and memory['schema']=='routing_saved_pose_memory_only_review_v22' and memory['inverse_whole_original_AST_equal'] is True
        and memory['AST_source_span_changes']==4 and memory['retained_full_actual_pose_packets'] is False
        and memory['streaming_JSON_encoder'] is True and memory['actual_pose_identity_scope'].startswith('PRIMARY_ONLY;')
        and memory['scientific_score_gate_routing_outcome_summary_formulas_changed'] is False
        and memory['parent_candidate_poses_or_metrics_or_settings_changed'] is False and memory['original_exit137_cause_confirmed'] is False
        and memory['original_attempt_outputs_modified'] is False and same(memory['ready_source'],b['routing_code'])
        and same(memory['original_source'],b['routing_history:original_diagnostic_source']), 'four-site memory-only ready/source scope')
    for key in ('original_diagnostic_source','original_attempt_protocol','original_attempt_started','original_attempt_exit_note'):
        require(same(r['inputs'][key],b['routing_history:'+key]),'preserved routing first attempt '+key)
    for name,n in (('combined',245),('easy',153),('medium',92)):
        require(m['strata'][name]['frames']==route['strata'][name]['frames']==n
            and all(m['strata'][name]['methods'][method]['denominator']==n for method in METHODS), 'unfiltered population '+name)
    old=read(f['old_full_scalar_failure']);require(old['complete'] is True and old['passed'] is False
        and old['checks']==198134007 and old['failure_count']==564, 'original full V16 FAIL retained')
    cp,cs,c=[read(f['c:'+n]) for n in ('RUNTIME_CHECK_PROTOCOL.json','RUNTIME_CHECK_STARTED.json','RUNTIME_CHECKS.json')]
    t,tp,tj=[read(f['t:'+n]) for n in ('RUNTIME_RECEIPT.json','RUNTIME_PROTOCOL.json','RUNTIME_STARTED.json')]
    require(cp['schema']=='independent_whole_U_point_closure_runtime_protocol_v21'
        and c['schema']=='independent_whole_U_point_closure_runtime_checks_v21'
        and cp['inputs']==cs['inputs']==c['inputs'] and same(c['own_protocol'],b['c:RUNTIME_CHECK_PROTOCOL.json'])
        and same(cs['own_protocol'],c['own_protocol']) and same(c['checker'],b['runtime_checker'])
        and same(c['inputs']['receipt'],b['t:RUNTIME_RECEIPT.json']) and same(c['inputs']['rows'],b['runtime_rows']), 'actual saved runtime audit envelope')
    require(t['schema']=='fresh_whole_U_point_closure_runtime_receipt_v21' and t['method']==PRIMARY
        and same(t['protocol'],b['t:RUNTIME_PROTOCOL.json']) and same(t['raw_rows'],b['runtime_rows'])
        and tj['inputs']==tp['inputs'] and same(tj['receipt'],b['t:RUNTIME_RECEIPT.json']), 'fresh runtime journal/rows')
    clean=t['complete'] is True and t['status']=='COMPLETE' and t['statistics_official'] is True
    if clean:
        require(c['complete'] is c['passed'] is True and c['failure_count']==0 and c['exception'] is None
            and c['rows']==150 and c['measured']==130 and c['summary_blocks']==8 and c['summary_slots']==56
            and t['serialized_rows']==150 and t['cleanup_errors']==[] and t['all_recorded_resource_snapshots_quiet'] is True,
            'actual full150/130 independently checked clean runtime')
    else:
        require(t['complete'] is False and t['statistics_official'] is False and t['status'].startswith('FAILED')
            and c['passed'] is False, 'preserved failed runtime; never fabricate official timing')
    require(c['old_full_V16_validation_passed'] is False and c['historical_V19_runtime_cleanup_failure_preserved'] is True,
        'historical failures remain failures')
    review=tp['runtime_output_review'];require(review==t['runtime_output_review']
        and review['schema']=='explicit_whole_U_runtime_output_relocation_v21' and review['output_folder_AST_changes']==1
        and review['inverse_whole_original_runtime_AST_equal'] is True and review['original_runtime_source_identity_retained'] is True
        and review['original_failed_preflight_preserved'] is True and review['first_attempt_model_fit_decode_calls']==0
        and review['scheduler_math_resource_or_threshold_changes']==0 and review['automatic_retry'] is False,'output-only runtime ready')
    for key,value in review['bindings'].items():require(same(value,b['runtime_output_review:'+key]),'runtime relocation source/history '+key)
    pre=read(f['runtime_output_review:first_failed_preflight'])
    require(pre['passed'] is False and pre['actual_model_fit_decode_calls']==0 and pre['resource']['quiet'] is False,'original quiet preflight FAIL0 retained')
    fig,visual=[read(f['f:'+n]) for n in ('FIGURE_BINDINGS.json','ROOT_VISUAL_REVIEW.json')]
    fp,fs=[read(f['f:'+n]) for n in ('FIGURE_PROTOCOL.json','FIGURES_STARTED.json')]
    require(fig['schema']=='whole_U_closure_results_figure_bindings_v21' and fig['complete'] is True
        and same(fig['renderer'],b['renderer']) and same(fig['inputs']['statistics_checks'],b['s:STATISTICS_CHECKS.json'])
        and fig['inputs']==fp['inputs']==fs['inputs'] and same(fig['own_protocol'],b['f:FIGURE_PROTOCOL.json'])
        and fig['actual_counts']['PNG_files']==9 and fig['actual_counts']['case_panels']==18
        and visual['schema']=='actual_ROOT_visual_review_v21' and visual['complete'] is visual['passed'] is True
        and visual['actual_viewed_PNG_count']==9 and visual['actual_case_panels_viewed']==18
        and same(visual['figure_bindings'],b['f:FIGURE_BINDINGS.json']),
        'nine actual viewed same fixed6-case figures')
    viewed={item['path']:item for item in visual['viewed_figures']}
    for i,item in enumerate(fig['figures']): require(same(item,b['PNG:'+str(i)]) and item['path'] in viewed and same(item,viewed[item['path']]), 'same viewed PNG')
    stress=read(f['stress:STRESS_CHECKS.json']);stress_p=read(f['stress:STRESS_PROTOCOL.json']);stress_s=read(f['stress:STRESS_STARTED.json'])
    require(stress['schema']=='fixed_synthetic_point_closure_stress_checks_v21'
        and stress['complete'] is stress['passed'] is True and stress['configured_cases']==stress['completed_cases']==9
        and stress['exception'] is None and len(stress['case_results'])==9
        and stress['inputs']==stress_p['inputs']==stress_s['inputs'] and same(stress_s['protocol'],stress['protocol'])
        and same(stress['protocol'],b['stress:STRESS_PROTOCOL.json']), 'actual fixed nine synthetic stress cases')
    cpu=read(f['cpu:CPU_CHECKS.json']);cpu_p=read(f['cpu:CPU_PROTOCOL.json']);cpu_s=read(f['cpu:CPU_STARTED.json'])
    require(cpu['schema']=='whole_U_closure_actual_CPU_checks_v21' and cpu['complete'] is cpu['passed'] is True
        and cpu['tests_run']==7 and cpu['failures']==cpu['errors']==cpu['skipped']==cpu['cleanup_errors']==[]
        and cpu['exception'] is None and cpu['inputs']==cpu_p['inputs']==cpu_s['inputs']
        and same(cpu['protocol'],b['cpu:CPU_PROTOCOL.json']), 'actual source contract CPU authority')
    for prefix,name in (('history_main','SUMMARY_CHECKS.json'),('history_full','REPORT_RESULTS_CHECKS.json')):
        hc=read(f[prefix+':'+name]);require(hc['complete'] is hc['passed'] is True and hc['error'] is None
            and same(hc['report'],b[prefix+':RESULT_KO.md.ready']), 'immutable old V19 report '+prefix)
    return dict(metrics=m,geometry=g,validation=v,statistics=s,routing=route,routing_checks=r,
        runtime=t,runtime_checks=c,runtime_clean=clean,figures=fig,stress=stress,cpu=cpu)

def cell(value):
    if value is None: return 'None'
    if isinstance(value,(list,dict)): value=json.dumps(value,ensure_ascii=False,separators=(',',':'))
    return str(value).replace('|','\\|').replace('\n',' ')

def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join('---' for _ in headers)+' |',
        *('| '+' | '.join(cell(v) for v in row)+' |' for row in rows)])

def label(path,a):
    path=Path(path).resolve()
    for root,family in ((a.private,a.family),(a.history,'pallet_finite_support_orthogonal_audit_20261011_v19'),(a.routing_private,a.routing_family)):
        if path.is_relative_to(root):
            rel=path.relative_to(root)
            target=(Path('scripts/research') if len(rel.parts)==1 and path.suffix=='.py' else Path('_docs/experiments'))/family/rel
            return os.path.relpath(target,Path('_docs/experiments')/a.family)
    require(path.is_relative_to(a.repo),'only declared public roots '+str(path))
    return os.path.relpath(path.relative_to(a.repo),Path('_docs/experiments')/a.family)

def link(path,a,text=None,line=None):
    target=label(path,a)
    if Path(path).suffix=='.gz' and Path(path).stat().st_size>40*1024*1024:
        target=os.path.relpath(Path('_docs/experiments')/a.family/'ARCHIVE_MANIFEST.json',Path('_docs/experiments')/a.family)
    return '['+(text or label(path,a))+']('+target+('#L'+str(line) if line is not None else '')+')'

def posthoc_rows(path,ids):
    seen=set()
    with gzip.open(path,'rt') as stream:
        for line in stream:
            require(bool(line.strip()),'blank saved POSTHOC');row=json.loads(line)
            require(row['method']==PRIMARY and row['id'] in ids and row['id'] not in seen,'unique saved245 POSTHOC')
            seen.add(row['id']);yield row
    require(seen==set(ids),'complete245 saved diagnostics, no performance filtering')

def reports(a,f,e):
    m=e['metrics'];main=[];full=[];statuses=[];brief=[];marginal=[];matched=[];ci=[];common=[]
    for name,z in m['strata'].items():
        for method in METHODS:
            s=z['methods'][method]
            statuses.append([name,LABELS[method],s['denominator'],s['operational'],s['new_pose'],s['fallback'],s['no_pose'],s['statuses']])
            brief.append([name,LABELS[method],s['new_pose'],s['fallback'],*[s['metrics']['operational'][metric][key]
                for metric in FIELDS for key in ('mean','population_std','median','P90')]])
            for scope,metrics in s['metrics'].items():
                for metric,moment in metrics.items(): marginal.append([name,method,scope,metric,*[moment[k] for k in MOMENTS],moment['max'],moment['unit']])
        for contrast,scopes in z['contrasts'].items():
            for scope,pair in scopes.items():
                for metric,value in pair['metrics'].items():
                    row=[name,contrast,scope,metric,pair['common_frames'],value['mean_delta'],value['CI95'],pair['bootstrap_nonempty_resamples'],pair['bootstrap_empty_resamples'],value['unit']]
                    ci.append(row)
                    if name=='combined' and scope=='common_operational': common.append(row)
                    for role,moment in pair['matched_recorded_error_and_delta_moments'][metric].items():
                        matched.append([name,contrast,scope,metric,role,*[moment[k] for k in MOMENTS],moment['max'],moment['unit']])
    require(len(marginal)==216 and len(matched)==567 and len(ci)==189,'complete783 distributions/189CI transcription')
    combined=m['strata']['combined'];current=combined['methods'][PRIMARY]['metrics']['operational']
    verdicts=m['full245_T_R_mean_comparisons']
    primary_status=combined['methods'][PRIMARY]
    masks=[];corners=[];diagnostic=combined['diagnostics'][PRIMARY]
    for key,value in diagnostic['mask_and_pose_groups_vs_fixed_N3'].items():
        mask,outcome=key.split('__');masks.append([mask,outcome,value['n'],value['new_pose'],value['fallback']])
    for group in ('DIRECT_VISIBLE','SELF_OCCLUDED','ACTUALLY_REPROJECTED'):
        value=diagnostic[group]
        for stage,moment in value['errors'].items(): corners.append([group,stage,value['corners'],value.get('damage_5px_to_over10px'),*[moment[k] for k in ('n','mean','sample_std','median','P90')]])
    stress_rows=[]
    for r in e['stress']['case_results']:
        stress_rows.append([r['id'],r['hidden_good_removed'],r['bad_coordinates_retained'],r['remaining_correct_count'],r['NEW'],r['state'],r.get('actual_fit_ids'),r.get('actual_final_inlier_ids'),r.get('true_pose_errors'),r.get('contract_failures')])
    routing=e['routing']['strata']['combined']
    goal=m['primary_both_T_R_means_better_than_N3_full245'] and verdicts['NATIVE_STANDARD_PNP']
    main.extend(['# 가림 판단이 일부 틀려도 다른 유효 대응점으로 자세를 구하고 자기 가림 코너를 재투영했을 때, 기존 단순 대조보다 실제 위치·회전이 좋아졌는가?',
        ('**현재 비교 기준에서는 아직 아닙니다.** ' if not goal else '**현재 평가 참조의 전체 평균 기준에서는 개선했습니다.** ')+
        '**현재 known DEV 245장 GEOMETRIC_PROXY 평가 참조 기준**에서 V21 whole-U 점 경로의 전체 평균은 T='+cell(current['translation_cm']['mean'])+'cm, R='+cell(current['rotation_deg']['mean'])+'deg, ADDsym='+cell(current['ADDsym_cm']['mean'])+'cm다. N3 대비 T/R 동시 개선='+cell(m['primary_both_T_R_means_better_than_N3_full245'])+', 이전 같은 q 점 경로 대비='+cell(m['primary_both_T_R_means_better_than_old_point_full245'])+', native ordinary 대비='+cell(verdicts['NATIVE_STANDARD_PNP'])+'다. **독립 실측 자세 개선까지 검증한 결과는 아니다.** 목표 성공과 saved arithmetic PASS는 구분한다.',
        '경계 교점 q 자체의 정의가 임의 한 ray 교점을 3D 코너로 넣는 구현은 아니다. 서로 다른 incident edge 두 TLS line의 교점만 해당 native corner q를 만든다. 학습 head의 실제 wire/query 목표와 virtual cuboid corner는 구분하며 실사 physical edge/corner ownership은 UNKNOWN이다. query 의미에서 inherited edge ID, 8px proxy 정합, rank6가 실제 올바른 대응을 독립 인증하지 않는다.',
        '## 무엇을 이미 고쳤고 무엇이 남았나',
        '별도 head 감독에서 **실제 대응이 존재했던2,164 query를 수치광선교차 실패만으로 no_match로 감독한 오류**는 POS/NONE/IGNORE 계약으로 수정했다. 원 감사 시점 수정 후 재학습은0이었고, 이후 별도 승인으로 수정 감독3×3000=9000 학습을 완료했다. 원 pre-fix IMAGE_ROLE0/319는 현재245 성능이 아니다. main N3를 바꾼 오류가 아니며 이번 새 학습은0이다. '+link(f['supervision_repair'],a,'실제 감독 수정 기록')+'.',
        'V18의161 extra veto 제거는 서로 다른 cost의 viable 해까지 ambiguity로 막던 결정정책 복원이다. 실제 numerical tie1과 관측/기하 부족4는 유지했다. 원 V16 전체198134007검사 중564FAIL은 모두 선 경로이며 원 FAIL로 보존한다. point-only 실패 predicate0은 전체 point 산술검사 횟수나 물리 정확성 인증을 뜻하지 않는다.',
        'LOO는 H∪{k}를 실제 fit에서 제외하고 native²−candidate²의 strict gain과 기존8px를 검사한다. q/H는 ordinary hybrid·V18 restored·V19 line/point·V21 closure 다섯 hybrid 후단에서 동일하다. Native ordinary는 원 native N3 q이고 fixed Base/N3의 H/fit은 unrecorded다. 승인 NEW에서만 H를 최종 fit/관측에서 제외하고 최종 자세로 한 번 재투영하며 다시 fit하지 않는다. fallback은 원 all8 N3 산출이다. 마스크가 완벽하지 않아도 충분한 유효 관측과 실제 배치가 남으면 풀어야 한다.',
        'V21은 원 raw4/refit/cache 후보와 count→SSE→dimension→start 결정 규칙을 유지하면서 live whole-U ordinary 후보를 고정적으로 추가했다. 성능을 보고 threshold·seed·학습·case를 바꾼 반복은 아니다. 동일 q의 pixel SSE 비악화가 proxy T/R 비악화를 보장하지 않는 부분은 실제 V22 routing 진단으로 구분했다. '+link(f['r:SUMMARY.json'],a,'전체245 routing 및 frame IDs')+'.',
        table(['확인할 코드','실제 역할'],[[link(f['corner_definition'],a,'V2 observations.py',147),'각 native corner의 서로 다른 incident TLS line 교점 q'],
            [link(f['corner_LOO'],a,'V4 selection.py',130),'H∪k 제외한 held-out native/candidate 비교와 strict gain'],
            [link(f['solver'],a,'V21 solver.py',149),'고정 whole-U ordinary coverage 추가·원 count/SSE 순위·실제fit guard'],
            [link(f['pipeline'],a,'V21 pipeline.py',82),'동일 q/H·승인 NEW의 최종H 투영·원 fallback 보존'],
            [link(f['evaluator'],a,'V21 evaluate.py'),'full245 proxy 오류와 공통집합·고정13session bootstrap']]),
        '## 전체245·Clean153·Moderate92',
        '원319는 Clean153+Moderate92+Severe74이며, 현재 Severe74는 원 요청 범위에서 제외했다. 아래 NEW/fallback 분모와 평균은 모든245/153/92를 유지한다. Tcm/Rdeg/ADDsym cm 각각 평균·모집단SD(ddof0)·중앙값·P90이며, 표본분산/SD(ddof1)·최대·공통NEW·전체189CI는 [FULL_AUDIT_KO.md](FULL_AUDIT_KO.md)에 있다. fixed controls는 NEW가 아니다.',
        table(['층','방법','NEW','fallback','T평균','T SD0','T중앙','T P90','R평균','R SD0','R중앙','R P90','ADD평균','ADD SD0','ADD중앙','ADD P90'],brief),
        '현재 primary 전체 상태: '+cell(dict(denominator=primary_status['denominator'],operational=primary_status['operational'],NEW=primary_status['new_pose'],fallback=primary_status['fallback'],failure=primary_status['no_pose'],states=primary_status['statuses']))+'. 큰 오차를 완전 실패로 빼거나 NEW만 전체평균인 것처럼 쓰지 않았다.',
        '## 같은 frame에서 T/R/ADD 차이와 CI',
        '앞−뒤 음수는 개선이다. 아래는 combined/common_operational이다. 모든 CI가0을 가로지른다고 일반화하지 않고 각 저장 CI를 그대로 표시한다. 후보NEW·양측NEW·층별63 paired groups/189CI의 서로 다른 분모와 frame IDs는 전체 부록에서 확인한다.',
        table(['층','차이','scope','지표','공통 n','평균차','95% CI','유효 resamples','빈 resamples','단위'],common),
        '## 마스크 오류와 남은 올바른 관측',
        'UNKNOWN/UNANNOTATED는 새 의미상 known mask 비교에서 제외한다. 기존 legacy mask audit은 별도 보존한다. CORRECT_WITHIN8PX는 저장 GEOMETRIC_PROXY의 수치 품질이며 physical truth가 아니다. DIRECT_VISIBLE와 proxy≤8px의 교집합도 독립 physical ownership 인증이 아니다.',
        table(['마스크 상태','N3 대비 실제 T/R','n','NEW','fallback'],masks),
        'eligible/solver_used/승인 fit/승인 final inlier의 전체 ID·GOOD/BAD/UNKNOWN·DIRECT_VISIBLE proxy GOOD 수, 잘못 남긴 head q, 실제 fit 배치·저장 rank/spectrum은 [FULL_AUDIT_KO.md](FULL_AUDIT_KO.md)에 모두 전사했다. rank6/weak 통과와 전역 유일해·정확한 semantic 대응은 별개다.',
        '## 직접 가시점 손상과 H',
        '직접 손상은 저장 N3≤5px→output>10px flag다. SELF_OCCLUDED와 실제 H재투영에 이 flag가 없는 None을 새 측정0으로 해석하지 않는다. H 점의 오차 개선만으로 R,t 개선이라고 결론내리지 않는다.',
        table(['집합','단계','코너','직접 손상 flag','n','평균 px','표본SD','중앙값','P90'],corners),
        '## 고정 합성 stress9: 자연 solver의 별도 검사',
        '잘못 제외한 GOOD와 남긴 BAD를 독립적으로 조합하고 같은 실제 solver를 호출한 고정9fixture다. synthetic analytic truth는 원 실사 독립 물리 GT가 아니다. contract PASS는 모든 fixture의 NEW나 실사 자세 성공을 뜻하지 않는다. fallback을 공급하지 않아 typed 부족/퇴화도 그대로 남겼다.',
        table(['fixture','잘못 제외GOOD H','남긴BAD','남은GOOD','NEW','state','실제fit','최종inliers','synthetic T/R/ADD','계약오류'],stress_rows),
        '## V22: 후보 coverage와 실제 출력 routing',
        '아래 membership/readout은 이미 저장된 자세·error에서 독립 전사/진단한 값이다. routing조건부 평균은 새 전체 성능·새CI가 아니다. accepted와 selected winner, stored best와 승인 NEW를 구분한다.',
        table(['routing','frame n'],routing['routing_counts'].items()),
        table(['ordinary branch 판정','frame n'],routing['ordinary_branch_routing_counts'].items()),
        '첫 V22 실행은 exit137로 종료됐으며 원 PROTOCOL/STARTED/exit note를 보존했다. full actual_pose packet을 오래 보유하는 source 메모리 배치를 바꾼 ready는4개 AST영역 역변환 동일성으로 분리했다. 실제 exit137 원인은 확정하지 않으며 새 pose·ranking·gate·summary 식·threshold를 바꾼 수정이 아니다.',
        '## 실제150회 전체 경로 시간',
        'detector→N3/cornerSubPix→initial N3·Base feature pose→고정 IMAGE_ROLE/CAL→corner H∪k LOO→hybrid point/native roots/cache→whole-U 후보·실제 refit/guard→H재투영·metadata 모두 t0–t1 내부다. 모델 로딩·26기존 RGB decode·post-return parity·로그·resources는 제외했다. warm20+measured130, 13session×5repeat 고정패널이다. 첫 predict detector2/나머지149×1과 N3/head150을 실제 계수한다. saved geometry replay elapsed나 단계 평균의 합산은 latency가 아니다.',
        'V21 runtime original status='+cell(e['runtime']['status'])+', statistics_official='+cell(e['runtime']['statistics_official'])+', independent passed='+cell(e['runtime_checks']['passed'])+'. V19의150행 cleanup 실패/overallFalse/비공식4374.615ms는 원 기록으로 보존하고 V21의 시간으로 대체하지 않는다.'] )
    if e['runtime_clean']:
        main.append(table(['단계','n','평균 ms','표본분산','표본SD','중앙값','P90','최대'],[[k,*[e['runtime_checks']['summaries'][k][q] for q in ('n','mean','sample_variance','sample_std','median','P90','maximum')]] for k in TIMES]))
    else: main.append('**이번 runtime은 실패 기록이다. clean 배포 시간이나 공식 summaries를 주장하지 않는다.** '+link(f['t:RUNTIME_RECEIPT.json'],a,'원 실패 receipt')+' / '+link(f['c:RUNTIME_CHECKS.json'],a,'실제 검산 실패 범위')+'.')
    main.extend(['## 고정6사례·실제9PNG',
        '3Clean+3Moderate를 결과에 따라 다시 고르지 않았다. ROOT가9PNG/18panel을 실제 열어 가독성을 확인했다. N3의 H(pred)=[] 또는 unrecorded는 실제 가림 없음이나 빈 fit 인증이 아니다. 실제 q·fit·H·NEW/fallback·정밀 raw값은 그대로이며 caption 반올림은 표시만이다. 시각 확인은 physical pose 인증이 아니다.'])
    main.extend('!['+Path(item['path']).name+']('+label(item['path'],a)+')' for item in e['figures']['figures'])
    main.extend(['## 검산 범위와 다음 최소 확인',
        '전체 기하 '+cell(e['validation']['checks'])+' checks/FAIL '+cell(e['validation']['failure_count'])+', saved 통계783분포/6264필드/63paired/189CI, routing '+cell(e['routing_checks']['checks'])+' checks/FAIL '+cell(e['routing_checks']['failure_count'])+'다. 전체 core 산술과 따로 생성한 proxy pool/diagnostic readout의 검산 범위를 혼동하지 않는다.',
        '현재 남은 문제는 후보 coverage·실제 semantic 대응/배치·pixel objective와 proxy pose의 관계다. 원 관측이 맞아도 imperfect mask 뒤 유효 대응과 관측가능성이 충분하면 풀어야 한다. 저장된 routing별 같은-q ordinary 대비 frame IDs/fit BAD·UNKNOWN·직접가시 GOOD를 먼저 대조해야 하며 현재 수치만으로 head나 가림분류기의 새 학습 필요성을 증명하지 않는다. 기존 무가림 robust 대조는 별도 역사 자료다. P0 외부 가림·비가림 distractor·저대비 네 통제 변형은 별도 미검증이다.',
        table(['역사 대조(전체245)','NEW','fallback','T평균 cm','R평균 deg','ADD평균 cm'],[
            ['고정 N3 baseline',None,None,9.754548,10.914842,17.028647],
            ['V3 N3_BASIN_NO_MASK_ROBUST',229,16,9.65211,11.01512,16.93721],
            ['V7 N3_INDEPENDENT_ROBUST_NO_MASK',244,1,11.59638,13.89985,20.51034]]),
        '위 no-mask 값은 '+link(f['historical_no_mask_v3'],a,'게시된 V3')+' 및 '+link(f['historical_no_mask_v7'],a,'게시된 V7')+'의 별도 기존 알고리즘·q/model controls다. 현재 V21과 backend·q가 같은 masked/no-mask 인과 대조가 아니며, 이 두 역사값 또는 이번 평균으로 학습 head나 가림분류기의 필요성을 증명하지 않는다.',
        '참조 R,t는 원 annotation XY+K+등록 cuboid에서 재구성한 GEOMETRIC_PROXY다. N3 pose를 정답으로 직접 되먹인 값은 아니지만 annotation에는 unknown/manual_click/pnp_projected가 섞여 전점 독립 사람 클릭·실측 GT가 아니다. N3-phase는 번호 정렬에 조건부다. sourceCAL virtual/native348corner 차이도 실사 ownership 인증이 아니다.',
        '전체 부록: [FULL_AUDIT_KO.md](FULL_AUDIT_KO.md). '+link(f['e:METRICS.json'],a,'실제 METRICS')+' · '+link(f['s:STATISTICS_CHECKS.json'],a,'독립 통계')+' · '+link(f['r:CHECKS.json'],a,'V22 진단')+' · '+link(f['f:ROOT_VISUAL_REVIEW.json'],a,'실제 그림 검토')+'. main 변경·새 학습·새RGB·force push를 수행했다는 주장은 없다. 이 report 생성은 공개 복원/portable replay 또는 새 원격 commit 완료 receipt가 아니다.'])
    full.extend(['# V21 전체 감사 부록','주 보고서 [RESULT_KO.md](RESULT_KO.md)의 모든 표는 저장 결과 전사다. 아래783분포×8 moment필드=6264와189CI는 전체 strata/scope를 포함한다. 표본 ddof1와 모집단 ddof0를 혼동하지 않는다. 빈집합 None은0이 아니다.',
        '## 전체24개 상태',table(['층','방법','분모','산출','NEW','fallback','완전실패','상태'],statuses),
        '## Marginal216분포',table(['층','방법','scope','지표',*MOMENTS,'max','단위'],marginal),
        '## Matched567분포',table(['층','차이','scope','지표','역할',*MOMENTS,'max','단위'],matched),
        '## 전체189CI',table(['층','차이','scope','지표','공통 n','평균차','95% CI','유효 resamples','빈 resamples','단위'],ci),
        '## 모든 저장 diagnostic 요약',table(['층','primary 진단'],[[name,z['diagnostics'][PRIMARY]] for name,z in m['strata'].items()]),
        '## V22 모든 strata/routing/readout/정확한 frame IDs',table(['층','저장 routing summary'],e['routing']['strata'].items()),
        '## 모든 실제 실행·검산 receipts',table(['권위','저장 값'],[[key,e[key]] for key in ('geometry','validation','statistics','runtime','runtime_checks','stress','cpu','routing_checks')]),
        '## 전체245의 남은 proxy 대응과 실제 배치',
        '아래 membership은 저장 corners/pools/fit ID에서만 읽는다. known visibility와 numerical GOOD, DIRECT_VISIBLE는 각각 별개다. UNKNOWN은 NONE이 아니다. 저장 rank/spectra를 새로 계산하지 않는다.'])
    diagnostics=[]
    for row in posthoc_rows(f['e:POSTHOC_ROWS.jsonl.gz'],m['strata']['combined']['ids']):
        corners_by_id={c['id']:c for c in row['corners']};require(set(corners_by_id)==set(range(8)),'eight stored corner records')
        pool_data={}
        for name,pool in row['pools'].items():
            ids=pool['ids'];good=[k for k in ids if corners_by_id[k]['input_quality']=='CORRECT_WITHIN8PX']
            pool_data[name]=dict(saved=pool,proxy_correct_ids=good,
                known_visibility_proxy_correct_ids=[k for k in good if corners_by_id[k]['human_state'] not in ('UNKNOWN','UNANNOTATED')],
                directly_visible_proxy_correct_ids=[k for k in good if corners_by_id[k]['human_state']=='DIRECT_VISIBLE'])
        diagnostics.append([row['id'],row['session'],row['output_status'],row['new_pose_estimated'],row['fallback_used'],row['mask_state'],row['pose_outcome_vs_N3'],
            pool_data,row['hidden_initial'],row['actually_reprojected_ids'],row['finite_geometry_diagnostics'],row['boundary_candidates'],row['corners']])
    full.append(table(['id','session','status','NEW','fallback','mask','N3 대비T/R','저장pools 및 GOOD IDs','initialH','실제Hprojection','실제fit/rank/closure','head 후보','전체8corners'],diagnostics))
    full.extend(['## 원행·코드·원 실패 보존',table(['alias','정확한 공개 경로'],[[k,link(p,a)] for k,p in f.items()]),
        '이 report는 이전 V19 main/full의 원 SHA를 결속하고 원문을 수정하지 않았다. 원 V19 JSON직렬화 실패2solve/46optimizer/0serialized와 JSON-only common.finite(row/ledger)2-site 복원 증거, 원 runtime_cleanup FAIL과 비공식 행 검산은 V19 [FULL_AUDIT_KO.md](../pallet_finite_support_orthogonal_audit_20261011_v19/FULL_AUDIT_KO.md)에서 확인한다. 모든 기하 replay elapsed는 새 초기 RGB 전체경로 latency가 아니다. 새 발표/원격 SHA는 별도 actualreceipt에서 확인해야 한다.'])
    counts=dict(status_rows=len(statuses),marginal_distributions=len(marginal),matched_distributions=len(matched),moment_fields=6264,
        paired_CI_rows=len(ci),paired_groups=63,POSTHOC_rows=len(diagnostics),PNG_images=9,case_panels=18,
        runtime_official=e['runtime_clean'],runtime_summary_slots=56 if e['runtime_clean'] else 0,
        routing_frames=e['routing']['frames'],fixed_synthetic_cases=len(stress_rows),new_scientific_computations=0)
    return '\n\n'.join(main)+'\n','\n\n'.join(full)+'\n',counts

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('freeze','run'));p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--private',type=Path,default=PRIVATE);p.add_argument('--history',type=Path,default=HISTORY);p.add_argument('--routing-private',type=Path,default=ROUTING)
    p.add_argument('--family',required=True);p.add_argument('--routing-family',required=True)
    defaults={'geometry_input':'geometry','validation_input':'validation','evaluation_input':'evaluation','statistics_input':'statistics',
        'runtime_input':'runtime','runtime_checks_input':'runtime_checks','render_input':'render','stress_input':'stress','cpu_input':'cpu','output':'report'}
    for name,folder in defaults.items():p.add_argument('--'+name.replace('_','-'),type=Path,default=PRIVATE/folder)
    p.add_argument('--routing-input',type=Path,default=ROUTING/'audit_ready');a=p.parse_args()
    for key,value in vars(a).items():
        if isinstance(value,Path):setattr(a,key,value.resolve())
    for family in (a.family,a.routing_family):require(Path(family).name==family and family not in ('.','..'),'public family basename')
    require(a.output.is_relative_to(a.private) and a.output!=a.private,'exclusive private report output')
    f=paths(a)
    require(not any(Path(v).is_relative_to(a.output) or a.output.is_relative_to(Path(v)) for v in f.values()),'disjoint inputs/output')
    b={k:binding(v) for k,v in f.items()};e=gate(a,f,b)
    a.output.mkdir(parents=True,exist_ok=True)
    for name in FILES if a.stage=='freeze' else FILES[1:]:require(not(a.output/name).exists(),'preserve '+name)
    if a.stage=='freeze':write(a.output/FILES[0],dict(schema='saved_whole_U_report_protocol_v21',inputs=b,checker=b['reporter'],limits=LIMITS,
        public_families=dict(V21=a.family,V22=a.routing_family),frozen_before_transcription=True));return
    own=read(a.output/FILES[0]);require(own['schema']=='saved_whole_U_report_protocol_v21' and own['inputs']==b and own['limits']==LIMITS
        and own['public_families']==dict(V21=a.family,V22=a.routing_family),'exact frozen report inputs')
    ob=binding(a.output/FILES[0]);write(a.output/FILES[1],dict(inputs=b,own_protocol=ob,checker=b['reporter'],new_production_calls=0));error=None;counts={}
    try:
        text,appendix,counts=reports(a,f,e);require(len(text.splitlines())<=300,'main report exceeds300 lines')
        require(b=={k:binding(v) for k,v in f.items()},'all frozen report inputs unchanged')
        for name,value in ((FILES[3],text),(FILES[4],appendix)):
            with (a.output/name).open('x',encoding='utf-8') as stream:stream.write(value);stream.flush();os.fsync(stream.fileno())
    except BaseException as exc:error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
    write(a.output/FILES[2],dict(schema='saved_whole_U_report_checks_v21',complete=error is None,passed=error is None,
        inputs=b,own_protocol=ob,checker=b['reporter'],error=error,transcription_counts=counts,
        reports={n:binding(a.output/n) for n in FILES[3:] if (a.output/n).is_file()},limits=LIMITS,new_production_calls=0,
        actual_runtime_contract_passed=e['runtime_clean'],publication_complete=False))
    if error:raise SystemExit(1)

if __name__=='__main__':main()
