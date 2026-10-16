"""Freeze and transcribe completed saved evidence into four exclusive reports.

This helper never decodes scored/geometry gzip rows or computes statistics,
confidence intervals, poses, projections, Jacobians or hardware timings.
Late reports and reader proofs are outside the earlier frozen byte bundle.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import traceback

REPO=Path(__file__).resolve().parents[3];CODE=Path(__file__).resolve().parent
DOC=REPO/'_docs/experiments/pallet_consensus_decision_repair_20261011_v18'
OLD=REPO/'_docs/experiments/pallet_hybrid_standard_controls_20261011_v17'
V16=REPO/'_docs/experiments/pallet_hybrid_factor_subset_consensus_20261011_v16'
FILES=('REPORT_BUILD_PROTOCOL.json','REPORT_BUILD_STARTED.json','REPORT_BUILD_CHECKS.json')
TARGETS=(DOC/'RESULT_KO.md',DOC/'README.md',DOC/'BUILD_LEDGER.md',OLD/'PUBLIC_REVIEW_KO.md')
METHODS=('BASE','N3_SUBPIX','HYBRID_STANDARD_PNP','NATIVE_STANDARD_PNP','HYBRID_POINT_CONSENSUS_ORIGINAL','HYBRID_POINT_CONSENSUS_RESTORED')
PRIMARY=METHODS[-1];STRATA=('combined','easy','medium');METRICS=('translation_cm','rotation_deg','ADDsym_cm')
CONTRASTS=((PRIMARY,METHODS[-2]),(PRIMARY,'N3_SUBPIX'),(PRIMARY,'NATIVE_STANDARD_PNP'),(PRIMARY,'HYBRID_STANDARD_PNP'),(METHODS[-2],'N3_SUBPIX'))
SCOPES=('common_operational','candidate_new_pose','both_new_pose');MOMENTS=('mean','sample_variance','sample_std','median','P90','max')
PROOFS=('V17_statistics','V17_runtime','V18_statistics','V18_runtime')
LIMITS=dict(new_moment_CI_pose_projection_J_SVD_GT_model_training_RGB_timing_calls=0,
    raw_gzip_rows_decoded=0,diagnostic_moments_independently_recomputed=False,
    late_reports_claimed_copied_in_earlier_bundle=False,Git_clone_claim=False,automatic_retry=False)

def require(value,message):
    if not value:raise ValueError(message)
def safe(path):
    p=Path(path).absolute();require(not any(q.is_symlink() for q in (p,*p.parents)),'symlink '+str(p));return p
def read(path):
    with safe(path).open(encoding='utf-8') as f:return json.load(f)
def same(a,b):return all(a.get(k)==b.get(k) for k in ('bytes','sha256'))
def bind(path):
    p=safe(path);h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return dict(path=str(p.relative_to(REPO)),bytes=p.stat().st_size,sha256=h.hexdigest(),origin='public_repository')
def declared(value):
    p=Path(value['path']);require(not p.is_absolute() and '..' not in p.parts and p.parts and p.parts[0] in ('scripts','_docs'),'full public proof path')
    p=safe(REPO/p);require(same(bind(p),value),'public proof bytes differ '+str(p));return p
def write(path,value):
    with safe(path).open('x',encoding='utf-8') as f:
        f.write(value if isinstance(value,str) else json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
def text(value):return 'None' if value is None else json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':')) if not isinstance(value,str) else value.replace('|','\\|').replace('\n',' ')
def link(path,label=None,folder=DOC):
    p=Path(path);p=p if p.is_absolute() else REPO/p
    return '['+(label or p.name)+']('+os.path.relpath(p,folder)+')'
def table(headers,rows):return '\n'+'|'+'|'.join(headers)+'|\n|'+'|'.join('---' for _ in headers)+'|\n'+''.join('|'+'|'.join(text(v) for v in row)+'|\n' for row in rows)+'\n'
def block(value):return '\n```json\n'+json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n```\n'

def reader_gate():
    paths={'reader':DOC/'PUBLIC_READER_CHECKS.json','bundle_checks':DOC/'PUBLIC_BUNDLE_CHECKS.json','bundle_manifest':DOC/'PUBLIC_BUNDLE_MANIFEST.json',
        'bundle_protocol':DOC/'PUBLIC_BUNDLE_PROTOCOL.json','bundle_started':DOC/'PUBLIC_BUNDLE_STARTED.json','bundle_code':CODE/'public_bundle_reader_closure.py'}
    reader=read(paths['reader']);bundle=read(paths['bundle_checks']);manifest=read(paths['bundle_manifest'])
    require(reader['schema']=='public_saved_reader_checks_v18' and reader['complete'] is True and reader['passed'] is True,'four actual copied readers required')
    require(bundle['schema']=='paired_standard_consensus_public_bundle_checks_v18' and bundle['complete'] is True and bundle['passed'] is True and bundle['error'] is None
        and bundle['target_initially_absent'] is True and manifest['original_public_bytes_preserved'] is True,'actual initially absent byte bundle PASS required')
    require(same(reader['bundle_checks'],bind(paths['bundle_checks'])) and same(reader['bundle_manifest'],bind(paths['bundle_manifest'])) and same(bundle['manifest'],bind(paths['bundle_manifest'])),'reader/bundle authority')
    require(same(bundle['own_protocol'],bind(paths['bundle_protocol'])) and same(read(paths['bundle_started'])['own_protocol'],bind(paths['bundle_protocol']))
        and same(bundle['code'],bind(paths['bundle_code'])),'actual bundle own protocol/STARTED/current code')
    require(reader['limits']==dict(new_model_PnP_optimizer_projection_J_SVD_GT_training_RGB_calls=0,fresh_saved_scalar_arithmetic_executed=True,
        hardware_remeasurement=False,Git_clone_claim=False,original_full_V16_scalar_PASS_claim=False),'saved-reader limits differ')
    require(set(reader['proofs'])==set(PROOFS),'exact four reader groups')
    for name in PROOFS:
        proof=reader['proofs'][name];version=17 if name.startswith('V17') else 18;stats=name.endswith('statistics')
        p,s,c=[declared(proof[k]) for k in ('protocol','started','checks')];own,start,check=[read(q) for q in (p,s,c)]
        paths.update({name+':'+k:q for k,q in zip(('protocol','started','checks'),(p,s,c))})
        require(check['complete'] is True and check['passed'] is True and check['failure_count']==0 and check['inputs']==own['inputs']==start['inputs'],'actual copied complete PASS/input closure')
        if stats:
            prefix='independent_native_hybrid_standard_statistics' if version==17 else 'independent_point_consensus_decision_statistics'
            expected=dict(saved_scored_rows_streamed=980 if version==17 else 1470,moment_scalars_checked=648 if version==17 else 972,
                distributions_checked=108 if version==17 else 162,pair_groups_checked=27 if version==17 else 45,metric_CI_slots_checked=81 if version==17 else 135)
            require(own['schema']==prefix+'_protocol_v'+str(version) and check['schema']==prefix+'_checks_v'+str(version) and check['error'] is None
                and all(check['actual_counts'][k]==v for k,v in expected.items()),'copied actual statistics topology')
            require(same(check['protocol'],bind(p)) and same(start['protocol'],bind(p)),'copied statistic protocol')
            checker=REPO/'scripts/research'/('pallet_hybrid_standard_controls_20261011_v17' if version==17 else CODE.name)/'verify.py'
        else:
            prefix='independent_hybrid_standard_runtime' if version==17 else 'independent_restored_point_consensus_runtime'
            schema=prefix+'_check_protocol_v17' if version==17 else prefix+'_protocol_v18'
            require(own['schema']==schema and check['schema']==prefix+'_checks_v'+str(version) and check['exception'] is None
                and check['rows']==150 and check['measured']==130 and check['summary_blocks']==8 and check['summary_slots']==56,'copied actual saved-time topology')
            require(same(check['own_protocol'],bind(p)) and same(start['own_protocol'],bind(p)),'copied time protocol')
            checker=REPO/'scripts/research'/('pallet_hybrid_standard_controls_20261011_v17' if version==17 else CODE.name)/('runtime_check.py' if version==17 else 'runtime_checks.py')
        require(same(check['checker'],bind(checker)),'copied reader/current public checker bytes');paths[name+':checker']=checker
    return paths

def snapshot():
    paths=reader_gate()
    for name in ('PROTOCOL.json','CONTROL_RECEIPT.json','GEOMETRY_SEAL.json','VALIDATION_PROTOCOL.json','VALIDATION_STARTED.json','VALIDATION_CHECKS.json',
        'EVALUATION_PROTOCOL.json','SCORING_AUTHORITY.json','SCORING_STARTED.json','SCORING_RECEIPT.json','STATISTICS_STARTED.json','METRICS.json',
        'STATISTICS_VERIFICATION_PROTOCOL.json','STATISTICS_VERIFICATION_STARTED.json','VERIFICATION.json','RUNTIME_PROTOCOL.json','RUNTIME_RECEIPT.json',
        'RUNTIME_CHECK_PROTOCOL.json','RUNTIME_CHECK_STARTED.json','RUNTIME_CHECKS.json','FIGURE_PROTOCOL.json','FIGURE_BINDINGS.json','ROOT_VISUAL_REVIEW.json','ARCHIVE_MANIFEST.json'):
        paths[name]=DOC/name
    for name in ('FIRST_SCALAR_ENVIRONMENT_FAILURE.json','VALIDATION_READY_CHECKS.json','TRACE_VALIDATION_CHECKS.json','FAILURE_WITNESS_SUMMARY.json','FAILURE_SUMMARY_CHECKS.json'):
        paths['V16:'+name]=V16/name
    m=read(paths['METRICS.json']);v=read(paths['VERIFICATION.json']);runtime=read(paths['RUNTIME_RECEIPT.json']);timing=read(paths['RUNTIME_CHECKS.json']);geo=read(paths['VALIDATION_CHECKS.json'])
    require(m['schema']=='point_consensus_decision_raw_row_metrics_v18' and m['complete'] is True and m['methods']==list(METHODS)
        and m['contrasts']==[list(c) for c in CONTRASTS] and m['V16_full_scalar_validation_passed'] is False
        and all(m['strata'][k]['frames']==n for k,n in (('combined',245),('easy',153),('medium',92))),'actual six-method same245 metrics')
    require(v['schema']=='independent_point_consensus_decision_statistics_checks_v18' and v['complete'] is True and v['passed'] is True
        and v['error'] is None and v['failure_count']==0 and same(v['inputs']['METRICS'],bind(paths['METRICS.json']))
        and all(v['actual_counts'][k]==n for k,n in dict(saved_scored_rows_streamed=1470,moment_scalars_checked=972,distributions_checked=162,
        pair_groups_checked=45,metric_CI_slots_checked=135).items()),'actual independent metric PASS')
    require(geo['schema']=='point_consensus_decision_scalar_checks_v18' and geo['complete'] is True and geo['passed'] is True
        and geo['failure_count']==0 and geo['exception'] is None and geo['new_production_calls']==0 and all(geo['scope'][k]==value for k,value in dict(original_full_scalar_validation_passed=False,
        point_only_original_failed_predicates=0,independent_parent_point_check_count_not_claimed=True).items()),'actual narrow point-only PASS, not full V16 PASS')
    for kind,path in (('statistic',paths['VERIFICATION.json']),('runtime',paths['RUNTIME_CHECKS.json'])):
        check=read(path);fresh=read(paths['V18_'+('statistics' if kind=='statistic' else 'runtime')+':checks'])
        require(check['inputs']==fresh['inputs'] and same(check['checker'],fresh['checker']),'fresh readers check identical production evidence')
    require(runtime['complete'] is True and runtime['statistics_official'] is True and runtime['serialized_rows']==150 and not runtime['cleanup_errors']
        and runtime['all_recorded_resource_snapshots_quiet'] is True and timing['schema']=='independent_restored_point_consensus_runtime_checks_v18'
        and timing['complete'] is True and timing['passed'] is True and timing['failure_count']==0 and timing['exception'] is None
        and timing['rows']==150 and timing['measured']==130 and timing['summary_blocks']==8 and timing['summary_slots']==56,'actual full fresh runtime/independent PASS')
    require(same(timing['inputs']['receipt'],bind(paths['RUNTIME_RECEIPT.json'])),'actual saved runtime receipt bytes')
    figures=read(paths['FIGURE_BINDINGS.json']);visual=read(paths['ROOT_VISUAL_REVIEW.json'])
    require(figures['complete'] is True and figures['PNG_files']==8 and figures['panels']==24 and figures['source_display_records']==1470
        and visual['actual_view_image_calls']==8 and visual['reviewed_case_panels']==24,'actual eight rendered/viewed images')
    require(len(figures['figures'])==8 and len(visual['reviewed_files'])==8,'eight fixed images/reviews')
    require(set(str(Path(v['path']).relative_to(DOC.relative_to(REPO))) for v in figures['figures'])==set(visual['reviewed_files']),'Root actually reviewed these eight figure names')
    for i,value in enumerate(figures['figures']):paths['figure:'+str(i)]=declared(value)
    paths['builder']=Path(__file__)
    return dict(inputs={k:bind(p) for k,p in paths.items()},limits=LIMITS,targets=[str(p.relative_to(REPO)) for p in TARGETS])

def documents():
    m=read(DOC/'METRICS.json');geo=read(DOC/'VALIDATION_CHECKS.json');v=read(DOC/'VERIFICATION.json');receipt=read(DOC/'CONTROL_RECEIPT.json')
    runtime=read(DOC/'RUNTIME_RECEIPT.json');timing=read(DOC/'RUNTIME_CHECKS.json');figures=read(DOC/'FIGURE_BINDINGS.json');reader=read(DOC/'PUBLIC_READER_CHECKS.json');bundle=read(DOC/'PUBLIC_BUNDLE_CHECKS.json')
    verdict={k:m[k] for k in ('primary_both_T_R_means_better_than_N3_on_full245','primary_both_T_R_means_better_than_original_on_full245',
        'primary_both_T_R_means_better_than_native_standard_on_full245','primary_both_T_R_means_better_than_hybrid_standard_on_full245')}
    history_note='공개 묶음의 '+link(DOC/'FIRST_PUBLIC_BUNDLE_FREEZE_FAILURE.json','첫 freeze 실패')+'와 '+link(DOC/'FIRST_READY_PUBLIC_BUNDLE_FREEZE_FAILURE.json','READY freeze 실패')+'는 own protocol·복사 target 생성 전이며 복사/새 수치·모델·GT·fit 호출 0으로 보존했다. 원 V2 '+link('_docs/experiments/pallet_boundary_corner_refiner_20261010_v2/PIPELINE_CHECKS.json','첫 CPU11 PASS 영수증')+'의 $.code와 $.checks_code 두 이전 소스 선언은 조사한 reachable Git path history에서 원 source가 없어 역사 metadata로 남겼으며, 첫 11 PASS를 현재 code로 새 재현했다고 주장하지 않는다. '+link('_docs/experiments/pallet_boundary_corner_refiner_20261010_v2/PIPELINE_CHECKS_FINAL.json','FINAL 영수증')+'과 현재 두 source는 별도로 그대로 결속한다. '+link(CODE/'public_bundle_reader_closure.py','공개 reader 입력 묶음 실행기')+'는 실제 reader4의 own protocol 필수 입력 집합을 권위로 삼아 현재 SHA를 엄격히 확인한다. 과거 보호 목록·영수증 JSON 자체의 byte는 보존하되, 그 안의 역사 선언을 현재 자산 의존성으로 재귀 해석하지 않는다. 두 실험 tree와 명시한 현재 source/kernel도 결속하며, 없는 파일이나 알 수 없는 origin을 일반적으로 무시하지 않는다. 원 bundle·READY source와 실패 영수증은 수정하지 않았다.\n'
    lines=['# V18: 같은 q의 합의 결정 복원과 실제 자세 결과\n',
        '“가림 판단이 일부 틀려도 다른 유효 대응점으로 자세를 구하고 자기 가림 코너를 재투영했을 때, 기존 단순 대조보다 실제 위치·회전이 좋아졌는가?”\n',
        '전체 245장의 저장된 양축 평균 판정: N3 대비 '+('위치·회전 평균이 모두 개선됐다' if verdict['primary_both_T_R_means_better_than_N3_on_full245'] else '위치·회전 평균이 모두 개선됐다고 확인하지 못했다')+'; 단순 native ordinary 대비 '+('두 평균이 모두 개선됐다' if verdict['primary_both_T_R_means_better_than_native_standard_on_full245'] else '두 평균이 모두 개선됐다고 확인하지 못했다')+'. 가림 오판 자체를 실패 조건으로 쓰지 않았다. 이는 Clean153·Moderate92의 알려진 DEV GEOMETRIC_PROXY 평가이며, 독립 물리 실측·미관측 실사 성능을 인증하지 않는다.\n',block(verdict),
        'N3와 native ordinary, 같은 q의 ORIGINAL을 각각 대조한다. RESTORED−ORIGINAL은 동일 관측·동일 후보에서 추가 veto를 제거한 결정 효과다. V17 HYBRID−NATIVE의 같은 ordinary backend 비교가 학습 q의 별도 대조이며, backend가 다른 RESTORED−NATIVE를 head 단독 효과로 해석하지 않는다.\n',
        '## 1. 무엇을 정의했고 어디가 잘못됐는가\n',
        '새 head의 표적은 query normal과 의도한 semantic physical wire의 대응 위치 또는 NONE이다. 수치 ray miss를 NONE으로 잘못 감독한 2,164 query는 별도 브랜치의 '+link('scripts/research/pallet_observation_refiner_20261009_v1/training.py','옛 targets():91–94')+' 문제이며 main N3→cornerSubPix 문제가 아니다. '+link('scripts/research/pallet_kp_supervision_gate_20261010_v1/full_source_prepare.py','mesh/wire POSITIVE·IGNORE 준비:98–110')+'와 '+link('scripts/research/pallet_kp_supervision_repair_20261010_v1/recover_front_none_v3.py','depth로 인증한 NONE:109')+' 및 CPU gradient 검사를 완료했다. 최초 읽기 전용 감사 당시 재학습은 없었고, 이후 승인된 수정 감독 3조건×3,000=9,000 update는 '+link('_docs/experiments/pallet_kp_corrected_supervision_20261010_v1/RESULT_KO.md','별도 역사 결과')+'다. 이번 새 학습·RGB·seed는 0이다. 수정 전 IMAGE_ROLE 0/319를 수정 후 결과로 쓰지 않는다.\n',
        'q는 임의 ray hit를 corner로 넣지 않고, '+link('scripts/research/pallet_boundary_corner_refiner_20261010_v2/observations.py','서로 다른 incident edge의 TLS 두 선 교점')+'으로 만든다. source의 owned-wire 감독과 실사 decoder의 physical ownership은 다르며 후자는 UNKNOWN이다. 역할 특성은 Base 초기 자세의 projected hull이고, source role parity는 과거 pose witness 또는 실사 ownership을 인증하지 않는다.\n',
        '이번 실제 오류는 같은 point consensus support의 다른 비용 해를 모두 ambiguity로 거부한 추가 veto였다. 161 frame의 이 veto만 원래 inlier 수→truncated SSE 결정으로 복원했다. 실제 수치 tie 1개와 대응/실제 관측 기하 부족 4개는 fallback으로 남았다. 후보·refit·q·H·head·CAL·8px·rank/weak/depth threshold는 바꾸지 않았다. '+link(CODE/'solver.py','결정 복원 코드')+' / '+link(DOC/'METHOD_CONTRACT_KO.md')+' / '+link(DOC/'CONTROL_LEDGERS.jsonl.gz','245 결정 원행')+'\n',
        '기존 H∪k LOO는 H와 시험 코너 k의 초기 좌표를 제외한 다른 관측으로 native와 q의 엄격한 개선을 시험한다. 남은 유효 대응으로 finite 4점 subset 합의와 실제 inlier geometry를 확인한다. 마스크 불일치만으로 frame을 거부하지 않는다. 승인 NEW에서는 H 초기 XY를 최종 fit에서 빼고 새 R,t로 한 번 재투영해 교체한다. 재투영점을 다시 PnP 관측으로 넣지 않는다. fallback은 이미 계산된 원래 all8 N3 출력을 유지하며 새 H 제외 fit/NEW로 세지 않는다. 가림을 완벽하게 분류해야 성공한다는 조건은 없다.\n',
        'V16 원 scalar CLI는 isolated environment의 ast 누락으로 raw decode/Audit 이전 실패했고 '+link(V16/'FIRST_SCALAR_ENVIRONMENT_FAILURE.json')+'를 보존했다. 환경만 보완한 READY의 full 검사 198,134,007 predicates 중 564 FAIL도 '+link(V16/'TRACE_VALIDATION_CHECKS.json')+'에 보존했다. 모두 line 경로이며 point-only failure는 0이라는 저장 분리 증거를 사용했다. '+link(V16/'FAILURE_WITNESS_SUMMARY.json')+'는 21 context/15 frame, hinge364와 unmapped column-norm200을 구분한다. 승인 selected NEW/best의 해당 실패는 0이지만 전체 기하 PASS로 바꾸지 않는다. 새 검사는 paired point decision/output만 인증하고 원 parent point J 검사 수를 별도로 주장하지 않는다.\n',
        '## 2. 산출률과 전체 운용 결과\n']
    lines.append(table(('층','방법','전체','산출','NEW','fallback','완전 실패','fixed','T mean cm','R mean deg','ADDsym mean cm'),
        ((st,method,s['denominator'],s['operational'],s['new_pose'],s['fallback'],s['no_pose'],s['fixed_control'],*(s['metrics']['operational'][k]['mean'] for k in METRICS))
        for st in STRATA for method in METHODS for s in (m['strata'][st]['methods'][method],))))
    lines+=['큰 오차 frame을 삭제하지 않았다. operational 전체와 NEW·fallback 분포를 모두 남기며, 공통 산출/PRIMARY NEW/both NEW의 paired ID는 '+link(DOC/'METRICS.json')+'에 있다. NEW는 수치 출력 상태이며 정확한 자세라는 뜻이 아니다.\n','## 3. 전체 분포 162개: 원행에서 계산된 972개 수치\n']
    lines.append(table(('층','방법','출력집합','오차','n',*MOMENTS,'단위'),((st,method,scope,metric,val['n'],*(val[k] for k in MOMENTS),val['unit'])
        for st in STRATA for method in METHODS for scope in ('operational','new_pose','fallback') for metric in METRICS for val in (m['strata'][st]['methods'][method]['metrics'][scope][metric],))))
    lines+=['분산·표준편차는 표본 ddof=1이다. None은 빈 집합/표본 부족이며 0으로 채우지 않는다. '+link(DOC/'VERIFICATION.json','독립 972 수치 검산')+' / '+link(DOC/'METRICS.csv','원집계 CSV')+'\n','## 4. 45 paired 집합과 135 CI\n','앞−뒤의 음수는 개선이다. 기존 13 session의 고정 10,000 bootstrap draw를 재사용했고 새 draw/seed는 0이다. 특정 평균 개선을 모든 난이도·median·P90·모든 CI 개선으로 확대하지 않는다.\n']
    lines.append(table(('층','대조','paired 집합','분모','공통 n','candidate NEW','comparator NEW','오차','평균 차','95% CI','빈 resample','비어 있지 않은 resample'),
        ((st,a+'_minus_'+b,scope,p['denominator'],p['common_frames'],p['candidate_new_pose_in_pairs'],p['comparator_new_pose_in_pairs'],metric,d['mean_delta'],d['CI95'],p['bootstrap_empty_resamples'],p['bootstrap_nonempty_resamples'])
        for st in STRATA for a,b in CONTRASTS for scope in SCOPES for p in (m['strata'][st]['contrasts'][a+'_minus_'+b][scope],) for metric in METRICS for d in (p['metrics'][metric],))))
    lines+=['## 5. 마스크와 자세, 가시점 손상과 숨은 점을 분리\n','현재 score의 semantic known은 UNKNOWN/UNANNOTATED를 제외한다. 기존 V7 adapter의 legacy known은 UNKNOWN을 포함할 수 있으며 원 legacy audit도 보존했다. 이 정의 차이는 '+link(DOC/'POSTHOC_ROWS.jsonl.gz','490 진단 원행')+'의 mask_definition_provenance에 있다. 마스크 분류는 자세 성공 gate가 아니다. 아래 producer 진단은 독립 972/135 검산 범위와 구분하며 이 builder가 다시 계산하지 않았다.\n']
    lines.append(table(('층','방법','mask__pose vs N3','n','NEW','fallback'),((st,method,key,val['n'],val['new_pose'],val['fallback'])
        for st in STRATA for method in METHODS[-2:] for key,val in m['strata'][st]['diagnostics'][method]['mask_and_pose_groups_vs_fixed_N3'].items())))
    lines.append(table(('층','방법','코너집합','좌표','n',*MOMENTS,'단위','DIRECT ≤5→>10 손상'),((st,method,cat,coord,val['n'],*(val[k] for k in MOMENTS),val['unit'],diag.get('damage_5px_to_over10px') if cat=='DIRECT_VISIBLE' else '해당 없음')
        for st in STRATA for method in METHODS[-2:] for cat in ('DIRECT_VISIBLE','SELF_OCCLUDED','ACTUALLY_REPROJECTED') for diag in (m['strata'][st]['diagnostics'][method][cat],) for coord,val in diag['errors'].items())))
    lines+=['사람 SELF_OCCLUDED와 실제 NEW에서 재투영한 H는 분모가 다르다. 숨은 점 pixel 오차 개선만으로 T/R 개선을 주장하지 않는다. '+block({st:{method:{key:m['strata'][st]['diagnostics'][method][key] for key in ('boundary_supply_counts','adopted_boundary_quality','hidden_set_changed_frames')} for method in METHODS[-2:]} for st in STRATA}),
        '## 6. 전체 실제 경로 비용\n','정확도 replay는 새 fit/projection/J/SVD 0이다. 아래 150회는 detector→N3→초기 자세→Base role feature→head→H∪k LOO→point consensus/actual geometry→복원 결정→H 재투영/metadata를 매번 실제 실행했다. 20 warmup 이후 26개 고정 RGB×5=130회와 13 session이다. 첫 예측 내부 detector 준비 1회를 포함한 forward 계수를 보존한다. 모델 로딩·26 image decode·반환 후 saved parity·로그/자원 snapshot은 clock 밖이며 GPU sync를 포함한다. 캐시 replay wall time을 latency로 쓰지 않았다. 원 candidate bank의 같은 호출 내부 공유는 실제 계수로 기록한다.\n']
    lines.append(table(('단계','n','mean','sample variance','sample std','median','P90','maximum','단위'),((key,val['n'],*(val[k] for k in ('mean','sample_variance','sample_std','median','P90','maximum')),val.get('unit','ms')) for key,val in runtime['summaries'].items())))
    lines.append(block({k:runtime[k] for k in ('actual_base_counts','actual_fresh_counts','consensus_operation_counts','direct_point_operation_counts','stable_incidence_operation_counts','decision_counts','point_bank_capture_counts','packet_compatibility_counts','model_forwards','model_forwards_after_loading','live_head_counts','live_cv_counts','live_optimizer_counts','live_SVD_counts','cleanup','cleanup_errors','all_recorded_resource_snapshots_quiet')}))
    lines+=['실제 runtime checker predicates='+text(timing['checks'])+'; 150 rows/130 measured/8 blocks/56 slots. '+link(DOC/'RUNTIME_CHECKS.json')+' / '+link(DOC/'RUNTIME_ROWS.jsonl.gz','150 전체 경로 원행')+' / '+link(DOC/'RUNTIME_CONTRACT_KO.md')+'\n','## 7. 같은 사전 고정 영상 6개와 집계 그림 2개\n','이미지 선택은 새 오차를 보고 바꾸지 않았다. Root가 8 PNG와 24 case panel을 실제 열어 본 '+link(DOC/'ROOT_VISUAL_REVIEW.json')+'를 보존했다. cyan=출력, green+=공급 q, orange=승인 NEW H 재투영, white=평가 proxy이다. unrecorded H/fit은 immutable baseline 기록 부재이며 빈 baseline fit/가림 없음 판정이 아니다. N3 패널의 H(pred)=[]도 fixed-row 기본 metadata이며 실제 예측 가림 없음 인증이 아니다; 해당 baseline의 가림 추정은 unrecorded다. 시각적으로 그럴듯한 재투영은 physical ownership/정확도 보증이 아니다.\n']
    lines.extend('!['+Path(value['path']).name+']('+os.path.relpath(REPO/value['path'],DOC)+')\n' for value in figures['figures'])
    lines+=['## 8. 검산·공개 재현·남은 한계\n',table(('증거','완료값','범위'),((link(DOC/'VALIDATION_CHECKS.json'),geo.get('checks'),'point-only 결정/output; full V16 FAIL 유지'),(link(DOC/'VERIFICATION.json'),v['actual_counts'],'1470 원 score/972 moment/135 CI'),(link(DOC/'PUBLIC_READER_CHECKS.json'),'4 실제 PASS','copied V17/V18 저장 정확도·시간 검사; 새 하드웨어 측정 아님'))),
        '정답 R,t는 N3 예측이 아니라 '+link('scripts/paper/pose_metric_closure_v1/build_axis_review_manifest.py','기존 주석 XY 이전')+'와 '+link('scripts/paper/pose_metric_closure_v1/build_geometry_resolved_pose_gt.py','주석8점+K+등록 cuboid의 SQPnP/LM proxy 재구성')+'에서 왔다. 다만 주석에 PnP 채움이 섞일 수 있고 source/direct 클릭 tag를 GT builder가 모두 유지/필터하지 않으므로 독립 물리 실측·전점 독립 클릭을 보증하지 않는다. N3-phase는 corner 번호·human mask/pixel quality 정렬이며 R,t truth를 N3로 생성하는 경로가 아니다. known DEV라 unseen 성능을 주장하지 않는다.\n',
        '가림 없는 robust PnP 대조는 '+link('_docs/experiments/pallet_cornerwise_independent_20261010_v4/RESULT_KO.md','이전 frozen 같은 좌표/backend 진단')+'를 유지했다. 이 조건을 이번 새 pair에서 다시 실행하지 않았고 과거 수치를 V18 수치처럼 붙이지 않는다. 새 가림 분류/큰 학습의 필요성을 이 결과만으로 주장하지 않는다. 최종 재투영은 얻은 R,t의 표시 교체이므로 그 자체가 이미 구한 R,t를 더 정확하게 만들었다는 독립 인과가 아니다.\n',
        '이번 변화로 무엇이 좋아졌는지는 같은-q ORIGINAL paired 표로 판단한다. q/physical line ownership, condition/global uniqueness, Moderate·tail 손상, 충분한 inlier에도 틀린 pose가 가능한 문제는 이 평균 판정과 별개다. 같은 seed/threshold/model을 나쁜 성능에 맞춰 바꿔 반복하지 않았다. Severe74·새 촬영·수동 주석·새 RGB/학습·main 수정·force push는 수행하지 않았다.\n',
        history_note,'공개 byte bundle '+text(bundle['total_files'])+' files / '+text(bundle['total_bytes'])+' bytes에서 V17/V18 저장 검사 4개를 실제 실행했다. '+link(DOC/'PUBLIC_READER_CHECKS.json')+'가 권위이며 private RGB/weights/원 annotation GT 복사 없이 saved reference를 보존했다. Git clone·새 하드웨어 측정이 아니다. 이 문서와 후기 reader proof는 bundle 뒤에 생성되어 앞선 복사본에 포함됐다고 주장하지 않는다. 전용 브랜치 게시 commit/remote 확인은 후속 실제 게시 영수증이 있어야 완료다.\n']
    report='\n'.join(lines)
    commands='''```bash
# 복사본 repo root에서, 아래 출력 디렉터리를 먼저 새로 만들고 비워 둔다.
mkdir -p /tmp/pallet-consensus-decision-repair-private-20261011-v18/reader-user-statistics-01 /tmp/pallet-consensus-decision-repair-private-20261011-v18/reader-user-runtime-01
python -I -S scripts/research/pallet_consensus_decision_repair_20261011_v18/verify.py freeze --output /tmp/pallet-consensus-decision-repair-private-20261011-v18/reader-user-statistics-01
python -I -S scripts/research/pallet_consensus_decision_repair_20261011_v18/verify.py run --output /tmp/pallet-consensus-decision-repair-private-20261011-v18/reader-user-statistics-01
python -I -S scripts/research/pallet_consensus_decision_repair_20261011_v18/runtime_checks.py freeze --output /tmp/pallet-consensus-decision-repair-private-20261011-v18/reader-user-runtime-01
python -I -S scripts/research/pallet_consensus_decision_repair_20261011_v18/runtime_checks.py run --output /tmp/pallet-consensus-decision-repair-private-20261011-v18/reader-user-runtime-01
```
'''
    commands17='''```bash
mkdir -p /tmp/pallet-hybrid-standard-controls-private-20261011-v17/reader-user-statistics-01 /tmp/pallet-hybrid-standard-controls-private-20261011-v17/reader-user-runtime-01
python -I -S scripts/research/pallet_hybrid_standard_controls_20261011_v17/verify.py freeze --output /tmp/pallet-hybrid-standard-controls-private-20261011-v17/reader-user-statistics-01
python -I -S scripts/research/pallet_hybrid_standard_controls_20261011_v17/verify.py run --output /tmp/pallet-hybrid-standard-controls-private-20261011-v17/reader-user-statistics-01
python -I -S scripts/research/pallet_hybrid_standard_controls_20261011_v17/runtime_check.py freeze --output /tmp/pallet-hybrid-standard-controls-private-20261011-v17/reader-user-runtime-01
python -I -S scripts/research/pallet_hybrid_standard_controls_20261011_v17/runtime_check.py run --output /tmp/pallet-hybrid-standard-controls-private-20261011-v17/reader-user-runtime-01
```
'''
    # Exact CLI output guards are shown from the standalone checkers, not executed here.
    readme='# V18 결과와 재현\n\n'+link(DOC/'RESULT_KO.md','전체 한국어 결과')+' / '+link(DOC/'BUILD_LEDGER.md','실행 증거')+' / '+link(DOC/'PUBLIC_READER_CHECKS.json','실제 공개 reader 4개 PASS')+'\n\n'+history_note+'\n'+commands+'\n출력은 checker가 허용하는 새 PRIVATE 하위 디렉터리여야 한다. 기존 receipt를 덮어쓰지 않는다. private asset/model/PnP/GT/새 timing 호출 없이 저장 원행 산술만 검산한다. 실제 학습/추론/하드웨어 재측정 재현은 원 asset 환경이 추가로 필요하다.\n'
    ledger='# V18 실행량과 게시 상태\n\n'+block(dict(geometry_execution=receipt['actual_execution'],geometry_scope=geo['scope'],geometry_checks=geo.get('checks'),statistics=v['actual_counts'],runtime=dict(rows=timing['rows'],measured=timing['measured'],checks=timing['checks'],summary_slots=timing['summary_slots']),bundle_files=bundle['total_files'],bundle_bytes=bundle['total_bytes'],publication_complete=False,goal_completion_not_inferred_from_audit=True))+'\n증거는 '+link(DOC/'CONTROL_RECEIPT.json')+' / '+link(DOC/'VALIDATION_CHECKS.json')+' / '+link(DOC/'VERIFICATION.json')+' / '+link(DOC/'RUNTIME_RECEIPT.json')+' / '+link(DOC/'RUNTIME_CHECKS.json')+' / '+link(DOC/'FIGURE_BINDINGS.json')+' / '+link(DOC/'PUBLIC_READER_CHECKS.json')+'이다. 부모 full FAIL와 기존 실패/원행을 삭제·PASS 치환하지 않는다. archive는 이미 frozen generic helper의 실제 manifest이며 새 학습/성능 재실험이 아니다. 후기 보고서는 원 bundle 뒤 생성된다. 게시 전용 브랜치 remote SHA 확인은 root의 후속 실제 영수증에 남긴다.\n'
    late='# V17/V18 공개 저장 결과 재현 검토\n\n이 후기 문서는 보호된 V17 RESULT/README/BUILD를 수정하지 않는다. 알려진 DEV proxy 결과와 원 V16 full FAIL도 그대로다.\n\n'+table(('증거','실제 완료','한계'),((link(DOC/'PUBLIC_BUNDLE_CHECKS.json',folder=OLD),bundle['total_files'],'처음 없는 공개 byte bundle; Git clone 아님'),(link(DOC/'PUBLIC_READER_CHECKS.json',folder=OLD),'V17/V18 statistics/runtime 4 PASS','saved scalar 산술; 하드웨어 측정·model/GT 호출 0')))+block(reader['limits'])+'\n'+commands17+'\n'+commands+'\n같은 복사본 repo root에서 먼저 각각 새 빈 허용 출력 폴더를 준비한다. original 13 session/10,000 draw와 저장 full-path timing을 다시 확인했으며 새 draw/학습/촬영은 없다. 외부 readonly 선언은 metadata로 보존하고 private RGB/weights/원 GT annotation은 복사하지 않았으며 saved 평가 reference는 보존했다. 네 실제 proof3 경로는 PUBLIC_READER_CHECKS.json에 결속돼 있다. 후기 report/reader 파일은 먼저 frozen한 bundle의 일부라고 주장하지 않는다. 이 후기 문서의 게시 완료/새 commit은 후속 실제 publication receipt가 있어야 인증된다.\n'
    return dict(zip(TARGETS,(report,readme,ledger,late)))

def guard(stage):
    require(DOC.is_dir() and OLD.is_dir(),'existing experiment folders')
    for p in (*TARGETS,*(DOC/n for n in FILES if stage=='freeze' or n!=FILES[0])):require(not safe(p).exists(),'preserve exclusive output '+str(p))
def freeze():
    guard('freeze');value=snapshot();write(DOC/FILES[0],dict(schema='saved_point_consensus_report_build_protocol_v18',snapshot=value,frozen_before_report_write=True,limits=LIMITS))
def run():
    guard('run');own=read(DOC/FILES[0]);value=snapshot();require(own['schema']=='saved_point_consensus_report_build_protocol_v18' and own['snapshot']==value,'own report freeze differs')
    write(DOC/FILES[1],dict(own_protocol=bind(DOC/FILES[0]),inputs=value['inputs'],limits=LIMITS));written=[];error=None
    try:
        for p,contents in documents().items():write(p,contents);written.append(bind(p))
        require(snapshot()==value,'saved evidence changed during report transcription')
    except BaseException as exc:error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
    complete=error is None and len(written)==4
    write(DOC/FILES[2],dict(schema='saved_point_consensus_report_build_checks_v18',complete=complete,passed=complete,own_protocol=bind(DOC/FILES[0]),
        inputs=value['inputs'],outputs=written,error=error,limits=LIMITS,partial_outputs_preserved=True,new_metric_or_CI_computations=0))
    require(complete,'report build incomplete; preserve first failure and prefix')
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('freeze','run'));args=p.parse_args();(freeze if args.stage=='freeze' else run)()
if __name__=='__main__':main()
