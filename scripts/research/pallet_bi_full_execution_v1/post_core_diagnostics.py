"""Stored fullnormal solver metadata joined to sealed post-bank outcomes.

No reference reader, pose metric, producer or numerical solver is imported.
Ranks and singular values are copied from saved local Jacobian audits. They
neither prove global uniqueness nor certify a material correspondence.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

SEEDS=('20261018','20261019','20261020')
STATES=('S','B','I','O','T','U')
FAMILIES=('ORIGINAL_WD','SWAPPED_WD')
POPS={'ALL238':('CAL','TEST','REMAINDER'),'PRIMARY192':('TEST','REMAINDER'),
      'TEST49':('TEST',),'REMAINDER143':('REMAINDER',),'CAL46':('CAL',)}
EXPECTED={'ALL238':238,'PRIMARY192':192,'TEST49':49,'REMAINDER143':143,'CAL46':46}
RANK_NOTE='Stored local image-Jacobian rank (visible or full+normal as named), excluding priors, outline, OT/U constraints; not global uniqueness or certified material correspondence.'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def read(path):
    path=Path(path)
    with (gzip.open if path.suffix=='.gz'else open)(path,'rt')as stream:return json.load(stream)


def rows(path):
    with gzip.open(path,'rt')as stream:
        for line in stream:
            if line.strip():yield json.loads(line)


def write(path,value,*,jsonl=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError('Preserve previous analysis; choose a fresh output: '+str(path))
    content=(b''.join((json.dumps(v,sort_keys=True,allow_nan=False)+'\n').encode()for v in value)if jsonl else
             (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode())
    if path.suffix=='.gz':
        with path.open('wb')as stream:
            with gzip.GzipFile(filename='',mode='wb',fileobj=stream,mtime=0)as zipped:zipped.write(content)
    else:path.write_bytes(content)


def distribution(values):
    valid=[float(v)for v in values if v is not None and math.isfinite(float(v))]
    return dict(n=len(valid),missing=len(values)-len(valid),minimum=min(valid)if valid else None,
        maximum=max(valid)if valid else None,mean=statistics.fmean(valid)if valid else None,
        median=statistics.median(valid)if valid else None)


def factor_metadata(spec):
    states=spec['states']
    if len(states)!=8 or any(s not in STATES for s in states):raise ValueError('Original eight automatic states required')
    visible=spec['visible'];boundary=spec['boundary_observations'];fullids=[p['native_corner']for p in visible]
    partialids=[p['native_corner']for p in boundary]
    if len(set(fullids))!=len(fullids)or len(set(partialids))!=len(partialids):raise ValueError('Stored effective factor IDs must be unique')
    if any(not isinstance(k,int)or not 0<=k<8 for k in fullids+partialids):raise ValueError('Invalid native factor ID')
    revoked=set(partialids);complete=[p for p in visible if p['native_corner']not in revoked]
    for point in visible+boundary:
        if point['state']!=states[point['native_corner']]or point['state']not in ('B','I'):raise ValueError('Factor role/state differs')
    dims={k:0 for k,s in enumerate(states)if s in ('B','I')}
    for point in complete:dims[point['native_corner']]=2
    for point in boundary:dims[point['native_corner']]=len(point['lines'])
    return dict(state_counts={s:states.count(s)for s in STATES},states=list(states),
        full2D=len(complete),full2D_by_role=dict(Counter(p['state']for p in complete)),
        effective_full_native_ids=sorted(p['native_corner']for p in complete),
        normal1D=sum(len(p['lines'])for p in boundary),normal1D_by_role={s:sum(len(p['lines'])for p in boundary if p['state']==s)for s in ('B','I')},
        normal_anchor_n=len(boundary),normal_native_ids=sorted(partialids),
        revoked_full_archive_native_ids=sorted(set(fullids)&revoked),
        observed_scalar_dimension=2*len(complete)+sum(len(p['lines'])for p in boundary),
        B_I_anchor_dimension_counts={str(d):sum(v==d for v in dims.values())for d in sorted({0,1,2,*dims.values()})},
        source_scope='already deduplicated stored solver_input; revoked complete archives excluded from full2D')


def rank_metadata(value):
    if value is None:return dict(recorded=False,rank=None,parameter_count=None,singular_values=None,condition=None,
        category='NOT_RECORDED',missing_reason='saved solution did not record this rank; never imputed as zero',
        interpretation=RANK_NOTE,global_uniqueness_inferred=False)
    rank=value.get('rank');parameters=value.get('parameter_count')
    if rank is not None and(not isinstance(rank,int)or rank<0 or(parameters is not None and rank>parameters)):raise ValueError('Invalid stored local rank')
    singular=value.get('singular_values')
    if singular is not None and any(not math.isfinite(float(s))or s<0 for s in singular):raise ValueError('Invalid stored singular values')
    return dict(recorded=True,rank=rank,parameter_count=parameters,singular_values=singular,condition=value.get('condition'),
        category='NOT_RECORDED'if rank is None else('FULL_LOCAL_'+str(rank)+'OF'+str(parameters)if rank==parameters else'LOCAL_'+str(rank)+'OF'+str(parameters)),
        source_rank_proves_global_uniqueness=value.get('rank_proves_global_uniqueness'),
        interpretation=RANK_NOTE,global_uniqueness_inferred=False)


def family_metadata(family):
    solution=family['solution'];rank=solution.get('rank')or{};search=solution.get('search')or{}
    endpoints=solution.get('all_visited_endpoints',[])
    return dict(name=family['name'],family_accepted=family.get('accepted'),solution_accepted=solution.get('accepted'),
        fallback_used=family.get('accepted')is not True,status=solution.get('status'),fallback_reason=solution.get('fallback_reason'),
        fallback_pose_stored=bool(solution.get('fallback_pose')),
        rank={key:rank_metadata(rank.get(key))for key in ('visible','BI_and_partial_boundary')},
        ordered_initialization_n=len(solution['ordered_actual_initializations'])if'ordered_actual_initializations'in solution else None,
        ordered_joint_initialization_n=len(solution['ordered_actual_joint_initializations'])if'ordered_actual_joint_initializations'in solution else None,
        search_joint_initializations_per_branch=search.get('joint_initializations_per_branch'),
        T_union_branches_searched=search.get('T_union_branches_searched'),
        full_saved_initial_solution_n=len(solution['full_saved_initial_solutions'])if'full_saved_initial_solutions'in solution else None,
        additional_initialization_n=len(solution['additional_actual_initializations'])if'additional_actual_initializations'in solution else None,
        stored_optimizer_calls=solution.get('optimizer_calls'),
        outline_stage_exact_reused_without_outline=solution.get('outline_stage_exact_reused_without_outline',False),
        starts_scope='array lengths are stored metadata; inherited arrays on exact stage reuse do not imply G10 executed those starts',
        minimum_joint_slack=solution.get('minimum_joint_slack'),
        OT_constraints_satisfied=solution.get('OT_constraints_satisfied'),U_constraints_satisfied=solution.get('U_constraints_satisfied'),
        outline_valid_projection=((solution.get('objective_components')or{}).get('global_outline_audit')or{}).get('valid_projection'),
        saved_endpoint_n=len(endpoints),saved_endpoint_slack=distribution([p.get('minimum_constraint_slack')for p in endpoints]),
        stored_ambiguity=solution.get('ambiguity'),numeric_caps=search.get('numeric_caps'))


def coverage_metadata(coverage):
    selected=coverage['selected']
    if set(selected)!=set(SEEDS):raise ValueError('Every requested arm needs all three scored head seeds')
    result={key:coverage[key]for key in ('generated_good','unique_good','eligible_good','generated_candidates','unique_candidates','eligible_candidates')}
    if any(not isinstance(result[k],bool)for k in ('generated_good','unique_good','eligible_good')):raise ValueError('Stored coverage booleans required')
    result['selected']={}
    for seed,source in selected.items():
        success=source['metric']['success']
        if not isinstance(success,bool):raise ValueError('Already-scored selected success required')
        expected='SUCCESS'if success else'GENERATED_ABSENT'if not result['generated_good']else'ADMISSION_LOSS'if not result['eligible_good']else'SELECTION_LOSS'
        if source['failure_stage']!=expected:raise ValueError('Sealed coverage failure attribution is inconsistent')
        result['selected'][seed]=dict(selected_good=success,failure_stage=source['failure_stage'],transition=source['transition'],
            selected_candidate_id=source.get('selected_candidate_id'),fallback=source.get('fallback'),prediction_status=source.get('prediction_status'))
    return result


def extract(result,coverage):
    if result['id']!=coverage['id']:raise ValueError('Stored solver and scored coverage identity differs')
    stage=result['stage_results']['G10'];families={f['name']:family_metadata(f)for f in stage['families']}
    if set(families)!=set(FAMILIES)or len(stage['families'])!=2:raise ValueError('Both original G10 families required')
    selected=result['selected_family']
    if selected not in families or stage['selected_family']!=selected:raise ValueError('Selected family differs from stored G10 readout')
    factors=factor_metadata(result['packet']['solver_input'])
    for family in families.values():
        for key,dimension in [('visible',2*factors['full2D']),('BI_and_partial_boundary',factors['observed_scalar_dimension'])]:
            rank=family['rank'][key]['rank']
            if rank is not None and rank>dimension:raise ValueError('Stored local image rank exceeds its observed scalar dimension')
    return {key:coverage[key]for key in ('arm','id','partition','session','cluster_id')}|dict(
        factors=factors,G10_families=families,selected_family=selected,
        selected_G10=families[selected],coverage=coverage_metadata(coverage),
        numeric_caps={key:result['packet']['solver_input']['config'].get(key)for key in
            ('max_p3p_triples','global_rotation_starts','max_seed_nfev','max_joint_iterations','maximum_union_branches')},
        G13_exact_reuses_G10=result.get('proof',{}).get('G13_exact_reuses_G10'),
        stored_full_pipeline_pose_optimizer_calls=result.get('operations',{}).get('pose_optimizer_calls'),
        operation_count_scope='stored result/stage counts, possibly from an exact reuse origin; not new campaign execution cost',
        rank_interpretation=RANK_NOTE)


def summarize(group):
    n=len(group);coverage=[r['coverage']for r in group]
    return dict(n=n,ids=[r['id']for r in group],cluster_counts=dict(Counter(r['cluster_id']for r in group)),
        coverage_bounds={key:dict(successes=sum(c[key]for c in coverage),denominator=n,rate=sum(c[key]for c in coverage)/n if n else None)
            for key in ('generated_good','unique_good','eligible_good')},
        bounds_scope='retrospective already-scored saved-pose coverage; not actual selection performance or global solver capability',
        seeds={seed:dict(selected_good=sum(c['selected'][seed]['selected_good']for c in coverage),
            denominator=n,selected_rate=sum(c['selected'][seed]['selected_good']for c in coverage)/n if n else None,
            failure_stages=dict(Counter(c['selected'][seed]['failure_stage']for c in coverage)),
            selected_success_outside_eligible_coverage=sum(c['selected'][seed]['selected_good']and not c['eligible_good']for c in coverage))for seed in SEEDS},
        stored_G10_family_bounds={family:dict(
            accepted_n=sum(r['G10_families'][family]['family_accepted']is True for r in group),
            fallback_n=sum(r['G10_families'][family]['fallback_used']for r in group),
            status_counts=dict(Counter(str(r['G10_families'][family]['status'])for r in group)),
            fallback_reason_counts=dict(Counter(str(r['G10_families'][family]['fallback_reason'])for r in group)),
            local_rank_categories={key:dict(Counter(r['G10_families'][family]['rank'][key]['category']for r in group))for key in ('visible','BI_and_partial_boundary')},
            optimizer_calls=distribution([r['G10_families'][family]['stored_optimizer_calls']for r in group]),
            minimum_joint_slack=distribution([r['G10_families'][family]['minimum_joint_slack']for r in group]),
            ordered_initialization_n=distribution([r['G10_families'][family]['ordered_initialization_n']for r in group]),
            ordered_joint_initialization_n=distribution([r['G10_families'][family]['ordered_joint_initialization_n']for r in group]),
            search_joint_initializations_per_branch=distribution([r['G10_families'][family]['search_joint_initializations_per_branch']for r in group]))for family in FAMILIES},
        numeric_cap_bounds={key:distribution([r['numeric_caps'][key]for r in group])for key in
            ('max_p3p_triples','global_rotation_starts','max_seed_nfev','max_joint_iterations','maximum_union_branches')},
        stored_selected_solver_bounds={
            'minimum_joint_slack':distribution([r['selected_G10']['minimum_joint_slack']for r in group]),
            'optimizer_calls':distribution([r['selected_G10']['stored_optimizer_calls']for r in group]),
            'ordered_initialization_n':distribution([r['selected_G10']['ordered_initialization_n']for r in group]),
            'ordered_joint_initialization_n':distribution([r['selected_G10']['ordered_joint_initialization_n']for r in group]),
            'rank_condition':distribution([r['selected_G10']['rank']['BI_and_partial_boundary']['condition']for r in group])})


def stratify(group):
    tables=defaultdict(lambda:defaultdict(list))
    for row in group:
        f=row['factors'];chosen=row['selected_G10']
        labels={'state_signature':'|'.join(s+'='+str(f['state_counts'][s])for s in STATES),
            'dimension_signature':f"full2D={f['full2D']}|normal1D={f['normal1D']}|scalar={f['observed_scalar_dimension']}",
            'full2D':str(f['full2D']),'normal1D':str(f['normal1D']),'normal_anchor_n':str(f['normal_anchor_n']),
            'observed_scalar_dimension':str(f['observed_scalar_dimension']),
            'zero_dimension_BI_anchors':str(f['B_I_anchor_dimension_counts']['0']),'selected_family':row['selected_family'],
            'selected_G10_status':str(chosen['status']),'selected_G10_fallback_reason':str(chosen['fallback_reason'])}
        labels.update({'state_count/'+s:str(f['state_counts'][s])for s in STATES})
        labels.update({'selected_local_rank/'+k:chosen['rank'][k]['category']for k in ('visible','BI_and_partial_boundary')})
        for family,record in row['G10_families'].items():
            labels['G10_family_status/'+family]=str(record['status'])
            labels['G10_family_local_rank/'+family]=record['rank']['BI_and_partial_boundary']['category']
        for name,value in labels.items():tables[name][value].append(row)
    return {name:{value:summarize(rr)for value,rr in sorted(values.items())}for name,values in sorted(tables.items())}


def authenticate_postbank(manifest,post_bank,arms=None):
    manifest,post_bank=Path(manifest),Path(post_bank);plan=read(manifest);seal=read(post_bank/'ANALYSIS_SEAL.json')
    if seal.get('complete')is not True or seal['manifest_sha256']!=sha(manifest):raise ValueError('Matching complete post-bank analysis required')
    required={'INPUT_BINDINGS.json','BANK_COVERAGE_ROWS.jsonl.gz','SUMMARY.json'}
    if not required<=set(seal['artifacts']):raise ValueError('Post-bank input bindings and coverage must be sealed')
    for name,value in seal['artifacts'].items():
        if sha(post_bank/name)!=value:raise ValueError('Sealed post-bank artifact changed')
    bindings=read(post_bank/'INPUT_BINDINGS.json');coverage=list(rows(post_bank/'BANK_COVERAGE_ROWS.jsonl.gz'))
    requested=list(arms)if arms is not None else[name for name,spec in plan['arms'].items()if spec['backend']=='fullnormal']
    if not requested or len(set(requested))!=len(requested):raise ValueError('Explicit unique fullnormal arm population required')
    if set(bindings)!=set(plan['arms'])or {r['arm']for r in coverage}!=set(plan['arms']):raise ValueError('Post-bank arm populations differ')
    grouped={}
    for arm in requested:
        if arm not in plan['arms']or plan['arms'][arm]['backend']!='fullnormal':raise ValueError('Only fullnormal arms supported')
        rr=[r for r in coverage if r['arm']==arm]
        if len(rr)!=238 or len({r['id']for r in rr})!=238:raise ValueError('All238 unique scored images required per arm')
        if dict(Counter(r['partition']for r in rr))!={'CAL':46,'TEST':49,'REMAINDER':143}:raise ValueError('Canonical populations changed')
        spec=plan['arms'][arm];binding=bindings[arm]
        for key in ('features','feature_seal'):
            expected_path=Path(spec[key]).resolve();item=binding[key]
            if Path(item['path']).resolve()!=expected_path or sha(expected_path)!=item['sha256']:raise ValueError('Post-bank feature input binding changed')
        own=read(spec['feature_seal'])
        if own.get('complete')is not True or own.get('rows')!=238:raise ValueError('Complete own-bank feature seal required')
        if binding['features']['sha256']not in (own.get('features_sha256'),own.get('head_features_sha256')):raise ValueError('Own feature bytes not bound by their seal')
        prediction=binding['prediction'];prediction_path=Path(spec['prediction_dir'])/'PREDICTIONS.jsonl.gz'
        if (Path(prediction['path']).resolve()!=prediction_path.resolve()or sha(prediction_path)!=prediction['sha256']
                or sha(prediction_path.parent/'PREDICTION_SEAL.json')!=prediction['seal_sha256']):raise ValueError('Post-bank prediction source changed')
        generated=Path(spec.get('generated_feature_seal',spec['feature_seal'])).resolve();gb=binding['generated_feature_seal']
        if Path(gb['path']).resolve()!=generated or sha(generated)!=gb['sha256']:raise ValueError('Generated feature ancestor differs')
        gs=read(generated)
        if gs.get('complete')is not True or gs.get('rows')!=238 or len(gs.get('frame_seal_sha256',{}))!=238:raise ValueError('Complete fullnormal generated-bank seal required')
        if binding['generated_frame_seals']!=gs['frame_seal_sha256']:raise ValueError('Post-bank generated row ancestry differs')
        if generated!=Path(spec['feature_seal']).resolve():
            if own.get('schema')!='OWN_FULL_NORMAL_KERNEL_EXTRA_FEATURE_SEAL_V1':raise ValueError('Explicit supported union ancestry required')
            union_rows={r['id']:r for r in own.get('frame_seals',[])}
            if len(union_rows)!=238 or set(union_rows)!={r['id']for r in rr}:raise ValueError('Full238 union ancestor rows required')
        grouped[arm]=(spec,binding,gs,rr)
    return grouped,dict(manifest_sha256=sha(manifest),post_bank_analysis_seal_sha256=sha(post_bank/'ANALYSIS_SEAL.json'),
        post_bank_input_bindings_sha256=sha(post_bank/'INPUT_BINDINGS.json'),post_bank_coverage_sha256=sha(post_bank/'BANK_COVERAGE_ROWS.jsonl.gz'))


def run(manifest,post_bank,output,*,arms=None):
    output=Path(output).resolve();grouped,inputs=authenticate_postbank(manifest,post_bank,arms)
    roots=[Path(post_bank).resolve(),*[Path(spec['result_root']).resolve()for spec,_,_,_ in grouped.values()]]
    if any(output==root or root in output.parents for root in roots):raise ValueError('New diagnostic output must be outside immutable input trees')
    ledger=[];reports={};source_rows={}
    for arm,(spec,binding,gs,coverage)in grouped.items():
        current=[];source_rows[arm]={}
        byid={r['id']:r for r in coverage};seen=set();own=read(spec['feature_seal'])
        union_rows={r['id']:r for r in own.get('frame_seals',[])}
        generated_path=Path(spec.get('generated_feature_seal',spec['feature_seal'])).resolve()
        for ordinal in range(238):
            folder=Path(spec['result_root'])/f'{ordinal:04d}';sp=folder/'FRAME_SEAL.json'
            if sha(sp)!=gs['frame_seal_sha256'][f'{ordinal:04d}']:raise ValueError('Requested generated frame seal changed')
            frame_seal=read(sp)
            if frame_seal['id']not in byid or frame_seal['id']in seen:raise ValueError('Generated/scored image bijection differs')
            cov=byid[frame_seal['id']];seen.add(cov['id'])
            if frame_seal.get('complete')is not True or frame_seal['id']!=cov['id']or frame_seal['domain']!='real'or frame_seal['ordinal']!=ordinal:
                raise ValueError('Generated frame identity differs')
            if frame_seal['source_binding']!=gs['source_binding']:raise ValueError('Generated numerical source differs')
            for name,value in frame_seal['artifacts'].items():
                if sha(folder/name)!=value:raise ValueError('Stored result/feature artifact changed')
            if 'RESULT.json.gz'not in frame_seal['artifacts']:raise ValueError('Sealed stored RESULT required')
            feature=read(folder/'FEATURE.json.gz')
            if any(feature[k]!=cov[k]for k in ('id','partition','session','cluster_id')):raise ValueError('Core feature and scored coverage metadata differ')
            if generated_path!=Path(spec['feature_seal']).resolve():
                union_entry=union_rows[cov['id']];union_path=Path(union_entry['seal_path'])
                if sha(union_path)!=union_entry['sha256']:raise ValueError('Union row seal changed')
                dependencies=read(union_path)['dependencies']
                if (dependencies['core_frame_seal_sha256']!=sha(sp)or dependencies['result_sha256']!=sha(folder/'RESULT.json.gz')):
                    raise ValueError('Union/core RESULT ancestry differs')
            result=read(folder/'RESULT.json.gz');record=extract(result,cov);record['ordinal']=ordinal
            current.append(record);ledger.append(record)
            source_rows[arm][f'{ordinal:04d}']=dict(frame_seal_sha256=sha(sp),result_sha256=sha(folder/'RESULT.json.gz'))
        if seen!=set(byid):raise ValueError('Every scored image must join exactly once')
        reports[arm]={}
        for name,parts in POPS.items():
            subset=[r for r in current if r['partition']in parts]
            if len(subset)!=EXPECTED[name]:raise ValueError('Full reporting denominator changed')
            reports[arm][name]=dict(summary=summarize(subset),strata=stratify(subset))
    write(output/'CORE_DIAGNOSTIC_ROWS.jsonl.gz',ledger,jsonl=True);write(output/'SUMMARY_BY_POPULATION_AND_STRATUM.json',reports)
    write(output/'INPUT_BINDINGS.json',dict(inputs=inputs,generated_row_bindings=source_rows))
    seal=dict(schema='POST_CORE_METADATA_DIAGNOSTIC_SEAL_V1',complete=True,arms=list(grouped),rows_per_arm=238,
        populations=EXPECTED,seeds=list(SEEDS),source_sha256=sha(__file__),inputs=inputs,
        artifacts={name:sha(output/name)for name in ('CORE_DIAGNOSTIC_ROWS.jsonl.gz','SUMMARY_BY_POPULATION_AND_STRATUM.json','INPUT_BINDINGS.json')},
        new_pose_optimizer_calls=0,new_H1_fits=0,reference_files_read=0,pose_metrics_calculated=0,
        rank_interpretation=RANK_NOTE,PRIMARY_is_repeated_development_data=True,independent_real_test_n=0,
        failure_stage_labels=dict(GENERATED_ABSENT='no good pose in stored generated candidates',
            ADMISSION_LOSS='a good generated pose exists but no eligible good candidate',
            SELECTION_LOSS='ranking/selection failed despite an eligible good candidate'),
        losses='GENERATED_ABSENT / ADMISSION_LOSS / SELECTION_LOSS copied and consistency-checked from sealed post-bank outcomes')
    write(output/'DIAGNOSTIC_SEAL.json',seal);return seal


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('manifest','post-bank','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--arms',nargs='+');a=parser.parse_args();print(run(a.manifest,a.post_bank,a.output,arms=a.arms))
