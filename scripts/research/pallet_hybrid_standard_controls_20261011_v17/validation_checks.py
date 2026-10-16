"""Own-frozen stdlib scalar audit of the two fixed V17 ordinary controls.

Reuse complete V8 scalar bodies unchanged. Whole-U SSE/ties and every stored
branch are checked; 8px inliers are diagnostic. Stored spectra are recounted,
not regenerated. No GT, model, image, solver, optimizer or SVD is called.
"""
import argparse
import ast
from collections import Counter
import copy
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import struct
import traceback

REPO=Path(__file__).resolve().parents[3];CODE=Path(__file__).resolve().parent
DOC=REPO/'_docs/experiments/pallet_hybrid_standard_controls_20261011_v17'
PRIVATE=Path('/tmp/pallet-hybrid-standard-controls-private-20261011-v17')
PARENT=REPO/'_docs/experiments/pallet_three_head_observation_20261010_v7'
V16=REPO/'_docs/experiments/pallet_hybrid_factor_subset_consensus_20261011_v16'
AUTHORITY=REPO/'scripts/research/pallet_hybrid_factor_subset_consensus_20261011_v16/validation_authority_ready.py'
SOURCE=REPO/'scripts/research/pallet_same_observation_controls_20261010_v8/validation_checks.py'
SOURCE_SHA='d9f8079d42515d6c72eac4c75676493766b95395c63e049414fb67fe48631121'
COHORT=REPO/'_docs/experiments/pallet_kp_corrected_supervision_20261010_v1/COHORT.json'
METHODS=('HYBRID_STANDARD_PNP','NATIVE_STANDARD_PNP')
PARENT_METHODS=('IMAGE_ROLE_BOUNDARY_ONLY','GEOMETRY_ONLY_BOUNDARY_ONLY','IMAGE_NO_ROLE_BOUNDARY_ONLY','GEOMETRY_ONLY_CORNERWISE_HYBRID','IMAGE_NO_ROLE_CORNERWISE_HYBRID','IMAGE_ROLE_CORNERWISE_HYBRID','N3_INDEPENDENT_ROBUST_H','N3_INDEPENDENT_ROBUST_NO_MASK')
HEADS=('GEOMETRY_ONLY','IMAGE_NO_ROLE','IMAGE_ROLE')
FILES=('VALIDATION_PROTOCOL.json','VALIDATION_STARTED.json','VALIDATION_CHECKS.json')
LIMITS=dict(new_production_GT_image_model_solver_optimizer_Jacobian_SVD_calls=0,
    all_stored_candidate_scalar_bodies_unchanged_from_V8=True,ordinary_whole_U_fit=True,
    diagnostic_inlier_count_is_not_acceptance_gate=True,stored_rank_spectrum_recount_only=True,
    local_rank6_not_global_uniqueness=True,physical_ownership_truth='UNKNOWN',
    null_does_not_certify_original_NaN_payload=True,resource_snapshots_can_miss_transient_contention=True,
    original_full319_legacy_evidence_preserved=True,automatic_retry=False)

def require(v,s):
    if not v:raise ValueError(s)
def read(p):return json.loads(Path(p).read_text())
def rows(p):
    with gzip.open(p,'rt') as f:
        for line in f:require(bool(line.strip()),'blank row');yield json.loads(line)
def binding(p):
    p=Path(p).absolute();require(p.is_file() and not any(q.is_symlink() for q in (p,*p.parents)),'missing/symlink '+str(p));d=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):d.update(block)
    return dict(path=str(p.relative_to(REPO)) if p.is_relative_to(REPO) else p.name,bytes=p.stat().st_size,sha256=d.hexdigest())
def same(a,b):return isinstance(a,dict) and isinstance(b,dict) and all(a.get(k)==b.get(k) for k in ('bytes','sha256'))
def write(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False,default=sorted);f.write('\n');f.flush();os.fsync(f.fileno())

def extraction():
    require(binding(SOURCE)['sha256']==SOURCE_SHA,'immutable complete V8 scalar source')
    names={'require','safe','Audit','finite','ids','eligible','dimensions','dot','model','project','determinant','physical_rotation','equivalent','native_hash','semantic_sha','stored_rank','ledger','candidate','tied','pose_packet','hidden_after'}
    constants={'FALSE_FLAGS','ATOL','RTOL','RADIUS','TIE','LEDGER_KEYS'};nodes=[];found=set();assigned=set()
    for n in ast.parse(SOURCE.read_text()).body:
        if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names:nodes.append(n);found.add(n.name)
        elif isinstance(n,ast.Assign):
            ids={q.id for t in n.targets for q in ast.walk(t) if isinstance(q,ast.Name)}
            if ids and ids<=constants:nodes.append(n);assigned.update(ids)
    require(found==names and assigned==constants,'exact scalar function/constant whitelist')
    tree=ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[]))
    return tree,dict(source=binding(SOURCE),names=sorted(names),constants=sorted(constants),whole_AST_sha256=hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest(),scalar_formula_AST_changes=0,method_label_adapter_only='ROLE_BOUNDARY_H_STANDARD')

def primary(path,key,allowed,target,n,known,a,label):
    seen={k:set() for k in allowed};total=0
    for r in rows(path):
        require(r[key] in seen and r['id'] in known and r['id'] not in seen[r[key]],'full original source unique identity')
        seen[r[key]].add(r['id']);total+=1
        require(not {'pose','corner','GT','ground_truth','human_states_native','mask_audit'}&set(r),'GT-free source rows')
        if r[key]==target:yield r
    a.eq(total,n,label+'.complete_population')
    for k,v in seen.items():a.eq(v,known,label+'.every_original_arm.'+k)

def packet(two,l,p,o,completion,h,a,counts,label):
    native=o['native_N3_points'];H=p['hidden_initial'];adopted=p['observation_contract']['hybrid_boundary_corner_ids']
    a.eq(o['predicted_N3_hidden'],H,label+'.same_H');a.eq(o['initial_N3_pose'],p['initial_pose'],label+'.same_initial');a.eq(o['GT_input'],False,label+'.GT_free')
    observed={v['id']:v for v in o['corners']};records={v['id']:v for v in p['observation_contract']['cornerwise_records']}
    for k in range(8):
        a.eq(p['input_points'][k],observed[k]['xy'] if k in adopted else native[k],label+'.original_hybrid_q.'+str(k))
        if k in adopted:a.eq(records[k]['accepted'],True,label+'.original_admission');a.eq(records[k]['fit_exclusion_verified'],True,label+'.original_heldout');a.eq(records[k]['projected_coordinate_used_as_observation'],False,label+'.not_fake_q')
    a.ok(not set(H)&set(adopted),label+'.H_not_adopted');a.eq(p['input_points'][8],native[8],label+'.source_center')
    sha=h['semantic_sha'];events={};routes=l['ordinary_routes'];unique={};cv_generic=cv_lm=jac=0
    for i,r in enumerate(two):
        s=r['solver'];q=p['input_points'] if i==0 else native;selected=adopted if i==0 else []
        for k,v in (('method',METHODS[i]),('parent_method',p['method']),('input_points',q),('original_native_N3_points',native),('hidden_initial',H),('initial_pose',p['initial_pose']),('selected_corner_ids',selected),('parent_completion',completion),('parent_geometry_row_semantic_sha256',sha(p)),('parent_observation_row_semantic_sha256',sha(o)),('parent_native_N3_semantic_sha256',sha(native)),('parent_observation_contract',p['observation_contract']),('stored_jacobian_fit_ids',s['used'] if s['geometry'].get('jacobian') else [])):a.eq(r[k],v,label+'.'+str(i)+'.'+k)
        for k in ('K','xyz','raw_hw','selected_index','fixed_metadata'):a.eq(r[k],p[k],label+'.preserved.'+k)
        a.eq(r['standard_control_policy'],h['policy'],label+'.policy');a.eq(s['policy'],h['policy']['solver'],label+'.same_ordinary_backend')
        a.eq(r['head_arm'],'IMAGE_ROLE' if i==0 else None,label+'.head');a.eq(r['observation_supply'],'CORNERWISE_HYBRID' if i==0 else 'NATIVE_N3',label+'.supply')
        if i==0:a.eq(r['observation_contract'],p['observation_contract'],label+'.original_hybrid_contract')
        if i==1:
            for k in ('validated_boundary_corner_ids','hybrid_boundary_corner_ids','corner_admission','cornerwise_records'):a.eq(r['observation_contract'].get(k,[]),[],label+'.native_no_boundary.'+k)
        for k in ('head_forward_calls','calibration_or_decode_calls','new_corner_selection_calls'):a.eq(r[k],0,label+'.no_new.'+k)
        for k in ('local_point_line_refinement','reprojections_reused_as_observations','final_numeric_pose_has_initial_prior','cached_accuracy_row_used','oracle'):a.eq(r[k],False,label+'.no_fake.'+k)
        a.eq(r['partial_lines_not_pose_inputs'],True,label+'.no_lines');a.eq(r['ordinary_inlier_count_is_diagnostic_only'],True,label+'.inliers_diagnostic')
        before=events.get(s['input_hash']);after=h['pose_packet'](dict(r,method='ROLE_BOUNDARY_H_STANDARD'),a,counts,before,label+'.ordinary.'+str(i));events[s['input_hash']]=after
        route=routes[i];zero={k:0 for k in after};bank_before=before if before is not None else zero
        for k,v in (('method',METHODS[i]),('input_hash',s['input_hash']),('bank_before',bank_before),('bank_after',after),('operation_counts',s['operation_counts']),('actual_active_ids',s['used']),('stored_jacobian_fit_ids',r['stored_jacobian_fit_ids']),('input_points_semantic_sha256',sha(q)),('line_factor_count',0)):a.eq(route[k],v,label+'.route.'+k)
        unique[s['input_hash']]=after;cv_generic+=s['operation_counts']['generic_calls'];cv_lm+=s['operation_counts']['lm_calls'];jac+=int(bool(s['geometry'].get('jacobian')))
        new=s['available'];fallback=bool(not new and p['initial_pose']['available']);expected=copy.deepcopy(native)
        a.eq(r['output_status'],'NEW_POSE' if new else 'N3_BASELINE_FALLBACK' if fallback else 'POSE_FAILURE',label+'.status');a.eq(r['new_pose_estimated'],new,label+'.new');a.eq(r['fallback_used'],fallback,label+'.fallback');a.eq(r['pose_available'],new or fallback,label+'.available');a.eq(r['no_pose'],not(new or fallback),label+'.complete_failure')
        if new:
            for k in ('R_cf','R_physical','centroid','cf_extents','projected'):a.eq(r['actual_pose'][k],s[k],label+'.actual_solver_'+k)
            for k in range(8):
                if k in H:expected[k]=s['projected'][k]
                elif h['finite'](q[k],(2,)) and q[k]!=[-1,-1]:expected[k]=q[k]
        else:a.eq(r['actual_pose'],p['initial_pose'],label+'.unchanged_full_fallback')
        a.near(r['native_points'],expected,label+'.output_H_once');a.eq(r['native_points'][8],native[8],label+'.center_exact');a.eq(r['reprojected_ids'],H if new else [],label+'.no_projection_refit');a.eq(r['hidden_reprojected'],bool(new and H),label+'.Hflag')
        ah=h['hidden_after'](s) if new else [];a.eq(r['hidden_after'],ah,label+'.H_diagnostic');a.eq(r['hidden_set_changed'],bool(new and set(ah)!=set(H)),label+'.no_mask_change_veto')
        a.eq(r['output_coordinate_sources'],['FINAL_POSE_REPROJECTION' if new and k in H else 'VALIDATED_BOUNDARY_INTERSECTION' if new and k in selected else 'NATIVE_N3_RGB' for k in range(8)]+['UNCHANGED_DETECTOR_CENTER'],label+'.sources')
        counts['status:'+METHODS[i]+':'+r['output_status']]+=1;counts['state:'+METHODS[i]+':'+s['state']]+=1;counts['rows']+=1
    a.eq(l['coordinate_banks'],[dict(input_hash=k,counts=v) for k,v in unique.items()],label+'.unique_banks');a.eq(l['coordinate_bank_count'],len(unique),label+'.bank_count');a.eq(l['same_coordinate_bank_shared'],len(unique)==1,label+'.share_exact_bank');a.eq(l['same_input_q'],p['input_points']==native,label+'.same_serialized_q')
    if all(h['finite'](q,(9,2)) for q in (p['input_points'],native)):a.eq(l['same_coordinate_bytes'],len(unique)==1,label+'.finite_float64_byte_identity')
    bank={k:sum(v[k] for v in unique.values()) for k in h['LEDGER_KEYS']};a.eq(l['bank_operation_counts'],bank,label+'.unique_bank_sum')
    for k,v in bank.items():counts['bank:'+k]+=v
    for k,v in (('methods',list(METHODS)),('actual_standard_solver_calls',2),('reported_output_views',2),('initial_H',H),('hybrid_adopted_boundary_ids',adopted),('input_packet_semantic_sha256',sha(dict(parent_row=p,native_N3_points=native,parent_completion=completion,role_observation=o))),('parent_geometry_row_semantic_sha256',sha(p)),('parent_observation_row_semantic_sha256',sha(o)),('parent_native_N3_semantic_sha256',sha(native)),('corner_selection_executed',False),('endpoint_LOO_executed',False),('line_factor_count',0)):a.eq(l[k],v,label+'.ledger.'+k)
    cv=('solvePnP','solvePnPGeneric','solvePnPRefineLM','projectPoints','Rodrigues','cornerSubPix')
    for name in cv:
        a.eq(l['primitive_entry_calls'][name],l['primitive_return_calls'][name]+l['primitive_exception_calls'][name],label+'.CVcomplete')
        for kind in ('entry','return','exception'):counts['cv:'+kind+':'+name]+=l['primitive_'+kind+'_calls'][name]
    a.eq(l['primitive_entry_calls']['solvePnPGeneric'],cv_generic,label+'.actual_Generic');a.eq(l['primitive_entry_calls']['solvePnPRefineLM'],cv_lm,label+'.actual_LM')
    for name in ('solvePnP','cornerSubPix'):a.eq(l['primitive_entry_calls'][name],0,label+'.no_extra_solver')
    for field,prefix in (('svd_dispatch','svd:'),('jacobian_dispatch','jac:')):
        a.eq(l[field]['entries'],l[field]['returns']+l[field]['exceptions'],label+'.'+field)
        for k,v in l[field].items():counts[prefix+k]+=v
    a.eq(l['jacobian_dispatch']['entries'],jac,label+'.whole_U_J_calls')
    for k,v in (('standard_solver_entries',2),('standard_solver_returns',2),('standard_solver_exceptions',0)):a.eq(l['actual_solver_dispatch'][k],v,label+'.actual2solve')
    counts['frames']+=1

def inputs(args):
    p=args.input;f={k:p/n for k,n in dict(geometry='GEOMETRY_SEALED.jsonl.gz',ledgers='CONTROL_LEDGERS.jsonl.gz',repair_protocol='PROTOCOL.json',geometry_seal='GEOMETRY_SEAL.json',repair_receipt='CONTROL_RECEIPT.json',parent_population_checks='PARENT_POPULATION_CHECKS.json',started='CONTROL_STARTED.json',preflight='PREFLIGHT.json').items()}
    f.update(checker=Path(__file__),ordinary_scalar=SOURCE,cohort=COHORT,authority_code=AUTHORITY,source_geometry=PARENT/'GEOMETRY_SEALED.jsonl.gz',source_observations=PARENT/'OBSERVATIONS.jsonl.gz')
    for n in ('pipeline.py','runner.py'):f['code:'+n]=CODE/n
    history=dict(original_checker=AUTHORITY.with_name('validation_checks.py'),original_protocol=V16/'VALIDATION_PROTOCOL.json',original_started=V16/'VALIDATION_STARTED.json',original_failure=V16/'FIRST_SCALAR_ENVIRONMENT_FAILURE.json',ready_checker=AUTHORITY.with_name('validation_environment_ready.py'),ready_protocol=V16/'VALIDATION_READY_PROTOCOL.json',ready_started=V16/'VALIDATION_READY_STARTED.json',ready_checks=V16/'VALIDATION_READY_CHECKS.json',authority_code=AUTHORITY)
    f.update({'v16_ready:'+k:v for k,v in history.items()})
    f.update(v7_validation=PARENT/'VALIDATION_CHECKS.json',v7_validation_protocol=PARENT/'VALIDATION_PROTOCOL.json',v7_validation_started=PARENT/'VALIDATION_STARTED.json',v7_checker=REPO/'scripts/research/pallet_three_head_observation_20261010_v7/validation_checks.py',v7_seal=PARENT/'GEOMETRY_SEAL.json')
    for i,_ in enumerate(read(f['repair_receipt'])['resource_snapshots']):f['resource:%03d'%i]=p/('RESOURCE_%03d.json'%i)
    return f
def metadata(f,b):
    core,seal,r,pop=[read(f[k]) for k in ('repair_protocol','geometry_seal','repair_receipt','parent_population_checks')]
    require(core['schema']=='fixed_native_hybrid_standard_protocol_v17' and core['methods']==list(METHODS) and core['frames']==core['ledger_rows']==245 and core['rows']==490,'fixed two ordinary paths')
    require(seal['schema']=='native_hybrid_standard_geometry_seal_v17' and seal['complete'] is True and seal['frames']==seal['ledger_rows']==245 and seal['rows']==490 and seal['GT_read_allowed'] is False and seal['cleanup_completed_before_seal'] is True and r['schema']=='native_hybrid_standard_execution_v17' and r['complete'] is True and r['cleanup_completed'] is True and r['error'] is r['cleanup_error'] is None,'complete cleanup before ownfreeze')
    for k,v in (('protocol','repair_protocol'),('geometry','geometry'),('ledgers','ledgers'),('control_receipt','repair_receipt'),('parent_population_checks','parent_population_checks')):require(same(seal[k],b[v]),'sealed '+k)
    require(same(r['protocol'],b['repair_protocol']) and all(same(r['raw_rows'][k],b[k]) for k in ('geometry','ledgers')) and same(core['parent_population_checks'],b['parent_population_checks']) and pop['complete'] is pop['passed'] is True and pop['frames']==245 and pop['full_V7_population_checked'] is True,'actual raw/population bindings')
    for k,v in b.items():
        if k in ('geometry','ledgers','repair_protocol','geometry_seal','repair_receipt','parent_population_checks','started','preflight') or k.startswith('resource:'):continue
        require(any(v['path']==z['path'] and same(v,z) for z in core['inputs'].values()),'ownsource/core binding '+k)
    for k in ('started','preflight'):require(same(read(f[k])['protocol'],b['repair_protocol']),'actual '+k)
    require(read(f['preflight'])['passed'] is True,'recorded quiet preflight')
    check,own,start,source_seal=[read(f[k]) for k in ('v7_validation','v7_validation_protocol','v7_validation_started','v7_seal')]
    require(check['schema']=='supplemental_v7_all_point_candidate_scalar_checks_v1' and check['complete'] is True and check['passed'] is True and check['failure_count']==0 and check['inputs']==own['inputs'] and same(check['protocol'],b['v7_validation_protocol']) and same(start['protocol'],b['v7_validation_protocol']) and same(own['checker'],b['v7_checker']),'V7 actual full scalar PASS, independent ordinary source authority')
    for alias,key in (('GEOMETRY_SEALED.jsonl.gz','source_geometry'),('OBSERVATIONS.jsonl.gz','source_observations'),('GEOMETRY_SEAL.json','v7_seal'),('checker_code','v7_checker')):require(same(check['inputs'][alias],b[key]),'V7 actual checked source '+alias)
    require(source_seal['complete'] is True and source_seal['frames']==245 and source_seal['rows']==1960 and source_seal['fixed_rows']==490 and source_seal['GT_read_allowed'] is False and same(source_seal['geometry'],b['source_geometry']) and same(source_seal['observations'],b['source_observations']),'V7 full source seal')
    env=runpy.run_path(str(AUTHORITY));original=env['failure_proof'](V16);_,marker=env['source_proof']();ro,rs,rc=[read(f['v16_ready:'+k]) for k in ('ready_protocol','ready_started','ready_checks')]
    require(rc['schema']=='hybrid_factor_subset_scalar_checks_v16' and rc['complete'] is True and rc['passed'] is False and rc['checks']==198134007 and rc['failure_count']==564 and rc['exception'] is None and rc['new_production_calls']==0,'preserved completed V16 scalar FAIL is not a V17 source PASS')
    require(ro['inputs']==rs['inputs']==rc['inputs'] and ro['environment_ready']==rs['environment_ready']==rc['environment_ready']==marker and same(rs['own_protocol'],b['v16_ready:ready_protocol']) and same(rc['own_protocol'],b['v16_ready:ready_protocol']) and same(ro['checker'],b['v16_ready:original_checker']) and same(rc['checker'],b['v16_ready:original_checker']),'failed READY source/protocol identity preserved')
    extras=dict(ready_checker='ready_checker',authority_code='authority_code',original_validation_protocol='original_protocol',original_validation_started='original_started',original_failure='original_failure')
    require(set(ro['inputs'])==set(original['inputs'])|set(extras) and all(ro['inputs'][k]==v for k,v in original['inputs'].items()) and all(same(ro['inputs'][k],b['v16_ready:'+v]) for k,v in extras.items()),'V16 failed READY exact original input and environment closure')
    return core,r,pop
def guard(args,stage):
    for p in (args.input,args.output):require(p.is_dir() and not any(q.is_symlink() for q in (p,*p.parents)),'existing non-symlink folder')
    require(args.output.resolve()==DOC or args.output.resolve().is_relative_to(PRIVATE),'new exclusive output')
    for n in FILES if stage=='freeze' else FILES[1:]:require(not(args.output/n).exists(),'preserve '+n)
def freeze(args):
    guard(args,'freeze');f=inputs(args);b={k:binding(p) for k,p in f.items()};metadata(f,b);_,info=extraction()
    write(args.output/FILES[0],dict(schema='native_hybrid_standard_scalar_protocol_v17',inputs=b,checker=b['checker'],helpers=info,limits=LIMITS,methods=list(METHODS),frames=245,rows=490,frozen_before_own_arithmetic=True))
def run(args):
    guard(args,'run');f=inputs(args);b={k:binding(p) for k,p in f.items()};own=read(args.output/FILES[0]);ob=binding(args.output/FILES[0]);tree,info=extraction()
    require(own['inputs']==b and own['helpers']==info and own['limits']==LIMITS and own['frozen_before_own_arithmetic'] is True,'ownfreeze differs');core,r,pop=metadata(f,b)
    write(args.output/FILES[1],dict(inputs=b,own_protocol=ob,new_production_calls=0));h=dict(math=math,json=json,hashlib=hashlib,struct=struct);exec(compile(tree,str(SOURCE)+':unchanged scalar bodies','exec'),h);h['policy']=core['policy'];a=h['Audit']();counts=Counter();error=None
    try:
        cohort=read(f['cohort']);ordered=cohort['ids'];known=set(ordered);require(len(ordered)==len(known)==245,'full245');a.eq(cohort['counts'],dict(clean=153,moderate=92,severe_excluded=74,original=319),'scope')
        streams=[iter(rows(f[k])) for k in ('geometry','ledgers')];parents=iter(primary(f['source_geometry'],'method',PARENT_METHODS,'IMAGE_ROLE_CORNERWISE_HYBRID',1960,known,a,'V7'));obs=iter(primary(f['source_observations'],'head_arm',HEADS,'IMAGE_ROLE',735,known,a,'ROLE'))
        for i,fid in enumerate(ordered):
            two=[next(streams[0],None),next(streams[0],None)];l,p,o=[next(s,None) for s in (streams[1],parents,obs)]
            require(all(isinstance(v,dict) and v['id']==fid and v['session']==cohort['frames'][i]['session'] for v in (*two,l,p,o)),'full ordered paired sourcejoin')
            packet(two,l,p,o,pop['completion'],h,a,counts,'frame.'+str(i))
        for s in (*streams,parents,obs):a.eq(next(s,None),None,'no_extra_rows')
        for k,v in dict(frames_started=245,frames_complete=245,standard_solver_entries=490,standard_solver_returns=490,standard_solver_exceptions=0,geometry_rows=490,ledger_rows=245).items():a.eq(r['actual_execution'][k],v,'actual.'+k)
        a.eq(r['ordinary_bank_operation_counts'],{k:counts['bank:'+k] for k in h['LEDGER_KEYS']},'actual.unique_bank_operations')
        for kind in ('entry','return','exception'):a.eq(r['actual_OpenCV_'+kind+'_calls'],{n:counts['cv:'+kind+':'+n] for n in ('solvePnP','solvePnPGeneric','solvePnPRefineLM','projectPoints','Rodrigues','cornerSubPix')},'actual.CV.'+kind)
        for field,prefix in (('actual_SVD_dispatch','svd:'),('actual_Jacobian_dispatch','jac:')):a.eq(r[field],{k:counts[prefix+k] for k in ('entries','returns','exceptions')},'actual.'+field)
        for field,prefix in (('output_status_counts','status:'),('solver_state_counts','state:')):a.eq(r[field],{m:{k.removeprefix(prefix+m+':'):v for k,v in counts.items() if k.startswith(prefix+m+':')} for m in METHODS},'actual.'+field)
        a.eq(r['forbidden_entries'],{},'no_forbidden_paths');a.eq(r['forbidden_asset_reads'],[],'no_GT_model_reads');a.eq(r['protection_before'],r['protection_after'],'protectedbytes')
        for k in ('primitive_and_pose_guards_restored','opencv_threads_restored','split_environment_restored','asset_canary_open_functions_restored'):a.eq(r['cleanup'][k],True,'cleanup.'+k)
        for i,snap in enumerate(r['resource_snapshots']):a.eq(read(f['resource:%03d'%i]),snap,'resource.byte_record');a.eq(snap['quiet'],True,'resource.recordedquiet');a.eq(snap['temperature_under_80'],True,'resource.recordedthermal')
        a.eq(counts['frames'],245,'all245');a.eq(counts['rows'],490,'all490');a.eq(counts['stored_candidate_records_checked'],counts['stored_candidate_records_in_packets'],'all_stored_candidates');a.eq({k:binding(p) for k,p in f.items()},b,'bytesunchanged');a.eq(binding(args.output/FILES[0]),ob,'ownfreezeunchanged')
    except BaseException as exc:error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
    result=dict(schema='native_hybrid_standard_scalar_checks_v17',complete=error is None and counts['frames']==245,passed=error is None and counts['frames']==245 and a.failure_count==0,checks=a.checks,failure_count=a.failure_count,failures=a.failures,exception=error,max_scalar_absolute_difference=a.max_difference,inputs=b,own_protocol=ob,checker=b['checker'],counts=dict(counts),limits=LIMITS,new_production_calls=0,first_failure_preserved=True)
    write(args.output/FILES[2],result);print('NATIVE_HYBRID_STANDARD_SCALAR',result['passed'],a.checks,a.failure_count,flush=True)
    if not result['passed']:raise SystemExit(1)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('freeze','run'));p.add_argument('--input',type=Path,default=DOC);p.add_argument('--output',type=Path,default=DOC);args=p.parse_args();{'freeze':freeze,'run':run}[args.stage](args)
if __name__=='__main__':main()
