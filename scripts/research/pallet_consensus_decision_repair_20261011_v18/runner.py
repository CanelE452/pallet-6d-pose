"""Own-frozen paired decision replay of the sealed V16 POINT_ONLY result.

No production numerical module, model, image, truth or scoring input is used.
The full paired scalar FAIL is retained; only its complete, failure-disjoint
point-only scope authorizes this decision replay. Wall time is not latency.
"""
from collections import Counter
import copy
import gzip
import json
import os
from pathlib import Path
import subprocess
import traceback

from . import pipeline as P
from ..pallet_sparse_local_line_20261010_v9 import common as C
from ..pallet_hybrid_factor_subset_consensus_20261011_v16 import saved_ambiguity_diagnostic as A
from ..pallet_hybrid_factor_subset_consensus_20261011_v16 import failure_witness_summary as W

REPO=C.REPO;CODE=Path(__file__).resolve().parent
DOC=REPO/'_docs/experiments/pallet_consensus_decision_repair_20261011_v18'
PRIVATE=Path('/tmp/pallet-consensus-decision-repair-private-20261011-v18')
SOURCE_DOC=A.DOC;METHODS=P.METHODS;PRIMARY=P.PRIMARY
PRIMITIVES=('solvePnP','solvePnPGeneric','solvePnPRefineLM','projectPoints','Rodrigues','cornerSubPix')
NAMES=('CONTROL_STARTED.json','GEOMETRY_SEALED.jsonl.gz','CONTROL_LEDGERS.jsonl.gz',
    'CONTROL_RECEIPT.json','GEOMETRY_SEAL.json')
LIMITS=dict(new_solver_optimizer_projection_J_SVD_model_GT_RGB_training_calls=0,
    numerical_candidate_or_observation_changes=0,original_full_paired_scalar_PASS=False,
    point_only_narrow_independent_scope_required=True,best_candidate_reselected=False,
    saved_decision_replay=True,cached_replay_latency_claim=False,global_uniqueness_proven=False,
    physical_ownership_certified=False,known_DEV=True,unseen_evaluation=False,automatic_retry=False)

def parser():
    p=C.parser(__doc__);p.set_defaults(output=str(DOC),protocol=str(DOC/'PROTOCOL.json'),
        prior_bindings=str(DOC/'PROTECTION_BEFORE.json'))
    p.add_argument('--decision-parent',default=str(SOURCE_DOC))
    p.add_argument('--v9-input',default=str(C.DOC))
    return p

def output(args,name):
    root=Path(args.output).absolute()
    C.require((root.resolve()==DOC.resolve() or root.resolve().is_relative_to(PRIVATE.resolve()))
        and not any(p.is_symlink() for p in (root,*root.parents)),'exclusive new V18 output required')
    for dep in (args.decision_parent,args.parent,args.source_root,args.baseline_root,args.fits):
        q=Path(dep).resolve();C.require(not root.resolve().is_relative_to(q)
            and not q.is_relative_to(root.resolve()),'output overlaps prior dependency')
    C.require(name==Path(name).name and not(root/name).exists(),'preserve '+name);return root/name

def input_paths(args):
    source=Path(args.decision_parent).resolve();C.require(source==SOURCE_DOC.resolve(),'fixed V16 parent required')
    f={'ambiguity:'+k:v for k,v in A.paths(source).items()}
    f.update({'failure:'+k:v for k,v in W.paths(source).items()})
    for name in ('FAILURE_SUMMARY_PROTOCOL.json','FAILURE_SUMMARY_STARTED.json','FAILURE_WITNESS_SUMMARY.json',
        'FAILURE_SUMMARY_CHECKS.json','AMBIGUITY_DIAGNOSTIC_PROTOCOL.json','AMBIGUITY_DIAGNOSTIC_STARTED.json',
        'AMBIGUITY_DIAGNOSTIC_ROWS.jsonl.gz','AMBIGUITY_DIAGNOSTIC_CHECKS.json'):
        f['v16:'+name]=source/name
    f['cohort']=Path(args.cohort)
    for name in ('__init__.py','solver.py','pipeline.py','runner.py','validation_checks.py'):
        f['code:'+name]=CODE/name
    for name in ('METHOD_CONTRACT_KO.md','PROTECTION_BEFORE.json'):f['own:'+name]=DOC/name
    old=REPO/'scripts/research/pallet_observation_refiner_20261009_v1'
    f.update(unchanged_mask_diagnostic=old/'inference.py',unchanged_cuboid=old/'solver.py',
        unchanged_point_equivalence=REPO/'scripts/research/pallet_cornerwise_independent_20261010_v4/pose.py',
        unchanged_common=Path(C.__file__),
        inherited_common=REPO/'scripts/research/pallet_same_observation_controls_20261010_v8/common.py',
        unchanged_protection=REPO/'scripts/research/pallet_boundary_corner_refiner_20261010_v2/protection.py')
    return f

def inputs(args):return {k:C.binding(v) for k,v in input_paths(args).items()}

def narrow_authority(args,b=None):
    """Consume complete actual audit proofs; never rename the paired FAIL."""
    b=inputs(args) if b is None else b;folder=Path(args.decision_parent)
    # Historical inspectors stored path/SHA/bytes without C's origin field.
    def historical(v):return {k:v[k] for k in ('path','sha256','bytes')}
    af=A.paths(folder);ab={k:historical(b['ambiguity:'+k]) for k in af};population,source_policy=A.authority(af,ab)
    wf=W.paths(folder);wb={k:historical(b['failure:'+k]) for k in wf};witnesses,trace=W.authority(wf,wb)
    summary,check,own,start=[C.read(folder/n) for n in ('FAILURE_WITNESS_SUMMARY.json',
        'FAILURE_SUMMARY_CHECKS.json','FAILURE_SUMMARY_PROTOCOL.json','FAILURE_SUMMARY_STARTED.json')]
    C.require(summary['schema']=='saved_failure_witness_summary_v16' and summary['complete'] is True
        and summary['original_scalar_validation_passed'] is False and summary['original_scalar_checks']==198134007
        and summary['original_scalar_failures']==564 and summary['geometry_or_operating_outputs_changed'] is False,
        'complete summary preserves the actual full paired scalar FAIL')
    C.require(check['schema']=='saved_failure_witness_summary_checks_v16' and check['complete'] is True
        and check['passed'] is True and check['error'] is None and check['original_geometry_scalar_passed'] is False
        and check['original_geometry_scalar_failure_count']==564 and check['inputs']==own['inputs']==start['inputs']==wb,
        'actual full failure-witness summary PASS and exact closure required')
    for value in (summary,check,start):C.bound(folder/'FAILURE_SUMMARY_PROTOCOL.json',value['own_protocol'],'summary own freeze')
    C.bound(folder/'FAILURE_WITNESS_SUMMARY.json',check['summary'],'actual complete summary bytes')
    C.require(summary['inputs']==wb and summary['limits']==W.LIMITS
        and summary['methods'][P.PARENT_METHOD].get('failed_predicates',0)==0
        and summary['methods'][P.PARENT_METHOD]['unique_candidate_contexts']==0
        and summary['methods'][P.PARENT_METHOD]['unique_frames']==0,'POINT_ONLY failure scope must be exactly empty')
    contexts=witnesses['contexts'];line_method=A.METHODS[0]
    C.require(len(trace['failures'])==len(summary['failed_predicates'])==564
        and [v['stored_failure'] for v in summary['failed_predicates']]==trace['failures']
        and all(v['method']==line_method for v in summary['failed_predicates'])
        and all(v['method']==line_method for v in contexts.values())
        and set(summary['contexts'])==set(contexts)
        and all(summary['contexts'][k]['full_original_context_semantic_sha256']==W.semantic(v) for k,v in contexts.items()),
        'all564 actual predicate/context failures must belong exclusively to the line method')
    diag,dp,ds=[C.read(folder/n) for n in ('AMBIGUITY_DIAGNOSTIC_CHECKS.json',
        'AMBIGUITY_DIAGNOSTIC_PROTOCOL.json','AMBIGUITY_DIAGNOSTIC_STARTED.json')]
    C.require(diag['schema']=='saved_ambiguity_diagnostic_checks_v16' and diag['complete'] is True
        and diag['passed'] is True and diag['rows_complete']==490 and diag['error'] is None and diag['cleanup_errors']==[]
        and diag['original_full_geometry_scalar_passed'] is False and diag['original_full_geometry_scalar_failure_count']==564
        and diag['inputs']==dp['inputs']==ds['inputs']==ab and diag['source_witness']==source_policy,
        'actual original saved ambiguity diagnostic and preserved FAIL required')
    C.bound(folder/'AMBIGUITY_DIAGNOSTIC_PROTOCOL.json',diag['own_protocol'],'ambiguity protocol')
    C.bound(folder/'AMBIGUITY_DIAGNOSTIC_PROTOCOL.json',ds['own_protocol'],'ambiguity STARTED')
    C.bound(folder/'AMBIGUITY_DIAGNOSTIC_ROWS.jsonl.gz',diag['rows'],'actual full paired classification rows')
    C.require(source_policy['numerical_score_tie_px2']==P.POLICY['numerical_score_tie_px2'],'unchanged exact tie constant')
    point=diag['groups'][P.PARENT_METHOD]
    C.require(point['ONLY_DIFFERENT_COST_SAME_SUPPORT_VETO']['count']==161
        and point['NUMERICAL_TIE_PRESENT']['count']==1 and point['NO_ACCEPTED_BEST']['count']==4
        and point['NO_REPORTED_VETO_WITH_ACCEPTED_BEST']['count']==79,'fixed complete source decision population')
    return population,dict(schema='independently_completed_point_only_narrow_geometry_authority_v18',
        full_paired_scalar_complete=True,full_paired_scalar_passed=False,full_paired_checks=198134007,
        full_paired_failures=564,all_failures_retained=True,point_only_failed_predicates=0,
        point_only_failed_contexts=0,point_only_failed_frames=0,point_only_scope_passed=True,
        failure_disjointness_rechecked=True,source_policy=source_policy,ambiguity_groups=copy.deepcopy(point),
        geometry_or_scalar_tolerance_changes=0,physical_ownership_or_accuracy_certified=False)

def protect(args):
    from ..pallet_boundary_corner_refiner_20261010_v2.protection import verify
    snapshot=C.read(args.prior_bindings);tracked=snapshot['tracked']
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    C.require(tracked['original_research_head']==head,'latest published HEAD protection required')
    check=verify(tracked,args.source_root);C.require(check['passed'],'protected published/source bytes changed')
    for item in snapshot.get('additional_completed_audits',[]):C.bound(REPO/item['path'],item,'immutable additional '+item['path'])
    return dict(published_parent_HEAD=head,unchanged_original_verifier=check,own_latest_HEAD_snapshot=True)

def frozen(args):
    own=C.read(args.protocol);b=inputs(args);pop,authority=narrow_authority(args,b)
    C.require(own['schema']=='fixed_point_consensus_decision_restoration_protocol_v18'
        and own['inputs']==b and own['policy']==P.POLICY and own['limits']==LIMITS
        and own['methods']==list(METHODS) and own['narrow_point_only_authority']==authority
        and own['source_root']==str(Path(args.source_root).resolve())
        and own['baseline_root']==str(Path(args.baseline_root).resolve()),'fixed decision replay inputs/policy differ')
    C.bound(DOC/'PARENT_POPULATION_CHECKS.json',own['parent_population_checks'],'new full245 population')
    C.require(C.read(DOC/'PARENT_POPULATION_CHECKS.json')['ordered_ids']==pop['ordered_ids'],'same ordered full source population')
    return own

def freeze(args):
    C.require(Path(args.protocol).resolve()==DOC/'PROTOCOL.json' and Path(args.output).resolve()==DOC,
        'formal V18 freeze required in own DOC')
    output(args,'PROTOCOL.json');output(args,'PARENT_POPULATION_CHECKS.json')
    b=inputs(args);pop,authority=narrow_authority(args,b);protected=protect(args)
    ids=pop['ordered_ids'];C.require(len(ids)==len(set(ids))==245,'full ordered245 source population')
    proof=dict(schema='point_consensus_decision_parent_population_v18',complete=True,passed=True,frames=245,
        ordered_ids=ids,source_method=P.PARENT_METHOD,parent_rows=490,parent_point_only_rows=245,
        full_original_population_proof=copy.deepcopy(pop),parent_population=C.binding(SOURCE_DOC/'PARENT_POPULATION_CHECKS.json'),
        selected_point_only_rows_redecoded_before_own_freeze=0,raw_geometry_or_GT_arithmetic_before_own_freeze=0,
        geometry_rows_visited_in_inherited_proof=1960,observation_rows_visited_in_inherited_proof=735,
        fixed_rows_visited_in_inherited_proof=490,scored_or_human_input=False)
    C.require(inputs(args)==b,'sources changed during own freeze')
    C.write_new(output(args,'PARENT_POPULATION_CHECKS.json'),proof)
    C.write_new(output(args,'PROTOCOL.json'),dict(schema='fixed_point_consensus_decision_restoration_protocol_v18',
        policy=P.POLICY,limits=LIMITS,inputs=b,methods=list(METHODS),primary=PRIMARY,frames=245,rows=490,ledger_rows=245,
        parent_population_checks=C.binding(DOC/'PARENT_POPULATION_CHECKS.json'),narrow_point_only_authority=authority,
        protected_before=protected,source_root=str(Path(args.source_root).resolve()),
        baseline_root=str(Path(args.baseline_root).resolve()),new_solver_optimizer_projection_J_SVD_calls_before_freeze=0,
        original_coordinate_head_LOO_H_registry_and_candidate_generation_unchanged=True,
        frozen_before_own_decision_and_mask_diagnostic_arithmetic=True,GT_access_allowed=False,
        independent_own_decision_output_validation_required_before_GT=True,no_runtime_or_accuracy_claim=True))
    frozen(args);print('POINT_DECISION_RESTORATION_FROZEN',C.binding(args.protocol)['sha256'],flush=True)

def preflight(args):
    frozen(args)
    for name in NAMES:output(args,name)
    resource=C.quiet_snapshot();passed=bool(resource['quiet'] and resource['temperature_under_80'])
    C.write_new(output(args,'PREFLIGHT.json'),dict(passed=passed,protocol=C.binding(args.protocol),
        resource_snapshot=resource,protection=protect(args),actual_new_solver_calls=0))
    C.require(passed,'competing workload/thermal guard; preserve preflight')

def run(args):
    own=frozen(args);root=Path(args.output);before=protect(args);resource=C.quiet_snapshot()
    C.require(resource['quiet'] and resource['temperature_under_80'],'independent quiet decision replay required')
    for name in NAMES:output(args,name)
    C.write_new(output(args,'CONTROL_STARTED.json'),dict(schema='point_consensus_decision_restoration_started_v18',
        protocol=C.binding(args.protocol),inputs=own['inputs'],methods=list(METHODS),limits=LIMITS,
        narrow_point_only_authority=own['narrow_point_only_authority'],resource_snapshot=resource,
        declared_execution='saved decision/display replay; zero newly fitted poses',new_production_calls=0))
    streams=[];error=None;cleanup=[];published={};c=Counter();statuses={m:Counter() for m in METHODS}
    active=None;after=None
    def encoded(v):return (json.dumps(v,ensure_ascii=False,allow_nan=False)+'\n').encode()
    try:
        for name in ('GEOMETRY_SEALED.jsonl.gz','CONTROL_LEDGERS.jsonl.gz'):
            pending=root/('PENDING_'+name);raw=pending.open('xb');streams.append([name,pending,raw,None])
            streams[-1][3]=gzip.GzipFile(fileobj=raw,mode='wb',mtime=0)
        geometry=iter(C.rows(SOURCE_DOC/'GEOMETRY_SEALED.jsonl.gz'))
        ledgers=iter(C.rows(SOURCE_DOC/'CONTROL_LEDGERS.jsonl.gz'))
        classifications=iter(C.rows(SOURCE_DOC/'AMBIGUITY_DIAGNOSTIC_ROWS.jsonl.gz'))
        ordered=C.read(DOC/'PARENT_POPULATION_CHECKS.json')['ordered_ids']
        for fid in ordered:
            active=dict(id=fid,phase='parent_stream');parent=None;diagnosis=None;session=None
            for method in A.METHODS:
                row=next(geometry);diag=next(classifications);c['parent_geometry_rows_visited']+=1;c['ambiguity_rows_visited']+=1
                C.require((row['id'],row['method'])==(fid,method)==(diag['id'],diag['method'])
                    and row['session']==diag['session'],'ordered full paired parent/classification identity')
                C.require(diag['original_solver_semantic_sha256']==P.semantic(row['solver']),'same diagnosed immutable solver')
                session=row['session'] if session is None else session;C.require(session==row['session'],'paired session identity')
                if method==P.PARENT_METHOD:parent,diagnosis=row,diag
                del row,diag
            ledger=next(ledgers);c['parent_ledger_rows_visited']+=1
            C.require((ledger['id'],ledger['session'])==(fid,session),'same full245 parent ledger')
            active.update(phase='pure_decision',parent_method=parent['method'])
            pair,newledger=P.restore_packet(parent);del parent,ledger
            newledger.update(parent_ledger_binding=own['inputs']['ambiguity:CONTROL_LEDGERS.jsonl.gz'],
                source_ambiguity_classification=diagnosis['classification'],
                narrow_point_only_scope_passed=True,original_full_paired_scalar_passed=False)
            restored=newledger['decision']['restored'];classification=diagnosis['classification']
            C.require(restored is (classification=='ONLY_DIFFERENT_COST_SAME_SUPPORT_VETO'),
                'only declared different-cost same-support veto may restore')
            c['original_point_only_rows_selected']+=1;c['pure_restoration_decisions']+=1;c['restored_rows']+=int(restored)
            c['true_tie_rows_retained']+=int(newledger['decision']['true_non_equivalent_numeric_tie_retained'])
            c['replay_display_H_assignments']+=newledger['replay_display_H_assignments']
            c['scalar_pose_to_mask_diagnostic_calls']+=newledger['scalar_pose_to_mask_diagnostic_calls']
            for row in pair:
                streams[0][3].write(encoded(row));c['geometry_rows']+=1;statuses[row['method']][row['output_status']]+=1
            streams[1][3].write(encoded(newledger));c['ledger_rows']+=1;c['frames_complete']+=1
            del pair,newledger,diagnosis;active=None
        C.require(next(geometry,None) is next(ledgers,None) is next(classifications,None) is None
            and c['frames_complete']==c['ledger_rows']==245 and c['geometry_rows']==490
            and c['parent_geometry_rows_visited']==c['ambiguity_rows_visited']==490
            and c['parent_ledger_rows_visited']==245 and c['restored_rows']==161 and c['true_tie_rows_retained']==1,
            'full245 paired completion and fixed original decision population')
        C.require(inputs(args)==own['inputs'],'all saved inputs changed during replay')
        after=protect(args)
    except BaseException as e:error=dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc(),active_frame=active)
    finally:
        for name,pending,raw,stream in streams:
            operations=[('gzip_close',stream.close)] if stream is not None else []
            operations.extend((('raw_flush',raw.flush),('raw_fsync',lambda raw=raw:os.fsync(raw.fileno())),('raw_close',raw.close)))
            for phase,op in operations:
                try:op()
                except BaseException as e:cleanup.append(dict(file=name,phase=phase,type=type(e).__name__,message=str(e)))
        if error is None and not cleanup:
            for name,pending,raw,stream in streams:
                try:os.link(pending,root/name);pending.unlink();published[name]=C.binding(root/name)
                except BaseException as e:error=dict(type=type(e).__name__,message=str(e),phase='exclusive_publish',file=name);break
        complete=error is None and not cleanup and len(published)==2
        execution=dict(c);execution.update(actual_new_solver_calls=0,actual_new_optimizer_calls=0,
            actual_new_projection_calls=0,actual_new_J_calls=0,actual_new_SVD_calls=0,
            actual_new_model_GT_RGB_training_calls=0)
        C.write_new(output(args,'CONTROL_RECEIPT.json'),dict(schema='point_consensus_decision_restoration_execution_v18',
            complete=complete,protocol=C.binding(args.protocol),methods=list(METHODS),limits=LIMITS,
            actual_execution=execution,output_status_counts={m:dict(v) for m,v in statuses.items()},
            narrow_point_only_authority=own['narrow_point_only_authority'],original_full_paired_scalar_passed=False,
            cleanup_completed=not cleanup,error=error,cleanup_error=cleanup or None,cleanup_errors=cleanup,
            active_incomplete_frame=active,protected_before=before,protected_after=after,
            source_inputs_unchanged=complete,published_outputs=published,
            preserved_prefixes={name:C.binding(pending) for name,pending,raw,stream in streams if pending.is_file()},
            latency_claim=False,new_numerical_pose_fit_calls=0,new_production_calls=0))
    C.require(complete,'decision replay incomplete; preserve first failure and prefixes')
    C.write_new(output(args,'GEOMETRY_SEAL.json'),dict(schema='point_consensus_decision_restoration_geometry_seal_v18',
        complete=True,protocol=C.binding(args.protocol),geometry=published['GEOMETRY_SEALED.jsonl.gz'],
        ledgers=published['CONTROL_LEDGERS.jsonl.gz'],control_receipt=C.binding(root/'CONTROL_RECEIPT.json'),
        parent_population_checks=C.binding(DOC/'PARENT_POPULATION_CHECKS.json'),frames=245,rows=490,ledger_rows=245,
        methods=list(METHODS),cleanup_completed_before_seal=True,GT_read_allowed=False,
        narrow_point_only_scope_passed=True,original_full_paired_scalar_passed=False,
        independent_own_decision_output_validation_required_before_GT=True,new_pose_fit_calls=0,latency_claim=False))
    print('POINT_DECISION_RESTORATION_COMPLETE',dict(c),flush=True)

def main():
    args=parser().parse_args();C.require(args.source_root and args.baseline_root,'explicit split source/baseline roots required')
    Path(args.output).mkdir(parents=True,exist_ok=True)
    {'freeze':freeze,'preflight':preflight,'run':run}[args.stage](args)

if __name__=='__main__':main()
