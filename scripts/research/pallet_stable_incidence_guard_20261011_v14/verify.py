"""Independent V14 saved-statistics audit with unchanged scalar arithmetic.

The frozen V13 verifier supplies the exact moment and paired-bootstrap code.
Only the six-method stream topology, completeness counts, method-name lookup
and receipt version change. No production import, GT, fit, SVD or new draw.
"""
import ast
import hashlib
from pathlib import Path

REPO=Path(__file__).resolve().parents[3];CODE=Path(__file__).resolve().parent
DOC=REPO/'_docs/experiments/pallet_stable_incidence_guard_20261011_v14'
PARENT=REPO/'_docs/experiments/pallet_finite_edge_consensus_repair_20261011_v13'
SOURCE=REPO/'scripts/research/pallet_finite_edge_consensus_repair_20261011_v13/verify.py'
SOURCE_SHA='bad35cd12dd449d2f02e772e715b4dc3b04c74f74b1e34f508a0288bb53e018e'
METHODS=('BASE','N3_SUBPIX','ROLE_BOUNDARY_H_ROBUST','ROLE_BOUNDARY_LOCAL_POINT_LINE','FINITE_EDGE_CONSENSUS','FINITE_EDGE_CONSENSUS_STABLE_INCIDENCE')
CONTRASTS=tuple((METHODS[-1],m) for m in ('N3_SUBPIX','FINITE_EDGE_CONSENSUS','ROLE_BOUNDARY_H_ROBUST'))
FILES=('STATISTICS_VERIFICATION_PROTOCOL.json','STATISTICS_VERIFICATION_STARTED.json','VERIFICATION.json')

def require(v,s):
    if not v:raise ValueError(s)

def kernel():
    require(hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SOURCE_SHA,'exact immutable V13 statistics verifier required')
    tree=ast.parse(SOURCE.read_text());original=ast.dump(tree,include_attributes=False)
    changes={
        'six_saved_streams':("(('new_scores',245,{METHODS[-1]}),('old_local_scores',245,{METHODS[3]}),('point_scores',245,{METHODS[2]}),('fixed_scores',490,set(METHODS[:2])))", "(('new_scores',245,{METHODS[-1]}),('finite_scores',245,{METHODS[-2]}),('old_local_scores',245,{METHODS[3]}),('point_scores',245,{METHODS[2]}),('fixed_scores',490,set(METHODS[:2])))"),
        'six_method_completeness':('dict(saved_scored_rows_streamed=1225,moment_scalars_checked=810,distributions_checked=135,pair_groups_checked=27,metric_CI_slots_checked=81)', 'dict(saved_scored_rows_streamed=1470,moment_scalars_checked=972,distributions_checked=162,pair_groups_checked=27,metric_CI_slots_checked=81)'),
        'new_primary_name':("stats['strata']['combined']['contrasts']['FINITE_EDGE_CONSENSUS_minus_N3_SUBPIX']['common_operational']", "stats['strata']['combined']['contrasts'][METHODS[-1]+'_minus_N3_SUBPIX']['common_operational']")}
    before={k:ast.parse(v[0],mode='eval').body for k,v in changes.items()};after={k:ast.parse(v[1],mode='eval').body for k,v in changes.items()};hits={k:0 for k in changes}
    class Replace(ast.NodeTransformer):
        def visit(self,node):
            for k,value in before.items():
                if ast.dump(node,include_attributes=False)==ast.dump(value,include_attributes=False):hits[k]+=1;return after[k]
            return super().visit(node)
    tree=Replace().visit(tree);require(all(v==1 for v in hits.values()),'exactly three data-topology AST sites')
    # Restore the original AST as a proof that formulas were not rewritten.
    class Undo(ast.NodeTransformer):
        def visit(self,node):
            for k,value in after.items():
                if ast.dump(node,include_attributes=False)==ast.dump(value,include_attributes=False):return before[k]
            return super().visit(node)
    derived=ast.dump(tree,include_attributes=False);restored=Undo().visit(ast.parse(ast.unparse(tree)))
    require(ast.dump(restored,include_attributes=False)==original,'canonical unchanged-arithmetic AST restoration')
    hits={k:0 for k in changes};tree=Replace().visit(ast.parse(SOURCE.read_text()));require(all(v==1 for v in hits.values()),'same exact three topology sites on compilation');versions={'independent_finite_edge_statistics_protocol_v13':'independent_stable_incidence_statistics_protocol_v14','independent_finite_edge_statistics_checks_v13':'independent_stable_incidence_statistics_checks_v14'}
    class Version(ast.NodeTransformer):
        def visit_Constant(self,node):
            if isinstance(node.value,str) and node.value in versions:return ast.copy_location(ast.Constant(versions[node.value]),node)
            return node
    tree=ast.fix_missing_locations(Version().visit(tree));witness=dict(original_source_sha256=SOURCE_SHA,original_full_AST_sha256=hashlib.sha256(original.encode()).hexdigest(),data_topology_AST_sites=list(changes),data_topology_AST_changes=3,derived_arithmetic_AST_sha256=hashlib.sha256(derived.encode()).hexdigest(),scalar_formula_AST_changes=0,receipt_schema_only_versions=versions)
    env=dict(__file__=str(Path(__file__).resolve()),__name__='isolated_stable_incidence_statistics_kernel')
    exec(compile(tree,str(SOURCE)+' [unchanged numeric kernel; six-method topology]','exec'),env)
    env.update(DOC=DOC,PRIVATE=Path('/tmp/pallet-stable-incidence-guard-private-20261011-v14'),METHODS=METHODS,CONTRASTS=CONTRASTS,FILES=FILES)
    return env,witness

def input_paths(args,h):
    p=args.input;stats=h['read'](p/'METRICS.json');f={key:h['public_path'](v) for key,v in stats['inputs'].items()}
    names=dict(METRICS='METRICS.json',statistics_started='STATISTICS_STARTED.json',scoring_started='SCORING_STARTED.json',scoring_receipt='SCORING_RECEIPT.json',geometry_seal='GEOMETRY_SEAL.json',repair_receipt='CONTROL_RECEIPT.json',geometry='GEOMETRY_SEALED.jsonl.gz',ledgers='CONTROL_LEDGERS.jsonl.gz',population='PARENT_POPULATION_CHECKS.json',geometry_validation='VALIDATION_CHECKS.json',geometry_validation_protocol='VALIDATION_PROTOCOL.json',geometry_validation_started='VALIDATION_STARTED.json')
    f.update({k:p/v for k,v in names.items()});f.update(checker=Path(__file__).resolve(),geometry_checker=CODE/'validation_checks.py',scalar_source=h['SOURCE'],original_verifier=SOURCE,protocol=Path(args.protocol),evaluation_protocol=p/'EVALUATION_PROTOCOL.json',scoring_authority=p/'SCORING_AUTHORITY.json',score_dispatch_review_protocol=p/'SCORING_DISPATCH_REVIEW_PROTOCOL.json',score_dispatch_review_started=p/'SCORING_DISPATCH_REVIEW_STARTED.json',score_dispatch_review=p/'SCORING_AUTHORITY_SUPPLEMENT.json',score_dispatch_failure=p/'SCORING_DISPATCH_FIRST_FAILURE.json',statistics_authority=p/'STATISTICS_AUTHORITY_SUPPLEMENT.json',score_review_source=CODE/'evaluation_review.py')
    evaluation=h['read'](f['evaluation_protocol'])
    for key,value in evaluation['inputs'].items():f['evaluation:'+key]=h['public_path'](value)
    review=h['read'](f['score_dispatch_review_protocol'])
    for key,value in review['inputs'].items():f['dispatch_review:'+key]=h['public_path'](value)
    for n in ('PROTOCOL.json','SCORING_RECEIPT.json','SCORING_STARTED.json','PREDICTIONS.jsonl.gz','METRICS.json','REVIEW_VALIDATION_CHECKS.json','REVIEW_VALIDATION_PROTOCOL.json','REVIEW_VALIDATION_STARTED.json','VALIDATION_CHECKS.json','VALIDATION_PROTOCOL.json'):
        f['parent:'+n]=PARENT/n
    require(f['new_scores']==p/'PREDICTIONS.jsonl.gz' and f['finite_scores']==PARENT/'PREDICTIONS.jsonl.gz','same guarded and immutable finite score streams')
    return f

def metadata(f,b,h):
    read,same=h['read'],h['same'];stats,core,score,seal,r,check,own=[read(f[k]) for k in ('METRICS','protocol','scoring_receipt','geometry_seal','repair_receipt','geometry_validation','geometry_validation_protocol')]
    require(stats['schema']=='stable_incidence_raw_row_metrics_v14' and stats['complete'] is True and stats['methods']==list(METHODS) and stats['contrasts']==[list(v) for v in CONTRASTS] and stats['scopes']==list(h['SCOPES']),'fixed six-method / three-contrast scope')
    evaluation,authority=read(f['evaluation_protocol']),read(f['scoring_authority'])
    require(evaluation['schema']=='postseal_stable_incidence_evaluation_protocol_v14' and evaluation['methods']==list(METHODS) and evaluation['contrasts']==[list(v) for v in CONTRASTS] and evaluation['frames']==245 and evaluation['scored_rows']==1470 and evaluation['moment_scalars']==972 and evaluation['paired_groups']==27 and evaluation['metric_CIs']==81 and evaluation['new_seed_draws']==0 and evaluation['frozen_before_own_GT_access_or_statistic_arithmetic'] is True,'own evaluation fixed before score and statistics')
    require(authority['schema']=='unchanged_V13_proxy_score_dispatch_for_guard_v14' and authority['methods']==[METHODS[-1]] and authority['inputs']==evaluation['inputs'] and same(authority['own_evaluation_protocol'],b['evaluation_protocol']) and authority['original_V13_source_changed'] is False and authority['method_and_output_dispatch_only'] is True and authority['reference_or_error_formula_changed'] is False and authority['new_independent_incidence_check_PASS'] is True,'unchanged scoring kernel dispatch authority')
    for key,value in evaluation['inputs'].items():require(same(value,b['evaluation:'+key]),'current own-evaluation input '+key)
    for key,alias in (('adapter','own_code'),('unchanged_scalar_statistics','statistics_kernel'),('cohort','cohort'),('protocol','protocol'),('geometry_seal','geometry_seal'),('geometry','geometry'),('geometry_validation','geometry_validation'),('validation_protocol','geometry_validation_protocol'),('validation_started','geometry_validation_started'),('checker','geometry_checker'),('parent_finite_scores','finite_scores'),('parent_finite_scoring','parent:SCORING_RECEIPT.json'),('fixed_scores','fixed_scores'),('old_local_scores','old_local_scores'),('point_scores','point_scores'),('bootstrap','bootstrap')):require(same(evaluation['inputs'][key],b[alias]),'evaluation fixed actual source '+key)
    failure,review,dispatch,dispatch_started,stat_authority=[read(f[k]) for k in ('score_dispatch_failure','score_dispatch_review_protocol','score_dispatch_review','score_dispatch_review_started','statistics_authority')]
    require(failure['schema']=='preserved_stable_incidence_score_dispatch_failure_v14' and failure['complete'] is False and failure['error']==dict(type='RuntimeError',message='SAME_OBSERVATION_GUARD: frozen repair source/policy/inputs differ') and failure['failed_before_SCORING_STARTED'] is True and failure['failed_before_GT_kernel_or_score_rows'] is True and failure['GT_score_calls']==failure['scored_rows']==0 and failure['source_files_changed'] is False and failure['original_frozen_parent_METHODS_restored_by_finally'] is True and same(failure['source'],b['own_code']),'preserved pre-GT failed global-method dispatch')
    for key,alias in (('EVALUATION_PROTOCOL.json','evaluation_protocol'),('SCORING_AUTHORITY.json','scoring_authority')):require(same(failure['authority'][key],b[alias]),'original failed dispatch authority '+key)
    require(review['schema']=='isolated_guard_score_dispatch_review_protocol_v14' and review['method']==METHODS[-1] and review['frames']==245 and review['original_failed_dispatch_preserved'] is True and review['frozen_before_own_GT_score_calls'] is True,'separate pre-GT method-local dispatch freeze')
    marker=review['review'];require(marker['local_method_import_AST_changes']==1 and marker['full_canonical_score_function_restore_proven'] is True and marker['score_reference_formula_AST_changes']==0 and marker['immutable_parent_method_tuple_modified'] is False and marker['original_module_globals_modified'] is False and marker['original_module_package_context_preserved'] is True,'one method-local import change without parent mutation')
    for key,value in review['inputs'].items():require(same(value,b['dispatch_review:'+key]),'actual isolated dispatch source '+key)
    for key,alias in (('review_code','score_review_source'),('original_adapter','own_code'),('original_score_driver','evaluation:unchanged_score_dispatch'),('original_evaluation_protocol','evaluation_protocol'),('original_dispatch_authority','scoring_authority'),('original_dispatch_failure','score_dispatch_failure')):require(same(review['inputs'][key],b[alias]),'method-local review byte closure '+key)
    for key,value in evaluation['inputs'].items():require(same(review['inputs']['original_evaluation:'+key],value),'unchanged original evaluation input '+key)
    require(dispatch['schema']=='isolated_guard_score_dispatch_authority_v14' and dispatch['method']==METHODS[-1] and dispatch['inputs']==review['inputs'] and dispatch['review']==marker and dispatch['original_failed_dispatch_preserved'] is True and dispatch['score_reference_formulas_changed'] is False and same(dispatch['own_protocol'],b['score_dispatch_review_protocol']) and same(dispatch['started'],b['score_dispatch_review_started']) and dispatch_started['inputs']==review['inputs'] and dispatch_started['review']==marker and same(dispatch_started['protocol'],b['score_dispatch_review_protocol']) and dispatch_started['configured_scored_rows']==245 and dispatch_started['GT_score_calls_before_STARTED']==0,'actual successful isolated score dispatch authority')
    require(stat_authority['schema']=='isolated_guard_score_statistics_authority_v14' and stat_authority['inputs']==review['inputs'] and stat_authority['original_full_statistics_function_changed'] is False and stat_authority['configured_saved_score_rows']==1470 and stat_authority['configured_moment_scalars']==972 and stat_authority['configured_metric_CIs']==81 and stat_authority['new_seed_draws']==0 and stat_authority['frozen_before_original_STATISTICS_STARTED_and_own_arithmetic'] is True,'actual unchanged statistics function authority')
    for key,alias in (('own_review_protocol','score_dispatch_review_protocol'),('review_source','score_review_source'),('original_statistics_source','own_code'),('successful_score_dispatch_authority','score_dispatch_review'),('actual_score_receipt','scoring_receipt')):require(same(stat_authority[key],b[alias]),'actual statistics review byte closure '+key)
    require(core['schema']=='fixed_selected_stable_incidence_protocol_v14' and core['methods']==[METHODS[-1]] and core['frames']==core['rows']==core['ledger_rows']==245,'fixed single guard method')
    require(seal['schema']=='selected_stable_incidence_geometry_seal_v14' and seal['complete'] is True and seal['frames']==seal['rows']==seal['ledger_rows']==245 and seal['GT_read_allowed'] is False and seal['cleanup_completed_before_seal'] is True and r['schema']=='selected_stable_incidence_execution_v14' and r['complete'] is True and r['cleanup_completed'] is True and r['error'] is None and r['cleanup_error'] is None,'complete restored GT-free geometry')
    for key,alias in (('protocol','protocol'),('geometry','geometry'),('ledgers','ledgers'),('control_receipt','repair_receipt'),('parent_population_checks','population')):require(same(seal[key],b[alias]),'current sealed bytes '+key)
    require(own['schema']=='stable_incidence_scalar_protocol_v14' and check['schema']=='stable_incidence_scalar_checks_v14' and check['complete'] is True and check['passed'] is True and check['exception'] is None and check['failure_count']==0 and check['inputs']==own['inputs'] and same(check['own_protocol'],b['geometry_validation_protocol']) and same(check['checker'],b['geometry_checker']),'actual independent guard scalar PASS')
    for key,alias in (('geometry','geometry'),('ledgers','ledgers'),('repair_protocol','protocol'),('geometry_seal','geometry_seal'),('repair_receipt','repair_receipt'),('parent_population_checks','population'),('checker','geometry_checker')):require(same(own['inputs'][key],b[alias]),'actual independently checked bytes '+key)
    start=read(f['geometry_validation_started']);require(start['inputs']==own['inputs'] and same(start['own_protocol'],b['geometry_validation_protocol']) and start['new_production_calls']==0,'guard scalar STARTED authority')
    require(score['schema']=='repair_postseal_proxy_scoring_v13' and score['complete'] is True and score['frames']==score['scored_rows']==245 and score['methods']==[METHODS[-1]] and score['GT_access_only_after_complete_geometry_and_cleanup'] is True and same(score['protocol'],b['protocol']) and same(score['geometry_seal'],b['geometry_seal']) and same(score['predictions'],b['new_scores']),'complete inherited-body postseal guard scoring')
    ss=read(f['scoring_started']);require(same(ss['independent_validation'],b['geometry_validation_protocol']) and same(ss['geometry_seal'],b['geometry_seal']) and ss['configured_scored_rows']==245 and ss['GT_only_after_full_seal_cleanup_and_independent_PASS'] is True,'GT score follows actual guard PASS')
    old=read(f['parent:SCORING_RECEIPT.json']);require(old['complete'] is True and old['scored_rows']==old['frames']==245 and old['methods']==[METHODS[-2]] and same(old['predictions'],b['finite_scores']) and same(old['protocol'],b['parent:PROTOCOL.json']),'immutable original finite scores')
    pv,po,pf=[read(f['parent:'+n]) for n in ('REVIEW_VALIDATION_CHECKS.json','REVIEW_VALIDATION_PROTOCOL.json','VALIDATION_CHECKS.json')]
    require(pv['complete'] is True and pv['passed'] is True and pv['checks']==713803 and pv['failure_count']==0 and pv['exception'] is None and pv['inputs']==po['inputs'] and same(pv['own_protocol'],b['parent:REVIEW_VALIDATION_PROTOCOL.json']) and pf['complete'] is True and pf['passed'] is False and pf['checks']==713802 and pf['failure_count']==74,'parent full separate review PASS / original FAIL preserved')
    for key,value in stats['inputs'].items():require(same(value,b[key]),'saved producing statistics input '+key)
    started=read(f['statistics_started']);require(started['inputs']==stats['inputs'] and started['configured_new_rows']==245 and started['configured_saved_score_rows']==1470 and started['new_seed_draws']==0 and started['methods']==list(METHODS) and started['contrasts']==[list(v) for v in CONTRASTS],'statistics fixed before producing arithmetic')
    expected=dict(scored_rows=1470,new_rows=245,posthoc_rows=245,methods=6,strata=3,moment_distributions=162,moment_scalars=972,paired_groups=27,metric_CIs=81,new_bootstrap_draws=0);require(stats['actual_arithmetic']==expected,'six-method scalar/CI execution scope')
    return stats

def main():
    h,witness=kernel();oldwrite=h['write'];h['inputs']=lambda args:input_paths(args,h);h['metadata']=lambda f,b:metadata(f,b,h)
    def write(path,value):
        value['source_topology_adapter']=witness;oldwrite(path,value)
    h['write']=write
    def freeze(args):
        out=h['guard'](args,'freeze');f=h['inputs'](args);b={k:h['binding'](v) for k,v in f.items()};h['metadata'](f,b);_,info=h['extraction']()
        write(out/FILES[0],dict(schema='independent_stable_incidence_statistics_protocol_v14',inputs=b,helpers=info,limits=h['LIMITS'],frames=245,methods=list(METHODS),contrasts=[list(v) for v in CONTRASTS],scopes=list(h['SCOPES']),moment_scalars=972,paired_groups=27,metric_CI_slots=81,existing_sessions=13,existing_draws=10000,new_draws=0,frozen_after_full_geometry_cleanup_PASS_scoring_statistics_before_own_arithmetic=True))
    h['freeze']=freeze;h['main']()
if __name__=='__main__':main()
