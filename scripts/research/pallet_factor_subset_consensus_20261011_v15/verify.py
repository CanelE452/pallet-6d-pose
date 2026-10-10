"""Own-frozen saved V15 statistics check with unchanged scalar arithmetic.

Reuse the immutable V13 full verifier, changing only one saved-stream alias,
the primary method lookup, method constants and receipt versions. No production
imports, GT, model, fit, SVD, timing or new bootstrap draws are invoked.
"""
import ast
import hashlib
from pathlib import Path

REPO=Path(__file__).resolve().parents[3];CODE=Path(__file__).resolve().parent
DOC=REPO/'_docs/experiments/pallet_factor_subset_consensus_20261011_v15'
PARENT=REPO/'_docs/experiments/pallet_stable_incidence_guard_20261011_v14'
SOURCE=REPO/'scripts/research/pallet_finite_edge_consensus_repair_20261011_v13/verify.py'
SOURCE_SHA='bad35cd12dd449d2f02e772e715b4dc3b04c74f74b1e34f508a0288bb53e018e'
METHODS=('BASE','N3_SUBPIX','ROLE_BOUNDARY_H_ROBUST','FINITE_EDGE_CONSENSUS_STABLE_INCIDENCE','FACTOR_SUBSET_FINITE_EDGE_CONSENSUS')
CONTRASTS=tuple((METHODS[-1],m) for m in ('N3_SUBPIX','ROLE_BOUNDARY_H_ROBUST','FINITE_EDGE_CONSENSUS_STABLE_INCIDENCE'))
FILES=('STATISTICS_VERIFICATION_PROTOCOL.json','STATISTICS_VERIFICATION_STARTED.json','VERIFICATION.json')

def require(v,s):
    if not v:raise ValueError(s)

def kernel():
    require(hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SOURCE_SHA,'immutable full V13 statistics verifier')
    tree=ast.parse(SOURCE.read_text());original=ast.dump(tree,include_attributes=False)
    changes={
        'saved_stable_stream':("(('new_scores',245,{METHODS[-1]}),('old_local_scores',245,{METHODS[3]}),('point_scores',245,{METHODS[2]}),('fixed_scores',490,set(METHODS[:2])))","(('new_scores',245,{METHODS[-1]}),('stable_scores',245,{METHODS[3]}),('point_scores',245,{METHODS[2]}),('fixed_scores',490,set(METHODS[:2])))"),
        'primary_method_lookup':("stats['strata']['combined']['contrasts']['FINITE_EDGE_CONSENSUS_minus_N3_SUBPIX']['common_operational']","stats['strata']['combined']['contrasts'][METHODS[-1]+'_minus_N3_SUBPIX']['common_operational']")}
    before={k:ast.parse(v[0],mode='eval').body for k,v in changes.items()};after={k:ast.parse(v[1],mode='eval').body for k,v in changes.items()};hits={k:0 for k in changes}
    class Replace(ast.NodeTransformer):
        def visit(self,node):
            for k,value in before.items():
                if ast.dump(node,include_attributes=False)==ast.dump(value,include_attributes=False):hits[k]+=1;return after[k]
            return super().visit(node)
    tree=Replace().visit(tree);require(all(v==1 for v in hits.values()),'exactly two data topology sites');derived=ast.dump(tree,include_attributes=False)
    class Undo(ast.NodeTransformer):
        def visit(self,node):
            for k,value in after.items():
                if ast.dump(node,include_attributes=False)==ast.dump(value,include_attributes=False):return before[k]
            return super().visit(node)
    require(ast.dump(Undo().visit(ast.parse(ast.unparse(tree))),include_attributes=False)==original,'full canonical arithmetic restoration')
    hits={k:0 for k in changes};tree=Replace().visit(ast.parse(SOURCE.read_text()));require(all(v==1 for v in hits.values()),'same two compilation sites')
    versions={'independent_finite_edge_statistics_protocol_v13':'independent_factor_subset_statistics_protocol_v15','independent_finite_edge_statistics_checks_v13':'independent_factor_subset_statistics_checks_v15'}
    class Version(ast.NodeTransformer):
        def visit_Constant(self,node):return ast.copy_location(ast.Constant(versions[node.value]),node) if isinstance(node.value,str) and node.value in versions else node
    tree=ast.fix_missing_locations(Version().visit(tree));witness=dict(original_source_sha256=SOURCE_SHA,original_full_AST_sha256=hashlib.sha256(original.encode()).hexdigest(),data_topology_AST_sites=list(changes),data_topology_AST_changes=2,derived_arithmetic_AST_sha256=hashlib.sha256(derived.encode()).hexdigest(),scalar_formula_AST_changes=0,receipt_schema_only_versions=versions)
    env=dict(__file__=str(Path(__file__).resolve()),__name__='isolated_factor_subset_statistics_kernel');exec(compile(tree,str(SOURCE)+' [unchanged arithmetic; stable stream topology]','exec'),env)
    env.update(DOC=DOC,PRIVATE=Path('/tmp/pallet-factor-subset-consensus-private-20261011-v15'),METHODS=METHODS,CONTRASTS=CONTRASTS,FILES=FILES);return env,witness

def input_paths(args,h):
    p=args.input;stats=h['read'](p/'METRICS.json');f={key:h['public_path'](v) for key,v in stats['inputs'].items()}
    names=dict(METRICS='METRICS.json',statistics_started='STATISTICS_STARTED.json',scoring_started='SCORING_STARTED.json',scoring_receipt='SCORING_RECEIPT.json',geometry_seal='GEOMETRY_SEAL.json',repair_receipt='CONTROL_RECEIPT.json',geometry='GEOMETRY_SEALED.jsonl.gz',ledgers='CONTROL_LEDGERS.jsonl.gz',population='PARENT_POPULATION_CHECKS.json',geometry_validation='VALIDATION_CHECKS.json',geometry_validation_protocol='VALIDATION_PROTOCOL.json',geometry_validation_started='VALIDATION_STARTED.json',evaluation_protocol='EVALUATION_PROTOCOL.json',scoring_authority='SCORING_AUTHORITY.json')
    f.update({k:p/v for k,v in names.items()});f.update(checker=Path(__file__).resolve(),geometry_checker=CODE/'validation_checks.py',scalar_source=h['SOURCE'],original_verifier=SOURCE,protocol=Path(args.protocol))
    evaluation=h['read'](f['evaluation_protocol'])
    for key,value in evaluation['inputs'].items():f['evaluation:'+key]=h['public_path'](value)
    for n in ('PROTOCOL.json','SCORING_RECEIPT.json','SCORING_STARTED.json','PREDICTIONS.jsonl.gz','METRICS.json','VALIDATION_CHECKS.json','VALIDATION_PROTOCOL.json','VALIDATION_STARTED.json','GEOMETRY_SEAL.json','CONTROL_RECEIPT.json'):
        f['parent:'+n]=PARENT/n
    require(f['new_scores']==p/'PREDICTIONS.jsonl.gz' and f['stable_scores']==PARENT/'PREDICTIONS.jsonl.gz','new and immutable stable score streams')
    return f

def metadata(f,b,h):
    read,same=h['read'],h['same'];stats,core,score,seal,r,check,own=[read(f[k]) for k in ('METRICS','protocol','scoring_receipt','geometry_seal','repair_receipt','geometry_validation','geometry_validation_protocol')]
    require(stats['schema']=='factor_subset_raw_row_metrics_v15' and stats['complete'] is True and stats['methods']==list(METHODS) and stats['contrasts']==[list(v) for v in CONTRASTS] and stats['scopes']==list(h['SCOPES']),'fixed five-method/three-contrast statistics')
    evaluation,authority=read(f['evaluation_protocol']),read(f['scoring_authority'])
    require(evaluation['schema']=='postseal_factor_subset_evaluation_protocol_v15' and evaluation['methods']==list(METHODS) and evaluation['contrasts']==[list(v) for v in CONTRASTS] and evaluation['frames']==245 and evaluation['scored_rows']==1225 and evaluation['moment_scalars']==810 and evaluation['paired_groups']==27 and evaluation['metric_CIs']==81 and evaluation['new_seed_draws']==0 and evaluation['frozen_before_own_GT_access_or_statistic_arithmetic'] is True,'evaluation fixed before score and statistics')
    for key,value in evaluation['inputs'].items():require(same(value,b['evaluation:'+key]),'current own evaluation input '+key)
    require(authority['schema']=='isolated_subset_proxy_score_authority_v15' and authority['methods']==[METHODS[-1]] and authority['inputs']==evaluation['inputs'] and same(authority['own_evaluation_protocol'],b['evaluation_protocol']) and authority['score_dispatch']==evaluation['score_dispatch'] and authority['original_module_globals_modified'] is False and authority['reference_or_error_formula_changed'] is False and authority['new_independent_geometry_check_PASS'] is True,'isolated actual score dispatch authority')
    marker=evaluation['score_dispatch'];require(marker['local_method_import_AST_changes']==1 and marker['full_canonical_score_function_restore_proven'] is True and marker['score_reference_formula_AST_changes']==0 and marker['original_module_globals_modified'] is False and marker['immutable_parent_method_tuple_modified'] is False and marker['original_module_package_context_preserved'] is True and same(marker['original_score_source'],b['evaluation:unchanged_score_dispatch']),'exact one local method import and unchanged score formulas')
    for key,alias in (('adapter','own_code'),('unchanged_scalar_statistics','statistics_kernel'),('cohort','cohort'),('protocol','protocol'),('geometry_seal','geometry_seal'),('geometry','geometry'),('geometry_validation','geometry_validation'),('validation_protocol','geometry_validation_protocol'),('validation_started','geometry_validation_started'),('checker','geometry_checker'),('stable_scores','stable_scores'),('stable_scoring','parent:SCORING_RECEIPT.json'),('fixed_scores','fixed_scores'),('point_scores','point_scores'),('bootstrap','bootstrap')):require(same(evaluation['inputs'][key],b[alias]),'frozen evaluation byte closure '+key)
    require(core['schema']=='fixed_factor_subset_consensus_protocol_v15' and core['methods']==[METHODS[-1]] and core['frames']==core['rows']==core['ledger_rows']==245,'fixed single-arm geometry')
    require(seal['schema']=='factor_subset_consensus_geometry_seal_v15' and seal['complete'] is True and seal['frames']==seal['rows']==seal['ledger_rows']==245 and seal['GT_read_allowed'] is False and seal['cleanup_completed_before_seal'] is True and r['schema']=='factor_subset_consensus_execution_v15' and r['complete'] is True and r['cleanup_completed'] is True and r['error'] is None and r['cleanup_error'] is None,'completed GT-free geometry and cleanup')
    for key,alias in (('protocol','protocol'),('geometry','geometry'),('ledgers','ledgers'),('control_receipt','repair_receipt'),('parent_population_checks','population')):require(same(seal[key],b[alias]),'current sealed bytes '+key)
    require(own['schema']=='factor_subset_consensus_scalar_protocol_v15' and check['schema']=='factor_subset_consensus_scalar_checks_v15' and check['complete'] is True and check['passed'] is True and check['exception'] is None and check['failure_count']==0 and check['inputs']==own['inputs'] and same(check['own_protocol'],b['geometry_validation_protocol']) and same(check['checker'],b['geometry_checker']),'actual independent full geometry PASS')
    for key,alias in (('geometry','geometry'),('ledgers','ledgers'),('repair_protocol','protocol'),('geometry_seal','geometry_seal'),('repair_receipt','repair_receipt'),('parent_population_checks','population'),('checker','geometry_checker')):require(same(own['inputs'][key],b[alias]),'actual independently checked bytes '+key)
    start=read(f['geometry_validation_started']);require(start['inputs']==own['inputs'] and same(start['own_protocol'],b['geometry_validation_protocol']) and start['new_production_calls']==0,'scalar STARTED authority')
    require(score['schema']=='repair_postseal_proxy_scoring_v13' and score['complete'] is True and score['frames']==score['scored_rows']==245 and score['methods']==[METHODS[-1]] and score['GT_access_only_after_complete_geometry_and_cleanup'] is True and same(score['protocol'],b['protocol']) and same(score['geometry_seal'],b['geometry_seal']) and same(score['predictions'],b['new_scores']),'complete unchanged-body postseal scoring')
    ss=read(f['scoring_started']);require(same(ss['independent_validation'],b['geometry_validation_protocol']) and same(ss['geometry_seal'],b['geometry_seal']) and ss['configured_scored_rows']==245 and ss['GT_only_after_full_seal_cleanup_and_independent_PASS'] is True,'score after actual full245 independent PASS')
    parent=read(f['parent:SCORING_RECEIPT.json']);require(parent['complete'] is True and parent['frames']==parent['scored_rows']==245 and parent['methods']==[METHODS[3]] and same(parent['predictions'],b['stable_scores']) and same(parent['protocol'],b['parent:PROTOCOL.json']),'immutable stable scores')
    pc,po=read(f['parent:VALIDATION_CHECKS.json']),read(f['parent:VALIDATION_PROTOCOL.json']);require(pc['complete'] is True and pc['passed'] is True and pc['failure_count']==0 and pc['exception'] is None and pc['inputs']==po['inputs'] and same(pc['own_protocol'],b['parent:VALIDATION_PROTOCOL.json']),'parent guard full scalar PASS')
    for key,value in stats['inputs'].items():require(same(value,b[key]),'producing statistics authority '+key)
    started=read(f['statistics_started']);require(started['inputs']==stats['inputs'] and started['configured_new_rows']==245 and started['configured_saved_score_rows']==1225 and started['new_seed_draws']==0 and started['methods']==list(METHODS) and started['contrasts']==[list(v) for v in CONTRASTS],'statistics fixed before own arithmetic')
    require(stats['actual_arithmetic']==dict(scored_rows=1225,new_rows=245,posthoc_rows=245,methods=5,strata=3,moment_distributions=135,moment_scalars=810,paired_groups=27,metric_CIs=81,new_bootstrap_draws=0),'1225/810/27/81 population')
    return stats

def main():
    h,witness=kernel();oldwrite=h['write'];h['inputs']=lambda args:input_paths(args,h);h['metadata']=lambda f,b:metadata(f,b,h)
    def write(path,value):value['source_topology_adapter']=witness;oldwrite(path,value)
    h['write']=write
    def freeze(args):
        out=h['guard'](args,'freeze');f=h['inputs'](args);b={k:h['binding'](v) for k,v in f.items()};h['metadata'](f,b);_,info=h['extraction']()
        write(out/FILES[0],dict(schema='independent_factor_subset_statistics_protocol_v15',inputs=b,helpers=info,limits=h['LIMITS'],frames=245,methods=list(METHODS),contrasts=[list(v) for v in CONTRASTS],scopes=list(h['SCOPES']),moment_scalars=810,paired_groups=27,metric_CI_slots=81,existing_sessions=13,existing_draws=10000,new_draws=0,frozen_after_full_geometry_cleanup_PASS_scoring_statistics_before_own_arithmetic=True))
    h['freeze']=freeze;h['main']()
if __name__=='__main__':main()
