"""Own-frozen stdlib recheck of1470 saved scores,972 moments and135 CIs.

The immutable complete V13 verifier supplies every arithmetic statement. Only
stream topology, count assertions, primary lookup and receipt versions change.
Explicit external declarations are retained as metadata and never opened.
"""
import ast
from collections import Counter
import gzip
import hashlib
from pathlib import Path

REPO=Path(__file__).resolve().parents[3];CODE=Path(__file__).resolve().parent
DOC=REPO/'_docs/experiments/pallet_hybrid_factor_subset_consensus_20261011_v16'
BOUNDARY=REPO/'_docs/experiments/pallet_factor_subset_consensus_20261011_v15'
SOURCE=REPO/'scripts/research/pallet_finite_edge_consensus_repair_20261011_v13/verify.py'
SOURCE_SHA='bad35cd12dd449d2f02e772e715b4dc3b04c74f74b1e34f508a0288bb53e018e'
METHODS=('BASE','N3_SUBPIX','IMAGE_ROLE_CORNERWISE_HYBRID','FACTOR_SUBSET_FINITE_EDGE_CONSENSUS','HYBRID_FACTOR_SUBSET_FINITE_EDGE_CONSENSUS','HYBRID_FACTOR_SUBSET_POINT_ONLY')
CONTRASTS=tuple((METHODS[4],m) for m in METHODS if m!=METHODS[4])
PARENT_METHODS=('IMAGE_ROLE_BOUNDARY_ONLY','GEOMETRY_ONLY_BOUNDARY_ONLY','IMAGE_NO_ROLE_BOUNDARY_ONLY','GEOMETRY_ONLY_CORNERWISE_HYBRID','IMAGE_NO_ROLE_CORNERWISE_HYBRID','IMAGE_ROLE_CORNERWISE_HYBRID','N3_INDEPENDENT_ROBUST_H','N3_INDEPENDENT_ROBUST_NO_MASK')
FILES=('STATISTICS_VERIFICATION_PROTOCOL.json','STATISTICS_VERIFICATION_STARTED.json','VERIFICATION.json')
ARITHMETIC=dict(scored_rows=1470,new_rows=490,posthoc_rows=490,methods=6,strata=3,moment_distributions=162,moment_scalars=972,paired_groups=45,metric_CIs=135,new_bootstrap_draws=0)

def require(value,message):
    if not value:raise ValueError(message)

def checked_stream(key,path,frames,compact_rows):
    if key!='hybrid_scores':yield from compact_rows(path);return
    seen={m:set() for m in PARENT_METHODS}
    for row in compact_rows(path):
        require(row['method'] in seen and row['id'] in frames and row['id'] not in seen[row['method']],'full original V7 unique method/ID')
        require(row['session']==frames[row['id']]['session'],'original V7 session')
        seen[row['method']].add(row['id'])
        if row['method']==METHODS[2]:yield row
    require(all(v==set(frames) for v in seen.values()),'full1960 original V7 scores retained')

def kernel():
    require(hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SOURCE_SHA,'immutable full V13 verifier')
    original_tree=ast.parse(SOURCE.read_text());original=ast.dump(original_tree,include_attributes=False)
    changes={
        'saved_stream_topology':("(('new_scores',245,{METHODS[-1]}),('old_local_scores',245,{METHODS[3]}),('point_scores',245,{METHODS[2]}),('fixed_scores',490,set(METHODS[:2])))","(('new_scores',490,set(METHODS[4:])),('boundary_scores',245,{METHODS[3]}),('hybrid_scores',245,{METHODS[2]}),('fixed_scores',490,set(METHODS[:2])))"),
        'full_parent_population_stream':("env['compact_rows'](paths[key])","checked_stream(key,paths[key],frames,env['compact_rows'])"),
        'fixed_complete_counts':("dict(saved_scored_rows_streamed=1225,moment_scalars_checked=810,distributions_checked=135,pair_groups_checked=27,metric_CI_slots_checked=81)","dict(saved_scored_rows_streamed=1470,moment_scalars_checked=972,distributions_checked=162,pair_groups_checked=45,metric_CI_slots_checked=135)"),
        'primary_method_lookup':("stats['strata']['combined']['contrasts']['FINITE_EDGE_CONSENSUS_minus_N3_SUBPIX']['common_operational']","stats['strata']['combined']['contrasts'][METHODS[4]+'_minus_N3_SUBPIX']['common_operational']")}
    before={k:ast.parse(v[0],mode='eval').body for k,v in changes.items()};after={k:ast.parse(v[1],mode='eval').body for k,v in changes.items()};hits=Counter()
    class Replace(ast.NodeTransformer):
        def visit(self,node):
            for k,value in before.items():
                if ast.dump(node,include_attributes=False)==ast.dump(value,include_attributes=False):hits[k]+=1;return after[k]
            return super().visit(node)
    tree=Replace().visit(original_tree);require(dict(hits)=={k:1 for k in changes},'exact four topology sites');derived=ast.dump(tree,include_attributes=False)
    class Undo(ast.NodeTransformer):
        def visit(self,node):
            for k,value in after.items():
                if ast.dump(node,include_attributes=False)==ast.dump(value,include_attributes=False):return before[k]
            return super().visit(node)
    require(ast.dump(Undo().visit(ast.parse(ast.unparse(tree))),include_attributes=False)==original,'full canonical arithmetic restoration')
    hits.clear();tree=Replace().visit(ast.parse(SOURCE.read_text()));require(dict(hits)=={k:1 for k in changes},'same four compilation sites')
    versions={'independent_finite_edge_statistics_protocol_v13':'independent_hybrid_factor_subset_statistics_protocol_v16','independent_finite_edge_statistics_checks_v13':'independent_hybrid_factor_subset_statistics_checks_v16'}
    class Version(ast.NodeTransformer):
        def visit_Constant(self,node):return ast.copy_location(ast.Constant(versions[node.value]),node) if isinstance(node.value,str) and node.value in versions else node
    tree=ast.fix_missing_locations(Version().visit(tree));witness=dict(original_source_sha256=SOURCE_SHA,
        original_full_AST_sha256=hashlib.sha256(original.encode()).hexdigest(),data_topology_AST_sites=list(changes),data_topology_AST_changes=4,
        derived_arithmetic_AST_sha256=hashlib.sha256(derived.encode()).hexdigest(),scalar_formula_AST_changes=0,receipt_schema_only_versions=versions)
    h=dict(__file__=str(Path(__file__).resolve()),__name__='isolated_hybrid_statistics_kernel',checked_stream=checked_stream)
    exec(compile(tree,str(SOURCE)+' [unchanged arithmetic; paired hybrid topology]','exec'),h)
    h.update(DOC=DOC,PRIVATE=Path('/tmp/pallet-hybrid-factor-subset-private-20261011-v16'),METHODS=METHODS,CONTRASTS=CONTRASTS,FILES=FILES)
    return h,witness

def public(item):
    if item.get('origin')=='external_readonly_dependency':return False
    p=Path(item['path']);require(item.get('origin')=='public_repository' and not p.is_absolute() and '..' not in p.parts and p.parts and p.parts[0] in ('scripts','_docs'),'explicit public repository binding')
    return True

def input_paths(args,h):
    p=args.input;stats=h['read'](p/'METRICS.json');f={k:h['public_path'](v) for k,v in stats['inputs'].items()}
    names=dict(METRICS='METRICS.json',statistics_started='STATISTICS_STARTED.json',scoring_started='SCORING_STARTED.json',scoring_receipt='SCORING_RECEIPT.json',
        geometry_seal='GEOMETRY_SEAL.json',repair_receipt='CONTROL_RECEIPT.json',geometry='GEOMETRY_SEALED.jsonl.gz',ledgers='CONTROL_LEDGERS.jsonl.gz',population='PARENT_POPULATION_CHECKS.json',
        geometry_validation='VALIDATION_CHECKS.json',geometry_validation_protocol='VALIDATION_PROTOCOL.json',geometry_validation_started='VALIDATION_STARTED.json',evaluation_protocol='EVALUATION_PROTOCOL.json',scoring_authority='SCORING_AUTHORITY.json')
    f.update({k:p/v for k,v in names.items()});f.update(checker=Path(__file__).resolve(),geometry_checker=CODE/'validation_checks.py',scalar_source=h['SOURCE'],original_verifier=SOURCE,protocol=Path(args.protocol))
    for k,v in h['read'](f['evaluation_protocol'])['inputs'].items():
        if public(v):f['evaluation:'+k]=h['public_path'](v)
    require(f['new_scores']==p/'PREDICTIONS.jsonl.gz' and f['boundary_scores']==BOUNDARY/'PREDICTIONS.jsonl.gz','current paired and immutable boundary score streams')
    return f

def metadata(f,b,h):
    read,same=h['read'],h['same'];stats,core,score,seal,r,check,own=[read(f[k]) for k in ('METRICS','protocol','scoring_receipt','geometry_seal','repair_receipt','geometry_validation','geometry_validation_protocol')]
    require(stats['schema']=='hybrid_factor_subset_raw_row_metrics_v16' and stats['complete'] is True and stats['methods']==list(METHODS) and stats['contrasts']==[list(v) for v in CONTRASTS] and stats['scopes']==list(h['SCOPES']) and stats['actual_arithmetic']==ARITHMETIC and stats['original_V7_scored_rows_streamed']==1960,'fixed6/1470/972/45/135 statistics')
    e,a=read(f['evaluation_protocol']),read(f['scoring_authority']);require(e['schema']=='postseal_hybrid_factor_subset_evaluation_protocol_v16' and e['methods']==list(METHODS) and e['new_methods']==list(METHODS[4:]) and e['contrasts']==[list(v) for v in CONTRASTS] and e['frames']==245 and e['scored_rows']==1470 and e['new_scored_rows']==490 and e['moment_scalars']==972 and e['paired_groups']==45 and e['metric_CIs']==135 and e['new_seed_draws']==0 and e['frozen_before_own_GT_access_or_statistic_arithmetic'] is True,'own evaluation fixed before score/statistics')
    for k,v in e['inputs'].items():
        if public(v):require(same(v,b['evaluation:'+k]),'current public evaluation input '+k)
    dispatch=e['score_dispatch'];require(dispatch['topology_AST_changes']==dict(local_method_import=1,configured_scored_rows=1,scored_rows=1,completion_count=1) and dispatch['full_canonical_score_function_restore_proven'] is True and dispatch['score_reference_formula_AST_changes']==0 and dispatch['original_module_globals_modified'] is False and dispatch['immutable_parent_method_tuple_modified'] is False and dispatch['original_module_package_context_preserved'] is True and same(dispatch['original_score_source'],b['evaluation:unchanged_score_dispatch']),'whole unchanged scorer except4 topology sites')
    require(e['statistics_dispatch']['formula_changes']==e['statistics_dispatch']['new_draws']==0 and e['statistics_dispatch']['names']==['moments','metric','summaries','paired'] and same(e['statistics_dispatch']['source'],b['statistics_kernel']),'whole unchanged statistics kernels')
    require(a['schema']=='isolated_hybrid_proxy_score_authority_v16' and a['inputs']==e['inputs'] and a['methods']==list(METHODS[4:]) and a['score_dispatch']==dispatch and a['original_module_globals_modified'] is False and a['reference_or_error_formula_changed'] is False and a['new_independent_geometry_check_PASS'] is True and same(a['own_evaluation_protocol'],b['evaluation_protocol']),'isolated paired actual score authority')
    for key,alias in (('adapter','own_code'),('unchanged_scalar_statistics','statistics_kernel'),('cohort','cohort'),('protocol','protocol'),('geometry_seal','geometry_seal'),('geometry','geometry'),('geometry_validation','geometry_validation'),('validation_protocol','geometry_validation_protocol'),('validation_started','geometry_validation_started'),('checker','geometry_checker'),('hybrid_scores','hybrid_scores'),('boundary_scores','boundary_scores'),('fixed_scores','fixed_scores'),('bootstrap','bootstrap')):require(same(e['inputs'][key],b[alias]),'frozen evaluation byte closure '+key)
    require(core['schema']=='fixed_hybrid_factor_subset_consensus_protocol_v16' and core['methods']==list(METHODS[4:]) and core['frames']==core['ledger_rows']==245 and core['rows']==490,'paired fixed geometry')
    require(seal['schema']=='hybrid_factor_subset_geometry_seal_v16' and seal['complete'] is True and seal['frames']==seal['ledger_rows']==245 and seal['rows']==490 and seal['GT_read_allowed'] is False and seal['cleanup_completed_before_seal'] is True and r['schema']=='hybrid_factor_subset_consensus_execution_v16' and r['complete'] is True and r['cleanup_completed'] is True and r['error'] is None and r['cleanup_error'] is None,'full paired GT-free seal and cleanup')
    for key,alias in (('protocol','protocol'),('geometry','geometry'),('ledgers','ledgers'),('control_receipt','repair_receipt'),('parent_population_checks','population')):require(same(seal[key],b[alias]),'current sealed bytes '+key)
    require(own['schema']=='hybrid_factor_subset_scalar_protocol_v16' and check['schema']=='hybrid_factor_subset_scalar_checks_v16' and check['complete'] is True and check['passed'] is True and check['failure_count']==0 and check['exception'] is None and check['new_production_calls']==0 and check['inputs']==own['inputs'] and same(check['own_protocol'],b['geometry_validation_protocol']) and same(check['checker'],b['geometry_checker']) and same(own['checker'],b['geometry_checker']),'actual independent full geometry PASS')
    for key,alias in (('geometry','geometry'),('ledgers','ledgers'),('repair_protocol','protocol'),('geometry_seal','geometry_seal'),('repair_receipt','repair_receipt'),('parent_population_checks','population'),('checker','geometry_checker')):require(same(own['inputs'][key],b[alias]),'current independently checked bytes '+key)
    gs=read(f['geometry_validation_started']);require(gs['inputs']==own['inputs'] and same(gs['own_protocol'],b['geometry_validation_protocol']) and gs['new_production_calls']==0,'geometry STARTED authority')
    require(score['schema']=='repair_postseal_proxy_scoring_v13' and score['complete'] is True and score['scored_rows']==490 and score['frames']==245 and score['methods']==list(METHODS[4:]) and score['GT_access_only_after_complete_geometry_and_cleanup'] is True and same(score['predictions'],b['new_scores']) and same(score['protocol'],b['protocol']) and same(score['geometry_seal'],b['geometry_seal']),'actual inherited paired490 scorer')
    ss=read(f['scoring_started']);require(ss['configured_scored_rows']==490 and ss['GT_only_after_full_seal_cleanup_and_independent_PASS'] is True and same(ss['independent_validation'],b['geometry_validation_protocol']) and same(ss['geometry_seal'],b['geometry_seal']),'score after actual checker PASS')
    hs=read(f['evaluation:hybrid_scoring']);require(hs['schema']=='observation_only_cornerwise_posthoc_scoring_v4' and hs['complete'] is True and hs['frames']==245 and hs['scored_method_rows']==1960 and hs['methods']==list(PARENT_METHODS) and same(hs['predictions'],b['hybrid_scores']),'immutable original V7 score authority')
    bs=read(f['evaluation:boundary_scoring']);require(bs['complete'] is True and bs['scored_rows']==245 and bs['methods']==[METHODS[3]] and same(bs['predictions'],b['boundary_scores']),'immutable original boundary score authority')
    for k,v in stats['inputs'].items():require(same(v,b[k]),'producing statistics byte authority '+k)
    st=read(f['statistics_started']);require(st['inputs']==stats['inputs'] and st['configured_new_rows']==490 and st['configured_saved_score_rows']==1470 and st['methods']==list(METHODS) and st['contrasts']==[list(v) for v in CONTRASTS] and st['new_seed_draws']==0,'statistics locked before own arithmetic')
    return stats

def main():
    h,witness=kernel();original_write=h['write'];h['inputs']=lambda args:input_paths(args,h);h['metadata']=lambda f,b:metadata(f,b,h)
    def write(path,value):
        if 'inputs' not in value:value['inputs']=h['read'](path.parent/FILES[0])['inputs']
        value['source_topology_adapter']=witness;value['external_declared_metadata']={k:v for k,v in h['read'](h['public_path'](value['inputs']['evaluation_protocol']))['inputs'].items() if not public(v)}
        value['external_declarations_are_not_current_asset_verification']=True;original_write(path,value)
    h['write']=write
    def freeze(args):
        out=h['guard'](args,'freeze');f=h['inputs'](args);b={k:h['binding'](v) for k,v in f.items()};h['metadata'](f,b);_,info=h['extraction']()
        write(out/FILES[0],dict(schema='independent_hybrid_factor_subset_statistics_protocol_v16',inputs=b,helpers=info,limits=h['LIMITS'],frames=245,methods=list(METHODS),contrasts=[list(v) for v in CONTRASTS],scopes=list(h['SCOPES']),moment_scalars=972,paired_groups=45,metric_CI_slots=135,existing_sessions=13,existing_draws=10000,new_draws=0,frozen_after_full_geometry_cleanup_PASS_scoring_statistics_before_own_arithmetic=True))
    h['freeze']=freeze;h['main']()

if __name__=='__main__':main()
