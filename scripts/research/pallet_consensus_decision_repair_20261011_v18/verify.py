"""Own-frozen stdlib recheck of 1470 saved scores, 972 moments and 135 CIs.

The immutable complete V13 verifier supplies every statistical arithmetic
statement. Only stream/count topology and primary lookup change. Explicit
external dependency declarations remain metadata and are never opened.
"""
import ast
from collections import Counter
import hashlib
from pathlib import Path

REPO=Path(__file__).resolve().parents[3];CODE=Path(__file__).resolve().parent
DOC=REPO/'_docs/experiments/pallet_consensus_decision_repair_20261011_v18'
SOURCE=REPO/'scripts/research/pallet_finite_edge_consensus_repair_20261011_v13/verify.py'
SOURCE_SHA='bad35cd12dd449d2f02e772e715b4dc3b04c74f74b1e34f508a0288bb53e018e'
METHODS=('BASE','N3_SUBPIX','HYBRID_STANDARD_PNP','NATIVE_STANDARD_PNP','HYBRID_POINT_CONSENSUS_ORIGINAL','HYBRID_POINT_CONSENSUS_RESTORED')
CONTRASTS=((METHODS[-1],METHODS[-2]),(METHODS[-1],'N3_SUBPIX'),(METHODS[-1],'NATIVE_STANDARD_PNP'),(METHODS[-1],'HYBRID_STANDARD_PNP'),(METHODS[-2],'N3_SUBPIX'))
FILES=('STATISTICS_VERIFICATION_PROTOCOL.json','STATISTICS_VERIFICATION_STARTED.json','VERIFICATION.json')
ARITHMETIC=dict(scored_rows=1470,new_rows=490,posthoc_rows=490,methods=6,strata=3,moment_distributions=162,moment_scalars=972,paired_groups=45,metric_CIs=135,new_bootstrap_draws=0)

def require(value,message):
    if not value:raise ValueError(message)

def kernel():
    require(hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SOURCE_SHA,'immutable full V13 statistical verifier')
    original_tree=ast.parse(SOURCE.read_text());original=ast.dump(original_tree,include_attributes=False)
    changes={
        'saved_stream_topology':("(('new_scores',245,{METHODS[-1]}),('old_local_scores',245,{METHODS[3]}),('point_scores',245,{METHODS[2]}),('fixed_scores',490,set(METHODS[:2])))","(('new_scores',490,set(METHODS[4:])),('ordinary_scores',490,set(METHODS[2:4])),('fixed_scores',490,set(METHODS[:2])))"),
        'fixed_complete_counts':("dict(saved_scored_rows_streamed=1225,moment_scalars_checked=810,distributions_checked=135,pair_groups_checked=27,metric_CI_slots_checked=81)","dict(saved_scored_rows_streamed=1470,moment_scalars_checked=972,distributions_checked=162,pair_groups_checked=45,metric_CI_slots_checked=135)"),
        'primary_method_lookup':("stats['strata']['combined']['contrasts']['FINITE_EDGE_CONSENSUS_minus_N3_SUBPIX']['common_operational']","stats['strata']['combined']['contrasts'][METHODS[-1]+'_minus_N3_SUBPIX']['common_operational']")}
    before={k:ast.parse(v[0],mode='eval').body for k,v in changes.items()};after={k:ast.parse(v[1],mode='eval').body for k,v in changes.items()};hits=Counter()
    class Replace(ast.NodeTransformer):
        def visit(self,node):
            for k,value in before.items():
                if ast.dump(node,include_attributes=False)==ast.dump(value,include_attributes=False):hits[k]+=1;return after[k]
            return super().visit(node)
    tree=Replace().visit(original_tree);require(dict(hits)=={k:1 for k in changes},'exact three topology sites');derived=ast.dump(tree,include_attributes=False)
    class Undo(ast.NodeTransformer):
        def visit(self,node):
            for k,value in after.items():
                if ast.dump(node,include_attributes=False)==ast.dump(value,include_attributes=False):return before[k]
            return super().visit(node)
    require(ast.dump(Undo().visit(ast.parse(ast.unparse(tree))),include_attributes=False)==original,'whole canonical arithmetic restoration')
    hits.clear();tree=Replace().visit(ast.parse(SOURCE.read_text()));require(dict(hits)=={k:1 for k in changes},'same three compilation sites')
    versions={'independent_finite_edge_statistics_protocol_v13':'independent_point_consensus_decision_statistics_protocol_v18','independent_finite_edge_statistics_checks_v13':'independent_point_consensus_decision_statistics_checks_v18'}
    class Version(ast.NodeTransformer):
        def visit_Constant(self,node):return ast.copy_location(ast.Constant(versions[node.value]),node) if isinstance(node.value,str) and node.value in versions else node
    tree=ast.fix_missing_locations(Version().visit(tree));witness=dict(original_source_sha256=SOURCE_SHA,
        original_full_AST_sha256=hashlib.sha256(original.encode()).hexdigest(),data_topology_AST_sites=list(changes),data_topology_AST_changes=3,
        derived_arithmetic_AST_sha256=hashlib.sha256(derived.encode()).hexdigest(),scalar_formula_AST_changes=0,receipt_schema_only_versions=versions)
    h=dict(__file__=str(Path(__file__).resolve()),__name__='isolated_point_consensus_decision_statistics')
    exec(compile(tree,str(SOURCE)+' [unchanged arithmetic; point decision paired topology]','exec'),h)
    h.update(DOC=DOC,PRIVATE=Path('/tmp/pallet-consensus-decision-repair-private-20261011-v18'),METHODS=METHODS,CONTRASTS=CONTRASTS,FILES=FILES)
    return h,witness

def public(item):
    if item.get('origin')=='external_readonly_dependency':return False
    p=Path(item['path']);require(item.get('origin')=='public_repository' and not p.is_absolute() and '..' not in p.parts and p.parts and p.parts[0] in ('scripts','_docs'),'explicit public repository binding')
    return True

def input_paths(args,h):
    p=args.input;stats=h['read'](p/'METRICS.json');f={k:h['public_path'](v) for k,v in stats['inputs'].items() if public(v)}
    names=dict(METRICS='METRICS.json',statistics_started='STATISTICS_STARTED.json',scoring_started='SCORING_STARTED.json',scoring_receipt='SCORING_RECEIPT.json',
        geometry_seal='GEOMETRY_SEAL.json',repair_receipt='CONTROL_RECEIPT.json',geometry='GEOMETRY_SEALED.jsonl.gz',ledgers='CONTROL_LEDGERS.jsonl.gz',population='PARENT_POPULATION_CHECKS.json',
        geometry_validation='VALIDATION_CHECKS.json',geometry_validation_protocol='VALIDATION_PROTOCOL.json',geometry_validation_started='VALIDATION_STARTED.json',
        evaluation_protocol='EVALUATION_PROTOCOL.json',scoring_authority='SCORING_AUTHORITY.json',protected_originals='PROTECTION_BEFORE.json')
    f.update({k:p/v for k,v in names.items()});f.update(checker=Path(__file__).resolve(),geometry_checker=CODE/'validation_checks.py',
        scalar_source=h['SOURCE'],original_verifier=SOURCE,protocol=Path(args.protocol))
    for k,v in h['read'](f['evaluation_protocol'])['inputs'].items():
        if public(v):f['evaluation:'+k]=h['public_path'](v)
    require(f['new_scores']==p/'PREDICTIONS.jsonl.gz' and Path(args.protocol).resolve()==h['public_path'](stats['inputs']['protocol']).resolve(),
        'exact paired new score stream and accuracy protocol')
    return f


def metadata(f,b,h):
    read,same=h['read'],h['same'];stats,core,score,seal,r,check,own=[read(f[k]) for k in
        ('METRICS','protocol','scoring_receipt','geometry_seal','repair_receipt','geometry_validation','geometry_validation_protocol')]
    require(stats['schema']=='point_consensus_decision_raw_row_metrics_v18' and stats['complete'] is True
        and stats['methods']==list(METHODS) and stats['contrasts']==[list(v) for v in CONTRASTS]
        and stats['scopes']==list(h['SCOPES']) and stats['actual_arithmetic']==ARITHMETIC
        and stats['V16_full_scalar_validation_passed'] is False and stats['same_observation_pair_source_join_verified'] is True,
        'fixed6/1470/972/45/135 saved statistics')
    e,a=read(f['evaluation_protocol']),read(f['scoring_authority'])
    require(e['schema']=='postseal_point_consensus_decision_evaluation_protocol_v18'
        and e['methods']==list(METHODS) and e['new_methods']==list(METHODS[4:]) and e['contrasts']==[list(v) for v in CONTRASTS]
        and e['frames']==245 and e['scored_rows']==1470 and e['new_scored_rows']==490
        and e['moment_distributions']==162 and e['moment_scalars']==972 and e['paired_groups']==45 and e['metric_CIs']==135
        and e['new_seed_draws']==0 and e['V16_full_scalar_validation_passed'] is False
        and e['frozen_before_own_GT_access_or_statistic_arithmetic'] is True,'own point-only evaluation fixed before score/statistics')
    for k,v in e['inputs'].items():
        if public(v):require(same(v,b['evaluation:'+k]),'current public evaluation input '+k)
    dispatch=e['score_dispatch']
    require(dispatch['topology_AST_changes']==dict(local_method_import=1,configured_scored_rows=1,scored_rows=1,completion_count=1)
        and dispatch['full_canonical_score_function_restore_proven'] is True and dispatch['score_reference_formula_AST_changes']==0
        and dispatch['original_module_globals_modified'] is False and dispatch['immutable_parent_method_tuple_modified'] is False
        and dispatch['original_module_package_context_preserved'] is True
        and same(dispatch['original_score_source'],b['evaluation:unchanged_score_dispatch']), 'unchanged whole scorer except four topology sites')
    require(e['statistics_dispatch']['formula_changes']==e['statistics_dispatch']['new_draws']==0
        and e['statistics_dispatch']['names']==['moments','metric','summaries','paired']
        and same(e['statistics_dispatch']['source'],b['statistics_kernel']), 'whole unchanged statistical kernels')
    require(a['schema']=='isolated_point_consensus_decision_proxy_score_authority_v18' and a['inputs']==e['inputs']
        and a['methods']==list(METHODS[4:]) and a['score_dispatch']==dispatch
        and a['original_module_globals_modified'] is False and a['reference_or_error_formula_changed'] is False
        and a['new_independent_point_only_decision_check_PASS'] is True and a['V16_full_scalar_validation_passed'] is False
        and same(a['own_evaluation_protocol'],b['evaluation_protocol']), 'isolated paired actual score authority')
    aliases=(('adapter','own_code'),('unchanged_scalar_statistics','statistics_kernel'),('cohort','cohort'),('protocol','protocol'),
        ('geometry_seal','geometry_seal'),('geometry','geometry'),('geometry_validation','geometry_validation'),
        ('validation_protocol','geometry_validation_protocol'),('validation_started','geometry_validation_started'),('checker','geometry_checker'),
        ('control_receipt','repair_receipt'),('ledgers','ledgers'),('parent_population_checks','population'),('fixed_scores','fixed_scores'),('bootstrap','bootstrap'))
    for key,alias in aliases:require(same(e['inputs'][key],b[alias]),'frozen evaluation byte closure '+key)
    require(core['schema']=='fixed_point_consensus_decision_restoration_protocol_v18' and core['methods']==list(METHODS[4:])
        and core['frames']==core['ledger_rows']==245 and core['rows']==490,'paired fixed decision geometry')
    require(seal['schema']=='point_consensus_decision_restoration_geometry_seal_v18' and seal['complete'] is True
        and seal['frames']==seal['ledger_rows']==245 and seal['rows']==490 and seal['GT_read_allowed'] is False
        and seal['cleanup_completed_before_seal'] is True and seal['original_full_paired_scalar_passed'] is False
        and r['schema']=='point_consensus_decision_restoration_execution_v18' and r['complete'] is True
        and r['cleanup_completed'] is True and r['error'] is None and r['cleanup_error'] is None
        and r['original_full_paired_scalar_passed'] is False and r['new_numerical_pose_fit_calls']==0,
        'full paired GT-free decision seal and cleanup, preserving full V16 FAIL')
    for key,alias in (('protocol','protocol'),('geometry','geometry'),('ledgers','ledgers'),('control_receipt','repair_receipt'),('parent_population_checks','population')):
        require(same(seal[key],b[alias]),'current sealed bytes '+key)
    require(own['schema']=='point_consensus_decision_scalar_protocol_v18' and check['schema']=='point_consensus_decision_scalar_checks_v18'
        and check['complete'] is True and check['passed'] is True and check['failure_count']==0 and check['exception'] is None
        and check['new_production_calls']==0 and check['inputs']==own['inputs'] and same(check['own_protocol'],b['geometry_validation_protocol'])
        and same(check['checker'],b['geometry_checker']) and same(own['checker'],b['geometry_checker']), 'actual point-only decision/output PASS')
    for key,alias in (('geometry','geometry'),('ledgers','ledgers'),('repair_protocol','protocol'),('geometry_seal','geometry_seal'),
        ('repair_receipt','repair_receipt'),('parent_population_checks','population'),('checker','geometry_checker')):
        require(same(own['inputs'][key],b[alias]),'current point-only checked bytes '+key)
    scope=check['scope']
    require(scope['original_full_scalar_validation_passed'] is False and scope['point_only_original_failed_predicates']==0
        and scope['independent_parent_point_check_count_not_claimed'] is True,'limited point-only PASS must not certify full V16')
    gs=read(f['geometry_validation_started'])
    require(gs['inputs']==own['inputs'] and same(gs['own_protocol'],b['geometry_validation_protocol']) and gs['new_production_calls']==0,
        'point-only inspector STARTED authority')
    parent=read(f['evaluation:preserved_V16_full_FAIL'])
    require(parent['complete'] is True and parent['passed'] is False and parent['checks']==198134007
        and parent['failure_count']==564 and parent['exception'] is None,'actual full V16 FAIL564 preserved')
    require(score['schema']=='repair_postseal_proxy_scoring_v13' and score['complete'] is True and score['scored_rows']==490
        and score['frames']==245 and score['methods']==list(METHODS[4:]) and score['GT_access_only_after_complete_geometry_and_cleanup'] is True
        and same(score['predictions'],b['new_scores']) and same(score['protocol'],b['protocol'])
        and same(score['geometry_seal'],b['geometry_seal']), 'actual inherited paired490 scorer')
    ss=read(f['scoring_started'])
    require(ss['configured_scored_rows']==490 and ss['GT_only_after_full_seal_cleanup_and_independent_PASS'] is True
        and same(ss['independent_validation'],b['geometry_validation_protocol']) and same(ss['geometry_seal'],b['geometry_seal']),
        'score after actual point-only PASS')
    fs=read(f['evaluation:fixed_scoring'])
    require(fs['complete'] is True and fs['frames']==245 and fs['fixed_control_rows']==490
        and same(fs['fixed_predictions'],b['fixed_scores']),'immutable Base/N3 fixed score authority')
    # Completed V17 ordinary rows and their own independent checks stay immutable.
    oscore,oseal,og,ogp,ogs,om,ov,ovp,ovs=[read(f['evaluation:ordinary:'+name]) for name in
        ('SCORING_RECEIPT.json','GEOMETRY_SEAL.json','VALIDATION_CHECKS.json','VALIDATION_PROTOCOL.json','VALIDATION_STARTED.json',
        'METRICS.json','VERIFICATION.json','STATISTICS_VERIFICATION_PROTOCOL.json','STATISTICS_VERIFICATION_STARTED.json')]
    require(oscore['complete'] is True and oscore['scored_rows']==490 and oscore['frames']==245
        and oscore['methods']==list(METHODS[2:4]) and same(oscore['predictions'],b['ordinary_scores'])
        and same(e['inputs']['ordinary:PREDICTIONS.jsonl.gz'],b['ordinary_scores']), 'immutable ordinary490 score join')
    require(oseal['complete'] is True and oseal['frames']==oseal['ledger_rows']==245 and oseal['rows']==490
        and og['schema']=='native_hybrid_standard_scalar_checks_v17' and og['complete'] is True and og['passed'] is True
        and og['checks']==394552 and og['failure_count']==0 and og['exception'] is None and og['inputs']==ogp['inputs']==ogs['inputs'],
        'actual V17 geometry394552 PASS preserved')
    require(om['schema']=='native_hybrid_standard_raw_row_metrics_v17' and om['complete'] is True
        and ov['schema']=='independent_native_hybrid_standard_statistics_checks_v17' and ov['complete'] is True and ov['passed'] is True
        and ov['failure_count']==0 and ov['error'] is None and ov['actual_counts']['saved_scored_rows_streamed']==980
        and ov['actual_counts']['moment_scalars_checked']==648 and ov['actual_counts']['metric_CI_slots_checked']==81
        and ov['inputs']==ovp['inputs']==ovs['inputs'] and same(ov['inputs']['METRICS'],b['evaluation:ordinary:METRICS.json'])
        and same(ov['inputs']['new_scores'],b['ordinary_scores']), 'actual V17 statistics980/648/81 PASS preserved')
    for key,v in stats['inputs'].items():
        if public(v):require(same(v,b[key]),'producing statistics byte authority '+key)
    st=read(f['statistics_started'])
    require(st['inputs']==stats['inputs'] and st['configured_new_rows']==490 and st['configured_saved_score_rows']==1470
        and st['methods']==list(METHODS) and st['contrasts']==[list(v) for v in CONTRASTS] and st['new_seed_draws']==0,
        'statistics locked before own arithmetic')
    require(same(core['inputs']['own:PROTECTION_BEFORE.json'],b['protected_originals']), 'source protection snapshot binding')
    protected=read(f['protected_originals']);tracked=protected.get('tracked',{}).get('files')
    require(isinstance(tracked,list) and all(isinstance(v,dict) and 'path' in v for v in tracked), 'wrapped tracked.files source-protection records')
    require(len({v['path'] for v in tracked})==len(tracked),'duplicate protected tracked original path')
    additional=protected.get('additional_completed_audits',[])
    require(isinstance(additional,list) and all(isinstance(v,dict) and 'path' in v for v in additional),'additional immutable audit record schema')
    return stats


def main():
    h,witness=kernel();original_write=h['write'];h['inputs']=lambda args:input_paths(args,h);h['metadata']=lambda f,b:metadata(f,b,h)
    def write(path,value):
        if 'inputs' not in value:value['inputs']=h['read'](path.parent/FILES[0])['inputs']
        value['source_topology_adapter']=witness
        e=h['read'](h['public_path'](value['inputs']['evaluation_protocol']))
        value['external_declared_metadata']={k:v for k,v in e['inputs'].items() if not public(v)}
        value['external_declarations_are_not_current_asset_verification']=True
        value['V16_original_full_scalar_validation_passed']=False
        value['geometry_gate_scope']='point-only paired decision/output audit; not full V16 line-scalar PASS or new independent parent check count'
        if path.name==FILES[2] and value.get('passed'):
            stats=h['read'](h['public_path'](value['inputs']['METRICS']))
            for comparator,field in ((METHODS[-2],'primary_both_T_R_means_better_than_original_on_full245'),
                ('NATIVE_STANDARD_PNP','primary_both_T_R_means_better_than_native_standard_on_full245'),
                ('HYBRID_STANDARD_PNP','primary_both_T_R_means_better_than_hybrid_standard_on_full245')):
                pair=stats['strata']['combined']['contrasts'][METHODS[-1]+'_minus_'+comparator]['common_operational']
                expected=pair['common_frames']==245 and all(pair['metrics'][m]['mean_delta']<0 for m in ('translation_cm','rotation_deg'))
                require(stats[field]==expected,'saved comparative verdict after independent pair verification '+comparator)
            value['saved_comparative_verdicts_checked_from_independently_verified_pairs']=True
        original_write(path,value)
    h['write']=write
    def freeze(args):
        out=h['guard'](args,'freeze');f=h['inputs'](args);b={k:h['binding'](v) for k,v in f.items()};h['metadata'](f,b);_,info=h['extraction']()
        write(out/FILES[0],dict(schema='independent_point_consensus_decision_statistics_protocol_v18',inputs=b,helpers=info,limits=h['LIMITS'],
            frames=245,methods=list(METHODS),contrasts=[list(v) for v in CONTRASTS],scopes=list(h['SCOPES']),moment_scalars=972,
            paired_groups=45,metric_CI_slots=135,existing_sessions=13,existing_draws=10000,new_draws=0,
            frozen_after_full_geometry_cleanup_PASS_scoring_statistics_before_own_arithmetic=True))
    h['freeze']=freeze;h['main']()

if __name__=='__main__':main()
