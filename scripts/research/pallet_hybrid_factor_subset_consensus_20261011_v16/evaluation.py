"""Postseal paired hybrid evaluation with unchanged proxy and statistics formulas.

Two new245-row arms share the frozen input coordinates. Original V7 HYBRID,
V15 boundary-only, Base and N3 scores remain immutable. Reference poses are
geometric proxies on known DEV, not independently measured physical truth.
"""
import ast
from collections import Counter
import copy
import csv
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from . import runner as R
from . import pipeline_ready as P
from ..pallet_finite_edge_consensus_repair_20261011_v13 import evaluation as E13
from ..pallet_finite_edge_consensus_repair_20261011_v13 import pipeline as P13
from ..pallet_factor_subset_consensus_20261011_v15 import evaluation as E15
from ..pallet_sparse_local_line_20261010_v9 import common as C
from ..pallet_three_head_observation_20261010_v7 import common as C7
from ..pallet_three_head_observation_20261010_v7 import statistics as S

METHODS=('BASE','N3_SUBPIX',P.PARENT_METHOD,'FACTOR_SUBSET_FINITE_EDGE_CONSENSUS',*P.METHODS)
CONTRASTS=tuple((P.PRIMARY,m) for m in METHODS if m!=P.PRIMARY)
ARITHMETIC=dict(scored_rows=1470,new_rows=490,posthoc_rows=490,methods=6,strata=3,
    moment_distributions=162,moment_scalars=972,paired_groups=45,metric_CIs=135,new_bootstrap_draws=0)

def geometry_gate(args):
    core=R.frozen(args);seal,receipt,check,own,start=[C.read(R.DOC/n) for n in
        ('GEOMETRY_SEAL.json','CONTROL_RECEIPT.json','VALIDATION_CHECKS.json','VALIDATION_PROTOCOL.json','VALIDATION_STARTED.json')]
    C.require(seal['schema']=='hybrid_factor_subset_geometry_seal_v16' and seal['complete'] is True
        and seal['frames']==seal['ledger_rows']==245 and seal['rows']==490 and seal['GT_read_allowed'] is False
        and seal['cleanup_completed_before_seal'] is True,'complete245/490 GT-free hybrid seal required')
    C.require(receipt['schema']=='hybrid_factor_subset_consensus_execution_v16' and receipt['complete'] is True
        and receipt['cleanup_completed'] is True and receipt['error'] is None and receipt['cleanup_error'] is None,
        'complete restored hybrid execution required')
    C.require(check['schema']=='hybrid_factor_subset_scalar_checks_v16' and own['schema']=='hybrid_factor_subset_scalar_protocol_v16'
        and check['complete'] is True and check['passed'] is True and check['failure_count']==0 and check['exception'] is None
        and check['inputs']==own['inputs'] and check['new_production_calls']==0,'actual full paired geometry PASS required')
    C.require(start['inputs']==own['inputs'] and start['new_production_calls']==0,'actual checker STARTED required')
    C.bound(R.DOC/'VALIDATION_PROTOCOL.json',check['own_protocol'],'checker protocol')
    C.bound(R.DOC/'VALIDATION_PROTOCOL.json',start['own_protocol'],'STARTED protocol')
    C.bound(R.CODE/'validation_checks.py',check['checker'],'current checker source')
    C.require(C.same_binding(own['checker'],check['checker']),'own checker differs')
    for alias,name in (('geometry','GEOMETRY_SEALED.jsonl.gz'),('ledgers','CONTROL_LEDGERS.jsonl.gz'),
        ('repair_protocol','PROTOCOL.json'),('geometry_seal','GEOMETRY_SEAL.json'),('repair_receipt','CONTROL_RECEIPT.json'),
        ('parent_population_checks','PARENT_POPULATION_CHECKS.json')):C.bound(R.DOC/name,own['inputs'][alias],'checked '+alias)
    for key,name in (('protocol','PROTOCOL.json'),('geometry','GEOMETRY_SEALED.jsonl.gz'),('ledgers','CONTROL_LEDGERS.jsonl.gz'),
        ('control_receipt','CONTROL_RECEIPT.json'),('parent_population_checks','PARENT_POPULATION_CHECKS.json')):C.bound(R.DOC/name,seal[key],'sealed '+key)
    return core,seal,receipt,check

def isolated_score():
    nodes=[n for n in ast.parse(Path(E13.__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='score']
    C.require(len(nodes)==1,'one immutable full score function');original=nodes[0];canonical=ast.dump(original,include_attributes=False)
    target=ast.parse('from .pipeline import METHODS').body[0];replacement=ast.parse('METHODS = _hybrid_METHODS').body[0]
    changes=Counter()
    class Replace(ast.NodeTransformer):
        def visit_ImportFrom(self,node):
            if ast.dump(node,include_attributes=False)==ast.dump(target,include_attributes=False):changes['local_method_import']+=1;return copy.deepcopy(replacement)
            return self.generic_visit(node)
        def visit_keyword(self,node):
            if node.arg in ('configured_scored_rows','scored_rows') and isinstance(node.value,ast.Constant) and node.value.value==245:
                changes[node.arg]+=1;return ast.keyword(arg=node.arg,value=ast.Constant(490))
            return self.generic_visit(node)
        def visit_Compare(self,node):
            if ast.dump(node,include_attributes=False)==ast.dump(ast.parse('count==245',mode='eval').body,include_attributes=False):
                changes['completion_count']+=1;return ast.parse('count==490',mode='eval').body
            return self.generic_visit(node)
    changed=Replace().visit(copy.deepcopy(original));expected=dict(local_method_import=1,configured_scored_rows=1,scored_rows=1,completion_count=1)
    C.require(dict(changes)==expected,'exact four method/population topology sites')
    undo=Counter()
    class Undo(ast.NodeTransformer):
        def visit_Assign(self,node):
            if ast.dump(node,include_attributes=False)==ast.dump(replacement,include_attributes=False):undo['local_method_import']+=1;return copy.deepcopy(target)
            return self.generic_visit(node)
        def visit_keyword(self,node):
            if node.arg in ('configured_scored_rows','scored_rows') and isinstance(node.value,ast.Constant) and node.value.value==490:
                undo[node.arg]+=1;return ast.keyword(arg=node.arg,value=ast.Constant(245))
            return self.generic_visit(node)
        def visit_Compare(self,node):
            if ast.dump(node,include_attributes=False)==ast.dump(ast.parse('count==490',mode='eval').body,include_attributes=False):
                undo['completion_count']+=1;return ast.parse('count==245',mode='eval').body
            return self.generic_visit(node)
    C.require(ast.dump(Undo().visit(copy.deepcopy(changed)),include_attributes=False)==canonical and dict(undo)==expected,
        'whole canonical score function restoration')
    facade=SimpleNamespace(**vars(R));facade.A=SimpleNamespace(protect=R.protect);facade.audit_args=lambda args:args
    env=dict(vars(E13));env.update(R=facade,geometry_gate=geometry_gate,_hybrid_METHODS=P.METHODS)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[changed],type_ignores=[])),str(E13.__file__)+' [paired topology only]','exec'),env)
    return env['score'],dict(original_score_source=C.binding(Path(E13.__file__)),topology_AST_changes=expected,
        original_score_function_AST_sha256=hashlib.sha256(canonical.encode()).hexdigest(),
        isolated_score_function_AST_sha256=hashlib.sha256(ast.dump(changed,include_attributes=False).encode()).hexdigest(),
        full_canonical_score_function_restore_proven=True,score_reference_formula_AST_changes=0,
        original_module_globals_modified=False,immutable_parent_method_tuple_modified=False,original_module_package_context_preserved=True)

def statistics_witness():
    names=('moments','metric','summaries','paired');nodes=[n for n in ast.parse(Path(S.__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
    C.require({n.name for n in nodes}==set(names),'whole unchanged four statistical functions')
    return dict(source=C.binding(Path(S.__file__)),names=list(names),whole_functions_AST_sha256=hashlib.sha256(ast.dump(ast.Module(body=nodes,type_ignores=[]),include_attributes=False).encode()).hexdigest(),formula_changes=0,new_draws=0)

def evaluation_inputs(args):
    values={'geometry_input:'+k:v for k,v in R.inputs(args).items()}
    paths=dict(adapter=Path(__file__),unchanged_score_dispatch=Path(E13.__file__),unchanged_scalar_statistics=Path(S.__file__),
        cohort=Path(args.cohort),protocol=Path(args.protocol),geometry_seal=R.DOC/'GEOMETRY_SEAL.json',geometry=R.DOC/'GEOMETRY_SEALED.jsonl.gz',
        geometry_validation=R.DOC/'VALIDATION_CHECKS.json',validation_protocol=R.DOC/'VALIDATION_PROTOCOL.json',validation_started=R.DOC/'VALIDATION_STARTED.json',
        checker=R.CODE/'validation_checks.py',control_receipt=R.DOC/'CONTROL_RECEIPT.json',ledgers=R.DOC/'CONTROL_LEDGERS.jsonl.gz',
        parent_population_checks=R.DOC/'PARENT_POPULATION_CHECKS.json',same_role_observations=Path(args.parent)/'OBSERVATIONS.jsonl.gz',
        hybrid_scores=Path(args.parent)/'PREDICTIONS.jsonl.gz',hybrid_scoring=Path(args.parent)/'SCORING_RECEIPT.json',hybrid_protocol=Path(args.parent)/'PROTOCOL.json',
        boundary_scores=E15.R.DOC/'PREDICTIONS.jsonl.gz',boundary_scoring=E15.R.DOC/'SCORING_RECEIPT.json',
        boundary_evaluation=E15.R.DOC/'EVALUATION_PROTOCOL.json',boundary_authority=E15.R.DOC/'SCORING_AUTHORITY.json',
        fixed_scores=Path(args.v9_input)/'FIXED_PREDICTIONS.jsonl.gz',bootstrap=S.BOOTSTRAP)
    values.update({k:C.binding(p) for k,p in paths.items()});return values

def freeze(args):
    geometry_gate(args);bindings=evaluation_inputs(args);_,dispatch=isolated_score()
    C.write_new(R.output(args,'EVALUATION_PROTOCOL.json'),dict(schema='postseal_hybrid_factor_subset_evaluation_protocol_v16',inputs=bindings,
        methods=list(METHODS),new_methods=list(P.METHODS),contrasts=[list(v) for v in CONTRASTS],frames=245,scored_rows=1470,new_scored_rows=490,
        moment_distributions=162,moment_scalars=972,paired_groups=45,metric_CIs=135,score_dispatch=dispatch,statistics_dispatch=statistics_witness(),
        reference='unchanged N3-phase GEOMETRIC_PROXY known DEV; not independent physical pose GT',
        original_frozen_sources_and_failures_preserved=True,new_seed_draws=0,new_models_fits_RGB=0,inherited_score_receipt_schema_is_v13=True,
        frozen_before_own_GT_access_or_statistic_arithmetic=True))

def frozen(args):
    geometry_gate(args);own=C.read(R.DOC/'EVALUATION_PROTOCOL.json');_,dispatch=isolated_score()
    C.require(own['schema']=='postseal_hybrid_factor_subset_evaluation_protocol_v16' and own['inputs']==evaluation_inputs(args)
        and own['methods']==list(METHODS) and own['new_methods']==list(P.METHODS) and own['contrasts']==[list(v) for v in CONTRASTS]
        and own['score_dispatch']==dispatch and own['statistics_dispatch']==statistics_witness(),'frozen paired evaluation differs');return own

def score(args):
    own=frozen(args);fn,dispatch=isolated_score();before=(P13.METHODS,E13.R,E13.geometry_gate)
    C.write_new(R.output(args,'SCORING_AUTHORITY.json'),dict(schema='isolated_hybrid_proxy_score_authority_v16',
        own_evaluation_protocol=C.binding(R.DOC/'EVALUATION_PROTOCOL.json'),inputs=own['inputs'],methods=list(P.METHODS),
        score_dispatch=dispatch,reference_or_error_formula_changed=False,original_module_globals_modified=False,new_independent_geometry_check_PASS=True))
    try:fn(args)
    finally:C.require(before==(P13.METHODS,E13.R,E13.geometry_gate),'parent globals changed')
    frozen(args)

def score_gate(args):
    own=frozen(args);r=C.read(R.DOC/'SCORING_RECEIPT.json');a=C.read(R.DOC/'SCORING_AUTHORITY.json');s=C.read(R.DOC/'SCORING_STARTED.json')
    C.require(r['schema']=='repair_postseal_proxy_scoring_v13' and r['complete'] is True and r['scored_rows']==490 and r['frames']==245
        and r['methods']==list(P.METHODS) and r['GT_access_only_after_complete_geometry_and_cleanup'] is True,'actual paired490 scores required')
    C.require(a['schema']=='isolated_hybrid_proxy_score_authority_v16' and a['inputs']==own['inputs'] and a['score_dispatch']==own['score_dispatch']
        and a['methods']==list(P.METHODS) and a['original_module_globals_modified'] is False and a['reference_or_error_formula_changed'] is False,'isolated paired authority differs')
    for path,item in ((R.DOC/'EVALUATION_PROTOCOL.json',a['own_evaluation_protocol']),(R.DOC/'PREDICTIONS.jsonl.gz',r['predictions']),
        (R.DOC/'PROTOCOL.json',r['protocol']),(R.DOC/'GEOMETRY_SEAL.json',r['geometry_seal'])):C.bound(path,item,'actual score bytes')
    C.require(s['configured_scored_rows']==490 and s['GT_only_after_full_seal_cleanup_and_independent_PASS'] is True
        and C.same_binding(s['protocol'],r['protocol']) and C.same_binding(s['independent_validation'],C.binding(R.DOC/'VALIDATION_PROTOCOL.json')),'score STARTED differs')
    return own,r

def posthoc(row,n3,observation):
    result=S.posthoc(dict(row,method=P.PARENT_METHOD),n3,observation);result['method']=row['method'];result['posthoc_kernel_label_adapter_only']=True
    result['hybrid_factor_subset_diagnostics']=dict(solver_state=row['solver']['state'],selected_hybrid_corner_ids=row['selected_corner_ids'],
        fit_point_ids=row['solver']['fit_input_ids'] if row['new_pose_estimated'] else [],fit_line_edges=row['solver']['fit_line_edges'] if row['new_pose_estimated'] else [],
        same_V7_corner_selection=True,new_corner_selection=False,point_only_ablation=row['partial_lines_not_pose_inputs'],
        physical_ownership_certified=False,global_uniqueness_proven=False,numerical_NEW_is_not_goal_success=True)
    return result

def stats(args):
    import numpy as np
    own,_=score_gate(args);parent=C.read(Path(args.parent)/'SCORING_RECEIPT.json');boundary=C.read(E15.R.DOC/'SCORING_RECEIPT.json')
    C.require(parent['complete'] is True and parent['frames']==245 and parent['scored_method_rows']==1960 and parent['methods']==list(C7.METHODS),'immutable full V7 scores required')
    C.require(boundary['complete'] is True and boundary['scored_rows']==245 and boundary['methods']==[METHODS[3]],'immutable V15 boundary-only scores required')
    C.bound(Path(args.parent)/'PREDICTIONS.jsonl.gz',parent['predictions'],'V7 reference scores');C.bound(E15.R.DOC/'PREDICTIONS.jsonl.gz',boundary['predictions'],'V15 boundary scores')
    paths=dict(new_scores=R.DOC/'PREDICTIONS.jsonl.gz',hybrid_scores=Path(args.parent)/'PREDICTIONS.jsonl.gz',boundary_scores=E15.R.DOC/'PREDICTIONS.jsonl.gz',
        fixed_scores=Path(args.v9_input)/'FIXED_PREDICTIONS.jsonl.gz',cohort=Path(args.cohort),bootstrap=S.BOOTSTRAP,statistics_kernel=Path(S.__file__),own_code=Path(__file__),
        scoring_receipt=R.DOC/'SCORING_RECEIPT.json',scoring_started=R.DOC/'SCORING_STARTED.json',protocol=Path(args.protocol),geometry_seal=R.DOC/'GEOMETRY_SEAL.json',
        geometry_validation=R.DOC/'VALIDATION_CHECKS.json',validation_protocol=R.DOC/'VALIDATION_PROTOCOL.json',validation_started=R.DOC/'VALIDATION_STARTED.json',
        evaluation_protocol=R.DOC/'EVALUATION_PROTOCOL.json',scoring_authority=R.DOC/'SCORING_AUTHORITY.json',same_role_observations=Path(args.parent)/'OBSERVATIONS.jsonl.gz')
    bindings={k:C.binding(p) for k,p in paths.items()}
    for n in ('STATISTICS_STARTED.json','METRICS.json','METRICS.csv','POSTHOC_ROWS.jsonl.gz'):R.output(args,n)
    C.write_new(R.output(args,'STATISTICS_STARTED.json'),dict(inputs=bindings,new_seed_draws=0,configured_new_rows=490,configured_saved_score_rows=1470,
        methods=list(METHODS),contrasts=[list(v) for v in CONTRASTS],known_DEV_only=True))
    groups={m:{} for m in METHODS};hybrid_population={m:set() for m in C7.METHODS}
    for key in ('fixed_scores','hybrid_scores','boundary_scores','new_scores'):
        for full in C.rows(paths[key]):
            if key=='hybrid_scores':
                C.require(full['method'] in hybrid_population and full['id'] not in hybrid_population[full['method']],'unique original V7 method/ID')
                hybrid_population[full['method']].add(full['id'])
                if full['method']!=P.PARENT_METHOD:continue
            row=E15.compact(full);del full
            C.require(row['method'] in groups and row['id'] not in groups[row['method']],'unique six-method/ID')
            groups[row['method']][row['id']]=row
    cohort=C.read(args.cohort);ids=cohort['ids'];labels={f['id']:f['label'] for f in cohort['frames']}
    C.require(len(ids)==245 and all(set(g)==set(ids) for g in groups.values()) and all(g==set(ids) for g in hybrid_population.values()),'six methods and all8 original methods retain245')
    with gzip.open(S.BOOTSTRAP,'rt') as stream:bootstrap=json.load(stream)
    draws=np.asarray(bootstrap['counts'],np.int64);sessions=bootstrap['sessions']
    C.require(draws.shape==(10000,13) and len(set(sessions))==13 and (draws>=0).all() and (draws.sum(1)==13).all(),'reuse original13session10000draws')
    index={session:i for i,session in enumerate(sessions)};observations={r['id']:r for r in C.rows(paths['same_role_observations']) if r['head_arm']=='IMAGE_ROLE'}
    C.require(set(observations)==set(ids),'full sameROLE245 observations')
    post=[posthoc(groups[method][fid],groups['N3_SUBPIX'][fid],observations[fid]) for method in P.METHODS for fid in ids];strata={}
    for name,selected in (('combined',ids),('easy',[fid for fid in ids if labels[fid]=='clean']),('medium',[fid for fid in ids if labels[fid]=='moderate'])):
        strata[name]=dict(frames=len(selected),ids=selected,methods={m:S.summaries([groups[m][fid] for fid in selected]) for m in METHODS},
            contrasts={a+'_minus_'+b:{scope:S.paired(groups[a],groups[b],selected,index,draws,scope) for scope in S.SCOPES} for a,b in CONTRASTS},
            diagnostics={m:S.diagnostic_summary([r for r in post if r['method']==m and r['id'] in set(selected)]) for m in P.METHODS})
    primary=strata['combined']['contrasts'][P.PRIMARY+'_minus_N3_SUBPIX']['common_operational']
    improved=primary['common_frames']==245 and all(primary['metrics'][m]['mean_delta']<0 for m in ('translation_cm','rotation_deg'))
    C.require({k:C.binding(p) for k,p in paths.items()}==bindings,'statistics bytes changed')
    S.C.save_rows(R.output(args,'POSTHOC_ROWS.jsonl.gz'),post)
    C.write_new(R.output(args,'METRICS.json'),dict(schema='hybrid_factor_subset_raw_row_metrics_v16',complete=True,methods=list(METHODS),strata=strata,
        contrasts=[list(v) for v in CONTRASTS],scopes=list(S.SCOPES),inputs=bindings,posthoc_rows=C.binding(R.DOC/'POSTHOC_ROWS.jsonl.gz'),actual_arithmetic=ARITHMETIC,
        original_V7_scored_rows_streamed=1960,primary_both_T_R_means_better_than_N3_on_full245=improved,reference=own['reference'],
        new_model_image_GT_score_PnP_optimizer_training_RGB_calls=0,statistics_does_not_change_proposals_or_operating_status=True))
    with R.output(args,'METRICS.csv').open('x',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=('stratum','method','population','metric','n','mean','sample_variance','sample_std','median','P90','max','unit'));writer.writeheader()
        for name,value in strata.items():
            for method,summary in value['methods'].items():
                for scope,metrics in summary['metrics'].items():
                    for metric,moment in metrics.items():writer.writerow(dict(stratum=name,method=method,population=scope,metric=metric,**moment))
    print('HYBRID_FACTOR_SUBSET_METRICS',improved,flush=True)

def main():
    p=R.parser();next(a for a in p._actions if a.dest=='stage').choices=('freeze','score','stats');args=p.parse_args()
    {'freeze':freeze,'score':score,'stats':stats}[args.stage](args)

if __name__=='__main__':main()
