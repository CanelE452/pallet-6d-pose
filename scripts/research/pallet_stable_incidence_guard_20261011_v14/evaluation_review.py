"""Isolate the scored method name without changing immutable parent globals.

The first dispatch failed before STARTED or GT. Its source and artifacts are
preserved. This review changes one local import in the unchanged V13 score
driver; the original score/reference/statistic formulas are not rewritten.
"""
import ast
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

from . import evaluation as E
from . import runner as R
from . import pipeline as G
from ..pallet_finite_edge_consensus_repair_20261011_v13 import evaluation as E13
from ..pallet_finite_edge_consensus_repair_20261011_v13 import pipeline as P13
from ..pallet_sparse_local_line_20261010_v9 import common as C

FILES=('SCORING_DISPATCH_REVIEW_PROTOCOL.json','SCORING_DISPATCH_REVIEW_STARTED.json','SCORING_AUTHORITY_SUPPLEMENT.json')


def inputs(args):
    paths=dict(review_code=Path(__file__),original_adapter=Path(E.__file__),original_score_driver=Path(E13.__file__),
        original_evaluation_protocol=R.DOC/'EVALUATION_PROTOCOL.json',original_dispatch_authority=R.DOC/'SCORING_AUTHORITY.json',
        original_dispatch_failure=R.DOC/'SCORING_DISPATCH_FIRST_FAILURE.json',immutable_parent_pipeline=Path(P13.__file__))
    values={k:C.binding(p) for k,p in paths.items()}
    values.update({'original_evaluation:'+k:v for k,v in E.evaluation_inputs(args).items()})
    return values


def history(args):
    E.frozen(args);failure=C.read(R.DOC/'SCORING_DISPATCH_FIRST_FAILURE.json')
    C.require(failure['schema']=='preserved_stable_incidence_score_dispatch_failure_v14' and failure['complete'] is False
        and failure['error']==dict(type='RuntimeError',message='SAME_OBSERVATION_GUARD: frozen repair source/policy/inputs differ')
        and failure['failed_before_SCORING_STARTED'] is True and failure['failed_before_GT_kernel_or_score_rows'] is True
        and failure['GT_score_calls']==failure['scored_rows']==0 and failure['source_files_changed'] is False
        and failure['original_frozen_parent_METHODS_restored_by_finally'] is True and failure['no_model_fit_training_RGB_work'] is True,
        'preserve the actual pre-GT dispatch failure')
    for name in ('EVALUATION_PROTOCOL.json','SCORING_AUTHORITY.json'):
        C.bound(R.DOC/name,failure['authority'][name],'preserved first dispatch '+name)
    C.bound(Path(E.__file__),failure['source'],'preserved original evaluation adapter')
    C.require(P13.METHODS==(G.PARENT_METHOD,),'immutable parent method tuple must remain unchanged')
    return failure


def reviewed_score():
    source=ast.parse(Path(E13.__file__).read_text())
    found=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='score']
    C.require(len(found)==1,'one unchanged V13 score function required')
    original=copy.deepcopy(found[0]);canonical=ast.dump(original,include_attributes=False)
    target=ast.parse('from .pipeline import METHODS').body[0]
    replacement=ast.parse('METHODS = _guard_METHODS').body[0]
    count=[]
    class Replace(ast.NodeTransformer):
        def visit_ImportFrom(self,node):
            if ast.dump(node,include_attributes=False)==ast.dump(target,include_attributes=False):
                count.append(1);return ast.copy_location(copy.deepcopy(replacement),node)
            return self.generic_visit(node)
    changed=Replace().visit(copy.deepcopy(original))
    C.require(len(count)==1,'exactly one local method import dispatch change')
    restored=[]
    class Undo(ast.NodeTransformer):
        def visit_Assign(self,node):
            if ast.dump(node,include_attributes=False)==ast.dump(replacement,include_attributes=False):
                restored.append(1);return ast.copy_location(copy.deepcopy(target),node)
            return self.generic_visit(node)
    C.require(ast.dump(Undo().visit(copy.deepcopy(changed)),include_attributes=False)==canonical and len(restored)==1,
        'canonical full score-function restoration proof')
    facade=SimpleNamespace(**vars(R));facade.A=SimpleNamespace(protect=R.protect);facade.audit_args=lambda a:a
    namespace=dict(vars(E13));namespace.update(R=facade,geometry_gate=E.geometry_gate,_guard_METHODS=G.METHODS)
    module=ast.fix_missing_locations(ast.Module(body=[changed],type_ignores=[]))
    exec(compile(module,str(E13.__file__)+' [isolated local method dispatch]','exec'),namespace)
    witness=dict(original_score_source=C.binding(Path(E13.__file__)),local_method_import_AST_changes=1,
        original_score_function_AST_sha256=hashlib.sha256(canonical.encode()).hexdigest(),
        reviewed_score_function_AST_sha256=hashlib.sha256(ast.dump(changed,include_attributes=False).encode()).hexdigest(),
        full_canonical_score_function_restore_proven=True,score_reference_formula_AST_changes=0,
        immutable_parent_method_tuple_modified=False,original_module_globals_modified=False,
        original_module_package_context_preserved=True)
    return namespace['score'],witness


def freeze(args):
    history(args);bindings=inputs(args);_,witness=reviewed_score()
    C.write_new(R.output(args,FILES[0]),dict(schema='isolated_guard_score_dispatch_review_protocol_v14',inputs=bindings,
        method=G.PRIMARY,frames=245,review=witness,original_failed_dispatch_preserved=True,
        frozen_before_own_GT_score_calls=True,new_models_fits_training_RGB_seed_calls=0))


def frozen(args):
    history(args);own=C.read(R.DOC/FILES[0]);_,witness=reviewed_score()
    C.require(own['schema']=='isolated_guard_score_dispatch_review_protocol_v14' and own['inputs']==inputs(args)
        and own['review']==witness and own['method']==G.PRIMARY and own['frames']==245,
        'own isolated score review source differs');return own


def score(args):
    own=frozen(args);fn,witness=reviewed_score();before=(P13.METHODS,E13.R,E13.geometry_gate)
    C.write_new(R.output(args,FILES[1]),dict(protocol=C.binding(R.DOC/FILES[0]),inputs=own['inputs'],review=witness,
        configured_scored_rows=245,GT_score_calls_before_STARTED=0))
    C.write_new(R.output(args,FILES[2]),dict(schema='isolated_guard_score_dispatch_authority_v14',
        own_protocol=C.binding(R.DOC/FILES[0]),started=C.binding(R.DOC/FILES[1]),inputs=own['inputs'],review=witness,
        original_failed_dispatch_preserved=True,method=G.PRIMARY,score_reference_formulas_changed=False))
    try:fn(args)
    finally:C.require(before==(P13.METHODS,E13.R,E13.geometry_gate),'original parent globals changed during isolated dispatch')
    frozen(args)


def geometry_gate(args):
    result=E.geometry_gate(args);own=frozen(args);authority=C.read(R.DOC/FILES[2]);started=C.read(R.DOC/FILES[1])
    C.require(authority['schema']=='isolated_guard_score_dispatch_authority_v14' and authority['inputs']==own['inputs']
        and authority['review']==own['review'] and authority['method']==G.PRIMARY and authority['score_reference_formulas_changed'] is False
        and authority['original_failed_dispatch_preserved'] is True and started['inputs']==own['inputs'] and started['review']==own['review'],
        'actual isolated score dispatch authority differs')
    C.bound(R.DOC/FILES[0],authority['own_protocol'],'actual isolated score protocol')
    C.bound(R.DOC/FILES[1],authority['started'],'actual isolated score STARTED')
    C.bound(R.DOC/FILES[0],started['protocol'],'isolated review STARTED protocol');return result


def stats(args):
    geometry_gate(args);own=frozen(args)
    C.write_new(R.output(args,'STATISTICS_AUTHORITY_SUPPLEMENT.json'),dict(
        schema='isolated_guard_score_statistics_authority_v14',own_review_protocol=C.binding(R.DOC/FILES[0]),
        review_source=C.binding(Path(__file__)),original_statistics_source=C.binding(Path(E.__file__)),
        successful_score_dispatch_authority=C.binding(R.DOC/FILES[2]),
        actual_score_receipt=C.binding(R.DOC/'SCORING_RECEIPT.json'),inputs=own['inputs'],
        original_full_statistics_function_changed=False,configured_saved_score_rows=1470,
        configured_moment_scalars=972,configured_metric_CIs=81,new_seed_draws=0,
        frozen_before_original_STATISTICS_STARTED_and_own_arithmetic=True))
    E.stats(args)


def main():
    p=R.parser();next(a for a in p._actions if a.dest=='stage').choices=('freeze','score','stats');args=p.parse_args()
    {'freeze':freeze,'score':score,'stats':stats}[args.stage](args)
if __name__=='__main__':main()
