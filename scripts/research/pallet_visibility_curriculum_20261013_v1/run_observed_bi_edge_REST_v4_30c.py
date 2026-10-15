"""CLI-safe operational retry of the zero-row V30b stdin launch.

The original prepared inputs and frozen R30._worker are unchanged. A private
AST copy changes only execution/progress/failure metadata names and serializes
pending identifiers instead of Future keys. Original source/attempt/CAL files
are preserved; the extra caller ancestry is bound through final all238 seals.
Invoke this module with ``python -m ...``, never a stdin main with spawn.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime,timezone
import inspect
from pathlib import Path
from types import FunctionType,SimpleNamespace

from . import rerank_observed_bi_edge_REST_v4_30b as W

C,R,OLD=W.C,W.R,W.OLD
OUTPUT=W.OUTPUT
PLAN=R.D.B/'observed_bi_association_REST_resume_plan_v4_30c/PREDECLARATION.json'
PARENT_SOURCE_SHA256='de1f69893a959030d9f6168295b3fb6abe791c83b3c5048fde564924dc1d36d2'
OLD_PLAN_SHA256='d8933269f08df314dfd918ce5be73a0be7519f3f886ea69cd6138e0254f06996'
OWNER_FAILURE='REST_LAUNCH_FAILURE_OWNER_RECEIPT_V30B_V1.json'
OWNER_FAILURE_SHA256='e9ff0aa48e21ad2c9459bfa07eb3b6df497e704d9edd5e6c67be72ff2b9532dd'
PROTOCOL='REST_RESUME_PROTOCOL_V30C.json'
PREP='REST_RESUME_PREPARATION_SEAL_V30C.json'
EXEC='REST_EXECUTION_PROTOCOL_V30C.json'
FAILURE='REST_FAILURE_RECEIPT_V30C.json'
LIVE='REST_LIVE_PROGRESS_V30C.json'


def declare():
    if PLAN.exists():raise FileExistsError('Preserve retry declaration')
    if C.sha(W.__file__)!=PARENT_SOURCE_SHA256:raise ValueError('Exact original caller source required')
    files=[Path(__file__),Path(__file__).with_name('test_run_observed_bi_edge_REST_v4_30c.py'),
           Path(W.__file__),Path(R.__file__)]
    PLAN.parent.mkdir(parents=True)
    C.write(PLAN,dict(schema='V30C_ZERO_ROW_OPERATIONAL_RETRY_PREDECLARATION',
        UTC=datetime.now(timezone.utc).isoformat(),
        source_sha256={str(p.resolve().relative_to(C.ROOT)):C.sha(p) for p in files},
        original_plan_sha256=OLD_PLAN_SHA256,owner_failure_sha256=OWNER_FAILURE_SHA256,
        original_attempt_preserved=True,expected_existing_REST_rows=0,
        same_worker='rerank_observed_bi_edge_population_v4_30._worker',
        numeric_hypothesis_factor_guard_or_selection_changes=0,
        operations=['CLI module main supports spawn','new EXEC/progress/failure names',
                    'pending_ids=list(pending.values()) on failure','additive source/input closure'],
        new_RGB_NMS_TLS_NN_labels_posefit=0,maximum_workers=3,maximum_pending=3))


def prepare(*,root_metadata_GO=False,output=OUTPUT):
    if root_metadata_GO is not True:raise PermissionError('Root retry metadata GO before IO')
    output=Path(output)
    if any((output/n).exists() for n in (PROTOCOL,PREP,EXEC)):
        raise FileExistsError('Preserve every retry attempt')
    plan=C.read(PLAN)
    if C.sha(W.__file__)!=PARENT_SOURCE_SHA256 or C.sha(W.PLAN)!=OLD_PLAN_SHA256:
        raise ValueError('Original caller and plan immutable')
    if C.sha(output/OWNER_FAILURE)!=OWNER_FAILURE_SHA256:
        raise ValueError('Exact zero-row launch-failure receipt required')
    old=W._load(output);failure=C.read(output/OWNER_FAILURE)
    if failure['new_REST_rerank_rows']!=0 or failure['original_CAL_immutable_failures']:
        raise ValueError('Only the verified zero-row attempt is reusable')
    if any(OLD._row_path(output,i).exists() for i in old['REST_ids']):
        raise ValueError('Do not overwrite any actual REST spool')
    paths=[W.PLAN,PLAN,output/OWNER_FAILURE,output/W.REST_PROTOCOL,output/W.REST_PREP,output/W.REST_EXEC]
    sources=W._union(old['source_sha256'],plan['source_sha256'])
    inputs=W._union(old['input_sha256'],{str(p.relative_to(C.ROOT)):C.sha(p) for p in paths})
    W._verify([sources,inputs,old['protected_user_sha256'],old['original_CAL_file_sha256']])
    protocol=dict(schema='V30C_CLI_SAFE_ZERO_ROW_RETRY_PROTOCOL',
        source_sha256=sources,input_sha256=inputs,protected_user_sha256=old['protected_user_sha256'],
        original_CAL_file_sha256=old['original_CAL_file_sha256'],REST_ids=old['REST_ids'],
        original_failed_execution=W._binding(output/W.REST_EXEC),owner_failure=W._binding(output/OWNER_FAILURE),
        original_preparation=W._binding(output/W.REST_PREP),plan=W._binding(PLAN),
        same_worker_source_sha256=C.sha(R.__file__),original_REST_source_sha256=PARENT_SOURCE_SHA256,
        maximum_workers=3,maximum_pending=3,new_RGB_NMS_TLS_NN_labels_posefit=0,
        hypothesis_or_mathematical_changes=0)
    C.write(output/PROTOCOL,protocol)
    C.write(output/PREP,dict(protocol_sha256=C.sha(output/PROTOCOL),rows=192,
        ordered_ids=old['REST_ids'],source_sha256=plan['source_sha256'],before_actual_retry=True))
    return W._binding(output/PREP)


def _bound_load(output):
    output=Path(output);old=W._load(output);protocol=C.read(output/PROTOCOL);prep=C.read(output/PREP)
    if C.sha(output/PROTOCOL)!=prep['protocol_sha256'] or prep['ordered_ids']!=old['REST_ids']:
        raise ValueError('Retry protocol/preparation changed')
    for key in ('source_sha256','input_sha256','protected_user_sha256','original_CAL_file_sha256'):
        W._verify([protocol[key]])
    extra={str((output/n).relative_to(C.ROOT)):C.sha(output/n) for n in (PROTOCOL,PREP)}
    if(output/EXEC).exists():extra[str((output/EXEC).relative_to(C.ROOT))]=C.sha(output/EXEC)
    return {**old,'source_sha256':W._union(old['source_sha256'],protocol['source_sha256']),
        'input_sha256':W._union(old['input_sha256'],protocol['input_sha256'],extra)}


def _closure_write(path,value):
    path=Path(path)
    if path.name in (EXEC,'REST_OBSERVATION_SEAL.json','FINAL_PROTOCOL.json',
                     'OBSERVATION_SEAL.json','FINAL_COMPLETION_RECEIPT_V30B.json'):
        output=path.parent
        refs=dict(resume_protocol=W._binding(output/PROTOCOL),resume_preparation=W._binding(output/PREP),
            original_failed_execution=W._binding(output/W.REST_EXEC),
            original_launch_failure=W._binding(output/OWNER_FAILURE),caller_source_sha256=C.sha(__file__),
            original_REST_worker_source_sha256=C.sha(R.__file__),mathematical_changes=0)
        if path.name!=EXEC:refs['resume_execution']=W._binding(output/EXEC)
        value={**value,'operational_resumption':refs}
    C.write(path,value)


def _resumed_rerank_function():
    if C.sha(W.__file__)!=PARENT_SOURCE_SHA256:
        raise ValueError('Frozen original rerank code required')
    tree=ast.parse(inspect.getsource(W.rerank));counts=dict(pending=0,failure=0,live=0)
    class MetadataOnly(ast.NodeTransformer):
        def visit_keyword(self,node):
            self.generic_visit(node)
            if(node.arg=='pending_ids' and isinstance(node.value,ast.Call)
                    and isinstance(node.value.func,ast.Name) and node.value.func.id=='list'
                    and len(node.value.args)==1 and isinstance(node.value.args[0],ast.Name)
                    and node.value.args[0].id=='pending'):
                node.value.args[0]=ast.Call(func=ast.Attribute(value=ast.Name(id='pending',ctx=ast.Load()),
                    attr='values',ctx=ast.Load()),args=[],keywords=[]);counts['pending']+=1
            return node
        def visit_Constant(self,node):
            if node.value=='REST_FAILURE_RECEIPT_V30B.json':counts['failure']+=1;return ast.copy_location(ast.Constant(FAILURE),node)
            if node.value=='REST_LIVE_PROGRESS_V30B.json':counts['live']+=1;return ast.copy_location(ast.Constant(LIVE),node)
            return node
    tree=MetadataOnly().visit(tree)
    if counts!=dict(pending=1,failure=1,live=1):raise ValueError('Exactly three metadata-only AST edits required')
    ast.fix_missing_locations(tree)
    namespace={**W.rerank.__globals__,'_load':_bound_load,'REST_EXEC':EXEC,
        'C':SimpleNamespace(**{**vars(C),'write':_closure_write})}
    exec(compile(tree,str(Path(__file__).resolve()),'exec'),namespace)
    return namespace['rerank']


def rerank(*,root_REST_rerank_GO=False,workers=3,output=OUTPUT):
    if root_REST_rerank_GO is not True:raise PermissionError('Root actual retry GO before IO')
    return _resumed_rerank_function()(root_REST_rerank_GO=True,workers=workers,output=output)


def finalize(*,root_finalize_GO=False,output=OUTPUT):
    if root_finalize_GO is not True:raise PermissionError('Root retry finalize GO before IO')
    namespace={**W.finalize.__globals__,'_load':_bound_load,'REST_EXEC':EXEC,
        'C':SimpleNamespace(**{**vars(C),'write':_closure_write})}
    function=FunctionType(W.finalize.__code__,namespace,W.finalize.__name__,W.finalize.__defaults__)
    function.__kwdefaults__=W.finalize.__kwdefaults__
    return function(root_finalize_GO=True,output=output)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('declare','prepare','rerank','finalize'))
    p.add_argument('--root-metadata-GO',action='store_true');p.add_argument('--root-REST-rerank-GO',action='store_true')
    p.add_argument('--root-finalize-GO',action='store_true');p.add_argument('--workers',type=int,default=3)
    a=p.parse_args()
    if a.stage=='declare':declare()
    elif a.stage=='prepare':print(prepare(root_metadata_GO=a.root_metadata_GO))
    elif a.stage=='rerank':print(rerank(root_REST_rerank_GO=a.root_REST_rerank_GO,workers=a.workers))
    else:print(finalize(root_finalize_GO=a.root_finalize_GO))
