"""Review only the wrong anchor literal in the preserved failed runtime audit.

Own freeze precedes string inspection. No production imports, inference,
timing moments, pose/projection math, optimizer, PnP, GT, J or SVD execution.
The old receipt remains FAILED. Join its other checks with 300 corrected
source-literal predicates, without rerunning those other checks.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import time
import traceback

REPO = Path(__file__).resolve().parents[3]
DOC = REPO/'_docs/experiments/pallet_sparse_local_line_20261010_v9'
PRIVATE = Path('/tmp/pallet-sparse-local-line-private-20261010-v9')
ARMS = ('BASE','N3_SUBPIX','ROLE_BOUNDARY_H_ROBUST','ROLE_BOUNDARY_LOCAL_POINT_LINE')
LEARNED = ARMS[2:]
WRONG = 'original Base predictions; unchanged training feature distribution'
ACTUAL = 'unchanged original Base predictions'
OLD_SHA = '420070cf0764fb24c5aff1c61f2dbec1a9782253d2e8fb5cd91d4b21ba41fbe3'
OLD_PROTOCOL_SHA = '8b78178b9f5ca8f152f65003a0996b3016ce14f3353c6fe2cced4750f73e103a'
OUTPUTS = ('RUNTIME_ANCHOR_REVIEW_PROTOCOL.json','RUNTIME_ANCHOR_REVIEW_STARTED.json',
           'RUNTIME_ANCHOR_REVIEW_CHECKS.json','JOINED_RUNTIME_CONTRACT_CHECKS.json')
LIMITS = dict(targeted_source_literal_review_only=True, original_failed_receipt_preserved=True,
    original_nonanchor_checks_not_rerun=111804, original_moment_slots_not_recomputed=196,
    new_production_import_model_image_GT_timing_pose_optimizer_PnP_J_SVD_calls=0,
    runtime_and_accuracy_policy_changes=0, original_checks_passed_remains=False,
    historical_hardware_independent_authentication=False, automatic_retry=False)


def require(value, reason):
    if not value:
        raise ValueError(reason)


def read(path):
    with Path(path).open(encoding='utf-8') as f:
        return json.load(f)


def binding(path):
    p=Path(path)
    require(p.is_file() and not any(x.is_symlink() for x in (p,*p.parents)), 'missing/symlink input '+str(p))
    digest=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):
            digest.update(b)
    absolute=p.resolve()
    return dict(path=str(absolute.relative_to(REPO)) if absolute.is_relative_to(REPO) else p.name,
        sha256=digest.hexdigest(),bytes=p.stat().st_size)


def same(a,b):
    return isinstance(b,dict) and all(a[k]==b.get(k) for k in ('sha256','bytes'))


def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush()


def guard(args,stage):
    require(args.input.is_dir() and args.output.is_dir(),'existing input/output directories required')
    for p in (args.input,args.output):
        require(not any(x.is_symlink() for x in (p,*p.parents)),'symlink ancestry')
    out=args.output.resolve()
    require(out==DOC.resolve() or out.is_relative_to(PRIVATE.resolve()),'new V9 DOC/private output only')
    for name in OUTPUTS if stage=='freeze' else OUTPUTS[1:]:
        require(not(out/name).exists(),'preserve first/existing '+name)
    return args.input.resolve(),out


def paths(args):
    folder=args.input.resolve()
    result={name:folder/name for name in ('RUNTIME_VALIDATION_PROTOCOL.json','RUNTIME_VALIDATION_STARTED.json',
        'RUNTIME_VALIDATION_CHECKS.json','RUNTIME_REVIEW_PROTOCOL.json','RUNTIME_REVIEW.json',
        'RUNTIME_REVIEW_ROWS.jsonl.gz')}
    result.update(checker=Path(__file__),original_checker=Path(__file__).with_name('runtime_scalar_check.py'),
        producer=REPO/'scripts/research/pallet_three_head_observation_20261010_v7/pipeline.py')
    return result


def source_predicates(inputs):
    """AST evidence only: never import or evaluate either production module."""
    old=ast.parse(inputs['original_checker'].read_text(encoding='utf-8'))
    prod=ast.parse(inputs['producer'].read_text(encoding='utf-8'))
    final=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name=='final_outcome')
    run=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    pipeline=next(n for n in prod.body if isinstance(n,ast.ClassDef) and n.name=='Pipeline')
    outcomes=next(n for n in pipeline.body if isinstance(n,ast.FunctionDef) and n.name=='outcomes')
    key=ast.dump(ast.parse("output['model_query_anchor']",mode='eval').body,include_attributes=False)
    matches=[n for n in ast.walk(final) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and
        isinstance(n.func.value,ast.Name) and n.func.value.id=='audit' and n.func.attr=='eq' and
        len(n.args)==3 and ast.dump(n.args[0],include_attributes=False)==key]
    require(len(matches)==1,'original anchor predicate must be unique')
    call=matches[0]
    require(isinstance(call.args[1],ast.Constant) and call.args[1].value==WRONG,'original wrong literal changed')
    require(isinstance(call.args[2],ast.BinOp) and isinstance(call.args[2].op,ast.Add) and
        isinstance(call.args[2].left,ast.Name) and call.args[2].left.id=='label' and
        isinstance(call.args[2].right,ast.Constant) and call.args[2].right.value=='.Base_query_anchor',
        'original failed predicate label differs')
    fixed=ast.dump(ast.parse("row['arm'] in ('BASE','N3_SUBPIX')",mode='eval').body,include_attributes=False)
    branches=[n for n in final.body if isinstance(n,ast.If) and ast.dump(n.test,include_attributes=False)==fixed]
    require(len(branches)==1 and any(n is call for stmt in branches[0].orelse for n in ast.walk(stmt)) and
        not any(n is call for stmt in branches[0].body for n in ast.walk(stmt)),
        'original failed predicate must occur once only in learned-route else branch')
    dispatch=[n for n in ast.walk(run) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='final_outcome']
    require(len(dispatch)==1,'original stream must have one final outcome dispatch')
    constants=[n.value for n in ast.walk(outcomes) if isinstance(n,ast.keyword) and n.arg=='model_query_anchor']
    require(len(constants)==1 and isinstance(constants[0],ast.Constant) and constants[0].value==ACTUAL,
        'unchanged producer actual anchor literal differs')
    def ast_hash(node):
        return hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
    return dict(original_wrong_literal=call.args[1].value,production_literal=constants[0].value,
        original_predicate_AST_sha256=ast_hash(call),producer_outcomes_AST_sha256=ast_hash(outcomes),
        original_final_outcome_AST_sha256=ast_hash(final),original_run_AST_sha256=ast_hash(run),
        unique_predicate_in_learned_else=True,one_original_per_row_outcome_dispatch=True)


def original_contract(inputs,bindings):
    old=read(inputs['RUNTIME_VALIDATION_CHECKS.json']); own=read(inputs['RUNTIME_VALIDATION_PROTOCOL.json'])
    runtime=read(inputs['RUNTIME_REVIEW.json']); started=read(inputs['RUNTIME_VALIDATION_STARTED.json'])
    require(old['schema']=='supplemental_sparse_local_line_runtime_scalar_checks_v9' and
        old['complete'] is True and old['passed'] is False and old['exception'] is None and
        old['checks']==112104 and old['failure_count']==300 and len(old['failures'])==100 and
        old['counts']['runtime_rows']==600 and old['counts']['independently_recomputed_moment_slots']==196,
        'preserved original failed audit accounting differs')
    require(bindings['original_checker']['sha256']==OLD_SHA and
        bindings['RUNTIME_VALIDATION_PROTOCOL.json']['sha256']==OLD_PROTOCOL_SHA,
        'original frozen code/protocol changed')
    require(same(old['checker'],bindings['original_checker']) and same(own['checker'],bindings['original_checker']) and
        same(old['own_protocol'],bindings['RUNTIME_VALIDATION_PROTOCOL.json']) and old['inputs']==own['inputs'],
        'old audit own binding differs')
    require(same(started['own_protocol'],bindings['RUNTIME_VALIDATION_PROTOCOL.json']) and
        same(started['checker'],bindings['original_checker']),'old STARTED binding differs')
    for alias,name in (('runtime','RUNTIME_REVIEW.json'),('runtime_rows','RUNTIME_REVIEW_ROWS.jsonl.gz'),
                       ('runtime_protocol','RUNTIME_REVIEW_PROTOCOL.json')):
        require(same(old['inputs'][alias],bindings[name]),'old arithmetic input changed: '+alias)
    require(runtime['complete'] is True and runtime['status']=='DONE' and runtime['statistics_official'] is True and
        same(runtime['raw_rows'],bindings['RUNTIME_REVIEW_ROWS.jsonl.gz']),'completed fresh runtime changed')
    rp=read(inputs['RUNTIME_REVIEW_PROTOCOL.json'])
    require(same(rp['inputs']['fresh_v7:pipeline.py'],bindings['producer']),'producer bytes differ from runtime freeze')
    return old


def freeze(args):
    _,out=guard(args,'freeze'); inputs=paths(args)
    bindings={k:binding(p) for k,p in inputs.items()}
    original_contract(inputs,bindings); extracted=source_predicates(inputs)
    write_new(out/OUTPUTS[0],dict(schema='supplemental_runtime_anchor_literal_review_protocol_v9',
        checker=bindings['checker'],inputs=bindings,source_predicates=extracted,limits=LIMITS,
        expected=dict(runtime_rows=600,learned_anchor_predicates=300,predicates_per_learned_route=150,
            original_checks=112104,original_failures=300,original_accepted_nonanchor_checks=111804),
        own_freeze_before_own_stream_string_checks=True,no_automatic_retry=True))
    print('V9_RUNTIME_ANCHOR_REVIEW_FROZEN',binding(out/OUTPUTS[0])['sha256'],flush=True)


def run(args):
    _,out=guard(args,'run'); inputs=paths(args); bindings={k:binding(p) for k,p in inputs.items()}
    own=read(out/OUTPUTS[0]); own_binding=binding(out/OUTPUTS[0])
    require(own['schema']=='supplemental_runtime_anchor_literal_review_protocol_v9' and
        own['checker']==bindings['checker'] and own['inputs']==bindings and own['limits']==LIMITS and
        own['source_predicates']==source_predicates(inputs),'own review frozen bytes/AST differ')
    write_new(out/OUTPUTS[1],dict(status='STARTED',own_protocol=own_binding,checker=bindings['checker'],limits=LIMITS))
    began=time.monotonic(); counts=Counter(); labels=[]; failure=None; finished=False
    try:
        old=original_contract(inputs,bindings)
        with gzip.open(inputs['RUNTIME_REVIEW_ROWS.jsonl.gz'],'rt',encoding='utf-8') as f:
            for index,line in enumerate(f):
                require(bool(line.strip()),'blank raw runtime row')
                row=json.loads(line); require(row['arm'] in ARMS,'unknown original runtime route')
                counts['runtime_rows']+=1; counts['route:'+row['arm']]+=1
                if row['arm'] in LEARNED:
                    actual=row['full_fresh_output']['model_query_anchor']
                    require(actual==own['source_predicates']['production_literal'],'actual anchor differs from frozen production source')
                    require(actual!=own['source_predicates']['original_wrong_literal'],'old wrong predicate no longer fails')
                    counts['corrected_production_anchor_predicates']+=1
                    counts['reconstructed_original_failed_predicates']+=1
                    labels.append(dict(check='row.%d.outcome.Base_query_anchor'%index,actual=actual,expected=WRONG))
                del row
        require(counts['runtime_rows']==600 and all(counts['route:'+arm]==150 for arm in ARMS),'full600 stream/4route scope differs')
        require(counts['corrected_production_anchor_predicates']==counts['reconstructed_original_failed_predicates']==300,
            'exact300 targeted old failure calls not reconstructed')
        require(labels[:100]==old['failures'],'preserved original truncated100 failure prefix differs')
        require(len(labels)==old['failure_count'],'all original failures not accounted by targeted predicate')
        require({k:binding(p) for k,p in inputs.items()}==bindings,'review inputs changed during string checks')
        finished=True
    except BaseException as error:
        failure=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
    receipt=dict(schema='supplemental_runtime_anchor_literal_review_checks_v9',complete=finished,passed=finished,
        own_protocol=own_binding,checker=bindings['checker'],inputs=bindings,limits=LIMITS,
        counts=dict(counts),exception=failure,source_predicates=own['source_predicates'],
        original_failed_checks_receipt=bindings['RUNTIME_VALIDATION_CHECKS.json'],
        reconstructed_original_anchor_failures=labels,original_nonanchor_checks_rerun=0,
        original196moment_slots_recomputed=0,elapsed_seconds=time.monotonic()-began)
    write_new(out/OUTPUTS[2],receipt)
    if finished:
        write_new(out/OUTPUTS[3],dict(schema='sparse_local_line_joined_runtime_contract_checks_v9',complete=True,passed=True,
            original_checks_passed=False,original_failed_attempt_preserved_unchanged=True,
            original_attempt=bindings['RUNTIME_VALIDATION_CHECKS.json'],original_protocol=bindings['RUNTIME_VALIDATION_PROTOCOL.json'],
            original_checker=bindings['original_checker'],review_protocol=own_binding,
            review_attempt=binding(out/OUTPUTS[2]),review_checker=bindings['checker'],
            runtime=bindings['RUNTIME_REVIEW.json'],runtime_rows=bindings['RUNTIME_REVIEW_ROWS.jsonl.gz'],
            runtime_protocol=bindings['RUNTIME_REVIEW_PROTOCOL.json'],
            original_checks=112104,original_failure_count=300,original_accepted_unchanged_nonanchor_checks=111804,
            corrected_source_anchor_predicates=300,accepted_contract_predicates=112104,
            original_nonanchor_checks_rerun=0,original196moment_slots_recomputed=0,
            all300_original_failures_accounted_by_AST_bound_wrong_literal=True,
            production_literal=ACTUAL,original_wrong_literal=WRONG,limits=LIMITS))
    print('V9_RUNTIME_ANCHOR_REVIEW_CHECKED',finished,counts['corrected_production_anchor_predicates'],flush=True)
    require(finished,'targeted anchor review failed; preserve first receipt')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('freeze','run'))
    p.add_argument('--input',type=Path,default=DOC);p.add_argument('--output',type=Path,default=DOC)
    args=p.parse_args();{'freeze':freeze,'run':run}[args.stage](args)


if __name__=='__main__':
    main()
