"""Audit the rejected first fit and absence of every downstream evaluation.

Reads only recorded training receipts/traces, protocol/PREFIT metadata and
code. Does not apply the rejected weight to any data, access reference arrays,
resume optimization, or treat a missing source gate as a failed source gate.
"""
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

from . import common as C


HASH_KEYS = ('signed_target_sha', 'input_difference_sha', 'base_context_sha',
             'errors_sha', 'scaled_excess_sha', 'original_valid_sha')


def array_sha(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def guard(readable):
    writes = {C.DOC/'EVALUATION_NOT_RUN.json', C.DOC/'EVALUATION_NOT_RUN_KO.md'}
    reads = set()
    def hook(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if not path.is_relative_to(C.ROOT):
            return
        mode = args[1]
        flags = args[2] if len(args)>2 and isinstance(args[2], int) else 0
        writing = (isinstance(mode, str) and any(v in mode for v in 'wax+')) or flags & (os.O_WRONLY|os.O_RDWR|os.O_APPEND|os.O_CREAT|os.O_TRUNC)
        if writing:
            assert path in writes, ('ABSENCE_AUDIT_WRITE_SCOPE', str(path))
        else:
            assert path in readable or path in writes or path.suffix in ('.py', '.pyc'), ('ABSENCE_AUDIT_READ_SCOPE', str(path))
            reads.add(str(path.relative_to(C.ROOT)))
    sys.addaudithook(hook)
    return reads


def run():
    sys.dont_write_bytecode = True
    folder = C.RAW/'fits'/'R0_ONLY'
    inputs = {
        'protocol': C.DOC/'TRAIN_PROTOCOL.json',
        'protocol_sha': C.DOC/'TRAIN_PROTOCOL_SHA.json',
        'prefit': C.DOC/'PREFIT_REVIEW.json',
        'prefit_note': C.DOC/'PREFIT_REVIEW_KO.md',
        'rejection': C.DOC/'REJECTED_R0_ONLY.json',
        'START': folder/'START.json',
        'trace': folder/'TRACE.jsonl',
        'failure': folder/'FAILED.json',
    }
    reads = guard({p.resolve() for p in inputs.values()})
    assert not (C.DOC/'EVALUATION_NOT_RUN.json').exists()
    C.verify(C.read(inputs['protocol_sha']))
    protocol = C.read(inputs['protocol'])
    for binding in protocol['codes']:
        C.verify(binding)
    protocol_binding = C.bind(inputs['protocol'])
    prefit = C.read(inputs['prefit'])
    assert prefit['complete'] and prefit['PASS']
    assert protocol['inputs']['prefit_review'] == C.bind(inputs['prefit'])
    assert prefit['note'] == C.bind(inputs['prefit_note'])
    rejected = C.read(inputs['rejection'])
    start = C.read(inputs['START'])
    failed = C.read(inputs['failure'])
    assert rejected['complete'] and rejected['accepted'] is False
    assert rejected['model'] == start['model'] == failed['model'] == 'R0_ONLY'
    assert rejected['protocol'] == start['protocol'] == failed['protocol'] == protocol_binding
    assert rejected['START'] == C.bind(inputs['START']) and rejected['trace'] == C.bind(inputs['trace'])
    assert not rejected['automatic_retry'] and not failed['automatic_retry']
    assert failed['further_iterations_authorized'] is False
    assert failed['error_type'] == 'RuntimeError'
    assert failed['error'] == 'SOLVER_OR_CERTIFICATE_REJECTED; no extra iterations or new fit authorized.'
    cert, solver = rejected['certificate'], rejected['solver']
    assert cert['PASS'] is False and cert['optimizer_success'] is False
    assert solver['success'] is False and solver['status'] == 1
    assert solver['message'] == 'STOP: TOTAL NO. of ITERATIONS REACHED LIMIT'
    assert cert['iterations'] == solver['iterations'] == failed['iterations'] == protocol['solver']['maxiter'] == 1000
    assert cert['objective_calls'] == solver['objective_calls'] == failed['objective_calls'] == 1108
    assert cert['objective_calls'] < protocol['solver']['maxfun'] == 2000
    assert cert['lambda_l2'] == protocol['lambda_l2'] == 1e-4
    assert cert['max_gap_upper_bound'] == protocol['certificate']['gap_upper_bound_max'] == 1e-6
    gap = cert['gradient_l2']**2/(2*cert['lambda_l2'])
    np.testing.assert_allclose(gap, cert['gradient_l2_squared_over_2lambda'], rtol=1e-15, atol=0)
    assert gap > cert['max_gap_upper_bound']
    assert cert['loss_rule'] == protocol['loss_rule'] and cert['output_dim'] == 2 and cert['huber_delta'] == 1.
    np.testing.assert_allclose(cert['objective_value'], cert['Huber']+cert['L2_penalty'], rtol=1e-15, atol=0)
    for key in HASH_KEYS:
        assert rejected[key] == start[key] == failed[key] == prefit['models']['R0_ONLY'][key]
    rows = [json.loads(line) for line in inputs['trace'].read_text().splitlines()]
    assert {row['event'] for row in rows} == {'objective', 'iteration'}
    objective = [row for row in rows if row['event'] == 'objective']
    iterations = [row for row in rows if row['event'] == 'iteration']
    assert len(rows) == 2108 and len(objective) == 1108 and len(iterations) == 1000
    assert [row['call'] for row in objective] == list(range(1,1109))
    assert [row['iteration'] for row in iterations] == list(range(1,1001))
    assert iterations[-1]['objective_calls'] == 1108
    for row in rows:
        for key in HASH_KEYS:
            assert row[key] == rejected[key]
        assert row['basis_SHA_bind'] == rejected['basis_SHA_bind'] == protocol['inputs']['rbf_basis']
        assert row['loss_rule'] == protocol['loss_rule']
    last = objective[-1]
    assert last['objective'] == cert['objective_value'] and last['Huber'] == cert['Huber']
    assert last['L2_penalty'] == cert['L2_penalty'] and last['gradient_l2'] == cert['gradient_l2']
    assert last['weight_sha'] == iterations[-1]['weight_sha'] == rejected['final_weight_sha']
    weight = np.asarray(rejected['final_weight'], np.float64)
    assert weight.shape == (253,2) and np.isfinite(weight).all()
    assert array_sha(weight) == rejected['final_weight_sha']
    assert array_sha(np.zeros((253,2),np.float64)) == start['initial_weight_sha']
    # The rejected matrix is authenticated only; it is never scored on data.
    assert sorted(p.name for p in (C.RAW/'fits').iterdir()) == ['R0_ONLY']
    assert sorted(p.name for p in folder.iterdir()) == ['FAILED.json','START.json','TRACE.jsonl']
    assert sorted(str(p.relative_to(C.RAW)) for p in C.RAW.rglob('*') if p.is_file()) == [
        'fits/R0_ONLY/FAILED.json','fits/R0_ONLY/START.json','fits/R0_ONLY/TRACE.jsonl']
    absent = [C.DOC/f'FIT_{model}.json' for model in C.MODEL_NAMES]
    absent += [C.RAW/'fits'/model/'final.json' for model in C.MODEL_NAMES]
    absent += [C.RAW/'fits'/model for model in C.MODEL_NAMES if model!='R0_ONLY']
    absent += [C.DOC/name for name in (
        'TRAINING_COMPLETE.json','SOURCE_VAL_ROUTING_LOCK.json','SOURCE_VAL_GATE.json',
        'SOURCE_VAL_VERIFICATION.json','REAL_PROTOCOL.json','REAL_PROTOCOL_SHA.json',
        'REAL_ROUTING_LOCK.json','REAL_RESULTS.json','REAL_FRAME_RESULTS.csv',
        'REAL_REFERENCE_BINDINGS.json','REAL_DETAILED_COMPARISONS.json','REAL_VERIFICATION.json')]
    absent += [C.RAW/name for name in ('SOURCE_VAL_CHOICES.json','SOURCE_VAL_METRICS.npz','REAL_CHOICES.json','POSE_METRICS.json')]
    assert all(not p.exists() for p in absent)
    assert sorted(p.name for p in C.DOC.glob('REJECTED_*.json')) == ['REJECTED_R0_ONLY.json']
    # This smoke test uses invented arrays and does not load any source inputs.
    from . import verify_source as unused
    unused.selfcheck()
    code_paths = {name: C.HERE/name for name in ('audit_not_run.py','verify_source.py','evaluate_source.py','evaluate_real.py')}
    result = dict(complete=True,PASS=True,status='TRAIN_CONVERGENCE_FAILED',created_at=C.now(),
        scope='Recorded fit and artifact-absence audit; not an evaluation or performance certificate.',
        training_attempts=1,attempted_models=['R0_ONLY'],unattempted_models=['UNION_s1','UNION_s2','UNION_s3'],
        certified_models=0,accepted_checkpoints=0,objective_calls=1108,optimizer_iterations=1000,
        attempted_models_with_restart=0,automatic_retry=False,budget_extended=False,
        certificate=cert,solver_status=solver['status'],solver_message=solver['message'],
        source_gate_status='NOT_RUN_TRAIN_CONVERGENCE_FAILED',source_gate_checks_computed=0,
        source_VAL_routes=0,source_VAL_quality_rows=0,real_protocol_sealed=False,real_routes=0,real_quality_rows=0,
        new_image_forwards=0,new_PnP_solves=0,real_GT_reads=0,source_GT_reads=0,
        rejected_weight_authenticated_only=True,rejected_weight_used_for_prediction=False,
        trace_counts=dict(total_rows=2108,objective_rows=1108,iteration_rows=1000),
        evidence={key:C.bind(path) for key,path in inputs.items()},
        absent_artifacts=[str(p.relative_to(C.ROOT)) for p in absent],
        unused_source_verifier=dict(implementation=C.bind(code_paths['verify_source.py']),synthetic_selfcheck='PASS',
            actual_data_execution=False,scope='Code-only invented data; no source route or quality array opened.'),
        implementations={key:C.bind(path) for key,path in code_paths.items()},
        source_and_real_evaluation_authorized=False,
        absence_scope='Observed current immutable namespace files; not a claim about unrecorded external actions.',
        audit_read_paths=sorted(reads),new_fits_in_audit=0,method_success=False,goal_complete=False)
    note=f'''# 평가 미실행 독립 확인

**중단 사유는 `TRAIN_CONVERGENCE_FAILED`다. Source VAL gate는 실행되지 않았다.**

첫 R0_ONLY 학습1회가 objective 호출 **1,108회**, 반복 **1,000회**에서 사전 반복 한도에 도달했다. optimizer success는 false이며 gradient gap 상한은 **{gap:.17g}**로 고정 기준 **1e-6**을 넘었다. 목적함수 값은 {cert['objective_value']:.17g}이다. 이 값은 TRAIN Huber+ridge이며 T/R 평가 결과가 아니다.

REJECTED·START·TRACE·FAILED의 protocol 및6개 입력/타깃 해시를 대조했다. TRACE 전체2,108행은 objective1,108행과 iteration1,000행으로 정확히 나뉜다. 마지막 기록과 거부된253×2 weight의 SHA가 일치한다. 해당 weight는 저장 상태 확인에만 사용했으며 입력에 적용하거나 후보를 선택하지 않았다.

현재 실험 RAW에는 R0_ONLY의 START/TRACE/FAILED3파일만 있다. **인증된 모델0개, 채택 checkpoint0개, UNION 시도0회**이며 final.json·FIT 영수증·TRAINING_COMPLETE가 없다. Source VAL 선택·잠금·오류 배열·gate, 실사 protocol·선택·잠금·오류·결과 CSV도 모두 없다. 상세 부재 경로를 JSON에 기록했다. 재시작·예산 연장·추가 fit은 없다.

준비된 source verifier는 합성 데이터 selfcheck만 통과했다. 실제 source/실사 데이터로 실행하지 않았으며 성능 증거가 아니다. 본 receipt의 PASS는 기록과 미실행 상태의 검산 통과를 뜻한다. 수렴을 인증하지 못해 source45 및 실사 조건 판정 단계에 도달하지 않았고, 안정적인 실제 T/R 개선은 입증되지 않았다.

[확인 JSON](EVALUATION_NOT_RUN.json) · [원본 거부 기록](REJECTED_R0_ONLY.json) · [봉인 TRAIN 계약](TRAIN_PROTOCOL.json)
'''
    note_path=C.DOC/'EVALUATION_NOT_RUN_KO.md'
    if note_path.exists():
        assert note_path.read_text()==note, 'Existing partial note must match exactly.'
    else:
        C.save(note_path,note)
    result['note']=C.bind(C.DOC/'EVALUATION_NOT_RUN_KO.md')
    C.save(C.DOC/'EVALUATION_NOT_RUN.json',result)
    print('EVALUATION_NOT_RUN_AUDIT_PASS',C.bind(C.DOC/'EVALUATION_NOT_RUN.json'),flush=True)


if __name__=='__main__':
    run()
