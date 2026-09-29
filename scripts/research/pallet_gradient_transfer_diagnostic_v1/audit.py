"""CPU-only integrity audit; never constructs a model or optimizer."""
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

from . import common as C
from .report import aggregate, point_scope, stable_sign


def run():
    protocol = C.read(C.DOC/'PROTOCOL.json')
    raw = C.read(C.RAW/'RESULTS_PRIVATE.json')
    public = C.read(C.DOC/'RESULTS.json')
    records = C.read(C.RAW/'INPUT_SELECTION_PRIVATE.json')
    execution = C.read(C.RAW/'EXECUTION_LOCK_PRIVATE.json')
    bindings = (protocol['inputs'] + raw['bindings'] + execution['bindings'] +
                [public[k] for k in ('protocol', 'measurement', 'report_code')] +
                public['figures'] + public['prior_context']['bindings'] +
                list(raw['checkpoint_paths'].values()) +
                [r[k] for r in records for k in ('image', 'label')])
    for binding in bindings:
        C.verify(binding)
    checks = dict(input_source_checkpoint_hashes=True,
        no_optimizer_or_fit=(not raw['optimizer_constructed'] and
                             raw['optimizer_steps'] == raw['new_fits'] == raw['checkpoint_writes'] == 0),
        no_evaluation_reference=raw['evaluation_reference_read'] is False,
        all_checkpoint_tensors_restored=all(v['exact'] and v['tensors'] == 879 and
            v['all_parameter_grad_buffers_none'] for v in raw['state_checks'].values()),
        train_selection_count=len(records) == 32 and all(sum(r['role'] == role for r in records) == 16
            for role in ('REAL','SOURCE')),
        train_selection_unique=all(len({r['image']['sha256'] for r in records if r['role'] == role}) == 16
            for role in ('REAL','SOURCE')),
        finite_difference_precision=execution['precision'] == 'FP32; autocast off; matmul TF32 off; cuDNN TF32 off')
    checks['all_saved_signs_recomputed'] = all(
        p['signs'][d] == (stable_sign(v, protocol['numerical_floor_px']) if p['residual_px'] > 1e-8 else 'NO_RESIDUAL')
        for r in raw['records'] for p in r['points']
        for d,v in p['response_px_at_common_reference_step'].items())
    checks['public_pooled_recomputed'] = all(public['pooled'][arm][role][scope] == aggregate(
        point_scope([p for r in raw['records'] if r['arm'] == arm for p in r['points']], role, scope), role.lower())
        for arm in protocol['checkpoints'] for role in ('REAL','SOURCE')
        for scope in ('ALL_9','CORNERS_0_7','CENTER_8'))
    checks['mixed_gradient_reconstruction'] = max(r['max_gradient_partition_error'] for r in raw['records']) < 3e-6
    checks['coordinate_additivity_below_fixed_floor'] = max(r['max_coordinate_additivity_residual_px']
        for r in raw['records']) < protocol['numerical_floor_px']
    checks['same_real_points_across_checkpoints'] = len({tuple(sorted((p['image_id'],p['corner'])
        for r in raw['records'] if r['arm'] == arm for p in r['points'] if p['role'] == 'REAL'))
        for arm in protocol['checkpoints']}) == 1
    report = (C.DOC/'REPORT_KO.md').read_text()
    images = re.findall(r'!\[[^\]]*\]\(([^)]+)\)', report)
    checks['report_figures_exist'] = len(images) >= 2 and all((C.DOC/p).is_file() for p in images)
    public_text = '\n'.join(p.read_text() for p in C.DOC.glob('*') if p.suffix in ('.json','.md'))
    checks['no_private_frame_ids_or_coordinate_arrays'] = all(p['image_id'] not in public_text
        for r in raw['records'] for p in r['points']) and 'response_px_at_common_reference_step' not in public_text
    changed = subprocess.check_output(['git','diff',protocol['head_start'],'--name-only'],cwd=C.ROOT,text=True).splitlines()
    allowed = ('scripts/research/pallet_gradient_transfer_diagnostic_v1/',
               '_docs/experiments/pallet_gradient_transfer_diagnostic_v1/')
    checks['no_existing_paper_or_other_tracked_edits'] = all(p.startswith(allowed) for p in changed)
    completed = subprocess.run([sys.executable,'-m','pytest','-q',str(Path(__file__).parent)],
                               cwd=C.ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    C.save(C.DOC/'TEST_OUTPUT.txt', completed.stdout)
    checks['unit_tests'] = completed.returncode == 0
    result = dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,
        binding_checks=len(bindings),unique_bound_files=len({b['path'] for b in bindings}),
        saved_sign_checks=sum(len(p['signs']) for r in raw['records'] for p in r['points']),
        test_output=C.bind(C.DOC/'TEST_OUTPUT.txt'),created_at=C.now(),
        code=[C.bind(p) for p in sorted(Path(__file__).parent.glob('*.py'))],
        caveat='Runtime assertions check assignment/restore equality; saved hash and aggregates independently rechecked. Not evidence of GT or 6D improvement.')
    C.save(C.DOC/'AUDIT.json',result)
    C.save(C.DOC/'AUDIT_KO.md','# 무학습 진단 검수\n\n'+
        f"상태: {result['status']}. SHA 검사 {len(bindings)}회, 저장된 부호 {result['saved_sign_checks']}개 재계산.\n\n"+
        '\n'.join(f'- {key}: {"PASS" if val else "FAIL"}' for key,val in checks.items())+
        '\n\n'+completed.stdout+'\n\n물리 GT 정확도·6D 개선 검증이 아닌 고정 TRAIN 타깃 방향 진단이다.\n')
    corners = public['pooled']['REF_LR5']['REAL']['CORNERS_0_7']
    C.save(C.DOC/'FINAL_DECISION.json', dict(status='NO_FIT_DIAGNOSTIC_COMPLETED',
        local_source_opposition='OBSERVED_IN_BOTH_BATCHES',
        dominant_replay_blocker='NOT_ESTABLISHED',
        current_student_real_corners=corners['points'],
        toward_to_away=corners['cancellation'],
        away_to_toward=corners['sign_transition_counts']['AWAY_TO_TOWARD'],
        real_component_toward=corners['sign_counts']['real']['TOWARD'],
        combined_toward=corners['sign_counts']['combined']['TOWARD'],
        replay_removal_or_downweighting='NOT_APPLIED_NOT_JUSTIFIED_AS_A_FIX',
        new_fits=0, optimizer_steps=0, model_changes_saved=0,
        unchanged=['published main comparison','manuscript','existing checkpoints','target labels'],
        unsupported=['Replay is the dominant transfer failure cause',
                     'Equal sign counts imply equal net error changes',
                     'Changing replay weight improves held-out 2D or 6D',
                     'This is the actual AdamW training trajectory'],
        next_action='Do not automatically change replay or start training. Any later intervention requires a separately fixed matched RAW/REF comparison.',
        report=C.bind(C.DOC/'REPORT_KO.md'),audit=C.bind(C.DOC/'AUDIT.json')))
    print(json.dumps(dict(status=result['status'],checks=checks),ensure_ascii=False))
    assert result['status'] == 'PASS'


if __name__ == '__main__':
    run()
