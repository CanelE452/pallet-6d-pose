"""기존 학습 엔진을 변경하지 않는 짝지은 반복 실행 어댑터.

이 파일의 존재는 새 학습 승인이 아니다. 무학습 대조 뒤 기록된 결정이 있어야
freeze가 가능하다. 모든 새 캐시/출력은 새 namespace에, 비용은 누적 원장에 쓴다.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import copy
from pathlib import Path
import re

from . import common as C


class Context:
    def __init__(self, stage='CLEAR_S43'):
        assert re.fullmatch(r'(CLEAR|OCC)_S43', stage), stage
        self.stage = stage
        self.DOC = C.DOC / 'training' / stage
        self.RAW = C.RAW / 'training' / stage
        self.OUT = C.OUT / 'training' / stage
        self.ROOT = C.ROOT

    def __getattr__(self, name):
        return getattr(C, name)

    def read(self, path):
        path = Path(path)
        if path == self.DOC / 'RESOURCE_LEDGER.json':
            path = C.DOC / 'RESOURCE_LEDGER.json'
        return C.read(path)

    def save(self, path, value, freeze=False):
        path = Path(path)
        assert path != self.DOC / 'RESOURCE_LEDGER.json', '하위 단계에서 비용 0 원장을 만들지 않는다'
        assert any(path.resolve().is_relative_to(root.resolve()) for root in (self.DOC, self.RAW, self.OUT))
        return C.save(path, value, freeze)

    def resource(self, event, *args, **kwargs):
        return C.resource(self.stage + '::' + event, *args, **kwargs)


@contextmanager
def scoped_context(context, modules):
    """단독 CLI 프로세스에서만 사용하며 실패해도 원래 모듈 전역을 복원한다."""
    previous = [(module, module.C) for module in modules]
    try:
        for module, _ in previous:
            module.C = context
        yield
    finally:
        for module, original in previous:
            module.C = original


def arms(condition):
    assert condition in ('CLEAR', 'OCC')
    return tuple(f'CLEAN_{target}_{condition}' for target in ('RAW', 'REF'))


def validate_decision(decision, stage):
    assert decision['approved'] is True and decision['locked_before_fit'] is True
    assert decision['kind'] == 'SAME_RECIPE_COMPLETION'
    assert decision['seed'] == 43 and stage == f"{decision['condition']}_S43"
    assert decision['condition'] in ('CLEAR', 'OCC') and decision['why'].strip()
    assert decision['evidence'], '무학습 대조의 고정 결과를 먼저 연결해야 한다'
    assert not any(key in decision for key in ('args', 'lr', 'updates', 'masking', 'datasets', 'initialization'))
    return arms(decision['condition'])


def verify_protocol(protocol):
    for binding in protocol['inputs'] + protocol['sources'] + [protocol['initialization'], protocol['resource_snapshot']]:
        C.verify(binding)


def freeze(stage, decision_path):
    """결정 뒤에만 실행한다. 기존 데이터 바이트는 읽고 새 alias/list만 만든다."""
    import yaml
    from scripts.research.pallet_clean_to_pose_transfer_v1 import prepare as P
    context = Context(stage)
    destination = context.DOC / 'PRIMARY_PROTOCOL.json'
    if destination.exists():
        protocol = context.read(destination)
        verify_protocol(protocol)
        assert protocol['decision'] == C.bind(decision_path)
        print('PAIR_PROTOCOL_ALREADY_LOCKED', stage, flush=True)
        return protocol
    decision_path = Path(decision_path).resolve()
    assert decision_path.is_relative_to(C.DOC)
    decision = C.read(decision_path)
    pair = validate_decision(decision, stage)
    for binding in decision['evidence']:
        C.verify(binding)
    # 이미 있는 OCC43 또는 다른 완료 fit을 새 반복처럼 다시 실행하지 않는다.
    existing = [p for arm in pair for p in C.OLD.DOC.rglob(f'FIT_{arm}_S43.json')]
    assert not existing, ('기존 완료 fit을 재사용해야 함', existing)
    ledger = C.read(C.DOC / 'RESOURCE_LEDGER.json')
    C.verify(ledger['inherited_ledger'])
    inherited = C.read(C.ROOT / ledger['inherited_ledger']['path'])
    assert ledger['inherited_totals'] == inherited['totals']
    assert ledger['caps'] == dict(student_fits=10, selector_fits=1, GPU_training_seconds=21600)
    assert ledger['totals']['student_fits'] + 2 <= ledger['caps']['student_fits']
    assert ledger['totals']['GPU_training_seconds'] < ledger['caps']['GPU_training_seconds']
    snapshot = context.DOC / 'RESOURCE_LEDGER_PREFIT.json'
    context.save(snapshot, ledger, True)
    base_path = C.OLD.DOC / 'PRIMARY_PROTOCOL.json'
    original = C.read(base_path)
    for binding in original['inputs'] + original['sources'] + [original['initialization']]:
        C.verify(binding)
    assert original['train_unique_images'] == 78 and original['updates_per_fit'] == 320
    assert original['source_slots_per_epoch'] == original['real_slots_per_epoch'] == 512
    assert original['args']['lr0'] == 1e-5 and original['masking']['schedule'] == .5
    protocol = copy.deepcopy(original)
    protocol.update(created_utc=C.now(), arms={name: original['arms'][name] for name in pair},
        seeds=[43], primary_fits=2, followup_kind='SAME_RECIPE_COMPLETION',
        decision=C.bind(decision_path), resource_snapshot=C.bind(snapshot),
        cumulative_resource_ledger_path=str((C.DOC/'RESOURCE_LEDGER.json').relative_to(C.ROOT)),
        locked_before_fit=True, base_seed=42, condition=decision['condition'],
        deviation='동일 clean78 recipe의 누락된 실제 seed43 대조 완성. 새 좌표/새 이미지/새 학습 설정 없음.',
        supervision_note='교사 9장/38점. GEO 간접 감독 포함 추적 가능 합집합 19장/86점은 기존 provenance 그대로 유지.')
    new_bindings = []
    with scoped_context(context, (P,)):
        for target in ('RAW', 'REF'):
            source = original['datasets'][target]
            directory = context.RAW / 'dataset' / target
            lists = {}
            for role in ('train', 'val'):
                C.verify(source[role + '_list'])
                oldlist = C.ROOT / source[role + '_list']['path']
                aliases = []
                for text in oldlist.read_text().splitlines():
                    image = Path(text)
                    alias = directory / 'images' / image.name
                    P.local_link(image, alias)
                    P.local_link(P.label_path(image), directory / 'labels' / image.with_suffix('.txt').name)
                    aliases.append(str(alias))
                path = directory / f'{role}.txt'
                context.save(path, '\n'.join(aliases) + '\n', True)
                lists[role] = path
                new_bindings.append(C.bind(path))
            data = yaml.safe_load((C.ROOT/source['data']['path']).read_text())
            data.update(path=str(directory), train=str(lists['train']), val=str(lists['val']))
            path = directory / 'data.yaml'
            context.save(path, yaml.safe_dump(data, sort_keys=False), True)
            protocol['datasets'][target] = dict(source, data=C.bind(path),
                train_list=C.bind(lists['train']), val_list=C.bind(lists['val']))
            new_bindings.append(C.bind(path))
    for name in ('PRIMARY_INPUT_BINDINGS_PRIVATE.json', 'CLEAN_LOCKED_PRIVATE.json'):
        context.save(context.RAW/name, C.read(C.OLD.RAW/name), True)
        new_bindings.append(C.bind(context.RAW/name))
    protocol['inputs'] = original['inputs'] + new_bindings
    protocol['sources'] = original['sources'] + [C.bind(path) for path in (
        base_path, decision_path, Path(__file__), Path(C.__file__),
        Path(P.__file__).with_name('followup_pair.py'),
        C.OLD.DOC/'CODE_LOCK.json', C.OLD.DOC/'PREFLIGHT.json',
        C.OLD.DOC/'SELECTOR_SUPERVISION_PROVENANCE.json')]
    for binding in C.read(C.OLD.DOC/'CODE_LOCK.json')['files']:
        C.verify(binding)
        protocol['sources'].append(binding)
    context.save(destination, protocol, True)
    print('PAIR_PROTOCOL_LOCKED', stage, list(pair), flush=True)
    return protocol


def cross_condition_checks(clear, occ, full=False):
    """같은 seed CLEAR/OCC의 바뀌지 않아야 하는 실제 입력 계약을 비교한다."""
    assert len(clear) == len(occ) == (320 if full else 8)
    changed = source = 0
    for a, b in zip(clear, occ):
        for key in ('names', 'before_images', 'boxes', 'support', 'coordinates', 'batch_idx', 'roles'):
            assert a[key] == b[key], ('CLEAR/OCC', key)
        for x, y in zip(a['transfer'], b['transfer']):
            for key in ('name', 'role', 'recording', 'plan', 'before_image', 'geometric_demotions'):
                assert x[key] == y[key], ('CLEAR/OCC occurrence', key)
            assert not x['applied'] and x['before_image'] == x['after_image']
            if x['role'] == 'SOURCE':
                source += 1
                assert not y['applied'] and x['after_image'] == y['after_image']
            elif y['applied']:
                changed += 1
                assert x['after_image'] != y['after_image']
            else:
                assert x['after_image'] == y['after_image']
    assert changed > 0
    return dict(passed=True, batches=len(clear), applied_real_occurrences=changed,
        source_exact_occurrences=source, base_RGB_coordinates_support_plan_exact=True)


def seed_difference(first, second):
    """단순 이름 재정렬뿐 아니라 동일 이미지의 기본 증강과 계획 RNG도 확인한다."""
    assert len(first) == len(second)
    counts = {key: sum(a[key] != b[key] for a, b in zip(first, second))
              for key in ('names', 'before_images', 'images')}
    counts['plan_batches'] = sum([v['plan'] for v in a['transfer']] != [v['plan'] for v in b['transfer']]
                               for a, b in zip(first, second))
    grouped = []
    for trace in (first, second):
        by_name = {}
        for batch in trace:
            for item in batch['transfer']:
                if item['role'] == 'REAL':
                    by_name.setdefault(item['name'], item)
        grouped.append(by_name)
    common = set(grouped[0]) & set(grouped[1])
    assert common
    counts['matched_real_identities'] = len(common)
    counts['matched_identity_base_RGB_changed'] = sum(grouped[0][n]['before_image'] != grouped[1][n]['before_image'] for n in common)
    counts['matched_identity_plan_seed_changed'] = sum(grouped[0][n]['plan']['seed'] != grouped[1][n]['plan']['seed'] for n in common)
    assert all(value > 0 for value in counts.values()), '실제 다른 입력 흐름이어야 한다'
    return counts


def preflight(stage):
    import cv2
    import torch
    from scripts.research.pallet_clean_to_pose_transfer_v1 import preflight as P, trainer as T
    from scripts.research.pallet_clean_to_pose_transfer_v1.augmentation import load_paired_labels
    from scripts.research.pallet_clean_to_pose_transfer_v1.followup_pair import pair_trace_checks
    context = Context(stage)
    path = context.DOC/'PREFLIGHT.json'
    protocol = context.read(context.DOC/'PRIMARY_PROTOCOL.json')
    verify_protocol(protocol)
    if path.exists():
        result = context.read(path)
        assert result['passed']
        for binding in result['bindings']:
            C.verify(binding)
        return result
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    parent = C.read(C.OLD.DOC/'PREFLIGHT.json')
    assert parent['passed']
    for binding in parent['bindings']:
        C.verify(binding)
    paired = load_paired_labels(*[C.ROOT/protocol['datasets'][target]['train_list']['path'] for target in ('RAW', 'REF')])
    assert len(paired) == 78
    recordings = {row['train_id']+'.png': row['recording'] for row in context.read(context.RAW/'CLEAN_LOCKED_PRIVATE.json')['rows']}
    condition = protocol['condition']
    streams, bindings, differences, cross = {}, [], {}, {}
    with scoped_context(context, (P, T)):
        for target in ('RAW', 'REF'):
            trace, binding, _ = P.collect(protocol, paired, recordings, 43, target, condition)
            streams[target] = trace
            prior = C.OLD.RAW/f'PREFLIGHT_TRACE_S43_{target}_{condition}_PRIVATE.json'
            assert trace == C.read(prior), '같은 recipe의 과거 CPU 입력과 달라짐'
            other = 'OCC' if condition == 'CLEAR' else 'CLEAR'
            other_path = C.OLD.RAW/f'PREFLIGHT_TRACE_S43_{target}_{other}_PRIVATE.json'
            other_trace = C.read(other_path)
            cross[target] = cross_condition_checks(trace, other_trace) if condition == 'CLEAR' else cross_condition_checks(other_trace, trace)
            previous_path = C.OLD.RAW/f'PREFLIGHT_TRACE_S42_{target}_{condition}_PRIVATE.json'
            differences[target] = seed_difference(C.read(previous_path), trace)
            bindings += [binding, C.bind(prior), C.bind(other_path), C.bind(previous_path)]
    result = dict(passed=True, created_at=C.now(), pair=pair_trace_checks(streams['RAW'], streams['REF']),
        same_seed_clear_occ=cross, actual_seed42_vs43=differences,
        matches_historical_seed43_preflight=True, K_batches=8, workers=2,
        true_ignore_test_reused=C.bind(C.OLD.DOC/'PREFLIGHT.json'),
        bindings=bindings+[C.bind(context.DOC/'PRIMARY_PROTOCOL.json'), C.bind(Path(__file__))],
        optimizer_updates=0, fits=0, GPU_seconds=0, evaluation_reference_read=False,
        scope='실제 worker2 입력 K8 검사이며 향후 전체320 parity를 대체하지 않는다')
    context.save(path, result, True)
    print('PAIR_PREFLIGHT_PASS', stage, differences, flush=True)
    return result


def train(stage, arm):
    from scripts.research.pallet_clean_to_pose_transfer_v1 import train as T, trainer as R
    context = Context(stage)
    protocol = context.read(context.DOC/'PRIMARY_PROTOCOL.json')
    verify_protocol(protocol)
    assert arm in protocol['arms']
    pre = context.read(context.DOC/'PREFLIGHT.json')
    assert pre['passed']
    for binding in pre['bindings']:
        C.verify(binding)
    # train.py를 복제하거나 optimizer/schedule을 바꾸지 않고 저장 위치만 바꾼다.
    with scoped_context(context, (T, R)):
        T.train(arm, 43)


def parity(stage):
    import torch
    from scripts.research.pallet_clean_to_pose_transfer_v1.followup_pair import pair_trace_checks
    from scripts.research.pallet_type_selftrain_v1.recovery_pose_trainer import pose_parameter
    context = Context(stage)
    protocol = context.read(context.DOC/'PRIMARY_PROTOCOL.json')
    verify_protocol(protocol)
    destination = context.DOC/'PAIR_INTEGRITY_S43.json'
    pair = arms(protocol['condition'])
    fits = {arm: context.read(context.DOC/f'FIT_{arm}_S43.json') for arm in pair}
    traces = {}
    base_model = torch.load(C.ROOT/protocol['initialization']['path'], map_location='cpu', weights_only=False)['model'].float()
    base = base_model.state_dict()
    allowed = {name for name, _ in base_model.named_parameters() if pose_parameter(name)}
    protected = set(base)-allowed
    assert (len(base), len(allowed), len(protected)) == (879, 132, 747)
    bindings, states, differences, cross = [], {}, {}, {}
    for arm, fit in fits.items():
        assert fit['complete'] and fit['optimizer_steps'] == 320 and fit['seed'] == 43
        assert fit['initialization'] == protocol['initialization']
        assert fit['protected_state_exact'] and fit['exact_R0_initialization']
        for key in ('checkpoint', 'trace', 'protocol', 'code_lock'):
            C.verify(fit[key])
            bindings.append(fit[key])
        final = torch.load(C.ROOT/fit['checkpoint']['path'], map_location='cpu', weights_only=False)['model'].float().state_dict()
        assert set(final) == set(base)
        assert all(torch.equal(final[name], base[name]) for name in protected)
        changed = {name for name in base if not torch.equal(final[name], base[name])}
        assert changed and changed <= allowed and changed == set(fit['changed_tensors'])
        trace = C.read(C.ROOT/fit['trace']['path'])
        assert [row['batch'] for row in trace] == list(range(320))
        assert Counter(row['epoch'] for row in trace) == {e: 64 for e in range(5)}
        target = fit['target']
        pre_path = context.RAW/f"PREFLIGHT_TRACE_S43_{target}_{protocol['condition']}_PRIVATE.json"
        for actual, expected in zip(trace[:8], context.read(pre_path)):
            for key in ('names', 'before_images', 'after_images', 'boxes', 'support', 'coordinates', 'batch_idx', 'roles', 'transfer'):
                assert actual[key] == expected[key], ('실제 학습/K8', key)
        previous = C.OLD.RAW/f'TRACE_{arm}_S42.json'
        differences[arm] = seed_difference(C.read(previous), trace)
        if protocol['condition'] == 'CLEAR':
            other = C.OLD.RAW/'followups/REPEAT_PRIMARY_S43'/f'TRACE_CLEAN_{target}_OCC_S43.json'
            cross[arm] = cross_condition_checks(trace, C.read(other), full=True)
            bindings.append(C.bind(other))
        prior_fit = C.read(C.OLD.DOC/f'FIT_{arm}_S42.json')
        C.verify(prior_fit['checkpoint'])
        prior_state = torch.load(C.ROOT/prior_fit['checkpoint']['path'], map_location='cpu', weights_only=False)['model'].float().state_dict()
        different = sum(not torch.equal(final[name], prior_state[name]) for name in final)
        assert different > 0
        states[arm] = dict(runtime_initialization_asserted=879, checkpoint_states=879,
            protected_exact=747, allowed_trainable=132, changed=len(changed),
            final_state_different_from_seed42=different, actual_first8_matches_preflight=True,
            initial_snapshot_limitation='초기 tensor 동일성은 런타임 assertion 및 R0 해시 근거; 저장되지 않은 step0를 사후 복원하지 않음')
        traces[arm] = trace
        bindings += [C.bind(previous), C.bind(pre_path), C.bind(context.DOC/f'FIT_{arm}_S43.json'), prior_fit['checkpoint']]
    result = pair_trace_checks(*[traces[a] for a in pair], full=True)
    result.update(states=states, actual_seed42_vs43=differences, same_seed_clear_occ=cross,
        initialization=protocol['initialization'], updates_per_arm=320,
        sources=bindings+[C.bind(context.DOC/'PRIMARY_PROTOCOL.json'), C.bind(Path(__file__))],
        new_audit_fits=0, GPU_seconds=0)
    context.save(destination, result, True)
    print('PAIR_FULL_PARITY_PASS', stage, flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['freeze', 'preflight', 'train', 'parity'])
    parser.add_argument('--stage', default='CLEAR_S43')
    parser.add_argument('--decision', type=Path)
    parser.add_argument('--arm')
    args = parser.parse_args()
    if args.phase == 'freeze':
        assert args.decision is not None
        freeze(args.stage, args.decision)
    elif args.phase == 'train':
        train(args.stage, args.arm)
    else:
        globals()[args.phase](args.stage)
