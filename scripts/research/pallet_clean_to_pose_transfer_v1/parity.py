"""Full actual four-arm trace audit; never run a model or open evaluation GT."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from . import common as C

ARMS = ('CLEAN_RAW_CLEAR','CLEAN_REF_CLEAR','CLEAN_RAW_OCC','CLEAN_REF_OCC')
ROLES = ('REAL','SOURCE')


def identity(arm):
    _, target, condition = arm.split('_')
    assert target in ('RAW','REF') and condition in ('CLEAR','OCC')
    return target, condition


def analyze(traces, expected_batches=320, expected_real=2560, expected_source=2560):
    assert set(traces) == set(ARMS)
    assert all(len(rows) == expected_batches for rows in traces.values())
    checks = Counter()
    differences = Counter()
    totals = {arm:dict(roles={role:Counter() for role in ROLES}, recordings={},
        plans=Counter(), reasons=Counter(), epochs=Counter(), real_names=set(), source_names=set(),
        bbox_fractions=[], masked_point_histogram=Counter(), remaining_point_histogram=Counter()) for arm in ARMS}

    for index in range(expected_batches):
        batch = {arm:traces[arm][index] for arm in ARMS}
        reference = batch[ARMS[0]]
        for arm, row in batch.items():
            assert row['batch'] == index and row['epoch'] == reference['epoch'], (arm,index,'batch/epoch')
            assert len(row['names']) == len(row['transfer']) == len(row['before_images']) == len(row['after_images'])
            for key in ('names','before_images','boxes','support','batch_idx'):
                assert row[key] == reference[key], (index,arm,key)
            checks['all4_base_RGB_boxes_support_order_batchidx_equal'] += 1
        for condition in ('CLEAR','OCC'):
            a, b = batch[f'CLEAN_RAW_{condition}'], batch[f'CLEAN_REF_{condition}']
            assert a['images'] == b['images'] and a['after_images'] == b['after_images'], (index,condition,'RGB')
            checks['RAW_REF_same_condition_actual_RGB_equal'] += 1
            differences[condition+'_RAW_REF_coordinate_different_batches'] += a['coordinates'] != b['coordinates']
        for target in ('RAW','REF'):
            a, b = batch[f'CLEAN_{target}_CLEAR'], batch[f'CLEAN_{target}_OCC']
            assert a['coordinates'] == b['coordinates'], (index,target,'OCC changed target')
            checks['CLEAR_OCC_same_target_coordinates_equal'] += 1
            differences[target+'_CLEAR_OCC_RGB_changed_batches'] += a['images'] != b['images']

        for position, name in enumerate(reference['names']):
            infos = {arm:batch[arm]['transfer'][position] for arm in ARMS}
            first = infos[ARMS[0]]
            for arm, info in infos.items():
                target, condition = identity(arm)
                assert info['name'] == name and info['target'] == target and info['condition'] == condition
                assert info['role'] == ('SOURCE' if name.startswith('syn__') else 'REAL')
                for key in ('plan','scheduled','recording','before_image','geometric_demotions'):
                    assert info[key] == first[key], (index,position,arm,key)
                assert info['before_image'] == batch[arm]['before_images'][position]
                assert info['after_image'] == batch[arm]['after_images'][position]
                if info['role'] == 'SOURCE':
                    assert not info['applied'] and not info['scheduled'] and info['plan'] is None
                    assert info['actual_covered'] == 0 and info['geometric_demotions'] == 0
                    assert info['before_image'] == info['after_image'], 'Source RGB was modified'
                else:
                    for key in ('actual_supervised','actual_supervised_all9','original_supervised','original_ignored'):
                        assert info[key] == first[key], (index,position,arm,key)
                    assert info['actual_supervised'] == info['actual_covered'] + info['actual_remaining']
                    if condition == 'CLEAR':
                        assert not info['applied'] and info['actual_covered'] == 0
                    else:
                        assert info['applied'] == bool(info['plan'] and info['plan']['applied'])
                        assert info['actual_covered'] == (info['planned_target_covered'] if info['applied'] else 0)
                    if not info['applied']:
                        assert info['before_image'] == info['after_image'], 'Unapplied input changed'
                    else:
                        assert info['plan']['scheduled'] and info['plan']['rectangle'] is not None
                        # The canonical REF plan covers>=1/leaves>=2; RAW has
                        # different coordinates and may cover a different count.
                        assert len(info['plan']['covered']) >= 1 and info['plan']['remaining'] >= 2
                checks['all4_shared_plan_and_role_invariants'] += 1
            assert infos['CLEAN_RAW_CLEAR']['after_image'] == infos['CLEAN_REF_CLEAR']['after_image']
            assert infos['CLEAN_RAW_OCC']['after_image'] == infos['CLEAN_REF_OCC']['after_image']

        for arm, row in batch.items():
            total = totals[arm]
            total['epochs'][str(row['epoch'])] += 1
            for role in ROLES:
                total['roles'][role].update(row['roles'][role])
            recording_check = {}
            for info in row['transfer']:
                if info['role'] == 'SOURCE':
                    total['source_names'].add(info['name'])
                    continue
                total['real_names'].add(info['name'])
                group = total['recordings'].setdefault(info['recording'], Counter())
                per_batch = recording_check.setdefault(info['recording'], Counter())
                values = dict(images=1, scheduled=int(info['scheduled']), applied=int(info['applied']),
                    supervised=info['actual_supervised'], covered=info['actual_covered'],
                    remaining=info['actual_remaining'], geometric_demotions=info['geometric_demotions'])
                group.update(values); per_batch.update(values)
                group.update(supervised_all9=info['actual_supervised_all9'],
                    original_supervised=info['original_supervised'], original_ignored=info['original_ignored'])
                plan = info['plan']
                if plan:
                    total['plans']['scheduled'] += int(plan['scheduled'])
                    total['plans']['feasible'] += int(plan['applied'])
                    total['reasons'][plan['reason']] += 1
                    group['scheduled_failed_placement'] += int(plan['scheduled'] and not plan['applied'])
                else:
                    total['reasons']['no_transformed_instance'] += 1
                    group['no_transformed_instance'] += 1
                total['plans']['actually_applied'] += int(info['applied'])
                total['plans']['actual_RGB_changed'] += info['before_image'] != info['after_image']
                total['plans']['masked_supervised_corners'] += info['actual_covered']
                total['plans']['remaining_supervised_corners'] += info['actual_remaining']
                total['plans']['planned_target_covered'] += info['planned_target_covered']
                total['plans']['geometric_demotions_all9'] += info['geometric_demotions']
                total['masked_point_histogram'][str(info['actual_covered'])] += 1
                total['remaining_point_histogram'][str(info['actual_remaining'])] += 1
                if info['applied']:
                    total['bbox_fractions'].append(float(plan['bbox_fraction']))
            assert {k:dict(v) for k,v in recording_check.items()} == row['recording_exposure'], (arm,index,'recording trace')

    exposures = {}
    for arm, total in totals.items():
        assert total['roles']['REAL']['images'] == expected_real
        assert total['roles']['SOURCE']['images'] == expected_source
        if expected_batches == 320:
            assert dict(total['epochs']) == {str(epoch):64 for epoch in range(5)}
            assert len(total['real_names']) == 78 and len(total['source_names']) == 512
        assert sum(r['images'] for r in total['recordings'].values()) == expected_real
        assert sum(r['supervised_all9'] for r in total['recordings'].values()) == total['roles']['REAL']['supervised']
        if arm.endswith('_CLEAR'):
            assert total['plans']['actually_applied'] == total['plans']['actual_RGB_changed'] == 0
        else:
            assert total['plans']['actually_applied'] > 0 and total['plans']['masked_supervised_corners'] > 0
        areas = sorted(total['bbox_fractions'])
        exposures[arm] = dict(roles={role:dict(values) for role,values in total['roles'].items()},
            recordings={key:dict(value) for key,value in sorted(total['recordings'].items())},
            real_unique=len(total['real_names']), source_unique=len(total['source_names']),
            per_epoch_batches=dict(total['epochs']), plans=dict(total['plans']), placement_reasons=dict(total['reasons']),
            scheduled_failed_placements=sum(row['scheduled_failed_placement'] for row in total['recordings'].values()),
            masked_point_histogram=dict(total['masked_point_histogram']), remaining_point_histogram=dict(total['remaining_point_histogram']),
            actual_applied_bbox_fraction=dict(n=len(areas),minimum=min(areas) if areas else None,
                mean=sum(areas)/len(areas) if areas else None,maximum=max(areas) if areas else None))
    for key in ('CLEAR_RAW_REF_coordinate_different_batches','OCC_RAW_REF_coordinate_different_batches'):
        assert differences[key] > 0, 'Coordinate intervention absent'
    return dict(passed=True, batches_per_arm=expected_batches, checks=dict(checks), differences=dict(differences),
        arms=exposures, original_images_and_labels_unchanged=True,
        source_RGB_unmasked_verified=True, RAW_REF_same_condition_RGB_exact=True,
        CLEAR_OCC_same_target_coordinates_exact=True, all4_common_augmented_support_exact=True,
        all4_order_boxes_before_RGB_plan_exact=True,
        coordinate_attribution_scope='Full-batch coordinate hashes verify CLEAR/OCC equality. SOURCE-only coordinate values are not individually hashed in this trace; unchanged source export and source-transform CPU tests provide separate evidence.',
        support_definition='Supervised v2; ignored v1; source invisible v0. Recording supervised/covered/remaining counts use8corners; supervised_all9 and role totals include center.',
        asymmetric_coverage_note='Canonical REF rectangle is shared. RAW/REF covered-point counts can differ because target coordinates differ; this is not a different input mask.',
        new_fits=0, optimizer_updates=0)


def run(seed=42):
    path = C.DOC/f'PAIR_INTEGRITY_S{seed}.json'
    if path.exists():
        result = C.read(path)
        for binding in result['sources'] + result['trace_bindings'] + result['fit_bindings']:
            C.verify(binding)
        assert result['passed']
        print('PAIR_INTEGRITY_ALREADY_VERIFIED',seed,flush=True)
        return result
    protocol_path = C.DOC/'PRIMARY_PROTOCOL.json'
    protocol = C.read(protocol_path)
    assert seed in protocol['seeds'] and set(protocol['arms']) == set(ARMS)
    for binding in protocol['inputs'] + protocol['sources'] + [protocol['initialization']]:
        C.verify(binding)
    traces, fits, trace_bindings, fit_bindings = {}, {}, [], []
    for arm in ARMS:
        fit_path = C.DOC/f'FIT_{arm}_S{seed}.json'
        fit = C.read(fit_path)
        assert fit['complete'] and fit['optimizer_steps'] == 320
        assert fit['exact_R0_initialization'] and fit['protected_state_exact']
        assert fit['initialization'] == protocol['initialization']
        assert fit['protocol'] == C.bind(protocol_path)
        for key in ('checkpoint','trace','results_csv','initialization'):
            C.verify(fit[key])
        assert fit['trace']['path'] == str((C.RAW/f'TRACE_{arm}_S{seed}.json').relative_to(C.ROOT))
        traces[arm] = C.read(C.ROOT/fit['trace']['path']); fits[arm] = fit
        trace_bindings.append(fit['trace']); fit_bindings.append(C.bind(fit_path))
    result = analyze(traces)
    result.update(seed=seed, created_utc=C.now(), trace_bindings=trace_bindings, fit_bindings=fit_bindings,
        sources=[C.bind(protocol_path), C.bind(C.DOC/'DATASET_PAIR_PREFLIGHT.json'), C.bind(Path(__file__))],
        checkpoints={arm:fit['checkpoint'] for arm,fit in fits.items()}, initialization=protocol['initialization'],
        same_initialization=True, same_updates=True, protected_state_exact=True,
        fits_audited=4, updates_audited=1280)
    C.save(path,result,True)
    print('PAIR_INTEGRITY_PASSED',seed,{arm:value['plans'] for arm,value in result['arms'].items()},flush=True)
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--seed',type=int,default=42)
    run(parser.parse_args().seed)
