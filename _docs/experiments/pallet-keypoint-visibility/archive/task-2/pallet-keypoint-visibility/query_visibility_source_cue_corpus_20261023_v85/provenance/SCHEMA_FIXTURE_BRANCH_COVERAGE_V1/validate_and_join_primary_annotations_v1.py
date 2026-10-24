"""Validate immutable public cue annotations; no RGB/model/current data reads.

Validation asserts correspondence and unknown semantics, not visual truth.
Primary AI observations remain weak source evidence, not native8 training GT.
"""
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import hashlib
import json
import math
import copy

ROOT = Path(__file__).resolve().parent
SLOTS = ('UPPER_WALL_LEFT', 'UPPER_WALL_RIGHT', 'LOWEST_WIDE_WALL_LEFT',
         'LOWEST_WIDE_WALL_RIGHT', 'BASE_RUNNER_LEFT', 'BASE_RUNNER_RIGHT')
VISIBLE = 'APPARENT_VISIBLE_STRUCTURAL_CUE_AI'
BEFORE_STATES = {VISIBLE, 'UNKNOWN_U', 'OUT_OF_FRAME_U', 'SELF_OR_UNRESOLVED_U'}
AFTER_STATES = {'APPARENT_SAME_CUE_VISIBLE_AI',
                'EXTERNAL_FOREGROUND_BLOCKER_COVERS_CUE_REGION_AI',
                'PARTIAL_OR_UNCLEAR_U', 'REGISTRATION_OR_TARGET_MOVEMENT_U',
                'MISSING_BEFORE_REFERENCE_U'}


def binding(path):
    f = Path(path)
    return {'path': str(f), 'sha256': hashlib.sha256(f.read_bytes()).hexdigest(), 'bytes': f.stat().st_size}


def read(path):
    return json.loads(Path(path).read_text())


def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def bound_read(item):
    f = Path(item['path'])
    assert f.is_file() and binding(f)['sha256'] == item['sha256'], str(f)
    return read(f)


def validate_pair(before, after, source):
    assert before['pair_id'] == after['pair_id'] == source['pair_id'], 'pair identity'
    assert before['source_split'] == after['source_split'] == source['split'], 'split identity'
    assert before['before_only_frozen'] is True, 'before not frozen'
    assert tuple(s['part_slot'] for s in before['slots']) == SLOTS, 'before slot order'
    assert tuple(s['part_slot'] for s in after['slots']) == SLOTS, 'after slot order'
    for side, ann in [('before', before), ('after', after)]:
        image = ann['source_' + side + '_RGB']
        expected = source[side + '_RGB']
        assert image['path'] == expected['path'] and image['sha256'] == expected['sha256'], 'RGB lineage'
        H, W = image['raw_hw']
        assert type(H) is int and type(W) is int and H > 0 and W > 0, 'native HW'
    assert before['source_before_RGB']['raw_hw'] == after['source_after_RGB']['raw_hw'], 'paired HW'
    H, W = before['source_before_RGB']['raw_hw']
    joined = []
    for left, right in zip(before['slots'], after['slots']):
        assert left['before_state'] in BEFORE_STATES, 'before state'
        assert right['after_state'] in AFTER_STATES, 'after state'
        assert left.get('native8_semantic_id') is None and right.get('native8_semantic_id') is None, 'fabricated native8'
        for field in ['reference_xy', 'uncertainty_bbox_xyxy_half_open']:
            assert canonical(left[field]) == canonical(right[field]), 'changed ' + field
        xy, box = left['reference_xy'], left['uncertainty_bbox_xyxy_half_open']
        if left['before_state'] != VISIBLE:
            assert xy is None and box is None, 'unknown before invented reference'
            assert right['after_state'] == 'MISSING_BEFORE_REFERENCE_U', 'unknown before converted into label'
        else:
            assert isinstance(xy, list) and len(xy) == 2 and all(isinstance(v, (float, int)) and math.isfinite(v) for v in xy), 'finite xy'
            assert 0 <= xy[0] < W and 0 <= xy[1] < H, 'visible reference out of frame'
            assert isinstance(box, list) and len(box) == 4 and all(isinstance(v, (float, int)) and math.isfinite(v) for v in box), 'finite box'
            assert box[0] <= xy[0] < box[2] and box[1] <= xy[1] < box[3], 'cue outside uncertainty box'
            assert isinstance(left.get('cue_description'), str) and left['cue_description'].strip(), 'no named cue'
            assert isinstance(left.get('evidence'), str) and left['evidence'].strip(), 'no before evidence'
            assert right['after_state'] != 'MISSING_BEFORE_REFERENCE_U', 'visible reference disappeared from schema'
        assert isinstance(right.get('evidence'), str) and right['evidence'].strip(), 'no after evidence'
        if right['after_state'] == 'EXTERNAL_FOREGROUND_BLOCKER_COVERS_CUE_REGION_AI':
            assert left['before_state'] == VISIBLE, 'cover without before witness'
            assert right['coverage_entire_region'] is True, 'partial cover called E'
            assert isinstance(right.get('foreign_blocker'), str) and right['foreign_blocker'].strip(), 'missing foreign blocker'
        joined.append({'pair_id': source['pair_id'], 'source_split': source['split'],
                       'source_family': 'ONE_PALLOX_PHYSICAL_BIN', 'session': source['session'],
                       'part_slot': left['part_slot'], 'reference_xy': xy,
                       'uncertainty_bbox_xyxy_half_open': box,
                       'before_state': left['before_state'], 'after_state': right['after_state'],
                       'before_cue_evidence': left, 'after_cue_evidence': right,
                       'human_gold': False, 'native8_semantic_id': None,
                       'training_or_forward_admission': False})
    return joined


def selftest():
    source = {'pair_id': 'TEST:0000_0001', 'split': 'TRAIN_LAB', 'session': 'TEST',
              'before_RGB': {'path': '/synthetic_test/before.png', 'sha256': 'a'},
              'after_RGB': {'path': '/synthetic_test/after.png', 'sha256': 'b'}}
    before = {'pair_id': source['pair_id'], 'source_split': source['split'],
              'before_only_frozen': True, 'source_before_RGB': {**source['before_RGB'], 'raw_hw': [20, 30]},
              'slots': [{'part_slot': s, 'reference_xy': [10., 10.],
                         'uncertainty_bbox_xyxy_half_open': [8., 8., 12., 12.],
                         'before_state': VISIBLE, 'cue_description': 'test contour',
                         'evidence': 'fixture only', 'native8_semantic_id': None} for s in SLOTS]}
    after = {'pair_id': source['pair_id'], 'source_split': source['split'],
             'source_after_RGB': {**source['after_RGB'], 'raw_hw': [20, 30]},
             'slots': [{k: v for k, v in s.items() if k in ['part_slot', 'reference_xy', 'uncertainty_bbox_xyxy_half_open', 'native8_semantic_id']} |
                       {'after_state': 'EXTERNAL_FOREGROUND_BLOCKER_COVERS_CUE_REGION_AI',
                        'evidence': 'fixture only', 'foreign_blocker': 'test foreign box',
                        'coverage_entire_region': True} for s in before['slots']]}
    assert len(validate_pair(before, after, source)) == 6
    mutations = {
        'changed_after_coordinate': lambda x, y: y['slots'][0].update(reference_xy=[11., 10.]),
        'unknown_before_becomes_external': lambda x, y: x['slots'][0].update(before_state='UNKNOWN_U', reference_xy=None, uncertainty_bbox_xyxy_half_open=None),
        'partial_cover_as_external': lambda x, y: y['slots'][0].update(coverage_entire_region=False),
        'invented_native8': lambda x, y: y['slots'][0].update(native8_semantic_id=0),
        'wrong_split': lambda x, y: y.update(source_split='VAL_HALL'),
        'slot_reorder': lambda x, y: y['slots'].reverse(),
    }
    results = []
    for name, mutate in mutations.items():
        x, y = copy.deepcopy(before), copy.deepcopy(after)
        mutate(x, y)
        try:
            validate_pair(x, y, source)
        except AssertionError as e:
            results.append({'case': name, 'rejected': True, 'reason': str(e)})
        else:
            raise AssertionError('Invalid fixture accepted: ' + name)
    return {'PASS': True, 'valid_fixture_accepted': True, 'invalid_cases': results,
            'scope': 'Tests immutable identity, unknown, foreign-full-cover and no native fabrication. Does not certify AI visual truth or train a model.'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest-only', action='store_true')
    ap.add_argument('--batch-a', type=Path)
    ap.add_argument('--batch-b', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(exist_ok=False, parents=True)
    report = selftest()
    (args.out / 'SCHEMA_INVARIANT_FIXTURE_TESTS.json').write_text(json.dumps(report, indent=2) + '\n')
    if args.selftest_only:
        print(json.dumps(report))
        return
    assert args.batch_a and args.batch_b
    cohort = read(ROOT / 'ALL50_PAIR_COHORT_LOCK_BEFORE_NEW_RGB.json')
    sources = {x['pair_id']: x for x in cohort['rows']}
    collected, bindings = [], []
    for worker, manifest in [('a', args.batch_a), ('b', args.batch_b)]:
        batch = read(manifest)
        assert batch['completed'] is True and batch['annotator'] == worker
        assigned = read(ROOT / f'ANNOTATOR_{worker.upper()}_BATCH01_BEFORE_NEW_RGB.json')
        assert [r['pair_id'] for r in batch['pairs']] == [r['pair_id'] for r in assigned['rows']], 'case skipped/replaced/reordered'
        bindings += [binding(manifest), binding(ROOT / f'ANNOTATOR_{worker.upper()}_BATCH01_BEFORE_NEW_RGB.json')]
        for unit in batch['pairs']:
            before = bound_read(unit['before_annotation'])
            after = bound_read(unit['after_annotation'])
            bound_read(unit['before_seal']); bound_read(unit['after_seal'])
            assert after['before_annotation']['sha256'] == unit['before_annotation']['sha256']
            assert after['before_seal']['sha256'] == unit['before_seal']['sha256']
            collected.extend(validate_pair(before, after, sources[unit['pair_id']]))
            bindings.extend(unit[k] for k in ['before_annotation', 'before_seal', 'after_annotation', 'after_seal'])
    assert len(collected) == 60 and len({r['pair_id'] for r in collected}) == 10
    by_split = defaultdict(Counter)
    for r in collected:
        by_split[r['source_split']][r['after_state']] += 1
    data = args.out / 'PRIMARY_CUE_OBSERVATIONS_GT_ONLY.jsonl'
    with data.open('x') as f:
        for r in collected:
            f.write(json.dumps(r, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')
    summary = {'PASS_metadata_invariants': True, 'pairs': 10, 'part_records': len(collected),
               'by_source_split_after_state': dict(by_split),
               'source_family_count': 1, 'physical_bin_instances': 1,
               'native8_gold': 0, 'human_gold': 0,
               'training_or_current_admission': False, 'new_forward_fit_current_PnP': 0,
               'visual_truth_not_certified_by_metadata_validator': True,
               'observations': binding(data), 'inputs': bindings,
               'code': binding(__file__), 'fixture_tests': report}
    (args.out / 'ANNOTATION_VALIDATION_AND_COUNTS.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'PASS': True, 'pairs': 10, 'slots': 60, 'counts': dict(by_split)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
