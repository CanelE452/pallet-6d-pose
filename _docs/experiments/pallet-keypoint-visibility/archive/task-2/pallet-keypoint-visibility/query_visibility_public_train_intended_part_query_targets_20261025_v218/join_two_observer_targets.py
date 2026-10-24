"""Join sealed weak SOURCE observations and check all saved jitter supports.

No image/array/model/GT payload read, no fit, no native gold replacement.
"""
import collections
import csv
import hashlib
import json
import math
import os
import struct
from pathlib import Path

V = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/query_visibility_public_train_intended_part_query_targets_20261025_v218')
OUT = V / 'joined_weak_intended_part_targets_v1'
D = 'D_VISIBLE_EXPECTED_PART_WEAK'
E = 'E_FOREIGN_COVER_OF_EXPECTED_PART_WEAK'
STATES = [D, E, 'SELF_OR_OWN_STRUCTURE_UNOBSERVABLE', 'WRONG_PART_OR_GEOMETRY_UNRESOLVED', 'UNKNOWN', 'OUT_OF_FRAME']

def body(p):
    p = Path(p)
    assert p.is_absolute() and '..' not in p.parts and 'v112' not in str(p).lower()
    assert p.is_relative_to(V)
    for q in reversed([p, *p.parents]):
        assert not q.is_symlink()
    with os.fdopen(os.open(p, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as f:
        return f.read()

def binding(p):
    b = body(p)
    return dict(path=str(p), sha256=hashlib.sha256(b).hexdigest(), size_bytes=len(b))

def bound_load(b):
    raw = body(b['path'])
    assert hashlib.sha256(raw).hexdigest() == b['sha256']
    return json.loads(raw)

def new_json(p, d):
    with p.open('x') as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
        f.write('\n')

def roi_ok(box):
    return isinstance(box, list) and len(box) == 4 and all(type(x) is int for x in box) and 0 <= box[0] <= box[2] < 96 and 0 <= box[1] <= box[3] < 96

def flat_support(runs):
    m = [False] * 9216
    previous_end = 0
    for start, length in runs:
        assert type(start) is int and type(length) is int and length > 0 and previous_end <= start < start + length <= 9216
        m[start:start + length] = [True] * length
        previous_end = start + length
    return m

def check_box(box, mask, dx, dy):
    shifted = [box[0] - dx, box[1] - dy, box[2] - dx, box[3] - dy]
    inside = roi_ok(shifted)
    valid = inside and all(mask[y * 96 + x] for y in range(shifted[1], shifted[3] + 1) for x in range(shifted[0], shifted[2] + 1))
    return dict(original_roi96=box, transformed_roi96=shifted, fully_inside=inside, all_selected_pixels_strict_valid=valid)

def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]

def condition32(row, group):
    k = row['native_slot']
    onehot = [float(j == k) for j in range(8)]
    available = [float(j != k and row['generated_query_available'] and group[j]['generated_query_available']) for j in range(8)]
    delta = []
    for j in range(8):
        for axis in range(2):
            value = (group[j]['generated_query_xy'][axis] - row['generated_query_xy'][axis]) / row['L'] if available[j] else 0.0
            delta.append(f32(max(-8., min(8., value))))
    return onehot + available + delta

def main():
    policy = V / 'ROOT_PROSPECTIVE_TWO_OBSERVER_JOIN_AND_JITTER_TARGET_POLICY.json'
    assert json.loads(body(policy))['actual_training_admission_this_phase'] is False
    completes = {}
    for who, expected in [('a', '2db56697c25d25c146d87936f28a00d6f6ad6c582f665d0d6d7c44ab0bc0c251'), ('c', '35d61119513af1ef110d8096f2443880593f29b5f1325ea3413f8cd797e3b12a')]:
        p = V / ('actual_gt_query_part_observations_' + who) / 'COMPLETE.json'
        b = body(p)
        assert hashlib.sha256(b).hexdigest() == expected
        completes[who] = json.loads(b)
        assert completes[who]['completed'] and completes[who]['PASS']
    ab = completes['a']['observations']
    raw = body(ab['path']); assert hashlib.sha256(raw).hexdigest() == ab['sha256']
    aa = [json.loads(line) for line in raw.decode().splitlines()]
    cc = bound_load(completes['c']['observations'])['rows120']
    source = V / 'public_source_gt_query_jitter_v1/ALL360_GENERATED_INPUTS.jsonl'
    rows = [json.loads(line) for line in body(source).decode().splitlines()]
    assert len(rows) == 360 and len(aa) == len(cc) == 120
    a = {r['row_id']: r for r in aa}; c = {r['row_id']: r for r in cc}
    nominal = [r for r in rows if r['normalized_offset'] == [0., 0.]]
    assert len(a) == len(c) == len(nominal) == 120 and set(a) == set(c) == {r['row_id'] for r in nominal}
    groups = collections.defaultdict(dict)
    for r in rows:
        groups[r['group_id']][r['native_slot']] = r
    assert len(groups) == 45 and all(set(g) == set(range(8)) for g in groups.values())
    joined = []
    matrix = [[0] * 6 for _ in range(6)]
    for r in nominal:
        x, y = a[r['row_id']], c[r['row_id']]
        for key in ['frame_position', 'candidate_gt_list_index', 'native_slot', 'canonical_index', 'anatomical_family', 'generated_query_xy', 'L', 'generated_query_available', 'nominal_target_in_frame']:
            assert x[key] == y[key] == r[key], (r['row_id'], key)
        assert x['source_role'] == y['SOURCE_role'] == r['source_role']
        assert x['state'] in STATES and y['state'] in STATES
        assert (x['state'] == 'OUT_OF_FRAME') == (y['state'] == 'OUT_OF_FRAME') == (not r['nominal_target_in_frame'])
        agree = x['state'] == y['state']
        state = x['state'] if agree else 'UNKNOWN'
        matrix[STATES.index(x['state'])][STATES.index(y['state'])] += 1
        boxes = {'A': x['optional_roi96'], 'C': y['optionalroi96']}
        reasons = [] if agree else ['OBSERVER_STATE_DISAGREEMENT']
        weak = state in [D, E]
        if weak:
            for who, observation in [('A', x), ('C', y)]:
                if not observation.get('part_identity_evidence') or not roi_ok(boxes[who]):
                    reasons.append(who + '_PART_EVIDENCE_OR_ROI_MISSING')
                if state == E and (not observation.get('foreign_evidence') or not observation.get('occluder')):
                    reasons.append(who + '_FOREIGN_EVIDENCE_MISSING')
            weak = not reasons
        joined.append(dict(nominal_row_id=r['row_id'], source_role=r['source_role'], frame_position=r['frame_position'], candidate_gt_list_index=r['candidate_gt_list_index'], native_slot=r['native_slot'], anatomical_family=r['anatomical_family'], original_input=r['original_input'], A=x, C=y, consensus_state=state, consensus_agreement=agree, nominal_weak_eligible=weak, failure_reasons=reasons, evidence_roi96_by_observer=boxes, weak_authority='ASSISTANT_CONSENSUS_EXPECTED_FINITE_PART_NOT_PHYSICAL_NATIVE_GOLD', training_admission=False))
    byid = {r['nominal_row_id']: r for r in joined}
    variants = []
    for r in rows:
        s0 = r['row_id'].replace('_S1_', '_S0_').replace('_S2_', '_S0_')
        j = byid[s0]
        dx, dy = [int(z * 96) for z in r['normalized_offset']]
        assert [dx, dy] in [[0, 0], [24, 0], [0, 24]]
        mask = flat_support(r['valid96_RLE'])
        target = [47.5 - dx, 47.5 - dy]
        anchor_valid = all(mask[y * 96 + x] for y in [math.floor(target[1]), math.ceil(target[1])] for x in [math.floor(target[0]), math.ceil(target[0])])
        checks = dict(query_available=r['generated_query_available'], nominal_target_in_frame=r['nominal_target_in_frame'], nominal_target_in_crop96=all(0 <= z <= 95 for z in target), nominal_target_strict_valid=anchor_valid, evidence_boxes={})
        reasons = list(j['failure_reasons'])
        eligible = j['nominal_weak_eligible']
        if eligible:
            for who, box in j['evidence_roi96_by_observer'].items():
                checks['evidence_boxes'][who] = check_box(box, mask, dx, dy)
            if not all(checks[k] for k in ['query_available', 'nominal_target_in_frame', 'nominal_target_in_crop96', 'nominal_target_strict_valid']):
                reasons.append('QUERY_OR_TARGET_SUPPORT_UNAVAILABLE')
            if not all(b['fully_inside'] and b['all_selected_pixels_strict_valid'] for b in checks['evidence_boxes'].values()):
                reasons.append('EVIDENCE_BOX_LOST_OR_INVALID_AFTER_FIXED_SHIFT')
            eligible = not reasons
        state = j['consensus_state']
        if state in [D, E] and not eligible:
            state = 'UNKNOWN'
        variants.append(dict(row_id=r['row_id'], nominal_row_id=s0, source_role=r['source_role'], generated_input=r, nominal_consensus_state=j['consensus_state'], target_state=state, support_checks=checks, failure_reasons=reasons, weak_loss_candidate=eligible, weak_loss_fit_candidate=eligible and r['source_role'] == 'SOURCE_FIT', weak_target_E_binary=(int(state == E) if eligible else None), deployable_condition32=condition32(r, groups[r['group_id']]), condition_origin='GT_DERIVED_TRAINING_QUERY8_NO_ADDITIONAL_GT_POSE_REF_ROI_MASK_FEATURE', primary_physical_native_gold=False, training_admission=False))
    assert sum(x['source_role'] == 'SOURCE_FIT' for x in variants) == 240
    assert all(not x['weak_loss_fit_candidate'] for x in variants if x['source_role'] == 'SOURCE_DEV')
    OUT.mkdir(exist_ok=False)
    for name, data in [('ALL120_JOINED_NOMINAL_OBSERVATIONS.jsonl', joined), ('ALL360_CHECKED_WEAK_TARGET_CANDIDATES.jsonl', variants)]:
        with (OUT / name).open('x') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False, separators=(',', ':')) + '\n')
    with (OUT / 'ALL360_TARGET_CENSUS.csv').open('x', newline='') as f:
        w = csv.writer(f)
        w.writerow(['row_id', 'nominal_row_id', 'source_role', 'native_slot', 'anatomical_family', 'nominal_state', 'target_state', 'weak_loss_candidate', 'weak_loss_fit_candidate', 'failure_reasons'])
        for r in variants:
            w.writerow([r['row_id'], r['nominal_row_id'], r['source_role'], r['generated_input']['native_slot'], r['generated_input']['anatomical_family'], r['nominal_consensus_state'], r['target_state'], r['weak_loss_candidate'], r['weak_loss_fit_candidate'], '|'.join(r['failure_reasons'])])
    summary = dict(completed=True, PASS=True, nominal_rows=120, rows360=360, observer_state_order=STATES, A_state_counts=dict(collections.Counter(x['state'] for x in aa)), C_state_counts=dict(collections.Counter(x['state'] for x in cc)), A_rows_C_columns_confusion=matrix, agreements=sum(x['consensus_agreement'] for x in joined), disagreements=sum(not x['consensus_agreement'] for x in joined), in_frame_nominal_rows=70, in_frame_agreements=sum(x['consensus_agreement'] for x in joined if x['consensus_state'] != 'OUT_OF_FRAME'), consensus_state_counts=dict(collections.Counter(x['consensus_state'] for x in joined)), nominal_weak_eligible=sum(x['nominal_weak_eligible'] for x in joined), generated_state_counts=dict(collections.Counter(x['target_state'] for x in variants)), support_checked_weak_candidates=sum(x['weak_loss_candidate'] for x in variants), FIT_weak_loss_candidates=sum(x['weak_loss_fit_candidate'] for x in variants), DEV_weak_evaluation_candidates=sum(x['weak_loss_candidate'] and x['source_role'] == 'SOURCE_DEV' for x in variants), by_source_role={role:dict(collections.Counter(x['target_state'] for x in variants if x['source_role'] == role)) for role in ['SOURCE_FIT', 'SOURCE_DEV']}, weak_candidates_by_state={s:dict(collections.Counter(x['target_state'] for x in variants if x['weak_loss_candidate'] and x['generated_input']['normalized_offset'] == offset)) for s,offset in [('S0',[0.,0.]),('S1',[.25,0.]),('S2',[0.,.25])]}, consensus_D_E_native_counts=dict(collections.Counter(str(x['native_slot']) for x in joined if x['nominal_weak_eligible'])), GT_derived_training_query_itself=True, original96_U11_unchanged=True, training_admission=False, training_model_protected_eval_or_PnP=0, perfect_physical_gold_blanket_requirement=False, primary_native_gold=0, fully_solved=False)
    new_json(OUT / 'SUMMARY.json', summary)
    new_json(OUT / 'COMPLETE.json', dict(completed=True, PASS=True, code=binding(Path(__file__)), policy=binding(policy), original_observer_completes={who:binding(V / ('actual_gt_query_part_observations_' + who) / 'COMPLETE.json') for who in ['a','c']}, original360=binding(source), outputs={name:binding(OUT / name) for name in ['ALL120_JOINED_NOMINAL_OBSERVATIONS.jsonl', 'ALL360_CHECKED_WEAK_TARGET_CANDIDATES.jsonl', 'ALL360_TARGET_CENSUS.csv', 'SUMMARY.json']}, original_inputs_mutated=False, new_source_payloads_read=0, image_decodes=0, training_admission=False))
    print(json.dumps(summary, ensure_ascii=False))

if __name__ == '__main__':
    main()
