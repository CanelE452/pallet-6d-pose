"""V223 SOURCE metadata-only input/target consistency. No payload/model imports."""
import csv
import hashlib
import io
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

P = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
V = P / 'query_visibility_expected_part_counterfactual_source_bridge_20261025_v222'
H = P / 'publication_root_preparation/V223_INPUT_TARGET_CONSISTENCY_B'
ACCESSES = []


def raw(p):
    p = Path(p)
    assert p.is_absolute() and '..' not in p.parts and 'v112' not in str(p).lower()
    assert p.suffix in ('.json', '.jsonl', '.py')
    fd = os.open(p.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in p.parts[1:-1]:
            newfd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = newfd
        leaf = os.open(p.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
        with os.fdopen(leaf, 'rb') as f:
            return f.read()
    finally:
        os.close(fd)


def read(b):
    p = Path(b['path'])
    value = raw(p)
    actual = dict(path=str(p), sha256=hashlib.sha256(value).hexdigest(), size_bytes=len(value))
    assert actual['sha256'] == b['sha256']
    if 'size_bytes' in b:
        assert actual['size_bytes'] == b['size_bytes']
    ACCESSES.append(actual)
    if p.suffix == '.py':
        return value.decode()
    return [json.loads(x) for x in value.splitlines()] if p.suffix == '.jsonl' else json.loads(value)


def write(name, data):
    b = (json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
    fd = os.open(H / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    with os.fdopen(fd, 'wb') as f:
        f.write(b)
    return dict(path=str(H / name), sha256=hashlib.sha256(b).hexdigest(), size_bytes=len(b))


def context(m):
    # Independent F64/clip/FP32 construction from saved input-only metadata.
    k = m['native_slot']
    a = np.zeros(32, np.float32)
    a[k] = 1
    if m['original_query_available']:
        q = np.asarray(m['query_xy'], np.float64)
        for j in range(8):
            if j != k and m['context_available8'][j]:
                a[8+j] = 1
                a[16+2*j:18+2*j] = np.clip(
                    (np.asarray(m['context_q8'][j], np.float64)-q)/np.float64(m['L']), -8, 8
                ).astype(np.float32)
    return a


def valid_digest(runs):
    a = np.zeros(9216, bool)
    for start, length in runs:
        assert type(start) is int and type(length) is int and length > 0
        assert 0 <= start < start + length <= 9216 and not a[start:start+length].any()
        a[start:start+length] = True
    return hashlib.sha256(a.tobytes()).hexdigest()


def pairs(a, b, keys):
    index = defaultdict(list)
    for x in b:
        index[tuple(x[k] for k in keys)].append(x)
    return [(x, y) for x in a for y in index[tuple(x[k] for k in keys)]]


def jkey(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False)


def main():
    entry = read(dict(path=str(V/'ROOT_ACTUAL_FIXED_SOURCE_PAIR_FIT_ENTRY.json'),
                      sha256='193ad96c4e69e1309cbb731e2fad2451baf8cfd85878c78169db9f9be620da3b'))
    complete = read(dict(path=str(V/'paired_source_fit_v1/COMPLETE.json'),
                         sha256='d16d461d70c78711dcc3918662491b0cf2a79521617314b61817b5ecdf61ccb2'))
    arm = complete['arms']['C66CONTINUATION_CONTROL']
    point_outputs = read(arm['strong_outputs'])['rows']
    weak_outputs = read(arm['weak57_outputs'])['rows57']
    point = read(entry['point_metadata7040'])
    source = read(entry['old_caches']['SYNTH_AUG65536']['metadata'])
    weak = read(entry['weak_rows360'])
    cf = read(entry['CF_rows144'])
    runner = read(entry['code'])
    contract = read(entry['contract'])
    assert "a['direct_end_external_logit']" in runner
    assert "model(inp,native,ct[:,8:16],ct[:,16:32])" in runner
    assert "STRONG_AUTHORITY='SYNTH_RENDERED_OWN_IMAGE_POINT0_TARGET_ALONE_D_COMBINED_DE'" in contract
    assert "WEAK_AUTHORITY='ASSISTANT_CONSENSUS_EXPECTED_FINITE_PART_SOURCE_GT_QUERY'" in contract
    assert 'point_contract.point_loss_target(r)' in runner
    assert all(x['native_point_DE_target'] == -1 for x in cf)

    pi = {x['row_id']: x for x in point}
    si = {x['row_id']: x for x in source}
    wi = {x['row_id']: x for x in weak}
    assert len(pi) == len(si) == 7040 and set(pi) == set(si)
    assert len(point_outputs) == 6096 and len(weak_outputs) == 57 and len(cf) == 144
    groups = defaultdict(list)
    for x in weak:
        groups[x['generated_input']['group_id']].append(x['generated_input'])

    strong_rows = []
    for x in point_outputs:
        p, s = pi[x['row_id']], si[x['row_id']]
        for k in ('native_slot', 'query_xy', 'L', 'original_query_available'):
            assert p[k] == s[k]
        a = context(s)
        assert hashlib.sha256(a.tobytes()).hexdigest() == s['context_FP32_native_mask_xy_sha256']
        assert p['point0_target_alone_state'] == 'D'
        assert x['target'] == {'D': 0, 'E': 1}[p['point0_combined_state']]
        strong_rows.append(dict(id=x['row_id'], target=x['target'], native=s['native_slot'],
                                qL=jkey([s['query_xy'], s['L']]),
                                full_query=jkey([s['query_xy'], s['L'], s['context_q8'], s['context_L8'], s['context_available8']]),
                                condition=a.tobytes(), valid=p['actual_valid96_sha256'], rgb=p['actual_rgb96_sha256']))

    weak_rows = []
    for x in weak_outputs:
        w = wi[x['row_id']]
        g = w['generated_input']
        gg = sorted(groups[g['group_id']], key=lambda r: r['native_slot'])
        assert [r['native_slot'] for r in gg] == list(range(8))
        m = dict(native_slot=g['native_slot'], original_query_available=g['generated_query_available'],
                 query_xy=g['generated_query_xy'], L=g['L'],
                 context_q8=[r['generated_query_xy'] for r in gg], context_L8=[r['L'] for r in gg],
                 context_available8=[r['generated_query_available'] for r in gg])
        c = context(m)
        assert np.array_equal(c, np.asarray(w['deployable_condition32'], np.float32))
        assert x['target'] == w['weak_target_E_binary'] and w['weak_loss_candidate']
        weak_rows.append(dict(id=x['row_id'], parent=x['row_id'], family='NATURAL_WEAK', role=x['source_role'],
                              target=x['target'], native=g['native_slot'], qL=jkey([g['generated_query_xy'], g['L']]),
                              full_query=jkey([g['generated_query_xy'], g['L'], m['context_q8'], m['context_L8'], m['context_available8']]),
                              condition=c.tobytes(), valid=valid_digest(g['valid96_RLE']), png=g['RGB96']['sha256']))

    cf_rows = []
    proofs = {}
    for x in cf:
        p = x['parent_row_id']
        w = next(r for r in weak_rows if r['id'] == p)
        assert x['native_slot'] == w['native'] and jkey([x['query_xy'], x['L']]) == w['qL']
        assert x['valid96_RLE'] == wi[p]['generated_input']['valid96_RLE']
        assert np.asarray(x['condition32'], np.float32).tobytes() == w['condition']
        assert jkey([x['query_xy'], x['L'], x['q8_xy'], x['q8_L'], x['q8_available']]) == w['full_query']
        assert x['weak_finite_part_target_D0_E1'] == int(x['variant'] == 'COVER')
        if p not in proofs:
            proof = read(x['paired_proof'])
            assert proof['parent_row_id'] == p and proof['constructed'] is True
            z = proof['proof']
            assert z['COVER_changed_target_pixels'] > 0 and z['COVER_changed_total'] > 0 and z['SHAM_changed_total'] > 0
            assert z['SHAM_changed_target_pixels'] == 0 and z['SHAM_target_RGB_byte_unchanged'] is True
            proofs[p] = proof
        cf_rows.append(dict(id=x['row_id'], parent=p, family=x['variant'], role=x['source_role'],
                            target=x['weak_finite_part_target_D0_E1'], native=x['native_slot'],
                            qL=w['qL'], full_query=w['full_query'], condition=w['condition'],
                            valid=w['valid'], png=x['RGB96']['sha256']))
        if x['variant'] == 'BEFORE':
            assert x['RGB96']['sha256'] == w['png'] and x['weak_finite_part_target_D0_E1'] == w['target'] == 0

    cross = {}
    for name, other in [('NATURAL57', weak_rows), ('CF_BEFORE48', [x for x in cf_rows if x['family'] == 'BEFORE']), ('CF_ALL144', cf_rows)]:
        cross[name] = {key: len(pairs(strong_rows, other, keys)) for key, keys in [
            ('native_qL_matches', ('native', 'qL')), ('full_q8_matches', ('native', 'full_query')),
            ('runtime_condition32_matches', ('condition',)), ('runtime_condition32_and_valid_matches', ('condition', 'valid'))]}
        assert cross[name]['runtime_condition32_matches'] == 0

    strong_groups = defaultdict(list)
    for x in strong_rows:
        strong_groups[(x['rgb'], x['valid'], x['condition'])].append(x)
    assert all(len({x['target'] for x in rr}) == 1 for rr in strong_groups.values())
    strong_duplicate_pairs = sum(len(rr)*(len(rr)-1)//2 for rr in strong_groups.values())
    rows = weak_rows + cf_rows
    conflict_candidates = []
    same_input_pairs = []
    for i, a in enumerate(rows):
        for b in rows[i+1:]:
            if a['condition'] != b['condition'] or a['valid'] != b['valid']:
                continue
            if a['png'] == b['png']:
                same_input_pairs.append((a, b))
                assert a['target'] == b['target']
            if a['target'] != b['target']:
                same_parent = a['parent'] == b['parent']
                assert a['png'] != b['png']
                conflict_candidates.append(dict(
                    row_A=a['id'], row_B=b['id'], family_A=a['family'], family_B=b['family'],
                    target_A=a['target'], target_B=b['target'], source_role=a['role'],
                    same_parent=same_parent, same_original_qL=a['qL'] == b['qL'],
                    same_full_q8=a['full_query'] == b['full_query'],
                    same_FP32_condition=True, same_valid=True, same_encoded_PNG_SHA=False,
                    conclusion=('NOT_IDENTICAL_RGB_PER_BOUND_SAVED_PIXEL_CHANGE_PROOF' if same_parent else
                                'DECODED_RGB_IDENTITY_NOT_ESTABLISHED_NO_COLLISION_CLAIM')))

    assert len(same_input_pairs) == 48 and len(conflict_candidates) == 348
    cc = Counter((x['same_parent'], x['source_role']) for x in conflict_candidates)
    assert sum(x['same_parent'] for x in conflict_candidates) == 144
    assert sum(not x['same_parent'] for x in conflict_candidates) == 204
    assert all(not x['same_original_qL'] and not x['same_full_q8'] for x in conflict_candidates if not x['same_parent'])
    text = io.StringIO()
    writer = csv.DictWriter(text, fieldnames=list(conflict_candidates[0]))
    writer.writeheader()
    writer.writerows(conflict_candidates)
    cb = text.getvalue().encode()
    fd = os.open(H/'CONDITION_MATCHES_NOT_INPUT_CONTRADICTIONS.csv', os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW, 0o644)
    with os.fdopen(fd, 'wb') as f:
        f.write(cb)

    result = dict(
        completed=True, PASS=True, kind='V223_SOURCE_SAVED_METADATA_INPUT_TARGET_CONSISTENCY',
        scope='Only exact approved SOURCE JSON/JSONL and two bound .py files; saved output logits are not model-selection criteria.',
        census=dict(strong_input_metadata=7040, strong_known_outputs=6096, strong_targets=dict(Counter(str(x['target']) for x in strong_rows)),
                    strong_other_not_loss_eligible=944, natural_known=57, natural_targets=dict(Counter(str(x['target']) for x in weak_rows)),
                    natural_FIT=45, natural_DEV=12, CF_parents=48, CF_rows=144, CF_FIT_rows=117, CF_DEV_rows=27),
        cross_strong_vs_weak=cross,
        conclusions=dict(
            confirmed_exact_input_opposite_target_pairs=0,
            this_is_not_a_global_proof_of_zero_conflicts_across_all_possible_SOURCE_inputs=True,
            strong_vs_natural_or_CF_exact_input_collision_excluded_by_runtime_condition=True,
            no_required_condition_match_is_sufficient_to_exclude_full_runtime_input_identity=True,
            namespace_difference_alone_used_as_proof=False,
            strong_metadata_RGB_valid_condition_duplicate_unordered_pairs=strong_duplicate_pairs,
            strong_metadata_unique_RGB_valid_condition_groups=len(strong_groups),
            strong_opposite_target_pairs_with_same_saved_RGB_valid_condition=0,
            natural_D_and_CF_BEFORE_same_input_same_D_pairs=48,
            natural_D_and_CF_BEFORE_same_input_same_D_FIT_pairs=39,
            natural_D_and_CF_BEFORE_same_input_same_D_DEV_pairs=9,
            weak_and_CF_same_encoded_PNG_valid_condition_opposite_pairs=0,
            opposite_label_same_condition_valid_pairs=348,
            same_parent_opposite_pairs_different_RGB_per_saved_proof=144,
            different_parent_opposite_pairs_without_common_decoded_RGB_digest=204,
            unresolved_same_role_FIT_pairs=cc[(False, 'SOURCE_FIT')],
            unresolved_same_role_DEV_pairs=cc[(False, 'SOURCE_DEV')],
            unresolved_different_parent_qL_or_fullq8_exact_matches=0,
            same_parent_pixel_change_evidence='48 exact bound PROOF JSON reports COVER target changes>0, SHAM target changes=0/unchanged; B did not replay pixels.',
            encoded_PNG_SHA_difference_is_not_independent_decoded_RGB_inequality_proof=True),
        authority_and_head=dict(
            strong='Synthetic actual own-image point0: target-alone D, combined state D/E; native ray/point reference, not query-center pixel or generic region owner.',
            natural='Assistant-consensus expected finite-part weak D/E at GT-derived training queries; not physical native-point gold.',
            CF='Controlled expected finite-part weak BEFORE D / COVER E / SHAM D; native_point_DE_target remains -1.',
            same_output_key='direct_end_external_logit',
            both_losses_update_same_scalar=True,
            no_exact_collision_does_not_make_target_semantic_mixture_harmless=True,
            semantic_negative_transfer_not_established_by_this_metadata_audit=True,
            shared_input_assembly='Same base.assemble_local_input(...,use_sam=False), native argmax(condition[:8]), other mask8 and XY16 across all three families.',
            actual_input_payloads_and_normalization_not_replayed=True),
        minimal_next_evidence=[
            'For any unresolved within-weak/CF condition match, generate a common decoded uint8 RGB96 + bool valid96 + int64 native + FP32 mask/XY digest at the same SOURCE input-assembly boundary; no model forward or protected data is needed. Current scope does not authorize that payload replay.',
            'First controlled hypothesis: retain encoder, old point output/loss, conditioning, initialization and common draws, and separate finite-part weak output semantics. This tests scalar-head semantic coupling before changing pooling/conditioning. Keep both SOURCE arms and role split; no assumed benefit.',
            'Condition-before-pooling changes the feature interaction/architecture in addition to the semantic question, so keep it as a separate later SOURCE-controlled factor rather than combine changes now.'
        ],
        limits=dict(
            original_NPZ_PT_RGB_GT_or_protected_payloads_read=0, images_viewed=0, new_models_or_forwards_or_training=0,
            broad_scan=0, symlink_follow=0, source_science_writes=0,
            strong_RGB_valid_hashes_are_saved_metadata_not_rehashed_payloads=True,
            weak_valid_hashes_derived_from_exact_RLE_bool_bytes=True,
            no_private_score_cause_claim=True, no_relabel_or_training_admission=True,
            prior_frozen_phases_untouched=True),
        input_bindings=ACCESSES,
        candidate_pair_table=dict(path=str(H/'CONDITION_MATCHES_NOT_INPUT_CONTRADICTIONS.csv'), sha256=hashlib.sha256(cb).hexdigest(), size_bytes=len(cb)),
        code_evidence=dict(runner='1e88783241417b97554bdc5b33225e0d6b8c2ae937d0f0288575acd1c26b3957',
                           contract='40fb7d9e1994d7cd1aa30d063c19d785789062b7e6176af9cdb3ed5bde973c94',
                           runner_lines=[165, 173, 202, 210, 238, 242, 249], contract_lines=[8, 9, 14, 27, 106]))
    binding = write('REVIEW.json', result)
    print(json.dumps(dict(completed=True, PASS=True, review=binding, cross=cross,
                          strong_unique_groups=len(strong_groups), strong_duplicate_pairs=strong_duplicate_pairs,
                          same_parent=144, unresolved_other_parent=204,
                          unresolved_FIT=cc[(False,'SOURCE_FIT')], unresolved_DEV=cc[(False,'SOURCE_DEV')]), ensure_ascii=False))


if __name__ == '__main__':
    main()
