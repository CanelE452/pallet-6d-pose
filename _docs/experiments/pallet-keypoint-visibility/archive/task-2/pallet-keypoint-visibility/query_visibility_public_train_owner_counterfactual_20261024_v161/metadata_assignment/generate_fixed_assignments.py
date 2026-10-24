"""V161 immutable metadata assignments; opens only three authorized JSON inputs.

No image, mask, NumPy, feature, model, or inherited payload hash is opened.
Pair hash serialization and fixed12 tie rules are declared before any pixels.
"""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math

H = Path(__file__).resolve().parent
V = H.parent
DESIGN = V/'ROOT_SOURCE_GENERATION_DESIGN_BEFORE_PIXELS.json'
DESIGN_SHA = '25f38eb0e8b098b681a8bb38e5a3f93086a1a24cdc9fdfa8f3300412382c332b'
READS = []


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def digest_text(text):
    return digest_bytes(text.encode('utf-8'))


def read_bound(binding, jsonl=False):
    p = Path(binding['path']).resolve()
    assert p.suffix in ('.json', '.jsonl')
    assert not any(s in str(p).lower() for s in ('_v112', 'current223', '/private/'))
    raw = p.read_bytes()
    assert digest_bytes(raw) == binding['sha256']
    READS.append(dict(path=str(p), sha256=digest_bytes(raw), bytes=len(raw)))
    return [json.loads(line) for line in raw.decode().splitlines() if line] if jsonl else json.loads(raw)


def bind_created(p):
    p = Path(p).resolve()
    assert p == Path(__file__).resolve() or p.parent == H
    assert p.suffix in ('.py', '.json', '.jsonl')
    return dict(path=str(p), sha256=digest_bytes(p.read_bytes()))


def write_json(name, obj):
    p = H/name
    with p.open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    return bind_created(p)


def write_rows(name, records):
    p = H/name
    with p.open('x') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False)+'\n')
    return bind_created(p)


def longside(anchor, scale):
    height, width = anchor['raw_hw']
    qx, qy = anchor['query_xy']
    L = anchor['L']
    assert isinstance(width, int) and isinstance(height, int) and width > 12 and height > 0
    assert all(math.isfinite(x) for x in (qx, qy, L)) and L > 0
    request = math.ceil(scale*L)
    center_cap = math.floor(min(qx-4, width-5-qx))
    width_cap = math.floor((width-12)/4)
    answer = max(1, min(request, center_cap, width_cap))
    return dict(requested_ceil_scale_L=request,
                horizontal_margin_floor=center_cap,
                image_quarter_width_floor=width_cap,
                longside_pixels=answer,
                cap_applied=answer < request,
                formula='max(1,min(ceil(scale*L),floor(min(qx-4,W-5-qx)),floor((W-12)/4)))',
                donor_pixels_or_model_used=False)


def main():
    design = read_bound(dict(path=str(DESIGN), sha256=DESIGN_SHA))
    anchors = read_bound(design['anchors'], jsonl=True)
    donor_container = read_bound(design['donor_catalog'])
    donors = donor_container['catalog']
    salt = design['assignment']['salt']
    scales = design['assignment']['scales_per_replica']
    assert len(anchors) == design['number_of_anchor_queries'] == 53
    assert len({r['query_id'] for r in anchors}) == 53
    assert len({r['frame_id'] for r in anchors}) == design['number_of_anchor_frames'] == 46
    assert len(donors) == len({r['id'] for r in donors}) == design['number_of_donors'] == 59
    assert scales == [.35, .5, .7] and design['assignment']['rotation_degrees'] == 0
    assert design['pairs_total'] == 159 and design['pairs_per_anchor'] == 3
    assert all(a['split'] == 'SOURCE_DEVELOPMENT' and a['existing_weak_center_owner'] == 'TARGET_VISIBLE_SUPPORTED'
               and a['center4_valid_count'] == 4 and a['center4_valid_bits'] == [[True,True],[True,True]]
               for a in anchors)
    assert all(d['split'] == 'TRAIN' and d['exclusive_alpha_certificate']['admitted']
               and d['scene_GT_owner_completeness']['complete'] and not d['replacement_allowed'] for d in donors)
    ordered_anchors = sorted(anchors, key=lambda a:(digest_text(salt+a['query_id']), a['query_id']))
    ordered_donors = sorted(donors, key=lambda d:(digest_text(salt+d['id']), d['id']))
    # No pair SHA serialization was specified in the root design. This exact
    # serialization is a metadata-only implementation choice frozen here.
    hash_rule = "SHA256_UTF8(salt+anchor_query_id+'::replica'+str(replica)+'::'+donor_id)"
    records = []
    for rank, anchor in enumerate(ordered_anchors):
        for replica, scale in enumerate(scales):
            donor_rank = (3*rank+replica) % 59
            donor = ordered_donors[donor_rank]
            pair_id = f'V161_A{rank:02d}_R{replica}'
            sha_input = salt+anchor['query_id']+'::replica'+str(replica)+'::'+donor['id']
            geometry = longside(anchor, scale)
            records.append(dict(schema='V161_FIXED_SOURCE_METADATA_ASSIGNMENT',
                assignment_index=len(records), pair_id=pair_id, pair_sha256=digest_text(sha_input),
                pair_sha256_rule=hash_rule, pair_sha256_input_utf8=sha_input,
                anchor_rank=rank, anchor_order_sha256=digest_text(salt+anchor['query_id']),
                anchor_query_id=anchor['query_id'], anchor_global_index=anchor['global_index'],
                query_id=anchor['query_id'], query_xy=anchor['query_xy'], L=anchor['L'],
                native_slot=anchor['native_slot'], frame_id=anchor['frame_id'],
                frame_index=anchor['frame_index'], source_id=anchor['source_id'], split=anchor['split'],
                raw_hw=anchor['raw_hw'], anchor=anchor,
                donor_rank=donor_rank, donor_order_sha256=digest_text(salt+donor['id']),
                donor_id=donor['id'], donor_split=donor['split'], donor=donor,
                replica=replica, scale=scale, rotation_degrees=0,
                longside_pixels=geometry['longside_pixels'], geometric_longside=geometry,
                variants=design['variants_per_pair'], all_attempts_retained=True,
                source_only=True, assignment_only=True,
                pixel_proof_status='NOT_RUN', owner_auxiliary_eligibility='NOT_PROVEN',
                new_native_visibility_target='UNKNOWN', new_point_quality_target='UNKNOWN',
                original_labels_immutable_lineage_only=True,
                q_L_native_source_frame_split_unchanged=True,
                inherited_RGB_NPZ_mask_payload_hashes_not_recomputed=True,
                training_admitted=False, selected_model=None, fully_solved=False))
    assert len(records) == 159 and len({r['pair_id'] for r in records}) == 159
    assert len({r['pair_sha256'] for r in records}) == 159
    use = Counter(r['donor_id'] for r in records)
    assert len(use) == 59 and Counter(use.values()) == {3:41, 2:18}
    assert Counter(r['anchor_query_id'] for r in records) == Counter({a['query_id']:3 for a in anchors})
    native_counts = Counter(a['native_slot'] for a in anchors)
    represented = sorted(native_counts)
    assert represented == [0,1,4,5,6,7] and design['missing_anchor_native_slots'] == [2,3]
    by_sha = sorted(records, key=lambda r:(r['pair_sha256'], r['pair_id']))
    selected, selected_anchors = [], set()
    for slot in represented:
        r = next(r for r in by_sha if r['native_slot'] == slot)
        assert r['anchor_query_id'] not in selected_anchors
        selected.append(dict(selection_rank=len(selected)+1, selection_reason=f'SMALLEST_PAIR_SHA_NATIVE{slot}', **r))
        selected_anchors.add(r['anchor_query_id'])
    for r in by_sha:
        if len(selected) == 12:
            break
        if r['anchor_query_id'] in selected_anchors:
            continue
        selected.append(dict(selection_rank=len(selected)+1, selection_reason='REMAINING_SMALLEST_PAIR_SHA_DISTINCT_ANCHOR', **r))
        selected_anchors.add(r['anchor_query_id'])
    assert len(selected) == len(selected_anchors) == 12
    assert set(r['native_slot'] for r in selected) == set(represented)
    assignments = write_rows('ASSIGNMENTS159_BEFORE_PIXELS.jsonl', records)
    visual = write_json('FIXED12_VISUAL_SELECTION_BEFORE_PIXELS.json', dict(
        schema='V161_PROSPECTIVE_FIXED12_DISTINCT_ANCHOR_VISUAL_SELECTION', completed=True,
        before_any_new_pixels=True, design=dict(path=str(DESIGN),sha256=DESIGN_SHA),
        assignments=assignments, pair_sha256_rule=hash_rule,
        tie_rule='pair_sha256 ascending, then pair_id lexicographic',
        selection_rule='For each represented native slot in ascending slot order, take smallest pair SHA; then scan all pair SHAs ascending and append unseen anchor_query_id until12.',
        represented_native_slots=represented, absent_native_slots=[2,3],
        absence_is_not_success_or_negative_evidence=True, no_failure_or_pixel_based_replacement=True,
        number_of_distinct_anchors=12, selected=selected,
        pixel_or_RGB_or_NPZ_or_feature_or_model_bodies_read=0, training_admitted=False))
    donor_use = write_json('DONOR59_FIXED_USE_COUNTS.json', dict(
        completed=True, ordered_donors=[dict(donor_rank=i, donor_id=d['id'],
            donor_order_sha256=digest_text(salt+d['id']), use_count=use[d['id']], split=d['split'])
            for i,d in enumerate(ordered_donors)],
        use_count_histogram={str(k):v for k,v in sorted(Counter(use.values()).items())},
        total_uses=159, all59_used=True, replacement_search_or_rejection=False))
    complete = dict(schema='V161_METADATA_ASSIGNMENT_COMPLETE_BEFORE_PIXELS', completed=True,PASS=True,
        UTC=datetime.now(timezone.utc).isoformat(), code=bind_created(__file__),
        design=dict(path=str(DESIGN),sha256=DESIGN_SHA), exact_input_JSON_reads=READS,
        inputs=dict(anchors=design['anchors'],donor_catalog=design['donor_catalog']),
        outputs=dict(assignments=assignments, fixed12_visual_selection=visual, donor_usage=donor_use),
        anchor_queries=53, anchor_frames=46, donors=59, pairs=159, variants_per_pair=3, prospective_variant_rows=477,
        all53_original_anchor_records_and_all59_donor_lineage_records_preserved=True,
        pairs_per_anchor=3, donor_usage_histogram={'2':18,'3':41},
        scales=scales, pair_sha256_rule=hash_rule,
        native_anchor_counts={str(k):native_counts.get(k,0) for k in range(8)},
        native_pair_counts={str(k):3*native_counts.get(k,0) for k in range(8)},
        represented_native_slots=represented, absent_anchor_native_slots=[2,3],
        all_native8_objective_preserved=True, absence_not_imputed_or_claimed_covered=True,
        geometry_longside_range=[min(r['longside_pixels'] for r in records),max(r['longside_pixels'] for r in records)],
        cap_applied_pairs=sum(r['geometric_longside']['cap_applied'] for r in records),
        selected12_distinct_anchor_query_ids=[r['anchor_query_id'] for r in selected],
        all_attempts_retained=True, donor_search_replacement_or_pixel_conditioning=False,
        metadata_scope_limit='Anchors contain selected q/L and bound original frame factual96 packet; all8 q/L, availability masks and context hashes are not materialized here and must come from original bound initializer/proof inputs in the separately reviewed builder.',
        pixel_RGB_mask_NPY_NPZ_feature_model_or_weight_bodies_read_or_hashed=0,
        source_forward_encoder_initializer_fit_or_PnP_calls=0,
        training_admitted=False, owner_auxiliary_eligibility_not_proven=True,
        current_allowed=False, selected_model=None,fully_solved=False)
    receipt = write_json('METADATA_ASSIGNMENT_COMPLETE.json',complete)
    print(json.dumps(dict(completed=True,PASS=True,receipt=receipt,outputs=complete['outputs'],native_anchor_counts=complete['native_anchor_counts'],cap_applied_pairs=complete['cap_applied_pairs'])))


if __name__ == '__main__':
    main()
