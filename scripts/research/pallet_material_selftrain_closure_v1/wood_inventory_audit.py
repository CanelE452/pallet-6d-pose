"""Metadata-only Wood inventory; never opens predictions or scores.

Coordinates in annotation JSON are not used or emitted. The model-independent
September material review and source-recording audit define sampling and roles.
The caller owns final protocol locking; this module emits proposed frozen IDs.
"""
from collections import Counter
import csv
import json
from pathlib import Path

from . import common as C

ROOT = C.ROOT
WS = ROOT / 'data/evaluation/pallet_eval_v1'
GROUPS = ROOT / 'data/pallet/results/site_environment_audit_v1/SOURCE_RECORDING_GROUPS.json'
SPLIT = ROOT / '_docs/experiments/pallet_replay_clean19_v1/SPLIT.json'
BUDGET = ROOT / '_docs/experiments/pallet_selftraining_paper_closure_v1/MANUAL_SUPERVISION_BUDGET.json'
MATERIAL = 'wood_small_80x59x14'


def overlap(a, b):
    """Compare identities and provenance without consulting model outputs."""
    def vals(rows, key):
        if key == 'sha256':
            return {r['image']['sha256'] for r in rows}
        if key == 'id':
            return {r.get('canonical_frame_id', r['id']) for r in rows}
        if key == 'session':
            return {r.get('canonical_session', r['session']) for r in rows}
        return {r[key] for r in rows}
    return {key: sorted(vals(a, key) & vals(b, key))
            for key in ('id', 'sha256', 'session', 'recording')}


def annotation_metadata(row):
    # The legacy file stores coordinates and metadata together. Read only the
    # provenance/visibility fields; no coordinate is inspected or used.
    d = C.read(ROOT / row['annotation']['path'])
    o = d['objects'][0]
    kp = o.get('keypoint_annotations', [])
    return dict(camera=d.get('camera_data'), dimensions_m=o.get('dimensions_m'),
                physical_dimensions_m=o.get('physical_dimensions_m'),
                gt_source=o.get('gt_source'),
                keypoint_source_counts=dict(Counter(k.get('source', 'missing') for k in kp)),
                direct_visible_corner_ids=[i for i, k in enumerate(kp[:8])
                    if k.get('source') == 'manual_click' and k.get('visibility') == 2],
                migration_status=o.get('migration_status'),
                manual_review_reasons=o.get('manual_review_reasons', []),
                pose_reference='legacy annotation-derived pose, not independent physical 6D truth')


def midpoint(rows, n):
    rows = sorted(rows, key=lambda r: r['source_frame'])
    assert n <= len(rows)
    return [rows[((2 * i + 1) * len(rows)) // (2 * n)] for i in range(n)]


def main():
    groups = C.read(GROUPS)
    alias = {s['session_key']: g['recording_id'] for g in groups['groups']
             if not g['is_collection'] for s in g['sessions']}
    split = C.read(SPLIT)
    all319 = split['train'] + split['evaluation']
    assert len(all319) == 319
    by_id = {r['id']: r for r in all319}
    teachers = [dict(by_id[r['id']], manual_corner_ids=r['manual_corner_ids'])
                for r in C.read(BUDGET)['teacher_records']]
    assert len(teachers) == 9
    teacher_recordings = {r['recording'] for r in teachers}
    assert teacher_recordings == {'REC_001', 'REC_002'}
    historical = [r for r in split['evaluation'] if r['object_type'] == 'wood']
    assert len(historical) == 116
    eligible_eval = [r for r in historical if r['recording'] not in teacher_recordings]
    assert len(eligible_eval) == 45
    assert {r['recording'] for r in eligible_eval} == {'REC_039', 'REC_042'}
    prohibited_hashes = {r['image']['sha256'] for r in all319 + teachers}
    sources = [C.bind(p) for p in (GROUPS, SPLIT, BUDGET, Path(__file__))]
    reviewed, eligible = [], []
    session_summaries = []
    for light in ('day', 'night'):
        session = WS / f'incoming/sessions/real_unlabeled_{light}_20260830'
        info = C.read(session / 'session.json')
        review_path = session / 'manifests/frame_review.csv'
        frames_path = session / 'manifests/frames.csv'
        reviews = {Path(r['frame']).stem: r['review_label']
                   for r in csv.DictReader(review_path.open())}
        frames = list(csv.DictReader(frames_path.open()))
        rel = str(session.relative_to(ROOT))
        current = []
        for r in frames:
            if reviews[r['frame']] != 'wood':
                continue
            image = dict(path=str((WS / r['image_path']).relative_to(ROOT)),
                         sha256=r['image_sha256'], bytes=int(r['byte_size']))
            current.append(dict(id='WOOD__' + r['image_sha256'], kind='WOOD',
                object_type=MATERIAL, source_frame=r['frame'], image=image,
                canonical_frame_id=f'wood_{light}_01:' + r['frame'],
                canonical_session=f'wood_{light}_01',
                session=rel, recording=alias[rel], recording_id=alias[rel],
                raw_hw=[int(r['height']), int(r['width'])], K=info['camera']['K'],
                physical_dimensions_xyz_m=[.8, .14, .59],
                camera_quality=info['camera']['intrinsics_quality'],
                type_source=str(review_path.relative_to(ROOT)),
                type_source_rule='review_label == wood; preexisting actual-pixel review, not model selection'))
        allowed = [r for r in current if r['image']['sha256'] not in prohibited_hashes]
        reviewed.extend(current)
        eligible.extend(allowed)
        session_summaries.append(dict(session=rel, recording=alias[rel],
            reviewed_wood=len(current), excluded_historical_annotated_sha=len(current)-len(allowed),
            eligible=len(allowed), camera=info['camera'], resolution=info['resolution']))
        sources.extend(C.bind(p) for p in (session/'session.json', review_path,
                                         frames_path, session/'cam_K.txt'))
    assert len(reviewed) == len({r['image']['sha256'] for r in reviewed}) == 13908
    assert len(eligible) == 13828
    # Largest-remainder proportional allocation, ties in fixed day/night order.
    cap = 1000
    pools = [[r for r in eligible if r['session'] == s['session']] for s in session_summaries]
    ideals = [cap * len(p) / len(eligible) for p in pools]
    allocations = [int(v) for v in ideals]
    for i in sorted(range(len(pools)), key=lambda i: (-(ideals[i]-allocations[i]), i))[:cap-sum(allocations)]:
        allocations[i] += 1
    selected = [r for p, n in zip(pools, allocations) for r in midpoint(p, n)]
    assert len(selected) == len({r['id'] for r in selected}) == 1000
    # Verify the selected candidate/evaluation/teacher RGB bytes, not only
    # metadata. Unselected incoming entries retain archive-extraction SHA hashes.
    for row in selected + historical + teachers:
        C.verify(row['image'])
    for row in historical:
        C.verify(row['annotation'])
    legacy_catalogue = []
    for subdir, rec in [('pallet_20260618_183705', 'REC_039'),
                        ('pallet_20260618_184309', 'REC_042'),
                        ('20260618_132917', 'REC_043')]:
        path = ROOT / 'data/pallet/raw_data/wood/selected' / subdir
        images = [C.bind(p) for p in sorted(path.iterdir()) if p.suffix.lower() in ('.jpg', '.png')]
        legacy_catalogue.append(dict(session=str(path.relative_to(ROOT)), recording=rec,
            images=images, count=len(images),
            role='EVALUATION_RECORDING_EXCLUDED_FROM_TRAIN' if rec != 'REC_043' else 'NO_VALIDATED_CAMERA_CONTRACT',
            camera_contract='scaled RealSense D435I sensor profile, 1280x720' if rec != 'REC_043'
                else '592x1280 portrait; missing K/hfov; no calibration invented'))
    derived = [str(p.relative_to(ROOT)) for p in (ROOT/'data/pallet/raw_data/wood/selected').glob('infer*')]
    meta = {r['id']: annotation_metadata(r) for r in historical}
    grouped = lambda rows: dict(Counter(r['recording'] for r in rows))
    stats = dict(historical116=len(historical), eligible45=len(eligible_eval),
        historical_recordings=grouped(historical), eligible_recordings=grouped(eligible_eval),
        eligible_severity=dict(Counter(r['severity'] for r in eligible_eval)),
        teacher_materials=dict(Counter(r['object_type'] for r in teachers)),
        teacher_recordings=grouped(teachers), incoming_reviewed_wood=len(reviewed),
        incoming_eligible_wood=len(eligible), candidate_cap=cap,
        candidate_allocation=dict(zip([s['session'] for s in session_summaries], allocations)),
        all_wood_primary_rgb_sha_unique=len({r['image']['sha256'] for r in reviewed} |
            {im['sha256'] for group in legacy_catalogue for im in group['images']}),
        historical_eval_trusted_visible=sum(len(meta[r['id']]['direct_visible_corner_ids']) for r in historical),
        eligible_eval_trusted_visible=sum(len(meta[r['id']]['direct_visible_corner_ids']) for r in eligible_eval))
    comparisons = dict(
        teacher_vs_historical116=overlap(teachers, historical),
        teacher_vs_main45=overlap(teachers, eligible_eval),
        reviewed_pool_vs_historical116=overlap(reviewed, historical),
        eligible_pool_vs_main45=overlap(eligible, eligible_eval),
        selected_candidates_vs_main45=overlap(selected, eligible_eval),
        selected_candidates_vs_all319=overlap(selected, all319),
        selected_candidates_vs_teacher=overlap(selected, teachers))
    result = dict(schema_version='wood_inventory_agent_v1', sources=sources, stats=stats,
        reviewed_wood_records=reviewed, eligible_pool_records=eligible,
        candidate_records=selected, historical_evaluation_records=historical,
        proposed_main_evaluation_records=eligible_eval, teacher_records=teachers,
        session_summaries=session_summaries, legacy_raw_catalogue=legacy_catalogue,
        derived_prediction_visualization_directories_not_rgb_candidates=derived,
        metadata_only_annotation_provenance=meta, overlaps=comparisons,
        session_aliases={g['recording_id']: [s['session_key'] for s in g['sessions']]
            for g in groups['groups'] if g['recording_id'] in
            {'REC_001', 'REC_002', 'REC_039', 'REC_042', 'REC_043'}},
        population_rule='Exclude all teacher-training recordings (both Wood day and Plastic night) from historical116. No prediction/score input.',
        candidate_rule='1000 total: largest-remainder proportional day/night allocation and midpoint-spaced sorted source frame IDs after removing all319 annotated/teacher RGB SHA.',
        hash_verification='candidate1000, historical116, teacher9 and legacy raw460 bytes verified now; all incoming13908 hashes reused from independently extracted per-entry SHA manifests',
        Q1_wood_main='UNRESOLVED: no direct-click provenance among eligible45; generic manual label is insufficient',
        scores_or_predictions_read=False, annotation_coordinates_used=False,
        proposed_population_not_independent_confirmation=True)
    C.save(C.RAW/'WOOD_INVENTORY_AGENT.json', result)
    C.save(C.RAW/'WOOD_INVENTORY_AGENT_SUMMARY.json', dict(stats=stats, overlaps=comparisons,
        population_rule=result['population_rule'], candidate_rule=result['candidate_rule'],
        Q1_wood_main=result['Q1_wood_main'], sources=sources))
    print(json.dumps(stats, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
