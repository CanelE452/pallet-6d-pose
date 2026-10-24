"""Join frozen outputs to existing human labels. No fitting, inference, or metrics."""
import csv
import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
RUNS = PROJECT.parent
DOCS = Path('/tmp/pallet-bi-full-execution-20261019-v1/_docs/experiments')
REPO = Path('/home/minjae/Documents/github/pallet-pose')
SOURCES = {
    'saved_outputs': RUNS/'presence_previous_models_audit_20261022_v62/SAME_QUERY_SAVED_OUTPUTS.csv',
    'v61_outputs': RUNS/'target_presence_classifier_20261022_v61/REAL_QUERY_PREDICTIONS.csv',
    'prepared_inputs': RUNS/'external_ray_proxy_head_20261022_v60/PREPARED_INPUTS_REAL_VERIFIED_V2.json.gz',
    'feature_metadata': RUNS/'external_ray_proxy_head_20261022_v60/REAL_FEATURE_METADATA.json',
    'human_sidecar': REPO/'_docs/experiments/pallet_combined_closeout_20261003_v1/closeout_20261006_v1/visibility_square/STATIC_VISIBILITY_MERGE_AUDIT.json',
    'review_manifest': REPO/'_docs/experiments/pallet_static_registry_review_20261003_v1/review/STATIC_REVIEW_MANIFEST.json',
    'scoring_metadata': DOCS/'pallet_visibility_curriculum_20261013_v1/SCORING_METADATA.json',
    'wood_journal': DOCS/'pallet_n3_subpix_20261008_v1/PREDICTIONS.jsonl.gz',
    'wood_targets': REPO/'data/pallet/results/pallet_posefix_replay_diagnosis_v1/TARGETS.json',
    'square_journal': DOCS/'pallet_paper_tables_245_20261011/SQUARE_ROWS.jsonl.gz',
    'legacy_cal': DOCS/'pallet_visibility_curriculum_20261013_v1/nonsevere_v1/OOF_pose_v1/ISOLATED_CAL_SCORING.json',
    'legacy_trace': Path('/home/minjae/Documents/pallet-research/pallet-statewise-correspondence-diagnosis-20261016-v1/e_trace_v2/E_TRACE.jsonl.gz'),
    'legacy_descriptor': RUNS/'presence_previous_models_audit_20261022_v62/EXACT_LEGACY_DESCRIPTOR.json',
    'v61_freeze': RUNS/'target_presence_classifier_20261022_v61/MODEL_FREEZE.json',
}


def read(path):
    with (gzip.open(path, 'rt') if str(path).endswith('.gz') else path.open()) as f:
        return json.load(f)


def lines(path):
    with gzip.open(path, 'rt') as f:
        return [json.loads(s) for s in f if s.strip()]


def csvread(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(name, data):
    (HERE/name).write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')


def write_csv(name, rows):
    with (HERE/name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def truth(value):
    assert value in ('True', 'False')
    return value == 'True'


inputs = read(SOURCES['prepared_inputs'])
frames = {x['id']: x for x in inputs}
assert len(frames) == len(inputs) == 223
outputs = csvread(SOURCES['saved_outputs'])
assert len(outputs) == len({(r['id'], r['native_corner']) for r in outputs}) == 1784
v61 = {(r['id'], r['native_corner']): r for r in csvread(SOURCES['v61_outputs'])}
feature = {r['id']: r for r in read(SOURCES['feature_metadata'])}
wood = {r['id']: r for r in lines(SOURCES['wood_journal']) if r['method'] == 'N3_SUBPIX'}
square_rows = [r for r in lines(SOURCES['square_journal']) if r['backbone'] == 'YOLO' and r['seed'] == 1 and r['method'] == 'N3_THEN_SUBPIX']
square = {f"SQUARE_YOLO:{r['id']}": r for r in square_rows}
assert len(square) == len(square_rows) == 118
targets = read(SOURCES['wood_targets'])
metadata = {r['id']: r for r in read(SOURCES['scoring_metadata'])['frames']}
sidecar = read(SOURCES['human_sidecar'])
review = {(r['population'], r['frame_id'], r['image']['sha256']): r for r in read(SOURCES['review_manifest'])['cases']}
human = {}
for r in sidecar['rows']:
    key = (r['population'], r['frame_id'], r['image_sha256'], int(r['corner_id']))
    assert key not in human
    human[key] = r
cal_ids = set(read(SOURCES['legacy_cal'])['rows'])
trace = {r['id']: r for r in lines(SOURCES['legacy_trace'])}
assert len(cal_ids) == 46 and cal_ids <= trace.keys()
cal_hashes = {trace[i]['image_sha256'] for i in cal_ids}
assert len(cal_hashes) == 46

mapping = {}
frame_rows = []
for u in inputs:
    fid = u['id']
    assert sha(Path(u['image'])) == u['image_sha256']
    is_wood = u['group'] == 'WOOD_PRIMARY105'
    pop, source_id = ('DEV319', fid) if is_wood else ('GREEN0918', fid.split(':')[1])
    journal = wood[fid] if is_wood else square[fid]
    assert journal['qFinal'] == u['initial_points']
    branch = journal['corner']['branch']
    candidate = (targets[fid]['permutations'] if is_wood else journal['evaluation_permutations'])[branch][:8]
    assert sorted(candidate) == list(range(8))
    matched = journal['corner']['matched']
    status = 'VERIFIED_FROZEN_BASELINE_MAPPING' if matched else 'UNKNOWN_UNMATCHED_BASELINE'
    if fid in metadata:
        assert metadata[fid]['native_to_reference_corner_ids'] == candidate
        assert metadata[fid]['frozen_baseline_branch'] == branch
    labels = [human[(pop, source_id, u['image_sha256'], k)] for k in range(8) if (pop, source_id, u['image_sha256'], k) in human]
    assert labels
    review_case = review[(pop, source_id, u['image_sha256'])]
    for kind in ('image', 'annotation'):
        entry = review_case[kind]
        assert sha(REPO/entry['path']) == entry['sha256']
    missing_sources = Counter(c['reference_source'] for c in review_case['corners'] if not c['metric_reference'])
    assert sum(missing_sources.values()) == 8-len(labels)
    for h in labels:
        refpoints = targets[fid]['gt'] if is_wood else journal['evaluation_reference_points']
        assert h['reference_xy'] == refpoints[h['corner_id']]
    mapping[fid] = (candidate, matched, branch, status, pop, source_id)
    counts = Counter(h['category'] for h in labels)
    frame_rows.append(dict(id=fid, group=u['group'], partition=u['partition'], image=u['image'], image_sha256=u['image_sha256'], mapping_status=status, stored_branch=branch, stored_native_to_reference_candidate=json.dumps(candidate), mapping_source=str(SOURCES['wood_journal'] if is_wood else SOURCES['square_journal']), stored_points_equal_current=True, preexisting_metadata_mapping=fid in metadata, human_reference_rows=len(labels), missing_reference_slots=8-len(labels), missing_reference_source_counts=json.dumps(missing_sources), DIRECT_VISIBLE=counts['DIRECT_VISIBLE'], EXTERNAL_OCCLUDED=counts['EXTERNAL_OCCLUDED'], SELF_OCCLUDED=counts['SELF_OCCLUDED'], OUT_OF_FRAME=counts['OUT_OF_FRAME'], historical_CAL46_member_by_id=fid in cal_ids, historical_CAL46_member_by_image_sha=u['image_sha256'] in cal_hashes, exposure='REPEATED_DEVELOPMENT_DATA'))

manifest = []
for r in outputs:
    fid, k = r['id'], int(r['native_corner'])
    u = frames[fid]
    p = v61[(fid, str(k))]
    qx, qy = float(r['query_x']), float(r['query_y'])
    assert [qx, qy] == u['initial_points'][k]
    assert (p['query_x'], p['query_y'], p['presence_probability'], p['valid']) == (r['query_x'], r['query_y'], r['presence_probability'], r['query_valid'])
    perm, matched, branch, status, pop, source_id = mapping[fid]
    ref_id = perm[k] if matched else None
    h = human.get((pop, source_id, u['image_sha256'], ref_id)) if matched else None
    state = h['category'] if h else ('UNANNOTATED' if matched else 'UNKNOWN_MAPPING')
    if fid in metadata:
        assert state == metadata[fid]['human_states_native'][k]
    finite = math.isfinite(qx) and math.isfinite(qy)
    height, width = u['raw_hw']
    inside = finite and 0 <= qx < width and 0 <= qy < height
    nearest_inside = finite and 0 <= math.floor(qx+.5) < width and 0 <= math.floor(qy+.5) < height
    assert truth(r['query_valid']) == inside == nearest_inside
    side = feature[fid]['side']
    offsets = [((i+.5)/9-.5)*side for i in range(9)]
    roi_valid_count = sum(0 <= qx+dx < width and 0 <= qy+dy < height for dy in offsets for dx in offsets) if finite else 0
    semantic_de = state in ('DIRECT_VISIBLE', 'EXTERNAL_OCCLUDED')
    semantic_des = state in ('DIRECT_VISIBLE', 'EXTERNAL_OCCLUDED', 'SELF_OCCLUDED')
    same_rgb = truth(r['legacy_direct_score_comparable_location'])
    if same_rgb:
        assert inside and truth(r['legacy_RGB_evaluated']) and truth(r['legacy_query_coordinate_equal'])
    reason = []
    if not matched: reason.append('mapping_unresolved')
    if state == 'UNANNOTATED': reason.append('unannotated_reference_slot')
    if state == 'OUT_OF_FRAME': reason.append('human_out_of_frame')
    if not inside: reason.append('initial_query_out_of_frame')
    if not truth(r['legacy_RGB_evaluated']): reason.append('legacy_no_RGB_score')
    if not truth(r['legacy_query_coordinate_equal']): reason.append('legacy_query_coordinate_differs')
    if r['legacy_stored_state'] == 'U': reason.append('legacy_abstention')
    if r['legacy_stored_state'] == 'S': reason.append('legacy_geometry_self_gate')
    refxy = h['reference_xy'] if h else None
    manifest.append(dict(**r, image=u['image'], image_sha256=u['image_sha256'], recording=u['recording'], partition=u['partition'], raw_height=height, raw_width=width, mapping_status=status, stored_branch=branch, stored_reference_candidate=perm[k], reference_corner_id=ref_id, human_state=state, human_reference_x=refxy[0] if refxy else None, human_reference_y=refxy[1] if refxy else None, human_label_source=h['label_source'] if h else None, human_reference_in_frame=h['reference_in_frame'] if h else None, initial_reference_error_px=math.hypot(qx-refxy[0], qy-refxy[1]) if refxy else None, query_finite=finite, query_inside_raw_canvas=inside, nearest_pixel_inside_raw_canvas=nearest_inside, roi_valid_sample_count=roi_valid_count, roi_has_valid_sample=roi_valid_count>0, v61_roi_side_raw_px=side, human_reference_inside_nominal_crop=(abs(refxy[0]-qx)<=side/2 and abs(refxy[1]-qy)<=side/2) if refxy else None, semantic_D_vs_E_label_eligible=semantic_de, semantic_D_vs_ES_label_eligible=semantic_des, paired_same_query_RGB_D_vs_E=semantic_de and same_rgb, paired_same_query_RGB_D_vs_ES=semantic_des and same_rgb, same_slot_pipeline_D_vs_E=semantic_de and inside, same_slot_pipeline_D_vs_ES=semantic_des and inside, exclusion_or_status_reasons=';'.join(reason), historical_CAL46_member_by_id=fid in cal_ids, historical_CAL46_member_by_image_sha=u['image_sha256'] in cal_hashes))

write_csv('PREPARED_QUERY_MANIFEST.csv', manifest)
write_csv('VERIFIED_FRAME_INVENTORY.csv', frame_rows)
index_rows = csvread(PROJECT/'PATH_INDEX.csv')
for entry in index_rows:
    path = Path(entry['path'])
    assert path.exists()
    if entry['bytes']: assert path.stat().st_size == int(entry['bytes'])
    if entry['sha256']: assert sha(path) == entry['sha256']
summary = {'schema':'PALLET_SAVED_EVALUATION_PREPARATION_V1', 'date_Asia_Seoul':'2026-10-22', 'new_fits':0, 'new_forward_calls':0, 'new_pose_calls':0, 'classification_metrics_computed':False, 'verified_path_index_entries':len(index_rows), 'verified_path_index_hashes':sum(bool(r['sha256']) for r in index_rows), 'verified_actual_prepared_image_hashes':223, 'verified_original_review_image_and_annotation_hashes':446, 'frames':223, 'native_slots':1784, 'centroid_excluded':True, 'frame_mapping_status':dict(Counter(r['mapping_status'] for r in frame_rows)), 'unresolved_mapping_ids':[r['id'] for r in frame_rows if r['mapping_status'].startswith('UNKNOWN')], 'historical_CAL46_id_overlap':sorted(set(frames)&cal_ids), 'historical_CAL46_image_sha_overlap':sorted({u['image_sha256'] for u in inputs}&cal_hashes), 'groups':{}}
for group in sorted({r['group'] for r in manifest}):
    rs = [r for r in manifest if r['group']==group]
    fs = [r for r in frame_rows if r['group']==group]
    summary['groups'][group] = {'frames':len(fs), 'native_slots':len(rs), 'source_human_reference_state_counts':{state:sum(f[state] for f in fs) for state in ('DIRECT_VISIBLE','EXTERNAL_OCCLUDED','SELF_OCCLUDED','OUT_OF_FRAME')}, 'source_missing_reference_slots':sum(f['missing_reference_slots'] for f in fs), 'mapped_native_state_inventory':dict(Counter(r['human_state'] for r in rs)), 'legacy_stored_state_inventory':dict(Counter(r['legacy_stored_state'] for r in rs)), **{key:sum(bool(r[key]) for r in rs) for key in ('query_inside_raw_canvas','nearest_pixel_inside_raw_canvas','roi_has_valid_sample','semantic_D_vs_E_label_eligible','semantic_D_vs_ES_label_eligible','paired_same_query_RGB_D_vs_E','paired_same_query_RGB_D_vs_ES','same_slot_pipeline_D_vs_E','same_slot_pipeline_D_vs_ES')}, 'same_query_RGB_score_available':sum(truth(r['legacy_direct_score_comparable_location']) for r in rs)}
write_json('PREPARATION_AUDIT.json', summary)
contract = {
    'scope':'Preparation only; existing semantic labels and frozen saved outputs; no new metrics, training, inference or PnP.',
    'evaluation_name':'Semantic keypoint visibility transfer evaluation of query-pixel presence',
    'positive_label':'DIRECT_VISIBLE',
    'tasks':{'D_vs_E':{'negative':['EXTERNAL_OCCLUDED'],'excluded':['SELF_OCCLUDED','OUT_OF_FRAME','UNANNOTATED','UNKNOWN_MAPPING']},'D_vs_ES':{'negative':['EXTERNAL_OCCLUDED','SELF_OCCLUDED'],'excluded':['OUT_OF_FRAME','UNANNOTATED','UNKNOWN_MAPPING']}},
    'v61':{'target':'M4 target visible silhouette at nearest initial-query pixel, not semantic corner visibility', 'threshold':0.5,'checkpoint_epoch':9,'training':'synthetic only; one head, 480 updates', 'crop':'same frozen P3 9x9 ROI; predicted 2D span scale, not pose/K/dimension crop comparison'},
    'legacy':{'descriptor_sha256':read(SOURCES['legacy_descriptor'])['sha256'],'training':'historical real CAL46; not synthetic-only comparator', 'visible_if':'p_DIRECT >= .98','external_if':'p_DIRECT < .96','abstain_if':'.96 <= p_DIRECT < .98','visible_states':['B','I'],'external_state':'O','self_state':'S (geometry before RGB; not a classifier score)','other_states':'U/T/missing remain separate; never force to negative'},
    'cohorts':{'paired_same_query_RGB':'Known mapped human task label + saved valid initial query + legacy RGB score at exactly that coordinate. Preserve legacy RGB U abstentions.', 'same_slot_pipeline':'Known mapped human task label + valid initial query. Preserve S gate and U/T/missing; report all registered denominators and selective decision coverage separately. This cohort does not claim identical query location for geometric gates.', 'all_registered_slots':'Keep all 1784 rows and each exclusion/status; do not silently drop unknowns or OOB.'},
    'future_decision_scoring':{'v61':'Saved present_at_fixed_0_5=True is positive; False is negative in each semantic transfer task.', 'legacy_D_vs_E':'B/I positive; O negative; S is a separate out-of-task prediction, retained in denominator and decision coverage, never forced to external.', 'legacy_D_vs_ES':'B/I positive; O or geometric S negative; identify S as a geometry decision separately from RGB.', 'legacy_unresolved':'U/T/missing remain unresolved outcomes in either task. Keep their rows in the task denominator; any selective confusion table must also show unresolved counts and coverage.', 'ranking':'Use p_present and p_DIRECT on paired_same_query_RGB rows, including RGB U. Never fabricate an RGB probability for S.'},
    'mapping':'Use stored baseline N3 branch/permutation only for evaluation. These are historical GT-evaluation branches, not independent physical identity certification and never inference input. Exact stored qFinal parity checked. No new GT-best branch selection. Unmatched SQUARE_YOLO:026048 stays UNKNOWN.',
    'native_slot_order':'Local cuboid convention: 0 front top left, 1 front top right, 2 front bottom right, 3 front bottom left; 4-7 corresponding back-face order; 8 centroid excluded. These are local cuboid names, not image-coordinate sorting. Synthetic perm_v4[dynamic_id]=native_id is separate from real evaluation native-to-reference mapping; do not reapply it to already-dynamic saved arrays.',
    'unannotated_slots':'Historical metric-reference-only review: wood13 NO_REFERENCE, square346 pnp_projected. Never convert these to EXTERNAL or DIRECT from raw visibility=1/reason=unknown.',
    'pixel_presence_reference':'No real pixel segmentation reference verified. Semantic evaluation cannot establish real single-pixel presence accuracy.',
    'diagnostics':'Initial-to-reference pixel distance and fixed nominal crop coverage are evaluation-only; never recenter or resize crop using reference.',
    'exposure':'All 223 frames are repeated development material. No CAL46 exact ID or image-SHA overlap does not establish independent final evaluation.',
    'reporting':'Separate wood/square and recording; keep OOF/unannotated/mapping/query-OOB/no-RGB/S/U counts. Preserve paired improvement and damage denominators when metrics are later computed.',
    'pro_conversation':{'url':'https://chatgpt.com/c/6ad599a5-f7c4-83ee-8f28-a9146700ff0a','title':'실험 감사와 실패 원인 진단','directly_read_message_ids':['cea1b746-b2b8-5f2d-a746-3c1dfa3529d3','74037ab2-5b79-5adb-a63b-b7ab60e7f63f','c0d5dbe9-2a2e-572b-8396-b431f06eaa17'],'full_historical_conversation_read':False,'relation_to_current_task':'Earlier design included localization/pose; current user goal is presence/occlusion classification.'},
}
write_json('EVALUATION_CONTRACT.json', contract)
bindings = {name:{'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path)} for name,path in SOURCES.items()}
bindings['preparation_script']={'path':str(Path(__file__).resolve()),'sha256':sha(Path(__file__).resolve())}
write_json('SOURCE_BINDINGS.json', bindings)
print(json.dumps(summary, ensure_ascii=False, indent=2))
