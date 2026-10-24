import json
import hashlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
EVIDENCE = BASE / 'review_evidence_preparation'
STATE = {'T': 'TARGET_VISIBLE_SUPPORTED', 'F': 'FOREIGN_VISIBLE_SUPPORTED', 'B': 'BACKGROUND_SUPPORTED', 'U': 'UNKNOWN'}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def binding(path):
    return {'path': str(path), 'sha256': sha(path)}

validity_path = EVIDENCE / 'CENTER4_VALIDITY_INPUT_ONLY_SIDECAR.json'
validity = json.loads(validity_path.read_text())
valid = {r['query_id']: r for r in validity['cases']}
viewed = [BASE / x for x in json.loads((OUT / 'ACTUALLY_VIEWED_IMAGE_PATHS.json').read_text())]
assert len(viewed) == len(set(viewed)) == 69
viewed_set = set(viewed)
outputs = {}
input_bindings = {}
for mode, manifest_name, notes_name, output_name, expected in [
    ('primary', 'REVIEW_INPUT_GROUP_A.json', 'VISUAL_NOTES_PRIMARY_INPUT_ONLY.json', 'PRIMARY195_OWNER_WITNESS.jsonl', 195),
    ('cross', 'BLIND_CROSSREVIEW_FROM_B_BY_A.json', 'VISUAL_NOTES_CROSS_INPUT_ONLY.json', 'CROSS20_OWNER_WITNESS.jsonl', 20),
]:
    manifest_path = EVIDENCE / manifest_name
    manifest = json.loads(manifest_path.read_text())
    notes = json.loads((OUT / notes_name).read_text())
    assert len(manifest['cases']) == len(notes) == expected
    assert {r['case_id'] for r in manifest['cases']} == set(notes)
    if mode == 'cross':
        assert manifest['first_reviewer_labels_exposed'] is False
    rows = []
    for c in manifest['cases']:
        label, reason, page = notes[c['case_id']]
        panel = c['images']['panel']
        assert sha(panel['path']) == panel['sha256']
        v = valid[c['query_id']]
        assert c['center4_valid'] == v['all4valid']
        if not v['all4valid']:
            assert label == 'U'
        if mode == 'primary':
            sheet = next(s for s in manifest['contact_sheets'] if s['page'] == page)
            assert c['case_id'] in sheet['case_ids']
            actually_viewed = sheet['image']
            assert Path(actually_viewed['path']) in viewed_set
            assert sha(actually_viewed['path']) == actually_viewed['sha256']
            reviewed_binding = {'method': 'tools.view_image', 'image': actually_viewed,
                                'contact_sheet_case_position_zero_based': sheet['case_ids'].index(c['case_id']),
                                'case_id': c['case_id']}
        else:
            assert Path(panel['path']) in viewed_set
            reviewed_binding = {'method': 'tools.view_image', 'image': panel, 'case_id': c['case_id']}
        row = {
            'global_index': c['global_index'], 'query_id': c['query_id'], 'case_id': c['case_id'],
            'owner_state': STATE[label], 'reason_KO': reason,
            'confidence': 'UNKNOWN' if label == 'U' else 'SUPPORTED',
            'panel': panel, 'actually_viewed_panel_binding': reviewed_binding,
            'center4_valid': bool(v['all4valid']), 'center4_valid_bits': v['valid2x2_bits'],
            'reviewer': '/root/owner_witness_a_v158', 'review_phase': mode,
            'original_query_xy': c['query_xy'], 'original_L': c['L'],
            'original_qL_sha256': hashlib.sha256(json.dumps({'query_id': c['query_id'], 'query_xy': c['query_xy'], 'L': c['L']}, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
            'source_only_weak_observation': True, 'human_gold': False,
            'physical_native_gold': False, 'coordinate_quality_gold': False,
            'center4_savedRGB96_indices': '[47:49,47:49]',
        }
        assert reason.strip() and (row['owner_state'] == 'UNKNOWN') == (row['confidence'] == 'UNKNOWN')
        rows.append(row)
    assert len({r['query_id'] for r in rows}) == expected
    dest = OUT / output_name
    dest.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    outputs[mode] = {**binding(dest), 'count': len(rows), 'owner_state_counts': dict(Counter(r['owner_state'] for r in rows)),
                     'notes': binding(OUT / notes_name)}
    input_bindings[mode] = binding(manifest_path)

receipt = {
    'schema': 'V158_OWNER_WITNESS_A_REVIEW_COMPLETE', 'completed': True,
    'UTC': datetime.now(timezone.utc).isoformat(), 'reviewer': '/root/owner_witness_a_v158',
    'primary_cases_actually_viewed': 195, 'blind_cross_cases_actually_viewed': 20,
    'input_manifests': input_bindings, 'validity_sidecar': binding(validity_path),
    'outputs': outputs, 'all_actually_viewed_images': [binding(p) for p in viewed],
    'viewed_image_count': len(viewed), 'view_method': 'tools.view_image',
    'primary_view': 'All 49 contact sheets were legible for source frame, raw96, central32 and literal center2x2. Each case was manually inspected; ambiguous boundaries remained UNKNOWN.',
    'cross_view': 'All 20 cross-review cases were viewed as individual 1600x1000 panels before any first-reviewer labels were read.',
    'weak_assistant_observations_only': True, 'human_gold': False,
    'native_coordinate_or_physical_occlusion_gold': False,
    'old_source_labels_or_qL_modified': False, 'training_admission': False,
    'original_roles_GT_reference_saved_predictions_read': False,
    'private_current223_or_V112_accessed': False,
    'models_weights_forward_fit_PnP_calls': 0,
}
(OUT / 'REVIEW_COMPLETE.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'outputs': outputs, 'actually_viewed_images': len(viewed)}, ensure_ascii=False, indent=2))
