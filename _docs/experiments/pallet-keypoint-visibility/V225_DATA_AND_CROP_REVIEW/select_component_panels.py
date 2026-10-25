"""Select one existing E frame per TRAIN component, with same-frame D control.

No model output is read. Existing labels remain weak, not revised here.
"""
import hashlib
import json
from pathlib import Path

P = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
R = P / 'publication_root_preparation/V225_ROOT_REVIEW'
SOURCE = P / 'publication_root_preparation/V224_ROOT_READINESS/EXISTING_ORIGINAL_QUERY_TRAIN_WEAK_CORRESPONDENCE1208.jsonl'
PREFIX = str(P / 'query_visibility_mr6d_full_public_source_20261024_v153/original_query_evidence_full230/frames') + '/'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def save(name, value):
    with (R / name).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def question(item):
    r = item['ancestor_record_unchanged']
    return {k: r[k] for k in ('query_id', 'frame_index', 'native_slot', 'query_xy', 'L',
                              'original_input_valid', 'selected_gt_instance_index',
                              'role_state', 'role_correspondence_supported',
                              'association_reason', 'source_role_review_reason',
                              'occluder_or_uncertainty')}


def main():
    assert '_v112' not in str(SOURCE).lower()
    raw = SOURCE.read_bytes()
    assert digest(raw) == '581a5b2f71f28c7cdbd7a5b313d0f8ded586ed6e498e484344beb18376d26451'
    rows = [json.loads(s) for s in raw.splitlines()]
    assert len(rows) == 1208 and all(r['component_role'] == 'TRAIN' for r in rows)
    e_rows = [r for r in rows if r['ancestor_record_unchanged']['original_input_valid']
              and r['ancestor_record_unchanged']['role_state'] == 'E']
    components = sorted({r['component_id'] for r in e_rows})
    assert len(e_rows) == 44 and len(components) == 15
    panels = []
    for component in components:
        e = min((r for r in e_rows if r['component_id'] == component),
                key=lambda r: r['original_global_index'])
        a = e['ancestor_record_unchanged']
        d_candidates = [r for r in rows
                        if r['component_id'] == component
                        and r['ancestor_record_unchanged']['frame_index'] == a['frame_index']
                        and r['ancestor_record_unchanged']['original_input_valid']
                        and r['ancestor_record_unchanged']['role_state'] == 'D']
        d = min(d_candidates, key=lambda r: r['original_global_index']) if d_candidates else None
        binding = a['original_query_evidence']
        assert binding['path'].startswith(PREFIX) and '_v112' not in binding['path'].lower()
        panels.append({'component_id': component, 'component_role': 'TRAIN',
                       'frame_index': a['frame_index'], 'source_id': a['source_id'],
                       'E_question': question(e), 'same_frame_D_question': question(d) if d else None,
                       'original_query_panel': binding})
    assert len({r['frame_index'] for r in panels}) == 15
    scope = {'completed_selection_before_panel_access': True,
             'source': {'path': str(SOURCE), 'sha256': digest(raw)},
             'selection': 'All15 E components, minimum original_global_index available E per component; minimum same-frame available D as control when present.',
             'model_scores_used': False, 'max_existing_panels': 15,
             'panels': panels, 'new_labels_or_training_admission': False,
             'allowed': 'Only exact15 existing TRAIN panel PNGs, original query and saved RGB96 display; no model, new render or crop correction.',
             'limits': 'Review is unblinded to existing weak proposals. Agreement is not independent human gold, 3px point certification or physical-scene independence.',
             'current223_protected_or_V112_filesystem': False,
             'new_fit_forward_PnP': 0}
    save('SOURCE_E15_COMPONENT_SCOPE_BEFORE_PANEL_READ.json', scope)
    for panel in panels:
        b = panel['original_query_panel']
        body = Path(b['path']).read_bytes()
        assert digest(body) == b['sha256']
        panel['panel_bytes'] = len(body)
        panel['panel_SHA_verified'] = True
    scope['same_frame_D_questions'] = sum(r['same_frame_D_question'] is not None for r in panels)
    scope['completed_panel_binding_check'] = True
    save('SOURCE_E15_COMPONENT_VISUAL_INPUTS.json', scope)
    print(json.dumps({'panels': 15, 'D_controls': scope['same_frame_D_questions'],
                      'frames': [r['frame_index'] for r in panels],
                      'manifest': str(R / 'SOURCE_E15_COMPONENT_VISUAL_INPUTS.json')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
