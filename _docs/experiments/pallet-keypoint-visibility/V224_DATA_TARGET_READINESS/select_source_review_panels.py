"""Choose four pre-existing TRAIN query panels without learned scores."""
import hashlib
import json
from pathlib import Path

R = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation/V224_ROOT_READINESS')
SOURCE = R / 'EXISTING_ORIGINAL_QUERY_TRAIN_WEAK_CORRESPONDENCE1208.jsonl'
PREFIX = '/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/query_visibility_mr6d_full_public_source_20261024_v153/original_query_evidence_full230/frames/'


def main():
    rows = [json.loads(s) for s in SOURCE.read_text().splitlines()]
    selected = []
    used = set()
    for state, seq in [('E', '000002'), ('E', '000006'), ('D', '000000'), ('D', '000002')]:
        candidates = sorted((r for r in rows
                             if r['ancestor_record_unchanged']['role_state'] == state
                             and r['ancestor_record_unchanged']['original_input_valid']
                             and r['ancestor_record_unchanged']['source_id'].startswith('val/' + seq + ':')),
                            key=lambda r: r['original_global_index'])
        item = next(r for r in candidates if r['ancestor_record_unchanged']['frame_index'] not in used)
        a = item['ancestor_record_unchanged']
        used.add(a['frame_index'])
        b = a['original_query_evidence']
        assert b['path'].startswith(PREFIX) and '_v112' not in b['path'].lower()
        selected.append({
            'row_id': item['row_id'], 'component_role': 'TRAIN',
            'frame_index': a['frame_index'], 'source_id': a['source_id'],
            'native_slot': a['native_slot'], 'existing_weak_role': state,
            'existing_association_reason': a['association_reason'],
            'existing_role_reason': a['source_role_review_reason'],
            'original_query_panel': b,
        })
    scope = {
        'selection_before_any_panel_read': True,
        'selection': 'First original global row in E sequence2/6 and D sequence0/2, distinct approved TRAIN frames; no visibility scores.',
        'panels': selected,
        'limit': 4,
        'scope': 'Read/view four already-existing original-query display panels only. Diagnostic visual plausibility, not new labels or RAW96 parity proof.',
        'no_new_training_forward_threshold_checkpoint_PnP': True,
        'no_current223_or_other_protected_frame_panel': True,
        'do_not_claim_new_human_or_native_physical_gold': True,
    }
    with (R / 'SOURCE_FIXED4_VISUAL_SCOPE_BEFORE_PANEL_READ.json').open('x') as f:
        json.dump(scope, f, ensure_ascii=False, indent=2); f.write('\n')
    for item in selected:
        b = item['original_query_panel']
        raw = Path(b['path']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == b['sha256']
        item['source_panel_SHA_verified'] = True
        item['source_panel_bytes'] = len(raw)
    with (R / 'SOURCE_FIXED4_VISUAL_INPUTS.json').open('x') as f:
        json.dump(scope, f, ensure_ascii=False, indent=2); f.write('\n')
    print(json.dumps(selected, ensure_ascii=False))


if __name__ == '__main__':
    main()
