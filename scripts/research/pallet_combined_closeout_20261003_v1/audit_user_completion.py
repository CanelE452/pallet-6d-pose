"""Preserve actual user drafts and audit completeness without submitting labels."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
ADAPTER = ROOT / 'scripts/research/pallet_lifter_case_review_20261003_v1'
REVIEW = ROOT / 'data/pallet/results/pallet_lifter_case_review_20261003_v1/review'
OUTPUT = ROOT / ('_docs/experiments/pallet_combined_closeout_20261003_v1/'
                 'human_review_20261004_v1/completion_20261005_v1')
ARCHIVE = REVIEW / 'user_completion_20261005_v1'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write_new(path, raw):
    """Never overwrite a different archival artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise RuntimeError(f'Archive exists with different content: {path}')
        return
    with path.open('xb') as stream:
        stream.write(raw)


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf-8')


def run():
    sys.path.insert(0, str(ADAPTER))
    from open_existing_annotation import Context, NativeReview
    from save_visibility_locally import validate_document

    # Context validates the frozen bindings and all original frame-image hashes.
    ctx = Context(REVIEW/'MANIFEST.json', REVIEW.parent/'LIFTER_EVALUATION_PLAN.json',
                  REVIEW/'CORNER_CONTRACT.json', REVIEW/'annotations_in_progress.json')
    native = NativeReview(ctx, None, REVIEW/'native_workspace', passes=('primary',),
                         native_pnp=False, batch_plan=REVIEW/'SMALL_BATCH_12_V1.json')
    frozen = [REVIEW/'MANIFEST.json', REVIEW.parent/'LIFTER_EVALUATION_PLAN.json',
              REVIEW/'CORNER_CONTRACT.json', REVIEW/'USER_EXCLUSIONS.json',
              REVIEW/'SMALL_BATCH_12_V1.json', REVIEW/'VIEWER_DRAFT_RECOVERY.json',
              REVIEW/'NATIVE_PNP_PROGRESS.json', REVIEW/'NATIVE_VISIBILITY_PROGRESS.json',
              ROOT/'scripts/annotate/annotate.py', ADAPTER/'review/serve.py']
    frozen += sorted((REVIEW/'native_annotations').rglob('*.json'))
    hashes = {str(path): digest(path.read_bytes()) for path in frozen}
    raw_recovery = (REVIEW/'VIEWER_DRAFT_RECOVERY.json').read_bytes()
    recovery = json.loads(raw_recovery)
    if recovery != native.recovery:
        raise RuntimeError('Recovery changed during audit startup')
    progress = json.loads((REVIEW/'NATIVE_PNP_PROGRESS.json').read_text())
    if progress['bindings'] != ctx.bindings:
        raise RuntimeError('PnP progress binding mismatch')
    saved_ids = native.saved_visibility_ids('primary')
    counts = dict(direct_visible=0, self_occlusion=0, external_occlusion=0,
                  out_of_frame=0, uncertain=0, unentered=0)
    rows, documents, missing = [], [], []
    for fid in native.batch['frame_ids']:
        key = fid+'|primary'
        draft = recovery['drafts'][key]
        if draft['frame_id'] != fid or draft['review_pass'] != 'primary':
            raise RuntimeError('Draft frame/pass mismatch')
        record = copy.deepcopy(draft['record'])
        corners = record['corners']
        if [c['id'] for c in corners] != list(range(8)):
            raise RuntimeError('Invalid canonical corner IDs')
        pnp_path = Path(progress['records'][key]['assisted_file'])
        pnp = json.loads(pnp_path.read_text())
        if (pnp['bindings'] != ctx.bindings or pnp['frame_id'] != fid
                or pnp['image_sha256'] != ctx.frames[fid]['image_sha256']):
            raise RuntimeError('PnP source identity/binding mismatch')
        missing_ids, visible_ids, self_ids, manual_sources, self_sources = [], [], [], {}, {}
        for c in corners:
            index = c['id']
            if c['visibility'] is None:
                counts['unentered'] += 1
                missing_ids.append(index)
                source = pnp['manual_reference_corners'][index]
                editor = pnp['editor_keypoint_annotations'][index]
                if source['visibility'] is not None or source['x'] is not None or source['y'] is not None:
                    raise RuntimeError('Missing point has a saved manual source; audit recovery first')
                missing.append(dict(frame_id=fid, review_pass='primary', corner_id=index,
                    current_state=None, reason='No actual current manual click or human visibility judgment',
                    editor_source=editor['source'], source_pnp=str(pnp_path),
                    pnp_coordinate_is_reference=False, requires_actual_user_input=True))
            elif c['visibility'] == 'direct_visible':
                source = pnp['manual_reference_corners'][index]
                if (source['visibility'] != 'direct_visible'
                        or (source['x'],source['y']) != (c['x'],c['y'])):
                    # Actual clicks made AFTER saving PnP must also be retained.
                    # Require the independently hash-validated actual S document
                    # and its post-PnP manual action, never the generated PnP point.
                    from save_visibility_locally import load_visibility_record
                    saved_record = load_visibility_record(native,fid,'primary')
                    later_clicks = [a for a in record['interaction_log']
                        if a.get('action') == 'manual_raw_click_and_physical_confirmation'
                        and a.get('performed_at','') > progress['records'][key]['saved_at']]
                    if (saved_record is None or saved_record['corners'][index] != c
                            or not later_clicks):
                        raise RuntimeError('Manual point changed; inspect its actual click provenance')
                    manual_sources[str(index)] = 'actual_manual_click_after_PnP_verified_in_actual_S_save'
                else:
                    manual_sources[str(index)] = 'original_PnP_manual_reference_corners'
                visible_ids.append(index)
                counts['direct_visible'] += 1
            else:
                if c['x'] is not None or c['y'] is not None:
                    raise RuntimeError('Hidden point has reference coordinates')
                axes = [k for k in ('self_occlusion','external_occlusion','out_of_frame',
                                   'definition_uncertain') if c[k]]
                if len(axes) != 1:
                    raise RuntimeError('Invalid hidden-point mask')
                axis = axes[0]
                counts['uncertain' if axis == 'definition_uncertain' else axis] += 1
                if axis == 'self_occlusion':
                    if c.get('geometry_confirmation',{}).get('source') != 'human_confirmed_manual_click_geometry':
                        from save_visibility_locally import load_visibility_record
                        saved_record = load_visibility_record(native,fid,'primary')
                        if (saved_record is None or saved_record['corners'][index] != c
                                or not any(a.get('action') == 'human_self_occlusion'
                                           for a in record['interaction_log'])):
                            raise RuntimeError('Self-occlusion lacks actual human evidence')
                        self_sources[str(index)] = 'actual_human_self_occlusion_button_verified_in_actual_S_save'
                    else:
                        self_sources[str(index)] = 'human_confirmed_manual_click_geometry'
                    self_ids.append(index)
        complete = not missing_ids
        if complete:
            # Reuse the existing shape validator only. Never invoke any save/action.
            validator_doc = dict(schema='lifter_native_visibility_save_v1',
                source_kind='actual_human_native_save', evaluation_use=False,
                provenance_status='UNVERIFIED', previous_prediction_exposure=None,
                bindings=ctx.bindings, frame_id=fid, review_pass='primary',
                image_sha256=ctx.frames[fid]['image_sha256'], record=record)
            validate_document(native,validator_doc)
        relative = f'{fid.replace(":", "_")}.ACTUAL_DRAFT_SNAPSHOT.json'
        doc = dict(schema='lifter_user_completion_snapshot_v1',
            source_kind='machine_preserved_actual_annotation', evaluation_use=False,
            training_use=False, independent_repeat=False, human_actions_synthesized=False,
            human_provenance_confirmation=False, previous_prediction_exposure=None,
            provenance_status='UNVERIFIED', bindings=copy.deepcopy(ctx.bindings),
            image_sha256=ctx.frames[fid]['image_sha256'], frame_id=fid, review_pass='primary',
            user_requested_closeout=True, annotation_complete=complete,
            actual_native_S_save_present=fid in saved_ids,
            original_recovery_sha256=digest(raw_recovery),
            original_record_sha256=digest(encoded(record)),
            draft=copy.deepcopy(draft))
        documents.append((ARCHIVE/relative,encoded(doc)))
        rows.append(dict(frame_id=fid, image_sha256=ctx.frames[fid]['image_sha256'],
            complete=complete, actual_native_S_save_present=fid in saved_ids,
            manual_corner_ids=visible_ids, human_self_occlusion_ids=self_ids,
            manual_corner_source_by_id=manual_sources,
            self_occlusion_source_by_id=self_sources,
            missing_corner_ids=missing_ids, preserved_snapshot=str(ARCHIVE/relative),
            preserved_snapshot_sha256=digest(encoded(doc)), source_pnp=str(pnp_path),
            source_pnp_sha256=digest(pnp_path.read_bytes())))

    if sum(counts.values()) != 8*len(rows):
        raise RuntimeError('Denominator mismatch')
    # Guard against concurrent GUI edits before writing any archival snapshots.
    for path,expected in hashes.items():
        if digest(Path(path).read_bytes()) != expected:
            raise RuntimeError('Source changed during audit: '+path)
    write_new(ARCHIVE/'VIEWER_DRAFT_RECOVERY_EXACT_COPY.json',raw_recovery)
    for path,raw in documents:
        write_new(path,raw)
    completed = [r['frame_id'] for r in rows if r['complete']]
    draft_complete = [r['frame_id'] for r in rows if r['complete'] and not r['actual_native_S_save_present']]
    audit = dict(schema='lifter_actual_user_completion_audit_v1',
        created_at=datetime.now(timezone.utc).isoformat(),
        source_kind='machine_audit_of_actual_annotation', bindings=ctx.bindings,
        actual_source_sha256=hashes, original_sources_preserved=True,
        frozen_evaluator_unchanged=True, evaluated_predictions_read=False,
        labels_inferred=False, human_actions_synthesized=False,
        original_batch_sha256=digest((REVIEW/'SMALL_BATCH_12_V1.json').read_bytes()),
        original_batch_frame_ids=native.batch['frame_ids'],
        selected_frames=len(rows), selected_corner_slots=8*len(rows),
        entered_corner_states=sum(counts.values())-counts['unentered'], counts=counts,
        complete_frames=len(completed), complete_frame_ids=completed,
        actual_native_S_saved_frames=len(saved_ids),
        complete_drafts_preserved_without_synthesized_S=len(draft_complete),
        complete_draft_frame_ids=draft_complete,
        pnp_saved_frames=len(native.saved_pnp_ids('primary')),
        eligible_primary_frames=native.requested_counts()['primary_required'],
        eligible_repeat_frames=native.requested_counts()['repeat_required'],
        official_primary_reviewed=0, official_repeat_reviewed=0,
        official_reference_exists=(REVIEW/'LIFTER_REFERENCE_REVIEWED.json').exists(),
        provenance_status='UNVERIFIED', previous_prediction_exposure=None,
        evaluation_use=False, independent_physical_T_R_reference=False,
        independent_T_R_accuracy='x', rows=rows, missing_corners=missing,
        execution=dict(new_training_runs=0, optimizer_updates=0, new_inference_runs=0,
                       generated_PDFs=0, pushes=0, external_uploads=0))
    # Keep the actual frozen status factual instead of manufacturing completion.
    if ctx.store['records'] or audit['official_reference_exists']:
        raise RuntimeError('Official reference state changed; inspect rather than reporting zero')
    (OUTPUT/'COMPLETION_AUDIT.json').write_bytes(encoded(audit))
    tasks = dict(schema='lifter_missing_corner_work_items_v1',
        source_kind='machine_list_of_unentered_actual_fields',
        bindings=ctx.bindings, parent_batch_sha256=audit['original_batch_sha256'],
        original_evaluation_batch_unchanged=True, statistical_selection=False,
        evaluation_use=False, labels_inferred=False, completed_work_should_not_repeat=True,
        tasks=missing)
    (OUTPUT/'MISSING_CORNER_TASKS.json').write_bytes(encoded(tasks))
    for path,expected in hashes.items():
        if digest(Path(path).read_bytes()) != expected:
            raise RuntimeError('Original source changed: '+path)
    return audit


if __name__ == '__main__':
    result = run()
    print(json.dumps({k:result[k] for k in ('selected_frames','entered_corner_states',
        'selected_corner_slots','counts','complete_frames','actual_native_S_saved_frames',
        'complete_drafts_preserved_without_synthesized_S','pnp_saved_frames')}, ensure_ascii=False))
