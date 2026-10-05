"""Submit existing inputs only after the native GUI's actual human confirmation.

There is intentionally no CLI entry point, model import, or automatic approval.
Callbacks follow Tk askokcancel(title, message) and the existing choose_profile
(path, change=False, preview=image) contracts.  Cancellation changes no review
store; the viewer's ordinary draft recovery remains available.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import uuid

from serve import ValidationError, utc_now, validate_reviewer


ACTION = 'human_approve_existing_manual_inputs_and_defer_unentered_states'
REASON = '사용자가 기존 수동좌표만 사용하고 미입력점은 추가 분류 없이 판단 보류하기로 일괄 확인'
AXES = ('external_occlusion', 'self_occlusion', 'out_of_frame', 'definition_uncertain')


def _require(condition, message):
    if not condition:
        raise ValidationError(message)


def _bytes(path):
    return Path(path).read_bytes()


def _digest(value):
    return hashlib.sha256(value).hexdigest()


def _read(path):
    raw = _bytes(path)
    value = json.loads(raw)
    _require(isinstance(value, dict), 'Existing-input evidence must be an object')
    return value, raw


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def _check_inputs(native, frame_ids):
    ctx = native.ctx
    for path, key in ((ctx.manifest_path, 'manifest_sha256'),
            (ctx.plan_path, 'plan_sha256'), (ctx.contract_path, 'corner_contract_sha256')):
        _require(_digest(_bytes(path)) == ctx.bindings[key], 'Input changed before approval: ' + key)
    _require(native.read_exclusions() == native.exclusions, 'User exclusions changed before approval')
    for fid in frame_ids:
        _require(fid in ctx.frames and fid not in native.exclusions, 'Invalid/excluded approval frame')
        _require(_digest(_bytes(ctx.images[fid])) == ctx.frames[fid]['image_sha256'],
                 'Raw image changed before approval: ' + fid)


def _bound_record(ctx, candidate, frame_id, reviewer, timestamp):
    """In-memory prevalidation only; pending states retain None here."""
    frame = ctx.frames[frame_id]
    record = copy.deepcopy(candidate)
    _require(record.get('frame_id') == frame_id and record.get('review_pass') == 'primary',
             'Reused draft frame/pass mismatch')
    for key in ('session_id', 'image_sha256', 'width', 'height'):
        _require(key not in record or record[key] == frame[key], 'Draft frame binding mismatch: ' + key)
        record[key] = frame[key]
    for key in ('plan_sha256', 'corner_contract_sha256', 'corner_definition_version'):
        _require(key not in record or record[key] == ctx.bindings[key], 'Draft input binding mismatch: ' + key)
        record[key] = ctx.bindings[key]
    record.update(status='draft', source_kind='human_in_progress', reviewer=copy.deepcopy(reviewer),
        review_time=dict(started_at=timestamp, finished_at=timestamp, duration_seconds=0.,
                         clock_source='server_wall_and_monotonic'))
    record.setdefault('skip_reason', '')
    record.setdefault('edit_reason', '')
    ctx.validate_record(record, allow_draft=True)
    for corner in record['corners']:
        if corner['visibility'] == 'direct_visible':
            _require(corner.get('x') is not None and corner.get('y') is not None,
                     'Direct reference is missing its actual click coordinates')
        if corner['visibility'] is None:
            _require(corner.get('x') is None and corner.get('y') is None,
                     'Unentered states cannot contain generated reference coordinates')
    return record


def approve_existing_clicks(native, state, confirm, choose_profile):
    """Return True after an actual human-approved batch submission, False on cancel.

    Only the current twelve-frame primary batch is eligible.  This function
    does not mark unentered corners as occluded, visible, or outside the image.
    Validation errors propagate to the native GUI's normal visible error path.
    """
    _require(callable(confirm) and callable(choose_profile), 'Actual GUI callbacks required')
    _require(native.batch is not None and native.batch_path is not None,
             'Existing-input approval requires the explicit twelve-frame batch')
    ids = list(native.batch['frame_ids'])
    _require(len(ids) == 12 and len(set(ids)) == 12, 'Existing-input approval scope must be twelve frames')
    _require(native.record is not None and native.record.get('review_pass') == 'primary'
             and native.record.get('frame_id') in ids, 'Open a primary batch image before approval')
    native.persist_recovery(state)
    recovery, recovery_raw = _read(native.recovery_path)
    _require(recovery.get('schema') == 'lifter_native_viewer_recovery_v1'
             and recovery.get('bindings') == native.ctx.bindings
             and recovery.get('evaluation_use') is False, 'Recovery binding/schema mismatch')
    batch, batch_raw = _read(native.batch_path)
    _require(batch.get('schema') == 'lifter_human_review_batch_v1'
             and batch.get('bindings') == native.ctx.bindings
             and batch.get('source_kind') == 'partial_human_review_queue'
             and batch.get('frozen_plan_modified') is False
             and batch.get('frame_ids') == ids and batch.get('repeat_frame_ids') == [],
             'Partial batch changed before approval')
    _check_inputs(native, ids)
    ctx = native.ctx
    progress_path = ctx.store_path.parent / 'NATIVE_PNP_PROGRESS.json'
    progress, progress_raw = _read(progress_path)
    _require(progress.get('schema') == 'lifter_native_pnp_progress_v1'
             and progress.get('bindings') == ctx.bindings, 'PnP progress binding mismatch')
    proof_sources = {}
    for fid in ids:
        entry = progress.get('records', {}).get(fid + '|primary', {})
        path = Path(entry.get('assisted_file', ''))
        _require(path.is_file(), 'Missing original saved PnP evidence: ' + fid)
        proof, raw = _read(path)
        _require(proof.get('schema') == 'lifter_native_pnp_assistance_v1'
                 and proof.get('source_kind') == 'human_assisted_annotation'
                 and proof.get('evaluation_use') is False
                 and proof.get('bindings') == ctx.bindings and proof.get('frame_id') == fid
                 and proof.get('review_pass') == 'primary'
                 and proof.get('image_sha256') == ctx.frames[fid]['image_sha256'],
                 'Original PnP evidence binding mismatch: ' + fid)
        proof_sources[fid] = (path, raw)
    # Current unsaved changes take precedence, then already validated submitted
    # records, then same-revision local drafts.  The PnP document never supplies
    # guessed coordinates or visibility labels to the reference records.
    candidates, origins, revisions = {}, {}, {}
    for fid in ids:
        previous = ctx.record_for(fid, 'primary')
        expected_revision = (previous or {}).get('revision')
        revisions[fid] = expected_revision
        entry = recovery.get('drafts', {}).get(fid + '|primary')
        if fid == native.record['frame_id']:
            candidate = copy.deepcopy(native.record)
            _require(candidate.get('revision') == expected_revision, 'Current draft revision mismatch: ' + fid)
            origin = 'current_native_record'
        elif previous and previous.get('status') == 'reviewed':
            candidate = copy.deepcopy(previous)
            origin = 'existing_validated_context_record'
        else:
            _require(isinstance(entry, dict) and entry.get('frame_id') == fid
                     and entry.get('review_pass') == 'primary'
                     and entry.get('base_revision') == expected_revision,
                     'Missing/stale same-revision draft: ' + fid)
            candidate = copy.deepcopy(entry.get('record'))
            _require(isinstance(candidate, dict), 'Invalid local draft: ' + fid)
            origin = 'same_revision_viewer_draft'
        _require(candidate.get('status') in ('draft', 'reviewed'), 'Skipped/unrecognized input record: ' + fid)
        candidates[fid], origins[fid] = candidate, origin
    reviewer = copy.deepcopy(native.reviewer)
    if reviewer is None or native.change_reviewer:
        reviewer = choose_profile(native.profile_path, change=native.change_reviewer, preview=state.img)
        if reviewer is None:
            return False
        reviewer = copy.deepcopy(reviewer)
    validate_reviewer(reviewer)
    for fid, candidate in candidates.items():
        owner = candidate.get('reviewer')
        _require(owner is None or (isinstance(owner, dict) and owner.get('id') == reviewer['id']),
                 'Another reviewer owns this existing input: ' + fid)
    timestamp = utc_now()
    prepared = {fid: _bound_record(ctx, record, fid, reviewer, timestamp)
                for fid, record in candidates.items()}
    summary = dict(frame_count=len(ids), direct_points=sum(c['visibility'] == 'direct_visible'
        for record in prepared.values() for c in record['corners']),
        confirmed_states=sum(c['visibility'] is not None
            for record in prepared.values() for c in record['corners']),
        pending_states=sum(c['visibility'] is None
            for record in prepared.values() for c in record['corners']))
    message = (f"{summary['frame_count']}장의 기존 직접 클릭 {summary['direct_points']}점과 "
        f"이미 확인한 상태 {summary['confirmed_states'] - summary['direct_points']}개를 그대로 사용합니다.\n"
        f"미입력 {summary['pending_states']}개만 '판단 보류'로 저장합니다.\n\n"
        '기존 클릭은 본인이 앞서 입력한 것임을 확인합니다.\n'
        '미입력 점의 가림 종류나 화면 밖 여부는 추정하지 않습니다.\n'
        '새로 재검사한 결과나 독립 반복 실험으로 기록하지 않습니다.\n\n'
        '이 내용으로 기존 입력만 일괄 저장하시겠습니까?')
    if confirm('기존 입력만 사용 · 미입력은 판단 보류', message) is not True:
        return False
    approved_at = utc_now()
    _check_inputs(native, ids)
    _require(_bytes(native.recovery_path) == recovery_raw and _bytes(native.batch_path) == batch_raw
             and _bytes(progress_path) == progress_raw, 'Input evidence changed during confirmation')
    for fid in ids:
        _require((ctx.record_for(fid, 'primary') or {}).get('revision') == revisions[fid],
                 'Review revision changed during confirmation: ' + fid)
        _require(_bytes(proof_sources[fid][0]) == proof_sources[fid][1],
                 'Original PnP evidence changed during confirmation: ' + fid)
    action_id = uuid.uuid4().hex
    final_records = {}
    for fid, source in prepared.items():
        record = copy.deepcopy(source)
        pending = [c['id'] for c in record['corners'] if c['visibility'] is None]
        for corner in record['corners']:
            if corner['visibility'] is None:
                corner.update(visibility='uncertain', x=None, y=None, definition_confirmed=False,
                    external_occlusion=False, self_occlusion=False, out_of_frame=False,
                    definition_uncertain=False, reason=REASON)
        if record['object'].get('presence') is None:
            record['object'].update(presence='uncertain', target_identity_confirmed=False, target_object_id=None)
        actor = copy.deepcopy(reviewer)
        actor.update(previous_annotation_exposure=True)
        if (record.get('native_pnp_assistance') or record.get('native_pnp_exposure')
                or record.get('actual_geometry_assistance') or record.get('actual_geometry_exposure')):
            actor['machine_assistance'] = True
        actor['exposure_notes'] += '\n기존 본인 입력을 재사용하는 일괄 확인이며 독립 재검사가 아님. 미입력 점은 직접 판단 보류로 확인함.'
        record.update(status='reviewed', source_kind='human_reviewed', reviewer=actor,
            independently_repeated=False, independent_repeat=False,
            edit_reason='실제 사람 버튼으로 기존 입력을 재사용하고 미입력만 판단 보류로 일괄 확인함')
        record.setdefault('interaction_log', []).append(dict(action=ACTION,
            performed_at=approved_at, action_id=action_id, explicit_human_confirmation=True,
            direct_coordinates_reused_unchanged=True, deferred_corner_ids=pending,
            independently_repeated=False))
        record['existing_input_approval'] = dict(action_id=action_id, approved_at=approved_at,
            source_kind='actual_human_gui_bulk_confirmation', input_action='explicit_existing_inputs_button',
            owner_confirmed_by_actual_human=True, independently_repeated=False,
            source_record_origin=origins[fid], source_base_revision=revisions[fid],
            original_reviewer=copy.deepcopy(candidates[fid].get('reviewer')),
            original_review_time=copy.deepcopy(candidates[fid].get('review_time')),
            review_time_scope='post_confirmation_submission_only',
            review_time_is_original_annotation_time=False,
            recovery_file=str(native.recovery_path), recovery_sha256=_digest(recovery_raw),
            assisted_file=str(proof_sources[fid][0]), assisted_sha256=_digest(proof_sources[fid][1]),
            directly_clicked_corners=sum(c['visibility'] == 'direct_visible' for c in record['corners']),
            already_confirmed_states=sum(c['visibility'] is not None for c in source['corners']),
            deferred_corner_ids=pending, deferred_count=len(pending),
            guessed_coordinates_created=False, occlusion_labels_inferred=False)
        ctx.validate_record(record)
        final_records[fid] = record
    # Preserve exact original evidence before any actual submission updates.
    archive = ctx.store_path.parent / 'existing_input_approval_sources' / action_id
    archive.mkdir(parents=True, exist_ok=False)
    (archive / 'VIEWER_DRAFT_RECOVERY.json').write_bytes(recovery_raw)
    (archive / 'NATIVE_PNP_PROGRESS.json').write_bytes(progress_raw)
    (archive / 'SMALL_BATCH_12_V1.json').write_bytes(batch_raw)
    if ctx.store_path.exists():
        (archive / 'REVIEW_STORE_BEFORE_APPROVAL.json').write_bytes(_bytes(ctx.store_path))
    for fid, (_, raw) in proof_sources.items():
        (archive / (fid.replace(':', '_') + '.PNP_ASSISTED.json')).write_bytes(raw)
    saved, tokens = {}, {}
    for fid, record in final_records.items():
        started = ctx.start_session(dict(frame_id=fid, review_pass='primary', reviewer=record['reviewer']))
        tokens[fid] = started['token']
        saved[fid] = ctx.save(dict(token=started['token'], record=record))['record']
    native.reviewer = reviewer
    native.record = saved[native.record['frame_id']]
    native.token = tokens[native.record['frame_id']]
    exported = ctx.export()
    exported['requested_use'] = dict(counts=native.requested_counts(),
        excluded_frame_ids=sorted(native.exclusions), frozen_population_preserved=True,
        partial_batch=dict(path=str(native.batch_path), sha256=_digest(batch_raw), frame_ids=ids,
            counts=native.batch_counts(), original_population_claimed_complete=False))
    _write(ctx.store_path.parent / 'LIFTER_REFERENCE_REVIEWED.json', exported)
    receipt = dict(schema='lifter_existing_input_approval_receipt_v1', action_id=action_id,
        source_kind='actual_human_gui_bulk_confirmation', approved_at=approved_at,
        input_bindings=ctx.bindings, reviewer_id=reviewer['id'], summary=summary,
        frame_ids=ids, source_archive=str(archive), independently_repeated=False,
        review_time_scope='post_confirmation_submission_only',
        full_population_claimed_complete=False, original_PnP_files_modified=False,
        records=[dict(frame_id=fid, revision=record['revision'],
            existing_input_approval=record['existing_input_approval']) for fid, record in saved.items()])
    _write(archive / 'APPROVAL_RECEIPT.json', receipt)
    _write(ctx.store_path.parent / 'EXISTING_CLICKS_APPROVAL_RECEIPT.json', receipt)
    state.dirty = state.annotation_dirty = state.frame_tags_dirty = False
    state.discard_armed = None
    native.notice = f"기존 직접 클릭 {summary['direct_points']}점 사용 · 미입력 {summary['pending_states']}개 판단 보류 저장 완료"
    return True
