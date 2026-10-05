"""Read visibility declarations separately from coordinate-reference completion.

This module never writes annotations, creates interaction logs, or promotes a
record to the frozen evaluator's human-reviewed schema.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

from serve import ValidationError


DECLARATIONS_FILENAME = 'USER_VISIBILITY_DECLARATIONS_20261005_V1.json'
AXES = ('external_occlusion', 'self_occlusion', 'out_of_frame', 'definition_uncertain')


def source_corner_sha256(corner):
    raw = json.dumps(corner, sort_keys=True, separators=(',', ':'),
                     ensure_ascii=False, allow_nan=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValidationError(message)


def _read(path):
    try:
        raw = Path(path).read_bytes()
        value = json.loads(raw)
    except (OSError, ValueError, TypeError) as error:
        raise ValidationError('Visibility declaration source cannot be read') from error
    _require(isinstance(value, dict), 'Visibility declaration source must be an object')
    return value, hashlib.sha256(raw).hexdigest()


def _declarations(native):
    path = native.ctx.store_path.parent / DECLARATIONS_FILENAME
    if not path.is_file():
        return []
    document, _ = _read(path)
    _require(document.get('schema') == 'lifter_user_visibility_declarations_v1'
             and document.get('source_kind') == 'human_chat_visibility_declaration'
             and document.get('bindings') == native.ctx.bindings
             and document.get('evaluation_use') is False
             and document.get('human_reference_review_complete') is False
             and document.get('previous_prediction_exposure') is None
             and isinstance(document.get('source_message'), str)
             and bool(document['source_message'].strip()),
             'Chat visibility declaration binding/provenance mismatch')
    queue, queue_hash = _read(document.get('source_queue_file', ''))
    _require(queue_hash == document.get('source_queue_sha256')
             and queue.get('schema') == 'lifter_human_review_batch_v1'
             and queue.get('source_kind') == 'partial_human_review_queue'
             and queue.get('bindings') == native.ctx.bindings
             and queue.get('task_scope') == 'remaining_missing_corners_of_existing_batch'
             and queue.get('frozen_plan_modified') is False,
             'Chat visibility declaration task-queue binding mismatch')
    parent, parent_hash = _read(queue.get('parent_batch_file', ''))
    _require(parent_hash == queue.get('parent_batch_sha256')
             and parent.get('schema') == 'lifter_human_review_batch_v1'
             and parent.get('source_kind') == 'partial_human_review_queue'
             and parent.get('bindings') == native.ctx.bindings
             and parent.get('frozen_plan_modified') is False,
             'Chat visibility declaration original-batch binding mismatch')
    queue_ids = queue.get('frame_ids', [])
    parent_ids = parent.get('frame_ids', [])
    _require(isinstance(queue_ids, list) and queue_ids and len(set(queue_ids)) == len(queue_ids)
             and isinstance(parent_ids, list) and set(queue_ids) <= set(parent_ids),
             'Chat visibility declaration task frame IDs mismatch')
    focus = queue.get('focus_corner_ids', {})
    declarations = document.get('declarations')
    _require(isinstance(declarations, list), 'Chat visibility declarations must be a list')
    seen = set()
    for item in declarations:
        _require(isinstance(item, dict), 'Chat visibility declaration must be an object')
        fid, review_pass, index = item.get('frame_id'), item.get('review_pass'), item.get('corner_id')
        key = (fid, review_pass, index)
        _require(fid in native.ctx.frames and fid in queue_ids
                 and fid not in getattr(native, 'exclusions', set())
                 and review_pass == 'primary' and type(index) is int and 0 <= index < 8
                 and index in focus.get(fid, []) and key not in seen,
                 'Chat visibility declaration frame/pass/corner mismatch')
        seen.add(key)
        _require(item.get('visibility') == 'direct_visible'
                 and item.get('x') is None and item.get('y') is None
                 and item.get('coordinate_source') == 'none'
                 and item.get('human_visibility_declared') is True
                 and item.get('source_kind') == 'human_chat_visibility_declaration'
                 and isinstance(item.get('source_corner_sha256'), str)
                 and len(item['source_corner_sha256']) == 64,
                 'Chat declaration cannot supply coordinates or evaluator approval')
    return declarations


def _valid_record(native, record, fid, review_pass):
    if not isinstance(record, dict) or record.get('frame_id') != fid or record.get('review_pass') != review_pass:
        return False
    frame = native.ctx.frames[fid]
    for key in ('image_sha256', 'width', 'height'):
        if key in record and record[key] != frame[key]:
            return False
    for key in ('plan_sha256', 'corner_contract_sha256', 'corner_definition_version'):
        if key in record and record[key] != native.ctx.bindings[key]:
            return False
    corners = record.get('corners')
    if (not isinstance(corners, list) or len(corners) != 8
            or any(not isinstance(c, dict) or type(c.get('id')) is not int for c in corners)
            or [c['id'] for c in corners] != list(range(8))):
        return False
    for corner in corners:
        visibility = corner.get('visibility')
        if visibility not in (None, 'direct_visible', 'not_direct_visible', 'uncertain'):
            return False
        if not all(type(corner.get(axis)) is bool for axis in AXES):
            return False
        if visibility == 'direct_visible':
            xy = corner.get('x'), corner.get('y')
            if (corner.get('definition_confirmed') is not True or any(corner[axis] for axis in AXES)
                    or not all(type(v) in (int, float) and math.isfinite(v) for v in xy)
                    or not (0 <= xy[0] < frame['width'] and 0 <= xy[1] < frame['height'])):
                return False
        elif corner.get('x') is not None or corner.get('y') is not None:
            return False
        elif visibility is None:
            if any(corner[axis] for axis in AXES):
                return False
        elif sum(corner[axis] for axis in AXES) != 1:
            return False
        elif corner['definition_uncertain'] != (visibility == 'uncertain'):
            return False
    return True


def _current_record(native, fid, review_pass, supplied):
    if supplied is not None:
        return supplied if _valid_record(native, supplied, fid, review_pass) else None
    active = getattr(native, 'record', None)
    if _valid_record(native, active, fid, review_pass):
        return active
    recovery = getattr(native, 'recovery', {})
    if (recovery.get('schema') == 'lifter_native_viewer_recovery_v1'
            and recovery.get('bindings') == native.ctx.bindings
            and recovery.get('evaluation_use') is False):
        draft = recovery.get('drafts', {}).get(fid+'|'+review_pass)
        if draft is None and recovery.get('frame_id') == fid and recovery.get('review_pass') == review_pass:
            draft = recovery
        revision = (native.ctx.record_for(fid, review_pass) or {}).get('revision')
        if (isinstance(draft, dict) and draft.get('frame_id') == fid
                and draft.get('review_pass') == review_pass
                and draft.get('base_revision') == revision
                and _valid_record(native, draft.get('record'), fid, review_pass)):
            return draft['record']
        if draft is not None:
            # A present newer draft, even an invalid one, must not be silently
            # replaced by an older saved record. That could resurrect deletions.
            return None
    from save_visibility_locally import load_visibility_record
    saved = load_visibility_record(native, fid, review_pass)
    return saved if _valid_record(native, saved, fid, review_pass) else None


def visibility_only_status(native, fid, review_pass, current_record=None):
    """Return category completion; chat-visible points have no reference xy."""
    _require(fid in native.ctx.frames and review_pass in ('primary', 'repeat'),
             'Visibility status requested for an unplanned frame/pass')
    _require(review_pass != 'repeat' or native.ctx.frames[fid].get('repeat_review') is True,
             'Visibility status requested for an unplanned repeat')
    declarations = _declarations(native)
    record = _current_record(native, fid, review_pass, current_record)
    result = dict(complete=False, declared_point_ids=[], corners=[],
                  confirmed_category_count=0, manual_coordinate_count=0,
                  raw_document_complete=False, evaluation_use=False,
                  human_reference_review_complete=False)
    if record is None:
        return result
    corners = copy.deepcopy(record['corners'])
    result['manual_coordinate_count'] = sum(c['visibility'] == 'direct_visible' for c in corners)
    result['raw_document_complete'] = all(c['visibility'] is not None for c in corners)
    for declaration in declarations:
        if declaration['frame_id'] != fid or declaration['review_pass'] != review_pass:
            continue
        corner = corners[declaration['corner_id']]
        # Later actual clicks or visibility edits take precedence. A stale
        # declaration cannot fill an edited/deleted corner without a new label.
        if (corner['visibility'] is not None
                or source_corner_sha256(corner) != declaration['source_corner_sha256']):
            continue
        corner.update(visibility='direct_visible', x=None, y=None,
                      coordinate_source='none', category_label_only=True,
                      visibility_label_source='human_chat_visibility_declaration')
        result['declared_point_ids'].append(corner['id'])
    result['corners'] = corners
    result['confirmed_category_count'] = sum(c['visibility'] is not None for c in corners)
    result['complete'] = result['confirmed_category_count'] == 8
    return result
