"""Save actual native annotation actions without claiming unconfirmed provenance."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

from serve import ValidationError, utc_now


SCHEMA = 'lifter_native_visibility_save_v1'
PROGRESS = 'lifter_native_visibility_progress_v1'
AXES = ('external_occlusion', 'self_occlusion', 'out_of_frame', 'definition_uncertain')


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def validate_document(native, document):
    fid, review_pass = document.get('frame_id'),document.get('review_pass')
    frame = native.ctx.frames.get(fid)
    if (document.get('schema') != SCHEMA or document.get('bindings') != native.ctx.bindings
            or document.get('source_kind') != 'actual_human_native_save'
            or document.get('evaluation_use') is not False
            or document.get('provenance_status') != 'UNVERIFIED'
            or document.get('previous_prediction_exposure') is not None
            or review_pass not in ('primary','repeat') or frame is None
            or document.get('image_sha256') != frame['image_sha256']):
        raise ValidationError('Local visibility annotation binding/provenance mismatch')
    record = document.get('record',{})
    corners = record.get('corners',[])
    if (record.get('frame_id') != fid or record.get('review_pass') != review_pass
            or record.get('reviewer') is not None or record.get('status') != 'draft'
            or len(corners) != 8 or any(type(c.get('id')) is not int for c in corners)
            or [c.get('id') for c in corners] != list(range(8))):
        raise ValidationError('Local visibility annotation frame/corner mismatch')
    for c in corners:
        if not all(type(c.get(axis)) is bool for axis in AXES):
            raise ValidationError('Local visibility annotation mask invalid')
        if c.get('visibility') == 'direct_visible':
            xy = (c.get('x'),c.get('y'))
            if (c.get('definition_confirmed') is not True or any(c[axis] for axis in AXES)
                    or not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in xy)
                    or not (0 <= xy[0] < frame['width'] and 0 <= xy[1] < frame['height'])):
                raise ValidationError('Local direct-visible annotation must contain an actual valid click')
        elif c.get('visibility') in ('not_direct_visible','uncertain'):
            if c.get('x') is not None or c.get('y') is not None or c.get('definition_confirmed') is not False:
                raise ValidationError('Hidden/uncertain annotation must not contain reference coordinates')
            if (sum(c[axis] for axis in AXES) != 1
                    or c['definition_uncertain'] != (c['visibility']=='uncertain')):
                raise ValidationError('Local visibility annotation status/mask mismatch')
        else:
            raise ValidationError('Incomplete visibility annotation cannot count as saved complete')
    return record


def load_visibility_record(native, frame_id, review_pass):
    if not native.visibility_progress_path.is_file():
        return None
    progress = json.loads(native.visibility_progress_path.read_text())
    if progress.get('schema') != PROGRESS or progress.get('bindings') != native.ctx.bindings:
        raise ValidationError('Local visibility progress binding mismatch')
    entry = progress.get('records',{}).get(frame_id+'|'+review_pass)
    if entry is None:
        return None
    path = Path(entry.get('saved_file',''))
    if not path.is_file():
        return None
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != entry.get('file_sha256'):
        raise ValidationError('Local visibility annotation hash mismatch')
    document = json.loads(raw)
    record = validate_document(native,document)
    if document['frame_id'] != frame_id or document['review_pass'] != review_pass:
        raise ValidationError('Local visibility annotation requested frame/pass mismatch')
    if document.get('base_revision') != (native.ctx.record_for(frame_id,review_pass) or {}).get('revision'):
        return None
    return copy.deepcopy(record)


def saved_visibility_ids(native, review_pass):
    if not native.visibility_progress_path.is_file():
        return set()
    progress = json.loads(native.visibility_progress_path.read_text())
    if progress.get('schema') != PROGRESS or progress.get('bindings') != native.ctx.bindings:
        raise ValidationError('Local visibility progress binding mismatch')
    complete = set()
    for key,entry in progress.get('records',{}).items():
        if entry.get('review_pass') != review_pass:
            continue
        path = Path(entry.get('saved_file',''))
        if not path.is_file():
            continue
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry.get('file_sha256'):
            raise ValidationError('Local visibility annotation changed; completion cannot be reused')
        document = json.loads(raw)
        record = validate_document(native,document)
        fid = document['frame_id']
        if key != fid+'|'+review_pass or entry.get('frame_id') != fid or document['review_pass'] != review_pass:
            raise ValidationError('Local visibility progress frame/pass mismatch')
        revision = (native.ctx.record_for(fid,review_pass) or {}).get('revision')
        if document.get('base_revision') != revision:
            continue
        recovery = native.recovery.get('drafts',{}).get(key)
        if recovery and recovery.get('base_revision') == revision:
            current = recovery.get('record',{})
            if current.get('corners') != record['corners'] or current.get('object') != record['object']:
                continue
        if native.record and native.record.get('frame_id') == fid and native.record.get('review_pass') == review_pass:
            if native.record['corners'] != record['corners'] or native.record['object'] != record['object']:
                continue
        complete.add(fid)
    return complete


def save_visibility_locally(native, state, path):
    """Called by an actual save action; no profile UI or frozen evaluator writes."""
    native.record['status'] = 'draft'
    native.action(state,'human_native_visibility_save_without_provenance_prompt')
    record = copy.deepcopy(native.record)
    record['reviewer'] = None
    fid,review_pass = record['frame_id'],record['review_pass']
    document = dict(schema=SCHEMA,source_kind='actual_human_native_save',evaluation_use=False,
        annotation_state='saved_complete',provenance_status='UNVERIFIED',
        previous_prediction_exposure=None,human_provenance_confirmation=False,
        bindings=copy.deepcopy(native.ctx.bindings),frame_id=fid,review_pass=review_pass,
        image_sha256=native.ctx.frames[fid]['image_sha256'],
        base_revision=(native.ctx.record_for(fid,review_pass) or {}).get('revision'),
        saved_at=utc_now(),record=record,independent_repeat=False,
        editor='scripts/annotate/annotate.py',training_use=False)
    validate_document(native,document)
    source = Path(path)
    target = source.with_name(source.stem+'.VISIBILITY_SAVED.json').resolve()
    progress = dict(schema=PROGRESS,bindings=copy.deepcopy(native.ctx.bindings),records={})
    if native.visibility_progress_path.is_file():
        previous = json.loads(native.visibility_progress_path.read_text())
        if previous.get('schema') != PROGRESS or previous.get('bindings') != native.ctx.bindings:
            raise ValidationError('Local visibility progress binding mismatch')
        progress = previous
    _write(target,document)
    progress['records'][fid+'|'+review_pass] = dict(frame_id=fid,review_pass=review_pass,
        saved_file=str(target),file_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        saved_at=document['saved_at'],provenance_status='UNVERIFIED',evaluation_use=False)
    _write(native.visibility_progress_path,progress)
    state.dirty = state.annotation_dirty = state.frame_tags_dirty = False
    state.discard_armed = None
    if not native.native_pnp:
        native.visibility_checkpoint = native.snapshot(state)
        native.visibility_checkpoint_history_index = len(native.history)
    native.persist_recovery(state)
    native.notice = '현재 이미지 주석 저장 완료.'
    print('SAVED_LOCAL_VISIBILITY',fid,review_pass,str(target),flush=True)
    return True
