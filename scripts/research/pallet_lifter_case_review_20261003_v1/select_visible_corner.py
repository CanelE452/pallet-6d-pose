"""Reuse an actual manual click or request a new click; never use PnP coordinates."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

from serve import ValidationError, utc_now


def _read_object(raw):
    try:
        value = json.loads(raw)
    except (ValueError, TypeError) as error:
        raise ValidationError('Manual-point source JSON is invalid') from error
    if not isinstance(value, dict):
        raise ValidationError('Manual-point source must be a JSON object')
    return value


def select_visible_corner(native, state):
    """Actual visible-button action.  Return True only for a restored manual point."""
    index = min(max(int(state.active), 0), 7)
    record = native.record
    fid, review_pass = record['frame_id'], record['review_pass']
    frame = native.ctx.frames[fid]

    def manual_candidate(source, provenance):
        if source.get('frame_id') != fid or source.get('review_pass') != review_pass:
            return None
        corner = next((c for c in source.get('corners', []) if c.get('id') == index), None)
        if not corner or corner.get('visibility') != 'direct_visible':
            return None
        for key in ('image_sha256', 'width', 'height'):
            if key in source and source[key] != frame[key]:
                raise ValidationError('Visible source image binding mismatch')
        for key in ('plan_sha256', 'corner_contract_sha256', 'corner_definition_version'):
            if key in source and source[key] != native.ctx.bindings[key]:
                raise ValidationError('Visible source input binding mismatch')
        x, y = corner.get('x'), corner.get('y')
        if (not corner.get('definition_confirmed')
                or not all(isinstance(v, (float, int)) and not isinstance(v, bool)
                           and math.isfinite(v) for v in (x, y))
                or not (0 <= x < frame['width'] and 0 <= y < frame['height'])
                or any(corner.get(axis) for axis in
                       ('external_occlusion', 'self_occlusion', 'out_of_frame', 'definition_uncertain'))):
            raise ValidationError('Original direct click is invalid; generated coordinates cannot replace it')
        return copy.deepcopy(corner), copy.deepcopy(source.get('object')), provenance

    found = manual_candidate(record, dict(source='current_manual_reference_record'))
    if found is None:
        for snapshot in reversed(native.history):
            found = manual_candidate(snapshot.get('record', {}), dict(source='same_frame_manual_undo_history'))
            if found is not None:
                break
    if found is None:
        paths = []
        progress_path = native.ctx.store_path.parent / 'NATIVE_PNP_PROGRESS.json'
        if progress_path.is_file():
            progress = _read_object(progress_path.read_text())
            if (progress.get('schema') != 'lifter_native_pnp_progress_v1'
                    or progress.get('bindings') != native.ctx.bindings):
                raise ValidationError('Saved manual-point progress binding mismatch')
            entry = progress.get('records', {}).get(fid + '|' + review_pass, {})
            if entry.get('assisted_file'):
                paths.append(Path(entry['assisted_file']))
        for path, key in native.path_map.items():
            if tuple(key) == (fid, review_pass):
                path = Path(path)
                paths.append(path.with_name(path.stem + '.PNP_ASSISTED.json'))
        for path in dict.fromkeys(paths):
            if not path.is_file():
                continue
            raw = path.read_bytes()
            document = _read_object(raw)
            if (document.get('schema') != 'lifter_native_pnp_assistance_v1'
                    or document.get('bindings') != native.ctx.bindings
                    or document.get('frame_id') != fid or document.get('review_pass') != review_pass
                    or document.get('image_sha256') != frame['image_sha256']
                    or document.get('source_kind') != 'human_assisted_annotation'
                    or document.get('evaluation_use') is not False
                    or document.get('reference_scope') != 'actual_manual_clicks_only'):
                raise ValidationError('Original manual-point proof binding mismatch')
            found = manual_candidate(dict(frame_id=fid, review_pass=review_pass,
                corners=document.get('manual_reference_corners', []), object=record.get('object')),
                dict(source='saved_actual_manual_reference_corners', file=str(path),
                     sha256=hashlib.sha256(raw).hexdigest()))
            if found is not None:
                break
    if found is None:
        native.visible_pending = index
        state.mode = 'click'
        state.active = index
        native.notice = f'{index}번 실제 코너를 사진에서 클릭하세요. 자동 생성 좌표는 사용하지 않습니다.'
        return False
    corner, source_object, provenance = found
    native.remember(state)
    restored = copy.deepcopy(corner)
    restored.update(visibility='direct_visible', definition_confirmed=True,
        external_occlusion=False, self_occlusion=False, out_of_frame=False, definition_uncertain=False)
    restored.pop('geometry_confirmation', None)
    restored['manual_visibility_reconfirmation'] = dict(
        source='actual_human_visible_button_reusing_manual_click', confirmed_at=utc_now(),
        provenance=provenance, generated_coordinates_used=False)
    record['corners'][index] = restored
    if source_object and source_object.get('presence') == 'present' and source_object.get('target_identity_confirmed'):
        record['object'] = copy.deepcopy(source_object)
    else:
        record['object'].update(presence='present', target_identity_confirmed=True,
            target_object_id='target:' + fid,
            notes='실제 수동점 출처를 재사용하고 사람이 보임 버튼으로 물리 코너 대응을 확인함')
    xy = [restored['x'], restored['y']]
    state.kps_2d[index] = xy
    state.extrap_mask[index] = False
    native.editor._set_keypoint_state(state, index, xy,
        source='manual_click', visibility=2, reason='visible')
    state._pose_key = None
    native.visible_pending = None
    native.action(state, 'human_visible_button_reused_actual_manual_click')
    native.notice = f'{index}번 기존 직접 클릭을 보임으로 복원했습니다.'
    native.persist_recovery(state)
    return True
