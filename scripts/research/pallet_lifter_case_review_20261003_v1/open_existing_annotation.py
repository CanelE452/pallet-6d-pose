"""Frozen lifter review with manual clicks and optional manual-point geometry help.

Process-local adapter: shared editor, original images, plan and references are preserved.
The same review validator records user actions, masks, IDs and real review times.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / 'data/pallet/results' / HERE.name
REVIEW = OUT / 'review'
sys.path.insert(0, str(HERE / 'review'))
from serve import Context, ValidationError, validate_reviewer, utc_now


def recorded_camera(session_id):
    """Read only frozen camera metadata; never read evaluated predictions."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.research.pallet_n3_completion_v3.lifter import PREFIX, EXPECTED_FILES
    from scripts.research.pallet_n3_completion_v3.lifter_run import camera_contract
    path = ROOT/'extracted/depth_cam/rec'/(PREFIX+session_id+'_meta.json')
    size, digest = EXPECTED_FILES[session_id]['meta']
    if path.stat().st_size != size or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValidationError('Recorded camera metadata hash mismatch')
    camera = camera_contract(json.loads(path.read_text(encoding='utf-8-sig')))
    camera['metadata_sha256'] = digest
    return camera


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def jsonable(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {k:jsonable(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [jsonable(v) for v in value]
    return value


def initial_window_size():
    """Leave space for desktop chrome and the editor's trackbar on small screens."""
    width,height = 1320,940
    try:
        result = subprocess.run(['xprop','-root','_NET_WORKAREA','_NET_CURRENT_DESKTOP'],
            capture_output=True,text=True,timeout=2,check=True)
        area = re.search(r'_NET_WORKAREA\([^)]*\)\s*=\s*([^\n]+)',result.stdout)
        desktop = re.search(r'_NET_CURRENT_DESKTOP\([^)]*\)\s*=\s*(\d+)',result.stdout)
        values = [int(value.strip()) for value in area.group(1).split(',')]
        index = int(desktop.group(1)) if desktop else 0
        _,_,screen_width,screen_height = values[index*4:index*4+4]
        if screen_width > 0 and screen_height > 0:
            width = min(width,max(200,screen_width-100))
            height = min(height,max(200,screen_height-140))
    except (OSError,subprocess.SubprocessError,ValueError,AttributeError):
        pass
    return width,height


def choose_profile(path, change=False, preview=None, known_history=None, history_sources=None):
    """Confirm provenance once, at save time, while showing the reviewed image."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0,str(ROOT))
    from scripts.annotate.korean_tk import install
    install()
    import tkinter as tk
    from tkinter import messagebox
    previous = json.loads(path.read_text()) if path.exists() else None
    if previous and not change:
        validate_reviewer(previous['reviewer'])
        return previous['reviewer']
    known_history = dict(known_history or {})
    allowed = ('machine_assistance', 'previous_prediction_exposure', 'previous_annotation_exposure')
    if any(key not in allowed or type(value) is not bool for key,value in known_history.items()):
        raise ValidationError('Invalid recorded history flags')
    prediction_only = set(allowed)-set(known_history) == {'previous_prediction_exposure'}
    window = tk.Tk()
    window.title('저장 전에 모델 예측을 본 적 있는지만 확인' if prediction_only else '저장할 주석의 작성 이력')
    window.geometry('680x555' if preview is not None else '660x330')
    window.attributes('-topmost', True)
    frame = tk.Frame(window, padx=22, pady=18)
    frame.pack(fill='both', expand=True)
    if preview is not None:
        from PIL import Image, ImageTk
        photo = Image.fromarray(preview[:, :, ::-1])
        photo.thumbnail((600, 200))
        window._review_photo = ImageTk.PhotoImage(photo)
        tk.Label(frame, image=window._review_photo).pack(pady=(0, 8))
    tk.Label(frame, text='이 검수 이미지의 모델 예측을 전에 보셨나요?' if prediction_only else '저장할 주석의 작성 이력', font=('', 15, 'bold')).pack(anchor='w')
    tk.Label(frame, text=('평가할 Base/N3 모델이 찍은 코너나 선택 박스를 본 적이 있는지 묻습니다.\n본인 클릭으로 만든 PnP 자동 채움 선은 이 질문에 포함하지 않습니다.' if prediction_only else
             '해당하는 이력만 표시하고 저장하세요. 최초 저장 때 한 번 확인합니다.'),
             justify='left', wraplength=610).pack(anchor='w', pady=8)
    alias = previous['reviewer']['id'] if previous else '검수자-' + uuid.uuid4().hex[:8]
    changed_person = tk.BooleanVar(value=False)
    if previous:
        tk.Checkbutton(frame, text='지난 검수자와 다른 사람이 작업함 (새 자동 별칭)',
                       variable=changed_person).pack(anchor='w')
    if known_history:
        tk.Label(frame,text='PnP·가림 제안 사용과 기존 입력점 재사용은 작업 기록에서 자동 저장합니다.',
                 justify='left',wraplength=610).pack(anchor='w',pady=(8,4))
    fields = [('machine_assistance', '이 코너 판정에 기계 도움을 받음'),
              ('previous_prediction_exposure', '이 표본의 모델 예측을 이미 본 적 있음'),
              ('previous_annotation_exposure', '이 표본의 이전 수동 주석을 본 적 있음')]
    flags = {}
    for key, label in fields:
        if key in known_history or prediction_only:
            continue
        flags[key] = tk.BooleanVar(value=bool(previous and previous['reviewer'][key]))
        tk.Checkbutton(frame, text=label, variable=flags[key]).pack(anchor='w')
    tk.Label(frame, text='코너의 보임·가림 상태는 이 창에서 바꾸지 않습니다.' if prediction_only else
             '표시하지 않은 항목은 해당 이력이 없다는 답변으로 저장됩니다.', justify='left').pack(anchor='w', pady=9)
    result = []
    def accept(prediction_answer=None):
        values = {**known_history, **{key: value.get() for key, value in flags.items()}}
        if prediction_only:
            if type(prediction_answer) is not bool:
                return
            values['previous_prediction_exposure'] = prediction_answer
        reviewer = dict(id='검수자-' + uuid.uuid4().hex[:8] if changed_person.get() else alias,
                        entered_by='human', confirmation=True,
                        exposure_notes='명시적 저장 때 실제 사람의 이력 답변과 실제 도구 사용·기존 입력 표시 기록을 분리 저장함',
                        **values)
        try:
            validate_reviewer(reviewer)
        except ValidationError as error:
            messagebox.showerror('검수 이력 확인', str(error), parent=window)
            return
        write_json(path, dict(reviewer=reviewer, confirmed_at=utc_now(), source='human_first_save_confirmation',
                             recorded_history_flags=known_history, recorded_history_sources=history_sources or {},
                             id_origin='existing_confirmed_alias' if previous and not changed_person.get()
                             else 'generated_local_alias_explicitly_confirmed'))
        result.append(reviewer)
        window.destroy()
    if prediction_only:
        tk.Button(frame,text='모델 예측을 본 적 없음 · 저장',command=lambda:accept(False),
                  bg='#176b45',fg='white',padx=15,pady=8).pack(fill='x',pady=3)
        tk.Button(frame,text='모델 예측을 본 적 있음 · 저장',command=lambda:accept(True),
                  bg='#315371',fg='white',padx=15,pady=8).pack(fill='x',pady=3)
        tk.Button(frame,text='기억나지 않음 · 초안 유지',command=window.destroy,
                  padx=15,pady=6).pack(fill='x',pady=3)
    else:
        tk.Button(frame, text='확인하고 저장 · 다음부터 기억', command=accept,
                  bg='#176b45', fg='white', padx=15, pady=8).pack(fill='x')
    # Pillow-rendered Korean controls can be taller than the Tk text metrics.
    # Size after packing so all history answers and the confirmation fit.
    window.update_idletasks()
    fit_width=min(max(680,window.winfo_reqwidth()+24),max(300,window.winfo_screenwidth()-100))
    fit_height=min(window.winfo_reqheight()+24,max(300,window.winfo_screenheight()-100))
    window.geometry(f'{fit_width}x{fit_height}')
    window.mainloop()
    return result[0] if result else None


class NativeReview:
    def __init__(self, context, reviewer, workspace, passes=('primary', 'repeat'), exclusions_path=None,
                 *, native_pnp=False, batch_plan=None, revisit_saved=False):
        self.ctx, self.reviewer, self.workspace, self.passes = context, reviewer, Path(workspace), passes
        self.native_pnp = native_pnp
        self.editor_state = None
        self.path_map = {}
        self.token = None
        self.record = None
        self.profile_path = self.ctx.store_path.parent / 'NATIVE_REVIEWER_PROFILE.json'
        self.visibility_progress_path = self.ctx.store_path.parent/'NATIVE_VISIBILITY_PROGRESS.json'
        self.change_reviewer = False
        self.frame_started = None
        self.history = []
        self.visibility_checkpoint = None
        self.visibility_checkpoint_history_index = 0
        self.queued_key = None
        self.panel_hits = []
        self.panel_scroll = 0
        self.panel_content_height = 0
        self.footer_height = 80
        self.notice = ''
        self.geometry = dict(proposals={}, projected=[], reason='NEED_FOUR_MANUAL_POINTS')
        self.geometry_signature = None
        self.camera_cache = {}
        self.camera = None
        self.restart_requested = False
        self.recovery_path = self.ctx.store_path.parent/'VIEWER_DRAFT_RECOVERY.json'
        self.recovery = {}
        self.pan_drag = None
        self.visible_pending = None
        self.loaded_existing_manual = False
        self.focus_position_hint = None
        if self.recovery_path.is_file():
            try:
                saved = json.loads(self.recovery_path.read_text())
                if (saved.get('schema') == 'lifter_native_viewer_recovery_v1'
                        and saved.get('bindings') == self.ctx.bindings
                        and saved.get('evaluation_use') is False):
                    self.recovery = saved
            except (OSError, ValueError, TypeError, AttributeError):
                pass
        self.exclusions_path = Path(exclusions_path) if exclusions_path else self.ctx.store_path.parent / 'USER_EXCLUSIONS.json'
        self.exclusions = self.read_exclusions()
        self.batch_path = Path(batch_plan).resolve() if batch_plan else None
        self.batch = None
        self.revisit_saved = revisit_saved
        if self.batch_path:
            batch = json.loads(self.batch_path.read_text())
            if not isinstance(batch, dict):
                raise ValidationError('Partial review queue must be an object')
            ids = batch.get('frame_ids')
            if (batch.get('schema') != 'lifter_human_review_batch_v1'
                    or batch.get('bindings') != self.ctx.bindings
                    or batch.get('source_kind') != 'partial_human_review_queue'
                    or batch.get('frozen_plan_modified') is not False
                    or not isinstance(ids, list) or not ids
                    or any(not isinstance(fid, str) for fid in ids)
                    or len(ids) != len(set(ids))
                    or any(fid not in self.ctx.frames or fid in self.exclusions for fid in ids)
                    or batch.get('repeat_frame_ids') != []):
                raise ValidationError('Partial review queue binding/IDs mismatch; originals preserved')
            self.batch = batch

    @property
    def focused_remaining_tasks(self):
        return bool(self.batch and self.batch.get('task_scope') ==
                    'remaining_missing_corners_of_existing_batch')

    @property
    def bulk_existing_clicks_enabled(self):
        return bool(self.batch and not self.native_pnp and not self.focused_remaining_tasks
                    and len(self.batch['frame_ids']) == 12)

    def translate_review_key(self, value):
        """Resolve focused UI shortcuts before annotate.py's global key gates."""
        if self.focused_remaining_tasks and not self.native_pnp:
            if value == ord('v'):
                return ord('V')
            if value == ord('S'):
                return ord('s')
        return value

    def batch_counts(self):
        if self.batch is None:
            return None
        ids = set(self.batch['frame_ids'])
        done = {fid for fid in ids if (self.ctx.record_for(fid, 'primary') or {}).get('status')
                in ('reviewed', 'skipped')}
        saved = self.saved_pnp_ids('primary') & ids
        local = self.saved_visibility_ids('primary') & ids
        category_complete = {fid for fid in ids if self.visibility_only_status(fid,'primary')['complete']}
        return dict(primary_required=len(ids), primary_reviewed=len(done),
                    primary_visibility_saved_locally=len(local),
                    primary_visibility_labels_completed=len(category_complete),
                    primary_visibility_labels_remaining=len(ids-category_complete),
                    primary_pnp_saved=len(saved), primary_pnp_remaining=len(ids-saved),
                    full_retained_primary=self.requested_counts()['primary_required'],
                    full_population_completed=False, repeat_deferred=True)

    def saved_visibility_ids(self, review_pass):
        from save_visibility_locally import saved_visibility_ids
        return saved_visibility_ids(self,review_pass)

    def visibility_only_status(self, frame_id, review_pass, current_record=None):
        from visibility_only_status import visibility_only_status
        return visibility_only_status(self,frame_id,review_pass,current_record=current_record)

    def saved_pnp_ids(self, review_pass):
        path = self.ctx.store_path.parent/'NATIVE_PNP_PROGRESS.json'
        if not path.is_file():
            return set()
        progress = json.loads(path.read_text())
        if (progress.get('schema') != 'lifter_native_pnp_progress_v1'
                or progress.get('bindings') != self.ctx.bindings):
            raise ValidationError('Saved PnP progress binding mismatch')
        saved = set()
        for entry in progress.get('records', {}).values():
            if entry.get('review_pass') != review_pass:
                continue
            assisted = Path(entry.get('assisted_file', ''))
            if not assisted.is_file():
                continue
            try:
                doc = json.loads(assisted.read_text())
                objects = doc.get('annotation', {}).get('objects', [])
                pose = np.asarray(objects[0]['pose_transform'], dtype=float) if objects else np.array([])
                valid_pose = pose.shape == (4, 4) and np.isfinite(pose).all()
            except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError):
                continue
            fid = entry.get('frame_id')
            if (doc.get('schema') == 'lifter_native_pnp_assistance_v1'
                    and doc.get('source_kind') == 'human_assisted_annotation'
                    and doc.get('evaluation_use') is False and valid_pose
                    and doc.get('bindings') == self.ctx.bindings
                    and doc.get('frame_id') == fid and doc.get('review_pass') == review_pass
                    and fid in self.ctx.frames
                    and doc.get('image_sha256') == self.ctx.frames[fid]['image_sha256']):
                saved.add(fid)
        return saved

    def read_exclusions(self):
        if not self.exclusions_path.exists():
            return set()
        data = json.loads(self.exclusions_path.read_text())
        if (data.get('schema') != 'lifter_user_exclusions_v1'
                or data.get('status') != 'EXCLUDED_BY_USER'
                or data.get('input_bindings') != self.ctx.bindings):
            raise ValidationError('Excluded-frame sidecar binding/status mismatch; originals preserved')
        ids = data.get('excluded_frame_ids')
        if (not isinstance(ids, list) or any(not isinstance(fid,str) for fid in ids)
                or len(ids) != len(set(ids))
                or any(fid not in self.ctx.frames for fid in ids)):
            raise ValidationError('Excluded-frame sidecar has invalid/duplicate frame IDs')
        return set(ids)

    def requested_counts(self):
        result = dict(excluded_primary=len(self.exclusions),
            excluded_repeat=sum(self.ctx.frames[fid]['repeat_review'] for fid in self.exclusions))
        for review_pass in ('primary', 'repeat'):
            rows = [f for f in self.ctx.frames.values() if f['frame_id'] not in self.exclusions
                    and (review_pass == 'primary' or f['repeat_review'])]
            completed = sum((self.ctx.record_for(f['frame_id'], review_pass) or {}).get('status')
                            in ('reviewed', 'skipped') for f in rows)
            result[review_pass + '_required'] = len(rows)
            result[review_pass + '_completed'] = completed
            result[review_pass + '_remaining'] = len(rows) - completed
        return result

    def contexts(self, manifest, cli_args, registry, repo):
        sessions, result = [], {}
        spec = registry.resolve('plastic_square')
        for review_pass in self.passes:
            rows = [f for f in self.ctx.frames.values() if f['frame_id'] not in self.exclusions
                    and (review_pass == 'primary' or f['repeat_review'])]
            if self.batch is not None:
                allowed = self.batch['frame_ids'] if review_pass == 'primary' else self.batch['repeat_frame_ids']
                positions = {fid: i for i, fid in enumerate(allowed)}
                rows = sorted((f for f in rows if f['frame_id'] in positions),
                              key=lambda f: positions[f['frame_id']])
            # User exclusions apply to this task list, never to the frozen reference plan.
            pending = [f for f in rows if not (self.ctx.record_for(f['frame_id'], review_pass) or {}).get('status') in ('reviewed', 'skipped')]
            if not self.revisit_saved:
                local = self.saved_visibility_ids(review_pass)
                if (self.recovery.get('resume_restored_frame_once') is True
                        and self.recovery.get('review_pass') == review_pass):
                    local.discard(self.recovery.get('frame_id'))
                pending = [f for f in pending if f['frame_id'] not in local]
                if (not self.native_pnp and
                        (self.ctx.store_path.parent/'USER_VISIBILITY_DECLARATIONS_20261005_V1.json').is_file()):
                    # Explicit visibility judgments can be complete while
                    # localization references remain absent. Skip only this
                    # classification task; keep the frozen reference gate.
                    pending = [f for f in pending if not self.visibility_only_status(
                        f['frame_id'],review_pass)['complete']]
            if self.batch is not None and self.native_pnp and not self.revisit_saved:
                saved = self.saved_pnp_ids(review_pass)
                pending = [f for f in pending if f['frame_id'] not in saved]
            if not self.focused_remaining_tasks and self.recovery.get('review_pass') == review_pass:
                saved_id = self.recovery.get('frame_id')
                position = next((i for i,f in enumerate(pending) if f['frame_id'] == saved_id), None)
                if position is not None:
                    pending = pending[position:] + pending[:position]
            if not pending:
                continue
            args = copy.copy(cli_args)
            args.object_type = spec.object_type
            args.population_role = 'DEV'
            args.default_split = 'eval'
            args.capture_session_id = 'LIFTER_FIXED_' + review_pass.upper()
            args.lighting_condition = None
            args.intrinsics_quality = 'UNKNOWN'
            args.intrinsics_source = 'reference auto-projection disabled; manual geometry helper uses separately frozen K'
            matrix = np.eye(3)
            if self.native_pnp:
                camera = recorded_camera(pending[0]['session_id'])
                matrix = np.asarray(camera['K'],np.float64)
                args.intrinsics_quality = 'UNKNOWN'
                args.intrinsics_source = 'SHA-256 verified session metadata; native annotation PnP'
            output = self.workspace / review_pass
            for f in pending:
                path = output / (self.ctx.images[f['frame_id']].stem + '.json')
                self.path_map[str(path.resolve())] = (f['frame_id'], review_pass)
            key = 'review:' + review_pass
            name = ('01_PRIMARY_' if review_pass == 'primary' else '02_REPEAT_') + str(len(pending))
            sessions.append((name, str(output), key))
            result[key] = dict(args=args, metadata={'population_role': 'DEV', 'object_type': spec.object_type},
                geometry_spec=spec, out_dir=str(output), K=matrix, K_source=args.intrinsics_source,
                frame_paths=[str(self.ctx.images[f['frame_id']]) for f in pending], frame_count=len(pending),
                writable=True, workspace_scope=None, display_role='DEV', source_session_dir=str(output),
                refresh_evaluation=False, force_explicit_object_type=True, active_evaluation_member=False)
        return sessions, result

    def reviewer_for_frame(self, frame_id, review_pass, base_reviewer):
        """Record known prior own annotation exposure without revealing old clicks."""
        if base_reviewer is None:
            return None, []
        reviewer = copy.deepcopy(base_reviewer)
        evidence = []
        for previous_pass in ('primary', 'repeat'):
            previous = self.ctx.record_for(frame_id, previous_pass)
            if (previous and previous.get('status') != 'skipped'
                    and previous.get('reviewer', {}).get('id') == reviewer['id']
                    and any(c.get('visibility') is not None for c in previous.get('corners', []))):
                evidence.append(dict(source='existing_same_person_annotation_record', frame_id=frame_id,
                    previous_review_pass=previous_pass, previous_revision=previous.get('revision'),
                    previous_finished_at=previous.get('review_time', {}).get('finished_at'),
                    image_sha256=previous.get('image_sha256'), current_review_pass=review_pass,
                    first_pass_coordinates_not_prefilled=review_pass == 'repeat'))
        if evidence:
            reviewer['previous_annotation_exposure'] = True
            reviewer['exposure_notes'] += '\n동일 프레임의 본인 기존 검수 기록이 있어 이전 주석 노출을 기록함. 반복 검수에는 최초 좌표를 불러오지 않음.'
        return reviewer, evidence

    def load(self, state, path, read_only=False):
        self.editor_state = state
        frame_id, review_pass = self.path_map[str(Path(path).resolve())]
        if frame_id in self.exclusions:
            raise ValidationError('This frame was excluded by the user')
        self.history = []
        self.visibility_checkpoint = None
        self.visibility_checkpoint_history_index = 0
        self.queued_key = None
        self.notice = ''
        self.geometry_signature = None
        self.geometry = dict(proposals={}, projected=[], reason='NEED_FOUR_MANUAL_POINTS')
        self.pan_drag = None
        self.visible_pending = None
        self.loaded_existing_manual = False
        self.focus_position_hint = None
        session_id = self.ctx.frames[frame_id]['session_id']
        self.panel_scroll = 0
        if session_id not in self.camera_cache:
            try:
                self.camera_cache[session_id] = recorded_camera(session_id)
            except (OSError, ValueError, KeyError):
                self.camera_cache[session_id] = None
        self.camera = self.camera_cache[session_id]
        self.frame_started = dict(started_at=utc_now(), start_mono=time.monotonic())
        frame_reviewer, exposure = self.reviewer_for_frame(frame_id, review_pass, self.reviewer)
        started = (self.ctx.start_session(dict(frame_id=frame_id, review_pass=review_pass, reviewer=frame_reviewer))
                   if frame_reviewer is not None else dict(token=None, record=None))
        self.token = started['token']
        from save_visibility_locally import load_visibility_record
        local_record = load_visibility_record(self,frame_id,review_pass) if not self.change_reviewer else None
        self.record = started['record'] or local_record or dict(frame_id=frame_id, review_pass=review_pass, status='draft',
            reviewer=copy.deepcopy(frame_reviewer), object=dict(presence=None, target_object_id=None,
            target_identity_confirmed=False, notes=''), skip_reason='', edit_reason='', interaction_log=[],
            corners=[dict(id=i, visibility=None, external_occlusion=False, self_occlusion=False,
                out_of_frame=False, definition_uncertain=False, definition_confirmed=False,
                x=None, y=None, reason='') for i in range(8)])
        self.record['reviewer'] = copy.deepcopy(frame_reviewer)
        self.record['actual_annotation_exposure_provenance'] = exposure
        recovery = self.recovery.get('drafts', {}).get(frame_id+'|'+review_pass, self.recovery)
        valid = False
        if (recovery.get('frame_id') == frame_id and recovery.get('review_pass') == review_pass
                and not self.change_reviewer
                and recovery.get('base_revision') == (started['record'] or {}).get('revision')):
            candidate = recovery.get('record', {})
            corners = candidate.get('corners', [])
            w,h = self.ctx.frames[frame_id]['width'], self.ctx.frames[frame_id]['height']
            valid = (candidate.get('frame_id') == frame_id and candidate.get('review_pass') == review_pass
                     and len(corners) == 8 and {c.get('id') for c in corners} == set(range(8)))
            old_reviewer = candidate.get('reviewer')
            if old_reviewer is not None and frame_reviewer is not None:
                valid = valid and old_reviewer.get('id') == frame_reviewer.get('id')
            elif old_reviewer is not None and frame_reviewer is None:
                valid = False
            for c in corners:
                if c.get('visibility') == 'direct_visible':
                    xy = [c.get('x'),c.get('y')]
                    valid = valid and all(isinstance(v,(int,float)) and np.isfinite(v) for v in xy)
                    valid = valid and 0 <= xy[0] < w and 0 <= xy[1] < h
                else:
                    valid = valid and c.get('visibility') in (None,'not_direct_visible','uncertain')
                    valid = valid and c.get('x') is None and c.get('y') is None
            if valid:
                self.record = copy.deepcopy(candidate)
                self.record['reviewer'] = copy.deepcopy(frame_reviewer)
                self.notice = '저장한 초안에서 이어갑니다.'
        self.loaded_existing_manual = (valid or started['record'] is not None or local_record is not None) and any(
            c.get('visibility') == 'direct_visible' for c in self.record['corners'])
        if self.record['status'] == 'skipped':
            raise ValidationError('Skipped records are not editable through the minimal adapter')
        state.kps_2d = [None] * 9
        state.keypoint_annotations = None
        state.extrap_mask = [False] * 9
        for c in self.record['corners']:
            if c['visibility'] == 'direct_visible':
                state.kps_2d[c['id']] = [c['x'], c['y']]
                self.editor._set_keypoint_state(state, c['id'], [c['x'], c['y']],
                    source='manual_click', visibility=2, reason='visible')
        state.pose = None
        # Show the entire raw image larger without changing coordinates or masks.
        h,w=state.img.shape[:2]
        canvas_w=w+self.editor.MARGIN_L+self.editor.MARGIN_R
        canvas_h=h+self.editor.MARGIN_T+self.editor.MARGIN_B
        state.zoom=max(1.,min(1.6,canvas_w/(w+20),max(1,canvas_h-134)/(h+20)))
        state.pan=[max(0,int((canvas_w-canvas_w/state.zoom)/2)),
                   max(0,int(self.editor.MARGIN_T+h/2-(52+canvas_h-82)/2/state.zoom))]
        state.active = next((c['id'] for c in self.record['corners'] if c['visibility'] is None), 0)
        if (not self.change_reviewer and recovery.get('frame_id') == frame_id
                and recovery.get('review_pass') == review_pass):
            view = recovery.get('view', {})
            zoom,pan = view.get('zoom'),view.get('pan')
            if isinstance(zoom,(int,float)) and np.isfinite(zoom) and 1 <= zoom <= 4:
                state.zoom = float(zoom)
            if isinstance(pan,list) and len(pan)==2 and all(isinstance(v,(int,float)) and np.isfinite(v) for v in pan):
                state.pan = [int(round(v)) for v in pan]
            if type(view.get('active')) is int and view['active'] in range(9 if self.native_pnp else 8):
                state.active = view['active']
            native_state = recovery.get('native_state')
            if self.native_pnp and valid and isinstance(native_state,dict):
                kps,entries,mask = (native_state.get(k) for k in ('kps_2d','keypoint_annotations','extrap_mask'))
                legal = isinstance(kps,list) and len(kps)==9 and isinstance(entries,list) and len(entries)==9 and isinstance(mask,list) and len(mask)==9
                if legal:
                    for i,p in enumerate(kps):
                        if p is not None:
                            legal = legal and isinstance(p,list) and len(p)==2 and all(isinstance(v,(int,float)) and np.isfinite(v) for v in p)
                        if entries[i].get('source') == 'manual_click' and i < 8:
                            c = self.record['corners'][i]
                            legal = legal and c['visibility']=='direct_visible' and p == [c['x'],c['y']]
                if legal:
                    state.kps_2d,state.keypoint_annotations,state.extrap_mask = copy.deepcopy(kps),copy.deepcopy(entries),copy.deepcopy(mask)
                    locked = native_state.get('locked_pose')
                    if isinstance(locked,dict):
                        rotation,translation = np.asarray(locked.get('R')),np.asarray(locked.get('t'))
                        if rotation.shape==(3,3) and translation.size==3 and np.isfinite(rotation).all() and np.isfinite(translation).all():
                            state.locked_pose = copy.deepcopy(locked)
                            state.locked_pose['R'],state.locked_pose['t'] = rotation.astype(float),translation.astype(float)
                            state.mode = 'manip' if native_state.get('mode')=='manip' else 'click'
                    state._pose_key = None
        if self.focused_remaining_tasks and not self.native_pnp:
            # Select the actual unentered field, even when an older view stored 0.
            missing = [c['id'] for c in self.record['corners'] if c['visibility'] is None]
            if missing:
                state.active = missing[0]
                self.notice = f'남은 {len(missing)}점만 입력하세요. 지금 {state.active}번입니다.'
                hint = self.batch.get('position_hints',{}).get(frame_id)
                if hint and hint.get('corner_id') == state.active:
                    source = Path(hint['source_pnp'])
                    raw = source.read_bytes()
                    document = json.loads(raw)
                    if (hashlib.sha256(raw).hexdigest() != hint.get('source_pnp_sha256')
                            or document.get('bindings') != self.ctx.bindings
                            or document.get('frame_id') != frame_id
                            or document.get('review_pass') != review_pass
                            or document.get('image_sha256') != self.ctx.frames[frame_id]['image_sha256']
                            or document['editor_keypoint_annotations'][state.active].get('source') != 'pnp_projected'
                            or document['editor_kps_2d'][state.active] != hint.get('xy')):
                        raise ValidationError('Own PnP position hint binding mismatch')
                    self.focus_position_hint = copy.deepcopy(hint)
        self.refresh_geometry()
        if not self.native_pnp:
            self.visibility_checkpoint = self.snapshot(state)
        if (self.recovery.get('resume_restored_frame_once') is True
                and self.recovery.get('frame_id') == frame_id
                and self.recovery.get('review_pass') == review_pass):
            self.recovery.pop('resume_restored_frame_once')
            write_json(self.recovery_path,self.recovery)
        return valid or started['record'] is not None or local_record is not None

    def persist_recovery(self, state):
        """Keep actual user input even before the one-time provenance confirmation."""
        if self.record is None:
            return
        entry = dict(frame_id=self.record['frame_id'],
            review_pass=self.record['review_pass'], base_revision=self.record.get('revision'),
            record=copy.deepcopy(self.record), view=dict(active=int(state.active),
                zoom=float(state.zoom), pan=[int(v) for v in state.pan]))
        if self.native_pnp:
            entry['native_state'] = jsonable(dict(kps_2d=state.kps_2d,
                keypoint_annotations=state.keypoint_annotations, extrap_mask=state.extrap_mask,
                mode=state.mode, locked_pose=state.locked_pose))
        key = self.record['frame_id']+'|'+self.record['review_pass']
        if (self.recovery.get('drafts', {}).get(key) == entry
                and self.recovery.get('frame_id') == self.record['frame_id']
                and self.recovery.get('review_pass') == self.record['review_pass']):
            return
        drafts = copy.deepcopy(self.recovery.get('drafts', {}))
        drafts[key] = entry
        value = dict(schema='lifter_native_viewer_recovery_v1',
            source_kind='automatic_local_draft_recovery', evaluation_use=False,
            bindings=copy.deepcopy(self.ctx.bindings), frame_id=self.record['frame_id'],
            review_pass=self.record['review_pass'], drafts=drafts, saved_at=utc_now())
        write_json(self.recovery_path, value)
        self.recovery = value

    def refresh_geometry(self):
        if self.record is None:
            return
        signature = json.dumps(self.record['corners'], sort_keys=True)
        if signature == self.geometry_signature:
            return
        self.geometry_signature = signature
        if self.camera is None:
            self.geometry = dict(proposals={}, projected=[], reason='CAMERA_NOT_VERIFIED')
            return
        from manual_geometry_proposals import proposal_for_manual_record
        frame = self.ctx.frames[self.record['frame_id']]
        self.geometry = proposal_for_manual_record(self.record, self.ctx.contract,
            np.asarray(self.camera['K'],np.float64), (frame['width'],frame['height']))

    def confirm_geometry(self, state):
        self.refresh_geometry()
        proposals = {int(i):p for i,p in self.geometry.get('proposals', {}).items()
                     if int(i) in range(8) and p.get('axis') in ('self_occlusion','out_of_frame')
                     and self.record['corners'][int(i)]['visibility'] is None}
        if not proposals:
            self.notice = ('실제 보이는 코너 4점 이상을 먼저 찍으세요.'
                if sum(c['visibility']=='direct_visible' for c in self.record['corners']) < 4
                else '확실한 기하 제안이 없습니다. 보이는 점만 추가하거나 보류하세요.')
            return
        self.remember(state)
        for i,p in proposals.items():
            self.mark(state, i, p['axis'], remember=False)
            self.record['corners'][i]['geometry_confirmation'] = dict(
                source='human_confirmed_manual_click_geometry', confirmed_at=utc_now(),
                proposal=copy.deepcopy(p), input_action='explicit_C_or_geometry_button',
                guessed_coordinate_saved_as_reference=False)
        self.record.setdefault('actual_geometry_assistance', []).append(dict(
            confirmed_at=utc_now(), corner_ids=sorted(proposals),
            helper_sha256=self.geometry.get('helper_sha256'),
            camera_metadata_sha256=self.camera.get('metadata_sha256'),
            evaluated_model_predictions_used=False, reference_coordinates_generated=False))
        self.action(state, 'human_confirmed_manual_geometry_proposals')
        self.notice = f'제안 {len(proposals)}개 확인. 보이는 나머지 점만 찍으세요.'
        self.advance_corner(state)
        self.maybe_autosave()

    def action(self, state, name):
        self.record.setdefault('interaction_log', []).append(dict(action=name, performed_at=utc_now()))
        self.editor._mark_annotation_dirty(state)

    def snapshot(self, state):
        value = dict(record=copy.deepcopy(self.record), kps_2d=copy.deepcopy(state.kps_2d),
                    keypoint_annotations=copy.deepcopy(state.keypoint_annotations),
                    extrap_mask=copy.deepcopy(state.extrap_mask), active=state.active)
        if self.native_pnp:
            value.update(mode=state.mode,locked_pose=copy.deepcopy(state.locked_pose),
                         line_mode=state.line_mode,line_pts=copy.deepcopy(state.line_pts))
        return value

    def remember(self, state):
        self.history.append(self.snapshot(state))

    def reset_visibility(self, state, index=None):
        """Revert state edits while retaining actual manual input geometry."""
        checkpoint = self.visibility_checkpoint
        if (checkpoint is None or checkpoint['record']['frame_id'] != self.record['frame_id']
                or checkpoint['record']['review_pass'] != self.record['review_pass']):
            self.notice = '복구할 이전 상태가 없습니다. 현재 키포인트를 유지합니다.'
            return
        actual = {}
        for source in [checkpoint,*self.history[self.visibility_checkpoint_history_index:],self.snapshot(state)]:
            record = source['record']
            if record['frame_id'] != self.record['frame_id'] or record['review_pass'] != self.record['review_pass']:
                continue
            for corner in record['corners']:
                if corner['visibility'] == 'direct_visible' and corner['definition_confirmed'] is True:
                    actual[corner['id']] = copy.deepcopy(corner)
        self.remember(state)
        selected = range(8) if index is None else [index]
        for i in selected:
            corner = copy.deepcopy(checkpoint['record']['corners'][i])
            if i in actual:
                corner = actual[i]
            self.record['corners'][i] = corner
            if corner['visibility'] == 'direct_visible':
                point = [corner['x'],corner['y']]
                state.kps_2d[i] = point
                state.extrap_mask[i] = False
                self.editor._set_keypoint_state(state,i,point,source='manual_click',visibility=2,reason='visible')
            else:
                state.kps_2d[i] = None
                state.extrap_mask[i] = False
                self.editor._clear_keypoint_state(state,i)
        if not any(c['visibility']=='direct_visible' for c in self.record['corners']):
            self.record['object'] = copy.deepcopy(checkpoint['record']['object'])
        state.pose = None
        self.visible_pending = self.queued_key = None
        self.geometry_signature = None
        self.action(state,'human_reset_visibility_only_preserving_actual_manual_points' if index is None
                    else 'human_cancel_selected_visibility_preserving_actual_manual_point')
        self.notice = '키포인트 유지 · 상태 변경만 복구했습니다.'
        self.persist_recovery(state)

    def undo(self, state):
        self.visible_pending = None
        if not self.history:
            return
        previous = self.history.pop()
        self.record = previous.pop('record')
        for name, value in previous.items():
            setattr(state, name, value)
        state.pose = None
        self.queued_key = None
        self.action(state, 'human_undo_last_click_or_visibility_action')

    def maybe_autosave(self):
        if self.editor_state is not None:
            self.persist_recovery(self.editor_state)
        if self.record and all(c['visibility'] is not None for c in self.record['corners']):
            self.queued_key = None
            self.notice = ('이 이미지 입력 완료. S 저장·다음을 누르세요.' if self.focused_remaining_tasks else
                           '코너 상태 초안 저장 완료. S 현재 저장 / B 기존 입력 일괄 저장')

    def choose_history_profile(self, path, change=False, preview=None):
        known, sources = {}, {}
        if self.record:
            assistance = [key for key in ('native_pnp_assistance', 'native_pnp_exposure',
                'actual_geometry_assistance', 'actual_geometry_exposure') if self.record.get(key)]
            if assistance:
                known['machine_assistance'] = True
                sources['machine_assistance'] = dict(source='actual_viewer_assistance_records',
                    frame_id=self.record['frame_id'], record_fields=assistance)
            if self.loaded_existing_manual or self.record.get('actual_annotation_exposure_provenance'):
                known['previous_annotation_exposure'] = True
                sources['previous_annotation_exposure'] = dict(source='existing_manual_clicks_displayed',
                    frame_id=self.record['frame_id'], recovery_path=str(self.recovery_path))
        return choose_profile(path, change=change, preview=preview,
                              known_history=known, history_sources=sources)

    def advance_corner(self, state):
        self.refresh_geometry()
        proposals = self.geometry.get('proposals', {})
        pending = [c['id'] for c in self.record['corners'] if c['visibility'] is None]
        state.active = next((i for i in pending if i not in proposals), next(iter(pending),min(state.active,7)))

    def mark(self, state, index, axis, remember=True):
        self.visible_pending = None
        if remember:
            self.remember(state)
        c = self.record['corners'][index]
        c.pop('geometry_confirmation', None)
        c.update(visibility='uncertain' if axis == 'definition_uncertain' else 'not_direct_visible',
                 definition_confirmed=False, x=None, y=None, reason='')
        for key in ('external_occlusion', 'self_occlusion', 'out_of_frame', 'definition_uncertain'):
            c[key] = key == axis
        state.kps_2d[index] = None
        self.editor._clear_keypoint_state(state, index)
        self.action(state, 'human_' + axis)

    def clear_corner(self, state, index, remember=True):
        self.visible_pending = None
        if remember:
            self.remember(state)
        self.record['corners'][index].update(visibility=None, definition_confirmed=False,
            external_occlusion=False, self_occlusion=False, out_of_frame=False,
            definition_uncertain=False, x=None, y=None, reason='')
        self.record['corners'][index].pop('geometry_confirmation',None)
        state.kps_2d[index] = None
        state.extrap_mask[index] = False
        self.editor._clear_keypoint_state(state, index)

    def save(self, state, path, status):
        if self.read_exclusions() != self.exclusions or self.record['frame_id'] in self.exclusions:
            raise ValidationError('User exclusions changed; reopen review without changing originals')
        missing = [c['id'] for c in self.record['corners'] if c['visibility'] is None]
        if status == 'reviewed' and missing:
            self.editor._toast(state, 'UNREVIEWED: ' + str(missing) + ' | U per point / Shift+U remaining',
                               log='남은 코너 상태를 확인하세요. 좌표를 추측하지 마세요.')
            return False
        if self.reviewer is None:
            if status == 'draft':
                self.persist_recovery(state)
                self.notice = '초안 저장 완료.'
                return True
            if self.record['object']['presence'] is None:
                self.record['object'].update(presence='uncertain',target_identity_confirmed=False,target_object_id=None)
            from save_visibility_locally import save_visibility_locally
            return save_visibility_locally(self,state,path)
        if status == 'reviewed' and self.record['object']['presence'] is None:
            # No direct clicks means no invented target identity/presence confirmation.
            self.record['object'].update(presence='uncertain', target_identity_confirmed=False,
                                         target_object_id=None)
        previous = self.ctx.record_for(self.record['frame_id'], self.record['review_pass'])
        if previous and previous['status'] != 'draft':
            self.record['edit_reason'] = '사람이 이전 프레임을 다시 열어 직접 재검수하고 저장함'
        outgoing = copy.deepcopy(self.record)
        if (self.record.get('actual_geometry_assistance') or self.record.get('actual_geometry_exposure')
                or self.record.get('native_pnp_assistance') or self.record.get('native_pnp_exposure')):
            outgoing['reviewer']['machine_assistance'] = True
            outgoing['reviewer']['exposure_notes'] += ('\n본인 클릭으로 annotation.py PnP/G/F/M 보조를 사용함. 직접 클릭과 생성 좌표는 분리 저장했고 Base/N3 예측은 표시하지 않았음.'
                if self.record.get('native_pnp_assistance') or self.record.get('native_pnp_exposure')
                else '\n본인이 직접 클릭한 점으로 만든 기하 제안을 봄. C 확인 여부는 interaction_log에 기록. Base/N3 예측은 표시하지 않았음.')
            self.ctx.sessions[self.token]['reviewer'] = copy.deepcopy(outgoing['reviewer'])
        outgoing['status'] = status
        self.action(state, 'human_save_' + status)
        outgoing['interaction_log'] = copy.deepcopy(self.record['interaction_log'])
        result = self.ctx.save(dict(token=self.token, record=outgoing))
        self.record = result['record']
        write_json(Path(path), dict(schema='lifter_native_manual_review_v1', record=self.record,
            editor='scripts/annotate/annotate.py', PnP_enabled=False, training_use=False))
        exported = self.ctx.export()
        exported['requested_use'] = dict(counts=self.requested_counts(),
            excluded_frame_ids=sorted(self.exclusions), exclusions_sidecar=str(self.exclusions_path),
            frozen_population_preserved=True)
        if self.batch is not None:
            exported['requested_use']['partial_batch'] = dict(
                path=str(self.batch_path), sha256=hashlib.sha256(self.batch_path.read_bytes()).hexdigest(),
                frame_ids=self.batch['frame_ids'], counts=self.batch_counts(),
                original_population_claimed_complete=False)
        write_json(self.ctx.store_path.parent / 'LIFTER_REFERENCE_REVIEWED.json', exported)
        state.dirty = state.annotation_dirty = state.frame_tags_dirty = False
        state.discard_armed = None
        if not self.native_pnp:
            self.visibility_checkpoint = self.snapshot(state)
            self.visibility_checkpoint_history_index = len(self.history)
        self.persist_recovery(state)
        print('SAVED', self.record['frame_id'], self.record['review_pass'], status, flush=True)
        return True

    def save_pnp_annotation(self, state, K, out_json, out_png, src_png):
        """Use native annotation creation; keep generated geometry out of direct references."""
        if self.camera is None or state.pose is None:
            self.notice = '키포인트 4점 이상을 찍어 PnP를 먼저 맞추세요.'
            return False
        for i,entry in enumerate(self.editor._ensure_keypoint_annotations(state)):
            if entry.get('source') in ('pnp_projected','centroid_auto','extrapolated'):
                state.extrap_mask[i] = True
        annotation = self.editor._make_state_annotation(state,np.asarray(self.camera['K'],np.float64))
        if annotation is None:
            return False
        self.record.setdefault('native_pnp_assistance', []).append(dict(
            saved_at=utc_now(), input_action='native_annotation_save_key',
            generated_reference_coordinates=False, evaluated_predictions_used=False,
            camera_metadata_sha256=self.camera['metadata_sha256']))
        path = Path(out_json).with_name(Path(out_json).stem+'.PNP_ASSISTED.json')
        write_json(path,jsonable(dict(schema='lifter_native_pnp_assistance_v1',
            source_kind='human_assisted_annotation', evaluation_use=False,
            independent_reference=False, bindings=self.ctx.bindings,
            frame_id=self.record['frame_id'],review_pass=self.record['review_pass'],
            image_sha256=self.ctx.frames[self.record['frame_id']]['image_sha256'],
            annotation=annotation, manual_reference_corners=copy.deepcopy(self.record['corners']),
            reference_scope='actual_manual_clicks_only', editor_kps_2d=state.kps_2d,
            editor_keypoint_annotations=state.keypoint_annotations,
            human_provenance_confirmed=self.reviewer is not None,
            projected_points_are_direct_ground_truth=False)))
        progress_path = self.ctx.store_path.parent/'NATIVE_PNP_PROGRESS.json'
        progress = dict(schema='lifter_native_pnp_progress_v1',bindings=self.ctx.bindings,
                        scope='keypoints_and_PnP_saved_separate_from_visibility_review',records={})
        if progress_path.exists():
            previous = json.loads(progress_path.read_text())
            if previous.get('schema') == progress['schema'] and previous.get('bindings') == self.ctx.bindings:
                progress = previous
        progress['records'][self.record['frame_id']+'|'+self.record['review_pass']] = dict(
            frame_id=self.record['frame_id'],review_pass=self.record['review_pass'],
            assisted_file=str(path.resolve()),saved_at=utc_now(),
            directly_clicked_corners=sum(c['visibility']=='direct_visible' for c in self.record['corners']),
            full_visibility_review_complete=all(c['visibility'] is not None for c in self.record['corners']))
        write_json(progress_path,progress)
        self.persist_recovery(state)
        if all(c['visibility'] is not None for c in self.record['corners']):
            if not self.save(state,out_json,'reviewed'):
                return False
        state.dirty = state.annotation_dirty = state.frame_tags_dirty = False
        self.notice = '키포인트·PnP 저장 완료. 가시성 검수는 별도입니다.'
        print('SAVED_NATIVE_PNP',self.record['frame_id'],self.record['review_pass'],str(path),flush=True)
        return True

    def install(self, editor):
        if str(ROOT) not in sys.path:
            sys.path.insert(0,str(ROOT))
        from scripts.annotate.korean_tk import install
        install()
        self.editor = editor
        original_mouse, original_key, original_render = editor.on_mouse, editor._handle_click_key, editor.render
        original_pose,original_manip = editor.update_pose,editor._handle_manip_key
        original_wait = editor.cv2.waitKey
        from annotate_pnp import make_pallet_keypoints_3d
        assert np.allclose(make_pallet_keypoints_3d(1.1, 1.1, .15)[:8], [c['xyz_m'] for c in self.ctx.contract['corners']])
        import annotate_draw
        from PIL import Image, ImageDraw, ImageFont
        font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
        font = ImageFont.truetype(font_path, 15)
        large = ImageFont.truetype(font_path, 19)
        def panel(h, active_idx, kps_2d, pose, frame_idx, total, zoom, dirty, **kwargs):
            # Lay out the complete sidebar first, then make a viewport.  Image
            # dimensions must not truncate controls or create invisible hitboxes.
            image = Image.new('RGB', (annotate_draw.PANEL_W, max(h, 1600)), (24, 32, 42))
            draw = ImageDraw.Draw(image)
            font = ImageFont.truetype(font_path, 14)
            large = ImageFont.truetype(font_path, 18)
            line_height = 20
            row_height = 32
            self.panel_hits = []
            y = 12
            def put(text, color='#dae9ed', big=False):
                nonlocal y
                selected_font = large if big else font
                for paragraph in text.split('\n'):
                    line = ''
                    for char in paragraph:
                        if line and draw.textlength(line+char, font=selected_font) > annotate_draw.PANEL_W-20:
                            draw.text((10,y),line,font=selected_font,fill=color)
                            y += 26 if big else line_height
                            line = ''
                        line += char
                    draw.text((10,y),line,font=selected_font,fill=color)
                    y += 26 if big else line_height
            def button(rect, text, key, selected=False):
                draw.rounded_rectangle(rect, radius=5, fill='#176b45' if selected else '#364359')
                button_font = font
                if draw.textlength(text,font=button_font) > rect[2]-rect[0]-12:
                    button_font = ImageFont.truetype(font_path, 13)
                draw.text((rect[0]+6, rect[1]+3), text, font=button_font, fill='white')
                self.panel_hits.append((rect, key))
            put('키포인트 → PnP' if self.native_pnp else
                '남은 코너 1점 확인' if self.focused_remaining_tasks else
                '리프터 · 입력점 재사용' if self.batch is not None else '리프터 · 보이는 코너', '#9ee7c0', True)
            manipulating = bool(self.editor_state is not None and self.editor_state.mode=='manip')
            put(f'{frame_idx+1} / {total}장 · 확대 {zoom:.1f}배')
            if self.batch is not None:
                put(f'{len(self.batch["frame_ids"])}장 · 각 이미지의 남은 점만' if self.focused_remaining_tasks else
                    f'{len(self.batch["frame_ids"])}장 부분 검수 · 반복 검수 보류', '#ffd27d')
            put('대상: 사진 속 초록색 팔레트')
            names = ['기준면 왼위','기준면 오른위','기준면 오른아래','기준면 왼아래',
                     '반대면 왼위','반대면 오른위','반대면 오른아래','반대면 왼아래']
            put(f'지금: {active_idx}번 · '+('중심점' if active_idx==8 else names[min(active_idx,7)]), '#ffd17e')
            if self.record and active_idx < 8:
                c = self.record['corners'][active_idx]
                selected_status = ('보임 · 직접 클릭됨' if c['visibility']=='direct_visible' else
                    '판단 불가' if c['visibility']=='uncertain' else '외부 가림' if c['external_occlusion'] else
                    '자체 가림' if c['self_occlusion'] else '화면 밖' if c['out_of_frame'] else '아직 미입력')
                put('보임 선택됨 → 사진에서 점 클릭' if self.visible_pending == active_idx else
                    f'현재 상태: {selected_status}', '#9ee7c0')
            if self.native_pnp:
                button((10,y,91,y+28), 'S 저장' if manipulating else 'G 채움', ord('s') if manipulating else ord('g'))
                button((98,y,179,y+28), 'L 회전' if manipulating else 'F 다음', ord('l') if manipulating else ord('f'))
                button((186,y,270,y+28), 'M 복귀' if manipulating else 'M 조작', ord('m'))
                y += row_height
            button((10,y,90,y+28), '+ 확대', ord('+'))
            button((98,y,178,y+28), '− 축소', ord('-'))
            button((186,y,270,y+28), '전체', ord('F') if self.focused_remaining_tasks and not self.native_pnp else ord('v'))
            y += row_height
            if not manipulating:
                button((10,y,270,y+26), 'V 보임 → 사진에서 점 클릭' if self.focused_remaining_tasks else
                       'V 보임 · 실제 점 선택', ord('V'),
                       selected=self.visible_pending == active_idx)
                y += row_height
            button((10,y,135,y+28), 'A 왼쪽 이동' if manipulating else 'E 외부 가림', ord('a') if manipulating else ord('e'))
            button((145,y,270,y+28), 'D 오른쪽 이동' if manipulating else 'I 자체 가림', ord('d') if manipulating else ord('i'))
            y += row_height
            button((10,y,135,y+28), 'Q 가까이 이동' if manipulating else 'O 화면 밖', ord('q') if manipulating else ord('o'))
            button((145,y,270,y+28), 'E 멀리 이동' if manipulating else 'U 판단 불가', ord('e') if manipulating else ord('u'))
            y += row_height
            n = len(self.geometry.get('proposals', {}))
            button((10,y,270,y+30), f'Shift+C 가림 확인 · {n}개' if self.native_pnp else f'C 자동 제안 확인 · {n}개', ord('C') if self.native_pnp else ord('c'),selected=bool(n))
            y += row_height + 2
            bulk = self.bulk_existing_clicks_enabled
            button((10,y,270,y+26), 'S 현재 이미지 저장 / 다음' if bulk else
                   'S 키포인트·PnP 저장 / 다음' if self.native_pnp else 'S 현재 이미지 저장 / 다음', ord('s'))
            y += row_height
            if bulk:
                button((10,y,270,y+26), 'B 12장 기존 입력 제출', ord('B'), selected=True)
                y += row_height
            button((10,y,135,y+28), 'Q 저장 · 닫기', ord('Q') if self.native_pnp else ord('q'))
            button((145,y,270,y+28), 'Shift+T 재열기' if self.native_pnp else 'T 저장 · 재열기', ord('T') if self.native_pnp else ord('t'))
            y += row_height
            button((10,y,93,y+28), 'Z 되돌림', ord('z'))
            button((100,y,182,y+28), 'D 삭제' if self.native_pnp else 'D 상태취소', ord('d'))
            button((189,y,270,y+28), 'R 초기화' if self.native_pnp else 'R 상태복구', ord('r'))
            y += row_height
            if not self.native_pnp:
                put('N/P 다음/이전 · Shift+U 일괄 보류')
            put('도식의 기준면 0–3 / 반대면 4–7', '#ffd17e')
            origin = y + 54
            # Front-facing schematic matching annotate.py's near/far names.
            pts = {0:(28,origin+5),1:(164,origin+5),2:(164,origin+48),3:(28,origin+48),
                   4:(99,origin-48),5:(235,origin-48),6:(235,origin-5),7:(99,origin-5)}
            for c in self.ctx.contract['corners']:
                for other in self.ctx.contract['corners'][c['id']+1:]:
                    if sum(a != b for a,b in zip(c['xyz_m'],other['xyz_m'])) == 1:
                        draw.line([pts[c['id']],pts[other['id']]], fill='#8699aa', width=2)
            draw.line([pts[i] for i in (0,1,2,3,0)], fill='#9ee7c0', width=4)
            draw.text((30,origin+50),'기준면 0–3',font=font,fill='#9ee7c0')
            for i,(x,v) in pts.items():
                draw.ellipse((x-10,v-10,x+10,v+10), fill='#176b45' if i==active_idx else '#364359')
                draw.text((x-5,v-12), str(i), font=font, fill='white')
                self.panel_hits.append(((x-15,v-15,x+15,v+15),ord(str(i))))
            y = origin + 68
            if self.record:
                status_y = y
                for c in self.record['corners']:
                    proposal = self.geometry.get('proposals', {}).get(c['id'])
                    status = ('제안·자체' if proposal and proposal['axis']=='self_occlusion' else '제안·밖') if proposal and c['visibility'] is None else '미입력' if c['visibility'] is None else '직접 클릭' if c['visibility']=='direct_visible' else '판단 불가' if c['visibility']=='uncertain' else '외부 가림' if c['external_occlusion'] else '자체 가림' if c['self_occlusion'] else '화면 밖'
                    x = 10 if c['id'] < 4 else 145
                    row_y = status_y+(c['id']%4)*line_height
                    draw.text((x,row_y),f'{">" if c["id"]==active_idx else " "}{c["id"]} {status}',
                        font=font,fill='#83cfff' if proposal and c['visibility'] is None else '#9ee7c0' if c['visibility'] else '#ffd17e')
                    self.panel_hits.append(((x,row_y,x+125,row_y+line_height),ord(str(c['id']))))
                y = status_y+4*line_height
            if dirty:
                put('저장하지 않은 변경 있음', '#ffb2a0')
            put('초안 자동 보존 · 제출은 저장 버튼')
            self.panel_content_height = y+10
            overflow = self.panel_content_height > h
            viewport_h = max(1,h-24) if overflow else h
            self.panel_scroll = min(max(0,self.panel_scroll),max(0,self.panel_content_height-viewport_h))
            if not overflow:
                self.panel_scroll = 0
            visible = image.crop((0,self.panel_scroll,annotate_draw.PANEL_W,self.panel_scroll+h))
            self.panel_hits = [((x0,y0-self.panel_scroll,x1,y1-self.panel_scroll),value)
                for (x0,y0,x1,y1),value in self.panel_hits
                if 0 <= y0-self.panel_scroll and y1-self.panel_scroll < viewport_h]
            if overflow:
                visible_draw = ImageDraw.Draw(visible)
                visible_draw.rectangle((0,viewport_h,annotate_draw.PANEL_W,h),fill='#16221c')
                visible_draw.text((10,viewport_h+2),'패널 위에서 휠로 위·아래 이동',font=font,fill='#ffd17e')
                thumb_h = max(12,int(viewport_h*viewport_h/self.panel_content_height))
                thumb_y = int(self.panel_scroll*(viewport_h-thumb_h)/max(1,self.panel_content_height-viewport_h))
                visible_draw.rectangle((273,0,279,viewport_h),fill='#1b2837')
                visible_draw.rectangle((273,thumb_y,279,thumb_y+thumb_h),fill='#9ee7c0')
            return np.asarray(visible)[:, :, ::-1].copy()
        annotate_draw.build_panel = panel

        def native_pose(state, K, force=False):
            self.editor_state = state
            if self.native_pnp and self.camera is not None:
                return original_pose(state,np.asarray(self.camera['K'],np.float64),force=force)
            state.pose = None
            state.mode = 'click'
            state.line_mode = False

        def mouse(event, x, y, flags, state):
            raw_x,raw_y = x,y
            x,y = editor._display_to_canvas(x,y,state)
            canvas_w = state.img.shape[1]+editor.MARGIN_L+editor.MARGIN_R if state.img is not None else 0
            if event == editor.cv2.EVENT_MOUSEWHEEL and state.img is not None and x >= canvas_w:
                self.panel_scroll = max(0,self.panel_scroll+(-80 if flags >> 16 > 0 else 80))
                return
            if event == editor.cv2.EVENT_MOUSEWHEEL:
                state.last_mouse = (x,y)
                delta = flags >> 16
                view_zoom(state,1.35 if delta>0 else 1/1.35)
                return
            if event == editor.cv2.EVENT_MBUTTONDOWN:
                self.pan_drag = (x,y,list(state.pan))
                return
            if event == editor.cv2.EVENT_MBUTTONUP:
                self.pan_drag = None
                return
            if event == editor.cv2.EVENT_MOUSEMOVE and self.pan_drag is not None:
                start_x,start_y,start_pan = self.pan_drag
                state.pan = [int(start_pan[0]-(x-start_x)/state.zoom),
                             int(start_pan[1]-(y-start_y)/state.zoom)]
                return
            if state.img is not None:
                if x >= canvas_w:
                    state.last_mouse = (x,y)
                    if event == editor.cv2.EVENT_LBUTTONDOWN:
                        px = x - canvas_w
                        for (x0,y0,x1,y1), value in self.panel_hits:
                            if x0 <= px <= x1 and y0 <= y <= y1:
                                if ord('0') <= value <= ord('7'):
                                    self.visible_pending = None
                                    state.active = value-ord('0')
                                else:
                                    self.queued_key = value
                                break
                    return
                canvas_h = state.img.shape[0]+editor.MARGIN_T+editor.MARGIN_B
                if event in (editor.cv2.EVENT_LBUTTONDOWN,editor.cv2.EVENT_RBUTTONDOWN) and (y < 49 or y >= canvas_h-self.footer_height):
                    return
            state.active = min(state.active, 8 if self.native_pnp else 7)
            idx = state.active
            previous = self.snapshot(state)
            before = copy.deepcopy(state.kps_2d)
            original_mouse(event, raw_x, raw_y, flags, state)
            if self.record is None or state.kps_2d == before:
                return
            p = state.kps_2d[idx]
            if p is not None:
                h, w = state.img.shape[:2]
                if not (0 <= p[0] < w and 0 <= p[1] < h):
                    state.kps_2d = before
                    if before[idx] is None:
                        editor._clear_keypoint_state(state, idx)
                    else:
                        editor._set_keypoint_state(state, idx, before[idx], source='manual_click', visibility=2, reason='visible')
                    state.active = idx
                    editor._toast(state, 'Click inside raw image only; O = outside frame')
                    return
                if idx == 8:
                    self.history.append(previous)
                    self.action(state,'human_native_centroid_click')
                    self.persist_recovery(state)
                    return
                c = self.record['corners'][idx]
                annotation_state = editor._ensure_keypoint_annotations(state)[idx]
                if (self.native_pnp and (annotation_state.get('source') != 'manual_click'
                        or state.extrap_mask[idx])):
                    self.history.append(previous)
                    self.action(state,'human_native_extrapolation_assistance')
                    self.persist_recovery(state)
                    return
                c.pop('geometry_confirmation',None)
                self.visible_pending = None
                self.history.append(previous)
                c.update(visibility='direct_visible', definition_confirmed=True, x=p[0], y=p[1], reason='',
                    external_occlusion=False, self_occlusion=False, out_of_frame=False, definition_uncertain=False)
                # The operator's raw click identifies this frame-local reference target.
                self.record['object'].update(presence='present', target_object_id='target:' + self.record['frame_id'],
                    target_identity_confirmed=True, notes='사람이 대상 팔레트의 실제 물리 코너를 직접 클릭함')
                self.action(state, 'manual_raw_click_and_physical_confirmation')
            else:
                self.history.append(previous)
                self.clear_corner(state,idx,remember=False)
                self.action(state, 'human_clear_click')
            self.advance_corner(state)
            self.maybe_autosave()

        def view_zoom(state, factor):
            old = state.zoom
            state.zoom = max(1.,min(4.,old*factor))
            if state.img is not None:
                h,w = state.img.shape[:2]
                x,y = state.last_mouse or ((w+editor.MARGIN_L+editor.MARGIN_R)/2,
                                          (h+editor.MARGIN_T+editor.MARGIN_B)/2)
                if not 0 <= x < w+editor.MARGIN_L+editor.MARGIN_R:
                    x,y = (w+editor.MARGIN_L+editor.MARGIN_R)/2,(h+editor.MARGIN_T+editor.MARGIN_B)/2
                state.pan = [int(state.pan[0]+x/old-x/state.zoom),
                             int(state.pan[1]+y/old-y/state.zoom)]

        def key(value, state, out_json, out_png, src_png, K):
            try:
                if self.bulk_existing_clicks_enabled and value in (ord('B'), ord('b')):
                    from tkinter import messagebox
                    from approve_existing_clicks import approve_existing_clicks
                    approved = approve_existing_clicks(self, state,
                        confirm=lambda title, message: messagebox.askokcancel(title, message),
                        choose_profile=self.choose_history_profile)
                    if approved:
                        self.notice = '12장 기존 입력 저장 완료. 대상 확인 단계로 진행하세요.'
                        return 'quit'
                    self.notice = '일괄 저장 취소. 기존 입력은 보존했습니다.'
                    return None
                if self.native_pnp and value==ord('8'):
                    state.active = 8
                    return None
                if self.native_pnp and value in map(ord,'gfxctwb'):
                    native_pose(state,K)
                    self.remember(state)
                    answer = original_key(value,state,out_json,out_png,src_png,
                        np.asarray(self.camera['K'],np.float64) if self.camera else K)
                    self.action(state,'human_native_annotation_'+chr(value))
                    self.persist_recovery(state)
                    return answer
                if ord('0') <= value <= ord('7'):
                    self.visible_pending = None
                    state.active = value - ord('0')
                    return None
                if value == ord('V'):
                    from select_visible_corner import select_visible_corner
                    select_visible_corner(self, state)
                    self.persist_recovery(state)
                    return None
                if value == ord('C') or (not self.native_pnp and value==ord('c')):
                    self.confirm_geometry(state)
                    self.persist_recovery(state)
                    return None
                if value in (ord('q'),ord('Q'),27,ord('T')) or (not self.native_pnp and value==ord('t')):
                    self.persist_recovery(state)
                    self.restart_requested = value in (ord('t'),ord('T'))
                    return 'quit'
                if value in (ord('n'),ord('p')):
                    self.persist_recovery(state)
                    state.dirty = state.annotation_dirty = state.frame_tags_dirty = False
                    return original_key(value,state,out_json,out_png,src_png,K)
                if value in (ord('+'),ord('='),ord('-'),ord('_')):
                    view_zoom(state,1.35 if value in (ord('+'),ord('=')) else 1/1.35)
                    return None
                if value == ord('v') or (self.focused_remaining_tasks and value == ord('F')):
                    state.zoom = 1.;state.pan = [0,0]
                    return None
                statuses = {'e': 'external_occlusion', 'i': 'self_occlusion', 'o': 'out_of_frame', 'u': 'definition_uncertain'}
                if value in map(ord, statuses):
                    self.mark(state, min(state.active, 7), statuses[chr(value)])
                    self.advance_corner(state)
                    self.maybe_autosave()
                    return None
                if value == ord('U'):
                    from tkinter import messagebox
                    if messagebox.askyesno('남은 점 직접 확인', '남은 미입력 코너는 모두 대응 또는 직접 가시 여부를 판정할 수 없습니까?\n확인한 경우에만 예를 누르세요.'):
                        self.remember(state)
                        for c in self.record['corners']:
                            if c['visibility'] is None:
                                self.mark(state, c['id'], 'definition_uncertain',remember=False)
                        self.maybe_autosave()
                    return None
                if value in (ord('s'), 13, 10):
                    if self.native_pnp:
                        native_pose(state,K)
                        if state.pose is not None:
                            return original_key(ord('s'),state,out_json,out_png,src_png,
                                np.asarray(self.camera['K'],np.float64))
                    if any(c['visibility'] is None for c in self.record['corners']):
                        self.persist_recovery(state)
                        missing = [c['id'] for c in self.record['corners'] if c['visibility'] is None]
                        self.notice = (f'아직 {missing[0]}번 미입력입니다. 보이면 사진에서 실제 점을 클릭하세요.'
                                       if self.focused_remaining_tasks else
                                       '초안 저장 완료. 보이는 점부터 계속 찍으세요.')
                        if self.focused_remaining_tasks:
                            print('S_NOT_COMPLETE',self.record['frame_id'],missing,flush=True)
                        return None
                    return 'save-next' if self.save(state, out_json, 'reviewed') else None
                if value in (ord('d'), ord('z'), ord('r')):
                    selected = min(state.active, 7)
                    self.queued_key = None
                    if value == ord('z'):
                        self.undo(state)
                    elif not self.native_pnp:
                        self.reset_visibility(state,selected if value == ord('d') else None)
                    elif value == ord('d'):
                        if self.record['corners'][selected]['visibility'] is not None:
                            self.clear_corner(state,selected)
                            self.action(state,'human_delete_corner_or_visibility')
                    elif any(c['visibility'] is not None for c in self.record['corners']):
                        self.remember(state)
                        original_key(value,state,out_json,out_png,src_png,K)
                        for i in range(8):
                            self.clear_corner(state,i,remember=False)
                        self.record['object'].update(presence=None,target_object_id=None,target_identity_confirmed=False)
                        self.action(state,'human_reset_all_clicks_and_visibility')
                    return None
                if value == ord('S'):
                    if self.reviewer is not None:
                        self.save(state, out_json, 'draft')
                    else:
                        self.persist_recovery(state)
                    self.notice = '초안 저장 완료.'
                    return None
                if value == ord('a'):
                    from tkinter import messagebox
                    if messagebox.askyesno('대상 부재 확인', '이 프레임에는 평가 대상 팔레트가 없습니까?'):
                        self.remember(state)
                        self.record['object'].update(presence='absent', target_object_id=None, target_identity_confirmed=False)
                        for c in self.record['corners']:
                            self.mark(state, c['id'], 'definition_uncertain',remember=False)
                            c['reason'] = '사람 확인: 평가 대상 부재'
                        self.maybe_autosave()
                    return None
                # Allow native view/navigation controls only; no projection, extrapolation, reordering or deletion.
                if value in map(ord, 'qnp,+-=_.hjkl'):
                    return original_key(value, state, out_json, out_png, src_png, K)
                return None
            except (ValidationError, OSError) as error:
                editor._toast(state, 'SAVE BLOCKED - see console', log=str(error))
                return None

        def render(state, *args):
            self.refresh_geometry()
            image = original_render(state, *args)
            if self.record:
                if self.native_pnp and state.pose is not None:
                    self.record['native_pnp_exposure'] = dict(source='own_manual_points_native_PnP',
                        machine_assistance=True,evaluated_predictions_displayed=False)
                hint = self.focus_position_hint
                if hint and self.record['corners'][hint['corner_id']]['visibility'] is None:
                    # A hollow guide is display-only; never put it in kps_2d or
                    # manual reference corners, and never synthesize a click.
                    px,py = hint['xy']
                    x = int(round((px+editor.MARGIN_L-state.pan[0])*state.zoom))
                    y = int(round((py+editor.MARGIN_T-state.pan[1])*state.zoom))
                    canvas_w = state.img.shape[1]+editor.MARGIN_L+editor.MARGIN_R
                    if 0 <= x < canvas_w and 49 <= y < image.shape[0]-self.footer_height:
                        editor.cv2.circle(image,(x,y),13,(90,225,255),2)
                        editor.cv2.putText(image,str(hint['corner_id'])+'?',(x+14,y-7),
                            editor.cv2.FONT_HERSHEY_SIMPLEX,.65,(90,225,255),2)
                        exposure = dict(source='same_frame_own_PnP_position_hint',
                            corner_ids=[hint['corner_id']], source_pnp=hint['source_pnp'],
                            source_pnp_sha256=hint['source_pnp_sha256'],
                            machine_assistance=True,evaluated_model_predictions_used=False,
                            reference_coordinates_generated=False)
                        existing = self.record.setdefault('actual_geometry_exposure',[])
                        if not any(all(row.get(k)==v for k,v in exposure.items()) for row in existing):
                            existing.append(dict(shown_at=utc_now(),**exposure))
                proposals = self.geometry.get('proposals', {})
                if proposals:
                    evidence = dict(corner_ids=sorted(proposals), evidence=self.geometry.get('evidence'),
                        camera_metadata_sha256=self.camera.get('metadata_sha256'),
                        evaluated_model_predictions_used=False, reference_coordinates_generated=False)
                    existing = self.record.setdefault('actual_geometry_exposure', [])
                    if not any(all(row.get(k)==v for k,v in evidence.items()) for row in existing):
                        existing.append(dict(shown_at=utc_now(), **evidence))
                    canvas_w = state.img.shape[1]+editor.MARGIN_L+editor.MARGIN_R
                    for i,proposal in proposals.items():
                        px,py = proposal['projected_xy']
                        x = int(round((px+editor.MARGIN_L-state.pan[0])*state.zoom))
                        y = int(round((py+editor.MARGIN_T-state.pan[1])*state.zoom))
                        if 0 <= x < canvas_w and 30 <= y < image.shape[0]-82:
                            editor.cv2.circle(image,(x,y),9,(255,170,80),2)
                            editor.cv2.putText(image,str(i),(x+10,y-6),editor.cv2.FONT_HERSHEY_SIMPLEX,.55,(255,190,100),2)
                # Korean guidance is readable in this Conda/OpenCV environment.
                rgb = Image.fromarray(image[:,:,::-1])
                canvas_w = image.shape[1]-annotate_draw.PANEL_W
                canvas = rgb.crop((0,0,canvas_w,image.shape[0]))
                draw = ImageDraw.Draw(canvas)
                draw.rectangle((0,0,canvas_w-1,48),fill='#172b25')
                title = (f'남은 1점 · 지금 {state.active}번 실제 코너를 클릭' if self.focused_remaining_tasks and not self.native_pnp else
                         '12장 기존 입력점 재사용 · B 일괄 저장' if self.bulk_existing_clicks_enabled else
                         'PnP 자세 조작 중 · M으로 클릭 모드 복귀' if self.native_pnp and state.mode=='manip'
                    else f'사진 속 초록색 팔레트 · 지금 {state.active}번 '+('중심점' if state.active==8 else '실제 코너')+'을 클릭')
                draw.text((12,5),title,font=large,fill='#bcebcf')
                draw.text((12,27),'4점 입력 → PnP → G 자동 채움 / F 저장·다음 / M 자세 조작' if self.native_pnp else
                    '노란 원은 위치 참고입니다. 보이면 실제 점 클릭 → S 저장·다음' if self.focused_remaining_tasks else
                    'B로 기존 12장 입력을 일괄 저장합니다. 미입력점은 판정 보류합니다.' if self.bulk_existing_clicks_enabled else
                    '파란 점은 가림 제안입니다. 맞으면 C, 보이는 점은 직접 클릭하세요.',font=font,fill='#c3deec')
                lines = [self.notice or ('키포인트 4점 이상을 입력하면 기존 annotation.py의 PnP가 표시됩니다.' if self.native_pnp else '보이는 코너를 클릭하면 다음 번호가 자동으로 선택됩니다.'),
                    '휠: 확대 · G/F: PnP 자동 채움 · M: 자세 조작 · t: 선 교점 · c: 중심점' if self.native_pnp else '휠: 확대/축소 · 휠 누르고 끌기: 이동 · C: 가림 제안 확인',
                    'S: 저장 · Shift+T: 저장·재열기 · Shift+C: 가림 확인 · Z: 되돌리기' if self.native_pnp else 'S: 저장 · Q: 저장 후 닫기 · T: 저장 후 다시 열기 · Z: 되돌리기']
                if self.bulk_existing_clicks_enabled:
                    lines = [self.notice or '기존 입력을 그대로 씁니다. 틀린 상태만 바꾸세요.',
                             '보임: V 버튼 → 사진에서 해당 점 클릭 · 자체 가림: I 버튼',
                             '상태 선택은 초안 보존만 합니다. S 현재 저장 / B 12장 입력 제출']
                elif self.focused_remaining_tasks and not self.native_pnp:
                    lines = [self.notice or '선택된 한 점만 확인하세요. 기존 입력은 보존됩니다.',
                             '보이면 사진에서 실제 점 클릭 · 안 보이면 오른쪽 상태 선택',
                             'S 저장·다음 · Q 저장·닫기 · Z 되돌림 · 노란 원: PnP 위치 참고']
                if self.native_pnp and state.mode=='manip':
                    lines[0] = 'M 자세 조작: A/D 좌우 · W/X 상하 · Q/E 앞뒤'
                    lines[1] = 'J/L 회전 · I/K 기울기 · U/O 롤 · 1/2 이동 간격 · 3/4 회전 간격'
                    lines[2] = 'S 자세 저장·다음 · M 클릭 모드 복귀 · Shift+T 저장·재열기'
                wrapped = []
                for text in lines:
                    line = ''
                    for char in text:
                        if line and draw.textlength(line+char,font=font) > canvas_w-24:
                            wrapped.append(line)
                            line = ''
                        line += char
                    wrapped.append(line)
                self.footer_height = max(80,len(wrapped)*22+10)
                footer_top = image.shape[0]-self.footer_height
                draw.rectangle((0,footer_top,canvas_w-1,image.shape[0]),fill='#16221c')
                for i,line in enumerate(wrapped):
                    draw.text((12,footer_top+4+i*22),line,font=font,fill='#bcebcf')
                rgb.paste(canvas,(0,0))
                image = np.asarray(rgb)[:,:,::-1].copy()
                self.persist_recovery(state)
            return image
        editor.load_existing_annotation = self.load
        editor.update_pose = native_pose
        editor.on_mouse = mouse
        editor._handle_click_key = key
        editor.render = render
        if self.native_pnp:
            editor._save_state_annotation = self.save_pnp_annotation
            def native_manip(value,state,out_json,out_png,src_png,K):
                if value in (ord('Q'),ord('T'),ord('C'),ord('z'),27):
                    return key(value,state,out_json,out_png,src_png,K)
                self.remember(state)
                result = original_manip(value,state,out_json,out_png,src_png,
                    np.asarray(self.camera['K'],np.float64) if self.camera else K)
                self.action(state,'human_native_manipulate_'+str(value))
                self.persist_recovery(state)
                return result
            editor._handle_manip_key = native_manip
        def wait_key(delay=0):
            if self.queued_key is not None:
                value, self.queued_key = self.queued_key, None
            else:
                value = original_wait(delay)
            return self.translate_review_key(value)
        editor.cv2.waitKey = wait_key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--change-reviewer', action='store_true')
    parser.add_argument('--pass', dest='review_pass', choices=['all', 'primary', 'repeat'], default='all')
    parser.add_argument('--visibility-only', action='store_true',help='키포인트·PnP 입력 뒤 기존 가시성 상태만 검수')
    parser.add_argument('--batch-plan', type=Path, help='원본 계획을 보존하는 부분 검수 목록')
    parser.add_argument('--revisit-saved', action='store_true', help='부분 목록의 이미 저장한 PnP도 다시 열기')
    args = parser.parse_args()
    context = Context(REVIEW/'MANIFEST.json', OUT/'LIFTER_EVALUATION_PLAN.json', REVIEW/'CORNER_CONTRACT.json', REVIEW/'annotations_in_progress.json')
    sys.path.insert(0, str(ROOT/'scripts/annotate'))
    import annotate
    import annotate_review
    from object_geometry_registry import load_object_geometry_registry
    registry_path = ROOT/'challenge/config/CHALLENGE_OBJECT_GEOMETRY_REGISTRY.json'
    passes = ('primary', 'repeat') if args.review_pass == 'all' else (args.review_pass,)
    native = NativeReview(context, None, REVIEW/'native_annotations', passes,
                         native_pnp=not args.visibility_only, batch_plan=args.batch_plan,
                         revisit_saved=args.revisit_saved)
    registry = load_object_geometry_registry(registry_path)
    sessions, contexts = native.contexts('', argparse.Namespace(), registry, ROOT)
    print(json.dumps(dict(status='READY_EXISTING_ANNOTATE', editor='scripts/annotate/annotate.py',
        frozen_counts=context.counts(), requested_counts=native.requested_counts(),
        partial_batch_counts=native.batch_counts(),
        sessions=[name for name, _, _ in sessions], PnP=native.native_pnp,
        native_G_F_M_enabled=native.native_pnp,
        manual_geometry_proposals=True, reference_coordinate_autofill=False,
        evaluated_predictions_displayed=False), ensure_ascii=False), flush=True)
    if args.prepare_only or not sessions:
        return
    lock = (REVIEW/'native_editor.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('기존 annotate.py 검수 창이 이미 실행 중입니다.')
    native.change_reviewer = args.change_reviewer
    if native.profile_path.exists() and not args.change_reviewer:
        native.reviewer = json.loads(native.profile_path.read_text())['reviewer']
        validate_reviewer(native.reviewer)
    native.install(annotate)
    annotate_review.load_review_contexts = native.contexts
    window_width,window_height = initial_window_size()
    annotate.main(['--review-manifest', str(REVIEW/'MANIFEST.json'), '--stride', '1',
        '--population-role', 'DEV', '--default_split', 'eval', '--object-type', 'plastic_square',
        '--geometry-registry', str(registry_path), '--win-w', str(window_width), '--win-h', str(window_height)])
    if native.restart_requested:
        annotate.cv2.destroyAllWindows()
        lock.close()
        os.execv(sys.executable,[sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]])


if __name__ == '__main__':
    main()
