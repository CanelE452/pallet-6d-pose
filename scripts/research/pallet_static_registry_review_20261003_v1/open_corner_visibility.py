"""Review only unresolved static reference-corner visibility in annotate.py.

Original coordinates and frozen annotations are read-only. Every new state needs
an actual keyboard/button action. Model predictions, errors and scores are never
loaded. Opening this tool alone cannot create a human visibility label.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts/research/pallet_lifter_case_review_20261003_v1'))
from open_existing_annotation import write_json
from serve import ValidationError, utc_now

MANIFEST = ROOT/'_docs/experiments/pallet_static_registry_review_20261003_v1/review/STATIC_REVIEW_MANIFEST.json'
SEVERITY = ROOT/'data/pallet/results/pallet_static_registry_review_20261003_v1/native_severity_review/STATIC_SEVERITY_INPUTS.json'
OUT = ROOT/'data/pallet/results/pallet_static_registry_review_20261003_v1/native_corner_visibility'
STATES = {'1': ('DIRECT_VISIBLE', '보임'), '2': ('EXTERNAL_OCCLUDED', '다른 물체에 가림'),
          '3': ('SELF_OCCLUDED', '뒤쪽 · 자체 가림'), '4': ('OUT_OF_FRAME', '화면 밖'),
          '5': ('UNKNOWN', '판단 보류')}
SCHEMA = 'static_native_reference_visibility_v1'
POPULATIONS = (('GREEN0918', '01_SQUARE_119', 'plastic_square', '정사각형 119'),
               ('DEV319', '02_RECTANGULAR_319', 'plastic_rectangular', '직사각형 319'))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_context(manifest_path=MANIFEST, severity_path=SEVERITY, root=ROOT):
    """Hash-check frozen inputs; coordinate flags are never visibility labels."""
    from PIL import Image
    from scripts.research.pallet_static_registry_review_20261003_v1 import geometry_visibility_proposals as geometry
    geometry_hash = sha256(geometry.__file__)
    manifest = json.loads(Path(manifest_path).read_text())
    severity = json.loads(Path(severity_path).read_text()) if Path(severity_path).exists() else None
    frames, images = {}, {}
    for case in manifest['cases']:
        image, annotation = root/case['image']['path'], root/case['annotation']['path']
        for path, binding in ((image, case['image']), (annotation, case['annotation'])):
            if not path.is_file() or sha256(path) != binding['sha256']:
                raise ValidationError('Frozen input missing/hash mismatch: '+str(path))
        source_annotation = json.loads(annotation.read_text())
        obj = source_annotation['objects'][0]
        with Image.open(image) as raw_image:
            image_size = raw_image.size
        proposals = geometry.proposal_for_object(obj, source_annotation.get('camera_data', {}), image_size)
        for proposal in proposals.values():
            proposal['implementation_sha256'] = geometry_hash
        annotations = obj.get('keypoint_annotations', [])
        fallback = obj.get('manual_kps') or obj.get('projected_cuboid') or []
        points = []
        for slot in case['corners']:
            i = slot['corner_id']
            point = annotations[i] if i < len(annotations) else {}
            xy = point.get('xy')
            if xy is None and i < len(fallback): xy = fallback[i]
            if xy is not None:
                xy = [float(xy[0]), float(xy[1])]
                if not np.isfinite(xy).all(): xy = None
            points.append(dict(slot, xy=xy, annotation_coordinate_source=point.get('source', 'legacy'),
                coordinate_origin='existing_frozen_annotation_not_new_independent_reference',
                geometry_proposal=proposals.get(i)))
        fid = case['case_id']
        label = severity and severity.get('records', {}).get(fid)
        grade = label['severity'] if label else case['frame_severity']['status']
        frames[fid] = dict(case, corners=points, current_frame_grade=grade)
        images[fid] = image
    # These hashes bind this snapshot only. A later grade revision cannot silently
    # change the context of an already recorded point review.
    return dict(frames=frames, images=images, bindings={
        'static_manifest_sha256': sha256(manifest_path),
        'severity_snapshot_sha256': sha256(severity_path) if severity else None,
        'source_bindings': manifest['source_bindings']})


class VisibilityReview:
    window_title = 'Annotate - Static corner visibility'

    def __init__(self, context, store_path, *, show_completed=False):
        self.ctx = context
        self.store_path = Path(store_path)
        self.workspace = self.store_path.parent/'viewer_navigation'
        self.resume_path = self.store_path.parent/'VIEWER_RESUME.json'
        self.resume = {}
        self.restart_requested = False
        self.path_map = {}
        self.show_completed = show_completed
        self.population_sessions = {}
        self.current = None
        self.cursor = None
        self.started_at = self.started_mono = None
        self.queued_key = self.queued_source = None
        self.buttons = []
        self.notice = ''
        self.store = dict(schema=SCHEMA, input_bindings=copy.deepcopy(context['bindings']),
            actor=dict(id='local-'+uuid.uuid4().hex[:12], id_origin='automatically_generated_local_alias',
                human_identity_confirmed=False, prior_prediction_exposure='NOT_CONFIRMED'),
            review_scope='static_metric_reference_visibility_only', coordinates_editable=False,
            independent_reference_claim=False, records={}, exposures={}, history=[])
        if self.store_path.exists():
            candidate = json.loads(self.store_path.read_text())
            if candidate.get('schema') != SCHEMA or candidate.get('input_bindings') != self.store['input_bindings']:
                raise ValidationError('Visibility sidecar contract mismatch; original preserved')
            for fid, record in candidate.get('records', {}).items():
                if fid not in context['frames']: raise ValidationError('Unknown saved frame: '+fid)
                case = context['frames'][fid]
                if (record.get('image_sha256') != case['image']['sha256']
                        or record.get('annotation_sha256') != case['annotation']['sha256']):
                    raise ValidationError('Saved frame binding mismatch: '+fid)
                for key, row in record.get('corners', {}).items():
                    point = self.point(fid, int(key))
                    if (not point['metric_reference'] or point['locked']
                            or row['status'] not in {v[0] for v in STATES.values()}
                            or row['reference_xy'] != point['xy']):
                        raise ValidationError('Invalid saved corner: '+fid+':'+key)
            self.store = candidate
        if self.resume_path.is_file():
            try:
                saved = json.loads(self.resume_path.read_text())
                if (isinstance(saved, dict) and saved.get('schema') == 'static_visibility_viewer_resume_v1'
                        and saved.get('input_bindings') == context['bindings']
                        and saved.get('case_id') in context['frames']):
                    self.resume = saved
            except (OSError, ValueError, TypeError):
                pass  # A viewport bookmark never invalidates human annotations.

    def resume_case(self):
        if self.resume:
            return self.resume['case_id']
        return next((event['case_id'] for event in reversed(self.store['history'])
                     if event.get('case_id') in self.ctx['frames']), None)

    def save_view(self, state):
        if self.current is None:
            return
        view = dict(schema='static_visibility_viewer_resume_v1',
            source_kind='automatic_viewer_bookmark', input_bindings=copy.deepcopy(self.ctx['bindings']),
            case_id=self.current, cursor=self.cursor, zoom=float(state.zoom),
            pan=[float(v) for v in state.pan], annotation_states_changed=False)
        if all(self.resume.get(key) == value for key, value in view.items()):
            return
        view['saved_at'] = utc_now()
        write_json(self.resume_path, view)
        self.resume = view

    def point(self, fid, index):
        return next(p for p in self.ctx['frames'][fid]['corners'] if p['corner_id'] == index)

    def targets(self, fid):
        return [p['corner_id'] for p in self.ctx['frames'][fid]['corners']
                if p['metric_reference'] and not p['locked']]

    def state_for(self, fid, index):
        point = self.point(fid, index)
        if point['locked']: return point['status']
        return self.store['records'].get(fid, {}).get('corners', {}).get(str(index), {}).get('status')

    def pending(self, fid):
        return [i for i in self.targets(fid) if self.state_for(fid, i) is None]

    def pending_proposals(self, fid):
        return {i: self.point(fid, i)['geometry_proposal'] for i in self.pending(fid)
                if (self.point(fid, i).get('geometry_proposal') or {}).get('status')
                in ('SELF_OCCLUDED', 'OUT_OF_FRAME')}

    def counts(self):
        pending = sum(len(self.pending(fid)) for fid in self.ctx['frames'])
        entered = sum(len(r['corners']) for r in self.store['records'].values())
        unknown = sum(p['status'] == 'UNKNOWN' for r in self.store['records'].values() for p in r['corners'].values())
        known = sum(p['locked'] for r in self.ctx['frames'].values() for p in r['corners'] if p['metric_reference'])
        suggestions = [p for fid in self.ctx['frames'] for p in self.pending_proposals(fid).values()]
        return dict(pending=pending, human_input_points=entered, preapproved_preserved=known,
                    unresolved_human_unknown=unknown, frames=len(self.ctx['frames']),
                    pending_geometry_proposals=len(suggestions),
                    pending_self_proposals=sum(p['status']=='SELF_OCCLUDED' for p in suggestions),
                    pending_outside_proposals=sum(p['status']=='OUT_OF_FRAME' for p in suggestions))

    def persist(self):
        self.store['updated_at'] = utc_now()
        self.store['counts'] = self.counts()
        write_json(self.store_path, self.store)

    def expose(self):
        changed = False
        if self.current not in self.store['exposures']:
            case = self.ctx['frames'][self.current]
            self.store['exposures'][self.current] = dict(opened_at=utc_now(),
                image_sha256=case['image']['sha256'], annotation_sha256=case['annotation']['sha256'],
                display='raw_RGB_with_existing_reference_corner_overlay', reference_overlay_exposure=True,
                model_predictions_displayed=False, source_coordinates_not_new_independent_GT=True)
            changed = True
        exposed = self.store['exposures'][self.current]
        proposals = self.pending_proposals(self.current)
        if proposals and not exposed.get('geometry_proposals_displayed'):
            exposed.update(geometry_proposals_displayed=True,
                geometry_proposals_first_shown_at=utc_now(),
                geometry_proposal_sources=copy.deepcopy(proposals), machine_assistance=True)
            changed = True
        if changed:
            self.persist()

    def contexts(self, manifest, cli_args, registry, repo):
        sessions, contexts = [], {}
        self.path_map.clear()
        self.population_sessions.clear()
        resume_case = self.resume_case()
        resume_population = self.ctx['frames'].get(resume_case, {}).get('population')
        populations = sorted(POPULATIONS, key=lambda row: row[0] != resume_population)
        for population, name, geometry, _ in populations:
            all_rows = [r for r in self.ctx['frames'].values() if r['population'] == population]
            if population == resume_population:
                index = next((i for i,r in enumerate(all_rows) if r['case_id']==resume_case), 0)
                all_rows = all_rows[index:] + all_rows[:index]
            rows = all_rows
            if not self.show_completed:
                rows = [r for r in rows if self.pending(r['case_id'])]
            if not rows:
                continue
            first = 0 if self.show_completed and resume_case == rows[0]['case_id'] else next((i for i,r in enumerate(rows) if self.pending(r['case_id'])), 0)
            rows = rows[first:]+rows[:first]
            try: spec = registry.resolve(geometry)
            except (KeyError, ValueError): spec = registry.resolve('plastic_square')
            args = copy.copy(cli_args)
            args.object_type=spec.object_type; args.population_role='DEV'; args.default_split='eval'
            args.capture_session_id=population; args.lighting_condition=None
            args.intrinsics_quality='UNKNOWN'; args.intrinsics_source='unused: visibility only; PnP OFF'
            output = self.workspace/population
            for row in rows:
                path = output/(self.ctx['images'][row['case_id']].stem+'.json')
                self.path_map[str(path.resolve())] = row['case_id']
            key = 'review:visibility:'+population
            self.population_sessions[population] = len(sessions)
            sessions.append((name, str(output), key))
            contexts[key] = dict(args=args, metadata={'population_role':'DEV','object_type':spec.object_type},
                geometry_spec=spec, out_dir=str(output), K=np.eye(3), K_source=args.intrinsics_source,
                frame_paths=[str(self.ctx['images'][r['case_id']]) for r in rows], frame_count=len(rows),
                writable=True, workspace_scope=None, display_role='DEV', source_session_dir=str(output),
                refresh_evaluation=False, force_explicit_object_type=True, active_evaluation_member=False)
        return sessions, contexts

    def population_buttons(self):
        """Map the fixed two labels to the filtered native session list."""
        buttons = []
        for population, _, _, label in POPULATIONS:
            has_pending = any(self.pending(fid) for fid, row in self.ctx['frames'].items()
                              if row['population'] == population)
            index = self.population_sessions.get(population)
            buttons.append(dict(population=population, label=label,
                session_index=index, complete=not has_pending,
                enabled=index is not None and (has_pending or self.show_completed)))
        return buttons

    def load(self, state, path, read_only=False):
        self.current = self.path_map[str(Path(path).resolve())]
        targets = self.pending(self.current) or self.targets(self.current)
        self.cursor = targets[0] if targets else None
        self.started_at, self.started_mono = utc_now(), time.monotonic()
        self.queued_key = self.queued_source = None
        state.kps_2d = [None]*9; state.keypoint_annotations=None; state.pose=None
        state.zoom=1.; state.pan=[0,0]
        if self.resume.get('case_id') == self.current:
            cursor = self.resume.get('cursor')
            if type(cursor) is int and cursor in targets:
                self.cursor = cursor
            try:
                zoom = float(self.resume.get('zoom', 1.))
                pan = np.asarray(self.resume.get('pan', [0,0]), dtype=float)
                if np.isfinite(zoom) and 1. <= zoom <= 4. and pan.shape == (2,) and np.isfinite(pan).all():
                    state.zoom, state.pan = zoom, [int(round(v)) for v in pan]
            except (TypeError, ValueError):
                pass
        self.notice = ''
        return False

    def choose(self, key, source='human_keyboard', all_remaining=False, geometry_confirm=False):
        if key not in STATES or self.current is None:
            raise ValidationError('No current image or invalid human action')
        proposals = self.pending_proposals(self.current) if geometry_confirm else {}
        targets = list(proposals) if geometry_confirm else self.pending(self.current) if all_remaining else [self.cursor]
        targets = [i for i in targets if i is not None]
        if not targets: return False
        if self.current not in self.store['exposures']:
            raise ValidationError('Reference was not displayed; no label recorded')
        for i in targets:
            if i not in self.targets(self.current): raise ValidationError('Locked/nonmetric point cannot be edited')
        case = self.ctx['frames'][self.current]
        record = self.store['records'].setdefault(self.current, dict(frame_id=case['frame_id'],
            case_id=self.current, population=case['population'], image_sha256=case['image']['sha256'],
            annotation_sha256=case['annotation']['sha256'], frame_grade_at_review=case['current_frame_grade'], corners={}))
        before = copy.deepcopy(record['corners'])
        for i in targets:
            point = self.point(self.current, i)
            record['corners'][str(i)] = dict(corner_id=i,
                status=proposals[i]['status'] if geometry_confirm else STATES[key][0],
                reference_xy=copy.deepcopy(point['xy']), reference_source=point['reference_source'],
                annotation_coordinate_source=point['annotation_coordinate_source'],
                reference_coordinate_origin=point['coordinate_origin'], actor_id=self.store['actor']['id'],
                human_identity_confirmed=False, input_action=source,
                human_input_status='HUMAN_CONFIRMED_GEOMETRY_PROPOSAL' if geometry_confirm else 'HUMAN_EXPLICIT_VISIBILITY_INPUT',
                selected_at=utc_now(),
                viewed_seconds=round(time.monotonic()-self.started_mono,3),
                reference_overlay_exposure=True, model_prediction_exposure='NOT_CONFIRMED',
                geometry_proposal_exposure=bool(self.store['exposures'][self.current].get('geometry_proposals_displayed')),
                machine_assistance=bool(self.store['exposures'][self.current].get('geometry_proposals_displayed')),
                does_not_create_independent_6D_reference=True)
            if geometry_confirm:
                record['corners'][str(i)]['geometry_proposal'] = copy.deepcopy(proposals[i])
        action = 'explicit_geometry_proposals_confirmed' if geometry_confirm else 'all_remaining_visible' if all_remaining else 'single_visibility'
        self.store['history'].append(dict(case_id=self.current, action=action,
            previous_corners=before, current_corners=copy.deepcopy(record['corners']), performed_at=utc_now()))
        self.persist()
        pending = self.pending(self.current)
        self.cursor = pending[0] if pending else self.cursor
        self.notice = f'{len(targets)}점 저장됨'
        return not pending

    def reset(self):
        record = self.store['records'].get(self.current)
        if not record or not record['corners']: return
        previous = copy.deepcopy(record['corners'])
        record['corners'] = {}
        self.store['history'].append(dict(case_id=self.current, action='human_reset_new_inputs',
            previous_corners=previous, current_corners={}, performed_at=utc_now()))
        self.cursor = self.targets(self.current)[0] if self.targets(self.current) else None
        self.persist(); self.notice='현재 이미지의 새 입력 취소 (기존 확정점 유지)'

    def undo(self):
        # Undo on the current frame preserves all frozen/preapproved statuses.
        for event in reversed(self.store['history']):
            if event['case_id'] == self.current and not event.get('undone'):
                self.store['records'][self.current]['corners'] = copy.deepcopy(event['previous_corners'])
                event['undone'] = True
                self.persist()
                pending = self.pending(self.current)
                self.cursor = pending[0] if pending else self.cursor
                self.notice='되돌림 저장됨'; return

    def install(self, editor):
        self.editor=editor
        from PIL import Image, ImageDraw, ImageFont
        font_path='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
        font=ImageFont.truetype(font_path,19); small=ImageFont.truetype(font_path,16)
        big=ImageFont.truetype(font_path,25)
        original_wait=editor.cv2.waitKey
        def wait_key(delay):
            value=original_wait(delay)
            if self.queued_key is not None: value,self.queued_key=self.queued_key,None
            return value
        def key(value,state,*args):
            if value in (ord('c'), ord('C')):
                done = self.choose('1', self.queued_source or 'human_keyboard_geometry_confirmation', geometry_confirm=True)
                self.queued_source = None
                return 'save-next' if done else None
            if value in [ord(k) for k in STATES] or value in (ord('a'),ord('A')):
                done=self.choose('1' if value in (ord('a'),ord('A')) else chr(value),
                    self.queued_source or 'human_keyboard',all_remaining=value in (ord('a'),ord('A')))
                self.queued_source=None
                return 'save-next' if done else None
            if value in (ord('q'),ord('Q'),27):
                self.persist();self.save_view(state)
                return 'quit'
            if value in (ord('t'),ord('T')):
                self.persist();self.save_view(state)
                self.restart_requested = True
                return 'quit'
            if value in (ord('p'),ord('n')): return 'prev' if value==ord('p') else 'next'
            if value in (ord('s'),ord('S'),13,10):
                self.persist();self.save_view(state)
                self.notice='저장 완료 · Q 닫기 / T 저장 후 다시 열기'
                return None
            if value==ord('r'): self.reset()
            if value==ord('z'): self.undo()
            if value in (ord(','),ord('.')):
                ids=self.targets(self.current)
                if ids: self.cursor=ids[(ids.index(self.cursor)+(1 if value==ord('.') else -1))%len(ids)]
            if value in (ord('+'),ord('=')): state.zoom=min(4.,state.zoom*1.5)
            if value in (ord('-'),ord('_')): state.zoom=max(1.,state.zoom/1.5)
            for code,dx,dy in [('h',-30,0),('l',30,0),('k',0,-30),('j',0,30)]:
                if value==ord(code): state.pan[0]+=dx;state.pan[1]+=dy
            return None
        def no_pose(state,K,force=False):
            state.pose=None;state.mode='click';state.condition_mode=False;state.line_mode=False
        def mouse(event,x,y,flags,state):
            if event!=editor.cv2.EVENT_LBUTTONDOWN:return
            for box,value in self.buttons:
                if box[0]<=x<=box[2] and box[1]<=y<=box[3]:
                    if value.startswith('session:'): state.sess_pick=int(value.split(':')[1])
                    elif value.startswith('corner:'): self.cursor=int(value.split(':')[1])
                    else: self.queued_key=ord(value);self.queued_source='human_mouse_button'
                    return
            if 0<=x<960 and 50<=y<770:
                h,w=state.img.shape[:2];cw,ch=max(1,int(w/state.zoom)),max(1,int(h/state.zoom))
                px,py=state.pan[0]+x*cw/960,state.pan[1]+(y-50)*ch/720
                distances=[(np.hypot((p['xy'][0]-px)*960/cw,(p['xy'][1]-py)*720/ch),p['corner_id'])
                    for p in self.ctx['frames'][self.current]['corners'] if p['corner_id'] in self.targets(self.current) and p['xy'] is not None]
                if distances:
                    distance,index=min(distances)
                    if distance<=25:self.cursor=index
        def render(state,frame_idx,total,name):
            self.expose()
            case=self.ctx['frames'][self.current];h,w=state.img.shape[:2]
            cw,ch=max(1,int(w/state.zoom)),max(1,int(h/state.zoom))
            state.pan[0]=max(0,min(w-cw,state.pan[0]));state.pan[1]=max(0,min(h-ch,state.pan[1]))
            self.save_view(state)
            crop=state.img[state.pan[1]:state.pan[1]+ch,state.pan[0]:state.pan[0]+cw]
            raw=editor.cv2.resize(crop,(960,720),interpolation=editor.cv2.INTER_LINEAR)
            view=Image.new('RGB',(1430,950),'#17202b');view.paste(Image.fromarray(raw[:,:,::-1]),(0,50))
            draw=ImageDraw.Draw(view);title='정사각형 119장' if case['population']=='GREEN0918' else '직사각형 319장'
            draw.text((12,10),f'{title} · {frame_idx+1}/{total} · 기존 코너의 보임 여부만 확인',font=font,fill='white')
            self.buttons=[]
            for i,button in enumerate(self.population_buttons()):
                box=(978+i*215,7,1184+i*215,44)
                draw.rounded_rectangle(box,radius=6,fill='#36596d' if button['enabled'] else '#303940')
                label=button['label']+(' · 입력 완료' if button['complete'] else '')
                draw.text((box[0]+9,box[1]+7),label,font=small,
                          fill='white' if button['enabled'] else '#9cb3c7')
                if button['enabled']:
                    self.buttons.append((box,'session:'+str(button['session_index'])))
            # Show existing reference slots only. Gray slots are not required and
            # never become reviewed because their coordinate exists.
            for point in case['corners']:
                if point['xy'] is None:continue
                x=(point['xy'][0]-state.pan[0])*960/cw;y=50+(point['xy'][1]-state.pan[1])*720/ch
                if not (0<=x<960 and 50<=y<770):
                    if point['corner_id']!=self.cursor:continue
                    # A boundary marker locates an off-image reference without
                    # creating an OUT_OF_FRAME classification automatically.
                    x=max(20,min(938,x));y=max(72,min(748,y))
                    draw.text((x-15,y+25),'경계 밖 참조 위치',font=small,fill='#efcb83',stroke_width=1,stroke_fill='#101010')
                index=point['corner_id'];status=self.state_for(self.current,index)
                color='#80db9a' if status=='DIRECT_VISIBLE' else '#ecba55' if status else '#f0f2f5'
                if not status and index in self.pending_proposals(self.current): color='#87baff'
                if not point['metric_reference']:color='#929599'
                radius=15 if index==self.cursor else 7
                draw.ellipse((x-radius,y-radius,x+radius,y+radius),outline=color,width=3)
                if index==self.cursor:draw.ellipse((x-20,y-20,x+20,y+20),outline='#ffffff',width=2)
                draw.text((x+10,y-25),str(index),font=big,fill=color,stroke_width=2,stroke_fill='#101010')
            x=978
            draw.text((x,58),f'선택 코너 {self.cursor} · 숫자로 판정',font=big,fill='#a5e4c3')
            draw.text((x,98),'좌표는 그대로 · 다시 찍을 필요 없음',font=font,fill='#dbe5ee')
            for i,(number,(_,label)) in enumerate(STATES.items()):
                y=140+i*57;box=(x,y,1410,y+46);draw.rounded_rectangle(box,radius=8,fill='#344c62')
                draw.text((x+12,y+9),f'{number}  {label}',font=font,fill='white');self.buttons.append((box,number))
            box=(x,434,1410,476);draw.rounded_rectangle(box,radius=8,fill='#286344')
            draw.text((x+12,442),'A  남은 코너 모두 보임',font=font,fill='white');self.buttons.append((box,'a'))
            proposals=self.pending_proposals(self.current)
            box=(x,486,1410,528);draw.rounded_rectangle(box,radius=8,fill='#345979' if proposals else '#303940')
            draw.text((x+12,494),f'C  자동 제안 확인 ({len(proposals)}점)',font=font,fill='white' if proposals else '#89949d')
            if proposals:self.buttons.append((box,'c'))
            # Click a numbered row to revise a required point; approved slots stay locked.
            for i in range(8):
                point=self.point(self.current,i);status=self.state_for(self.current,i)
                text=next((v[1] for v in STATES.values() if v[0]==status),'미입력')
                if not status and i in proposals:
                    text='제안: '+('자체 가림' if proposals[i]['status']=='SELF_OCCLUDED' else '화면 밖')
                if point['locked']:text+=' (기존 확정)'
                if not point['metric_reference']:text='이번 평가 대상 아님'
                y=547+i*27;draw.text((x,y),f'{">" if i==self.cursor else " "} {i}  {text}',font=small,fill='#87baff' if not status and i in proposals else '#dbe5ee')
                if i in self.targets(self.current):self.buttons.append(((x,y,1410,y+27),'corner:'+str(i)))
            counts=self.counts()
            draw.text((978,782),f'새 입력 {counts["human_input_points"]} · 남음 {counts["pending"]}',font=small,fill='#dbe5ee')
            for box,value,text in [((978,816,1104,852),'s','S 저장'),
                    ((1114,816,1410,852),'t','T 저장 후 다시 열기')]:
                draw.rounded_rectangle(box,radius=6,fill='#36596d')
                draw.text((box[0]+10,box[1]+6),text,font=small,fill='white')
                self.buttons.append((box,value))
            draw.text((978,861),'Q 저장·닫기 · P/N 이동 · Z 되돌림',font=small,fill='#dbe5ee')
            # A separate magnifier preserves both original context and legibility.
            if self.cursor is not None:
                point=self.point(self.current,self.cursor)
                if point['xy'] is not None:
                    px,py=map(lambda v:int(round(v)),point['xy']);half=65
                    x0,x1=max(0,px-half),min(w,px+half);y0,y1=max(0,py-half),min(h,py+half)
                    zoom=state.img[y0:y1,x0:x1]
                    if zoom.size:
                        zoom=editor.cv2.resize(zoom,(170,170),interpolation=editor.cv2.INTER_NEAREST)
                        view.paste(Image.fromarray(zoom[:,:,::-1]),(8,778));draw=ImageDraw.Draw(view)
                        zx=8+(px-x0)*170/(x1-x0);zy=778+(py-y0)*170/(y1-y0)
                        draw.ellipse((zx-9,zy-9,zx+9,zy+9),outline='#ffffff',width=2)
                        draw.text((193,787),f'선택 코너 {self.cursor} 확대',font=font,fill='#a5e4c3')
            draw.text((193,829),'회색 점은 평가 대상이 아닙니다. 기존 확정점 71개는 유지됩니다.',font=small,fill='#dbe5ee')
            draw.text((193,865),'기존 수동 / PnP 참조 좌표를 표시합니다. 새로운 독립 정답이 아닙니다.',font=small,fill='#dbe5ee')
            draw.text((193,902),'파란 점은 기하 제안 · 모양을 보고 맞으면 C · 모두 실제 보일 때만 A',font=small,fill='#87baff')
            draw.text((193,926),self.notice,font=small,fill='#9debbe')
            return np.asarray(view)[:,:,::-1].copy()
        editor.WIN=self.window_title;editor.cv2.waitKey=wait_key
        editor.load_existing_annotation=self.load;editor.update_pose=no_pose
        editor.on_mouse=mouse;editor._handle_click_key=key;editor.render=render


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--show-completed',action='store_true',
                        help='Include previously entered images for optional inspection/revision.')
    args=parser.parse_args()
    native=VisibilityReview(load_context(),OUT/'STATIC_CORNER_VISIBILITY_INPUTS.json',
                           show_completed=args.show_completed)
    no_pending=native.counts()['pending']==0 and not args.show_completed
    print(json.dumps(dict(status='NO_PENDING_VISIBILITY_INPUT' if no_pending else 'READY_NATIVE_STATIC_VISIBILITY',
        counts=native.counts(),show_completed=args.show_completed,
        editor='scripts/annotate/annotate.py',new_training=0),ensure_ascii=False),flush=True)
    if args.prepare_only or no_pending:return
    OUT.mkdir(parents=True,exist_ok=True);lock=(OUT/'native_editor.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('코너 가시성 창이 이미 열려 있습니다.')
    sys.path.insert(0,str(ROOT/'scripts/annotate'))
    import annotate,annotate_review
    native.install(annotate);annotate_review.load_review_contexts=native.contexts
    annotate.main(['--review-manifest',str(MANIFEST),'--stride','1','--population-role','DEV',
        '--default_split','eval','--object-type','plastic_square','--geometry-registry',
        str(ROOT/'challenge/config/CHALLENGE_OBJECT_GEOMETRY_REGISTRY.json'),'--win-w','1430','--win-h','1010'])
    if native.restart_requested:
        annotate.cv2.destroyAllWindows()
        lock.close()
        os.execv(sys.executable, [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])


if __name__=='__main__':main()
