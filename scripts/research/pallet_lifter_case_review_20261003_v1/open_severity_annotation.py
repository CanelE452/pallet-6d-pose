"""Show frozen raw lifter images in annotate.py for direct three-class review.

Only explicit keyboard/button choices create labels. This sidecar never promotes
corner, target identity, pose accuracy, or reviewer provenance to reviewed status.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
from pathlib import Path
import sys
import time
import uuid

import numpy as np

from open_existing_annotation import HERE, ROOT, OUT, REVIEW, write_json
from serve import Context, ValidationError, utc_now

LABELS = {
    '1': ('clean', 'none', '가림 없음'),
    '2': ('moderate', 'partial', '중간'),
    '3': ('severe', 'heavy', '어려움 · 심한 가림'),
}
SCHEMA = 'lifter_direct_severity_review_v1'
RUBRIC = 'human_direct_three_class_lifter_v1'


class SeverityReview:
    schema = SCHEMA
    rubric = RUBRIC
    display_title = '리프터 원본'
    window_title = 'Annotate - Lifter occlusion 1 2 3'

    def __init__(self, context, store_path):
        self.ctx = context
        self.store_path = Path(store_path)
        self.workspace = self.store_path.parent / 'severity_annotations'
        self.path_map = {}
        self.current = None
        self.started_at = None
        self.started_mono = None
        self.queued_key = None
        self.queued_source = None
        self.buttons = []
        self.store = dict(schema=self.schema, rubric_version=self.rubric,
            input_bindings=copy.deepcopy(context.bindings),
            actor=dict(id='local-' + uuid.uuid4().hex[:12], id_origin='automatically_generated_local_alias',
                human_identity_confirmed=False, exposure_status='NOT_CONFIRMED'),
            full_reference_review_status='NOT_REVIEWED', records={}, history=[])
        if self.store_path.exists():
            candidate = json.loads(self.store_path.read_text())
            if (candidate.get('schema') != self.schema or candidate.get('rubric_version') != self.rubric
                    or candidate.get('input_bindings') != self.store['input_bindings']):
                raise ValidationError('Severity sidecar input/rubric mismatch; original preserved')
            for fid, row in candidate['records'].items():
                if (fid not in context.frames or row.get('severity') not in ('clean', 'moderate', 'severe')
                        or row.get('image_sha256') != context.frames[fid]['image_sha256']):
                    raise ValidationError('Invalid saved severity record: ' + fid)
            self.store = candidate

    def record_for_display(self, frame_id):
        return self.store['records'].get(frame_id)

    def counts(self):
        counts = {severity: 0 for severity in ('clean', 'moderate', 'severe')}
        for record in self.store['records'].values():
            counts[record['severity']] += 1
        counts['unreviewed'] = len(self.ctx.frames) - len(self.store['records'])
        return counts

    def contexts(self, manifest, cli_args, registry, repo):
        rows = list(self.ctx.frames.values())
        start = next((i for i, row in enumerate(rows) if row['frame_id'] not in self.store['records']), 0)
        rows = rows[start:] + rows[:start]
        spec = registry.resolve('plastic_square')
        args = copy.copy(cli_args)
        args.object_type = spec.object_type
        args.population_role = 'DEV'
        args.default_split = 'eval'
        args.capture_session_id = 'LIFTER_FIXED_SEVERITY_ONLY'
        args.lighting_condition = None
        args.intrinsics_quality = 'UNKNOWN'
        args.intrinsics_source = 'unused: severity-only; PnP disabled'
        for row in rows:
            path = self.workspace / (self.ctx.images[row['frame_id']].stem + '.json')
            self.path_map[str(path.resolve())] = row['frame_id']
        key = 'review:severity-only'
        context = dict(args=args, metadata={'population_role': 'DEV', 'object_type': spec.object_type},
            geometry_spec=spec, out_dir=str(self.workspace), K=np.eye(3), K_source=args.intrinsics_source,
            frame_paths=[str(self.ctx.images[row['frame_id']]) for row in rows], frame_count=len(rows),
            writable=True, workspace_scope=None, display_role='DEV', source_session_dir=str(self.workspace),
            refresh_evaluation=False, force_explicit_object_type=True, active_evaluation_member=False)
        return [('LIFTER_SEVERITY_ONLY_120', str(self.workspace), key)], {key: context}

    def load(self, state, path, read_only=False):
        self.current = self.path_map[str(Path(path).resolve())]
        self.started_at, self.started_mono = utc_now(), time.monotonic()
        self.queued_key = self.queued_source = None
        state.kps_2d = [None] * 9
        state.keypoint_annotations = None
        state.pose = None
        state.zoom = 1.
        state.pan = [0, 0]
        # These are sidecars, not ordinary corner/pose annotation JSON files.
        return False

    def persist(self):
        write_json(self.store_path, self.store)
        groups = dict(schema='lifter_severity_groups_v1', rubric_version=self.rubric,
            input_bindings=self.store['input_bindings'], counts=self.counts(),
            source=str(self.store_path), groups={key: [] for key in ('clean', 'moderate', 'severe', 'unreviewed')})
        for fid, row in self.ctx.frames.items():
            severity = self.store['records'].get(fid, {}).get('severity', 'unreviewed')
            groups['groups'][severity].append(dict(frame_id=fid, image_path=str(self.ctx.images[fid])))
        write_json(self.store_path.parent / 'SEVERITY_GROUPS.json', groups)

    def choose(self, key, path, source='human_keyboard'):
        if self.current is None or key not in LABELS:
            raise ValidationError('No current image or invalid explicit choice')
        frame = self.ctx.frames[self.current]
        severity, canonical, _ = LABELS[key]
        previous = copy.deepcopy(self.store['records'].get(self.current))
        record = dict(frame_id=self.current, session_id=frame['session_id'],
            saved_frame_index=frame['saved_frame_index'], image_sha256=frame['image_sha256'],
            severity=severity, occlusion_level=canonical, rubric_version=self.rubric,
            label_source='HUMAN_DIRECT_CLASS', input_action=source,
            classification_status='HUMAN_INPUT_RECORDED', actor_id=self.store['actor']['id'],
            human_identity_confirmed=False, started_at=self.started_at, selected_at=utc_now(),
            elapsed_view_seconds=round(time.monotonic() - self.started_mono, 3),
            corner_review_status='NOT_REVIEWED', target_identity_status='NOT_CONFIRMED',
            pose_reference_status='NOT_REVIEWED')
        self.store['records'][self.current] = record
        self.store['history'].append(dict(frame_id=self.current, action='explicit_class_selection',
            previous=previous, current=copy.deepcopy(record), performed_at=record['selected_at']))
        self.persist()
        write_json(Path(path), dict(schema=self.schema, record=record))
        print('SEVERITY_SAVED', self.current, severity, self.counts(), flush=True)

    def install(self, editor):
        self.editor = editor
        from PIL import Image, ImageDraw, ImageFont
        font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
        font = ImageFont.truetype(font_path, 18)
        large = ImageFont.truetype(font_path, 24)
        small = ImageFont.truetype(font_path, 16)
        original_wait = editor.cv2.waitKey

        def wait_key(delay):
            value = original_wait(delay)
            if self.queued_key is not None:
                value, self.queued_key = self.queued_key, None
            return value

        def key(value, state, out_json, out_png, src_png, K):
            if value in (ord('1'), ord('2'), ord('3')):
                try:
                    self.choose(chr(value), out_json, self.queued_source or 'human_keyboard')
                    self.queued_source = None
                    return 'save-next'
                except (OSError, ValidationError) as error:
                    print('SAVE FAILED:', error, flush=True)
                    editor._toast(state, 'SAVE FAILED - see console')
                    return None
            if value in (ord('q'), 27):
                return 'quit'
            if value in (ord('p'), ord('n')):
                return 'prev' if value == ord('p') else 'next'
            if value in (ord('s'), ord('S'), 13, 10):
                if self.current in self.store['records']:
                    return 'next'
                editor._toast(state, 'Choose 1 / 2 / 3; N skips without a label')
            if value in (ord('+'), ord('=')):
                state.zoom = min(4., state.zoom * 1.5)
            if value in (ord('-'), ord('_')):
                state.zoom = max(1., state.zoom / 1.5)
            if value == ord('h'): state.pan[0] -= 25
            if value == ord('l'): state.pan[0] += 25
            if value == ord('k'): state.pan[1] -= 25
            if value == ord('j'): state.pan[1] += 25
            if value == ord('r'):
                previous = self.store['records'].pop(self.current, None)
                if previous:
                    self.store['history'].append(dict(frame_id=self.current, action='human_reset_to_unreviewed',
                        previous=previous, performed_at=utc_now()))
                    self.persist()
                    Path(out_json).unlink(missing_ok=True)
                state.zoom, state.pan = 1., [0, 0]
            return None

        def mouse(event, x, y, flags, state):
            if event == editor.cv2.EVENT_LBUTTONDOWN:
                for box, value in self.buttons:
                    if box[0] <= x <= box[2] and box[1] <= y <= box[3]:
                        self.queued_key, self.queued_source = ord(value), 'human_mouse_button'
                        break

        def no_pose(state, K, force=False):
            state.pose = None
            state.mode = 'click'
            state.condition_mode = False
            state.line_mode = False

        def render(state, frame_idx, total, name):
            h, w = state.img.shape[:2]
            cw, ch = max(1, int(w / state.zoom)), max(1, int(h / state.zoom))
            state.pan[0] = max(0, min(w-cw, state.pan[0]))
            state.pan[1] = max(0, min(h-ch, state.pan[1]))
            crop = state.img[state.pan[1]:state.pan[1]+ch, state.pan[0]:state.pan[0]+cw]
            raw = editor.cv2.resize(crop, (960, 720), interpolation=editor.cv2.INTER_LINEAR)
            view = Image.new('RGB', (1290, 770), '#17202b')
            view.paste(Image.fromarray(raw[:, :, ::-1]), (0, 50))
            draw = ImageDraw.Draw(view)
            draw.text((16, 10), f'{self.display_title} · {frame_idx+1}/{total} · {self.current} · 확대 {state.zoom:.1f}배', font=font, fill='white')
            x = 978
            draw.text((x, 55), '가림만 분류하세요', font=large, fill='#a5e4c3')
            draw.text((x, 95), '버튼 클릭 또는 숫자 키', font=font, fill='#d9e5ee')
            draw.text((x, 123), '선택 → 자동 저장 → 다음', font=font, fill='#d9e5ee')
            colors = ['#24553b', '#655521', '#693733']
            self.buttons = []
            for i, (number, (_, _, label)) in enumerate(LABELS.items()):
                y = 176 + i*88
                box = (x, y, 1272, y+68)
                draw.rounded_rectangle(box, radius=10, fill=colors[i], outline='#aebcc8', width=2)
                draw.text((x+14, y+19), f'{number}  {label}', font=large if i<2 else font, fill='white')
                self.buttons.append((box, number))
            record = self.record_for_display(self.current)
            label = next((v[2] for v in LABELS.values() if record and v[0] == record['severity']), '미분류')
            draw.text((x, 446), '현재: ' + label, font=font, fill='#a5e4c3')
            counts = self.counts()
            for i, line in enumerate((f'없음 {counts["clean"]} · 중간 {counts["moderate"]}',
                    f'어려움 {counts["severe"]} · 남음 {counts["unreviewed"]}',
                    'P 이전 · N 보류하고 다음', 'R 현재 분류 취소 · Q 종료',
                    '+/− 확대 · H/J/K/L 이동', 'S: 이미 저장한 이미지에서 다음',
                    '뒤쪽 면이 안 보이는 자체 가림은', '외부 가림과 구분해서 판단하세요.',
                    '식별자 입력·코너 클릭 없음')):
                draw.text((x, 484+i*29), line, font=small, fill='#d9e5ee')
            if counts['unreviewed'] == 0:
                draw.text((15, 730), '분류 완료 · Q 종료 / P 다시 확인', font=font, fill='white')
            return np.asarray(view)[:, :, ::-1].copy()

        editor.WIN = self.window_title
        editor.cv2.waitKey = wait_key
        editor.load_existing_annotation = self.load
        editor.update_pose = no_pose
        editor.on_mouse = mouse
        editor._handle_click_key = key
        editor.render = render


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    # Verify the existing immutable inputs; use a separate empty corner store.
    context = Context(REVIEW/'MANIFEST.json', OUT/'LIFTER_EVALUATION_PLAN.json',
        REVIEW/'CORNER_CONTRACT.json', REVIEW/'severity_unused_corner_store.json')
    native = SeverityReview(context, REVIEW/'SEVERITY_REVIEW_IN_PROGRESS.json')
    print(json.dumps(dict(status='READY_NATIVE_SEVERITY_ONLY', frames=len(context.frames),
        counts=native.counts(), editor='scripts/annotate/annotate.py', new_training=0), ensure_ascii=False), flush=True)
    if args.prepare_only:
        return
    lock = (REVIEW/'severity_editor.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('가림 분류 창이 이미 실행 중입니다.')
    sys.path.insert(0, str(ROOT/'scripts/annotate'))
    import annotate
    import annotate_review
    native.install(annotate)
    annotate_review.load_review_contexts = native.contexts
    annotate.main(['--review-manifest', str(REVIEW/'MANIFEST.json'), '--stride', '1',
        '--population-role', 'DEV', '--default_split', 'eval', '--object-type', 'plastic_square',
        '--geometry-registry', str(ROOT/'challenge/config/CHALLENGE_OBJECT_GEOMETRY_REGISTRY.json'),
        '--win-w', '1290', '--win-h', '830'])


if __name__ == '__main__':
    main()
