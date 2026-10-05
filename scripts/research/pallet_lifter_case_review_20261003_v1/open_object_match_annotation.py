"""Review the frozen selected box with annotate.py, after primary corner review.

Only three explicit human decisions are possible. No method, predicted corner,
error, or reference corner is displayed. Each queue binds an immutable snapshot
of the completed primary reference, so repeat review cannot invalidate decisions.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / 'data/pallet/results' / HERE.name
REVIEW = OUT / 'review'
sys.path.insert(0, str(HERE / 'combined_integration'))
from bridge import (GateError, read_json, sha256, write_json,
                    build_object_match_queue, apply_sidecar)
from object_match_review import Context, validate_reviewer, utc_now
sys.path.insert(0, str(HERE / 'review'))
from serve import Context as CornerContext

CHOICES = {'1': ('same', '같은 파렛트'), '2': ('different', '다른 대상'),
           '3': ('undetermined', '판단하기 어려움')}


def primary_snapshot(reference, required_ids, bindings, scope_metadata=None):
    """Stable data only: no mutable export timestamp or repeat-pass records."""
    if reference.get('bindings') != bindings:
        raise GateError('사람 코너 참조의 입력 계약이 일치하지 않습니다.')
    records = {r['frame_id']: r for r in reference['records']
               if r['review_pass'] == 'primary' and r['status'] in ('reviewed', 'skipped')}
    missing = sorted(set(required_ids) - set(records))
    if missing:
        raise GateError(f'주 검수 코너 {len(missing)}장이 남았습니다. 먼저 코너 작업을 마쳐 주세요.')
    if reference.get('source_kind') != 'human_reviewed':
        raise GateError('제출된 사람 코너 참조가 필요합니다.')
    result = dict(schema_version=reference['schema_version'], source_kind='human_reviewed',
                bindings=copy.deepcopy(bindings),
                records=[copy.deepcopy(records[fid]) for fid in sorted(required_ids)],
                scope='USER_EXCLUSIONS_APPLIED_PRIMARY_SNAPSHOT',
                frozen_original_primary_denominator=120,
                retained_primary_denominator=len(required_ids))
    if scope_metadata is not None:
        result.update(copy.deepcopy(scope_metadata))
    return result


def batch_scope(batch_plan, bindings, eligible_ids, original_count):
    """An explicit partial task; never change the plan or auto-select done rows."""
    path = Path(batch_plan).resolve()
    raw = path.read_bytes()
    batch = json.loads(raw)
    if (not isinstance(batch, dict)
            or batch.get('schema', batch.get('schema_version')) != 'lifter_human_review_batch_v1'
            or batch.get('bindings') != bindings
            or batch.get('source_kind') != 'partial_human_review_queue'
            or batch.get('frozen_plan_modified') is not False):
        raise GateError('부분 검수 배치의 스키마·입력 해시·원본 보존 계약이 일치하지 않습니다.')
    ids = batch.get('frame_ids')
    if (not isinstance(ids, list) or len(ids) != 12
            or any(not isinstance(fid, str) or not fid for fid in ids)
            or len(set(ids)) != len(ids)
            or not set(ids) <= set(eligible_ids)
            or batch.get('repeat_frame_ids') != []):
        raise GateError('부분 검수 배치는 제외되지 않은 고정 주 표본 12장과 빈 반복 목록이어야 합니다.')
    metadata = dict(scope='PARTIAL_HUMAN_REVIEW_BATCH_SNAPSHOT',
        batch_plan_path=str(path), batch_plan_sha256=hashlib.sha256(raw).hexdigest(),
        batch_schema='lifter_human_review_batch_v1',
        frozen_original_primary_denominator=original_count,
        retained_primary_denominator=len(eligible_ids), batch_primary_denominator=len(ids),
        not_selected_retained_primary_count=len(eligible_ids)-len(ids),
        excluded_primary_count=original_count-len(eligible_ids),
        full_frozen_plan_completed=False, frozen_plan_modified=False,
        repeat_review_in_this_batch=False,
        analysis_scope='partial_exploratory_visible_corner_evaluation',
        contract_source='EXECUTE_COMBINED_CLI_KO.txt:L7B,L7C;L6 repeat quality separated',
        limitations='12-frame partial batch; does not complete the original 120/24 or retained 115/23; '
                    'no independent physical T/R reference or full-cohort generalization')
    return set(ids), metadata


def prepare(batch_plan=None):
    manifest = REVIEW / 'MANIFEST.json'
    corner = CornerContext(manifest, OUT / 'LIFTER_EVALUATION_PLAN.json',
                           REVIEW / 'CORNER_CONTRACT.json', REVIEW / 'annotations_in_progress.json')
    exclusions_path = REVIEW / 'USER_EXCLUSIONS.json'
    excluded = set()
    if exclusions_path.is_file():
        exclusions = read_json(exclusions_path)
        if exclusions.get('input_bindings') != corner.bindings:
            raise GateError('제외 기록의 입력 해시가 바뀌었습니다.')
        excluded = set(exclusions['excluded_frame_ids'])
        if not excluded <= set(corner.frames):
            raise GateError('검수 목록에 없는 제외 프레임입니다.')
    eligible = set(corner.frames) - excluded
    required, scope_metadata = eligible, None
    if batch_plan is not None:
        required, scope_metadata = batch_scope(batch_plan, corner.bindings, eligible, len(corner.frames))
    reference = corner.export()
    snapshot = primary_snapshot(reference, required, corner.bindings, scope_metadata)
    corner.validate_bundle(snapshot)
    digest = hashlib.sha256(json.dumps(snapshot, sort_keys=True, ensure_ascii=False,
                            allow_nan=False).encode('utf-8')).hexdigest()
    folder = REVIEW / 'native_object_match' / digest
    folder.mkdir(parents=True, exist_ok=True)
    reference_path = folder / 'PRIMARY_REFERENCE_FROZEN.json'
    if reference_path.exists():
        if read_json(reference_path) != snapshot:
            raise GateError('잠긴 주 검수 복사본이 변경되었습니다.')
    else:
        write_json(reference_path, snapshot)
    predictions = OUT / 'raw_predictions/ALL_STORED_FRAMES.jsonl'
    queue_path = folder / 'OBJECT_MATCH_QUEUE.json'
    if not queue_path.exists():
        build_object_match_queue(predictions, reference_path, manifest, queue_path)
    ctx = Context(queue_path, manifest, predictions, reference_path,
                  folder / 'object_match_in_progress.json')
    # Whole-task progress is outside the stable reference: later repeat or
    # outside-batch submissions cannot invalidate completed object decisions.
    submitted = {r['frame_id'] for r in reference['records']
                 if r['review_pass'] == 'primary' and r['status'] in ('reviewed', 'skipped')}
    progress = dict(frozen_original_primary_denominator=len(corner.frames),
        retained_primary_denominator=len(eligible), batch_primary_denominator=len(required),
        retained_primary_submitted_count=len(submitted & eligible),
        retained_primary_pending_count=len(eligible-submitted),
        batch_primary_pending_count=len(required-submitted),
        pending_outside_batch_count=len((eligible-required)-submitted),
        excluded_primary_count=len(excluded), frozen_plan_modified=False,
        partial_batch=batch_plan is not None)
    if scope_metadata is not None:
        progress.update(batch_plan_path=scope_metadata['batch_plan_path'],
                        batch_plan_sha256=scope_metadata['batch_plan_sha256'],
                        analysis_scope=scope_metadata['analysis_scope'],
                        full_frozen_plan_completed=False)
    ctx.native_scope_metadata = progress
    write_json(REVIEW / 'native_object_match/ACTIVE_QUEUE.json', dict(
        source_kind='machine_prepared_human_task', snapshot_sha256=sha256(reference_path),
        queue_sha256=sha256(queue_path), folder=str(folder),
        counts=ctx.counts(), retained_primary=len(eligible), excluded=len(excluded),
        primary_scope=progress))
    return ctx


class NativeObjectReview:
    window_title = 'Annotate - Selected pallet match 1 2 3'

    def __init__(self, ctx):
        self.ctx = ctx
        self.workspace = ctx.store_path.parent / 'native_frames'
        self.paths = {}
        self.current = self.token = None
        self.queued_key = None
        self.buttons = []
        self.profile_path = ctx.store_path.parent / 'REVIEWER_PROFILE.json'
        self.reviewer = read_json(self.profile_path) if self.profile_path.exists() else None
        if self.reviewer:
            validate_reviewer(self.reviewer)

    def contexts(self, manifest, cli_args, registry, repo):
        rows = list(self.ctx.records.values())
        submitted = {r['frame_id'] for r in self.ctx.store['records']}
        start = next((i for i, r in enumerate(rows) if r['frame_id'] not in submitted), 0)
        rows = rows[start:] + rows[:start]
        spec = registry.resolve('plastic_square')
        args = copy.copy(cli_args)
        args.object_type, args.capture_session_id = spec.object_type, 'LIFTER_OBJECT_MATCH'
        args.population_role, args.default_split = 'DEV', 'eval'
        args.lighting_condition, args.intrinsics_quality = None, 'UNKNOWN'
        args.intrinsics_source = 'unused: selected-box identity only; PnP disabled'
        for r in rows:
            self.paths[str((self.workspace / (self.ctx.images[r['frame_id']].stem + '.json')).resolve())] = r['frame_id']
        key = 'review:frozen-selected-object'
        context = dict(args=args, metadata={'population_role': 'DEV', 'object_type': spec.object_type},
                       geometry_spec=spec, out_dir=str(self.workspace), K=np.eye(3),
                       K_source=args.intrinsics_source,
                       frame_paths=[str(self.ctx.images[r['frame_id']]) for r in rows],
                       frame_count=len(rows), writable=True, workspace_scope=None,
                       display_role='DEV', source_session_dir=str(self.workspace),
                       refresh_evaluation=False, force_explicit_object_type=True,
                       active_evaluation_member=False)
        return [('선택 박스의 대상 확인', str(self.workspace), key)], {key: context}

    def load(self, state, path, read_only=False):
        self.current = self.paths[str(Path(path).resolve())]
        self.token = None
        self.queued_key = None
        state.kps_2d, state.keypoint_annotations, state.pose = [None] * 9, None, None
        state.zoom, state.pan = 1., [0, 0]
        if self.reviewer:
            self.token = self.ctx.start(dict(frame_id=self.current, reviewer=self.reviewer))['token']
        return False

    def confirm_profile(self):
        # Shown on the first explicit choice, after the actual boxed image is visible.
        sys.path.insert(0,str(ROOT))
        from scripts.annotate.korean_tk import install
        install()
        import tkinter as tk
        popup = tk.Tk()
        popup.title('첫 저장 한 번만 확인')
        popup.attributes('-topmost', True)
        tk.Label(popup, text=
            '자동 별칭으로 저장합니다. 식별자 입력은 필요 없습니다.\n\n'
            '화면의 선택 박스는 모델이 만든 보조 정보입니다.\n'
            '앞에서 직접 확인한 파렛트와 같은 대상인지 판단해 주세요.\n'
            '이 박스를 본 이력을 기록합니다.', justify='left', padx=20, pady=18).pack()
        seen_annotation = tk.BooleanVar(value=False)
        tk.Checkbutton(popup, text='이 표본의 이전 수동 코너 주석을 본 적 있음',
                       variable=seen_annotation).pack(anchor='w', padx=20)
        accepted = []
        def accept():
            accepted.append(seen_annotation.get())
            popup.destroy()
        tk.Button(popup, text='확인하고 저장 · 다음부터 기억', command=accept,
                  bg='#176b45', fg='white', padx=15, pady=9).pack(fill='x', padx=20, pady=18)
        popup.mainloop()
        if not accepted:
            return False
        corner_profile = REVIEW / 'NATIVE_REVIEWER_PROFILE.json'
        alias = read_json(corner_profile).get('reviewer', {}).get('id', '검수자1') if corner_profile.exists() else '검수자1'
        self.reviewer = dict(id=alias, entered_by='human', confirmation=True,
                             machine_assistance=True, previous_prediction_exposure=True,
                             previous_annotation_exposure=accepted[0],
                             exposure_notes='Raw image and fixed selected box only; previous annotation exposure explicitly confirmed.',
                             confirmed_at=utc_now())
        validate_reviewer(self.reviewer)
        write_json(self.profile_path, self.reviewer)
        return True

    def choose(self, value):
        if not self.reviewer and not self.confirm_profile():
            return False
        if self.token is None:
            self.token = self.ctx.start(dict(frame_id=self.current, reviewer=self.reviewer))['token']
        old = next((r for r in self.ctx.store['records'] if r['frame_id'] == self.current), None)
        self.ctx.save(dict(token=self.token, record=dict(frame_id=self.current,
            decision=CHOICES[value][0], decision_reason='Explicit human button/key selection',
            edit_reason='Human explicitly reselected this frame in the native review tool' if old else '')))
        self.token = None
        self.export()
        return True

    def export(self):
        if not self.ctx.store['records']:
            return
        path = self.ctx.store_path.parent / 'LIFTER_OBJECT_MATCH_REVIEWED.json'
        exported = self.ctx.export()
        if hasattr(self.ctx, 'native_scope_metadata'):
            exported['primary_scope'] = copy.deepcopy(self.ctx.native_scope_metadata)
        write_json(path, exported)
        if self.ctx.counts()['pending'] == 0:
            apply_sidecar(self.ctx.predictions_path, self.ctx.queue_path, path,
                          self.ctx.store_path.parent / 'ALL_STORED_FRAMES_EVALUATOR.jsonl',
                          self.ctx.store_path.parent / 'L4_EVALUATOR_DERIVATION.json')

    def install(self, editor):
        from PIL import Image, ImageDraw, ImageFont
        font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
        font, small = ImageFont.truetype(font_path, 21), ImageFont.truetype(font_path, 17)
        original_wait = editor.cv2.waitKey

        def wait(delay):
            value = original_wait(delay)
            if self.queued_key is not None:
                value, self.queued_key = self.queued_key, None
            return value

        def key(value, state, *unused):
            if value in map(ord, CHOICES):
                try:
                    return 'save-next' if self.choose(chr(value)) else None
                except (GateError, OSError) as err:
                    print('저장 오류:', err, flush=True)
                    editor._toast(state, 'SAVE FAILED - see console')
            if value in (ord('q'), 27):
                self.export()
                return 'quit'
            if value in (ord('p'), ord('n')):
                return 'prev' if value == ord('p') else 'next'
            if value in (ord('s'), 13):
                if any(r['frame_id'] == self.current for r in self.ctx.store['records']):
                    return 'next'
                editor._toast(state, 'Choose 1 / 2 / 3 first; N skips without review')
            return None

        def mouse(event, x, y, flags, state):
            if event == editor.cv2.EVENT_LBUTTONDOWN:
                for box, value in self.buttons:
                    if box[0] <= x <= box[2] and box[1] <= y <= box[3]:
                        self.queued_key = ord(value)
                        break

        def no_pose(state, K, force=False):
            state.pose, state.mode, state.condition_mode, state.line_mode = None, 'click', False, False

        def render(state, frame_idx, total, name):
            raw = editor.cv2.resize(state.img, (960, 720))
            view = Image.new('RGB', (1290, 770), '#17202b')
            view.paste(Image.fromarray(raw[:, :, ::-1]), (0, 50))
            draw = ImageDraw.Draw(view)
            h, w = state.img.shape[:2]
            box = self.ctx.records[self.current]['selected_box_xyxy']
            if box is not None:
                coords = [box[0]*960/w, 50+box[1]*720/h, box[2]*960/w, 50+box[3]*720/h]
                draw.rectangle(coords, outline='#ffff60', width=4)
            draw.text((16, 10), f'대상 확인 · {frame_idx+1}/{total}', font=font, fill='white')
            draw.text((978, 70), '노란 박스가 앞서 확인한', font=font, fill='#a5e4c3')
            draw.text((978, 104), '같은 파렛트인가요?', font=font, fill='#a5e4c3')
            self.buttons = []
            for i, (number, (_, label)) in enumerate(CHOICES.items()):
                box = (978, 174+i*92, 1272, 244+i*92)
                draw.rounded_rectangle(box, radius=8, fill=['#24553b', '#693733', '#655521'][i], outline='#c3d1d9', width=2)
                draw.text((992, 193+i*92), number+'  '+label, font=font, fill='white')
                self.buttons.append((box, number))
            counts = self.ctx.counts()
            old = next((r for r in self.ctx.store['records'] if r['frame_id'] == self.current), None)
            saved = next((label for decision, label in CHOICES.values() if old and decision == old['decision']), '미확인')
            lines = [f'현재: {saved}', f'완료 {counts["reviewed"]} · 남음 {counts["pending"]}',
                     '선택 → 자동 저장 → 다음', 'P 이전 · N 보류하고 다음', 'Q 닫기 · 다시 열면 이어서',
                     '좌표 클릭·식별자 입력 없음', '모델 코너·오차·방법명 비표시']
            scope = getattr(self.ctx, 'native_scope_metadata', {})
            if scope.get('partial_batch'):
                lines[-1] = f'부분 {scope["batch_primary_denominator"]}장 · 전체 완료 아님'
            for i, line in enumerate(lines):
                draw.text((978, 476+i*35), line, font=small, fill='#d9e5ee')
            if counts['pending'] == 0:
                draw.text((16, 727), '대상 확인 완료 · Q 종료 / P 수정', font=font, fill='white')
            return np.asarray(view)[:, :, ::-1].copy()

        editor.WIN, editor.cv2.waitKey = self.window_title, wait
        editor.load_existing_annotation, editor.update_pose = self.load, no_pose
        editor.on_mouse, editor._handle_click_key, editor.render = mouse, key, render


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--batch-plan', type=Path,
                        help='Explicit 12-frame partial batch; original 120/24 plan is preserved')
    args = parser.parse_args()
    try:
        ctx = prepare(args.batch_plan)
    except (GateError, ValueError, OSError) as err:
        print(json.dumps(dict(status='WAITING_HUMAN_PRIMARY_CORNERS', reason=str(err)), ensure_ascii=False), flush=True)
        if not args.prepare_only:
            sys.path.insert(0,str(ROOT))
            from scripts.annotate.korean_tk import install
            install()
            import tkinter as tk
            from tkinter import messagebox
            popup = tk.Tk(); popup.withdraw()
            messagebox.showinfo('먼저 코너 작업을 마쳐 주세요', str(err), parent=popup)
            popup.destroy()
        return 0
    print(json.dumps(dict(status='READY_NATIVE_OBJECT_MATCH', counts=ctx.counts(),
                         snapshot=str(ctx.reviewed_path), editor='scripts/annotate/annotate.py',
                         primary_scope=getattr(ctx, 'native_scope_metadata', None)), ensure_ascii=False), flush=True)
    if args.prepare_only or not ctx.records:
        return 0
    lock = (REVIEW / 'native_object_match.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('대상 확인 창이 이미 열려 있습니다.')
    sys.path.insert(0, str(ROOT / 'scripts/annotate'))
    import annotate
    import annotate_review
    native = NativeObjectReview(ctx)
    native.install(annotate)
    annotate_review.load_review_contexts = native.contexts
    annotate.main(['--review-manifest', str(REVIEW / 'MANIFEST.json'), '--stride', '1',
        '--population-role', 'DEV', '--default_split', 'eval', '--object-type', 'plastic_square',
        '--geometry-registry', str(ROOT / 'challenge/config/CHALLENGE_OBJECT_GEOMETRY_REGISTRY.json'),
        '--win-w', '1290', '--win-h', '830'])
    native.export()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
