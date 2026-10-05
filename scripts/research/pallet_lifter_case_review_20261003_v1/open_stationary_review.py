"""Review genuine stationary intervals from the four raw videos, without predictions.

This separate native viewer never imports inference, control, model outputs or
frozen source. Human decisions and exposure history are saved beside the review
files; the video, frame plan and previous annotations are preserved.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
from pathlib import Path
import sys
import time


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / 'data/pallet/results' / HERE.name
REVIEW = OUT / 'review'
SCHEMA = 'lifter_stationary_review_v1'
SESSION_IDS = ('173507', '174126', '174342', '174925')
WINDOW = 'Pallet raw video - stationary interval review'


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def validate_profile(profile):
    if (not isinstance(profile, dict) or not isinstance(profile.get('id'), str)
            or not profile['id'].strip() or profile.get('entered_by') != 'human'
            or profile.get('confirmation') is not True
            or type(profile.get('previous_prediction_exposure')) is not bool
            or type(profile.get('blind_selection_confirmation')) is not bool
            or not profile.get('confirmed_at')):
        raise ValueError('실제 사람이 별칭과 예측 노출 이력을 한 번 확인해야 합니다.')
    # Prior exposure cannot be neutralized by checking a second box.
    expected = (not profile['previous_prediction_exposure']
                and profile['blind_selection_confirmation'])
    if profile.get('fixed_before_predictions') is not expected:
        raise ValueError('노출 이력과 정지 구간의 블라인드 여부가 일치하지 않습니다.')
    return profile


class StationaryReview:
    def __init__(self, plan_path, video_root, store_path, *, verify_videos=True):
        self.plan_path = Path(plan_path)
        self.video_root = Path(video_root)
        self.store_path = Path(store_path)
        self.plan = json.loads(self.plan_path.read_text(encoding='utf-8'))
        self.plan_hash = sha256(self.plan_path)
        self.frames = {sid: [] for sid in SESSION_IDS}
        for frame in self.plan['frames']:
            sid = frame['session_id']
            if sid not in self.frames:
                raise ValueError('지원하지 않는 원영상 세션: ' + str(sid))
            self.frames[sid].append(frame)
        self.videos, bindings = {}, {}
        for sid, frames in self.frames.items():
            if not frames:
                raise ValueError('원영상 계획이 비어 있습니다: ' + sid)
            if [f['saved_frame_index'] for f in frames] != list(range(len(frames))):
                raise ValueError('저장 프레임 인덱스가 연속적이지 않습니다: ' + sid)
            times = [f['camera_sensor_timestamp_ms'] for f in frames]
            if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in times):
                raise ValueError('잘못된 센서 시각')
            if any(b < a for a, b in zip(times, times[1:])):
                raise ValueError('센서 시각 역전')
            hashes = {f['raw_video_sha256'] for f in frames}
            if len(hashes) != 1:
                raise ValueError('세션 원영상 해시 불일치')
            video = self.video_root / ('forklift_v4_recording_20260901_' + sid + '_raw.mp4')
            if not video.is_file():
                raise ValueError('원영상 없음: ' + str(video))
            expected = next(iter(hashes))
            if verify_videos and sha256(video) != expected:
                raise ValueError('원영상 SHA-256 불일치: ' + sid)
            self.videos[sid] = video
            bindings[sid] = {'raw_video_sha256': expected, 'bytes': video.stat().st_size,
                             'stored_frame_count': len(frames)}
        self.store = dict(schema_version=SCHEMA, source_kind='human_in_progress',
                          plan_sha256=self.plan_hash, video_bindings=bindings,
                          intervals=[], session_reviews={}, action_log=[],
                          draft_interval=None, profile=None,
                          fixed_before_predictions=False,
                          confirmed_no_stop_intervals=False,
                          status='WAITING_HUMAN')
        if self.store_path.exists():
            existing = json.loads(self.store_path.read_text(encoding='utf-8'))
            if (existing.get('schema_version') != SCHEMA
                    or existing.get('plan_sha256') != self.plan_hash
                    or existing.get('video_bindings') != bindings):
                raise ValueError('기존 검수는 다른 원영상 또는 계획에 연결돼 있습니다.')
            self.store = existing
            self._validate_store()

    def _validate_store(self):
        if self.store.get('profile') is not None:
            validate_profile(self.store['profile'])
        ids = set()
        for interval in self.store['intervals']:
            if interval['interval_id'] in ids:
                raise ValueError('중복 정지 구간 ID')
            ids.add(interval['interval_id'])
            sid = interval['session_id']
            a, b = interval['saved_start_index'], interval['saved_end_index']
            self._validate_range(sid, a, b)
            if (interval['sensor_start_ms'] != self.frames[sid][a]['camera_sensor_timestamp_ms']
                    or interval['sensor_end_ms'] != self.frames[sid][b]['camera_sensor_timestamp_ms']
                    or interval.get('source_kind') != 'human_reviewed'
                    or not interval.get('reviewer_id') or not interval.get('reviewed_at')
                    or not interval.get('evidence')):
                raise ValueError('정지 구간의 시각 또는 실제 사람 근거 불일치')
            validate_profile(interval['reviewer'])
        for sid, review in self.store.get('session_reviews', {}).items():
            if (sid not in self.frames or review.get('decision') not in ('intervals_reviewed', 'no_stop')
                    or review.get('source_kind') != 'human_reviewed'
                    or not review.get('reviewed_at')):
                raise ValueError('실제 세션 검수 기록 불일치')
            validate_profile(review['reviewer'])
            if review['decision'] == 'no_stop' and any(i['session_id'] == sid for i in self.store['intervals']):
                raise ValueError('정지 없음 확인에 정지 구간이 존재합니다.')
        if self.store.get('draft_interval') is not None:
            draft = self.store['draft_interval']
            self.frame(draft['session_id'], draft['saved_start_index'])
            validate_profile(draft['reviewer'])

    def frame(self, sid, index):
        if sid not in self.frames or type(index) is not int or not 0 <= index < len(self.frames[sid]):
            raise ValueError('원영상 범위를 벗어났습니다.')
        return self.frames[sid][index]

    def _validate_range(self, sid, start, end):
        a, b = self.frame(sid, start), self.frame(sid, end)
        if end <= start or b['camera_sensor_timestamp_ms'] <= a['camera_sensor_timestamp_ms']:
            raise ValueError('정지 끝은 시작보다 뒤여야 합니다. 두 개 이상 다른 시각의 프레임을 확인하세요.')

    def set_profile(self, profile):
        self.store['profile'] = copy.deepcopy(validate_profile(profile))
        self._persist('human_profile_confirmation')

    def _profile(self):
        return copy.deepcopy(validate_profile(self.store.get('profile')))

    def _persist(self, action, **details):
        profiles = [i['reviewer'] for i in self.store['intervals']]
        profiles.extend(r['reviewer'] for r in self.store['session_reviews'].values())
        self.store['fixed_before_predictions'] = bool(profiles) and all(p['fixed_before_predictions'] for p in profiles)
        done = len(self.store['session_reviews']) == len(SESSION_IDS)
        self.store['confirmed_no_stop_intervals'] = (done and not self.store['intervals']
            and all(r['decision'] == 'no_stop' for r in self.store['session_reviews'].values()))
        self.store['source_kind'] = 'human_reviewed' if profiles else 'human_in_progress'
        self.store['status'] = ('HUMAN_REVIEW_SUBMITTED' if done else 'WAITING_HUMAN')
        self.store['strict_evaluator_status'] = ('ELIGIBLE' if profiles and self.store['fixed_before_predictions']
                                                else 'WAITING_HUMAN' if not profiles else 'BLOCKED_EXPOSURE')
        self.store['updated_at'] = utc_now()
        self.store['action_log'].append(dict(action=action, performed_at=utc_now(), **details))
        # Append the replaced revision before atomic replacement; source media is never touched.
        if self.store_path.exists():
            history = self.store_path.with_suffix('.history.jsonl')
            previous = json.loads(self.store_path.read_text(encoding='utf-8'))
            with history.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(dict(superseded_at=utc_now(), action=action, previous=previous),
                                        ensure_ascii=False, allow_nan=False) + '\n')
        write_json(self.store_path, self.store)

    def start(self, sid, index):
        frame, reviewer = self.frame(sid, index), self._profile()
        if self.store.get('draft_interval') is not None:
            raise ValueError('시작한 구간이 있습니다. 끝 저장 또는 시작 취소를 먼저 누르세요.')
        self.store['draft_interval'] = dict(session_id=sid, saved_start_index=index,
            sensor_start_ms=frame['camera_sensor_timestamp_ms'], started_at=utc_now(), reviewer=reviewer)
        self._persist('human_stop_start', session_id=sid, saved_frame_index=index)

    def cancel_start(self):
        if self.store.get('draft_interval') is None:
            return False
        old = self.store['draft_interval']
        self.store['draft_interval'] = None
        self._persist('human_cancel_stop_start', previous_start=old)
        return True

    def finish(self, sid, index):
        draft = self.store.get('draft_interval')
        if draft is None or draft['session_id'] != sid:
            raise ValueError('이 영상에서 정지 시작을 먼저 선택하세요.')
        start = draft['saved_start_index']
        self._validate_range(sid, start, index)
        if any(i['session_id'] == sid and start <= i['saved_end_index']
               and index >= i['saved_start_index'] for i in self.store['intervals']):
            raise ValueError('기존 구간과 겹칩니다. 기존 구간을 확인하거나 삭제하세요.')
        reviewer = copy.deepcopy(draft['reviewer'])
        interval = dict(interval_id=sid + ':' + str(start) + '-' + str(index), session_id=sid,
            saved_start_index=start, saved_end_index=index,
            sensor_start_ms=self.frame(sid, start)['camera_sensor_timestamp_ms'],
            sensor_end_ms=self.frame(sid, index)['camera_sensor_timestamp_ms'],
            source_kind='human_reviewed', reviewer_id=reviewer['id'], reviewer=reviewer,
            reviewed_at=utc_now(), evidence='사람이 모델 출력을 표시하지 않은 원영상에서 실제 정지 시작과 끝을 직접 확인',
            raw_video_sha256=self.frames[sid][0]['raw_video_sha256'],
            fixed_before_predictions=reviewer['fixed_before_predictions'],
            selection_started_at=draft['started_at'])
        self.store['intervals'].append(interval)
        self.store['draft_interval'] = None
        self.store['session_reviews'].pop(sid, None)
        self._persist('human_stop_end_and_save', interval_id=interval['interval_id'])
        return interval

    def complete_session(self, sid, no_stop=False):
        self.frame(sid, 0)
        reviewer = self._profile()
        if self.store.get('draft_interval') is not None:
            raise ValueError('시작한 구간의 끝을 저장하거나 취소하세요.')
        intervals = [i for i in self.store['intervals'] if i['session_id'] == sid]
        if no_stop and intervals:
            raise ValueError('이 영상에 저장된 정지 구간이 있습니다.')
        if not no_stop and not intervals:
            raise ValueError('정지 구간을 저장하거나, 직접 확인한 뒤 정지 없음 버튼을 누르세요.')
        self.store['session_reviews'][sid] = dict(session_id=sid,
            decision='no_stop' if no_stop else 'intervals_reviewed', source_kind='human_reviewed',
            reviewer=reviewer, reviewer_id=reviewer['id'], reviewed_at=utc_now(),
            evidence='사람이 해당 원영상을 직접 확인하고 세션 검수 완료 버튼을 선택함',
            raw_video_sha256=self.frames[sid][0]['raw_video_sha256'])
        self._persist('human_session_no_stop' if no_stop else 'human_session_completed', session_id=sid)

    def delete_last(self, sid):
        matching = [i for i in self.store['intervals'] if i['session_id'] == sid]
        if not matching:
            raise ValueError('삭제할 저장 구간이 없습니다.')
        old = matching[-1]
        self.store['intervals'].remove(old)
        self.store['session_reviews'].pop(sid, None)
        self._persist('human_delete_last_interval', removed_interval=old,
                      edit_reason='사람이 현재 영상의 마지막 구간 삭제 버튼을 선택함')

    def counts(self):
        return dict(total_sessions=4, completed_sessions=len(self.store['session_reviews']),
                    interval_count=len(self.store['intervals']),
                    strict_evaluator_status=self.store.get('strict_evaluator_status', 'WAITING_HUMAN'))


def choose_profile(preview, previous=None):
    """One genuine confirmation; no implicit true for blindness or exposure."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.annotate.korean_tk import install
    install()
    import tkinter as tk
    from PIL import Image, ImageTk
    window = tk.Tk()
    window.title('정지 구간 검수 · 처음 한 번만 확인')
    window.attributes('-topmost', True)
    frame = tk.Frame(window, padx=18, pady=16)
    frame.pack(fill='both', expand=True)
    picture = Image.fromarray(preview[:, :, ::-1])
    picture.thumbnail((560, 270))
    photo = ImageTk.PhotoImage(picture)
    tk.Label(frame, image=photo).pack()
    tk.Label(frame, text='보고 있는 원영상으로 직접 정지 구간을 선택합니다.',
             font=('', 13, 'bold')).pack(anchor='w', pady=7)
    alias = (previous or {}).get('id', '검수자1')
    tk.Label(frame, text='별칭은 자동으로 준비했습니다. 식별자를 입력할 필요 없습니다.').pack(anchor='w')
    tk.Label(frame, text='이번 검수 별칭: ' + alias).pack(anchor='w', pady=4)
    different_person = tk.BooleanVar(value=False)
    tk.Checkbutton(frame, text='다른 사람이 검수함 · 새 별칭을 자동으로 만듭니다.',
                   variable=different_person).pack(anchor='w')
    exposure = tk.StringVar(value='unanswered')
    tk.Label(frame, text='이 리프터 영상의 모델 예측(상자·점·결과 그래프)을 이미 보았습니까?').pack(anchor='w', pady=(9, 2))
    tk.Radiobutton(frame, text='보지 않았음', variable=exposure, value='no').pack(anchor='w')
    tk.Radiobutton(frame, text='본 적 있음 / 확실하지 않음', variable=exposure, value='yes').pack(anchor='w')
    blind = tk.BooleanVar(value=False)
    tk.Checkbutton(frame, text='예측을 보기 전에 원영상만으로 구간을 선택·확정합니다.', variable=blind).pack(anchor='w', pady=4)
    tk.Label(frame, text='예측을 이미 본 경우에도 구간 저장은 가능합니다.\n그 경우 정지 잡음 표는 기존 블라인드 규약을 충족하지 않아 보류됩니다.',
             justify='left', wraplength=560).pack(anchor='w', pady=5)
    error = tk.Label(frame, text='', fg='#c03333')
    error.pack(anchor='w')
    result = []
    def accept():
        if exposure.get() == 'unanswered':
            error.configure(text='예측 노출 여부를 직접 선택하세요. 식별자 입력은 필요 없습니다.')
            return
        prior = exposure.get() == 'yes'
        confirmed_alias = ('검수자_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
                           if different_person.get() else alias)
        profile = dict(id=confirmed_alias, entered_by='human', confirmation=True,
            confirmed_at=utc_now(), previous_prediction_exposure=prior,
            blind_selection_confirmation=blind.get(), fixed_before_predictions=not prior and blind.get(),
            machine_assistance=False, exposure_notes='원영상 전용 도구에서 사람이 자동 별칭과 실제 노출 이력을 직접 확인')
        result.append(validate_profile(profile))
        window.destroy()
    tk.Button(frame, text='직접 확인 · 다음부터 기억', command=accept,
              bg='#176b45', fg='white', pady=8).pack(fill='x', pady=6)
    window.mainloop()
    return result[0] if result else None


class Viewer:
    def __init__(self, review, *, change_reviewer=False):
        import cv2
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        self.cv2, self.np, self.Image, self.ImageDraw = cv2, np, Image, ImageDraw
        font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
        self.font = ImageFont.truetype(font_path, 19)
        self.small = ImageFont.truetype(font_path, 16)
        self.big = ImageFont.truetype(font_path, 22)
        self.review, self.cap = review, None
        self.sid = next((s for s in SESSION_IDS if s not in review.store['session_reviews']), SESSION_IDS[0])
        self.index, self.image, self.loaded_index = 0, None, None
        self.playing, self.speed, self.next_due = False, 1., time.monotonic()
        self.message = '원영상이 보입니다. 실제 멈춘 구간만 시작과 끝을 선택하세요.'
        self.regions, self.setting_slider, self.dirty_display = [], False, True
        self.change_reviewer = change_reviewer
        self.profile_path = review.store_path.parent / 'STATIONARY_REVIEWER_PROFILE.json'
        if not change_reviewer and review.store.get('profile') is None and self.profile_path.exists():
            review.store['profile'] = validate_profile(json.loads(self.profile_path.read_text()))

    def open_session(self, sid):
        self.playing = False
        capture = self.cv2.VideoCapture(str(self.review.videos[sid]))
        if not capture.isOpened() or int(capture.get(self.cv2.CAP_PROP_FRAME_COUNT)) != len(self.review.frames[sid]):
            capture.release()
            raise ValueError('원영상 프레임 수가 계획과 다릅니다: ' + sid)
        success, frame = capture.read()
        if not success or hashlib.sha256(frame.tobytes()).hexdigest() != self.review.frame(sid, 0)['decoded_bgr_sha256']:
            capture.release()
            raise ValueError('원영상 첫 프레임의 픽셀 해시가 계획과 다릅니다: ' + sid)
        if self.cap is not None:
            self.cap.release()
        self.cap, self.image, self.sid = capture, frame, sid
        self.index, self.loaded_index, self.dirty_display = 0, 0, True
        self.setting_slider = True
        self.cv2.setTrackbarMax('Frame', WINDOW, len(self.review.frames[sid]) - 1)
        self.cv2.setTrackbarPos('Frame', WINDOW, 0)
        self.setting_slider = False

    def read_frame(self):
        if self.loaded_index == self.index:
            return
        if self.loaded_index is None or self.index != self.loaded_index + 1:
            self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, self.index)
        success, frame = self.cap.read()
        if not success:
            raise ValueError('원영상 프레임을 읽을 수 없습니다.')
        planned = self.review.frame(self.sid, self.index)
        if hashlib.sha256(frame.tobytes()).hexdigest() != planned['decoded_bgr_sha256']:
            raise ValueError('표시 프레임 픽셀 해시가 frozen 계획과 다릅니다: ' + planned['frame_id'])
        self.image, self.loaded_index, self.dirty_display = frame, self.index, True

    def seek(self, index):
        previous_index = self.index
        self.index = max(0, min(len(self.review.frames[self.sid]) - 1, int(index)))
        try:
            self.read_frame()
        except ValueError:
            # Never leave a new sensor index paired with the old displayed image.
            self.index, self.loaded_index, self.playing = previous_index, None, False
            self.setting_slider = True
            self.cv2.setTrackbarPos('Frame', WINDOW, previous_index)
            self.setting_slider = False
            raise
        self.setting_slider = True
        self.cv2.setTrackbarPos('Frame', WINDOW, self.index)
        self.setting_slider = False

    def slider(self, index):
        if not self.setting_slider and self.cap is not None:
            self.playing = False
            try:
                self.seek(index)
            except ValueError as error:
                self.message, self.dirty_display = str(error), True

    def ensure_profile(self):
        if self.review.store.get('profile') is not None and not self.change_reviewer:
            return True
        profile = choose_profile(self.image, self.review.store.get('profile'))
        if profile is None:
            self.message = '확인을 취소했습니다. 원영상 확인과 이동은 계속 가능합니다.'
            return False
        self.review.set_profile(profile)
        write_json(self.profile_path, profile)
        self.change_reviewer = False
        return True

    def action(self, command):
        try:
            if command.startswith('session:'):
                self.open_session(command.split(':', 1)[1])
            elif command == 'play':
                self.playing = not self.playing
                next_index = min(self.index+1, len(self.review.frames[self.sid])-1)
                delay = (self.review.frame(self.sid, next_index)['camera_sensor_timestamp_ms']
                         - self.review.frame(self.sid, self.index)['camera_sensor_timestamp_ms'])
                self.next_due = time.monotonic() + max(1., delay) / 1000 / self.speed
            elif command == 'speed':
                speeds = (1., 2., 4., .5)
                self.speed = speeds[(speeds.index(self.speed) + 1) % len(speeds)]
            elif command in ('back', 'forward'):
                self.playing = False
                self.seek(self.index + (-1 if command == 'back' else 1))
            elif command == 'cancel':
                self.review.cancel_start()
                self.message = '정지 시작 선택을 취소했습니다.'
            elif command in ('start', 'end', 'no_stop', 'complete', 'delete'):
                self.playing = False
                if not self.ensure_profile():
                    return
                if command == 'start':
                    self.review.start(self.sid, self.index)
                    self.message = '시작 선택 완료. 원영상을 확인한 뒤 끝 위치에서 끝 저장을 누르세요.'
                elif command == 'end':
                    interval = self.review.finish(self.sid, self.index)
                    self.message = f"구간 저장: {(interval['sensor_end_ms'] - interval['sensor_start_ms']) / 1000:.2f}초"
                elif command == 'delete':
                    self.review.delete_last(self.sid)
                    self.message = '이 영상의 마지막 저장 구간을 삭제했습니다. 이전 기록은 보존했습니다.'
                else:
                    self.review.complete_session(self.sid, no_stop=command == 'no_stop')
                    self.message = '영상 검수 완료. 위쪽 버튼에서 다음 영상을 선택하세요.'
            self.dirty_display = True
        except ValueError as error:
            self.message, self.dirty_display = str(error), True

    def mouse(self, event, x, y, flags, parameter):
        if event == self.cv2.EVENT_LBUTTONDOWN:
            for rectangle, action in self.regions:
                a, b, c, d = rectangle
                if a <= x <= c and b <= y <= d:
                    self.action(action)
                    break

    def render(self):
        canvas = self.Image.new('RGB', (1260, 900), '#14202b')
        draw = self.ImageDraw.Draw(canvas)
        self.regions = []
        def text(x, y, label, fill='#eaf1f5', font=None):
            draw.text((x, y), label, font=font or self.font, fill=fill)
        def button(rect, label, action, active=False, color=None):
            draw.rounded_rectangle(rect, radius=7, fill=color or ('#276a55' if active else '#30475d'),
                                   outline='#64849a')
            text(rect[0] + 12, rect[1] + 10, label)
            self.regions.append((rect, action))
        text(16, 10, '원영상에서 실제 정지 구간 확인', font=self.big)
        text(755, 13, '예측 · 코너 · 성능 정보는 표시하지 않습니다.', font=self.small)
        for n, sid in enumerate(SESSION_IDS):
            done = sid in self.review.store['session_reviews']
            button((16+n*306, 51, 309+n*306, 96), f"영상 {n+1}  {'완료' if done else '대기'}", 'session:' + sid,
                   active=sid == self.sid)
        picture = self.Image.fromarray(self.image[:, :, ::-1]).resize((832, 624))
        canvas.paste(picture, (16, 108))
        frame = self.review.frame(self.sid, self.index)
        elapsed = (frame['camera_sensor_timestamp_ms'] - self.review.frames[self.sid][0]['camera_sensor_timestamp_ms']) / 1000
        text(16, 741, f'현재 {elapsed:.2f}초 · {self.index+1}/{len(self.review.frames[self.sid])} 프레임 · 재생 {self.speed:g}배')
        button((16, 783, 185, 832), '일시정지' if self.playing else '▶ 재생', 'play')
        button((198, 783, 345, 832), '재생 속도', 'speed')
        button((358, 783, 505, 832), '이전 프레임', 'back')
        button((518, 783, 665, 832), '다음 프레임', 'forward')
        x = 868
        button((x, 108, 1244, 163), 'B  정지 시작', 'start', color='#285979')
        button((x, 175, 1244, 230), 'E  정지 끝 · 구간 저장', 'end', color='#276a55')
        button((x, 242, 1244, 287), 'C  시작 선택 취소', 'cancel')
        button((x, 305, 1244, 361), 'F  이 영상 검수 완료', 'complete')
        button((x, 373, 1244, 429), 'N  직접 확인함 · 정지 없음', 'no_stop')
        text(x, 443, '실제 원영상에서 확인한 경우에만 선택.', font=self.small)
        text(x, 469, '시작/끝은 센서 시각에 연결해 저장합니다.', font=self.small)
        draft = self.review.store.get('draft_interval')
        if draft:
            text(x, 507, f"선택한 시작: 영상 {SESSION_IDS.index(draft['session_id'])+1}", fill='#ffe28a')
            text(x, 535, f"프레임 {draft['saved_start_index']+1}", fill='#ffe28a')
        rows = [i for i in self.review.store['intervals'] if i['session_id'] == self.sid]
        text(x, 579, f'이 영상에 저장된 구간 {len(rows)}개')
        for n, interval in enumerate(rows[-4:]):
            start = (interval['sensor_start_ms'] - self.review.frames[self.sid][0]['camera_sensor_timestamp_ms']) / 1000
            end = (interval['sensor_end_ms'] - self.review.frames[self.sid][0]['camera_sensor_timestamp_ms']) / 1000
            text(x, 612+n*27, f'{start:.2f} ~ {end:.2f}초', font=self.small)
        button((x, 742, 1244, 784), 'D  이 영상 마지막 구간 삭제', 'delete')
        counts = self.review.counts()
        text(x, 799, f"전체 완료 {counts['completed_sessions']}/4 · 구간 {counts['interval_count']}개", font=self.small)
        if counts['strict_evaluator_status'] == 'BLOCKED_EXPOSURE':
            text(x, 826, '블라인드 조건 미확인: 정지 평가 보류', fill='#ffb985', font=self.small)
        else:
            text(x, 826, '별칭·노출 확인은 처음 한 번만 합니다.', font=self.small)
        text(16, 854, self.message[:88], fill='#ffe28a', font=self.small)
        return self.np.asarray(canvas)[:, :, ::-1].copy()

    def run(self):
        self.cv2.namedWindow(WINDOW, self.cv2.WINDOW_AUTOSIZE)
        self.cv2.createTrackbar('Frame', WINDOW, 0, len(self.review.frames[self.sid])-1, self.slider)
        self.cv2.setMouseCallback(WINDOW, self.mouse)
        self.open_session(self.sid)
        try:
            while True:
                if self.playing and time.monotonic() >= self.next_due:
                    if self.index == len(self.review.frames[self.sid])-1:
                        self.playing = False
                        self.dirty_display = True
                    else:
                        self.seek(self.index+1)
                        next_index = min(self.index+1, len(self.review.frames[self.sid])-1)
                        a = self.review.frame(self.sid, self.index)['camera_sensor_timestamp_ms']
                        b = self.review.frame(self.sid, next_index)['camera_sensor_timestamp_ms']
                        self.next_due = time.monotonic() + max(1., b-a) / 1000 / self.speed
                if self.dirty_display:
                    self.cv2.imshow(WINDOW, self.render())
                    self.dirty_display = False
                key = self.cv2.waitKeyEx(12)
                if key in (27, ord('q')) or self.cv2.getWindowProperty(WINDOW, self.cv2.WND_PROP_VISIBLE) < 1:
                    break
                if ord('A') <= key <= ord('Z'):
                    key += ord('a') - ord('A')
                actions = {ord(' '): 'play', ord('b'): 'start', ord('e'): 'end', ord('c'): 'cancel',
                           ord('f'): 'complete', ord('n'): 'no_stop', ord('d'): 'delete',
                           ord(','): 'back', ord('.'): 'forward', ord('v'): 'speed',
                           65361: 'back', 65363: 'forward'}
                if key in actions:
                    self.action(actions[key])
                elif ord('1') <= key <= ord('4'):
                    self.action('session:' + SESSION_IDS[key-ord('1')])
        finally:
            if self.cap is not None:
                self.cap.release()
            self.cv2.destroyWindow(WINDOW)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=OUT/'LIFTER_EVALUATION_PLAN.json')
    parser.add_argument('--video-root', type=Path, default=ROOT/'extracted/depth_cam/rec')
    parser.add_argument('--store', type=Path, default=REVIEW/'STATIONARY_INTERVALS_REVIEWED.json')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--change-reviewer', action='store_true')
    args = parser.parse_args(argv)
    review = StationaryReview(args.plan, args.video_root, args.store)
    print(json.dumps(dict(status='READY_RAW_VIDEO_REVIEW', counts=review.counts(),
        source='raw_video_only', prediction_files_loaded=0, plan_sha256=review.plan_hash,
        output=str(args.store), sessions=[dict(session_id=sid, frames=len(review.frames[sid]),
        duration_seconds=(review.frames[sid][-1]['camera_sensor_timestamp_ms']-
                          review.frames[sid][0]['camera_sensor_timestamp_ms'])/1000,
        video=str(review.videos[sid])) for sid in SESSION_IDS]), ensure_ascii=False), flush=True)
    if args.prepare_only:
        return 0
    args.store.parent.mkdir(parents=True, exist_ok=True)
    lock = (args.store.parent/'stationary_editor.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('정지 구간 검수 창이 이미 실행 중입니다.')
    Viewer(review, change_reviewer=args.change_reviewer).run()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
