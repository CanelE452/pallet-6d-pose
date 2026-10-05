#!/usr/bin/env python3
"""Local-only manual reference review. No model, camera, CAN, or control imports."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import mimetypes
import os
import struct
import threading
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

SCHEMA = 'lifter_reference_review_v1'
AXES = ('external_occlusion', 'self_occlusion', 'out_of_frame', 'definition_uncertain')


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def image_dimensions(path):
    with Path(path).open('rb') as stream:
        header = stream.read(24)
        if header[:8] == b'\x89PNG\r\n\x1a\n' and header[12:16] == b'IHDR':
            return struct.unpack('>II', header[16:24])
        stream.seek(0)
        if stream.read(2) == b'\xff\xd8':
            while True:
                byte = stream.read(1)
                if not byte:
                    break
                if byte != b'\xff':
                    continue
                marker = stream.read(1)
                while marker == b'\xff':
                    marker = stream.read(1)
                if marker in (b'\xd8', b'\x01') or b'\xd0' <= marker <= b'\xd7':
                    continue
                if marker in (b'\xd9', b'\xda'):
                    break
                raw_length = stream.read(2)
                require(len(raw_length) == 2, 'Truncated JPEG')
                length = struct.unpack('>H', raw_length)[0]
                if marker in [bytes([x]) for x in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF)]:
                    data = stream.read(5)
                    require(len(data) == 5, 'Truncated JPEG dimensions')
                    height, width = struct.unpack('>HH', data[1:])
                    return width, height
                require(length >= 2, 'Invalid JPEG segment')
                stream.seek(length - 2, 1)
    raise ValidationError('Only raw PNG/JPEG images are supported: ' + str(path))


def validate_reviewer(reviewer):
    require(isinstance(reviewer, dict), 'Reviewer must be manually entered')
    require(isinstance(reviewer.get('id'), str) and reviewer['id'].strip(), 'Reviewer ID is required')
    require(reviewer.get('entered_by') == 'human', 'Reviewer entered_by must be human')
    for key in ('machine_assistance', 'previous_prediction_exposure', 'previous_annotation_exposure'):
        require(type(reviewer.get(key)) is bool, 'Explicit reviewer answer required: ' + key)
    require(isinstance(reviewer.get('exposure_notes'), str), 'Exposure notes must be text')
    require(reviewer.get('confirmation') is True, 'Human provenance confirmation required')


class Context:
    def __init__(self, manifest, plan, contract, store):
        self.manifest_path = Path(manifest).resolve()
        self.plan_path = Path(plan).resolve()
        self.contract_path = Path(contract).resolve()
        self.store_path = Path(store).resolve()
        for path in (self.manifest_path, self.plan_path, self.contract_path):
            require(path.is_file(), 'BLOCKED_REFERENCE: missing input ' + str(path))
        self.manifest = read_json(self.manifest_path)
        self.contract = read_json(self.contract_path)
        require(self.manifest.get('schema_version') == 'lifter_review_manifest_v1', 'Bad manifest schema')
        require(self.contract.get('schema_version') == 'lifter_corner_contract_v1', 'Bad corner contract schema')
        require(self.contract.get('version'), 'BLOCKED_CONTRACT: missing corner version')
        require(self.contract.get('object_definition') and self.contract.get('source'), 'BLOCKED_CONTRACT: definition/source missing')
        require(self.contract.get('direct_click_policy') in ('human_confirmation_required', 'confirmed_physical_corners'), 'BLOCKED_CONTRACT: unsupported direct-click policy')
        corners = self.contract.get('corners', [])
        require(len(corners) == 8 and {x.get('id') for x in corners} == set(range(8)), 'BLOCKED_CONTRACT: external canonical IDs must be 0 through 7 exactly once')
        require(all(type(x['id']) is int and x.get('name') and x.get('definition') for x in corners), 'BLOCKED_CONTRACT: incomplete canonical definitions')
        self.bindings = {'manifest_sha256': sha256(self.manifest_path), 'plan_sha256': sha256(self.plan_path), 'corner_contract_sha256': sha256(self.contract_path), 'corner_definition_version': self.contract['version']}
        for key in ('plan_sha256', 'corner_contract_sha256'):
            require(self.manifest.get(key) == self.bindings[key], 'Hash mismatch: ' + key)
        self.frames = {}
        self.images = {}
        frames = self.manifest.get('frames')
        require(isinstance(frames, list) and frames, 'BLOCKED_REFERENCE: no selected raw frames')
        for frame in frames:
            frame_id = frame.get('frame_id')
            require(isinstance(frame_id, str) and frame_id and frame_id not in self.frames, 'Duplicate/invalid frame ID')
            require(isinstance(frame.get('session_id'), str) and frame['session_id'], 'Missing session ID')
            require(type(frame.get('saved_frame_index')) is int and frame['saved_frame_index'] >= 0, 'Bad stored frame index')
            require(finite(frame.get('camera_sensor_timestamp_ms')), 'Bad sensor timestamp')
            require(type(frame.get('repeat_review')) is bool, 'Repeat selection must be frozen')
            require(all(type(frame.get(k)) is int and frame[k] > 0 for k in ('width', 'height')), 'Bad image dimensions')
            image_path = (self.manifest_path.parent / frame['image_path']).resolve()
            require(image_path.is_file(), 'BLOCKED_REFERENCE: missing raw image ' + str(image_path))
            require(sha256(image_path) == frame.get('image_sha256'), 'Raw image hash mismatch: ' + frame_id)
            require(image_dimensions(image_path) == (frame['width'], frame['height']), 'Raw image dimensions mismatch: ' + frame_id)
            self.frames[frame_id] = frame
            self.images[frame_id] = image_path
        self.lock = threading.Lock()
        self.sessions = {}
        self.store = self.empty_store()
        if self.store_path.exists():
            self.store = self.validate_bundle(read_json(self.store_path), allow_drafts=True)

    def empty_store(self):
        return {'schema_version': SCHEMA, 'source_kind': 'human_in_progress', 'bindings': copy.deepcopy(self.bindings), 'records': []}

    def validate_record(self, record, allow_draft=False):
        require(isinstance(record, dict), 'Bad record')
        frame = self.frames.get(record.get('frame_id'))
        require(frame is not None, 'Unplanned frame ID')
        require(record.get('session_id') == frame['session_id'], 'Session mismatch')
        require(record.get('review_pass') in ('primary', 'repeat'), 'Bad review pass')
        require(record['review_pass'] != 'repeat' or frame['repeat_review'], 'Unplanned repeat frame')
        for key in ('image_sha256', 'width', 'height'):
            require(record.get(key) == frame[key], 'Frame binding mismatch: ' + key)
        for key in ('plan_sha256', 'corner_contract_sha256', 'corner_definition_version'):
            require(record.get(key) == self.bindings[key], 'Record binding mismatch: ' + key)
        status = record.get('status')
        require(status in ('draft', 'reviewed', 'skipped'), 'Bad review status')
        if status == 'draft':
            require(allow_draft and record.get('source_kind') == 'human_in_progress', 'Incomplete/unreviewed record rejected')
        else:
            require(record.get('source_kind') == 'human_reviewed', 'Unreviewed source rejected')
        validate_reviewer(record.get('reviewer'))
        review_time = record.get('review_time', {})
        require(review_time.get('clock_source') == 'server_wall_and_monotonic', 'Actual review clock provenance required')
        require(finite(review_time.get('duration_seconds')) and review_time['duration_seconds'] >= 0, 'Bad review duration')
        try:
            start = datetime.fromisoformat(review_time['started_at'].replace('Z', '+00:00'))
            end = datetime.fromisoformat(review_time['finished_at'].replace('Z', '+00:00'))
            require(start.tzinfo is not None and end.tzinfo is not None and end >= start, 'Invalid review time')
        except (KeyError, ValueError, TypeError) as error:
            raise ValidationError('Missing/invalid actual review timestamps') from error
        require(isinstance(record.get('skip_reason', ''), str), 'Bad skip reason')
        require(isinstance(record.get('edit_reason', ''), str), 'Bad edit reason')
        if status == 'skipped':
            require(record['skip_reason'].strip(), 'Skip reason required')
            require(record.get('corners') == [], 'Skipped frame cannot carry reference clicks')
            return record
        obj = record.get('object', {})
        require(obj.get('presence') in ('present', 'absent', 'uncertain') or (allow_draft and obj.get('presence') is None), 'Object presence is unreviewed')
        require(type(obj.get('target_identity_confirmed')) is bool, 'Explicit target identity flag required')
        require(obj.get('target_object_id') is None or isinstance(obj['target_object_id'], str), 'Bad target object ID')
        if obj['presence'] == 'present' and obj['target_identity_confirmed']:
            require(obj.get('target_object_id') and obj['target_object_id'].strip(), 'Confirmed target needs an object ID')
        corners = record.get('corners')
        require(isinstance(corners, list) and len(corners) == 8, 'All eight canonical corner states required')
        require({c.get('id') for c in corners} == set(range(8)) and all(type(c.get('id')) is int for c in corners), 'Duplicate/missing canonical corner ID')
        for corner in corners:
            visibility = corner.get('visibility')
            require(visibility in ('direct_visible', 'not_direct_visible', 'uncertain') or (allow_draft and visibility is None), 'Unreviewed corner state')
            require(all(type(corner.get(k)) is bool or (allow_draft and corner.get(k) is None) for k in AXES), 'Occlusion axes must be separate explicit booleans')
            require(type(corner.get('definition_confirmed')) is bool, 'Physical corner confirmation required')
            require(isinstance(corner.get('reason'), str), 'Corner reason must be text')
            if visibility == 'direct_visible':
                require(obj['presence'] == 'present' and obj['target_identity_confirmed'], 'Direct visible requires confirmed present target')
                require(corner['definition_confirmed'] and not any(corner[k] for k in AXES), 'Direct click cannot be occluded/outside/unconfirmed')
                if not (status == 'draft' and allow_draft and corner.get('x') is None and corner.get('y') is None):
                    require(finite(corner.get('x')) and finite(corner.get('y')), 'Visible corner has no valid coordinates')
                    require(0 <= corner['x'] < frame['width'] and 0 <= corner['y'] < frame['height'], 'Coordinates outside original image')
            else:
                require(corner.get('x') is None and corner.get('y') is None, 'Non-visible corner cannot carry guessed coordinates')
                if not allow_draft or status != 'draft':
                    require(any(corner[k] for k in AXES) or corner['reason'].strip(), 'Non-visible state needs an axis or reason')
        return record

    def validate_bundle(self, bundle, allow_drafts=False):
        require(bundle.get('schema_version') == SCHEMA, 'Wrong reference schema')
        require(bundle.get('bindings') == self.bindings, 'Plan/contract/manifest binding mismatch')
        require(bundle.get('source_kind') in ('human_in_progress', 'human_reviewed') if allow_drafts else bundle.get('source_kind') == 'human_reviewed', 'Unreviewed bundle rejected')
        require(isinstance(bundle.get('records'), list), 'Records must be a list')
        seen = set()
        for record in bundle['records']:
            self.validate_record(record, allow_draft=allow_drafts)
            key = (record['frame_id'], record['review_pass'])
            require(key not in seen, 'Duplicate frame/pass record')
            seen.add(key)
        return copy.deepcopy(bundle)

    def counts(self):
        result = {'total_primary': len(self.frames), 'total_repeat': sum(f['repeat_review'] for f in self.frames.values()), 'primary_reviewed': 0, 'repeat_reviewed': 0, 'skipped': 0, 'drafts': 0, 'visible_corners': 0}
        for record in self.store['records']:
            if record['status'] == 'draft':
                result['drafts'] += 1
            else:
                result[record['review_pass'] + '_reviewed'] += 1
                result['skipped'] += record['status'] == 'skipped'
                result['visible_corners'] += sum(c['visibility'] == 'direct_visible' for c in record['corners'])
        result['incomplete_primary'] = result['total_primary'] - result['primary_reviewed']
        result['incomplete_repeat'] = result['total_repeat'] - result['repeat_reviewed']
        return result

    def record_for(self, frame_id, review_pass):
        return next((copy.deepcopy(r) for r in self.store['records'] if r['frame_id'] == frame_id and r['review_pass'] == review_pass), None)

    def atomic_write(self):
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.store_path.with_name(self.store_path.name + '.' + uuid.uuid4().hex + '.tmp')
        temporary.write_text(json.dumps(self.store, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        os.replace(temporary, self.store_path)

    def catalog(self):
        statuses = {r['frame_id'] + '|' + r['review_pass']: r['status'] for r in self.store['records']}
        safe_frames = [{k: v for k, v in f.items() if k != 'image_path'} for f in self.frames.values()]
        return {'bindings': self.bindings, 'contract': self.contract, 'frames': safe_frames, 'counts': self.counts(), 'statuses': statuses, 'store_path': str(self.store_path), 'status': 'WAITING_HUMAN'}

    def start_session(self, data):
        frame_id, review_pass = data.get('frame_id'), data.get('review_pass')
        require(frame_id in self.frames, 'Unknown frame')
        require(review_pass in ('primary', 'repeat'), 'Unknown review pass')
        require(review_pass != 'repeat' or self.frames[frame_id]['repeat_review'], 'Unplanned repeat')
        validate_reviewer(data.get('reviewer'))
        token = uuid.uuid4().hex
        self.sessions[token] = {'frame_id': frame_id, 'review_pass': review_pass, 'reviewer': copy.deepcopy(data['reviewer']), 'started_at': utc_now(), 'start_mono': time.monotonic()}
        # This lookup is strictly same-pass and same-person. Independent reviewers
        # never receive another person's clicks, including another repeat review.
        previous = self.record_for(frame_id, review_pass)
        if previous and previous['reviewer']['id'] != data['reviewer']['id']:
            previous = None
        return {'token': token, 'record': previous, 'started_at': self.sessions[token]['started_at']}

    def save(self, data):
        with self.lock:
            session = self.sessions.get(data.get('token'))
            require(session is not None, 'Start a review session before saving')
            record = copy.deepcopy(data.get('record', {}))
            require(record.get('frame_id') == session['frame_id'] and record.get('review_pass') == session['review_pass'], 'Session/frame mismatch')
            require(record.get('reviewer') == session['reviewer'], 'Reviewer changed; start a new session')
            frame = self.frames[record['frame_id']]
            for path, key in ((self.manifest_path, 'manifest_sha256'), (self.plan_path, 'plan_sha256'), (self.contract_path, 'corner_contract_sha256')):
                require(sha256(path) == self.bindings[key], 'Input changed during review: ' + key)
            require(sha256(self.images[record['frame_id']]) == frame['image_sha256'], 'Raw image changed during review')
            record.update({key: frame[key] for key in ('session_id', 'image_sha256', 'width', 'height')})
            record.update({key: self.bindings[key] for key in ('plan_sha256', 'corner_contract_sha256', 'corner_definition_version')})
            record['source_kind'] = 'human_in_progress' if record.get('status') == 'draft' else 'human_reviewed'
            record['review_time'] = {'started_at': session['started_at'], 'finished_at': utc_now(), 'duration_seconds': round(time.monotonic() - session['start_mono'], 6), 'clock_source': 'server_wall_and_monotonic'}
            previous = self.record_for(record['frame_id'], record['review_pass'])
            if previous and previous['status'] != 'draft':
                require(record.get('edit_reason', '').strip(), 'Editing a submitted review requires a reason')
            if record['review_pass'] == 'repeat':
                primary = self.record_for(record['frame_id'], 'primary')
                record['repeat_relationship'] = 'first_review_unavailable' if primary is None else ('same_person_repeat' if primary['reviewer']['id'] == record['reviewer']['id'] else 'different_person_repeat')
                record['independent_repeat'] = record['repeat_relationship'] == 'different_person_repeat' and not any(record['reviewer'][key] for key in ('previous_annotation_exposure', 'previous_prediction_exposure', 'machine_assistance'))
            self.validate_record(record, allow_draft=True)
            record['revision'] = 1 if previous is None else previous.get('revision', 1) + 1
            self.store_path.parent.mkdir(parents=True, exist_ok=True)
            if previous:
                history = self.store_path.with_name(self.store_path.stem + '.history.jsonl')
                with history.open('a', encoding='utf-8') as output:
                    output.write(json.dumps({'superseded_at': utc_now(), 'record': previous}, ensure_ascii=False, allow_nan=False) + '\n')
            self.store['records'] = [r for r in self.store['records'] if (r['frame_id'], r['review_pass']) != (record['frame_id'], record['review_pass'])] + [record]
            self.atomic_write()
            return {'saved': True, 'record': record, 'counts': self.counts()}

    def export(self):
        bundle = {'schema_version': SCHEMA, 'source_kind': 'human_reviewed', 'bindings': copy.deepcopy(self.bindings), 'records': [copy.deepcopy(r) for r in self.store['records'] if r['status'] != 'draft'], 'counts': self.counts(), 'exported_at': utc_now(), 'status': 'WAITING_HUMAN'}
        bundle['complete_primary'] = bundle['counts']['incomplete_primary'] == 0
        bundle['complete_repeat'] = bundle['counts']['incomplete_repeat'] == 0
        if bundle['complete_primary'] and bundle['complete_repeat']:
            bundle['status'] = 'HUMAN_REVIEW_SUBMITTED'
        if not bundle['records']:
            bundle['source_kind'] = 'human_in_progress'
            return self.validate_bundle(bundle, allow_drafts=True)
        return self.validate_bundle(bundle)


class ReviewServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, context):
        self.context = context
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # No request bodies or reviewer identifiers in console logs.
        pass

    def send(self, payload, status=200, content_type='application/json; charset=utf-8', attachment=None):
        if not isinstance(payload, bytes):
            payload = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(payload)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; connect-src 'self'; frame-ancestors 'none'")
        if attachment:
            self.send_header('Content-Disposition', 'attachment; filename="' + attachment + '"')
        self.end_headers()
        self.wfile.write(payload)

    def check_origin(self, post=False):
        host = self.headers.get('Host', '')
        port = self.server.server_address[1]
        require(host in (f'127.0.0.1:{port}', f'localhost:{port}'), 'Only localhost Host allowed')
        if post:
            require(self.headers.get('Origin') in (f'http://127.0.0.1:{port}', f'http://localhost:{port}'), 'Local browser Origin required')

    def do_GET(self):
        try:
            self.check_origin()
            request = urlsplit(self.path)
            if request.path == '/api/catalog':
                self.send(self.server.context.catalog())
            elif request.path == '/api/export':
                self.send(self.server.context.export(), attachment='LIFTER_REFERENCE_REVIEWED.json')
            elif request.path == '/api/image':
                frame_id = parse_qs(request.query).get('frame_id', [''])[0]
                image_path = self.server.context.images.get(frame_id)
                require(image_path is not None, 'Unknown image')
                require(sha256(image_path) == self.server.context.frames[frame_id]['image_sha256'], 'Image changed during review')
                self.send(image_path.read_bytes(), content_type=mimetypes.guess_type(image_path)[0] or 'application/octet-stream')
            elif request.path in ('/', '/index.html', '/app.js', '/coords.js', '/style.css'):
                path = Path(__file__).parent / ('index.html' if request.path == '/' else request.path[1:])
                self.send(path.read_bytes(), content_type=mimetypes.guess_type(path)[0] or 'text/plain')
            else:
                self.send({'error': 'Not found'}, status=404)
        except (ValidationError, OSError, ValueError) as error:
            self.send({'error': str(error)}, status=400)

    def do_POST(self):
        try:
            self.check_origin(post=True)
            length = int(self.headers.get('Content-Length', '0'))
            require(0 < length <= 2 * 1024 * 1024, 'Request size invalid')
            data = json.loads(self.rfile.read(length))
            route = urlsplit(self.path).path
            if route == '/api/start':
                self.send(self.server.context.start_session(data))
            elif route == '/api/save':
                self.send(self.server.context.save(data))
            else:
                self.send({'error': 'Not found'}, status=404)
        except (ValidationError, OSError, ValueError, KeyError, TypeError) as error:
            self.send({'error': str(error)}, status=400)


def parser():
    arg = argparse.ArgumentParser(description=__doc__)
    sub = arg.add_subparsers(dest='command', required=True)
    for name in ('serve', 'validate', 'import', 'export'):
        cmd = sub.add_parser(name)
        for key in ('manifest', 'plan', 'contract', 'store'):
            cmd.add_argument('--' + key, required=True, type=Path)
        if name in ('validate', 'import'):
            cmd.add_argument('--input', required=True, type=Path)
        if name == 'import':
            cmd.add_argument('--merge', action='store_true', help='Add non-conflicting reviewed records; never replace clicks')
        if name == 'export':
            cmd.add_argument('--output', required=True, type=Path)
        if name == 'serve':
            cmd.add_argument('--port', type=int, default=8765)
    return arg


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        context = Context(args.manifest, args.plan, args.contract, args.store)
        if args.command == 'serve':
            server = ReviewServer(('127.0.0.1', args.port), context)
            print(json.dumps({'status': 'WAITING_HUMAN', 'url': f'http://127.0.0.1:{server.server_address[1]}', 'counts': context.counts(), 'bindings': context.bindings}, ensure_ascii=False), flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                server.server_close()
        elif args.command in ('validate', 'import'):
            incoming = context.validate_bundle(read_json(args.input))
            if args.command == 'import':
                require(not args.store.exists() or args.merge, 'Store exists; use --merge for non-conflicting import')
                for record in incoming['records']:
                    previous = context.record_for(record['frame_id'], record['review_pass'])
                    require(previous is None or previous == record, 'Conflicting existing review; edit through UI with reason')
                    if previous is None:
                        context.store['records'].append(record)
                context.atomic_write()
            print(json.dumps({'valid': True, 'records': len(incoming['records']), 'counts': context.counts(), 'status': 'WAITING_HUMAN'}, ensure_ascii=False))
        else:
            exported = context.export()
            require(exported['records'], 'WAITING_HUMAN: no submitted human reviews; no reference file created')
            require(not args.output.exists(), 'Refusing to overwrite export file')
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(exported, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
            print(json.dumps({'exported': str(args.output), 'counts': context.counts()}, ensure_ascii=False))
    except (ValidationError, OSError, ValueError, KeyError, TypeError) as error:
        state = 'WAITING_HUMAN' if 'WAITING_HUMAN' in str(error) else ('BLOCKED_CONTRACT' if 'CONTRACT' in str(error) else 'BLOCKED_REFERENCE')
        print(json.dumps({'error': str(error), 'status': state}, ensure_ascii=False))
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
