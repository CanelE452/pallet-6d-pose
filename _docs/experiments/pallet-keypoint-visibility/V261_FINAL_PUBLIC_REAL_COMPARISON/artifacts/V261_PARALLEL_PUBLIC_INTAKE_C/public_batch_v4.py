"""Explicitly authorized opaque intake: verified reuse, then at most eight GETs.

No decoder, label compiler, model, training, retry or directory discovery.
The V3 utility module is read by its exact binding and executed without a loader.
"""
import argparse
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import stat
import threading
import types


V3_SHA = '3ca2fbc8746ed284dc7b91b9d62629e3e1b4f96489762f7ea400d04f63f01948'
V3_SIZE = 24311
WORKERS = 8


def need(ok, message):
    if not bool(ok):
        raise ValueError(message)


def guarded(path):
    p = Path(path)
    need(p.is_absolute() and '..' not in p.parts, 'Exact absolute artifact path')
    need(not any(t in str(p).casefold() for t in
                 ('v112', 'cache', '.pyc', 'current223', 'publicheld', 'holdout', 'oldval')),
         'Forbidden artifact before FS')
    for a in [*reversed(p.parents), p]:
        need(not stat.S_ISLNK(os.lstat(a).st_mode), 'No ancestor or leaf symlink')
    return p


def raw(path):
    fd = os.open(guarded(path), os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as f:
        return f.read()


def code_binding(path):
    b = raw(path)
    return {'path': str(path), 'sha256': hashlib.sha256(b).hexdigest(), 'size_bytes': len(b)}


def load_utilities(b):
    need(b['sha256'] == V3_SHA and b['size_bytes'] == V3_SIZE, 'Exact frozen V3 utilities')
    source = raw(b['path'])
    need(len(source) == b['size_bytes'] and hashlib.sha256(source).hexdigest() == b['sha256'],
         'V3 utility bytes')
    module = types.ModuleType('v261_bound_public_intake_v3_utilities')
    module.__file__ = b['path']
    exec(compile(source, b['path'], 'exec'), module.__dict__)
    return module


def validate_metadata(e, u):
    manifest = json.loads(u.bound_read(e['metadata_manifest']))
    need(manifest['source_API']['sha256'] == u.API_SHA and
         manifest['source_README']['sha256'] == u.README_SHA,
         'Same sealed official metadata')
    bundles = {r['bundle_id']: r for r in manifest['complete_bundles']}
    need(len(bundles) == len(manifest['complete_bundles']) == 284 and
         len(e['bundle_ids']) == len(set(e['bundle_ids'])) == 284 and
         set(e['bundle_ids']) == set(bundles), 'All fixed 284 bundles; no selection')
    needed = {}
    for name in e['bundle_ids']:
        row = bundles[name]
        for f in [*row['files'].values(), row['K']]:
            if f['file_id'] in needed:
                need(needed[f['file_id']] == f, 'Shared file ID metadata unchanged')
            needed[f['file_id']] = f
    allowed = e['exact_download_allowlist']
    need(len(allowed) == len(needed) == 1142 and
         {x['file_id'] for x in allowed} == set(needed), 'Exact 1142 unique file IDs')
    destinations = set()
    for x in allowed:
        f = needed[x['file_id']]
        stage = x['publisher_relative_path']
        relative = Path(stage)
        need(not relative.is_absolute() and stage == relative.as_posix() and
             '..' not in relative.parts and
             not any(t in stage.casefold() for t in ('v112', 'cache', '.pyc')),
             'Safe exact publisher destination')
        need(stage not in destinations, 'One worker owner for each destination')
        destinations.add(stage)
        need(stage == f['publisher_relative_path'] and x['size_bytes'] == f['size_bytes'] and
             x['publisher_MD5'] == f['publisher_MD5'], 'Unchanged publisher identity')
        need(x['url'] == 'https://rdr.kuleuven.be/api/access/datafile/'+str(x['file_id']),
             'Only exact official public file-ID URL')
    return needed, sorted(allowed, key=lambda x: x['publisher_relative_path'])


def validate_reuse(e, u, needed):
    before_entry = json.loads(u.bound_read(e['previous_download_entry']))
    before_failure = json.loads(u.bound_read(e['previous_failure']))
    need(before_entry['code']['sha256'] == V3_SHA and
         before_entry['code']['size_bytes'] == V3_SIZE and
         before_entry.get('approved_actual_public_allowlist_download') is True and
         before_entry.get('opaque_only') is True and before_entry.get('automatic_retry') is False,
         'Exact explicitly authorized V3 predecessor')
    for name in ('metadata_manifest', 'bundle_ids', 'exact_download_allowlist', 'redirect_scope'):
        need(before_entry[name] == e[name], 'Same V3 allowlist, metadata and redirect scope')
    need(before_failure.get('completed') is False and before_failure.get('PASS') is False and
         before_failure.get('automatic_retry') is False,
         'Preserved interrupted V3 failure receipt')
    complete = before_failure['completed_downloads']
    old = {r['file_id']: r for r in complete}
    need(len(old) == len(complete) and set(old) <= set(needed),
         'Unique completed V3 IDs inside the same allowlist')
    plan = e['reuse_files']
    need(len(plan) == len(old) == e['expected_reused_completed_files'] and
         {r['publisherfile_id'] for r in plan} == set(old),
         'Reuse every and only recorded complete predecessor file')
    reuse = {}
    for r in plan:
        file_id = r['publisherfile_id']
        prior, f, source = old[file_id], needed[file_id], r['source']
        need(source == {k: prior[k] for k in ('path', 'sha256', 'size_bytes')} and
             source['size_bytes'] == f['size_bytes'] and
             prior['publisher_relative_path'] == f['publisher_relative_path'] and
             r['original_MD5'] == prior['publisher_MD5_verified'] == f['publisher_MD5'],
             'Exact completed SHA/size/path and verified publisher MD5')
        need(len(source['sha256']) == 64 and
             not str(source['path']).casefold().endswith('.part'), 'Never copy partial bytes')
        reuse[file_id] = r
    return reuse


def fetch_one(u, x, destination, stop):
    """Exactly one request attempt for this file; URL tokens never leave utilities."""
    if stop.is_set():
        raise CancelledError()
    part = u.safe(destination.with_name(destination.name+'.part'), False)
    count, md5, sha = 0, hashlib.md5(), hashlib.sha256()
    request = u.urllib.request.Request(x['url'], headers={
        'User-Agent': 'PublicMetadataBoundOpaqueCollector/1.0'})
    redirect = u.DatasetScopedRedirect(x['url'])
    opener = u.urllib.request.build_opener(redirect)
    with opener.open(request, timeout=90) as response, part.open('xb') as output:
        final_url = response.geturl()
        need(response.status == 200 and
             (final_url == x['url'] or u.storage_url_in_scope(final_url)),
             'Final response outside the official initial or exact dataset storage scope')
        final_origin = u.redacted_url_metadata(final_url)
        while True:
            chunk = response.read(1024*1024)
            if not chunk:
                break
            count += len(chunk)
            need(count <= x['size_bytes'], 'Response exceeds exact publisher size')
            md5.update(chunk)
            sha.update(chunk)
            output.write(chunk)
    need(count == x['size_bytes'] and md5.hexdigest() == x['publisher_MD5'],
         'Opaque publisher size and original MD5')
    part.rename(destination)
    return {'file_id': x['file_id'], 'publisher_relative_path': x['publisher_relative_path'],
            'url': x['url'], 'path': str(destination), 'sha256': sha.hexdigest(),
            'size_bytes': count, 'publisher_MD5_verified': md5.hexdigest(),
            'official_redirect_events': redirect.events, 'final_origin_redacted': final_origin,
            'operation': 'GET_OFFICIAL_SCOPE'}


def sanitized_error(exc, x=None):
    status = getattr(exc, 'code', None)
    return {'file_id': None if x is None else x['file_id'],
            'publisher_relative_path': None if x is None else x['publisher_relative_path'],
            'error_type': type(exc).__name__,
            'HTTP_status': status if isinstance(status, int) else None,
            'error_summary': 'Opaque intake failed; exception text and URL query values suppressed.'}


def parallel_gets(u, jobs, out, completed, fetch=fetch_one):
    """Drain running attempts after a failure/interrupt; cancel all queued attempts.

    Only the main thread writes ledgers. A successful file is recorded even if it
    finishes while another worker fails. No retry, fallback or overwrite occurs.
    """
    stop, futures, errors = threading.Event(), {}, []
    pool = ThreadPoolExecutor(max_workers=WORKERS, thread_name_prefix='public_opaque')
    try:
        for x in jobs:
            destination = u.safe(out/Path(x['publisher_relative_path']), False)
            destination.parent.mkdir(parents=True, exist_ok=True)
            futures[pool.submit(fetch, u, x, destination, stop)] = x
        for future in as_completed(futures):
            x = futures[future]
            try:
                future.result()
            except CancelledError:
                continue
            except BaseException as exc:
                errors.append(sanitized_error(exc, x))
                stop.set()
                for queued in futures:
                    queued.cancel()
                break
            if sum(f.done() and not f.cancelled() for f in futures) % 25 == 0:
                print(json.dumps({'stage': 'opaque_public_parallel_GET',
                                  'finished_attempts': sum(f.done() for f in futures),
                                  'GET_jobs': len(jobs), 'max_workers': WORKERS}), flush=True)
    except BaseException as exc:
        errors.append(sanitized_error(exc))
        stop.set()
        for queued in futures:
            queued.cancel()
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    # Deterministic ledger independent of completion order. Never expose raw exceptions.
    seen_errors = {(x['file_id'], x['error_type']) for x in errors}
    for future, x in sorted(futures.items(), key=lambda item: item[1]['publisher_relative_path']):
        if future.cancelled():
            continue
        try:
            completed.append(future.result())
        except CancelledError:
            pass
        except BaseException as exc:
            key = (x['file_id'], type(exc).__name__)
            if key not in seen_errors:
                errors.append(sanitized_error(exc, x))
                seen_errors.add(key)
    return sorted(errors, key=lambda r: (r['publisher_relative_path'] or '', r['error_type']))


def download(entry_path):
    e = json.loads(raw(entry_path))
    need(e.get('approved_actual_public_allowlist_download') is True and
         e.get('opaque_only') is True and e.get('explicit_user_authorized_resumption') is True and
         e.get('approved_bounded_parallel_intake') is True,
         'Root explicit technical resumption and bounded parallel intake required')
    need(e.get('training_admission') is False and e.get('automatic_retry') is False and
         e.get('max_workers') == WORKERS, 'Exactly eight workers; no retry or training permission')
    need(e['code'] == code_binding(Path(__file__).absolute()), 'Pinned V4 downloader')
    u = load_utilities(e['utility_code'])
    need(e.get('Rootapproved_same_dataset_official_redirect') is True and
         e.get('redirect_scope') == u.STORAGE_SCOPE, 'Unchanged explicit scoped official redirect')
    needed, allowed = validate_metadata(e, u)
    reuse = validate_reuse(e, u, needed)
    out = u.safe(e['output_directory'], False)
    need(not out.exists(), 'Fresh V4 output; no in-place resume')
    out.mkdir()
    completed, stage, errors = [], 'before_reuse', []
    try:
        # Finish and verify all authorized local copies before initiating network work.
        for x in allowed:
            if x['file_id'] not in reuse:
                continue
            stage = x['publisher_relative_path']
            destination = u.safe(out/Path(stage), False)
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = reuse[x['file_id']]['source']
            copied = u.copy_verified_reuse(source, destination, needed[x['file_id']])
            completed.append({'file_id': x['file_id'], 'publisher_relative_path': stage,
                              'url': x['url'], 'path': str(destination), **copied,
                              'operation': 'REUSE_VERIFIED_COPY', 'source': source})
        print(json.dumps({'stage': 'verified_reuse_complete', 'reused_files': len(completed),
                          'required_files': len(needed), 'max_workers': WORKERS}), flush=True)
        stage = 'parallel_GET'
        jobs = [x for x in allowed if x['file_id'] not in reuse]
        errors = parallel_gets(u, jobs, out, completed)
        need(not errors, 'One or more intake attempts failed or intake was interrupted')
        completed.sort(key=lambda r: r['publisher_relative_path'])
        need(len(completed) == len(needed) and
             {r['file_id'] for r in completed} == set(needed), 'Complete exact opaque allowlist')
    except BaseException as exc:
        if not errors:
            errors = [sanitized_error(exc)]
        completed.sort(key=lambda r: r['publisher_relative_path'])
        failed = u.dump(out/'FAILURE.json', {
            'completed': False, 'PASS': False, 'entry': code_binding(entry_path),
            'code': e['code'], 'utility_code': e['utility_code'], 'stage': stage,
            'error_type': errors[0]['error_type'], 'errors': errors,
            'completed_downloads': completed, 'completed_files': len(completed),
            'required_files': len(needed), 'max_workers': WORKERS,
            'partial_bytes_preserved': True, 'running_attempts_drained': True,
            'previous_failure': e['previous_failure'],
            'previous_download_entry': e['previous_download_entry'],
            'explicit_user_authorized_resumption': True, 'automatic_retry': False,
            'decoded_files': 0, 'labels_models_training_or_evaluation': 0,
            'training_admission': False})
        print(json.dumps({'completed': False, 'PASS': False, 'failure': failed,
                          'completed_files': len(completed)}, sort_keys=True), flush=True)
        raise RuntimeError('Opaque public intake stopped; signed URL query values suppressed.') from None
    receipt = u.dump(out/'OPAQUE_DOWNLOAD_COMPLETE.json', {
        'completed': True, 'PASS': True, 'entry': code_binding(entry_path),
        'code': e['code'], 'utility_code': e['utility_code'],
        'metadata_manifest': e['metadata_manifest'], 'downloads': completed,
        'previous_failure': e['previous_failure'],
        'previous_download_entry': e['previous_download_entry'],
        'explicit_user_authorized_resumption': True, 'reused_verified_files': len(reuse),
        'new_GET_files': len(completed)-len(reuse), 'required_files': len(needed),
        'max_workers': WORKERS, 'ledger_sorted_by': 'publisher_relative_path',
        'decoded_files': 0, 'labels_models_training_or_evaluation': 0,
        'training_admission': False, 'automatic_retry': False})
    print(json.dumps({'completed': True, 'PASS': True, 'receipt': receipt}, sort_keys=True), flush=True)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--download-entry', required=True)
    download(parser.parse_args().download_entry)
