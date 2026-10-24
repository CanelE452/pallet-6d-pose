"""Metadata-only, root-entry-gated public GitHub verification. No Git writes.

Only GitHub repository/ref/commit/tree JSON APIs are allowed. No contents/raw/blob
API, base64 payload, image decoder, dataset/model library or source discovery.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import re
import stat
import urllib.request

HOME = Path('/home/minjae/Documents/publication_remote_verification_b_v196_20261024')
REPO = Path('/home/minjae/Documents/pallet-visibility-publication-20261024')
OWNER_REPOSITORY = 'CanelE452/pallet-6d-pose'
PREFIX = '_docs/experiments/pallet-keypoint-visibility/'
PHASE = 'query_visibility_grouped_source_holdout_target_readout_20261024_v196'
SELECTED = (
    PREFIX + 'publication/deltas/V196_20261024_v1/DELTA_MANIFEST.json',
    PREFIX + 'archive/task-2/pallet-keypoint-visibility/' + PHASE + '/PORTABLE_REPORT_KO.md',
    PREFIX + 'archive/task-2/pallet-keypoint-visibility/' + PHASE + '/ROOT_RESULTS_KO.json',
    PREFIX + 'archive/task-2/pallet-keypoint-visibility/' + PHASE + '/independent_saved_viewer_review_c/REVIEW.json',
    PREFIX + 'EXPERIMENT_INDEX_THROUGH_V196_KO.md',
)

PROTOCOL = HOME / 'PROTOCOL_BEFORE_REMOTE_API_OR_PUBLISHED_FILE_READS.json'


def require(value, message):
    if not bool(value):
        raise AssertionError(message)


def safe_path(value, absolute=True):
    path = Path(value)
    require(path.is_absolute() == absolute, 'Exact absolute/relative path contract')
    require(bool(path.parts) and '..' not in path.parts, 'No traversal')
    require(all('v112' not in part.casefold() for part in path.parts), 'Forbidden name before filesystem access')
    return path


def open_nofollow(value):
    path = safe_path(value)
    fd = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in path.parts[1:-1]:
            nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = nxt
        final = os.open(path.parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
    finally:
        os.close(fd)
    if not stat.S_ISREG(os.fstat(final).st_mode):
        os.close(final)
        raise AssertionError('Regular file only')
    return os.fdopen(final, 'rb')


def file_hash(path):
    h = hashlib.sha256()
    with open_nofollow(path) as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def bound_json(binding):
    require(re.fullmatch('[0-9a-f]{64}', binding['sha256']) is not None, 'Bound SHA256')
    with open_nofollow(binding['path']) as stream:
        data = stream.read(64 * 1024**2 + 1)
    require(len(data) <= 64 * 1024**2, 'Metadata JSON size bound')
    require(hashlib.sha256(data).hexdigest() == binding['sha256'], 'Metadata binding drift')
    return json.loads(data)


def local_blob(path, expected):
    require(isinstance(expected['size_bytes'], int) and 0 <= expected['size_bytes'] < 100 * 1024**2, 'Physical file bound')
    require(re.fullmatch('[0-9a-f]{64}', expected['sha256']) is not None, 'Sealed physical SHA256')
    sha256 = hashlib.sha256()
    blob = hashlib.sha1(('blob ' + str(expected['size_bytes']) + '\0').encode())
    count = 0
    with open_nofollow(path) as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            count += len(block)
            sha256.update(block)
            blob.update(block)
    require(count == expected['size_bytes'] and sha256.hexdigest() == expected['sha256'], 'Local published bytes differ from sealed push plan')
    return {'size_bytes': count, 'sha256': sha256.hexdigest(), 'git_blob_sha1': blob.hexdigest()}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise AssertionError('GitHub API redirect refused; inspect explicitly')


class MetadataAPI:
    def __init__(self):
        self.calls = []
        self.opener = urllib.request.build_opener(NoRedirect)

    def get(self, suffix):
        require(re.fullmatch(r'/(?:git/ref/heads/main|git/commits/[0-9a-f]{40}|git/trees/[0-9a-f]{40})?', suffix) is not None or suffix == '', 'Metadata endpoint whitelist')
        url = 'https://api.github.com/repos/' + OWNER_REPOSITORY + suffix
        request = urllib.request.Request(url, headers={
            'Accept': 'application/vnd.github+json',
            'X-GitHub-Api-Version': '2022-11-28',
            'User-Agent': 'pallet-v196-independent-publication-metadata-review',
        })
        with self.opener.open(request, timeout=30) as response:
            require(response.status == 200 and response.url == url, 'Exact public API HTTP200')
            raw = response.read(32 * 1024**2 + 1)
            require(len(raw) <= 32 * 1024**2, 'API metadata response size bound')
        self.calls.append({'endpoint': url, 'status': 200, 'response_sha256': hashlib.sha256(raw).hexdigest(), 'response_bytes': len(raw)})
        return json.loads(raw)


def find_remote_blob(api, root_tree, relative, cache):
    parts = safe_path(relative, absolute=False).parts
    current = root_tree
    for index, name in enumerate(parts):
        if current not in cache:
            tree = api.get('/git/trees/' + current)
            require(tree['sha'] == current and tree.get('truncated') is False, 'Complete exact nonrecursive Git tree')
            cache[current] = tree['tree']
        matches = [row for row in cache[current] if row['path'] == name]
        require(len(matches) == 1, 'Exactly one selected tree path component')
        row = matches[0]
        require(re.fullmatch('[0-9a-f]{40}', row['sha']) is not None, 'Git SHA1')
        if index < len(parts) - 1:
            require(row['type'] == 'tree' and row['mode'] == '040000', 'Directory tree, no symlink/submodule')
            current = row['sha']
        else:
            require(row['type'] == 'blob' and row['mode'] == '100644', 'Regular published Git blob')
            return row


def write_new(path, obj):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def run(entry_path):
    require(Path(__file__).absolute().parent == HOME, 'Pinned standalone tool directory')
    with open_nofollow(entry_path) as stream:
        entry = json.load(stream)
    require(entry['approved_actual_remote_metadata_verification'] is True and entry['payload_push_completed'] is True, 'Root pushed-commit authorization required')
    require(entry['code'] == {'path': str(Path(__file__).absolute()), 'sha256': file_hash(__file__)}, 'Exact frozen verifier')
    require(entry['protocol']['path'] == str(PROTOCOL), 'Exact protocol path')
    protocol = bound_json(entry['protocol'])
    require(protocol['code'] == entry['code'], 'Protocol/code closure')
    scope = bound_json(entry['root_scope'])
    require(bool(scope), 'Explicit root metadata-only scope binding')
    require(entry['repository'] == OWNER_REPOSITORY and entry['branch'] == 'main', 'Fixed public repository/main')
    oid = entry['expected_payload_commit']
    require(re.fullmatch('[0-9a-f]{40}', oid) is not None, 'Root-provided pushed payload commit')
    require(entry['local_repository'] == str(REPO) and entry['selected_paths'] == list(SELECTED), 'Only five fixed published files')
    plan = bound_json(entry['sealed_push_plan'])
    require(plan['authorized_main_push'] is True and plan['repository'] == str(REPO) and plan['remote'] == 'https://github.com/' + OWNER_REPOSITORY + '.git' and plan['branch'] == 'main', 'Exact sealed push-plan repository contract')
    flat = [row for batch in plan['batches'] for row in batch['files']]
    require(len({row['path'] for row in flat}) == len(flat), 'Unique sealed physical paths')
    expected = {}
    for relative in SELECTED:
        matches = [row for row in flat if row['path'] == relative]
        require(len(matches) == 1, 'Selected fixed file in sealed physical push plan')
        expected[relative] = local_blob(REPO / relative, matches[0])
    out = safe_path(entry['output_directory'])
    require(out.parent == HOME and out.name == 'actual_remote_metadata_review_v1', 'Fresh outside-phase review directory')
    out.mkdir(exist_ok=False)
    api = MetadataAPI()
    try:
        repository = api.get('')
        require(repository['full_name'] == OWNER_REPOSITORY and repository['private'] is False and repository['visibility'] == 'public' and repository['default_branch'] == 'main', 'Public repository/default main')
        before = api.get('/git/ref/heads/main')
        require(before['ref'] == 'refs/heads/main' and before['object']['type'] == 'commit' and before['object']['sha'] == oid, 'Remote main exactly pushed payload commit')
        commit = api.get('/git/commits/' + oid)
        require(commit['sha'] == oid and re.fullmatch('[0-9a-f]{40}', commit['tree']['sha']) is not None, 'Exact commit tree authority')
        records = []
        cache = {}
        for relative in SELECTED:
            blob = find_remote_blob(api, commit['tree']['sha'], relative, cache)
            require(blob['sha'] == expected[relative]['git_blob_sha1'] and blob['size'] == expected[relative]['size_bytes'], 'Remote selected blob metadata differs from sealed local bytes')
            records.append({'path': relative, 'local_sealed': expected[relative], 'remote_blob_sha1': blob['sha'], 'remote_size_bytes': blob['size'], 'remote_mode': blob['mode'], 'PASS': True})
        after = api.get('/git/ref/heads/main')
        require(after['object']['sha'] == oid and after['ref'] == 'refs/heads/main', 'Remote main stable through selected metadata verification')
        review = {'completed': True, 'PASS': True, 'recorded_at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'expected_payload_commit': oid, 'repository': OWNER_REPOSITORY, 'public': True, 'default_branch': 'main', 'remote_main_before': oid, 'remote_main_after': oid, 'root_tree': commit['tree']['sha'], 'selected_files_verified': len(records), 'selected_files': records, 'entry': {'path': str(entry_path), 'sha256': file_hash(entry_path)}, 'code': entry['code'], 'protocol': entry['protocol'], 'scope': entry['root_scope'], 'sealed_push_plan': entry['sealed_push_plan'], 'API_calls': api.calls, 'remote_content_raw_blob_contents_base64_or_pixel_reads': 0, 'Git_writes_or_commands': 0, 'source_phase_or_model_or_evaluation_body_reads': 0, 'limits': 'Selected five Git blob identities and remote public/main metadata only; not a whole-repository roundtrip or model/scientific verification.'}
        write_new(out / 'REVIEW.json', review)
        print(json.dumps({'completed': True, 'PASS': True, 'review': {'path': str(out / 'REVIEW.json'), 'sha256': file_hash(out / 'REVIEW.json')}, 'files': len(records), 'API_calls': len(api.calls)}), flush=True)
    except Exception as exc:
        write_new(out / 'FAILURE.json', {'completed': False, 'PASS': False, 'error_type': type(exc).__name__, 'error': str(exc), 'API_calls': api.calls, 'entry': {'path': str(entry_path), 'sha256': file_hash(entry_path)}, 'automatic_retry': False})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--entry', type=Path, required=True)
    run(parser.parse_args().entry)
