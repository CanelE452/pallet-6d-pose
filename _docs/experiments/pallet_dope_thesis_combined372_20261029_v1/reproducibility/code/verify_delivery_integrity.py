"""Fresh opaque byte hashes for final delivery; never import old verifiers.

Source-only until ROOT writes the separate hash-bound GO. The only directory
inventory is the two original experiment roots, using original snapshot_paths:
ALL regular files, no .pyc/cache exclusions, no symlinks. Added paths are listed
but never hashed. All content reads are exact declared paths. Raw bytes are
only streamed through SHA256, never decoded, parsed, loaded or evaluated.
"""
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import datetime
import hashlib
import json
import os
import re
import stat
import time

OUT = Path(__file__).resolve().parents[1]
BASE = OUT.parent
MAIN = BASE / 'dope-filter-ablation-per-type-20261027-v1'
WOOD = BASE / 'dope-wood640-eval80-20261027-v1'
FLIP = BASE / 'dope-loo-flip-per-type-20261026-v1'
NOFLIP = BASE / 'dope-loo-no-flip-per-type-20261026-v1'
R0 = Path('/home/minjae/Documents/github/pallet-pose/weights/paper_s2_stageB/net_epoch_0057_noseg.pth')
GO = OUT / 'preflight/DELIVERY_INTEGRITY_GO.json'
RESULT = OUT / 'preflight/DELIVERY_FINAL_INTEGRITY.json'
LOG = OUT / 'logs/DELIVERY_FINAL_INTEGRITY.jsonl'
META = {
    MAIN / 'preflight/SNAPSHOT_BEFORE.json': 'c308d494e196850dd20e7b82163c0ba7bf6a196548405937884d9bc8b6d0f21e',
    MAIN / 'preflight/ALLOWLIST.json': '4c90b54afda2194dc4c715f0578a87af82ecbae87b7fd9d1cc7a415ea22bdd25',
    WOOD / 'AUTHORIZATION.json': '1112cc8bb7814e5faa2ef0bcd5e15cc8453e7b2f54712220a78154e22655fee9',
    WOOD / 'ALLOWLIST.json': '38f6fe7b4da4ff67f0904e83bece084ffe72269cc4404ede6848e83ea663c445',
    WOOD / 'preflight/SNAPSHOT_BEFORE.json': '9d5fc0de12700836274e7a516e9fb94d53f1bef15cedaac01d293356c758bf44',
}
POLICY_SOURCES = {
    MAIN / 'code/prepare_inputs.py': '14c417fc4d9bafff69e2039e437aa57b28fed483db8568564c3bd10383abcccc',
    MAIN / 'code/verify_integrity.py': '6db31c119393f372e559c6435fe49e1ceb4908d5029f0eaba94e76633e61807f',
}
CATEGORIES = {'source_code': (20, {'.py'}), 'train_real_rgb': (3000, {'.png'}),
    'dev_rgb': (292, {'.png', '.jpg'}), 'train_synthetic_gt': (2212, {'.json'}),
    'train_synthetic_rgb': (2212, {'.png'}), 'checkpoint': (15, {'.pth'}),
    'metadata': (24, {'.json'}), 'protocol': (7, {'.json'}), 'r0_journal': (6, {'.jsonl'}),
    'checkpoint_receipt': (21, {'.json'}), 'cached_predictions': (3, {'.json'}),
    'cached_scores': (2, {'.json'}), 'dev_two_view_predictions': (7, {'.json'})}
PROTECTED_PARTS = {'final', 'physical_final', 'final_existing', 'final_physical'}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def digest_ok(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def canonical(path):
    p = Path(path)
    require(p.is_absolute() and p.resolve() == p and not p.is_symlink(), 'Noncanonical/symlink path: ' + str(p))
    return p


def fresh_hash(path):
    """Opaque, uncached SHA256; stat stability matches original prepare_inputs."""
    p = canonical(path)
    before = p.stat()
    require(stat.S_ISREG(before.st_mode), 'Not a regular file: ' + str(p))
    signature = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    h = hashlib.sha256()
    fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as stream:
        opened = os.fstat(stream.fileno())
        require(signature == (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns),
                'File identity changed before open: ' + str(p))
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    after = p.stat()
    require(signature == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
            'File changed while hashing: ' + str(p))
    return dict(sha256=h.hexdigest(), bytes=after.st_size)


def read_metadata(path, expected=None):
    p = canonical(path)
    require(p == GO or p in META, 'Unlisted metadata body read blocked')
    before = fresh_hash(p)
    require(expected is None or before['sha256'] == expected, 'Frozen metadata SHA differs: ' + str(p))
    obj = json.loads(p.read_text())
    require(fresh_hash(p) == before, 'Metadata changed during read')
    return obj, before['sha256']


def contract():
    return dict(GO_path=str(GO), GO_schema='delivery_final_integrity_go_v1',
        GO=dict(passed=True, integrity_verification_authorized=True, raw_access_mode='opaque_sha256_only',
            unknown_protected_FINAL_raw_authorized=False, original_file_count=599,
            MAIN_allowed_input_count=7821, authorized_WOOD80_raw_input_count=160,
            input_metadata_sha256={str(p): h for p, h in META.items()},
            source_sha256={str(Path(__file__).resolve()): 'CURRENT_SOURCE_SHA256',
                           **{str(p): h for p, h in POLICY_SOURCES.items()}}),
        outputs=[str(RESULT), str(LOG)], workers='1..4; default4',
        original_inventory_policy='All regular files in exactly FLIP and NOFLIP roots, plus R0; '
                                  'no exclusions, no symlinks, added paths never hashed',
        protected_exception='Only the exact160 paths from authorized WOOD80 allowlist and before snapshot',
        old_modules_imported_or_executed=False, execution_requires_separate_ROOT_GO=True)


def expectation(item):
    require(isinstance(item, dict) and digest_ok(item.get('sha256'))
            and isinstance(item.get('bytes'), int) and not isinstance(item['bytes'], bool)
            and item['bytes'] >= 0, 'Missing authoritative SHA/byte size')
    return dict(sha256=item['sha256'], bytes=item['bytes'])


def declarations():
    go, go_sha = read_metadata(GO)
    own_sha = fresh_hash(__file__)['sha256']
    source_sha = {str(Path(__file__).resolve()): own_sha, **{str(p): h for p, h in POLICY_SOURCES.items()}}
    require(go['schema'] == 'delivery_final_integrity_go_v1' and go['passed'] is True
            and go['integrity_verification_authorized'] is True and go['raw_access_mode'] == 'opaque_sha256_only'
            and go['unknown_protected_FINAL_raw_authorized'] is False
            and go['original_file_count'] == 599 and go['MAIN_allowed_input_count'] == 7821
            and go['authorized_WOOD80_raw_input_count'] == 160
            and go['input_metadata_sha256'] == {str(p): h for p, h in META.items()}
            and go['source_sha256'] == source_sha, 'ROOT GO/source/metadata contract absent or stale')
    for p, digest in POLICY_SOURCES.items():
        require(fresh_hash(p)['sha256'] == digest, 'Original inventory/verifier source changed')
    metadata = {p: read_metadata(p, h)[0] for p, h in META.items()}
    before = metadata[MAIN / 'preflight/SNAPSHOT_BEFORE.json']
    allow = metadata[MAIN / 'preflight/ALLOWLIST.json']
    auth = metadata[WOOD / 'AUTHORIZATION.json']
    wood_allow = metadata[WOOD / 'ALLOWLIST.json']
    wood_before = metadata[WOOD / 'preflight/SNAPSHOT_BEFORE.json']
    require(before['schema'] == 'original_experiment_byte_snapshot_v1' and before['total_files'] == 599
            and before['tree_files'] == 598 and before['plus_R0_files'] == 1
            and before['tree_file_counts'] == {str(FLIP): 417, str(NOFLIP): 181}, 'Original599 inventory authority differs')
    original = {p: expectation(v) for p, v in before['files'].items()}
    require(len(original) == 599 and str(R0) in original
            and sum(v['bytes'] for v in original.values()) == before['total_bytes'], 'Original baseline count/bytes differ')
    for name in original:
        p = Path(name)
        require(p == R0 or p.is_relative_to(FLIP) or p.is_relative_to(NOFLIP), 'Baseline path outside exact two roots/R0')
        require(not any(part.lower() in PROTECTED_PARTS for part in p.parts), 'Unexpected protected path in original snapshot')
    require(allow['schema'] == 'dope_filter_ablation_exact_inputs_v1' and len(allow['input_files']) == 7821
            and allow['protected_FINAL_raw_allowed'] is False and allow['global_content_search_allowed'] is False,
            'MAIN semantic allowlist changed')
    protected_sha = set(allow['protected_image_sha256'])
    protected_ids = set(allow['protected_frame_ids'])
    require(len(protected_sha) == len(protected_ids) == 229, 'Original protected namespace differs')
    main_inputs = {}
    for name, item in allow['input_files'].items():
        p = Path(name)
        category = item['category']
        require(p.is_absolute() and category in CATEGORIES and p.suffix in CATEGORIES[category][1]
                and not any(part.lower() in PROTECTED_PARTS for part in p.parts), 'Unsafe MAIN category/path')
        main_inputs[name] = expectation(item)
        require(item['sha256'] not in protected_sha, 'Protected raw image SHA cannot be a MAIN hash input')
    category_counts = Counter(v['category'] for v in allow['input_files'].values())
    require(dict(category_counts) == {k: v[0] for k, v in CATEGORIES.items()}, 'Exact13 MAIN category counts differ')
    require(wood_allow['authorization_sha256'] == META[WOOD / 'AUTHORIZATION.json']
            and wood_before['allowlist_sha256'] == META[WOOD / 'ALLOWLIST.json']
            and len(wood_allow['records']) == 80 and len(wood_allow['input_files']) == len(wood_before['files']) == 160
            and set(wood_allow['input_files']) == set(wood_before['files'])
            and set(auth['allowed_sessions']) == {'wood_day_01', 'wood_night_01'}, 'Exact80 authorization/baseline differs')
    wood_inputs = {}
    approved_paths = set()
    records = wood_allow['records']
    require(len({r['frame_id'] for r in records}) == 80
            and Counter(r['session_id'] for r in records) == {'wood_day_01': 24, 'wood_night_01': 56}
            and {r['frame_id'] for r in records} <= protected_ids, 'Authorized80 frame/session identity differs')
    for r in records:
        require(r['population_role'] == 'FINAL' and r['object_type'] == 'wood', 'Raw exception role/object changed')
        for key, category in (('image_path', 'authorized_WOOD80_RGB'), ('annotation_path', 'authorized_WOOD80_annotation')):
            name = r[key]
            item = wood_allow['input_files'][name]
            old = wood_before['files'][name]
            require(name not in approved_paths and item['category'] == old['category'] == category
                    and item['frame_id'] == r['frame_id'] and item['session'] == r['session_id']
                    and old['manifest_matches_actual'] is True
                    and old['sha256'] == old['manifest_sha256'] == item['manifest_sha256'], 'Exact raw80 pair declaration differs')
            if category == 'authorized_WOOD80_RGB':
                require(old['sha256'] in protected_sha, 'Approved FINAL80 RGB is not in the original protected namespace')
            approved_paths.add(name)
            wood_inputs[name] = expectation(old)
    require(approved_paths == set(wood_allow['input_files']) and not (approved_paths & set(main_inputs)),
            'Extra raw input or overlap with MAIN semantic inputs')
    return dict(go_sha256=go_sha, source_sha256=source_sha, original=original, main=main_inputs,
                wood=wood_inputs, tree_counts=before['tree_file_counts'], category_counts=dict(category_counts))


def inventory():
    """Same file-inclusion policy as original snapshot_paths; no outside walk."""
    paths, counts = set(), {}
    for root in (FLIP, NOFLIP):
        require(canonical(root).is_dir(), 'Original experiment root is missing')
        local = []
        for p in root.rglob('*'):
            require(not p.is_symlink(), 'Original experiment contains symlink: ' + str(p))
            if p.is_file():
                require(p.resolve().is_relative_to(root), 'Inventory escaped original root')
                local.append(str(p))
        counts[str(root)] = len(local)
        paths.update(local)
    paths.add(str(R0))
    return paths, counts


def append(stream, event, **fields):
    stream.write(json.dumps(dict(utc=utc(), event=event, **fields), ensure_ascii=False, allow_nan=False) + '\n')
    stream.flush()


def unchanged_contract(d):
    require(fresh_hash(GO)['sha256'] == d['go_sha256'], 'ROOT GO changed during final byte verification')
    for p, digest in META.items():
        require(fresh_hash(p)['sha256'] == digest, 'Authoritative input metadata changed')
    for p, digest in d['source_sha256'].items():
        require(fresh_hash(p)['sha256'] == digest, 'Verifier/policy source changed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--describe-contract', action='store_true')
    args = parser.parse_args()
    if args.describe_contract:
        print(json.dumps(contract(), indent=2), flush=True)
        return
    require(1 <= args.workers <= 4, 'Worker count must be1..4')
    require(RESULT.resolve() == RESULT and LOG.resolve() == LOG and not RESULT.exists() and not LOG.exists(),
            'Preserve existing final integrity receipt/partial log')
    d = declarations()  # No declared content/raw hashes before ROOT GO checks.
    started = time.monotonic()
    result = dict(passed=False, complete=False, started_utc=utc(), source_sha256=d['source_sha256'],
        input_metadata_sha256={str(p): h for p, h in META.items()}, ROOT_GO_sha256=d['go_sha256'],
        original_scope_files=599, MAIN_allowed_scope_files=7821, authorized_WOOD80_raw_scope_files=160,
        original_inventory_policy='All regular files; no exclusions including pyc; no symlinks',
        expected_tree_file_counts=d['tree_counts'], MAIN_category_counts=d['category_counts'],
        image_decodes=0, annotation_body_parses=0, checkpoint_deserializations=0, neural_forwards=0,
        PnP_fits=0, training_updates=0, unknown_protected_FINAL_raw_paths_hashed=0,
        original_output_paths_modified=0, raw_content_access='opaque_SHA256_streaming_only')
    with LOG.open('x') as log:
        try:
            append(log, 'begin', workers=args.workers)
            current, counts = inventory()
            missing = sorted(set(d['original']) - current)
            added = sorted(current - set(d['original']))
            result.update(missing_original_files=missing, added_original_files=added, actual_tree_file_counts=counts)
            require(not missing and not added and counts == d['tree_counts'],
                    'Original path set changed; newly discovered paths will NOT be hashed')
            expected = {}
            for label in ('original', 'main', 'wood'):
                for p, item in d[label].items():
                    require(p not in expected or expected[p] == item, 'Conflicting byte authority for a shared exact path')
                    expected[p] = item
            # Validate exact path semantics before opening any declared content.
            for name in expected:
                canonical(name)
            append(log, 'exact_union_ready', unique_paths=len(expected), expected_bytes=sum(v['bytes'] for v in expected.values()))
            actual, done_bytes, last = {}, 0, time.monotonic()
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(fresh_hash, p): p for p in sorted(expected)}
                for future in as_completed(futures):
                    p = futures[future]
                    actual[p] = future.result()
                    done_bytes += actual[p]['bytes']
                    if time.monotonic() - last >= 10 or len(actual) == len(expected):
                        append(log, 'hash_progress', completed=len(actual), total=len(expected), bytes=done_bytes)
                        print('DELIVERY_HASH', len(actual), '/', len(expected), done_bytes, flush=True)
                        last = time.monotonic()
            changed = {label: sorted(p for p, want in d[label].items() if actual[p] != want)
                       for label in ('original', 'main', 'wood')}
            after, counts_after = inventory()
            missing_after = sorted(set(d['original']) - after)
            added_after = sorted(after - set(d['original']))
            unchanged_contract(d)
            result.update(complete=True, exact_original_path_set_unchanged=not (missing_after or added_after),
                missing_original_files=missing_after, added_original_files=added_after, actual_tree_file_counts=counts_after,
                changed_original_files=changed['original'], changed_MAIN_allowed_inputs=changed['main'],
                changed_authorized_WOOD80_raw_inputs=changed['wood'], unique_content_paths_hashed=len(actual),
                unique_content_bytes_hashed=done_bytes, authorized_protected_raw_paths_hashed=160,
                all_original_experiment_and_R0_bytes_unchanged=not (changed['original'] or missing_after or added_after),
                all_MAIN_allowed_input_bytes_unchanged=not changed['main'],
                all_authorized_WOOD80_raw_bytes_unchanged=not changed['wood'])
            result['passed'] = not any(changed.values()) and not missing_after and not added_after and counts_after == d['tree_counts']
            require(result['passed'], 'Original or exact allowed bytes changed; final delivery must stop')
            append(log, 'passed', unique_paths=len(actual), elapsed_seconds=time.monotonic() - started)
        except Exception as error:
            result['error'] = repr(error)
            append(log, 'failed', error=repr(error), automatic_retry=False)
        result.update(completed_utc=utc(), elapsed_seconds=time.monotonic() - started, task_log_path=str(LOG))
    result['task_log_sha256'] = fresh_hash(LOG)['sha256']
    with RESULT.open('x') as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print('DELIVERY_FINAL_INTEGRITY', 'PASS' if result['passed'] else 'FAIL', str(RESULT), flush=True)
    require(result['passed'], 'Final integrity failed; preserve receipt/log, no automatic retry')


if __name__ == '__main__':
    main()
