"""Bounded metadata inventory proposal; no package build, arrays/models or Git.

Only the named directories below are listed, one level. Data leaves use saved
receipt bindings without stat/read; public interrupted .part bytes alone may be
hashed as explicitly archived incomplete bytes. Root adds final outputs later.
"""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import stat

P = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
R = P/'publication_root_preparation'
H = R/'V261_PUBLICATION_INVENTORY_C'
OWNED = (
    'V260_PUBLIC_BATCH_METADATA_C', 'V261_PARALLEL_PUBLIC_INTAKE_C',
    'V260_ROOT_PUBLIC_BATCH', 'V261_ROOT_PUBLIC_BATCH_TEACHER',
    'V260_PUBLIC_BATCH_TEACHER_A', 'V260_SINGLE_QUERY_FEATURE_TRANSPORT_B',
    'V261_FINAL_PUBLIC_AUX_TRAINER_B', 'V261_FROZEN_CURRENT_COMPARISON_B',
    'V261_FINAL_CROP_GALLERY_B', 'V261_FINAL_PUBLICATION_C',
    'V261_PUBLIC_BATCH_CONSISTENCY_C', 'V260_ROOT_PUBLIC_FEATURE_PILOT',
    'V261_ROOT_FINAL_PUBLIC_REAL_COMPARISON', 'V261_PUBLICATION_INVENTORY_C')
DENIED = ('v112', 'cache', '.pyc')
MODEL_SUFFIXES = ('.pt', '.pth', '.ckpt', '.safetensors')


def need(ok, message):
    if not bool(ok):
        raise ValueError(message)


def lexical(path):
    p = Path(path)
    need(p.is_absolute() and '..' not in p.parts and
         not any(t in str(p).casefold() for t in DENIED), 'Lexical rejection before FS')
    return p


def safe(path):
    p = lexical(path)
    for a in [*reversed(p.parents), p]:
        need(not stat.S_ISLNK(os.lstat(a).st_mode), 'No ancestor or leaf symlink')
    return p


def measure(path):
    p = safe(path)
    h, n = hashlib.sha256(), 0
    fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
            n += len(block)
    return {'path': str(p), 'sha256': h.hexdigest(), 'size_bytes': n}


def json_metadata(path, expected=None):
    b = measure(path)
    if expected:
        need(b == expected, 'Exact admitted metadata binding')
    fd = os.open(safe(path), os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as f:
        return json.load(f), b


def simple(b):
    return {k: b[k] for k in ('path', 'sha256', 'size_bytes')}


def bindings(value):
    if isinstance(value, dict):
        if ({'path', 'sha256', 'size_bytes'} <= set(value) and
                type(value['path']) is str and type(value['sha256']) is str and
                type(value['size_bytes']) is int):
            yield simple(value)
        for v in value.values():
            yield from bindings(v)
    elif isinstance(value, list):
        for v in value:
            yield from bindings(v)


def logical(path):
    p = lexical(path)
    if p.is_relative_to(R):
        return 'artifacts/'+p.relative_to(R).as_posix()
    if p.is_relative_to(P):
        return 'dependencies/P/'+p.relative_to(P).as_posix()
    return 'dependencies/exact_'+hashlib.sha256(str(p).encode()).hexdigest()[:16]+'/'+p.name


def run():
    target = H/'ALLOWLIST_PROPOSAL.json'
    need(not target.exists(), 'Fresh proposal; no overwrite')
    rows, source_paths, metadata, metadata_bindings = {}, {}, [], []
    inert, aliases, partials, stale = [], {}, [], []

    def add(b, group, rel=None, evidence='saved binding only; source leaf not accessed'):
        b = simple(b)
        lexical(b['path'])  # No source stat/read for receipt-derived arrays, models or images.
        rel = rel or logical(b['path'])
        need(not Path(rel).is_absolute() and '..' not in Path(rel).parts and
             not any(t in rel.casefold() for t in DENIED), 'Safe proposed logical path')
        if b['path'] in source_paths:
            before = source_paths[b['path']]
            if rows[before]['source'] != b:
                stale.append({'saved_binding': b, 'current_included_binding': rows[before]['source'],
                              'action': 'Inert historical mismatch; no source replacement or fallback'})
            return
        need(rel not in rows or rows[rel]['source'] == b, 'Unique logical owner')
        rows[rel] = {'relative_path': rel, 'group': group, 'source': b, 'inventory_evidence': evidence}
        source_paths[b['path']] = rel

    def load(path, group='attempts_code_protocols', expected=None):
        value, b = json_metadata(path, expected)
        add(b, group, evidence='named metadata SHA/size measured; JSON parsed, serialized pointers inert')
        metadata.append(value)
        metadata_bindings.append(b)
        return value, b

    # Explicit top-level code/protocol/attempt records. No recursion into actual output dirs.
    for name in OWNED:
        directory = safe(R/name)
        for child in sorted(os.listdir(directory)):
            if any(t in child.casefold() for t in DENIED):
                continue  # Do not stat a rejected basename.
            p = directory/child
            mode = os.lstat(lexical(p)).st_mode
            need(not stat.S_ISLNK(mode), 'No owned artifact symlink')
            if not stat.S_ISREG(mode):
                continue
            if name == 'V261_ROOT_FINAL_PUBLIC_REAL_COMPARISON' and p.suffix == '.json' and not any(
                    tag in p.name for tag in ('ENTRY', 'POLICY', 'SCOPE', 'VERIFICATION', 'REPLAY')):
                continue  # Root adds final report/current summaries; do not consume their bodies here.
            need(not p.name.casefold().endswith(MODEL_SUFFIXES+('.npy', '.npz', '.png', '.jpg')),
                 'Top-level scope is code/metadata only; actual outputs Root will add')
            if p.suffix == '.json':
                load(p)
            else:
                add(measure(p), 'attempts_code_protocols', evidence='named shallow code/text bytes measured')

    # Frozen metadata-only collection and C known-two actual consistency receipt.
    for directory in (R/'V260_PUBLIC_BATCH_METADATA_C/metadata_v1',
                      R/'V261_PUBLIC_BATCH_CONSISTENCY_C/actual_review_v1'):
        for child in sorted(os.listdir(safe(directory))):
            if any(t in child.casefold() for t in DENIED):
                continue
            p = directory/child
            need(stat.S_ISREG(os.lstat(lexical(p)).st_mode) and p.suffix == '.json', 'Named metadata-only child')
            load(p, 'source_metadata_and_QA')

    # Canonical completed public raw files exactly once; no raw file stat/hash/read.
    raw_receipt, rb = load(R/'V260_ROOT_PUBLIC_BATCH/public_284_raw_v4/OPAQUE_DOWNLOAD_COMPLETE.json',
                           'intake_receipts')
    need(raw_receipt['completed'] is True and raw_receipt['PASS'] is True and
         len(raw_receipt['downloads']) == raw_receipt['required_files'] == 1142, 'All raw V4 public files complete')
    by_id = {r['file_id']: r for r in raw_receipt['downloads']}
    need(len(by_id) == 1142, 'Unique complete official file IDs')
    for item in raw_receipt['downloads']:
        rel = 'public_data/DOI10.48804_DJLDFS_V1_0/'+item['publisher_relative_path']
        add(item, 'public_raw_v4', rel)

    # All failed/intentionally interrupted intake receipts; success bytes map to V4, not copied again.
    for version in (1, 2, 3):
        failure_path = R/('V260_ROOT_PUBLIC_BATCH/public_284_raw_v%d/FAILURE.json' % version)
        failure, fb = load(failure_path, 'intake_failure_records')
        for prior in failure.get('completed_downloads', []):
            canonical = by_id[prior['file_id']]
            need(prior['sha256'] == canonical['sha256'] and prior['size_bytes'] == canonical['size_bytes'] and
                 prior['publisher_relative_path'] == canonical['publisher_relative_path'] and
                 prior['publisher_MD5_verified'] == canonical['publisher_MD5_verified'], 'Previous success exactly aliases V4')
            aliases[prior['path']] = {'relative_path': source_paths[canonical['path']],
                                     'sha256': prior['sha256'], 'size_bytes': prior['size_bytes'],
                                     'reason': 'Same verified official file ID and bytes; no duplicate successful raw asset'}
        stage = failure.get('stage')
        if stage and stage != 'before_first_request':
            rp = Path(stage)
            need(not rp.is_absolute() and '..' not in rp.parts and
                 not any(t in stage.casefold() for t in DENIED), 'Only exact failure stage can name a partial')
            partial = lexical(failure_path.parent/rp)
            partial = Path(str(partial)+'.part')
            safe(partial.parent)
            try:
                mode = os.lstat(partial).st_mode
            except FileNotFoundError:
                continue
            need(stat.S_ISREG(mode) and not stat.S_ISLNK(mode), 'Explicit partial is a regular opaque artifact')
            b = measure(partial)
            add(b, 'incomplete_intake_parts',
                'archive/incomplete_intake_v%d/' % version+rp.as_posix()+'.part',
                'Exact failed stage .part; opaque SHA only, not decoded or admitted as complete')
            partials.append({'source': b, 'status': 'INCOMPLETE_HTTP_BYTES_NOT_PUBLISHER_MD5_VALIDATED'})

    # One completed teacher receipt for each named immediate group directory.
    teacher_root = safe(R/'V261_ROOT_PUBLIC_BATCH_TEACHER')
    groups = []
    for child in sorted(os.listdir(teacher_root)):
        if not child.startswith('group_') or any(t in child.casefold() for t in DENIED):
            continue
        directory = safe(teacher_root/child)
        need(stat.S_ISDIR(os.lstat(directory).st_mode), 'Named immediate teacher group')
        complete, cb = load(directory/'COMPLETE.json', 'public_teacher284')
        need(complete['completed'] is True and complete['PASS'] is True, 'Every teacher group complete')
        groups.append({'directory': child, 'complete': cb, 'listed_frames': complete['listed_frames']})
        for b in complete['outputs'].values():
            add(b, 'public_teacher284')
    need(len(groups) == 6 and sum(g['listed_frames'] for g in groups) == 284, 'Exact all284 teacher groups, no filtering')

    # Root-admitted prior actual144 features/full86400 ledger (saved bindings only).
    pilot, pb = load(R/'V260_ROOT_PUBLIC_FEATURE_PILOT/single_query_feature_transport_v1/COMPLETE.json',
                     'pilot144_features_ledger',
                     {'path': str(R/'V260_ROOT_PUBLIC_FEATURE_PILOT/single_query_feature_transport_v1/COMPLETE.json'),
                      'sha256': 'b88e6949b459bd16cab973183d904de4945310d10e92f6f5272bd6c443f12580', 'size_bytes': 8949})
    need(pilot['completed'] is True and pilot['PASS'] is True, 'Pilot complete')
    for b in pilot['outputs'].values():
        add(b, 'pilot144_features_ledger')

    # Root late admitted final public features and both fixed-trained outputs.
    final_root = R/'V261_ROOT_FINAL_PUBLIC_REAL_COMPARISON'
    feature, fcb = load(final_root/'public_aux_frozen_features_v1/COMPLETE.json',
                        'final_public_features',
                        {'path': str(final_root/'public_aux_frozen_features_v1/COMPLETE.json'),
                         'sha256': '9a1207ce9fd546836559438108956cad5203b1d8621096cb1e4957259cf34d88',
                         'size_bytes': 193504})
    need(feature['completed'] is True and feature['PASS'] is True, 'Final public features completed')
    for b in feature['outputs'].values():
        add(b, 'final_public_features')
    trained, tcb = load(final_root/'final_public_aux_matched_v1/COMPLETE.json',
                        'final_matched_fit',
                        {'path': str(final_root/'final_public_aux_matched_v1/COMPLETE.json'),
                         'sha256': 'ecb5e038217653187b798e926691eaace02ecbe40a9cf7548940236feb9c03cb',
                         'size_bytes': 12903})
    need(trained['completed'] is True and trained['PASS'] is True, 'Final matched fit completed')
    for key in ('checkpoints', 'logs', 'draw_ledgers', 'common_uniform_streams',
                'fit_admission', 'freeze', 'head_state_after_readout_QA', 'manifest',
                'row_exposures', 'summary', 'source_outputs', 'public_outputs'):
        for b in bindings(trained[key]):
            add(b, 'final_matched_fit')
    for name in ('prepare_v261_final_fit.py', 'replay_v261_public_selection.py'):
        add(measure(R/name), 'attempts_code_protocols', evidence='Root explicitly named admin CODE bytes measured')

    # Root authorized generated current/gallery publication bindings only, not original image inputs.
    for dirname, digest, size, group in (
        ('frozen_current_comparison_v1', '8ed2a77eada63ead5545869f714f8fd22b5d19686624fd02f8fd315093e481dd', 10165, 'final_current_generated'),
        ('final_current_crop_gallery_v1', 'fc88482ec7f68a0dda311a8c31e29b59377aefabb292ad9999720054495b54f9', 12576, 'final_current_gallery')):
        directory = final_root/dirname
        complete, cb = load(directory/'COMPLETE.json', group,
                            {'path': str(directory/'COMPLETE.json'), 'sha256': digest, 'size_bytes': size})
        need(complete['completed'] is True and complete['PASS'] is True, 'Root final current/gallery complete')
        for b in bindings(complete):
            if lexical(b['path']).is_relative_to(directory):
                add(b, group)

    # Literal CODE / fixed geometry / official encoder closure only. Never follow old arrays/heads.
    for value in metadata:
        for b in bindings(value):
            text = b['path'].casefold()
            if any(t in text for t in DENIED):
                inert.append({'binding': b, 'reason': 'Rejected lexically before source FS'})
                continue
            name = Path(b['path']).name
            if name.endswith('.py'):
                if b['path'] in source_paths:
                    add(b, 'runtime_dependency_code')
                    continue
                actual = measure(b['path'])
                if actual != b:
                    stale.append({'saved_binding': b, 'actual_CODE_binding': actual,
                                  'action': 'Inert stale CODE pointer; preserved parent metadata; no replacement'})
                    continue
                add(b, 'runtime_dependency_code', evidence='literal CODE closure SHA/size independently measured')
            elif name == 'dinov2_vits14_reg4_pretrain.pth':
                add(b, 'official_frozen_encoder', 'runtime/official_encoder/'+name)
            elif name == 'pallox.obj':
                add(b, 'public_geometry', 'runtime/public_geometry/'+name)
            elif name in ('OFFICIAL_API_RESPONSE.json', 'OFFICIAL_README.md') and Path(b['path']).parent == R/'V255_PUBLIC_PALLET_METADATA_ROOT':
                add(b, 'official_public_source_metadata')
            elif name.endswith(MODEL_SUFFIXES+('.npy', '.npz')) and b['path'] not in source_paths:
                inert.append({'binding': b, 'reason': 'Old source/model/cache dependency only; no stat/read/copy. Root may reference existing publication; final new outputs added separately.'})

    ordered = sorted(rows.values(), key=lambda row: row['relative_path'])
    coverage = {g: [r['source'] for r in ordered if r['group'] == g] for g in sorted({r['group'] for r in ordered})}
    result = {'completed': True, 'proposal_only_not_package_or_admission': True,
              'files': ordered, 'coverage_contract': coverage, 'file_count': len(ordered),
              'group_counts': dict(Counter(r['group'] for r in ordered)),
              'source_raw_V4_files_once': 1142, 'all_teacher_groups': groups,
              'metadata_read_bindings': metadata_bindings,
              'previous_successful_raw_source_aliases': aliases,
              'archived_incomplete_parts': partials,
              'inert_exclusions_and_unfollowed_bindings': inert,
              'stale_historical_CODE_bindings': stale,
              'future_required_additions_by_Root': ['final saved fit/public/source/current arithmetic QA and CPU replay',
                  'final readout/report/status/registry prefixes and any additional named A/B CODE directories',
                  'this frozen inventory proposal and its postexecution FINAL_READY metadata'],
              'large_originals_above90MiB_only': 'portable_publication.py uses deterministic gzip and <=64MiB parts; exact originals restored before stage',
              'data_array_model_image_decodes': 0, 'model_or_private_or_current_body_reads': 0,
              'old_successful_raw_body_reads': 0, 'Git_network_copy_or_package': 0,
              'raw_source_leaf_FS_for_receipt_bindings': 0,
              'public_incomplete_parts_hashed_only': len(partials),
              'model_data_architecture_threshold_or_parent_selection': False}
    with target.open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')
    print(json.dumps({'completed': True, 'proposal': measure(target), 'files': len(ordered),
                      'groups': result['group_counts'], 'part_archives': len(partials),
                      'old_raw_aliases': len(aliases), 'stale_CODE_bindings': len(stale)}, sort_keys=True), flush=True)
    return result


if __name__ == '__main__':
    run()
