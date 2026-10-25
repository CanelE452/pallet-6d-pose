"""Root's final named metadata/CODE/attribution supplement; no payload decoding."""
import hashlib
import json
import os
from pathlib import Path
import stat

R = Path(__file__).absolute().parent.parent
H = Path(__file__).absolute().parent


def safe(p):
    p = Path(p)
    assert p.is_absolute() and '..' not in p.parts
    assert not any(t in str(p).casefold() for t in ('v112', 'cache', '.pyc'))
    for a in [*reversed(p.parents), p]:
        assert not stat.S_ISLNK(os.lstat(a).st_mode)
    return p


def raw(p):
    with os.fdopen(os.open(safe(p), os.O_RDONLY | os.O_NOFOLLOW), 'rb') as f:
        return f.read()


def bind(p):
    data = raw(p)
    return {'path': str(p), 'sha256': hashlib.sha256(data).hexdigest(), 'size_bytes': len(data)}


def all_bindings(v):
    if isinstance(v, dict):
        if {'path', 'sha256', 'size_bytes'} <= set(v) and type(v['path']) is str:
            yield {k: v[k] for k in ('path', 'sha256', 'size_bytes')}
        for x in v.values():
            yield from all_bindings(x)
    elif isinstance(v, list):
        for x in v:
            yield from all_bindings(x)


def run():
    base = json.loads(raw(H/'ALLOWLIST_PROPOSAL.json'))
    prior_path = H/'NAMED_CODE_POLICY_SUPPLEMENT.json'
    prior = json.loads(raw(prior_path))
    rows = list(prior['additional_files'])
    seen = {r['source']['path'] for r in base['files']+rows}

    def add(b, group, rel=None):
        if b['path'] in seen:
            return
        p = Path(b['path'])
        assert p.is_absolute() and '..' not in p.parts and not any(t in str(p).casefold() for t in ('v112', 'cache', '.pyc'))
        rel = rel or 'artifacts/'+p.relative_to(R).as_posix()
        rows.append({'relative_path': rel, 'group': group, 'source': b,
                     'inventory_evidence': 'Root final named metadata/CODE measured or exact receipt output binding; arrays/models not read'})
        seen.add(b['path'])

    prep = R/'V261_ROOT_FINAL_PUBLIC_REAL_COMPARISON/teacher_feature_entry_preparation_v1'
    prep_complete = json.loads(raw(prep/'COMPLETE.json'))
    for name in sorted(os.listdir(safe(prep))):
        if any(t in name.casefold() for t in ('v112', 'cache', '.pyc')):
            continue
        p = prep/name
        assert stat.S_ISREG(os.lstat(p).st_mode) and p.suffix == '.json'
        add(bind(p), 'teacher_feature_entry_preparation')
    for b in all_bindings(prep_complete):
        if Path(b['path']).is_relative_to(prep):
            add(b, 'teacher_feature_entry_preparation')
    add(bind(R/'prepare_v261_teacher_groups.py'), 'attempts_code_protocols')

    cpu = R/'V261_PUBLIC_CPU_REPLAY_A'
    for name in sorted(os.listdir(safe(cpu))):
        if any(t in name.casefold() for t in ('v112', 'cache', '.pyc')):
            continue
        p = cpu/name
        mode = os.lstat(p).st_mode
        assert not stat.S_ISLNK(mode)
        if stat.S_ISREG(mode):
            assert p.suffix in ('.py', '.json', '.md', '.txt')
            add(bind(p), 'public_CPU_replay')
    cb = bind(cpu/'actual_review_v1/COMPLETE.json')
    assert cb['sha256'] == 'b4f52258a2479c037b347368c281bd9692be4569e756a719e2f8b1a5be314c1b' and cb['size_bytes'] == 2186
    add(cb, 'public_CPU_replay')
    complete = json.loads(raw(cpu/'actual_review_v1/COMPLETE.json'))
    for b in all_bindings(complete):
        if Path(b['path']).parent == cpu/'actual_review_v1' and Path(b['path']).name in ('SUMMARY.json', 'REPLAY_ROWS.csv'):
            add(b, 'public_CPU_replay')

    # Root explicitly admitted only these two leaf names under the known vendor root.
    entry = json.loads(raw(R/'V260_ROOT_PUBLIC_FEATURE_PILOT/ACTUAL_PUBLIC_SINGLE_QUERY_FEATURE_ENTRY.json'))
    vendor = safe(entry['model_loader_config']['vendor_root'])
    licenses = []
    for name in ('LICENSE', 'NOTICE'):
        p = vendor/name
        try:
            mode = os.lstat(p).st_mode
        except FileNotFoundError:
            licenses.append({'named_leaf': name, 'exists': False})
            continue
        assert stat.S_ISREG(mode) and not stat.S_ISLNK(mode)
        b = bind(p)
        add(b, 'official_runtime_attribution', 'runtime/official_encoder/'+name)
        licenses.append({'named_leaf': name, 'exists': True, 'binding': b})
    result = {'completed': True, 'proposal_only': True,
              'base_proposal': bind(H/'ALLOWLIST_PROPOSAL.json'), 'preserved_prior_supplement': bind(prior_path),
              'additional_files': sorted(rows, key=lambda row: row['relative_path']), 'additional_count': len(rows),
              'named_vendor_attribution': licenses,
              'runtime_vendor_revision': entry['model_loader_config']['asset_identity']['vendor_revision'],
              'runtime_limit': prior['runtime_limit'], 'existing_source_packet_limit': prior['existing_source_packet_limit'],
              'data_arrays_checkpoint_models_images_current_rows_read': 0,
              'Git_network_copy_or_package': 0,
              'Root_remaining_additions': ['final readout/figures/status/third-party prose/registry prefixes',
                                           'Root manual review/report and all inventory own files including this supplement/FINAL_READY']}
    path = H/'PUBLICATION_SUPPLEMENT_V2.json'
    with path.open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write('\n')
    print(json.dumps({'completed': True, 'supplement': bind(path), 'additional_files': len(rows)}, sort_keys=True))


if __name__ == '__main__':
    run()
