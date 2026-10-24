"""Build source-only SAM2 input identities. Never decode RGB or read features."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent
V66 = BASE / 'query_visibility_controlled_real_20261022_v66'
V63 = BASE / 'query_visibility_public_real_20261022_v63'
HERE = ROOT / 'source_inputs'
JOINED = V66 / 'joined_source_cache/CONTROLLED_PLUS_PALLOX_VAL.npz'
EXPECTED = dict(SYNTH=(1742, 235), ORIGINAL_YCB=(3200, 400), CONTROLLED_YCB=(3200, 400), PALLOX=(448, 56))
IDENTITY = ('sample_id', 'group_id', 'image_sha256', 'source', 'global_id', 'global_source',
            'global_image_sha256', 'global_box', 'target_index')
FORBIDDEN = {'x', 'rgb_native', 'sam', 'context', 'box_grid', 'global_rgb', 'global_sam', 'global_prompt'}
ACCESSED = []

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def ref(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))

def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def load_json(path):
    return json.loads(Path(path).read_text())

def read_arrays(path, keys):
    assert not (set(keys) & FORBIDDEN)
    with np.load(path, allow_pickle=False) as z:
        assert all(k in z.files for k in keys)
        values = {k: z[k] for k in keys}
    ACCESSED.append(dict(path=str(Path(path).resolve()), arrays=list(keys)))
    return values

def array_binding(name, value):
    a = np.ascontiguousarray(value)
    return dict(name=name, dtype=str(a.dtype), shape=list(a.shape), raw_array_sha256=hashlib.sha256(a.tobytes()).hexdigest())

def metadata(path):
    records = [json.loads(line) for line in Path(path).open()]
    assert len({r['id'] for r in records}) == len(records)
    return {r['id']: r for r in records}

def whitelist(row, keys):
    assert set(row) == set(keys), sorted(set(row) ^ set(keys))

def main():
    sealpath = HERE / 'SEAL.json'
    assert not sealpath.exists(), 'Preserve an earlier manifest'
    protocol = load_json(ROOT / 'PROTOCOL.json')
    assert protocol['fixed_inputs']['queries'] == 8590 and protocol['fixed_inputs']['targets'] == 1091
    merge_seal_path = V66 / 'joined_source_cache/SEAL.json'
    merge_seal = load_json(merge_seal_path)
    assert merge_seal['complete'] and not merge_seal['current223_used']
    registered = next(x for x in merge_seal['outputs'] if x['candidate'] == 'CONTROLLED_PLUS_PALLOX' and x['split'] == 'VAL')
    assert (V66 / registered['path']).resolve() == JOINED.resolve()
    assert JOINED.stat().st_size == registered['bytes']
    merge_lock_path = V66 / merge_seal['input_lock']['path']
    assert sha(merge_lock_path) == merge_seal['input_lock']['sha256']
    lock = load_json(merge_lock_path)
    assert lock['validation_sources'] == list(EXPECTED)
    frozen = {str(Path(b['path']).resolve()): b for b in lock['bindings']}
    source_files = dict(
        legacy=V63 / 'sam_source/INPUTS.jsonl',
        controlled=V66 / 'source_features/INPUTS.jsonl',
        pallox=V66 / 'public_pallet/PAIRED_QUERIES/INPUTS_RGB_ONLY.jsonl')
    # Freeze the recipe and document identities before any output construction.
    input_lock = dict(schema='V68_SOURCE_MANIFEST_INPUT_RECIPE_LOCK_V1', code=ref(__file__), protocol=ref(ROOT / 'PROTOCOL.json'),
        prior_joined_seal=ref(merge_seal_path), prior_merge_lock=ref(merge_lock_path),
        joined_NPZ_prior_sealed_binding=registered,
        source_metadata=[ref(p) for p in source_files.values()],
        NPZ_access_rule='Named identity/query/box arrays only; joined dense_valid is read later for native-valid parity audit only. No feature arrays.',
        NPZ_whole_file_SHA_policy='Trust immutable prior source seal/lock file SHA and exact size; no full compressed feature payload hash/read in this builder.',
        source_RGB_policy='Trust previously sealed image_SHA and metadata shape. File-existence metadata only; no RGB bytes or decoder.',
        no_GT_in_model_manifest=True, no_model_forward=True, no_current_read=True)
    recipe = HERE / 'MANIFEST_INPUT_LOCK.json'
    assert not recipe.exists()
    write(recipe, input_lock)
    data = read_arrays(JOINED, IDENTITY)
    n = len(data['sample_id']); t = len(data['global_id'])
    assert (n, t) == (8590, 1091)
    assert len(set(data['sample_id'])) == n and len(set(data['global_id'])) == t
    assert data['target_index'].dtype == np.int64 and set(data['target_index']) == set(range(t))
    assert np.array_equal(data['source'], data['global_source'][data['target_index']])
    assert np.array_equal(data['image_sha256'], data['global_image_sha256'][data['target_index']])
    assert data['global_box'].shape == (t, 4)
    catalogs = {name: metadata(path) for name, path in source_files.items()}
    witnesses = {}
    witness_bindings = []
    for source, paths in lock['sources']['VAL'].items():
        path = Path(paths.get('label', paths['feature']))
        binding = frozen[str(path.resolve())]
        assert path.stat().st_size > 0
        if source == 'SYNTH':
            keys = ('sample_id', 'id', 'native_corner', 'group_id', 'image_sha256', 'global_id',
                    'global_image_sha256', 'global_box', 'target_index')
        else:
            keys = ('sample_id', 'native_corner', 'image_sha256', 'global_id', 'global_image_sha256',
                    'global_box', 'target_index', 'query', 'side')
        w = read_arrays(path, keys)
        w['sample_lookup'] = {str(s): i for i, s in enumerate(w['sample_id'])}
        w['global_lookup'] = {str(s): i for i, s in enumerate(w['global_id'])}
        assert len(w['sample_lookup']) == len(w['sample_id']) and len(w['global_lookup']) == len(w['global_id'])
        witnesses[source] = w
        witness_bindings.append(dict(source=source, prior_frozen_binding=binding,
            accessed_identity_array_bindings=[array_binding(k, w[k]) for k in keys]))
    rows = []
    targets = []
    alias_counts = Counter()
    query_keyset = ('sample_id', 'joined_query_index', 'native_corner', 'query', 'pixel', 'native_input_valid')
    target_keyset = ('id', 'global_id', 'joined_target_index', 'source', 'source_metadata_label', 'group_id',
                     'split', 'image', 'image_sha256', 'raw_hw', 'box', 'side', 'queries')
    for target_index in range(t):
        gid = str(data['global_id'][target_index]); source = str(data['global_source'][target_index]); w = witnesses[source]
        wi = w['global_lookup'][gid]
        assert str(w['global_image_sha256'][wi]) == str(data['global_image_sha256'][target_index])
        assert np.array_equal(w['global_box'][wi], data['global_box'][target_index])
        if source in ('SYNTH', 'ORIGINAL_YCB'):
            m = catalogs['legacy'][gid]
            original_label = m['source']
            assert original_label == ('SYNTH' if source == 'SYNTH' else 'YCB')
            assert m['split'] == 'validation'
        elif source == 'CONTROLLED_YCB':
            m = catalogs['controlled'][gid]; original_label = 'ABSENT_SOURCE_KEY_IN_CONTROLLED_MANIFEST'
            assert 'source' not in m and m['split'] == 'VALIDATION'
        else:
            m = catalogs['pallox'][gid]; original_label = m['source']
            assert original_label == 'PALLOX' and m['split'] == 'VALIDATION'
            assert m['global_id'] == gid and m['query8'] == m['query']
        assert m['id'] == gid and np.array_equal(m['box'], data['global_box'][target_index])
        assert m['image_sha256'] == str(data['global_image_sha256'][target_index])
        assert m['raw_hw'] == [480, 640] and np.asarray(m['query']).shape == (8, 2)
        assert Path(m['image']).is_file() and np.isfinite(m['query']).all() and np.isfinite(m['box']).all()
        assert 0 <= m['box'][0] < m['box'][2] <= 639 and 0 <= m['box'][1] < m['box'][3] <= 479
        indices = np.flatnonzero(data['target_index'] == target_index)
        groups = {str(data['group_id'][i]) for i in indices}; assert len(groups) == 1
        group = next(iter(groups))
        if 'group_id' in m:
            assert m['group_id'] == group
        queries = []
        for i in indices:
            sid = str(data['sample_id'][i]); j = w['sample_lookup'][sid]
            k = int(w['native_corner'][j]); assert 0 <= k < 8
            assert sid == ('SYNTH:' + gid + ':' + str(k) if source == 'SYNTH' else gid + ':' + str(k))
            assert str(w['global_id'][w['target_index'][j]]) == gid
            assert str(w['image_sha256'][j]) == str(data['image_sha256'][i]) == m['image_sha256']
            q = np.asarray(m['query'][k], np.float64)
            if source == 'SYNTH':
                assert str(w['id'][j]) == gid and str(w['group_id'][j]) == group
            else:
                assert np.array_equal(w['query'][j], q) and float(w['side'][j]) == m['side']
            pixel = np.floor(q + .5).astype(np.int64)
            valid = bool(0 <= q[0] < 640 and 0 <= q[1] < 480 and 0 <= pixel[0] < 640 and 0 <= pixel[1] < 480)
            query = dict(sample_id=sid, joined_query_index=int(i), native_corner=k,
                         query=q.tolist(), pixel=pixel.tolist(), native_input_valid=valid)
            whitelist(query, query_keyset); queries.append(query)
            rows.append(dict(joined_query_index=int(i), joined_target_index=target_index, sample_id=sid, global_id=gid,
                source=source, group_id=group, native_corner=k, query_x=float(q[0]), query_y=float(q[1]),
                pixel_x=int(pixel[0]), pixel_y=int(pixel[1]), native_input_valid=valid,
                side=float(m['side']), image=m['image'], image_sha256=m['image_sha256']))
        assert len({x['native_corner'] for x in queries}) == len(queries)
        if source != 'SYNTH':
            assert {x['native_corner'] for x in queries} == set(range(8))
        target = dict(id=gid, global_id=gid, joined_target_index=target_index, source=source,
            source_metadata_label=original_label, group_id=group, split='VAL', image=m['image'],
            image_sha256=m['image_sha256'], raw_hw=m['raw_hw'], box=m['box'], side=m['side'], queries=queries)
        whitelist(target, target_keyset); targets.append(target)
        alias_counts[(source, original_label)] += 1
    rows.sort(key=lambda r: r['joined_query_index'])
    assert [r['sample_id'] for r in rows] == data['sample_id'].tolist()
    assert [r['global_id'] for r in rows] == data['global_id'][data['target_index']].tolist()
    # A ground-truth native-valid field is audit-only; it cannot select rows.
    native_truth = read_arrays(JOINED, ('dense_valid',))['dense_valid'][:, 4, 4]
    assert np.array_equal(native_truth, np.asarray([r['native_input_valid'] for r in rows]))
    counts = []
    for source, (expected_n, expected_t) in EXPECTED.items():
        rr = [r for r in rows if r['source'] == source]; tt = [x for x in targets if x['source'] == source]
        assert (len(rr), len(tt)) == (expected_n, expected_t)
        counts.append(dict(source=source, queries=len(rr), targets=len(tt), unique_RGB_SHA=len({x['image_sha256'] for x in tt}),
            native_input_valid=sum(r['native_input_valid'] for r in rr),
            target_query_count_histogram=dict(sorted(Counter(len(x['queries']) for x in tt).items()))))
    for path in (HERE / 'INPUTS_RGB_ONLY.jsonl', HERE / 'QUERY_ID_MAP.jsonl', HERE / 'QUERY_ID_MAP.csv'):
        assert not path.exists()
    (HERE / 'INPUTS_RGB_ONLY.jsonl').write_text(''.join(json.dumps(x, allow_nan=False) + '\n' for x in targets))
    (HERE / 'QUERY_ID_MAP.jsonl').write_text(''.join(json.dumps(x, allow_nan=False) + '\n' for x in rows))
    with (HERE / 'QUERY_ID_MAP.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    audit = dict(schema='V68_SOURCE_MANIFEST_IDENTITY_AUDIT_V1', PASS=True, source_counts=counts,
        source_aliases=[dict(canonical_source=s, metadata_source_label=a, targets=n) for (s, a), n in sorted(alias_counts.items())],
        exact_native_query_witness='SYNTH native_corner/id/sample/global index checked against oldcache identity; q8 from immutableV63metadata. Otherthree sourcequeries/side additionally bit-exact against label-only identity arrays.',
        query_order_preserved=True, target_order_preserved=True, missing_SYNTH_slots_added=0,
        native_valid_audit_mismatches=0, identity_or_box_or_SHA_or_query_mismatches=0,
        target_manifest_fields=list(target_keyset), query_manifest_fields=list(query_keyset),
        hidden_GT_or_mask_fields_in_model_manifest=0, fields_outside_exact_recursive_whitelist=0,
        NPZ_array_access_log=ACCESSED, input_witnesses=witness_bindings,
        joined_identity_array_bindings=[array_binding(k, data[k]) for k in IDENTITY],
        decoded_RGB=0, RGB_body_hash_reads=0, source_image_SHA_trusted_from_prior_seals=True,
        feature_arrays_read=0, model_weight_reads=0, model_forward_calls=0, optimizer_updates=0, GPU_calls=0, current_data_reads=0)
    auditpath = ROOT / 'source_audit/MANIFEST_IDENTITY_AUDIT.json'; assert not auditpath.exists(); write(auditpath, audit)
    assert sha(ROOT / 'PROTOCOL.json') == input_lock['protocol']['sha256'] and sha(__file__) == input_lock['code']['sha256']
    seal = dict(schema='V68_SOURCE_RGB_AND_QUERY_MANIFEST_SEAL_V1', completed=True, source_only=True,
        input_lock=ref(recipe), protocol=ref(ROOT / 'PROTOCOL.json'), builder=ref(__file__),
        queries=n, targets=t, source_counts=counts, native_valid=int(native_truth.sum()),
        inherited_joined_VAL_binding=registered, audit=ref(auditpath),
        outputs=[ref(HERE / name) for name in ('INPUTS_RGB_ONLY.jsonl', 'QUERY_ID_MAP.jsonl', 'QUERY_ID_MAP.csv')],
        GT_fields_in_model_input=0, no_source_forward_or_training=True, current_inference_authorized=False)
    write(sealpath, seal)
    print(json.dumps(dict(completed=True, seal=ref(sealpath), source_counts=counts, native_valid=seal['native_valid'],
                         hidden_GT_fields=0, decoded_RGB=0, feature_arrays_read=0, model_forward_calls=0)))

if __name__ == '__main__':
    main()
