"""Saved PUBLIC feature/metadata audit only. No producer or learned-model import."""
from pathlib import Path
import collections
import datetime
import hashlib
import json
import numpy as np

V = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
EXPECTED_RECEIPT = '25730178daf6db93071cf9d9b8244102099bdf5765fbed9a4c1a50e36952af3c'


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def bind(path):
    return dict(path=str(Path(path).resolve()), sha256=digest(path))


def pixel(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def main():
    assert not (OUT / 'REVIEW.json').exists(), 'No output overwrite'
    receipt_path = V / 'spatial_features_v1/FEATURE_COMPLETE.json'
    assert digest(receipt_path) == EXPECTED_RECEIPT
    receipt = read(receipt_path)
    assert receipt['completed'] is True and receipt['PASS'] is True
    verified = []

    def verify(binding):
        assert digest(binding['path']) == binding['sha256'], binding['path']
        verified.append(binding)
        return Path(binding['path'])

    for key in ('output', 'manifest', 'protocol', 'code', 'entry',
                'root_conditional_admission', 'corpus_seal', 'input_occurrences',
                'new_query_lineage', 'input_unit_lineage', 'input_RGB_spatial_facts'):
        verify(receipt[key])
    protocol = read(receipt['protocol']['path'])
    manifest = read(receipt['manifest']['path'])
    assert manifest['completed'] is True
    assert manifest['output'] == receipt['output']
    assert manifest['units'] == receipt['input_unit_lineage']
    assert manifest['occurrence_map'] == receipt['new_query_lineage']
    assert manifest['input_occurrences'] == receipt['input_occurrences']
    assert manifest['old_C11_features'] == protocol['old_output']
    old_path = verify(protocol['old_output'])
    units = rows(receipt['input_unit_lineage']['path'])
    maps = rows(receipt['new_query_lineage']['path'])
    occurrences = rows(receipt['input_occurrences']['path'])
    facts = rows(receipt['input_RGB_spatial_facts']['path'])
    assert len(units) == len(facts) == 152 and len(maps) == len(occurrences) == 924
    assert [u['input_unit_index'] for u in units] == list(range(152))
    assert [f['input_unit_index'] for f in facts] == list(range(152))
    with np.load(receipt['output']['path'], allow_pickle=False) as z:
        arrays = {k: np.array(z[k], copy=True) for k in z.files}
    expected_keys = {'features', 'occurrence_unit', 'unit_Q_inside', 'query_available',
                     'factual_query_available', 'context_available', 'global_occurrence_index'}
    assert set(arrays) == expected_keys
    x, mapping = arrays['features'], arrays['occurrence_unit']
    assert x.shape == (152, 1928) and x.dtype == np.float32 and np.isfinite(x).all()
    assert mapping.shape == (924,) and mapping.dtype == np.int32
    assert np.array_equal(np.unique(mapping), np.arange(152))
    assert arrays['global_occurrence_index'].dtype == np.int32
    assert np.array_equal(arrays['global_occurrence_index'], np.arange(924))
    for name in ('query_available', 'factual_query_available', 'context_available'):
        assert arrays[name].shape == (924,) and arrays[name].dtype == bool
    assert arrays['unit_Q_inside'].shape == (152,) and arrays['unit_Q_inside'].dtype == bool
    assert np.array_equal(arrays['query_available'],
                          arrays['factual_query_available'] & arrays['context_available'])
    with np.load(old_path, allow_pickle=False) as z:
        old = {k: np.array(z[k], copy=True) for k in
               ('features', 'occurrence_unit', 'unit_Q_inside', 'query_available',
                'factual_query_available', 'context_available')}
    old_features_equal = 0
    for u in units[:28]:
        i, original_unit, original_row = (u['input_unit_index'],
                                          u['old_C11_unit_index'],
                                          u['old_C11_occurrence_index'])
        assert u['kind'] == 'EXACT_C11_BEFORE_REUSE'
        assert int(old['occurrence_unit'][original_row]) == original_unit
        assert x[i].tobytes() == old['features'][original_unit].tobytes()
        assert arrays['unit_Q_inside'][i] == old['unit_Q_inside'][original_unit]
        for key in ('query_available', 'factual_query_available', 'context_available'):
            assert arrays[key][33 * i] == old[key][original_row]
        assert facts[i]['feature_vector_sha256'] == pixel(x[i])
        old_features_equal += 1
    next_fresh = 28
    for n, occurrence in enumerate(occurrences):
        base, member = divmod(n, 33)
        expected_variant = ('BEFORE' if member == 0 else
                            'OPAQUE_COVER' if member % 2 else 'CUE_PRESERVING_SHAM')
        assert occurrence['occurrence_index'] == n and occurrence['base_index'] == base
        assert occurrence['variant'] == expected_variant
        assert maps[n]['occurrence_index'] == n and maps[n]['occurrence_id'] == occurrence['occurrence_id']
        assert maps[n]['input_unit_index'] == int(mapping[n])
        unit = units[int(mapping[n])]
        for field in ('query_xy', 'L', 'raw_split'):
            assert maps[n][field] == occurrence[field] == unit[field]
        assert occurrence['frozen_context_box_xyxy'] == unit['box_xyxy']
        if occurrence['full_rgb_changed']:
            assert int(mapping[n]) == next_fresh and unit['kind'] == 'FRESH_CHANGED_FULLRGB'
            assert unit['occurrence_index'] == n
            next_fresh += 1
        else:
            assert int(mapping[n]) == base
        for key in ('fullRGB', 'decoded_RGB_sha256', 'rgb96_sha256'):
            assert occurrence[key] == unit[key]
        assert occurrence['crop_valid96_sha256'] == unit['valid96_sha256']
        for key in ('query_available', 'factual_query_available', 'context_available'):
            assert arrays[key][n] == arrays[key][base * 33]
    assert next_fresh == 152
    for i, (u, fact) in enumerate(zip(units, facts)):
        assert fact['kind'] == u['kind'] and fact['occurrence_index'] == u['occurrence_index']
        assert fact['RGB96_sha256'] == u['rgb96_sha256']
        assert fact['valid96_sha256'] == u['valid96_sha256']
        assert fact['Q_inside'] == bool(arrays['unit_Q_inside'][i])
        q = np.asarray(u['query_xy'], dtype=np.float64)
        box = np.asarray(u['box_xyxy'], dtype=np.float64)
        inside = bool(np.isfinite(q).all() and np.isfinite(box).all() and
                      (box[2:] > box[:2]).all() and (q >= box[:2]).all() and
                      (q <= box[2:]).all() and x[i, 1924] > 0)
        assert inside == bool(arrays['unit_Q_inside'][i])
        assert x[i, 1927] == np.float32(inside)
        if not inside:
            assert not x[i, 384:768].any()
        if i >= 28:
            assert fact['fresh_context_from_changed_fullRGB'] is True
            assert fact['native_RGB_sha256'] == u['decoded_RGB_sha256']
            assert np.array_equal(x[i, 1924:1927], np.asarray([
                fact['context_valid_fraction'], fact['local_valid_fraction'],
                fact['central_valid_fraction']], dtype=np.float32))
    assert ((x[:, 1924:1927] >= 0) & (x[:, 1924:1927] <= 1)).all()
    assert receipt['model_state_unchanged'] is True and receipt['first_batch_repeat_exact'] is True
    assert receipt['frozen_encoder_state_before'] == receipt['frozen_encoder_state_after']
    assert receipt['encoder_calls'] == 17 and receipt['max_views'] == 24
    assert receipt['available'] == int(arrays['query_available'].sum())
    assert receipt['factual_U'] == int((~arrays['factual_query_available']).sum())
    assert receipt['context_U'] == int((~arrays['context_available']).sum())
    descriptor_norms = {}
    for i, name in enumerate(('T', 'Q', 'F', 'C', 'P')):
        norms = np.linalg.norm(x[:, i * 384:(i + 1) * 384].astype(np.float64), axis=1)
        descriptor_norms[name] = {'minimum': float(norms.min()), 'maximum': float(norms.max()),
                                  'max_deviation_from_one': float(np.abs(norms - 1).max())}
    report = dict(
        completed=True, PASS=True, discrepancies=[],
        created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        feature_pack=receipt['output'], feature_receipt=bind(receipt_path),
        feature_manifest=receipt['manifest'], protocol=receipt['protocol'],
        preparation_code_review=bind(V / 'independent_feature_preparation_review/REVIEW.json'),
        audit_code=bind(__file__),
        population=dict(units=152, occurrences=924, reused_old_units_exact_bytes=old_features_equal,
                        fresh_changed_units=next_fresh-28,
                        factual_available=int(arrays['factual_query_available'].sum()),
                        query_available=int(arrays['query_available'].sum()),
                        context_available=int(arrays['context_available'].sum()),
                        Q_inside_units=int(arrays['unit_Q_inside'].sum())),
        verified=['All metadata/hash bindings and exact feature NPZ schema/finite32-bit values.',
                  'All924 original IDs/order/base/variant/q/L/context/rawfold maps and complete152-unit coverage.',
                  'All28 reused C11 feature vectors byte-identical, with original availability and Q-inside bits.',
                  'All124 changed units preserve own fullRGB/RGB96/context facts and distinct fresh unit references.',
                  'All152 Q-inside flags independently recomputed from original float64 q/context geometry.',
                  'All924 availability flags equal old factual/base geometry; query=factual AND context.',
                  'All fresh fractions equal saved feature scalars and valid ranges; unavailable flags never dropped.'],
        descriptive_descriptor_norms_only=descriptor_norms,
        producer_certificates=dict(state_unchanged=True, first_batch_repeat_exact=True,
                                   encoder_calls=17, primary_batches=16, repeat_batches=1),
        limitations=['No independent DINO encoder/weights/model execution or raw source pixel replay.',
                     'Fresh embeddings and RGB/context pixel hashes are checked through the bound producer facts, not independently re-encoded.',
                     'This review establishes source feature integrity, not semantic target/native-gold accuracy or training/current admission.'],
        actual_source_pixel_reads=0, model_or_weight_reads=0, model_forward=0,
        fit=0, current=False, V112=False, V117=False, PnP=0,
        bindings_verified=verified)
    report_path = OUT / 'REVIEW.json'
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    seal = dict(completed=True, PASS=True, report=bind(report_path),
                feature_pack=receipt['output'], feature_receipt=bind(receipt_path), audit_code=bind(__file__))
    (OUT / 'SEAL.json').write_text(json.dumps(seal, indent=2) + '\n')
    print(json.dumps(dict(report=bind(report_path), seal=bind(OUT/'SEAL.json'), population=report['population'])))


if __name__ == '__main__':
    main()
