"""Frozen public source query emulation; no RGB/features/model/current I/O.

Freeze protocol and complete TRAIN bank identities before deriving any error
vectors. Public queries are GT-derived source augmentation, not an initializer.
"""
import argparse
import ast
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
P = ROOT.parent
PREP = P.parent / 'external_ray_proxy_head_20261022_v60/PREPARED_INPUTS_SYNTH.json.gz'
CORPUS = ROOT / 'gt_corpus_build/GT_ONLY_TEACHER_CORPUS.npz'
INDEX = ROOT / 'gt_corpus_build/ORDERED_METADATA_INDEX_GT_ONLY.jsonl'
YCB = P / 'query_visibility_public_real_20261022_v63/ycb/instances.jsonl'
GRID = P.parent / 'external_ray_proxy_head_20261022_v60/head_and_roi.py'
BOX = P / 'query_visibility_cue_occlusion_20261023_v72/private_evaluation/current_input_adapter.py'
SALT = 'V76_PUBLIC_YCB_N3_TRAIN768_CORRELATED_NATIVE8_V1'
FIELDS = ['frame_index', 'id', 'source', 'split', 'image', 'image_sha256', 'raw_hw', 'initial_points', 'roi_side', 'box']


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def bind(p):
    return {'path': str(p), 'sha256': sha(p), 'bytes': Path(p).stat().st_size}


def clean(v):
    if isinstance(v, np.ndarray): return clean(v.tolist())
    if isinstance(v, dict): return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [clean(x) for x in v]
    if isinstance(v, (np.integer, np.bool_)): return v.item()
    if isinstance(v, (float, np.floating)): return float(v) if np.isfinite(v) else None
    return v


def write(p, d):
    with p.open('x') as f: json.dump(clean(d), f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')


def lines(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines()]


def pure(p, name):
    node = next(n for n in ast.parse(p.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)
    env = {'np': np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(p), 'exec'), env)
    return env[name]


def freeze():
    # Identity metadata only: no source initial_points or reference arrays used.
    idx = lines(INDEX)
    assert len(idx) == 5203
    bank = [{'bank_index': len([]), 'corpus_row': x['row_index'], 'id': x['id'], 'split': x['split']}
            for x in idx if x['source'] == 'SYNTHETIC' and x['split'] == 'train']
    for i, x in enumerate(bank): x['bank_index'] = i
    assert len(bank) == 768 and len(set(x['id'] for x in bank)) == 768
    write(OUT / 'ERROR_BANK_IDENTITIES_BEFORE_VALUES.json', {
        'completed': True, 'selection': 'ALL original N3 synthetic TRAIN768 in corpus order; no quality/state/matched/query-valid filtering',
        'heldout_rows_in_error_bank': 0, 'bank': bank, 'identity_index': bind(INDEX)})
    protocol = {
        'schema': 'V76_PUBLIC_QUERY_EMULATION_PROTOCOL_V1', 'frozen_before_delta_derivation_and_use': True,
        'scope': 'SOURCE ONLY. GT-derived augmentation; not an actual deployment initializer or human semantic evidence.',
        'bindings': {k: bind(v) for k, v in {'N3_prepared': PREP, 'teacher_corpus': CORPUS, 'ordered_index': INDEX,
            'YCB_instances': YCB, 'native_grid_producer': GRID, 'initial_box_producer': BOX,
            'teacher_seal': ROOT / 'gt_corpus_build/GT_ONLY_CORPUS_SEAL.json',
            'error_bank_identities': OUT / 'ERROR_BANK_IDENTITIES_BEFORE_VALUES.json', 'builder_code': Path(__file__)}.items()},
        'bank': {'count': 768, 'all_TRAIN_rows': True, 'quality_filter': False, 'native_order': 'original dynamic native slots0..7 preserved',
            'q_source_stage': 'SEALED_ACTUAL_N3_SUBPIX_F, original stored initial_points[:8]',
            'reference_source': 'same-id first1003 GT-only corpus native_GTref8',
            'normalizer': 'max(ptp(reference8,axis=0)); all8 original references, no per-slot filtering',
            'delta8': '(stored_q8 - reference8) / normalizer',
            'invalid_policy': 'Keep all768 rows. If nonfinite reference or nonpositive/nonfinite span, delta8 allNaN; if individual q nonfinite, preserve nonfinite delta. Report and retain mapped public rows; JSON nonfinite is null. No bank substitution.'},
        'mapping': {'salt': SALT, 'identity': 'YCB:{scene_id:06d}:{frame_id:06d}:{instance_id:06d}',
            'hash': 'SHA256((salt+"\\n"+identity).encode("utf-8"))',
            'bank_index': 'int.from_bytes(digest,"big") % 768',
            'query8': 'public_reference8 + selected_delta8 * max(ptp(public_reference8,axis=0))',
            'no_clip_no_reorder_no_query_refinement': True, 'native_slot_correlation': 'One complete8x2 bank vector per target; never independent slot draws'},
        'public_rows': {'count': 4200, 'unique_RGB': 1000, 'source_order': 'Existing YCB instances.jsonl order, exact corpus[1003:] id join',
            'split': 'Original scenes0..7 TRAIN3400,8..9 VAL800 unchanged. Both use synthetic TRAIN-only error bank.',
            'filter': 'None: no teacher, class, mask, matched, quality, current or crop-visibility filtering'},
        'input_geometry': {'L': 'Exact native_grid(initial_points,raw_hw)[2]: clip(.25*maxptp(finite non-(-1,-1) q8),16,max(H,W)); fewer2=>16',
            'box': 'Exact initial_query_box: finite non-(-1,-1) q8 minmax, lo=max(lo,0), hi=min(hi,[W-1,H-1]); fewer2 or anyspan<2 => null',
            'centroid_query': 'Not synthesized; exactly8 input points', 'helpers': 'Pure function AST only; no module imports/model loads'},
        'model_input_manifest': {'name': 'INPUTS_RGB_ONLY.jsonl', 'whitelist': FIELDS,
            'identity_fields_are_lineage_only': True, 'forward_forbidden': ['reference8', 'mask', 'teacher', 'label', 'state', 'GTpose', 'GTperm', 'errorbank_id', 'delta8', 'GTspan', 'reference_offset']},
        'loss_admission_proposal': {
            'teacher_state': 'Independent GT label8:1VISIBLE/0COVERED/-1U; preserve original label even when query misses, training target separatelyU.',
            'self': 'convex_self_proxy8_separate is visibility:1=may-visible,0=self-hidden,-1unknown; never self1=S and never merge into binary support target',
            'admit': 'teacher known0/1 & query_center_valid & ROI_anyvalid & local_valid96.any & target_valid128.any & finite GTref & (-.5 <= (r-q)/L < .5).all',
            'query_center_valid': 'finite/nonmissing & continuous native image bounds & floor(q+.5) nearest pixel bounds',
            'ROI_anyvalid': 'Exact native_grid9 continuous sampled coordinate bounds; input-only',
            'near_boundary': 'Retain already known teacher labels; no extra teacher quality filtering',
            'query_miss': 'Reference outside unchanged half-open local square => trainingU/loss0, even if underlying teacherCOVERED; never relabel miss toCOVERED',
            'unknown_and_invalid': 'U or invalid q/ref/input coverage => loss0; all rows retained',
            'fit_authorization': False, 'current_authorization': False},
        'execution': {'RGB_reads': False, 'feature_reads': False, 'model_forward': False, 'fit': False, 'current_data': False, 'PnP': False}}
    write(OUT / 'PROTOCOL_BEFORE_ERROR_VALUES.json', protocol)
    print('PROTOCOL_FROZEN', sha(OUT / 'PROTOCOL_BEFORE_ERROR_VALUES.json'), flush=True)


def good_query(q):
    return np.isfinite(q).all(-1) & ~(q == -1).all(-1)


def center_valid(q, hw):
    h, w = hw; good = good_query(q)
    pixel = np.floor(np.where(good[:, None], q, -1e8) + .5)
    cont = good & (q[:, 0] >= 0) & (q[:, 0] < w) & (q[:, 1] >= 0) & (q[:, 1] < h)
    nearest = good & (pixel[:, 0] >= 0) & (pixel[:, 0] < w) & (pixel[:, 1] >= 0) & (pixel[:, 1] < h)
    return cont, nearest


def any_bilinear(q, L, hw, n):
    h, w = hw
    if not np.isfinite(q).all() or not np.isfinite(L) or L <= 0: return False
    u = ((np.arange(n) + .5) / n - .5) * L
    return bool(np.any((q[0] + u >= -.5) & (q[0] + u < w - .5)) and
                np.any((q[1] + u >= -.5) & (q[1] + u < h - .5)))


def build():
    protocol = json.loads((OUT / 'PROTOCOL_BEFORE_ERROR_VALUES.json').read_text())
    for b in protocol['bindings'].values(): assert sha(b['path']) == b['sha256'], b['path']
    identities = json.loads((OUT / 'ERROR_BANK_IDENTITIES_BEFORE_VALUES.json').read_text())['bank']
    with gzip.open(PREP, 'rt') as f: prepared = json.load(f)
    assert isinstance(prepared, list) and len(prepared) == 1003
    prepared_by_id = {x['id']: x for x in prepared}; assert len(prepared_by_id) == 1003
    # Load only typed reference/identity/teacher fields, never any feature array.
    with np.load(CORPUS, allow_pickle=False) as z:
        ids = z['id']; sources = z['source']; splits = z['split']; refs = z['native_GTref8']
        labels = z['label8']; self_flags = z['convex_self_proxy8_separate']
    assert len(ids) == 5203 and np.all(sources[:1003] == 'SYNTHETIC') and np.all(sources[1003:] == 'YCB_PUBLIC_REAL')
    qbank = []; rbank = []; spans = []; delta = []; invalid_bank = []
    for b in identities:
        j = b['corpus_row']; row = prepared_by_id[b['id']]
        assert str(ids[j]) == b['id'] and str(splits[j]) == 'train' and row['split'] == 'train'
        q = np.asarray(row['initial_points'], np.float64)[:8]; r = refs[j].astype(np.float64)
        assert q.shape == r.shape == (8, 2)
        span = float(np.max(np.ptp(r, axis=0)))
        valid_span = bool(np.isfinite(r).all() and np.isfinite(span) and span > 0)
        d = (q - r) / span if valid_span else np.full((8, 2), np.nan)
        if not np.isfinite(d).all(): invalid_bank.append({'bank_index': b['bank_index'], 'id': b['id'], 'bad_delta_components': int((~np.isfinite(d)).sum())})
        qbank.append(q); rbank.append(r); spans.append(span); delta.append(d)
    qbank, rbank, spans, delta = map(np.asarray, [qbank, rbank, spans, delta])
    np.savez_compressed(OUT / 'ERROR_BANK_GT_ONLY.npz', id=np.asarray([b['id'] for b in identities]),
        corpus_row=np.asarray([b['corpus_row'] for b in identities]), q8=qbank, reference8=rbank, reference_span=spans, delta8=delta)
    rows = lines(YCB); assert len(rows) == 4200
    grid = pure(GRID, 'native_grid'); boxfun = pure(BOX, 'initial_query_box')
    counts = Counter(); stats = Counter(); mapping = []; q_all = []; side_all = []; box_all = []
    qvalid_all = []; roi_all = []; inside_all = []; target_all = []; local_all = []; admitted_all = []
    manifest_path = OUT / 'INPUTS_RGB_ONLY.jsonl'
    with manifest_path.open('x') as f:
        for i, row in enumerate(rows):
            identity = f'YCB:{row["scene_id"]:06d}:{row["frame_id"]:06d}:{row["instance_id"]:06d}'
            j = i + 1003; assert str(ids[j]) == identity and str(splits[j]) == row['split']
            hw = [row['height'], row['width']]; r = refs[j].astype(np.float64)
            span = float(np.max(np.ptp(r, axis=0)))
            digest = hashlib.sha256((SALT + '\n' + identity).encode('utf-8')).digest()
            bi = int.from_bytes(digest, 'big') % len(delta)
            q = r + delta[bi] * span
            _, gv, L = grid(q, hw); box = boxfun(q, hw)
            rec = dict(frame_index=i, id=identity, source='YCB_PUBLIC_REAL', split=row['split'],
                image=row['image'], image_sha256=row['image_sha256'], raw_hw=hw, initial_points=q, roi_side=L, box=box)
            assert list(rec) == FIELDS
            f.write(json.dumps(clean(rec), separators=(',', ':'), allow_nan=False) + '\n')
            cv, nv = center_valid(q, hw); qv = cv & nv; roi = gv.reshape(8, -1).any(1)
            local = np.asarray([any_bilinear(p, L, hw, 96) for p in q])
            target = False if box is None else any_bilinear((box[:2] + box[2:]) / 2, float(1.25 * max(box[2:] - box[:2])), hw, 128)
            offset = (r - q) / L
            inside = np.isfinite(r).all(1) & np.isfinite(offset).all(1) & ((offset >= -.5) & (offset < .5)).all(1)
            known = np.isin(labels[j], [0, 1]); admit = known & qv & roi & local & target & inside
            prefix = row['split'] + ':'
            for name, mask in {'teacher_known': known, 'teacher_U': ~known, 'query_center_valid': qv, 'ROI_anyvalid': roi,
                'local_anyvalid': local, 'reference_in_local_square': inside, 'admitted': admit,
                'known_query_miss': known & ~inside, 'known_invalid_query': known & ~qv,
                'admitted_VISIBLE': admit & (labels[j] == 1), 'admitted_COVERED': admit & (labels[j] == 0)}.items(): stats[prefix + name] += int(mask.sum())
            counts[row['split']] += 1; counts['bank:' + str(bi)] += 1
            stats[prefix + 'slots'] += 8; stats[prefix + 'target_anyvalid_rows'] += int(target)
            stats[prefix + 'nonfinite_query_components'] += int((~np.isfinite(q)).sum())
            stats[prefix + 'box_null_rows'] += int(box is None)
            mapping.append({'frame_index': i, 'id': identity, 'corpus_row': j, 'split': row['split'], 'selected_bank_index_GT_ONLY': bi,
                'selected_bank_id_GT_ONLY': identities[bi]['id'], 'SHA256_digest': digest.hex(), 'public_reference_span_GT_ONLY': span})
            q_all.append(q); side_all.append(L); box_all.append(np.full(4, np.nan) if box is None else box)
            qvalid_all.append(qv); roi_all.append(roi); inside_all.append(inside); target_all.append(target); local_all.append(local); admitted_all.append(admit)
    with (OUT / 'BANK_MAPPING_GT_ONLY.jsonl').open('x') as f:
        for row in mapping: f.write(json.dumps(clean(row), separators=(',', ':'), allow_nan=False) + '\n')
    np.savez_compressed(OUT / 'QUERY_ADMISSION_GT_ONLY.npz', id=ids[1003:], source=sources[1003:], split=splits[1003:],
        query8=np.asarray(q_all), L=np.asarray(side_all), box=np.asarray(box_all), query_center_valid=np.asarray(qvalid_all),
        ROI_anyvalid=np.asarray(roi_all), local_anyvalid=np.asarray(local_all), target_anyvalid=np.asarray(target_all),
        reference_in_local_square=np.asarray(inside_all), underlying_teacher_label8=labels[1003:],
        training_label8=np.where(np.asarray(admitted_all), labels[1003:], -1).astype(np.int8), loss_admission=np.asarray(admitted_all),
        convex_self_visible_proxy8_separate=self_flags[1003:])
    summary = {'completed': True, 'rows': 4200, 'slots': 33600, 'unique_RGB': len(set(r['image_sha256'] for r in rows)),
        'split_rows': {k: counts[k] for k in ['train', 'val']}, 'bank_rows': 768, 'bank_invalid_rows_preserved': invalid_bank,
        'bank_used_unique': sum(counts['bank:' + str(i)] > 0 for i in range(768)), 'counts': dict(stats),
        'no_rows_filtered': True, 'no_query_clipping_or_reordering': True, 'source_emulation_not_actual_initializer': True,
        'model_manifest_contains_GT_fields': False, 'self_flag_meaning': '1=convex may-visible;0=self-hidden;-1U; independent of binarylocal support',
        'protocol': bind(OUT / 'PROTOCOL_BEFORE_ERROR_VALUES.json'), 'builder': bind(Path(__file__)),
        'RGB_feature_weight_current_reads': 0, 'forward_fit_PnP': 0}
    write(OUT / 'BUILD_SUMMARY.json', summary)
    print('BUILD_COMPLETE', summary['split_rows'], dict(stats), flush=True)


def verify():
    protocol = json.loads((OUT / 'PROTOCOL_BEFORE_ERROR_VALUES.json').read_text())
    for b in protocol['bindings'].values(): assert sha(b['path']) == b['sha256']
    manifest = lines(OUT / 'INPUTS_RGB_ONLY.jsonl'); mappings = lines(OUT / 'BANK_MAPPING_GT_ONLY.jsonl'); rows = lines(YCB)
    with np.load(CORPUS, allow_pickle=False) as z: refs = z['native_GTref8']; ids = z['id']; splits = z['split']
    with np.load(OUT / 'ERROR_BANK_GT_ONLY.npz', allow_pickle=False) as z:
        delta = z['delta8']; qbank = z['q8']; rbank = z['reference8']; spanbank = z['reference_span']; bankids = z['id']
    np.testing.assert_array_equal(delta, (qbank - rbank) / spanbank[:, None, None])
    assert len(manifest) == len(mappings) == len(rows) == 4200
    assert len(set(r['id'] for r in manifest)) == 4200
    grid = pure(GRID, 'native_grid'); boxfun = pure(BOX, 'initial_query_box')
    for i, (m, b, y) in enumerate(zip(manifest, mappings, rows)):
        assert list(m) == FIELDS and m['frame_index'] == i and m['id'] == str(ids[i + 1003]) == b['id']
        assert m['split'] == y['split'] == str(splits[i + 1003]) and m['image'] == y['image'] and m['image_sha256'] == y['image_sha256']
        bi = int.from_bytes(hashlib.sha256((SALT + '\n' + m['id']).encode()).digest(), 'big') % 768
        assert bi == b['selected_bank_index_GT_ONLY'] and str(bankids[bi]) == b['selected_bank_id_GT_ONLY']
        r = refs[i + 1003]; q = np.asarray(m['initial_points'], np.float64)
        np.testing.assert_array_equal(q, r + delta[bi] * np.max(np.ptp(r, axis=0)))
        _, _, L = grid(q, m['raw_hw']); assert L == m['roi_side']
        expected_box = boxfun(q, m['raw_hw'])
        if expected_box is None: assert m['box'] is None
        else: np.testing.assert_array_equal(expected_box, np.asarray(m['box']))
    write(OUT / 'EXACT_FORMULA_AND_IDENTITY_VERIFICATION.json', {'PASS': True, 'all4200_formula_mapping_order_native_fields_exact': True,
        'all768_error_formula_exact': True, 'manifest_GT_field_whitelist_exact': FIELDS, 'originals_unchanged': True,
        'RGB_feature_weight_current_reads': 0, 'forward_fit_PnP': 0})
    write(OUT / 'SEAL.json', {'completed': True, 'scope': 'Source-only GT-derived publicqueryemulation, not deployedinitializer',
        'protocol': bind(OUT / 'PROTOCOL_BEFORE_ERROR_VALUES.json'), 'code': bind(Path(__file__)),
        'outputs': {p.name: bind(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name not in ['SEAL.json']},
        'fit_or_current_authorized': False})
    print('EXACT_VERIFY_PASS_SEAL', sha(OUT / 'SEAL.json'), flush=True)


if __name__ == '__main__':
    a = argparse.ArgumentParser(); a.add_argument('mode', choices=['freeze', 'build', 'verify']); args = a.parse_args()
    {'freeze': freeze, 'build': build, 'verify': verify}[args.mode]()
