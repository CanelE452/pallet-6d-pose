"""Source TRAIN augmentation compositor. GT is not opened or passed to a model.

SELF crops use their original scene SAM prompt; BEFORE family uses frozen
BEFORE-anchor SAM. Both origin metadata and new augmentation IDs are explicit.
"""
from pathlib import Path
import argparse, ast, hashlib, json, time
import numpy as np

V = Path(__file__).resolve().parent
P = V.parent
SCOPE = V / 'ROOT_SCOPE_BEFORE_NEW_AUGMENTATION_OR_FIT.json'
SCOPE_SHA = 'd8394deb1c0a5844ecb987447391d7fd61ce5aa80d073cf33adf76ff385fcb95'
SAMPLER = P / 'query_visibility_local_support_raw_head_20261024_v181/raw_cache_preparation/build_source_raw96_cache.py'
SAMPLER_SHA = '01b6be15ad7941384ef5a209b397ef1a0b240dec8b403564e5e54c40a702225f'
META = ('row_id', 'original_global_index', 'native_slot', 'query_xy', 'L', 'original_query_available')
KEYS8 = set(META) | {'rgb96', 'valid96'}


def require(ok, msg):
    if not bool(ok):
        raise AssertionError(msg)


def bind(p):
    p = Path(p).resolve()
    require(p.is_relative_to(P), 'Project only')
    require(not any(x in str(p).lower() for x in ('_v112', '_v169', '_v177', '_v178', '_v182', 'current223', 'synth_val')), 'Forbidden training source')
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return dict(path=str(p), sha256=h.hexdigest())


def checked(b):
    require(bind(b['path']) == b, 'Binding drift ' + b['path'])
    return Path(b['path'])


def read(p):
    return json.loads(Path(p).read_text())


def rows(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x]


def write(p, x):
    with Path(p).open('x') as f:
        json.dump(x, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype.kind == b.dtype.kind == 'U':
        return a.shape == b.shape and np.array_equal(a, b)
    return a.shape == b.shape and a.dtype == b.dtype and (np.array_equal(a, b, equal_nan=True) if a.dtype.kind in 'fc' else np.array_equal(a, b))


def qarray(q):
    return np.array([np.nan, np.nan] if q is None else q, np.float64)


def helper():
    require(bind(SAMPLER)['sha256'] == SAMPLER_SHA, 'Pinned sampler')
    names = {'require', 'same', 'qarray', 'validq', 'grid96', 'sample_sam'}
    nodes = [n for n in ast.parse(SAMPLER.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in names]
    require(len(nodes) == len(names), 'Pure sampler AST')
    ns = {'np': np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SAMPLER), 'exec'), ns)
    return ns


def self_test():
    h = helper()
    a, av = h['sample_sam'](None, None, None, False, [512, 512])
    require(not a.any() and not av.any(), 'Missing no fake availability')
    a, av = h['sample_sam'](np.zeros((8, 8), np.float32), [3., 3.], 4., True, [8, 8])
    require(not a.any() and av.all(), 'Strict zero threshold independent availability')
    require(same(np.array(['A'], dtype='U1'), np.array(['A'], dtype='U10')), 'Unicode padding is not identity')
    require(not same(np.array([1], np.int8), np.array([1], np.int64)), 'Numeric dtype is identity')
    return dict(PASS=True, checks=4, actual_source_pixels_logits_or_targets_read=0, code=bind(__file__))


def run(entry_path):
    e = read(entry_path)
    require(e.get('approved_training_input_composition') is True, 'Separate root composition entry')
    require(e['code'] == bind(__file__) and bind(SCOPE)['sha256'] == SCOPE_SHA, 'Frozen code/scope')
    require(read(checked(e['checks']))['PASS'], 'Meaningful invented checks')
    raw = read(checked(e['raw_complete']))
    before = read(checked(e['before_SAM_complete']))
    original = read(checked(e['original_SAM_complete']))
    source = rows(checked(e['original_TRAIN_manifest']))
    require(raw['PASS'] and raw['rows'] == 65536 and raw['SAM_arrays_present'] is False, 'Raw8 admission')
    require(raw['original_nominal4096_raw_valid_byteparity'] is True and raw['original_aug_inputU'] == 1264, 'Original nominal parity/U')
    require(before['PASS'] and original['PASS'], 'Both SAM receipts PASS')
    rr = rows(checked(raw['metadata']))
    require(len(rr) == 65536 and len(source) == 4096, 'All rows')
    with np.load(checked(raw['inputs']), allow_pickle=False) as z:
        require(set(z.files) == KEYS8, 'Only raw input whitelist')
        x = {k: z[k].copy() for k in z.files}
    require(x['rgb96'].shape == (65536, 96, 96, 3) and x['rgb96'].dtype == np.uint8 and x['valid96'].dtype == bool, 'Raw shapes')
    require(x['original_global_index'].tolist() == list(range(65536)) and int((~x['original_query_available']).sum()) == 1264, 'New IDs/U')
    old_groups = {g['scene_index']: g for g in original['scenes']}
    new_groups = {(g['base_index'], g['variant']): g for g in before['groups']}
    require(len(old_groups) == len(new_groups) == 512, 'Both512 SAM group maps')
    h = helper()
    states = read(SCOPE)['training_synthetic']['states15']
    mask = np.zeros((65536, 96, 96), bool)
    available = np.zeros_like(mask)
    cache = {}
    used = {}
    facts = []
    start = time.monotonic()
    for i, r in enumerate(rr):
        family, native = r['family'], r['native_slot']
        require(r['original_global_index'] == i and r['split'] == 'TRAIN', 'Aug row order/split')
        for k in ('row_id', 'original_global_index', 'native_slot', 'original_query_available'):
            require(x[k][i] == r[k], 'Raw metadata ' + k)
        require(same(x['query_xy'][i], qarray(r['query_xy'])) and same(np.array(x['L'][i]), np.array(np.nan if r['L'] is None else r['L'], np.float64)), 'Aug q/L identity')
        im = source[r['source_image_original_global_index']]
        anchor = source[r['anchor_original_global_index']]
        require(im['scene_index'] == r['scene_index'] and im['native_slot'] == anchor['native_slot'] == native, 'Original image/native origins')
        if family == 'SELF_SCENE15':
            require(i < 61440 and i // 15 == r['source_image_original_global_index'] == r['anchor_original_global_index'], 'SELF original row/state')
            state = states[i % 15]
            q = None if anchor['original_query_xy'] is None else (np.array(anchor['original_query_xy'], np.float64) + [state['dx'], state['dy']]).tolist()
            L = None if anchor['original_L'] is None else anchor['original_L'] * state['scale']
            require(r['query_xy'] == q and r['L'] == L, 'Prospective crop-only transform')
            g = old_groups[r['scene_index']]
            meta = g['original_input_metadata8'][native]
            require(meta == anchor, 'SELF SAM original metadata remains original')
            logical = ('SELF', r['scene_index'])
            expected_id, expected_occ = anchor['original_global_index'], anchor['original_occurrence_id']
        else:
            require(family == 'BEFORE_ANCHOR4' and i >= 61440 and anchor['variant'] == 'BEFORE' and anchor['base_index'] == r['base_index'], 'BEFORE family origins')
            require(r['query_xy'] == anchor['original_query_xy'] and r['L'] == anchor['original_L'], 'BEFORE unchanged anchor')
            g = new_groups[(r['base_index'], r['variant'])]
            logical = ('BEFORE', r['scene_index'])
            expected_id, expected_occ = i, r['row_id']
        require(bool(x['original_query_available'][i]) == anchor['original_input_flags']['query_available'], 'Original anchor availability')
        output = g['outputs']
        if logical not in cache:
            if output is None:
                require(not g['prompt_available'] or not g['runtime_available'], 'Missing SAM status explicit')
                cache[logical] = None
            else:
                with np.load(checked(output), allow_pickle=False) as z:
                    cache[logical] = {k: z[k].copy() for k in ('image_logits', 'box_xyxy', 'original_query_xy', 'original_L', 'native_slot', 'original_global_index', 'original_occurrence_id', 'original_query_available', 'prediction_available')}
                require(cache[logical]['image_logits'].dtype == np.float32 and cache[logical]['image_logits'].shape == (512, 512) and np.isfinite(cache[logical]['image_logits']).all(), 'Finite full image SAM logits')
                used[output['path']] = output
        z = cache[logical]
        if z is None:
            logits, pred_av = None, False
        else:
            require(z['image_logits'].dtype == np.float32 and z['image_logits'].shape == (512, 512), 'Full image SAM')
            require(same(z['box_xyxy'], np.array(g['box_xyxy'], np.float64)), 'Exact original anchor context box')
            require(int(z['native_slot'][native]) == native and int(z['original_global_index'][native]) == expected_id and str(z['original_occurrence_id'][native]) == expected_occ, 'Explicit old or new SAM identity')
            require(same(z['original_query_xy'][native], qarray(anchor['original_query_xy'])) and same(z['original_L'][native], np.float64(np.nan if anchor['original_L'] is None else anchor['original_L'])), 'SAM prompt metadata is original anchor, never jittered pixel alias')
            require(bool(z['original_query_available'][native]) == bool(x['original_query_available'][i]), 'SAM original availability')
            logits, pred_av = z['image_logits'], bool(z['prediction_available'][native])
        mask[i], available[i] = h['sample_sam'](logits, r['query_xy'], r['L'], pred_av, [512, 512])
        facts.append(dict(row_id=r['row_id'], family=family, scene_index=r['scene_index'], native_slot=native,
            source_image_original_global_index=r['source_image_original_global_index'], anchor_original_global_index=r['anchor_original_global_index'], SAM_output=output,
            prompt_unchanged_by_local_crop_jitter=family == 'SELF_SCENE15', sam_mask96_count=int(mask[i].sum()), sam_available96_count=int(available[i].sum())))
        # Bound the full-logit cache while processing the compact row ordering.
        if i % 4096 == 4095:
            cache.clear()
            print(json.dumps(dict(rows=i+1, seconds=time.monotonic()-start)), flush=True)
    x.update(sam_mask96=mask, sam_available96=available)
    out = Path(e['output_dir']).resolve()
    require(out.parent == V and not out.exists(), 'Fresh output')
    out.mkdir()
    inp = out / 'SYNTH_AUG65536_INPUTS.npz'
    np.savez_compressed(inp, **x)
    fp = out / 'ALL65536_SAM_INPUT_LINEAGE.jsonl'
    fp.write_text(''.join(json.dumps(r) + '\n' for r in facts))
    for b in used.values():
        checked(b)
    checked(raw['inputs'])
    result = dict(completed=True, PASS=True, scope=bind(SCOPE), code=bind(__file__), entry=bind(entry_path), raw_complete=e['raw_complete'],
        rows=65536, domain='SYNTH_AUG65536', inputs=bind(inp), targets=raw['targets'], metadata=raw['metadata'], scene_metadata=raw['scene_metadata'],
        original_aug_inputU=1264, SAM_arrays_present=True, SAM_lineage=bind(fp), original_SAM_complete=e['original_SAM_complete'], before_SAM_complete=e['before_SAM_complete'],
        original_nominal4096_raw_valid_byteparity=True, target_body_reads=0, source_TRAIN_only=True, model_forward_fit_or_GTinput=0,
        fit_admission=True, selected_model=None, fully_solved=False, seconds=time.monotonic()-start)
    write(out / 'COMPLETE.json', result)
    print(json.dumps(dict(completed=True, PASS=True, complete=bind(out / 'COMPLETE.json'))))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--self-test', action='store_true')
    g.add_argument('--entry', type=Path)
    args = ap.parse_args()
    print(json.dumps(self_test())) if args.self_test else run(args.entry)
