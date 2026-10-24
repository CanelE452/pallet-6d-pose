"""One prospective V91 source-only physical cue-region auxiliary fit.

Only ``generated`` is authorized during preparation. The actual run is guarded
by a separately approved immutable root entry. No current evaluation reader.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import hashlib
import importlib.util
import inspect
import json
import random
import shutil
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / 'FIT_PROTOCOL_BEFORE_SOURCE_ARRAYS.json'
FIELDS = ('p3_roi', 'local_rgb96', 'local_valid96')
REQUIRED_BINDINGS = ('model_code', 'mosaic_code', 'source_checkpoint', 'source_checkpoint_freeze',
    'source_checkpoint_completion', 'teacher_contract', 'teacher_seal', 'teacher_units', 'teacher_views',
    'source_grid', 'source_frames', 'source_pack_manifest', 'source_pack_seal', 'source_pack_QA',
    'source_pack_independent_QA')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def bind(path): return dict(path=str(Path(path).resolve()), sha256=sha(path))
def read(path): return json.loads(Path(path).read_text())
def lines(path): return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def check(binding):
    assert sha(binding['path']) == binding['sha256'], binding['path']
    return Path(binding['path'])


def write(path, value):
    with Path(path).open('x') as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def protocol():
    p = read(PROTOCOL)
    check(read(ROOT / 'FIT_PROTOCOL_BINDING.json')['protocol'])
    assert set(REQUIRED_BINDINGS) <= set(p['bindings'])
    for item in p['bindings'].values(): check(item)
    assert p['parameter_count'] == 25381 and p['forward_fields'] == list(FIELDS)
    assert p['optimizer']['updates'] == 2000 and p['selection']['every_updates'] == 100
    assert p['selection']['threshold'] == .5
    assert p['sampling']['batch_queries'] == 96 and p['sampling']['per_class'] == 48
    assert p['sampling']['seed'] == 20261023
    assert p['runtime']['torch'] == '2.5.1+cu121'
    assert p['selection']['unavailable_region_NLL'] == 'FIXED_ZERO_LOGIT_BCE_LOG2_DIAGNOSTIC_PENALTY'
    assert p['current223_reads_allowed'] is False and p['PnP_allowed'] is False
    source_freeze = read(p['bindings']['source_checkpoint_freeze']['path'])
    source_complete = read(p['bindings']['source_checkpoint_completion']['path'])
    assert source_freeze['completed'] is True and source_complete['completed'] is True
    assert source_freeze['checkpoint']['sha256'] == p['bindings']['source_checkpoint']['sha256']
    assert source_complete['model_freeze']['sha256'] == p['bindings']['source_checkpoint_freeze']['sha256']
    assert source_freeze['current_for_fit_selection'] is False
    return p


def sample_terms(class_indices, class_probabilities, rng):
    return np.concatenate([rng.choice(class_indices[c], 48, replace=True, p=class_probabilities[c]) for c in (0, 1)])


def masked_region_loss(module, maps, valid, coordinates, labels, admitted):
    # Unknown coordinates/labels are not passed to interpolation or BCE.
    use = admitted.bool() & ((labels == 0) | (labels == 1))
    if not use.any(): return maps.sum() * 0, dict(eligible=0, factual_available=0)
    pooled, available = module.region_logit(maps[use], valid[use], coordinates[use])
    target = labels[use]
    if not available.any(): return maps.sum() * 0, dict(eligible=int(use.sum()), factual_available=0)
    loss = F.binary_cross_entropy_with_logits(pooled[available], target[available].to(pooled.dtype))
    return loss, dict(eligible=int(use.sum()), factual_available=int(available.sum()))


def factual_region_NLL(logit, available, labels):
    """All known validation units retain a prospective fixed abstention penalty.

    Zero-logit BCE is log(2), not a forced D/E prediction. Typed teacher U is
    excluded by all callers, even though numeric diagnostics can be stored.
    """
    z = np.where(available, logit, 0).astype(np.float32)
    return F.binary_cross_entropy_with_logits(torch.from_numpy(z),
        torch.from_numpy(np.clip(labels, 0, 1).astype(np.float32)), reduction='none').numpy()


def macro_class_loss(nll, labels, weights, mask):
    by_class = {}
    for c in (0, 1):
        use = mask & (labels == c)
        ww = weights[use].astype(np.float64)
        assert len(ww) and ww.sum() > 0
        by_class[str(c)] = dict(N=int(use.sum()), weight=float(ww.sum()),
            weighted_mean=float(np.sum(nll[use].astype(np.float64) * ww) / ww.sum()))
    return float(np.mean([x['weighted_mean'] for x in by_class.values()])), by_class


def confusion(labels, logits, available, mask):
    prediction = np.where(available, (logits >= 0).astype(np.int8), 2)
    cm = np.zeros((2, 3), np.int64)
    truth = labels[mask]; pp = prediction[mask]
    assert np.isin(truth, (0, 1)).all()
    np.add.at(cm, (truth, pp), 1)
    ratio = lambda n, d: float(n / d) if d else None
    return dict(N=int(mask.sum()), rows_D_E_cols_D_E_U=cm.tolist(),
        D_precision=ratio(cm[0, 0], cm[:, 0].sum()), D_recall=ratio(cm[0, 0], cm[0].sum()),
        E_precision=ratio(cm[1, 1], cm[:, 1].sum()), E_recall=ratio(cm[1, 1], cm[1].sum()),
        U_rate=ratio(cm[:, 2].sum(), cm.sum()))


def score_report(scored):
    rows = scored['rows']; y = scored['labels']; partitions = np.asarray([r['split'] for r in rows])
    result = {}
    for partition in ('TRAIN_LAB', 'VAL_HALL'):
        base = partitions == partition; known = base & np.isin(y, (0, 1))
        if not base.any(): continue
        section = dict(all_views=int(base.sum()), known_views=int(known.sum()), teacher_U=int(np.sum(base & (y == -1))))
        for kind in ('region', 'reference'):
            section[kind] = confusion(y, scored[kind+'_logit'], scored[kind+'_available'], known)
            pred = np.where(scored[kind+'_available'], scored[kind+'_logit'] >= 0, 2).astype(np.int8)
            section[kind+'_teacher_U_prediction_distribution'] = {str(c): int(np.sum(base & (y == -1) & (pred == c))) for c in (0, 1, 2)}
        lookup = {(r['cue_id'], r['view']): i for i, r in enumerate(rows) if r['split'] == partition}
        pairs = {}
        for after_label, name in ((1, 'D_to_E'), (0, 'D_to_D')):
            pair_indices = []
            for i, r in enumerate(rows):
                before = lookup.get((r['cue_id'], 'before'))
                if r['split'] == partition and r['view'] == 'after' and y[i] == after_label and before is not None and y[before] == 0:
                    pair_indices.append((before, i))
            pair_result = dict(N=len(pair_indices), effective_pair_weight=float(sum(rows[j]['duplicate_weight'] for _, j in pair_indices)))
            for kind in ('region', 'reference'):
                correct = [bool(scored[kind+'_available'][i] and scored[kind+'_available'][j]
                    and scored[kind+'_logit'][i] < 0 and (scored[kind+'_logit'][j] >= 0 if after_label == 1 else scored[kind+'_logit'][j] < 0)) for i, j in pair_indices]
                pair_result[kind] = dict(both_correct=int(sum(correct)), rate=float(np.mean(correct)) if correct else None)
            pairs[name] = pair_result
        section['paired_views'] = pairs
        result[partition] = section
    return result


def gates(report):
    val = report['VAL_HALL']; result = {}
    for kind in ('region', 'reference'):
        row = val[kind]
        for key, threshold, direction in (('D_recall', .8, 'ge'), ('E_recall', .8, 'ge'),
                                          ('E_precision', .5, 'ge'), ('U_rate', .1, 'le')):
            value = row[key]
            result[kind+'_'+key+'_'+direction] = value is not None and (value >= threshold if direction == 'ge' else value <= threshold)
    rate = val['paired_views']['D_to_E']['region']['rate']
    result['region_D_to_E_pair_both_correct_ge_0p7'] = rate is not None and rate >= .7
    return result


def load_inputs(p, device):
    """Only immutable V90 input pack; no GT parameter or annotation reader."""
    manifest = read(p['bindings']['source_pack_manifest']['path'])
    seal = read(p['bindings']['source_pack_seal']['path'])
    qa = read(p['bindings']['source_pack_QA']['path'])
    assert manifest['completed'] and manifest['tile_count'] == 12900 and manifest['frame_count'] == 100
    assert seal['completed'] and seal['PASS'] and qa['completed'] and qa['PASS']
    assert seal['manifest']['sha256'] == p['bindings']['source_pack_manifest']['sha256']
    tiles = lines(p['bindings']['source_grid']['path']); frames = lines(p['bindings']['source_frames']['path'])
    assert len(tiles) == 12900 and len(frames) == 100
    assert sum(f['split'] == 'TRAIN_LAB' for f in frames) == 44
    assert sum(f['split'] == 'VAL_HALL' for f in frames) == 56
    fmap = {f['frame_input_id']: i for i, f in enumerate(frames)}
    parts = {k: [] for k in ('p3_roi', 'rgb96', 'crop_valid96', 'query_xy', 'L', 'tile_index',
        'tile_id', 'split', 'image_frame_index')}
    end = 0
    for chunk in manifest['chunks']:
        assert chunk['start'] == end; end = chunk['end']; check(chunk['file'])
        with np.load(chunk['file']['path'], allow_pickle=False) as z:
            for k in parts: parts[k].append(z[k])
    assert end == 12900
    data = {k: np.concatenate(v) for k, v in parts.items()}
    assert np.array_equal(data['tile_index'], np.arange(12900))
    assert np.array_equal(data['tile_id'], [t['tile_id'] for t in tiles])
    assert np.array_equal(data['split'], [t['split'] for t in tiles])
    assert np.array_equal(data['query_xy'], [t['query_xy'] for t in tiles])
    assert np.array_equal(data['L'], [t['roi_side'] for t in tiles])
    assert np.array_equal(data['image_frame_index'], [fmap[t['frame_input_id']] for t in tiles])
    assert data['p3_roi'].shape == (12900, 65, 9, 9) and data['p3_roi'].dtype == np.float16
    assert data['rgb96'].shape == (12900, 3, 96, 96) and data['rgb96'].dtype == np.uint8
    assert data['crop_valid96'].shape == (12900, 1, 96, 96) and data['crop_valid96'].dtype == bool
    bank = dict(p3_roi=torch.from_numpy(data.pop('p3_roi')).to(device),
        local_rgb96=torch.from_numpy(data.pop('rgb96')).to(device),
        local_valid96=torch.from_numpy(data.pop('crop_valid96')).to(device))
    return bank, dict(tiles=tiles, frames=frames, data=data)


def load_terms(p, input_meta, device):
    """GT-only loss join. Does not add any field to the three-input bank."""
    with np.load(p['bindings']['teacher_units']['path'], allow_pickle=False) as z:
        terms = {k: z[k] for k in ('tile_index', 'cue_view_index', 'training_label', 'teacher_label',
            'loss_admitted', 'loss_weight', 'uncertainty9_normalized_xy_safe', 'full', 'uncertainty_inside_image')}
    assert terms['tile_index'].shape == (77916,)
    assert np.all((terms['tile_index'] >= 0) & (terms['tile_index'] < 12900))
    assert np.all((terms['cue_view_index'] >= 0) & (terms['cue_view_index'] < 604))
    assert terms['uncertainty9_normalized_xy_safe'].shape == (77916, 9, 2)
    assert np.isfinite(terms['uncertainty9_normalized_xy_safe']).all()
    known = np.isin(terms['training_label'], (0, 1))
    assert np.array_equal(known, terms['loss_admitted'])
    assert np.all((terms['loss_weight'] > 0) == known)
    assert np.all(~known | (terms['full'] & terms['uncertainty_inside_image']))
    split = np.asarray([t['split'] for t in input_meta['tiles']])[terms['tile_index']]
    train = split == 'TRAIN_LAB'
    pools = {c: np.flatnonzero(train & known & (terms['training_label'] == c)) for c in (0, 1)}
    probabilities = {c: terms['loss_weight'][pools[c]].astype(np.float64) for c in (0, 1)}
    masses = {str(c): float(probabilities[c].sum()) for c in (0, 1)}
    assert masses == {'0': 109., '1': 15.}
    for c in (0, 1): probabilities[c] /= probabilities[c].sum()
    loss_bank = dict(coordinates=torch.from_numpy(terms['uncertainty9_normalized_xy_safe']).to(device),
                     labels=torch.from_numpy(terms['training_label'].astype(np.int64)).to(device))
    return terms, pools, probabilities, loss_bank


def probe_nominal_term_support(bank, meta, terms, loss_bank, device):
    """Report factual support loss without adding an admission gate or retry.

    This uses only fixed validity arrays, not learned logit values. All nominal
    FULL terms remain in the corpus/sampler; unavailable sampled terms get zero
    gradient. Validation U retains the predeclared fixed log2 selection penalty.
    """
    available = np.zeros(77916, bool)
    nominal = np.flatnonzero(terms['loss_admitted'])
    with torch.inference_mode():
        for start in range(0, len(nominal), 128):
            ix = nominal[start:start+128]
            ti = torch.as_tensor(terms['tile_index'][ix], device=device)
            lv = bank['local_valid96'][ti].float()
            valid = (F.adaptive_avg_pool2d(lv, (9, 9))[:, 0] > 0) & bank['p3_roi'][ti, 64].bool()
            grid = loss_bank['coordinates'][torch.as_tensor(ix, device=device)][:, None]
            den = F.grid_sample(valid[:, None].float(), grid, mode='bilinear', padding_mode='border', align_corners=False)[:, 0, 0]
            available[ix] = (den > 0).all(1).cpu().numpy()
    split = np.asarray([t['split'] for t in meta['tiles']])[terms['tile_index']]
    rows = []
    for partition in ('TRAIN_LAB', 'VAL_HALL'):
        for c in (0, 1):
            use = terms['loss_admitted'] & (split == partition) & (terms['training_label'] == c)
            weights = terms['loss_weight']
            rows.append(dict(split=partition, label=c, nominal_terms=int(use.sum()), available_terms=int(np.sum(use & available)),
                unavailable_terms=int(np.sum(use & ~available)), nominal_effective_mass=float(weights[use].sum()),
                available_effective_mass=float(weights[use & available].sum()), unavailable_effective_mass=float(weights[use & ~available].sum())))
    return available, rows


def predict_mosaics(model, bank, meta, frame_indices, mosaic_module, device):
    """No GT argument, reference read or teacher-based tile/mask selection."""
    frame_indices = list(frame_indices)
    selected_frames = set(frame_indices)
    ix = np.asarray([i for i, f in enumerate(meta['data']['image_frame_index']) if int(f) in selected_frames], np.int64)
    assert len(ix) == 129 * len(frame_indices)
    model.eval(); all_maps = []; all_valid = []
    with torch.inference_mode():
        for start in range(0, len(ix), 128):
            index = torch.as_tensor(ix[start:start+128], device=device)
            output = model(**{k: v[index] for k, v in bank.items()})
            all_maps.append(output['physical_external_logit'].float().cpu().numpy())
            all_valid.append(output['map_valid'].cpu().numpy())
    maps = np.concatenate(all_maps); valid = np.concatenate(all_valid)
    assert maps.shape == valid.shape == (len(ix), 9, 9) and np.isfinite(maps).all()
    lookup = {int(t): j for j, t in enumerate(ix)}
    mosaics = {}
    for fi in frame_indices:
        frame = meta['frames'][fi]
        ti = np.flatnonzero(meta['data']['image_frame_index'] == fi)
        rows = [lookup[int(t)] for t in ti]
        mosaic, available, support = mosaic_module.make_frame_mosaic(maps[rows], valid[rows],
            meta['data']['query_xy'][ti], meta['data']['L'][ti], frame['raw_hw'], device=device)
        assert mosaic.shape == tuple(frame['raw_hw']) and np.isfinite(mosaic).all()
        mosaics[frame['frame_input_id']] = dict(frame_index=fi, logit=mosaic, valid=available, support=support)
    assert len(mosaics) == len(frame_indices)
    return dict(mosaics=mosaics, tile_indices=ix, tile_logit=maps, tile_valid=valid,
                boundary=dict(all_fixed_tiles_processed=True, frames_completed=len(frame_indices),
                    tiles_processed=len(ix), GT_reference_or_label_used_in_mosaic=False))


def score_completed_mosaics(prediction, teacher_view_path, mosaic_module):
    """Only after all requested frame mosaics are complete, read scoring GT."""
    assert prediction['boundary']['GT_reference_or_label_used_in_mosaic'] is False
    viewrows = lines(teacher_view_path)  # Deliberately outside map construction.
    assert len(viewrows) == 604
    rows = [r for r in viewrows if r['frame_input_id'] in prediction['mosaics']]
    region = []; reference = []; region_valid = []; reference_valid = []
    for row in rows:
        native = prediction['mosaics'][row['frame_input_id']]
        if row['reference_xy_unchanged'] is None:
            region.append(0.); reference.append(0.); region_valid.append(False); reference_valid.append(False)
            continue
        u = row['uncertainty9_xy_GT_only']; r = row['reference_xy_unchanged']
        values, available = mosaic_module.sample_native_region(native['logit'], native['valid'], u)
        rvalue, ravailable = mosaic_module.sample_native_region(native['logit'], native['valid'], [r])
        ok = bool(available.all()); rok = bool(ravailable[0])
        region.append(float(values.mean()) if ok else 0.)
        reference.append(float(rvalue[0]) if rok else 0.)
        region_valid.append(ok); reference_valid.append(rok)
    result = dict(rows=rows, region_logit=np.asarray(region, np.float32), reference_logit=np.asarray(reference, np.float32),
        region_available=np.asarray(region_valid, bool), reference_available=np.asarray(reference_valid, bool),
        labels=np.asarray([r['teacher_label'] for r in rows], np.int8),
        weights=np.asarray([r['duplicate_weight'] for r in rows], np.float64))
    for kind in ('region', 'reference'):
        result[kind+'_NLL_FP32_DIAGNOSTIC_U_CLAMPED'] = factual_region_NLL(result[kind+'_logit'], result[kind+'_available'], result['labels'])
    return result


def validate(model, bank, meta, p, mosaic_module, device, frame_indices):
    prediction = predict_mosaics(model, bank, meta, frame_indices, mosaic_module, device)
    scored = score_completed_mosaics(prediction, p['bindings']['teacher_views']['path'], mosaic_module)
    mask = np.asarray([r['split'] == 'VAL_HALL' for r in scored['rows']]) & np.isin(scored['labels'], (0, 1))
    assert mask.sum() == 170
    value, classes = macro_class_loss(scored['region_NLL_FP32_DIAGNOSTIC_U_CLAMPED'],
        scored['labels'], scored['weights'], mask)
    return value, classes, prediction, scored


def save_scored(path, scored):
    rows = scored['rows']
    arrays = {k: v for k, v in scored.items() if k != 'rows'}
    arrays.update(cue_view_index=np.asarray([r['cue_view_index'] for r in rows], np.int32),
        cue_id=np.asarray([r['cue_id'] for r in rows]), pair_id=np.asarray([r['pair_id'] for r in rows]),
        frame_input_id=np.asarray([r['frame_input_id'] for r in rows]), split=np.asarray([r['split'] for r in rows]),
        view=np.asarray([r['view'] for r in rows]), reference_defined=np.asarray([r['reference_xy_unchanged'] is not None for r in rows]))
    np.savez_compressed(path, **arrays)


def generated():
    """Generated CPU arrays/parameters only; never read protocol/pack/weights."""
    torch.set_num_threads(1); torch.manual_seed(20261023)
    module = load_module(ROOT / 'local_cue_evidence_map.py', 'generated_localcue')
    mosaic = load_module(ROOT / 'spatial_mosaic.py', 'generated_mosaic')
    model = module.LocalCueEvidenceMap()
    assert tuple(inspect.signature(model.forward).parameters) == FIELDS
    assert sum(p.numel() for p in model.parameters()) == 25381
    source = {k: torch.randn_like(v) for k, v in model.state_dict().items() if k.startswith(('p3_encoder.', 'local_encoder.'))}
    source['old_GT_classifier.weight'] = torch.randn(2, 3)
    before_fusion = {k: v.clone() for k, v in model.state_dict().items() if k.startswith('fusion.')}
    copied = module.initialize_shared_encoders(model, source)
    assert all(k.startswith(('p3_encoder.', 'local_encoder.')) for k in copied['copied'])
    assert all(torch.equal(model.state_dict()[k], v) for k, v in before_fusion.items())
    assert not model.evidence.weight.any() and not model.evidence.bias.any()
    fake = dict(p3_roi=torch.randn(3, 65, 9, 9), local_rgb96=torch.randint(0, 256, (3, 3, 96, 96), dtype=torch.uint8),
        local_valid96=torch.ones(3, 1, 96, 96, dtype=torch.bool))
    fake['p3_roi'][:, 64] = 1
    result = model(**fake)
    assert result['physical_external_logit'].shape == (3, 9, 9) and not result['physical_external_logit'].any()
    try: model(**fake, uncertainty=torch.zeros(3, 9, 2))
    except TypeError: pass
    else: raise AssertionError('GT must not enter forward')
    maps = torch.randn(5, 9, 9, requires_grad=True); valid = torch.ones_like(maps, dtype=torch.bool)
    coords = torch.zeros(5, 9, 2); coords[2:] = float('nan')
    labels = torch.tensor([0, 1, -1, 1, 0]); admitted = torch.tensor([1, 1, 0, 0, 0], dtype=torch.bool)
    loss, count = masked_region_loss(module, maps, valid, coords, labels, admitted); loss.backward()
    assert count == {'eligible':2, 'factual_available':2}
    assert maps.grad[:2].abs().sum() > 0 and not maps.grad[2:].any()
    empty = torch.randn(2, 9, 9, requires_grad=True)
    zero, _ = masked_region_loss(module, empty, torch.ones_like(empty, dtype=torch.bool), torch.full((2, 9, 2), float('nan')),
        torch.full((2,), -1), torch.zeros(2, dtype=torch.bool)); zero.backward()
    assert zero == 0 and not empty.grad.any()
    unavailable = torch.randn(2, 9, 9, requires_grad=True)
    uzero, ucount = masked_region_loss(module, unavailable, torch.zeros_like(unavailable, dtype=torch.bool), torch.zeros(2, 9, 2),
        torch.tensor([0, 1]), torch.ones(2, dtype=torch.bool)); uzero.backward()
    assert ucount == {'eligible':2, 'factual_available':0} and uzero == 0 and not unavailable.grad.any()
    pools = {0: np.array([0, 1]), 1: np.array([2, 3])}; probability = {0: np.array([.25, .75]), 1: np.array([.6, .4])}
    rng = np.random.default_rng(20261023); rng2 = np.random.default_rng(20261023)
    for _ in range(20):
        a = sample_terms(pools, probability, rng); b = sample_terms(pools, probability, rng2)
        assert np.array_equal(a, b) and np.isin(a[:48], [0, 1]).all() and np.isin(a[48:], [2, 3]).all()
    # Every map is fused first; GT-only scoring is a separate callable afterward.
    logit, available, _ = mosaic.make_frame_mosaic(np.full((1, 9, 9), 2., np.float32), np.ones((1, 9, 9), bool),
        np.array([[8., 8.]]), np.array([16]), (17, 17))
    vv, av = mosaic.sample_native_region(logit, available, [[8., 8.], [6., 6.]])
    assert av.all() and np.all(vv == 2.)
    penalty = factual_region_NLL(np.array([20., -20.]), np.array([False, False]), np.array([0, 1]))
    assert np.array_equal(penalty, np.array([np.log(2), np.log(2)], np.float32))
    y = np.array([0, 0, 1, 1]); z = np.array([-1., 1., 1., -1.]); av = np.array([1, 0, 1, 1], bool)
    assert confusion(y, z, av, np.ones(4, bool))['rows_D_E_cols_D_E_U'] == [[1, 0, 1], [1, 1, 0]]
    nll = np.array([.1, .3, .5, .9]); weights = np.array([.5, .5, 1, 1]);
    macro, _ = macro_class_loss(nll, y, weights, np.ones(4, bool)); assert abs(macro-.45) < 1e-12
    best = float('inf'); chosen = None
    for update, score in [(0, 0.), (100, .5), (200, .4), (300, .4), (400, .6)]:
        if update and score < best: best, chosen = score, update
    assert chosen == 200
    report = dict(completed=True, PASS=True, trainer=bind(__file__), model=bind(ROOT / 'local_cue_evidence_map.py'),
        mosaic=bind(ROOT / 'spatial_mosaic.py'), parameters=25381, forward_fields=list(FIELDS),
        GT_forward_rejected=True, transferred_encoders_only=True, new_fusion_preserved_final_output_zero=True,
        unknown_MISS_PARTIAL_loss_gradient_zero=True, all_unknown_graph_zero=True, factual_unavailable_loss_gradient_zero=True,
        class48_weighted_replacement_sampler_reproducible=True, GTfree_mosaic_then_GTsample=True,
        unavailable_NLL_fixed_log2_not_prediction=True, known_weighted_class_macro=True,
        CM_D_E_U_exact=True, earliest_strict_min_update0_excluded=True,
        source_feature_GT_or_weight_reads=0, actual_model_forward_fit_current_PnP=0,
        generated_model_forward=True, generated_CPU_only=True)
    write(ROOT / 'GENERATED_TRAINER_TESTS.json', report)
    print(json.dumps(report))


def freeze():
    p = protocol(); generated_proof = read(ROOT / 'GENERATED_TRAINER_TESTS.json')
    assert generated_proof['PASS'] and generated_proof['trainer'] == bind(__file__)
    assert generated_proof['model']['sha256'] == p['bindings']['model_code']['sha256']
    assert generated_proof['mosaic']['sha256'] == p['bindings']['mosaic_code']['sha256']
    review_path = ROOT / 'independent_prefit_review/REVIEW.json'; review = read(review_path)
    assert review['PASS'] is True and review['blockers'] == []
    assert review['trainer']['sha256'] == sha(__file__) and review['protocol']['sha256'] == sha(PROTOCOL)
    write(ROOT / 'TRAINER_ENTRYLOCK.json', dict(approved=False, protocol=bind(PROTOCOL), trainer=bind(__file__),
        generated=bind(ROOT / 'GENERATED_TRAINER_TESTS.json'), independent_review=bind(review_path),
        run_directory=p['run_directory'], restart_after_first_update=False,
        source_only=True, current_read_inference_allowed=False, PnP=0))


def run(entry_path):
    p = protocol(); entry = read(entry_path)
    assert entry['approved'] is True and entry['protocol'] == bind(PROTOCOL) and entry['trainer'] == bind(__file__)
    for k in ('generated', 'independent_review'): check(entry[k])
    assert read(entry['generated']['path'])['PASS'] and read(entry['independent_review']['path'])['PASS']
    assert entry['restart_after_first_update'] is False and entry['current_read_inference_allowed'] is False and entry['PnP'] == 0
    run_dir = Path(p['run_directory']); assert run_dir == Path(entry['run_directory']) and not run_dir.exists()
    assert Path(sys.executable).resolve() == Path(p['runtime']['python']).resolve()
    assert str(torch.__version__) == p['runtime']['torch'] and torch.cuda.is_available()
    run_dir.mkdir(parents=True); (run_dir / 'code_snapshot').mkdir()
    for path in (Path(__file__), PROTOCOL, Path(p['bindings']['model_code']['path']), Path(p['bindings']['mosaic_code']['path'])):
        shutil.copy2(path, run_dir / 'code_snapshot' / path.name)
    write(run_dir / 'RUN_STARTED.json', dict(protocol=bind(PROTOCOL), entry=bind(entry_path), trainer=bind(__file__), optimizer_updates=0,
        source_only=True, restart_after_first_update=False, current_PnP=0))
    torch.set_num_threads(1); torch.manual_seed(20261023); torch.cuda.manual_seed_all(20261023)
    np.random.seed(20261023); random.seed(20261023)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.cuda.reset_peak_memory_stats(); started=time.monotonic(); device='cuda:0'
    module=load_module(p['bindings']['model_code']['path'], 'immutable_v91_model')
    mosaic=load_module(p['bindings']['mosaic_code']['path'], 'immutable_v91_mosaic')
    bank, meta=load_inputs(p, device)
    terms, pools, probabilities, loss_bank=load_terms(p, meta, device)
    nominal_available, nominal_support_report=probe_nominal_term_support(bank,meta,terms,loss_bank,device)
    np.savez_compressed(run_dir/'NOMINAL_GT_TERMS_FACTUAL_SUPPORT_DIAGNOSTIC.npz',
        unit_index=np.arange(77916,dtype=np.int32),loss_admitted=terms['loss_admitted'],factual_available=nominal_available)
    model=module.LocalCueEvidenceMap().to(device)
    initialization=module.initialize_shared_encoders(model, torch.load(p['bindings']['source_checkpoint']['path'], map_location=device, weights_only=True))
    assert sum(x.numel() for x in model.parameters()) == 25381
    assert not model.evidence.weight.any() and not model.evidence.bias.any()
    old=[v for k,v in model.named_parameters() if k.startswith(('p3_encoder.', 'local_encoder.'))]
    new=[v for k,v in model.named_parameters() if k.startswith(('fusion.', 'evidence.'))]
    assert len(old)+len(new)==len(list(model.parameters()))
    optimizer=torch.optim.AdamW([{'params':old,'lr':1e-4},{'params':new,'lr':3e-4}],weight_decay=.001)
    val_frames=[i for i,f in enumerate(meta['frames']) if f['split']=='VAL_HALL']
    rng=np.random.default_rng(20261023)
    def batch_loss(indices):
        ti=terms['tile_index'][indices]; tile_indices=torch.as_tensor(ti,device=device)
        result=model(**{k:v[tile_indices] for k,v in bank.items()})
        term_indices=torch.as_tensor(indices,device=device)
        return masked_region_loss(module,result['physical_external_logit'],result['map_valid'],
            loss_bank['coordinates'][term_indices],loss_bank['labels'][term_indices],torch.ones(len(indices),dtype=torch.bool,device=device))
    pre_indices=sample_terms(pools,probabilities,np.random.default_rng(20261023))
    loss,count=batch_loss(pre_indices); assert torch.isfinite(loss)
    loss.backward(); model.zero_grad(set_to_none=True)
    assert count['eligible']==96
    write(run_dir/'SOURCE_GPU_PREFLIGHT.json',dict(PASS=True,optimizer_updates=0,loss=float(loss.detach()),
        training_term_counts={str(c):len(pools[c])for c in (0,1)},
        training_class_effective_mass={str(c):float(terms['loss_weight'][pools[c]].sum())for c in (0,1)},
        sampled96_factual_available=count['factual_available'],initialization=initialization,
        nominal_term_support_by_split_class=nominal_support_report,new_all_supported_admission_gate=False,
        parameter_groups=dict(transferred_lr=1e-4,new_lr=3e-4,weight_decay=.001),current_PnP=0))
    torch.save({k:v.detach().cpu()for k,v in model.state_dict().items()},run_dir/'UNTRAINED_UPDATE0.pt')
    baseline,baseclass,baseprediction,basescored=validate(model,bank,meta,p,mosaic,device,val_frames)
    write(run_dir/'BASELINE0_RESULTS.json',dict(selection_eligible=False,macroBCE=baseline,per_class=baseclass,metrics=score_report(basescored)))
    save_scored(run_dir/'BASELINE0_VAL_SCORES.npz',basescored);del baseprediction,basescored
    best=float('inf');best_update=None;history=[]
    cumulative_sampled_available=0;cumulative_sampled_nominal=0
    for update in range(1,2001):
        indices=sample_terms(pools,probabilities,rng)
        model.train();optimizer.zero_grad(set_to_none=True)
        loss,count=batch_loss(indices);assert torch.isfinite(loss)
        assert count['eligible']==96
        cumulative_sampled_available+=count['factual_available'];cumulative_sampled_nominal+=count['eligible']
        loss.backward();gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);assert torch.isfinite(gradient)
        optimizer.step()
        if update==1:
            write(run_dir/'FIRST_UPDATE.json',dict(update=1,restart_allowed=False,loss=float(loss.detach()),current_PnP=0))
            print('FIRST_UPDATE_COMMITTED',flush=True)
        if update%100==0:
            criterion,pc,prediction,scored=validate(model,bank,meta,p,mosaic,device,val_frames)
            saved=criterion<best
            if saved:
                best=criterion;best_update=update
                torch.save({k:v.detach().cpu()for k,v in model.state_dict().items()},run_dir/'BEST_PHYSICAL_CUE_HEAD.pt')
            row=dict(update=update,train_loss=float(loss.detach()),heldout_macroBCE=criterion,per_class_BCE=pc,
                saved=saved,best_update=best_update,known_VAL_regions=170,validation_frames=56,validation_tiles=7224,
                region_factual_U=int(np.sum(np.isin(scored['labels'],[0,1])&~scored['region_available'])),
                cumulative_sampled_nominal=cumulative_sampled_nominal,cumulative_sampled_factual_available=cumulative_sampled_available,
                mosaic_before_scoring_GT=True,seconds=time.monotonic()-started)
            history.append(row);(run_dir/'HISTORY.json').write_text(json.dumps(history,indent=2)+'\n');print(json.dumps(row),flush=True)
            del prediction,scored
    model.load_state_dict(torch.load(run_dir/'BEST_PHYSICAL_CUE_HEAD.pt',map_location=device,weights_only=True));model.eval().requires_grad_(False)
    write(run_dir/'MODEL_FREEZE.json',dict(completed=True,optimizer_updates=2000,best_update=best_update,selection_macroBCE=best,
        checkpoint=bind(run_dir/'BEST_PHYSICAL_CUE_HEAD.pt'),model_code=p['bindings']['model_code'],mosaic_code=p['bindings']['mosaic_code'],
        trainer=bind(__file__),protocol=bind(PROTOCOL),external_threshold=.5,source_only=True,current_used_for_fit_selection=False,
        native8_mapping='UNKNOWN',human_gold=False,PnP=0,fully_solved=False))
    prediction=predict_mosaics(model,bank,meta,range(100),mosaic,device)
    scored=score_completed_mosaics(prediction,p['bindings']['teacher_views']['path'],mosaic)
    assert len(scored['rows'])==604
    save_scored(run_dir/'SOURCE_SCORES.npz',scored)
    np.savez_compressed(run_dir/'ALL12900_TILE_MAPS_INPUT_ONLY.npz',tile_index=prediction['tile_indices'],
        tile_id=np.asarray([t['tile_id']for t in meta['tiles']]),physical_external_logit=prediction['tile_logit'],map_valid=prediction['tile_valid'])
    (run_dir/'native_mosaics').mkdir();mosaics_manifest=[]
    for frame_id,value in prediction['mosaics'].items():
        fi=value['frame_index'];frame=meta['frames'][fi];path=run_dir/'native_mosaics'/f'frame_{fi:03d}.npz'
        np.savez_compressed(path,physical_external_logit=value['logit'],native_valid=value['valid'],input_support=value['support'])
        mosaics_manifest.append(dict(frame_index=fi,frame_input_id=frame_id,image_sha256=frame['image_sha256'],raw_hw=frame['raw_hw'],file=bind(path)))
    write(run_dir/'NATIVE_MOSAICS_INPUT_ONLY_MANIFEST.json',dict(completed=True,frame_count=100,tile_count=12900,GT_used_for_construction=False,frames=mosaics_manifest))
    result=score_report(scored);assert result['VAL_HALL']['paired_views']['D_to_E']['N']==22
    assert result['VAL_HALL']['paired_views']['D_to_D']['N']==63
    all_gates=gates(result)
    knownval=np.asarray([r['split']=='VAL_HALL'for r in scored['rows']])&np.isin(scored['labels'],[0,1])
    finalmacro,finalpc=macro_class_loss(scored['region_NLL_FP32_DIAGNOSTIC_U_CLAMPED'],scored['labels'],scored['weights'],knownval)
    write(run_dir/'SOURCE_RESULTS.json',dict(completed=True,model_freeze=bind(run_dir/'MODEL_FREEZE.json'),source_metrics=result,
        final_alltile_macroBCE=finalmacro,final_alltile_per_class=finalpc,history_selected_macroBCE=best,
        final_minus_history_macroBCE=finalmacro-best,checkpoint_selection_unchanged=True,
        predeclared_source_gate=all_gates,all_nine_pass=all(all_gates.values()),current_evaluation_admission=False,
        meaning='AI-conditional physical structural cue-region auxiliary. One bin/environment holdout; no native8 mapping, human gold or target identity input. Passing allows only separately frozen diagnostic preparation.'))
    write(run_dir/'FIT_COMPLETE.json',dict(completed=True,optimizer_updates=2000,best_update=best_update,
        model_freeze=bind(run_dir/'MODEL_FREEZE.json'),history=bind(run_dir/'HISTORY.json'),scores=bind(run_dir/'SOURCE_SCORES.npz'),
        tilemaps=bind(run_dir/'ALL12900_TILE_MAPS_INPUT_ONLY.npz'),mosaics=bind(run_dir/'NATIVE_MOSAICS_INPUT_ONLY_MANIFEST.json'),
        source_results=bind(run_dir/'SOURCE_RESULTS.json'),seconds=time.monotonic()-started,
        peak_GPU_bytes=torch.cuda.max_memory_allocated(),source_frames=100,source_tiles=12900,source_cue_views=604,current_PnP=0))
    print(json.dumps(dict(COMPLETE=bind(run_dir/'FIT_COMPLETE.json'),best_update=best_update,all_nine_pass=all(all_gates.values()))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['generated','freeze','run'])
    parser.add_argument('--entry',type=Path,default=ROOT/'TRAINER_ENTRYLOCK.json');args=parser.parse_args()
    if args.stage=='generated':generated()
    elif args.stage=='freeze':freeze()
    else:run(args.entry)
