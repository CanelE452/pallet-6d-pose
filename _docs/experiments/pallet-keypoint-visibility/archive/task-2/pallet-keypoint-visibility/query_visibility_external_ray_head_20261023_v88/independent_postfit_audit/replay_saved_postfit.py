"""Independent saved-only V88 audit. No torch or trainer/model imports.

Preparation: --generated uses only arrays created below.
Actual: requires a parent-authorized exact completed FIT_COMPLETE SHA and the
previously frozen preparation lock. No source result is opened by preparation.
"""
import argparse
import ast
import csv
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SCOPE = HERE / 'REVIEW_SCOPE_BEFORE_FIT_COMPLETE.json'
NUMERICAL_DISCUSSION_ATOL = 1e-6
FORWARD_FIELDS = ('p3_roi', 'local_rgb96', 'local_valid96', 'target_rgb128', 'target_valid128', 'context42')
SCORE_KEYS = {'id', 'split', 'external_logit', 'input_available', 'query_center_valid',
              'loss_admitted_GT_only', 'reference_inside_qcrop_GT_only', 'NLL_FP32'}
GT_KEYS = ('id', 'split', 'label8', 'reason8', 'reference_xy8',
           'convex_self_visible_proxy8_separate', 'human_semantic_gold8')
PACK_KEYS = ('query8_xy', 'L', 'query_center_valid', 'ROI_anyvalid',
             'crop_valid96', 'target_valid128', 'pure_continuous_query_valid',
             'nearest_pixel_valid')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def binding(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def verify(b):
    assert set(('path', 'sha256')) <= set(b), b
    assert sha(b['path']) == b['sha256'], b['path']
    return Path(b['path'])


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, data):
    with Path(path).open('x') as f:
        f.write(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def stable_bce(logit, label):
    z = np.asarray(logit, np.float64)
    y = np.asarray(label, np.float64)
    return np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))


def safe_ratio(n, d):
    return float(n / d) if d else None


def independent_admission(y, query, side, reference, qvalid, roi_any, local_any, target_any):
    with np.errstate(invalid='ignore', divide='ignore'):
        offset = (reference - query) / side[:, None, None]
    square = np.isfinite(offset).all(-1) & (offset >= -.5).all(-1) & (offset < .5).all(-1)
    available = roi_any & local_any & target_any[:, None]
    known = (y == 0) | (y == 1)
    return known & qvalid & available & square, square, offset, available


def independent_qvalid(query, raw_hw):
    q = np.asarray(query, np.float64)
    h = np.asarray(raw_hw)[:, 0, None]
    w = np.asarray(raw_hw)[:, 1, None]
    finite = np.isfinite(q).all(-1)
    not_missing = ~np.all(q == -1, axis=-1)
    continuous = finite & not_missing & (q[..., 0] >= 0) & (q[..., 0] < w) & (q[..., 1] >= 0) & (q[..., 1] < h)
    pix = np.floor(np.where(np.isfinite(q), q, -1e12) + .5)
    nearest = finite & not_missing & (pix[..., 0] >= 0) & (pix[..., 0] < w) & (pix[..., 1] >= 0) & (pix[..., 1] < h)
    return continuous & nearest, continuous, nearest


def decisions(logit, factual):
    # Frozen p>=.5 is represented by the root's exact logit>=0 decision.
    # Never use rounded sigmoid outputs for class decisions near zero.
    return np.where(factual, np.asarray(logit) >= 0, 2).astype(np.int8)


def metrics(y, pred, selected):
    truth = np.asarray(y)[selected]
    prediction = np.asarray(pred)[selected]
    assert np.isin(truth, (0, 1)).all()
    assert np.isin(prediction, (0, 1, 2)).all()
    cm = np.zeros((2, 3), np.int64)
    # Independent explicit enumeration; no trainer metric helper.
    for row in (0, 1):
        for col in (0, 1, 2):
            cm[row, col] = np.count_nonzero((truth == row) & (prediction == col))
    return dict(N=int(len(truth)),
                confusion_rows_GT_CLEAR_EXTERNAL_cols_pred_CLEAR_EXTERNAL_U=cm.tolist(),
                clear_precision=safe_ratio(cm[0, 0], cm[:, 0].sum()),
                clear_recall=safe_ratio(cm[0, 0], cm[0].sum()),
                external_precision=safe_ratio(cm[1, 1], cm[:, 1].sum()),
                external_recall=safe_ratio(cm[1, 1], cm[1].sum()),
                factual_U_rate=safe_ratio(cm[:, 2].sum(), cm.sum()))


def class_nll(loss, y, mask):
    result = {}
    for c in (0, 1):
        v = np.asarray(loss)[mask & (y == c)].astype(np.float64)
        result[str(c)] = dict(N=int(len(v)), mean=float(v.mean()) if len(v) else None)
    means = [result[str(c)]['mean'] for c in (0, 1)]
    return dict(classes=result, macro_equal_class_mean=float(np.mean(means)) if all(v is not None for v in means) else None)


def report(y, pred, split, admitted, square, self_visible):
    result = {}
    for partition in ('train', 'val'):
        base = np.repeat(split == partition, 8)
        known = base & np.isin(y, (0, 1))
        result[partition] = dict(
            all_slots=int(base.sum()), teacher_known=int(known.sum()),
            teacher_U=int(np.count_nonzero(base & (y == -1))),
            admitted_loss=int(np.count_nonzero(base & admitted)),
            known_reference_outside_qcrop=int(np.count_nonzero(known & ~square)),
            all_known_fixed_gate_metrics=metrics(y, pred, known),
            conditional_admitted_metrics=metrics(y, pred, base & admitted),
            all_known_by_convex_self={str(c): metrics(y, pred, known & (self_visible == c)) for c in (0, 1)})
    return result


def fixed_gates(m):
    return dict(external_recall_ge_0p8=m['external_recall'] is not None and m['external_recall'] >= .8,
                clear_recall_ge_0p8=m['clear_recall'] is not None and m['clear_recall'] >= .8,
                external_precision_ge_0p5=m['external_precision'] is not None and m['external_precision'] >= .5,
                factual_U_rate_le_0p1=m['factual_U_rate'] is not None and m['factual_U_rate'] <= .1)


def count_values(values):
    u, n = np.unique(values, return_counts=True)
    return {str(k): int(v) for k, v in zip(u.tolist(), n.tolist())}


def numerical_summary(a, b, mask=None):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    if mask is not None:
        a, b = a[mask], b[mask]
    d = a - b
    return dict(N=int(d.size), exact_equal=int(np.count_nonzero(d == 0)),
                max_abs=float(np.abs(d).max()) if d.size else None,
                mean_signed=float(d.mean()) if d.size else None,
                mean_abs=float(np.abs(d).mean()) if d.size else None,
                abs_gt_1e_6=int(np.count_nonzero(np.abs(d) > NUMERICAL_DISCUSSION_ATOL)),
                within_discussion_1e_6=bool(np.all(np.abs(d) <= NUMERICAL_DISCUSSION_ATOL)))


def assert_tree_equal(actual, expected, path='root'):
    assert type(actual) is type(expected), (path, type(actual), type(expected))
    if isinstance(actual, dict):
        assert set(actual) == set(expected), (path, set(actual) ^ set(expected))
        for k in actual:
            assert_tree_equal(actual[k], expected[k], path + '.' + k)
    elif isinstance(actual, list):
        assert len(actual) == len(expected), path
        for i, (a, b) in enumerate(zip(actual, expected)):
            assert_tree_equal(a, b, path + f'[{i}]')
    else:
        assert actual == expected, (path, actual, expected)


def verify_history(history):
    assert len(history) == 20 and [r['update'] for r in history] == list(range(100, 2001, 100))
    best = float('inf')
    best_update = None
    maximum_macro_discrepancy = 0.
    for row in history:
        assert set(row['per_class_BCE']) == {'0', '1'}
        pc = [float(row['per_class_BCE'][str(c)]) for c in (0, 1)]
        assert np.isfinite(pc).all() and min(pc) >= 0
        expected_macro = float(np.mean(pc))
        maximum_macro_discrepancy = max(maximum_macro_discrepancy, abs(expected_macro - row['heldout_macroBCE']))
        assert expected_macro == row['heldout_macroBCE']
        assert np.isfinite(row['train_loss']) and np.isfinite(row['seconds'])
        selected = row['heldout_macroBCE'] < best
        if selected:
            best = row['heldout_macroBCE']
            best_update = row['update']
        assert row['saved'] is selected and row['best_update'] == best_update
    return dict(rows=20, eligible_updates=list(range(100, 2001, 100)),
                strict_earliest_selected_update=best_update,
                selected_history_macroBCE=best, per_class_mean_max_difference=maximum_macro_discrepancy,
                selection_replayed_from_saved_history_only=True,
                limitation='Per-step prediction arrays were not saved; history loss values cannot be independently re-forwarded in this saved-only audit.')


def code_separation_audit(protocol, trainer_path):
    tree = ast.parse(Path(trainer_path).read_text())
    defs = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    fwd = defs['forward']
    model_calls = [n for n in ast.walk(fwd) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'model']
    assert len(model_calls) == 1 and len(model_calls[0].keywords) == 1 and model_calls[0].keywords[0].arg is None
    assert ast.unparse(model_calls[0].keywords[0].value).startswith('{k: v[')
    source = ast.get_source_segment(Path(trainer_path).read_text(), fwd)
    assert 'gpu.items()' in source and 'labels' not in source and 'reference' not in source
    load = defs['load_source']
    assigns = [n for n in ast.walk(load) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'gpu' for t in n.targets)]
    assert len(assigns) == 1 and isinstance(assigns[0].value, ast.Call)
    assert tuple(k.arg for k in assigns[0].value.keywords) == FORWARD_FIELDS
    model_tree = ast.parse(Path(protocol['bindings']['model_code']['path']).read_text())
    model_forward = next(n for n in ast.walk(model_tree) if isinstance(n, ast.FunctionDef) and n.name == 'forward')
    assert tuple(x.arg for x in model_forward.args.args[1:]) == FORWARD_FIELDS
    return dict(PASS=True, trainer_AST_forward_keyword_fields=list(FORWARD_FIELDS),
                forward_index='slot indices for local inputs; integer slot_index//8 for target crop',
                GT_reference_depth_owner_label_forward=False,
                scope='Static source code, not tensor dataflow instrumentation or a new model forward.')


def generated_tests(output):
    # Six exact half-open/validity/unknown cases.
    y = np.array([[0, 1, -1, 0, 1, 1, 0, 1]])
    q = np.zeros((1, 8, 2)); r = q.copy(); r[0, 1] = [-.5, 0]; r[0, 3] = [.5, 0]
    r[0, 4] = [0, np.nan]
    qv = np.ones((1, 8), bool); qv[0, 5] = False
    roi = np.ones_like(qv); roi[0, 6] = False
    local = np.ones_like(qv); local[0, 7] = False
    admitted, square, _, avail = independent_admission(y, q, np.ones(1), r, qv, roi, local, np.ones(1, bool))
    assert admitted.tolist() == [[True, True, False, False, False, False, False, False]]
    assert square[0, 1] and not square[0, 3]
    z = np.array([-2, 2, 0, -2, 2, 2, -2, 2], np.float32)
    pred = decisions(z, (qv & avail).ravel())
    known = np.isin(y.ravel(), (0, 1))
    m = metrics(y.ravel(), pred, known)
    assert m['confusion_rows_GT_CLEAR_EXTERNAL_cols_pred_CLEAR_EXTERNAL_U'] == [[2, 0, 1], [0, 2, 2]]
    # A GT-reference miss does not cause prediction U or disappear from all-known.
    assert pred[3] == 0 and not admitted.ravel()[3]
    assert metrics(y.ravel(), pred, admitted.ravel())['N'] == 2 and m['N'] == 7
    # Unknown-clamped diagnostics must not alter known class losses.
    loss = stable_bce(z, np.clip(y.ravel(), 0, 1))
    alt = loss.copy(); alt[y.ravel() == -1] = 1e300
    assert class_nll(loss, y.ravel(), known) == class_nll(alt, y.ravel(), known)
    # Stable extreme logits and sign ties/subnormal decisions.
    assert np.isfinite(stable_bce(np.array([-1e4, 1e4]), np.array([0, 1]))).all()
    tiny = np.nextafter(np.float32(0), np.float32(1))
    assert decisions(np.array([-tiny, -0., 0., tiny], np.float32), np.ones(4, bool)).tolist() == [0, 1, 1, 1]
    assert decisions(np.array([1.]), np.array([False])).tolist() == [2]
    # Coordinate formula: continuous-in-frame can still fail nearest raster bound.
    cq = np.array([[[0, 0], [9.49, 9.49], [9.5, 0], [-1, -1], [np.nan, 1], [.2, -.01], [1, 9.99], [2, 2]]])
    qc, continuous, nearest = independent_qvalid(cq, np.array([[10, 10]]))
    assert qc.tolist() == [[True, True, False, False, False, False, False, True]]
    assert continuous[0, 2] and not nearest[0, 2]
    # History earliest strict minimum includes ties, never update0.
    history = []
    best = float('inf'); best_update = None
    for i, update in enumerate(range(100, 2001, 100)):
        value = .8 if i < 2 else .4 if i < 5 else .6
        saved = value < best
        if saved: best, best_update = value, update
        history.append(dict(update=update, per_class_BCE={'0': value, '1': value}, heldout_macroBCE=value,
                            train_loss=value, seconds=i, saved=saved, best_update=best_update))
    assert verify_history(history)['strict_earliest_selected_update'] == 300
    rejected = 0
    for mutate in ('saved', 'best_update', 'macro', 'update'):
        bad = json.loads(json.dumps(history))
        if mutate == 'saved': bad[3]['saved'] = True
        elif mutate == 'best_update': bad[-1]['best_update'] = 400
        elif mutate == 'macro': bad[0]['heldout_macroBCE'] += .01
        else: bad[0]['update'] = 0
        try: verify_history(bad)
        except AssertionError: rejected += 1
    assert rejected == 4
    # Empty known cohorts have explicit null ratios, not artificial perfect U.
    empty = metrics(np.array([-1]), np.array([2]), np.array([False]))
    assert empty['N'] == 0 and empty['external_recall'] is None
    result = dict(PASS=True, reader=binding(__file__), scope=binding(SCOPE),
                  independent_NumPy_only=True, no_trainer_import=True,
                  generated_half_open_admission=True, factualU_not_GTmiss=True,
                  CM_exact=True, unknown_NLL_exclusion=True, stable_extreme_BCE=True,
                  zero_and_subnormal_logit_sign=True, continuous_nearest_bounds=True,
                  history20_earliest_strict_min=True, malformed_history_rejections=4,
                  actual_GT_output_feature_RGB_weight_reads=0, new_forward_fit_current_PnP=0)
    write_new(output, result)
    print(json.dumps(result, ensure_ascii=False))


def load_public_flags(protocol):
    source = read(verify(protocol['bindings']['source_input_bindings']))
    pack = next(x for x in source['packs'] if x['source'] == 'YCB_PUBLIC_REAL')
    for k in ('manifest', 'seal', 'lineage'): verify(pack[k])
    manifest = read(pack['manifest']['path'])
    assert manifest['completed'] is True and manifest['frame_count'] == 4200 and manifest['query_count'] == 33600
    lines = [json.loads(x) for x in Path(pack['lineage']['path']).read_text().splitlines()]
    query_manifest = manifest['input_query_manifest']; verify(query_manifest)
    query_lines = [json.loads(x) for x in Path(query_manifest['path']).read_text().splitlines()]
    assert len(lines) == len(query_lines) == 4200
    fields = {k: [] for k in ('query8_xy', 'L', 'query_center_valid', 'ROI_anyvalid',
                              'pure_continuous_query_valid', 'nearest_pixel_valid', 'local_any', 'target_any')}
    chunk_bindings = []
    next_start = 0
    for chunk in manifest['chunks']:
        assert chunk['start'] == next_start and chunk['end'] > chunk['start']
        next_start = chunk['end']; verify(chunk['file']); chunk_bindings.append(chunk['file'])
        with np.load(chunk['file']['path'], allow_pickle=False) as z:
            assert set(PACK_KEYS) <= set(z.files)
            for k in PACK_KEYS:
                # Deliberately never access p3_roi/rgb96/target_rgb128/context42.
                if k == 'crop_valid96':
                    v = z[k]; assert v.dtype == bool
                    fields['local_any'].append(v.reshape(len(v), 8, -1).any(-1))
                elif k == 'target_valid128':
                    v = z[k]; assert v.dtype == bool
                    fields['target_any'].append(v.reshape(len(v), -1).any(-1))
                else: fields[k].append(z[k])
    assert next_start == 4200
    data = {k: np.concatenate(v) for k, v in fields.items()}
    assert data['query8_xy'].shape == (4200, 8, 2) and data['L'].shape == (4200,)
    assert np.isfinite(data['L']).all() and (data['L'] > 0).all()
    hw = np.asarray([q['raw_hw'] for q in query_lines], np.int64)
    qvalid, continuous, nearest = independent_qvalid(data['query8_xy'], hw)
    assert np.array_equal(qvalid, data['query_center_valid'])
    assert np.array_equal(continuous, data['pure_continuous_query_valid'])
    assert np.array_equal(nearest, data['nearest_pixel_valid'])
    for i, (line, q) in enumerate(zip(lines, query_lines)):
        assert line['id'] == q['id'] and line['split'] == q['split']
        assert np.array_equal(data['query8_xy'][i], np.asarray(q['initial_points'], np.float64), equal_nan=True)
        assert data['L'][i] == q['roi_side']
    return data, lines, dict(pack=pack, query_manifest=query_manifest, chunks=chunk_bindings,
                            accessed_NPZ_keys=list(PACK_KEYS),
                            unaccessed_forbidden_arrays=['p3_roi', 'rgb96', 'target_rgb128', 'context42'],
                            feature_arrays_RGB_arrays_loaded=0,
                            note='SHA checks read compressed file bytes; only explicitly listed coordinate/factual-valid arrays are deserialized.')


def actual_replay(completed, completed_sha, preparation_lock_sha, output):
    # Fail closed before any source GT/scores/pack arrays are accessed.
    lock_path = HERE / 'PREPARATION_SEAL.json'
    assert sha(lock_path) == preparation_lock_sha
    lock = read(lock_path); assert lock['completed'] is True and lock['PASS'] is True
    for k in ('scope', 'reader', 'generated'): verify(lock[k])
    assert lock['reader'] == binding(__file__)
    scope = read(SCOPE)
    for k in ('protocol', 'trainer', 'model'): verify(scope[k])
    completed = Path(completed)
    assert sha(completed) == completed_sha
    done = read(completed)
    assert done['completed'] is True and done['optimizer_updates'] == 2000 and done['current_PnP'] == 0
    run = completed.parent
    assert run == ROOT / 'runs/YCB_EXTERNAL_RAY_V1'
    for k in ('model_freeze', 'history', 'scores', 'source_results'): verify(done[k])
    protocol = read(scope['protocol']['path'])
    assert protocol['forward_fields'] == list(FORWARD_FIELDS)
    assert protocol['current223_underlying_reads_allowed'] is False and protocol['PnP_allowed'] is False
    assert protocol['selection']['threshold'] == .5 and protocol['optimizer']['updates'] == 2000
    for name, b in protocol['bindings'].items(): verify(b)
    entry_path = ROOT / 'TRAINER_ENTRYLOCK.json'; entry = read(entry_path)
    assert entry['trainer'] == scope['trainer'] and entry['protocol'] == scope['protocol']
    assert entry['restart_allowed_after_first_update'] is False and Path(entry['run_directory']) == run
    verify(entry['generated']); verify(entry['independent_review'])
    freeze = read(done['model_freeze']['path']); source_result = read(done['source_results']['path'])
    assert freeze['completed'] is True and freeze['optimizer_updates'] == 2000
    assert freeze['protocol'] == scope['protocol'] and freeze['trainer'] == scope['trainer']
    assert freeze['model_code'] == scope['model'] and freeze['base_architecture'] == protocol['bindings']['base_architecture']
    assert freeze['external_probability_threshold'] == .5 and freeze['semantic_gold'] is False
    assert freeze['current_for_fit_selection'] is False and freeze['PnP'] == 0 and freeze['fully_solved'] is False
    verify(freeze['checkpoint'])  # SHA bytes only; no checkpoint deserialization.
    assert source_result['completed'] is True and source_result['model_freeze'] == done['model_freeze']
    assert source_result['current_evaluation_admission'] is False
    history = read(done['history']['path']); history_audit = verify_history(history)
    assert freeze['best_update'] == done['best_update'] == history_audit['strict_earliest_selected_update']
    assert freeze['selection_macroBCE'] == history_audit['selected_history_macroBCE']
    started = read(run / 'RUN_STARTED.json'); first = read(run / 'FIRST_UPDATE.json')
    assert started['protocol'] == scope['protocol'] and started['entry'] == binding(entry_path)
    assert started['optimizer_updates'] == 0 and first['update'] == 1 and first['restart_allowed'] is False
    assert started['current_PnP'] == first['current_PnP'] == 0
    preflight = read(run / 'SOURCE_GPU_PREFLIGHT.json'); assert preflight['PASS'] is True and preflight['optimizer_updates'] == 0
    code_audit = code_separation_audit(protocol, scope['trainer']['path'])
    # Only now access completed saved source outputs/GT and original input flags.
    with np.load(done['scores']['path'], allow_pickle=False) as z:
        assert set(z.files) == SCORE_KEYS
        score = {k: z[k] for k in SCORE_KEYS}
    with np.load(protocol['bindings']['teacher_GT']['path'], allow_pickle=False) as z:
        gt = {k: z[k] for k in GT_KEYS}
    data, lineage, pack_bindings = load_public_flags(protocol)
    ids = np.asarray([x['id'] for x in lineage]); split = np.asarray([x['split'] for x in lineage])
    assert len(ids) == 4200 and len(np.unique(ids)) == 4200
    assert count_values(split) == {'train': 3400, 'val': 800}
    for field in ('id', 'split'):
        assert np.array_equal(score[field], gt[field])
        assert np.array_equal(score[field], ids if field == 'id' else split)
    for field in SCORE_KEYS - {'id', 'split'}: assert score[field].shape == (4200, 8)
    assert score['external_logit'].dtype == np.float32 and score['NLL_FP32'].dtype == np.float32
    for k in ('input_available', 'query_center_valid', 'loss_admitted_GT_only', 'reference_inside_qcrop_GT_only'):
        assert score[k].dtype == bool
    assert gt['label8'].shape == (4200, 8) and np.isin(gt['label8'], (-1, 0, 1)).all()
    assert np.all(gt['human_semantic_gold8'] == -1)
    assert np.array_equal(gt['label8'] == 1, gt['reason8'] == 'EXTERNAL_OTHER_FIRST_HIT_PROXY')
    assert np.array_equal(gt['label8'] == 0, gt['reason8'] == 'CLEAR_TO_VIRTUAL_ENDPOINT_PROXY')
    assert np.isfinite(score['external_logit']).all() and np.isfinite(score['NLL_FP32']).all()
    assert (score['NLL_FP32'] >= 0).all()
    admitted, square, offset, available = independent_admission(
        gt['label8'], data['query8_xy'], data['L'], gt['reference_xy8'],
        data['query_center_valid'], data['ROI_anyvalid'], data['local_any'], data['target_any'])
    assert np.array_equal(admitted, score['loss_admitted_GT_only'])
    assert np.array_equal(square, score['reference_inside_qcrop_GT_only'])
    assert np.array_equal(available, score['input_available'])
    assert np.array_equal(data['query_center_valid'], score['query_center_valid'])
    y, logits = gt['label8'].ravel(), score['external_logit'].ravel()
    factual = (data['query_center_valid'] & available).ravel()
    predicted = decisions(logits, factual)
    independent = report(y, predicted, split, admitted.ravel(), square.ravel(), gt['convex_self_visible_proxy8_separate'].ravel())
    assert_tree_equal(independent, source_result['source_metrics'], 'source_metrics')
    gates = fixed_gates(independent['val']['all_known_fixed_gate_metrics'])
    assert_tree_equal(gates, source_result['predeclared_source_gate'], 'gates')
    assert source_result['all_four_pass'] is all(gates.values())
    pools = {str(c): {'train': int(np.count_nonzero(admitted & (split == 'train')[:, None] & (gt['label8'] == c))),
                     'val': int(np.count_nonzero(admitted & (split == 'val')[:, None] & (gt['label8'] == c)))} for c in (0, 1)}
    assert_tree_equal(pools, source_result['pool_counts'], 'pool_counts')
    assert_tree_equal(pools, preflight['pool_counts'], 'preflight_pools')
    known = np.isin(y, (0, 1)); val_admit = admitted.ravel() & np.repeat(split == 'val', 8)
    analytic = stable_bce(logits, np.clip(y, 0, 1))
    stored = score['NLL_FP32'].ravel()
    nll = {part: class_nll(analytic, y, admitted.ravel() & np.repeat(split == part, 8)) for part in ('train', 'val')}
    stored_nll = {part: class_nll(stored, y, admitted.ravel() & np.repeat(split == part, 8)) for part in ('train', 'val')}
    selected_history_row = next(r for r in history if r['update'] == freeze['best_update'])
    history_vs_final = {
        'history_selected_validation_only_pass': {'macro': selected_history_row['heldout_macroBCE'], 'per_class': selected_history_row['per_class_BCE']},
        'final_all_slot_pass_stored_FP32_NLL': stored_nll['val'],
        'final_all_slot_pass_analytic_FP64_NLL': nll['val'],
        'stored_final_minus_history_macro': stored_nll['val']['macro_equal_class_mean'] - selected_history_row['heldout_macroBCE'],
        'analytic_final_minus_history_macro': nll['val']['macro_equal_class_mean'] - selected_history_row['heldout_macroBCE'],
        'per_class_final_stored_minus_history': {str(c): stored_nll['val']['classes'][str(c)]['mean'] - selected_history_row['per_class_BCE'][str(c)] for c in (0, 1)},
        'numerical_discussion_atol': NUMERICAL_DISCUSSION_ATOL,
        'selection_kept': 'Original frozen history validation-only pass; final-pass values are audit diagnostics only.',
        'no_saved_per_query_history_logits': True,
        'cannot_verify_history_pass_class_decision_flips_without_history_logits': True}
    numeric = dict(stored_FP32_minus_analytic_FP64_known=numerical_summary(stored, analytic, known),
                   stored_FP32_minus_analytic_FP64_admitted_VAL=numerical_summary(stored, analytic, val_admit),
                   history_vs_final=history_vs_final)
    extra = {}
    for part in ('train', 'val'):
        base = np.repeat(split == part, 8)
        pk = base & known
        extra[part] = dict(
            teacher_reasons=count_values(gt['reason8'].ravel()[base]),
            teacher_label_counts=count_values(y[base]), factual_prediction_counts=count_values(predicted[base]),
            teacher_unknown_prediction_distribution=count_values(predicted[base & ~known]),
            reference_miss_only=metrics(y, predicted, pk & ~square.ravel()),
            known_inside_ref_square=metrics(y, predicted, pk & square.ravel()),
            teacher_known_by_self={str(c): int(np.count_nonzero(pk & (gt['convex_self_visible_proxy8_separate'].ravel() == c))) for c in (-1, 0, 1)},
            exclusion_counts_nonexclusive=dict(teacher_U=int(np.count_nonzero(base & ~known)),
                query_invalid=int(np.count_nonzero(base & ~data['query_center_valid'].ravel())),
                input_unavailable=int(np.count_nonzero(base & ~available.ravel())),
                reference_outside_or_nonfinite=int(np.count_nonzero(base & ~square.ravel()))))
    output = Path(output); output.mkdir(exist_ok=True)
    per_query = output / 'INDEPENDENT_ALL33600_REPLAY.csv'
    with per_query.open('x', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame_index', 'id', 'split', 'native_k', 'teacher', 'reason', 'logit', 'prediction_0clear_1external_2U',
                         'qvalid', 'input_available', 'ref_inside_halfopen', 'loss_admitted', 'convex_self_visible_separate',
                         'known_NLL_analytic_FP64', 'stored_FP32_NLL_DIAGNOSTIC_unknown_not_label', 'stored_minus_analytic_if_known'])
        for i in range(33600):
            frame, k = divmod(i, 8)
            writer.writerow([frame, ids[frame], split[frame], k, int(y[i]), gt['reason8'].ravel()[i], float(logits[i]), int(predicted[i]),
                bool(data['query_center_valid'].ravel()[i]), bool(available.ravel()[i]), bool(square.ravel()[i]), bool(admitted.ravel()[i]),
                int(gt['convex_self_visible_proxy8_separate'].ravel()[i]), float(analytic[i]) if known[i] else '',
                float(stored[i]), float(stored[i] - analytic[i]) if known[i] else ''])
    artifacts = dict(completed_seal=binding(completed), preparation= binding(lock_path), scope=binding(SCOPE), reader=binding(__file__),
                     protocol=scope['protocol'], trainer=scope['trainer'], model=scope['model'], checkpoint_SHA_only=freeze['checkpoint'],
                     teacher_GT=protocol['bindings']['teacher_GT'], selected_scores=done['scores'], history=done['history'],
                     model_freeze=done['model_freeze'], published_results=done['source_results'], input_pack=pack_bindings,
                     run_started=binding(run / 'RUN_STARTED.json'), first_update=binding(run / 'FIRST_UPDATE.json'),
                     preflight=binding(run / 'SOURCE_GPU_PREFLIGHT.json'), all_query_replay=binding(per_query))
    audit = dict(completed=True, PASS=True, blockers=[], created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 bindings=artifacts, exact_reconciliation=dict(all4200_identity_order=True, all33600_native_slots=True,
                    GT_reason_codes=True, pack_query_formula=True, qvalid_nearest_continuous=True,
                    factual_available=True, halfopen_admission=True, published_CM_precision_recall_U=True,
                    all_known_misses_preserved=True, conditional_admitted=True, source_pools=True, all_four_frozen_gates=True),
                 source_metrics=independent, source_gate=gates, all_four_pass=all(gates.values()),
                 independent_known_only_NLL_FP64=nll, saved_known_only_NLL_FP32=stored_nll,
                 numerical_reconciliation=numeric, history_selection=history_audit,
                 additional_cohorts_and_exclusions=extra, GT_forward_static_audit=code_audit,
                 execution_integrity=dict(optimizer_updates=2000, restart_allowed_after_first_update=False,
                    single_expected_run_directory=str(run), run_started_and_first_update_bound=True,
                    completed_after_first_update=True,
                    limitation='No hidden-process/restart proof beyond the sealed code and provided execution records; no weight tensor was loaded.'),
                 source_feature_RGB_weight_parameter_loads=0, new_model_forward_fit_current_PnP=0,
                 current_evaluation_admission=False, semantic_gold=False,
                 scientific_limits=scope['external_limits'])
    audit_path = output / 'INDEPENDENT_POSTFIT_AUDIT.json'; write_new(audit_path, audit)
    seal_path = output / 'AUDIT_SEAL.json'; write_new(seal_path, dict(completed=True, PASS=True, audit=binding(audit_path), all_query_replay=binding(per_query), current_admission=False))
    print(json.dumps(dict(audit=binding(audit_path), seal=binding(seal_path), PASS=True, source_gate=gates), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='stage', required=True)
    g = sub.add_parser('generated'); g.add_argument('--output', type=Path, required=True)
    a = sub.add_parser('actual'); a.add_argument('--fit-complete', type=Path, required=True)
    a.add_argument('--fit-complete-sha', required=True); a.add_argument('--preparation-lock-sha', required=True)
    a.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.stage == 'generated': generated_tests(args.output)
    else: actual_replay(args.fit_complete, args.fit_complete_sha, args.preparation_lock_sha, args.output)
