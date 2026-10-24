"""Independent, NumPy-only replay of frozen SOURCE outputs. No models or RGB.

All decisions/strata are registered source rules. This program reports metrics;
it never selects a checkpoint, radius, threshold or private candidate.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from source_support_contract import (CLASS_NAMES, JOINT_STATES, confident_joint_map,
                                     hierarchical_states, joint_codes, joint_marginals)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PAIRS = np.asarray(JOINT_STATES)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def rows(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def safe_ratio(a, b):
    return float(a / b) if b else None


def csv_bool(value):
    if str(value).lower() in {'true', '1'}:
        return True
    if str(value).lower() in {'false', '0'}:
        return False
    raise ValueError('Unregistered CSV Boolean: '+str(value))


def softmax(z):
    z = np.asarray(z, float)
    if z.ndim != 2 or z.shape[1] != 6 or not np.isfinite(z).all():
        raise ValueError('Finite N,6 saved logits required')
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)


def confusion(y, predicted, nclass):
    """Columns 0..nclass-1 plus final U column, all ground truth retained."""
    y, predicted = np.asarray(y, int), np.asarray(predicted, int)
    if ((y < 0) | (y >= nclass)).any() or ((predicted < -1) | (predicted >= nclass)).any():
        raise ValueError('Invalid confusion labels')
    c = np.zeros((nclass, nclass + 1), np.int64)
    np.add.at(c, (y, np.where(predicted < 0, nclass, predicted)), 1)
    return c


def state_metrics(y, predicted, nclass=3):
    c = confusion(y, predicted, nclass)
    support, called, decided = c.sum(1), c[:, :nclass].sum(0), c[:, :nclass].sum(1)
    by_class = []
    for k in range(nclass):
        by_class.append(dict(class_index=k, support=int(support[k]), predicted=int(called[k]),
                             correct=int(c[k, k]), withheld=int(c[k, -1]),
                             precision=safe_ratio(c[k, k], called[k]),
                             recall_all_GT=safe_ratio(c[k, k], support[k]),
                             recall_decided_GT=safe_ratio(c[k, k], decided[k]),
                             coverage_given_GT=safe_ratio(decided[k], support[k])))
    recalls = [v['recall_all_GT'] for v in by_class if v['support']]
    return dict(n=len(y), confusion_truth_rows_prediction_columns_U_last=c.tolist(),
                coverage=safe_ratio(c[:, :nclass].sum(), len(y)),
                withheld=int(c[:, -1].sum()),
                accuracy_all_GT=safe_ratio(np.trace(c[:, :nclass]), len(y)),
                balanced_recall_all_GT=float(np.mean(recalls)) if recalls else None,
                per_class=by_class)


def binary_metrics(truth, probability):
    truth = np.asarray(truth, bool)
    p = np.asarray(probability, float)
    pred = p >= .5
    tp = int((truth & pred).sum()); fp = int((~truth & pred).sum())
    fn = int((truth & ~pred).sum()); tn = int((~truth & ~pred).sum())
    pp = np.clip(p, 1e-12, 1-1e-12)
    return dict(n=len(p), truth_positive=int(truth.sum()), tp=tp, fp=fp, fn=fn, tn=tn,
                precision=safe_ratio(tp, tp+fp), recall=safe_ratio(tp, tp+fn),
                BCE=float(-(truth*np.log(pp)+(~truth)*np.log(1-pp)).mean()) if len(p) else None,
                threshold=.5, comparison='>=')


def probability_metrics(y, p):
    if not len(y):
        return dict(joint_NLL=None, joint_Brier=None)
    return dict(joint_NLL=float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()),
                joint_Brier=float(np.square(p-np.eye(6)[y]).sum(1).mean()))


def summarize(y, p):
    statep = joint_marginals(p)
    logical = hierarchical_states(p)
    confident = confident_joint_map(p)
    best = p.argmax(1)
    result = dict(**probability_metrics(y, p),
                  joint_MAP=state_metrics(y, best, 6),
                  confident_joint_MAP_U=state_metrics(y, confident['joint'], 6),
                  scales={})
    truth = PAIRS[y]
    for scale, name in enumerate(['exact', 'radius2']):
        full = 1-statep[:, scale, 2]
        result['scales'][name] = dict(
            logical_support_diagnostic=state_metrics(truth[:, scale], logical[:, scale]),
            joint_MAP_diagnostic=state_metrics(truth[:, scale], PAIRS[best, scale]),
            confident_joint_MAP_U=state_metrics(truth[:, scale], confident['classes'][:, scale]),
            primary_binary_visible=binary_metrics(truth[:, scale] == 0, statep[:, scale, 0]),
            binary_full_support=binary_metrics(truth[:, scale] != 2, full))
    return result


def load_cache(path):
    with np.load(path, allow_pickle=False) as z:
        keys = ['joint', 'classes', 'dense_valid', 'source', 'sample_id', 'group_id', 'image_sha256']
        if 'sam' in z.files:
            keys.append('sam')
        d = {k: z[k] for k in keys}
    if d['joint'].shape[1:] != (9, 9) or d['classes'].shape[1:] != (2, 9, 9):
        raise ValueError('Registered native9 targets required')
    joint = joint_codes(d['classes'][:, 0], d['classes'][:, 1], d['dense_valid'])
    if not np.array_equal(joint[d['dense_valid']], d['joint'][d['dense_valid']]):
        raise AssertionError('Class to joint cache labels disagree')
    if not set(d['source']) <= {'SYNTH', 'YCB'}:
        raise ValueError('Only registered source data permitted')
    if len(set(d['sample_id'])) != len(d['sample_id']):
        raise ValueError('Duplicate source sample ID')
    return d


def validate_outputs(path, cache):
    with np.load(path, allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    index = a['index'].astype(int)
    valid = np.flatnonzero(cache['dense_valid'][:, 4, 4])
    if not np.array_equal(index, valid):
        raise AssertionError('Outputs must cover every valid cache query in the same order')
    truth = cache['joint'][index, 4, 4].astype(int)
    p = softmax(a['joint_logits'])
    for key, expected in [('truth_joint', truth), ('truth_classes', PAIRS[truth]),
                          ('source', cache['source'][index])]:
        if not np.array_equal(a[key], expected):
            raise AssertionError('Frozen output mismatch: '+key)
    marginal = joint_marginals(p)
    expected = {'joint_probability': p, 'state_probability': marginal,
                'visible_probability': marginal[:, :, 0],
                'full_probability': marginal[:, :, :2].sum(2)}
    for key, value in expected.items():
        if key in a and not np.allclose(a[key], value, atol=1e-7, rtol=1e-6):
            raise AssertionError('Output probability mismatch: '+key)
    return index, truth, p


def source_queries(cache, indices, query_files):
    lookup = {}
    for file in query_files:
        for row in rows(file):
            if row['sample_id'] in lookup:
                raise ValueError('Duplicate query CSV sample')
            lookup[row['sample_id']] = row
    result = []
    for i in indices:
        row = lookup[str(cache['sample_id'][i])]
        if row['image_sha256'] != cache['image_sha256'][i] or row['source'] != cache['source'][i]:
            raise AssertionError('Query identity/SHA mismatch')
        if int(row['joint_state']) != int(cache['joint'][i, 4, 4]):
            raise AssertionError('Query CSV labels disagree')
        result.append(row)
    return result


def rough_box_sam(cache, indices, queries, input_manifest):
    """Frozen v63 raw visible support + disclosed prompt-box full heuristic.

    A box is never GT amodal support. OR visible support enforces V subset F.
    Membership is evaluated at the same nearest native integer query pixel.
    """
    inputs = {v['id']: v for v in map(json.loads, Path(input_manifest).read_text().splitlines())}
    score = cache['sam'][indices, :, 4, 4].astype(float)
    visible = score > 0  # raw SAM threshold is strictly >0, unlike head >=.5.
    full = np.zeros_like(visible)
    for j, row in enumerate(queries):
        item = inputs[row['id']]
        if item['image_sha256'] != row['image_sha256']:
            raise AssertionError('Input-only source box identity mismatch')
        q = np.array([float(row['query_x']), float(row['query_y'])])
        if not np.allclose(q, item['query'][int(row['query_index'])], atol=1e-7):
            raise AssertionError('Source query differs from frozen SAM input')
        x, y = int(row['pixel_x']), int(row['pixel_y'])
        x0, y0, x1, y1 = item['box']
        for scale, radius in enumerate([0, 2]):
            full[j, scale] = ((x0-radius <= x <= x1+radius) and (y0-radius <= y <= y1+radius))
    full |= visible
    classes = np.where(visible, 0, np.where(full, 1, 2))
    joint = joint_codes(classes[:, 0], classes[:, 1])
    return np.eye(6)[joint], score


def train_prior(train, val_source, laplace=0):
    valid = train['dense_valid'][:, 4, 4]
    y = train['joint'][:, 4, 4]
    probability = np.empty((len(val_source), 6))
    priors = {}
    for source in set(val_source):
        selected = valid & (train['source'] == source)
        # A SYNTH-only checkpoint transfer may request unseen YCB: global TRAIN
        # prior is then the only available training prior, explicitly recorded.
        fallback = not selected.any()
        if fallback:
            selected = valid
        counts = np.bincount(y[selected], minlength=6)
        p = (counts+laplace)/(counts.sum()+6*laplace)
        probability[val_source == source] = p
        priors[source] = dict(counts=counts.tolist(), probability=p.tolist(),
                              missing_domain_uses_global_TRAIN=fallback, laplace=laplace)
    return probability, priors


def evidence_strata(queries, evidence_file):
    evidence = {r['sample_id']: r for r in rows(evidence_file)}
    n = len(queries)
    strata = {'ALL': np.ones(n, bool)}
    fields = ['exact_external_corroborated', 'exact_external_ambiguous',
              'strict_external_corroborated_5x5', 'strict_external_ambiguous_5x5',
              'visible_boundary_r1', 'full_boundary_r1', 'visible_boundary_r2', 'full_boundary_r2',
              'exact_enclosed_full_void', 'radius2_enclosed_void_any', 'radius2_clipped_by_frame',
              'depth_missing_at_query', 'depth_file_available']
    for key in fields:
        strata['YCB_'+key] = np.zeros(n, bool)
    for state in ['EXTERNAL_SUPPORT_CORROBORATED_ANY', 'EXTERNAL_SUPPORT_AMBIGUOUS']:
        strata['YCB_radius2_'+state] = np.zeros(n, bool)
    strata['YCB_depth_valid_at_query'] = np.zeros(n, bool)
    strata['SYNTH_native_boundary'] = np.zeros(n, bool)
    for i, row in enumerate(queries):
        if row['source'] == 'SYNTH':
            strata['SYNTH_native_boundary'][i] = csv_bool(row['native_boundary'])
            continue
        e = evidence[row['sample_id']]
        if row['image_sha256'] != e['image_sha256']:
            raise AssertionError('Evidence image identity mismatch')
        if int(row['exact_state']) != (0 if int(e['exact_visible']) else 1 if int(e['exact_full']) else 2):
            raise AssertionError('Evidence exact source label disagrees')
        near = 0 if int(e['radius2_visible_support']) else 1 if int(e['radius2_full_support']) else 2
        if int(row['radius2_state']) != near:
            raise AssertionError('Evidence radius2 source label disagrees')
        for key in fields:
            strata['YCB_'+key][i] = bool(int(e[key]))
        for state in ['EXTERNAL_SUPPORT_CORROBORATED_ANY', 'EXTERNAL_SUPPORT_AMBIGUOUS']:
            strata['YCB_radius2_'+state][i] = e['radius2_evidence_state'] == state
        strata['YCB_depth_valid_at_query'][i] = bool(int(e['depth_file_available'])) and not int(e['depth_missing_at_query'])
    return strata


def flatten_report(model, domain, stratum, report, metric_rows, confusion_rows):
    for scale, record in report['scales'].items():
        for decoder in ['logical_support_diagnostic', 'joint_MAP_diagnostic', 'confident_joint_MAP_U']:
            stat = record[decoder]
            for cls in stat['per_class']:
                metric_rows.append(dict(model=model, domain=domain, stratum=stratum, scale=scale,
                                        decoder=decoder, class_name=CLASS_NAMES[cls['class_index']],
                                        n=stat['n'], coverage=stat['coverage'], **cls))
            cm = stat['confusion_truth_rows_prediction_columns_U_last']
            for truth in range(3):
                for pred in range(4):
                    confusion_rows.append(dict(model=model, domain=domain, stratum=stratum, scale=scale,
                                               decoder=decoder, truth=CLASS_NAMES[truth],
                                               prediction=CLASS_NAMES[pred] if pred < 3 else 'U', count=cm[truth][pred]))


def write_csv(path, records):
    if not records:
        return
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, list(records[0])); w.writeheader(); w.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-cache', required=True, type=Path)
    parser.add_argument('--val-cache', required=True, type=Path)
    parser.add_argument('--outputs', required=True, type=Path)
    parser.add_argument('--freeze', required=True, type=Path)
    parser.add_argument('--output-cache-binding', type=Path,
                        help='Required when saved output cache is not the fit selection VAL')
    parser.add_argument('--query-csv', type=Path, action='append')
    parser.add_argument('--inputs', type=Path, default=ROOT.parent/'query_visibility_public_real_20261022_v63/sam_source/INPUTS.jsonl')
    parser.add_argument('--evidence', type=Path, default=ROOT/'source_audit/YCB_VALIDATION_QUERY_EVIDENCE.csv')
    parser.add_argument('--prefix', required=True, type=Path)
    args = parser.parse_args()
    freeze = json.loads(args.freeze.read_text())
    if not freeze['completed'] or freeze['eval_data_used'] or freeze['PnP_used']:
        raise ValueError('Completed source-only freeze required')
    if sha(args.train_cache) != freeze['train_sha256']:
        raise AssertionError('TRAIN cache changed from frozen fit')
    val_sha = sha(args.val_cache)
    if val_sha != freeze['val_sha256'] and not args.output_cache_binding:
        raise ValueError('Cross-source output needs explicit frozen cache/output hash binding')
    if args.output_cache_binding:
        binding = json.loads(args.output_cache_binding.read_text())
        if binding['cache_sha256'] != val_sha or binding['outputs_sha256'] != sha(args.outputs):
            raise AssertionError('Cross-source output binding mismatch')
        if binding['checkpoint_sha256'] != freeze['checkpoint_sha256']:
            raise AssertionError('Cross-source checkpoint mismatch')
    train, val = load_cache(args.train_cache), load_cache(args.val_cache)
    denylist_path = Path(freeze['arguments']['eval_denylist'])
    if sha(denylist_path) != freeze['eval_denylist_sha256']:
        raise AssertionError('Frozen image exclusion manifest changed')
    excluded = set(json.loads(denylist_path.read_text())['image_sha256'])
    for split, cache in [('TRAIN', train), ('VALIDATION', val)]:
        if set(cache['image_sha256']) & excluded:
            raise AssertionError('Evaluation image SHA in source '+split)
    for field in ['sample_id', 'group_id', 'image_sha256']:
        if set(train[field]) & set(val[field]):
            raise AssertionError('TRAIN/VAL overlap: '+field)
    index, y, p = validate_outputs(args.outputs, val)
    source = val['source'][index]
    query_files = args.query_csv or [ROOT/'source_cache/SYNTH_VALIDATION_QUERIES.csv', ROOT/'source_cache/YCB_VALIDATION_QUERIES.csv']
    queries = source_queries(val, index, query_files)
    rough, raw_sam = rough_box_sam(val, index, queries, args.inputs)
    prior, prior_info = train_prior(train, source)
    smooth_prior, smooth_info = train_prior(train, source, laplace=1)
    models = {'FROZEN_HEAD': p, 'TRAIN_EMPIRICAL_JOINT_PRIOR': prior,
              'TRAIN_LAPLACE1_JOINT_PRIOR': smooth_prior,
              'UNIFORM_JOINT': np.full_like(p, 1/6),
              'ALWAYS_VISIBLE': np.tile(np.eye(6)[0], (len(y), 1)),
              'ALWAYS_EXTERNAL_SUPPORT': np.tile(np.eye(6)[2], (len(y), 1)),
              'ALWAYS_OUTSIDE': np.tile(np.eye(6)[5], (len(y), 1)),
              'FROZEN_RAW_SAM_PLUS_ROUGH_BOX_FULL': rough}
    strata = evidence_strata(queries, args.evidence)
    result = dict(schema='V64_INDEPENDENT_FROZEN_SOURCE_EVALUATION_V1',
                  scope='Source validation only; no RGB/model/private inference or checkpoint selection',
                  hashes={str(path): sha(path) for path in [Path(__file__), HERE/'source_support_contract.py',
                                                           args.train_cache, args.val_cache, args.outputs, args.freeze,
                                                           args.evidence, args.inputs, denylist_path, *query_files]},
                  n=len(y), sources={s: int((source == s).sum()) for s in sorted(set(source))},
                  exclusions=dict(eval_image_SHA_count=len(excluded), train_overlap=0, validation_overlap=0,
                                  eval_image_RGB_or_labels_opened=False),
                  thresholds=dict(binary_visible_and_full='>=.5', mechanism='joint MAP max>=.5 else U', raw_SAM='logit strictly >0'),
                  caveats=['E means source full-minus-visible support, never trained semantic corner self.',
                           'SAM full is prompt-box membership OR visible, not GT amodal support.',
                           'Radius2 is separately supervised fixed local support, not exact pixel visibility.',
                           'Other-visible overlap corroborates source external support; depth-only values do not establish occlusion.',
                           'Enclosed full-mask void is a 2D topology attribute, not a verified physical hole.',
                           'Deterministic baselines have one-hot probabilities; clipped NLL is diagnostic only.'],
                  train_priors=dict(empirical=prior_info, laplace1=smooth_info),
                  strata_denominators={k: int(v.sum()) for k, v in strata.items()}, results={}, evidence_results={})
    metric_rows, confusion_rows, evidence_rows = [], [], []
    for name, prob in models.items():
        result['results'][name] = {}
        result['evidence_results'][name] = {}
        for domain in ['ALL', *sorted(set(source))]:
            select = np.ones(len(y), bool) if domain == 'ALL' else source == domain
            report = summarize(y[select], prob[select])
            result['results'][name][domain] = report
            flatten_report(name, domain, 'ALL', report, metric_rows, confusion_rows)
        for label, mask in strata.items():
            if label == 'ALL' or not mask.any():
                continue
            report = summarize(y[mask], prob[mask])
            result['evidence_results'][name][label] = report
            flatten_report(name, 'SOURCE_STRATUM', label, report, evidence_rows, confusion_rows)
    result['macro_source_joint_NLL'] = {
        model: float(np.mean([result['results'][model][domain]['joint_NLL']
                              for domain in sorted(set(source))]))
        for model in models}
    # Source-only hold diagnostic does not change source-NLL checkpoint selection.
    holds = []
    for domain in sorted(set(source)):
        head, base = result['results']['FROZEN_HEAD'][domain], result['results']['TRAIN_EMPIRICAL_JOINT_PRIOR'][domain]
        if head['joint_NLL'] >= base['joint_NLL']:
            holds.append(dict(domain=domain, reason='joint_NLL does not beat TRAIN empirical prior'))
        for scale, value in head['scales'].items():
            for cls in value['confident_joint_MAP_U']['per_class']:
                if cls['support'] and cls['recall_all_GT'] == 0:
                    holds.append(dict(domain=domain, scale=scale, state=CLASS_NAMES[cls['class_index']],
                                      reason='confidence-qualified mechanism has zero recall on supported source class'))
    result['independent_source_mechanism_HOLD'] = dict(HOLD=bool(holds), reasons=holds,
                                                      role='Diagnostic only; no model or threshold selection')
    result['raw_SAM_binary'] = {}
    for domain in ['ALL', *sorted(set(source))]:
        mask = np.ones(len(y), bool) if domain == 'ALL' else source == domain
        result['raw_SAM_binary'][domain] = {}
        for scale, label in enumerate(['exact', 'radius2']):
            # A hard >0 decision is recorded, avoiding a sigmoid/tie convention change.
            result['raw_SAM_binary'][domain][label] = binary_metrics(PAIRS[y[mask], scale] == 0,
                                                                           (raw_sam[mask, scale] > 0).astype(float))
    args.prefix.parent.mkdir(parents=True, exist_ok=True)
    args.prefix.with_suffix('.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    write_csv(str(args.prefix)+'_CLASS_METRICS.csv', metric_rows)
    write_csv(str(args.prefix)+'_EVIDENCE_METRICS.csv', evidence_rows)
    write_csv(str(args.prefix)+'_CONFUSION.csv', confusion_rows)
    # Explicit pE distinguishes low-E logical calls and joint confidence U.
    statep = joint_marginals(p)
    confidence = confident_joint_map(p)
    logical = hierarchical_states(p)
    prediction_rows = []
    for i, row in enumerate(queries):
        record = dict(sample_id=row['sample_id'], source=row['source'], image_sha256=row['image_sha256'],
                      truth_joint=int(y[i]), joint_MAP=int(p[i].argmax()),
                      mechanism_joint_or_U=int(confidence['joint'][i]),
                      max_joint_probability=float(confidence['max_joint_probability'][i]))
        for scale, name in enumerate(['exact', 'radius2']):
            record[name+'_truth_class'] = int(PAIRS[y[i], scale])
            record[name+'_logical_class'] = int(logical[i, scale])
            record[name+'_mechanism_class_or_U'] = int(confidence['classes'][i, scale])
            for cls, clsname in enumerate(['pV', 'pE', 'pO']):
                record[name+'_'+clsname] = float(statep[i, scale, cls])
        prediction_rows.append(record)
    write_csv(str(args.prefix)+'_HEAD_QUERY_PROBABILITIES.csv', prediction_rows)
    print(json.dumps(dict(output=str(args.prefix.with_suffix('.json')), n=len(y), sources=result['sources'],
                          head_joint_NLL={k:v['joint_NLL'] for k,v in result['results']['FROZEN_HEAD'].items()},
                          mechanism_HOLD=result['independent_source_mechanism_HOLD'])))


if __name__ == '__main__':
    main()
