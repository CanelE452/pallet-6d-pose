"""Evaluate sealed SAM2/source scores only; no models, RGB, features or current data.

All actual GT/score reads are behind a root-approved source evaluation ENTRYLOCK
and a completed extraction seal. --mock uses generated arrays only.
"""
from pathlib import Path
from collections import Counter
import argparse
import copy
import csv
import hashlib
import json
import math
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent
V66 = BASE / 'query_visibility_controlled_real_20261022_v66'
V67 = BASE / 'query_visibility_visible_preservation_20261023_v67'
V63 = BASE / 'query_visibility_public_real_20261022_v63'
PROTOCOL_SHA = '1e6a99030faf8b2c6cd1144b70b09395d55609c2b746c9fd7d12ad6f7e622128'
INPUT_SEAL_SHA = '081c4a58cce0a20830872358d1559b59ef75a57df333bdc357a164d35abec979'
EXTRACTOR_SHA = '185ccea1e43ff8cf43e57694c8e382d2dd8911538286a1a8f44218fdefdb81aa'
CHECKPOINT_SHA = 'a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5'
DENY_SHA = '58ef8b2ca79b9cb511fe7b5813d22b31f6ff82ef5049a2be340ec293bdbef987'
SOURCES = ('SYNTH', 'ORIGINAL_YCB', 'CONTROLLED_YCB', 'PALLOX')
COUNTS = dict(SYNTH=1742, ORIGINAL_YCB=3200, CONTROLLED_YCB=3200, PALLOX=448)
PRIMARY = 'SAM21_BASE_PLUS_FROZEN'
PRED_NAMES = ('VISIBLE', 'ABSENT', 'U')
GT_NAMES = ('V', 'E', 'O')
GT_KEYS = ('sample_id', 'source', 'group_id', 'image_sha256', 'global_id', 'global_source',
           'global_image_sha256', 'target_index', 'dense_allowed', 'dense_valid',
           'dense_exact_state', 'dense_radius2_state')
SAM_KEYS = ('sample_id', 'source', 'group_id', 'global_id', 'native_valid', 'exact_logit',
            'radius2_logit', 'exact_visible', 'radius2_visible', 'selected_mask_index')
OLD_KEYS = ('sample_id', 'source', 'group_id', 'image_sha256', 'center_native_valid',
            'center_allowed', 'GT_state', 'visible_probability', 'raw_SAM_visible_probability')
FORBIDDEN_ARRAYS = {'x', 'rgb_native', 'sam', 'context', 'box_grid', 'global_rgb', 'global_sam', 'global_prompt'}

def require(value, message):
    if not value:
        raise ValueError(message)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()

def ref(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))

def bound(binding, path=None, expected_sha=None):
    actual = Path(binding['path']).resolve()
    if path is not None:
        require(actual == Path(path).resolve(), 'Unexpected source binding path: ' + str(actual))
    if expected_sha is not None:
        require(binding['sha256'] == expected_sha, 'Unapproved source digest: ' + str(actual))
    require(sha(actual) == binding['sha256'], 'Changed source metadata/output: ' + str(actual))
    return actual

def same_binding(a, b):
    """Optional byte-count metadata does not change the path/SHA identity."""
    return Path(a['path']).resolve() == Path(b['path']).resolve() and a['sha256'] == b['sha256']

def read_json(path):
    return json.loads(Path(path).read_text())

def csv_rows(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))

def boolean(value):
    if value in (True, 'True', 'true', 1, '1'):
        return True
    if value in (False, 'False', 'false', 0, '0'):
        return False
    raise ValueError('Explicit boolean required')

def arrays(path, keys):
    require(not (set(keys) & FORBIDDEN_ARRAYS), 'Feature arrays forbidden')
    with np.load(path, allow_pickle=False) as z:
        require(all(k in z.files for k in keys), 'Missing saved identity/GT/score array')
        return {k: z[k] for k in keys}

def fraction(a, b):
    return int(a) / int(b) if b else None

def states_from_allowed(allowed, valid):
    a = np.asarray(allowed)
    valid = np.asarray(valid)
    require(a.shape == valid.shape and a.ndim == 1 and valid.dtype == bool, 'Aligned native centers')
    require(np.isin(a, np.arange(1, 64)).all(), 'Nonempty compatible sets1..63 required')
    require((a[~valid] == 63).all(), 'Invalid centers must remain uninformative')
    result = np.full((len(a), 2), -1, np.int8)
    for k, groups in enumerate(((1, 6, 56), (11, 20, 32))):
        possible = np.stack([(a & bits) != 0 for bits in groups], 1)
        unique = possible.sum(1) == 1
        result[valid & unique, k] = possible[valid & unique].argmax(1)
    return result

def binary_truth(allowed, valid, support):
    """Additional marginal-known cohort; no E/O mechanism is invented."""
    a = np.asarray(allowed)
    vbits = 1 if support == 0 else 11
    positive = (a & (63 ^ vbits)) == 0
    negative = (a & vbits) == 0
    answer = np.full(len(a), -1, np.int8)
    answer[valid & positive] = 1
    answer[valid & negative] = 0
    return answer

def decisions(score, valid, comparison, threshold):
    s = np.asarray(score)
    require(comparison in ('>', '>='), 'Only registered strict/inclusive comparisons')
    require(s.shape == (len(valid), 2), 'Exact/radius2 scores required')
    require(np.isfinite(s[valid]).all(), 'Finite scores required for valid native queries')
    answer = np.full(s.shape, -1, np.int8)
    answer[valid] = ((s[valid] > threshold) if comparison == '>' else (s[valid] >= threshold)).astype(np.int8)
    return answer

def binary_metrics(truth, prediction):
    positive = truth == 1
    negative = truth == 0
    tp = int((positive & (prediction == 1)).sum())
    fp = int((negative & (prediction == 1)).sum())
    fn = int((positive & (prediction != 1)).sum())
    tn = int((negative & (prediction == 0)).sum())
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, N=len(truth), positive=int(positive.sum()),
        negative=int(negative.sum()), U=int((prediction < 0).sum()), precision=fraction(tp, tp + fp),
        recall=fraction(tp, positive.sum()), nonvisible_recall=fraction(tn, negative.sum()))

def support_metrics(allowed, valid, gt, prediction, support):
    require(prediction.shape == gt.shape and np.isin(prediction, [-1, 0, 1]).all(), 'Saved binary decision shape/range')
    truth = gt[:, support]; pred = prediction[:, support]
    known = valid & (truth >= 0); unknown = valid & (truth < 0)
    table = np.zeros((3, 3), np.int64)
    # Model columns are VISIBLE / ABSENT / U, not V/E/O causal classes.
    column = np.where(pred == 1, 0, np.where(pred == 0, 1, 2))
    for g, p in zip(truth[known], column[known]):
        table[int(g), int(p)] += 1
    binary = binary_metrics((truth[known] == 0).astype(np.int8), pred[known])
    def nonvisible_for(c):
        ix = valid & (truth == c)
        return dict(N=int(ix.sum()), nonvisible=int((ix & (pred == 0)).sum()),
            visible=int((ix & (pred == 1)).sum()), U=int((ix & (pred < 0)).sum()),
            recall=fraction((ix & (pred == 0)).sum(), ix.sum()),
            interpretation='Binary nonvisibility on known ' + GT_NAMES[c] + '; not recognition precision/recall for a predicted causal class')
    marginal_truth = binary_truth(allowed, valid, support)
    mk = marginal_truth >= 0
    marginal = binary_metrics(marginal_truth[mk], pred[mk])
    return dict(total_rows=len(valid), native_valid=int(valid.sum()), native_invalid=int((~valid).sum()),
        known_class_GT=int(known.sum()), UNKNOWN_class_GT=int(unknown.sum()),
        native_valid_all63=int((valid & (allowed == 63)).sum()), informative_partial=int((valid & (allowed != 63)).sum()),
        confusion=dict(GT_rows=list(GT_NAMES), prediction_columns=list(PRED_NAMES), counts=table.tolist()),
        visible_binary_on_class_known=binary,
        known_E_to_binary_nonvisible=nonvisible_for(1), known_O_to_binary_nonvisible=nonvisible_for(2),
        UNKNOWN_class_predictions={name: int((unknown & (column == j)).sum()) for j, name in enumerate(PRED_NAMES)},
        all_native_valid_predictions={name: int((valid & (column == j)).sum()) for j, name in enumerate(PRED_NAMES)},
        invalid_predictions={name: int((~valid & (column == j)).sum()) for j, name in enumerate(PRED_NAMES)},
        binary_known_partial_diagnostic=dict(scope='All compatible-set marginal-known facts; a different denominator from class-known primary',
            unknown_marginal=int((valid & (marginal_truth < 0)).sum()), **marginal))

def summarize(gt, models):
    output = {}
    for name, model in models.items():
        output[name] = dict(decision_rule=model['rule'], ties=model.get('ties'), per_source={})
        for source in SOURCES:
            ix = gt['source'] == source
            output[name]['per_source'][source] = {
                support_name: support_metrics(gt['allowed'][ix], gt['valid'][ix], gt['states'][ix], model['prediction'][ix], support)
                for support, support_name in enumerate(('exact', 'radius2_diagnostic'))}
    return output

def diagnostic_checks(spec, result, strict):
    rows = []
    def add(name, source, metric, value, minimum):
        rows.append(dict(condition=name, source=source, metric=metric, value=value, minimum=minimum,
                         passed=value is not None and value >= minimum))
    for source in spec['public_exact_visible']['sources']:
        m = result['per_source'][source]['exact']['visible_binary_on_class_known']
        for key in ('precision', 'recall'):
            add('public_exact_visible', source, key, m[key], spec['public_exact_visible'][key + '_at_least'])
    for source in spec['known_external_to_binary_nonvisible_recall']['sources']:
        m = result['per_source'][source]['exact']['known_E_to_binary_nonvisible']
        add('known_E_to_binary_nonvisible', source, 'recall', m['recall'], spec['known_external_to_binary_nonvisible_recall']['recall_at_least'])
    for source, rule in spec['strict25_visible_nonregression'].items():
        require(strict[source]['N'] == rule['N'], 'Frozen strict25 denominator changed')
        add('strict25_visible_nonregression', source, 'TP', strict[source]['TP'], rule['TP_at_least'])
    require(len(rows) == 10, 'All ten prefrozen source diagnostic conditions required')
    return rows

def preflight(entry_path, out):
    require(entry_path is not None, 'Root-approved source evaluation ENTRYLOCK required')
    entry = read_json(entry_path)
    require(entry['schema'] == 'V68_SAVED_SOURCE_EVALUATION_ENTRYLOCK_V1', 'Evaluation entry schema')
    for k, value in dict(approved=True, source_only=True, current223_allowed=False,
                         model_inference_allowed=False, model_training_allowed=False).items():
        require(entry.get(k) is value, 'Invalid source evaluation scope: ' + k)
    bound(entry['evaluator'], __file__)
    require(Path(out).resolve() == Path(entry['output_directory']).resolve() and not Path(out).exists(), 'Preserve locked output')
    require(Path(out).resolve().is_relative_to(ROOT), 'Evaluation outputs must remain inside V68')
    require(entry['expected'] == dict(targets=1091, queries=8590, native_valid=8584, native_invalid=6), 'Fixed source cohort changed')
    require(entry['thresholds'] == dict(SAM21_logit_strict_gt=0., MobileSAM_probability_strict_gt=.5, head_probability_ge=.5), 'No threshold search')
    protocol_path = bound(entry['inputs']['protocol'], ROOT / 'PROTOCOL.json', PROTOCOL_SHA)
    protocol = read_json(protocol_path)
    input_seal_path = bound(entry['inputs']['input_seal'], ROOT / 'source_inputs/SEAL.json', INPUT_SEAL_SHA)
    input_seal = read_json(input_seal_path)
    require(input_seal['completed'] and input_seal['queries'] == 8590 and input_seal['targets'] == 1091, 'Source input seal incomplete')
    extractor = bound(entry['inputs']['extractor_code'], ROOT / 'extract_source_masks.py', EXTRACTOR_SHA)
    extraction_path = bound(entry['inputs']['extraction_seal'])
    require(extraction_path.resolve().is_relative_to(ROOT / 'source_outputs'), 'Extraction must remain source-only')
    extraction = read_json(extraction_path)
    for k, value in dict(completed=True, source_only=True, current_inference_allowed=False,
            model_weights_unchanged=True, checkpoint_unchanged=True, GT_depth_pose_features_annotations_read=False,
            current_RGB_GT_scores_inputs_read=False, training_calls=0, optimizer_updates=0, PnP_calls=0).items():
        require(extraction.get(k) is value if isinstance(value, bool) else extraction.get(k) == value, 'Invalid extraction seal: ' + k)
    for k, value in entry['expected'].items():
        require(extraction[k] == value, 'Extraction source counts changed')
    require(same_binding(extraction['bindings']['source_protocol'], entry['inputs']['protocol']) and same_binding(extraction['bindings']['input_seal'], entry['inputs']['input_seal']), 'Extraction used different source contract')
    require(extraction['bindings']['extractor']['sha256'] == EXTRACTOR_SHA and extraction['bindings']['checkpoint']['sha256'] == CHECKPOINT_SHA, 'Extraction model/code drift')
    require(extraction['model_state_sha256'] == extraction['final_model_state_sha256'], 'Frozen model state changed')
    require(extraction['model_input_whitelist'] == ['sealed original RGB', 'unchanged float box'] and not extraction['query_is_prompt'] and not extraction['GT_branch_selection'], 'Box-only visible baseline required')
    require(extraction['runtime']['float32_weights'] and extraction['runtime']['cuda_autocast'] == 'bfloat16'
            and extraction['runtime']['compile_calls'] == 0 and extraction['runtime']['batch_targets'] == 1, 'Frozen numerical recipe changed')
    require(Path(extraction['bindings']['entry_lock']['path']).resolve().is_relative_to(ROOT), 'Extraction ENTRYLOCK must remain source-only')
    extract_entry_path = bound(extraction['bindings']['entry_lock'])
    extract_entry = read_json(extract_entry_path)
    require(extract_entry['approved'] is True and extract_entry['source_only'] is True and extract_entry['current223_allowed'] is False, 'Extraction entry scope')
    # No checkpoint/wheel/model deserialization/hash is needed in a saved evaluator.
    identity = {}
    for key, name in (('manifest', 'INPUTS_RGB_ONLY.jsonl'), ('query_id_map', 'QUERY_ID_MAP.csv')):
        binding = next(x for x in input_seal['outputs'] if Path(x['path']).name == name)
        require(same_binding(extraction['bindings'][key], binding), 'Extraction source identity drift')
        identity[key] = bound(binding, ROOT / 'source_inputs' / name)
    output_bindings = {Path(x['path']).name: x for x in extraction['outputs']}
    sam_npz = bound(output_bindings['SOURCE_QUERY_OUTPUTS.npz'], extraction_path.parent / 'SOURCE_QUERY_OUTPUTS.npz')
    sam_csv = bound(output_bindings['PER_QUERY.csv'], extraction_path.parent / 'PER_QUERY.csv')
    prior_seal_path = bound(entry['inputs']['joined_GT_seal'], V66 / 'joined_source_cache/SEAL.json')
    prior_seal = read_json(prior_seal_path)
    registered = next(x for x in prior_seal['outputs'] if x['candidate'] == 'CONTROLLED_PLUS_PALLOX' and x['split'] == 'VAL')
    gt_path = V66 / 'joined_source_cache/CONTROLLED_PLUS_PALLOX_VAL.npz'
    require(entry['inputs']['GT_cache']['sha256'] == registered['sha256']
            and Path(entry['inputs']['GT_cache']['path']).resolve() == gt_path.resolve()
            and gt_path.stat().st_size == registered['bytes'], 'Frozen source GT cache registration changed')
    # Trust prior immutable full-file seal; do not read compressed feature payload.
    deny_path = bound(entry['inputs']['denylist'], V63 / 'contract/EVALUATION_SHA256_DENYLIST.json', DENY_SHA)
    deny = set(read_json(deny_path)['image_sha256']); require(len(deny) == 223, 'Complete evaluation image exclusion list')
    baseline_files, baseline_meta = {}, {}
    allowed_roots = dict(V65_BASELINE=V66 / 'source_evaluation/V65_BASELINE',
        V66_CONTROLLED_ONLY=V66 / 'source_evaluation/CONTROLLED_ONLY',
        V66_CONTROLLED_PLUS_PALLOX=V66 / 'source_evaluation/CONTROLLED_PLUS_PALLOX',
        V67_VISIBLE_BCE1=V67 / 'source_evaluation')
    require(set(entry['baselines']) == set(allowed_roots), 'Four frozen source heads required, no private model selection')
    for name, binding in entry['baselines'].items():
        bp = bound(binding)
        require(bp.name == 'SOURCE_METRICS.json' and bp.is_relative_to(allowed_roots[name]), 'Only registered source score reports allowed')
        bm = read_json(bp)
        require(bm['completed'] is True and bm['source_only'] is True and bm['current223_RGB_GT_scores_read'] is False, 'Source baseline provenance')
        require(bm['N'] == 8590 and bm['sources'] == list(SOURCES) and not bm['thresholds_fitted'] and not bm['model_or_checkpoint_selected'], 'Frozen source score comparison only')
        npzb = next(x for x in bm['outputs'] if Path(x['path']).name == 'SOURCE_OUTPUTS.npz')
        baseline_files[name] = bound(npzb, bp.parent / 'SOURCE_OUTPUTS.npz')
        baseline_meta[name] = dict(report=ref(bp), output=npzb)
    strict_audit_path = bound(entry['inputs']['strict_reference_audit'], V66 / 'source_audit/SOURCE_BOUNDARY_FAILURE_DIAGNOSIS.json')
    strict_audit = read_json(strict_audit_path)
    require(strict_audit['completed'] and strict_audit['current_RGB_GT_scores_inputs_read'] == 0, 'Source-only strict25 reference required')
    strict_b = next(x for x in strict_audit['output_bindings'] if Path(x['path']).name == 'SOURCE_BOUNDARY_FAILURE_DIAGNOSIS_QUERY_ROWS.csv')
    strict_path = bound(strict_b)
    return dict(entry=entry, protocol=protocol, GT=gt_path, deny=deny, sam_npz=sam_npz, sam_csv=sam_csv,
        identity=identity, baseline_files=baseline_files, strict_path=strict_path,
        bindings=dict(evaluation_entry=ref(entry_path), evaluator=ref(__file__), source_protocol=ref(protocol_path),
            input_seal=ref(input_seal_path), extractor_code=ref(extractor), extraction_seal=ref(extraction_path),
            extraction_entry=ref(extract_entry_path), GT_prior_seal=ref(prior_seal_path), GT_cache_prior_binding=registered,
            strict_reference_audit=ref(strict_audit_path), strict_reference_rows=ref(strict_path),
            denylist=ref(deny_path), SAM_saved_npz=ref(sam_npz), SAM_saved_csv=ref(sam_csv), baselines=baseline_meta))

def load_truth(plan):
    d = arrays(plan['GT'], GT_KEYS)
    require(len(d['sample_id']) == 8590 and len(d['global_id']) == 1091, 'GT source cohort')
    require(len(set(d['sample_id'])) == 8590 and len(set(d['global_id'])) == 1091, 'Unique source IDs')
    ti = d['target_index']; require(ti.shape == (8590,) and set(ti) == set(range(1091)), 'Target index coverage')
    require(np.array_equal(d['global_source'][ti], d['source']) and np.array_equal(d['global_image_sha256'][ti], d['image_sha256']), 'GT target identity mapping')
    require(not (set(d['image_sha256'].astype(str)) & plan['deny']), 'Current evaluation SHA in source GT')
    require(dict(Counter(d['source'])) == COUNTS, 'Four source query counts')
    allowed = d['dense_allowed'][:, 4, 4]; valid = d['dense_valid'][:, 4, 4]
    require(d['dense_allowed'].shape == d['dense_valid'].shape == (8590, 9, 9), 'Native label map shape')
    states = states_from_allowed(allowed, valid)
    require(np.array_equal(states[:, 0], d['dense_exact_state'][:, 4, 4]) and np.array_equal(states[:, 1], d['dense_radius2_state'][:, 4, 4]), 'Partial compatible bits versus saved exact/radius2 GT')
    require(int(valid.sum()) == 8584 and (allowed[valid & (states[:, 0] == 0)] == 1).all(), 'Frozen native-valid and knownV=VV invariants')
    return dict(sample_id=d['sample_id'], source=d['source'], group_id=d['group_id'], image_sha256=d['image_sha256'],
        global_id=d['global_id'][ti], allowed=allowed, valid=valid, states=states)

def load_scores(plan, gt):
    d = arrays(plan['sam_npz'], SAM_KEYS); rows = csv_rows(plan['sam_csv']); mapping = csv_rows(plan['identity']['query_id_map'])
    with Path(plan['identity']['manifest']).open() as f:
        target_identity = {r['global_id']: r for r in map(json.loads, f)}
    require(len(target_identity) == 1091, 'Source native shape identity coverage')
    require(len(rows) == len(mapping) == 8590, 'Sealed SAM/source map row count')
    for key in ('sample_id', 'source', 'group_id', 'global_id'):
        require(np.array_equal(d[key], gt[key]), 'Canonical SAM/GT identity order: ' + key)
    require(np.array_equal(d['native_valid'], gt['valid']), 'SAM native validity differs from fixed GT canvas')
    score = np.stack((d['exact_logit'], d['radius2_logit']), 1)
    require(score.dtype == np.float32 and d['native_valid'].dtype == bool, 'Native float32 logits and Boolean validity required')
    pred = decisions(score, gt['valid'], '>', 0.)
    require(np.array_equal(pred, np.stack((d['exact_visible'], d['radius2_visible']), 1)), 'SAM strict zero decisions')
    require(np.isnan(score[~gt['valid']]).all(), 'Invalid SAM scores must remain NaN/U')
    require((score[gt['valid'], 0] <= score[gt['valid'], 1]).all(), 'Exact/radius2 selected-mask inclusion')
    for i, (r, q) in enumerate(zip(rows, mapping)):
        require(int(r['joined_query_index']) == int(q['joined_query_index']) == i, 'Canonical saved query order')
        for key in ('sample_id', 'global_id', 'source', 'group_id', 'image_sha256'):
            require(r[key] == q[key] == str(gt[key][i]), 'SAM CSV/map/GT identity: ' + key)
        h, w = int(r['native_H']), int(r['native_W']); x, y = float(r['query_x']), float(r['query_y'])
        require([h, w] == target_identity[r['global_id']]['raw_hw'], 'Saved native shape changed from source metadata')
        px, py = math.floor(x + .5), math.floor(y + .5)
        valid = 0 <= x < w and 0 <= y < h and 0 <= px < w and 0 <= py < h
        require(valid == boolean(r['native_valid']) == boolean(q['native_input_valid']) == bool(gt['valid'][i]), 'Native continuous/rounded bounds')
        require((px, py) == (int(r['pixel_x']), int(r['pixel_y'])) == (int(q['pixel_x']), int(q['pixel_y'])), 'Floor(q+.5) native sampling')
        require((x, y) == (float(q['query_x']), float(q['query_y'])) and float(r['side']) == float(q['side']), 'Original query/side preserved')
        quality = [float(r['predicted_quality_' + str(j)]) for j in range(3)]
        require(np.isfinite(quality).all() and int(r['selected_mask_index']) == int(np.argmax(quality)) == int(d['selected_mask_index'][i]), 'Predicted-quality max/tie-first')
        for k, support in enumerate(('exact', 'radius2')):
            require(int(r[support + '_visible']) == int(pred[i, k]), 'CSV/NPZ strict decisions')
            v = float(r[support + '_logit'])
            require(v == float(score[i, k]) if valid else math.isnan(v), 'CSV/NPZ raw logits')
        require(r['status'] == ('VALID' if valid else 'U'), 'Invalid status cannot be padded negative')
    models = {PRIMARY: dict(prediction=pred, rule='saved native float32 logit >0 strict', ties=[int((score[gt['valid'], k] == 0).sum()) for k in range(2)])}
    raw = None
    for name, path in plan['baseline_files'].items():
        old = arrays(path, OLD_KEYS)
        for key in ('sample_id', 'source', 'group_id', 'image_sha256'):
            require(np.array_equal(old[key], gt[key]), 'Frozen baseline canonical identity: ' + name + '/' + key)
        require(np.array_equal(old['center_native_valid'], gt['valid']) and np.array_equal(old['center_allowed'], gt['allowed'])
                and np.array_equal(old['GT_state'], gt['states']), 'Frozen baseline partial GT/canvas must be identical')
        probability = old['visible_probability']
        require(probability.shape == (8590, 2) and np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all(), 'Saved head probability range')
        models[name] = dict(prediction=decisions(probability, gt['valid'], '>=', .5), rule='saved pVisible >=.5 fixed', ties=[int((probability[gt['valid'], k] == .5).sum()) for k in range(2)])
        if raw is None:
            raw = old['raw_SAM_visible_probability']
        else:
            require(np.array_equal(raw, old['raw_SAM_visible_probability']), 'Raw MobileSAM recipe differs across baseline reports')
    require(raw.shape == (8590, 2) and np.isfinite(raw).all() and ((raw >= 0) & (raw <= 1)).all(), 'Saved raw MobileSAM probabilities')
    models['MOBILE_SAM_RAW'] = dict(prediction=decisions(raw, gt['valid'], '>', .5), rule='saved sigmoid(raw logit) >.5 strict, equivalent to raw logit>0; zero tie negative',
        ties=[int((raw[gt['valid'], k] == .5).sum()) for k in range(2)])
    return models

def strict_counts(plan, gt, models):
    reference = csv_rows(plan['strict_path']); lookup = {str(s): i for i, s in enumerate(gt['sample_id'])}
    result = {name: {} for name in models}
    for source, target in plan['protocol']['source_diagnostic_quality']['strict25_visible_nonregression'].items():
        old = [r for r in reference if r['model'] == 'V65_BASELINE' and r['source'] == source]
        require(len(old) == 3200 and {r['sample_id'] for r in old} == set(gt['sample_id'][gt['source'] == source]), 'Strict25 complete same-query reference')
        selected = []
        for r in old:
            i = lookup[r['sample_id']]
            require(int(r['allowed']) == int(gt['allowed'][i]) and boolean(r['native_valid']) == bool(gt['valid'][i])
                    and int(r['exact_GT']) == int(gt['states'][i, 0]) and int(r['radius2_GT']) == int(gt['states'][i, 1]), 'Strict25 fixed partial truth')
            if boolean(r['strict25_V']):
                require(gt['valid'][i] and gt['allowed'][i] == 1, 'Strict25 must be known nativeVV')
                selected.append(i)
        require(len(selected) == target['N'], 'Strict25 fixed denominator')
        require(sum(float(r['exact_pV']) >= .5 for r in old if boolean(r['strict25_V'])) == target['TP_at_least'], 'FrozenV65 strict25 anchor')
        id_hash = hashlib.sha256(('\n'.join(sorted(str(gt['sample_id'][i]) for i in selected)) + '\n').encode()).hexdigest()
        for name, model in models.items():
            p = model['prediction'][selected, 0]; tp = int((p == 1).sum())
            result[name][source] = dict(N=len(selected), TP=tp, FN=len(selected) - tp, U=int((p < 0).sum()), recall=fraction(tp, len(selected)), sorted_sample_ids_sha256=id_hash)
    return result

def write_result(out, report):
    out = Path(out); require(not out.exists(), 'Preserve saved evaluation'); out.mkdir(parents=True)
    flat = []
    for model, arm in report['models'].items():
        for source, supports in arm['per_source'].items():
            for support, m in supports.items():
                b = m['visible_binary_on_class_known']
                flat.append(dict(model=model, source=source, support=support, native_valid=m['native_valid'], native_invalid=m['native_invalid'],
                    class_known=m['known_class_GT'], class_UNKNOWN=m['UNKNOWN_class_GT'],
                    visible_TP=b['TP'], visible_FP=b['FP'], visible_FN=b['FN'], visible_TN=b['TN'], visible_precision=b['precision'], visible_recall=b['recall'],
                    known_E_N=m['known_E_to_binary_nonvisible']['N'], known_E_to_nonvisible_recall=m['known_E_to_binary_nonvisible']['recall'],
                    known_O_N=m['known_O_to_binary_nonvisible']['N'], known_O_to_nonvisible_recall=m['known_O_to_binary_nonvisible']['recall'],
                    confusion_GT_V_E_O_pred_VISIBLE_ABSENT_U=json.dumps(m['confusion']['counts'])))
    with (out / 'SOURCE_RESULTS.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0])); w.writeheader(); w.writerows(flat)
    (out / 'evaluate_saved_source.py').write_bytes(Path(__file__).read_bytes())
    report['outputs'] = [ref(out / name) for name in ('SOURCE_RESULTS.csv', 'evaluate_saved_source.py')]
    report['completed'] = True
    (out / 'SOURCE_RESULTS.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(completed=True, diagnostics=report['source_diagnostic_status'], current_inference_allowed=False, report=ref(out / 'SOURCE_RESULTS.json'))))

def evaluate(entry, out):
    plan = preflight(entry, out)  # All seals/guards before any GT/score array.
    gt = load_truth(plan); models = load_scores(plan, gt); result = summarize(gt, models)
    strict = strict_counts(plan, gt, models)
    checks = diagnostic_checks(plan['protocol']['source_diagnostic_quality'], result[PRIMARY], strict[PRIMARY])
    require(sha(__file__) == plan['entry']['evaluator']['sha256'] and sha(ROOT / 'PROTOCOL.json') == PROTOCOL_SHA, 'Evaluation code/protocol changed during saved scoring')
    report = dict(schema='V68_SAVED_SOURCE_BINARY_VISIBILITY_RESULTS_V1', models=result, strict25_visible=strict,
        primary_model=PRIMARY, source_diagnostic_checks=checks,
        source_diagnostic_status='PASS' if all(c['passed'] for c in checks) else 'SOURCE_HOLD', bindings=plan['bindings'],
        queries=8590, targets=1091, native_valid=8584, native_invalid=6,
        primary_precision_scope='Native-valid exact-class-known V/E/O GT; UNKNOWN separate. Additional compatible-set binary-known cohort uses a distinct denominator.',
        mechanism_limit='SAM mask absence predicts binary nonvisibility only. E/O labels are source truth strata, not predicted causal classes. No confidentE/O class precision exists for this model.',
        evidence_limit='Synthetic render, BOP render/depth masks, controlled cutout, and PALLOX conditional teacher are separate source evidence tiers, not human pixel gold or semantic pallet-corner truth.',
        source_only=True, solved=False, current_inference_allowed=False, automatic_current_gate_issued=False,
        current_RGB_GT_scores_inputs_read=False, source_RGB_reads=0, feature_array_reads=0, model_weight_reads=0,
        new_model_forward_calls=0, optimizer_updates=0, PnP_calls=0, thresholds_changed=False, checkpoint_or_model_selected=False,
        GT_array_whitelist=list(GT_KEYS), score_array_whitelist=list(SAM_KEYS), head_saved_score_whitelist=list(OLD_KEYS))
    write_result(out, report)

def mock(path):
    path = Path(path); require(not path.exists(), 'Preserve generated proof')
    a = np.tile(np.arange(1, 64, dtype=np.uint8), 4); valid = np.ones(len(a), bool); valid[-1] = False
    states = states_from_allowed(a, valid)
    require((a[valid & (states[:, 0] == 0)] == 1).all(), 'KnownV onlyVV')
    exact = binary_truth(a, valid, 0)
    require(int((exact[:63] == 1).sum()) == 1 and int((exact[:63] == 0).sum()) == 31 and int((exact[:63] < 0).sum()) == 31, 'All63 mask marginal facts')
    score = np.zeros((len(a), 2), np.float32); score[0] = np.nextafter(np.float32(0), np.float32(1))
    sam = decisions(score, valid, '>', 0.)
    require(sam[0, 0] == 1 and sam[1, 0] == 0 and (sam[~valid] == -1).all(), 'Strict tie/subnormal/invalid')
    mp = np.full((len(a), 2), .5, np.float64)
    require((decisions(mp, valid, '>', .5)[valid] == 0).all() and (decisions(mp, valid, '>=', .5)[valid] == 1).all(), 'Preserve Mobile/head tie distinction')
    source = np.repeat(np.asarray(SOURCES), 63)
    gt = dict(source=source, allowed=a, valid=valid, states=states)
    summary = summarize(gt, {'MOCK_SAM': dict(prediction=sam, rule='strict>0')})['MOCK_SAM']
    for s in SOURCES:
        ix = source == s; m = summary['per_source'][s]['exact']; table = np.asarray(m['confusion']['counts'])
        require(int(table.sum()) == int((valid[ix] & (states[ix, 0] >= 0)).sum()), 'Class-known confusion count')
        require(table.sum() == 11 and table[1].sum() == 3 and table[2].sum() == 7, 'Independent exhaustive exact-class-known counts')
        require(m['binary_known_partial_diagnostic']['N'] == 32, 'Additional binary-known facts use separate denominator')
        require(sum(m['UNKNOWN_class_predictions'].values()) == m['UNKNOWN_class_GT'], 'Retain UNKNOWN prediction coverage')
        require(m['known_E_to_binary_nonvisible']['N'] == int((valid[ix] & (states[ix, 0] == 1)).sum()), 'External stratum denominator')
    empty = binary_metrics(np.zeros(3, np.int8), np.zeros(3, np.int8))
    require(empty['precision'] is None and empty['recall'] is None, 'Undefined visibleP/R remain undefined')
    # Pure metric spec uses generated scalar summaries; every one can fail closed.
    spec = dict(public_exact_visible=dict(sources=list(SOURCES[1:]), precision_at_least=.8, recall_at_least=.8),
        known_external_to_binary_nonvisible_recall=dict(sources=['CONTROLLED_YCB', 'PALLOX'], recall_at_least=.6),
        strict25_visible_nonregression=dict(ORIGINAL_YCB=dict(N=714, TP_at_least=679), CONTROLLED_YCB=dict(N=296, TP_at_least=278)))
    pass_result = dict(per_source={s: dict(exact=dict(visible_binary_on_class_known=dict(precision=.8, recall=.8),
        known_E_to_binary_nonvisible=dict(recall=.6))) for s in SOURCES})
    strict = dict(ORIGINAL_YCB=dict(N=714, TP=679), CONTROLLED_YCB=dict(N=296, TP=278))
    checks = diagnostic_checks(spec, pass_result, strict); require(all(x['passed'] for x in checks), 'Threshold equality passes')
    require(len(checks) == 10, 'Ten prefrozen diagnostics')
    for c in checks:
        rr, ss = copy.deepcopy(pass_result), copy.deepcopy(strict)
        if c['metric'] == 'TP':
            ss[c['source']]['TP'] = c['minimum'] - 1
        elif c['condition'] == 'public_exact_visible':
            rr['per_source'][c['source']]['exact']['visible_binary_on_class_known'][c['metric']] = math.nextafter(c['minimum'], -math.inf)
        else:
            rr['per_source'][c['source']]['exact']['known_E_to_binary_nonvisible']['recall'] = math.nextafter(c['minimum'], -math.inf)
        failed = [x for x in diagnostic_checks(spec, rr, ss) if not x['passed']]
        require(len(failed) == 1 and failed[0]['condition'] == c['condition'] and failed[0]['source'] == c['source'] and failed[0]['metric'] == c['metric'], 'Every diagnostic must fail independently')
    try:
        preflight(None, ROOT / 'UNUSED_GENERATED_OUTPUT')
    except ValueError:
        pass
    else:
        raise AssertionError('Missing evaluation gate must fail before arrays')
    report = dict(schema='V68_GENERATED_SAVED_SOURCE_EVALUATOR_MOCK_V1', PASS=True, code=ref(__file__),
        generated_rows=len(a), all63_compatible_sets=True, knownV_allowed1=True,
        class_known_confusion_V_E_O_by_VISIBLE_ABSENT_U=True, UNKNOWN_coverage_separate=True,
        allowed_binary_known_cohort_separate=True, strict_zero_and_invalid_unknown=True,
        MobileSAM_strict_vs_head_inclusive_tie=True, no_causal_E_O_precision=True,
        prefrozen_10_diagnostic_conditions_equality_and_individual_failure_PASS=True, no_automatic_current_authorization=True,
        actual_GT_or_saved_score_or_feature_or_weight_or_RGB_or_current_reads=0,
        model_forward_calls=0, optimizer_updates=0, PnP_calls=0, GPU_calls=0)
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(PASS=True, report=ref(path))))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--entry-lock', type=Path); p.add_argument('--out', type=Path); p.add_argument('--mock', type=Path)
    a = p.parse_args()
    if a.mock:
        require(a.entry_lock is None and a.out is None, 'Mock cannot mix source inputs'); mock(a.mock)
    else:
        if a.entry_lock is None or a.out is None:
            p.error('--entry-lock and --out required')
        evaluate(a.entry_lock, a.out)

if __name__ == '__main__':
    main()
