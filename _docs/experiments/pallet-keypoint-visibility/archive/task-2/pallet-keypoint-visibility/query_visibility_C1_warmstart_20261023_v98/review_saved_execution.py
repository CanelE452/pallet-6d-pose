"""Posthoc root verification of saved C4 outputs; no model or optimizer IO."""
from pathlib import Path
import hashlib
import json
import numpy as np

V = Path(__file__).resolve().parent
R = V / 'runs/C4_C1_WARMSTART_SOURCE_V1'
OLD = V.parent / 'query_visibility_query_anchored_cue_20261023_v96'

def read(p): return json.loads(Path(p).read_text())
def rows(p): return [json.loads(s) for s in Path(p).read_text().splitlines()]
def bind(p): return dict(path=str(p), sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def arrays(p):
    with np.load(p, allow_pickle=False) as f:
        return {k: f[k].copy() for k in f.files}

complete = read(R / 'FIT_COMPLETE.json')
for k in ('MODEL_FREEZE', 'source_results', 'source_gates', 'scores', 'history', 'train_updates', 'update0_parity'):
    b = complete[k]
    assert bind(b['path'])['sha256'] == b['sha256']
a = arrays(R / 'SELECTED_ALL31680_SOURCE_SCORES.npz')
b = arrays(OLD / 'source_baselines/C1_FROZEN_ON_V96/SELECTED_ALL31680_SOURCE_SCORES.npz')
fixed = ['occurrence_index', 'base_row_index', 'donor_rank', 'input_unit_index', 'occurrence_id', 'input_id',
         'native_slot_logical_only', 'image_sha256', 'original_image_sha256', 'query_xy_raw', 'L_raw',
         'candidate_known', 'scalar_target_D0_E1_Uminus1', 'source', 'split', 'original_split', 'variant', 'accepted_pair']
for k in fixed:
    assert np.array_equal(a[k], b[k], equal_nan=True) if a[k].dtype.kind == 'f' else np.array_equal(a[k], b[k]), k
z, available, y = a['query_logit'], a['query_available'], a['scalar_target_D0_E1_Uminus1']
pred = np.where(available, (z >= 0).astype(np.int8), -1)
assert len(z) == 31680 and np.isfinite(z).all() and np.all(z[~available] == 0)
assert np.array_equal(pred, a['prediction']) and np.array_equal(y >= 0, a['candidate_known'])
saved = read(R / 'SOURCE_RESULTS.json')
summary, matrices = {}, {}
for split in ('TRAIN', 'VALIDATION'):
    m = a['split'] == split
    before = a['variant'] == 'BEFORE'
    masks = dict(LOCO_BEFORE_D=m & before & (a['source'] == 'LOCO') & (y == 0),
                 PALLOX_BEFORE_D=m & before & (a['source'] == 'PALLOX') & (y == 0),
                 SHAM_D=m & (a['variant'] == 'CUE_PRESERVING_SHAM') & (y == 0),
                 BEFORE_E=m & before & (y == 1), COVER_E=m & (a['variant'] == 'OPAQUE_COVER') & (y == 1))
    expect = saved['metrics'][split]
    cm = [[int(np.sum(m & (y == i) & (pred == j))) for j in (0, 1, -1)] for i in (0, 1)]
    assert cm == expect['all_known']['matrix']
    matrices[split] = cm
    result = {}
    for name, mask in masks.items():
        target = 0 if name.endswith('_D') else 1
        n, correct = int(mask.sum()), int(np.sum(mask & (pred == target)))
        key = 'D_recall' if target == 0 else 'E_recall'
        assert n == expect['five_cohorts'][name]['known_N'] and correct / n == expect['five_cohorts'][name][key]
        result[name] = dict(N=n, correct=correct, recall=correct / n)
    pp = pred.reshape(960, 33)
    ready = a['accepted_pair'] & m[::33, None]
    good = (pp[:, 0, None] == 0) & (pp[:, 1::2] == 1) & (pp[:, 2::2] == 0)
    n, correct = int(ready.sum()), int((ready & good).sum())
    assert n == expect['paired']['accepted_pairs'] and correct == expect['paired']['all3_correct']
    result['paired'] = dict(N=n, correct=correct, recall=correct / n)
    result['known_U_rate'] = float(np.sum(m & (y >= 0) & (pred == -1)) / np.sum(m & (y >= 0)))
    summary[split] = result
history = rows(R / 'HISTORY.jsonl')
assert [h['update'] for h in history] == list(range(0, 2001, 100))
for h in history:
    assert h['criterion_five_VAL_cohorts_BCE'] == float(np.mean([c['mean_BCE'] for c in h['VAL_cohorts'].values()]))
assert min(history, key=lambda h: h['criterion_five_VAL_cohorts_BCE'])['update'] == complete['selected_update'] == 2000
c4log, c3log = rows(R / 'TRAIN_UPDATES.jsonl'), rows(OLD / 'runs/C3_QUERY_ANCHORED_SOURCE_V1/TRAIN_UPDATES.jsonl')
assert len(c4log) == len(c3log) == 2000
for i, (x, old) in enumerate(zip(c4log, c3log), 1):
    assert x['update'] == old['update'] == i
    for k in ('occurrence_indices', 'input_unit_indices', 'cohort_quotas'):
        assert x[k] == old[k]
    assert np.isfinite(x['loss']) and np.isfinite(x['gradient_norm_before_clip'])
g = read(R / 'SOURCE_GATE_RESULTS.json')
v = summary['VALIDATION']
checks = [('LOCO_before_D_recall', v['LOCO_BEFORE_D']['recall'], .85),
          ('PALLOX_before_D_recall', v['PALLOX_BEFORE_D']['recall'], .85),
          ('natural_before_E_recall', v['BEFORE_E']['recall'], .75),
          ('sham_D_recall', v['SHAM_D']['recall'], .85), ('cover_E_recall', v['COVER_E']['recall'], .75),
          ('accepted_VAL_pair_all3_correct_rate', v['paired']['recall'], .75)]
for name, value, limit in checks:
    assert g['gates'][name]['value'] == value and g['gates'][name]['PASS'] == (value >= limit)
assert g['gates']['known_VAL_U_rate']['value'] == v['known_U_rate']
assert not g['PASS'] and sum(c['PASS'] for c in g['gates'].values()) == 3
out = dict(completed=True, PASS=True, discrepancies=[], fit_complete=bind(R / 'FIT_COMPLETE.json'),
           review_code=bind(Path(__file__)), root_reviewed_trainer=bind(V / 'source_fit_preparation/train_source_once.py'),
           all31680_fixed_fields_exact=fixed, raw0_and_U_exact=True, all2000_draws_match_audited_C3=True,
           all21_selector_means_and_earliest_minimum_exact=True, selected_update=2000,
           model_source_gates_pass=False, gate_PASS_count=3, gate_FAIL_count=4, metrics=summary,
           confusion_matrices=matrices, pre_optimizer_C1_parity=bind(R / 'UPDATE0_C1_PARITY.json'),
           code_review='Exact C1 state loaded after model construction; full parity precedes fresh AdamW. Shared immutable loss/sampler/loader/selector/metrics, all parameters trainable.',
           limitations=['Saved verification and pinned code/provenance, no independent optimizer/model re-execution.',
                        'Conditional AUX, native association UNKNOWN; repeated sourceVAL and clustered controls are not independent transfer proof.'],
           new_model_checkpoint_parameter_deserializations=0, new_private_current_PnP=0)
with (V / 'ROOT_SAVED_EXECUTION_REVIEW.json').open('x') as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
    f.write('\n')
print(json.dumps(dict(review=bind(V / 'ROOT_SAVED_EXECUTION_REVIEW.json'), selected_update=2000, source_gates_pass=False)))
