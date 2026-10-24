"""Independent V70 evaluator review: source code and generated arrays only."""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import math
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'source_evaluation/evaluate_saved_ownership.py'
EXPECTED = '8e2c7ff74624c9d88e32d620ca1e0c29f2dbd57a06a3a7297e715cc943ecf768'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def binding(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


assert sha(CODE) == EXPECTED
spec = importlib.util.spec_from_file_location('independent_generated_v70_evaluator', CODE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

# Derive class knowledge from literal joint pairs, independently of bit constants.
pairs = ((0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2))
allowed = np.arange(1, 64, dtype=np.int64)
valid = np.ones(63, bool)
expected = np.full((63, 2), -1, np.int8)
for i, mask in enumerate(allowed):
    for scale in (0, 1):
        options = {pairs[j][scale] for j in range(6) if int(mask) & (1 << j)}
        if len(options) == 1:
            expected[i, scale] = next(iter(options))
assert np.array_equal(m.states(allowed, valid), expected)
gt = dict(states=expected, allowed=allowed, valid=valid)
for prediction in (-1, 0, 1):
    model = dict(prediction=np.full(63, prediction, np.int8),
                 p=np.full(63, .37), logit=None)
    result = m.cohort_summary(gt, model, valid)
    known = [i for i in range(63) if expected[i, 0] >= 0]
    ve = [i for i in known if expected[i, 0] in (0, 1)]
    binary_known = [i for i, mask in enumerate(allowed)
                    if mask == 1 or not (int(mask) & 1)]
    assert (result['known_class_GT'], result['conditional_known_VE']['N'],
            result['partial_binary_known']['N']) == (len(known), len(ve), len(binary_known)) == (11, 4, 32)
    assert sum(map(sum, result['class_known_confusion']['counts'])) == 11
    assert result['visible_binary_ALL_known_VEO']['TP'] == int(prediction == 1)
    assert result['visible_binary_ALL_known_VEO']['FP'] == (10 if prediction == 1 else 0)
    assert result['conditional_known_VE']['FP'] == (3 if prediction == 1 else 0)
    assert result['partial_binary_known']['FP'] == (31 if prediction == 1 else 0)
    assert result['visible_binary_ALL_known_VEO']['U'] == (11 if prediction == -1 else 0)
    assert result['UNKNOWN_class_GT'] == 52

logits = np.array([0., np.nextafter(np.float32(0), np.float32(1)),
                   np.nextafter(np.float32(0), np.float32(-1)), np.nan], np.float64)
native_valid = np.array([True, True, True, False])
assert m.decide(logits, native_valid, True, 0.).tolist() == [0, 1, 0, -1]
assert m.decide(np.array([.5, .51, .49, np.nan]), native_valid).tolist() == [1, 1, 0, -1]
m.probability_parity(np.array([0., np.nan]), np.array([.5, np.nan]),
                     np.array([True, False]), 1e-7)
for z in (-1000., 1000.):
    loss = m.nll(dict(logit=np.array([z]), p=None),
                 np.array([float(z < 0)]), np.array([True]))
    assert loss['mean'] == 1000. and loss['nonfinite'] == 0

# Run exactly the final source code's isolated history-check block on generated metadata.
text = CODE.read_text()
history_block = text.split('    history=read(history_path);', 1)[1].split('    # Earlier completed', 1)[0]
history_block = 'history=read(history_path);' + history_block
history_block = '\n'.join(line[4:] if line.startswith('    ') else line
                          for line in history_block.splitlines())
history_code = compile(history_block, str(CODE) + ':generated_history_block', 'exec')


def history_check(history, freeze):
    exec(history_code, dict(read=lambda _: history, history_path='GENERATED_NO_READ',
                           require=m.require, math=math, freeze=freeze))


history = []
best = float('inf')
for update in range(100, 2001, 100):
    value = 1. if update == 100 else .5
    improved = value < best
    history.append(dict(update=update, VAL_macro_knownVE_NLL=value, saved_best=improved))
    best = min(best, value)
freeze = dict(best_update=200, best_source_macro_knownVE_NLL=.5)
history_check(history, freeze)
for bad_history, bad_freeze in [
    (history, dict(best_update=300, best_source_macro_knownVE_NLL=.5)),
    (history[:-1], freeze),
    ([dict(row, saved_best=True) if row['update'] == 300 else row for row in history], freeze),
    ([dict(row, VAL_macro_knownVE_NLL=float('nan')) if row['update'] == 300 else row
      for row in history], freeze),
]:
    try:
        history_check(bad_history, bad_freeze)
    except ValueError:
        pass
    else:
        raise AssertionError('Generated malformed history admitted')

tree = ast.parse(text)
functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
order = [n.value.func.id for n in functions['evaluate'].body
         if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
         and isinstance(n.value.func, ast.Name)]
assert order[:3] == ['preflight', 'load_gt', 'load_models']
imports = {x.name.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for x in n.names}
assert not imports & {'torch', 'cv2', 'PIL', 'ultralytics', 'sam2'}

# Author's separate generated pair proof is itself code-only; stop before its output write.
pair_code = ROOT / 'source_audit/review_generated_saved_pair_metrics.py'
prefix = pair_code.read_text().split("out=ROOT/'source_audit/GENERATED_SAVED_PAIR_METRICS_REVIEW.json'")[0]
exec(compile(prefix, str(pair_code), 'exec'), {'__file__': str(pair_code)})

builtin = ROOT / 'source_audit/INDEPENDENT_EVALUATOR_BUILTIN_GENERATED_MOCK.json'
assert json.loads(builtin.read_text())['status'] == 'PASS'
assert sha(CODE) == EXPECTED
out = ROOT / 'source_audit/INDEPENDENT_SAVED_OWNERSHIP_EVALUATOR_REVIEW.json'
assert not out.exists()
report = dict(
    schema='V70_INDEPENDENT_SAVED_OWNERSHIP_EVALUATOR_CODE_GENERATED_REVIEW_V1',
    completed=True, status='PASS', blockers=[],
    evaluator=binding(CODE), review_script=binding(__file__),
    author_mock=binding(ROOT / 'source_evaluation/GENERATED_SAVED_EXACT_EVALUATOR_MOCK_V2.json'),
    independent_builtin_mock=binding(builtin),
    generated_pair_review=binding(ROOT / 'source_audit/GENERATED_SAVED_PAIR_METRICS_REVIEW.json'),
    generated_pair_review_code=binding(pair_code),
    tests=dict(
        all_63_joint_compatible_masks_independent_set_projection=True,
        ALL_knownVEO_conditionalVE_partial_binary_known_denominators=[11, 4, 32],
        all_visible_all_absent_and_U_confusions=True,
        UNKNOWN_class_and_invalid_native_rows_retained=True,
        valid_only_v61_logit_probability_NaN_parity=True,
        strict_SAM_zero_and_positive_subnormal_vs_inclusive_head_half=True,
        stable_NLL_extreme_logits=True,
        full_20_step_earliest_strict_minimum_history_and_four_rejections=True,
        generated_VAL_pairs=3200, preserved_V_pairs=1126, changed_VE_pairs=445,
        pair_transition_counts_U_TRAIN_and_identity_rejection=True,
        ten_fixed_diagnostics_vs_eight_prefit_eligibility_no_auto_current=True,
        preflight_before_GT_or_saved_scores=True,
        no_model_or_image_imports=True),
    static_review=dict(
        source_selection_freeze_protocol_history_and_code_hashes_bound_before_load=True,
        strict25_same_ID_native_valid_knownV_and_frozen_denominator_TP_checks=True,
        strict25_ORIGINAL_N_TP=[714, 679], strict25_CONTROLLED_N_TP=[296, 278],
        selected_saved_output_macro_NLL_replay_does_not_reselect_checkpoint=True,
        main_invalid_scores_blank_and_prediction_U=True,
        absent_not_causal_external_no_E_precision_or_invented_radius2=True),
    scope='Code and generated arrays only; review does not certify actual run output integrity or actual gate outcome.',
    actual_source_GT_scores_RGB_features_weights_current_reads=0,
    model_forward_calls=0, optimizer_updates=0, PnP_calls=0,
    current_inference_allowed=False)
out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
print(json.dumps(dict(path=str(out), sha256=sha(out), status=report['status'])))
