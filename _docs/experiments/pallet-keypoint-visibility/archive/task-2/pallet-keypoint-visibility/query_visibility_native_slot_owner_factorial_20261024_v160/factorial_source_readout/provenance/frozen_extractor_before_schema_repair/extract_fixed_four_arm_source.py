"""Fixed V160 source readout; no model import, forward, fit, or arm selection.

--prepare reads code/protocol/input-contract metadata and invented arrays only.
--extract requires root's explicit all-four-completion message before any result
body is opened. All output paths are fixed and writes are exclusive.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
import numpy as np

H = Path(__file__).resolve().parent
V, P = H.parent, H.parent.parent
PROTOCOL = V / 'source_fit_preparation/PROTOCOL_BEFORE_SOURCE_FEATURE_OR_MODEL_BODIES.json'
CONTRACT = V / 'input_contract_audit/INPUT_BINDINGS_AND_NATIVE_SLOT_PROVENANCE.json'
DESIGN = V / 'ROOT_DESIGN_BEFORE_NEW_SOURCE_FEATURE_OR_MODEL_BODIES.json'
RUN = V / 'runs/C39_C40_C41_C42_NATIVE_SLOT_OWNER_FIXED8000'
ARMS = ('C39_BASE_NATIVE0_OWNER0', 'C40_NATIVE_SLOT8_OWNER0',
        'C41_NATIVE0_OWNER3_AUX', 'C42_NATIVE_SLOT8_OWNER3_AUX')
FAMILIES = ('OLD6144', 'NEW20480', 'PALLOX512', 'MR1840')
COUNTS = (6144, 20480, 512, 1840)
EFFECTS = {
    'native_slot_at_owner0_C40_minus_C39': (-1, 1, 0, 0),
    'owner_at_native0_C41_minus_C39': (-1, 0, 1, 0),
    'native_slot_at_owner1_C42_minus_C41': (0, 0, -1, 1),
    'owner_at_native1_C42_minus_C40': (0, -1, 0, 1),
    'interaction_C42_minus_C41_minus_C40_plus_C39': (1, -1, -1, 1),
    'average_native_slot_main_effect': (-.5, .5, -.5, .5),
    'average_owner_main_effect': (-.5, -.5, .5, .5),
}
RATES = ('D_precision', 'D_recall', 'E_precision', 'E_recall', 'prediction_U_rate_known')
AUDIT_READS = []


def safe(path):
    p = Path(path).resolve()
    assert p.is_relative_to(P.resolve()), p
    assert not any(x in str(p).lower() for x in ('_v112', 'current223', '/private/')), p
    return p


def sha(path):
    return hashlib.sha256(safe(path).read_bytes()).hexdigest()


def bind(path):
    p = safe(path)
    return dict(path=str(p), sha256=sha(p))


def read(path):
    p = safe(path)
    AUDIT_READS.append(str(p))
    return json.loads(p.read_text())


def check(b):
    p = safe(b['path'])
    assert sha(p) == b['sha256'], p
    return p


def rows(binding):
    p = check(binding)
    AUDIT_READS.append(str(p))
    return [json.loads(line) for line in p.read_text().splitlines() if line]


def write(name, obj):
    with (H / name).open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def ratio(n, d):
    return float(n / d) if d else None


def metric(y, pred, av, mask, states):
    """CM includes known truth with factual U; unknown truth stays separate."""
    assert y.shape == pred.shape == av.shape == mask.shape == states.shape
    assert av.dtype == mask.dtype == bool
    assert np.isin(y, (-1, 0, 1)).all() and np.isin(pred, (-1, 0, 1)).all()
    assert np.array_equal(pred == -1, ~av)
    known = mask & (y >= 0)
    cm = np.asarray([[int((known & (y == t) & (pred == p)).sum())
                      for p in (0, 1, -1)] for t in (0, 1)])
    n = int(cm.sum())
    out = dict(CM_D_E_rows_D_E_U_columns=cm.tolist(), known_DE=n,
               known_D=int(cm[0].sum()), known_E=int(cm[1].sum()),
               D_precision=ratio(cm[0, 0], cm[:, 0].sum()),
               E_precision=ratio(cm[1, 1], cm[:, 1].sum()),
               D_recall=ratio(cm[0, 0], cm[0].sum()),
               E_recall=ratio(cm[1, 1], cm[1].sum()),
               prediction_U_known=int(cm[:, 2].sum()),
               prediction_U_rate_known=ratio(cm[:, 2].sum(), n),
               all_original_rows=int(mask.sum()),
               unknown_truth=int((mask & (y < 0)).sum()),
               factual_input_U_all=int((mask & ~av).sum()),
               factual_input_U_known=int((known & ~av).sum()),
               factual_input_U_unknown_truth=int((mask & (y < 0) & ~av).sum()),
               prediction_D_E_U_all=[int((mask & (pred == p)).sum()) for p in (0, 1, -1)],
               truth_state_counts={str(s): int((mask & (states == s)).sum()) for s in np.unique(states)})
    assert n + out['unknown_truth'] == out['all_original_rows']
    assert out['prediction_U_known'] == out['factual_input_U_known']
    assert sum(out['prediction_D_E_U_all']) == out['all_original_rows']
    return out


def paired_metric(y, pred, av, mask, states):
    return dict(all_original_rows=metric(y, pred, av, mask, states),
                input_valid_diagnostic=metric(y, pred, av, mask & av, states))


def normalized_existing(rr, binary=False):
    return dict(CM_D_E_rows_D_E_U_columns=rr['CM_rows0_1_columns0_1_U' if binary else 'CM_D_E_rows_D_E_U_columns'],
                D_precision=rr['negative_precision' if binary else 'D_precision'],
                E_precision=rr['positive_precision' if binary else 'E_precision'],
                D_recall=rr['negative_recall' if binary else 'D_recall'],
                E_recall=rr['positive_recall' if binary else 'E_recall'])


def same_metric(computed, existing, binary=False):
    for key, value in normalized_existing(existing, binary).items():
        assert computed[key] == value, (key, computed[key], value)


def factorial(group_by_arm):
    """No optimum: fixed signed count changes and percentage-point effects."""
    assert set(group_by_arm) == set(ARMS)
    names = set(group_by_arm[ARMS[0]])
    assert all(set(group_by_arm[a]) == names for a in ARMS)
    out = {}
    for group in sorted(names):
        metrics = [group_by_arm[a][group] for a in ARMS]
        supports = [(m['all_original_rows'], m['known_D'], m['known_E'], m['unknown_truth'],
                     m['factual_input_U_all']) for m in metrics]
        assert all(s == supports[0] for s in supports)
        out[group] = {}
        for effect, weights in EFFECTS.items():
            cm = sum(w * np.asarray(m['CM_D_E_rows_D_E_U_columns'], float)
                     for w, m in zip(weights, metrics))
            active = [(w, m) for w, m in zip(weights, metrics) if w != 0]
            rates = {key: None if any(m[key] is None for w, m in active)
                     else float(100 * sum(w * m[key] for w, m in active))
                     for key in RATES}
            out[group][effect] = dict(CM_signed_count_change=cm.tolist(), rate_percentage_point_change=rates)
    return out


def metadata(protocol, contract):
    expected = rows(protocol['expected_source_metadata'])
    assert len(expected) == 28976
    result = {}
    for family, count in zip(FAMILIES, COUNTS):
        inp = [r for r in expected if r['family'] == family]
        assert len(inp) == count
        labels = rows(contract['source_families'][family]['role_labels'])
        if family in FAMILIES[:2]:
            assert len(labels) == count * 2
            labels = [r for r in labels if r['convention'] == 'CF_PRIMARY_DECLARED']
            assert len(labels) == count
            state = np.asarray([r['teacher_state'] for r in labels])
            y = np.where(state == 'D', 0, np.where(state == 'E', 1, -1)).astype(np.int8)
        elif family == 'PALLOX512':
            state = np.asarray([r['semantic_state'] for r in labels])
            y = np.where(state == 'D', 0, np.where(state == 'E', 1, -1)).astype(np.int8)
        else:
            state = np.asarray([r['role_state'] for r in labels])
            y = np.asarray([r['role_target'] for r in labels], np.int8)
        assert len(labels) == count
        for i, (r, label) in enumerate(zip(inp, labels)):
            assert r['global_index'] == i and label['native_slot'] == r['native_slot']
            assert label['split'] == r['split']
            assert label['query_id'] == r['query_id']
        av = np.asarray([r['query_available'] for r in inp], bool)
        assert int((~av).sum()) == contract['source_families'][family]['inputU']
        meta = dict(input=inp, labels=labels, y=y, state=state, av=av,
                    split=np.asarray([r['split'] for r in inp]),
                    native=np.asarray([r['native_slot'] for r in inp], np.int8))
        if family in FAMILIES[:2]:
            for factor in ('material', 'aspect', 'geometry_family', 'foreign_family'):
                meta[factor] = np.asarray([r['factors'][factor] for r in labels])
            assert set(meta['material']) == {'PROCEDURAL_WOOD', 'DARK_POLYMER'}
            assert set(meta['aspect']) == {'RECTANGULAR', 'SQUARE'}
        result[family] = meta
    assert sum(int((~m['av']).sum()) for m in result.values()) == 535
    return result


def masks(meta):
    out = {}
    for fold in ['ALL', *sorted(set(meta['split']))]:
        f = np.ones(len(meta['y']), bool) if fold == 'ALL' else meta['split'] == fold
        out[fold] = f
        for slot in range(8):
            out[f'{fold}::native{slot}'] = f & (meta['native'] == slot)
        for factor in ('material', 'aspect', 'geometry_family', 'foreign_family'):
            if factor in meta:
                for value in sorted(set(meta[factor])):
                    out[f'{fold}::{factor}::{value}'] = f & (meta[factor] == value)
        if 'material' in meta:
            for material in sorted(set(meta['material'])):
                for aspect in sorted(set(meta['aspect'])):
                    ma = f & (meta['material'] == material) & (meta['aspect'] == aspect)
                    name = f'{fold}::material_aspect::{material}::{aspect}'
                    out[name] = ma
                    for slot in range(8):
                        out[f'{name}::native{slot}'] = ma & (meta['native'] == slot)
    return out


def output_prediction(path, meta):
    """Read saved final output and metadata members only; no feature/weight input."""
    path = safe(path)
    AUDIT_READS.append(str(path))
    with np.load(path, allow_pickle=False) as f:
        assert 'features' not in f.files
        av = f['query_available'].copy()
        slot = f['native_slot'].copy()
        index = f['global_index'].copy()
        pred = f['primary_visibility_prediction'].copy()
        raw = f['raw_three_logits'].copy()
        identity = f['occurrence_id' if 'occurrence_id' in f.files else 'query_id'].astype(str)
        assert np.array_equal(av, meta['av']) and np.array_equal(slot, meta['native'])
        assert np.array_equal(index, np.arange(len(av)))
        assert identity.tolist() == [r['query_id'] for r in meta['input']]
        assert raw.shape == (len(av), 3) and np.isfinite(raw[av]).all()
        assert np.array_equal(pred, np.where(av, (raw[:, 0] >= 0).astype(np.int8), -1))
    return pred


def prepare():
    protocol, contract = read(PROTOCOL), read(CONTRACT)
    assert protocol['arms'] == list(ARMS) and protocol['updates_each'] == 8000
    assert contract['source_rows_total'] == 28976
    invented = generated()
    write('GENERATED_EXTRACTOR_CHECKS.json', invented)
    plan = dict(schema='V160_FIXED_SOURCE_READOUT_PLAN', completed=True, prepared_only=True,
                code=bind(__file__), protocol=bind(PROTOCOL), contract=bind(CONTRACT), design=bind(DESIGN),
                source_families=dict(zip(FAMILIES, COUNTS)), factual_input_U_total=535,
                arms=list(ARMS), updates_each=8000, threshold_logit=0,
                completion_trigger='Root must explicitly confirm all four fits completed; then pass --extract --root-all4-completion-note with that message.',
                result_bodies_read_during_preparation=False,
                predicted_or_score_or_checkpoint_or_metric_bodies_read_during_preparation=0,
                outputs=['ARM_DATASET_SUMMARY.json', 'FULL_NATIVE_MATERIAL_ASPECT_TABLES.json',
                         'FACTORIAL_EFFECTS_DIAGNOSTIC.json', 'ORIGINAL_GATE_AND_SEPARATE_AUXILIARY_DIAGNOSTICS.json', 'EXTRACTION_COMPLETE.json'],
                top_summary='Each arm: all-source and four family ALL/train/held cohorts, full known D/E CM rows D/E columns D/E/U, precision/recall, unknown truth, factual U; separate input-valid diagnostic.',
                full_tables='Every family/fold/native0..7; OLD and NEW full material/aspect/material-aspect/native cross and geometry/foreign groups. All empty/unknown cohorts retained.',
                MR_held_full=dict(all_rows=360, known_D=102, known_E=52, known_DE=154, known_D_factual_U=9,
                                  input_valid_known_D=93, input_valid_known_E=52, input_valid_known_DE=145),
                denominators='Known factual U remains in D/E recall and CM. Unknown truth is separate; all-source U535 is an input fact, not a known-truth count. Precision is among known-truth predictions of that class.',
                fixed_effects={name:dict(zip(ARMS, weights)) for name, weights in EFFECTS.items()},
                effect_units='CM signed count change; precision/recall/U-rate change in percentage points. No causal population/generalization claim.',
                gate_reporting='Copy all original 35 limits/results; separately report primary visibility component, unchanged full prospective source gate, and historical AUX29/original64 not retested/not claimed.',
                owner_reporting='Separate full253 weak TRAIN owner diagnostic only; no held owner truth, physical native/pixel/coordinate gold, visibility gate or selection.',
                selected_model=None, fully_solved=False, current_allowed=False,
                new_model_forward_or_fit_or_feature_or_checkpoint_or_RGB_or_private_reads=0)
    write('OUTPUT_PLAN_BEFORE_RESULT_BODIES.json', plan)
    write('PREPARATION_COMPLETE.json', dict(completed=True, prepared_only=True, code=bind(__file__),
          plan=bind(H/'OUTPUT_PLAN_BEFORE_RESULT_BODIES.json'), generated_checks=bind(H/'GENERATED_EXTRACTOR_CHECKS.json'),
          reads=AUDIT_READS.copy(), no_V160_result_bodies_read=True, selected_model=None))
    print(json.dumps(dict(prepared=True, plan=bind(H/'OUTPUT_PLAN_BEFORE_RESULT_BODIES.json'))))


def generated():
    y = np.asarray([0, 0, 1, 1, -1, -1], np.int8)
    av = np.asarray([1, 0, 1, 1, 0, 1], bool)
    pred = np.asarray([0, -1, 0, 1, -1, 1], np.int8)
    state = np.asarray(['D', 'D', 'E', 'E', 'U', 'S'])
    pair = paired_metric(y, pred, av, np.ones(6, bool), state)
    full, valid = pair['all_original_rows'], pair['input_valid_diagnostic']
    assert full['CM_D_E_rows_D_E_U_columns'] == [[1, 0, 1], [1, 1, 0]]
    assert full['D_recall'] == .5 and full['D_precision'] == .5
    assert full['E_recall'] == .5 and full['E_precision'] == 1.
    assert full['factual_input_U_all'] == 2 and full['prediction_U_known'] == 1
    assert full['unknown_truth'] == 2 and full['factual_input_U_unknown_truth'] == 1
    assert valid['known_DE'] == 3 and valid['D_recall'] == 1 and valid['unknown_truth'] == 1
    empty = metric(y, pred, av, np.zeros(6, bool), state)
    assert all(empty[k] is None for k in RATES) and empty['known_DE'] == 0
    fake = {a:dict(test=dict(full)) for a in ARMS}
    for i, a in enumerate(ARMS):
        fake[a]['test'] = dict(full, D_recall=(.1, .3, .4, .8)[i])
    effects = factorial(fake)['test']
    assert abs(effects['interaction_C42_minus_C41_minus_C40_plus_C39']['rate_percentage_point_change']['D_recall'] - 20) < 1e-12
    assert abs(effects['average_native_slot_main_effect']['rate_percentage_point_change']['D_recall'] - 30) < 1e-12
    assert np.asarray(effects['owner_at_native0_C41_minus_C39']['CM_signed_count_change']).sum() == 0
    bad = {a:dict(test=dict(fake[a]['test'])) for a in ARMS}
    bad[ARMS[2]]['test']['known_D'] += 1
    try:
        factorial(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('Differing support must fail closed')
    return dict(completed=True, PASS=True, code=bind(__file__), invented_arrays_only=True,
                factual_U_in_full_recall_denominator=True, unknown_truth_U_separate=True,
                input_valid_conditioning_explicit=True, precision_denominators_checked=True,
                empty_support_rates_null=True, fixed_factorial_interaction_and_average_units_checked=True,
                unequal_support_rejected=True, source_result_feature_weight_RGB_or_model_calls=0,
                runtime=dict(python=sys.version.split()[0], numpy=np.__version__))


def extract(root_note):
    assert root_note and len(root_note.strip()) >= 20, 'Explicit root ALL4 completion message required'
    preparation = read(H/'PREPARATION_COMPLETE.json')
    assert preparation['no_V160_result_bodies_read']
    assert check(preparation['code']) == Path(__file__).resolve()
    protocol, contract = read(PROTOCOL), read(CONTRACT)
    # The completion seal is opened first; no partial arm inspection is allowed.
    control = read(RUN/'CONTROL_COMPLETE.json')
    assert control['completed'] and control['PASS'] and control['updates_each'] == 8000
    assert set(control['arms']) == set(ARMS) and control['selected_model'] is None
    assert control['source_only'] and control['all_original_native26624_public512_MR1840_retained']
    assert control['equal_original_initial_states_and_auxiliary_initial_final_outputs_and_common_draws']
    assert control['protocol'] == bind(PROTOCOL)
    frozen = read(check(control['all4_frozen_before_source_comparison']))
    assert frozen['completed'] and frozen['arms'] == list(ARMS)
    assert frozen['source_validation_reports_emitted'] is False
    all_meta = metadata(protocol, contract)
    summary, tables, separate, flat = {}, {}, {}, {}
    output_bindings = []
    for arm in ARMS:
        directory = RUN / arm
        fit = read(check(control['arms'][arm]['fit']))
        assert fit['completed'] and fit['optimizer_updates'] == 8000
        report = read(check(fit['source_results']))
        freeze = read(check(fit['model_freeze']))
        assert report['completed'] and report['arm'] == arm and report['update'] == freeze['update'] == 8000
        assert freeze['threshold_logit'] == 0 and report['selected_model'] is None
        assert not report['current_allowed'] and report['primary_visibility_never_filtered_by_entity_quality_or_GTerror']
        summary[arm], tables[arm], flat[arm] = {}, {}, {}
        joined = {k:[] for k in ('y', 'state', 'av', 'pred')}
        for family in FAMILIES:
            meta = all_meta[family]
            output = directory / f'FINAL_{family}_OUTPUTS.npz'
            pred = output_prediction(output, meta)
            output_bindings.append(bind(output))
            family_table = {name:paired_metric(meta['y'], pred, meta['av'], mask, meta['state'])
                            for name, mask in masks(meta).items()}
            tables[arm][family] = family_table
            summary[arm][family] = {fold: family_table[fold] for fold in ['ALL', *sorted(set(meta['split']))]}
            for name, pair in family_table.items():
                for conditioning, values in pair.items():
                    flat[arm][f'{family}::{name}::{conditioning}'] = values
            for key in ('y', 'state', 'av'):
                joined[key].append(meta[key])
            joined['pred'].append(pred)
            for fold, pair in summary[arm][family].items():
                if family in FAMILIES[:2]:
                    rr = report['native']['old_native' if family == FAMILIES[0] else 'new_native'][fold]
                    same_metric(pair['all_original_rows'], rr)
                elif family == 'PALLOX512':
                    same_metric(pair['all_original_rows'], report['Pallox']['groups'][fold])
                    same_metric(pair['input_valid_diagnostic'], report['Pallox']['groups'][fold+'::INPUT_VALID'])
                else:
                    same_metric(pair['all_original_rows'], report['MR'][fold]['visibility'], binary=True)
                    same_metric(pair['input_valid_diagnostic'], report['MR'][fold]['visibility_inputvalid'], binary=True)
        j = {key:np.concatenate(values) for key, values in joined.items()}
        all_source = paired_metric(j['y'], j['pred'], j['av'], np.ones(28976, bool), j['state'])
        assert all_source['all_original_rows']['factual_input_U_all'] == 535
        summary[arm]['ALL_SOURCE28976'] = dict(ALL=all_source)
        for conditioning, values in all_source.items():
            flat[arm][f'ALL_SOURCE28976::ALL::{conditioning}'] = values
        held = summary[arm]['MR1840']['SOURCE_VALIDATION']
        assert (held['all_original_rows']['known_D'], held['all_original_rows']['known_E']) == (102, 52)
        assert (held['input_valid_diagnostic']['known_D'], held['input_valid_diagnostic']['known_E']) == (93, 52)
        assert held['all_original_rows']['factual_input_U_known'] == 9
        gates = report['native']['gates']
        assert len(gates) == 35 and sum(g['PASS'] for g in gates.values()) == report['native']['native35_passed']
        vis = report['primary_visibility_component']['MR_held_visibility']
        primary_pass = bool(report['native']['native35_all_pass'] and report['Pallox']['sanity']['PASS']
                            and vis['negative_recall'] is not None and vis['negative_recall'] >= .90
                            and vis['positive_recall'] is not None and vis['positive_recall'] >= .80)
        owner = report['owner_diagnostic']
        cm_owner = np.asarray(owner['CM_rows_truth_T_F_BG_columns_prediction_T_F_BG'])
        assert cm_owner.shape == (3,3) and int(cm_owner.sum()) == 253
        gate_summary = dict(native35_passed=report['native']['native35_passed'], native35_total=35,
                            native35_all_pass=report['native']['native35_all_pass'],
                            primary_visibility_component=report['primary_visibility_component'],
                            primary_visibility_component_diagnostic_PASS=primary_pass,
                            MR_full_visibility_and_entity_weak_sanity_PASS=report['MR_weak_sanity_PASS'],
                            original_full_prospective_source_checks_all_pass=report['prospective_source_checks_all_pass'],
                            original_AUX29_not_retested=True, original64_qualification_not_claimed=True,
                            fully_solved=False, adopted=False, current_allowed=False, selected_model=None)
        summary[arm]['gates'] = gate_summary
        separate[arm] = dict(original_native35_gates=gates, gate_summary=gate_summary,
                            Pallox_original_sanity=report['Pallox']['sanity'],
                            MR_entity_and_secondary_gate_diagnostics=report['MR'],
                            synthetic_point_quality_diagnostic=report['synthetic_point_quality'],
                            owner_weak_TRAIN_only=dict(original=owner, accuracy=float(np.trace(cm_owner)/253),
                                per_class_precision=[ratio(cm_owner[k,k], cm_owner[:,k].sum()) for k in range(3)],
                                per_class_recall=[ratio(cm_owner[k,k], cm_owner[k].sum()) for k in range(3)],
                                no_held_owner_accuracy=True, no_native_or_pixel_gold=True))
    header = dict(schema='V160_FIXED_FOUR_ARM_SOURCE_READOUT', source_only=True, selected_model=None,
                  fully_solved=False, adopted=False, current_allowed=False, threshold_logit=0,
                  CM_order=dict(rows=['D','E'], columns=['D','E','U']),
                  truth='Existing separate CF_PRIMARY_DECLARED synthetic native role and public assistant weak role labels; literal owner is separate.',
                  factual_U='Original input fact; cannot be revived by any branch. Known factual U included in CM and recall; unknown truth reported separately.',
                  rates='Fractions; precision denominator contains known truth predictions of that class. Factorial rate changes use percentage points.')
    write('ARM_DATASET_SUMMARY.json', dict(**header, arms=summary))
    write('FULL_NATIVE_MATERIAL_ASPECT_TABLES.json', dict(**header, arms=tables))
    write('FACTORIAL_EFFECTS_DIAGNOSTIC.json', dict(**header, diagnostic_only=True,
          formulas={name:dict(zip(ARMS,w)) for name,w in EFFECTS.items()}, groups=factorial(flat)))
    write('ORIGINAL_GATE_AND_SEPARATE_AUXILIARY_DIAGNOSTICS.json', dict(**header, arms=separate))
    names = ('ARM_DATASET_SUMMARY.json','FULL_NATIVE_MATERIAL_ASPECT_TABLES.json',
             'FACTORIAL_EFFECTS_DIAGNOSTIC.json','ORIGINAL_GATE_AND_SEPARATE_AUXILIARY_DIAGNOSTICS.json')
    write('EXTRACTION_COMPLETE.json', dict(completed=True, UTC=datetime.now(timezone.utc).isoformat(),
          root_all4_completion_message=root_note, code=bind(__file__), preparation=bind(H/'PREPARATION_COMPLETE.json'),
          protocol=bind(PROTOCOL), contract=bind(CONTRACT), control=bind(RUN/'CONTROL_COMPLETE.json'),
          all4_frozen=control['all4_frozen_before_source_comparison'], outputs=[bind(H/name) for name in names],
          final_saved_output_bindings=output_bindings, exact_reads=AUDIT_READS,
          original28976_rows_and535_factual_U_per_arm_preserved=True,
          MR_held154_known_including9_factual_U_and145_inputvalid_both_preserved=True,
          original_fixed_threshold_masks_gates_and_truth_unchanged=True, selected_model=None,
          source_features_or_checkpoint_weights_or_RGB_or_model_or_encoder_or_PnP_or_private_calls=0))
    print(json.dumps(dict(completed=True, summary=bind(H/'ARM_DATASET_SUMMARY.json'), receipt=bind(H/'EXTRACTION_COMPLETE.json'))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--extract', action='store_true')
    parser.add_argument('--root-all4-completion-note')
    args = parser.parse_args()
    if args.prepare:
        assert args.root_all4_completion_note is None
        prepare()
    else:
        extract(args.root_all4_completion_note)
