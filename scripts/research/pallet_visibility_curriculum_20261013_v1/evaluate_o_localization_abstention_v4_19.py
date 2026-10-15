"""Uniform continuous O-localization-abstention causal control, all238.

Frozen grazing states, actual NMS observations/CAL noise, q/Sigma/reliability,
U/T/S and physical models remain fixed. Only O ellipses become their existing
closed canvas with extra halfplanes retained. Native and native+old-mask use
the unchanged six-diverse-plus-coarse optimizer and all cached contenders.
This is an exploratory constraint ablation, never a new default by definition.
"""
import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path
import shutil
import sys
import types

from . import refine_grazing_image_chain_mixtures_v4_18c as PARENT
from . import o_localization_abstention_v4_19 as ADAPTER

C, E, BASE, CAL, REFINE = PARENT.C, PARENT.E, PARENT.BASE, PARENT.CAL, PARENT.REFINE
OUTPUT = PARENT.OUTPUT.parent/'o_localization_abstention_v4_19'
ARMS = ('SAME_FROZEN_E', 'O_LOCALIZATION_ABSTAINED_NATIVE', 'O_LOCALIZATION_ABSTAINED_PLUS_MASK')
_verified_model = PARENT._verified_model
_WORKER = types.FunctionType(PARENT._run.__code__, {**PARENT._run.__globals__, 'ARMS': ARMS}, PARENT._run.__name__)


def adapt_task(task):
    before = E._digest(task)
    result = ADAPTER.abstain_external_o_localization(task['spec']['domains'], task['states'],
        task['spec']['image_hw'], initial_self_hidden=task['initial_self_hidden'])
    changed = deepcopy(task)
    changed['spec']['domains'] = result['domains']
    changed['O_localization_abstention'] = {k: v for k, v in result.items() if k != 'domains'}
    changed['O_localization_abstention']['source_task_sha256'] = before
    # The observer already ran from states/model/RGB/visible q; it never used
    # the numerical O circle. The complete saved chain archive is kept exact.
    for key in task:
        if key == 'spec':
            for field in task[key]:
                if field != 'domains' and E._digest(task[key][field]) != E._digest(changed[key][field]):
                    raise ValueError('Non-domain solver input changed')
        elif E._digest(task[key]) != E._digest(changed[key]):
            raise ValueError('State/observation/bank/archive changed: '+key)
    if E._digest(task) != before:
        raise ValueError('Source task mutated')
    return changed


def _run(task, model):
    if 'O_localization_abstention' not in task:
        raise ValueError('Prepared O abstention profile required')
    row = _WORKER(task, model)
    row['schema'] = 'O_LOCALIZATION_ABSTENTION_CONTINUOUS_PREDICTION_V4_19'
    row['O_localization_abstention'] = deepcopy(task['O_localization_abstention'])
    row['tangent_control_audit'] = {a: dict(tangent_variance_inflation=4.,
        unchanged_archived_prediction_uncertainty_not_precise_RGB_point=True,
        source='same frozen v18c native objective; metadata supplied before score') for a in ARMS[1:]}
    row['contract'].pop('observation_noise_guards_and_start_budget_unchanged', None)
    row['contract'].update(observation_noise_and_start_budget_unchanged=True,
        only_O_ellipse_localization_hard_bounds_abstained=True,
        original_O_closed_canvas_and_extra_halfplanes_retained=True,
        O_state_preserved=True, O_point_or_soft_losses=0,
        T_U_BI_S_geometry_and_local_queries_unchanged=True,
        causal_constraint_ablation_not_range_method_validation=True)
    changed_ids = task['O_localization_abstention']['changed_native_ids']
    for arm in ARMS[1:]:
        output = row['arms'][arm]
        output['O_localization_abstention'] = deepcopy(task['O_localization_abstention'])
        if output['accepted']:
            for k in changed_ids:
                output['keypoint_readout']['point_sources'][k] = 'JOINT_CANVAS_FEASIBLE_PROJECTION_O_LOCALIZATION_ABSTAINED_NOT_UNIQUE'
        output['keypoint_readout']['original_O_ellipse_constraints_claimed_solved'] = False
        output['keypoint_readout']['O_localization_abstained_native_ids'] = list(changed_ids)
    return row


def prepare(output=OUTPUT):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Preserve every prior preparation')
    parent, seal = PARENT.PARENT._sealed(PARENT.OUTPUT, 'INPUT_SEAL.json', 'INPUTS.jsonl.gz', 'input_sha256', 238)
    prediction = C.read(PARENT.OUTPUT/'PREDICTION_SEAL.json')
    if (prediction['rows'] != 238 or C.sha(PARENT.OUTPUT/'PREDICTIONS.jsonl.gz') != prediction['predictions_sha256']
            or prediction['protocol_sha256'] != C.sha(PARENT.OUTPUT/'PROTOCOL.json')):
        raise ValueError('Full frozen continuous parent predictions required')
    score = C.read(PARENT.OUTPUT/'SCORE_SEAL.json')
    for part, digest in score['score_sha256'].items():
        if C.sha(PARENT.OUTPUT/(part+'_SCORE.json')) != digest:
            raise ValueError('Parent fixed-reference score binding changed')
    _verified_model()
    tasks = [adapt_task(t) for t in C.rows(PARENT.OUTPUT/'INPUTS.jsonl.gz')]
    if (len(tasks) != 238 or [t['id'] for t in tasks] != parent['ids']
            or Counter(t['partition'] for t in tasks) != Counter(CAL=46, TEST=49, REMAINDER=143)):
        raise ValueError('All ordered238 must remain, including zero-BI/fallbacks')
    sources, inputs = dict(parent['source_sha256']), dict(parent['input_sha256'])
    for module in (PARENT, ADAPTER, sys.modules[__name__]):
        path = Path(module.__file__).resolve(); sources[str(path.relative_to(C.ROOT))] = C.sha(path)
    adapter_path = Path(PARENT.__file__).with_name('score_refined_grazing_mixtures_v4_18c.py')
    sources[str(adapter_path.relative_to(C.ROOT))] = C.sha(adapter_path)
    for name in ('PROTOCOL.json', 'INPUT_SEAL.json', 'INPUTS.jsonl.gz', 'PREDICTION_SEAL.json', 'SCORE_SEAL.json',
                 'SCORER_COMPATIBILITY_RECEIPT.json', 'SCORER_COMPATIBILITY_SEAL.json'):
        path = PARENT.OUTPUT/name; inputs[str(path.relative_to(C.ROOT))] = C.sha(path)
    for mapping in (sources, inputs, parent['protected_user_sha256']): E._verify_sources(mapping)
    output.mkdir(parents=True)
    protocol = dict(schema='O_LOCALIZATION_ABSTENTION_ALL238_PROTOCOL_V4_19', ids=parent['ids'], arms=list(ARMS),
        population=parent['population'], source_sha256=sources, input_sha256=inputs,
        protected_user_sha256=parent['protected_user_sha256'], new_NMS_model_binding=parent['new_NMS_model_binding'],
        observation_generation=parent['observation_generation'], refinement_configuration=REFINE.DEFAULT_CONFIG,
        parent_prediction_seal=E._binding(PARENT.OUTPUT/'PREDICTION_SEAL.json'),
        change='Only empirical O ellipse hard localization abstains; existing closed canvas/extra halfplanes remain.',
        changed_O_points=sum(len(t['O_localization_abstention']['changed_native_ids']) for t in tasks),
        changed_O_frames=sum(bool(t['O_localization_abstention']['changed_native_ids']) for t in tasks),
        same_frozen_states_NMS_noise_visible_partial_U_T_S_models_Q_and_bank=True,
        actual_chain_archives_byte_values_unchanged=True, observer_did_not_use_O_circle_for_local_queries=True,
        observer_source_lines={'observe_native_edge_chains_v4_17b.py': [172, 173, 181, 184],
            'image_supported_edge_chains_v4_17c.py': [320]},
        candidate_policy=parent['candidate_policy'], selection=parent['selection'],
        six_diverse_native_ranked_starts_and_own_coarse_per_family_and_T_union=True,
        actual_six_start_subset_can_change_when_more_candidates_become_feasible=True,
        O_point_or_soft_losses=0, new_NN_RGB_or_calibration_calls=0, no_default_adoption=True,
        parameter_free_causal_ablation_not_final_range_method=True,
        all238_prediction_seal_before_new_metrics=True, exploratory_reused_data=True,
        additional143_reference_not_independent_metrology=True)
    C.write(output/'PROTOCOL.json', protocol); C.save_rows(output/'INPUTS.jsonl.gz', tasks)
    for relative, digest in sources.items():
        dest = output/'code_snapshot'/relative; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(C.ROOT/relative, dest)
        if C.sha(dest) != digest: raise ValueError('Source snapshot mismatch')
    for mapping in (sources, inputs, parent['protected_user_sha256']): E._verify_sources(mapping)
    C.write(output/'INPUT_SEAL.json', dict(rows=238, ordered_ids=parent['ids'], input_sha256=C.sha(output/'INPUTS.jsonl.gz'),
        protocol_sha256=C.sha(output/'PROTOCOL.json'), before_new_predictions_or_metric_reads=True))
    print(dict(prepared=238, changed_O_points=protocol['changed_O_points'], changed_O_frames=protocol['changed_O_frames']), flush=True)


def predict(output=OUTPUT, workers=8):
    fn = types.FunctionType(PARENT.OLD.predict.__code__, {**PARENT.OLD.predict.__globals__,
        '_run': _run, '_verified_model': _verified_model}, PARENT.OLD.predict.__name__)
    return fn(Path(output), workers)


def score(output=OUTPUT):
    fn = types.FunctionType(BASE.ARCHIVE.score.__code__, {**BASE.ARCHIVE.score.__globals__,
        'ARMS': ARMS}, BASE.ARCHIVE.score.__name__)
    return fn(Path(output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('command', choices=('prepare', 'predict', 'score'))
    parser.add_argument('--output', type=Path, default=OUTPUT); parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if args.command == 'prepare': prepare(args.output)
    elif args.command == 'predict': predict(args.output, args.workers)
    else: score(args.output)
