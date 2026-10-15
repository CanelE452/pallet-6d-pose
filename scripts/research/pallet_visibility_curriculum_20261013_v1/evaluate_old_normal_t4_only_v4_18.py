"""Matched OFF-NMS control: old normal/T4 costs and identical v18 proposals.

The shared prepared input is copied byte for byte.  A private, read-only
function adapter disables NMS authentication and line replacement before
the objective is constructed.  No frozen module globals are mutated.  Full
BI, old vertex normals, archived T4 and every hard guard remain unchanged.
This separates the effect of NMS from discarding old window-line angles.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import sys
import types

import numpy as np

from . import common as C
from . import evaluate_image_chain_mixtures_v4_18 as ON


OUTPUT = ON.OUTPUT.parent/'old_normal_t4_only_candidates_v4_18'
ARMS = ('SAME_FROZEN_E','OLD_NORMAL_T4_NATIVE','OLD_NORMAL_T4_PLUS_MASK')


def _no_noise_parameters(ignored_model):
    """Do not inspect or parse NMS noise parameters in this OFF control."""
    return {}


def _no_image_chain_groups(base, ignored_observation, *, initial_self_hidden,
                           image_binding, **ignored_parameters):
    """No observation fields, fitted noise, group or replacement is read."""
    hidden = np.asarray(initial_self_hidden, bool)
    if hidden.shape != (8,) or list(hidden) != [s=='S' for s in base.states]:
        raise ValueError('Exact permanent initial-S gate required')
    return dict(groups=[], replaces_normal_factor_refs=[], skipped=[],
        NMS_groups_disabled=True, NMS_observation_fields_read=0, NMS_noise_parameters_read=0,
        old_partial_normals_retained_exactly=True, old_projected_edge_angles_used=False,
        S_local_queries=0, U_point_queries=0, new_image_queries=0)


def _unchanged_old_problem(base, disabled_bundle):
    if disabled_bundle['groups'] or disabled_bundle['replaces_normal_factor_refs'] or base.edges:
        raise ValueError('OFF control cannot replace old normals or add projected-edge angles')
    return base


# Each adapter owns a private globals dictionary.  No production module or
# process-wide function attribute is replaced, even transiently.
_OFF_CAL = types.SimpleNamespace(mixture_noise_parameters=_no_noise_parameters)
_OFF_MIX = types.SimpleNamespace(authenticate_image_chain_groups=_no_image_chain_groups,
    ImageChainMixtureProblem=_unchanged_old_problem)
_OFF_WORKER = types.FunctionType(ON._run.__code__,
    {**ON._run.__globals__, 'ARMS':ARMS, 'CAL':_OFF_CAL, 'MIX':_OFF_MIX}, ON._run.__name__)


def _run(task, ignored_model=None):
    row = _OFF_WORKER(task, None)
    audits = row.pop('image_chain_mixture_audit')
    for arm in ARMS[1:]:
        out = row['arms'][arm]
        disabled = out.pop('new_image_line_group_authentication')
        if any(b['groups'] or b['replaces_normal_factor_refs'] for b in disabled.values()):
            raise ValueError('No NMS group may enter the OFF objective')
        solution = out['solution']
        if solution.get('projected_edge_factor_audit') or solution.get('image_chain_mixture_audit'):
            raise ValueError('Neither old projected-edge angles nor NMS likelihoods are allowed')
        out['old_normal_T4_only_audit'] = dict(disabled_NMS_by_family=disabled,
            old_vertex_normal_factors_preserved=True, archived_tangent_variance_inflation=4.,
            original_complete_BI_covariance_reliability_unchanged=True,
            old_projected_native_edge_angle_factors_used=False,
            NMS_observations_used_in_objective=False, NMS_noise_model_used_in_objective=False)
        audit = audits[arm]
        audit.pop('old_partial_normal_refs_replaced_once', None)
        audit.update(old_partial_normal_factors_replaced=0, old_partial_normal_factors_retained=True,
            projected_native_edge_angle_factors=0, NMS_mixture_factors=0,
            NMS_noise_model_used_in_objective=False, exact_shared_v18_candidate_bank=True)
    row['schema'] = 'OLD_NORMAL_T4_ONLY_MATCHED_V18_CACHED_PREDICTION'
    row['old_normal_T4_only_audit'] = audits
    row['contract'].pop('all_new_NMS_alternatives_retained', None)
    row['contract'].update(NMS_groups_disabled=True, old_partial_normals_preserved=True,
        old_projected_native_edge_angles_used=False, NMS_observations_used_in_objective=False,
        NMS_noise_model_used_in_objective=False, no_new_full_2D_KPs=True,
        unchanged_shared_input_contains_NMS_archive_for_matching_only=True)
    return row


def _parent_inputs():
    protocol = C.read(ON.OUTPUT/'PROTOCOL.json')
    seal = C.read(ON.OUTPUT/'INPUT_SEAL.json')
    predictions = C.read(ON.OUTPUT/'PREDICTION_SEAL.json')
    if (seal['rows'] != 238 or predictions['rows'] != 238
            or C.sha(ON.OUTPUT/'INPUTS.jsonl.gz') != seal['input_sha256']
            or C.sha(ON.OUTPUT/'PROTOCOL.json') != seal['protocol_sha256']
            or C.sha(ON.OUTPUT/'PROTOCOL.json') != predictions['protocol_sha256']
            or C.sha(ON.OUTPUT/'INPUT_SEAL.json') != predictions['input_seal_sha256']
            or C.sha(ON.OUTPUT/'PREDICTIONS.jsonl.gz') != predictions['predictions_sha256']):
        raise ValueError('Complete immutable v18 prepared238 input and prediction seals required')
    for key in ('source_sha256','input_sha256','protected_user_sha256'):
        ON.E._verify_sources(protocol[key])
    return protocol, seal


def prepare(output=OUTPUT):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Preserve every previous control preparation')
    parent, parent_seal = _parent_inputs()
    sources = dict(parent['source_sha256']); inputs = dict(parent['input_sha256'])
    for module in (sys.modules[__name__], ON):
        path = Path(module.__file__).resolve(); sources[str(path.relative_to(C.ROOT))] = C.sha(path)
    for filename in ('PROTOCOL.json','INPUT_SEAL.json','INPUTS.jsonl.gz','PREDICTION_SEAL.json'):
        path = ON.OUTPUT/filename; inputs[str(path.relative_to(C.ROOT))] = C.sha(path)
    for values in (sources, inputs, parent['protected_user_sha256']):
        ON.E._verify_sources(values)
    output.mkdir(parents=True)
    protocol = dict(schema='OLD_NORMAL_T4_ONLY_MATCHED_V18_ALL238_PROTOCOL',
        ids=parent['ids'], arms=list(ARMS), population=deepcopy(parent['population']),
        source_sha256=sources, input_sha256=inputs,
        protected_user_sha256=parent['protected_user_sha256'],
        parent_input_seal=ON.E._binding(ON.OUTPUT/'INPUT_SEAL.json'),
        shared_v18_input_sha256=parent_seal['input_sha256'], prepared_inputs_byte_identical_to_v18=True,
        comparison='NMS off versus frozen v18 on, with identical proposals, old guards and selectors',
        controls='old-normal/T4 native sum with mask numeric ties; same native plus frozen old mask weight',
        NMS_observations_used_in_objective=False, NMS_noise_model_used_in_objective=False,
        inherited_NMS_files_bound_for_shared_input_provenance_only=True,
        all_old_partial_normals_retained=True, projected_native_edge_angle_factors=False,
        full_BI_covariance_reliability_and_all_T4_hard_guards_preserved=True,
        proposal_bank='all frozen v18 physical proposals, both explicit WD families',
        global_module_monkeypatch=False, new_pose_RGB_or_NN_fits=0,
        S_model_projection_only=True, U_regions_only=True, no_new_full_KPs=True,
        all238_predictions_sealed_before_new_control_metrics=True, no_default_adoption=True,
        reused_TEST_and_remaining_exploratory=True,
        additional143_reference_not_independent_metrology=True)
    C.write(output/'PROTOCOL.json', protocol)
    shutil.copyfile(ON.OUTPUT/'INPUTS.jsonl.gz', output/'INPUTS.jsonl.gz')
    if C.sha(output/'INPUTS.jsonl.gz') != parent_seal['input_sha256']:
        raise ValueError('Exact byte-identical v18 input copy required')
    for relative,digest in sources.items():
        target = output/'code_snapshot'/relative; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(C.ROOT/relative,target)
        if C.sha(target) != digest:
            raise ValueError('Source snapshot mismatch')
    for values in (sources,inputs,parent['protected_user_sha256']):
        ON.E._verify_sources(values)
    C.write(output/'INPUT_SEAL.json', dict(rows=238, ordered_ids=protocol['ids'],
        input_sha256=C.sha(output/'INPUTS.jsonl.gz'), protocol_sha256=C.sha(output/'PROTOCOL.json'),
        prepared_inputs_byte_identical_to_v18=True, before_new_predictions_and_metrics=True))
    print(dict(prepared=238, inputs_byte_identical_to_v18=True, NMS_objective=False),flush=True)


def predict(output=OUTPUT,workers=8):
    runner = types.FunctionType(ON.predict.__code__,
        {**ON.predict.__globals__, '_run':_run, '_verified_model':lambda:None}, ON.predict.__name__)
    return runner(Path(output),workers)


def score(output=OUTPUT):
    scorer = types.FunctionType(ON.BASE.ARCHIVE.score.__code__,
        {**ON.BASE.ARCHIVE.score.__globals__,'ARMS':ARMS}, ON.BASE.ARCHIVE.score.__name__)
    return scorer(Path(output))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','predict','score'))
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    if args.command=='prepare':prepare(args.output)
    elif args.command=='predict':predict(args.output,args.workers)
    else:score(args.output)
