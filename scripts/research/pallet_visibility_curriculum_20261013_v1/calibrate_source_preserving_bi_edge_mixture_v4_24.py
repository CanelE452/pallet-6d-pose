"""Separate CAL edge noise for source-preserving v24 active groups.

Exactly the immutable v20d support-balanced, uncentered cov-volume NLL and
optimizer/LOGO recipe. Only declared observation population/schema changes.
Preparation reads CAL metadata, not reference coordinates. Fit is a separate
step, to be run only after essential source/preparation review. No TEST or
remaining references, image/NN/pose fits, or tangent-gamma fitting occur here.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
import types

import numpy as np

from . import calibrate_complete_bi_edge_mixture_v4_20d as PARENT
from . import source_preserving_complete_bi_evidence_v4_24 as AUTH


C = PARENT.C
PREPARED = PARENT.PREPARED.parent / 'source_preserving_bi_evidence_v4_24_checks'
OUTPUT = PARENT.OUTPUT.parent / 'CAL_source_preserving_bi_edge_mixture_v4_24'
LABELS = PARENT.LABELS
SCHEMA = 'CAL_SOURCE_PRESERVING_COMPLETE_PARTIAL_SUPPORT_BALANCED_NMS_NOISE_V24'
PROTOCOL_SCHEMA = 'SOURCE_PRESERVING_BI_SUPPORT_BALANCED_PREFIT_PROTOCOL_V24'
RECIPE = {**deepcopy(PARENT.RECIPE),
    'observation_generation': AUTH.NOISE_OBSERVATION_FAMILY,
    'applicability': 'ONLY exact fallback-self-owner or existing partial-normal v24 active groups; all CAL46 inventory retained',
    'complete_source_policy': AUTH.SOURCE_POLICY,
    'unknown_complete_source': 'retain exact original complete 2D q/covariance/reliability',
    'replacement': 'unchanged v20d finite/topology/overlap plan, then exact source owner filtering',
    'tangent_gamma_model': 'not fitted here; root controller shares its separate encompassing model across both controls'}
conditional_covariance_floors = PARENT.conditional_covariance_floors
mixture_noise_nll_and_gradient = PARENT.mixture_noise_nll_and_gradient


# Same likelihood, frame/session/edge weighting, optimizer starts and LOGO
# exclusion algebra. No model is silently relabelled from the old population.
fit_native_edge_mixture_model = types.FunctionType(PARENT.fit_native_edge_mixture_model.__code__,
    {**PARENT.fit_native_edge_mixture_model.__globals__, 'SCHEMA': SCHEMA, 'RECIPE': RECIPE},
    'fit_source_preserving_native_edge_noise')
fit_native_edge_mixture_model.__kwdefaults__ = deepcopy(PARENT.fit_native_edge_mixture_model.__kwdefaults__)
_EXPORT = types.FunctionType(PARENT.OLD.mixture_noise_parameters.__code__,
    {**PARENT.OLD.mixture_noise_parameters.__globals__, 'SCHEMA': SCHEMA, 'RECIPE': RECIPE,
     'MIX': types.SimpleNamespace(NOISE_OBSERVATION_FAMILY=AUTH.NOISE_OBSERVATION_FAMILY)},
    'export_source_preserving_noise')


def mixture_noise_parameters(model):
    result = _EXPORT(model)
    result['noise_provenance'].update(applicability_profile=RECIPE['applicability'],
        source_policy=AUTH.SOURCE_POLICY, support_prior_policy=RECIPE['component_prior'],
        archived_complete_tangent_calibrated=False, tangent_gamma_fitted_by_this_model=False)
    return result


def _verify_preparation(output):
    protocol = PARENT.OLD._verify_preparation(output)
    if protocol['recipe'] != RECIPE or protocol['schema'] != PROTOCOL_SCHEMA:
        raise ValueError('Separate source-preserving active-population preparation required')
    return protocol


def prepare(output=OUTPUT):
    """Freeze exact active groups/support priors and CAL label bytes only."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Preserve earlier CAL preparation')
    prep = C.read(PREPARED / 'PREPARATION_SEAL.json')
    seal = C.read(PREPARED / 'CAL_PREFLIGHT_SEAL_V1.json')
    report = C.read(PREPARED / 'CAL_PREFLIGHT_V1.json')
    journal = PREPARED / 'CAL_PREPARED_GROUPS.jsonl.gz'
    if (C.sha(PREPARED / 'PREPARATION_SEAL.json') != seal['preparation_sha256']
            or C.sha(PREPARED / 'CAL_PREFLIGHT_V1.json') != seal['report_sha256']
            or C.sha(journal) != seal['prepared_groups_sha256']
            or report['error_count'] != 0 or report['successful_metadata_rows'] != 46
            or report['source_policy'] != AUTH.SOURCE_POLICY):
        raise ValueError('Successful sealed source-preserving CAL46 metadata required')
    for key in ('source_sha256', 'input_sha256', 'protected_user_sha256'):
        PARENT.OLD.E._verify_sources(prep[key])
    rows = list(C.rows(journal))
    ids = prep['ids']
    if (len(rows) != 46 or len(set(ids)) != 46 or [row['id'] for row in rows] != ids
            or any(row['partition'] != 'CAL' for row in rows)):
        raise ValueError('Exact CAL46 only; no other partition decode')
    sources = dict(prep['source_sha256'])
    ancestry = []
    for relative in prep['input_sha256']:
        if not relative.endswith('/PROTOCOL.json'):
            continue
        ancestor = C.read(C.ROOT / relative)
        for key in ('source_sha256', 'input_sha256', 'protected_user_sha256'):
            if key in ancestor:
                PARENT.OLD.E._verify_sources(ancestor[key])
        for path, digest in ancestor.get('source_sha256', {}).items():
            if path in sources and sources[path] != digest:
                raise ValueError('Frozen source ancestry conflict')
            sources[path] = digest
        ancestry.append(relative)
    for module in (sys.modules[__name__], C, PARENT, PARENT.OLD,
            PARENT.OLD.ALIGN, PARENT.OLD.FACTOR, AUTH, AUTH.OLD):
        path = Path(module.__file__).resolve()
        sources[str(path.relative_to(C.ROOT))] = C.sha(path)
    inputs = {str(path.relative_to(C.ROOT)): C.sha(path) for path in (journal,
        PREPARED / 'PREPARATION_SEAL.json', PREPARED / 'CAL_PREFLIGHT_SEAL_V1.json',
        PREPARED / 'CAL_PREFLIGHT_V1.json', LABELS)}
    for row in rows:
        bundle, plan = row['authenticated_group_bundle'], row['replacement_plan']
        if (bundle['groups'] != plan['active_groups'] or plan['source_policy'] != AUTH.SOURCE_POLICY
                or bundle['noise_observation_family'] != AUTH.NOISE_OBSERVATION_FAMILY):
            raise ValueError('Exact new runtime active population required')
        for group in bundle['groups']:
            prior = PARENT._validate_support_priors(group['components'])
            if not np.allclose(prior, group['component_weights'], atol=1e-14, rtol=1e-12):
                raise ValueError('Same exact support/fit mixture priors as runtime required')
        row.update(all_NMS_physical_edges=len(row['authentication_inventory_groups']),
            all_NMS_alternatives=sum(len(g['components']) for g in row['authentication_inventory_groups']),
            all_NMS_source_chain_alternatives=len(bundle['actual_lattice_gradient_subpixel_relation']),
            all_NMS_alternatives_are_unique_fit_variants_not_source_directed_chain_count=True)
    output.mkdir(parents=True)
    protocol = dict(schema=PROTOCOL_SCHEMA, recipe=RECIPE,
        prepared_before_CAL_reference_decode_UTC=datetime.now(timezone.utc).isoformat(), ids=ids, rows=46,
        source_sha256=sources, input_sha256=inputs, protected_user_sha256=prep['protected_user_sha256'],
        CAL_label_path=str(LABELS.relative_to(C.ROOT)), other_partition_prediction_rows_decoded=0,
        TEST_or_remaining_label_paths=False, new_pose_fits=0, new_RGB_or_model_forwards=0,
        prior_identity_binding='source owner and support/fit identity fixed before CAL reference decode',
        source_binding_scope='conservative inherited union plus explicit new helper/fitter; not every module executes',
        ancestor_protocols=ancestry)
    C.write(output / 'PROTOCOL.json', protocol)
    for relative, digest in sources.items():
        target = output / 'code_snapshot' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(C.ROOT / relative, target)
        if C.sha(target) != digest:
            raise ValueError('Source snapshot mismatch')
    C.save_rows(output / 'CAL_AUTHENTICATED_INPUTS.jsonl.gz', rows)
    for mapping in (sources, inputs, prep['protected_user_sha256']):
        PARENT.OLD.E._verify_sources(mapping)
    C.write(output / 'PREPARATION_SEAL.json', dict(protocol_sha256=C.sha(output / 'PROTOCOL.json'),
        CAL_inputs_sha256=C.sha(output / 'CAL_AUTHENTICATED_INPUTS.jsonl.gz'), rows=46,
        source_sha256=sources, before_CAL_reference_decode=True, no_TEST_or_remaining_label_decode=True,
        runtime_applicable_group_count=sum(len(r['authenticated_group_bundle']['groups']) for r in rows),
        support_and_fit_variant_priors_fixed_before_labels=True, source_policy=AUTH.SOURCE_POLICY))
    print(dict(prepared_CAL=46, active_groups=sum(len(r['authenticated_group_bundle']['groups']) for r in rows),
        label_decodes=0, fit_executed=False), flush=True)


_FIT = types.FunctionType(PARENT.fit.__code__, {**PARENT.fit.__globals__,
    '_verify_preparation': _verify_preparation,
    'fit_native_edge_mixture_model': fit_native_edge_mixture_model,
    'RECIPE': RECIPE, 'SCHEMA': SCHEMA, 'AUTH': AUTH, 'OUTPUT': OUTPUT},
    'fit_source_preserving_CAL46_same_v20d_recipe')


def fit(output=OUTPUT):
    return _FIT(Path(output).resolve())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'fit'))
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    (prepare if args.mode == 'prepare' else fit)(args.output)
