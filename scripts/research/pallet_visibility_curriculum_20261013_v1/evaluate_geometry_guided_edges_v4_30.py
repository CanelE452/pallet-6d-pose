"""Uniform V30 association selection with frozen V29 pose solver arithmetic.

The legacy task/model dictionary keys named ``v29`` are private wire-format
aliases. Every external method, observation population, noise recipe and
model export is separately V30 and bound to its selection profile.
"""
from copy import deepcopy
import gzip
import importlib
import json
from itertools import zip_longest
from pathlib import Path
import shutil
import sys
from types import FunctionType, SimpleNamespace
from collections import Counter

from . import evaluate_geometry_guided_edges_v4_29 as PARENT
from . import geometry_guided_edge_problem_v4_30 as G
from . import geometry_guided_edge_authentication_v4_30 as AUTH
from . import declare_observed_bi_associated_edges_v4_30 as DECLARE

C, E = PARENT.C, PARENT.E
ARMS = DECLARE.ARMS
POLICY_ARMS = dict(zip(G.SOURCE_POLICIES, (ARMS[1:3], ARMS[3:5])))
PLAN = DECLARE.PLAN
DECLARATION_SHA256 = '1598f5ac0a4308ce758f792656f05bfba7f351adcfd2fc1b4000f564a33d22b1'
OUTPUT = DECLARE.B / 'geometry_guided_edge_eval_v4_30'
OBSERVATION_PROTOCOL = 'FINAL_PROTOCOL.json'


def _modules():
    prefix = __package__ + '.'
    return (importlib.import_module(prefix + 'prepare_geometry_guided_native_evidence_v4_30'),
        importlib.import_module(prefix + 'calibrate_geometry_guided_edge_noise_v4_30'))


def _clone(function, updates):
    result = FunctionType(function.__code__, {**function.__globals__, **updates},
        function.__name__, function.__defaults__)
    result.__kwdefaults__ = deepcopy(function.__kwdefaults__)
    return result


def _verify_plan():
    if C.sha(PLAN) != DECLARATION_SHA256:
        raise ValueError('Frozen V30 central method declaration required')
    plan = C.read(PLAN)
    if plan['arms'] != list(ARMS) or plan['selection_profile'] != G.SELECTION_PROFILE:
        raise ValueError('Uniform declared V30 arms/profile required')
    for key in ('source_sha256', 'input_sha256', 'protected_user_sha256'):
        E._verify_sources(plan[key])
    return plan


def _save_rows(path, values):
    """Same JSON values, new storage compression1 instead of parent5."""
    path = Path(path)
    if path.exists():
        raise FileExistsError('Preserve every V30 journal attempt')
    with gzip.open(path, 'wt', encoding='utf-8', compresslevel=1) as stream:
        for value in values:
            stream.write(json.dumps(C.finite(value), ensure_ascii=False,
                separators=(',', ':'), allow_nan=False) + '\n')


def _models():
    _, cal = _modules()
    return _clone(PARENT._models, dict(CAL=cal, G=G))()


def _problem(task, family, models, policy):
    prep, _ = _modules()
    base = prep.native_baseline(task, family)
    parameters = models['V29'][policy]  # Parent internal key; the contained models are V30.
    if parameters['available']:
        noise = parameters['noise']
        bundle = AUTH.authenticate_geometry_guided_edges(base, task['v29_observation'],
            image_binding=dict(sha256=task['image_sha256'], image_hw=task['spec']['image_hw']),
            initial_geometry=task['v29_initial_geometry'], expected_source_policy=policy, **noise)
        gamma = parameters['tangent']['gamma']
    else:
        bundle = dict(observation_family=G.OBSERVATION_FAMILY, population=G.POPULATION,
            selection_profile=G.SELECTION_PROFILE,
            association_configuration=task['v29_observation']['association_configuration'],
            groups=[], ordered_corner_pairs=[], noise_unavailable_abstention=True,
            old_model_substituted=False)
        gamma = 0.
    problem = G.GeometryGuidedEdgeProblem(base, bundle, source_policy=policy, tangent_gamma=gamma)
    problem.reference_start_problem = PARENT.OLD.V.PARENT._problem(task, family, models['NMS']['OLD'])[0]
    problem.plan['CAL_noise_binding'] = deepcopy(parameters)
    return problem, bundle


def _readout(task, selected, pose):
    result = PARENT._readout(task, selected, pose)
    result.update(v30_selection_profile=G.SELECTION_PROFILE,
        v30_new_owner_ids=result['v29_new_owner_ids'])
    return result


def _run(task, models):
    run = _clone(PARENT._run, dict(ARMS=ARMS, POLICY_ARMS=POLICY_ARMS,
        _problem=_problem, _readout=_readout))
    result = run(task, models)
    result.update(schema='UNIFORM_OBSERVED_BI_ASSOCIATED_EDGE_PREDICTION_V30',
        selection_profile=G.SELECTION_PROFILE, noise_observation_family=G.OBSERVATION_FAMILY,
        raw_observation_generation=G.RAW_OBSERVATION_GENERATION)
    return result


def _full_observation_metadata(folder, ordered_ids):
    """The immutable CAL protocol and new all238 closure have distinct roles."""
    folder = Path(folder)
    protocol = C.read(folder / OBSERVATION_PROTOCOL)
    seal = C.read(folder / 'OBSERVATION_SEAL.json')
    if (len(ordered_ids) != 238 or len(set(ordered_ids)) != 238
            or protocol['ids'] != ordered_ids
            or protocol['selection_profile'] != G.SELECTION_PROFILE
            or protocol['raw_observation_generation'] != G.RAW_OBSERVATION_GENERATION
            or seal['rows'] != 238 or seal['ordered_ids'] != ordered_ids
            or seal['protocol_sha256'] != C.sha(folder / OBSERVATION_PROTOCOL)):
        raise ValueError('Separate complete V30 all238 protocol/selection seal required')
    for name, count, ids in (('CAL', 46, ordered_ids[:46]), ('REST', 192, ordered_ids[46:])):
        path = folder / (name + '_OBSERVATION_SEAL.json')
        partition = C.read(path)
        if (partition['rows'] != count or partition['ordered_ids'] != ids
                or seal['partition_seals'][name] != C.sha(path)):
            raise ValueError('Exact CAL46 + REST192 observation closure required')
    return protocol, seal


def prepare(output=OUTPUT):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Preserve every V30 preparation')
    declaration = _verify_plan()
    prep, cal = _modules()
    _models()
    parent = C.read(prep.TASKS / 'PROTOCOL.json')
    taskseal = C.read(prep.TASKS / 'INPUT_SEAL.json')
    obsprotocol, obsseal = _full_observation_metadata(prep.OBS, parent['ids'])
    if (taskseal['rows'] != 238 or obsseal['rows'] != 238
            or obsseal['ordered_ids'] != parent['ids']
            or C.sha(prep.TASKS / 'INPUTS.jsonl.gz') != taskseal['input_sha256']
            or C.sha(prep.OBS / 'OBSERVATIONS.jsonl.gz') != obsseal['observations_sha256']
            or C.sha(prep.OBS / OBSERVATION_PROTOCOL) != obsseal['protocol_sha256']
            or obsprotocol['selection_profile'] != G.SELECTION_PROFILE
            or obsprotocol['protected_user_sha256'] != parent['protected_user_sha256']):
        raise ValueError('Complete uniform V30 observations and original all238 physics required')
    modelseal = C.read(cal.OUTPUT / 'MODEL_SEAL.json')
    sources, inputs = {}, {}
    for ancestor in (declaration, parent, obsprotocol, modelseal):
        for key, mapping in (('source_sha256', sources), ('input_sha256', inputs)):
            E._verify_sources(ancestor[key])
            mapping.update(PARENT._merge_frozen(mapping, ancestor[key]))
    for module in (sys.modules[__name__], PARENT, G, AUTH, G.PARENT,
            AUTH.PARENT, DECLARE, prep, cal, cal.M):
        path = Path(module.__file__).resolve()
        sources[str(path.relative_to(C.ROOT))] = C.sha(path)
    paths = [PLAN, prep.TASKS / 'PROTOCOL.json', prep.TASKS / 'INPUT_SEAL.json',
        prep.TASKS / 'INPUTS.jsonl.gz', prep.OBS / OBSERVATION_PROTOCOL,
        prep.OBS / 'PREPARATION_SEAL.json', prep.OBS / 'OBSERVATION_SEAL.json',
        prep.OBS / 'OBSERVATIONS.jsonl.gz', cal.OUTPUT / 'MODEL_SEAL.json',
        cal.OUTPUT / 'PREPARATION_SEAL.json']
    inputs.update({str(p.relative_to(C.ROOT)): C.sha(p) for p in paths})
    inputs.update({str((cal.OUTPUT / rel).relative_to(C.ROOT)): digest
        for rel, digest in modelseal['assets'].items()})
    for mapping in (sources, inputs, parent['protected_user_sha256']):
        E._verify_sources(mapping)
    output.mkdir(parents=True)
    ids, counts = [], Counter()
    def joined():
        for task, row in zip_longest(C.rows(prep.TASKS / 'INPUTS.jsonl.gz'),
                C.rows(prep.OBS / 'OBSERVATIONS.jsonl.gz')):
            if task is None or row is None:
                raise ValueError('All238 original ordered rows required')
            prep.validate_runtime_join(task, row)
            observation = row['observation']
            if (observation.get('selection_profile') != G.SELECTION_PROFILE
                    or observation['observation_generation'] != G.RAW_OBSERVATION_GENERATION):
                raise ValueError('Raw V29 evidence plus new V30 selection required on every row')
            task['v29_observation'] = observation
            task['v29_initial_geometry'] = row['initial_geometry']
            task['v30_selection_profile'] = G.SELECTION_PROFILE
            ids.append(task['id']); counts[task['partition']] += 1
            yield task
    _save_rows(output / 'INPUTS.jsonl.gz', joined())
    if ids != parent['ids'] or counts != Counter(CAL=46, TEST=49, REMAINDER=143):
        raise ValueError('No frame, zero-support case or primary family may be filtered')
    protocol = {**parent, 'schema': 'UNIFORM_OBSERVED_BI_ASSOCIATED_EDGE_ALL238_PROTOCOL_V30',
        'arms': list(ARMS), 'source_sha256': sources, 'input_sha256': inputs,
        'declaration': E._binding(PLAN), 'selection_profile': G.SELECTION_PROFILE,
        'raw_observation_generation': G.RAW_OBSERVATION_GENERATION,
        'noise_observation_family': G.OBSERVATION_FAMILY,
        'new_noise_policies': list(G.SOURCE_POLICIES),
        'runtime_objective': 'unchanged V29 factor arithmetic + bounded O/S + optional one-sided C1',
        'original_legacy_v29_dictionary_keys_are_internal_aliases_only': True,
        'old_V29_fitted_noise_relabelled': False, 'old_e_numeric_pose_added_as_prior_or_fallback': False,
        'full_pose_TP_topnormal_prior_weights': 0., 'no_default_adoption': True,
        'storage_compression_level': 1, 'storage_parent_compression_level': 5,
        'all5arms_all238_seal_before_reference_reads': True}
    C.write(output / 'PROTOCOL.json', protocol)
    for relative, digest in sources.items():
        target = output / 'code_snapshot' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(C.ROOT / relative, target)
        if C.sha(target) != digest:
            raise ValueError('V30 source snapshot differs')
    for mapping in (sources, inputs, parent['protected_user_sha256']):
        E._verify_sources(mapping)
    C.write(output / 'INPUT_SEAL.json', dict(rows=238, ordered_ids=ids,
        protocol_sha256=C.sha(output / 'PROTOCOL.json'),
        input_sha256=C.sha(output / 'INPUTS.jsonl.gz'), before_new_predictions_or_primary_metrics=True))


def predict(output=OUTPUT, workers=6):
    _verify_plan()
    protocol = C.read(Path(output) / 'PROTOCOL.json')
    if protocol['selection_profile'] != G.SELECTION_PROFILE:
        raise ValueError('V30 full238 selected population required')
    storage = SimpleNamespace(**{**vars(C), 'save_rows': _save_rows})
    return _clone(PARENT.predict, dict(ARMS=ARMS, _models=_models, _run=_run, C=storage))(
        output, workers=workers)


def score(output=OUTPUT):
    _verify_plan()
    return _clone(PARENT.score, dict(ARMS=ARMS))(output)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'predict', 'score'))
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    if args.stage == 'predict':
        predict(args.output, args.workers)
    else:
        dict(prepare=prepare, score=score)[args.stage](args.output)
