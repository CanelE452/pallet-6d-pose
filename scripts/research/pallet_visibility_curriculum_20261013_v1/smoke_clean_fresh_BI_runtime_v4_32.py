"""First ordered CAL only: positive V32 packet, fresh seeds and static costs.

The operational controller supplies the same parent packet/core construction.
This smoke never calls its solver, continuous refiner, prior/score selector or
historical evaluator. There is no accuracy claim. Metadata declaration and
actual CAL runtime have separate explicit root gates. No RGB, label or primary
row is decoded. The parent reads only the first sealed original task and its
one canonical CAL spool; raw history is never a worker input or a filter.
"""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np

from . import evaluate_clean_fresh_BI_poses_v4_32 as M

C, G, META, OBS = M.C, M.G, M.META, M.OBS
OUTPUT = M.B/'clean_fresh_BI_actual_runtime_smoke_v4_32'
PLAN = M.B/'clean_fresh_BI_actual_runtime_smoke_plan_v4_32/PREDECLARATION.json'
PROOF_PINS = {
    'clean_fresh_BI_input_projection_peer_review_v4_32/SOURCE_REVIEW_V1.json':
        '53d8e1a50b52bbd3de2b0af828de5aa71c8784c87810784a7d1f8d54dac0e9cb',
    'clean_fresh_BI_pose_solver_peer_review_v4_32/SOURCE_REVIEW_V1.json':
        'f2329f56caaf8cfb70b996c56222a5b23a9ffd38334877f5301f4de8dd644a19',
}
PROJECTOR_SHA256 = '5dee94f2d4975a50a97564f1a4f7795c8c3994f69bb80db3f7e89f9c8d2c6026'


def _expected(value, name):
    if not M._sha(value):
        raise ValueError('Explicit exact '+name+' SHA required before IO')
    return value


def _map(*maps):
    result = {}
    for mapping in maps:
        for key, value in mapping.items():
            if key in result and result[key] != value:
                raise ValueError('Frozen source/input conflict: '+key)
            result[key] = value
    return result


def _path_sha(path):
    path = Path(path).resolve()
    return {str(path.relative_to(C.ROOT)):C.sha(path)}


def _plain_digest(value):
    # Prepared O regions contain arrays. Hash the same finite representation
    # saved by C.write, without changing the numerical packet or plan.
    return G._digest(C.finite(value))


def _first(rows):
    iterator = iter(rows)
    try:
        return next(iterator)
    finally:
        if hasattr(iterator, 'close'):
            iterator.close()


def _native_samples(packet, clean, cores):
    def without_repeated_plan(value):
        return {k:v for k,v in value.items() if k not in ('v29_cost_plan','v31_cost_plan','v32_cost_plan')}

    records = {}
    for policy in G.SOURCE_POLICIES:
        records[policy] = {}
        for name in M.FAMILIES:
            core = cores[policy][name]
            bank = clean['fresh_seed_banks'][name]
            starts = M.CLEAN.select_clean_fresh_native_starts(core, clean, name)
            origin = {p['name']:p['source'] for p in bank['proposals']}
            origin['OWN_INITIAL_COARSE'] = M.FACTORY.FRESH.COARSE_SOURCE
            probes = [('INDEPENDENT_COARSE_STATIC', clean['families'][name]['coarse_pose'])]
            probes += [(start['name'], start) for start in starts['starts']]
            seed_evaluations = []
            for proposal in bank['proposals']:
                # Retain rejected seeds as well. Never compare a fresh number
                # to e or use a saved method's pose to filter this bank.
                seed_evaluations.append(dict(name=proposal['name'],origin=proposal['source'],
                    R=deepcopy(proposal['R']),native_t=deepcopy(proposal['t']),
                    evaluation=without_repeated_plan(core.evaluate(proposal['R'],proposal['t']))))
            samples = []
            for token, pose in probes:
                R, t = np.asarray(pose['R']), np.asarray(pose['t'])
                p = G.V._pose_vector(R, t, core.scale)
                value, gradient, parts = core.objective(p)
                evaluated = core.evaluate(R, t)
                if (not np.isfinite(value) or not np.isfinite(gradient).all()
                        or evaluated.get('objective') is None
                        or not np.isclose(value, evaluated['objective'], atol=1e-9, rtol=1e-10)):
                    raise ValueError('Current native objective/evaluate disagreement or nonfinite derivative')
                scalar = sum(cost['weighted_robust_cost'] for cost in parts['native_costs'])
                if not np.isclose(value, scalar, atol=1e-9, rtol=1e-10):
                    raise ValueError('Current native observation ledger must sum once')
                constraints = []
                for side in core.sides:
                    slack, jacobian = core.constraints(p,side)
                    constraints.append(dict(side=deepcopy(side), slack=np.asarray(slack).tolist(),
                                            jacobian=np.asarray(jacobian).tolist()))
                samples.append(dict(name=token, origin=(M.FACTORY.FRESH.COARSE_SOURCE
                    if token == 'INDEPENDENT_COARSE_STATIC' else origin[token]),
                    R=R.tolist(), native_t=t.tolist(), t=t.tolist(),
                    model_center_camera_xyz=(R@np.asarray(core.xyz).mean(axis=0)+t).tolist(),
                    pose_vector=p.tolist(), objective=value,
                    objective_gradient=np.asarray(gradient).tolist(), native_cost_sum=scalar,
                    parts=without_repeated_plan(parts), evaluation=without_repeated_plan(evaluated),
                    constraint_branches=constraints))
            records[policy][name] = dict(start_packet=starts, samples=samples,
                all_fresh_seed_evaluations=seed_evaluations,
                fresh_seed_bank=deepcopy(bank), current_cost_plan=deepcopy(core.plan),
                current_cost_plan_sha256=_plain_digest(core.plan),
                family=deepcopy(clean['families'][name]),pose_vector_scale=float(core.scale),
                omitted_repeated_probe_keys=['v29_cost_plan','v31_cost_plan','v32_cost_plan'],
                current_cost_plan_complete_not_pruned=True,
                digest_encoding='G32._digest(C.finite(complete plan)) before alias omission; no byteexact claim',
                native_factory_uses_full_original_BI_guard_archive=True,
                tangent_parameters=deepcopy(packet['source_contract']['tangent_parameters'][policy]))
    return records


def _worker(packet):
    if not isinstance(packet,dict) or set(packet) != M.PROJECT.WORKER_KEYS:
        raise ValueError('Exact positive seven-key worker packet required')
    if packet['identifiers']['partition'] != 'CAL':
        raise ValueError('Fixed CAL-only gate before positive coordinate payload')
    clean, cores = M.build_clean_native_cores(packet)
    if not clean['ready']:
        return dict(schema='FIRST_CAL_CLEAN_FRESH_NATIVE_STATIC_SMOKE_V32',ready=False,
            row_retained=True,reason=clean['reason'],identifiers=deepcopy(clean['identifiers']),
            clean=clean,records={},continuous_optimizer_calls=0,old_model_or_pose_fallback_used=False)
    M.SOLVER._cores(clean,cores)
    records = _native_samples(packet,clean,cores)
    M.CLEAN._validate(clean)
    return dict(schema='FIRST_CAL_CLEAN_FRESH_NATIVE_STATIC_SMOKE_V32',ready=True,
        identifiers=deepcopy(clean['identifiers']),row_retained=True,clean=clean,records=records,
        packet_sha256=_plain_digest(packet),selected_native_start_sets=4,
        source_policy_count=2,family_count=2,no_numeric_equality_to_e_filter=True,
        original_BI_guard_archive_preserved=True,continuous_optimizer_calls=0,
        no_full_four_arm_solve=True,old_model_or_pose_fallback_used=False,
        population_accuracy_or_global_uniqueness_claim=False)


def smoke_worker_packet(packet):
    """Pure same-controller construction; calls are counted without module edits."""
    old_profile = sys.getprofile()
    counts = dict(fresh_BI_task=0,fresh_PnP_bank=0,SQPNP_Generic=0,
                  continuous_refiner=0,full_solver=0,scipy_optimizer=0,legacy_evaluator_run=0,
                  legacy_refiner_wrapper=0)
    codes = {M.CLEAN.prepare_clean_fresh_BI_task.__code__:'fresh_BI_task',
        M.FACTORY.FRESH.generate_fresh_observed_BI_pnp_proposals.__code__:'fresh_PnP_bank',
        M.CLEAN.refine_clean_fresh_native_problem.__code__:'continuous_refiner',
        M.SOLVER.solve_clean_fresh_BI_poses.__code__:'full_solver',
        M.SOLVER.SCALAR._run.__code__:'legacy_evaluator_run'}
    legacy = M.SOLVER.SCALAR.PARENT.V.V._REFINER
    if hasattr(legacy,'__code__'):
        codes[legacy.__code__] = 'legacy_refiner_wrapper'

    def profile(frame,event,arg):
        if event == 'call':
            key = codes.get(frame.f_code)
            if key: counts[key] += 1
            if '/scipy/optimize/' in frame.f_code.co_filename.replace('\\','/'):
                counts['scipy_optimizer'] += 1
        elif event == 'c_call' and getattr(arg,'__name__',None) == 'solvePnPGeneric':
            counts['SQPNP_Generic'] += 1

    try:
        sys.setprofile(profile)
        result = _worker(packet)
    finally:
        sys.setprofile(old_profile)
    if any(counts[key] for key in ('continuous_refiner','full_solver','scipy_optimizer',
                                  'legacy_evaluator_run','legacy_refiner_wrapper')):
        raise ValueError('Static first-CAL smoke called a forbidden optimizer/legacy wrapper')
    result['observed_call_counts'] = counts
    result['worker_packet_keys'] = sorted(packet)
    return result


def declare(*,root_metadata_GO=False,expected_central_plan_sha256=None,
        expected_controller_sha256=None,expected_CAL_seal_sha256=None,
        expected_CAL_owner_sha256=None,expected_model_seal_sha256=None):
    if root_metadata_GO is not True:
        raise PermissionError('Root smoke metadata declaration GO before IO')
    expected = {name:_expected(value,name) for name,value in dict(
        central=expected_central_plan_sha256,controller=expected_controller_sha256,
        CALseal=expected_CAL_seal_sha256,CALowner=expected_CAL_owner_sha256,
        modelseal=expected_model_seal_sha256).items()}
    if PLAN.exists(): raise FileExistsError('Preserve each smoke declaration')
    if C.sha(M.__file__) != expected['controller'] or C.sha(M.PROJECT.__file__) != PROJECTOR_SHA256:
        raise ValueError('Exact reviewed controller/projector source required')
    central,protocol,ids,entries,immutable = META._observation_closure(expected['central'])
    binding,proof_sources = META._initial_source_closure()
    sealpath=OBS.OUTPUT/'CAL_OBSERVATION_SEAL.json';ownerpath=OBS.OUTPUT/'CAL_OWNER_COMPLETION_RECEIPT_V32_V1.json'
    modelpath=M.CAL.OUTPUT/'MODEL_SEAL.json'
    for path,key in ((sealpath,'CALseal'),(ownerpath,'CALowner'),(modelpath,'modelseal')):
        if C.sha(path) != expected[key]: raise ValueError('Explicit approved '+key+' mismatch')
    modelseal=C.read(modelpath)
    if (modelseal['observation_generation'] != G.OBSERVATION_FAMILY
            or modelseal['noise_population'] != G.POPULATION
            or modelseal['selection_profile'] != G.SELECTION_PROFILE
            or modelseal['raw_observation_generation'] != G.RAW_OBSERVATION_GENERATION
            or modelseal['input_sha256'].get(str(M.CAL.CENTRAL_PLAN.relative_to(C.ROOT))) != expected['central']):
        raise ValueError('Same central and newly fitted32 modelseal required')
    taskseal=C.read(M.TASKS/'INPUT_SEAL.json');taskprotocol=C.read(M.TASKS/'PROTOCOL.json')
    if (taskseal['ordered_ids'][:46] != ids or taskprotocol['ids'][:46] != ids
            or C.sha(M.TASKS/'INPUTS.jsonl.gz') != taskseal['input_sha256']
            or C.sha(M.TASKS/'PROTOCOL.json') != taskseal['protocol_sha256']):
        raise ValueError('First ordered CAL must bind the original task archive')
    for rel,digest in PROOF_PINS.items():
        if C.sha(M.B/rel) != digest: raise ValueError('Exact projector/solver source peer required')
    sources=_map(central['source_sha256'],protocol['source_sha256'],modelseal['source_sha256'],proof_sources,
        {str(Path(module.__file__).resolve().relative_to(C.ROOT)):C.sha(module.__file__)
            for module in (M,M.PROJECT,M.CLEAN,M.FACTORY,M.AUTH,M.G,M.NOISE,M.CAL,M.SOLVER,META,OBS)},
        _path_sha(__file__),_path_sha(Path(__file__).with_name('test_smoke_clean_fresh_BI_runtime_v4_32.py')))
    paths=[sealpath,ownerpath,modelpath,M.TASKS/'INPUTS.jsonl.gz',M.TASKS/'INPUT_SEAL.json',
        M.TASKS/'PROTOCOL.json',M.CAL.CENTRAL_PLAN,*[M.B/rel for rel in PROOF_PINS]]
    inputs=_map(central['input_sha256'],protocol['input_sha256'],modelseal['input_sha256'],immutable,
        *[_path_sha(p) for p in paths],
        {str((M.CAL.OUTPUT/rel).relative_to(C.ROOT)):digest for rel,digest in modelseal['assets'].items()})
    protected=_map(central['protected_user_sha256'],protocol['protected_user_sha256'],taskprotocol['protected_user_sha256'])
    for mapping in (sources,inputs,protected): M._verify(mapping)
    C.write(PLAN,dict(schema='FIRST_ORDERED_CAL_POSITIVE_FRESH_STATIC_RUNTIME_SMOKE_PLAN_V32',
        expected_sha256=expected,source_sha256=sources,input_sha256=inputs,protected_user_sha256=protected,
        original_CAL_file_sha256=immutable,first_CAL_id=ids[0],first_CAL_index_entry=entries[0],
        initial_source_binding=binding,case_selection='first canonical CAL by original inventory order; no errors/GT routing',
        selection_profile=G.SELECTION_PROFILE,first_task_and_one_raw_CAL_spool_only=True,
        no_full_OBSERVATIONS238_needed=True,metadata_declaration_models_or_rows_decoded=0,
        runtime_scope='same parent positive packet and reviewed worker build; fresh seeds/native starts/static objective+guards only',
        no_full_solver_or_continuous_optimizer=True,primary_reference_RGB_NN_or_score_queries=0))
    C.write(PLAN.parent/'PLAN_SEAL.json',dict(plan_sha256=C.sha(PLAN)))


def _verify(expected_central_plan_sha256):
    _expected(expected_central_plan_sha256,'central')
    plan=C.read(PLAN)
    if (C.sha(PLAN) != C.read(PLAN.parent/'PLAN_SEAL.json')['plan_sha256']
            or plan['expected_sha256']['central'] != expected_central_plan_sha256):
        raise ValueError('Exact same-central static smoke plan required')
    for key in ('source_sha256','input_sha256','protected_user_sha256','original_CAL_file_sha256'):
        M._verify(plan[key])
    return plan


def runtime(*,root_actual_CAL_smoke_GO=False,expected_central_plan_sha256=None):
    if root_actual_CAL_smoke_GO is not True:
        raise PermissionError('Separate root actual first-CAL smoke GO before IO')
    _expected(expected_central_plan_sha256,'central')
    if OUTPUT.exists(): raise FileExistsError('Preserve each smoke attempt')
    plan=_verify(expected_central_plan_sha256)
    models,_,modelseal=M._models(expected_central_plan_sha256)
    if C.sha(M.CAL.OUTPUT/'MODEL_SEAL.json') != plan['expected_sha256']['modelseal']:
        raise ValueError('Exact approved fitted modelseal required')
    entry=plan['first_CAL_index_entry']
    task=_first(C.rows(M.TASKS/'INPUTS.jsonl.gz'))
    row=OBS.OLD._read_single(OBS.OLD._row_path(OBS.OUTPUT,entry['id']),entry)
    if (task['partition'] != 'CAL' or row['partition'] != 'CAL'
            or task['id'] != plan['first_CAL_id'] or row['id'] != plan['first_CAL_id']
            or OBS.D.R._canonical_sha(row) != entry['canonical_row_sha256']):
        raise ValueError('Exact first canonical CAL raw row/task required')
    packet=M._prepare_packet(task,row,plan['initial_source_binding'],models)
    del task,row,models
    result=smoke_worker_packet(packet)
    _verify(expected_central_plan_sha256)
    OUTPUT.mkdir(parents=True)
    C.write(OUTPUT/'WORKER_PACKET.json',packet)
    C.write(OUTPUT/'RESULT.json',result)
    C.write(OUTPUT/'SEAL.json',dict(schema='FIRST_CAL_CLEAN_FRESH_STATIC_SMOKE_SEAL_V32',
        plan_sha256=C.sha(PLAN),central_plan_sha256=expected_central_plan_sha256,
        model_seal_sha256=plan['expected_sha256']['modelseal'],case=plan['first_CAL_id'],
        assets={name:C.sha(OUTPUT/name) for name in ('WORKER_PACKET.json','RESULT.json')},
        actual_primary_labels_RGB_NN_or_continuous_pose_optimization=0))
    return result


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('declare','runtime'))
    parser.add_argument('--central-plan-sha256',required=True)
    parser.add_argument('--root-metadata-GO',action='store_true')
    parser.add_argument('--root-actual-CAL-smoke-GO',action='store_true')
    for key in ('controller','CAL-seal','CAL-owner','model-seal'):
        parser.add_argument('--expected-'+key+'-sha256')
    args=parser.parse_args()
    if args.stage == 'declare':
        declare(root_metadata_GO=args.root_metadata_GO,expected_central_plan_sha256=args.central_plan_sha256,
            expected_controller_sha256=args.expected_controller_sha256,
            expected_CAL_seal_sha256=args.expected_CAL_seal_sha256,
            expected_CAL_owner_sha256=args.expected_CAL_owner_sha256,
            expected_model_seal_sha256=args.expected_model_seal_sha256)
    else:
        runtime(root_actual_CAL_smoke_GO=args.root_actual_CAL_smoke_GO,
                expected_central_plan_sha256=args.central_plan_sha256)
