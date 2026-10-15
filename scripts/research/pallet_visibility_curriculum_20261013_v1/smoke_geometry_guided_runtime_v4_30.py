"""Fixed first-CAL V30 fitted-runtime smoke and independent native check.

Only names/dependencies change in the two frozen smoke drivers. Their actual
objective/guard finite differences, continuous five-arm run and independent
native arithmetic remain unchanged. No reference, primary row, RGB or new
model fit is accessed. This is not a population accuracy evaluation.
"""
from copy import deepcopy
from pathlib import Path
from types import FunctionType, SimpleNamespace

from . import evaluate_geometry_guided_edges_v4_30 as M
from . import smoke_geometry_guided_runtime_v4_29 as RUNTIME
from . import smoke_fitted_geometry_native_audit_v4_29d as INDEPENDENT

C = M.C
OUTPUT = M.DECLARE.B / 'geometry_guided_actual_runtime_smoke_v4_30'
INDEPENDENT_OUTPUT = M.DECLARE.B / 'geometry_guided_fitted_native_independent_smoke_v4_30'
PLAN = M.DECLARE.B / 'geometry_guided_actual_runtime_smoke_plan_v4_30/PREDECLARATION.json'
PARENT_SHA256 = {RUNTIME: '63e30cd942e89890f877149676fe81e2fae360b3de681b12d08a699057ff0113',
    INDEPENDENT: '609e49bdab6f18b7667180df483c231161ead8f9710aa9c903bad43f18f861d4'}


def _clone(function, updates):
    result = FunctionType(function.__code__, {**function.__globals__, **updates},
        function.__name__, function.__defaults__)
    result.__kwdefaults__ = deepcopy(function.__kwdefaults__)
    return result


def _facade():
    prep, cal = M._modules()
    facade = SimpleNamespace(**{**vars(M.PARENT), **vars(M), 'PREP': prep, 'CAL': cal})
    return facade, prep, cal


def declare():
    if PLAN.exists():
        raise FileExistsError('Preserve each actual-model smoke declaration')
    M._verify_plan()
    _, prep, cal = _facade()
    for module, digest in PARENT_SHA256.items():
        if C.sha(module.__file__) != digest:
            raise ValueError('Exact unchanged smoke driver required')
    modelseal = C.read(cal.OUTPUT / 'MODEL_SEAL.json')
    sources = dict(modelseal['source_sha256'])
    for module in (M, RUNTIME, INDEPENDENT, prep, cal, M.G, M.AUTH):
        sources = M.PARENT._merge_frozen(sources,
            {str(Path(module.__file__).resolve().relative_to(C.ROOT)): C.sha(module.__file__)})
    sources = M.PARENT._merge_frozen(sources,
        {str(Path(__file__).resolve().relative_to(C.ROOT)): C.sha(__file__)})
    inputs = dict(modelseal['input_sha256'])
    paths = [prep.TASKS / 'INPUTS.jsonl.gz', prep.TASKS / 'INPUT_SEAL.json',
        prep.OBS / 'CAL_OBSERVATIONS.jsonl.gz', prep.OBS / 'CAL_OBSERVATION_SEAL.json',
        cal.OUTPUT / 'PREPARATION_SEAL.json', cal.OUTPUT / 'MODEL_SEAL.json', prep.DECLARATION]
    inputs = M.PARENT._merge_frozen(inputs,
        {str(p.relative_to(C.ROOT)): C.sha(p) for p in paths})
    inputs = M.PARENT._merge_frozen(inputs,
        {str((cal.OUTPUT / rel).relative_to(C.ROOT)): digest
            for rel, digest in modelseal['assets'].items()})
    protected = C.read(prep.TASKS / 'PROTOCOL.json')['protected_user_sha256']
    for mapping in (sources, inputs, protected):
        M.E._verify_sources(mapping)
    C.write(PLAN, dict(schema='FIXED_FIRST_CAL_V30_ACTUAL_MODEL_RUNTIME_SMOKE_PLAN',
        case_selection='first original CAL task and first new CAL observation; fixed ordering, no score selector',
        source_sha256=sources, input_sha256=inputs, protected_user_sha256=protected,
        selection_profile=M.G.SELECTION_PROFILE, model_seal_sha256=C.sha(cal.OUTPUT / 'MODEL_SEAL.json'),
        objective_guard_finite_differences=True, continuous_five_arm_run=True,
        independent_native_cost_and_support_after_run=True,
        reference_RGB_primary_rows_parameter_search_or_model_refits=0,
        population_accuracy_or_default_adoption_claim=False))
    C.write(PLAN.parent / 'PLAN_SEAL.json', dict(plan_sha256=C.sha(PLAN)))


def _verify():
    plan = C.read(PLAN)
    if C.sha(PLAN) != C.read(PLAN.parent / 'PLAN_SEAL.json')['plan_sha256']:
        raise ValueError('Exact actual-model smoke declaration required')
    for key in ('source_sha256', 'input_sha256', 'protected_user_sha256'):
        M.E._verify_sources(plan[key])
    return plan


def runtime(*, root_actual_CAL_smoke_GO=False):
    if root_actual_CAL_smoke_GO is not True:
        raise PermissionError('Root actual fixed-CAL smoke GO before IO')
    _verify()
    facade, prep, _ = _facade()
    result = _clone(RUNTIME.main, dict(M=facade, C=C, E=M.E, G=M.G, PREP=prep,
        __file__=__file__))(OUTPUT)
    _verify()
    C.write(OUTPUT / 'CALLER_CLOSURE.json', dict(plan_sha256=C.sha(PLAN),
        numerical_driver_sha256=PARENT_SHA256[RUNTIME], runtime_seal_sha256=C.sha(OUTPUT / 'SEAL.json')))
    return result


def independent(*, root_actual_CAL_smoke_GO=False):
    if root_actual_CAL_smoke_GO is not True:
        raise PermissionError('Root independent fixed-CAL smoke GO before IO')
    _verify()
    _, prep, _ = _facade()
    result = _clone(INDEPENDENT.main, dict(OBS=prep.OBS, SMOKE=OUTPUT, __file__=__file__))(
        INDEPENDENT_OUTPUT)
    _verify()
    C.write(INDEPENDENT_OUTPUT / 'CALLER_CLOSURE.json', dict(plan_sha256=C.sha(PLAN),
        numerical_driver_sha256=PARENT_SHA256[INDEPENDENT],
        independent_seal_sha256=C.sha(INDEPENDENT_OUTPUT / 'SEAL.json')))
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('declare', 'runtime', 'independent'))
    parser.add_argument('--root-actual-CAL-smoke-GO', action='store_true')
    args = parser.parse_args()
    if args.stage == 'declare':
        declare()
    else:
        dict(runtime=runtime, independent=independent)[args.stage](
            root_actual_CAL_smoke_GO=args.root_actual_CAL_smoke_GO)
