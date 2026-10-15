"""Metadata-only V32 central declaration; no rows, labels, RGB or fitting.

The caller supplies final full-solver bytes and independent solver/operator
peer bindings. An explicit root GO precedes every IO. The declaration records
initial-only observation selection and positive fresh-PnP construction, not a
negative label filter over the old e-contaminated contender bank. Only the
parent output writer may add the separately sealed e comparison row.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from . import common as C
from . import geometry_guided_edge_problem_v4_32 as G
from . import geometry_guided_edge_authentication_v4_32 as AUTH
from . import geometry_guided_edge_noise_calibration_v4_32 as NOISE
from . import initial_only_observed_ray_association_v4_32 as ADMISSION
from . import initial_only_observed_ray_observation_v4_32 as ROW
from . import rerank_initial_only_observed_ray_population_v4_32 as OPERATOR
from . import fresh_observed_BI_pnp_proposals_v4_32 as PNP
from . import clean_fresh_BI_runtime_v4_32 as CLEAN
from . import prepare_geometry_guided_native_evidence_v4_32 as FACTORY
from . import solve_clean_fresh_BI_poses_v4_32 as SOLVER
from . import run_initial_only_observed_ray_CAL_v4_32 as CAL3_OPERATOR
from . import calibrate_coarse_visibility_prior_v4_20 as SCAL
from . import calibrate_bounded_o_localization_v4_21 as OCAL


B = OPERATOR.B
OUTPUT = B / 'initial_only_observed_ray_method_plan_v4_32'
PLAN = OUTPUT / 'PREDECLARATION.json'
ARMS = ('SAME_FROZEN_E', *SOLVER.ARMS)
RAW = OPERATOR.OLD.OUTPUT
CAL3 = B / 'initial_only_observed_ray_CAL_cases_v4_32/RERANK_SEAL.json'
CAL3_SHA256 = 'eac238f222a469f0d1eee3263b359dbd4eb09292a91781afc0b5de34644b6fb0'
CAL3_AUDIT = B / 'initial_only_observed_ray_CAL_actual_checks_v4_32/AUDIT_V1.json'
CAL3_AUDIT_SHA256 = 'bd8ea752ede9bd0fb1d779398a2253b6a9862fe35856139704f5fa428fcc9c8f'
FIXED_CAL3_IDS = ('wood_184309:000925', 'wood_184309:000804', 'eval_cad:1778653017736058368')
PARTITIONS = dict(CAL=46, TEST=49, REMAINDER=143, PRIMARY=192, severe=0)
PEERS = (
    ('initial_only_observed_ray_association_peer_review_v4_32/SOURCE_REVIEW_V1.json',
     'fc5ed589894fa89312b1924e9c86b4e38b45c0e672e6543bc6716798f60383ce', (ADMISSION,)),
    ('initial_only_observed_ray_observation_peer_review_v4_32/SOURCE_REVIEW_V1.json',
     '5f7aeb7717addfe0749b3c18ba2e3f213db98ba7dea4d170b34f904fdd7c163b', (ROW,)),
    ('geometry_guided_edges_peer_review_v4_32/PROBLEM_AUTH_NOISE_SOURCE_REVIEW_V1.json',
     '2f617569daeb39e6cbee1a5a8f34866aaa31c832dd714a15585261bc890cef06', (G, AUTH, NOISE)),
    ('fresh_observed_BI_pnp_proposals_peer_review_v4_32/SOURCE_REVIEW_V1.json',
     'c26673caa9b3d1df2d6022aacd4d46a5c4c9dec309374ef3b9262da28d522004', (PNP,)),
    ('clean_fresh_BI_runtime_peer_review_v4_32/SOURCE_REVIEW_V1.json',
     '2b71e475f983b069342ff259778ca487b970f08815d67b948477fd3f8ef3af3e', (CLEAN,)),
    ('positive_native_evidence_factory_peer_review_v4_32/SOURCE_REVIEW_V1.json',
     'd007551cf7b26dd9a3939600972c5cb842aa4fdbef4113dc4e4eae7ece938b10', (FACTORY,)),
    ('initial_only_observed_ray_CAL_operator_peer_review_v4_32/SOURCE_REVIEW_V1.json',
     'c4f31946b03ee85a32d127b5710ce057b20dc694b5bf7290f2d8686ac734b0b0', (CAL3_OPERATOR,)),
)
TEST_NAMES = ('test_initial_only_observed_ray_association_v4_32.py',
    'test_initial_only_observed_ray_observation_v4_32.py',
    'test_geometry_guided_edge_problem_v4_32.py', 'test_geometry_guided_edge_authentication_v4_32.py',
    'test_geometry_guided_edge_noise_calibration_v4_32.py',
    'test_fresh_observed_BI_pnp_proposals_v4_32.py', 'test_clean_fresh_BI_runtime_v4_32.py',
    'test_prepare_geometry_guided_native_evidence_v4_32.py',
    'test_run_initial_only_observed_ray_CAL_v4_32.py',
    'test_solve_clean_fresh_BI_poses_v4_32.py', 'test_rerank_initial_only_observed_ray_population_v4_32.py',
    'test_declare_initial_only_observed_ray_v4_32.py')


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _binding(path):
    path = Path(path).resolve()
    return dict(path=str(path.relative_to(C.ROOT)), sha256=C.sha(path))


def _verify(mapping):
    for relative, expected in mapping.items():
        if not _sha(expected) or C.sha(C.ROOT / relative) != expected:
            raise ValueError('Frozen source/input/protected bytes changed: ' + relative)


def _merge(*maps):
    return OPERATOR._union(*maps)


def _source_bindings(value):
    """Collect explicit review source maps/bindings, without opening targets."""
    result = {}
    if isinstance(value, dict):
        if isinstance(value.get('path'), str) and _sha(value.get('sha256')):
            result[value['path']] = value['sha256']
        for key, item in value.items():
            if key == 'source_sha256' and isinstance(item, dict):
                result = _merge(result, {p: digest for p, digest in item.items()
                    if isinstance(p, str) and _sha(digest)})
            result = _merge(result, _source_bindings(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            result = _merge(result, _source_bindings(item))
    return result


def _peer(binding, modules):
    if (not isinstance(binding, dict) or set(binding) != {'path', 'sha256'}
            or not isinstance(binding['path'], str) or not _sha(binding['sha256'])):
        raise ValueError('Explicit final independent peer path/SHA binding required')
    path = C.ROOT / binding['path']
    if C.sha(path) != binding['sha256']:
        raise ValueError('Independent peer bytes changed before metadata decode')
    review = C.read(path)
    verdict = review.get('status', review.get('verdict',
        review.get('essential_source_review', review.get('essential_review'))))
    if (not isinstance(verdict, str)
            or not (verdict.startswith('PASS') or verdict.startswith('ESSENTIAL_PASS'))
            or any(bool(review.get(k, 0)) for k in ('blockers', 'essential_blockers', 'failure_count', 'failures', 'material_findings'))):
        raise ValueError('Independent essential PASS without blockers/failures required')
    sources = _source_bindings(review)
    for module in modules:
        target = Path(module.__file__).resolve()
        if sources.get(str(target.relative_to(C.ROOT))) != C.sha(target):
            raise ValueError('Peer must bind the exact current critical source: ' + target.name)
    return path


def _contracts():
    if (len(ARMS) != 5 or len(set(ARMS)) != 5 or ARMS[0] != 'SAME_FROZEN_E'
            or tuple(ARMS[1:]) != tuple(SOLVER.ARMS)
            or set(SOLVER.POLICY_ARMS) != set(G.SOURCE_POLICIES)
            or list(SOLVER.POLICY_ARMS.values()) != [tuple(ARMS[1:3]), tuple(ARMS[3:5])]):
        raise ValueError('Exact comparison e plus four uniform policy/native-C solver arms required')
    if (G.OBSERVATION_FAMILY != NOISE.OBSERVATION_FAMILY or G.POPULATION != NOISE.POPULATION
            or G.SELECTION_PROFILE != ADMISSION.SELECTION_PROFILE
            or G.SELECTION_PROFILE != NOISE.SELECTION_PROFILE or G.SELECTION_PROFILE != ROW.SELECTION_PROFILE
            or G.SELECTION_PROFILE != OPERATOR.SELECTION_PROFILE
            or G.RAW_OBSERVATION_GENERATION != ROW.RAW_OBSERVATION_GENERATION
            or G.RAW_OBSERVATION_GENERATION != NOISE.RAW_OBSERVATION_GENERATION):
        raise ValueError('One typed32 row/core/noise/profile and unchanged raw generation required')


def _critical_bindings(expected_fullsolver_sha256, fullsolver_peer_binding, operator_peer_binding):
    if not _sha(expected_fullsolver_sha256) or C.sha(SOLVER.__file__) != expected_fullsolver_sha256:
        raise ValueError('Caller must pin final independently reviewed fullsolver bytes')
    peer_paths = []
    for relative, digest, modules in PEERS:
        peer_paths.append(_peer(dict(path=str((B / relative).relative_to(C.ROOT)), sha256=digest), modules))
    peer_paths.append(_peer(fullsolver_peer_binding, (SOLVER,)))
    peer_paths.append(_peer(operator_peer_binding, (OPERATOR,)))
    sources = OPERATOR._sources()
    modules = (C, G, AUTH, NOISE, PNP, CLEAN, FACTORY, SOLVER, CAL3_OPERATOR, SCAL, OCAL,
        G.BASELINE, G.N, G.J, G.EDGE, SOLVER.BOUNDED, SOLVER.CONTAIN, SOLVER.SCALAR,
        CLEAN.REFINE, CLEAN.WD)
    paths = [Path(m.__file__).resolve() for m in modules]
    paths += [Path(__file__).resolve(), *(Path(__file__).with_name(n) for n in TEST_NAMES)]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError('Critical source/test missing: ' + path.name)
        sources = _merge(sources, {str(path.relative_to(C.ROOT)): C.sha(path)})
    return sources, peer_paths


def _raw_parent_metadata():
    closure = OPERATOR.OP.verify_operational_closure()
    parent, seal = C.read(RAW / 'PROTOCOL.json'), C.read(RAW / 'OBSERVATION_SEAL.json')
    ids = parent['ids']
    if (len(ids) != 238 or len(set(ids)) != 238 or ids != closure['ordered_ids']
            or seal['rows'] != 238 or seal['ordered_ids'] != ids
            or seal['observations_sha256'] != C.sha(RAW / 'OBSERVATIONS.jsonl.gz')
            or seal['protocol_sha256'] != C.sha(RAW / 'PROTOCOL.json')
            or parent['protected_user_sha256'] != closure['protected_user_sha256']
            or len(parent['protected_user_sha256']) != 6):
        raise ValueError('Complete exact raw238 and original protected6 required')
    paths = [RAW / name for name in ('PROTOCOL.json', 'PREPARATION_SEAL.json',
        'OBSERVATION_SEAL.json', 'OBSERVATIONS.jsonl.gz')]
    indexed = []
    for part, count, selected in (('CAL', 46, ids[:46]), ('REST', 192, ids[46:])):
        indexpath, sealpath = RAW / (part + '_ROW_INDEX.json'), RAW / (part + '_OBSERVATION_SEAL.json')
        index, partseal = C.read(indexpath)['entries'], C.read(sealpath)
        if (len(index) != count or [e['id'] for e in index] != selected
                or partseal['rows'] != count or partseal['ordered_ids'] != selected
                or partseal['row_index_sha256'] != C.sha(indexpath)
                or seal['partition_seals'][part] != C.sha(sealpath)):
            raise ValueError('Raw CAL46+REST192 order and partition seals required')
        paths += [indexpath, sealpath]
        indexed += index
    if Counter(entry['partition'] for entry in indexed) != Counter(CAL=46, TEST=49, REMAINDER=143):
        raise ValueError('Exact CAL46/TEST49/REMAINDER143 without dropped/severe rows required')
    return parent, closure, paths


def _prior_cal_metadata():
    if C.sha(CAL3) != CAL3_SHA256 or C.sha(CAL3_AUDIT) != CAL3_AUDIT_SHA256:
        raise ValueError('Fixed initial-only CAL3 rerank and independent333 audit required')
    seal, audit = C.read(CAL3), C.read(CAL3_AUDIT)
    calplan = CAL3.parent.parent / 'initial_only_observed_ray_CAL_plan_v4_32/PREDECLARATION.json'
    planseal = calplan.parent / 'PLAN_SEAL.json'
    if (seal['rows'] != 3 or tuple(seal['ordered_ids']) != FIXED_CAL3_IDS
            or seal['selection_profile'] != G.SELECTION_PROFILE
            or audit['checks'] != 333 or audit['failures']
            or audit['execution_seal_sha256'] != CAL3_SHA256
            or audit['plan_sha256'] != seal['plan_sha256']
            or C.sha(calplan) != seal['plan_sha256']
            or C.read(planseal)['plan_sha256'] != seal['plan_sha256']
            or seal['external_pose_guides_consumed'] != 0):
        raise ValueError('Fixed CAL3 metadata/audit contract mismatch')
    _verify(seal['source_sha256'])
    audit_source = _independent_cal_audit_source(audit, CAL3_AUDIT)
    return [CAL3, CAL3_AUDIT, audit_source, calplan, planseal,
        OPERATOR.SOURCE_CLOSURE, OPERATOR.ANCESTRY]


def _independent_cal_audit_source(audit, audit_path):
    """The standalone333 audit records one source digest, not a source map."""
    source = Path(audit_path).with_name('audit_selected_query_equations_v1.py')
    if not _sha(audit.get('source_sha256')) or C.sha(source) != audit['source_sha256']:
        raise ValueError('Independent333 exact standalone audit source required')
    return source


def _prior_model_paths():
    """Bind existing exported prior files without decoding model values."""
    paths = [folder / name for folder in (SCAL.OUTPUT, OCAL.OUTPUT)
        for name in ('PROTOCOL.json', 'PREPARATION_SEAL.json', 'MODEL.json',
            'MODEL_SEAL.json', 'CAL_LOGO_REPORT.json')]
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError('Complete frozen O/S prior source export required')
    return paths


def declare(*, root_metadata_GO=False, expected_fullsolver_sha256=None,
        fullsolver_peer_binding=None, operator_peer_binding=None):
    if root_metadata_GO is not True:
        raise PermissionError('Explicit ROOT metadata declaration GO before any IO')
    if not _sha(expected_fullsolver_sha256) or fullsolver_peer_binding is None or operator_peer_binding is None:
        raise ValueError('Final solver source and independent solver/operator peers required before IO')
    if OUTPUT.exists():
        raise FileExistsError('Preserve every central32 declaration attempt')
    _contracts()
    sources, peers = _critical_bindings(expected_fullsolver_sha256, fullsolver_peer_binding, operator_peer_binding)
    parent, closure, raw_paths = _raw_parent_metadata()
    prior_paths = _prior_cal_metadata()
    OPERATOR._pins()
    prior_models = _prior_model_paths()
    paths = [*raw_paths, *prior_paths, *peers, *prior_models]
    sources = _merge(closure['source_sha256'], sources,
        {str(path.relative_to(C.ROOT)): C.sha(path) for path in prior_paths if path.suffix == '.py'})
    inputs = _merge(closure['input_sha256'], {str(p.relative_to(C.ROOT)): C.sha(p) for p in paths})
    protected = parent['protected_user_sha256']
    for mapping in (sources, inputs, protected):
        _verify(mapping)
    plan = dict(schema='INITIAL_ONLY_QUERY_AND_STRICT_FRESH_BI_METHOD_PLAN_V32',
        UTC=datetime.now(timezone.utc).isoformat(), arms=list(ARMS), partitions=PARTITIONS,
        observation_family=G.OBSERVATION_FAMILY, population=G.POPULATION,
        selection_profile=G.SELECTION_PROFILE, raw_observation_generation=G.RAW_OBSERVATION_GENERATION,
        ordered_ids=parent['ids'], admission_recipe=ADMISSION.RECIPE,
        independent_initial_source_closure=_binding(OPERATOR.SOURCE_CLOSURE),
        raw_query_budget_ancestry=_binding(OPERATOR.ANCESTRY),
        fixed_CAL3_rerank=_binding(CAL3), independent_CAL3_333check=_binding(CAL3_AUDIT),
        final_fullsolver=_binding(Path(SOLVER.__file__)), fullsolver_peer=fullsolver_peer_binding,
        population_operator_peer=operator_peer_binding,
        observation_selection='Uniform F32 initial17c/measuredBI queries; external pose_guides=None before graph; raw extras kept as evidence only.',
        raw_arrays_q_covariance_states_initialS_masks_and_indices_unchanged=True,
        old_raw_guide_bank_or_old_selected_graph_used_numerically=False,
        source_policies=list(G.SOURCE_POLICIES), numerical_N_O_S_C_and_hardguard_math='UNCHANGED_FROZEN_KERNELS',
        native_factory='positive named cleanfamily/physics/q/OT/U/boundary only; no opaque old task/spec or initialization bank',
        fresh_proposals='full BI set plus fixed unique3/4 subsets, SQPNPGeneric all proper/all8cheirality solutions, max127calls/family; numeric physical-pose dedup only',
        starts='new32 policy-native-core six diverse fresh starts + independent own coarse, fixed per policy/family across native/C; C also keeps same-run same-policy native endpoints',
        old_reference_start_problem_or_old_evaluator_run_used=False,
        cached_old_method_pose_prior_start_candidate_familylock_or_fallback_used=False,
        numeric_e_pose_prior_start_candidate_or_fallback_added=False,
        e_comparison='separate sealed parent-only comparison stream merged after four-arm physical worker returns; e never a numerical worker input',
        invalid_initial_coarse='retain every row as acceptedFalse/poseUnavailable; no other-method fallback',
        full_pose_TP_topnormal_prior_weights=0., frozen_bounded_O_S_regularizers_retained=True,
        optional_one_sided_containment_weight=1.,
        S_image_or_raw_2D_or_region_queries=0, S_only_model_prior_and_PnP_projection=True,
        U_precise_point_or_local_queries=0, U_only_approximate_regions=True,
        target_mask_absence_is_occlusion=False, current_other_instance_mask_consumer_connection=False,
        automatic_instance_mask_provider_scope='diagnostic only; not an input to this observation/core/solver generation',
        new_noise='Select/seal all CAL46 before labels; fixed compatible44; two policies times(line+tangent) times(full+4LOGO)=20models; fixed3starts each=60 starts; unavailable population abstains, no old model substitution.',
        CAL_inventory=46, compatible_CAL_inventory=44, LOGO_sessions=4, new_CAL_model_count=20,
        new_CAL_optimizer_starts=60, noise_recipe=NOISE.RECIPE, tangent_recipe=NOISE.TANGENT_RECIPE,
        old_V29_V30_V31_noise_transferred_or_relabelled=False,
        runtime_GT_ID_error_or_partition_routing=False, rows_or_fallbacks_or_zero_cases_filtered=False,
        full_evaluation='all238x5 prediction seal, independent fresh-provenance/selected-energy/guards audit, then fixed-reference scores; primary192 all retained',
        reused_exploratory_CAL_and_TEST_not_new_holdout=True,
        default='SAME_FROZEN_E until all goal criteria independently verified',
        source_sha256=sources, input_sha256=inputs, protected_user_sha256=protected,
        original_raw_parent_immutable_file_sha256=closure['original_immutable_file_sha256'],
        current_V29_V30_V31_sources_models_inputs_results_changed=False,
        actual_new32_CAL_selection_labels_fits_RGB_or_pose_evaluation_in_declare=0)
    OUTPUT.mkdir(parents=True)
    C.write(PLAN, plan)
    for relative, digest in sources.items():
        target = OUTPUT / 'code_snapshot' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((C.ROOT / relative).read_bytes())
        if C.sha(target) != digest:
            raise ValueError('Exact central32 source snapshot required')
    for mapping in (sources, inputs, protected):
        _verify(mapping)
    C.write(OUTPUT / 'PLAN_SEAL.json', dict(plan_sha256=C.sha(PLAN),
        source_sha256=sources, input_sha256=inputs, protected_user_sha256=protected,
        before_new_CAL46_selection_labels_fit_or_full_pose_prediction=True))
    return dict(plan_sha256=C.sha(PLAN), arms=list(ARMS), rows=238, new32_numeric_execution=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-metadata-GO', action='store_true')
    parser.add_argument('--fullsolver-sha256', required=True)
    parser.add_argument('--fullsolver-peer', required=True)
    parser.add_argument('--fullsolver-peer-sha256', required=True)
    parser.add_argument('--operator-peer', required=True)
    parser.add_argument('--operator-peer-sha256', required=True)
    args = parser.parse_args()
    print(declare(root_metadata_GO=args.root_metadata_GO, expected_fullsolver_sha256=args.fullsolver_sha256,
        fullsolver_peer_binding=dict(path=args.fullsolver_peer, sha256=args.fullsolver_peer_sha256),
        operator_peer_binding=dict(path=args.operator_peer, sha256=args.operator_peer_sha256)), flush=True)
