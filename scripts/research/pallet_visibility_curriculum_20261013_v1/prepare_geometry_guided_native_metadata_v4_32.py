"""CAL46 operational metadata around the positive V32 physics factory.

Only the ordered CAL prefix of the original sealed task archive is decoded.
Old baseline/proposals/initializations never enter a numerical factory, worker
or hash used to choose a pose. Actual independent source receipts and source
bytes, original task/observation joins, and all53 CAL immutable files close
before the floors-only populations are written. No labels, PnP, objective,
optimizer, new measurement, primary row or fitted noise is accessed here.
"""
import argparse
from copy import deepcopy
from itertools import zip_longest
from pathlib import Path

import numpy as np

from . import common as C
from . import prepare_geometry_guided_native_evidence_v4_32 as M
from . import geometry_guided_edge_noise_calibration_v4_32 as NOISE
from . import clean_fresh_BI_runtime_v4_32 as CLEAN
from . import rerank_initial_only_observed_ray_population_v4_32 as OP
from . import prepare_prepartial_RGB_evidence_v4_26 as PREFIX

G, AUTH, B = M.G, M.AUTH, OP.B
TASKS = B / 'coarse_prior_containment_eval_v4_25'
OBS = OP.OUTPUT
OUTPUT = B / 'geometry_guided_native_evidence_v4_32_checks'
DECLARATION = OP.DECLARATION
SOURCE_CLOSURE = OP.SOURCE_CLOSURE
RAW_QUERY_AUDIT = OP.ANCESTRY
FACTORY_SHA256 = 'db6467591e237e5d6689935fd744f3d28903465a1c4990feaf09f5b564822b20'
RAW_QUERY_AUDIT_SHA256 = 'e0db2eab479fe2527aa394465f7650c67999de385f4be8cae5c2b26e6b908620'
CAL_FILES = ('PROTOCOL.json', 'PREPARATION_SEAL.json', 'CAL_EXECUTION_PROTOCOL.json',
    'CAL_ROW_INDEX.json', 'CAL_OBSERVATIONS.jsonl.gz', 'CAL_SUMMARY.json', 'CAL_OBSERVATION_SEAL.json')


def _union(*maps):
    return OP._union(*maps)


def validate_CAL_closure(protocol, seal, entries, summary, execution, hashes,
                         expected_sha, expected_operator_sha):
    """Pure inventory/link checks; opaque reader supplies actual file hashes."""
    ids = protocol['CAL_ids']
    if (protocol['schema'] != 'V32_INITIAL_ONLY_CACHED_CAL46_THEN_REST192_PROTOCOL'
            or len(protocol['ids']) != 238 or len(set(protocol['ids'])) != 238
            or len(ids) != 46 or len(set(ids)) != 46 or ids != protocol['ids'][:46]
            or len(entries) != 46 or [e['id'] for e in entries] != ids
            or any(e['partition'] != 'CAL' for e in entries)
            or seal['rows'] != 46 or seal['ordered_ids'] != ids
            or summary['rows'] != 46 or [r['id'] for r in summary['records']] != ids):
        raise ValueError('Exact ordered CAL46 of238 including zero rows required')
    links = dict(protocol_sha256='PROTOCOL.json', preparation_seal_sha256='PREPARATION_SEAL.json',
        execution_protocol_sha256='CAL_EXECUTION_PROTOCOL.json', row_index_sha256='CAL_ROW_INDEX.json',
        observations_sha256='CAL_OBSERVATIONS.jsonl.gz', summary_sha256='CAL_SUMMARY.json')
    operator_path = str(Path(OP.__file__).resolve().relative_to(C.ROOT))
    if (any(seal.get(field) != hashes[name] for field, name in links.items())
            or protocol['central_plan_sha256'] != expected_sha
            or execution['central_plan_sha256'] != expected_sha
            or execution['source_sha256'] != expected_operator_sha
            or protocol['source_sha256'].get(operator_path) != expected_operator_sha
            or seal['source_sha256'] != protocol['source_sha256']):
        raise ValueError('Exact CAL PREP/EXEC/index/journal/summary/source/central closure required')
    if (any(doc['selection_profile'] != G.SELECTION_PROFILE for doc in (seal, protocol, summary))
            or any(doc['raw_observation_generation'] != G.RAW_OBSERVATION_GENERATION for doc in (seal, protocol))
            or seal['canonical_validation'] != dict(rows=46, canonical_row_digests_verified=46,
                                                   missing_duplicate_or_extra_rows=0)):
        raise ValueError('New V32 selection/raw generation and canonical inventory required')
    return list(ids)


def _initial_source_closure():
    if (C.sha(M.__file__) != FACTORY_SHA256
            or C.sha(SOURCE_CLOSURE) != G.INITIAL_SOURCE_CLOSURE_SHA256
            or C.sha(RAW_QUERY_AUDIT) != RAW_QUERY_AUDIT_SHA256):
        raise ValueError('Exact factory and independent initial/raw-query source proofs required')
    source, raw = C.read(SOURCE_CLOSURE), C.read(RAW_QUERY_AUDIT)
    if (source['verdicts']['original_coarse_q_S_and_originalguide'] !=
            'NO_DOWNSTREAM_E_POSE_COPY_FOUND_IN_TRACED_SOURCE'
            or source['verdicts']['statistical_independence_claimed'] is not False
            or tuple(source['clean_cached_measurement_candidate_design']['allow_guide_kinds']) != G.ASSOCIATION.ALLOWED_GUIDE_KINDS
            or raw['verdict'] != 'NO_CROSS_QUERY_NMS_OR_PREGRAPH_CACHED_GUIDE_QUOTA_FOUND'
            or raw['actual_rows_or_images_or_arrays_read'] != 0):
        raise ValueError('Independent original input and no-pregraph-query-quota source findings required')
    source_map = _union(source['source_sha256'], raw['source_sha256'])
    PREFIX._verify(source_map)
    binding = dict(path=str(SOURCE_CLOSURE.relative_to(C.ROOT)),
        sha256=G.INITIAL_SOURCE_CLOSURE_SHA256, verified_by_caller=True,
        independent_initial_geometry_and_BI_q=True)
    return binding, source_map


def _observation_closure(expected_sha):
    central = OP._central(expected_sha)
    protocol = OP._load(OBS, expected_sha)
    documents = {name: C.read(OBS/name) for name in CAL_FILES if name != 'CAL_OBSERVATIONS.jsonl.gz'}
    hashes = {name: C.sha(OBS/name) for name in CAL_FILES}
    if documents['PREPARATION_SEAL.json']['protocol_sha256'] != hashes['PROTOCOL.json']:
        raise ValueError('Observation preparation/protocol mismatch')
    entries = documents['CAL_ROW_INDEX.json']['entries']
    ids = validate_CAL_closure(protocol, documents['CAL_OBSERVATION_SEAL.json'], entries,
        documents['CAL_SUMMARY.json'], documents['CAL_EXECUTION_PROTOCOL.json'], hashes,
        expected_sha, C.sha(OP.__file__))
    immutable = OP._cal_immutable(OBS)
    if len(immutable) != 53 or any(immutable[str((OBS/name).relative_to(C.ROOT))] != digest for name, digest in hashes.items()):
        raise ValueError('Exact seven CAL files plus46 original sealed spools required')
    PREFIX._verify(immutable)
    return central, protocol, ids, entries, immutable


def positive_inputs(task, row, *, initial_source_binding):
    """Project archived CAL input to named positive fields, never old poses.

    The binding has been byte/source verified by the operational caller.
    This function then checks this row's ORIGINAL coarse, image BI archives
    and guard metadata before creating a clean family; no SQPnP is called.
    """
    if task['partition'] != 'CAL' or row['partition'] != 'CAL' or task['id'] != row['id']:
        raise ValueError('CAL-only same identity gate before physics/coordinate payload')
    binding = M._source_binding(initial_source_binding)
    spec, geometry = task['spec'], row['initial_geometry']
    states, hidden = list(task['states']), task['initial_self_hidden']
    if list(spec['states']) != states or list(row['states']) != states:
        raise ValueError('Same authoritative state binding before q/archive payload')
    task_visible = M._BI_records(states, hidden, spec['visible'])
    row_visible = M._BI_records(states, hidden, row['visible_measurements'])
    if (task_visible != row_visible or row['domains'] != spec['domains']
            or row['u_regions'] != task['u_regions'] or row['session'] != task['session']
            or row['image_sha256'] != task['image_sha256']
            or geometry['K'] != spec['K'] or list(geometry['image_hw']) != list(spec['image_hw'])
            or list(geometry['initial_self_hidden']) != list(hidden)):
        raise ValueError('Exact archived BI q/source/covariance/weight/canvas/domain/U/image join required')
    original = task['families']['ORIGINAL_WD']
    xyz = G._array(original['native_xyz'], (8, 3), 'Original known native XYZ')
    Q = M.FRESH._rotation(original['physical_registration_Q'], 'Original known Q')
    R = M.FRESH._rotation(original['coarse_pose']['R'], 'Independent original initial R')
    t = G._array(original['coarse_pose']['t'], (3,), 'Independent original native t')
    if (not np.array_equal(xyz, np.asarray(geometry['native_xyz'], float))
            or not np.array_equal(Q, np.asarray(geometry['physical_registration_Q'], float))
            or not np.allclose(R, geometry['R_cf'], atol=1e-12, rtol=0)
            or not np.allclose(R @ xyz.mean(axis=0)+t, geometry['centroid'], atol=1e-12, rtol=0)):
        raise ValueError('Original independent coarse/native geometry/model center join required')
    # Read only nonS MODEL slots. Unknown observed coordinates never enter.
    for k, is_hidden in enumerate(hidden):
        if is_hidden: continue
        first = task['complete_BI_evidence']['initial_projection_xy'][k]
        other = geometry['initial_projection_xy'][k]
        if first != other:
            raise ValueError('Original nonS MODEL projection source changed')
    coarse = dict(R=R.tolist(), t=t.tolist(), physical_registration_Q=Q.tolist(),
        R_physical=(R @ Q).tolist(), source=M.FRESH.COARSE_SOURCE,
        independent_of_cached_pose_bank=True)
    family = CLEAN._families(xyz, coarse, Q)['ORIGINAL_WD']  # Geometry only, never fresh PnP.
    inputs = dict(family=family, K=deepcopy(spec['K']), states=states,
        visible_measurements=task_visible, domains=deepcopy(spec['domains']),
        image_hw=list(spec['image_hw']), config=deepcopy(spec['config']),
        boundary_merged=deepcopy(task['boundary_merged']), u_regions=deepcopy(task['u_regions']),
        initial_self_hidden=list(hidden), initial_source_binding=binding)
    metadata = dict(identifier=task['id'], partition='CAL', session=task['session'],
        initial_geometry=geometry, image_binding=dict(sha256=task['image_sha256'], image_hw=list(spec['image_hw'])),
        observation=row['observation'])
    if row['selection_profile'] != G.SELECTION_PROFILE or row['observation'].get('selection_profile') != G.SELECTION_PROFILE:
        raise ValueError('Exact V32 initial-only selected profile required')
    return inputs, metadata


def prepared_CAL_rows(tasks, observations, entries, ids, records, errors, *, initial_source_binding):
    observed = []
    for task, row, entry in zip_longest(tasks, observations, entries):
        if task is None or row is None or entry is None:
            raise ValueError('Exact CAL-only task/row/index counts required')
        i = len(observed)
        if (i >= 46 or task['id'] != ids[i] or row['id'] != ids[i] or entry['id'] != ids[i]
                or task['partition'] != 'CAL' or row['partition'] != 'CAL'
                or row['selection_profile'] != G.SELECTION_PROFILE
                or OP.D.R._canonical_sha(row) != entry['canonical_row_sha256']):
            raise ValueError('Ordered CAL rows must match sealed canonical spools')
        OP._validate_selection_counts(row)
        observed.append(row['id'])
        identifier = row['id']
        inputs = metadata = prepared = None
        try:
            inputs, metadata = positive_inputs(task, row, initial_source_binding=initial_source_binding)
            prepared, record = M.prepare_one(**inputs, **metadata)
            records.append(record)
        except Exception as error:
            errors.append(dict(id=identifier, exception=type(error).__name__, reason=str(error)))
        finally:
            # Release old raw arrays before yielding or fetching the next row.
            del inputs, metadata, task, row, entry
        if prepared is not None:
            yield prepared
            del prepared
    if observed != ids:
        raise ValueError('All CAL46 rows including zero support required')


def _sources():
    modules = (M, G, G.PREVIOUS, G.PARENT, AUTH, AUTH.PREVIOUS, AUTH.PARENT,
        NOISE, NOISE.PREVIOUS, NOISE.PARENT, CLEAN, M.FRESH, G.BASELINE,
        G.N, G.V, G.J, G.P, G.EDGE, G.ASSOCIATION, AUTH.RELATION, AUTH.OBS, C, PREFIX, OP)
    paths = [Path(module.__file__).resolve() for module in modules]
    paths += [Path(__file__).resolve(), Path(__file__).with_name('test_prepare_geometry_guided_native_metadata_v4_32.py').resolve()]
    return {str(path.relative_to(C.ROOT)): C.sha(path) for path in paths}


def main(output=OUTPUT, *, root_metadata_GO=False, expected_central_plan_sha256=None):
    if root_metadata_GO is not True:
        raise PermissionError('Root CAL46 metadata GO before any IO')
    if (not isinstance(expected_central_plan_sha256, str) or len(expected_central_plan_sha256) != 64
            or any(c not in '0123456789abcdef' for c in expected_central_plan_sha256)):
        raise ValueError('Explicit exact central32 SHA required; no default/substitution')
    output = Path(output).resolve()
    if output.exists(): raise FileExistsError('Preserve every metadata attempt')
    binding, proof_sources = _initial_source_closure()
    central, obs_protocol, ids, entries, immutable = _observation_closure(expected_central_plan_sha256)
    task_protocol, task_seal = C.read(TASKS/'PROTOCOL.json'), C.read(TASKS/'INPUT_SEAL.json')
    if (task_seal['rows'] != 238 or task_seal['ordered_ids'] != task_protocol['ids']
            or ids != task_protocol['ids'][:46]
            or C.sha(TASKS/'PROTOCOL.json') != task_seal['protocol_sha256']
            or C.sha(TASKS/'INPUTS.jsonl.gz') != task_seal['input_sha256']):
        raise ValueError('Exact original sealed238 task bytes and CAL46 prefix required')
    source = _union(task_protocol['source_sha256'], obs_protocol['source_sha256'], central['source_sha256'], proof_sources, _sources())
    inputs = _union(task_protocol['input_sha256'], obs_protocol['input_sha256'], central['input_sha256'], immutable,
        {str(path.relative_to(C.ROOT)): C.sha(path) for path in
         (DECLARATION, SOURCE_CLOSURE, RAW_QUERY_AUDIT, TASKS/'PROTOCOL.json', TASKS/'INPUT_SEAL.json', TASKS/'INPUTS.jsonl.gz')})
    protected = _union(task_protocol['protected_user_sha256'], obs_protocol['protected_user_sha256'], central['protected_user_sha256'])
    if len(protected) != 6: raise ValueError('All original six protected user files required')
    for mapping in (source, inputs, protected): PREFIX._verify(mapping)
    output.mkdir(parents=True)
    C.write(output/'PREPARATION_SEAL.json', dict(schema='ZERO_COMPONENT_CAL46_V32_POSITIVE_NATIVE_METADATA_PREPARATION',
        ids=ids, source_sha256=source, input_sha256=inputs, protected_user_sha256=protected,
        central_declaration_sha256=expected_central_plan_sha256, original_CAL_file_sha256=immutable,
        observation_generation=G.OBSERVATION_FAMILY, raw_observation_generation=G.RAW_OBSERVATION_GENERATION,
        selection_profile=G.SELECTION_PROFILE, noise_population=G.POPULATION,
        initial_source_binding=binding, raw_query_source_audit_sha256=RAW_QUERY_AUDIT_SHA256,
        positive_physics_only_no_old_pose_task_or_bank_forwarding=True,
        before_CAL_reference_decode=True, CAL_labels_read=False, TEST_REMAINDER_rows_decoded=0,
        workers=1, memory_scope='one CAL task/raw row and transient canonical copy; small46 summaries only',
        zero_prefit_components_not_fitted_noise=True))
    for relative, digest in source.items():
        target = output/'code_snapshot'/relative
        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes((C.ROOT/relative).read_bytes())
        if C.sha(target) != digest: raise ValueError('Source snapshot changed')
    records, errors = [], []
    try:
        OP.save_rows_large(output/'CAL_PREPARED_GROUPS.jsonl.gz', prepared_CAL_rows(
            PREFIX._cal_task_prefix(TASKS/'INPUTS.jsonl.gz', ids), C.rows(OBS/'CAL_OBSERVATIONS.jsonl.gz'),
            entries, ids, records, errors, initial_source_binding=binding))
        report = dict(rows=46, successful_metadata_rows=len(records), error_count=len(errors), errors=errors, records=records,
            policy_totals={p: {k: sum(r['profiles'][p][k] for r in records) for k in ('groups', 'tangents', 'owners')}
                           for p in G.SOURCE_POLICIES},
            observation_generation=G.OBSERVATION_FAMILY, raw_observation_generation=G.RAW_OBSERVATION_GENERATION,
            selection_profile=G.SELECTION_PROFILE, noise_population=G.POPULATION,
            CAL_reference_decodes=0, primary_rows_decoded=0, pose_objective_or_fit_calls=0, fresh_PnP_calls=0,
            new_image_queries=0, selected_graph_population_and_measured_tangent_axes_frozen_before_labels=True)
        C.write(output/'CAL_PREFLIGHT_V1.json', report)
        if errors or len(records) != 46: raise ValueError('Complete CAL46 metadata46/0 required before seal')
        if [r['id'] for r in C.rows(output/'CAL_PREPARED_GROUPS.jsonl.gz')] != ids:
            raise ValueError('Prepared CAL46 output order changed')
        for mapping in (source, inputs, protected, immutable): PREFIX._verify(mapping)
        C.write(output/'CAL_PREFLIGHT_SEAL_V1.json', dict(preparation_sha256=C.sha(output/'PREPARATION_SEAL.json'),
            report_sha256=C.sha(output/'CAL_PREFLIGHT_V1.json'), prepared_groups_sha256=C.sha(output/'CAL_PREPARED_GROUPS.jsonl.gz')))
    except BaseException as error:
        C.write(output/'FAILURE_RECEIPT_V1.json', dict(exception=type(error).__name__, reason=str(error),
            successful_metadata_rows=len(records), errors=errors, central_plan_sha256=expected_central_plan_sha256,
            CAL_labels_primary_rows_RGB_PnP_objective_posefit_or_noise_fit=0))
        raise
    return dict(CAL46=len(records), errors=0, policy_totals=report['policy_totals'],
        preflight_seal_sha256=C.sha(output/'CAL_PREFLIGHT_SEAL_V1.json'), CAL_labels=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--central-plan-sha256', required=True)
    parser.add_argument('--root-metadata-GO', action='store_true')
    args = parser.parse_args()
    print(main(args.output, root_metadata_GO=args.root_metadata_GO,
               expected_central_plan_sha256=args.central_plan_sha256))
