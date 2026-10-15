"""Operational CAL46-only metadata writer around the frozen pure V31 API.

An explicit central declaration SHA is required on every invocation. The
producer CAL seal/index, exact53 immutable CAL files and source/input/user
maps are verified before decoding the ordered CAL46 prefix. No reference,
label, RGB query, primary row, noise fit, objective or pose optimizer is used.
The output file/field contract is the unchanged M30 fitter preparation API.
"""
from copy import deepcopy
from itertools import zip_longest
from pathlib import Path
import argparse

from . import prepare_geometry_guided_native_evidence_v4_31 as M
from . import rerank_observed_anchor_native_ray_population_v4_31 as OP

C, G, AUTH, PREFIX, B = M.C, M.G, M.AUTH, M.PREFIX, M.B
TASKS, OBS, OUTPUT = M.TASKS, OP.OUTPUT, M.OUTPUT
DECLARATION = OP.DECLARATION
CAL_FILES = ('PROTOCOL.json', 'PREPARATION_SEAL.json', 'CAL_EXECUTION_PROTOCOL.json',
             'CAL_ROW_INDEX.json', 'CAL_OBSERVATIONS.jsonl.gz', 'CAL_SUMMARY.json', 'CAL_OBSERVATION_SEAL.json')


def _union(*maps):
    output = {}
    for mapping in maps:
        for path, digest in mapping.items():
            if path in output and output[path] != digest:
                raise ValueError('Immutable metadata ancestry conflict: ' + path)
            output[path] = digest
    return output


def validate_CAL_closure(protocol, seal, index, summary, execution, hashes, expected_sha):
    """Pure schema/link closure; hashes are supplied by the opaque-byte reader."""
    ids = protocol['CAL_ids']
    if (len(ids) != 46 or len(set(ids)) != 46 or ids != protocol['ids'][:46]
            or len(index) != 46 or [e['id'] for e in index] != ids
            or any(e['partition'] != 'CAL' for e in index)
            or seal['rows'] != 46 or seal['ordered_ids'] != ids
            or summary['rows'] != 46 or [r['id'] for r in summary['records']] != ids):
        raise ValueError('Exact ordered CAL46 inventory/index/summary including zeros required')
    fields = dict(protocol_sha256='PROTOCOL.json', preparation_seal_sha256='PREPARATION_SEAL.json',
        execution_protocol_sha256='CAL_EXECUTION_PROTOCOL.json', row_index_sha256='CAL_ROW_INDEX.json',
        observations_sha256='CAL_OBSERVATIONS.jsonl.gz', summary_sha256='CAL_SUMMARY.json')
    if (any(seal.get(key) != hashes[name] for key, name in fields.items())
            or protocol['central_plan_sha256'] != expected_sha
            or execution['central_plan_sha256'] != expected_sha
            or seal['source_sha256'] != protocol['source_sha256']):
        raise ValueError('Same central/source and exact CAL PREP/EXEC/index/journal/summary links required')
    if (seal['selection_profile'] != G.SELECTION_PROFILE
            or protocol['selection_profile'] != G.SELECTION_PROFILE
            or summary['selection_profile'] != G.SELECTION_PROFILE
            or seal['raw_observation_generation'] != G.RAW_OBSERVATION_GENERATION
            or protocol['raw_observation_generation'] != G.RAW_OBSERVATION_GENERATION
            or seal['canonical_validation'] != dict(rows=46, canonical_row_digests_verified=46,
                                                    missing_duplicate_or_extra_rows=0)):
        raise ValueError('New V31 profile/raw V29 generation and complete canonical validation required')
    return list(ids)


def _observation_closure(expected_sha):
    central = OP._central(expected_sha)
    protocol = OP._load(OBS, expected_sha)
    documents = {name: C.read(OBS/name) for name in CAL_FILES if name != 'CAL_OBSERVATIONS.jsonl.gz'}
    hashes = {name: C.sha(OBS/name) for name in CAL_FILES}
    if documents['PREPARATION_SEAL.json']['protocol_sha256'] != hashes['PROTOCOL.json']:
        raise ValueError('Actual producer preparation must bind its original protocol')
    index = documents['CAL_ROW_INDEX.json']['entries']
    ids = validate_CAL_closure(protocol, documents['CAL_OBSERVATION_SEAL.json'], index,
        documents['CAL_SUMMARY.json'], documents['CAL_EXECUTION_PROTOCOL.json'], hashes, expected_sha)
    immutable = OP._cal_immutable(OBS)
    if len(immutable) != 53:
        raise ValueError('Exactly seven sealed CAL files and46 exact spool files required')
    for name, digest in hashes.items():
        if immutable[str((OBS/name).relative_to(C.ROOT))] != digest:
            raise ValueError('CAL immutable snapshot must agree with direct seal/index links')
    OP._verify(immutable)
    return central, protocol, ids, index, immutable


def _sources():
    modules = (M, M.PARENT, G, G.PARENT, G.V30, G.BASELINE, G.N, G.V, G.J, G.P, G.EDGE,
               G.ASSOCIATION, AUTH, AUTH.PARENT, AUTH.RELATION, AUTH.OBS, C, PREFIX, OP)
    paths = [Path(module.__file__).resolve() for module in modules]
    paths.extend((Path(__file__).resolve(),
                  Path(__file__).with_name('test_prepare_geometry_guided_native_metadata_v4_31b.py').resolve()))
    return {str(path.relative_to(C.ROOT)): C.sha(path) for path in paths}


def prepared_CAL_rows(tasks, observations, entries, ids, records, errors):
    """Stream one task/raw row at a time, keeping every zero-evidence row."""
    observed = []
    for task, row, entry in zip_longest(tasks, observations, entries):
        if task is None or row is None or entry is None:
            raise ValueError('Exact CAL-only task/observation/index counts required')
        i = len(observed)
        if (i >= 46 or task['id'] != ids[i] or row['id'] != ids[i] or entry['id'] != ids[i]
                or task['partition'] != 'CAL' or row['partition'] != 'CAL'
                or row['selection_profile'] != G.SELECTION_PROFILE
                or OP.D.R._canonical_sha(row) != entry['canonical_row_sha256']):
            raise ValueError('Every CAL row ordered/profile-bound and canonically equal to its sealed spool')
        OP._validate_selection_counts(row)
        observed.append(row['id'])
        try:
            prepared, metadata = M.prepare_one(task, row)
            records.append(metadata)
            yield prepared
        except Exception as error:
            errors.append(dict(id=task['id'], exception=type(error).__name__, reason=str(error)))
    if observed != ids:
        raise ValueError('All original CAL46 inventory rows including zero evidence required')


def main(output=OUTPUT, *, root_metadata_GO=False, expected_central_plan_sha256=None):
    if root_metadata_GO is not True:
        raise PermissionError('Root actual CAL46 metadata GO before any IO')
    if (not isinstance(expected_central_plan_sha256, str) or len(expected_central_plan_sha256) != 64
            or any(c not in '0123456789abcdef' for c in expected_central_plan_sha256)):
        raise ValueError('Explicit exact central31 SHA required; no embedded default or substitution')
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Preserve every CAL46 metadata attempt')
    central, observation_protocol, ids, index, immutable = _observation_closure(expected_central_plan_sha256)
    task_protocol, task_seal = C.read(TASKS/'PROTOCOL.json'), C.read(TASKS/'INPUT_SEAL.json')
    if ids != task_protocol['ids'][:46] or C.sha(TASKS/'INPUTS.jsonl.gz') != task_seal['input_sha256']:
        raise ValueError('Exact immutable task bytes and original CAL46 prefix required')
    source = _union(task_protocol['source_sha256'], observation_protocol['source_sha256'],
                    central['source_sha256'], _sources())
    inputs = _union(observation_protocol['input_sha256'], central['input_sha256'], immutable,
        {str(path.relative_to(C.ROOT)): C.sha(path) for path in
         (DECLARATION, TASKS/'PROTOCOL.json', TASKS/'INPUT_SEAL.json', TASKS/'INPUTS.jsonl.gz')})
    protected = _union(task_protocol['protected_user_sha256'], observation_protocol['protected_user_sha256'],
                       central['protected_user_sha256'])
    for mapping in (source, inputs, protected):
        PREFIX._verify(mapping)
    output.mkdir(parents=True)
    C.write(output/'PREPARATION_SEAL.json', dict(schema='ZERO_COMPONENT_CAL46_V31_NATIVE_METADATA_PREPARATION',
        ids=ids, source_sha256=source, input_sha256=inputs, protected_user_sha256=protected,
        central_declaration_sha256=expected_central_plan_sha256,
        observation_generation=G.OBSERVATION_FAMILY, raw_observation_generation=G.RAW_OBSERVATION_GENERATION,
        selection_profile=G.SELECTION_PROFILE, noise_population=G.POPULATION,
        original_CAL_file_sha256=immutable, before_CAL_reference_decode=True, CAL_labels_read=False,
        TEST_REMAINDER_rows_decoded=0, workers=1,
        memory_scope='one task and raw observation plus transient canonical copy; small46 metadata summaries only',
        zero_prefit_components_not_fitted_noise=True))
    for relative, digest in source.items():
        target = output/'code_snapshot'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((C.ROOT/relative).read_bytes())
        if C.sha(target) != digest:
            raise ValueError('Exact operational source snapshot required')
    records, errors = [], []
    try:
        C.save_rows(output/'CAL_PREPARED_GROUPS.jsonl.gz', prepared_CAL_rows(
            PREFIX._cal_task_prefix(TASKS/'INPUTS.jsonl.gz', ids),
            C.rows(OBS/'CAL_OBSERVATIONS.jsonl.gz'), index, ids, records, errors))
        report = dict(rows=46, successful_metadata_rows=len(records), error_count=len(errors),
            errors=errors, records=records,
            policy_totals={policy: {key: sum(record['profiles'][policy][key] for record in records)
                                   for key in ('groups', 'tangents', 'owners')} for policy in G.SOURCE_POLICIES},
            observation_generation=G.OBSERVATION_FAMILY, raw_observation_generation=G.RAW_OBSERVATION_GENERATION,
            selection_profile=G.SELECTION_PROFILE, noise_population=G.POPULATION,
            CAL_reference_decodes=0, primary_rows_decoded=0, pose_objective_or_fit_calls=0,
            new_image_queries=0, selected_graph_population_and_measured_tangent_axes_frozen_before_labels=True)
        C.write(output/'CAL_PREFLIGHT_V1.json', report)
        if errors or len(records) != 46:
            raise ValueError('CAL46 metadata must complete46/0 before a successful seal')
        if [row['id'] for row in C.rows(output/'CAL_PREPARED_GROUPS.jsonl.gz')] != ids:
            raise ValueError('Prepared CAL46 output ordered roundtrip required')
        for mapping in (source, inputs, protected, immutable):
            PREFIX._verify(mapping)
        C.write(output/'CAL_PREFLIGHT_SEAL_V1.json', dict(
            preparation_sha256=C.sha(output/'PREPARATION_SEAL.json'),
            report_sha256=C.sha(output/'CAL_PREFLIGHT_V1.json'),
            prepared_groups_sha256=C.sha(output/'CAL_PREPARED_GROUPS.jsonl.gz')))
    except BaseException as error:
        C.write(output/'FAILURE_RECEIPT_V1.json', dict(exception=type(error).__name__, reason=str(error),
            successful_metadata_rows=len(records), errors=errors, central_plan_sha256=expected_central_plan_sha256,
            CAL_labels_primary_rows_RGB_objective_posefit_or_noise_fit=0))
        raise
    return dict(CAL46=len(records), errors=0, policy_totals=report['policy_totals'],
                preflight_seal_sha256=C.sha(output/'CAL_PREFLIGHT_SEAL_V1.json'), CAL_labels=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--central-plan-sha256', required=True)
    parser.add_argument('--root-metadata-GO', action='store_true')
    arguments = parser.parse_args()
    print(main(arguments.output, root_metadata_GO=arguments.root_metadata_GO,
               expected_central_plan_sha256=arguments.central_plan_sha256))
