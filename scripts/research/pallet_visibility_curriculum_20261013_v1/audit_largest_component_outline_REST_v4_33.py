"""Independent cached-mask replay of REST192 after the sealed CAL46 gate.

Reuse the independently reviewed pixel/clip/sample arithmetic, never the
LCC producer or frozen outline observer. This checks saved observations;
it makes no claim about semantic mask identity or pose accuracy.
"""
from pathlib import Path

import numpy as np

from . import audit_largest_component_outline_v4_33 as A

ROOT = A.ROOT
PROFILE = A.PROFILE
CENTRAL = ROOT/'_docs/experiments/pallet_visibility_curriculum_20261013_v1/nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/largest_component_global_outline_method_plan_v4_33/PREDECLARATION.json'


def sha_text(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def validate_metadata(central, final, full, rest, entries, cal_audit):
    ids = central['ordered_ids']
    expected_parts = ['CAL']*46+['TEST']*49+['REMAINDER']*143
    records = central['mask_source_packet_records']
    if (len(ids) != 238 or len(set(ids)) != 238 or len(records) != 238
            or [r['id'] for r in records] != ids
            or [r['partition'] for r in records] != expected_parts
            or full['rows'] != 238 or full['ordered_ids'] != ids
            or full['partitions'] != dict(CAL=46, TEST=49, REMAINDER=143)
            or full['selection_profile'] != PROFILE
            or final['selection_profile'] != PROFILE or final['ordered_ids'] != ids
            or [r['id'] for r in entries] != ids
            or [r['partition'] for r in entries] != expected_parts
            or rest['rows'] != 192 or rest['ordered_ids'] != ids[46:]
            or rest['selection_profile'] != PROFILE or rest['union_exact_match_rows'] != 192
            or cal_audit['rows'] != 46 or cal_audit['failure_count'] != 0
            or cal_audit['CAL_observation_seal_sha256'] != full['partition_seals']['CAL']
            or final['protected_user_sha256'] != central['protected_user_sha256']
            or final['original_CAL_file_sha256'] != central['original_CAL_file_sha256']
            or any(final['source_sha256'].get(p) != s for p,s in central['source_sha256'].items())
            or len(final['protected_user_sha256']) != 6
            or len(final['original_CAL_file_sha256']) != 53
            or len(final['immutable_LCC_CAL_file_sha256']) != 53):
        raise ValueError('Exact all238/CAL46/REST192 populations and immutable CAL closures required')
    return ids[46:], records[46:], entries[46:]


def audit_rest(*, folder, expected_full_seal_sha256, expected_central_sha256,
               expected_cal_audit_sha256, root_REST_observation_audit_GO=False):
    if root_REST_observation_audit_GO is not True:
        raise PermissionError('ROOT REST192 pixel audit GO before IO')
    if not all(sha_text(v) for v in (expected_full_seal_sha256, expected_central_sha256, expected_cal_audit_sha256)):
        raise ValueError('Explicit full/central/CAL audit SHA before IO')
    folder = Path(folder).resolve()
    if not folder.is_relative_to(ROOT):
        raise ValueError('Bound repository observation folder required')
    calpath = folder/'independent_mask_outline_audit_v1/AUDIT.json'
    fullpath = folder/'OUTLINE_OBSERVATION_SEAL.json'
    for path, expected in ((CENTRAL, expected_central_sha256),
                           (fullpath, expected_full_seal_sha256), (calpath, expected_cal_audit_sha256)):
        if A.sha_file(path) != expected:
            raise ValueError('Current caller-pinned sealed metadata required')
    full = A.read(fullpath); final = A.read(folder/'FINAL_PROTOCOL.json')
    rest = A.read(folder/'REST_OBSERVATION_SEAL.json'); cal = A.read(calpath)
    entries = A.read(folder/'OUTLINE_ROW_INDEX.json')['entries']
    central = A.read(CENTRAL)
    ids, records, selected = validate_metadata(central, final, full, rest, entries, cal)
    if (final['central_plan_sha256'] != expected_central_sha256
            or cal['central_sha256'] != expected_central_sha256
            or A.sha_file(folder/'FINAL_PROTOCOL.json') != full['protocol_sha256']
            or A.sha_file(folder/'OUTLINE_ROW_INDEX.json') != full['row_index_sha256']
            or A.sha_file(folder/'OUTLINE_OBSERVATIONS.jsonl.gz') != full['observations_sha256']
            or A.sha_file(folder/'REST_OBSERVATION_SEAL.json') != full['partition_seals']['REST']
            or A.sha_file(folder/'CAL_OBSERVATION_SEAL.json') != full['partition_seals']['CAL']
            or A.sha_file(folder/'REST_ROW_INDEX.json') != rest['row_index_sha256']
            or A.sha_file(folder/'REST_OUTLINE_OBSERVATIONS.jsonl.gz') != rest['observations_sha256']
            or A.read(folder/'REST_ROW_INDEX.json')['entries'] != selected):
        raise ValueError('Full protocol/journals/partition seals/indices must agree')
    maps = [final[k] for k in ('source_sha256','input_sha256','protected_user_sha256',
                              'original_CAL_file_sha256','immutable_LCC_CAL_file_sha256')]
    A.verify_maps(*maps)
    dest = folder/'independent_REST_mask_outline_audit_v1'
    dest.mkdir(exist_ok=False)
    sources = {str(p.relative_to(ROOT)):A.sha_file(p) for p in (
        Path(__file__).resolve(), Path(__file__).with_name('test_audit_largest_component_outline_REST_v4_33.py').resolve(),
        Path(A.__file__).resolve(), Path(A.__file__).with_name('test_audit_largest_component_outline_v4_33.py').resolve())}
    metadata = {str(p.relative_to(ROOT)):A.sha_file(p) for p in (CENTRAL, fullpath, calpath,
        folder/'FINAL_PROTOCOL.json',folder/'REST_ROW_INDEX.json',folder/'OUTLINE_ROW_INDEX.json',
        folder/'REST_OBSERVATION_SEAL.json',folder/'CAL_OBSERVATION_SEAL.json',
        folder/'REST_OUTLINE_OBSERVATIONS.jsonl.gz',folder/'OUTLINE_OBSERVATIONS.jsonl.gz')}
    A.write(dest/'PLAN.json',dict(schema='INDEPENDENT_REST192_LCC8_PIXEL_OUTLINE_AUDIT_V33',
        source_sha256=sources,input_sha256=metadata,ordered_ids=ids,rows=192,
        numeric_mask_or_observation_decodes_before_plan=0,new_LCC_producer_or_old_observer_calls=0,
        scope='Same cached mask pixels, independent clip/sample arithmetic; no semantic or pose accuracy proof'))
    results=[]; failures=[]; last=None
    try:
        for entry, record in zip(selected, records):
            last=record['id']; path=A.relative_path(entry['path'])
            if A.sha_file(path) != entry['sha256']:
                raise ValueError('Bound saved observation row changed')
            row=A.read(path)
            if (A.digest(row) != entry['canonical_row_sha256'] or row['id'] != last
                    or row['partition'] != record['partition']
                    or row['positive_packet_sha256'] != record['positive_packet_sha256']):
                raise ValueError('Canonical original positive/observation identity changed')
            binding=row['target_mask_binding']; maskpath=A.relative_path(binding['path'])
            if A.sha_file(maskpath) != binding['sha256']:
                raise ValueError('Original cached mask bytes changed')
            with np.load(maskpath,allow_pickle=False) as arrays:
                mask=arrays['pallet_mask_clean'].copy()
            value=A.audit_row(mask,row)
            results.append(dict(id=last,partition=record['partition'],**value))
            failures.extend(dict(id=last,**check) for check in value['checks'] if not check['passed'])
            del mask,row
        A.verify_maps(*maps,metadata,sources)
    except BaseException as error:
        A.write(dest/'FAILURE.json',dict(rows_completed=len(results),attempted_id=last,
            exception=type(error).__name__,reason=str(error),attempt_preserved=True))
        raise
    report=dict(schema='INDEPENDENT_REST192_LCC8_PIXEL_OUTLINE_AUDIT_V33',rows=len(results),
        checks=sum(len(r['checks']) for r in results),failure_count=len(failures),failures=failures,
        row_results=results,maximum_coordinate_difference=max(r['maximum_coordinate_difference'] for r in results),
        source_sha256=sources,input_sha256=metadata,central_sha256=expected_central_sha256,
        full_observation_seal_sha256=expected_full_seal_sha256,CAL_audit_sha256=expected_cal_audit_sha256,
        CAL46_previously_checked=46,all238_observation_replays_complete=len(results)==192 and not failures,
        new_LCC_producer_or_old_observer_calls=0,GT_reference_NN_PnP_or_pose_inputs=0,
        semantic_or_pose_accuracy_certification=False)
    A.write(dest/'AUDIT.json',report)
    A.write(dest/'AUDIT_SEAL.json',dict(report_sha256=A.sha_file(dest/'AUDIT.json'),
        plan_sha256=A.sha_file(dest/'PLAN.json'),source_sha256=sources,input_sha256=metadata))
    if failures:
        raise SystemExit(1)
    return dict(rows=len(results),checks=report['checks'],failures=0,audit_sha256=A.sha_file(dest/'AUDIT.json'))
