"""Uniform CAL46 operation of the frozen standalone instance candidate supplier.

Only index + flat mask-manifest file bindings are decoded for preparation.
Raw line rows, keypoint/state/pose payloads and labels are never inputs.
Actual inference requires a separate GO; the CAL3 sampler/math is unchanged.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time
from types import FunctionType

from . import capture_generic_instance_mask_candidates_v1 as P

C = P.C
B = P.B
INDEX_FOLDER = B / 'geometry_guided_edge_observations_v4_29b'
MASKS = B / 'image_evidence_v4_6'
PLAN = B / 'generic_instance_mask_population_CAL_plan_v1/PREDECLARATION.json'
OUTPUT = B / 'generic_instance_mask_candidates_CAL46_v1'
infer_case = P.infer_case
CONFIGURATION = P.CONFIGURATION
WHITESPACE = ' \n\t\r'


def _skip_value(text, start):
    """Find a JSON value's end without decoding nested arrays/objects."""
    while start < len(text) and text[start] in WHITESPACE: start += 1
    if start == len(text): raise ValueError('Missing JSON value')
    if text[start] == '"':
        i = start + 1; escaped = False
        while i < len(text):
            if escaped: escaped = False
            elif text[i] == '\\': escaped = True
            elif text[i] == '"': return i + 1
            i += 1
        raise ValueError('Unterminated skipped string')
    if text[start] not in '[{':
        end = start
        while end < len(text) and text[end] not in ',}]' + WHITESPACE: end += 1
        if end == start: raise ValueError('Missing scalar')
        return end
    stack = [text[start]]; i = start + 1; quoted = False; escaped = False
    while i < len(text):
        char = text[i]
        if quoted:
            if escaped: escaped = False
            elif char == '\\': escaped = True
            elif char == '"': quoted = False
        elif char == '"': quoted = True
        elif char in '[{': stack.append(char)
        elif char in ']}':
            if not stack or (stack[-1], char) not in (('[',']'),('{','}')):
                raise ValueError('Unbalanced JSON container')
            stack.pop()
            if not stack: return i + 1
        i += 1
    raise ValueError('Unterminated JSON container')


def flat_fields(text, allowed):
    """Decode whitelisted outer scalars only; unrequested payloads are skipped."""
    i = 0
    while i < len(text) and text[i] in WHITESPACE: i += 1
    if i == len(text) or text[i] != '{': raise ValueError('Manifest outer object required')
    i += 1; result = {}; seen = set(); decoder = json.JSONDecoder()
    while True:
        while i < len(text) and text[i] in WHITESPACE: i += 1
        if i == len(text): raise ValueError('Unterminated manifest object')
        if text[i] == '}': return result
        key, i = decoder.raw_decode(text, i)
        if not isinstance(key, str) or key in seen: raise ValueError('Unique string manifest keys required')
        seen.add(key)
        while text[i] in WHITESPACE: i += 1
        if text[i] != ':': raise ValueError('Manifest key colon required')
        i += 1
        while text[i] in WHITESPACE: i += 1
        end = _skip_value(text, i)
        if key in allowed:
            if text[i] in '[{': raise ValueError('Only outer scalar metadata is permitted')
            result[key] = decoder.raw_decode(text, i)[0]
        i = end
        while i < len(text) and text[i] in WHITESPACE: i += 1
        if i < len(text) and text[i] == ',': i += 1
        elif i < len(text) and text[i] == '}': return result
        else: raise ValueError('Manifest separator required')


def _records(text):
    i = 0
    while i < len(text) and text[i] in WHITESPACE: i += 1
    if i == len(text) or text[i] != '[': raise ValueError('Mask manifest array required')
    i += 1
    while True:
        while i < len(text) and text[i] in WHITESPACE: i += 1
        if i == len(text): raise ValueError('Unterminated mask manifest')
        if text[i] == ']':
            if text[i+1:].strip(): raise ValueError('Trailing manifest data')
            return
        if text[i] != '{': raise ValueError('Mask manifest records must be objects')
        end = _skip_value(text, i)
        yield text[i:end]
        i = end
        while i < len(text) and text[i] in WHITESPACE: i += 1
        if i < len(text) and text[i] == ',': i += 1
        elif i < len(text) and text[i] == ']': continue
        else: raise ValueError('Manifest array separator required')


def join_CAL_bindings(index, seal, manifest_text):
    entries = index['entries']; ids = [row['id'] for row in entries]
    if (len(ids) != 46 or len(set(ids)) != 46 or any(row['partition'] != 'CAL' for row in entries)
            or seal['rows'] != 46 or ids != seal['ordered_ids']):
        raise ValueError('Exact ordered unique CAL46 index required')
    allowed = set(ids); found = {}; skipped = 0; total = 0
    fields = {'id','partition','image','image_sha256','mask_npz','mask_file_sha256'}
    for raw in _records(manifest_text):
        total += 1
        ident = flat_fields(raw, {'id'})['id']
        if ident not in allowed:
            skipped += 1
            continue
        row = flat_fields(raw, fields)
        if row['partition'] != 'CAL' or ident in found: raise ValueError('Duplicate/nonCAL requested mask metadata')
        name = row['mask_npz']
        if not isinstance(name, str) or Path(name).name != name or name in ('.','..'):
            raise ValueError('Mask evidence must stay inside the existing archive')
        found[ident] = dict(id=ident, partition='CAL',
            image_binding=dict(path=row['image'], sha256=row['image_sha256']),
            target_mask_binding=dict(path=str((MASKS/name).relative_to(C.ROOT)),sha256=row['mask_file_sha256']))
    if set(found) != allowed: raise ValueError('Missing CAL mask metadata')
    return [found[ident] for ident in ids], dict(manifest_records_scanned=total,
        nonCAL_image_mask_or_payload_values_decoded=0, nonCAL_records_skipped_after_id=skipped,
        provider_provenance_raw_points_states_or_line_payloads_decoded=0)


def model_factory(output):
    """Private output-path rebind only; frozen factory bytecode is unchanged."""
    factory = FunctionType(P._load_models.__code__, {**P._load_models.__globals__, 'OUTPUT':Path(output)},
                           '_frozen_factory_new_output', P._load_models.__defaults__)
    factory.__kwdefaults__ = deepcopy(P._load_models.__kwdefaults__)
    return factory


def runtime_config_directory(output):
    # The installed API appends 'Ultralytics' before checking writability.
    path = Path(output) / 'ultralytics_runtime_config' / 'Ultralytics'
    path.mkdir(parents=True, exist_ok=False)
    return path


def prepare(*, root_CAL_metadata_GO=False):
    if root_CAL_metadata_GO is not True: raise PermissionError('Metadata-only CAL46 GO before IO')
    if PLAN.exists(): raise FileExistsError('Preserve the prior plan')
    parent = C.read(P.PLAN)
    if C.sha(P.PLAN) != C.read(P.PLAN.parent/'PLAN_SEAL.json')['plan_sha256']:
        raise ValueError('Frozen CAL3 source plan required')
    if parent['configuration'] != CONFIGURATION: raise ValueError('Exact same sampler/configuration required')
    P._verify(parent['source_sha256']); P._verify(parent['protected_user_sha256'])
    index_path = INDEX_FOLDER/'CAL_ROW_INDEX.json'; cal_seal_path = INDEX_FOLDER/'CAL_OBSERVATION_SEAL.json'
    index, seal = C.read(index_path), C.read(cal_seal_path)
    if C.sha(index_path) != seal['row_index_sha256']: raise ValueError('Original CAL index changed')
    mask_seal = C.read(MASKS/'SEAL.json')
    if C.sha(MASKS/'RECORDS.json') != mask_seal['records_sha256'] or C.sha(MASKS/'PROTOCOL.json') != mask_seal['protocol_sha256']:
        raise ValueError('Original target-mask manifest seal changed')
    cases, inventory = join_CAL_bindings(index, seal, (MASKS/'RECORDS.json').read_text())
    if inventory['manifest_records_scanned'] != mask_seal['records']: raise ValueError('Manifest outer inventory changed')
    sources = dict(parent['source_sha256'])
    for path in (Path(__file__),Path(__file__).with_name('test_capture_generic_instance_mask_population_CAL_v1.py')):
        sources[P._binding(path)['path']] = C.sha(path)
    paths = [P.PLAN,P.PLAN.parent/'PLAN_SEAL.json',index_path,cal_seal_path,
        MASKS/'RECORDS.json',MASKS/'SEAL.json',MASKS/'PROTOCOL.json',P.DETECTOR,P.SAM]
    inputs = {P._binding(path)['path']:C.sha(path) for path in paths}
    if inputs[P._binding(P.DETECTOR)['path']] != P.DETECTOR_SHA or inputs[P._binding(P.SAM)['path']] != P.SAM_SHA:
        raise ValueError('Same existing explicit checkpoints required')
    for case in cases:
        for name in ('image_binding','target_mask_binding'):
            binding=case[name]
            if C.sha(P._resolve(binding['path']))!=binding['sha256']:raise ValueError('Bound CAL image/mask file changed')
            inputs[binding['path']]=binding['sha256']
    for mapping in (sources,inputs,parent['protected_user_sha256']):P._verify(mapping)
    PLAN.parent.mkdir(parents=True)
    value = dict(schema='UNIFORM_CAL46_GENERIC_INSTANCE_CANDIDATE_SUPPLIER_PLAN_V1',
        UTC=datetime.now(timezone.utc).isoformat(),cases=cases,ordered_ids=seal['ordered_ids'],inventory=inventory,
        configuration=deepcopy(CONFIGURATION),source_sha256=sources,input_sha256=inputs,
        protected_user_sha256=parent['protected_user_sha256'],detector=parent['detector'],
        SAM_checkpoint=parent['SAM_checkpoint'],COCO_names=parent['COCO_names'],
        parent_CAL3_plan=P._binding(P.PLAN),uniform_population='all existing CAL46, no error/GT/class/frame-based case selection',
        same_numerical_supplier='frozen infer_case object, same settings/status/overlap/all alternatives and renderer; source/output operations only',
        operational_change='create OUTPUT/ultralytics_runtime_config/Ultralytics before importing the same factory',
        workers=1,rows_in_memory=1,checkpoint_download_or_AMG=False,
        unavailable_and_zero_rows_retained=True,current_pipeline_mask_supply=False,other_object_or_depth_confirmation=False,
        rawline_spool_decodes=0,CAL_label_or_primary_reads=0,image_NPZ_array_or_model_loads_during_prepare=0,
        actual_ROOT_GO_required=True,pose_noise_fits=0,default_adoption=False)
    C.write(PLAN,value)
    for relative,digest in sources.items():
        path=PLAN.parent/'code_snapshot'/relative.lstrip('/')
        path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(P._resolve(relative),path)
        if C.sha(path)!=digest:raise ValueError('Source snapshot differs')
    for mapping in (sources,inputs,parent['protected_user_sha256']):P._verify(mapping)
    C.write(PLAN.parent/'PLAN_SEAL.json',dict(plan_sha256=C.sha(PLAN),source_sha256=sources,
        cases=46,actual_image_mask_model_or_label_decodes=0))
    return dict(plan_sha256=C.sha(PLAN),cases=46,sources=len(sources),inputs=len(inputs),inventory=inventory)


def run(*, expected_plan_sha256, root_CAL_actual_inference_GO=False):
    if root_CAL_actual_inference_GO is not True:raise PermissionError('Separate root CAL46 actual inference GO before IO')
    if not isinstance(expected_plan_sha256,str) or len(expected_plan_sha256)!=64:raise ValueError('Explicit frozen plan SHA required')
    if C.sha(PLAN)!=expected_plan_sha256 or C.read(PLAN.parent/'PLAN_SEAL.json')['plan_sha256']!=expected_plan_sha256:
        raise ValueError('Exact declared CAL46 metadata/source plan required')
    plan=C.read(PLAN)
    if plan['configuration']!=CONFIGURATION or len(plan['cases'])!=46 or [r['id']for r in plan['cases']]!=plan['ordered_ids']:
        raise ValueError('Exact uniform CAL46/configuration required')
    for key in ('source_sha256','input_sha256','protected_user_sha256'):P._verify(plan[key])
    if OUTPUT.exists():raise FileExistsError('Preserve every partial or complete attempt')
    OUTPUT.mkdir(parents=True);(OUTPUT/'masks').mkdir();started=time.monotonic()
    C.write(OUTPUT/'EXECUTION_START.json',dict(plan_sha256=expected_plan_sha256,UTC=datetime.now(timezone.utc).isoformat()))
    try:
        runtime_config_directory(OUTPUT)
        detector,predictor=model_factory(OUTPUT)(plan)
        import cv2
        cv2.setNumThreads(1);rows=[]
        for number,case in enumerate(plan['cases']):
            image=cv2.imread(str(P._resolve(case['image_binding']['path'])))
            if image is None:raise ValueError('Original CAL RGB decode failed')
            with P.np.load(P._resolve(case['target_mask_binding']['path']),allow_pickle=False)as data:target=data['pallet_mask_clean'].copy()
            def save(index,masks,quality):
                path=OUTPUT/'masks'/f'case{number:02d}_instance{index:03d}.npz'
                if path.exists():raise FileExistsError('Preserve actual mask alternatives')
                P.np.savez_compressed(path,sam_candidate_masks=masks,sam_candidate_quality=quality)
                return P._binding(path)
            row=infer_case(image,target,detector,predictor,case_id=case['id'],save_candidates=save)
            row.update(image_binding=case['image_binding'],target_mask_binding=case['target_mask_binding'])
            row['figure_binding']=P._render_case(image,row,OUTPUT,number)
            C.write(OUTPUT/f'case{number:02d}.json',row);rows.append(row)
            print(json.dumps(dict(completed=len(rows),total=46,id=row['id'],detections=row['returned_detections'])),flush=True)
            del image,target
        C.write(OUTPUT/'RESULTS.json',dict(schema=plan['schema'],cases=rows,plan_sha256=expected_plan_sha256))
        text='# CAL46 전수 자동 instance 후보\n\n기존 fixedCAL3와 같은 generic detector/SAM/configuration을 전체CAL46에 적용했습니다. 후보·target 의심·unknown이며 other-object/앞가림/물리O확정이 아닙니다. 빈 결과를 포함하며 기존 pose pipeline에 공급하지 않았습니다.\n\n'
        for number,row in enumerate(rows):text+=f"- {row['id']}: {row['returned_detections']} detections/{row['status']} ([actual](case{number}_candidates.png))\n"
        (OUTPUT/'REPORT_KO.md').write_text(text)
        for key in ('source_sha256','input_sha256','protected_user_sha256'):P._verify(plan[key])
        C.write(OUTPUT/'EXECUTION_SEAL.json',dict(complete=True,cases=46,ordered_ids=plan['ordered_ids'],
            plan_sha256=expected_plan_sha256,source_sha256=plan['source_sha256'],input_sha256=plan['input_sha256'],
            protected_user_sha256=plan['protected_user_sha256'],elapsed_seconds=time.monotonic()-started,
            assets={str(path.relative_to(OUTPUT)):C.sha(path)for path in OUTPUT.rglob('*')if path.is_file()},
            rawline_reference_primary_label_decodes=0,current_pose_pipeline_changes=0,pose_noise_fits=0))
    except BaseException as exc:
        C.write(OUTPUT/'FAILURE_RECEIPT.json',dict(complete=False,plan_sha256=expected_plan_sha256,
            error_type=type(exc).__name__,error=str(exc),attempt_preserved=True))
        raise
    return dict(output=str(OUTPUT.relative_to(C.ROOT)),seal_sha256=C.sha(OUTPUT/'EXECUTION_SEAL.json'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('prepare','run'))
    parser.add_argument('--root-CAL-metadata-GO',action='store_true');parser.add_argument('--root-CAL-actual-inference-GO',action='store_true')
    parser.add_argument('--expected-plan-sha256');args=parser.parse_args()
    print(json.dumps(prepare(root_CAL_metadata_GO=args.root_CAL_metadata_GO)if args.stage=='prepare'else run(
        expected_plan_sha256=args.expected_plan_sha256,root_CAL_actual_inference_GO=args.root_CAL_actual_inference_GO),ensure_ascii=False),flush=True)
