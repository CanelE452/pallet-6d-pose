"""CAL-first union/LCC8 mask observation operation, gated before numeric IO.

Preparation reads flat manifest/index/seal metadata and opaque file hashes.
Only a later ROOT-gated run opens the first 46 positive packets and cached
``pallet_mask_clean`` arrays. No RGB, models, E comparison, raw graph, labels,
pose generation, sampling changes or primary packet payloads are consumed.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from . import common as C
from . import largest_component_global_outline_v4_33 as LCC
from . import capture_generic_instance_mask_population_CAL_v1 as FLAT


B = FLAT.B
POSITIVE = B/'clean_fresh_BI_pose_eval_v4_32_r2'
CAL_OBSERVATIONS = B/'initial_only_observed_ray_observations_v4_32'
MASKS = B/'image_evidence_v4_6'
OUTPUT = B/'largest_component_global_outline_observations_v4_33'
SCHEMA = 'SEALED_CAL46_UNION_AND_LCC8_GLOBAL_OUTLINE_OBSERVATIONS_V33'
SELECTION_PROFILE = LCC.SELECTION_PROFILE
BASELINE_PROFILE = LCC.BASELINE_PROFILE
CAL_COUNT = 46
COMPATIBLE_CAL_REFERENCE_COUNT = 44  # Inventory contract only; no references opened here.


def _sha(value):
    return isinstance(value, str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def _digest(value):
    return hashlib.sha256(json.dumps(C.finite(value), sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _relative(path):
    return str(Path(path).resolve().relative_to(C.ROOT))


def _resolve(relative):
    path=(C.ROOT/relative).resolve()
    if not path.is_relative_to(C.ROOT):
        raise ValueError('Repository-relative bound input required')
    return path


def _union(*maps):
    result={}
    for mapping in maps:
        for path,digest in mapping.items():
            if not _sha(digest) or (path in result and result[path]!=digest):
                raise ValueError('Exact source/input ancestry union required')
            result[path]=digest
    return result


def _verify(*maps):
    for mapping in maps:
        for path,expected in mapping.items():
            if C.sha(_resolve(path))!=expected:
                raise ValueError('Bound bytes changed: '+path)


def _sources():
    modules=(C,LCC,LCC.FROZEN_OUTLINE,LCC.FROZEN_CONTAINMENT,
        LCC.FROZEN_CONTAINMENT.OLD,LCC.FROZEN_CONTAINMENT.J,
        LCC.FROZEN_CONTAINMENT.BASE,FLAT,FLAT.P)
    paths=[Path(module.__file__) for module in modules]
    paths += [Path(__file__),Path(__file__).with_name('test_observe_largest_component_global_outline_CAL_v4_33.py')]
    return {_relative(path):C.sha(path) for path in paths}


def _central(path, expected, *, root_GO):
    if root_GO is not True:
        raise PermissionError('Explicit ROOT stage GO before any IO')
    if not _sha(expected):
        raise ValueError('Explicit central33 lowercase SHA before IO')
    path=Path(path).resolve()
    if C.sha(path)!=expected:
        raise ValueError('Exact central33 declaration required')
    value=C.read(path)
    if value['selection_profile']!=LCC.SELECTION_PROFILE:
        raise ValueError('Current single LCC8 profile required')
    for key in ('source_sha256','input_sha256','protected_user_sha256'):
        _verify(value[key])
    if len(value['protected_user_sha256'])!=6:
        raise ValueError('Original protected six bindings required')
    for source in (Path(LCC.__file__),Path(__file__),Path(__file__).with_name('test_observe_largest_component_global_outline_CAL_v4_33.py')):
        if value['source_sha256'].get(_relative(source))!=C.sha(source):
            raise ValueError('Central declaration must pin the final helper/operator/test')
    return value


def _CAL53(folder,index):
    paths=[Path(folder)/name for name in ('PROTOCOL.json','PREPARATION_SEAL.json',
        'CAL_EXECUTION_PROTOCOL.json','CAL_ROW_INDEX.json','CAL_OBSERVATIONS.jsonl.gz',
        'CAL_SUMMARY.json','CAL_OBSERVATION_SEAL.json')]
    for entry in index['entries']:
        path=Path(folder)/'row_spool'/(hashlib.sha256(entry['id'].encode()).hexdigest()+'.jsonl.gz')
        if C.sha(path)!=entry['sha256']:
            raise ValueError('Original CAL spool changed')
        paths.append(path)
    if len(paths)!=53 or len(set(paths))!=53:
        raise ValueError('Seven original CAL files plus ordered46 spools required')
    return {_relative(path):C.sha(path) for path in paths}


def join_bindings(index, cal_seal, manifest_text, positive_protocol, positive_seal):
    """Flat metadata only, exact CAL46 order, nonCAL skipped after ID."""
    cases,inventory=FLAT.join_CAL_bindings(index,cal_seal,manifest_text)
    ids=[case['id'] for case in cases]
    if (positive_seal['rows']!=238 or positive_seal['ordered_ids']!=positive_protocol['ordered_ids']
            or ids!=positive_protocol['ordered_ids'][:CAL_COUNT]
            or len(positive_protocol['records'])!=238):
        raise ValueError('Same sealed positive238 and CAL46 prefix required')
    for case,record in zip(cases,positive_protocol['records'][:CAL_COUNT]):
        if case['id']!=record['id'] or record['partition']!='CAL' or not _sha(record['positive_packet_sha256']):
            raise ValueError('Exact CAL positive packet record binding required')
        case['positive_packet_sha256']=record['positive_packet_sha256']
    return cases,inventory


def prepare(*, central_plan, expected_central_sha256, root_metadata_GO=False, output=OUTPUT):
    central=_central(central_plan,expected_central_sha256,root_GO=root_metadata_GO)
    output=Path(output).resolve()
    if output.exists():
        raise FileExistsError('Preserve every preparation attempt')
    index=C.read(CAL_OBSERVATIONS/'CAL_ROW_INDEX.json')
    cal_seal=C.read(CAL_OBSERVATIONS/'CAL_OBSERVATION_SEAL.json')
    for name,key in (('PROTOCOL.json','protocol_sha256'),('PREPARATION_SEAL.json','preparation_seal_sha256'),
            ('CAL_EXECUTION_PROTOCOL.json','execution_protocol_sha256'),('CAL_ROW_INDEX.json','row_index_sha256'),
            ('CAL_OBSERVATIONS.jsonl.gz','observations_sha256'),('CAL_SUMMARY.json','summary_sha256')):
        if C.sha(CAL_OBSERVATIONS/name)!=cal_seal[key]:
            raise ValueError('Original CAL seal reference changed: '+name)
    immutable=_CAL53(CAL_OBSERVATIONS,index)
    mask_seal=C.read(MASKS/'SEAL.json')
    if (C.sha(MASKS/'RECORDS.json')!=mask_seal['records_sha256']
            or C.sha(MASKS/'PROTOCOL.json')!=mask_seal['protocol_sha256']):
        raise ValueError('Original cached mask manifest seal changed')
    protocol=C.read(POSITIVE/'PROTOCOL.json');seal=C.read(POSITIVE/'INPUT_SEAL.json')
    if C.sha(POSITIVE/'PROTOCOL.json')!=seal['protocol_sha256'] or C.sha(POSITIVE/'INPUTS.jsonl.gz')!=seal['input_sha256']:
        raise ValueError('Exact unchanged R2 positive input archive required')
    if (central['ordered_ids']!=protocol['ordered_ids']
            or central['partitions']!=dict(CAL=46,TEST=49,REMAINDER=143,PRIMARY=192,severe=0)
            or central['original_CAL_file_sha256']!=immutable):
        raise ValueError('Central all238 order and original CAL53 must match the positive source')
    records=central['mask_source_packet_records']
    if len(records)!=238 or any(any(record[key]!=original[key] for key in
            ('id','partition','positive_packet_sha256')) for record,original in zip(records,protocol['records'])):
        raise ValueError('Central mask-source packet metadata must match all sealed positive records')
    cases,inventory=join_bindings(index,cal_seal,(MASKS/'RECORDS.json').read_text(),protocol,seal)
    if inventory['manifest_records_scanned']!=mask_seal['records']:
        raise ValueError('Original outer manifest inventory changed')
    source=_union(central['source_sha256'],_sources())
    files=[Path(central_plan),POSITIVE/'PROTOCOL.json',POSITIVE/'INPUT_SEAL.json',POSITIVE/'INPUTS.jsonl.gz',
        MASKS/'RECORDS.json',MASKS/'PROTOCOL.json',MASKS/'SEAL.json']
    inputs=_union(central['input_sha256'],immutable,{_relative(path):C.sha(path) for path in files})
    for case in cases:
        binding=case['target_mask_binding']
        inputs=_union(inputs,{binding['path']:binding['sha256']})
    protected=central['protected_user_sha256'];_verify(source,inputs,protected)
    output.mkdir(parents=True)
    prepared=dict(schema=SCHEMA,central_plan=dict(path=_relative(central_plan),sha256=expected_central_sha256),
        selection_profile=LCC.SELECTION_PROFILE,baseline_profile=LCC.BASELINE_PROFILE,
        source_sha256=source,input_sha256=inputs,protected_user_sha256=protected,
        original_CAL_file_sha256=immutable,cases=cases,ordered_CAL_ids=[case['id'] for case in cases],
        positive_archive=dict(folder=_relative(POSITIVE),input_sha256=seal['input_sha256']),
        CAL_reference_inventory=46,compatible_CAL_reference_inventory=44,
        reference_alignment_or_labels_read=False,workers=1,large_rows_in_memory=1,
        memory='one CAL positive packet and mask; union/LCC arrays and serialization temporaries; small46 summaries',
        inventory=inventory,actual_packet_mask_RGB_model_or_label_decodes=0,
        union_must_equal_all_saved_outline_fields=True,coefficient_C_noise_physics_or_seed_changes=0,
        primary_application=False,default_adoption=False)
    C.write(output/'PREPARATION.json',prepared)
    _verify(source,inputs,protected)
    C.write(output/'PREPARATION_SEAL.json',dict(preparation_sha256=C.sha(output/'PREPARATION.json'),
        source_sha256=source,input_sha256=inputs,protected_user_sha256=protected,
        original_CAL_file_sha256=immutable,cases=46,actual_array_or_packet_decodes=0))
    return dict(output=_relative(output),preparation_sha256=C.sha(output/'PREPARATION.json'))


def CAL_prefix(rows, ids):
    """Never request row47; the parent seal already certifies the full archive."""
    if len(ids)!=CAL_COUNT or len(set(ids))!=CAL_COUNT:
        raise ValueError('Exact ordered46 CAL IDs required')
    stream=iter(rows)
    try:
        for identifier in ids:
            packet=next(stream)
            if packet['identifiers']['id']!=identifier or packet['identifiers']['partition']!='CAL':
                raise ValueError('Only exact ordered CAL46 prefix may be decoded')
            yield packet
    finally:
        close=getattr(stream,'close',None)
        if close is not None:
            close()


def process_one(packet, mask, binding):
    """Pure synthetic-testable join; no pose/state/keypoint fields are queried."""
    if packet['identifiers']['partition']!='CAL' or packet['identifiers']['id']!=binding['id']:
        raise ValueError('Exact CAL identity required')
    if _digest(packet)!=binding['positive_packet_sha256']:
        raise ValueError('Canonical positive packet differs from its sealed record')
    saved=packet['regularizers']['outline_observation']
    image=saved['image_binding'];file=binding['target_mask_binding']
    if (image['image_sha256']!=binding['image_binding']['sha256']
            or image['image_sha256']!=packet['identifiers']['image_sha256']
            or image['mask_file_sha256']!=file['sha256']
            or image['mask_npz']!=Path(file['path']).name
            or saved['image_hw']!=packet['physics']['image_hw']
            or saved['configuration']!=packet['regularizers']['outline_configuration']):
        raise ValueError('Same cached mask/image/canvas/configuration binding required')
    before=_digest(packet)
    bundle=LCC.observe_union_and_largest_component(mask,saved['image_hw'],
        image_binding=image,configuration=saved['configuration'])
    union={key:value for key,value in bundle['original_union'].items() if key!='selection_profile'}
    per_field={key:_digest(union.get(key))==_digest(saved.get(key)) for key in set(union)|set(saved)}
    same=set(union)==set(saved) and all(per_field.values()) and _digest(union)==_digest(saved)
    if _digest(packet)!=before:
        raise ValueError('Original positive packet mutated')
    return dict(schema=SCHEMA,id=binding['id'],partition='CAL',row_retained=True,
        selection_profile=LCC.SELECTION_PROFILE,baseline_profile=LCC.BASELINE_PROFILE,
        target_mask_binding=deepcopy(file),positive_packet_sha256=binding['positive_packet_sha256'],
        stored_union_sha256=_digest(saved),regenerated_union_sha256=_digest(union),
        union_exact_match=bool(same),union_per_field_equal=per_field,
        stored_union_on_mismatch=None if same else saved,
        original_union=bundle['original_union'],largest_component=bundle['largest_component'],
        C_coefficient_or_noise_physics_seed_or_mask_bytes_changed=False,
        reference_or_GT_or_E_pose_or_RGB_or_new_inference_inputs=0)


def run(*, central_plan, expected_central_sha256, expected_preparation_sha256,
        root_CAL_mask_GO=False, output=OUTPUT):
    _central(central_plan,expected_central_sha256,root_GO=root_CAL_mask_GO)
    if not _sha(expected_preparation_sha256):
        raise ValueError('Explicit exact prepared-stage SHA required')
    output=Path(output).resolve();prepared_path=output/'PREPARATION.json'
    if C.sha(prepared_path)!=expected_preparation_sha256:
        raise ValueError('Exact prepared CAL46 stage required')
    prepared=C.read(prepared_path);seal=C.read(output/'PREPARATION_SEAL.json')
    if (seal['preparation_sha256']!=expected_preparation_sha256
            or prepared['central_plan']!=dict(path=_relative(central_plan),sha256=expected_central_sha256)
            or prepared['selection_profile']!=LCC.SELECTION_PROFILE):
        raise ValueError('Same prepared profile/central binding required')
    for key in ('source_sha256','input_sha256','protected_user_sha256','original_CAL_file_sha256'):
        _verify(prepared[key])
    execution=output/'EXECUTION_START.json'
    if execution.exists():
        raise FileExistsError('Never restart or overwrite a partial attempt')
    threads={key:os.environ.get(key) for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}
    if any(value!='1' for value in threads.values()):
        raise ValueError('The ROOT operation must start with OMP/OpenBLAS/MKL thread environment equal to1')
    C.write(execution,dict(preparation_sha256=expected_preparation_sha256,central_sha256=expected_central_sha256,
        workers=1,verified_thread_environment=threads))
    (output/'rows').mkdir();summaries=[];attempted=None
    try:
        packets=CAL_prefix(C.rows(POSITIVE/'INPUTS.jsonl.gz'),prepared['ordered_CAL_ids'])
        for number,(packet,binding) in enumerate(zip(packets,prepared['cases'])):
            attempted=binding['id'];maskpath=_resolve(binding['target_mask_binding']['path'])
            if C.sha(maskpath)!=binding['target_mask_binding']['sha256']:
                raise ValueError('Original cached mask bytes changed')
            with np.load(maskpath,allow_pickle=False) as arrays:
                mask=arrays['pallet_mask_clean'].copy()
            row=process_one(packet,mask,binding)
            path=output/'rows'/f'{number:02d}.json';C.write(path,row)
            if not row['union_exact_match']:
                raise ValueError('Union differs from original saved R2 observation; mismatch row preserved')
            audit=row['largest_component']['foreground_component_selection']
            summaries.append(dict(id=binding['id'],partition='CAL',path=_relative(path),sha256=C.sha(path),
                canonical_row_sha256=_digest(row),union_exact_match=True,union_supported=row['original_union']['supported'],
                LCC_supported=row['largest_component']['supported'],
                original_component_count=audit['foreground_component_count'],selected_label=audit['selected_label'],
                discarded_area_px=audit['discarded_foreground_area_px'],tie_count=len(audit['maximum_area_tied_labels'])))
            print(json.dumps(dict(completed=len(summaries),total=46,id=binding['id'])),flush=True)
            del packet,mask,row
        if len(summaries)!=46 or [row['id'] for row in summaries]!=prepared['ordered_CAL_ids']:
            raise ValueError('Every CAL row, including empty/unsupported cases, must be retained')
        index=dict(entries=summaries,rows=46,ordered_ids=prepared['ordered_CAL_ids'],selection_profile=LCC.SELECTION_PROFILE)
        C.write(output/'CAL_ROW_INDEX.json',index)
        journal=output/'CAL_OUTLINE_OBSERVATIONS.jsonl.gz'
        with gzip.open(journal,'xt',encoding='utf-8',compresslevel=1) as stream:
            for summary in summaries:
                path=_resolve(summary['path'])
                if C.sha(path)!=summary['sha256']:
                    raise ValueError('Completed observation row changed')
                stream.write(json.dumps(C.finite(C.read(path)),ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
        for number,row in enumerate(C.rows(journal)):
            if number>=46 or row['id']!=summaries[number]['id'] or _digest(row)!=summaries[number]['canonical_row_sha256']:
                raise ValueError('Canonical saved-row roundtrip mismatch')
        if number!=45:
            raise ValueError('Ordered46 journal roundtrip required')
        C.write(output/'SUMMARY.json',dict(rows=46,ordered_ids=prepared['ordered_CAL_ids'],summaries=summaries,
            union_exact_match_count=46,empty_or_unsupported_rows_retained=True,primary_rows_decoded=0,
            compatible_CAL_reference_inventory=44,reference_or_label_decodes=0,model_or_pose_calls=0))
        for key in ('source_sha256','input_sha256','protected_user_sha256','original_CAL_file_sha256'):
            _verify(prepared[key])
        assets={_relative(path):C.sha(path) for path in output.rglob('*') if path.is_file()}
        C.write(output/'CAL_OBSERVATION_SEAL.json',dict(schema=SCHEMA,rows=46,ordered_ids=prepared['ordered_CAL_ids'],
            preparation_sha256=expected_preparation_sha256,protocol_sha256=expected_preparation_sha256,
            central_sha256=expected_central_sha256,selection_profile=LCC.SELECTION_PROFILE,
            observations_sha256=C.sha(journal),row_index_sha256=C.sha(output/'CAL_ROW_INDEX.json'),
            assets=assets,source_sha256=prepared['source_sha256'],
            input_sha256=prepared['input_sha256'],protected_user_sha256=prepared['protected_user_sha256'],
            original_CAL_file_sha256=prepared['original_CAL_file_sha256'],union_exact_match_count=46,union_exact_match_rows=46,
            RGB_model_NN_PnP_GT_reference_or_primary_decodes=0,coefficient_noise_physics_or_seed_changes=0))
    except BaseException as error:
        C.write(output/'FAILURE_RECEIPT.json',dict(exception=type(error).__name__,reason=str(error),
            completed_rows=len(summaries),attempted_id=attempted,attempt_preserved=True,no_silent_source_or_math_correction=True))
        raise
    return dict(rows=46,output=_relative(output),seal_sha256=C.sha(output/'CAL_OBSERVATION_SEAL.json'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('prepare','run'))
    parser.add_argument('--central-plan',required=True);parser.add_argument('--central-sha256',required=True)
    parser.add_argument('--output',type=Path,default=OUTPUT);parser.add_argument('--preparation-sha256')
    parser.add_argument('--root-metadata-GO',action='store_true');parser.add_argument('--root-CAL-mask-GO',action='store_true')
    args=parser.parse_args();kwargs=dict(central_plan=args.central_plan,expected_central_sha256=args.central_sha256,output=args.output)
    if args.stage=='prepare':result=prepare(**kwargs,root_metadata_GO=args.root_metadata_GO)
    else:result=run(**kwargs,expected_preparation_sha256=args.preparation_sha256,root_CAL_mask_GO=args.root_CAL_mask_GO)
    print(json.dumps(result))
