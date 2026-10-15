"""Append REST192 to immutable CAL46 using the single reviewed LCC8 rule.

No dynamic code construction. The small process_one body is the CAL source
with only its two partition metadata expressions generalized; an AST test
normalizes those two spots and proves the remaining body identical.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import gzip
import hashlib
from itertools import zip_longest
import json
import os
from pathlib import Path
import shutil

import numpy as np

from . import observe_largest_component_global_outline_CAL_v4_33 as CAL
from . import declare_largest_component_global_outline_v4_33 as DECLARE

C, LCC, FLAT = CAL.C, CAL.LCC, CAL.FLAT
OUTPUT, POSITIVE = CAL.OUTPUT, CAL.POSITIVE
SELECTION_PROFILE, BASELINE_PROFILE = CAL.SELECTION_PROFILE, CAL.BASELINE_PROFILE
SCHEMA = 'SEALED_REST192_UNION_AND_LCC8_GLOBAL_OUTLINE_OBSERVATIONS_V33'
MASK_FOLDERS = (CAL.MASKS, C.DOC/'nonsevere_v1/remainder_after_test49_v1/fresh_observations_v4_14/masks')
_digest, _sha, _relative, _resolve, _union, _verify = CAL._digest, CAL._sha, CAL._relative, CAL._resolve, CAL._union, CAL._verify


def _sources():
    paths=[Path(__file__),Path(__file__).with_name('test_observe_largest_component_global_outline_REST_v4_33.py'),Path(DECLARE.__file__)]
    return _union(CAL._sources(),{_relative(path):C.sha(path) for path in paths})


def _gate(binding, central_sha256, output):
    DECLARE._validate_binding(binding)
    path=_resolve(binding['path'])
    if C.sha(path)!=binding['sha256']:
        raise ValueError('Exact caller-pinned ROOT CAL independent gate required')
    gate=C.read(path)
    if (gate['schema']!='ROOT_CAL46_LCC33_INDEPENDENT_GATE_V1' or gate['status']!='PASS'
            or gate['rows']!=46 or gate['failure_count']!=0 or gate['union_exact_match_rows']!=46
            or gate['central_plan_sha256']!=central_sha256):
        raise ValueError('ROOT independent CAL46 PASS/zero-failure/union exactness required')
    expected=dict(path=_relative(Path(output)/'CAL_OBSERVATION_SEAL.json'),sha256=C.sha(Path(output)/'CAL_OBSERVATION_SEAL.json'))
    if gate['CAL_observation_seal']!=expected:
        raise ValueError('ROOT gate must bind the exact current CAL observation seal')
    if C.read(Path(output)/'CAL_OBSERVATION_SEAL.json')['central_sha256']!=central_sha256:
        raise ValueError('CAL observation and ROOT gate must bind the same central declaration')
    for key in ('independent_audit','independent_audit_seal'):
        DECLARE._validate_binding(gate[key])
        if C.sha(_resolve(gate[key]['path']))!=gate[key]['sha256']:
            raise ValueError('ROOT independent audit byte binding changed')
    return gate


def _stage(central_plan, expected_central_sha256, gate_binding, root_GO):
    if root_GO is not True:
        raise PermissionError('Explicit ROOT REST stage GO before any IO')
    if not _sha(expected_central_sha256):
        raise ValueError('Explicit central33 SHA before IO')
    DECLARE._validate_binding(gate_binding)
    return CAL._central(central_plan,expected_central_sha256,root_GO=True)


def _CAL_snapshot(output, central):
    output=Path(output);seal=C.read(output/'CAL_OBSERVATION_SEAL.json')
    index=C.read(output/'CAL_ROW_INDEX.json');prepared=C.read(output/'PREPARATION.json')
    ids=central['ordered_ids'][:46]
    if (seal['rows']!=46 or seal['ordered_ids']!=ids or seal['selection_profile']!=SELECTION_PROFILE
            or seal['union_exact_match_rows']!=46 or seal['protocol_sha256']!=C.sha(output/'PREPARATION.json')
            or seal['row_index_sha256']!=C.sha(output/'CAL_ROW_INDEX.json')
            or seal['observations_sha256']!=C.sha(output/'CAL_OUTLINE_OBSERVATIONS.jsonl.gz')
            or [entry['id'] for entry in index['entries']]!=ids
            or any(entry['partition']!='CAL' for entry in index['entries'])
            or prepared['central_plan']['sha256']!=C.sha(_resolve(prepared['central_plan']['path']))):
        raise ValueError('Complete unchanged ordered CAL46 with exact union controls required')
    names=('PREPARATION.json','PREPARATION_SEAL.json','EXECUTION_START.json','CAL_ROW_INDEX.json',
        'CAL_OUTLINE_OBSERVATIONS.jsonl.gz','SUMMARY.json','CAL_OBSERVATION_SEAL.json')
    paths=[output/name for name in names]
    for entry in index['entries']:
        path=_resolve(entry['path'])
        if not path.is_relative_to((output/'rows').resolve()) or C.sha(path)!=entry['sha256']:
            raise ValueError('CAL observation row changed; never regenerate it in REST')
        paths.append(path)
    if len(paths)!=53 or len(set(paths))!=53:
        raise ValueError('Exactly53 new LCC CAL files must stay immutable')
    immutable={_relative(path):C.sha(path) for path in paths}
    for key in ('source_sha256','input_sha256','protected_user_sha256','original_CAL_file_sha256'):
        _verify(prepared[key])
    return prepared,seal,index,immutable


def join_REST_bindings(protocol, manifests):
    """Flat metadata from the original95 and fresh143 mask archives only."""
    records=protocol['records'][46:];ids=protocol['ordered_ids'][46:]
    if (len(records)!=192 or len(ids)!=192 or len(set(ids))!=192
            or [row['id'] for row in records]!=ids
            or [row['partition'] for row in records]!=['TEST']*49+['REMAINDER']*143):
        raise ValueError('Exact ordered TEST49 plus REMAINDER143 required')
    wanted={row['id']:row for row in records};found={};scanned=0;skipped=0
    allowed={'id','partition','image','image_sha256','mask_npz','mask_file_sha256'}
    for folder,text in manifests:
        for raw in FLAT._records(text):
            scanned+=1;identifier=FLAT.flat_fields(raw,{'id'})['id']
            if identifier not in wanted:
                skipped+=1;continue
            value=FLAT.flat_fields(raw,allowed);record=wanted[identifier]
            if identifier in found or (value.get('partition') is not None and value['partition']!=record['partition']):
                raise ValueError('Duplicate or contradictory REST mask metadata')
            name=value['mask_npz']
            if not isinstance(name,str) or Path(name).name!=name or name in ('.','..'):
                raise ValueError('Original mask basename required')
            found[identifier]=dict(id=identifier,partition=record['partition'],
                image_binding=dict(path=value['image'],sha256=value['image_sha256']),
                target_mask_binding=dict(path=_relative(Path(folder)/name),sha256=value['mask_file_sha256']),
                positive_packet_sha256=record['positive_packet_sha256'])
    if set(found)!=set(wanted):
        raise ValueError('Every REST mask metadata binding, including empty masks, is required')
    return [found[identifier] for identifier in ids],dict(manifest_records_scanned=scanned,
        unrequested_records_skipped_after_id=skipped,nonREST_payload_fields_decoded=0,
        source_pose_state_keypoint_or_line_fields_decoded=0)


def prepare(*, central_plan, expected_central_sha256, gate_binding, source_peer_binding,
        root_metadata_GO=False, output=OUTPUT):
    central=_stage(central_plan,expected_central_sha256,gate_binding,root_metadata_GO)
    gate=_gate(gate_binding,expected_central_sha256,output)
    peer=DECLARE._peer(source_peer_binding,(Path(__file__),Path(__file__).with_name('test_observe_largest_component_global_outline_REST_v4_33.py')))
    output=Path(output).resolve()
    if (output/'REST_PREPARATION.json').exists() or (output/'REST_PREPARATION_SEAL.json').exists():
        raise FileExistsError('Preserve every REST preparation attempt')
    cal,_,_,immutable=_CAL_snapshot(output,central)
    protocol=C.read(POSITIVE/'PROTOCOL.json');input_seal=C.read(POSITIVE/'INPUT_SEAL.json')
    if (input_seal['ordered_ids']!=central['ordered_ids'] or protocol['ordered_ids']!=central['ordered_ids']
            or C.sha(POSITIVE/'PROTOCOL.json')!=input_seal['protocol_sha256']
            or C.sha(POSITIVE/'INPUTS.jsonl.gz')!=input_seal['input_sha256']):
        raise ValueError('Exact original positive packet archive required')
    manifests=[];files=[];expected_records=0
    for folder in MASK_FOLDERS:
        seal=C.read(folder/'SEAL.json')
        if C.sha(folder/'RECORDS.json')!=seal['records_sha256'] or C.sha(folder/'PROTOCOL.json')!=seal['protocol_sha256']:
            raise ValueError('Original mask metadata changed')
        manifests.append((folder,(folder/'RECORDS.json').read_text()));expected_records+=seal['records']
        files += [folder/name for name in ('SEAL.json','RECORDS.json','PROTOCOL.json')]
    cases,inventory=join_REST_bindings(protocol,manifests)
    if inventory['manifest_records_scanned']!=expected_records:
        raise ValueError('Complete original mask metadata inventories required')
    source=_union(central['source_sha256'],cal['source_sha256'],_sources())
    files += [Path(central_plan),_resolve(gate_binding['path']),peer,POSITIVE/'PROTOCOL.json',POSITIVE/'INPUT_SEAL.json',POSITIVE/'INPUTS.jsonl.gz']
    files += [_resolve(gate[key]['path']) for key in ('independent_audit','independent_audit_seal')]
    inputs=_union(central['input_sha256'],cal['input_sha256'],immutable,{_relative(path):C.sha(path) for path in files},
        {case['target_mask_binding']['path']:case['target_mask_binding']['sha256'] for case in cases})
    protected=central['protected_user_sha256'];_verify(source,inputs,protected,cal['original_CAL_file_sha256'])
    prepared=dict(schema=SCHEMA,selection_profile=SELECTION_PROFILE,
        central_plan_sha256=expected_central_sha256,central_plan_path=_relative(central_plan),
        gate_binding=gate_binding,source_peer_binding=source_peer_binding,cases=cases,
        ordered_ids=central['ordered_ids'][46:],source_sha256=source,input_sha256=inputs,
        protected_user_sha256=protected,original_CAL_file_sha256=cal['original_CAL_file_sha256'],
        immutable_LCC_CAL_file_sha256=immutable,inventory=inventory,workers=1,
        memory='one REST packet/mask plus derived arrays and serialization temporaries; small192 summaries',
        CAL_positive_packet_and_mask_decodes=0,actual_REST_packet_mask_or_label_decodes_in_prepare=0)
    C.write(output/'REST_PREPARATION.json',prepared)
    C.write(output/'REST_PREPARATION_SEAL.json',dict(preparation_sha256=C.sha(output/'REST_PREPARATION.json'),
        source_sha256=source,input_sha256=inputs,protected_user_sha256=protected,
        immutable_LCC_CAL_file_sha256=immutable,original_CAL_file_sha256=cal['original_CAL_file_sha256']))
    return dict(preparation_sha256=C.sha(output/'REST_PREPARATION.json'),rows=192)


def REST_packets(path, bindings):
    """Skip CAL46 serialized lines without JSON-decoding their packet payloads."""
    if len(bindings)!=192 or [row['partition'] for row in bindings]!=['TEST']*49+['REMAINDER']*143:
        raise ValueError('Uniform ordered192 bindings required')
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        for _ in range(46):
            next(stream)  # Opaque text only; no CAL packet JSON decode.
        for binding in bindings:
            packet=json.loads(next(stream))
            if packet['identifiers']['id']!=binding['id'] or packet['identifiers']['partition']!=binding['partition']:
                raise ValueError('Exact REST packet identity/order required')
            yield packet
            del packet
        if next(stream,None) is not None:
            raise ValueError('No extra positive packet rows permitted')


def process_one(packet, mask, binding):
    """Pure synthetic-testable join; no pose/state/keypoint fields are queried."""
    if packet['identifiers']['partition']!=binding['partition'] or packet['identifiers']['id']!=binding['id']:
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
    return dict(schema=SCHEMA,id=binding['id'],partition=binding['partition'],row_retained=True,
        selection_profile=LCC.SELECTION_PROFILE,baseline_profile=LCC.BASELINE_PROFILE,
        target_mask_binding=deepcopy(file),positive_packet_sha256=binding['positive_packet_sha256'],
        stored_union_sha256=_digest(saved),regenerated_union_sha256=_digest(union),
        union_exact_match=bool(same),union_per_field_equal=per_field,
        stored_union_on_mismatch=None if same else saved,
        original_union=bundle['original_union'],largest_component=bundle['largest_component'],
        C_coefficient_or_noise_physics_seed_or_mask_bytes_changed=False,
        reference_or_GT_or_E_pose_or_RGB_or_new_inference_inputs=0)


def _prepared(central_plan, central_sha256, gate_binding, expected, root_GO, output):
    central=_stage(central_plan,central_sha256,gate_binding,root_GO)
    _gate(gate_binding,central_sha256,output)
    if not _sha(expected) or C.sha(Path(output)/'REST_PREPARATION.json')!=expected:
        raise ValueError('Explicit exact REST preparation SHA required')
    value=C.read(Path(output)/'REST_PREPARATION.json')
    if (value['central_plan_sha256']!=central_sha256 or value['gate_binding']!=gate_binding
            or value['ordered_ids']!=central['ordered_ids'][46:]
            or C.read(Path(output)/'REST_PREPARATION_SEAL.json')['preparation_sha256']!=expected):
        raise ValueError('Prepared REST central/gate/order changed')
    for key in ('source_sha256','input_sha256','protected_user_sha256','original_CAL_file_sha256','immutable_LCC_CAL_file_sha256'):
        _verify(value[key])
    return central,value


def run(*, central_plan, expected_central_sha256, gate_binding, expected_preparation_sha256,
        root_REST_mask_GO=False, output=OUTPUT):
    if root_REST_mask_GO is not True:
        raise PermissionError('Explicit ROOT REST execution GO before any IO')
    output=Path(output).resolve()
    _,prepared=_prepared(central_plan,expected_central_sha256,gate_binding,expected_preparation_sha256,root_REST_mask_GO,output)
    if (output/'REST_EXECUTION_START.json').exists():
        raise FileExistsError('Preserve every REST attempt; no automatic resume')
    threads={key:os.environ.get(key) for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}
    if any(value!='1' for value in threads.values()):
        raise ValueError('OMP/OpenBLAS/MKL thread environment must all equal1')
    C.write(output/'REST_EXECUTION_START.json',dict(preparation_sha256=expected_preparation_sha256,workers=1,verified_thread_environment=threads))
    (output/'rows_REST').mkdir();entries=[];attempted=None
    try:
        for number,(packet,binding) in enumerate(zip(REST_packets(POSITIVE/'INPUTS.jsonl.gz',prepared['cases']),prepared['cases'])):
            attempted=binding['id']
            if binding['partition'] not in ('TEST','REMAINDER'):
                raise ValueError('REST stages must never relabel CAL or drop rows')
            maskpath=_resolve(binding['target_mask_binding']['path'])
            if C.sha(maskpath)!=binding['target_mask_binding']['sha256']:
                raise ValueError('Original REST cached mask bytes changed')
            with np.load(maskpath,allow_pickle=False) as arrays:
                mask=arrays['pallet_mask_clean'].copy()
            row=process_one(packet,mask,binding);path=output/'rows_REST'/f'{number:03d}.json';C.write(path,row)
            if not row['union_exact_match']:
                raise ValueError('Original union mismatch; comparison evidence row preserved')
            audit=row['largest_component']['foreground_component_selection']
            entries.append(dict(id=binding['id'],partition=binding['partition'],path=_relative(path),sha256=C.sha(path),
                canonical_row_sha256=_digest(row),union_exact_match=True,
                union_supported=row['original_union']['supported'],LCC_supported=row['largest_component']['supported'],
                original_component_count=audit['foreground_component_count'],selected_label=audit['selected_label'],
                discarded_area_px=audit['discarded_foreground_area_px'],tie_count=len(audit['maximum_area_tied_labels'])))
            print(json.dumps(dict(completed=len(entries),total=192,id=binding['id'])),flush=True)
            del packet,mask,row
        if len(entries)!=192 or [row['id'] for row in entries]!=prepared['ordered_ids']:
            raise ValueError('Every REST row must be retained')
        C.write(output/'REST_ROW_INDEX.json',dict(entries=entries,rows=192,ordered_ids=prepared['ordered_ids'],selection_profile=SELECTION_PROFILE))
        journal=output/'REST_OUTLINE_OBSERVATIONS.jsonl.gz'
        with gzip.open(journal,'xt',encoding='utf-8',compresslevel=1) as stream:
            for entry in entries:
                if C.sha(_resolve(entry['path']))!=entry['sha256']:
                    raise ValueError('Completed REST row changed')
                stream.write(json.dumps(C.finite(C.read(_resolve(entry['path']))),ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
        for number,row in enumerate(C.rows(journal)):
            if number>=192 or row['id']!=entries[number]['id'] or _digest(row)!=entries[number]['canonical_row_sha256']:
                raise ValueError('REST192 canonical roundtrip mismatch')
        if number!=191:
            raise ValueError('Complete REST192 roundtrip required')
        C.write(output/'REST_SUMMARY.json',dict(rows=192,ordered_ids=prepared['ordered_ids'],union_exact_match_rows=192,
            entries=entries,CAL_positive_packet_or_mask_decodes=0,reference_or_GT_NN_PnP_or_fit_calls=0))
        for key in ('source_sha256','input_sha256','protected_user_sha256','original_CAL_file_sha256','immutable_LCC_CAL_file_sha256'):
            _verify(prepared[key])
        C.write(output/'REST_OBSERVATION_SEAL.json',dict(schema=SCHEMA,rows=192,ordered_ids=prepared['ordered_ids'],
            selection_profile=SELECTION_PROFILE,protocol_sha256=expected_preparation_sha256,
            observations_sha256=C.sha(journal),row_index_sha256=C.sha(output/'REST_ROW_INDEX.json'),
            union_exact_match_rows=192,source_sha256=prepared['source_sha256'],input_sha256=prepared['input_sha256'],
            protected_user_sha256=prepared['protected_user_sha256'],immutable_LCC_CAL_file_sha256=prepared['immutable_LCC_CAL_file_sha256'],
            original_CAL_file_sha256=prepared['original_CAL_file_sha256']))
    except BaseException as error:
        C.write(output/'REST_FAILURE_RECEIPT.json',dict(exception=type(error).__name__,reason=str(error),completed_rows=len(entries),attempted_id=attempted,attempt_preserved=True))
        raise
    return dict(rows=192,seal_sha256=C.sha(output/'REST_OBSERVATION_SEAL.json'))


def finalize(*, central_plan, expected_central_sha256, gate_binding, expected_preparation_sha256,
        root_finalize_GO=False, output=OUTPUT):
    if root_finalize_GO is not True:
        raise PermissionError('Explicit ROOT finalization GO before any IO')
    output=Path(output).resolve()
    central,prepared=_prepared(central_plan,expected_central_sha256,gate_binding,expected_preparation_sha256,root_finalize_GO,output)
    names=('FINAL_PROTOCOL.json','OUTLINE_ROW_INDEX.json','OUTLINE_OBSERVATIONS.jsonl.gz','OUTLINE_OBSERVATION_SEAL.json')
    if any((output/name).exists() for name in names):
        raise FileExistsError('Preserve every finalization attempt')
    partitions={};indices=[];inputs={}
    for part,count,ids in (('CAL',46,central['ordered_ids'][:46]),('REST',192,central['ordered_ids'][46:])):
        seal=C.read(output/(part+'_OBSERVATION_SEAL.json'));index=C.read(output/(part+'_ROW_INDEX.json'))['entries']
        if (seal['rows']!=count or seal['ordered_ids']!=ids or seal['selection_profile']!=SELECTION_PROFILE
                or seal['union_exact_match_rows']!=count or [entry['id'] for entry in index]!=ids
                or seal['protocol_sha256']!=C.sha(output/('PREPARATION.json' if part=='CAL' else 'REST_PREPARATION.json'))
                or seal['observations_sha256']!=C.sha(output/(part+'_OUTLINE_OBSERVATIONS.jsonl.gz'))
                or seal['row_index_sha256']!=C.sha(output/(part+'_ROW_INDEX.json'))):
            raise ValueError('Both complete partition seals required')
        partitions[part]=C.sha(output/(part+'_OBSERVATION_SEAL.json'));indices.extend(index)
        for name in (part+'_OBSERVATION_SEAL.json',part+'_ROW_INDEX.json',part+'_OUTLINE_OBSERVATIONS.jsonl.gz'):
            inputs[_relative(output/name)]=C.sha(output/name)
    if len(indices)!=238 or len({entry['id'] for entry in indices})!=238 or Counter(entry['partition'] for entry in indices)!=Counter(CAL=46,TEST=49,REMAINDER=143):
        raise ValueError('Exact CAL46 TEST49 REMAINDER143, no dropped or extra rows')
    for entry in indices:
        if C.sha(_resolve(entry['path']))!=entry['sha256']:
            raise ValueError('A completed partition row changed')
    final=dict(schema='SEALED_ALL238_LCC8_OUTLINE_ONLY_PROTOCOL_V33',
        central_plan_sha256=expected_central_sha256,ordered_ids=central['ordered_ids'],selection_profile=SELECTION_PROFILE,
        source_sha256=prepared['source_sha256'],input_sha256=_union(prepared['input_sha256'],inputs),
        protected_user_sha256=prepared['protected_user_sha256'],original_CAL_file_sha256=prepared['original_CAL_file_sha256'],
        immutable_LCC_CAL_file_sha256=prepared['immutable_LCC_CAL_file_sha256'],partition_seals=partitions,
        final_gzip_members=['CAL_OUTLINE_OBSERVATIONS.jsonl.gz','REST_OUTLINE_OBSERVATIONS.jsonl.gz'],
        original_CAL_positive_mask_decodes_in_REST_or_final=0,new_CAL_outline_outputs_read_for_final_canonical_validation=46,
        line_noise_physics_seed_C_coefficient_and_parent_outputs_changed=False)
    C.write(output/'FINAL_PROTOCOL.json',final);C.write(output/'OUTLINE_ROW_INDEX.json',dict(entries=indices,rows=238,selection_profile=SELECTION_PROFILE))
    journal=output/'OUTLINE_OBSERVATIONS.jsonl.gz'
    with journal.open('xb') as destination:
        for part in ('CAL','REST'):
            with (output/(part+'_OUTLINE_OBSERVATIONS.jsonl.gz')).open('rb') as stream:
                shutil.copyfileobj(stream,destination,length=1024*1024)
    verified=0
    for row,entry in zip_longest(C.rows(journal),indices):
        if row is None or entry is None or row['id']!=entry['id'] or row['partition']!=entry['partition'] or row['selection_profile']!=SELECTION_PROFILE or row['union_exact_match'] is not True or _digest(row)!=entry['canonical_row_sha256']:
            raise ValueError('All238 canonical two-member gzip validation failed')
        verified+=1
    for key in ('source_sha256','input_sha256','protected_user_sha256','original_CAL_file_sha256','immutable_LCC_CAL_file_sha256'):
        _verify(final[key])
    C.write(output/'OUTLINE_OBSERVATION_SEAL.json',dict(schema='SEALED_ALL238_LCC8_OUTLINE_ONLY_V33',
        rows=238,ordered_ids=central['ordered_ids'],partitions=dict(CAL=46,TEST=49,REMAINDER=143),
        selection_profile=SELECTION_PROFILE,protocol_sha256=C.sha(output/'FINAL_PROTOCOL.json'),
        observations_sha256=C.sha(journal),row_index_sha256=C.sha(output/'OUTLINE_ROW_INDEX.json'),
        partition_seals=partitions,canonical_validation=dict(rows=238,canonical_row_digests_verified=verified,missing_duplicate_or_extra_rows=0)))
    return dict(rows=verified,seal_sha256=C.sha(output/'OUTLINE_OBSERVATION_SEAL.json'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('prepare','run','finalize'))
    parser.add_argument('--central-plan',required=True);parser.add_argument('--central-sha256',required=True)
    parser.add_argument('--CAL-independent-gate',required=True);parser.add_argument('--CAL-independent-gate-sha256',required=True)
    parser.add_argument('--source-peer');parser.add_argument('--source-peer-sha256');parser.add_argument('--preparation-sha256')
    parser.add_argument('--output',type=Path,default=OUTPUT);parser.add_argument('--root-metadata-GO',action='store_true')
    parser.add_argument('--root-REST-mask-GO',action='store_true');parser.add_argument('--root-finalize-GO',action='store_true')
    args=parser.parse_args();kwargs=dict(central_plan=args.central_plan,expected_central_sha256=args.central_sha256,
        gate_binding=dict(path=args.CAL_independent_gate,sha256=args.CAL_independent_gate_sha256),output=args.output)
    if args.stage=='prepare':result=prepare(**kwargs,source_peer_binding=dict(path=args.source_peer,sha256=args.source_peer_sha256),root_metadata_GO=args.root_metadata_GO)
    elif args.stage=='run':result=run(**kwargs,expected_preparation_sha256=args.preparation_sha256,root_REST_mask_GO=args.root_REST_mask_GO)
    else:result=finalize(**kwargs,expected_preparation_sha256=args.preparation_sha256,root_finalize_GO=args.root_finalize_GO)
    print(json.dumps(result))
