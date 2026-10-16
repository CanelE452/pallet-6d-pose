"""Bind the six preselected scenes and twelve stored poses, without inference."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

SCORE_NAMESPACE='_docs/experiments/pallet_selection_score_diagnosis_v1'
SOURCE_COMMIT='c56a4e92027a8e56f6704773f24d7c3da78082f1'
STATE_COMMIT='407de865445e4987614f50f893d2ccea78d507b6'
EXPECTED_IDS=('plastic_night_01:038630','wood_day_01:005286','wood_night_01:030698',
              'plastic_day_01:020955','plastic_day_01:014704','wood_night_01:030898')
REQUIRED_DOCUMENTS=('NEXT_CHANGE_PROPOSAL_KO.md','RESULTS_KO.md','OBJECTIVE_MAP.md','SELECTION_RULE_MAP.md',
 'SOURCE_BINDINGS.json','PACKAGING_BINDINGS.json','VALIDATION.json','REPRODUCE.md','PAIR_LEDGER.jsonl.gz',
 'PER_OBSERVATION_PAIR_LEDGER.csv.gz','gallery/REPRESENTATIVE_SELECTION.json','gallery/GALLERY_PROOF.json',
 'gallery/outline_witness/WITNESS_RECEIPT.json','gallery/outline_witness/INDEX_KO.md')
PERMISSION_INVENTORY='_docs/experiments/pallet_visibility_curriculum_20261013_v1/publication_followup_20261015_v11/repro_inventory_current_files_v2.jsonl.gz'
PERMISSION_INVENTORY_SHA256='75c5aaa33d534faefc028c3358d9be6b5cbe688599488a29d34478502fac47b5'


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):value.update(block)
    return value.hexdigest()


def array_sha(array):return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def save(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as stream:json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)


def selected_json_rows(path,ids):
    """Stream bytes but decode only the requested IDs; never evaluate a pose."""
    found={};tokens=[f'"id":"{fid}"' for fid in ids]+[f'"id": "{fid}"' for fid in ids]
    with gzip.open(path,'rt') as stream:
        for number,line in enumerate(stream,1):
            if not any(token in line for token in tokens):continue
            row=json.loads(line)
            if row.get('id') in ids:
                if row['id'] in found:raise ValueError('Duplicate selected frame '+row['id'])
                found[row['id']]=(number,row)
    if set(found)!=set(ids):raise ValueError('Selected original rows missing: '+str(set(ids)-set(found)))
    return found


def load_mask_arrays(case):
    if sha(case['mask_path'])!=case['mask_file_sha256']:raise ValueError('Original mask bytes changed')
    with np.load(case['mask_path'],allow_pickle=False) as source:
        return {key:source[key].copy() for key in source.files}


def permission_bindings(source_root,cases):
    """Existing publication permission evidence; no new release is requested."""
    source_root=Path(source_root);path=source_root/PERMISSION_INVENTORY
    if sha(path)!=PERMISSION_INVENTORY_SHA256:raise ValueError('Published original permission inventory changed')
    wanted={str(Path(case[key]).relative_to(source_root)) for case in cases for key in ('image_path','mask_path')}
    records={}
    with gzip.open(path,'rt') as stream:
        for line in stream:
            row=json.loads(line)
            if row['relpath'] in wanted:records[row['relpath']]=row
    if set(records)!=wanted:raise ValueError('Selected RGB/mask missing from existing publication inventory')
    for case in cases:
        for pathkey,digestkey in (('image_path','image_sha256'),('mask_path','mask_file_sha256')):
            entry=records[str(Path(case[pathkey]).relative_to(source_root))]
            if entry['sha256']!=case[digestkey] or not entry['prior_release_exact']:
                raise ValueError('Selected RGB/mask prior-release exact bytes not established')
    return dict(path=PERMISSION_INVENTORY,sha256=PERMISSION_INVENTORY_SHA256,selected_members=list(records.values()),
                original_RGB_and_masks_already_in_public_release=True,raw_RGB_or_NPZ_not_copied_to_new_git_artifacts=True)


def load_dense(case,score_private):
    """Read only the two original candidate rows in this one sealed frame."""
    root=Path(score_private)/'cost_v1';seal=read(root/'COST_SEAL.json')
    indexes={p['path']:p for p in seal['dense_frame_journals']}
    result=[]
    for pointer,panel in zip(case['witness']['original_dense_pointers'],case['panels']):
        path=root/pointer['path'];entry=indexes[pointer['path']]
        if entry['id']!=case['id'] or sha(path)!=entry['sha256']:raise ValueError('Dense selected frame binding differs')
        with gzip.open(path,'rt') as stream:
            line=next(value for number,value in enumerate(stream,1) if number==pointer['line_number'])
        row=json.loads(line)
        if (row['id'],row['candidate_id'])!=(case['id'],panel['candidate_id']):raise ValueError('Wrong original selected candidate pointer')
        if row['stage']!=panel['stage'] or row['family']!=panel['family']:raise ValueError('Stage/family cannot be mixed')
        if abs(row['objective']-panel['objective'])>1e-9:raise ValueError('Original candidate scalar differs')
        result.append(row)
    return result


def load_cases(source_root,score_private,state_private,*,include_dense=False):
    source_root,score_private,state_private=map(Path,(source_root,score_private,state_private))
    public=source_root/SCORE_NAMESPACE
    reps=read(public/'gallery/REPRESENTATIVE_SELECTION.json')['examples']
    if tuple(r['id'] for r in reps)!=EXPECTED_IDS:raise ValueError('Original representative six order changed')
    proof={c['id']:c for c in read(public/'gallery/GALLERY_PROOF.json')['cases'] if c['id'] in EXPECTED_IDS}
    witnesses=read(public/'gallery/outline_witness/WITNESS_RECEIPT.json')['examples']
    if tuple(w['id'] for w in witnesses)!=EXPECTED_IDS:raise ValueError('Original witness order differs')
    packets=selected_json_rows(state_private/'e_trace_v2/TYPED_PACKETS.jsonl.gz',EXPECTED_IDS)
    reference_path=state_private/'score_v2/FIXED_REFERENCES.json';reference_seal=read(state_private/'score_v2/SCORE_SEAL.json')
    if sha(reference_path)!=reference_seal['artifacts']['FIXED_REFERENCES.json']:raise ValueError('Original fixed references changed')
    references=read(reference_path)['references'];cases=[]
    for ordinal,(rep,witness) in enumerate(zip(reps,witnesses),1):
        fid=rep['id'];packet_line,packet=packets[fid];old=proof[fid]
        image=source_root/packet['image'];mask=source_root/packet['provenance']['original_mask_binding']['path']
        image_sha=sha(image);mask_sha=sha(mask);outline=packet['solver_input']['outline_observation']
        if image_sha!=packet['image_sha256'] or image_sha!=old['source_RGB_sha256']:raise ValueError('Source RGB SHA differs')
        if mask_sha!=packet['provenance']['original_mask_binding']['sha256'] or mask_sha!=outline['image_binding']['mask_file_sha256']:
            raise ValueError('Actual outline mask SHA differs')
        if outline['image_binding']['image_sha256']!=image_sha:raise ValueError('Mask outline RGB binding differs')
        from PIL import Image
        with Image.open(image) as original_RGB:
            raw_hw=[original_RGB.height,original_RGB.width]
        if raw_hw!=packet['solver_input']['image_hw'] or raw_hw!=outline['image_hw']:
            raise ValueError('Original RGB and saved observation canvas differ')
        frame_path=mask.parent.parent/'frames'/(fid.replace(':','__')+'.json')
        fresh=read(frame_path) if frame_path.exists() else None
        if fresh is not None and (fresh['id']!=fid or fresh['image_sha256']!=image_sha):raise ValueError('Fresh mask provenance row differs')
        if fresh is not None and (fresh['runtime']['initial_geometry']['K']!=packet['solver_input']['K'] or fresh['runtime']['initial_geometry']['image_hw']!=raw_hw):
            raise ValueError('Original observation K/canvas contract differs')
        panels=old['panels'][-2:]
        for panel in panels:
            family=packet['original_e_stage1_family_seed_bank'][panel['family']]['family']
            if not np.array_equal(panel['native_xyz'],family['native_xyz']) or not np.array_equal(panel['physical_registration_Q'],family['physical_registration_Q']):
                raise ValueError('Candidate family XYZ/Q differs')
            if panel['stage']!='G13':raise ValueError('Original six comparisons must remain G13')
        case=dict(id=fid,ordinal=ordinal,group=rep['group'],partition=packet['partition'],
            image_path=str(image),image_sha256=image_sha,mask_path=str(mask),mask_file_sha256=mask_sha,
            source_root=str(source_root),score_private=str(score_private),state_private=str(state_private),
            packet=packet,packet_line_number=packet_line,outline=outline,panels=panels,reference=references[packet['partition']][fid],
            reference_path=str(reference_path),reference_sha256=sha(reference_path),K=packet['solver_input']['K'],
            crop_xyxy=old['viewport_original_xyxy'],old_case=old,witness=witness,
            original_observation_frame_path=str(frame_path),original_observation_frame=fresh,
            provenance_status='BOUND_ORIGINAL_REMAINDER_FRAME' if fresh is not None else 'ORIGINAL_MASK_PROVENANCE_FRAME_UNAVAILABLE',
            pixel_transform=dict(original_hw=raw_hw,original_RGB_dimensions_checked=True,working_hw=outline['image_hw'],
                raw_to_observation='identity original 640x480 canvas; no crop/resize/letterbox in saved solver input',
                display_crop_xyxy=old['viewport_original_xyxy'],display_crop_only=True))
        if include_dense:case['dense']=load_dense(case,score_private)
        cases.append(case)
    return cases


def bind(source_root,score_private,state_private,output):
    from .audit_guard import SupportAuditOnly
    started,cpu=time.monotonic(),time.process_time()
    source_root,score_private,state_private,output=map(Path,(source_root,score_private,state_private,output))
    public=source_root/SCORE_NAMESPACE
    with SupportAuditOnly() as guard:
        cases=load_cases(source_root,score_private,state_private,include_dense=True)
        permission=permission_bindings(source_root,cases)
    documents=[]
    for name in REQUIRED_DOCUMENTS:
        path=public/name;digest=sha(path);relative=str(path.relative_to(source_root))
        original=subprocess.run(['git','show',SOURCE_COMMIT+':'+relative],cwd=source_root,capture_output=True,check=True).stdout
        if hashlib.sha256(original).hexdigest()!=digest:raise ValueError('Required original result artifact changed: '+relative)
        documents.append(dict(path=relative,sha256=digest,source_commit=SOURCE_COMMIT))
    inputs=[]
    for case in cases:
        arrays=load_mask_arrays(case);source_frame=case['original_observation_frame_path']
        pointers=case['witness']['original_dense_pointers']
        inputs.append(dict(id=case['id'],ordinal=case['ordinal'],group=case['group'],partition=case['partition'],
            RGB=dict(path=case['image_path'],sha256=case['image_sha256']),
            mask=dict(path=case['mask_path'],sha256=case['mask_file_sha256'],arrays=[dict(key=k,shape=list(v.shape),dtype=str(v.dtype),pixel_bytes_sha256=array_sha(v)) for k,v in arrays.items()]),
            original_observation_frame=dict(path=source_frame,sha256=sha(source_frame) if Path(source_frame).exists() else None,status=case['provenance_status']),
            packet=dict(path=str(state_private/'e_trace_v2/TYPED_PACKETS.jsonl.gz'),line_number=case['packet_line_number'],sha256=sha(state_private/'e_trace_v2/TYPED_PACKETS.jsonl.gz')),
            reference=dict(path=case['reference_path'],sha256=case['reference_sha256'],role='unchanged geometric reconstruction; not independent physical truth'),
            K=case['K'],candidates=[dict(candidate_id=p['candidate_id'],stage=p['stage'],family=p['family'],R=p['R'],t=p['t'],
                native_xyz=p['native_xyz'],registered_dimensions_xyz_m=np.ptp(np.asarray(p['native_xyz']),axis=0).tolist(),
                physical_registration_Q=p['physical_registration_Q'],original_dense_pointer=pointer) for p,pointer in zip(case['panels'],pointers)],
            pixel_transform=case['pixel_transform'],new_pose_or_cost_calls=0))
    result=dict(schema='OUTLINE_PHYSICAL_SUPPORT_INPUT_BINDINGS_V1',source_result_commit=SOURCE_COMMIT,upstream_state_result_commit=STATE_COMMIT,
        representative_ids=list(EXPECTED_IDS),representative_count=6,candidate_count=12,original_required_artifacts=documents,cases=inputs,
        source_files={str(Path(__file__).relative_to(source_root)):sha(__file__)},publication_permission=permission,
        missing_provenance_cases=[c['id'] for c in cases if c['original_observation_frame'] is None],
        original_data_or_prediction_rewritten=False,new_model_forwards=0,new_pose_solves=0,new_cost_evaluations=0,
        source_mask_or_threshold_changed=False,semantic_owner_verified=False,runtime_guard=guard.receipt(),
        elapsed_seconds=time.monotonic()-started,process_cpu_seconds=time.process_time()-cpu)
    save(output/'INPUT_BINDINGS.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('source-root','score-private','state-private','output'):parser.add_argument('--'+key,required=True)
    args=parser.parse_args();value=bind(args.source_root,args.score_private,args.state_private,args.output)
    print(json.dumps({k:value[k] for k in ('representative_count','candidate_count','missing_provenance_cases','elapsed_seconds','process_cpu_seconds')},indent=2))
