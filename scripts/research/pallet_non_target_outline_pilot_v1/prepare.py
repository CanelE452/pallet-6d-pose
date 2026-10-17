"""Privileged read-only exporter: whitelist neutral inputs before selection.

The supplier receives RGB/prompts only. The selector receives no reference,
audit support category, candidate success flag, or historical comparison cost.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time

from PIL import Image
import numpy as np

IDS = ('plastic_night_01:038630','wood_day_01:005286','wood_night_01:030698',
       'plastic_day_01:020955','plastic_day_01:014704','wood_night_01:030898')
POSE = ('available','R_cf','native_t','centroid','R_physical','physical_registration_Q',
        'cf_extents','source','joint_feasible','fallback','fallback_pose_does_not_solve_OT',
        'fallback_pose_does_not_solve_U_regions')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def selected(path):
    result=[]
    with gzip.open(path,'rt') as f:
        for line in f:
            if not any(fid in line for fid in IDS):continue
            row=json.loads(line)
            if row.get('id') in IDS:result.append(row)
    return result


def save(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)


def rows(path,values):
    path=Path(path)
    with path.open('xb') as raw:
        if path.suffix=='.gz':
            with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as f:
                for row in values:f.write((json.dumps(row,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode())
        else:
            for row in values:raw.write((json.dumps(row,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode())


def subset(row,keys):return {k:row[k] for k in keys if k in row}


def reject_forbidden(value):
    if isinstance(value,dict):
        for k,v in value.items():
            if any(token in k.lower() for token in ('reference','oracle','ground_truth','success_candidate','final_reference','support_category')):
                raise ValueError('Forbidden neutral field '+k)
            reject_forbidden(v)
    elif isinstance(value,list):
        for v in value:reject_forbidden(v)


def run(source_root,score_private,state_private,output):
    started,cpu=time.monotonic(),time.process_time()
    source,score,state,out=map(Path,(source_root,score_private,state_private,output))
    out.mkdir(parents=True,exist_ok=False)
    paths=dict(frames=score/'selection_v1/FRAME_SELECTIONS.jsonl.gz',
        candidates=score/'selection_v1/RETAINED_CANDIDATES.jsonl.gz',
        packets=state/'e_trace_v2/TYPED_PACKETS.jsonl.gz',cost_seal=score/'cost_v1/COST_SEAL.json')
    fs={x['id']:x for x in selected(paths['frames'])}
    packets={x['id']:x for x in selected(paths['packets'])}
    raw_candidates=selected(paths['candidates'])
    if set(fs)!=set(IDS)or set(packets)!=set(IDS):raise ValueError('Exact six original frames required')
    seal=read(paths['cost_seal']);dense_map={x['id']:x for x in seal['dense_frame_journals']}
    frames=[];candidates=[];dense_bindings=[];rgb=[];original_assets=[]
    for ordinal,fid in enumerate(IDS,1):
        original,packet=fs[fid],packets[fid]
        observation=packet['solver_input']['outline_observation']
        stages={}
        for stage,value in original['stages'].items():
            families=[]
            for fam in value['families']:
                neutral=subset(fam,('stage','family','solver_schema','accepted','fallback','status',
                    'winner_candidate_id','original_objective','retained_count'))
                neutral['published_actual_pose']=subset(fam['published_actual_pose'],POSE)
                neutral['published_fallback_audit']=subset(fam['published_fallback_audit'],
                    ('applied','pose_source','reason','O_T_reported_as_solved'))
                families.append(neutral)
            stages[stage]=dict(selected_family=value['selected_family'],families=families)
        frame=subset(original,('id','index','partition','session','image_sha256','states',
            'selected_family','selected_candidate_id','original_e_accepted','cached_g13_no_typed_revocation'))
        frame.update(ordinal=ordinal,image=packet['image'],K=packet['solver_input']['K'],
            image_hw=packet['solver_input']['image_hw'],stages=stages,
            original_e_actual_pose=subset(original['original_e_actual_pose'],POSE),
            outlines={stage:{fam['family']:observation for fam in value['families']}for stage,value in stages.items()})
        frames.append(frame)
        entry=dense_map[fid];dense_path=score/'cost_v1'/entry['path']
        if sha(dense_path)!=entry['sha256']:raise ValueError('Original sealed dense frame changed')
        dense={r['candidate_id']:r for r in selected(dense_path)}
        own=[r for r in raw_candidates if r['id']==fid]
        if set(dense)!={r['candidate_id']for r in own}:raise ValueError('All retained candidates must have dense rows')
        for row in own:
            d=dense[row['candidate_id']]
            for k in ('id','stage','family','candidate_id'):
                if d[k]!=row[k]:raise ValueError('Composite candidate identity mismatch')
            neutral=subset(row,('id','index','partition','session','candidate_id','stage','family',
                'source_order','retained','original_outcome_order','original_branch','solver_schema',
                'accepted_family','R','t','native_xyz','physical_registration_Q','cf_extents'))
            neutral['original_record']=subset(row['original_record'],('objective','initialization',
                'all_OT_individual_constraints_pass','all_U_individual_constraints_pass','all_partial_B_tangent_constraints_pass'))
            neutral['pose']=subset(row['pose'],POSE)
            neutral['dense']=subset(d,('objective','weighted_components','outline_active','outline_valid',
                'recomputed_joint_feasible','original_joint_feasible','minimum_constraint_slack','constraints','actual_backend'))
            # Existing per-direction arrays only, no re-fitting or cost recomputation.
            neutral['dense']['per_observation']=[o for o in d['per_observation']if o['loss_family']=='outline']
            candidates.append(neutral)
        dense_bindings.append(dict(id=fid,path=str(dense_path),sha256=sha(dense_path),retained_candidates=len(own)))
        image=source/packet['image']
        if sha(image)!=packet['image_sha256']:raise ValueError('Source RGB changed')
        pixels=np.asarray(Image.open(image).convert('RGB'))
        if list(pixels.shape[:2])!=frame['image_hw']:raise ValueError('RGB canvas mismatch')
        rgb.append(dict(schema='NON_TARGET_RGB_NEUTRAL_V1',image_id=fid,rgb_path=str(image),
            rgb_file_sha256=sha(image),rgb_pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),image_hw=frame['image_hw']))
        mask=source/packet['provenance']['original_mask_binding']['path']
        if sha(mask)!=packet['provenance']['original_mask_binding']['sha256']:raise ValueError('Original target changed')
        original_assets.extend([dict(path=str(image),sha256=sha(image),kind='original_RGB'),
                                dict(path=str(mask),sha256=sha(mask),kind='original_target_mask')])
    reject_forbidden(frames);reject_forbidden(candidates);reject_forbidden(rgb)
    if len(candidates)!=351:raise ValueError('Expected original six retained bank 351')
    rows(out/'NEUTRAL_FRAMES.jsonl.gz',frames);rows(out/'NEUTRAL_CANDIDATES.jsonl.gz',candidates)
    rows(out/'NEUTRAL_RGB.jsonl',rgb)
    package=dict(schema='NEUTRAL_FIXED_BANK_PACKAGE_V1',ids=list(IDS),
        frames=dict(path='NEUTRAL_FRAMES.jsonl.gz',sha256=sha(out/'NEUTRAL_FRAMES.jsonl.gz')),
        candidates=dict(path='NEUTRAL_CANDIDATES.jsonl.gz',sha256=sha(out/'NEUTRAL_CANDIDATES.jsonl.gz')),
        RGB=dict(path='NEUTRAL_RGB.jsonl',sha256=sha(out/'NEUTRAL_RGB.jsonl')),
        frame_count=6,retained_candidate_count=351,allowed_field_whitelist=True,
        reference_or_success_fields_exported=False,new_model_forward=0,new_pose_calls=0)
    save(out/'NEUTRAL_PACKAGE.json',package)
    save(out/'EXPORT_RECEIPT.json',dict(schema='PRIVILEGED_WHITELIST_EXPORT_RECEIPT_V1',
        inputs={str(v):sha(v)for v in paths.values()},selected_dense_bindings=dense_bindings,
        original_assets=original_assets,neutral_package_sha256=sha(out/'NEUTRAL_PACKAGE.json'),
        neutral_forbidden_keys_absent=True,previous_support_decisions_read=False,
        references_read=False,new_model_forward=0,new_pose_calls=0,
        elapsed_seconds=time.monotonic()-started,process_cpu_seconds=time.process_time()-cpu))
    return package


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('source-root','score-private','state-private','output'):p.add_argument('--'+k,required=True)
    a=p.parse_args();r=run(a.source_root,a.score_private,a.state_private,a.output)
    print(json.dumps({k:r[k]for k in ('frame_count','retained_candidate_count','reference_or_success_fields_exported')}))
