"""Root actual entry後のみ、全5406 native-scene datasetを独立再構成。fit/modelなし。"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import traceback
import numpy as np
from independent_scene_math import need,sha,lift_scene,teacher_proof,sample_all,changed_counts
from qa_io import identity,safe,raw,verify,binding,read_json,read_jsonl,iter_jsonl,write_json,pinned_math,LRU,decode_base,map_npy

ROLES=('COVER','ADJACENT_SHAM')
ROOT_PROTOCOL_SHA='03177adde9dca4e91e04663bd0a7a1976d176a876c114fb67844acc6f27ecc2c'
SHAPES={
 'CF_NODE_RGB96.npy':('uint8',(5406,2,8,3,96,96)),
 'CF_CONTEXT_RGB224_F32.npy':('float32',(5406,2,3,224,224)),
 'PARENT_INDICES.npy':('int64',(5406,)), 'PAIR_TARGETS.npy':('int8',(5406,2)),
 'CONSTRUCTED.npy':('bool',(5406,)), 'PROOF_PASS.npy':('bool',(5406,)),
 'GROUP_NODE_VALID96.npy':('bool',(1198,8,96,96)),
 'GROUP_CONTEXT_VALID224.npy':('bool',(1198,224,224)),
 'PAIR_LAYER_RGB96.npy':('uint8',(5406,2,3,96,96)),
 'PAIR_ALPHA_PACKBITS.npy':('uint8',(5406,2,1152))}
GEO={'group_index':('int64',(1198,)), 'q8':('float64',(1198,8,2)),
 'L8':('float64',(1198,8)), 'original_available8':('bool',(1198,8)),
 'node_available':('bool',(1198,8)), 'node_fraction':('float32',(1198,8)),
 'context_fraction':('float32',(1198,)), 'box_xyxy':('float64',(1198,4)),
 'box_supported':('bool',(1198,)), 'context_supported':('bool',(1198,))}

def equal(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()

def load_geometry(b,white):
    p=verify(b,white)
    with os.fdopen(os.open(p,os.O_RDONLY|os.O_NOFOLLOW),'rb') as f,np.load(f,allow_pickle=False) as z:
        need(set(z.files)==set(GEO),'exact10 geometry members before decode')
        values={k:z[k] for k in GEO}
    for k,(dtype,shape) in GEO.items():need(values[k].dtype==np.dtype(dtype) and values[k].shape==shape,'geometry dtype/shape '+k)
    return values

def group_metadata_geometry(groups,g):
    need(len(groups)==1198 and len({x['group_index'] for x in groups})==1198,'1198 original groups')
    groups=sorted(groups,key=lambda x:x['group_index'])
    need(np.array_equal(g['group_index'],[x['group_index'] for x in groups]),'sparse original index position map')
    for i,group in enumerate(groups):
        for key in ('q8','L8','original_available8'):
            expected=np.asarray(group[key],dtype=g[key].dtype)
            need(np.array_equal(expected,g[key][i],equal_nan=True),'original group '+key)
        box=np.full(4,np.nan,np.float64) if group['box_xyxy'] is None else np.asarray(group['box_xyxy'],np.float64)
        need(np.array_equal(box,g['box_xyxy'][i],equal_nan=True),'original box')
        need(bool(g['box_supported'][i])==(group['box_xyxy'] is not None),'box support flag')
    need(not (g['node_available']&~g['original_available8']).any(),'original U not promoted')
    return groups

def check_before(before,g,a,position):
    need(equal(before['node_valid'],a['GROUP_NODE_VALID96.npy'][position]) and
         equal(before['context_valid'],a['GROUP_CONTEXT_VALID224.npy'][position]),'all8/context original validity exact')
    need(equal(before['node_support'],g['node_available'][position]) and
         equal(before['node_fraction'],g['node_fraction'][position]) and
         before['context_fraction'].tobytes()==g['context_fraction'][position].tobytes(), 'input-only support/fractions exact')
    need(bool(before['context_valid'].any())==bool(g['context_supported'][position]),'context support flag')

def check_proof(proof,saved):
    need(saved is not None and bool(saved['PASS'])==proof['PASS'],'independent native teacher PASS')
    for name,item in proof['checks'].items():
        for k,v in item.items():need(saved['checks'][name][k]==v,'independent positive-weight teacher tap '+name+'/'+k)

def run(entry_path):
    e=json.loads(raw(entry_path))
    need(e.get('approved_independent_SOURCE_scene_dataset_QA') is True and e.get('automatic_retry') is False,
         'Root separate exact actual entry before any dataset/base access')
    codes,meta,arrays,basewhite=e['code_whitelist'],e['metadata_whitelist'],e['array_whitelist'],e['base_image_whitelist']
    for key,name in [('code','verify_scene_pairs.py'),('math_code','independent_scene_math.py'),('io_code','qa_io.py')]:
        verify(e[key],codes);need(identity(e[key])==identity(binding(Path(__file__).parent/name)),'checker code pin')
    verify(e['protocol'],codes)
    need(e['root_protocol']['sha256']==ROOT_PROTOCOL_SHA,'Root prospective pin')
    read_json(e['root_protocol'],meta);read_json(e['root_scope'],meta)
    c=read_json(e['dataset_complete'],meta);need(c['completed'] is True and c['PASS'] is True,'producer terminal COMPLETE')
    need(identity(c['protocol'])==identity(e['root_protocol']),'producer protocol closure')
    need(c['code']['sha256']=='83a84fb22e177b0ea012b04f5c622e0ac71f9c5bdd612b9b63e4f1bae76538ca'
         and c['constructed_UNKNOWN_inputs_preserved'] is True and c['only_UNCONSTRUCTED_child_inputs_zero'] is True
         and c['training_executed'] is False and c['classifier_or_encoder_executed'] is False
         and c['SOURCE_DEV_payloads']==0 and c['current_or_held_payloads']==0,'pinned data-only producer schema')
    for ours,theirs in [('parents','parents'),('groups','groups'),('parent_image_whitelist','parent_images')]:
        need(identity(e[ours])==identity(c['metadata'][theirs]),'original closed metadata binding '+ours)
    for name in set(SHAPES)|{'GROUP_GEOMETRY.npz'}:
        need(identity(e['outputs'][name])==identity(c['outputs'][name]),'exact dataset output binding '+name)
    need(identity(e['pair_metadata'])==identity(c['outputs']['PAIR_METADATA.jsonl']),'exact saved metadata binding')
    for key,pin in [('parents','b439b10b866953b51706eb0195abc32e4de9be2ee15f4e51dd00313c43d21e3d'),
                    ('groups','c98785e6e5d26a9788491cd6400f9af5ab533538ef6d8e2a49795f3593ac4236'),
                    ('parent_image_whitelist','bfdfa28241fece306b6435ca5ae1ba943865dffbf12cb5ac34d9999d33b3f44d')]:
        need(e[key]['sha256']==pin,'frozen all5406 metadata pin')
    groups=read_jsonl(e['groups'],meta);closed=read_json(e['parent_image_whitelist'],meta)['images']
    closed_bases=[x['image'] for x in closed]
    need(len(closed_bases)==804 and {identity(x) for x in basewhite}=={identity(x) for x in closed_bases},'base804 only; no donor/DEV additions')
    crop=pinned_math(e['crop90_code'],codes,['crop'],'6a6c2543e28715e3d47e15c543c25f75ff7d3efa3b881e523bd3451817fc004b').crop
    context=pinned_math(e['context107_code'],codes,['query_extent','valid_box','target_context_grid','sample_target_context'],
        'aa7cd8e2568c8b7d30827e8681b4b87899944bd23cfa1b27948ceda5b55d07f3')
    out=safe(e['output_directory']);need(out.is_relative_to(Path(__file__).absolute().parent) and not out.exists(),'fresh own QA output');out.mkdir()
    try:
        a={name:map_npy(e['outputs'][name],arrays,dtype,shape) for name,(dtype,shape) in SHAPES.items()}
        g=load_geometry(e['outputs']['GROUP_GEOMETRY.npz'],arrays);groups=group_metadata_geometry(groups,g)
        bygroup={x['group_index']:(i,x) for i,x in enumerate(groups)}
        need(np.isin(a['PAIR_TARGETS.npy'],[-1,0,1]).all() and
             not (a['PROOF_PASS.npy']&~a['CONSTRUCTED.npy']).any(),'explicit input/label masks')
        rgbcache=LRU(64*1024*1024);beforecache=LRU(64*1024*1024);assets=set();checked_groups=set()
        stats=Counter();observations=[];collisions=[];all_ids=[]
        metadata=iter(iter_jsonl(e['pair_metadata'],meta));parent_count=0
        for i,p in enumerate(iter_jsonl(e['parents'],meta)):
            parent_count+=1;r=next(metadata);original=p['original_input'];sample=p['parent_sample_index']
            need(i<5406 and r['pair_index']==i and r['parent_sample_index']==sample==int(a['PARENT_INDICES.npy'][i]),'all5406 original ordered parent identity')
            need(original['fit'] is True and original['target']==0 and original['sample_index']==sample
                 and original['crop_available'] is True and original['original_query_available'] is True,'original FIT knownD only')
            need(r['original_input']==original and r['donor_id']==p['same_V235_donor_id']
                 and r['teacher_reference_xy']==p['teacher_reference_xy'] and r['teacher_authority']==p['teacher_authority']
                 and r['reference_is_supervision_only'] is True,'source/donor/teacher sidecar identity')
            need(identity(r['donor_RGB'])==identity(p['same_V235_donor_RGB']) and
                 identity(r['donor_visible_mask'])==identity(p['same_V235_donor_visible_mask']) and
                 r['same_V235_proposal_plan_sha256']==hashlib.sha256(json.dumps(p['same_V235_donor_plan'],sort_keys=True,separators=(',',':')).encode()).hexdigest(),
                 'same declared donor/old proposal metadata, no donor pixel reload')
            need(r['role_order']==list(ROLES) and r['alpha_packbits_bitorder']=='big'
                 and r['old_V235_success_diagnostic_only']==p['old_V235_success_diagnostic_only']
                 and r['old_V235_pair_index']==p['old_V235_pair_index'] and r['old_proof_inherited'] is False
                 and r['original_D_retained'] is True and r['new_native_gold'] is False,'old39/history/authority preserved')
            pos,group=bygroup[p['group_index']];slot=p['active_slot']
            need(r['group_cache_index']==pos and r['group_index']==group['group_index']
                 and r['group_id']==group['group_id'] and r['active_slot']==slot,'sparse group/slot address')
            need(identity(p['image'])==identity(group['image']) and identity(r['image'])==identity(group['image']), 'actualscene image not anchor substitution')
            need(np.array_equal(np.asarray(original['q'],np.float64),np.asarray(group['q8'],np.float64)[slot],equal_nan=True)
                 and original['L']==group['L8'][slot] and original['original_query_available']==group['original_available8'][slot], 'parent original q/L/input flag vs q8')
            key=identity(group['image']);base=rgbcache.get(key,lambda:decode_base(group['image'],basewhite));assets.add(key)
            need(group['raw_hw'] is None or list(base.shape[:2])==list(group['raw_hw']),'known original raw dimensions')
            before=beforecache.get(pos,lambda:sample_all(base,group,crop,context));checked_groups.add(pos)
            check_before(before,g,a,pos)
            before_rgb,valid_sha=sha(before['nodes'][slot]),sha(before['node_valid'][slot])
            need(before_rgb==p['expected_BEFORE_RGB96_CHW_sha256']==r['expected_BEFORE_RGB96_CHW_sha256']==r['actual_BEFORE_RGB96_CHW_sha256']
                 and valid_sha==p['expected_BEFORE_valid96_bool_sha256']==r['expected_BEFORE_valid96_bool_sha256']==r['actual_BEFORE_valid96_bool_sha256'], 'BEFORE5406 exact original pixel/valid proof')
            need(r['native_scene_sha256']['BEFORE']==sha(base),'native BEFORE SHA')
            made=bool(a['CONSTRUCTED.npy'][i]);passed=bool(a['PROOF_PASS.npy'][i]);family=int(original['family_code'])
            need(r['constructed']==made and r['proof_pass']==passed,'saved construction/proof flags')
            need((r['unknown_reason'] is None) if passed else bool(r['unknown_reason']), 'explicit unsupported/proof-failure reason')
            all_ids.append(sample);stats[f'family{family}_parents']+=1
            stats['old_success_retained' if p['old_V235_success_diagnostic_only'] else 'old_failure_retained']+=1
            ref=p['teacher_reference_xy'];q=np.asarray(original['q'],np.float64)
            distance=None if ref is None else float(np.linalg.norm(q-np.asarray(ref,np.float64)))
            bin_name='UNKNOWN' if distance is None else 'LE3' if distance<=3 else 'GT3'
            stats[f'family{family}_{bin_name}_parents']+=1
            observation=dict(pair_index=i,parent_sample_index=sample,family=family,q_reference_distance_px=distance,
                q_reference_bin=bin_name,constructed=made,r_proof_pass=passed,q_observation=None,
                labels_unchanged_by_q_diagnostic=True)
            if not made:
                need(not passed and np.all(a['PAIR_TARGETS.npy'][i]==-1) and r['targets_D0_E1_Uminus1']==[-1,-1]
                     and r['input_available']==[False,False] and r['target_known']==[False,False], 'unconstructed explicit input absent/targetunknown')
                for name in ('CF_NODE_RGB96.npy','CF_CONTEXT_RGB224_F32.npy','PAIR_LAYER_RGB96.npy','PAIR_ALPHA_PACKBITS.npy'):
                    need(not np.count_nonzero(a[name][i]),'unconstructed child arrays zero')
                for variant,role in enumerate(ROLES):
                    need(r['native_scene_sha256'][role] is None and r['native_alpha_sha256_bool'][role] is None
                         and r['native_alpha_pixels'][role]==0,'unconstructed child native SHA null')
                    need(r['saved_variant_features'][role]['NODE_RGB96_sha256']==sha(a['CF_NODE_RGB96.npy'][i,variant])
                         and r['saved_variant_features'][role]['CONTEXT_RGB224_F32_sha256']==sha(a['CF_CONTEXT_RGB224_F32.npy'][i,variant]), 'unconstructed zero feature SHA')
                need(r['carrier_RGB96_CHW_sha256']==[None,None] and r['alpha_bool_sha256']==[None,None]
                     and r['alpha_pixels']==[0,0], 'unconstructed no layer claims')
                stats[f'family{family}_unconstructed']+=1;observations.append(observation);continue
            stats[f'family{family}_constructed']+=1;stats[f'family{family}_{bin_name}_constructed']+=1
            scenes=[];alphas=[];outputs=[]
            for variant,role in enumerate(ROLES):
                carrier=np.asarray(a['PAIR_LAYER_RGB96.npy'][i,variant])
                alpha=np.unpackbits(a['PAIR_ALPHA_PACKBITS.npy'][i,variant],bitorder='big',count=9216).reshape(96,96).astype(bool)
                need(not np.count_nonzero(carrier[:,~alpha]),'carrier RGB outside alpha zero')
                need(r['carrier_RGB96_CHW_sha256'][variant]==sha(carrier) and r['alpha_bool_sha256'][variant]==sha(alpha)
                     and r['alpha_pixels'][variant]==int(alpha.sum()),'stored layer SHA/mass')
                scene,opaque=lift_scene(base,carrier,alpha,original['q'],original['L']);child=sample_all(scene,group,crop,context)
                need(sha(scene)==r['native_scene_sha256'][role],'independent native composite SHA')
                need(sha(opaque)==r['native_alpha_sha256_bool'][role] and int(opaque.sum())==r['native_alpha_pixels'][role], 'independent native alpha SHA/mass')
                need(equal(child['nodes'],a['CF_NODE_RGB96.npy'][i,variant]) and
                     equal(child['context'],a['CF_CONTEXT_RGB224_F32.npy'][i,variant]), 'allconstructed same-native all8/V107 outputs byteexact')
                check_before(child,g,a,pos)
                need(r['saved_variant_features'][role]['NODE_RGB96_sha256']==sha(child['nodes']) and
                     r['saved_variant_features'][role]['CONTEXT_RGB224_F32_sha256']==sha(child['context']), 'saved feature pixel SHA')
                need(changed_counts(base,scene,before,child,slot)==r['changed_pixels'][role], 'native/active/context changed counts')
                scenes.append(scene);alphas.append(opaque);outputs.append(child)
            proof=teacher_proof(base,*scenes,*alphas,ref);need(proof['PASS']==passed,'renewed reference proof independently computed')
            check_proof(proof,r['proof']);targets=[1,0] if passed else [-1,-1]
            need(a['PAIR_TARGETS.npy'][i].tolist()==targets==r['targets_D0_E1_Uminus1']
                 and r['input_available']==[True,True] and r['target_known']==[passed,passed], 'constructed proofFAIL inputs retained/labels masked')
            qproof=teacher_proof(base,*scenes,*alphas,original['q']);observation['q_observation']=qproof
            for footprint,items in qproof['checks'].items():
                stats[f'family{family}_q_{footprint}_supported']+=int(items['supported'])
                if passed:
                    if not items['supported']:stats[f'family{family}_rpass_q_{footprint}_UNKNOWN']+=1
                    elif not items['COVER_all_opaque']:stats[f'family{family}_rpass_q_{footprint}_COVER_not_all_opaque']+=1
            if passed:
                stats[f'family{family}_proof_pass']+=1;stats[f'family{family}_{bin_name}_proof_pass']+=1
                if not qproof['PASS']:stats[f'family{family}_rpass_q_combined_notpass']+=1
                same_cs=equal(outputs[0]['nodes'],outputs[1]['nodes']) and equal(outputs[0]['context'],outputs[1]['context'])
                same_before=equal(outputs[0]['nodes'],before['nodes']) and equal(outputs[0]['context'],before['context'])
                if same_cs or same_before:
                    collisions.append(dict(parent_sample_index=sample,family=family,
                        COVER_SHAM_identical_all_input_RGB=same_cs,COVER_BEFORE_identical_all_input_RGB=same_before,
                        targets_preserved_no_RGB_difference_gate=True))
            observations.append(observation)
            if i%100==0:print(json.dumps(dict(stage='independent_native_dataset_replay',parents=i+1,total=5406)),flush=True)
        need(parent_count==5406 and next(metadata,None) is None and len(set(all_ids))==5406,'all5406/no extra rows')
        need(len(assets)==804 and len(checked_groups)==1198,'all closed bases/groups processed')
        need(stats['family0_parents']==4565 and stats['family1_parents']==841
             and stats['old_success_retained']==5367 and stats['old_failure_retained']==39,'fixed source/oldfailure census')
        for family in (0,1):
            for key,ours in [('attempted','parents'),('constructed','constructed'),('proof_pass','proof_pass')]:
                need(c['by_family'][str(family)][key]==stats[f'family{family}_{ours}'],'producer family counts independently recounted')
        write_json(out/'Q_VS_REFERENCE_OBSERVATIONS.json',observations)
        write_json(out/'IDENTICAL_RGB_OPPOSITE_TARGET_DIAGNOSTIC.json',collisions)
        result=dict(completed=True,PASS=True,dataset_complete=e['dataset_complete'],entry=binding(entry_path),
            code=binding(__file__),counts=dict(stats),closed_base_RGB_assets_read=804,all5406_BEFORE_active_original_SHA_exact=True,
            all_constructed_native_layers_all8_context_proofs_independently_checked=True,
            Python_image_and_before_LRU_peak_bytes=rgbcache.peak+beforecache.peak,
            additional_donor_role_PNG_or_mask_loads=0,rawGT_model_current_or_held_reads=0,new_labels_or_parent_selection=False,
            current_admission=False,new_training_or_model_selection=False,
            limitations=['Stored carrier reconstructs scene; no additional donor-role or donor-mask load. 61 RGB paths shared with donors may be read solely as approved base images.',
             'Teacher footprint authority retained; no q-center/native physical/whole-part gold upgrade.',
             'q-vs-r and identical-RGB diagnostics do not relabel, filter parents or require RGB difference.'])
        receipt=write_json(out/'REVIEW.json',result);print(json.dumps(dict(completed=True,PASS=True,REVIEW=receipt)))
    except BaseException:
        write_json(out/'FAILED_EXECUTION.json',dict(completed=False,PASS=False,automatic_retry=False,traceback=traceback.format_exc()));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--entry',required=True);run(p.parse_args().entry)
