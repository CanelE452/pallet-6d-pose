"""V116 explicit wood-foreground wrapper. Import opens no source metadata/payload."""
import argparse
from collections import Counter, OrderedDict
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time
import traceback
import numpy as np
import PIL
from PIL import Image

sys.dont_write_bytecode = True
HERE=Path(__file__).resolve().parent; V=HERE.parent; P=V.parent
META=V/'metadata_preparation'; OUT=V/'corpus_build_v1'
POLICY=V/'ROOT_SOURCE_POLICY_BEFORE_DONORS_AND_PIXELS.json'
POLICY_SHA='c2bb6ab955bb9d726281397a0f3df281192cd9acde964fe12cf843d24af0902c'
V96=P/'query_visibility_query_anchored_cue_20261023_v96'
KERNEL=V96/'query_anchored_pair_preparation/query_anchored_pair_kernel.py'
CROP=P/'query_visibility_public_cue_inputs_20261023_v90/producer/spatial_input_math.py'
CROP_SHA='6a6c2543e28715e3d47e15c543c25f75ff7d3efa3b881e523bd3451817fc004b'
KERNEL_SHA='0299a3f6af75b3907875675499054b88bfc8f9a59203bd4ea4b58529f661b5f9'
PROTOCOL=HERE/'SOURCE_PROTOCOL_BEFORE_PIXELS.json'


def require(ok,message):
    if not ok: raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def bind(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))
def read(path):return json.loads(Path(path).read_text())
def rows(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s]
def pixels(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def clean(a):
    if isinstance(a,np.generic):return a.item()
    if isinstance(a,np.ndarray):return a.tolist()
    if isinstance(a,dict):return {k:clean(v) for k,v in a.items()}
    if isinstance(a,(tuple,list)):return [clean(v) for v in a]
    return a


def write(path,value):
    with Path(path).open('x') as stream:json.dump(clean(value),stream,indent=2,allow_nan=False);stream.write('\n')


def jsonl(path,values):
    with Path(path).open('x') as stream:
        for value in values:stream.write(json.dumps(clean(value),allow_nan=False,separators=(',',':'))+'\n')


def check(binding):
    require(sha(binding['path'])==binding['sha256'],'Changed bound file '+binding['path']);return Path(binding['path'])


def load_math(expected_kernel_sha):
    require(sha(KERNEL)==expected_kernel_sha,'Immutable V96 kernel changed')
    spec=importlib.util.spec_from_file_location('v116_exact_v96_lower_math',KERNEL)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def donor_allowed_v116(row,donor,locked_donors,old_val_rgb_shas):
    """New explicit role allowlist; no call/patch to V96 odd-ID TRAIN guard."""
    if donor is None:return False
    chosen=locked_donors.get(donor.get('id'))
    return (chosen is not None and chosen==donor and row['split'] in ('TRAIN','TRAIN_LAB')
        and donor.get('object_id')==16 and donor.get('scene_id') in (2,4)
        and donor.get('split')=='TRAIN' and donor.get('original_split')=='train'
        and donor.get('image_sha256')!=row['image_sha256']
        and donor.get('image_sha256') not in old_val_rgb_shas)


def prepare_pair(k,native,row,target,donor_rgb,selected,others,complete,donor,locked,old_val):
    """One assigned donor only; reuse exact protected/alpha/scale/placement/math."""
    require(native.dtype==np.uint8 and native.shape==tuple(row['raw_hw'])+(3,),'Native base RGB')
    protected,reason=k.protected_mask(row,target)
    facts=dict(pair_ready=False,reason=reason,protected_pixels=0 if protected is None else int(protected.sum()),
        original_q=row['query_xy'],original_L=row['L'],same_full_RGBA_transform=False,donor_id=donor.get('id'))
    def failed(why):
        facts['reason']=why
        return dict(images=(native,native),labels=(-1,-1),facts=facts,protected=protected,cutout=None)
    if protected is None:return failed(reason)
    if not donor_allowed_v116(row,donor,locked,old_val):return failed('V116_EXPLICIT_LOCKED_WOOD_TRAIN_ALLOWLIST_U')
    if selected is None or others is None:return failed('UNAVAILABLE_ASSIGNED_DONOR_OR_COMPLETE_OWNER_MASKS_U')
    exclusive,ownership=k.alpha_rule.exclusive_alpha(selected,others,complete);facts['exclusive_alpha']=ownership
    if not ownership['admitted']:return failed(ownership['reason'])
    if donor_rgb is None:return failed('UNAVAILABLE_ASSIGNED_DONOR_RGB_U')
    cut,scale=k.V2.support_scaled_cutout(donor_rgb,exclusive,protected,row['L']);facts['scale']=scale
    if cut is None:return failed(scale['reason'])
    pair,mass=k.V2.core_aligned_pair(row,cut,protected);facts['mass']=mass
    if pair is None:return failed(mass['reason'])
    (cxy,ca),(sxy,sa)=pair
    cover=k.V1.compose(native,cut,cxy,ca);sham=k.V1.compose(native,cut,sxy,sa)
    passed,recheck=k.V2.mass_certificate(row,ca,sa)
    require(passed and ca[protected].all() and not sa[protected].any(),'Frozen opacity/mass certificates')
    require(np.array_equal(sham[protected],native[protected]) and ca.sum()==sa.sum()==cut['alpha'].sum(),'Same RGBA/global mass and exact sham protected RGB')
    facts.update(pair_ready=True,reason='V116_FIXED_WOOD_FOREGROUND_PAIR_READY',cover_xy=list(cxy),sham_xy=list(sxy),
        same_full_RGBA_transform=True,protected_cover_all1=True,protected_sham_all0=True,sham_protected_RGB_bitexact=True,mass_recheck=recheck,
        source_cutout_bbox_xyxy=cut['donor_native_bbox_xyxy'],resized_cutout_hw=cut['resized_hw'])
    return dict(images=(cover,sham),labels=(1,0),facts=facts,protected=protected,cutout=cut)


def entry_structure(e):
    require(e.get('schema')=='V116_ROOT_IMAGE_ONLY_CONSTRUCTION_ENTRY_V1' and e.get('approved') is True and e.get('allow_source_image_construction') is True,'Approved root entry required before bodies')
    require(e.get('source_only') is True and e.get('model_forward_or_fit')==0 and e.get('current223') is False and e.get('V112') is False and e.get('PnP')==0,'Source image-only scope')
    require(e.get('expected_bases')==28 and e.get('expected_donors')==16 and e.get('expected_attempts')==448 and e.get('expected_occurrences')==924,'Fixed28x16/all924')
    require(Path(e['output_directory']).resolve()==OUT,'Fixed new output directory')
    require(set(e['bindings'])=={'producer','protocol','generated','metadata_seal','root_policy'},'Exact root binding names')
    require(e['bindings']['producer']['path']==str(Path(__file__).resolve()) and e['bindings']['protocol']['path']==str(PROTOCOL) and e['bindings']['root_policy']['path']==str(POLICY),'Fixed code/protocol/policy paths')


def guarded_metadata(entry):
    e=read(entry);entry_structure(e)
    require(not OUT.exists(),'No overwrite/restart; preserve partial failures')
    for b in e['bindings'].values():check(b)
    require(sha(POLICY)==POLICY_SHA,'Exact root pre-pixel policy')
    protocol=read(PROTOCOL);require(protocol['producer']==e['bindings']['producer'] and protocol['root_policy']==e['bindings']['root_policy'],'Protocol code/policy binding')
    require(protocol['metadata_seal']==e['bindings']['metadata_seal'],'Metadata selection seal binding')
    proof=read(e['bindings']['generated']['path']);require(proof['PASS'] and proof['producer']==e['bindings']['producer'] and proof['source_payload_reads']==0,'Exact generated-only proof')
    metadata=read(e['bindings']['metadata_seal']['path']);require(metadata['completed'] is True and metadata['PASS'] is True,'Completed metadata lock')
    for b in metadata['outputs'].values():check(b)
    for b in protocol['bindings'].values():check(b)
    return e,protocol


def validate_metadata(protocol):
    """Real locked manifests/schema only. Never follows RGB/mask/array refs."""
    for b in protocol['bindings'].values():check(b)
    B=protocol['bindings']
    donors=rows(B['selected_donors']['path']);bases=rows(B['bases']['path']);targets=rows(B['targets']['path']);assign=rows(B['assignments']['path']);plan=rows(B['input_plan']['path'])
    contexts=rows(B['context_readiness']['path'])
    old_val=set(read(B['old_val_donor_RGB_shas']['path'])['RGB_SHA256'])
    locked={d['id']:d for d in donors}
    require(len(locked)==len(donors)==16 and len(bases)==len(targets)==28 and len(assign)==448 and len(plan)==924,'All fixed metadata rows')
    require(len({d['image_sha256'] for d in donors})==16,'Distinct locked donor RGBs')
    all200=rows(B['all200_donors']['path'])
    require(len(all200)==200 and [d['id'] for d in donors]==[d['id'] for d in sorted(all200,key=lambda d:(d['image_sha256'],d['original_source_instance_id']))[:16]],'Exact pre-pixel first16 SHA/instance ID donor lock')
    require(len(contexts)==28,'All28 original context/readiness rows')
    for i,(base,target) in enumerate(zip(bases,targets)):
        row,t=base['original_row'],target['original_target']
        require(base['base_index']==target['base_index']==i and row['slot_index']==t['base_row_index']==base['original_base_row_index'],'Original base joins')
        require(row['query_xy']==t['original_q'] and row['L']==t['original_L'] and t['query_anchored_label']==0 and row['split'] in ('TRAIN','TRAIN_LAB'),'Unchanged TRAIN anchored D q/L')
        s=base['source_lineage'];ctx=contexts[i]
        require(s['decoded_RGB_sha256']==s['before_decoded_sha256'] and ctx['base_index']==i and ctx['original_base_row_index']==row['slot_index'],'Source decode/context original row identity')
        require(ctx['query_xy']==row['query_xy'] and ctx['L']==row['L'] and ctx['context_full_RGB']==s['fullRGB_file_binding'] and ctx['context_decoded_RGB_sha256']==s['decoded_RGB_sha256'],'Original C11 context/RGB/qL metadata')
        for rank,d in enumerate(donors):
            a=assign[16*i+rank];require(a['attempt_index']==16*i+rank and a['base_index']==i and a['donor_rank']==rank and a['donor_id']==d['id'],'All448 assigned donors exact order')
            require(donor_allowed_v116(row,d,locked,old_val),'Explicit locked donor metadata allowlist')
        for j,o in enumerate(plan[33*i:33*i+33]):
            rank=None if j==0 else (j-1)//2
            variant='BEFORE' if j==0 else 'OPAQUE_COVER' if j%2 else 'CUE_PRESERVING_SHAM'
            require(o['occurrence_index']==33*i+j and o['base_index']==i and o['original_base_row_index']==row['slot_index'] and o['donor_rank']==rank and o['variant']==variant,'All924 canonical occurrence order')
            require(o['query_xy']==row['query_xy'] and o['L']==row['L'] and o['raw_split']==row['split'] and o['native_slot_logical_only']==row['native_slot_logical_only'] and o['selected_box_xyxy']==row['selected_box_xyxy'],'All924 original qL/slot/rawfold/candidate unchanged')
    require(Counter(b['source'] for b in bases)=={'LOCO':3,'PALLOX':25},'Exact original3LOCO25PALLOX')
    for rank,d in enumerate(donors):
        owners=d['all_visible_owners'];n=d['complete_scene_GT_owner_count']
        require(d['donor_rank']==rank and d['image']['sha256']==d['image_sha256'] and [a['instance_index'] for a in owners]==list(range(n)),'Actual donor owner schema/ordered completeness metadata')
        own=owners[d['instance_id']]
        require(own['object_id']==16 and own['visible_mask']==d['own_visible_mask'],'Selected owner index/object/mask identity')
    return donors,bases,targets,assign,plan,contexts,old_val


def run(entry):
    e,protocol=guarded_metadata(entry);B=protocol['bindings'];k=load_math(B['kernel']['sha256'])
    crop,crop_ast=k.V1.frozen_rgb96_operator(CROP,CROP_SHA)
    donors,bases,targets,assign,plan,contexts,old_val=validate_metadata(protocol)
    locked={d['id']:d for d in donors}
    OUT.mkdir();[(OUT/n).mkdir() for n in ['edited_fullRGB','protected_masks','donor_certificates','cutout_RGBA']]
    write(OUT/'RUN_STARTED.json',dict(entry=bind(entry),protocol=bind(PROTOCOL),source_only=True,model_forward_or_fit=0,current223=False,V112=False,PnP=0))
    occurrence_rows=[];GT=[];feasibility=[];donor_facts=[];source_facts=[]
    crops=np.empty((924,3,96,96),np.uint8);valids=np.empty((924,1,96,96),bool)
    rgbcache=OrderedDict();start=time.monotonic()
    def rgb_file(b,fmt):
        path=str(check(b))
        if path not in rgbcache:
            image=np.load(path,allow_pickle=False) if fmt in ('NPY','CANONICAL_RGB_UINT8_HWC_NPY') else np.array(Image.open(path).convert('RGB'),copy=True)
            require(image.dtype==np.uint8 and image.ndim==3 and image.shape[2]==3,'Native source RGB uint8 HWC')
            rgbcache[path]=image
            if len(rgbcache)>4:rgbcache.popitem(last=False)
        return rgbcache[path]
    def mask_file(b,hw):
        path=Path(b['path'])
        if not path.is_file():return None
        check(b);a=np.array(Image.open(path))
        require(a.dtype==np.uint8 and a.shape==tuple(hw) and np.isin(a,[0,255]).all(),'Native binary public visible mask')
        return a>0
    try:
        donor_data={}
        for d in donors:
            controls_available=all(Path(c['path']).is_file() for c in d['controls'].values())
            for control in d['controls'].values():
                if Path(control['path']).is_file():check(control)
            gt=read(d['controls']['scene_gt.json']['path'])[str(d['frame_id'])] if controls_available else []
            info=read(d['controls']['scene_gt_info.json']['path'])[str(d['frame_id'])] if controls_available else []
            owners=d['all_visible_owners']
            complete=(controls_available and d['all_visible_owner_files_available_recorded'] is True and len(gt)==d['complete_scene_GT_owner_count']
                and [a['instance_index'] for a in owners]==list(range(len(gt)))
                and all(a['object_id']==gt[a['instance_index']]['obj_id'] for a in owners))
            selected=mask_file(d['own_visible_mask'],d['raw_hw'])
            other=np.zeros(tuple(d['raw_hw']),bool)
            for owner in owners:
                m=mask_file(owner['visible_mask'],d['raw_hw'])
                if m is None:complete=False
                elif owner['instance_index']!=d['instance_id']:other|=m
            if selected is None:complete=False;selected=np.zeros_like(other)
            expected_visible_pixels=info[d['instance_id']]['px_count_visib'] if info else None
            if int(selected.sum())!=expected_visible_pixels:complete=False
            exclusive,cert=k.alpha_rule.exclusive_alpha(selected,other,complete)
            image=rgb_file(d['image'],'PNG') if Path(d['image']['path']).is_file() else None
            if image is not None:require(image.shape==tuple(d['raw_hw'])+(3,),'Donor RGB HW')
            proof_path=OUT/'donor_certificates'/f"DONOR_{d['donor_rank']:02d}_NATIVE_OWNERSHIP.npz"
            padded=np.pad(exclusive,1);core=np.logical_and.reduce([padded[y:y+exclusive.shape[0],x:x+exclusive.shape[1]] for y in range(3) for x in range(3)])
            np.savez_compressed(proof_path,selected_visible=selected,other_visible_union=other,exclusive_alpha=exclusive,erode1_exclusive_alpha=core)
            donor_data[d['id']]=(image,selected,other,complete)
            donor_facts.append(dict(donor_rank=d['donor_rank'],donor_id=d['id'],image=d['image'],decoded_RGB_sha256=None if image is None else pixels(image),own_visible_mask=d['own_visible_mask'],all_visible_owners=owners,complete_owner_instance_set=complete,expected_visible_pixels=expected_visible_pixels,actual_selected_visible_pixels=int(selected.sum()),donor_RGB_available=image is not None,exclusive_alpha=cert,native_ownership_proof=bind(proof_path),native_ownership_keys=['selected_visible','other_visible_union','exclusive_alpha','erode1_exclusive_alpha'],replacement_allowed=False))
        def save_occ(index,base,native,image,label,ready,reason,protected_ref=None):
            original=base['original_row'];source=base['source_lineage'];o=dict(plan[index])
            changed=not np.array_equal(image,native);b=source['fullRGB_file_binding'];fmt=source['fullRGB_format']
            if changed:
                path=OUT/'edited_fullRGB'/f'occ_{index:04d}.png';Image.fromarray(image).save(path);b=bind(path);fmt='PNG'
            digest=pixels(image)
            o.update(fullRGB=b,format='NPY' if fmt=='CANONICAL_RGB_UINT8_HWC_NPY' else 'PNG',decoded_RGB_sha256=digest,raw_hw=original['raw_hw'],query_xy=original['query_xy'],L=original['L'],full_rgb_changed=changed,source_original_image_sha256=original['image_sha256'],base_index=base['base_index'],original_base_row_index=base['original_base_row_index'])
            require(o['occurrence_index']==index,'All924 occurrence order')
            rgb96,valid96=crop(image,np.asarray(original['query_xy']),original['L'],96)
            o.update(rgb96_row=index,rgb96_sha256=pixels(rgb96),crop_valid96_sha256=pixels(valid96),frozen_context_box_xyxy=contexts[base['base_index']]['frozen_context_box_xyxy'],context_feature_rule='Fresh target context from this exact native fullRGB when changed; original fixed box, no AFTER reinitialization')
            crops[index]=rgb96;valids[index]=valid96;occurrence_rows.append(o)
            GT.append(dict(occurrence_index=index,occurrence_id=o['occurrence_id'],base_index=base['base_index'],original_base_row_index=base['original_base_row_index'],donor_rank=o['donor_rank'],variant=o['variant'],candidate_aux_hidden_label_GT_only=label,pair_ready=ready,pair_reason=reason,protected_union_file=protected_ref,native_mapping='UNKNOWN',human_gold=False,training_admission=False))
        for i,(base,tref) in enumerate(zip(bases,targets)):
            row,target=base['original_row'],tref['original_target'];s=base['source_lineage']
            native=rgb_file(s['fullRGB_file_binding'],s['fullRGB_format'])
            require(native.shape==tuple(row['raw_hw'])+(3,) and pixels(native)==s['decoded_RGB_sha256'],'Exact original source pixel lineage')
            before_crop,before_valid=crop(native,np.asarray(row['query_xy']),row['L'],96)
            require(pixels(before_crop)==contexts[i]['saved_RGB96_sha256'] and pixels(before_valid)==contexts[i]['saved_valid96_sha256'],'Exact original C11 cached BEFORE RGB96/valid96 hashes')
            protected,why=k.protected_mask(row,target)
            pm=OUT/'protected_masks'/f'BASE_{i:02d}.npz';np.savez_compressed(pm,protected_union=protected if protected is not None else np.zeros(tuple(row['raw_hw']),bool));pm_binding=bind(pm)
            source_facts.append(dict(base_index=i,original_base_row_index=base['original_base_row_index'],fullRGB=s['fullRGB_file_binding'],decoded_RGB_sha256=pixels(native),protected_union_file=pm_binding,protected_support_available=protected is not None,protected_support_reason=why))
            save_occ(33*i,base,native,native,0,False,'UNCHANGED_ORIGINAL_QUERY_ANCHORED_D_BEFORE',pm_binding)
            for rank,d in enumerate(donors):
                drgb,selected,other,complete=donor_data[d['id']]
                result=prepare_pair(k,native,row,target,drgb,selected,other,complete,d,locked,old_val)
                facts=result['facts'];cutout_ref=None
                if result['cutout'] is not None:
                    path=OUT/'cutout_RGBA'/f'ATTEMPT_{16*i+rank:03d}.npz';cut=result['cutout'];np.savez_compressed(path,rgb=cut['rgb'],alpha=cut['alpha']);cutout_ref=bind(path)
                facts.update(attempt_index=16*i+rank,base_index=i,original_base_row_index=base['original_base_row_index'],donor_rank=rank,donor_id=d['id'],protected_union_file=pm_binding,protected_union_key='protected_union',protected_support_available=result['protected'] is not None,cutout_RGBA_file=cutout_ref,cutout_RGBA_keys=None if cutout_ref is None else ['rgb','alpha'])
                feasibility.append(facts)
                for j,(image,label) in enumerate(zip(result['images'],result['labels'])):
                    save_occ(33*i+1+2*rank+j,base,native,image,label,facts['pair_ready'],facts['reason'],pm_binding)
            print(json.dumps(dict(bases_done=i+1,total28=28,attempts=len(feasibility),accepted=sum(a['pair_ready'] for a in feasibility),seconds=time.monotonic()-start)),flush=True)
        require(len(occurrence_rows)==len(GT)==924 and len(feasibility)==448,'All attempted records retained')
        jsonl(OUT/'OCCURRENCES924_INPUT_ONLY.jsonl',occurrence_rows);jsonl(OUT/'OCCURRENCES924_GT_ONLY.jsonl',GT)
        jsonl(OUT/'FEASIBILITY448_GT_ONLY.jsonl',feasibility);jsonl(OUT/'SELECTED16_DONOR_CERTIFICATES.jsonl',donor_facts);jsonl(OUT/'SOURCE28_RGB_CERTIFICATES.jsonl',source_facts)
        np.savez_compressed(OUT/'RGB96_AND_VALID924_INPUT_ONLY.npz',rgb96=crops,crop_valid96=valids)
        summary=dict(completed=True,PASS=True,base_count=28,donor_count=16,attempts=448,occurrences=924,accepted_pairs=sum(a['pair_ready'] for a in feasibility),failures_U=sum(not a['pair_ready'] for a in feasibility),label_counts=dict(Counter(str(g['candidate_aux_hidden_label_GT_only']) for g in GT)),reasons=dict(Counter(a['reason'] for a in feasibility)),source_only=True,training_admission=False,native_mapping='UNKNOWN',human_gold=False,original_q_L_order_folds_unchanged=True,all_attempts_retained=True,no_donor_replacement=True,crop_operator_AST_sha256=crop_ast,seconds=time.monotonic()-start,model_forward_or_fit=0,current223=False,V112=False,PnP=0)
        summary['runtime']=dict(Python=platform.python_version(),NumPy=np.__version__,Pillow=PIL.__version__)
        write(OUT/'SUMMARY.json',summary)
        files=[bind(path) for path in sorted(OUT.rglob('*')) if path.is_file()]
        write(OUT/'CORPUS_OUTPUT_SEAL.json',dict(completed=True,PASS=True,root_entry=bind(entry),protocol=bind(PROTOCOL),metadata_seal=e['bindings']['metadata_seal'],files=files,occurrences=bind(OUT/'OCCURRENCES924_INPUT_ONLY.jsonl'),GT=bind(OUT/'OCCURRENCES924_GT_ONLY.jsonl'),feasibility=bind(OUT/'FEASIBILITY448_GT_ONLY.jsonl'),donor_certificates=bind(OUT/'SELECTED16_DONOR_CERTIFICATES.jsonl'),summary=bind(OUT/'SUMMARY.json'),source_only=True,training_admission=False,model_forward_or_fit=0,current223=False,V112=False,PnP=0))
        print(json.dumps(dict(completed=True,summary=summary,seal=bind(OUT/'CORPUS_OUTPUT_SEAL.json'))))
    except BaseException:
        write(OUT/'FAILED.json',dict(completed=False,exception=traceback.format_exc(),completed_attempts=len(feasibility),automatic_retry=False,model_forward_or_fit=0,current223=False,V112=False))
        raise


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--entry');ap.add_argument('--validate-metadata',action='store_true');ap.add_argument('--output');args=ap.parse_args()
    if args.validate_metadata:
        require(sha(POLICY)==POLICY_SHA,'Exact root policy');protocol=read(PROTOCOL)
        require(protocol['producer']==bind(__file__),'Exact protocol producer');metadata=read(check(protocol['metadata_seal']))
        require(metadata['completed'] and metadata['PASS'],'Metadata completed/PASS')
        for b in metadata['outputs'].values():check(b)
        result=validate_metadata(protocol)
        facts=dict(PASS=True,producer=bind(__file__),protocol=bind(PROTOCOL),metadata_seal=protocol['metadata_seal'],bases=len(result[1]),donors=len(result[0]),assignments=len(result[3]),occurrences=len(result[4]),owner_schema='all_visible_owners/instance_index/visible_mask',source_payload_reads=0,source_metadata_only=True,model_forward_or_fit=0,current223=False,V112=False,PnP=0)
        if args.output:write(args.output,facts)
        print(json.dumps(facts))
    else:
        require(args.entry is not None,'--entry required before construction');run(args.entry)
