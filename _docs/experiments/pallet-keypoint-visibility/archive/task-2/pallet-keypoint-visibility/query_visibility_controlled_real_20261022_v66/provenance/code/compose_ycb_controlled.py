"""Frozen public-YCB opaque occlusion composer; no model/GPU/PnP work.

`freeze` registers source pools, input hashes, recipe and code BEFORE any
composition. `generate` consumes that immutable lock and preserves partial
outputs/errors rather than silently changing the accepted recipe.
"""
import argparse
from collections import Counter,OrderedDict
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time
import cv2
import numpy as np
from PIL import Image
import composer_core as core

ROOT=Path(__file__).resolve().parents[1]
V63=ROOT.parent/'query_visibility_public_real_20261022_v63'
V64=ROOT.parent/'query_visibility_multitarget_20261022_v64'
YCB=V63/'ycb'
PROTOCOL_SHA='7ab257789998262dba23508aa4b5e7cce4450d85293c64e3dba948d78c040109'
PROV=ROOT/'provenance'
OUT=ROOT/'controlled'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def write(path,obj):
    path=Path(path)
    if path.exists():raise FileExistsError('Preserve existing artifact: '+str(path))
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def lines(path):return [json.loads(s)for s in Path(path).read_text().splitlines()]


def write_lines(path,rows):
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    with path.open('w')as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')


def binding(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))


def check(b):
    if sha(b['path'])!=b['sha256']:raise ValueError('Frozen binding changed: '+b['path'])


def ident(r):return f"YCB:{int(r['scene_id']):06d}:{int(r['frame_id']):06d}:{int(r['instance_id']):06d}"


def read_csv(path):
    with Path(path).open(newline='')as f:return list(csv.DictReader(f))


def freeze():
    if(PROV/'GENERATION_INPUT_LOCK.json').exists():raise FileExistsError('Generation already registered')
    protocol=ROOT/'DATA_LABEL_PROTOCOL.json'
    if sha(protocol)!=PROTOCOL_SHA:raise ValueError('The approved controlled-label protocol changed')
    contract=ROOT/'contract/controlled_truth.py'
    if not contract.exists():raise FileNotFoundError('Wait for frozen pure partial-label helper')
    inputs=[protocol,contract,V63/'sam_source/INPUTS.jsonl',V63/'sam_source/LOCK.json',YCB/'train_instances.jsonl',
        YCB/'SOURCE_VAL_LOCKED_200.jsonl',YCB/'SELECTED_FILES_SHA256.jsonl',V63/'contract/EVALUATION_SHA256_DENYLIST.json',
        V64/'source_audit/YCB_TRAIN_QUERY_EVIDENCE.csv',V64/'source_audit/YCB_VALIDATION_QUERY_EVIDENCE.csv',
        ROOT/'contract/CONTROLLED_TRUTH_MOCK_PROOF.json',OUT/'COMPOSER_CORE_SMOKE.json']
    protocol_data=json.loads(protocol.read_text())
    for b in protocol_data['bindings']:check(b)
    sam_lock=json.loads((V63/'sam_source/LOCK.json').read_text())
    if sha(V63/'sam_source/INPUTS.jsonl')!=sam_lock['input_sha256']:raise ValueError('Original input-only query manifest changed')
    denied=set(json.loads((V63/'contract/EVALUATION_SHA256_DENYLIST.json').read_text())['image_sha256'])
    official={str((YCB/r['relative_path']).resolve()):r['sha256']for r in lines(YCB/'SELECTED_FILES_SHA256.jsonl')}
    def public_file(path):
        p=Path(path).resolve()
        if not p.is_relative_to((YCB/'data/train_real').resolve())or str(p)not in official:raise ValueError('Unregistered public source path')
        return dict(path=str(p),sha256=official[str(p)])
    pool=[]
    for split,path,scenes in [('TRAIN',YCB/'train_instances.jsonl',set(range(8))),('VALIDATION',YCB/'SOURCE_VAL_LOCKED_200.jsonl',{8,9})]:
        for r in lines(path):
            if r['scene_id']not in scenes or r['image_sha256']in denied:raise ValueError('Invalid split/current-eval source')
            image=public_file(r['image'])
            if image['sha256']!=r['image_sha256']:raise ValueError('Source image manifest hashes disagree')
            pool.append(dict(id=ident(r),split=split,scene_id=r['scene_id'],frame_id=r['frame_id'],instance_id=r['instance_id'],
                object_id=r['object_id'],image=image,visible=public_file(r['mask_vis']),full_proxy=public_file(r['mask_full']),
                raw_hw=[r['height'],r['width']],vis_pixels=r['vis_pixels'],source_license=r.get('license','MIT'),
                official_revision=r.get('official_revision'),source_mask_assumption='BOP rendered/depth mask, original-visible completeness; not human physical-corner truth'))
    pool.sort(key=lambda r:(r['split'],r['id']));by_id={r['id']:r for r in pool}
    if len(by_id)!=4200:raise ValueError('Expected frozen4200 original instance donor pool')
    targets=[]
    for r in lines(V63/'sam_source/INPUTS.jsonl'):
        if r['source']!='YCB':continue
        base=by_id[r['id']];split='TRAIN'if r['split'].upper()=='TRAIN'else'VALIDATION'
        if split!=base['split']or r['image_sha256']!=base['image']['sha256']:raise ValueError('Query/instance source identity mismatch')
        q=np.asarray(r['query'],np.float64)
        if q.shape!=(8,2)or not np.isfinite(q).all()or r['side']<=0:raise ValueError('Original query8 malformed')
        targets.append(dict(**base,query=r['query'],side=r['side'],box=r['box'],original_sample_ids=[r['id']+':'+str(k)for k in range(8)]))
    targets.sort(key=lambda r:(r['split'],r['id']))
    counts=Counter(r['split']for r in targets)
    if dict(counts)!={'TRAIN':1600,'VALIDATION':400}:raise ValueError('Expected unchanged2000 target instances')
    hashes={s:{r['image']['sha256']for r in pool if r['split']==s}for s in counts}
    if hashes['TRAIN']&hashes['VALIDATION']:raise ValueError('Source split image overlap')
    PROV.mkdir(exist_ok=True);(PROV/'code').mkdir(exist_ok=False)
    code=[Path(__file__),Path(core.__file__),contract]
    for p in code:shutil.copy2(p,PROV/'code'/p.name)
    write_lines(PROV/'TARGETS.jsonl',targets);write_lines(PROV/'OCCLUDER_POOL.jsonl',pool)
    recipe=dict(schema='V66_APPROVED_FIXED_PUBLIC_REAL_BINARY_COMPOSITOR_V1',seed=core.SEED,
        targets=2000,target_counts=dict(counts),donor_counts=dict(Counter(r['split']for r in pool)),max_composites_per_target=1,
        donor_eligibility='same original split, different image SHA AND scene/frame, different object_id; no extra outcome-filtered donors',
        search='anchor order SHA(seed,target,anchor,k), then up to32 donors SHA(seed,target,donor,id), then scale order [.25,.40,.60]; first accepted only',
        donor_limit=32,scales=list(core.SCALES),size='longest side=max(9,floor(scale*sqrt(original_input_box_area)+.5)); aspect kept, each dimension floor(size*ratio+.5)',
        cutout='tight bbox of original visible mask only; RGB INTER_LINEAR, binary mask INTER_NEAREST; no antialias/color/alpha blending',
        donor_anchor='row-major first argmax Euclidean distanceTransform of resized mask padded1zero; align to nearest original query pixel with integer translation',
        acceptance=dict(anchor='all25 native pixels origVis AND actual clipped paste',other_V='>=1 other valid original-visible query nearest pixel left unpasted',
            untouched_O='>=1 valid original-full-negative query nearest pixel left unpasted',visible_coverage=[.03,.60],minimum_visible_pixels_left=25),
        original_geometry='query8 floatvalues, ROI side and preexisting jittered source SAMbox unchanged; no GT extrema new prompt',
        certificate='DATA_LABEL_PROTOCOL plus pure controlled_truth helper; original-hidden full UNKNOWN retained; semantic self UNKNOWN',
        strict_core='erode2(origVis &actual paste), border0; strict query certificate requires actual25 pixels, never clipped border',
        conditioning_gap='Original source query sampling and jittered target roughbox are source-GT-derived. Donor cutout/mask and size are data construction only; no pasted/GT masks as future model input.',
        resampling='Native target RGB canvas unchanged; zero outside donor paste clipping; generated RGB lossless PNG',
        future_features='New features must be extracted from composite RGB after independent data admission and frozen feature protocol under the existing goal authorization; original P3 cannot be reused as altered-image input.',
        no_Pallox_admitted=True,no_model_or_backbone_or_training=True,no_private_reads=True,
        source_bindings=[binding(p)for p in inputs],code_bindings=[binding(p)for p in code],
        target_manifest=binding(PROV/'TARGETS.jsonl'),occluder_manifest=binding(PROV/'OCCLUDER_POOL.jsonl'))
    write(ROOT/'GENERATION_RECIPE.json',recipe)
    lock=dict(schema='V66_BEFORE_ANY_COMPOSITION_IMMUTABLE_SOURCE_LOCK',recipe=binding(ROOT/'GENERATION_RECIPE.json'),
        source_bindings=recipe['source_bindings'],code_bindings=recipe['code_bindings'],code_snapshots=[binding(PROV/'code'/p.name)for p in code],
        targets=recipe['target_manifest'],occluders=recipe['occluder_manifest'],private_images_opened=0,model_calls=0,GPU_calls=0,PnP_calls=0)
    write(PROV/'GENERATION_INPUT_LOCK.json',lock);print(json.dumps({'frozen':True,'lock':binding(PROV/'GENERATION_INPUT_LOCK.json'),'targets':dict(counts)}),flush=True)


class SourceImages:
    def __init__(self):self.cache=OrderedDict();self.verified={};self.decodes=Counter()
    def get(self,b,kind):
        key=(b['path'],kind)
        if key in self.cache:self.cache.move_to_end(key);return self.cache[key]
        check(b);self.verified[b['path']]=b['sha256']
        with Image.open(b['path'])as im:value=np.array(im.convert('RGB'))if kind=='RGB'else np.array(im)
        if kind!='RGB':
            if value.ndim!=2 or not set(np.unique(value))<={0,255}:raise ValueError('Source masks must be binary0/255')
            value=value>0
        self.cache[key]=value;self.decodes[kind]+=1
        while len(self.cache)>192:self.cache.popitem(last=False)
        return value


def validate_lock():
    lock=json.loads((PROV/'GENERATION_INPUT_LOCK.json').read_text())
    for b in [lock['recipe'],*lock['source_bindings'],*lock['code_bindings'],*lock['code_snapshots'],lock['targets'],lock['occluders']]:check(b)
    if sha(ROOT/'DATA_LABEL_PROTOCOL.json')!=PROTOCOL_SHA:raise ValueError('Controlled-label contract changed')
    return lock


def load_truth(lock):
    b=next(b for b in lock['code_bindings']if Path(b['path']).name=='controlled_truth.py');check(b)
    name='_v66_frozen_partial_truth_'+b['sha256'][:12];spec=importlib.util.spec_from_file_location(name,b['path'])
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module


def generate():
    lock=validate_lock();truth=load_truth(lock)
    if sha(ROOT/'contract/controlled_truth.py')!='0bfedfb48744aa9ae2bd16941afcb0ec568a464119a4c99177d97ccc21290d64':
        raise ValueError('Approved partial-label implementation changed')
    if(OUT/'BUILD_STARTED.json').exists():raise FileExistsError('Preserve complete/partial composition run')
    targets=lines(lock['targets']['path']);pool=lines(lock['occluders']['path'])
    denied=set(json.loads((V63/'contract/EVALUATION_SHA256_DENYLIST.json').read_text())['image_sha256'])
    if any(r['image']['sha256']in denied for r in targets+pool):raise ValueError('Current-eval SHA in source pool')
    evidence={}
    for split in ('TRAIN','VALIDATION'):
        for row in read_csv(V64/'source_audit'/f'YCB_{split}_QUERY_EVIDENCE.csv'):evidence[row['sample_id']]=row
    write(OUT/'BUILD_STARTED.json',dict(lock=binding(PROV/'GENERATION_INPUT_LOCK.json'),recipe=lock['recipe'],targets=len(targets),
        no_private_read=True,model_calls=0,GPU_calls=0,PnP_calls=0))
    for split in ('TRAIN','VALIDATION'):
        (OUT/'images'/split).mkdir(parents=True,exist_ok=False);(OUT/'labels'/split).mkdir(parents=True,exist_ok=False)
    cache=SourceImages();attempt_reasons=Counter();statuses=[];manifests=[];query_rows=[];output_bindings=[]
    summary=Counter();state_counts=Counter();partial_counts=Counter();start=time.monotonic();output_hashes={}
    recipe_hash=lock['recipe']['sha256']
    with(OUT/'TARGET_STATUS.jsonl').open('w')as sf,(OUT/'COMPOSITES.jsonl').open('w')as mf:
      for ti,target in enumerate(targets):
        rgb=cache.get(target['image'],'RGB');vis=cache.get(target['visible'],'MASK');full=cache.get(target['full_proxy'],'MASK')
        if rgb.shape!=(*vis.shape,3)or full.shape!=vis.shape or list(vis.shape)!=target['raw_hw']or(vis&~full).any():raise ValueError('Source target mask/RGB invariants fail')
        points=np.asarray(target['query'],np.float64);pixels,valid=core.query_pixels(points,vis.shape)
        anchors=[k for k,(px,py)in enumerate(pixels)if valid[k]and core.strict_square(vis,int(px),int(py))]
        anchors.sort(key=lambda k:core.rank_key(target['id'],'anchor',k))
        eligible=[d for d in pool if d['split']==target['split']and d['image']['sha256']!=target['image']['sha256']
            and(d['scene_id'],d['frame_id'])!=(target['scene_id'],target['frame_id'])and d['object_id']!=target['object_id']]
        donors=sorted(eligible,key=lambda d:core.rank_key(target['id'],'donor',d['id']))[:32]
        box=np.asarray(target['box'],np.float64);boxscale=float(np.sqrt((box[2]-box[0])*(box[3]-box[1])))
        accepted=None;attempts=0;local_reasons=Counter()
        for anchor in anchors:
          if accepted is not None:break
          for donor_rank,donor in enumerate(donors):
            if accepted is not None:break
            donor_rgb=cache.get(donor['image'],'RGB');donor_vis=cache.get(donor['visible'],'MASK');donor_full=cache.get(donor['full_proxy'],'MASK')
            if donor_rgb.shape!=(*donor_vis.shape,3)or list(donor_vis.shape)!=donor['raw_hw']or donor_full.shape!=donor_vis.shape or(donor_vis&~donor_full).any():raise ValueError('Source donor RGB/mask shape mismatch')
            for scale in core.SCALES:
                attempts+=1;length=max(9,int(np.floor(scale*boxscale+.5)))
                cutout=core.resize_visible_cutout(donor_rgb,donor_vis,length)
                if cutout is None:local_reasons['EMPTY_RESIZED_DONOR_VISIBLE_MASK']+=1;continue
                paste,donor_canvas,placement=core.paste_support(vis.shape,cutout,pixels[anchor])
                ok,reason,facts=core.candidate_checks(vis,full,paste,points,anchor)
                local_reasons[reason]+=1
                if ok:
                    accepted=dict(donor=donor,donor_rank=donor_rank,scale=scale,cutout=cutout,paste=paste,donor_canvas=donor_canvas,
                        placement=placement,facts=facts,anchor=anchor);break
        attempt_reasons.update(local_reasons)
        status=dict(target_id=target['id'],split=target['split'],object_id=target['object_id'],eligible_anchors=anchors,
            ranked_donors=[d['id']for d in donors],candidate_attempts=attempts,attempt_reasons=dict(local_reasons),accepted=accepted is not None)
        if accepted is None:
            reason='NO_STRICT_VISIBLE_ANCHOR'if not anchors else'NO_ELIGIBLE_DONOR'if not donors else'NO_ACCEPTABLE_CANDIDATE_WITHIN_FIXED_BUDGET'
            status['skip_reason']=reason;summary[target['split']+'_SKIPPED']+=1;summary['SKIP_'+reason]+=1
        else:
            a=accepted;paste=a['paste'];composite=core.compose_rgb(rgb,a['donor_canvas'],paste)
            cert=truth.construct_certainty(vis,full,paste,visible_negative_trusted=True)
            base=truth.construct_certainty(vis,full,np.zeros_like(paste),visible_negative_trusted=True)
            labels=truth.label_queries(cert,points);baseline=truth.label_queries(base,points)
            if not labels[a['anchor']]['radius2_strict_new_controlled_E_core']:raise AssertionError('Accepted anchor lost strict25pixel certificate')
            if(cert['exact_state'][full&~vis]!=-1).any():raise AssertionError('Originalhidden was relabelled')
            stable=hashlib.sha256((recipe_hash+':'+target['id']).encode()).hexdigest()[:20]
            composite_id='CTRL:'+target['id']+':'+stable
            image_path=OUT/'images'/target['split']/(stable+'.png');label_path=OUT/'labels'/target['split']/(stable+'.npz')
            if image_path.exists()or label_path.exists():raise FileExistsError('Composite stable ID collision')
            Image.fromarray(composite).save(image_path,format='PNG',compress_level=6)
            cert_arrays={k:v for k,v in cert.items()if isinstance(v,np.ndarray)}
            cert_arrays['strict_controlled_E_core_r2']=core.erode_core(cert['controlled_E_pixel'])
            cert_arrays['source_visible_boundary_r1']=cv2.dilate(vis.astype(np.uint8),np.ones((3,3),np.uint8))!=cv2.erode(vis.astype(np.uint8),np.ones((3,3),np.uint8))
            cert_arrays['source_full_boundary_r1']=cv2.dilate(full.astype(np.uint8),np.ones((3,3),np.uint8))!=cv2.erode(full.astype(np.uint8),np.ones((3,3),np.uint8))
            cert_arrays['paste_boundary_r1']=cv2.dilate(paste.astype(np.uint8),np.ones((3,3),np.uint8))!=cv2.erode(paste.astype(np.uint8),np.ones((3,3),np.uint8))
            np.savez_compressed(label_path,**cert_arrays)
            image_binding=binding(image_path);label_binding=binding(label_path)
            if image_binding['sha256']in denied:raise ValueError('Composite/current-eval image hash overlap')
            prior_split=output_hashes.get(image_binding['sha256'])
            if prior_split and prior_split!=target['split']:raise ValueError('Composite image hash overlaps TRAIN/VAL')
            if prior_split:summary['SAME_SPLIT_DUPLICATE_RGB']+=1
            output_hashes[image_binding['sha256']]=target['split'];output_bindings.extend([image_binding,label_binding])
            query_path=label_path.with_suffix('.queries.json')
            write(query_path,dict(composite_id=composite_id,source_target_id=target['id'],query=target['query'],side=target['side'],box=target['box'],
                original_zero_new_occlusion=baseline,controlled_composite=labels,label_protocol_sha256=PROTOCOL_SHA,
                original_visible_negative_completeness_assumption=True,semantic_self='UNKNOWN'))
            output_bindings.append(binding(query_path))
            c=a['cutout'];manifest=dict(id=composite_id,source='YCB_CONTROLLED_PUBLIC_REAL',split=target['split'],
                target_original=target,occluder_original=a['donor'],image=image_binding,labels=label_binding,query_labels=binding(query_path),
                query=target['query'],side=target['side'],box=target['box'],original_sample_ids=target['original_sample_ids'],
                label_protocol_sha256=PROTOCOL_SHA,recipe_sha256=recipe_hash,source_visible_negative_completeness_assumption=True,
                transform=dict(seed=core.SEED,donor_rank=a['donor_rank'],scale_multiplier=a['scale'],target_box_geomean=boxscale,
                    donor_visible_bbox_xyxy=c['bbox'],resized_hw=c['resized_hw'],source_crop_hw=c['source_crop_hw'],longest_side=c['longest_side'],
                    resize_scale_factor=c['scale_factor'],donor_anchor_xy=c['donor_anchor_xy'],donor_anchor_distance=c['donor_anchor_distance'],**a['placement']),
                acceptance=a['facts'],RGB_outside_paste_bit_exact=True,RGB_inside_paste_matches_resized_donor=True,composite_RGB_equals_original=bool(np.array_equal(composite,rgb)),
                changed_RGB_pixels_inside_paste=int(np.any(composite!=rgb,axis=2)[paste].sum()),
                original_hidden_full_is_UNKNOWN=True,GT_or_paste_masks_are_not_model_inputs=True,
                new_backbone_model_GPU_training_PnP_calls=0)
            mf.write(json.dumps(manifest,allow_nan=False)+'\n');mf.flush();manifests.append(manifest)
            for k,(label,old)in enumerate(zip(labels,baseline)):
                source_sample=target['original_sample_ids'][k];e=evidence[source_sample]
                row=dict(composite_id=composite_id,sample_id=composite_id+':'+str(k),original_sample_id=source_sample,
                    source_target_id=target['id'],split=target['split'],native_corner=k,query_x=target['query'][k][0],query_y=target['query'][k][1],
                    pixel_x=int(pixels[k,0]),pixel_y=int(pixels[k,1]),roi_side=target['side'],prompt_box_json=json.dumps(target['box']),
                    target_object_id=target['object_id'],occluder_object_id=a['donor']['object_id'],target_scene_id=target['scene_id'],target_frame_id=target['frame_id'],
                    target_original_image_sha256=target['image']['sha256'],composite_image_sha256=image_binding['sha256'],
                    donor_original_image_sha256=a['donor']['image']['sha256'],covered_visible_fraction=a['facts']['covered_fraction'],
                    valid=label['valid'],exact_state=label['exact_state'],radius2_state=label['radius2_state'],joint_known=label['joint_known'],joint_state=label['joint_state'],
                    joint_compatible_mask=label['joint_compatible_mask'],joint_compatible_codes_json=json.dumps(label['joint_compatible_codes']),
                    original_zero_occlusion_exact_state=old['exact_state'],original_zero_occlusion_radius2_state=old['radius2_state'],
                    original_zero_occlusion_joint_compatible_mask=old['joint_compatible_mask'],original_proxy_exact_state=int(cert['original_proxy_state'][pixels[k,1],pixels[k,0]])if valid[k]else-1,
                    exact_details_json=json.dumps(label.get('exact',{})),radius2_details_json=json.dumps(label.get('radius2',{})),
                    exact_controlled_E=label.get('exact_controlled_E',False),exact_original_hidden_proxy_unknown=label.get('exact_original_hidden_proxy_unknown',False),
                    radius2_contains_unknown_full=label.get('radius2_contains_unknown_full',False),radius2_E_uses_source_visible_negative_assumption=label.get('radius2_E_uses_source_visible_negative_assumption',False),
                    radius2_strict_new_controlled_E_core=label.get('radius2_strict_new_controlled_E_core',False),radius2_strict_known_O_core=label.get('radius2_strict_known_O_core',False),
                    radius2_clipped_by_frame=label.get('radius2_clipped_by_frame',False),is_anchor=k==a['anchor'],
                    is_unpasted_V_control=k in a['facts']['visible_control_queries'],is_untouched_O_control=k in a['facts']['outside_control_queries'],
                    original_visible_negative_assumption=True,semantic_self='UNKNOWN',corner_semantic_truth='UNKNOWN')
                for key in ('visible_boundary_r1','full_boundary_r1','visible_boundary_r2','full_boundary_r2','exact_enclosed_full_void','depth_missing_at_query',
                            'strict_external_corroborated_5x5','exact_external_ambiguous','visible_inside_distance_px','full_inside_distance_px'):
                    row['source_'+key]=e.get(key,'')
                query_rows.append(row);state_counts[(target['split'],'exact',label['exact_state'])]+=1;state_counts[(target['split'],'radius2',label['radius2_state'])]+=1
                partial_counts[(target['split'],label['joint_known'],label['joint_compatible_mask'])]+=1
            summary[target['split']+'_ACCEPTED']+=1
            status.update(composite_id=composite_id,image=image_binding,labels=label_binding,acceptance=a['facts'])
        sf.write(json.dumps(status,allow_nan=False)+'\n');sf.flush();statuses.append(status)
        if(ti+1)%100==0:print(json.dumps({'processed':ti+1,'total':len(targets),'counts':dict(summary),'seconds':round(time.monotonic()-start,1)}),flush=True)
    with(OUT/'QUERIES.csv').open('w',newline='')as f:
        writer=csv.DictWriter(f,fieldnames=list(query_rows[0]));writer.writeheader();writer.writerows(query_rows)
    verified=[dict(path=p,sha256=d)for p,d in sorted(cache.verified.items())]
    write_lines(PROV/'VERIFIED_SOURCE_FILES.jsonl',verified)
    audit=dict(schema='V66_PUBLIC_CONTROLLED_DATA_BUILD_AUDIT',completed=True,targets_requested=len(targets),target_status_rows=len(statuses),
        counts=dict(summary),accepted=len(manifests),skipped=len(targets)-len(manifests),query_rows=len(query_rows),candidate_attempt_reasons=dict(attempt_reasons),
        class_counts=[dict(split=s,scale=scale,state=state,N=count)for(s,scale,state),count in sorted(state_counts.items())],
        joint_known_compatibility_counts=[dict(split=s,joint_known=known,compatible_mask=mask,N=count)for(s,known,mask),count in sorted(partial_counts.items())],
        source_files_verified=len(verified),source_file_decodes=dict(cache.decodes),all_accepted_have_strict_new_E25=True,
        RGB_outside_actual_paste_unchanged=True,RGB_inside_paste_matches_transformed_donor=True,
        target_donor_same_split_different_image_frame_and_object=True,query8_side_box_unchanged=True,
        original_hidden_exact_UNKNOWN=True,current_eval_RGB_or_GT_or_scores_read=False,private_image_SHA_overlap=0,
        new_model_backbone_GPU_training_PnP_calls=0,seconds=time.monotonic()-start)
    write(OUT/'BUILD_AUDIT.json',audit)
    for p in (OUT/'TARGET_STATUS.jsonl',OUT/'COMPOSITES.jsonl',OUT/'QUERIES.csv',OUT/'BUILD_AUDIT.json',PROV/'VERIFIED_SOURCE_FILES.jsonl'):output_bindings.append(binding(p))
    validate_lock()
    write(OUT/'SEAL.json',dict(schema='V66_CONTROLLED_PUBLIC_REAL_DATA_GENERATION_SEAL',completed=True,lock=binding(PROV/'GENERATION_INPUT_LOCK.json'),
        recipe=lock['recipe'],label_protocol=binding(ROOT/'DATA_LABEL_PROTOCOL.json'),label_helper=binding(ROOT/'contract/controlled_truth.py'),
        requested_targets=len(targets),accepted=len(manifests),skipped=len(targets)-len(manifests),outputs=output_bindings,
        current_eval_RGB_GT_scores_read=False,model_backbone_GPU_training_PnP_calls=0,admission='HOLD_PENDING_INDEPENDENT_DATA_AUDIT',training_allowed=False))
    print(json.dumps({'completed':True,'accepted':len(manifests),'skipped':len(targets)-len(manifests),'seal':binding(OUT/'SEAL.json')}),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('freeze','generate'));args=p.parse_args()
    cv2.setNumThreads(1)
    if args.command=='freeze':freeze()
    else:generate()

if __name__=='__main__':main()
