"""V96 one image-only public run; no torch/model/feature/optimizer loader."""
from pathlib import Path
from collections import defaultdict,Counter,OrderedDict
import argparse,hashlib,importlib.util,json,shutil,time,traceback
import numpy as np
from PIL import Image,ImageDraw

HERE=Path(__file__).resolve().parent;V=HERE.parent;P=V.parent
PREP=V/'query_anchored_pair_preparation';SCOPE=V/'ROOT_ACTUAL_QUERY_ANCHORED_IMAGE_SCOPE.json'
SCOPE_SHA='9b4f938386f4f81aca4a53057bc9b6e4b5f07c9810d842b77e577c01189047af'
SOURCE=P/'query_visibility_same_q_counterfactual_20261023_v95/source_pair_intake/SOURCE960_DRIVER_LINEAGE.jsonl'
PROTOCOL=HERE/'IMAGE_ONLY_PROTOCOL_BEFORE_PIXELS_V2.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(p),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(x)for x in Path(p).read_text().splitlines()if x]
def clean(x):
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,dict):return {k:clean(v)for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [clean(v)for v in x]
    return x
def write(path,x):
    with Path(path).open('x')as f:json.dump(clean(x),f,indent=2,allow_nan=False);f.write('\n')
def jsonl(path,x):
    with Path(path).open('x')as f:
        for r in x:f.write(json.dumps(clean(r),separators=(',',':'),allow_nan=False)+'\n')
def check(b):assert sha(b['path'])==b['sha256'],b['path'];return Path(b['path'])
def pixels(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def kernel():
    path=PREP/'query_anchored_pair_kernel.py';s=importlib.util.spec_from_file_location('v96_exact_kernel',path)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def freeze_metadata():
    assert sha(SCOPE)==SCOPE_SHA
    scope=read(SCOPE);assert scope['root_approved']and scope['actual_image_construction_allowed']and scope['feature_model_training_current_PnP_allowed']is False
    for key in ['root_actual_crop_review','independent_inventory_review','inventory_seal','metadata_assignment_lock','kernel','generated']:check(scope[key])
    lock=read(scope['metadata_assignment_lock']['path'])
    for b in lock['files']:check(b)
    proto=read(PREP/'PAIR_PROTOCOL_BEFORE_ASSIGNMENT_AND_SOURCE_PIXELS.json')
    pool=lines(check(proto['inputs']['donor_pool']));pool_by_id={d['id']:d for d in pool};byframe=defaultdict(list)
    for d in pool:byframe[(d['scene_id'],d['frame_id'])].append(d)
    targets=lines(PREP/'MATCHED_TARGET960_GT_ONLY.jsonl');assign=lines(PREP/'DONOR_ASSIGNMENT960_K16_BEFORE_PIXELS.jsonl')
    needed={r['donor_id']for t,a in zip(targets,assign)if t['construction_eligible']for r in a['ranks']}
    complete=[];gtcache={}
    for sid,fid in sorted({(pool_by_id[x]['scene_id'],pool_by_id[x]['frame_id'])for x in needed}):
        available=sorted(byframe[sid,fid],key=lambda d:d['instance_id'])
        gp=Path(available[0]['visible_alpha_source']['path']).parent.parent/'scene_gt.json'
        if str(gp)not in gtcache:gtcache[str(gp)]=read(gp)
        expected=gtcache[str(gp)].get(str(fid),[]);ids=[d['instance_id']for d in available]
        objects_ok=ids==list(range(len(expected)))and all(d['object_id']==expected[d['instance_id']]['obj_id']for d in available)
        exists=all(Path(d['visible_alpha_source']['path']).is_file()for d in available)
        complete.append(dict(scene_id=sid,frame_id=fid,scene_GT=bind(gp),expected_instance_ids=list(range(len(expected))),available_instance_ids=ids,
            object_ids_match=objects_ok,all_expected_masks_exist=exists,complete=objects_ok and exists,
            instance_masks=[dict(instance_id=d['instance_id'],object_id=d['object_id'],mask=d['visible_alpha_source'])for d in available]))
    completeness=dict(completed=True,unique_selected_donors=len(needed),frames=complete,incomplete_frames=sum(not f['complete']for f in complete),source_pixel_reads=0)
    cp=HERE/'COMPLETE_DONOR_INSTANCE_METADATA_BEFORE_PIXELS.json'
    if cp.exists():assert read(cp)==completeness
    else:write(cp,completeness)
    bindings={k:bind(PREP/n)for k,n in [('input_plan','INPUT_PLAN31680_INPUT_ONLY.jsonl'),('target_GT','MATCHED_TARGET960_GT_ONLY.jsonl'),('assignments','DONOR_ASSIGNMENT960_K16_BEFORE_PIXELS.jsonl'),('kernel','query_anchored_pair_kernel.py'),('generated','GENERATED_QUERY_ANCHORED_PAIR_TESTS.json')]}
    bindings.update(rows960=proto['inputs']['rows960'],donor_pool=proto['inputs']['donor_pool'],source960=bind(SOURCE),complete_instances=bind(HERE/'COMPLETE_DONOR_INSTANCE_METADATA_BEFORE_PIXELS.json'))
    write(PROTOCOL,dict(completed=True,root_scope=bind(SCOPE),runner=bind(__file__),metadata_assignment_lock=scope['metadata_assignment_lock'],bindings=bindings,
        original_rows=960,occurrences=31680,K=16,eligible_attempts=976,source_only=True,training_admission=False,current_allowed=False,
        run_policy='One pass all976 D rank attempts, all15360 rank records/all31680 occurrences. No replacement/resize search/query change; retain failures. E4 BEFORE only.',
        opacity='Complete BOP instance set; selected&~all_other, native erode1, unchanged pinned geometry/mass. RGB boundary interpolation is not pure optical ownership or human gold.',
        preview_policy='First attempted base/rank per source/rawsplit, plus first successful pair per cell if different; metadata order, no model scores. Raw exact RGB96 plus annotated derivatives.',
        resource_policy='LRU source RGB4 and mask24 files; before sources read only as needed for eligible construction. No P3/full features.'))
    print(json.dumps(dict(stage='PRE_PIXEL_BINDINGS_FROZEN',protocol=bind(PROTOCOL),selected_donors=len(needed),donor_frames=len(complete),incomplete_frames=sum(not f['complete']for f in complete))),flush=True)

def main():
    protocol=read(PROTOCOL);assert sha(SCOPE)==SCOPE_SHA
    check(protocol['root_scope']);assert read(SCOPE)['actual_image_construction_allowed']is True
    check(protocol['runner']);check(protocol['metadata_assignment_lock'])
    for b in protocol['bindings'].values():check(b)
    assert protocol['runner']==bind(__file__)and not(HERE/'RUN_STARTED.json').exists()
    k=kernel();B=protocol['bindings'];rows=lines(B['rows960']['path']);source=lines(B['source960']['path']);targets=lines(B['target_GT']['path'])
    assign=lines(B['assignments']['path']);plan=lines(B['input_plan']['path']);pool=lines(B['donor_pool']['path']);donors={d['id']:d for d in pool}
    frames={(x['scene_id'],x['frame_id']):x for x in read(B['complete_instances']['path'])['frames']}
    assert len(rows)==len(source)==len(targets)==len(assign)==960 and len(plan)==31680
    assert sum(t['construction_eligible']for t in targets)==61
    for i,(r,s,t,a)in enumerate(zip(rows,source,targets,assign)):
        assert r['slot_index']==s['slot_index']==t['base_row_index']==a['base_row_index']==i
        assert r['query_xy']==s['query_xy']==t['original_q']and r['L']==s['L']==t['original_L']
        assert [d['id']for d in k.fixed_donors(r,pool)]==[d['donor_id']for d in a['ranks']]
    for name in ['donor_certificates','protected_masks','edited_fullRGB','fixed_order_preview']:(HERE/name).mkdir(exist_ok=False)
    write(HERE/'RUN_STARTED.json',dict(completed=False,protocol=bind(PROTOCOL),runner=bind(__file__),actual_model_feature_fit_current_PnP=0))
    masks=OrderedDict();rgbs=OrderedDict();donor_cert={};target_cert={};feasibility=[];occurrences=[];GT=[];attempted=0;success=0;start=time.monotonic()
    first_attempt={};first_success={}
    def mask_file(b,hw):
        path=str(check(b))
        if path not in masks:
            a=np.array(Image.open(path));assert a.dtype==np.uint8 and a.shape==tuple(hw)and np.isin(a,[0,255]).all();masks[path]=a>0
            if len(masks)>24:masks.popitem(last=False)
        else:masks.move_to_end(path)
        return masks[path]
    def rgb_file(b,fmt='PNG_RGB'):
        path=str(check(b))
        if path not in rgbs:
            a=np.load(path,allow_pickle=False)if fmt=='CANONICAL_RGB_UINT8_HWC_NPY'else np.array(Image.open(path).convert('RGB'),copy=True)
            assert a.dtype==np.uint8 and a.ndim==3 and a.shape[2]==3;rgbs[path]=a
            if len(rgbs)>4:rgbs.popitem(last=False)
        else:rgbs.move_to_end(path)
        return rgbs[path]
    def donor_data(d):
        owner=frames[d['scene_id'],d['frame_id']]
        if not owner['complete']:return None,None,None,dict(d,foreign_opaque_cutout_admitted=False)
        selected=mask_file(d['visible_alpha_source'],d['raw_hw']);other=np.zeros_like(selected)
        for m in owner['instance_masks']:
            if m['instance_id']!=d['instance_id']:other|=mask_file(m['mask'],d['raw_hw'])
        exclusive,ec=k.alpha_rule.exclusive_alpha(selected,other,True)
        admitted=ec['admitted']and int(selected.sum())==d['metadata_vis_pixels']
        meta=dict(d,foreign_opaque_cutout_admitted=bool(admitted));image=rgb_file(d['image'])
        if d['id']not in donor_cert:
            safe=d['id'].replace(':','_');npz=HERE/'donor_certificates'/f'{safe}.npz'
            padded=np.pad(exclusive,1);core=np.logical_and.reduce([padded[y:y+exclusive.shape[0],x:x+exclusive.shape[1]]for y in range(3)for x in range(3)])
            np.savez_compressed(npz,selected_visible=selected,other_visible_union=other,exclusive_alpha=exclusive,erode1_exclusive_alpha=core)
            donor_cert[d['id']]=dict(id=d['id'],split=d['split'],scene_id=d['scene_id'],frame_id=d['frame_id'],instance_id=d['instance_id'],object_id=d['object_id'],
                image=d['image'],mask=d['visible_alpha_source'],decoded_RGB_sha256=pixels(image),metadata_visible_pixels=d['metadata_vis_pixels'],
                exclusive_alpha=ec,foreign_opaque_cutout_admitted=bool(admitted),complete_owner_frame=[d['scene_id'],d['frame_id']],native_alpha_proof=bind(npz),replacement_allowed=False)
        return image,selected,other,meta
    def base_image(i):
        s=source[i];a=rgb_file(s['fullRGB_file_binding'],s['fullRGB_format']);assert a.shape==(*rows[i]['raw_hw'],3)and pixels(a)==s['decoded_RGB_sha256']
        target_cert[s['fullRGB_path']]=dict(file=s['fullRGB_file_binding'],format=s['fullRGB_format'],decoded_RGB_sha256=pixels(a),raw_hw=rows[i]['raw_hw']);return a
    try:
        for i,(row,t,s,assignment)in enumerate(zip(rows,targets,source,assign)):
            before_sha=s['decoded_RGB_sha256'];native=base_image(i)if t['construction_eligible']else None
            def output_occ(j,image=None,label=-1,pair_ready=False,reason='INVENTORY_NOT_D_AFTER_U'):
                o=dict(plan[33*i+j]);assert o['occurrence_index']==33*i+j
                changed=image is not None and not np.array_equal(image,native)
                digest=pixels(image)if changed else before_sha;file=None
                if changed:
                    file_path=HERE/'edited_fullRGB'/f'occ_{33*i+j:05d}.png';Image.fromarray(image).save(file_path);file=bind(file_path)
                o.update(slot_index=i,query_xy=row['query_xy'],L=row['L'],image_sha256=file['sha256']if file else row['image_sha256'],
                    image_status='ACTUAL_CHANGED_FULLRGB'if changed else'EXACT_UNCHANGED_BEFORE_REUSE',
                    full_rgb_changed=bool(changed),edited_fullRGB_file=file,edited_fullRGB_format='PNG'if file else None,
                    full_rgb_decoded_sha256=digest,before_decoded_sha256=before_sha,base_ref=dict(source_driver_lineage=str(SOURCE),row_index=i))
                occurrences.append(o)
                GT.append(dict(occurrence_index=o['occurrence_index'],occurrence_id=o['occurrence_id'],base_row_index=i,slot_index=i,donor_rank=o['donor_rank'],variant=o['variant'],
                    source=t['source_family'],raw_split=row['split'],candidate_aux_hidden_label_GT_only=label,known_candidate_GT_only=label>=0,
                    query_anchored_before_label=t['query_anchored_label'],original_C1_V95_label=t['original_C1_V95_label'],pair_ready=pair_ready,pair_reason=reason,
                    native_expected_part_state='UNKNOWN',human_gold=False,training_admission=False))
            output_occ(0,label=t['query_anchored_label'],reason='ORIGINAL_QUERY_ANCHORED_BEFORE')
            for rank,dref in enumerate(assignment['ranks']):
                assert dref['donor_rank']==rank
                if not t['construction_eligible']:
                    facts=dict(pair_ready=False,reason='QUERY_ANCHORED_BEFORE_NOT_D_NO_EDIT',protected_pixels=0,protected_area_over_L_squared=None)
                    pair=None
                else:
                    donor=donors[dref['donor_id']];drgb,dmask,other,dmeta=donor_data(donor)
                    result=k.prepare_rank(native,row,t,drgb,dmask,other,frames[donor['scene_id'],donor['frame_id']]['complete'],dmeta)
                    facts=clean(result['facts']);pair=result['images'];attempted+=1;success+=int(facts['pair_ready'])
                    if rank==0:
                        pm,why=k.protected_mask(row,t);path=HERE/'protected_masks'/f'base_{i:04d}.npz';np.savez_compressed(path,protected_union=pm)
                    facts['protected_union_file']=bind(HERE/'protected_masks'/f'base_{i:04d}.npz')
                    key=(t['source_family'],row['split']);first_attempt.setdefault(key,(i,rank))
                    if facts['pair_ready']:first_success.setdefault(key,(i,rank))
                    if attempted%32==0:print(json.dumps(dict(stage='ELIGIBLE_D_RANK_ATTEMPTS',done=attempted,total=976,pairs_ready=success,seconds=time.monotonic()-start)),flush=True)
                facts.update(base_row_index=i,donor_rank=rank,assigned_donor_id=dref['donor_id'],source=t['source_family'],raw_split=row['split'],original_q=row['query_xy'],original_L=row['L'])
                feasibility.append(facts)
                ready=bool(facts['pair_ready'])
                output_occ(1+2*rank,None if pair is None else pair[0],1 if ready else-1,ready,facts['reason'])
                output_occ(2+2*rank,None if pair is None else pair[1],0 if ready else-1,ready,facts['reason'])
        assert attempted==976 and len(occurrences)==len(GT)==31680 and len(feasibility)==15360
        assert [r['occurrence_index']for r in occurrences]==list(range(31680))
        jsonl(HERE/'OCCURRENCES31680_LINEAGE.jsonl',occurrences);jsonl(HERE/'OCCURRENCES31680_GT_ONLY.jsonl',GT)
        jsonl(HERE/'FEASIBILITY15360_GT_ONLY.jsonl',feasibility);jsonl(HERE/'SELECTED_DONOR_RGBA_CERTIFICATES.jsonl',list(donor_cert.values()))
        jsonl(HERE/'SOURCE_RGB_SHA_CERTIFICATES.jsonl',list(target_cert.values()))
        shutil.copyfile(SOURCE,HERE/'SOURCE960_DRIVER_LINEAGE.jsonl')
        jsonl(HERE/'ORIGINAL_ANCHORED_AND_C1_V95_LABELS960_GT_ONLY.jsonl',[dict(base_row_index=i,query_anchored_label=t['query_anchored_label'],original_C1_V95_label=t['original_C1_V95_label'],native_expected_part_state='UNKNOWN',human_gold=False)for i,t in enumerate(targets)])
        crop,_=k.V1.frozen_rgb96_operator(P/'query_visibility_loco_native_context_20261023_v74/native_context_cache/compile_input_pack.py','802f2e17f1ee98ae09d8468b8ab749653a50054a85f24ef0af16385a7c262b66')
        selected=sorted(set(first_attempt.values())|set(first_success.values()));preview=[]
        for number,(i,rank)in enumerate(selected):
            row=rows[i];native=base_image(i);imgs=[native]
            for j in [1+2*rank,2+2*rank]:
                o=occurrences[33*i+j];imgs.append(rgb_file(o['edited_fullRGB_file'])if o['full_rgb_changed']else native)
            canvas=Image.new('RGB',(972,610),'white');draw=ImageDraw.Draw(canvas)
            f=feasibility[16*i+rank];draw.text((8,5),f"{targets[i]['source_family']} {row['split']} base{i} rank{rank} ready={f['pair_ready']} | {f['reason']}",fill='black')
            for j,(name,image)in enumerate(zip(['BEFORE','COVER','SHAM'],imgs)):
                full=Image.fromarray(image);full.thumbnail((316,245));canvas.paste(full,(j*324,30))
                rgb96,valid=crop(image,np.asarray(row['query_xy']),row['L'],96);rawpath=HERE/'fixed_order_preview'/f'case_{number:02d}_{name}_rgb96.png'
                Image.fromarray(rgb96.transpose(1,2,0)).save(rawpath)
                zoom=Image.fromarray(rgb96.transpose(1,2,0)).resize((288,288),Image.Resampling.NEAREST);canvas.paste(zoom,(j*324+12,295));draw.text((j*324+12,278),name,fill='black')
                draw.ellipse((j*324+12+141,295+141,j*324+12+147,295+147),outline='red',width=2)
            path=HERE/'fixed_order_preview'/f'case_{number:02d}.png';canvas.save(path)
            preview.append(dict(base_row_index=i,donor_rank=rank,pair_ready=f['pair_ready'],reason=f['reason'],figure=bind(path),selection='first attempted or first successful in source/rawsplit, fixed metadata order'))
        write(HERE/'fixed_order_preview/PREVIEW_MANIFEST.json',dict(completed=True,cases=preview,actual_views_by_builder=0,source_pixels_only=True,no_model_scores=True))
        signatures=defaultdict(set)
        for o,g in zip(occurrences,GT):
            if g['known_candidate_GT_only']:signatures[(o['full_rgb_decoded_sha256'],json.dumps(o['original_q']),o['original_L'])].add(g['candidate_aux_hidden_label_GT_only'])
        stats=dict(completed=True,PASS=True,source_only=True,base_rows=960,occurrences=31680,eligible_attempts=attempted,accepted_pairs=success,
            before_D_E_U=dict(Counter(str(t['query_anchored_label'])for t in targets)),all_GT_labels=dict(Counter(str(g['candidate_aux_hidden_label_GT_only'])for g in GT)),
            changed_fullRGB_occurrences=sum(o['full_rgb_changed']for o in occurrences),attempt_reasons=dict(Counter(f['reason']for f in feasibility if targets[f['base_row_index']]['construction_eligible'])),
            by_source_raw_split={s+'|'+sp:dict(attempts=sum(f['source']==s and f['raw_split']==sp and targets[f['base_row_index']]['construction_eligible']for f in feasibility),accepted=sum(f['source']==s and f['raw_split']==sp and f['pair_ready']for f in feasibility))for s,sp in sorted({(t['source_family'],r['split'])for t,r in zip(targets,rows)})},
            complete_owner_frames=len(frames),donor_certificates=len(donor_cert),known_fullRGB_qL_label_collisions=sum(len(v)>1 for v in signatures.values()),
            original_q_L_folds_boxes_ranks_unchanged=True,all_failures_retained=True,natural_E4_BEFORE_only=True,training_admission=False,native_expected_part_state='UNKNOWN',human_gold=False,
            source_RGB_mask_only_no_features_models_fit_current_PnP=0,seconds=time.monotonic()-start,preview=bind(HERE/'fixed_order_preview/PREVIEW_MANIFEST.json'))
        write(HERE/'IMAGE_ONLY_SUMMARY.json',stats)
        files=[bind(p)for p in sorted(HERE.rglob('*'))if p.is_file()and p.name!='IMAGE_ONLY_INTAKE_SEAL.json'and '__pycache__'not in str(p)]
        write(HERE/'IMAGE_ONLY_INTAKE_SEAL.json',dict(completed=True,PASS=True,root_scope=protocol['root_scope'],protocol=bind(PROTOCOL),metadata_assignment_lock=protocol['metadata_assignment_lock'],
            source_bindings=B,files=files,summary=bind(HERE/'IMAGE_ONLY_SUMMARY.json'),training_admission=False,feature_model_fit_current_PnP=0))
        print(json.dumps(dict(stage='COMPLETE',summary=stats,seal=bind(HERE/'IMAGE_ONLY_INTAKE_SEAL.json'))),flush=True)
    except BaseException:
        write(HERE/'FAILED_EXECUTION.json',dict(completed=False,attempted=attempted,success=success,exception=traceback.format_exc(),automatic_restart=False))
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze-metadata',action='store_true');a=p.parse_args()
    freeze_metadata()if a.freeze_metadata else main()
