"""Root-admitted source controls -> exact saved raw96 + recomputed frozen SAM.

No source body, checkpoint or model is opened on import or --prepare. Root's
separate cache entry must bind the complete census/admission manifest first.
"""
import argparse, os, time, traceback
import numpy as np
import direct_source_common as c
H,V,P=c.H,c.V,c.P
PROTOCOL=H/'PROTOCOL_BEFORE_DIRECT_CONTROL_CACHE_OR_FIT.json'
DOMAINS={'SYNTH_FINITE_END':1024,'REAL_SEMANTIC_END_WEAK':66}
VARIANTS=('BEFORE','COVER','SHAM')

def source_population(admission):
    c.need(admission['completed'] is True and admission['PASS'] is True and admission['source_only'] is True and admission['training_admission_direct_end'] is True,'Separate root direct-end source admission')
    c.need({s['domain'] for s in admission['sources']}==set(DOMAINS) and len(admission['sources'])==2,'All synthetic/real source domains')
    physical={};census=[];closures=[]
    for source in admission['sources']:
        domain=source['domain'];done=c.read_bound(source['complete']);review=c.read_bound(source['independent_review'])
        c.need(done['completed'] is True and done['PASS'] is True and review['completed'] is True and review['PASS'] is True,'Closed source controls and independent audit')
        key='attempts' if domain=='SYNTH_FINITE_END' else 'manifest'
        c.need(done[key]==source['census'] and source['expected_attempts']==DOMAINS[domain],'Exact full source census authority')
        rows=c.lines(source['census']);c.need(len(rows)==DOMAINS[domain],'Full attempt count, not only supported subset')
        ids=[]
        for r in rows:
            aid=r['attempt_id'] if domain=='SYNTH_FINITE_END' else r['pair_id'];ids.append(aid)
            supported=(r['direct_end_states']==[0,1,0]) if domain=='SYNTH_FINITE_END' else (r['PASS'] is True and r['direct_end_weak_states']==dict(BEFORE='VISIBLE_WEAK',COVER='EXTERNALLY_BLOCKED_WEAK',SHAM='VISIBLE_WEAK'))
            selected=r['native_slot'] if domain=='SYNTH_FINITE_END' else r['original_input']['native_slot']
            for vi,var in enumerate(VARIANTS):
                packet=r['outputs'] if domain=='SYNTH_FINITE_END' else r['variants'][{'SHAM':'ADJACENT_SHAM'}.get(var,var)]
                if packet is not None and 'path' in packet:
                    physical[(domain,aid,var)]=dict(packet=packet,layout='SYNTH_TRIPLET' if domain=='SYNTH_FINITE_END' else 'REAL_SINGLE',variant_index=vi,selected=selected,supported=supported,status=r['status'])
                census.append(dict(domain=domain,attempt_id=aid,variant=var,source_status=r['status'],selected_native_slot=selected,source_supported=supported,source_packet=packet if packet is not None and 'path' in packet else None,cache_row_indices=None,direct_training_admitted=False,new_native_visibility_target=-1,new_quality_target=-1,new_owner_target=-1))
        c.need(len(set(ids))==len(ids),'Unique source attempts, source global-index repeats remain valid')
        closures.append(source)
    groups=admission['groups'];c.need(len({g['group_id'] for g in groups})==len(groups),'Unique control group identities')
    c.need({(g['domain'],g['attempt_id'],g['variant']) for g in groups}==set(physical) and len(groups)==len(physical),'Every physically saved variant included; missing variants never fabricated')
    admitted={}
    for g in groups:
        key=(g['domain'],g['attempt_id'],g['variant']);r=physical[key]
        c.need(g['source_packet']==r['packet'] and g['layout']==r['layout'] and g['selected_native_slot']==r['selected'] and g['source_status']==r['status'],'Exact source payload/status/native join')
        c.need(g['layout']!='SYNTH_TRIPLET' or g['variant_index']==r['variant_index'],'Exact original triplet variant index')
        c.need(type(g['admitted']) is bool and g['admitted_target']==((1 if g['variant']=='COVER' else 0) if g['admitted'] else -1),'Root direct admission remains separate from old labels')
        c.need(not g['admitted'] or r['supported'],'No UNKNOWN/failure automatically admitted')
        c.need(g['authority'].startswith('CONSTRUCTIVE_SOURCE_FIXED_NEAR_ANCHOR_END') if g['domain']=='SYNTH_FINITE_END' else g['authority'].startswith('SOURCE_RGB_SEMANTIC_END_WEAK'),'Distinct finite-synthetic and assistant-real authority')
        mm=g['metadata8'];c.need(len(mm)==8 and [m['native_slot'] for m in mm]==list(range(8)),'All eight original slots retained')
        c.need(all(m['row_id']==g['group_id']+'/native%d'%m['native_slot'] for m in mm),'Unique occurrence identity, original indices separate')
        c.need(not g['admitted'] or mm[g['selected_native_slot']]['original_query_available'] is True,'Original inputU cannot become known')
        admitted.setdefault((g['domain'],g['attempt_id']),[]).append((g['variant'],g['admitted']))
    for rows in admitted.values():
        if any(v for _,v in rows):c.need(len(rows)==3 and all(v for _,v in rows),'Admit complete verified triplet, not outcome-selected variants')
    return groups,census,closures

def load_pixels(g):
    if g['layout']=='SYNTH_TRIPLET':
        a=c.arrays(g['source_packet'],('full_RGB','rgb96','valid96','query_xy','L','native_slot','original_global_index','original_occurrence_id','original_query_available'))
        v=g['variant_index'];c.need(a['full_RGB'].ndim==4 and a['full_RGB'].shape[0]==3 and a['rgb96'].shape==(3,8,96,96,3) and a['valid96'].shape==(3,8,96,96),'Saved synthetic exact three variants')
        rgb,raw,valid=a['full_RGB'][v],a['rgb96'][v],a['valid96'][v]
    else:
        a=c.arrays(g['source_packet'],('full_RGB','rgb96','valid96','query_xy','L','native_slot','original_global_index','query_id','query_available'))
        a['original_occurrence_id']=a.pop('query_id');a['original_query_available']=a.pop('query_available')
        rgb,raw,valid=a['full_RGB'],a['rgb96'],a['valid96']
    c.need(rgb.dtype==np.uint8 and rgb.shape==tuple(g['raw_hw'])+(3,) and raw.dtype==np.uint8 and raw.shape==(8,96,96,3) and valid.dtype==bool and valid.shape==(8,96,96),'Original RGB/valid packet schema')
    mm=g['metadata8']
    expected=dict(query_xy=np.asarray([c.qarray(m['query_xy']) for m in mm],np.float64),L=np.asarray([np.nan if m['L'] is None else m['L'] for m in mm],np.float64),native_slot=np.arange(8,dtype=np.int8),original_global_index=np.asarray([m['original_global_index'] for m in mm],np.int64),original_occurrence_id=np.asarray([m['original_occurrence_id'] for m in mm]),original_query_available=np.asarray([m['original_query_available'] for m in mm],bool))
    for k in expected:
        # Original real packet's source global-index is int32. This is an
        # explicit lossless metadata widening, never a new concatenated ID.
        value=a[k].astype(np.int64) if k=='original_global_index' else a[k]
        c.need(c.equal(value,expected[k]),'Exact original packet q/L/native/identity/availability '+k)
    return np.ascontiguousarray(rgb),raw,valid,expected

def build(entry_path):
    e=c.read(entry_path);c.need(e['approved_direct_control_cache'] is True and e['source_only'] is True,'Root cache entry required')
    p=c.read_bound(e['protocol']);c.need(e['code']==c.bind(__file__) and p['cache_code']==e['code'],'Frozen cache source/protocol')
    for b in p['source_bindings'].values():c.checked(b)
    admission=c.read_bound(e['admission']);groups,census,closures=source_population(admission)
    c.read_bound(e['root_scope']);out=c.fresh_output(e['output_directory']);start=time.monotonic()
    c.write(out/'STARTED.json',dict(entry=c.bind(entry_path),code=c.bind(__file__),admission=e['admission'],source_only=True))
    c.write_rows(out/'GROUPS_BEFORE_SAM_OR_TARGET_FORWARD.jsonl',groups)
    helper,old,pin=c.sam_helpers();sample=c.sampled_helpers()['sample_sam']
    checkpoint=old['checkpoint'];c.need(e['checkpoint']==checkpoint,'Exact inherited frozen SAM checkpoint')
    model=None;allrows=[];savedgroups=[]
    try:
        c.checked(checkpoint)
        c.need(helper['git']('rev-parse','HEAD')==pin['exact_commit'] and helper['git']('status','--porcelain','--untracked-files=no')=='','Frozen SAM source unchanged')
        for b in pin['source_files']:c.checked(b)
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';torch,Predictor,registry,runtime=helper['import_runtime']()
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        guard=helper['actual_cuda_cumsum_guard'](torch);c.need(guard['PASS'],'Inherited CUDA cumsum guard');c.write(out/'CUDA_CUMSUM_GUARD.json',guard)
        state=torch.load(checkpoint['path'],map_location='cpu',weights_only=True);model=registry['vit_b'](checkpoint=None);model.load_state_dict(state,strict=True);del state
        model.eval().requires_grad_(False).float().cuda();initial=helper['state_sha'](model);predictor=Predictor(model)
        n=8*len(groups);x=dict(rgb96=np.empty((n,96,96,3),np.uint8),valid96=np.empty((n,96,96),bool),sam_mask96=np.zeros((n,96,96),bool),sam_available96=np.zeros((n,96,96),bool))
        targets=np.full(n,-1,np.int8);authorities=[];numeric={k:[] for k in c.META};reuse={};calls=0;repeat=None
        for gi,g in enumerate(groups):
            rgb,raw,valid,meta=load_pixels(g);decoded=c.ash(rgb);box,reason=helper['prompt_box'](meta['query_xy'])
            sm=dict(g,decoded_RGB_sha256=decoded,prompt_available=box is not None,prompt_reason=reason,runtime_available=False,outputs=None)
            a=None
            if box is not None:
                # Any changed fullRGB has a distinct key and must be forwarded.
                # Exact RGB+box duplicates may reuse only the raw4 result;
                # every group receives fresh original-metadata arrays.
                key=(rgb.shape,decoded,c.ash(box));reused=key in reuse
                if reused:a={k:v.copy() for k,v in reuse[key].items()}
                else:
                    with torch.inference_mode():
                        predictor.set_image(rgb,image_format='RGB');audit={}
                        z,q,low=helper['predict_warn_only_call'](torch,lambda:predictor.predict(point_coords=None,point_labels=None,box=box.copy(),mask_input=None,multimask_output=False,return_logits=True),audit)
                    c.need(z.shape==(1,*rgb.shape[:2]) and q.shape==(1,) and low.shape==(1,256,256) and all(v.dtype==np.float32 and np.isfinite(v).all() for v in (z,q,low)),'Exact frozen SAM output')
                    a=dict(image_logits=z[0].copy(),lowres_logits=low[0].copy(),predicted_quality=q.copy(),mask_logits_gt0=z[0]>0);calls+=1;reuse[key]={k:v.copy() for k,v in a.items()}
                    if repeat is None:
                        with torch.inference_mode():
                            predictor.set_image(rgb,image_format='RGB');bq={}
                            zz,qq,ll=helper['predict_warn_only_call'](torch,lambda:predictor.predict(point_coords=None,point_labels=None,box=box.copy(),mask_input=None,multimask_output=False,return_logits=True),bq)
                        repeat=dict(PASS=c.equal(zz[0],a['image_logits']) and c.equal(qq,a['predicted_quality']) and c.equal(ll[0],a['lowres_logits']),group_id=g['group_id']);c.need(repeat['PASS'],'Fixed first eligible SAM exact repeat');calls+=1
                packet=dict(a,box_xyxy=box.copy(),original_query_xy=meta['query_xy'],original_L=meta['L'],native_slot=meta['native_slot'],original_global_index=meta['original_global_index'],original_occurrence_id=meta['original_occurrence_id'],original_query_available=meta['original_query_available'],prediction_available=meta['original_query_available'].copy())
                sm.update(runtime_available=True,raw4_reused_after_exact_RGB_box_equality=reused,outputs=c.save_arrays(out/('SAM_%05d.npz'%gi),packet))
            for k,m in enumerate(g['metadata8']):
                i=8*gi+k;x['rgb96'][i]=raw[k];x['valid96'][i]=valid[k]
                if a is not None:x['sam_mask96'][i],x['sam_available96'][i]=sample(a['image_logits'],m['query_xy'],m['L'],m['original_query_available'],g['raw_hw'])
                target=g['admitted_target'] if k==g['selected_native_slot'] else -1;targets[i]=target;authorities.append(g['authority'])
                row=dict(m,domain=g['domain'],group_id=g['group_id'],attempt_id=g['attempt_id'],variant=g['variant'],authority=g['authority'],source_packet=g['source_packet'],source_status=g['source_status'],root_admitted=g['admitted'],raw_RGB96_sha256=c.ash(raw[k]),raw_valid96_sha256=c.ash(valid[k]))
                allrows.append(row)
                for key in c.META:numeric[key].append(m[key])
            savedgroups.append(sm)
        x.update(row_id=np.asarray(numeric['row_id']),original_global_index=np.asarray(numeric['original_global_index'],np.int64),native_slot=np.asarray(numeric['native_slot'],np.int8),query_xy=np.asarray([c.qarray(q) for q in numeric['query_xy']],np.float64),L=np.asarray([np.nan if q is None else q for q in numeric['L']],np.float64),original_query_available=np.asarray(numeric['original_query_available'],bool))
        c.validate_input10(x,n);c.need((targets[~x['original_query_available']]==-1).all(),'Original inputU unknown')
        for label in [0,1]:c.need((targets==label).any(),'Both direct classes, no fabricated empty-class denominator')
        for gi,g in enumerate(groups):
            found=[r for r in census if (r['domain'],r['attempt_id'],r['variant'])==(g['domain'],g['attempt_id'],g['variant'])];c.need(len(found)==1,'Census cache map unique');found[0].update(cache_row_indices=list(range(8*gi,8*gi+8)),direct_training_admitted=g['admitted'])
        target_packet=dict(direct_end_external_target=targets,authority=np.asarray(authorities),**{k:x[k] for k in c.META})
        ip=c.save_arrays(out/'DIRECT_CONTROL_INPUTS10.npz',x);tp=c.save_arrays(out/'DIRECT_END_TARGETS_WITH_META6.npz',target_packet)
        c.write_rows(out/'ALL_CONSTRUCTED_ROWS.jsonl',allrows);c.write_rows(out/'ALL3270_VARIANT_CENSUS.jsonl',census);c.write_rows(out/'ALL_SAM_GROUPS.jsonl',savedgroups)
        c.need(helper['state_sha'](model)==initial,'Frozen SAM full state unchanged');c.checked(checkpoint)
        for s in closures:c.checked(s['complete']);c.checked(s['census']);c.checked(s['independent_review'])
        for b in {g['source_packet']['path']:g['source_packet'] for g in groups}.values():c.checked(b)
        c.checked(e['admission'])
        for b in p['source_bindings'].values():c.checked(b)
        c.write(out/'COMPLETE.json',dict(completed=True,PASS=True,source_only=True,domain='DIRECT_CONTROL',rows=n,groups=len(groups),inputs=ip,targets=tp,metadata=c.bind(out/'ALL_CONSTRUCTED_ROWS.jsonl'),census=c.bind(out/'ALL3270_VARIANT_CENSUS.jsonl'),SAM_groups=c.bind(out/'ALL_SAM_GROUPS.jsonl'),admission=e['admission'],entry=c.bind(entry_path),code=c.bind(__file__),protocol=e['protocol'],source_controls=closures,original_inputU=int((~x['original_query_available']).sum()),direct_known_counts=[int((targets==i).sum()) for i in [0,1]],distinct_target_authorities=sorted(set(authorities)),SAM_checkpoint=checkpoint,SAM_source_commit=pin['exact_commit'],SAM_model_state_sha=initial,SAM_state_unchanged=True,SAM_forward_calls_including_repeat=calls,SAM_first_repeat=repeat,new_native_owner_quality_targets_created=False,original_targets_changed=False,new_fit=0,seconds=time.monotonic()-start,selected_model=None,fully_solved=False))
        print('DIRECT_CONTROL_CACHE_COMPLETE',flush=True)
    except BaseException:
        c.write(out/'FAILED_NO_AUTOMATIC_RETRY.json',dict(completed=False,PASS=False,traceback=traceback.format_exc(),automatic_retry=False));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--entry',type=c.Path,required=True);a=p.parse_args();build(a.entry)
