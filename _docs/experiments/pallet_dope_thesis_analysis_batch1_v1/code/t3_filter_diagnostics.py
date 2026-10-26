"""T3 only: infer -> verify both original filters -> separately score stage masks.

No training/annotation reads. `infer` is the only command with model forwards.
`verify` is reference-free. `stats` reads previously frozen SCORES reference.xy
only after a SHA-bound, successful main/no-flip equivalence receipt exists.
Run commands only on the root agent's authorization after T0/T1 gates.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
import sys
import time
import numpy as np
from common import OUT, FLIP, NOFLIP, KINDS, read, write, sha, start, log, forward_log, csvwrite, mdtable, load_module
from t3_adapter import (MODEL_SOURCE,ADAPTER_SOURCE,GEOMETRY_INPUTS,require,clean,
    execution_gates,geometry_modules,load_geometry_inputs,teacher_jobs,source_bindings,
    assert_bindings,native_predictions)

STAGES=OUT/'cache/T3_STAGE_MASKS.json'
EQUIVALENCE=OUT/'preflight/T3_FILTER_EQUIVALENCE.json'
RUNTIME=dict(device='cuda',batch=1,dtype='float32',autocast=False,TF32=False,
             cudnn_benchmark=False,cudnn_deterministic=True)

def cache_path(job):
    return OUT/'cache'/f"T3_PREDICTIONS_{job['model']}.json"

def cache_contract(job,bindings):
    paths=[FLIP/'MANIFEST.json',GEOMETRY_INPUTS,Path(job['protocol_path'])]
    if 'receipt_path' in job: paths.append(Path(job['receipt_path']))
    return dict(schema='batch1_t3_two_view_predictions_v1',model=job['model'],role=job['role'],
                checkpoint=job['checkpoint'],runtime=RUNTIME,sources=bindings,
                inputs={str(p):sha(p) for p in paths},
                frame_ids=[r['frame']['id'] for r in job['records']],
                population='Own-kind DEV only; shared R0 covers all292; 876 teacher-frame pairs total',
                GT_inputs=False,training=False)

def validate_cache(job,bindings):
    path=cache_path(job);require(path.is_file(),f'T3 original/flip predictions not prepared: {path}')
    payload=read(path);expected=cache_contract(job,bindings)
    require(payload.get('complete') is True and payload.get('contract')==expected,'T3 cache contract/source/checkpoint differs')
    rows=payload['records']
    require([r['id'] for r in rows]==expected['frame_ids'],'T3 cache population/order differs')
    require(payload['image_forwards']==2*len(rows),'T3 cache forward count differs')
    for row,item in zip(rows,job['records']):
        frame=item['frame']
        require(row['image_sha256']==frame['image_sha256'],'T3 cache image binding differs')
        for view in ('original','flip'):
            p=np.asarray(row[view]['points'],float);v=np.asarray(row[view]['valid'],bool);c=np.asarray(row[view]['confidence'],float)
            require(p.shape==(9,2) and v.shape==c.shape==(9,),'T3 cache requires9 semantic slots')
            require(np.isfinite(p[v]).all() and np.isfinite(c[v]).all(),'T3 valid predictions must be finite')
    return rows

def infer():
    gates=execution_gates();items,_=load_geometry_inputs();geometry_modules()
    jobs=teacher_jobs(items);bindings=source_bindings()
    start('T3_infer',[GEOMETRY_INPUTS,FLIP/'MANIFEST.json'])
    import cv2
    import torch
    require(torch.cuda.is_available(),'Canonical CUDA FP32 inference is required')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    model_module=load_module(MODEL_SOURCE,'batch1_t3_dope_models')
    adapter=load_module(ADAPTER_SOURCE,'batch1_t3_canonical_adapter')
    new_forwards=0
    for job in jobs:
        path=cache_path(job);cp=job['checkpoint']
        require(sha(cp['path'])==cp['sha256'],'Teacher checkpoint changed')
        if path.exists():
            rows=validate_cache(job,bindings)
            log('cache_reuse',task='T3',model=job['model'],frames=len(rows),path=str(path),sha256=sha(path))
            continue
        net=model_module.DopeNetwork(pretrained=False,stop_at_stage=6)
        state=torch.load(cp['path'],map_location='cpu',weights_only=True)
        net.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()},strict=True)
        net.float().requires_grad_(False).eval().to('cuda')
        rows=[];began=time.monotonic()
        for index,item in enumerate(job['records']):
            frame=item['frame'];image_bytes=Path(frame['image_path']).read_bytes()
            import hashlib
            require(hashlib.sha256(image_bytes).hexdigest()==frame['image_sha256'],'DEV image changed')
            image=cv2.imdecode(np.frombuffer(image_bytes,np.uint8),cv2.IMREAD_COLOR)
            require(image is not None and image.shape[:2]==(frame['height'],frame['width']),'DEV image shape differs')
            row=dict(id=frame['id'],kind=item['kind'],image_sha256=frame['image_sha256'])
            for view,im in (('original',image),('flip',cv2.flip(image,1))):
                prepared=adapter.prepare(im,source_pre_padded=False)
                log('neural_forward_attempt',task='T3',model=job['model'],frame_id=frame['id'],view=view,image_count=1)
                with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False):
                    beliefs,_=net(prepared['tensor'].unsqueeze(0).to(device='cuda',dtype=torch.float32))
                decoded=adapter.decode(beliefs[-1][0].detach().cpu().numpy(),prepared)
                forward_log('T3',job['model'],1,frame_id=frame['id'],view=view,checkpoint_sha256=cp['sha256'])
                new_forwards+=1
                row[view]=clean(dict(points=decoded['points_original'],valid=decoded['valid'],
                    confidence=decoded['confidence'],input_shape=prepared['input_shape']))
            rows.append(row)
            if (index+1)%25==0 or index+1==len(job['records']):
                print('T3_FORWARD',job['model'],index+1,len(job['records']),f'{time.monotonic()-began:.1f}s',flush=True)
        assert_bindings(bindings);assert_bindings(gates)
        require(sha(cp['path'])==cp['sha256'],'Teacher checkpoint changed during inference')
        write(path,dict(complete=True,contract=cache_contract(job,bindings),records=rows,image_forwards=2*len(rows)))
        validate_cache(job,bindings)
        del net,state,beliefs
        torch.cuda.empty_cache()
    total=sum(2*len(validate_cache(job,bindings)) for job in jobs)
    require(total==1752,'Expected exactly1752 image predictions/876 two-view pairs')
    write(OUT/'cache/T3_INFERENCE_COMPLETION.json',dict(complete=True,teacher_frame_pairs=876,
        cached_image_predictions=total,new_image_forwards_this_invocation=new_forwards,GT_inputs=False,
        sources=bindings,gates=gates,caches={str(cache_path(j)):sha(cache_path(j)) for j in jobs}))
    log('step_complete',task='T3_infer',image_predictions=total,new_image_forwards=new_forwards)

def mask9(mask):
    result=np.zeros(9,dtype=bool);result[:8]=mask
    return result

def decompose(result,config,use_flip):
    """Pure Boolean decomposition on the original result's SAME fixed I,D/errors.

    No new PnP/LOO fits, no updated medians, and no hypothesis reselection.
    Scalar frame median gates are broadcast into the cumulative point masks.
    """
    s0=np.asarray(result['both_view_mask'],bool).copy()
    I=np.asarray(result['ransac_inlier_mask'],bool)
    s1=I & s0
    L=np.asarray(result['loo_errors_normalized'],float)
    lm=result['loo_median_normalized']
    lmed=bool(np.isfinite(lm) and lm<config.loo_median_threshold_normalized)
    s2=s1 & np.isfinite(L) & (L<config.loo_threshold_normalized) & lmed
    stages={'s0':mask9(s0),'s1':mask9(s1),'s2':mask9(s2)}
    last=s2
    if use_flip:
        F=np.asarray(result['flip_errors_normalized'],float);fm=result['flip_median_normalized']
        fmed=bool(np.isfinite(fm) and fm<config.flip_median_threshold_normalized)
        last=s2 & np.isfinite(F) & (F<config.flip_threshold_normalized) & fmed
        stages['s3']=mask9(last)
    passed=bool(last.sum()>=config.min_support)
    stages['final']=mask9(last if passed else np.zeros(8,bool))
    return stages,passed

def original_candidates(prediction,item,mode,modules):
    G,config,functions=modules[mode]
    p,v,c=native_predictions(prediction['original'],item['frame'])
    flip=prediction['flip']
    # Match the original caller: bound the restored flip-back pixels, without
    # adding a new pre-unflip bound on the decoded flipped-view coordinates.
    back,bv,bc=G.unflip_keypoints(np.asarray(flip['points'],float),item['frame']['width'],
        np.asarray(flip['valid'],bool),np.asarray(flip['confidence'],float))
    bv &= (np.isfinite(back).all(1)&(back[:,0]>=0)&(back[:,0]<item['frame']['width'])
           &(back[:,1]>=0)&(back[:,1]<item['frame']['height']))
    candidates=[]
    for name,xyz in item['hypotheses']:
        result=G.filter_geometry(p,v,c,back,bv,item['K'],xyz,flip_confidence=bc,config=config)
        candidates.append((name,result))
    name,result=functions['choose'](candidates)
    return name,result,candidates,p

def equivalence_failure(model,frame_id,mode,hypothesis,original,stages,passed,checked):
    expected=np.asarray(original['trusted_mask'],bool);actual=stages['final']
    report=dict(complete=False,pass_all=False,precision_computed=False,reference_coordinates_read=False,
        failure=dict(model=model,frame_id=frame_id,filter=mode,hypothesis=hypothesis,
            original_passed=bool(original['passed']),decomposed_passed=passed,
            original_trusted_mask=expected.tolist(),decomposed_trusted_mask=actual.tolist(),
            differing_channels=np.flatnonzero(expected!=actual).tolist()),checked=checked)
    write(EQUIVALENCE,report)
    log('gate_failed',task='T3_equivalence',**report['failure'])
    raise RuntimeError(f'T3 original/wrapper mismatch: {model}/{frame_id}/{mode}/{hypothesis}; precision forbidden')

def verify():
    gates=execution_gates();items,_=load_geometry_inputs();modules=geometry_modules()
    jobs=teacher_jobs(items);bindings=source_bindings()
    start('T3_equivalence',[GEOMETRY_INPUTS,FLIP/'MANIFEST.json'])
    require((OUT/'cache/T3_INFERENCE_COMPLETION.json').is_file(),'T3 inference must be complete first')
    checked={mode:dict(frames=0,nine_channel_comparisons=0,frame_mismatches=0,mask_mismatches=0,channel_mismatches=0,
                      candidate_frames=0,candidate_mismatches=0) for mode in modules}
    rows=[]
    for job in jobs:
        predictions=validate_cache(job,bindings)
        for prediction,item in zip(predictions,job['records']):
            per_mode={}
            for mode in ('main','noflip'):
                name,result,candidates,p=original_candidates(prediction,item,mode,modules)
                config=modules[mode][1];use_flip=mode=='main'
                for candidate_name,candidate in candidates:
                    candidate_stages,candidate_pass=decompose(candidate,config,use_flip)
                    checked[mode]['candidate_frames']+=1
                    if not (candidate_pass==bool(candidate['passed']) and np.array_equal(candidate_stages['final'],candidate['trusted_mask'])):
                        checked[mode]['candidate_mismatches']+=1
                        equivalence_failure(job['model'],item['frame']['id'],mode,candidate_name,candidate,candidate_stages,candidate_pass,checked)
                stages,passed=decompose(result,config,use_flip)
                frame_equal=passed==bool(result['passed'])
                mask_equal=np.array_equal(stages['final'],np.asarray(result['trusted_mask'],bool))
                checked[mode]['frames']+=1;checked[mode]['nine_channel_comparisons']+=9
                checked[mode]['frame_mismatches']+=int(not frame_equal)
                checked[mode]['mask_mismatches']+=int(not mask_equal)
                checked[mode]['channel_mismatches']+=int(np.count_nonzero(stages['final']!=np.asarray(result['trusted_mask'],bool)))
                if not (frame_equal and mask_equal):
                    equivalence_failure(job['model'],item['frame']['id'],mode,name,result,stages,passed,checked)
                require(np.array_equal(np.asarray(result['points_raw']),p,equal_nan=True),'Original filter altered raw targets')
                require(not any(mask[8] for mask in stages.values()),'Centroid must remain excluded')
                ordered=[stages[k] for k in ('s0','s1','s2','s3','final') if k in stages]
                require(all(not np.any(b & ~a) for a,b in zip(ordered,ordered[1:])),'Stages are not cumulative')
                per_mode[mode]=dict(selected_registered_hypothesis=name,passed=passed,reason=result['reason'],
                    stages={k:v.tolist() for k,v in stages.items()},diagonal_px=result['diagonal_px'],
                    loo_median_normalized=result['loo_median_normalized'],flip_median_normalized=result['flip_median_normalized'],
                    original_trusted_mask=np.asarray(result['trusted_mask'],bool).tolist(),
                    ransac_inlier_mask=np.asarray(result['ransac_inlier_mask'],bool).tolist())
            original_s0=np.asarray(per_mode['noflip']['stages']['s0'],bool)
            both_s0=np.asarray(per_mode['main']['stages']['s0'],bool)
            rows.append(clean(dict(model=job['model'],teacher_role=job['role'],id=item['frame']['id'],kind=item['kind'],
                original=prediction['original'],filters=per_mode,confidence_only={
                    'original_only':(original_s0 if original_s0.sum()>=6 else np.zeros(9,bool)).tolist(),
                    'both_view':(both_s0 if both_s0.sum()>=6 else np.zeros(9,bool)).tolist()})))
    require(len(rows)==876 and all(v['frames']==876 and not v['frame_mismatches'] and not v['mask_mismatches'] for v in checked.values()),'Incomplete equivalence population')
    assert_bindings(bindings);assert_bindings(gates)
    contract=dict(main_s0='Both-view valid & finite & confidence>=0.3 & native bounds intersection, unchanged original usable.',
        original_only_comparator='Original valid & finite & confidence>=0.3 & native bounds; all s0 corners from frames with >=6.',
        both_view_comparator='All main s0 corners from frames with >=6; no PnP/LOO/error gate.',
        hypothesis_selection='Original choose on all original candidate results, THEN expose fixed selected-hypothesis stages. Conditional diagnostics; no stage-specific reselection.',
        fixed_statistics='Same original RANSAC I, projected D, per-fold LOO errors, and medians over original I. No refits or median recomputation.',
        noflip='Original-only s0 -> s1 -> s2 -> final; s3 not applicable.',
        frame_counts='Report frames_nonempty and frames_ge6 separately; final requires >=6.',
        references_used=False,GT_inputs=False)
    write(STAGES,dict(complete=True,schema='batch1_t3_original_filter_stage_masks_v1',contract=contract,records=rows))
    report=dict(complete=True,pass_all=True,method='Every teacher-frame and every registry candidate: Boolean decomposition compared to unchanged original filter result.',
        checked=checked,teacher_frame_pairs=876,unique_DEV_frames=292,channels_per_frame=9,
        sources=bindings,gates=gates,stage_masks=dict(path=str(STAGES),sha256=sha(STAGES)),
        prediction_caches={str(cache_path(j)):sha(cache_path(j)) for j in jobs},
        contract=contract,neural_forwards=0,reference_coordinates_read=False,precision_computed=False)
    write(EQUIVALENCE,report)
    log('step_complete',task='T3_equivalence',pass_all=True,receipt=str(EQUIVALENCE),sha256=sha(EQUIVALENCE),checked=checked)
    print('T3_EQUIVALENCE_PASS',checked,flush=True)

def stats():
    execution_gates();require(EQUIVALENCE.is_file(),'Precision is forbidden before real equivalence verification')
    proof=read(EQUIVALENCE)
    require(proof.get('complete') is True and proof.get('pass_all') is True,'Equivalence verification did not pass')
    assert_bindings(proof['sources']);assert_bindings(proof['gates']);assert_bindings(proof['prediction_caches'])
    require(sha(STAGES)==proof['stage_masks']['sha256'],'Verified stage masks changed')
    for mode in ('main','noflip'):
        require(proof['checked'][mode]['frames']==876 and proof['checked'][mode]['frame_mismatches']==proof['checked'][mode]['mask_mismatches']==0,'Incomplete main/no-flip equivalence')
    score_path=NOFLIP/'SCORES.json'
    start('T3_stats',[STAGES,EQUIVALENCE,score_path])
    # Only now are existing score/reference coordinates parsed. No annotation file is opened.
    scores=read(score_path)
    references={r['id']:r['all_annotated_reference'] for r in scores['models']['R0_before_selftraining']['frames']}
    require(len(references)==292,'Reference score population differs')
    stage_rows=read(STAGES)['records'];groups=defaultdict(list)
    for row in stage_rows:
        for mode,result in row['filters'].items():
            for stage,mask in result['stages'].items():
                groups[(row['model'],row['teacher_role'],row['kind'],mode,stage,'all_annotated_reference')].append((row,mask))
                if row['kind']=='GREEN':groups[(row['model'],row['teacher_role'],row['kind'],mode,stage,'verified_manual_only')].append((row,mask))
        for mode,mask in row['confidence_only'].items():
            groups[(row['model'],row['teacher_role'],row['kind'],'confidence_only',mode,'all_annotated_reference')].append((row,mask))
            if row['kind']=='GREEN':groups[(row['model'],row['teacher_role'],row['kind'],'confidence_only',mode,'verified_manual_only')].append((row,mask))
    output=[]
    for (model,role,kind,mode,stage,panel),entries in groups.items():
        counts=dict(total_mask_corners=0,frames_nonempty=0,frames_ge6=0,eligible_ref_count=0,
                    annotated_eligible_selected_ref_count=0,selected_ref_observed_count=0,
                    rejected_ref_count=0,rejected_ref_observed_count=0)
        correct={part:{tau:0 for tau in (5,10,20)} for part in ('selected','rejected')}
        for row,rawmask in entries:
            mask=np.asarray(rawmask,bool);counts['total_mask_corners']+=int(mask[:8].sum())
            counts['frames_nonempty']+=int(mask[:8].any());counts['frames_ge6']+=int(mask[:8].sum()>=6)
            p=np.asarray(row['original']['points'],float);valid=np.asarray(row['original']['valid'],bool)&np.isfinite(p).all(1)
            refs=references[row['id']]
            if panel=='verified_manual_only':refs=[r for r in refs if r['verified_manual'] is True]
            counts['eligible_ref_count']+=len(refs)
            for ref in refs:
                i=ref['index'];require(0<=i<8,'Reference contains non-corner index')
                part='selected' if mask[i] else 'rejected'
                counts['annotated_eligible_selected_ref_count' if part=='selected' else 'rejected_ref_count']+=1
                if valid[i]:
                    counts[part+'_ref_observed_count']+=1
                    xy=np.asarray(ref['xy'],float);require(xy.shape==(2,) and np.isfinite(xy).all(),'Invalid frozen reference coordinate')
                    error=float(np.linalg.norm(p[i]-xy))
                    for tau in (5,10,20):correct[part][tau]+=int(error<=tau)
        expected=600 if panel=='verified_manual_only' else {'PLASTIC':985,'WOOD':346,'GREEN':943}[kind]
        require(counts['eligible_ref_count']==expected,'T3 reference denominator differs from frozen primary/manual population')
        result=dict(model=model,teacher_role=role,kind=kind,filter=mode,stage=stage,panel=panel,n_frames=len(entries),**counts)
        for tau in (5,10,20):
            selected=counts['annotated_eligible_selected_ref_count'];rejected=counts['rejected_ref_count'];obs=counts['rejected_ref_observed_count']
            result[f'selected_correct_le{tau}px']=correct['selected'][tau]
            result[f'selected_ref_precision_le{tau}px']=correct['selected'][tau]/selected if selected else None
            result[f'rejected_correct_le{tau}px']=correct['rejected'][tau]
            result[f'rejected_ref_PCK_le{tau}px_full_denominator']=correct['rejected'][tau]/rejected if rejected else None
            result[f'rejected_ref_precision_le{tau}px_observed_only']=correct['rejected'][tau]/obs if obs else None
        output.append(result)
    csvwrite(OUT/'tables/T3_FILTER_STAGES.csv',output)
    note=('T3 filter diagnostics. Native-pixel same-index references; no threshold or hypothesis reselection. '
          'Main stages use both-view s0. Stage statistics are conditional on each original final selected hypothesis. '
          'Unannotated selected corners are counted in total_mask_corners only; their accuracy is unknown. '
          'Selected precision uses eligible selected reference points. Rejected PCK includes missing original predictions as failures; '
          'observed-only rejected precision excludes missing predictions. Both confidence-only definitions are reported.\n\n')
    headers=['kind','model','filter','stage','panel','frames_nonempty','frames_ge6','total_mask_corners',
             'annotated_eligible_selected_ref_count','selected_ref_precision_le10px']
    (OUT/'tables/T3_FILTER_STAGES.md').write_text(note+mdtable(headers,[[r[k] for k in headers] for r in output]))
    write(OUT/'tables/T3_FILTER_STAGES.json',dict(complete=True,rows=output,unit='Native original-image pixels; inclusive <=5/10/20.',
        equivalence_receipt=dict(path=str(EQUIVALENCE),sha256=sha(EQUIVALENCE)),
        reference_source=dict(path=str(score_path),sha256=sha(score_path),field='models.R0_before_selftraining.frames[].all_annotated_reference[].xy'),
        annotations_opened=False,thresholds_changed=False,contract=read(STAGES)['contract']))
    log('step_complete',task='T3_stats',rows=len(output),annotations_opened=False)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('infer','verify','stats'))
    args=parser.parse_args()
    {'infer':infer,'verify':verify,'stats':stats}[args.command]()

if __name__=='__main__':main()
