"""Frozen K8 TRAIN cross-input inference; no fit, optimizer, or evaluation GT.

The same CLEAR/OCC RGB and common support are used for every checkpoint.
Existing seed42 own-input predictions are reused; twelve missing model/input
contexts are inferred. Highest-confidence deployment detection, not TAL or a
target-nearest candidate. Results describe pseudo-target following only.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

import numpy as np

from . import common as C
from scripts.research.pallet_clean_to_pose_transfer_v1 import train_augmented_following as A

OUT=C.RAW/'train_cross_input'
PROTOCOL=C.DOC/'TRAIN_CROSS_INPUT_PROTOCOL.json'
LOCK=C.DOC/'TRAIN_CROSS_INPUT_PREDICTIONS_LOCK.json'
RESULT=C.DOC/'TRAIN_CROSS_INPUT_RESULTS.json'
SPLITS=('ALL_SUPERVISED','CANONICAL_REF_PLANNED_COVERED','CANONICAL_REF_UNMASKED')


def deny_eval_reference_reads():
    reads=[]
    forbidden=('TRUTH_FOR_DISPLAY','GEOMETRY_RESOLVED','VERIFIED_LABELS','FRAME_METRICS',
               'POSE_METRICS','EVAL_RESULTS','CURRENT_GEO_RESULTS','ORACLE_METRICS')
    def hook(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):
            return
        path=os.path.abspath(os.fsdecode(args[0]))
        writing=isinstance(args[1],str) and any(c in args[1] for c in 'wax+')
        if not writing:
            assert not any(s in path for s in forbidden), 'Evaluation reference denied: '+path
            if path.startswith(str(C.ROOT)):
                reads.append(path)
    sys.addaudithook(hook)
    return reads


def model_specs():
    result={}
    for seed in (42,43):
        for target in ('RAW','REF'):
            for condition in ('CLEAR','OCC'):
                arm=f'CLEAN_{target}_{condition}'
                folder=(C.OLD.DOC if seed==42 else C.DOC/'training/CLEAR_S43' if condition=='CLEAR'
                        else C.OLD.DOC/'followups/REPEAT_PRIMARY_S43')
                path=folder/f'FIT_{arm}_S{seed}.json'
                fit=C.read(path)
                assert fit['complete'] and fit['optimizer_steps']==320
                assert fit['protected_state_exact'] and fit['protected_tensors']==747
                C.verify(fit['checkpoint'])
                result[f'{target}_{condition}_S{seed}']=dict(seed=seed,target=target,training_condition=condition,
                    checkpoint=fit['checkpoint'],fit=C.bind(path),old_arm=arm)
    return result


def prepare():
    if PROTOCOL.exists():
        verify_protocol();print('TRAIN_CROSS_INPUT_ALREADY_PREPARED',flush=True);return
    reads=deny_eval_reference_reads()
    import torch
    from scripts.research.pallet_clean_to_pose_transfer_v1.augmentation import digest
    start=time.monotonic();models=model_specs();_,cache,bindings=A.prefix();rows=A.targets_for(cache)
    old_lock=C.read(A.LOCK)
    for b in old_lock['files']+old_lock['sources']:C.verify(b)
    assert rows==C.read(A.OUT/'TARGETS_PRIVATE.json')
    images={condition:[] for condition in ('CLEAR','OCC')}
    for row in rows:
        index,position=row['batch'],row['position']
        for condition in ('CLEAR','OCC'):
            raw=cache['RAW',condition][index];ref=cache['REF',condition][index]
            assert torch.equal(raw['images'][position],ref['images'][position])
            images[condition].append(ref['images'][position].clone())
        for target in ('RAW','REF'):
            clear=cache[target,'CLEAR'][index];occ=cache[target,'OCC'][index]
            assert torch.equal(clear['keypoints'],occ['keypoints'])
            assert torch.equal(clear['batch_idx'],occ['batch_idx'])
    images={k:torch.stack(v) for k,v in images.items()}
    changed=[not torch.equal(images['CLEAR'][i],images['OCC'][i]) for i in range(len(rows))]
    assert sum(changed)==sum(bool(r['plan'] and r['plan']['applied']) for r in rows)
    OUT.mkdir(parents=True,exist_ok=True)
    tensor_path=OUT/'INPUT_RGB_PRIVATE.pt'
    assert not tensor_path.exists(), 'Partial prepare exists; preserve and inspect before retry'
    torch.save(images,tensor_path)
    row_path=OUT/'TARGETS_PRIVATE.json';C.save(row_path,rows,True)
    reused={}
    for name,spec in models.items():
        if spec['seed']==42:
            p=A.OUT/(spec['old_arm']+'.json');d=C.read(p)
            assert d['stamp']['checkpoint']==spec['checkpoint']
            assert d['state_unchanged'] and not d['optimizer_constructed']
            reused[name+'/'+spec['training_condition']]=C.bind(p)
    sources=bindings+[C.bind(A.LOCK),C.bind(Path(A.__file__)),C.bind(Path(__file__)),
        C.bind(C.OLD.DOC/'PRIMARY_PROTOCOL.json'),C.bind(C.OLD.DOC/'PREFLIGHT.json'),
        C.bind(C.OLD.DOC/'PAIR_INTEGRITY_S42.json'),
        C.bind(Path(A.__file__).with_name('trainer.py')),C.bind(Path(A.__file__).with_name('augmentation.py'))]
    sources.extend(m['fit'] for m in models.values())
    sources.extend(reused.values())
    value=dict(created_at=C.now(),models=models,reused_contexts=reused,
        files=[C.bind(tensor_path),C.bind(row_path)],sources=sources,
        input_hashes={key:digest(value) for key,value in images.items()},
        protocol_locked_before_new_outputs=True,K_batches=8,input_seed=42,
        real_occurrences=len(rows),real_unique=len({r['name'] for r in rows}),
        supervised_corner_occurrences=sum(sum(r['support']) for r in rows),
        canonical_covered_occurrences=sum(len(r['canonical_REF_planned_covered']) for r in rows),
        RGB_changed_occurrences=sum(changed),common_RGB_targets_support_exact=True,
        evaluation_refs_read=False,read_paths=sorted(set(reads)),
        selection='Original pre-fit seed42 K8 prefix, not selected by residuals or DEV. Same62 occurrences for all models.',
        prediction_selection='Highest confidence only; no target-nearest or TAL assigned-anchor replacement',
        score_contract='Both raw/ref pseudo targets, common supervised corners0..7, canonicalREF covered group; native deployment decoder at augmented640',
        contrasts='OCC-trained minus CLEAR-trained on exactly same OCC and CLEAR inputs, for each target/seed; own target primary, both targets retained',
        missing_policy='Explicit missing count and640sqrt2 L2 penalty; conditional mean/median/P90 kept separately',
        limitations='Small dependent TRAIN prefix; seed43 models use same seed42 diagnostic inputs. Not physicalGT, not a capacity/attention assay.',
        new_fits=0,optimizer_updates=0,GPU_seconds=0,CPU_prepare_seconds=time.monotonic()-start)
    C.save(PROTOCOL,value,True)
    print('TRAIN_CROSS_INPUT_PREPARED',len(rows),value['canonical_covered_occurrences'],sum(changed),flush=True)


def verify_protocol():
    p=C.read(PROTOCOL)
    for b in p['files']+p['sources']:C.verify(b)
    for m in p['models'].values():C.verify(m['checkpoint'])
    assert p['protocol_locked_before_new_outputs'] and p['evaluation_refs_read'] is False
    return p


def infer():
    if LOCK.exists():
        for b in C.read(LOCK)['files']+C.read(LOCK)['sources']:C.verify(b)
        print('TRAIN_CROSS_INPUT_ALREADY_INFERRED',flush=True);return
    reads=deny_eval_reference_reads();p=verify_protocol()
    import torch
    from ultralytics import YOLO
    from scripts.research.pallet_selector_recovery_v1.common import state_hash
    from scripts.research.pallet_material_selftrain_closure_v1.infer_eval import thermal_guard,assert_detector_parity
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.benchmark=False
    images=torch.load(OUT/'INPUT_RGB_PRIVATE.pt',map_location='cpu',weights_only=True)
    rows=C.read(OUT/'TARGETS_PRIVATE.json')
    kwargs=dict(conf=.001,imgsz=640,rect=True,augment=False,half=False,device='cuda',verbose=False,save=False)
    allpred={};files=[];new_count=0;reused_count=0;start=time.monotonic();states={}
    for name,spec in p['models'].items():
        contexts={};needed=[]
        for condition in ('CLEAR','OCC'):
            key=name+'/'+condition
            if key in p['reused_contexts']:
                contexts[condition]=C.read(C.ROOT/p['reused_contexts'][key]['path'])['predictions'];reused_count+=1
            else:needed.append(condition)
        output_path=OUT/(name+'_PREDICTIONS_PRIVATE.json');seal=OUT/(name+'_LOCK.json')
        if output_path.exists():
            sealed=C.read(seal);C.verify(sealed['file'])
            stored=C.read(output_path);assert stored['checkpoint']==spec['checkpoint'] and stored['protocol']==C.bind(PROTOCOL)
            contexts=stored['contexts'];states[name]=stored['state_audit']
        else:
            thermal_guard();model=YOLO(str(C.ROOT/spec['checkpoint']['path']),task='pose')
            model.predict(np.zeros((640,640,3),np.uint8),**kwargs)
            model.model.eval()
            for parameter in model.model.parameters():parameter.requires_grad_(False)
            before=state_hash(model.model);count=len(model.model.state_dict())
            for condition in needed:
                prediction={}
                for index in range(8):
                    indices=[i for i,r in enumerate(rows) if r['batch']==index]
                    batch=[images[condition][i].permute(1,2,0).numpy()[:,:,::-1].copy() for i in indices]
                    with torch.inference_mode():results=model.predict(batch,**kwargs)
                    assert len(results)==len(indices)
                    for i,result in zip(indices,results):
                        assert result.orig_shape==(640,640)
                        cc=[]
                        if result.boxes is not None:
                            for j in range(len(result.boxes)):
                                cc.append(dict(candidate_index=j,score=float(result.boxes.conf[j]),
                                    box_xyxy=result.boxes.xyxy[j].cpu().tolist(),keypoints_xy=result.keypoints.xy[j].cpu().tolist(),
                                    keypoints_conf=result.keypoints.conf[j].cpu().tolist()))
                        prediction[rows[i]['id']]=dict(candidates=cc,selected_index=int(np.argmax([c['score'] for c in cc])) if cc else None)
                contexts[condition]=prediction;new_count+=1
            assert state_hash(model.model)==before
            assert all(parameter.grad is None and not parameter.requires_grad for parameter in model.model.parameters())
            states[name]=dict(state_hash_before_after=before,state_tensors=count,state_unchanged=True,
                gradients_absent=True,scope='Inference model after normal YOLO setup/fusion; checkpoint bytes separately verified unchanged')
            stored=dict(checkpoint=spec['checkpoint'],protocol=C.bind(PROTOCOL),contexts=contexts,state_audit=states[name])
            C.save(output_path,stored,True);C.save(seal,dict(file=C.bind(output_path)),True)
            del model;torch.cuda.empty_cache()
        allpred[name]=contexts;files.extend([C.bind(output_path),C.bind(seal)])
        print('TRAIN_CROSS_INPUT_MODEL',name,flush=True)
    for condition in ('CLEAR','OCC'):
        reference=next(iter(allpred.values()))[condition]
        for contexts in allpred.values():
            for row in rows:assert_detector_parity(reference[row['id']],contexts[condition][row['id']])
    seconds=time.monotonic()-start
    C.save(LOCK,dict(created_at=C.now(),files=files,sources=[C.bind(PROTOCOL)],read_paths=sorted(set(reads)),
        states=states,model_contexts=16,new_contexts_this_attempt=new_count,reused_seed42_contexts=reused_count,
        inputs_same_across_models=True,detector_parity_all_models=True,evaluation_refs_read=False,
        new_fits=0,optimizer_updates=0,optimizer_constructed=False,seconds=seconds,
        inference=kwargs,coordinate_space='augmented640px',model_weights_not_written=True),True)
    print('TRAIN_CROSS_INPUT_INFERRED_SECONDS',seconds,flush=True)


def point_residuals(predictions, rows, target):
    result=[]
    for row in rows:
        prediction=predictions[row['id']];i=prediction['selected_index']
        q=np.asarray(prediction['candidates'][i]['keypoints_xy']) if i is not None else np.full((9,2),np.nan)
        for corner,keep in enumerate(row['support']):
            if not keep:continue
            delta=q[corner]-np.asarray(row[target][corner]);missing=not np.isfinite(delta).all()
            result.append(dict(id=row['id'],name=row['name'],corner=corner,recording=row['recording'],missing=missing,
                split='CANONICAL_REF_PLANNED_COVERED' if corner in row['canonical_REF_planned_covered'] else 'CANONICAL_REF_UNMASKED',
                l1_mean_xy_px=None if missing else float(np.abs(delta).mean()),l2_px=None if missing else float(np.linalg.norm(delta))))
    return result


def paired(points_before,points_after):
    assert [(p['id'],p['corner']) for p in points_before]==[(p['id'],p['corner']) for p in points_after]
    observed=[(b,a) for b,a in zip(points_before,points_after) if not b['missing'] and not a['missing']]
    deltas=np.asarray([a['l2_px']-b['l2_px'] for b,a in observed])
    def val(p):return 640*np.sqrt(2) if p['missing'] else p['l2_px']
    full=[val(a)-val(b) for b,a in zip(points_before,points_after)]
    return dict(point_occurrences=len(points_before),common_observed=len(observed),
        improved=int((deltas<0).sum()),worsened=int((deltas>0).sum()),equal=int((deltas==0).sum()),
        paired_L2_delta_mean_px=float(deltas.mean()) if len(deltas) else None,
        paired_L2_delta_median_px=float(np.median(deltas)) if len(deltas) else None,
        full_missing_penalty_delta_mean_px=float(np.mean(full)) if full else None)


def score():
    deny_eval_reference_reads();protocol=verify_protocol();lock=C.read(LOCK)
    for b in lock['files']+lock['sources']:C.verify(b)
    if RESULT.exists():
        for b in C.read(RESULT)['sources']+C.read(RESULT)['private_artifacts']:C.verify(b)
        print('TRAIN_CROSS_INPUT_ALREADY_SCORED');return
    rows=C.read(OUT/'TARGETS_PRIVATE.json');points={};summaries={}
    for name in protocol['models']:
        stored=C.read(OUT/(name+'_PREDICTIONS_PRIVATE.json'));points[name]={};summaries[name]={}
        for condition,predictions in stored['contexts'].items():
            points[name][condition]={};summaries[name][condition]={}
            for target in ('raw','ref'):
                pp=point_residuals(predictions,rows,target);points[name][condition][target]=pp
                summaries[name][condition][target]={s:A.summary(pp if s=='ALL_SUPERVISED' else [r for r in pp if r['split']==s]) for s in SPLITS}
    contrasts={};own_table=[]
    for seed in (42,43):
        for target in ('RAW','REF'):
            before=f'{target}_CLEAR_S{seed}';after=f'{target}_OCC_S{seed}'
            for input_condition in ('CLEAR','OCC'):
                key=f'{after}-minus-{before}__INPUT_{input_condition}'
                contrasts[key]={}
                for reference in ('raw','ref'):
                    bb=points[before][input_condition][reference];aa=points[after][input_condition][reference]
                    contrasts[key][reference]={s:paired(bb if s=='ALL_SUPERVISED' else [p for p in bb if p['split']==s],
                                                       aa if s=='ALL_SUPERVISED' else [p for p in aa if p['split']==s]) for s in SPLITS}
                own_table.append(dict(seed=seed,target=target,input_condition=input_condition,
                    CLEARtrained=summaries[before][input_condition][target.lower()],
                    OCCtrained=summaries[after][input_condition][target.lower()],
                    paired=contrasts[key][target.lower()]))
    private=OUT/'POINT_RESIDUALS_PRIVATE.json';C.save(private,points,True)
    value=dict(created_at=C.now(),status='COMPLETE',groups=summaries,paired=contrasts,own_target_comparison=own_table,
        protocol=C.bind(PROTOCOL),real_occurrences=len(rows),real_unique=protocol['real_unique'],
        common_supervised_points=protocol['supervised_corner_occurrences'],canonical_covered=protocol['canonical_covered_occurrences'],
        targets='RAW/REF pseudo coordinates only; own-target comparisons primary, both targets retained',
        input_contract='Same saved seed42 K8 RGB, support and coordinate targets for all8 models; CLEAR planned-covered is counterfactual',
        limitations='TRAIN prefix, dependent points, highest-confidence deployment instance not TAL, no physicalGT or inference of attention mechanism. Both seeds diagnosed on same input stream.',
        new_fits=0,optimizer_updates=0,GPU_inference_seconds=lock['seconds'],
        sources=[C.bind(PROTOCOL),C.bind(LOCK),C.bind(Path(__file__))],private_artifacts=[C.bind(private)])
    C.save(RESULT,value,True)
    text=['# 동일 TRAIN 입력에서 CLEAR/OCC 학습 모델 교차 진단','',
        '동일 seed42 사전 K8의62 occurrence/45 unique,476지원코너(계획 covered21)를 모든 모델에 입력했다. 새 fit0. 값은 augmented640px의 **각 모델 RAW/REF 의사 타깃** 추종이며 물리 GT·자연 가림6D·attention의 증명이 아니다. seed43 모델도 동일 seed42 진단 입력을 사용했다.','',
        '| seed/target | 같은 입력 | covered L2 mean CLEAR-trained→OCC-trained | unmasked L2 mean CLEAR-trained→OCC-trained | covered 개선/악화/동률 |','|---|---|---:|---:|---:|']
    for r in own_table:
        a,b=r['CLEARtrained'],r['OCCtrained'];s='CANONICAL_REF_PLANNED_COVERED';u='CANONICAL_REF_UNMASKED';q=r['paired'][s]
        text.append('| %s/%s | %s | %.5f→%.5f | %.5f→%.5f | %s/%s/%s |' % (r['seed'],r['target'],r['input_condition'],
            a[s]['l2_px']['mean'],b[s]['l2_px']['mean'],a[u]['l2_px']['mean'],b[u]['l2_px']['mean'],q['improved'],q['worsened'],q['equal']))
    text+=['','CLEAR 입력의 covered는 실제 가림이 아닌 동일 계획 위치다. missing과640대각선 패널티, paired 평균/중앙값, 양쪽 RAW/REF 타깃 전체 결과는 JSON에 있다. 가림학습 모델의 이득/손해는 동일 입력에 대해 비교하지만, 이 소표본만으로 노출 부족과 학습능력·타깃 오차·instance 선택을 완전히 분해할 수 없다.','',
           '[결과 JSON](TRAIN_CROSS_INPUT_RESULTS.json) · [사전 입력 잠금](TRAIN_CROSS_INPUT_PROTOCOL.json) · [예측 잠금](TRAIN_CROSS_INPUT_PREDICTIONS_LOCK.json)']
    C.save(C.DOC/'TRAIN_CROSS_INPUT_KO.md','\n'.join(text)+'\n',True)
    print('TRAIN_CROSS_INPUT_SCORED',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('prepare','infer','score'))
    args=parser.parse_args();globals()[args.phase]()
