"""One source-only recalibration of the old Linear94 recipe.

Separate prepare/infer/labels/fit invocations keep renderer targets out of
feature generation. No real evaluation reference or outcome is permitted.
GPU phases are explicit; all old files remain read-only.
"""
from __future__ import annotations

import argparse
from collections import Counter
import os
from pathlib import Path
import sys
import time

import numpy as np

from . import common as C

ARMS = ('CLEAN_RAW_CLEAR', 'CLEAN_REF_CLEAR')
PARTS = ('TRAIN', 'VAL')
OLD_DOC = C.ROOT/'_docs/experiments/pallet_selector_recovery_v1'
OLD_SOURCE = OLD_DOC/'stage2_synth_scorer'
PRIVATE = C.RAW/'calibration'
PROTOCOL = C.DOC/'SELECTOR_CALIBRATION_PROTOCOL.json'
FEATURE_LOCK = C.DOC/'SYNTH_CURRENT_FEATURE_LOCK.json'
LABEL_LOCK = C.DOC/'CALIBRATION_LABEL_LOCK.json'
FIT = C.DOC/'SELECTOR_CALIBRATION_FIT.json'


def guard(allow_source_labels=False):
    reads = []
    forbidden = ('GEOMETRY_RESOLVED', 'TRUTH_FOR_DISPLAY', 'VERIFIED_LABELS',
        'AXIS_REVIEW', '/data/evaluation/', '/evaluation/S', 'ORACLE_METRICS',
        'POSE_METRICS', 'FRAME_METRICS', 'CONTROL_RESULTS', 'CONTROL_FRAME',
        'EVAL_RESULTS', 'SELECTOR_PAIR_RESULTS', 'ZERO_FIT_SELECTOR_RESULTS',
        'REAL_SCORER_RESULTS', 'REAL_ROUTER_RESULTS')
    if not allow_source_labels:
        forbidden += ('SYNTH_LABELS.npz', 'TRAINVAL_LABELS.npz', 'GEOMETRY_SIDETABLE.npz')
    def hook(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = os.path.abspath(os.fsdecode(args[0]))
        writing = isinstance(args[1], str) and any(c in args[1] for c in 'wax+')
        if not writing:
            assert not any(token in path for token in forbidden), 'CALIBRATION_READ_DENIED: '+path
            if path.startswith(str(C.ROOT)):
                reads.append(path)
    sys.addaudithook(hook)
    return reads


def select_rows(rows):
    selected = [row for row in rows if row['split'] in PARTS]
    assert Counter(row['split'] for row in selected) == {'TRAIN':4096, 'VAL':1024}
    assert len(selected) == len({row['id'] for row in selected}) == len({row['image']['sha256'] for row in selected})
    return selected


def verify_protocol():
    value = C.read(PROTOCOL)
    assert value['locked_before_new_feature_generation'] and value['new_selector_fits_max']==1
    assert value['feature_generator_arms']==list(ARMS)
    for binding in value['sources'] + list(value['checkpoints'].values()) + [value['inputs']]:
        C.verify(binding)
    return value


def prepare():
    if PROTOCOL.exists():
        verify_protocol(); print('CALIBRATION_PROTOCOL_ALREADY_LOCKED', flush=True); return
    reads = guard()
    from scripts.research.pallet_selector_recovery_v1 import features as F, models as M
    from scripts.research.pallet_selector_recovery_v1 import extract_features as X
    split_path = OLD_SOURCE/'SYNTHETIC_SPLIT_LOCK.json'
    split = C.read(split_path); C.verify(split['inputs'])
    rows = select_rows(C.read(C.ROOT/split['inputs']['path']))
    for row in rows:
        assert set(row) == {'id','split','image','hw','pad','K','dims'}
        C.verify(row['image'])
    primary = C.read(C.OLD.DOC/'PRIMARY_PROTOCOL.json')
    train_list = primary['datasets']['RAW']['train_list']; C.verify(train_list)
    source = [Path(p) for p in (C.ROOT/train_list['path']).read_text().splitlines() if Path(p).name.startswith('syn__')]
    current_source_sha = {C.sha(path) for path in source}
    assert len(source)==len(current_source_sha)==512
    assert current_source_sha == set(split['replay512_sha256'])
    assert not current_source_sha & {row['image']['sha256'] for row in rows}
    checkpoints = {}; sources = [C.bind(split_path), split['inputs'], C.bind(C.OLD.DOC/'PRIMARY_PROTOCOL.json'), train_list]
    for arm in ARMS:
        path = C.OLD.DOC/f'FIT_{arm}_S42.json'; fit = C.read(path)
        assert fit['complete'] and fit['optimizer_steps']==320 and fit['protected_state_exact']
        C.verify(fit['checkpoint']); checkpoints[arm] = fit['checkpoint']; sources.append(C.bind(path))
    path = PRIVATE/'TRAINVAL_INPUTS.json'; C.save(path, rows, True)
    sources += [C.bind(Path(__file__)), C.bind(Path(C.__file__)), C.bind(Path(F.__file__)),
        C.bind(Path(M.__file__)), C.bind(Path(X.__file__)), C.bind(OLD_DOC/'SELECTOR_FEATURE_CONTRACT.json'),
        C.bind(OLD_SOURCE/'EXACT_LABEL_AUDIT.json'), C.bind(OLD_SOURCE/'SCORER_VAL_RESULTS.json'), C.bind(C.DOC/'START.json')]
    protocol = dict(created_at=C.now(), locked_before_new_feature_generation=True,
        rationale='Replace old S0/S1 feature-generator distribution with the fixed matched CLEAR42 RAW/REF pair; do not tune against real DEV.',
        feature_generator_arms=list(ARMS), checkpoints=checkpoints, inputs=C.bind(path),
        counts={'TRAIN':4096,'VAL':1024}, candidate_features=94,
        source_replay_images=512, source_replay_sha_equal_old_exclusion=True, calibration_replay_exact_SHA_overlap=0,
        split='Unchanged old renderer-group-disjoint seed20260925 TRAIN/VAL; no TEST generation or scoring.',
        inference=dict(imgsz=640,conf=.001,rect=True,augment=False,half=False,batch=16,
            image_canvas='Existing stored synthetic RGB; no added padding or external coordinate transformation.',
            instance_selection='Highest detector confidence; no GT matching'),
        selector=dict(variant='GEO_LINEAR',feature_dimension=94,target='Exact renderer physical W/D parity, NOT T/R loss',
            normalization='Both candidates of valid TRAIN pairs only, std floor1e-6',
            optimizer='AdamW',lr=.001,weight_decay=.0001,batch=256,seed=42,max_epochs=30,
            early_stop_patience=5,checkpoint_selection='Earliest strict best synthetic VAL accuracy',
            valid_pairs='Retain each arm own finite pair; report actual count differences rather than assume unchanged updates',
            source_arm_weight='Each eligible pair weight1; both arms share original4096/1024 images'),
        forbidden=['Real-reference fitting','DEV hyperparameter selection','TEST scoring','MLP variant','New teacher fit','Additional selector retry'],
        supervision='Current Replay9images/38corners inherited through REF_CLEAR; old10/48 S0/S1 manual ancestry is not a dependency of this new scorer.',
        broader_research_supervision='Historical controls and decisions already used old GEO with additional10images/48corners: cumulative research19images/86corners is not erased.',
        deployment_dependency='R0 plus new GEO would require no self-trained student at deployment, but the new scorer still indirectly depends on RAW/REF student training during calibration.',
        scope_limit='Synthetic VAL is repeated development; R0 was trained on broader source. This is not an unseen-source test or independent real confirmation.',
        new_selector_fits_max=1,new_student_fits=0,new_training_real_RGB=0,new_manual_coordinates=0,
        runtime_read_guard=True,read_paths=sorted(set(reads)),sources=sources)
    C.save(PROTOCOL, protocol, True)
    print('CALIBRATION_PROTOCOL_LOCKED',protocol['counts'],flush=True)


def verify_features():
    value=C.read(FEATURE_LOCK); verify_protocol()
    assert value['target_values_read'] is False and value['base_weights_unchanged']
    for binding in value['files']+value['sources']: C.verify(binding)
    return value


def infer(batch=16):
    reads=guard(); protocol=verify_protocol()
    if FEATURE_LOCK.exists():
        verify_features(); print('CALIBRATION_FEATURES_ALREADY_LOCKED',flush=True); return
    import cv2
    import torch
    from ultralytics import YOLO
    from scripts.research.pallet_selector_recovery_v1 import common as U, features as F
    U.setup(); gpu=U.gpu(); torch.backends.cudnn.allow_tf32=True
    assert 1 <= batch <= protocol['inference']['batch']
    rows=C.read(C.ROOT/protocol['inputs']['path']); arrays={}; sources=[C.bind(PROTOCOL)]; files=[]; audits={}
    started=time.monotonic()
    for arm in ARMS:
        output=PRIVATE/(arm+'_FEATURES.npz'); audit_path=PRIVATE/(arm+'_FEATURES_LOCK.json')
        if audit_path.exists():
            audit=C.read(audit_path); assert audit['checkpoint']==protocol['checkpoints'][arm]
            C.verify(audit['file']); assert audit['base_weights_unchanged']
        else:
            assert not output.exists(), 'Unsealed feature file: preserve it and inspect before rerun'
            model=YOLO(str(C.ROOT/protocol['checkpoints'][arm]['path']),task='pose')
            model.predict(np.zeros((680,840,3),np.uint8),conf=.001,imgsz=640,rect=True,half=False,device='cuda',verbose=False)
            model.model.eval()
            for parameter in model.model.parameters(): parameter.requires_grad_(False)
            initial=U.state_hash(model.model); geo=[]; valid=[]; selected=[]; begun=time.monotonic()
            for offset in range(0,len(rows),batch):
                if offset%256==0:
                    U.gpu(); print('CALIBRATION_INFER',arm,offset,len(rows),flush=True)
                items=rows[offset:offset+batch]; images=[cv2.imread(str(C.ROOT/row['image']['path'])) for row in items]
                assert all(image is not None and list(image.shape[:2])==list(row['hw']) for image,row in zip(images,items))
                with torch.no_grad():
                    predictions=model.predict(images,conf=.001,imgsz=640,rect=True,augment=False,half=False,
                        device='cuda',verbose=False,save=False)
                for row,prediction in zip(items,predictions):
                    candidates=[]
                    if prediction.boxes is not None:
                        for index in range(len(prediction.boxes)):
                            candidates.append(dict(candidate_index=index,score=float(prediction.boxes.conf[index]),
                                box_xyxy=prediction.boxes.xyxy[index].cpu().tolist(),
                                keypoints_xy=prediction.keypoints.xy[index].cpu().tolist(),
                                keypoints_conf=prediction.keypoints.conf[index].cpu().tolist()))
                    pred=dict(candidates=candidates,selected_index=int(np.argmax([c['score'] for c in candidates])) if candidates else None)
                    features=F.extract(pred,row['K'],row['dims'],row['hw'])
                    geo.append(features['features'] if features['valid'] else np.zeros((2,94)))
                    valid.append(features['valid']); selected.append(U.HYP.index(features['selection']) if features['selection'] in U.HYP else -1)
            assert initial==U.state_hash(model.model)
            assert all(p.grad is None and not p.requires_grad for p in model.model.parameters())
            C.verify(protocol['checkpoints'][arm])
            output.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(output,ids=np.array([r['id'] for r in rows]),geo=np.asarray(geo,np.float32),
                valid=np.asarray(valid,bool),current=np.asarray(selected))
            audit=dict(file=C.bind(output),checkpoint=protocol['checkpoints'][arm],base_weights_unchanged=True,
                base_grad_zero=True,base_optimizer_steps=0,seconds=time.monotonic()-begun,batch=batch,frames=len(rows))
            C.save(audit_path,audit,True); del model; torch.cuda.empty_cache()
        with np.load(output) as data:
            assert data['ids'].tolist()==[row['id'] for row in rows]
            for key in ('geo','valid','current'): arrays[arm+'_'+key]=data[key]
        audits[arm]=audit; files.extend([C.bind(output),C.bind(audit_path)])
    combined=PRIVATE/'TRAINVAL_FEATURES.npz'
    assert not combined.exists()
    np.savez_compressed(combined,ids=np.array([r['id'] for r in rows]),split=np.array([r['split'] for r in rows]),**arrays)
    files.append(C.bind(combined))
    C.save(FEATURE_LOCK,dict(created_at=C.now(),features=C.bind(combined),files=files,sources=sources,
        arms=audits,target_values_read=False,real_reference_read=False,base_weights_unchanged=True,
        optimizer_steps=0,new_selector_fits=0,frames=len(rows),gpu_start=gpu,gpu_end=U.gpu(),
        seconds=time.monotonic()-started,read_paths=sorted(set(reads))),True)
    print('CALIBRATION_FEATURES_LOCKED',len(rows),flush=True)


def labels():
    reads=guard(allow_source_labels=True); features=verify_features()
    if LABEL_LOCK.exists():
        value=C.read(LABEL_LOCK); C.verify(value['labels']); print('CALIBRATION_LABELS_ALREADY_LOCKED',flush=True); return
    audit=C.read(OLD_SOURCE/'EXACT_LABEL_AUDIT.json'); C.verify(audit['labels'])
    rows=C.read(C.ROOT/verify_protocol()['inputs']['path'])
    with np.load(C.ROOT/audit['labels']['path']) as data:
        indices={str(fid):i for i,fid in enumerate(data['ids'])}; take=np.array([indices[r['id']] for r in rows])
        labels={key:data[key][take] for key in ('ids','split','parity')}
    assert labels['ids'].tolist()==[r['id'] for r in rows]
    assert labels['split'].tolist()==[r['split'] for r in rows]
    assert set(labels['split'])==set(PARTS) and set(labels['parity'])=={0,1}
    path=PRIVATE/'TRAINVAL_LABELS.npz'; assert not path.exists(); np.savez_compressed(path,**labels)
    C.save(LABEL_LOCK,dict(created_at=C.now(),labels=C.bind(path),features_lock=C.bind(FEATURE_LOCK),
        source_labels=audit['labels'],read_paths=sorted(set(reads)),TEST_used=False,
        physical_container_disclosure='Old compressed label container includes TEST arrays; only fixed TRAIN/VAL IDs are retained or used, no TEST score/selection.',
        labels_opened_after_feature_lock=True,exact_old_renderer_parity_preserved=True),True)
    print('CALIBRATION_LABELS_LOCKED',len(rows),flush=True)


def fit():
    reads=guard(allow_source_labels=True); protocol=verify_protocol(); features=verify_features()
    if FIT.exists():
        value=C.read(FIT); C.verify(value['checkpoint']); print('CALIBRATION_FIT_ALREADY_COMPLETE',flush=True); return
    import torch
    from scripts.research.pallet_selector_recovery_v1 import models as M,common as U
    U.setup(); gpu=U.gpu(); label_lock=C.read(LABEL_LOCK); C.verify(label_lock['labels']); C.verify(label_lock['features_lock'])
    with np.load(C.ROOT/features['features']['path']) as data: z=dict(data)
    with np.load(C.ROOT/label_lock['labels']['path']) as data: l=dict(data)
    assert np.array_equal(z['ids'],l['ids']) and np.array_equal(z['split'],l['split'])
    y=np.tile(l['parity'],2); parts=np.tile(l['split'],2)
    valid=np.concatenate([z[arm+'_valid'] for arm in ARMS])
    train=(parts=='TRAIN')&valid; val=(parts=='VAL')&valid
    assert train.sum()>=2048 and val.sum()>=512, 'INSUFFICIENT_VALID_SYNTHETIC_PAIRS'
    assert set(y[train])==set(y[val])=={0,1}
    preflight=dict(created_at=C.now(),features=C.bind(FEATURE_LOCK),labels=C.bind(LABEL_LOCK),
        valid_train=int(train.sum()),valid_val=int(val.sum()),invalid_excluded=int((~valid).sum()),
        old_valid_train=8190,old_valid_val=2048,valid_pair_change=int(train.sum())-8190,
        updates_per_epoch=int(np.ceil(train.sum()/256)),old_updates_per_epoch=32,
        arm_valid={arm:{part:int(np.sum(z[arm+'_valid']&(z['split']==part))) for part in PARTS} for arm in ARMS},
        no_real_reference_reads=True,TEST_used=False,passed=True)
    C.save(C.DOC/'CALIBRATION_PAIR_PREFLIGHT.json',preflight,True)
    attempt=C.DOC/'SELECTOR_CALIBRATION_ATTEMPT.json'
    assert not attempt.exists(), 'Only one selector fit: inspect preserved attempt instead of retrying'
    C.save(attempt,dict(started_at=C.now(),protocol=C.bind(PROTOCOL),preflight=C.bind(C.DOC/'CALIBRATION_PAIR_PREFLIGHT.json'),new_selector_fits=1),True)
    x=np.concatenate([z[arm+'_geo'] for arm in ARMS]); assert x.shape==(10240,2,94)
    x,mean,std=M.normalize(x,train); started=time.monotonic()
    state,result=M.fit(x,y,train,val,linear=True)
    checkpoint=PRIVATE/'CURRENT_GEO_LINEAR.pt'; assert not checkpoint.exists()
    torch.save(dict(state=state,mean=mean,std=std,d=94,variant='GEO_LINEAR'),checkpoint)
    C.save(FIT,dict(created_at=C.now(),checkpoint=C.bind(checkpoint),variant='GEO_LINEAR',
        result=result,preflight=preflight,selector_fits=1,student_fits=0,
        optimizer_steps=len(result['curve'])*preflight['updates_per_epoch'],seconds=time.monotonic()-started,
        no_TEST_evaluation=True,no_real_reference_read=True,old_historical_10_48_not_inherited=True,
        inherited_current_manual_budget=dict(images=9,corners=38),read_paths=sorted(set(reads)),
        sources=[C.bind(PROTOCOL),C.bind(FEATURE_LOCK),C.bind(LABEL_LOCK),C.bind(attempt),C.bind(Path(M.__file__))],
        gpu_start=gpu,gpu_end=U.gpu()),True)
    print('CALIBRATION_FIT_COMPLETE',result['best_epoch'],result['best_val_accuracy'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('phase',choices=('prepare','infer','labels','fit'))
    parser.add_argument('--batch',type=int,default=16); args=parser.parse_args()
    if args.phase=='infer': infer(args.batch)
    else: globals()[args.phase]()
