"""C3: bounded existing-manual38 versus RAW38 student capability control.

CPU preparation/tests are separate from explicitly authorized GPU stages.
All generated files stay inside this cycle; historical artifacts are read-only.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import copy
import csv
import inspect
import math
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
import torch

from . import common as C
from . import cycle_affine as A
from . import cycle_affine_eval as E
from . import pose_oracle as O
from scripts.research.pallet_type_selftrain_v1.recovery_pose_trainer import pose_parameter

IDENT = 'C3_MANUAL38_CAPABILITY'
RAW = C.RAW / 'cycles' / IDENT
DOC = C.DOC / 'cycles' / IDENT
ARMS = ('RAW9', 'MANUAL9')
SPLIT = C.ROOT / '_docs/experiments/pallet_large_error_refiner_v1/SPLIT.json'
SUPPORT = C.ROOT / '_docs/experiments/pallet_posefix_large_error_v1/TRAIN_SUPPORT.json'
REPLAY_DOC = C.ROOT / '_docs/experiments/pallet_posefix_replay_v1'
BASE_ARMS = E.BASE_ARMS


def protocol_path():
    revised = DOC/'PROTOCOL_V2.json'
    return revised if revised.exists() else DOC/'PROTOCOL.json'


def read_protocol():
    p=C.read(DOC/'PROTOCOL.json')
    if protocol_path().name=='PROTOCOL_V2.json':
        revision=C.read(protocol_path());C.verify(revision['base_protocol'])
        replacements={r['old']['path']:r for r in revision['source_replacements']}
        sources=[]
        for binding in p['sources']:
            if binding['path'] in replacements:
                replacement=replacements[binding['path']]
                assert binding==replacement['old']
                binding=replacement['new']
            sources.append(binding)
        p['sources']=sources
    return p


def save(path, value, freeze=False):
    assert Path(path).is_relative_to(RAW) or Path(path).is_relative_to(DOC)
    C.save(path, value, freeze)


def link(source, destination):
    assert destination.is_relative_to(RAW)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        assert destination.resolve() == source.resolve()
    else:
        destination.symlink_to(source.resolve())


def fixed_label(box, points, mask, hw, pad=100):
    """Same predicted box, explicit support; never infer support from confidence.

    A finite raw point outside native RGB but inside the established padded
    canvas stays supervised. No clipping, identity repair or PnP completion.
    """
    h, w = map(int, hw)
    box = np.asarray(box, float).copy()
    points, mask = np.asarray(points, float), np.asarray(mask, bool)
    assert points.shape == (9, 2) and mask.shape == (9,) and not mask[8]
    assert np.isfinite(points[mask]).all()
    assert ((points[mask] >= [-pad, -pad]) & (points[mask] < [w+pad, h+pad])).all()
    box[[0, 2]] = np.clip(box[[0, 2]], 0, w)
    box[[1, 3]] = np.clip(box[[1, 3]], 0, h)
    assert np.isfinite(box).all() and (box[2:]-box[:2] > 1).all()
    size = np.asarray([w+2*pad, h+2*pad])
    values = [0., *((box[:2]+box[2:])/2+pad)/size, *(box[2:]-box[:2])/size]
    for j in range(9):
        values.extend([*((points[j]+pad)/size), 2.] if mask[j] else [.5, .5, 1.])
    assert len(values) == 32 and np.isfinite(values).all()
    return ' '.join(f'{v:.12f}' for v in values)+'\n'


def manual_points(annotation, expected_mask):
    entries = annotation['objects'][0]['keypoint_annotations']
    h, w = annotation['camera_data']['height'], annotation['camera_data']['width']
    assert len(entries) == 9
    points = np.zeros((9, 2), float)
    mask = np.zeros(9, bool)
    for j, entry in enumerate(entries[:8]):
        if entry.get('source') != 'manual_click':
            continue
        point = np.asarray(entry.get('xy'), float)
        assert point.shape == (2,) and np.isfinite(point).all()
        assert entry.get('visibility') == 2 and entry.get('reason') == 'visible'
        assert entry.get('in_frame') is True and 0 <= point[0] < w and 0 <= point[1] < h
        points[j], mask[j] = point, True
    assert np.array_equal(mask, expected_mask)
    return points, mask


def verify_prepared():
    p = read_protocol()
    for binding in p['inputs']+p['sources']+[p['initialization'],p['targets'],p['RGB_metadata'],p['occurrences']]:
        C.verify(binding)
    assert p['arms'] == list(ARMS) and p['optimizer_updates_per_arm'] == 320
    assert p['real_affine_off'] and p['manual_corners'] == 38
    return p


def prepare():
    if (DOC/'PROTOCOL.json').exists():
        verify_prepared(); print('C3_PREPARED_REUSED', flush=True); return
    start = time.monotonic()
    assert (DOC/'SPEC.md').exists()
    old_path = C.P.REC/'pose_only/PROTOCOL.json'
    old = C.read(old_path)
    replay = C.read(REPLAY_DOC/'PROTOCOL.json')
    replay_input = C.read(REPLAY_DOC/'INPUT_LOCK.json')
    C.verify(replay['bindings'][str(SPLIT.relative_to(C.ROOT))])
    C.verify(replay_input['real_support'])
    assert replay_input['real_support'] == C.bind(SUPPORT)
    split = C.read(SPLIT); teacher = split['train']
    assert len(teacher) == 9 and sum(r['manual_corners'] for r in teacher) == 38
    assert [r['id'] for r in teacher] == replay_input['real_ids']
    population = C.P.records()+C.read(C.M.DOC/'EVAL_POPULATION_LOCK.json')['records']
    assert len(population) == 173
    train_ids, train_sha = {r['id'] for r in teacher}, {r['image']['sha256'] for r in teacher}
    assert not train_ids & {r['id'] for r in population}
    assert not train_sha & {r['image']['sha256'] for r in population}
    teacher_roles = C.read(C.P.DOC/'TEACHER_RECORDING_AUDIT.json')
    teacher_groups = set(teacher_roles['teacher_groups'])
    assert teacher_groups == {'REC_001', 'REC_002'}
    assert not teacher_groups & {r['recording'] for r in population}
    inputs, records, rgb_meta = [], [], []
    support_lock = {r['id']: r for r in C.read(SUPPORT)['records']}
    for r in teacher:
        for key in ('image', 'annotation', 'cache'):
            C.verify(r[key]); inputs.append(r[key])
        annotation = C.read(C.ROOT/r['annotation']['path'])
        points, mask = manual_points(annotation, r['manual_mask'])
        assert mask.tolist() == support_lock[r['id']]['crop_supported']
        image = cv2.imread(str(C.ROOT/r['image']['path'])); assert image is not None
        hw = list(image.shape[:2]); assert hw == [480, 640]
        cap = torch.load(C.ROOT/r['cache']['path'], map_location='cpu', weights_only=False)['captured']
        assert cap['selected_index'] is not None
        selected = cap['candidates'][cap['selected_index']]
        raw = np.asarray(selected['keypoints_xy'], float); box = np.asarray(selected['box_xyxy'], float)
        name = 'manual__'+r['id'].replace(':', '__')+'.png'
        destination = RAW/'dataset/shared/images'/name
        padded = cv2.copyMakeBorder(image, 100, 100, 100, 100, cv2.BORDER_REFLECT_101)
        if destination.exists():
            assert np.array_equal(cv2.imread(str(destination)), padded)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            assert cv2.imwrite(str(destination), padded)
        labels = {a: fixed_label(box, q, mask, hw) for a, q in [('RAW9', raw), ('MANUAL9', points)]}
        native = np.isfinite(raw).all(1) & (raw >= 0).all(1) & (raw < [640, 480]).all(1)
        obj = annotation['objects'][0]
        record = dict(id=r['id'], name=name, image=r['image'], padded_image=C.bind(destination),
                      annotation=r['annotation'], cache=r['cache'], hw=hw, mask=mask,
                      raw_target=raw, manual_target=points, box=box, labels=labels,
                      material='WOOD' if r['object_type'].startswith('wood') else 'PLASTIC',
                      recording='REC_001' if r['session']=='wood_day_01' else 'REC_002',
                      manual_corners=int(mask.sum()), raw_native_outside=np.flatnonzero(mask & ~native),
                      keypoint_frame=obj.get('keypoint_frame'), pose_status=obj.get('pose_status'),
                      migration_status=obj.get('migration_status'))
        records.append(record); inputs.append(C.bind(destination))
        rgb_meta.append({k: record[k] for k in ('id','image','hw','material','recording')})
    assert sum(len(r['raw_native_outside']) for r in records) == 1
    source_paths = [Path(p) for p in (C.ROOT/old['datasets']['RAW']['train_list']['path']).read_text().splitlines()
                    if Path(p).name.startswith('syn__')]
    assert len(source_paths) == len(set(p.name for p in source_paths)) == 512
    names = sorted(r['name'] for r in records)
    replacement = np.random.default_rng(9021).choice(names, 512, replace=True).tolist()
    assert set(replacement) == set(names)
    val_text = (C.P.CACHE/'dataset/val.txt').read_text()
    from scripts.research.pallet_material_selftrain_closure_v1.train_pair import parity_signature, assert_pair
    datasets, signatures = {}, {}
    for arm in ARMS:
        folder = RAW/'dataset'/arm; syn = []
        for image in source_paths:
            label = image.parent.parent/'labels'/f'{image.stem}.txt'
            link(image, folder/'images'/image.name); link(label, folder/'labels'/label.name)
            inputs.extend([C.bind(image), C.bind(label)])
            syn.append(str(folder/'images'/image.name))
        for record in records:
            link(C.ROOT/record['padded_image']['path'], folder/'images'/record['name'])
            label = folder/'labels'/f'{Path(record["name"]).stem}.txt'
            save(label, record['labels'][arm], True); inputs.append(C.bind(label))
        slots = syn+[str(folder/'images'/name) for name in replacement]
        save(folder/'train.txt', '\n'.join(slots)+'\n', True)
        save(folder/'val.txt', val_text, True)
        save(folder/'data.yaml', f'path: {folder}\ntrain: {folder/"train.txt"}\nval: {folder/"val.txt"}\nnc: 1\nnames: [pallet]\nkpt_shape: [9, 3]\nflip_idx: [1, 0, 3, 2, 5, 4, 7, 6, 8]\n', True)
        for name in ('train.txt','val.txt','data.yaml'): inputs.append(C.bind(folder/name))
        datasets[arm] = dict(data=C.bind(folder/'data.yaml'), train_list=C.bind(folder/'train.txt'))
        signatures[arm] = parity_signature(slots)
    assert_pair(signatures['RAW9'], signatures['MANUAL9'])
    for record in records: record.pop('labels')
    save(RAW/'TRAIN_TARGETS_PRIVATE.json', records, True)
    save(RAW/'TRAIN_RGB_METADATA.json', rgb_meta, True)
    save(RAW/'OCCURRENCES.json', dict(seed=9021, names=replacement, counts=dict(Counter(replacement))), True)
    args = dict(old['args'], lr0=1e-5)
    sources = [C.bind(p) for p in (old_path, SPLIT, SUPPORT, REPLAY_DOC/'PROTOCOL.json',
        REPLAY_DOC/'INPUT_LOCK.json', C.P.SPLIT, C.M.DOC/'EVAL_POPULATION_LOCK.json',
        C.P.DOC/'TEACHER_RECORDING_AUDIT.json', DOC/'SPEC.md', Path(__file__),
        Path(__file__).with_name('test_cycle_manual.py'), Path(A.__file__),
        C.ROOT/'scripts/research/pallet_type_selftrain_v1/recovery_pose_trainer.py',
        C.ROOT/'scripts/self_training_yolo/v3/true_ignore_pose_loss.py',
        C.ROOT/'scripts/self_training_yolo/v3/true_ignore_trainer.py')]
    save(DOC/'DATA_AUDIT.json', dict(train_images=9, manual_corners=38,
        materials={m:dict(images=sum(r['material']==m for r in records),
                          corners=sum(r['manual_corners'] for r in records if r['material']==m)) for m in ('PLASTIC','WOOD')},
        current_vs_teacher_frozen_annotations_exact=True, current_vs_teacher_images_exact=True,
        teacher_protocol_chain_verified=True, current_DEV_ID_overlap=[], current_DEV_SHA_overlap=[],
        current_DEV_recording_overlap=[], teacher_recordings=sorted(teacher_groups),
        verified66_is_subset_of_current128=True, raw_native_outside_points=1,
        padded_canvas_support_retained=38, source_order_and_files_same=True,
        no_new_labels=True, center_supervised=False, no_PnP_or_unknown_supervision=True,
        physical_signed_axes_not_confirmed=True, evaluated_only_after_prediction_lock=True,
        prior_same9_student_fit_found=False,
        prior_search_limit='Bounded scripts/docs/FIT inventory; PoseFix9/38 fit and hard8/36 full-student fit are different contracts',
        no_GPU_used=True, fits=0, updates=0), True)
    unique_inputs = list({b['path']: b for b in inputs}.values())
    save(DOC/'PROTOCOL.json', dict(cycle=IDENT, arms=list(ARMS), args=args, datasets=datasets,
        inputs=unique_inputs, sources=sources, initialization=old['initialization'],
        manual_images=9, manual_corners=38, new_labels=0, optimizer_updates_per_arm=320,
        fit_limit=2, total_update_limit=640, real_affine_off=True,
        augmentation='real translate=scale=0; same C2 wrapper, HSV and all synthetic augmentation unchanged',
        population='Mixed Plastic3/15 + Wood6/23 existing teacher TRAIN; unlike main217/361',
        manual_information='RAW9 also uses known manual support locations; this is a support-matched coordinate-source contrast',
        interpretation='Stored-index finite TRAIN capability / practical same-information-budget control, not pose GT or oracle upper bound',
        loss='Unchanged true-ignore; center and nonmanual visibility1, manual38 visibility2 before augmentation',
        checkpoint_selection='last after exactly320 optimizer updates; no DEV selection',
        parity=signatures, targets=C.bind(RAW/'TRAIN_TARGETS_PRIVATE.json'),
        RGB_metadata=C.bind(RAW/'TRAIN_RGB_METADATA.json'), occurrences=C.bind(RAW/'OCCURRENCES.json'),
        CPU_preparation_seconds=time.monotonic()-start, GPU_seconds=0), True)
    print('C3_PREPARED', len(records), sum(r['manual_corners'] for r in records), flush=True)


def make_dataset(arm):
    from ultralytics.cfg import get_cfg
    from ultralytics.data.dataset import YOLODataset
    p = read_protocol()
    return YOLODataset(img_path=str(C.ROOT/p['datasets'][arm]['train_list']['path']), imgsz=640,
        batch_size=16, augment=True, hyp=get_cfg(overrides=p['args']), rect=False, cache=False,
        stride=32, pad=0., task='pose', data=dict(names={0:'pallet'}, nc=1, kpt_shape=[9,3],
        flip_idx=[1,0,3,2,5,4,7,6,8]), prefix='C3_PREFLIGHT ')


def preflight():
    if (DOC/'PREFLIGHT.json').exists():
        assert C.read(DOC/'PREFLIGHT.json')['passed']; verify_prepared(); return
    p = verify_prepared(); start = time.monotonic()
    torch.set_num_threads(4); cv2.setNumThreads(1)
    datasets = {a:make_dataset(a) for a in ARMS}
    assert [Path(x).name for x in datasets['RAW9'].im_files] == [Path(x).name for x in datasets['MANUAL9'].im_files]
    original = {a:copy.deepcopy(d.transforms) for a,d in datasets.items()}
    for d in datasets.values(): d.transforms, count=A.wrap_affine(d.transforms); assert count>=1
    indices = {}
    for i,path in enumerate(datasets['RAW9'].im_files): indices.setdefault(Path(path).name, i)
    real = [(name,index) for name,index in indices.items() if not name.startswith('syn__')]
    assert len(real) == 9
    expected_support={r['name']:r['manual_corners'] for r in C.read(RAW/'TRAIN_TARGETS_PRIVATE.json')}
    mismatch, differences, tested = [], [], 0
    for name,index in real:
        for draw in range(64):
            sample_seed = 20260928+draw*10000+index
            old, new, rng = {}, {}, {}
            for arm in ARMS:
                dataset=datasets[arm]
                A.seed(sample_seed); old[arm]=original[arm](copy.deepcopy(dataset.get_image_and_label(index)))
                old_rng=A.rng_digest()
                A.seed(sample_seed); new[arm]=dataset[index]; rng[arm]=A.rng_digest()
                assert rng[arm]==old_rng
            b,c=map(A.sample_fingerprint,[new['RAW9'],new['MANUAL9']])
            assert all(b[k]==c[k] for k in ('image','boxes','support'))
            expected=expected_support[name]
            assert int((new['RAW9']['keypoints'][...,2]==2).sum())==expected
            tested+=1
            if not np.array_equal(old['RAW9']['keypoints'][...,2],old['MANUAL9']['keypoints'][...,2]):
                mismatch.append(dict(name=name,draw=draw,seed=sample_seed,
                    raw_mask=old['RAW9']['keypoints'][...,2].tolist(),manual_mask=old['MANUAL9']['keypoints'][...,2].tolist()))
            differences.append(b['xy'] != c['xy'])
    source=[i for name,i in indices.items() if name.startswith('syn__')]
    selected=[source[int((j+.5)*len(source)/64)] for j in range(64)]
    for i in selected:
        A.seed(20260928+i); old=original['RAW9'](copy.deepcopy(datasets['RAW9'].get_image_and_label(i))); old_rng=A.rng_digest()
        A.seed(20260928+i); new=datasets['RAW9'][i]; new_rng=A.rng_digest()
        A.seed(20260928+i); manual=datasets['MANUAL9'][i]
        assert A.sample_fingerprint(old)==A.sample_fingerprint(new)==A.sample_fingerprint(manual)
        assert old_rng==new_rng
    save(RAW/'ORIGINAL_AFFINE_MASK_DIAGNOSTIC.json',dict(samples=tested,mismatch_count=len(mismatch),
        mismatches=mismatch,decision='Real affine OFF fixed in SPEC before this diagnostic, not selected using DEV'),True)
    save(DOC/'PREFLIGHT.json',dict(passed=True,protocol=C.bind(protocol_path()),
        real_images=9,seeds_per_real_image=64,real_pairs=tested,all38_initial_and_transformed_support_retained=True,
        paired_RGB_boxes_support_exact=True,coordinate_difference_pairs=sum(differences),
        source_samples_bit_exact=64,source_RNG_unchanged=True,
        original_affine_mask_mismatch_count=len(mismatch),
        true_ignore='Existing unchanged implementation and inherited tests; new per-label support export tested here',
        no_actual_all_batch_parity_claim_yet=True,seconds=time.monotonic()-start,
        GPU_seconds=0,fits=0,optimizer_updates=0),True)
    print('C3_PREFLIGHT_PASS',tested,'original_affine_mask_mismatches',len(mismatch),flush=True)


def train(arm):
    assert arm in ARMS
    p=verify_prepared(); assert C.read(DOC/'PREFLIGHT.json')['passed']
    fit_path=RAW/f'FIT_{arm}.json'
    if fit_path.exists():
        fit=C.read(fit_path);assert fit['complete'] and fit['optimizer_steps']==320;C.verify(fit['checkpoint']);return
    run=RAW/'runs'/arm;assert not run.exists(),'Preserve incomplete fit, no automatic restart'
    from scripts.research.pallet_type_selftrain_v1 import train as T
    T.C.N.setup();torch.set_num_interop_threads(1)
    assert torch.cuda.is_available();A.gpu()
    trainer=A.AffineOffTrainer(overrides=dict(p['args'],model=str(C.ROOT/p['initialization']['path']),
        data=str(C.ROOT/p['datasets'][arm]['data']['path']),project=str(RAW/'runs'),name=arm,exist_ok=False))
    trainer.affine_trace=[]
    base=torch.load(C.ROOT/p['initialization']['path'],map_location='cpu',weights_only=False)['model'].float().state_dict()
    steps,epochs=[],[];begin=time.monotonic()
    def started(t):
        actual=t.model.state_dict()
        assert set(actual)==set(base) and all(torch.equal(base[k],actual[k].detach().cpu()) for k in base)
        assert all(pose_parameter(n) for n in t.recovery_trainable)
        t.optimizer.register_step_post_hook(lambda *_:steps.append(1))
    def epoch(t):
        assert len(steps)<=320
        epochs.append(dict(epoch=t.epoch+1,steps=len(steps),protected=t.check_frozen(),temperature_C=A.gpu()))
        save(RAW/f'RUN_STATE_{arm}.json',dict(status='RUNNING',optimizer_steps=len(steps),seconds=time.monotonic()-begin,epochs=epochs))
        print('C3_EPOCH',arm,t.epoch+1,len(steps),flush=True)
    trainer.add_callback('on_train_start',started);trainer.add_callback('on_train_epoch_end',epoch)
    save(RAW/f'RUN_STATE_{arm}.json',dict(status='STARTED',utc=C.now(),optimizer_steps=0))
    try:
        trainer.train()
        checkpoint=run/'weights/last.pt'
        final=torch.load(checkpoint,map_location='cpu',weights_only=False)['model'].float().state_dict()
        protected=[k for k in base if not pose_parameter(k) or k.endswith(('.running_mean','.running_var','.num_batches_tracked'))]
        assert all(torch.equal(base[k],final[k]) for k in protected)
        changed=[k for k in base if not torch.equal(base[k],final[k])]
        assert changed and all(pose_parameter(k) for k in changed)
        assert len(steps)==len(trainer.affine_trace)==320
        csv_path=run/'results.csv';assert len(list(csv.DictReader(csv_path.open())))==5
        save(RAW/f'TRACE_{arm}.json',trainer.affine_trace,True)
        fit=dict(complete=True,arm=arm,optimizer_steps=320,epochs=5,checkpoint=C.bind(checkpoint),
            initialization=p['initialization'],protocol=C.bind(protocol_path()),spec=C.bind(DOC/'SPEC.md'),
            implementation=C.bind(Path(__file__)),exact_R0_initialization=True,frozen_state_exact=True,
            protected_tensors=len(protected),changed_tensors=changed,trace=C.bind(RAW/f'TRACE_{arm}.json'),
            results_csv=C.bind(csv_path),history=epochs,seconds=time.monotonic()-begin,GPU_seconds=time.monotonic()-begin,
            existing_manual_coordinates_used=arm=='MANUAL9',existing_manual_support_used=True,
            new_manual_labels=0,manual_image_budget=9,manual_corner_budget=38,
            checkpoint_selection='last after exactly320updates',source_augmentation_unchanged=True)
        save(fit_path,fit,True);save(RAW/f'RUN_STATE_{arm}.json',dict(status='COMPLETE',optimizer_steps=320,fit=C.bind(fit_path)))
        print('C3_FIT_COMPLETE',arm,fit['seconds'],flush=True)
    except BaseException as error:
        save(RAW/f'RUN_STATE_{arm}.json',dict(status='FAILED_PRESERVED',error=repr(error),optimizer_steps=len(steps),
            seconds=time.monotonic()-begin,GPU_seconds=time.monotonic()-begin,epochs=epochs))
        save(RAW/f'FAILED_TRACE_{arm}.json',trainer.affine_trace,True)
        raise


def paired_trace():
    a,b=[C.read(RAW/f'TRACE_{arm}.json') for arm in ARMS]
    assert len(a)==len(b)==320
    keys=('batch','epoch','names','images','boxes','support','batch_idx','roles')
    for x,y in zip(a,b):assert all(x[k]==y[k] for k in keys),(x['batch'],'C3 paired input mismatch')
    totals={role:{k:sum(x['roles'][role][k] for x in a) for k in ('images','instances','supervised','ignore','invisible')}
            for role in ('SOURCE','REAL')}
    assert totals['SOURCE']['images']==totals['REAL']['images']==2560
    records=C.read(RAW/'TRAIN_TARGETS_PRIVATE.json');counts=C.read(RAW/'OCCURRENCES.json')['counts']
    expected=5*sum(counts[r['name']]*r['manual_corners'] for r in records)
    assert totals['REAL']['supervised']==expected and totals['REAL']['invisible']==0
    save(DOC/'TRAINING_PARITY.json',dict(batches=320,paired_images_boxes_masks_order_exact=True,
        source_and_real_exposures=totals,expected_manual_supervised_exposures=expected,
        coordinates_differing_batches=sum(x['coordinates']!=y['coordinates'] for x,y in zip(a,b))),True)


def train_all():
    for arm in ARMS:
        if (RAW/f'FIT_{arm}.json').exists():continue
        log=RAW/f'TRAIN_{arm}.log';assert not log.exists(),'Preserve earlier attempt'
        with log.open('w') as stream:
            result=subprocess.run([sys.executable,'-u','-m','scripts.research.pallet_oracle_mechanism_followup_v1.cycle_manual','train','--arm',arm,'--allow-gpu'],stdout=stream,stderr=subprocess.STDOUT)
        assert result.returncode==0,(arm,result.returncode,str(log))
    paired_trace()


def infer():
    lockpath=RAW/'PREDICTIONS_LOCK.json'
    if lockpath.exists():
        for b in C.read(lockpath)['files']:C.verify(b)
        return
    from ultralytics import YOLO
    from scripts.research.pallet_visible_transfer_closure_v1.infer_train import predict
    from scripts.research.pallet_material_selftrain_closure_v1.infer_eval import assert_detector_parity
    assert (DOC/'TRAINING_PARITY.json').exists()
    assert torch.cuda.is_available();torch.set_num_threads(4);cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.benchmark=False
    start=time.monotonic();files=[]
    checkpoints={'R0':read_protocol()['initialization']}
    for arm in ARMS:
        fit=C.read(RAW/f'FIT_{arm}.json');assert fit['complete'] and fit['optimizer_steps']==320
        checkpoints[arm]=fit['checkpoint'];C.verify(fit['checkpoint'])
    train_rows=C.read(RAW/'TRAIN_RGB_METADATA.json');train_predictions={}
    for arm,checkpoint in checkpoints.items():
        model=YOLO(str(C.ROOT/checkpoint['path']),task='pose');train_predictions[arm]={}
        for row in train_rows:
            A.gpu();C.verify(row['image']);image=cv2.imread(str(C.ROOT/row['image']['path']));assert image is not None
            train_predictions[arm][row['id']]=predict(model,image)
            if arm!='R0':assert_detector_parity(train_predictions['R0'][row['id']],train_predictions[arm][row['id']])
        del model;torch.cuda.empty_cache()
    path=RAW/'TRAIN_PREDICTIONS.json';save(path,train_predictions,True);files.append(path)
    for material in BASE_ARMS:
        rows,old_predictions,old_poses=E.metadata(material)
        predictions={alias:old_predictions[original] for alias,original in BASE_ARMS[material].items()}
        poses={alias:old_poses[original] for alias,original in BASE_ARMS[material].items()}
        for arm in ARMS:
            model=YOLO(str(C.ROOT/checkpoints[arm]['path']),task='pose');values={}
            for j,row in enumerate(rows):
                if j%32==0:A.gpu();print('C3_INFER',material,arm,j,len(rows),flush=True)
                C.verify(row['image']);image=cv2.imread(str(C.ROOT/row['image']['path']))
                assert image is not None and list(image.shape[:2])==row['hw']
                values[row['id']]=predict(model,image)
                assert_detector_parity(old_predictions['R0'][row['id']],values[row['id']])
            predictions[arm]=values
            poses[arm]={r['id']:O.D.Pose.infer(O.D.points(values[r['id']]),np.asarray(r['K']),np.asarray(r['xyz']),False) for r in rows}
            del model;torch.cuda.empty_cache()
        for name,value in [(f'{material}_PREDICTIONS.json',predictions),(f'{material}_POSES.json',poses),(f'{material}_METADATA.json',rows)]:
            path=RAW/name;save(path,value,True);files.append(path)
    files.extend([RAW/'TRAIN_RGB_METADATA.json'])
    save(lockpath,dict(utc=C.now(),files=[C.bind(p) for p in files],checkpoints=checkpoints,
        protocol=C.bind(protocol_path()),spec=C.bind(DOC/'SPEC.md'),code=C.bind(Path(__file__)),
        no_DEV_reference_coordinates_read=True,RGB_only_student=True,original_D9=True,
        detector_parity=True,seconds=time.monotonic()-start,GPU_seconds=time.monotonic()-start),True)
    print('C3_PREDICTIONS_FROZEN',flush=True)


def residual_summary(values):
    x=np.asarray(values,float)
    return dict(corners=len(x),mean_px=float(x.mean()),median_px=float(np.median(x)),P90_px=float(np.quantile(x,.9)),
                PCK={str(t):float((x<=t).mean()) for t in (5,10,20)},over20=int((x>20).sum()))


def score_train():
    assert (RAW/'PREDICTIONS_LOCK.json').exists()
    rows=C.read(RAW/'TRAIN_TARGETS_PRIVATE.json');predictions=C.read(RAW/'TRAIN_PREDICTIONS.json')
    points=[]
    for row in rows:
        for arm,values in predictions.items():
            pred=C.P.selected(values[row['id']])
            for j in np.flatnonzero(row['mask']):
                q=None if pred is None else np.asarray(pred['keypoints_xy'][j],float)
                valid=q is not None and np.isfinite(q).all() and not np.all(q==-1)
                errors={target:float(np.linalg.norm(q-np.asarray(row[key][j]))) if valid else math.hypot(*row['hw'])
                        for target,key in [('RAW9','raw_target'),('MANUAL9','manual_target')]}
                points.append(dict(id=row['id'],corner=int(j),material=row['material'],arm=arm,errors=errors,missing=not valid))
    groups={'ALL':points,**{m:[r for r in points if r['material']==m] for m in ('PLASTIC','WOOD')}}
    summary={g:{a:{t:residual_summary([r['errors'][t] for r in rr if r['arm']==a]) for t in ARMS}
                for a in predictions} for g,rr in groups.items()}
    save(RAW/'TRAIN_POINT_METRICS.json',points,True)
    return dict(groups=summary,coverage={a:sum(not r['missing'] for r in points if r['arm']==a) for a in predictions},
        interpretation='Finite stored-index TRAIN fit, not independent accuracy; same teacher-exposed9/38 and no physical pose claim')


def verified(predictions,rows,truth):
    from scripts.research.pallet_verified_anchor_v1.evaluate import point,metrics
    final=C.read(C.P.FINAL);assert final['reference_version']=='VERIFIED_VISIBLE_ANCHOR_FINAL_V2'
    lookup={r['id']:r for r in rows};values=[]
    for fi,ci in final['review_queue']:
        frame=final['frames'][fi];fid=frame['frame_id'];corner=frame['corners'][ci]
        if fid not in lookup or corner['status']!='DIRECT_VISIBLE':continue
        assert ci<8 and corner['coordinate_source']=='manual_click'
        assert frame['image_sha256']==lookup[fid]['image']['sha256']
        errors={};missing={}
        for arm,pp in predictions.items():
            q=point(pp[fid],ci);missing[arm]=q is None
            errors[arm]=float(np.linalg.norm(q-corner['xy'])) if q is not None else math.hypot(*truth[fid]['hw'])
        values.append(dict(id=fid,corner=ci,recording=lookup[fid]['recording'],severity=lookup[fid]['severity'],errors=errors,missing=missing))
    assert len(values)==66 and len({r['id'] for r in values})==16
    gg={'ALL':values}
    for key in ('recording','severity'):gg.update({f'{key}:{v}':[r for r in values if r[key]==v] for v in sorted({r[key] for r in values})})
    save(RAW/'VERIFIED66_POINT_METRICS.json',values,True)
    return dict(points=66,frames=16,fixed_identity=True,DEV_reference_never_trained=True,
        groups={g:{a:metrics([r['errors'][a] for r in rr]) for a in predictions} for g,rr in gg.items()},
        coverage={a:sum(not r['missing'][a] for r in values) for a in predictions})


def score():
    if (DOC/'RESULTS.json').exists():return
    start=time.monotonic();lockpath=RAW/'PREDICTIONS_LOCK.json';lock=C.read(lockpath)
    for b in lock['files']+list(lock['checkpoints'].values()):C.verify(b)
    save(RAW/'SCORING_START.json',dict(utc=C.now(),prediction_lock=C.bind(lockpath)),True)
    # Evaluation-only references are first opened after both arms/all populations are frozen.
    from scripts.research.pallet_material_selftrain_closure_v1.score_eval import score_frame
    truth=C.read(C.P.TRUTH);_,pose_truth=O.D.Pose.metadata('REAL_DEV')
    results={};visible=None
    scoring_sources=[C.bind(p) for p in (C.P.TRUTH,C.P.FINAL,Path(inspect.getsourcefile(score_frame)),
        Path(inspect.getsourcefile(O.D.metric)),Path(inspect.getsourcefile(O.D.Pose.metadata)),Path(E.__file__))]
    for material,bindings in BASE_ARMS.items():
        rows=C.read(RAW/f'{material}_METADATA.json');gg=E.groups(rows)
        predictions=C.read(RAW/f'{material}_PREDICTIONS.json');poses=C.read(RAW/f'{material}_POSES.json')
        oldroot=C.P.RAW if material=='PLASTIC' else C.M.RAW
        scoring_sources.extend(C.bind(oldroot/name) for name in ('FRAME_METRICS.json','POSE_METRICS.json','FIXED_ID_METRICS.json'))
        oldframe=C.read(oldroot/'FRAME_METRICS.json');oldpose=C.read(oldroot/'POSE_METRICS.json');oldfixed=C.read(oldroot/'FIXED_ID_METRICS.json')
        frames={a:oldframe[b] for a,b in bindings.items()};metrics={a:oldpose[b] for a,b in bindings.items()};fixed={a:oldfixed[b] for a,b in bindings.items()}
        for arm in ARMS:
            frames[arm]={r['id']:score_frame(r['id'],predictions[arm][r['id']],truth[r['id']]) for r in rows}
            fixed[arm]={r['id']:score_frame(r['id'],predictions[arm][r['id']],truth[r['id']],fixed=True) for r in rows}
            with ProcessPoolExecutor(max_workers=4) as pool:
                metrics[arm]=dict(pool.map(E.pose_job,[(r['id'],poses[arm][r['id']],pose_truth[r['id']]) for r in rows],chunksize=8))
        summaries={g:{a:dict(twoD=E.summary2([frames[a][i] for i in ids]),fixed_ID=E.summary2([fixed[a][i] for i in ids]),
                    sixD=E.summary6([metrics[a][i] for i in ids])) for a in frames} for g,ids in gg.items()}
        expected=985 if material=='PLASTIC' else 346
        assert all(v['twoD']['corners']==expected for v in summaries['ALL'].values())
        assert all(v['fixed_ID']['corners']==expected and v['sixD']['frames']==len(rows) for v in summaries['ALL'].values())
        assert len(rows)==(128 if material=='PLASTIC' else 45)
        assert all(set(predictions[a])==set(poses[a])==set(gg['ALL']) for a in predictions)
        comparisons=[('RAW9','MANUAL9'),('R0','MANUAL9'),('OLD_REF','MANUAL9'),('R0','RAW9')]
        contrasts={g:{f'{b}-minus-{a}':E.contrast(frames[a],frames[b],metrics[a],metrics[b],ids) for a,b in comparisons} for g,ids in gg.items()}
        loro={rec:E.contrast(frames['RAW9'],frames['MANUAL9'],metrics['RAW9'],metrics['MANUAL9'],
            [r['id'] for r in rows if r['recording']!=rec]) for rec in sorted({r['recording'] for r in rows})}
        results[material]=dict(groups=summaries,contrasts=contrasts,leave_one_recording_out_manual_minus_raw=loro,
            frames=len(rows),corners=expected,recordings=len({r['recording'] for r in rows}))
        for name,value in [(f'{material}_FRAME_METRICS.json',frames),(f'{material}_POSE_METRICS.json',metrics),(f'{material}_FIXED_ID_METRICS.json',fixed)]:save(RAW/name,value,True)
        if material=='PLASTIC':visible=verified(predictions,rows,truth)
    fits={a:C.read(RAW/f'FIT_{a}.json') for a in ARMS};train=score_train()
    save(DOC/'RESULTS.json',dict(cycle=IDENT,materials=results,verified66=visible,TRAIN_capability=train,
        primary='MANUAL9 versus RAW9, same38-support mixed9 TRAIN capability recipe',
        evaluation_role='Repeated DEV, optimizer seed42 only; no independent confirmation',
        reference='Legacy coordinates/geometry-derived pose, verified66 supplement; not physical6D ground truth',
        baseline_comparison='R0/OLD_REF are descriptive across different training populations; not same217/361 target intervention',
        manual_budget=dict(existing_images=9,existing_corners=38,new_images=0,new_corners=0,teacher_exposed=True),
        raw_and_pose_frozen_before_scoring=True,prediction_lock=C.bind(lockpath),
        fits={a:{k:f[k] for k in ('checkpoint','optimizer_steps','seconds','GPU_seconds','frozen_state_exact','exact_R0_initialization')} for a,f in fits.items()},
        resources=dict(fits=2,optimizer_updates=640,fit_GPU_seconds=sum(f['GPU_seconds'] for f in fits.values()),
            inference_GPU_seconds=lock['GPU_seconds'],scoring_CPU_wall_seconds=time.monotonic()-start),
        training_parity=C.bind(DOC/'TRAINING_PARITY.json'),spec=C.bind(DOC/'SPEC.md'),
        scoring_sources=scoring_sources,
        oracle_upper_bound=False,strict_Wood_visible='NA_REFERENCE_NOT_VERIFIED'),True)
    report()


def report():
    result=C.read(DOC/'RESULTS.json');lines=['# C3 기존 9장/38점 직접감독 capability 대조','',
        '[확인] 새로운 수동 라벨 없이 기존 교사 TRAIN 정보를 직접 학생에 사용했다. RAW9/MANUAL9는 같은 9장·38점 support·R0·source512·real512·증강·320update이고 감독 좌표만 다르다. 실사 affine OFF, HSV/합성 증강은 유지했다. MAIN217/361 전체 GT 학습 upper bound나 새 self-training recipe의 단독 효과가 아니다.','',
        '## TRAIN 저장-index 적합성','',C.table(['모델','RAW38 평균px','MANUAL38 평균px','MANUAL38 PCK10'],[
            [a,f"{v['RAW9']['mean_px']:.4f}",f"{v['MANUAL9']['mean_px']:.4f}",f"{100*v['MANUAL9']['PCK']['10']:.2f}%"]
            for a,v in result['TRAIN_capability']['groups']['ALL'].items()]),'',
        '교사에 이미 노출된 TRAIN 재평가다. 성공은 이 유한한 stored-index 타깃을 학습할 수 있다는 증거이며 물리적 축 정답 또는 독립 일반화를 뜻하지 않는다. 실패는 한 LR/320update/고정부 조건의 실패이지 표현 불가능의 증명이 아니다.','',
        '## 현재 반복 DEV — 전부 보고','',C.table(['재료','모델','PCK10','full-penalty median px','full-penalty P90 px','ADDsym AUC'],[
            [m,a,f"{100*v['twoD']['PCK']['10']:.3f}%",f"{v['twoD']['full_penalty_median_px']:.3f}",
             f"{v['twoD']['full_penalty_P90_px']:.3f}",f"{v['sixD']['ADDsym_AUC']:.6f}"]
            for m,r in result['materials'].items() for a,v in r['groups']['ALL'].items()]),'',
        '통제된 비교는 MANUAL9−RAW9다. R0/OLD_REF와의 비교는 다른 학습 모집단을 가진 참고선이다. recording별 손익·leave-one-recording-out·verified66는 RESULTS.json에 모두 남겼다. Wood45의 직접 visible 출처 검증은 없어 그 별도 점수는 NA다.', '',
        'camera_dynamic_0123_v4 / UNCONFIRMED_SIGNED_AXIS / MANUAL_REVIEW_REQUIRED 메타데이터를 그대로 보존했다. 중심·unknown·PnP projected 점은 감독하지 않았다. 기존 모델 자동 교체 없음.']
    save(DOC/'REPORT_KO.md','\n'.join(lines)+'\n',True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','preflight','train','train-all','paired-trace','infer','score','report'])
    parser.add_argument('--arm',choices=ARMS);parser.add_argument('--allow-gpu',action='store_true');args=parser.parse_args()
    if args.stage in ('train','train-all','infer'):assert args.allow_gpu,'Explicit parent GPU handoff required; CPU preparation must not start fits'
    if args.stage=='train':train(args.arm)
    else:{'prepare':prepare,'preflight':preflight,'train-all':train_all,'paired-trace':paired_trace,'infer':infer,'score':score,'report':report}[args.stage]()


if __name__=='__main__':main()
