"""Compare exact real/source gradient components in predicted-coordinate space.

Functional +/- parameter perturbations are restored by torch.func; no optimizer
is constructed, no checkpoint is saved, and no evaluation reference is read.
"""
import argparse
import gc
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
import torch

from . import common as C
from scripts.research.pallet_pose_objective_followup_v2.loss_signal import (
    load_model, seed, tensor_digest, make_criterion, attach_signal_hooks)

EPS = (1e-4, 3e-4, 1e-3)
NUMERICAL_FLOOR_PX = 1e-3
ARMS = ('R0', 'REF_LR5')


def prepare():
    destination = C.DOC / 'PROTOCOL.json'
    if destination.exists():
        for b in C.read(destination)['inputs']:
            C.verify(b)
        return C.read(destination)
    parent_path = C.V.P.REC / 'pose_only/PROTOCOL.json'
    parent = C.read(parent_path)
    train_path = C.ROOT / parent['datasets']['REF']['train_list']['path']
    all_paths = [Path(p) for p in train_path.read_text().splitlines()]
    roles = {r: sorted({p for p in all_paths if p.name.startswith('syn__') == (r == 'SOURCE')})
             for r in ('REAL', 'SOURCE')}
    assert len(roles['REAL']) == 217 and len(roles['SOURCE']) == 512
    chosen = {r: [v[int((j+.5)*len(v)/16)] for j in range(16)] for r, v in roles.items()}
    records = []
    for j in range(2):
        for role in ('REAL', 'SOURCE'):
            for path in chosen[role][j*8:j*8+8]:
                label = path.parent.parent / 'labels' / path.with_suffix('.txt').name
                records.append(dict(batch=j, role=role, image=C.bind(path), label=C.bind(label),
                                    sample_seed=290929+len(records)))
    selected = C.RAW / 'INPUT_SELECTION_PRIVATE.json'
    C.save(selected, records, True)
    checkpoints = {a: C.V.checkpoint('PLASTIC', a) for a in ARMS}
    protocol = dict(created_at=C.now(), head_start=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        material='PLASTIC', unique_real=16, unique_source=16, real_population=217,
        selection='Midpoint systematic sample from sorted unique existing TRAIN paths; no error or DEV ranking',
        batches=2, batch_size=16, real_per_batch=8, source_per_batch=8,
        targets='Existing corrected REF pseudo coordinates and original source labels; no evaluation reference',
        checkpoints=checkpoints, criterion_schedule={'R0': 0, 'REF_LR5': 4},
        schedule_note='Original initial/final E2E weights; frozen saved EMA for REF, not reconstructed online trajectory',
        directions=['REAL_COMPONENT', 'SOURCE_COMPONENT', 'COMBINED'],
        partition='Exact original mixed-batch loss with global denominators and same RLE clamp derivative gate; not separately renormalized real-only training',
        normalization='d_role = -g_role / max(norm(g_real),norm(g_source),norm(g_combined)); common scale preserves additivity',
        perturbation='theta +/- relative_epsilon * norm(theta_trainable) * d_role using functional_call',
        epsilons=EPS, displayed_response='Central derivative multiplied by middle epsilon0.0003, input640 pixel units',
        robust_sign='Same nonzero sign with magnitude above1e-3px at ALL three scaled derivative estimates; otherwise unresolved',
        numerical_floor_px=NUMERICAL_FLOOR_PX,
        decision='Report real-positive to combined-negative/nearzero counts per image/batch/checkpoint; local support only if replicated across batches, never auto fit',
        forbidden=['new fit', 'optimizer creation/step', 'checkpoint writes', 'evaluation labels', 'posthoc filtering or epsilon tuning', 'new RGB/manual coordinates'],
        inputs=[C.bind(parent_path), C.bind(train_path), C.bind(selected), *checkpoints.values()],
        max_new_fits=0, max_optimizer_updates=0, max_GPU_seconds=600,
        private_selection=C.bind(selected), augmentation='Original vanilla affine/HSV, fixed input seed; no new occlusion',
        scope_limits=['16 Plastic TRAIN images only', 'Pseudo target following is not physical correctness',
                      'Raw gradient directions, not AdamW moments/clipping/weight decay/EMA steps',
                      'Evaluation18 beneficial/8 harmful teacher cases are NOT training inputs or selection criteria'])
    C.save(destination, protocol, True)
    return protocol


def fixture_image_name(row):
    # bind() resolves original RGB symlinks; preserve the training label alias.
    return Path(row['label']['path']).with_suffix(Path(row['image']['path']).suffix).name


def batches(protocol):
    from ultralytics.cfg import get_cfg
    from ultralytics.data.dataset import YOLODataset
    parent = C.read(C.ROOT / protocol['inputs'][0]['path'])
    records = C.read(C.RAW / 'INPUT_SELECTION_PRIVATE.json')
    fixture = C.RAW / 'fixture_alias_corrected'
    for row in records:
        for field, folder in [('image', 'images'), ('label', 'labels')]:
            src = C.ROOT / row[field]['path']; C.verify(row[field])
            dst = fixture / folder / (fixture_image_name(row) if field == 'image' else src.name)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists(): dst.symlink_to(src.resolve())
            assert dst.resolve() == src.resolve()
    listing = fixture / 'train.txt'
    C.save(listing, '\n'.join(str(fixture/'images'/fixture_image_name(r)) for r in records)+'\n', True)
    ds = YOLODataset(img_path=str(listing), imgsz=640, batch_size=16, augment=True,
        hyp=get_cfg(overrides=parent['args']), rect=False, cache=False, stride=32, pad=0., task='pose',
        data=dict(names={0:'pallet'}, nc=1, kpt_shape=[9,3], flip_idx=[1,0,3,2,5,4,7,6,8]), prefix='DIRECTION_DIAGNOSTIC ')
    lookup = {Path(p).name: i for i,p in enumerate(ds.im_files)}
    result = []
    for batch_id in range(2):
        rows = [r for r in records if r['batch'] == batch_id]
        entries = []
        for row in rows:
            seed(row['sample_seed'])
            entries.append(ds[lookup[fixture_image_name(row)]])
        batch = ds.collate_fn(entries)
        assert len(batch['img']) == len(batch['keypoints']) == 16
        assert batch['img'].shape[-2:] == (640,640)
        result.append(batch)
    return parent, result


def extract(hook):
    row = hook.rows['one2one']
    masks, idx, _, _, stride, _, _ = row['args']
    prediction, target, support, _ = row['location_inputs']
    image, anchor = torch.where(masks)
    scale = stride[anchor].reshape(-1,1,1)
    return dict(prediction=prediction[...,:2]*scale, target=target[...,:2]*scale,
                support=support, image=image, anchor=anchor, target_index=idx[masks],
                assignment={name:dict(mask=v['args'][0].detach().clone(), index=v['args'][1].detach().clone()) for name,v in hook.rows.items()})


def flat_gradient(value, parameters):
    grad = torch.autograd.grad(value, parameters, allow_unused=True, retain_graph=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).detach().reshape(-1) for p,g in zip(parameters,grad)])


def vector_override(names, parameters, vector, scale):
    out = {}; offset = 0
    for name, parameter in zip(names,parameters):
        n = parameter.numel()
        out[name] = parameter.detach()+scale*vector[offset:offset+n].reshape_as(parameter)
        offset += n
    assert offset == vector.numel()
    return out


def stable_sign(values):
    if min(values) > NUMERICAL_FLOOR_PX: return 'TOWARD'
    if max(values) < -NUMERICAL_FLOOR_PX: return 'AWAY'
    if max(map(abs,values)) <= NUMERICAL_FLOOR_PX: return 'NEAR_ZERO'
    return 'EPS_SENSITIVE'


def run():
    from .loss_partition import partition
    from scripts.research.pallet_material_selftrain_closure_v1.infer_eval import thermal_guard
    protocol = prepare()
    destination = C.RAW / 'RESULTS_PRIVATE.json'
    if destination.exists():
        for b in C.read(destination)['bindings']: C.verify(b)
        print('DIAGNOSTIC_ALREADY_COMPLETE'); return
    assert torch.cuda.is_available(), 'Host GPU access required; do not silently train or alter environment'
    torch.set_num_threads(4); cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    parent, cpu_batches = batches(protocol)
    fingerprint = [dict(batch=i, image=tensor_digest(b['img']), box=tensor_digest(b['bboxes']),
                        support=tensor_digest(b['keypoints'][...,2]), target=tensor_digest(b['keypoints']),
                        names=b['im_file']) for i,b in enumerate(cpu_batches)]
    C.save(C.RAW/'INPUT_TENSOR_LOCK_PRIVATE.json', fingerprint, True)
    start = time.monotonic(); results = []; state_checks = {}
    import inspect
    import ultralytics.utils.loss as installed_loss
    import ultralytics.nn.modules.block as installed_block
    from scripts.self_training_yolo.v3 import true_ignore_pose_loss
    bindings = [C.bind(C.DOC/'PROTOCOL.json'), C.bind(C.RAW/'INPUT_TENSOR_LOCK_PRIVATE.json'),
                *[C.bind(Path(__file__).with_name(name)) for name in
                  ('probe.py', 'loss_partition.py', 'common.py')],
                C.bind(Path(inspect.getfile(load_model))),
                *[C.bind(Path(m.__file__)) for m in (true_ignore_pose_loss, installed_loss, installed_block)]]
    C.save(C.RAW/'EXECUTION_LOCK_PRIVATE.json', dict(bindings=bindings,
        precision='FP32; autocast off; matmul TF32 off; cuDNN TF32 off',
        coordinate_semantics='one2one TAL-assigned TRAIN anchor, augmented input640 pixels'), True)
    for arm in ARMS:
        thermal_guard()
        model = load_model(protocol['checkpoints'][arm], parent['args'], 'cuda:0')
        before = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        criterion = make_criterion(model)
        for _ in range(protocol['criterion_schedule'][arm]): criterion.update()
        names, parameters = zip(*[(n,p) for n,p in model.named_parameters() if p.requires_grad])
        theta_norm = float(torch.cat([p.detach().reshape(-1).double() for p in parameters]).norm())
        flow_mask = torch.cat([torch.full((p.numel(),), 'flow_model' in n, device='cuda',dtype=torch.bool) for n,p in zip(names,parameters)])
        for batch_id, cpu_batch in enumerate(cpu_batches):
            assert time.monotonic()-start < protocol['max_GPU_seconds']
            thermal_guard()
            batch = {k:v.to('cuda') if torch.is_tensor(v) else v for k,v in cpu_batch.items()}
            batch['img'] = batch['img'].float()/255
            real = torch.tensor([not Path(n).name.startswith('syn__') for n in batch['im_file']],device='cuda')
            with attach_signal_hooks(criterion) as hook:
                prediction = model(batch['img']); loss,_ = criterion(prediction,batch); hook.full_loss=loss
                base = extract(hook)
                pieces = partition(model,criterion,hook,real)
                gradients = {key:flat_gradient(pieces[key],parameters) for key in ('real','source','combined')}
                actual = flat_gradient(loss.sum()/16,parameters)
                additive = gradients['real']+gradients['source']
                assert torch.allclose(additive,actual,rtol=2e-4,atol=3e-6), 'Mixed gradient partition mismatch'
                assert torch.allclose(gradients['combined'],actual,rtol=2e-4,atol=3e-6)
                direction_norm = max(float(g.double().norm()) for g in gradients.values())
                assert direction_norm > 0
                directions = {key:-g/direction_norm for key,g in gradients.items()}
                record = dict(arm=arm,batch=batch_id,branch_weights=[criterion.o2m,criterion.o2o],
                    gradients={k:dict(norm=float(g.double().norm()),head_norm=float(g[~flow_mask].double().norm()),
                                      flow_norm=float(g[flow_mask].double().norm())) for k,g in gradients.items()},
                    gradient_cosine=float((gradients['real'].double()@gradients['source'].double())/(gradients['real'].double().norm()*gradients['source'].double().norm())),
                    partition=pieces['diagnostics'], max_gradient_partition_error=float((actual-additive).abs().max()),
                    theta_norm=theta_norm,common_gradient_norm=direction_norm, points=[])
                base = {k:(v.detach().clone() if torch.is_tensor(v) else v) for k,v in base.items()}
            del prediction,loss,pieces,hook,actual,additive
            torch.cuda.empty_cache()
            measured = {role:[] for role in directions}
            max_assignment_change = 0
            with torch.no_grad():
                for role, direction in directions.items():
                    for eps in EPS:
                        coordinates = []
                        for sign in (1,-1):
                            overrides = vector_override(names,parameters,direction,sign*eps*theta_norm)
                            with attach_signal_hooks(criterion) as h:
                                pred = torch.func.functional_call(model,overrides,(batch['img'],),strict=False)
                                criterion(pred,batch)
                                point = extract(h)
                                for branch in base['assignment']:
                                    for field in ('mask','index'):
                                        assert torch.equal(point['assignment'][branch][field],base['assignment'][branch][field]), 'Assignment changed'
                                assert torch.equal(point['support'],base['support'])
                                assert torch.equal(point['target'],base['target'])
                                coordinates.append(point['prediction'].detach().clone())
                            del pred,overrides,h,point
                        measured[role].append((coordinates[0]-coordinates[1])/(2*eps)*EPS[1])
                with attach_signal_hooks(criterion) as h:
                    criterion(model(batch['img']),batch)
                    restored = extract(h)
                    assert torch.equal(restored['prediction'],base['prediction']), 'Functional restore output mismatch'
                del restored,h
            desired = base['target']-base['prediction']; residual = desired.norm(dim=-1)
            unit = desired/residual.clamp_min(1e-12)[...,None]
            responses = {role:torch.stack([(delta*unit).sum(-1) for delta in values]).cpu().numpy() for role,values in measured.items()}
            additive_delta = measured['combined'][1]-measured['real'][1]-measured['source'][1]
            record['median_coordinate_additivity_residual_px'] = float(additive_delta.norm(dim=-1)[base['support']].median())
            record['max_coordinate_additivity_residual_px'] = float(additive_delta.norm(dim=-1)[base['support']].max())
            for anchor in range(len(base['image'])):
                image = int(base['image'][anchor])
                for corner in range(9):
                    if not bool(base['support'][anchor,corner]): continue
                    vals = {r:responses[r][:,anchor,corner].tolist() for r in responses}
                    record['points'].append(dict(image_id=Path(batch['im_file'][image]).stem,role='REAL' if real[image] else 'SOURCE',
                        corner=corner,anchor=int(base['anchor'][anchor]),residual_px=float(residual[anchor,corner]),
                        response_px_at_common_reference_step=vals,
                        signs={r:stable_sign(v) if residual[anchor,corner]>1e-8 else 'NO_RESIDUAL' for r,v in vals.items()}))
            C.save(C.RAW/f'{arm}_BATCH{batch_id}_PRIVATE.json',record,True)
            results.append(record)
            print('PROBED',arm,batch_id,'points',len(record['points']),'gradient_cos',record['gradient_cosine'],flush=True)
            assert all(torch.equal(before[k],v.detach().cpu()) for k,v in model.state_dict().items()), 'Checkpoint state changed'
            del gradients,directions,measured,batch,base
            gc.collect(); torch.cuda.empty_cache()
        state_checks[arm] = dict(tensors=len(before),exact=True,all_parameter_grad_buffers_none=all(p.grad is None for p in parameters))
        del model,criterion,before,parameters
        gc.collect();torch.cuda.empty_cache()
    result = dict(created_at=C.now(),bindings=bindings,records=results,state_checks=state_checks,
        GPU_seconds=time.monotonic()-start,optimizer_constructed=False,optimizer_steps=0,new_fits=0,
        evaluation_reference_read=False,checkpoint_writes=0,checkpoint_paths=protocol['checkpoints'],
        device=torch.cuda.get_device_name(0),private=True)
    C.save(destination,result,True)
    print('NO_FIT_DIAGNOSTIC_COMPLETE',result['GPU_seconds'],flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run'])
    {'prepare':prepare,'run':run}[parser.parse_args().action]()
