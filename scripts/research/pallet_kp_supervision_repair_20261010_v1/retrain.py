"""Reuse immutable source features/order with separately repaired target arrays.

``freeze`` and ``check`` perform no optimizer updates. ``train`` requires a
separate explicit authorization receipt bound to the frozen protocol/checks.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[3]
DOC = REPO / '_docs/experiments/pallet_kp_supervision_repair_20261010_v1'
OLD_DOC = REPO / '_docs/experiments/pallet_observation_refiner_20261009_v1'
PRIVATE_CACHE = Path('/dev/shm/pallet-observation-private-20261009/learned_cache')
READY = Path('/dev/shm/pallet-kp-supervision-gate-private-20261010/depth_recovery_v3')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def binding(path):
    path = Path(path).resolve()
    if path.is_relative_to(REPO):
        name, origin = str(path.relative_to(REPO)), 'public_repository'
    else:
        name, origin = path.name, 'external_readonly_dependency'
    return dict(path=name, origin=origin, sha256=sha(path), bytes=path.stat().st_size)


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def modules(source_root):
    assert source_root, 'Set PALLET_SOURCE_ROOT or --source-root to the original read-only source checkout'
    os.environ.setdefault('PALLET_SOURCE_ROOT', str(Path(source_root).resolve()))
    import numpy as np
    import torch
    from scripts.research.pallet_observation_refiner_20261009_v1 import model as M
    return np, torch, M


def state_sha(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def reject_protected_output(output, source_root, allow_repair_doc=False):
    output=Path(output).resolve()
    if allow_repair_doc and output==DOC.resolve():return
    assert source_root, 'Original source checkout is required for output protection'
    roots=[REPO.resolve(),Path(source_root).resolve(),PRIVATE_CACHE.parent.resolve(),READY.parent.resolve()]
    assert all(not output.is_relative_to(root) and not root.is_relative_to(output) for root in roots), 'Output must be isolated from repository/source/prior private experiment directories'


def input_paths(args):
    old_code = REPO / 'scripts/research/pallet_observation_refiner_20261009_v1'
    return dict(features=Path(args.features), order=Path(args.order),
                repaired_targets=Path(args.targets), cache_manifest=Path(args.manifest),
                source_ready_validation=Path(args.ready_validation),
                original_protocol=OLD_DOC / 'LEARNING_PROTOCOL.json',
                original_model=old_code / 'model.py', original_training=old_code / 'training.py',
                original_decoder=old_code / 'learned_infer.py',
                original_evaluation=old_code / 'learned_evaluate.py',
                original_solver=old_code / 'solver.py', original_point_line=old_code / 'point_line.py',
                original_common=old_code / 'common.py', original_pose_inference=old_code / 'inference.py',
                original_pose_scoring=old_code / 'evaluate.py',
                frozen_real_inputs=OLD_DOC / 'INPUTS.json',
                frozen_real_initial_observations=OLD_DOC / 'OBSERVATIONS.jsonl.gz',
                frozen_real_controls=OLD_DOC / 'FIXED_CONTROLS.jsonl.gz',
                driver=Path(__file__).resolve(), downstream_adapter=Path(__file__).with_name('downstream.py'))


def snapshots(args):
    return {key: binding(path) for key, path in input_paths(args).items()}


def load_arrays(args, np):
    features = np.load(args.features, mmap_mode='r')
    order = np.load(args.order, mmap_mode='r')
    with np.load(args.targets, allow_pickle=False) as z:
        targets = {key: z[key].copy() for key in z.files}
    assert features.shape == (1024, 84, 28, 65) and features.dtype == np.float16
    assert order.shape == (3000, 16) and order.dtype == np.int64
    assert (order >= 0).all() and (order < 768).all()
    assert set(targets) == {'lo', 'hi', 'weight', 'valid', 'source_index', 'partitions'}
    for key, dtype in [('lo', np.int16), ('hi', np.int16), ('weight', np.float32), ('valid', np.bool_)]:
        assert targets[key].shape == (1024, 84) and targets[key].dtype == dtype
    assert targets['source_index'].dtype == np.int64
    assert np.array_equal(targets['source_index'], np.arange(1024, dtype=np.int64))
    expected = np.array(['train'] * 768 + ['calibration'] * 128 + ['source_test'] * 128)
    assert np.array_equal(targets['partitions'], expected)
    assert ((targets['lo'] >= 0) & (targets['lo'] <= 65)).all()
    assert ((targets['hi'] >= targets['lo']) & (targets['hi'] <= 65)).all()
    assert np.isfinite(targets['weight']).all()
    assert ((targets['weight'] >= 0) & (targets['weight'] <= 1)).all()
    valid = targets['valid']; positive = valid & (targets['lo'] < 65); none = valid & ~positive
    assert (targets['hi'][positive] < 65).all()
    assert (targets['hi'][none] == 65).all() and (targets['weight'][none] == 0).all()
    assert (targets['hi'][positive] - targets['lo'][positive] <= 1).all()
    assert np.isfinite(features[order[0]]).all()
    manifest = read(args.manifest)
    assert manifest['complete'] and len(manifest['records']) == 1024
    for index, row in enumerate(manifest['records']):
        assert row['index'] == index and row['partition'] == str(targets['partitions'][index])
        assert row['variant'] == 0 and row['composition']['generated'] is False
    return features, order, targets, manifest


def target_counts(targets, np):
    result = {}
    for split in ['train', 'calibration', 'source_test']:
        selected = targets['partitions'] == split
        v = targets['valid'][selected]; lo = targets['lo'][selected]
        result[split] = dict(images=int(selected.sum()), POSITIVE=int((v & (lo < 65)).sum()),
                             NONE=int((v & (lo == 65)).sum()), IGNORE=int((~v).sum()))
        assert result[split]['POSITIVE'] > 0 and result[split]['NONE'] > 0
        assert sum(result[split][key] for key in ['POSITIVE', 'NONE', 'IGNORE']) == int(selected.sum()) * 84
    return result


def freeze(args):
    np, torch, M = modules(args.source_root)
    torch.set_num_threads(1)
    original = read(OLD_DOC / 'LEARNING_PROTOCOL.json')
    assert list(M.ARMS) == original['arms'] and M.CONFIG == original['model']
    assert original['seed'] == 1 and original['updates'] == 3000 and original['batch'] == 16
    ready = read(args.ready_validation)
    assert ready['complete'] and ready['source_supervision_ready']
    assert sha(args.targets) == ready['ready_arrays_sha256']
    features, order, targets, manifest = load_arrays(args, np)
    torch.manual_seed(1); head = M.CorrespondenceHead()
    initial = state_sha(head.state_dict())
    assert initial == original['initial_state_sha256']
    assert hashlib.sha256(order.tobytes()).hexdigest() == original['batch_order_sha256']
    assert sum(p.numel() for p in head.parameters()) == original['parameters'] == 5890
    protocol = dict(schema='fixed_source_supervision_repair_retraining_protocol_v1',
        status='PREPARED_AWAITING_ADDITIONAL_UPDATE_AUTHORIZATION',
        original_formal_updates_completed=9000, original_remaining_formal_updates=0,
        requested_additional_formal_updates=9000, executed_additional_formal_updates=0,
        source_repair_only=True, arms=list(M.ARMS), model=M.CONFIG, parameters=5890,
        seed=1, updates_per_arm=3000, batch=16, formal_RGB_exposures=144000,
        optimizer=original['optimizer'], learning_rate_formula='.001*(step/100 if step<=100 else .5*(1+cos(pi*(step-100)/2900)))',
        checkpoint='last only; no best-source or real-result checkpoint selection',
        initial_state_sha256=initial, batch_order_sha256=original['batch_order_sha256'],
        batch_order_file_sha256=sha(args.order), first_batch_source_indices=order[0].tolist(),
        target_counts=target_counts(targets, np), fixed_inputs=snapshots(args),
        source_splits=dict(train=768, calibration=128, source_test=128),
        input_channels=dict(GEOMETRY_ONLY='zero image0..18; preserve geometry19..24 and geometric role25..27 (exact original implementation)',
                            IMAGE_NO_ROLE='preserve image0..18/geometry19..24; zero role25..27',
                            IMAGE_ROLE='all original28 channels'),
        loss='Original per-image mean soft 66-way CE; POS linearly interpolated adjacent bins0..64; NONE index65; IGNORE contributes exactly zero and all-ignored images excluded from mean',
        source_probes=dict(initial_source_test_128=True, steps=[1000,2000,3000],
                           train_indices=list(range(128)), source_test_indices=list(range(896,1024)),
                           head_forward_calls=168, image_exposures=2688, selection=False),
        throwaway_updates=0, justification='Original throwaway numerical preflight is not repeated; zero-update checks replace it. Requested extra budget covers exactly three formal3000-update fits.',
        fresh_original_RGB=0, fresh_feature_detector_calls=0, feature_cache_reused_readonly=True,
        source_test_limit='Source-test was inspected to diagnose a deterministic supervisor bug; it remains excluded from fitting and cannot tune this fixed replay. The corrected ideal graph is an upper bound, not learned performance.',
        downstream=dict(frames=319, sessions=13, decode='unchanged original66-way argmax + per-edge TLS + fixed corner intersections; no match-mass or threshold change',
                        method_paths=['GEOMETRY_ONLY','IMAGE_NO_ROLE','IMAGE_ROLE','IMAGE_ROLE_NO_MASK_ROBUST','IMAGE_ROLE_STANDARD','IMAGE_ROLE_POINT_LINE'],
                        rows=1914, point_solver_paths=1595, point_line_paths=319,
                        robust='original finite4-subset bank,8px,top3 refinement; same bank across masks; excluded self corners never fit',
                        hidden_output='if and only if newR,t available replace initial predicted-self corners with final reprojections; never refit using reprojections',
                        point_line='separate original same-IMAGE_ROLE ablation; remove all lines supporting consumed point intersections; no duplicated source-edge support',
                        no_correspondence_filling=True, independent_pose_not_classification_success_gate=True,
                        reference_reads='only after all1914 geometry rows sealed; GEOMETRIC_PROXY limits disclosed',
                        comparators=['BASE','N3_SUBPIX','original IMAGE_ROLE','original IMAGE_ROLE_POINT_LINE'],
                        primary='repaired IMAGE_ROLE versus fixed N3_SUBPIX on all319 operational rows',
                        report_all_three_models=True, no_winner_reselection=True,
                        metrics=['position_cm','rotation_deg','ADDsym_cm'], summary=['mean','sample_variance','sample_sd','median','linearP90','max'],
                        statuses=['NEW_POSE','BASELINE_FALLBACK','POSE_FAILURE'], common_sets=True,
                        visibility='DIRECT native damage and SELF/hidden reprojection separately; mask-error vs pose-quality separate',
                        paired_bootstrap='same published10000×13 session multiplicity matrix; zero additional draws',
                        runtime='measure original full detector+features+initial pose+observations+final solver path in exclusive quiet window; cached replay time is not runtime'),
        authorization_required='Separate JSON statusAPPROVED, additional_formal_updates9000 and exact frozen protocol/checksSHA; absent or different receipt aborts before any optimizer object. Exclusive adjacent EXECUTION_CLAIM prevents accidental second training run with the same authorization.',
        readiness_does_not_establish_real_success=True)
    write_new(args.protocol, protocol)
    print('FROZEN_REPAIR_PROTOCOL', sha(args.protocol), flush=True)


def verify_protocol(args):
    protocol = read(args.protocol)
    assert protocol['status'] == 'PREPARED_AWAITING_ADDITIONAL_UPDATE_AUTHORIZATION'
    assert snapshots(args) == protocol['fixed_inputs'], 'Frozen input or code changed'
    return protocol


def tensor_batch(features, targets, ids, device, np, torch):
    x = torch.tensor(np.array(features[ids]), device=device, dtype=torch.float32)
    ts = {key: torch.tensor(np.array(targets[key][ids]), device=device,
                           dtype=torch.float32 if key == 'weight' else torch.bool if key == 'valid' else torch.long)
          for key in ['lo','hi','weight','valid']}
    return x, ts


def check(args):
    np, torch, M = modules(args.source_root); torch.set_num_threads(1)
    begin = time.monotonic(); protocol = verify_protocol(args)
    features, order, targets, manifest = load_arrays(args, np)
    ids = np.array(order[0]); x, ts = tensor_batch(features, targets, ids, 'cpu', np, torch)
    counts = dict(images=16, queries=16*84, POSITIVE=int((ts['valid'] & (ts['lo']<65)).sum()),
                  NONE=int((ts['valid'] & (ts['lo']==65)).sum()), IGNORE=int((~ts['valid']).sum()))
    results = []; autograd_calls = 0; backward_calls = 0
    # Inspect all arms at exactly the original initialization and source batch.
    for arm in M.ARMS:
        torch.manual_seed(1); head = M.CorrespondenceHead().cpu()
        before = state_sha(head.state_dict()); assert before == protocol['initial_state_sha256']
        logits = head(x,arm); logits.retain_grad()
        total = M.loss(logits,**ts); assert torch.isfinite(total)
        scopes = {}
        for name, mask in [('POSITIVE',ts['valid'] & (ts['lo']<65)), ('NONE',ts['valid'] & (ts['lo']==65)), ('ALL_IGNORED',torch.zeros_like(ts['valid']))]:
            loss = M.loss(logits,ts['lo'],ts['hi'],ts['weight'],mask)
            grad = torch.autograd.grad(loss,head.body[0].weight,retain_graph=True)[0]; autograd_calls += 1
            scopes[name] = dict(queries=int(mask.sum()),loss=float(loss.detach()),first_layer_gradient_norm=float(grad.norm()))
            assert torch.isfinite(grad).all()
            assert (float(grad.norm()) == 0) if name == 'ALL_IGNORED' else (float(grad.norm()) > 0)
        logits.grad = None
        total.backward(); backward_calls += 1
        gradient = logits.grad.detach(); valid = ts['valid']
        ignored_max = float(gradient[~valid].abs().max()) if (~valid).any() else 0.
        valid_min = float(gradient[valid].abs().sum(-1).min())
        assert ignored_max == 0 and valid_min > 0
        distribution = torch.zeros_like(logits)
        distribution.scatter_add_(-1,ts['lo'][...,None],(1-ts['weight'])[...,None])
        distribution.scatter_add_(-1,ts['hi'][...,None],ts['weight'][...,None])
        divisor = valid.sum(-1).clamp(min=1).to(logits.dtype)[:,None,None]
        image_count = int((valid.sum(-1)>0).sum())
        expected = (logits.detach().softmax(-1)-distribution)*valid[...,None]/divisor/max(1,image_count)
        derivative_error = float((gradient-expected).abs().max()); assert derivative_error < 1e-7
        gradients = {key: float(p.grad.norm()) if p.grad is not None else None for key,p in head.named_parameters()}
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
        image_gradient=float(head.body[0].weight.grad[:,:19].norm())
        role_gradient=float(head.body[0].weight.grad[:,25:28].norm())
        if arm=='GEOMETRY_ONLY': assert image_gradient==0
        if arm=='IMAGE_NO_ROLE': assert role_gradient==0
        if arm=='IMAGE_ROLE': assert image_gradient>0 and role_gradient>0
        after=state_sha(head.state_dict());assert after==before
        results.append(dict(arm=arm,parameters=5890,initial_state_sha256=before,after_backward_state_sha256=after,
                            loss=float(total.detach()),scoped_gradient=scopes,ignored_logit_gradient_max=ignored_max,
                            valid_query_logit_gradient_L1_min=valid_min,analytic_CE_derivative_max_error=derivative_error,
                            image_channel_gradient_norm=image_gradient,role_channel_gradient_norm=role_gradient,
                            parameter_gradient_norms=gradients))
    uniform=torch.zeros((16,84,66),requires_grad=True)
    uniform_loss=M.loss(uniform,**ts); uniform_grad=torch.autograd.grad(uniform_loss,uniform)[0];autograd_calls+=1
    assert abs(float(uniform_loss.detach())-math.log(66))<1e-6
    assert torch.count_nonzero(uniform_grad[~ts['valid']])==0
    assert snapshots(args)==protocol['fixed_inputs']
    value=dict(schema='zero_update_repaired_supervision_cpu_checks_v1',complete=True,passed=True,
               protocol=binding(args.protocol),fixed_inputs=protocol['fixed_inputs'],target_counts=target_counts(targets,np),
               first_batch_indices=ids.tolist(),first_batch_ids=[manifest['records'][i]['id'] for i in ids],
               first_batch_target_counts=counts,
               first_batch_features_float32_sha256=hashlib.sha256(x.numpy().tobytes()).hexdigest(),
               first_batch_targets={key:targets[key][ids].tolist() for key in ['lo','hi','weight','valid']},
               head_checks=results,uniform_logit_loss=float(uniform_loss.detach()),expected_uniform_logit_loss=math.log(66),
               execution=dict(device='cpu',head_forward_calls=3,head_image_exposures=48,
                              loss_only_uniform_evaluations=1,autograd_grad_calls=autograd_calls,
                              backward_calls=backward_calls,optimizer_instances=0,optimizer_steps=0,
                              training_updates=0,model_weight_files_written=0,detector_forwards=0,
                              rays=0,PnP=0,raw_RGB_generated=0,source_feature_files_changed=0),
               after_input_hashes_equal=True,wall_seconds=time.monotonic()-begin,
               limits='Actual CPU initialization/loss/backward checks on one frozen train batch; no learning convergence or independent private-feature reproduction, source-test guarantee or real pose improvement is claimed.')
    write_new(args.checks,value)
    print('ZERO_UPDATE_CHECKS_PASS',sha(args.checks),json.dumps(value['execution']),flush=True)


def probe(head,arm,features,targets,indices,np,torch,M):
    values=[];counts=Counter();positive=[];negative=[];head.eval()
    with torch.no_grad():
        for start in range(0,len(indices),16):
            x,ts=tensor_batch(features,targets,indices[start:start+16],'cuda',np,torch)
            logits=head(x,arm);values.append(float(M.loss(logits,**ts)));choice=logits.argmax(-1)
            valid=ts['valid'];pos=valid&(ts['lo']!=65);neg=valid&(ts['lo']==65);accepted=pos&(choice<65)
            counts.update(images=len(x),valid=int(valid.sum()),ignored=int((~valid).sum()),positive=int(pos.sum()),
                          no_match=int(neg.sum()),selected=int(((choice!=65)&valid).sum()),positive_accepted=int(accepted.sum()))
            if accepted.any():positive.extend(abs(choice[accepted].float()-(ts['lo'][accepted]+ts['weight'][accepted])).cpu().tolist())
            if neg.any():negative.extend((choice[neg]!=65).cpu().tolist())
    head.train()
    return dict(loss_image_average=float(np.mean(values)),counts=dict(counts),
                positive_adoption_rate=counts['positive_accepted']/max(1,counts['positive']),
                positive_candidate_error_px_mean_conditional_accepted=float(np.mean(positive)) if positive else None,
                no_match_false_acceptance=float(np.mean(negative)) if negative else None)


def authorization(args):
    assert args.authorization,'Additional9000 updates require a separate user-authorization receipt'
    receipt=read(args.authorization);assert receipt['status']=='APPROVED'
    assert receipt['additional_formal_updates']==9000
    assert receipt['protocol_sha256']==sha(args.protocol)
    assert receipt['checks_sha256']==sha(args.checks)
    assert receipt.get('user_authorization_evidence'), 'Receipt must identify explicit user authorization'
    checks=read(args.checks);assert checks['complete'] and checks['passed']
    assert checks['execution']['training_updates']==0
    return receipt


def train(args):
    # Authorization is checked before module/CUDA/optimizer creation.
    approved=authorization(args)
    np,torch,M=modules(args.source_root);torch.set_num_threads(1)
    protocol=verify_protocol(args); features,order,targets,_=load_arrays(args,np)
    assert torch.cuda.is_available()
    output=Path(args.output).resolve();assert not output.exists(),'Use a new isolated output directory'
    reject_protected_output(output,args.source_root)
    for path in input_paths(args).values():
        assert not path.resolve().is_relative_to(output),'Output may not enclose any read-only input'
    claim=Path(args.authorization).with_name(Path(args.authorization).stem+'.EXECUTION_CLAIM.json')
    write_new(claim,dict(schema='single_additional_update_authorization_claim_v1',authorization=binding(args.authorization),
                        protocol=binding(args.protocol),requested_formal_updates=9000,output_directory_name=output.name,
                        limit='One execution claim for this authorization; an interrupted run must be audited before any further updates.'))
    output.mkdir(parents=True)
    write_new(output/'AUTHORIZED_TRAINING_START.json',dict(protocol=binding(args.protocol),checks=binding(args.checks),authorization=binding(args.authorization),updates=0))
    begin=time.monotonic();checkpoints=[];formal_updates=0;source_forward_calls=0
    logs=(output/'TRAIN_LOGS.jsonl').open('x'); formal=(output/'FORMAL_UPDATE_ROWS.jsonl').open('x')
    def log(value):logs.write(json.dumps(value,allow_nan=False)+'\n');logs.flush()
    try:
        for arm in M.ARMS:
            torch.manual_seed(1);torch.cuda.manual_seed_all(1);head=M.CorrespondenceHead().cuda()
            initial=state_sha(head.state_dict());assert initial==protocol['initial_state_sha256']
            optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.0001)
            arm_begin=time.monotonic();initial_probe=probe(head,arm,features,targets,np.arange(896,1024),np,torch,M);source_forward_calls+=8
            first=None
            for step in range(1,3001):
                ids=order[step-1];x,ts=tensor_batch(features,targets,ids,'cuda',np,torch)
                lr=.001*(step/100 if step<=100 else .5*(1+math.cos(math.pi*(step-100)/2900)))
                for group in optimizer.param_groups:group['lr']=lr
                optimizer.zero_grad(set_to_none=True);logits=head(x,arm);value=M.loss(logits,**ts);assert torch.isfinite(value)
                value.backward();gradient=float(torch.nn.utils.clip_grad_norm_(head.parameters(),10,error_if_nonfinite=True))
                if step==1:
                    first=dict(parameter_gradient_norms={k:float(p.grad.norm()) for k,p in head.named_parameters()},
                               image_channel_gradient=float(head.body[0].weight.grad[:,:19].norm()),
                               role_channel_gradient=float(head.body[0].weight.grad[:,25:28].norm()))
                    if arm=='GEOMETRY_ONLY':assert first['image_channel_gradient']==0
                    if arm=='IMAGE_NO_ROLE':assert first['role_channel_gradient']==0
                optimizer.step();formal_updates+=1
                formal.write(json.dumps(dict(arm=arm,step=step,source_indices=ids.tolist(),lr=lr,loss=float(value),gradient_norm=gradient),allow_nan=False)+'\n')
                formal.flush()
                if step==1 or step%100==0:
                    torch.cuda.synchronize();probs=logits.detach().softmax(-1)
                    correct=(1-ts['weight'])*probs.gather(-1,ts['lo'][...,None])[...,0]+ts['weight']*probs.gather(-1,ts['hi'][...,None])[...,0]
                    row=dict(kind='formal',arm=arm,step=step,loss_image_average=float(value),lr=lr,gradient_norm=gradient,
                             seconds=time.monotonic()-arm_begin,batch_images=16,valid_queries=int(ts['valid'].sum()),
                             positive_queries=int((ts['valid']&(ts['lo']!=65)).sum()),no_match_queries=int((ts['valid']&(ts['lo']==65)).sum()),
                             ignored_queries=int((~ts['valid']).sum()),target_correct_probability=float(correct[ts['valid']].mean()))
                    log(row);print('REPAIR_TRAIN',arm,step,3000,round(float(value),4),flush=True)
                if step in [1000,2000,3000]:
                    log(dict(kind='source_curve',arm=arm,step=step,
                             train=probe(head,arm,features,targets,np.arange(128),np,torch,M),
                             source_test=probe(head,arm,features,targets,np.arange(896,1024),np,torch,M)))
                    source_forward_calls+=16
                if time.monotonic()-begin>3600:raise RuntimeError('Original formal60min budget reached; preserve partial outputs')
            checkpoint=output/(arm+'.pt')
            torch.save(dict(model=head.cpu().state_dict(),arm=arm,config=M.CONFIG,steps=3000,
                            initial_state_sha256=initial,batch_order_sha256=protocol['batch_order_sha256'],
                            protocol_sha256=sha(args.protocol),repaired_target_sha256=sha(args.targets)),checkpoint)
            checkpoints.append(dict(arm=arm,updates=3000,exposures=48000,checkpoint=binding(checkpoint),
                                    initial_probe=initial_probe,first_step=first,seconds=time.monotonic()-arm_begin))
            del head,optimizer;torch.cuda.empty_cache()
        assert formal_updates==9000 and source_forward_calls==168
        assert snapshots(args)==protocol['fixed_inputs']
        torch.cuda.synchronize()
        write_new(output/'TRAINING_COMPLETION.json',dict(schema='fixed_supervision_repair_training_completion_v1',complete=True,
                  formal_updates=9000,throwaway_updates=0,total_updates=9000,formal_RGB_exposures=144000,
                  source_probe_head_calls=168,source_probe_image_exposures=2688,total_head_forward_calls=9168,
                  batch=16,seed=1,checkpoints=checkpoints,protocol=binding(args.protocol),authorization=binding(args.authorization),
                  same_initial_tensor_sha=True,same_batch_order=True,same_update_budget=True,source_scores_model_selection=False,
                  input_hashes_unchanged=True,wall_seconds=time.monotonic()-begin))
    finally:
        logs.close();formal.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['freeze','check','train'])
    p.add_argument('--source-root',default=os.environ.get('PALLET_SOURCE_ROOT'))
    p.add_argument('--features',default=str(PRIVATE_CACHE/'features.npy'));p.add_argument('--order',default=str(PRIVATE_CACHE/'order.npy'))
    p.add_argument('--manifest',default=str(PRIVATE_CACHE/'CACHE_MANIFEST.json'))
    p.add_argument('--targets',default=str(READY/'READY_PREPARED_TARGETS.npz'))
    p.add_argument('--ready-validation',default=str(READY/'DEPTH_RECOVERY_VALIDATION.json'))
    p.add_argument('--protocol',default=str(DOC/'RETRAINING_PROTOCOL.json'))
    p.add_argument('--checks',default=str(DOC/'ZERO_UPDATE_CPU_CHECKS.json'))
    p.add_argument('--authorization');p.add_argument('--output',default='/dev/shm/pallet-kp-supervision-repair-private-20261010/learned_fits')
    args=p.parse_args()
    {'freeze':freeze,'check':check,'train':train}[args.stage](args)


if __name__=='__main__':main()
