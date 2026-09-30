"""Matched recording-composition experiment; optimization never opens real GT."""
import argparse
import gc
import json
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch

from . import common as C
from scripts.research.pallet_posefix_limited_adaptation_pilot_v1 import train as T
from scripts.research.pallet_sensors_submission_v1.prior_model import TFAdam

L=C.L


def verify_inputs():
    p=C.protocol();inputs=C.read(C.DOC/'INPUTS.json')
    assert inputs['paired_count']==p['sample_size']==251
    assert set(inputs['arms'])==set(C.ARMS) and inputs['GT_input'] is False
    for b in inputs['target_recipe_bindings']+inputs['cache_bindings']:
        C.verify(b)
    for key in ('source_orders','initialization','data_audit'):
        C.verify(inputs[key])
    assert inputs['initialization']['sha256']==p['initialization']['sha256']
    for seed in C.SEEDS:
        b=inputs['real_orders'][str(seed)];C.verify(b)
        order=torch.load(C.ROOT/b['path'],map_location='cpu',weights_only=True).numpy()
        expected=np.random.default_rng(6400+seed).integers(0,251,size=(300,8))
        np.testing.assert_array_equal(order,expected)
    for arm in C.ARMS:
        rows=inputs['arms'][arm]
        assert len(rows)==251 and len({x['id'] for x in rows})==251
        for row in rows:
            C.verify(row['image']);C.verify(row['pair'])
    return inputs


def load_real(inputs,arm):
    return [torch.load(C.ROOT/r['pair']['path'],map_location='cpu',weights_only=False)['pair']['OCC']
            for r in inputs['arms'][arm]]


def preflight():
    assert not (C.DOC/'PRETRAIN_TESTS.json').exists(),'No preflight overwrite'
    inputs=verify_inputs();L.L.setup('cpu');cv2.setNumThreads(1)
    pair_checks={};protected={}
    for arm in C.ARMS:
        valid_corners=0
        for row in inputs['arms'][arm]:
            entry=torch.load(C.ROOT/row['pair']['path'],map_location='cpu',weights_only=False)
            pair=entry['pair'];meta=entry['metadata']
            np.testing.assert_array_equal(pair['CLEAN']['target_valid'],pair['OCC']['target_valid'])
            for item in pair.values():
                valid=np.asarray(item['target_valid'],bool)
                assert len(valid)==9 and not valid[8] and valid[:8].sum()>=4
                assert np.isfinite(item['rgb']).all()
                assert np.isfinite(item['points'][np.asarray(item['valid'],bool)]).all()
                assert np.isfinite(item['target'][valid]).all()
                mapped=L.E.N.C.transform_points(item['target'],np.linalg.inv(item['matrix']))
                np.testing.assert_allclose(mapped[valid],np.asarray(meta['target_original'])[valid],atol=1e-4,rtol=0)
            valid_corners+=int(pair['OCC']['target_valid'].sum())
            protected[row['pair']['path']]=row['pair']
        pair_checks[arm]=dict(frames=251,supervised_corner_slots=valid_corners,common_mask=True,inverse_affine_atol=1e-4)
    source=L.L.SourceData();orders=np.load(C.ROOT/inputs['source_orders']['path'])
    assert orders['source_rows'].shape==(300,8) and len(orders['held_rows'])==256
    assert np.isin(orders['source_rows'],source.train_rows).all()
    assert not np.intersect1d(orders['source_rows'],orders['held_rows']).size
    source_lock=L.read(L.E.N.DOC/'INPUT_LOCK.json')
    for b in source_lock['cache_bindings']+[source_lock['orders']]:C.verify(b);protected[b['path']]=b
    m=L.model('FULL','cpu');initial=L.state_hash(L.A.original_state(m))
    assert initial==L.read(L.DOC/'PREFLIGHT.json')['initial_model_state_sha']
    frozen=L.state_hash(L.frozen_state(m));bn=L.state_hash(L.bn_state(m))
    item=load_real(inputs,C.ARMS[0])[0]
    batch={k:torch.as_tensor(np.stack([item[k]])) for k in ('rgb','points','valid','target','target_valid')}
    base=L.L.load_base()
    with torch.no_grad():
        z=m.eval()(batch['rgb'],batch['points'],batch['valid'])
        reference=base(batch['rgb'],batch['points'],batch['valid'])
        torch.testing.assert_close(z,reference,atol=0,rtol=0)
    m.train();z=m(batch['rgb'],batch['points'],batch['valid'])
    parts=T.parts(z,batch['target'],batch['target_valid'])
    center_changed=batch['target'].clone();center_changed[:,8]=1e6
    center_valid=batch['target_valid'].clone();center_valid[:,8]=True
    ignored=T.parts(z,center_changed,center_valid)
    for k in parts:torch.testing.assert_close(parts[k],ignored[k],atol=0,rtol=0)
    sum(parts.values()).backward()
    assert all(q.grad is None for q in m.parameters() if not q.requires_grad)
    assert any(q.grad is not None and torch.isfinite(q.grad).all() and q.grad.abs().sum()>0 for q in m.parameters() if q.requires_grad)
    assert L.state_hash(L.bn_state(m))==bn and L.state_hash(L.frozen_state(m))==frozen
    assert L.state_hash(L.A.original_state(m))==initial,'Preflight makes no optimizer step'
    modules=(C,T,L,L.L,L.A,L.E,L.E.N.C)
    code={__file__,*(module.__file__ for module in modules)}
    import scripts.research.pallet_sensors_submission_v1.prior_model as prior_model
    code.add(prior_model.__file__)
    for name in ('prepare_diverse.py','finalize_inputs.py'):code.add(str(C.HERE/name))
    C.save(C.DOC/'TRAIN_CODE_LOCK.json',dict(files=[C.bind(f) for f in sorted(code)],created_at=C.now()))
    C.save(C.DOC/'PRETRAIN_TESTS.json',dict(PASS=True,created_at=C.now(),protocol=C.bind(C.DOC/'PROTOCOL_EFFECTIVE.json'),
        original_protocol=C.bind(C.DOC/'GOAL_PROTOCOL.json'),amendment=C.bind(C.DOC/'PROTOCOL_AMENDMENT_01.json'),
        inputs=C.bind(C.DOC/'INPUTS.json'),code_lock=C.bind(C.DOC/'TRAIN_CODE_LOCK.json'),
        protected=list(protected.values()),pair_checks=pair_checks,source_train_only=True,source_heldout_disjoint=True,
        initial_model_state_sha=initial,trainable_contract=L.A.trainable_contract(m),frozen_state_sha=frozen,BN_sha=bn,
        wrapped_base_output_exact=True,center_target_ignored=True,frozen_gradient_absent=True,
        trainable_gradient_finite_nonzero=True,BN_unchanged_after_backward=True,optimizer_updates=0,
        training_seeds=list(C.SEEDS),real_order_verified=True,no_real_evaluation_GT_read=True))
    print('PRETRAIN_PASS',initial,flush=True)


def train(arm,seed):
    assert arm in C.ARMS and seed in C.SEEDS
    p=C.protocol();test=C.read(C.DOC/'PRETRAIN_TESTS.json');assert test['PASS']
    for k in ('protocol','inputs','code_lock'):C.verify(test[k])
    for b in C.read(C.DOC/'TRAIN_CODE_LOCK.json')['files']:C.verify(b)
    name=f'{arm}_s{seed}';folder=C.RAW/'fits'/name;folder.mkdir(parents=True,exist_ok=True)
    fit=C.DOC/f'FIT_{name}.json';dest=folder/'final.pt';trace=folder/'trace.jsonl'
    assert not any(path.exists() for path in (fit,dest,trace,folder/'START.json')),'No duplicate fit or overwrite'
    inputs=verify_inputs();real=load_real(inputs,arm)
    real_order=torch.load(C.ROOT/inputs['real_orders'][str(seed)]['path'],map_location='cpu',weights_only=True).numpy()
    source=L.L.SourceData();orders=np.load(C.ROOT/inputs['source_orders']['path'])
    assert np.isin(orders['source_rows'],source.train_rows).all()
    L.L.setup('cuda');cv2.setNumThreads(1)
    m=L.model('FULL','cuda');torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats()
    initial=L.state_hash(L.A.original_state(m));assert initial==test['initial_model_state_sha']
    frozen=L.state_hash(L.frozen_state(m));bn=L.state_hash(L.bn_state(m));trainable0=L.state_hash(L.trainable_state(m))
    opt=TFAdam([q for q in m.parameters() if q.requires_grad],lr=p['optimizer']['lr'])
    assert {id(q) for g in opt.param_groups for q in g['params']}=={id(q) for q in m.parameters() if q.requires_grad}
    m.train();rng=np.random.default_rng(7102+seed);start=time.monotonic()
    C.save(folder/'START.json',dict(created_at=C.now(),arm=arm,seed=seed,protocol=test['protocol'],inputs=test['inputs'],
        device=torch.cuda.get_device_name(),initial_state_sha=initial,real_order=inputs['real_orders'][str(seed)],source_corruption_rng=7102+seed))
    with trace.open('x') as log,ThreadPoolExecutor(max_workers=4) as workers:
        pending=[workers.submit(source.item,int(i)) for i in orders['source_rows'][0]]
        for step in range(300):
            gpu=L.L.gpu_guard();items=[f.result() for f in pending]
            if step<299:pending=[workers.submit(source.item,int(i)) for i in orders['source_rows'][step+1]]
            rr=[real[int(i)] for i in real_order[step]];ss=[L.L.corrupted(x,rng) for x in items]
            assert all(x['partition']=='train' for x in ss)
            opt.zero_grad(set_to_none=True);losses={}
            for branch,rows in (('real',rr),('source',ss)):
                logs=[]
                for i in range(0,8,2):
                    b=T.batch(rows[i:i+2]);z=m(b['rgb'],b['points'],b['valid'])
                    detail=T.parts(z,b['target'],b['target_valid'])
                    regularizer=L.A.original_weight_l2(m) if branch=='real' else z.new_zeros(())
                    loss=(detail['heatmap']+detail['coordinate']+regularizer)*.25
                    assert torch.isfinite(loss);loss.backward()
                    logs.append(dict(heatmap=float(detail['heatmap'].detach()),coordinate=float(detail['coordinate'].detach()),original_L2=float(regularizer.detach())))
                    del b,z,detail,loss,regularizer
                losses[branch]={k:float(np.mean([v[k] for v in logs])) for k in logs[0]}
            assert all(q.grad is None for q in m.parameters() if not q.requires_grad)
            update=float(opt.step());assert np.isfinite(update) and update>0
            bnh=L.state_hash(L.bn_state(m));assert bnh==bn
            row=dict(step=step+1,real_indices=real_order[step].tolist(),real_ids=[x['id'] for x in rr],
                real_supervised=sum(int(x['target_valid'].sum()) for x in rr),source_ids=[x['id'] for x in ss],
                source_rows=orders['source_rows'][step].tolist(),source_corrupted_points_sha=L.P.array_sha(np.stack([x['points'] for x in ss])),
                real_points_sha=L.P.array_sha(np.stack([x['points'] for x in rr])),real_target_sha=L.P.array_sha(np.stack([x['target'] for x in rr])),
                real_mask_sha=L.P.array_sha(np.stack([x['target_valid'] for x in rr])),loss=losses,update_norm=update,
                trainable_sha=L.state_hash(L.trainable_state(m)),BN_sha=bnh)
            log.write(json.dumps(row)+'\n');log.flush()
            if step==0 or (step+1)%25==0:print('TRAIN',name,step+1,'seconds',round(time.monotonic()-start,1),'gpu',gpu,flush=True)
    assert L.state_hash(L.frozen_state(m))==frozen and L.state_hash(L.trainable_state(m))!=trainable0
    state={k:v.detach().cpu() for k,v in L.A.original_state(m).items()};final_sha=L.state_hash(state)
    C.tensor_save(dest,dict(state=state,step=300,complete=True,arm=arm,seed=seed,protocol_sha256=test['protocol']['sha256'],
        original_protocol_sha256=test['original_protocol']['sha256'],final_state_sha=final_sha))
    C.save(fit,dict(complete=True,arm=arm,seed=seed,updates=300,real_exposures=2400,source_exposures=2400,
        checkpoint=C.bind(dest),trace=C.bind(trace),initial_state_sha=initial,final_state_sha=final_sha,
        frozen_parameters_buffers_unchanged=True,BN_all_unchanged=True,trainable_changed=True,
        optimizer_trainables_only=True,seconds=time.monotonic()-start,final_only=True,
        input_lock=test['inputs'],protocol=test['protocol'],trainable_contract=L.A.trainable_contract(m),
        peak_allocated_GPU_bytes=torch.cuda.max_memory_allocated(),no_real_evaluation_GT_input=True))
    del m,opt,state,source,real;gc.collect();torch.cuda.empty_cache()
    print('FIT_COMPLETE',name,flush=True)


def audit_pairing():
    evidence=[]
    for seed in C.SEEDS:
        logs=[]
        for arm in C.ARMS:
            fit=C.read(C.DOC/f'FIT_{arm}_s{seed}.json');C.verify(fit['checkpoint']);C.verify(fit['trace'])
            logs.append([json.loads(line) for line in (C.ROOT/fit['trace']['path']).read_text().splitlines()])
        assert all(len(x)==300 for x in logs)
        for left,right in zip(*logs):
            for key in ('step','real_indices','source_rows','source_ids','source_corrupted_points_sha','BN_sha'):
                assert left[key]==right[key],(seed,left['step'],key)
        evidence.append(dict(seed=seed,updates=300,identical_real_indices=True,identical_source_RGB_ids_and_corruptions=True))
    C.save(C.DOC/'TRAINING_COMPLETE.json',dict(created_at=C.now(),fits=6,optimizer_updates=1800,paired_seed_checks=evidence,
        fits_bindings=[C.bind(C.DOC/f'FIT_{arm}_s{seed}.json') for seed in C.SEEDS for arm in C.ARMS]))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['preflight','all','one','audit'])
    parser.add_argument('--arm',choices=C.ARMS);parser.add_argument('--seed',type=int,choices=C.SEEDS);args=parser.parse_args()
    if args.command=='preflight':preflight()
    elif args.command=='audit':audit_pairing()
    elif args.command=='one':train(args.arm,args.seed)
    else:
        for seed in C.SEEDS:
            for arm in C.ARMS:train(arm,seed)
        audit_pairing()
