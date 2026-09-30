"""Exactly three authorized arms; evaluation GT is never opened by this module."""
import argparse
import copy
import gc
import time
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import torch
from . import common as C
from scripts.research.pallet_sensors_submission_v1.prior_model import expectation,target_distribution,TFAdam


def parts(logits,target,valid):
    valid=valid.clone();valid[:,8]=False
    q,mask=target_distribution(target,valid);safe=torch.where(mask[...,None],target,torch.zeros_like(target))
    ce=(-(q*logits.flatten(2).log_softmax(-1)).sum(-1)*mask).mean()
    coord=((expectation(logits)/4-safe/4).abs()*mask[...,None]).mean()
    return dict(heatmap=ce,coordinate=coord)


def batch(rows):return {k:torch.as_tensor(np.stack([r[k] for r in rows]),device='cuda') for k in ('rgb','points','valid','target','target_valid')}


def train(arm):
    assert arm in C.ARMS;p=C.protocol()
    test=C.read(C.DOC/'PRETRAIN_TESTS.json');assert test['PASS'];C.verify(test['protocol']);C.verify(test['code_lock'])
    C.L.setup('cuda');cv2.setNumThreads(1)
    fit=C.DOC/f'FIT_{arm}.json';dest=C.RAW/f'{arm}_last300.pt'
    assert not fit.exists() and not dest.exists() and not (C.RAW/f'TRACE_{arm}.jsonl').exists(),'No rerun/overwrite'
    real,realorder,orders=C.load_inputs();source=C.L.SourceData()
    assert np.isin(orders['source_rows'],source.train_rows).all()
    m=C.model(arm,'cuda');initial=C.state_hash(C.A.original_state(m))
    assert initial==C.read(C.DOC/'PREFLIGHT.json')['initial_model_state_sha']
    freeze_hash=C.state_hash(C.frozen_state(m));bn_hash=C.state_hash(C.bn_state(m));trainable0=C.state_hash(C.trainable_state(m))
    opt=TFAdam([p for p in m.parameters() if p.requires_grad],lr=1e-4)
    assert {id(p) for g in opt.param_groups for p in g['params']}=={id(p) for p in m.parameters() if p.requires_grad}
    m.train();rng=np.random.default_rng(7103);start=time.monotonic();history=[]
    log=(C.RAW/f'TRACE_{arm}.jsonl').open('x')
    try:
        with ThreadPoolExecutor(max_workers=4) as workers:
            pending=[workers.submit(source.item,int(i)) for i in orders['source_rows'][0]]
            for step in range(300):
                gpu=C.L.gpu_guard();items=[f.result() for f in pending]
                if step<299:pending=[workers.submit(source.item,int(i)) for i in orders['source_rows'][step+1]]
                rr=[real[int(i)] for i in realorder[step]]
                ss=[C.L.corrupted(x,rng) for x in items]
                assert all(x['partition']=='train' for x in ss)
                opt.zero_grad(set_to_none=True);losses={}
                for branch,rows in (('real',rr),('source',ss)):
                    logs=[]
                    for i in range(0,8,2):
                        b=batch(rows[i:i+2]);z=m(b['rgb'],b['points'],b['valid']);detail=parts(z,b['target'],b['target_valid'])
                        regularizer=C.A.original_weight_l2(m) if branch=='real' else z.new_zeros(())
                        loss=(detail['heatmap']+detail['coordinate']+regularizer)*.25
                        assert torch.isfinite(loss);loss.backward()
                        logs.append(dict(heatmap=float(detail['heatmap'].detach()),coordinate=float(detail['coordinate'].detach()),original_L2=float(regularizer.detach())))
                        del b,z,detail,loss,regularizer
                    losses[branch]={k:float(np.mean([v[k] for v in logs])) for k in logs[0]}
                assert all(q.grad is None for q in m.parameters() if not q.requires_grad)
                update=float(opt.step());assert np.isfinite(update) and update>0
                bnh=C.state_hash(C.bn_state(m));assert bnh==bn_hash
                row=dict(step=step+1,real_ids=[x['id'] for x in rr],real_supervised=sum(int(x['target_valid'].sum()) for x in rr),
                    source_ids=[x['id'] for x in ss],source_rows=orders['source_rows'][step].tolist(),
                    source_corrupted_points_sha=C.P.array_sha(np.stack([x['points'] for x in ss])),
                    real_points_sha=C.P.array_sha(np.stack([x['points'] for x in rr])),
                    real_target_sha=C.P.array_sha(np.stack([x['target'] for x in rr])),real_mask_sha=C.P.array_sha(np.stack([x['target_valid'] for x in rr])),
                    loss=losses,update_norm=update,trainable_sha=C.state_hash(C.trainable_state(m)),BN_sha=bnh)
                log.write(__import__('json').dumps(row)+'\n');log.flush();history.append(row)
                if step==0 or (step+1)%25==0:print('TRAIN',arm,step+1,'seconds',round(time.monotonic()-start,1),'gpu',gpu,flush=True)
    finally:log.close()
    assert C.state_hash(C.frozen_state(m))==freeze_hash
    assert C.state_hash(C.trainable_state(m))!=trainable0
    if arm=='LORA':assert C.state_hash(C.A.original_state(m))==initial
    state={k:v.detach().cpu() for k,v in m.state_dict().items()}
    C.tensor_save(dest,dict(model_state_dict=state,step=300,arm=arm,protocol_sha256=C.bind(C.DOC/'EXPERIMENT_PROTOCOL.json')['sha256'],base=p['base']))
    C.save(fit,dict(complete=True,arm=arm,updates=300,real_exposures=2400,source_exposures=2400,checkpoint=C.bind(dest),
        trace=C.bind(C.RAW/f'TRACE_{arm}.jsonl'),initial_base_state_sha=initial,final_state_sha=C.state_hash(state),
        frozen_parameters_buffers_unchanged=True,BN_all_unchanged=True,trainable_changed=True,optimizer_trainables_only=True,
        seconds=time.monotonic()-start,final_only=True,input_lock=p['inputs'],protocol=C.bind(C.DOC/'EXPERIMENT_PROTOCOL.json'),
        trainable_contract=C.A.trainable_contract(m),peak_allocated_GPU_bytes=torch.cuda.max_memory_allocated()))
    del m,opt,state,source,real;gc.collect();torch.cuda.empty_cache();print('FIT_COMPLETE',arm,flush=True)


def load_fit(arm):
    p=C.protocol();fit=C.read(C.DOC/f'FIT_{arm}.json');C.verify(fit['checkpoint'])
    ck=torch.load(C.ROOT/fit['checkpoint']['path'],map_location='cpu',weights_only=False)
    assert ck['protocol_sha256']==C.bind(C.DOC/'EXPERIMENT_PROTOCOL.json')['sha256']
    m=C.model(arm);m.load_state_dict(ck['model_state_dict'],strict=True)
    assert C.state_hash(m.state_dict())==fit['final_state_sha']
    return m.cuda().eval().requires_grad_(False)


@torch.no_grad()
def infer(arm):
    C.L.setup('cuda');p=C.protocol();m=load_fit(arm)
    lock=C.read(C.DOC/'INPUT_LOCK.json');C.verify(lock['frozen_baseline'])
    baseline=C.read(C.ROOT/lock['frozen_baseline']['path'])['predictions']['R0'];result={}
    for i,r in enumerate(p['eval_records']):
        C.verify(r['image']);image=cv2.imread(str(C.ROOT/r['image']['path']));raw=baseline[r['id']];original=copy.deepcopy(raw)
        q=C.E.N.C.predict(m,image,raw);assert raw==original;C.E.assert_preserved(raw,q);result[r['id']]=q
        if (i+1)%70==0:C.L.gpu_guard();print('INFER',arm,i+1,flush=True)
    C.save(C.RAW/f'PREDICTIONS_{arm}.json',dict(predictions=result,checkpoint=C.read(C.DOC/f'FIT_{arm}.json')['checkpoint'],GT_input=False,evaluation_filtering=0))
    C.save(C.DOC/f'PREDICTIONS_{arm}.json',dict(binding=C.bind(C.RAW/f'PREDICTIONS_{arm}.json'),frames=len(result),frozen=True))
    del m;gc.collect();torch.cuda.empty_cache();print('INFER_COMPLETE',arm,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('arm',choices=C.ARMS);args=ap.parse_args();train(args.arm);infer(args.arm)
