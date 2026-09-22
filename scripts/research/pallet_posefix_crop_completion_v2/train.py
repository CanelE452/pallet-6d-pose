"""Exactly one C300 fit from PRIOR1. No evaluation imports or labels."""
import gc
import json
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import cv2
from . import common as C
from . import data as D
from scripts.research.pallet_sensors_submission_v1.prior_model import TFAdam
from scripts.research.pallet_posefix_full_preserve_v1.run import task_backward
B=C.B

def main():
    p=C.protocol();assert C.read(C.DOC/'PRETRAIN_TESTS.json')['PASS'];assert C.read(C.DOC/'PREDICTIONS_B.json')['frozen']
    for b in C.read(C.DOC/'CODE_LOCK.json')['files']:C.verify(b)
    assert not (C.RAW/'TRACE_C.jsonl').exists() and not (C.DOC/'FIT_C.json').exists(),'Single-use run; no rescue'
    B.L.setup('cuda');cv2.setNumThreads(1);real,ro,orders,corruption=D.load_real();source=D.Source()
    oldtrace=[json.loads(s) for s in (B.RAW/'TRACE_FULL.jsonl').read_text().splitlines()]
    m=B.model('FULL','cuda').train();initial=C.state_hash(B.A.original_state(m));assert initial==C.read(B.DOC/'PREFLIGHT.json')['initial_model_state_sha']
    opt=TFAdam([q for q in m.parameters() if q.requires_grad],lr=p['same_as_FULL']['lr']);assert not opt.state
    bn=C.state_hash(B.bn_state(m));frozen=C.state_hash(B.frozen_state(m));contract=B.A.trainable_contract(m)
    assert contract==C.read(B.DOC/'TRAINABLE_CONTRACTS.json')['FULL']
    C.save(C.DOC/'TRAIN_START.json',dict(initial_state_sha=initial,BN_sha=bn,contract=contract,optimizer_empty=True,seed=1,expansion=C.EXPANSION))
    start=time.monotonic();real_new=source_new=realcount=sourcecount=0
    with (C.RAW/'TRACE_C.jsonl').open('x') as out,ThreadPoolExecutor(max_workers=4) as workers:
        pending=[workers.submit(source.pair,int(i)) for i in orders['source_rows'][0]]
        for step in range(p['same_as_FULL']['updates']):
            gpu=B.L.gpu_guard();pairs=[f.result() for f in pending]
            if step<299:pending=[workers.submit(source.pair,int(i)) for i in orders['source_rows'][step+1]]
            rr=[real[int(i)] for i in ro[step]]
            ss=[D.with_original_points(pair[1],xy,valid) for pair,xy,valid in zip(pairs,corruption['points'][step],corruption['valid'][step])]
            assert [x['id'] for x in rr]==oldtrace[step]['real_ids'];assert [x['id'] for x in ss]==oldtrace[step]['source_ids']
            original_check=np.stack([D.transform_points(x['points'],np.linalg.inv(x['matrix'])) for x in ss])
            np.testing.assert_allclose(original_check,corruption['points'][step],atol=1e-4,rtol=0)
            for x in rr+ss:
                assert all(np.isfinite(x[k]).all() for k in ('rgb','points','target'))
            opt.zero_grad(set_to_none=True);loss=task_backward(m,rr,ss)
            assert all(q.grad is None for q in m.parameters() if not q.requires_grad)
            update=float(opt.step());assert np.isfinite(update) and update>0
            assert C.state_hash(B.bn_state(m))==bn
            rc=sum(int(x['target_valid'].sum()) for x in rr);sc=sum(int(x['target_valid'].sum()) for x in ss)
            rn=rc-oldtrace[step]['real_supervised'];sn=sum(int((b['target_valid']&~a['target_valid']).sum()) for a,b in pairs)
            assert rn>=0 and sn>=0;realcount+=rc;sourcecount+=sc;real_new+=rn;source_new+=sn
            trace=dict(step=step+1,expansion=C.EXPANSION,real_ids=[x['id'] for x in rr],source_ids=[x['id'] for x in ss],
                real_rows=ro[step].tolist(),source_rows=orders['source_rows'][step].tolist(),real_supervised=rc,source_supervised=sc,
                real_newly_supervised=rn,source_newly_supervised=sn,original_source_points_sha=B.P.array_sha(corruption['points'][step]),
                loss=loss,update_norm=update,BN_sha=bn,trainable_sha=C.state_hash(B.trainable_state(m)),gpu=gpu)
            out.write(json.dumps(trace)+'\n');out.flush()
            if step==0 or (step+1)%25==0:print('TRAIN_C',step+1,'seconds',round(time.monotonic()-start,1),gpu,flush=True)
    assert C.state_hash(B.frozen_state(m))==frozen
    state={k:v.detach().cpu() for k,v in m.state_dict().items()};dest=C.RAW/'C_last300.pt'
    C.tensor_save(dest,dict(model_state_dict=state,step=300,protocol=C.bind(C.DOC/'PROTOCOL.json'),initialization=p['same_as_FULL']['base']))
    fit=dict(complete=True,updates=300,new_train_runs=1,checkpoint=C.bind(dest),initial_state_sha=initial,final_state_sha=C.state_hash(state),
        trainable_contract=contract,BN_frozen=True,frozen_parameters_unchanged=True,real_exposures=2400,source_exposures=2400,
        real_supervised_exposures=realcount,source_supervised_exposures=sourcecount,real_newly_supervised_exposures=real_new,source_newly_supervised_exposures=source_new,
        final_only=True,evaluation_during_training=False,seconds=time.monotonic()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated(),trace=C.bind(C.RAW/'TRACE_C.jsonl'))
    C.save(C.DOC/'FIT_C.json',fit);C.save(C.DOC/'TRACE_C.json',dict(trace=fit['trace'],updates=300))
    del m,opt,state,real;gc.collect();torch.cuda.empty_cache();print('C_FIT_COMPLETE',flush=True)

if __name__=='__main__':main()
