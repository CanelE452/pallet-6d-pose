"""One minimum contract run, including exactly three real parity forwards."""
import copy
import inspect
import time
from pathlib import Path
import numpy as np
import torch
from .common import (ROOT, DOC, OUTPUT, BANKS, OLD_DOC, Data, Banks, read, write,
                     sha, dcp_env, numeric_contract, load_setup, verify_derived, pose_loss)
from .model import (initialize, ResidualJointScorer, projected_tokens, residual_score,
                    descriptors_from_inherited, action_scores, decode_bank)
from scripts.research.pallet_pose_target_6d_20261006_v1.training import initialize as old_initialize


def run_tests():
    started=time.monotonic()
    assert (DOC/'PROTOCOL.json').exists(), 'Protocol must precede model creation'
    protocol=read(DOC/'PROTOCOL.json'); setup=load_setup(); verify_derived(setup)
    bindings={name:sha(Path(__file__).with_name(name)) for name in ('model.py','tests.py')}
    attempt_path=OUTPUT/'TEST_ATTEMPT_STATE.json'
    existing=read(DOC/'RUN_RECEIPTS.json').get('tests')
    if existing:
        assert existing['status']=='PASS' and existing['code_sha256']==bindings
        return existing
    assert not attempt_path.exists(), 'Prior incomplete test cannot silently repeat real forwards'
    attempt=dict(status='RUNNING',code_sha256=bindings,actual_model_batches=0,
                 actual_model_examples=0,actual_F_calls=0,optimizer_updates=0)
    write(attempt_path,attempt)
    numeric_contract(); config=protocol['config']; checks={}
    local=initialize('LOCAL_CAP',config)
    joint=initialize('JOINT8',config)
    assert sum(p.numel() for p in local.residual_head.parameters())==4680
    assert sum(p.numel() for p in joint.residual_head.parameters())==4680
    assert sum(p.numel() for p in local.parameters())==sum(p.numel() for p in joint.parameters())==26169
    assert local.common_state_sha()==joint.common_state_sha()
    assert set(local.common_state_dict())==set(joint.common_state_dict())
    assert all(torch.equal(v,joint.common_state_dict()[k]) for k,v in local.common_state_dict().items())
    assert local.initialization_metadata['inherited_state_sha256']==joint.initialization_metadata['inherited_state_sha256']
    assert not local.residual_head[2].weight.any() and not joint.residual_head[2].weight.any()
    checks['parameter_counts_common_state_and_zero_final_weight']=True

    # Explicit interaction difference: changing two corners crosses one joint
    # ReLU boundary; independently applied local nonlinearities remain additive.
    lh=copy.deepcopy(local.residual_head); jh=copy.deepcopy(joint.residual_head)
    with torch.no_grad():
        for h in (lh,jh):
            for p in h.parameters():p.zero_()
            h[0].bias[0]=-1;h[2].weight[0,0]=1
        lh[0].weight[0,0]=1
        jh[0].weight[0,0]=1;jh[0].weight[0,16]=1
    support=torch.ones((1,8),dtype=torch.bool); actions=torch.ones((1,1),dtype=torch.bool)
    tokens=torch.zeros((1,1,8,16));tokens[...,15]=1
    variants=[]
    for first,second in ((0,0),(.75,0),(0,.75),(.75,.75)):
        v=tokens.clone();v[0,0,0,0]=first;v[0,0,1,0]=second;variants.append(v)
    interaction={}
    for method,h in (('LOCAL_CAP',lh),('JOINT8',jh)):
        v=[residual_score(method,h,t,support,actions) for t in variants]
        interaction[method]=float((v[3]-v[1]-v[2]+v[0]).item())
    assert interaction['LOCAL_CAP']==0 and interaction['JOINT8']==.5
    checks['independent_local_and_nonadditive_joint_interaction']=True

    # Shared descriptor construction includes valid-motion NoOp aggregates.
    pooled=torch.arange(2*8*2*48,dtype=torch.float32).reshape(2,8,2,48)/100
    action_valid=torch.tensor([[True,True,False],[True,True,True]])
    weights=action_valid[:,1:].float()
    null=(pooled*weights[:,None,:,None]).sum(-2)/weights.sum(-1)[:,None,None]
    raw=torch.zeros((2,8,13));disp=torch.ones((2,8,3,2));coverage=torch.full((2,8,2),.5)
    coverage[0,:,1]=.9;embedding=torch.zeros((2,16))
    supported=torch.ones((2,8),dtype=torch.bool);supported[0,3]=False;supported[1]=False
    # Nonfinite missing data must be removed before phi, including its bias.
    pooled[0,3]=float('nan');null[0,3]=float('nan');raw[0,3]=float('inf')
    desc=descriptors_from_inherited(pooled,null,raw,disp,coverage,embedding,action_valid,supported)
    assert desc.shape==(2,3,8,81) and torch.isfinite(desc).all()
    assert not desc[0,:,3].any() and not desc[1].any() and not desc[0,2].any()
    assert not desc[0,0,0,61:63].any() and desc[0,0,0,63]==.5 and desc[0,0,0,80]==1
    assert torch.equal(desc[0,0,0,:48],pooled[0,0,0])
    tok=projected_tokens(local.phi,desc,supported,action_valid)
    assert tok.shape==(2,3,8,16) and torch.isfinite(tok).all()
    assert not tok[0,:,3].any() and not tok[1].any() and not tok[0,2].any()
    checks['NoOp_descriptor_coverage_and_prediction_masks']=True
    for method,h in (('LOCAL_CAP',local.residual_head),('JOINT8',joint.residual_head)):
        delta=residual_score(method,h,tok,supported,action_valid)
        assert torch.isfinite(delta).all() and not delta.any()
        out=dict(point_support=supported,logits=torch.zeros((2,8,3))+delta[:,None],action_valid=action_valid)
        score=action_scores(out)
        assert score.argmax(-1).tolist()==[0,0] and torch.isneginf(score[0,2])
        assert torch.equal(score[1],torch.tensor([0.,float('-inf'),float('-inf')]))
        one=dict(point_support=torch.zeros((2,8),dtype=torch.bool),logits=torch.zeros((2,8,1)),action_valid=torch.ones((2,1),dtype=torch.bool))
        assert torch.equal(action_scores(one),torch.zeros((2,1)))
    checks['variable_counts_padding_support0_single_NoOp']=True

    # Pure token/descriptor CE backwards only: no feature CNN, F or optimizer.
    toy_backward=0; gradient_evidence={}
    rng=torch.Generator().manual_seed(113)
    for model in (local,joint):
        model.zero_grad(set_to_none=True)
        d=torch.randn((2,3,8,81),generator=rng)
        valid=torch.ones((2,3),dtype=torch.bool);sup=torch.ones((2,8),dtype=torch.bool)
        t=projected_tokens(model.phi,d,sup,valid)
        delta=residual_score(model.method,model.residual_head,t,sup,valid)
        y=torch.tensor([[.1,.3,.6],[.6,.1,.3]])
        loss,meta=pose_loss(delta,y,torch.ones(2,dtype=torch.bool),'SOFT6D')
        assert torch.isfinite(loss) and meta['eligible_frames']==2
        loss.backward();toy_backward+=1
        grad={n:float(p.grad.norm()) for n,p in model.named_parameters() if p.grad is not None}
        assert all(np.isfinite(v) for v in grad.values())
        assert grad['residual_head.2.weight']>0
        assert grad['phi.0.weight']==grad['phi.0.bias']==0
        assert grad['residual_head.0.weight']==grad['residual_head.0.bias']==0
        gradient_evidence[model.method]=grad
        model.zero_grad(set_to_none=True)
    checks['finite_saved_SOFT6D_CE_gradient_and_zero_last_initialization']=True
    sig=inspect.signature(ResidualJointScorer.forward_bank)
    assert set(sig.parameters)=={'self','p3','p4','points','boxes','point_valid','input_shape','context','candidate_points','action_valid'}
    checks['inference_signature_excludes_GT_cost_oracle_ID']=True

    data=Data(ROOT);banks=Banks(data,BANKS,setup)
    order_path=ROOT/'data/pallet/results/pallet_final_ml_contribution_test_v1/B_line_vs_point/order_seed1.npy'
    rows=np.load(order_path,mmap_mode='r')[0]
    assert len(rows)==16 and np.isin(rows,data.train_rows).all()
    batch=data.batch(rows,supervision=False)
    assert not any(k.startswith('gt') or k in ('truth','oracle','cost','permutations','group_valid') for k in batch)
    candidates,valid=banks.tensor_batch(rows,batch)
    assert candidates.shape==(16,201,9,2) and valid.all()
    models=[('ORIGINAL',old_initialize(1,config,'cuda').eval()),
            ('LOCAL_CAP',local.to('cuda').eval()),('JOINT8',joint.to('cuda').eval())]
    results={}
    with torch.no_grad():
        for name,head in models:
            attempt['actual_model_batches']+=1;attempt['actual_model_examples']+=len(rows)
            attempt['currently_attempting']=name;write(attempt_path,attempt)
            out=head.forward_bank(*(batch[k] for k in ('p3','p4','points','boxes','point_valid','input_shape')),
                context=batch['context'],candidate_points=candidates,action_valid=valid)
            results[name]=out
    original=results['ORIGINAL'];selected,index=decode_bank(original)
    for name in ('LOCAL_CAP','JOINT8'):
        out=results[name]
        for key in ('logits','point_support','point_valid','candidate_points','points_raw','box_diagonal','coverage'):
            assert torch.equal(original[key],out[key]), 'Initial real parity failed: '+name+'/'+key
        assert torch.equal(original['logits'],out['inherited_logits']) and not out['residual_scores'].any()
        assert torch.equal(action_scores(original),action_scores(out))
        q,i=decode_bank(out);assert torch.equal(q,selected) and torch.equal(i,index)
        assert out['descriptors'].shape==(16,201,8,81) and out['tokens'].shape==(16,201,8,16)
    assert torch.equal(results['LOCAL_CAP']['descriptors'],results['JOINT8']['descriptors'])
    assert torch.equal(results['LOCAL_CAP']['tokens'],results['JOINT8']['tokens'])
    checks['real_fixed_first_batch_bit_exact_zero_residual']=True
    torch.cuda.synchronize()
    report=dict(status='PASS',checks=checks,check_count=len(checks),code_sha256=bindings,
        seconds=time.monotonic()-started,actual_model_batches=3,actual_model_examples=48,
        actual_F_calls=0,backbone_forwards=0,optimizer_updates=0,toy_backward_calls=toy_backward,
        toy_scope='descriptor/projection/residual heads only; no cached RGB feature CNN or F',
        source_rows=rows.tolist(),order_sha256=sha(order_path),executed_descriptor_dim=81,
        local_joint_interaction_difference=interaction,initialization={m.method:m.initialization_metadata for m in (local,joint)},
        zero_last_weight_gradient_evidence=gradient_evidence,
        inherited_unused_parameters=[n for n,_ in local.named_parameters() if n.startswith('radial_')],
        parameter_equality_does_not_imply_equal_compute=True,
        NoOp_descriptor='existing null_pool and valid-motion average coverage; no new RAW patch sampling')
    attempt.update(status='PASS',currently_attempting=None,completed_model_batches=3)
    write(attempt_path,attempt)
    # Runner stores this small report in its single RUN_RECEIPTS artifact.
    del results,models,original,out,batch,candidates
    torch.cuda.empty_cache()
    return report
