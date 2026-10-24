#!/usr/bin/env python3
"""Generated arrays/random head only: no source cache, weights, RGB or fit."""
from pathlib import Path
import argparse,hashlib,inspect,json
import numpy as np
import torch

from ownership_head import TargetOwnershipHead,parameter_count
from input_prototype import input_prototypes
from fit_ownership_head import known_ve,source_macro_bce,pair_ranking_loss,build_batch,preflight


def main(output):
    out=Path(output);assert not out.exists();torch.set_num_threads(1);torch.manual_seed(20261023)
    rng=np.random.default_rng(20261023);n=8
    x=np.zeros((n,65,9,9),np.float32);x[:,64]=1
    for i in range(n):x[i,:64]=i+1
    sam=np.full((n,2,9,9),.5,np.float32);box=np.zeros((n,4,9,9),np.float32);ctx=np.full((n,64),-.5,np.float32)
    ti=np.repeat(np.arange(2,dtype=np.int64),4)
    proto,info,audit=input_prototypes(x,sam,box,ctx,ti)
    assert np.all(proto[0]==3) and np.all(proto[3]==2)
    changed=x.copy();changed[0,:64]=1000
    proto_changed,_,_=input_prototypes(changed,sam,box,ctx,ti)
    assert np.array_equal(proto[0],proto_changed[0]) and not np.array_equal(proto[1],proto_changed[1])
    changed_sam=sam.copy();changed_sam[0]=4
    proto_sam,_,_=input_prototypes(x,changed_sam,box,ctx,ti)
    assert np.array_equal(proto[0],proto_sam[0]),'Own-ROI mask must not influence own prototype'
    # Extreme input-mask-negative targets have a declared context fallback, not a veto.
    neg_sam=np.full_like(sam,-4);fallback,flag,fa=input_prototypes(x,neg_sam,box,ctx,ti)
    assert np.array_equal(fallback,ctx) and (flag[:,2]==1).all() and fa['fallback_count']==n
    zero_fallback,zero_flag,_=input_prototypes(x,np.zeros_like(sam),box,ctx,ti)
    assert np.array_equal(zero_fallback,ctx) and (zero_flag[:,2]==1).all(),'Strict zero is not predicted foreground'
    alone,aloneinfo,_=input_prototypes(x[:1],sam[:1],box[:1],ctx[:1],np.zeros(1,np.int64))
    assert np.array_equal(alone,ctx[:1]) and aloneinfo[0,2]==1
    assert set(inspect.signature(input_prototypes).parameters)=={'x','sam','box_grid','context','target_index'}
    # Exhaust every compatible mask: E is ONLY2/4/6, never all even negatives.
    masks=np.arange(1,64,dtype=np.uint8);known,y=known_ve(masks,np.ones(63,bool))
    assert np.array_equal(masks[known],np.array([1,2,4,6])) and y[0]==1 and y[known].sum()==1
    invalid,_=known_ve(masks,np.zeros(63,bool));assert not invalid.any()
    z=torch.randn(63,requires_grad=True);loss=source_macro_bce(z[known],torch.as_tensor(y[known]),torch.zeros(int(known.sum()),dtype=torch.long));loss.backward()
    assert (z.grad[~known]==0).all() and (z.grad[known]!=0).all(),'Excluded O/U must have zero supervised gradient'
    pair=torch.tensor([2.,-2.],requires_grad=True);good=pair_ranking_loss(pair,torch.tensor([0]),torch.tensor([1]));good.backward()
    bad=pair_ranking_loss(torch.tensor([-2.,2.]),torch.tensor([0]),torch.tensor([1]))
    assert good<bad and pair.grad[0]<0 and pair.grad[1]>0
    pools={s:np.arange(100*i,100*i+100) for i,s in enumerate(('SYNTH','ORIGINAL_YCB','CONTROLLED_YCB','PALLOX'))}
    pairs=np.stack((np.arange(100,132),np.arange(200,232)),1)
    batch,op,cp=build_batch(rng,pools,pairs);assert batch.shape==(256,) and op.shape==cp.shape==(32,)
    assert np.array_equal(batch[cp]-batch[op],np.full(32,100))
    counts={s:int(((batch>=100*i)&(batch<100*i+100)).sum()) for i,s in enumerate(pools)};assert set(counts.values())=={64}
    model=TargetOwnershipHead();assert parameter_count(model)<100000
    inp=dict(x=torch.randn(n,65,9,9),rgb_native=torch.rand(n,75,9,9),sam=torch.full((n,2,9,9),-4.),
        box_grid=torch.randn(n,4,9,9),prototype=torch.randn(n,64,requires_grad=True),prototype_info=torch.rand(n,4))
    logits,dense=model(**inp,return_dense=True);assert logits.shape==(n,) and dense.shape==(n,9,9) and torch.equal(logits,dense[:,4,4])
    logits.sum().backward();assert torch.isfinite(inp['prototype'].grad).all() and inp['prototype'].grad.abs().sum()>0
    assert model.rgb_encoder[0].weight.grad.abs().sum()>0 and model.appearance_projection.weight.grad.abs().sum()>0
    with torch.no_grad():model.readout.weight.zero_();model.readout.bias.fill_(7.)
    assert (model(**inp)>0).all(),'SAM-negative queries must be able to output V; no sign prior/veto'
    forbidden=('reference_xy','semantic_slot_id','native_corner','GT','allowed','original_pair_features','controlled_pair_features')
    for key in forbidden:
        try:model(**dict(inp,**{key:torch.zeros(n)}))
        except TypeError:pass
        else:raise AssertionError('Forbidden inference field '+key)
    try:preflight(None,Path('UNUSED_GENERATED_RUN'))
    except ValueError:pass
    else:raise AssertionError('Missing root ENTRYLOCK must fail')
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();folder=Path(__file__).parent
    report=dict(schema='V70_GENERATED_OWNERSHIP_MODEL_PROTOTYPE_LOSS_GUARD_TESTS_V1',PASS=True,
        code={p.name:dict(path=str(p.resolve()),sha256=sha(p)) for p in (folder/'ownership_head.py',folder/'input_prototype.py',folder/'fit_ownership_head.py',Path(__file__))},
        parameters=parameter_count(model),leave_one_entire_ROI_out_and_own_mask_invariance=True,other_ROIs_affect_prototype=True,
        prototype_all_input_ROIs_before_GT_filter=True,empty_foreground_and_single_ROI_context_fallback_flag=True,
        only_strict_positive_predicted_SAM_foreground_used=True,zero_logit_not_foreground=True,epsilon_division_guard_only=True,
        known_V_only1_E_only2_4_6=True,O_U_excluded_zero_supervised_gradient=True,
        four_sources64_each_paired32_sampler=True,pairranking_direction=True,center_direct_readout=True,
        prototype_and_RGB_gradient_nonzero=True,negative_MobileSAM_can_still_predict_V=True,
        forbidden_reference_slot_GT_pair_inference_inputs_rejected=list(forbidden),missing_ENTRYLOCK_fails_before_source_access=True,
        source_cache_RGB_weights_current_reads=0,actual_source_model_forward_calls=0,generated_random_head_forward_only=True,
        optimizer_updates=0,PnP_calls=0,fit_executed=False,goal_scope='Conditional V/E ownership component only; no semantic reference/self/O success claim')
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(PASS=True,parameters=report['parameters'],report=str(out),sha256=sha(out))),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);main(ap.parse_args().out)
