"""Generated-array CPU checks only; no source/private image reads or optimizer steps."""
import json
from pathlib import Path
import tempfile
import numpy as np
import torch
from torch.nn import functional as F
from query_global_amodal_head import (QueryGlobalAmodalHead, sample_nearest_onehot,
                                     marginals, parameter_count, load_model, save_checkpoint)
from fit_global_amodal_head import load_cache, source_balanced_loss, source_balanced_global_loss


def main():
    torch.set_num_threads(1);torch.manual_seed(20261022);torch.use_deterministic_algorithms(True)
    rng=np.random.default_rng(20261022);n,t=4,2
    model=QueryGlobalAmodalHead()
    assert parameter_count(model)==95508
    assert parameter_count(model.global_encoder)==49006
    indices=torch.arange(81).reshape(1,9,9).expand(n,-1,-1).clone()
    indices[0]=4095
    inverse=torch.tensor([0,1,0,1])
    valid=torch.ones(n,9,9,dtype=torch.bool);valid[1,0,0]=False
    field=torch.arange(t*2*64*64,dtype=torch.float32).reshape(t,2,64,64).requires_grad_()
    sampled=sample_nearest_onehot(field,inverse,indices,valid)
    reference=torch.stack([field[inverse[i]].flatten(1)[:,indices[i].flatten()].reshape(2,9,9)for i in range(n)])*valid[:,None]
    assert torch.equal(sampled,reference)
    gradient=torch.autograd.grad(sampled.square().sum(),field,retain_graph=True)[0]
    reference_gradient=torch.autograd.grad(reference.square().sum(),field)[0]
    assert torch.equal(gradient,reference_gradient)
    assert sampled[1,:,0,0].eq(0).all()
    i8=((indices//64)//8)*8+(indices%64)//8
    args=dict(x=torch.randn(n,65,9,9),rgb_native=torch.rand(n,75,9,9),sam=torch.randn(n,2,9,9),
        context=torch.randn(n,64),box_grid=torch.randn(n,4,9,9),global_input=torch.randn(t,5,64,64),
        target_inverse=inverse,global_index64=indices,global_index8=i8,global_query_valid=valid)
    center,dense,global_logits=model(**args,return_dense=True)
    assert center.shape==(n,6) and dense.shape==(n,6,9,9) and global_logits.shape==(t,6,64,64)
    assert torch.equal(center,dense[:,:,4,4])
    joint=torch.randint(6,(n,9,9));gjoint=torch.randint(6,(t,64,64))
    codes=torch.tensor([0,1,0,1]);gcodes=torch.tensor([0,1])
    loss=source_balanced_loss(center,joint[:,4,4],valid[:,4,4],torch.ones(n),codes)
    loss=loss+.25*source_balanced_loss(dense,joint,valid,torch.ones(n,9,9),codes)
    loss=loss+.25*source_balanced_global_loss(global_logits,gjoint,torch.ones(t,64,64,dtype=torch.bool),gcodes)
    loss.backward()
    branches={name:float(sum(p.grad.abs().sum() for key,p in model.named_parameters()if key.startswith(name)))
        for name in ('global_encoder.encoder0','global_encoder.encoder3','global_encoder.decoder0','global_encoder.readout',
                     'p3_encoder','rgb_encoder','sam_encoder','context_encoder','fusion','readout')}
    assert all(value>0 and np.isfinite(value)for value in branches.values())
    probs=marginals(center.softmax(1))
    assert (probs['visible_probability']<=probs['full_probability']+1e-6).all()
    assert (probs['visible_probability'][:,0]<=probs['visible_probability'][:,1]+1e-6).all()
    assert (probs['full_probability'][:,0]<=probs['full_probability'][:,1]+1e-6).all()
    # Generated schema fixture verifies per-axis 64->8 mapping and image linkage.
    i64=indices.numpy().copy();v=valid.numpy();i64[~v]=0
    i8=(((i64//64)//8)*8+(i64%64)//8).astype(np.int64)
    data=dict(x=args['x'].numpy(),rgb_native=rng.integers(256,size=(n,75,9,9),dtype=np.uint8),
        sam=np.zeros((n,2,9,9),np.float16),context=args['context'].numpy(),box_grid=np.zeros((n,4,9,9),np.float32),
        dense_valid=np.ones((n,9,9),bool),joint=joint.numpy(),source=np.array(['SYNTH','YCB','SYNTH','YCB']),
        group_id=np.array(['g0','g1','g0','g1']),sample_id=np.array(['q0','q1','q2','q3']),
        image_sha256=np.array(['a'*64,'b'*64,'a'*64,'b'*64]),target_index=inverse.numpy(),
        global_rgb=np.zeros((t,3,64,64),np.uint8),global_sam=np.zeros((t,1,64,64),np.float16),
        global_prompt=np.ones((t,1,64,64),bool),global_joint=gjoint.numpy().astype(np.int8),
        global_valid=np.ones((t,64,64),bool),global_id=np.array(['t0','t1']),global_image_sha256=np.array(['a'*64,'b'*64]),
        global_box=np.array([[0,0,100,50],[20,30,40,60]],np.float32),global_index64=i64,global_index8=i8,global_query_valid=v)
    with tempfile.TemporaryDirectory(prefix='v65_cpu_mock_')as tmp:
        path=Path(tmp)/'cache.npz';np.savez(path,**data);loaded=load_cache(path)
        assert np.array_equal(loaded['global_source'],['SYNTH','YCB'])
        save_checkpoint(Path(tmp)/'MOCK_NOT_TRAINED.pt',model,optimizer_updates=0)
        restored,_=load_model(Path(tmp)/'MOCK_NOT_TRAINED.pt')
        with torch.inference_mode():assert torch.equal(restored(**args),center)
        broken=dict(data);broken['global_index8']=i8.copy();broken['global_index8'][0,0,0]=0
        np.savez(path,**broken)
        try:load_cache(path)
        except ValueError as exc:assert 'divide each' in str(exc)
        else:raise AssertionError('invalid flattened mapping accepted')
    result={'PASS':True,'scope':'generated CPU arrays only','parameters':parameter_count(model),
        'global_parameters':parameter_count(model.global_encoder),'local_parameters':parameter_count(model)-parameter_count(model.global_encoder),
        'sampler_value_and_gradient_exact':True,'invalid_sample_zero':True,'query_center_exact':True,
        'marginals_nested':True,'checkpoint_roundtrip_exact':True,'schema_and_bad_mapping_rejection':True,
        'gradient_abs_sums':branches,'optimizer_updates':0,'private_data_read':False,'PnP_calls':0}
    out=Path(__file__).with_name('CPU_MODEL_SCHEMA_SMOKE.json');out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
