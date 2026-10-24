"""Invented CPU arrays/states only. No real data, model weights or feature cache."""
from pathlib import Path
import copy,hashlib,json
import numpy as np
import torch
from independent_query_head import (IndependentAuxiliaryQueryHead, OriginalHead,
                                    ORIGINAL_SOURCE, ORIGINAL_SHA, per_output_loss)

H=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def write(p,x):
    with Path(p).open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def same(a,b):
    if torch.is_tensor(a):
        assert torch.is_tensor(b) and a.dtype==b.dtype and a.shape==b.shape
        assert a.detach().cpu().contiguous().numpy().tobytes()==b.detach().cpu().contiguous().numpy().tobytes()
    elif isinstance(a,dict):
        assert set(a)==set(b)
        for k in a:same(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):same(x,y)
    else:assert a==b
def visstate(m):return {k:v for k,v in m.state_dict().items() if not k.startswith(('entity_head.','quality_head.'))}
def auxstate(m):return {k:v for k,v in m.state_dict().items() if k.startswith(('entity_head.','quality_head.'))}
def digest(state):
    d=hashlib.sha256()
    for k,v in sorted(state.items()):
        a=v.detach().cpu().contiguous().numpy();d.update(k.encode()+str(a.dtype).encode()+str(a.shape).encode()+a.tobytes())
    return d.hexdigest()
def optimizer(params):return torch.optim.AdamW(params,lr=1e-4,weight_decay=.001,foreach=False)
def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.manual_seed(1550);reference=OriginalHead().float()
    torch.manual_seed(1550);normal=IndependentAuxiliaryQueryHead().float()
    torch.manual_seed(1550);huge=IndependentAuxiliaryQueryHead().float()
    same(reference.state_dict(),visstate(normal));same(normal.state_dict(),huge.state_dict())
    assert sum(p.numel()for p in normal.parameters())==471043
    vp=normal.visibility_parameters();ap=normal.auxiliary_parameters()
    assert len(set(map(id,vp)))==len(vp) and len(set(map(id,ap)))==len(ap)
    assert not set(map(id,vp))&set(map(id,ap))
    assert set(map(id,vp+ap))==set(map(id,normal.parameters()))
    assert sum(p.numel()for p in vp)==224513
    assert sum(p.numel()for p in normal.entity_parameters())==24705
    assert sum(p.numel()for p in normal.quality_parameters())==221825
    # Loading a different invented warm visibility state cannot consume RNG or
    # change auxiliary initialization. The fresh/warm source control uses seed1550.
    torch.manual_seed(321);invented_warm=OriginalHead().float().state_dict()
    before_aux=copy.deepcopy(auxstate(normal));before_rng=torch.get_rng_state().clone()
    normal.load_visibility_state(invented_warm)
    same(auxstate(normal),before_aux);same(torch.get_rng_state(),before_rng)
    same(auxstate(normal),auxstate(huge));assert digest(visstate(normal))!=digest(visstate(huge))
    normal.load_visibility_state(reference.state_dict())
    invalid=dict(reference.state_dict());invalid['not_an_original_key']=torch.zeros(1)
    rejected=False
    try:normal.load_visibility_state(invalid)
    except ValueError:rejected=True
    assert rejected
    gen=torch.Generator().manual_seed(1551)
    x=torch.randn(16,1928,generator=gen);G=torch.zeros(16,42)
    first=normal(x,G);same(first['external_occlusion'],reference(x,G))
    assert set(first)=={'external_occlusion','target_entity','coarse_point_quality'}
    assert all(v.shape==(16,) for v in first.values())
    assert not first['target_entity'].any() and not first['coarse_point_quality'].any()
    labels=dict(external_occlusion=torch.tensor([0.,1.,0.,1.,0.,1.,-1.,-1.,0.,1.,0.,1.,-1.,-1.,0.,1.]),
                target_entity=torch.tensor([1.,1.,1.,1.,-1.,-1.,0.,0.,1.,0.,1.,0.,-1.,1.,0.,1.]),
                coarse_point_quality=torch.tensor([-1.,-1.,-1.,-1.,0.,1.,0.,1.,-1.,-1.,0.,1.,0.,1.,0.,1.]))
    available=torch.ones(16,dtype=torch.bool);available[14]=False
    training=torch.ones(16,dtype=torch.bool);training[15]=False
    # Raw-output gradients independently check S/U-like unknown, unavailable,
    # held rows. The first four D/E labels retain gradient with unknown quality.
    raw={k:torch.randn(16,generator=gen,requires_grad=True)for k in labels}
    sum(per_output_loss(raw,labels,available,training).values()).backward()
    for key,t in raw.items():
        eligible=available&training&((labels[key]==0)|(labels[key]==1))
        assert not t.grad[~eligible].any() and (t.grad[eligible].abs()>0).all()
    assert (raw['external_occlusion'].grad[:4].abs()>0).all()
    empty={k:torch.randn(4,generator=gen,requires_grad=True)for k in labels}
    loss=per_output_loss(empty,{k:torch.full((4,),-1.)for k in labels},torch.ones(4,dtype=torch.bool),torch.ones(4,dtype=torch.bool))
    sum(loss.values()).backward();assert all(not t.grad.any()for t in empty.values())
    # Separate optimizers and separate clipping make visibility invariant even
    # when arbitrary auxiliary gradients are made a million times larger.
    ropt=optimizer(reference.parameters());opts=[(optimizer(m.visibility_parameters()),optimizer(m.auxiliary_parameters()))for m in(normal,huge)]
    records=[]
    for step in range(1,5):
        x=torch.randn(16,1928,generator=gen);ropt.zero_grad(set_to_none=True)
        zr=reference(x,G);lr=per_output_loss({'external_occlusion':zr},{'external_occlusion':labels['external_occlusion']},available,training)['external_occlusion'];lr.backward()
        rgrad=[p.grad.detach().clone()for p in reference.parameters()]
        rnorm=torch.nn.utils.clip_grad_norm_(list(reference.parameters()),5)
        clipped=[p.grad.detach().clone()for p in reference.parameters()]
        ropt.step();norms=[];auxnorms=[]
        for i,(model,(vo,ao),scale) in enumerate(zip((normal,huge),opts,(1.,1.e6))):
            vo.zero_grad(set_to_none=True);ao.zero_grad(set_to_none=True)
            out=model(x,G);same(out['external_occlusion'],zr)
            terms=per_output_loss(out,labels,available,training)
            aux=terms['target_entity']+terms['coarse_point_quality']
            cross=torch.autograd.grad(aux,model.visibility_parameters(),retain_graph=True,allow_unused=True)
            assert all(v is None for v in cross)
            (terms['external_occlusion']+scale*aux).backward()
            for a,p in zip(rgrad,model.visibility_parameters()):same(a,p.grad)
            vn=torch.nn.utils.clip_grad_norm_(model.visibility_parameters(),5)
            an=torch.nn.utils.clip_grad_norm_(model.auxiliary_parameters(),5)
            assert torch.isfinite(vn) and torch.isfinite(an);same(vn,rnorm)
            for a,p in zip(clipped,model.visibility_parameters()):same(a,p.grad)
            vo.step();ao.step();same(reference.state_dict(),visstate(model));same(ropt.state_dict(),vo.state_dict())
            assert not model.geometry.weight.any();norms.append(float(vn));auxnorms.append(float(an))
        same(visstate(normal),visstate(huge));same(opts[0][0].state_dict(),opts[1][0].state_dict())
        records.append(dict(update=step,reference_visibility_norm=float(rnorm),normal_visibility_norm=norms[0],million_aux_visibility_norm=norms[1],normal_aux_norm=auxnorms[0],million_aux_norm=auxnorms[1],visibility_gradient_clip_params_and_AdamW_state_bitexact=True))
    assert any(a['million_aux_norm']>1.e4 for a in records)
    report=dict(completed=True,PASS=True,generated_CPU_invented_inputs_states_only=True,
        code=bind(H/'independent_query_head.py'),checker=bind(__file__),original_class_source=dict(path=str(ORIGINAL_SOURCE),sha256=ORIGINAL_SHA),
        parameters=dict(total=471043,visibility=224513,entity=24705,quality=221825),
        visibility_keys_expression_forward_gradients_and_updates_original_bitexact=True,
        disjoint_exhaustive_deterministic_parameter_lists=True,
        warm_fresh_seed1550_auxiliary_initialization_identical=True,
        load_visibility_state_leaves_aux_and_RNG_unchanged=True,
        neutral_aux_initial_raw_logits_zero=True,
        auxiliary_gradients_do_not_reach_visibility_parameters=True,
        millionfold_aux_loss_cannot_change_separately_clipped_visibility_norm_parameters_or_optimizer_state=True,
        independent_known_masks_no_quality_filter_on_visibility=True,
        unknown_unavailable_and_held_gradients_zero=True,
        no_xy_output=True,training_protocol_created=False,
        source_data_feature_or_checkpoint_bodies_read=0,private_or_V112_access=0,
        actual_source_model_forward_or_fit_calls=0,updates_on_invented_CPU_only=4,
        paired_step_evidence=records,final_visibility_state_sha256=digest(visstate(normal)))
    write(H/'GENERATED_INDEPENDENCE_CHECKS.json',report)
    print(json.dumps(dict(completed=True,PASS=True,checks=bind(H/'GENERATED_INDEPENDENCE_CHECKS.json'),parameters=report['parameters'])))
if __name__=='__main__':main()
