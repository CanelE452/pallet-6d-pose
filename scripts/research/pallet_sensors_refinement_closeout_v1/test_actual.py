"""Real cache descriptor parity and one disposable optimizer step."""
import numpy as np
import torch
from env import *
from direct_residual_control import DirectResidualControl,direct_loss,decode_direct,REF
from paired_stats_helper import _prepare,_median,paired_pooled_median
def run():
    verify();assert torch.cuda.is_available();g=gpu();assert not g['foreign_compute'],g
    if (DOC/'REGRESSION_TESTS.json').exists():
        r=read(DOC/'REGRESSION_TESTS.json');assert r['PASS'];print('EXISTING_SMOKE_VERIFIED');return
    config=read(B/'PARAMETER_BUDGET_LOCK.json')['config'];data=dataset();batch=data.batch(data.train_rows[:2])
    torch.manual_seed(1);p=REF.GenericPointRefiner(**config).cuda()
    torch.manual_seed(1);d=DirectResidualControl(**config).cuda()
    shared={k:v for k,v in p.state_dict().items() if k in d.state_dict()}
    assert all(torch.equal(v,d.state_dict()[k]) for k,v in shared.items())
    descriptors=[];hook=p.scorer[0].register_forward_pre_hook(lambda m,a:descriptors.append(a[0].detach().clone()))
    with torch.no_grad():
        po=fwd(p,batch)
        out=d(*(batch[k] for k in ('p3','p4','points','boxes','point_valid','input_shape')),lam=0,return_descriptors=True)
    hook.remove();assert torch.equal(out['candidate_descriptors'],descriptors[0])
    assert sum(t.numel() for t in d.parameters())==19450
    assert torch.equal(decode_direct(out,0),batch['points'])
    assert torch.equal(decode_direct(out,1,0),batch['points'])
    changed=decode_direct(out,4,1.5);assert torch.equal(changed[:,8],batch['points'][:,8])
    assert (torch.linalg.vector_norm(changed-batch['points'],dim=-1)<=1.5001).all()
    # Invalid corner and invalid box are passed through; valid finite sentinel input.
    bad={k:v.clone() for k,v in batch.items()};bad['point_valid'][:,0]=False
    b=fwd(d,bad);assert torch.equal(decode_direct(b,1)[:,0],bad['points'][:,0])
    bad['boxes'][:,2:]=bad['boxes'][:,:2]-1
    b=fwd(d,bad);assert torch.equal(decode_direct(b,1),bad['points'])
    output=fwd(d,batch);z=direct_loss(output,batch['gt_points'],torch.zeros_like(batch['gt_valid']));assert float(z)==0
    opt=torch.optim.AdamW(d.parameters(),lr=.001,betas=(.9,.999),weight_decay=.0001)
    before=torch.nn.utils.parameters_to_vector(d.parameters()).detach().clone()
    loss=direct_loss(output,batch['gt_points'],batch['gt_valid']);assert torch.isfinite(loss);loss.backward()
    gradients={k:float(v.grad.norm()) for k,v in d.named_parameters()};assert gradients['adapt3.0.weight']>0 and gradients['adapt4.0.weight']>0 and gradients['residual_head.2.weight']>0
    torch.nn.utils.clip_grad_norm_(d.parameters(),5,error_if_nonfinite=True);opt.step()
    change=float((torch.nn.utils.parameters_to_vector(d.parameters()).detach()-before).norm());assert change>0
    from point_inference import replace_selected
    pts=np.arange(18,dtype=np.float32).reshape(9,2);base=[dict(keypoints_xy=pts.astype(float),box_xyxy=[0,0,20,20],score=.9)]
    # decoder already scaled lambda: restoration must not multiply .25 a second time.
    refined=pts.copy();refined[:8]+=.25*4
    got=replace_selected(base,0,pts,refined,.5,.25)[0]['keypoints_xy']
    assert np.array_equal(got[:8],pts[:8]+2) and np.array_equal(got[8],pts[8])
    for gain in (.2,.5,1.3):
        test=dict(out);test['delta_normalized']=torch.ones_like(out['delta_normalized'])*10
        moved=decode_direct(test,.25,3*gain)-out['points_raw']
        assert float((moved[:,:8].norm(dim=-1)/gain).max())<=3.0002
    rng=np.random.default_rng(1)
    for _ in range(60):
        rows=[rng.normal(size=5),rng.normal(size=4),rng.normal(size=3)];counts=rng.integers(1,4,size=3)
        prepared=_prepare(rows,np.arange(3));expanded=np.concatenate([np.tile(row,n) for row,n in zip(rows,counts)])
        assert abs(_median(prepared,counts)-np.median(expanded))<1e-12
    rows=[np.array([1.,2.]),np.array([3.,4.])];stats=paired_pooled_median(rows,[[r-.5 for r in rows]]*3,['a','b'],resamples=100)
    assert stats['delta']==stats['low']==stats['high']==-.5
    try:paired_pooled_median(rows,[[np.array([1]),np.array([2])]],['a','b']);raise AssertionError('expected mismatch rejection')
    except ValueError:pass
    write(DOC/'REGRESSION_TESTS.json',dict(PASS=True,actual_cache_rows=data.train_rows[:2].tolist(),shared_initial_parameters_exact=True,shared_descriptors_exact=True,
        zero_lambda=True,zero_cap=True,center_invalid_box_point_preserved=True,cap_unit_parity=True,lambda_once=True,empty_supervision_zero=True,weighted_median_cases=60,
        support_mismatch_rejected=True,smoke_updates=1,smoke_exposures=2,smoke_reused_for_training=False,finite_gradient=gradients,post_step_parameter_norm=change,gpu=g,
        D_params=19450,P_params=18962,relative_params=19450/18962-1,source_sha256=sha(HERE/'direct_residual_control.py')))
    print('ACTUAL_CACHE_GATE_PASS',change,flush=True)
if __name__=='__main__':run()
