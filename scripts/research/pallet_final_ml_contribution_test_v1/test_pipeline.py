import numpy as np
import torch
from common import old
from generic_point_refiner import GenericPointRefiner,decode,targets
from select_point import summarize9
import statistics_and_mechanism as stats

def test_weighted_median_exact_repeated_data():
    values=np.array([1.,9.,3.,2.]);weights=np.array([[1,2,3,1],[0,1,0,3]])
    for q in (.5,.9):
        expected=[np.quantile(np.repeat(values,w),q) for w in weights]
        assert np.allclose(old('aggregate_results').weighted_quantile(values,weights,q),expected)

def test_same_seed_paired_estimand():
    # Every L-P median contrast is a constant, unlike an incorrectly pooled
    # or independently resampled seed comparison.
    left=[];right=[]
    for seed,shift in enumerate((1.,3.,8.)):
        base={str(i):dict(session_id=str(i),errors=np.array([float(i+seed*30)])) for i in range(13)}
        right.append(base);left.append({k:{**v,'errors':v['errors']-shift} for k,v in base.items()})
    previous=stats.BOOT;stats.BOOT=256
    try:r=stats.paired(left,right,'keypoint_location_median_px')
    finally:stats.BOOT=previous
    assert r['difference']==-4 and r['paired_sessions']==13
    for s in ('frame_level','session_cluster'):
        assert np.isclose(r[s]['low'],-4) and np.isclose(r[s]['high'],-4)

def test_synthetic_coverage_denominator():
    r=dict(errors9_px=[1.,2.],gt9=4,available9=2,move8_px=[0.]*8,residual_in_bank=[True,False],mean_null_probability=.2)
    s=summarize9([r]);assert s['coverage']==.5 and s['pck10_all_gt']==.5 and s['candidate_target_radius_coverage']==.5

def test_expected_vote_no_gt_path():
    head=GenericPointRefiner(c3=1,c4=1);p=torch.zeros(1,9,2)
    logits=torch.full((1,8,222),-1000.);logits[:,:,16]=0
    output=dict(points_raw=p,point_support=torch.ones(1,8,dtype=torch.bool),logits=logits,candidate_displacements=head.displacements[None]*100,box_diagonal=torch.tensor([100.]))
    q=decode(output,lam=1);assert torch.allclose(q[0,:8],torch.tensor([8.,0.]).expand(8,2),atol=1e-6)
    target=targets(output,q,torch.ones(1,9,dtype=torch.bool));assert (target['distribution'].argmax(-1)==16).all()
    logits.fill_(-1000);logits[:,:,-1]=0
    assert torch.equal(decode(output),p)
