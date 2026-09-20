import numpy as np
from .camera_facing_contract_audit import observe,ORIGIN
from scripts.annotate.convert_to_camera_facing_v4 import compute_perm_v4


def fixture():
    p=np.array([[-2,-1,4],[2,-1,5],[2,1,5],[-2,1,4],[-1,-1,7],[3,-1,8],[3,1,8],[-1,1,7]],float)
    return p[:,:2]/p[:,2:]*500+[320,240]


def test_exact_legacy_algorithm_no_input_mutation_and_affine_invariance():
    q=fixture();before=q.copy();a=observe(q,np.ones(8,bool))
    assert a['legacy_perm']==compute_perm_v4(ORIGIN,q)
    b=observe(q*3+[20,-10],np.ones(8,bool))
    assert a['legacy_perm']==b['legacy_perm']
    for k in ['front_vs_rear_area','front_axis_advantage','min_LR_margin']:np.testing.assert_allclose(a[k],b[k],atol=1e-12)
    np.testing.assert_array_equal(q,before)


def test_incomplete_gt_not_filled_and_not_counted():
    q=fixture();v=np.ones(8,bool);v[3]=False
    assert observe(q,v)==dict(complete=False)
    q[1]=np.nan
    assert observe(q,np.ones(8,bool))==dict(complete=False)


def test_reordered_fixture_agrees_and_quarter_is_not_C2():
    q=fixture();p=compute_perm_v4(ORIGIN,q)[:8];x=q[p]
    a=observe(x,np.ones(8,bool));assert a['legacy_identity']
    quarter=[4,0,3,7,5,1,2,6]
    b=observe(x[quarter],np.ones(8,bool))
    assert not b['legacy_in_approved_C2'] and not b['legacy_same_axis']
