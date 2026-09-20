import torch
from .heatmap_candidate_headroom import top_five
from .heatmap_mode_decoder import local_mode


def test_candidates_gt_free_separated_first_is_fixed_decoder():
    q=torch.full((1,1,96,72),-100.)
    for i,(x,y) in enumerate([(10,10),(30,10),(50,10),(10,30),(30,30)]):q[0,0,y,x]=5.-i
    before=q.clone();a=top_five(q)
    torch.testing.assert_close(a['points'][0,0],torch.tensor([[40.,40.],[120.,40.],[200.,40.],[40.,120.],[120.,120.]]))
    torch.testing.assert_close(a['points'][:,:,0],local_mode(q))
    torch.testing.assert_close(a['mass'].sum(-1),torch.ones(1,1))
    torch.testing.assert_close(q,before,atol=0,rtol=0)
    anchors=a['anchors'][0,0]
    assert all((anchors[i]-anchors[j]).abs().max()>5 for i in range(5) for j in range(i))
