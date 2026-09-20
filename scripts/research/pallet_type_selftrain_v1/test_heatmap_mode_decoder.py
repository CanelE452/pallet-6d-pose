import numpy as np
import pytest
import torch
from .heatmap_mode_decoder import local_mode,ambiguity


def test_two_separated_peaks_do_not_average_in_empty_middle():
    q=torch.full((1,1,96,72),-100.)
    q[0,0,30,10]=np.log(.6);q[0,0,30,60]=np.log(.4)
    a=ambiguity(q)
    torch.testing.assert_close(a['mean'],torch.tensor([[[120.,120.]]]))
    torch.testing.assert_close(a['mode'],torch.tensor([[[40.,120.]]]))
    torch.testing.assert_close(a['local_mass'],torch.tensor([[.6]]))


@pytest.mark.parametrize('x,y',[(0,0),(71,95),(35,47)])
def test_exact_peak_and_boundaries(x,y):
    q=torch.full((1,1,96,72),-100.)
    q[0,0,y,x]=0
    torch.testing.assert_close(local_mode(q),torch.tensor([[[4.*x,4.*y]]]))


def test_subcell_average_preserved_inside_mode():
    q=torch.full((1,1,96,72),-100.)
    q[0,0,20,10]=np.log(.75);q[0,0,20,11]=np.log(.25)
    torch.testing.assert_close(local_mode(q),torch.tensor([[[41.,80.]]]))


def test_channel_permutation_and_constant_shift_and_input_immutable():
    q=torch.randn(2,9,96,72,generator=torch.Generator().manual_seed(1))
    before=q.clone();perms=torch.tensor([5,4,7,6,1,0,3,2,8])
    torch.testing.assert_close(local_mode(q[:,perms]),local_mode(q)[:,perms])
    torch.testing.assert_close(local_mode(q+40),local_mode(q),atol=1e-4,rtol=1e-5)
    torch.testing.assert_close(q,before,atol=0,rtol=0)


def test_invalid_heatmap_rejected():
    with pytest.raises(ValueError):local_mode(torch.zeros(1,9,95,72))
    with pytest.raises(ValueError):local_mode(torch.full((1,9,96,72),torch.nan))
