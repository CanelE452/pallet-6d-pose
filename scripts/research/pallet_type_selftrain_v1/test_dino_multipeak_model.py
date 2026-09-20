import pytest
import torch
from . import dino_multipeak_model as M


def test_first_mode_matches_frozen_decoder_exactly():
    torch.manual_seed(1);z=torch.randn(2,9,192,144)
    torch.testing.assert_close(M.modes(z)['points'][:,:,0],M.W.decode(z),atol=0,rtol=0)


def test_three_separate_peaks_ranked_and_exhausted():
    y,x=torch.meshgrid(torch.arange(192),torch.arange(144),indexing='ij')
    z=torch.stack([10-((x-20)**2+(y-30)**2).float(),8-((x-90)**2+(y-130)**2).float(),6-((x-100)**2+(y-20)**2).float()]).max(0).values[None,None]
    r=M.modes(z)
    torch.testing.assert_close(r['points'][0,0,:3],torch.tensor([[80.,120.],[360.,520.],[400.,80.]]),atol=1e-4,rtol=0)
    assert r['valid'][0,0].tolist()==[True,True,True,False,False]
    assert r['mass'][0,0,0]>r['mass'][0,0,1]>r['mass'][0,0,2]>0
    assert r['mass'][0,0,3:].sum()==0


def test_adjacent_local_maximum_not_counted_twice():
    z=torch.full((1,1,192,144),-100.)
    z[0,0,50,50]=10;z[0,0,50,55]=9;z[0,0,80,80]=8
    r=M.modes(z,count=2)
    assert r['anchors'][0,0].tolist()==[[50,50],[80,80]]


def test_border_peak_and_mass_bounds():
    z=torch.full((1,1,192,144),-50.);z[0,0,0,0]=10;z[0,0,-1,-1]=9
    r=M.modes(z,count=2)
    assert r['anchors'][0,0].tolist()==[[0,0],[143,191]]
    assert (r['mass']>=0).all() and (r['mass'].sum(-1)<=1.000001).all()


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):M.modes(torch.zeros(1,1,12,12))
    with pytest.raises(ValueError):M.modes(torch.full((1,1,192,144),float('nan')))
    with pytest.raises(ValueError):M.modes(torch.zeros(1,1,192,144),count=6)
