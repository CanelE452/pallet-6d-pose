import importlib.util
from pathlib import Path
import numpy as np
import pytest
spec=importlib.util.spec_from_file_location('al_pilot',Path(__file__).with_name('run.py'))
al=importlib.util.module_from_spec(spec);spec.loader.exec_module(al)

def rows(n):return [dict(capture_session='s'+str(i%3),timestamp_ns=i*3_000_000_000,image_sha256=str(i)) for i in range(n)]

def test_overlap_and_dedup_exclusion():
    rr=rows(6)+[rows(6)[5]]
    keep,bad=al.eligible(rr,{'0','1'},{'2'})
    assert [r['image_sha256'] for r in keep]==['3','4','5']
    assert len(bad)==4

@pytest.mark.parametrize('method',al.METHODS)
def test_selection_deterministic_unique_budget_and_temporal_gap(method):
    rr=rows(40);rr[1]['capture_session']=rr[0]['capture_session'];rr[1]['timestamp_ns']=rr[0]['timestamp_ns']+1
    x=np.random.default_rng(5).normal(size=(40,6));u=np.linspace(0,1,40)
    a=al.select(x,u,rr,10,method);b=al.select(x,u,rr,10,method)
    assert a==b and len(set(a))==10
    for i in a:
        for j in a:
            if i!=j and rr[i]['capture_session']==rr[j]['capture_session']:
                assert abs(rr[i]['timestamp_ns']-rr[j]['timestamp_ns'])>=2_000_000_000

def test_equal_instability_reduces_to_diversity():
    x=np.random.default_rng(7).normal(size=(40,6));u=np.ones(40);rr=rows(40)
    assert al.select(x,u,rr,10,'diversity')==al.select(x,u,rr,10,'geometry_weighted_diversity')

def test_weights_bounded_and_tie_symmetric():
    w=1+al.midranks([0,0,1,2]);assert w[0]==w[1] and np.all((w>=1)&(w<=2))

def test_labels_cannot_influence_selection():
    x=np.random.default_rng(8).normal(size=(40,6));u=np.arange(40);a=rows(40)
    b=[dict(**r,annotation={'arbitrary_GT':i}) for i,r in enumerate(a)]
    for m in al.METHODS:assert al.select(x,u,a,10,m)==al.select(x,u,b,10,m)

def test_nonfinite_signal_rejected():
    with pytest.raises(AssertionError):al.select(np.ones((3,2)),[1,np.nan,2],rows(3),2,'diversity')

def test_constraints_never_silently_relaxed():
    rr=[dict(capture_session='same',timestamp_ns=0) for _ in range(5)]
    assert len(al.select(np.eye(5),np.ones(5),rr,3,'random'))==1
