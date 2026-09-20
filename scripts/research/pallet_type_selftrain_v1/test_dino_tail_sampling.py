from collections import Counter
import numpy as np
import pytest
from . import dino_tail_sampling as T


def test_sampling_is_deterministic_and_exactly_balanced():
    pool=list(range(20));hard=[1,4,7];s=T.schedule(pool,hard,50);assert s==T.schedule(pool,hard,50)
    for a,rows in s.items():
        assert np.shape(rows)==(50,8) and set(np.asarray(rows).ravel())==set(pool)
    for r in s['TAIL']:assert set(r[:4])<=set(hard) and not set(r[4:])&set(hard)
    counts=Counter(np.asarray(s['UNIFORM']).ravel());assert max(counts.values())-min(counts.values())<=1
    assert Counter(np.asarray(s['TAIL'])[:,:4].ravel()).total()==200


def test_empty_or_invalid_strata_are_rejected():
    for pool,hard in [([1],[]),([1],[1]),([1,2],[3])]:
        with pytest.raises(ValueError):T.schedule(pool,hard,5)


def test_spatial_recovery_distinguishes_reindex_from_new_locations():
    q=np.zeros((9,2));gt=np.zeros((9,2));gt[0]=[50,0];gt[1]=[0,0]
    mask=np.zeros(9,bool);mask[:2]=True
    r=dict(row=1,points=q,valid=np.ones(9,bool),target=gt,target_valid=mask,matrix=np.eye(3))
    out=T.spatial_summary([r],[gt]);assert out['far']==1 and out['far_recovered']==1 and out['damaged']==0
    q[2]=[50,0];out=T.spatial_summary([r],[gt]);assert out['far']==0
