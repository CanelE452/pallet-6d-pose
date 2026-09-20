from collections import Counter
import pytest
from .recovery_real_only import real_slots


def test_exact_real_membership_and_doubled_exposure():
    old=[f'/source/syn__{i}.png' for i in range(512)]+[f'/real/PLASTIC__{i%217}.png' for i in range(512)]
    new=real_slots(old)
    assert len(new)==1024 and len(set(new))==217
    assert Counter(new)==Counter({k:2*v for k,v in Counter(old[512:]).items()})
    assert not set(new)&set(old[:512])


def test_reject_accidental_other_population():
    old=[f'/source/syn__{i}.png' for i in range(512)]+[f'/real/GREEN__{i%217}.png' for i in range(512)]
    with pytest.raises(AssertionError):real_slots(old)
