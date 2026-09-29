import pytest
from .repeat_audit import direction_counts,close,Verifier


def test_directions_separate_ties_and_invalid():
    before={};after={}
    for i,(t,r) in enumerate((t,r) for t in (-1,0,1) for r in (-1,0,1)):
        before[str(i)]=dict(available=True,translation_cm=3.,rotation_deg=3.)
        after[str(i)]=dict(available=True,translation_cm=3.+t,rotation_deg=3.+r)
    before['bad']=dict(available=False);after['bad']=dict(available=False)
    result=direction_counts(before,after,list(before))
    assert len(result)==9 and all(v==1 for v in result.values())


def test_close_rejects_number_or_denominator_change():
    close({'frames':99,'error':1.},{'frames':99,'error':1.+1e-9})
    with pytest.raises(AssertionError):close({'frames':99},{'frames':98})
    with pytest.raises(AssertionError):close({'error':1.},{'error':1.1})


def test_verifier_deduplicates_equal_bindings(monkeypatch,tmp_path):
    from . import repeat_audit as A
    file=tmp_path/'x';file.write_text('test')
    calls=[];monkeypatch.setattr(A.C,'verify',lambda row:calls.append(row))
    verifier=Verifier();row=dict(path=str(file),sha256='x',bytes=4)
    verifier.nested({'one':row,'two':[row]})
    assert len(calls)==len(verifier.seen)==1
