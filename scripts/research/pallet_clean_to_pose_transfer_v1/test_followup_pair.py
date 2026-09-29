"""반복2팔에서namespace/자원회계/입력짝계약을지키는가."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import pytest

from . import followup_pair as F


def test_stage_reads_global_ledger_and_namespaces_resource_event(monkeypatch):
    calls=[]
    monkeypatch.setattr(F.ROOT_C,'read',lambda p:calls.append(('read',p)) or {'totals':{'student_fits':4}})
    monkeypatch.setattr(F.ROOT_C,'resource',lambda event,*a,**kw:calls.append(('event',event,kw)))
    stage=F.StageContext('REPEAT_PRIMARY_S43')
    assert stage.read(stage.DOC/'RESOURCE_LEDGER.json')['totals']['student_fits']==4
    stage.resource('FIT_RAW',fits=1,updates=320)
    assert calls[0][1]==F.ROOT_C.DOC/'RESOURCE_LEDGER.json'
    assert calls[1][1]=='REPEAT_PRIMARY_S43::FIT_RAW'
    with pytest.raises(AssertionError):stage.save(stage.DOC/'RESOURCE_LEDGER.json',{})


def test_stage_path_rejects_escape():
    for name in ('../PRIMARY','PRIMARY','/tmp/x','lower_case'):
        with pytest.raises(AssertionError):F.StageContext(name)
    stage=F.StageContext('REPEAT_PRIMARY_S43')
    with pytest.raises(AssertionError):stage.save(F.ROOT_C.DOC/'PRIMARY_PROTOCOL.json',{})


def test_runtime_restores_frozen_engine_context(monkeypatch):
    previous=object();module=SimpleNamespace(C=previous)
    monkeypatch.setattr(F.StageContext,'read',lambda self,p:dict(followup_kind='SEED_REPEAT',masking={'schedule':.5}))
    with pytest.raises(RuntimeError):
        with F.runtime('REPEAT_PRIMARY_S43',(module,)):
            assert module.C is not previous
            raise RuntimeError('preserved failure')
    assert module.C is previous


def trace(coordinate):
    output=[]
    for index in range(8):
        real=dict(name='real.png',role='REAL',recording='R',plan={'bbox_fraction':.2},scheduled=True,applied=True,
            geometric_demotions=0,before_image='a',after_image='b',actual_covered=1,actual_remaining=7,reason='random_valid')
        real['plan'].update(scheduled=True,applied=True)
        source=dict(name='syn__x.png',role='SOURCE',recording='S',plan=None,scheduled=False,applied=False,
            geometric_demotions=0,before_image='s',after_image='s')
        output.append(dict(names=['real.png','syn__x.png'],images='rgb',before_images=['a','s'],after_images=['b','s'],
            boxes='box',support='mask',batch_idx='i',roles='role',coordinates=coordinate,transfer=[real,source]))
    return output


def test_pair_parity_rejects_RGB_or_support_change():
    raw,ref=trace('raw'),trace('ref')
    assert F.pair_trace_checks(raw,ref)['passed']
    ref[0]['support']='different'
    with pytest.raises(AssertionError):F.pair_trace_checks(raw,ref)
    ref=trace('ref');ref[0]['images']='different'
    with pytest.raises(AssertionError):F.pair_trace_checks(raw,ref)
