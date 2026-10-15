"""The O floor counts held-out frames and never fits held-out sessions."""
from copy import deepcopy
import pytest

from . import calibrate_bounded_o_localization_v4_21 as F


def samples():
    rows=[]
    for session,frame,errors in [('a','0',[100.,90.]),('a','1',[80.,60.]),
                                 ('b','0',[4.]),('b','1',[3.]),('b','2',[5.])]:
        for k,error in enumerate(errors):
            rows.append(dict(record_id=f'{session}:{frame}',session=session,
                native_corner=k,error_px=error,scale_px=1000.,normalized_error=error/1000.))
    return rows


def test_holdout_is_whole_frame_and_beta_denominator_is_frames():
    rows=samples();result=F.fit_event_floor(rows,8.)
    assert result['informative_frames']==5 and result['fit_sample_count']==7
    assert result['mismatch_frames']==1 and result['epsilon']==pytest.approx(2/7)
    first=result['frame_heldout_replays'][0]
    assert first['id']=='a:0' and len(first['points'])==2
    assert first['id'] not in first['training_ids']
    assert first['points'][0]['fold_radius_px']==80.
    assert first['points'][1]['fold_radius_px']==80.
    assert all(f['id']not in f['training_ids']for f in result['frame_heldout_replays'])


def test_logo_fit_never_uses_heldout_frame_or_error():
    rows=samples();expected=F.fit_event_floor(rows,8.,heldout_session='a')
    changed=deepcopy(rows)
    for r in changed:
        if r['session']=='a':
            r['error_px']=1e6;r['normalized_error']=1000.
    assert F.fit_event_floor(changed,8.,heldout_session='a')==expected
    assert expected['fit_ids']==['b:0','b:1','b:2']
    assert expected['epsilon']==pytest.approx(1/5)
    other=F.fit_event_floor(rows,8.,heldout_session='b')
    assert other['fit_ids']==['a:0','a:1'] and other['epsilon']==.5


def test_existing_pixel_floor_and_frame_any_miss_are_preserved():
    rows=samples();train=[r for r in rows if r['session']=='b']
    held=[r for r in rows if r['record_id']=='a:0']
    result=F._events(train,held,8.)
    assert len(result)==1 and result[0]['event'] is True
    assert all(p['fold_radius_px']==8. for p in result[0]['points'])
    assert not any(f['event']for f in F._events(train,train,8.))
    with pytest.raises(ValueError,match='At least two'):
        F.fit_event_floor(held,8.)


def test_source_inventory_rejects_noncal_duplicate_and_corrupt_scale():
    rows=samples();ids={r['record_id']for r in rows}
    model=dict(samples=rows,training_record_ids=sorted(ids))
    assert len(F._samples(model,ids))==7
    with pytest.raises(ValueError,match='Unique declared CAL'):
        F._samples(model,ids-{'a:0'})
    duplicate=deepcopy(model);duplicate['samples'][1]=deepcopy(duplicate['samples'][0])
    with pytest.raises(ValueError,match='Unique declared CAL'):
        F._samples(duplicate,ids)
    bad=deepcopy(model);bad['samples'][0]['normalized_error']=2.
    with pytest.raises(ValueError,match='nonnegative CAL error'):
        F._samples(bad,ids)


def test_runtime_parameters_require_new_model_schema_recipe_and_digest():
    model=dict(schema=F.SCHEMA,recipe=deepcopy(F.RECIPE),fit_partition='CAL',epsilon=2/7)
    model['model_digest_sha256']=F.SOURCE._digest(model)
    assert F.prior_parameters(model)==dict(epsilon=2/7,huber_scale=3.)
    bad=deepcopy(model);bad['epsilon']=.1
    with pytest.raises(ValueError,match='digest changed'):F.prior_parameters(bad)
    bad=deepcopy(model);bad['fit_partition']='TEST'
    with pytest.raises(ValueError,match='Explicit new CAL'):F.prior_parameters(bad)
