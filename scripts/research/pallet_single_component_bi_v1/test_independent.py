"""Source/cache and all-failure arithmetic contracts, no model execution."""
from copy import deepcopy
import json
import pytest
from . import independent as I


def test_predictor_field_reader_skips_unselected_scores(monkeypatch):
    text='{"id":"f1", "reference_distance_m":123456.789, "pose":{"rotation_error":987654.321}, "qFinal":[[1,2]], "method":"N3"}'
    decoded=[];loads=json.loads
    def trace(value):
        decoded.append(value)
        return loads(value)
    monkeypatch.setattr(I.json,'loads',trace)
    out=I.selected_fields(text,{'id','method','qFinal'})
    assert out==dict(id='f1',qFinal=[[1,2]],method='N3')
    assert not any('123456' in s or '987654' in s for s in decoded)
    with pytest.raises(ValueError,match='Duplicate'):
        I.selected_fields('{"id":"f1","id":"f2"}',{'id'})


def test_recount_retains_no_pose_and_paired_damage_denominators():
    rows=[]
    for i in range(4):
        metrics={a:dict(pose_available=True,translation_cm=1.,rotation_deg=1.,ADDsym_cm=1.,success_le5=True,success_strict_lt5=True) for a in I.ARMS}
        statuses={a:dict(accepted=None,fallback=None,status='baseline',reason=None) for a in I.ARMS}
        if i==0:
            metrics['N3_BI']=dict(pose_available=False,translation_cm=None,rotation_deg=None,ADDsym_cm=None,success_le5=False,success_strict_lt5=False)
        rows.append(dict(id=str(i),session='s'+str(i//2),cluster_id='c'+str(i//2),metrics=metrics,prediction_status=statuses))
    out=I.recount(rows,[('N3_BI','N3_F')]);a=out['arms']['N3_BI'];c=out['comparisons']['N3_BI-N3_F']
    assert a['denominator']==4 and a['successes_le5']==3 and a['no_pose_n']==1
    assert a['errors']['translation_cm']['n']==3 and a['errors']['translation_cm']['missing']==1
    assert c['success_rate']['full_denominator']==4 and c['success_rate']['mean_difference']==-.25
    assert c['success_rate']['losses']==1 and c['success_rate']['gains']==0
    assert c['transition']['ids']['success_to_failure']==['0']
