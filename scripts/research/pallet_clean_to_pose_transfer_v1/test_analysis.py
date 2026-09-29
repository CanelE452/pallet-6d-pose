"""미실행결과를만들지않고두축/후보상한을분리하는분석검사."""
from .analysis import delta, selector_evidence


def test_delta_signs_keep_p90_tradeoff():
    before=dict(conditional={'translation_cm':{'median':10.,'P90':20.},'rotation_deg':{'median':5.,'P90':9.}},valid_pose=99,frames=99,twoD={'PCK':{'10':.5}},ADDsym_AUC=.2)
    after=dict(conditional={'translation_cm':{'median':9.,'P90':25.},'rotation_deg':{'median':4.,'P90':10.}},valid_pose=99,frames=99,twoD={'PCK':{'10':.6}},ADDsym_AUC=.3)
    result=delta(before,after)
    assert result['translation_cm']=={'median':-1.,'P90':5.}
    assert result['rotation_deg']=={'median':-1.,'P90':1.}


def test_missing_oracle_is_pending_not_failure_or_fit():
    assert selector_evidence({},None)==dict(status='PENDING_CANDIDATE_ORACLE',fit_selected=False)
