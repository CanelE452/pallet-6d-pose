"""Mapping/gate/render contracts only; no actual scores or journals are read."""
from copy import deepcopy

import pytest

from . import build_repair_campaign_progress_v9_v26 as B


def summary(count):
    return dict(count=count,translation_le5=count-1,successes_5cm_5deg=count-2,
        translation_cm=dict(mean=2.,max=9.),rotation_deg=dict(mean=3.,max=8.),
        rotation_unresolved=0,fallback_count=1,max_translation_id='synthetic<maximum>')


def method(name):
    return dict(method=name,label_KO='synthetic '+name,current_default=False,
        original_TEST49_all_translation_le5=False,numerical_primary192_target_met=False,
        populations={part:summary(count) for part,count in {**B.P.COUNTS,'PRIMARY192':192}.items()})


def test_noP_TP_and_containment_profiles_have_exact_zero_or_one_weights():
    profiles=B._profile_contracts()
    assert list(profiles)==list(B.V.ARMS[1:])
    for arm in B.V.ARMS[1:3]:
        assert profiles[arm]['coarse_prior_schema_mode']=='FULL'
        assert profiles[arm]['coarse_prior_weight']==0.
        assert profiles[arm]['noP_FULL_schema_is_zero_weight_placeholder'] is True
        assert profiles[arm]['active_rotation_prior'] is False
    for arm in B.V.ARMS[3:]:
        assert profiles[arm]['coarse_prior_schema_mode']=='TRANSLATION'
        assert profiles[arm]['coarse_prior_weight']==1.
        assert profiles[arm]['new_CAL_edge_noise_fits']==1
        assert profiles[arm]['shared_v24_TCAL_gamma'] is True
    assert [profiles[a]['containment_weight'] for a in B.V.ARMS[1:]]==[0.,0.,0.,0.,1.,1.]
    assert [profiles[a]['e_predicted_family_constraint'] for a in B.V.ARMS[1:]]==[False,True,False,True,False,True]
    assert all(not p['old_symmetric_mask_cost_or_numeric_tie_break_used'] for p in profiles.values())


def test_all_eight_pairs_are_matched_and_multichanges_are_disclosed():
    pairs=B._paired_definitions()
    assert len(pairs)==8
    expected=[(B.SOURCE24.ARMS[3],B.V.ARMS[1]),(B.SOURCE25.ARMS[1],B.V.ARMS[3]),
        (B.SOURCE25.ARMS[3],B.V.ARMS[5]),(B.V.ARMS[1],B.V.ARMS[3]),
        (B.V.ARMS[3],B.V.ARMS[5]),(B.V.ARMS[1],B.V.ARMS[2]),
        (B.V.ARMS[3],B.V.ARMS[4]),(B.V.ARMS[5],B.V.ARMS[6])]
    assert [(p['before'],p['after'])for p in pairs]==expected
    assert '단일 요소 효과 아님' in pairs[0]['interpretation']
    assert 'CAL noise' in pairs[1]['interpretation'] and 'CAL noise' in pairs[2]['interpretation']
    assert 'endpoint는 달라질' in pairs[3]['interpretation']
    assert all(p['intervention']=='PREDICTED_E_FAMILY_CONSTRAINT' for p in pairs[5:])


def test_previous23_unchanged_and_new_six_no_default_adoption():
    previous=dict(method_count=23,methods=[method('old_'+str(i))for i in range(23)])
    methods=deepcopy(previous['methods'])+[method(a)for a in B.V.ARMS[1:]]
    B._same_previous_methods(previous,methods)
    methods[0]['label_KO']='changed historical recipe'
    with pytest.raises(ValueError,match='previous23'):B._same_previous_methods(previous,methods)
    methods=deepcopy(previous['methods'])+[method(a)for a in B.V.ARMS[1:]]
    methods[-1]['current_default']=True
    with pytest.raises(ValueError,match='default adoption'):B._same_previous_methods(previous,methods)


def test_fixed_reference_path_and_sha_match_before_any_pairing():
    before={part:dict(reference_binding=dict(path='same/'+part,sha256=part))for part in B.P.COUNTS}
    after=deepcopy(before)
    B._require_same_reference_bindings(before,after)
    after['REMAINDER']['reference_binding']['sha256']='changed derived reference'
    with pytest.raises(ValueError,match='REMAINDER'):B._require_same_reference_bindings(before,after)


def test_input_byte_identity_never_accepts_only_equal_row_counts(tmp_path,monkeypatch):
    current,previous=tmp_path/'new',tmp_path/'old'
    bindings={str(current/'INPUT_SEAL.json'):dict(rows=238,input_sha256='same'),
              str(previous/'INPUT_SEAL.json'):dict(rows=238,input_sha256='same')}
    digests={str(current/'INPUTS.jsonl.gz'):'same',str(previous/'INPUTS.jsonl.gz'):'same'}
    monkeypatch.setattr(B.C,'read',lambda p:bindings[str(p)])
    monkeypatch.setattr(B.C,'sha',lambda p:digests[str(p)])
    B._require_same_input_bytes(dict(input_bytes_identical_to_v25=True),current,previous)
    digests[str(current/'INPUTS.jsonl.gz')]='modified'
    with pytest.raises(ValueError,match='Byte-identical'):
        B._require_same_input_bytes(dict(input_bytes_identical_to_v25=True),current,previous)


def test_missing_GO_rejects_before_score_or_prediction_access(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('Actual file/metric/journal access before GO')
    monkeypatch.setattr(B.C,'read',forbidden)
    monkeypatch.setattr(B.C,'sha',forbidden)
    monkeypatch.setattr(B.P,'_load_run',forbidden)
    with pytest.raises(ValueError,match='root accounting GO'):B.build()


def test_exact116_HTML_rows_include_metrics_maxid_fallback_and49_preservation():
    methods=[method('old_'+str(i))for i in range(23)]+[method(a)for a in B.V.ARMS[1:]]
    rendered=B._render(dict(methods=methods))
    assert rendered.count('<tr>')+rendered.count('<tr class="new">')==120 # 116 methods + four headers.
    assert rendered.count('<tr class="new">')==24
    assert 'synthetic&lt;maximum&gt;' in rendered and 'synthetic<maximum>' not in rendered
    for text in ('최대 T 사례','평균 R(°)','최대 R(°)','R 미확정','fallback','기존49 보존',
                 'FULL schema는 0 가중치 placeholder','단일 요인 실험이 아닙니다','PROGRESS_V9_V26.json'):
        assert text in rendered
