"""Analytic and contract tests; no dataset labels, images or inference."""
from copy import deepcopy
import hashlib
import numpy as np
import pytest

from scripts.research.pallet_minimal_bi_ablation_v1 import statistics as T
from scripts.research.pallet_minimal_bi_ablation_v1 import scoring as S
from scripts.research.pallet_minimal_bi_ablation_v1 import diagnostics as D
from scripts.research.pallet_minimal_bi_ablation_v1 import reportmaker as R


def scored_row(i, cluster='a', values=None):
    values = values or {a: bool(i%2) for a in T.ARMS}
    metrics = {a:dict(pose_available=True, translation_cm=float(i+1), rotation_deg=1., ADDsym_cm=1.,
               success_le5=values[a], success_strict_lt5=values[a]) for a in T.ARMS}
    return dict(id=str(i), partition='TEST', session='s'+cluster, cluster_id=cluster, metrics=metrics,
        prediction_status={a:dict(accepted=None if a in ('R0','N0') else True,
            fallback=None if a in ('R0','N0') else False, status='AVAILABLE',reason=None) for a in T.ARMS})


def test_distribution_keeps_unavailable_and_sample_sd():
    x = T.distribution([1.,3.,None])
    assert (x['n'],x['denominator'],x['missing']) == (2,3,1)
    assert x['mean'] == 2 and x['sample_std'] == pytest.approx(np.sqrt(2)) and x['max'] == 3
    assert T.distribution([])['mean'] is None and T.distribution([1])['sample_std'] is None
    with pytest.raises(ValueError): T.distribution([1,np.inf])


def test_bootstrap_original_frame_weight_not_equal_cluster_weight():
    rows = [scored_row(0,'a')]+[scored_row(i,'b') for i in range(1,4)]
    b = T.PairedClusterBootstrap(rows)
    result = b.contrast([1.,0.,0.,0.])
    assert result['mean_difference'] == .25  # Equal-cluster would incorrectly give .5.
    draws = np.random.default_rng(20260917).multinomial(2,[.5,.5],size=10000).astype('uint16')
    expected = draws[:,0]/(draws[:,0]+3*draws[:,1])
    np.testing.assert_array_equal(b.draws, draws)
    assert result['CI95'] == np.quantile(expected,[.025,.975]).tolist()
    assert b.metadata()['draw_sha256'] == hashlib.sha256(draws.tobytes()).hexdigest()


def test_common_availability_bootstrap_reports_empty_resamples():
    b = T.PairedClusterBootstrap([scored_row(0,'a'),scored_row(1,'b')])
    x = b.contrast([2.,None])
    assert x['common_available_n']==1 and x['full_denominator']==2
    assert x['empty_resamples']>0 and x['valid_resamples']+x['empty_resamples']==10000
    assert x['CI95'] is None
    assert b.contrast([None,None])['mean_difference'] is None


def test_single_cluster_has_no_inferential_interval():
    x = T.PairedClusterBootstrap([scored_row(0),scored_row(1)]).contrast([1.,2.])
    assert x['mean_difference']==1.5 and x['CI95'] is None


def test_shared_draws_reproducible_and_pair_order_preserved():
    rows=[scored_row(0,'a'),scored_row(1,'b')]
    a,b=T.PairedClusterBootstrap(rows),T.PairedClusterBootstrap(rows)
    assert a.metadata()['draw_sha256']==b.metadata()['draw_sha256']
    assert a.contrast([1.,-2.])==b.contrast([1.,-2.])
    with pytest.raises(ValueError): a.contrast([1.])
    with pytest.raises(ValueError): T.PairedClusterBootstrap([rows[0],rows[0]])


def test_four_transitions_and_p_not_derived_from_CI():
    pairs=[(True,False),(False,True),(True,True),(False,False)]
    rows=[scored_row(i,str(i%2),dict(R0=b,R1=a,N0=b,N1=a)) for i,(a,b) in enumerate(pairs)]
    r=T.summarize_population(rows)
    v=r['comparisons']['R1-R0']
    assert [v['transition']['counts'][k] for k in ('failure_to_success','success_to_failure','both_success','both_failure')]==[1]*4
    assert v['success_rate']['mean_difference']==0.
    assert v['p_raw'] is None and v['p_Holm'] is None and v['p_value_reason']
    assert len(r['Holm_family']['hypotheses'])==4


def test_no_pose_stays_failure_and_baseline_acceptance_NA():
    rows=[scored_row(0,'a'),scored_row(1,'b')]
    rows[0]['metrics']['R1'].update(pose_available=False,translation_cm=None,rotation_deg=None,ADDsym_cm=None,
                                  success_le5=False,success_strict_lt5=False)
    rows[0]['prediction_status']['R1'].update(accepted=False,fallback=False,status='NO_POSE')
    r=T.summarize_population(rows)
    assert r['arms']['R1']['denominator']==2 and r['arms']['R1']['no_pose_n']==1
    assert r['arms']['R1']['errors']['translation_cm']['n']==1
    assert r['common_all_four_available_n']['translation_cm']==1
    assert r['arms']['R0']['accepted_status']=='NA' and r['arms']['R0']['accepted_available_n']==0
    assert r['comparisons']['R1-R0']['continuous']['translation_cm']['common_available_n']==1


def test_unresolved_reference_not_fabricated_into_known_accuracy():
    rows=[scored_row(0,'a'),scored_row(1,'b')]
    rows[0]['metrics']['R1'].update(success_le5=None,success_strict_lt5=None,rotation_deg=None)
    r=T.summarize_population(rows)
    assert r['arms']['R1']['reference_unresolved_n']==1
    assert r['arms']['R1']['success_rate_is_confirmed_lower_bound']
    assert r['comparisons']['R1-R0']['transition']['counts']['reference_unresolved']==1
    assert r['comparisons']['R1-R0']['success_rate']['CI95'] is None


def test_inclusive_five_boundary_and_historical_strict_separate(monkeypatch):
    from scripts.research.pallet_visibility_curriculum_20261013_v1 import scoring as OLD
    monkeypatch.setattr(OLD,'pose_metric',lambda *args:dict(translation_cm=5.,rotation_deg=5.,ADDsym_cm=1.))
    x=S.metric_for_pose(dict(available=True),dict(R=np.eye(3).tolist()))
    assert x['success_le5'] is True and x['success_strict_lt5'] is False
    rows=[scored_row(0,'a'),scored_row(1,'b')]
    rows[0]['metrics']['R1']=x
    y=T.summarize_population(rows)['arms']['R1']
    assert y['translation_exactly_5_n']==y['rotation_exactly_5_n']==1
    assert y['strict_vs_inclusive_changed_ids']==['0']


def test_unavailable_pose_never_reads_numeric_payload():
    class Poison(dict):
        def __getitem__(self,key):
            if key!='available': raise AssertionError('Unavailable coordinates read')
            return False
        def get(self,key,default=None): return False if key=='available' else default
    e=dict(actual_pose=Poison(available=False),accepted=False,fallback=False,status='NO_POSE')
    assert S.normalize_entry(e,'R1')['actual_pose']['available'] is False
    assert not S.metric_for_pose(Poison(available=False),dict(R=np.eye(3).tolist()))['success_le5']


def input_pair():
    rows=[];frames=[];ids=[]
    for p,n in S.PARTITIONS.items():
        for j in range(n):
            fid=f'{p}:{j}';ids.append(fid)
            f=dict(id=fid,partition=p,session='toy',cluster_id='toy_cluster')
            rows.append(dict(**f,predictions={a:dict(actual_pose=dict(available=True,R_physical=np.eye(3).tolist(),centroid=[0,0,1]),
                accepted=None if a in ('R0','N0') else True,fallback=None if a in ('R0','N0') else False,
                status='AVAILABLE',reason=None) for a in S.ARMS}));
            f['arms']={b:dict(coarse_pose=deepcopy(rows[-1]['predictions'][a]['actual_pose']))
                       for a,b in (('R0','RAW_YOLO'),('N0','N3_SUBPIX'))}
            frames.append(f)
    return rows,frames,ids


def test_full238x4_identity_no_drop_and_inputs_immutable():
    rows,frames,ids=input_pair();before=deepcopy(rows)
    assert len(S.compact_predictions(rows,frames,ids))==238 and rows==before
    rows[5]['predictions'].pop('N1')
    with pytest.raises(ValueError): S.compact_predictions(rows,frames,ids)
    with pytest.raises(ValueError): S.compact_predictions(before[:-1],frames,ids)


def test_wrong_basis_and_accepted_no_pose_rejected():
    row=input_pair()[0][0]['predictions']['R1']
    row['actual_pose'].update(R_cf=np.eye(3).tolist(),Q=np.diag([-1,1,-1]).tolist())
    with pytest.raises(ValueError): S.normalize_entry(row,'R1')
    row['actual_pose']=dict(available=False)
    with pytest.raises(ValueError): S.normalize_entry(row,'R1')


def test_baseline_pose_and_missing_acceptance_cannot_be_silently_changed():
    rows,frames,ids=input_pair()
    rows[1]['predictions']['R0']['actual_pose']['centroid'][0]=.1
    with pytest.raises(ValueError,match='baseline'):S.compact_predictions(rows,frames,ids)
    rows,frames,ids=input_pair();rows[1]['predictions']['N0']['accepted']=True
    with pytest.raises(ValueError,match='baseline'):S.compact_predictions(rows,frames,ids)


def test_current_source_and_input_byte_closure(tmp_path):
    source=tmp_path/'legacy.py';source.write_text('original source')
    inp=tmp_path/'cache.json';inp.write_text('opaque bytes')
    model=tmp_path/'model.json';model.write_text('descriptor only')
    proto=dict(source_sha256={},legacy_source_sha256={str(source):S.sha(source)},
               model_descriptors={'m':dict(path='model.json',sha256=S.sha(model))})
    seal=dict(source_sha256={},input_sha256={'cache.json':S.sha(inp)})
    assert S.verify_current_bindings(proto,seal,tmp_path)==3
    inp.write_text('changed')
    with pytest.raises(ValueError,match='changed'):S.verify_current_bindings(proto,seal,tmp_path)


def test_score_GO_and_full_prediction_gate_precede_reference_IO(monkeypatch,tmp_path):
    def poison(*args,**kwargs):raise AssertionError('IO before GO')
    monkeypatch.setattr(S,'sha',poison)
    args=dict(folder=tmp_path,input_dir=tmp_path,source_root=tmp_path,reference_manifest=tmp_path/'m',
              expected_reference_manifest_sha256='a'*64,expected_prediction_seal_sha256='b'*64,output=tmp_path/'out')
    with pytest.raises(PermissionError): S.score(**args)
    with pytest.raises(ValueError,match='SHA256'):S.prediction_gate(folder=tmp_path,input_dir=tmp_path,expected_prediction_seal_sha256=None)
    monkeypatch.setattr(S,'prediction_gate',lambda **kwargs:(_ for _ in ()).throw(ValueError('incomplete predictions')))
    monkeypatch.setattr(S,'read_fixed_references',poison)
    with pytest.raises(ValueError,match='incomplete predictions'):S.score(**args,root_scoring_GO=True)


def test_diagnostic_BI_gating_fallback_and_real_motion():
    class Poison:
        def __array__(self,*args):raise AssertionError('Excluded coordinates read')
    q=[Poison() for _ in range(9)];q[1]=[5.,6.];q[2]=[8.,9.]
    obs=dict(states=['S','B','I','O','T','U','S','U'],
        visible_correction=dict(records=[dict(native_corner=k,pixel_queried=k in (1,2),corrected=k==2,reason='saved') for k in range(8)]),
        frozen_visible_measurements=[dict(native_corner=1,xy=[5,6],covariance_px2=np.eye(2).tolist(),source='VISIBLE_BACKBONE_FALLBACK'),
                                    dict(native_corner=2,xy=[8+2.842e-14,9],covariance_px2=np.eye(2).tolist(),source='I_INTERNAL_RGB_JUNCTION')])
    x=D.observation_summary(obs,q)
    assert x['new_intersection_n']==1 and x['retained_backbone_n']==1 and x['moved_n']==0
    obs['frozen_visible_measurements'][1]['xy']=[9,9]
    assert D.observation_summary(obs,q)['moved_n']==1


def test_diagnostic_excluded_query_or_bad_covariance_fails():
    obs=dict(states=['S']*8,visible_correction=dict(records=[dict(native_corner=0,pixel_queried=True)]),frozen_visible_measurements=[])
    with pytest.raises(ValueError):D.observation_summary(obs,[])


def test_gallery_categories_fixedID_not_best_error_and_absences_explicit():
    rows=[scored_row(10,'b',dict(R0=False,R1=True,N0=False,N1=True)),
          scored_row(2,'a',dict(R0=False,R1=True,N0=False,N1=True))]
    g=D.changed_case_groups(rows)
    assert g['first_id']['R1_RECOVERY']=='10'  # Lexicographic source ID, not score or arrival order.
    assert 'R1_DAMAGE' in g['absent_categories']
    assert D.COORDINATE_DIAGNOSTIC_ARMS==('R1_PRECORRECTION_XY_FIXED_OBSERVATION','N1_PRECORRECTION_XY_FIXED_OBSERVATION')


def test_report_keeps_n_SD_threshold_pNA_and_no_default_change():
    rows=[scored_row(0,'a'),scored_row(1,'b')];x=T.summarize_population(rows)
    results=dict(populations={p:x for p in ('PRIMARY','TEST','REMAINDER','CAL')})
    paired={p:dict(comparisons=x['comparisons']) for p in results['populations']}
    out=R.make_report(results,paired,judgement='TRADEOFF_OR_UNRESOLVED',rationale='사전 고정 비교 결과를 해석한다.')
    assert '≤5cm' in out and 'strict <5/<5' in out and 'ddof=1' in out and '운영 기본 e는 변경하지 않았다' in out
    assert 'Holm' in out and 'NA' in out and 'n=2' in out
