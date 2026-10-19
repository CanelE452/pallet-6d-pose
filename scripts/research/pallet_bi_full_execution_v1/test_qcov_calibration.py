"""Analytic transport, revocation, covariance and CAL isolation contracts."""
from copy import deepcopy
import json

import numpy as np
import pytest

from scripts.research.pallet_bi_conditioned_observation_v1.test_solver import fixture, partial_selection
from scripts.research.pallet_bi_full_execution_v1 import qcov as Q
from scripts.research.pallet_bi_full_execution_v1 import calibration as C


def test_strict_cross_rejects_changed_physical_source_but_keeps_diagonals():
    frame,a,*_=fixture();b=deepcopy(a);b['full_points'][0]['physical_source_ids']=['different_actual_RGB']
    assert not Q.comparable(a,b)['eligible']
    with pytest.raises(ValueError,match='invalid'):Q.mix(a,b)
    assert Q.comparable(a,a)['eligible']


def test_fixed_graph_transports_q_and_covariance_with_honest_separate_sources():
    frame,a,*_=fixture();b=deepcopy(a);original=deepcopy(a)
    for point in b['full_points']:
        point['xy'][0]+=3;point['covariance_px2']=[[4.,0.],[0.,9.]]
        point['physical_source_ids']=['other_RGB:'+str(point['native_corner'])]
    for qb,cb in ((0,0),(1,0),(0,1),(1,1)):
        out,provenance=Q.fixed_common_packet(frame,a,b,qb,cb)
        for k,(old,donor,result,factor)in enumerate(zip(a['full_points'],b['full_points'],out['full_points'],provenance['factors'])):
            assert result['xy']==(donor if qb else old)['xy']
            assert result['covariance_px2']==(donor if cb else old)['covariance_px2']
            assert factor['q_donor_source_ids']==(donor if qb else old)['physical_source_ids']
            assert factor['canonical_original_source_ids']==old['physical_source_ids']
            assert result['physical_source_ids']==['CANONICAL_LOGICAL_FULL_FACTOR:'+str(old['native_corner'])]
        assert all(f['changed_source_association']==bool(qb or cb)for f in provenance['factors'])
    assert a==original


def test_fixed_graph_preserves_typed_archive_and_exact_numeric_baseline_cache():
    frame,a,_,_,truth=fixture();a=partial_selection(frame,a,truth);b=deepcopy(a)
    partial_id=a['partial_lines'][0]['native_corner']
    for point in b['full_points']:
        if point['native_corner']!=partial_id:point['xy'][0]+=2
    baseline,provenance=Q.fixed_common_packet(frame,a,b,0,0)
    assert Q.parsed_solver_payload(frame,a)==Q.parsed_solver_payload(frame,baseline)
    shifted,prov=Q.fixed_common_packet(frame,a,b,1,1)
    assert shifted['partial_lines']==a['partial_lines']
    assert next(p for p in shifted['full_points']if p['native_corner']==partial_id)==next(p for p in a['full_points']if p['native_corner']==partial_id)
    Q.S._prepare(frame['native_xyz'],frame['K'],shifted['states'],shifted['full_points'],frame['raw_hw'],shifted['partial_lines'],Q.S.SOLVER_CONFIG)
    corrupt=deepcopy(shifted);corrupt['partial_lines'][0]['association_provenance']['original_observed_xy'][0]+=1
    with pytest.raises(ValueError):Q.validate_fixed_packet(a,corrupt,prov)


def test_missing_effective_donor_retains_common_and_records_reason():
    frame,a,*_=fixture();b=deepcopy(a);missing=b['full_points'].pop(0)['native_corner']
    out,prov=Q.fixed_common_packet(frame,a,b,1,1)
    result=next(p for p in out['full_points']if p['native_corner']==missing)
    original=next(p for p in a['full_points']if p['native_corner']==missing)
    assert result['xy']==original['xy']and result['covariance_px2']==original['covariance_px2']
    assert next(p for p in prov['factors']if p['native_corner']==missing)['missing_BI_donor_fallback_reason']


def sample_points():
    return [dict(id=str(i),partition='CAL',cluster_id='A'if i<5 else 'B',state='B'if i%2 else 'I',
        native_corner=0,reference_error_xy=[4.,0.],covariance_px2=[[1.,0.],[0.,1.]],nonself_scale_px=100.)for i in range(10)]


def test_floor_matches_analytic_zero_centered_80percent_ellipse_no_bias():
    points=sample_points();cases=[dict(id=p['id'],partition='CAL',nonself_scale_px=100.)for p in points]
    model=C.fit(points,cases);expected=16/C.RECIPE['target_mahalanobis2']-1
    assert model['pooled_floor_px2']==pytest.approx(expected,abs=1e-12)
    assert model['bias_parameters']==0
    assert model['roles']['I']['shrinkage_weight']==pytest.approx(5/25)
    points[0]['partition']='PRIMARY'
    with pytest.raises(ValueError,match='CAL'):C.fit(points,cases)


def test_calibration_never_changes_q_conditional_or_partial_and_c0_is_exact():
    frame,a,_,_,truth=fixture();a=partial_selection(frame,a,truth)
    points=sample_points();model=C.fit(points,[dict(id=p['id'],partition='CAL',nonself_scale_px=100.)for p in points])
    assert C.apply_selection(a,model,'C0',100.)==a
    result=C.apply_selection(a,model,'CB',100.)
    assert result['partial_lines']==a['partial_lines']
    partial_ids={p['native_corner']for p in a['partial_lines']}
    for old,new in zip(a['full_points'],result['full_points']):
        assert new['xy']==old['xy']and new['covariance_px2']==old['covariance_px2']
        if old['native_corner']in partial_ids:assert new==old
        else:assert np.linalg.eigvalsh(new['fit_covariance_px2']).min()>0


def test_reference_extractor_lexically_skips_primary_numeric_objects(tmp_path,monkeypatch):
    # Invalid JSON numeric tokens in PRIMARY prove that only CAL is decoded.
    refs={str(i):dict(order=2,R=np.eye(3).tolist(),t=[0.,0.,1.])for i in range(46)}
    reference=tmp_path/'source.json';reference.write_text('{"references":{"PRIMARY":{"forbidden":THIS_IS_NOT_A_JSON_NUMBER},"CAL":'+json.dumps(refs)+'}}')
    plan=tmp_path/'plan.json';plan.write_text('{}')
    with pytest.raises(PermissionError):C.isolate_cal_references(reference,list(refs),plan,Q.sha(plan),tmp_path/'out')
    result,receipt=C.isolate_cal_references(reference,list(refs),plan,Q.sha(plan),tmp_path/'out',root_GO=True)
    assert result==refs
    assert receipt['PRIMARY_numeric_reference_objects_decoded']is False
    assert receipt['source_entire_file_bytes_read']is True
    with pytest.raises(ValueError,match='whitelist'):C.isolate_cal_references(reference,list(refs)[:-1],plan,Q.sha(plan),tmp_path/'out2',root_GO=True)
