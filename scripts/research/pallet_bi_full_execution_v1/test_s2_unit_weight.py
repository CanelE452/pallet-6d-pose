from copy import deepcopy

import numpy as np

from scripts.research.pallet_bi_conditioned_observation_v1.test_solver import fixture,partial_selection
from scripts.research.pallet_bi_full_execution_v1 import s2_unit_weight as U,calibration as C,qcov as Q
from scripts.research.pallet_bi_full_execution_v1.test_qcov_calibration import sample_points


def test_unit_correction_removes_covariance_reliability_confound_without_touching_typed_archive():
    frame,selection,_,_,truth=fixture()
    for row in selection['full_points']:row['fit_reliability_weight']=.8
    selection=partial_selection(frame,selection,truth)
    partial_ids={r['native_corner']for r in selection['partial_lines']}
    original=deepcopy(selection);out,changed=U.corrected_selection(selection)
    U.assert_selection_contract(original,out)
    for old,new in zip(original['full_points'],out['full_points']):
        if old['native_corner']in partial_ids:assert new==old
        else:
            assert new['fit_reliability_weight']==1.
            assert new['xy']==old['xy']
            assert np.array_equal(Q.covariance(old),Q.covariance(new))
    assert selection==original
    assert set(changed)=={r['native_corner']for r in selection['full_points']}-partial_ids
    visible,partial,_=Q.S._deduplicate(out['full_points'],out['partial_lines'])
    Q.S._prepare(frame['native_xyz'],frame['K'],out['states'],visible,frame['raw_hw'],partial,Q.S.SOLVER_CONFIG)


def test_calibration_fit_is_independent_of_solver_reliability_gamma():
    points=sample_points();cases=[dict(id=p['id'],partition='CAL',nonself_scale_px=100.)for p in points]
    model=C.fit(points,cases)
    for i,p in enumerate(points):p['fit_reliability_weight']=.1 if i%2 else .8
    assert C.fit(points,cases)==model
    frame,selection,*_=fixture()
    for row in selection['full_points']:row['fit_reliability_weight']=.8
    cp=C.apply_selection(selection,model,'CP',100.)
    corrected,_=U.corrected_selection(selection)
    assert C.apply_selection(corrected,model,'CP',100.)==cp
    assert C.apply_selection(corrected,model,'CB',100.)==C.apply_selection(selection,model,'CB',100.)


def test_already_unit_or_implicit_unit_packet_is_exactly_reusable():
    _,selection,*_=fixture()
    for row in selection['full_points']:row.pop('fit_reliability_weight',None)
    result,changed=U.corrected_selection(selection)
    assert result==selection and changed==[]
