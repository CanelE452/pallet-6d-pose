import numpy as np
import pytest
from .renderer_front_visibility_audit import visibility, summarize
from .camera_facing_contract_audit import ORIGIN


def test_front_and_rigid_transform_invariance_without_mutation():
    q = ORIGIN.copy(); cam = np.array([0., -5., 3.])
    a, el = visibility(q, cam)
    assert a.argmax() == 0 and a[0] > 0 and a[1] < 0
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    b, el2 = visibility(q @ rotation.T*2+[2, 8, -3], cam @ rotation.T*2+[2, 8, -3])
    np.testing.assert_allclose(a, b, atol=1e-12)
    assert abs(el-el2) < 1e-12
    np.testing.assert_array_equal(q, ORIGIN)


def test_front_need_not_be_max_and_degeneracy_is_explicit():
    a, _ = visibility(ORIGIN, np.array([6., -2., 3.]))
    assert a.argmax() == 3
    with pytest.raises(ValueError): visibility(np.zeros((8, 3)), [0, 0, 3])
    with pytest.raises(ValueError): visibility(ORIGIN, [np.nan, 0, 3])


def test_missing_records_stay_in_denominator():
    s = summarize([dict(available=False, id='missing')])
    assert s['total'] == 1 and s['available'] == 0 and s['unavailable'] == 1
    assert s['stored_cos_max_error'] is None
