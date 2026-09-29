from pathlib import Path
import pytest
from . import final_audit as F


def test_missing_inputs_never_means_pass(tmp_path):
    absent = F.missing_inputs(tmp_path)
    assert set(absent) == set(F.REQUIRED)


def test_public_bindings_allowed_but_pose_arrays_denied():
    F.privacy({'predictions': {'path': 'private/predictions.json', 'sha256': 'hash'}})
    with pytest.raises(AssertionError):
        F.privacy({'R_physical': [[1., 0., 0.]]})


def test_cluster_bootstrap_resamples_recordings_not_points():
    rows = [dict(id='a', recording='r1'), dict(id='b', recording='r1'), dict(id='c', recording='r2')]
    before = {r['id']: dict(available=True, translation_cm=5., rotation_deg=2.) for r in rows}
    after = {r['id']: dict(available=True, translation_cm=4., rotation_deg=1.) for r in rows}
    result = F.bootstrap(before, after, rows, repeats=20, seed=2)
    assert result['recording_count'] == 2 and result['recording_sizes'] == {'r1': 2, 'r2': 1}
    assert result['intervals']['translation_cm']['percentile95'] == [-1., -1.]
