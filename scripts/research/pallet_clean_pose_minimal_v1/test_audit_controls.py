"""독립 감사의 실패 처리, paired 의미, 선택/비공개 경계를 검사한다."""
import numpy as np
import pytest

from . import audit_controls as A


def metric(t, r, available=True):
    return dict(available=available, translation_cm=t, rotation_deg=r, yaw_deg=r,
                ADDsym_normalized=.01, IoU3D=.5, axis_correct=True)


def test_failed_pose_is_in_full_denominator_not_zero():
    x = A.summary([metric(1., 2.), metric(0., 0., False)])
    assert x['coverage'] == .5 and x['frames'] == 2 and x['valid_pose'] == 1
    assert x['conditional']['translation_cm']['median'] == 1.
    assert x['full_population']['translation_cm']['median'] is None
    assert x['full_population']['translation_cm']['median_status'] == 'POSITIVE_INFINITY'


def test_paired_median_and_median_difference_are_not_conflated():
    a = {str(i): metric(v, 5.) for i, v in enumerate((0., 10., 11.))}
    b = {str(i): metric(v, 5.) for i, v in enumerate((9., 1., 12.))}
    out = A.paired(a, b, list(a))
    assert out['difference_of_conditional_medians']['translation_cm'] == -1.
    assert out['median_of_common_frame_differences']['translation_cm'] == 1.
    assert sum(out['paired_direction_counts'].values()) == 3
    assert out['paired_direction_counts']['T_IMPROVE__R_TIE'] == 1


def test_selected_name_not_better_pose_error():
    row = dict(current={'error': 5}, hypotheses=[dict(name='chosen', pose={'error': 90}), dict(name='other', pose={'error': 0})])
    assert A.selected_pose(row, 'chosen') == {'error': 90}
    assert A.selected_pose(row, None) == row['current']


def test_guard_read_paths_fail_on_reference_access():
    row = dict(reference_or_metrics_read_before_decisions=False, read_guard_active=True,
        candidate_order_swap_test=True, final_poses_from_cached_candidates=True,
        new_selector_fits=0, read_paths=['/safe/PREDICTIONS.json'])
    assert A.validate_decision_reads(row) == 1
    row['read_paths'].append('/bad/POSE_METRICS.json')
    with pytest.raises(AssertionError):
        A.validate_decision_reads(row)


def test_public_privacy_rejects_coordinates_and_image_bytes():
    A.public_privacy({'source': {'path': 'private/PREDICTIONS.json', 'sha256': 'hash'}, 'Tmedian': 1.})
    for row in ({'centroid': [1, 2, 3]}, {'image': 'data:image/png;base64,abc'}):
        with pytest.raises(AssertionError):
            A.public_privacy(row)


def test_distribution_matches_numpy_for_finite_values():
    values = [2., 7., 3., 20., 9.]
    out = A.distribution(values)
    assert np.isclose(out['median'], np.median(values))
    assert np.isclose(out['P90'], np.quantile(values, .9))
