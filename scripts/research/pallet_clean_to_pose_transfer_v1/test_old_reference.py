"""과거 reference 재집계의 분모·동률·짝지은 변화 회귀검사."""
import pytest

from .old_reference import pair, scalar_pose_recheck, summary


def row(t, r, available=True):
    return dict(available=available, translation_cm=t, rotation_deg=r, yaw_deg=r, axis_correct=True)


def test_pair_partitions_all_nine_directions_and_failed():
    before, after = {}, {}
    for index, (dt, dr) in enumerate((t,r) for t in (-1.,0.,1.) for r in (-1.,0.,1.)):
        before[str(index)] = row(2.,2.)
        after[str(index)] = row(2.+dt,2.+dr)
    before['bad'] = row(2.,2.)
    after['bad'] = row(2.,2., False)
    result, private = pair(before, after, list(before))
    assert result['common_valid'] == len(private) == 9
    assert result['both_improve'] == result['T_only'] == result['R_only'] == result['both_worsen'] == 1
    assert result['any_tie'] == 5
    assert result['available_to_failed'] == 1
    assert all(value == 1 for value in result['direction_counts'].values())


def test_failed_pose_is_not_zero_or_dropped_from_full_population():
    result = summary([row(1.,1.),row(2.,2.,False),row(3.,3.,False)])
    assert result['frames'] == 3 and result['valid_pose'] == 1
    assert result['conditional']['translation_cm']['median'] == 1.
    assert result['full_population']['translation_cm']['median'] is None
    assert result['full_population']['translation_cm']['median_status'] == 'POSITIVE_INFINITY'


def test_difference_of_medians_is_not_median_of_differences():
    before = {str(i):row(t,t) for i,t in enumerate([0.,10.,11.])}
    after = {str(i):row(t,t) for i,t in enumerate([9.,10.,100.])}
    result, _ = pair(before, after, list(before))
    assert result['difference_of_medians']['translation_cm'] == 0.
    assert result['median_of_paired_differences']['translation_cm'] == 9.


def test_pose_scalar_check_preserves_centimeter_and_c2():
    eye = [[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]
    pose = dict(available=True, centroid=[0.,0.,1.], R_physical=eye)
    truth = dict(order=2,t=[0.,0.,0.],R=eye)
    scalar_pose_recheck(row(100.,0.), pose, truth)
    with pytest.raises(AssertionError):
        scalar_pose_recheck(row(1.,0.), pose, truth)
