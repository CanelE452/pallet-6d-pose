import numpy as np
from .train_target_following import restore_native,selected,summarize_errors


def test_native_inverse_padding_exactly_once():
    label=[0.,.5,.5,.2,.2]+[.5,.5,2.]*9
    xy,mask=restore_native(label,[480,640])
    np.testing.assert_allclose(xy,np.tile([320.,240.],(9,1)))
    assert mask.all()


def test_selected_is_not_target_nearest():
    predictions=dict(selected_index=1,candidates=[dict(keypoints_xy='near_target'),dict(keypoints_xy='highest_score')])
    assert selected(predictions)['keypoints_xy']=='highest_score'


def test_occurrence_weights_are_not_unique_point_weights():
    points=[dict(error_px=2.,occurrences_per_epoch=1,missing=False),dict(error_px=10.,occurrences_per_epoch=3,missing=False)]
    assert summarize_errors(points)['mean_px']==6.
    assert summarize_errors(points,True)['mean_px']==8.
