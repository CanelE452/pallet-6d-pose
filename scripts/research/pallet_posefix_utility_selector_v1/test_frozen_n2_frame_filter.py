import copy
import numpy as np
from .frozen_n2_frame_filter import keep_unchanged


def test_kept_prediction_is_exact_and_independent():
    pred={'points':[[1.0,2.0],[3.0,4.0]],'score':.9}
    expected=copy.deepcopy(pred)
    kept=keep_unchanged(pred,True,.05,.05)
    assert pred==kept==expected and kept is not pred
    kept['points'][0][0]=99
    assert pred==expected


def test_rejection_omits_whole_prediction_without_fallback():
    for confidence,loo in [(False,.01),(True,.0501),(True,None),(True,np.nan),(True,np.inf)]:
        pred={'points':[[1.,2.],[3.,4.]]};expected=copy.deepcopy(pred)
        assert keep_unchanged(pred,confidence,loo,.05) is None
        assert pred==expected
