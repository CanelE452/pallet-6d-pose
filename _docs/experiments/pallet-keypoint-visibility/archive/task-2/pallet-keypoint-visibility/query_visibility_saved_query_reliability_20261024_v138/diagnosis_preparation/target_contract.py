"""GT-only target math, called only after completed INPUT bin seal + root entry."""
import numpy as np

def targets(teacher_state,error_px,c29_prediction,c29_available):
    state=np.asarray(teacher_state);error=np.asarray(error_px,np.float64);pred=np.asarray(c29_prediction);available=np.asarray(c29_available,bool)
    assert state.shape==error.shape==pred.shape==available.shape and np.isin(state,list('DESU')).all()
    assert np.isin(pred,[-1,0,1]).all()and np.array_equal(pred==-1,~available)
    assert (error[np.isfinite(error)]>=0).all()and not np.isinf(error).any()
    knownDE=np.isin(state,['D','E']);truth=np.where(state=='D',0,np.where(state=='E',1,-1))
    location=np.full(len(state),-1,np.int8);location[np.isfinite(error)]=(error[np.isfinite(error)]>3).astype(np.int8)
    available_mistake=np.full(len(state),-1,np.int8);mask=knownDE&available;available_mistake[mask]=(pred[mask]!=truth[mask]).astype(np.int8)
    full_failure=np.full(len(state),-1,np.int8);full_failure[knownDE]=(pred[knownDE]!=truth[knownDE]).astype(np.int8)
    return dict(EXTERNAL_TEACHER_E1_D0=truth.astype(np.int8),GT_ERROR_GT3=location,C29_DE_MISTAKE_WHEN_INPUT_AVAILABLE=available_mistake,C29_KNOWN_FAILURE_INCLUDING_INPUT_U=full_failure)

def generated_checks():
    out=targets(list('DDEESU'),[np.nan,2,4,1,6,3],[-1,1,0,1,1,0],[0,1,1,1,1,1])
    assert out['EXTERNAL_TEACHER_E1_D0'].tolist()==[0,0,1,1,-1,-1]
    assert out['GT_ERROR_GT3'].tolist()==[-1,0,1,0,1,0]
    assert out['C29_DE_MISTAKE_WHEN_INPUT_AVAILABLE'].tolist()==[-1,1,1,0,-1,-1]
    assert out['C29_KNOWN_FAILURE_INCLUDING_INPUT_U'].tolist()==[1,1,1,0,-1,-1]
    return dict(PASS=True,invented_data_only=True,external_E1_D0_teacher_SU_unknown_not_relabelled=True,NOQUERY_location_unknown_not_automatically_GTerror=True,inputU_knownDE_failure_positive_preserved=True,
        available_only_DE_mistake_separate_target=True,teacherSU_outside_external_binary_not_relabelled=True,teacherSU_finite_location_targets_retained=True)
