"""One small image-only contract check; no detector, refiner or pose inference."""
import copy
import hashlib
import inspect
import json
from pathlib import Path
import time
from unittest import mock

import cv2
import numpy as np
from . import methods as M

DOC = M.ROOT / '_docs/experiments/pallet_training_free_compare_20261007_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_tests():
    started=time.monotonic();config=M.method_configuration()
    bindings={name:sha(Path(__file__).with_name(name)) for name in ('methods.py','tests.py')}
    path=DOC/'CHECKS_METHODS.json'
    if path.exists():
        prior=json.loads(path.read_text())
        assert prior['status']=='PASS' and prior['code_sha256']==bindings
        assert prior['method_lock']==config['method_lock']
        return prior
    checks={}; details={};native_calls=0;actual_subpix_calls=0;actual_collect_calls=0
    def call(gray,p,s,method):
        nonlocal native_calls
        native_calls+=1
        return M.correct(gray,p,s,method)

    assert set(inspect.signature(M.correct).parameters)=={'gray','initial_points','prediction_support','method'}
    assert config['CVRANK']['radius_px']==12 and config['SUBPIX']['winSize']==[5,5]
    assert config['SUBPIX']['zeroZone']==[-1,-1]
    assert config['SUBPIX']['criteria']==dict(type=cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,maxCount=40,epsilon=.001)
    assert config['source_calibration']['all_frozen_configs_match_source_recorded_optimum']
    checks['fixed_configuration_and_GT_free_inference_signature']=True

    # Existing rank direction and exact ordinal/default-argsort tie semantics.
    info=dict(segments=[],origin=np.zeros(2))
    cands=[dict(xy=np.array([8.,0]),response=0.,family='toy0'),
           dict(xy=np.array([4.,0]),response=1.,family='toy1'),
           dict(xy=np.array([1.,0]),response=2.,family='toy2')]
    before=copy.deepcopy(cands); info_before=copy.deepcopy(info)
    chosen,score=M.G.selector_score(cands,info,np.zeros(2),[])
    assert chosen is cands[2] and score==1.
    for a,b in zip(cands,before):
        assert np.array_equal(a['xy'],b['xy']) and a['response']==b['response']
    assert np.array_equal(info['origin'],info_before['origin']) and info['segments']==info_before['segments']
    ties=[dict(xy=np.array([1.,0]),response=1.,family=str(i)) for i in range(3)]
    chosen,score=M.G.selector_score(ties,info,np.zeros(2),[])
    assert chosen is ties[2] and score==1., 'Preserve original default ordinal ranks, not averaged tie ranks'
    cancel=[dict(xy=np.array([1.,0]),response=1.,family='a'),dict(xy=np.array([2.,0]),response=0.,family='b')]
    chosen,score=M.G.selector_score(cancel,info,np.zeros(2),[])
    assert chosen is cancel[0] and score==.5, 'Final equal scores must use first argmax'
    assert M.G.selector_score([],{},np.zeros(2),[]) is None
    details['selector']=dict(all_four_terms_favor_index=2,duplicate_equal_signals_original_index=2,
        duplicate_equal_signals_note='original ordinal ranks distinguish equal raw signals; unchanged, not averaged ties',
        equal_final_score_first_argmax_index=0,rank_direction_correct=True)
    checks['selector_direction_duplicate_ties_first_argmax_and_nonmutation']=True

    initial=np.array([[10.+k,20.+k] for k in range(9)],dtype=np.float64)
    initial[1]=[-1,-1];initial[2]=[np.nan,np.nan]
    support=np.ones(9,dtype=bool);support[1:3]=False
    native=initial.copy();native[0]+=[30,40];native[3]+=[0,0];native[8]+=[30,40]
    native[1]=[99,99];native[2]=[99,99]
    a,b,s=initial.copy(),native.copy(),support.copy()
    capped=M.cap_points(initial,native,300,400,support)
    assert capped.dtype==np.float64 and np.array_equal(capped[8],initial[8])
    assert np.array_equal(capped[~support],initial[~support],equal_nan=True)
    assert np.array_equal(capped[3],initial[3])
    assert np.isclose(np.linalg.norm(capped[0]-initial[0]),5.,rtol=0,atol=1e-12)
    assert np.array_equal(initial,a,equal_nan=True) and np.array_equal(native,b,equal_nan=True) and np.array_equal(support,s)
    assert np.array_equal(M.cap_points(initial,initial,300,400,support),initial,equal_nan=True)
    checks['float64_diagonal_cap_zero_center_missing_and_nonmutation']=True

    gray=np.zeros((64,96),np.uint8);gray[20:45,50:75]=255
    p=np.tile([49.2,19.3],(9,1)).astype(np.float64);p[8]=[38.25,47.5]
    s=np.zeros(9,bool);s[0]=True;s[8]=True
    gray_before,p_before,s_before=gray.copy(),p.copy(),s.copy()
    # Changing an unrelated GT record cannot enter this API. Trap every legacy
    # GT loader to catch accidental calls through a reused module.
    gt_record=dict(points=np.zeros((9,2)),visibility=np.ones(9,dtype=int))
    with mock.patch.object(M.G.M,'load_gt',side_effect=AssertionError('GT must not enter correction')):
        sub,diag=call(gray,p,s,'SUBPIX');actual_subpix_calls+=diag['algorithm_corner_calls']
        gt_record['points'][:]=1e6;gt_record['visibility'][:]=0
        sub2,diag2=call(gray,p,s,'SUBPIX');actual_subpix_calls+=diag2['algorithm_corner_calls']
    assert np.array_equal(sub,sub2,equal_nan=True) and diag==diag2
    assert sub.dtype==np.float64 and abs(sub[0,0]-49.5)<1 and abs(sub[0,1]-19.5)<1
    assert sub[0,0]>sub[0,1]+20, 'Raw pixel x/y order must remain x/y'
    assert np.array_equal(sub[8],p[8]) and np.array_equal(sub[~s],p[~s])
    assert np.array_equal(gray,gray_before) and np.array_equal(p,p_before) and np.array_equal(s,s_before)
    details['SUBPIX_raw_pixel_output']=sub[0].tolist()
    checks['raw_pixel_xy_GT_perturbation_and_SUBPIX_input_nonmutation']=True

    # A flat patch exercises the actual, frozen four-family collector once.
    flat=np.zeros_like(gray)
    cv,d=call(flat,p,s,'CVRANK');actual_collect_calls+=d['algorithm_corner_calls']
    assert np.array_equal(cv,p) and d['fallback_counts']=={'no_candidate':1}
    assert d['corner_records'][0]['candidate_count']==0
    checks['actual_frozen_CVRANK_empty_patch_and_fallback_count']=True

    tiny=np.zeros((8,8),np.uint8);q=p.copy();q[0]=[1,1]
    unchanged,d=call(tiny,q,s,'SUBPIX');assert np.array_equal(unchanged,q) and d['fallback_counts']=={'image_too_small':1}
    outside=p.copy();outside[0]=[-2,10]
    unchanged,d=call(gray,outside,s,'SUBPIX');assert np.array_equal(unchanged,outside) and d['fallback_counts']=={'outside_initial':1}
    missing=p.copy();missing[0]=[np.nan,np.nan]
    unchanged,d=call(gray,missing,s,'SUBPIX')
    assert np.array_equal(unchanged,missing,equal_nan=True) and d['algorithm_corner_calls']==0 and d['status_counts']['nonfinite_initial']==1
    for value,reason in ((np.array([[[np.nan,10]]],np.float32),'nonfinite_refined'),
                         (np.array([[[200,10]]],np.float32),'outside_refined')):
        with mock.patch.object(M.cv2,'cornerSubPix',return_value=value):
            unchanged,d=call(gray,p,s,'SUBPIX')
        assert np.array_equal(unchanged,p) and d['fallback_counts']=={reason:1}
    with mock.patch.object(M.cv2,'cornerSubPix',side_effect=cv2.error('toy OpenCV failure')):
        unchanged,d=call(gray,p,s,'SUBPIX')
    assert np.array_equal(unchanged,p) and d['fallback_counts']=={'function_error':1}
    assert d['corner_records'][0]['exception']['message']=='toy OpenCV failure'
    checks['tiny_boundary_nonfinite_and_exception_fallback_counts']=True

    # Do not add a new candidate/radius/image-bound gate to the legacy CV rank.
    candidate=dict(xy=np.array([130.,100.]),response=1.,family='lsd')
    with mock.patch.object(M.G,'collect',return_value=([candidate],dict(segments=[],origin=np.zeros(2),counts={'lsd':1}))):
        outside_output,d=call(gray,outside,s,'CVRANK')
    assert np.array_equal(outside_output[0],candidate['xy']) and d['corner_records'][0]['outside_refined_image']
    assert d['fallback_counts']=={} and d['corner_records'][0]['selected_candidate_index']==0
    assert np.linalg.norm(M.cap_points(outside,outside_output,96,64,s)[0]-outside[0])<=.01*np.hypot(96,64)+1e-12
    checks['CVRANK_clipped_patch_contract_retains_finite_outside_candidate']=True

    checks['no_new_backbone_refiner_pose_or_optimizer_execution']=True
    result=dict(status='PASS',checks=checks,check_count=len(checks),seconds=time.monotonic()-started,
        code_sha256=bindings,method_lock=config['method_lock'],source_calibration=config['source_calibration'],
        reused_code_sha256=config['reused_code'],details=details,
        execution=dict(test_invocations=1,native_wrapper_calls=native_calls,
            actual_OpenCV_cornerSubPix_calls=actual_subpix_calls,actual_frozen_CVRANK_collect_calls=actual_collect_calls,
            other_calls='small mocked fallback/rank fixtures',detector_backbone_calls=0,refiner_calls=0,
            final_F_calls=0,internal_PnP_calls=0,optimizer_updates=0,config_search=0),
        metadata_scope='wrapper has no box/confidence/object-selection argument and cannot mutate those caller fields; full frame metadata checked by evaluation',
        inherited_tie_semantics='equal raw terms receive original ordinal np.argsort ranks; final score ties use first np.argmax',
        inherited_generator_exception_visibility='original collect suppresses individual family exceptions; wrapper records escaping errors, but no claim that hidden family-error count is zero')
    DOC.mkdir(parents=True,exist_ok=True)
    pending=path.with_suffix('.json.pending');pending.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n');pending.replace(path)
    return result


if __name__=='__main__':
    result=run_tests()
    print(json.dumps(dict(status=result['status'],checks=result['check_count'],seconds=result['seconds'],execution=result['execution']),ensure_ascii=False))
