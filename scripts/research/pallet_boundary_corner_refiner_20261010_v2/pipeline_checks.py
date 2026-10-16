"""Small CPU-only interface/admission/guard checks; no image/model/PnP run."""
from __future__ import annotations

import copy
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np

from . import common as C
from . import pipeline as P


def run(args):
    C.require(args.source_root and args.baseline_root,'explicit source/baseline required')
    os.environ['PALLET_SOURCE_ROOT']=str(Path(args.source_root).resolve())
    os.environ['PALLET_BASELINE_ROOT']=str(Path(args.baseline_root).resolve())
    from scripts.research.pallet_observation_refiner_20261009_v1 import common as old
    checked=[]
    def check(name, fn):
        fn();checked.append(dict(name=name,passed=True))
    def rejects(fn, text):
        try:
            fn()
        except (RuntimeError,AssertionError) as error:
            C.require(text in str(error),'guard rejected for unrelated reason: '+str(error))
        else:
            raise AssertionError('Expected rejection: '+text)
    n3=np.arange(18,dtype=float).reshape(9,2)+20
    line0=dict(edge=0,queries=[0,1,2],support_points=[[0,1],[1,1],[2,1]],normal=[0,1],offset=1.,query_radii_px=[.5,.5,.5])
    line1=dict(edge=3,queries=[21,22,23],support_points=[[1,0],[1,1],[1,2]],normal=[1,0],offset=1.,query_radii_px=[.5,.5,.5])
    observation=dict(corners=[dict(id=0,xy=(n3[0]+[8.,0]).tolist(),edges=[0,3],radius_px=8.)],lines=[line0,line1])
    def equal_radius():
        sparse,hybrid,contract=P.observation_points(n3,observation,[])
        assert contract['validated_boundary_corner_ids']==contract['hybrid_boundary_corner_ids']==[0]
        assert np.isnan(sparse[1:8]).all() and np.array_equal(hybrid[0],n3[0]+[8.,0])
        assert np.array_equal(sparse[8],n3[8])
    check('exact8px radius and distance accepted; missing ROLE_ONLY stays NaN',equal_radius)
    def uncertainty():
        case=copy.deepcopy(observation);case['corners'][0]['radius_px']=8.000001
        sparse,hybrid,contract=P.observation_points(n3,case,[])
        assert not contract['validated_boundary_corner_ids'] and np.isnan(sparse[:8]).all()
        assert contract['corner_admission'][0]['reason']=='UNCERTAINTY_EXCEEDS_PNP_OBSERVATION_RADIUS'
        assert np.array_equal(hybrid,n3)
    check('above-consensus uncertainty abstains without native point fabrication',uncertainty)
    def inconsistent_final_line():
        case=copy.deepcopy(observation);case['lines'][0]['support_points'][1]=[1.,2.]
        sparse,hybrid,contract=P.observation_points(n3,case,[])
        assert not contract['validated_boundary_corner_ids'] and contract['invalid_final_line_edges']==[0]
        assert contract['corner_admission'][0]['reason']=='FINAL_LINE_CONSENSUS_INCONSISTENT'
        assert contract['final_line_consensus_checks'][0]['absolute_residuals_px']==[0.,1.,0.]
        assert np.array_equal(hybrid,n3) and np.isnan(sparse[0]).all()
    check('final-TLS membership inconsistency vetoes line and its corner',inconsistent_final_line)
    def native_distance():
        case=copy.deepcopy(observation);case['corners'][0]['xy']=(n3[0]+[8.000001,0]).tolist()
        sparse,hybrid,contract=P.observation_points(n3,case,[])
        assert contract['validated_boundary_corner_ids']==[0] and not contract['hybrid_boundary_corner_ids']
        assert np.isfinite(sparse[0]).all() and np.array_equal(hybrid,n3)
    check('valid boundary outside native8px basin is ROLE_ONLY only',native_distance)
    def hidden_admission():
        sparse,hybrid,contract=P.observation_points(n3,observation,[0])
        assert np.isfinite(sparse[0]).all() and not contract['hybrid_boundary_corner_ids']
        assert contract['per_validated_corner'][0]['reason']=='initial_self_hidden'
        assert np.array_equal(hybrid,n3)
    check('frozen hidden ID cannot change hybrid or its no-mask coordinate bank',hidden_admission)
    before=dict(selected_index=0,candidates=[dict(score=.9,cls=0,keypoints_xy=n3.tolist()),dict(score=.1,cls=1,keypoints_xy=(n3+50).tolist())])
    def metadata():
        after=copy.deepcopy(before);after['candidates'][0]['keypoints_xy'][0]=[999.,777.]
        assert P.preserve_prediction(before,after)
        after['candidates'][0]['score']=.91
        rejects(lambda:P.preserve_prediction(before,after),'confidence/class')
        after=copy.deepcopy(before);after['candidates'][1]['keypoints_xy'][0]=[0.,0.]
        rejects(lambda:P.preserve_prediction(before,after),'nonselected')
        after=copy.deepcopy(before);after['candidates'][0]['keypoints_xy'][8]=[0.,0.]
        rejects(lambda:P.preserve_prediction(before,after),'center')
    check('selection confidence classes nonselected points and center protected',metadata)
    def camera_registry():
        dummy=P.Pipeline.__new__(P.Pipeline)
        dummy.models=SimpleNamespace(inf=SimpleNamespace(registry_input=lambda _:([1.1,1.3,.11],None)))
        row=dummy.registry_metadata(np.zeros((480,640,3),np.uint8),[[600,0,320],[0,600,240],[0,0,1]],[1.1,.11,1.3],dict(object_type='registry_fixture'))
        assert row['dimensions_wdh_m']==[1.1,1.3,.11] and row['xyz']==[1.1,.11,1.3]
        rejects(lambda:dummy.registry_metadata(np.zeros((2,2,3),np.uint8),np.eye(3),[1.1,1.3,.11],dict(object_type='registry_fixture')),'registered dimensions')
    check('full-native K and physical WHD to registry WDH explicit metadata',camera_registry)
    initial=dict(available=True,R_cf=np.eye(3).tolist(),R_physical=np.eye(3).tolist(),centroid=[0,0,3],cf_extents=[1.1,.11,1.3])
    contract=P.observation_points(n3,observation,[])[2]
    sparse=P.observation_points(n3,observation,[])[0]
    def finish_states():
        projected=np.arange(16,dtype=float).reshape(8,2)+100
        new=dict(initial,fit_input_ids=[0,1,2,3],projected=projected.tolist())
        row=P.assemble(n3,sparse,initial,[6,7],new,'VALIDATED_ROLE_ONLY',contract)
        assert row['new_pose_estimated'] and row['output_status']=='NEW_POSE'
        assert np.array_equal(row['native_points'][[6,7]],projected[[6,7]]) and np.array_equal(row['native_points'][8],n3[8])
        rejects(lambda:P.assemble(n3,sparse,initial,[6],dict(new,fit_input_ids=[0,1,2,6]),'VALIDATED_ROLE_ONLY',contract),'hidden initial coordinate')
        fallback=P.assemble(n3,sparse,initial,[6,7],dict(available=False,state='INSUFFICIENT_OBSERVATIONS'),'VALIDATED_ROLE_ONLY',contract)
        assert fallback['fallback_used'] and fallback['output_status']=='N3_BASELINE_FALLBACK' and np.array_equal(fallback['native_points'],n3)
        failure=P.assemble(n3,sparse,dict(available=False),[],dict(available=False),'VALIDATED_ROLE_ONLY',contract)
        assert failure['no_pose'] and not failure['fallback_used'] and failure['output_status']=='POSE_FAILURE'
    check('new hidden reprojection excludes its input; full-N3 fallback vs no-pose',finish_states)
    def truth_canary():
        # The file is a new fixture, never the actual reference source.
        with tempfile.TemporaryDirectory(prefix='boundary-canary-',dir='/dev/shm') as temp:
            fixture=Path(temp)/'TARGETS.json';fixture.write_text('{"fixture":true}')
            with old.no_truth_reads():
                rejects(lambda:fixture.read_text(),'GT_CANARY')
                rejects(lambda:fixture.open('rb'),'GT_CANARY')
    check('real Path.read_text/open GT canary denies fixture before read',truth_canary)
    def scoring_guard():
        import cv2
        from .evaluate import forbid_fits
        with forbid_fits():
            rejects(lambda:cv2.solvePnPGeneric(None,None,None,None),'pose fitting')
    check('posthoc scorer denies any PnP entry before numerical work',scoring_guard)
    def runtime_schedule():
        from .runtime import schedules,ARMS
        warm,measured=schedules()
        assert len(warm+measured)==600 and len(ARMS)==4
    check('fresh600 fixed runtime routes and schedule',runtime_schedule)
    payload=dict(schema='validated_boundary_pipeline_cpu_contract_checks_v2',passed=True,checks=checked,
        check_count=len(checked),actual_real_images=0,detector_forwards=0,N3_forwards=0,ROLE_forwards=0,
        actual_PnP_calls=0,new_training_updates=0,GT_reads=0,rays=0,
        source='synthetic point/line fixtures and one new denied-file fixture; no accuracy measurement',
        code=C.binding(P.__file__),checks_code=C.binding(__file__))
    C.write_new(C.output_path(args,args.receipt_name),payload)
    print('BOUNDARY_PIPELINE_CHECKS',len(checked),'PASS',flush=True)
    return payload


if __name__=='__main__':
    parser=C.parser(__doc__)
    parser.set_defaults(output=str(C.DOC))
    parser.add_argument('--receipt-name',choices=('PIPELINE_CHECKS.json','PIPELINE_CHECKS_FINAL.json'),default='PIPELINE_CHECKS.json')
    run(parser.parse_args())
