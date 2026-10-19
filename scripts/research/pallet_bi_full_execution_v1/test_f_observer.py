"""Constructed policy tests; no real reference or pose optimization."""
from copy import deepcopy
from pathlib import Path
import argparse
import numpy as np
from . import f_observer as F
from .s1_union import sha,write
from scripts.research.pallet_visibility_curriculum_20261013_v1 import statewise_partial_BI_u_solver_v4_13 as P

def run():
    frame=dict(image_sha256='a'*64,raw_hw=[100,100])
    states=['B']+['S']*7
    def ray(cue,name,rms):
        return dict(source_line_id=name,cue=cue,mask_sources=['tight'],native_neighbor=1,
            supported=True,normal=[0.,1.],offset=-20.,actual_inlier_tangent_interval_px=[10.,30.],
            physical_source_id='pixels:'+name,inlier_rms_px=rms,samples=15,inlier_count=15)
    rays=[ray('B','mask',1.),ray('I','rgb',.1)]
    pool=dict(id='CONSTRUCTED',states=states,corners=[dict(native_corner=0,state='B',queried=True,
             q0=[20.,20.],rays=rays,candidates=[]),dict(native_corner=1,state='S',queried=False)])
    before=deepcopy(pool);common=F.choose(pool,frame,'COMMON');bi=F.choose(pool,frame,'BI')
    assert common['selected_candidates'][0]['selected']=='rgb'
    assert bi['selected_candidates'][0]['selected']=='mask'
    assert pool==before
    for selection in (common,bi):
        assert len(selection['partial_lines'])==1
        parsed=P._boundary_inputs(selection['partial_lines'],states,frame['raw_hw'])
        assert P.validate_complete_BI_likelihood_revocations(parsed,selection['full_points'])==[0]
        assert selection['selected_candidates'][0]['new_cue_dimension']==1
    empty=deepcopy(pool);empty['corners'][0]['rays']=[]
    selection=F.choose(empty,frame,'COMMON')
    assert not selection['partial_lines'] and selection['full_points'][0]['xy']==[20.,20.]
    assert selection['selected_candidates'][0]['new_cue_dimension']==0
    changed=deepcopy(pool);changed['states'][0]='I';changed['corners'][0]['state']='I'
    cc=F.choose(changed,frame,'COMMON')
    assert cc['selected_candidates']==common['selected_candidates']
    bad=deepcopy(common);bad['full_points'][0]['xy'][0]+=1
    try:P.validate_complete_BI_likelihood_revocations(P._boundary_inputs(bad['partial_lines'],states,frame['raw_hw']),bad['full_points'])
    except ValueError:pass
    else:raise AssertionError('Corrupted archive was accepted')
    return dict(passed=True,constructed_tests=['same cue pool, different adoption policies','COMMON role invariant',
        'supported single line becomes1D','no line retainsN0 with0newimage dimensions','strict unchanged revocation validator',
        'corrupted revocation rejected','immutable evidence','S excluded before fields'],new_pose_solves=0,real_reference_reads=0)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();r=run()
    write(a.run/'F_OBSERVER_CONSTRUCTED_TESTS.json',r)
    write(a.run/'F_OBSERVER_POLICY_LOCK_V2.json',dict(intervention_table=F.intervention_table(),source_sha256=sha(F.__file__),
        tests_sha256=sha(__file__),locked_before_F_production_and_scoring=True,
        supersedes='F_OBSERVER_POLICY_LOCK.json; source guard repair rejects a claimed2D junction made from one actual pixel source twice; no F scientific calls before repair',
        activation='afterS1-S4 fixed validation-based system selection; this shared observation pool/matching producer cannot be replaced by a condition-bit comparison',
        calibration='own full-pointCAL residuals per arm, same pooled recipe selected by preregistered S2LOGO, no changed-q covariance inheritance'))
    print(r)
