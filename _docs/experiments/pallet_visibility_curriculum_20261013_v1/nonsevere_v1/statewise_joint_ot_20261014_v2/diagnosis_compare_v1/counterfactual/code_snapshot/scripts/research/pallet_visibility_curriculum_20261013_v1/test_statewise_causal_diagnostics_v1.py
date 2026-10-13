"""Focused transform and phase contracts for the new postseal diagnostics."""
from copy import deepcopy
import numpy as np

from . import common as C
from . import joint_ot_pnp as P
from . import statewise_causal_diagnostics_v1 as D
from .scoring import rotation_y


def fixture():
    xyz=C.cuboid(1.3,.11,1.1);Q=rotation_y(np.pi/2);physical=rotation_y(.37);R=physical@Q.T;t=np.array([.03,.02,5.]);K=np.array([[400.,5.,320.],[0.,410.,240.],[0.,0.,1.]])
    q,depth=P.project_native(xyz,R,t,K);raw=np.c_[q].tolist();raw.append([320.,240.]);states=['B','I','O','T','S','U','B','S'];roles=['B','I','B','B','B','I','B','I']
    visible=[dict(native_corner=k,state=states[k],xy=(q[k]+[.1,-.2]).tolist(),covariance_px2=[[4.,.2],[.2,5.]],protection_px=2.,source='FROZEN_BI',image_measurement=True,measurement_is_pose_projection=False)for k in(0,1,6)]
    model=dict(minimum_radius_px=8.,normalized_radius=.1,model_sha256='fitted_CAL_model')
    domains=D.diagnostic_domains(states,q,[480,640],model,300.,center_source='RAW_IMAGE_PREDICTION')
    coarse=dict(R_cf=R.tolist(),R_physical=physical.tolist(),centroid=t.tolist(),cf_extents=[1.3,.11,1.1])
    rt=dict(initial_geometry=dict(native_xyz=xyz.tolist(),physical_registration_Q=Q.tolist(),initial_roles=roles,initial_projection_xy=q.tolist()),states=states,original_raw_points=raw,frozen_visible_measurements=visible,ot_domains=dict(domains=domains,nonself_prediction_scale_px=300.))
    row=dict(id='fixture:1',runtime=rt,K=K.tolist(),raw_hw=[480,640],initial_pose=coarse)
    human=['DIRECT_VISIBLE','DIRECT_VISIBLE','EXTERNAL_OCCLUDED','DIRECT_VISIBLE','SELF_OCCLUDED','UNANNOTATED','DIRECT_VISIBLE','DIRECT_VISIBLE']
    truth=dict(R=physical.tolist(),t=t.tolist(),xyz=[1.1,.11,1.3],body_xyz=[1.1,.11,1.3],body_R=physical.tolist(),order=2)
    # Force phase1 by a predeclared physical native-to-reference permutation.
    mapped=xyz@Q@rotation_y(np.pi).T;body=C.cuboid(*truth['body_xyz']);mapping=np.linalg.norm(mapped[:,None]-body[None,:],axis=2).argmin(axis=1).tolist();assert len(set(mapping))==8
    targetq,_=P.project_native(body,np.array(truth['body_R']),t,K)
    label=dict(target=dict(truth=truth,gt=targetq.tolist(),valid=[True]*8),metadata=dict(native_to_reference_corner_ids=mapping,human_states_native=human))
    alignment=D.reference_alignment(row,label);assert alignment['phase']==1 and alignment['compatible']
    plan=dict(solver_configuration={**P._DEFAULTS,'analytic_jacobian':True})
    return row,label,alignment,plan,model


def test_raw_same_sigma_changes_only_xy_and_no_ot_is_independent():
    row,label,a,plan,model=fixture();before=D.digest(row)
    s,_=D.transform(row,'RAW_BI_OT',plan,model,a,label);expected=deepcopy(row['runtime']['frozen_visible_measurements'])
    for r in expected:r['xy']=row['runtime']['original_raw_points'][r['native_corner']]
    assert s['visible']==expected and s['states']==row['runtime']['states'] and s['domains']==row['runtime']['ot_domains']['domains']
    no,_=D.transform(row,'RAW_BI_NO_OT',plan,model,a,label);assert no['visible']==s['visible']and no['domains']==[]and no['states']==['B','I','U','U','S','U','B','S']
    budget,_=D.transform(row,'CURRENT_BI_BUDGET6',plan,model,a,label);assert budget['visible']==row['runtime']['frozen_visible_measurements']and budget['domains']==s['domains']
    assert {k:v for k,v in budget['config'].items()if k!='maximum_visible_mahalanobis'}=={k:v for k,v in plan['solver_configuration'].items()if k!='maximum_visible_mahalanobis'}
    assert D.digest(row)==before
    full,_=D.transform(row,'FULL_HUMAN_RAW_STANDARD_COV',plan,model,a,label)
    joint,_=D.transform(row,'FULL_HUMAN_RAW_STANDARD_COV_BUDGET6',plan,model,a,label)
    assert full['visible']==joint['visible'] and full['domains']==joint['domains'] and full['states']==joint['states']
    assert joint['config']['maximum_visible_mahalanobis']==6.
    incompatible=deepcopy(a);incompatible['compatible']=False
    full9,_=D.transform(row,'FULL_HUMAN_RAW_STANDARD_COV',plan,model,incompatible,label)
    joint9,_=D.transform(row,'FULL_HUMAN_RAW_STANDARD_COV_BUDGET6',plan,model,incompatible,label)
    assert D.physics_key(full9)==D.physics_key(joint9)


def test_oracle_missing_direct_never_fabricates_T_and_fixed_radius():
    row,label,a,plan,model=fixture();row['runtime']['original_raw_points'][0]=[-1.,-1.];row['runtime']['original_raw_points'][1]=[640.,240.];row['runtime']['original_raw_points'][7]=[10000.,-9999.]
    binary,_=D.transform(row,'ORACLE_BINARY_RAW_STANDARD_COV',plan,model,a,label);full,_=D.transform(row,'FULL_HUMAN_RAW_STANDARD_COV',plan,model,a,label)
    assert binary['states'][3]=='T' and binary['states'][7]=='S'
    assert full['states'][0]=='U'and full['states'][1]=='U'and full['states'][7]=='U'and full['states'][3]=='B'
    assert all(m['native_corner']not in(0,1,7)for m in full['visible'])
    domain=next(d for d in full['domains']if d['native_corner']==2);assert domain['half_axes_px']==[30.,30.]
    assert next(m for m in full['visible']if m['native_corner']==3)['covariance_px2']==[[4.,0.],[0.,4.]]
    assert not any(s=='T'for s in full['states'])


def test_phase_uses_only_fixed_3D_mapping_and_reference_init_Q():
    row,label,a,plan,model=fixture();changed=deepcopy(label);changed['target']['gt']=(np.asarray(label['target']['gt'])+1000).tolist()
    b=D.reference_alignment(row,changed);assert a['phase']==b['phase']==1 and a['reference_R_cf']==b['reference_R_cf']
    Q=np.asarray(row['runtime']['initial_geometry']['physical_registration_Q']);G=np.asarray(label['target']['truth']['R']);expected=G@rotation_y(np.pi)@Q.T
    assert np.max(np.abs(np.asarray(a['reference_R_cf'])-expected))<1e-12
    spec,_=D.transform(row,'CURRENT_REFERENCE_INIT',plan,model,a,label);assert len(spec['initializations'])==3
    assert np.max(np.abs(np.asarray(spec['initializations'][2]['R'])-expected))<1e-12
    assert spec['visible']==row['runtime']['frozen_visible_measurements'] and spec['domains']==row['runtime']['ot_domains']['domains']


def test_ideal_reference_feasibility_and_incompatible_controls_exact():
    row,label,a,plan,model=fixture();spec,_=D.transform(row,'MODEL_IDEAL_PROJECTIONS',plan,model,a,label);audit=D.reference_audit(row,spec,label)
    assert audit['any_representative_feasible']
    assert all(m['measurement_is_pose_projection']and not m['image_measurement']and m['covariance_px2']==[[1.,0.],[0.,1.]]for m in spec['visible'])
    # Model positive control includes its construction reference, independent of initial coarse.
    check=deepcopy(spec);check['config'].update(global_rotation_starts=0,max_p3p_triples=0,max_joint_iterations=30)
    result=P.solve_joint_ot_pnp(check['xyz'],check['K'],check['states'],check['visible'],check['domains'],check['image_hw'],initializations=check['initializations'],config=check['config'])
    assert result['accepted'] and result['objective']<1e-8
    incompatible=deepcopy(a);incompatible['compatible']=False
    standard,_=D.transform(row,'CURRENT_RAW_STANDARD_COV',plan,model,a,label)
    for arm in('ORACLE_BINARY_RAW_STANDARD_COV','FULL_HUMAN_RAW_STANDARD_COV'):
        control,p=D.transform(row,arm,plan,model,incompatible,label);assert D.physics_key(control)==D.physics_key(standard)and p['incompatible_control_reused']
    ideal,_=D.transform(row,'MODEL_IDEAL_PROJECTIONS',plan,model,incompatible,label);Q=np.asarray(row['runtime']['initial_geometry']['physical_registration_Q']);G=np.asarray(label['target']['truth']['R'])
    assert np.max(np.abs(np.asarray(ideal['initializations'][-1]['R'])-G@Q.T))<1e-12
    assert 'O'not in ideal['states']
