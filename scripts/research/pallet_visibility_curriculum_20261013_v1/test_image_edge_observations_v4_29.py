"""Uniform observation contracts on synthetic RGB only, no dataset or labels."""
from copy import deepcopy
import numpy as np
import pytest

from . import image_edge_observations_v4_29 as P
from .test_joint_geometry_guided_lines_v4_28 import fixture as old_fixture,Poison


def fixture():
    args,kw=old_fixture()
    args[2]['centroid']=[0.,.15,1.5]
    args[2]['K']=[[100.,0,80.],[0,100.,50.],[0,0,1.]]
    return args,kw


def call(args,kw,**extra):
    return P.observe_joint_geometry_edge_evidence(*args,**kw,**extra)


def selected_lines(out):
    return [out['chain_observation']['edge_records'][r['ray_index']]['chains'][r['chain_index']]
            for r in out['selected_chain_refs']]


def test_fresh_full_line_has_exact_per_guide_NMS_provenance_and_selected_reference():
    args,kw=fixture();before=deepcopy(args[-1]);out=call(args,kw)
    assert out['selected_chain_refs'] and args[-1]==before
    assert out['selected_physical_edges']==len({tuple(r['native_edge'])for r in out['selected_chain_refs']})
    for line in selected_lines(out):
        assert line['full_actual_image_provenance'] and line['measurement_source']==P.MEASUREMENT_SOURCE
        lattice=np.asarray(line['actual_image_lattice_xy'])
        actual=lattice+np.asarray(line['measured_subpixel_shift_px'])[:,None]*np.asarray(line['image_gradient_normal_xy'])
        assert np.allclose(actual,line['observed_points'],atol=1e-9,rtol=0)
        assert line['guide_reference']['unoriented_angle_to_proposal_deg']==pytest.approx(line['angle_change_deg'],abs=1e-5)
    assert not out['full_2D_corner_measurements_created'] and not out['labels_or_reference_access']


def test_fresh_span_partial_has_all_actual_arrays_not_cached_diag_imputation():
    args,kw=fixture();args[3][1]='I'
    args[-1].append(dict(native_corner=1,state='I',xy=[130.,55.],source='VISIBLE_BACKBONE_FALLBACK',
        covariance_px2=[[3.,0],[0,3.]],fit_covariance_px2=[[25.,0],[0,25.]],fit_reliability_weight=.8))
    args[0][:]=50;args[0][60:,35:55]=180
    out=call(args,kw)
    partial=[line for row in out['chain_observation']['edge_records']for line in row.get('chains',[])
             if line['proposal_grade']==P.PARTIAL_GRADE]
    assert partial and out['partial_line_ledger']
    for line in partial:
        assert not line['original_fitter_accepted'] and not line['original_span_guard_pass']
        assert line['actual_finite_inlier_span_px']>=11
        count=len(line['observed_points'])
        for field in ('RGB_gradient_energy','RGB_alignment','RGB_contrast','mask_plus','mask_minus',
                      'measured_subpixel_shift_px','actual_image_lattice_xy'):
            assert len(line[field])==count
        assert line['full_actual_image_provenance']
    assert all(n['RGB_gradient_lattice_provenance_complete']for n in out['joint_graph']['line_candidates'])


def test_blank_image_abstains_without_fake_line_or_state_or_U_point():
    args,kw=fixture();args[0][:]=50
    out=call(args,kw)
    assert out['selected_chain_refs']==[] and out['full_2D_corner_measurements_created']==0
    assert out['U_intersections_created']==out['S_U_local_coordinate_or_corner_queries']==0


def test_S_U_poison_observation_fields_never_accessed():
    args,kw=fixture()
    for k in range(1,8):args[-1].append(dict(native_corner=k,state=args[3][k],xy=Poison(),covariance_px2=Poison()))
    out=call(args,kw)
    assert out['S_U_local_coordinate_or_corner_queries']==0
    assert out['selected_chain_refs']
    assert all(r['native_corner']==0 for r in out['selected_chain_refs'])


def test_fresh_fitter_partial_rechecks_later_RMS_and_finite_span_guards():
    cfg=P.G.NMS._config(None);fit=P._fit_proxy(cfg)
    settings=dict(along_interval=[0.,1.],huber_delta_px=cfg['maximum_line_rms_px'],
        maximum_angle_change_deg=cfg['maximum_direction_change_deg'],minimum_inlier_count=6,
        minimum_inlier_fraction=.6,minimum_inlier_span_fraction=.55,maximum_line_rms_px=1.25)
    actual=np.column_stack([np.linspace(0,17,18),np.ones(18)*50])
    result,diag=fit(actual,np.ones(18),np.linspace(0,.535,18),np.array([0.,1.]),18,settings)
    assert result is not None and diag['proposal_grade']==P.PARTIAL_GRADE
    tiny=actual.copy();tiny[:,0]*=.1
    result,diag=fit(tiny,np.ones(18),np.linspace(0,.535,18),np.array([0.,1.]),18,settings)
    assert result is None


def test_task_guides_use_only_unregularized_input_bank_and_not_numeric_baseline():
    args,_=fixture();g=args[2];xyz=np.asarray(g['native_xyz']);Q=np.array([[0.,0,1],[0,1,0],[-1,0,0]])
    families={}
    for name,q in (('ORIGINAL_WD',np.eye(3)),('SWAPPED_WD',Q)):
        families[name]=dict(native_xyz=(xyz@q.T).tolist(),physical_registration_Q=q.tolist(),
            coarse_pose=dict(R=q.T.tolist(),t=g['centroid']))
    class Problem:
        scale=1.
        def __init__(self,xyz):self.xyz=np.asarray(xyz)
        def evaluate(self,R,t):return dict(accepted=True,R=R,t=t,objective=float(np.linalg.norm(t)))
    task=dict(families=families,proposals=[dict(R_physical=np.eye(3).tolist(),t=[i*.02,.15,1.5])for i in range(8)],
        spec=dict(K=g['K'],image_hw=g['image_hw']),initial_self_hidden=g['initial_self_hidden'],states=args[3],baseline=Poison())
    guides,audit=P.prepare_task_pose_guides(task,{n:Problem(f['native_xyz'])for n,f in families.items()})
    assert len(guides)<=14 and len([v for v in guides if v['source_kind']=='ORIGINAL_TASK_COARSE'])==2
    assert audit['old_numeric_e_pose_or_GT_error_read']is False
    assert all(v['geometry']['initial_projection_xy'][2]is None for v in guides)


def test_guide_initial_S_cannot_be_reclassified_before_query():
    args,kw=fixture();guide={k:deepcopy(v)for k,v in args[2].items()if k!='initial_projection_xy'}
    guide['initial_projection_xy']=args[2]['initial_projection_xy'];guide['initial_self_hidden'][2]=False
    with pytest.raises(ValueError,match='permanent initial-S'):
        call(args,kw,pose_guides=[dict(guide_id='bad',source_kind='ORIGINAL_TASK_COARSE',geometry=guide)])


def test_both_family_guides_rebind_one_native_table_preserving_physical_pose_and_nonzero_center():
    args,_=fixture();g=args[2]
    xyz=np.asarray(g['native_xyz'])+np.array([.13,.04,-.08])
    Q=np.array([[0.,0,1],[0,1,0],[-1,0,0]])
    from scipy.spatial.transform import Rotation
    physical=Rotation.from_rotvec([.2,-.3,.1]).as_matrix();center=np.array([.1,.2,2.])
    permutation=[4,0,3,7,5,1,2,6]
    families={}
    for name,q,points in [('ORIGINAL_WD',np.eye(3),xyz),('SWAPPED_WD',Q,xyz[permutation]@Q.T)]:
        R=physical@q.T
        families[name]=dict(native_xyz=points.tolist(),physical_registration_Q=q.tolist(),
            coarse_pose=dict(R=R.tolist(),t=(center-R@points.mean(0)).tolist()))
    class Problem:
        scale=1.
        def __init__(self,points):self.xyz=np.asarray(points)
        def evaluate(self,*args):raise AssertionError('No proposal should be evaluated')
    task=dict(families=families,proposals=[],spec=dict(K=g['K'],image_hw=g['image_hw']),
        initial_self_hidden=g['initial_self_hidden'],states=args[3],baseline=Poison())
    guides,audit=P.prepare_task_pose_guides(task,{n:Problem(f['native_xyz'])for n,f in families.items()})
    assert len(guides)==2
    for item in guides:
        trial=item['geometry']
        assert np.array_equal(trial['native_xyz'],xyz)and np.array_equal(trial['physical_registration_Q'],np.eye(3))
        assert np.allclose(trial['R_cf'],physical,atol=1e-14,rtol=0)
        assert np.allclose(trial['centroid'],center,atol=1e-14,rtol=0)
    for k in (0,1):
        assert np.allclose(guides[0]['geometry']['initial_projection_xy'][k],
            guides[1]['geometry']['initial_projection_xy'][k],atol=1e-12,rtol=0)
    assert audit['source_family_corner_permutation_not_treated_as_same_native_image_edge']


def test_unrebound_family_native_table_is_rejected_before_image_queries():
    args,kw=fixture();guide=dict(args[2]);guide['native_xyz']=list(reversed(guide['native_xyz']))
    with pytest.raises(ValueError,match='ORIGINAL fixed native'):
        call(args,kw,pose_guides=[dict(guide_id='SWAPPED_WD/coarse',
            source_kind='ORIGINAL_TASK_COARSE',geometry=guide)])
