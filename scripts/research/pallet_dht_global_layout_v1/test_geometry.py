"""Generated analytic fixtures; no real/synthetic image GT is evaluated here."""
import inspect
from itertools import product

import numpy as np
import pytest

from scripts.research.pallet_dht_global_layout_v1 import geometry as G


def project(camera):
    h = G.CUBOID_H @ camera.T
    q = h[:,:2]/h[:,2:]
    center = camera[:,3]
    return np.vstack([q,center[:2]/center[2]])


def example(affine=False):
    camera = np.array([[220.,20.,75.,1920.],[-15.,130.,-35.,1440.],
                       [.15,.1,.35,6.]])
    if affine:camera[2]=[0,0,0,6.]
    return project(camera)


def make_prepared(points=None, truth=None, sigma_modes=True):
    truth = example() if truth is None else truth
    points = truth.copy() if points is None else points
    lines = []
    for u,v in G.EDGES:
        h = np.cross(np.r_[truth[u],1],np.r_[truth[v],1])
        h /= np.linalg.norm(h[:2])
        modes = np.repeat(h[None],4,axis=0)
        if sigma_modes:modes[:,2] += np.array([0.,3.,-3.,6.])
        lines.append(modes)
    old = np.repeat(points[:8,None],49,axis=1)
    mask = np.zeros((8,49),bool);mask[:,0]=True
    for i in range(8):
        old[i,1:9] = truth[i]+np.array([[0,0],[1,0],[-1,0],[0,1],[0,-1],[2,0],[-2,0],[0,2]])
        mask[i,1:9]=True
    return G.prepare_inputs(points,np.ones(9,bool),640,480,old,mask,
                            np.asarray(lines),np.ones((12,4)),np.ones((12,4),bool))


def test_perspective_cube_dlt_is_exact_without_image_parallelism():
    q=example();r=G.projective_penalty(q[:8],800,8)
    assert r['valid'] and r['maximum_residual_px']<1e-9
    first=q[1]-q[0];second=q[5]-q[4]
    assert abs(np.cross(first,second))>1.


def test_affine_camera_and_vanishing_points_at_infinity_allowed():
    r=G.projective_penalty(example(True)[:8],800,8)
    assert r['valid'] and r['maximum_residual_px']<1e-9


def test_frontal_views_allow_coincident_front_rear_vertices():
    affine=np.array([[100.,0.,0.,320.],[0.,30.,0.,240.],[0.,0.,0.,1.]])
    perspective=np.array([[100.,0.,0.,1920.],[0.,30.,0.,1440.],[0.,0.,.2,6.]])
    qa=project(affine)
    np.testing.assert_array_equal(qa[:4],qa[4:8])
    for camera in (affine,perspective):
        r=G.projective_penalty(project(camera)[:8],800,8)
        assert r['valid'] and r['maximum_residual_px']<1e-9


def test_independent_corner_perturbation_has_nonzero_joint_residual():
    q=example();q[0]+=[9,-7]
    r=G.projective_penalty(q[:8],800,8)
    assert r['valid'] and r['mean_residual_px']>.1 and r['cost']>0


def test_camera_with_mixed_projective_depth_is_rejected():
    p=np.array([[200.,0.,30.,320.],[0.,80.,50.,240.],[.1,.2,1.,0.]])
    h=G.CUBOID_H@p.T
    q=h[:,:2]/h[:,2:]
    assert not G.projective_penalty(q,800,8)['valid']


@pytest.mark.parametrize('q,reason',[(np.zeros((8,2)),'collapsed_image_layout'),
                                  (np.column_stack([np.arange(8),np.arange(8)*2]),'collinear_image_layout')])
def test_collapsed_and_collinear_layouts_fail(q,reason):
    r=G.projective_penalty(q,800,8)
    assert not r['valid'] and r['reason']==reason


def test_yaw_is_graph_automorphism_but_does_not_validate_semantic_ids():
    edges={tuple(sorted(e)) for e in G.EDGES};q=example();prepared=make_prepared(sigma_modes=False)
    values=[]
    for perm in G.YAW_PERMUTATIONS:
        assert {tuple(sorted((int(perm[u]),int(perm[v])))) for u,v in G.EDGES}==edges
        r=G.projective_penalty(q[perm],800,8)
        assert r['valid'] and r['maximum_residual_px']<1e-9
        layout=q.copy();layout[:8]=q[perm]
        values.append(G.score_layout(prepared,layout,.25,1)['raw_terms']['shared_line'])
    assert values[0]<1e-12 and min(values[1:])>.05


def test_same_line_mode_for_both_endpoints_differs_from_separate_modes():
    h=np.array([[1.,0.,0.],[1.,0.,-20.]]);w=np.array([.5,.5])
    a=np.array([[0.,0.]]);b=np.array([[20.,10.]])
    independent=.5*(G.endpoint_cost(a,h,w,1)[0]+G.endpoint_cost(b,h,w,1)[0])
    shared=G.pair_cost(a,b,h,w,1)[0,0]
    assert shared>independent+5
    assert G.pair_cost(a,np.array([[0.,10.]]),h,w,1)[0,0]<shared


def test_single_mode_degree_normalization_independent_equals_shared():
    q=example();p=q.copy();p[:8]+=np.arange(8)[:,None]*[.9,-.4]
    a=make_prepared(p,q,sigma_modes=False)
    r=G.score_layout(a,p,.25,0,baseline_exception=True)
    assert np.isclose(r['raw_terms']['independent_line'],r['raw_terms']['shared_line'],atol=1e-12)


def test_antipodal_line_sign_and_endpoint_order_invariance():
    h=np.array([[.6,.8,-20.],[-.8,.6,-5.]]);w=np.array([.3,.7])
    a=np.array([[5.,8.],[9.,1.]]);b=np.array([[3.,4.],[2.,7.]])
    cost=G.pair_cost(a,b,h,w,8)
    np.testing.assert_allclose(cost,G.pair_cost(a,b,-h,w,8),atol=1e-12)
    np.testing.assert_allclose(cost,G.pair_cost(b,a,h,w,8).T,atol=1e-12)


def test_raw_affine_feature_line_incidence_is_preserved():
    theta=np.deg2rad(179.);rho=2.;affine=np.array([[.7,0,70.],[0,.7,82.]])
    h=G.feature_line_to_raw(theta,rho,(30,40),(480,640),affine)
    n=np.array([np.cos(theta),np.sin(theta)]);q_feature=n*rho+np.array([-n[1],n[0]])*3
    q_input=(q_feature+[20,15])*16
    raw=np.linalg.solve(affine[:,:2],q_input-affine[:,2])
    assert abs(h[:2]@raw+h[2])<1e-10
    other=G.feature_line_to_raw(theta+np.pi,-rho,(30,40),(480,640),affine)
    np.testing.assert_allclose(h,-other,atol=1e-10)


def test_candidate_pool_keeps_all8_and_prunes_only_by_line_unary():
    a=make_prepared()
    assert a['candidate_xy'].shape==(8,16,2) and a['candidate_valid'].all()
    for i in range(8):
        np.testing.assert_array_equal(a['candidate_xy'][i,:8],a['points_xy'][:8])
        scores=[]
        for slot in range(1,9):
            q=a['original_candidates_xy'][i,slot:slot+1]
            scores.append(np.mean([G.endpoint_cost(q,a['lines'][r],a['line_weights'][r],a['sigma'])[0] for r in G.INCIDENT_ROLES[i]]))
        expected=np.lexsort((np.arange(1,9),scores))+1
        np.testing.assert_array_equal(a['candidate_source_slot'][i,8:],expected)


def test_independent_has_original_point_first_tie_and_copies_centroid():
    a=make_prepared();a['unary_line_cost'][:]=0;a['anchor_cost'][:]=0
    s=G.independent_select(a,1)
    np.testing.assert_array_equal(s['indices'],np.arange(8))
    np.testing.assert_array_equal(s['points_xy'],a['points_xy'])


def test_beam_full_score_matches_exhaustive_small_graph():
    a=make_prepared();rng=np.random.default_rng(10)
    a['candidate_valid'][:]=False;a['candidate_valid'][:,0]=True;a['candidate_valid'][:3,1]=True
    a['anchor_cost']=rng.random((8,16));a['pair_line_cost']=rng.random((12,16,16))
    choices=[range(2) if i<3 else range(1) for i in range(8)]
    def cost(ii):
        _,anchor,shared,_=G._layout_terms(a,ii)
        return shared+.25*anchor
    brute=sorted((cost(ii),tuple(ii)) for ii in product(*choices))
    for order in G.ORDERS:
        beam=G._beam(a,order,.25)
        assert len(beam)==8
        assert abs(cost(beam[0])-brute[0][0])<1e-12
    # This tiny fully enumerated fixture does not imply full K16 search optimality.


def test_whole_layout_selection_and_public_score_terms_agree():
    a=make_prepared();bank=G.build_layouts(a,.25)
    for wg in (0.,.25,1.):
        out=G.select_layout(a,bank,wg)
        s=G.score_layout(a,out['points_xy'],.25,wg,baseline_exception=out['selected_index']==0)
        assert abs(s['total_score']-out['selected_score'])<1e-12
        np.testing.assert_array_equal(out['points_xy'],bank['hypothesis_xy'][out['selected_index']])
        np.testing.assert_array_equal(out['points_xy'][8],a['points_xy'][8])
        assert set(out['explicit_hypotheses'])=={'baseline','baseline_yaw0','baseline_yaw90','baseline_yaw180','baseline_yaw270','independent'}
        assert not out['uses_gt'] and not out['global_optimality_claim']


def test_geometry_weight_reranks_the_exact_same_bank():
    a=make_prepared();bank=G.build_layouts(a,1);before=bank['candidate_indices'].copy()
    G.select_layout(a,bank,0);G.select_layout(a,bank,1)
    np.testing.assert_array_equal(bank['candidate_indices'],before)
    assert bank['bank_is_geometry_weight_independent']


@pytest.mark.parametrize('reason',['missing_predicted_corner','no_detection','no_line_evidence'])
def test_input_failure_preserves_original_layout_and_mask(reason):
    a=make_prepared();a['fallback_reason']=reason
    if reason=='missing_predicted_corner':a['point_valid'][3]=False
    bank=G.build_layouts(a,4);out=G.select_layout(a,bank,1)
    assert out['identity_forced_by_input_failure'] and out['n_hypotheses']==1
    np.testing.assert_array_equal(out['points_xy'],a['points_xy'])
    np.testing.assert_array_equal(out['point_valid'],a['point_valid'])


def test_invalid_baseline_exception_is_recorded_not_claimed_valid():
    a=make_prepared();a['points_xy'][:8]=5.;a['candidate_xy'][:]=5.
    bank=G.build_layouts(a,.25);out=G.select_layout(a,bank,1)
    assert out['baseline_exception_selected']
    assert not out['selected']['geometry_valid']
    assert out['selected']['raw_terms']['geometry']==0
    with pytest.raises(ValueError,match='exact original'):
        G.score_layout(a,example(),.25,1,baseline_exception=True)


def test_nonfinite_predictions_fail_clearly():
    a=make_prepared();p=a['points_xy'].copy();p[0,0]=np.nan
    with pytest.raises(ValueError,match='Nonfinite'):
        G.prepare_inputs(p,a['point_valid'],640,480,a['original_candidates_xy'],
                         a['original_candidate_valid'],np.ones((12,4,3)),
                         np.ones((12,4)),np.ones((12,4),bool))


def test_prediction_only_constructor_has_no_gt_or_assignment_arguments():
    for fn in (G.prepare_inputs,G.prepare,G.independent_select,G.build_layouts,G.select_layout,G.score_layout):
        assert not any('gt' in name.lower() or 'target' in name.lower() for name in inspect.signature(fn).parameters)


def test_prepare_operates_with_only_whitelisted_prediction_fields():
    from scripts.research.pallet_dht_decoder_probe_v1.geometry import build_candidates
    class Whitelist(dict):
        def __getitem__(self,key):
            if key not in self:raise AssertionError('Unexpected field read: '+key)
            return super().__getitem__(key)
    points=example();theta=np.arange(90)*np.pi/90;rho=np.arange(-28,28.01,.5)
    logits=np.full((12,90,113),-5.)
    for r in range(12):logits[r,(r*7)%90,50+r%7]=5.
    footprint=np.abs(rho[None])<=20*np.abs(np.cos(theta[:,None]))+15*np.abs(np.sin(theta[:,None]))
    affine=np.array([[.7,0,70.],[0,.7,82.]])
    c=build_candidates(logits,theta,rho,footprint,(30,40),(480,640),affine,points,np.ones(9))
    record=Whitelist(baseline=Whitelist(points=points,point_valid=np.ones(9,bool),detected=True),
                     width=640,height=480,id='generated_no_gt_fixture')
    frame=Whitelist(theta=theta,rho=rho,logits=logits,lattice_valid=footprint,
                    feature_shape_hw=(30,40),input_shape_hw=(480,640),raw_to_input_affine=affine)
    evidence=Whitelist({'line_fusion__'+k:c[k] for k in
                       ['line_peaks_h_raw','line_peak_valid','candidates_xy','candidate_valid']})
    prepared=G.prepare(record,frame,evidence)
    assert prepared['candidate_xy'].shape==(8,16,2) and prepared['id']=='generated_no_gt_fixture'
    assert not prepared['uses_gt']


def test_geometry_scale_and_translation_covariance():
    q=example();q[0]+=[9,-7];r=G.projective_penalty(q[:8],800,8)
    q2=q[:8]*3+[500,-200];s=G.projective_penalty(q2,2400,24)
    assert r['valid'] and s['valid']
    assert abs(r['cost']-s['cost'])<1e-11
