"""Small actual-gradient and archived-likelihood contracts; no dataset/GT."""
from copy import deepcopy
import numpy as np
import pytest
from . import common as C
from . import face_aware_partial_bi_observations_v4_13b as P
from . import truncation_interval_v4_2 as T
from . import statewise_partial_BI_u_solver_v4_13 as SOLVER


class Poison:
    def __array__(self, *args, **kwargs):
        raise AssertionError('Excluded coordinate converted')
    def __deepcopy__(self, memo):
        raise AssertionError('Excluded coordinate copied')


class OnlyAnchorRaw:
    def __init__(self, anchor=(30., 55.)):
        self.anchor = anchor; self.reads = []
    def __array__(self, *args, **kwargs):
        raise AssertionError('Full raw container converted')
    def __iter__(self):
        raise AssertionError('Full raw container iterated')
    def __getitem__(self, k):
        assert k == 0, 'Unknown/S corner raw coordinate read'
        self.reads.append(k)
        return self.anchor


def fixture():
    image = np.zeros((160, 160, 3), np.uint8)+50
    image[60:] = [180, 180, 180]
    mask = np.ones((160, 160), np.float32)
    # Camera lies exactly in the x=-.5 face plane: the left face is grazing.
    states = ['I', 'U', 'S', 'S', 'S', 'S', 'S', 'S']
    geometry = dict(native_xyz=C.cuboid(1., .1, 1.).tolist(), R_cf=np.eye(3).tolist(),
                    centroid=[.5, 4., 4.], image_hw=[160, 160],
                    initial_self_hidden=[False, False]+[True]*6,
                    initial_projection_xy=[[30., 60.], [130., 60.]]+[Poison() for _ in range(6)])
    fit = [dict(native_corner=0, state='I', xy=[30., 55.], source='I_PREVIOUS_RGB_CORRECTION',
                covariance_px2=[[3., 0.], [0., 3.]], fit_covariance_px2=[[25., 0.], [0., 25.]],
                fit_reliability_weight=.8)]
    model = dict(schema=T.SCHEMA, fit_partition='CAL', radius_ratio=.5,
                 recipe='ALL_VALID_INITIAL_NON_S_DIRECT_AND_EXTERNAL_UNTRIMMED_MAX')
    model['model_sha256'] = T._model_digest(model)
    kw = dict(raw_native_points=OnlyAnchorRaw(), localization_model=model, nonself_scale_px=100.,
              image_binding=dict(sha256='0'*64, image_hw=[160, 160]))
    return [image, mask, geometry, states, fit], kw


def test_actual_local_internal_line_can_use_U_model_guide_without_U_coordinate():
    args, kw = fixture(); before = deepcopy(args[-1])
    out = P.observe_face_aware_partial_bi(*args, **kw)
    assert len(out['boundary_observations']) == 1
    b = out['boundary_observations'][0]
    assert b['state'] == 'I' and b['explicit_complete_BI_likelihood_revocation']
    assert b['association_provenance']['revocation_reason'] == 'grazing_incident_face_junction'
    ray = b['lines'][0]['provenance']['line_record']
    assert ray['native_neighbor'] == 1 and not ray['whole_edge_search']
    assert ray['face_association']['role'] == 'INTERNAL_TWO_VISIBLE_FACES'
    assert ray['neighbor_coordinate_reads'] == ray['neighbor_corner_queries'] == 0
    assert args[-1] == before and out['visible_measurements'] == before
    assert kw['raw_native_points'].reads == [0]
    assert out['queried_native_ids'] == [0] and out['S_local_queries'] == out['U_corner_queries'] == 0
    assert abs(b['lines'][0]['offset_px']) > 50
    normalized = SOLVER._boundary_inputs([b], args[3], args[0].shape[:2], np.asarray(args[2]['native_xyz']))
    assert SOLVER.validate_complete_BI_likelihood_revocations(normalized, args[-1]) == [0]


def test_blank_RGB_abstains_without_raw_reads_or_fake_corner():
    args, kw = fixture(); args[0][:] = 50
    out = P.observe_face_aware_partial_bi(*args, **kw)
    assert out['boundary_observations'] == [] and kw['raw_native_points'].reads == []
    assert out['full_corner_measurements_created'] == 0 and out['visible_measurements'] == args[-1]


def test_S_endpoint_excluded_before_projection_and_local_patch():
    args, kw = fixture(); args[3][1] = 'S'; args[2]['initial_self_hidden'][1] = True
    args[2]['initial_projection_xy'][1] = Poison()
    out = P.observe_face_aware_partial_bi(*args, **kw)
    assert not out['boundary_observations']
    assert all(not r['pixel_queried'] for r in out['records'][0]['rays'])
    assert not kw['raw_native_points'].reads


def test_state_gates_precede_excluded_fit_fields_and_raw_container_reads():
    args, kw = fixture()
    for k in range(1, 8):
        args[-1].append(dict(native_corner=k, state=args[3][k], xy=Poison(), covariance_px2=Poison()))
    out = P.observe_face_aware_partial_bi(*args, **kw)
    assert [r['native_corner'] for r in out['visible_measurements']] == [0]
    assert out['U_coordinate_reads'] == out['O_T_coordinate_reads'] == out['S_coordinate_reads'] == 0


def test_unknown_mate_neighbourhood_is_replaced_before_all_remap_queries(monkeypatch):
    args, kw = fixture(); mate = np.array([45., 60.]); xy = np.array([[[45., 60.], [100., 60.]]])
    seen = []
    def trace(lab, gradients, mask, query, normal, offsets, internal, cfg):
        seen.append(query.copy())
        return dict(best=np.array([0]), keep=np.array([True]))
    monkeypatch.setattr(P.G, '_measure_trace', trace)
    out, audit = P._guarded_trace(None, None, args[1], xy, np.array([0., 1.]), np.array([0., 1.]), True, P._config(None), mate)
    assert np.array_equal(seen[0][0, 0], [-1000000., -1000000.])
    assert not out['keep'][0] and audit['exclusion_applied_before_RGB_and_mask_sampling']


def test_face_pair_role_is_independent_of_I_or_B_neighbor_label():
    args, kw = fixture(); context = P._face_context(args[2])
    assert P._edge_face_role(context, 0, 1)['role'] == 'INTERNAL_TWO_VISIBLE_FACES'
    assert P._edge_face_role(context, 0, 3)['role'] == 'GRAZING_FACE_ASSOCIATION_UNRESOLVED'
    # Move the camera outside both front/top/left faces: all three are strong.
    args[2]['centroid'] = [4., 4., 4.]
    context = P._face_context(args[2])
    assert all(P._edge_face_role(context, 0, mate)['role'] == 'INTERNAL_TWO_VISIBLE_FACES' for mate in (1, 3, 4))


def test_tangent_range_is_borrowed_from_first_raw_forecast_not_normal_display():
    args, kw = fixture(); out = P.observe_face_aware_partial_bi(*args, **kw)
    b = out['boundary_observations'][0]; n = np.asarray(b['lines'][0]['normal_unit_xy']); tangent = np.array([-n[1], n[0]])
    center = float(tangent @ np.array([30., 55.]))
    assert np.allclose(b['lines'][0]['provenance']['raw_tangent_CAL_borrowed_interval_px'], [center-50, center+50])
    assert not b['association_provenance']['tangent_range_is_probability']
    assert not b['association_provenance']['display_coordinate_is_full_2D_measurement']
    assert b['association_provenance']['original_measurement_source'] == 'I_PREVIOUS_RGB_CORRECTION'
    assert b['lines'][0]['normal_variance_px2'] == pytest.approx(25.)


def test_no_runtime_GT_file_or_ID_configuration():
    args, kw = fixture()
    with pytest.raises(ValueError, match='Unknown'):
        P.observe_face_aware_partial_bi(*args, **kw, configuration={'GT_corner': [0, 0]})
    kw['localization_model'] = '/tmp/not_a_CAL_mapping.json'
    with pytest.raises(ValueError, match='mapping'):
        P.observe_face_aware_partial_bi(*args, **kw)


def test_whole_BI_edge_uses_actual_two_endpoint_support_and_face_context():
    args, kw = fixture(); image, mask = args[:2]
    lab = P.cv2.GaussianBlur(P.cv2.cvtColor(image, P.cv2.COLOR_BGR2LAB).astype(np.float32),(3,3),.8)
    gradients = P.cv2.Sobel(lab,P.cv2.CV_32F,1,0,ksize=3)/8.,P.cv2.Sobel(lab,P.cv2.CV_32F,0,1,ksize=3)/8.
    points = {0:dict(xy=np.array([30.,55.]),corrected=False),
              1:dict(xy=np.array([130.,60.]),corrected=True)}
    projection = {0:np.array([30.,60.]),1:np.array([130.,60.])}
    ray = P._whole_edge(image,mask,lab,gradients,0,1,points,projection,True,P._config(None))
    assert ray['supported'] and ray['independent_both_endpoint_support']['passed']
    assert ray['mask_context_kind']=='TARGET_TARGET_INTERNAL'
    assert ray['central_sample_count']==15 and ray['whole_edge_search']
    assert ray['corrected_endpoint_normal_residual_px']['1']<=11
    assert ray['measurement_source']=='OBSERVED_FULL_EDGE_FACE_RGB_WITH_MASK_CONTEXT'
