"""Actual provenance/guide/owner gates with synthetic samples only."""
from copy import deepcopy
import numpy as np
import pytest
from . import geometry_guided_edge_authentication_v4_29 as A
from .test_geometry_guided_edge_problem_v4_29 import fixture as cost_fixture
from .test_complete_bi_image_chain_evidence_v4_20d import ray


def fixture(two=False,partial=False):
    base,_,(k,a,b),R,t=cost_fixture(partial=partial)
    projection,_=A.G.V.project_native(base.xyz,R,t,base.K)
    geometry=dict(K=base.K.tolist(),initial_self_hidden=[s=='S'for s in base.states],initial_projection_xy=projection.tolist(),
        native_xyz=base.xyz.tolist(),R_cf=R.tolist(),centroid=t.tolist(),physical_registration_Q=base.Q.tolist(),image_hw=base.image_hw)
    records=[];refs=[];nodes=[]
    for i,mate in enumerate([a,b]if two else[a]):
        record=ray(base,projection,k,mate);chain=record['chains'][0]
        d=projection[mate]-projection[k];d/=np.linalg.norm(d)
        guide_reference=dict(seed_direction_xy=d.tolist(),actual_search_direction_xy=d.tolist(),
            seed_normal_xy=[-float(d[1]),float(d[0])],native_corner=k,native_neighbor=mate)
        guide_id=A.G._digest(guide_reference)
        guide_reference['guide_reference_id']=guide_id
        chain.update(observation_confidence=[1.]*len(chain['observed_points']),generating_guide_direction_xy=d.tolist(),
            generating_guide_reference_id=guide_id,proposal_grade='FULL_LENGTH_IMAGE_LINE_PROPOSAL')
        ref=dict(line_name='line'+str(i),ray_index=i,chain_index=0,native_corner=k,native_neighbor=mate,
                 guide_reference_id=guide_id,guide_reference=guide_reference)
        refs.append(ref);records.append(record)
        nodes.append(dict(name=ref['line_name'],normal=chain['normal'],offset=chain['offset'],actual_inliers=chain['observed_points'],
            image_quality_energy=.5,global_external_context=dict(available=True,actual_support_near_global_external_fraction=1.,same_foreground_component_fraction=1.)))
    obs=dict(observation_generation=A.PRODUCER_GENERATION,image_binding=dict(sha256='a'*64,image_hw=base.image_hw),
        initial_geometry_digest=A.G._digest({key:geometry[key]for key in A.GEOMETRY_DIGEST_FIELDS}),
        initial_geometry_digest_fields=list(A.GEOMETRY_DIGEST_FIELDS),chain_observation=dict(edge_records=records),
        selected_chain_refs=refs,joint_graph=dict(selected_line_names=[r['line_name']for r in refs],line_candidates=nodes))
    kwargs=dict(image_binding=obs['image_binding'],initial_geometry=geometry,
        noise_provenance=dict(fit_partition='CAL',observation_family=A.G.OBSERVATION_FAMILY),calibration_preparation=True)
    return base,obs,kwargs,(k,a,b)


def test_actual_full_provenance_is_bound_and_both_BI_owners_are_typed():
    base,obs,kw,(k,mate,_)=fixture();bundle=A.authenticate_geometry_guided_edges(base,obs,**kw)
    assert len(bundle['groups'])==1
    group=bundle['groups'][0]
    assert group['selected_chain_ref']==obs['selected_chain_refs'][0]
    assert group['lattice_relation_error_px']<1e-7
    assert {o['native_corner']for o in group['activation_owners']}=={k,mate}
    assert {o['owner_provenance']for o in group['activation_owners']}=={
        'SELF_ANCHOR_ACTUAL_SUPPORT','NAMED_PHYSICAL_EDGE_MODEL_CONSTRAINT_ONLY'}
    assert all(o['order_provenance']=='IMMUTABLE_BI_ENDPOINT_HEMISPHERE'for o in group['activation_owners'])
    assert all(not o['opposite_corner_new_image_measurement_created']for o in group['activation_owners'])


@pytest.mark.parametrize('change',['lattice','gradient','confidence','RGB','mask','guide','selected_node','reference'])
def test_unbound_actual_samples_or_new_guide_identity_reject(change):
    base,obs,kw,_=fixture();chain=obs['chain_observation']['edge_records'][0]['chains'][0]
    if change=='lattice':chain['actual_inlier_image_lattice_xy'][0][0]+=1
    elif change=='gradient':chain['image_gradient_normal_xy'][0]=[0.,0.]
    elif change=='confidence':chain['observation_confidence']=[]
    elif change=='RGB':chain['RGB_contrast'][0]=0.
    elif change=='mask':chain['mask_plus'][0]=0.
    elif change=='guide':chain['generating_guide_direction_xy']=[1.,0.]
    elif change=='selected_node':obs['joint_graph']['line_candidates'][0]['offset']+=1.
    else:obs['selected_chain_refs'][0]['guide_reference_id']='unrelated'
    with pytest.raises(ValueError):A.authenticate_geometry_guided_edges(base,obs,**kw)


def test_new_partial_has_full_RGB_provenance_and_does_not_fake_old_guard_pass():
    base,obs,kw,_=fixture();chain=obs['chain_observation']['edge_records'][0]['chains'][0]
    chain.update(proposal_grade='PARTIAL_LINE_PROPOSAL',inlier_span_fraction=.53,
                 corrected_endpoint_normal_guard_pass=False,corrected_endpoint_normal_residual_px={'old':22.})
    bundle=A.authenticate_geometry_guided_edges(base,obs,**kw)
    component=bundle['groups'][0]
    assert component['proposal_grade']=='PARTIAL_LINE_PROPOSAL'
    assert component['original_q_endpoint_guard_and_12px_corner_cap_superseded']
    assert not component['full_2D_corner_measurement']
    assert obs['chain_observation']['edge_records'][0]['chains'][0]['corrected_endpoint_normal_guard_pass']is False
    del chain['RGB_alignment']
    with pytest.raises(KeyError):A.authenticate_geometry_guided_edges(base,obs,**kw)


def test_two_measured_finite_directed_edges_are_joined_by_exact_component_ids():
    base,obs,kw,(k,_,_)=fixture(two=True);bundle=A.authenticate_geometry_guided_edges(base,obs,**kw)
    pairs=[p for p in bundle['ordered_corner_pairs']if p['native_corner']==k]
    assert len(pairs)==1 and pairs[0]['passed']
    assert set(pairs[0]['component_ids'])=={g['component_id']for g in bundle['groups']}
    assert max(pairs[0]['finite_actual_support']['original_actual_support_gaps_px'])<11.


def test_S_gate_precedes_chain_array_lookup_and_copy():
    base,obs,kw,_=fixture();hidden=next(i for i,s in enumerate(base.states)if s=='S')
    class Forbidden(dict):
        def __getitem__(self,key):
            if key in('native_corner','native_neighbor'):return super().__getitem__(key)
            raise AssertionError('Forbidden S payload access')
    obs['chain_observation']['edge_records']=[Forbidden(native_corner=hidden,native_neighbor=0)]
    with pytest.raises(ValueError,match='State/S gates'):
        A.authenticate_geometry_guided_edges(base,obs,**kw)


def test_negative_selected_array_index_rejects_instead_of_aliasing_last_entry():
    base,obs,kw,_=fixture();obs['selected_chain_refs'][0]['ray_index']=-1
    with pytest.raises(ValueError,match='nonnegative'):
        A.authenticate_geometry_guided_edges(base,obs,**kw)


def test_runtime_noise_source_policy_is_fail_closed():
    base,obs,kw,_=fixture();kw['calibration_preparation']=False
    kw['noise_provenance'].update(model_digest_sha256='c'*64,source_policy='RGB_NONFALLBACK_PRESERVE')
    with pytest.raises(ValueError,match='source policy'):
        A.authenticate_geometry_guided_edges(base,obs,expected_source_policy='ALL_BI_REPLACE',**kw)


def test_base_query_direction_and_TLS_seed_normal_are_separately_bound():
    base,obs,kw,_=fixture();chain=obs['chain_observation']['edge_records'][0]['chains'][0]
    reference=obs['selected_chain_refs'][0];guide=reference['guide_reference']
    d=np.asarray(guide['actual_search_direction_xy']);angle=np.deg2rad(5.)
    rotated=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])@d
    guide['actual_search_direction_xy']=rotated.tolist();new_id=A.G._digest({k:v for k,v in guide.items()if k!='guide_reference_id'})
    guide['guide_reference_id']=new_id
    reference['guide_reference_id']=new_id;chain['generating_guide_reference_id']=new_id
    chain['generating_guide_direction_xy']=rotated.tolist()
    result=A.authenticate_geometry_guided_edges(base,obs,**kw)
    assert result['groups'][0]['per_guide_angle_change_deg']==pytest.approx(0.,abs=1e-5)
    assert np.array_equal(result['groups'][0]['generating_guide_direction_xy'],rotated)


def test_real_new_CAL_exporter_kwargs_bind_directly_to_runtime_AUTH_once():
    from . import geometry_guided_edge_noise_calibration_v4_29 as M
    from .test_geometry_guided_edge_noise_calibration_v4_29 import records
    model=M.fit_geometry_guided_line_model(records(),source_policy='ALL_BI_REPLACE')
    parameters=M.mixture_noise_parameters(model,source_policy='ALL_BI_REPLACE')
    base,obs,kw,_=fixture();kw.pop('noise_provenance');kw.pop('calibration_preparation')
    bundle=A.authenticate_geometry_guided_edges(base,obs,expected_source_policy='ALL_BI_REPLACE',**kw,**parameters)
    group=bundle['groups'][0]
    assert parameters['normal_angle_correlation']==0. and group['normal_angle_correlation']==0.
    assert group['normal_variance_px2']==pytest.approx(parameters['normal_variance_component_px2']+group['normal_floor_px']**2)
    bad={**parameters,'normal_angle_correlation':.1}
    with pytest.raises(ValueError,match='zero normal/angle'):
        A.authenticate_geometry_guided_edges(base,obs,expected_source_policy='ALL_BI_REPLACE',**kw,**bad)
