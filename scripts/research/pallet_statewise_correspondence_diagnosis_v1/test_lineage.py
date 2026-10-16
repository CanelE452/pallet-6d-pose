"""Pure typed/owner/serialization contracts in the actual archive schema."""
from copy import deepcopy
import json
import unittest

from . import lineage as L


def fixture():
    I=[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]
    states=['B','I','S','O','T','U','B','I']
    pose=dict(available=True,R_cf=I,R_physical=I,centroid=[0.,0.,5.],cf_extents=[2.,.2,1.])
    visible=[dict(native_corner=k,state=states[k],xy=[float(k),float(k+1)],
                  covariance_px2=[[4.,0.],[0.,4.]],fit_covariance_px2=[[9.,0.],[0.,9.]],
                  fit_reliability_weight=.8,source='VISIBLE_BACKBONE_FALLBACK'if k==0 else'B_TARGET_BACKGROUND_LINES')
             for k in (0,1,6,7)]
    line=dict(normal_unit_xy=[1.,0.],offset_px=-5.,normal_variance_px2=2.,
              support_tangent_interval_px=[-5.,10.],support_extrapolation_px=3.,provenance={})
    boundary=[dict(native_corner=0,state='B',lines=[deepcopy(line)],fit_reliability_weight=.8,
                   replaces_raw_B_likelihood=True,association_provenance={'display_coordinate_is_full_2D_measurement':False}),
              dict(native_corner=1,state='I',lines=[deepcopy(line)],fit_reliability_weight=.8,
                   explicit_complete_BI_likelihood_revocation=True,replaces_complete_BI_likelihood=True,
                   association_provenance={'display_coordinate_is_full_2D_measurement':False})]
    coarse=dict(R=deepcopy(I),t=[0.,0.,5.],physical_registration_Q=deepcopy(I))
    family=dict(name='ORIGINAL_WD',native_xyz=L.EI.cuboid(pose['cf_extents']).tolist(),
                cf_extents=pose['cf_extents'],physical_registration_Q=deepcopy(I),coarse_pose=deepcopy(coarse))
    row=dict(id='synthetic:0',partition='CAL',session='synthetic',image='synthetic.jpg',image_sha256='a'*64,
             finish_input={'_original_source':{'states':states},
                           'partial_boundary_observation':{'boundary_observations':deepcopy(boundary)},
                           'frozen_BI_fit':deepcopy(visible)},
             boundary_typed=deepcopy(boundary),boundary_all=deepcopy(boundary),boundary_original_legacy=deepcopy(boundary[:1]),
             spec=dict(xyz=family['native_xyz'],K=[[500.,0.,320.],[0.,500.,240.],[0.,0.,1.]],
                       states=states,visible=deepcopy(visible),domains=[],image_hw=[480,640],coarse_pose=coarse,config={'prior_weight':0.}),
             u_regions=[],outline={'supported':False,'configuration':{}},family=family,fixed_family='ORIGINAL_WD',
             baseline=dict(actual_pose=deepcopy(pose),WD_family='ORIGINAL_WD',accepted=True,
                           solution={'schema':'statewise_partial_BI_outline_solver_v4_13','accepted':True,
                                     'objective_components':{'removed_complete_BI_2D_likelihood_ids':[0,1]}}),
             stage1_family_seed_bank={'ORIGINAL_WD':{'family':deepcopy(family),'initial_solutions':[]}})
    frame={'id':row['id'],'arms':{'E_FROZEN':{'coarse_pose':deepcopy(pose)}}}
    return row,frame


class LineageContracts(unittest.TestCase):
    def test_legacy_and_explicit_partial_both_remove_full_cost(self):
        r,f=fixture();p=L.build_packet(r,f,{})
        self.assertEqual(p['activation']['complete_ids'],[6,7])
        self.assertEqual(p['activation']['typed_partial_ids'],[1])
        self.assertEqual(p['activation']['legacy_partial_ids'],[0])
        self.assertEqual(p['activation']['legacy_e_trace_false_complete_ids'],[0])

    def test_later_tangent_augmentation_is_not_original_input(self):
        r,f=fixture();r['boundary_all'][0]['later_tangent_prediction']={'made_for_v15':True}
        p=L.build_packet(r,f,{})
        self.assertNotIn('later_tangent_prediction',p['solver_input']['boundary_observations'][0])

    def test_fallback_flag_does_not_make_actual_rgb_point(self):
        m={'source':'VISIBLE_BACKBONE_FALLBACK','image_measurement':True,'is_actual_image_observation':True}
        self.assertEqual(L.source_kind(m,{'corrected':False}),'inherited_predictor_point')

    def test_S_and_region_states_rejected_before_full_point_usage(self):
        r,_=fixture()
        for state in ('S','O','T','U'):
            states=deepcopy(r['spec']['states']);states[6]=state
            with self.assertRaisesRegex(ValueError,'cannot be point'):
                L.consumer_activation(r['spec']['visible'],r['boundary_typed'],states)

    def test_stored_component_disagreement_fails_closed(self):
        r,f=fixture();r['baseline']['solution']['objective_components']['removed_complete_BI_2D_likelihood_ids']=[1]
        with self.assertRaisesRegex(ValueError,'removal IDs'):
            L.build_packet(r,f,{})

    def test_E0_start_kept_separate_from_original_coarse_prior(self):
        r,f=fixture();r['baseline']['actual_pose']['centroid']=[.1,0.,5.]
        f['arms']['E_FROZEN']['coarse_pose']['centroid']=[.1,0.,5.]
        p=L.build_packet(r,f,{})
        self.assertEqual(p['solver_input']['coarse_pose']['t'],[0.,0.,5.])
        self.assertEqual(p['e0_pose']['centroid'],[.1,0.,5.])

    def test_serialization_keeps_active_factor_view(self):
        r,f=fixture();p=L.build_packet(r,f,{})
        q=json.loads(json.dumps(p,allow_nan=False))
        self.assertEqual(L.active_factor_view(p),L.active_factor_view(q))
        self.assertEqual(L.digest(L.active_factor_view(q)),p['active_factor_view_sha256'])

    def test_S_readout_and_revoked_display_poison_leave_active_factors(self):
        r,f=fixture();p=L.build_packet(r,f,{});before=L.active_factor_view(p)
        p['display_model_projection']={'S2':[1e8,-1e8]}
        p['solver_input']['visible'][0]['xy']=[1e8,-1e8]
        p['solver_input']['boundary_observations'][0]['association_provenance']['normal_projected_display_xy']=[1e9,1e9]
        self.assertEqual(before,L.active_factor_view(p))

    def test_endpoint_and_sample_reversal_keep_dependency_group(self):
        ray={'endpoints':[0,4],'normal':[1.,0.],'offset':-5.,'observed_points':[[5.,1.],[5.,2.]],'observation_confidence':[.3,.7]}
        reverse=deepcopy(ray);reverse.update(endpoints=[4,0],normal=[-1.,0.],offset=5.)
        reverse['observed_points'].reverse();reverse['observation_confidence'].reverse()
        self.assertEqual(L.line_source_group('a'*64,ray),L.line_source_group('a'*64,reverse))

    def test_model_bijection_does_not_assert_semantic_owner(self):
        r,f=fixture();p=L.build_packet(r,f,{})
        self.assertEqual(p['source_group_semantic_owner_status'],'SEMANTIC_OWNER_UNVERIFIED')
        self.assertTrue(p['provenance']['original_e_native_channels_not_permuted'])


if __name__=='__main__':unittest.main()
