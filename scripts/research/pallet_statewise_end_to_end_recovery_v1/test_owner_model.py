"""A's new scientific invariants; synthetic geometry only, no evaluation IO."""
import math
import unittest

import numpy as np
from scipy.integrate import quad

from . import owner_model as A


XYZ = np.array([[-.55,-.075,-.65],[.55,-.075,-.65],[.55,.075,-.65],[-.55,.075,-.65],
                [-.55,-.075,.65],[.55,-.075,.65],[.55,.075,.65],[-.55,.075,.65]])
K = np.array([[600.,0.,320.],[0.,600.,240.],[0.,0.,1.]])
Q, _ = A.project_native(XYZ, np.eye(3), [0.,0.,3.], K)


def observation(k, xy=None, variance=4.):
    return dict(native_corner=k,state='B',xy=Q[k].tolist() if xy is None else list(xy),
                covariance_px2=(np.eye(2)*variance).tolist(),fit_reliability_weight=.9,
                source='SYNTHETIC_EXACT_COORDINATE_TECHNICAL_TEST')


def compile_model(measurements=None,boundary=None,**kwargs):
    return A.compile_owner_groups(XYZ,K,[observation(k) for k in range(8)] if measurements is None else measurements,
                                   boundary or [],(480,640),none_prior=.05,**kwargs)


class OwnerModelTests(unittest.TestCase):
    def test_global_permutation_is_proper_bijective_and_preserves_incidence(self):
        perms=A.upright_owner_permutations(XYZ)
        self.assertEqual(len(set(perms)),4)
        edges={tuple(sorted((i,j))) for i in range(8) for j in range(i+1,8)
               if np.count_nonzero(XYZ[i]!=XYZ[j])==1}
        signs=np.sign(XYZ)
        for p in perms:
            self.assertEqual(set(p),set(range(8)))
            self.assertEqual({tuple(sorted((p[i],p[j]))) for i,j in edges},edges)
            self.assertTrue(np.array_equal(signs[:,1],signs[list(p),1]))
            rotation=np.linalg.lstsq(signs,signs[list(p)],rcond=None)[0]
            self.assertAlmostEqual(np.linalg.det(rotation),1.)
        reflection=(1,0,3,2,5,4,7,6)
        with self.assertRaises(ValueError):A.validate_owner_permutation(XYZ,reflection)

    def test_actual_different_global_owner_can_win_not_same_edge_alternative(self):
        p=A.upright_owner_permutations(XYZ)[1]
        model=compile_model([observation(k,Q[p[k]]) for k in range(8)])
        values=A.energies_per_global_h(model,Q)
        self.assertEqual(int(np.argmin(values)),1)
        self.assertAlmostEqual(values[1],0.,places=10)
        self.assertGreater(values[0],1.)
        self.assertNotEqual(p[0],0)

    def test_duplicate_hypotheses_and_measurements_cannot_multiply_prior_information(self):
        model=compile_model()
        duplicate=compile_model([observation(k) for k in range(8)]*3,
                                permutations=list(model['permutations'])*4)
        self.assertEqual(len(duplicate['groups']),8)
        self.assertEqual(len(duplicate['permutations']),4)
        np.testing.assert_allclose(A.energy(model,Q),A.energy(duplicate,Q),atol=0,rtol=0)

    def test_normalizer_is_exact_for_original_pseudo_huber(self):
        for d in (1,2,3):
            sphere=2.*math.pi**(d/2.)/math.gamma(d/2.)
            integral=sphere*quad(lambda r:math.exp(-.5*float(A.pseudo_huber_squared(r*r)))*r**(d-1),0.,np.inf)[0]
            self.assertAlmostEqual(math.log(integral),A.log_radial_normalizer(d),places=10)

    def test_covariance_volume_and_none_does_not_turn_unsupported_into_zero_cost(self):
        model=compile_model()
        wrong=Q+10000.
        bad, audit=A.energy(model,wrong,return_audit=True)
        good=A.energy(model,Q)
        self.assertGreater(bad,good)
        self.assertGreater(bad,0.)
        self.assertTrue(np.allclose(audit['per_h_total_energies'],audit['none_constant_energy_limit_per_h']))
        narrow=compile_model([observation(0,variance=1.)])['groups'][0]
        broad=compile_model([observation(0,variance=100.)])['groups'][0]
        self.assertLess(broad['log_inlier_peak'],narrow['log_inlier_peak'])

    def test_same_observation_set_every_h_and_no_free_hypothesis_deletion(self):
        model=compile_model()
        _,_,a=A.energies_per_global_h(model,Q,return_audit=True)
        self.assertEqual(a['same_group_count_every_h'],8)
        penalty=A.energy(model,Q,extra_h_energies=[0.,math.inf,math.inf,math.inf])
        self.assertGreaterEqual(penalty,A.energy(model,Q))
        with self.assertRaises(TypeError):A.energy(model,Q,exclude_group_ids=[{'full:0'},{'full:1'}])

    def test_partial_revocation_never_promotes_tangent_to_precise_full_point(self):
        partial=dict(native_corner=0,state='B',fit_reliability_weight=.8,
                     lines=[dict(normal_unit_xy=[1.,0.],offset_px=-Q[0,0],normal_variance_px2=4.,
                                 support_tangent_interval_px=[0.,480.])])
        model=compile_model([observation(0)],[partial])
        self.assertEqual(model['global_observation_ids'],('partial:0',))
        p=A.upright_owner_permutations(XYZ)[0]
        single=A.compile_owner_groups(XYZ,K,[observation(0)],[partial],(480,640),none_prior=.05,permutations=[p])
        shifted=Q.copy();shifted[0,1]+=30.
        self.assertEqual(A.energy(single,Q),A.energy(single,shifted))
        with self.assertRaises(ValueError):compile_model([{**observation(0),'measurement_is_pose_projection':True}])

    def test_projection_gradient_matches_finite_difference(self):
        model=compile_model([observation(k) for k in (0,2,4,6)])
        q=Q+np.array([1.3,-.7])
        _,gradient=A.energy(model,q,return_gradient=True)
        numerical=np.zeros_like(q);eps=1e-5
        for i in range(8):
            for j in range(2):
                plus=q.copy();minus=q.copy();plus[i,j]+=eps;minus[i,j]-=eps
                numerical[i,j]=(A.energy(model,plus)-A.energy(model,minus))/(2*eps)
        np.testing.assert_allclose(gradient,numerical,atol=1e-7,rtol=1e-7)

    def test_resolution_conversion_preserves_conditional_energy(self):
        model=compile_model([observation(k) for k in range(4)])
        scale=2.;m=[observation(k) for k in range(4)]
        for item in m:
            item['xy']=(np.array(item['xy'])*scale).tolist()
            item['covariance_px2']=(np.array(item['covariance_px2'])*scale**2).tolist()
        k=K.copy();k[:2]*=scale
        resized=A.compile_owner_groups(XYZ,k,m,[],(960,1280),none_prior=.05)
        np.testing.assert_allclose(A.energy(model,Q+1.2),A.energy(resized,(Q+1.2)*scale),atol=1e-10,rtol=0)

    def test_hard_domain_owner_transport_preserves_pixel_payload_and_count(self):
        p=A.upright_owner_permutations(XYZ)[1]
        domain=[dict(native_corner=0,state='O',center_xy=[13.,17.],half_axes_px=[8.,6.])]
        mapped=A.map_owner_records(domain,p,XYZ)
        self.assertEqual(mapped[0]['native_corner'],p[0]);self.assertEqual(len(mapped),len(domain))
        self.assertEqual(mapped[0]['center_xy'],domain[0]['center_xy'])
        self.assertEqual(domain[0]['native_corner'],0)

    def test_prepared_geometry_normals_are_rebuilt_at_new_owner(self):
        p=A.upright_owner_permutations(XYZ)[1]
        record=[dict(native_corner=0,object_xyz=XYZ[0].copy(),
                     incident_normals=np.diag(np.sign(XYZ[0])),xy=[10.,20.])]
        mapped=A.map_owner_records(record,p,XYZ)[0]
        np.testing.assert_array_equal(mapped['object_xyz'],XYZ[p[0]])
        np.testing.assert_array_equal(mapped['incident_normals'],np.diag(np.sign(XYZ[p[0]])))
        self.assertEqual(mapped['xy'],[10.,20.])

    def test_partial_projection_gradient_matches_finite_difference(self):
        partial=dict(native_corner=0,state='B',fit_reliability_weight=.8,lines=[
            dict(normal_unit_xy=[1.,0.],offset_px=-Q[0,0],normal_variance_px2=4.),
            dict(normal_unit_xy=[.6,.8],offset_px=-float(np.array([.6,.8])@Q[0]),normal_variance_px2=9.)])
        model=compile_model([], [partial]);q=Q+np.array([.7,-.3])
        _,gradient=A.energy(model,q,return_gradient=True)
        numerical=np.zeros_like(q);eps=1e-5
        for i in range(8):
            for j in range(2):
                plus=q.copy();minus=q.copy();plus[i,j]+=eps;minus[i,j]-=eps
                numerical[i,j]=(A.energy(model,plus)-A.energy(model,minus))/(2*eps)
        np.testing.assert_allclose(gradient,numerical,atol=1e-7,rtol=1e-7)

    def test_no_input_identity(self):
        model=compile_model([], [])
        value,gradient=A.energy(model,Q,return_gradient=True)
        self.assertEqual(value,0.);np.testing.assert_array_equal(gradient,np.zeros((8,2)))

    def test_AB_extra_energy_enters_before_global_sum_not_after(self):
        model=compile_model([observation(0)])
        values=A.energies_per_global_h(model,Q)
        extra=np.array([4.,0.,2.,5.])
        expected=-2.*math.log(sum(math.exp(-.5*(v+b))/4. for v,b in zip(values,extra)))
        self.assertAlmostEqual(A.energy(model,Q,extra_h_energies=extra),expected,places=12)


if __name__=='__main__':unittest.main()
