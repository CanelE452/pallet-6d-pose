"""Meaningful numerical contract checks using synthetic arrays only, no DEV/GT."""
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from . import fusion as F


def inputs():
    gray=np.zeros((480,640),np.float32)
    q0=np.tile([300.,200.],(9,1))
    qN=q0.copy();qN[:8]+=[2.,-1.]
    angle=np.arange(221)*2*np.pi/221
    d=np.zeros((222,2));d[:221]=np.stack([4*np.cos(angle),2*np.sin(angle)],1)
    logits=np.zeros((8,222))
    return gray,q0,qN,logits,d,.5,np.ones(9,bool)


class FusionContracts(unittest.TestCase):
    def test_zero_image_and_zero_equation_exact_N3(self):
        args=inputs()
        for method in F.METHODS:
            q,diag=F.joint_refine(*args,method=method)
            self.assertTrue(np.array_equal(q,args[2]))
            self.assertEqual(diag['status_counts'],{'flat_gradient':8})
            self.assertFalse(diag['cap_active'].any())
        mu=np.array([327.183719,-45.974348])
        q,diag=F.joint_solve(mu,np.array([[4.,1.],[1.,3.]]),np.zeros((2,2)),np.zeros(2))
        self.assertTrue(np.array_equal(q,mu))

    def test_rank_one_edge_prior_supplies_SPD_tangent(self):
        gray,q0,qN,z,d,gain,support=inputs()
        gray[:,301:]=1
        gradients=F.prepare_gradients(gray)
        A,b,S=F.image_normal_equation(gradients,qN[0])
        self.assertGreater(S,F.FLAT_S_MIN)
        self.assertAlmostEqual(np.trace(A),1.,places=12)
        self.assertAlmostEqual(np.linalg.det(A),0.,places=12)
        self.assertEqual(A[1,1],0.)
        out,diag=F.joint_refine(gray,q0,qN,z,d,gain,support)
        self.assertTrue(np.isfinite(out).all())
        self.assertAlmostEqual(out[0,1],qN[0,1],places=11)
        self.assertNotEqual(out[0,0],qN[0,0])
        self.assertGreater(min(diag['corner_records'][0]['joint_system_eigenvalues']),0)

    def test_pixel_center_remap_and_boundary_reflection(self):
        gray=np.linspace(0,1,640,dtype=np.float32)[None].repeat(480,0)
        gradients=F.prepare_gradients(gray)
        gx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3,borderType=cv2.BORDER_REFLECT_101)
        gy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3,borderType=cv2.BORDER_REFLECT_101)
        np.testing.assert_array_equal(gradients['gx'],gx)
        for center in ([320.,240.],[.25,.5],[639.,479.]):
            center=np.asarray(center);A,b,S=F.image_normal_equation(gradients,center)
            locations=center+F.OFFSETS
            sampled=np.stack([cv2.remap(v,locations[:,0].astype(np.float32).reshape(11,11),
                locations[:,1].astype(np.float32).reshape(11,11),cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101).ravel() for v in (gx,gy)],1).astype(float)
            expectedS=0.;expectedA=np.zeros((2,2));expectedb=np.zeros(2)
            for g,x,w in zip(sampled,locations,F.SPATIAL_WEIGHTS):
                expectedS+=w*float(g@g)
                tensor=w*np.outer(g,g);expectedA+=tensor;expectedb+=tensor@x
            self.assertAlmostEqual(S,expectedS,places=11)
            np.testing.assert_allclose(A,expectedA/expectedS,atol=1e-12,rtol=0)
            np.testing.assert_allclose(b,expectedb/expectedS,atol=1e-10,rtol=0)
        # Exact constant derivatives isolate the pixel-center convention from
        # float32 ramp quantization, which makes neighboring Sobel values vary.
        constant=dict(raw_hw=gray.shape,gx=np.ones_like(gray),gy=np.zeros_like(gray))
        A,b,S=F.image_normal_equation(constant,np.array([320.,240.]))
        self.assertAlmostEqual(b[0],320.,places=12)
        with self.assertRaisesRegex(ValueError,'outside_initial'):
            F.image_normal_equation(gradients,[-.01,10])

    def test_null_only_and_high_entropy_covariance_and_native_gain(self):
        _,_,_,z,d,gain,_=inputs()
        null=np.full(222,-1000.);null[-1]=0.
        Sigma,meta=F.posterior_covariance(null,d/gain,1.,8.)
        np.testing.assert_array_equal(Sigma,np.eye(2))
        np.testing.assert_array_equal(meta['C'],np.zeros((2,2)))
        self.assertEqual(meta['null_probability'],1.)
        self.assertEqual(meta['probability_sum'],1.)
        self.assertEqual(meta['posterior_entropy'],0.)
        Sigma,meta=F.posterior_covariance(z[0],d/gain,1.,8.)
        self.assertAlmostEqual(meta['posterior_entropy'],np.log(222),places=12)
        self.assertAlmostEqual(meta['null_probability'],1/222,places=15)
        self.assertGreater(np.min(np.linalg.eigvalsh(Sigma)),1)
        _,native1=F.posterior_covariance(z[0],d,1.,8.)
        np.testing.assert_allclose(meta['C'],native1['C']/gain**2,atol=1e-12)
        huge=d*10000
        clipped,_=F.posterior_covariance(z[0],huge,1.,8.)
        np.testing.assert_allclose(np.linalg.eigvalsh(clipped),[64.,64.],atol=1e-10)

    def test_center_missing_unsupported_and_original_Base_cap(self):
        args=list(inputs());args[1][0]=[-1,-1];args[2][0]=[-1,-1]
        args[1][1]=[np.nan,np.nan];args[2][1]=[np.nan,np.nan]
        args[6][2]=False
        args[2][3]=args[1][3]+[30,0]
        model_support=np.ones(8,bool);model_support[4]=False
        q,diag=F.joint_refine(*args,point_support=model_support)
        for k in (0,1,2,4,8):
            self.assertTrue(np.array_equal(q[k],args[1][k],equal_nan=True))
        self.assertAlmostEqual(np.linalg.norm(q[3]-args[1][3]),8.,places=12)
        self.assertTrue(diag['cap_active'][3])
        self.assertEqual(diag['corner_records'][0]['status'],'missing_initial_sentinel')
        self.assertEqual(diag['corner_records'][1]['status'],'nonfinite_initial')

    def test_no_cornerSubPix_or_inverse_calls(self):
        args=list(inputs());args[0][:,301:]=1
        with patch.object(cv2,'cornerSubPix',side_effect=AssertionError('serial forbidden')),\
             patch.object(np.linalg,'inv',side_effect=AssertionError('explicit inverse forbidden')):
            for method in F.METHODS:
                q,diag=F.joint_refine(*args,method=method)
                self.assertTrue(np.isfinite(q).all())
                self.assertEqual(diag['cornerSubPix_calls'],0)
                self.assertEqual(diag['status_counts'],{'joint_solved':8})

    def test_isotropic_changes_only_Sigma_and_corresponding_solve(self):
        args=list(inputs());args[0][:,301:]=1
        qp,dp=F.joint_refine(*args,method='FG_JOINT_POSTERIOR')
        qi,di=F.joint_refine(*args,method='JOINT_FIXED_ISOTROPIC')
        for k,(p,i) in enumerate(zip(dp['corner_records'],di['corner_records'])):
            for field in ('A','b','S','C','C_eigenvalues_px2','fixed_window_center',
                          'candidate_mean_native_px','null_probability','posterior_entropy'):
                np.testing.assert_array_equal(p[field],i[field])
            np.testing.assert_array_equal(i['Sigma'],np.eye(2)*16)
            expected,_=F.joint_solve(args[2][k],np.eye(2)*16,p['A'],p['b'])
            np.testing.assert_array_equal(expected,di['q_unconstrained'][k])
        self.assertFalse(np.array_equal(qp[:8],qi[:8]))

    def test_invalid_inputs_and_outside_image_fallback_not_silent_success(self):
        args=list(inputs());args[2][0]=[-2,200];args[1][0]=[-3,200]
        args[3][1,3]=np.nan
        q,diag=F.joint_refine(*args)
        self.assertTrue(np.array_equal(q[0],args[2][0]))
        self.assertEqual(diag['corner_records'][0]['fallback_reason'],'outside_initial')
        self.assertTrue(np.array_equal(q[1],args[2][1]))
        self.assertEqual(diag['corner_records'][1]['fallback_reason'],'nonfinite_posterior')
        with self.assertRaisesRegex(ValueError,'invalid_prior_spd'):
            F.joint_solve([1,2],np.array([[1,0],[0,0]]),np.eye(2),np.ones(2))


if __name__=='__main__':unittest.main(verbosity=2)
