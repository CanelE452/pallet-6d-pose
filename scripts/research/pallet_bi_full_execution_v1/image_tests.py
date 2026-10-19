"""Constructed registration, Jacobian, dimensionality and image-path checks.

Every pose in this module is an authored mathematical fixture. No current real
or original synthetic pose reference is opened or used.
"""
from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
import unittest

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

from .material_assets import axis_registration,project,unproject,VisibleRenderer
from .material_support import compile_factors,generic_residuals,_policy
from .render_refine import choose_starts,optimize_factors,_image_update,old_point_line_factors,retain_nonoverlapping_original


def fixture():
    xyz=np.array([[x,y,z]for z in (-.65,.65)for y in (-.06,.06)for x in (-.55,.55)])
    R=Rotation.from_euler('xyz',[15.,-8.,3.],degrees=True).as_matrix()
    t=np.array([.03,-.05,3.])
    K=np.array([[608.,0,320],[0,608.,240],[0,0,1.]])
    p=dict(available=True,R_physical=R.tolist(),R_cf=R.tolist(),centroid=t.tolist(),cf_extents=[1.1,.12,1.3])
    frame=dict(id='CONSTRUCTED_ONLY',native_xyz=xyz.tolist(),initial_pose=p,K=K.tolist(),raw_hw=[480,640])
    return frame,xyz,R,t,K


class ConstructedTests(unittest.TestCase):
    def test_01_axis_rank_proper_affine(self):
        vertices=np.array([[x,y,z]for x in (-2.,2.)for y in (-.2,.2)for z in (-3.,3.)])
        for dimensions in ([1.1,.11,1.3],[.8,.14,.59]):
            unit,meta=axis_registration(vertices,dimensions)
            A=np.asarray(meta['source_to_object_affine'])
            transformed=(np.c_[vertices,np.ones(len(vertices))]@A.T)[:,:3]
            np.testing.assert_allclose(unit*np.asarray(dimensions),transformed,atol=1e-12)
            np.testing.assert_allclose(np.ptp(transformed,axis=0),dimensions,atol=1e-12)
            self.assertGreater(np.linalg.det(meta['proper_axis_rotation']),.999999)

    def test_02_general_support_projection_roundtrip(self):
        _,xyz,R,t,K=fixture()
        points=np.vstack([xyz,[[.231,.012,-.173],[-.121,-.011,.283]]])
        q,z=project(points,R,t,K)
        np.testing.assert_allclose(unproject(q,z,R,t,K),points,atol=2e-14)

    def test_03_physical_starts_basis_alias(self):
        frame,_,R,t,_=fixture()
        erun=dict(candidates=[dict(candidate_id='E0',source='E0',R_physical=R.tolist(),centroid=t.tolist())])
        pf=deepcopy(frame['initial_pose']);pf['centroid'][0]+=.01
        starts,audit=choose_starts(frame,erun,[dict(method='POSEFIX_SUBPIX_PF_ANCHOR',actual_pose=pf)])
        self.assertEqual(len(starts),2)
        self.assertEqual(starts[0]['source_aliases'],['N0','E0'])
        self.assertIn('SAME_PHYSICAL_POSE_ALIAS',[r['status']for r in audit])

    def test_04_analytic_mixture_jacobian(self):
        frame,xyz,R,t,K=fixture();q,_=project(xyz,R,t,K)
        factors=[]
        for i in range(3):
            factors.append(dict(observation_dimension=2,object_xyz=xyz[i].tolist(),
                xy=(q[i]+[.7,-.3]).tolist(),effective_covariance_px2=[[5.,1.],[1.,4.]]))
        factors.append(dict(observation_dimension=1,object_xyz=[.123,.017,.273],xy=[0.,0.],
            normal_unit_xy=[.6,.8],offset_px=-400.,effective_variance_px2=5.))
        c=compile_factors(factors);x=np.r_[cv2.Rodrigues(R)[0].ravel(),t]
        _,j=cv2.projectPoints(c['xyz'],x[:3],x[3:],K,None)
        residual,jac=generic_residuals(c,R,t,K,projection_jacobian=j[:,:6].reshape(-1,2,6))
        numerical=[]
        for k in range(6):
            plus=x.copy();minus=x.copy();plus[k]+=1e-6;minus[k]-=1e-6
            a=generic_residuals(c,cv2.Rodrigues(plus[:3])[0],plus[3:],K)
            b=generic_residuals(c,cv2.Rodrigues(minus[:3])[0],minus[3:],K)
            numerical.append((a-b)/2e-6)
        np.testing.assert_allclose(jac,np.asarray(numerical).T,atol=2e-5,rtol=2e-5)

    def test_05_generic_pose_update(self):
        frame,xyz,R,t,K=fixture();q,_=project(xyz,R,t,K)
        factors=[dict(observation_dimension=2,object_xyz=p.tolist(),xy=xy.tolist(),
            effective_covariance_px2=[[4.,0.],[0.,4.]])for p,xy in zip(xyz,q)]
        p=deepcopy(frame['initial_pose']);p['centroid'][0]+=.01;p['centroid'][1]-=.01
        result,audit=optimize_factors(frame,p,factors)
        self.assertTrue(audit['proposal_accepted'])
        self.assertLess(np.linalg.norm(np.asarray(result['centroid'])-t),1e-5)
        self.assertLessEqual(audit['objective_calls'],10)

    def test_06_role_does_not_force_dimension(self):
        frame,_,R,t,K=fixture()
        vertices=np.array([[0.,0.,0.],[.2,0.,0.],[0.,.2,0.]])
        q,_=project(vertices,R,t,K)
        records=[]
        for i,j in [(0,1),(0,2)]:
            direction=q[j]-q[i];direction/=np.linalg.norm(direction)
            n=np.array([-direction[1],direction[0]])
            cue=dict(normal_unit_xy=n.tolist(),offset_px=-float(n@q[0]),xy=q[0].tolist(),
                normal_variance_px2=4.,conditional_variance_px2=4.,effective_variance_px2=4.,
                physical_source_ids=[f'INDEPENDENT_DIRECTION_{j}'],normalized_cue_cost=0.,cue='RGB_FULL')
            records.append(dict(support_id=f'EDGE/{j}',cue_candidates={'RGB_FULL':cue},
                role='B',object_segment_xyz=vertices[[i,j]].tolist(),vertex_ids=[i,j],
                visibility_source='CONSTRUCTED',owner_evidence_status='CONSTRUCTED',image_sha256='fixture'))
        packet=_policy(records,vertices,R,t,K,'BI')
        self.assertEqual(packet['evidence_dimensions'],{'2':1})
        packet=_policy(records[:1],vertices,R,t,K,'BI')
        self.assertEqual(packet['evidence_dimensions'],{'1':1})
        altered=deepcopy(records[:1]);masked=deepcopy(altered[0]['cue_candidates']['RGB_FULL'])
        masked['normalized_cue_cost']=2.;masked['cue']='MASK_TIGHT'
        altered[0]['cue_candidates']['MASK_TIGHT']=masked
        common=_policy(altered,vertices,R,t,K,'COMMON');bi=_policy(altered,vertices,R,t,K,'BI')
        self.assertEqual(common['factors'][0]['cue'],'RGB_FULL');self.assertEqual(bi['factors'][0]['cue'],'MASK_TIGHT')

    def test_07_renderer_integer_pixel_depth(self):
        mesh=trimesh.Trimesh([[-.5,-.5,0],[.5,-.5,0],[.5,.5,0],[-.5,.5,0]],
            [[0,2,1],[0,3,2]],process=False)
        renderer=VisibleRenderer()
        try:
            R=Rotation.from_euler('xyz',[20,10,0],degrees=True).as_matrix()
            t=np.array([0.,0.,3.]);K=np.array([[608.,0,320],[0,608.,240],[0,0,1.]])
            _,depth,mask=renderer.render(SimpleNamespace(render_meshes=[mesh]),[1.,1.,1.],R,t,K,[480,640])
            yy,xx=np.where(mask);self.assertGreater(len(xx),1000)
            rays=np.c_[xx,yy,np.ones(len(xx))]@np.linalg.inv(K).T
            normal=R[:,2];expected=(normal@t)/(rays@normal)
            self.assertLess(np.max(abs(depth[yy,xx]-expected)),2e-5)
            self.assertEqual(renderer.raster_passes,2)
        finally:renderer.close()

    def test_08_whole_image_correspondence_is_pose_update(self):
        frame,_,R,t,K=fixture();h,w=frame['raw_hw']
        yy,xx=np.mgrid[:h,:w]
        gray=np.clip(100+55*np.sin(xx*.1)+35*np.cos(yy*.13),0,255).astype(np.uint8)
        rgb=cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
        depth=np.zeros((h,w),np.float32);depth[120:360,180:460]=3.
        mask=depth>0
        result,audit=_image_update(frame,frame['initial_pose'],rgb,depth,mask,gray,mask.astype(np.float32))
        self.assertEqual(audit['DIS_calls'],2)
        self.assertEqual(audit['PnP_calls'],1)
        self.assertTrue(audit['depth_recovered_object_support'])
        self.assertTrue(result['available'])
        self.assertIsNone(audit['native_crop_transform'])

    def test_09_old_normal_uses_actual_support_and_weight(self):
        frame,xyz,R,t,K=fixture();frame['image_sha256']='CONSTRUCTED_RGB'
        line=dict(normal_unit_xy=[0.,1.],offset_px=-100.,normal_variance_px2=16.,
            provenance=dict(actual_RGB_line_id=7,line_record=dict(observed_points=[[100.,100.],[110.,100.],[120.,100.]],
                inlier_mask=[True,True,False])))
        missing=dict(normal_unit_xy=[1.,0.],offset_px=-40.,normal_variance_px2=9.,provenance={})
        erun=dict(families=dict(ORIGINAL_WD=dict(native_xyz=xyz.tolist(),physical_registration_Q=np.eye(3).tolist())),
            packet=dict(solver_input=dict(visible=[],boundary_observations=[dict(native_corner=0,lines=[line],fit_reliability_weight=.25),
                dict(native_corner=1,lines=[missing],fit_reliability_weight=1.)])))
        original,_=old_point_line_factors(frame,erun)
        np.testing.assert_equal(original[0]['raw_image_support_xy'],[[100.,100.],[110.,100.]])
        self.assertEqual(original[0]['effective_variance_px2'],64.)
        self.assertFalse(original[0]['display_coordinate_is_full_2D_measurement'])
        self.assertEqual(original[1]['xy'],[None,None]);self.assertEqual(original[1]['physical_source_ids'],[])
        generic=[dict(observation_dimension=1,xy=[105.,102.],raw_image_support_xy=[[105.,102.]])]
        kept=retain_nonoverlapping_original(original,generic)
        self.assertEqual(len(kept),1);self.assertEqual(kept[0]['correlation_with_generic'],'UNKNOWN_NO_OLD_RAW_SUPPORT')
        self.assertTrue(np.isfinite(generic_residuals(compile_factors(original),R,t,K)).all())


if __name__=='__main__':unittest.main()
