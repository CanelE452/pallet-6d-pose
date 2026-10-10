"""Small synthetic incidence-guard contracts; root freezes before executing.

These tests construct explicit selected-candidate packets with the unchanged
V13 forward geometry. They do not claim optimizer convergence or global pose
uniqueness. No real image, trained head, GT dataset or checkpoint is opened.
"""
import copy
import unittest
from unittest.mock import patch
import cv2
import numpy as np

from . import pipeline as G
from ..pallet_finite_edge_consensus_repair_20261011_v13 import solver as S
from ..pallet_finite_edge_consensus_repair_20261011_v13.test_solver import fixture


def selected_fixture(ids=(0,1,4), edges=(2,), *, fractions=(.2,.5,.8),
                     hidden=(6,7), dimension=0, xyz=None):
    """Controlled original NEW packet; no least_squares/PnP dispatch occurs."""
    data=fixture(square=dimension==0, fractions=fractions)
    if xyz is not None:data['xyz']=np.asarray(xyz,float)
    bank=S.FiniteEdgeConsensusBank(data['points'],data['K'],data['xyz'])
    model=bank.bank.models[dimension];z=data['z'];q,_=S.V5.projection_jacobian(model,z,data['K'])
    points=np.full((9,2),np.nan);points[list(ids)]=q[list(ids)];points[8]=data['points'][8]
    # H coordinates are deliberately finite and wrong; they are never fit IDs.
    points[list(hidden)]=[19.,29.]
    bank=S.FiniteEdgeConsensusBank(points,data['K'],data['xyz'])
    raw=[]
    for edge in edges:
        a,b=S.EDGES[edge];direction=q[b]-q[a];normal=np.array([-direction[1],direction[0]])/np.linalg.norm(direction)
        support=[(q[a]+f*direction).tolist() for f in fractions]
        raw.append(dict(edge=edge,endpoints=[a,b],normal=normal.tolist(),offset=float(normal@q[a]),
            queries=[edge*7+i for i in range(len(fractions))],support_points=support,
            query_radii_px=[1.]*len(fractions)))
    lines=bank.factor_validator._lines(dict(corners=[],lines=raw,queries=[]),(),set(ids))[0]
    geometry,_,projected=bank._geometry(model,z,ids,lines)
    Rcf=cv2.Rodrigues(z[:3])[0];Q=np.eye(3) if dimension==0 else np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
    candidate=dict(accepted=True,dimension_index=dimension,final_z=z.tolist(),
        cf_extents=bank.bank.dims[dimension].tolist(),fit_input_ids=list(ids),fit_line_edges=list(edges),
        point_line_geometry=geometry,fixture_origin='controlled selected candidate; no optimizer convergence claim')
    display=np.vstack((q,data['points'][8]));display[list(hidden)]=projected[list(hidden)]
    solved=dict(available=True,pose_available=True,new_pose_estimated=True,no_pose=False,
        state='FINITE_EDGE_CONSENSUS_NEW',output_status='NEW_POSE',
        local_refinement_estimated=True,R_cf=Rcf.tolist(),R_physical=(Rcf@Q).tolist(),
        centroid=z[3:].tolist(),cf_extents=bank.bank.dims[dimension].tolist(),projected=projected.tolist(),
        fit_input_ids=list(ids),fit_line_edges=list(edges),used=list(ids),line_edges=list(edges),
        consumed_edges=[],hidden=list(hidden),input_inliers=list(ids),final_inliers=list(ids),
        inliers=list(ids),final_line_inliers=list(edges),point_inlier_count=len(ids),
        unique_line_inlier_count=len(edges),total_inlier_factor_count=len(ids)+len(edges),
        actual_source_lines=[dict(edge=l['edge'],normal=l['normal'].tolist(),offset=l['offset']) for l in lines],
        point_line_geometry=copy.deepcopy(geometry),best_candidate=candidate,
        all_candidate_solutions=[copy.deepcopy(candidate)],points_final=display.tolist(),
        hidden_reprojected=bool(hidden),reprojected_points_reused_as_observations=False,
        operation_counts=dict(local_optimizer_calls=0,fixture_forward_geometry_only=True))
    native=np.vstack((q+[1.,-1.],data['points'][8]));initial=dict(available=True,
        R_cf=Rcf.tolist(),R_physical=(Rcf@Q).tolist(),centroid=z[3:].tolist(),
        cf_extents=bank.bank.dims[dimension].tolist())
    row=dict(id='synthetic:stable-incidence',method=G.PARENT_METHOD,solver=solved,
        input_points=points.tolist(),K=data['K'].tolist(),xyz=data['xyz'].tolist(),
        hidden_initial=list(hidden),original_native_N3_points=native.tolist(),initial_pose=initial,
        observation_contract=dict(validated_boundary_corner_ids=list(ids),hybrid_boundary_corner_ids=[]),
        native_points=display.tolist(),actual_pose=copy.deepcopy(solved),output_status='NEW_POSE',
        pose_available=True,new_pose_estimated=True,fallback_used=False,no_pose=False,
        hidden_reprojected=bool(hidden),reprojected_ids=list(hidden),
        output_coordinate_sources=['CONTROLLED_ORIGINAL_DISPLAY']*9,
        reprojections_reused_as_observations=False)
    return row


class StableIncidenceContract(unittest.TestCase):
    def test_two_lines_have_only_four_incidence_rows_despite_outside_hinges(self):
        row=selected_fixture(ids=(),edges=(2,4),fractions=(-.01,.5,1.01))
        blocks=row['solver']['best_candidate']['point_line_geometry']['edge_blocks']
        self.assertTrue(all('BEFORE_FIRST_ENDPOINT' in b['hinge_branches'] and 'AFTER_SECOND_ENDPOINT' in b['hinge_branches'] for b in blocks))
        result,w=G.guard_row(row)
        self.assertEqual(w['incidence_scalar_rows'],4);self.assertEqual(w['removed_longitudinal_rows'],6)
        self.assertFalse(w['accepted']);self.assertEqual(w['guard_state'],'STABLE_INCIDENCE_INSUFFICIENT')
        self.assertLessEqual(w['stable_incidence_geometry']['observed_normal_joint']['numerical_rank'],4)
        self.assertLessEqual(w['stable_incidence_geometry']['modeled_normal_joint']['numerical_rank'],4)
        self.assertEqual(result['output_status'],'N3_BASELINE_FALLBACK')

    def test_three_actual_points_and_unused_line_keep_same_pose_and_display(self):
        row=selected_fixture();before=G.C.semantic_sha256(row)
        with patch.object(G,'assemble',side_effect=AssertionError('accepted projection must not be assigned again')):
            result,w=G.guard_row(row)
        self.assertTrue(w['accepted']);self.assertEqual(w['actual_fit_input_ids'],[0,1,4])
        self.assertEqual(w['actual_fit_line_edges'],[2]);self.assertTrue(set(w['actual_fit_input_ids']).isdisjoint({6,7,8}))
        self.assertEqual(result['native_points'],row['native_points']);self.assertEqual(result['solver']['centroid'],row['solver']['centroid'])
        self.assertEqual(result['reprojected_ids'],[6,7]);self.assertFalse(result['accepted_H_projection_assigned_again'])
        self.assertEqual(before,G.C.semantic_sha256(row));self.assertEqual(w['counts']['new_optimizer_calls'],0)
        self.assertEqual(w['counts']['new_PnP_calls'],0);self.assertEqual(w['counts']['geometry_projection_entries'],2)

    def test_H_coordinate_changes_and_normal_sign_do_not_change_guard(self):
        row=selected_fixture();changed=copy.deepcopy(row);changed['input_points'][6:8]=[[590.,450.],[580.,440.]]
        for line in changed['solver']['actual_source_lines']:
            line['normal']=[-v for v in line['normal']];line['offset']=-line['offset']
        geometry=changed['solver']['best_candidate']['point_line_geometry'];indices,_=G.normal_row_indices(geometry,[0,1,4],[2])
        # Reverse observed line rows too: this is the same geometric line.
        for k in indices[6:]:geometry['observed_normal_J'][k]=[-v for v in geometry['observed_normal_J'][k]]
        A,wa=G.guard_row(row);B,wb=G.guard_row(changed)
        self.assertTrue(wa['accepted'] and wb['accepted']);self.assertEqual(A['native_points'],B['native_points'])
        for key in ('observed_normal_joint','modeled_normal_joint'):
            np.testing.assert_allclose(wa['stable_incidence_geometry'][key]['singular_values'],wb['stable_incidence_geometry'][key]['singular_values'],atol=1e-12)

    def test_rank_six_weak_selected_candidate_declines_without_alt_reselection(self):
        row=selected_fixture();old=G.V5.geometry_info
        def controlled(*args,**kwargs):
            info=old(*args,**kwargs)
            for key in ('observed_normal_joint','modeled_normal_joint'):
                info[key].update(numerical_rank=6,singular_values=[1.,1.,1.,1.,1.,1e-6])
            return info
        original_candidates=G.C.semantic_sha256(row['solver']['all_candidate_solutions'])
        with patch.object(G.V5,'geometry_info',side_effect=controlled):result,w=G.guard_row(row)
        self.assertEqual(w['guard_state'],'STABLE_INCIDENCE_WEAK');self.assertFalse(result['new_pose_estimated'])
        self.assertEqual(G.C.semantic_sha256(result['solver']['all_candidate_solutions']),original_candidates)
        self.assertEqual(result['solver']['best_candidate'],row['solver']['best_candidate'])
        self.assertEqual(result['solver']['fit_input_ids'],[]);self.assertEqual(result['solver']['fit_line_edges'],[])
        self.assertEqual(result['native_points'].tolist(),row['original_native_N3_points'])
        self.assertEqual(result['actual_pose'],row['initial_pose']);self.assertEqual(result['reprojected_ids'],[])

    def test_unavailable_parent_skips_geometry_and_numerical_error_is_typed(self):
        row=selected_fixture();row['solver']['available']=False
        with patch.object(G.V5,'geometry_info',side_effect=AssertionError('unavailable parent has no guard fit')):
            result,w=G.guard_solver(row['solver'],row['input_points'],row['K'],row['xyz'],row['hidden_initial'])
        self.assertFalse(w['gate_applied']);self.assertEqual(w['counts']['geometry_info_calls'],0)
        self.assertFalse(result['available'])
        row=selected_fixture();before=G.C.semantic_sha256(row)
        with patch.object(G.V5,'geometry_info',side_effect=ValueError('explicit controlled forward failure')):
            result,w=G.guard_row(row)
        self.assertEqual(w['guard_state'],'STABLE_INCIDENCE_NUMERICAL_FAILURE');self.assertEqual(result['output_status'],'N3_BASELINE_FALLBACK')
        self.assertEqual(before,G.C.semantic_sha256(row))
        row['initial_pose']['available']=False
        with patch.object(G.V5,'geometry_info',side_effect=ValueError('explicit controlled forward failure')):result,_=G.guard_row(row)
        self.assertEqual(result['output_status'],'POSE_FAILURE');self.assertTrue(result['no_pose'])

    def test_consumed_or_H_fit_identity_is_rejected_before_geometry(self):
        for fault in ('consumed','hidden'):
            row=selected_fixture()
            if fault=='consumed':row['solver']['consumed_edges']=[2]
            else:row['hidden_initial']=[0,6,7]
            with self.subTest(fault=fault),patch.object(G.V5,'geometry_info',side_effect=AssertionError('invalid identity must fail before forward work')):
                with self.assertRaises(ValueError):G.guard_row(row)

    def test_registry_DW_and_original_near_square_enumeration(self):
        for row in (selected_fixture(ids=(0,1,4),edges=(),dimension=1),
                    selected_fixture(ids=(0,1,4),edges=(),xyz=[1.,.2,1.+.5e-9])):
            with self.subTest(xyz=row['xyz'],dim=row['solver']['best_candidate']['dimension_index']):
                result,w=G.guard_row(row);self.assertTrue(w['accepted'])
                self.assertEqual(result['solver']['R_physical'],row['solver']['R_physical'])
                self.assertEqual(w['selected_extents'],row['solver']['cf_extents'])


if __name__=='__main__':
    raise SystemExit('Root must freeze the synthetic CPU protocol before running these seven groups.')
