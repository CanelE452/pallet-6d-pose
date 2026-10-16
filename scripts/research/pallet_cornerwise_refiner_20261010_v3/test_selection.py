"""CPU-only synthetic checks for actual corner-held-out boundary selection.

Run once after the decision code and this test have been reviewed and frozen.
Actual OpenCV PnP/LM calls are counted. No images, model, GPU, source cache,
private GT, performance evaluation or training are used. Toy projections and
line supports are mathematical fixtures, not physical-boundary evidence.
"""
from __future__ import annotations

import argparse
import builtins
from collections import Counter
from contextlib import contextmanager
import copy
import hashlib
import inspect
import io
import json
from pathlib import Path
import sys
import traceback

import cv2
import numpy as np

from . import selection as S
from ..pallet_boundary_corner_refiner_20261010_v2 import pipeline as B
from ..pallet_boundary_corner_refiner_20261010_v2 import pose as P
from ..pallet_observation_refiner_20261009_v1.solver import Candidate, cuboid, project, visibility

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[3]
DOC = REPO/'_docs/experiments/pallet_cornerwise_refiner_20261010_v3'
EDGES = ((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))


def fixture(rvec=(.3,.25,.1), t=(0.,.1,4.)):
    K=np.array([[600.,0.,640.],[0.,600.,480.],[0.,0.,1.]])
    xyz=np.array([1.1,.11,1.3]);R=cv2.Rodrigues(np.asarray(rvec,float))[0];t=np.asarray(t,float)
    q=np.vstack([project(cuboid(*xyz),R,t,K),[641.25,481.75]])
    initial=dict(available=True,R_cf=R.tolist(),R_physical=R.tolist(),centroid=t.tolist(),
                 cf_extents=xyz.tolist(),selected_hypothesis='FIXED_SYNTHETIC_INITIAL')
    return K,xyz,q,initial


def observation(q, ids):
    """Incident straight-line support fixtures with exact observed intersections."""
    lines={};corners=[]
    for k in ids:
        incident=[e for e,(a,b) in enumerate(EDGES) if k in (a,b)][:2]
        assert len(incident)==2
        corners.append(dict(id=k,xy=np.asarray(q[k]).tolist(),edges=incident,radius_px=.5))
        for edge in incident:
            if edge in lines:continue
            a,b=EDGES[edge];pa,pb=np.asarray(q[a]),np.asarray(q[b])
            tangent=(pb-pa)/np.linalg.norm(pb-pa);normal=np.array([-tangent[1],tangent[0]])
            support=np.array([(1-u)*pa+u*pb for u in (.25,.5,.75)])
            lines[edge]=dict(edge=edge,normal=normal.tolist(),offset=float(normal@pa),
                support_points=support.tolist(),query_radii_px=[1.,1.,1.],
                queries=[edge*7+j for j in range(3)],support_length_px=float(np.linalg.norm(pb-pa)))
    return dict(corners=corners,lines=[lines[k] for k in sorted(lines)],queries=[],
                model_query_anchor='unchanged original Base; synthetic decoder-output fixture',
                fixed_metadata=dict(candidate_index=2,score=.73,class_id=0,source_RGB_changed=False))


def one_shifted_boundary(q, k, delta):
    # Shift both incident observed lines consistently; their actual intersection
    # differs from the held-out projection. No projection is supplied to select.
    shifted=np.asarray(q,float).copy();incident=[e for e,(a,b) in enumerate(EDGES) if k in (a,b)][:2]
    vertices={v for e in incident for v in EDGES[e]}
    for v in vertices:shifted[v]+=np.asarray(delta)
    return observation(shifted,[k])


class AuditedBank(P.PoseBank):
    """Use genuine solver methods while recording their consumed corner IDs."""
    def __init__(self,*args,**kw):
        self.calls=[];self.trace=[];self.requested_exclusions=None
        super().__init__(*args,**kw)

    def _generic(self,ids,dim,phase):
        self.trace.append(dict(kind='generic',phase=phase,ids=list(ids),dimension=dim,
                               requested_exclusions=self.requested_exclusions))
        return super()._generic(ids,dim,phase)

    def _lm(self,candidate,fit_ids):
        self.trace.append(dict(kind='LM',ids=list(fit_ids),
                               requested_exclusions=self.requested_exclusions))
        return super()._lm(candidate,fit_ids)

    def solve(self,initial_pose,hidden=(),robust=True):
        before=len(self.trace);self.requested_exclusions=sorted(set(hidden))
        answer=super().solve(initial_pose,hidden=hidden,robust=robust)
        consumed=self.trace[before:]
        # The common bank enumerates all subsets once, even when the first
        # requested mask excludes k. Only selected generators/refits consume a
        # mask's allowed IDs; cached global subset construction is not its fit.
        for entry in consumed:
            if entry['kind']=='LM' or entry.get('phase')!='subset':
                assert not set(entry['ids']) & set(hidden),entry
        assert not set(answer['fit_input_ids']) & set(hidden),answer['fit_input_ids']
        if answer.get('available'):
            assert not set(answer['generator_ids']) & set(hidden)
        self.calls.append(dict(hidden=sorted(set(hidden)),robust=robust,answer=copy.deepcopy(answer),
                               actual_refit_trace=[r for r in consumed if r['kind']=='LM' or r.get('phase')!='subset']))
        self.requested_exclusions=None
        return answer


@contextmanager
def no_truth_or_model_reads():
    old=(builtins.open,io.open,Path.open);attempts=[]
    tokens=('TARGETS','READY_SOURCE','GT_CACHE','STATIC_VISIBILITY','GROUND_TRUTH','INPUTS.JSON',
            'OBSERVATIONS.JSON','PREDICTIONS.JSON','_LABEL.JSON','MASK_AMODAL','MASK_VISIBLE',
            'FEATURES.NPY','BEST.PT','LAST.PT','IMAGE_ROLE.PT')
    def guarded(fn):
        def call(path,*args,**kw):
            if isinstance(path,(str,Path)) and any(t in str(path).upper() for t in tokens):
                attempts.append(str(path));raise AssertionError('forbidden truth/model/cache read')
            return fn(path,*args,**kw)
        return call
    builtins.open,io.open,Path.open=map(guarded,old)
    try:yield attempts
    finally:builtins.open,io.open,Path.open=old


def assert_native_output(selected,native,contract,obs,H):
    assert np.array_equal(native[8],selected[8],equal_nan=True)
    actual={c['id']:np.asarray(c['xy']) for c in obs['corners']}
    accepted=set(contract['hybrid_boundary_corner_ids'])
    for k in range(8):
        if k in accepted:
            assert k not in H and k in actual
            assert np.array_equal(selected[k],actual[k]),'selected coordinate was manufactured'
        else:assert np.array_equal(selected[k],native[k],equal_nan=True)
    assert not contract['heldout_projection_is_observation']
    assert not contract['heldout_is_self_occlusion']


def run():
    cv2.setNumThreads(1);checks=[];actual=Counter()
    originals={name:getattr(cv2,name) for name in ('solvePnPGeneric','solvePnPRefineLM','solvePnP')}
    def counted(name):
        def call(*args,**kw):
            actual[name]+=1
            return originals[name](*args,**kw)
        return call
    for name in originals:setattr(cv2,name,counted(name))
    def check(name,fn):
        before=actual.copy()
        try:
            with no_truth_or_model_reads() as reads:
                detail=fn() or {}
                assert not reads
            checks.append(dict(name=name,passed=True,details=detail,
                actual_calls={k:actual[k]-before[k] for k in originals}))
        except Exception as exc:
            checks.append(dict(name=name,passed=False,error=type(exc).__name__+': '+str(exc),
                traceback=traceback.format_exc(),actual_calls={k:actual[k]-before[k] for k in originals}))
    try:
        def heldout_actual_fit():
            K,xyz,q,initial=fixture();native=q.copy();native[0]+=[1.5,-.5];H=[6,7]
            obs=one_shifted_boundary(q,0,[.2,.1]);before=copy.deepcopy(obs);native_before=native.copy()
            bank=AuditedBank(native,K,xyz,image_size=(1280,960))
            selected,contract=S.select_corners(native,obs,initial,H,K,xyz,(1280,960),bank=bank)
            assert contract['hybrid_boundary_corner_ids']==[0],contract['cornerwise_records']
            assert len(bank.calls)==1 and bank.calls[0]['hidden']==[0,6,7]
            record=contract['cornerwise_records'][0];solved=bank.calls[0]['answer']
            assert solved['available'] and record['fit_exclusion_verified']
            assert set(solved['fit_input_ids']).isdisjoint([0,6,7])
            assert len(solved['fit_input_ids'])>=4 and bank.calls[0]['actual_refit_trace']
            prediction=np.asarray(solved['projected'])[0]
            assert np.linalg.norm(selected[0]-prediction)>.1,'test requires actual boundary distinct from prediction'
            assert np.allclose(prediction,q[0],atol=1e-5,rtol=0)
            assert np.array_equal(native,native_before) and obs==before
            assert record['initial_prior_includes_heldout_influence']
            assert_native_output(selected,native,contract,obs,H)
            return dict(actual_heldout_ids=[0,6,7],fit_ids=solved['fit_input_ids'],
                chosen_coordinate=selected[0].tolist(),LOO_prediction=prediction.tolist(),
                actual_boundary_not_projection=True,fully_independent_validation=False,
                source_RGB_or_metadata_modified=False)
        check('actual_k_LOO_and_H_fit_exclusion_selects_boundary_not_projection',heldout_actual_fit)

        def no_candidates_no_fill():
            K,xyz,q,initial=fixture();native=q.copy();native[0]+=[3.,0.]
            obs=observation(q,[]);bank=AuditedBank(native,K,xyz,image_size=(1280,960))
            selected,contract=S.select_corners(native,obs,initial,[7],K,xyz,(1280,960),bank=bank)
            assert np.array_equal(selected,native) and not bank.calls
            assert contract['heldout_pose_calls']==0 and not contract['hybrid_boundary_corner_ids']
            assert_native_output(selected,native,contract,obs,[7])
            return dict(no_missing_boundary_filled=True,no_LOO_calls=True)
        check('no_actual_boundary_never_filled_with_prior_or_LOO_projection',no_candidates_no_fill)

        def residual_tie_and_worse():
            K,xyz,q,initial=fixture();records=[]
            for offset,reason in (([0.,0.],'NUMERICAL_RESIDUAL_TIE_PRESERVE_NATIVE_N3'),
                                  ([.4,.1],'HELDOUT_POSE_PREFERS_NATIVE_N3')):
                obs=one_shifted_boundary(q,0,offset);bank=AuditedBank(q,K,xyz,image_size=(1280,960))
                selected,contract=S.select_corners(q,obs,initial,[7],K,xyz,(1280,960),bank=bank)
                assert len(bank.calls)==1 and bank.calls[0]['answer']['available']
                assert np.array_equal(selected,q) and not contract['hybrid_boundary_corner_ids']
                assert contract['cornerwise_records'][0]['reason']==reason,contract['cornerwise_records']
                records.append(dict(offset_px=offset,reason=reason,fit_ids=bank.calls[0]['answer']['fit_input_ids']))
            return dict(cases=records)
        check('actual_LOO_numeric_tie_and_worse_boundary_preserve_native',residual_tie_and_worse)

        def insufficient_loo_valid_final():
            K,xyz,q,initial=fixture();native=np.full((9,2),np.nan);native[[0,1,2,4,8]]=q[[0,1,2,4,8]]
            native[0]+=[1.,0.];obs=observation(q,[0]);H=[6,7]
            bank=AuditedBank(native,K,xyz,image_size=(1280,960))
            selected,contract=S.select_corners(native,obs,initial,H,K,xyz,(1280,960),bank=bank)
            assert len(bank.calls)==1 and bank.calls[0]['answer']['state']=='INSUFFICIENT_OBSERVATIONS'
            assert np.array_equal(selected,native,equal_nan=True)
            assert contract['cornerwise_records'][0]['reason']=='HELDOUT_POSE_UNAVAILABLE'
            assert contract['operation_counts']['generic_calls']==0
            final=P.refine(selected,K,xyz,initial,H,(1280,960))
            assert final['available'],final['state']
            assert len(final['fit_input_ids'])==4 and set(final['fit_input_ids'])<={0,1,2,4}
            return dict(LOO_state='INSUFFICIENT_OBSERVATIONS',final_state=final['state'],
                        final_fit_ids=final['fit_input_ids'],selection_failure_not_frame_failure=True)
        check('insufficient_LOO_preserves_native_and_allows_actual_four_point_final_pose',insufficient_loo_valid_final)

        def missing_native_supported():
            K,xyz,q,initial=fixture();native=q.copy();native[0]=np.nan;obs=observation(q,[0]);H=[7]
            bank=AuditedBank(native,K,xyz,image_size=(1280,960))
            selected,contract=S.select_corners(native,obs,initial,H,K,xyz,(1280,960),bank=bank)
            record=contract['cornerwise_records'][0]
            assert bank.calls[0]['answer']['available'] and contract['hybrid_boundary_corner_ids']==[0]
            assert contract['missing_native_boundary_ids']==[0] and record['native_distance_check']=='unavailable'
            assert record['native_N3_LOO_residual_px'] is None and record['native_minus_candidate_squared_residual_px2'] is None
            assert np.array_equal(selected[0],q[0])
            assert_native_output(selected,native,contract,obs,H)
            return dict(missing_native_comparison_unavailable=True,actual_boundary_used=True,fit_ids=record['fit_input_ids'])
        check('missing_native_boundary_is_used_only_with_actual_other_corner_support',missing_native_supported)

        def cache_reuse():
            K,xyz,q,initial=fixture();native=q.copy()
            native[:6]+=[.6,.2];obs=observation(q,range(6));bank=AuditedBank(native,K,xyz,image_size=(1280,960))
            selected,contract=S.select_corners(native,obs,initial,[6,7],K,xyz,(1280,960),bank=bank)
            assert len(bank.calls)==6 and [c['hidden'] for c in bank.calls]==[[k,6,7] for k in range(6)]
            expected=70*len(bank.dims)
            assert bank.ledger['subsets_considered']==expected==140
            assert bank.ledger['subset_generic_calls']<=expected
            assert bank.calls[0]['answer']['operation_counts']['subset_generic_calls']==expected
            assert all(c['answer']['operation_counts']['subset_generic_calls']==0 for c in bank.calls[1:])
            assert np.array_equal(bank.points,native)
            assert contract['same_N3_bank_reused_across_heldouts']
            assert_native_output(selected,native,contract,obs,[6,7])
            # Even after a second mask, identical N3 coordinates reuse the same
            # global numeric bank; masks never cause four-ID regeneration.
            before=bank.ledger['subset_generic_calls']
            _,second=S.select_corners(native,obs,initial,[7],K,xyz,(1280,960),bank=bank)
            assert bank.ledger['subset_generic_calls']==before and second['operation_counts']['subset_generic_calls']==0
            changed=native.copy();changed[0,0]+=.1;rejected=False
            try:S.select_corners(changed,obs,initial,[7],K,xyz,(1280,960),bank=bank)
            except ValueError:rejected=True
            assert rejected
            return dict(dimension_hypotheses=len(bank.dims),four_subsets_total=expected,
                        max_four_subsets_per_dimension=70,heldout_solves_first_pass=6,
                        later_heldout_subset_generation=0,mismatched_coordinates_bank_rejected=True)
        check('one_native_bank_reuses_at_most_seventy_subsets_per_dimension_across_k_and_H',cache_reuse)

        def hidden_final_no_refit():
            K,xyz,q,initial=fixture();native=q.copy();native[0]+=[1.,0.];native[[6,7]]+=[[100.,-80.],[-80.,100.]]
            obs=observation(q,[0,6]);H=[6,7]
            selection_bank=AuditedBank(native,K,xyz,image_size=(1280,960))
            selected,contract=S.select_corners(native,obs,initial,H,K,xyz,(1280,960),bank=selection_bank)
            assert not set(contract['hybrid_boundary_corner_ids']) & set(H)
            assert np.array_equal(selected[H],native[H])
            record=next(r for r in contract['cornerwise_records'] if r['id']==6)
            assert record['reason']=='INITIAL_SELF_HIDDEN' and record['heldout_pose_calls']==0
            final_bank=AuditedBank(selected,K,xyz,image_size=(1280,960))
            final=P.refine(selected,K,xyz,initial,H,(1280,960),bank=final_bank)
            assert final['available'] and set(final['fit_input_ids']).isdisjoint(H)
            before=actual.copy();before_ledger=copy.deepcopy(final_bank.ledger)
            output=B.assemble(native,selected,initial,H,final,'N3_VALIDATED_ROLE',contract)
            assert actual==before and final_bank.ledger==before_ledger and len(final_bank.calls)==1
            independently_projected=project(cuboid(*final['cf_extents']),np.asarray(final['R_cf']),np.asarray(final['centroid']),K)
            assert np.allclose(output['native_points'][H],independently_projected[H],atol=1e-10,rtol=0)
            assert not np.array_equal(output['native_points'][H],native[H])
            assert not output['reprojections_reused_as_observations']
            assert output['excluded']==H and output['reprojected_ids']==H
            assert np.array_equal(output['native_points'][8],native[8])
            # The same frozen assembly helper preserves center, all scores and
            # other candidates. Only selected corner coordinates may change.
            raw=dict(selected_index=1,candidates=[dict(candidate_index=0,score=.2,class_id=3,
                keypoints_xy=q.copy(),keypoints_conf=[.2]*9,box_xyxy=[1.,2.,3.,4.]),
                dict(candidate_index=1,score=.8,class_id=0,keypoints_xy=native.copy(),
                     keypoints_conf=[.9]*9,box_xyxy=[4.,5.,6.,7.])])
            prediction=copy.deepcopy(raw);prediction['candidates'][1]['keypoints_xy']=output['native_points']
            assert B.preserve_prediction(raw,prediction)
            return dict(final_fit_ids=final['fit_input_ids'],actual_self_H=H,
                projection_calls_after_final_PnP=0,additional_fit_after_output=0,
                final_H_actual_projection_max_abs_px=float(np.max(np.abs(output['native_points'][H]-independently_projected[H]))),
                center_and_candidate_metadata_preserved=True,assembly_scope='unchanged v2 helper used by parent driver')
        check('self_H_excluded_then_reprojected_once_without_refit_and_metadata_damage',hidden_final_no_refit)

        def wrong_mask_not_frame_veto():
            theta0,theta1=np.deg2rad(5.),np.deg2rad(6.5)
            K,xyz,q0,initial=fixture((0.,theta0,0.),(0.,.5,4.))
            R=cv2.Rodrigues(np.array([0.,theta1,0.]))[0]
            q=np.vstack([project(cuboid(*xyz),R,np.array([0.,.5,4.]),K),q0[8]])
            H0=np.flatnonzero(visibility(cuboid(*xyz),np.asarray(initial['R_cf']),np.asarray(initial['centroid']))[0]).tolist()
            H1=np.flatnonzero(visibility(cuboid(*xyz),R,np.array([0.,.5,4.]))[0]).tolist()
            assert H0!=H1
            k=next(k for k in range(8) if k not in H0);native=q.copy();native[k]+=[.02,0.]
            obs=observation(q,[k]);bank=AuditedBank(native,K,xyz,image_size=(1280,960))
            selected,contract=S.select_corners(native,obs,initial,H0,K,xyz,(1280,960),bank=bank)
            final=P.refine(selected,K,xyz,initial,H0,(1280,960))
            assert final['available'],final['state']
            output=B.assemble(native,selected,initial,H0,final,'N3_VALIDATED_ROLE',contract)
            assert output['new_pose_estimated'] and output['output_status']=='NEW_POSE'
            assert output['hidden_set_changed'] and not set(H0)&set(final['fit_input_ids'])
            return dict(initial_H=H0,final_H=output['hidden_after'],hidden_difference_recorded=True,
                        final_pose_not_vetoed=True)
        check('mask_disagreement_after_refinement_is_recorded_without_frame_veto',wrong_mask_not_frame_veto)

        def one_wrong_actual_input():
            K,xyz,q,initial=fixture();native=q.copy();native[0]+=[30.,-35.];native[4]+=[.8,.2]
            obs=observation(q,[4]);H=[7]
            selected,contract=S.select_corners(native,obs,initial,H,K,xyz,(1280,960))
            assert np.array_equal(selected[0],native[0]),'no oracle removal/injection of bad coordinate'
            assert contract['hybrid_boundary_corner_ids']==[4],contract['cornerwise_records']
            final=P.refine(selected,K,xyz,initial,H,(1280,960))
            assert final['available'],final['state']
            assert 0 in final['used'] and 0 not in final['final_inliers']
            assert len(final['final_inliers'])>=4 and np.max(np.abs(np.asarray(final['projected'])-q[:8]))<1e-5
            return dict(retained_wrong_input_id=0,not_oracle_removed=True,
                        final_used=final['used'],final_inliers=final['final_inliers'],
                        accurate_other_correspondences_provide_pose=True)
        check('one_wrong_selected_N3_input_can_be_consensus_outlier_without_oracle',one_wrong_actual_input)

        def evidence_veto_not_failure():
            K,xyz,q,initial=fixture();native=q.copy();native[0]+=[1.,0.];obs=observation(q,[0])
            line=obs['lines'][0];line['support_points'][0]=(np.asarray(line['support_points'][0])+3*np.asarray(line['normal'])).tolist()
            bank=AuditedBank(native,K,xyz,image_size=(1280,960))
            selected,contract=S.select_corners(native,obs,initial,[7],K,xyz,(1280,960),bank=bank)
            assert np.array_equal(selected,native) and not bank.calls
            assert contract['corner_admission'][0]['reason']=='FINAL_LINE_CONSENSUS_INCONSISTENT'
            assert P.refine(selected,K,xyz,initial,[7],(1280,960))['available']
            return dict(inconsistent_evidence_abstains_only_this_corner=True,native_final_pose_remains_available=True)
        check('inconsistent_boundary_line_abstains_corner_without_vetoing_frame',evidence_veto_not_failure)

        def standard_four_five():
            K,xyz,q,initial=fixture();cases=[]
            for ids in ((0,1,2,3),(0,1,2,4),(0,1,2,4,5)):
                for robust in (False,True):
                    sparse=np.full((9,2),np.nan);sparse[list(ids)]=q[list(ids)];sparse[8]=q[8]
                    answer=P.refine(sparse,K,xyz,initial,[6,7],(1280,960),robust=robust)
                    assert answer['available'],answer['state']
                    assert set(answer['fit_input_ids'])<=set(ids) and len(answer['fit_input_ids'])>=4
                    assert answer['geometry']['jacobian']['numerical_rank']==6
                    assert np.max(np.abs(np.asarray(answer['projected'])-q[:8]))<1e-5
                    if not robust:
                        assert answer['generator'].startswith('IPPE_PLANE' if ids==(0,1,2,3) else 'SQPNP')
                    else:
                        assert answer['generator'].startswith(('IPPE_PLANE','SQPNP'))
                    cases.append(dict(ids=list(ids),robust=robust,generator=answer['generator'],
                                      fit_ids=answer['fit_input_ids'],multiple_solutions=answer['multiple_solutions']))
            # Borrow the original test_solver.py exact-plane 4/5-point standard
            # control. Both IPPE returned solutions are read; no six->four count
            # edit or IPPE_SQUARE substitution is involved.
            plane=np.array([[-.65,-.55,0.],[.65,-.55,0.],[.65,.55,0.],[-.65,.55,0.],[-.4,.2,0.]])
            planar=[];R=np.asarray(initial['R_cf']);t=np.asarray(initial['centroid'])
            for n in (4,5):
                measured=project(plane[:n],R,t,K)
                ret=cv2.solvePnPGeneric(plane[:n],measured,K,None,flags=cv2.SOLVEPNP_IPPE)
                assert ret[0] and len(ret[1])==len(ret[2])==2
                errors=[float(np.max(np.abs(project(plane[:n],cv2.Rodrigues(rv)[0],tv,K)-measured))) for rv,tv in zip(ret[1],ret[2])]
                assert min(errors)<1e-5
                planar.append(dict(points=n,standard='IPPE',retained_solutions=len(ret[1]),projection_errors_px=errors))
            return dict(actual_registry_cases=cases,actual_planar_standard_controls=planar,
                        implementation='unchanged v2 PoseBank and original finite solver')
        check('actual_four_five_planar_nonplanar_standard_generators_and_IPPE_multisolution',standard_four_five)

        def distinct_tie_rejected():
            K,xyz,q,initial=fixture();native=np.full((9,2),np.nan);native[[0,1,2,4,8]]=q[[0,1,2,4,8]]
            class ControlledCandidates(P.PoseBank):
                def _generic(self,*args,**kw):return []
                def _lm(self,*args,**kw):return None
            bank=ControlledCandidates(native,K,xyz,image_size=(1280,960));bank.hypotheses=[]
            rv=cv2.Rodrigues(np.asarray(initial['R_cf']))[0]
            for j,shift in enumerate((-.02,.02)):
                tv=np.asarray(initial['centroid'])+[shift,0.,0.]
                projected=project(cuboid(*xyz),np.asarray(initial['R_cf']),tv,K)
                bank.hypotheses.append(Candidate(0,(0,1,2,4),j,'CONTROLLED_POSITIVE_DEPTH_POSE',rv,tv.reshape(3,1),projected))
            before=actual.copy();answer=bank.solve(initial,hidden=[6,7],robust=True)
            assert not answer['available'] and answer['state']=='AMBIGUOUS_PNP'
            assert answer['unresolved_ambiguity'] and answer['multiple_solutions']
            assert actual==before and not answer['global_uniqueness_proven']
            return dict(controlled_arbitration_only=True,actual_PnP_calls=0,
                        distinct_equal_numeric_score_is_not_new_pose=True,
                        natural_IPPE_multisolution_checked_separately=True)
        check('unchanged_pose_policy_rejects_distinct_numeric_tied_pose_candidates',distinct_tie_rejected)

        def interface_and_inputs():
            signature=inspect.signature(S.select_corners)
            assert set(signature.parameters)=={'native_n3','observation','initial_pose','hidden','K','xyz','image_size','bank'}
            assert not S.POLICY['truth_input'] and not S.POLICY['fully_independent_validation']
            assert S.POLICY['maximum_heldout_solves']==8
            assert 'torch' not in sys.modules,'selection unit tests must not import a model/GPU runtime'
            return dict(no_image_or_GT_or_score_input=True,no_Torch_import=True,
                        original_source_RGB_and_frozen_features_not_opened=True,
                        heldout_fit_exclusion_not_full_statistical_independence=True)
        check('deployable_selection_has_no_image_model_GT_or_performance_tuning_interface',interface_and_inputs)
    finally:
        for name,fn in originals.items():setattr(cv2,name,fn)
    return dict(schema='actual_synthetic_cornerwise_selection_checks_v3',complete=True,
        passed=all(c['passed'] for c in checks),checks=checks,actual_calls={k:actual[k] for k in originals},
        selection_policy=copy.deepcopy(S.POLICY),pose_policy=copy.deepcopy(P.POLICY),opencv=cv2.__version__,
        images_decoded=0,detector_forwards=0,N3_forwards=0,ROLE_forwards=0,GPU_calls=0,
        new_training_updates=0,new_RGB=0,private_GT_reads=0,rays=0,real_pose_fits=0,timing_intervals=0,
        limitations=['Synthetic geometry/selection integrity tests, not real accuracy or latency.',
            'LOO excludes held-out fit residuals; the initial N3 prior still includes that corner.',
            'Synthetic line-support provenance is a fixture, not actual real-image physical ownership.',
            'Unchanged v2 assembly helper is tested; full live-pipeline metadata needs its own sealed parity.',
            'Numerical rank/multiple-solution handling does not prove global pose uniqueness.'])


def binding(path):
    path=Path(path).resolve()
    return dict(path=str(path.relative_to(REPO)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DOC/'SELECTION_CHECKS.json')
    args=parser.parse_args();out=args.output
    if out.is_symlink() or out.exists():raise FileExistsError('Preserve completed/interrupted check receipt: '+str(out))
    if out.parent.is_symlink():raise ValueError('Check output parent is a symlink')
    out=out.resolve()
    if out!=DOC/'SELECTION_CHECKS.json':
        raise ValueError('Only the exact new selection-check receipt destination is supported')
    # Check output claims before any CPU PnP tests. An interrupted execution
    # leaves this exclusive STARTED record and cannot be silently restarted.
    started=out.with_name(out.stem+'_STARTED.json')
    if started.exists() or started.is_symlink():raise FileExistsError('Preserve selection check execution claim')
    code=dict(test=binding(__file__),selection=binding(S.__file__),pose=binding(P.__file__),
        original_solver=binding(REPO/'scripts/research/pallet_observation_refiner_20261009_v1/solver.py'),
        borrowed_standard_checks=binding(REPO/'scripts/research/pallet_observation_refiner_20261009_v1/test_solver.py'),
        original_pipeline_helper=binding(B.__file__))
    out.parent.mkdir(parents=True,exist_ok=True)
    with started.open('x') as stream:
        json.dump(dict(schema='synthetic_cornerwise_selection_check_claim_v3',status='STARTED',code=code,
                       real_or_GPU_execution=False,no_automatic_retry=True),stream,indent=2,allow_nan=False)
        stream.write('\n')
    result=run();result['code']=code
    with out.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(passed=result['passed'],checks=len(result['checks']),
        failed=[c['name'] for c in result['checks'] if not c['passed']],actual_calls=result['actual_calls'])))
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
