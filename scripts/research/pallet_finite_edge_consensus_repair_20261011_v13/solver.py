"""One fixed local finite-edge consensus repair of actual sparse observations.

The original initial/available-point poses supply only local starts. Existing
line incidence components and actual-support longitudinal hinges form one
semantic edge block. Each existing branch gets at most one actual-inlier refit.
All comparisons score the identical full original observation pool. No model,
image, truth, native fill, random start or inferred support-point 3D ID is used.
"""
from __future__ import annotations
import copy
import cv2
import numpy as np
from scipy.optimize import least_squares
from ..pallet_cornerwise_independent_20261010_v4 import pose as P
from ..pallet_partial_line_independent_20261010_v5 import solver as V5
from ..pallet_sparse_local_line_20261010_v9 import solver as V9
from ..pallet_observation_refiner_20261009_v1.solver import Candidate

EDGES = V5.EDGES
METHOD = 'FINITE_EDGE_CONSENSUS'
POLICY = dict(schema='finite_edge_actual_inlier_consensus_policy_v13',
    loss='soft_l1', f_scale_px=8., max_nfev=50,
    line_block='2 endpoint-normal residuals/sqrt(2), N actual-support longitudinal outside hinges/sqrt(N)',
    line_consensus_unit='one distinct unused semantic edge, not N query votes',
    point_inlier_threshold_px=8., edge_block_inlier_threshold_px=8.,
    whole_pool_score='descending point+edge inlier count, truncated factor SSE, dimension and start order',
    truncated_factor_squared_cap=64.,
    maximum_start_sources=2, maximum_proposal_starts=4, maximum_fixed_inlier_refits=4,
    maximum_actual_optimizer_dispatches=8, refit_iterations=1,
    starts='unchanged initial N3 and available point-only physical pose, remapped to both registry dimensions',
    initial_pose_residual_prior=False, initial_dimension_prior=False, random_starts=False,
    minimum_actual_point_gate=False, minimum_scalar_rows=6,
    acceptance='finite positive-depth successful actual-inlier refit, all fitted factors final8px, observed and modeled-normal fit J rank6 and inherited weak-safe',
    numerical_rank_relative_threshold=1e-10, weak_relative_threshold=1e-4,
    weak_normalization='unchanged V5 unit L2 Jacobian columns',
    numerical_score_tie_px2=P.POLICY['numerical_score_tie_px2'],
    numerical_equivalence={k:v for k,v in P.POLICY.items() if k.startswith('equivalent_')},
    observation_ambiguity='distinct accepted physical branches explaining identical final actual8px inlier IDs; cost equality not required',
    conservative_ambiguity_is_not_global_uniqueness=True, global_unique_pose_proven=False,
    partial_support_endpoints_are_not_model_endpoints=True,
    radius_gate=False, source_radius_is_uncertainty_surrogate=True,
    point_line_double_count=False, consumed_edges_never_reintroduced_by_outlier_refit=True,
    physical_real_edge_ownership_certified=False, known_DEV_structural_repair=True,
    hidden_policy='exclude H initial XY from every score and fit; NEW final H projection once, no refit',
    native_fill=False, center_used_for_fit_or_score=False, new_training_seeds_RGB=False)


def finite_residual_jacobian(model, z, K, points, point_ids, lines, *, modeled_normals=False):
    """Exact forward derivative, except modeled normals are held diagnostic-fixed.

    Each line's query support has a free location along its finite 3D edge;
    query fractions never become independent 3D point correspondences.
    At u=0/L the hinge's inside derivative is zero. A collapsed projected edge
    is a typed numerical failure, not a jittered or fabricated observation.
    """
    q, J = V5.projection_jacobian(model, z, K)
    residual, matrices, blocks = [], [], []
    for k in point_ids:
        residual.extend(q[k]-points[k]); matrices.extend(J[k])
    for line in lines:
        a,b=EDGES[line['edge']]; A,B=q[a],q[b]; JA,JB=J[a],J[b]
        direction=B-A; length=float(np.linalg.norm(direction))
        if not np.isfinite(length) or length<=1e-12: raise ValueError('FINITE_PROJECTED_EDGE_DEGENERATE')
        tangent=direction/length;normal=np.asarray(line['normal'],float)
        if modeled_normals:normal=np.array([-tangent[1],tangent[0]])
        signed=q[[a,b]]@normal-line['offset']
        start=len(residual);residual.extend(signed/np.sqrt(2.));matrices.extend([normal@JA/np.sqrt(2.),normal@JB/np.sqrt(2.)])
        support=np.asarray(line['source_line']['support_points'],float);N=len(support)
        if N<3 or support.shape!=(N,2) or not np.isfinite(support).all():raise ValueError('FINITE_ACTUAL_SUPPORT_CONTRACT')
        d_direction=JB-JA;d_length=tangent@d_direction
        d_tangent=(np.eye(2)-np.outer(tangent,tangent))@d_direction/length
        along=(support-A)@tangent;hinges=[];hinge_J=[];branches=[]
        for xy,u in zip(support,along):
            du=-tangent@JA+(xy-A)@d_tangent
            if u<0: h,row,branch=u,du,'BEFORE_FIRST_ENDPOINT'
            elif u>length: h,row,branch=u-length,du-d_length,'AFTER_SECOND_ENDPOINT'
            else:h,row,branch=0.,np.zeros(6),'INSIDE_CLOSED_SEGMENT'
            hinges.append(float(h));hinge_J.append(row);branches.append(branch)
        residual.extend(np.asarray(hinges)/np.sqrt(N));matrices.extend(np.asarray(hinge_J)/np.sqrt(N))
        blocks.append(dict(edge=line['edge'],endpoints=[a,b],scalar_start=start,
            scalar_end=len(residual),normal_signed_px=signed.tolist(),outside_hinges_px=hinges,
            longitudinal_support_px=along.tolist(),projected_segment_length_px=length,
            hinge_branches=branches,support_query_ids=list(line['support_query_ids']),
            actual_support_xy=support.tolist(),query_radii_px=list(line['source_line']['query_radii_px']),
            radius_gate_applied=False,block_factor_count=1,
            physical_ownership_truth='UNKNOWN',modeled_normals_held_fixed=modeled_normals))
    return np.asarray(residual,float),np.asarray(matrices,float).reshape(-1,6),q,blocks


def weak(info):
    singular=info['singular_values']
    return bool(len(singular)!=6 or singular[-1]<singular[0]*1e-4)


class FiniteEdgeConsensusBank(V9.LocalPointLineBank):
    """Reuse unchanged source-factor and initial-pose validation, never V9.solve."""
    def __init__(self, points, K, xyz, image_size=(640,480)):
        super().__init__(points,K,xyz,image_size=image_size)
        if self.points.shape!=(9,2):raise ValueError('finite sparse packet requires eight corners and preserved center8')
        self.counts.update(proposal_start_entries=0,proposal_cache_hits=0,refit_start_entries=0,
            refit_cache_hits=0,proposal_optimizer_calls=0,proposal_optimizer_returns=0,
            proposal_optimizer_exceptions=0,refit_optimizer_calls=0,refit_optimizer_returns=0,
            refit_optimizer_exceptions=0,local_optimizer_exceptions=0,
            finite_score_calls=0,score_projectPoints_calls=0,geometry_projectPoints_calls=0)

    def _geometry(self, model, z, point_ids, lines):
        self.counts['geometry_diagnostic_calls']+=1
        self.counts['geometry_projectPoints_calls']+=1
        residual,observed,q,blocks=finite_residual_jacobian(model,z,self.K,self.points,point_ids,lines)
        self.counts['geometry_projectPoints_calls']+=1
        _,modeled,_,_=finite_residual_jacobian(model,z,self.K,self.points,point_ids,lines,modeled_normals=True)
        split=2*len(point_ids)
        info=dict(observed_normal_joint=V5.rank_info(observed),modeled_normal_joint=V5.rank_info(modeled),
            point_only=V5.rank_info(modeled[:split]),observed_normal_lines_only=V5.rank_info(observed[split:]),
            modeled_normal_lines_only=V5.rank_info(modeled[split:]),point_ids=list(point_ids),
            line_edges=[l['edge'] for l in lines],observed_normal_J=observed.tolist(),
            modeled_normal_J=modeled.tolist(),actual_scalar_residuals_px=residual.tolist(),
            edge_blocks=blocks,modeled_normals_held_fixed_for_derivative=True,global_unique_pose_proven=False,
            observed_weak=None,modeled_weak=None)
        info['observed_weak']=weak(info['observed_normal_joint']);info['modeled_weak']=weak(info['modeled_normal_joint'])
        return info,residual,q

    def _score(self, model, z, U, lines):
        self.counts['finite_score_calls']+=1;self.counts['score_projectPoints_calls']+=1
        q,_=V5.projection_jacobian(model,z,self.K)
        point=np.linalg.norm(q[list(U)]-self.points[list(U)],axis=1)
        edge=[];blocks=[];scalar=list((q[list(U)]-self.points[list(U)]).reshape(-1))
        for line in lines:
            a,b=EDGES[line['edge']];A,B=q[a],q[b];direction=B-A;length=float(np.linalg.norm(direction))
            support=np.asarray(line['source_line']['support_points'],float);N=len(support);start=len(scalar)
            common=dict(edge=line['edge'],endpoints=[a,b],scalar_start=start,scalar_end=start+2+N,
                support_query_ids=list(line['support_query_ids']),actual_support_xy=support.tolist(),
                query_radii_px=list(line['source_line']['query_radii_px']),radius_gate_applied=False,
                block_factor_count=1,physical_ownership_truth='UNKNOWN',modeled_normals_held_fixed=False)
            if not np.isfinite(length) or length<=1e-12:
                # A scoring-only edge can be an outlier even though it is
                # absent from the actual refit. It must not crash or veto all
                # remaining observations. None is an explicit invalid norm.
                edge.append(None);scalar.extend([None]*(2+N));common.update(
                    score_state='DEGENERATE_PROJECTED_EDGE_OUTLIER',normal_signed_px=None,
                    outside_hinges_px=None,longitudinal_support_px=None,projected_segment_length_px=length)
            else:
                along=(support-A)@(direction/length);hinge=np.minimum(along,0.)+np.maximum(along-length,0.)
                signed=q[[a,b]]@line['normal']-line['offset'];block=np.r_[signed/np.sqrt(2.),hinge/np.sqrt(N)]
                edge.append(float(np.linalg.norm(block)));scalar.extend(block.tolist());common.update(
                    score_state='FINITE_EDGE_BLOCK',normal_signed_px=signed.tolist(),outside_hinges_px=hinge.tolist(),
                    longitudinal_support_px=along.tolist(),projected_segment_length_px=length)
            blocks.append(common)
        ip=[k for k,error in zip(U,point) if error<=8.];il=[line['edge'] for line,error in zip(lines,edge) if error is not None and error<=8.]
        norms=list(point)+edge;truncated=sum(64. if value is None else min(value*value,64.) for value in norms)
        return dict(point_ids=list(U),line_edges=[l['edge'] for l in lines],point_residual_norm_px=point.tolist(),
            edge_block_norm_px=edge,edge_blocks=blocks,final_inliers=ip,final_line_inliers=il,
            inlier_count=len(ip),unique_line_inlier_count=len(il),total_inlier_factor_count=len(ip)+len(il),
            factor_sse_px2=None if any(value is None for value in norms) else float(sum(value*value for value in norms)),
            factor_truncated_sse_px2=float(truncated),scalar_residuals_px=scalar,
            score_only_degenerate_edge_is_outlier=True,fixed_full_pool=True),q

    def _attempt(self, model, z0, point_ids, lines, phase, entry):
        entry.update(actual_optimizer_called=True,success=False,accepted=False,rejected=None,
            fit_point_ids=list(point_ids),fit_line_edges=[l['edge'] for l in lines],initial_z=np.asarray(z0,float).tolist())
        def residual(z):
            self.counts['optimizer_residual_calls']+=1;self.counts['optimizer_projectPoints_calls']+=1
            return finite_residual_jacobian(model,z,self.K,self.points,point_ids,lines)[0]
        def jacobian(z):
            self.counts['optimizer_jacobian_calls']+=1;self.counts['optimizer_projectPoints_calls']+=1
            return finite_residual_jacobian(model,z,self.K,self.points,point_ids,lines)[1]
        self.counts['local_optimizer_calls']+=1;self.counts[phase+'_optimizer_calls']+=1
        try:
            fit=least_squares(residual,z0,jac=jacobian,loss='soft_l1',f_scale=8.,max_nfev=50)
            self.counts['local_optimizer_completions']+=1;self.counts[phase+'_optimizer_returns']+=1
        except BaseException as error:
            self.counts['local_optimizer_exceptions']+=1;self.counts[phase+'_optimizer_exceptions']+=1
            entry.update(rejected='LOCAL_NUMERICAL_FAILURE',exception=dict(type=type(error).__name__,message=str(error)))
            if not isinstance(error,(ValueError,TypeError,FloatingPointError,np.linalg.LinAlgError,cv2.error)):raise
            return None
        entry.update(success=bool(fit.success),scipy_status=int(fit.status),nfev=int(fit.nfev),
            njev=None if fit.njev is None else int(fit.njev),final_z=np.asarray(fit.x,float).tolist(),reported_soft_l1_cost=float(fit.cost))
        if not fit.success:entry['rejected']='OPTIMIZER_FAILED';return None
        self.counts['candidate_validation_calls']+=1
        if not self.bank._valid(np.asarray(fit.x[:3]),np.asarray(fit.x[3:]),entry['dimension_index']):
            entry['rejected']='NONFINITE_OR_NONPOSITIVE_DEPTH';return None
        try:
            geometry,rr,q=self._geometry(model,np.asarray(fit.x,float),point_ids,lines)
            if not np.isfinite(rr).all() or not np.isfinite(q).all():raise ValueError('NONFINITE_FINITE_EDGE_FACTOR')
            cost=V9.soft_l1_cost(rr)
            if not np.isfinite(fit.cost) or not np.isclose(cost,fit.cost,rtol=1e-9,atol=1e-8):raise ValueError('FIXED_FIT_POOL_COST_MISMATCH')
        except (ValueError,FloatingPointError,np.linalg.LinAlgError,cv2.error) as error:
            entry.update(rejected='FINAL_GEOMETRY_UNAVAILABLE',exception=dict(type=type(error).__name__,message=str(error)));return None
        self.counts['candidate_Rodrigues_calls']+=1
        Rcf=cv2.Rodrigues(np.asarray(fit.x[:3]))[0];Q=np.eye(3) if entry['dimension_index']==0 else np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
        witness=dict(phase=phase,start_index=entry['start_index'],source=entry['source'],dimension_index=entry['dimension_index'],
            selected_hypothesis=self.bank.names[entry['dimension_index']],final_z=np.asarray(fit.x,float).tolist(),
            R_cf=Rcf.tolist(),R_physical=(Rcf@Q).tolist(),centroid=np.asarray(fit.x[3:],float).tolist(),
            cf_extents=self.bank.dims[entry['dimension_index']].tolist(),projected=q.tolist(),
            model_camera_depths_m=((Rcf@model.T).T+np.asarray(fit.x[3:]))[:,2].tolist(),
            fit_input_ids=list(point_ids),fit_line_edges=[l['edge'] for l in lines],
            scalar_residuals_px=rr.tolist(),soft_l1_cost=cost,reported_soft_l1_cost=float(fit.cost),
            sse_px2=float(rr@rr),point_line_geometry=geometry,optimizer_success=True,
            initial_pose_residual_prior=False,local_only=True,global_uniqueness_proven=False,
            accepted=False,rejected=None)
        return witness

    def solve(self, observation, adopted_boundary_corner_ids=(), hidden=(), *, initial_pose=None, point_result=None):
        before=self.counts.copy();self.counts['local_solve_calls']+=1
        H=P._ids(hidden,'hidden');adopted=P._ids(adopted_boundary_corner_ids,'adopted_boundary_corner_ids');U=tuple(k for k in self.bank.eligible if k not in H)
        result=dict(available=False,pose_available=False,new_pose_estimated=False,no_pose=True,fallback_used=False,
            state='INSUFFICIENT_ACTUAL_FACTORS',reason='too_few_actual_scalar_factors',output_status='NO_NEW_POSE',
            solver='FINITE_EDGE_ACTUAL_INLIER_LOCAL_CONSENSUS',policy=copy.deepcopy(POLICY),
            eligible=list(self.bank.eligible),used=list(U),excluded=list(H),hidden=list(H),actual_self_hidden_ids=list(H),
            ineligible=copy.deepcopy(self.bank.ineligible),input_hash=self.bank.digest,
            fit_input_ids=[],input_inliers=[],final_inliers=[],inliers=[],fit_line_edges=[],final_line_inliers=[],
            line_edges=[],consumed_edges=[],proposal_solutions=[],all_candidate_solutions=[],optimizer_attempts=[],
            start_source_witnesses=[],alternatives=[],per_dimension_best=[],candidate_count=0,refit_count=0,
            multiple_solutions=False,unresolved_ambiguity=False,numerical_tied_solution_count=0,numerical_duplicate_solution_count=0,
            R_cf=None,R_physical=None,centroid=None,cf_extents=None,projected=None,points_final=self.points.tolist(),
            prior_used=False,initial_pose_used=False,initial_pose_start_used=False,initial_pose_residual_prior=False,
            initial_projection_used=False,initial_dimension_prior_used=False,known_dimension_constraint_used=False,
            independent_PnP_or_PnL=False,local_initialization_only=True,rank_is_local_only=True,global_uniqueness_proven=False,
            hidden_initial_excluded=True,hidden_reprojected=False,reprojected_points_reused_as_observations=False,
            same_edge_point_line_double_count=False,derived_line_points=0,excluded_image_coordinates_used_for_scoring=False,
            excluded_image_coordinates_used_for_equivalence=False,equivalence_uses_all_eight_model_predictions=True,
            inlier_geometry_is_acceptance_gate=True,inlier_counts_are_diagnostic_only=False,
            observation_binding=V9._binding(observation),local_refinement_estimated=False,
            no_new_coordinate_or_native_numeric_fill=True,center_used_for_fit_or_score=False,
            source_line_support_is_not_real_ownership_certificate=True)
        def finish():
            result['operation_counts']={k:self.counts[k]-before[k] for k in self.counts};result['hypothesis_bank_counts']=self.bank.ledger.copy();result['local_bank_counts']=self.counts.copy();result['initial_pose_used']=result['initial_pose_start_used']
            return result
        try:lines,consumed,checks=self.factor_validator._lines(observation,adopted,set(U))
        except (KeyError,TypeError,ValueError,IndexError) as error:
            result.update(state='INVALID_OBSERVATION_CONTRACT',reason=str(error));return finish()
        edges=tuple(l['edge'] for l in lines);line_map={l['edge']:l for l in lines}
        result.update(line_edges=list(edges),consumed_edges=consumed,line_contract_checks=checks,
            actual_adopted_boundary_ids=[k for k in adopted if k in U],
            factor_pool=dict(point_ids=list(U),line_edges=list(edges),unique_line_factors=len(lines),actual_point_factors=len(U),
                scalar_residuals=2*len(U)+sum(2+len(l['support_query_ids']) for l in lines)),
            actual_source_lines=[dict(edge=l['edge'],normal=l['normal'].tolist(),offset=l['offset'],raw=copy.deepcopy(l['source_line'])) for l in lines],
            factor_pool_binding=V9._binding(dict(points=self.points[list(U)],point_ids=U,lines=lines)),
            registered_dimensions=[dict(index=i,name=self.bank.names[i],extents=d.tolist()) for i,d in enumerate(self.bank.dims)])
        if result['factor_pool']['scalar_residuals']<6:return finish()
        if not V5._camera(self.K):result.update(state='UNSUPPORTED_CAMERA_CONTRACT',reason='finite_J_requires_zero_skew_pinhole_K');return finish()
        seeds=[]
        for source,raw in (('SEALED_INITIAL_N3',initial_pose),('SEALED_AVAILABLE_POINT_ONLY',point_result)):
            self.counts['start_sources_examined']+=1;seed,witness=self._seed(raw,source);result['start_source_witnesses'].append(witness)
            if seed is not None:seeds.append((source,seed))
        if not seeds:result.update(state='NO_LOCAL_INITIALIZATION',reason='no_available_valid_original_start');return finish()
        proposal_cache,refit_cache={},{};viable=[];failures=[]
        for source,(Rphysical,centroid) in seeds:
            for dim,model in enumerate(self.bank.models):
                Q=np.eye(3) if dim==0 else np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
                self.counts['seed_Rodrigues_calls']+=1;z0=np.r_[cv2.Rodrigues(Rphysical@Q.T)[0].reshape(3),centroid]
                index=len(result['optimizer_attempts']);entry=dict(start_index=index,source=source,dimension_index=dim,
                    proposal=None,refit=None,initial_pose_residual_prior=False);result['initial_pose_start_used']=True
                self.counts['local_start_entries']+=1;self.counts['proposal_start_entries']+=1
                proposal_entry=dict(start_index=index,source=source,dimension_index=dim,cache_reused=False)
                key=(dim,z0.astype('<f8').tobytes())
                if key in proposal_cache:
                    self.counts['proposal_cache_hits']+=1;self.counts['local_start_cache_hits']+=1
                    proposal,old=copy.deepcopy(proposal_cache[key]);proposal_entry.update(old,actual_optimizer_called=False,cache_reused=True,cache_source_start_index=old['start_index'],start_index=index,source=source)
                else:
                    proposal=self._attempt(model,z0,U,lines,'proposal',proposal_entry);proposal_cache[key]=(copy.deepcopy(proposal),copy.deepcopy(proposal_entry))
                entry['proposal']=proposal_entry
                if proposal is None:
                    failures.append(proposal_entry['rejected']);result['optimizer_attempts'].append(entry);continue
                score,_=self._score(model,np.asarray(proposal['final_z']),U,lines);proposal['whole_pool_score']=score
                proposal['proposal_index']=len(result['proposal_solutions']);proposal['logical_start_index']=index;proposal['logical_start_source']=source
                result['proposal_solutions'].append(copy.deepcopy(proposal));ip=tuple(score['final_inliers']);il=tuple(score['final_line_inliers']);fit_lines=[line_map[e] for e in il]
                fit_rows=2*len(ip)+sum(2+len(l['support_query_ids']) for l in fit_lines)
                refit_entry=dict(start_index=index,source=source,dimension_index=dim,cache_reused=False,
                    proposal_index=proposal['proposal_index'],proposal_consensus_point_ids=list(ip),proposal_consensus_line_ids=list(il),
                    removed_point_ids=[k for k in U if k not in ip],removed_line_ids=[e for e in edges if e not in il],actual_optimizer_called=False)
                entry['refit']=refit_entry
                if fit_rows<6:
                    refit_entry.update(rejected='INSUFFICIENT_ACTUAL_INLIER_FACTORS',accepted=False,success=False);failures.append(refit_entry['rejected']);result['optimizer_attempts'].append(entry);continue
                self.counts['refit_start_entries']+=1;refit_key=(dim,np.asarray(proposal['final_z'],dtype='<f8').tobytes(),ip,il)
                if refit_key in refit_cache:
                    self.counts['refit_cache_hits']+=1;candidate,old=copy.deepcopy(refit_cache[refit_key]);refit_entry.update(old,actual_optimizer_called=False,cache_reused=True,cache_source_start_index=old['start_index'],start_index=index,source=source,proposal_index=proposal['proposal_index'])
                else:
                    candidate=self._attempt(model,np.asarray(proposal['final_z']),ip,fit_lines,'refit',refit_entry);refit_cache[refit_key]=(copy.deepcopy(candidate),copy.deepcopy(refit_entry))
                if candidate is None:
                    failures.append(refit_entry['rejected']);result['optimizer_attempts'].append(entry);continue
                score,projected=self._score(model,np.asarray(candidate['final_z']),U,lines);geometry=candidate['point_line_geometry']
                rank_ok=all(geometry[k]['numerical_rank']==6 for k in ('observed_normal_joint','modeled_normal_joint'));weak_pose=geometry['observed_weak'] or geometry['modeled_weak'];stable=set(ip)<=set(score['final_inliers']) and set(il)<=set(score['final_line_inliers'])
                rejected='UNSTABLE_INLIER_REFIT' if not stable else 'INSUFFICIENT_ACTUAL_INLIER_GEOMETRY' if not rank_ok else 'WEAK_OBSERVABILITY' if weak_pose else None
                candidate.update(candidate_index=len(result['all_candidate_solutions']),logical_start_index=index,
                    logical_start_source=source,
                    proposal_index=proposal['proposal_index'],whole_pool_score=score,whole_pool_point_ids=list(U),whole_pool_line_edges=list(edges),
                    final_inliers=score['final_inliers'],final_line_inliers=score['final_line_inliers'],
                    fit_pool_all_final8px_inliers=stable,accepted=rejected is None,rejected=rejected,
                    score_count=score['total_inlier_factor_count'],truncated_sse_px2=score['factor_truncated_sse_px2'],
                    actual_inlier_geometry_is_acceptance_gate=True,whole_pool_scoring_factors_not_all_fit_observations=True)
                result['all_candidate_solutions'].append(candidate);refit_entry.update(candidate_index=candidate['candidate_index'],accepted=candidate['accepted'],rejected=rejected)
                if candidate['accepted']:
                    numeric=Candidate(dim,ip,candidate['candidate_index'],'FINITE_EDGE_CONSENSUS_REFIT',np.asarray(candidate['final_z'][:3]).reshape(3,1),np.asarray(candidate['final_z'][3:]).reshape(3,1),projected);viable.append((numeric,candidate))
                else:failures.append(rejected)
                result['optimizer_attempts'].append(entry)
        result.update(candidate_count=len(result['all_candidate_solutions']),proposal_candidate_count=len(result['proposal_solutions']),
            refit_count=self.counts['refit_optimizer_calls']-before['refit_optimizer_calls'],viable_candidate_count=len(viable),
            local_start_cache=dict(proposal_unique=len(proposal_cache),proposal_reuse=self.counts['proposal_cache_hits']-before['proposal_cache_hits'],refit_unique=len(refit_cache),refit_reuse=self.counts['refit_cache_hits']-before['refit_cache_hits']),failure_reasons=failures)
        categories=dict(numeric=0,insufficient=0,weak=0,unstable=0,other=0)
        for failure in failures:
            category=('insufficient' if failure.startswith('INSUFFICIENT_ACTUAL_INLIER') else
                'weak' if failure=='WEAK_OBSERVABILITY' else 'unstable' if failure=='UNSTABLE_INLIER_REFIT' else
                'numeric' if failure in ('LOCAL_NUMERICAL_FAILURE','FINAL_GEOMETRY_UNAVAILABLE','NONFINITE_OR_NONPOSITIVE_DEPTH','OPTIMIZER_FAILED') else 'other')
            categories[category]+=1
        result['failure_category_counts']=categories
        if not viable:
            active=[key for key,count in categories.items() if count]
            types=dict(numeric='NUMERICAL_FAILURE',insufficient='INSUFFICIENT_ACTUAL_INLIER_GEOMETRY',weak='WEAK_OBSERVABILITY',unstable='UNSTABLE_INLIER_REFIT')
            state=types.get(active[0],'MIXED_NO_VALID_CONSENSUS') if len(active)==1 else 'MIXED_NO_VALID_CONSENSUS'
            result.update(state=state,reason='no_stable_sufficient_weak_safe_actual_inlier_refit');return finish()
        ranked=sorted(viable,key=lambda item:(-item[1]['score_count'],item[1]['truncated_sse_px2'],item[0].dim,item[1]['logical_start_index']))
        chosen,best=ranked[0];result['best_candidate']=copy.deepcopy(best)
        for dim in range(len(self.bank.dims)):
            group=[w for c,w in ranked if c.dim==dim]
            if group:result['per_dimension_best'].append(copy.deepcopy(group[0]))
        ambiguities=[]
        for other,witness in ranked[1:]:
            equivalent=self.bank._equivalent(chosen,other);delta=witness['truncated_sse_px2']-best['truncated_sse_px2']
            tied=witness['score_count']==best['score_count'] and abs(delta)<=POLICY['numerical_score_tie_px2']
            same_support=witness['final_inliers']==best['final_inliers'] and witness['final_line_inliers']==best['final_line_inliers']
            relation=copy.deepcopy(witness);relation.update(numerically_equivalent_to_selected=equivalent,numerical_objective_tie=tied,
                objective_difference_from_selected_px2=float(delta),same_actual8px_consensus_as_selected=same_support)
            result['alternatives'].append(relation);result['numerical_duplicate_solution_count']+=int(equivalent);result['numerical_tied_solution_count']+=int(tied)
            if not equivalent:
                result['multiple_solutions']=True
                if same_support:ambiguities.append('OBSERVATION_CONSENSUS_AMBIGUITY')
                elif tied:ambiguities.append('NUMERICAL_CONSENSUS_TIE')
        if ambiguities:
            result.update(state='OBSERVATION_CONSENSUS_AMBIGUITY' if 'OBSERVATION_CONSENSUS_AMBIGUITY' in ambiguities else 'NUMERICAL_CONSENSUS_TIE',reason='distinct_physical_accepted_branches_share_actual_consensus_or_fixed_numerical_score',unresolved_ambiguity=True,ambiguity_reasons=ambiguities);return finish()
        output=self.points.copy()
        if H:output[list(H)]=chosen.projected[list(H)];self.counts['hidden_reprojection_assignments']+=len(H)
        if not set(U).isdisjoint(H) or not np.array_equal(output[8],self.points[8],equal_nan=True):raise AssertionError('self-hidden or center contract violated')
        result.update(available=True,pose_available=True,new_pose_estimated=True,no_pose=False,state='NEW_POSE',reason='finite_edge_actual_inlier_consensus_local_refit',output_status='NEW_POSE',
            local_refinement_estimated=True,R_cf=best['R_cf'],R_physical=best['R_physical'],centroid=best['centroid'],cf_extents=best['cf_extents'],
            dimension_index=chosen.dim,selected_hypothesis=best['selected_hypothesis'],projected=best['projected'],points_final=output.tolist(),hidden_reprojected=bool(H),
            fit_input_ids=best['fit_input_ids'],fit_line_edges=best['fit_line_edges'],input_inliers=best['fit_input_ids'],
            final_inliers=best['final_inliers'],inliers=best['final_inliers'],final_line_inliers=best['final_line_inliers'],
            point_line_geometry=best['point_line_geometry'],whole_pool_score=best['whole_pool_score'],soft_l1_cost=best['soft_l1_cost'],
            sse_px2=best['sse_px2'],truncated_sse_px2=best['truncated_sse_px2'],
            point_inlier_count=len(best['final_inliers']),unique_line_inlier_count=len(best['final_line_inliers']),total_inlier_factor_count=best['score_count'],
            generator='FINITE_EDGE_CONSENSUS_REFIT',generator_ids=[],accepted_rank_scope='only actually fitted final8px inlier point/edge pool',
            fit_has_fewer_than_four_actual_points=len(best['fit_input_ids'])<4)
        return finish()
