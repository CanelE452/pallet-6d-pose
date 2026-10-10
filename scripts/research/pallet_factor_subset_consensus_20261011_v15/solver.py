"""Generate independent factor-subset proposals before whole-pool consensus.

Finite point roots are direct geometric proposals, including unselected roots.
Three mixed factor blocks supply necessary six incidence rows, not a proof of
unique pose. Their local optimization uses the same original N3 start only.
The unchanged finite-edge fit and stable incidence guard decide each refit.
"""
from __future__ import annotations
import copy
from itertools import combinations
from math import comb
import cv2
import numpy as np

from ..pallet_observation_refiner_20261009_v1.solver import Candidate
from ..pallet_cornerwise_independent_20261010_v4 import pose as P
from ..pallet_sparse_local_line_20261010_v9 import solver as V9
from ..pallet_finite_edge_consensus_repair_20261011_v13 import solver as S13
from ..pallet_stable_incidence_guard_20261011_v14 import pipeline as G14

METHOD='FACTOR_SUBSET_FINITE_EDGE_CONSENSUS'
EDGES=S13.EDGES
POLICY=dict(schema='finite_point_and_mixed_factor_subset_consensus_policy_v15',
    frozen_source_finite_policy=copy.deepcopy(S13.POLICY),stable_incidence=copy.deepcopy(G14.POLICY),
    source_V13_whole_pool_start_and_call_caps_replaced_by_subset_enumeration=True,
    proposal_sources='all valid same-bank four-point roots; all three-factor mixed subsets with at least one unused physical edge',
    direct_point_roots_conditioned_on_available_or_selected=False,
    direct_point_generator='unchanged V4 cached SQPNP/IPPE Generic on four actual IDs',
    maximum_point_corner_ids=8,maximum_unused_edge_ids=12,maximum_factor_blocks=20,
    maximum_four_ID_subsets_per_dimension=70,maximum_registered_dimensions=2,
    maximum_mixed_three_block_subsets=1140,maximum_mixed_LOCAL_proposals=2280,
    maximum_LOCAL_optimizer_calls='4560 + R; R is actual valid direct four-ID roots, not LM/refit aliases',
    mixed_LOCAL_start='original available initial N3 physical pose remapped to each registry dimension',
    selected_point_pose_is_additional_LOCAL_start=False,three_point_only_subsets=False,
    mixed_subset_exhaustive=True,minimum_subset_incidence_scalar_rows=6,
    point_only_actual_inlier_refit_minimum=4,
    minimum_six_rows_is_sufficiency=False,mixed_LOCAL_root_completeness_proven=False,
    subset_objective_used_for_final_ranking=False,
    final_score='unchanged full-pool point+one-edge vote and truncated factor SSE',
    refit='one fixed actual-inlier refit per distinct proposal pose/dimension/inlier pool',
    all_refit_branches_retained=True,stable_incidence_required_per_accepted_candidate=True,
    numeric_pose_cache='exact dimension, pose bytes and same fit IDs; distinct physical branches are retained',
    final_ambiguity='unchanged non-equivalent accepted branches with the same final actual consensus, or numerical consensus tie',
    random_starts=False,new_training_RGB_seeds=False,new_threshold_search=False,
    oracle_quality_or_GT_input=False,initial_pose_residual_prior=False,
    physical_ownership_certified=False,global_uniqueness_proven=False,known_DEV=True,
    H_initial_coordinates_excluded=True,center_used_for_fit_or_score=False,
    accepted_NEW_H_projected_once=True,reprojected_observations_refit=False)


def _copy(v):return copy.deepcopy(v)


class FactorSubsetConsensusBank(S13.FiniteEdgeConsensusBank):
    """Reuse S13 kernels/source checks and V4 cache without calling S13.solve."""
    def __init__(self,points,K,xyz,image_size=(640,480),*,point_bank=None):
        super().__init__(points,K,xyz,image_size=image_size)
        self.point_bank_reused=point_bank is not None
        if point_bank is not None:
            if not isinstance(point_bank,P.PoseBank) or point_bank.digest!=self.bank.digest or point_bank.known_dimension_index is not None:
                raise ValueError('shared point bank must match actual q/K/xyz/image size without dimension prior')
            self.bank=point_bank;self.factor_validator.bank=point_bank
            self.points,self.K,self.xyz=point_bank.points,point_bank.K,point_bank.xyz
            self.factor_validator.points,self.factor_validator.K,self.factor_validator.xyz=self.points,self.K,self.xyz
        self.direct_generation_state_cache={}
        self.counts.update(direct_four_ID_requests=0,direct_point_roots=0,
            direct_geometry_calls=0,direct_geometry_failures=0,
            cached_empty_root_layout_diagnostic_calls=0,
            mixed_subset_requests=0,mixed_dimension_requests=0,
            whole_pool_proposal_scores=0,inlier_refit_requests=0,
            stable_incidence_guard_calls=0,stable_incidence_geometry_calls=0,
            stable_incidence_geometry_projection_entries=0)

    def _empty(self,U,H,observation):
        return dict(available=False,pose_available=False,new_pose_estimated=False,no_pose=True,
            fallback_used=False,state='INSUFFICIENT_ACTUAL_INCIDENCE',reason='fewer_than_six_actual_incidence_rows',
            output_status='NO_NEW_POSE',solver=METHOD,policy=_copy(POLICY),
            eligible=list(self.bank.eligible),used=list(U),excluded=list(H),hidden=list(H),actual_self_hidden_ids=list(H),
            ineligible=_copy(self.bank.ineligible),input_hash=self.bank.digest,
            fit_input_ids=[],input_inliers=[],final_inliers=[],inliers=[],fit_line_edges=[],final_line_inliers=[],
            line_edges=[],consumed_edges=[],proposal_solutions=[],all_candidate_solutions=[],optimizer_attempts=[],
            direct_point_generation=[],mixed_subset_enumeration=[],start_source_witnesses=[],
            alternatives=[],per_dimension_best=[],candidate_count=0,refit_count=0,
            R_cf=None,R_physical=None,centroid=None,cf_extents=None,projected=None,points_final=self.points.tolist(),
            multiple_solutions=False,unresolved_ambiguity=False,numerical_tied_solution_count=0,
            numerical_duplicate_solution_count=0,prior_used=False,initial_pose_used=False,initial_pose_start_used=False,
            initial_pose_residual_prior=False,initial_projection_used=False,initial_dimension_prior_used=False,
            known_dimension_constraint_used=False,independent_PnP_or_PnL=False,local_initialization_only=True,
            rank_is_local_only=True,global_uniqueness_proven=False,hidden_initial_excluded=True,
            hidden_reprojected=False,reprojected_points_reused_as_observations=False,
            same_edge_point_line_double_count=False,derived_line_points=0,
            excluded_image_coordinates_used_for_scoring=False,excluded_image_coordinates_used_for_equivalence=False,
            equivalence_uses_all_eight_model_predictions=True,center_used_for_fit_or_score=False,
            no_new_coordinate_or_native_numeric_fill=True,inlier_geometry_is_acceptance_gate=True,
            inlier_counts_are_diagnostic_only=False,local_refinement_estimated=False,
            observation_binding=V9._binding(observation),source_line_support_is_not_real_ownership_certificate=True,
            direct_point_coordinate_bank_reused=self.point_bank_reused,
            direct_point_roots_do_not_require_point_solver_available=True,
            mixed_root_completeness_proven=False,point_only_and_partial_line_paths_are_separate_ablation_components=True)

    def _direct(self,numeric,entry):
        """Expose the same raw Generic root without fitting it to all factors."""
        z=np.r_[numeric.rvec.reshape(3),numeric.tvec.reshape(3)]
        self.counts['direct_geometry_calls']+=1
        try:geometry,residual,q=self._geometry(self.bank.models[numeric.dim],z,numeric.ids,[])
        except (ValueError,FloatingPointError,np.linalg.LinAlgError,cv2.error) as error:
            self.counts['direct_geometry_failures']+=1
            entry.update(rejected='DIRECT_POINT_GEOMETRY_UNAVAILABLE',exception=dict(type=type(error).__name__,message=str(error)))
            return None
        self.counts['candidate_Rodrigues_calls']+=1
        R=cv2.Rodrigues(numeric.rvec)[0];Q=np.eye(3) if numeric.dim==0 else np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
        return dict(phase='direct_point_proposal',source='ALL_VALID_FINITE_FOUR_POINT_ROOTS',
            start_index=entry['start_index'],dimension_index=numeric.dim,selected_hypothesis=self.bank.names[numeric.dim],
            point_generator=numeric.generator,point_generator_ids=list(numeric.ids),point_solution_index=numeric.solution,
            final_z=z.tolist(),R_cf=R.tolist(),R_physical=(R@Q).tolist(),centroid=z[3:].tolist(),
            cf_extents=self.bank.dims[numeric.dim].tolist(),projected=q.tolist(),
            model_camera_depths_m=((R@self.bank.models[numeric.dim].T).T+z[3:])[:,2].tolist(),
            fit_input_ids=list(numeric.ids),fit_line_edges=[],scalar_residuals_px=residual.tolist(),
            soft_l1_cost=V9.soft_l1_cost(residual),sse_px2=float(residual@residual),point_line_geometry=geometry,
            optimizer_success=None,actual_optimizer_called=False,accepted=False,rejected=None,
            initial_pose_residual_prior=False,local_only=False,global_uniqueness_proven=False)

    def _incidence(self,candidate,U,lines,consumed,H):
        """Normal-only observability of this candidate; never reselect or fit."""
        temporary=dict(available=True,state='CANDIDATE_FOR_STABLE_INCIDENCE',used=list(U),
            fit_input_ids=list(candidate['fit_input_ids']),fit_line_edges=list(candidate['fit_line_edges']),
            consumed_edges=list(consumed),line_edges=[l['edge'] for l in lines],
            actual_source_lines=[dict(edge=l['edge'],normal=l['normal'].tolist(),offset=l['offset']) for l in lines],
            centroid=candidate['centroid'],cf_extents=candidate['cf_extents'],R_cf=candidate['R_cf'],
            R_physical=candidate['R_physical'],projected=candidate['projected'],
            best_candidate=_copy(candidate),all_candidate_solutions=[_copy(candidate)])
        self.counts['stable_incidence_guard_calls']+=1
        _,w=G14.guard_solver(temporary,self.points,self.K,self.xyz,H)
        self.counts['stable_incidence_geometry_calls']+=w['counts']['geometry_info_calls']
        self.counts['stable_incidence_geometry_projection_entries']+=w['counts']['geometry_projection_entries']
        return w

    def solve(self,observation,adopted_boundary_corner_ids=(),hidden=(),*,initial_pose=None,point_result=None):
        before=self.counts.copy();bank_before=self.bank.ledger.copy();self.counts['local_solve_calls']+=1
        H=P._ids(hidden,'hidden');adopted=P._ids(adopted_boundary_corner_ids,'adopted_boundary_corner_ids')
        U=tuple(k for k in self.bank.eligible if k not in H);result=self._empty(U,H,observation)
        def finish():
            result['operation_counts']={k:self.counts[k]-before[k] for k in self.counts}
            result['hypothesis_bank_operation_counts']={k:self.bank.ledger[k]-bank_before[k] for k in self.bank.ledger}
            result['hypothesis_bank_counts']=self.bank.ledger.copy();result['local_bank_counts']=self.counts.copy()
            result['initial_pose_used']=result['initial_pose_start_used'];return result
        try:lines,consumed,checks=self.factor_validator._lines(observation,adopted,set(U))
        except (KeyError,TypeError,ValueError,IndexError) as error:
            result.update(state='INVALID_OBSERVATION_CONTRACT',reason=str(error));return finish()
        edges=tuple(l['edge'] for l in lines);line_map={l['edge']:l for l in lines}
        result.update(line_edges=list(edges),consumed_edges=consumed,line_contract_checks=checks,
            actual_adopted_boundary_ids=[k for k in adopted if k in U],
            actual_source_lines=[dict(edge=l['edge'],normal=l['normal'].tolist(),offset=l['offset'],raw=_copy(l['source_line'])) for l in lines],
            factor_pool=dict(point_ids=list(U),line_edges=list(edges),actual_point_factors=len(U),unique_line_factors=len(lines),
                normal_incidence_scalar_rows=2*(len(U)+len(lines)),
                scalar_residuals=2*len(U)+sum(2+len(l['support_query_ids']) for l in lines)),
            factor_pool_binding=V9._binding(dict(points=self.points[list(U)],point_ids=U,lines=lines)),
            registered_dimensions=[dict(index=i,name=self.bank.names[i],extents=d.tolist()) for i,d in enumerate(self.bank.dims)])
        F=len(U)+len(lines);M=comb(F,3)-comb(len(U),3) if F>=3 else 0
        D=len(self.bank.dims);result['call_bounds']=dict(point_count=len(U),line_count=len(lines),factor_blocks=F,
            registered_dimensions=D,four_ID_subsets=comb(len(U),4) if len(U)>=4 else 0,
            maximum_direct_Generic_calls=D*(comb(len(U),4) if len(U)>=4 else 0),
            mixed_subsets=M,mixed_LOCAL_proposal_calls_at_most=D*M,universal_mixed_subsets=1140,
            universal_mixed_LOCAL_proposals=2280,direct_valid_roots=0,LOCAL_optimizer_calls_at_most=None)
        if F>20 or D>2:raise ValueError('fixed physical factor/registry bound exceeded')
        if not lines and len(U)<4:
            result.update(state='INSUFFICIENT_POINT_OBSERVATIONS',reason='point_only_standard_path_requires_at_least_four_actual_corners');return finish()
        if 2*F<6:return finish()
        if not S13.V5._camera(self.K):result.update(state='UNSUPPORTED_CAMERA_CONTRACT',reason='zero_skew_pinhole_K_required');return finish()
        result['point_result_is_selected_start_only_not_used']=True
        result['point_result_supplied_but_available_verdict_not_a_filter']=isinstance(point_result,dict)
        proposals=[];failures=[];subset_cache={};refit_cache={};viable=[]
        # Generate every valid raw four-ID branch on actual unmasked coordinates.
        for dim in range(D):
            for ids in combinations(U,4):
                self.counts['direct_four_ID_requests']+=1
                generic_before=self.bank.ledger.copy();roots=self.bank._generic(ids,dim,'subset')
                delta={k:self.bank.ledger[k]-generic_before[k] for k in self.bank.ledger}
                state_key=(dim,ids)
                if state_key not in self.direct_generation_state_cache:
                    if roots:state='VALID_GENERIC_ROOTS'
                    elif delta['geometry_rejected']:state='DEGENERATE_POINT_SUBSET'
                    elif delta.get('generic_cache_hits',0):
                        # The legacy cache stores no per-key rejection reason.
                        # Recount its unchanged layout criterion, not Generic.
                        self.counts['cached_empty_root_layout_diagnostic_calls']+=1
                        degenerate=P._shape(self.bank.models[dim][list(ids)])['numerical_rank']<2 or P._shape(self.points[list(ids)])['numerical_rank']<2
                        state='DEGENERATE_POINT_SUBSET' if degenerate else 'NUMERICAL_GENERATION_FAILURE'
                    else:state='NUMERICAL_GENERATION_FAILURE'
                    self.direct_generation_state_cache[state_key]=state
                generation=dict(dimension_index=dim,generator_ids=list(ids),returned_valid_roots=len(roots),
                    original_available_or_best_not_used=True,roots=[],
                    bank_operation_counts=delta,
                    generation_state=self.direct_generation_state_cache[state_key])
                for numeric in roots:
                    self.counts['direct_point_roots']+=1
                    entry=dict(start_index=len(proposals),dimension_index=dim,generator_ids=list(ids),
                        solution_index=numeric.solution,generator=numeric.generator,actual_optimizer_called=False)
                    proposal=self._direct(numeric,entry);generation['roots'].append(_copy(entry))
                    if proposal is None:failures.append(entry['rejected']);continue
                    proposal['proposal_kind']='DIRECT_FINITE_FOUR_POINT_ROOT';proposals.append(proposal)
                result['direct_point_generation'].append(generation)
        self.counts['start_sources_examined']+=1;seed,seed_witness=self._seed(initial_pose,'ORIGINAL_INITIAL_N3_ONLY')
        result['start_source_witnesses'].append(seed_witness)
        blocks=[('POINT',k) for k in U]+[('EDGE',e) for e in edges]
        for subset in combinations(blocks,3):
            ids=tuple(v for kind,v in subset if kind=='POINT');es=tuple(v for kind,v in subset if kind=='EDGE')
            if not es:continue
            self.counts['mixed_subset_requests']+=1;sub=dict(index=len(result['mixed_subset_enumeration']),
                factor_blocks=[list(v) for v in subset],point_ids=list(ids),line_edges=list(es),
                normal_incidence_scalar_rows=6,necessary_not_sufficient=True,branches=[])
            for dim,model in enumerate(self.bank.models):
                self.counts['mixed_dimension_requests']+=1
                entry=dict(start_index=len(proposals),source='INITIAL_N3_MIXED_THREE_FACTOR_SUBSET',
                    dimension_index=dim,subset_index=sub['index'],subset_point_ids=list(ids),subset_line_edges=list(es),
                    actual_optimizer_called=False,cache_reused=False)
                if seed is None:
                    entry.update(accepted=False,rejected='NO_LOCAL_INITIALIZATION');sub['branches'].append(entry)
                    failures.append(entry['rejected']);continue
                R,t=seed;Q=np.eye(3) if dim==0 else np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
                self.counts['seed_Rodrigues_calls']+=1;z0=np.r_[cv2.Rodrigues(R@Q.T)[0].reshape(3),t]
                result['initial_pose_start_used']=True;self.counts['local_start_entries']+=1;self.counts['proposal_start_entries']+=1
                key=(dim,ids,es,z0.astype('<f8').tobytes())
                if key in subset_cache:
                    self.counts['proposal_cache_hits']+=1;self.counts['local_start_cache_hits']+=1
                    proposal,old=_copy(subset_cache[key]);entry.update(old,actual_optimizer_called=False,cache_reused=True)
                else:
                    proposal=self._attempt(model,z0,ids,[line_map[e] for e in es],'proposal',entry)
                    subset_cache[key]=(_copy(proposal),_copy(entry))
                sub['branches'].append(_copy(entry))
                if proposal is None:failures.append(entry['rejected']);continue
                proposal.update(proposal_kind='MIXED_THREE_FACTOR_LOCAL_SUBSET',subset_index=sub['index'],
                    subset_factor_blocks=_copy(sub['factor_blocks']),subset_root_completeness_proven=False)
                proposals.append(proposal)
            result['mixed_subset_enumeration'].append(sub)
        R=self.counts['direct_point_roots']-before['direct_point_roots']
        result['call_bounds'].update(direct_valid_roots=R,LOCAL_optimizer_calls_at_most=2*D*M+R,
            universal_LOCAL_optimizer_calls_at_most=4560+R)
        for index,proposal in enumerate(proposals):
            dim=proposal['dimension_index'];model=self.bank.models[dim]
            proposal.update(proposal_index=index,logical_start_index=index,logical_start_source=proposal['source'])
            self.counts['whole_pool_proposal_scores']+=1
            score,_=self._score(model,np.asarray(proposal['final_z']),U,lines);proposal['whole_pool_score']=score
            result['proposal_solutions'].append(_copy(proposal));ip=tuple(score['final_inliers']);il=tuple(score['final_line_inliers'])
            entry=dict(start_index=index,source=proposal['source'],dimension_index=dim,proposal_index=index,
                proposal_kind=proposal['proposal_kind'],proposal_consensus_point_ids=list(ip),proposal_consensus_line_ids=list(il),
                removed_point_ids=[k for k in U if k not in ip],removed_line_ids=[e for e in edges if e not in il],
                actual_optimizer_called=False,cache_reused=False)
            attempt=dict(start_index=index,source=proposal['source'],dimension_index=dim,
                proposal_kind=proposal['proposal_kind'],proposal_index=index,refit=entry,initial_pose_residual_prior=False)
            if (not il and len(ip)<4) or 2*(len(ip)+len(il))<6:
                entry.update(accepted=False,rejected='INSUFFICIENT_POINT_INLIER_OBSERVATIONS' if not il and len(ip)<4
                    else 'INSUFFICIENT_ACTUAL_INLIER_INCIDENCE');failures.append(entry['rejected'])
                result['optimizer_attempts'].append(attempt);continue
            self.counts['inlier_refit_requests']+=1;self.counts['refit_start_entries']+=1
            fit_lines=[line_map[e] for e in il];key=(dim,np.asarray(proposal['final_z'],dtype='<f8').tobytes(),ip,il)
            if key in refit_cache:
                self.counts['refit_cache_hits']+=1;candidate,old=_copy(refit_cache[key]);entry.update(old,
                    actual_optimizer_called=False,cache_reused=True,cache_source_start_index=old['start_index'],
                    start_index=index,source=proposal['source'],proposal_index=index)
            else:
                candidate=self._attempt(model,np.asarray(proposal['final_z']),ip,fit_lines,'refit',entry)
                refit_cache[key]=(_copy(candidate),_copy(entry))
            if candidate is None:
                failures.append(entry['rejected']);result['optimizer_attempts'].append(attempt);continue
            score,projected=self._score(model,np.asarray(candidate['final_z']),U,lines);geometry=candidate['point_line_geometry']
            stable=set(ip)<=set(score['final_inliers']) and set(il)<=set(score['final_line_inliers'])
            rank=all(geometry[k]['numerical_rank']==6 for k in ('observed_normal_joint','modeled_normal_joint'))
            weak=geometry['observed_weak'] or geometry['modeled_weak']
            rejected='UNSTABLE_INLIER_REFIT' if not stable else 'INSUFFICIENT_ACTUAL_INLIER_GEOMETRY' if not rank else 'WEAK_OBSERVABILITY' if weak else None
            candidate.update(candidate_index=len(result['all_candidate_solutions']),logical_start_index=index,
                logical_start_source=proposal['source'],proposal_kind=proposal['proposal_kind'],proposal_index=index,
                whole_pool_score=score,whole_pool_point_ids=list(U),whole_pool_line_edges=list(edges),
                final_inliers=score['final_inliers'],final_line_inliers=score['final_line_inliers'],
                fit_pool_all_final8px_inliers=stable,accepted=rejected is None,rejected=rejected,
                score_count=score['total_inlier_factor_count'],truncated_sse_px2=score['factor_truncated_sse_px2'],
                actual_inlier_geometry_is_acceptance_gate=True,whole_pool_scoring_factors_not_all_fit_observations=True)
            if rejected is None:
                witness=self._incidence(candidate,U,lines,consumed,H);candidate['stable_incidence_guard']=witness
                if not witness['accepted']:candidate.update(accepted=False,rejected=witness['guard_state'])
            else:candidate['stable_incidence_guard_not_called_reason']='ORIGINAL_FINITE_CANDIDATE_GATE_REJECTED'
            result['all_candidate_solutions'].append(candidate);entry.update(candidate_index=candidate['candidate_index'],
                accepted=candidate['accepted'],rejected=candidate['rejected'])
            if candidate['accepted']:
                numeric=Candidate(dim,ip,index,'FACTOR_SUBSET_ACTUAL_INLIER_REFIT',np.asarray(candidate['final_z'][:3]).reshape(3,1),
                    np.asarray(candidate['final_z'][3:]).reshape(3,1),projected)
                viable.append((numeric,candidate))
            else:failures.append(candidate['rejected'])
            result['optimizer_attempts'].append(attempt)
        result.update(candidate_count=len(result['all_candidate_solutions']),proposal_candidate_count=len(result['proposal_solutions']),
            refit_count=self.counts['refit_optimizer_calls']-before['refit_optimizer_calls'],viable_candidate_count=len(viable),
            local_start_cache=dict(proposal_unique=len(subset_cache),proposal_reuse=self.counts['proposal_cache_hits']-before['proposal_cache_hits'],
                refit_unique=len(refit_cache),refit_reuse=self.counts['refit_cache_hits']-before['refit_cache_hits']),failure_reasons=failures)
        if len(result['mixed_subset_enumeration'])!=M:raise ValueError('exhaustive mixed subset enumeration incomplete')
        if self.counts['local_optimizer_calls']-before['local_optimizer_calls']>2*D*M+R:raise ValueError('fixed optimizer work bound exceeded')
        categories=dict(numeric=0,insufficient=0,weak=0,unstable=0,initialization=0,other=0)
        for failure in failures:
            category='insufficient' if 'INSUFFICIENT' in failure else 'weak' if 'WEAK' in failure else 'unstable' if failure=='UNSTABLE_INLIER_REFIT' else 'initialization' if failure=='NO_LOCAL_INITIALIZATION' else 'numeric' if failure in ('DIRECT_POINT_GEOMETRY_UNAVAILABLE','LOCAL_NUMERICAL_FAILURE','FINAL_GEOMETRY_UNAVAILABLE','NONFINITE_OR_NONPOSITIVE_DEPTH','OPTIMIZER_FAILED','STABLE_INCIDENCE_NUMERICAL_FAILURE') else 'other'
            categories[category]+=1
        result['failure_category_counts']=categories
        if not viable:
            if not proposals and M==0 and len(U)>=4:
                degenerate=bool(result['direct_point_generation']) and all(g['generation_state']=='DEGENERATE_POINT_SUBSET' for g in result['direct_point_generation'])
                result.update(state='DEGENERATE_POINT_OBSERVATIONS' if degenerate else 'NUMERICAL_GENERATION_FAILURE',
                    reason='all_four_ID_subsets_layout_degenerate' if degenerate else 'no_valid_finite_point_roots');return finish()
            active=[k for k,v in categories.items() if v];states=dict(numeric='NUMERICAL_FAILURE',insufficient='INSUFFICIENT_ACTUAL_INLIER_GEOMETRY',weak='WEAK_OBSERVABILITY',unstable='UNSTABLE_INLIER_REFIT',initialization='NO_LOCAL_INITIALIZATION')
            result.update(state=states.get(active[0],'MIXED_NO_VALID_CONSENSUS') if len(active)==1 else 'MIXED_NO_VALID_CONSENSUS',
                reason='no_sufficient_stable_incidence_actual_inlier_refit');return finish()
        ranked=sorted(viable,key=lambda x:(-x[1]['score_count'],x[1]['truncated_sse_px2'],x[0].dim,x[1]['logical_start_index']))
        chosen,best=ranked[0];result['best_candidate']=_copy(best)
        for dim in range(D):
            group=[w for c,w in ranked if c.dim==dim]
            if group:result['per_dimension_best'].append(_copy(group[0]))
        ambiguity=[]
        for other,w in ranked[1:]:
            equivalent=self.bank._equivalent(chosen,other);delta=w['truncated_sse_px2']-best['truncated_sse_px2']
            tied=w['score_count']==best['score_count'] and abs(delta)<=S13.POLICY['numerical_score_tie_px2']
            same_support=w['final_inliers']==best['final_inliers'] and w['final_line_inliers']==best['final_line_inliers']
            relation=_copy(w);relation.update(numerically_equivalent_to_selected=equivalent,numerical_objective_tie=tied,
                objective_difference_from_selected_px2=float(delta),same_actual8px_consensus_as_selected=same_support)
            result['alternatives'].append(relation);result['numerical_duplicate_solution_count']+=int(equivalent);result['numerical_tied_solution_count']+=int(tied)
            if not equivalent:
                result['multiple_solutions']=True
                if same_support:ambiguity.append('OBSERVATION_CONSENSUS_AMBIGUITY')
                elif tied:ambiguity.append('NUMERICAL_CONSENSUS_TIE')
        if ambiguity:
            result.update(state='AMBIGUOUS_FINITE_CONSENSUS',reason=';'.join(sorted(set(ambiguity))),unresolved_ambiguity=True);return finish()
        output=self.points.copy()
        if H:output[list(H)]=chosen.projected[list(H)];self.counts['hidden_reprojection_assignments']+=len(H)
        result.update(available=True,pose_available=True,new_pose_estimated=True,no_pose=False,state='NEW_POSE',reason='new_factor_subset_actual_inlier_pose',
            output_status='NEW_POSE',local_refinement_estimated=True,R_cf=best['R_cf'],R_physical=best['R_physical'],centroid=best['centroid'],
            cf_extents=best['cf_extents'],dimension_index=chosen.dim,selected_hypothesis=best['selected_hypothesis'],projected=best['projected'],
            fit_input_ids=best['fit_input_ids'],input_inliers=best['fit_input_ids'],final_inliers=best['final_inliers'],inliers=best['final_inliers'],
            fit_line_edges=best['fit_line_edges'],final_line_inliers=best['final_line_inliers'],
            point_inlier_count=len(best['final_inliers']),unique_line_inlier_count=len(best['final_line_inliers']),
            total_inlier_factor_count=best['score_count'],points_final=output.tolist(),hidden_reprojected=bool(H),
            point_line_geometry=_copy(best['point_line_geometry']),stable_incidence_guard=_copy(best['stable_incidence_guard']),
            scalar_residuals_px=best['scalar_residuals_px'],soft_l1_cost=best['soft_l1_cost'],sse_px2=best['sse_px2'],
            truncated_sse_px2=best['truncated_sse_px2'],factor_sse_px2=best['whole_pool_score']['factor_sse_px2'],
            residuals_used_px=best['whole_pool_score']['point_residual_norm_px'],line_residual_RMS_px=best['whole_pool_score']['edge_block_norm_px'])
        return finish()
