"""Point consensus with one deterministic live whole-U standard closure.

All original four-ID proposals and their local inlier refits remain intact.
The additional standard Generic fits and one LM refit are actual same-q fits,
not stored poses or extra observations. ROOT alone imports and runs this code.
"""
from __future__ import annotations

import ast
import copy
import hashlib
from pathlib import Path

from scripts.research.pallet_finite_support_orthogonal_audit_20261011_v19 import solver as S19

P4, S13, S15 = S19.P, S19.S13, S19.S15
G14, np, cv2, Candidate = S15.G14, S19.np, S19.cv2, S15.Candidate
METHOD = 'HYBRID_POINT_CONSENSUS_WHOLE_U_CLOSURE'
POLICY = dict(schema='whole_U_standard_point_consensus_closure_policy_v21',
    inherited_point_solver_policy=copy.deepcopy(S19.POLICY),
    observation_supply='unchanged actual hybrid q/K/xyz/U/H', point_only=True,
    four_ID_proposals_and_inlier_refits_unchanged=True,
    maximum_four_ID_subsets_per_dimension=70, maximum_registered_dimensions=2,
    extra_standard_problems='one whole-U problem per registered dimension, exact old cache',
    extra_LM_refits='one original ordinary-L2-best raw root, whole U',
    maximum_extra_Generic_calls=2, maximum_extra_LM_calls=1,
    raw_Generic_is_actual_standard_fit_not_LM_refit=True,
    LM_is_actual_refit_with_no_exposed_iterative_status=True,
    closure_SciPy_least_squares_calls=0,
    closure_admission='every actual U stays final8px; unchanged depth/rank/weak/G14',
    ranking='unchanged full-U count/truncated SSE/dimension/logical order',
    numerical_score_tie_px2=S19.DECISION_POLICY['numerical_score_tie_px2'],
    non_equivalent_numerical_tie_veto_retained=True,
    extra_independent_observations=0, extra_initial_or_point_pose_starts=0,
    old_stored_pose_inputs=False, H_and_center_excluded=True,
    final_H_assigned_once_without_refit=True, parent_globals_mutated=False,
    threshold_weights_loss_seeds_head_CAL_changed=False,
    algorithm_coverage_addition_not_proven_implementation_bug_repair=True,
    numerical_NEW_is_not_accuracy_success=True, global_uniqueness_proven=False,
    physical_ownership_certified=False, performance_improvement_proven=False)
EXPECTED_SHA = {
    'S19': '990be7c054917259f4323bb0c26efb68dbb6f2c580ec863f0a5e8f62afdb1d58',
    'S15': '4f447f8fa21ae11d90be19f7423b6c42b8dbde4cd8bacb32450b54dfe4c495a9',
    'S13': '3b8cdeb5bc189cb196be4066d095ead081f8a5c3de1b6ee62c28e9402621e4d7',
    'P4': 'ef827022bef6807c5f7e4016d7ebefa1838c5f013f6d48be6f4932e0c9ad4a40',
    'G14': '17828a9f0ff7f895898cab761d6aee8e20533a7564c49353ecfe3d03ee7bd6ec'}


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(node):
    return ast.dump(node, include_attributes=False)


def source(module, name):
    path = Path(module.__file__); raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == EXPECTED_SHA[name], 'immutable source changed: ' + name)
    return ast.parse(raw), dict(path=str(path), bytes=len(raw), sha256=EXPECTED_SHA[name])


def method(tree, cls, name):
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls]
    require(len(classes) == 1, 'unique source class: ' + cls)
    methods = [n for n in classes[0].body if isinstance(n, ast.FunctionDef) and n.name == name]
    require(len(methods) == 1, 'unique source method: ' + name)
    return copy.deepcopy(methods[0])


def prepare():
    trees, bindings = {}, {}
    for name, module in dict(S19=S19,S15=S15,S13=S13,P4=P4,G14=G14).items():
        trees[name], bindings[name] = source(module, name)
    original = method(trees['S15'], 'FactorSubsetConsensusBank', 'solve')
    repaired = copy.deepcopy(original)
    vetoes = [n for n in ast.walk(repaired) if isinstance(n, ast.If)
              and isinstance(n.test, ast.Name) and n.test.id == 'same_support']
    require(len(vetoes) == 1 and len(vetoes[0].orelse) == 1
        and isinstance(vetoes[0].orelse[0], ast.If)
        and canonical(vetoes[0].orelse[0].test) == canonical(ast.Name(id='tied',ctx=ast.Load())),
        'one original any-cost support veto')
    old_veto = copy.deepcopy(vetoes[0])
    vetoes[0].test, vetoes[0].body, vetoes[0].orelse = copy.deepcopy(old_veto.orelse[0].test), copy.deepcopy(old_veto.orelse[0].body), []
    undo_tie = copy.deepcopy(repaired)
    repaired_veto = [n for n in ast.walk(undo_tie) if isinstance(n, ast.If) and canonical(n) == canonical(vetoes[0])]
    require(len(repaired_veto) == 1, 'unique current true-tie veto')
    repaired_veto[0].test, repaired_veto[0].body, repaired_veto[0].orelse = old_veto.test, old_veto.body, old_veto.orelse
    require(canonical(undo_tie) == canonical(original), 'current true-tie inverse whole AST')
    require(S19.SOURCE_PROOF['solve_inverse_whole_AST_equal'] is True
        and S19.DECISION_POLICY['numerical_score_tie_px2'] == S13.POLICY['numerical_score_tie_px2'],
        'original current decision rule unchanged')
    solved = copy.deepcopy(repaired)
    slots = [i for i,n in enumerate(solved.body) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Attribute) and isinstance(n.value.func.value, ast.Name)
        and n.value.func.value.id == 'result' and n.value.func.attr == 'update'
        and any(k.arg == 'candidate_count' for k in n.value.keywords)]
    require(len(slots) == 1, 'unique final viable-pool insertion boundary')
    insertion = ast.parse('self._append_closure_candidates(result, viable, U, H, lines, failures)').body[0]
    solved.body.insert(slots[0], insertion)
    undone = copy.deepcopy(solved); undone.body.pop(slots[0])
    require(canonical(undone) == canonical(repaired), 'one coverage insertion; otherwise exact current solve')

    ordinary = method(trees['P4'], 'PoseBank', 'solve')
    robust_blocks = [n for n in ordinary.body if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == 'robust']
    require(len(robust_blocks) == 2, 'exact ordinary generation and refit branches')
    generation, refit = robust_blocks[0].orelse, robust_blocks[1].orelse
    require(len(generation) == 3 and len(refit) == 3, 'ordinary source fragment shapes')
    rank = [n for n in ordinary.body if isinstance(n, ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ranked' for t in n.targets)]
    require(len(rank) == 2, 'ordinary prefit and postfit rankings')
    pool_start = next(i for i,n in enumerate(ordinary.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='pool' for t in n.targets))
    pool = ordinary.body[pool_start:pool_start+3]
    require(len(pool)==3 and isinstance(pool[-1],ast.For), 'exact original raw plus returned LM identity pool')
    function = ast.parse('def _live_standard_pool(self, U):\n    result = {}\n').body[0]
    function.body += copy.deepcopy(generation)
    function.body += ast.parse("""
robust = False
if not candidates:
    return [], {}, candidates, [], result
dimension_candidates = candidates
refined = []
regenerated = []
fit_history = {}
""").body
    function.body += [copy.deepcopy(rank[0])] + copy.deepcopy(refit) + copy.deepcopy(pool)
    function.body += ast.parse('return pool, fit_history, candidates, refined, result').body
    require(all(canonical(a)==canonical(b) for a,b in zip(generation,function.body[1:4])), 'generation AST unchanged')
    compiled = ast.Module(body=[function],type_ignores=[])
    ordinary_ns = dict(vars(P4)); ordinary_ns['__name__'] = __name__+'.isolated_standard_pool'
    exec(compile(ast.fix_missing_locations(compiled), str(P4.__file__)+':unchanged ordinary fragments', 'exec'),ordinary_ns)
    solved.name = '_closure_solve'
    namespace = dict(S19._isolated_solve.__globals__); namespace['__name__'] = __name__+'.isolated_current_solve'
    exec(compile(ast.fix_missing_locations(ast.Module(body=[solved],type_ignores=[])),str(__file__)+':one coverage insertion','exec'),namespace)
    proof = dict(schema='point_whole_U_closure_source_proof_v21',source_bindings=bindings,
        solve_inverse_whole_AST_equal=True,solve_coverage_insertions=1,
        original_true_tie_repair_whole_AST_equal=True,ordinary_fragment_AST_changes=0,
        ordinary_fragment_AST_sha256=hashlib.sha256(canonical(compiled).encode()).hexdigest(),
        ordinary_whole_U_H_assignment_called=False,parent_globals_mutated=False,
        point_callbacks_delegate_frozen_S19_S13=True,
        all_original_four_ID_generation_refit_rank_score_H_body_unchanged=True,
        new_coverage_is_not_global_root_completeness_or_accuracy_certificate=True)
    return namespace['_closure_solve'], ordinary_ns['_live_standard_pool'], proof


_closure_solve, _live_standard_pool, SOURCE_PROOF = prepare()


class PointClosureSolver(S19.ObservedFrameFactorSubsetConsensusBank):
    """One bank, unchanged point consensus, plus live full-U standard fits."""
    def __init__(self, bank):
        require(isinstance(bank,P4.PoseBank) and bank.known_dimension_index is None,
                'exact observation bank without external dimension prior required')
        super().__init__(bank.points,bank.K,bank.xyz,image_size=bank.image_size,point_bank=bank)

    def _append_closure_candidates(self, result, viable, U, H, lines, failures):
        require(not lines and len(U)>=4 and set(U).isdisjoint(H) and all(0<=k<8 for k in U), 'point-only whole U without H/center')
        before = self.bank.ledger.copy(); old_counts = self.counts.copy()
        ledger = dict(schema='live_whole_U_standard_closure_ledger_v21',used=list(U),hidden=list(H),
            raw4_bank_counts_before_closure=before.copy(),raw4_solver_counts_before_closure=old_counts,
            bank_before=before,maximum_extra_Generic_calls=len(self.bank.dims),maximum_extra_LM_calls=1,
            maximum_four_ID_subsets_per_dimension=70,extra_independent_observations=0,
            SciPy_closure_refit_entries=0,standard_cache_hit=U in self.bank.standard_cache,candidates=[])
        pool, history, raw, fitted, standard = _live_standard_pool(self.bank,U)
        after = self.bank.ledger.copy(); delta = {k:after[k]-before[k] for k in after}
        require(delta['generic_calls']<=len(self.bank.dims) and delta['lm_calls']<=1,
                'one full-U Generic per dimension and one LM bound')
        require(delta['subset_generic_calls']==delta['refit_generic_calls']==delta['subsets_considered']==0,
                'closure work is never counted as four-ID subset generation')
        ledger.update(bank_after=after,bank_operation_counts=delta,raw_valid_roots=len(raw),LM_valid_returns=len(fitted),
            original_standard_refit_count=standard.get('refit_count',0),
            LM_result='RETURNED_VALID' if fitted else 'NO_VALID_RETURN' if raw else 'NO_RAW_STANDARD_ROOT',
            LM_convergence_status='NOT_EXPOSED_BY_OPENCV',original_standard_producer_AST_changes=0)
        result['closure_ledger']=ledger
        first_start = len(result['proposal_solutions'])
        for index,numeric in enumerate(pool):
            is_lm = id(numeric) in history
            kind = 'WHOLE_U_STANDARD_LM_REFIT' if is_lm else 'WHOLE_U_STANDARD_GENERIC_FIT'
            require(tuple(numeric.ids)==U and (not is_lm or tuple(history[id(numeric)])==U), 'standard fit is exactly whole U')
            entry=dict(index=index,proposal_kind=kind,dimension_index=numeric.dim,
                generator=numeric.generator,generator_ids=list(numeric.ids),solution_index=numeric.solution,
                actual_refit_performed=is_lm,actual_OpenCV_LM_refit=is_lm,
                mathematical_standard_Generic_fit=not is_lm,actual_optimizer_called=False,
                actual_optimizer_called_scope='SCIPY_LEAST_SQUARES_ONLY',
                actual_SciPy_least_squares_called=False,actual_iterative_OpenCV_optimizer_called=is_lm,
                optimizer_success=None,scipy_status=None,accepted=False,rejected=None,candidate_index=None)
            ledger['candidates'].append(entry)
            model=self.bank.models[numeric.dim];z=np.r_[numeric.rvec.reshape(3),numeric.tvec.reshape(3)]
            try:
                if not self.bank._valid(numeric.rvec,numeric.tvec,numeric.dim):
                    entry['rejected']='NONFINITE_OR_NONPOSITIVE_DEPTH';failures.append(entry['rejected']);continue
                geometry,rr,q=self._geometry(model,z,U,[])
                score,projected=self._score(model,z,U,[])
                if not np.isfinite(rr).all() or not np.isfinite(q).all():raise ValueError('NONFINITE_POINT_CLOSURE_GEOMETRY')
            except (ValueError,TypeError,FloatingPointError,np.linalg.LinAlgError,cv2.error) as error:
                entry.update(rejected='FINAL_GEOMETRY_UNAVAILABLE',exception=dict(type=type(error).__name__,message=str(error)))
                failures.append(entry['rejected']);continue
            self.counts['candidate_Rodrigues_calls']+=1
            R=cv2.Rodrigues(numeric.rvec)[0];Q=np.eye(3) if numeric.dim==0 else np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
            stable=set(U)<=set(score['final_inliers'])
            rank=all(geometry[k]['numerical_rank']==6 for k in ('observed_normal_joint','modeled_normal_joint'))
            weak=geometry['observed_weak'] or geometry['modeled_weak']
            rejected='UNSTABLE_INLIER_REFIT' if not stable else 'INSUFFICIENT_ACTUAL_INLIER_GEOMETRY' if not rank else 'WEAK_OBSERVABILITY' if weak else None
            candidate=dict(phase='whole_U_standard_fit',source='LIVE_SAME_Q_WHOLE_U_STANDARD',
                proposal_kind=kind,start_index=first_start+index,logical_start_index=first_start+index,
                logical_start_source='LIVE_SAME_Q_WHOLE_U_STANDARD',dimension_index=numeric.dim,
                selected_hypothesis=self.bank.names[numeric.dim],point_generator=numeric.generator,
                point_generator_ids=list(numeric.ids),point_solution_index=numeric.solution,
                final_z=z.tolist(),R_cf=R.tolist(),R_physical=(R@Q).tolist(),centroid=z[3:].tolist(),
                cf_extents=self.bank.dims[numeric.dim].tolist(),projected=projected.tolist(),
                model_camera_depths_m=((R@model.T).T+z[3:])[:,2].tolist(),
                fit_input_ids=list(U),fit_line_edges=[],scalar_residuals_px=rr.tolist(),
                soft_l1_cost=S19.V9.soft_l1_cost(rr),soft_l1_cost_is_diagnostic_not_this_fit_objective=True,
                sse_px2=float(rr@rr),actual_fit_objective='WHOLE_U_L2_REPROJECTION_LM' if is_lm else 'OPENCV_GENERIC_STANDARD_PROBLEM',
                standard_candidate_selection_objective='WHOLE_U_PIXEL_SSE',
                raw_generator_internal_objective_not_independently_certified=not is_lm,
                optimizer_success=None,scipy_status=None,actual_optimizer_called=False,
                actual_optimizer_called_scope='SCIPY_LEAST_SQUARES_ONLY',
                actual_SciPy_least_squares_called=False,actual_iterative_OpenCV_optimizer_called=is_lm,
                actual_refit_performed=is_lm,actual_OpenCV_LM_refit=is_lm,
                mathematical_standard_Generic_fit=not is_lm,LM_convergence_status='NOT_EXPOSED_BY_OPENCV',
                point_line_geometry=geometry,whole_pool_score=score,whole_pool_point_ids=list(U),whole_pool_line_edges=[],
                final_inliers=score['final_inliers'],final_line_inliers=[],fit_pool_all_final8px_inliers=stable,
                score_count=score['total_inlier_factor_count'],truncated_sse_px2=score['factor_truncated_sse_px2'],
                candidate_index=len(result['all_candidate_solutions']),accepted=rejected is None,rejected=rejected,
                actual_inlier_geometry_is_acceptance_gate=True,whole_pool_scoring_factors_not_all_fit_observations=False,
                initial_pose_residual_prior=False,local_only=True,global_uniqueness_proven=False)
            if rejected is None:
                witness=self._incidence(candidate,U,[],[],H);candidate['stable_incidence_guard']=witness
                if not witness['accepted']:candidate.update(accepted=False,rejected=witness['guard_state'])
            else:candidate['stable_incidence_guard_not_called_reason']='ORIGINAL_POINT_CANDIDATE_GATE_REJECTED'
            result['all_candidate_solutions'].append(candidate)
            entry.update(candidate_index=candidate['candidate_index'],accepted=candidate['accepted'],rejected=candidate['rejected'])
            if candidate['accepted']:
                scored_numeric=Candidate(numeric.dim,U,numeric.solution,numeric.generator,
                    numeric.rvec.copy(),numeric.tvec.copy(),projected)
                viable.append((scored_numeric,candidate))
            else:failures.append(candidate['rejected'])
        ledger['closure_solver_operation_counts']={k:self.counts[k]-old_counts[k] for k in self.counts}
        ledger['accepted_candidates']=sum(c['accepted'] for c in ledger['candidates'])
        ledger['rejected_candidates']=sum(not c['accepted'] for c in ledger['candidates'])
        ledger['original_raw_four_ID_proposal_count']=len(result['proposal_solutions'])
        ledger['closure_candidate_count']=len(pool)

    def solve(self,used,hidden=(),line_blocks=(),*,observation=None,initial_pose=None,point_result=None):
        H=P4._ids(hidden,'hidden');U=tuple(k for k in self.bank.eligible if k not in H)
        require(P4._ids(used,'used')==U,'used must equal original actual eligible minus H')
        require(not line_blocks,'this closure is point-only; no line blocks permitted')
        observation=dict(lines=[]) if observation is None else copy.deepcopy(observation)
        require(isinstance(observation,dict) and not observation.get('lines',[]),'point-only observation required')
        result=_closure_solve(self,observation,(),H,initial_pose=initial_pose,point_result=point_result)
        result.update(solver=METHOD,inherited_solver_policy=result['policy'],policy=copy.deepcopy(POLICY),
            operative_output_decision_policy=copy.deepcopy(S19.DECISION_POLICY),
            inherited_solver_policy_is_candidate_generation_history=True,
            whole_U_closure_source_proof=copy.deepcopy(SOURCE_PROOF),
            closure_is_algorithm_coverage_addition_not_accuracy_or_bug_certificate=True)
        best=result.get('best_candidate')
        if result['available'] and best is not None and best.get('proposal_kind','').startswith('WHOLE_U_STANDARD_'):
            result['local_refinement_estimated']=best['actual_refit_performed']
        if 'closure_ledger' not in result:
            result['closure_ledger']=dict(schema='live_whole_U_standard_closure_ledger_v21',skipped=True,
                reason=result['state'],used=list(U),hidden=list(H),candidates=[],accepted_candidates=0,rejected_candidates=0,
                closure_candidate_count=0,raw_valid_roots=0,LM_valid_returns=0,SciPy_closure_refit_entries=0,
                maximum_extra_Generic_calls=len(self.bank.dims),maximum_extra_LM_calls=1,
                extra_independent_observations=0)
        return result
