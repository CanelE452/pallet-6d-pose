"""Actual CPU synthetic checks of conditional observation-only independence.

Run only after peer review and freezing. No images, models, GPU, source cache,
private truth, rays, training or real accuracy/timing evaluation are used.
Fixed H and remaining observations define the independence test; inherited H
and Base-derived proposal features are not claimed statistically independent.
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

from . import pose as P
from . import selection as S
from ..pallet_observation_refiner_20261009_v1.solver import Candidate, cuboid, project, visibility

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
DOC = ROOT / '_docs/experiments/pallet_cornerwise_independent_20261010_v4'
SIZE = (1280, 960)
PIXEL_ATOL = 1e-6
EDGES = ((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))


def fixture():
    K = np.array([[600.,0.,640.],[0.,600.,480.],[0.,0.,1.]])
    xyz = np.array([1.1,.11,1.3])
    R = cv2.Rodrigues(np.array([.3,.25,.1]))[0]
    t = np.array([0.,.1,4.])
    q = np.vstack([project(cuboid(*xyz), R, t, K), [641.25,481.75]])
    return K, xyz, q, R, t


def observation(q, ids):
    """Mathematical line supports, not evidence of real physical ownership."""
    lines = {}; corners = []
    for k in ids:
        incident = [e for e, pair in enumerate(EDGES) if k in pair][:2]
        corners.append(dict(id=k, xy=np.asarray(q[k]).tolist(), edges=incident, radius_px=.5))
        for e in incident:
            if e in lines:
                continue
            a, b = EDGES[e]; pa, pb = np.asarray(q[a]), np.asarray(q[b])
            tangent = (pb-pa) / np.linalg.norm(pb-pa)
            normal = np.array([-tangent[1], tangent[0]])
            support = np.array([(1-u)*pa + u*pb for u in (.25,.5,.75)])
            lines[e] = dict(edge=e, normal=normal.tolist(), offset=float(normal@pa),
                support_points=support.tolist(), query_radii_px=[1.,1.,1.],
                queries=[e*7+j for j in range(3)], support_length_px=float(np.linalg.norm(pb-pa)))
    return dict(corners=corners, lines=[lines[e] for e in sorted(lines)], queries=[])


class AuditedBank(P.PoseBank):
    def __init__(self, *a, **kw):
        self.trace = []; self.calls = []; self.blocked = None
        super().__init__(*a, **kw)

    def _generic(self, ids, dim, phase):
        self.trace.append(dict(kind='generic', phase=phase, ids=list(ids), dimension=dim))
        return super()._generic(ids, dim, phase)

    def _lm(self, candidate, ids):
        self.trace.append(dict(kind='LM', ids=list(ids), dimension=candidate.dim))
        return super()._lm(candidate, ids)

    def solve(self, excluded=(), hidden=(), robust=True):
        start = len(self.trace); blocked = set(excluded) | set(hidden)
        result = super().solve(excluded=excluded, hidden=hidden, robust=robust)
        for event in self.trace[start:]:
            # Building a shared bank enumerates all coordinates once. A mask's
            # selected generators and actual refits, not unused bank entries,
            # are required to exclude its blocked coordinates.
            if event['kind'] == 'LM' or event.get('phase') != 'subset':
                assert set(event['ids']).isdisjoint(blocked), event
        for key in ('used','fit_input_ids','final_inliers','generator_ids'):
            assert set(result.get(key, [])).isdisjoint(blocked), (key, result)
        for candidate in result.get('all_candidate_solutions', []):
            assert set(candidate['generator_ids']).isdisjoint(blocked)
            assert set(candidate['actual_fit_input_ids']).isdisjoint(blocked)
        assert result['prior_used'] is False and result['initial_pose_used'] is False
        self.calls.append(dict(excluded=list(excluded), hidden=list(hidden), result=copy.deepcopy(result)))
        return result


@contextmanager
def no_truth_or_models():
    saved = [(builtins,'open',builtins.open),(io,'open',io.open),(Path,'open',Path.open)]
    attempts = []
    forbidden = ('TARGETS','GROUND_TRUTH','STATIC_VISIBILITY','READY_SOURCE','FEATURES.NPY',
                 'PREDICTIONS.JSON','INPUTS.JSON','OBSERVATIONS.JSON','BEST.PT','LAST.PT','IMAGE_ROLE.PT')
    def guard(fn):
        def call(path, *a, **kw):
            if isinstance(path,(str,Path)) and any(s in str(path).upper() for s in forbidden):
                attempts.append(str(path)); raise AssertionError('forbidden cache/model/truth read')
            return fn(path,*a,**kw)
        return call
    for obj, name, fn in saved:
        setattr(obj,name,guard(fn))
    try:
        yield attempts
    finally:
        for obj,name,fn in saved:
            setattr(obj,name,fn)


def same_numeric_solution(a, b):
    for key in ('available','state','used','fit_input_ids','final_inliers','selected_hypothesis'):
        assert a.get(key) == b.get(key), key
    for key in ('projected','R_cf','R_physical','centroid','cf_extents','residuals_used_px'):
        if a.get(key) is not None:
            assert np.allclose(a[key], b[key], rtol=0, atol=PIXEL_ATOL), key
    for key in ('truncated_sse_px2','sse_px2'):
        if key in a:
            assert abs(a[key]-b[key]) <= PIXEL_ATOL**2, key
    assert a['prior_used'] is b['prior_used'] is False


def run():
    cv2.setNumThreads(1)
    checks = []; actual = Counter()
    names = ('solvePnPGeneric','solvePnPRefineLM','solvePnP','projectPoints')
    saved = {name:getattr(cv2,name) for name in names}
    for name, fn in saved.items():
        def counted(*a, _name=name, _fn=fn, **kw):
            actual[_name] += 1
            return _fn(*a,**kw)
        setattr(cv2,name,counted)
    def check(name, fn):
        before = actual.copy()
        try:
            with no_truth_or_models() as reads:
                detail = fn() or {}
                assert not reads
            checks.append(dict(name=name, passed=True, details=detail,
                actual_calls={k:actual[k]-before[k] for k in names}))
        except Exception as error:
            checks.append(dict(name=name, passed=False, error=type(error).__name__+': '+str(error),
                traceback=traceback.format_exc(), actual_calls={k:actual[k]-before[k] for k in names}))
    try:
        def signatures():
            assert 'initial_pose' not in inspect.signature(P.PoseBank.solve).parameters
            assert 'initial_pose' not in inspect.signature(P.refine).parameters
            assert 'initial_pose' not in inspect.signature(S.select_corners).parameters
            assert not P.POLICY['initial_pose_prior'] and not P.POLICY['initial_dimension_choice_input']
            K,xyz,q,_,_ = fixture()
            try:
                P.PoseBank(q,K,xyz,image_size=SIZE).solve(initial_pose={})
            except TypeError:
                pass
            else:
                raise AssertionError('initial pose unexpectedly accepted')
            assert 'torch' not in sys.modules
            return dict(initial_pose_api_rejected=True, no_Torch_import=True,
                        conditional_H_and_Base_feature_dependence_still_disclosed=True)
        check('no_initial_pose_projection_or_dimension_prior_API', signatures)

        def excluded_invariance():
            K,xyz,q,_,_ = fixture(); H=[6]; heldout=[7]; answers=[]; cases=[]
            variants = [('original',q[[6,7]]), ('large_inframe_wrong',[[20.,30.],[1230.,910.]]),
                        ('sentinel_minus_one',[[-1.,-1.],[-1.,-1.]]),
                        ('nonfinite_nan',[[np.nan,np.nan],[np.nan,np.nan]])]
            for label, values in variants:
                native=q.copy();native[[6,7]]=values
                bank=AuditedBank(native,K,xyz,image_size=SIZE)
                answer=bank.solve(excluded=heldout,hidden=H)
                assert answer['available'], answer['state']
                assert answer['used']==list(range(6)) and answer['geometry']['jacobian']['numerical_rank']==6
                assert np.max(np.abs(np.asarray(answer['projected'])-q[:8]))<=PIXEL_ATOL
                if answers:
                    same_numeric_solution(answers[0],answer)
                answers.append(answer)
                cases.append(dict(label=label,state=answer['state'],eligible=answer['eligible'],
                    used=answer['used'], ledger=dict(bank.ledger),excluded_coordinates_are_not_scored=True))
            return dict(fixed_H=H,fixed_heldout=heldout,cases=cases,
                        pose_projection_and_score_invariant=True,
                        global_unused_bank_generation_counts_may_change=True)
        check('excluded_H_and_k_large_wrong_sentinel_NaN_do_not_change_numeric_solution', excluded_invariance)

        def cache_reuse():
            K,xyz,q,_,_=fixture();bank=AuditedBank(q,K,xyz,image_size=SIZE)
            answers=[bank.solve(excluded=[k],hidden=[7]) for k in (0,1,2,3,4)]
            assert bank.ledger['subsets_considered']==2*70,bank.ledger
            assert bank.ledger['subset_generic_calls']<=2*70
            numeric_before=dict(bank.ledger)
            repeated=bank.solve(excluded=[0],hidden=[7])
            same_numeric_solution(answers[0],repeated)
            assert bank.ledger['subsets_considered']==numeric_before['subsets_considered']
            assert bank.ledger['generic_calls']==numeric_before['generic_calls']
            assert bank.ledger['generic_cache_hits']>0
            return dict(heldouts=[0,1,2,3,4],subsets_per_dimension=70,ledger=dict(bank.ledger),
                        repeated_solve_refines_again_but_does_not_regenerate_identical_numeric_problems=True)
        check('same_N3_bank_reuses_finite_subsets_and_exact_refit_problems_across_heldouts', cache_reuse)

        def four_five():
            K,xyz,q,R,t=fixture(); cases=[]
            provenance=dict(source='synthetic fixture declared canonical dimension',independent_of_initial_pose=True)
            for ids in ((0,1,2,3),(0,1,2,4),(0,1,2,4,5)):
                for robust in (False,True):
                    sparse=np.full((9,2),np.nan);sparse[list(ids)]=q[list(ids)];sparse[8]=q[8]
                    bank=AuditedBank(sparse,K,xyz,image_size=SIZE,
                        known_dimension_index=0,known_dimension_provenance=provenance)
                    answer=bank.solve(robust=robust)
                    assert answer['state'] in ('NEW_POSE','AMBIGUOUS_PNP'),answer['state']
                    assert answer['geometry']['jacobian']['numerical_rank']==6
                    assert answer['all_candidate_solutions'] and len(answer['fit_input_ids'])>=4
                    candidates=answer['all_candidate_solutions']
                    assert min(np.max(np.abs(np.asarray(c['projected'])[list(ids)]-q[list(ids)]))
                               for c in candidates if c['dimension_index']==0)<=PIXEL_ATOL
                    if answer['available']:
                        assert np.max(np.abs(np.asarray(answer['projected'])[list(ids)]-q[list(ids)]))<=PIXEL_ATOL
                    assert any(c['generator'].startswith('IPPE_PLANE' if ids==(0,1,2,3) else 'SQPNP')
                               for c in candidates)
                    cases.append(dict(ids=list(ids),robust=robust,state=answer['state'],
                        rank=answer['geometry']['jacobian']['numerical_rank'],
                        returned_candidates=len(candidates),known_dimension_fixture_only=True))
            # Direct standard planar4/5 controls exercise both real IPPE branches;
            # a five-point plane is not manufactured from the cuboid's four-point face.
            plane=np.array([[-.65,-.55,0.],[.65,-.55,0.],[.65,.55,0.],[-.65,.55,0.],[-.4,.2,0.]])
            planar=[]
            for n in (4,5):
                measured=project(plane[:n],R,t,K)
                ret=cv2.solvePnPGeneric(plane[:n],measured,K,None,flags=cv2.SOLVEPNP_IPPE)
                assert ret[0] and len(ret[1])==len(ret[2])==2
                errors=[float(np.max(np.abs(project(plane[:n],cv2.Rodrigues(rv)[0],tv,K)-measured)))
                        for rv,tv in zip(ret[1],ret[2])]
                assert min(errors)<=PIXEL_ATOL
                planar.append(dict(points=n,returned_solutions=2,projection_errors_px=errors))
            return dict(actual_registry_cases=cases,actual_standard_planar_controls=planar,
                        no_six_point_count_substitution=True,rank6_not_global_uniqueness=True)
        check('real_four_five_nonplane_plane_generators_all_IPPE_branches_and_rank',four_five)

        def insufficient_and_degenerate():
            K,xyz,q,_,_=fixture();cases=[]
            for label,ids,coords in (('three_points',[0,1,4],q[[0,1,4]]),
                                    ('four_collinear',[0,1,2,4],[[610.,430.],[620.,440.],[630.,450.],[640.,460.]])):
                sparse=np.full((9,2),np.nan);sparse[ids]=coords;sparse[8]=q[8]
                bank=AuditedBank(sparse,K,xyz,image_size=SIZE);answer=bank.solve()
                assert not answer['available'] and not answer['new_pose_estimated']
                assert answer['state']==('INSUFFICIENT_OBSERVATIONS' if label=='three_points' else 'DEGENERATE_OBSERVATIONS')
                cases.append(dict(label=label,state=answer['state'],actual_calls=answer['operation_counts']))
            return dict(cases=cases,no_basic_output_mislabeled_new_pose=True)
        check('insufficient_and_degenerate_layout_are_explicit_unavailable_states', insufficient_and_degenerate)

        def wrong_and_excluded():
            K,xyz,q,_,_=fixture();cases=[]
            configs=[('one_bad',[0],[]),('two_bad',[0,4],[]),('one_good_excluded',[],[7]),
                     ('two_good_excluded',[],[6,7]),('one_bad_one_good_excluded',[0],[7]),
                     ('two_bad_two_good_excluded',[0,4],[6,7])]
            for label,bad,dropped in configs:
                native=q.copy()
                for k in bad:native[k]+=[45.,-35.]
                accurate=sorted(set(range(8))-set(bad)-set(dropped));assert len(accurate)>=4
                for robust in (True,False):
                    bank=AuditedBank(native,K,xyz,image_size=SIZE)
                    # Deliberately classify otherwise correct visible fixture
                    # correspondences as H. The mask must not veto the frame.
                    answer=bank.solve(hidden=dropped,robust=robust)
                    assert set(answer['used'])==set(range(8))-set(dropped)
                    assert set(bad)<=set(answer['used']) # no oracle removal or coordinate repair
                    if answer['available']:
                        assert answer['geometry']['jacobian']['numerical_rank']==6
                        assert len(answer['fit_input_ids'])>=4
                    if robust and not bad:
                        assert answer['available'],answer['state']
                        assert np.max(np.abs(np.asarray(answer['projected'])-q[:8]))<=PIXEL_ATOL
                    if robust and label=='one_bad':
                        assert answer['available'] and set(bad).isdisjoint(answer['final_inliers'])
                        assert np.max(np.abs(np.asarray(answer['projected'])-q[:8]))<=PIXEL_ATOL
                    cases.append(dict(label=label,robust=robust,wrong_retained=bad,
                        accurate_excluded=dropped,remaining_accurate_ids=accurate,state=answer['state'],
                        available=answer['available'],used=answer['used'],fit=answer['fit_input_ids'],
                        final_inliers=answer['final_inliers'],multiple_solutions=answer['multiple_solutions']))
            return dict(cases=cases,no_universal_recovery_assertion_for_two_bad_plus_four_good=True,
                        ordinary_solver_outcome_is_diagnostic_not_required_to_be_worse=True)
        check('actual_1_2_bad_1_2_good_exclusions_and_combined_masks_without_oracle',wrong_and_excluded)

        def projection_output():
            K,xyz,q,_,_=fixture();native=q.copy();native[[6,7]]=[[20.,30.],[1230.,910.]]
            bank=AuditedBank(native,K,xyz,image_size=SIZE);answer=bank.solve(excluded=[7],hidden=[6])
            assert answer['available']
            after=actual.copy();output=np.asarray(answer['points_final'])
            independently_projected=project(cuboid(*answer['cf_extents']),np.asarray(answer['R_cf']),np.asarray(answer['centroid']),K)
            assert np.max(np.abs(output[6]-independently_projected[6]))<=PIXEL_ATOL
            assert np.array_equal(output[7],native[7]) and np.array_equal(output[8],native[8])
            assert not answer['reprojected_points_reused_as_observations']
            assert answer['temporary_excluded']==[7] and answer['hidden']==[6]
            assert actual==after # independent projection here uses plain matrix arithmetic
            return dict(only_hidden6_reprojected=True,temporary7_kept_unchanged=True,
                        center8_preserved=True,additional_PnP_calls_after_output=0,
                        independent_projection_verification_calls=1)
        check('H_reprojected_after_new_pose_temporary_k_unchanged_no_refit_center_preserved', projection_output)

        def selector_actual():
            K,xyz,q,_,_=fixture();native=q.copy();native[0]+=[2.,-.5]
            observed=q.copy();observed[[0,1,3]]+=[.2,.1]
            obs=observation(observed,[0]);before=copy.deepcopy(obs)
            bank=AuditedBank(native,K,xyz,image_size=SIZE)
            selected,contract=S.select_corners(native,obs,[7],K,xyz,SIZE,bank=bank)
            record=contract['cornerwise_records'][0]
            assert len(bank.calls)==1 and bank.calls[0]['excluded']==[0] and bank.calls[0]['hidden']==[7]
            assert bank.calls[0]['result']['excluded']==[0,7]
            assert bank.calls[0]['result']['available'],bank.calls[0]['result']['state']
            assert contract['hybrid_boundary_corner_ids']==[0] and np.array_equal(selected[0],np.asarray(obs['corners'][0]['xy']))
            assert np.linalg.norm(selected[0]-np.asarray(record['loo_solver']['projected'])[0])>PIXEL_ATOL
            assert np.array_equal(selected[8],native[8]) and np.array_equal(selected[7],native[7])
            assert obs==before and record['numeric_validation_pose_prior_used'] is False
            assert contract['LOO_initial_prior_includes_heldout_influence'] is False
            assert contract['fully_independent_validation'] is False
            return dict(actual_boundary_not_projection=True,fit_excluded=[0,7],
                        conditional_H_and_Base_feature_dependence_not_erased=True)
        check('actual_prior_free_LOO_selects_observed_boundary_not_validation_projection',selector_actual)

        def selector_unavailable():
            K,xyz,q,_,_=fixture();native=np.full((9,2),np.nan);native[[0,1,2,4,8]]=q[[0,1,2,4,8]]
            native[0]+=[1.,0.];obs=observation(q,[0]);bank=AuditedBank(native,K,xyz,image_size=SIZE)
            selected,contract=S.select_corners(native,obs,[6,7],K,xyz,SIZE,bank=bank)
            assert bank.calls[0]['result']['state']=='INSUFFICIENT_OBSERVATIONS'
            assert np.array_equal(selected,native,equal_nan=True) and not contract['hybrid_boundary_corner_ids']
            assert contract['selection_does_not_fail_frame_for_mask_disagreement'] is True
            return dict(LOO_3_remaining_points=True,selection_abstains_only_corner=True,
                        native_four_point_final_input_still_preserved=True)
        check('insufficient_LOO_preserves_native_without_mask_or_whole_frame_veto',selector_unavailable)

        def tied_pose():
            K,xyz,q,R,t=fixture()
            class Tied(P.PoseBank):
                def _generic(self,*a,**kw):return []
                def _lm(self,*a,**kw):return None
            bank=Tied(q,K,xyz,image_size=SIZE);bank.hypotheses=[]
            rv=cv2.Rodrigues(R)[0]
            for j,shift in enumerate((-.02,.02)):
                tv=t+np.array([shift,0.,0.])
                bank.hypotheses.append(Candidate(0,(0,1,2,4),j,'CONTROLLED_PHYSICAL_POSE',rv,tv.reshape(3,1),project(cuboid(*xyz),R,tv,K)))
            before=actual.copy();answer=bank.solve(excluded=[7])
            assert not answer['available'] and answer['state']=='AMBIGUOUS_PNP'
            assert answer['unresolved_ambiguity'] and answer['multiple_solutions']
            assert all(actual[k]==before[k] for k in ('solvePnPGeneric','solvePnPRefineLM','solvePnP'))
            assert not answer['global_uniqueness_proven']
            class AmbiguousBank(AuditedBank):
                def solve(self,*a,**kw):
                    value=super().solve(*a,**kw)
                    value.update(available=False,state='AMBIGUOUS_PNP',unresolved_ambiguity=True)
                    return value
            native=q.copy();native[0]+=[1.,0.]
            selected,contract=S.select_corners(native,observation(q,[0]),[7],K,xyz,SIZE,
                                               bank=AmbiguousBank(native,K,xyz,image_size=SIZE))
            assert np.array_equal(selected,native) and contract['cornerwise_records'][0]['reason']=='HELDOUT_POSE_AMBIGUOUS'
            return dict(controlled_arbitration_fixture=True,pose_tie_not_success=True,
                        ambiguous_validation_preserves_native=True,natural_IPPE_returns_checked_separately=True)
        check('distinct_numerically_tied_pose_is_unavailable_and_cannot_choose_boundary',tied_pose)

        def square_and_bad_provenance():
            K,_,q,R,t=fixture();xyz=np.array([1.1,.15,1.1]);q[:8]=project(cuboid(*xyz),R,t,K)
            bank=AuditedBank(q,K,xyz,image_size=SIZE);answer=bank.solve()
            assert len(bank.dims)==1 and bank.ledger['subsets_considered']==70
            assert answer['available'] and np.max(np.abs(np.asarray(answer['projected'])-q[:8]))<=PIXEL_ATOL
            try:
                P.PoseBank(q,K,xyz,image_size=SIZE,known_dimension_index=0)
            except ValueError:pass
            else:raise AssertionError('unproven dimension constraint accepted without declaration')
            return dict(square_dimension_deduplicated=True,unproven_dimension_constraint_rejected=True,
                        declared_external_provenance_is_not_independently_certified=True)
        check('square_dimensions_deduplicate_and_known_dimension_needs_explicit_provenance',square_and_bad_provenance)
    finally:
        for name,fn in saved.items():setattr(cv2,name,fn)
    return dict(schema='actual_observation_only_independence_synthetic_checks_v4',complete=True,
        passed=all(c['passed'] for c in checks),checks=checks,actual_calls={k:actual[k] for k in names},
        pose_policy=copy.deepcopy(P.POLICY),selection_policy=copy.deepcopy(S.POLICY),opencv=cv2.__version__,
        fixed_numeric_projection_atol_px=PIXEL_ATOL,models_loaded=0,images_decoded=0,GPU_calls=0,
        detector_forwards=0,N3_forwards=0,ROLE_forwards=0,new_training_updates=0,new_RGB=0,
        private_GT_reads=0,rays=0,real_pose_fits=0,timing_intervals=0,no_automatic_retry=True,
        limitations=['Conditional numerical exclusion test with H and remaining observations fixed.',
            'Inherited H and Base proposal features still depend on the original fixed estimators.',
            'Synthetic ownership and known-dimension fixtures do not establish physical real-image roles.',
            'Local rank6 or lowest residual does not prove global uniqueness or real accuracy.',
            'Two retained errors with only four correct correspondences may remain ambiguous.'])


def binding(path):
    path=Path(path).resolve()
    return dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DOC/'INDEPENDENCE_CHECKS.json')
    args=parser.parse_args();out=args.output
    if out.is_symlink() or out.exists() or out.parent.is_symlink():
        raise FileExistsError('Preserve existing/symlink independence check output')
    out=out.resolve()
    if out != DOC/'INDEPENDENCE_CHECKS.json':
        raise ValueError('Only the exact new independence-check output is supported')
    started=DOC/'INDEPENDENCE_CHECKS_STARTED.json'
    if started.exists() or started.is_symlink():
        raise FileExistsError('Preserve interrupted/completed independence-check execution claim')
    code=dict(test=binding(__file__),selection=binding(S.__file__),pose=binding(P.__file__),
              borrowed_solver=binding(ROOT/'scripts/research/pallet_observation_refiner_20261009_v1/solver.py'))
    out.parent.mkdir(parents=True,exist_ok=True)
    with started.open('x') as stream:
        json.dump(dict(schema='independence_check_execution_claim_v4',status='STARTED',code=code,
                       no_automatic_retry=True,real_or_GPU_execution=False),stream,indent=2,allow_nan=False)
        stream.write('\n')
    result=run();result['code']=code
    # Finite-only JSON: fixture raw NaNs are intentionally omitted from receipts.
    with out.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(passed=result['passed'],checks=len(result['checks']),
        failed=[c['name'] for c in result['checks'] if not c['passed']],actual_calls=result['actual_calls'])))
    return 0 if result['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
