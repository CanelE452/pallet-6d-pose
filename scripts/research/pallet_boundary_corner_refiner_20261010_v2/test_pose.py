"""Synthetic geometry checks, with actual PnP/LM calls counted separately.

No RGB, detector, head, private GT or source modules are opened. These unit
checks do not measure real-image accuracy or latency.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

from ..pallet_observation_refiner_20261009_v1.solver import cuboid, project, visibility, Candidate
from .pose import PoseBank, refine, POLICY


def bind(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def fixture(rvec=(.3, .25, .1), t=(0., .1, 4.)):
    K = np.array([[600., 0., 640.], [0., 600., 480.], [0., 0., 1.]])
    xyz = np.array([1.1, .11, 1.3])
    R = cv2.Rodrigues(np.asarray(rvec, float))[0]
    t = np.asarray(t, float)
    q = np.vstack([project(cuboid(*xyz), R, t, K), [641., 481.]])
    initial = dict(available=True, R_cf=R.tolist(), R_physical=R.tolist(),
                   centroid=t.tolist(), cf_extents=xyz.tolist(), selected_hypothesis="FIXED_SYNTHETIC_INITIAL")
    return K, xyz, q, initial


def sparse(points, ids):
    q = np.full((9, 2), np.nan)
    q[list(ids)] = points[list(ids)]
    q[8] = points[8]
    return q


def run():
    checks = []
    actual = dict(solvePnPGeneric=0, solvePnPRefineLM=0)
    generic, lm = cv2.solvePnPGeneric, cv2.solvePnPRefineLM
    def counted_generic(*args, **kwargs):
        actual["solvePnPGeneric"] += 1
        return generic(*args, **kwargs)
    def counted_lm(*args, **kwargs):
        actual["solvePnPRefineLM"] += 1
        return lm(*args, **kwargs)
    cv2.solvePnPGeneric, cv2.solvePnPRefineLM = counted_generic, counted_lm
    def check(name, function):
        before = actual.copy()
        try:
            detail = function() or {}
            checks.append(dict(name=name, passed=True, details=detail,
                               actual_calls={k:actual[k]-before[k] for k in actual}))
        except Exception as exc:
            checks.append(dict(name=name, passed=False, error=type(exc).__name__+": "+str(exc),
                               actual_calls={k:actual[k]-before[k] for k in actual}))
    try:
        K, xyz, q, initial = fixture()
        for ids in ((0, 1, 2, 3), (0, 1, 2, 4), (0, 1, 2, 4, 5)):
            for robust in (False, True):
                def exact(ids=ids, robust=robust):
                    p = sparse(q, ids)
                    answer = refine(p, K, xyz, initial, [6, 7], (1280, 960), robust=robust)
                    assert answer["available"], answer["state"]
                    assert answer["state"] == "NEW_POSE" and len(answer["fit_input_ids"]) >= 4
                    assert set(answer["fit_input_ids"]) <= set(ids)
                    assert answer["cf_extents"] == initial["cf_extents"]
                    error = float(np.max(np.abs(np.asarray(answer["projected"]) - q[:8])))
                    assert error < 1e-5, error
                    assert answer["geometry"]["jacobian"]["numerical_rank"] == 6
                    return dict(ids=list(ids), robust=robust, max_projection_error_px=error,
                                exact_plane=len(ids)==4 and ids==(0,1,2,3))
                check("exact_"+str(len(ids))+"_"+str(ids)+"_"+str(robust), exact)

        for count in (1, 2):
            def outliers(count=count):
                p = q.copy();bad=[0, 6][:count]
                for i, offset in zip(bad, ([30., -35.], [-40., 30.])):
                    p[i] += offset
                answer = refine(p, K, xyz, initial, [7], (1280, 960))
                assert answer["available"], answer["state"]
                assert not set(bad) & set(answer["final_inliers"])
                assert set(answer["final_inliers"]) == set(range(8)) - set(bad) - {7}
                assert np.max(np.abs(np.asarray(answer["projected"])-q[:8])) < 1e-5
                return dict(outlier_ids=bad, final_inliers=answer["final_inliers"])
            check("exclude_"+str(count)+"_outliers_from_consensus", outliers)

        def incorrect_mask():
            trueH = np.flatnonzero(visibility(cuboid(*xyz), np.asarray(initial["R_cf"]), np.asarray(initial["centroid"]))[0]).tolist()
            visible = [i for i in range(8) if i not in trueH]
            answers = []
            for count in (1, 2):
                H = sorted(set(trueH + visible[:count]))
                answer = refine(q, K, xyz, initial, H, (1280, 960))
                assert answer["available"], answer["state"]
                assert set(H).isdisjoint(answer["fit_input_ids"])
                assert answer["points_final"][8] == q[8].tolist()
                assert np.max(np.abs(np.asarray(answer["points_final"])[H]-np.asarray(answer["projected"])[H])) < 1e-10
                answers.append(dict(falsely_excluded_visible=visible[:count], used=answer["used"], inliers=answer["final_inliers"]))
            return dict(actual_synthetic_hidden=trueH, remaining_pose_succeeds=answers)
        check("one_two_wrong_H_exclusions_do_not_abort_frame", incorrect_mask)

        def altered_hidden_coordinate():
            p = q.copy();p[6] += [200., -170.];p[7] += [-200., 170.]
            clean = refine(q, K, xyz, initial, [6, 7], (1280, 960))
            dirty = refine(p, K, xyz, initial, [6, 7], (1280, 960))
            assert clean["available"] and dirty["available"]
            assert np.max(np.abs(np.asarray(clean["projected"])-np.asarray(dirty["projected"]))) < 1e-8
            assert dirty["reprojected_points_reused_as_observations"] is False
            assert set(dirty["fit_input_ids"]).isdisjoint([6, 7])
            assert np.allclose(np.asarray(dirty["points_final"])[[6,7]], np.asarray(dirty["projected"])[[6,7]])
            return dict(hidden_input_corruption_does_not_change_pose=True,
                        hidden_replaced_after_fit=True, projection_reused_as_observation=False)
        check("H_initial_coordinates_never_reinserted_into_fit", altered_hidden_coordinate)

        def mask_changes():
            theta0, theta1 = np.deg2rad(5.), np.deg2rad(6.5)
            k, x, p0, prior = fixture((0., theta0, 0.), (0., .5, 4.))
            R = cv2.Rodrigues(np.array([0., theta1, 0.]))[0]
            p1 = np.vstack([project(cuboid(*x), R, np.array([0., .5, 4.]), k), p0[8]])
            H0 = np.flatnonzero(visibility(cuboid(*x), np.asarray(prior["R_cf"]), np.asarray(prior["centroid"]))[0]).tolist()
            H1 = np.flatnonzero(visibility(cuboid(*x), R, np.array([0., .5, 4.]))[0]).tolist()
            assert H0 != H1, (H0, H1)
            answer = refine(p1, k, x, prior, H0, (1280, 960))
            assert answer["available"], answer["state"]
            assert np.max(np.abs(np.asarray(answer["projected"])-p1[:8])) < 1e-5
            assert set(H0).isdisjoint(answer["fit_input_ids"])
            return dict(initial_H=H0, final_H=H1, H_change_not_failure=True,
                        all_eight_prior_rms_px=answer["prior_projection_rms_px"])
        check("changed_H_does_not_reject_valid_pose", mask_changes)

        def cache():
            bank = PoseBank(q, K, xyz, image_size=(1280, 960))
            a = refine(q, K, xyz, initial, [], (1280, 960), bank=bank)
            generated = bank.ledger["subset_generic_calls"]
            b = refine(q, K, xyz, initial, [6, 7], (1280, 960), bank=bank)
            assert a["available"] and b["available"]
            assert generated == 140 and bank.ledger["subset_generic_calls"] == generated
            assert b["operation_counts"]["subset_generic_calls"] == 0
            assert len(a["per_dimension_best"]) == 2
            rejected = False
            try:refine(q, K, xyz, initial, [], (640, 480), bank=bank)
            except ValueError:rejected = True
            assert rejected
            return dict(all_dimension_subset_calls=generated, second_mask_new_subset_calls=0,
                        inconsistent_bank_image_size_rejected=True)
        check("same_coordinate_bank_reuses_all_four_ID_subsets_across_masks", cache)

        def dimension_cliff():
            k, x, p, prior = fixture((.05, .1, .02), (0., .1, 12.))
            R = np.asarray(prior["R_cf"])
            p[:8] = project(cuboid(*x[[2,1,0]]), R, np.asarray(prior["centroid"]), k)
            answer = refine(p, k, x, prior, [], (1280, 960))
            assert answer["available"], answer["state"]
            assert answer["cf_extents"] == prior["cf_extents"]
            opposing = next(d for d in answer["per_dimension_best"] if not d["same_initial_dimension"])
            assert opposing["truncated_sse_px2"] < 1e-6
            assert opposing["accepted_by_prior"] is False
            assert opposing["truncated_sse_px2"] < answer["truncated_sse_px2"]
            return dict(opposing_dimension_objective_px2=opposing["truncated_sse_px2"],
                        selected_objective_px2=answer["truncated_sse_px2"],
                        initial_dimension_explicitly_preserved=True,
                        initial_error_can_be_inherited=True)
        check("competing_dimension_lower_residual_is_diagnostic_not_branch_jump", dimension_cliff)

        def prior_jump():
            k, x, p, prior = fixture()
            moved = project(cuboid(*x), np.asarray(prior["R_cf"]), np.asarray(prior["centroid"])+[.2,0.,0.], k)
            p[:8] = moved
            answer = refine(sparse(p, [0,1,2,4]), k, x, prior, [], (1280, 960))
            assert not answer["available"] and answer["state"] == "BRANCH_JUMP_REJECTED", answer["state"]
            assert not answer["fallback_used"] and answer["projected"] is None
            return dict(state=answer["state"], fallback_must_be_assembled_by_driver=True)
        check("all_eight_projection_prior_rejects_large_pose_jump", prior_jump)

        def missing():
            for n in (0,1,2,3):
                p=sparse(q,list(range(n)))
                answer=refine(p,K,xyz,initial,[],(1280,960))
                assert not answer["available"] and answer["state"]=="INSUFFICIENT_OBSERVATIONS"
                assert answer["operation_counts"]["generic_calls"]==0
                assert np.array_equal(np.asarray(answer["points_final"]),p,equal_nan=True)
            unavailable=refine(q,K,xyz,dict(available=False),[],(1280,960))
            assert unavailable["state"]=="INITIAL_POSE_UNAVAILABLE" and not unavailable["available"]
            return dict(under_four_never_generates_pose=True, no_initial_observations_filled=True)
        check("observation_shortage_and_missing_prior_have_distinct_states", missing)

        def tied_distinct_candidates():
            # A controlled bank of two genuine, positive-depth pose candidates
            # tests the tie arbiter only. It does not claim a solver naturally
            # produces this pair or prove global ambiguity detection.
            class FixedCandidates(PoseBank):
                def _generic(self,*args,**kwargs):return []
                def _lm(self,*args,**kwargs):return None
            p=sparse(q,[0,1,2,4]);bank=FixedCandidates(p,K,xyz,image_size=(1280,960))
            bank.hypotheses=[]
            rv=cv2.Rodrigues(np.asarray(initial["R_cf"]))[0]
            for j,shift in enumerate((-.02,.02)):
                tv=np.asarray(initial["centroid"])+[shift,0.,0.]
                projection=project(cuboid(*xyz),np.asarray(initial["R_cf"]),tv,K)
                bank.hypotheses.append(Candidate(0,(0,1,2,4),j,"CONTROLLED_VALID_POSE",rv,tv.reshape(3,1),projection))
            answer=bank.solve(initial,hidden=[],robust=True)
            assert not answer["available"] and answer["state"]=="AMBIGUOUS_PNP",answer["state"]
            assert answer["unresolved_ambiguity"] and answer["multiple_solutions"]
            assert not answer["global_uniqueness_proven"]
            return dict(decision_policy_only=True, actual_new_PnP_calls=0,
                        distinct_numeric_score_tie_not_counted_as_new_pose=True)
        check("equal_score_distinct_pose_candidates_report_ambiguity", tied_distinct_candidates)
    finally:
        cv2.solvePnPGeneric, cv2.solvePnPRefineLM = generic, lm
    return dict(complete=True,passed=all(r["passed"] for r in checks),checks=checks,
                actual_calls=actual,policy=POLICY,
                limits=["Synthetic mathematical unit checks, not real accuracy or latency.",
                        "Numerical tie ambiguity handling does not certify global uniqueness.",
                        "The explicit initial-pose prior can inherit initial pose errors."],
                new_RGB=0,detector_forwards=0,head_forwards=0,optimizer_updates=0,private_GT_reads=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();started=time.monotonic();result=run()
    result.update(wall_seconds=time.monotonic()-started,code=[bind(__file__),bind(Path(__file__).with_name('pose.py'))],
                  original_solver=bind(Path(__file__).parents[1]/'pallet_observation_refiner_20261009_v1/solver.py'))
    if args.output:
        if args.output.exists():raise FileExistsError("Preserve prior unit-check receipt: "+str(args.output))
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(passed=result['passed'],checks=[dict(name=r['name'],passed=r['passed'],error=r.get('error')) for r in result['checks']],actual_calls=result['actual_calls']),ensure_ascii=False))
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
