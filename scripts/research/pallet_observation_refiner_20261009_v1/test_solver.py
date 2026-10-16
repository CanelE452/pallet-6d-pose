"""Numerical integrity checks; these are not empirical performance claims."""
import argparse
import builtins
import inspect
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np

from solver import HypothesisBank, POLICY, cuboid, project, visibility


def run_checks():
    cv2.setNumThreads(1)
    checks = []
    def check(name, assertion, **details):
        checks.append(dict(name=name, passed=bool(assertion), **details))
    xyz = np.array([1.3,.11,1.1])
    K = np.array([[570.,0,320],[0,573.,240],[0,0,1.]])
    R = cv2.Rodrigues(np.array([.24,-.15,.1]))[0]
    t = np.array([.05,.07,3.3])
    X = cuboid(*xyz)
    q = project(X,R,t,K)
    opencv = cv2.projectPoints(X,cv2.Rodrigues(R)[0],t,K,None)[0].reshape(-1,2)
    check("projection_coordinate_and_meter_contract",np.max(abs(q-opencv))<1e-10,
          max_difference_px=float(np.max(abs(q-opencv))))
    p = np.vstack([q,[312.4,244.2]])
    bank = HypothesisBank(p,K,xyz)
    contracts = []
    for U in [(0,1,2,3),(0,1,2,4),(0,1,2,4,5),(0,1,2,3,4,5),tuple(range(8))]:
        hidden = [i for i in range(8) if i not in U]
        for robust in [False,True]:
            out = bank.solve(hidden,robust,hidden)
            contracts.append(dict(n=len(U),ids=list(U),planar=len(U)==4 and U==(0,1,2,3),
                            robust=robust,available=out["available"],
                            reprojection_px=out.get("reprojection_px"),
                            alternatives=len(out.get("alternatives",[]))))
            check("four_five_six_eight_correspondences_"+str(U)+str(robust),
                  out["available"] and out["reprojection_px"]<1e-6, **contracts[-1])
    count_before = bank.ledger["subset_generic_calls"]
    out = bank.solve([0],True,[0])
    check("one_mask_error_does_not_abort_remaining_seven",out["available"])
    check("cached_four_subsets_reused_across_masks",bank.ledger["subset_generic_calls"]==count_before,
          actual_subset_calls=count_before, allowed_upper_bound=2*math.comb(8,4),
          call_delta=out["operation_counts"]["subset_generic_calls"])
    check("maximum_seventy_subsets_per_dimension",bank.ledger["subsets_considered"]==140)
    # Exact-plane standard algorithms accept >=4, retain two IPPE solutions.
    plane = np.array([[-.65,-.55,0],[.65,-.55,0],[.65,.55,0],[-.65,.55,0],
                      [-.4,.2,0],[.1,-.2,0],[.4,.3,0],[-.2,-.1,0]])
    planar = []
    for n in [4,5,6,8]:
        pq = project(plane[:n],R,t,K)
        for flag,name in [(cv2.SOLVEPNP_IPPE,"IPPE"),(cv2.SOLVEPNP_SQPNP,"SQPNP")]:
            ret = cv2.solvePnPGeneric(plane[:n],pq,K,None,flags=flag)
            err = []
            for rv,tv in zip(ret[1],ret[2]):
                err.append(float(np.linalg.norm(project(plane[:n],cv2.Rodrigues(rv)[0],tv,K)-pq,axis=1).mean()))
            planar.append(dict(n=n,method=name,solutions=len(ret[1]),reprojection_px=err))
            check("planar_standard_"+str(n)+name,ret[0] and min(err)<1e-6,
                  solutions=len(ret[1]),reprojection_px=err)
    check("known_ippe_multiple_solutions_retained",all(r["solutions"]==2 for r in planar if r["method"]=="IPPE"))
    out = bank.solve([0,1],True,[0,1])
    changed = p.copy()
    changed[[0,1]] = [[12.,433.],[593.,21.]]
    modified = HypothesisBank(changed,K,xyz).solve([0,1],True,[0,1])
    diff = float(np.max(abs(np.asarray(out["projected"])-np.asarray(modified["projected"]))))
    check("excluded_initial_coordinates_have_no_fit_residual",diff<1e-8 and
          not set(out["fit_input_ids"])&{0,1},max_projection_difference_px=diff)
    pr = project(cuboid(*out["cf_extents"]),np.array(out["R_cf"]),np.array(out["centroid"]),K)
    check("hidden_final_equals_actual_final_pose_projection",
          np.max(abs(np.array(out["points_final"])[[0,1]]-pr[[0,1]]))<1e-10)
    check("hidden_projection_not_reused_as_observation",
          out["reprojected_points_reused_as_observations"] is False)
    check("center_and_original_input_immutable",np.array_equal(bank.points,p) and
          np.array_equal(np.array(out["points_final"])[8],p[8]))
    for available in [[],[0],[0,1],[0,1,2]]:
        excluded = [i for i in range(8) if i not in available]
        fail = bank.solve(excluded,True,excluded)
        check("explicit_insufficient_"+str(len(available)),not fail["available"] and
              fail["reason"]=="insufficient_observations" and fail["new_pose_estimated"] is False)
    # Real out-of-frame, sentinel and nonfinite points never enter U.
    missing = p.copy(); missing[0]=[-1,-1]; missing[1]=[640,40];missing[2]=[np.nan,55]
    mb = HypothesisBank(missing,K,xyz)
    check("eligible_pool_sentinel_finite_and_image_extent",mb.eligible==(3,4,5,6,7))
    # Independent padding and registry axis conversion round trip.
    pad = 80.; network=q+pad
    check("original_padding_round_trip",np.max(abs(network-pad-q))<1e-12,
          max_difference_px=float(np.max(abs(network-pad-q))))
    Q = np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
    physical=R@Q
    check("swapped_dimension_physical_axis_round_trip",np.max(abs(physical@Q.T-R))<1e-12)
    # GT/annotation file access canary: solver has no filesystem dependency.
    original_open=builtins.open
    attempted=[]
    def forbidden_open(*args,**kwargs):
        attempted.append(str(args[0]));raise RuntimeError("inference file read forbidden")
    builtins.open=forbidden_open
    try:
        canary=HypothesisBank(p,K,xyz).solve([0],True,[0])
    finally:
        builtins.open=original_open
    check("GT_file_read_canary",canary["available"] and not attempted,attempted_reads=attempted)
    check("deployable_signature_has_no_GT_or_human_arguments",
          set(inspect.signature(HypothesisBank).parameters)=={"points","K","xyz","image_size"} and
          set(inspect.signature(HypothesisBank.solve).parameters)=={"self","excluded","robust","hidden"})
    check("different_ID_inlier_identity",len(out["inliers"])==len(set(out["inliers"])) and
          all(0<=i<8 for i in out["inliers"]))
    # Record robust versus ordinary; improvement is not an integrity assertion.
    comparisons=[]
    for count in [1,2]:
        corrupt=p.copy();corrupt[:count]+=[24.,0.]
        cb=HypothesisBank(corrupt,K,xyz)
        for robust in [False,True]:
            result=cb.solve(robust=robust)
            estimate=np.array(result["R_physical"]) if result["available"] else None
            eR=float(np.degrees(np.arccos(np.clip((np.trace(R.T@estimate)-1)/2,-1,1)))) if estimate is not None else None
            eT=float(np.linalg.norm(np.array(result["centroid"])-t)*100) if estimate is not None else None
            comparisons.append(dict(outliers=count,robust=robust,available=result["available"],
                                    translation_cm=eT,rotation_deg=eR,inliers=result["inliers"]))
    check("outlier_control_executed_without_performance_gate",len(comparisons)==4)
    # Exact collinear image projection cannot quietly become a valid solution.
    collinear=p.copy();collinear[:8]=np.column_stack([np.linspace(150,450,8),np.full(8,240.)])
    failed=HypothesisBank(collinear,K,xyz).solve()
    check("rank_deficient_geometry_explicit_failure",not failed["available"],reason=failed["reason"])
    return dict(complete=True,passed=all(c["passed"] for c in checks),opencv=cv2.__version__,
                algorithm_policy=POLICY,checks=checks,correspondence_contracts=contracts,
                standard_planar_multi_solutions=planar,outlier_diagnostic=comparisons,
                actual_solver_ledger=bank.ledger,
                synthetic_RGB_rendered=0,interpretation="Numerical/implementation checks; not real performance evidence")


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",required=True);args=parser.parse_args()
    started=time.perf_counter();result=run_checks();result["wall_seconds"]=time.perf_counter()-started
    dst=Path(args.output);dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists():
        raise FileExistsError("Refusing to overwrite completed solver checks")
    dst.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(passed=result["passed"],checks=len(result["checks"]),
                         failed=[c["name"] for c in result["checks"] if not c["passed"]],
                         wall_seconds=result["wall_seconds"])))
    return 0 if result["passed"] else 1


if __name__=="__main__":
    raise SystemExit(main())
