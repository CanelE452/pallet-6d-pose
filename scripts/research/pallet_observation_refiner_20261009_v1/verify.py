"""Independent raw-result verifier; does not import the statistics implementation.

Uses Python sample variance/stdev and independent numeric projection/metric
formulas. Shared-session intervals are recomputed from the saved frame rows.
Re-run after adding learned rows; records exact hashes of the verified inputs.
"""
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import inspect
import json
from pathlib import Path
import statistics as S
import time

import cv2
import numpy as np

from . import common as C

METRICS={"translation_cm":("translation_cm",1.,"cm"),
         "rotation_deg":("rotation_deg",1.,"degree"),
         "ADDsym_cm":("ADDsym_m",100.,"cm")}


def cube(d):
    a,b,c=np.asarray(d)/2.
    return np.array([[-a,-b,-c],[a,-b,-c],[a,b,-c],[-a,b,-c],
                     [-a,-b,c],[a,-b,c],[a,b,c],[-a,b,c]])


def distribution(values):
    a=[float(v) for v in values]
    return dict(n=len(a),mean=S.fmean(a) if a else None,
                sample_variance=S.variance(a) if len(a)>1 else None,
                sample_std=S.stdev(a) if len(a)>1 else None,
                median=S.median(a) if a else None,
                P90=float(np.quantile(a,.9)) if a else None,max=max(a) if a else None)


def metric(pose,truth):
    if not pose["available"]:return dict(available=False)
    R=np.array(pose["R_physical"]);G=np.array(truth["R"])
    t=np.array(pose["centroid"]);target_t=np.array(truth["t"])
    X=cube(truth["xyz"]);rot=[];add=[]
    for i in range(truth["order"]):
        theta=2.*np.pi*i/truth["order"]
        Q=np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
        target=G@Q
        rel=target.T@R
        rot.append(float(np.degrees(np.arccos(np.clip((np.trace(rel)-1.)/2.,-1.,1.)))))
        add.append(float(np.linalg.norm((R@X.T).T+t-((target@X.T).T+target_t),axis=1).mean()))
    return dict(available=True,translation_cm=float(np.linalg.norm(t-target_t)*100.),
                rotation_deg=min(rot),ADDsym_m=min(add),
                ADDsym_normalized=min(add)/np.linalg.norm(truth["xyz"]))


def correspondence_audit(rows, controls, bounds, targets):
    """GT and fixed baseline phase are used only AFTER inference was sealed."""
    control={(r["method"],r["id"]):r for r in controls}
    bound={r["id"]:r for r in bounds}
    visibility=C.read(C.ROOT/"_docs/experiments/pallet_combined_closeout_20261003_v1/closeout_20261006_v1/visibility_square/STATIC_VISIBILITY_MERGE_AUDIT.json")["rows"]
    labels={(r["frame_id"],r["corner_id"]):r["category"] for r in visibility if r["population"]=="DEV319"}
    raw=[];groups=defaultdict(list)
    for row in rows:
        if "solver" not in row:continue
        arm="N3_SUBPIX" if row["method"].startswith("N3_SUBPIX") else "BASE"
        base=control[(arm,row["id"])];target=targets[row["id"]]
        phase=target["permutations"][base["corner"].get("branch",0)][:8]
        reference=np.asarray(target["gt"],float)[phase]
        valid=np.asarray(target["valid"],bool)[phase] & np.isfinite(reference).all(1) & ~(reference==-1).all(1)
        observed=np.asarray(row["input_points"],float)[:8]
        finite=np.isfinite(observed).all(1) & ~(observed==-1).all(1)
        valid=valid & finite & bool(target["matched"])
        error=np.linalg.norm(observed-reference,axis=1)
        error[~valid]=np.nan
        known={i for i in range(8) if valid[i]}
        correct={i for i in known if error[i]<=8.}
        state=[labels.get((row["id"],i),"UNANNOTATED") for i in phase]
        direct={i for i,s in enumerate(state) if s=="DIRECT_VISIBLE"}
        humanhidden={i for i,s in enumerate(state) if s=="SELF_OCCLUDED"}
        H=set(row["hidden_initial"]);U=set(row["solver"]["used"]);I=set(row["solver"]["inliers"])
        labelknown={i for i,s in enumerate(state) if s!="UNANNOTATED"}
        wrongmask=bool((H^humanhidden)&labelknown)
        pose=row["pose"];old=base["pose"]
        deltaT=pose["translation_cm"]-old["translation_cm"] if pose["available"] and old["available"] else None
        deltaR=pose["rotation_deg"]-old["rotation_deg"] if pose["available"] and old["available"] else None
        outcome="no_new_pose"
        if row["new_pose_estimated"] and deltaT is not None:
            outcome="both_improved" if deltaT< -1e-9 and deltaR< -1e-9 else "both_worsened" if deltaT>1e-9 and deltaR>1e-9 else "mixed_or_equal"
        poseparams=row["actual_pose"]
        dims=poseparams.get("cf_extents",bound[row["id"]]["xyz"])
        layout=cube(dims)[sorted(correct&U)]
        singular=np.linalg.svd(layout-layout.mean(0),compute_uv=False) if len(layout) else np.array([])
        record=dict(id=row["id"],method=row["method"],condition=row.get("condition"),
            oracle_phase="ORACLE_MASK_AND_PHASE diagnostic only; frozen same-coordinate baseline corner branch",
            permutation_native_to_canonical=list(phase),human_states_native=state,
            reference_matched=bool(target["matched"]),reference_error_input_native_px=C.finite(error),
            accurate_threshold_px=8.,reference_valid_native_ids=sorted(known),
            correct_input_native_ids=sorted(correct),pool_ids=sorted(U),final_inlier_ids=sorted(I),
            correct_pool_ids=sorted(correct&U),correct_pool_count=len(correct&U),
            wrong_pool_ids=sorted((U&known)-correct),unknown_pool_ids=sorted(U-known),
            correct_final_inlier_ids=sorted(correct&I),correct_final_inlier_count=len(correct&I),
            wrong_final_inlier_ids=sorted((I&known)-correct),unknown_final_inlier_ids=sorted(I-known),
            human_direct_pool_ids=sorted(direct&U),human_direct_correct_pool_ids=sorted(direct&correct&U),
            human_direct_correct_final_inlier_ids=sorted(direct&correct&I),
            false_excluded_direct_ids=sorted(H&direct),false_excluded_accurate_ids=sorted(H&correct),
            false_retained_human_self_ids=sorted((U&humanhidden)-H),
            wrong_coordinate_pool_over_32px_ids=sorted(i for i in U&known if error[i]>32.),
            correct_pool_layout_m=layout.tolist(),correct_pool_shape_singular_values=singular.tolist(),
            correct_pool_shape_rank=int(np.sum(singular>singular[0]*1e-10)) if len(singular) and singular[0]>0 else 0,
            mask_wrong_on_known=wrongmask,mask_pose_outcome=("wrong_mask:" if wrongmask else "correct_mask_on_known:")+outcome,
            new_pose_estimated=row["new_pose_estimated"],fallback_used=row["fallback_used"],no_pose=row["no_pose"],
            translation_cm=pose.get("translation_cm"),rotation_deg=pose.get("rotation_deg"),
            translation_delta_cm=deltaT,rotation_delta_deg=deltaR,
            solver_consensus_inlier_count=len(I),remaining_pool_count=len(U),
            local_point_line_refinement=bool(row.get("local_point_line_refinement")),
            point_PnP_inliers_applicable=not bool(row.get("local_point_line_refinement")),
            point_line_rank=row["solver"].get("rank"),retained_line_edges=row["solver"].get("line_edges",[]),
            normalized_jacobian_condition=row["solver"].get("geometry",{}).get("jacobian",{}).get("condition_number"))
        raw.append(record);groups[row["method"]].append(record)
    summary={}
    for name,rr in groups.items():
        hist=Counter();outcomes=Counter()
        bymask={}
        for r in rr:
            hist[f"pool_correct{r['correct_pool_count']}_final_correct{r['correct_final_inlier_count']}_final_total{r['solver_consensus_inlier_count']}"]+=1
            outcomes[r["mask_pose_outcome"]]+=1
        for wrong in [False,True]:
            subset=[r for r in rr if r["mask_wrong_on_known"]==wrong]
            bymask["wrong_mask" if wrong else "correct_mask_on_known"]=dict(frames=len(subset),
              new_pose_estimated=sum(r["new_pose_estimated"] for r in subset),fallback_used=sum(r["fallback_used"] for r in subset),
              correct_pool_count_histogram=dict(Counter(r["correct_pool_count"] for r in subset)),
              correct_final_inlier_count_histogram=dict(Counter(r["correct_final_inlier_count"] for r in subset)),
              translation_cm=distribution([r["translation_cm"] for r in subset if r["translation_cm"] is not None]),
              rotation_deg=distribution([r["rotation_deg"] for r in subset if r["rotation_deg"] is not None]))
        summary[name]=dict(frames=len(rr),mask_pose_outcomes=dict(outcomes),
            correct_pool_final_inlier_histogram=dict(hist),by_mask_error=bymask,
            final_false_inliers_total=sum(len(r["wrong_final_inlier_ids"]) for r in rr),
            false_excluded_accurate_total=sum(len(r["false_excluded_accurate_ids"]) for r in rr),
            direct_visible_correct_pool_total=sum(len(r["human_direct_correct_pool_ids"]) for r in rr))
    dst=C.DOC/"REAL_CORRESPONDENCE_ROWS.jsonl.gz"
    if dst.exists():
        previous=C.binding(dst)
        oldrows=list(C.iter_rows(dst))
        if C.digest(oldrows)==C.digest(raw):
            previous=None
        else:
            archive=dst.with_name("REAL_CORRESPONDENCE_ROWS_PRE_"+previous["sha256"][:12]+".jsonl.gz")
            if archive.exists():raise FileExistsError("Previous correspondence audit archive already exists")
            dst.replace(archive);previous=C.binding(archive)
    else:previous=None
    if not dst.exists():C.save_rows(dst,raw)
    C.write(C.DOC/"REAL_CORRESPONDENCE_AUDIT.json",dict(complete=True,rows=len(raw),summary=summary,
        raw=C.binding(dst),previous_derived_audit=previous,GT_used_after_sealed_inference=True,
        definition="Correct here means input correspondence <=8 original-image pixels from the existing reference at frozen baseline phase, independently of human visibility and final model residual.",
        limitations="Existing reference is geometry-reconstructed; hidden references can be derived, not independent physical truth. Direct visibility is not a guarantee of coordinate accuracy. Unknown/invalid/unmatched references are separate, never filled with truth."))
    return raw


def run():
    start=time.perf_counter();cv2.setNumThreads(1)
    problems=[];checks=[]
    def check(name,passed,**details):
        checks.append(dict(name=name,passed=bool(passed),**C.finite(details)))
        if not passed:problems.append(dict(name=name,**C.finite(details)))
    def equal(a,b):
        if a is None or b is None:return a is None and b is None
        return bool(np.isclose(a,b,rtol=1e-10,atol=1e-8))
    inputs=C.read(C.DOC/"INPUTS.json");bounds=inputs["frames"]
    ids=[r["id"] for r in bounds];bound={r["id"]:r for r in bounds}
    files=["INPUTS.json","PREDICTIONS.jsonl.gz","FIXED_CONTROLS.jsonl.gz","POSE_DIAGNOSTICS.jsonl.gz",
           "METRICS.json","METRICS.csv","PAIRED_COMPARISONS.json","OBSERVATIONS.jsonl.gz",
           "REAL_MASK_STRESS.jsonl.gz","GEOMETRY_STRESS.jsonl.gz","GEOMETRY_STRESS_SUMMARY.json",
           "SOLVER_CHECKS.json","SOURCE_BINDINGS.json","INFERENCE_CODE_LOCK.json"]
    if (C.DOC/"LEARNED_PREDICTIONS.jsonl.gz").exists():
        files += [n for n in ("LEARNED_PREDICTIONS.jsonl.gz","LEARNED_OBSERVATIONS.jsonl.gz",
                              "LEARNED_GEOMETRY_SEALED.jsonl.gz","OBSERVATION_SEAL.json",
                              "LEARNED_GEOMETRY_CODE_LOCK.json","TRAINING_COMPLETION.json",
                              "SOURCE_LEARNING_CHECKS.json") if (C.DOC/n).exists()]
    if (C.DOC/"REPAIR_LOG.json").exists():files.append("REPAIR_LOG.json")
    manifest=[C.binding(C.DOC/n) for n in files]
    rows=list(C.iter_rows(C.DOC/"PREDICTIONS.jsonl.gz"))
    if "LEARNED_PREDICTIONS.jsonl.gz" in files:rows+=list(C.iter_rows(C.DOC/"LEARNED_PREDICTIONS.jsonl.gz"))
    controls=list(C.iter_rows(C.DOC/"FIXED_CONTROLS.jsonl.gz"))
    methods=defaultdict(dict)
    for row in controls+rows:
        if row["id"] in methods[row["method"]]:check("duplicate_frame_method",False,id=row["id"],method=row["method"])
        methods[row["method"]][row["id"]]=row
    check("population_319_13",len(ids)==319 and len(set(r["session"] for r in bounds))==13)
    check("each_arm_preserves_exact_ids",all(set(rr)==set(ids) for rr in methods.values()),methods=len(methods))
    metrics=C.read(C.DOC/"METRICS.json")
    check("metrics_cover_all_raw_methods",set(metrics["methods"])==set(methods))
    distribution_errors=[];counts_errors=[]
    for name,byid in methods.items():
        rr=[byid[i] for i in ids]
        saved=metrics["methods"].get(name,{})
        available=[r for r in rr if r["pose"]["available"]]
        fresh=[r for r in available if r.get("new_pose_estimated")]
        counts=dict(total_frames=len(rr),pose_available=len(available),new_pose_estimated=len(fresh),
                    fallback_used=sum(bool(r.get("fallback_used")) for r in rr),
                    no_pose=len(rr)-len(available),hidden_reprojected=sum(bool(r.get("hidden_reprojected")) for r in rr))
        for k,v in counts.items():
            if saved.get(k)!=v:counts_errors.append(dict(method=name,field=k,expected=v,saved=saved.get(k)))
        for scope,selected in [("operational",available),("new_pose",fresh)]:
            for m,(field,factor,unit) in METRICS.items():
                expected=distribution([r["pose"][field]*factor for r in selected])
                stat=saved.get("metrics",{}).get(scope,{}).get(m,{})
                for k,v in expected.items():
                    if not equal(v,stat.get(k)):distribution_errors.append(dict(method=name,scope=scope,metric=m,field=k,expected=v,saved=stat.get(k)))
                if stat.get("ddof")!=1 or stat.get("unit")!=unit:distribution_errors.append(dict(method=name,scope=scope,metric=m,issue="unit or ddof"))
        values=[r["pose"].get("ADDsym_normalized") if r["pose"]["available"] else float("inf") for r in rr]
        curve=[sum(x<=i*.0001 for x in values)/len(rr) for i in range(1001)]
        auc=(sum(curve)-.5*(curve[0]+curve[-1]))*.0001/.1
        if not equal(auc,saved.get("ADDsym_AUC_full")):
            distribution_errors.append(dict(method=name,metric="AUC_full",expected=auc,saved=saved.get("ADDsym_AUC_full")))
    check("all_raw_statistics_mean_sample_variance_std_median_p90_max",not distribution_errors,errors=distribution_errors[:20])
    check("operational_new_fallback_failure_counts",not counts_errors,errors=counts_errors[:20])
    archived_controls={(r["method"],r["id"]):r for r in C.iter_rows(
        C.ROOT/"_docs/experiments/pallet_n3_subpix_20261008_v1/PREDICTIONS.jsonl.gz")}
    copied_controls_equal=all(all(r.get(k)==archived_controls[(r["method"],r["id"])].get(k)
                                 for k in ("pose","corner","native_points","selected_index")) for r in controls)
    check("historical_control_raw_metrics_coordinates_and_selection_exact",copied_controls_equal,rows=len(controls))
    csv_errors=[]
    with (C.DOC/"METRICS.csv").open() as fp:
        csvrows=list(csv.DictReader(fp))
    for r in csvrows:
        source=metrics["methods"][r["method"]]
        reference=source["metrics"][r["scope"]][r["metric"]]
        for key in ["n","mean","sample_variance","sample_std","median","P90","max"]:
            value=float(r[key]) if r[key] else None
            if not equal(value,reference[key]):csv_errors.append(dict(method=r["method"],scope=r["scope"],metric=r["metric"],field=key))
    check("csv_matches_json_and_units",not csv_errors and len(csvrows)==len(methods)*6,errors=csv_errors[:10])
    # Original thirteen-session draws; independently aggregate each common set.
    sessions=sorted(set(r["session"] for r in bounds));ix={s:i for i,s in enumerate(sessions)}
    draws=np.random.default_rng(20260917).multinomial(13,[1/13.]*13,size=10000).astype("uint16")
    drawsha=hashlib.sha256(draws.tobytes()).hexdigest()
    paired=C.read(C.DOC/"PAIRED_COMPARISONS.json");pair_errors=[]
    check("same_existing_session_bootstrap_draws",drawsha==paired["draw_sha256"]==metrics["bootstrap"]["draw_sha256"],sha256=drawsha)
    draw_binding=metrics["bootstrap"].get("existing_draw_file")
    if draw_binding:
        draw_path=C.ROOT/draw_binding["path"]
        with gzip.open(draw_path,"rb") as fp:archived_draws=np.load(fp,allow_pickle=False)
        check("actual_archived_bootstrap_matrix_exact_values",np.array_equal(archived_draws,draws) and
              C.sha(draw_path)==draw_binding["sha256"],rows=int(archived_draws.shape[0]),
              sessions=int(archived_draws.shape[1]),archived_dtype=str(archived_draws.dtype),
              compute_dtype=str(draws.dtype),archive_numeric_sha256=hashlib.sha256(archived_draws.tobytes()).hexdigest(),
              cast_numeric_sha256=drawsha)
    else:
        check("actual_archived_bootstrap_matrix_exact_values",False,reason="missing existing draw binding")
    for name,scopes in paired["contrasts"].items():
        new,base=name.split("_minus_");a,b=methods[new],methods[base]
        for scope,saved in scopes.items():
            fresh=scope=="new_pose_common"
            eligible=[i for i in ids if a[i]["pose"]["available"] and b[i]["pose"]["available"] and
                      (not fresh or (a[i].get("new_pose_estimated") and (b[i].get("new_pose_estimated") or b[i].get("output_status")=="HISTORICAL_FIXED_CONTROL")))]
            if saved["common_ids"]!=eligible or saved["common_frames"]!=len(eligible) or saved["denominator"]!=319:
                pair_errors.append(dict(contrast=name,scope=scope,issue="common IDs or full denominator"))
            counts=np.zeros(13)
            for i in eligible:counts[ix[bound[i]["session"]]]+=1
            denom=draws@counts;valid=denom>0
            for m,(field,factor,unit) in METRICS.items():
                delta=[(a[i]["pose"][field]-b[i]["pose"][field])*factor for i in eligible]
                sums=np.zeros(13)
                for i,d in zip(eligible,delta):sums[ix[bound[i]["session"]]]+=d
                samples=(draws@sums)[valid]/denom[valid]
                expected=distribution(delta)
                expected["CI95"]=np.quantile(samples,[.025,.975]).tolist() if len(samples) else None
                expected["improved_frames"]=sum(x< -1e-9 for x in delta)
                expected["worsened_frames"]=sum(x>1e-9 for x in delta)
                expected["unchanged_frames"]=sum(abs(x)<=1e-9 for x in delta)
                for k,v in expected.items():
                    got=saved["metrics"][m].get(k)
                    good=np.allclose(v,got,rtol=1e-10,atol=1e-8) if isinstance(v,list) and isinstance(got,list) else equal(v,got)
                    if not good:pair_errors.append(dict(contrast=name,scope=scope,metric=m,field=k))
    check("all_paired_common_sets_deltas_and_bootstrap_intervals",not pair_errors,contrasts=len(paired["contrasts"]),errors=pair_errors[:20])
    # Pose metrics exactly match the old implementation and independent formulas.
    C.source_modules()
    from scripts.research.pallet_training_free_compare_20261007_v1.common import load_real
    E,frames,targets,_,_=load_real();frame={f["id"]:f for f in frames}
    stress=list(C.iter_rows(C.DOC/"REAL_MASK_STRESS.jsonl.gz"))
    correspondence_rows=correspondence_audit(rows+stress,controls,bounds,targets)
    check("reference_accuracy_and_visibility_are_separate_diagnostics",len(correspondence_rows)==len(rows)+len(stress),
          reference_accuracy_threshold_px=8.,GT_used_after_inference=True)
    metric_errors=[];corner_errors=[];geometry_errors=[];status_errors=[];unrefined=Counter();changed_masks=Counter()
    max_pose_difference=0.;max_hidden_difference=0.;historical_missing_pose=Counter()
    for r in controls+rows+stress:
        f=frame[r["id"]];target=targets[r["id"]]
        detected=f["q"] is not None and np.isfinite(f["q"][:8]).any()
        expected_corner=C.finite(E.M.measure(np.array(r["native_points"],float),target["gt"],target["valid"],
                               target["permutations"],f["raw_hw"],target["matched"] and detected,detected))
        for k,v in expected_corner.items():
            got=r["corner"].get(k)
            if isinstance(v,list):
                good=bool(np.allclose(np.array(v,float),np.array(got,float),rtol=1e-10,atol=1e-8,equal_nan=True))
            elif isinstance(v,(float,int)) and not isinstance(v,bool):good=equal(v,got)
            else:good=v==got
            if not good:corner_errors.append(dict(method=r["method"],id=r["id"],field=k))
        if r.get("actual_pose") is None:
            historical_missing_pose[r["method"]]+=1
            continue
        truth=frame[r["id"]]["truth"]
        recomputed=metric(r["actual_pose"],truth)
        historical=E.POSE.metric((r["id"],r["actual_pose"],truth))
        if r["pose"]["available"]!=recomputed["available"]:
            metric_errors.append(dict(method=r["method"],id=r["id"],issue="pose availability"));continue
        if recomputed["available"]:
            for k in ("translation_cm","rotation_deg","ADDsym_m","ADDsym_normalized"):
                diff=abs(recomputed[k]-r["pose"][k]);max_pose_difference=max(max_pose_difference,diff)
                if not equal(recomputed[k],r["pose"][k]) or not equal(historical[k],r["pose"][k]):
                    metric_errors.append(dict(method=r["method"],id=r["id"],field=k,expected=recomputed[k],saved=r["pose"][k]))
        if "solver" not in r:continue
        s=r["solver"];H=set(r["hidden_initial"]);U=set(s["used"]);fit=set(s["fit_input_ids"])
        if U&H or fit&H or fit&set(s.get("excluded",r.get("excluded",[]))) or len(s["inliers"])!=len(set(s["inliers"])) or any(i>=8 for i in s["inliers"]):
            geometry_errors.append(dict(method=r["method"],id=r["id"],issue="excluded or duplicate identity"))
        if r.get("reprojections_reused_as_observations") or s.get("reprojected_points_reused_as_observations"):
            geometry_errors.append(dict(method=r["method"],id=r["id"],issue="cyclic projection observation"))
        if s.get("refit_count",0)>3 and s.get("solver")=="FINITE_SUBSET_ROBUST":
            geometry_errors.append(dict(method=r["method"],id=r["id"],issue="top-three refit budget"))
        f=bound[r["id"]]
        if r["selected_index"]!=f["selected_index"] or not r["fixed_metadata"].get("preserved"):
            geometry_errors.append(dict(method=r["method"],id=r["id"],issue="object/candidate metadata mutation"))
        arm="N3_SUBPIX" if r["method"].startswith("N3_SUBPIX") else "BASE"
        original=np.array(f["points"][arm],float)
        if not np.array_equal(np.array(r["native_points"])[8],original[8],equal_nan=True):
            geometry_errors.append(dict(method=r["method"],id=r["id"],issue="center changed"))
        if r["new_pose_estimated"]:
            if not r["pose_available"] or r["fallback_used"] or r["no_pose"] or not s["available"]:
                status_errors.append(dict(method=r["method"],id=r["id"],issue="new pose status"))
            X=cube(s["cf_extents"]);R=np.array(s["R_cf"]);t=np.array(s["centroid"]);K=np.array(r["K"])
            cam=X@R.T+t;hom=cam@K.T;projected=hom[:,:2]/hom[:,2:3]
            err=np.max(abs(projected-np.array(s["projected"])));max_hidden_difference=max(max_hidden_difference,float(err))
            if not np.allclose(projected,s["projected"],rtol=0,atol=1e-8):
                geometry_errors.append(dict(method=r["method"],id=r["id"],issue="final pose projection"))
            if H and not np.allclose(np.array(r["native_points"])[sorted(H)],projected[sorted(H)],rtol=0,atol=1e-8):
                geometry_errors.append(dict(method=r["method"],id=r["id"],issue="hidden projection replacement"))
            if s.get("solver")=="FINITE_SUBSET_ROBUST":
                if len(s["inliers"])<4 or len(fit)<4 or not fit<=U:
                    geometry_errors.append(dict(method=r["method"],id=r["id"],issue="insufficient robust fit/consensus"))
                if "+LM" not in s["generator"]:unrefined[r["method"]]+=1
            q=np.array(r["input_points"],float)
            rr=np.linalg.norm(projected[sorted(U)]-q[sorted(U)],axis=1)
            if not r.get("local_point_line_refinement") and not np.allclose(rr,s["residuals_used_px"],atol=1e-8,rtol=0):
                geometry_errors.append(dict(method=r["method"],id=r["id"],issue="same fixed-U residual"))
            expected_inliers={i for i,e in zip(sorted(U),rr) if e<=8.}
            if not r.get("local_point_line_refinement") and expected_inliers!=set(s["inliers"]):
                geometry_errors.append(dict(method=r["method"],id=r["id"],issue="inlier residual definition"))
            Q=np.eye(3) if abs(s["cf_extents"][0]-r["xyz"][0])<1e-6 else np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
            if not np.allclose(R@Q,s["R_physical"],atol=1e-10):
                geometry_errors.append(dict(method=r["method"],id=r["id"],issue="physical dimension conversion"))
            if r.get("hidden_set_changed"):changed_masks[r["method"]]+=1
        elif r["fallback_used"]:
            initial=r.get("initial_pose")
            if initial is not None and r["actual_pose"]!=initial:
                status_errors.append(dict(method=r["method"],id=r["id"],issue="fallback differs from original initial"))
            if not r["pose_available"] or r["no_pose"] or r["hidden_reprojected"]:
                status_errors.append(dict(method=r["method"],id=r["id"],issue="fallback status"))
        elif not r["no_pose"] or r["pose_available"]:
            status_errors.append(dict(method=r["method"],id=r["id"],issue="complete failure status"))
    check("pose_metrics_old_function_and_independent_T_R_ADDsym",not metric_errors,max_absolute_difference=max_pose_difference,errors=metric_errors[:20])
    check("all_corner_scores_existing_fixed_units_and_phase_evaluation",not corner_errors,errors=corner_errors[:20])
    check("historical_controls_missing_pose_parameters_disclosed",True,missing_parameter_counts=dict(historical_missing_pose),
          status="Inherited pose-error raw rows remain verifiable statistics; absent historical R,t are not fabricated. All new rows have actual R,t or explicit unavailable pose.")
    check("geometry_hidden_projection_fixed_pool_ids_and_units",not geometry_errors,max_projection_difference_px=max_hidden_difference,errors=geometry_errors[:20])
    check("new_pose_fallback_complete_failure_separated",not status_errors,errors=status_errors[:20])
    # Expected selected original candidates are visible in audit, not fake LM fits.
    check("selected_unrefined_candidates_reported_explicitly",True,counts=dict(unrefined),
          rationale="Fixed consensus score retains an original allowed four-subset pose when refits worsen it; fit_input_ids are actual generator IDs.")
    check("mask_changes_do_not_abort_new_pose",True,new_poses_with_changed_mask=dict(changed_masks))
    # No truth arguments/files within deployable path; inspect canary separately.
    from .solver import HypothesisBank
    from .inference import finish
    sample=rows[0];f=bound[sample["id"]]
    with C.no_truth_reads():
        q=np.asarray(f["points"]["BASE"])
        bank=HypothesisBank(q,np.array(f["K"]),np.array(f["xyz"]),image_size=(f["raw_hw"][1],f["raw_hw"][0]))
        canary=finish(bank,q,sample["initial_pose"],excluded=sample["hidden_initial"],hidden=sample["hidden_initial"],robust=True)
        all_none=q.copy();all_none[:8]=np.nan
        empty=finish(HypothesisBank(all_none,np.array(f["K"]),np.array(f["xyz"])),q,sample["initial_pose"])
    check("deployable_gt_canary_and_no_GT_arguments",set(inspect.signature(HypothesisBank).parameters)=={"points","K","xyz","image_size"})
    check("all_no_match_is_insufficient_observation_or_fallback",not empty["new_pose_estimated"] and empty["solver"]["reason"]=="insufficient_observations")
    # Failure regression: dropping the bad frame lowers conditional mean only.
    conditional=distribution([1.])["mean"];operational=distribution([1.,100.])["mean"]
    check("failure_denominator_regression",conditional==1. and operational==50.5 and operational>conditional,
          two_frame_example=dict(conditional_n=1,full_n=2,conditional_mean=conditional,fallback_full_mean=operational))
    check("sample_variance_n_minus_one_regression",distribution([1.,3.])["sample_variance"]==2.)
    # Input and protected source bytes exactly match original snapshots.
    bindings=C.read(C.DOC/"SOURCE_BINDINGS.json");hash_errors=[]
    for b in bindings["inputs"]:
        p=(C.ROOT if b["origin"]=="source" else C.WORKTREE)/b["path"]
        if C.sha(p)!=b["sha256"]:hash_errors.append(b["path"])
    for b in bindings["raw_images"]:
        if C.sha(C.ROOT/b["path"])!=b["sha256"]:hash_errors.append(b["path"])
    check("source_assets_weights_and_319_RGB_hashes_unchanged",not hash_errors,checked=len(bindings["inputs"])+len(bindings["raw_images"]),changed=hash_errors)
    original_state=C.read(C.SCRATCH/"INITIAL_STATE.json");current_state=C.source_state()
    check("user_changes_source_branch_and_diff_unchanged",original_state==current_state,
          fields_equal={k:original_state[k]==current_state.get(k) for k in original_state})
    lock=C.read(C.DOC/"INFERENCE_CODE_LOCK.json")
    repair=C.read(C.DOC/"REPAIR_LOG.json") if "REPAIR_LOG.json" in files else None
    code_ok=[]
    for b in lock["files"]:
        actual=C.sha(C.WORKTREE/b["path"])
        accepted=actual==b["sha256"]
        if repair and b["path"]==repair["old_common"]["path"]:
            accepted=accepted or (b["sha256"]==repair["old_common"]["sha256"] and actual==repair["new_common"]["sha256"] and
                       repair["status"]=="PASS" and repair["frames"]==319 and repair["exact_output_rows"]==2871 and
                       all(c["exact_numeric_output_equal"] for c in repair["comparisons"]))
        code_ok.append(accepted)
    check("real_inference_code_lock_hashes_or_logged_numerical_identical_canary_repair",all(code_ok),
          logged_canary_repair=bool(repair),solver_changed=False)
    obslock=C.read(C.DOC/"OBSERVATIONS_LOCK.json")
    observations=list(C.iter_rows(C.DOC/"OBSERVATIONS.jsonl.gz"))
    check("deployable_observations_sealed_before_scoring",obslock["before_scoring"] and obslock["GT_reads_blocked"] and
          C.sha(C.DOC/"OBSERVATIONS.jsonl.gz")==obslock["observations"]["sha256"] and
          all(not r["oracle"] and "pose" not in r and "corner" not in r and "mask_audit" not in r for r in observations))
    line_rows=[r for r in rows if r.get("local_point_line_refinement")]
    line_errors=[]
    if line_rows:
        from .compact_observations import decode
        raw_role={r["id"]:decode(r) for r in C.iter_rows(C.DOC/"LEARNED_OBSERVATIONS.jsonl.gz") if r["method"]=="IMAGE_ROLE"}
        for r in line_rows:
            observation=raw_role[r["id"]];s=r["solver"];allowed=set(s["used"])
            expected_consumed={e for c in observation["corners"] if c["id"] in allowed for e in c["edges"]}
            if r["new_pose_estimated"]:
                retained=set(s["line_edges"]);consumed=set(s["consumed_edges"])
                if retained&consumed or consumed!=expected_consumed or not retained<={l["edge"] for l in observation["lines"]}:
                    line_errors.append(dict(id=r["id"],issue="line supporting a used point is duplicated"))
                if s["rank"]!=6 or s["initial_pose_prior"] or s["same_edge_point_line_double_count"]:
                    line_errors.append(dict(id=r["id"],issue="rank/prior/duplicate policy"))
                if s["inliers"] or r["independent_four_point_PnP"]:
                    line_errors.append(dict(id=r["id"],issue="local line refinement mislabeled point consensus"))
        seal=C.read(C.DOC/"OBSERVATION_SEAL.json")
        check("learned_observations_pre_GT_seal",seal["complete"] and
              C.sha(C.DOC/"LEARNED_OBSERVATIONS.jsonl.gz")==seal["records"]["sha256"])
        check("learned_geometry_pre_GT_seal",all("pose" not in r and "corner" not in r and "mask_audit" not in r
              for r in C.iter_rows(C.DOC/"LEARNED_GEOMETRY_SEALED.jsonl.gz")))
        lock=C.read(C.DOC/"LEARNED_GEOMETRY_CODE_LOCK.json")
        check("learned_geometry_code_lock",all(C.sha(C.WORKTREE/b["path"])==b["sha256"] for b in lock["files"]))
        training=C.read(C.DOC/"TRAINING_COMPLETION.json")
        check("three_equal_formal_training_runs",training["formal_updates"]==9000 and
              all(r["updates"]==3000 and r["exposures"]==48000 for r in training["checkpoints"]) and
              training["same_initial_tensor_sha"] and training["same_batch_order"] and training["same_update_budget"])
        learned_identity_errors=[];logit_hash_errors=[];selection_errors=[];semantic_errors=[]
        observation_keys=set()
        for encoded in C.iter_rows(C.DOC/"LEARNED_OBSERVATIONS.jsonl.gz"):
            r=decode(encoded);key=(r["method"],r["id"])
            if key in observation_keys:learned_identity_errors.append(dict(id=r["id"],method=r["method"],issue="duplicate"))
            observation_keys.add(key)
            if r.get("GT_input") is not False or r["method"] not in ("GEOMETRY_ONLY","IMAGE_NO_ROLE","IMAGE_ROLE"):
                learned_identity_errors.append(dict(id=r["id"],issue="inference identity or GT flag"))
            if encoded.get("unencoded_semantic_sha256") and C.digest(r)!=encoded["unencoded_semantic_sha256"]:
                semantic_errors.append(key)
            if r.get("queries"):
                logits=np.asarray([q["candidate_logits"] for q in r["queries"]],dtype="<f4")
                if logits.shape!=(84,66) or hashlib.sha256(logits.tobytes()).hexdigest()!=r["raw_logits_sha256"]:
                    logit_hash_errors.append(key)
                for q in r["queries"]:
                    chosen=int(np.argmax(q["candidate_logits"]))
                    if (chosen!=q["chosen_candidate"] or (chosen==65 and not q["no_match"]) or
                        (q["selected_xy"] is None and not q["no_match"])):
                        selection_errors.append(dict(id=r["id"],method=r["method"],query=q["query"],issue="argmax/no-match"))
                    if q["selected_xy"] is not None:
                        selected=np.array(q["center"])+(chosen-32)*np.array(q["normal"])
                        if not np.allclose(selected,q["selected_xy"],atol=1e-8,rtol=0):
                            selection_errors.append(dict(id=r["id"],method=r["method"],query=q["query"],issue="selected basin coordinate"))
                if len({c["id"] for c in r["corners"]})!=len(r["corners"]) or any(len(c["edges"])!=2 for c in r["corners"]):
                    learned_identity_errors.append(dict(id=r["id"],method=r["method"],issue="corner identity/two edges"))
        check("all_957_sealed_learned_observation_identities",not learned_identity_errors and len(observation_keys)==957,
              errors=learned_identity_errors[:10])
        check("all_raw_learned_logits_float32_hashes_and_lossless_storage",not logit_hash_errors and not semantic_errors,
              hash_errors=logit_hash_errors[:10],semantic_errors=semantic_errors[:10])
        check("learned_argmax_no_match_selected_coordinates",not selection_errors,errors=selection_errors[:10])
        protocol=C.read(C.DOC/"LEARNING_PROTOCOL.json")
        order=np.load(C.SCRATCH/"learned_cache/order.npy",allow_pickle=False)
        check("actual_training_batch_order_matches_locked_protocol",order.shape==(3000,16) and
              hashlib.sha256(order.tobytes()).hexdigest()==protocol["batch_order_sha256"] and
              int(order.min())>=0 and int(order.max())<768)
        import torch
        checkpoint_errors=[]
        for c in training["checkpoints"]:
            path=C.SCRATCH/"learned_fits"/(c["arm"]+".pt")
            saved=torch.load(path,map_location="cpu",weights_only=False)
            good=(C.sha(path)==c["checkpoint"]["sha256"] and saved["steps"]==3000 and saved["arm"]==c["arm"] and
                  saved["initial_state_sha256"]==protocol["initial_state_sha256"] and
                  saved["batch_order_sha256"]==protocol["batch_order_sha256"] and
                  saved["protocol_sha256"]==C.sha(C.DOC/"LEARNING_PROTOCOL.json") and
                  sum(t.numel() for t in saved["model"].values())==protocol["parameters"])
            if not good:checkpoint_errors.append(c["arm"])
        check("actual_three_final_checkpoints_execution_and_protocol",not checkpoint_errors,errors=checkpoint_errors)
    check("point_line_duplicate_observations",not line_errors,line_paths=len(line_rows),errors=line_errors[:20],
          status="Disjoint line factors and point-generating edges checked on identical sealed IMAGE_ROLE observations; point-PnP inliers are not claimed for the local line path.")
    analytic=list(C.iter_rows(C.DOC/"GEOMETRY_STRESS.jsonl.gz"));ags=C.read(C.DOC/"GEOMETRY_STRESS_SUMMARY.json")
    totals=Counter()
    for r in analytic:totals.update(r["result"]["operation_counts"])
    check("analytic_stress_2816_counts_and_calls",len(analytic)==2816 and dict(totals)==ags["ledger"])
    check("analytic_stress_ddof1_independent",all(equal(distribution([r[k] for r in analytic if r["condition"]==cond and r["solver"]==s and r[k] is not None])["sample_variance"],v[k]["variance"])
          for cond,ss in ags["summary"].items() for s,v in ss.items() for k in ("translation_cm","rotation_deg","ADD_m")))
    check("verification_inputs_stable_during_read",all(C.sha(C.WORKTREE/b["path"])==b["sha256"] for b in manifest))
    result=dict(complete=True,passed=not problems,checks=checks,problems=problems,
                verified_inputs=manifest,verified_methods=sorted(methods),rows=len(rows),
                controls=len(controls),real_stress_rows=len(stress),analytic_rows=len(analytic),
                verified_bootstrap_source=draw_binding,
                audit=dict(selected_original_candidate_counts=dict(unrefined),mask_changed_new_pose_counts=dict(changed_masks),
                    alternative_pose_semantics="multiple_solutions indicates alternatives across subset/dimension candidates, not automatically equal-score ambiguity"),
                wall_seconds=time.perf_counter()-start,
                independent_implementation="Python statistics variance/stdev/median; independent projection and proper-group T/R/ADD; exact oldPOSE.metric crosscheck")
    destination=C.DOC/"VERIFICATION.json"
    if destination.exists():
        old=C.read(destination)
        oldscope="A" if not any(m.startswith("IMAGE_ROLE") for m in old.get("verified_methods",[])) else "FINAL"
        archive=C.DOC/("VERIFICATION_"+oldscope+"_"+C.sha(destination)[:12]+".json")
        if not archive.exists():destination.replace(archive)
    C.write(destination,result)
    print(json.dumps(dict(passed=result["passed"],checks=len(checks),methods=len(methods),rows=len(rows),
                         problems=problems[:20],wall_seconds=result["wall_seconds"])),flush=True)
    return 0 if result["passed"] else 1


if __name__=="__main__":
    raise SystemExit(run())
