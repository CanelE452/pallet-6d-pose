"""Independent postseal arithmetic from actual physical poses, NumPy/stdlib.

No campaign evaluator, old metric, optimizer or method implementation imports.
Reference decoding occurs only after explicit root GO and all seal checks.
The metric uses atan2 rotation geometry; moments use math.fsum and quantiles
use an independent sorted linear interpolation. Only the prescribed NumPy
multinomial draw schedule is shared with the descriptive bootstrap policy.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import resource
import time

import numpy as np

BASELINES=("R0","N0","E0","POSEFIX_RAW","POSEFIX_SUBPIX_PF_ANCHOR","POSEFIX_SUBPIX_YOLO_ANCHOR")
COMPARATORS=("E0","N0","POSEFIX_SUBPIX_PF_ANCHOR","POSEFIX_SUBPIX_YOLO_ANCHOR")
POPULATIONS={"PRIMARY192":("TEST","REMAINDER"),"TEST49":("TEST",),"REMAINDER143":("REMAINDER",),"CAL46":("CAL",),"ALL238":("CAL","TEST","REMAINDER")}
EXPECTED_N={"PRIMARY192":192,"TEST49":49,"REMAINDER143":143,"CAL46":46,"ALL238":238}
ARM_MODE={"CTRL_CARRY":"fixed","CTRL_BUDGET":"fixed","CTRL_FIXED":"fixed","A_FIXED_BANK":"A","B_FIXED_BANK":"B",
          "A_FULL":"A","B_FULL":"B","AB_FULL":"AB","CTRL_EDGE_BOUNDARY":"edge"}
SEED=20260917
DRAWS=10000
_DRAW_CACHE={}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda:f.read(4*1024*1024),b""):h.update(part)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def rows(path):
    with gzip.open(path,"rt") as f:
        for line in f:
            if line.strip():yield json.loads(line)


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n")


def linear_quantile(values,p):
    ordered=sorted(float(v) for v in values)
    if not ordered:return None
    position=(len(ordered)-1)*p;lo=int(math.floor(position));hi=int(math.ceil(position))
    return ordered[lo]+(ordered[hi]-ordered[lo])*(position-lo)


def pose_metric(pose,reference):
    """Same-pose joint center/C2 result using an independent stable angle."""
    absent=dict(pose_available=False,translation_cm=None,rotation_deg=None,success=False)
    if not isinstance(pose,dict) or pose.get("available") is not True:return absent
    try:
        R=np.asarray(pose["R_physical"],float);t=np.asarray(pose["centroid"],float)
        G=np.asarray(reference["R"],float);g=np.asarray(reference["t"],float)
        if reference["order"]!=2:raise ValueError("Only original physical C2 is admissible")
        if R.shape!=(3,3) or t.shape!=(3,) or G.shape!=(3,3) or g.shape!=(3,):return absent
        if not np.isfinite(R).all() or not np.isfinite(t).all():return absent
        if np.max(np.abs(R.T@R-np.eye(3)))>1e-7 or abs(float(np.linalg.det(R))-1)>1e-7:return absent
        delta=t-g
        translation=100*math.sqrt(math.fsum(float(x)*float(x) for x in delta))
        angles=[]
        for symmetry in (np.eye(3),np.diag([-1.,1.,-1.])):
            relative=(G@symmetry).T@R
            cosine=max(-1.,min(1.,(float(np.trace(relative))-1)/2))
            skew=(float(relative[2,1]-relative[1,2]),float(relative[0,2]-relative[2,0]),float(relative[1,0]-relative[0,1]))
            sine=.5*math.sqrt(math.fsum(x*x for x in skew))
            angles.append(math.degrees(math.atan2(sine,cosine)))
        angle=min(angles)
        return dict(pose_available=True,translation_cm=translation,rotation_deg=angle,
                    success=translation<=5 and angle<=5)
    except (KeyError,TypeError,ValueError,ArithmeticError):return absent


def strict_direction_values(pose,reference):
    """Secondary replay of the original floating-point metric expression.

    This does not replace the independent atan2/fsum metric or recompute
    success labels. Its only purpose is verifying strict delta<0/>0/==0
    counts whose sign can depend on an ulp at zero. No evaluator is imported.
    """
    absent=dict(translation_cm=None,rotation_deg=None)
    if not isinstance(pose,dict) or pose.get("available") is not True:return absent
    try:
        R=np.asarray(pose["R_physical"],float);t=np.asarray(pose["centroid"],float)
        G=np.asarray(reference["R"],float);g=np.asarray(reference["t"],float)
        if R.shape!=(3,3) or t.shape!=(3,) or not np.isfinite(R).all() or not np.isfinite(t).all():return absent
        if np.max(np.abs(R.T@R-np.eye(3)))>1e-7 or abs(float(np.linalg.det(R))-1)>1e-7:return absent
        if reference["order"]!=2:raise ValueError("Original physical C2 required")
        translation=float(np.linalg.norm(t-g)*100.)
        c,s=math.cos(math.pi),math.sin(math.pi)
        symmetries=(np.eye(3),np.array([[c,0.,s],[0.,1.,0.],[-s,0.,c]]))
        angle=min(math.degrees(math.acos(max(-1.,min(1.,float((np.trace((G@S).T@R)-1.)/2.))))) for S in symmetries)
        return dict(translation_cm=translation,rotation_deg=angle)
    except (KeyError,TypeError,ValueError,ArithmeticError):return absent


def distribution(values,denominator):
    valid=[float(x) for x in values if x is not None]
    n=len(valid);mean=math.fsum(valid)/n if n else None
    variance=math.fsum((x-mean)*(x-mean) for x in valid)/(n-1) if n>1 else None
    return dict(n=n,full_denominator=denominator,missing=denominator-n,mean=mean,
                sample_std=math.sqrt(variance) if variance is not None else None,sample_variance=variance,
                median=linear_quantile(valid,.5),P90=linear_quantile(valid,.9),P95=linear_quantile(valid,.95),
                max=max(valid) if valid else None,ddof=1,available_only=True,tail_trimming=False)


def aggregate(scored,arms):
    result={}
    for arm in arms:
        metrics=[r["metrics"][arm] for r in scored];successes=sum(m["success"] for m in metrics)
        status_counts={}
        for key in ("accepted","fallback","technical_failure","solver_rejected","timeout"):
            flags=[r.get("status",{}).get(arm,{}).get(key) for r in scored]
            status_counts[key]=dict(true_n=sum(x is True for x in flags),false_n=sum(x is False for x in flags),
                                    unknown_n=sum(x is None for x in flags))
        status_counts["raw_status"]=dict(Counter(str(r.get("status",{}).get(arm,{}).get("status")) for r in scored))
        result[arm]=dict(denominator=len(scored),successes=successes,success_rate=successes/len(scored),
            no_pose=sum(not m["pose_available"] for m in metrics),
            translation_cm=distribution([m["translation_cm"] for m in metrics],len(scored)),
            rotation_deg=distribution([m["rotation_deg"] for m in metrics],len(scored)),
            T_gt50cm=sum(m["translation_cm"] is not None and m["translation_cm"]>50 for m in metrics),
            T_gt100cm=sum(m["translation_cm"] is not None and m["translation_cm"]>100 for m in metrics),
            R_gt45deg=sum(m["rotation_deg"] is not None and m["rotation_deg"]>45 for m in metrics),
            status_counts=status_counts,failures=[r["id"] for r in scored if not r["metrics"][arm]["success"]])
    return result


def cluster_ci(values,scored,unit):
    labels=sorted({r[unit] for r in scored})
    if len(labels)<2:return dict(CI95=None,reason="fewer than two original clusters",cluster_count=len(labels))
    k=len(labels)
    if k not in _DRAW_CACHE:
        _DRAW_CACHE[k]=np.random.default_rng(SEED).multinomial(k,np.full(k,1/k),size=DRAWS)
    weights=_DRAW_CACHE[k]
    # Own grouped arithmetic. The shared RNG schedule does not share any
    # evaluator-produced sum, count, draw value or interval endpoint.
    numerator=np.zeros(DRAWS,float);denominator=np.zeros(DRAWS,np.int64)
    for j,label in enumerate(labels):
        v=[float(x) for r,x in zip(scored,values) if r[unit]==label and x is not None]
        numerator+=weights[:,j]*math.fsum(v)
        denominator+=weights[:,j]*len(v)
    valid=[float(a/b) for a,b in zip(numerator,denominator) if b>0]
    return dict(CI95=[linear_quantile(valid,.025),linear_quantile(valid,.975)] if valid else None,
        cluster_count=k,clusters=labels,cluster_sizes=dict(Counter(r[unit] for r in scored)),
        valid_resamples=len(valid),empty_resamples=DRAWS-len(valid),seed=SEED,resamples=DRAWS,unit=unit)


def paired(scored,before,after,*,direction_view="stable"):
    groups={k:[] for k in ("BOTH_SUCCESS","RECOVERY","DAMAGE","BOTH_FAILURE")}
    differences={k:[] for k in ("success_rate","translation_cm","rotation_deg")}
    for row in scored:
        a,b=row["metrics"][before],row["metrics"][after]
        key="BOTH_SUCCESS" if a["success"] and b["success"] else "RECOVERY" if b["success"] else "DAMAGE" if a["success"] else "BOTH_FAILURE"
        groups[key].append(row["id"])
        differences["success_rate"].append(int(b["success"])-int(a["success"]))
        for key in ("translation_cm","rotation_deg"):
            differences[key].append(b[key]-a[key] if a[key] is not None and b[key] is not None else None)
    deltas={}
    for name,values in differences.items():
        valid=[v for v in values if v is not None]
        # Magnitudes, means and CIs retain the independent stable metric.
        # Only strict direction counts use the separately audited expression.
        direction=values
        if name!="success_rate" and direction_view=="original_expression":
            direction=[r["strict_direction_metrics"][after][name]-r["strict_direction_metrics"][before][name]
                       if r["strict_direction_metrics"][before][name] is not None and r["strict_direction_metrics"][after][name] is not None
                       else None for r in scored]
        direction=[v for v in direction if v is not None]
        deltas[name]=dict(mean=math.fsum(valid)/len(valid) if valid else None,common_available_n=len(valid),
            improved=sum((v>0 if name=="success_rate" else v<0) for v in direction),
            worsened=sum((v<0 if name=="success_rate" else v>0) for v in direction),
            recording_cluster_CI=cluster_ci(values,scored,"cluster_id"),
            named_session_cluster_sensitivity_CI=cluster_ci(values,scored,"session"))
    count=sum(differences["success_rate"])
    common_failure=[r for r in scored if not r["metrics"][before]["success"] and not r["metrics"][after]["success"]]
    continuous={}
    for key in ("translation_cm","rotation_deg"):
        details=[dict(id=r["id"],before=r["metrics"][before][key],after=r["metrics"][after][key],
                      delta=r["metrics"][after][key]-r["metrics"][before][key]) for r in common_failure
                 if r["metrics"][before][key] is not None and r["metrics"][after][key] is not None]
        direction=[d["delta"] for d in details]
        if direction_view=="original_expression":
            direction=[r["strict_direction_metrics"][after][key]-r["strict_direction_metrics"][before][key]
                       for r in common_failure if r["strict_direction_metrics"][before][key] is not None and
                       r["strict_direction_metrics"][after][key] is not None]
        continuous[key]=dict(common_available_n=len(details),both_failure_full_n=len(common_failure),
            improved_n=sum(d<0 for d in direction),worsened_n=sum(d>0 for d in direction),tied_n=sum(d==0 for d in direction),
            improvement_sum=math.fsum(d["delta"] for d in details if d["delta"]<0),
            worsening_sum=math.fsum(d["delta"] for d in details if d["delta"]>0))
    return dict(before=before,after=after,denominator=len(scored),counts={k:len(v) for k,v in groups.items()},ids=groups,
        success_difference_count=count,success_difference_percentage_points=100*count/len(scored),deltas=deltas,
        both_failure_tail=aggregate(common_failure,(before,after)) if common_failure else {},both_failure_continuous=continuous)


def candidate_pose(candidate):
    if isinstance(candidate.get("actual_pose"),dict):return candidate["actual_pose"]
    if "R_physical" in candidate and "centroid" in candidate:
        return dict(available=True,R_physical=candidate["R_physical"],centroid=candidate["centroid"])
    value=candidate.get("solve",{}).get("actual_pose")
    return value if isinstance(value,dict) else dict(available=False)


def bank_summary(bank,reference):
    metrics=[pose_metric(candidate_pose(c),reference) for c in bank]
    return dict(candidate_count=len(bank),has_success=any(m["success"] for m in metrics),
        success_ids=[c.get("candidate_id","UNRECORDED") for c,m in zip(bank,metrics) if m["success"]],
        invalid_or_unavailable_candidate_count=sum(not m["pose_available"] for m in metrics))


def stage_summary(prediction,reference,new_arms,baseline_poses=None):
    carried=prediction.get("carried",[]);result=dict(carried=bank_summary(carried,reference),arms={},
        new_attempt_count=len(prediction.get("attempts",[])),carried_unique_count=len(carried))
    archived=prediction.get("archived_initial_candidates")
    if isinstance(archived,list):
        result["original_candidate_archive_counts"]=dict(available_source_pose_roles=len(archived),
            positive_depth_admitted_source_pose_roles=sum(c.get("initial_projection_admitted") is True for c in archived),
            projection_ineligible_preserved_source_pose_roles=sum(c.get("initial_projection_admitted") is not True for c in archived),
            original_outcome_Rt_unavailable=sum(v for k,v in prediction.get("initial_candidate_source_counts",{}).items()
                                               if k.endswith("_Rt_unavailable")))
    else:result["original_candidate_archive_counts"]=prediction.get("original_candidate_archive_counts",prediction.get("candidate_archive_counts"))
    for arm in new_arms:
        final=pose_metric(prediction["predictions"][arm]["actual_pose"],reference)
        if arm=="CTRL_RANSAC":
            result["arms"][arm]=dict(status="ARCHIVE_GAP",candidate_oracle_available=False,
                family_intermediate_status="ARCHIVE_GAP",
                ransac_attempt_metadata=prediction["predictions"][arm].get("ransac_attempts",[]),
                final_success=final["success"],final_candidate_id=prediction["predictions"][arm].get("selected_candidate_id"))
            continue
        selection=(prediction.get("selection_banks") or {}).get(arm)
        if selection is not None:
            pool=selection["candidates"];pool_ids={c["candidate_id"] for c in pool}
            eligible_ids=set(selection.get("eligible_ids",pool_ids))
            if eligible_ids-pool_ids:raise ValueError("Learned eligibility points outside its sealed pool")
            generated=bank_summary(pool,reference)
            admitted=bank_summary([c for c in pool if c["candidate_id"] in eligible_ids],reference)
            failure="FINAL_SUCCESS" if final["success"] else "NO_SUCCESS_IN_LEARNED_POOL" if not generated["has_success"] else "LEARNED_ELIGIBILITY_FILTER_LOSS" if not admitted["has_success"] else "LEARNED_SELECTION_OR_ABSTENTION_LOSS"
            result["arms"][arm]=dict(status="DIAGNOSED_POSTSEAL_ONLY",candidate_oracle_available=True,
                mode="LEARNED_FINITE_CANDIDATE_SELECTOR",generated_before_feasibility=generated,
                after_constraints=admitted,retained_bank=generated,eligibility_is_not_new_geometric_constraint=True,
                family_intermediate_status="NOT_APPLICABLE_DIRECT_SOFTMAX_SELECTION",
                final_success=final["success"],final_candidate_id=prediction["predictions"][arm].get("selected_candidate_id"),stage_loss_class=failure)
            continue
        if arm=="CTRL_FIXED":
            attempts=[c for c in prediction.get("attempts",[]) if (c.get("solve") or {}).get("start_id")=="N0_LOCAL_START"]
            admitted_bank=[c for c in attempts if (c.get("solve") or {}).get("feasible") is True]
            n0=(baseline_poses or {}).get("N0") or (prediction.get("features") or {}).get("fallback")
            if n0 is None:n0=next((candidate_pose(c) for c in carried if c.get("candidate_id")=="N0"),None)
            initial=[] if n0 is None else [dict(candidate_id="N0_INITIAL_OR_FALLBACK",actual_pose=n0)]
            generated=bank_summary(initial+attempts,reference);admitted=bank_summary(admitted_bank,reference)
            failure="FINAL_SUCCESS" if final["success"] else "NO_SUCCESS_IN_N0_AND_TWO_FIXED_START_ATTEMPTS" if not generated["has_success"] else "SUCCESS_LOST_BEFORE_ADMISSIBLE_FIXED_TWO_START_BANK" if not admitted["has_success"] else "FIXED_SELECTION_LOSS_FAMILY_ARCHIVE_GAP"
            result["arms"][arm]=dict(status="DIAGNOSED_POSTSEAL_ONLY",mode="fixed",candidate_oracle_available=True,
                generated_before_feasibility=generated,after_constraints=admitted,retained_bank=admitted,
                family_intermediate_status="ARCHIVE_GAP_CROSS_FAMILY_FIXED_SCORES_NOT_SAVED",
                original_N0_available_for_scope=n0 is not None,new_attempt_count=len(attempts),
                final_success=final["success"],final_candidate_id=prediction["predictions"][arm].get("selected_candidate_id"),stage_loss_class=failure)
            continue
        mode=ARM_MODE.get(arm)
        if mode is None:
            result["arms"][arm]=dict(status="NO_METHOD_BANK_MAP",final_success=final["success"]);continue
        fixed=arm in ("CTRL_CARRY","A_FIXED_BANK","B_FIXED_BANK")
        bank=carried if fixed else prediction.get("banks",{}).get(mode)
        scores=prediction.get("carried_scores",[]) if fixed else prediction.get("bank_scores",{}).get(mode)
        if bank is None or scores is None:
            result["arms"][arm]=dict(status="BANK_NOT_PRODUCED",mode=mode);continue
        if len(bank)!=len(scores):raise ValueError("Stored bank/score alignment differs")
        attempts=[] if fixed else [c for c in prediction.get("attempts",[]) if c.get("solve",{}).get("mode")==mode or c.get("source")=="NEW_"+mode]
        generated=bank_summary(carried+attempts,reference)
        eligible=set();winners={}
        for i,entry in enumerate(scores):
            for family,value in entry.items():
                energy=value.get("scores",{}).get(mode)
                if value.get("feasible",{}).get(mode) and energy is not None and math.isfinite(energy):
                    eligible.add(i);key=(float(energy),i)
                    if family not in winners or key<winners[family][0]:winners[family]=(key,i)
        admitted=bank_summary([bank[i] for i in sorted(eligible)],reference)
        winner_success={f:pose_metric(candidate_pose(bank[i]),reference)["success"] for f,(_,i) in winners.items()}
        failure="FINAL_SUCCESS" if final["success"] else "NO_SUCCESS_IN_CARRIED_OR_FINITE_NEW_ATTEMPTS" if not generated["has_success"] else "SUCCESS_LOST_BEFORE_ADMISSIBLE_BANK_OR_BY_CONSTRAINTS" if not admitted["has_success"] else "WITHIN_FAMILY_SELECTION_LOSS" if not any(winner_success.values()) else "FAMILY_OR_FINAL_SELECTION_LOSS"
        result["arms"][arm]=dict(status="DIAGNOSED_POSTSEAL_ONLY",mode=mode,fixed_bank_only=fixed,candidate_oracle_available=True,
            generated_before_feasibility=generated,retained_bank=bank_summary(bank,reference),after_constraints=admitted,
            family_winner_ids={f:bank[i].get("candidate_id") for f,(_,i) in winners.items()},family_winner_success=winner_success,
            any_family_winner_success=any(winner_success.values()),final_success=final["success"],
            final_candidate_id=prediction["predictions"][arm].get("selected_candidate_id"),stage_loss_class=failure,
            success_created_beyond_carried=generated["has_success"] and not result["carried"]["has_success"])
    return result


def stage_population(stage_rows,new_arms):
    archive=[r["original_candidate_archive_counts"] for r in stage_rows if isinstance(r["original_candidate_archive_counts"],dict)]
    keys=sorted({k for item in archive for k,v in item.items() if type(v) in (int,float)})
    result=dict(denominator=len(stage_rows),carried_oracle_successes=sum(r["carried"]["has_success"] for r in stage_rows),
        original_archive_count_known_frames=len(archive),
        original_archive_vs_positive_depth_admission_totals={k:math.fsum(v.get(k,0) for v in archive) for k in keys},arms={})
    for arm in new_arms:
        values=[r["arms"][arm] for r in stage_rows]
        generated=[v["generated_before_feasibility"]["has_success"] for v in values if "generated_before_feasibility" in v]
        admitted=[v["after_constraints"]["has_success"] for v in values if "after_constraints" in v]
        family=[v["any_family_winner_success"] for v in values if "any_family_winner_success" in v]
        result["arms"][arm]=dict(status_counts=dict(Counter(v["status"] for v in values)),
            stage_loss_class_counts=dict(Counter(v.get("stage_loss_class","NO_BANK_MAP") for v in values)),
            generated_oracle_known_frames=len(generated),after_constraints_oracle_known_frames=len(admitted),family_intermediate_known_frames=len(family),
            family_intermediate_status_counts=dict(Counter(v.get("family_intermediate_status","APPLICABLE") for v in values)),
            generated_before_constraints_oracle_successes=sum(generated) if generated else None,
            after_constraints_oracle_successes=sum(admitted) if admitted else None,
            any_family_winner_successes=sum(family) if family else None,final_successes=sum(v.get("final_success",False) for v in values))
    return result


def execution_inventory(registry,execution_status,executed):
    declared=registry.get("arms",registry)
    if isinstance(declared,list):declared={r.get("name",r.get("arm")):r for r in declared}
    if set(executed)-set(declared):raise ValueError("Unregistered executed method")
    inactive={}
    for arm,definition in declared.items():
        actual=execution_status.get(arm,{})
        if isinstance(actual,str):actual={"status":actual}
        if arm in executed:
            if actual.get("status") not in (None,"EXECUTED","DONE"):raise ValueError("Inactive status has actual predictions: "+arm)
            continue
        if actual.get("status") in ("EXECUTED","DONE"):raise ValueError("Executed status has no actual predictions: "+arm)
        merged={**definition,**actual}
        if merged.get("status") in (None,"REGISTERED"):merged["status"]="NOT_EXECUTED"
        merged.setdefault("reason","No sealed numerical output produced; full-denominator result N/A")
        inactive[arm]=merged
    return inactive


def source_status(entry):
    logged=entry.get("prediction_status")
    status=dict(logged) if isinstance(logged,dict) else {}
    for key in ("accepted","fallback","technical_failure","solver_rejected","timeout","status","reason"):
        if key not in status:status[key]=entry.get(key)
    if isinstance(logged,str):status["status"]=logged
    for key in ("accepted","fallback","technical_failure","solver_rejected","timeout"):
        if status[key] is not None and type(status[key]) is not bool:raise ValueError("Unresolved source status flag: "+key)
    return status


def _compare_pose(comp,expected,actual,path):
    for field in ("available","R_physical","centroid"):
        comp.check(expected.get(field),actual.get(field),path+"."+field)


def audit_stored_scores(prediction,baseline_poses,metadata,comp,counts):
    """Check stored terms and actual selection; no solver/model imports."""
    ident=prediction["id"];seen={}
    sets=[("carried",prediction.get("carried",[]),prediction.get("carried_scores",[]))]
    sets += [(mode,bank,prediction.get("bank_scores",{}).get(mode,[])) for mode,bank in prediction.get("banks",{}).items()]
    sets.append(("learning",prediction.get("learning_bank",[]),prediction.get("learning_scores",[])))
    tolerance=metadata.get("feasibility_tolerance",1e-7)
    consumed=set(prediction.get("boundary",{}).get("consumed_native_ids",[]))
    expected_replaced={g for g in metadata.get("active_group_ids",[]) if int(g.split(":")[1]) in consumed}
    for label,bank,scores in sets:
        if len(bank)!=len(scores):comp.fail("BANK_SCORE_LENGTH",ident+"."+label,len(bank),len(scores));continue
        for candidate,all_scores in zip(bank,scores):
            for family,values in all_scores.items():
                cid=candidate.get("candidate_id");cache=(cid,family)
                if cache in seen:
                    if values!=seen[cache]:comp.fail("SAME_POSE_SCORE_CHANGED",ident+"."+str(cache),"identical stored score record","different")
                    continue
                seen[cache]=values
                for mode,value in values.get("scores",{}).items():
                    detail=values.get("audits",{}).get(mode,{})
                    path=f"score.{ident}.{cid}.{family}.{mode}";counts["unique_mode_scores"]+=1
                    slacks=values.get("h_min_slack",{}).get(mode,[])
                    good=[float(s)>=-tolerance for s in slacks]
                    if good:comp.check(any(good),values["feasible"][mode],path+".feasible")
                    if mode in ("fixed","B","edge"):
                        parts=detail.get("parts")
                        if parts is None:counts["invalid_projection_term_archives"]+=1;continue
                        if metadata.get("prior_weight",0)!=0:
                            counts["prior_term_not_in_parts_ARCHIVE_GAP"]+=1;continue
                        expected=math.fsum(float(parts.get(k,0)) for k in ("full","partial","outline"))+float(detail.get("region",0))
                        comp.check(expected,value,path+".term_sum")
                        if mode in ("B","edge") and detail.get("B_supported") is True:
                            comp.check(0.,float(parts.get("outline",0)),path+".replaced_hull")
                            if detail.get("region",-1)<0:comp.fail("NEGATIVE_REGION_ENERGY",path,">=0",detail.get("region"))
                            counts["B_or_edge_replacement_scores"]+=1
                        continue
                    if mode not in ("A","AB"):continue
                    owner=detail.get("owner",{})
                    per_h=owner.get("per_h_total_energies");priors=owner.get("normalized_global_priors")
                    if per_h is None or priors is None:comp.fail("OWNER_TERM_ARCHIVE_MISSING",path,"per-h energies and priors",None);continue
                    if len(per_h)!=len(priors) or len(good)!=len(priors):comp.fail("OWNER_TERM_LENGTH",path,len(priors),len(per_h));continue
                    comp.check(1.,math.fsum(priors),path+".prior_sum")
                    comp.check(good,detail.get("jointly_feasible_global_h"),path+".feasible_h_mask")
                    logits=[math.log(p)-.5*e for p,e,g in zip(priors,per_h,good) if g]
                    posterior=[0.]*len(priors)
                    if logits:
                        peak=max(logits);normalizer=peak+math.log(math.fsum(math.exp(v-peak) for v in logits))
                        expected=-2*normalizer+float(detail.get("outline",0))
                        posterior=[math.exp(math.log(p)-.5*e-normalizer) if g else 0. for p,e,g in zip(priors,per_h,good)]
                    else:expected=1e20
                    comp.check(expected,value,path+".feasible_logsumexp")
                    comp.check(posterior,owner.get("global_owner_responsibilities"),path+".responsibilities")
                    for h,g in enumerate(good):
                        if not g:comp.check(0.,owner.get("global_owner_responsibilities",[])[h],path+f".infeasible_h{h}_mass")
                    group_energies=owner.get("per_h_group_energies",[])
                    if len(group_energies)==len(per_h):
                        remainder=[float(e)-math.fsum(v) for e,v in zip(per_h,group_energies)]
                        if mode=="A" or detail.get("B_supported") is not True:
                            for h,extra in enumerate(remainder):comp.check(0.,extra,path+f".no_extra_region_h{h}")
                        elif min(remainder)<-1e-7:comp.fail("NEGATIVE_INFERRED_AB_REGION",path,">=0",min(remainder))
                    active=owner.get("active_group_ids",[])
                    if any(len(v)!=len(active) for v in group_energies):comp.fail("GROUP_COUNT_DIFFERS_BY_H",path,len(active),[len(v) for v in group_energies])
                    replaced=set(detail.get("replacements",[]))
                    comp.check(sorted(replaced),owner.get("excluded_once_for_entire_arm",[]),path+".fixed_exclusions")
                    comp.check(False,owner.get("hypothesis_specific_observation_deletion"),path+".no_h_specific_deletion")
                    if mode=="AB" and detail.get("B_supported") is True:
                        comp.check(0.,float(detail.get("outline",0)),path+".replaced_hull")
                        comp.check(sorted(expected_replaced),sorted(replaced),path+".B_group_replacement")
                    else:comp.check([],sorted(replaced),path+".no_unrequested_replacement")
                    counts["owner_feasible_mixture_scores"]+=1
    for arm,entry in prediction["predictions"].items():
        path=f"selection.{ident}.{arm}";actual=entry["actual_pose"]
        if arm=="CTRL_RANSAC":
            counts["RANSAC_minimum_root_ARCHIVE_GAP"]+=1
            if entry.get("fallback") is True:_compare_pose(comp,baseline_poses["N0"],actual,path+".fallback")
            continue
        selection=(prediction.get("selection_banks") or {}).get(arm)
        if selection is not None:
            counts["learned_selection_membership_checked"]+=1
            if entry.get("none_selected") is True:
                comp.check(True,entry.get("fallback"),path+".abstention_fallback")
                _compare_pose(comp,baseline_poses["N0"],actual,path+".abstention_pose")
            else:
                allowed=set(selection.get("eligible_ids",[c["candidate_id"] for c in selection["candidates"]]))
                cid=entry.get("selected_candidate_id")
                if cid not in allowed:comp.fail("LEARNED_PICK_NOT_ELIGIBLE",path,"eligible ID",cid)
                chosen=next((c for c in selection["candidates"] if c["candidate_id"]==cid),None)
                if chosen is None:comp.fail("LEARNED_PICK_NOT_ARCHIVED",path,"candidate in pool",cid)
                else:_compare_pose(comp,candidate_pose(chosen),actual,path+".pose")
            counts["learned_all_logits_not_archived_ARGMAX_GAP"]+=1
            continue
        if arm=="CTRL_FIXED":
            valid=[c for c in prediction.get("attempts",[]) if (c.get("solve") or {}).get("start_id")=="N0_LOCAL_START" and (c.get("solve") or {}).get("feasible") is True]
            chosen=next((c for c in valid if c.get("candidate_id")==entry.get("selected_candidate_id")),None)
            if entry.get("fallback") is True:_compare_pose(comp,baseline_poses["N0"],actual,path+".fallback")
            elif chosen is None:comp.fail("FIXED_PICK_OUTSIDE_OWN_TWO_SOLVES",path,"own feasible N0_LOCAL_START attempts",entry.get("selected_candidate_id"))
            else:_compare_pose(comp,candidate_pose(chosen),actual,path+".pose")
            counts["CTRL_FIXED_cross_family_score_ARCHIVE_GAP"]+=1
            continue
        mode=ARM_MODE.get(arm)
        if mode is None:counts["unmapped_selection_ARCHIVE_GAP"]+=1;continue
        fixed=arm in ("CTRL_CARRY","A_FIXED_BANK","B_FIXED_BANK")
        bank=prediction.get("carried",[]) if fixed else prediction.get("banks",{}).get(mode,[])
        scores=prediction.get("carried_scores",[]) if fixed else prediction.get("bank_scores",{}).get(mode,[])
        choices=[]
        for i,score in enumerate(scores):
            for family,values in score.items():
                energy=values.get("scores",{}).get(mode)
                if values.get("feasible",{}).get(mode) and energy is not None and math.isfinite(energy):choices.append((float(energy),i,family))
        if not choices:
            comp.check(True,entry.get("fallback"),path+".no_eligible_fallback")
            comp.check("N0",entry.get("selected_candidate_id"),path+".fallback_ID")
            _compare_pose(comp,baseline_poses["N0"],actual,path+".fallback_pose")
        else:
            energy,index,family=min(choices)
            comp.check(False,entry.get("fallback"),path+".selected_not_fallback")
            comp.check(bank[index]["candidate_id"],entry.get("selected_candidate_id"),path+".candidate_ID")
            comp.check(family,entry.get("selected_family"),path+".family")
            comp.check(energy,entry.get("selected_energy"),path+".energy")
            _compare_pose(comp,candidate_pose(bank[index]),actual,path+".pose")
        counts["exact_common_score_minimum_selections"]+=1


class Comparison:
    def __init__(self):self.count=0;self.failures=Counter();self.examples=[];self.max_abs=Counter()

    def check(self,actual,reported,path="",*,rotation=False):
        if isinstance(actual,dict):
            if not isinstance(reported,dict):self.fail("TYPE",path,actual,reported);return
            for key,value in actual.items():
                if key not in reported:self.fail("MISSING",path+"."+key,value,None)
                else:self.check(value,reported[key],path+"."+key,rotation=rotation or key=="rotation_deg")
            return
        if isinstance(actual,list):
            if not isinstance(reported,list) or len(actual)!=len(reported):self.fail("LIST_LENGTH",path,len(actual),len(reported) if isinstance(reported,list) else None);return
            for i,(a,b) in enumerate(zip(actual,reported)):self.check(a,b,path+f"[{i}]",rotation=rotation)
            return
        self.count+=1
        if isinstance(actual,(float,np.floating)) and isinstance(reported,(float,int)):
            difference=abs(float(actual)-float(reported));field="rotation" if rotation else "other"
            self.max_abs[field]=max(self.max_abs[field],difference)
            tolerance=(1e-5 if rotation else 5e-9)+1e-9*max(abs(float(actual)),abs(float(reported)))
            if difference>tolerance:self.fail("NUMERIC",path,actual,reported)
        elif actual!=reported:self.fail("VALUE",path,actual,reported)

    def fail(self,kind,path,actual,reported):
        self.failures[kind]+=1
        if len(self.examples)<30:self.examples.append(dict(kind=kind,path=path,independent=actual,reported=reported))


def rounding_direction_audit(scored,reported,reported_rows,*,previous_failed_check_dir=None):
    """Account for every representation-sensitive strict direction count.

    No tolerance changes a sign. Both signs are preserved. Every replay value
    must equal the sealed evaluator value exactly, and aggregate count changes
    must equal the sum of explicit per-row sign transitions. The discrepancy
    envelope is numerical expression disagreement, not physical uncertainty.
    """
    sign=lambda value: -1 if value<0 else 1 if value>0 else 0
    switches=[];expression_mismatches=[];value_count=0
    maximum={k:0. for k in ("translation_cm","rotation_deg")}
    new_arms=[a for a in scored[0]["metrics"] if a not in BASELINES]
    for row in scored:
        for arm,stable in row["metrics"].items():
            for metric in maximum:
                replay=row["strict_direction_metrics"][arm][metric]
                target=reported_rows[row["id"]]["metrics"][arm][metric]
                value_count+=1
                if replay!=target:expression_mismatches.append(dict(id=row["id"],arm=arm,metric=metric,replay=replay,reported=target))
                if replay is not None and stable[metric] is not None:maximum[metric]=max(maximum[metric],abs(stable[metric]-replay))
        for after in new_arms:
            for before in COMPARATORS:
                for metric in maximum:
                    a,b=(row["metrics"][arm][metric] for arm in (before,after))
                    x,y=(row["strict_direction_metrics"][arm][metric] for arm in (before,after))
                    if any(v is None for v in (a,b,x,y)):continue
                    stable_delta,replay_delta=b-a,y-x
                    if sign(stable_delta)==sign(replay_delta):continue
                    envelope=abs(a-x)+abs(b-y)
                    outward_envelope=math.nextafter(envelope,math.inf)
                    assert abs(stable_delta-replay_delta)<=outward_envelope,"Sign transition exceeds expression-disagreement envelope"
                    assert min(stable_delta,replay_delta)<=0<=max(stable_delta,replay_delta),"Sign transition does not include zero"
                    pa,pb=(row["poses"][arm] for arm in (before,after))
                    switches.append(dict(id=row["id"],partition=row["partition"],before=before,after=after,metric=metric,
                        stable_before=a,stable_after=b,original_expression_before=x,original_expression_after=y,
                        stable_delta=stable_delta,original_expression_delta=replay_delta,
                        stable_sign=sign(stable_delta),original_expression_sign=sign(replay_delta),
                        value_hex=dict(stable_before=a.hex(),stable_after=b.hex(),expression_before=x.hex(),expression_after=y.hex()),
                        expression_disagreement_envelope=envelope,
                        center_max_component_difference_m=float(np.max(np.abs(np.asarray(pa["centroid"])-np.asarray(pb["centroid"])))),
                        rotation_max_component_difference=float(np.max(np.abs(np.asarray(pa["R_physical"])-np.asarray(pb["R_physical"]))))))
    assert not expression_mismatches,"Secondary expression does not exactly reproduce sealed metric columns"
    previous=None;previous_detail=None;previous_bindings={}
    if previous_failed_check_dir is not None:
        folder=Path(previous_failed_check_dir);receipt_path=folder/"INDEPENDENT_CHECK.json"
        previous=read(receipt_path)
        assert previous["status"]=="FAIL","Previous attempt must remain explicitly FAIL"
        detail_path=folder/previous["recomputed_detail"]["file"]
        assert sha(detail_path)==previous["recomputed_detail"]["sha256"],"Previous failed recomputation changed"
        snapshot=folder/"independent_check_source.py"
        assert sha(snapshot)==previous["source_sha256"],"Previous failed checker source not preserved"
        for path,digest in previous["input_bindings"].items():assert sha(path)==digest,"Previously checked prediction/reference/evaluation bytes changed"
        with gzip.open(detail_path,"rt") as f:previous_detail=json.load(f)
        previous_bindings={str(p):sha(p) for p in (receipt_path,detail_path,snapshot)}
    changed=[];previous_paths=set();current_paths=set();count_fields_verified=0
    for population,parts in POPULATIONS.items():
        if population=="ALL238":continue
        subset=[r for r in scored if r["partition"] in parts]
        for after in new_arms:
            for before in COMPARATORS:
                contrast=after+"_minus_"+before;target=reported["paired"][population][contrast]
                for metric in maximum:
                    for scope in ("all_common_available","both_failure"):
                        selected=[r for r in subset if scope!="both_failure" or
                                  (not r["metrics"][before]["success"] and not r["metrics"][after]["success"])]
                        stable_counts=Counter();expression_counts=Counter()
                        for r in selected:
                            a,b=(r["metrics"][arm][metric] for arm in (before,after))
                            x,y=(r["strict_direction_metrics"][arm][metric] for arm in (before,after))
                            if any(v is None for v in (a,b,x,y)):continue
                            stable_counts[sign(b-a)]+=1;expression_counts[sign(y-x)]+=1
                        relevant=[v for v in switches if v["id"] in {r["id"] for r in selected} and
                                  v["before"]==before and v["after"]==after and v["metric"]==metric]
                        mapping={"improved":-1,"worsened":1} if scope=="all_common_available" else {"improved_n":-1,"worsened_n":1,"tied_n":0}
                        for field,direction in mapping.items():
                            section="deltas" if scope=="all_common_available" else "both_failure_continuous"
                            path=f"paired.{population}.{contrast}.{section}.{metric}.{field}"
                            delta=sum(int(v["original_expression_sign"]==direction)-int(v["stable_sign"]==direction) for v in relevant)
                            stable_count,expression_count=stable_counts[direction],expression_counts[direction]
                            assert expression_count==stable_count+delta,"Per-row sign effects do not reconstruct strict count"
                            assert expression_count==target[section][metric][field],"Independently replayed strict count disagrees with sealed report"
                            count_fields_verified+=1
                            if previous_detail is not None:
                                old_count=previous_detail["paired"][population][contrast][section][metric][field]
                                assert old_count==stable_count,"Independent stable metric direction changed during repair"
                                if old_count!=target[section][metric][field]:previous_paths.add(path)
                            if stable_count!=target[section][metric][field]:
                                current_paths.add(path)
                                changed.append(dict(path=path,stable_count=stable_count,original_expression_count=expression_count,
                                    unchanged_reported_count=target[section][metric][field],net_effect_of_listed_sign_switches=delta,
                                    sign_switch_ids=[v["id"] for v in relevant]))
    if previous is not None:
        assert previous_paths==current_paths,"Previous failed count fields are not exactly the accounted sign-sensitive fields"
        assert len(previous_paths)==previous["failure_count"],"A prior failure other than these strict-direction fields is unexplained"
        assert all(v["path"] in previous_paths for v in previous["failure_examples"]),"Prior failure example not explained"
    return dict(schema="STRICT_ZERO_DIRECTION_ROUNDOFF_AUDIT_V1",status="PASS",
        stable_metric="independent atan2 C2 rotation + fsum translation, unchanged for joint success/magnitudes/distributions/CIs",
        strict_direction_verifier="separate original norm/acos/Y(pi) expression computed from raw poses and fixed reference; evaluator not imported",
        original_expression_value_checks=value_count,original_expression_values_not_bitwise_equal_to_report=0,
        maximum_stable_vs_original_expression_metric_difference=maximum,
        strict_direction_count_fields_verified=count_fields_verified,representation_sensitive_count_fields=len(changed),
        previous_failed_count_fields_exactly_explained=len(previous_paths) if previous is not None else None,
        sign_switch_pair_metric_rows=len(switches),unique_sign_switch_frame_count=len({v["id"] for v in switches}),
        all_count_changes_equal_explicit_per_row_sign_effects=True,zero_tolerance_changed=False,
        predictions_references_models_success_criteria_weights_or_reported_numbers_changed=False,
        limitation="Strict zero-direction counts depend on the floating-point expression at near-zero deltas. The envelope is expression disagreement, not calibrated physical uncertainty; these microscopic signs do not establish material improvement or damage.",
        previous_failure_artifacts=previous_bindings,changed_count_fields=changed,sign_switch_rows=switches,
        source_sha256=sha(__file__))


def run(run_dir,*,baseline_folder,baseline_comparison_folder,evaluation_dir,output_dir,checking_GO=False,
        expected_plan_sha256=None,expected_prediction_seal_sha256=None,previous_failed_check_dir=None):
    if checking_GO is not True:raise PermissionError("Root final postseal checking GO required before any file read")
    started,cpu=time.perf_counter(),time.process_time()
    run_dir,baseline_folder,baseline_comparison_folder,evaluation_dir,output_dir=map(Path,
        (run_dir,baseline_folder,baseline_comparison_folder,evaluation_dir,output_dir))
    plan=run_dir/"PLAN_LOCK.json";prediction_seal_path=run_dir/"PREDICTION_SEAL.json"
    ps=read(prediction_seal_path);es=read(evaluation_dir/"EVALUATION_SEAL.json")
    if expected_plan_sha256 is not None and sha(plan)!=expected_plan_sha256:raise ValueError("Frozen plan changed")
    if expected_prediction_seal_sha256 is not None and sha(prediction_seal_path)!=expected_prediction_seal_sha256:raise ValueError("Final prediction seal changed")
    if ps.get("before_scoring") is not True or ps.get("reference_inputs") is not False or ps.get("complete") is not True or ps.get("plan_lock_sha256")!=sha(plan):raise ValueError("Final reference-free prediction seal incomplete")
    registry_path=run_dir/"ARM_REGISTRY.json";execution_path=run_dir/"ARM_EXECUTION_STATUS.json"
    if read(plan).get("arm_registry_sha256")!=sha(registry_path):raise ValueError("Frozen arm registry changed")
    prediction_path=run_dir/ps.get("prediction_filename","PREDICTIONS.jsonl.gz")
    if sha(prediction_path)!=ps["predictions_sha256"]:raise ValueError("Final predictions changed")
    batch_seal_path=run_dir/"real_batch_all_available_v1/BATCH_SEAL.json"
    batch_seal=read(batch_seal_path)
    inputs_path=run_dir/"inputs_all_available_v1/PREDICTION_INPUTS238.jsonl.gz"
    if ps.get("batch_seal_sha256")!=sha(batch_seal_path) or batch_seal.get("input_sha256")!=sha(inputs_path):
        raise ValueError("Final common-score/input bank binding changed")
    metadata={}
    for f in rows(inputs_path):
        packet=f["packet"];activation=packet["activation"];cfg=packet["solver_input"]["config"]
        metadata[f["id"]]=dict(prior_weight=cfg.get("prior_weight",0),feasibility_tolerance=cfg.get("feasibility_tolerance",1e-7),
            active_group_ids=[f"full:{k}" for k in activation["complete_ids"]]+[f"partial:{k}" for k in activation["all_partial_ids"]])
    if es.get("explicit_root_scoring_GO") is not True or es["input_bindings"]["new_prediction_seal"]!=sha(prediction_seal_path):raise ValueError("Evaluation not bound to final predictions")
    old_seal_path=baseline_comparison_folder/"EVALUATION_SEAL.json";old_seal=read(old_seal_path)
    old_pose_path=baseline_comparison_folder/"FAIR_SCORED_ROWS.json"
    reference_path=baseline_folder/"FIXED_REFERENCES.json";baseline_seal_path=baseline_folder/"SCORE_SEAL.json"
    if sha(old_pose_path)!=old_seal["artifacts"][old_pose_path.name]:raise ValueError("Old six-arm actual poses changed")
    if es["input_bindings"].get("old_six_pose_ledger")!=sha(old_pose_path) or es["input_bindings"].get("old_evaluation_seal")!=sha(old_seal_path):
        raise ValueError("Evaluation is not bound to the inspected six-arm cache")
    reference_sha=sha(reference_path)
    if (reference_sha!=read(baseline_seal_path)["artifacts"][reference_path.name] or
        reference_sha!=old_seal["input_bindings"]["original"][reference_path.name] or
        reference_sha!=es["input_bindings"]["fixed_references"]):raise ValueError("Original reference seal mismatch")
    for name in ("RESULTS.json","SCORED_ROWS.jsonl.gz","CANDIDATE_STAGE_DIAGNOSIS.json","COMPARISON_ALL_ARMS.csv"):
        if sha(evaluation_dir/name)!=es["artifacts"][name]:raise ValueError("Evaluation artifact changed: "+name)
    # All reference-free prediction and original-reference seals are verified
    # above. This is the first decoding of any reference pose in the checker.
    refs=read(reference_path)["references"]
    old={r["id"]:r for r in read(old_pose_path)["rows"]}
    reported_rows={r["id"]:r for r in rows(evaluation_dir/"SCORED_ROWS.jsonl.gz")}
    reported=read(evaluation_dir/"RESULTS.json")
    stage_document=read(evaluation_dir/"CANDIDATE_STAGE_DIAGNOSIS.json")
    reported_stages={r["id"]:r for r in stage_document["rows"]}
    comp=Comparison();score_comp=Comparison();score_counts=Counter();scored=[];stages=[];new_arms=None;seen=[]
    for prediction in rows(prediction_path):
        ident=prediction["id"];seen.append(ident)
        if new_arms is None:new_arms=tuple(prediction["predictions"])
        if set(new_arms)!=set(prediction["predictions"]):raise ValueError("New arm lost on a frame")
        identity={k:prediction[k] for k in ("id","partition","session","cluster_id")}
        source=old[ident];reference=refs[prediction["partition"]][ident]
        comp.check(identity,{k:source[k] for k in identity},"identity_old."+ident)
        comp.check(identity,{k:reported_rows[ident][k] for k in identity},"identity_eval."+ident)
        metrics={a:pose_metric(source["poses"][a],reference) for a in BASELINES}
        metrics.update({a:pose_metric(e["actual_pose"],reference) for a,e in prediction["predictions"].items()})
        poses={a:source["poses"][a] for a in BASELINES}
        poses.update({a:e["actual_pose"] for a,e in prediction["predictions"].items()})
        strict_metrics={a:strict_direction_values(p,reference) for a,p in poses.items()}
        statuses={a:source["status"][a] for a in BASELINES}
        statuses.update({a:source_status(e) for a,e in prediction["predictions"].items()})
        value={**identity,"metrics":metrics,"strict_direction_metrics":strict_metrics,"poses":poses,"status":statuses};scored.append(value)
        for arm,m in metrics.items():comp.check(m,reported_rows[ident]["metrics"][arm],"row."+ident+"."+arm)
        for arm,s in statuses.items():comp.check(s,reported_rows[ident]["status"][arm],"status."+ident+"."+arm)
        audit_stored_scores(prediction,source["poses"],metadata[ident],score_comp,score_counts)
        stage=stage_summary(prediction,reference,new_arms,baseline_poses=source["poses"])
        comp.check(stage["carried"],reported_stages[ident]["carried"],"stage.carried."+ident)
        comp.check(stage["new_attempt_count"],reported_stages[ident]["new_attempt_count"],"stage.attempt_count."+ident)
        for key in ("original_candidate_archive_counts","carried_unique_count"):
            comp.check(stage[key],reported_stages[ident][key],"stage."+key+"."+ident)
        for arm,v in stage["arms"].items():comp.check(v,reported_stages[ident]["arms"][arm],"stage."+ident+"."+arm)
        stages.append({**identity,**stage})
    if seen!=ps["ordered_ids"] or len(seen)!=238 or len(set(seen))!=238:raise ValueError("Exact sealed238 order differs")
    if Counter(r["partition"] for r in scored)!=Counter(CAL=46,TEST=49,REMAINDER=143):raise ValueError("Full population denominators differ")
    if set(new_arms)&set(BASELINES):raise ValueError("New arms overwrite old comparators")
    inactive=execution_inventory(read(registry_path),read(execution_path),new_arms)
    comp.check(inactive,reported["unexecuted_arms"],"unexecuted_arms")
    if set(inactive)!=set(reported["unexecuted_arms"]):comp.fail("UNEXECUTED_INVENTORY","unexecuted_arms",sorted(inactive),sorted(reported["unexecuted_arms"]))
    arms=BASELINES+new_arms;populations={};contrasts={};stage_populations={}
    for name,parts in POPULATIONS.items():
        subset=[r for r in scored if r["partition"] in parts]
        if len(subset)!=EXPECTED_N[name]:raise ValueError("Incorrect independent population size")
        populations[name]=aggregate(subset,arms)
        contrasts[name]={after+"_minus_"+before:paired(subset,before,after,direction_view="original_expression") for after in new_arms for before in COMPARATORS}
        stage_populations[name]=stage_population([r for r in stages if r["partition"] in parts],new_arms)
        if name!="ALL238":
            comp.check(populations[name],reported["populations"][name],"population."+name)
            comp.check(contrasts[name],reported["paired"][name],"paired."+name)
            comp.check(stage_populations[name],stage_document["populations"][name],"stage_population."+name)
            for session in sorted({r["session"] for r in subset}):
                comp.check(aggregate([r for r in subset if r["session"]==session],arms),
                    reported["session_summaries"][name][session],"session."+name+"."+session)
    with (evaluation_dir/"COMPARISON_ALL_ARMS.csv").open(newline="") as f:
        comparison_csv={(r["population"],r["arm"]):r for r in csv.DictReader(f)}
    expected_csv={(population,arm) for population in POPULATIONS if population!="ALL238" for arm in (*arms,*inactive)}
    if set(comparison_csv)!=expected_csv:comp.fail("COMPARISON_CSV_INVENTORY","csv",len(expected_csv),len(comparison_csv))
    for (population,arm),row in comparison_csv.items():
        if population not in populations:continue
        comp.check(EXPECTED_N[population],int(row["denominator"]),f"csv.{population}.{arm}.denominator")
        if arm in inactive:
            comp.check(inactive[arm]["status"],row["execution_status"],f"csv.{population}.{arm}.execution_status")
            for key in ("successes","success_rate","no_pose","translation_cm_mean","rotation_deg_mean"):
                comp.check("",row.get(key,""),f"csv.{population}.{arm}.NA.{key}")
            continue
        value=populations[population][arm]
        comp.check("EXECUTED",row["execution_status"],f"csv.{population}.{arm}.execution_status")
        for key in ("successes","success_rate","no_pose","T_gt50cm","T_gt100cm","R_gt45deg"):
            parsed=float(row[key]) if key=="success_rate" else int(row[key])
            comp.check(value[key],parsed,f"csv.{population}.{arm}.{key}")
        for metric_name in ("translation_cm","rotation_deg"):
            for key in ("n","mean","sample_std","sample_variance","median","P90","P95","max"):
                parsed=None if row.get(metric_name+"_"+key) in (None,"") else float(row[metric_name+"_"+key])
                comp.check(value[metric_name][key],parsed,f"csv.{population}.{arm}.{metric_name}.{key}",rotation=metric_name=="rotation_deg")
    if previous_failed_check_dir is None and (run_dir/"independent_v1/INDEPENDENT_CHECK.json").is_file():
        previous_failed_check_dir=run_dir/"independent_v1"
    rounding_audit=rounding_direction_audit(scored,reported,reported_rows,previous_failed_check_dir=previous_failed_check_dir)
    output_dir.mkdir(parents=True,exist_ok=False)
    rounding_path=output_dir/"ROUNDING_DIRECTION_AUDIT.json";write(rounding_path,rounding_audit)
    score_report=dict(schema="INDEPENDENT_STORED_SCORE_SELECTION_AUDIT_V1",status="PASS" if not score_comp.failures else "FAIL",
        failure_count=sum(score_comp.failures.values()),failure_counts=dict(score_comp.failures),failure_examples=score_comp.examples,
        scalar_checks=score_comp.count,max_absolute_differences=dict(score_comp.max_abs),counts=dict(score_counts),
        limitations=["CTRL_FIXED full cross-family scores were not saved: verify only own two-solve membership and pose identity",
                     "RANSAC complete roots were not saved: minimum-root oracle/selection audit unavailable",
                     "Learned complete logits were not saved: verify eligible-pool/abstention membership and pose identity; argmax audit unavailable"],
        method_and_solver_imports=0,new_solves=0,new_RGB_queries=0,source_sha256=sha(__file__),
        input_bindings={str(p):sha(p) for p in (prediction_seal_path,prediction_path,batch_seal_path,inputs_path)})
    score_report_path=output_dir/"PREDICTION_SCORE_AUDIT.json";write(score_report_path,score_report)
    detail_path=output_dir/"INDEPENDENT_RECOMPUTED.json.gz"
    with gzip.open(detail_path,"wt") as f:
        json.dump(dict(populations=populations,paired=contrasts,scored_rows=scored,stage_rows=stages,
                       stage_populations=stage_populations,unexecuted_arms=inactive),f,separators=(",",":"),allow_nan=False)
    failures={**{"arithmetic:"+k:v for k,v in comp.failures.items()},**{"stored_scores:"+k:v for k,v in score_comp.failures.items()}}
    result=dict(schema="INDEPENDENT_ALL_ARMS_PHYSICAL_POSE_CHECK_V1",status="PASS" if not failures else "FAIL",
        failure_count=sum(failures.values()),failure_counts=failures,failure_examples=comp.examples+score_comp.examples,
        scalar_checks=comp.count,max_absolute_differences=dict(comp.max_abs),
        arms=list(arms),new_arms=list(new_arms),population_denominators=EXPECTED_N,
        declared_new_arm_count=len(new_arms)+len(inactive),executed_new_arm_count=len(new_arms),unexecuted_arms=inactive,
        independently_recomputed_successes={name:{a:s["successes"] for a,s in data.items()} for name,data in populations.items()},
        unit_tests=self_check(),same_pose_joint_criterion=True,symmetry="I and physical Y180 only; rectangular Y90 excluded",
        missing_pose_stays_failure_in_full_denominator=True,all_fitted_seeds_kept_as_methods_not_independent_images=True,
        stats="own fsum sample moments and sorted linear quantiles; no trimming",
        bootstrap=dict(seed=SEED,draws=DRAWS,schedule="NumPy default_rng multinomial, reset per contrast; own grouped sums and own linear CI quantiles",
                       units=["original cluster_id","named session sensitivity"],paired_frame_weighted=True),
        stage_checks="all stored carried, retained-bank and finite/failed-attempt poses scored independently; source score fields used only to reproduce postseal eligibility/selection diagnosis",
        reference_decoded_only_after_prediction_and_reference_seals=True,
        input_bindings={str(p):sha(p) for p in (plan,registry_path,execution_path,prediction_seal_path,prediction_path,batch_seal_path,inputs_path,old_seal_path,old_pose_path,baseline_seal_path,reference_path,
                          evaluation_dir/"EVALUATION_SEAL.json",evaluation_dir/"RESULTS.json",evaluation_dir/"SCORED_ROWS.jsonl.gz",evaluation_dir/"CANDIDATE_STAGE_DIAGNOSIS.json",evaluation_dir/"COMPARISON_ALL_ARMS.csv")},
        source_sha256=sha(__file__),recomputed_detail=dict(file=detail_path.name,sha256=sha(detail_path),bytes=detail_path.stat().st_size),
        stored_score_audit=dict(file=score_report_path.name,sha256=sha(score_report_path),status=score_report["status"],
                               scalar_checks=score_report["scalar_checks"],counts=score_report["counts"]),
        rounding_direction_audit=dict(file=rounding_path.name,sha256=sha(rounding_path),status=rounding_audit["status"],
            previous_failed_count_fields_exactly_explained=rounding_audit["previous_failed_count_fields_exactly_explained"],
            sign_switch_pair_metric_rows=rounding_audit["sign_switch_pair_metric_rows"],
            numerical_direction_policy="strict sign independently replayed under original expression; stable atan2/fsum magnitudes and CIs unchanged"),
        wall_seconds=time.perf_counter()-started,process_cpu_seconds=time.process_time()-cpu,process_peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    write(output_dir/"INDEPENDENT_CHECK.json",result)
    return result


def self_check():
    reference=dict(order=2,R=np.eye(3).tolist(),t=[0,0,1.])
    def pose(angle,center):
        c,s=math.cos(angle),math.sin(angle)
        return dict(available=True,R_physical=[[c,0,s],[0,1,0],[-s,0,c]],centroid=center)
    assert pose_metric(pose(math.pi,[0,0,1]),reference)["success"]
    assert not pose_metric(pose(math.pi/2,[0,0,1]),reference)["success"]
    assert not pose_metric(pose(0,[.06,0,1]),reference)["success"]
    assert not pose_metric(pose(math.radians(6),[0,0,1]),reference)["success"]
    assert pose_metric(pose(math.radians(5),[.05,0,1]),reference)["success"]
    absent=pose_metric(dict(available=False),reference)
    assert absent["translation_cm"] is None and not absent["success"]
    stats=distribution([1.,2.,3.,None],4)
    assert stats["n"]==3 and stats["missing"]==1 and stats["sample_variance"]==1 and stats["median"]==2
    assert abs(stats["P95"]-2.9)<1e-12
    # Paired values have unequal recording sizes: resample clusters and divide
    # by resampled frame counts; never average unweighted cluster means.
    r=[dict(cluster_id="a",session="s1"),dict(cluster_id="a",session="s1"),dict(cluster_id="b",session="s2")]
    ci=cluster_ci([1.,1.,-1.],r,"cluster_id")
    assert ci["CI95"]==[-1.,1.] and ci["valid_resamples"]==DRAWS
    good=dict(candidate_id="good",**{k:v for k,v in pose(0,[0,0,1]).items() if k!="available"})
    bad=dict(candidate_id="bad",**{k:v for k,v in pose(math.radians(20),[0,0,1]).items() if k!="available"})
    attempt={**bad,"solve":dict(start_id="N0_LOCAL_START",mode="fixed",feasible=True)}
    fixed_row=dict(carried=[good],banks={"fixed":[good]},attempts=[attempt],
        predictions={"CTRL_FIXED":dict(actual_pose=candidate_pose(bad),selected_candidate_id="bad")})
    fixed=stage_summary(fixed_row,reference,["CTRL_FIXED"],baseline_poses={"N0":candidate_pose(bad)})["arms"]["CTRL_FIXED"]
    assert not fixed["generated_before_feasibility"]["has_success"] and fixed["new_attempt_count"]==1
    ransac=stage_summary(dict(carried=[],predictions={"CTRL_RANSAC":dict(actual_pose=candidate_pose(good))}),reference,["CTRL_RANSAC"])["arms"]["CTRL_RANSAC"]
    assert ransac["status"]=="ARCHIVE_GAP" and "generated_before_feasibility" not in ransac
    learned=stage_summary(dict(carried=[],selection_banks={"L":dict(candidates=[good,bad],eligible_ids=["bad"])},
        predictions={"L":dict(actual_pose=candidate_pose(bad),selected_candidate_id="bad")}),reference,["L"])["arms"]["L"]
    assert learned["stage_loss_class"]=="LEARNED_ELIGIBILITY_FILTER_LOSS" and learned["family_intermediate_status"]=="NOT_APPLICABLE_DIRECT_SOFTMAX_SELECTION"
    unavailable=execution_inventory({"arms":[dict(name="L",status="REGISTERED")]},{"L":dict(status="NOT_APPLICABLE",reason="synthetic support gate")},[])
    assert unavailable["L"]["status"]=="NOT_APPLICABLE"
    energies=[2.,4.,6.,8.];prior=[.25]*4;mask=[True,False,True,False]
    z=.25*math.exp(-1)+.25*math.exp(-3);mixture=-2*math.log(z)+3
    posterior=[.25*math.exp(-1)/z,0.,.25*math.exp(-3)/z,0.]
    owner=dict(per_h_total_energies=energies,normalized_global_priors=prior,global_owner_responsibilities=posterior,
        per_h_group_energies=[[e] for e in energies],active_group_ids=["full:0"],excluded_once_for_entire_arm=[],hypothesis_specific_observation_deletion=False)
    scores={"F":dict(scores={"A":mixture},feasible={"A":True},h_min_slack={"A":[0.,-1.,0.,-1.]},
        audits={"A":dict(owner=owner,outline=3.,B_supported=False,jointly_feasible_global_h=mask,replacements=[])})}
    fixture=dict(id="synthetic-score",carried=[good],carried_scores=[scores],predictions={"A_FIXED_BANK":dict(actual_pose=candidate_pose(good),
        selected_candidate_id="good",selected_family="F",selected_energy=mixture,fallback=False)})
    comparison=Comparison();audit_stored_scores(fixture,{"N0":candidate_pose(good)},dict(active_group_ids=["full:0"]),comparison,Counter())
    assert not comparison.failures
    return dict(C2_180_accepted=True,rectangle_90_rejected=True,same_pose_joint_success=True,
                unavailable_full_denominator=True,sample_moments_and_linear_quantiles=True,cluster_frame_weighted_ratio=True,
                CTRL_FIXED_own_two_solve_scope=True,RANSAC_archive_gap_not_oracle_zero=True,
                learned_pool_eligibility_and_family_NA=True,actual_execution_NA_merge=True,
                feasible_owner_logsumexp_without_prior_renormalization=True,saved_common_score_selection_and_pose_identity=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--self-check",action="store_true")
    p.add_argument("--run-dir");p.add_argument("--baseline-folder");p.add_argument("--baseline-comparison-folder")
    p.add_argument("--evaluation-dir");p.add_argument("--output-dir");p.add_argument("--checking-GO",action="store_true")
    p.add_argument("--expected-plan-sha256");p.add_argument("--expected-prediction-seal-sha256")
    p.add_argument("--previous-failed-check-dir")
    a=p.parse_args()
    if a.self_check:print(json.dumps(self_check(),indent=2))
    else:
        r=run(a.run_dir,baseline_folder=a.baseline_folder,baseline_comparison_folder=a.baseline_comparison_folder,
              evaluation_dir=a.evaluation_dir,output_dir=a.output_dir,checking_GO=a.checking_GO,
              expected_plan_sha256=a.expected_plan_sha256,expected_prediction_seal_sha256=a.expected_prediction_seal_sha256,
              previous_failed_check_dir=a.previous_failed_check_dir)
        print(json.dumps({k:r[k] for k in ("status","failure_count","failure_counts","scalar_checks","max_absolute_differences","wall_seconds")},indent=2))
