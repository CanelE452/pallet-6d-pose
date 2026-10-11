"""Independent sealed-row supplement, donor, preservation and disclosure audit.

No fitting, neural model import, RGB read or pose solve is performed. Producer
statistics functions are not imported. Existing files are never rewritten.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from fractions import Fraction
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

import numpy as np


PRIMARY=[("G_V_MLP","G_MLP"),("G_V_D_MLP","G_V_MLP"),
         ("D_JOINT_DIM","D_CONTROL"),("G_V_D_MLP","OFFICIAL")]
C_PAIRS=[(method,"OFFICIAL") for method in ["TRAIN_PRIOR","D_LOGISTIC",
         "G_LOGISTIC","V_LOGISTIC","V_MLP","G_MLP","G_V_MLP",
         "G_V_D_MLP","V_SHUFFLED_LOGISTIC"]]+PRIMARY[:2]
D_PAIRS=[]
for suffix,n3 in [("","N3_DIM_SYM"),("_SUBPIX","N3_THEN_SUBPIX")]:
    D_PAIRS.extend([(n3,"OLD_P"+suffix),("D_CONTROL"+suffix,"OLD_P"+suffix),
                    ("D_JOINT_DIM"+suffix,"OLD_P"+suffix),
                    ("D_JOINT_DIM"+suffix,"D_CONTROL"+suffix),
                    ("D_CONTROL"+suffix,n3),("D_JOINT_DIM"+suffix,n3)])
D_PAIRS.extend([("D_JOINT_MEAN","D_JOINT_DIM"),("D_JOINT_SHUFFLE","D_JOINT_DIM")])
FORBIDDEN={"manual_points","manual_points8","gt_points","reference_points",
           "evaluation_reference_points","q_ref","qref","R_gt","t_gt",
           "reference_R_cf","reference_bottom_center","reference_raw_rgb"}


def read(path):return json.loads(Path(path).read_text())


def rows(path):
    with gzip.open(path,"rt",encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda:handle.read(4*1024*1024),b""):digest.update(block)
    return digest.hexdigest()


def quantile(values,q):
    ordered=sorted(values);position=(len(ordered)-1)*q
    a=math.floor(position);b=math.ceil(position)
    return ordered[a] if a==b else ordered[a]*(b-position)+ordered[b]*(position-a)


def moments(values):
    values=[float(value) for value in values if value is not None and math.isfinite(value)]
    mean=math.fsum(values)/len(values) if values else None
    variance=math.fsum((value-mean)**2 for value in values)/(len(values)-1) if len(values)>1 else None
    return dict(n=len(values),mean=mean,variance=variance,SD=math.sqrt(variance) if variance is not None else None,
                median=quantile(values,.5) if values else None,P90=quantile(values,.9) if values else None,
                P95=quantile(values,.95) if values else None,maximum=max(values) if values else None)


class Comparator:
    def __init__(self):self.count=0;self.maximum=0.
    def check(self,a,b,tolerance=1e-8):
        self.count+=1
        if a is None or b is None:assert a is None and b is None,(a,b);return
        delta=abs(float(a)-float(b));self.maximum=max(self.maximum,delta)
        assert delta<=tolerance,(a,b,delta)
    def values(self,values,stored):
        expected=moments(values)
        for key,value in stored.items():
            if key in expected:self.check(expected[key],value)


def state(before,after):
    if before is None or after is None:return "UNCLEAR"
    return "RECOVERY" if before and not after else "DAMAGE" if not before and after else "UNCHANGED"


def identity(row):return row["method"],row["id"],int(row["seed"])


def binary_pose(row):
    pose=row["pose"];available=int(bool(pose["available"]))
    success=int(available and pose["translation_cm"]<5 and pose["rotation_deg"]<5)
    confusion=int(available and pose["rotation_deg"]>45 and abs(pose["yaw_deg"])>=60)
    return available,success,confusion


def verify_primary(doc,frame,ids,sessions,draw,check):
    requested=read(doc/"REQUESTED_PRIMARY_PAIRED.json")
    assert requested["status"]=="POST_EXECUTION_PROTOCOL_ALIGNMENT_EXPLORATORY"
    assert requested["family_size"]==8 and "after execution" in requested["deviation"]
    expected=[(a,b,key) for a,b in PRIMARY for key in ["wd_confusion","pose_success"]]
    assert [(r["after"],r["before"],r["metric"]) for r in requested["results"]]==expected
    computed=[]
    for row in requested["results"]:
        seeds=[1] if row["after"]=="D_JOINT_DIM" else [1,2,3]
        totals=[];counts=[]
        for session in sessions:
            selected=[fid for fid in ids if fid.split(":")[0]==session]
            totals.append(sum(frame[row["after"],fid,seed][row["metric"]]-frame[row["before"],fid,seed][row["metric"]]
                              for fid in selected for seed in seeds))
            counts.append(len(selected)*len(seeds))
        numerator=sum(totals);denominator=sum(counts);effect=Fraction(numerator,denominator)
        sampled=[];extreme=0
        for indices in draw:
            result=Fraction(sum(totals[int(index)] for index in indices),sum(counts[int(index)] for index in indices))
            sampled.append(float(100*result));extreme+=int(abs(result-effect)>=abs(effect))
        p=Fraction(extreme+1,len(draw)+1)
        assert row["numerator"]==numerator and row["denominator"]==denominator
        assert row["frames"]==173 and row["seeds"]==len(seeds) and row["sessions"]==sessions
        check.check(float(100*effect),row["effect_pp"])
        check.check(float(p),row["p_raw"],1e-15)
        ci=[quantile(sampled,.025),quantile(sampled,.975)]
        for a,b in zip(ci,row["CI95"]):check.check(a,b)
        equal=math.fsum(float(Fraction(t,n)*100) for t,n in zip(totals,counts))/9
        check.check(equal,row["session_equal_effect_pp"])
        computed.append(dict(after=row["after"],before=row["before"],metric=row["metric"],effect_pp=float(effect*100),
                             CI95=ci,extreme_draws=extreme,p=p,seeds=len(seeds)))
    order=sorted(range(8),key=lambda i:(computed[i]["p"],i));running=Fraction(0)
    for rank,index in enumerate(order):
        running=max(running,min(Fraction(1),computed[index]["p"]*(8-rank)))
        check.check(float(running),requested["results"][index]["p_Holm"],1e-15)
        computed[index]["p_Holm"]=float(running)
    for row in computed:row["p_raw"]=float(row.pop("p"))
    # Verify all existing Stage1 mean/SD/quantile/CI columns from raw seed1 rows.
    stage_summaries=[row for row in read(doc/"METRICS.json")["summaries"]
                     if row["population"]=="REAL173" and row["backbone"]=="YOLO" and row["method"] in {"D_CONTROL","D_JOINT_DIM"}]
    assert len(stage_summaries)==2
    for summary in stage_summaries:
        assert summary["frames"]==173 and summary["seeds"]==[1]
        for field,stored in summary["metrics"].items():
            values={fid:frame[summary["method"],fid,1][field] for fid in ids if frame[summary["method"],fid,1][field] is not None}
            check.values(list(values.values()),stored)
            totals=[math.fsum(value for fid,value in values.items() if fid.split(":")[0]==session) for session in sessions]
            counts=[sum(fid.split(":")[0]==session for fid in values) for session in sessions]
            samples=[];empty=0
            for indices in draw:
                den=sum(counts[int(i)] for i in indices)
                if den:samples.append(math.fsum(totals[int(i)] for i in indices)/den)
                else:empty+=1
            assert stored["empty_bootstrap_draws"]==empty
            for a,b in zip([quantile(samples,.025),quantile(samples,.975)],stored["CI95_mean_session_cluster"]):check.check(a,b)
    return dict(status="PASS",family="Named Stage1 pair; post-execution alignment, not retrospectively preregistered",
                exact_rational_results=computed,Stage1_moment_CI_summaries_checked=2)


def verify_transitions(doc,frame,c_rows,d_rows,ids,check):
    receipt=read(doc/"COMPARISON_COMPLETION_RECEIPT.json")
    lookup={"C":{identity(row):row for row in c_rows},"D":{identity(row):row for row in d_rows}}
    reused={identity(row):row for row in rows(doc/"AE_CANDIDATE_METRICS.jsonl.gz") if row["backbone"]=="YOLO"}
    def expected_candidates(row,kind):
        if kind=="C":return row["candidates"]
        if "candidate_selector_values" in row:return row["candidate_selector_values"]
        if identity(row) in reused:return reused[identity(row)]["candidate_metrics"]
        assert row["method"]=="OLD_P"
        return {hyp:dict(status="NA_NOT_RETAINED_IN_HISTORICAL_CACHE") for hyp in ["long-face-front","short-face-front"]}
    expected_keys=set()
    for kind,pairs,seeds in [("C",C_PAIRS,[1,2,3]),("D",D_PAIRS,[1])]:
        expected_keys.update((kind,a,b,fid,seed) for a,b in pairs for fid in ids for seed in seeds)
    with (doc/"COMPARISON_TRANSITIONS_COMPLETE.csv").open(newline="") as handle:complete=list(csv.DictReader(handle))
    actual_keys=[(row["source"],row["after"],row["before"],row["id"],int(row["seed"])) for row in complete]
    assert len(complete)==len(actual_keys)==len(set(actual_keys))==8131 and set(actual_keys)==expected_keys
    counts=defaultdict(Counter)
    def number(value):return None if value=="" else float(value)
    for row in complete:
        kind,a,b,fid,seed=row["source"],row["after"],row["before"],row["id"],int(row["seed"])
        after=frame[a,fid,seed];before=frame[b,fid,seed]
        raw_after=lookup[kind][a,fid,seed];raw_before=lookup[kind][b,fid,seed]
        assert row["session"]==after["session"]==fid.split(":")[0] and row["grade"]==after["grade"]
        assert row["from_hypothesis"]==(before["selected_hypothesis"] or "")
        assert row["to_hypothesis"]==(after["selected_hypothesis"] or "")
        transitions={"axis":state(before["axis_wrong"],after["axis_wrong"]),
                     "strict":state(1-before["pose_success"],1-after["pose_success"]),
                     "availability":state(1-before["available"],1-after["available"])}
        counter=counts[a+" - "+b]
        for key,value in transitions.items():assert row[key+"_transition"]==value;counter[key+"_"+value]+=1
        changed=int(before["selected_hypothesis"]!=after["selected_hypothesis"])
        assert int(row["selection_changed"])==changed;counter["selection_flips"]+=changed
        for prefix,source in [("old",before),("new",after)]:
            for target,key in [("available","available"),("success","pose_success"),("axis_wrong","axis_wrong"),
                               ("wd_confusion","wd_confusion"),("T_cm","translation_cm"),("R_deg","rotation_deg")]:
                check.check(number(row[prefix+"_"+target]),source[key])
        assert json.loads(row["old_candidates_json"])==expected_candidates(raw_before,kind)
        assert json.loads(row["new_candidates_json"])==expected_candidates(raw_after,kind)
        delta=raw_after["corner"]["frame_mean_px"]-raw_before["corner"]["frame_mean_px"] if kind=="D" else 0
        check.check(float(row["corner_delta_px"]),delta)
        if kind=="D":assert "centroid Z" in row["candidate_semantics"] and "OLD_P both candidates NA" in row["candidate_semantics"]
        else:assert row["candidate_semantics"]=="unchanged final8 candidate metrics"
    assert {name:dict(value) for name,value in counts.items()}==receipt["comparisons"]
    assert receipt["rows"]==8131 and receipt["C_comparisons"]==11 and receipt["D_comparisons"]==14
    assert receipt["new_fits"]==receipt["new_RGB_forwards"]==receipt["new_PnP_calls"]==0
    for name,digest in receipt["old_artifacts_unchanged"].items():assert sha(doc/name)==digest
    for arm,flips in [("D_JOINT_MEAN",2),("D_JOINT_SHUFFLE",3)]:
        assert counts[arm+" - D_JOINT_DIM"]["selection_flips"]==counts[arm+" - D_JOINT_DIM"]["axis_DAMAGE"]==flips
    return dict(status="PASS",complete_rows=8131,C_pair_comparisons=11,D_pair_comparisons=14,
                no_outcome_based_filtering=True,all_mean_and_shuffle_flips_present=True,
                D_unselected_candidate_fields="predicted selector fits; unavailable final8 metrics explicitly not fabricated",
                comparisons={name:dict(value) for name,value in counts.items()})


def verify_subgroups(doc,source,private,c_rows,ids,check):
    result=read(doc/"C5_SUBGROUPS_SUPPLEMENT.json")
    allowed=set(ids)
    meta=read(source/"_docs/experiments/pallet_dim_conditioned_p_v1/DEV_CACHE_COMPLETE.json")
    types={row["id"]:row["object_type"] for row in meta["records"] if row["id"] in allowed}
    assert set(types)==allowed
    assert dict(Counter(types.values()))==result["object_type_counts"]=={"plastic_standard_110x130x11":128,"wood_small_80x59x14":45}
    registry={row["object_type"]:row["physical_dimensions_m"] for row in read(source/"challenge/config/CHALLENGE_OBJECT_GEOMETRY_REGISTRY.json")["objects"]}
    tables_checked=0
    for object_type,table in result["object_type_tables"].items():
        selected=[row for row in c_rows if types[row["id"]]==object_type]
        assert table["registered_physical_dimensions_XYZ_m"]==registry[object_type]
        assert table["frames"]==len({row["id"] for row in selected})
        for method,stored in table["methods"].items():
            selected_method=[row for row in selected if row["method"]==method]
            by=defaultdict(list)
            for row in selected_method:by[row["id"]].append(row)
            assert all(sorted(int(row["seed"]) for row in records)==[1,2,3] for records in by.values())
            assert stored["frames"]==len(by) and stored["prediction_rows"]==3*len(by)
            for field in ["translation_cm","rotation_deg","yaw_deg","IoU3D","ADDsym_normalized"]:
                values=[]
                for records in by.values():
                    valid=[row["pose"][field] for row in records if row["pose"]["available"]]
                    if valid:values.append(math.fsum(valid)/len(valid))
                check.values(values,stored[field])
            available,success,confusion=zip(*(binary_pose(row) for row in selected_method))
            check.check(sum(available)/3,stored["available_seed_mean_count"])
            assert sum(available)==stored["available_prediction_rows"]
            check.check(sum(available)/len(selected_method),stored["coverage"])
            check.check(sum(success)/len(selected_method),stored["strict_success"])
            check.check(sum(confusion)/len(selected_method),stored["legacy_WD_confusion"])
            assert sum(success)==stored["strict_success_seed_rows"] and sum(confusion)==stored["legacy_WD_confusion_seed_rows"]
            check.check(sum(success)/3,stored["strict_success_frame_equivalent_count"])
            check.check(sum(confusion)/3,stored["legacy_WD_confusion_frame_equivalent_count"])
            evaluable=[row for row in selected_method if row["axis_error"] is not None]
            assert len(evaluable)==stored["direct_axis_evaluable_seed_rows"]
            wrong=sum(row["axis_error"] for row in evaluable)
            assert wrong==stored["direct_axis_error_seed_rows"]
            check.check(wrong/3,stored["direct_axis_error_frame_equivalent_count"])
            check.check(wrong/len(evaluable),stored["axis_error_rate"])
            assert stored["axis_evaluable_frames"]==len(evaluable)//3
            assert stored["UNCLEAR_frames"]==len({row["id"] for row in selected_method if row["axis_error"] is None})
            matrix=[[0,0],[0,0]];brier=[]
            for row in evaluable:
                guess=int(row["selected_hypothesis"]=="long-face-front")
                target=guess if not row["axis_error"] else 1-guess
                matrix[target][guess]+=1
                if row["p_long"] is not None:brier.append((row["p_long"]-target)**2)
            classification=stored["classification"]
            assert matrix==classification["confusion_matrix_seed_rows_true_short_long"]
            check.check(sum(matrix[i][i] for i in (0,1))/len(evaluable),classification["accuracy"])
            recalls=[matrix[i][i]/sum(matrix[i]) for i in (0,1) if sum(matrix[i])]
            check.check(math.fsum(recalls)/2 if len(recalls)==2 else None,classification["balanced_accuracy"])
            check.check(math.fsum(brier)/len(brier) if brier else None,classification["Brier"])
            # Exact integration of threshold accuracy counts, including failed poses.
            errors=[row["pose"].get("ADDsym_normalized",float("inf")) if row["pose"]["available"] else float("inf") for row in selected_method]
            accuracy=[sum(value<=threshold for value in errors) for threshold in np.linspace(0,.1,1001)]
            auc=Fraction(accuracy[0]+accuracy[-1]+2*sum(accuracy[1:-1]),2000*len(errors))
            check.check(float(auc),stored["ADDsym_AUC_full"])
            tables_checked+=1
    size=result["size_subgroups"]
    assert size["status"]=="NA_NO_PRELOCKED_BIN_CONTRACT" and size["quantity_is_GT_free"]
    assert all(row["status"]=="NA" and row["size_bins"] is None and row["metrics"] is None for row in size["methods"].values())
    with np.load(private/"C_REAL_FEATURES.npz",allow_pickle=False) as features:
        feature_ids=features["ids"].tolist();geometry=features["G"]
    column=read(doc/"C_GEOMETRY_RECEIPT.json")["geometry_names"].index("bbox_area_image_fraction")
    assert geometry.shape==(519,2,94) and set(feature_ids)==allowed
    assert np.array_equal(geometry[:,0,column],geometry[:,1,column])
    size_values=defaultdict(list)
    for fid,value in zip(feature_ids,geometry[:,0,column]):size_values[fid].append(float(value))
    assert all(len(values)==3 and len(set(values))==1 for values in size_values.values())
    values=[group[0] for group in size_values.values()]
    expected=dict(min=min(values),P10=quantile(values,.1),median=quantile(values,.5),P90=quantile(values,.9),max=max(values))
    for key,value in expected.items():check.check(value,size["frame_quantiles_only_descriptive"][key])
    assert size["frames"]==size["finite_frames"]==173
    assert result["no_new_fit"] and result["new_model_forwards"]==result["new_OpenCV_calls"]==result["protected_per_ID_data_reads"]==0
    return dict(status="PASS",object_type_counts=result["object_type_counts"],method_subgroup_tables_checked=tables_checked,
                size_subgroups="Explicit NA: no prelocked bin boundaries; continuous runtime size descriptive quantiles independently matched")


def verify_permutation(doc,private,source,published,ids,check):
    root=doc.parents[2]
    lock=read(doc/"B1_PERMUTATION_REPAIR_LOCK.json")
    donor_seal=read(doc/"B1_PERMUTATION_REPAIR_DONOR_SEAL.json")
    prediction_seal=read(doc/"B1_PERMUTATION_REPAIR_PREDICTION_SEAL.json")
    result=read(doc/"B1_PERMUTATION_REPAIR_RESULTS.json")
    compute=read(doc/"B1_PERMUTATION_REPAIR_COMPUTE_RECEIPT.json")
    roots={"historical_source":source,"published_source":published,"new_worktree":root,"private_new":private}
    for binding in lock["source_bindings"]+lock["preserve"]+[lock["code"],lock["donor_seal"],
            donor_seal["private_donors"],prediction_seal["private_predictions"],prediction_seal["source_checkpoint"]]+prediction_seal["cache_bindings"]+prediction_seal["logit_bindings"]:
        assert sha(roots[binding["owner"]]/binding["path"])==binding["sha256"],binding["path"]
    assert lock["new_fits"]==lock["new_CNN_forwards"]==lock["raw_RGB_reads"]==0
    assert prediction_seal["sealed_before_reference_scoring"] and donor_seal["sealed_without_reference_pose_or_error"]
    assert lock["correct_decode_parity_tolerance_px"]==prediction_seal["tolerance_px"]==1e-4
    assert lock["seed"]==donor_seal["seed"]==20261011 and lock["checkpoint_seed"]==1
    source_manifest=read(source/"data/pallet/results/pallet_line_pose_v1/SOURCE_MANIFEST.json")["records"]
    pool=np.array([i for i,row in enumerate(source_manifest) if row["partition"]=="train"],dtype=np.int64)
    assert len(pool)==lock["pool_count"]==donor_seal["pool_count"]==55980
    permutation=np.random.default_rng(20261011).permutation(pool)
    pool_sha=hashlib.sha256(pool.astype("<i8").tobytes()).hexdigest()
    perm_sha=hashlib.sha256(permutation.astype("<i8").tobytes()).hexdigest()
    assert pool_sha==lock["pool_index_array_sha256"]==donor_seal["pool_index_array_sha256"]
    assert perm_sha==lock["full_permutation_index_array_sha256"]==donor_seal["full_permutation_index_array_sha256"]
    donors=read(private/"B1_PERMUTATION_REPAIR_DONORS_SEALED.json")
    assert donors["seed"]==20261011 and [row["id"] for row in donors["rows"]]==ids==donor_seal["assigned_ids"]
    assigned=[row["original_TRAIN_record_index"] for row in donors["rows"]]
    assert len(assigned)==len(set(assigned))==173 and assigned==permutation[:173].tolist()
    with np.load(source/"data/pallet/results/pallet_dim_conditioned_p_v1/DIMENSION_SIDECAR.npz",allow_pickle=False) as archive:dimensions=archive["dimensions"]
    for row in donors["rows"]:
        index=row["original_TRAIN_record_index"]
        assert row["donor_id"]==source_manifest[index]["id"] and source_manifest[index]["partition"]=="train"
        assert row["canonical_dimensions_WDH_m"]==dimensions[index].tolist()
    del source_manifest
    # Parse only already authorized predictions; no reference poses are read.
    allowed=set(ids);old_published={};pattern=re.compile(r'"id"\s*:\s*"([^"\\]+)"')
    with gzip.open(published/"_docs/experiments/pallet_feature_gradient_joint_20261010/PREDICTIONS.jsonl.gz","rt") as stream:
        for line in stream:
            match=pattern.search(line)
            if match and match.group(1) in allowed:
                row=json.loads(line)
                if row["method"]=="N3_DIM_SYM" and int(row["seed"])==1:
                    old_published[row["id"]]={key:row[key] for key in ["qFinal","q0","raw_hw"]}
    assert set(old_published)==allowed
    old_private={row["id"]:row for row in read(private/"B1_PREDICTIONS_SEALED.json") if row["method"]=="CORRECT"}
    fresh=read(private/"B1_PERMUTATION_REPAIR_PREDICTIONS_SEALED.json")
    assert [row["id"] for row in fresh]==ids and set(old_private)==allowed
    parity=0.;cached_parity=0.;cap_overshoot=0.
    for row,donor in zip(fresh,assigned):
        fid=row["id"];q=np.asarray(row["q"],float);correct=np.asarray(row["q_correct"],float)
        original=np.asarray(old_published[fid]["qFinal"],float);base=np.asarray(old_published[fid]["q0"],float)
        assert q.shape==correct.shape==(9,2) and np.isfinite(q).all() and np.isfinite(correct).all()
        cached_parity=max(cached_parity,float(np.linalg.norm(correct-np.asarray(old_private[fid]["q"]),axis=-1).max()))
        parity=max(parity,float(np.linalg.norm(correct-original,axis=-1).max()))
        assert np.array_equal(q[8],base[8]) and np.array_equal(correct[8],base[8])
        cap=.01*math.hypot(*old_published[fid]["raw_hw"])
        cap_overshoot=max(cap_overshoot,float(np.linalg.norm(q[:8]-base[:8],axis=-1).max()-cap))
        diagnostics=row["diagnostic"]
        assert diagnostics["donor_original_TRAIN_record_index"]==donor
        assert diagnostics["supported_corners"]==8 and len(diagnostics["coordinate_delta_px"])==8
        for actual,expected in zip(np.linalg.norm(q[:8]-original[:8],axis=-1),diagnostics["coordinate_delta_px"]):check.check(actual,expected)
    assert parity<=1e-4 and cached_parity==0 and cap_overshoot<=1e-4
    check.check(parity,prediction_seal["correct_decode_max_error_px"],1e-12)
    scored=rows(doc/"B1_PERMUTATION_REPAIR_FRAME_ROWS.jsonl.gz")
    old={row["id"]:row for row in rows(doc/"B1_FRAME_ROWS.jsonl.gz") if row["method"]=="B1_CORRECT"}
    assert len(scored)==173 and [row["id"] for row in scored]==ids and set(old)==allowed
    for row,pred in zip(scored,fresh):
        assert row["method"]==lock["method"]==result["method"] and row["seed"]==1
        assert row["diagnostic"]==pred["diagnostic"] and row["session"]==row["id"].split(":")[0]
        available,success,confusion=binary_pose(row)
        assert row["success"]==success and row["legacy_confusion"]==confusion
    def summary(records,stored):
        assert stored["frames"]==stored["prediction_rows"]==len(records)==173
        available,success,confusion=zip(*(binary_pose(row) for row in records))
        check.check(sum(available),stored["available_seed_mean_count"])
        assert sum(available)==stored["available_prediction_rows"]
        check.check(sum(available)/173,stored["coverage"])
        check.check(sum(success)/173,stored["strict_success"])
        check.check(sum(confusion)/173,stored["legacy_WD_confusion"])
        evaluable=[row for row in records if row["axis_error"] is not None]
        assert len(evaluable)==stored["axis_evaluable_frames"]==170
        check.check(sum(row["axis_error"] for row in evaluable)/170,stored["axis_error_rate"])
        for field in ["translation_cm","rotation_deg","yaw_deg","IoU3D","ADDsym_normalized"]:
            check.values([row["pose"][field] for row in records if row["pose"]["available"]],stored[field])
        errors=[row["pose"]["ADDsym_normalized"] if row["pose"]["available"] else float("inf") for row in records]
        accuracy=[sum(value<=threshold for value in errors) for threshold in np.linspace(0,.1,1001)]
        check.check(float(Fraction(accuracy[0]+accuracy[-1]+2*sum(accuracy[1:-1]),2000*173)),stored["ADDsym_AUC_full"])
    summary(scored,result["metrics"]);summary(list(old.values()),result["correct_reused_metrics"])
    diagnostics=[row["diagnostic"] for row in fresh];coordinate_values=[value for row in diagnostics for value in row["coordinate_delta_px"]]
    check.values(coordinate_values,result["coordinate_delta_px"])
    check.check(max(row["centered_logit_maxabs"] for row in diagnostics),result["centered_logit_maxabs"])
    check.check(math.fsum(row["probability_L1_mean"] for row in diagnostics)/173,result["probability_L1_mean"])
    for field in ["argmax_changed","supported_corners","cap_active"]:assert sum(row[field] for row in diagnostics)==result[field]
    switches=sum(row["selected_hypothesis"]!=old[row["id"]]["selected_hypothesis"] for row in scored)
    assert switches==result["WD_choice_switches"]==0
    assert sum(row["success"] for row in scored)==result["strict_success_count"]==75
    assert sum(row["success"] for row in old.values())==result["correct_strict_success_count"]==77
    expected_transitions=[]
    for row in scored:
        before=old[row["id"]]
        for event,a,b in [("AXIS_SELECTION",before["selected_hypothesis"],row["selected_hypothesis"]),
                          ("POSE_AVAILABLE",int(before["pose"]["available"]),int(row["pose"]["available"])),
                          ("STRICT_SUCCESS",before["success"],row["success"])]:
            if a!=b:expected_transitions.append((row["id"],event,str(a),str(b)))
    with (doc/"B1_PERMUTATION_REPAIR_TRANSITIONS.csv").open(newline="") as handle:transitions=list(csv.DictReader(handle))
    assert [(r["id"],r["event"],r["before"],r["after"]) for r in transitions]==expected_transitions
    assert len(transitions)==result["transitions"]==4
    sessions=sorted({fid.split(":")[0] for fid in ids});draw=np.load(doc/"BOOTSTRAP_DRAWS.npy",allow_pickle=False)
    descriptive=[]
    for field,stored in result["paired_descriptive"].items():
        delta={row["id"]:row[field]-old[row["id"]][field] for row in scored}
        totals=[sum(delta[fid] for fid in ids if fid.split(":")[0]==s) for s in sessions]
        counts=[sum(fid.split(":")[0]==s for fid in ids) for s in sessions]
        effect=Fraction(sum(totals),173);sampled=[];extreme=0
        for indices in draw:
            value=Fraction(sum(totals[int(i)] for i in indices),sum(counts[int(i)] for i in indices))
            sampled.append(float(value*100));extreme+=int(abs(value-effect)>=abs(effect))
        p=Fraction(1+extreme,10001)
        check.check(float(effect*100),stored["effect_pp"])
        check.check(float(p),stored["p_raw"],1e-15)
        for a,b in zip([quantile(sampled,.025),quantile(sampled,.975)],stored["CI95"]):check.check(a,b)
        assert stored["numerator"]==sum(totals) and stored["denominator"]==173
        descriptive.append(dict(metric=field,effect_pp=float(effect*100),exact_p=float(p),extreme_draws=extreme))
    # Compare the physical append-only ledger, including all failed first attempts.
    ledger_bytes=(private/"PNP_RECEIPTS.jsonl").read_bytes();ledger=[json.loads(line) for line in ledger_bytes.splitlines()]
    prior_binding=lock["previous_ledger"]
    assert hashlib.sha256(ledger_bytes[:prior_binding["bytes"]]).hexdigest()==prior_binding["sha256"]
    after_binding=compute["private_ledger_after"]
    assert hashlib.sha256(ledger_bytes[:after_binding["bytes"]]).hexdigest()==after_binding["sha256"]
    repair_positions=[i for i,receipt in enumerate(ledger) if receipt["job"]=="B1_PERMUTATION_REPAIR"]
    assert len(repair_positions)==1
    repair_index=repair_positions[0];repair_receipt=ledger[repair_index]
    assert compute["counter_receipt"]==repair_receipt
    for receipt in ledger:assert sum(receipt["calls"].values())==receipt["total"]
    total=sum(receipt["total"] for receipt in ledger)
    assert 49315<=total<=50000 and sum(receipt["total"] for receipt in ledger[:repair_index])==48277
    assert repair_receipt["total"]==result["actual_OpenCV_calls"]==1038
    assert repair_receipt["calls"]["solvePnP"]==repair_receipt["calls"]["solvePnPRefineLM"]==519
    assert lock["old_reservation"]+lock["supplemental_reservation"]==compute["reservation_total"]==49754<=50000
    assert result["new_fits"]==result["new_CNN_forwards"]==compute["new_fits"]==compute["new_CNN_forwards"]==compute["new_GPU_device_seconds"]==0
    assert compute["protected_per_ID_JSON_decodes"]==compute["raw_RGB_reads"]==0
    assert not result["modifies_prelocked_Holm_family"] and result["historical_hash_sampling_preserved"]
    return dict(status="PASS",original_TRAIN_pool=55980,assigned_unique_donors=173,donor_duplicates=0,
                full_permutation_index_array_sha256=perm_sha,CORRECT_new_decode_vs_old_private_decode_max_px=cached_parity,
                CORRECT_new_decode_vs_original_published_points_max_px=parity,new_coordinates_preserve_center8=True,
                native_final_cap_max_overshoot_px=cap_overshoot,strict_success_before=77,strict_success_after=75,
                selection_flips=0,all_changed_outcome_rows_present=4,descriptive_paired=descriptive,
                added_OpenCV_calls=1038,cumulative_actual_OpenCV_calls=total,reserved_OpenCV_calls_through_permutation=49754,
                supplemental_fits=0,supplemental_backbone_forwards=0,
                frozen_metadata_head_evaluation="Existing frozen metadata residual only; no new backbone pass or training",
                donor_sealed_before_reference=True,predictions_sealed_before_reference=True)


def verify_ratio_correction(doc,root,source,published,check):
    lock=read(doc/"A3_RATIO_AXIS_CORRECTION_LOCK.json");result=read(doc/"A3_RATIO_AXIS_CORRECTION.json")
    roots={"historical_source":source,"published_source":published,"new_worktree":root}
    for binding in lock["source_bindings"]+[lock["code_binding"]]+result["preserved_source_bindings_after"]:
        assert sha(roots[binding["owner"]]/binding["path"])==binding["sha256"],binding["path"]
    assert result["sample_frames"]==16 and result["derived_rows"]==len(result["rows"])==48 and result["conditions"]==3
    assert {row["id"] for row in result["rows"]}==set(lock["sample_ids"])
    assert result["sample_sha256"]==lock["sample_sha256"]
    assert result["original_pose_gate"]=="FAIL_2_OF_16_UNRESOLVED"
    assert set(result["original_pose_failure_ids"])=={"G38__G__f30814","G38__G__f30131"}
    assert result["corrected_6D_metrics_status"]=="NOT_EVALUATED_NO_NEW_PNP"
    for key in ["new_PnP_calls","fits","model_inference","RGB_reads","new_RGB_generation","protected_REAL_processing"]:
        assert result[key]==lock[key]==0
    by=defaultdict(list)
    for row in result["rows"]:
        by[row["ratio_condition"]].append(row)
        assert row["signed_normalized_corner_order_max_error"]<1e-12
        assert row["corrected_coordinate_span_max_error_m"]<1e-12 and row["footprint_area_max_error_m2"]<1e-12
        check.check(row["positive_axis_scale_determinant"],1,1e-12)
        assert row["center_projection_displacement_px"]<1e-10
        assert not row["raw_reference_R_t_K_or_points_public"] and row["corrected_6D_scores"]=="NOT_EVALUATED_NO_NEW_PNP"
    registered={"plastic_ratio":1.1/1.3,"wood_ratio":.8/.59,"square":1.}
    for summary in result["summaries"]:
        records=by[summary["ratio_condition"]];assert len(records)==16
        check.check(summary["registered_canonical_W_over_D"],registered[summary["ratio_condition"]])
        check.values([row["nine_point_projection_displacement_px"]["mean"] for row in records],summary["mean_projection_displacement_per_frame_px"])
        check.check(max(row["nine_point_projection_displacement_px"]["maximum"] for row in records),summary["maximum_projection_displacement_px"])
        assert sum(row["original_and_corrected_geometry_identical"] for row in records)==summary["geometry_unchanged_frames"]
        assert sum(row["nonsquare_parity_changed"] for row in records)==summary["nonsquare_parity_changed_frames"]
        assert summary["corrected_pose_metrics"]=="NOT_EVALUATED_NO_NEW_PNP"
    assert sum(row["nonsquare_parity_changed"] for row in by["wood_ratio"])==16
    return dict(status="PASS",source_bindings_unchanged=True,derived_projection_rows=48,new_PnP=0,new_fits=0,
                shape_condition_axis_repaired_only=True,original_signed_C1_pose_gate="FAIL_2_OF_16_UNRESOLVED")


def verify_clean_preflight(doc,private,check):
    root=doc.parents[2];prepared=read(doc/"C_CLEAN_REPLAY_PREPARATION.json")
    details=read(doc/"C5_CONTRACT_DETAILS_SUPPLEMENT.json")
    for binding in [prepared["code"],prepared["classifier_source"]]+prepared["prior_published_outputs_unchanged"]:
        assert sha(root/binding["path"])==binding["sha256"]
    fresh=private/prepared["fresh_root_relative_to_private"]
    extraction_lock=read(fresh/"_docs/experiments/pallet_wd_info_fast_20261011/C_CODE_LOCK.json")
    fit_lock=read(fresh/"_docs/experiments/pallet_wd_info_fast_20261011/C_IO_PREFIT_LOCK.json")
    expected=sha(root/"scripts/research/pallet_wd_info_fast_20261011/classifier.py")
    assert expected==extraction_lock["code"]["sha256"]==fit_lock["code"]["sha256"]==prepared["classifier_source"]["sha256"]
    assert expected==sha(fresh/"scripts/research/pallet_wd_info_fast_20261011/classifier.py")
    for key in ["arms","logistic","MLP","geometry_dimensions","visual_dimensions","dimension_dimensions","common_MLP_input_dimensions","tie"]:
        assert extraction_lock[key]==fit_lock[key]
    fresh_plan=read(fresh/"_docs/experiments/pallet_wd_info_fast_20261011/PLAN_LOCK.json")
    for binding in fresh_plan["new_code_bindings"]:assert sha(fresh/binding["path"])==binding["sha256"]
    assert not (fresh/"_docs/experiments/pallet_wd_info_fast_20261011/C_CODE_FINAL_LOCK.json").exists()
    assert not (fresh/"private/C_SYNTH_PREDICTIONS_SEALED.npz").exists()
    assert not (fresh/"_docs/experiments/pallet_wd_info_fast_20261011/C_FITS.json").exists()
    for key in ["PLAN_guard","extract_source_guard","fit_source_guard"]:assert prepared["actual_guard_receipt"][key]=="PASS"
    assert prepared["new_fits"]==prepared["new_CNN_forwards"]==prepared["new_OpenCV_calls"]==0
    assert "Does not claim a second full experiment" in prepared["scope"]
    with np.load(private/"C_SYNTH_PREDICTIONS_SEALED.npz",allow_pickle=False) as archive:split=archive["split"];ids=archive["ids"]
    with np.load(private/"C_SYNTH_LABELS.npz",allow_pickle=False) as archive:elevation=archive["elevation"];label_ids=archive["ids"]
    assert np.array_equal(ids,label_ids) and len(ids)==6144
    for name,stored in details["counts"].items():
        mask=split==name
        assert stored==dict(frames=int(mask.sum()),elevation_finite=int((mask & np.isfinite(elevation)).sum()),
                           elevation_ge30=int((mask & (elevation>=30)).sum()))
    assert details["counts"]["TRAIN"]["elevation_ge30"]==1854 and details["counts"]["VAL"]["elevation_ge30"]==2
    assert details["counts"]["TEST"]["elevation_ge30"]==0 and details["high_elevation_test_metrics"] is None
    assert details["new_fits"]==details["new_CNN_forwards"]==details["new_OpenCV_calls"]==details["protected_per_ID_reads"]==0
    return dict(status="PASS",fresh_locks_match_final_current_source=True,fresh_guard_conditions_verified=True,
                fresh_full_rerun_executed=False,additional_fits_or_backbone_forwards_or_PnP=0,
                synthetic_ge30_frame_counts={name:row["elevation_ge30"] for name,row in details["counts"].items()},
                TEST_high_elevation_metric="NA: zero fixed TEST frames")


def verify_changed_b1(doc,private,source,published,check):
    root=doc.parents[2];lock=read(doc/"CHANGED_B1_CANDIDATES_LOCK.json")
    receipt=read(doc/"CHANGED_B1_CANDIDATES_RECEIPT.json")
    roots={"new_worktree":root,"private_new":private,"historical_source":source,"published_source":published}
    for binding in lock["inputs"]+[lock["code"],receipt["code"],receipt["output"]]+receipt["source_bindings_after"]:
        assert sha(roots[binding["owner"]]/binding["path"])==binding["sha256"],binding["path"]
    old_rows=rows(doc/"B1_FRAME_ROWS.jsonl.gz");perm=rows(doc/"B1_PERMUTATION_REPAIR_FRAME_ROWS.jsonl.gz")
    baseline={row["id"]:row for row in old_rows if row["method"]=="B1_CORRECT"}
    outcomes={(row["method"],row["id"]):row for row in old_rows+perm}
    expected=[]
    for row in old_rows+perm:
        before=baseline[row["id"]]
        for event,a,b in [("AXIS_SELECTION",before["selected_hypothesis"],row["selected_hypothesis"]),
                          ("STRICT_SUCCESS",before["success"],row["success"]),
                          ("POSE_AVAILABLE",int(before["pose"]["available"]),int(row["pose"]["available"]))]:
            if a!=b:expected.append(dict(method=row["method"],id=row["id"],event=event,before=a,after=b))
    assert len(expected)==14 and lock["events"]==expected
    pairs=sorted({(event["method"],event["id"]) for event in expected},
                 key=lambda pair:(hashlib.sha256(json.dumps(pair,separators=(",",":")).encode()).hexdigest(),pair))
    assert [list(pair) for pair in pairs]==lock["fixed_hash_sorted_method_id_pairs"]
    digest=hashlib.sha256(json.dumps(pairs,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode()).hexdigest()
    # The frozen geometry digest helper uses the compact JSON list convention.
    assert digest==lock["pair_sha256"]==receipt["pair_sha256"]
    cached={row["id"]:row for row in rows(doc/"AE_CANDIDATE_METRICS.jsonl.gz")
            if row["backbone"]=="YOLO" and row["method"]=="N3_DIM_SYM" and row["seed"]==1}
    with (doc/"CHANGED_B1_CANDIDATES_COMPLETE.csv").open(newline="") as stream:complete=list(csv.DictReader(stream))
    assert len(complete)==14 and [(row["method"],row["id"],row["event"]) for row in complete]==[(row["method"],row["id"],row["event"]) for row in expected]
    for record,event in zip(complete,expected):
        after=outcomes[record["method"],record["id"]];before=baseline[record["id"]]
        assert record["before"]==str(event["before"]) and record["after"]==str(event["after"])
        assert json.loads(record["old_pose_metrics_json"])==before["pose"]
        assert json.loads(record["new_pose_metrics_json"])==after["pose"]
        assert json.loads(record["old_candidates_json"])==dict(candidate_metrics=cached[record["id"]]["candidate_metrics"],
                                                                  candidate_scores=cached[record["id"]]["candidate_scores"])
        for prefix,row in [("old",before),("new",after)]:
            check.check(float(record[prefix+"_axis_error"]) if record[prefix+"_axis_error"] else None,row["axis_error"])
            assert int(record[prefix+"_success"])==row["success"]
        candidates=json.loads(record["new_candidates_json"])
        selected=json.loads(record["new_selected_final8_pose_json"])
        assert set(candidates)=={"long-face-front","short-face-front"}
        assert bool(selected["available"])==bool(after["pose"]["available"])
        assert selected["selected_hypothesis"]==after["selected_hypothesis"]
        for candidate in candidates.values():
            assert "initial selector fit" in candidate["semantics"] and "not final8" in candidate["semantics"]
            if candidate["success"]:
                rotation=np.asarray(candidate["R_cf"],float);centroid=np.asarray(candidate["centroid_m"],float)
                assert rotation.shape==(3,3) and centroid.shape==(3,) and np.isfinite(rotation).all() and np.isfinite(centroid).all()
                assert np.max(np.abs(rotation.T@rotation-np.eye(3)))<1e-8 and abs(np.linalg.det(rotation)-1)<1e-8
                assert math.isfinite(candidate["score"]) and centroid[2]>0
    assert receipt["events"]==receipt["method_id_pairs"]==14 and receipt["event_counts"]=={"STRICT_SUCCESS":14}
    assert not lock["reference_inputs"] and not lock["new_configuration"]
    for key in ["new_fits","new_CNN_forwards","RGB_reads","references_read"]:assert receipt[key]==0
    ledger=[json.loads(line) for line in (private/"PNP_RECEIPTS.jsonl").read_text().splitlines()]
    matching=[row for row in ledger if row["job"]=="CHANGED_B1_CANDIDATES"]
    assert matching==[receipt["counter_receipt"]]
    assert matching[0]["total"]==lock["new_call_limit"]==84
    assert matching[0]["calls"]["solvePnP"]==matching[0]["calls"]["solvePnPRefineLM"]==42
    assert matching[0]["prior_global_actual"]==lock["prior_global_actual"]==49315
    assert matching[0]["global_actual_after"]==receipt["global_actual_after"]==49399
    assert lock["reservation_total"]==receipt["reservation_total"]==49838
    return dict(status="PASS",all_changed_events=14,all_changed_method_ID_pairs=14,new_OpenCV_calls=84,
                reference_inputs_or_additional_fits_or_backbone_forwards=0,original_outcomes_unchanged=True,
                candidate_semantics="New initial selector fits and selected final8 pose distinguished; baseline final8 cache reused")


def verify_changed_oldp(doc,private,source,published,check):
    root=doc.parents[2];lock=read(doc/"OLD_P_CHANGED_CANDIDATES_LOCK.json")
    receipt=read(doc/"OLD_P_CHANGED_CANDIDATES_RECEIPT.json")
    roots={"new_worktree":root,"historical_source":source,"published_source":published}
    for binding in lock["inputs"]+[lock["code"],receipt["code"],receipt["output"]]+receipt["source_bindings_after"]:
        assert sha(roots[binding["owner"]]/binding["path"])==binding["sha256"],binding["path"]
    with (doc/"COMPARISON_TRANSITIONS_COMPLETE.csv").open(newline="") as stream:original=list(csv.DictReader(stream))
    changed=[row for row in original if row["source"]=="D" and row["before"]=="OLD_P" and
             (row["selection_changed"]=="1" or any(row[field] in {"RECOVERY","DAMAGE"} for field in ["axis_transition","strict_transition","availability_transition"]))]
    assert len(changed)==15
    ids=sorted({row["id"] for row in changed},key=lambda fid:(hashlib.sha256(fid.encode()).hexdigest(),fid))
    assert len(ids)==12 and ids==lock["hash_sorted_ids"]==receipt["ids"]
    assert lock["fixed_pairs"]==[dict(after=row["after"],id=row["id"]) for row in changed]
    assert hashlib.sha256(json.dumps(ids,sort_keys=True,separators=(",",":")).encode()).hexdigest()==lock["sample_sha256"]
    decoder=json.JSONDecoder()
    def field(text,key):
        match=re.search(r'"'+key+r'"\s*:\s*',text);assert match,key
        return decoder.raw_decode(text[match.end():])[0]
    predictions={};buffer=[]
    prediction_path=source/"data/pallet/results/pallet_dim_conditioned_p_v1/predictions/REAL_DEV/OLD_P_seed1.json"
    with prediction_path.open() as stream:
        for line in stream:
            if line.rstrip()=="    {":buffer=[line]
            elif buffer:
                buffer.append(line)
                if line.rstrip() in {"    },","    }"}:
                    text="".join(buffer).rstrip().removesuffix(",")
                    fid=field(text,"id")
                    if fid in ids:
                        selected=field(text,"selected_index");assert selected is not None
                        predictions[fid]=np.asarray(field(text,"candidates")[selected]["keypoints_xy"],float)
                    buffer=[]
    historical={};q_parity=0.
    with gzip.open(published/"_docs/experiments/pallet_paper_tables_245_20261011/RECT_ROWS.jsonl.gz","rt") as stream:
        for line in stream:
            fid=field(line,"id")
            if fid not in ids:continue
            if field(line,"backbone")!="YOLO" or field(line,"seed")!=1 or field(line,"method")!="P":continue
            historical[fid]=field(line,"actual_pose")
            delta=float(np.abs(predictions[fid]-np.asarray(field(line,"qFinal"),float)).max())
            assert delta==0;q_parity=max(q_parity,delta)
    assert set(predictions)==set(historical)==set(ids)
    with (doc/"OLD_P_CHANGED_CANDIDATES_COMPLETE.csv").open(newline="") as stream:complete=list(csv.DictReader(stream))
    assert len(complete)==len(changed)==15
    checked_poses={}
    for old,row in zip(changed,complete):
        assert (row["after"],row["id"],row["seed"])==(old["after"],old["id"],old["seed"])
        for key,value in old.items():
            if key!="old_candidates_json":assert row[key]==value,(key,row["id"])
        assert "NA_NOT_RETAINED" not in row["old_candidates_json"]+row["new_candidates_json"]
        candidates=json.loads(row["old_candidates_json"])
        assert set(candidates)=={"long-face-front","short-face-front"}
        selected=json.loads(row["old_selected_final8_pose_json"]);cached=historical[row["id"]]
        assert selected["available"]==cached["available"] and selected["selected_hypothesis"]==cached["selected_hypothesis"]==row["from_hypothesis"]
        difference={key:float(np.max(np.abs(np.asarray(selected[key])-np.asarray(cached[key])))) for key in ["R_cf","centroid"]}
        assert max(difference.values())<=1e-10
        if row["id"] in checked_poses:assert checked_poses[row["id"]]==difference
        checked_poses[row["id"]]=difference
        for candidate in candidates.values():
            assert "initial selector fit" in candidate["semantics"]
            if candidate["success"]:
                rotation=np.asarray(candidate["R_cf"],float);centroid=np.asarray(candidate["centroid_m"],float)
                assert rotation.shape==(3,3) and centroid.shape==(3,) and np.isfinite(rotation).all() and np.isfinite(centroid).all()
                assert np.max(np.abs(rotation.T@rotation-np.eye(3)))<1e-8 and abs(np.linalg.det(rotation)-1)<1e-8
    for row in receipt["cached_pose_parity"]:
        assert row["selected_hypothesis_equal"]
        for key,value in checked_poses[row["id"]].items():check.check(value,row[key],1e-12)
    check.check(q_parity,receipt["cached_prediction_max_delta_px"],1e-12)
    assert receipt["comparison_rows"]==15 and receipt["unique_baseline_frames"]==12
    for key in ["new_fits","new_CNN_forwards","raw_RGB_reads","raw_reference_poses_annotations_read"]:assert lock[key]==receipt[key]==0
    ledger=[json.loads(line) for line in (private/"PNP_RECEIPTS.jsonl").read_text().splitlines()]
    matching=[row for row in ledger if row["job"]=="OLD_P_CHANGED_CANDIDATES"]
    assert matching==[receipt["counter_receipt"]] and matching[0]["total"]==lock["new_call_limit"]==72
    assert matching[0]["calls"]["solvePnP"]==matching[0]["calls"]["solvePnPRefineLM"]==36
    assert matching[0]["prior_global_actual"]==lock["prior_global_actual"]==49399
    assert sum(row["total"] for row in ledger)==matching[0]["global_actual_after"]==receipt["global_actual_after"]==49471<=50000
    reservations=read(doc/"PLAN_LOCK.json")["pnp_reservations"]
    assert sum(reservations.values())+1038+84+72==lock["reservation_total"]==receipt["reservation_total"]==49910<=50000
    original_counts=Counter()
    for row in ledger:
        assert sum(row["calls"].values())==row["total"]
        original_counts[row["job"]]+=row["total"]
    assert set(original_counts)==set(reservations)|{"B1_PERMUTATION_REPAIR","CHANGED_B1_CANDIDATES","OLD_P_CHANGED_CANDIDATES"}
    for job,limit in reservations.items():assert original_counts[job]<=limit
    assert original_counts["B1_PERMUTATION_REPAIR"]==1038 and original_counts["CHANGED_B1_CANDIDATES"]==84 and original_counts["OLD_P_CHANGED_CANDIDATES"]==72
    return dict(status="PASS",all_changed_stage1_OLD_P_comparison_rows=15,unique_baseline_IDs=12,
                independently_verified_historical_prediction_parity_px=q_parity,
                independently_verified_R_cf_and_centroid_parity_max=max(value for delta in checked_poses.values() for value in delta.values()),
                both_candidates_complete_for_all_changed_comparisons=True,unchanged_original_NA_rows_preserved=True,
                new_OpenCV_calls=72,cumulative_actual_OpenCV_calls=49471,cumulative_reserved_OpenCV_calls=49910,
                per_job_actual_calls=dict(original_counts),additional_fits_or_backbone_forwards_or_reference_inputs=0)


def preservation(root,doc,source,published):
    old=read(doc/"SOURCE_PRESERVATION.json")
    roots={"historical_source":source,"published_source":published}
    for binding in old["historical_bound_files"]:assert sha(roots[binding["owner"]]/binding["path"])==binding["sha256"],binding["path"]
    original_status=subprocess.check_output(["git","-C",str(source),"status","--porcelain=v1","-z"])
    original_diff=subprocess.check_output(["git","-C",str(source),"diff","--binary","HEAD"])
    assert hashlib.sha256(original_status).hexdigest()==old["status_sha256"]
    assert hashlib.sha256(original_diff).hexdigest()==old["tracked_diff_sha256"]
    plan=read(doc/"PLAN_LOCK.json")
    for binding in plan["new_code_bindings"]+plan["input_contract_bindings"]:
        assert sha(root/binding["path"])==binding["sha256"],binding["path"]
    original_commit=read(doc/"PUBLICATION_VERIFICATION.json")["published_science_commit_sha"]
    scientific=[]
    listing=subprocess.check_output(["git","-C",str(root),"ls-tree","-r","--name-only",original_commit],text=True).splitlines()
    prefixes=["_docs/experiments/pallet_wd_info_fast_20261011/","scripts/research/pallet_wd_info_fast_20261011/"]
    for name in listing:
        if not any(name.startswith(prefix) for prefix in prefixes) or Path(name).suffix==".md":continue
        before=subprocess.check_output(["git","-C",str(root),"show",original_commit+":"+name])
        assert hashlib.sha256(before).hexdigest()==sha(root/name),name
        scientific.append(name)
    return dict(status="PASS",historical_frozen_bindings=len(old["historical_bound_files"]),original_git_status_and_diff_identical=True,
                unchanged_original_scientific_and_code_files=len(scientific),original_science_commit=original_commit,
                old_PLAN_and_core_inputs_unchanged=True,Markdown_excluded_from_scientific_freeze="Report-only supplementation and delimiter fixes permitted")


def disclosure_and_markdown(root,doc):
    issues=[];keys=[];links=[];pending=[];table_rows=0;file_count=0
    deferred={"COMPLETION_COMPUTE_RECEIPT.json","COMPLETION_PREPUBLICATION_AUDIT.json"}
    def walk(value,filename):
        if isinstance(value,dict):
            for key,child in value.items():
                descriptor=isinstance(child,dict) and set(child)=={"file","dtype","shape"}
                if key in FORBIDDEN and not descriptor:keys.append(dict(file=filename,key=key))
                walk(child,filename)
        elif isinstance(value,list):
            for child in value:walk(child,filename)
    for base in [doc,root/"scripts/research/pallet_wd_info_fast_20261011"]:
        for path in base.rglob("*"):
            if not path.is_file():continue
            file_count+=1
            if path.suffix in {".pt",".pth",".npz",".jpg",".jpeg",".pdf",".tex",".pyc"} or "__pycache__" in path.parts:
                issues.append(dict(file=str(path.relative_to(root)),reason="Unwanted public container"))
            if path.stat().st_size>20*1024*1024:issues.append(dict(file=path.name,reason="Artifact exceeds20MiB"))
            if path.suffix in {".png",".npy"}:continue
            try:
                if path.suffix==".gz":
                    with gzip.open(path,"rt",encoding="utf-8") as handle:text=handle.read()
                else:text=path.read_text()
            except UnicodeDecodeError:issues.append(dict(file=path.name,reason="Unexpected binary"));continue
            if re.search(r"/(?:home/minjae|dev/shm|tmp)(?:/|\b)",text):issues.append(dict(file=path.name,reason="Private absolute path"))
            if path.suffix==".json":walk(json.loads(text),path.name)
            elif path.name.endswith(".jsonl.gz"):
                for line in text.splitlines():walk(json.loads(line),path.name)
            elif path.suffix==".csv":
                for row in csv.DictReader(text.splitlines()):
                    for key,value in row.items():
                        if key and key.endswith("_json") and value:walk(json.loads(value),path.name)
            if path.suffix!=".md":continue
            fenced=False;columns=None
            for line_number,line in enumerate(text.splitlines(),1):
                if line.lstrip().startswith("```"):fenced=not fenced;columns=None;continue
                if fenced:continue
                if re.fullmatch(r"\s*\|(?:\s*:?-+:?\s*\|)+\s*",line):columns=len(re.findall(r"(?<!\\)\|",line));continue
                if columns is not None:
                    if not line.lstrip().startswith("|"):columns=None
                    else:
                        table_rows+=1;observed=len(re.findall(r"(?<!\\)\|",line))
                        if observed!=columns:issues.append(dict(file=path.name,line=line_number,reason="Markdown table delimiter count differs",expected=columns,actual=observed))
            for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)",text):
                if target.startswith(("http:","https:","mailto:")):continue
                target=target.split("#")[0].strip("<>")
                if not target:continue
                linked=(path.parent/target).resolve();exists=linked.exists()
                if not exists and linked.is_relative_to(root):
                    exists=subprocess.run(["git","-C",str(root),"cat-file","-e","HEAD:"+str(linked.relative_to(root))],capture_output=True).returncode==0
                item=dict(document=path.name,target=target,exists=exists)
                if not exists and linked.parent==doc and linked.name in deferred:
                    item["status"]="PENDING_DOWNSTREAM_PUBLICATION_AUDIT"
                    pending.append(item)
                elif not exists and linked==doc/"SUPPLEMENT_INDEPENDENT_VERIFICATION.json":
                    item["status"]="SELF_RECEIPT_WRITTEN_AFTER_THIS_CHECK"
                elif not exists:issues.append(dict(file=path.name,reason="Broken local Markdown link",target=target))
                links.append(item)
    issues.extend(dict(file=item["file"],reason="Reference-original field",key=item["key"]) for item in keys)
    return dict(status="PASS" if not issues else "FAIL",public_files_checked=file_count,issues=issues,
                forbidden_reference_original_keys=keys,Markdown_table_data_rows_checked=table_rows,local_Markdown_links=links,
                deferred_downstream_links=pending,downstream_closure_required_before_publication=True)


def verify(args):
    doc=Path(args.public_dir).resolve();root=doc.parents[2];source=Path(args.source_root);private=Path(args.private_dir)
    contract=read(doc/"INPUT_CONTRACT.json");ids=sorted(contract["allowed_ids"]);allowed=set(ids);protected=set(contract["excluded_protected_ids"])
    c_rows=rows(doc/"C_FRAME_METRICS.jsonl.gz");d_rows=rows(doc/"D_REPAIRED_ROWS.jsonl.gz");raw=rows(doc/"FRAME_METRICS.jsonl.gz")
    assert len(c_rows)==5190 and len(d_rows)==1730 and len(raw)==10726
    assert all(row["id"] in allowed and row["id"] not in protected for row in c_rows+d_rows+raw)
    assert len({row["id"] for row in raw if row["grade"]=="severe"})==63
    assert len({(row["backbone"],*identity(row)) for row in raw})==10726
    frame={identity(row):row for row in raw if row["backbone"]=="YOLO"}
    for records,kind in [(c_rows,"C"),(d_rows,"D")]:
        for row in records:
            normalized=frame[identity(row)];available,success,confusion=binary_pose(row)
            assert (available,success,confusion)==(normalized["available"],normalized["pose_success"],normalized["wd_confusion"])
            axis=row["axis_error"] if kind=="C" else row["axis_wrong"]
            assert axis==normalized["axis_wrong"]
    sessions=sorted({fid.split(":")[0] for fid in ids});draw=np.load(doc/"BOOTSTRAP_DRAWS.npy",allow_pickle=False)
    assert len(ids)==173 and len(sessions)==9 and draw.shape==(10000,9)
    assert np.array_equal(draw,np.random.default_rng(20260917).integers(0,9,size=(10000,9)))
    assert sha(doc/"BOOTSTRAP_DRAWS.npy")==read(doc/"BOOTSTRAP_LOCK.json")["sha256"]
    check=Comparator();primary=verify_primary(doc,frame,ids,sessions,draw,check)
    transitions=verify_transitions(doc,frame,c_rows,d_rows,ids,check)
    subgroups=verify_subgroups(doc,source,private,c_rows,ids,check)
    permutation=verify_permutation(doc,private,source,Path(args.published_root),ids,check)
    ratio=verify_ratio_correction(doc,root,source,Path(args.published_root),check)
    clean=verify_clean_preflight(doc,private,check)
    changed_b1=verify_changed_b1(doc,private,source,Path(args.published_root),check)
    changed_oldp=verify_changed_oldp(doc,private,source,Path(args.published_root),check)
    preserved=preservation(root,doc,source,Path(args.published_root))
    public=disclosure_and_markdown(root,doc)
    result=dict(status="PASS" if public["status"]=="PASS" else "STATISTICS_PASS_PUBLIC_AUDIT_FAIL",
                implementation="Independent math.fsum, sample variance, sorted linear quantiles, Fraction centered-bootstrap ties and Holm; no producer statistics import",
                frames=173,sessions=9,protected_scored=0,Severe_retained=63,
                original_C_rows=5190,original_D_rows=1730,original_deduplicated_rows=10726,
                scalar_comparisons=check.count,maximum_scalar_absolute_difference=check.maximum,
                requested_primary=primary,complete_transitions=transitions,C5_subgroups=subgroups,
                proper_permutation=permutation,source_and_original_result_preservation=preserved,
                A3_registered_axis_projection_correction=ratio,
                C_current_source_clean_replay_preflight=clean,
                B1_all_changed_candidate_traces=changed_b1,
                OLD_P_all_changed_candidate_traces_and_final_budget=changed_oldp,
                protocol_limitations=dict(global_C_D_common_split="FAIL: D fitting exposed C VAL253/TEST271; cross-task independent TEST superiority blocked",
                                          original_D_primary_route="Original pre-execution family used SubPix; named Stage1 supplement is post-execution alignment",
                                          common_preregistration_of_new_Stage1_family="FAIL: do not describe requested supplement as retroactively preregistered",
                                          C5_size_bins="NA: absent prelocked bin contract, no outcome-chosen bins",
                                          arithmetic_audit_status_distinct_from_experimental_validation=True),
                disclosure_and_Markdown=public,auditor_additional_PnP=0,auditor_additional_training=0,auditor_additional_model_forwards=0,
                inputs_sha256={name:sha(doc/name) for name in ["REQUESTED_PRIMARY_PAIRED.json","COMPARISON_TRANSITIONS_COMPLETE.csv",
                "C5_SUBGROUPS_SUPPLEMENT.json","FRAME_METRICS.jsonl.gz","C_FRAME_METRICS.jsonl.gz","D_REPAIRED_ROWS.jsonl.gz","BOOTSTRAP_DRAWS.npy"]})
    output=doc/args.output_name;assert not output.exists();output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":result["status"],"comparisons":check.count,"maximum_difference":check.maximum,
                      "primary_Stage1":[row for row in primary["exact_rational_results"] if row["after"]=="D_JOINT_DIM"],
                      "public_issues":public["issues"]},ensure_ascii=False))
    assert result["status"]=="PASS",public["issues"]


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for option in ["public-dir","source-root","published-root","private-dir"]:parser.add_argument("--"+option,required=True)
    parser.add_argument("--output-name",default="SUPPLEMENT_INDEPENDENT_VERIFICATION.json")
    args_global=parser.parse_args();verify(args_global)
