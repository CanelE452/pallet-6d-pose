"""Independent original-outcome/endpoint coverage audit, prediction inputs only.

Does not import the input producer, solver, evaluator or metric modules. Raw
native source R/t and source family registrations independently reconstruct
the expected carried physical poses. No oracle/reference/accuracy rows open.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np


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


def physical_pose(record,family):
    """Source availability is audited independently of producer helper logic."""
    if not isinstance(record,dict):return None,"ABSENT_RECORD"
    native_R=record.get("R_cf") if "R_cf" in record else record.get("R")
    translation=record.get("native_t") if "native_t" in record else record.get("t")
    center=record.get("centroid")
    if native_R is None:return None,"NO_NATIVE_ROTATION"
    rotation=np.asarray(native_R,float)
    if rotation.shape!=(3,3):return None,"BAD_NATIVE_ROTATION_SHAPE"
    if center is None and translation is None:return None,"NO_NATIVE_TRANSLATION_OR_CENTER"
    xyz=np.asarray(family["native_xyz"],float)
    registration=np.asarray(family["physical_registration_Q"],float)
    center=np.asarray(center,float) if center is not None else rotation@xyz.mean(axis=0)+np.asarray(translation,float)
    # Keep an R/t-bearing source even when its optimizer/feasibility flag is
    # false. A false explicit pose-availability flag is recorded separately.
    return (rotation@registration,center),"DECLARED_UNAVAILABLE_BUT_RT_PRESENT" if record.get("available") is False else "RT_PRESENT"


def producer_static_checks(path):
    tree=ast.parse(Path(path).read_text())
    keys=[];calls=[];imports=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Subscript) and isinstance(node.slice,ast.Constant) and isinstance(node.slice.value,str):keys.append(node.slice.value)
        if isinstance(node,ast.Call):
            if isinstance(node.func,ast.Name):calls.append(node.func.id)
            elif isinstance(node.func,ast.Attribute):calls.append(node.func.attr)
        if isinstance(node,(ast.Import,ast.ImportFrom)):
            imports.append(ast.unparse(node))
    return dict(direct_label_or_oracle_field_access=sorted(set(keys)&{
                    "reference","references","metrics","reference_success","joint_success_candidate_ids","oracle_rank","success"}),
                rank_selection_calls=sorted(set(calls)&{"sort","sorted","argsort","argmin","argmax"}),
                imports=imports,old_objective_field_access=sorted(set(keys)&{"objective","original_objective","original_rank","score"}))


def audit(old_private,new_input_dir,*,output,expected_input_sha256=None):
    started,cpu=time.perf_counter(),time.process_time()
    old_private,new_input_dir,output=map(Path,(old_private,new_input_dir,output))
    paths=dict(inputs=new_input_dir/"PREDICTION_INPUTS238.jsonl.gz",binding=new_input_dir/"INPUT_BINDING.json",
               bank=old_private/"original_candidate_archive_v1/ORIGINAL_E_CANDIDATE_BANK.jsonl.gz",
               bank_manifest=old_private/"original_candidate_archive_v1/CANDIDATE_MANIFEST.json",
               endpoints=old_private/"GENERATED_CANDIDATES.jsonl.gz",endpoint_seal=old_private/"CANDIDATE_SEAL.json",
               baselines=old_private/"score_v2/BASELINE_PREDICTIONS.jsonl.gz",
               baseline_seal=old_private/"score_v2/SCORE_SEAL.json")
    digest={name:sha(path) for name,path in paths.items()}
    binding=read(paths["binding"]);bank_manifest=read(paths["bank_manifest"]);endpoint_seal=read(paths["endpoint_seal"])
    baseline_seal=read(paths["baseline_seal"])
    if expected_input_sha256 is not None and digest["inputs"]!=expected_input_sha256:raise ValueError("Requested repaired input SHA differs")
    if binding["output_sha256"]!=digest["inputs"]:raise ValueError("Repaired input binding differs")
    if bank_manifest["artifacts"][paths["bank"].name]!=digest["bank"]:raise ValueError("Original candidate bank seal differs")
    if endpoint_seal["generated_candidates_sha256"]!=digest["endpoints"]:raise ValueError("Logged endpoint seal differs")
    if baseline_seal["artifacts"][paths["baselines"].name]!=digest["baselines"]:raise ValueError("Original prediction seal differs")
    for key in ("bank","endpoints","baselines"):
        if binding["inputs_sha256"][str(paths[key])]!=digest[key]:raise ValueError("Repaired input names another source: "+key)
    counts=Counter();failures=Counter();examples=[];frame_results=[];ids=[];partitions=Counter()
    max_numeric=0.;thresholds=Counter();nonpositive=[];proper_invalid=[];missing_rt=Counter()
    def fail(kind,ident,**detail):
        failures[kind]+=1
        if len(examples)<30:examples.append(dict(kind=kind,id=ident,**detail))
    streams=[rows(paths[k]) for k in ("inputs","bank","endpoints","baselines")]
    for frame,bank,endpoints,baseline in zip(*streams,strict=True):
        ident=frame["id"];ids.append(ident);partitions[frame["partition"]]+=1
        if len({ident,bank["id"],endpoints["id"],baseline["id"]})!=1:
            fail("SOURCE_ORDER_ID_MISMATCH",ident);continue
        if bank.get("GT_reference_or_success_inputs") is not False or endpoints.get("GT_inputs") is not False:
            fail("SOURCE_NOT_PREDICTION_ONLY",ident)
        expected={};source_counter=Counter()
        def add(cid,source,record,family):
            pose,status=physical_pose(record,family)
            if pose is None:
                missing_rt[source+":"+status]+=1;return
            if status!="RT_PRESENT":fail(status,ident,candidate_id=cid)
            if cid in expected:fail("DUPLICATED_EXPECTED_CANDIDATE_ID",ident,candidate_id=cid)
            expected[cid]=(source,*pose)
            source_counter[source]+=1
        for stage in ("G10","G13"):
            for family in bank["stages"][stage]["families"]:
                for sample in family["feasible_samples"]:
                    add(f"{stage}/{family['name']}/sample/{sample['original_index']}","ORIGINAL_"+stage,sample["pose"],family)
                for role in ("selected_solver_pose","published_actual_pose"):
                    add(f"{stage}/{family['name']}/{role}","ORIGINAL_"+stage,family.get(role),family)
                for outcome in family["original_outcomes"]:
                    add(f"{stage}/{family['name']}/actual_outcome/{outcome['original_index']}","ORIGINAL_"+stage+"_OUTCOME",outcome.get("original_record"),family)
                    counts[stage+"_original_outcomes_total"]+=1
        family_by_name=frame["families"]
        for n,endpoint in enumerate(endpoints["candidates"]):
            stage,name,kind=endpoint["stage"],endpoint["family"],endpoint["kind"]
            add(f"LOGGED/{stage}/{name}/{kind}/{n}","EXISTING_LOGGED_"+stage+"_ENDPOINT",
                {"R":endpoint["R"],"t":endpoint["t"]},family_by_name[name])
            counts["logged_endpoint_records_total"]+=1
        for arm in ("N0","E0"):
            pose=baseline["predictions"][arm]["actual_pose"]
            if pose.get("available") is not True:fail("MANDATORY_BASELINE_POSE_UNAVAILABLE",ident,arm=arm);continue
            expected[arm]=(arm,np.asarray(pose["R_physical"],float),np.asarray(pose["centroid"],float))
            source_counter[arm]+=1
        archived=frame.get("candidate_archive_all",[]);by_id={c["candidate_id"]:c for c in archived}
        if len(by_id)!=len(archived):fail("DUPLICATED_ARCHIVE_ID",ident)
        missing=set(expected)-set(by_id);extra=set(by_id)-set(expected)
        for cid in sorted(missing):fail("MISSING_SOURCE_POSE",ident,candidate_id=cid)
        for cid in sorted(extra):fail("EXTRA_UNSOURCED_ARCHIVE_POSE",ident,candidate_id=cid)
        admitted=[];rejected=Counter()
        threshold=frame["packet"]["solver_input"]["config"].get("positive_depth_m",1e-5)
        thresholds[str(threshold)]+=1
        if threshold!=1e-5:fail("POSITIVE_DEPTH_THRESHOLD_NOT_ORIGINAL_1E_MINUS5",ident,value=threshold)
        original=family_by_name["ORIGINAL_WD"]
        xyz=np.asarray(original["native_xyz"],float);Q=np.asarray(original["physical_registration_Q"],float)
        # Independently form physical model vertices once, then project each
        # physical rotation; producer uses the inverse native-R formulation.
        physical_vertices=(xyz-xyz.mean(axis=0))@Q
        for cid,(source,R,center) in expected.items():
            if cid not in by_id:continue
            c=by_id[cid];actual_R=np.asarray(c["R_physical"],float);actual_center=np.asarray(c["centroid"],float)
            if actual_R.shape!=R.shape or actual_center.shape!=center.shape:
                fail("POSE_SHAPE_MISMATCH",ident,candidate_id=cid);continue
            difference=max(float(np.max(np.abs(actual_R-R))),float(np.max(np.abs(actual_center-center))))
            max_numeric=max(max_numeric,difference)
            if difference>1e-12:fail("SOURCE_PHYSICAL_POSE_MISMATCH",ident,candidate_id=cid,max_abs=difference)
            if c["source"]!=source:fail("SOURCE_ROLE_MISMATCH",ident,candidate_id=cid)
            proper=bool(R.shape==(3,3) and center.shape==(3,) and np.isfinite(R).all() and np.isfinite(center).all()
                        and np.max(np.abs(R.T@R-np.eye(3)))<=1e-6 and abs(np.linalg.det(R)-1)<1e-6)
            depths=(physical_vertices@R.T+center)[:,2] if proper else np.array([-np.inf])
            good=proper and bool(np.min(depths)>1e-5)
            reason=None if good else "INVALID_NONFINITE_OR_NONPROPER_POSE" if not proper else "NONPOSITIVE_MODEL_VERTEX_DEPTH"
            if c.get("initial_projection_admitted") is not good or c.get("initial_projection_rejection_reason")!=reason:
                fail("PROJECTION_ADMISSION_OR_REASON_MISMATCH",ident,candidate_id=cid,expected_admitted=good,expected_reason=reason)
            if good:admitted.append(cid)
            else:
                rejected[reason]+=1
                (nonpositive if proper else proper_invalid).append(dict(id=ident,candidate_id=cid,min_depth_m=float(np.min(depths)) if proper else None))
            extra_keys=set(c)-{"candidate_id","source","R_physical","centroid","original_native_family","initial_projection_admitted","initial_projection_rejection_reason"}
            if extra_keys:fail("UNEXPECTED_CANDIDATE_INPUT_FIELDS",ident,candidate_id=cid,keys=sorted(extra_keys))
        actual_admitted=frame.get("candidates",[])
        if [c["candidate_id"] for c in actual_admitted]!=admitted:fail("ADMITTED_SEED_ORDER_OR_COVERAGE_MISMATCH",ident)
        for c in actual_admitted:
            if c!=by_id.get(c["candidate_id"]):fail("ADMITTED_SEED_DIFFERS_FROM_ARCHIVED_POSE",ident,candidate_id=c["candidate_id"])
        for arm in ("N0","E0"):
            if arm not in by_id:fail("MANDATORY_BASELINE_NOT_ARCHIVED",ident,arm=arm)
        if frame.get("supervised_inputs") is not False:fail("INPUT_SUPERVISION_FLAG_NOT_FALSE",ident)
        counts.update(source_counter)
        counts["archived_source_pose_roles"]+=len(expected);counts["actual_archived_pose_roles"]+=len(archived)
        counts["projection_admitted"]+=len(admitted);counts["projection_ineligible_preserved"]+=sum(rejected.values())
        frame_results.append(dict(id=ident,partition=frame["partition"],source_roles=dict(source_counter),
            expected_archive_count=len(expected),actual_archive_count=len(archived),admitted_count=len(admitted),
            missing_source_count=len(missing),extra_pose_count=len(extra),rejection_counts=dict(rejected)))
    if ids!=binding["ordered_ids"] or ids!=bank_manifest["ordered_ids"]:fail("SEALED_ORDERED_POPULATION_DIFFERS","ALL")
    if len(ids)!=238 or len(set(ids))!=238 or partitions!=Counter(CAL=46,TEST=49,REMAINDER=143):fail("FULL238_POPULATION_NOT_PRESERVED","ALL")
    producer=Path(__file__).with_name("inputs.py");static=producer_static_checks(producer)
    if any(static[k] for k in ("direct_label_or_oracle_field_access","rank_selection_calls","old_objective_field_access")):
        fail("PRODUCER_LABEL_OR_RANK_FIELD_ACCESS","SOURCE",details=static)
    report=dict(schema="INDEPENDENT_PREDICTION_INPUT_COVERAGE_V1",status="PASS" if not failures else "FAIL",
        independent_of_input_producer_and_solver=True,expected_input_sha256=expected_input_sha256,input_sha256=digest["inputs"],
        counts=dict(counts),source_missing_Rt_reasons=dict(missing_rt),frame_count=len(ids),partition_counts=dict(partitions),
        failure_count=sum(failures.values()),failure_counts=dict(failures),failure_examples=examples,
        all_original_available_Rt_roles_and_logged_endpoints_retained=not any(failures[k] for k in
            ("MISSING_SOURCE_POSE","SOURCE_PHYSICAL_POSE_MISMATCH","SOURCE_ORDER_ID_MISMATCH")),
        all_mandatory_N0_E0_poses_archived=not any(failures[k] for k in ("MANDATORY_BASELINE_NOT_ARCHIVED","MANDATORY_BASELINE_POSE_UNAVAILABLE")),
        original_positive_depth_threshold_m=1e-5,observed_threshold_counts=dict(thresholds),
        max_absolute_physical_pose_bridge_difference=max_numeric,
        nonpositive_projection_poses_preserved=nonpositive,invalid_projection_poses_preserved=proper_invalid,
        candidate_pose_deduplication_performed=False,source_cost_ranking_performed=False,
        original_optimizer_or_feasibility_failure_used_to_exclude_Rt=False,
        source_checks=static,PRIMARY_metrics_or_reference_opens=0,oracle_files_opened=0,
        numeric_read_allowlist=[str(paths[k]) for k in ("inputs","bank","endpoints","baselines")],
        input_bindings={str(paths[k]):v for k,v in digest.items()},producer_source_sha256=sha(producer),
        auditor_source_sha256=sha(__file__),rows=frame_results,
        wall_seconds=time.perf_counter()-started,process_cpu_seconds=time.process_time()-cpu,
        process_peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--old-private",required=True)
    p.add_argument("--new-input-dir",required=True);p.add_argument("--output",required=True);p.add_argument("--expected-input-sha256")
    a=p.parse_args();r=audit(a.old_private,a.new_input_dir,output=a.output,expected_input_sha256=a.expected_input_sha256)
    print(json.dumps({k:r[k] for k in ("status","frame_count","counts","failure_count","failure_counts","max_absolute_physical_pose_bridge_difference","wall_seconds")},indent=2))
