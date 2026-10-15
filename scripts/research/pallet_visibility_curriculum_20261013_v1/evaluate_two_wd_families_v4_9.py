"""Versioned CAL46 then TEST49 two-W/D predictions, sealed before scoring.

The original family reuses the sealed V48 no-prior solution only after exact
mathematical-input hash equality. The swapped family is freshly fitted using
the same current observations, covariance, state/S gates and O/T/U ranges.
No images, masks or labels are opened during prediction. A separate caller
can inspect all explicit-family feasible poses using global image evidence.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[_key]="1"

import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
from multiprocessing import get_context
from pathlib import Path
import shutil
import time

import numpy as np

from . import common as C
from . import statewise_evaluation_v4 as E
from . import evaluate_visibility_pose_v4_8 as PREVIOUS
from . import evaluate_shared_edge_partial_b_v4_7 as V47
from . import statewise_two_wd_family_v4_9 as FAMILY
from .statewise_partial_boundary_u_solver_v4_6 import solve_boundary_u_region_joint_ot_pnp
from .statewise_uncertainty_calibration_v4 import _reference_alignment
from .scoring import pose_metric

OUTPUT=PREVIOUS.P.BOUNDARY/"two_wd_family_eval_v4_9"
ARMS=("ORIGINAL_NO_PRIOR","SWAPPED_WD_NO_PRIOR","MIN_IMAGE_OBJECTIVE_WD")


def _load():
    rows,model,config,bindings,parent=PREVIOUS._load()
    directory=PREVIOUS.OUTPUT
    protocol=C.read(directory/"PROTOCOL.json")
    complete=C.read(directory/"PREDICTION_COMPLETE.json")
    if complete["rows"]!=95 or complete["severe"]!=0 or not complete["before_any_new_pose_scoring"]:
        raise ValueError("Complete sealed V48 primary95 required")
    E._verify_sources(protocol["source_sha256"])
    E._verify_sources(protocol["input_sha256"])
    E._verify_sources(protocol["protected_user_sha256"])
    cache={}
    for part,count in (("CAL",46),("TEST",49)):
        seal_path=directory/f"{part}_PREDICTION_SEAL.json"
        seal=C.read(seal_path)
        journal=directory/f"{part}_PREDICTIONS.jsonl.gz"
        if (seal["rows"]!=count or not seal["before_new_annotation_score"]
                or C.sha(journal)!=seal["predictions_sha256"]
                or C.sha(directory/"PROTOCOL.json")!=seal["protocol_sha256"]
                or C.sha(seal_path)!=complete["ordered_partition_prediction_seal_sha256"][part]):
            raise ValueError("V48 no-prior prediction seal mismatch")
        values=list(C.rows(journal))
        expected=[r["id"] for r in rows if r["partition"]==part]
        if [r["id"] for r in values]!=expected:
            raise ValueError("Same complete ordered partition required")
        for value in values:
            cache[value["id"]]=value
        for path in (seal_path,journal):
            bindings[str(path.relative_to(C.ROOT))]=C.sha(path)
    for path in (directory/"PROTOCOL.json",directory/"PREDICTION_COMPLETE.json"):
        bindings[str(path.relative_to(C.ROOT))]=C.sha(path)
    config=deepcopy(config)
    config["prior_weight"]=0.
    for row in rows:
        cached=cache[row["id"]]
        if (cached["states"]!=row["states"] or cached["frozen_BI_fit"]!=row["frozen_BI_fit"]
                or cached["partial_boundary_observation"]!=row["partial_boundary_observation"]
                or cached["u_region_observation"]!=row["u_region_observation"]):
            raise ValueError("Original-family frozen observation binding mismatch")
        spec=PREVIOUS._spec(row,config)
        if E._digest(spec)!=cached["no_prior_mathematical_input_sha256"]:
            raise ValueError("Original-family mathematical input differs from sealed V48")
        row["_cached_V48"]=cached
    return rows,model,config,bindings,protocol


def _family_finish(row,model,family,arm,solution=None):
    source=deepcopy(row["_original_source"])
    geometry=source["initial_geometry"]
    pose=family["coarse_pose"]
    geometry.update(native_xyz=deepcopy(family["native_xyz"]),cf_extents=deepcopy(family["cf_extents"]),
        physical_registration_Q=deepcopy(family["physical_registration_Q"]),
        R_cf=deepcopy(pose["R"]),centroid=deepcopy(pose["t"]),
        R_physical=(np.asarray(pose["R"])@np.asarray(family["physical_registration_Q"])).tolist(),
        initial_projection_xy=C.project(family["native_xyz"],pose["R"],pose["t"],geometry["K"]).tolist(),
        S_gate_origin="immutable_original_geometry_before_family_hypotheses",
        candidate_geometry_did_not_requery_or_reclassify_RGB=True)
    result=V47._finish(source,row["frozen_BI_fit"],model,family["solution"] if solution is None else solution,
        row["u_region_observation"]["u_regions"],arm,row["partial_boundary_observation"])
    if result["accepted"]:
        result["actual_pose"]["source"]="NO_PRIOR_EXPLICIT_WD_FAMILY_JOINT_SE3"
    result["WD_family"]=family["name"]
    result["original_initial_state_gate_preserved"]=True
    return result


def _run(task):
    row,model,config=task
    start=time.monotonic()
    spec=PREVIOUS._spec(row,config)
    original_hash=E._digest(spec)
    cached=row["_cached_V48"]["arms"][PREVIOUS.ARMS[1]]
    calls={"original_cache":0,"new_fits":0}
    def solver(xyz,K,states,visible,domains,hw,coarse,**kw):
        actual=dict(xyz=xyz,K=K,states=states,visible=visible,domains=domains,image_hw=hw,
            coarse_pose=coarse,initializations=kw.get("initializations"),config=kw["config"],
            boundary_observations=kw.get("boundary_observations"),u_regions=kw.get("u_regions"),
            boundary_factor_weight=1.)
        if E._digest(actual)==original_hash:
            calls["original_cache"]+=1
            return deepcopy(cached["solution"])
        calls["new_fits"]+=1
        return solve_boundary_u_region_joint_ot_pnp(xyz,K,states,visible,domains,hw,coarse,
            initializations=kw.get("initializations"),config=kw["config"],
            boundary_observations=kw.get("boundary_observations"),u_regions=kw.get("u_regions"))
    result=FAMILY.solve_two_wd_families(spec["xyz"],spec["K"],spec["states"],spec["visible"],
        spec["domains"],spec["image_hw"],spec["coarse_pose"],initializations=spec["initializations"],
        config=spec["config"],boundary_observations=spec["boundary_observations"],
        u_regions=spec["u_regions"],solver_callable=solver)
    if calls["original_cache"]!=1 or calls["new_fits"]!=len(result["families"])-1:
        raise ValueError("Original family must reuse its exact V48 input; all other families require fresh fits")
    if E._digest(spec)!=original_hash:
        raise ValueError("Family fitting modified immutable source observations")
    families=result["families"]
    arms={ARMS[0]:deepcopy(cached)}
    arms[ARMS[1]]=_family_finish(row,model,families[-1],ARMS[1])
    selected=families[result["selected_family_index"]]
    arms[ARMS[2]]=_family_finish(row,model,selected,ARMS[2],result["solution"])
    # Every feasible sample remains attached to its own explicit native basis.
    for family in families:
        family["solution"]=E._compact_solution(family["solution"])
    return C.finite(dict(id=row["id"],partition=row["partition"],session=row["session"],
        cohort=row["cohort"],image=row["image"],image_sha256=row["image_sha256"],states=row["states"],
        runtime=row["runtime"],frozen_BI_fit=row["frozen_BI_fit"],
        partial_boundary_observation=row["partial_boundary_observation"],u_region_observation=row["u_region_observation"],
        families=families,selected_family=result["selected_family"],family_selection=result["family_selection"],
        family_ambiguity=result["family_ambiguity"],selected_model_geometry=result["selected_model_geometry"],
        arms=arms,original_mathematical_input_sha256=original_hash,solver_calls=calls,
        contract=dict(prior_weight=0.,same_BI_q_observed_and_fit_covariances_and_reliability=True,
            same_partial_B_lines_and_OT_U_domains=True,S_RGB_queries=0,U_RGB_queries=0,
            source_initial_S_state_gate_fixed=True,known_physical_vertex_set_preserved=True,
            registration_Q_changes_explicit_and_proper=True,native_channels_not_GT_remapped=True,
            image_reads=0,GT_inputs=False),elapsed_seconds=time.monotonic()-start))


def predict(output=OUTPUT,workers=3):
    output=Path(output).resolve()
    if output.exists():
        raise FileExistsError("Preserve completed family evaluation")
    if workers not in (1,2,3):
        raise ValueError("One to three CPU1 workers required")
    rows,model,config,bindings,parent=_load()
    sources=deepcopy(parent["source_sha256"])
    for path in (Path(__file__),Path(FAMILY.__file__)):
        sources[str(path.relative_to(C.ROOT))]=C.sha(path)
    protected=parent["protected_user_sha256"]
    for value in (sources,bindings,protected):E._verify_sources(value)
    output.mkdir(parents=True)
    protocol=dict(schema="TWO_WD_FAMILY_SAME_OBSERVATION_EVAL_V4_9",arms=list(ARMS),
        ids=[r["id"] for r in rows],primary_population={"CAL":46,"TEST":49,"severe":0},
        source_sha256=sources,input_sha256=bindings,protected_user_sha256=protected,
        solver_configuration=config,family_rule="swap native cuboid width/depth; Qnew=Ry90@Qold; Rguess=Rold@Ry-90",
        family_selection="min unchanged image likelihood among feasible families; original exact ties",
        physical_size_and_initial_unordered_rendered_shape_preserved=True,
        original_family_cache="exact V48 mathematical input hash",new_fits_expected=95,
        original_observations_states_initial_S_and_CAl_weights_preserved=True,
        mask_or_GT_selection_in_prediction=False,prediction_order="CAL46 then TEST49 with CAL seal first",
        labels_accessed_only_after_all95_prediction_seals=True,workers=workers,CPU_threads_per_worker=1,
        TEST_is_reused_iterative_exploratory=True,all95_including_fallbacks_and_native_flagged_kept=True)
    C.write(output/"PROTOCOL.json",protocol)
    for relative,digest in sources.items():
        dest=output/"code_snapshot"/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(C.ROOT/relative,dest)
        if C.sha(dest)!=digest:raise ValueError("Source snapshot changed")
    started=time.monotonic();seals={};done={}
    with ProcessPoolExecutor(max_workers=workers,mp_context=get_context("spawn")) as pool:
        for partition,count in (("CAL",46),("TEST",49)):
            population=[r for r in rows if r["partition"]==partition]
            futures={pool.submit(_run,(r,model,config)):r["id"] for r in population}
            for future in as_completed(futures):
                value=future.result();done[value["id"]]=value
                C.write(output/f"{partition}_frames"/(value["id"].replace(":","__")+".json"),value)
                C.write(output/"LIVE_PROGRESS.json",dict(completed=len(done),total=95,
                    latest_id=value["id"],partition=partition,elapsed_seconds=time.monotonic()-started))
                print(f'{len(done)}/95 {value["id"]} family={value["selected_family"]}',flush=True)
            for value in (sources,bindings,protected):E._verify_sources(value)
            journal=output/f"{partition}_PREDICTIONS.jsonl.gz"
            C.save_rows(journal,[done[r["id"]] for r in population])
            if len(population)!=count:raise ValueError("Full partition count changed")
            C.write(output/f"{partition}_PREDICTION_SEAL.json",dict(rows=count,
                protocol_sha256=C.sha(output/"PROTOCOL.json"),predictions_sha256=C.sha(journal),
                source_sha256=sources,input_sha256=bindings,before_new_annotation_score=True,
                labels_read_in_prediction=False,partition=partition))
            seals[partition]=C.sha(output/f"{partition}_PREDICTION_SEAL.json")
    C.write(output/"PREDICTION_COMPLETE.json",dict(rows=95,CAL=46,TEST=49,severe=0,
        ordered_partition_prediction_seal_sha256=seals,before_any_new_pose_scoring=True,
        original_family_cache_reuses=sum(x["solver_calls"]["original_cache"] for x in done.values()),
        new_family_pose_fits=sum(x["solver_calls"]["new_fits"] for x in done.values()),
        elapsed_seconds=time.monotonic()-started))


def score(output=OUTPUT):
    output=Path(output).resolve();protocol=C.read(output/"PROTOCOL.json");complete=C.read(output/"PREDICTION_COMPLETE.json")
    if complete["rows"]!=95 or not complete["before_any_new_pose_scoring"]:
        raise ValueError("Both complete partitions must be sealed before labels")
    for key in ("source_sha256","input_sha256","protected_user_sha256"):E._verify_sources(protocol[key])
    parts={}
    for part,count in (("CAL",46),("TEST",49)):
        sealpath=output/f"{part}_PREDICTION_SEAL.json";seal=C.read(sealpath);journal=output/f"{part}_PREDICTIONS.jsonl.gz"
        if (seal["rows"]!=count or not seal["before_new_annotation_score"]
                or C.sha(journal)!=seal["predictions_sha256"] or C.sha(output/"PROTOCOL.json")!=seal["protocol_sha256"]
                or C.sha(sealpath)!=complete["ordered_partition_prediction_seal_sha256"][part]):
            raise ValueError("Prediction seal changed")
        parts[part]=list(C.rows(journal))
        expected=[r["id"] for r in C.rows(PREVIOUS.OUTPUT/f"{part}_PREDICTIONS.jsonl.gz")]
        if [r["id"] for r in parts[part]]!=expected:raise ValueError("All-frame identity changed")
    reports={}
    for part in ("CAL","TEST"):
        target=E.COHORT/("OOF_pose_v1/ISOLATED_CAL_SCORING.json" if part=="CAL" else "test_reused_v1/ISOLATED_TEST_SCORING.json")
        labels=C.read(target)["rows"];labels=labels if isinstance(labels,dict) else {x["id"]:x for x in labels}
        values=[]
        for row in parts[part]:
            label=labels[row["id"]];alignment=_reference_alignment(row,label)
            values.append(dict(id=row["id"],image=row["image"],states=row["states"],selected_family=row["selected_family"],
                metrics={a:pose_metric(row["arms"][a]["actual_pose"],label["target"]["truth"]) for a in ARMS},
                accepted={a:row["arms"][a]["accepted"] for a in ARMS},
                ambiguous={a:row["arms"][a]["ambiguity_audit"]["material_ambiguity"] for a in ARMS},
                physical_native_compatible=bool(alignment["compatible"]),geometry_gap_m=float(alignment["geometry_gap_m"]),
                family_objectives={f["name"]:f["objective"] for f in row["families"]}))
        report=dict(schema="POSTSEAL_TWO_WD_FAMILY_SCORE_V4_9",partition=part,severe=0,
            all_frames_include_fallbacks=True,summary_all={a:PREVIOUS._summary(values,a) for a in ARMS},
            summary_reference_native_compatible={a:PREVIOUS._summary([v for v in values if v["physical_native_compatible"]],a) for a in ARMS},
            swapped_selected_count=sum(v["selected_family"]=="SWAPPED_WD" for v in values),rows=values,
            target_binding=E._binding(target),source_prediction_seal=E._binding(output/f"{part}_PREDICTION_SEAL.json"),
            no_GT_family_choice=True,TEST_is_reused_iterative_exploratory=True)
        C.write(output/f"{part}_POSE_SCORE.json",report);reports[part]=report
    for key in ("source_sha256","input_sha256","protected_user_sha256"):E._verify_sources(protocol[key])
    C.write(output/"SCORE_SEAL.json",dict(CAL_score_sha256=C.sha(output/"CAL_POSE_SCORE.json"),
        TEST_score_sha256=C.sha(output/"TEST_POSE_SCORE.json"),
        prediction_complete_sha256=C.sha(output/"PREDICTION_COMPLETE.json"),all95_kept=True,
        TEST49_all_translation_le5cm=reports["TEST"]["summary_all"][ARMS[2]]["all_translation_le_5cm"]))
    print(C.finite({part:r["summary_all"] for part,r in reports.items()}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("command",choices=("predict","score"))
    parser.add_argument("--output",type=Path,default=OUTPUT);parser.add_argument("--workers",type=int,default=3)
    args=parser.parse_args()
    predict(args.output,args.workers) if args.command=="predict" else score(args.output)
