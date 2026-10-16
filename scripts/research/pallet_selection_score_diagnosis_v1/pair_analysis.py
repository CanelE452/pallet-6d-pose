"""Paired original-cost accounting on the already sealed candidate bank.

All numbers are joins/subtractions of immutable candidate costs and reference
metrics. No pose, measurement, optimizer, or alternative score is generated.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
import statistics

from .common import rows, read, save_rows, sha, timed_stage, write, write_csv
from .runtime_guard import FixedPoseOnly

TERMS = ("full_BI", "partial_normal", "outline", "prior")
ATOL = 1e-9


def require(condition, message):
    if not condition:
        raise ValueError(message)


def distribution(values):
    values = list(values)
    return dict(n=len(values), mean=statistics.mean(values) if values else None,
        sample_SD=statistics.stdev(values) if len(values)>1 else None,
        min=min(values) if values else None, max=max(values) if values else None)


def component_delta(selected, other):
    require(selected["stage"] == other["stage"], "Never subtract G10 and G13 objectives")
    delta = {term:other["weighted_components"][term]-selected["weighted_components"][term] for term in TERMS}
    total = other["objective"]-selected["objective"]
    difference = sum(delta.values())-total
    require(abs(difference)<=ATOL, "Pair weighted-component sum does not equal original objective difference")
    return delta, total, difference


def lineage_link(observation, ray_rows, stage):
    """Conditional source connection, without inventing semantic ownership.

    Full-corner links identify original attempted source rays only. For a
    fallback point those rays are explicitly NOT its measurement dependency.
    Partial factors additionally require the stored line normal/offset match.
    """
    corner = observation.get("native_corner")
    if corner is None:
        return dict(matches=[],status="GLOBAL_OUTLINE_HAS_NO_NATIVE_CORNER_SOURCE_RAY_LINK",
                    source_group_is_independence_proof=False)
    origin = ("original_e_typed_partial" if observation["loss_family"] == "partial_normal" and stage == "G13"
              and observation.get("association_provenance",{}).get("no_precise_2D_corner_or_neighbor_unknown_measurement_created")
              else "original_pre_e_BI_recovery")
    candidates = [r for r in ray_rows if r["native_corner"] == corner and r["origin"] == origin]
    matches = []
    for row in candidates:
        ray = row["ray"]
        matched_line = None
        if observation["loss_family"] == "partial_normal":
            for line in observation.get("lines",[]):
                normal, offset = ray.get("normal"), ray.get("offset")
                if normal is not None and offset is not None and len(normal)==2 and all(
                    abs(a-b)<=ATOL for a,b in zip(normal,line["normal_unit_xy"])) and abs(offset-line["offset_px"])<=ATOL:
                    matched_line = line["line_index"]
                    break
            if matched_line is None:
                continue
        matches.append(dict(native_corner=corner, origin=row["origin"],
            exact_sample_source_group=row.get("exact_sample_source_group"),
            original_edge=ray.get("edge"), native_neighbor=ray.get("native_neighbor"),
            line_index=matched_line, supported=ray.get("supported"),
            measurement_source=ray.get("measurement_source"),
            semantic_owner_status=row["semantic_owner_status"]))
    fallback = observation.get("inherited_predictor_fallback") is True
    return dict(matches=matches,
        status=("NO_RECORDED_RAY_MATCH_AT_NATIVE_CORNER_AND_ORIGIN" if not matches else
                "ATTEMPTED_RAYS_NOT_A_DEPENDENCY_OF_PREDICTOR_FALLBACK" if fallback else
                "NORMAL_OFFSET_AND_NATIVE_CORNER_ORIGIN_MATCH" if observation["loss_family"] == "partial_normal" else
                "NATIVE_CORNER_ORIGIN_MATCH_ONLY_COST_DEPENDENCY_NOT_CERTIFIED"),
        candidate_rays_at_native_corner_and_origin=len(candidates),
        source_group_is_independence_proof=False,
        semantic_physical_owner_verified=False)


def observation_deltas(selected, other, ray_rows):
    a = {(o["loss_family"],o["observation_id"]):o for o in selected["per_observation"]}
    b = {(o["loss_family"],o["observation_id"]):o for o in other["per_observation"]}
    require(len(a)==len(selected["per_observation"]) and len(b)==len(other["per_observation"]), "Unique original observation accounting IDs")
    result = []
    for key in sorted(a.keys() | b.keys()):
        s, g = a.get(key), b.get(key)
        sv = s["weighted_contribution"] if s is not None else 0.
        gv = g["weighted_contribution"] if g is not None else 0.
        outline = key[0] == "outline"
        alignment = ("DIRECTION_TOTALS_ONLY_DIFFERENT_PROJECTED_SAMPLES_NO_SAMPLE_INDEX_MATCH" if outline else
            "SAME_MEASUREMENT_CHANNEL_DIFFERENT_PHYSICAL_OWNER_HYPOTHESIS" if selected["family"] != other["family"] else
            "SAME_ORIGINAL_OBSERVATION_CHANNEL_AND_FAMILY")
        result.append(dict(loss_family=key[0],observation_id=key[1],
            native_corner=(s or g).get("native_corner"), state=(s or g).get("state"),
            weighted_contribution_s=sv, weighted_contribution_other=gv, delta=gv-sv,
            observation_alignment=alignment,
            observation_s=deepcopy(s), observation_other=deepcopy(g),
            source_lineage_s=None if s is None else lineage_link(s,ray_rows,selected["stage"]),
            source_lineage_other=None if g is None else lineage_link(g,ray_rows,other["stage"]),
            original_factor_absence_s=s is None,original_factor_absence_other=g is None,
            dense_outline_pointer_s=selected.get("dense_outline_stored_in") if outline else None,
            dense_outline_pointer_other=other.get("dense_outline_stored_in") if outline else None,
            hard_constraints_are_not_cost=True))
    return result


def make_pair(definition, selected, other, metrics, ray_rows):
    require(definition["id"] == selected["id"] == other["id"], "Pair exact frame binding")
    require(definition["stage"] == selected["stage"] == other["stage"], "Pair exact objective stage binding")
    deltas,total,error = component_delta(selected,other)
    observations = observation_deltas(selected,other,ray_rows)
    obs_error = {term:sum(o["delta"] for o in observations if o["loss_family"] == term)-deltas[term]
                 for term in TERMS if term != "prior"}
    require(all(abs(e)<=ATOL for e in obs_error.values()), "Observation-pair difference does not sum to component difference")
    positive = [k for k in TERMS if deltas[k]>ATOL]
    negative = [k for k in TERMS if deltas[k]<-ATOL]
    strongest = max(positive,key=lambda k:deltas[k]) if positive else None
    fallback = sum(o["delta"] for o in observations if o["loss_family"] == "full_BI" and
                   ((o["observation_s"] or {}).get("inherited_predictor_fallback") is True))
    outline_delta = {o["observation_id"].split(":",1)[1]:o["delta"] for o in observations if o["loss_family"] == "outline"}
    row = {**deepcopy(definition),"pair_id":definition["id"]+"/"+definition["stage"]+"/"+definition["pair_role"],
        "s_family":selected["family"],"other_family":other["family"],
        "objective_s":selected["objective"],"objective_other":other["objective"],"delta_total":total,
        "weighted_components_s":selected["weighted_components"],"weighted_components_other":other["weighted_components"],
        "component_deltas":deltas,"component_delta_sum_error":error,
        "observation_delta_sum_errors":obs_error,"positive_terms_prefer_s":positive,
        "negative_terms_prefer_other":negative,"largest_positive_term":strongest,
        "delta_full_BI_predictor_fallback":fallback,
        "outline_direction_deltas":outline_delta,
        "metric_s":metrics[(definition["id"],definition["s_candidate_id"])],
        "metric_other":metrics[(definition["id"],definition["other_candidate_id"])],
        "per_observation_deltas":observations,
        "constraints_s":selected["constraints"],"constraints_other":other["constraints"],
        "minimum_slack_s":selected["minimum_constraint_slack"],"minimum_slack_other":other["minimum_constraint_slack"],
        "full_2D_revoked_ids_s":selected["full_2D_revoked_ids"],"full_2D_revoked_ids_other":other["full_2D_revoked_ids"],
        "S_point_terms_s":selected["S_point_terms"],"S_point_terms_other":other["S_point_terms"],
        "U_point_terms_s":selected["U_point_terms"],"U_point_terms_other":other["U_point_terms"],
        "prior_weight_s":selected["actual_prior_weight"],"prior_weight_other":other["actual_prior_weight"],
        "term_status_s":selected["term_status"],"term_status_other":other["term_status"],
        "dense_outline_pointer_s":selected.get("dense_outline_stored_in"),
        "dense_outline_pointer_other":other.get("dense_outline_stored_in"),
        "source_cost_file":"cost_v1/CANDIDATE_COSTS.jsonl.gz",
        "source_metric_file":"metrics_v2/CANDIDATE_METRICS.jsonl.gz",
        "selection_replay_status":"PASS","candidate_cost_role":"POSTHOC_FIXED_ORIGINAL_CANDIDATE_COMPARISON",
        "new_selector_validation_performance":False}
    if definition["pair_role"] == "selected_vs_comparator_first_reference_failure_same_family":
        require(row["metric_s"]["success_le5"] is True and row["metric_other"]["success_le5"] is False,
                "Success control must compare actual reference success with a real reference failure")
    elif definition["pair_role"].startswith(("selected_vs_comparator_first_reference_success", "frozen_e_selected_vs_comparator_first_reference_success")):
        require(row["metric_s"]["success_le5"] is False and row["metric_other"]["success_le5"] is True,
                "Selection loss must compare actual reference failure with one jointly reference-successful pose")
    return row


def flat_pair(row):
    result = {key:row[key] for key in ("pair_id","id","partition","group","stage","pair_role", "s_candidate_id","other_candidate_id",
        "s_family","other_family","objective_s","objective_other","delta_total","largest_positive_term",
        "delta_full_BI_predictor_fallback","minimum_slack_s","minimum_slack_other",
        "full_2D_revoked_ids_s","full_2D_revoked_ids_other","positive_terms_prefer_s","negative_terms_prefer_other")}
    for term in TERMS:
        result["J_s_"+term]=row["weighted_components_s"][term]
        result["J_other_"+term]=row["weighted_components_other"][term]
        result["Delta_"+term]=row["component_deltas"][term]
    for name,metric in (("s",row["metric_s"]),("other",row["metric_other"])):
        result[name+"_translation_cm"]=metric["translation_cm"]
        result[name+"_rotation_deg"]=metric["rotation_deg"]
        result[name+"_reference_success"]=metric["success_le5"]
    result["Delta_outline_model_to_observed"]=row["outline_direction_deltas"].get("model_to_observed",0.)
    result["Delta_outline_observed_to_model"]=row["outline_direction_deltas"].get("observed_to_model",0.)
    result["component_delta_sum_error"]=row["component_delta_sum_error"]
    return result


def flat_observation(pair,obs):
    s,g = obs["observation_s"] or {},obs["observation_other"] or {}
    return dict(pair_id=pair["pair_id"],id=pair["id"],partition=pair["partition"],group=pair["group"],
        stage=pair["stage"],pair_role=pair["pair_role"],s_candidate_id=pair["s_candidate_id"],other_candidate_id=pair["other_candidate_id"],
        loss_family=obs["loss_family"],observation_id=obs["observation_id"],native_corner=obs["native_corner"],state=obs["state"],
        weighted_contribution_s=obs["weighted_contribution_s"],weighted_contribution_other=obs["weighted_contribution_other"],Delta=obs["delta"],
        measurement_source_s=s.get("measurement_source"),measurement_source_other=g.get("measurement_source"),
        predictor_fallback_s=s.get("inherited_predictor_fallback"),predictor_fallback_other=g.get("inherited_predictor_fallback"),
        active_s=s.get("active"),active_other=g.get("active"),count_s=s.get("count"),count_other=g.get("count"),
        normalizer_s=s.get("normalizer"),normalizer_other=g.get("normalizer"),
        raw_radius_px_s=s.get("raw_radius_px"),raw_radius_px_other=g.get("raw_radius_px"),
        reliability_weight_s=s.get("reliability_weight"),reliability_weight_other=g.get("reliability_weight"),
        covariance_s=s.get("fit_covariance_px2"),covariance_other=g.get("fit_covariance_px2"),
        source_lineage_s=obs["source_lineage_s"],source_lineage_other=obs["source_lineage_other"],
        observation_alignment=obs["observation_alignment"],
        dense_outline_pointer_s=obs["dense_outline_pointer_s"],dense_outline_pointer_other=obs["dense_outline_pointer_other"],
        complete_details_pointer="PAIR_LEDGER.jsonl.gz#"+pair["pair_id"]+"/"+obs["observation_id"])


def summaries(pairs):
    result = {}
    for cohort in ("PRIMARY","TEST","REMAINDER","CAL"):
        subset=[p for p in pairs if (p["partition"]!="CAL" if cohort=="PRIMARY" else p["partition"]==cohort)]
        groups={}
        for key in sorted({(p["group"],p["stage"],p["pair_role"]) for p in subset}):
            group,stage,role=key
            values=[p for p in subset if (p["group"],p["stage"],p["pair_role"])==key]
            groups["/".join(key)]=dict(images=len({p["id"] for p in values}),pairs=len(values),
                delta_total=distribution(p["delta_total"] for p in values),
                component_delta={term:distribution(p["component_deltas"][term] for p in values) for term in TERMS},
                largest_positive_term_counts=dict(Counter(str(p["largest_positive_term"]) for p in values)),
                terms_preferring_s={term:sum(p["component_deltas"][term]>ATOL for p in values) for term in TERMS},
                terms_preferring_other={term:sum(p["component_deltas"][term]<-ATOL for p in values) for term in TERMS},
                full_BI_predictor_fallback_delta=distribution(p["delta_full_BI_predictor_fallback"] for p in values),
                retained_candidate_pair_not_independent_image_replicates=True)
        result[cohort]=groups
    return result


def run(folder,old_root,output):
    folder,old_root,output=map(Path,(folder,old_root,output))
    if output.exists():raise FileExistsError("Preserve every previous pair analysis attempt")
    output.mkdir(parents=True)
    with timed_stage(folder,"fixed_original_candidate_pair_accounting",new_pose_solve_calls=0,new_model_forwards=0) as timing:
        with FixedPoseOnly() as guard:
            frames=list(rows(folder/"selection_v1/FRAME_SELECTIONS.jsonl.gz"))
            frame_by_id={f["id"]:f for f in frames}
            require(len(frame_by_id)==238 and all(f["selection_replay_status"]=="PASS" for f in frames), "Full238 selection gate")
            definitions=list(rows(folder/"selection_v1/PAIR_DEFINITIONS.jsonl.gz"))
            metrics={(r["id"],r["candidate_id"]):r["metric"] for r in rows(folder/"metrics_v2/CANDIDATE_METRICS.jsonl.gz")}
            meta={}
            by_frame_family=defaultdict(list)
            for c in rows(folder/"selection_v1/RETAINED_CANDIDATES.jsonl.gz"):
                value={k:c[k] for k in ("id","candidate_id","stage","family","source_order","reference_success","original_outcome_order")}
                value.update(initialization=c["original_record"]["initialization"],objective=c["original_record"]["objective"])
                meta[(c["id"],c["candidate_id"])]=value
                by_frame_family[(c["id"],c["stage"],c["family"])].append(value)
            require(len(meta)==len(metrics)==11603,"Exactly11603 original retained cost/metric records")
            for f in frames:
                if f["group"]=="FAMILY_SELECTION_LOSS":
                    require(f["selected_candidate_id"] is not None and f["representative_success_candidate_id"] is not None,"Family loss real saved pair")
                    definitions.append(dict(id=f["id"],partition=f["partition"],group=f["group"],stage="G13",family=f["selected_family"],
                        s_candidate_id=f["selected_candidate_id"],other_candidate_id=f["representative_success_candidate_id"],
                        pair_role="frozen_e_selected_vs_comparator_first_reference_success_other_family",
                        same_stage_family_specific_objective_packages=True, cross_stage_values_compared=False))
            needed={(p["id"],p[k]) for p in definitions for k in ("s_candidate_id","other_candidate_id")}
            costs={}; scalar_rows=[]
            for c in rows(folder/"cost_v1/CANDIDATE_COSTS.jsonl.gz"):
                key=c["id"],c["candidate_id"]
                require(key in meta and c["stored_objective_parity"] is True and c["stored_component_parity"] is True,"Every source fixed-cost parity required")
                require(meta[key]["reference_success"] is metrics[key]["success_le5"],"Sealed oracle and fixed-pose metric candidate success must agree")
                if key in needed:costs[key]=c
                for term in TERMS:
                    scalar_rows.append(dict(id=c["id"],partition=c["partition"],session=c["session"],stage=c["stage"],family=c["family"],
                        candidate_id=c["candidate_id"],initialization=meta[key]["initialization"],retained_rank=meta[key]["source_order"],
                        original_outcome_order=meta[key]["original_outcome_order"],loss_family=term,
                        weighted_component=c["weighted_components"][term],objective=c["objective"],
                        active=c["term_status"][term]["active"],reason=c["term_status"][term]["reason"],
                        T_cm=metrics[key]["translation_cm"],R_deg=metrics[key]["rotation_deg"],reference_success=metrics[key]["success_le5"],
                        minimum_constraint_slack=c["minimum_constraint_slack"],full_2D_revoked_ids=c["full_2D_revoked_ids"],
                        S_point_terms=c["S_point_terms"],U_point_terms=c["U_point_terms"],actual_prior_weight=c["actual_prior_weight"],
                        dense_outline_pointer=c.get("dense_outline_stored_in")))
            require(set(costs)==needed,"Every pair points to a costed original retained candidate")
            lineage={r["id"]:r["rays"] for r in rows(old_root/"e_trace_v2/OBSERVATION_LINEAGE.jsonl.gz")}
            pairs=[make_pair(p,costs[(p["id"],p["s_candidate_id"])],costs[(p["id"],p["other_candidate_id"])],metrics,lineage[p["id"]]) for p in definitions]
            require(len({p["pair_id"] for p in pairs})==len(pairs),"Unique role/stage/frame pair IDs")
            flat=[flat_pair(p) for p in pairs]
            write_csv(output/"SELECTION_PAIR_TABLE.csv",flat)
            save_rows(output/"PAIR_LEDGER.jsonl.gz",pairs)
            write_csv(output/"PER_OBSERVATION_PAIR_LEDGER.csv.gz",(flat_observation(p,o) for p in pairs for o in p["per_observation_deltas"]))
            write_csv(output/"CANDIDATE_COMPONENTS_ALL.csv.gz",scalar_rows)
            cases=[]
            for f in frames:
                if f["group"] not in ("FAMILY_SELECTION_LOSS","WITHIN_FAMILY_SELECTION_LOSS","FINAL_SUCCESS_CONTROL"):continue
                available=[p for p in pairs if p["id"]==f["id"]]
                mainrole=("selected_vs_comparator_first_reference_failure_same_family" if f["group"]=="FINAL_SUCCESS_CONTROL" else
                    "selected_vs_comparator_first_reference_success_same_family" if f["group"]=="WITHIN_FAMILY_SELECTION_LOSS" else
                    "frozen_e_selected_vs_comparator_first_reference_success_other_family")
                main=next((p for p in available if p["pair_role"]==mainrole),None)
                cid=f["representative_failure_candidate_id"] if f["group"]=="FINAL_SUCCESS_CONTROL" else f["representative_success_candidate_id"]
                family=(meta[(f["id"],cid)]["family"] if cid is not None else f["selected_family"])
                pool=by_frame_family[(f["id"],"G13",family)]
                # Preserve ALL reference-successful candidates and their actual
                # comparator ranks/costs. Controls additionally preserve every
                # eligible reference-failure candidate in their selected family.
                good=[c for c in pool if c["reference_success"]]
                bad=[c for c in pool if not c["reference_success"]]
                target=bad if f["group"]=="FINAL_SUCCESS_CONTROL" else good
                case=dict(id=f["id"],partition=f["partition"],session=f["session"],group=f["group"],
                    selected_family=f["selected_family"],g13_reselected_family=f["g13_reselected_family"],
                    selected_candidate_id=f["selected_candidate_id"],representative_other_candidate_id=cid,
                    pair_status="PASS" if main is not None else "NO_ELIGIBLE_REFERENCE_FAILURE_CANDIDATE",
                    family_reference_diagnosis=f["family_reference_diagnosis"],
                    all_selected_family_reference_success_ids=f["reference_success_candidates_g13_selected_family"],
                    all_G13_reference_success_ids=f["reference_success_candidates_g13_all"],
                    representative_pool_family=family,representative_pool_count=len(pool),
                    successful_candidate_count=len(good),failed_candidate_count=len(bad),
                    target_candidate_ids=[c["candidate_id"] for c in target],
                    target_original_comparator_ranks=[c["source_order"] for c in target],
                    target_original_costs=[c["objective"] for c in target],
                    target_cost_min=min((c["objective"] for c in target),default=None),
                    target_cost_max=max((c["objective"] for c in target),default=None),
                    G10_actual_family_pair_pointer=f["id"]+"/G10/accepted_family_winner_comparison",
                    G13_pair_pointer=None if main is None else main["pair_id"])
                if main is not None:case.update({k:v for k,v in flat_pair(main).items() if k not in case})
                cases.append(case)
            write_csv(output/"FAMILY10_CASES.csv",[c for c in cases if c["partition"]!="CAL" and c["group"]=="FAMILY_SELECTION_LOSS"])
            write_csv(output/"WITHIN22_CASES.csv",[c for c in cases if c["partition"]!="CAL" and c["group"]=="WITHIN_FAMILY_SELECTION_LOSS"])
            write_csv(output/"SUCCESS114_CONTROLS.csv",[c for c in cases if c["partition"]!="CAL" and c["group"]=="FINAL_SUCCESS_CONTROL"])
            cal_cases=[c for c in cases if c["partition"]=="CAL"]
            for f in frames:
                if f["partition"]=="CAL" and f["group"]=="NO_SUCCESS_CANDIDATE":
                    cal_cases.append(dict(id=f["id"],partition="CAL",session=f["session"],group=f["group"],
                        selected_family=f["selected_family"],g13_reselected_family=f["g13_reselected_family"],
                        selected_candidate_id=f["selected_candidate_id"],representative_other_candidate_id=None,
                        pair_status="NO_REFERENCE_SUCCESS_CANDIDATE_SEE_NO_CANDIDATE_LEDGER",
                        family_reference_diagnosis=f["family_reference_diagnosis"],
                        all_selected_family_reference_success_ids=[],all_G13_reference_success_ids=[]))
            require(len(cal_cases)==46,"Preserve CAL46 including six no-success-candidate cases")
            write_csv(output/"CAL_CASES.csv",cal_cases)
            summary=summaries(pairs)
            write(output/"PAIR_SUMMARY.json",dict(cohorts=summary,
                interpretation="Candidate selection cost contribution; not causal attribution or new-selector performance",
                Delta_sign="other minus actual selected: positive prefers actual selected, negative prefers reference diagnostic comparator",
                original_weight_and_hard_guards_preserved=True,semantic_owner_verified=False,
                source_groups_are_dependency_diagnostics_not_independence_proofs=True,
                no_cross_stage_objective_subtraction=True))
            guard_receipt=guard.receipt()
    check=dict(schema="ORIGINAL_FIXED_CANDIDATE_PAIR_CHECK_V1",all_passed=True,
        candidate_scalar_component_rows=len(scalar_rows),original_candidate_rows=len(meta),
        pair_count=len(pairs),per_observation_pair_rows=sum(len(p["per_observation_deltas"]) for p in pairs),
        maximum_component_delta_sum_error=max(abs(p["component_delta_sum_error"]) for p in pairs),
        maximum_observation_delta_sum_error=max(abs(v) for p in pairs for v in p["observation_delta_sum_errors"].values()),
        PRIMARY_case_counts=dict(Counter(c["group"] for c in cases if c["partition"]!="CAL")),
        CAL_case_counts=dict(Counter(c["group"] for c in cases if c["partition"]=="CAL")),
        controls_without_eligible_bad_ids=[c["id"] for c in cases if c["group"]=="FINAL_SUCCESS_CONTROL" and c["pair_status"]!="PASS"],
        partial_ray_lineage_links=dict(Counter(o["source_lineage_"+side]["status"]
            for p in pairs for o in p["per_observation_deltas"] if o["loss_family"]=="partial_normal"
            for side in ("s","other"))),
        atol=ATOL,rtol=0.,source_sha256=sha(Path(__file__)),runtime_guard=guard_receipt,**timing,
        input_sha256={str(p.relative_to(folder)):sha(p) for p in (folder/"cost_v1/CANDIDATE_COSTS.jsonl.gz",
            folder/"selection_v1/FRAME_SELECTIONS.jsonl.gz",folder/"selection_v1/RETAINED_CANDIDATES.jsonl.gz",
            folder/"selection_v1/PAIR_DEFINITIONS.jsonl.gz",folder/"metrics_v2/CANDIDATE_METRICS.jsonl.gz")},
        artifacts={p.name:sha(p) for p in sorted(output.iterdir())})
    write(output/"PAIR_CHECK.json",check)
    return check


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder",required=True,type=Path)
    parser.add_argument("--old-root",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    value=run(args.folder,args.old_root,args.output)
    print({k:value[k] for k in ("all_passed","pair_count","original_candidate_rows","maximum_component_delta_sum_error","maximum_observation_delta_sum_error","elapsed_seconds")})
