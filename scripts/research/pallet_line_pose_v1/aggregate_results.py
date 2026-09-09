"""Strict aggregation of completed real paper evaluations; never run inference.

The primary arm is preregistered image_joint. Pooled keypoint medians/P90 and
MAIN pose statistics are recomputed using the evaluator definitions. Seed
statistics, rather than independent seed/corner observations, are averaged.
Paired resampling uses the same frames or complete sessions for all three seeds.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ARMS = ("image_joint", "geometry_joint", "image_line_only")
SEEDS = (1, 2, 3)
KEYPOINT_METRICS = ("keypoint_location_median_px", "keypoint_location_p90_px")
POSE_METRICS = ("rotation_median_deg", "translation_median_cm", "iou3d_median", "add_sym_auc")
METRICS = KEYPOINT_METRICS + POSE_METRICS
LOWER = set(KEYPOINT_METRICS + ("rotation_median_deg", "translation_median_cm"))
POSE_FIELDS = dict(rotation_median_deg="rotation_error_deg", translation_median_cm="translation_error_cm",
                   iou3d_median="iou3d", add_sym_auc="add_sym_m")
N_RESAMPLES = 10000


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda:handle.read(1048576),b""):
            value.update(block)
    return value.hexdigest()


def write(path,value):
    temporary = Path(path).with_suffix(".pending.json")
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    temporary.replace(path)


def bound(path):
    path = Path(path).resolve()
    return dict(path=str(path),sha256=sha(path))


def checked(path, digest):
    if sha(path) != digest:
        raise ValueError(f"Artifact SHA changed: {path}")


def load_keypoint_rows(path):
    result = {}
    with Path(path).open() as handle:
        for row in csv.DictReader(handle):
            if row["kind"] != "POSITIVE":
                continue
            key = row["frame_id"]
            if key in result:
                raise ValueError("Duplicate paper keypoint frame id")
            errors = row["top_keypoint_supervised_errors_px"]
            result[key] = dict(frame_id=key,session_id=row["session_id"],
                errors=np.array([float(value) for value in errors.split(";")],dtype=np.float64) if errors else np.empty(0))
    if len(result) != 319:
        raise ValueError("Paper2D must retain all319 positive frame rows")
    return result


def pose_auc(values, diameters):
    """Exact evaluator 1001-threshold trapezoid, evaluated per observation.

    Summation can move outside the trapezoid because both are linear. This
    avoids constructing threshold x frame matrices in each bootstrap draw.
    Threshold positions and <= comparisons remain unchanged.
    """
    diameter = float(np.median(diameters))
    thresholds = np.linspace(0.,.1*diameter,1001)
    first = np.searchsorted(thresholds,np.asarray(values),side="left")
    contribution = np.where(first == 0,1.,np.where(first > 1000,0.,(1000-first+.5)/1000))
    return float(np.mean(contribution))


def keypoint_summary(rows):
    errors = [row["errors"] for row in rows if len(row["errors"])]
    values = np.concatenate(errors) if errors else np.empty(0)
    return dict(frames=len(rows),matched_frames=len(errors),keypoints=len(values),
        keypoint_location_median_px=float(np.median(values)) if len(values) else None,
        keypoint_location_p90_px=float(np.quantile(values,.9)) if len(values) else None)


def pose_summary(rows):
    if not rows:
        return dict(n=0,**{name:None for name in POSE_METRICS})
    values = lambda name:np.array([row[name] for row in rows],dtype=np.float64)
    return dict(n=len(rows),rotation_median_deg=float(np.median(values("rotation_error_deg"))),
        translation_median_cm=float(np.median(values("translation_error_cm"))),
        iou3d_median=float(np.median(values("iou3d"))),
        add_sym_auc=pose_auc(values("add_sym_m"),values("diameter_m")))


def weighted_quantile(values, weights, quantile):
    """numpy's linear empirical quantile for integer-weighted repetitions."""
    order = np.argsort(values,kind="stable")
    values, weights = np.asarray(values)[order],np.asarray(weights)[:,order]
    cumulative = np.cumsum(weights,axis=1)
    size = cumulative[:,-1]
    if (size <= 0).any():
        raise ValueError("Bootstrap sample contains no paired observations")
    rank = (size-1)*quantile
    lower,upper = np.floor(rank).astype(int),np.ceil(rank).astype(int)
    lo = (cumulative <= lower[:,None]).sum(-1)
    hi = (cumulative <= upper[:,None]).sum(-1)
    return values[lo]+(values[hi]-values[lo])*(rank-lower)


def weighted_statistic(values, groups, counts, metric, diameters=None):
    weights = counts[:,groups]
    if metric != "add_sym_auc":
        return weighted_quantile(values,weights,.9 if metric == "keypoint_location_p90_px" else .5)
    median_diameter = weighted_quantile(diameters,weights,.5)
    result = np.empty(len(counts),dtype=np.float64)
    for diameter in np.unique(median_diameter):
        selected = np.flatnonzero(median_diameter == diameter)
        thresholds = np.linspace(0.,.1*float(diameter),1001)
        first = np.searchsorted(thresholds,values,side="left")
        contribution = np.where(first == 0,1.,np.where(first > 1000,0.,(1000-first+.5)/1000))
        result[selected] = (weights[selected]*contribution).sum(-1)/weights[selected].sum(-1)
    return result


def prepare_series(store, keys, metric, group_lookup):
    values,groups,diameters = [],[],[]
    for key in keys:
        row = store[key]
        current = row["errors"] if metric in KEYPOINT_METRICS else [row[POSE_FIELDS[metric]]]
        values.extend(current)
        groups.extend([group_lookup[key]]*len(current))
        if metric == "add_sym_auc":
            diameters.append(row["diameter_m"])
    values = np.asarray(values,dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite evaluator values cannot be silently dropped")
    return values,np.asarray(groups,dtype=int),np.asarray(diameters,dtype=float) if diameters else None


def interval(draws, lower):
    low,high = np.quantile(draws,[.025,.975])
    return dict(low=float(low),high=float(high),excludes_zero=bool(low > 0 or high < 0),
                confirmed_benefit=bool(high < 0 if lower else low > 0),
                bootstrap_fraction_better=float(np.mean(draws < 0 if lower else draws > 0)))


def paired_metric(left, right, metric, resamples=N_RESAMPLES):
    """Three paired seed statistics with identical resampled frame/session IDs."""
    all_stores = left+right
    keys = sorted(set.intersection(*(set(store) for store in all_stores)))
    if metric in KEYPOINT_METRICS:
        keys = [key for key in keys if all(len(store[key]["errors"]) for store in all_stores)]
    if not keys:
        raise ValueError("No frames common to all compared seeds")
    for key in keys:
        if len({store[key]["session_id"] for store in all_stores}) != 1:
            raise ValueError("Paired session assignments disagree")
    sessions = sorted({left[0][key]["session_id"] for key in keys})
    if any(not value for value in sessions):
        raise ValueError("Missing session ID prevents session-paired inference")
    seed_base = 20260902 if metric in KEYPOINT_METRICS else 20260903
    rng = np.random.default_rng(seed_base)
    distributions = {}
    observed = None
    for scheme in ("frame_level","session_cluster"):
        lookup = ({key:i for i,key in enumerate(keys)} if scheme == "frame_level" else
                  {key:sessions.index(left[0][key]["session_id"]) for key in keys})
        n_groups = len(keys) if scheme == "frame_level" else len(sessions)
        series = [prepare_series(store,keys,metric,lookup) for store in all_stores]
        if observed is None:
            ones = np.ones((1,n_groups),dtype=np.int32)
            stats = [weighted_statistic(values,groups,ones,metric,diameters)[0]
                     for values,groups,diameters in series]
            observed = float(np.mean(stats[:3])-np.mean(stats[3:]))
        draws = np.empty(resamples,dtype=np.float64)
        for start in range(0,resamples,128):
            batch = min(128,resamples-start)
            counts = rng.multinomial(n_groups,np.full(n_groups,1/n_groups),size=batch).astype(np.int32)
            statistics = np.stack([weighted_statistic(values,groups,counts,metric,diameters)
                                  for values,groups,diameters in series])
            draws[start:start+batch] = statistics[:3].mean(0)-statistics[3:].mean(0)
        distributions[scheme] = interval(draws,metric in LOWER)
    return dict(difference=observed,better="lower" if metric in LOWER else "higher",
        paired_frames=len(keys),paired_sessions=len(sessions),
        per_seed_available_frames_left=[len(store) for store in left],
        per_seed_available_frames_right=[len(store) for store in right],
        frames_not_used_left=[len(store)-len(keys) for store in left],
        frames_not_used_right=[len(store)-len(keys) for store in right],
        resamples=resamples,seed=seed_base,**distributions,
        estimand="mean of3 seed-specific evaluator statistic differences on common paired frames; baseline repeated only as a reference, never as additional independent samples")


def mean_std(values):
    values = np.asarray(values,dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Missing/nonfinite primary metric")
    return dict(mean=float(values.mean()),std=float(values.std(ddof=1)) if len(values)>1 else 0.,
                seeds=values.tolist(),std_definition="sample standard deviation across3 trained seeds, ddof1")


def close(actual,expected,name):
    # Paper keypoint CSV errors are rounded to six decimal places.
    is_csv_keypoint = name.rsplit("/",1)[-1] in KEYPOINT_METRICS
    rtol,atol = (0.,5.1e-7) if is_csv_keypoint else (1e-10,1e-9)
    if not np.isclose(actual,expected,rtol=rtol,atol=atol):
        raise ValueError(f"Independent evaluator-statistic recomputation differs for {name}: {actual} vs {expected}")


def aggregate(run_dir):
    run_dir = Path(run_dir).resolve()
    train_protocol = read(run_dir/"TRAIN_PROTOCOL.json")
    decision_path = run_dir/"DECISION_PROTOCOL.json"
    decision = read(decision_path)
    selection = read(run_dir/"SELECTION.json")
    real = read(run_dir/"REAL_EVALUATION_COMPLETE.json")
    if not all(value.get("complete") for value in (train_protocol,decision,selection,real)) or not real.get("PASS"):
        raise ValueError("Training/decision/selection/real evaluation are not all complete")
    if selection.get("no_real_selection") is not True or selection.get("selection_population") != "synth_val":
        raise ValueError("Selection was not frozen solely on synthetic data")
    checked(run_dir/"SELECTION.json",real["selection_sha256"])
    for path,digest in real["source_sha256"].items():
        checked(path,digest)
    checked(run_dir/"TRAIN_PROTOCOL.json",selection["training_protocol"]["sha256"])
    checked(selection["source_manifest"]["path"],selection["source_manifest"]["sha256"])
    checked(HERE/"model.py",selection["model_sha256"])
    checked(HERE/"readout.py",selection["readout_sha256"])
    # The parent writes this explicit pre-real extension of TRAIN_PROTOCOL.
    decision_binding = decision.get("train_protocol_sha256",decision.get("training_protocol_sha256"))
    if decision_binding != sha(run_dir/"TRAIN_PROTOCOL.json"):
        raise ValueError("Decision protocol does not bind the frozen training protocol")
    if decision.get("primary_arm","image_joint") != "image_joint":
        raise ValueError("Primary arm cannot change after observing real results")
    entries = {(row["arm"],int(row["seed"])):row for row in real["runs"]}
    if len(real["runs"]) != 9 or set(entries) != {(arm,seed) for arm in ARMS for seed in SEEDS}:
        raise ValueError("Exactly9 completed arm/seed actual evaluations are required")
    runtime_path = Path(real["runtime"]["path"])
    checked(runtime_path,real["runtime"]["sha256"])
    runtime = read(runtime_path)
    if not runtime.get("complete") or not runtime.get("PASS"):
        raise ValueError("Actual runtime measurements are incomplete")
    selected_runs = {(row["arm"],int(row["seed"])):row for row in selection["runs"]}
    baseline = None
    stores_2d,stores_pose = {},{}
    result_rows,sources = [],{}
    identity_hashes,box_hashes = set(),set()
    for arm in ARMS:
        stores_2d[arm],stores_pose[arm] = [],[]
        for seed in SEEDS:
            key,label = (arm,seed),f"{arm}_seed{seed}"
            entry = entries[key]
            directory = Path(entry.get("directory",entry.get("output_dir")))
            if entry["evaluation_arm"] != label:
                raise ValueError("Evaluation label and arm/seed disagree")
            for name,digest_key in (("RESULTS.json","result_sha256"),("COMPLETION.json","completion_sha256"),
                                    ("IDENTITY_AUDIT.json","identity_audit_sha256")):
                checked(directory/name,entry[digest_key])
            result,completion,audit = (read(directory/name) for name in ("RESULTS.json","COMPLETION.json","IDENTITY_AUDIT.json"))
            if not all(value.get("complete") for value in (result,completion,audit)) or not completion["PASS"] or not audit["PASS"]:
                raise ValueError("A per-run paper evaluation is incomplete")
            for name,digest in completion["output_sha256"].items():
                checked(directory/name,digest)
            if (result["population"] != dict(positive=319,negative=2689,role="DEV",held_out_final=False)
                    or not result["negative_outputs_preserved"] or not audit["all_boxes_scores_order_unchanged"]
                    or audit["negative_frames"] != 2689):
                raise ValueError("Paper population or box/negative preservation contract changed")
            identity_hashes.add(audit["negative_raw_candidates_sha256"])
            box_hashes.add(audit["all_box_score_sha256"])
            trained = read(run_dir/"runs"/label/"COMPLETION.json")
            selected = selected_runs[key]
            checkpoint = Path(selected["checkpoint"])
            checked(checkpoint,selected["checkpoint_sha256"])
            if (not trained.get("complete") or not trained.get("PASS") or trained["step"] != 6000
                    or trained.get("smoke") or trained["checkpoint_sha256"] != selected["checkpoint_sha256"]):
                raise ValueError("Actual final6000 training is not proven")
            saved = torch.load(checkpoint,map_location="cpu",weights_only=False)
            if (not saved.get("complete") or saved.get("smoke") or saved["step"] != 6000
                    or saved["arm"] != arm or saved["seed"] != seed
                    or saved["train_protocol_sha256"] != sha(run_dir/"TRAIN_PROTOCOL.json")
                    or saved["source_manifest_sha256"] != selection["source_manifest"]["sha256"]):
                raise ValueError("Checkpoint identity/protocol/budget is inconsistent")
            if not all(torch.isfinite(value).all() for value in saved["model_state_dict"].values()):
                raise ValueError("Nonfinite final checkpoint")
            del saved
            for reference in result["reference_sources"].values():
                checked(reference["path"],reference["sha256"])
            reference_pose = read(result["reference_sources"]["main_6d"]["path"])
            current_reference = dict(**result["reference_values"],coverage=reference_pose["paths"]["MAIN"]["coverage"])
            if baseline is None:
                baseline = current_reference
                reference_csv = Path(result["reference_sources"]["two_d"]["path"]).with_name("R0_per_frame.csv")
                reference_2d = load_keypoint_rows(reference_csv)
                stores_2d["R0"] = [reference_2d]*3
            elif current_reference != baseline:
                raise ValueError("Different R0 references across runs")
            kp = load_keypoint_rows(directory/"PAPER_2D_per_frame.csv")
            pose_payload = read(directory/"POSE_PER_FRAME_BY_ARM.json")
            pp = {row["frame_id"]:row for row in pose_payload["per_frame"][label]}
            rp = {row["frame_id"]:row for row in pose_payload["per_frame"]["R0"]}
            if "R0" not in stores_pose:
                stores_pose["R0"] = [rp]*3
            elif rp != stores_pose["R0"][0]:
                raise ValueError("Baseline pose rows differ across arms")
            for metric,value in keypoint_summary(list(kp.values())).items():
                if metric in KEYPOINT_METRICS:
                    close(value,result["two_d"][metric],label+"/"+metric)
            for metric,value in pose_summary(list(pp.values())).items():
                close(value,result["main_6d"][metric],label+"/"+metric)
            if set(kp) != set(stores_2d["R0"][0]):
                raise ValueError("Positive2D frame set differs from baseline")
            for frame in kp:
                if (kp[frame]["session_id"] != stores_2d["R0"][0][frame]["session_id"]
                        or len(kp[frame]["errors"]) != len(stores_2d["R0"][0][frame]["errors"])):
                    raise ValueError("Detection-preserved2D supervision or sessions changed")
            stores_2d[arm].append(kp)
            stores_pose[arm].append(pp)
            sessions = sorted({row["session_id"] for row in kp.values()})
            per_session = {sid:dict(two_d=keypoint_summary([row for row in kp.values() if row["session_id"] == sid]),
                                   main_6d=pose_summary([row for row in pp.values() if row["session_id"] == sid])) for sid in sessions}
            if label not in runtime["by_run"]:
                raise ValueError("Missing actual paired runtime for a run")
            result_rows.append(dict(arm=arm,seed=seed,evaluation_arm=label,two_d=result["two_d"],
                main_6d=result["main_6d"],coverage=result["main_6d_coverage"],per_session=per_session,
                runtime=runtime["by_run"][label],training=trained,
                existing_keypoint_bootstrap=read(directory/"KEYPOINT_PAIRED_BOOTSTRAP.json"),
                existing_pose_bootstrap=read(directory/"POSE_PAIRED_BOOTSTRAP.json")))
            sources[label] = {name:bound(directory/name) for name in ("RESULTS.json","COMPLETION.json","IDENTITY_AUDIT.json",
                "PAPER_2D_per_frame.csv","POSE_PER_FRAME_BY_ARM.json","KEYPOINT_PAIRED_BOOTSTRAP.json","POSE_PAIRED_BOOTSTRAP.json")}
            sources[label]["checkpoint"] = bound(checkpoint)
    if len(identity_hashes) != 1 or len(box_hashes) != 1:
        raise ValueError("Copied negatives or detection outputs differ across runs")
    for metric in KEYPOINT_METRICS:
        close(keypoint_summary(list(stores_2d["R0"][0].values()))[metric],baseline["two_d"][metric],"R0/"+metric)
    for metric,value in pose_summary(list(stores_pose["R0"][0].values())).items():
        close(value,baseline["main_6d"][metric],"R0/"+metric)
    families = {}
    for arm in ARMS:
        rows = [row for row in result_rows if row["arm"] == arm]
        metrics = {metric:mean_std([row["two_d" if metric in KEYPOINT_METRICS else "main_6d"][metric] for row in rows]) for metric in METRICS}
        per_session = {}
        for sid in rows[0]["per_session"]:
            per_session[sid] = dict(metrics={metric:mean_std([row["per_session"][sid]["two_d" if metric in KEYPOINT_METRICS else "main_6d"][metric]
                for row in rows]) if all(row["per_session"][sid]["two_d" if metric in KEYPOINT_METRICS else "main_6d"][metric] is not None for row in rows) else None for metric in METRICS})
        families[arm] = dict(metrics=metrics,per_session=per_session,
            coverage=mean_std([row["coverage"] for row in rows]),selected_rule=selection["selected_rules"][arm],
            temperatures=selection["temperatures"][arm])
    comparisons = {}
    for left,right in [(arm,"R0") for arm in ARMS]+[("image_joint","geometry_joint"),("image_joint","image_line_only")]:
        metrics = {}
        for metric in METRICS:
            store = stores_2d if metric in KEYPOINT_METRICS else stores_pose
            metrics[metric] = paired_metric(store[left],store[right],metric)
            print(f"paired bootstrap {left} vs {right}: {metric} done",flush=True)
        comparisons[f"{left}_vs_{right}"] = dict(left=left,right=right,metrics=metrics)
    reference_sessions = {sid:dict(two_d=keypoint_summary([row for row in stores_2d["R0"][0].values() if row["session_id"] == sid]),
        main_6d=pose_summary([row for row in stores_pose["R0"][0].values() if row["session_id"] == sid]))
        for sid in sorted({row["session_id"] for row in stores_2d["R0"][0].values()})}
    baseline["per_session"] = reference_sessions
    methods = dict(resamples=N_RESAMPLES,keypoint_seed=20260902,pose_seed=20260903,
        statistic="pooled9-supervised-point median/P90; MAIN6D medians and exact1001-threshold ADDsym AUC; first calculate each seed statistic, then mean3seed differences",
        resampling="paired frame and paired whole-session multinomial resampling; same draw applied to both methods and all3seeds; seeds/corners not independent replicates",
        extension="Original per-run paper bootstrap is retained. This aggregate additionally covers primary P90 and rotation (original pose bootstrap included yaw), and jointly averages seed statistics.",
        sources={name:bound(ROOT/path) for name,path in {
            "keypoint_bootstrap":"scripts/paper/paired_uncertainty_and_tails.py",
            "pose_bootstrap":"scripts/paper/pose_metric_closure_v1/paired_bootstrap_pose.py",
            "pose_evaluator":"scripts/paper/pose_metric_closure_v1/run_pose_evaluation.py",
            "pose_per_frame":"scripts/paper/pose_metric_closure_v1/evaluate_pose_by_session.py"}.items()})
    limits=["All319 positive and2689 negative paper frames are reused DEV; no independent final real generalization claim.",
        "Sessions are few; session-cluster intervals are conditional exploratory intervals and do not correct all repeated comparisons.",
        "Three trained seeds are averaged within the same paired resamples; intervals do not model a broad population of future training seeds.",
        "MAIN6D intervals use frames available for every compared seed; dropped frames and separate full-population coverage are reported.",
        "Negative candidates are copied from R0 exactly; negative neural forwards and PnP runtime are not measured by this comparison.",
        "Runtime includes image-to-2D-output computation after image loading, not disk I/O or PnP.",
        "New line branches use extra training compute beyond R0; geometry and line-only controls have the same added budget.",
        "Initial100step smoke reduced total/line loss but increased corner loss; it was a plumbing check, not evidence of accuracy gain."]
    identity = dict(PASS=True,all_boxes_scores_order_unchanged=True,negative_frames=2689,
        negative_raw_candidates_sha256=next(iter(identity_hashes)),all_box_score_sha256=next(iter(box_hashes)),
        negative_reinference_performed=False)
    summary = dict(schema="pallet_line_pose_summary_v1",complete=True,PASS=True,primary_arm="image_joint",
        population=dict(positive=319,negative=2689,role="DEV",held_out_final=False),
        baseline=baseline,runs=result_rows,arms=families,comparisons=comparisons,runtime=runtime,
        identity=identity,methods=methods,limitations=limits,sources=sources,
        source_bindings={name:bound(run_dir/name) for name in ("TRAIN_PROTOCOL.json","DECISION_PROTOCOL.json","SELECTION.json",
            "REAL_EVALUATION_COMPLETE.json","SOURCE_MANIFEST.json","TRAIN_CODE_REVIEW.json")},
        aggregate_source_sha256=sha(__file__))
    verdict = make_verdict(summary)
    write(run_dir/"SUMMARY.json",summary)
    write(run_dir/"VERDICT.json",verdict)
    write(run_dir/"AGGREGATE_COMPLETE.json",dict(schema="pallet_line_pose_aggregate_complete_v1",complete=True,PASS=True,
        completed_training_runs=9,steps_per_run=6000,actual_paper_evaluation_runs=9,
        primary_arm="image_joint",overall_accuracy_improved=verdict["overall_accuracy_improved"],
        scope="Training, fixed synthetic selection, actual9run paper evaluation and aggregation verified; HTML delivery checked separately by finalizer.",
        input_sha256={str(run_dir/name):sha(run_dir/name) for name in ("SELECTION.json","REAL_EVALUATION_COMPLETE.json","DECISION_PROTOCOL.json","TRAIN_PROTOCOL.json")},
        output_sha256={str(run_dir/name):sha(run_dir/name) for name in ("SUMMARY.json","VERDICT.json")},
        source_sha256={str(Path(__file__).resolve()):sha(__file__)}))
    return verdict


def make_verdict(summary):
    """Pre-real conservative decision: all primary metrics, plus coverage."""
    primary = summary["arms"]["image_joint"]
    comparison = summary["comparisons"]["image_joint_vs_R0"]["metrics"]
    criteria = {}
    for metric in METRICS:
        base = summary["baseline"]["two_d" if metric in KEYPOINT_METRICS else "main_6d"][metric]
        value = primary["metrics"][metric]["mean"]
        improves = value < base if metric in LOWER else value > base
        criteria[metric] = dict(seed_mean_improved=bool(improves),
            session_ci_supports_improvement=comparison[metric]["session_cluster"]["confirmed_benefit"],
            confirmed=bool(improves and comparison[metric]["session_cluster"]["confirmed_benefit"]))
    coverage = all(row["coverage"] >= summary["baseline"]["coverage"] for row in summary["runs"] if row["arm"] == "image_joint")
    keypoint = all(criteria[name]["confirmed"] for name in KEYPOINT_METRICS)
    pose = all(criteria[name]["confirmed"] for name in POSE_METRICS) and coverage
    preserved = summary["identity"]["PASS"]
    overall = bool(keypoint and pose and preserved)
    reasons = [f"{name}: 평균 {'개선' if value['seed_mean_improved'] else '개선 미확인'}, 세션 대응 구간 {'지지' if value['session_ci_supports_improvement'] else '미지지'}"
               for name,value in criteria.items()]
    if not coverage:
        reasons.append("하나 이상의 primary seed에서 MAIN6D pose coverage가 baseline보다 낮음")
    headline = ("공유 특징 기반 점·선 모델이 재사용 논문 DEV에서 사전등록한 2D·6D 개선 기준을 충족했습니다."
                if overall else "공유 특징 기반 점·선 모델의 논문 baseline 대비 종합 우월성은 확인되지 않았습니다.")
    med = primary["metrics"]["keypoint_location_median_px"]["mean"]
    p90 = primary["metrics"]["keypoint_location_p90_px"]["mean"]
    bp = summary['baseline']['main_6d']
    pm = primary['metrics']
    lines=[f"Primary=image_joint, 3 seeds 평균: 9점 median {summary['baseline']['two_d']['keypoint_location_median_px']:.3f}→{med:.3f}px, P90 {summary['baseline']['two_d']['keypoint_location_p90_px']:.3f}→{p90:.3f}px.",
        f"6D rotation {bp['rotation_median_deg']:.3f}→{pm['rotation_median_deg']['mean']:.3f}°, translation {bp['translation_median_cm']:.3f}→{pm['translation_median_cm']['mean']:.3f}cm.",
        f"6D IoU3D {bp['iou3d_median']:.4f}→{pm['iou3d_median']['mean']:.4f}, ADDsym AUC {bp['add_sym_auc']:.4f}→{pm['add_sym_auc']['mean']:.4f}.",
        f"2D 개선 확정={keypoint}, 6D 개선 확정={pose}; 검출·negative 출력 보존={preserved}.",
        "동일 실사319장·negative2689장의 재사용 DEV 결과이며 독립 final 일반화 결론은 아닙니다."]
    return dict(schema="pallet_line_pose_verdict_v1",complete=True,PASS=True,
        PASS_semantics="Execution/source/statistic validity, not model superiority",
        overall_accuracy_improved=overall,keypoint_gain_confirmed=bool(keypoint),pose_gain_confirmed=bool(pose),
        headline_ko=headline,reasons=reasons,primary_arm="image_joint",metric_criteria=criteria,
        pose_coverage_preserved=coverage,negative_detection_preserved=preserved,discord_lines_ko=lines,
        decision="Both keypoint median/P90 and all4 MAIN6D primary metrics need mean3seed improvement with correctly directed session-paired95% intervals; all primary pose coverages must be preserved. No arm is selected from real outcomes.")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,required=True)
    arguments=parser.parse_args()
    result=aggregate(arguments.run_dir)
    print(json.dumps({key:result[key] for key in ("complete","PASS","overall_accuracy_improved","headline_ko")},ensure_ascii=False,indent=2))
