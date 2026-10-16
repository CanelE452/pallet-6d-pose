"""Frozen-evidence counterfactual query locations: GT diagnostic, never fit."""
from pathlib import Path
import sys
import importlib.util
import importlib
import hashlib
import gzip
import json
import time
from datetime import datetime, timezone

import cv2
import numpy as np
import torch


DEST = Path(__file__).resolve().parent
ROOT = DEST.parents[4]
RUN = DEST.parent / "test_reused_v1"
IDS = ["eval_pallet07:1778652140531310080", "eval_pallet07:1778652166837872128"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def freeze_imports():
    name = "_boundary_oracle_query_frozen_test49"
    folder = RUN / "code_snapshot"
    spec = importlib.util.spec_from_file_location(name, folder / "__init__.py",
        submodule_search_locations=[str(folder)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)
    return [importlib.import_module(name+"."+s) for s in ("segmentation", "segmented_appearance", "appearance", "calibration")]


def selected_evidence(appearance, predicted, k):
    row = appearance["diagnostics"]["corners"][k]
    keys = ["inward_material_fraction", "outward_material_fraction", "patch_material_fraction",
            "nearby_inward_material_fraction", "soft_inward_material_support",
            "soft_nearby_inward_material_support", "patch_lab_mean", "patch_lab_median", "patch_lab_std",
            "inward_lab_mean", "inward_lab_std", "patch_chroma_mean", "patch_chroma_std",
            "edge_distance_px", "corner_structure", "edge_energy",
            "segmentation_probability_patch_mean", "segmentation_probability_inward_mean",
            "segmentation_probability_outward_mean", "segmentation_probability_nearby_inward_mean"]
    return dict(P_visible=predicted["visibility_score"][k], resulting_state=predicted["states"][k],
        boundary_score=predicted["boundary_score"][k], local_accuracy_score=predicted["local_accuracy_score"][k],
        local_evidence={key:row.get(key) for key in keys},
        raw_palette=appearance["diagnostics"]["palette"],patch_radius_px=appearance["diagnostics"]["patch_radius_px"],
        all_visibility_scores=predicted["visibility_score"],all_states=predicted["states"])


def main():
    output = DEST / "ORACLE_QUERY_LOCATION_DIAGNOSTIC.json"
    if output.exists():
        raise FileExistsError("Preserve completed oracle diagnostic")
    started = time.perf_counter()
    seal = read(RUN / "PREDICTION_SEAL.json")
    assert seal["complete"] and seal["before_scoring"] and seal["GT_read_canary"] == "PASS"
    assert sha(RUN / "PREDICTIONS.jsonl.gz") == seal["sha256"]
    protocol = read(RUN / "PROTOCOL.json")
    originals = read(DEST / "BOUNDARY_RGB_FAILURE_DIAGNOSTICS_V1.json")
    assert {r["id"] for r in originals["records"]} == set(IDS)
    assert set(IDS) <= set(protocol["ids"])
    assert not set(IDS) & set(read(DEST.parent/"COHORT_PROTOCOL.json")["excluded_severe_ids"])
    config_path = DEST.parent / "CAL_VISIBILITY_CONFIG.json"
    config = read(config_path)
    model_path = (config_path.parent/config["visibility_model"]).resolve()
    weights = (config_path.parent/config["segmentation_weights"]).resolve()
    assert sha(config_path) == protocol["config_sha256"][str(config_path)]
    for path in (model_path, weights):
        assert sha(path) == protocol["model_sha256"][str(path)]
    for name in ("__init__.py", "common.py", "appearance.py", "segmented_appearance.py", "segmentation.py", "calibration.py"):
        assert sha(RUN/"code_snapshot"/name) == protocol["code_sha256"][name]
    bindings={str(p):sha(p) for p in [RUN/"PREDICTIONS.jsonl.gz",RUN/"PREDICTION_SEAL.json",
        RUN/"PROTOCOL.json",RUN/"ISOLATED_TEST_INPUTS.json",RUN/"ISOLATED_TEST_SCORING.json",
        DEST/"BOUNDARY_RGB_FAILURE_DIAGNOSTICS_V1.json",config_path,model_path,weights]}
    # Only two authorised rows from the isolated TEST49 inputs/scoring bundle.
    frames={f["id"]:f for f in read(RUN/"ISOLATED_TEST_INPUTS.json")["frames"] if f["id"] in IDS}
    isolated = read(RUN/"ISOLATED_TEST_SCORING.json")["rows"]
    scoring={fid:isolated[fid] for fid in IDS}
    predictions={}
    with gzip.open(RUN/"PREDICTIONS.jsonl.gz", "rt") as handle:
        for line in handle:
            row=json.loads(line)
            if row["id"] in IDS and row["method"]=="SEGMENTED_RGB_CURRICULUM":predictions[row["id"]]=row
    assert set(frames)==set(scoring)==set(predictions)==set(IDS)
    segmenter, segmented, raw, calibration = freeze_imports()
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    cv2.setNumThreads(1)
    device="cuda" if torch.cuda.is_available() else "cpu"
    model=read(model_path)
    assert model["visible_threshold"]==.2
    cases=[]
    for fid in IDS:
        f,s,saved=frames[fid],scoring[fid],predictions[fid]
        image_path=ROOT/f["image"]
        assert sha(image_path)==f["image_sha256"]
        image=cv2.imread(str(image_path)); assert image is not None
        image_sha=array_sha(image)
        q=np.asarray(f["points"]["N3_SUBPIX"],float)[:8]
        q_before=q.copy()
        hidden=np.asarray(saved["self_hidden"],bool)
        hidden_before=hidden.copy()
        metadata=s["metadata"]
        mapping=np.asarray(metadata["native_to_reference_corner_ids"],int)
        assert np.array_equal(mapping,np.arange(8))
        probability=segmenter.segment(image,weights,device)
        probability_sha=array_sha(probability)
        baseline=segmented.infer_segmented_appearance(image,q,hidden,probability,config.get("appearance"))
        baseline_pred=calibration.predict(baseline,model)
        active=~hidden & np.array([state!="OUT_OF_FRAME" for state in saved["appearance"]["states"]])
        difference=np.abs(np.asarray(baseline_pred["visibility_score"])-np.asarray(saved["appearance"]["visibility_score"]))
        assert difference[active].max()<1e-9, difference.tolist()
        rows=[]
        for k in ([5] if fid==IDS[0] else [4,5]):
            assert not hidden[k]
            oracle_q=q.copy()
            oracle_q[k]=s["target"]["gt"][mapping[k]]
            assert np.array_equal(oracle_q[np.arange(8)!=k],q[np.arange(8)!=k])
            oracle=segmented.infer_segmented_appearance(image,oracle_q,hidden,probability,config.get("appearance"))
            oracle_pred=calibration.predict(oracle,model)
            # Raw colour evidence is logged as context, without changing the
            # frozen pipeline's sourceCNN+RGB classifier feature basis.
            raw_baseline=raw.infer_appearance(image,q,hidden,config.get("appearance"))
            raw_oracle=raw.infer_appearance(image,oracle_q,hidden,config.get("appearance"))
            original_evidence=selected_evidence(baseline,baseline_pred,k)
            oracle_evidence=selected_evidence(oracle,oracle_pred,k)
            state=metadata["human_states_native"][k]
            correct_after = oracle_pred["visibility_score"][k] >= model["visible_threshold"] if state=="DIRECT_VISIBLE" else oracle_pred["visibility_score"][k] < model["visible_threshold"]
            rows.append(dict(native_corner=k,locked_reference_corner=int(mapping[k]),locked_human_state=state,
                original_query=q[k].tolist(),ORACLE_GT_query=oracle_q[k].tolist(),query_displacement_px=float(np.linalg.norm(q[k]-oracle_q[k])),
                all_other_queries_exactly_preserved=True,fixed_initial_H_exactly_preserved=True,
                classifier_feature_basis="same frozen sourceCNN binary material + raw Lab26 descriptors as original pipeline",
                original=original_evidence,ORACLE_query=oracle_evidence,
                raw_RGB_colour_context_original=raw_baseline["diagnostics"]["corners"][k],
                raw_RGB_colour_context_ORACLE=raw_oracle["diagnostics"]["corners"][k],
                P_visible_change=oracle_pred["visibility_score"][k]-baseline_pred["visibility_score"][k],
                crosses_frozen_threshold=bool((baseline_pred["visibility_score"][k]>=model["visible_threshold"]) !=
                    (oracle_pred["visibility_score"][k]>=model["visible_threshold"])),
                ORACLE_query_matches_locked_binary_label=bool(correct_after),
                counterfactual_role="GT_query_diagnostic_not_deployable",
                correspondence_confounded=fid==IDS[1],
                interpretation="GT query changes appearance hull/palette context as defined by the unchanged function. This is an oracle-location intervention, not an isolated single-pixel classifier or a deployable locator."))
        assert np.array_equal(image,np.asarray(image)) and array_sha(image)==image_sha
        assert np.array_equal(q,q_before) and np.array_equal(hidden,hidden_before)
        assert array_sha(probability)==probability_sha
        cases.append(dict(id=fid,severity="MODERATE_OCCLUSION",image_sha256=f["image_sha256"],
            fullframe_CNN_probability_sha256=probability_sha,fullframe_CNN_forward_calls=1,
            original_H=hidden.tolist(),locked_native_reference_mapping=mapping.tolist(),
            baseline_saved_probability_max_error=float(difference[active].max()),
            baseline_saved_states_exact=baseline_pred["states"]==saved["appearance"]["states"],
            all_oracle_outputs_only_diagnostic=True,points=rows))
    assert all(sha(path)==value for path,value in bindings.items())
    first=cases[0]["points"][0]
    report=dict(schema="GT_QUERY_LOCATION_POSTSEAL_DIAGNOSTIC_V1",created_UTC=datetime.now(timezone.utc).isoformat(),
        GT_query_diagnostic_not_deployable=True,postseal_test_diagnostic_only=True,
        images=2,ORACLE_point_queries=3,device=device,frozen_visibility_threshold=model["visible_threshold"],
        models_source_and_weights_bound_to_original_protocol=True,bindings=bindings,
        analysis_source_sha256=sha(__file__),new_training_calls=0,pose_solver_calls=0,pose_outputs=0,
        model_policy_or_threshold_changes=0,native_reference_remapping_applied=False,
        original_initial_H_and_native_order_fixed=True,severe_images_or_heads_read=0,FINAL_reads=0,
        primary_predictions_unchanged=True,records=cases,
        firstcase_summary=dict(original_P_visible=first["original"]["P_visible"],
            ORACLE_GT_query_P_visible=first["ORACLE_query"]["P_visible"],
            frozen_threshold_crossed=first["crosses_frozen_threshold"],
            oracle_label_agreement=first["ORACLE_query_matches_locked_binary_label"],
            attribution="This tests whether the frozen classifier recovers at the locked query location. It cannot establish that all failure causation is localisation, nor resolve inherited reference/model correspondence."),
        limitations=["GT is used solely to choose an oracle query; these coordinates are never independent detector evidence or pose-fit inputs.",
          "Replacing one query can affect the unchanged hull/palette extraction context. The full-frame CNN probability is shared unchanged across all queries in each image.",
          "The first historical object-coordinate phase is incompatible even though spatial IDs match. The second oracle queries remain strongly correspondence-confounded and never remap native channels.",
          "No model, threshold, correction policy or deployable coordinate locator will be selected from these already observed TEST examples."])
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(output)
    for c in cases:
        for row in c["points"]:
            print(c["id"],row["native_corner"],row["original"]["P_visible"],"->",row["ORACLE_query"]["P_visible"],
                "crossed",row["crosses_frozen_threshold"],"oracle_label_agreement",row["ORACLE_query_matches_locked_binary_label"])
    print("DONE seconds",time.perf_counter()-started)


if __name__=="__main__":
    main()
