"""Read-only six-case registration/camera/reference provenance checks.

No pose reconstruction, objective evaluation, mask/edge inference, or reference
revision. Original fixed projections are checked by direct matrix arithmetic.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

IDS=("plastic_night_01:038630","wood_day_01:005286","wood_night_01:030698",
     "plastic_day_01:020955","plastic_day_01:014704","wood_night_01:030898")


def sha(path):
    value=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b""):value.update(block)
    return value.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def rows(path):
    with gzip.open(path,"rt") as stream:
        for line in stream:yield json.loads(line)


def require(condition,message):
    if not condition:raise ValueError(message)


def cuboid(dimensions):
    a,b,c=np.asarray(dimensions,float)/2
    return np.array([[-a,-b,-c],[a,-b,-c],[a,b,-c],[-a,b,-c],[-a,-b,c],[a,-b,c],[a,b,c],[-a,b,c]])


def project(xyz,R,t,K):
    h=(np.asarray(xyz)@np.asarray(R).T+np.asarray(t))@np.asarray(K).T
    return h[:,:2]/h[:,2,None]


def gap(a,b):
    d=np.linalg.norm(np.asarray(a)[:,None]-np.asarray(b)[None],axis=2)
    return float(max(d.min(0).max(),d.min(1).max()))


def run(source_root,asset_root,score_private,state_private,output):
    started,cpu=time.monotonic(),time.process_time()
    source_root,asset_root,score_private,state_private,output=map(Path,(source_root,asset_root,score_private,state_private,output))
    require(not output.exists(),"Preserve previous registration audit attempts")
    score=source_root/"_docs/experiments/pallet_selection_score_diagnosis_v1"
    reps=read(score/"gallery/REPRESENTATIVE_SELECTION.json")["examples"]
    require(tuple(r["id"] for r in reps)==IDS,"Preserve exact fixed six cases/order")
    proof={r["id"]:r for r in read(score/"gallery/GALLERY_PROOF.json")["cases"] if r["id"] in IDS}
    packet_path=state_private/"e_trace_v2/TYPED_PACKETS.jsonl.gz"
    packets={r["id"]:r for r in rows(packet_path) if r["id"] in IDS}
    require(set(packets)==set(proof)==set(IDS),"Exactly six source-bound cases")
    candidate_path=score_private/"selection_v1/RETAINED_CANDIDATES.jsonl.gz"
    wanted={(ident,p["candidate_id"]) for ident in IDS for p in proof[ident]["panels"][-2:]}
    candidates={(r["id"],r["candidate_id"]):r for r in rows(candidate_path) if (r["id"],r["candidate_id"]) in wanted}
    require(set(candidates)==wanted and len(candidates)==12,"Exactly twelve original fixed candidates")
    reference_path=state_private/"score_v2/FIXED_REFERENCES.json"
    reference_document=read(reference_path)
    references=reference_document["references"]["REMAINDER"]
    readiness_path=source_root/"_docs/experiments/pallet_visibility_curriculum_20261013_v1/nonsevere_v1/remainder_after_test49_v1/run_input_readiness_v1/RUN_INPUTS143.json"
    readiness={r["id"]:r for r in read(readiness_path)["rows"] if r["id"] in IDS}
    registry_path=asset_root/"challenge/config/CHALLENGE_OBJECT_GEOMETRY_REGISTRY.json"
    registry={r["object_type"]:r for r in read(registry_path)["objects"]}
    sources=("scripts/research/pallet_visibility_curriculum_20261013_v1/remainder_native_inputs_v1.py",
             "scripts/research/pallet_visibility_curriculum_20261013_v1/score_remainder_derived_reference_v4_14.py",
             "scripts/research/pallet_statewise_correspondence_diagnosis_v1/analysis.py",
             "scripts/research/pallet_selection_score_diagnosis_v1/gallery.py",
             "scripts/annotate/pallet_geometry.py","scripts/annotate/annotate_io.py","scripts/annotate/annotate_pnp.py")
    checks=[]
    for ident in IDS:
        session,frame=ident.split(":");p=packets[ident];case=proof[ident];ready=readiness[ident]
        annotation_relative=f"data/evaluation/pallet_eval_v1/final/positive/annotations/{session}/{frame}.json"
        annotation_path=source_root/annotation_relative;annotation=read(annotation_path);obj=annotation["objects"][0]
        require(sha(annotation_path)==reference_document["annotation_sha256"][annotation_relative],"Original annotation hash changed")
        image_path=source_root/p["image"]
        require(sha(image_path)==p["image_sha256"]==ready["image_sha256"],"Exact raw RGB binding")
        with Image.open(image_path) as image:
            raw=image.convert("RGB")
        height,width=p["solver_input"]["image_hw"]
        require(raw.size==(width,height)==(annotation["camera_data"]["width"],annotation["camera_data"]["height"]),"Same raw RGB/camera canvas")
        intr=annotation["camera_data"]["intrinsics"]
        K=np.array([[intr["fx"],0.,intr["cx"]],[0.,intr["fy"],intr["cy"]],[0.,0.,1.]])
        require(np.array_equal(K,p["solver_input"]["K"]) and np.array_equal(K,ready["K"]),"Exact raw-canvas K agreement")
        dims=np.array([obj["physical_dimensions_m"][axis] for axis in ("x","y","z")])
        object_registry=registry[annotation["object_type"]]
        require(np.array_equal(dims,ready["known_physical_dimensions_xyz_m"]) and
                np.array_equal(dims,[object_registry["physical_dimensions_m"][axis]for axis in ("x","y","z")]),"Exact external registered dimensions")
        viewport=case["viewport_original_xyxy"]
        source_crop=score/"gallery"/case["source_crop"]
        encoded=io.BytesIO();raw.crop(tuple(viewport)).save(encoded,format="JPEG",quality=87,optimize=True)
        require(encoded.getvalue()==source_crop.read_bytes(),"Exact original RGB crop and original JPEG parameters")
        pair=[]
        for panel in case["panels"][-2:]:
            cid=panel["candidate_id"];c=candidates[(ident,cid)]
            require(c["stage"]=="G13" and c["family"]==panel["family"],"Original fixed stage/family")
            for name in ("R","t","native_xyz","physical_registration_Q"):
                require(np.array_equal(c[name],panel[name]),"Exact original candidate field: "+name)
            xyz,Q,R,t=map(np.asarray,(c["native_xyz"],c["physical_registration_Q"],c["R"],c["t"]))
            require(np.allclose(xyz.mean(0),0.,atol=1e-12,rtol=0),"Centroid-origin native XYZ")
            shape_gap=gap(xyz@Q,cuboid(dims))
            require(shape_gap<=1e-9 and np.allclose(Q.T@Q,np.eye(3),atol=1e-9,rtol=0) and abs(np.linalg.det(Q)-1.)<=1e-9,"Proper registered unordered physical cuboid")
            projection=project(xyz,R,t,K)
            projection_error=float(np.max(np.abs(projection-panel["model_projection_xy"])))
            require(projection_error<=1e-9,"Saved original model projection in raw pixels")
            require(np.allclose(R@Q,c["pose"]["R_physical"],atol=1e-9,rtol=0),"Native-to-physical rotation")
            pair.append(dict(candidate_id=cid,stage="G13",family=c["family"],R_native=c["R"],t_native_m=c["t"],
                Q_native_from_physical=c["physical_registration_Q"],native_extents_m=c["cf_extents"],
                registered_unordered_corner_gap_m=shape_gap,projection_maximum_difference_px=projection_error,
                original_role=c["role"],no_candidate_changed=True))
        cf=obj["camera_facing_pnp"];T=np.array(cf["pose_transform"]);candidates_ref=obj["canonical_pose_candidates"]
        require(len(candidates_ref)==2 and obj["canonical_pose"] is None,"Preserved unconfirmed two-pose C2 reference policy")
        original_zero=np.array(candidates_ref[0]["pose_transform"]);original_one=np.array(candidates_ref[1]["pose_transform"])
        require(np.array_equal(original_zero[:3,3],T[:3,3]) and np.array_equal(original_one[:3,3],T[:3,3]),"Exact reference translation copies")
        require(np.array_equal(original_one[:3,:3],original_zero[:3,:3]@np.diag([-1.,1.,-1.])),"Exact original C2 pair")
        turn=np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
        assignment=int(candidates_ref[0]["axis_assignment"].split("_")[1])//90
        require(np.allclose(original_zero[:3,:3],T[:3,:3]@np.linalg.matrix_power(turn,assignment),atol=1e-6,rtol=0),"Original signed CF-to-canonical reference conversion")
        ref=references[ident]
        require(np.array_equal(ref["t"],T[:3,3]) and np.array_equal(ref["R"],original_zero[:3,:3]) and np.array_equal(ref["xyz"],dims),"Exact unchanged fixed reference")
        cf_dims=cf["dimensions_m"]
        reference_projection=project(cuboid([cf_dims[k] for k in ("width","height","depth")]),T[:3,:3],T[:3,3],K)
        canonical_projection=project(cuboid(dims),original_zero[:3,:3],original_zero[:3,3],K)
        reference_unordered_gap=gap(reference_projection,canonical_projection)
        require(reference_unordered_gap<=1e-9,"CF and canonical reference project the same unordered registered cuboid")
        sources_count=Counter(k["source"]for k in obj["keypoint_annotations"][:8])
        checks.append(dict(id=ident,group=next(r["group"]for r in reps if r["id"]==ident),
            raw_RGB=dict(path=p["image"],sha256=p["image_sha256"],size_wh=[width,height]),
            camera=dict(K=K.tolist(),annotation_K_equals_packet_K=True,readiness_K_equal=True,
                intrinsics_quality=annotation.get("intrinsics_quality"),intrinsics_source=annotation.get("intrinsics_source"),
                distortion_coefficients_stored_in_annotation="distortion" in annotation["camera_data"],
                independent_calibration_verified=False),
            display=dict(original_viewport_xyxy=viewport,original_source_crop=case["source_crop"],
                original_source_crop_sha256=sha(source_crop),JPEG_reconstruction_byte_equal=True,
                crop_transform_original_to_crop=[[1,0,-viewport[0]],[0,1,-viewport[1]],[0,0,1]],
                resized=False,letterboxed=False,K_offset_added=False,
                upstream_detector_letterbox_roundtrip_newly_validated=False,
                upstream_raw_canvas_contract="remainder_native_inputs_v1.predictor_record requires cached raw_hw; raw qFinal equals native_points; emits crop_transform=None"),
            registration=dict(object_type=annotation["object_type"],physical_dimensions_xyz_m=dims.tolist(),
                source_measurement_cm=object_registry["source_measurement_cm"],source_measurement_order=object_registry["source_measurement_order"],
                dimension_availability="external frozen prediction-registry condition, not measured again here",
                symmetry_registry_status=object_registry["symmetry_status"],
                origin="cuboid geometric centroid; canonical and native share origin",
                axes="canonical +X registered first footprint axis; +Y down height; +Z registered second footprint axis",
                physical_points_row_rule="x_physical_row = x_native_row @ Q; R_physical = R_native @ Q",
                candidates=pair,independent_physical_corner_semantics_verified=False),
            reference=dict(annotation=annotation_relative,annotation_sha256=sha(annotation_path),
                schema=annotation["schema_version"],gt_source_metadata=obj.get("gt_source"),
                keypoint_source_counts_first_eight=dict(sources_count),
                manual_click_corner_ids=[i for i,k in enumerate(obj["keypoint_annotations"][:8])if k["source"]=="manual_click"],
                extrapolated_corner_ids=[i for i,k in enumerate(obj["keypoint_annotations"][:8])if k["source"]=="extrapolated"],
                pnp_projected_corner_ids=[i for i,k in enumerate(obj["keypoint_annotations"][:8])if k["source"]=="pnp_projected"],
                keypoint_frame=obj["keypoint_frame"],pose_status=obj["pose_status"],migration_status=obj["migration_status"],
                selected_camera_facing_hypothesis=cf["selected_hypothesis"],hypothesis_selection_reason=cf["selection_reason"],
                axis_assignment_confirmed=cf["axis_assignment_confirmed"],canonical_candidate_assignments=[r["axis_assignment"]for r in candidates_ref],
                saved_reported_reprojection_error_px=obj.get("reproj_error_px"),
                fixed_reference_equals_original_candidate_zero=True,exact_C2_pair_and_centroid_copy=True,
                CF_to_canonical_signed_conversion_checked=True,
                CF_vs_canonical_unordered_reference_projection_gap_px=reference_unordered_gap,
                policy="translation=stored camera_facing_pnp t; rotation=fixed original canonical candidate[0] after C2/exact-t-copy checks; no nearest-prediction selection",
                annotation_reference_is_independent_physical_ground_truth=False,
                CF_reference_projection_raw_xy=reference_projection.tolist(),
                stored_annotation_projection_maximum_difference_px=float(np.max(np.abs(reference_projection-obj["projected_cuboid"]))),
                unused_reference_hypothesis_poses_missing=all("pose_transform"not in h for h in cf["hypotheses"])),
            A_mask_RGB_support=dict(status="REQUIRES_DIRECT_RGB_REVIEW_BY_ROOT",numeric_cost_not_used_as_semantic_judgment=True),
            B_outline_cuboid_identity=dict(numeric_registered_geometry="PASS",physical_silhouette_correspondence="UNRESOLVED_FROM_REGISTRATION_ALONE"),
            C_image_model_reference_compatibility=dict(numeric_K_Q_size_reference_contract="PASS",physical_calibration_and_corner_meaning="UNVERIFIED")))
    result=dict(schema="SIX_CASE_REGISTERED_GEOMETRY_AND_REFERENCE_PROVENANCE_AUDIT_V1",cases=checks,
        source_result_commit="c56a4e92027a8e56f6704773f24d7c3da78082f1",upstream_result_commit="407de865445e4987614f50f893d2ccea78d507b6",
        fixed_candidate_count=12,source_file_sha256={p:sha(source_root/p)for p in sources},
        audit_source_sha256=sha(Path(__file__)),
        inputs=dict(typed_packets_sha256=sha(packet_path),original_retained_candidate_sha256=sha(candidate_path),
            fixed_references_sha256=sha(reference_path),registry_path="ASSET_ROOT/challenge/config/CHALLENGE_OBJECT_GEOMETRY_REGISTRY.json",registry_sha256=sha(registry_path),
            representative_selection_sha256=sha(score/"gallery/REPRESENTATIVE_SELECTION.json"),gallery_proof_sha256=sha(score/"gallery/GALLERY_PROOF.json")),
        all_numeric_registration_checks_passed=True,independent_physical_pose_or_semantic_owner_validation_claimed=False,
        PnP_optimizer_forward_training_new_candidates_calls=0,existing_source_prediction_reference_files_modified=False,
        elapsed_seconds=time.monotonic()-started,process_cpu_seconds=time.process_time()-cpu)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x")as stream:json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write("\n")
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("source-root","asset-root","score-private","state-private","output"):p.add_argument("--"+key,required=True,type=Path)
    args=p.parse_args();result=run(**vars(args))
    print(dict(cases=len(result["cases"]),fixed_candidate_count=result["fixed_candidate_count"],passed=result["all_numeric_registration_checks_passed"],elapsed_seconds=result["elapsed_seconds"]))
