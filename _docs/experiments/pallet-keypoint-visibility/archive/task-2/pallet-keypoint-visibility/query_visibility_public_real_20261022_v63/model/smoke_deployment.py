"""Deployment smoke using mock features and one approved SYNTH source fixture."""
import csv
import gzip
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from deploy_query_visibility import QueryVisibility, sha256


def main():
    root = Path(__file__).resolve().parent.parent
    model_dir = root / "model"
    bundle = model_dir / "DEPLOYMENT_BUNDLE.json"
    api = QueryVisibility(bundle, "cpu")
    rng = np.random.default_rng(20261022)
    q = np.array([[20,20],[70,20],[70,45],[20,45],[25,15],[75,15],[75,40],[25,40]], float)
    x = rng.normal(size=(8,65,9,9)).astype(np.float16)
    x[:,64] = 1
    rgb = rng.integers(0,256,size=(8,3,65,65),dtype=np.uint8)
    sam = rng.normal(size=(8,1,9,9)).clip(-4,4).astype(np.float16)
    before = {key: value.clone() for key,value in api.model.state_dict().items()}
    a = api.predict_precomputed(points=q,raw_hw=[64,96],x=x,rgb=rgb,sam=sam)
    b = api.predict_precomputed(points=q,raw_hw=[64,96],x=x,rgb=rgb,sam=sam)
    assert a == b and all(row["status"]=="OK" for row in a["queries"])
    assert all(row["nearby_r2_support"]["status"]=="UNKNOWN" for row in a["queries"])
    assert all(row["geometry_status"].startswith("UNKNOWN") for row in a["queries"])
    x_empty=x.copy();x_empty[0,64]=0
    empty=api.predict_precomputed(points=q,raw_hw=[64,96],x=x_empty,rgb=rgb,sam=sam)
    assert empty["queries"][0]["reason"]=="INVALID_OR_EMPTY_ROI"
    assert all(row["status"]=="OK" for row in empty["queries"][1:])
    invalid_q=q.copy();invalid_q[0]=[np.nan,np.nan];invalid_q[1]=[-1,-1]
    invalid=api.predict_precomputed(points=invalid_q,raw_hw=[64,96],x=x,rgb=rgb,sam=sam)
    assert all(invalid["queries"][k]["status"]=="UNKNOWN" for k in [0,1])
    missing=api.predict_precomputed(points=np.full((8,2),-1),raw_hw=[64,96],x=x,rgb=rgb,sam=sam)
    assert all(row["status"]=="UNKNOWN" and row["p_visible_exact"]is None for row in missing["queries"])
    full=np.full((64,96),-1.,np.float32);full[20,22]=3
    neighborhood=api.predict_precomputed(points=q,raw_hw=[64,96],x=x,rgb=rgb,sam=sam,sam_full_logits=full)
    assert neighborhood["queries"][0]["raw_SAM_center_logit"]==-1
    assert neighborhood["queries"][0]["nearby_r2_support"]["present"]is True
    assert neighborhood["queries"][0]["p_visible_exact"]==a["queries"][0]["p_visible_exact"]
    pose={"available":True,"cf_extents":[1.,1.,1.],"R_cf":np.eye(3).tolist(),"centroid":[0.,0.,10.]}
    geometry=api.predict_precomputed(points=q,raw_hw=[64,96],x=x,rgb=rgb,sam=sam,saved_initial_pose=pose)
    assert all(row["self_hidden_geometry_proxy"]is not None for row in geometry["queries"])
    assert [row["p_visible_exact"]for row in geometry["queries"]]==[row["p_visible_exact"]for row in a["queries"]]
    assert all(torch.equal(value,api.model.state_dict()[key])for key,value in before.items())

    v60=root.parent.parent/"external_ray_proxy_head_20261022_v60"
    with gzip.open(v60/"PREPARED_INPUTS_SYNTH.json.gz","rt")as stream:
        frames=json.load(stream)
    with np.load(root/"source_cache/v4/SYNTH_VALIDATION_RGB_SAM.npz")as cache:
        ids=cache["id"].astype(str)
        ident=next(ident for ident in sorted(set(ids))if (ids==ident).sum()==8)
        chosen=np.flatnonzero(ids==ident);chosen=chosen[np.argsort(cache["native_corner"][chosen])]
        stored_roi=cache["x"][chosen]
        stored_rgb,stored_sam,stored_context=cache["rgb"][chosen],cache["sam"][chosen],cache["context"][chosen]
    frame_index=next(i for i,frame in enumerate(frames)if frame["id"]==ident)
    frame=frames[frame_index]
    denied=set(json.loads((root/"contract/EVALUATION_SHA256_DENYLIST.json").read_text())["image_sha256"])
    assert frame["image_sha256"]not in denied and sha256(frame["image"])==frame["image_sha256"]
    with np.load(v60/"SYNTH_ROI_CACHE.npz")as original:
        assert np.array_equal(stored_roi,original["x"][frame_index])
    source_light=api.predict_precomputed(points=frame["initial_points"][:8],raw_hw=frame["raw_hw"],x=stored_roi,
                                         rgb=stored_rgb,sam=stored_sam,context=stored_context)
    with np.load(root/"runs/MIXED_CONTEXT_RGB_SAM_V4/VAL_SOURCE_OUTPUTS.npz")as outputs:
        frozen_prob=outputs["probability"][chosen]
    source_light_prob=np.array([row["p_visible_exact"]for row in source_light["queries"]])
    saved_score_error=float(np.abs(source_light_prob-frozen_prob).max())
    assert saved_score_error<2e-5,saved_score_error
    api._load_backbone()
    capture={};original_patches=api.roi_module.patches
    def recording_patches(*args,**kwargs):
        result=original_patches(*args,**kwargs);capture["roi"]=result[0].numpy();capture["side"]=float(result[2]);return result
    api.roi_module.patches=recording_patches
    with Image.open(frame["image"])as image:
        image_rgb=np.array(image.convert("RGB"))
    start=time.monotonic()
    source_result=api.predict_rgb(image_rgb,frame["initial_points"][:8])
    elapsed=time.monotonic()-start
    assert all(row["status"]=="OK" for row in source_result["queries"])
    assert all(row["nearby_r2_support"]["status"]=="OK" for row in source_result["queries"])
    fresh,expected=capture["roi"].astype(np.float32),stored_roi.astype(np.float32)
    delta=np.abs(fresh-expected)
    # CPU and the original CUDA backbone can round a few FP16 cells differently.
    feature_close=np.allclose(fresh,expected,atol=.004,rtol=.002)
    assert feature_close, float(delta.max())
    fresh_probability=np.array([row["p_visible_exact"]for row in source_result["queries"]])
    fresh_score_error=float(np.abs(fresh_probability-frozen_prob).max())
    assert fresh_score_error<.02,fresh_score_error
    fixture_dir=model_dir/"deployment_fixture";fixture_dir.mkdir(exist_ok=True)
    request={"image":frame["image"],"initial_points":frame["initial_points"][:8],"request_id":"SOURCE_ONLY_"+ident}
    (fixture_dir/"SOURCE_REQUEST.json").write_text(json.dumps(request,indent=2)+"\n")
    (fixture_dir/"SOURCE_RESULT.json").write_text(json.dumps(source_result,indent=2,allow_nan=False)+"\n")
    report={"PASS":True,"device":"cpu","mock_repeat_exact":True,"frozen_head_state_unchanged":True,
            "invalid_queries_UNKNOWN":True,"degenerate_box_UNKNOWN":True,"empty_ROI_reason_correct":True,
            "r2_distinct_from_exact_and_missing_fullmap_UNKNOWN":True,"geometry_never_changes_learned_probability":True,
            "fresh_source_pipeline_executed":True,"fixture_source":"SYNTH_VALIDATION","fixture_id":ident,
            "fixture_image_sha256":frame["image_sha256"],"fixture_denylist_overlap":0,
            "original_v60_ROI_equals_carried_source_ROI":True,"fresh_ROI_shape":list(fresh.shape),
            "fresh_CPU_vs_original_CUDA_FP16_max_abs_error":float(delta.max()),
            "fresh_CPU_vs_original_CUDA_FP16_mean_abs_error":float(delta.mean()),
            "fresh_CPU_vs_original_CUDA_FP16_identical_fraction":float((fresh==expected).mean()),
            "feature_parity_tolerance":{"atol":.004,"rtol":.002},"feature_parity_PASS":bool(feature_close),
            "cached_source_API_vs_saved_frozen_GPU_max_probability_error":saved_score_error,
            "fresh_source_pipeline_vs_saved_frozen_GPU_max_probability_error":fresh_score_error,
            "fresh_pipeline_seconds":elapsed,"evaluation_images_read":0,"GT_masks_read":0,"PnP_calls":0,
            "source_model_checkpoint_sha256":api.bundle["assets"]["checkpoint"]["sha256"],
            "deployment_bundle_sha256":sha256(bundle)}
    api.close()
    (model_dir/"DEPLOYMENT_SMOKE.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report))


if __name__=="__main__":
    main()
