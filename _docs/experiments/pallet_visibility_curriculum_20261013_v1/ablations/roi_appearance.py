"""Experimental ROI implementation wrapper; no new learned/decision policy.

The CNN must already have run on the complete original image. This module only
crops downstream RGB calculations, translating native coordinates by an even
integer origin to preserve OpenCV nearest-even polygon rasterization.
"""
from __future__ import annotations

import cv2
import numpy as np

from scripts.research.pallet_visibility_curriculum_20261013_v1 import appearance as A
from scripts.research.pallet_visibility_curriculum_20261013_v1 import segmented_appearance as S


def crop_descriptor(image_bgr,points,self_hidden,config=None):
    cfg=A._config(config)
    image,q,hidden,valid,inside=A._inputs(image_bgr,points,self_hidden)
    h,w=image.shape[:2];finite=q[valid]
    extent=float(np.hypot(*(np.ptp(finite,axis=0)))) if len(finite) else 0.
    radius=int(np.clip(round(extent*cfg["patch_radius_fraction"]),cfg["patch_radius_min"],cfg["patch_radius_max"]))
    # _fill_polygon's extreme-coordinate clamp is canvas-size-dependent.
    # Retain the original canvas for huge hulls instead of changing that clamp.
    if not len(finite) or not inside.any() or np.max(np.abs(finite))>2*max(h,w):
        bounds=(0,0,w,h)
    else:
        padding=8*radius
        lo=np.floor(finite.min(0)-padding).astype(int)
        hi=np.ceil(finite.max(0)+padding+1).astype(int)
        # Even origins make rint(x-offset)+offset identical at half-integers.
        x0=max(0,min(w,lo[0]//2*2));y0=max(0,min(h,lo[1]//2*2))
        x1=max(x0+1,min(w,hi[0]));y1=max(y0+1,min(h,hi[1]))
        bounds=(x0,y0,x1,y1)
    x0,y0,x1,y1=bounds
    translated=q-np.array([x0,y0])
    translated[~valid]=q[~valid] # retain the exact (-1,-1) missing sentinel
    return dict(image=image[y0:y1,x0:x1],points=translated,hidden=hidden,bounds=bounds,
                radius=radius,original_hw=(h,w),area_fraction=(x1-x0)*(y1-y0)/(h*w),
                full_frame=bounds==(0,0,w,h),config=cfg)


def infer_appearance_roi(image_bgr,points,self_hidden,config=None):
    roi=crop_descriptor(image_bgr,points,self_hidden,config)
    return A.infer_appearance(roi["image"],roi["points"],roi["hidden"],roi["config"])


def infer_segmented_appearance_roi(image_bgr,points,self_hidden,probability_mask,config=None):
    image=np.asarray(image_bgr)
    probability=S._probability(probability_mask,image.shape[:2])
    roi=crop_descriptor(image,points,self_hidden,config)
    x0,y0,x1,y1=roi["bounds"]
    if roi["full_frame"]:
        return S.infer_segmented_appearance(image_bgr,points,self_hidden,probability_mask,config)
    cfg=roi["config"]
    cropped,translated,hidden,valid,inside=A._inputs(roi["image"],roi["points"],roi["hidden"])
    cropped_probability=probability[y0:y1,x0:x1]
    scene=A._scene(cropped,translated,hidden,valid,cfg)
    palette=scene["palette"];raw_material=scene["material"]
    hull,mask_roi,radius=S._hull_roi(cropped_probability,translated,cfg)
    if radius!=scene["radius"] or not np.array_equal(hull,scene["hull"]):
        raise AssertionError("Original ROI/CNN hull geometry must agree")
    material=((cropped_probability>=.5)&(mask_roi>0)).astype(np.uint8)*255
    scene["material"]=material
    # OpenCV float32 chamfer accumulation can differ by a few ULPs when the
    # exact same mask is scanned in a translated crop. Keep its original global
    # coordinate canvas, then slice its result before local feature extraction.
    full_material=np.zeros(image.shape[:2],np.uint8)
    full_material[y0:y1,x0:x1]=material
    full_edge=cv2.morphologyEx(full_material,cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))
    scene["edge"]=full_edge[y0:y1,x0:x1]
    full_distance=cv2.distanceTransform((full_edge==0).astype(np.uint8),cv2.DIST_L2,3) if full_edge.any() else None
    scene["distance"]=None if full_distance is None else full_distance[y0:y1,x0:x1]
    scene["palette"]=dict(palette,confidence=1.,centers_lab=palette["centers_lab"] or [[0.,0.,0.]])
    states=[];visibility=[];boundary=[];accuracy=[];corners=[]
    for index in range(8):
        if not valid[index]:
            state,vis,bound,acc,row="UNKNOWN",0.,0.,0.,dict(reason="missing_or_nonfinite_prediction")
        elif not inside[index]:
            state,vis,bound,acc,row="OUT_OF_FRAME",0.,0.,0.,dict(reason="predicted_coordinate_outside_frame")
        elif hidden[index]:
            state,vis,bound,acc,row="SELF_OCCLUDED",0.,0.,0.,dict(reason="supplied_initial_pose_self_occlusion")
        else:
            state,vis,bound,acc,row=A._local(translated[index],scene,cfg)
            row["reason"]="frozen_source_CNN_mask_and_raw_RGB_features"
        if valid[index] and inside[index]:
            row.update(S._patch_probability(translated[index],scene,cropped_probability,mask_roi))
        row["binary_material_feature_source"]="frozen_source_RGB_CNN_with_predicted_hull_ROI"
        row["soft_material_feature_source"]="original_RGB_Lab_palette_support"
        row["appearance_version"]="v3_source_CNN_mask_raw_v2_Lab_features"
        states.append(state);visibility.append(vis);boundary.append(bound);accuracy.append(acc)
        corners.append(dict(corner=index,state=state,**row))
    result=dict(states=states,visibility_score=visibility,boundary_score=boundary,
        local_accuracy_score=accuracy,diagnostics=dict(palette=palette,corners=corners,
            patch_radius_px=radius,face_seed_count=scene["face_seed_count"],config=cfg,
            mask_source="frozen_source_RGB_CNN",mask_threshold=.5,
            mask_ROI="predicted_cuboid_hull_dilated_by_patch_radius",
            mask_ROI_pixels=int(np.count_nonzero(mask_roi)),mask_pixels=int(np.count_nonzero(material)),
            original_colour_material_pixels=int(np.count_nonzero(raw_material)),
            CNN_colour_material_intersection_pixels=int(np.count_nonzero((raw_material>0)&(material>0))),
            effective_palette_gate=dict(raw_colour_gate_applied_to_CNN=False,calibrated_CNN_confidence_claimed=False),
            inference_inputs=["current_image_bgr","predicted_points","supplied_self_hidden","frozen_CNN_probability_mask"],
            ground_truth_inputs=False,scores_calibrated=False,
            limits=["source_graphite_single_shape_domain","CNN_background_false_positives",
                    "incorrect_initial_pose_or_xy","virtual_bbox_corners","same_coloured_occluder_ownership",
                    "a_mask_support_pixel_is_not_proof_of_the_requested_3D_corner"]))
    # This diagnostic deliberately counts colour matches over the original
    # entire image, even though local CNN evidence is hull-restricted. Retain
    # its meaning instead of silently changing it into an ROI count.
    if not roi["full_frame"]:
        lab=cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
        centers=[np.asarray(c,np.float32) for c in result["diagnostics"]["palette"]["centers_lab"]]
        global_material=A._match(lab,centers,np.asarray(roi["config"]["lab_tolerance"],np.float32),
                                 float(roi["config"]["palette_match_radius"]))
        result["diagnostics"]["original_colour_material_pixels"]=int(np.count_nonzero(global_material))
    return result
