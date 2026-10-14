"""State-specific, image-only B/I measurements frozen before joint O/T PnP.

B searches positively oriented target/background mask transitions; I searches
incident RGB gradient lines. Initial model directions guide local searches.
S/O/T/U never receive a per-keypoint correction or pixel query in this module.
"""
from copy import deepcopy
import hashlib

import cv2
import numpy as np

from . import common as C
from . import boundary_refine as F
from .interior_observation import NEIGHBORS
from scripts.research.pallet_foundation_mask_20261013_v1.uncertainty import intersection_uncertainty


DEFAULT_CONFIG = dict(normal_search_radius_px=12., normal_step_px=.5, along_start_px=4.,
    along_end_px=36., along_samples=15, maximum_direction_change_deg=12.,
    maximum_line_rms_px=1.25, minimum_inlier_count=6, minimum_inlier_fraction=.6,
    minimum_inlier_span_fraction=.55, minimum_intersection_sine=.25,
    maximum_corner_shift_px=12., detector_sigma_px=2., maximum_alpha=.5,
    boundary_probe_px=2.5, boundary_context_px=7., minimum_transition=.45,
    minimum_inside=.65, maximum_outside=.35, minimum_gradient=4.,
    minimum_gradient_alignment=.8, minimum_color_transition=10.,
    fallback_boundary_sigma_px=2., fallback_interior_sigma_px=3.,
    visible_protection_px=2., line_resampling=64, seed=20261014)


def image_sha256(image):
    a = np.asarray(image)
    return hashlib.sha256(b"BI-BGR-v2\0"+str(a.shape).encode()+np.ascontiguousarray(a).tobytes()).hexdigest()


def _sample(array, xy):
    return cv2.remap(array, xy[...,0].astype(np.float32), xy[...,1].astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT).astype(float)


def _line(image, mask, lab, gradients, q, k, mate, guide, model_center, role, cfg):
    """Measure an actual local image line; mate supplies a direction only."""
    length = float(np.linalg.norm(guide))
    if length < 10.:
        return dict(edge=k*8+mate, endpoints=[k,mate], supported=False, reason="short_model_guide")
    direction = guide/length
    normal = np.array([-direction[1],direction[0]])
    if normal @ (model_center-q[k]) < 0:
        normal = -normal
    end = min(cfg["along_end_px"], .8*length)
    if end <= cfg["along_start_px"]:
        return dict(edge=k*8+mate, endpoints=[k,mate], supported=False, reason="short_search_span")
    along = np.linspace(cfg["along_start_px"], end, cfg["along_samples"])
    offsets = np.arange(-cfg["normal_search_radius_px"], cfg["normal_search_radius_px"]+.01, cfg["normal_step_px"])
    xy = q[k]+along[:,None,None]*direction+offsets[None,:,None]*normal
    h,w = image.shape[:2]
    if role == "B":
        probe, context = cfg["boundary_probe_px"], cfg["boundary_context_px"]
        inside = sum(_sample(mask,xy+normal*probe*s) for s in (.5,1.,1.5))/3
        outside = sum(_sample(mask,xy-normal*probe*s) for s in (.5,1.,1.5))/3
        inward_context, outward_context = _sample(mask,xy+normal*context), _sample(mask,xy-normal*context)
        contrast = inside-outside
        margin = max(context,1.5*probe)
        valid = (inside>=cfg["minimum_inside"])&(outside<=cfg["maximum_outside"])&(contrast>=cfg["minimum_transition"])
        valid &= (inward_context>=.50)&(outward_context<=.45)
        score = .65*contrast+.15*(inward_context-outward_context)-.015*np.abs(offsets)
        evidence = "OBSERVED_TARGET_BACKGROUND_TRANSITION"
    else:
        gx,gy = gradients
        x,y = _sample(gx,xy),_sample(gy,xy)
        energy = np.linalg.norm(x*normal[0]+y*normal[1],axis=-1)
        magnitude = np.sqrt(np.sum(x*x+y*y,axis=-1))
        alignment = energy/np.maximum(magnitude,1e-12)
        contrast = np.linalg.norm(_sample(lab,xy+normal*1.5)-_sample(lab,xy-normal*1.5),axis=-1)
        valid = (energy>=cfg["minimum_gradient"])&(alignment>=cfg["minimum_gradient_alignment"])&(contrast>=cfg["minimum_color_transition"])
        score = energy*alignment-.05*np.abs(offsets)
        margin = 1.5
        evidence = "OBSERVED_INTERNAL_RGB_GRADIENT"
    a,b = xy+normal*margin,xy-normal*margin
    valid &= (a[...,0]>=0)&(a[...,0]<w-1)&(a[...,1]>=0)&(a[...,1]<h-1)&(b[...,0]>=0)&(b[...,0]<w-1)&(b[...,1]>=0)&(b[...,1]<h-1)
    score = np.where(valid,score,-np.inf)
    best = np.argmax(score,axis=1)
    keep = np.isfinite(score[np.arange(len(along)),best])
    result = dict(edge=k*8+mate,endpoints=[k,mate],native_corner=k,native_neighbor=mate,
        samples=len(along),transition_sample_count=int(keep.sum()),supported=False,
        guide_source="initial_registered_model_direction_only",measurement_source=evidence)
    if keep.sum()<cfg["minimum_inlier_count"]:
        return dict(result,reason="insufficient_actual_image_samples")
    observations = xy[np.arange(len(along))[keep],best[keep]]
    confidence = np.clip(contrast[np.arange(len(along))[keep],best[keep]]/(1. if role=="B" else 40.),.05,1.)
    positions = .15+.7*(along[keep]-along[0])/(along[-1]-along[0])
    line_cfg = dict(along_interval=[.15,.85],huber_delta_px=1.25,
        maximum_angle_change_deg=cfg["maximum_direction_change_deg"],minimum_inlier_count=cfg["minimum_inlier_count"],
        minimum_inlier_fraction=cfg["minimum_inlier_fraction"],minimum_inlier_span_fraction=cfg["minimum_inlier_span_fraction"],
        maximum_line_rms_px=cfg["maximum_line_rms_px"])
    fitted, diagnostics = F._fit_line(observations,confidence,positions,normal,len(along),line_cfg)
    result.update(diagnostics,supported=fitted is not None)
    return result


def correct_visible_bi(image_bgr,raw_points,K,initial_pose,states,*,pallet_mask,config=None,native_xyz=None):
    cfg=dict(DEFAULT_CONFIG);cfg.update(config or {})
    if set(cfg)!=set(DEFAULT_CONFIG):raise ValueError("Unknown B/I correction field")
    if not all(np.isfinite(v) for v in cfg.values()):raise ValueError("Finite B/I correction configuration required")
    for key in ("normal_search_radius_px","normal_step_px","along_start_px","along_end_px",
                "maximum_corner_shift_px","detector_sigma_px","fallback_boundary_sigma_px",
                "fallback_interior_sigma_px","visible_protection_px","boundary_probe_px","boundary_context_px"):
        if cfg[key]<=0:raise ValueError(f"{key} must be positive")
    if not 0<cfg["maximum_alpha"]<=1 or cfg["along_end_px"]<=cfg["along_start_px"]:
        raise ValueError("Valid blend and line search span required")
    for key in ("along_samples","minimum_inlier_count","line_resampling"):
        if int(cfg[key])!=cfg[key] or cfg[key]<3:raise ValueError(f"{key} must be an integer >=3")
    if cfg["line_resampling"]<16:raise ValueError("At least 16 line resamples required")
    image=np.asarray(image_bgr);q=np.asarray(raw_points,float)
    if image.ndim!=3 or image.shape[2]!=3 or image.dtype!=np.uint8 or q.shape not in ((8,2),(9,2)) or len(states)!=8:
        raise ValueError("Actual BGR image, native q8/9 and eight states required")
    if any(s not in ("S","B","I","O","T","U") for s in states):raise ValueError("Unknown state")
    mask=np.asarray(pallet_mask,float)
    if mask.shape!=image.shape[:2] or not np.isfinite(mask).all() or np.any((mask<0)|(mask>1)):
        raise ValueError("Bound actual target mask in [0,1] required")
    K=np.asarray(K,float);X=C.cuboid(*initial_pose["cf_extents"])
    if native_xyz is not None:
        registered=np.asarray(native_xyz,float)
        if registered.shape!=(8,3) or not np.isfinite(registered).all() or not np.allclose(registered,X,rtol=0,atol=1e-9):
            raise ValueError("Registered native cuboid/order must match initial cf_extents")
        X=registered.copy()
    P=C.project(X,initial_pose["R_cf"],initial_pose["centroid"],K)
    if not np.isfinite(P).all():raise ValueError("Projectable initial model required")
    hull=cv2.convexHull(P.astype(np.float32),returnPoints=False).ravel().tolist()
    hull_neighbors={k:(hull[(i-1)%len(hull)],hull[(i+1)%len(hull)]) for i,k in enumerate(hull)}
    # Global image transforms do not perform a query at any excluded keypoint.
    lab=cv2.GaussianBlur(cv2.cvtColor(image,cv2.COLOR_BGR2LAB).astype(np.float32),(3,3),.8)
    gradients=cv2.Sobel(lab,cv2.CV_32F,1,0,ksize=3)/8.,cv2.Sobel(lab,cv2.CV_32F,0,1,ksize=3)/8.
    measured=q.copy();visible=[];records=[];h,w=image.shape[:2]
    for k,state in enumerate(states):
        if state not in ("B","I"):
            records.append(dict(native_corner=k,state=state,pixel_queried=False,corrected=False,reason="excluded_before_image_query"));continue
        if not np.isfinite(q[k]).all() or np.all(q[k]==-1) or not (0<=q[k,0]<w and 0<=q[k,1]<h):
            raise ValueError("Visible B/I requires an original in-frame image coordinate")
        neighbors=hull_neighbors.get(k,()) if state=="B" else tuple(m for m in NEIGHBORS[k] if states[m]!="S")
        rays=[_line(image,mask,lab,gradients,q,k,m,P[m]-P[k],P.mean(0),state,cfg) for m in neighbors]
        supported=[r for r in rays if r["supported"]]
        candidates=[]
        for i in range(len(supported)):
            for j in range(i+1,len(supported)):
                first,second=supported[i],supported[j]
                matrix=np.array([first["normal"],second["normal"]]);sine=abs(float(np.linalg.det(matrix)))
                if sine>=cfg["minimum_intersection_sine"]:
                    xy=np.linalg.solve(matrix,-np.array([first["offset"],second["offset"]]))
                    candidates.append((sine,first,second,xy))
        row=dict(native_corner=k,state=state,pixel_queried=True,corrected=False,
            method="B_TARGET_BACKGROUND_LINES" if state=="B" else "I_INTERNAL_RGB_JUNCTION",rays=rays,
            raw_xy=q[k].tolist(),reason="no_two_supported_nonparallel_lines")
        covariance=np.eye(2)*(cfg["fallback_boundary_sigma_px"] if state=="B" else cfg["fallback_interior_sigma_px"])**2
        if candidates:
            sine,first,second,candidate=max(candidates,key=lambda c:c[0])
            shift=float(np.linalg.norm(candidate-q[k]));disagreement=max(float(np.linalg.norm(a[3]-b[3])) for a in candidates for b in candidates)
            row.update(intersection_xy=candidate.tolist(),intersection_shift_px=shift,intersection_sine=sine)
            if shift<=cfg["maximum_corner_shift_px"] and disagreement<=3. and 0<=candidate[0]<w and 0<=candidate[1]<h:
                correction=dict(edge_records=[first,second],corner_records=[dict(corner=k,refined=True,candidate=candidate.tolist(),incident_shared_edges=[first["edge"],second["edge"]])])
                covs,_=intersection_uncertainty(correction,repetitions=int(cfg["line_resampling"]),block_length=3,normal_floor_px=1.,seed=int(cfg["seed"]))
                cov=covs.get(k)
                if cov and cov["successful_pair_resamples"]>=16:
                    conditional=np.asarray(cov["covariance_px2"]);variance=float(np.linalg.eigvalsh(conditional).max())
                    alpha=min(cfg["maximum_alpha"],cfg["detector_sigma_px"]**2/(cfg["detector_sigma_px"]**2+variance))
                    measured[k]=q[k]+alpha*(candidate-q[k])
                    covariance=(1-alpha)**2*np.eye(2)*cfg["detector_sigma_px"]**2+alpha**2*conditional+np.eye(2)
                    row.update(corrected=True,blend_alpha=alpha,applied_shift_px=float(np.linalg.norm(measured[k]-q[k])),
                               reason="actual_image_intersection",conditional_covariance=deepcopy(cov))
                else:row["reason"]="insufficient_line_resamples"
            else:row["reason"]="intersection_outside_image_trust_or_inconsistent"
        row.update(measured_xy=measured[k].tolist(),covariance_px2=covariance.tolist())
        visible.append(dict(native_corner=k,state=state,xy=measured[k].tolist(),covariance_px2=covariance.tolist(),
            protection_px=cfg["visible_protection_px"],source=row["method"] if row["corrected"] else "VISIBLE_BACKBONE_FALLBACK",
            image_measurement=True,measurement_is_pose_projection=False))
        records.append(row)
    return C.finite(dict(schema="visible_BI_measurements_v2",raw_points=q.tolist(),measured_points=measured.tolist(),
        visible_measurements=visible,records=records,image_pixel_sha256=image_sha256(image),
        configuration=cfg,queried_native_ids=[r["native_corner"] for r in records if r["pixel_queried"]],
        provenance=dict(S_queried=False,O_T_queried_for_correction=False,projected_coordinates_used_as_measurements=False,
            image_coordinates_frozen_before_joint_OT=True,conditional_covariance_is_calibrated=False,
            nonvisible_raw_coordinates_do_not_affect_measurements=True)))
