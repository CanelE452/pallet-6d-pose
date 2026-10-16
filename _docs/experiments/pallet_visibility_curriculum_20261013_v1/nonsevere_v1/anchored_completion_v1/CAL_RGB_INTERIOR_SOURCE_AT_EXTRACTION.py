"""Optional lower-evidence RGB junction measurements, never pose completions.

Initial cuboid rays guide local searches. Samples, fitted lines and their
intersection come from actual RGB gradients. Repeated grids, virtual/rounded
corners and wrong model phase remain semantic ambiguity. Conditional line
resampling covariance is not calibrated keypoint/visibility confidence.
"""
from copy import deepcopy
import hashlib

import cv2
import numpy as np

from . import boundary_refine as B
from . import common as C
from scripts.research.pallet_foundation_mask_20261013_v1.uncertainty import intersection_uncertainty

DEFAULT_CONFIG = dict(minimum_visibility=.65, normal_search_radius_px=12., normal_step_px=.5,
    along_min_px=4., along_max_px=28., along_samples=13, minimum_projected_incident_length_px=16.,
    edge_probe_px=1.5, gaussian_sigma_px=.8, minimum_normal_gradient=4., minimum_color_transition=10.,
    minimum_normal_alignment=.8, minimum_inlier_count=6, minimum_inlier_fraction=.6,
    minimum_inlier_span_fraction=.55, huber_delta_px=1.25, maximum_line_rms_px=1.25,
    maximum_angle_change_deg=12., minimum_intersection_sine=.25, maximum_corner_shift_px=12.,
    maximum_pair_disagreement_px=3., repetitions=64, block_length=3, normal_floor_px=1.,
    seed=20261013, minimum_successful_pair_resamples=16, detector_sigma_assumed_px=2., maximum_blend_alpha=.5)
NEIGHBORS = ((1,3,4),(0,2,5),(1,3,6),(0,2,7),(0,5,7),(1,4,6),(2,5,7),(3,4,6))


def _config(config):
    cfg = dict(DEFAULT_CONFIG)
    if config:
        if set(config)-set(cfg):
            raise ValueError("Unknown interior-observation fields")
        cfg.update(config)
    if not all(np.isfinite(value) for value in cfg.values()):
        raise ValueError("Finite interior constants required")
    if any(cfg[key] <= 0 for key in cfg if key != "seed"):
        raise ValueError("Positive interior constants required")
    for key in ("along_samples", "minimum_inlier_count", "repetitions", "block_length", "seed", "minimum_successful_pair_resamples"):
        if cfg[key] != int(cfg[key]):
            raise ValueError("Integer count/seed required")
        cfg[key] = int(cfg[key])
    if (cfg["along_min_px"] >= cfg["along_max_px"] or cfg["maximum_corner_shift_px"] > 12.
            or cfg["maximum_blend_alpha"] > .5 or cfg["detector_sigma_assumed_px"] != 2.):
        raise ValueError("Bounded declared search/blend required")
    return cfg


def image_pixel_sha256(image_bgr):
    """Hash actual BGR uint8 pixels, dimensions and channel encoding; no files."""
    image = np.asarray(image_bgr)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or min(image.shape[:2]) < 1:
        raise ValueError("Nonempty uint8 BGR image required")
    digest = hashlib.sha256(b"interior-BGR-uint8-v1\0")
    digest.update(np.asarray(image.shape, dtype="<i8").tobytes())
    digest.update(np.ascontiguousarray(image).tobytes())
    return digest.hexdigest()


def _context(points, K, initial_pose, observation, image_hw, cfg):
    q, K = np.asarray(points, float), np.asarray(K, float)
    hw = np.asarray(image_hw, float)
    if q.shape not in ((8,2),(9,2)) or K.shape != (3,3) or not np.isfinite(K).all() or min(K[0,0],K[1,1]) <= 0:
        raise ValueError("Native q8/9 and raw positive-focal K required")
    if hw.shape != (2,) or not np.isfinite(hw).all() or np.any(hw <= 0) or np.any(hw != np.floor(hw)):
        raise ValueError("Actual integral image_hw required")
    R, t, d = np.asarray(initial_pose["R_cf"], float), np.asarray(initial_pose["centroid"], float), np.asarray(initial_pose["cf_extents"], float)
    if not initial_pose.get("available") or R.shape != (3,3) or t.shape != (3,) or d.shape != (3,) or not all(np.isfinite(a).all() for a in (R,t,d)) or np.any(d <= 0) or not np.allclose(R.T@R,np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(R),1.,atol=1e-6):
        raise ValueError("Proper finite initial cuboid pose required")
    X = C.cuboid(*d)
    if np.any((X@R.T+t)[:,2] <= 1e-6):
        raise ValueError("Initial model must be in front of camera")
    H = np.asarray(observation["self_hidden"], bool)
    if H.shape != (8,) or not np.array_equal(H,C.self_hidden(X,R,t)):
        raise ValueError("Fixed H must bind original initial pose")
    if not np.array_equal(np.asarray(observation["raw_points"],float),q,equal_nan=True):
        raise ValueError("Original native raw query binding differs")
    P = C.project(X,R,t,K)
    hull = np.zeros(8,bool)
    hull[cv2.convexHull(P.astype(np.float32),returnPoints=False).ravel()] = True
    h,w = map(int,hw)
    projected_inside = (P[:,0]>=0)&(P[:,0]<w)&(P[:,1]>=0)&(P[:,1]<h)
    valid = np.isfinite(q[:8]).all(1)&~np.all(q[:8]==-1,axis=1)
    inside = valid&(q[:8,0]>=0)&(q[:8,0]<w)&(q[:8,1]>=0)&(q[:8,1]<h)
    bv = observation["boundary_visibility"]
    if not np.array_equal(bv["expected_outline"],hull) or not np.array_equal(bv["conditional_truncation"],~projected_inside):
        raise ValueError("Original outline/truncation must bind initial pose/canvas")
    app = observation["visibility_appearance"]
    p = np.asarray(app["visibility_score"],float)
    if (p.shape != (8,) or not np.isfinite(p).all() or np.any((p<0)|(p>1))
            or app.get("diagnostics",{}).get("joint_observation")):
        raise ValueError("Independent RGB visibility required, not P_usable")
    geometric = ~hull&~H&projected_inside&inside
    eligible = geometric&(p>=cfg["minimum_visibility"])
    # An independent other-instance cue must not be overruled by same-color RGB.
    from .boundary_evidence import _occluder_evidence
    external = observation.get("occluding_evidence",{})
    for k in np.flatnonzero(eligible):
        if _occluder_evidence(external.get(int(k),external.get(str(k))),int(k),q[k])["independent_region_supported"]:
            eligible[k] = False
    return dict(q=q,K=K,R=R,t=t,d=d,H=H,P=P,inside=inside,hull=hull,geometric=geometric,
                eligible=eligible,hw=[h,w],probability=p)


def _sample(array, positions):
    return cv2.remap(array,positions[...,0].astype(np.float32),positions[...,1].astype(np.float32),
                     cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT).astype(float)


def _observe_ray(lab,gx,gy,context,k,mate,cfg):
    delta = context["P"][mate]-context["P"][k]
    length = float(np.linalg.norm(delta))
    metadata = dict(edge=k*3+NEIGHBORS[k].index(mate),endpoints=[k,mate],native_corner=k,
                    native_neighbor=mate,shared_edge_key=":".join(map(str,sorted((k,mate)))),supported=False)
    if length < cfg["minimum_projected_incident_length_px"] or context["H"][mate]:
        return dict(metadata,reason="short_or_conditionally_hidden_incident_ray")
    guide = delta/length
    normal = np.array([-guide[1],guide[0]])
    start,end = cfg["along_min_px"],min(cfg["along_max_px"],.8*length)
    distances = np.linspace(start,end,cfg["along_samples"])
    offsets = np.arange(-cfg["normal_search_radius_px"],cfg["normal_search_radius_px"]+cfg["normal_step_px"]/2,cfg["normal_step_px"])
    pos = context["q"][k]+distances[:,None,None]*guide+offsets[None,:,None]*normal
    x,y = _sample(gx,pos),_sample(gy,pos)
    energy = np.linalg.norm(x*normal[0]+y*normal[1],axis=-1)
    total = np.sqrt(np.sum(x*x+y*y,axis=-1))
    alignment = energy/np.maximum(total,1e-12)
    probe = cfg["edge_probe_px"]
    color = np.linalg.norm(_sample(lab,pos+normal*probe)-_sample(lab,pos-normal*probe),axis=-1)
    h,w = context["hw"]
    a,b = pos+normal*probe,pos-normal*probe
    valid = (a[...,0]>=0)&(a[...,0]<w-1)&(a[...,1]>=0)&(a[...,1]<h-1)&(b[...,0]>=0)&(b[...,0]<w-1)&(b[...,1]>=0)&(b[...,1]<h-1)
    available = valid&(energy>=cfg["minimum_normal_gradient"])&(color>=cfg["minimum_color_transition"])&(alignment>=cfg["minimum_normal_alignment"])
    score = np.where(available,energy*alignment-.05*np.abs(offsets)[None,:],-np.inf)
    best = np.argmax(score,axis=1)
    use = np.isfinite(score[np.arange(len(distances)),best])
    metadata.update(samples=len(distances),transition_sample_count=int(use.sum()),expected_direction=guide.tolist(),
                    raw_query_xy=context["q"][k].tolist(),along_range_px=[start,end])
    if use.sum() < cfg["minimum_inlier_count"]:
        return dict(metadata,reason="insufficient_actual_RGB_gradient_samples")
    points = pos[np.arange(len(distances))[use],best[use]]
    contrasts = color[np.arange(len(distances))[use],best[use]]
    confidence = np.clip(contrasts/40.,.05,1.)
    t = .15+.7*(distances[use]-start)/(end-start)
    fitcfg = dict(cfg,along_interval=[.15,.85])
    fitted,diagnostic = B._fit_line(points,confidence,t,normal,len(distances),fitcfg)
    metadata.update(diagnostic,normal_gradient_at_samples=energy[np.arange(len(distances))[use],best[use]].tolist(),
                    normal_alignment_at_samples=alignment[np.arange(len(distances))[use],best[use]].tolist(),
                    color_transition_at_samples=contrasts.tolist(),supported=fitted is not None)
    return metadata


def infer_interior_observations(image_bgr,points,K,initial_pose,observation_result,config=None):
    """Return separate typed observations; never modify raw/fused/center arrays."""
    cfg = _config(config)
    image = np.asarray(image_bgr)
    digest = image_pixel_sha256(image)
    context = _context(points,K,initial_pose,observation_result,image.shape[:2],cfg)
    lab = cv2.GaussianBlur(cv2.cvtColor(image,cv2.COLOR_BGR2LAB).astype(np.float32),(3,3),cfg["gaussian_sigma_px"])
    gx = cv2.Sobel(lab,cv2.CV_32F,1,0,ksize=3)/8.
    gy = cv2.Sobel(lab,cv2.CV_32F,0,1,ksize=3)/8.
    records,diagnostics = {},[]
    for k in np.flatnonzero(context["eligible"]):
        k = int(k)
        rays = [_observe_ray(lab,gx,gy,context,k,mate,cfg) for mate in NEIGHBORS[k]]
        supported = [r for r in rays if r["supported"]]
        pairs = []
        for i in range(len(supported)):
            for j in range(i+1,len(supported)):
                first,second = supported[i],supported[j]
                matrix = np.array([first["normal"],second["normal"]])
                sine = abs(float(np.linalg.det(matrix)))
                if sine >= cfg["minimum_intersection_sine"]:
                    candidate = np.linalg.solve(matrix,-np.array([first["offset"],second["offset"]]))
                    pairs.append((sine,first,second,candidate))
        diagnostic = dict(native_corner=k,rays=rays,accepted=False,reason="fewer_than_two_nonparallel_RGB_incident_lines")
        if pairs:
            disagreement = max(float(np.linalg.norm(a[3]-b[3])) for a in pairs for b in pairs)
            sine,first,second,candidate = max(pairs,key=lambda p:(p[0],-p[1]["edge"],-p[2]["edge"]))
            shift = float(np.linalg.norm(candidate-context["q"][k]))
            h,w = context["hw"]
            if disagreement > cfg["maximum_pair_disagreement_px"]:
                diagnostic["reason"] = "inconsistent_three_line_junctions"
            elif shift > cfg["maximum_corner_shift_px"] or not (0<=candidate[0]<w and 0<=candidate[1]<h):
                diagnostic["reason"] = "intersection_outside_query_trust_window_or_canvas"
            else:
                correction = dict(edge_records=[first,second],corner_records=[dict(corner=k,refined=True,
                    incident_shared_edges=[first["edge"],second["edge"]],candidate=candidate.tolist())])
                covariance,_ = intersection_uncertainty(correction,repetitions=cfg["repetitions"],block_length=cfg["block_length"],normal_floor_px=cfg["normal_floor_px"],seed=cfg["seed"])
                cov = covariance.get(k)
                if not cov or cov["successful_pair_resamples"] < cfg["minimum_successful_pair_resamples"]:
                    diagnostic["reason"] = "insufficient_conditional_line_resamples"
                else:
                    variance = cov["eigenvalues_px2"][-1]
                    alpha = min(cfg["maximum_blend_alpha"],cfg["detector_sigma_assumed_px"]**2/(cfg["detector_sigma_assumed_px"]**2+variance))
                    measured = context["q"][k]+alpha*(candidate-context["q"][k])
                    records[str(k)] = dict(native_corner=k,coordinate_source="OBSERVED_RGB_JUNCTION",
                        evidence_level="LOWER_EVIDENCE_RGB_INCIDENT_JUNCTION",raw_query_xy=context["q"][k].tolist(),
                        candidate_xy=candidate.tolist(),measured_xy=measured.tolist(),blend_alpha=alpha,
                        detector_sigma_assumed_px=cfg["detector_sigma_assumed_px"],image_hw=context["hw"],
                        image_pixel_sha256=digest,lines=[first,second],incident_native_neighbor_ids=[first["native_neighbor"],second["native_neighbor"]],
                        intersection_sine=sine,intersection_shift_px=shift,applied_shift_px=float(np.linalg.norm(measured-context["q"][k])),
                        covariance_px2=cov["covariance_px2"],conditional_covariance=cov,
                        successful_pair_resamples=cov["successful_pair_resamples"],p_visible=float(context["probability"][k]),
                        coordinate_confidence_is_calibrated=False,semantic_visibility_certified=False,
                        projected_points_used=False,original_initial_geometry_nonhull=True)
                    diagnostic.update(accepted=True,reason="actual_RGB_incident_line_intersection_lower_evidence")
        diagnostics.append(diagnostic)
    result = C.finite(dict(schema="interior_rgb_incident_junction_v1",configuration=cfg,
        input_binding=dict(raw_points=np.asarray(points,float).tolist(),K=np.asarray(K,float).tolist(),
            R_cf=context["R"].tolist(),centroid=context["t"].tolist(),cf_extents=context["d"].tolist(),self_hidden=context["H"].tolist()),
        image_binding=dict(image_pixel_sha256=digest,image_hw=context["hw"],encoding="BGR_uint8_contiguous_with_shape_v1"),
        eligible_interior_ids=np.flatnonzero(context["eligible"]).tolist(),supplemental_observations=records,
        diagnostics=diagnostics,provenance=dict(GT_inputs=False,models_loaded=0,pose_fits=0,extra_SubPix_passes=0,
            projected_coordinates_used_as_measurements=False,initial_projected_incident_directions_search_guide_only=True,
            original_raw_fused_and_center_unchanged=True,semantic_corner_identity_verified=False,
            covariance_is_conditional=True,detector_sigma_is_assumed=True,shared_incident_edges_can_be_correlated=True),
        limits=["Repeated pallet grid or virtual/rounded junction can pass local gradient support; no requested3Dcorner certainty",
                "Covariance omits semantic/model bias and rawdetector/gradient correlation; sigma2 blend is not calibrated Bayes",
                "Initialpose/nativefamily/visibility assumptions may be wrong; no hidden or truncated completion is an observation"]))
    validate_interior_observations(result,points,K,initial_pose,observation_result,image_hw=image.shape[:2],image_pixel_sha256=digest)
    return result


def validate_interior_observations(result,points,K,initial_pose,observation_result,*,image_hw,image_pixel_sha256=None):
    """Validate producer/data bindings, not semantic identity or source pixels.

Production callers pass the hash of their actual image. Cached callers without
that hash cannot independently bind a supplied record to the actual pixels.
"""
    if result.get("schema") != "interior_rgb_incident_junction_v1":
        raise ValueError("Typed interior RGB junction schema required")
    cfg = _config(result["configuration"])
    context = _context(points,K,initial_pose,observation_result,image_hw,cfg)
    binding = result["input_binding"]
    for key,expected in (("raw_points",context["q"]),("K",context["K"]),("R_cf",context["R"]),("centroid",context["t"]),("cf_extents",context["d"]),("self_hidden",context["H"])):
        if not np.allclose(np.asarray(binding[key],float),expected,atol=1e-10,rtol=0,equal_nan=True):
            raise ValueError("Interior input binding differs: "+key)
    image_binding = result["image_binding"]
    digest = image_binding.get("image_pixel_sha256")
    actual_hash = image_pixel_sha256 or observation_result.get("image_pixel_sha256")
    if (image_binding.get("image_hw") != context["hw"] or not isinstance(digest,str) or len(digest)!=64
            or image_binding.get("encoding") != "BGR_uint8_contiguous_with_shape_v1"
            or actual_hash is not None and digest != actual_hash):
        raise ValueError("Interior actual-image binding differs")
    if result["provenance"].get("projected_coordinates_used_as_measurements") is not False:
        raise ValueError("Projected corners cannot be interior measurements")
    output = {}
    for key,record in result.get("supplemental_observations",{}).items():
        k = int(key)
        if not 0<=k<8 or record.get("native_corner")!=k or not context["eligible"][k]:
            raise ValueError("Interior native eligibility/query identity differs")
        if record.get("coordinate_source")!="OBSERVED_RGB_JUNCTION" or record.get("projected_points_used") is not False or record.get("image_pixel_sha256")!=digest or record.get("image_hw")!=context["hw"] or not np.allclose(record["raw_query_xy"],context["q"][k],atol=1e-10,rtol=0):
            raise ValueError("Interior record query/source/image binding differs")
        lines = record["lines"]
        if len(lines)!=2 or len({line["edge"] for line in lines})!=2 or len({line["native_neighbor"] for line in lines})!=2:
            raise ValueError("Two distinct actual incident lines required")
        for line in lines:
            mate = line["native_neighbor"]
            if mate not in NEIGHBORS[k] or line["endpoints"] != [k,mate] or not line.get("supported"):
                raise ValueError("Native incident line binding differs")
            delta = context["P"][mate]-context["P"][k]
            guide = delta/np.linalg.norm(delta)
            seed_normal = np.array([-guide[1],guide[0]])
            if not np.allclose(line["expected_direction"],guide,atol=1e-10,rtol=0):
                raise ValueError("Original model incident search direction differs")
            points = np.asarray(line["observed_points"],float)
            n = np.asarray(line["normal"],float)
            inlier = np.asarray(line["inlier_mask"],bool)
            confidence = np.asarray(line["observation_confidence"],float)
            if points.ndim!=2 or points.shape[1]!=2 or not np.isfinite(points).all() or inlier.shape!=(len(points),) or confidence.shape!=(len(points),) or n.shape!=(2,) or not np.isfinite(n).all() or not np.isclose(np.linalg.norm(n),1.,atol=1e-6):
                raise ValueError("Actual finite line sample payload required")
            angle = float(np.degrees(np.arccos(np.clip(n@seed_normal,-1,1))))
            if angle>cfg["maximum_angle_change_deg"]+1e-6 or not np.isclose(angle,line["angle_change_deg"],atol=1e-6):
                raise ValueError("Actual line is inconsistent with original incident direction")
            if line.get("native_corner")!=k or line.get("edge")!=k*3+NEIGHBORS[k].index(mate) or not np.allclose(line["raw_query_xy"],context["q"][k],atol=1e-10,rtol=0):
                raise ValueError("Actual line raw/native identity differs")
            start,end = line["along_range_px"]
            location = (points-context["q"][k])@guide
            normal_offset = (points-context["q"][k])@np.array([-guide[1],guide[0]])
            if np.any(location<start-1e-6) or np.any(location>end+1e-6) or np.any(np.abs(normal_offset)>cfg["normal_search_radius_px"]+1e-6):
                raise ValueError("Samples outside original local RGB search window")
            h,w = context["hw"]
            for probe in (points+seed_normal*cfg["edge_probe_px"],points-seed_normal*cfg["edge_probe_px"]):
                if np.any((probe[:,0]<0)|(probe[:,0]>=w-1)|(probe[:,1]<0)|(probe[:,1]>=h-1)):
                    raise ValueError("RGB sample probe outside actual canvas")
            residual = np.abs(points@n+float(line["offset"]))
            expected_inlier = residual<=cfg["huber_delta_px"]
            count = int(inlier.sum())
            span = float(np.ptp(location[inlier]))/(end-start) if count else 0.
            rms = float(np.sqrt(np.mean(residual[inlier]**2))) if count else np.inf
            if (not np.array_equal(inlier,expected_inlier) or count<cfg["minimum_inlier_count"]
                    or line["samples"]!=cfg["along_samples"] or count/line["samples"]<cfg["minimum_inlier_fraction"]
                    or span<cfg["minimum_inlier_span_fraction"] or rms>cfg["maximum_line_rms_px"]
                    or not np.isclose(rms,line["inlier_rms_px"],atol=1e-7) or line["inlier_count"]!=count
                    or not np.isclose(line["inlier_fraction"],count/line["samples"],atol=1e-8)
                    or not np.isclose(line["inlier_span_fraction"],span,atol=1e-7)):
                raise ValueError("Actual line support quality/binding differs")
            for name,threshold in (("normal_gradient_at_samples",cfg["minimum_normal_gradient"]),("normal_alignment_at_samples",cfg["minimum_normal_alignment"]),("color_transition_at_samples",cfg["minimum_color_transition"])):
                values = np.asarray(line[name],float)
                if values.shape!=(len(points),) or not np.isfinite(values).all() or np.any(values<threshold-1e-8):
                    raise ValueError("Actual RGB gradient support differs")
            if not np.allclose(confidence,np.clip(np.asarray(line["color_transition_at_samples"])/40.,.05,1.),atol=1e-8,rtol=0):
                raise ValueError("Observed line confidence must bind retained RGB contrast")
        matrix = np.asarray([line["normal"] for line in lines],float)
        sine = abs(float(np.linalg.det(matrix)))
        if sine<cfg["minimum_intersection_sine"]:
            raise ValueError("Parallel image lines cannot localize an interior point")
        candidate = np.linalg.solve(matrix,-np.asarray([line["offset"] for line in lines]))
        shift = float(np.linalg.norm(candidate-context["q"][k]));h,w=context["hw"]
        if shift>cfg["maximum_corner_shift_px"]+1e-8 or not (0<=candidate[0]<w and 0<=candidate[1]<h) or not np.allclose(candidate,record["candidate_xy"],atol=1e-8,rtol=0):
            raise ValueError("Bounded actual image intersection required")
        correction = dict(edge_records=lines,corner_records=[dict(corner=k,refined=True,incident_shared_edges=[line["edge"] for line in lines],candidate=candidate.tolist())])
        cov,_ = intersection_uncertainty(correction,repetitions=cfg["repetitions"],block_length=cfg["block_length"],normal_floor_px=cfg["normal_floor_px"],seed=cfg["seed"])
        expected = cov.get(k)
        if expected is None or expected["successful_pair_resamples"]<cfg["minimum_successful_pair_resamples"] or record["successful_pair_resamples"]!=expected["successful_pair_resamples"] or not np.allclose(record["covariance_px2"],expected["covariance_px2"],atol=1e-8,rtol=0):
            raise ValueError("Conditional covariance must bind actual retained lines")
        for name,value in expected.items():
            if name not in record["conditional_covariance"] or not np.allclose(record["conditional_covariance"][name],value,atol=1e-8,rtol=0):
                raise ValueError("Conditional covariance diagnostic binding differs")
        alpha = min(cfg["maximum_blend_alpha"],cfg["detector_sigma_assumed_px"]**2/(cfg["detector_sigma_assumed_px"]**2+expected["eigenvalues_px2"][-1]))
        measured = context["q"][k]+alpha*(candidate-context["q"][k])
        if record["detector_sigma_assumed_px"]!=cfg["detector_sigma_assumed_px"] or not np.isclose(record["blend_alpha"],alpha,atol=1e-10) or not np.allclose(record["measured_xy"],measured,atol=1e-8,rtol=0) or np.linalg.norm(measured-context["q"][k])>6.+1e-8:
            raise ValueError("Declared sigma2 bounded blend differs")
        if (not np.isclose(record["intersection_sine"],sine,atol=1e-8)
                or not np.isclose(record["intersection_shift_px"],shift,atol=1e-8)
                or not np.isclose(record["applied_shift_px"],np.linalg.norm(measured-context["q"][k]),atol=1e-8)
                or record["incident_native_neighbor_ids"]!=[line["native_neighbor"] for line in lines]):
            raise ValueError("Measured junction diagnostic/source binding differs")
        output[str(k)] = deepcopy(record)
        output[str(k)]["actual_image_hash_binding_verified"] = actual_hash is not None
        output[str(k)]["semantic_corner_identity_verified"] = False
        output[str(k)]["source_RGB_gradient_values_verified_by_validator"] = False
    return output
