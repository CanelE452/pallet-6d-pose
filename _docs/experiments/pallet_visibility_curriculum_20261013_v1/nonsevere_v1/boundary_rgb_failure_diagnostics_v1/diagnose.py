"""Two preselected TEST49 failures, postseal diagnosis only; no inference change."""
from pathlib import Path
import hashlib
import json
import gzip
from datetime import datetime, timezone

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[5]
NAMESPACE = Path(__file__).resolve().parent
RUN = NAMESPACE.parent / "test_reused_v1"
IDS = ["eval_pallet07:1778652140531310080", "eval_pallet07:1778652166837872128"]
QCOLOR, PCOLOR, GCOLOR = (255, 255, 0), (0, 255, 255), (255, 0, 255)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cuboid(dims):
    a, b, c = np.asarray(dims, float) / 2
    return np.array([[-a,-b,-c],[a,-b,-c],[a,b,-c],[-a,b,-c],
                     [-a,-b,c],[a,-b,c],[a,b,c],[-a,b,c]])


def project(xyz, R, t, K):
    camera = xyz @ np.asarray(R).T + np.asarray(t)
    h = camera @ np.asarray(K).T
    return h[:, :2] / h[:, 2:3]


def ry(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])


def phase_diagnostic(frame, target, metadata):
    """Object-coordinate diagnostics; never replace the locked correspondence."""
    pose = frame["initial_poses"]["N3_SUBPIX"]
    Q = np.asarray(pose["R_cf"]).T @ np.asarray(pose["R_physical"])
    truth = target["truth"]
    reference_Q = np.asarray(truth["body_R"]).T @ np.asarray(truth["R"])
    initial_physical = cuboid(pose["cf_extents"]) @ Q.T
    reference_physical = cuboid(truth["body_xyz"]) @ reference_Q.T
    mapping = np.asarray(metadata["native_to_reference_corner_ids"], int)
    gaps, hypothetical = [], []
    for j in range(truth["order"]):
        points = initial_physical @ ry(j * 2 * np.pi / truth["order"]).T
        gaps.append(float(np.linalg.norm(points-reference_physical[mapping], axis=1).max()))
        distances = np.linalg.norm(points[:, None]-reference_physical[None], axis=-1)
        # This geometric permutation is disclosed as QA, never applied to labels,
        # native points, either overlay, model probabilities or primary metrics.
        hypothetical.append(dict(symmetry_index=j, native_to_reference_geometry_only=distances.argmin(1).tolist(),
            maximum_geometric_gap_m=float(distances.min(1).max()), used_for_any_remapping=False))
    R, G = np.asarray(pose["R_physical"]), np.asarray(truth["R"])
    angles = []
    for j in range(truth["order"]):
        relative = (G @ ry(j*2*np.pi/truth["order"])).T @ R
        angles.append(float(np.degrees(np.arccos(np.clip((np.trace(relative)-1)/2,-1,1)))))
    return dict(locked_native_to_reference_corner_ids=mapping.tolist(),
        fixed_correspondence_compatible=min(gaps)<1e-6,
        maximum_locked_object_correspondence_gap_m=min(gaps),
        initial_cf_extents=pose["cf_extents"], reference_body_extents=truth["body_xyz"],
        registered_extents=truth["xyz"], physical_from_initial_cf=Q.tolist(),
        physical_from_reference_body=reference_Q.tolist(),
        disclosed_geometric_correspondence_diagnostics=hypothetical,
        reconstructed_reference_rotation_difference_deg=min(angles),
        reconstructed_reference_translation_difference_cm=float(np.linalg.norm(
            np.asarray(pose["centroid"])-np.asarray(truth["t"]))*100),
        independent_physical_pose_truth=metadata["independent_physical_pose_truth"],
        physical_pose_conclusion="Historical annotation reconstruction only; correspondence mismatch alone does not prove a wrong physical pose.")


def pixel_summary(image, point):
    x, y = np.rint(point).astype(int)
    h, w = image.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return dict(in_frame=False)
    patch = image[max(0,y-2):min(h,y+3),max(0,x-2):min(w,x+3)]
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
    return dict(in_frame=True, nearest_integer_xy=[int(x),int(y)],
        point_BGR=image[y,x].astype(int).tolist(),
        patch_5x5_BGR_mean=patch.reshape(-1,3).mean(0).tolist(),
        patch_5x5_Lab_mean=lab.reshape(-1,3).mean(0).tolist(),
        diagnostic_only_never_input_to_fit=True)


def classifier_replay(appearance, model, k):
    """Reproduce the frozen saved evidence, without a CNN/head refit."""
    row = appearance["diagnostics"]["corners"][k]
    scalars = model["feature_scalars"]
    x = [float(row.get(key) or 0.) for key in scalars]
    names = list(scalars)
    for key in model["feature_vectors"]:
        x.extend(row.get(key) or [0.,0.,0.])
        names.extend(key+"_"+str(j) for j in range(3))
    x.append(min(4.,float(row.get("edge_distance_px") or 0.)/
                   max(1.,appearance["diagnostics"]["patch_radius_px"])))
    names.append("edge_distance_px/patch_radius")
    x = np.asarray(x)
    terms = (x-np.asarray(model["mean"]))/np.asarray(model["scale"])*np.asarray(model["coefficients"])
    logit = float(terms.sum()+model["intercept"])
    probability = float(1/(1+np.exp(-np.clip(logit,-60,60))))
    actual = appearance["visibility_score"][k]
    assert abs(probability-actual) < 1e-12
    order = np.argsort(-np.abs(terms))[:5]
    return dict(replayed_probability=probability, absolute_saved_probability_difference=abs(probability-actual),
        intercept=model["intercept"], logit=logit,
        dominant_absolute_logit_terms=[dict(feature=names[i], raw_value=float(x[i]), contribution=float(terms[i])) for i in order],
        feature_version=model.get("feature_version","legacy_v1"), input="saved sourceCNN+rawRGB diagnostics, exactly as frozen inference")


STATE = {"DIRECT_VISIBLE":"DIR", "EXTERNAL_OCCLUDED":"EXT", "SELF_OCCLUDED":"SELF", "UNANNOTATED":"UNK"}


def draw_full(image, q, p, gt, states, valid, hull):
    canvas = image.copy()
    cv2.polylines(canvas,[np.rint(p[hull]).astype(np.int32)],True,PCOLOR,1,cv2.LINE_AA)
    h, w = image.shape[:2]
    for k in range(8):
        for point,color,marker in [(q[k],QCOLOR,cv2.MARKER_CROSS),(p[k],PCOLOR,cv2.MARKER_DIAMOND),
                                   (gt[k],GCOLOR,cv2.MARKER_TILTED_CROSS)]:
            cv2.drawMarker(canvas,tuple(np.rint(point).astype(int)),color,marker,8,1,cv2.LINE_AA)
        if 0<=q[k,0]<w and 0<=q[k,1]<h:
            cv2.putText(canvas,"q"+str(k),tuple(np.rint(q[k]).astype(int)+np.array([4,-6])),
                        cv2.FONT_HERSHEY_SIMPLEX,.35,QCOLOR,1,cv2.LINE_AA)
        if 0<=gt[k,0]<w and 0<=gt[k,1]<h:
            label="G"+str(k)+":"+STATE[states[k]]+("*" if not valid[k] else "")
            cv2.putText(canvas,label,tuple(np.rint(gt[k]).astype(int)+np.array([3,14])),
                        cv2.FONT_HERSHEY_SIMPLEX,.31,GCOLOR,1,cv2.LINE_AA)
    return canvas


def crop_panel(image, q, p, gt, k, label, marked=True):
    h, w = image.shape[:2]
    center = q[k]
    x0 = max(0,min(w-160,int(round(center[0]))-80))
    y0 = max(0,min(h-90,int(round(center[1]))-45))
    cut = image[y0:y0+90,x0:x0+160].copy()
    if marked:
        offset=np.array([x0,y0])
        for point,color,marker in [(q[k],QCOLOR,cv2.MARKER_CROSS),(p[k],PCOLOR,cv2.MARKER_DIAMOND),(gt[k],GCOLOR,cv2.MARKER_TILTED_CROSS)]:
            cv2.drawMarker(cut,tuple(np.rint(point-offset).astype(int)),color,marker,5,1)
    panel = cv2.resize(cut,(640,360),interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(panel,(0,0),(640,35),(22,22,22),-1)
    cv2.putText(panel,label+f" | native crop ({x0},{y0})..({x0+160},{y0+90})",(10,23),
                cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA)
    return panel, [x0,y0,x0+160,y0+90]


def render(image, q, p, gt, states, valid, hull, records, fid):
    canvas=np.full((1020,1600,3),24,np.uint8)
    cv2.putText(canvas,"TEST49 POSTHOC DIAGNOSTIC ONLY | "+fid,(18,29),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(canvas,"Observed q: cyan +   Initial projection: yellow diamond/outline   Locked GT: magenta x  (DIR/EXT/SELF/UNK)",(18,62),cv2.FONT_HERSHEY_SIMPLEX,.63,(255,255,255),1,cv2.LINE_AA)
    annotated=draw_full(image,q,p,gt,states,valid,hull)
    canvas[80:800,:960]=cv2.resize(annotated,(960,720),interpolation=cv2.INTER_LINEAR)
    crops=[]
    focus=[5,5] if len(records)==1 else [5,4]
    for j,k in enumerate(focus):
        marked=not (len(records)==1 and j==1)
        label=f"q{k}: observed / initial / locked GT" if marked else "Same native RGB crop; no markers"
        panel,rect=crop_panel(image,q,p,gt,k,label,marked)
        canvas[80+j*360:80+(j+1)*360,960:1600]=panel
        crops.append(dict(focus_native_corner=k,native_xyxy=rect,marked=marked))
    y=836
    for r in records:
        line=(f"q{r['native_corner']} locked label {STATE[r['human_state']]}: p_visible={r['RGB_p_visible']:.6f}, "
              f"raw-GT={r['raw_to_locked_GT_distance_px']:.2f}px, raw-initial={r['raw_to_initial_distance_px']:.2f}px")
        cv2.putText(canvas,line,(18,y),cv2.FONT_HERSHEY_SIMPLEX,.65,(240,240,240),1,cv2.LINE_AA);y+=35
    lines=["Fixed native/reference mapping is identity. All reported labels, scores and metrics retain that mapping.",
           "q5 samples exposed pallet material; this does not prove visibility of the labelled hidden corner."]
    if len(records)>1:
        lines.append("q4 samples blue stool; locked DIRECT G4 is on the opposite pallet corner. Do not silently relabel either point.")
    else:
        lines.append("G5 lies at the red occluder edge; q5 is displaced onto an exposed patch. Colour cannot verify corner ownership.")
    for line in lines:
        cv2.putText(canvas,line,(18,y),cv2.FONT_HERSHEY_SIMPLEX,.6,(220,220,220),1,cv2.LINE_AA);y+=34
    return canvas,crops


def main():
    output=NAMESPACE/"BOUNDARY_RGB_FAILURE_DIAGNOSTICS_V1.json"
    if output.exists():
        raise FileExistsError("Preserve the completed posthoc diagnosis")
    seal=read(RUN/"PREDICTION_SEAL.json")
    assert seal["complete"] and seal["before_scoring"] and seal["GT_read_canary"]=="PASS"
    assert sha(RUN/"PREDICTIONS.jsonl.gz")==seal["sha256"]
    protocol=read(RUN/"PROTOCOL.json")
    cohort=read(NAMESPACE.parent/"COHORT_PROTOCOL.json")
    assert set(IDS)<=set(protocol["ids"])
    assert not set(IDS)&set(cohort["excluded_severe_ids"])
    hashes={str(path):sha(path) for path in [RUN/"PREDICTIONS.jsonl.gz",RUN/"PREDICTION_SEAL.json",
        RUN/"PROTOCOL.json",RUN/"ISOLATED_TEST_INPUTS.json",RUN/"ISOLATED_TEST_SCORING.json",
        RUN/"GEOMETRIC_BOUNDARY_VISIBILITY_AUDIT_V1.json",NAMESPACE.parent/"visibility_logistic_CAL46_v2.json"]}
    all_frames=read(RUN/"ISOLATED_TEST_INPUTS.json")["frames"]
    frames={f["id"]:f for f in all_frames if f["id"] in IDS}
    assert len(all_frames)==49 and set(frames)==set(IDS)
    # Isolated labels are read only after the seal checks above; only these two
    # selected rows are accessed, with no main broad TARGETS or metadata access.
    isolated=read(RUN/"ISOLATED_TEST_SCORING.json")["rows"]
    scoring={fid:isolated[fid] for fid in IDS}
    audit=read(RUN/"GEOMETRIC_BOUNDARY_VISIBILITY_AUDIT_V1.json")
    failures={fid:[r for r in audit["failure_cases"] if r["id"]==fid] for fid in IDS}
    prediction={}
    with gzip.open(RUN/"PREDICTIONS.jsonl.gz","rt") as handle:
        for line in handle:
            row=json.loads(line)
            if row["id"] in IDS and row["method"]=="SEGMENTED_RGB_CURRICULUM":prediction[row["id"]]=row
    model=read(NAMESPACE.parent/"visibility_logistic_CAL46_v2.json")
    assert model["visible_threshold"]==.2
    cases=[]
    for fid in IDS:
        f=frames[fid];s=scoring[fid];rgb=prediction[fid]
        image_path=ROOT/f["image"]
        assert sha(image_path)==f["image_sha256"]
        image=cv2.imread(str(image_path));assert image is not None and list(image.shape[:2])==f["raw_hw"]
        q=np.asarray(f["points"]["N3_SUBPIX"])[:8]
        init=f["initial_poses"]["N3_SUBPIX"]
        p=project(cuboid(init["cf_extents"]),init["R_cf"],init["centroid"],f["K"])
        gt=np.asarray(s["target"]["gt"])[:8]
        locked=np.asarray(s["metadata"]["native_to_reference_corner_ids"])
        assert np.array_equal(locked,np.arange(8))
        assert np.array_equal(q,np.asarray(rgb["original_points"]))
        states=s["metadata"]["human_states_native"]
        hull=cv2.convexHull(p.astype(np.float32),returnPoints=False).ravel()
        selection=next(r for r in audit["frame_selection_records"] if r["id"]==fid)
        assert selection["initial_hull_native_ids"]==hull.tolist()
        assert selection["fixed_H"]==rgb["self_hidden"]
        records=[]
        for failure in failures[fid]:
            k=failure["corner"];assert k in selection["selected_native_ids"] and not rgb["self_hidden"][k]
            assert np.max(np.abs(p[k]-failure["initial_projected_xy"]))<1e-10
            records.append(dict(native_corner=k,locked_reference_corner=int(locked[k]),
                human_state=states[k],human_label_locked=True,initial_geometric_boundary_selected=True,
                original_self_hidden=rgb["self_hidden"][k],observed_q_xy=q[k].tolist(),initial_projected_xy=p[k].tolist(),
                locked_GT_xy=gt[locked[k]].tolist(),raw_to_locked_GT_distance_px=float(np.linalg.norm(q[k]-gt[locked[k]])),
                raw_to_initial_distance_px=float(np.linalg.norm(q[k]-p[k])),RGB_p_visible=rgb["appearance"]["visibility_score"][k],
                RGB_state=rgb["appearance"]["states"][k],RGB_boundary_score=rgb["appearance"]["boundary_score"][k],
                saved_RGB_diagnostics=rgb["appearance"]["diagnostics"]["corners"][k],
                frozen_classifier_replay=classifier_replay(rgb["appearance"],model,k),
                observed_pixel=pixel_summary(image,q[k]),locked_GT_pixel=pixel_summary(image,gt[locked[k]]),
                initial_projection_pixel=pixel_summary(image,p[k])))
        D=np.linalg.norm(q[:,None]-gt[None],axis=-1)
        # Disclose all-nearest distances only as spatial context; never change
        # correspondence or compute an alternative primary label/metric.
        spatial=dict(nearest_reference_corner_each_raw_query=D.argmin(1).tolist(),
            nearest_distance_px=D.min(1).tolist(),locked_identity_distance_px=np.diag(D).tolist(),
            used_to_change_labels_or_metrics=False)
        phase=phase_diagnostic(f,s["target"],s["metadata"])
        if fid==IDS[0]:
            interpretation=dict(observed_scene="q5 is on an exposed dark pallet grid/edge patch between the red sign and blue stool. Locked G5 lies at the red sign edge.",
                failure_mechanism="Observed query displaced onto visible pallet material, so local colour/material membership is insufficient to establish visibility of the labelled corner.",
                query_phase_context="All eight nearest image-reference IDs equal the locked identity. Object-coordinate W/D basis remains incompatible, so this is a local displacement example with a separate historical model/reference confound.",
                same_colour_occluder_verified=False,exact_corner_ownership_verified=False,
                head_failure_conclusion="High visibility is correct for a visible material patch at q5, but fails the locked requested-corner label. The available evidence cannot separate all upstream localisation/model error from classifier error.",
                Korean="관측 q5가 빨간 가림물 옆의 보이는 파렛트 표면으로 약19.55px 이동했다. 그 위치의 파렛트 색은 맞아도, 가려진 정답 코너의 가시성을 증명하지 않는다.")
        else:
            interpretation=dict(observed_scene="q5 is on the exposed pallet rim/grid; locked G5 is222.17px away on the blue stool. q4 is on the blue stool; locked DIRECT G4 is275.07px away on the opposite pallet corner.",
                failure_mechanism="Large native/reference correspondence discrepancy changes which physical location is queried. Local RGB accepts the grey pallet query and rejects the blue-stool query, while the locked labels refer to distant locations.",
                query_phase_context="Nearest image-reference IDs are[4,0,3,7,5,1,2,6], also disclosed by the geometric180-symmetry diagnostic. No permutation was applied to predictions, overlays, labels or primary metrics.",
                same_colour_occluder_verified=False,exact_corner_ownership_verified=False,
                head_failure_conclusion="These two apparent failures are strongly confounded by locked channel/reference correspondence. They do not establish that the local RGB head misclassified the actually sampled material.",
                Korean="q5는 보이는 파렛트 테두리를, q4는 파란 의자를 읽고 있다. 잠긴 EXTERNAL/DIRECT 라벨은 각각222px와275px 떨어진 다른 코너에 대응하므로 순수한 색 판별 실패로 해석할 수 없다.")
        overlay_path=NAMESPACE/(fid.split(":")[1]+"_RGB_OVERLAY.png")
        canvas,crops=render(image,q,p,gt,states,s["target"]["valid"][:8],hull,records,fid)
        assert cv2.imwrite(str(overlay_path),canvas)
        cases.append(dict(id=fid,severity="MODERATE_OCCLUSION",image=str(image_path),image_sha256=f["image_sha256"],
            initial_pose=init,initial_outline_native_ids=hull.tolist(),fixed_initial_H=rgb["self_hidden"],
            selected_initial_geometry_corner_ids=selection["selected_native_ids"],all_original_q=q.tolist(),
            all_initial_projected_q=p.tolist(),all_locked_reference_GT=gt.tolist(),all_locked_human_states=states,
            phase_diagnostic=phase,spatial_context_only=spatial,points=records,interpretation=interpretation,
            overlay=str(overlay_path),overlay_sha256=sha(overlay_path),overlay_crops=crops))
    assert all(sha(path)==value for path,value in hashes.items())
    report=dict(schema="NONSEVERE_TEST49_BOUNDARY_RGB_FAILURE_POSTHOC_V1",
        created_UTC=datetime.now(timezone.utc).isoformat(),postseal_diagnostic_not_fit=True,
        selection="Parent preselected both HIGH_CONFIDENCE_FALSE_VISIBLE_EXTERNAL cases and second frame FALSE_ABSENT_DIRECT from frozen geometric boundary audit.",
        cases=len(cases),points=sum(len(c["points"]) for c in cases),
        inference_model_policy_changes=0,threshold_changes=0,training_or_selection_from_TEST=0,
        primary_population_and_predictions_unchanged=True,GT_or_projection_coordinates_used_for_fit=False,
        source_images_read=2,severe_images_or_old_severe_trained_heads_read=0,FINAL_reads=0,
        native_reference_remapping_applied=False,visibility_threshold=model["visible_threshold"],
        bindings=hashes,analysis_source_sha256=sha(__file__),records=cases,
        conclusions=["Initial projected-cuboid silhouette membership is conditional on initial pose/model, not certification that the sampled q pixel is the actual object/background corner.",
          "These examples verify local observation/correspondence ambiguity. They do not verify same-colour external occluders.",
          "The first frame samples exposed pallet material near a hidden labelled corner; the second frame has severe channel/reference mismatch within a MODERATE_OCCLUSION image, not severe-occlusion population inclusion.",
          "The second reconstructed rotation gap is only4.81deg despite incompatible point correspondence. The phase flag alone must not be called a wrong physical pose.",
          "Saved probabilities replay exactly from frozen sourceCNN+rawRGB diagnostics. No fresh inference, SAM mask, alternative threshold or model was selected.",
          "This posthoc TEST inspection must not be used to silently correct mappings or retune this already observed test population."])
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(output)
    print("PASS: seal/input/image hashes unchanged; frozen probabilities reproduced; two posthoc RGB overlays; no fit/remap/tuning")


if __name__=="__main__":
    main()
