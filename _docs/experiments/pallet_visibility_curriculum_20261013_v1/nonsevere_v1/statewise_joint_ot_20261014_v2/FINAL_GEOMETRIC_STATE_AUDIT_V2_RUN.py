import sys
from pathlib import Path
sys.path.insert(0,"/home/minjae/Documents/github/pallet-pose")
from collections import Counter,defaultdict
import shutil
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from scripts.research.pallet_visibility_curriculum_20261013_v1 import common as C
from scripts.research.pallet_visibility_curriculum_20261013_v1.joint_ot_pnp import project_native,solve_joint_ot_pnp

STUDY=C.DOC/"nonsevere_v1/statewise_joint_ot_20261014_v2"
EVALUATION=STUDY/"evaluation_v2"


def geometry(xyz,R,t):
    xyz=np.asarray(xyz,float);R=np.asarray(R,float);t=np.asarray(t,float)
    camera=-R.T@t;planes=ConvexHull(xyz).equations
    scale=max(float(np.linalg.norm(np.ptp(xyz,axis=0))),1e-9)
    values=[]
    for i,x in enumerate(xyz):
        incident=np.abs(planes[:,:3]@x+planes[:,3]) <= 1e-8*scale
        normals=planes[incident,:3]
        cosines=normals@(camera-x)/(np.linalg.norm(normals,axis=1)*np.linalg.norm(camera-x))
        maximum=float(cosines.max())
        values.append(dict(native_corner=i,maximum_incident_outward_ray_cosine=maximum,
            self_hidden_strict=maximum < -1e-10,
            self_hidden_initial_2deg_policy=maximum < -float(np.sin(np.deg2rad(2))),
            grazing_2deg_band=abs(maximum)<=float(np.sin(np.deg2rad(2)))))
    return values


def audit():
    destination=STUDY/"FINAL_GEOMETRIC_STATE_AUDIT_V2.json"
    if destination.exists():raise FileExistsError(destination)
    seal=C.read(EVALUATION/"PREDICTION_SEAL.json")
    assert seal["before_scoring"] and C.sha(EVALUATION/"PREDICTIONS.jsonl.gz")==seal["predictions_sha256"]
    protocol=C.read(EVALUATION/"PROTOCOL.json")
    assert C.sha(EVALUATION/"PROTOCOL.json")==seal["protocol_sha256"]
    predictions=list(C.rows(EVALUATION/"PREDICTIONS.jsonl.gz"));assert[p["id"]for p in predictions]==protocol["ids"]
    rows=[]
    for p in predictions:
        runtime=p["runtime"];xyz=runtime["initial_geometry"]["native_xyz"]
        coarse=p["initial_pose"];final=runtime["actual_pose"]
        initial=geometry(xyz,coarse["R_cf"],coarse["centroid"])
        chosen=geometry(xyz,final["R_cf"],final["centroid"])
        states=runtime["states"]
        BI=[i for i,s in enumerate(states)if s in["B","I"]]
        initial_S=[i for i,s in enumerate(states)if s=="S"]
        final_S=[i for i,v in enumerate(chosen)if v["self_hidden_initial_2deg_policy"]]
        BI_strict_bad=[i for i in BI if chosen[i]["self_hidden_strict"]]
        BI_2deg_bad=[i for i in BI if chosen[i]["self_hidden_initial_2deg_policy"]]
        rows.append(dict(id=p["id"],partition=p["partition"],cohort=p["cohort"],accepted=runtime["accepted"],
            pose_source=final["source"],states=states,BI_ids=BI,
            initial_S_ids=initial_S,final_geometric_S_ids=final_S,
            initial_S_now_nonself_ids=sorted(set(initial_S)-set(final_S)),
            initial_nonS_now_self_ids=sorted(set(final_S)-set(initial_S)),
            S_set_changed=sorted(initial_S)!=sorted(final_S),
            final_BI_selfhidden_strict_ids=BI_strict_bad,final_BI_selfhidden_2deg_ids=BI_2deg_bad,
            initial_BI_selfhidden_strict_ids=[i for i in BI if initial[i]["self_hidden_strict"]],
            final_initial_nonS_state_selfhidden_2deg={state:[i for i,s in enumerate(states)
                if s==state and chosen[i]["self_hidden_initial_2deg_policy"]] for state in["B","I","O","T","U"]},
            initial_geometry=initial,final_geometry=chosen,
            BI_measurement_fit_audit=runtime["joint_pose"].get("visible_protection_audit")))
    summaries={}
    for part in["CAL","TEST","ALL"]:
        primary=[r for r in rows if r["cohort"]=="PRIMARY_NONSEVERE" and (part=="ALL"or r["partition"]==part)]
        accepted=[r for r in primary if r["accepted"]]
        summaries[part]=dict(primary_frames=len(primary),accepted_primary_frames=len(accepted),
            predicted_state_counts_all_primary=dict(Counter(s for r in primary for s in r["states"])),
            predicted_state_counts_accepted_primary=dict(Counter(s for r in accepted for s in r["states"])),
            accepted_with_BI_selfhidden_strict=sum(bool(r["final_BI_selfhidden_strict_ids"])for r in accepted),
            accepted_BI_selfhidden_strict_corner_count=sum(len(r["final_BI_selfhidden_strict_ids"])for r in accepted),
            accepted_with_BI_selfhidden_initial2deg=sum(bool(r["final_BI_selfhidden_2deg_ids"])for r in accepted),
            accepted_BI_selfhidden_initial2deg_corner_count=sum(len(r["final_BI_selfhidden_2deg_ids"])for r in accepted),
            accepted_S_set_changed=sum(r["S_set_changed"]for r in accepted),
            accepted_initialS_now_nonself_corner_count=sum(len(r["initial_S_now_nonself_ids"])for r in accepted),
            accepted_initial_nonS_now_self_corner_count=sum(len(r["initial_nonS_now_self_ids"])for r in accepted),
            accepted_problematic_BI_ids=[dict(id=r["id"],strict=r["final_BI_selfhidden_strict_ids"],
                conservative_2deg=r["final_BI_selfhidden_2deg_ids"])for r in accepted if r["final_BI_selfhidden_strict_ids"]],
            accepted_finalself_by_initial_nonS_state={s:sum(len(r["final_initial_nonS_state_selfhidden_2deg"][s])for r in accepted)
                                                     for s in["B","I","O","T","U"]})
    report=dict(schema="FINAL_GEOMETRIC_STATE_AUDIT_V2",prediction_seal_sha256=C.sha(EVALUATION/"PREDICTION_SEAL.json"),
        predictions_sha256=seal["predictions_sha256"],GT_reads=0,scoring_outputs_read=False,
        original_predictions_or_sources_changed=False,
        geometry_method="registered native convex-hull outward planes; camera_in_object=-R_cf.T@centroid",
        strict_hidden_definition="all incident outward ray cosines<0 (1e-10 numerical tolerance)",
        conservative_definition="max incident cosine < -sin(2deg), same initial S policy",
        thresholds_selected_from_test_labels=False,summary=summaries,rows=rows,
        limitations=["Diagnostic only; frozen selection rules were not changed",
            "Checks model self-visibility consistency, not physical-corner or RGB correspondence correctness",
            "S retains its initial conditional assignment; a changed final geometric S set is logged without relabeling"])
    C.write(destination,report)
    shutil.copyfile(__file__,STUDY/"FINAL_GEOMETRIC_STATE_AUDIT_V2_RUN.py")
    print(summaries,flush=True)


def fixture():
    from itertools import product
    destination=STUDY/"PHYSICALLY_VISIBLE_BI_P3P_COUNTEREXAMPLE_V2.json"
    if destination.exists():raise FileExistsError(destination)
    xyz=np.array(list(product([-.5,.5],[-.2,.2],[-.6,.6])))
    K=np.array([[600.,25.,320.],[0.,620.,240.],[0.,0.,1.]])
    R=Rotation.from_rotvec([-.4611806149383468,.4522670776283566,-.7933895902003064]).as_matrix()
    t=np.array([.03840692188819134,-.037316089277514036,4.700156034328923])
    q,_=project_native(xyz,R,t,K);ids=[0,2,3]
    states=["B","S","B","B","U","U","U","U"]
    measurements=[dict(native_corner=i,xy=q[i].tolist(),covariance_px2=np.eye(2).tolist(),protection_px=.05)for i in ids]
    result=solve_joint_ot_pnp(xyz,K,states,measurements,[],[480,640],
        config=dict(global_rotation_starts=0,analytic_jacobian=True))
    initial=geometry(xyz,R,t);selected=geometry(xyz,result["R"],result["t"])
    assert all(not initial[i]["self_hidden_strict"] for i in ids)
    assert result["accepted"] and selected[2]["self_hidden_initial_2deg_policy"]
    C.write(destination,dict(schema="PHYSICALLY_VISIBLE_BI_P3P_COUNTEREXAMPLE_V2",GT_reads=0,
        source="generated cuboid/pose analytic image projections, no dataset or external labels",
        native_xyz=xyz.tolist(),K=K.tolist(),image_hw=[480,640],states=states,visible=measurements,
        generated_initial_pose=dict(R=R.tolist(),t=t.tolist()),initial_geometry=initial,
        accepted_selected_geometry=selected,solver_result=result,
        root_problem="three physically visible initial B corners admit a positive-depth exact-P3P branch whose B2 is model-self-hidden",
        conceptual_conclusion="positive depth and immutable BI reprojection constraints alone do not establish direct-corner model visibility",
        no_core_selection_rule_change=True))
    print(dict(physically_valid_fixture_BI=ids,initial_maxcos=[initial[i]["maximum_incident_outward_ray_cosine"]for i in ids],
               accepted_maxcos=[selected[i]["maximum_incident_outward_ray_cosine"]for i in ids]),flush=True)

if __name__=="__main__":audit();fixture()
