"""Postseal process plates and a local, full-population interactive gallery.

Nothing is read, selected or rendered before an explicit root postseal GO.
Representative arms are supplied by the campaign's prior selection, never
chosen here by PRIMARY performance. The renderer computes no pose accuracy,
new observations, model forwards, PnP, optimization or learned predictions.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import html
import json
import math
from pathlib import Path
import resource
import textwrap
import time

import numpy as np

SELECTION_RULES = {
    "RECOVERY": "PRIMARY; E0 failure/new success; lexicographically first immutable ID",
    "DAMAGE": "PRIMARY; E0 success/new failure; largest max(newT/5,newR/5), tie immutable ID",
    "BOTH_FAILURE_WORSENING": "PRIMARY; both fail; largest positive max(deltaT/5,deltaR/5), then normalized sum, then immutable ID",
    "MAINTAINED_SUCCESS_B_UNAPPLIED": "PRIMARY; E0/new success and explicit B_treatment_active=False; lexicographically first ID",
}
STATE_COLORS = {"B":"#00B050", "I":"#2098F3", "S":"#B0B0B0", "O":"#FF9B21", "T":"#FF4D4D", "U":"#F0D52A"}
ARM_MODE = {"A_FIXED_BANK":"A", "B_FIXED_BANK":"B", "A_FULL":"A", "B_FULL":"B", "AB_FULL":"AB",
            "CTRL_EDGE_BOUNDARY":"edge", "CTRL_CARRY":"fixed", "CTRL_BUDGET":"fixed", "CTRL_FIXED":"fixed"}
FIXED_REPRESENTATIVE_ARMS=("AB_FULL","L_HEAD_s20261018","L_LINEAR_s20261018")
DISPLAY_SETTINGS={"translation_spacing_default":"log1p", "translation_spacing_options":["log1p","linear"],
                  "translation_ticks_and_tooltips":"original cm", "rotation_spacing":"linear degrees",
                  "domain":"zero through complete selected range with 4.5% margin", "tail_values_removed":0}


def _sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(4*1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _rows(path):
    opener=gzip.open if str(path).endswith(".gz") else open
    with opener(path,"rt") as f:
        for line in f:
            if line.strip():yield json.loads(line)


def _write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")


def _normalized_error(metric):
    values=[metric.get("translation_cm"),metric.get("rotation_deg")]
    return max(float(v)/5 if v is not None else float("inf") for v in values)


def select_cases(scored_rows, representative_arms,execution_status=None):
    """Postseal descriptive witnesses; no methods/thresholds are selected."""
    primary=sorted((r for r in scored_rows if r["partition"] in ("TEST","REMAINDER")),key=lambda r:r["id"])
    result=[]
    for arm in representative_arms:
        if any(arm not in r["metrics"] for r in primary):
            actual=(execution_status or {}).get(arm,{})
            if isinstance(actual,str):actual={"status":actual}
            if all(arm not in r["metrics"] for r in primary) and actual.get("status") not in (None,"EXECUTED","DONE"):
                result.extend(dict(arm=arm,kind=kind,selection_rule=rule,status="NOT_EXECUTED",execution_status=actual)
                              for kind,rule in SELECTION_RULES.items())
                continue
            raise ValueError("Explicit representative arm is absent/partly absent from sealed metrics: "+arm)
        recovery=[r for r in primary if not r["metrics"]["E0"]["success"] and r["metrics"][arm]["success"]]
        damage=[r for r in primary if r["metrics"]["E0"]["success"] and not r["metrics"][arm]["success"]]
        both=[]
        for r in primary:
            a,b=r["metrics"]["E0"],r["metrics"][arm]
            if a["success"] or b["success"]:continue
            if any(a.get(k) is None or b.get(k) is None for k in ("translation_cm","rotation_deg")):continue
            deltas=[(b[k]-a[k])/5 for k in ("translation_cm","rotation_deg")]
            if max(deltas)>0:both.append((r,max(deltas),sum(deltas)))
        kept=[r for r in primary if r["metrics"]["E0"]["success"] and r["metrics"][arm]["success"] and
              r.get("runtime_diagnostics",{}).get(arm,{}).get("B_treatment_active") is False]
        picks={"RECOVERY":recovery[0] if recovery else None,
               "DAMAGE":min(damage,key=lambda r:(-_normalized_error(r["metrics"][arm]),r["id"])) if damage else None,
               "BOTH_FAILURE_WORSENING":min(both,key=lambda v:(-v[1],-v[2],v[0]["id"]))[0] if both else None,
               "MAINTAINED_SUCCESS_B_UNAPPLIED":kept[0] if kept else None}
        for kind, row in picks.items():
            case=dict(arm=arm,kind=kind,selection_rule=SELECTION_RULES[kind],status="SELECTED" if row else "NO_CASE")
            if row:
                case.update(id=row["id"],partition=row["partition"],metrics={a:row["metrics"][a] for a in ("N0","E0",arm)},
                            runtime_diagnostics=row.get("runtime_diagnostics",{}).get(arm,{}))
            result.append(case)
    return result


def _native_projection(xyz,Q,K,pose):
    if not pose or not pose.get("available",True):return None
    R=np.asarray(pose.get("R_physical",pose.get("R")),float)
    center=np.asarray(pose.get("centroid",pose.get("t")),float)
    physical=(np.asarray(xyz)-np.mean(xyz,axis=0))@np.asarray(Q)
    camera=physical@R.T+center
    if not np.isfinite(camera).all() or np.min(camera[:,2])<=0:return None
    uv=camera@np.asarray(K).T
    return uv[:,:2]/uv[:,2:]


def _edges(xyz):
    return [(i,j) for i in range(8) for j in range(i+1,8)
            if np.count_nonzero(np.abs(xyz[i]-xyz[j])>1e-7)==1]


def _wire(ax,xy,edges,color,label,*,style="-",alpha=1.):
    if xy is None:return
    for n,(a,b) in enumerate(edges):
        ax.plot(xy[[a,b],0],xy[[a,b],1],style,color=color,alpha=alpha,linewidth=1.25,
                label=label if n==0 else None)


def _valid_upstream_points(points,image):
    """Display only predictions with finite coordinates inside the raw canvas."""
    points=np.asarray(points,float)
    h,w=image.shape[:2]
    return (np.isfinite(points).all(axis=1)&(points[:,0]>=0)&(points[:,0]<w)&
            (points[:,1]>=0)&(points[:,1]<h))


def _viewport(image,frame):
    points=np.asarray(frame["trace"]["initial_points"],float)[:8]
    displayed=points[_valid_upstream_points(points,image)]
    h,w=image.shape[:2]
    if not len(displayed):return [0,0,w,h]
    low=np.floor(displayed.min(0)-24);high=np.ceil(displayed.max(0)+24)
    low=np.maximum(low,[0,0]);high=np.minimum(high,[w,h])
    if np.any(high-low<32):return [0,0,w,h]
    return [int(low[0]),int(low[1]),int(high[0]),int(high[1])]


def _base(ax,image,roi):
    ax.imshow(image)
    ax.set_xlim(roi[0],roi[2]);ax.set_ylim(roi[3],roi[1])
    ax.set_xlabel("raw x (px)",fontsize=7);ax.set_ylabel("raw y (px)",fontsize=7)
    ax.tick_params(labelsize=6)


def _observations(ax,packet,*,consumed_native_ids=()):
    spec=packet["solver_input"];complete=set(packet["activation"]["complete_ids"])
    for m in spec["visible"]:
        k=m["native_corner"]
        if k not in complete or spec["states"][k] not in ("B","I"):continue
        x,y=m["xy"]
        color="#AAAAAA" if k in consumed_native_ids else STATE_COLORS[spec["states"][k]]
        alpha=.45 if k in consumed_native_ids else 1.
        ax.plot(x,y,"o",mfc="none",mec=color,ms=6,mew=1.4,alpha=alpha)
        ax.annotate(str(k),(x,y),color=color,fontsize=6,alpha=alpha)
    for boundary in spec["boundary_observations"]:
        k=boundary["native_corner"]
        if spec["states"][k] not in ("B","I"):continue
        for line in boundary["lines"]:
            p=line.get("provenance",{})
            interval=p.get("actual_observed_inlier_tangent_interval_px")
            if interval is None:continue  # A broad allowed interval is not an observed segment.
            n=np.asarray(line["normal_unit_xy"]);tangent=np.array([-n[1],n[0]])
            xy=-float(line["offset_px"])*n+np.asarray(interval)[:,None]*tangent
            color="#AAAAAA" if k in consumed_native_ids else STATE_COLORS[spec["states"][k]]
            ax.plot(xy[:,0],xy[:,1],color=color,linewidth=2.5,alpha=.4 if k in consumed_native_ids else 1.)


def _selection_details(prediction,arm):
    entry=prediction["predictions"][arm]
    cid=entry.get("selected_candidate_id")
    family=entry.get("selected_family")
    mode=entry.get("mode",ARM_MODE.get(arm))
    selection=(prediction.get("selection_banks") or {}).get(arm)
    if selection is not None:
        chosen=next((c for c in selection["candidates"] if c.get("candidate_id")==cid),None)
        return dict(candidate_id=cid,family="N/A: global finite-candidate selector",mode="LEARNED_FINITE_CANDIDATE_SELECTOR",
            candidate_source="N0 abstention/fallback" if entry.get("none_selected") else (chosen or {}).get("source","not in stored selection pool"),
            candidate_metadata={k:chosen[k] for k in ("candidate_id","source","original_native_family") if chosen is not None and k in chosen},
            stored_score_audit={},family_intermediate_status="NOT_APPLICABLE_DIRECT_SOFTMAX_SELECTION",
            eligible_pool_count=len(selection.get("eligible_ids",selection["candidates"])),
            prediction_metadata={k:v for k,v in entry.items() if k!="actual_pose"})
    banks=[(prediction.get("carried",[]),prediction.get("carried_scores",[])),
           (prediction.get("banks",{}).get(mode,[]),prediction.get("bank_scores",{}).get(mode,[])),
           (prediction.get("learning_bank",[]),prediction.get("learning_scores",[]))]
    for bank,scores in banks:
        for i,c in enumerate(bank):
            if c.get("candidate_id")==cid:
                detail=(scores[i].get(family,{}) if family is not None and i<len(scores) else {})
                audits=detail.get("audits",{})
                audit=audits.get(mode,{}) if mode in audits else {}
                return dict(candidate_id=cid,family=family,mode=mode,candidate_source=c.get("source"),
                            candidate_metadata={k:c[k] for k in ("candidate_id","source","original_native_family") if k in c},
                            stored_score_audit=audit,prediction_metadata={k:v for k,v in entry.items() if k!="actual_pose"})
    return dict(candidate_id=cid,family=family,mode=mode,candidate_source="not retained in mapped bank",
                stored_score_audit={},prediction_metadata={k:v for k,v in entry.items() if k!="actual_pose"})


def _profile_application_text(bundle,applied):
    ids=bundle["consumed_native_ids"]
    if applied is True:return f"B profiles APPLIED\nactual consumed B IDs: {ids}"
    if applied is False:return f"B context only / NOT APPLIED\npotential bundle B IDs: {ids}\nNOT CONSUMED by this arm"
    return f"B application NOT LOGGED\npotential bundle B IDs: {ids}\nactual consumption unknown"


def _profile_status_text(q):
    status="SUPPORTED query" if q["supported"] else "UNSUPPORTED: diagnostic distribution; NOT USED"
    return status+"\n"+textwrap.fill(str(q["reason"]),width=48)


def _candidate_text(details):
    def shortened(value):
        value=str(value)
        return value if len(value)<=34 else value[:31]+"..."
    owner=details.get("stored_score_audit",{}).get("owner",{})
    weights=owner.get("global_owner_responsibilities")
    lines=["candidate: "+shortened(details["candidate_id"]),"family: "+str(details["family"]),"mode: "+str(details["mode"])]
    lines.append(("owner h mass: "+", ".join(f"{w:.3f}" for w in weights)) if weights else "owner hypothesis: not logged / not used")
    lines.append("Full ID and metadata: case JSON")
    return "\n".join(textwrap.fill(line,width=44) for line in lines)


def _profile_overlay(ax,bundle,*,applied):
    if bundle is None:
        ax.text(.02,.97,"No recorded B bundle",transform=ax.transAxes,va="top",fontsize=7,color="white",
                bbox=dict(facecolor="black",alpha=.65,pad=2));return
    for group in bundle["groups"]:
        for q in group["queries"]:
            xy=np.asarray(q["sample_xy"])
            color="#00FFFF" if q["supported"] else "#E6D58C"
            ax.plot(xy[:,0],xy[:,1],color=color,linewidth=.55,alpha=.48 if applied else .15)
    ax.text(.02,.97,_profile_application_text(bundle,applied),transform=ax.transAxes,va="top",fontsize=7,color="white",
            bbox=dict(facecolor="black",alpha=.65,pad=2))


def _metric_text(metric):
    if not metric.get("pose_available"):return "NO POSE; retained failure"
    return f"T={metric['translation_cm']:.2f} cm, R={metric['rotation_deg']:.2f} deg; {'SUCCESS' if metric['success'] else 'FAILURE'}"


def process_plate(case,frame,prediction,scored,reference,bundle,output):
    from PIL import Image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    image_path=Path(frame["image"])
    if _sha(image_path)!=frame["image_sha256"]:raise ValueError("Bound case RGB changed")
    image=np.asarray(Image.open(image_path).convert("RGB"));roi=_viewport(image,frame)
    packet=frame["packet"];spec=packet["solver_input"];trace=frame["trace"]
    family=frame["families"]["ORIGINAL_WD"]
    xyz=np.asarray(family["native_xyz"]);Q=np.asarray(family["physical_registration_Q"]);K=np.asarray(spec["K"])
    edges=_edges(xyz)
    initial=_native_projection(xyz,Q,K,trace["initial_pose"])
    refpose=dict(available=True,R_physical=reference["R"],centroid=reference["t"])
    refxy=_native_projection(xyz,Q,K,refpose)
    methods=("N0","E0",case["arm"])
    fig,axes=plt.subplots(4,3,figsize=(12.0,13.4),constrained_layout=True)
    details=_selection_details(prediction,case["arm"])
    for col,arm in enumerate(methods):
        pose=scored["poses"][arm];final=_native_projection(xyz,Q,K,pose)
        for row in range(4):_base(axes[row,col],image,roi)
        ax=axes[0,col]
        raw=np.asarray(trace["initial_points"],float)[:8]
        valid_raw=_valid_upstream_points(raw,image)
        for k,state in enumerate(trace["states"]):
            if not valid_raw[k]:continue
            ax.plot(*raw[k],marker="x" if state=="S" else "+",color=STATE_COLORS[state],ms=6,mew=1.3)
            ax.annotate(f"{k}:{state}",raw[k],color=STATE_COLORS[state],fontsize=6)
        _wire(ax,initial,edges,"white","initial virtual cuboid",style="--",alpha=.5)
        ax.set_title(arm+"\n1. Same upstream N3/SubPix predictions, state colors",fontsize=9)
        active=scored.get("runtime_diagnostics",{}).get(arm,{}).get("B_treatment_active") if arm==case["arm"] else False
        consumed=bundle["consumed_native_ids"] if bundle is not None and active is True else ()
        ax=axes[1,col];_observations(ax,packet,consumed_native_ids=consumed)
        if arm==case["arm"]:_profile_overlay(ax,bundle,applied=active)
        ax.set_title("2. Actual full circles / observed partial segments\nGray source factors replaced by B; no S measurement",fontsize=8)
        ax=axes[2,col];_wire(ax,final,edges,"#00E5FF","selected candidate projection")
        if arm==case["arm"]:
            text=_candidate_text(details)
        elif arm=="E0":
            text="Original E0 archived final candidate\nfamily: "+str(trace.get("selected_candidate_audit",{}).get("WD_family"))+"\nNo new owner mixture"
        else:text="Original common F output\nN3 + SubPix upstream\nNo new owner mixture or B distribution"
        ax.text(.02,.97,text,transform=ax.transAxes,va="top",fontsize=6.5,color="white",clip_on=True,bbox=dict(facecolor="black",alpha=.7,pad=3))
        ax.set_title("3. Selected candidate / correspondence metadata",fontsize=8)
        ax=axes[3,col]
        _wire(ax,refxy,edges,"#FF4DE3","existing geometric reference",style="--",alpha=.9)
        _wire(ax,final,edges,"#32FF71","final pose projection")
        ax.set_title("4. "+_metric_text(scored["metrics"][arm]),fontsize=8)
        ax.legend(fontsize=6,loc="lower left")
        if final is None:ax.text(.5,.5,"NO AVAILABLE FINITE PROJECTION",transform=ax.transAxes,ha="center",color="red",fontsize=8)
        elif np.any((final[:,0]<roi[0])|(final[:,0]>roi[2])|(final[:,1]<roi[1])|(final[:,1]>roi[3])):
            ax.text(.02,.97,"Some final vertices outside this input-defined crop",transform=ax.transAxes,va="top",fontsize=6,
                    color="white",bbox=dict(facecolor="black",alpha=.7,pad=2))
    fig.suptitle(f"{case['kind']} | {case['arm']} | {case['id']}\n"
                 "Upstream predictions, actual image evidence, virtual projections and geometric reference have distinct roles.\n"
                 "Physical material owner remains unverified; the geometric reference is not independent physical ground truth.",fontsize=10)
    fig.savefig(output,dpi=110,facecolor="white",pil_kwargs={"compress_level":9})
    plt.close(fig)
    return dict(image_sha256=frame["image_sha256"],viewport_original_xyxy=roi,geometry_family="ORIGINAL_WD display registration",
                selected_candidate=details,initial_and_final_projection_count=8,reference_kind="existing derived geometric; no physical-owner certification")


def profile_plate(case,bundle,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups=bundle["groups"] if bundle else []
    queries=[q for g in groups for q in g["queries"]]
    shown=queries[:2]  # Immutable query order, independent of peaks/outcomes.
    fig,axes=plt.subplots(max(len(shown),1),3,figsize=(11.5,2.7*max(len(shown),1)),squeeze=False,constrained_layout=True)
    if not shown:
        for ax in axes.ravel():ax.axis("off")
        axes[0,0].text(.05,.5,"No B query was supported/recorded.\nOriginal B factors remain; no precise boundary created.")
    for row,q in enumerate(shown):
        ax,pf_ax,p_ax=axes[row]
        rgb=np.asarray(q["sample_rgb"],np.uint8)[None,:,:]
        r=q["sample_offsets_base_px"]
        step=(r[1]-r[0]) if len(r)>1 else 1.
        ax.imshow(rgb,aspect="auto",extent=(r[0]-step/2,r[-1]+step/2,0,1));ax.set_yticks([])
        ax.set_title(f"Fixed RGB strip B{q['native_corner']} -> native {q['native_neighbor']}",fontsize=9)
        ax.set_xlabel("normal coordinate, canonical px")
        if "foreground_probability" in q:
            pf_ax.plot(r,q["foreground_probability"],color="#008B45");pf_ax.set_ylim(-.03,1.03)
            pf_ax.set_ylabel("p(foreground | RGB color)");pf_ax.set_xlabel("normal coordinate, canonical px")
            p_ax.plot(q["offset_grid_base_px"],q["offset_probability"],color="#7651A8" if q["supported"] else "#777777",
                      linestyle="-" if q["supported"] else "--")
            p_ax.set_xlabel("F -> B offset d, canonical px");p_ax.set_ylabel("discrete normalized mass p(d)")
        else:
            for a in (pf_ax,p_ax):a.text(.03,.5,"Unsupported shared appearance",transform=a.transAxes);a.set_axis_off()
        p_ax.set_title(_profile_status_text(q),fontsize=7)
        for a in (pf_ax,p_ax):a.grid(alpha=.2)
    fig.suptitle(f"{case['arm']} | {case['id']}\nFirst two fixed query profiles, including unsupported cases. All bins retained; no peak selection.",fontsize=10)
    fig.savefig(output,dpi=110,facecolor="white",pil_kwargs={"compress_level":9});plt.close(fig)


def _table_payload(scored_rows):
    values=[]
    for row in scored_rows:
        for arm,m in row["metrics"].items():
            status=row.get("status",{}).get(arm,{})
            values.append(dict(id=row["id"],partition=row["partition"],session=row["session"],arm=arm,
                success=bool(m["success"]),available=bool(m["pose_available"]),T=m["translation_cm"],R=m["rotation_deg"],
                fallback=status.get("fallback"),status=status.get("status"),
                B_active=row.get("runtime_diagnostics",{}).get(arm,{}).get("B_treatment_active")))
    return values


def _write_html(output,scored_rows,cases,*,execution_status=None,numeric_only=False):
    data=json.dumps(_table_payload(scored_rows),separators=(",",":"),ensure_ascii=False).replace("<","\\u003c")
    cards=[]
    for c in cases:
        label=html.escape(c["arm"]+" · "+c["kind"])
        if c["status"]=="NOT_EXECUTED":
            cards.append(f"<article><h3>{label}</h3><p>N/A: no actual sealed method output. {html.escape(str(c.get('execution_status',{}).get('reason','')))}</p></article>")
        elif c["status"]=="NO_CASE":
            cards.append(f"<article><h3>{label}</h3><p>No frame satisfies this category.</p></article>")
        else:
            file=html.escape(c["process_png"]);profiles=html.escape(c["profiles_png"])
            cards.append(f"<article><h3>{label}</h3><p>{html.escape(c['id'])}</p><a href='{file}'><img loading='lazy' src='{file}' alt='Parallel N0, E0, new-method process plate with input, actual evidence, candidate and final projections.'></a><p><a href='{profiles}'>Fixed RGB strips and full offset distributions</a> · <a href='{html.escape(c['case_json'])}'>Case provenance</a></p></article>")
    actual_arms=list(dict.fromkeys(a for r in scored_rows for a in r["metrics"]))
    inventory={a:dict(status="EXECUTED") for a in actual_arms}
    for a,v in (execution_status or {}).items():
        if a not in inventory:inventory[a]=dict(status=v) if isinstance(v,str) else v
    inventory_html="<h2>Complete method inventory</h2><div class='scroll'><table><thead><tr><th>Method</th><th>Execution</th><th>Reason when N/A</th></tr></thead><tbody>"+"".join(
        f"<tr><td>{html.escape(a)}</td><td>{html.escape(str(v.get('status','UNKNOWN')))}</td><td>{html.escape(str(v.get('reason') or ''))}</td></tr>" for a,v in inventory.items())+"</tbody></table></div>"
    template="""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Statewise recovery: sealed exploratory research gallery</title>
<style>body{font:15px/1.5 system-ui,sans-serif;color:#20262e;background:#f7f8fa;margin:0}main{max-width:1200px;margin:28px auto;padding:0 20px 40px}h1{line-height:1.2;font-size:28px}h2{margin-top:30px}a{color:#075caa}.notice{padding:14px;background:#fff3d3;border-left:4px solid #af7000}article{background:white;border:1px solid #dce1e6;margin:18px 0;padding:14px;border-radius:8px}article img{width:100%;height:auto}label{margin:5px 12px 5px 0;display:inline-block}#methods{padding:12px;background:white;border:1px solid #dde2e7}.scroll{max-height:520px;overflow:auto;background:white}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:6px 8px;text-align:right;border-bottom:1px solid #e0e4e8}th:first-child,td:first-child{text-align:left}thead th{position:sticky;top:0;background:#eaf0f5;cursor:pointer}canvas{background:white;display:block;width:100%;height:auto;border:1px solid #dde2e7}#tooltip{white-space:pre-wrap;min-height:62px;background:white;padding:10px;border:1px solid #dde2e7;font:13px/1.4 monospace}small{color:#485562}button,select,input{font:inherit}</style>
<main><h1>Statewise recovery: sealed exploratory comparison</h1>
<p class="notice">All 238 source images were already exposed during development. PRIMARY192 = TEST49 + REMAINDER143; CAL46 is separate. This is not an independent test. The reference is existing geometric reconstruction, not independent physical ground truth. Material/cuboid owner correspondence remains unverified.</p>
<p>Success is center translation ≤5 cm <strong>and</strong> physical rotation ≤5° on one final pose, with C2 I/Y (180°) only; no 90° width/depth equivalence. Every frame, missing pose and tail value remains in the data. The gallery only presents sealed evaluator outputs.</p>
__INVENTORY__
<h2>Full population errors</h2><label>Population <select id="population"><option value="PRIMARY">PRIMARY192</option><option value="ALL">All238</option><option value="CAL">CAL46</option><option value="TEST">TEST49</option><option value="REMAINDER">REMAINDER143</option></select></label>
<label>Translation spacing <select id="xscale"><option value="log1p" selected>log1p, complete range</option><option value="linear">Linear, complete range</option></select></label>
<label>Frame/session search <input id="search" placeholder="ID or session"></label><button id="all">All methods</button> <button id="none">Clear methods</button><div id="methods"></div>
<p id="coverage"></p><canvas id="scatter" width="1120" height="590"></canvas><div id="tooltip">Hover a point for exact per-image metrics. Circles: joint success; crosses: failure. Missing poses are counted above and retained in the table.</div>
<p><small>Both axes preserve the complete selected population range, beginning at zero. Translation spacing can use log1p(T/cm) or linear spacing; tick labels and tooltips always show the original cm values. Every selected point, including extreme errors, is retained. There is no outlier removal, jitter or axis cropping. Method/population/search controls select records explicitly. A marginal location on this plot does not certify owner identity.</small></p>
<div class="scroll"><table><thead><tr><th data-key="id">Frame</th><th data-key="arm">Method</th><th data-key="partition">Partition</th><th data-key="success">Success</th><th data-key="T">T (cm)</th><th data-key="R">R (deg)</th><th data-key="available">Pose available</th><th data-key="fallback">Fallback</th><th data-key="B_active">B applied</th></tr></thead><tbody id="rows"></tbody></table></div>
<h2>Mechanically selected process witnesses</h2><p>Representative arms were supplied by the campaign; they were not chosen here by PRIMARY score. Recovery uses the first immutable ID. Damage uses the largest threshold-normalized error. Both-failure worsening uses the largest positive max(ΔT/5 cm, ΔR/5°), with explicit ties. These witnesses describe mechanisms and do not estimate prevalence. Missing categories remain visible.</p>
<p><small>Process legend: crosses/pluses = upstream predictions; state colors B green / I blue / S gray / O orange / T red / U yellow. Actual full image observations are open circles; partial factors show only their recorded observed segment. Faint gray source B factors are replaced when B is applied. S has no image measurement. Cyan profiles are fixed RGB queries; virtual cuboid projections and the magenta geometric reference are distinct from actual material observations. The final projected pose is green.</small></p>
__CARDS__
<p><a href="GALLERY_CASES.json">All selection rules and categories</a> · <a href="GALLERY_RECEIPT.json">Seals, costs and source hashes</a> · <a href="SHA256SUMS.txt">Artifact checksums</a> · <a href="../COMPARISON_ALL_ARMS.csv">Full comparison CSV</a> · <a href="../ROW_METRICS.csv">All row metrics</a></p>
</main><script id="metrics" type="application/json">__DATA__</script><script>
const records=JSON.parse(document.getElementById('metrics').textContent),arms=[...new Set(records.map(r=>r.arm))];
const colors=Object.fromEntries(arms.map((a,i)=>[a,`hsl(${(i*137.508)%360} 68% 38%)`]));
const selected=new Set(arms),box=document.getElementById('methods'),canvas=document.getElementById('scatter'),ctx=canvas.getContext('2d');let plotted=[],sortKey='id',sortSign=1;
const esc=s=>String(s??'unknown').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
arms.forEach(a=>{const l=document.createElement('label'),i=document.createElement('input');i.type='checkbox';i.checked=true;i.onchange=()=>{i.checked?selected.add(a):selected.delete(a);render()};l.style.color=colors[a];l.append(i,document.createTextNode(a));box.append(l)});
function filtered(){const pop=document.getElementById('population').value,s=document.getElementById('search').value.toLowerCase();return records.filter(r=>selected.has(r.arm)&&(pop==='ALL'||pop==='PRIMARY'&&r.partition!=='CAL'||r.partition===pop)&&(!s||(r.id+' '+r.session).toLowerCase().includes(s)))}
function render(){const rows=filtered(),available=rows.filter(r=>r.available&&Number.isFinite(r.T)&&Number.isFinite(r.R)),ids=new Set(rows.map(r=>r.id)),logX=document.getElementById('xscale').value==='log1p',rawMaxT=Math.max(0,...available.map(r=>r.T)),rawMaxR=Math.max(0,...available.map(r=>r.R));document.getElementById('coverage').textContent=`${ids.size} images × ${selected.size} selected methods; ${rows.length} records, ${available.length} available plotted poses, ${rows.length-available.length} missing/nonfinite poses retained as failures. Joint successes: ${rows.filter(r=>r.success).length}. Complete-range maxima: T=${rawMaxT.toPrecision(7)} cm; R=${rawMaxR.toPrecision(7)} deg. Translation spacing: ${logX?'log1p':'linear'}.`;
ctx.clearRect(0,0,canvas.width,canvas.height);const m={l:80,r:28,t:28,b:68},w=canvas.width-m.l-m.r,h=canvas.height-m.t-m.b,maxT=Math.max(5,rawMaxT)*1.045,maxR=Math.max(5,rawMaxR)*1.045,tx=t=>logX?Math.log1p(t):t,x=t=>m.l+tx(t)/tx(maxT)*w,y=r=>m.t+h-r/maxR*h,format=v=>v===0?'0':v>=10000?v.toExponential(1):v.toFixed(1);
ctx.font='13px system-ui';ctx.strokeStyle='#dce1e6';ctx.fillStyle='#333';ctx.lineWidth=1;for(let i=0;i<=5;i++){let t=logX?Math.expm1(Math.log1p(maxT)*i/5):maxT*i/5,r=maxR*i/5;ctx.beginPath();ctx.moveTo(x(t),m.t);ctx.lineTo(x(t),m.t+h);ctx.stroke();ctx.beginPath();ctx.moveTo(m.l,y(r));ctx.lineTo(m.l+w,y(r));ctx.stroke();ctx.fillText(format(t),x(t)-18,m.t+h+23);ctx.fillText(r.toFixed(1),18,y(r)+4)}
ctx.strokeStyle='#222';ctx.setLineDash([5,4]);ctx.beginPath();ctx.moveTo(x(5),m.t);ctx.lineTo(x(5),m.t+h);ctx.moveTo(m.l,y(5));ctx.lineTo(m.l+w,y(5));ctx.stroke();ctx.setLineDash([]);ctx.fillText(`Per-image center translation error (cm), ${logX?'log1p':'linear'} spacing`,m.l+w/2-190,canvas.height-17);ctx.save();ctx.translate(17,m.t+h/2+120);ctx.rotate(-Math.PI/2);ctx.fillText('Per-image physical C2 rotation error (deg)',0,0);ctx.restore();
plotted=available.map(r=>({...r,x:x(r.T),y:y(r.R)}));for(const r of plotted){ctx.strokeStyle=colors[r.arm];ctx.fillStyle=colors[r.arm];ctx.globalAlpha=.55;ctx.beginPath();if(r.success){ctx.arc(r.x,r.y,2.8,0,Math.PI*2);ctx.fill()}else{ctx.moveTo(r.x-2.6,r.y-2.6);ctx.lineTo(r.x+2.6,r.y+2.6);ctx.moveTo(r.x-2.6,r.y+2.6);ctx.lineTo(r.x+2.6,r.y-2.6);ctx.stroke()}}ctx.globalAlpha=1;
rows.sort((a,b)=>{let x=a[sortKey],y=b[sortKey];if(x==null)return y==null?0:1;if(y==null)return-1;return sortSign*(typeof x==='number'?x-y:String(x).localeCompare(String(y)))});document.getElementById('rows').innerHTML=rows.map(r=>`<tr><td>${esc(r.id)}</td><td style="color:${colors[r.arm]}">${esc(r.arm)}</td><td>${esc(r.partition)}</td><td>${r.success?'yes':'no'}</td><td>${r.T==null?'missing':r.T.toFixed(4)}</td><td>${r.R==null?'missing':r.R.toFixed(4)}</td><td>${r.available?'yes':'no'}</td><td>${esc(r.fallback)}</td><td>${esc(r.B_active)}</td></tr>`).join('')}
canvas.addEventListener('mousemove',e=>{const b=canvas.getBoundingClientRect(),x=(e.clientX-b.left)*canvas.width/b.width,y=(e.clientY-b.top)*canvas.height/b.height;let best=null,d=64;for(const r of plotted){let q=(r.x-x)**2+(r.y-y)**2;if(q<d){best=r;d=q}}if(best)document.getElementById('tooltip').textContent=`${best.id} | ${best.arm} | ${best.partition}\nT=${best.T.toFixed(6)} cm, R=${best.R.toFixed(6)} deg | ${best.success?'joint success':'failure'}\nB applied=${best.B_active??'unknown'}; fallback=${best.fallback??'unknown'}`});
document.getElementById('population').onchange=render;document.getElementById('xscale').onchange=render;document.getElementById('search').oninput=render;document.getElementById('all').onclick=()=>{arms.forEach(a=>selected.add(a));box.querySelectorAll('input').forEach(i=>i.checked=true);render()};document.getElementById('none').onclick=()=>{selected.clear();box.querySelectorAll('input').forEach(i=>i.checked=false);render()};document.querySelectorAll('th[data-key]').forEach(th=>th.onclick=()=>{const k=th.dataset.key;sortSign=sortKey===k?-sortSign:1;sortKey=k;render()});render();
</script></html>"""
    document=template.replace("__DATA__",data).replace("__CARDS__","\n".join(cards)).replace("__INVENTORY__",inventory_html)
    if numeric_only:
        beginning=document.index("<h2>Mechanically selected process witnesses</h2>")
        end=document.index("</main><script id=",beginning)
        document=document[:beginning]+"<p>RGB process plates and case-specific boundary strips remain in the private research gallery. This public artifact contains numeric evaluator records only.</p><p><a href='../COMPARISON_ALL_ARMS.csv'>Complete comparison CSV</a> · <a href='../ROW_METRICS.csv'>All row metrics</a> · <a href='NUMERIC_GALLERY_RECEIPT.json'>Numeric export provenance</a></p>\n"+document[end:]
        if "<img " in document:raise ValueError("Public numeric export cannot contain RGB plates")
    output.write_text(document,encoding="utf-8")


def run(*,run_dir,evaluation_dir,inputs_path,references_path,batch_dir,output_dir,representative_arms=None,
        numeric_output_dir=None,representative_lock_path=None,presentation_amendment_path=None,postseal_GO=False):
    if postseal_GO is not True:
        raise PermissionError("Root explicit postseal gallery GO is required before any input read or case selection")
    started,cpu=time.perf_counter(),time.process_time()
    run_dir,evaluation_dir,inputs_path,references_path,batch_dir,output_dir=map(Path,
        (run_dir,evaluation_dir,inputs_path,references_path,batch_dir,output_dir))
    representative_arms=list(representative_arms or FIXED_REPRESENTATIVE_ARMS)
    if tuple(representative_arms)!=FIXED_REPRESENTATIVE_ARMS:raise ValueError("Use the three pre-outcome fixed representative arms, without PRIMARY model selection")
    representative_lock_path=Path(representative_lock_path or Path(__file__).resolve().parents[3]/"_docs/experiments/pallet_statewise_end_to_end_recovery_v1/GALLERY_REPRESENTATIVE_LOCK.json")
    representative_lock=_read(representative_lock_path)
    if representative_lock.get("representative_arms")!=representative_arms or representative_lock.get("selection_rules")!=SELECTION_RULES:
        raise ValueError("Representative gallery definition changed after its pre-outcome lock")
    presentation_binding={}
    if representative_lock.get("gallery_source_sha256")!=_sha(__file__):
        presentation_amendment_path=Path(presentation_amendment_path or representative_lock_path.with_name("GALLERY_PRESENTATION_AMENDMENT.json"))
        amendment=_read(presentation_amendment_path)
        if (amendment.get("schema") not in ("GALLERY_PRESENTATION_AMENDMENT_V1","GALLERY_PRESENTATION_AMENDMENT_V2","GALLERY_PRESENTATION_AMENDMENT_V3") or
            amendment.get("original_representative_lock_sha256")!=_sha(representative_lock_path) or
            amendment.get("original_gallery_source_sha256")!=representative_lock.get("gallery_source_sha256") or
            amendment.get("amended_gallery_source_sha256")!=_sha(__file__) or
            amendment.get("selection_semantics_unchanged") is not True or
            amendment.get("presentation_only") is not True or amendment.get("display_settings")!=DISPLAY_SETTINGS):
            raise ValueError("Gallery source requires a separately bound presentation-only amendment")
        presentation_binding[str(presentation_amendment_path)]=_sha(presentation_amendment_path)
        while amendment["schema"] in ("GALLERY_PRESENTATION_AMENDMENT_V2","GALLERY_PRESENTATION_AMENDMENT_V3"):
            previous_path=presentation_amendment_path.with_name(amendment["previous_amendment_filename"])
            previous=_read(previous_path)
            expected_previous_schema=("GALLERY_PRESENTATION_AMENDMENT_V2" if amendment["schema"]=="GALLERY_PRESENTATION_AMENDMENT_V3" else "GALLERY_PRESENTATION_AMENDMENT_V1")
            if (_sha(previous_path)!=amendment.get("previous_amendment_sha256") or
                previous.get("schema")!=expected_previous_schema or
                previous.get("amended_gallery_source_sha256")!=amendment.get("previous_gallery_source_sha256") or
                previous.get("original_representative_lock_sha256")!=_sha(representative_lock_path) or
                previous.get("original_gallery_source_sha256")!=representative_lock.get("gallery_source_sha256")):
                raise ValueError("Presentation amendment must preserve its preceding amendment binding")
            presentation_binding[str(previous_path)]=_sha(previous_path)
            presentation_amendment_path=previous_path
            amendment=previous
    predseal=_read(run_dir/"PREDICTION_SEAL.json");evalseal=_read(evaluation_dir/"EVALUATION_SEAL.json")
    if predseal.get("before_scoring") is not True or predseal.get("complete") is not True or predseal.get("reference_inputs") is not False:
        raise ValueError("Complete sealed reference-free predictions are required")
    if evalseal.get("explicit_root_scoring_GO") is not True:
        raise ValueError("Complete postseal evaluation GO is required")
    prediction_path=run_dir/predseal.get("prediction_filename","PREDICTIONS.jsonl.gz")
    if _sha(prediction_path)!=predseal["predictions_sha256"]:raise ValueError("Prediction bytes changed")
    if evalseal["input_bindings"]["new_prediction_seal"]!=_sha(run_dir/"PREDICTION_SEAL.json"):
        raise ValueError("Evaluation uses another prediction seal")
    scored_path=evaluation_dir/"SCORED_ROWS.jsonl.gz"
    if _sha(scored_path)!=evalseal["artifacts"][scored_path.name]:raise ValueError("Scored rows changed")
    if _sha(references_path)!=evalseal["input_bindings"]["fixed_references"]:raise ValueError("Original geometric reference binding changed")
    batchseal=_read(batch_dir/"BATCH_SEAL.json")
    if batchseal.get("complete") is not True or batchseal["input_sha256"]!=_sha(inputs_path):
        raise ValueError("Process inputs are not the sealed prediction-only batch inputs")
    if batchseal.get("ordered_ids")!=predseal.get("ordered_ids"):
        raise ValueError("Frame-folder order differs from final sealed population")
    scored=list(_rows(scored_path))
    if len(scored)!=238 or Counter(r["partition"] for r in scored)!=Counter(CAL=46,TEST=49,REMAINDER=143):
        raise ValueError("Exact all238 denominator required")
    execution_status=_read(run_dir/"ARM_EXECUTION_STATUS.json")
    cases=select_cases(scored,representative_arms,execution_status)
    wanted={c["id"] for c in cases if c["status"]=="SELECTED"}
    frames={f["id"]:f for f in _rows(inputs_path) if f["id"] in wanted}
    predictions={};frame_indices={}
    for i,p in enumerate(_rows(prediction_path)):
        if p["id"] in wanted:predictions[p["id"]]=p;frame_indices[p["id"]]=i
    if set(frames)!=wanted or set(predictions)!=wanted:raise ValueError("Selected frame not in prediction/input population")
    references=_read(references_path)["references"]
    by_id={r["id"]:r for r in scored}
    output_dir.mkdir(parents=True,exist_ok=False)
    for i,case in enumerate(cases):
        if case["status"]!="SELECTED":continue
        ident=case["id"];frame=frames[ident];prediction=predictions[ident]
        folder=batch_dir/"frames"/f"{frame_indices[ident]:04d}"
        if _read(folder/"FRAME_SEAL.json")["id"]!=ident:raise ValueError("Frame-folder ID differs")
        bundle_path=folder/"BOUNDARY_BUNDLE.jsonl.gz";bundle=None
        if prediction.get("boundary",{}).get("bundle_sha256") is not None and not bundle_path.exists():
            raise ValueError("Recorded sealed boundary bundle is absent")
        if bundle_path.exists():
            expected=prediction.get("boundary",{}).get("bundle_sha256")
            if expected is not None and _sha(bundle_path)!=expected:raise ValueError("Sealed B bundle changed")
            bundle=next(_rows(bundle_path))
        stem=f"case_{i+1:02d}"
        case.update(process_png=stem+"_process.png",profiles_png=stem+"_profiles.png",case_json=stem+".json")
        detail=process_plate(case,frame,prediction,by_id[ident],references[case["partition"]][ident],bundle,output_dir/case["process_png"])
        profile_plate(case,bundle,output_dir/case["profiles_png"])
        _write(output_dir/case["case_json"],{**case,**detail,
            "source_frame_prediction_only":True,"geometric_reference_access":"postseal rendering only",
            "boundary_bundle_sha256":_sha(bundle_path) if bundle_path.exists() else None})
    _write(output_dir/"GALLERY_CASES.json",dict(schema="POSTSEAL_MECHANICAL_WITNESSES_V1",rules=SELECTION_RULES,
        explicit_representative_arms=representative_arms,case_selection_population="PRIMARY192",cases=cases,
        no_primary_best_arm_or_seed_selection=True))
    _write_html(output_dir/"index.html",scored,cases,execution_status=execution_status)
    numeric_export=None
    if numeric_output_dir is not None:
        numeric_output_dir=Path(numeric_output_dir);numeric_output_dir.mkdir(parents=True,exist_ok=False)
        numeric_path=numeric_output_dir/"index.html"
        _write_html(numeric_path,scored,[],execution_status=execution_status,numeric_only=True)
        numeric_receipt=dict(schema="PUBLIC_NUMERIC_GALLERY_V1",source_sha256=_sha(__file__),
            input_bindings={"PREDICTION_SEAL.json":_sha(run_dir/"PREDICTION_SEAL.json"),
                            "EVALUATION_SEAL.json":_sha(evaluation_dir/"EVALUATION_SEAL.json"),
                            "SCORED_ROWS.jsonl.gz":_sha(scored_path),
                            str(representative_lock_path):_sha(representative_lock_path),**presentation_binding},
            population_frames=238,PRIMARY_frames=192,metric_records=sum(len(r["metrics"]) for r in scored),
            actual_arms=list(scored[0]["metrics"]),unexecuted_arms={a:v for a,v in execution_status.items() if a not in scored[0]["metrics"]},
            RGB_files_copied=0,process_case_images_published=0,tail_values_removed=0,
            display_settings=DISPLAY_SETTINGS,
            numeric_HTML=dict(file=numeric_path.name,bytes=numeric_path.stat().st_size,sha256=_sha(numeric_path)))
        numeric_receipt_path=numeric_output_dir/"NUMERIC_GALLERY_RECEIPT.json";_write(numeric_receipt_path,numeric_receipt)
        numeric_export=dict(directory=str(numeric_output_dir),files={p.name:_sha(p) for p in (numeric_path,numeric_receipt_path)})
    outputs=[p for p in output_dir.iterdir() if p.is_file()]
    receipt=dict(schema="POSTSEAL_PROCESS_GALLERY_V1",input_bindings={str(p):_sha(p) for p in
        (run_dir/"PREDICTION_SEAL.json",evaluation_dir/"EVALUATION_SEAL.json",scored_path,prediction_path,inputs_path,references_path,representative_lock_path)}|presentation_binding,
        source_sha256=_sha(__file__),selection_rules=SELECTION_RULES,all_population_frames=238,
        PRIMARY_frames=192,all_metric_records=sum(len(r["metrics"]) for r in scored),
        point_plot_tail_filtering=False,new_NN_PnP_optimizer_or_accuracy_calls=0,
        display_settings=DISPLAY_SETTINGS,
        explicit_postseal_GO=True,physical_owner_certification="unknown",reference="existing geometric reconstruction, not independent physical GT",
        representative_arms=representative_arms,RGB_process_plates_private=True,public_numeric_export=numeric_export,
        wall_seconds=time.perf_counter()-started,process_cpu_seconds=time.process_time()-cpu,
        process_peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        outputs=[dict(file=p.name,bytes=p.stat().st_size,sha256=_sha(p)) for p in outputs])
    _write(output_dir/"GALLERY_RECEIPT.json",receipt)
    outputs.append(output_dir/"GALLERY_RECEIPT.json")
    (output_dir/"SHA256SUMS.txt").write_text("".join(f"{_sha(p)}  {p.name}\n" for p in sorted(outputs)))
    return receipt


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ("run-dir","evaluation-dir","inputs-path","references-path","batch-dir","output-dir"):
        p.add_argument("--"+flag,required=True)
    p.add_argument("--representative-arm",action="append")
    p.add_argument("--representative-lock-path");p.add_argument("--numeric-output-dir")
    p.add_argument("--presentation-amendment-path")
    p.add_argument("--postseal-GO",action="store_true")
    args=p.parse_args()
    print(json.dumps(run(run_dir=args.run_dir,evaluation_dir=args.evaluation_dir,inputs_path=args.inputs_path,
        references_path=args.references_path,batch_dir=args.batch_dir,output_dir=args.output_dir,
        representative_arms=args.representative_arm,numeric_output_dir=args.numeric_output_dir,
        representative_lock_path=args.representative_lock_path,
        presentation_amendment_path=args.presentation_amendment_path,postseal_GO=args.postseal_GO),indent=2))
