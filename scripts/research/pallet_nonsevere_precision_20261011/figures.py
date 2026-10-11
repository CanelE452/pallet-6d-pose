"""Native scientific PNG figures from sealed metrics; no RGB is published."""
from __future__ import annotations
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rows(paths):
    result={}
    for path in paths:
        with gzip.open(path,"rt",encoding="utf-8") as handle:
            for line in handle:
                r=json.loads(line);key=(r["method"],r["id"],int(r["seed"]))
                if key in result:assert result[key]==r
                result[key]=r
    return list(result.values())


def private_diagrams(public_rows,changed,selection,prediction_paths,reference_path):
    """Evaluation-aligned diagnostics only; never enters runtime inference."""
    if not prediction_paths or not reference_path:return None
    selected=selection.get("selected_arm") if selection else None
    target=f"{selected}_SUBPIX" if selected else None
    wanted=[r for r in changed["rows"] if r["before"]=="POSEFIX_SUBPIX" and (target is None or r["after"]==target)]
    displayed=[]
    for event in ("RECOVERY","DAMAGE"):
        candidates=sorted((r for r in wanted if r["event"]==event),key=lambda r:(r["id"],r["seed"]))
        displayed.extend(candidates[:6])
    keys={(r[m],r["id"],r["seed"]) for r in displayed for m in ("before","after")}
    predictions={}
    for path in prediction_paths:
        with gzip.open(path,"rt",encoding="utf-8") as handle:
            for line in handle:
                r=json.loads(line);key=r["method"],r["id"],int(r["seed"])
                if key in keys:
                    q=r.get("qFinal",r.get("q",r.get("points")))
                    assert q is not None,key
                    predictions[key]=np.asarray(q,float)
    public={(r["method"],r["id"],int(r["seed"])):r for r in public_rows}
    references=read(reference_path);output={}
    for r in displayed:
        reference=references[r["id"]];gt=np.asarray(reference["gt"],float);mask=np.asarray(reference["valid"],bool)[:8]
        aligned=[]
        for method in (r["before"],r["after"]):
            key=method,r["id"],r["seed"];q=predictions[key];p=public[key]
            permutation=np.asarray(reference["permutations"][int(p["corner_evaluation_branch"])],int)
            assert len(q)==len(permutation)==9
            canonical=np.empty_like(q);canonical[permutation]=q
            errors=np.linalg.norm(canonical[:8]-gt[:8],axis=1)
            assert np.allclose(errors[mask],np.asarray(p["corner_errors_px"],float)[mask],atol=1e-4,rtol=0),key
            aligned.append(canonical[:8])
        output[f"{r['id']}|{r['seed']}|{r['after']}"]=dict(reference_corners=gt[:8].tolist(),
                    before_corners=aligned[0].tolist(),after_corners=aligned[1].tolist(),corner_valid=mask.tolist())
    return output


def private_contact_sheets(destination,source,runtime_path,changed,selection,diagrams):
    """Deterministic review sheets remain private; no inference is performed."""
    if not destination or not diagrams:return []
    out=Path(destination);out.mkdir(parents=True,exist_ok=True)
    inputs=np.load(runtime_path,allow_pickle=False)
    images={str(fid):(str(inputs["image_paths"][k]),str(inputs["image_sha256"][k])) for k,fid in enumerate(inputs["ids"])}
    selected=selection["selected_arm"]+"_SUBPIX";receipts=[]
    for event in ("RECOVERY","DAMAGE"):
        cases=sorted((r for r in changed["rows"] if r["event"]==event and r["before"]=="POSEFIX_SUBPIX" and r["after"]==selected),key=lambda r:(r["id"],r["seed"]))[:6]
        if not cases:continue
        fig,axes=plt.subplots((len(cases)+1)//2,2,figsize=(14,5*((len(cases)+1)//2)),squeeze=False)
        for ax in axes.ravel():ax.axis("off")
        for ax,item in zip(axes.ravel(),cases):
            key=f"{item['id']}|{item['seed']}|{item['after']}";data=diagrams[key]
            path,expected=images[item["id"]];path=Path(path)
            if not path.is_absolute():path=Path(source)/path
            assert sha(path)==expected
            import cv2
            image=cv2.imread(str(path),cv2.IMREAD_COLOR);assert image is not None
            ax.imshow(image[:,:,::-1]);mask=np.asarray(data["corner_valid"],bool)
            for key_,label,color,marker in [("reference_corners","Evaluation reference","#37dd83","o"),("before_corners","PoseFix + SP","#4db6ff","x"),("after_corners",selected,"#ffab40","+")]:
                q=np.asarray(data[key_],float);ax.scatter(q[mask,0],q[mask,1],c=color,marker=marker,s=45,lw=1.5,label=label)
            fmt=lambda value:"NA" if value is None else f"{value:.2f}"
            ax.set_title(f"{event} {item['id']} seed{item['seed']}\nT cm {fmt(item['before_Tcm'])} → {fmt(item['after_Tcm'])}; R deg {fmt(item['before_Rdeg'])} → {fmt(item['after_Rdeg'])}",fontsize=9)
            ax.legend(fontsize=7,loc="lower right")
        fig.suptitle("PRIVATE RGB review: "+event+"; first six sorted image/seed cases; evaluation alignment only",fontsize=13)
        fig.tight_layout();path=out/(event+".png");assert not path.exists();fig.savefig(path,dpi=130,bbox_inches="tight");plt.close(fig)
        receipts.append(dict(owner="private_new",relative_path=out.name+"/"+path.name,sha256=sha(path),event=event,cases=len(cases),RGB_public=False))
    return receipts


COLORS={"BASE":"#b0b0b0","SUBPIX":"#697986","OLD_P_SUBPIX":"#a569bd","N3_THEN_SUBPIX":"#aa8820",
        "POSEFIX_SUBPIX":"#343a40","M0_SUBPIX":"#a17171","M1_SUBPIX":"#1b9e77",
        "M2_SUBPIX":"#7570b3","M3_SUBPIX":"#d95f02","M4_SUBPIX":"#0072b2"}


def setup():
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
                        "axes.spines.top":False,"axes.spines.right":False,
                        "figure.facecolor":"white","savefig.facecolor":"white"})


def pipeline(destination,completed_methods):
    fig,ax=plt.subplots(figsize=(13,5.8));ax.set(xlim=(0,13),ylim=(0,6));ax.axis("off")
    def box(x,y,w,h,label,color,fontsize=9):
        patch=FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.08,rounding_size=.08",fc=color,ec="#394150",lw=1.1)
        ax.add_patch(patch);ax.text(x+w/2,y+h/2,label,ha="center",va="center",fontsize=fontsize)
    def arrow(a,b,label=None):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=13,color="#526172",lw=1.3))
        if label:ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.13,label,ha="center",fontsize=8,color="#526172")
    box(.2,3.65,1.75,1.25,"Native RGB image", "#f2f4f7")
    box(2.5,3.65,1.9,1.25,"Frozen YOLO\nPredicted box/corners\nP3/P4 features", "#e1edf8",8.5)
    box(5,3.65,2.45,1.25,"OLD_P + residual slots\n222 candidate logits\nDifferentiable\nexpectation", "#e9f4ec",8.5)
    box(8.1,3.65,1.75,1.25,"Synthetic-only\ncalibration\nUncapped native\ncoordinates", "#e9f4ec",8.5)
    box(10.45,3.65,2.2,1.25,"Shared cornerSubPix\nBase-anchor cap once\nW/D selection + PnP\nC2-symmetric metrics", "#fceee0",8.5)
    box(10.7,5.12,1.75,.36,"K / registered dimensions", "#f2f4f7",7.5)
    arrow((11.55,5.12),(11.55,4.92))
    arrow((1.95,4.3),(2.5,4.3));arrow((4.4,4.3),(5,4.3));arrow((7.45,4.3),(8.1,4.3));arrow((9.85,4.3),(10.45,4.3))
    box(.25,.5,3.75,1.85,"COMMON TRAIN\nSame split, order, seed 1/2, final step\nM0: original candidate CE\nM1: CE + native expectation SmoothL1", "#faf0f0")
    box(4.65,.5,3.35,1.85,"GEOMETRY SUPERVISION\nM2 / M4 only, training-time\nDetached X(d), K, R/t, Jacobian\nFixed-weight covariance term", "#efecfa")
    box(8.6,.5,3.9,1.85,"RGB LOCAL EVIDENCE\nM3 / M4 only, training + inference\nNative tiles from predicted corners\nShared tiny CNN, sampled candidates", "#fff0df")
    arrow((2.15,2.35),(5.65,3.65),"CE / coordinate loss");arrow((6.3,2.35),(6.3,3.65),"M2 / M4 covariance loss")
    arrow((10.5,2.35),(7,3.65),"M3 / M4 RGB residual")
    ax.text(.25,5.78,"Non-Severe precision screen: five arms, common frozen base",fontsize=13,fontweight="bold")
    completed=[m for m in ("M0","M1","M2","M3","M4") if m in completed_methods]
    ax.text(.25,.12,"Dimensions enter every PnP. Declared arm design above; evaluated new arms: "+(", ".join(completed) or "none (blocked)"),fontsize=9)
    fig.savefig(destination,dpi=180,bbox_inches="tight");plt.close(fig)


def metric(row,key):
    if not row.get("available",row.get("pose",{}).get("available")):return np.inf
    p=row["pose"]
    for alias in {"Tcm":["Tcm","translation_cm","T_cm"],"Rdeg":["Rdeg","rotation_deg","R_deg"]}[key]:
        if p.get(alias) is not None:return float(p[alias])
    return np.inf


def ecdf(ax,values,label,color):
    values=np.asarray(values,float);finite=np.sort(values[np.isfinite(values)])
    if len(finite):
        ax.step(np.r_[0,finite],np.r_[0,np.arange(1,len(finite)+1)/len(values)],where="post",label=label,color=color,lw=1.8)
    else:ax.plot([],[],label=label+" (all unavailable)",color=color)
    return finite[-1] if len(finite) else 0


def cdfs(destination,rr):
    groups=defaultdict(list)
    for r in rr:
        if int(r["seed"]) in (1,2) and r["method"] in COLORS:groups[r["method"]].append(r)
    fig,axes=plt.subplots(1,3,figsize=(15.8,4.7));maxima=[0.,0.,0.]
    for method,color in COLORS.items():
        if method not in groups:continue
        records=groups[method];n=len(records)
        assert n==220,(method,n)
        errors=[float(e) if e is not None else np.inf for r in records for e,v in zip(r["corner_errors_px"],r["corner_valid"]) if v]
        maxima[0]=max(maxima[0],ecdf(axes[0],errors,method,color))
        for k,key in enumerate(("Tcm","Rdeg"),1):
            values=[metric(r,key) for r in records];missing=sum(not np.isfinite(v) for v in values)
            label=method+(f" [missing {missing}]" if missing else "")
            maxima[k]=max(maxima[k],ecdf(axes[k],values,label,color))
    for ax,maximum,title,xlabel in zip(axes,maxima,["PCK: every valid corner", "Translation: every image × seed", "Rotation: every image × seed"],
                                     ["2D error (px)","Translation error (cm)","Rotation error (deg)"]):
        ax.set_xscale("symlog",linthresh=1);ax.set(xlim=(0,max(10,maximum)*1.02),ylim=(0,1.02),xlabel=xlabel,ylabel="Cumulative fraction",title=title)
        ax.grid(alpha=.2);ax.legend(fontsize=7,loc="lower right")
    for t in (1,2,3,5,10):axes[0].axvline(t,color="#c2c8cf",lw=.7,ls=":")
    for ax in axes[1:]:ax.axvline(5,color="#555",ls="--",lw=.9)
    fig.suptitle("Full finite error range; log-linear x axes; unavailable poses remain in the denominator",fontsize=12)
    fig.tight_layout();fig.savefig(destination,dpi=180,bbox_inches="tight");plt.close(fig)


def cases(destination,changed,selection,private_cases):
    selected=selection.get("selected_arm") if selection else None
    target=f"{selected}_SUBPIX" if selected else None
    candidates=[r for r in changed["rows"] if r["before"]=="POSEFIX_SUBPIX" and (target is None or r["after"]==target)]
    # Prespecified deterministic display: first ID/seed from each direction;
    # if absent, show that absence, never substitute a flattering alternative.
    shown=[]
    for event in ("RECOVERY","DAMAGE"):
        records=sorted((r for r in candidates if r["event"]==event),key=lambda r:(r["id"],r["seed"]))
        shown.append(records[0] if records else None)
    fig,axes=plt.subplots(2,2,figsize=(12.5,8.6));details=[]
    data=private_cases if isinstance(private_cases,dict) else {}
    for row_index,(event,item) in enumerate(zip(("RECOVERY","DAMAGE"),shown)):
        ax,summary=axes[row_index];summary.axis("off")
        if item is None:
            ax.axis("off");ax.text(.5,.5,f"No {event.lower()} for fixed VAL-selected candidate",ha="center",va="center",transform=ax.transAxes)
            summary.text(.05,.5,"All changed success rows remain in the JSON supplement.",va="center")
            details.append(dict(event=event,status="NO_EVENT"));continue
        key=f"{item['id']}|{item['seed']}|{item['after']}"
        record=data.get(key)
        if record:
            ref=np.asarray(record["reference_corners"],float);before=np.asarray(record["before_corners"],float);after=np.asarray(record["after_corners"],float)
            mask=np.asarray(record["corner_valid"],bool)
            # Public image contains normalized geometry only, never RGB or a
            # serialized absolute reference-coordinate array.
            origin=ref[mask].mean(0);scale=max(np.ptp(ref[mask,0]),np.ptp(ref[mask,1]),1.)
            ref,before,after=(a-origin for a in (ref,before,after));ref,before,after=(a/scale for a in (ref,before,after))
            for points,label,color,marker in [(ref,"Reference","#2d9d65","o"),(before,"PoseFix + SubPix","#444","x"),(after,item["after"],"#d95f02","+")]:
                ax.scatter(points[mask,0],-points[mask,1],label=label,c=color,marker=marker,s=42)
            for k in np.where(mask)[0]:
                ax.plot([before[k,0],after[k,0]],[-before[k,1],-after[k,1]],color="#d95f02",alpha=.7,lw=1)
                ax.text(ref[k,0],-ref[k,1]+.025,str(k),fontsize=8)
            ax.axis("equal");ax.set_xlabel("Normalized reference-centered x");ax.set_ylabel("Normalized y");ax.legend(fontsize=8);status="ACTUAL_CORNER_DIAGRAM"
        else:
            values=[item["before_Tcm"],item["after_Tcm"],item["before_Rdeg"],item["after_Rdeg"]]
            ax.bar(range(4),[v if v is not None else np.nan for v in values],color=["#444","#d95f02","#444","#d95f02"])
            ax.set_xticks(range(4),["PoseFix T","New T","PoseFix R","New R"])
            for i,v in enumerate(values):
                if v is None:ax.text(i,0,"NA",ha="center",va="bottom")
            ax.axhline(5,ls="--",color="#999");ax.set_ylabel("T cm / R deg (separate labeled values)");status="METRIC_ONLY_PRIVATE_COORDINATES_UNAVAILABLE"
        ax.set_title(f"{event}: {item['id']}, seed {item['seed']}",fontsize=10)
        fmt=lambda value:"NA" if value is None else f"{value:.3f}"
        summary.text(.05,.85,f"{item['before']} → {item['after']}\nStrict success {item['before_success']} → {item['after_success']}\n\nT (cm): {fmt(item['before_Tcm'])} → {fmt(item['after_Tcm'])}\nR (deg): {fmt(item['before_Rdeg'])} → {fmt(item['after_Rdeg'])}\n\nCorners improved: {item['corners_improved']}\nCorners damaged: {item['corners_damaged']}\n\nDisplay rule: first sorted ID/seed in each direction.\nFull recovery AND damage lists are retained.",va="top",fontsize=11)
        details.append(dict(event=event,id=item["id"],seed=item["seed"],method=item["after"],status=status))
    fig.suptitle("VAL-selected candidate: actual same-image recovery and damage, no raw RGB",fontsize=14)
    fig.tight_layout();fig.savefig(destination,dpi=180,bbox_inches="tight");plt.close(fig)
    return details


def paired_plot(destination,paired):
    records=paired["primary"]+paired["mechanism"]
    fig,ax=plt.subplots(figsize=(10,6.4))
    for y,r in enumerate(records):
        if r["effect_pp"] is None:ax.text(0,y,"BLOCKED / NA",ha="center",color="#999");continue
        low,high=r["CI95"];effect=r["effect_pp"]
        color="#0072b2" if y<5 else "#d95f02"
        ax.hlines(y,low,high,color=color);ax.vlines([low,high],y-.07,y+.07,color=color)
        ax.plot(effect,y,"o",color=color)
        ax.text(high+.15,y,f"Holm {r['p_Holm']:.3f}",va="center",fontsize=8)
    ax.set_yticks(range(len(records)),[r["after"]+" − "+r["before"] for r in records]);ax.invert_yaxis()
    ax.axvline(0,color="#555",lw=1);ax.axhline(4.5,color="#bbb",lw=.8)
    ax.set_xlabel("Strict 5cm / 5deg success difference (percentage points)")
    ax.set_title("Fixed primary (5) and mechanism (4) families: separate Holm correction")
    ax.grid(axis="x",alpha=.2);fig.tight_layout();fig.savefig(destination,dpi=180,bbox_inches="tight");plt.close(fig)


def repair_pipeline(args):
    """Render the authorized diagram-only repair; preserve original evidence."""
    start=time.perf_counter();setup();doc=Path(args.public_dir);private=Path(args.private_dir)
    output=doc/"PIPELINE_VISUAL_REPAIR.json";lock=doc/"PIPELINE_VISUAL_REPAIR_LOCK.json"
    assert not output.exists() and not lock.exists()
    image=doc/"FIGURES/01_pipeline.png";oldsource=private/"figures_pre_visual_repair.py"
    original=read(doc/"FIGURE_RECEIPT.json")
    oldentry=next(r for r in original["figures"] if r["path"]=="FIGURES/01_pipeline.png")
    assert sha(image)==oldentry["sha256"] and sha(oldsource)==original["source_sha256"]
    archive=private/"01_pipeline_before_visual_repair.png";assert not archive.exists();shutil.copyfile(image,archive)
    unchanged={p.name:sha(p) for p in (doc/"FIGURES").glob("*.png") if p!=image}
    evidence=dict(status="LOCKED_BEFORE_RENDER",old_PNG_sha256=sha(archive),old_source_sha256=sha(oldsource),
                  new_source_sha256=sha(Path(__file__)),original_figure_receipt_sha256=sha(doc/"FIGURE_RECEIPT.json"),
                  reason="Native RGB alone feeds YOLO; predicted box follows YOLO; K/dimensions bypass detector to PnP; wrap labels inside boxes",
                  original_PNG_private_relative_path=archive.name,original_source_private_relative_path=oldsource.name,
                  additional_model_forwards=0,additional_PnP=0,additional_training=0)
    with lock.open("x") as f:json.dump(evidence,f,ensure_ascii=False,indent=2);f.write("\n")
    pipeline(image,{"M0","M1","M2","M3","M4"})
    assert unchanged=={p.name:sha(p) for p in (doc/"FIGURES").glob("*.png") if p!=image}
    assert sha(doc/"FIGURE_RECEIPT.json")==evidence["original_figure_receipt_sha256"]
    result={**evidence,"status":"PASS","new_PNG_sha256":sha(image),"other_PNGs_unchanged":unchanged,
            "wall_seconds":time.perf_counter()-start,"repair_lock_sha256":sha(lock),"original_receipt_preserved":True}
    with output.open("x") as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write("\n")
    print(json.dumps(result,ensure_ascii=False))


def run(args):
    setup();doc=Path(args.public_dir);out=doc/"FIGURES";out.mkdir(exist_ok=True)
    destinations=[out/name for name in ["01_pipeline.png","02_PCK_pose_CDF.png","03_recovery_damage.png","04_fixed_paired_effects.png"]]
    assert not any(p.exists() for p in destinations),"Existing figures preserved"
    rr=rows(args.rows);selection=read(args.selection) if args.selection else None
    changed=read(doc/"CHANGED_SUCCESS_CASES.json")
    private=read(args.private_cases) if args.private_cases else private_diagrams(rr,changed,selection,args.private_predictions,args.private_reference)
    pipeline(destinations[0],{r["method"] for r in rr});cdfs(destinations[1],rr)
    details=cases(destinations[2],changed,selection,private)
    paired_plot(destinations[3],read(doc/"PAIRED_RESULTS.json"))
    contacts=private_contact_sheets(args.private_contact_dir,args.source_root,args.runtime_inputs,changed,selection,private) if args.private_contact_dir else []
    receipt=dict(status="COMPLETE",source_sha256=sha(Path(__file__)),raw_RGB_published=False,
                 inputs=[dict(name=Path(p).name,sha256=sha(p)) for p in args.rows],displayed_cases=details,
                 selection_sha256=sha(args.selection) if args.selection else None,
                 case_coordinate_alignment="Each method uses its recorded evaluation whole-object permutation; canonical GT diagnostic, never inference branching",
                 private_RGB_contact_sheets=contacts,
                 figures=[dict(path="FIGURES/"+p.name,sha256=sha(p)) for p in destinations],
                 additional_model_forwards_or_PnP_or_training=0)
    with (doc/"FIGURE_RECEIPT.json").open("x",encoding="utf-8") as handle:json.dump(receipt,handle,ensure_ascii=False,indent=2);handle.write("\n")
    print(json.dumps(receipt,ensure_ascii=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--public-dir",required=True)
    parser.add_argument("--rows",nargs="+");parser.add_argument("--selection");parser.add_argument("--private-cases")
    parser.add_argument("--repair-pipeline",action="store_true");parser.add_argument("--private-dir")
    parser.add_argument("--private-predictions",nargs="*",default=[]);parser.add_argument("--private-reference")
    parser.add_argument("--private-contact-dir");parser.add_argument("--source-root");parser.add_argument("--runtime-inputs")
    args=parser.parse_args()
    if args.repair_pipeline:
        assert args.private_dir
        repair_pipeline(args)
    else:
        assert args.rows
        run(args)
