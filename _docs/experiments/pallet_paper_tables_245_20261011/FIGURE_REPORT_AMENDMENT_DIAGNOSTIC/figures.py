"""Fixed numeric/coordinate figures and private RGB overlays; no fit or model call."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DOC = Path(os.environ.get("PALLET_PAPER_OUTPUT", ROOT / "_docs/experiments/pallet_paper_tables_245_20261011"))
BACKBONES = ("YOLO", "DOPE", "ResNet-18")
METHODS = ("BASE", "N3_THEN_SUBPIX")
FINAL4 = {"eval_pallet07", "eval_pallet09", "eval_night08", "eval_night09"}
COLORS = {"BASE": "#657882", "N3_THEN_SUBPIX": "#c45034"}
LABELS = {"BASE": "Base", "N3_THEN_SUBPIX": "Two-stage"}
EDGES = ((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def write(name, value):
    with (DOC/name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def save(fig, path):
    assert not path.exists(), "Preserve existing figure"
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return {"path": str(path.relative_to(DOC)), "sha256": sha(path), "bytes": path.stat().st_size}


def figure_a(metrics, figdir):
    fig, axes = plt.subplots(2, 2, figsize=(15, 8.8), sharey="row")
    receipt = {"groups": [], "zero_values": [], "all_finite_outliers_retained": True}
    for column, (scope, title) in enumerate((("RECT245", "Rectangular 245: selected clean/moderate"), ("SQUARE118", "Square 118: manual-PnP reference"))):
        packet = metrics["scopes"][scope] if "scopes" in metrics else metrics[scope]
        groups = packet.get("groups", packet)
        for row_index, (key, label) in enumerate((("T_cm", "Position error (cm)"), ("R_deg", "Rotation error (deg)"))):
            ax = axes[row_index, column]
            for backbone_index, backbone in enumerate(BACKBONES):
                for method_index, method in enumerate(METHODS):
                    group = groups[backbone+"::"+method]
                    frames = group["frame_values"]
                    valid = [(x["id"], float(x[key])) for x in frames if x.get(key) is not None and np.isfinite(x[key])]
                    assert all(value >= 0 for _, value in valid)
                    positive = np.array([value for _, value in valid if value > 0])
                    zeros = [fid for fid, value in valid if value == 0]
                    pos = backbone_index*3 + method_index
                    if len(positive):
                        bp = ax.boxplot([positive], positions=[pos], widths=.70, patch_artist=True, showfliers=False,
                                        medianprops={"color":"black","linewidth":1.4}, whiskerprops={"linewidth":1.0})
                        bp["boxes"][0].set(facecolor=COLORS[method], alpha=.70)
                        q1, q3 = np.quantile(positive, [.25,.75]);iqr = q3-q1
                        outside = positive[(positive < q1-1.5*iqr) | (positive > q3+1.5*iqr)]
                        if len(outside):
                            jitter = np.linspace(-.22,.22,len(outside)) if len(outside)>1 else [0.]
                            ax.scatter(pos+np.asarray(jitter),outside,s=14,alpha=.70,c=COLORS[method],zorder=3,edgecolors="none")
                    ax.text(pos, 1.02, f"n={len(valid)}", transform=ax.get_xaxis_transform(), ha="center", fontsize=7)
                    receipt["groups"].append({"scope":scope,"backbone":backbone,"method":method,"metric":key,"available":len(valid),"positive_on_log_axis":len(positive),"zero_count":len(zeros),"max":max((v for _,v in valid),default=None)})
                    if zeros:
                        receipt["zero_values"].append({"scope":scope,"backbone":backbone,"method":method,"metric":key,"ids":zeros})
                        ax.text(pos,.02,f"zero:{len(zeros)}",transform=ax.get_xaxis_transform(),ha="center",fontsize=6)
            ax.set_yscale("log")
            ax.set_xticks([.5,3.5,6.5],BACKBONES)
            ax.set_xlim(-.75,7.75)
            ax.grid(axis="y",which="both",alpha=.17)
            ax.set_title(title if row_index==0 else "")
            if column==0:ax.set_ylabel(label+"; logarithmic axis")
    fig.legend(handles=[Patch(facecolor=COLORS[m],alpha=.70,label=LABELS[m]) for m in METHODS],loc="upper center",bbox_to_anchor=(.5,1.025),ncol=2,frameon=False)
    fig.text(.5,.002,"Per-frame mean of all seeds; outliers are retained. Missing poses excluded only from continuous errors. Different reference procedures: no pooled estimate.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.03,1,.98))
    receipt.update(save(fig,figdir/"FIGURE_A_errors.png"))
    return receipt


def confusion(pose):
    return bool(pose.get("available") and pose["rotation_deg"] > 45 and abs(pose["yaw_deg"]) >= 60)


def figure_b(rect, figdir):
    ids = {r["id"] for r in rect if r["grade"] in ("clean","moderate")}
    old = rows(ROOT/"_docs/experiments/pallet_wd_hypothesis_diag_20261010/STAGE1_ROWS_REAL.jsonl.gz")
    selected = [r for r in old if r["id"] in ids and r["backbone"]=="YOLO" and r["method"]=="N3_THEN_SUBPIX"]
    byid = {}
    for r in selected:byid.setdefault(r["id"],[]).append(r)
    assert len(byid)==245 and all(sorted(x["seed"] for x in v)==[1,2,3] for v in byid.values())
    bins = [(0,5,"<5"),(5,10,"5–10"),(10,20,"10–20"),(20,30,"20–30"),(30,float("inf"),"≥30")]
    packet=[]
    for low,high,label in bins:
        values=[v for v in byid.values() if low <= v[0]["elevation_deg"] < high]
        rates={kind:float(np.mean([np.mean([confusion(r[kind]) for r in v]) for v in values]))*100 if values else None for kind in ("S0","ORACLE")}
        packet.append({"bin":label,"frames":len(values),"ids":sorted(v[0]["id"] for v in values),"S0_percent":rates["S0"],"GT_parity_oracle_percent":rates["ORACLE"]})
    fig,ax=plt.subplots(figsize=(10.7,5.2));x=np.arange(5);width=.34
    for off,key,label,color in ((-.5,"S0_percent","S0: prediction-selected",COLORS["N3_THEN_SUBPIX"]),(.5,"GT_parity_oracle_percent","GT-parity oracle: diagnostic", "#417d9b")):
        height=[p[key] if p[key] is not None else 0 for p in packet]
        bars=ax.bar(x+off*width,height,width,label=label,color=color)
        for bar,p in zip(bars,packet):
            val=p[key]
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.6,"NA" if val is None else f"{val:.1f}%",ha="center",fontsize=8)
    ax.set_xticks(x,[p["bin"]+f"°\nn={p['frames']}" for p in packet]);ax.set_ylabel("W/D confusion (% of frames; binary before seed mean)");ax.set_xlabel("Reference elevation bin")
    ax.set_title("YOLO Two-stage, rectangular 245: S0 vs GT-parity oracle")
    ax.set_ylim(0,max([p[k] or 0 for p in packet for k in ("S0_percent","GT_parity_oracle_percent")]+[5])*1.22+1)
    ax.grid(axis="y",alpha=.2);ax.legend(fontsize=9)
    fig.text(.5,.005,"Confusion = R >45 deg and |yaw| ≥60 deg. Oracle receives GT parity; it is not deployable and does not minimize pose error.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.035,1,1))
    return {**save(fig,figdir/"FIGURE_B_wd_elevation.png"),"bins":packet,"oracle_definition":"GT parity supplied, not min-error oracle","source_sha256":sha(ROOT/"_docs/experiments/pallet_wd_hypothesis_diag_20261010/STAGE1_ROWS_REAL.jsonl.gz")}


def project(row):
    p = row["actual_pose"]
    if not p.get("available"):return None
    w,h,d = np.asarray(p["cf_extents"],float)/2
    model=np.array([[-w,-h,-d],[w,-h,-d],[w,h,-d],[-w,h,-d],[-w,-h,d],[w,-h,d],[w,h,d],[-w,h,d]])
    camera=model@np.asarray(p["R_cf"],float).T+np.asarray(p["centroid"],float)
    k=np.asarray(row["fixed_metadata"]["K"],float)
    homogeneous=camera@k.T
    return homogeneous[:,:2]/homogeneous[:,2:]


def overlay(ax, base, seq, rgb=None, title=""):
    h,w=base["raw_hw"]
    if rgb is not None:ax.imshow(rgb)
    else:ax.set_facecolor("#f5f6f8")
    extent_points=[np.array([[0.,0.],[w,h]])]
    for method,row in (("BASE",base),("N3_THEN_SUBPIX",seq)):
        q=np.asarray(row["qFinal"],float)[:8]
        good=np.isfinite(q).all(1)&np.asarray(row.get("prediction_support",[True]*9),bool)[:8]
        extent_points.append(q[good])
        ax.scatter(q[good,0],q[good,1],s=18,marker="o" if method=="BASE" else "x",c=COLORS[method],label=LABELS[method]+" corners",linewidths=1,zorder=4)
        for index in np.flatnonzero(good):ax.annotate(str(index),q[index],color=COLORS[method],fontsize=5,xytext=(2,2),textcoords="offset points")
        pts=project(row)
        if pts is not None:
            extent_points.append(pts[np.isfinite(pts).all(1)])
            for a,b in EDGES:ax.plot(pts[[a,b],0],pts[[a,b],1],color=COLORS[method],lw=.8,ls="--" if method=="BASE" else "-",alpha=.85)
    if rgb is None:
        limits=np.concatenate(extent_points);low=limits.min(0);high=limits.max(0);pad=np.maximum((high-low)*.03,1.)
        ax.plot([0,w,w,0,0],[0,0,h,h,0],color="#8e9398",lw=.7,ls=":")
        ax.set_xlim(low[0]-pad[0],high[0]+pad[0]);ax.set_ylim(high[1]+pad[1],low[1]-pad[1])
    else:ax.set_xlim(0,w);ax.set_ylim(h,0)
    ax.set_aspect("equal")
    ax.set_title(title,fontsize=8);ax.set_xlabel("Native image x (px)",fontsize=6);ax.set_ylabel("Native image y (px)",fontsize=6);ax.tick_params(labelsize=6)


def choose_cases(data):
    cases=[];excluded=[]
    for population,backbones in (("RECT",BACKBONES),("SQUARE",("YOLO",))):
        for backbone in backbones:
            pairs={}
            for row in data[population]:
                if row["seed"]!=1 or row["backbone"]!=backbone or row["method"] not in METHODS:continue
                if population=="RECT" and (row["grade"] not in ("clean","moderate") or row["session"] in FINAL4):continue
                pairs.setdefault(row["id"],{})[row["method"]]=row
            complete=[]
            for fid,pair in pairs.items():
                if set(pair)!=set(METHODS) or not all(p["pose"].get("available") for p in pair.values()):
                    excluded.append({"population":population,"backbone":backbone,"id":fid,"reason":"paired pose unavailable"});continue
                delta=pair["N3_THEN_SUBPIX"]["pose"]["translation_cm"]-pair["BASE"]["pose"]["translation_cm"]
                complete.append((fid,delta,pair))
            improvements=[x for x in complete if x[1]<0]
            worsening=[x for x in complete if x[1]>0]
            assert improvements and worsening, "Fixed good/bad case selection requires both signs"
            median=float(np.median([x[1] for x in improvements]))
            good=min(improvements,key=lambda x:(abs(x[1]-median),x[0]))
            bad=min(worsening,key=lambda x:(-x[1],x[0]))
            for kind,item in (("median_improvement",good),("worst_worsening",bad)):
                fid,delta,pair=item
                cases.append({"population":population,"backbone":backbone,"kind":kind,"id":fid,"seed":1,"delta_T_cm":delta,"improvement_pool_median_cm":median,"eligible_paired_frames":len(complete),"improvements":len(improvements),"worsenings":len(worsening),"pair":pair})
    return cases,excluded


def choose_examples(data, material_map):
    examples=[]
    for material,population in (("plastic","RECT"),("wood","RECT"),("square","SQUARE")):
        candidates=[]
        for row in data[population]:
            if row["backbone"]!="YOLO" or row["method"]!="BASE" or row["seed"]!=1:continue
            if population=="RECT" and (row["grade"] not in ("clean","moderate") or row["session"] in FINAL4):continue
            rowmaterial=material_map.get(row["id"]) if population=="RECT" else "square"
            if rowmaterial!=material:continue
            diag=row["corner"].get("pallet_diagonal_px")
            if diag is not None and np.isfinite(diag):candidates.append((row["id"],float(diag),row))
        assert candidates, "No fixed example candidate for "+material
        med=float(np.median([x[1] for x in candidates]));fid,diag,row=min(candidates,key=lambda x:(abs(x[1]-med),x[0]))
        examples.append({"population":population,"material":material,"id":fid,"diagonal_px":diag,"pool_median_px":med,"eligible_frames":len(candidates),"row":row})
    return examples


def private_image(binding):
    path=Path(binding["image_path"])
    assert sha(path)==binding["sha256"], "Private image changed"
    bgr=cv2.imread(str(path),cv2.IMREAD_COLOR)
    assert bgr is not None, "Private RGB unavailable"
    return cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)


def prepare_private_inputs(private_dir):
    """Rebuild only private path bindings from already authenticated audits."""
    path=private_dir/"FIGURE_INPUTS.json"
    if path.exists():return
    private_dir.mkdir(parents=True,exist_ok=True)
    source=Path(os.environ.get("PALLET_SOURCE_ROOT",ROOT))
    materials={r["id"]:r["material"] for r in rows(ROOT/"_docs/experiments/pallet_wd_hypothesis_diag_20261010/STAGE1_ROWS_REAL.jsonl.gz") if r["backbone"]=="YOLO" and r["method"]=="BASE" and r["seed"]==1}
    rect=read(ROOT/"_docs/experiments/pallet_n3_subpix_final_20261010/INPUT_AUDIT.json")["inputs"]
    square=read(ROOT/"_docs/experiments/pallet_square6d_manualpnp_20261010/INPUT_AUDIT.json")["inputs"]
    frames=[]
    for population,audit in (("RECT",rect),("SQUARE",square)):
        for r in audit:
            if population=="SQUARE" and not r["eligible"]:continue
            bound=r["image"];p=source/bound["path"]
            assert sha(p)==bound["sha256"], "Private RGB audit changed"
            frames.append({"population":population,"id":r["id"],"image_path":str(p.resolve()),"sha256":bound["sha256"],"material":materials[r["id"]] if population=="RECT" else "square"})
    assert len(frames)==437 and len({r["id"] for r in frames})==437
    with path.open("x",encoding="utf-8") as stream:json.dump({"frames":frames,"scope":"Existing319 REAL and118 eligibleSQUARE RGB; private paths only"},stream,ensure_ascii=False,indent=2);stream.write("\n")


def figure_cd(data,figdir,private_dir):
    prepare_private_inputs(private_dir)
    inputs=read(private_dir/"FIGURE_INPUTS.json")
    rgb_dir=private_dir/"rgb_figures";rgb_dir.mkdir(exist_ok=True)
    binding_map=inputs.get("frames",inputs)
    if isinstance(binding_map,list):binding_map={r["id"]:r for r in binding_map}
    material_map={fid:v.get("material") for fid,v in binding_map.items()}
    cases,excluded=choose_cases(data);examples=choose_examples(data,material_map)
    case_receipts=[];private_outputs=[]
    fig,axes=plt.subplots(4,2,figsize=(13,15))
    for ax,case in zip(axes.flat,cases):
        pair=case["pair"];base=pair["BASE"];seq=pair["N3_THEN_SUBPIX"]
        title=f"{case['population']} {case['backbone']} | {case['kind']}\n{case['id']} | seed1 ΔT={case['delta_T_cm']:+.3f} cm"
        overlay(ax,base,seq,title=title)
        rgb=private_image(binding_map[case["id"]])
        pf,pa=plt.subplots(figsize=(8.5,6.7));overlay(pa,base,seq,rgb=rgb,title=title);pa.legend(fontsize=7)
        name=f"rgb_figures/FIGURE_C_{len(case_receipts)+1:02d}_private.png";pp=private_dir/name
        assert not pp.exists();pf.savefig(pp,dpi=160,bbox_inches="tight");plt.close(pf)
        private_outputs.append({"filename":name,"sha256":sha(pp),"bytes":pp.stat().st_size,"kind":"RGB overlay","id":case["id"],"input_RGB_sha256":binding_map[case["id"]]["sha256"]})
        case_receipts.append({k:v for k,v in case.items() if k!="pair"})
    axes[0,0].legend(fontsize=7);fig.tight_layout()
    cr=save(fig,figdir/"FIGURE_C_coordinates.png")
    fig,axes=plt.subplots(1,3,figsize=(15,4.8));example_receipts=[]
    index={(r["population"],r["backbone"],r["method"],r["seed"],r["id"]):r for records in data.values() for r in records}
    for ax,example in zip(axes,examples):
        row=example["row"];seq=index[(example["population"],"YOLO","N3_THEN_SUBPIX",1,example["id"])]
        title=f"{example['material']} | {example['id']}\npallet diagonal={example['diagonal_px']:.2f}px (pool median {example['pool_median_px']:.2f}px)"
        overlay(ax,row,seq,title=title)
        reference=row.get("evaluation_reference_points");valid=row.get("evaluation_reference_valid")
        if reference is not None:
            ref=np.asarray(reference,float)[:8];finite=np.isfinite(ref).all(1);mask=finite.copy()
            if valid is not None:mask &= np.asarray(valid,bool)[:8]
            ax.scatter(ref[mask,0],ref[mask,1],s=15,c="#3f925b",marker="+",label="Reference manual/observed")
            auxiliary=finite&~mask
            if auxiliary.any():ax.scatter(ref[auxiliary,0],ref[auxiliary,1],s=12,facecolors="none",edgecolors="#3f925b",marker="s",label="Existing auxiliary reference")
            low=ref[finite].min(0);high=ref[finite].max(0)
            ax.plot([low[0],high[0],high[0],low[0],low[0]],[low[1],low[1],high[1],high[1],low[1]],color="#3f925b",lw=.6,ls=":")
            ax.plot([low[0],high[0]],[low[1],high[1]],color="#3f925b",lw=.6,ls=":")
            xlim=ax.get_xlim();ylim=ax.get_ylim();ax.set_xlim(min(xlim[0],low[0]-5),max(xlim[1],high[0]+5));ax.set_ylim(max(ylim[0],high[1]+5),min(ylim[1],low[1]-5))
        rgb=private_image(binding_map[example["id"]]);pf,pa=plt.subplots(figsize=(8.5,6.7));overlay(pa,row,seq,rgb=rgb,title=title);pa.legend(fontsize=7)
        name=f"rgb_figures/FIGURE_D_{example['material']}_private.png";pp=private_dir/name
        assert not pp.exists();pf.savefig(pp,dpi=160,bbox_inches="tight");plt.close(pf)
        private_outputs.append({"filename":name,"sha256":sha(pp),"bytes":pp.stat().st_size,"kind":"RGB example","id":example["id"],"input_RGB_sha256":binding_map[example["id"]]["sha256"]})
        example_receipts.append({k:v for k,v in example.items() if k!="row"})
    axes[0].legend(fontsize=6);fig.tight_layout();dr=save(fig,figdir/"FIGURE_D_coordinates.png")
    write("PRIVATE_FIGURES_RECEIPT.json",{"status":"COMPLETE","private_directory_argument":"--private-dir","local_manifest":"FIGURE_INPUTS.json","local_manifest_sha256":sha(private_dir/"FIGURE_INPUTS.json"),"outputs":private_outputs,"raw_RGB_published":False,"local_paths_recorded_only_in_private_manifest":True})
    write("CASE_SELECTION.json",{"status":"COMPLETE","selection_seed":1,"tie_break":"lexicographic ID","rectangular_final_test4_excluded":True,"rectangular_case_pool":218,"square_case_pool":118,"diagonal_definition":"bbox hypot(ptp(x),ptp(y)) over ALL8 finite reference corners0..7, native pixels, no clipping; existing projected/extrapolated corners included only for this diagnostic","C":case_receipts,"C_pair_unavailable":excluded,"D":example_receipts})
    return cr,dr


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--private-dir",type=Path,required=True);args=parser.parse_args()
    assert (DOC/"METHOD_LOCK.json").exists(), "Methods must be locked before result-dependent figures"
    assert not (DOC/"FIGURE_INDEX.json").exists(), "Do not overwrite completed figures"
    data={pop:rows(DOC/(pop+"_ROWS.jsonl.gz")) for pop in ("RECT","SQUARE")}
    metrics=read(DOC/"METRICS.json")
    figdir=DOC/"figures";figdir.mkdir(exist_ok=True)
    a=figure_a(metrics,figdir);b=figure_b(data["RECT"],figdir);c,d=figure_cd(data,figdir,args.private_dir)
    write("FIGURE_INDEX.json",{"status":"COMPLETE","numeric_and_coordinate_only":True,"raw_RGB_published":False,"A":a,"B":b,"C":c,"D":d,"public_figures":[a["path"],b["path"],c["path"],d["path"]]})
    print("PAPER_FIGURES_COMPLETE",flush=True)


if __name__=="__main__":main()
