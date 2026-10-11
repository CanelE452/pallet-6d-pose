"""Independent aggregate arithmetic, rational bootstrap p-values and disclosure audit.

No producer statistics function is imported. This script performs no inference,
fitting, image extraction or PnP. Public reference originals are rejected.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter, defaultdict
from fractions import Fraction
import gzip
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np


CONTRASTS = [("G_V_MLP", "G_MLP"), ("G_V_D_MLP", "G_V_MLP"),
             ("D_JOINT_DIM_SUBPIX", "D_CONTROL_SUBPIX"), ("G_V_D_MLP", "OFFICIAL")]
KEYS = ["translation_cm", "rotation_deg", "yaw_deg", "ADDsym_m", "IoU3D",
        "available", "pose_success", "wd_confusion", "axis_wrong"]
FORBIDDEN_KEYS = {"evaluation_reference_points", "manual_points8", "manual_points", "gt_points",
                  "reference_points", "q_ref", "qref", "R_gt_representative", "t_gt",
                  "reference_R_cf", "reference_bottom_center", "reference_raw_rgb"}


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quantile(sequence, q):
    values = sorted(sequence)
    position = (len(values)-1)*q; a=math.floor(position); b=math.ceil(position)
    return values[a] if a==b else values[a]*(b-position)+values[b]*(position-a)


def compare(a, b, tolerance=1e-8):
    if a is None or b is None:
        assert a is None and b is None
        return 0.0
    delta=abs(float(a)-float(b)); assert delta<=tolerance, (a,b,delta)
    return delta


def audit_classifier_amendment(root, doc, private):
    """Verify archived execution sources; permit no unrecorded code mutation."""
    amendment=read(doc/"C_IO_PREFIT_AMENDMENT.json")
    latest=read(doc/"C_IO_PREFIT_LOCK.json")
    prior=read(doc/"C_CODE_FINAL_LOCK.json")
    reporting=read(doc/"C_REPORTING_PREEXEC_AMENDMENT.json")
    extraction=read(doc/"C_CODE_LOCK.json")
    current=root/latest["code"]["path"]
    archived=private/"classifier_pre_io_source.py"
    extraction_copy=private/"classifier_extraction_source.py"
    assert amendment["fitting_calls"]==amendment["optimizer_updates"]==0
    assert amendment["C_OpenCV_calls_already_completed"]==26652
    assert latest["fitting_calls"]==latest["optimizer_updates"]==0
    assert latest["PnP_calls_already_executed"]==26652
    assert sha(archived)==amendment["source_before_sha256"]==prior["code"]["sha256"]==latest["prior_code_sha256"]
    assert sha(current)==amendment["source_after_sha256"]==latest["code"]["sha256"]
    assert sha(extraction_copy)==extraction["code"]["sha256"]==reporting["loaded_extractor_source"]["sha256"]
    assert reporting["new_code"]["sha256"]==sha(archived)
    assert reporting["fitting_calls"]==reporting["PnP_calls"]==reporting["predictor_or_model_or_split_or_hyperparameter_changes"]==0
    unchanged=["arms","geometry_dimensions","geometry","visual_dimensions","dimension_dimensions",
               "common_MLP_input_dimensions","logistic","MLP","tie","one_candidate","no_candidate",
               "extraction","probe","parity","reporting"]
    for key in unchanged:assert latest[key]==prior[key],key
    expected_helper=ast.parse('''def loaded_npz(path):
    """Decompress each requested input array once before any row loop."""
    with np.load(path, allow_pickle=False) as packet:
        return {key: packet[key] for key in packet.files}
''').body[0]
    original=ast.parse(archived.read_text());replacement=ast.parse(current.read_text())
    helpers=[node for node in replacement.body if isinstance(node,ast.FunctionDef) and node.name=="loaded_npz"]
    assert len(helpers)==1 and ast.dump(helpers[0],include_attributes=False)==ast.dump(expected_helper,include_attributes=False)
    replacement.body=[node for node in replacement.body if node not in helpers]
    class Normalize(ast.NodeTransformer):
        def visit_Constant(self,node):
            if node.value=="C_IO_PREFIT_LOCK.json":node.value="C_CODE_FINAL_LOCK.json"
            return node
        def visit_Call(self,node):
            self.generic_visit(node)
            if isinstance(node.func,ast.Name) and node.func.id=="loaded_npz":
                node.func=ast.Attribute(value=ast.Name(id="np",ctx=ast.Load()),attr="load",ctx=ast.Load())
            if isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=="np" and node.func.attr=="load":
                if not any(key.arg=="allow_pickle" for key in node.keywords):
                    node.keywords.append(ast.keyword(arg="allow_pickle",value=ast.Constant(value=False)))
            return node
    assert ast.dump(Normalize().visit(original),include_attributes=False)==ast.dump(Normalize().visit(replacement),include_attributes=False)
    archived_hashes={sha(archived):"classifier_pre_io_source.py",sha(extraction_copy):"classifier_extraction_source.py"}
    return dict(status="PASS",current_source=latest["code"],preserved_archived_sources=archived_hashes,
                identical_complete_AST_after_explicit_IO_helper_and_guard_normalizations=True,
                locked_optimization_and_selection_fields_unchanged=unchanged,
                fits_before_IO_amendment=0,geometry_calls_before_IO_amendment=26652),archived_hashes


def audit_public(root, doc, source, published, private):
    paths=[p for p in doc.rglob("*") if p.is_file()]
    issues=[]; forbidden=[]; bindings=[]; byte_count=0
    roots={"new_worktree":root, "historical_source":source, "published_source":published,
           "private_new":private, "published_worktree":published}
    geometry_amendment=read(doc/"GEOMETRY_EXECUTION_AMENDMENT.json")
    assert geometry_amendment["actual_A3_PnP_calls_before_amendment"]==0
    assert sha(doc/"GEOMETRY_CODE_LOCK.json")==geometry_amendment["original_code_lock_sha256"]
    classifier_amendment,archived_classifier_hashes=audit_classifier_amendment(root,doc,private)
    def walk(value, context):
        if isinstance(value, dict):
            for key in value:
                metadata_only = (isinstance(value[key],dict) and set(value[key])=={"file","dtype","shape"}
                                 and isinstance(value[key]["file"],str) and isinstance(value[key]["shape"],list))
                if key in FORBIDDEN_KEYS and not metadata_only:
                    forbidden.append(dict(file=context,key=key))
            if "path" in value and "sha256" in value and isinstance(value["path"],str):
                bindings.append((context,value))
            for child in value.values():walk(child,context)
        elif isinstance(value,list):
            for child in value:walk(child,context)
    for path in paths:
        byte_count+=path.stat().st_size
        if path.suffix.lower() in {".jpg",".jpeg",".bmp",".tif",".tiff",".webp",".pt",".pth",".ckpt",".safetensors",".npz"}:
            issues.append(dict(file=str(path.relative_to(doc)),reason="Unexpected public RGB/model/feature container"))
        if path.stat().st_size>20*1024*1024:
            issues.append(dict(file=str(path.relative_to(doc)),reason="Public artifact exceeds20MiB"))
        if path.suffix in (".png",".npy"):continue
        try:
            if path.suffix==".gz":
                with gzip.open(path,"rt",encoding="utf-8") as f:text=f.read()
            else:text=path.read_text()
        except UnicodeDecodeError:
            issues.append(dict(file=path.name,reason="Unexpected binary public artifact"));continue
        if re.search(r"/(?:home/minjae|dev/shm|tmp)(?:/|\b)",text):
            issues.append(dict(file=path.name,reason="Private absolute path"))
        if path.name.endswith(".jsonl.gz"):
            for line in text.splitlines():walk(json.loads(line),path.name)
        elif path.suffix==".json":walk(json.loads(text),path.name)
    source_checks=[]; stale_new_code=[]
    seen=set()
    for context,binding in bindings:
        item=(context,binding.get("owner"),binding["path"],binding["sha256"])
        if item in seen:continue
        seen.add(item)
        if binding["path"].startswith("/"):
            issues.append(dict(file=context,reason="Absolute binding path"));continue
        bases=[roots[binding["owner"]]] if binding.get("owner") in roots else [source,published,root,private,doc]
        candidates=[base/binding["path"] for base in bases if (base/binding["path"]).is_file()]
        matching=[path for path in candidates if sha(path)==binding["sha256"]]
        if not matching:
            # Initial execution locks intentionally preserve pre-amendment code.
            amended_geometry = (binding["path"]==geometry_amendment["final_code_binding"]["path"]
                                 and binding["sha256"]==geometry_amendment["old_script_sha256"]
                                 and sha(root/binding["path"])==geometry_amendment["final_code_binding"]["sha256"])
            amended_classifier=(binding["path"]==classifier_amendment["current_source"]["path"]
                                and binding["sha256"] in archived_classifier_hashes
                                and binding.get("owner","new_worktree")=="new_worktree")
            if amended_geometry or amended_classifier:
                stale_new_code.append(dict(file=context,path=binding["path"],sha256=binding["sha256"],
                                          archived_source=archived_classifier_hashes.get(binding["sha256"]),
                                          reason="Verified preserved execution source; explicit pre-execution amendment is authoritative"))
            else:
                issues.append(dict(file=context,path=binding["path"],reason="Binding missing or SHA differs"))
        else:
            source_checks.append(dict(file=context,path=binding["path"],sha256=binding["sha256"],owner=binding.get("owner","resolved_existing_source")))
    issues.extend(dict(file=v["file"],reason="Forbidden reference-original key",key=v["key"]) for v in forbidden)
    code_root=root/"scripts/research/pallet_wd_info_fast_20261011"
    code_paths=[p for p in code_root.rglob("*") if p.is_file() and p.suffix in {".py",".mplstyle"}]
    for path in code_paths:
        if re.search(r"/(?:home/minjae|dev/shm|tmp)(?:/|\b)",path.read_text()):
            issues.append(dict(file=str(path.relative_to(root)),reason="Private absolute path in public source"))
    figure_sources=[code_root/"figures_report.py",code_root/"d_diagram.py"]
    for path in figure_sources:
        # Public figures must use derived metrics and schematic drawing primitives.
        assert not re.search(r"\b(?:imread|imshow|read_rgb)\s*\(|Image\.open\s*\(",path.read_text())
    return dict(status="PASS" if not issues else "FAIL", files=len(paths), bytes=byte_count,
                issues=issues, forbidden_reference_keys=forbidden, verified_bindings=source_checks,
                preserved_pre_amendment_code_bindings=stale_new_code,
                classifier_IO_amendment_independent_verification=classifier_amendment,
                public_source_files_audited=len(code_paths),public_figure_producers_RGB_read_or_plot_calls=0,
                PNG_scope="Repo-generated scientific plots/diagrams only; no new RGB copy", big_checkpoint_files=0)


def verify(args):
    doc=Path(args.public_dir);root=doc.parents[2]
    contract=read(doc/"INPUT_CONTRACT.json");ids=sorted(contract["allowed_ids"]);allowed=set(ids)
    raw=rows(doc/"FRAME_METRICS.jsonl.gz");metrics=read(doc/"METRICS.json");paired=read(doc/"PAIRED.json")
    draw=np.load(doc/"BOOTSTRAP_DRAWS.npy",allow_pickle=False);draw_lock=read(doc/"BOOTSTRAP_LOCK.json")
    sessions=sorted({fid.split(":")[0] for fid in ids})
    assert len(ids)==173 and len(sessions)==9 and draw.shape==(10000,9)
    assert sessions==draw_lock["sessions"] and sha(doc/"BOOTSTRAP_DRAWS.npy")==draw_lock["sha256"]
    assert np.array_equal(draw,np.random.default_rng(20260917).integers(0,9,size=(10000,9)))
    assert len(raw)==10726 and len({(r["backbone"],r["method"],r["id"],r["seed"]) for r in raw})==10726
    assert all(r["id"] in allowed and r["session"]==r["id"].split(":")[0] for r in raw)
    protected=set(contract["excluded_protected_ids"])
    assert not protected.intersection(r["id"] for r in raw)
    assert {r["id"] for r in raw if r["grade"]=="severe"}=={fid for fid in ids if any(q["id"]==fid and q["grade"]=="severe" for q in raw)}
    assert len({r["id"] for r in raw if r["grade"]=="severe"})==63
    grouped=defaultdict(list)
    for r in raw:grouped[r["backbone"],r["method"]].append(r)
    # Direct source joins prove the invalid first D rows were not normalized.
    source_d={(r["id"],r["method"],r["seed"]):r for r in rows(doc/"D_REPAIRED_ROWS.jsonl.gz")}
    source_c={(r["id"],r["method"],r["seed"]):r for r in rows(doc/"C_FRAME_METRICS.jsonl.gz")}
    joins=0
    for r in raw:
        expected=source_d.get((r["id"],r["method"],r["seed"])) if r["source_kind"]=="D" else source_c.get((r["id"],r["method"],r["seed"])) if r["source_kind"]=="C" else None
        if expected:
            assert r["available"]==int(expected["pose"]["available"])
            for field in ["translation_cm","rotation_deg","yaw_deg","ADDsym_m","IoU3D"]:
                compare(r[field],expected["pose"].get(field));joins+=1
        expected_success=int(bool(r["available"]) and r["translation_cm"]<5 and r["rotation_deg"]<5)
        expected_confusion=int(bool(r["available"]) and r["rotation_deg"]>45 and abs(r["yaw_deg"])>=60)
        assert r["pose_success"]==expected_success and r["wd_confusion"]==expected_confusion
    summary_checks=0;maximum=0;ci_maximum=0;auc_maximum=0
    multiplicities=np.array([list(Counter(map(int,row)).get(i,0) for i in range(9)) for row in draw],dtype=np.int64)
    subsets={"REAL173":allowed,"RECT245_INTERSECTION110":set(contract["rect245_intersection_ids"])}
    assert len(metrics["summaries"])==52
    for summary in metrics["summaries"]:
        subset=subsets[summary["population"]];rr=[r for r in grouped[summary["backbone"],summary["method"]] if r["id"] in subset]
        by=defaultdict(list)
        for r in rr:by[r["id"]].append(r)
        assert set(by)==subset and summary["frames"]==len(subset) and summary["prediction_rows"]==len(rr)
        assert summary["seeds"]==sorted({r["seed"] for r in rr})
        assert len({len(value) for value in by.values()})==1
        for field in KEYS:
            frame_values={}
            for fid,values in by.items():
                vv=[r[field] for r in values if r[field] is not None]
                if vv:frame_values[fid]=math.fsum(vv)/len(vv)
            values=list(frame_values.values());stored=summary["metrics"][field]
            assert stored["n"]==len(values)
            mean=math.fsum(values)/len(values) if values else None
            variance=math.fsum((value-mean)**2 for value in values)/(len(values)-1) if len(values)>1 else None
            expected=dict(mean=mean,variance=variance,SD=math.sqrt(variance) if variance is not None else None,
                          median=quantile(values,.5) if values else None,P90=quantile(values,.9) if values else None,
                          P95=quantile(values,.95) if values else None,maximum=max(values) if values else None)
            for name,value in expected.items():
                maximum=max(maximum,compare(value,stored[name]));summary_checks+=1
            totals=[math.fsum(value for fid,value in frame_values.items() if fid.split(":")[0]==name) for name in sessions]
            counts=[sum(fid.split(":")[0]==name for fid in frame_values) for name in sessions]
            numerator=multiplicities@np.asarray(totals);denominator=multiplicities@np.asarray(counts)
            sampled=[float(n/d) for n,d in zip(numerator,denominator) if d>0]
            assert stored["empty_bootstrap_draws"]==int((denominator==0).sum())
            expected_ci=[quantile(sampled,.025),quantile(sampled,.975)] if sampled else None
            if expected_ci is None:assert stored["CI95_mean_session_cluster"] is None
            else:
                for a,b in zip(expected_ci,stored["CI95_mean_session_cluster"]):ci_maximum=max(ci_maximum,compare(a,b))
        # Independent exact grid integration: endpoints weight1, interior2.
        errors=[r["ADDsym_normalized"] if r["ADDsym_normalized"] is not None else float("inf") for r in rr]
        accuracy=[sum(value<=threshold for value in errors) for threshold in np.linspace(0,.1,1001)]
        expected_auc=Fraction(accuracy[0]+accuracy[-1]+2*sum(accuracy[1:-1]),2000*len(errors))
        auc_maximum=max(auc_maximum,compare(float(expected_auc),summary["ADDsym_AUC_full"]))
    assert [(r["after"],r["before"],r["metric"]) for r in paired["results"]]==[(a,b,k) for a,b in CONTRASTS for k in ["wd_confusion","pose_success"]]
    independent=[]
    for stored in paired["results"]:
        aa={(r["id"],r["seed"]):r for r in grouped["YOLO",stored["after"]]}
        bb={(r["id"],r["seed"]):r for r in grouped["YOLO",stored["before"]]}
        assert set(aa)==set(bb)
        seeds=sorted({seed for fid,seed in aa});count=len(seeds)
        delta={fid:sum(aa[fid,seed][stored["metric"]]-bb[fid,seed][stored["metric"]] for seed in seeds) for fid in ids}
        totals=[sum(delta[fid] for fid in ids if fid.split(":")[0]==name) for name in sessions]
        counts=[sum(fid.split(":")[0]==name for fid in ids)*count for name in sessions]
        observed=Fraction(sum(totals),sum(counts));samples=[];extreme=0
        for row in draw:
            sample=Fraction(sum(totals[int(i)] for i in row),sum(counts[int(i)] for i in row))
            samples.append(float(sample*100));extreme+=int(abs(sample-observed)>=abs(observed))
        p=Fraction(1+extreme,10001)
        assert stored["numerator"]==observed.numerator*(sum(counts)//observed.denominator)
        assert stored["denominator"]==sum(counts) and stored["frames"]==173 and stored["seeds"]==count
        compare(float(observed*100),stored["effect_pp"]);compare(float(p),stored["p_raw"],1e-15)
        independent_ci=[quantile(samples,.025),quantile(samples,.975)]
        for a,b in zip(independent_ci,stored["CI95"]):compare(a,b)
        equal=math.fsum(float(Fraction(total,n)*100) for total,n in zip(totals,counts))/9
        compare(equal,stored["session_equal_effect_pp"])
        independent.append(dict(after=stored["after"],before=stored["before"],metric=stored["metric"],
                                p=p,extreme_draws=extreme,effect_pp=float(observed*100),CI95=independent_ci,seeds=count))
    ordered=sorted(range(8),key=lambda i:(independent[i]["p"],i));running=Fraction(0)
    for rank,index in enumerate(ordered):
        running=max(running,min(Fraction(1),independent[index]["p"]*(8-rank)))
        compare(float(running),paired["results"][index]["p_Holm"],1e-15)
        independent[index]["p_Holm"]=float(running)
    for row in independent:row["p_raw"]=float(row.pop("p"))
    privacy=audit_public(root,doc,Path(args.source_root),Path(args.published_root),Path(args.private_dir))
    result=dict(status="PASS" if privacy["status"]=="PASS" else "STATISTICS_PASS_PUBLIC_AUDIT_FAIL",
                independent_implementation="Python math.fsum/sample variance/sorted linear quantiles; Fraction rational bootstrap tail tests; exact trapezoidal count AUC; no producer stats import",
                frames=173,sessions=sessions,protected_frames_in_aggregate=0,severe_frames_preserved=63,
                raw_deduplicated_rows=len(raw),summaries=52,primary_hypotheses=8,
                source_row_joins=joins,summary_statistic_comparisons=summary_checks,
                maximum_moment_absolute_difference=maximum,maximum_CI_absolute_difference=ci_maximum,
                maximum_ADDsym_AUC_absolute_difference=auc_maximum,primary_exact_rational_checks=independent,
                D_invalid_first_attempt_used_in_aggregate=False,D_source="D_REPAIRED_ROWS.jsonl.gz",
                draw_file_sha256=sha(doc/"BOOTSTRAP_DRAWS.npy"),privacy_and_source_audit=privacy,
                additional_PnP=0,additional_training=0,additional_inference=0)
    destination=doc/args.output_name
    assert not destination.exists();destination.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    print(json.dumps({key:value for key,value in result.items() if key!="privacy_and_source_audit"},ensure_ascii=False))
    print("PUBLIC_AUDIT",json.dumps({key:value for key,value in privacy.items() if key!="verified_bindings"},ensure_ascii=False))
    assert privacy["status"]=="PASS",privacy["issues"]


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for name in ["public-dir","source-root","published-root","private-dir"]:parser.add_argument("--"+name,required=True)
    parser.add_argument("--output-name",default="FINAL_AGGREGATE_VERIFICATION.json")
    verify(parser.parse_args())
