"""Independent sealed-row arithmetic and disclosure checks.

This verifier imports no training, inference, pose or producer statistics code.
Scalar moments use math.fsum and linear sorted quantiles; bootstrap ties use
integer cross-products. It writes a new receipt and preserves existing files.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

PRIMARY=[(f"M{i}_SUBPIX","POSEFIX_SUBPIX") for i in range(5)]
MECHANISM=[("M1_SUBPIX","M0_SUBPIX"),("M2_SUBPIX","M1_SUBPIX"),
           ("M3_SUBPIX","M1_SUBPIX"),("M4_SUBPIX","M3_SUBPIX")]
METRICS=("Tcm","Rdeg","ADDsym_m","IoU3D","corner_error_px")
FORBIDDEN={"q0","qFinal","gt","R_gt","t_gt","manual_points","reference_corners",
           "reference_points","reference_R_cf","reference_bottom_center","raw_RGB"}


def read(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
    return h.hexdigest()
def load(paths):
    out={}
    for path in paths:
        with gzip.open(path,"rt",encoding="utf-8") as f:
            for line in f:
                r=json.loads(line);key=r["method"],r["id"],int(r["seed"])
                if key in out:assert out[key]==r
                out[key]=r
    return out
def finite(value):return value is not None and math.isfinite(float(value))
def quantile(values,q):
    values=sorted(values);p=(len(values)-1)*q;i=math.floor(p);j=math.ceil(p)
    return values[i] if i==j else values[i]*(j-p)+values[j]*(p-i)
def moments(values):
    v=[float(x) for x in values if finite(x)]
    if not v:return dict(n=0,mean=None,variance=None,SD=None,median=None,P90=None,P95=None,minimum=None,maximum=None)
    mean=math.fsum(v)/len(v);variance=math.fsum((x-mean)**2 for x in v)/(len(v)-1) if len(v)>1 else None
    return dict(n=len(v),mean=mean,variance=variance,SD=math.sqrt(variance) if variance is not None else None,
                median=quantile(v,.5),P90=quantile(v,.9),P95=quantile(v,.95),minimum=min(v),maximum=max(v))
def corners(row):
    assert len(row["corner_errors_px"])==len(row["corner_valid"])==8
    return [float(e) if finite(e) else math.inf for e,v in zip(row["corner_errors_px"],row["corner_valid"]) if v]
def val(row,key):
    if key=="corner_error_px":
        c=corners(row);return math.fsum(c)/len(c) if c else None
    if not row["available"]:return None
    aliases={"Tcm":["Tcm","translation_cm","T_cm"],"Rdeg":["Rdeg","rotation_deg","R_deg"],
             "ADDsym_m":["ADDsym_m"],"IoU3D":["IoU3D","IoU"]}
    for k in aliases[key]:
        if k in row["pose"]:return float(row["pose"][k]) if finite(row["pose"][k]) else None
    return None
def success(row):
    t,r=val(row,"Tcm"),val(row,"Rdeg")
    return bool(row["available"] and t is not None and r is not None and t<5 and r<5)


class Compare:
    def __init__(self):self.count=0;self.maxdiff=0.
    def check(self,a,b,tolerance=1e-8):
        self.count+=1
        if a is None or b is None:assert a is None and b is None,(a,b);return
        delta=abs(float(a)-float(b));self.maxdiff=max(self.maxdiff,delta)
        assert delta<=tolerance,(a,b,delta)
    def packet(self,values,stored):
        expected=moments(values)
        for key,value in expected.items():self.check(value,stored[key])


class Draws:
    def __init__(self,ids,sessions,draw):
        self.ids=ids;self.names=sorted(set(sessions));self.sessions=sessions
        self.indices=[[k for k,s in enumerate(sessions) if s==name] for name in self.names]
        self.multiplicity=np.array([[int(np.count_nonzero(row==i)) for i in range(len(self.names))] for row in draw],dtype=np.int64)
    def ci(self,numerator,denominator=None):
        if denominator is None:denominator=[1 if finite(x) else 0 for x in numerator]
        totals=[];counts=[]
        for indices in self.indices:
            valid=[i for i in indices if finite(numerator[i]) and finite(denominator[i]) and denominator[i]>0]
            totals.append(math.fsum(float(numerator[i]) for i in valid));counts.append(math.fsum(float(denominator[i]) for i in valid))
        # Separate per-column accumulation avoids the producer's bincount and
        # matrix multiplication implementation.
        n=np.zeros(len(self.multiplicity));d=np.zeros(len(self.multiplicity))
        for k,(t,c) in enumerate(zip(totals,counts)):
            n+=self.multiplicity[:,k]*t;d+=self.multiplicity[:,k]*c
        samples=[float(a/b) for a,b in zip(n,d) if b>0]
        return [quantile(samples,.025),quantile(samples,.975)] if samples else None
    def binary(self,before,after):
        delta=[[int(a)-int(b) for a,b in zip(ar,br)] for ar,br in zip(after,before)]
        totals=[sum(sum(delta[i]) for i in indices) for indices in self.indices]
        counts=[2*len(indices) for indices in self.indices]
        fulln=sum(totals);fulld=sum(counts);samples=[];extreme=0
        for multiplicity in self.multiplicity:
            n=sum(int(m)*t for m,t in zip(multiplicity,totals));d=sum(int(m)*c for m,c in zip(multiplicity,counts))
            samples.append(n/d*100)
            extreme+=abs(n*fulld-fulln*d)>=abs(fulln*d)
        return dict(effect_pp=fulln/fulld*100,CI95=[quantile(samples,.025),quantile(samples,.975)],
                    p_raw=(1+extreme)/(1+len(samples)),extreme_draws=extreme,numerator=fulln,denominator=fulld,
                    per_seed_effect_pp=[sum(row[s] for row in delta)/len(self.ids)*100 for s in range(2)])


def audit_group(packet,method,index,ids,draw,check):
    seeds=packet["seeds"];rr=[[index[method,fid,s] for s in seeds] for fid in ids]
    assert packet["frames"]==len(ids) and packet["prediction_rows"]==len(ids)*len(seeds)
    for metric in METRICS:
        values=[]
        for records in rr:
            finite_values=[val(r,metric) for r in records if finite(val(r,metric))]
            values.append(math.fsum(finite_values)/len(finite_values) if finite_values else None)
        stored=packet["metrics"][metric];check.packet(values,stored)
        ci=draw.ci(values)
        if ci is None:assert stored["mean_CI95"] is None
        else:
            for a,b in zip(ci,stored["mean_CI95"]):check.check(a,b)
        for seed in seeds:
            check.packet([val(index[method,fid,seed],metric) for fid in ids],packet["per_seed"][str(seed)]["metrics"][metric])
    for label,func in [("strict_success",success),("pose_availability",lambda r:bool(r["available"]))]:
        values=[sum(func(r) for r in records)/len(seeds) for records in rr];p=packet[label]
        count=sum(sum(func(r) for r in records) for records in rr)
        check.check(p["rate"],math.fsum(values)/len(ids));check.check(p["denominator"],len(ids)*len(seeds))
        check.check(p["successes" if label=="strict_success" else "available"],count)
        for a,b in zip(draw.ci(values),p["CI95"]):check.check(a,b)
    for threshold in (1,2,3,5,10):
        n=[sum(sum(e<=threshold for e in corners(r)) for r in records)/len(seeds) for records in rr]
        d=[sum(len(corners(r)) for r in records)/len(seeds) for records in rr]
        p=packet["PCK"][str(threshold)]
        for a,b in [(math.fsum(n)/math.fsum(d),p["rate"]),(math.fsum(n),p["hits_seed_mean"]),(math.fsum(d),p["reference_corners"])]:check.check(a,b)
        for a,b in zip(draw.ci(n,d),p["CI95"]):check.check(a,b)
        for seed in seeds:
            records=[index[method,fid,seed] for fid in ids]
            rate=sum(sum(e<=threshold for e in corners(r)) for r in records)/sum(len(corners(r)) for r in records)
            check.check(rate,packet["per_seed"][str(seed)]["PCK"][str(threshold)])
    axes=[r["axis_wrong"] for records in rr for r in records if r.get("axis_wrong") is not None]
    check.check(sum(bool(x) for x in axes),packet["axis_wrong"]["errors"]);check.check(len(axes),packet["axis_wrong"]["denominator"])
    if axes:check.check(sum(bool(x) for x in axes)/len(axes),packet["axis_wrong"]["rate"])
    confusion=[sum(bool(r["wd_confusion"]) for r in records)/len(seeds) for records in rr]
    check.check(math.fsum(confusion)/len(ids),packet["legacy_WD_confusion"]["rate"])
    for a,b in zip(draw.ci(confusion),packet["legacy_WD_confusion"]["CI95"]):check.check(a,b)
    for seed in seeds:
        records=[index[method,fid,seed] for fid in ids];p=packet["per_seed"][str(seed)]
        check.check(sum(success(r) for r in records),p["strict_successes"]);check.check(sum(not r["available"] for r in records),p["unavailable"])


def audit_family(records,pairs,index,ids,draw,check):
    assert [(r["after"],r["before"]) for r in records]==pairs
    independent=[]
    for row,(after,before) in zip(records,pairs):
        complete=all((m,fid,s) in index for m in (after,before) for fid in ids for s in (1,2))
        if not complete:
            assert row["status"]=="BLOCKED_OR_INCOMPLETE" and row["p_raw"] is None
            independent.append(None);continue
        a=[[success(index[after,fid,s]) for s in (1,2)] for fid in ids];b=[[success(index[before,fid,s]) for s in (1,2)] for fid in ids]
        packet=draw.binary(b,a);independent.append(packet)
        for key in ("effect_pp","p_raw","extreme_draws","numerator","denominator"):check.check(packet[key],row[key])
        for key in ("CI95","per_seed_effect_pp"):
            for x,y in zip(packet[key],row[key]):check.check(x,y)
    ordered=sorted((r["p_raw"],i) for i,r in enumerate(independent) if r is not None);running=0.
    for rank,(p,index_in_family) in enumerate(ordered):
        running=max(running,min(1.,(len(pairs)-rank)*p));check.check(running,records[index_in_family]["p_Holm"])
    return independent


def audit_transitions(doc,index,ids,check):
    with (doc/"ALL_PAIRED_TRANSITIONS.csv").open(newline="") as f:records=list(csv.DictReader(f))
    expected={(a,b,fid,s) for a,b in PRIMARY+MECHANISM for fid in ids for s in (1,2) if (a,fid,s) in index and (b,fid,s) in index}
    assert len(records)==len(expected);actual=set();changed=[]
    for row in records:
        key=row["after"],row["before"],row["id"],int(row["seed"]);assert key in expected and key not in actual;actual.add(key)
        after,before,fid,seed=key;a,b=index[after,fid,seed],index[before,fid,seed]
        ea,eb=corners(a),corners(b);sa,sb=success(a),success(b)
        event="RECOVERY" if sa and not sb else "DAMAGE" if sb and not sa else "UNCHANGED"
        assert row["event"]==event
        for key_,v in [("before_Tcm",val(b,"Tcm")),("after_Tcm",val(a,"Tcm")),("before_Rdeg",val(b,"Rdeg")),("after_Rdeg",val(a,"Rdeg")),
                       ("corners_damaged",sum(x>y+1e-8 for x,y in zip(ea,eb))),("corners_improved",sum(x<y-1e-8 for x,y in zip(ea,eb)))]:check.check(v,float(row[key_]) if row[key_]!="" else None)
        assert row["before_success"]==str(sb) and row["after_success"]==str(sa)
        if event!="UNCHANGED":changed.append(key)
    stored=read(doc/"CHANGED_SUCCESS_CASES.json")["rows"]
    assert set(changed)=={(r["after"],r["before"],r["id"],r["seed"]) for r in stored} and len(changed)==len(stored)
    failure=read(doc/"FAILURES.json")["unavailable"]
    for method,seeds in failure.items():
        for seed,failures in seeds.items():assert failures==[fid for fid in ids if not index[method,fid,int(seed)]["available"]]
    return dict(status="PASS",all_pair_rows=len(records),all_changed_success_rows=len(changed),all_damage_and_recovery_retained=True)


def audit_public(doc,root,pending_names):
    issues=[];pending=[];count=0;table_rows=0
    def walk(value,name):
        if isinstance(value,dict):
            for key,child in value.items():
                # METRICS.PCK.reference_corners is a scalar denominator count,
                # independently checked above; an array with the same name
                # would contain original reference coordinates and is blocked.
                scalar_reference_count=(key=="reference_corners" and isinstance(child,(int,float)) and finite(child) and child>=0)
                if key in FORBIDDEN and not scalar_reference_count:issues.append(dict(file=name,reason="Public original reference/prediction coordinate field",key=key))
                walk(child,name)
        elif isinstance(value,list):
            for child in value:walk(child,name)
    for base in (doc,root/"scripts/research/pallet_nonsevere_precision_20261011"):
        for path in base.rglob("*"):
            if not path.is_file():continue
            count+=1
            if path.suffix in {".pt",".pth",".npz",".jpg",".jpeg",".pyc",".tex",".pdf"} or "__pycache__" in path.parts:issues.append(dict(file=path.name,reason="Forbidden public container"))
            if path.stat().st_size>20*1024*1024:issues.append(dict(file=path.name,reason="Public artifact exceeds20MiB"))
            if path.suffix in {".png",".npy"}:continue
            if path.suffix==".gz":
                with gzip.open(path,"rt",encoding="utf-8") as f:text=f.read()
            else:text=path.read_text()
            if re.search(r"/(?:home/minjae|dev/shm|tmp)(?:/|\b)",text):issues.append(dict(file=path.name,reason="Private absolute path"))
            if path.suffix==".json":walk(json.loads(text),path.name)
            elif path.name.endswith(".jsonl.gz"):
                for line in text.splitlines():walk(json.loads(line),path.name)
            if path.suffix!=".md":continue
            fenced=False;columns=None
            for lineno,line in enumerate(text.splitlines(),1):
                if line.lstrip().startswith("```"):fenced=not fenced;columns=None;continue
                if fenced:continue
                if re.fullmatch(r"\s*\|(?:\s*:?-+:?\s*\|)+\s*",line):columns=len(re.findall(r"(?<!\\)\|",line));continue
                if columns is not None:
                    if not line.lstrip().startswith("|"):columns=None
                    else:
                        table_rows+=1;n=len(re.findall(r"(?<!\\)\|",line))
                        if n!=columns:issues.append(dict(file=path.name,line=lineno,reason="Markdown delimiter mismatch"))
            for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)",text):
                if target.startswith(("https:","http:","mailto:")):continue
                target=target.split("#")[0].strip("<>")
                if not target:continue
                linked=(path.parent/target).resolve()
                if not linked.exists():
                    item=dict(file=path.name,target=target)
                    if linked.parent==doc and linked.name in pending_names:pending.append(item)
                    else:issues.append({**item,"reason":"Missing local Markdown target"})
    return dict(status="PASS" if not issues else "FAIL",files=count,issues=issues,Markdown_table_rows=table_rows,pending_generated_after_this_audit=pending)


def audit_sources(doc,owners):
    result=[]
    bindings=read(doc/"SOURCE_BINDINGS.json")["bindings"]
    for binding in bindings:
        if binding["owner"] not in owners:continue
        path=owners[binding["owner"]]/binding["path"];assert sha(path)==binding["sha256"],binding["path"]
        result.append(dict(owner=binding["owner"],path=binding["path"],sha256=binding["sha256"]))
    return dict(status="PASS" if len(result)==len(bindings) else "PARTIAL_OWNER_ROOTS_NOT_SUPPLIED",verified=len(result),declared=len(bindings))


def audit_latency(doc,ids,check):
    path=doc/"LATENCY.json"
    if not path.exists():return dict(status="NOT_YET_MEASURED",accuracy_speed_claim_checked=False)
    aggregate=read(path);count=0;panels=set()
    for method,packet in aggregate["methods"].items():
        if packet["status"]=="NOT_MEASURED":continue
        receipt=read(doc/packet["receipt"]);assert sha(doc/packet["receipt"])==packet["receipt_sha256"]
        timed=[r for r in receipt["per_call"] if r["phase"]=="TIMED"]
        warm=[r for r in receipt["per_call"] if r["phase"]=="WARMUP"]
        assert len(timed)==90 and len(warm)==20
        assert all(r["id"] in ids for r in timed+warm)
        assert Counter(r["id"] for r in timed)==Counter({fid:5 for fid in receipt["panel_ids"]})
        assert len(receipt["panel_ids"])==18 and len({fid.split(":")[0] for fid in receipt["panel_ids"]})==9
        expected=[fid for session in sorted({i.split(":")[0] for i in ids}) for fid in sorted(i for i in ids if i.split(":")[0]==session)[:2]]
        assert receipt["panel_ids"]==expected
        panels.add(tuple(receipt["panel_ids"]))
        check.packet([r["latency_ms"] for r in timed],packet["latency_ms"])
        for key in ("peak_memory_MiB","peak_reserved_MiB"):check.check(packet[key],receipt[key])
        assert receipt["precision"]==dict(dtype="FP32",autocast=False,matmul_TF32=False,cudnn_TF32=False,cudnn_benchmark=False)
        assert receipt["training_or_gradient_updates"]==0 and receipt["GT_inputs"] is False
        assert receipt["CUDA_sync_before_and_after_every_timed_call"] and receipt["independent_process_peak_reset_after_load"]
        count+=1
    assert len(panels)<=1
    return dict(status="PASS",independent_process_routes=count,all_same18image_panel_and_precision=True,
                measured_cost_moments_checked=True,missing_routes_explicit=True,global_optimality_not_claimed=True)


def audit_training(doc,private,check):
    path=doc/"TRAIN_RECEIPTS.json"
    if not path.exists():return dict(status="NOT_YET_EXECUTED",matched_training_checked=False)
    train=read(path);assert train["seeds"]==[1,2] and train["planned_arms"]==["M0","M1","M2","M3","M4"]
    executed=train["executed_arms"];assert set(executed)|set(train["blocked_arms"])==set(train["planned_arms"])
    assert not set(executed)&set(train["blocked_arms"])
    assert 0<train["common_steps"]<=3000 and train["FP32"] and not train["AMP"] and not train["matmul_TF32"] and not train["cudnn_TF32"]
    assert not train["real_selection"]
    total=0;records=[];device=[]
    for name in train["receipts"]:
        receipt=read(doc/name);assert receipt["updates_per_arm"]==train["common_steps"]
        assert receipt["arms"]==executed and receipt["batch"]<=16
        assert receipt["last_checkpoint_only"] and receipt["source_checkpoint_unchanged"]
        assert receipt["backbone_training_updates"]==receipt["protected_or_Severe_REAL_training_reads"]==0
        assert receipt["history"][-1]["step"]==train["common_steps"]
        assert all(set(h["arms"])==set(executed) for h in receipt["history"])
        assert receipt["total_updates"]==len(executed)*train["common_steps"]
        if private:
            for ck in receipt["checkpoints"].values():assert sha(private/ck["private_namespace"])==ck["sha256"]
        total+=receipt["total_updates"];device.append(receipt["actual_device_seconds"])
        records.append(dict(seed=receipt["seed"],updates_per_arm=receipt["updates_per_arm"],arms=receipt["arms"],
                            source_checkpoint_sha256=receipt["original_checkpoint_sha256"],last_step=receipt["history"][-1]["step"]))
    assert sorted(r["seed"] for r in records)==[1,2] and total==train["total_updates"]
    weights=read(doc/"AUX_WEIGHT_LOCK.json")
    assert weights["real_reads"]==weights["optimizer_updates"]==0 and weights["minibatches"]==32
    assert weights["shared_step0_logits_exact"] and weights["gradient_target_ratio"]==.25
    for kind in ("COORD","GEO"):
        if weights["weights"].get(kind) is not None:
            check.check(weights["weights"][kind]*weights["mean_norms"][kind]/weights["mean_norms"]["CE"],.25,tolerance=1e-5)
    check.check(sum(device)+weights["actual_device_seconds"],train["actual_device_seconds"])
    assert train["actual_device_seconds"]<=train["device_budget_seconds"]==7200
    return dict(status="PASS",matched_two_seed_last_steps=records,total_actual_updates=total,blocked_arms_preserved=train["blocked_arms"],
                synthetic_only_one_time_aux_gradient_weights=True,private_checkpoint_hashes_checked=private is not None,
                training_device_seconds=train["actual_device_seconds"],training_only_budget_checked=True,
                global_device_budget_requires_complete_compute_receipt=True)


def audit_figures(doc,root,private):
    original=read(doc/"FIGURE_RECEIPT.json")
    repair=read(doc/"PIPELINE_VISUAL_REPAIR.json") if (doc/"PIPELINE_VISUAL_REPAIR.json").exists() else None
    assert original["raw_RGB_published"] is False and original["additional_model_forwards_or_PnP_or_training"]==0
    for image in original["figures"]:
        expected=repair["new_PNG_sha256"] if repair and image["path"]=="FIGURES/01_pipeline.png" else image["sha256"]
        assert sha(doc/image["path"])==expected
    current=root/"scripts/research/pallet_nonsevere_precision_20261011/figures.py"
    if repair:
        assert repair["status"]=="PASS" and repair["original_receipt_preserved"]
        assert repair["original_figure_receipt_sha256"]==sha(doc/"FIGURE_RECEIPT.json")
        assert repair["repair_lock_sha256"]==sha(doc/"PIPELINE_VISUAL_REPAIR_LOCK.json")
        assert repair["new_source_sha256"]==sha(current)
        assert repair["additional_model_forwards"]==repair["additional_PnP"]==repair["additional_training"]==0
        assert repair["old_source_sha256"]==original["source_sha256"]
        assert repair["old_PNG_sha256"]==next(r["sha256"] for r in original["figures"] if r["path"]=="FIGURES/01_pipeline.png")
        if private:
            assert sha(private/repair["original_source_private_relative_path"])==repair["old_source_sha256"]
            assert sha(private/repair["original_PNG_private_relative_path"])==repair["old_PNG_sha256"]
        for name,digest in repair["other_PNGs_unchanged"].items():assert sha(doc/"FIGURES"/name)==digest
    else:
        assert original["source_sha256"]==sha(current)
    if private:
        for contact in original["private_RGB_contact_sheets"]:
            assert contact["RGB_public"] is False and sha(private/contact["relative_path"])==contact["sha256"]
    return dict(status="PASS",public_figure_count=len(original["figures"]),visual_only_repair_originals_preserved=bool(repair),
                private_RGB_contacts_checked=private is not None,scientific_rows_or_model_or_PnP_changed=0)


def audit_compute(doc,check):
    path=doc/"COMPUTE_RECEIPT.json"
    if not path.exists():return dict(status="NOT_YET_ACCOUNTED",global_budgets_checked=False)
    receipt=read(path);assert receipt["status"]=="PASS"
    training=read(doc/"TRAIN_RECEIPTS.json");selection=read(doc/"SYNTHETIC_SELECTION.json")
    expected_device={"train_and_auxiliary_calibration":training["actual_device_seconds"],
                     "forward_backward_dry_run":read(doc/"MODEL_DRY_RUN.json")["total_setup_seconds"]}
    stage_status={}
    for phase in ("selection","accuracy"):
        paths=sorted(doc.glob(f"STAGE_{phase}*.json"))
        assert paths and any(read(path)["status"]=="DONE" for path in paths)
        for path in paths:
            actual=read(path)
            assert actual["status"] in ("DONE","FAILED") and actual["device"]=="cuda"
            stage=actual["stage"]
            assert stage not in expected_device
            expected_device[stage]=actual["device_reserved_upper_seconds"]
            stage_status[stage]=(actual["status"],path.name)
    for latency in sorted(doc.glob("LATENCY_*.json")):
        actual=read(latency);assert actual["status"]=="COMPLETE"
        expected_device[latency.stem]=actual["GPU_device_job_wall_seconds_conservative"]
    got={row["stage"]:row["seconds"] for row in receipt["device_stage_receipts"]}
    assert len(got)==len(receipt["device_stage_receipts"]) and set(got)==set(expected_device)
    for stage,seconds in expected_device.items():check.check(got[stage],seconds)
    for row in receipt["device_stage_receipts"]:
        if row["stage"] in stage_status:
            status,filename=stage_status[row["stage"]]
            assert row["status"]==status and row["receipt"]==filename
    device=math.fsum(expected_device.values())
    check.check(receipt["measured_device_stage_seconds"],device)
    assert receipt["setup_and_failed_startup_upper_allowance_seconds"]==300
    check.check(receipt["accounted_device_upper_seconds"],device+300)
    check.check(receipt["accounted_device_upper_minutes"],(device+300)/60)
    assert receipt["accounted_device_upper_seconds"]<receipt["device_budget_seconds"]==7200
    cpu=receipt["CPU_analysis_stage_receipts"]
    assert len({row["stage"] for row in cpu})==len(cpu)
    for row in cpu:
        assert math.isfinite(row["seconds"]) and row["seconds"]>=0
        stage=row["stage"]
        if stage.startswith("STAGE_") or stage.startswith("whole_process_"):
            filename=("STAGE_"+stage.removeprefix("whole_process_")+".json") if stage.startswith("whole_process_") else (stage if stage.endswith(".json") else stage+".json")
            stage_path=doc/filename
            actual=read(stage_path);assert actual["status"]=="DONE" and actual["device"]=="cpu"
            check.check(row["seconds"],actual["wall_seconds"])
        elif stage=="synthetic_pose_scoring_also_in_device_wall":check.check(row["seconds"],selection["CPU_scoring_seconds"])
        elif stage.endswith(".json"):
            check.check(row["seconds"],read(doc/stage)["wall_seconds"])
        else:
            check.check(row["seconds"],read(doc/"DECODER_EXECUTION_RECEIPT.json")[stage])
    for mandatory in ("statistics","figures"):
        if (doc/f"STAGE_{mandatory}.json").exists():
            assert any(row["stage"] in (f"STAGE_{mandatory}",f"STAGE_{mandatory}.json",f"whole_process_{mandatory}") for row in cpu),("Recorded stage omitted",mandatory)
    assert receipt["CPU_uninstrumented_contract_publication_and_later_verification_upper_allowance_seconds"]==300
    cpu_total=math.fsum(row["seconds"] for row in cpu)
    check.check(receipt["accounted_CPU_analysis_upper_seconds"],cpu_total+300)
    assert receipt["accounted_CPU_analysis_upper_seconds"]<receipt["CPU_analysis_budget_seconds"]==3600
    assert receipt["actual_optimizer_updates"]==training["total_updates"]==30000
    assert receipt["common_updates_per_arm_seed"]==training["common_steps"]==3000
    assert receipt["actual_new_corrector_fits"]==10 and receipt["simultaneous_owned_GPU_jobs"]==1
    for key in ("new_backbone_updates","new_full_PoseFix_updates","new_synthetic_images",
                "REAL_Severe_training_selection_evaluation_rows","protected_FINAL_training_selection_evaluation_rows"):
        assert receipt[key]==0,key
    assert receipt["hyperparameter_search"] is False and receipt["no_source_files_modified"] is True
    evaluation=read(doc/"NEW_EVALUATION_RECEIPT.json")
    assert receipt["new_REAL_F_calls"]==evaluation["actual_F_calls"]
    assert receipt["new_REAL_OpenCV_calls"]==evaluation["total_OpenCV_calls"]
    assert receipt["synthetic_selection_F_calls"]==selection["pose_F_calls"]
    assert receipt["synthetic_selection_OpenCV_calls"]==selection["OpenCV_calls"]
    return dict(status="PASS",global_budgets_checked=True,device_stage_seconds=device,
                accounted_device_upper_seconds=device+300,CPU_stage_seconds=cpu_total,
                accounted_CPU_upper_seconds=cpu_total+300,independent_stage_is_later_and_covered_by_explicit300second_CPU_allowance=True,
                failed_reserved_device_stages_retained=[stage for stage,(status,_) in stage_status.items() if status=="FAILED"],
                actual_updates=30000,fits=10,new_backbone_updates=0,new_synthetic_data=0,
                Severe_and_protected_scoring=0)


def run(args):
    doc=Path(args.public_dir).resolve();root=doc.parents[2]
    output=doc/args.output_name;assert not output.exists(),"Existing independent receipt preserved"
    contract=read(args.contract);ids=sorted(contract["allowed_ids"]);assert len(ids)==110
    assert contract["current_protected_per_ID_access"]==0
    index=load(args.rows);assert all(fid in ids for _,fid,_ in index)
    assert all(r["grade"].lower() in ("clean","moderate") for r in index.values())
    evaluation=read(doc/"NEW_EVALUATION_RECEIPT.json")
    assert evaluation["rows"]==len(index) and evaluation["frames"]==110
    assert set(evaluation["methods"])=={method for method,_,_ in index}
    for method in evaluation["methods"]:
        new=bool(re.fullmatch(r"M[0-4](?:_SUBPIX)?",method))
        assert sorted({seed for m,_,seed in index if m==method})==([1,2] if new else [1,2,3])
    assert all(r["inference_reference_inputs"] is False for r in index.values())
    assert all(str(r["precision_basis"]).startswith("Fresh shared frozen YOLO FP32 inference") for r in index.values())
    metadata={}
    for r in index.values():
        assert success(r)==r["strict_success"]
        meta=r["session"],r["grade"].lower(),r["corner_valid"]
        if r["id"] in metadata:assert metadata[r["id"]]==meta
        metadata[r["id"]]=meta
    assert set(metadata)==set(ids)
    grade_counts=Counter(metadata[fid][1] for fid in ids);assert dict(grade_counts)==contract["grade_counts"]
    sessions=[metadata[fid][0] for fid in ids];names=sorted(set(sessions))
    rawdraw=np.load(doc/"BOOTSTRAP_DRAWS.npy",allow_pickle=False)
    assert rawdraw.shape==(10000,len(names)) and np.array_equal(rawdraw,np.random.default_rng(20260917).integers(0,len(names),size=rawdraw.shape))
    assert sha(doc/"BOOTSTRAP_DRAWS.npy")==read(doc/"BOOTSTRAP_LOCK.json")["sha256"]
    draw=Draws(ids,sessions,rawdraw);check=Compare();metrics=read(doc/"METRICS.json");paired=read(doc/"PAIRED_RESULTS.json")
    for method,packets in metrics["groups"].items():
        for packet in packets.values():
            if packet:audit_group(packet,method,index,ids,draw,check)
    primary=audit_family(paired["primary"],PRIMARY,index,ids,draw,check)
    mechanism=audit_family(paired["mechanism"],MECHANISM,index,ids,draw,check)
    transitions=audit_transitions(doc,index,ids,check)
    owners={}
    for item in args.owner_root:
        name,path=item.split("=",1);owners[name]=Path(path)
    sources=audit_sources(doc,owners)
    latency=audit_latency(doc,ids,check)
    figures=audit_figures(doc,root,Path(args.private_dir) if args.private_dir else None)
    training=audit_training(doc,Path(args.private_dir) if args.private_dir else None,check)
    compute=audit_compute(doc,check)
    public=audit_public(doc,root,set(args.allow_pending)|{args.output_name})
    receipt=read(doc/"STATISTICS_RECEIPT.json");assert receipt["additional_PnP"]==receipt["additional_training"]==receipt["additional_model_forwards"]==0
    assert receipt["rows"]==len(index)
    result=dict(status="PASS" if public["status"]=="PASS" else "ARITHMETIC_PASS_DISCLOSURE_FAIL",frames=len(ids),sessions=len(names),
                grades=dict(grade_counts),Severe_scored=0,protected_current_per_ID_access=contract["current_protected_per_ID_access"],
                row_count=len(index),scalar_comparisons=check.count,maximum_scalar_absolute_difference=check.maxdiff,
                arithmetic="Independent math.fsum and sample moments, sorted linear percentiles, separately accumulated session draws, exact integer centered bootstrap ties",
                primary_independent=primary,mechanism_independent=mechanism,transitions=transitions,source_bindings=sources,latency=latency,figures=figures,matched_training=training,complete_compute=compute,
                public_disclosure_and_Markdown=public,auditor_new_training=0,auditor_model_forwards=0,auditor_PnP=0,
                experiment_interpretation="Reused non-Severe development data, screen2seeds; arithmetic PASS does not establish independent physical accuracy",
                inputs_sha256={Path(p).name:sha(p) for p in args.rows},source_sha256=sha(Path(__file__)))
    with output.open("x",encoding="utf-8") as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")
    print(json.dumps(dict(status=result["status"],scalar_comparisons=check.count,maximum_difference=check.maxdiff,public_issues=public["issues"]),ensure_ascii=False))
    assert result["status"]=="PASS",public["issues"]


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--public-dir",required=True);p.add_argument("--rows",nargs="+",required=True)
    p.add_argument("--contract",required=True);p.add_argument("--owner-root",action="append",default=[])
    p.add_argument("--private-dir")
    p.add_argument("--allow-pending",action="append",default=[]);p.add_argument("--output-name",default="INDEPENDENT_VERIFICATION.json")
    run(p.parse_args())
