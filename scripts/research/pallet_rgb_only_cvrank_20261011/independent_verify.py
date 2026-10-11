"""Independent arithmetic for the fixed RGB-only/CVRANK development comparison.

No producer-statistics, model, training or pose implementation is imported.
The reused independent math uses fsum/sample moments/sorted linear quantiles,
separate session accumulation and exact integer centered-bootstrap ties.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv,gzip,hashlib,json,math,re,time
from datetime import datetime
from pathlib import Path
import numpy as np

PRIMARY=[("M3_SUBPIX","RGB_ONLY_SUBPIX"),("M3_SUBPIX","M1_SUBPIX"),("M3_SUBPIX","CVRANK_THEN_SUBPIX")]
DESCRIPTIVE=[("RGB_ONLY_SUBPIX","M1_SUBPIX"),("RGB_ONLY_SUBPIX","CVRANK_THEN_SUBPIX"),("M1_SUBPIX","CVRANK_THEN_SUBPIX"),
 ("RGB_ONLY","M1"),("M3","RGB_ONLY"),("M3","CVRANK_CAP1"),("RGB_ONLY","CVRANK_CAP1")]
WITHIN=[(m+"_SUBPIX",m) for m in ("OLD_P","M1","RGB_ONLY","M3","POSEFIX")]+[("CVRANK_THEN_SUBPIX","CVRANK_NATIVE")]
PAIRS=PRIMARY+DESCRIPTIVE+WITHIN
MAIN_METHODS=("BASE","SUBPIX","CVRANK_NATIVE","CVRANK_CAP1","CVRANK_THEN_SUBPIX","OLD_P","OLD_P_SUBPIX","M1","M1_SUBPIX","RGB_ONLY","RGB_ONLY_SUBPIX","M3","M3_SUBPIX","POSEFIX","POSEFIX_SUBPIX")
NEW_METHODS={"RGB_ONLY","RGB_ONLY_SUBPIX","CVRANK_NATIVE","CVRANK_CAP1","CVRANK_THEN_SUBPIX"}
METRICS=("Tcm","Rdeg","ADDsym_m","IoU3D","corner_error_px")
FORBIDDEN={"q0","qFinal","gt","R_gt","t_gt","manual_points","reference_corners","reference_points","reference_R_cf","reference_bottom_center","raw_RGB"}


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
    for base in (doc,root/"scripts/research/pallet_rgb_only_cvrank_20261011"):
        for path in base.rglob("*"):
            if not path.is_file():continue
            count+=1
            if "__pycache__" in path.parts:continue # transient interpreter files, never staged/public artifacts
            if path.suffix in {".pt",".pth",".npz",".jpg",".jpeg",".pyc",".tex",".pdf"}:issues.append(dict(file=path.name,reason="Forbidden public container"))
            if path.stat().st_size>20*1024*1024:issues.append(dict(file=path.name,reason="Public artifact exceeds20MiB"))
            if path.suffix==".npy" and path.name!="BOOTSTRAP_DRAWS.npy":issues.append(dict(file=path.name,reason="Unapproved public array"))
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


def csv_rows(path):
    with Path(path).open(newline="",encoding="utf-8") as handle:return list(csv.DictReader(handle))

def boolean(text):
    assert text in ("True","False");return text=="True"

def expected_transition(after,before,fid,seed,index):
    a,b=index[after,fid,seed],index[before,fid,seed];ea,eb=corners(a),corners(b)
    assert len(ea)==len(eb)
    am=math.fsum(ea)/len(ea) if ea and all(map(math.isfinite,ea)) else None
    bm=math.fsum(eb)/len(eb) if eb and all(map(math.isfinite,eb)) else None
    sa,sb=success(a),success(b)
    tradeoff=am is not None and bm is not None and am<bm-1e-8 and any(
        val(a,k) is not None and val(b,k) is not None and val(a,k)>val(b,k)+1e-8 for k in ("Tcm","Rdeg"))
    return dict(after=after,before=before,id=fid,seed=seed,session=a["session"],grade=a["grade"],
        event="RECOVERY" if sa and not sb else "DAMAGE" if sb and not sa else "UNCHANGED",
        before_success=sb,after_success=sa,before_available=bool(b["available"]),after_available=bool(a["available"]),
        before_Tcm=val(b,"Tcm"),after_Tcm=val(a,"Tcm"),before_Rdeg=val(b,"Rdeg"),after_Rdeg=val(a,"Rdeg"),
        before_corner_mean_px=bm,after_corner_mean_px=am,
        corners_damaged=sum(x>y+1e-8 for x,y in zip(ea,eb)),corners_improved=sum(x<y-1e-8 for x,y in zip(ea,eb)),
        before_hyp=b.get("selected_hyp"),after_hyp=a.get("selected_hyp"),corner_better_pose_worse=tradeoff,
        availability_transition=bool(a["available"])!=bool(b["available"]))

def audit_transitions(doc,index,ids,check):
    allrows=csv_rows(doc/"ALL_PAIRED_TRANSITIONS.csv");changed=csv_rows(doc/"CHANGED_CASES.csv")
    def key(r):return r["after"],r["before"],r["id"],int(r["seed"])
    stored={key(r):r for r in allrows};assert len(stored)==len(allrows)==3520
    expected={}
    for after,before in PAIRS:
        for fid in ids:
            for seed in (1,2):
                k=after,before,fid,seed;e=expected_transition(*k,index);expected[k]=e
                s=stored[k]
                for name,value in e.items():
                    if isinstance(value,bool):assert boolean(s[name])==value
                    elif isinstance(value,(int,float)):check.check(float(s[name]),value)
                    elif value is None:assert s[name]==""
                    else:assert s[name]==value
    want={k for k,e in expected.items() if e["event"]!="UNCHANGED" or e["corner_better_pose_worse"] or e["availability_transition"]}
    actual={key(r) for r in changed};assert len(actual)==len(changed) and actual==want
    assert all(row==stored[key(row)] for row in changed)
    summary=read(doc/"PAIRED_TRANSITION_SUMMARY.json")
    assert summary["all_pairs"]==[list(p) for p in PAIRS] and summary["all_rows"]==3520
    assert summary["changed_or_corner_pose_tradeoff_rows"]==len(changed) and summary["no_unfavorable_case_deleted"]
    perpair=[]
    for s,(after,before) in zip(summary["pairs"],PAIRS):
        rr=[r for k,r in expected.items() if k[:2]==(after,before)];c=Counter(r["event"] for r in rr)
        assert (s["after"],s["before"])==(after,before) and s["rows"]==220
        for name,event in [("recovery","RECOVERY"),("damage","DAMAGE"),("unchanged","UNCHANGED")]:check.check(s[name],c[event])
        for name in ("corners_damaged","corners_improved","corner_better_pose_worse"):check.check(s[name],sum(r[name] for r in rr))
        check.check(s["net_success_difference"],c["RECOVERY"]-c["DAMAGE"])
        assert s["descriptive"]==((after,before) not in PRIMARY)
        perpair.append(dict(after=after,before=before,recovery=c["RECOVERY"],damage=c["DAMAGE"],tradeoffs=sum(r["corner_better_pose_worse"] for r in rr)))
    return dict(status="PASS",all_rows=3520,changed_rows=len(changed),fixed_pairs=16,paired_counts=perpair)

def audit_table(doc,metrics,check):
    rr=csv_rows(doc/"ACCURACY_TABLE.csv");assert len(rr)==15 and {r["method"] for r in rr}==set(MAIN_METHODS)
    for row in rr:
        p=metrics["groups"][row["method"]]["common_seed12"]
        assert int(row["frames"])==110 and row["seeds"]=="1,2" and int(row["denominator"])==220
        for field,want in [("success",p["strict_success"]["successes"]),("success_percent",p["strict_success"]["rate"]*100),
             ("pose_unavailable",220-p["pose_availability"]["available"]),("axis_wrong",p["axis_wrong"]["errors"]),
             ("axis_denominator",p["axis_wrong"]["denominator"]),("legacy_WD_percent",p["legacy_WD_confusion"]["rate"]*100)]:check.check(float(row[field]),want)
        for metric in METRICS:
            for stat in ("mean","variance","SD","median","P90","P95"):
                text=row[metric+"_"+stat];check.check(float(text) if text else None,p["metrics"][metric][stat])
            for a,b in zip(json.loads(row[metric+"_mean_CI95"]),p["metrics"][metric]["mean_CI95"]):check.check(a,b)
        for t in (1,2,3,5,10):check.check(float(row[f"PCK_{t}px_percent"]),p["PCK"][str(t)]["rate"]*100)
    return dict(status="PASS",rows=15)

def audit_bindings(contract,owners):
    checked=[]
    for binding in contract["bindings"]:
        path=owners[binding["owner"]]/binding["path"]
        assert path.is_file() and path.stat().st_size==binding["bytes"] and sha(path)==binding["sha256"],binding["path"]
        checked.append(dict(owner=binding["owner"],path=binding["path"],sha256=binding["sha256"]))
    return dict(status="PASS",files=len(checked),all_original_bytes_match=True)

def private_rows(path):
    with gzip.open(path,"rt",encoding="utf-8") as handle:return [json.loads(line) for line in handle]

def independent_cap(initial,native,support,hw):
    initial=np.asarray(initial,float);native=np.asarray(native,float);support=np.asarray(support,bool)
    usable=support[:8]&np.isfinite(initial[:8]).all(-1)&~(initial[:8]==-1).all(-1)
    result=initial.copy();limit=.01*math.hypot(*hw)
    for k in range(8):
        if not usable[k]:continue
        displacement=native[k]-initial[k];length=math.sqrt(math.fsum(float(x*x) for x in displacement))
        result[k]=initial[k]+displacement*(min(1.,limit/length) if length>0 else 1.)
    return result

def audit_private_predictions(private,ids):
    predictions=private_rows(private/"CVRANK_PREDICTIONS_SEALED.jsonl.gz")
    index={(r["method"],r["id"],r["seed"]):r for r in predictions};assert len(index)==len(predictions)==660
    diag={r["id"]:r for r in private_rows(private/"CVRANK_DIAGNOSTICS.jsonl.gz")};assert set(diag)==set(ids)
    maxdelta=0.
    for fid in ids:
        native=index["CVRANK_NATIVE",fid,1]
        for method,start in [("CVRANK_CAP1",native["qFinal"]),("CVRANK_THEN_SUBPIX",diag[fid]["SUBPIX"]["raw_subpix"])]:
            row=index[method,fid,1];expected=independent_cap(row["q0"],start,row["prediction_support"],row["raw_hw"])
            got=np.asarray(row["qFinal"],float);assert np.array_equal(np.isnan(expected),np.isnan(got))
            delta=float(np.max(np.abs(expected[np.isfinite(expected)]-got[np.isfinite(got)])))
            assert delta<1e-10;maxdelta=max(maxdelta,delta)
            assert np.array_equal(np.asarray(row["subpix_input"]),np.asarray(native["qFinal"]),equal_nan=True)
        for method in ("CVRANK_NATIVE","CVRANK_CAP1","CVRANK_THEN_SUBPIX"):
            a,b=index[method,fid,1],index[method,fid,2]
            for key in ("q0","qFinal","prediction_support","raw_hw","fixed_metadata","subpix_input"):
                assert a[key]==b[key]
            q0,q=np.asarray(a["q0"],float),np.asarray(a["qFinal"],float);support=np.asarray(a["prediction_support"],bool)
            usable=support[:8]&np.isfinite(q0[:8]).all(-1)&~(q0[:8]==-1).all(-1)
            assert np.array_equal(q[8],q0[8],equal_nan=True) and np.array_equal(q[:8][~usable],q0[:8][~usable],equal_nan=True)
    return dict(status="PASS",CV_frames=110,CV_rows=660,CV_seeds_duplicate_identically=True,
        native_to_SubPix_input_exact=True,independent_cap_max_abs=maxdelta,original_center_and_missing_slots_preserved=True,
        new_image_decodes=0,new_classical_calls=0,new_PnP=0)

def audit_latency(doc,prior,ids,check):
    aggregate=read(doc/"LATENCY.json");assert set(aggregate["methods"])==set(MAIN_METHODS)
    panels=set();measured=[]
    for method,packet in aggregate["methods"].items():
        assert packet["status"]!="NOT_MEASURED",method
        receiptpath=(doc/packet.get("original_receipt",packet["receipt"])).resolve();receipt=read(receiptpath)
        assert sha(receiptpath)==packet["receipt_sha256"]
        timed=[r for r in receipt["per_call"] if r["phase"]=="TIMED"];warm=[r for r in receipt["per_call"] if r["phase"]=="WARMUP"]
        assert len(timed)==90 and len(warm)==20
        expected=[fid for session in sorted({i.split(":")[0] for i in ids}) for fid in sorted(i for i in ids if i.split(":")[0]==session)[:2]]
        assert receipt["panel_ids"]==expected and Counter(r["id"] for r in timed)==Counter({fid:5 for fid in expected})
        panels.add(tuple(expected));check.packet([r["latency_ms"] for r in timed],packet["latency_ms"])
        for key in ("peak_memory_MiB","peak_reserved_MiB"):check.check(packet[key],receipt[key])
        assert receipt["precision"]==dict(dtype="FP32",autocast=False,matmul_TF32=False,cudnn_TF32=False,cudnn_benchmark=False)
        assert receipt["training_or_gradient_updates"]==0 and receipt["GT_inputs"] is False
        assert receipt["CUDA_sync_before_and_after_every_timed_call"] and receipt["independent_process_peak_reset_after_load"]
        if method not in NEW_METHODS:
            oldpacket=read(prior/"LATENCY.json")["methods"][method];assert packet["receipt_sha256"]==oldpacket["receipt_sha256"]
        else:
            physical=Counter()
            for call in receipt["per_call"]:
                assert call["counters"]["F_calls"]==1
                physical.update(call["counters"]["opencv_calls"])
            assert dict(physical)==receipt["actual_OpenCV_calls"]
            assert sum(physical.values())==receipt["actual_OpenCV_total"]
        measured.append(method)
    assert len(panels)==1
    return dict(status="PASS",all_routes=15,new_measured_routes=5,reused_routes_exact=10,same_panel_frames=18,
        timed_calls_per_route=90,warmups_per_route=20,latency_moments_independent=True)

def audit_compute(doc,check):
    receipt=read(doc/"COMPUTE_RECEIPT.json");assert receipt["status"]=="COMPLETE_WITH_MEASURED_PHASES"
    actual_stages={p.name:read(p) for p in doc.glob("STAGE_*.json") if p.name!="STAGE_final_verify.json"}
    # The final audit runs after the producer ledger; its own conservative wall
    # receipt is added by publication without mutating that ledger.
    recorded={"STAGE_"+r["stage"]+".json":r for r in receipt["stages"]}
    assert recorded==actual_stages
    gpu=math.fsum(r["device_reserved_upper_seconds"] for r in recorded.values())
    cpu=math.fsum(r["wall_seconds"] for r in recorded.values() if r["device"]=="cpu")
    selection=read(doc/"SYNTHETIC_SELECTION.json");contract=read(doc/"INPUT_REUSE_CONTRACT.json")
    cpu+=contract["wall_seconds"]+selection["CPU_scoring_seconds"]+read(doc/'FEATURE_EXCLUSION_AUDIT.json')['wall_seconds']
    check.check(gpu,receipt["GPU_device_reserved_seconds"]);check.check(cpu,receipt["CPU_analysis_reserved_seconds"])
    assert gpu<receipt["GPU_budget_seconds"]==2700 and cpu<receipt["CPU_analysis_budget_seconds"]==3600
    start_epoch=datetime.fromisoformat(receipt['CPU_conservative_interval_start_UTC'].replace('Z','+00:00')).timestamp()
    upper=receipt['CPU_conservative_interval_end_epoch']-start_epoch+cpu
    check.check(upper,receipt['CPU_analysis_conservative_upper_seconds'],tolerance=1e-6)
    assert upper<3600 and receipt['CPU_conservative_interval_end_epoch']>=start_epoch
    costs=[read(doc/f"LATENCY_{m}.json") for m in ("RGB_ONLY","RGB_ONLY_SUBPIX","CVRANK_NATIVE","CVRANK_CAP1","CVRANK_THEN_SUBPIX")]
    benchmark_f=sum(c["counters"].get("F_calls",0) for row in costs for c in row["per_call"])
    benchmark_cv=sum(row["actual_OpenCV_total"] for row in costs)
    cv,rgb=read(doc/"CVRANK_COMPUTE_RECEIPT.json"),read(doc/"RGB_ONLY_EVALUATION_RECEIPT.json")
    expected=dict(synthetic_selection=selection["new_pose_F_calls"],REAL=cv["actual_F_calls"]+rgb["actual_F_calls"],benchmark=benchmark_f)
    assert expected==receipt["expected_F_calls"]==dict(synthetic_selection=1564,REAL=770,benchmark=550)
    assert receipt["total_new_F_calls"]==sum(expected.values())==2884
    opencv=dict(synthetic_selection=sum(selection["OpenCV_calls"].values()),REAL=cv["total_OpenCV_calls"]+rgb["total_OpenCV_calls"],benchmark=benchmark_cv)
    opencv["total"]=sum(opencv.values());assert opencv==receipt["actual_OpenCV_calls"]
    assert opencv["total"]<=receipt["expected_OpenCV_upper_bound"]==17304
    assert receipt["new_fits"]==2 and receipt["new_updates"]==6000 and receipt["updates_per_seed"]==3000
    for key in ("old_model_fits","old_model_new_inference","new_detector_accuracy_forwards","old_18_route_benchmark_repeated",
                "new_geometry_cache","new_synthetic_images","new_backbone_updates","new_seeds_beyond1_2","new_hyperparameter_search",
                "protected_or_Severe_REAL_access","manuscript_changes"):assert receipt[key]==0
    assert receipt["new_cost_processes"]==5 and receipt["new_cost_timed_calls"]==450 and receipt["new_cost_warmup_calls"]==100
    assert receipt["scientific_stage_failures"]==[r["stage"] for r in recorded.values() if r["returncode"]!=0]
    assert receipt["raw_RGB_and_GT_coordinates_public"] is False and receipt["existing_results_deleted"] is False
    return dict(status="PASS",GPU_seconds=gpu,CPU_seconds_before_final_audit=cpu,CPU_conservative_upper_before_final_audit=upper,GPU_budget_seconds=2700,CPU_budget_seconds=3600,
        final_audit_own_wall_accounted_at_publication=True,new_fits=2,new_updates=6000,new_F_calls=expected,actual_OpenCV_calls=opencv,
        preparation_failure_accounting=receipt["preparation_repair_record"])

def audit_figures(doc,root):
    receipt=read(doc/"FIGURE_RECEIPT.json");assert receipt["status"]=="COMPLETE"
    assert receipt["source_sha256"]==sha(root/"scripts/research/pallet_rgb_only_cvrank_20261011/figures_new.py")
    assert len(receipt["outputs"])==5 and receipt["all_routes"]==15 and receipt["independent_images"]==110
    assert receipt["full_CDF_ranges"] and receipt["CDF_denominator"]==220 and receipt["missing_poses_retained"]
    for r in receipt["inputs"]:assert sha(doc/r["path"])==r["sha256"]
    for r in receipt["outputs"]:assert sha(doc/r["path"])==r["sha256"] and (doc/r["path"]).stat().st_size==r["bytes"]
    for key in ("new_GPU_forwards","optimizer_updates","production_F_calls","new_RGB_reads"):assert receipt[key]==0
    assert receipt["public_native_or_reference_coordinates"] is False
    return dict(status="PASS",figures=5,all_scientific_inputs_and_PNG_hashes_match=True,
        all_routes_in_accuracy_and_full_CDF=True,actual_case_selection=receipt["case_selection"])

def audit_preservation(doc,prior,root,private,source):
    import subprocess
    receipt=read(doc/"SOURCE_PRESERVATION.json");assert receipt["status"]=="PASS"
    status=subprocess.check_output(['git','--no-optional-locks','-C',str(source),'status','--porcelain=v1','-z','--untracked-files=normal'])
    diff=subprocess.check_output(['git','--no-optional-locks','-C',str(source),'diff','--binary','HEAD'])
    assert status==(private/'original-status.bin').read_bytes() and diff==(private/'original-diff.bin').read_bytes()
    assert sha(private/'original-status.bin')==receipt['source_status_sha256'] and sha(private/'original-diff.bin')==receipt['source_diff_sha256']
    assert receipt['source_status_bytes_exact'] and receipt['source_diff_bytes_exact']
    manuscript=read(prior/'SOURCE_PRESERVATION.json')['tracked_manuscript_inventory']
    assert len(manuscript)==receipt['tracked_manuscript_files']==341
    for r in manuscript:assert sha(source/r['path'])==r['current_sha256'] and (source/r['path']).stat().st_size==r['current_bytes']
    previous=read(private/'prior_namespace_snapshot.json')['files'];assert len(previous)==receipt['existing_precision_namespace_files']==139
    for r in previous:assert sha(root/r['path'])==r['sha256'] and (root/r['path']).stat().st_size==r['bytes']
    assert receipt['immutable_input_bindings']==45 and receipt['original_checkpoints']==10 and receipt['all_input_bindings_and_checkpoints_exact']
    assert receipt['source_writes']==receipt['manuscript_writes']==receipt['old_result_writes']==0
    assert not receipt['untracked_content_equivalence_claimed']
    return dict(status='PASS',original_dirty_status_and_binary_diff_exact=True,tracked_manuscript_files_exact=341,
        prior_precision_namespace_files_exact=139,original_input_bindings=45,checkpoints=10,
        untracked_full_content_proof_claim=False,root_receipt_sha256=sha(doc/'SOURCE_PRESERVATION.json'))

def run(args):
    start=time.monotonic();doc=Path(args.public_dir).resolve();root=doc.parents[2];prior=Path(args.prior_public_dir).resolve()
    output=doc/args.output_name;assert not output.exists(),"Existing receipt is immutable"
    contract=read(doc/"INPUT_REUSE_CONTRACT.json");plan=read(doc/"PLAN_AND_CONTRACT.json")
    assert plan['status']=='LOCKED_BEFORE_FITS' and plan['training']['seeds']==[1,2]
    assert plan['training']['updates_each']==3000 and plan['training']['max_updates']==6000
    assert plan['input']['frames']==110 and plan['input']['id_sha256']==contract['id_sha256']
    assert plan['routes']==list(MAIN_METHODS) and plan['bootstrap']['family']==[list(p) for p in PRIMARY]
    assert plan['bootstrap']['Holm_size']==3 and plan['bootstrap']['draws']==10000 and plan['bootstrap']['seed']==20260917
    ids=sorted(contract["allowed_ids"]);assert len(ids)==110
    assert hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()==contract["id_sha256"]
    index=load([doc/"ROW_METRICS.jsonl.gz"]);assert len(index)==3300
    assert set(index)=={(m,fid,s) for m in MAIN_METHODS for fid in ids for s in (1,2)}
    metadata={}
    for r in index.values():
        assert r["inference_reference_inputs"] is False and r["strict_success"]==success(r)
        assert len(r["corner_valid"])==8
        key=r["session"],r["grade"].lower(),r["corner_valid"]
        if r["id"] in metadata:assert metadata[r["id"]]==key
        metadata[r["id"]]=key
        if r["available"]:
            assert 0<=val(r,"IoU3D")<=1 and val(r,"ADDsym_m")>=0
        yaw=r["pose"].get("yaw_deg") if r["pose"] else None
        assert bool(r["wd_confusion"])==bool(r["available"] and val(r,"Rdeg")>45 and yaw is not None and abs(yaw)>=60)
    grades=dict(Counter(metadata[fid][1] for fid in ids));assert grades==contract["grade_counts"]==dict(clean=68,moderate=42)
    axis_eligible={fid:index['BASE',fid,1]['axis_reference_evaluable'] for fid in ids}
    assert sum(axis_eligible.values())==109 and [fid for fid in ids if not axis_eligible[fid]]==['eval_outside:1778653367706938112']
    for row in index.values():
        assert row['axis_reference_evaluable']==axis_eligible[row['id']]
        if not axis_eligible[row['id']]:assert row['axis_wrong'] is None
    sessions=[metadata[fid][0] for fid in ids];names=sorted(set(sessions));assert len(names)==9
    rawdraw=np.load(doc/"BOOTSTRAP_DRAWS.npy",allow_pickle=False)
    assert rawdraw.shape==(10000,9) and np.array_equal(rawdraw,np.random.default_rng(20260917).integers(0,9,size=(10000,9)))
    assert sha(doc/"BOOTSTRAP_DRAWS.npy")==sha(prior/"BOOTSTRAP_DRAWS.npy")==read(doc/"BOOTSTRAP_LOCK.json")["sha256"]
    draw=Draws(ids,sessions,rawdraw);check=Compare();metrics=read(doc/"METRICS.json");paired=read(doc/"PAIRED_RESULTS.json")
    assert metrics["frames"]==110 and metrics["prediction_rows"]==3300 and metrics["sessions"]==9
    assert set(metrics["groups"])==set(MAIN_METHODS)
    for method,packets in metrics["groups"].items():
        assert set(packets)=={"common_seed12"};p=packets["common_seed12"];assert p["seeds"]==[1,2]
        audit_group(p,method,index,ids,draw,check)
        axis=p["axis_wrong"];assert axis["reference_eligible_frames"]==109 and axis["reference_eligible_prediction_rows"]==218
        assert axis["missing_selected_hyp_rows"]==218-axis["denominator"] and axis["excluded_reference_UNCLEAR_frames"]==1
    assert paired["primary_declared_family_size"]==3 and paired["mechanism"]==[]
    primary=audit_family(paired["primary"],PRIMARY,index,ids,draw,check)
    oldindex=load([prior/"ROW_METRICS.jsonl.gz"]);reused=0
    for (m,fid,seed),row in index.items():
        if m in NEW_METHODS:continue
        old=oldindex[m,fid,seed]
        for key in ("available","pose","strict_success","axis_wrong","wd_confusion","corner_errors_px","corner_valid","selected_hyp"):
            assert row[key]==old[key],(m,fid,seed,key)
        reused+=1
    assert reused==2200
    transitions=audit_transitions(doc,index,ids,check);table=audit_table(doc,metrics,check)
    feature=read(doc/"FEATURE_EXCLUSION_AUDIT.json");assert feature["status"]=="PASS" and len(feature["execution_probes"])==4
    assert feature["plan_sha256"]==sha(doc/"PLAN_AND_CONTRACT.json")
    train=read(doc/"RGB_ONLY_TRAIN_RECEIPTS.json");assert train["status"]=="DONE" and train["fits"]==2 and train["actual_updates"]==6000 and train["updates_per_seed"]==3000
    owners=dict(new_worktree=root,historical_source=Path(args.source_root),prior_private=Path(args.prior_private_dir),private_new=Path(args.private_dir))
    bindings=audit_bindings(contract,owners)
    private_predictions=audit_private_predictions(Path(args.private_dir),ids)
    latency=audit_latency(doc,prior,ids,check);compute=audit_compute(doc,check)
    preservation=audit_preservation(doc,prior,root,Path(args.private_dir),Path(args.source_root))
    figures=audit_figures(doc,root)
    public=audit_public(doc,root,set(args.allow_pending)|{args.output_name,"STAGE_final_verify.json"})
    receipt=read(doc/"STATISTICS_RECEIPT.json");assert receipt["rows"]==3300 and receipt["groups"]==15 and receipt["all_pair_rows"]==3520
    result=dict(status="PASS" if public["status"]=="PASS" else "ARITHMETIC_PASS_DISCLOSURE_FAIL",
        frames=110,sessions=9,grades=grades,prediction_rows=3300,reused_prediction_rows_exact=2200,new_prediction_rows=1100,
        Severe_scored=0,scalar_comparisons=check.count,maximum_scalar_absolute_difference=check.maxdiff,
        arithmetic="Independent fsum/sample variance/sorted linear quantiles; separately accumulated identical session draws; exact integer centered-bootstrap ties",
        primary_independent=primary,transitions=transitions,accuracy_table=table,feature_exclusion_receipt_sha256=sha(doc/"FEATURE_EXCLUSION_AUDIT.json"),
        matched_training=dict(status="PASS",fits=2,updates=6000,steps_each=3000),
        public_disclosure_and_Markdown=public,source_bindings=bindings,private_CV_prediction_contract=private_predictions,
        runtime_costs=latency,global_compute=compute,original_preservation=preservation,figures=figures,
        source_rows_sha256=sha(doc/"ROW_METRICS.jsonl.gz"),producer_statistics_sha256=sha(doc/"METRICS.json"),
        verifier_sha256=sha(__file__),auditor_GPU_calls=0,auditor_model_forwards=0,auditor_PnP=0,auditor_fits=0,
        full_experiment_rerun_claim=False,wall_seconds=time.monotonic()-start)
    assert compute["CPU_conservative_upper_before_final_audit"]+result["wall_seconds"]<3600
    with output.open("x",encoding="utf-8") as handle:json.dump(result,handle,indent=2,ensure_ascii=False,allow_nan=False);handle.write("\n")
    print(json.dumps(dict(status=result["status"],scalar_comparisons=check.count,maximum_difference=check.maxdiff,public_issues=public["issues"]),ensure_ascii=False))
    assert result["status"]=="PASS",public["issues"]

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--public-dir",required=True);parser.add_argument("--prior-public-dir",required=True)
    parser.add_argument("--private-dir",required=True);parser.add_argument("--source-root",required=True);parser.add_argument("--prior-private-dir",required=True)
    parser.add_argument("--allow-pending",action="append",default=[]);parser.add_argument("--output-name",default="INDEPENDENT_VERIFICATION.json")
    run(parser.parse_args())
