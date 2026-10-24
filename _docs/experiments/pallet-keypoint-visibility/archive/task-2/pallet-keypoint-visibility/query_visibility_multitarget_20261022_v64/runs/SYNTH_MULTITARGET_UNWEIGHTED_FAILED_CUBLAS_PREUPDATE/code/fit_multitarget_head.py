"""Source-only fixed-budget six-joint training; no evaluation-file imports."""
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F

from query_multitarget_head import (HeadConfig, JOINT_PAIRS, QueryMultiTargetHead,
                                    classes_to_joint, joint_to_classes, load_model,
                                    marginals, parameter_count, save_checkpoint)


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open("rb")as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):digest.update(block)
    return digest.hexdigest()


def write_json(path,data):
    Path(path).write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+"\n")


def load_cache(path):
    with np.load(path,allow_pickle=False)as cache:data={key:cache[key]for key in cache.files}
    n=len(data["x"])
    expected={"x":(n,65,9,9),"rgb_native":(n,75,9,9),"sam":(n,2,9,9),"context":(n,64),
              "box_grid":(n,4,9,9),"dense_valid":(n,9,9),"source":(n,),
              "group_id":(n,),"sample_id":(n,),"image_sha256":(n,)}
    for key,shape in expected.items():
        if key not in data or data[key].shape!=shape:raise ValueError(f"{key} must have shape {shape}")
    for key in ("x","rgb_native","sam","context","box_grid"):
        if not np.isfinite(data[key]).all():raise ValueError(f"nonfinite {key}")
    if data["rgb_native"].dtype!=np.uint8:raise ValueError("rgb_native must be uint8 RGB patch channels")
    if np.abs(data["sam"]).max()>4.000001 or np.abs(data["box_grid"]).max()>4.000001:
        raise ValueError("SAM and box input channels must be clipped to [-4,4]")
    data["dense_valid"]=data["dense_valid"].astype(bool)
    valid=data["dense_valid"]
    if "classes"in data:
        joint=classes_to_joint(data["classes"],valid)
        if "joint"in data and not np.array_equal(joint[valid],data["joint"][valid]):
            raise ValueError("provided joint and classes disagree")
        data["joint"]=joint
    elif "joint"in data:
        if data["joint"].shape!=(n,9,9)or not np.isin(data["joint"][valid],np.arange(6)).all():
            raise ValueError("valid joint labels must be 0..5 in N,9,9")
        data["joint"]=data["joint"].astype(np.int64)
    else:raise ValueError("classes or joint supervision is required")
    aliases={"SYNTHETIC":"SYNTH","SYNTH":"SYNTH","YCB":"YCB","YCBV":"YCB","YCB-V":"YCB"}
    data["source"]=np.asarray([aliases.get(str(s).upper(),str(s))for s in data["source"]])
    if not set(data["source"])<={"SYNTH","YCB"}:raise ValueError("only SYNTH/YCB sources are permitted")
    evidence=np.asarray(data.get("evidence_weight",np.ones((n,9,9),np.float32)))
    if evidence.shape==(n,2,9,9):evidence=evidence.min(axis=1)
    if evidence.shape!=(n,9,9)or not np.isfinite(evidence).all()or(evidence<0).any():
        raise ValueError("evidence_weight must be nonnegative finite N,9,9 (or N,2,9,9, minimum aggregated)")
    data["evidence_weight"]=evidence.astype(np.float32)
    if not valid[:,4,4].any():raise ValueError("no valid scalar center labels")
    return data


def source_balanced_loss(logits,target,valid,weights,source_codes):
    """Mean of within-source weighted losses, not a pooled-domain mean."""
    ce=F.cross_entropy(logits,target.clamp_min(0),reduction="none")
    losses=[]
    for code in torch.unique(source_codes):
        row=source_codes==code
        w=weights[row]*valid[row].float()
        if w.sum()>0:losses.append((ce[row]*w).sum()/w.sum())
    if not losses:raise ValueError("training batch has no positively weighted valid label")
    return torch.stack(losses).mean()


def confusion(truth,prediction,count):
    matrix=np.zeros((count,count),np.int64)
    np.add.at(matrix,(truth,prediction),1)
    per_class=[]
    for state in range(count):
        tp=int(matrix[state,state]);actual=int(matrix[state].sum());predicted=int(matrix[:,state].sum())
        per_class.append({"code":state,"N":actual,"precision":tp/predicted if predicted else None,
                          "recall":tp/actual if actual else None})
    recalls=[row["recall"]for row in per_class if row["recall"]is not None]
    return {"confusion":matrix.tolist(),"per_class":per_class,"balanced_accuracy":float(np.mean(recalls))}


def abstaining_confusion(truth,prediction,count):
    matrix=np.zeros((count,count+1),np.int64)
    safe=np.where(prediction<0,count,prediction)
    np.add.at(matrix,(truth,safe),1)
    rows=[]
    for state in range(count):
        tp=int(matrix[state,state]);actual=int(matrix[state].sum());decided=int(matrix[state,:count].sum());predicted=int(matrix[:,state].sum())
        rows.append({"code":state,"N":actual,"decided_GT":decided,
                     "precision":tp/predicted if predicted else None,"recall":tp/actual if actual else None,
                     "recall_all_GT":tp/actual if actual else None,"recall_decided_GT":tp/decided if decided else None})
    return {"confusion_columns":list(range(count))+["U"],"confusion":matrix.tolist(),"per_class":rows,
            "coverage":float((prediction>=0).mean()),"abstention_rate":float((prediction<0).mean()),
            "balanced_accuracy_all_GT":float(np.mean([row["recall_all_GT"]for row in rows if row["N"]])),
            "balanced_accuracy_decided_GT":float(np.mean([row["recall_decided_GT"]for row in rows if row["decided_GT"]]))if(prediction>=0).any()else None}


def summarize(y,p):
    from sklearn.metrics import average_precision_score,roc_auc_score
    states=joint_to_classes(y)
    probabilities=marginals(p)
    paired=joint_to_classes(p.argmax(1))
    hard=np.full(states.shape,2,np.int64)
    hard[probabilities["full_probability"]>=.5]=1
    hard[probabilities["visible_probability"]>=.5]=0
    pair_lookup={pair:index for index,pair in enumerate(JOINT_PAIRS)}
    hard_joint=np.asarray([pair_lookup[tuple(pair)]for pair in hard])
    confident_joint=np.where(p.max(1)>=.5,p.argmax(1),-1)
    confident_states=paired.copy();confident_states[confident_joint<0]=-1
    result={"N":len(y),"joint_NLL":float(-np.log(np.clip(p[np.arange(len(y)),y],1e-12,1)).mean()),
            "logical_support_decoder":"pVisible>=.5 else pFull>=.5; not a confidence-qualified external label",
            "joint":confusion(y,hard_joint,6),"joint_MAP_diagnostic":confusion(y,p.argmax(1),6),
            "confident_joint_MAP":abstaining_confusion(y,confident_joint,6),"scales":{}}
    for scale,name in enumerate(("exact","radius2")):
        state=states[:,scale];sp=probabilities["state_probability"][:,scale]
        report={"primary_threshold_decode":confusion(state,hard[:,scale],3),
                "joint_MAP_decode_diagnostic":confusion(state,paired[:,scale],3),
                "confidence_mechanism_joint_MAP_or_U":abstaining_confusion(state,confident_states[:,scale],3)}
        for task,target,pp in (("visible",state==0,probabilities["visible_probability"][:,scale]),
                               ("full",state!=2,probabilities["full_probability"][:,scale])):
            report[task]={**confusion(target.astype(int),(pp>=.5).astype(int),2),
                          "AP":float(average_precision_score(target,pp))if target.any()else None,
                          "AUC":float(roc_auc_score(target,pp))if len(np.unique(target))==2 else None,
                          "BCE":float(-(target*np.log(np.clip(pp,1e-12,1))+(~target)*np.log(np.clip(1-pp,1e-12,1))).mean()),
                          "threshold":.5}
        result["scales"][name]=report
    return result


def fit(args):
    out=Path(args.out)
    if(out/"RUN_STARTED.json").exists():raise FileExistsError("preserve prior training run")
    train,val=load_cache(args.train),load_cache(args.val)
    denied=set(json.loads(Path(args.eval_denylist).read_text())["image_sha256"])
    if not denied:raise ValueError("nonempty evaluation exclusion list required")
    exclusion={}
    for split,data in (("TRAIN",train),("VAL",val)):
        overlap=set(data["image_sha256"].astype(str))&denied
        if overlap:raise ValueError(f"evaluation image found in {split}")
        exclusion[split]={"queries":len(data["x"]),"denied_image_overlap":0}
    split_checks={}
    for key in ("group_id","sample_id","image_sha256"):
        if set(train[key].astype(str))&set(val[key].astype(str)):raise ValueError(f"TRAIN/VAL {key} overlap")
        split_checks[key]=0
    torch.set_num_threads(args.threads);random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
    config=HeadConfig();model=QueryMultiTargetHead(config).to(args.device)
    parameters=parameter_count(model)
    if parameters>100000:raise ValueError("parameter budget exceeded")
    out.mkdir(parents=True,exist_ok=True)
    code_paths=[Path(__file__),Path(__file__).with_name("query_multitarget_head.py")]
    audit={"arguments":vars(args),"head_config":asdict(config),"parameters":parameters,
           "joint_pairs":JOINT_PAIRS,"target":"exact/radius2 visible and full instance support",
           "loss":{"center_joint_CE":1.,"dense_joint_CE":args.dense_loss_weight,"source_balanced":True},
           "selection":"lowest macro(source unweighted valid-center joint NLL)",
           "class_weights":[1.]*6,"evidence_weights_used_for_training":args.use_evidence_weights,
           "evidence_weights_used_for_selection":False,"threshold":.5,
           "paired_hard_state_decoder":"per-scale V if pVisible>=.5 else E if pFull>=.5 else O; joint MAP diagnostic only",
           "logical_decoder_scope":"presence/full logical support, not confidence-qualified external classification",
           "confidence_mechanism_decoder":"joint MAP only if maxjoint>=.5, otherwise U; threshold fixed before evaluation",
           "train_sha256":sha256(args.train),"val_sha256":sha256(args.val),
           "eval_denylist_sha256":sha256(args.eval_denylist),"excluded_images":len(denied),
           "exclusion_checks":exclusion,"split_overlap_checks":split_checks,
           "eval_data_used":False,"PnP_used":False,"code_sha256":{p.name:sha256(p)for p in code_paths}}
    if args.provenance:audit["source_provenance"]={"path":args.provenance,"sha256":sha256(args.provenance),"content":json.loads(Path(args.provenance).read_text())}
    if args.protocol:audit["protocol"]={"path":args.protocol,"sha256":sha256(args.protocol)}
    write_json(out/"RUN_STARTED.json",audit);(out/"code").mkdir(exist_ok=False)
    for path in code_paths:shutil.copy2(path,out/"code"/path.name)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)
    train_indices=np.flatnonzero(train["dense_valid"][:,4,4]);val_indices=np.flatnonzero(val["dense_valid"][:,4,4])
    source_map={source:index for index,source in enumerate(sorted(set(train["source"])))}
    source_codes=np.array([source_map[s]for s in train["source"]])
    def tensor(data,key,ix,dtype=torch.float32):return torch.as_tensor(np.asarray(data[key][ix]),device=args.device,dtype=dtype)
    def inputs(data,ix):return {"x":tensor(data,"x",ix),"rgb_native":tensor(data,"rgb_native",ix)/255,
                              "sam":tensor(data,"sam",ix),"context":tensor(data,"context",ix),"box_grid":tensor(data,"box_grid",ix)}
    def predict(data,indices):
        model.eval();result=[]
        with torch.inference_mode():
            for start in range(0,len(indices),args.batch_size):result.append(model(**inputs(data,indices[start:start+args.batch_size])).cpu().numpy())
        return np.concatenate(result)
    update,best,best_update,history,start=0,float("inf"),None,[],time.monotonic()
    losses=[];epochs=math.ceil(args.max_updates/math.ceil(len(train_indices)/args.batch_size))
    for epoch in range(1,epochs+1):
        model.train();permutation=np.random.permutation(train_indices)
        for offset in range(0,len(permutation),args.batch_size):
            ix=permutation[offset:offset+args.batch_size]
            center,dense=model(**inputs(train,ix),return_dense=True)
            y=tensor(train,"joint",ix,torch.long);valid=tensor(train,"dense_valid",ix,torch.bool)
            weight=tensor(train,"evidence_weight",ix)if args.use_evidence_weights else torch.ones_like(y,dtype=torch.float32)
            codes=torch.as_tensor(source_codes[ix],device=args.device)
            scalar_loss=source_balanced_loss(center,y[:,4,4],valid[:,4,4],weight[:,4,4],codes)
            dense_loss=source_balanced_loss(dense,y,valid,weight,codes)
            loss=scalar_loss+args.dense_loss_weight*dense_loss
            if not torch.isfinite(loss):raise FloatingPointError("nonfinite loss")
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();update+=1;losses.append(float(loss.detach().cpu()))
            if update%args.val_every_updates==0 or update==args.max_updates:
                z=predict(val,val_indices).astype(np.float64);logp=z-np.logaddexp.reduce(z,axis=1)[:,None]
                yv=val["joint"][val_indices,4,4];nll=-logp[np.arange(len(yv)),yv]
                by_source={s:float(nll[val["source"][val_indices]==s].mean())for s in np.unique(val["source"][val_indices])}
                selection=float(np.mean(list(by_source.values())));improved=selection<best
                if improved:
                    best,best_update=selection,update
                    save_checkpoint(out/"BEST_MULTITARGET_HEAD.pt",model,updates=update,epoch=epoch,
                                    selection_joint_NLL=selection,selection_NLL_by_source=by_source,threshold=.5,
                                    train_sha256=audit["train_sha256"],val_sha256=audit["val_sha256"])
                row={"epoch":epoch,"updates":update,"train_loss_mean":float(np.mean(losses)),"VAL_joint_NLL":selection,
                     "VAL_joint_NLL_by_source":json.dumps(by_source,sort_keys=True),"saved_best":improved,"elapsed_seconds":time.monotonic()-start}
                history.append(row);write_json(out/"PROGRESS.json",row);print(json.dumps(row),flush=True);losses=[];model.train()
            if update>=args.max_updates:break
        if update>=args.max_updates:break
    if update!=args.max_updates:raise AssertionError("fixed update budget incomplete")
    model,payload=load_model(out/"BEST_MULTITARGET_HEAD.pt",args.device);reports=[];sam_reports=[]
    priors={"ALL":np.bincount(train["joint"][train_indices,4,4],minlength=6).astype(float)}
    for source in np.unique(train["source"]):priors[source]=np.bincount(train["joint"][train_indices[train["source"][train_indices]==source],4,4],minlength=6).astype(float)
    priors={key:value/value.sum()for key,value in priors.items()}
    for split,data,indices in (("TRAIN",train,train_indices),("VAL",val,val_indices)):
        z=predict(data,indices).astype(np.float64);p=np.exp(z-np.logaddexp.reduce(z,axis=1)[:,None]);y=data["joint"][indices,4,4]
        mm=marginals(p);np.savez_compressed(out/f"{split}_SOURCE_OUTPUTS.npz",index=indices,joint_logits=z,truth_joint=y,
                                           truth_classes=joint_to_classes(y),source=data["source"][indices],**mm)
        for source in ("ALL",*np.unique(data["source"])):
            mask=np.ones(len(indices),bool)if source=="ALL"else data["source"][indices]==source
            prior=priors.get(source,priors["ALL"]);always=np.zeros((int(mask.sum()),6));always[:,5]=1
            always_visible=np.zeros_like(always);always_visible[:,0]=1
            for arm,prediction in (("MULTITARGET_HEAD",p[mask]),("TRAIN_JOINT_PRIOR",np.repeat(prior[None],int(mask.sum()),0)),("ALWAYS_OUTSIDE",always),("ALWAYS_VISIBLE",always_visible)):
                reports.append({"split":split,"source":str(source),"arm":arm,**summarize(y[mask],prediction)})
            truth=joint_to_classes(y[mask])
            for scale,name in enumerate(("exact","radius2")):
                raw=8*data["sam"][indices[mask],scale,4,4].astype(float);prob=1/(1+np.exp(-np.clip(raw,-80,80)))
                target=truth[:,scale]==0
                from sklearn.metrics import average_precision_score,roc_auc_score
                sam_reports.append({"split":split,"source":str(source),"scale":name,"arm":"FROZEN_RAW_SAM_VISIBLE_ONLY",
                                    **confusion(target.astype(int),(raw>0).astype(int),2),
                                    "AP":float(average_precision_score(target,prob))if target.any()else None,
                                    "AUC":float(roc_auc_score(target,prob))if len(np.unique(target))==2 else None,
                                    "threshold":"raw logit > 0","amodal_full_prediction":False})
    with(out/"TRAINING_HISTORY.csv").open("w",newline="")as stream:
        writer=csv.DictWriter(stream,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history)
    write_json(out/"SOURCE_METRICS.json",reports)
    write_json(out/"RAW_SAM_SOURCE_METRICS.json",sam_reports)
    zero_recall=[]
    for report in reports:
        if report["split"]!="VAL"or report["source"]=="ALL"or report["arm"]!="MULTITARGET_HEAD":continue
        for scale,metric in report["scales"].items():
            for state in metric["primary_threshold_decode"]["per_class"]:
                if state["N"]>0 and state["recall"]==0:
                    zero_recall.append({"source":report["source"],"scale":scale,"state":state["code"],"N":state["N"]})
    gate={"PASS":not zero_recall,"HOLD":bool(zero_recall),"supported_zero_recall_states":zero_recall,
          "scope":"source mechanism only; checkpoint remains lowest-source-NLL selection",
          "private_evaluation_used":False,"threshold_changed":False}
    write_json(out/"SOURCE_MECHANISM_GATE.json",gate)
    complete={**audit,"optimizer_updates":update,"best_update":best_update,"best_source_VAL_joint_NLL":best,
              "best_VAL_NLL_by_source":payload["selection_NLL_by_source"],"checkpoint_sha256":sha256(out/"BEST_MULTITARGET_HEAD.pt"),
              "TRAIN_joint_priors":{k:v.tolist()for k,v in priors.items()},"seconds":time.monotonic()-start,"completed":True}
    complete["source_mechanism_gate"]=gate
    write_json(out/"MODEL_FREEZE.json",complete);print(json.dumps({"completed":True,"updates":update,"best_update":best_update}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("train","val","out","eval-denylist"):parser.add_argument("--"+key,required=True)
    parser.add_argument("--provenance");parser.add_argument("--protocol")
    parser.add_argument("--max-updates",type=int,default=2000);parser.add_argument("--val-every-updates",type=int,default=100)
    parser.add_argument("--batch-size",type=int,default=256);parser.add_argument("--learning-rate",type=float,default=.001)
    parser.add_argument("--weight-decay",type=float,default=.0001);parser.add_argument("--dense-loss-weight",type=float,default=.25)
    parser.add_argument("--use-evidence-weights",action="store_true");parser.add_argument("--seed",type=int,default=20261022)
    parser.add_argument("--threads",type=int,default=1);parser.add_argument("--device",default="cpu")
    args=parser.parse_args()
    if min(args.max_updates,args.val_every_updates,args.batch_size,args.learning_rate)<=0 or args.dense_loss_weight<0:parser.error("positive budget, batch, learning rate required")
    fit(args)


if __name__=="__main__":main()
