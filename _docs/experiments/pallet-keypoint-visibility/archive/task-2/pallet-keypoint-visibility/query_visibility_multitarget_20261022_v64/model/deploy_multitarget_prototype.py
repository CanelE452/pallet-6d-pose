"""V64 inference prototype: new RGB+q8 or frozen precomputed input features.

Generic YOLO/MobileSAM and pure geometry runtime come from the SHA-bound v63
deployment bundle. No v63 learned head is instantiated. A V64 candidate must
be explicit; this prototype does not select a model or read evaluation labels.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bound_module(path,expected,role):
    if sha(path)!=expected:raise ValueError(f"bound {role} code changed")
    name="_v64_deploy_"+role+"_"+expected[:16]
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec)
        sys.modules[name]=value;spec.loader.exec_module(value)
    return sys.modules[name]


def native_inputs(rgb,points,side,box,logits):
    """Input-only native75 RGB, relative box and exact/r2 SAM grids."""
    import cv2
    q=np.asarray(points,float);h,w=rgb.shape[:2]
    offsets=(np.arange(9)-4)*side/9;dx,dy=np.meshgrid(offsets,offsets)
    xy=q[:,None,None]+np.stack((dx,dy),-1)[None]
    finite=np.isfinite(xy).all(-1);pixel=np.floor(np.where(np.isfinite(xy),xy,-1e8)+.5).astype(np.int64)
    valid=finite&(xy[...,0]>=0)&(xy[...,0]<w)&(xy[...,1]>=0)&(xy[...,1]<h)
    valid&=(pixel[...,0]>=0)&(pixel[...,0]<w)&(pixel[...,1]>=0)&(pixel[...,1]<h)
    patches=[]
    for oy in range(-2,3):
        for ox in range(-2,3):
            px,py=pixel[...,0]+ox,pixel[...,1]+oy
            inside=(px>=0)&(px<w)&(py>=0)&(py<h)
            value=rgb[np.clip(py,0,h-1),np.clip(px,0,w-1)].copy();value[~inside]=0
            patches.append(value.transpose(0,3,1,2))
    rgb_native=np.stack(patches,axis=2).reshape(8,75,9,9)
    x0,y0,x1,y1=box;bw,bh=x1-x0,y1-y0
    if min(bw,bh)<2:raise ValueError("nondegenerate input prompt box required")
    box_grid=np.stack((2*(xy[...,0]-x0)/bw-1,2*(xy[...,1]-y0)/bh-1,
                       np.full(xy.shape[:-1],np.log2(side/bw)),np.full(xy.shape[:-1],np.log2(side/bh))),axis=1)
    box_grid=np.nan_to_num(np.clip(box_grid,-4,4)).astype(np.float32)
    near=cv2.dilate(np.asarray(logits,np.float32),np.ones((5,5),np.uint8))
    grids=[]
    for field in (logits,near):
        sampled=field[np.clip(pixel[...,1],0,h-1),np.clip(pixel[...,0],0,w-1)]
        grids.append(np.where(valid,np.clip(sampled/8,-4,4),0).astype(np.float16))
    return rgb_native,box_grid,np.stack(grids,axis=1),valid


class QueryMultiTargetVisibility:
    def __init__(self,candidate,manifest=None,device="cpu"):
        self.manifest_path=Path(manifest or Path(__file__).with_name("DEPLOYMENT_PROTOTYPE.json"))
        self.manifest=json.loads(self.manifest_path.read_text());self.device=device
        if sha(__file__)!=self.manifest["entrypoint"]["sha256"]:raise ValueError("prototype entrypoint changed")
        os.environ.setdefault("YOLO_CONFIG_DIR",str(Path(__file__).parent/"deployment_runtime/ultralytics"))
        binding=self.manifest["generic_runtime_bundle"]
        if sha(binding["path"])!=binding["sha256"]:raise ValueError("generic runtime bundle changed")
        self.bundle=json.loads(Path(binding["path"]).read_text())
        entry=self.bundle["entrypoint"]
        self.generic=bound_module(entry["path"],entry["sha256"],"generic_frontend")
        models=self.manifest["candidates"]
        if candidate not in models:raise ValueError("explicit frozen source candidate is required")
        selected=models[candidate];self.candidate=candidate
        for role in ("checkpoint","freeze","model_code"):
            if sha(selected[role]["path"])!=selected[role]["sha256"]:raise ValueError(f"frozen {role} changed")
        freeze=json.loads(Path(selected["freeze"]["path"]).read_text())
        if freeze["checkpoint_sha256"]!=selected["checkpoint"]["sha256"]or freeze["eval_data_used"]or freeze["PnP_used"]:
            raise ValueError("source-only model freeze mismatch")
        self.head=bound_module(selected["model_code"]["path"],selected["model_code"]["sha256"],"head")
        self.model,self.payload=self.head.load_model(selected["checkpoint"]["path"],device)
        self.model.eval().requires_grad_(False)
        rule=self.manifest["support_contract"]
        self.rules=bound_module(rule["path"],rule["sha256"],"support_contract")
        self.backbone=self.sam_predictor=self.sam_model=None;self._sam_image_key=None
        torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False

    def _load_backbone(self):return self.generic.QueryVisibility._load_backbone(self)
    def _load_sam(self):return self.generic.QueryVisibility._load_sam(self)
    def _geometry(self,pose):return self.generic.QueryVisibility._geometry(self,pose)
    def close(self):return self.generic.QueryVisibility.close(self)

    def empty(self,points,raw_hw,reason,pose=None):
        q,pixel,valid,box=self.generic.query_metadata(points,raw_hw)
        flags,geometry_status=self._geometry(pose)
        rows=[]
        for k in range(8):
            rows.append(dict(native_corner=k,query_xy=[float(v)if np.isfinite(v)else None for v in q[k]],
                             pixel_xy=pixel[k].tolist()if valid[k]else None,status="UNKNOWN",
                             reason=reason if valid[k]else"INVALID_QUERY",joint_probability=None,
                             exact=None,radius2=None,mechanism=dict(status="UNKNOWN",joint=None,classes=[None,None]),
                             logical_support=None,nearby_support_gap=None,raw_SAM_exact_logit=None,
                             raw_SAM_radius2_logit=None,self_hidden_geometry_proxy=flags[k],geometry_status=geometry_status))
        return dict(schema="V64_FROZEN_MULTITARGET_PROTOTYPE",candidate=self.candidate,
                    prompt_box_xyxy=box.tolist()if box is not None else None,raw_hw=list(raw_hw),queries=rows,
                    thresholds=dict(visible=.5,full=.5,joint_confidence=.5),PnP_calls=0,
                    geometry_changes_learned_probabilities=False)

    def predict_precomputed(self,*,points,raw_hw,x,rgb_native,sam,box_grid,context=None,
                            full_SAM_logits=None,saved_initial_pose=None):
        result=self.empty(points,raw_hw,"MISSING_OR_DEGENERATE_QUERY_BOX",saved_initial_pose)
        if result["prompt_box_xyxy"]is None:return result
        q,pixel,valid,_=self.generic.query_metadata(points,raw_hw)
        x,rgb_native,sam,box_grid=map(np.asarray,(x,rgb_native,sam,box_grid))
        for value,shape in ((x,(8,65,9,9)),(rgb_native,(8,75,9,9)),(sam,(8,2,9,9)),(box_grid,(8,4,9,9))):
            if value.shape!=shape or not np.isfinite(value).all():raise ValueError(f"finite input shape {shape} required")
        if rgb_native.dtype!=np.uint8 or np.abs(sam).max()>4.000001 or np.abs(box_grid).max()>4.000001:
            raise ValueError("uint8 native RGB and normalized clipped SAM/box grids required")
        roi_valid=x[:,64].astype(bool).any(axis=(1,2))
        for k in np.flatnonzero(valid&~roi_valid):result["queries"][k]["reason"]="INVALID_OR_EMPTY_ROI"
        valid&=roi_valid;ix=np.flatnonzero(valid)
        if not len(ix):return result
        if context is None:
            mask=x[:,64].astype(np.float32)
            context=(x[:,:64].astype(np.float32)*mask[:,None]).sum(axis=(0,2,3))/max(float(mask.sum()),1)
        context=np.asarray(context)
        if context.shape==(64,):context=np.repeat(context[None],8,0)
        if context.shape!=(8,64)or not np.isfinite(context).all():raise ValueError("finite shared target64 context required")
        tensor=lambda a:torch.as_tensor(np.asarray(a),device=self.device).float()
        with torch.inference_mode():
            z=self.model(tensor(x[ix]),tensor(rgb_native[ix])/255,tensor(sam[ix]),tensor(context[ix]),tensor(box_grid[ix])).cpu().numpy().astype(float)
        p=np.exp(z-np.logaddexp.reduce(z,axis=1)[:,None]);statep=self.rules.joint_marginals(p)
        logical=self.rules.hierarchical_states(p);mechanism=self.rules.confident_joint_map(p)
        if full_SAM_logits is not None:
            full_SAM_logits=np.asarray(full_SAM_logits)
            if full_SAM_logits.shape!=tuple(raw_hw)or not np.isfinite(full_SAM_logits).all():raise ValueError("finite full-resolution SAM logit map required")
        for j,k in enumerate(ix):
            row=result["queries"][k];scales=[]
            for scale in range(2):
                values=statep[j,scale]
                scales.append(dict(p_visible=float(values[0]),p_external_support=float(values[1]),
                                   p_outside_full=float(values[2]),p_full=float(values[0]+values[1])))
            accepted=bool(mechanism["accepted"][j]);pair=mechanism["classes"][j]
            row.update(status="OK",reason=None,joint_probability=p[j].tolist(),exact=scales[0],radius2=scales[1],
                       mechanism=dict(status="ACCEPTED"if accepted else"UNKNOWN",joint=int(mechanism["joint"][j])if accepted else None,
                                      classes=pair.tolist()if accepted else[None,None],max_joint_probability=float(p[j].max()),
                                      interpretation="source-trained support class, not calibrated semantic corner visibility"),
                       logical_support=dict(classes=logical[j].tolist(),interpretation="fixed presence/full diagnostic, not confidence-qualified external classification"),
                       nearby_support_gap=dict(outside_exact_visible_near=float(p[j,3]),outside_exact_hidden_support_near=float(p[j,4]),
                                               full_support_only_near=float(p[j,3]+p[j,4]),location_error_proof=False),
                       raw_SAM_exact_logit=float(8*sam[k,0,4,4]),raw_SAM_radius2_logit=float(8*sam[k,1,4,4]),
                       raw_SAM_source="CLIPPED_NORMALIZED_NATIVE_CACHE")
            if full_SAM_logits is not None:
                px,py=pixel[k];h,w=raw_hw
                row.update(raw_SAM_exact_logit=float(full_SAM_logits[py,px]),
                           raw_SAM_radius2_logit=float(full_SAM_logits[max(0,py-2):min(h,py+3),max(0,px-2):min(w,px+3)].max()),
                           raw_SAM_source="FULL_RESOLUTION_SAME_SELECTED_MASK")
        return result

    def predict_rgb(self,rgb,points,saved_initial_pose=None):
        rgb=np.asarray(rgb)
        if rgb.ndim!=3 or rgb.shape[2]!=3 or rgb.dtype!=np.uint8:raise ValueError("uint8 H,W,3 RGB required")
        hw=rgb.shape[:2];q,_,_,box=self.generic.query_metadata(points,hw)
        if box is None:return self.empty(points,hw,"MISSING_OR_DEGENERATE_QUERY_BOX",saved_initial_pose)
        self._load_backbone();self._load_sam()
        safe=q.copy();safe[~np.isfinite(safe).all(1)]=[-1,-1]
        captured=self.backbone.predict(np.ascontiguousarray(rgb[...,::-1]),cpu_features=False)
        gain,offset=self.backbone_module.canvas_affine(captured["canvas_shape"],captured["input_shape"])
        roi,_,side=self.roi_module.patches(captured,safe,hw,gain,offset)
        image_key=hashlib.sha256(str(rgb.shape).encode()+rgb.tobytes()).hexdigest()
        if image_key!=self._sam_image_key:self.sam_predictor.set_image(rgb);self._sam_image_key=image_key
        with torch.inference_mode():alternatives,quality,_=self.sam_predictor.predict(box=box.astype(np.float32),multimask_output=True,return_logits=True)
        if not np.isfinite(quality).all():return self.empty(points,hw,"INVALID_SAM_MASK_QUALITY",saved_initial_pose)
        choice=int(np.argmax(quality));selected=np.asarray(alternatives[choice])
        patches,bg,sg,_=native_inputs(rgb,q,side,box,selected)
        result=self.predict_precomputed(points=q,raw_hw=hw,x=roi.numpy(),rgb_native=patches,sam=sg,box_grid=bg,
                                         full_SAM_logits=selected,saved_initial_pose=saved_initial_pose)
        result.update(roi_side=float(side),sam_selected_index=choice,sam_uncalibrated_quality=np.asarray(quality).tolist())
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--candidate",required=True)
    parser.add_argument("--manifest");parser.add_argument("--input-json",required=True);parser.add_argument("--out",required=True)
    parser.add_argument("--device",default="cpu");args=parser.parse_args();out=Path(args.out)
    if out.exists():raise FileExistsError("preserve previous prediction output")
    request=json.loads(Path(args.input_json).read_text())
    if set(request)-{"image","initial_points","initial_pose","request_id"}:raise ValueError("only new RGB/N0 queries and optional saved geometry are accepted")
    from PIL import Image
    with Image.open(request["image"])as image:rgb=np.array(image.convert("RGB"))
    api=QueryMultiTargetVisibility(args.candidate,args.manifest,args.device)
    try:
        result=api.predict_rgb(rgb,request["initial_points"],request.get("initial_pose"));result["request_id"]=request.get("request_id")
        out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n");print(out.resolve())
    finally:api.close()


if __name__=="__main__":main()
