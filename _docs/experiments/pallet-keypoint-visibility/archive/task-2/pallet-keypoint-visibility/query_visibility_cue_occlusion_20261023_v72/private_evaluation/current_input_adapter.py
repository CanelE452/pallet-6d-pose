"""Frozen current-input adapter. Imports no model and reads no data on import.

Only root's explicit post-source-freeze gate permits private cache/RGB reads.
This head emits D/E-conditional scores; it has no learned U/S/O detector.
"""
from pathlib import Path
import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import shutil
import sys
import time
import numpy as np

PROTOCOL_SHA = 'be6df166527072e5135e3fba402063c502043784173e772a8ab2f4182aef3c70'
FORWARD_KEYS = {'p3_roi','rgb96','crop_valid96','query_xy','box_xyxy','side_L'}
INPUT_KEYS = {'roi_cache','prepared_inputs','feature_metadata'}


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()


def bind(path):
    return dict(path=str(Path(path).resolve()),sha256=sha(path))


def write(path, value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def need(value, why):
    if not value: raise PermissionError(why)


def binding(entry):
    need(set(entry)=={'path','sha256'} and len(entry['sha256'])==64,'Expected exact path/SHA binding')
    need(sha(entry['path'])==entry['sha256'],'Bound file changed: '+entry['path'])


def read_json(path):
    p=Path(path)
    with (gzip.open if p.suffix=='.gz' else open)(p,'rt') as f: return json.load(f)


def import_bound(entry):
    binding(entry)
    spec=importlib.util.spec_from_file_location('_v72_bound_'+entry['sha256'][:16],entry['path'])
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    return mod


def validate_gate(path):
    """No private inputs are hashed, opened, or imported in this function."""
    g=read_json(path)
    need(g.get('approved') is True and g.get('source_selection_complete') is True and g.get('allow_current_evaluation_inference') is True,'Post-freeze explicit execution gate required')
    need(g.get('phase')=='FROZEN_ONE_SHOT_DIAGNOSTIC','Only one frozen diagnostic is permitted')
    need(g.get('current_used_for_model_or_threshold_selection') is False,'Current-based selection forbidden')
    need(g.get('threshold_probability_D')==.5 and g.get('threshold_logit')==0. and g.get('tie_is_D') is True,'Fixed .5 inclusive D threshold')
    need(g.get('expected_frames')==223 and g.get('expected_slots')==1784,'All 223 frames / native8 slots required')
    need(g.get('code')==bind(__file__),'Adapter code must be explicitly bound')
    binding(g['source_selection_seal']);binding(g['source_model_freeze'])
    need(g['source_selection_seal']==g['source_model_freeze'],'Single source-selected model freeze is the selection seal')
    f=read_json(g['source_model_freeze']['path'])
    need(f.get('completed') is True and f.get('optimizer_updates')==1000 and f.get('parameter_count')==40682,'Incomplete/unexpected source fit')
    need(f.get('current_used_for_fit_selection') is False and f.get('current_used_for_threshold') is False and f.get('PnP_calls')==0,'Source-only freeze required')
    need(f.get('threshold_probability_D')==.5 and f.get('threshold_logit')==0. and f.get('tie_is_D') is True,'Freeze threshold mismatch')
    for key in ['checkpoint','model_code','trainer_code','protocol']:
        binding(f[key])
    need(f['checkpoint']==g['checkpoint'] and f['model_code']==g['model_code'],'Selected checkpoint/code mismatch')
    need(f['protocol']==g['source_protocol'] and f['protocol']['sha256']==PROTOCOL_SHA,'Source protocol mismatch')
    binding(g['sampler_recipe']);binding(g['geometry_helper']);binding(g['prior_v63_input_gate'])
    prior=read_json(g['prior_v63_input_gate']['path'])
    need(set(g['inputs'])==INPUT_KEYS and g['inputs']==prior['inputs'],'Only unchanged registered N0 input artifacts allowed')
    for entry in g['inputs'].values():
        need(set(entry)=={'path','sha256'} and len(entry['sha256'])==64,'Unbound private input descriptor')
    need(g['runtime']['device']=='cpu' and g['runtime']['num_threads']==1,'Frozen CPU one-thread recipe')
    need(g.get('roughbox_rule')=='V65_INITIAL_Q8_MINMAX_CLIP_UNCHANGED','Registered input-only box rule')
    need(g.get('RGB96_recipe')=='V71_YCB_NATIVE_PIXEL_CENTER_BILINEAR_CONSTANT0_UINT8_ROUND','Registered RGB sampler')
    return g,f


def crop96(rgb,q,L):
    """Exact V71 grid/V72 builder sampler; no q rounding or recentering.

    Sample q+((j+.5)/96-.5)*L; bilinear native pixel centers; floor(value+.5).
    Pixel-cell footprint outside[-.5,W-.5)x[-.5,H-.5) is zero/invalid.
    floor(q+.5) is used only to audit query validity, never to shift the crop.
    """
    h,w=rgb.shape[:2];u=(np.arange(96,dtype=np.float64)+.5)/96-.5
    sx=q[0]+u*L;sy=q[1]+u*L;xx,yy=np.meshgrid(sx,sy)
    x0=np.floor(xx).astype(int);y0=np.floor(yy).astype(int);dx=xx-x0;dy=yy-y0
    acc=np.zeros((96,96,3),np.float64)
    for xo,yo,weight in [(0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)]:
        x=x0+xo;y=y0+yo;inside=(x>=0)&(x<w)&(y>=0)&(y<h)
        acc[inside]+=rgb[y[inside],x[inside]].astype(np.float64)*weight[inside,None]
    valid=(xx>=-.5)&(xx<w-.5)&(yy>=-.5)&(yy<h-.5)
    result=np.floor(np.clip(acc,0,255)+.5).astype(np.uint8);result[~valid]=0
    return result.transpose(2,0,1).copy(),valid[None].copy()


def initial_query_box(points,hw):
    q=np.asarray(points,float)[:8];h,w=hw;good=np.isfinite(q).all(1)&~(q==-1).all(1)
    if good.sum()<2:return None
    lo=np.maximum(q[good].min(0),[0.,0.]);hi=np.minimum(q[good].max(0),[w-1.,h-1.])
    return np.r_[lo,hi] if (hi-lo>=2).all() else None


def query_validity(points,hw,pixel_valid):
    q=np.asarray(points,float)[:8];h,w=hw
    finite=np.isfinite(q).all(1)&~(q==-1).all(1)
    pixel=np.floor(np.where(np.isfinite(q),q,-1e8)+.5).astype(np.int64)
    valid=finite&(q[:,0]>=0)&(q[:,0]<w)&(q[:,1]>=0)&(q[:,1]<h)
    valid&=(pixel[:,0]>=0)&(pixel[:,0]<w)&(pixel[:,1]>=0)&(pixel[:,1]<h)&np.asarray(pixel_valid,bool)
    return valid,pixel


def saved_geometry(frame,helper):
    pose=frame.get('initial_pose',{})
    if not pose.get('available',False):return [None]*8,'UNAVAILABLE_SAVED_INITIAL_POSE'
    try:return helper.conditional_self_hidden(pose['cf_extents'],pose['R_cf'],pose['centroid']).astype(bool).tolist(),'SAVED_INITIAL_REGISTERED_CUBOID_PROXY'
    except (KeyError,ValueError,TypeError):return [None]*8,'INVALID_SAVED_INITIAL_GEOMETRY'


def adapt_frame(rgb,frame,meta,x,pixel_valid,geometry):
    """No labels/reference/state accepted by forward dict. Geometry is output only."""
    q=np.asarray(frame['initial_points'],np.float64)[:8]
    if q.shape!=(8,2) or x.shape!=(8,65,9,9) or len(pixel_valid)!=8:raise ValueError('Native8 saved shape mismatch')
    if meta['id']!=frame['id'] or list(rgb.shape[:2])!=frame['raw_hw']:raise ValueError('Frame order/canvas mismatch')
    L=float(meta['side']);box=initial_query_box(q,frame['raw_hw'])
    finite=np.isfinite(q).all(1)
    valid,pixels=query_validity(q,frame['raw_hw'],pixel_valid)
    valid&=x[:,64].astype(bool).any((1,2))
    callable_mask=finite & np.isfinite(L) & (L>0) & (box is not None)
    valid&=callable_mask
    rgb96=np.zeros((8,3,96,96),np.uint8);cv=np.zeros((8,1,96,96),bool)
    for k in np.flatnonzero(callable_mask):rgb96[k],cv[k]=crop96(rgb,q[k],L)
    flags,geometry_reason=saved_geometry(frame,geometry)
    out=[]
    for k in range(8):
        reason='' if valid[k] else ('MISSING_OR_DEGENERATE_QUERY_BOX' if box is None else 'INVALID_QUERY_OR_EMPTY_ROI')
        out.append(dict(id=frame['id'],native_corner=k,query_x=float(q[k,0]) if finite[k] else None,query_y=float(q[k,1]) if finite[k] else None,
          pixel_x=int(pixels[k,0]),pixel_y=int(pixels[k,1]),image_sha256=frame['image_sha256'],roi_side=L,
          saved_roi_pixel_valid=bool(pixel_valid[k]),valid=bool(valid[k]),network_callable=bool(callable_mask[k]),status='OK' if valid[k] else 'UNKNOWN',reason=reason,
          geometry_self_hidden=flags[k],geometry_self_source=geometry_reason,geometry_overrides_learned_probability=False,
          logit=None,p_visible=None,present_at_frozen_threshold=None,p_D_given_DE=None,p_E_given_DE=None,predicted_D_or_E='UNKNOWN',
          raw_logit_D_given_DE=None,raw_p_D_given_DE=None,raw_p_E_given_DE=None,
          source_reference_or_crop_availability_known=False,semantic_U_detector=False,self_occlusion_learned=False))
    # Noncallable rows have safe placeholders; never forwarded or interpreted.
    features=dict(p3_roi=np.asarray(x,np.float32),rgb96=rgb96,crop_valid96=cv,query_xy=q.astype(np.float32),
                  box_xyxy=np.repeat(np.zeros((1,4),np.float32) if box is None else box[None].astype(np.float32),8,axis=0),side_L=np.full(8,L,np.float32))
    if set(features)!=FORWARD_KEYS:raise AssertionError('Unexpected forward field')
    return features,out,callable_mask


def runtime_check(g):
    import torch
    need(Path(sys.executable).resolve()==Path(g['runtime']['executable']).resolve(),'Runtime executable changed')
    need(str(torch.__version__)==g['runtime']['torch_version'],'Torch runtime changed')
    torch.set_num_threads(1);torch.manual_seed(20261023)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
    return torch


def prepare(args):
    g,f=validate_gate(args.gate);runtime_check(g)
    output=Path(g['output_directory']).resolve()
    if output.exists():raise FileExistsError('Preserve prior preparation/output directory')
    # First private file accesses only after every source/code/gate condition.
    for entry in g['inputs'].values():binding(entry)
    with np.load(g['inputs']['roi_cache']['path'],allow_pickle=False) as z:x=z['x'];pv=z['pixel_valid']
    frames=read_json(g['inputs']['prepared_inputs']['path']);metadata=read_json(g['inputs']['feature_metadata']['path'])
    n=len(frames)
    if n!=223 or x.shape!=(n,8,65,9,9) or pv.shape!=(n,8) or len(metadata)!=n:raise ValueError('All223 native8 registered arrays required')
    if not np.isfinite(x).all() or len({r['id'] for r in frames})!=n:raise ValueError('Nonfinite cached P3/duplicate frame')
    if not np.isin(x[:,:,64],(0,1)).all():raise ValueError('Native validity channel must be binary')
    geometry=import_bound(g['geometry_helper'])
    output.mkdir(parents=True);shutil.copy2(__file__,output/'ADAPTER_CODE.py')
    write(output/'PREPARATION_STARTED.json',dict(gate=bind(args.gate),source_freeze=g['source_model_freeze'],current_ref_or_annotation_or_scores_read=False,optimizer_updates=0,PnP_calls=0))
    from PIL import Image
    fields={k:[] for k in FORWARD_KEYS};rows=[];images=[];start=time.monotonic()
    for i,(frame,meta) in enumerate(zip(frames,metadata)):
        entry=dict(path=frame['image'],sha256=frame['image_sha256']);binding(entry)
        rgb=np.array(Image.open(frame['image']).convert('RGB'))
        features,frame_rows,callable_mask=adapt_frame(rgb,frame,meta,x[i],pv[i],geometry)
        for k in fields:fields[k].append(features[k])
        rows.extend(frame_rows);images.append(entry)
    np.savez_compressed(output/'CURRENT_MODEL_INPUT.npz',**{k:np.ascontiguousarray(np.concatenate(v)) for k,v in fields.items()})
    (output/'ROWS_INPUT_ONLY.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows))
    write(output/'PREPARATION_SEAL.json',dict(completed=True,gate=bind(args.gate),source_freeze=g['source_model_freeze'],code=bind(__file__),
        outputs=[bind(output/'CURRENT_MODEL_INPUT.npz'),bind(output/'ROWS_INPUT_ONLY.jsonl')],images=images,frames=n,slots=len(rows),valid=sum(r['valid'] for r in rows),
        original_q_L_order_preserved=True,roughbox_rule=g['roughbox_rule'],RGB96_recipe=g['RGB96_recipe'],
        new_backbone_or_SAM_forwards=0,optimizer_updates=0,PnP_calls=0,current_ref_or_annotation_or_scores_read=False,seconds=time.monotonic()-start))


def infer(args):
    g,f=validate_gate(args.gate);torch=runtime_check(g);output=Path(g['output_directory']).resolve()
    if (output/'INFERENCE_SEAL.json').exists() or (output/'REAL_QUERY_SCORES.csv').exists():raise FileExistsError('No inference output overwrite')
    prep=read_json(output/'PREPARATION_SEAL.json')
    need(prep['completed'] is True and prep['gate']==bind(args.gate) and prep['source_freeze']==g['source_model_freeze'],'Preparation seal mismatch')
    for entry in prep['outputs']:binding(entry)
    with np.load(output/'CURRENT_MODEL_INPUT.npz',allow_pickle=False) as z:
        need(set(z.files)==FORWARD_KEYS,'Input-only tensor whitelist');features={k:z[k] for k in z.files}
    rows=[json.loads(s) for s in (output/'ROWS_INPUT_ONLY.jsonl').read_text().splitlines()]
    need(len(rows)==1784 and all(v.shape[0]==1784 for v in features.values()),'All1784 slots required')
    model=import_bound(g['model_code']).CueVisibilityHead()
    need(sum(p.numel() for p in model.parameters())==40682,'Architecture count changed')
    model.load_state_dict(torch.load(g['checkpoint']['path'],map_location='cpu',weights_only=True),strict=True)
    model.eval().requires_grad_(False);before={k:v.detach().clone() for k,v in model.state_dict().items()}
    ids=np.array([i for i,r in enumerate(rows) if r['network_callable']],np.int64)
    logits=np.full(1784,np.nan,np.float32);heat=np.full((1784,9,9),np.nan,np.float32);attention=heat.copy();repeat=0.
    start=time.monotonic()
    with torch.inference_mode():
        for j in range(0,len(ids),64):
            ix=ids[j:j+64];inputs={k:torch.from_numpy(np.ascontiguousarray(v[ix])) for k,v in features.items()}
            result=model(**inputs);again=model(**inputs)
            repeat=max(repeat,float((result['logit_D_given_DE']-again['logit_D_given_DE']).abs().max()))
            logits[ix]=result['logit_D_given_DE'].numpy();heat[ix]=result['cue_heatmap_logits'].numpy();attention[ix]=result['cue_attention_weights'].numpy()
    need(all(torch.equal(before[k],v) for k,v in model.state_dict().items()),'Model state changed')
    for i,r in enumerate(rows):
        if not r['network_callable']:continue
        z=float(logits[i]);p=float(np.exp(-np.logaddexp(0.,-z)))
        need(np.isfinite(z),'Nonfinite network score')
        r.update(raw_logit_D_given_DE=z,raw_p_D_given_DE=p,raw_p_E_given_DE=1-p)
        if r['valid']:
            r.update(logit=z,p_visible=p,present_at_frozen_threshold=bool(z>=0),p_D_given_DE=p,p_E_given_DE=1-p,predicted_D_or_E='D' if z>=0 else 'E')
    with open(output/'REAL_QUERY_SCORES.csv','w',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    np.savez_compressed(output/'RAW_NETWORK_OUTPUTS.npz',logit_D_given_DE=logits,cue_heatmap_logits=heat,cue_attention_weights=attention,
                        valid=np.array([r['valid'] for r in rows]),network_callable=np.array([r['network_callable'] for r in rows]))
    binding(g['checkpoint'])
    write(output/'INFERENCE_SEAL.json',dict(completed=True,gate=bind(args.gate),source_model_freeze=g['source_model_freeze'],checkpoint=g['checkpoint'],
        code=bind(__file__),preparation=bind(output/'PREPARATION_SEAL.json'),outputs=[bind(output/'REAL_QUERY_SCORES.csv'),bind(output/'RAW_NETWORK_OUTPUTS.npz')],
        frames=223,slots=1784,valid=sum(r['valid'] for r in rows),threshold_probability_D=.5,threshold_logit=0.,tie_is_D=True,
        score_scope='Conditional source-reference D/E score. p_visible is compatibility alias of p_D_given_DE, not native-pixel occupancy or calibrated causal E.',
        semantic_U_detector=False,self_occlusion_learned=False,geometry_overrides_learned_probability=False,source_reference_or_crop_availability_known=False,
        model_state_unchanged=True,repeat_max_logit_difference=repeat,current_ref_or_annotation_or_scores_read=False,
        new_backbone_or_SAM_forwards=0,optimizer_updates=0,PnP_calls=0,model_or_threshold_selection=False,seconds=time.monotonic()-start))


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','infer','run']);p.add_argument('--gate',required=True);a=p.parse_args()
    if a.stage in ('prepare','run'):prepare(a)
    if a.stage in ('infer','run'):infer(a)


if __name__=='__main__':main()
