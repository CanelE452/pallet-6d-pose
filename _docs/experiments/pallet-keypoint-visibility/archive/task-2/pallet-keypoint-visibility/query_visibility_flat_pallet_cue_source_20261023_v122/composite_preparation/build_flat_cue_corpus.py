"""V122 concrete source-only opaque-cutout corpus; no model imports or calls.

--prepare records metadata/code bindings, without reading an image or mask.
--run is the one root-reviewed invocation. Full edited RGB stays transient;
shared resized RGB/alpha patches plus exact placements reconstruct it later.
"""
import argparse
import ast
from collections import Counter, OrderedDict
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np
from PIL import Image

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
V = HERE.parent
P = V.parent
POLICY = V/'ROOT_CONCRETE_CORPUS_POLICY_BEFORE_COMPOSITES.json'
ALPHA = V/'public_occluders/ROOT_OCCLUDER_ALPHA_ADMISSION.json'
META = V/'input_preparation'
OUT = V/'actual_corpus'
PROTOCOL = HERE/'CORPUS_PROTOCOL_BEFORE_PIXELS.json'
POOL = P/'query_visibility_matched_query_scalar_20261023_v94/paired_source_preparation/YCB4200_DONOR_METADATA_CANDIDATES_NOT_ADMITTED.jsonl'
KERNEL = P/'query_visibility_query_anchored_cue_20261023_v96/query_anchored_pair_preparation/query_anchored_pair_kernel.py'
KERNEL_SHA = '0299a3f6af75b3907875675499054b88bfc8f9a59203bd4ea4b58529f661b5f9'
CROP = P/'query_visibility_public_cue_inputs_20261023_v90/producer/spatial_input_math.py'
CROP_SHA = '6a6c2543e28715e3d47e15c543c25f75ff7d3efa3b881e523bd3451817fc004b'
FACTORS = (.6, 1., 1.4, 2.)
DIRECTIONS = ((1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1))
RADII_OVER_L = (.125,.25,.375,.5,.75,1.,1.5,2.,3.,4.)
COVER_OFFSETS = ((0,0),(-1,0),(1,0),(0,-1),(0,1),(-2,0),(2,0),(0,-2),(0,2))


def require(ok, why):
    if not ok:
        raise ValueError(why)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20), b''):
            h.update(b)
    return h.hexdigest()


def bind(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def check(b):
    require(sha(b['path']) == b['sha256'], 'Changed binding '+b['path'])
    return Path(b['path'])


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x]


def clean(x):
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, dict): return {k:clean(v) for k,v in x.items()}
    if isinstance(x, (tuple,list)): return [clean(v) for v in x]
    return x


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(clean(value), f, indent=2, allow_nan=False)
        f.write('\n')


def jsonl(path, values):
    with Path(path).open('x') as f:
        for value in values:
            f.write(json.dumps(clean(value), separators=(',',':'), allow_nan=False)+'\n')


def load_math():
    require(sha(KERNEL)==KERNEL_SHA and sha(CROP)==CROP_SHA, 'Pinned lower math changed')
    spec = importlib.util.spec_from_file_location('v122_exact_lower', KERNEL)
    k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
    tree = ast.parse(CROP.read_text())
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('crop','query_validity')]
    ns = dict(np=np); exec(compile(ast.Module(body=keep,type_ignores=[]),str(CROP),'exec'),ns)
    return k, ns['crop'], ns['query_validity']


def fixed_donors(pool, cone_manifest):
    """Prospective metadata availability only, never image/alpha/score search."""
    present = {fold:{r['object_id'] for r in pool if r['split']==fold and r['metadata_vis_pixels']>0}
               for fold in ('TRAIN','VALIDATION')}
    common = sorted((present['TRAIN'] & present['VALIDATION'])-{5,15,16})
    require(common and common[0]==4, 'Reconcile the root prospective replacement rule')
    ids = [common[0],5,15,16]
    chosen = []
    for fold in ('TRAIN','VALIDATION'):
        cone = next(x for x in cone_manifest['files'] if x['split']==fold)
        for rank,factor in enumerate(FACTORS):
            chosen.append(dict(fold=fold,rank=rank,kind='EXTERNAL_RGBA',factor=factor,
                               donor_id=cone['id'],source=cone))
        for rank,object_id in enumerate(ids,4):
            # Original metadata order, positive recorded visibility, not actual
            # alpha success. Missing files/invalid owner proof remain U later.
            donor = next((r for r in pool if r['split']==fold and r['object_id']==object_id
                          and r['metadata_vis_pixels']>0), None)
            chosen.append(dict(fold=fold,rank=rank,kind='YCB_EXCLUSIVE',object_id=object_id,
                               donor_id=None if donor is None else donor['id'],source=donor))
    train_sha = {d['source']['image_sha256'] for d in chosen if d['fold']=='TRAIN' and d['kind']=='YCB_EXCLUSIVE' and d['source']}
    val_sha = {d['source']['image_sha256'] for d in chosen if d['fold']=='VALIDATION' and d['kind']=='YCB_EXCLUSIVE' and d['source']}
    require(not train_sha & val_sha, 'Donor RGB cross-fold leakage')
    return chosen


def protected_mask(cue, evidence):
    """Exact original ranges intersecting RGB, without target-bbox clipping."""
    h,w = cue['raw_hw']; out = np.zeros((h,w),bool); outside = False
    require(evidence['before_visible_weak_supported'] is True, 'Weak before cue support required')
    ranges = []
    for region in evidence['original_regions']:
        b = np.asarray(region['bbox'],np.float64)
        require(b.shape==(4,) and np.isfinite(b).all(), 'Finite original region')
        if region['kind']=='HALF_OPEN_PIXEL_EDGE':
            require((b[2:]>b[:2]).all(), 'Positive half-open range')
            lo = np.floor(b[:2]+.5).astype(int); hi = np.ceil(b[2:]+.5).astype(int)
        elif region['kind']=='CLOSED_NATIVE_UNCERTAINTY':
            require((b[2:]>=b[:2]).all(), 'Ordered closed uncertainty')
            lo = np.floor(b[:2]+.5).astype(int); hi = np.floor(b[2:]+.5).astype(int)+1
        else: raise ValueError('Explicit source region convention required')
        outside |= bool((lo<0).any() or hi[0]>w or hi[1]>h)
        x0,y0 = np.maximum(lo,0); x1,y1 = np.minimum(hi,[w,h])
        if x1>x0 and y1>y0: out[y0:y1,x0:x1] = True
        ranges.append(dict(original_region=region,unclipped_native_cells_xyxy=[*lo.tolist(),*hi.tolist()]))
    ref = np.asarray(evidence['reference_xy'],np.float64)
    require(ref.shape==(2,) and np.isfinite(ref).all(), 'Finite original reference')
    rx,ry = np.floor(ref+.5).astype(int)
    if 0<=rx<w and 0<=ry<h: out[ry,rx] = True
    else: outside = True
    return out, dict(original_ranges=ranges,original_reference_xy=ref.tolist(),
                     original_uncertainty_has_outside_image_cells=outside,
                     protected_in_image_pixels=int(out.sum()),target_bbox_clipping=False)


def erode1(a):
    p=np.pad(a,1);h,w=a.shape
    return np.logical_and.reduce([p[y:y+h,x:x+w] for y in range(3) for x in range(3)])


def native_cutout(rgb, alpha):
    yy,xx=np.nonzero(alpha)
    if not len(xx): return None
    x0,y0,x1,y1=int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)
    return dict(rgb=rgb[y0:y1,x0:x1].copy(),alpha=alpha[y0:y1,x0:x1].copy(),
                native_bbox_xyxy=[x0,y0,x1,y1])


def resize_cutout(base, factor, L0, k):
    h,w=base['alpha'].shape; scale=float(factor*L0/max(h,w))
    nh,nw=max(1,int(np.floor(h*scale+.5))),max(1,int(np.floor(w*scale+.5)))
    rgb=np.array(Image.fromarray(base['rgb']).resize((nw,nh),Image.Resampling.BILINEAR),copy=True)
    alpha=np.array(Image.fromarray(base['alpha']).resize((nw,nh),Image.Resampling.NEAREST),dtype=bool,copy=True)
    core=k.V2.largest_opaque_rectangle(alpha)
    return dict(rgb=rgb,alpha=alpha,opaque_core_xyxy=core,
                factor_over_L0=float(factor),nominal_longest=float(factor*L0),
                rounded_resized_hw=[nh,nw],source_native_bbox_xyxy=base['native_bbox_xyxy'],
                aspect_rounding_rule='round-half-up each uniform-scaled dimension; min1')


def alpha_at(cut, xy, xs, ys, raw_hw):
    px=np.asarray(xs)-xy[0];py=np.asarray(ys)-xy[1];ph,pw=cut['alpha'].shape;h,w=raw_hw
    good=(px>=0)&(px<pw)&(py>=0)&(py<ph)&(np.asarray(xs)>=0)&(np.asarray(xs)<w)&(np.asarray(ys)>=0)&(np.asarray(ys)<h)
    out=np.zeros(np.shape(px),bool);out[good]=cut['alpha'][py[good],px[good]]
    return out


def alpha_mass96(cut, xy, q, L, raw_hw):
    h,w=raw_hw;u=(np.arange(96,dtype=np.float64)+.5)/96-.5
    xx,yy=np.meshgrid(q[0]+u*L,q[1]+u*L);ix=np.floor(xx).astype(int);iy=np.floor(yy).astype(int)
    dx,dy=xx-ix,yy-iy;mass=np.zeros((96,96),np.float64)
    for ax,ay,weight in ((0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)):
        mass+=alpha_at(cut,xy,ix+ax,iy+ay,raw_hw)*weight
    mass[~((xx>=-.5)&(xx<w-.5)&(yy>=-.5)&(yy<h-.5))]=0
    return float(mass.sum())


def patch_slices(hw, patch_hw, xy):
    h,w=hw;ph,pw=patch_hw;x,y=map(int,xy)
    x0,y0=max(0,x),max(0,y);x1,y1=min(w,x+pw),min(h,y+ph)
    if x1<=x0 or y1<=y0: return None
    return (slice(y0,y1),slice(x0,x1)),(slice(y0-y,y1-y),slice(x0-x,x1-x))


def reconstruct_full_RGB(native_rgb, patch_rgb, patch_alpha, xy):
    """Opaque replacement only, native clipping explicit; no feathering/painting."""
    require(native_rgb.dtype==patch_rgb.dtype==np.uint8 and patch_alpha.dtype==bool,'RGB/alpha dtype')
    require(patch_rgb.shape==patch_alpha.shape+(3,), 'Shared patch RGB/alpha shape')
    out=native_rgb.copy();s=patch_slices(out.shape[:2],patch_alpha.shape,xy)
    if s is not None:
        dst,src=s;alpha=patch_alpha[src];region=out[dst];region[alpha]=patch_rgb[src][alpha]
    return out


def placement_pair(cut, protected, q0, L0):
    """Fixed geometry enumeration, no model/color search or local-containment cap."""
    ys,xs=np.nonzero(protected);core=cut['opaque_core_xyxy']
    trials=[]
    if not len(xs) or core is None:return None,dict(reason='EMPTY_PROTECTED_OR_OPAQUE_CORE_U',cover_trials=[])
    x0,y0,x1,y1=core
    if x1-x0<xs.max()-xs.min()+1 or y1-y0<ys.max()-ys.min()+1:
        return None,dict(reason='FIXED_SCALE_CORE_TOO_SMALL_U',cover_trials=[])
    c0=np.floor(np.array([(xs.min()+xs.max())/2,(ys.min()+ys.max())/2])-
                np.array([(x0+x1-1)/2,(y0+y1-1)/2])+.5).astype(int)
    for dx,dy in COVER_OFFSETS:
        cxy=(int(c0[0]+dx),int(c0[1]+dy));covered=bool(alpha_at(cut,cxy,xs,ys,protected.shape).all())
        ct=dict(xy=list(cxy),protected_all_opaque=covered,sham_trials=[]);trials.append(ct)
        if not covered:continue
        seen=set()
        for radius in RADII_OVER_L:
            for direction_index,(sx,sy) in enumerate(DIRECTIONS):
                direction=np.asarray([sx,sy],np.float64);direction/=np.linalg.norm(direction)
                sxy=tuple((np.asarray(cxy)+np.floor(direction*radius*L0+.5).astype(int)).tolist())
                if sxy in seen:continue
                seen.add(sxy);clear=not bool(alpha_at(cut,sxy,xs,ys,protected.shape).any())
                mass=alpha_mass96(cut,sxy,q0,L0,protected.shape) if clear else 0.
                ct['sham_trials'].append(dict(xy=list(sxy),radius_over_L0=radius,direction_index=direction_index,
                                             protected_all_clear=clear,bilinear_alpha_mass96=mass))
                if clear and mass>0:
                    cm=alpha_mass96(cut,cxy,q0,L0,protected.shape)
                    return (cxy,sxy),dict(reason='WHOLE_UNCERTAINTY_COVER_AND_VISIBLE_LOCAL_SHAM_READY',
                        cover_trials=trials,local_alpha_mass96_at_r_L0=[cm,mass],
                        local_alpha_mass96_absolute_difference=abs(cm-mass),
                        local_alpha_mass_equal_claim=False,whole_localcrop_containment_required=False,
                        inherited_point8L_cap_applied=False)
    return None,dict(reason='NO_FIXED_COVER_AND_VISIBLE_LOCAL_SHAM_PLACEMENT_U',cover_trials=trials)


def context_any(box, hw):
    b=np.asarray(box,np.float64);h,w=hw
    if b.shape!=(4,) or not np.isfinite(b).all() or (b[2:]<=b[:2]).any():return False
    u=(np.arange(224,dtype=np.float64)+.5)/224
    x=b[0]+u*(b[2]-b[0]);y=b[1]+u*(b[3]-b[1])
    # Strict positive bilinear-neighbour support, rather than box intersection.
    return bool(((x>-1)&(x<w)).any() and ((y>-1)&(y<h)).any())


def prepare():
    bindings=dict(producer=bind(__file__),root_policy=bind(POLICY),cone_alpha_admission=bind(ALPHA),
        cues=bind(META/'CUES32_INPUT_ONLY.jsonl'),evidence=bind(META/'CUES32_WEAK_EVIDENCE_GT_ONLY.jsonl'),
        queries=bind(META/'QUERIES288_INPUT_ONLY.jsonl'),pool=bind(POOL),kernel=bind(KERNEL),crop_math=bind(CROP))
    pool=rows(POOL);donors=fixed_donors(pool,read(ALPHA))
    protocol=dict(completed=True,stage='SOURCE_IMAGE_ONLY_PREPARATION',bindings=bindings,
        selected_donors=donors,geometry=dict(factors_over_L0=FACTORS,cover_offsets=COVER_OFFSETS,
        sham_directions=DIRECTIONS,sham_radii_over_L0=RADII_OVER_L,
        cone_scale_tries=1,YCB_scale_tries='fixed factor order; first geometric pair; no donor replacement',
        opaque_alpha_rule='External RGBA exactly alpha255; YCB selected&~allother then erode1',
        clipping='Patches may cross image; preserve flag/count; cover tests all original in-image uncertainty cells',
        no_old_whole_localcrop_or_point8L_cap=True),
        ordering='cue*153 + jitter*17: BEFORE, then COVER/SHAM for ranks0..7',
        planned=dict(cues=32,queries=288,rank_attempts=256,occurrences=4896),
        root_reviewed_execution_command=f'python {__file__} --run',
        model_encoder_head_fit_current_V112_PnP=0)
    write(PROTOCOL,protocol)
    print(json.dumps(dict(protocol=bind(PROTOCOL),producer=bind(__file__),prepared=True,source_pixel_reads=0)))


def run():
    require(not OUT.exists(),'One invocation only: preserve partial failure and do not restart')
    p=read(PROTOCOL)
    for b in p['bindings'].values():check(b)
    require(p['bindings']['producer']==bind(__file__),'Changed producer after preparation')
    cues=rows(p['bindings']['cues']['path']);evidence=rows(p['bindings']['evidence']['path']);queries=rows(p['bindings']['queries']['path'])
    require(len(cues)==len(evidence)==32 and len(queries)==288,'Exact32x9 input metadata')
    source_groups={};image_folds={}
    for c in cues:
        require(source_groups.setdefault(c['source_frame_group'],c['split'])==c['split'],'Source frame/episode split leakage')
        require(image_folds.setdefault(c['decoded_RGB_sha256'],c['split'])==c['split'],'Source RGB cross-fold leakage')
    require(read(POLICY)['planned']['all_rows']==4896,'Root fixed4896 scope')
    k,crop,qvalid=load_math();OUT.mkdir();(OUT/'patches').mkdir();(OUT/'donor_proofs').mkdir();(OUT/'protected').mkdir()
    write(OUT/'RUN_STARTED.json',dict(protocol=bind(PROTOCOL),code=bind(__file__),source_only=True,model_calls=0))
    start=time.monotonic();source_cache=OrderedDict();donor_cache={};donor_facts=[]
    def rgb_file(binding,fmt):
        path=str(check(binding))
        if path not in source_cache:
            a=np.load(path,allow_pickle=False) if 'NPY' in fmt else np.array(Image.open(path).convert('RGB'),copy=True)
            require(a.dtype==np.uint8 and a.ndim==3 and a.shape[2]==3,'Canonical native RGB')
            source_cache[path]=a
            if len(source_cache)>4:source_cache.popitem(last=False)
        return source_cache[path]
    def mask(path,hw):
        if not Path(path).is_file():return None
        a=np.array(Image.open(path),copy=True)
        require(a.shape==tuple(hw) and np.isin(a,[0,255]).all(),'Binary original visible mask')
        return a>0
    def donor_payload(d):
        key=(d['fold'],d['donor_id'],d['kind'])
        if key in donor_cache:return donor_cache[key]
        r=d['source'];fact=dict(fold=d['fold'],donor_id=d['donor_id'],kind=d['kind'],source=r,admitted=False)
        rgb=alpha=None
        if r is None:fact['reason']='MISSING_ASSIGNED_METADATA_DONOR_U'
        elif d['kind']=='EXTERNAL_RGBA':
            if Path(r['RGBA_path']).is_file():
                check(dict(path=r['RGBA_path'],sha256=r['sha256']))
                rgba=np.array(Image.open(r['RGBA_path']).convert('RGBA'),copy=True);rgb=rgba[...,:3].copy();alpha=rgba[...,3]==255
                fact.update(admitted=bool(alpha.any()),reason='DECLARED_PUBLIC_RGBA_EXACT_OPAQUE_255',
                            native_hw=list(alpha.shape),opaque255_pixels=int(alpha.sum()),nonopaque_alpha_not_pasted=True)
            else:fact['reason']='MISSING_ASSIGNED_RGBA_U'
        else:
            scene=Path(r['visible_alpha_source']['path']).parent.parent
            gp=scene/'scene_gt.json';ip=scene/'scene_gt_info.json'
            if gp.is_file() and ip.is_file() and Path(r['image']['path']).is_file():
                gt=read(gp)[str(r['frame_id'])];info=read(ip)[str(r['frame_id'])]
                n=len(gt);idx=r['instance_id'];complete=0<=idx<n and gt[idx]['obj_id']==r['object_id']
                selected=mask(r['visible_alpha_source']['path'],r['raw_hw']);other=np.zeros(tuple(r['raw_hw']),bool);owners=[]
                for i,g in enumerate(gt):
                    path=scene/'mask_visib'/f"{r['frame_id']:06d}_{i:06d}.png";a=mask(path,r['raw_hw'])
                    owners.append(dict(instance_id=i,object_id=g['obj_id'],mask=None if a is None else bind(path)))
                    if a is None:complete=False
                    elif i!=idx:other|=a
                if selected is None:complete=False;selected=np.zeros_like(other)
                else:check(r['visible_alpha_source'])
                complete &= int(selected.sum())==int(info[idx]['px_count_visib']) if 0<=idx<len(info) else False
                exclusive,cert=k.alpha_rule.exclusive_alpha(selected,other,bool(complete));alpha=erode1(exclusive)
                rgb=rgb_file(r['image'],'PNG')
                proof=OUT/'donor_proofs'/f"{d['fold']}_OBJ{r['object_id']:02d}.npz"
                np.savez_compressed(proof,selected_visible=selected,other_visible_union=other,exclusive_alpha=exclusive,erode1_exclusive=alpha)
                fact.update(admitted=bool(cert['admitted'] and alpha.any()),reason=cert['reason'],
                            official_scene_GT=bind(gp),official_scene_info=bind(ip),complete_owner_instance_set=bool(complete),
                            complete_owner_masks=owners,exclusive_alpha=cert,native_proof=bind(proof))
            else:fact['reason']='MISSING_ASSIGNED_DONOR_OR_OFFICIAL_OWNER_METADATA_U'
        base=native_cutout(rgb,alpha) if fact['admitted'] else None
        fact['native_decoded_RGB_sha256']=None if rgb is None else digest(rgb)
        donor_facts.append(fact);donor_cache[key]=(base,fact)
        return base,fact
    occurrences=[];teachers=[];feasibility=[];crop_rgb=np.empty((4896,3,96,96),np.uint8);crop_valid=np.empty((4896,1,96,96),bool)
    try:
        # All donor choices are serialized before first donor/image read.
        jsonl(OUT/'DONOR16_CONDITIONS_BEFORE_PIXELS.jsonl',p['selected_donors'])
        for ci,(c,e) in enumerate(zip(cues,evidence)):
            require(c['cue_index']==e['cue_index']==ci,'Cue order/GT join')
            native=rgb_file(c['full_RGB'],c['full_RGB_format'])
            require(native.shape==tuple(c['raw_hw'])+(3,) and digest(native)==c['decoded_RGB_sha256'],'Bound source canonical decoded pixels')
            qrows=queries[9*ci:9*ci+9]
            for j,q in enumerate(qrows):
                require(q['query_index']==9*ci+j and q['cue_index']==ci and q['jitter_index']==j,'All288 unchanged query order')
                require(q['full_RGB']==c['full_RGB'] and q['split']==c['split'],'Same source/RGB/fold')
            protected,pfacts=protected_mask(c,e);pm=OUT/'protected'/f'CUE_{ci:02d}.npz';np.savez_compressed(pm,protected_union=protected)
            native_digest=digest(native)
            variants={};fold=c['split'];q0=np.asarray(e['reference_xy'],np.float64);L0=float(c['L0'])
            for rank in range(8):
                d=next(x for x in p['selected_donors'] if x['fold']==fold and x['rank']==rank)
                base,df=donor_payload(d);attempt=dict(cue_index=ci,donor_rank=rank,donor_id=d['donor_id'],
                    source=c['source'],split=fold,L0=L0,pair_ready=False,protected=pfacts,
                    protected_file=bind(pm),scale_trials=[])
                result=None
                factors=[d['factor']] if d['kind']=='EXTERNAL_RGBA' else FACTORS
                if base is not None:
                    for factor in factors:
                        cut=resize_cutout(base,factor,L0,k);pair,geo=placement_pair(cut,protected,q0,L0)
                        attempt['scale_trials'].append(dict(factor=factor,resized_hw=cut['rounded_resized_hw'],geometry=geo))
                        if pair is not None:result=(cut,pair,geo);break
                if result is None:
                    attempt['reason']=df['reason'] if base is None else attempt['scale_trials'][-1]['geometry']['reason']
                    variants[rank]=(native,native,None,attempt)
                else:
                    cut,(cxy,sxy),geo=result
                    cover=reconstruct_full_RGB(native,cut['rgb'],cut['alpha'],cxy)
                    sham=reconstruct_full_RGB(native,cut['rgb'],cut['alpha'],sxy)
                    require(np.array_equal(sham[protected],native[protected]),'Sham original protected RGB exact')
                    pp=OUT/'patches'/f'CUE_{ci:02d}_RANK_{rank:02d}.npz'
                    np.savez_compressed(pp,rgb=cut['rgb'],alpha=cut['alpha'])
                    patch=dict(file=bind(pp),rgb_key='rgb',alpha_key='alpha',resized_hw=cut['rounded_resized_hw'])
                    attempt.update(pair_ready=True,reason=geo['reason'],patch=patch,cover_xy=list(cxy),sham_xy=list(sxy),
                        selected_factor=cut['factor_over_L0'],source_native_bbox_xyxy=cut['source_native_bbox_xyxy'],
                        shared_patch_RGBA_transform=True,cover_protected_all_opaque=True,sham_protected_all_clear=True,
                        sham_protected_RGB_exact=True,cover_decoded_RGB_sha256=digest(cover),sham_decoded_RGB_sha256=digest(sham),
                        local_alpha_mass96_at_r_L0=geo['local_alpha_mass96_at_r_L0'],
                        local_alpha_mass_equal_claim=False,native_frame_clipping=[])
                    for xy in (cxy,sxy):
                        ph,pw=cut['alpha'].shape;x,y=xy;attempt['native_frame_clipping'].append(bool(x<0 or y<0 or x+pw>native.shape[1] or y+ph>native.shape[0]))
                    variants[rank]=(cover,sham,patch,attempt)
                feasibility.append(attempt)
            for j,q in enumerate(qrows):
                roles=[('BEFORE',None,native,None,None)]
                for rank in range(8):
                    cv,sm,patch,f=variants[rank]
                    roles.extend([('OPAQUE_COVER',rank,cv,patch,f),('CUE_PRESERVING_SHAM',rank,sm,patch,f)])
                for role,(variant,rank,image,patch,f) in enumerate(roles):
                    i=153*ci+17*j+role;rr,vv=crop(image,np.asarray(q['query_xy']),q['L'],96)
                    factual=bool(qvalid(np.asarray([q['query_xy']]),q['raw_hw'])[0][0] and vv[0,47:49,47:49].all())
                    ctx=context_any(q['context_box_xyxy'],q['raw_hw']);available=factual and ctx
                    crop_rgb[i]=rr;crop_valid[i]=vv
                    ready=variant=='BEFORE' or bool(f['pair_ready'])
                    label=(0 if variant!='OPAQUE_COVER' else 1) if ready and factual else -1
                    image_digest=native_digest if patch is None else f['cover_decoded_RGB_sha256' if variant=='OPAQUE_COVER' else 'sham_decoded_RGB_sha256']
                    recipe=dict(kind='ORIGINAL',original_full_RGB=c['full_RGB'],original_format=c['full_RGB_format'],raw_hw=c['raw_hw'],decoded_RGB_sha256=image_digest)
                    if patch is not None:
                        recipe.update(kind='OPAQUE_PATCH',patch=patch['file'],placement_xy=f['cover_xy' if variant=='OPAQUE_COVER' else 'sham_xy'])
                    o=dict(q,occurrence_index=i,occurrence_id=f"V122:{ci:02d}:{j}:{variant}:"+('NONE' if rank is None else str(rank)),
                        variant=variant,donor_rank=rank,composition_key=f'C{ci:02d}_'+('BEFORE' if variant=='BEFORE' else f'R{rank:02d}_{variant}'),
                        full_RGB_recipe=recipe,rgb96_row=i,rgb96_sha256=digest(rr),valid96_sha256=digest(vv),
                        factual_query_available=factual,context_available=ctx,query_available=available)
                    occurrences.append(o)
                    teachers.append(dict(occurrence_index=i,occurrence_id=o['occurrence_id'],cue_index=ci,jitter_index=j,
                        donor_rank=rank,variant=variant,candidate_aux_hidden_label_GT_only=label,
                        original_before_weak_label_unchanged=0,geometry_pair_ready=False if variant=='BEFORE' else f['pair_ready'],
                        reason='UNCHANGED_VISIBLE_STRUCTURAL_CUE_AUX_BEFORE' if variant=='BEFORE' else f['reason'],
                        weak_region_component_label_not_native_pixel_gold=True,native_mapping='UNKNOWN',human_gold=False,training_admission=False))
            print(json.dumps(dict(cue_completed=ci+1,total_cues=32,accepted_ranks=sum(variants[r][3]['pair_ready'] for r in range(8)))),flush=True)
        require(len(occurrences)==4896 and [o['occurrence_index'] for o in occurrences]==list(range(4896)),'All4896 retained')
        jsonl(OUT/'OCCURRENCES4896_INPUT_ONLY.jsonl',occurrences);jsonl(OUT/'ORIGINAL_WEAK_AND_COMPOSITE_LABELS4896_GT_ONLY.jsonl',teachers)
        jsonl(OUT/'FEASIBILITY256_GT_ONLY.jsonl',feasibility);jsonl(OUT/'DONOR_OWNERSHIP_AND_ALPHA_FACTS.jsonl',donor_facts)
        np.savez_compressed(OUT/'INPUT_CROPS4896_INPUT_ONLY.npz',rgb96=crop_rgb,valid96=crop_valid,occurrence_index=np.arange(4896,dtype=np.int32))
        summary=dict(completed=True,PASS=True,source_only=True,cues=32,query_conditions=288,rank_attempts=256,occurrences=4896,
            accepted_cue_rank_pairs=sum(f['pair_ready'] for f in feasibility),labels=Counter(str(t['candidate_aux_hidden_label_GT_only']) for t in teachers),
            feasibility_by_source_fold={},failures=Counter(f['reason'] for f in feasibility if not f['pair_ready']),
            canonical_crop_shape=list(crop_rgb.shape),valid_shape=list(crop_valid.shape),seconds=time.monotonic()-start,
            full_native_edited_RGB_persisted=0,unchanged_original_q8_modified=0,local_mass_match_claim=False,
            model_encoder_head_fit_current_V112_PnP=0,training_admission=False,native_mapping='UNKNOWN',human_gold=False)
        for f in feasibility:
            key=f"{f['source']}:{f['split']}";s=summary['feasibility_by_source_fold'].setdefault(key,dict(attempts=0,accepted=0))
            s['attempts']+=1;s['accepted']+=int(f['pair_ready'])
        write(OUT/'SUMMARY.json',summary)
        outputs={path.stem:bind(path) for path in sorted(OUT.glob('*')) if path.is_file()}
        seal=dict(completed=True,PASS=True,protocol=bind(PROTOCOL),code=bind(__file__),outputs=outputs,
                  patches=[bind(x) for x in sorted((OUT/'patches').glob('*.npz'))],
                  protected=[bind(x) for x in sorted((OUT/'protected').glob('*.npz'))],
                  donor_proofs=[bind(x) for x in sorted((OUT/'donor_proofs').glob('*.npz'))],
                  all4896_retained=True,model_encoder_head_fit_current_V112_PnP=0,training_admission=False)
        write(OUT/'CORPUS_SEAL.json',seal);print(json.dumps(dict(seal=bind(OUT/'CORPUS_SEAL.json'),summary=summary)),flush=True)
    except Exception:
        write(OUT/'FAILED_NO_RESTART.json',dict(error=traceback.format_exc(),seconds=time.monotonic()-start,
              preserve_all_partial_outputs=True,model_encoder_head_fit_current_V112_PnP=0))
        raise


if __name__=='__main__':
    a=argparse.ArgumentParser();g=a.add_mutually_exclusive_group(required=True)
    g.add_argument('--prepare',action='store_true');g.add_argument('--run',action='store_true');args=a.parse_args()
    prepare() if args.prepare else run()
