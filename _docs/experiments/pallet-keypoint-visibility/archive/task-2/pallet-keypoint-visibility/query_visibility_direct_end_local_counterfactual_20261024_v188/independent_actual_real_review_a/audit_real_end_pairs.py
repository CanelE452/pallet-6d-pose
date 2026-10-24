"""Independent saved source-real pixel audit. No producer/helper imports or models."""
import argparse, hashlib, itertools, json, math, os, stat, traceback
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.ndimage import distance_transform_edt

P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
V=P/'query_visibility_direct_end_local_counterfactual_20261024_v188'
HERE=V/'independent_actual_real_review_a'
GEN=V/'real_end_local_pairs_v1'
PROTOCOL=HERE/'PROTOCOL_BEFORE_SAVED_REAL_PIXEL_AUDIT.json'
NAMES=('BEFORE','COVER','ADJACENT_SHAM')

def need(v,msg):
    if not bool(v):raise AssertionError(msg)
def safe(p):
    p=Path(p);need(p.is_absolute() and '..' not in p.parts and not any('v112' in s.casefold() for s in p.parts),'explicit allowed path')
    return p
def opened(p):
    p=safe(p);fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for part in p.parts[1:-1]:
            f2=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=f2
        f2=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
    finally:os.close(fd)
    need(stat.S_ISREG(os.fstat(f2).st_mode),'regular file');return os.fdopen(f2,'rb')
def sha(p):
    h=hashlib.sha256()
    with opened(p) as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def bind(p):return {'path':str(p),'sha256':sha(p)}
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def eq(a,b):return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def jread(p):
    with opened(p) as f:return json.load(f)
def checked(b):need(sha(b['path'])==b['sha256'],'file SHA '+b['path']);return jread(b['path'])
def lines(b):
    need(sha(b['path'])==b['sha256'],'JSONL SHA')
    with opened(b['path']) as f:return [json.loads(s) for s in f if s.strip()]
def arrays(b,keys=None):
    need(sha(b['path'])==b['sha256'],'array file SHA '+b['path'])
    with opened(b['path']) as f:
        a=np.load(f,allow_pickle=False)
        if isinstance(a,np.lib.npyio.NpzFile):
            with a:return {k:a[k].copy() for k in (keys or a.files)}
        return a.copy()
def write(p,j):
    with p.open('x') as f:json.dump(j,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def taps(x,y,hw):
    h,w=hw
    if not (-.5<=x<w-.5 and -.5<=y<h-.5):return []
    i,j=math.floor(x),math.floor(y);fx,fy=x-i,y-j;out=[]
    for xx,wx in ((i,1-fx),(i+1,fx)):
        for yy,wy in ((j,1-fy),(j+1,fy)):
            if wx*wy>0 and 0<=xx<w and 0<=yy<h:out.append(((xx,yy),wx*wy))
    return out
def footprint(q,L,hw):
    s=set();by=[]
    for y in range(96):
        for x in range(96):
            p=[q[0]+((x+.5)/96-.5)*L,q[1]+((y+.5)/96-.5)*L]
            a={v for v,w in taps(*p,hw)};s.update(a);by.append(a)
    return s,by
def enclosure(boxes):
    result=set()
    for x0,y0,x1,y1 in boxes:
        need(x0<=x1 and y0<=y1 and all(math.isfinite(x) for x in (x0,y0,x1,y1)),'finite closed enclosure')
        result.update(itertools.product(range(math.floor(x0),math.ceil(x1)+1),range(math.floor(y0),math.ceil(y1)+1)))
    return result
def local_sample(rgb,q,L):
    h,w=rgb.shape[:2];g=(np.arange(96,dtype=np.float64)+.5)/96-.5
    x,y=np.meshgrid(q[0]+g*L,q[1]+g*L);ix=np.floor(x).astype(np.int64);iy=np.floor(y).astype(np.int64)
    dx=x-ix;dy=y-iy;z=np.zeros((96,96,3),np.float64)
    # Same mathematical four-neighbor interpolation, implemented without source helpers.
    for ox,oy,ww in ((0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)):
        xx,yy=ix+ox,iy+oy;good=(xx>=0)&(yy>=0)&(xx<w)&(yy<h)
        z[good]+=rgb[yy[good],xx[good]].astype(np.float64)*ww[good,None]
    valid=(x>=-.5)&(x<w-.5)&(y>=-.5)&(y<h-.5)
    z=np.floor(np.clip(z,0,255)+.5).astype(np.uint8);z[~valid]=0
    return z,valid
def context_sample(rgb,box):
    if box is None or not all(math.isfinite(v) for v in box) or box[2]<=box[0] or box[3]<=box[1]:return np.zeros((224,224,3),np.float32),np.zeros((224,224),bool)
    h,w=rgb.shape[:2];x,y=np.meshgrid(box[0]+(np.arange(224)+.5)*(box[2]-box[0])/224,box[1]+(np.arange(224)+.5)*(box[3]-box[1])/224)
    ix,iy=np.floor(x).astype(np.int64),np.floor(y).astype(np.int64);dx,dy=x-ix,y-iy
    z=np.zeros((224,224,3),np.float64);valid=np.ones((224,224),bool)
    for ox,oy,ww in ((0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)):
        xx,yy=ix+ox,iy+oy;inside=(xx>=0)&(xx<w)&(yy>=0)&(yy<h)
        z+=rgb[np.clip(yy,0,h-1),np.clip(xx,0,w-1)].astype(np.float64)*(ww*inside)[...,None]
        valid&=inside|(ww==0)
    return z.astype(np.float32),valid
def raw_mask(shape,alpha,origin):
    h,w=shape;y,x=origin[1],origin[0];ch,cw=alpha.shape
    need(0<=x and 0<=y and x+cw<=w and y+ch<=h,'unclipped hard-alpha placement')
    a=np.zeros(shape,bool);a[y:y+ch,x:x+cw]=alpha;return a
def resized_donor(rgb,a,longside):
    h,w=a.shape;scale=max(1,int(longside))/max(h,w);size=(max(1,math.floor(w*scale+.5)),max(1,math.floor(h*scale+.5)))
    out=np.array(Image.fromarray(rgb).resize(size,Image.Resampling.BILINEAR))
    b=np.array(Image.fromarray(a).resize(size,Image.Resampling.NEAREST),bool);z=np.pad(b,1)
    er=np.ones_like(b)
    for y in range(3):
        for x in range(3):er&=z[y:y+b.shape[0],x:x+b.shape[1]]
    return out,er

def invented():
    checks={};rgb=np.arange(11*13*3,dtype=np.uint8).reshape(11,13,3)
    r,v=local_sample(rgb,[6,5],96);checks['integer_grid_exact_at_center_offsets']=eq(r[47:49,47:49],np.floor((rgb[4:6,5:7].astype(float)+rgb[4:6,6:8]+rgb[5:7,5:7]+rgb[5:7,6:8])/4+.5).astype(np.uint8))
    checks['integer_tap_zero_weight_absent']=taps(3.,4.,[11,13])==[((3,4),1.)]
    checks['invalid_position_has_no_positive_source']=not taps(-.51,4,[11,13])
    s,_=footprint([400.,400.],288.,[800,800]);e=enclosure([[395,395,405,405]])
    xs,ys=zip(*s);missing=e-s
    checks['internal_downsample_holes_distinguished_from_extent']=bool(missing) and all(min(xs)<=x<=max(xs) and min(ys)<=y<=max(ys) for x,y in missing)
    checks['candidate_closed_integer_endpoint_retained']=enclosure([[2,3,2,3]])=={(2,3)}
    a=np.zeros((4,5),bool);a[1:3,1:4]=True;ca=raw_mask((20,20),a,[2,2]);sa=raw_mask((20,20),a,[12,2])
    checks['same_positive_mass_and_disjoint_controls']=ca.sum()==sa.sum()==6 and not (ca&sa).any()
    checks['surrounding_witness_outside_both']=not ca[10,10] and not sa[10,10]
    checks['uint8_rounding_preserves_noop_possible']=np.uint8(math.floor(.1+.5))==0
    need(all(checks.values()),'invented failed '+str(checks));return {k:bool(v) for k,v in checks.items()}

def marked_source(case,observation,folder):
    ev=observation['endpoint_observation']['evidence_binding'];b=[x for x in ev['actual_viewed_images'] if x['kind']=='nominal_ref_diagnostic'][0]
    need(sha(b['path'])==b['sha256'],'original diagnostic SHA')
    with opened(b['path']) as f:
        with Image.open(f) as im:original=im.convert('RGB').copy()
    need(original.size==(224,224),'unmarked original224 diagnostic')
    canvas=Image.new('RGB',(1394,840),'white');draw=ImageDraw.Draw(canvas);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    for x in (14,708):canvas.paste(original.resize((672,672),Image.Resampling.NEAREST),(x,100))
    offset=case['existing_nominal_diagnostic_raw_xyxy'][:2];ep=observation['endpoint_observation']
    for box in ep['candidate_enclosures_xyxy']:
        a=[708+(box[0]-offset[0])*3,100+(box[1]-offset[1])*3,708+(box[2]-offset[0])*3,100+(box[3]-offset[1])*3];draw.rectangle(a,outline='red',width=3)
    for box in observation['nonlocal_visual_candidate_enclosures_raw_xyxy']:
        draw.rectangle([708+(box[0]-offset[0])*3,100+(box[1]-offset[1])*3,708+(box[2]-offset[0])*3,100+(box[3]-offset[1])*3],outline='orange',width=3)
    for w in ep['surrounding_material_witnesses']:
        px,py=w['raw_pixel_xy'];x=708+(px-offset[0])*3;y=100+(py-offset[1])*3
        draw.ellipse((x-7,y-7,x+7,y+7),outline='yellow',width=3);draw.line((x-10,y,x+10,y),fill='yellow',width=2);draw.line((x,y-10,x,y+10),fill='yellow',width=2)
    draw.text((14,12),case['case_id']+' / native'+str(case['native_slot'])+' / '+observation['endpoint_observation_status'],font=font,fill='black')
    draw.text((14,40),'LEFT clean original diagnostic224 (3x NN); RIGHT same pixels + separate display annotations.',font=font,fill='black')
    draw.text((14,67),'RED finite end candidate; YELLOW chosen same-part witness; ORANGE nonlocal UNKNOWN candidate.',font=font,fill='black')
    draw.text((14,788),'Assistant weak semantic end; not native pixel/coordinate gold. Original q/L/inputs and frozen observation unchanged.',font=font,fill='black')
    f=folder/(case['case_id']+'_SOURCE_ENDPOINT_ANNOTATION.png');canvas.save(f)
    with Image.open(f) as im:arr=np.array(im)
    expected=np.array(original.resize((672,672),Image.Resampling.NEAREST));need(eq(arr[100:772,14:686],expected),'clean left panel exact display NN')
    return {'case_id':case['case_id'],'source':b,'output':bind(f),'clean_panel_byteparity':True}

def geometry_annotation(pid,r,rgb,folder):
    h,w=rgb.shape[:2];canvas=Image.new('RGB',(w,h+120),'white');canvas.paste(Image.fromarray(rgb),(0,120));d=ImageDraw.Draw(canvas);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    d.text((12,10),pid+' / first original-order geometry failure per case; no paste performed',fill='black',font=font)
    d.text((12,38),'RED full COVER bbox / BLUE full SHAM bbox / ORANGE end enclosure / YELLOW chosen witness',fill='black',font=font)
    d.text((12,66),'Original fullRGB unchanged. Display-only plan; out-of-frame portions clipped by display canvas only.',fill='black',font=font)
    p=r['placement_geometry'];b=p['cover_bbox_xyxy'];dx,dy=p['sham_translation_xy']
    for bb,c in ((b,'red'),([b[0]+dx,b[1]+dy,b[2]+dx,b[3]+dy],'blue')):d.rectangle((bb[0],bb[1]+120,bb[2],bb[3]+120),outline=c,width=3)
    for bb in r['endpoint_observation']['candidate_enclosures_xyxy']:d.rectangle((bb[0],bb[1]+120,bb[2],bb[3]+120),outline='orange',width=2)
    for v in r['endpoint_observation']['surrounding_material_witnesses']:
        x,y=v['raw_pixel_xy'];y+=120;d.ellipse((x-5,y-5,x+5,y+5),outline='yellow',width=2)
    f=folder/(pid+'_PLAN_ONLY.png');canvas.save(f);return bind(f)

def run(trigger):
    proto=jread(PROTOCOL);need(proto['code']==bind(Path(__file__)),'frozen auditor source')
    need(bind(trigger)==proto['root_actual_generation_entry'],'exact actual root entry')
    e=jread(trigger);need(e['approved_actual_source_real_end_compositing'] is True and e['automatic_retry'] is False,'authorized root scope')
    complete_path=GEN/'COMPLETE.json';need(sha(complete_path)==proto['actual_complete_expected_sha256'],'closed generation trigger')
    done=jread(complete_path);need(done['completed'] and done['PASS'],'actual completed census')
    gp=checked(e['protocol']);src=checked(gp['source_plan']);obs=lines(e['direct_end_observations']);om={r['case_id']:r for r in obs}
    frames={r['frame_index']:r for r in src['frames']};attempts=src['attempts'];original_plan=checked(proto['original_observation_plan']);cm={r['case_id']:r for r in original_plan['cases22']}
    need(len(attempts)==66 and len(om)==22 and len(frames)==21,'all original source population')
    rows=lines(done['manifest']);need([x['pair_id'] for x in rows]==[x['pair_id'] for x in attempts],'all66 exact order')
    whitelist={b['path']:b['sha256'] for b in e['source_asset_whitelist']};initial={};source={};donors={};fp={};aud=[];status=[];packets=0;slotU=0;selectedU=0;geovis=[];geocases=set()
    out=HERE/'actual_saved_real_review_v1';out.mkdir(exist_ok=False);vis=out/'source22_annotation';vis.mkdir();geov=out/'geometry_failure_annotation';geov.mkdir()
    def load_source(b,keys=None):
        need(whitelist.get(b['path'])==b['sha256'],'exact source/donor whitelist only');initial[b['path']]=b['sha256'];return arrays(b,keys)
    for x in obs:
        fr=frames[x['frame_index']];k=x['native_slot'];ep=x['endpoint_observation']
        if not ep['queried_end_localization_supported']:
            fp[x['case_id']]={'status':'NO_WEAK_LOCAL_END','locality_detail':'NONLOCAL_VISUAL_END' if x['no_local_opportunity_due_to_outside_original_crop'] else 'PART_IDENTITY_AMBIGUITY'};continue
        pts=enclosure(ep['candidate_enclosures_xyxy']);s,by=footprint(fr['query_xy'][k],fr['L'][k],fr['raw_hw']);miss=pts-s;xs,ys=zip(*s)
        beyond={p for p in miss if not(min(xs)<=p[0]<=max(xs) and min(ys)<=p[1]<=max(ys))};holes=miss-beyond
        fp[x['case_id']]={'status':'FULL_POSITIVE_RAW_SUPPORT' if not miss else 'INTERNAL_DOWNSAMPLING_SUPPORT_HOLES' if not beyond else 'OUTSIDE_OR_INVALID_SUPPORT_EXTENT','candidate_raw_pixels':len(pts),'missing_raw_pixels':len(miss),'outside_support_extent_pixels':len(beyond),'internal_hole_pixels':len(holes),'L':fr['L'][k],'grid_step':fr['L'][k]/96,'support_extent_xyxy':[min(xs),min(ys),max(xs),max(ys)]}
    for original,r in zip(attempts,rows):
        pid=r['pair_id'];cid=original['case_id'];fr=frames[original['frame_index']];k=original['native_slot'];ep=om[cid]['endpoint_observation'];record={'pair_id':pid,'checks':{}}
        def check(name,v):need(v,pid+': '+name);record['checks'][name]=True
        proof_path=GEN/pid/'PROOF.json';check('perpair_JSON_equals_manifest',jread(proof_path)==r)
        check('case_donor_frame_native_exact',r['case_id']==cid and r['donor_id']==original['donor_id'] and r['old_donor_lineage']==original['donor'] and r['original_input']['native_slot']==k)
        check('frozen_endpoint_observation_exact',r['endpoint_observation']==ep)
        check('original_qL_ID_flags_exact',r['original_input']['query_xy']==fr['query_xy'][k] and r['original_input']['L']==fr['L'][k] and r['original_input']['original_row_id']==fr['query_id'][k] and r['original_input']['original_global_index']==fr['global_index'][k] and r['original_input']['original_flags']==fr['original_query_input_metadata'][k] and r['original_input']['original_query_available']==fr['native_continuous_valid'][k])
        check('native_quality_owner_unknown_and_no_training_admission',r['new_native_quality_owner_all8']==-1 and r['training_admission'] is False)
        available={name:b for name,b in r['variants'].items() if 'path' in b}
        for name in NAMES:status.append({'pair_id':pid,'case_id':cid,'variant':name,'constructed_packet':name in available,'original_native_rows':8,'original_input_U':sum(not v for v in fr['native_continuous_valid']),'direct_end_weak_state':r['direct_end_weak_states']['SHAM' if name=='ADJACENT_SHAM' else name]})
        if not available:
            check('no_input_created_for_UNKNOWN',not r['constructed'] and not r['PASS'] and all(v=='UNKNOWN' for v in r['direct_end_weak_states'].values()));aud.append(record);continue
        i=fr['frame_index']
        if i not in source:
            rgb=load_source(fr['canonical_RGB_NPY']);fact=load_source(fr['existing_factual96_packet'],('rgb96','valid96','crop_constructed','global_index'))
            need(rgb.dtype==np.uint8 and list(rgb.shape)==fr['raw_hw']+[3] and ash(rgb)==fr['decoded_RGB_sha256'],'canonical decoded hash')
            source[i]=(rgb,fact)
        rgb,fact=source[i];pack={}
        for name,b in available.items():
            need(Path(b['path']).is_relative_to(GEN/pid),'generated packet exact own attempt');a=arrays(b);pack[name]=a;packets+=1;slotU+=int((~a['query_available']).sum())
            need(a['query_available'].dtype==bool and a['rgb96'].dtype==np.uint8 and a['valid96'].dtype==bool,'packet input dtype')
            expected={'query_xy':np.array(fr['query_xy'],np.float64),'L':np.array(fr['L'],np.float64),'native_slot':np.arange(8,dtype=np.int8),'query_id':np.array(fr['query_id']),'original_global_index':np.array(fr['global_index'],np.int32),'query_available':np.array(fr['native_continuous_valid'],bool),'native_continuous_valid':np.array(fr['native_continuous_valid'],bool),'nearest_valid':np.array(fr['nearest_valid'],bool),'crop_constructed':np.array(fr['feature_constructed'],bool)}
            for key in ('no_detection','finite_non_sentinel','query_RGB96_centre_domain_valid','native_grid_any_valid'):expected['original_'+key]=np.array([z[key] for z in fr['original_query_input_metadata']],bool)
            check(name+'_all8_original_metadata_exact',all(eq(a[key],val) for key,val in expected.items()))
            check(name+'_all8_allnewtargets_minus1',all(a[t].dtype==np.int8 and a[t].shape==(8,) and (a[t]==-1).all() for t in ('native_visibility_target','point_quality_target','owner_target')))
            crop=np.zeros((8,96,96,3),np.uint8);valid=np.zeros((8,96,96),bool)
            for n in range(8):
                if fr['feature_constructed'][n]:crop[n],valid[n]=local_sample(a['full_RGB'],fr['query_xy'][n],fr['L'][n])
            check(name+'_all8_RGB96_valid96_independent_F64_replay',eq(crop,a['rgb96']) and eq(valid,a['valid96']))
            check(name+'_central32_exact_slice',eq(a['central32'],a['rgb96'][:,32:64,32:64]))
            ctx,cv=context_sample(a['full_RGB'],fr['context_box_xyxy']);fra=np.array([[np.float32(cv.mean()),np.float32(valid[n].mean()),np.float32(valid[n,32:64,32:64].mean())] for n in range(8)],np.float32)
            check(name+'_context224_valid_fractions_exact_replay',eq(a['target224'],ctx) and eq(a['target_valid'],cv) and eq(a['fractions'],fra))
            check(name+'_center4_and_context_diagnostics_exact',eq(a['central4_available'],valid[:,47:49,47:49].all((1,2))) and eq(a['context_available'],np.full(8,cv.any(),bool)))
        before=pack['BEFORE'];check('BEFORE_fullRGB_original_exact',eq(before['full_RGB'],rgb));check('BEFORE_factual_all8_raw_valid_construct_indices_exact',eq(before['rgb96'],fact['rgb96']) and eq(before['valid96'],fact['valid96']) and eq(before['crop_constructed'],fact['crop_constructed']) and np.array_equal(fact['global_index'],fr['global_index']))
        check('BEFORE_full8_original_fact_array_hashes',all(ash(before['rgb96'][n])==fr['facts'][n]['RGB96_HWC_sha256'] and ash(before['valid96'][n])==fr['facts'][n]['valid96_sha256'] for n in range(8)) and ash(before['target224'])==fr['facts'][0]['target224_F32_HWC_sha256'])
        check('BEFORE_alpha_empty',not before['pasted_alpha'].any())
        did=r['donor_id']
        if did not in donors:donors[did]=load_source(original['donor']['compact_RGBA_cutout'],('rgb','alpha'))
        donor=donors[did];need(ash(donor['rgb'])==original['donor']['array_proof']['rgb']['array_sha256'] and ash(donor['alpha'])==original['donor']['array_proof']['alpha']['array_sha256'],'original donor array SHA')
        pts=enclosure(ep['candidate_enclosures_xyxy']);xs,ys=zip(*pts);bbox=[min(xs),min(ys),max(xs)+1,max(ys)+1]
        radius=float(distance_transform_edt(np.pad(donor['alpha'],1))[1:-1,1:-1].max());required=.5*math.hypot(bbox[2]-bbox[0]-1,bbox[3]-bbox[1]-1)+4
        size=max(1,math.ceil(max(donor['alpha'].shape)*required/(radius-1)));td=arrays(r['transformed_donor']);dr,da=resized_donor(donor['rgb'],donor['alpha'],size)
        check('frozen_once_resize_and_alpha_erode_exact',eq(dr,td['rgb']) and eq(da,td['alpha']) and size==r['scale_geometry']['longside_pixels'])
        dist=distance_transform_edt(np.pad(da,1))[1:-1,1:-1];ay,ax=np.unravel_index(int(dist.argmax()),dist.shape);center=[(bbox[0]+bbox[2]-1)/2,(bbox[1]+bbox[3]-1)/2];co=np.rint(np.array(center)-[ax,ay]).astype(int);ch,cw=da.shape;direction=1 if center[0]<=(fr['raw_hw'][1]-1)/2 else -1;dx=int(bbox[2]+4-co[0]) if direction==1 else int(bbox[0]-4-(co[0]+cw));so=co+[dx,0]
        pg=r['placement_geometry'];check('frozen_single_COVER_SHAM_placement_exact',pg['cover_origin_xy']==co.tolist() and pg['sham_origin_xy']==so.tolist() and pg['sham_translation_xy']==[dx,0])
        s,_=footprint(fr['query_xy'][k],fr['L'][k],fr['raw_hw']);cb=[int(co[0]),int(co[1]),int(co[0]+cw),int(co[1]+ch)];sb=[cb[0]+dx,cb[1],cb[2]+dx,cb[3]]
        inside=lambda point,b:b[0]<=point[0]<b[2] and b[1]<=point[1]<b[3]
        hw=fr['raw_hw'];gw=[w['raw_pixel_xy'] for w in ep['surrounding_material_witnesses']]
        gc={'cover_bbox_contains_all_end_support':all(inside(pt,cb) for pt in pts),'sham_bbox_disjoint_all_end_support':all(not inside(pt,sb) for pt in pts),'cover_and_sham_in_frame':all(0<=b[0]<b[2]<=hw[1] and 0<=b[1]<b[3]<=hw[0] for b in (cb,sb)),'same_bbox_and_positive_mass_translation':dx!=0,'sham_has_some_original_rgb96_support':any(inside(pt,sb) for pt in s),'certified_surrounding_part_retained_by_both_bbox_transforms':bool(gw) and all(tuple(pt) in s and tuple(pt) not in pts and not inside(pt,cb) and not inside(pt,sb) for pt in gw)}
        check('independent_frozen_geometry_controls_equal',gc==r['finite_end_geometry']['geometry_controls'])
        if not r['constructed']:
            check('geometry_failure_retains_BEFORE_only_and_UNKNOWN',not all(gc.values()) and set(pack)=={'BEFORE'} and r['status']=='GEOMETRY_FAILURE_RETAINED' and all(v=='UNKNOWN' for v in r['direct_end_weak_states'].values()))
            if cid not in geocases:geovis.append({'case_id':cid,'pair_id':pid,'image':geometry_annotation(pid,r,rgb,geov)});geocases.add(cid)
            aud.append(record);continue
        check('constructed_all_three_variants',set(pack)==set(NAMES));cover=pack['COVER'];sham=pack['ADJACENT_SHAM'];ca,sa=raw_mask(fr['raw_hw'],da,co),raw_mask(fr['raw_hw'],da,so)
        pixel={}
        pixel['end_every_candidate_raw_tap_COVER_alpha1']=all(ca[y,x] for x,y in pts)
        pixel['end_every_candidate_raw_tap_SHAM_alpha0']=all(not sa[y,x] for x,y in pts)
        pixel['equal_positive_pasted_alpha_mass']=int(ca.sum())==int(sa.sum())>0
        pixel['outside_paste_fullRGB_exact']=eq(cover['full_RGB'][~ca],rgb[~ca]) and eq(sham['full_RGB'][~sa],rgb[~sa])
        pixel['same_transformed_RGB_alpha_pasted_at_both_locations']=True
        for a,m,origin in ((cover,ca,co),(sham,sa,so)):
            pixel['same_transformed_RGB_alpha_pasted_at_both_locations'] &= eq(a['pasted_alpha'],m) and eq(a['full_RGB'][origin[1]:origin[1]+ch,origin[0]:origin[0]+cw][da],dr[da])
        s,by=footprint(fr['query_xy'][k],fr['L'][k],fr['raw_hw']);sample=np.array([bool(x&pts) for x in by]).reshape(96,96)
        pixel['end_support_actual_original96_bytes_changed_COVER']=bool(sample.any() and np.any(cover['rgb96'][k][sample]!=before['rgb96'][k][sample]))
        pixel['end_support_actual_original96_bytes_unchanged_SHAM']=eq(sham['rgb96'][k][sample],before['rgb96'][k][sample])
        pixel['SHAM_donor_positive_original96_RGB_support']=any(sa[y,x] for x,y in s)
        pixel['chosen_surrounding_samepart_RGB_unchanged_in_both']=bool(ep['surrounding_material_witnesses']) and all(not ca[w['raw_pixel_xy'][1],w['raw_pixel_xy'][0]] and not sa[w['raw_pixel_xy'][1],w['raw_pixel_xy'][0]] and eq(rgb[w['raw_pixel_xy'][1],w['raw_pixel_xy'][0]],cover['full_RGB'][w['raw_pixel_xy'][1],w['raw_pixel_xy'][0]]) and eq(rgb[w['raw_pixel_xy'][1],w['raw_pixel_xy'][0]],sham['full_RGB'][w['raw_pixel_xy'][1],w['raw_pixel_xy'][0]]) for w in ep['surrounding_material_witnesses'])
        for key,val in pixel.items():check('independent_pixel_'+key,bool(val)==r['pixel_checks'][key])
        check('producer_PASS_matches_independent_all_pixel_checks',r['PASS']==all(pixel.values()))
        check('weak_states_only_after_actualproof',r['direct_end_weak_states']==({'BEFORE':'VISIBLE_WEAK','COVER':'EXTERNALLY_BLOCKED_WEAK','SHAM':'VISIBLE_WEAK'} if r['PASS'] else {'BEFORE':'UNKNOWN','COVER':'UNKNOWN','SHAM':'UNKNOWN'}))
        for name,b in r['clean_selected96'].items():
            need(sha(b['path'])==b['sha256'],'clean PNG file SHA')
            with opened(b['path']) as f:
                with Image.open(f) as im:a=np.array(im)
            check(name+'_clean_savedPNG_selected96_byteparity',eq(a,pack[name]['rgb96'][k]) and ash(a)==b['decoded_array_sha256'])
        record['independent_pixel_checks']={k:bool(v) for k,v in pixel.items()};aud.append(record)
        print(json.dumps({'audited_pair':pid,'all66_index':len(aud),'constructed':True}),flush=True)
    need(len(status)==198 and len(aud)==66,'full census')
    for p,h in initial.items():need(sha(p)==h,'original source unchanged after audit')
    visual=[marked_source(case,om[case['case_id']],vis) for case in original_plan['cases22']]
    contacts=[]
    for start in range(0,22,2):
        sheet=Image.new('RGB',(1394,1680),'white')
        for j,vr in enumerate(visual[start:start+2]):
            with Image.open(vr['output']['path']) as im:sheet.paste(im,(0,840*j))
        f=vis/('CONTACT_%02d.png'%(start//2+1));sheet.save(f);contacts.append({'case_ids':[x['case_id'] for x in visual[start:start+2]],'image':bind(f)})
    write(out/'SOURCE22_DISPLAY_MANIFEST.json',{'completed':True,'rows':visual,'contacts':contacts,'first_geometry_failure_per_case':geovis,'actual_tool_view':'PENDING'})
    statuscount=Counter(r['status'] for r in rows);supportcount=Counter(x['status'] for x in fp.values());need(sum(r['PASS'] for r in rows)==done['supported_weak_end_pairs'],'complete count')
    report={'completed':True,'PASS':True,'source_only':True,'actual_generation_COMPLETE':bind(complete_path),'root_generation_entry':bind(trigger),'auditor':bind(Path(__file__)),'protocol':bind(PROTOCOL),
        'counts':{'cases':22,'attempts':66,'variant_status_rows':198,'potential_all8_native_rows':1584,'actual_variant_packets':packets,'actual_native_input_rows':packets*8,'actual_original_input_U_rows':slotU,'constructed_pairs':sum(r['constructed'] for r in rows),'supported_weak_pairs':sum(r['PASS'] for r in rows),'actual_source_frames_independently_opened':len(source),'assigned_donors_in_original66':len({x['donor_id'] for x in attempts}),'actual_opened_assigned_donors_independently_compared':len(donors)},
        'attempt_status_counts':dict(statuscount),'finite_end_support_status_counts':dict(supportcount),'per_case_independent_positive_raw_support':fp,
        'all66_independent_checks':aud,'all198_variant_statuses':status,
        'all_original_assets_SHA_unchanged_after_audit':[{'path':p,'sha256':h} for p,h in initial.items()],
        'actual_source_visual_render':{'all22_source_panels_created':True,'manifest':bind(out/'SOURCE22_DISPLAY_MANIFEST.json'),'directory':str(vis),'actual_tools_view_image':'PENDING_SEPARATE_RECORDED_VISUAL_QA','display_only_not_model_inputs':True},
        'new_native_quality_owner_targets_all_minus1':True,'training_admission':False,'new_models_or_forward_fit_or_labels':0,
        'limitations':['Numerical proof only. Witness material/part semantics require separate actual marked-source visual scrutiny; original frozen observation never edited.',
        'Full raw-tap candidate coverage is a conservative geometry gate; internal downsampling holes are not proof the end is visually outside the original crop.',
        'Original sampled-end semantics differ from literal query-center owner, broad part support and physical corner/visibility gold.','All66 failures and UNKNOWN retained; no retries/replacements/shrinking, no model/data selection, no private/held/VAL/V112 access.']}
    write(out/'REVIEW.json',report);print(json.dumps({'completed':True,'PASS':True,'review':bind(out/'REVIEW.json'),'counts':report['counts'],'statuses':dict(statuscount),'support':dict(supportcount)}),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--invented',action='store_true');ap.add_argument('--trigger',type=Path);args=ap.parse_args()
    if args.invented:print(json.dumps({'completed':True,'PASS':True,'checks':invented(),'actual_asset_reads':0}))
    else:
        need(args.trigger is not None,'root trigger required')
        try:run(args.trigger)
        except Exception:
            p=HERE/'ACTUAL_AUDIT_FAILURE.json'
            if not p.exists():write(p,{'completed':False,'PASS':False,'traceback':traceback.format_exc(),'automatic_retry':False})
            raise
