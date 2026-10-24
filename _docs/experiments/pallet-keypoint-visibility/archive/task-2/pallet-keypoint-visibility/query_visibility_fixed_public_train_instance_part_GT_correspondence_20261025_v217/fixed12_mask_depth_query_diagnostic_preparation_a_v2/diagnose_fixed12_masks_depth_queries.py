#!/usr/bin/env python3
"""Future root-gated SOURCE candidate-mask/depth diagnostics; never native labels."""
import argparse
import collections
import csv
import hashlib
import io
import json
import math
import os
import stat
import numpy as np
from PIL import Image,ImageDraw

D='/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/query_visibility_fixed_public_train_instance_part_GT_correspondence_20261025_v217'
H=D+'/fixed12_mask_depth_query_diagnostic_preparation_a_v2'
OUT=D+'/fixed12_mask_depth_query_diagnostics_v2'
PRIOR=D+'/fixed12_GT_payload_preparation_a'
PIXEL_KINDS={'FULL_INSTANCE_MASK_PNG','VISIBLE_INSTANCE_MASK_PNG','DEPTH_PNG'}

def need(x,msg):
    if not x:raise ValueError(msg)

def opened(path,flags=os.O_RDONLY):
    need(isinstance(path,str) and 'v112' not in path.lower(),'Forbidden lexical path before FS')
    need(os.path.isabs(path) and os.path.normpath(path)==path,'Normalized absolute path')
    fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY)
    try:
        for c in path.strip('/').split('/')[:-1]:
            n=os.open(c,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
        r=os.open(path.rsplit('/',1)[1],flags|os.O_NOFOLLOW,0o644,dir_fd=fd)
        if not stat.S_ISREG(os.fstat(r).st_mode):os.close(r);raise ValueError('Regular file required')
        return r
    finally:os.close(fd)

def raw(path):
    with os.fdopen(opened(path),'rb') as f:b=f.read(67108865)
    need(len(b)<=67108864,'Byte cap');return b

def sha(b):return hashlib.sha256(b).hexdigest()
def binding(p,b):return {'path':p,'sha256':sha(b),'size_bytes':len(b)}
def checked(b):
    v=raw(b['path']);need(sha(v)==b['sha256'],'SHA mismatch')
    if b.get('size_bytes') is not None:need(len(v)==b['size_bytes'],'Size mismatch')
    return v

def save(name,b):
    p=OUT+'/'+name
    with os.fdopen(opened(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL),'wb') as f:f.write(b)
    return binding(p,b)

def savej(name,x):return save(name,(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
def savecsv(name,rows):
    s=io.StringIO(newline='');w=csv.DictWriter(s,fieldnames=list(rows[0]));w.writeheader()
    for row in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})
    return save(name,s.getvalue().encode())

def new_output():
    need('v112' not in OUT.lower(),'Forbidden output')
    fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY)
    try:
        for c in D.strip('/').split('/'):
            n=os.open(c,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
        os.mkdir(OUT.rsplit('/',1)[1],dir_fd=fd)
    finally:os.close(fd)

def decode_png(data):
    with Image.open(io.BytesIO(data)) as im:
        need(im.format=='PNG','PNG required');a=np.asarray(im).copy()
    need(a.ndim==2 and np.issubdtype(a.dtype,np.number),'Single-channel numeric PNG required')
    return a

def scale_from_camera(camera):
    x=camera.get('depth_scale') if isinstance(camera,dict) else None
    return float(x) if type(x) in (float,int) and math.isfinite(x) and x>0 else None

def candidate_summary(full,visible,depth,depth_scale):
    need(full.ndim==visible.ndim==depth.ndim==2,'2D arrays required')
    need(full.shape==visible.shape==depth.shape,'DIMENSION_MISMATCH_NO_RESIZE_OR_REPAIR')
    f=full>0;v=visible>0;dv=np.isfinite(depth)&(depth>0)
    fa=int(f.sum());va=int(v.sum());outside=int((v&~f).sum())
    return {'height':int(f.shape[0]),'width':int(f.shape[1]),'full_area':fa,'visible_area':va,'visible_outside_full_area':outside,'visible_subset_full':outside==0,'visible_full_area_ratio':va/fa if fa else None,'depth_positive_finite_pixels':int(dv.sum()),'depth_zero_or_invalid_pixels':int((~dv).sum()),'full_mask_positive_depth_pixels':int((f&dv).sum()),'visible_mask_positive_depth_pixels':int((v&dv).sum()),'full_mask_positive_depth_fraction':float((f&dv).sum()/fa) if fa else None,'visible_mask_positive_depth_fraction':float((v&dv).sum()/va) if va else None,'depth_scale':depth_scale,'metric_depth_status':'BOP_DEPTH_SCALE_AVAILABLE_MM_PER_RAW_UNIT' if depth_scale is not None else 'UNKNOWN_CAMERA_DEPTH_SCALE_UNAVAILABLE','full_dtype':str(full.dtype),'visible_dtype':str(visible.dtype),'depth_dtype':str(depth.dtype)}

def query_diagnostic(q,available,full,visible,depth,depth_scale):
    base={'measure':'ORIGINAL_RAW_IMAGE_NEAREST_FLOOR_Q_PLUS_0_5_AND_CLIPPED_3X3_NOT_RAW96_SAMPLER','original_inputU':not available,'query_xy_unchanged':q,'finite_original_q':False,'nearest_x':None,'nearest_y':None,'original_q_in_continuous_image_canvas':None,'nearest_pixel_in_frame':False,'full_mask_at_nearest':None,'visible_mask_at_nearest':None,'raw_depth_at_nearest':None,'positive_finite_depth_at_nearest':None,'scaled_depth_mm_at_nearest':None,'window3x3_in_frame_samples':0,'window3x3_full_pixels':None,'window3x3_visible_pixels':None,'window3x3_positive_depth_pixels':None,'new_native_visibility_label':'UNKNOWN','training_admission':False}
    if q is None or len(q)!=2 or not all(type(x) in (int,float) and math.isfinite(x) for x in q):return base
    x,y=float(q[0]),float(q[1]);h,w=full.shape
    ix,iy=math.floor(x+.5),math.floor(y+.5)
    base.update(finite_original_q=True,nearest_x=ix,nearest_y=iy,original_q_in_continuous_image_canvas=(-.5<=x<w-.5 and -.5<=y<h-.5),nearest_pixel_in_frame=(0<=ix<w and 0<=iy<h))
    if not base['nearest_pixel_in_frame']:return base
    z=float(depth[iy,ix]);positive=math.isfinite(z) and z>0
    x0,x1=max(0,ix-1),min(w,ix+2);y0,y1=max(0,iy-1),min(h,iy+2)
    dz=depth[y0:y1,x0:x1]
    base.update(full_mask_at_nearest=bool(full[iy,ix]>0),visible_mask_at_nearest=bool(visible[iy,ix]>0),raw_depth_at_nearest=z if math.isfinite(z) else None,positive_finite_depth_at_nearest=positive,scaled_depth_mm_at_nearest=z*depth_scale if positive and depth_scale is not None else None,window3x3_in_frame_samples=int(dz.size),window3x3_full_pixels=int((full[y0:y1,x0:x1]>0).sum()),window3x3_visible_pixels=int((visible[y0:y1,x0:x1]>0).sum()),window3x3_positive_depth_pixels=int((np.isfinite(dz)&(dz>0)).sum()))
    return base

def gate_pixels(records):
    need(len(records)==42 and len({r['path'] for r in records})==42,'Exact42 unique PNGs')
    need(collections.Counter(r['kind'] for r in records)=={'FULL_INSTANCE_MASK_PNG':15,'VISIBLE_INSTANCE_MASK_PNG':15,'DEPTH_PNG':12},'Exact15full15vis12depth')
    for r in records:
        need('v112' not in r['path'].lower(),'Forbidden path before FS')
        need(r['kind'] in PIXEL_KINDS and isinstance(r['sha256'],str) and len(r['sha256'])==64,'Only known rawSHA PNG records')

def GT_info_context(info,height,width):
    out={'GT_info_context_only':info,'GT_bbox_obj_oob_status':'UNKNOWN_METADATA_MISSING','global_visib_fract_is_not_native_occlusion':True}
    box=info.get('bbox_obj') if isinstance(info,dict) else None
    if isinstance(box,list) and len(box)==4 and all(type(x) in (int,float) and math.isfinite(x) for x in box):
        x,y,w,h=box
        if w<0 or h<0:out['GT_bbox_obj_oob_status']='NEGATIVE_BBOX_SENTINEL_NO_GEOMETRIC_SUPPORT'
        if w>=0 and h>=0:
            out['GT_bbox_obj_oob_status']='BBOX_PARTLY_OUTSIDE' if x<0 or y<0 or x+w>width or y+h>height else 'BBOX_WITHIN_CANVAS'
            out['GT_bbox_obj_clipped_area_fraction']=max(0,min(width,x+w)-max(0,x))*max(0,min(height,y+h)-max(0,y))/(w*h) if w*h else None
    return out


def panel(full,visible,depth,rows,title,context):
    need(full.shape==visible.shape==depth.shape==context.shape[:2] and context.ndim==3 and context.shape[2]==3,'No source alignment resizing/repair')
    h,w=full.shape;pw,ph=480,420
    canvas=Image.new('RGB',(pw*4,ph+105),'white');draw=ImageDraw.Draw(canvas)
    draw.text((14,8),title,fill='black')
    variants=[(None,'ORIGINAL derived context RGB'),(full>0,'FULL candidate overlay'),(visible>0,'VISIBLE candidate overlay'),(np.isfinite(depth)&(depth>0),'DEPTH positive/finite')]
    for col,(mask,label) in enumerate(variants):
        if col==3:src=np.repeat((mask.astype(np.uint8)*255)[:,:,None],3,axis=2)
        else:
            src=context.copy()
            if mask is not None:
                tint=np.array([255,80,40] if col==1 else [0,220,150],np.float64)
                src[mask]=np.rint(src[mask].astype(np.float64)*(159/255)+tint*(96/255)).astype(np.uint8)
        scale=min((pw-20)/w,(ph-35)/h);sw=max(1,int(round(w*scale)));sh=max(1,int(round(h*scale)))
        im=Image.fromarray(src).resize((sw,sh),Image.Resampling.NEAREST)
        ox=col*pw+(pw-sw)//2;oy=48+(ph-35-sh)//2
        canvas.paste(im,(ox,oy));draw.text((col*pw+12,29),label,fill='black')
        for row in rows:
            q=row['query_xy']
            if q is None or len(q)!=2 or not all(type(v) in (int,float) and math.isfinite(v) for v in q):continue
            x,y=q
            if not (-.5<=x<w-.5 and -.5<=y<h-.5):continue
            sx=ox+(x+.5)*sw/w-.5;sy=oy+(y+.5)*sh/h-.5
            color='#ff8c00' if not row['original_query_available'] else '#00bcd4'
            draw.line((sx-4,sy,sx+4,sy),fill=color,width=2);draw.line((sx,sy-4,sx,sy+4),fill=color,width=2)
            draw.text((sx+5,sy-10),str(row['native_slot'])+('U' if not row['original_query_available'] else ''),fill=color)
    draw.text((12,ph+51),'SOURCE candidate overlays; original derived PNG/q/native/U. Display only; not RAW96 sampling.',fill='black')
    draw.text((12,ph+70),'Global visibility fraction/OOB/depth support are separate; no automatic GT-best instance or native labels.',fill='black')
    f=io.BytesIO();canvas.save(f,format='PNG');return f.getvalue()


def run(entry_path):
    need(entry_path.startswith(D+'/') and 'v112' not in entry_path.lower(),'Exact new-phase entry')
    entry=json.loads(raw(entry_path));need(entry.get('approved_actual_fixed12_mask_depth_diagnostic') is True,'New root actual entry required')
    need(entry.get('approved_exact12_derived_context_PNG') is True,'Exact12 context PNG approval required')
    need(entry.get('training_admission') is False and entry.get('output_directory')==OUT,'No admission; fresh fixed output')
    need(entry['protocol']['path']==H+'/PROTOCOL_BEFORE_MASK_DEPTH_PIXEL_READS.json','Exact protocol')
    need(entry['payload_whitelist']['path']==PRIOR+'/EXACT_PUBLIC_GT_PAYLOAD_WHITELIST.json','Exact known whitelist')
    need(entry['fixed12_frames']['path']==PRIOR+'/FIXED12_TRAIN_FRAME_AND_INSTANCE_CANDIDATE_METADATA.json','Exact fixed frames')
    need(entry['original96_inputs']['path']==PRIOR+'/ALL96_ORIGINAL_INPUTS_AND_GT_POINTERS.json','Exact original inputs')
    need(entry['selected_GT_complete']['path']==D+'/fixed12_selected_GT_metadata_v1/COMPLETE.json','Exact generated selected GT completion')
    need(entry['context_PNG_selection']['path']==H+'/EXACT12_ORIGINAL_DERIVED_CONTEXT_PNG_BINDINGS.json','Exact12 context selection')
    need(entry['code']['path']==os.path.abspath(__file__),'Exact reader');checked(entry['code']);checked(entry['protocol'])
    wl=json.loads(checked(entry['payload_whitelist']));frames=json.loads(checked(entry['fixed12_frames']))['frames12']
    original=json.loads(checked(entry['original96_inputs']))['rows96']
    joined=json.loads(checked(entry['V216_joined96']))['rows96'];semantic=json.loads(checked(entry['V216_root_semantic']))['rows96']
    gtc=json.loads(checked(entry['selected_GT_complete']));need(gtc['completed'] and gtc['PASS'],'Selected GT reader completed PASS')
    gtf=json.loads(checked(gtc['selected_GT_metadata']))['frames12']
    gf={f['frame_position']:f for f in gtf};jr={r['input']['row_id']:r for r in joined};sem={r['row_id']:r for r in semantic}
    need(len(gf)==12 and set(gf)==set(range(12)),'All selected GT frames unique')
    need(len(frames)==12 and len(original)==len(jr)==len(sem)==96,'Fixed12/all96')
    need(sum(not r['original_query_available'] for r in original)==11,'Original U11')
    need(collections.Counter(r['internal_source_role'] for r in original)=={'SOURCE_FIT':64,'SOURCE_DEV':32},'Original internal roles')
    for r in original:
        i=jr[r['row_id']]['input'];s=sem[r['row_id']]
        need(r['original_role']==i['role']==s['original_role']=='TRAIN','Outer TRAIN before pixels')
        for k in ['native_slot','query_xy','L','original_query_available','component_id','internal_source_role']:
            need(r[k]==i[k],'Original input drift: '+k)
        need(r['native_slot']==s['native_slot'] and r['internal_source_role']==s['internal_source_role'],'Root weak identity')
    contexts=json.loads(checked(entry['context_PNG_selection']))['frames12']
    need(len(contexts)==12 and len({x['binding']['path'] for x in contexts})==12,'Exact12 unique derived context PNGs')
    context_by_position={x['frame_position']:x['binding'] for x in contexts}
    for r in original:
        candidates=[b for b in jr[r['row_id']]['display_bindings']['full_context'] if b['path'].endswith('/ORIGINAL_CANONICAL_RGB.png')]
        need(len(candidates)==1 and candidates[0]['path']==context_by_position[r['frame_position']]['path'] and candidates[0]['sha256']==context_by_position[r['frame_position']]['sha256'],'Context binding mismatch; no RGB fallback')
    records=[r for r in wl['payloads'] if r['kind'] in PIXEL_KINDS];gate_pixels(records)
    data={};byte_QA=[]
    for r in records:
        b=checked(r);data[r['path']]=b;byte_QA.append(binding(r['path'],b))
    context_data={pos:checked(b) for pos,b in context_by_position.items()}
    new_output()
    context_copies=[{'frame_position':pos,'image':save('FRAME_%02d_ORIGINAL_CONTEXT.png'%pos,b)} for pos,b in sorted(context_data.items())]
    opaque=savej('BEFORE_PIXEL_DECODE_OPAQUE_SHA_QA.json',{'completed':True,'PASS':True,'files42':byte_QA,'context12':[binding(context_by_position[pos]['path'],b) for pos,b in sorted(context_data.items())],'decoded_images_before_receipt':0})
    candidate_rows=[];query_rows=[];panels=[];failures=[]
    for frame in frames:
        pos=frame['frame_position'];rr=[r for r in original if r['frame_position']==pos]
        need(len(rr)==8 and sorted(r['native_slot'] for r in rr)==list(range(8)),'Exact group8')
        g=gf[pos];need(g['frame_key']==frame['frame_key'] and g['original_role']=='TRAIN' and g['internal_source_role']==frame['internal_source_role'],'Selected GT frame mismatch')
        need([x['gt_list_index'] for x in g['obj1_candidates']]==frame['obj1_candidate_gt_list_indices'],'All source candidates unchanged')
        scale=scale_from_camera(g.get('camera'));depth=decode_png(data[frame['depth']['path']])
        with Image.open(io.BytesIO(context_data[pos])) as image:
            need(image.format=='PNG' and image.mode=='RGB','Original derived RGB PNG mode only')
            context=np.asarray(image).copy()
        for ix in frame['obj1_candidate_gt_list_indices']:
            paths=frame['candidate_masks'][str(ix)]
            full=decode_png(data[paths['full_mask']['path']]);vis=decode_png(data[paths['visible_mask']['path']])
            base={'frame_position':pos,'frame_key':frame['frame_key'],'component_id':frame['component_id'],'internal_source_role':frame['internal_source_role'],'outer_role':'TRAIN','candidate_gt_list_index':ix,'selected_instance_correspondence':'UNKNOWN','native_visibility_admission':False}
            try:summary=candidate_summary(full,vis,depth,scale)
            except ValueError as e:
                status='FAILED_DIMENSIONS_PRESERVED';summary={'diagnostic_error':str(e),'full_shape':list(full.shape),'visible_shape':list(vis.shape),'depth_shape':list(depth.shape)};failures.append({**base,**summary})
            else:
                if context.shape[:2]!=full.shape:
                    status='DIAGNOSTIC_COMPLETE_CONTEXT_DIMENSIONS_UNALIGNED';summary['context_shape']=list(context.shape);failures.append({**base,**summary})
                else:status='DIAGNOSTIC_COMPLETE_NO_LABEL'
            info=next(x for x in g['obj1_candidates'] if x['gt_list_index']==ix).get('GT_info')
            summary.update(GT_info_context(info,full.shape[0],full.shape[1]))
            candidate_rows.append({**base,'diagnostic_status':status,**summary})
            for r in rr:
                s=sem[r['row_id']];j=jr[r['row_id']]
                q=query_diagnostic(r['query_xy'],r['original_query_available'],full,vis,depth,scale) if status.startswith('DIAGNOSTIC') else {'measure':'NOT_COMPUTED_DIMENSION_MISMATCH','original_inputU':not r['original_query_available'],'new_native_visibility_label':'UNKNOWN','training_admission':False}
                query_rows.append({**base,'row_id':r['row_id'],'native_slot':r['native_slot'],'query_xy':r['query_xy'],'L':r['L'],'original_query_available':r['original_query_available'],'original_flags':r['original_flags'],'diagnostic_status':status,'saved_V216_root_weak_part_state':s['root_weak_part_state'],'A_ROI_raw96':j['reader_A_axes']['witness']['roi'],'C_ROI_raw96':j['reader_C_axes']['witness']['roi'],'ROI_support_sampling':'NOT_COMPUTED_NO_RAW96_MASK_SAMPLER_REPLAY',**q})
            if status.startswith('DIAGNOSTIC') and context.shape[:2]==full.shape:
                image=save('FRAME_%02d_CANDIDATE_%06d_MASKS_DEPTH_VALID.png'%(pos,ix),panel(full,vis,depth,rr,'Frame %02d | candidate GT list %d | %s'%(pos,ix,frame['internal_source_role']),context))
                panels.append({**base,'image':image})
    need(len(candidate_rows)==15 and len(query_rows)==120,'All15 candidates x8 rows retained')
    # Different dimension failures retain the same CSV columns through explicit union.
    def rectangular(rows):
        keys=list(dict.fromkeys(k for r in rows for k in r));return [{k:r.get(k) for k in keys} for r in rows]
    qa=savecsv('ALL120_CANDIDATE_NATIVE_QUERY_DIAGNOSTICS.csv',rectangular(query_rows))
    ca=savecsv('ALL15_CANDIDATE_MASK_DEPTH_DIAGNOSTICS.csv',rectangular(candidate_rows))
    allj=savej('ALL_DIAGNOSTIC_ROWS.json',{'completed':True,'candidate_rows15':candidate_rows,'candidate_native_rows120':query_rows,'dimension_failures_preserved':failures})
    gallery=['# 고정 SOURCE 후보 mask/depth 진단','', '저장된 원 derived context PNG 위에 full/visible mask를 표시합니다. 원 q/native/U를 유지하며 모든 후보 인스턴스 대응·새 native D/E는 UNKNOWN입니다. 표시 확대·축소만 사용하고 RAW96 sampler를 재현한 화면이 아닙니다.','']
    for p in context_copies:
        gallery+=['## Frame %02d original context'%p['frame_position'],'![]('+p['image']['path'].rsplit('/',1)[1]+')','']
    for p in panels:
        name=p['image']['path'].rsplit('/',1)[1];gallery+=['## Frame %02d / candidate %d'%(p['frame_position'],p['candidate_gt_list_index']),'![]('+name+')','']
    ga=save('PORTABLE_MASK_DEPTH_GALLERY.md','\n'.join(gallery).encode())
    sm=savej('SUMMARY.json',{'completed':True,'frames':12,'original_native_rows':96,'original_inputU':11,'candidate_instances':15,'candidate_native_rows':120,'rows_with_original_U':sum(r['original_inputU'] for r in query_rows),'candidate_dimension_failures':len(failures),'panels_generated':len(panels),'metric_depth_camera_available_frames':sum(scale_from_camera(g.get('camera')) is not None for g in gtf),'selected_instance_correspondence':'UNKNOWN_ALL_CANDIDATES','all_new_native_visibility_labels':'UNKNOWN','training_admission':False,'query_measure':'nearest raw-image pixel floor(q+.5) and clipped 3x3; NOT RAW96 bilinear sampling','mask_absence_or_query_foreign_is_not_E':True,'no_ROI_mapping_no_native_projection_no_q_change':True})
    c=savej('COMPLETE.json',{'completed':True,'PASS':True,'entry':binding(entry_path,raw(entry_path)),'code':entry['code'],'protocol':entry['protocol'],'selected_GT_complete':entry['selected_GT_complete'],'opaque_SHA_QA':opaque,'summary':sm,'all_rows':allj,'candidate_CSV':ca,'query_CSV':qa,'gallery':ga,'panels':panels,'original_context12':context_copies,'dimension_failures_preserved':len(failures),'source_mask_depth_files_decoded':42,'derived_context_PNGs_decoded':12,'original_JPEG_RGB_NPY_CAD_NPZ_model_projection_PnP_or_fit_reads':0,'training_admission':False})
    print(json.dumps(c,ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--entry',required=True);run(p.parse_args().entry)
