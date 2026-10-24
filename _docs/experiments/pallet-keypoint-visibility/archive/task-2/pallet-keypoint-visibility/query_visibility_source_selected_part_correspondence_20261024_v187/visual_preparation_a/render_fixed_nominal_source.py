"""Four predeclared source frames, original q crops, separate nominal overlays.
Preparation reads sealed metadata only; pixel/NPZ reads require a root entry.
"""
from pathlib import Path
import argparse, ast, hashlib, html, json, math, os, stat, traceback
import numpy as np
from PIL import Image, ImageDraw, ImageFont

H=Path(__file__).absolute().parent;V=H.parent;P=V.parent
PROJECTION=V/'nominal_source_native_projection_v1/COMPLETE.json'
REG_SCOPE=P/'query_visibility_public_source_registration_probe_20261024_v186/ROOT_SCOPE_BEFORE_SOURCE_RGB_DEPTH_MASK_POSE_MESH_BODIES.json'
C_COMPLETE=V/'selected_instance_metadata_diagnostic_v2_nested_R/COMPLETE.json'
CROP=P/'query_visibility_public_cue_inputs_20261023_v90/producer/spatial_input_math.py'
CROP_SHA='6a6c2543e28715e3d47e15c543c25f75ff7d3efa3b881e523bd3451817fc004b'
PLAN=H/'FIXED4_FRAMES_32_ORIGINAL_CROPS_METADATA_ONLY.json'
PIXEL_SCOPE=H/'PROSPECTIVE_PIXEL_SCOPE_BEFORE_SOURCE_PIXELS.json'
PROTOCOL=H/'PROTOCOL_BEFORE_FIXED_SOURCE_VISUAL_PIXELS.json'
CHECKS=H/'GENERATED_INVENTED_VISUAL_CHECKS.json'
OUT=V/'fixed_source_nominal_visual_v1'
FIXED=[0,118,166,199]
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
COLORS=['#ffcc33','#ff66dd','#a3ff66','#ff9966','#d0b3ff']

def require(v,s):
    if not bool(v):raise AssertionError(s)
def permitted(path):
    p=Path(path)
    require(p.is_absolute()and '..'not in p.parts,'Explicit absolute path')
    require(all('v112'not in s.casefold()for s in p.parts),'Forbidden name before access')
    require(P in p.parents,'Project-only data path');return p
def safe_open(path):
    p=permitted(path);parts=p.relative_to(P).parts;fd=os.open(P,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for part in parts[:-1]:
            nd=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=nd
        final=os.open(parts[-1],os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
    finally:os.close(fd)
    require(stat.S_ISREG(os.fstat(final).st_mode),'Ordinary file only');return os.fdopen(final,'rb')
def sha(path):
    h=hashlib.sha256()
    with safe_open(path)as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def bind(path):return {'path':str(permitted(path)),'sha256':sha(path)}
def checked(b):
    require(isinstance(b.get('sha256'),str)and len(b['sha256'])==64,'SHA required')
    require(sha(b['path'])==b['sha256'],'Binding drift '+b['path']);return permitted(b['path'])
def read(path):
    with safe_open(path)as f:return json.load(f)
def lines(path):
    with safe_open(path)as f:return [json.loads(s)for s in f if s.strip()]
def write(path,value):
    with permitted(path).open('x')as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def font(size):return ImageFont.truetype(FONT,size)
def text(draw,xy,s,size=18,fill='#172b40',width=None):
    ft=font(size);box=draw.textbbox(xy,s,font=ft)
    if width is not None:require(box[2]-box[0]<=width,'Caption width: '+s)
    draw.text(xy,s,font=ft,fill=fill);return box
def fmt(x):return 'null'if x is None else f'{x:.3f}'
def cross(draw,xy,color,r=8):
    x,y=xy;draw.line((x-r,y,x+r,y),fill=color,width=2);draw.line((x,y-r,x,y+r),fill=color,width=2)
def zoom(a,n):return Image.fromarray(a).resize((a.shape[1]*n,a.shape[0]*n),Image.Resampling.NEAREST)

def load_crop():
    require(sha(CROP)==CROP_SHA,'Exact original V90 crop source')
    with safe_open(CROP)as f:tree=ast.parse(f.read().decode())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='crop'];require(len(nodes)==1,'Only original pure crop')
    ns={'np':np};exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(CROP),'exec'),ns)
    return ns['crop']

def render_crop(raw,valid,row):
    board=Image.new('RGB',(1280,650),'#f2f5f8');d=ImageDraw.Draw(board)
    k=row['native_slot'];fi=row['original_frame_index'];n=row['nominal_correspondence'];ref=n['nominal_reference'];dg=n['nominal_discrepancy_and_crop_diagnostic']
    text(d,(24,12),f'FRAME {fi:03d} | original global {row["original_global_index"]} | native {k}',28,width=1230)
    q=row['original_query_xy'];L=row['original_L']
    text(d,(24,55),f'Original q=({fmt(q[0])},{fmt(q[1])})  L={fmt(L)}  inputU={not row["original_query_available"]}  | weak selected GT={n["selected_gt_instance_index"]}, obj={n["selected_obj_id"]}',18,width=1230)
    for x,s in [(24,'Clean original RGB96'),(440,'Display copy: q + nominal ref'),(864,'Original valid96')]:text(d,(x,96),s,19,width=392)
    clean=zoom(raw,4);display=clean.copy();dd=ImageDraw.Draw(display)
    cross(dd,(191.5,191.5),'#00ffff',12)
    t=dg['reference_crop_index_xy']
    if t is not None:
        pt=(4*(t[0]+.5)-.5,4*(t[1]+.5)-.5)
        if -.5<=t[0]<95.5 and -.5<=t[1]<95.5:cross(dd,pt,'#ff9900',10)
    board.paste(clean,(24,128));board.paste(display,(440,128));board.paste(zoom((valid*255).astype(np.uint8),2).convert('RGB'),(864,128))
    text(d,(24,530),'Exact saved pixels; nearest 4x.',18,width=392)
    text(d,(440,530),'Cyan q at (47.5,47.5); orange ref.',17,width=392)
    text(d,(864,334),'White valid / black padding.',17,width=392)
    text(d,(864,366),'Center4 valid: '+str(int(valid[47:49,47:49].sum()))+'/4',18,width=392)
    text(d,(864,396),'Original U is not a D/E label.',17,width=392)
    text(d,(864,426),'Ref: '+('no obj1 lookup'if ref is None else ref['native_name']),16,width=392)
    text(d,(864,456),'Inside crop: '+str(dg['inside_outer_crop_extent']),17,width=392)
    text(d,(864,486),'Ref marker: '+('absent / outside'if t is None or not dg['inside_outer_crop_extent']else 'display copy only'),16,width=392)
    xy='null'if t is None else f'({t[0]:.3f},{t[1]:.3f})'
    text(d,(24,570),f'Nominal crop index={xy} | nominal discrepancy px={fmt(dg["nominal_distance_px"])} / L={fmt(dg["nominal_distance_over_L"])}',18,width=1230)
    text(d,(24,610),'Nominal geometry diagnostic only; no physical coordinate gold or native visibility / quality label.',18,width=1230)
    require(np.array_equal(np.asarray(board)[128:512,24:408],np.repeat(np.repeat(raw,4,0),4,1)),'Clean raw panel unchanged')
    return board

def render_overview(rgb,frame,instances):
    overlay=Image.fromarray(rgb).copy();d=ImageDraw.Draw(overlay);h,w=rgb.shape[:2]
    for j,g in enumerate(instances):
        color=COLORS[j%len(COLORS)];refs=g['references']
        for a,b in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]:
            p,q=refs[a]['nominal_reference_xy'],refs[b]['nominal_reference_xy']
            if p is not None and q is not None:d.line(tuple(p+q),fill=color,width=2)
        for r in refs:
            p=r['nominal_reference_xy']
            if p is not None and 0<=p[0]<w and 0<=p[1]<h:
                x,y=p;d.ellipse((x-5,y-5,x+5,y+5),outline=color,width=2)
                d.text((x+6,y-18),f'G{g["gt_instance_index"]}:{r["native_slot"]}',font=font(17),fill=color,stroke_width=2,stroke_fill='black')
    box=frame['original_selected_detection_bbox_xyxy']
    if box is not None:d.rectangle(box,outline='white',width=3)
    for k,q in enumerate(frame['original_q8']):
        if q is not None and len(q)==2 and all(x is not None and math.isfinite(x)for x in q)and 0<=q[0]<w and 0<=q[1]<h:
            cross(d,q,'#00ffff',7);d.text((q[0]+8,q[1]+4),'q'+str(k),font=font(19),fill='#00ffff',stroke_width=2,stroke_fill='black')
    board=Image.new('RGB',(w,944),'#f2f5f8');bd=ImageDraw.Draw(board)
    text(bd,(24,12),f'FRAME {frame["frame_index"]:03d} | {frame["source_id"]} | original canonical RGB + separate overlay',25,width=w-48)
    legend=' / '.join('GT'+str(g['gt_instance_index'])+' nominal8'for g in instances)
    text(bd,(24,55),legend+' | cyan=q8 | white=selected detector box',18,width=w-48)
    board.paste(overlay,(0,96))
    text(bd,(24,834),'All obj1 instances shown; selected detector index is not a GT-instance ID.',19,width=w-48)
    a=frame['diagnostic_association']
    text(bd,(24,867),f'Weak bbox association: GT={a["selected_gt_instance_index"]}, obj={a["selected_obj_id"]}. Original q/L unchanged.',19,width=w-48)
    text(bd,(24,900),'Nominal references are not visible-part / native D/E / physical corner gold.',19,width=w-48)
    return board,overlay

def invented():
    tests=[]
    def add(n,v):tests.append({'name':n,'PASS':bool(v)})
    raw=np.arange(96*96*3,dtype=np.uint8).reshape(96,96,3);valid=np.ones((96,96),bool);valid[:8]=False
    n={'selected_gt_instance_index':None,'selected_obj_id':7,'nominal_reference':None,'nominal_discrepancy_and_crop_diagnostic':{'reference_crop_index_xy':None,'inside_outer_crop_extent':None,'nominal_distance_px':None,'nominal_distance_over_L':None}}
    r={'native_slot':7,'original_frame_index':199,'original_global_index':1599,'original_query_xy':[None,None],'original_L':None,'original_query_available':False,'nominal_correspondence':n}
    b=render_crop(raw,valid,r);add('clean_raw_ROI_byteexact_with_unknown_U',np.array_equal(np.asarray(b)[128:512,24:408],np.repeat(np.repeat(raw,4,0),4,1)))
    add('valid_mask_distinct_padding',np.asarray(b)[129,865,0]==0 and np.asarray(b)[160,865,0]==255)
    n2=dict(n,nominal_reference={'native_name':'front_bottom_right'},nominal_discrepancy_and_crop_diagnostic={'reference_crop_index_xy':[47.5,47.5],'inside_outer_crop_extent':True,'nominal_distance_px':0.,'nominal_distance_over_L':0.})
    b2=render_crop(raw,valid,dict(r,nominal_correspondence=n2,original_query_xy=[1.25,2.5],original_L=96.))
    add('marker_only_in_display_copy',np.array_equal(np.asarray(b2)[128:512,24:408],np.asarray(b)[128:512,24:408])and not np.array_equal(np.asarray(b2)[128:512,440:824],np.asarray(b)[128:512,440:824]))
    add('nearest_zoom_alignment',4*(47.5+.5)-.5==191.5)
    crop=load_crop();rgb=np.full((40,50,3),137,np.uint8);a,v=crop(rgb,[2.25,3.5],16.,96)
    add('exact_original_sampler_boundary_and_dtype',a.shape==(3,96,96)and v.shape==(1,96,96)and a.dtype==np.uint8 and v.dtype==bool and not v.all())
    add('all32_selection_fixed_before_outcomes',len(FIXED)*8==32 and FIXED==[0,118,166,199])
    try:permitted(P/'blocked_v112'/'never_access');ok=False
    except AssertionError:ok=True
    add('forbidden_name_before_access',ok)
    return {'completed':True,'PASS':all(t['PASS']for t in tests),'invented_only':True,'checks':tests,'source_RGB_NPZ_or_pixel_hash_reads':0}

def prepare():
    pc=read(PROJECTION);cc=read(C_COMPLETE);reg=read(REG_SCOPE)
    require(pc['completed']and pc['PASS']and pc['C_complete']==bind(C_COMPLETE),'Closed metadata source')
    frames=lines(checked(cc['outputs']['ALL185_FRAME_INSTANCE_ASSOCIATIONS.jsonl']));queries=lines(checked(pc['outputs']['queries']));instances=lines(checked(pc['outputs']['instances']))
    fi={r['frame_index']:r for r in frames};assets={r['frame_index']:r for r in reg['fixed_frames_before_body_reads']}
    plan=[]
    for i in FIXED:
        f=fi[i];a=assets[i];rr=[r for r in queries if r['original_frame_index']==i];gg=[g for g in instances if g['frame_index']==i]
        require(f['split']==a['internal_split']=='SOURCE_DEVELOPMENT'and f['source_id']==a['source_id'],'Exact four source frames')
        require([r['native_slot']for r in rr]==list(range(8))and [g['gt_instance_index']for g in gg]==a['all_obj1_GT_instance_indices'],'All8/all obj1, no selection')
        plan.append({'frame':f,'RGB_input':a['canonical_RGB_NPY'],'original_encoded_RGB_provenance_only':a['RGB'],'queries':rr,'instances':gg})
    write(PLAN,{'frames':plan,'fixed_indices':FIXED,'rows':32,'original_inputU':sum(not r['original_query_available']for x in plan for r in x['queries']),'selection':'Same four predeclared V186 first-source frames; all8 slots and every obj1 instance. No outcomes/labels used.'})
    pixels={'prepared_only':True,'approved_actual_pixel_reads':False,'plan':bind(PLAN),'selection':FIXED,'allowed_actual_pixel_inputs':[{'frame_index':x['frame']['frame_index'],'canonical_RGB_NPY':x['RGB_input'],'original_factual96_NPZ':x['frame']['original_factual96_packet']}for x in plan],
            'only_after_root_entry':'Hash/load exactly4 canonical RGB NPY and4 saved RGB96/valid96 packets. No depth, masks, mesh, upstream GT bodies, model/feature/score/evaluation assets.',
            'input_preservation':'V90 exact original sampler AST must replay all32 original raw96/valid96 bytes. Original U remains separate; markers only display copies. No GT-centered crop, fallback instance, changed q/L/native or new labels/admission.',
            'outputs':'Four full canonical clean RGB + four nominal/q8 overview cards, all32 clean raw96+valid PNG and32 display cards, four contact sheets, offline HTML/portableMD and manifest. Actual tool-view QA separate.'}
    write(PIXEL_SCOPE,pixels);tests=invented();tests['code']=bind(__file__);write(CHECKS,tests);require(tests['PASS'],'Invented visual checks')
    proto={'completed':True,'prepared_only':True,'code':bind(__file__),'plan':bind(PLAN),'pixel_scope':bind(PIXEL_SCOPE),'checks':bind(CHECKS),'source_projection_complete':bind(PROJECTION),'source_C_complete':bind(C_COMPLETE),'V186_scope':bind(REG_SCOPE),'source_crop_math':bind(CROP),
           'metadata_sources':{'frames':cc['outputs']['ALL185_FRAME_INSTANCE_ASSOCIATIONS.jsonl'],'queries':pc['outputs']['queries'],'instances':pc['outputs']['instances']},'fixed_frames':FIXED,'original_native_rows':32,'output':str(OUT),'automatic_retry':False,'new_labels_or_training_admission':False,'raw_pixels_read_in_prepare':0,'HTML_browser_rendered':False}
    write(PROTOCOL,proto)
    template={'approved_actual_source_nominal_visualization':False,'source_TRAIN_only':True,'code':proto['code'],'protocol':bind(PROTOCOL),'plan':proto['plan'],'pixel_scope':proto['pixel_scope'],'checks':proto['checks'],'output':str(OUT),'model_or_feature_or_evaluation_or_GT_mask_depth_mesh_reads':False,'new_labels_or_training_admission':False,'automatic_retry':False}
    write(H/'ROOT_ACTUAL_VISUAL_ENTRY_TEMPLATE_APPROVED_FALSE.json',template)
    ready={'completed':True,'prepared_only':True,'code':proto['code'],'protocol':bind(PROTOCOL),'plan':proto['plan'],'pixel_scope':proto['pixel_scope'],'checks':proto['checks'],'template':bind(H/'ROOT_ACTUAL_VISUAL_ENTRY_TEMPLATE_APPROVED_FALSE.json'),'fixed_frames':FIXED,'rows':32,'actual_source_pixel_NPZ_or_pixel_hash_reads':0,'actual_render_calls':0,'command':'/home/minjae/anaconda3/envs/pallet-yolo26/bin/python '+str(Path(__file__).absolute())+' --entry <root-entry>'}
    write(H/'PREPARATION_COMPLETE.json',ready);print(json.dumps(ready),flush=True)

def run(entry):
    e=read(entry);require(e['approved_actual_source_nominal_visualization']is True and e['source_TRAIN_only']is True,'Root pixel entry required')
    for k in ('model_or_feature_or_evaluation_or_GT_mask_depth_mesh_reads','new_labels_or_training_admission','automatic_retry'):require(e[k]is False,'Forbidden action '+k)
    p=read(checked(e['protocol']));require(e['protocol']==bind(PROTOCOL)and e['code']==p['code']==bind(__file__),'Exact frozen renderer')
    for k in ('plan','pixel_scope','checks'):require(e[k]==p[k],'Frozen '+k);checked(p[k])
    for k in ('source_projection_complete','source_C_complete','V186_scope','source_crop_math'):checked(p[k])
    for b in p['metadata_sources'].values():checked(b)
    plan=read(checked(p['plan']));require(plan['fixed_indices']==FIXED and plan['rows']==32,'Fixed all32')
    require(e['output']==p['output']==str(OUT),'New fixed output');OUT.mkdir(exist_ok=False)
    write(OUT/'STARTED.json',{'completed':False,'entry':bind(entry),'automatic_retry':False})
    try:
        crop=load_crop();records=[];used=[];body=['<html><meta charset="utf-8"><title>Fixed source nominal correspondence</title><h1>Fixed four SOURCE frames / all32 original crops</h1><p>Weak identity and nominal geometry only. Original q/L/U unchanged; no native D/E/quality gold. Browser rendering not verified.</p>'];md=['# 고정 SOURCE 명목 대응 시각화','원래 q/L의 전체32 crop과 모든 obj1 instance의 명목 reference. 약한 identity 진단이며 가시성·좌표 gold가 아닙니다.']
        for x in plan['frames']:
            f=x['frame'];fi=f['frame_index'];directory=OUT/f'FRAME_{fi:03d}';directory.mkdir()
            with safe_open(checked(x['RGB_input']))as fd:rgb=np.load(fd,allow_pickle=False)
            require(rgb.dtype==np.uint8 and rgb.shape==tuple(f['raw_hw'])+(3,),'Original canonical RGB uint8/HWC')
            with safe_open(checked(f['original_factual96_packet']))as fd:
                with np.load(fd,allow_pickle=False)as z:packet={k:z[k].copy()for k in z.files}
            require(set(packet)=={'rgb96','valid96','global_index','crop_constructed'},'Factual96 input-only keys')
            require(packet['rgb96'].shape==(8,96,96,3)and packet['rgb96'].dtype==np.uint8 and packet['valid96'].shape==(8,96,96)and packet['valid96'].dtype==bool,'Original all8 crop schema')
            require(packet['global_index'].tolist()==f['original_global_indices'],'Original global IDs')
            Image.fromarray(rgb).save(directory/'ORIGINAL_CANONICAL_RGB.png')
            overview,overlay=render_overview(rgb,f,x['instances']);overview.save(directory/'OVERVIEW.png');overlay.save(directory/'DISPLAY_ONLY_NOMINAL_OVERLAY.png')
            body.append(f'<h2>FRAME {fi:03d}</h2><img width="1280" src="FRAME_{fi:03d}/OVERVIEW.png">');md.extend([f'## FRAME {fi:03d}',f'![overview](FRAME_{fi:03d}/OVERVIEW.png)'])
            contact=Image.new('RGB',(1280,1300),'#f2f5f8')
            for k,r in enumerate(x['queries']):
                raw=packet['rgb96'][k];valid=packet['valid96'][k];constructed=bool(packet['crop_constructed'][k])
                require(constructed==r['original_input_flags']['feature_constructed'],'Original constructed flag')
                if constructed:
                    a,v=crop(rgb,r['original_query_xy'],r['original_L'],96);require(np.array_equal(a.transpose(1,2,0),raw)and np.array_equal(v[0],valid),'All32 original sampler byte parity')
                else:require(not raw.any()and not valid.any(),'Unconstructed original placeholder unchanged')
                require(ash(raw)==r['saved_rgb96_sha256']and ash(valid)==r['saved_valid96_sha256'],'Original row pixel hashes')
                rawfile=directory/f'NATIVE_{k}_RAW96.png';validfile=directory/f'NATIVE_{k}_VALID96.png';cardfile=directory/f'NATIVE_{k}_CARD.png'
                Image.fromarray(raw).save(rawfile);Image.fromarray((valid*255).astype(np.uint8)).save(validfile);card=render_crop(raw,valid,r);card.save(cardfile)
                require(np.array_equal(np.asarray(Image.open(rawfile)),raw)and np.array_equal(np.asarray(Image.open(validfile))>0,valid),'Saved clean pixel parity')
                contact.paste(card.resize((640,325),Image.Resampling.LANCZOS),((k%2)*640,(k//2)*325))
                records.append({'original_global_index':r['original_global_index'],'frame_index':fi,'native_slot':k,'original_q':r['original_query_xy'],'original_L':r['original_L'],'original_query_available':r['original_query_available'],'original_input_flags':r['original_input_flags'],'original_RGB96_sha256':ash(raw),'original_valid96_sha256':ash(valid),'sampler_replay_byteexact':True,'images':{'raw96':bind(rawfile),'valid96':bind(validfile),'card':bind(cardfile)},'nominal_correspondence':r['nominal_correspondence'],'new_labels_or_training_admission':False})
                body.append(f'<h3>original global {r["original_global_index"]} / native{k}</h3><img width="1280" src="FRAME_{fi:03d}/NATIVE_{k}_CARD.png">');md.append(f'![original global {r["original_global_index"]} native{k}](FRAME_{fi:03d}/NATIVE_{k}_CARD.png)')
            contact.save(directory/'ALL8_CONTACT.png');used.extend([x['RGB_input'],f['original_factual96_packet']])
            require(np.array_equal(np.asarray(Image.open(directory/'ORIGINAL_CANONICAL_RGB.png')),rgb),'Saved full canonical RGB parity')
        for b in used:checked(b)
        write(OUT/'ALL32_VISUAL_MANIFEST.json',records);(OUT/'index.html').write_text('\n'.join(body+['</html>']));(OUT/'PORTABLE_GALLERY_KO.md').write_text('\n\n'.join(md)+'\n')
        for r in records:
            for b in r['images'].values():checked(b)
        done={'completed':True,'PASS':True,'entry':bind(entry),'code':bind(__file__),'protocol':bind(PROTOCOL),'plan':bind(PLAN),'pixel_scope':bind(PIXEL_SCOPE),'fixed_frames':FIXED,'original_native_rows':len(records),'original_inputU':sum(not r['original_query_available']for r in records),'all32_original_RGB96_valid96_sampler_and_cleanPNG_byteparity':True,'full_canonical_RGB_cleanPNG_byteparity':True,'actual_source_pixel_bindings_unchanged':used,'manifest':bind(OUT/'ALL32_VISUAL_MANIFEST.json'),'index':bind(OUT/'index.html'),'portable_gallery':bind(OUT/'PORTABLE_GALLERY_KO.md'),'actual_tool_view_QA':'pending separate view_image receipt','HTML_browser_rendered':False,'new_labels_or_training_admission':False,'selected_model':None,'goal_solved':False,'automatic_retry':False}
        write(OUT/'COMPLETE.json',done);print(json.dumps({'completed':True,'PASS':True,'complete':bind(OUT/'COMPLETE.json')}),flush=True)
    except Exception:
        write(OUT/'FAILED_EXECUTION_PRESERVED.json',{'completed':False,'automatic_retry':False,'entry':bind(entry),'traceback':traceback.format_exc()});raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--entry',type=Path);args=ap.parse_args()
    if args.prepare:prepare()
    elif args.entry:run(args.entry)
    else:ap.error('--prepare or --entry required')
