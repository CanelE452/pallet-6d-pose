#!/usr/bin/env python3
"""Display-only fixed4 original-query/crop vs saved nominal reference cards."""
import argparse,hashlib,io,json,math,os,stat
from PIL import Image,ImageDraw

D='/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/query_visibility_fixed_public_train_instance_part_GT_correspondence_20261025_v217'
H=D+'/fixed_query_crop_nominal_reference_visual_preparation_a'
OUT=D+'/fixed_query_crop_nominal_reference_visual_v1'
CASE_KEYS=((0,0),(0,1),(6,3),(10,0))
CYAN='#00bfd7';MAGENTA='#e100bc'

def need(v,msg):
    if not v:raise ValueError(msg)
def opened(p,flags=os.O_RDONLY):
    need(isinstance(p,str) and 'v112' not in p.lower(),'Forbidden lexical path before FS')
    need(os.path.isabs(p) and os.path.normpath(p)==p,'Normalized absolute path')
    fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY)
    try:
        for c in p.strip('/').split('/')[:-1]:
            n=os.open(c,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
        r=os.open(p.rsplit('/',1)[1],flags|os.O_NOFOLLOW,0o644,dir_fd=fd)
        if not stat.S_ISREG(os.fstat(r).st_mode):os.close(r);raise ValueError('Regular file required')
        return r
    finally:os.close(fd)
def raw(p):
    with os.fdopen(opened(p),'rb') as f:return f.read()
def digest(b):return hashlib.sha256(b).hexdigest()
def binding(p,b):return {'path':p,'sha256':digest(b),'size_bytes':len(b)}
def checked(b):
    need(isinstance(b.get('sha256'),str) and len(b['sha256'])==64,'Exact SHA binding before FS')
    z=raw(b['path']);need(digest(z)==b['sha256'],'SHA mismatch')
    if b.get('size_bytes') is not None:need(len(z)==b['size_bytes'],'Size mismatch')
    return z
def save(name,z):
    p=OUT+'/'+name
    with os.fdopen(opened(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL),'wb') as f:f.write(z)
    return binding(p,z)
def savej(name,obj):return save(name,(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
def mkdir_output():
    need('v112' not in OUT.lower(),'Forbidden output')
    fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY)
    try:
        for c in D.strip('/').split('/'):
            n=os.open(c,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
        os.mkdir(OUT.rsplit('/',1)[1],dir_fd=fd)
    finally:os.close(fd)

def finite_xy(p):return isinstance(p,(list,tuple)) and len(p)==2 and all(type(x) in (int,float) and math.isfinite(x) for x in p)
def crop_reference(ref,q,L):
    if not finite_xy(ref) or not finite_xy(q) or type(L) not in (float,int) or not math.isfinite(L) or L<=0:return None
    return [(float(ref[i])-float(q[i]))*(96.0/float(L))+47.5 for i in range(2)]
def in_canvas(p,w,h):return finite_xy(p) and -.5<=p[0]<w-.5 and -.5<=p[1]<h-.5
def display_coordinate(p,scale,offset):
    scales=[scale,scale] if isinstance(scale,(int,float)) else scale
    return [offset[i]+(p[i]+.5)*scales[i]-.5 for i in range(2)]
def boundary_arrow(p,w,h,inset=5):
    """Only a display endpoint. The returned coordinate never replaces p/input q."""
    if not finite_xy(p):return None
    cx,cy=(w-1)/2,(h-1)/2;dx,dy=p[0]-cx,p[1]-cy
    if dx==0 and dy==0:return [cx,cy]
    ts=[]
    if dx:ts.append(((w-1-inset if dx>0 else inset)-cx)/dx)
    if dy:ts.append(((h-1-inset if dy>0 else inset)-cy)/dy)
    t=min(x for x in ts if x>=0)
    return [cx+t*dx,cy+t*dy]
def cross(draw,p,color,label):
    x,y=p;draw.line((x-5,y,x+5,y),fill=color,width=2);draw.line((x,y-5,x,y+5),fill=color,width=2)
    draw.text((x+6,y-12),label,fill=color)
def marker(draw,p,w,h,offset,scale,color,label):
    if not finite_xy(p):return 'UNKNOWN'
    if in_canvas(p,w,h):cross(draw,display_coordinate(p,scale,offset),color,label);return 'IN_CANVAS'
    endpoint=boundary_arrow(p,w,h)
    disp=display_coordinate(endpoint,scale,offset)
    center=display_coordinate([(w-1)/2,(h-1)/2],scale,offset)
    dx,dy=disp[0]-center[0],disp[1]-center[1];norm=math.hypot(dx,dy) or 1
    start=[disp[0]-14*dx/norm,disp[1]-14*dy/norm]
    draw.line((*start,*disp),fill=color,width=3)
    draw.text((disp[0]+3,disp[1]-10),label+' OUT',fill=color)
    return 'OUTSIDE_DISPLAY_ARROW_ONLY'

def png_image(z,size=None):
    with Image.open(io.BytesIO(z)) as im:
        need(im.format=='PNG' and im.mode=='RGB','Original derived RGB PNG only')
        need(size is None or im.size==size,'Original PNG dimensions')
        return im.copy()

def make_card(case,context,crops):
    # No original pixel array/crop is changed: overlays are on the display canvas.
    canvas=Image.new('RGB',(1600,1610),'white');draw=ImageDraw.Draw(canvas)
    draw.text((20,16),'SOURCE fixed frame %02d / GT candidate %d'%(case['frame_position'],case['candidate_gt_list_index']),fill='black')
    draw.text((20,38),'CYAN original q / MAGENTA saved NOMINAL reference; unknown physical native gold; no recenter, no native labels.',fill='black')
    w,h=case['source_hw'][1],case['source_hw'][0];need(context.size==(w,h),'Context/source dimensions match')
    scale=min(1480/w,700/h);sw,sh=int(round(w*scale)),int(round(h*scale));ox=(1600-sw)//2;oy=70
    canvas.paste(context.resize((sw,sh),Image.Resampling.NEAREST),(ox,oy))
    # Actual rounded display dimensions get their own scale for pixel centers.
    scales=(sw/w,sh/h)
    source_notes=[]
    for r in case['rows8']:
        n=r['native_slot'];q=r['query_xy'];ref=r['nominal_reference_xy']
        qs=marker(draw,q,w,h,(ox,oy),scales,CYAN,'q'+str(n)+(' U' if not r['original_query_available'] else ''))
        rs=marker(draw,ref,w,h,(ox,oy),scales,MAGENTA,'r'+str(n))
        if rs!='IN_CANVAS':source_notes.append('r%d=%s %s'%(n,str(ref),rs))
        if qs!='IN_CANVAS':source_notes.append('q%d=%s %s'%(n,str(q),qs))
    y0=800
    for r in case['rows8']:
        n=r['native_slot'];col=n%4;row=n//4;tx=col*400+20;ty=y0+row*355
        draw.text((tx,ty),'native%d | input%s'%(n,' U (unchanged)' if not r['original_query_available'] else ' available'),fill='black')
        crop=crops[n];need(crop.size==(96,96),'Same original clean96')
        x=tx+90;y=ty+27;canvas.paste(crop.resize((192,192),Image.Resampling.NEAREST),(x,y))
        marker(draw,[47.5,47.5],96,96,(x,y),2,CYAN,'q')
        local=r['nominal_reference_raw96_xy'];state=marker(draw,local,96,96,(x,y),2,MAGENTA,'r'+str(n))
        draw.text((tx,ty+225),'q='+str(r['query_xy']),fill='black')
        draw.text((tx,ty+243),'L='+str(r['L']),fill='black')
        draw.text((tx,ty+261),'ref96='+('UNKNOWN' if local is None else '[%.3f, %.3f]'%tuple(local)),fill=MAGENTA)
        draw.text((tx,ty+279),state,fill=MAGENTA)
        draw.text((tx,ty+297),'Not GT-q error / not RAW96 replay proof',fill='black')
    draw.text((20,1525),'OUT arrows are clipped DISPLAY pointers; listed reference coordinates remain unchanged.',fill='black')
    for i,line in enumerate(source_notes[:3]):draw.text((20,1545+18*i),line,fill=MAGENTA)
    if len(source_notes)>3:draw.text((830,1545),'Additional outside/unknown coordinates: see saved JSON rows.',fill='black')
    f=io.BytesIO();canvas.save(f,format='PNG');return f.getvalue()

def validate_plan(plan):
    need([tuple(c['case_key']) for c in plan['cases4']]==list(CASE_KEYS),'Fixed4 case identity')
    bs=plan['unique_PNG_bindings27'];need(len(bs)==27 and len({b['path'] for b in bs})==27,'Unique27 only')
    bypath={b['path']:b['sha256'] for b in bs};used=set()
    for c in plan['cases4']:
        need(c['case_key']==[c['frame_position'],c['candidate_gt_list_index']],'Case identity')
        need(c['selected_gt_instance_index'] is None and c['training_admission'] is False,'No selected-instance/admission')
        need(len(c['source_hw'])==2 and all(type(x) is int and x>0 for x in c['source_hw']),'Saved source HW')
        need([r['native_slot'] for r in c['rows8']]==list(range(8)),'Original all8 native order')
        used.add(c['context']['path']);need(bypath.get(c['context']['path'])==c['context']['sha256'],'Context exact whitelist')
        for r in c['rows8']:
            need(type(r['original_query_available']) is bool,'Original availability bool')
            need(r['training_admission'] is False and r['physical_native_gold']=='UNKNOWN','No native promotion')
            local=r['nominal_reference_raw96_xy']
            if not r['original_query_available']:need(local is None,'InputU remains local UNKNOWN')
            elif local is not None:need(local==crop_reference(r['nominal_reference_xy'],r['query_xy'],r['L']),'Exact saved algebraic address')
            used.add(r['clean96']['path']);need(bypath.get(r['clean96']['path'])==r['clean96']['sha256'],'Clean96 exact whitelist')
    need(used==set(bypath),'No unused or undeclared PNG')
    return True

def run(entry_path):
    need(entry_path.startswith(D+'/') and 'v112' not in entry_path.lower(),'Exact phase entry')
    e=json.loads(raw(entry_path));need(e.get('approved_actual_fixed4_query_reference_cards') is True,'New root approval before PNG bodies')
    need(e.get('output_directory')==OUT and e.get('training_admission') is False,'Fresh fixed output/no admission')
    need(e['code']['path']==os.path.abspath(__file__) and e['protocol']['path']==H+'/PROTOCOL_BEFORE_FIXED4_PNG_BODIES.json','Exact renderer/protocol');checked(e['code']);checked(e['protocol'])
    plan=json.loads(checked(e['fixed_cases']));need([tuple(c['case_key']) for c in plan['cases4']]==list(CASE_KEYS),'Fixed a-priori4 cards')
    validate_plan(plan)
    need(e['input_metadata_bindings']==plan['input_metadata_bindings'],'Exact generated metadata closure')
    for b in plan['input_metadata_bindings'].values():checked(b)
    data={b['path']:checked(b) for b in plan['unique_PNG_bindings27']}
    mkdir_output();opaque=savej('BEFORE_IMAGE_DECODE_SHA_QA.json',{'completed':True,'PASS':True,'verified_PNG27':plan['unique_PNG_bindings27'],'decoded_before_receipt':0})
    decoded={b['path']:png_image(data[b['path']]) for b in plan['unique_PNG_bindings27']}
    copied={};contexts={}
    for c in plan['cases4']:
        f=c['frame_position']
        if f not in contexts:contexts[f]=save('FRAME_%02d_ORIGINAL_CONTEXT.png'%f,data[c['context']['path']])
        for r in c['rows8']:
            key=(f,r['native_slot'])
            if key not in copied:copied[key]=save('FRAME_%02d_NATIVE_%d_ORIGINAL_CLEAN96.png'%key,data[r['clean96']['path']])
    cards=[]
    for c in plan['cases4']:
        context=decoded[c['context']['path']]
        crops={r['native_slot']:decoded[r['clean96']['path']] for r in c['rows8']}
        z=make_card(c,context,crops);cb=save('FRAME_%02d_CANDIDATE_%06d_QUERY_REFERENCE_CARD.png'%tuple(c['case_key']),z)
        cards.append({'case_key':c['case_key'],'image':cb,'physical_native_gold':'UNKNOWN','training_admission':False})
    rows=savej('SAVED_FIXED4_DISPLAY_COORDINATES.json',{'completed':True,'cases4':plan['cases4'],'no_projection_recalculation':True,'no_q_or_source_pixels_changed':True})
    md=['# 고정4 original query/crop 대 nominal reference','', 'Cyan은 원q, magenta는 저장된 불확실한 nominal reference입니다. 범위 밖 화살표는 표시용이며 점을 옮기지 않습니다. native visibility/좌표 gold나 RAW96/control proof가 아닙니다.','']
    for c in cards:md+=['## frame%d / candidate%d'%tuple(c['case_key']),'![]('+c['image']['path'].rsplit('/',1)[1]+')','']
    gallery=save('PORTABLE_GALLERY_KO.md','\n'.join(md).encode())
    done=savej('COMPLETE.json',{'completed':True,'PASS':True,'entry':binding(entry_path,raw(entry_path)),'code':e['code'],'protocol':e['protocol'],'fixed_cases':e['fixed_cases'],'opaque_PNG_SHA_QA':opaque,'cards4':cards,'coordinates':rows,'original_clean96_copies24':list(copied.values()),'original_context_copies3':list(contexts.values()),'gallery':gallery,'original_pixel_inputs_unchanged':True,'nominal_projection_or_permutation_search':0,'new_native_labels_training_model_GT_or_sourcebinary_reads':0,'owned_derived_PNGs_decoded':27,'training_admission':False})
    print(json.dumps(done,ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--entry',required=True);run(p.parse_args().entry)
