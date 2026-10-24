"""Diagnostic selectedbox+one-span context, not a model crop/input replacement."""
from pathlib import Path
import hashlib,json,math
import numpy as np
from PIL import Image,ImageDraw,ImageFont
H=Path(__file__).resolve().parent;P=H.parents[1];V141=P/'query_visibility_public_native_initializer_association_20261024_v141'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(p),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def check(b):assert sha(b['path'])==b['sha256'];return Path(b['path'])
def font(n):return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',n)
seal=read(V141/'actual_public_initializer_canonical_transport_v2/PREDICTION_SEAL.json');outputs={Path(b['path']).name:b for b in seal['outputs']}
result=read(check(outputs['RESULT.json']));transport=read(check(result['canonical_transport_seal']));manifest=read(check(transport['manifest']))['rows']
association=read(check(read(V141/'saved_public_association_v1/ASSOCIATION_COMPLETE.json')['results']))
out=H/'four_selected_entity_zooms_v1';assert not out.exists();out.mkdir();records=[]
for i in(2,3,8,10):
    f=association['frame_diagnostics'][i];m=manifest[i];rgb=np.load(check(m['canonical_RGB_NPY']),allow_pickle=False)
    assert hashlib.sha256(rgb.tobytes()).hexdigest()==m['decoded_RGB_sha256']
    h,w=rgb.shape[:2];box=np.array(f['selected_bbox_xyxy']);bw,bh=box[2:]-box[:2]
    context=[max(0,math.floor(box[0]-bw)),max(0,math.floor(box[1]-bh)),min(w,math.ceil(box[2]+bw)),min(h,math.ceil(box[3]+bh))]
    x0,y0,x1,y1=context;raw=Image.fromarray(rgb[y0:y1,x0:x1]);marked=raw.copy();d=ImageDraw.Draw(marked)
    d.rectangle(tuple(box-np.array([x0,y0,x0,y0])),outline=(255,145,0),width=2)
    anns=[]
    for a in f['official_instance_diagnostics']:
        b=np.array(a['official_bbox_xyxy'])
        if min(x1,b[2])<=max(x0,b[0])or min(y1,b[3])<=max(y0,b[1]):continue
        d.rectangle(tuple(b-np.array([x0,y0,x0,y0])),outline=(0,230,80),width=1)
        d.text((max(0,b[0]-x0),max(0,b[1]-y0)),str(a['annotation_id']),font=font(12),fill=(0,230,80),stroke_width=1,stroke_fill='black');anns.append(a['annotation_id'])
    for q in f['queries']:
        p=q['query_xy'];k=q['native_slot']
        if p is None or any(v is None for v in p):continue
        x,y=p[0]-x0,p[1]-y0
        if 0<=x<raw.width and 0<=y<raw.height:
            d.ellipse((x-2,y-2,x+2,y+2),outline=(255,0,210),width=1)
            d.text((min(max(0,x+3),raw.width-15),min(max(0,y),raw.height-14)),str(k),font=font(12),fill=(255,0,210),stroke_width=1,stroke_fill='black')
    scale=min(960/raw.width,650/raw.height);sz=(round(raw.width*scale),round(raw.height*scale));canvas=Image.new('RGB',(1920,sz[1]+125),'white');cd=ImageDraw.Draw(canvas)
    cd.text((12,8),f['id']+' | input selectedbox + one bbox span each side | diagnostic NEAREST only',font=font(22),fill='black')
    cd.text((12,42),'Left raw pixels; right same pixels with original q8/selectedbox and ALL intersecting official boxes. No target re-selection.',font=font(19),fill='black')
    cd.text((12,72),'Canonical RGB, context '+str(context)+' | this display zoom is NOT RGB96/model input or a corrected coordinate/reference.',font=font(18),fill='black')
    canvas.paste(raw.resize(sz,Image.Resampling.NEAREST),(0,115));canvas.paste(marked.resize(sz,Image.Resampling.NEAREST),(960,115))
    path=out/f'FRAME_{i:02d}_INPUT_BOX_CONTEXT_RAW_AND_Q.png';canvas.save(path);records.append(dict(frame_index=i,source_frame=f['id'],selected_bbox=f['selected_bbox_xyxy'],context_unexpanded_source_box_one_span_eachside_xyxy=context,all_intersecting_official_ids=anns,figure=bind(path),diagnostic_only=True,query_L_or_native_order_change=False))
with(out/'ZOOM_FACTS.json').open('x')as f:json.dump(dict(completed=True,rows=records,model_crop_replacement=False,GTcenter=False,querycorrection=False,model_forward_fit_private_V112_PnP=0),f,indent=2)
print(json.dumps(dict(completed=True,output=str(out),frames=4)))
