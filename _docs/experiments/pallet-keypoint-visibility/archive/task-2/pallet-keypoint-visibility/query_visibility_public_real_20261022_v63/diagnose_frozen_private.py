"""Post-freeze descriptive crop/error audit. No model, fitting, or PnP imports."""
from pathlib import Path
import csv,hashlib,json
from collections import Counter
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent
OUT=ROOT/'diagnostics';OUT.mkdir(exist_ok=True)
PRIMARY='MIXED_CONTEXT_RGB_SAM_V4'
def rows(p):
    with Path(p).open(newline='')as f:return list(csv.DictReader(f))
def key(r):return r['id'],int(r['native_corner'])
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def outcsv(p,x):
    with Path(p).open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(x[0]));w.writeheader();w.writerows(x)

seal=json.loads((ROOT/'SOURCE_SELECTION_SEAL.json').read_text());assert seal['selected_primary']==PRIMARY
inf=json.loads((ROOT/'private_evaluation'/PRIMARY/'INFERENCE_SEAL.json').read_text());assert inf['new_optimizer_updates']==inf['new_PnP_calls']==0
manifest=rows(PARENT/'evaluation_preparation_20261022/PREPARED_QUERY_MANIFEST.csv')
scores={key(r):r for r in rows(ROOT/'private_evaluation'/PRIMARY/'REAL_QUERY_SCORES.csv')}
sam={key(r):r for r in rows(ROOT/'private_evaluation'/PRIMARY/'RAW_SAM_QUERY_SCORES.csv')}
near={key(r):r for r in rows(PARENT/'near_error_diagnosis_20261022/real/NEAR_DIRECT_ROWS_WITH_PROVENANCE.csv')}
maskrecords={r['id']:r for r in json.loads((ROOT/'private_evaluation'/PRIMARY/'SAM_TARGET_RECORDS.json').read_text())}
allrows=[]
for r in manifest:
    s=scores[key(r)];a=sam[key(r)];n=near.get(key(r),{})
    assert r['image_sha256']==s['image_sha256']==a['image_sha256']
    allrows.append({**r,'new_probability':s['p_visible'],'new_present':s['present_at_frozen_threshold'],
                    'SAM_raw_logit':a['raw_sam_logit'],'SAM_present':a['SAM_present_at_logit0'],'SAM_radius2_present':a['SAM_radius2_present_at_logit0'],
                    'geometry_self_proxy':s['geometry_self_hidden'],'annotation_coordinate_source':n.get('annotation_coordinate_source','NOT_IN_NEAR_DIRECT_COHORT')})
outcsv(OUT/'REAL_QUERY_DIAGNOSTIC_ROWS.csv',allrows)

summary=[]
for cohort in ['same_slot_pipeline','paired_same_query_RGB']:
 for group in ['WOOD_PRIMARY105','SQUARE118']:
  for scope in ['ALL_DIRECT','ERROR_LE3_DIRECT','ERROR_LE3_MANUAL_DIRECT','ERROR_LE3_SAME_NEAREST_PIXEL_DIRECT']:
   rr=[r for r in allrows if r['group']==group and r[cohort+'_D_vs_ES']=='True'and r['human_state']=='DIRECT_VISIBLE']
   if scope!='ALL_DIRECT':rr=[r for r in rr if float(r['initial_reference_error_px'])<=3]
   if scope=='ERROR_LE3_MANUAL_DIRECT':rr=[r for r in rr if r['annotation_coordinate_source']=='manual_click']
   if scope=='ERROR_LE3_SAME_NEAREST_PIXEL_DIRECT':rr=[r for r in rr if near[key(r)]['same_nearest_pixel']=='True']
   n=len(rr)
   for name,col in [('v61','present_at_fixed_0_5'),(PRIMARY,'new_present'),('SAM_EXACT','SAM_present'),('SAM_RADIUS2_SUPPORT','SAM_radius2_present')]:
    present=sum(r[col]=='True'for r in rr);summary.append(dict(cohort=cohort,group=group,scope=scope,model=name,N=n,pred_present=present,pred_absent=n-present,visible_recall=present/n if n else None))
outcsv(OUT/'DIRECT_LOCATION_AUDIT.csv',summary)

FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
font=ImageFont.truetype(FONT,18);small=ImageFont.truetype(FONT,15);titlefont=ImageFont.truetype(FONT,23)
def patch(image,q,side,n=300):
    im=np.asarray(image);h,w=im.shape[:2];axis=(np.arange(n)+.5)/n-.5
    xx,yy=np.meshgrid(q[0]+axis*side,q[1]+axis*side)
    px,py=np.floor(xx+.5).astype(int),np.floor(yy+.5).astype(int)
    valid=(px>=0)&(px<w)&(py>=0)&(py<h)
    a=im[np.clip(py,0,h-1),np.clip(px,0,w-1)].copy();a[~valid]=0
    return Image.fromarray(a)
def cross(d,xy,color,xshape=False):
    x,y=xy;s=6
    if xshape:d.line((x-s,y-s,x+s,y+s),fill=color,width=2);d.line((x-s,y+s,x+s,y-s),fill=color,width=2)
    else:d.line((x-s,y,x+s,y),fill=color,width=2);d.line((x,y-s,x,y+s),fill=color,width=2)
def panel(cases,path,title):
    width=1320;rowh=420;canvas=Image.new('RGB',(width,70+len(cases)*rowh),'white');d=ImageDraw.Draw(canvas)
    d.text((12,8),title,fill='black',font=titlefont)
    d.text((12,39),'Cyan + initial query | Magenta x reference | SAM is an inferred mask, not real pixel GT',fill='black',font=small)
    notes=[]
    for i,(r,reason)in enumerate(cases):
        y=70+i*rowh;q=np.array([float(r['query_x']),float(r['query_y'])]);ref=np.array([float(r['human_reference_x']),float(r['human_reference_y'])]);L=float(r['v61_roi_side_raw_px'])
        assert sha(r['image'])==r['image_sha256'];image=Image.open(r['image']).convert('RGB')
        record=maskrecords[r['id']];assert record['available']and sha(record['mask_path'])==record['mask_sha256']
        with np.load(record['mask_path'])as z:m=z['mask']
        assert m.shape==(int(r['raw_height']),int(r['raw_width']))
        mask=Image.fromarray(np.repeat((m*255).astype(np.uint8)[...,None],3,axis=2))
        for x,im,side,label in [(12,image,L,'Saved N0-centered RGB crop'),(342,mask,L,'Frozen box-SAM selected mask'),(672,image,13.,'13px native RGB detail')]:
            p=patch(im,q,side);pd=ImageDraw.Draw(p);cross(pd,(149.5,149.5),'cyan');rxy=(ref-q)/side*300+149.5
            if ((rxy>=0)&(rxy<300)).all():cross(pd,rxy,'magenta',True)
            canvas.paste(p,(x,y+48));d.text((x,y+24),label,fill='black',font=small)
        d.text((12,y),f"{r['id']} k{r['native_corner']} | {r['human_state']} | error {float(r['initial_reference_error_px']):.3f}px",fill='black',font=font)
        tx=995;d.text((tx,y+50),'Frozen outputs',fill='black',font=font)
        lines=[f"v61 p={float(r['presence_probability']):.3f}",f"Mixed V4 p={float(r['new_probability']):.3f}",f"SAM logit={float(r['SAM_raw_logit']):.2f}",f"SAM nearby r2={r['SAM_radius2_present']}",f"Self proxy={r['geometry_self_proxy']}"]
        for j,line in enumerate(lines):d.text((tx,y+85+29*j),line,fill='black',font=small)
        words=reason.split();text='';liney=y+250
        for word in words:
            if len(text+' '+word)>33:d.text((tx,liney),text,fill='black',font=small);liney+=20;text=word
            else:text=(text+' '+word).strip()
        d.text((tx,liney),text,fill='black',font=small)
        d.text((12,y+365),f"Coordinate annotation: {r['annotation_coordinate_source']}; pixel membership remains unannotated in real evaluation.",fill='black',font=small)
        notes.append(dict(id=r['id'],native_corner=r['native_corner'],reason=reason,v61_p=float(r['presence_probability']),new_p=float(r['new_probability']),SAM_logit=float(r['SAM_raw_logit']),self_proxy=r['geometry_self_proxy'],error_px=float(r['initial_reference_error_px']),real_pixel_GT_created=False))
    canvas.save(path);return notes

by={key(r):r for r in allrows}
fixed=[('wood_night_01:032494',5),('SQUARE_YOLO:025747',5),('wood_night_01:032376',0),('SQUARE_YOLO:027620',1)]
notes=panel([(by[k],'Previously fixed <=1px boundary case; compare outputs without changing query.')for k in fixed],OUT/'FIXED_LOW_ERROR_REAL_CROPS.png','Frozen comparison on the same four pre-existing low-error cases')
eligible=[r for r in allrows if r['same_slot_pipeline_D_vs_ES']=='True']
selected=[]
for group in ['WOOD_PRIMARY105','SQUARE118']:
    a=[r for r in eligible if r['group']==group and r['human_state']=='EXTERNAL_OCCLUDED'and r['new_present']=='True']
    a.sort(key=lambda r:(-float(r['new_probability']),r['id'],int(r['native_corner'])))
    if a:selected.append((a[0],'Highest-confidence external false presence in this group; diagnostic only.'))
a=[r for r in eligible if r['human_state']=='SELF_OCCLUDED'and r['new_present']=='True'and r['geometry_self_proxy']=='True'];a.sort(key=lambda r:(-float(r['new_probability']),r['id']))
if a:selected.append((a[0],'Self corner projects onto visible pallet pixels; binary support cannot identify semantic self.'))
a=[r for r in eligible if r['human_state']=='DIRECT_VISIBLE'and r['human_reference_inside_nominal_crop']=='False'and r['new_present']=='False'];a.sort(key=lambda r:(-float(r['initial_reference_error_px']),r['id']))
if a:selected.append((a[0],'Reference is outside unchanged query crop; low score is not proof of occlusion.'))
failnotes=panel(selected,OUT/'FAILURE_MECHANISMS_REAL_CROPS.png','Remaining mechanisms after the source-frozen comparison')
dump(OUT/'CROP_SELECTION_AND_SCORES.json',dict(fixed_preexisting_cases=notes,postfreeze_failure_diagnostics=failnotes,selection_does_not_change_model=True,new_training_or_threshold_selection=False))
dump(OUT/'DIAGNOSTIC_SEAL.json',dict(current_selection_source_only=seal['selected_primary'],new_training=0,new_PnP_calls=0,real_pixel_GT_created=False,interpretation='Semantic transfer; visible keypoint annotation is not exact query-pixel GT. Descriptive location error never proves crop absence.',outputs=[dict(path=str(p),sha256=sha(p))for p in OUT.iterdir()if p.is_file()and p.name!='DIAGNOSTIC_SEAL.json']))
print(json.dumps(dict(near_summary=summary,images=['FIXED_LOW_ERROR_REAL_CROPS.png','FAILURE_MECHANISMS_REAL_CROPS.png']),indent=2))
