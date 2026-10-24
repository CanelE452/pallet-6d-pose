"""Frozen, fixed public-source cases; stored outputs only, no model inference."""
import csv,hashlib,json,pathlib
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=pathlib.Path(__file__).resolve().parent.parent;OUT=ROOT/'ycb';BASE=ROOT.parent/'near_error_diagnosis_20261022/synthetic'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def truth(v):return str(v).lower()=='true'
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17);small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14);tiny=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12)
review=json.loads((ROOT/'sam/SOURCE_SAM_VISUAL_REVIEW.json').read_text());cases=[dict(c)for c in review['cases']]
near=json.loads((BASE/'SYNTH_CROP_CASES.json').read_text())[:2]
for r in near:cases.append({'title':'SYNTH saved cause '+f'{r["query_to_native_renderer_gt_px"]:.3f}px','id':r['id'],'source':'SYNTH','query_id':f'SYNTH:{r["id"]}:{r["native_corner"]}','native_corner':r['native_corner'],'query_x':r['query_x'],'query_y':r['query_y'],'near_error_record':r})
with np.load(ROOT/'source_cache/v4/MIXED_VALIDATION_RGB_SAM.npz')as z:data={k:z[k]for k in ['sample_id','id','source','rgb','sam','dense_y','y','image_sha256','boundary']}
ids={str(v):i for i,v in enumerate(data['sample_id'])};sourceinputs={r['id']:r for r in map(json.loads,(ROOT/'sam_source/INPUTS.jsonl').read_text().splitlines())if r['split']=='validation'}
ycb={r['sample_id']:r for r in csv.DictReader((ROOT/'source_cache/v2/YCB_VALIDATION_QUERIES.csv').open())}
proxy={r['sample_id']:r for r in csv.DictReader((OUT/'SOURCE_EXTERNAL_PROXY_SUPPORT.csv').open())}
models=['V61','FROZEN_MOBILESAM','SYNTH_CONTEXT_RGB_SAM_V4','MIXED_CONTEXT_RGB_SAM_V4'];preds={};bindings={}
for model in models:
 p=ROOT/f'source_comparison/{model}/SOURCE_OUTPUTS.npz'
 with np.load(p)as z:preds[model]={int(ix):{'p':float(prob),'logit':float(log),'y':int(y)}for ix,prob,log,y in zip(z['index'],z['probability'],z['logits'],z['y'])}
 bindings[model]={'path':str(p),'sha256':sha(p)}
deny=set(json.loads((ROOT/'contract/EVALUATION_SHA256_DENYLIST.json').read_text())['image_sha256']);records=[]
for c in cases:
 i=ids[c['query_id']];s=sourceinputs[c['id']];image=pathlib.Path(s['image']);ih=sha(image);assert ih==data['image_sha256'][i]and ih not in deny
 qx,qy=float(c['query_x']),float(c['query_y']);px,py=np.floor([qx+.5,qy+.5]).astype(int);L=float(s['side']);rgb=np.asarray(Image.open(image).convert('RGB'))
 if c['source']=='YCB':vpath=pathlib.Path(ycb[c['query_id']]['mask_vis']);fpath=pathlib.Path(ycb[c['query_id']]['mask_full']);threshold=0
 else:stem=image.name.replace('_rgb.png','');vpath=image.parent.parent/'mask_visible'/f'{stem}.png';fpath=image.parent.parent/'mask_amodal'/f'{stem}.png';threshold=127
 vis=np.asarray(Image.open(vpath))>threshold;full=np.asarray(Image.open(fpath))>threshold;assert not np.any(vis&~full);label=int(vis[py,px]);assert label==data['y'][i]
 for model in models:assert preds[model][i]['y']==label
 state='VISIBLE_TARGET_SUPPORT'if label else'EXTERNAL_PROXY'if full[py,px]else'OUTSIDE_TARGET_SUPPORT'
 radii={str(r):int(vis[max(0,py-r):min(vis.shape[0],py+r+1),max(0,px-r):min(vis.shape[1],px+r+1)].any())for r in [0,1,2]}
 rec=dict(c);rec.update(cache_index=i,image=str(image),image_sha256=ih,mask_visible=str(vpath),mask_full=str(fpath),mask_binary_threshold=threshold,pixel_x=int(px),pixel_y=int(py),roi_side=L,exact_target=label,source_support_state=state,nearby_visible_support=radii,scores={m:preds[m][i]for m in models},external_proxy_evidence=proxy.get(c['query_id']),new_model_inference=False,private_evaluation=False)
 records.append((rec,rgb,vis,full))
canvas=Image.new('RGB',(1110,1590),'white');d=ImageDraw.Draw(canvas)
d.text((12,8),'Fixed earlier public-source cases: unchanged query and stored frozen scores',font=font,fill='black')
d.text((12,32),'Exact target = visible target instance at floor(query+.5). Nearby support is a different label. Green=visible; orange=external proxy.',font=tiny,fill='black')
d.text((12,49),'RGB uses the saved65x65 input crop. SAM panel shows the frozen9x9 sampled mask input; SAM logit0 and head p=.5 stay fixed.',font=tiny,fill='black')
for j,(r,rgb,vis,full)in enumerate(records):
 top=78+j*248;i=r['cache_index'];d.text((12,top),f'{j+1}. {r["title"]} | {r["source"]} {r["id"]} / query {r["native_corner"]}',font=font,fill='black')
 txt=f'exact={r["exact_target"]}, state={r["source_support_state"]}; nearby r1/r2={r["nearby_visible_support"]["1"]}/{r["nearby_visible_support"]["2"]}; pixel=({r["pixel_x"]},{r["pixel_y"]})'
 if'near_error_record'in r:
  nr=r['near_error_record'];txt+=f'; reference pixel=({nr["gt_nearest_pixel_x"]},{nr["gt_nearest_pixel_y"]}), ref M4={int(nr["gt_pixel_M4"])}'
 d.text((12,top+23),txt,font=tiny,fill='black')
 inp=Image.fromarray(data['rgb'][i].transpose(1,2,0)).resize((185,185),Image.Resampling.NEAREST)
 color=np.full((*vis.shape,3),75,np.uint8);color[full&~vis]=(255,95,30);color[vis]=(35,200,100);L=r['roi_side'];px,py=r['pixel_x'],r['pixel_y'];box=(px-L/2,py-L/2,px+L/2,py+L/2)
 ov=(rgb*.45+color*.55).astype(np.uint8);gt=Image.fromarray(ov).transform((185,185),Image.Transform.EXTENT,box,Image.Resampling.NEAREST)
 sm=data['sam'][i,0]>0;smcolor=np.full((9,9,3),75,np.uint8);smcolor[sm]=(35,200,100);sam=Image.fromarray(smcolor).resize((185,185),Image.Resampling.NEAREST)
 for left,label,panel in [(12,'actual RGB input',inp),(220,'GT target support',gt),(428,'raw SAM mask9',sam)]:
  pd=ImageDraw.Draw(panel);pd.line((84,92,100,92),fill=(0,255,255),width=2);pd.line((92,84,92,100),fill=(0,255,255),width=2);canvas.paste(panel,(left,top+60));d.text((left,top+44),label,font=tiny,fill='black')
 labels={'V61':'v61 p','FROZEN_MOBILESAM':'SAM raw logit','SYNTH_CONTEXT_RGB_SAM_V4':'SYNTH RGB+SAM V4 p','MIXED_CONTEXT_RGB_SAM_V4':'MIXED RGB+SAM V4 p'}
 for k,m in enumerate(models):
  z=r['scores'][m];pred=int(z['logit']>0 if m=='FROZEN_MOBILESAM'else z['p']>=.5);value=f'{z["logit"]:.3f}'if m=='FROZEN_MOBILESAM'else f'{z["p"]:.3f}';color=(0,120,55)if pred==r['exact_target']else(190,25,25)
  d.text((640,top+64+37*k),labels[m],font=small,fill='black');d.text((905,top+64+37*k),f'{value} -> {pred}',font=small,fill=color)
 if r.get('external_proxy_evidence'):
  text=r['external_proxy_evidence']['external_support_evidence'].replace('_',' ');d.text((640,top+215),text,font=tiny,fill=(90,70,10))
canvas.save(OUT/'SOURCE_FINAL_FIXED_CASE_COMPARISON.png')
# Preserve enlarged nearest-pixel explanations for the two saved tiny offsets.
canvas=Image.new('RGB',(1000,620),'white');d=ImageDraw.Draw(canvas);d.text((12,10),'Two fixed synthetic causes: small coordinate error can change the rounded-pixel target',font=font,fill='black')
for j,(r,rgb,vis,full)in enumerate(records[-2:]):
 top=45+j*280;nr=r['near_error_record'];px,py=r['pixel_x'],r['pixel_y'];h,w=vis.shape;box=(px-6.5,py-6.5,px+6.5,py+6.5)
 crop=Image.fromarray(rgb).transform((234,234),Image.Transform.EXTENT,box,Image.Resampling.NEAREST)
 col=np.full((*vis.shape,3),60,np.uint8);col[full&~vis]=(255,95,30);col[vis]=(35,200,100);mask=Image.fromarray(col).transform((234,234),Image.Transform.EXTENT,box,Image.Resampling.NEAREST)
 for panel in[crop,mask]:
  pd=ImageDraw.Draw(panel)
  for k in range(14):pd.line((k*18,0,k*18,234),fill=(110,110,110),width=1);pd.line((0,k*18,234,k*18),fill=(110,110,110),width=1)
  pd.rectangle((108,108,125,125),outline=(255,255,0),width=2)
  gx,gy=nr['gt_nearest_pixel_x']-px,nr['gt_nearest_pixel_y']-py;pd.rectangle((108+gx*18,108+gy*18,125+gx*18,125+gy*18),outline=(255,0,255),width=2)
 canvas.paste(crop,(12,top+30));canvas.paste(mask,(262,top+30));d.text((12,top),f'{r["id"]}: error {nr["query_to_native_renderer_gt_px"]:.6f}px',font=font,fill='black')
 d.text((515,top+40),f'query pixel=({px},{py}), M4={r["exact_target"]}',font=small,fill='black');d.text((515,top+65),f'reference pixel=({nr["gt_nearest_pixel_x"]},{nr["gt_nearest_pixel_y"]}), M4={int(nr["gt_pixel_M4"])}',font=small,fill='black');d.text((515,top+90),'Yellow=query nearest pixel; magenta=reference',font=tiny,fill='black')
 for k,m in enumerate(models):
  z=r['scores'][m];value=f'logit {z["logit"]:.3f}'if m=='FROZEN_MOBILESAM'else f'p {z["p"]:.3f}';d.text((515,top+125+27*k),f'{m}: {value}',font=tiny,fill='black')
canvas.save(OUT/'SOURCE_FINAL_TWO_TINY_OFFSET_CASES.png')
report={'scope':'Six cases fixed in earlier source reviews before final V4 comparisons; source RGB/masks and stored frozen model outputs only','input_case_bindings':{'earlier_SAM_review_sha256':sha(ROOT/'sam/SOURCE_SAM_VISUAL_REVIEW.json'),'earlier_synthetic_two_causes_sha256':sha(BASE/'SYNTH_CROP_CASES.json')},'score_bindings':bindings,'cases':[r for r,_,_,_ in records],'private_data_or_scores_read':False,'new_model_inference':False,'GPU_used':False,'model_cache_or_threshold_changed':False,'figures':['SOURCE_FINAL_FIXED_CASE_COMPARISON.png','SOURCE_FINAL_TWO_TINY_OFFSET_CASES.png'],'physical_external_caveat':'full1/vis0 is an official rendered/depth visibility proxy. An unsupported internal stripe is not certified physical occlusion; see SOURCE_EXTERNAL_PROXY_SUPPORT.json.','direct_visual_review_pending':True}
(OUT/'SOURCE_FINAL_FIXED_CASE_COMPARISON.json').write_text(json.dumps(report,indent=2));print(OUT/'SOURCE_FINAL_FIXED_CASE_COMPARISON.png')
