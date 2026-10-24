"""Display sealed saved crops, never recompute model crops or visibility."""
from pathlib import Path
from collections import Counter
import hashlib,html,json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
H=Path(__file__).resolve().parent;V=H.parent
FEATURE=V/'full230_native_DINO_v1/FEATURE_COMPLETE.json';R0=V/'actual_full230_initializer_v1/PREDICTION_SEAL.json'
EXPECTED_R0='e986a15f998974e6c1aabade48aff7cbf02565fbc6512339425bae175d272f5d'
def bind(p):return dict(path=str(Path(p).resolve()),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def check(b):assert bind(b['path'])==b;return Path(b['path'])
def read(p):return json.loads(Path(p).read_text())
def lines(p):return[json.loads(s)for s in Path(p).read_text().splitlines()if s]
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def exact(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
F=ImageFont.truetype(FONT,13);SMALL=ImageFont.truetype(FONT,11);BIG=ImageFont.truetype(FONT,16)

def gallery(rows,title,path):
 cards='\n'.join(f'<a class="card" href="{html.escape(r["PNG_relative"])}"><img loading="lazy" src="{html.escape(r["PNG_relative"])}" alt="{html.escape(r["source_id"])}"><span>#{r["frame_index"]:03d} {html.escape(r["source_id"])} | valid {r["continuous_count"]}/8</span></a>'for r in rows)
 return f'''<!doctype html><meta charset="utf-8"><title>{title}</title><style>body{{background:#edf2f7;color:#17304a;font:15px sans-serif;margin:20px}}h1{{font-size:24px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px}}.card{{display:block;background:white;padding:8px;color:inherit;text-decoration:none}}.card img{{display:block;width:100%;height:auto}}.card span{{display:block;margin:7px 0}}p{{max-width:1100px;line-height:1.5}}</style><h1>{title}</h1><p>입력 증거입니다. 새로운 가림 예측이나 의미 라벨이 아닙니다. 원 R0 좌표·L과 저장 RGB96을 그대로 사용합니다. 8개 crop은 native0→7 순서이며 1.5배 nearest 확대만 적용했습니다. 검은 crop/padding도 삭제하지 않았습니다. Magenta: 원 좌표. Cyan: 원 q8 min/max context. Crop 자체에는 표식을 넣지 않았습니다.</p>{path}<div class="grid">{cards}</div>'''

def run():
 assert bind(R0)['sha256']==EXPECTED_R0 and not(H/'STARTED.json').exists()
 feature=read(FEATURE);s=read(R0);assert feature['completed']and feature['PASS']and s['completed']and s['PASS']and feature['source_initializer']==bind(R0)
 assert feature['frames']==s['frames']==230 and feature['rows']==s['rows']==1840
 files={Path(b['path']).name:check(b)for b in s['outputs']}
 props=lines(files['FROZEN_PROPOSALS_ALL230_INPUT_ONLY.jsonl']);queries=lines(files['ORIGINAL_NATIVE_QUERY1840_INPUT_ONLY.jsonl'])
 with np.load(files['ORIGINAL_NATIVE_QUERY1840_AND_VALID9X9_INPUT_ONLY.npz'],allow_pickle=False)as f:q={k:f[k].copy()for k in f.files}
 crops=read(check(feature['crop_manifest']))['sheets'];facts=lines(check(feature['inputfacts']))
 intake=read(check(feature['source_intake']));manifest=read(check(intake['RGB_manifest']))['rows']
 assert len(props)==len(crops)==len(manifest)==230 and len(facts)==len(queries)==1840
 assert np.array_equal(q['global_index'],np.arange(1840))and np.array_equal(q['frame_index'],np.repeat(np.arange(230),8))and np.array_equal(q['native_slot'],np.tile(np.arange(8),230))
 write(H/'STARTED.json',dict(completed=True,code=bind(__file__),feature_seal=bind(FEATURE),initializer_seal=bind(R0),source_intake=feature['source_intake'],purpose='Saved original-query input evidence only; no visibility predictions or new labels.',GT_body_or_visibilityscore_reads=0,learned_calls=0))
 out=H/'frames';out.mkdir();records=[]
 priority=[73,95,109]
 for i in priority+[j for j in range(230)if j not in priority]:
  p,m,c=props[i],manifest[i],crops[i]
  assert p['frame_index']==m['frame_index']==c['frame_index']==i and p['source_id']==m['source_id']==c['source_id']
  assert all(p[k]==v for k,v in m.items())and p['split']in('SOURCE_DEVELOPMENT','SOURCE_VALIDATION')
  check(p['canonical_RGB_NPY']);assert bind(p['image'])['sha256']==p['image_sha256']
  rgb=np.load(p['canonical_RGB_NPY']['path'],allow_pickle=False);assert rgb.dtype==np.uint8 and list(rgb.shape)==p['raw_hw']+[3]and ash(rgb)==p['decoded_RGB_sha256']
  with np.load(check(c['crop_packet']),allow_pickle=False)as f:raw=f['rgb96'].copy();valid=f['valid96'].copy();idx=f['global_index'].copy();constructed=f['crop_constructed'].copy()
  assert raw.shape==(8,96,96,3)and raw.dtype==np.uint8 and valid.shape==(8,96,96)and valid.dtype==bool and np.array_equal(idx,np.arange(i*8,i*8+8))
  sl=slice(i*8,i*8+8);qq=q['query_xy'][sl];ll=q['L'][sl];ff=facts[sl]
  rowhash=[]
  for k,f in enumerate(ff):
   g=i*8+k;r=queries[g]
   assert f['global_index']==g and f['frame_index']==i and f['frame_id']==p['id']and f['source_id']==p['source_id']and f['native_slot']==k and f['split']==p['split']and f['query_id']==r['query_id']==str(q['query_id'][g])
   assert np.array_equal(np.asarray(f['query_xy'],np.float64),qq[k],equal_nan=True)and np.array_equal(np.asarray(f['L'],np.float64),ll[k],equal_nan=True)
   assert f['native_continuous_valid']==bool(q['continuous_valid'][g])and f['nearest_valid']==bool(q['nearest_valid'][g])and f['feature_constructed']==bool(constructed[k])
   assert ash(raw[k])==f['RGB96_HWC_sha256']and ash(raw[k].transpose(2,0,1))==f['RGB96_CHW_sha256']and ash(valid[k])==f['valid96_sha256']
   rowhash.append(dict(global_index=g,query_id=r['query_id'],query_xy=f['query_xy'],L=f['L'],continuous_valid=f['native_continuous_valid'],nearest_valid=f['nearest_valid'],crop_constructed=f['feature_constructed'],RGB96_HWC_sha256=ash(raw[k]),valid96_sha256=ash(valid[k]),valid_fraction=float(valid[k].mean())))
  sheet=Image.new('RGB',(800,820),'#edf2f7');d=ImageDraw.Draw(sheet)
  d.text((10,7),f"#{i:03d} {p['source_id']}  |  {p['split']}",font=BIG,fill='#17304a')
  d.text((10,29),'INPUT EVIDENCE | original q/L + saved RGB96 | no visibility prediction',font=F,fill='#17304a')
  # All overlays are clipped on this per-scene image, never on neighboring cells.
  scene=Image.fromarray(rgb.copy()).resize((640,360),Image.Resampling.BILINEAR);sd=ImageDraw.Draw(scene)
  box=ff[0]['context_box_xyxy']
  assert all(f['context_box_xyxy']==box for f in ff)
  if box is not None and np.isfinite(np.asarray(box,np.float64)).all():sd.rectangle(tuple(float(x)*.5 for x in box),outline='#00d9e8',width=2)
  for k,xy in enumerate(qq):
   if not q['finite_non_sentinel'][i*8+k]:continue
   x,y=xy*.5;sd.ellipse((x-4,y-4,x+4,y+4),fill='#ff33c7',outline='white',width=1)
   if 0<=x<640 and 0<=y<360:sd.text((min(620,max(0,x+5)),min(342,max(0,y-15))),str(k),font=BIG,fill='#ff33c7',stroke_width=1,stroke_fill='white')
  sheet.paste(scene,(8,53));d.text((656,55),'Original native order',font=SMALL,fill='#17304a')
  for k in range(8):
   g=i*8+k;y=79+k*37;state='NOQUERY'if q['no_detection'][g]else('VALID'if q['continuous_valid'][g]else('OOB'if q['finite_non_sentinel'][g]else'UNDEFINED'))
   xy=qq[k];xytext=f'{xy[0]:.1f},{xy[1]:.1f}'if np.isfinite(xy).all()else'--,--';lt=f'{ll[k]:.1f}'if np.isfinite(ll[k])else'--'
   d.text((656,y),f'n{k} {state}',font=SMALL,fill='#17304a');d.text((656,y+14),f'{xytext}  L{lt}',font=SMALL,fill='#17304a')
  d.text((10,421),'Saved crops: native0..3 then native4..7. Black padding preserved; no crop markers.',font=SMALL,fill='#17304a')
  for k in range(8):
   x=10+(k%4)*198;y=443+(k//4)*180;display=Image.fromarray(raw[k]).resize((144,144),Image.Resampling.NEAREST)
   sheet.paste(display,(x,y));d.rectangle((x-1,y-1,x+144,y+144),outline='#738399')
   state='U:noquery'if q['no_detection'][i*8+k]else('input-valid'if q['continuous_valid'][i*8+k]else'input-OOB')
   d.text((x,y+147),f'n{k} {state}',font=SMALL,fill='#17304a');d.text((x,y+161),f'valid pixels {valid[k].mean():.1%}',font=SMALL,fill='#17304a')
  d.text((10,807),'RGB96 bytes unchanged; display enlargement nearest x1.5. Input availability is not visibility.',font=SMALL,fill='#17304a')
  filename=f'FRAME_{i:03d}_{p["source_id"].replace("/","_").replace(":","_")}.png';path=out/filename;sheet.save(path)
  records.append(dict(frame_index=i,frame_id=p['id'],source_id=p['source_id'],split=p['split'],PNG=bind(path),PNG_relative='frames/'+filename,source_RGB_NPY=p['canonical_RGB_NPY'],original_JPEG=dict(path=p['image'],sha256=p['image_sha256']),decoded_RGB_sha256=p['decoded_RGB_sha256'],saved_crop_packet=c['crop_packet'],continuous_count=int(q['continuous_valid'][sl].sum()),noquery=bool(q['no_detection'][sl].all()),native_order=list(range(8)),rows=rowhash))
 records.sort(key=lambda r:r['frame_index'])
 assert len(records)==230 and sum(len(r['rows'])for r in records)==1840
 val=[r for r in records if r['split']=='SOURCE_VALIDATION'];dev=[r for r in records if r['split']=='SOURCE_DEVELOPMENT'];assert len(val)==45 and len(dev)==185
 vh=H/'VALIDATION45_CONTACT_INDEX.html';vh.write_text(gallery(val,'MR6D validation45: original-query input evidence','<p><a href="index.html">All230 index</a></p>'))
 devbody=gallery(dev,'Development185','').split('<div class="grid">',1)[1]
 index=H/'index.html';index.write_text(gallery(val,'MR6D all230 input evidence — validation45 first','<p><a href="VALIDATION45_CONTACT_INDEX.html">Validation45 only</a></p>')+'<h2>Development185 — original numeric order</h2><div class="grid">'+devbody)
 manifest=H/'ALL230_SAVED_CROP_EVIDENCE.json';write(manifest,dict(completed=True,frames=230,rows=1840,validation45_first_display=True,original_numeric_identity_order_preserved=True,records=records))
 seal=H/'EVIDENCE_RENDER_COMPLETE.json';write(seal,dict(completed=True,PASS=True,code=bind(__file__),feature_seal=bind(FEATURE),initializer_seal=bind(R0),source_intake=feature['source_intake'],crop_manifest=feature['crop_manifest'],inputfacts=feature['inputfacts'],evidence_manifest=bind(manifest),HTML_index=bind(index),validation45_contact_index=bind(vh),frames=230,rows=1840,PNG_count=230,board_HW=[820,800],crop_display_HW=[144,144],crop_enlargement='NEAREST x1.5, saved96 bytes unchanged',continuous_valid=int(q['continuous_valid'].sum()),noquery_rows=int(q['no_detection'].sum()),all1840_saved_RGB96_and_validmask_hashes_verified=True,all230_original_RGB_NPY_JPEG_and_query_identity_verified=True,GT_recrop_or_coordinate_L_correction=False,new_visibility_predictions_or_semantic_labels=0,GT_pose_mask_depth_reference_or_visibilityscore_body_reads=0,model_encoder_R0_or_training_calls=0,private_access=0,representative_validation_view_plan=priority))
 print(json.dumps(dict(completed=True,PASS=True,seal=bind(seal),index=bind(index),validation_index=bind(vh))),flush=True)
if __name__=='__main__':run()
