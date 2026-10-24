"""Primary source annotation I/O; no mask/depth/q/model operations."""
from pathlib import Path
import hashlib,json,math
import numpy as np
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).resolve().parent
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(p):return json.loads(Path(p).read_text())
def write(p,obj):
 with Path(p).open('x') as f:json.dump(obj,f,indent=2,ensure_ascii=False);f.write('\n')
INDEX=read(R/'PAIR_DIRECTORY_INDEX.json')['rows']
POLICY=read(R/'ANNOTATOR_POLICY_BEFORE_NEW_RGB.json')
def pair(i):return Path(INDEX[i]['path'])
def rgb(i,stage):
 d=pair(i);m=read(d/'PAIR_METADATA_INPUT_ONLY.json');b=m[stage+'_RGB'];assert bind(Path(b['path']))['sha256']==b['sha256']
 if stage=='after':
  assert all((pair(k)/'BEFORE_SEAL.json').exists() for k in range(5)), 'All before seals required by A outcome-quarantine policy'
 with Image.open(b['path']) as im:a=np.asarray(im.convert('RGB')).copy()
 assert a.dtype==np.uint8 and list(a.shape)==[480,640,3]
 return a,m,b
def prepare_before(i):
 d=pair(i);a,m,b=rgb(i,'before');im=Image.fromarray(a);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',10)
 grid=im.copy();draw=ImageDraw.Draw(grid)
 for x in range(0,640,20):draw.line((x,0,x,480),fill='#55b9b9');draw.text((x+1,0),str(x),fill='yellow',font=font)
 for y in range(0,480,20):draw.line((0,y,640,y),fill='#55b9b9');draw.text((0,y+1),str(y),fill='yellow',font=font)
 grid.resize((1280,960),Image.Resampling.NEAREST).save(d/'BEFORE_NATIVE20_GRID.png')
 write(d/'BEFORE_RGB_RECEIPT.json',dict(completed=True,pair_id=m['pair_id'],source=b,raw_hw=[480,640],decoded_RGB_SHA256=hashlib.sha256(a.tobytes()).hexdigest(),policy=bind(R/'ANNOTATOR_POLICY_BEFORE_NEW_RGB.json'),code=bind(Path(__file__)),after_mask_depth_q_model_current_reads=0))
def seal_before(i,records,wall_note):
 d=pair(i);a,m,b=rgb(i,'before');assert [r['part_slot'] for r in records]==POLICY['slots']
 for r in records:
  xy=r['reference_xy'];box=r['uncertainty_bbox_xyxy_half_open']
  assert (xy is None)==(box is None)
  if xy is not None:assert len(xy)==2 and len(box)==4 and box[0]<=xy[0]<box[2] and box[1]<=xy[1]<box[3] and 0<=box[0]<box[2]<=640 and 0<=box[1]<box[3]<=480
  r.update(cue_id=m['pair_id']+'::'+r['part_slot'],native8_semantic_id=None,native8_status='UNKNOWN',human_gold=False,official_gold=False,training_admitted=False)
 obj=dict(completed=True,phase='BEFORE_ONLY',pair_id=m['pair_id'],source_split=m['split'],slots=records,principal_wall_description=wall_note,source_before_RGB=dict(b,raw_hw=[480,640]),before_only_frozen=True,raw_hw=[480,640],known_visible_cues=sum(x['reference_xy'] is not None for x in records),unknown_cues=sum(x['reference_xy'] is None for x in records),uncertainty_not_pixel_material_GT=True,after_mask_depth_q_model_current_reads=0,new_model_forward_fit_composites_PnP=0,policy=bind(R/'ANNOTATOR_POLICY_BEFORE_NEW_RGB.json'),format=bind(R.parent.parent/'ANNOTATION_RECORD_FORMAT_BEFORE_LABELS.json'))
 write(d/'BEFORE_CUES.json',obj)
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
 canvas=Image.new('RGB',(1440,860),'#14212d');draw=ImageDraw.Draw(canvas);draw.text((15,10),m['pair_id']+' / BEFORE ONLY / physical6 slots, native8 UNKNOWN',fill='white',font=font)
 scene=Image.fromarray(a).resize((768,576),Image.Resampling.NEAREST);sd=ImageDraw.Draw(scene)
 tiles=[]
 for k,row in enumerate(records):
  xy=row['reference_xy'];xx=805+(k%2)*315;yy=75+(k//2)*260;draw.text((xx,yy-24),str(k)+' '+row['part_slot'].replace('LOWEST_WIDE_WALL','LOW_WALL').replace('UPPER_WALL','UP_WALL'),fill='white',font=font)
  if xy is None:tile=Image.new('RGB',(280,205),'#26333f');ImageDraw.Draw(tile).text((10,80),'UNKNOWN / no coordinate',fill='#ffd17e',font=font)
  else:
   x,y=xy;box=row['uncertainty_bbox_xyxy_half_open'];sd.rectangle(tuple(v*1.2 for v in box),outline='#5de2e9',width=2);sd.text((x*1.2+4,y*1.2-15),str(k),fill='#ffdf52',font=font)
   cropbox=[max(0,int(x)-28),max(0,int(y)-28),min(640,int(x)+29),min(480,int(y)+29)];tile=Image.fromarray(a).crop(tuple(cropbox));s=min(280/tile.width,205/tile.height);tile=tile.resize((round(tile.width*s),round(tile.height*s)),Image.Resampling.NEAREST);td=ImageDraw.Draw(tile);td.rectangle(((box[0]-cropbox[0])*s,(box[1]-cropbox[1])*s,(box[2]-cropbox[0])*s,(box[3]-cropbox[1])*s),outline='#5de2e9',width=2);td.line(((x-cropbox[0])*s-5,(y-cropbox[1])*s,(x-cropbox[0])*s+5,(y-cropbox[1])*s),fill='#ffdf52',width=2)
  canvas.paste(tile,(xx,yy))
 canvas.paste(scene,(15,70));draw.text((15,690),'Uncertainty from BEFORE appearance; no after/mask/depth/model/query used.',font=font,fill='white');canvas.save(d/'BEFORE_CONTACT.png')
 write(d/'BEFORE_SEAL.json',dict(completed=True,PASS=True,pair_id=m['pair_id'],before_proposals=bind(d/'BEFORE_CUES.json'),contact=bind(d/'BEFORE_CONTACT.png'),RGB_receipt=bind(d/'BEFORE_RGB_RECEIPT.json'),policy=bind(R/'ANNOTATOR_POLICY_BEFORE_NEW_RGB.json'),code=bind(Path(__file__)),before_only_sealed_before_after=True,source_6slots_all_retained=True,after_mask_depth_q_model_current_reads=0))
def prepare_after(i):
 d=pair(i);a,m,b=rgb(i,'after');before,_,_=rgb(i,'before');refs=read(d/'BEFORE_CUES.json')['slots'];font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
 scene=Image.new('RGB',(1300,530),'#14212d');draw=ImageDraw.Draw(scene)
 for j,(name,v) in enumerate((('BEFORE',before),('AFTER',a))):draw.text((j*650+5,5),m['pair_id']+' '+name,font=font,fill='white');scene.paste(Image.fromarray(v),(j*650+5,40))
 scene.save(d/'PAIRED_FULL_SCENE.png');contact=Image.new('RGB',(1240,6*310),'#14212d');items=[]
 for k,row in enumerate(refs):
  canvas=Image.new('RGB',(1240,310),'#14212d');cd=ImageDraw.Draw(canvas);cd.text((10,5),row['part_slot']+' / same fixed reference/uncertainty',font=font,fill='white')
  if row['reference_xy'] is None:cd.text((20,100),'BEFORE UNKNOWN => AFTER UNKNOWN; no coordinate invented',font=font,fill='#ffd17e');box=None
  else:
   x,y=row['reference_xy'];u=row['uncertainty_bbox_xyxy_half_open'];box=[max(0,int(x)-30),max(0,int(y)-30),min(640,int(x)+31),min(480,int(y)+31)]
   for j,(name,v) in enumerate((('before',before),('after',a))):
    raw=Image.fromarray(v).crop(tuple(box));s=min(270/raw.width,235/raw.height);tile=raw.resize((round(raw.width*s),round(raw.height*s)),Image.Resampling.NEAREST);marked=tile.copy();md=ImageDraw.Draw(marked);md.rectangle(((u[0]-box[0])*s,(u[1]-box[1])*s,(u[2]-box[0])*s,(u[3]-box[1])*s),outline='#5de2e9',width=2)
    for title,t,xx in [(name+' RAW',tile,10+j*305),(name+' FIXED RANGE',marked,620+j*305)]:cd.text((xx,33),title,font=font,fill='white');canvas.paste(t,(xx,60))
  path=d/(f'AFTER_PAIR_{k:02d}.png');canvas.save(path);contact.paste(canvas,(0,k*310));items.append(dict(part_slot=row['part_slot'],native_crop_box=box,figure=bind(path)))
 contact.save(d/'AFTER_ALL6_CONTACT.png');write(d/'AFTER_DISPLAY_RECEIPT.json',dict(completed=True,before_seal=bind(d/'BEFORE_SEAL.json'),source_after=b,decoded_RGB_SHA256=hashlib.sha256(a.tobytes()).hexdigest(),all6_same_refs=True,figures=items,full_scene=bind(d/'PAIRED_FULL_SCENE.png'),contact=bind(d/'AFTER_ALL6_CONTACT.png'),no_warp_or_relocalization=True,mask_depth_q_model_current_reads=0,code=bind(Path(__file__))))
