"""Fixed-five raw RGB inspection renderer; never reads primary after artifacts."""
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
import hashlib,json,math,argparse
R=Path(__file__).resolve().parent;V=R.parent
PACKET=V/'SECONDARY5_REVIEW_PACKET_WITHOUT_PRIMARY_AFTER.json'
def rd(p):return json.loads(Path(p).read_text())
def bind(p):
 p=Path(p).resolve();return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
def check(x):assert bind(x['path'])==x,x['path']
def save(p,x):
 with p.open('x')as f:json.dump(x,f,indent=2,ensure_ascii=False);f.write('\n')
def txt(im,xy,s,size=15):ImageDraw.Draw(im).text(xy,s,font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',size),fill='white')
def marked(raw,slots):
 im=raw.copy();d=ImageDraw.Draw(im)
 for k,s in enumerate(slots):
  if s['reference_xy']is None:continue
  x,y=s['reference_xy'];box=s['uncertainty_bbox_xyxy_half_open'];d.rectangle(box,outline='yellow',width=1);d.line((x-3,y,x+3,y),fill='red');d.line((x,y-3,x,y+3),fill='red');d.text((x+4,y),str(k),fill='red')
 return im
p=argparse.ArgumentParser();p.add_argument('stage',choices=['before','after']);stage=p.parse_args().stage
if stage=='after':assert rd(R/'SECONDARY_BEFORE_SUPPORT_SEAL.json')['completed']
packet=rd(PACKET);policy=rd(R/'REVIEW_POLICY_BEFORE_RGB.json');assert policy['all5_fixed_pairs_all6_primarybefore_slots']
files=[];verified=[]
for n,row in enumerate(packet['rows']):
 check(row['before_annotation']);check(row['before_seal']);before=rd(row['before_annotation']['path']);slots=before['slots'];assert len(slots)==6
 rec=row['before_RGB']if stage=='before'else row['after_RGB'];check(rec);raw=Image.open(rec['path']).convert('RGB');assert raw.size==(640,480)
 name=f'{n:02d}_'+row['pair_id'].replace(':','__');out=R/name;out.mkdir(exist_ok=True)
 full=Image.new('RGB',(1280,520),(25,25,25));full.paste(raw,(0,40));full.paste(marked(raw,slots),(640,40));txt(full,(5,5),f'{row["pair_id"]} {stage.upper()} raw | fixed primarybefore refs (unchanged)',16)
 f=out/(stage.upper()+'_FULL_RAW_MARKED.png');full.save(f);files.append(bind(f))
 sheet=Image.new('RGB',(1152,3*330),(25,25,25))
 if stage=='after':
  br=Image.open(row['before_RGB']['path']).convert('RGB');assert br.size==raw.size
 for k,s in enumerate(slots):
  lx=k%2*576;ly=k//2*330;txt(sheet,(lx+3,ly+2),f'{row["pair_id"]} slot{k} {s["part_slot"]}',12)
  if s['reference_xy']is None:
   txt(sheet,(lx+20,ly+80),'NULL PRIMARY BEFORE -> MISSING U',18);continue
  x,y=s['reference_xy'];x0=math.floor(x)-48;y0=math.floor(y)-48
  crop=raw.crop((x0,y0,x0+96,y0+96))
  if stage=='before':
   left=crop;right=marked(raw,slots).crop((x0,y0,x0+96,y0+96));label='BEFORE raw then same annotated; '
  else:
   left=br.crop((x0,y0,x0+96,y0+96));right=crop;label='BEFORE raw then AFTER raw; '
  txt(sheet,(lx+3,ly+20),label+f'xy=({x},{y})',12)
  sheet.paste(left.resize((288,288),Image.Resampling.NEAREST),(lx,ly+42));sheet.paste(right.resize((288,288),Image.Resampling.NEAREST),(lx+288,ly+42))
  if stage=='after':
   # Display-only red outline; raw same96 panel is separately unmodified above.
   pass
 f=out/(stage.upper()+'_FIXED6_RAW96_DETAILS.png');sheet.save(f);files.append(bind(f))
 verified.append({'pair_id':row['pair_id'],'stage':stage,'source_RGB':rec,'decoded_raw_hw':[480,640],'primarybefore':row['before_annotation'],'original_slots_geometry':[{'part_slot':s['part_slot'],'reference_xy':s['reference_xy'],'uncertainty_bbox_xyxy_half_open':s['uncertainty_bbox_xyxy_half_open']}for s in slots]})
save(R/(stage.upper()+'_RENDER_RECEIPT.json'),{'completed':True,'stage':stage,'packet':bind(PACKET),'policy':bind(R/'REVIEW_POLICY_BEFORE_RGB.json'),'code':bind(__file__),'verified':verified,'files':files,'primary_after_artifacts_read':0,'mask_depth_sourceproxy_q_model_current_reads':0,'new_forward_fit_PnP':0})
print(json.dumps({'stage':stage,'rendered':len(files),'receipt':bind(R/(stage.upper()+'_RENDER_RECEIPT.json'))}))
