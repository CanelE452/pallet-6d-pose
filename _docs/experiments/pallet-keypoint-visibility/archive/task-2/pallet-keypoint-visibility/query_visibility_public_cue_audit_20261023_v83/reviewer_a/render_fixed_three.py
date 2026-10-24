"""Review-only displays of three frozen public cue proposals; no model/GT masks."""
from pathlib import Path
import hashlib,json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).resolve().parent
P=R.parent.parent
policy=json.loads((R/'REVIEW_POLICY_BEFORE_RGB.json').read_text())
assert policy['frozen_before_raw_RGB_decode_or_view']
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def mark(d,xy,color='#ffdf57',radius=7):
 x,y=xy;d.line((x-radius,y,x+radius,y),fill=color,width=2);d.line((x,y-radius,x,y+radius),fill=color,width=2)
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
bold=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',22)
contact=Image.new('RGB',(1500,440*3),'#14202b');records=[]
for n,row in enumerate(policy['fixed_records']):
 f=Path(row['image']);assert hashlib.sha256(f.read_bytes()).hexdigest()==row['image_sha256']
 with Image.open(f) as opened:raw=opened.convert('RGB')
 assert list(raw.size[::-1])==row['raw_hw']
 x,y=row['original_xy'];rad=row['original_uncertainty_radius_px'];box=row['old_detail_box']
 full=raw.copy().resize((640,360),Image.Resampling.BILINEAR)
 fd=ImageDraw.Draw(full);mark(fd,(x/2,y/2));fd.rectangle(((x-rad)/2,(y-rad)/2,(x+rad)/2,(y+rad)/2),outline='#5ee2ed',width=1)
 detail=raw.crop(tuple(box));detail.thumbnail((440,330),Image.Resampling.NEAREST)
 nativebox=(int(x)-48,int(y)-48,int(x)+48,int(y)+48)
 native=raw.crop(nativebox).resize((288,288),Image.Resampling.NEAREST)
 marked=native.copy();md=ImageDraw.Draw(marked);mark(md,((x-nativebox[0])*3,(y-nativebox[1])*3));md.rectangle(((x-rad-nativebox[0])*3,(y-rad-nativebox[1])*3,(x+rad-nativebox[0])*3,(y+rad-nativebox[1])*3),outline='#5ee2ed',width=2)
 unmarked_path=R/f'{row["id"]}_UNMARKED_NATIVE96_X3.png';native.save(unmarked_path)
 detail_path=R/f'{row["id"]}_UNMARKED_OLD_DETAIL.png';raw.crop(tuple(box)).save(detail_path)
 canvas=Image.new('RGB',(1500,440),'#14202b');d=ImageDraw.Draw(canvas)
 d.text((16,12),f'{row["id"]}: unchanged r=({x:g},{y:g}), prior uncertainty +/-{rad:g}px / native8 UNKNOWN',fill='white',font=bold)
 for xx,text in ((16,'Raw source frame (display 0.5x)'),(680,'Unmarked old native detail'),(1180,'Native96 x3 / r and prior range')):d.text((xx,47),text,fill='white',font=font)
 canvas.paste(full,(16,75));canvas.paste(detail,(680,75));canvas.paste(marked,(1180,75))
 d.text((680,397),'AI weak-source cue review only; no human GT / mask / mesh / query repair.',fill='#d1dae2',font=font)
 path=R/f'{row["id"]}_REVIEW_PANEL.png';canvas.save(path);contact.paste(canvas,(0,n*440))
 records.append(dict(id=row['id'],source_file=dict(path=str(f),sha256=row['image_sha256']),native_r=row['original_xy'],old_uncertainty_radius_px=rad,old_detail_box=box,native96_box=list(nativebox),unmarked_native96=bind(unmarked_path),unmarked_old_detail=bind(detail_path),panel=bind(path)))
contact.save(R/'THREE_FIXED_PUBLIC_CUES_CONTACT.png')
with (R/'DISPLAY_RECEIPT.json').open('x') as f:json.dump(dict(completed=True,policy=bind(R/'REVIEW_POLICY_BEFORE_RGB.json'),code=bind(Path(__file__)),records=records,contact=bind(R/'THREE_FIXED_PUBLIC_CUES_CONTACT.png'),source_RGB_decodes=3,new_model_mask_depth_GT_current_forward_fit_PnP=0),f,indent=2);f.write('\n')
print(json.dumps(bind(R/'THREE_FIXED_PUBLIC_CUES_CONTACT.png')))
