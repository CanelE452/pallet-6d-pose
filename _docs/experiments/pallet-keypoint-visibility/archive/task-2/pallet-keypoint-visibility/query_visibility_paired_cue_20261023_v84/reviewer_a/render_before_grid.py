"""Before-only native coordinate aid; no annotation, after, masks or model."""
from pathlib import Path
import hashlib,json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).resolve().parent
p=json.loads((R/'BEFORE_REVIEW_POLICY.json').read_text());source=p['allowed_only_RGB_file']
assert hashlib.sha256(Path(source['path']).read_bytes()).hexdigest()==source['sha256']
with Image.open(source['path']) as im:raw=im.convert('RGB')
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',11)
canvas=raw.copy();d=ImageDraw.Draw(canvas)
for x in range(0,raw.width,20):d.line((x,0,x,raw.height),fill='#48b7b2',width=1);d.text((x+1,0),str(x),fill='yellow',font=font)
for y in range(0,raw.height,20):d.line((0,y,raw.width,y),fill='#48b7b2',width=1);d.text((0,y+1),str(y),fill='yellow',font=font)
canvas.resize((1280,960),Image.Resampling.NEAREST).save(R/'BEFORE_NATIVE20_GRID_X2.png')
boxes={'TOP_LEFT':[50,155,110,215],'TOP_RIGHT':[545,155,605,215],'BASE_LEFT':[55,430,125,480],'BASE_RIGHT':[515,430,585,480]}
outputs=[]
for name,box in boxes.items():
 c=raw.crop(tuple(box)).resize(((box[2]-box[0])*6,(box[3]-box[1])*6),Image.Resampling.NEAREST);path=R/(name+'_UNMARKED_X6.png');c.save(path)
 outputs.append(dict(name=name,native_box=box,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
with (R/'BEFORE_COORDINATE_AID_RECEIPT.json').open('x') as f:json.dump(dict(completed=True,source=source,grid_step_native20=True,unmarked_crops=outputs,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),after_mask_depth_model_current_read=0),f,indent=2);f.write('\n')
