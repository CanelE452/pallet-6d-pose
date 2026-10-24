from pathlib import Path
import json,numpy as np
from PIL import Image,ImageDraw,ImageFont
r=Path(__file__).resolve().parent;pack=r/'native_context_cache/SYNTH5120_INPUT_PACK';m=json.loads((pack/'PACK_MANIFEST.json').read_text());line=[json.loads(x) for x in (pack/'INDEX_LINEAGE_INPUT_ONLY.jsonl').read_text().splitlines()]
idx=[0,1366,2731,4096];font=ImageFont.truetype('/usr/share/fonts/truetype/nanum/NanumGothic.ttf',17);small=ImageFont.truetype('/usr/share/fonts/truetype/nanum/NanumGothic.ttf',13)
can=Image.new('RGB',(1220,1230),'#f4f6f8');d=ImageDraw.Draw(can);d.text((12,8),'실제 동결 입력 | 초기 query 그대로 | GT / 예측 분류 없음',font=font,fill='black')
colors=['#ff4040','#ffa000','#ffe000','#80e000','#00cfb0','#00aaff','#8060ff','#ff60d0']
for row,i in enumerate(idx):
 c=next(c for c in m['chunks'] if c['start']<=i<c['end']);z=np.load(c['file']['path']);k=i-c['start'];y=40+row*295
 meta=line[i];d.text((12,y),f"row {i}: {meta.get('id',meta.get('frame_id',''))} · {meta.get('source','')} · {meta.get('partition','')}",font=font,fill='black')
 im=Image.fromarray(z['target_rgb128'][k].transpose(1,2,0));im=im.resize((256,256),Image.Resampling.NEAREST);dr=ImageDraw.Draw(im);q=z['query8_xy'][k];center=z['target_crop_center_xy'][k];side=z['target_crop_side'][k]
 for j in range(8):
  p=(q[j]-center)/side*256+128
  if np.isfinite(p).all() and np.all((p>=0)&(p<256)):
   x1,y1=map(float,p);dr.ellipse((x1-2,y1-2,x1+2,y1+2),fill=colors[j]);dr.text((x1+3,y1),str(j),font=small,fill=colors[j])
 can.paste(im,(12,y+29))
 for j in range(8):
  crop=Image.fromarray(z['rgb96'][k,j].transpose(1,2,0));x=285+j*114;can.paste(crop,(x,y+80));d.text((x,y+45),f"k{j} valid={int(z['query_valid'][k,j])}",font=small,fill='black');d.text((x,y+184),'local RGB96',font=small,fill='black')
 can.save(r/'SOURCE_INPUT_ACTUAL_FOUR_FRAMES.png')
print('wrote SOURCE_INPUT_ACTUAL_FOUR_FRAMES.png; noGT reads')
