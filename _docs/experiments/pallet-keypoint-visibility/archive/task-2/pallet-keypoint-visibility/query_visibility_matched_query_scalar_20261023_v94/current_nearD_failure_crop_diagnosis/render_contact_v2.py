"""Display spacing repair only; original crops/first contacts/selection stay immutable."""
from pathlib import Path
import json,math,hashlib
import numpy as np
from PIL import Image,ImageDraw
import build_saved_all11 as base
H=Path(__file__).resolve().parent
r=json.loads((H/'ALL11_CASES_AND_PROVENANCE.json').read_text());cases=r['all11']
with np.load(H/'ALL11_SAVED_RGB96_AND_METADATA.npz',allow_pickle=False)as z:images=z['rgb96']
outs=[]
for no,start in enumerate([0,8],1):
 end=min(start+8,11);rows=math.ceil((end-start)/4);fig=Image.new('RGB',(1680,100+rows*390),'#e9edf3');d=ImageDraw.Draw(fig)
 d.text((14,10),f'V94 C1: all 11 manual-near humanD misses | fixed cases {start+1}-{end}',font=base.font(24),fill='#101727')
 d.text((14,43),'Original cached RGB96 and fixed 2x display; q/L/r unchanged. No fullRGB decode, recenter or model call.',font=base.font(16),fill='#202a38')
 d.text((14,69),'Human manual D evidence; C1 auxiliary hidden is not a learned occlusion cause. Root has not viewed this new gallery yet.',font=base.font(14),fill='#202a38')
 for j in range(start,end):
  rgb=images[j].transpose(1,2,0).copy();assert base.array_sha(rgb)==cases[j]['display_rgb96_HWC_uint8_sha256']
  p=base.make_panel(cases[j],rgb);pd=ImageDraw.Draw(p);pd.rectangle((0,128,419,149),fill='white')
  pd.text((12,132),'RGB96 raw',font=base.font(11),fill='#151b26');pd.text((138,132),'RGB96 2x + q/r',font=base.font(11),fill='#151b26')
  fig.paste(p,((j-start)%4*420,100+(j-start)//4*390))
 name='CONTACT_01_ALL8_V2.png'if no==1 else'CONTACT_02_REMAINING3_V2.png';path=H/name;assert not path.exists();fig.save(path);outs.append(base.binding(path))
(H/'CONTACT_LAYOUT_V2_QA.json').write_text(json.dumps(dict(completed=True,display_only_change='Shortened overlapping per-image captions. V1 figures/selection/crop bytes preserved; no source/score re-read.',code=base.binding(__file__),figures=outs,all11_rgb_hashes_rechecked=True,actual_new_original_RGB_decode_or_model=0),indent=2)+'\n')
print(json.dumps(outs,indent=2))
