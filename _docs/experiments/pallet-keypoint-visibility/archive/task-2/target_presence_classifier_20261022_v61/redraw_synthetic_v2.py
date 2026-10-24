"""Presentation repair only: distinguish outside-image crop padding from target white mask."""
from pathlib import Path
import json,gzip,sys
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
R=Path(__file__).resolve().parent;OLD=R.parent/'external_ray_proxy_head_20261022_v60';sys.path.insert(0,str(OLD));from common import read
ss=read(R/'SYNTH_FIGURE_SELECTION.json');man=read(R/'SYNTH_FIGURE_MANIFEST.json');u={x['id']:x for x in read(OLD/'PREPARED_INPUTS_SYNTH.json.gz')};import csv
audit={x['id']:x for x in csv.DictReader(open(R/'MASK_CONTRACT_AUDIT.csv'))};fig,axs=plt.subplots(6,4,figsize=(12,13.7),layout='constrained')
for row,(s,m)in enumerate(zip(ss,man)):
 x=u[s['id']];rgb=np.array(Image.open(x['image']).convert('RGB'));mask=np.array(Image.open(audit[s['id']]['visible_path']))>127;q=np.array(m['query_xy']);lo=np.array(m['same_crop_xyxy'][:2]);hi=np.array(m['same_crop_xyxy'][2:]);L=hi[0]-lo[0];xx,yy=m['nearest_pixel_xy'];color='#00bd7f'if s['predicted_present']=='True'else'#dc267f'
 for col in range(4):
  ax=axs[row,col];ax.set_axis_off()
  if col==0:ax.imshow(rgb);ax.add_patch(Rectangle(lo,L,L,fill=False,edgecolor='#00aaaa',lw=1.2));ax.set_title(s['purpose']+' | '+s['id']+f' / k{m["native_corner"]}',fontsize=7.5)
  else:
   ax.add_patch(Rectangle(lo,L,L,facecolor='#888888',edgecolor='none',zorder=-1));ax.imshow(mask if col==2 else rgb,cmap='gray',vmin=0,vmax=1,interpolation='nearest');ax.set_xlim(lo[0],hi[0]);ax.set_ylim(hi[1],lo[1]);ax.plot(q[0],q[1],marker='+',c=color if col==3 else'#25bbee',ms=10,mew=1.5);ax.add_patch(Rectangle((xx-.5,yy-.5),1,1,fill=False,edgecolor='yellow',lw=.8));ax.set_title('Native query crop'if col==1 else'M4: white=target, gray=padding'if col==2 else f'p(present)={float(s["presence_probability"]):.3f}',fontsize=8)
  if col==3:ax.text(.5,-.04,f'Center pixel target={s["presence_target"]}; predicted={int(s["predicted_present"]=="True")}\n'+s['boundary'],ha='center',va='top',transform=ax.transAxes,fontsize=7)
fig.suptitle('Synthetic validation: visible target presence at the initial predicted query',fontsize=13);fig.supxlabel('Nearest single pixel defines presence. Yellow 1-pixel square marks that pixel; + is the unchanged query.\nGray crop padding is outside the source image and is not a target label. Raw RGB and binary mask use nearest-neighbor display.\n3x3 mixed support marks boundary uncertainty only; no patch-vote target. Green predicted present / magenta predicted absent.\nTarget-pixel labels are not semantic-corner correctness or physical DIRECT visibility. No pose evaluation.',fontsize=8)
for ext in ['png','pdf','svg']:fig.savefig(R/('Synthetic_Presence_Cases_v2.'+ext),dpi=160)
plt.close(fig);print('SYNTHETIC_FIGURE_PADDING_REPAIRED_NO_MODEL_OR_LABEL_CHANGE',flush=True)
