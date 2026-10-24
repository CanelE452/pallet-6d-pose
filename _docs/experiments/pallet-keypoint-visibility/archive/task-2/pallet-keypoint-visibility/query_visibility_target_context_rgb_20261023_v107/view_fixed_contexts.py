"""Only the four frozen source diagnostic triplets; no model or new selection."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import json,hashlib
import numpy as np
from PIL import Image,ImageDraw,ImageFont,__version__ as PILVERSION
from target_context_math import sample_target_context
V=Path(__file__).resolve().parent;P=V.parent;O=V/'actual_fixed_context_feasibility'
CASES=P/'query_visibility_frozen_rgb_head_20261023_v105/saved_failure_views/CASES.json'
def bind(p):return dict(path=str(Path(p).resolve()),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def rows(p):return [json.loads(x)for x in Path(p).read_text().splitlines()if x]
def write(p,r):
 with p.open('x')as f:json.dump(r,f,indent=2,allow_nan=False);f.write('\n')
cs=json.loads(CASES.read_text())['cases'];assert len(cs)==4 and sum(len(c['indices'])for c in cs)==12
rs=rows(V/'OCCURRENCES33984_TARGET_CONTEXT_INPUT_ONLY.jsonl');ds={r['context_id']:r for r in rows(V/'TARGET_CONTEXT_DEFINITIONS216_INPUT_ONLY.jsonl')}
O.mkdir(exist_ok=False)
write(O/'SCOPE_BEFORE_PIXEL_READ.json',dict(source_cases=bind(CASES),source_inventory=bind(V/'RESULT.json'),math=bind(V/'target_context_math.py'),code=bind(__file__),indices=[c['indices']for c in cs],full_overviews='BEFORE at each fixed triplet first index; exactly4',no_new_case_selection=True,no_box_q_L_changes=True,no_RGB96_parity_claim=True,new_feature_model_label_fit_private_PnP=0))
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
outputs=[];panels=[]
for ci,c in enumerate(cs):
 sheet=Image.new('RGB',(3*240,280),'white');dd=ImageDraw.Draw(sheet);dd.text((8,5),f'Fixed case{ci}: {c["source"]} {c["split"]}',font=font,fill='black')
 boxes=[];qry=[];scale=[]
 for si,idx in enumerate(c['indices']):
  r=rs[idx];d=ds[r['context_id']];p=Path(r['full_RGB']['path']);actual=bind(p);assert actual==r['full_RGB']
  if r['full_RGB_format']=='CANONICAL_RGB_UINT8_HWC_NPY':
   a=np.load(p,allow_pickle=False);decoder='numpy.load existing canonical RGB'
  else:a=np.array(Image.open(p).convert('RGB'),dtype=np.uint8,copy=True);decoder='Pillow.Image.open.convert(RGB)'
  assert a.dtype==np.uint8 and list(a.shape)==r['raw_hw']+[3]
  decoded=hashlib.sha256(a.tobytes()).hexdigest();assert decoded==r['decoded_RGB_sha256'],(idx,decoded,r['decoded_RGB_sha256'])
  box=d['target_box_xyxy'];assert box is not None
  crop,valid=sample_target_context(a,box,(224,224));u8=np.clip(np.floor(crop.astype(np.float64)+.5),0,255).astype(np.uint8)
  cp=O/f'case_{ci:02d}_{si}_context224_unmarked.png';vp=O/f'case_{ci:02d}_{si}_context_valid.png'
  Image.fromarray(u8).save(cp);Image.fromarray(valid.astype(np.uint8)*255).save(vp)
  sheet.paste(Image.fromarray(u8),(si*240+8,48));dd.text((si*240+8,27),f'{r["variant"]} #{idx}',font=font,fill='black')
  full=None
  if si==0:
   im=Image.fromarray(a).copy();di=ImageDraw.Draw(im);di.rectangle(tuple(box),outline=(255,0,255),width=3)
   x,y=r['query_xy'];di.line((x-7,y,x+7,y),fill=(0,255,255),width=2);di.line((x,y-7,x,y+7),fill=(0,255,255),width=2)
   full=O/f'case_{ci:02d}_before_full_box_q.png';im.save(full)
  outputs.append(dict(case_index=ci,triplet_position=si,global_occurrence_index=idx,variant=r['variant'],full_RGB=actual,decoded_RGB_sha256=decoded,decoder=decoder,raw_hw=r['raw_hw'],context_id=r['context_id'],box_xyxy=box,box_origin=d['box_origin'],query_xy=r['query_xy'],L=r['L'],crop=bind(cp),valid_mask=bind(vp),valid_fraction=float(valid.mean()),box_clipped=False,crop_display_quantization='float32 bilinear values -> nearest integer floor(v+.5),clip[0,255],uint8; stored unmarkedPNG',before_full_box_q=bind(full)if full else None))
  boxes.append(box);qry.append(r['query_xy']);scale.append(r['L'])
 assert all(b==boxes[0]for b in boxes)and all(q==qry[0]for q in qry)and all(l==scale[0]for l in scale)
 sp=O/f'case_{ci:02d}_context_triplet.png';sheet.save(sp);panels.append(bind(sp))
write(O/'ARTIFACTS.json',dict(completed=True,decoder=dict(python=sys.version,numpy=np.__version__,Pillow=PILVERSION),all12_file_and_decoded_SHA_exact=True,cases=outputs,context_triplet_panels=panels,counts=dict(context_crops=12,before_fullframe_box_q_overviews=4),no_new_features_model_labels_fit_current_PnP=True,RGB96_parity_checked=False))
print(json.dumps(dict(completed=True,artifacts=bind(O/'ARTIFACTS.json'),allvalidfractions=[r['valid_fraction']for r in outputs],panels=panels)))
