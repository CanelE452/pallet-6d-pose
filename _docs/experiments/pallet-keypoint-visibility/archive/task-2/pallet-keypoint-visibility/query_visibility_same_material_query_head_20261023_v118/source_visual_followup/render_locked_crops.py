"""Nine exact cached PUBLIC RGB96 crops only. No model/fullframe/GT relabel."""
from pathlib import Path
import hashlib,json,sys
sys.dont_write_bytecode=True
import numpy as np
from PIL import Image,ImageDraw,ImageFont
O=Path(__file__).resolve().parent
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
lock=json.loads((O/'SELECTION_LOCK_BEFORE_PIXELS.json').read_text())
spec=dict(completed=True,code=bind(Path(__file__)),selection=bind(O/'SELECTION_LOCK_BEFORE_PIXELS.json'),operator='Load only bound cached rgb96/crop_valid96 keys, exact9 selected rows; no fullframe decoding, resampling/recentering, encoder or head.',raw_panel='Original96x96 nearest-neighbour x4, unmarked. Separate128px thumbnail has q centre display marker only.',expected={'BEFORE':'D','COVER':'E','SHAM':'D'},native_mapping='UNKNOWN',human_gold=False)
with(O/'RENDER_SPEC_BEFORE_PIXELS.json').open('x')as f:json.dump(spec,f,indent=2)
font='/usr/share/fonts/truetype/nanum/NanumGothic.ttf';normal=ImageFont.truetype(font,17);small=ImageFont.truetype(font,15);big=ImageFont.truetype(font,23)
cache={};checked=set()
def load(ref):
 path=Path(ref['file']['path'])
 if str(path)not in checked:assert bind(path)['sha256']==ref['file']['sha256'];checked.add(str(path))
 key=(str(path),ref['key'])
 if key not in cache:
  with np.load(path,allow_pickle=False)as f:cache[key]=np.array(f[ref['key']],copy=True)
 return np.array(cache[key][ref['index']],copy=True)
titles=['남은 VAL cover E→D 실패','VAL cover C12 D→C13 E 개선','남은 VAL sham D→E 오탐']
cards=[];receipts=[]
for ci,item in enumerate(lock['roles']):
 r=item['original_record'];card=Image.new('RGB',(1440,810),'#eff3f6');draw=ImageDraw.Draw(card)
 draw.text((20,14),f'{ci+1}. {titles[ci]} | base {r["base_row_index"]}, rank {r["donor_rank"]}, donor object {r["donor_object_id"]}',font=big,fill='#15334a')
 draw.text((20,48),f'{r["input_id"]} | q={r["query_xy"]} | L={r["L"]:.4f}px',font=normal,fill='#29475a')
 draw.text((20,76),'공개 통제 합성 · 조건부 cue AUX · native 대응 UNKNOWN · 자연 가림/인간 junction gold 아님',font=normal,fill='#845220')
 images=[];rowreceipts=[]
 for si,s in enumerate(item['stages']):
  a=load(s['rgb96_ref']);v=load(s['valid96_ref']);assert a.shape==(3,96,96)and a.dtype==np.uint8 and v.shape==(1,96,96)
  assert ash(a)==s['RGB96_sha256']and ash(v)==s['valid96_sha256'];images.append(a)
  x=20+si*470;state='E'if si==1 else'D';draw.text((x,112),f'{s["stage"]} | row {s["global_occurrence_index"]} | AUX {state}',font=big,fill='#15334a')
  for k,model in enumerate(('C11','C12','C13')):
   z=s[model+'_z'];pred='E/hidden'if z>=0 else'D/visible';draw.text((x,147+23*k),f'{model}: {z:+.5f} → {pred}',font=normal,fill='#9a3d28'if(model=='C13'and(pred.startswith('E')!=(state=='E')))else'#29475a')
  im=Image.fromarray(a.transpose(1,2,0));im.save(O/f'CASE_{ci:02d}_{s["stage"]}_RGB96_RAW.png');card.paste(im.resize((384,384),Image.Resampling.NEAREST),(x,228))
  marked=im.resize((128,128),Image.Resampling.NEAREST);md=ImageDraw.Draw(marked);md.line((56,63.5,71,63.5),fill='#00ffff',width=2);md.line((63.5,56,63.5,71),fill='#00ffff',width=2);card.paste(marked,(x,630))
  draw.text((x+140,637),'왼쪽: q 표시 전용 사본',font=small,fill='#29475a');draw.text((x+140,664),f'valid {float(v.mean()):.4f}',font=small,fill='#29475a');draw.text((x+140,691),'위: 원 RGB96 무표시 ×4',font=small,fill='#29475a')
  rowreceipts.append(dict(**s,actual_RGB96_sha256=ash(a),actual_valid96_sha256=ash(v),exact_cached_hash_match=True,raw_png=bind(O/f'CASE_{ci:02d}_{s["stage"]}_RGB96_RAW.png')))
 diffs={}
 for si,stage in((1,'COVER'),(2,'SHAM')):
  diffs[stage]={}
  for region,sl in(('central4',slice(47,49)),('central32',slice(32,64)),('full96',slice(0,96))):
   delta=images[si][:,sl,sl].astype(np.int16)-images[0][:,sl,sl].astype(np.int16)
   diffs[stage][region]=dict(changed_pixels=int((delta!=0).any(0).sum()),total_pixels=int(delta.shape[1]*delta.shape[2]),mean_absolute_channel_difference=float(np.abs(delta).mean()))
 draw.text((20,778),'좌표·L·label·문턱 변경0 · C11/C12/C13 저장 logit만 표시 · 상대 margin 통과는 절대 E 정답과 다름',font=normal,fill='#845220')
 out=O/f'CASE_{ci:02d}.png';card.save(out);cards.append(card);receipts.append(dict(role=item['role'],metadata=r,stages=rowreceipts,literal_RGB_differences_vs_BEFORE=diffs,card=bind(out)))
contact=Image.new('RGB',(1440,2430),'white')
for i,im in enumerate(cards):contact.paste(im,(0,i*810))
contact.save(O/'CONTACT_FIXED3_SOURCE_TRIPLES.png')
receipt=dict(completed=True,PASS=True,code=bind(Path(__file__)),selection=bind(O/'SELECTION_LOCK_BEFORE_PIXELS.json'),spec=bind(O/'RENDER_SPEC_BEFORE_PIXELS.json'),cases=receipts,contact=bind(O/'CONTACT_FIXED3_SOURCE_TRIPLES.png'),actual_saved_RGB96_rows=9,actual_saved_valid96_rows=9,all_cached_pixel_hashes_exact=True,fullframe_or_new_crop_calls=0,feature_P3_encoder_head_model_fit_current_V112_V117_PnP=0)
with(O/'RENDER_RECEIPTS.json').open('x')as f:json.dump(receipt,f,indent=2)
print(json.dumps(dict(completed=True,PASS=True,contact=receipt['contact'],cards=[r['card']for r in receipts]),indent=2))
