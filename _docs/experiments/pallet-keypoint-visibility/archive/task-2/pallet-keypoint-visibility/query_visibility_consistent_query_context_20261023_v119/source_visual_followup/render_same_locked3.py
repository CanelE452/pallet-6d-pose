"""PUBLIC saved pixels only: exact C13 lock, unchanged RGB96, old/new contexts."""
from pathlib import Path
import hashlib,importlib.util,json,sys
sys.dont_write_bytecode=True
import numpy as np
from PIL import Image,ImageDraw,ImageFont
H=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(p),sha256=sha(p))
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
lock=json.loads((H/'SELECTION_AND_RENDER_LOCK_BEFORE_PIXELS.json').read_text())
for k in('original_selection','original_render_receipts','C14_source_comparison','input_metadata','context_math','native_RGB_replay','actual_C14_context_facts'):
 assert sha(lock[k]['path'])==lock[k]['sha256'],k
spec=dict(completed=True,code=bind(Path(__file__)),selection=bind(H/'SELECTION_AND_RENDER_LOCK_BEFORE_PIXELS.json'),operation='Reuse exact9 saved RGB96 PNGs; hashcheck decoded CHW bytes. Decode9 bound PUBLIC nativeRGB and sample224 with immutable V107 bilinear noexpansion/no-clipping math, once old selected box and once original finiteq8minmax. No encoder/head.',raw_display='RGB96 nearest x3 unmarked; target224 is F32sample rounded/clipped only for unmarked PNG display. No q/GT pixel painting.',model_input_note='C13/C14 local F/C/P exactly same. Context T/Q and associated scalars differ; head retrained with identical frozen source schedule.',no_new_case_selection=True,new_model_fit_private_V112_V117_PnP=0)
write(H/'RENDER_SPEC_BEFORE_PIXELS.json',spec)
s=importlib.util.spec_from_file_location('v119_public_visual_native_replay',lock['native_RGB_replay']['path']);replay=importlib.util.module_from_spec(s);s.loader.exec_module(replay)
recipes=[t['context_recipe']for r in lock['roles']for t in r['stages']];reader=replay.NativeRGBReplay(recipes,allow_public_payload_reads=True);math=replay.exact_math()
facts={r['context_group_id']:r for r in map(json.loads,Path(lock['actual_C14_context_facts']['path']).read_text().splitlines())}
font='/usr/share/fonts/truetype/nanum/NanumGothic.ttf';normal=ImageFont.truetype(font,18);small=ImageFont.truetype(font,15);big=ImageFont.truetype(font,25)
titles=['남은 cover 실패: C13·C14 모두 D','C13에서 고친 cover: C14도 E','sham 오탐: C13·C14 모두 E'];cards=[];receipts=[]
def response(z):return'D 보임'if z<0 else'E 외부가림'
for ci,item in enumerate(lock['roles']):
 r=item['record'];card=Image.new('RGB',(1700,900),'#eef3f6');draw=ImageDraw.Draw(card)
 draw.text((20,15),f'{ci+1}. {titles[ci]} | base {r["base_row_index"]} / donor rank {r["donor_rank"]}, object {r["donor_object_id"]}',font=big,fill='#17374e')
 draw.text((20,54),f'{r["input_id"]} | native logical slot {r["native_slot_logical_only"]} | q={r["query_xy"]} | L={r["L"]:.4f}px',font=normal,fill='#294a5d')
 draw.text((20,86),'같은 C13 고정 사례 재사용 · 공개 cutout 조건부 AUX · native 대응 UNKNOWN · 자연 가림/인간 정답 아님',font=normal,fill='#80501d')
 rr=[]
 for j,t in enumerate(item['stages']):
  u=t['context_recipe'];x=20+560*j;expected='E'if j==1 else'D';draw.text((x,128),f'{t["stage"]} | row {t["global_occurrence_index"]} | AUX {expected}',font=big,fill='#17374e')
  for k,model in enumerate(('C13','C14')):
   z=t[model+'_z'];wrong=(z>=0)!=(expected=='E');draw.text((x,168+27*k),f'{model}: z {z:+.6f} → {response(z)}',font=normal,fill='#a13527'if wrong else'#294a5d')
  assert sha(t['raw_local_png']['path'])==t['raw_local_png']['sha256']
  raw=np.array(Image.open(t['raw_local_png']['path']).convert('RGB'));assert raw.shape==(96,96,3)and ash(raw.transpose(2,0,1))==t['RGB96_sha256']
  card.paste(Image.fromarray(raw).resize((288,288),Image.Resampling.NEAREST),(x,235))
  draw.text((x+300,242),'원 RGB96 ×3',font=normal,fill='#294a5d');draw.text((x+300,270),'q/L·local crop 동일',font=small,fill='#294a5d');draw.text((x+300,298),'무표시/재중심화 없음',font=small,fill='#294a5d')
  rgb=reader.image(t['input_unit_index']);out=[]
  for k,boxkey in enumerate(('old_box_xyxy','proposed_q8_box_xyxy')):
   a,valid=math.sample_target_context(rgb,u[boxkey]);fraction=float(np.float32(valid.mean()));name='C13_OLD_BOX'if k==0 else'C14_Q8_BOX';fn=H/f'CASE_{ci:02d}_{t["stage"]}_{name}_224.png';Image.fromarray(np.rint(a).clip(0,255).astype(np.uint8)).save(fn)
   card.paste(Image.open(fn),(x+240*k,572));draw.text((x+240*k,542),'C13 기존 selected-box'if k==0 else'C14 초기 q8 minmax',font=normal,fill='#294a5d');draw.text((x+240*k,807),f'valid fraction {fraction:.4f}',font=small,fill='#294a5d')
   out.append(dict(kind=name,box_xyxy=u[boxkey],raw_F32sample_sha256=ash(a),valid_sha256=ash(valid),valid_fraction=fraction,display=bind(fn)))
   if k==1:
    saved=facts[u['context_group_id']];assert ash(a)==saved['target_RGB224_F32_sha256']and fraction==saved['target_valid_fraction']
  rr.append(dict(stage=t['stage'],global_occurrence_index=t['global_occurrence_index'],input_unit_index=t['input_unit_index'],query_xy=t['query_xy'],L=t['L'],localPNG=t['raw_local_png'],decoded_local_RGB96_exact=True,context_fullRGB=u['full_RGB'],decoded_RGB_sha256=ash(rgb),C13_z=t['C13_z'],C14_z=t['C14_z'],contexts=out,new_context_matches_actual_producer_sample_exact=True))
 draw.text((20,858),'표시용224는 같은 nativeRGB의 고정 box 샘플. 전체 target 소유권·native 정답·재질 원인·개별 성능을 증명하지 않습니다.',font=normal,fill='#80501d')
 fn=H/f'CASE_{ci:02d}.png';card.save(fn);cards.append(card);receipts.append(dict(original_role=item['original_role'],indices=item['triple_global_indices'],card=bind(fn),stages=rr))
contact=Image.new('RGB',(1700,2700),'white')
for i,card in enumerate(cards):contact.paste(card,(0,900*i))
contact.save(H/'CONTACT_SAME_FIXED3_C13_C14.png')
write(H/'RENDER_RECEIPTS.json',dict(completed=True,PASS=True,code=bind(Path(__file__)),selection=bind(H/'SELECTION_AND_RENDER_LOCK_BEFORE_PIXELS.json'),spec=bind(H/'RENDER_SPEC_BEFORE_PIXELS.json'),cases=receipts,contact=bind(H/'CONTACT_SAME_FIXED3_C13_C14.png'),saved_RGB96_rows=9,decoded_PUBLIC_fullRGB=9,sampled_context_views18=18,all9_C14_new_context_sample_hashes_match_actual_producer=True,new_encoder_head_fit_private_current_V112_V117_PnP=0))
print(json.dumps(dict(completed=True,cards=[r['card']for r in receipts],contact=bind(H/'CONTACT_SAME_FIXED3_C13_C14.png')),indent=2))
