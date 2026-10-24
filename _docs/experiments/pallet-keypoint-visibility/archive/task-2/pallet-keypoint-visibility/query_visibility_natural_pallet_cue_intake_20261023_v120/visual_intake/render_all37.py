"""All37 PUBLIC natural cue pairs. Stored RGB only; no model or label changes."""
from pathlib import Path
import hashlib,json,math,sys,traceback
sys.dont_write_bytecode=True
import numpy as np
from PIL import Image,ImageDraw,ImageFont
H=Path(__file__).resolve().parent;V=H.parent
POLICY=V/'ROOT_PUBLIC37_INTAKE_SCOPE_BEFORE_PIXELS.json'
POLICY_SHA='09cf91dac301efafe7de6e38929751bcecb284f6f5636180551fa8d8742be93e'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def crop128(rgb,r):
 h,w=rgb.shape[:2];r=np.asarray(r,np.float64);assert r.shape==(2,)and np.isfinite(r).all()
 x0,y0=[math.floor(float(z))-63 for z in r];out=np.zeros((128,128,3),np.uint8);valid=np.zeros((128,128),bool)
 xa,ya=max(0,x0),max(0,y0);xb,yb=min(w,x0+128),min(h,y0+128)
 if xb>xa and yb>ya:out[ya-y0:yb-y0,xa-x0:xb-x0]=rgb[ya:yb,xa:xb];valid[ya-y0:yb-y0,xa-x0:xb-x0]=True
 return out,valid,[x0,y0,x0+128,y0+128],[float(r[0]-x0),float(r[1]-y0)]
def contain(im,size):
 im=im.copy();im.thumbnail(size,Image.Resampling.LANCZOS);return im
def render():
 assert sha(POLICY)==POLICY_SHA
 protocol=json.loads((H/'PROTOCOL_BEFORE_RGB.json').read_text());assert protocol['code']==bind(__file__)
 for k in('source_manifest','source_seal','render_manifest'):assert sha(protocol[k]['path'])==protocol[k]['sha256']
 rows=[json.loads(l)for l in Path(protocol['render_manifest']['path']).read_text().splitlines()];assert len(rows)==37 and [r['pair_index']for r in rows]==list(range(37))
 assert [r['before_cue_view_index']for r in rows]==sorted(r['before_cue_view_index']for r in rows)
 assert sum(r['fold']=='TRAIN_LAB'for r in rows)==15 and sum(r['fold']=='VAL_HALL'for r in rows)==22
 assert all(r['before']['teacher_label']==0 and r['after']['teacher_label']==1 and r['before']['reference_xy']==r['after']['reference_xy']and r['before']['uncertainty_box_xyxy']==r['after']['uncertainty_box_xyxy']for r in rows)
 out=H/'figures';out.mkdir();font='/usr/share/fonts/truetype/nanum/NanumGothic.ttf';big=ImageFont.truetype(font,23);normal=ImageFont.truetype(font,17);small=ImageFont.truetype(font,14)
 receipts=[];cache={};cards=[]
 for row in rows:
  i=row['pair_index'];card=Image.new('RGB',(1380,1020),'#f0f4f7');d=ImageDraw.Draw(card)
  d.text((18,12),f'{i+1:02d}/37 | {row["pair_id"]} | {row["fold"]} | {row["slot_or_proposal_id"]}',font=big,fill='#16364e')
  d.text((18,49),'기존 before D → after E 약한 AI 주석 · 원 참조/불확실성 유지 · PUBLIC oracle 표시, 모델 입력 아님',font=normal,fill='#805020')
  d.text((18,77),'registration·소유권·native 대응 미입증 / human gold false / 새 레이블·학습 허가 아님',font=normal,fill='#805020')
  stagefacts=[]
  for j,name in enumerate(('before','after')):
   s=row[name];ref=s['RGB'];key=(ref['path'],ref['sha256'])
   if key not in cache:
    assert sha(ref['path'])==ref['sha256'];a=np.array(Image.open(ref['path']).convert('RGB'),dtype=np.uint8,copy=True);assert list(a.shape[:2])==s['raw_hw']and ash(a)==s['decoded_RGB_sha256'];cache[key]=a
   a=cache[key];r=s['reference_xy'];box=s['uncertainty_box_xyxy'];c,valid,bounds,rc=crop128(a,r);base=out/f'PAIR_{i:02d}_{name.upper()}';Image.fromarray(c).save(str(base)+'_CROP128_RAW.png');Image.fromarray(valid.astype(np.uint8)*255).save(str(base)+'_CROP128_VALID.png')
   full=Image.fromarray(a);marked=full.copy();md=ImageDraw.Draw(marked);md.rectangle(box,outline='#ffb000',width=2);x,y=r;md.line((x-9,y,x+9,y),fill='#00ffff',width=2);md.line((x,y-9,x,y+9),fill='#00ffff',width=2)
   x=18+690*j;d.text((x,113),f'{name.upper()} | 원 weak label '+('D 보임'if j==0 else'E 외부가림'),font=big,fill='#16364e');fullview=contain(full,(650,480));card.paste(fullview,(x,150));d.text((x,638),'위: 원 fullRGB 무표시 (표시 크기만 조정)',font=normal,fill='#294c60')
   card.paste(Image.fromarray(c).resize((256,256),Image.Resampling.NEAREST),(x,675));card.paste(contain(marked,(335,252)),(x+285,675));d.text((x,938),'원 native128 무표시 ×2',font=normal,fill='#294c60');d.text((x+285,938),'별도 전체표시: 청록 r / 주황 uncertainty',font=small,fill='#294c60')
   inside=bool(0<=r[0]<a.shape[1]and 0<=r[1]<a.shape[0]);unc_in_crop=bool(box[0]>=bounds[0]and box[1]>=bounds[1]and box[2]<=bounds[2]and box[3]<=bounds[3])
   d.text((x,967),f'r=({r[0]:.3f},{r[1]:.3f}) | crop valid {valid.mean():.3f} | uncertainty in crop={unc_in_crop}',font=small,fill='#805020')
   stagefacts.append(dict(view=name,RGB=ref,decoded_RGB_sha256=ash(a),raw_hw=list(a.shape[:2]),reference_xy=r,uncertainty_box_xyxy=box,native_crop_bounds_half_open=bounds,reference_in_crop_xy=rc,reference_in_native_image=inside,crop_valid_fraction=float(valid.mean()),uncertainty_entirely_in_crop=unc_in_crop,crop128_raw=bind(str(base)+'_CROP128_RAW.png'),crop128_raw_pixel_sha256=ash(c),crop128_valid=bind(str(base)+'_CROP128_VALID.png'),teacher_label_unchanged=s['teacher_label']))
  fn=out/f'PAIR_{i:02d}.png';card.save(fn);cards.append(card);receipts.append(dict(pair_index=i,pair_id=row['pair_id'],fold=row['fold'],before_cue_view_index=row['before_cue_view_index'],after_cue_view_index=row['after_cue_view_index'],stages=stagefacts,card=bind(fn)))
  print(json.dumps(dict(pairs_complete=i+1,total=37)),flush=True)
 contacts=[]
 for start in range(0,37,6):
  sheet=Image.new('RGB',(1800,2040),'#dce6ed')
  for j,card in enumerate(cards[start:start+6]):sheet.paste(card.resize((900,665),Image.Resampling.LANCZOS),((j%2)*900,(j//2)*680))
  fn=out/f'CONTACT_{start//6+1:02d}.png';sheet.save(fn);contacts.append(bind(fn))
 write(H/'RENDER_RECEIPTS.json',dict(completed=True,PASS=True,code=bind(__file__),protocol=bind(H/'PROTOCOL_BEFORE_RGB.json'),pairs=37,views=74,unique_RGB_files=len(cache),native128_crops=74,all_originalorder_and_weaklabels_preserved=True,cases=receipts,contacts=contacts,source_only=True,model_training_current_V112_PnP=0))
 print(json.dumps(dict(completed=True,PASS=True,receipt=bind(H/'RENDER_RECEIPTS.json'),contacts=contacts)),flush=True)
if __name__=='__main__':
 try:render()
 except BaseException:
  write(H/'FAILED_RENDER.json',dict(completed=False,traceback=traceback.format_exc(),no_automatic_restart=True));raise
