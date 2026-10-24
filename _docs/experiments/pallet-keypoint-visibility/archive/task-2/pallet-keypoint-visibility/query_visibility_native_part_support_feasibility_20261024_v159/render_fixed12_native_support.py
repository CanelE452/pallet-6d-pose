"""Exact-source diagnostic views; original query inputs remain immutable."""
from pathlib import Path
import hashlib,json,datetime,math,textwrap
import numpy as np
import cv2
from PIL import Image,ImageDraw,ImageFont
V=Path(__file__).resolve().parent;P=V.parent
OLD=P/'query_visibility_train_query_owner_witness_20261024_v158'
SRC=P/'query_visibility_mr6d_full_public_source_20261024_v153'
O=V/'native_support_visual12'
def read(p):return json.loads(Path(p).read_text())
def lines(p):return[json.loads(x)for x in Path(p).read_text().splitlines()if x]
def bind(p):
 p=Path(p).resolve();assert '_v112'not in str(p).lower()
 return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def check(b):assert bind(b['path'])==b;return Path(b['path'])
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
FONT='/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
def font(n):return ImageFont.truetype(FONT,n)
def label(d,xy,t,n=24,color='#20364b'):d.text(xy,t,font=font(n),fill=color)
def marker(d,x,y,c):
 d.line((x-11,y,x+11,y),fill=c,width=4);d.line((x,y-11,x,y+11),fill=c,width=4)
def main():
 assert not O.exists();O.mkdir()
 mapping_path=V/'native_mask_support_audit/ALL390_EXACT_SELECTED_NATIVE_REFERENCE_MAPPING.jsonl'
 assert bind(mapping_path)['sha256']=='0a79af475f916dc67e75ca8b323745bdf7a60059fce59a2b18868b5252424504'
 rows=lines(mapping_path);assert len(rows)==390 and all(x['split']=='SOURCE_DEVELOPMENT'for x in rows)
 used=set();chosen=[]
 def pick(name,pred):
  pool=[r for r in rows if pred(r)and r['global_index']not in used]
  assert pool
  row=min(pool,key=lambda r:hashlib.sha256((r['query_id']+'V159_NATIVE_VISUAL').encode()).hexdigest())
  used.add(row['global_index']);chosen.append(dict(bucket=name,row=row))
 for k in range(8):pick('D_NATIVE_'+str(k),lambda r,k=k:r['original_role_state']=='D'and r['native_slot']==k)
 for k in (4,5):pick('E_NATIVE_'+str(k),lambda r,k=k:r['original_role_state']=='E'and r['native_slot']==k)
 pick('D_FOREIGN',lambda r:r['original_role_state']=='D'and r['weak_visible_owner_support']=='FOREIGN_VISIBLE_SUPPORTED')
 pick('E_TARGET',lambda r:r['original_role_state']=='E'and r['weak_visible_owner_support']=='TARGET_VISIBLE_SUPPORTED')
 rgbman=read(SRC/'full_source_v1/RGB230_INPUT_ONLY_MANIFEST.json')
 rgbs={x['frame_index']:x for x in rgbman['rows']}
 cases={c['global_index']:c for g in ('A','B')for c in read(OLD/('review_evidence_preparation/REVIEW_INPUT_GROUP_'+g+'.json'))['cases']}
 selection=dict(completed=True,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),rule='D each native0..7, E native4/5, D_FOREIGN, E_TARGET; smallest SHA256(query_id+V159_NATIVE_VISUAL), excluding previously chosen IDs. No performance selection.',mapping=bind(mapping_path),cases=[dict(bucket=c['bucket'],global_index=c['row']['global_index'],query_id=c['row']['query_id'],frame_index=c['row']['frame_index'],native_slot=c['row']['native_slot'],q=c['row']['query_xy'],L=c['row']['L'],selected_gt_instance_index=c['row']['selected_gt_instance_index'],nominal_reference=c['row']['reference_xy'])for c in chosen],TRAIN_RGB_or_crop_body_read_or_hashed_before_seal=0,
  display_only_nominal_zoom='The second96 view is a diagnostic display at the provided-pose nominal reference with original L. It is never a model input and does not alter q/L/GT.',no_replacement=True)
 write(O/'FIXED12_SELECTION_BEFORE_RGB_BODIES.json',selection)
 cards=[];cache={}
 for no,item in enumerate(chosen,1):
  r=item['row'];i=r['global_index'];c=cases[i];rgbrec=rgbs[r['frame_index']]
  assert rgbrec['split']=='SOURCE_DEVELOPMENT'and c['query_xy']==r['query_xy']and c['L']==r['L']
  if r['frame_index']not in cache:
   pp=check(rgbrec['canonical_RGB_NPY']);a=np.load(pp,allow_pickle=False)
   assert a.shape==(720,1280,3)and a.dtype==np.uint8
   assert hashlib.sha256(a.tobytes()).hexdigest()==rgbrec['decoded_RGB_sha256'];cache[r['frame_index']]=a
  rgb=cache[r['frame_index']];rawpath=check(c['images']['raw96']);raw=np.asarray(Image.open(rawpath).convert('RGB'))
  assert raw.shape==(96,96,3)and hashlib.sha256(raw.tobytes()).hexdigest()==c['raw_RGB96_array_sha256']
  q=np.array(r['query_xy']);ref=np.array(r['reference_xy']);L=r['L'];delta=(np.arange(96,dtype=np.float32)+.5)/96-.5
  mx,my=np.meshgrid((ref[0]+delta*L).astype(np.float32),(ref[1]+delta*L).astype(np.float32))
  nom=cv2.remap(rgb,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
  canvas=Image.new('RGB',(1800,1120),'white');d=ImageDraw.Draw(canvas)
  label(d,(24,18),f'{no:02d}  Q{i:04d}  native{r["native_slot"]} · {r["native_name"]}',31)
  label(d,(24,65),f'원래 약한 역할 {r["original_role_state"]} / 중심 소유 {r["weak_visible_owner_support"]}',24)
  dist=float(np.linalg.norm(q-ref));label(d,(24,102),f'q=({q[0]:.2f}, {q[1]:.2f}) · L={L:.2f} · 명목 위치와 거리 {dist:.2f}px (실제 오차 정답 아님)',23)
  full=Image.fromarray(rgb).resize((960,540),Image.Resampling.BILINEAR);fd=ImageDraw.Draw(full);sc=.75
  for k,xy in enumerate(r['selected_instance_geometry']['dynamic_projected8']):
   x,y=np.array(xy)*sc
   if 0<=x<960 and 0<=y<540:
    fd.ellipse((x-4,y-4,x+4,y+4),outline='#43c7ff',width=2);fd.text((x+5,y+2),str(k),font=font(16),fill='#43c7ff')
  x,y=q*sc;xx,yy=ref*sc
  fd.rectangle(((q[0]-L/2)*sc,(q[1]-L/2)*sc,(q[0]+L/2)*sc,(q[1]+L/2)*sc),outline='#e82ad4',width=2)
  if 0<=xx<960 and 0<=yy<540:fd.ellipse((xx-11,yy-11,xx+11,yy+11),outline='#ffe23b',width=4)
  marker(fd,x,y,'#f329df');fd.line((x,y,xx,yy),fill='#ffe23b',width=2)
  canvas.paste(full,(24,190));label(d,(24,151),'원본 장면 · 자홍색 q/원래 crop · 노란 원=제공 pose의 명목 native 위치',23)
  label(d,(1010,151),'원래 query RGB96 · 모델 입력 그대로',23)
  canvas.paste(Image.fromarray(raw).resize((384,384),Image.Resampling.NEAREST),(1010,190))
  label(d,(1414,151),'명목 위치 확대 · 입력 아님',22)
  canvas.paste(Image.fromarray(nom).resize((384,384),Image.Resampling.NEAREST),(1414,190))
  label(d,(1010,595),'두 확대 모두 L은 동일 · 최근접 표시 4×',23)
  label(d,(1010,635),'오른쪽은 진단용 새 창이며 q를 고치지 않음',22)
  label(d,(24,780),'확인할 질문: 원래 q의 crop 안에 해당 native 부위가 있는가?',29)
  label(d,(24,835),'노란 위치는 CAD/제공 자세의 명목 참조입니다. 실사 물리 좌표·가림 정답으로 확정하지 않습니다.',25)
  label(d,(24,887),'목재가 보여도 같은 native라는 보장은 없습니다. 배경이라고 외부 가림·좌표 오류로 바꾸지 않습니다.',24)
  label(d,(24,940),'고정 공개 TRAIN 12개 진단 예시 · 전체 390개/원래 역할/좌표/분할 유지 · 학습 편입 없음',24)
  dest=O/f'CASE_{no:02d}_Q{i:04d}.png';canvas.save(dest)
  cards.append(dict(bucket=item['bucket'],global_index=i,query_id=r['query_id'],original_q=r['query_xy'],original_L=L,nominal_reference=r['reference_xy'],nominal_delta_px=dist,selected_gt_instance_index=r['selected_gt_instance_index'],source_RGB=rgbrec['canonical_RGB_NPY'],original96=c['images']['raw96'],original96_array_sha=c['raw_RGB96_array_sha256'],panel=bind(dest),native_physical_coordinate_or_visibility_certification='UNKNOWN',training_admitted=False))
 contacts=[]
 for page in range(3):
  im=Image.new('RGB',(3600,2240),'#e1e8ed')
  for j,c in enumerate(cards[page*4:(page+1)*4]):im.paste(Image.open(c['panel']['path']),(j%2*1800,j//2*1120))
  dest=O/f'CONTACT_{page+1:02d}.png';im.save(dest);contacts.append(bind(dest))
 write(O/'RENDER_COMPLETE.json',dict(completed=True,source_only=True,selection=bind(O/'FIXED12_SELECTION_BEFORE_RGB_BODIES.json'),renderer=bind(__file__),cards=cards,contacts=contacts,original_inputs_or_roles_or_qL_changed=False,nominal_zoom_used_as_model_input=False,new_training_model_PnP_private_held_V112_calls=0,visual_QA_pending=True))
 print(json.dumps(dict(completed=True,selected=[x['global_index']for x in cards],contacts=contacts),ensure_ascii=False))
if __name__=='__main__':main()
