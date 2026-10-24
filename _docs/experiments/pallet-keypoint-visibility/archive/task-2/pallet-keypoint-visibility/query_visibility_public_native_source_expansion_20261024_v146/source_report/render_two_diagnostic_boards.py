"""Saved-public-source diagnostic boards; untouched stored factual RGB96."""
from pathlib import Path
from collections import Counter
import hashlib,json
import numpy as np
from PIL import Image,ImageDraw,ImageFont

H=Path(__file__).resolve().parent;V=H.parent;S=V/'saved_source64_contact_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(p),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def check(b):assert sha(b['path'])==b['sha256'];return Path(b['path'])
def font(n,bold=False):return ImageFont.truetype('/usr/share/fonts/truetype/nanum/NanumGothic'+('Bold'if bold else'')+'.ttf',n)
def write(p,v):
 with Path(p).open('x')as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def wrap(d,text,f,w):
 words=text.split();out=[];line=''
 for word in words:
  trial=(line+' '+word).strip()
  if line and d.textlength(trial,font=f)>w:out.append(line);line=word
  else:line=trial
 if line:out.append(line)
 return out

rootpath=V/'ROOT_SOURCE64_WEAK_LABEL_ADJUDICATION_COMPLETE.json'
assert sha(rootpath)=='3afed99e41eb58ccbf29e181703630d0ae42d43c03165251c597f7faf4f30a8b'
root=read(rootpath);assert root['completed']and root['PASS']
labels=[json.loads(x)for x in check(root['labels']).read_text().splitlines()]
facts=[json.loads(x)for x in check(root['facts']).read_text().splitlines()]
assert len(labels)==len(facts)==512
for j,(r,f)in enumerate(zip(labels,facts)):
 assert r['global_index']==f['global_index']==j
 for k in('id','native_slot','query_xy','L','selected_index','original_input_valid','no_detection','rgb96_SHA256','valid96_SHA256'):
  assert r[k]==f[k]
assert dict(Counter(r['semantic_state']for r in labels))==dict(U=428,D=70,E=14)
seal=read(S/'CONTACT_RENDER_COMPLETE.json');assert seal['completed']and seal['PASS']and seal['frames']==64 and seal['slots']==512
transport=read(check(seal['RGB_transport_seal']));rgbrows=read(check(transport['manifest']))['rows'];assert len(rgbrows)==64
init=read(check(seal['initializer_prediction_seal']));op={Path(b['path']).name:b for b in init['outputs']}
props=[json.loads(x)for x in check(op['FROZEN_PROPOSALS_ALL64_INPUT_ONLY.jsonl']).read_text().splitlines()]
assert len(props)==64

cases=[
 (12,3,'D','LAB · 보이는 하부','같은 포즈 계열에서 왼쪽 하부 지지 구조가 노출되어 있습니다.'),
 (15,3,'E','LAB · 외부 가림','별도 종이상자가 같은 조립체의 왼쪽 하부 역할을 가립니다.'),
 (37,3,'D','HALL · 보이는 하부','상자는 왼쪽에 떨어져 있고 조립체의 왼쪽 하부가 보입니다.'),
 (33,3,'E','HALL · 외부 가림','앞의 종이상자가 조립체 왼쪽 하부를 가립니다.'),
 (2,3,'U','다른 개체 선택 · U','추론은 노란 기둥 받침을 선택했습니다. 뒤 목재 조립체의 라벨을 옮길 수 없습니다.'),
 (50,2,'D','teacher D · 입력 q는 영상 밖','오른쪽 하부 역할은 보이지만 원 q는 OOB입니다. 입력 보류와 의미 D를 구분합니다.'),
 (8,3,'D','큰 좌표 어긋남 · 역할은 보임','원 crop은 바닥을 보지만 같은 조립체의 왼쪽 하부 역할은 노출되어 있습니다.'),
 (57,3,'U','개체 대응 모호 · 주 판정 U','추론 cuboid가 상자와 목재 통 사이에 걸칩니다. 원 조건부 D 제안은 별도 보존합니다.')]
color={'D':(17,111,68),'E':(171,61,36),'U':(88,86,112)}
receipts=[]
def panel(canvas,x,y,case):
 i,k,state,title,note=case;r=labels[i*8+k];p=props[i];t=rgbrows[i]
 assert r['semantic_state']==state and r['native_slot']==k and r['id']==p['id']==t['id']
 rgb=np.load(check(t['canonical_RGB_NPY']),allow_pickle=False)
 assert rgb.dtype==np.uint8 and hashlib.sha256(rgb.tobytes()).hexdigest()==t['decoded_RGB_sha256']==p['decoded_RGB_sha256']
 original=Image.fromarray(rgb);context=original.copy();d=ImageDraw.Draw(context)
 selected=p['selected_index']
 if selected is not None:d.rectangle(p['candidates'][selected]['box_xyxy'],outline=(255,145,0),width=2)
 q=np.asarray(r['query_xy'],np.float64);ref=np.asarray(r['source_reference_xy'],np.float64)
 hh,ww=rgb.shape[:2]
 qin=np.isfinite(q).all()and 0<=q[0]<ww and 0<=q[1]<hh
 rin=np.isfinite(ref).all()and 0<=ref[0]<ww and 0<=ref[1]<hh
 if qin and rin and state in('D','E'):d.line([tuple(q),tuple(ref)],fill=(245,245,245),width=1)
 if qin:
  xx,yy=q;d.ellipse((xx-5,yy-5,xx+5,yy+5),outline=(255,0,210),width=2)
  d.text((max(0,min(xx+7,ww-70)),max(0,min(yy-25,hh-25))),f'q{k}',font=font(22,True),fill=(255,0,210),stroke_width=1,stroke_fill='black')
 else:d.text((ww-165,12),f'q{k}: 영상 밖',font=font(20,True),fill=(255,0,210),stroke_width=1,stroke_fill='black')
 if rin:
  xx,yy=ref;d.line((xx-6,yy,xx+6,yy),fill=(0,255,255),width=2);d.line((xx,yy-6,xx,yy+6),fill=(0,255,255),width=2)
  d.text((max(0,min(xx+7,ww-70)),max(0,min(yy+2,hh-25))),f'R{k}',font=font(22,True),fill=(0,255,255),stroke_width=1,stroke_fill='black')
 sheet=seal['sheets'][i];assert sheet['frame_index']==i
 with np.load(check(sheet['crop_packet']),allow_pickle=False)as z:
  assert int(z['global_index'][k])==i*8+k;crop=z['rgb96'][k].copy();valid=z['valid96'][k].copy()
 assert crop.shape==(96,96,3)and crop.dtype==np.uint8
 assert hashlib.sha256(crop.tobytes()).hexdigest()==r['rgb96_SHA256']
 assert hashlib.sha256(valid.tobytes()).hexdigest()==r['valid96_SHA256']
 cd=ImageDraw.Draw(canvas)
 cd.rounded_rectangle((x+8,y+8,x+952,y+648),radius=13,fill='white',outline=(216,221,229),width=2)
 cd.text((x+24,y+23),f'{i:02d} · {r["source_id"]} · native{k}',font=font(26,True),fill=(28,35,45))
 cd.text((x+24,y+62),title,font=font(28,True),fill=color[state])
 cd.text((x+24,y+102),'전체 장면 표시용 복사본 · q 자홍 / 역할 참조 R 청록',font=font(19),fill=(74,82,95))
 canvas.paste(context.resize((576,432),Image.Resampling.NEAREST),(x+24,y+132))
 cd.text((x+628,y+104),'원 q/L의 RGB96',font=font(23,True),fill=(28,35,45))
 canvas.paste(Image.fromarray(crop).resize((288,288),Image.Resampling.NEAREST),(x+628,y+145))
 cd.text((x+628,y+449),f'약한 의미 판정: {state}',font=font(26,True),fill=color[state])
 cd.text((x+628,y+488),'입력 유효'if r['original_input_valid']else'입력 사용 불가 · OOB',font=font(22,True),fill=(45,63,82)if r['original_input_valid']else(166,67,39))
 cd.text((x+628,y+524),f'L={r["L"]:.2f} · 원96 → 표시3배',font=font(19),fill=(85,91,101))
 yy=y+578
 for line in wrap(cd,note,font(22),904):
  cd.text((x+24,yy),line,font=font(22),fill=(40,49,62));yy+=29
 assert yy<=y+646,(i,yy)
 receipts.append(dict(frame_index=i,native_slot=k,global_index=i*8+k,source_id=r['source_id'],teacher_state=state,
  original_input_valid=r['original_input_valid'],query_xy=r['query_xy'],L=r['L'],selected_index=r['selected_index'],
  crop_packet=sheet['crop_packet'],rgb96_SHA256=r['rgb96_SHA256'],valid96_SHA256=r['valid96_SHA256'],
  RGB96_loaded_untouched_no_marks_or_recenter=True,display_only_NEAREST_scale=3,context_overlay_copy_only=True,
  note_KO=note,root_conditional_original_proposal=r.get('conditional_original_proposal')))

outputs=[]
for board,title,subset,filename in[(1,'같은 포즈 계열에서 보임과 외부 가림 대조',cases[:4],'BOARD_01_VISIBLE_EXTERNAL_PAIRS.png'),
 (2,'입력 위치·개체 대응과 보임 판정은 따로 확인',cases[4:],'BOARD_02_ENTITY_INPUT_AND_ROLE_CASES.png')]:
 im=Image.new('RGB',(1920,1530),(241,244,249));d=ImageDraw.Draw(im)
 d.text((25,21),title,font=font(40,True),fill=(22,32,45))
 d.text((25,79),'공개 실사 · 하부2/3의 구조적 assistant weak 판정 · 사람 gold 아님',font=font(26),fill=(68,80,95))
 for j,c in enumerate(subset):panel(im,(j%2)*960,130+(j//2)*650,c)
 d.text((25,1455),'전체64장·고유512입력 유지 | 원 q/L·첫 선택 유지 | 위8사례는 진단 예시이며 모델 성능 결과가 아닙니다.',font=font(23),fill=(50,62,78))
 d.text((25,1491),'R은 제공 포즈의 제한된 가상 하부 역할 참조입니다. 정밀 물리 꼭짓점·픽셀 gold 또는 전체8점 라벨을 뜻하지 않습니다.',font=font(20),fill=(79,88,101))
 target=H/filename;assert not target.exists();im.save(target);outputs.append(bind(target))
index=H/'INDEX.html'
index.write_text('<!doctype html><meta charset="utf-8"><title>공개 source 가림 진단</title><style>body{font-family:sans-serif;max-width:1500px;margin:auto}img{width:100%}</style><h1>공개 source 실제 crop 진단</h1><p>제한된 하부 역할의 assistant weak 판정입니다. 원64장·고유512입력은 모두 유지했습니다.</p>'+''.join(f'<img src="{Path(o["path"]).name}">'for o in outputs))
write(H/'RENDER_RECEIPT.json',dict(completed=True,QA_view_pending=True,code=bind(__file__),root_adjudication=bind(rootpath),labels=root['labels'],contact_seal=bind(S/'CONTACT_RENDER_COMPLETE.json'),
 outputs=outputs,index=bind(index),fixed_cases=receipts,all_original64_frames_and_unique512_slots_retained=True,all512_q_L_ID_firstargmax_inputflags_cropSHA_join_exact=True,
 source_root_counts=dict(Counter(r['semantic_state']for r in labels)),source_weak_not_human_gold=True,partial_lower_bridge_only=True,new_GT_centered_crop_or_q_L_correction=False,
 new_model_predictions_or_fit_or_private_current223_V112_PnP=0,original_files_modified=False))
print(json.dumps(dict(outputs=outputs,render_receipt=bind(H/'RENDER_RECEIPT.json'))))
