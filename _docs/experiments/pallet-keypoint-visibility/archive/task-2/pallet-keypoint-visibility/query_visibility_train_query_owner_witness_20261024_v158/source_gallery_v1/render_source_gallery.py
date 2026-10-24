import json, hashlib, shutil
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUT=Path(__file__).resolve().parent
BASE=OUT.parent
EVIDENCE=BASE/'review_evidence_preparation'
selected=json.loads((OUT/'SELECTED_SIX_CASES.json').read_text())
manifests=[json.loads((EVIDENCE/f'REVIEW_INPUT_GROUP_{g}.json').read_text()) for g in ['A','B']]
cases={c['query_id']:c for m in manifests for c in m['cases']}
REG='/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf'
BOLD='/usr/share/fonts/truetype/nanum/NanumBarunGothicBold.ttf'
font=lambda n,b=False:ImageFont.truetype(BOLD if b else REG,n)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
binding=lambda p:{'path':str(p),'sha256':sha(p)}
SIZE=(1540,820)
FG='#172735';MUTED='#516170';ACCENT='#246579';BG='#f3f6f8'
titles=['D + TARGET','D + FOREIGN','D + BACKGROUND','E + FOREIGN','E + TARGET','final UNKNOWN']
captions=[
 '중앙 네 샘플: 같은 팔레트에 이어진 목재 면. 원래 D는 그대로 유지.',
 '중앙 네 샘플: 팔레트 위 흰 포장 화물의 측면. 외부 E로 바꾸는 근거가 아님.',
 '중앙 네 샘플: 팔레트 밖 카펫. 배경은 가상 모서리 위치일 수도 있음.',
 '중앙 네 샘플: 뒤 검은 지지대. 원래 E와 보이는 소유 관찰을 따로 기록.',
 '중앙 네 샘플: 흰 화물 옆 팔레트 목재. 해당 네이티브 키포인트의 가시성을 뜻하지 않음.',
 '중앙 네 샘플: 목재 윗윤곽과 카펫 경계. 단일 소유를 확정하지 않음.',
]
records=[];cards=[]
for i,s in enumerate(selected):
 r=s['row'];c=cases[r['query_id']];case_id=c['case_id']
 srcdir=OUT/'source_assets'/case_id;srcdir.mkdir(parents=True,exist_ok=True)
 source={}
 for name in ['panel','raw96','central32']:
  b=c['images'][name];p=Path(b['path']);assert sha(p)==b['sha256']
  cp=srcdir/p.name;shutil.copyfile(p,cp);assert sha(cp)==b['sha256']
  source[name]={'sealed_input':b,'byte_identical_copy':binding(cp)}
 panel=Image.open(srcdir/'REVIEW_PANEL.png').convert('RGB')
 assert panel.size==(1600,1000)
 # Pixel-for-pixel extraction of the existing panel context; its magenta q is a display overlay.
 context=panel.crop((20,130,930,642))
 context_path=srcdir/'CONTEXT_FROM_SEALED_PANEL.png';context.save(context_path)
 raw=Image.open(srcdir/'RAW_RGB96.png').convert('RGB')
 center=Image.open(srcdir/'RAW_CENTRAL32.png').convert('RGB')
 assert raw.size==(96,96) and center.size==(32,32)
 assert raw.crop((32,32,64,64)).tobytes()==center.tobytes()
 four=raw.crop((47,47,49,49))
 card=Image.new('RGB',SIZE,'white');d=ImageDraw.Draw(card)
 d.rectangle((0,0,1540,10),fill=ACCENT)
 d.text((24,28),f'{i+1:02d}  {titles[i]}',font=font(36,True),fill=FG)
 badge='교차 3관찰 일치 · 약관찰' if s['preferred_pool_used'] else ('단독 약관찰 · UNKNOWN' if r['final_visible_owner_support']=='UNKNOWN' else '단독 약관찰')
 badgew=d.textbbox((0,0),badge,font=font(25,True))[2]+34
 d.rounded_rectangle((1516-badgew,26,1516,70),radius=10,fill='#e8f0f3')
 d.text((1533-badgew,35),badge,font=font(25,True),fill=ACCENT)
 q=r['original_query_xy'];d.text((24,84),f'{case_id}  |  원래 역할 {r["original_role_state"]}  |  q=({q[0]:2.2f}, {q[1]:2.2f})  L={r["original_L"]:.2f}',font=font(24),fill=MUTED)
 d.text((24,120),'저장된 문맥 · 자홍색 표시는 원래 q',font=font(24,True),fill=FG)
 d.text((968,120),'정확한 RGB96 · 3×',font=font(23,True),fill=FG)
 d.text((1284,120),'중앙32 · 6×',font=font(23,True),fill=FG)
 card.paste(context,(24,158))
 card.paste(raw.resize((288,288),Image.Resampling.NEAREST),(968,158))
 card.paste(center.resize((192,192),Image.Resampling.NEAREST),(1284,158))
 d.text((1284,382),'중앙 2×2 · 64×',font=font(23,True),fill=FG)
 card.paste(four.resize((128,128),Image.Resampling.NEAREST),(1284,420))
 d.text((968,477),'원본 픽셀 유지',font=font(23),fill=MUTED)
 d.text((968,512),'RGB96 [47:49, 47:49]',font=font(23),fill=MUTED)
 d.text((968,554),'확대: 최근접 이웃',font=font(23),fill=MUTED)
 d.text((968,594),f'중앙 4샘플 유효: {r["center4_valid"]}',font=font(23),fill=MUTED)
 d.rounded_rectangle((24,694,1516,751),radius=8,fill=BG)
 assert d.textbbox((0,0),captions[i],font=font(26))[2]<1450
 d.text((40,709),captions[i],font=font(26),fill=FG)
 footer='중앙 소유 약관찰 ≠ 네이티브 가시성·좌표·물리적 가림 정답'
 d.text((24,775),footer,font=font(24,True),fill=ACCENT)
 dest=OUT/f'CARD_{i+1:02d}_{case_id}.png';card.save(dest);cards.append(card)
 records.append({'category':s['category'],'case_id':case_id,'query_id':r['query_id'],
                 'original_role':r['original_role_state'],'final_visible_owner_support':r['final_visible_owner_support'],
                 'observation_scope':r['observation_scope'],'caption_KO':captions[i],
                 'source_evidence_reason_KO':r['primary_reason_KO'],'source_assets':source,
                 'context':{**binding(context_path),'crop_box_in_existing_panel':[20,130,930,642],
                            'pixel_sha256':hashlib.sha256(context.tobytes()).hexdigest(),'contains_original_q_display_overlay':True},
                 'card':binding(dest)})
contact=Image.new('RGB',(3104,2492),'#dbe3e8')
for i,card in enumerate(cards):contact.paste(card,((i%2)*1552,(i//2)*832))
contact_path=OUT/'SOURCE_DIAGNOSTIC_GALLERY_SIX.png';contact.save(contact_path)
(OUT/'GALLERY_RENDER_MANIFEST.json').write_text(json.dumps({'cases':records,'contact':binding(contact_path),'display_resampling':'nearest only','cards_size':SIZE,'contact_size':contact.size,'source_raw_png_bytes_preserved':True},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'cards':[r['card'] for r in records],'contact':binding(contact_path)},ensure_ascii=False,indent=2))
