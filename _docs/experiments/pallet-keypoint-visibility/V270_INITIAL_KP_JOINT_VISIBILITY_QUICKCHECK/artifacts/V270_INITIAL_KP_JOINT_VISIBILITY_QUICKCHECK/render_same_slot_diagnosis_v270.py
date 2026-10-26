"""Contact sheet from sealed display-only V270 JOINT panels. No inference."""
from pathlib import Path
import hashlib,json
from PIL import Image,ImageDraw,ImageFont
W=Path(__file__).absolute().parent

def safe(p):
 p=Path(p);assert p.is_absolute() and not any(x in str(p).lower()for x in ['v112','cache','.pyc','..']),str(p)
 for a in [p,*p.parents]:assert not a.is_symlink(),str(a)
 return p

def binding(p):
 p=safe(p);b=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),size_bytes=len(b))

font_path=safe(Path('/usr/share/fonts/truetype/nanum/NanumGothic.ttf'))
def font(n):return ImageFont.truetype(str(font_path),n)

cases=[dict(g=2,native=2,human='D',pred='E',p_E=.951,caption='외부 물체 없는 보임 점 → E 오판',detail='같은 native2: 가림 없어도 E',color='#a32e19'),
       dict(g=370,native=2,human='E',pred='E',p_E=.948,caption='콘에 가려진 점 → E 정답',detail='같은 native2: 콘이 있어도 비슷한 점수',color='#11683c'),
       dict(g=800,native=0,human='E',pred='D',p_E=.323,caption='콘에 가려진 점 → D 오판',detail='native0: 콘 위 좌표인데도 D',color='#a32e19')]
canvas=Image.new('RGB',(1440,1050),'#edf2f7');d=ImageDraw.Draw(canvas)
d.text((24,14),'V270 실제 이미지 진단: 가림보다 native 슬롯에 의존',font=font(32),fill='#14304c')
d.text((24,58),'JOINT 저장 출력 · 임계값 p(E) ≥ 0.5 · 새 추론/학습 없음',font=font(22),fill='#40536a')
complete=json.loads(safe(W/'visual_v3/COMPLETE.json').read_bytes());assert complete['PASS']
source={}
for col,c in enumerate(cases):
 x=15+col*475
 d.rounded_rectangle((x,98,x+460,976),radius=12,fill='white',outline='#d0dbe6',width=1)
 d.text((x+12,110),f'G{c["g"]:04d} / native{c["native"]} / GT {c["human"]}',font=font(24),fill='#193a57')
 d.text((x+12,147),c['caption'],font=font(22),fill=c['color'])
 p=safe(W/f'visual_v3/CASE_G{c["g"]:04d}.png');b=binding(p);assert b==complete['outputs'][p.name]
 im=Image.open(p).convert('RGB');assert im.size==(1800,1030)
 # Third panel contains the complete frame, original markers and display-only
 # zoom. No pixels or model coordinates are regenerated or relabeled.
 panel=im.crop((1200,85,1800,1030)).resize((450,709),Image.Resampling.LANCZOS)
 canvas.paste(panel,(x+5,187));d=ImageDraw.Draw(canvas)
 d.text((x+12,909),c['detail'],font=font(21),fill='#2d475f')
 d.text((x+12,943),f'JOINT {c["pred"]} · p(E)={c["p_E"]:.3f}',font=font(21),fill=c['color'])
 source[str(c['g'])]=dict(source=b,extraction_box=[1200,85,1800,1030],output_size=[450,709],native_slot=c['native'],human=c['human'],prediction=c['pred'],p_E_rounded=c['p_E'])
d.text((24,992),'초록 D / 주황 E / 노란 원: 해당 native · 확대는 표시용이며 모델 입력은 전체 이미지',font=font(22),fill='#334e68')
d.text((24,1023),'native0/1은 223장 모두 D, native2/3/6/7은 223장 모두 E. 성공 사례만으로 가림 단서 학습을 주장할 수 없음.',font=font(20),fill='#40536a')
out=safe(W/'SAME_SLOT_DIAGNOSIS.png');assert not out.exists();canvas.save(out)
receipt=safe(W/'SAME_SLOT_DIAGNOSIS_SOURCE.json');assert not receipt.exists();receipt.write_text(json.dumps(dict(PASS=True,completed=True,code=binding(Path(__file__).absolute()),output=binding(out),source_visual_complete=binding(W/'visual_v3/COMPLETE.json'),source_cases=source,source_slot_diagnostic=binding(W/'evaluation_v4/SLOT_PRIOR_DIAGNOSTIC.json'),font=binding(font_path),new_model_forwards=0,new_fits=0,new_predictions=0,changes='Only extract and resize the already-rendered JOINT panel into a 3-card explanation; preserve full-frame and display-only crop markers.',excluded_disputed_cases=[300,996]),ensure_ascii=False,indent=2,sort_keys=True)+'\n')
print(binding(out))
