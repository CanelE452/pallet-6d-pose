from pathlib import Path
import hashlib,json,math
import numpy as np
from PIL import Image,ImageDraw,ImageFont

B=Path(__file__).resolve().parent
O=B/'BUILD_V2'
rows=[json.loads(x) for x in (O/'ROWS.jsonl').read_text().splitlines()]
props=json.loads((B.parent.parent.parent/'query_visibility_reference_semantic_20261023_v69/source_cue_reference_proposals/AI_OPERATIONAL_CUE_REFERENCE_PROPOSAL_LEDGER.json').read_text())['proposals']
font='/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
F=ImageFont.truetype(font,22);small=ImageFont.truetype(font,18);title=ImageFont.truetype(font,30)

def bind(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def marked_crop(row):
    p=props[row['cue_rank']];im=Image.open(row['crop']['path']).convert('RGB').resize((288,288),Image.Resampling.NEAREST)
    d=ImageDraw.Draw(im);q=np.array(row['model_inputs']['initial_query_xy']);L=row['model_inputs']['input_crop_L']
    # Only diagnostic overlays; the saved model input PNG remains unmarked.
    if p['associated_cue_xy'] is not None:
        r=(np.array(p['associated_cue_xy'])-q)*(288/L)+143.5
        b=np.array(p['uncertainty_bbox_half_open_xyxy']);bb=(b-np.tile(q,2))*(288/L)+143.5
        if -10<r[0]<298 and -10<r[1]<298:
            d.rectangle(tuple(bb),outline='#00ccff',width=2);d.line((r[0]-6,r[1],r[0]+6,r[1]),fill='#00ccff',width=2);d.line((r[0],r[1]-6,r[0],r[1]+6),fill='#00ccff',width=2)
    d.line((136,144,152,144),fill='#ff00b4',width=2);d.line((144,136,144,152),fill='#ff00b4',width=2)
    return im

def context(row):
    p=props[row['cue_rank']];im=Image.open(p['original_image']).convert('RGB');d=ImageDraw.Draw(im)
    q=row['model_inputs']['initial_query_xy'];L=row['model_inputs']['input_crop_L'];r=p['associated_cue_xy']
    d.rectangle((q[0]-L/2,q[1]-L/2,q[0]+L/2,q[1]+L/2),outline='#ffd000',width=2)
    for xy,color,label in [(q,'#ff00b4','q'),(r,'#00ccff','cue')]:
        d.line((xy[0]-6,xy[1],xy[0]+6,xy[1]),fill=color,width=2);d.line((xy[0],xy[1]-6,xy[0],xy[1]+6),fill=color,width=2)
        d.text((xy[0]+7,xy[1]+2),label,font=small,fill=color,stroke_width=1,stroke_fill='#000000')
    box=p['original_target_box_xyxy'];lo=[min(box[0],q[0]-L/2,r[0])-20,min(box[1],q[1]-L/2,r[1])-20];hi=[max(box[2],q[0]+L/2,r[0])+20,max(box[3],q[1]+L/2,r[1])+20]
    im=im.crop((max(0,math.floor(lo[0])),max(0,math.floor(lo[1])),min(im.width,math.ceil(hi[0])),min(im.height,math.ceil(hi[1]))))
    im.thumbnail((360,288));return im

case1=next(r for r in rows if r['variant']=='ORIGINAL' and r['training_target_GT_only']=='D_teacher' and r['native_query_support_GT_only'].get('exact') is False)
case2=next(r for r in rows if r['variant']=='FULL_COVER' and r['training_target_GT_only']=='E_teacher' and r['native_query_support_GT_only'].get('exact') is True)
case3=next(r for r in rows if r['cue_rank']==4 and r['variant']=='ORIGINAL' and r['jitter_role_GT_only']=='PRIMARY' and r['delta_L_GT_only']==[0,0])
cases=[('cue는 보임 · query 픽셀은 배경',case1),('cue는 외부 가림 · query에는 대상이 보임',case2),('cue 자체는 보임 · 초기 crop이 놓침',case3)]
canvas=Image.new('RGB',(2020,1410),'#eef2f7');d=ImageDraw.Draw(canvas)
d.text((24,18),'실제 합성 source: reference 상태와 query/crop 상태를 분리',font=title,fill='#172536')
d.text((24,62),'고정 cue/불확실성 영역 · 동일 q/L/box의 원본과 개입 쌍 · 새 모델 예측 없음 · AI 조건부 teacher, human gold 아님',font=F,fill='#45586c')
d.text((24,97),'자홍 q = 입력 중심 / 청록 cue = GT-only 진단 표시 / 노랑 = 입력 footprint. 입력용 PNG에는 GT 표시가 없습니다.',font=small,fill='#45586c')
kept=[]
for case_i,(label,base) in enumerate(cases):
    y=145+case_i*400;rank=base['cue_rank'];j=base['row_id'].split('_J')[-1]
    d.text((24,y),f'{case_i+1}. {label} | rank{rank}, q=({base["model_inputs"]["initial_query_xy"][0]:.2f},{base["model_inputs"]["initial_query_xy"][1]:.2f}) L={base["model_inputs"]["input_crop_L"]:.2f}',font=F,fill='#172536')
    context_im=context(base);canvas.paste(context_im,(24,y+60));d.text((24,y+32),'원본 문맥: q/cue/crop',font=small,fill='#45586c')
    selected=[]
    for col,variant in enumerate(['ORIGINAL','FULL_COVER','PARTIAL_COVER','QUERY_ONLY_CONTROL']):
        row=next(r for r in rows if r['cue_rank']==rank and r['variant']==variant and r['row_id'].endswith('_J'+j))
        x=425+col*390
        d.text((x,y+32),variant,font=small,fill='#172536');canvas.paste(marked_crop(row),(x,y+60))
        st=row['reference_state_GT_only'].replace('_teacher','');av=row['availability_GT_only'];tt=row['training_target_GT_only'].replace('_teacher','');px=row['native_query_support_GT_only'].get('exact')
        d.text((x,y+352),f'ref={st} / crop={av}',font=small,fill='#172536')
        d.text((x,y+375),f'train={tt} / exact M4(q)={px}',font=small,fill='#172536')
        selected.append(row['row_id'])
    kept.append({'criterion':label,'row_ids':selected})
d.text((24,1360),'MISS/PARTIAL crop과 부분 cue 가림은 U로 보존합니다. crop 중심·크기를 정답으로 바꾸지 않았고, 이 그림은 모델 성능 향상의 증거가 아닙니다.',font=small,fill='#45586c')
p=O/'ACTUAL_FIXED_REFERENCE_COUNTEREXAMPLES.png';assert not p.exists();canvas.save(p)
f=O/'COUNTEREXAMPLE_FIGURE.json';assert not f.exists();f.write_text(json.dumps({'figure':bind(p),'code':bind(Path(__file__)),'cases':kept,'selection':'First row in frozen row order satisfying named source GT counterexample, plus rank4 original q0 MISS. Diagnostic selection only; no model scores/current.','input_crop_images_modified':False,'new_model_or_PnP':0},ensure_ascii=False,indent=2)+'\n')
print(json.dumps(bind(p)))
