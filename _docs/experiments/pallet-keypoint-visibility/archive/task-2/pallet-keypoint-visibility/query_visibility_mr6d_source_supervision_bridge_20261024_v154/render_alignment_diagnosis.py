"""Visualize saved public query/GT-role differences; no learned model calls."""
from pathlib import Path
import hashlib
import json
import textwrap
import numpy as np
from PIL import Image, ImageDraw, ImageFont

V=Path(__file__).resolve().parent
S=V.parent/'query_visibility_mr6d_full_public_source_20261024_v153'
regular='/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
bold='/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf'
font=lambda n,b=False:ImageFont.truetype(bold if b else regular,n)
im=Image.new('RGB',(1320,1180),'#f4f7fa');d=ImageDraw.Draw(im)
d.text((24,18),'대상 개체 오류와 native 역할 불일치는 가림과 분리해야 합니다',font=font(28,True),fill='#173348')
d.text((24,60),'공개 실사의 실제 저장 입력 · 원래 q 유지 · 가림 모델의 예측 결과가 아닙니다',font=font(18),fill='#536575')
bindings=[]
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
for row,(i,key,title) in enumerate([(73,0,'A. crop은 정상인데 초기 모델이 팔레트 옆 금속 받침대를 선택'),(117,44,'B. 같은 팔레트를 선택했지만 앞·뒤 역할의 대응이 달라짐')]):
    y=101+row*452
    d.text((24,y),title,font=font(23,True),fill='#173348')
    qp=S/f'original_query_evidence_full230/frames/FRAME_{i:03d}_val_000001_{key}.png'
    gp=S/f'native_reference_preparation/source_reference_contact_v1/frames/FRAME_{i:03d}_000001_{key:06d}_ALL_INSTANCES_NATIVE.png'
    q=Image.open(qp).convert('RGB').crop((8,52,648,412))
    g=Image.open(gp).convert('RGB').resize((640,360),Image.Resampling.LANCZOS)
    d.text((24,y+36),'원래 초기 query: 분홍색 native0…7',font=font(17),fill='#9b1c70')
    d.text((674,y+36),'제공 pose/CAD의 명목 역할: 청록색 native0…7',font=font(17),fill='#087c97')
    im.paste(q,(24,y+66));im.paste(g,(674,y+66));bindings.extend([bind(qp),bind(gp)])

facts=[json.loads(x) for x in (S/'full230_native_DINO_v1/FEATURE_INPUT_FACTS1840_INPUT_ONLY.jsonl').read_text().splitlines()]
refs=[json.loads(x) for x in (S/'native_reference_preparation/provided_reference_v1/ALL2424_DYNAMIC_NATIVE_REFERENCE_GT_ONLY.jsonl').read_text().splitlines()]
q=facts[117*8+5]
r=next(x for x in refs if x['frame_index']==117 and x['gt_instance_index']==0 and x['native_slot']==5)
distance=float(np.linalg.norm(np.asarray(q['query_xy'])-np.asarray(r['reference_xy'])))
cropfile=S/'full230_native_DINO_v1/FRAME_0117_FACTUAL96_INPUT_ONLY.npz'
with np.load(cropfile,allow_pickle=False) as z:
    crop=Image.fromarray(z['rgb96'][5]).resize((160,160),Image.Resampling.NEAREST)
im.paste(crop,(24,1000));bindings.append(bind(cropfile))
d.text((205,1005),'B의 원래 native5 crop: 팔레트 끝이 보임',font=font(20,True),fill='#173348')
d.text((205,1040),f'명목 GT native5는 바구니 뒤쪽 역할. 같은 채널의 명목 거리: {distance:.1f}px',font=font(18),fill='#536575')
d.text((205,1072),'따라서 GT 역할의 가림 라벨을 이 crop에 그대로 복사하지 않습니다.',font=font(18),fill='#536575')
d.text((205,1104),'이 거리는 실측 좌표 정답이 아닙니다. 좌표 보정·다른 GT점 선택은 하지 않습니다.',font=font(16),fill='#536575')
out=V/'ROLE_AND_ENTITY_ALIGNMENT_FAILURES_KO.png'
im.save(out)
qa=dict(completed=True,figure=bind(out),inputs=bindings,source_only=True,root_case73_and117_actually_viewed=True,
    example117_same_native5_nominal_reference_distance_px=distance,exact_real_pixel_gold=False,
    crop_pixels_saved_RGB96_nearest_only=True,original_query_or_labels_changed=False,
    new_GT_assignment_permutation_or_coordinate_correction=False,new_model_calls=0,
    figure_visual_QA_pending=True)
with (V/'ALIGNMENT_FIGURE_RENDER_RECEIPT.json').open('x') as f:json.dump(qa,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(bind(out)))
