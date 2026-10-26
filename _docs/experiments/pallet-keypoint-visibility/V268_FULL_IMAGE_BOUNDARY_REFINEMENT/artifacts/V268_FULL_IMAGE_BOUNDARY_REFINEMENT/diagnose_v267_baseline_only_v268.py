"""Saved V267-only failures; manual-coordinate oracle is evaluation-only.

No new model inference, fit, threshold, q replacement in output, or PnP.
"""
from pathlib import Path
import csv
import hashlib
import io
import json
import sys
import types
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.nn import functional as F

HERE=Path(__file__).absolute().parent
SELECTED=(200,852,665,1354,300)


def safe(p):
    p=Path(p)
    assert p.is_absolute()and not any(x in str(p).lower()for x in('v112','cache','.pyc','..'))
    for a in reversed((p,*p.parents)):assert not a.is_symlink()
    return p


def read(b):
    raw=safe(Path(b['path'])).read_bytes();assert len(raw)==b['size_bytes']and hashlib.sha256(raw).hexdigest()==b['sha256'];return raw


def dump(p,d):
    p=safe(p)
    if p.exists():
        assert json.loads(p.read_bytes())==json.loads(json.dumps(d))
        return
    with p.open('x')as f:json.dump(d,f,ensure_ascii=False,sort_keys=True,indent=2);f.write('\n')


def sample(z,xy,hw):
    h,w=hw;grid=torch.tensor((np.asarray(xy,np.float64)+.5)/np.array([w,h])*2-1,dtype=torch.float32).reshape(1,1,1,2)
    return float(F.grid_sample(torch.from_numpy(z).reshape(1,1,32,32),grid,mode='bilinear',padding_mode='border',align_corners=False)[0,0,0,0])


def font(size):return ImageFont.truetype('/usr/share/fonts/truetype/nanum/NanumGothic.ttf',size)


def panel(im,w,h):
    base=Image.new('RGB',(w,h),'#1e2939');im=im.copy();im.thumbnail((w,h),Image.Resampling.BICUBIC);base.paste(im,((w-im.width)//2,(h-im.height)//2));return base


def marks(im,q,ref):
    im=im.copy();d=ImageDraw.Draw(im);x,y=q;r=max(3,round(im.width/160));d.ellipse((x-r,y-r,x+r,y+r),outline='#ffe529',width=2)
    if ref is not None:
        a,b=ref;d.line((a-r,b,a+r,b),fill='#ff3fce',width=2);d.line((a,b-r,a,b+r),fill='#ff3fce',width=2)
    return im


def main():
    torch.set_num_threads(1)
    out=safe(HERE/'v267_baseline_failures_v1');out.mkdir(exist_ok=True)
    dump(out/'SELECTION_BEFORE_RGB.json',dict(completed=True,global_indices=SELECTED,
        reason='FixedpriorvisiblefailuresG200/G852/G665 + firstsmall-errorV267EfalseD(G1354) + firstV267EfalseDlarge-errorG300; no representativefrequencyclaim.',
        models='V267FULL_CLIP_SURGERY_ARCHonly; notnewV268refiners',new_model_or_fit_or_threshold_changes=0))
    receipt=json.loads(safe(HERE/'CURRENT_PREDICTIONS_FROZEN.json').read_bytes());cfg=json.loads(read(receipt['configuration']));policy=json.loads(read(cfg['protocol']))
    inp=json.loads(read(policy['current_input_complete']));frames=[json.loads(x)for x in read(inp['frames']).splitlines()];queries=[json.loads(x)for x in read(inp['queries_manifest']).splitlines()]
    with np.load(io.BytesIO(read(policy['V267_current_frame_masks'])),allow_pickle=False)as z:maps=np.array(z['FULL_CLIP_SURGERY_ARCH_logit'],copy=True)
    with safe(HERE/'evaluation_v1/compact_inputs/LEDGER.csv').open()as f:ledger=list(csv.DictReader(f))
    reference=[json.loads(x)for x in safe(HERE/'evaluation_v1/compact_inputs/V267_REFERENCE_PREDICTIONS.jsonl').read_bytes().splitlines()]
    init=[];oracle=[];valid=[];manual=[];near=[]
    for i,(r,q)in enumerate(zip(ledger,queries)):
        a=sample(maps[q['frame_index']],q['original_q0'],q['raw_hw'])if q['available']else None
        if a is not None:assert abs(a+reference[i]['FULL_CLIP_SURGERY_ARCH']['E_logit'])<1e-5
        ismanual=r['human_state']=='DIRECT_VISIBLE'and r['annotation_coordinate_source']=='manual_click'and q['available']
        try:refxy=np.array([float(r['human_reference_x']),float(r['human_reference_y'])]);finite=bool(np.isfinite(refxy).all())
        except ValueError:refxy=None;finite=False
        h,w=q['raw_hw'];inside=finite and 0<=refxy[0]<w and 0<=refxy[1]<h
        b=sample(maps[q['frame_index']],refxy,q['raw_hw'])if ismanual and inside else None
        init.append(a);oracle.append(b);manual.append(bool(ismanual and inside))
        try:near.append(bool(ismanual and inside and float(r['initial_reference_error_px'])<=3))
        except ValueError:near.append(False)
    groups={}
    for group in('ALL','WOOD_PRIMARY105','SQUARE118'):
        groups[group]={}
        for cohort,mask in [('MANUAL_VISIBLE_VALID',manual),('MANUAL_VISIBLE_INITIAL_LE3',near)]:
            ix=[i for i,v in enumerate(mask)if v and(group=='ALL'or ledger[i]['group']==group)]
            groups[group][cohort]=dict(N=len(ix),original_q_D=sum(init[i]>=0 for i in ix),manual_reference_oracle_D=sum(oracle[i]>=0 for i in ix),
                E_to_D=[i for i in ix if init[i]<0 and oracle[i]>=0],D_to_E=[i for i in ix if init[i]>=0 and oracle[i]<0],
                still_E_at_manual_reference=[i for i in ix if oracle[i]<0])
    dump(out/'MANUAL_REFERENCE_ORACLE_DIAGNOSTIC.json',dict(completed=True,PASS=True,groups=groups,
        scope='HumanDmanualcoordinatesonly, evaluation-onlyoracleprobeofSAMEsavedV267map;notdeployablecorrection/policy/metricreplacement.',
        actual_predictions_or_queries_changed=0,new_model_forwards_or_fits_or_threshold_or_PnP=0,real_pixel_GT_available=False))
    ioapi=types.ModuleType('v268_baseline_diag_rgb');ioapi.__file__=policy['current_input_helper']['path'];sys.modules[ioapi.__name__]=ioapi
    exec(compile(read(policy['current_input_helper']),ioapi.__file__,'exec'),ioapi.__dict__)
    titles={200:'보이는 모서리를 가림으로 읽음',852:'정사각형의 보이는 외곽을 놓침',665:'작고 어두운 팔레트 영역을 놓침',1354:'가림 주석인데 팔레트 확률이 높음',300:'다른 위치의 팔레트와 숨은 native는 다름'}
    cards=[]
    for i in SELECTED:
        row=ledger[i];q=queries[i];rgb=ioapi.read_rgb(frames[q['frame_index']]);h,w=rgb.shape[:2];xy=q['original_q0']
        try:ref=[float(row['human_reference_x']),float(row['human_reference_y'])];ref=ref if np.isfinite(ref).all()else None
        except ValueError:ref=None
        fullz=F.interpolate(torch.from_numpy(maps[q['frame_index']]).reshape(1,1,32,32),(h,w),mode='bilinear',align_corners=False)[0,0].numpy()
        p=1/(1+np.exp(-np.clip(fullz,-80,80)));color=np.empty((h,w,3),np.uint8);color[:,:,0]=(14+10*p).astype(np.uint8);color[:,:,1]=(25+230*p).astype(np.uint8);color[:,:,2]=(19+10*p).astype(np.uint8)
        hard=fullz>=0;overlay=rgb.astype(np.float32);overlay[hard]=.55*overlay[hard]+.45*np.array([0,255,45]);overlay=np.clip(overlay,0,255).astype(np.uint8)
        a,b=xy;box=(round(a)-48,round(b)-48,round(a)+48,round(b)+48)
        crop=Image.fromarray(rgb).crop(box).resize((330,330),Image.Resampling.NEAREST);dd=ImageDraw.Draw(crop)
        sx,sy=(a-box[0])*330/96,(b-box[1])*330/96;dd.ellipse((sx-4,sy-4,sx+4,sy+4),outline='#ffe529',width=2)
        if ref is not None:
            rx,ry=(ref[0]-box[0])*330/96,(ref[1]-box[1])*330/96
            dd.line((rx-6,ry,rx+6,ry),fill='#ff3fce',width=2);dd.line((rx,ry-6,rx,ry+6),fill='#ff3fce',width=2)
        canvas=Image.new('RGB',(1560,620),'#f6f8fb');d=ImageDraw.Draw(canvas)
        d.text((24,16),f'V267 Full-image CLIP Surgery · G{i} · '+titles[i],font=font(27),fill='#17253c')
        score=reference[i]['FULL_CLIP_SURGERY_ARCH'];label='보임 D'if score['pred']=='D'else'비대상 E'
        d.text((24,62),f"주석 {row['human_state']} · 모델 {label} · P(target@초기q)={score['p_target']:.4f} · 참조대비 {row['initial_reference_error_px'][:6]}px",font=font(22),fill='#475b78')
        labels=['전체 RGB · 입력 범위','V267 P(target) · 고정0–1','V267 고정0.5 마스크','초기 q 중심 확대 · 표시만']
        imgs=[marks(Image.fromarray(rgb),xy,ref),marks(Image.fromarray(color),xy,ref),marks(Image.fromarray(overlay),xy,ref),crop]
        for j,(name,im)in enumerate(zip(labels,imgs)):
            x=24+j*384;d.text((x,112),name,font=font(19),fill='#475b78');canvas.paste(panel(im,360,330),(x,145))
        d.text((24,494),'노랑 원=초기 q · 분홍 +=기존 주석 참조(manual/PnP; 숨은 점의 위치 GT는 아님)',font=font(22),fill='#17253c')
        text='수동 참조 위치에서도 P(target)='+f'{1/(1+np.exp(-oracle[i])):.4f}'if oracle[i]is not None else'가림 참조좌표 출처: '+row['annotation_coordinate_source']+' (기존 정의 유지)'
        d.text((24,535),text+' · 진단 전용, 실제 판정/좌표는 변경하지 않음',font=font(20),fill='#ad420c')
        d.text((24,575),'초록은 ARCH Surgery 특징+합성 M4 head 출력 · 원 논문의 full zero-shot text/feature-surgery 출력은 아님',font=font(19),fill='#475b78')
        name=f'CASE_G{i:04d}.png';canvas.save(safe(out/name));cards.append(dict(global_index=i,title=titles[i],file=name,original_query=xy,reference=ref,
            coordinate_source=row['annotation_coordinate_source'],human_state=row['human_state'],prediction=score,reference_distance_px=float(row['initial_reference_error_px']),
            manual_reference_oracle_p_target=None if oracle[i]is None else float(1/(1+np.exp(-oracle[i])))))
    # A concise baseline-only pair for inline viewing. All5largercards remain.
    overview=Image.new('RGB',(1560,1240),'white')
    for k,i in enumerate((200,1354)):
        with Image.open(safe(out/f'CASE_G{i:04d}.png'))as im:overview.paste(im,(0,k*620))
    overview.save(safe(out/'OVERVIEW_KO.png'))
    dump(out/'CASE_SUMMARY.json',dict(completed=True,cases=cards,source_model='V267FULL_CLIP_SURGERY_ARCHonly',new_model_forwards_or_fits_or_thresholds=0))
    nearall=groups['ALL']['MANUAL_VISIBLE_INITIAL_LE3']
    html='''<!doctype html><meta charset="utf-8"><title>V267이 좋아 보이는데 어디가 문제인가</title><style>body{font-family:sans-serif;max-width:1560px;margin:30px auto;padding:0 24px;line-height:1.7;background:#f5f7fb;color:#17253c}img{max-width:100%}.note{background:#fff0ce;padding:18px}a{color:#165db8}</style><h1>Full-image CLIP Surgery는 영역 앞단으로 사용할 수 있습니다</h1>
<p class="note">보류한 것은 이 지도 한 지점의 값으로 native 보임/외부 가림을 확정하는 방식입니다. 전체 팔레트 영역을 잘 찾은 사례도 모서리 판정은 틀릴 수 있습니다. 아래는 새 V268 RGB 보정 모델이 아닌, 사용자가 좋아 보인다고 한 V267 결과만 보여줍니다.</p>
<p>원래 PRIMARY에서 D1147중595보임/552비대상, E90중84비대상/6보임입니다. 입력 crop 없이 전체 영상부터 처리했고 초기 q는 지도 저장 뒤 읽습니다. 자기 가림은 별도입니다. 실사 픽셀 GT는 없으므로 픽셀 분할이 맞다/틀리다를 모두 확정하지 않습니다.</p>
<p><a href="../ROOT_READOUT_KO.html">새 보정 비교 전체</a> · <a href="../PRIMARY_SOURCE_DCLIP_REVIEW_KO.md">DeCLIP/DCLIP 공식 자료 확인</a> · <a href="MANUAL_REFERENCE_ORACLE_DIAGNOSTIC.json">수동 좌표 진단</a></p>'''
    html+=f"<p>초기오차≤3px 수동D {nearall['N']}점: 원래q에서 D{nearall['original_q_D']}, 수동 참조좌표를 넣은 평가용 oracle에서는 D{nearall['manual_reference_oracle_D']}; 그 좌표에서도 E{len(nearall['still_E_at_manual_reference'])}입니다. 이는 GT를 쓰는 진단이며 새 성능/보정 방식이 아닙니다.</p>"
    for c in cards:html+=f'<h2>G{c["global_index"]} · {c["title"]}</h2><img src="{c["file"]}">'
    html+='''<p>영역 표현의 실패(작고 어두운 대상/가림 경계 번짐), 초기 좌표가 다른 표면에 놓이는 문제, 픽셀 존재와 특정 코너 보임 목표의 차이가 함께 있습니다. CLIP 영역 앞단을 유지하면서 뒤의 native 판단을 개선할 이유이며, 전체 모델을 버려야 한다는 결론이 아닙니다.</p>'''
    safe(out/'INDEX_KO.html').write_text(html)
    print(json.dumps(dict(completed=True,selected=SELECTED,manual_near3_oracle={k:len(v)if isinstance(v,list)else v for k,v in nearall.items()})))


if __name__=='__main__':main()
