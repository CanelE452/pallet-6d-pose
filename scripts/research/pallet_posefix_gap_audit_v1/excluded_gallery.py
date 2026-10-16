"""Show every excluded RGB from the fixed common-image outlier analysis."""
from pathlib import Path
import argparse
import csv
import gzip
import hashlib
import html
import json
import math
import time

from PIL import Image,ImageDraw,ImageFont,ImageOps

ARMS=('POSEFIX_RAW','R0','N0','N1','E0')
SHORT={'POSEFIX_RAW':'PF','R0':'RAW','N0':'N3+Sub','N1':'BI','E0':'e'}
NAMES={'POSEFIX_RAW':'PoseFix','R0':'RAW YOLO','N0':'N3→SubPix','N1':'최소 B/I','E0':'기존 전체 방법 e'}
INVENTORY='_docs/experiments/pallet_visibility_curriculum_20261013_v1/publication_followup_20261015_v11/repro_inventory_current_files_v2.jsonl.gz'
INVENTORY_SHA='75c5aaa33d534faefc028c3358d9be6b5cbe688599488a29d34478502fac47b5'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,value):
    with Path(p).open('x') as f:f.write(value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def run(source_root,inputs,summary,outliers,output):
    start=time.monotonic();source_root,inputs,summary,outliers,output=map(Path,(source_root,inputs,summary,outliers,output))
    if output.exists():raise FileExistsError('Preserve previous gallery')
    analysis=read(outliers);cohort=analysis['cohorts']['ALL5_IQR_T_R'];ids=cohort['excluded_ids']
    if len(ids)!=len(set(ids))!=71:raise ValueError('Exactly the same71 excluded images required')
    if not len(ids)==len(set(ids))==71:raise ValueError('Exactly the same71 excluded images required')
    frames={r['id']:r for r in read(inputs)['frames']}
    with summary.open(newline='') as f:scored={r['id']:r for r in csv.DictReader(f)}
    inventory=source_root/INVENTORY
    if sha(inventory)!=INVENTORY_SHA:raise ValueError('Already-public RGB inventory changed')
    with gzip.open(inventory,'rt') as f:
        permitted={r['sha256'] for r in map(json.loads,f) if Path(r['relpath']).suffix.lower() in ('.jpg','.jpeg','.png')}
    thresholds={m:analysis['thresholds'][m]['upper'] for m in ('translation_cm','rotation_deg')}
    reconstructed=[]
    for fid,row in scored.items():
        if row['partition']=='CAL':continue
        if any(float(row[a+'__'+m])>limit for a in ARMS for m,limit in thresholds.items()):reconstructed.append(fid)
    if reconstructed!=ids:raise ValueError('Excluded IDs/order must match saved rule and CSV')
    fontfile='/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
    font=ImageFont.truetype(fontfile,16);small=ImageFont.truetype(fontfile,14);heading=ImageFont.truetype(fontfile,25)
    output.mkdir(parents=True);(output/'images').mkdir()
    cards=[];records=[];contact_crops=[]
    for number,fid in enumerate(ids,1):
        frame=frames[fid];row=scored[fid];path=source_root/frame['image']
        if frame['partition']=='CAL':raise ValueError('CAL must not be included')
        if sha(path)!=frame['image_sha256'] or frame['image_sha256'] not in permitted:raise ValueError('Original already-public RGB bytes required')
        image=Image.open(path).convert('RGB');w,h=image.size
        if [h,w]!=frame['raw_hw']:raise ValueError('Original canvas changed')
        x0,y0,x1,y1=frame['candidate_metadata']['box_xyxy'];marginx=max((x1-x0)*.18,16);marginy=max((y1-y0)*.35,16)
        roi=(max(0,math.floor(x0-marginx)),max(0,math.floor(y0-marginy)),min(w,math.ceil(x1+marginx)),min(h,math.ceil(y1+marginy)))
        if roi[0]>=roi[2] or roi[1]>=roi[3]:roi=(0,0,w,h)
        crop=image.crop(roi);contact_crops.append(crop)
        full=ImageOps.contain(image,(960,720));thumbnail=ImageOps.contain(crop,(640,400))
        full_name=f'images/{number:03d}_full.jpg';crop_name=f'images/{number:03d}_crop.jpg'
        full.save(output/full_name,quality=86,optimize=True);thumbnail.save(output/crop_name,quality=88,optimize=True)
        reasons=[];metrics={};table=[]
        for arm in ARMS:
            t=float(row[arm+'__translation_cm']);r=float(row[arm+'__rotation_deg']);tx=t>thresholds['translation_cm'];rx=r>thresholds['rotation_deg']
            metrics[arm]=dict(translation_cm=t,rotation_deg=r,translation_exceeds=tx,rotation_exceeds=rx)
            if tx or rx:reasons.append(dict(arm=arm,translation_exceeds=tx,rotation_exceeds=rx))
            table.append(f'<tr><td>{NAMES[arm]}</td><td class="{"bad" if tx else ""}">{t:.2f}cm</td><td class="{"bad" if rx else ""}">{r:.2f}°</td></tr>')
        if not reasons:raise ValueError('Every excluded frame needs an exact crossing reason')
        summary_text=' · '.join(SHORT[r['arm']]+':'+('T' if r['translation_exceeds'] else '')+('R' if r['rotation_exceeds'] else '') for r in reasons)
        records.append(dict(number=number,id=fid,session=frame['session'],partition=frame['partition'],source_image_sha256=frame['image_sha256'],
            source_already_public=True,raw_hw=[h,w],display_full_hw=[full.height,full.width],crop_xyxy=list(roi),metrics=metrics,reasons=reasons,
            full_display=full_name,crop_display=crop_name,no_pose_or_GT_overlay=True))
        arms=' '.join(r['arm'] for r in reasons)
        cards.append(f'<article data-session="{html.escape(frame["session"])}" data-arms="{arms}" data-id="{html.escape(fid)}" id="frame-{number}"><h2>{number:02d}. {html.escape(fid)}</h2><p class="reason">{html.escape(summary_text)}</p><a href="{full_name}" target="_blank"><img src="{crop_name}" alt="{html.escape(fid)} 예측 bbox 주변 실제 RGB" loading="lazy"></a><table><thead><tr><th>방법</th><th>위치 오차</th><th>회전 오차</th></tr></thead><tbody>'+''.join(table)+f'</tbody></table><details><summary>전체 프레임 보기</summary><img src="{full_name}" alt="실제 전체 프레임" loading="lazy"></details></article>')
    sheets=[]
    for page in range(math.ceil(len(ids)/24)):
        canvas=Image.new('RGB',(1800,1930),'#edf1f5');draw=ImageDraw.Draw(canvas)
        start_index=page*24;stop=min(start_index+24,len(ids))
        draw.text((22,15),f'공통 제외 71장 | {page+1}/3 페이지 | {start_index+1}–{stop}',font=heading,fill='#172b3b')
        draw.text((22,53),'PF=PoseFix  RAW=YOLO  N3+Sub=기본  BI=최소 B/I  e=기존 방법 | T:위치초과 R:회전초과',font=font,fill='#243649')
        draw.text((22,80),'어느 방법이든 위치 >18.728cm 또는 회전 >7.222° | 실제 RGB의 예측 bbox 주변 확대',font=font,fill='#243649')
        for j,index in enumerate(range(start_index,stop)):
            x=(j%4)*450+10;y=(j//4)*300+125
            draw.rounded_rectangle((x,y,x+430,y+285),radius=8,fill='white')
            crop=ImageOps.contain(contact_crops[index],(414,218));canvas.paste(crop,(x+8+(414-crop.width)//2,y+7+(218-crop.height)//2))
            record=records[index];draw.text((x+8,y+230),f'{index+1:02d}. '+record['id'],font=font,fill='#112234')
            reason=' '.join(SHORT[r['arm']]+':'+('T' if r['translation_exceeds'] else '')+('R' if r['rotation_exceeds'] else '') for r in record['reasons'])
            draw.text((x+8,y+254),reason,font=small,fill='#b63b24')
        name=f'contact_{page+1:02d}.jpg';canvas.save(output/name,quality=88,optimize=True);sheets.append(name)
    sessions=sorted({r['session'] for r in records})
    options=''.join(f'<option value="{html.escape(s)}">{html.escape(s)}</option>' for s in sessions)
    arm_options=''.join(f'<option value="{a}">{NAMES[a]}에서 상한 초과</option>' for a in ARMS)
    style='body{font-family:sans-serif;background:#edf1f5;max-width:1700px;margin:20px auto;padding:16px;line-height:1.6;color:#172b3b}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}article{background:white;border-radius:10px;padding:14px}h2{font-size:17px;overflow-wrap:anywhere}img{width:100%;height:auto}table{width:100%;border-collapse:collapse}td,th{border-bottom:1px solid #ddd;padding:4px;text-align:left}.bad,.reason{color:#b52d21}.bad{font-weight:bold;background:#fff0e8}select,input{padding:9px;margin:6px}nav{position:sticky;top:0;background:#edf1f5;padding:8px;z-index:3}@media(max-width:1000px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:650px){.grid{grid-template-columns:1fr}}'
    introduction='전체192장 중 ALL5 공통 IQR 기준으로 제외된71장 전부입니다. 어느 경로에서든 위치18.728cm 또는 회전7.222°를 넘으면 공통 제외했습니다. 빨간 값이 실제 제외 원인입니다. 사진은 원 RGB의 표시용 JPEG이며 GT/예측 pose를 그리지 않았습니다. 작은 사진은 예측 bbox 주변 확대이고, 클릭하거나 펼치면 전체 프레임을 볼 수 있습니다. 원래 전체192장 평가·실패는 그대로 보존했습니다.'
    script='function filterCards(){let s=document.getElementById("session").value,a=document.getElementById("arm").value,q=document.getElementById("query").value.toLowerCase(),n=0;document.querySelectorAll("article").forEach(c=>{let show=(!s||c.dataset.session===s)&&(!a||c.dataset.arms.split(" ").includes(a))&&c.dataset.id.toLowerCase().includes(q);c.hidden=!show;if(show)n++});document.getElementById("count").textContent=n+" / 71장";}document.querySelectorAll("select,input").forEach(e=>e.addEventListener("input",filterCards));'
    document='<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>공통 제외71장 실제 이미지</title><style>'+style+'</style><h1>공통 제외71장 실제 이미지</h1><p>'+introduction+'</p><p>개요: '+''.join(f'<a href="{s}">모아보기{i+1}</a> ' for i,s in enumerate(sheets))+'</p><nav><select id="session"><option value="">모든 세션</option>'+options+'</select><select id="arm"><option value="">모든 제외 원인</option>'+arm_options+'</select><input id="query" placeholder="ID 검색"><strong id="count">71 / 71장</strong></nav><div class="grid">'+''.join(cards)+'</div><script>'+script+'</script></html>'
    write(output/'index.html',document)
    md=['# 공통 제외71장 실제 이미지',introduction,'[전체71장 HTML·제외 원인·방법별 오차](index.html)']
    for i,name in enumerate(sheets,1):md.extend([f'## {i}/3 페이지',f'![공통 제외 이미지 {i}](<'+name+'>)'])
    write(output/'GALLERY_KO.md','\n\n'.join(md)+'\n')
    write(output/'CARDS.json',dict(schema='EXACT_COMMON_EXCLUDED71_RGB_GALLERY_V1',cohort='ALL5_IQR_T_R',excluded_n=71,ordered_ids=ids,thresholds=thresholds,cards=records,
        contacts=sheets,all_actual_RGB_SHA_verified=True,already_public_RGB_only=True,new_inference_or_PnP_or_GT_read=0))
    write(output/'RECEIPT.json',dict(source_sha256=sha(__file__),inputs_sha256={'PRIVATE/inputs':sha(inputs),'summary/ALL_ROWS.csv':sha(summary),'outliers/OUTLIER_RESULTS.json':sha(outliers),INVENTORY:INVENTORY_SHA},
        images=71,crops=71,contact_sheets=3,full_original_RGB_canvases_available=True,display_JPEGs_are_reencoded_not_original_bytes=True,
        exact_excluded_IDs_and_order=True,exclusion_reasons=sum(len(r['reasons']) for r in records),wall_seconds=time.monotonic()-start))
    write(output/'SEAL.json',dict(assets={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()}))
    return output/'index.html'


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('source-root','inputs','summary','outliers','output'):p.add_argument('--'+arg,type=Path,required=True)
    a=p.parse_args();print(run(a.source_root,a.inputs,a.summary,a.outliers,a.output))
