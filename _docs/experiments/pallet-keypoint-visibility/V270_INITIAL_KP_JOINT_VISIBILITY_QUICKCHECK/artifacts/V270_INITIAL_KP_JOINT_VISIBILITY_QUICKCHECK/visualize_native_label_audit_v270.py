"""SYN-only saved-label pixel-risk audit and display-only native-point crops.

M0/M4 are diagnostic evidence, never used to revise the frozen proxy targets.
No model, real data, fit, or input crops. All RGB choices saved before RGB reads.
"""
from pathlib import Path
import collections
import csv
import hashlib
import html
import io
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE=Path(__file__).absolute().parent
TARGET=HERE/'syn_joint_targets_v1'
SOURCE=HERE.parent/'V267_FULL_IMAGE_CLIP_VISIBLE_MASK/data_audit/FRAME_MANIFEST.jsonl'


def safe(value):
    p=Path(value)
    if not p.is_absolute() or any(x in str(p).lower() for x in ('v112','cache','.pyc','..')):
        raise ValueError(str(p))
    for a in reversed((p,*p.parents)):
        if a.is_symlink(): raise ValueError(str(a))
    return p


def binding(path):
    p=safe(path);b=p.read_bytes()
    return {'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'size_bytes':len(b)}


def checked(bound):
    assert binding(bound['path'])==bound,bound
    return safe(bound['path']).read_bytes()


def savej(path,x):
    p=safe(path);assert not p.exists(),p
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n')


def center_sample(mask,xy):
    h,w=mask.shape;x,y=np.floor(np.asarray(xy)+.5).astype(int)
    if not (0<=x<w and 0<=y<h):return None,None
    patch=mask[max(0,y-1):min(h,y+2),max(0,x-1):min(w,x+2)]
    return bool(mask[y,x]),bool(patch.any())


def crop80(im,xy):
    x,y=np.floor(np.asarray(xy)+.5).astype(int)
    return im.crop((x-40,y-40,x+40,y+40)).resize((320,320),Image.Resampling.NEAREST)


def thumb(im,size=(640,360)):
    z=Image.new('RGB',size,(15,18,22));a=im.copy();a.thumbnail(size,Image.Resampling.BILINEAR)
    z.paste(a,((size[0]-a.width)//2,(size[1]-a.height)//2));return z


def main():
    out=safe(HERE/'syn_label_visual_audit_v1');assert not out.exists(),out
    complete=json.loads(safe(TARGET/'COMPLETE.json').read_text());assert complete['PASS'] and complete['completed']
    rows=[json.loads(z) for z in checked(complete['outputs']['FRAME_TARGETS1003.jsonl']).decode().splitlines()]
    manifest=[json.loads(z) for z in safe(SOURCE).read_text().splitlines()]
    assert len(rows)==len(manifest)==1003
    pair_selection=json.loads(checked(complete['outputs']['EXAMPLES_SELECTED_BEFORE_RGB.json']))
    out.mkdir();savej(out/'PIXEL_AUDIT_PROTOCOL_BEFORE_MASK_READS.json',{
        'code':binding(Path(__file__).absolute()),'target_complete':binding(TARGET/'COMPLETE.json'),
        'diagnostic_only':True,'target_changes':0,'model_forwards':0,'current223_reads':0,
        'pixel_rule':'M0/M4 uint8>127 at floor(GT_xy+.5); independent 3x3 patch-any flag. Raster alias/virtual anchors prevent treating these flags as physical gold.',
        'risk_selection':'First source-order point per split and four named risk predicates; save selected cases before RGB decode.',
        'pair_selection':complete['outputs']['EXAMPLES_SELECTED_BEFORE_RGB.json']})
    counters=collections.defaultdict(collections.Counter);points=[];risk_first={};masks={}
    for r,m in zip(rows,manifest):
        assert r['id']==m['input']['id'] and r['frame_index']==m['input']['frame_index']
        a0=np.asarray(Image.open(io.BytesIO(checked(m['target']['M0_auxiliary']))))>127
        a4=np.asarray(Image.open(io.BytesIO(checked(m['target']['M4']))))>127
        assert a0.shape==a4.shape==tuple(r['raw_hw']) and not (a4&~a0).any()
        masks[r['frame_index']]=(a0,a4)
        for k,y in enumerate(r['visibility_y8']):
            if y<0:continue
            p=r['projected_native_dynamic8_xy'][k]
            c0,p0=center_sample(a0,p);c4,p4=center_sample(a4,p)
            c=counters[(r['split'],y)];c['N']+=1
            c['M0_center_target']+=int(c0 is True);c['M4_center_target']+=int(c4 is True)
            c['M0_patch3_any_target']+=int(p0 is True);c['M4_patch3_any_target']+=int(p4 is True)
            c['raster_center_unsampleable']+=int(c0 is None)
            risks=[]
            if y==0 and c0 is False:risks.append('CLEAR_PROXY_M0_CENTER_NON_TARGET')
            if y==0 and c0 is True and c4 is False:risks.append('CLEAR_PROXY_M0_CENTER_TARGET_REMOVED_FINAL')
            if y==1 and c4 is True:risks.append('BLOCKED_PROXY_M4_CENTER_TARGET')
            if y==1 and c0 is False:risks.append('BLOCKED_PROXY_M0_CENTER_NON_TARGET')
            z={'frame_index':r['frame_index'],'id':r['id'],'split':r['split'],'native_slot':k,
               'visibility_y':y,'external_ray_fraction':r['external_ray_fraction8'][k],
               'M0_center_target':c0,'M4_center_target':c4,'M0_patch3_any_target':p0,'M4_patch3_any_target':p4,
               'risk_flags':'|'.join(risks)}
            points.append(z)
            for risk in risks:risk_first.setdefault((r['split'],risk),dict(z,risk=risk))
    selected=[]
    for p in pair_selection['examples']:
        selected.append({'kind':p['kind'],'frame_index':p['frame_index'],'id':p['id'],
                         'slots':[p['clear_slot'],p['blocked_slot']]})
    for k,v in sorted(risk_first.items()):selected.append({'kind':v['risk'],'frame_index':v['frame_index'],
                                                          'id':v['id'],'slots':[v['native_slot']]})
    savej(out/'CASES_SELECTED_BEFORE_RGB.json',{'rules':'Frozen source-order same-frame proxy pairs + first split/risk point; never model-result selection.',
        'selected':selected,'RGB_decodes':0,'targets_unchanged':True})
    summary={'completed':True,'PASS':True,'census':[
        {'split':key[0],'visibility_y':key[1],**dict(v)} for key,v in sorted(counters.items())],
        'risks_are_not_proven_false_labels':'A cuboid anchor or boundary may land outside target raster; 9-ray fraction is not central-pixel visibility. Pixel disagreement diagnoses objective mismatch, not certified relabeling.',
        'risk_examples':list(risk_first.values()),'mask_arrays_decoded':2006,
        'targets_modified':0,'current223_reads':0,'model_forwards':0}
    savej(out/'PIXEL_PROXY_AUDIT.json',summary)
    with safe(out/'PIXEL_PROXY_POINTS.csv').open('x')as f:
        w=csv.DictWriter(f,fieldnames=list(points[0]));w.writeheader();w.writerows(points)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
    colors={0:(40,225,80),1:(245,75,75),-1:(235,190,45)}
    outputs=[];rgb_seen={}
    for n,s in enumerate(selected):
        r=rows[s['frame_index']];m=manifest[s['frame_index']]
        if r['frame_index'] not in rgb_seen:
            rgb_seen[r['frame_index']]=Image.open(io.BytesIO(checked(r['RGB']))).convert('RGB')
        im=rgb_seen[r['frame_index']];a0,a4=masks[r['frame_index']]
        assert (im.height,im.width)==tuple(r['raw_hw'])
        marked=im.copy();d=ImageDraw.Draw(marked)
        for k,p in enumerate(r['projected_native_dynamic8_xy']):
            col=colors[r['visibility_y8'][k]];x,y=p;d.ellipse((x-4,y-4,x+4,y+4),outline=col,width=2)
            d.text((x+4,y-9),str(k),font=small,fill=col,stroke_width=1,stroke_fill='black')
        canvas=Image.new('RGB',(1320,810),(20,24,30));draw=ImageDraw.Draw(canvas)
        draw.text((12,10),f"SYN frame {r['frame_index']} | {r['split']} | {r['id']} | {r['material_variant']}",font=font,fill='white')
        draw.text((12,38),s['kind'],font=small,fill=(255,205,120))
        draw.text((12,61),'FULL RGB input. Lower zooms are display only. GREEN=D_proxy; RED=E_proxy; YELLOW=unknown.',font=small,fill='white')
        canvas.paste(thumb(marked,(640,350)),(10,88))
        draw.text((670,94),'Targets remain frozen. These are virtual cuboid projections,',font=small,fill='white')
        draw.text((670,120),'not certified physical surface points or human visibility gold.',font=small,fill='white')
        draw.text((670,158),'D_proxy: source self-visible + fraction exactly 0.',font=small,fill=colors[0])
        draw.text((670,184),'E_proxy: source self-visible + fraction >= 0.5.',font=small,fill=colors[1])
        draw.text((670,222),'M0: target-alone silhouette. M4: final target silhouette.',font=small,fill='white')
        draw.text((670,248),'A silhouette at this pixel cannot certify native identity.',font=small,fill='white')
        # Each selected slot has RGB/M0/M4 zoom; two-pair cards show RGB+M4 per slot.
        slots=s['slots']
        panels=[]
        for k in slots:
            p=r['projected_native_dynamic8_xy'][k];y=r['visibility_y8'][k];f=r['external_ray_fraction8'][k]
            z=crop80(im,p);zd=ImageDraw.Draw(z);zd.line((150,160,170,160),fill=colors[y],width=2);zd.line((160,150,160,170),fill=colors[y],width=2)
            panels.append((f'k{k} RGB y={y} ray={f:.4f}',z))
            for key,a in [('M0',a0),('M4',a4)]:
                if len(slots)==2 and key=='M0':continue
                maskim=Image.fromarray((a.astype(np.uint8)*255)).convert('RGB');v=crop80(maskim,p)
                vd=ImageDraw.Draw(v);vd.line((150,160,170,160),fill=colors[y],width=2);vd.line((160,150,160,170),fill=colors[y],width=2)
                panels.append((f'k{k} {key} (binary >127)',v))
        for j,(title,z) in enumerate(panels):
            x=10+330*j;draw.text((x,453),title,font=small,fill='white');canvas.paste(z,(x,480))
        filename=f'CASE_{n:03d}_F{r["frame_index"]}_K{"_".join(map(str,slots))}.png'
        canvas.save(safe(out/filename));outputs.append({'file':filename,**s})
    index=['<!doctype html><html lang="ko"><meta charset="utf-8"><title>V270 합성 라벨 감사</title>',
      '<style>body{background:#111820;color:#eee;font:17px sans-serif;max-width:1440px;margin:auto}img{width:100%;height:auto}article{margin:28px 0}a{color:#8bcaff}</style>',
      '<h1>V270 합성 native 좌표·외부 ray 라벨 감사</h1>',
      '<p>원본 전체 이미지 입력을 유지한다. 아래 crop은 시각화 전용이다. 초록 D_proxy=fraction 0, 빨강 E_proxy≥0.5이며 self/outside/부분가림은 제외한다.</p>',
      '<p>이는 virtual cuboid 주변 ray 차단 교사다. 물리 표면점이나 사람의 native 보임·외부가림 정답으로 인증한 라벨이 아니다. M0/M4 대조는 위험 진단이며 라벨 수정에 쓰지 않았다.</p>']
    for x in outputs:index.append(f'<article><h2>{html.escape(x["kind"])} | {html.escape(x["id"])}</h2><img src="{x["file"]}"></article>')
    index.append('</html>');safe(out/'INDEX_KO.html').write_text('\n'.join(index)+'\n')
    savej(out/'COMPLETE.json',{'completed':True,'PASS':True,'code':binding(Path(__file__).absolute()),
        'target_complete':binding(TARGET/'COMPLETE.json'),'pixel_audit':binding(out/'PIXEL_PROXY_AUDIT.json'),
        'selection':binding(out/'CASES_SELECTED_BEFORE_RGB.json'),'index':binding(out/'INDEX_KO.html'),
        'RGB_images_decoded':len(rgb_seen),'cards':len(outputs),'targets_modified':0,'current223_reads':0,
        'model_forwards':0,'optimizer_updates':0,'images':[{**x,'binding':binding(out/x['file'])} for x in outputs]})
    print(json.dumps({'COMPLETE':binding(out/'COMPLETE.json'),'cards':len(outputs),'census':summary['census']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
