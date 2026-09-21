"""Render frozen corner evidence into GitHub-readable comparison figures."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE=Path(__file__).resolve().parent
ARMS=('R0','N2','REPLAY','A10','A11')


def main():
    rows=json.loads((HERE/'D1_TRANSITIONS.json').read_text())['rows']
    review=json.loads((HERE/'D3_REVIEW_SELECTION.json').read_text())['rows']
    lookup={(r['frame_id'],r['corner_id']):r for r in rows}
    delta=lambda r:r['A11']['error_px']-r['A10']['error_px']
    groups=[('gain','A11이 새로 맞힌 사례',sorted([r for r in rows if r['transition10']=='BAD->GOOD'],key=lambda r:(delta(r),r['frame_id'],r['corner_id']))[:2]),
        ('loss','A10은 맞았지만 A11이 놓친 사례',sorted([r for r in rows if r['transition10']=='GOOD->BAD'],key=lambda r:(-delta(r),r['frame_id'],r['corner_id']))[:2]),
        ('hard','R0의 큰 오류를 복구한 사례',sorted([r for r in rows if r['hard_recovery']],key=lambda r:(-r['R0']['error_px'],r['frame_id'],r['corner_id']))[:2]),
        ('damage','원래 정상점이 손상된 사례',[r for r in rows if r['good_damage']]),
        ('random','기존 고정 랜덤 대조 사례',[lookup[(r['frame_id'],r['corner_id'])] for r in review if 'RANDOM_CONTROL' in r['reasons']][:2])]
    dest=HERE/'inline_figures';dest.mkdir(exist_ok=True)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
    manifest=[];lines=[]
    for group,title,rr in groups:
        lines += ['### '+title,'']
        for rank,r in enumerate(rr,1):
            image=Image.open(HERE/'review_images'/(r['frame_id'].replace(':','__')+'.jpg')).convert('RGB')
            gt=np.asarray(r['gt_xy']);points=np.array([gt,*[r[a]['xy'] for a in ARMS]],float)
            lo=points.min(0)-45;hi=points.max(0)+45;center=(lo+hi)/2;wh=np.maximum(hi-lo,[160,120])
            box=(int(max(0,center[0]-wh[0]/2)),int(max(0,center[1]-wh[1]/2)),int(min(640,center[0]+wh[0]/2)),int(min(480,center[1]+wh[1]/2)))
            assert box[2]>box[0] and box[3]>box[1]
            canvas=Image.new('RGB',(1440,840),'#10212b')
            for i,a in enumerate(('FULL_RGB',*ARMS)):
                tile=Image.new('RGB',(480,420),'#10212b');draw=ImageDraw.Draw(tile)
                titletext=f'Full RGB | GT G{r["corner_id"]}' if i==0 else f'{a} | error {r[a]["error_px"]:.2f}px'
                draw.text((8,6),titletext,font=font,fill='white')
                raw=image.copy();rd=ImageDraw.Draw(raw)
                x,y=gt;rd.line((x-5,y,x+5,y),fill='#66ff66',width=2);rd.line((x,y-5,x,y+5),fill='#66ff66',width=2)
                if i==0:
                    rd.rectangle(box,outline='#ffa64d',width=2);display=raw.resize((480,360));tile.paste(display,(0,35))
                else:
                    raw=raw.crop(box);s=min(480/raw.width,360/raw.height);w,h=round(raw.width*s),round(raw.height*s)
                    display=raw.resize((w,h));pd=ImageDraw.Draw(display)
                    p=np.asarray(r[a]['xy']);gx,gy=(gt-np.array(box[:2]))*s;px,py=(p-np.array(box[:2]))*s
                    pd.line((gx,gy,px,py),fill='#ffa64d',width=2)
                    pd.ellipse((px-5,py-5,px+5,py+5),outline='#38baff',width=3)
                    pd.line((gx-7,gy,gx+7,gy),fill='#66ff66',width=3);pd.line((gx,gy-7,gx,gy+7),fill='#66ff66',width=3)
                    tile.paste(display,((480-w)//2,35+(360-h)//2))
                draw.text((8,398),'Green +: GT | Blue circle: prediction',font=small,fill='#bde6ce')
                canvas.paste(tile,((i%3)*480,(i//3)*420))
            name=f'{group}_{rank:02d}.jpg';path=dest/name
            assert not path.exists();canvas.save(path,quality=90,subsampling=0)
            lines += [f'[확인] `{r["frame_id"]}` / **G{r["corner_id"]}** — R0 {r["R0"]["error_px"]:.2f}px → A10 {r["A10"]["error_px"]:.2f}px → A11 {r["A11"]["error_px"]:.2f}px. 외부/자기 가림 subtype은 미확인이다.','',f'![{title} {rank}: 원본과 다섯 후보 비교](inline_figures/{name})','']
            manifest.append(dict(group=group,rank=rank,frame_id=r['frame_id'],corner_id=r['corner_id'],errors={a:r[a]['error_px'] for a in ARMS},crop=list(box),path=str(path.relative_to(HERE)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    payload=dict(status='[확인]',rows=manifest,selection='Gain/loss: largest A11-A10 change within transition group; hard: largest R0 error; all good damage; first two existing seed1 random controls. Groups may overlap.',
        no_new_prediction_or_training=True,original_analysis_audit_is_historical=True,
        report_before_sha256=hashlib.sha256((HERE/'POSTMORTEM_KO.md').read_bytes()).hexdigest())
    for name,text in [('INLINE_FIGURES.json',json.dumps(payload,ensure_ascii=False,indent=2)+'\n'),('INLINE_FIGURES.md','\n'.join(lines))]:
        with (HERE/name).open('x') as f:f.write(text)
    print('\n'.join(lines))


if __name__=='__main__':main()
