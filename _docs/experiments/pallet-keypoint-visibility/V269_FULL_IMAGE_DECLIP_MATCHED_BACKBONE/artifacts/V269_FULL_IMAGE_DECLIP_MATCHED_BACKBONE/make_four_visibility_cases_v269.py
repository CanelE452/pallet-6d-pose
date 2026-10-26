"""Actual images: four D/E classification outcomes, no coordinate movement."""
from pathlib import Path
import csv,hashlib,io,json,sys
from PIL import Image,ImageDraw,ImageFont

W=Path(__file__).absolute().parent

def safe(p):
    p=Path(p);assert p.is_absolute() and not any(s in str(p).lower()for s in('v112','cache','.pyc','..'))
    for a in reversed((p,*p.parents)):assert not a.is_symlink()
    return p
def load(p):return json.loads(safe(p).read_bytes())
def font(n):return ImageFont.truetype('/usr/share/fonts/truetype/nanum/NanumGothic.ttf',n)

def main():
    cases=[(85,'주석 보임 / 판정 보임','#16663e'),(1354,'주석 가림 / 판정 가림','#16663e'),
           (665,'주석 보임 / 판정 가림 → 오판','#a43d1c'),(300,'주석 가림 / 판정 보임 · 대응 확인','#a43d1c')]
    out=safe(W/'FOUR_VISIBILITY_CASES_KO.png');assert not out.exists() or '--redraw' in sys.argv
    p=safe(W/'FOUR_VISIBILITY_CASE_SELECTION.json');assert not p.exists() or '--redraw' in sys.argv
    p.write_text(json.dumps(dict(completed=True,selected=[x[0]for x in cases],coordinate_changes=0,
        selection='Fixed observed cases illustrating all four outcomes; not a prevalence estimate or blind selection.',
        prior_RGB_exposure=True,model_inputs='Whole image; zoom below is display-only'),ensure_ascii=False,indent=2)+'\n')
    meta=load(W/'evaluation_v1/compact_visual_inputs/INPUTS.json')
    frames={r['frame_index']:r for r in meta['frame_metadata']}
    files={r['frame_index']:r['file']for r in meta['full_RGBs']}
    rows=list(csv.DictReader(safe(W/'evaluation_v1/ALL1784_COMPARISON.csv').open()))
    canvas=Image.new('RGB',(1760,1160),'#f4f7fc');d=ImageDraw.Draw(canvas)
    d.text((24,18),'DeCLIP · 같은 초기 좌표에서 보임/외부 가림만 판정',font=font(34),fill='#182a42')
    d.text((24,68),'노란 표시=처음 추론한 점 · 좌표 이동0 · 전체 영상 입력 / 오른쪽 확대는 표시 전용',font=font(23),fill='#485b74')
    for k,(g,title,color) in enumerate(cases):
        x=24+(k%2)*870;y=135+(k//2)*490;fi=g//8;b=files[fi];raw=safe(b['path']).read_bytes()
        assert len(raw)==b['size_bytes'] and hashlib.sha256(raw).hexdigest()==b['sha256']
        with Image.open(io.BytesIO(raw))as im:rgb=im.convert('RGB').copy()
        xy=frames[fi]['original_query_xy8'][g%8];qx,qy=xy
        full=rgb.copy();fd=ImageDraw.Draw(full);r=max(4,round(full.width/140));fd.ellipse((qx-r,qy-r,qx+r,qy+r),outline='#ffea25',width=3)
        full.thumbnail((520,320),Image.Resampling.BICUBIC)
        crop=rgb.crop((round(qx)-48,round(qy)-48,round(qx)+48,round(qy)+48)).resize((280,280),Image.Resampling.NEAREST)
        cd=ImageDraw.Draw(crop);cd.line((132,140,148,140),fill='#ffea25',width=3);cd.line((140,132,140,148),fill='#ffea25',width=3)
        d.text((x,y),f'G{g} · '+title,font=font(27),fill=color)
        canvas.paste(full,(x,y+53+(320-full.height)//2));canvas.paste(crop,(x+540,y+70))
        pt=float(rows[g]['EVA_DECLIP_p_target']);pred='보임'if rows[g]['EVA_DECLIP_pred']=='D'else'가림'
        d.text((x,y+382),f'판정: {pred} · 초기 점의 대상 마스크 점수 {pt:.3f}',font=font(21),fill=color)
        if g==300:d.text((x,y+419),'G300: query와 숨은 native 참조 차이103.57px',font=font(20),fill='#a43d1c')
        elif g==665:d.text((x,y+419),'작고 어두운 대상의 보이는 점을 가림으로 오판',font=font(20),fill='#a43d1c')
    d.text((24,1120),'보임/가림 GT = 기존 사람 주석 · G300은 native/q 대응 차이 포함 · 마스크 점수는 보임 확률 보장값이 아님',font=font(22),fill='#485b74')
    canvas.save(out);print(json.dumps(dict(completed=True,PASS=True,figure=str(out),new_model_forwards_or_fits=0)))

if __name__=='__main__':main()
