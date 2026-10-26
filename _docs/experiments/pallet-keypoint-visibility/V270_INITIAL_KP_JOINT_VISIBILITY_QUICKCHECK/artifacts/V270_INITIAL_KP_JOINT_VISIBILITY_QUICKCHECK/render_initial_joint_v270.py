"""Saved V270 outputs: full frames and display-only zooms. No neural inference."""
from pathlib import Path
import argparse, csv, hashlib, html, json, math, sys
import PIL
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE=Path(__file__).absolute().parent

def safe(p):
    p=Path(p);assert p.is_absolute() and not any(x in str(p).lower() for x in ('v112','cache','.pyc','..')),str(p)
    for a in [p,*p.parents]:assert not a.is_symlink(),str(a)
    return p

def binding(p):
    p=safe(p);b=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),size_bytes=len(b))

def dump(p,d):safe(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n')

def read(p):return json.loads(safe(p).read_bytes())

def verify(b):
    q=binding(b['path']);assert q['sha256']==b['sha256']
    for k in ('size','size_bytes','bytes'):
        if k in b:assert q['size_bytes']==b[k]
    return safe(b['path'])

def font(size):
    p=safe(Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
    return ImageFont.truetype(str(p),size) if p.exists() else ImageFont.load_default()

def marker(draw,xy,fill,r=6,outline=None):
    x,y=xy;draw.ellipse((x-r,y-r,x+r,y+r),fill=fill,outline=outline,width=3)

def cross(draw,xy,fill='#ff38ff',r=9,width=3):
    x,y=xy;draw.line((x-r,y,x+r,y),fill=fill,width=width);draw.line((x,y-r,x,y+r),fill=fill,width=width)

def main():
    p=argparse.ArgumentParser();p.add_argument('--evaluation',type=Path,default=HERE/'evaluation_v1');p.add_argument('--output',type=Path,default=HERE/'visual_v1');a=p.parse_args()
    ev=safe(a.evaluation);out=safe(a.output);assert not out.exists()
    complete=read(ev/'COMPLETE.json');assert complete['PASS'];verify(complete['results']);verify(complete['consumer_seal'])
    freeze=read(HERE/'CURRENT_PREDICTIONS_FROZEN.json');predpath=verify(freeze['predictions_npz']);verify(freeze['input_channel_native_identity'])
    with np.load(predpath,allow_pickle=False) as z:pred={k:z[k] for k in z.files}
    selection=read(ev/'VISUAL_CASE_SELECTION_BEFORE_RGB.json')['cases']
    with safe(ev/'ALL1784_CLASSIFICATION_COORDINATE_JOIN.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    assert len(rows)==1784
    inputs=safe(HERE.parent/'V267_FULL_IMAGE_CLIP_VISIBLE_MASK/current_fullframe_inputs_v2/FRAMES223_INPUT_ONLY.jsonl')
    frames=[json.loads(s) for s in inputs.read_text().splitlines() if s];assert len(frames)==223
    out.mkdir();dump(out/'VISUAL_INPUT_SELECTION_BEFORE_RGB.json',dict(PASS=True,completed=True,current_RGB_reads=0,
        fixed_selection=binding(ev/'VISUAL_CASE_SELECTION_BEFORE_RGB.json'),prediction_seal=binding(HERE/'CURRENT_PREDICTIONS_FROZEN.json'),
        input_manifest=binding(inputs),cases=selection,model_forwards=0,new_fits=0,display_crop_use='Only visualization. The saved model used the complete frame, without query/crop input.'))
    f18,f22,f28=font(18),font(22),font(28)
    source_bindings={}; rendered=[];rgb_cache={}
    for choice in selection:
        gi=int(choice['global_index']);r=rows[gi];fi,ni=gi//8,gi%8;fr=frames[fi]
        assert int(r['frame_index'])==fi and int(r['native_slot'])==ni and fr['frame_index']==fi
        if fi not in rgb_cache:
            imagepath=verify(fr['image_binding']);im=Image.open(imagepath).convert('RGB');arr=np.asarray(im)
            assert list(arr.shape[:2])==fr['raw_hw'] and hashlib.sha256(arr.tobytes()).hexdigest()==fr['decoded_RGB_sha256']
            rgb_cache[fi]=im;source_bindings[str(fi)]=binding(imagepath)
        raw=rgb_cache[fi];rw,rh=raw.size
        panelw=600;can=Image.new('RGB',(1800,1030),'#f6f8fb');d=ImageDraw.Draw(can)
        d.text((20,12),f'G{gi:04d} | {r["frame_id"]} | native {ni} | human {r["truth"]}',font=f28,fill='#151e2d')
        d.text((20,52),choice['reason'],font=f18,fill='#495568')
        center=pred['R0_xy_original'][fi,ni];side=160.;box=(float(center[0]-side/2),float(center[1]-side/2),float(center[0]+side/2),float(center[1]+side/2))
        patch=raw.crop(box)
        for col,arm in enumerate(('R0','VIS_ONLY','JOINT')):
            x0=col*panelw;xy=pred[arm+'_xy_original'][fi]
            prob=None if arm=='R0' else float(pred[arm+'_E_probability'][fi,ni]); available=bool(pred['frame_available'][fi])
            label='canvas confidence only' if arm=='R0' else ('U' if not available else 'E' if prob>=.5 else 'D')
            title='R0 matched square640' if arm=='R0' else arm
            d.text((x0+18,85),title,font=f22,fill='#14223b')
            d.text((x0+18,116),label+(f'  p(E)={prob:.3f}' if prob is not None else f'  kobj={float(pred["R0_kobj_probability"][fi,ni]):.3f}'),font=f18,fill='#293c55')
            oob=int(((xy[:,0]<0)|(xy[:,0]>=rw)|(xy[:,1]<0)|(xy[:,1]>=rh)).sum())
            d.text((x0+18,140),f'Predicted points outside frame: {oob}/9',font=f18,fill='#495568')
            scale=min(564/rw,338/rh);dw,dh=round(rw*scale),round(rh*scale);ox=x0+(600-dw)//2;oy=158+(338-dh)//2
            can.paste(raw.resize((dw,dh),Image.Resampling.LANCZOS),(ox,oy));d=ImageDraw.Draw(can)
            for k,(x,y) in enumerate(xy):
                if not (0<=x<rw and 0<=y<rh):continue
                if k==8:color='#c5c5c5'
                elif arm=='R0':color='#3cc9ff'
                else:color='#16f15b' if pred[arm+'_E_probability'][fi,k]<.5 else '#ff6f21'
                sx,sy=ox+x*scale,oy+y*scale;marker(d,(sx,sy),color,4)
                if k<8:d.text((sx+5,sy-12),str(k),font=f18,fill=color,stroke_width=1,stroke_fill='black')
                if k==ni:marker(d,(sx,sy),None,11,'#fff600')
            if r['manual_D']=='True':
                ref=np.array([float(r['manual_reference_x']),float(r['manual_reference_y'])]);cross(d,(ox+ref[0]*scale,oy+ref[1]*scale))
            d.text((x0+18,510),'Complete frame / fixed native identities',font=f18,fill='#293c55')
            zoom=patch.resize((450,450),Image.Resampling.NEAREST);zx,zy=x0+75,551;can.paste(zoom,(zx,zy));d=ImageDraw.Draw(can)
            factor=450/side
            for k,(x,y) in enumerate(xy):
                px,py=(x-box[0])*factor,(y-box[1])*factor
                if not (0<=px<=450 and 0<=py<=450):continue
                if k==8:color='#c5c5c5'
                elif arm=='R0':color='#3cc9ff'
                else:color='#16f15b' if pred[arm+'_E_probability'][fi,k]<.5 else '#ff6f21'
                marker(d,(zx+px,zy+py),color,6)
                if k==ni:marker(d,(zx+px,zy+py),None,14,'#fff600')
                if k<8:d.text((zx+px+8,zy+py-12),str(k),font=f22,fill=color,stroke_width=1,stroke_fill='black')
            if r['manual_D']=='True':
                ref=np.array([float(r['manual_reference_x']),float(r['manual_reference_y'])]);px,py=(ref-np.array(box[:2]))*factor
                if 0<=px<=450 and 0<=py<=450:cross(d,(zx+px,zy+py),r=14,width=4)
            d.text((x0+18,1005),'Zoom is DISPLAY ONLY; same fixed R0 center',font=f18,fill='#293c55')
        name=f'CASE_G{gi:04d}.png';safe(out/name);can.save(out/name)
        rendered.append(dict(global_index=gi,frame_index=fi,native_slot=ni,frame_id=r['frame_id'],human=r['truth'],
            VIS_ONLY=r['VIS_ONLY_D_E_U'],JOINT=r['JOINT_D_E_U'],VIS_ONLY_p_E=float(r['VIS_ONLY_p_E']),JOINT_p_E=float(r['JOINT_p_E']),
            reason=choice['reason'],file=name,display_crop_center_R0_original_xy=center.tolist(),display_crop_side_original_px=side,
            manual_D_reference_only=r['manual_D']=='True'))
    intro='<h1>V270 전체 이미지 초기 KP 공동 출력: 성공/실패</h1><p>보임 D=초록, 외부 가림 E=주황, 선택 native=노란 원, 수동 보임 참조=자홍 십자. R0 파랑은 기존 좌표이며 canvas 신뢰도를 D/E로 사용하지 않았습니다. 아래 확대는 표시용입니다. 실제 새 모델은 전체 640 입력을 사용했습니다. 숨은 점 위치를 정답으로 주장하지 않습니다.</p>'
    cards=''.join(f'<article><h2>G{x["global_index"]:04d} · {html.escape(x["frame_id"])} · human {x["human"]} · VIS {x["VIS_ONLY"]} → JOINT {x["JOINT"]}</h2><p>{html.escape(x["reason"])}</p><a href="{x["file"]}"><img loading="lazy" src="{x["file"]}"></a></article>'for x in rendered)
    safe(out/'INDEX_KO.html').write_text('<!doctype html><meta charset="utf-8"><title>V270 초기 KP 공동 출력 성공/실패</title><style>body{font:17px sans-serif;background:#edf2f8;color:#18314d;max-width:1600px;margin:24px auto}article{background:white;margin:24px 0;padding:18px;border-radius:12px}img{width:100%;height:auto}p{line-height:1.65}</style>'+intro+cards)
    dump(out/'CASES.json',rendered)
    dump(out/'COMPLETE.json',dict(PASS=True,completed=True,code=binding(Path(__file__).absolute()),
        evaluation_complete=binding(ev/'COMPLETE.json'),prediction_seal=binding(HERE/'CURRENT_PREDICTIONS_FROZEN.json'),
        source_images=source_bindings,case_count=len(rendered),model_forwards=0,new_fits=0,
        display_only_zoom=True,semantic_channels_source_fixed=True,RGB_decode_environment=dict(python=sys.executable,Pillow=PIL.__version__),fullframe_OOB_markers_clipped_for_display_only=True,
        outputs={x:binding(out/x) for x in ['INDEX_KO.html','CASES.json']+[r['file'] for r in rendered]}))
    print('PASS saved-output display-only visual cases',len(rendered))

if __name__=='__main__':main()
