"""Actual saved current crops, only after completed sealed evaluation.

No model imports/forward, learning, threshold choice or coordinate correction.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

R=Path(__file__).resolve().parent
def load(p):return json.loads(Path(p).read_text())
def rows(p):
    with Path(p).open(newline='')as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(b):assert sha(b['path'])==b['sha256']
def key(r):return r['id'],str(r['native_corner'])
def order(r):return hashlib.sha256((r['id']+':'+str(r['native_corner'])).encode()).hexdigest()
def write(p,d):
    with Path(p).open('x')as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def cross(d,xy,color,size=7,width=2):
    x,y=xy;d.line((x-size,y,x+size,y),fill=color,width=width);d.line((x,y-size,x,y+size),fill=color,width=width)

def main(entry_path):
    e=load(entry_path);assert e['approved_by_root'] and e['schema']=='V91_POST_EVAL_ACTUAL_CROP_VISUAL_ENTRY'
    for k in ['protocol','code','evaluation_seal','evaluation_report','per_query','truth','manual_provenance','input_pack','raw_outputs']:check(e[k])
    assert e['code']==bind(__file__)
    assert load(e['evaluation_seal']['path'])['completed']
    assert load(e['evaluation_report']['path'])['completed']
    protocol=load(e['protocol']['path']);assert protocol['PnP_allowed'] is False
    out=Path(e['output_directory']);assert not out.exists();out.mkdir()
    truth=rows(e['truth']['path']);tmap={key(r):r for r in truth}
    saved=rows(e['per_query']['path']);smap={key(r):r for r in saved};assert len(saved)==1784 and set(smap)==set(tmap)
    manual={key(r):r for r in rows(e['manual_provenance']['path']) if r['annotation_coordinate_source']=='manual_click' and r['human_state']=='DIRECT_VISIBLE' and float(r['initial_reference_error_px'])<=3 and r['query_valid']=='True'}
    selected=[]
    for group in ['WOOD_PRIMARY105','SQUARE118']:
        for name in ['manualnearD_newabsent_v61absent','manualnearD_newvisible_v61absent','knownE_newvisible','knownE_newabsent']:
            candidates=[]
            for t in truth:
                k=key(t);s=smap[k]
                if t['group']!=group or s['V91_valid']!='True':continue
                if name.startswith('manualnearD'):
                    if k not in manual or t['present_at_fixed_0_5']!='False':continue
                    expected='ABSENT_NONVISIBLE' if name=='manualnearD_newabsent_v61absent' else 'VISIBLE'
                    if s['V91_raw_decision']!=expected:continue
                else:
                    if t['human_state']!='EXTERNAL_OCCLUDED' or t['paired_same_query_RGB_D_vs_E']!='True':continue
                    expected='VISIBLE' if name=='knownE_newvisible' else 'ABSENT_NONVISIBLE'
                    if s['V91_raw_decision']!=expected:continue
                candidates.append(t)
            if candidates:
                t=min(candidates,key=order);selected.append(dict(group=group,stratum=name,truth=t,saved=smap[key(t)],manual_provenance=manual.get(key(t))))
    # Selection is saved before ANY source RGB decode or input array deserialization.
    write(out/'SELECTION_BEFORE_RGB_AND_CROP_ARRAYS.json',dict(completed=True,protocol=e['protocol'],evaluation=e['evaluation_seal'],
        rule='Fixed CURRENT_PROTOCOL four strata per type; minSHA256(id:native) each, original paired-knownE and manualnearD groups. Missing strata remain missing. No replacement after viewing.',cases=selected,new_forward_fit_PnP=0))
    ids={key(r):i for i,r in enumerate(saved)}
    with np.load(e['input_pack']['path'],allow_pickle=False)as z:rgb96=z['rgb96']
    with np.load(e['raw_outputs']['path'],allow_pickle=False)as z:
        maps=z['physical_external_logit_map'];validmaps=z['map_valid']
    assert rgb96.shape==(1784,3,96,96) and maps.shape==(1784,9,9)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
    figures=[]
    for n,c in enumerate(selected):
        t=c['truth'];s=c['saved'];ix=ids[key(t)]
        assert sha(t['image'])==t['image_sha256'];original=Image.open(t['image']).convert('RGB')
        q=np.array([float(t['query_x']),float(t['query_y'])]);rr=np.array([float(t['human_reference_x']),float(t['human_reference_y'])]);L=float(s['roi_side'])
        im=original.copy();d=ImageDraw.Draw(im)
        marker=max(5,round(max(im.size)/180));cross(d,q,'magenta',marker,3);cross(d,rr,'lime',marker,2)
        d.rectangle((q[0]-L/2,q[1]-L/2,q[0]+L/2,q[1]+L/2),outline='magenta',width=2)
        scale=min(1280/im.width,500/im.height);im=im.resize((round(im.width*scale),round(im.height*scale)),Image.Resampling.LANCZOS)
        canvas=Image.new('RGB',(1280,1080),'#fafafa');d=ImageDraw.Draw(canvas)
        d.text((16,12),f"{n+1}. {c['group']} | {c['stratum']}",fill='black',font=font)
        d.text((16,38),f"id={t['id']} native={t['native_corner']} | human={t['human_state']} | error={float(t['initial_reference_error_px']):.3f}px",fill='black',font=font)
        d.text((16,64),f"V91 raw={s['V91_raw_decision']} zE={float(s['V91_raw_zE']):.4f} | v61 visible={t['present_at_fixed_0_5']} | legacy={t['legacy_stored_state']}",fill='black',font=font)
        d.text((16,90),'Magenta: original initial q/crop; green: existing evaluation r. Overlay only, no coordinate correction.',fill='black',font=small)
        canvas.paste(im,((1280-im.width)//2,120))
        top=650
        # The left panel is the actual immutable model input, no recropping.
        actual=Image.fromarray(rgb96[ix].transpose(1,2,0)).resize((288,288),Image.Resampling.NEAREST)
        ad=ImageDraw.Draw(actual);cross(ad,(143.5,143.5),'magenta',9,3)
        pr=96*(rr-q)/L+47.5
        if ((pr>=0)&(pr<96)).all():cross(ad,(pr+.5)*3-.5,'lime',8,2)
        canvas.paste(actual,(16,top))
        d.text((16,top-28),f'Actual saved RGB96; L={L:.3f}',fill='black',font=font)
        # A native-resolution diagnostic display only; it is not fed to the head.
        pixel=np.floor(q+.5).astype(int);native=original.crop((int(pixel[0]-32),int(pixel[1]-32),int(pixel[0]+32),int(pixel[1]+32))).resize((288,288),Image.Resampling.NEAREST)
        nd=ImageDraw.Draw(native);cross(nd,(q-(pixel-32)+.5)*4.5-.5,'magenta',9,3);cross(nd,(rr-(pixel-32)+.5)*4.5-.5,'lime',8,2)
        canvas.paste(native,(446,top));d.text((446,top-28),'Native64 display (not model input)',fill='black',font=font)
        pe=np.exp(-np.logaddexp(0,-maps[ix].astype(np.float64)))
        color=np.stack([255*pe,80*(1-np.abs(2*pe-1)),255*(1-pe)],-1).astype(np.uint8);color[~validmaps[ix]]=0
        heat=Image.fromarray(color).resize((288,288),Image.Resampling.NEAREST);hd=ImageDraw.Draw(heat);hd.rectangle((128,128,159,159),outline='yellow',width=3)
        canvas.paste(heat,(876,top));d.text((876,top-28),'Saved evidence9x9; yellow=q',fill='black',font=font)
        d.text((16,956),f"q=({q[0]:.3f},{q[1]:.3f}) | r=({rr[0]:.3f},{rr[1]:.3f}) | original image={original.size}",fill='black',font=font)
        d.text((16,984),'Red evidence: larger external score; blue: smaller. Score is conditional cue evidence, not a material mask.',fill='black',font=small)
        d.text((16,1011),'Human r and state are evaluation overlays only. All model outputs were frozen before this selection.',fill='black',font=small)
        fig=out/f'case_{n:02d}.png';canvas.save(fig)
        figures.append(dict(index=n,group=c['group'],stratum=c['stratum'],id=t['id'],native_corner=t['native_corner'],
            figure=bind(fig),image=bind(t['image']),image_raw_hw=list(original.size[::-1]),query_xy=q.tolist(),reference_xy=rr.tolist(),roi_side=L,
            initial_reference_error_px=float(t['initial_reference_error_px']),V91_raw_zE=float(s['V91_raw_zE']),actual_model_input_rgb96_SHA256=hashlib.sha256(rgb96[ix].tobytes()).hexdigest(),
            source_coordinate_provenance=c['manual_provenance']['annotation_coordinate_source'] if c['manual_provenance'] else 'SEE_REGISTERED_EVALUATION_REFERENCE',new_forward_fit_PnP=0))
    write(out/'ACTUAL_CROPS_AND_FIGURES.json',dict(completed=True,entry=bind(entry_path),selection=bind(out/'SELECTION_BEFORE_RGB_AND_CROP_ARRAYS.json'),cases=figures,
        actual_input_source=e['input_pack'],raw_output_source=e['raw_outputs'],new_forward_fit_PnP=0))
    print(json.dumps(dict(completed=True,cases=len(figures),manifest=bind(out/'ACTUAL_CROPS_AND_FIGURES.json'))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--entry',required=True);a=p.parse_args();main(a.entry)
