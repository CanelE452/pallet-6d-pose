"""Fixed first pair per source/split preview; no model/scores/current IO."""
import json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
HERE=Path(__file__).resolve().parent
OUT=HERE/'fixed_previews'
def b(p):return {'path':str(p),'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()}
def lines(p):return[json.loads(x)for x in p.read_text().splitlines()]
def font(n,bold=False):return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans'+('-Bold'if bold else'')+'.ttf',n)
def mark(image,xy,color):
    x,y=map(float,xy);d=ImageDraw.Draw(image);d.line((x-7,y,x+7,y),fill=color,width=2);d.line((x,y-7,x,y+7),fill=color,width=2)
def main():
    assert not OUT.exists();OUT.mkdir()
    seal=json.loads((HERE/'SEAL.json').read_text());assert seal['completed']
    selected=lines(HERE/'PAIR_SELECTION_GT_ONLY.jsonl');pairs=lines(HERE/'PAIR_LINEAGE_GT_ONLY.jsonl')
    assert len(selected)==len(pairs)==4799
    indices=[]
    for source in ['SYNTHETIC','YCB_PUBLIC_REAL']:
        for split in ['train','val']:indices.append(next(i for i,x in enumerate(selected)if x['source']==source and x['split']==split))
    lock={'selection':'First metadataorder afterrow per source×split; frozen originalselection, no newlabel/error/modelscore ranking',
        'after_indices':indices,'bindings':[b(HERE/'SEAL.json'),b(HERE/'PAIR_SELECTION_GT_ONLY.jsonl'),b(HERE/'PAIR_LINEAGE_GT_ONLY.jsonl'),b(Path(__file__))]}
    (OUT/'SELECTION_BEFORE_PREVIEW_RGB.json').write_text(json.dumps(lock,indent=2)+'\n')
    tiles=[];records=[]
    for j,i in enumerate(indices):
        s=selected[i];p=pairs[i];before=np.asarray(Image.open(p['before_RGB']['path']).convert('RGB'));after=np.asarray(Image.open(p['after_RGB']['path']).convert('RGB'))
        paste=np.asarray(Image.open(p['pasted_mask']['path']))>0
        assert b(p['after_RGB']['path'])==p['after_RGB']and b(p['pasted_mask']['path'])==p['pasted_mask']
        k=p['anchor_k'];r=np.asarray(p['native_GTref8'][k]);q=np.asarray(s['initial_points'][k]);rho=p['transform']['radius_px']-2;L=s['roi_side']
        h,w=before.shape[:2];scale=min(430/w,310/h)
        def marked(rgb,overlay=False):
            a=rgb.astype(float).copy()
            if overlay:a[paste]=.5*a[paste]+.5*np.array([255,50,0])
            im=Image.fromarray(np.floor(np.clip(a,0,255)+.5).astype(np.uint8)).resize((round(w*scale),round(h*scale)),Image.Resampling.LANCZOS)
            d=ImageDraw.Draw(im);d.rectangle(tuple(np.r_[(q-L/2)*scale,(q+L/2)*scale]),outline=(0,240,255),width=2)
            x,y=r*scale;rr=(rho+2)*scale;d.ellipse((x-rr,y-rr,x+rr,y+rr),outline=(255,0,255),width=2)
            mark(im,q*scale,(0,240,255));mark(im,r*scale,(255,0,255));return im
        tile=Image.new('RGB',(1400,480),'#111820');d=ImageDraw.Draw(tile)
        d.text((15,10),f'{j+1}. {s["source"]}/{s["split"]} {s["original_id"]} anchor{k}: originalV -> afterC={p["anchor_success_COVERED"]}',font=font(20,True),fill='white')
        for x,title,im in [(15,'BEFORE original RGB',marked(before)),(475,'AFTER actual opaque donor RGB',marked(after)),(935,'AFTER + GTpaste display overlay',marked(after,True))]:
            d.text((x,45),title,font=font(17),fill='#dddddd');tile.paste(im,(x,75))
        d.text((15,395),f'q cyan / r magenta / certifieddisk rho+2={rho+2:.3f}px; scale={p["transform"]["scale"]:.5f}; quarterturn={p["quarterturn"]}; donor={p["donor"]["id"]}',font=font(16),fill='#dddddd')
        d.text((15,423),f'PaintedM0fraction={p["painted_expected_target_fraction"]:.4f}; same q8,L,box,reference,source/split; convexself flag kept separate.',font=font(16),fill='#dddddd')
        d.text((15,451),'Controlled source weak local-support teacher; public q is GT-derived emulation. No semantic human gold, model scores, PnP or fit.',font=font(16),fill='#dddddd')
        png=OUT/f'PAIR_{i:06d}.png';tile.save(png);tiles.append(tile);records.append({'after_row_index':i,'original_id':s['original_id'],'anchor_k':k,'panel':b(png)})
    sheet=Image.new('RGB',(1400,480*4),'black')
    for i,t in enumerate(tiles):sheet.paste(t,(0,i*480))
    sheet.save(OUT/'CONTACT_SHEET.png')
    (OUT/'PREVIEW_SEAL.json').write_text(json.dumps({'completed':True,'dataset_seal':b(HERE/'SEAL.json'),'cases':records,'outputs':[b(x)for x in sorted(OUT.iterdir())if x.is_file()and x.name!='PREVIEW_SEAL.json']},indent=2)+'\n')
    print(json.dumps({'preview':b(OUT/'CONTACT_SHEET.png'),'seal':b(OUT/'PREVIEW_SEAL.json')}),flush=True)
if __name__=='__main__':main()
