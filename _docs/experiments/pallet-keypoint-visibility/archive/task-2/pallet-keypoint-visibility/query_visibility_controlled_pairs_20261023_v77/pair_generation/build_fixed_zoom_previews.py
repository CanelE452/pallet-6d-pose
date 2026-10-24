"""Same locked four previews, exact before/after local input crop display."""
import ast,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
HERE=Path(__file__).resolve().parent
OUT=HERE/'fixed_zoom_previews'
P=HERE.parents[1]
CROP=P/'query_visibility_loco_native_context_20261023_v74/native_context_cache/compile_input_pack.py'
def b(p):return {'path':str(p),'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()}
def lines(p):return[json.loads(x)for x in p.read_text().splitlines()]
def font(n,bold=False):return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans'+('-Bold'if bold else'')+'.ttf',n)
def marker(im,p,col):
    x,y=map(float,p);d=ImageDraw.Draw(im);d.line((x-7,y,x+7,y),fill=col,width=2);d.line((x,y-7,x,y+7),fill=col,width=2)
def main():
    assert not OUT.exists();OUT.mkdir()
    meta=json.loads((HERE/'fixed_previews/SELECTION_BEFORE_PREVIEW_RGB.json').read_text());indices=meta['after_indices']
    selection=lines(HERE/'PAIR_SELECTION_GT_ONLY.jsonl');pairs=lines(HERE/'PAIR_LINEAGE_GT_ONLY.jsonl')
    n=next(n for n in ast.parse(CROP.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='crop');env={'np':np};exec(compile(ast.Module(body=[n],type_ignores=[]),str(CROP),'exec'),env);crop=env['crop']
    panels=[];records=[]
    for j,i in enumerate(indices):
        s=selection[i];p=pairs[i];before=np.asarray(Image.open(p['before_RGB']['path']).convert('RGB'));after=np.asarray(Image.open(p['after_RGB']['path']).convert('RGB'))
        q=np.asarray(s['initial_points'][s['anchor_k']],float);r=np.asarray(p['native_GTref8'][s['anchor_k']],float);L=s['roi_side'];h,w=before.shape[:2]
        bc,bv=crop(before,q,L,96);ac,av=crop(after,q,L,96);assert np.array_equal(bv,av)
        bp=OUT/f'{i:06d}_BEFORE_FULL_L_RGB96.png';ap=OUT/f'{i:06d}_AFTER_FULL_L_RGB96.png'
        Image.fromarray(bc.transpose(1,2,0)).save(bp);Image.fromarray(ac.transpose(1,2,0)).save(ap)
        paste=np.asarray(Image.open(p['pasted_mask']['path']))>0;u=((np.arange(96)+.5)/96-.5)*L;xx,yy=np.meshgrid(q[0]+u,q[1]+u);px=np.floor(xx+.5).astype(int);py=np.floor(yy+.5).astype(int)
        valid=(xx>=0)&(xx<w)&(yy>=0)&(yy<h)&(px>=0)&(px<w)&(py>=0)&(py<h);mask=np.zeros((96,96),bool);mask[valid]=paste[py[valid],px[valid]]
        changed=np.any(bc!=ac,axis=0);assert changed.any()
        tile=Image.new('RGB',(1450,430),'#111820');d=ImageDraw.Draw(tile)
        d.text((15,10),f'{s["source"]}/{s["split"]} {s["original_id"]} anchor{s["anchor_k"]}: originalV -> afterC',font=font(20,True),fill='white')
        for x,title,a in [(15,'BEFORE entireL RGB96',(bc.transpose(1,2,0))),(355,'AFTER entireL RGB96',(ac.transpose(1,2,0))),(695,'Same AFTER + GTpaste',(ac.transpose(1,2,0)))]:
            if x==695:
                a=a.astype(float);a[mask]=.5*a[mask]+.5*np.array([255,30,0]);a=np.floor(a+.5).astype(np.uint8)
            im=Image.fromarray(a).resize((288,288),Image.Resampling.NEAREST);marker(im,[143.5,143.5],(0,240,255));marker(im,((r-q)*96/L+48)*3-.5,(255,0,255))
            tile.paste(im,(x,75));d.text((x,45),title,font=font(17),fill='#dddddd')
        d.text((1030,75),f'q-r={np.linalg.norm(q-r):.3f}px\nL={L:.3f}px\nrho+2={p["transform"]["radius_px"]:.3f}px\nnewRGBpixels={int(changed.sum())}/9216\nalpha crop={int(mask.sum())}/9216\nanchor teacherC=True',font=font(19),fill='#dddddd')
        d.text((15,380),'Cyan q / magenta reference are display-only. Raw unmarked crop PNGs separate. No model score or GT-guided q movement.',font=font(17),fill='#dddddd')
        png=OUT/f'PAIR_{i:06d}_ZOOM.png';tile.save(png);panels.append(tile)
        records.append({'after_row_index':i,'original_id':s['original_id'],'anchor_k':s['anchor_k'],'q':q.tolist(),'r':r.tolist(),'L':L,'RGB96_changed_pixels':int(changed.sum()),'native_RGB_changed_pixels':int(np.any(before!=after,axis=2).sum()),'pasted_crop_pixels':int(mask.sum()),'before_crop':b(bp),'after_crop':b(ap),'panel':b(png)})
    sheet=Image.new('RGB',(1450,430*len(panels)),'black')
    for j,tile in enumerate(panels):sheet.paste(tile,(0,j*430))
    sheet.save(OUT/'CONTACT_SHEET.png')
    (OUT/'SEAL.json').write_text(json.dumps({'completed':True,'selection':b(HERE/'fixed_previews/SELECTION_BEFORE_PREVIEW_RGB.json'),'dataset':b(HERE/'SEAL.json'),'code':b(Path(__file__)),'cropcode':b(CROP),'cases':records,'outputs':[b(p)for p in sorted(OUT.iterdir())if p.is_file()and p.name!='SEAL.json'],'model_current_fit_PnP':0},indent=2)+'\n')
    print(json.dumps({'contact':b(OUT/'CONTACT_SHEET.png'),'seal':b(OUT/'SEAL.json')}))
if __name__=='__main__':main()
