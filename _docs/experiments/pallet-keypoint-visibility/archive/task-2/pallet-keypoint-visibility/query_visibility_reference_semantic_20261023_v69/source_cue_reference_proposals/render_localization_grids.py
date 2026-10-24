"""Recover existing native96 review raster, no original source RGB reads."""
from pathlib import Path
import hashlib,json,math
from PIL import Image,ImageDraw,ImageFont
OUT=Path(__file__).resolve().parent
F='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
font=lambda n:ImageFont.truetype(F,n)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def raw_crop(c):
    p=Path(c['raw96_sheet']['path']);assert sha(p)==c['raw96_sheet']['sha256']
    k=c['original_native_index'];x=8+(k%4)*304;y=45+(k//4)*345+28
    # Saved raster is native96 repeated nearest x3. Recover values exactly.
    im=Image.open(p).convert('RGB').crop((x,y,x+288,y+288))
    native=im.resize((96,96),Image.Resampling.NEAREST)
    assert native.resize((288,288),Image.Resampling.NEAREST).tobytes()==im.tobytes()
    r=c['original_renderer_r_xy'];cx,cy=[math.floor(v+.5) for v in r]
    return native,(cx-48,cy-48)
def main():
    lock=json.loads((OUT/'CUE_PROPOSAL_SELECTION_LOCK.json').read_text());dest=OUT/'LOCALIZATION_GRIDS';dest.mkdir(exist_ok=True)
    assert not (OUT/'LOCALIZATION_GRID_OUTPUTS.json').exists()
    outputs=[]
    for c in lock['candidates']:
        im,origin=raw_crop(c);im8=im.resize((768,768),Image.Resampling.NEAREST)
        panel=Image.new('RGB',(1680,910),(24,26,29));panel.paste(im8,(70,90));panel.paste(im8,(890,90));d=ImageDraw.Draw(panel)
        d.text((12,8),f"{c['selection_rank']} {c['candidate_point_ID']}   AI cue localization grid; raw raster duplicated at right",font=font(22),fill='white')
        d.text((12,42),f"old r={c['original_renderer_r_xy']} q={c['original_query_xy']} L={c['original_side_L']:.3f}; no coordinates moved",font=font(16),fill=(220,220,220))
        for i in range(97):
            if i%4:continue
            col=(90,90,90) if i%8 else (150,150,150)
            x=70+i*8;y=90+i*8
            d.line((x,90,x,858),fill=col,width=1);d.line((70,y,838,y),fill=col,width=1)
            if i%8==0:
                d.text((x-12,70),str(origin[0]+i),font=font(12),fill='white');d.text((26,y-5),str(origin[1]+i),font=font(12),fill='white')
        rx=70+(c['original_renderer_r_xy'][0]-origin[0]+.5)*8;ry=90+(c['original_renderer_r_xy'][1]-origin[1]+.5)*8
        # Open-center r cross preserves cue pixels near reference.
        for b in [(rx-24,ry,rx-10,ry),(rx+10,ry,rx+24,ry),(rx,ry-24,rx,ry-10),(rx,ry+10,rx,ry+24)]:d.line(b,fill=(0,255,255),width=2)
        d.text((12,875),'Grid:4-native-pixel lines; labels8px. Cyan open cross = OLD renderer r. Right raw center untouched.',font=font(18),fill='white')
        path=dest/f'{c["selection_rank"]:02d}_{c["candidate_point_ID"].replace(":","_k")}.png';panel.save(path)
        outputs.append({'candidate_point_ID':c['candidate_point_ID'],'path':str(path),'sha256':sha(path),'native96_origin_xy':origin,'native96_RGB_sha256':hashlib.sha256(im.tobytes()).hexdigest()})
    (OUT/'LOCALIZATION_GRID_OUTPUTS.json').write_text(json.dumps({'original_source_RGB_opened':0,'existing_raw_sheet_count':len({c['raw96_sheet']['path'] for c in lock['candidates']}),'recovered_native_rasters':8,'nearest3_roundtrip_exact':True,'frames':outputs},indent=2)+'\n')
    print(json.dumps({'outputs':[r['path'] for r in outputs]}))
if __name__=='__main__':main()
