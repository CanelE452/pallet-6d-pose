"""Render the already frozen8 TRAIN source frames, no mask/model imports."""
from pathlib import Path
import hashlib, json, math
from PIL import Image, ImageDraw, ImageFont

OUT=Path(__file__).resolve().parent
fontpath='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
font=lambda n:ImageFont.truetype(fontpath,n)
CYAN=(0,255,255); MAGENTA=(255,0,220); YELLOW=(255,225,30)

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def coords(a):
    return tuple(float(a[k]) for k in ('reference_x','reference_y')),tuple(float(a[k]) for k in ('initial_query_x','initial_query_y'))
def markers(im, r, q, s=1, offset=(0,0), label=None):
    d=ImageDraw.Draw(im); x,y=r[0]*s+offset[0],r[1]*s+offset[1]; u,v=q[0]*s+offset[0],q[1]*s+offset[1]
    d.line((x,y,u,v),fill=YELLOW,width=2)
    d.line((x-6,y,x+6,y),fill=CYAN,width=2); d.line((x,y-6,x,y+6),fill=CYAN,width=2)
    d.ellipse((u-5,v-5,u+5,v+5),outline=MAGENTA,width=2)
    if label is not None:
        for tx,ty,txt,col in [(x+7,y-17,'r'+str(label),CYAN),(u+7,v+4,'q'+str(label),MAGENTA)]:
            d.text((tx,ty),txt,font=font(14),fill=col,stroke_width=2,stroke_fill='black')

def main():
    p=OUT/'SELECTION_AND_REVIEW_PROTOCOL.json'; lock=json.loads(p.read_text())
    assert sha(p)=='a2111d02aaf33a4550e73eca7141bc8d065a83c70ba94dc1f9de2a44a9baf868'
    (OUT/'FRAME_SHEETS').mkdir(exist_ok=True)
    assert not (OUT/'RENDER_OUTPUTS.json').exists()
    outputs=[]; overview=[]
    for j,row in enumerate(lock['frames']):
        assert row['split']=='train' and sha(row['image'])==row['image_sha256']
        assert sha(row['annotation'])==row['annotation_sha256']
        image=Image.open(row['image']).convert('RGB'); assert list(image.size[::-1])==row['raw_hw']
        marked=image.copy()
        for a in row['anchors8']: markers(marked,*coords(a),label=int(a['native_corner']))
        panel=Image.new('RGB',(1760,740),(24,26,29)); draw=ImageDraw.Draw(panel)
        draw.text((16,8),f"{j}: {row['id']}  {row['source_asset']}  TRAIN {image.width}x{image.height}",font=font(22),fill='white')
        draw.text((16,40),'Cyan + = projected reference r; magenta ring = saved initial q; yellow = displacement. Native8 already dynamic; no re-permutation.',font=font(15),fill=(220,220,220))
        panel.paste(marked,(12,86))
        draw.text((16,650),'All8 reference-centered native96 crops, nearest x2; dark padding is outside RGB.',font=font(17),fill='white')
        draw.text((16,679),'AI observation only. No mask, model, PnP, sourceVAL or current data. OOB/proxy does not certify semantic D/E/S.',font=font(15),fill=(210,210,210))
        corners=[]
        for a in row['anchors8']:
            k=int(a['native_corner']); r,q=coords(a); cx,cy=math.floor(r[0]+.5),math.floor(r[1]+.5)
            box=(cx-48,cy-48,cx+48,cy+48)
            crop=Image.new('RGB',(96,96),(40,42,45))
            valid=(max(0,box[0]),max(0,box[1]),min(image.width,box[2]),min(image.height,box[3]))
            if valid[0]<valid[2] and valid[1]<valid[3]: crop.paste(image.crop(valid),(valid[0]-box[0],valid[1]-box[1]))
            crop=crop.resize((192,192),Image.Resampling.NEAREST)
            markers(crop,r,q,s=2,offset=(-2*box[0],-2*box[1]))
            x=980+(k%4)*194; y=85+(k//4)*295
            draw.text((x,y),f"r{k} ({r[0]:.1f},{r[1]:.1f})",font=font(14),fill=CYAN)
            draw.text((x,y+19),f"q ({q[0]:.1f},{q[1]:.1f})",font=font(13),fill=MAGENTA)
            panel.paste(crop,(x,y+41))
            draw.text((x,y+237),f"|q-r|={float(a['source_coordinate_displacement_px']):.2f}px",font=font(14),fill='white')
            inframe=0<=r[0]<image.width and 0<=r[1]<image.height
            draw.text((x,y+258),'r in RGB' if inframe else 'r OUTSIDE RGB',font=font(13),fill=(180,220,180) if inframe else (255,170,100))
            corners.append({'native_corner':k,'reference':r,'initial_query':q,'reference_crop_native_xyxy':box,'reference_inframe':inframe})
        target=OUT/'FRAME_SHEETS'/f'{j:02d}_{row["id"]}.png'; panel.save(target)
        thumb=Image.new('RGB',(600,440),(24,26,29)); dd=ImageDraw.Draw(thumb)
        dd.text((8,5),f"{j}: {row['id']} {row['source_asset']}",font=font(16),fill='white')
        view=marked.copy(); view.thumbnail((584,390)); thumb.paste(view,(8,40)); overview.append(thumb)
        outputs.append({'id':row['id'],'frame_sheet':str(target),'sha256':sha(target),'corners8':corners})
    sheet=Image.new('RGB',(1200,1760),(18,20,23))
    for j,t in enumerate(overview): sheet.paste(t,((j%2)*600,(j//2)*440))
    target=OUT/'FIXED8_TRAIN_REFERENCE_QUERY_CONTACT_SHEET.png'; sheet.save(target)
    result={'selection_lock_sha256':sha(p),'source_RGB_decoded':8,'mask_model_VAL_current_data_decoded':0,'frame_outputs':outputs,'contact_sheet':{'path':str(target),'sha256':sha(target)},'new_coordinate_estimation':False}
    (OUT/'RENDER_OUTPUTS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'contact_sheet':str(target),'frame_sheets':[r['frame_sheet'] for r in outputs],'manifest_sha256':sha(OUT/'RENDER_OUTPUTS.json')}))

if __name__=='__main__': main()
