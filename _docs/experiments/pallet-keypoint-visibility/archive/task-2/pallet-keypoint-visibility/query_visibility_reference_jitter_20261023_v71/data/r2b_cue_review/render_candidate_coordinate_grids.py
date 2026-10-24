"""Eight proposed review regions, only already fixed public RGB."""
from pathlib import Path
import hashlib,json,math
from PIL import Image,ImageDraw,ImageFont
OUT=Path(__file__).resolve().parent;F='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf';font=lambda n:ImageFont.truetype(F,n)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    protocol=json.loads((OUT/'RGB_REVIEW_PROTOCOL.json').read_text());assert sha(OUT/'RGB_REVIEW_PROTOCOL.json')=='8d72f37f979e95ad296e04d930ea980fa383e6cc46617f6c79a2e02eae3e4ec4'
    rows=[];order=protocol['fixed48_review_order']
    for old in protocol['prior3_records']:
        pt=next(p for p in old['points'] if p['id'].startswith('P'))
        frame=next(r for r in order if r['stream']=='STORAGE_D455' and r['sensor_slot']==old['slot'])
        x,y=pt['xy'];rows.append({'record_id':'LEGACY_'+pt['id'],'kind':'prior3_recheck','frame':frame,'legacy_point':pt,'legacy_target':old['target'],'detail_region':[math.floor(x)-80,math.floor(y)-80,math.floor(x)+80,math.floor(y)+80]})
    # Fixed round-robin first5 non-prior frames. All have an actual visible lowwood pallet.
    regions=[(1,[300,880,460,1040],'Left loaded low wood pallet support block, below cardboard and distinct from green steelrack.'),
             (2,[100,210,260,370],'Low wood pallet under two cardboard cartons at left, behind blue metalcart; not the cart or carton.'),
             (3,[220,440,380,600],'Far/right deck end of foreground left loaded low wood pallet; not transport crate on right.'),
             (4,[440,730,600,890],'Far/right deck/support end of left low wood pallet beneath cardboard; not purplecrate.'),
             (5,[100,210,260,370],'Lowwood pallet under left cardboard stack; a distinct local wooddeck corner cue, same/near-duplicate view as GALILEOslot0.')]
    for rank,(i,b,target) in enumerate(regions):rows.append({'record_id':f'NEW_{rank}','kind':'new_RGB_only_proposal','frame':order[i],'legacy_point':None,'legacy_target':target,'detail_region':b})
    dest=OUT/'CANDIDATE_NATIVE_COORDINATE_GRIDS';dest.mkdir(exist_ok=True);outputs=[]
    for j,row in enumerate(rows):
        f=row['frame'];assert sha(f['image'])==f['image_sha256'];im=Image.open(f['image']).convert('RGB');b=row['detail_region'];crop=im.crop(b)
        assert crop.size==(160,160)
        panel=Image.new('RGB',(1480,800),(24,26,29));panel.paste(crop.resize((640,640),Image.Resampling.NEAREST),(85,90));panel.paste(crop.resize((640,640),Image.Resampling.NEAREST),(810,90));d=ImageDraw.Draw(panel)
        d.text((12,8),f"{j} {row['record_id']} / {f['stream']} slot{f['sensor_slot']} / grid+RAW / no masks or model",font=font(21),fill='white')
        d.text((12,40),'Native160 crop x4; coordinate guides at integer pixel centers every8px (labels16px). Raw raster right untouched.',font=font(16),fill='white')
        for i in range(0,160,8):
            xx=85+(i+.5)*4;yy=90+(i+.5)*4
            d.line((xx,90,xx,730),fill=(95,95,95),width=1);d.line((85,yy,725,yy),fill=(95,95,95),width=1)
            if i%16==0:d.text((xx-10,70),str(b[0]+i),font=font(12),fill='white');d.text((35,yy-5),str(b[1]+i),font=font(12),fill='white')
        if row['legacy_point']:
            x,y=row['legacy_point']['xy'];xx=85+(x-b[0]+.5)*4;yy=90+(y-b[1]+.5)*4
            for l in [(xx-18,yy,xx-7,yy),(xx+7,yy,xx+18,yy),(xx,yy-18,xx,yy-7),(xx,yy+7,xx,yy+18)]:d.line(l,fill=(0,255,255),width=2)
            d.text((12,750),f"Prior coordinate retained for recheck: {row['legacy_point']['xy']}, prior radiusguess={row['legacy_point']['coordinate_uncertainty_px']}px; not human GT.",font=font(17),fill=(0,255,255))
        else:d.text((12,750),'New exact xy/visual uncertainty not yet frozen; choose only a visible woodpallet cue or U/null.',font=font(17),fill=(230,210,150))
        path=dest/f'{j:02d}_{row["record_id"]}_GRID_RAW.png';panel.save(path);row['grid_raw_figure']={'path':str(path),'sha256':sha(path)};outputs.append(row)
    target=OUT/'CANDIDATE_FRAME_VISUAL_ORDER.json';assert not target.exists();target.write_text(json.dumps({'schema':'V71_R2B_CUE_REVIEW_REGIONS_V1','protocol_sha256':sha(OUT/'RGB_REVIEW_PROTOCOL.json'),'selected_new_review_orders':[1,2,3,4,5],'all48_overview_reviewed':True,'records':outputs},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'path':str(target),'sha256':sha(target),'figures':[r['grid_raw_figure']['path'] for r in outputs]}))
if __name__=='__main__':main()
