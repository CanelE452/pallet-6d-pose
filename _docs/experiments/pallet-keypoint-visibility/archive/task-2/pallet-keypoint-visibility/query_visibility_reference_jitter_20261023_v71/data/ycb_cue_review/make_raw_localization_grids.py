"""Raw RGB + native20px grid only, before GT-only cue coordinate localization."""
from pathlib import Path
import hashlib,json
import numpy as np
from PIL import Image,ImageDraw,ImageFont

HERE=Path(__file__).resolve().parent
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def txt(d,xy,s,size=21,fill='#172536'):d.text(xy,s,font=ImageFont.truetype(FONT,size),fill=fill)


def main():
    out=HERE/'RAW_NATIVE20_LOCALIZATION';out.mkdir(exist_ok=False)
    source=[json.loads(s)for s in (HERE/'SOURCE_LINEAGE_GT_ONLY.jsonl').read_text().splitlines()]
    files=[]
    for k,row in enumerate(source):
        r=row['original'];b=r['image'];assert sha(b['path'])==b['sha256']
        original=Image.open(b['path']).convert('RGB');assert original.size==(640,480)
        grid=original.resize((1280,960),Image.Resampling.NEAREST);d=ImageDraw.Draw(grid)
        for x in range(0,640,20):
            d.line((x*2,0,x*2,959),fill='#c4c147',width=1)
            d.text((x*2+2,3),str(x),font=ImageFont.truetype(FONT,14),fill='#ffff00',stroke_width=1,stroke_fill='#000000')
        for y in range(0,480,20):
            d.line((0,y*2,1279,y*2),fill='#c4c147',width=1)
            d.text((3,y*2+1),str(y),font=ImageFont.truetype(FONT,14),fill='#ffff00',stroke_width=1,stroke_fill='#000000')
        b=r['box'];d.rectangle(tuple(x*2 for x in b),outline='#00d9ee',width=2)
        canvas=Image.new('RGB',(2600,1080),'#f5f7fa');dd=ImageDraw.Draw(canvas)
        txt(dd,(20,12),r['split']+' '+r['id']+' / object '+str(r['object_id'])+' · RAW RGB / native20px GRID',29)
        txt(dd,(20,58),'No masks/query/labels on this view. Cyan is the preexisting input roughbox. Prior review DID expose masks; this is not blind annotation.',21)
        canvas.paste(original.resize((1280,960),Image.Resampling.NEAREST),(10,100));canvas.paste(grid,(1310,100))
        f=out/f'{k+1:02d}_{r["split"]}_{r["id"].replace(":","_")}.png';canvas.save(f)
        files.append(dict(id=r['id'],split=r['split'],source_RGB=r['image'],file=ref(f),native_grid_step_px=20,raw_rgb_interpolation='nearest2x',GTmask_read=False))
    report=dict(schema='V71_YCB_RAW_RGB_NATIVE20_LOCALIZATION_V1',completed=True,code=ref(__file__),selection_lock=ref(HERE/'SELECTION_LOCK.json'),files=files,prior_mask_exposure=True,GT_mask_query_or_score_used_in_this_render=False,reference_xy_not_yet_localized=True,current_model_forward_training_PnP=0)
    (out/'MANIFEST.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(ref(out/'MANIFEST.json')))


if __name__=='__main__':main()
