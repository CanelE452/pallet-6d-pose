"""AI working localization from actual raw RGB/native grid, without mask reads."""
from pathlib import Path
import hashlib,json
import numpy as np
from PIL import Image,ImageDraw,ImageFont

HERE=Path(__file__).resolve().parent
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
# At most one new operational physical cue per preselected target. Native8 UNKNOWN.
PROPOSALS=[
 ((241,338),(237,333,246,344),'캔 앞면 upper-left rim이 옆면으로 꺾이는 지점','medium'),
 ((171,199),(167,194,176,204),'원통 캔 상단 타원 rim의 왼쪽 끝','medium'),
 ((183,88),(179,83,188,93),'흰 bottle cap/neck의 왼쪽 상단 외곽 전환점','medium'),
 ((220,245),(215,240,225,250),'marker 흰 cap 상단의 왼쪽 rounded 끝','medium'),
 ((275,301),(270,296,281,307),'가위 closed blade의 맨 끝 tip','medium'),
 ((486,75),(481,70,492,81),'노란 bottle 작은 cap의 왼쪽 상단 turn','medium'),
 ((180,124),(176,120,185,129),'빨간 상자 앞/왼쪽/위 세 면이 만나는 실제 모서리','high'),
 ((187,204),(182,199,193,210),'설탕 상자 앞/왼쪽/윗면이 만나는 실제 모서리','high'),
 ((226,386),(221,380,232,392),'목재 block 윗면/앞면/왼쪽 면의 전면 모서리','medium'),
 ((302,112),(297,106,308,119),'파란 pitcher 상단 rim의 왼쪽 끝','medium'),
 ((102,132),(97,126,109,139),'파란 pitcher 왼쪽 pouring lip 끝의 turn','medium'),
 ((322,161),(317,156,328,168),'노란 bottle body의 왼쪽 하단 foot turn','medium')]


def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def ref(f):return dict(path=str(Path(f).resolve()),sha256=sha(f))
def text(d,xy,s,size=20,color='#172536'):d.text(xy,s,font=ImageFont.truetype(FONT,size),fill=color)


def main():
    out=HERE/'RGB_ONLY_WORKING_LOCALIZATION';out.mkdir(exist_ok=False)
    rows=[json.loads(s)for s in (HERE/'SOURCE_LINEAGE_GT_ONLY.jsonl').read_text().splitlines()]
    ledger=[];pages=[]
    for page in range(3):
        canvas=Image.new('RGB',(1280,1235),'#f5f7fa');d=ImageDraw.Draw(canvas)
        text(d,(18,10),'RGB-only working cue localization · native64 raw / grid + proposed point',25)
        text(d,(18,48),'Mask exposure occurred earlier; NO mask read in this stage. Provisional geometry, not training gold.',19)
        for i in range(4):
            k=page*4+i;r=rows[k]['original'];xy,bb,description,confidence=PROPOSALS[k]
            assert sha(r['image']['path'])==r['image']['sha256'];im=Image.open(r['image']['path']).convert('RGB')
            x,y=xy;box=(x-32,y-32,x+32,y+32);raw=im.crop(box).resize((256,256),Image.Resampling.NEAREST)
            grid=raw.copy();gd=ImageDraw.Draw(grid)
            for xx in range(((x-32)//20+1)*20,x+32,20):
                px=(xx-(x-32))*4;gd.line((px,0,px,255),fill='#bfb66c');gd.text((px+2,2),str(xx),font=ImageFont.truetype(FONT,13),fill='#ffff00',stroke_width=1,stroke_fill='#000000')
            for yy in range(((y-32)//20+1)*20,y+32,20):
                py=(yy-(y-32))*4;gd.line((0,py,255,py),fill='#bfb66c');gd.text((2,py+1),str(yy),font=ImageFont.truetype(FONT,13),fill='#ffff00',stroke_width=1,stroke_fill='#000000')
            gd.rectangle(tuple((v-(x-32 if j%2==0 else y-32))*4 for j,v in enumerate(bb)),outline='#ffbb00',width=2)
            gd.line((120,128,136,128),fill='#ff00bb',width=2);gd.line((128,120,128,136),fill='#ff00bb',width=2)
            ox=18+(i%2)*630;oy=130+(i//2)*540
            text(d,(ox,oy-45),f'{k+1:02d} {r["split"]} {r["id"]}',20)
            text(d,(ox,oy-18),f'proposed r={xy} / bbox={bb}',18)
            canvas.paste(raw,(ox,oy+15));canvas.paste(grid,(ox+275,oy+15))
            text(d,(ox,oy+285),description,19)
            text(d,(ox,oy+318),'중심/황색 bbox는 RGB-only 제안. GT 소유권 sanity 전.',18)
            ledger.append(dict(target_id=r['id'],split=r['split'],source_RGB=r['image'],reference_xy=list(xy),uncertainty_bbox_xyxy=list(bb),uncertainty_semantics='Half-open native coordinate rectangle localizing the visually extended cue; per-cue RGB-only proposal.',cue_description_ko=description,AI_localization_confidence=confidence,physical_cue_visible_proposal=True,native8_semantic_ID='UNKNOWN',GT_mask_read_during_localization=False,source_native64_crop_xyxy=list(box)))
        f=out/f'WORKING_RAW64_{page+1}.png';canvas.save(f);pages.append(ref(f))
    report=dict(schema='V71_YCB_RGB_ONLY_WORKING_LOCALIZATION_V1',completed=True,code=ref(__file__),selection_lock=ref(HERE/'SELECTION_LOCK.json'),raw_grid_manifest=ref(HERE/'RAW_NATIVE20_LOCALIZATION/MANIFEST.json'),prior_mask_exposure=True,mask_ownership_not_yet_checked=True,not_frozen_for_training=True,proposals=ledger,figures=pages,GT_mask_reads_this_stage=0,current_model_forward_training_PnP=0)
    (out/'WORKING_PROPOSALS.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(ref(out/'WORKING_PROPOSALS.json')))


if __name__=='__main__':main()
