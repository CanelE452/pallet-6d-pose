"""Display-only privacy redactions. Never imported by inference or training."""
from __future__ import annotations
import argparse
from pathlib import Path
import math
from PIL import Image, ImageDraw, ImageFont
from scripts.research.pallet_statewise_end_to_end_recovery_v1.common import read, rows, sha, write


def display_image(frame,privacy=None):
    if sha(frame['image'])!=frame['image_sha256']:raise ValueError('Actual RGB bytes changed from the sealed inference image')
    image=Image.open(frame['image']).convert('RGB')
    expected=frame.get('trace',{}).get('raw_hw',frame.get('raw_hw'))
    if expected is None or list(image.size[::-1])!=expected:raise ValueError('Actual RGB dimensions differ from the sealed raw canvas')
    if privacy is None:return image
    if privacy.get('measurement_or_inference_changes')is not False:raise ValueError('Display-only plan required')
    row=privacy['frames'].get(frame['id'])
    if row is None or row['original_image_sha256']!=frame['image_sha256']:raise ValueError('Privacy plan must cover exact original image ID/hash')
    for box in row.get('redactions',[]):
        x0,y0,x1,y1=box['xyxy_raw_pixels']
        if not(0<=x0<x1<=image.width and 0<=y0<y1<=image.height):raise ValueError('Redaction box outside original canvas')
        ImageDraw.Draw(image).rectangle((x0,y0,x1,y1),fill=(112,124,131))
    return image


def contacts(metadata,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    frames=[r for r in rows(metadata)if r['partition']in('TEST','REMAINDER')]
    if len(frames)!=192:raise ValueError('All PRIMARY192 required')
    records=[];tw,th=320,240;per=24
    try:font=ImageFont.truetype('DejaVuSans.ttf',12)
    except OSError:font=ImageFont.load_default()
    for block in range(math.ceil(len(frames)/per)):
        selected=frames[block*per:(block+1)*per];canvas=Image.new('RGB',(tw*4,(th+35)*6),'white');draw=ImageDraw.Draw(canvas)
        for j,frame in enumerate(selected):
            image=Image.open(frame['image']).convert('RGB');image.thumbnail((tw,th))
            x=(j%4)*tw;y=(j//4)*(th+35);dx=(tw-image.width)//2;dy=(th-image.height)//2
            canvas.paste(image,(x+dx,y+35+dy));draw.text((x+3,y+2),str(block*per+j)+' '+frame['id'],font=font,fill='#182f40')
            records.append(dict(index=block*per+j,id=frame['id'],raw_hw=frame['trace']['raw_hw'],
                original_image_sha256=frame['image_sha256'],sheet=f'contact_{block:02d}.png',
                thumbnail_transform=dict(scale_x=image.width/Image.open(frame['image']).width,scale_y=image.height/Image.open(frame['image']).height,
                    canvas_x=x+dx,canvas_y=y+35+dy)))
        canvas.save(output/f'contact_{block:02d}.png')
    write(output/'CONTACTS.json',dict(schema='NO_REFERENCE_PRIVACY_CONTACTS_V1',frames=records,rows=192,
        pose_or_reference_decoding=False,new_model_forward_calls=0,source_sha256=sha(__file__),
        manual_review='not implied by generation; root model pixel review is recorded separately'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--metadata',required=True);p.add_argument('--output',required=True);a=p.parse_args();contacts(a.metadata,a.output)


if __name__=='__main__':main()
