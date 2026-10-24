"""All eight root-locked regions, identical raw-coordinate crops in both RGBs."""
from pathlib import Path
import hashlib,json,math
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).resolve().parent
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
policy=json.loads((R/'AFTER_POLICY_BEFORE_RGB.json').read_text())
assert bind(Path(policy['bindings']['root_union']['path']))==policy['bindings']['root_union']
union=json.loads(Path(policy['bindings']['root_union']['path']).read_text())
source={}
for key,b in policy['only_RGB_files_allowed'].items():
    assert bind(Path(b['path']))['sha256']==b['sha256']
    with Image.open(b['path']) as opened:source[key]=opened.convert('RGB')
assert source['before'].size==source['after'].size==(640,480)
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
bold=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',22)
def mark(d,xy,color='#ffdf54'):
    x,y=xy;d.line((x-7,y,x+7,y),fill=color,width=2);d.line((x,y-7,x,y+7),fill=color,width=2)
overview=Image.new('RGB',(1320,540),'#13202a');od=ImageDraw.Draw(overview)
for n,(key,im) in enumerate(source.items()):od.text((10+660*n,10),key+' / raw native640x480 / NO WARP',font=bold,fill='white');overview.paste(im,(10+660*n,50))
overview.save(R/'RAW_PAIR_OVERVIEW.png')
contact=Image.new('RGB',(1240,360*8),'#13202a');records=[]
for n,cue in enumerate(union['proposals']):
    x,y=cue['reference_xy_unchanged'];rect=cue['uncertainty_bbox_xyxy_half_open_unchanged']
    box=[max(0,math.floor(x)-36),max(0,math.floor(y)-36),min(640,math.ceil(x)+37),min(480,math.ceil(y)+37)]
    assert box[0]<=rect[0]<rect[2]<=box[2] and box[1]<=rect[1]<rect[3]<=box[3]
    row=Image.new('RGB',(1240,360),'#13202a');d=ImageDraw.Draw(row)
    d.text((15,10),f'{n+1}. {cue["union_id"]} / same r=({x:g},{y:g}) / unc={rect}',font=bold,fill='white')
    crop_bindings={}
    for j,key in enumerate(('before','after')):
        crop=source[key].crop(tuple(box));path=R/f'{n:02d}_{cue["union_id"]}_{key}_UNMARKED.png';crop.save(path);crop_bindings[key]=bind(path)
        scale=min(280/crop.width,256/crop.height);display=crop.resize((round(crop.width*scale),round(crop.height*scale)),Image.Resampling.NEAREST)
        marked=display.copy();md=ImageDraw.Draw(marked)
        mark(md,((x-box[0])*scale,(y-box[1])*scale))
        md.rectangle(((rect[0]-box[0])*scale,(rect[1]-box[1])*scale,(rect[2]-box[0])*scale,(rect[3]-box[1])*scale),outline='#5ce0e7',width=2)
        for title,tile,xx in [(key+' UNMARKED',display,15+305*j),(key+' fixed r/range',marked,625+305*j)]:d.text((xx,48),title,font=small,fill='white');row.paste(tile,(xx,78))
    d.text((15,331),'Same coordinate windows; native8 UNKNOWN; no mask/depth/registration/model. AI source evidence only.',font=small,fill='#d5e0e8')
    path=R/f'{n:02d}_{cue["union_id"]}_PAIRED_REGION.png';row.save(path);contact.paste(row,(0,360*n))
    records.append(dict(union_id=cue['union_id'],r_unchanged=cue['reference_xy_unchanged'],uncertainty_unchanged=rect,native_crop_box=box,unmarked_crops=crop_bindings,paired_figure=bind(path)))
contact.save(R/'ALL8_LOCKED_REGIONS_CONTACT.png')
with (R/'PAIR_REGION_DISPLAY_RECEIPT.json').open('x') as f:json.dump(dict(completed=True,policy=bind(R/'AFTER_POLICY_BEFORE_RGB.json'),union=policy['bindings']['root_union'],code=bind(Path(__file__)),all8_order_retained=True,records=records,full_pair=bind(R/'RAW_PAIR_OVERVIEW.png'),contact=bind(R/'ALL8_LOCKED_REGIONS_CONTACT.png'),no_coordinate_relocalization_or_registration=True,mask_depth_otherreview_model_current_reads=0),f,indent=2);f.write('\n')
print(json.dumps(bind(R/'ALL8_LOCKED_REGIONS_CONTACT.png')))
