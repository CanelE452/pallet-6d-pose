"""Display-only additive fix; original V1 panels/selection/seal stay immutable."""
from pathlib import Path
import hashlib,json
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).resolve().parent
def b(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def main():
    meta=json.loads((R/'GALLERY_METADATA.json').read_text());panels=[];outputs=[];boxes=[]
    assert not (R/'DISPLAY_ONLY_READABILITY_ADDENDUM.json').exists()
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',21)
    for c in meta['cases']:
        assert b(Path(c['panel']['path']))==c['panel']
        panel=Image.open(c['panel']['path']).convert('RGB');d=ImageDraw.Draw(panel)
        d.rectangle((0,538,1800,591),fill='#111922')
        for x,name in zip((20,610,1200),('V76','V77','V78')):
            s=c['scores'][name];text=f'{name}: pV={s["p_visible"]:.6f}  {s["prediction"]}'
            bbox=d.textbbox((x,554),text,font=font);assert bbox[2]<x+580 and bbox[3]<590
            color='#66e0a0' if (s['prediction']=='VISIBLE')==bool(c['GT_label']) else '#ff9d88'
            d.text((x,554),text,font=font,fill=color);boxes.append(dict(id=c['id'],model=name,text=text,bbox=bbox))
        path=R/(Path(c['panel']['path']).stem+'_READABLE_V2.png')
        assert not path.exists();panel.save(path);outputs.append(b(path));panels.append(panel)
    contact=Image.new('RGB',(1800,820*4),'#111922')
    for i,p in enumerate(panels):contact.paste(p,(0,i*820))
    path=R/'CONTACT_SHEET_READABLE_V2.png';assert not path.exists();contact.save(path);outputs.append(b(path))
    addendum=dict(completed=True,reason='V1 three score strings overlapped at350px spacing; replace display strip only with590px columns. All original V1 artifacts preserved.',
        metadata=b(R/'GALLERY_METADATA.json'),original_gallery_seal=b(R/'GALLERY_SEAL.json'),selection=b(R/'SELECTION_LOCK_BEFORE_RGB.json'),
        code=b(Path(__file__).resolve()),outputs=outputs,text_bounding_boxes=boxes,
        probabilities_labels_coordinates_RGB_target_masks_changed=False,
        new_RGB_mask_cache_score_reads=0,source_original_panels_and_metadata_only=True,current_forward_fit_PnP=0)
    with (R/'DISPLAY_ONLY_READABILITY_ADDENDUM.json').open('x') as f:json.dump(addendum,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(b(R/'DISPLAY_ONLY_READABILITY_ADDENDUM.json')))
if __name__=='__main__':main()
