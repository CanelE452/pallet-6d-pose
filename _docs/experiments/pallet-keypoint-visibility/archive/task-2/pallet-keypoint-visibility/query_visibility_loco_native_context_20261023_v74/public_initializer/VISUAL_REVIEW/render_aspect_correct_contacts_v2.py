"""Additive aspect-preserving all20 contact sheets; source audit unchanged."""
from pathlib import Path
import json
import hashlib
from PIL import Image,ImageDraw,ImageFont
HERE=Path(__file__).resolve().parent

def main():
    rows=[json.loads(s) for s in (HERE.parent/'OUTPUT_AUDIT/PER_FRAME_BBOX_DIAGNOSTIC.jsonl').read_text().splitlines()]
    figs=[]
    for subset in range(1,6):
        selected=[r for r in rows if r['source_subset']==subset];assert len(selected)==4
        target=HERE/f'SUBSET_{subset}_ALL4_NATIVE_RAW9_ASPECT_V2.png';assert not target.exists()
        canvas=Image.new('RGB',(1520,2500),(248,249,252));draw=ImageDraw.Draw(canvas)
        draw.text((20,12),f'SUBSET {subset}: all four fixed frames, unchanged raw9 / selected box / official GT boxes',font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',24),fill=(20,30,45))
        for j,r in enumerate(selected):
            pic=Image.open(r['native_overlay']['path']).convert('RGB');assert pic.size==(1520,610)
            canvas.paste(pic,(0,60+j*610))
        canvas.save(target);figs.append(dict(path=str(target),sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    p=HERE/'ASPECT_CORRECT_CONTACTS_V2.json';assert not p.exists()
    p.write_text(json.dumps(dict(completed=True,figures=figs,individual_panels_preserved=True,V1_contact_note='Initial2x2 contact layout distorted aspect when compressed horizontally; retained unchanged. These additiveV2 vertical contacts preserve native-overlay image aspect exactly.',model_forward_training_current_PnP=0),indent=2)+'\n')
    print(json.dumps(figs))

if __name__=='__main__':main()
