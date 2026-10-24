import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
R=P/'query_visibility_matched_query_scalar_20261023_v94'
S=P/'query_visibility_flat_pallet_cue_bridge_20261023_v92/rgb96_review'
O=R/'reviewer_C_actual_cues'
def bind(p):
    b=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def write(p,x):
    assert not p.exists();p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')

def main():
    policy=dict(schema='V94_REVIEWER_C_ALL20_ACTUAL_SAVED_RGB96_REVIEW_BEFORE_PIXELS',scope=bind(R/'ROOT_PUBLIC_CUE_REVIEW_SCOPE_BEFORE_ACTUAL_CROPS.json'),created_UTC=datetime.now(timezone.utc).isoformat(),all_frames=20,all_slots=160,valid122=True,
       action='View all20 original fullcontext figures and all saved96 rawcrop sheets. Enlargements are nearest-neighbour visualization of exact stored96 pixels, no new crop/recenter. No B new annotations or model text.',
       definition='YES requires distinct pallet structural cue plus context-compatible selected instance ownership. Material-only is NO appearance, identity/repeated/blur ambiguity UNKNOWN. NO/UNKNOWN is not externalE/selfS. Any uncertain visible candidate remains separately described; no native gold or training admission.',
       annotation_geometry='Half-open bbox in unchanged96x96 pixelcoordinates; uncertainty is visual localization uncertainty, not mask/materialgold; q centre is (47.5,47.5). Never relocate q/L or expected reference.',
       prior_exposure='Prior V92 coarse AI source observations and earlier figures existed; not blinded and not human gold. Reviewer B V94 annotations not read.',
       original_source=bind(S/'RGB96_REVIEW_SEAL.json'),rows=bind(S/'ROWS_INPUT_ONLY.jsonl'),pack=bind(S/'INPUT_RGB96_ONLY.npz'),figures=bind(S/'RGB_PROVENANCE_AND_FIGURES.json'),new_model_current_fit_PnP=0)
    write(O/'POLICY_BEFORE_ACTUAL_CROP_PIXELS.json',policy)
    rows=[json.loads(x) for x in (S/'ROWS_INPUT_ONLY.jsonl').read_text().splitlines()]
    a=np.load(S/'INPUT_RGB96_ONLY.npz',allow_pickle=False)
    write(O/'PACK_SCHEMA_METADATA.json',{k:dict(shape=list(a[k].shape),dtype=str(a[k].dtype)) for k in a.files})
    print('KEYS',a.files)
    rgb=a['rgb96']
    if rgb.shape==(160,3,96,96): rgb=rgb.reshape(20,8,3,96,96)
    assert rgb.shape==(20,8,3,96,96)
    fdir=O/'raw_saved96_sheets';fdir.mkdir()
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13)
    receipts=[]
    for fi in range(20):
        canvas=Image.new('RGB',(4*336,2*370+48),'#f4f4f4');d=ImageDraw.Draw(canvas)
        d.text((12,8),f'FRAME {fi:02d} | EXACT SAVED96 x3 NEAREST | no new crop | q=(47.5,47.5)',fill='black',font=font)
        for k in range(8):
            x0=(k%4)*336;y0=48+(k//4)*370
            rr=rows[fi*8+k]
            d.text((x0+8,y0),f'k{k} {rr["input_status"]}',fill='black',font=small)
            im=Image.fromarray(np.transpose(rgb[fi,k],(1,2,0))).resize((288,288),Image.Resampling.NEAREST)
            canvas.paste(im,(x0+32,y0+38))
            for t in [0,16,32,48,64,80,96]:
                xx=x0+32+t*3;yy=y0+38+t*3
                d.text((xx-7,y0+21),str(t),fill='black',font=small)
                if t<96:d.text((x0+3,yy-6),str(t),fill='black',font=small)
            d.text((x0+8,y0+334),'Native state UNKNOWN; no GT/fit admission',fill='#444444',font=small)
        out=fdir/f'frame_{fi:02d}_raw96.png';canvas.save(out)
        receipts.append(dict(frame_index=fi,original_figure=bind(S/'figures'/f'frame_{fi:02d}.png'),derived_raw_sheet=bind(out)))
    write(O/'RAW_SHEET_RENDER_MANIFEST.json',dict(all20_retained=True,all160_retained=True,scale_only=True,no_new_source_crops=True,frames=receipts))

if __name__=='__main__':main()
