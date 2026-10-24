"""Render only the eight preselected original factual96 arrays; no model."""
from pathlib import Path
import hashlib,json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
H=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def main():
    selection=H/'FIXED8_CASE_SELECTION_BEFORE_RGB.json';s=json.loads(selection.read_text());assert s['posthoc_diagnostic_selection_before_any_RGB_read']
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
    title=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',23)
    board=Image.new('RGB',(1280,1920),'#f7f8fa');d=ImageDraw.Draw(board)
    d.text((18,10),'Fixed 8 original source crops: C35 vs C36',font=title,fill='black')
    d.text((18,42),'Weak labels unchanged. Nominal pose/reference distances are NOT real pixel gold.',font=font,fill='#8c3333')
    records=[]
    for n,c in enumerate(s['cases']):
        p=c['saved_crop_packet'];assert sha(p['path'])==p['sha256']
        with np.load(p['path'],allow_pickle=False) as f:
            rgb=f['rgb96'][c['native_slot']].copy();valid=f['valid96'][c['native_slot']].copy()
            assert rgb.shape==(96,96,3) and rgb.dtype==np.uint8 and valid.shape==(96,96)
        assert hashlib.sha256(rgb.tobytes()).hexdigest()==c['expected_RGB96_sha256']
        assert hashlib.sha256(valid.tobytes()).hexdigest()==c['expected_valid96_sha256']
        out=H/f"CASE_{n+1:02d}_g{c['global_index']:04d}_ORIGINAL96.png";Image.fromarray(rgb).save(out)
        x=(n%2)*640;y=78+(n//2)*455
        d.rectangle((x+5,y,x+635,y+447),fill='white',outline='#c3c9d0',width=1)
        d.text((x+16,y+8),c['category'],font=font,fill='black')
        d.text((x+16,y+31),f"g{c['global_index']} {c['source_id']} native{c['native_slot']}  weak {c['teacher_state']} / entity {c['entity_truth']}",font=font,fill='#222222')
        board.paste(Image.fromarray(rgb).resize((288,288),Image.Resampling.NEAREST),(x+16,y+59))
        board.paste(Image.fromarray(rgb[32:64,32:64]).resize((192,192),Image.Resampling.NEAREST),(x+329,y+59))
        d.text((x+329,y+256),'original central32',font=font,fill='#555555')
        d.text((x+329,y+280),f"entity TRAIN draws: {c['entity_sample_count']}",font=font,fill='#333333')
        dist=c['nominal_reference_distance_px'];side=c['nominal_reference_distance_over_L']
        d.text((x+329,y+306),('nominal: unsupported' if dist is None else f'nominal: {dist:.1f}px / {side:.2f}L'),font=font,fill='#8c3333')
        for k,arm in enumerate(('C35','C36')):
            z=c[arm+'_logits'];text=f'{arm}  visibility {z[0]:+.3f}  entity {z[1]:+.3f}  quality {z[2]:+.3f}'
            d.text((x+16,y+358+k*27),text,font=font,fill='#183f71' if k==0 else '#8c3333')
        d.text((x+16,y+415),'All scores raw; quality negative does not certify real error.',font=font,fill='#555555')
        records.append(dict(global_index=c['global_index'],category=c['category'],source_crop_packet=p,original96=bind(out),decoded_RGB96_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),valid96_sha256=hashlib.sha256(valid.tobytes()).hexdigest(),saved_original_crop_bytes_exact=True))
    target=H/'FIXED8_SAVED_CROP_DIAGNOSIS.png';board.save(target)
    write(H/'CROP_RENDER_COMPLETE.json',dict(completed=True,PASS=True,selection=bind(selection),code=bind(__file__),board=bind(target),cases=records,RGB_inputs='only eight selected original factual96 arrays; no full RGB/GT recrop',model_feature_weight_private_reads_or_calls=0,labels_unchanged=True))
    print(json.dumps(bind(target)))
if __name__=='__main__':main()
