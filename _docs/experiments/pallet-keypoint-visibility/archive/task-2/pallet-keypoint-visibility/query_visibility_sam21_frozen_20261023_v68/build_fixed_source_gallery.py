"""Compare saved SAM2 masks on four source examples fixed before SAM2 extraction.

Read-only saved outputs and original public-source RGB/masks. No inference or fit.
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
P = ROOT.parent
SELECTION = P / 'query_visibility_controlled_real_20261022_v66/source_audit/SOURCE_VISIBLE_REGRESSION_CROPS_V3/SELECTION_AND_PIXEL_CHECK.json'
V67 = P / 'query_visibility_visible_preservation_20261023_v67/source_evaluation/VISIBLE_BCE1/PER_QUERY.csv'
OUT = ROOT / 'source_audit/FIXED_PUBLIC_VISIBLE_CROPS'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def record(path):
    return {'path': str(path), 'sha256': sha(path)}

def main():
    OUT.mkdir(parents=True, exist_ok=False)
    selection = json.loads(SELECTION.read_text())
    sam_path = ROOT / 'source_outputs/SAM21_BASE_PLUS/PER_QUERY.csv'
    targets_path = ROOT / 'source_outputs/SAM21_BASE_PLUS/TARGET_OUTPUTS.json'
    sam = {r['sample_id']: r for r in csv.DictReader(sam_path.open())}
    old = {r['sample_id']: r for r in csv.DictReader(V67.open())}
    targets = {r['global_id']: r for r in json.loads(targets_path.read_text())}
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    panel, margin, rowh = 288, 24, 420
    sheet = Image.new('RGB', (3 * panel + 4 * margin, 80 + rowh * len(selection['chosen'])), '#111827')
    draw = ImageDraw.Draw(sheet)
    draw.text((margin, 16), 'Public YCB source: four fixed visible-regression examples', fill='white', font=font)
    draw.text((margin, 42), 'Same RGB/query; target-visible mask GT is a pixel proxy, not semantic corner GT.', fill='#d1d5db', font=small)
    results = []
    for row, item in enumerate(selection['chosen']):
        sid = item['sample_id']
        pred = sam[sid]
        target = targets[pred['global_id']]
        image_path, gt_path = Path(item['RGB']['path']), Path(item['visible_mask']['path'])
        assert sha(image_path) == item['RGB']['sha256'] == target['image_sha256']
        assert sha(gt_path) == item['visible_mask']['sha256']
        assert sha(target['mask_output']['path']) == target['mask_output']['sha256']
        rgb = np.asarray(Image.open(image_path).convert('RGB'))
        gt = np.asarray(Image.open(gt_path)) > 0
        z = np.load(target['mask_output']['path'], allow_pickle=False)
        logits = z['chosen_logit']
        assert logits.dtype == np.float32 and logits.shape == gt.shape
        sm = logits > 0
        px, py = int(pred['pixel_x']), int(pred['pixel_y'])
        assert gt[py-2:py+3, px-2:px+3].all()
        assert float(logits[py, px]) == float(pred['exact_logit'])
        box = item['crop_box_native_pixels']
        x0,y0,x1,y1 = box
        assert x1-x0 == y1-y0 == 96
        patches = [rgb.copy(), rgb.copy(), rgb.copy()]
        for idx, mask, color in [(1,gt,[52,211,153]), (2,sm,[96,165,250])]:
            patches[idx][mask] = (0.58 * patches[idx][mask] + 0.42 * np.asarray(color)).astype(np.uint8)
        top = 80 + row * rowh
        for col, (arr, title) in enumerate(zip(patches, ['Original RGB + query', 'Saved target-visible GT mask', 'Frozen SAM2.1 chosen mask'])):
            left = margin + col * (panel + margin)
            draw.text((left, top), title, fill='white', font=small)
            patch = Image.fromarray(arr).crop(box).resize((panel,panel),Image.Resampling.NEAREST)
            pd = ImageDraw.Draw(patch)
            xx, yy = (px-x0)*3+1, (py-y0)*3+1
            pd.line((xx-8, yy, xx+8, yy), fill='#ff3030', width=2)
            pd.line((xx, yy-8, xx, yy+8), fill='#ff3030', width=2)
            sheet.paste(patch,(left,top+26))
        text_y = top + 322
        draw.text((margin,text_y),sid,fill='white',font=small)
        vals = {
            'sample_id':sid, 'crop_box_native':box, 'pixel':[px,py],
            'GT_strict25_visible_verified':True,
            'raw_MobileSAM_pV':float(item['v66']['rawSAM_exact_pV']),
            'V65_pV':float(item['v65']['exact_pV']),
            'V66_pV':float(item['v66']['exact_pV']),
            'V67_pV':float(old[sid]['exact_pV']),
            'SAM21_exact_logit':float(logits[py,px]),
            'SAM21_exact_visible':bool(sm[py,px]),
            'SAM21_radius2_logit':float(pred['radius2_logit']),
            'selected_mask_index':int(pred['selected_mask_index']),
            'RGB':record(image_path), 'target_visible_mask':record(gt_path),
            'saved_SAM21_mask':target['mask_output'],
        }
        draw.text((margin,text_y+24),
            'raw MobileSAM pV %.4f | V65 %.4f | V66 %.4f | V67 %.4f' %
            tuple(vals[k] for k in ['raw_MobileSAM_pV','V65_pV','V66_pV','V67_pV']),
            fill='#d1d5db',font=small)
        draw.text((margin,text_y+48),
            f"SAM2.1 logit {vals['SAM21_exact_logit']:.4f} -> {'VISIBLE' if vals['SAM21_exact_visible'] else 'ABSENT'}; mask branch {vals['selected_mask_index']} chosen by predicted quality",
            fill='#93c5fd',font=small)
        results.append(vals)
    png = OUT / 'FIXED_PUBLIC_VISIBLE_CROPS.png'
    sheet.save(png)
    report = {'schema':'V68_FIXED_SOURCE_CROP_REVIEW_V1','selection_unchanged':True,
        'selection_preceded_SAM21_extraction':True,'selection':record(SELECTION),
        'saved_query_outputs':record(sam_path),'saved_target_outputs':record(targets_path),
        'saved_V67_outputs':record(V67),'examples':results,'image':record(png),
        'new_model_forward':0,'training':0,'current223_read':False,'PnP':0,
        'interpretation':'Four preselected visible pixel-support examples; not a representative metric or semantic reference-corner proof.'}
    (OUT/'PIXEL_AND_PROVENANCE_CHECK.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'completed':True,'image':str(png),'SAM21_visible':[r['SAM21_exact_visible'] for r in results]}))

if __name__ == '__main__':
    main()
