"""Render first successful FIT pair per family before classifier training."""
from pathlib import Path
import json
import hashlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

H = Path(__file__).parent
D = H / 'pairs_v1'
OUT = H / 'fixed2_prefit_gallery_v2'


def main():
    complete = json.loads((D / 'COMPLETE.json').read_text())
    assert hashlib.sha256((D / 'COMPLETE.json').read_bytes()).hexdigest() == '8958c47fc5cf65d592346d5b9671c592ad470e799cfdcdb926e6e2788ca2fc13'
    meta = [json.loads(line) for line in (D / 'PAIR_METADATA.jsonl').read_text().splitlines()]
    cases = [next(m for m in meta if m['family'] == family and m['success']) for family in (0, 1)]
    OUT.mkdir(exist_ok=False)
    policy = {'rule': 'Minimum successful pair_index per family; before new head fit/output; no score selection', 'cases': [{'pair_index': m['pair_index'], 'parent_sample_index': m['parent_sample_index'], 'row_id': m['row_id']} for m in cases]}
    (OUT / 'FIXED_CASE_IDS_BEFORE_RENDER.json').write_text(json.dumps(policy, indent=2) + '\n')
    for p in (complete['source_bindings']['rgb96']['path'], D / 'PAIR_RGB96.npy', D / 'PAIR_ALPHA_PACKBITS.npy'):
        assert 'v112' not in str(p).lower() and '__pycache__' not in Path(p).parts
    raw = np.load(complete['source_bindings']['rgb96']['path'], mmap_mode='r', allow_pickle=False)
    images = np.load(D / 'PAIR_RGB96.npy', mmap_mode='r', allow_pickle=False)
    masks = np.load(D / 'PAIR_ALPHA_PACKBITS.npy', mmap_mode='r', allow_pickle=False)
    board = Image.new('RGB', (1560, 1020), '#101827')
    draw = ImageDraw.Draw(board)
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font = ImageFont.truetype(font_path, 20)
    small = ImageFont.truetype(font_path, 16)
    title = ImageFont.truetype(font_path, 27)
    draw.text((20, 15), 'V235 same-q SOURCE FIT counterfactuals: pre-training inspection', font=title, fill='white')
    draw.text((20, 55), 'Cyan = original query; magenta = teacher reference, never a recrop/forward channel.', font=font, fill='#a7d9fa')
    for r, m in enumerate(cases):
        j, i = m['pair_index'], m['parent_sample_index']
        y = 110 + r * 410
        family = 'SYN intended-native witness0' if m['family'] == 0 else 'YCB associated-r surface'
        draw.text((20, y), f'{family} | parent {i} | donor {m["donor_id"]}', font=font, fill='white')
        base = np.array(raw[i])
        assert hashlib.sha256(base.tobytes()).hexdigest() == m['original_pixel_sha256']
        ref96 = (np.asarray(m['teacher_reference_xy']) - np.asarray(m['original_query_xy'])) * 96 / m['original_L'] + 47.5
        views = [base, np.array(images[j, 0]), np.array(images[j, 1])]
        names = ['BEFORE: D (visible)', 'COVER: E (generated)', 'SHAM: D (generated)']
        for c, (pixels, name) in enumerate(zip(views, names)):
            x, py = 20 + c * 385, y + 70
            image = Image.fromarray(pixels.transpose(1, 2, 0))
            image.save(OUT / f'CASE_{r}_{c}_RAW96.png')
            shown = image.resize((288, 288), Image.Resampling.NEAREST)
            overlay = ImageDraw.Draw(shown)
            overlay.line((131.5, 143.5, 155.5, 143.5), fill='#00ffff', width=2)
            overlay.line((143.5, 131.5, 143.5, 155.5), fill='#00ffff', width=2)
            rx, ry = ref96 * 3 + 1
            overlay.ellipse((rx - 7, ry - 7, rx + 7, ry + 7), outline='#ff42df', width=2)
            board.paste(shown, (x, py))
            draw.text((x, y + 38), name, font=font, fill='white')
        alpha = np.unpackbits(np.array(masks[j]), axis=1).reshape(2, 96, 96)
        mask_view = np.zeros((192, 96, 3), np.uint8)
        mask_view[:96, :, 0] = alpha[0] * 255
        mask_view[96:, :, 2] = alpha[1] * 255
        for x, z in np.asarray(m['teacherchecks']['required_taps_xy']).reshape(-1, 2):
            mask_view[z, x] = [0, 255, 0]
            mask_view[96 + z, x] = [0, 255, 0]
        board.paste(Image.fromarray(mask_view).resize((144, 288), Image.Resampling.NEAREST), (1190, y + 70))
        draw.text((1165, y + 38), 'COVER / SHAM alpha', font=small, fill='white')
        draw.text((1165, y + 365), 'Green: required reference support', font=small, fill='#7ef0a4')
    draw.text((20, 955), 'Successful pairs: SYN 4537/4565, YCB 830/841. 39 failed constructions remain masked; originals are retained.', font=small, fill='#cad5e7')
    draw.text((20, 983), 'Public-real cutout composites are synthetic training examples. Natural SOURCE DEV and current223 are not donors.', font=small, fill='#ffc98b')
    board.save(OUT / 'PREFIT_COUNTERFACTUALS.png')
    (OUT / 'CASE_DATA.json').write_text(json.dumps(cases, ensure_ascii=False, indent=2) + '\n')
    (OUT / 'COMPLETE.json').write_text(json.dumps({'PASS': True, 'cases': policy['cases'], 'new_fit_or_model_forward': 0, 'gallery_v1_failure': '1aacdb exit1: required_taps_xy is [9,4,2], initial inline renderer expected[36,2]. Source dataset unchanged; v2 reshapes for display only.'}, indent=2) + '\n')
    print(json.dumps({'PASS': True, 'gallery': str(OUT / 'PREFIT_COUNTERFACTUALS.png'), 'parents': [m['parent_sample_index'] for m in cases]}))


if __name__ == '__main__':
    main()
