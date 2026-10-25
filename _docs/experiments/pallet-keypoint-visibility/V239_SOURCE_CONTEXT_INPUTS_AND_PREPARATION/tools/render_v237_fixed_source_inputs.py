"""Predeclared SOURCE input display; no model or reference/pose inputs."""
from pathlib import Path
import hashlib
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

R = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation')
H = R / 'V237_ROOT_SOURCE_CONTEXT_INPUTS'
OUT = H / 'fixed4_original_input_gallery'


def safe(p):
    p = Path(p)
    assert 'v112' not in str(p).lower() and '__pycache__' not in str(p) and p.suffix != '.pyc'
    assert not p.is_symlink()
    return p


def sha(p):
    return hashlib.sha256(safe(p).read_bytes()).hexdigest()


def write(p, value):
    with safe(p).open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False); f.write('\n')


def main():
    assert not OUT.exists()
    OUT.mkdir()
    groups_path = R / 'V237_SOURCE_INPUT_MANIFEST_C/GROUPS_INPUT_ONLY.jsonl'
    rows_path = R / 'V237_SOURCE_INPUT_MANIFEST_C/ALL15040_GROUP_JOIN.jsonl'
    assert sha(groups_path) == 'adcf139a49d5acb672c0a1a4f8505c1b92b2b01870c6c177aac33356a020ea01'
    assert sha(rows_path) == '82d4ca430e038e6567538e7dc373069f621913a480fc3a69c17d32634a1e12c6'
    groups = [json.loads(x) for x in safe(groups_path).read_text().splitlines()]
    rows = [json.loads(x) for x in safe(rows_path).read_text().splitlines()]
    cases = []
    for family in (0, 1):
        for target in (0, 1):
            pool = [r for r in rows if r['sample_index'] is not None and r['family_code'] == family and r['target'] == target]
            cases.append(min(pool, key=lambda r: r['sample_index']))
    policy = {'selection': 'Minimum original sample_index in each SOURCE family x D/E cell, before new features/head predictions; no substitution.',
              'sample_indices': [r['sample_index'] for r in cases], 'rows': cases,
              'original_query_q_L_and_U_unchanged': True, 'GT_reference_or_pose_displayed': False,
              'limits': 'SOURCE labels are display headings only, not input features. YCB q8 is existing GT-jitter emulation/nativeUNKNOWN. Display rounds context F32 to uint8; model does not.'}
    write(OUT / 'CASE_POLICY_BEFORE_PIXEL_READ.json', policy)
    data = H / 'original_source_inputs_v1'
    complete = json.loads(safe(data / 'COMPLETE.json').read_text())
    assert complete['PASS'] and complete['known7617_original_RGB96_and_valid_SHA_exact'] == 7617
    nodes = np.load(safe(data / 'NODE_RGB96.npy'), mmap_mode='r', allow_pickle=False)
    contexts = np.load(safe(data / 'CONTEXT_RGB224_F32.npy'), mmap_mode='r', allow_pickle=False)
    with np.load(safe(data / 'INPUT_GEOMETRY.npz'), allow_pickle=False) as z:
        node_available = z['node_available'].copy()
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    normal = ImageFont.truetype(font_path, 15)
    small = ImageFont.truetype(font_path, 12)
    title = ImageFont.truetype(font_path, 22)
    board = Image.new('RGB', (1770, 1100), '#edf1f5')
    draw = ImageDraw.Draw(board)
    draw.text((20, 15), 'V237: original SOURCE target context + all eight original local query crops', font=title, fill='black')
    draw.text((20, 48), 'Orange = active q. Gates use input geometry only. Existing S/U labels never remove a supported context node.', font=normal, fill='black')
    records = []
    for ci, row in enumerate(cases):
        gi, active = row['group_index'], row['active_slot']; g = groups[gi]
        y0 = 92 + ci * 247
        heading = f"{'SYN MAIN' if row['family_code'] == 0 else 'YCB AUX/native UNKNOWN'} | {'D visible' if row['target'] == 0 else 'E external'} | {'FIT' if row['fit'] else 'DEV'} | sample {row['sample_index']} | active {active}"
        draw.text((20, y0), heading, font=normal, fill='black')
        context = np.floor(np.clip(np.array(contexts[gi]), 0, 255) + .5).astype(np.uint8).transpose(1, 2, 0)
        im = Image.fromarray(context); imdraw = ImageDraw.Draw(im)
        box = np.asarray(g['box_xyxy'], np.float64); q8 = np.asarray(g['q8'], np.float64)
        uv = (q8 - box[:2]) / (box[2:] - box[:2]) * 224 - .5
        for k, (x, y) in enumerate(uv):
            x, y = float(np.clip(x, 3, 220)), float(np.clip(y, 3, 220))
            color = '#ff8700' if k == active else '#00ffdd'
            imdraw.ellipse((x-3, y-3, x+3, y+3), outline=color, width=2)
            imdraw.text((x+3, y+2), str(k), font=small, fill=color)
        board.paste(im.resize((208, 208)), (20, y0 + 27))
        local_records = []
        for k in range(8):
            local = np.array(nodes[gi, k]).transpose(1, 2, 0)
            raw_path = OUT / f'CASE{ci}_NODE{k}_RAW96.png'
            Image.fromarray(local).save(safe(raw_path))
            tile = Image.fromarray(local).resize((174, 174), Image.Resampling.NEAREST)
            td = ImageDraw.Draw(tile); td.ellipse((84, 84, 90, 90), outline='#ff8700', width=2)
            x = 252 + k * 187; yy = y0 + 38
            board.paste(tile, (x, yy))
            draw.rectangle((x-2, yy-2, x+175, yy+175), outline='#ff8700' if k == active else '#64748b', width=3 if k == active else 1)
            draw.text((x, y0 + 27), f'q{k} gate={int(node_available[gi,k])}', font=small, fill='black')
            local_records.append({'slot': k, 'input_gate': bool(node_available[gi,k]), 'clean_png': raw_path.name, 'sha256': sha(raw_path),
                                  'raw_CHW_sha256': hashlib.sha256(local.transpose(2, 0, 1).tobytes()).hexdigest()})
        records.append({'sample_index': row['sample_index'], 'group_index': gi, 'active_slot': active, 'local_pngs': local_records})
    image_path = OUT / 'FIXED4_SOURCE_INPUT_CONTEXT_AND_Q8.png'
    board.save(safe(image_path))
    write(OUT / 'COMPLETE.json', {'completed': True, 'PASS': True, 'code_sha256': sha(__file__), 'input_complete_sha256': sha(data / 'COMPLETE.json'),
                                 'case_policy_sha256': sha(OUT / 'CASE_POLICY_BEFORE_PIXEL_READ.json'), 'cases': records,
                                 'board': {'path': image_path.name, 'sha256': sha(image_path)}, 'source_only': True, 'models_run': 0, 'new_current_or_held_payload': 0})
    print(json.dumps({'PASS': True, 'image': str(image_path), 'sample_indices': policy['sample_indices'], 'board_sha256': sha(image_path)}))


if __name__ == '__main__':
    main()
