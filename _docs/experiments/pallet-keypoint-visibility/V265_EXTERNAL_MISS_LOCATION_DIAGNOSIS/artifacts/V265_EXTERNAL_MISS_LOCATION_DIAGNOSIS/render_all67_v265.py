#!/usr/bin/env python3
"""Render every frozen V264 external miss in original index order. Display only."""
from pathlib import Path
import csv, hashlib, html, io, json, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'all67_gallery_v1'

def guard(path):
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if any(s in str(p).lower() for s in ('v112', 'cache', '.pyc')) or '..' in p.parts:
        raise ValueError('Rejected lexical path')
    for a in [*reversed(p.parents), p]:
        if a.is_symlink():
            raise ValueError('Rejected symlink ancestor')
    return p

def blob(p):
    return guard(p).read_bytes()

def sha(data):
    return hashlib.sha256(data).hexdigest()

def binding(p):
    p = guard(p)
    data = p.read_bytes()
    return {'path': str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p), 'sha256': sha(data), 'size_bytes': len(data)}

def verify(b, portable_relative=None):
    p = guard(ROOT / portable_relative) if portable_relative else guard(b['path'])
    data = p.read_bytes()
    if len(data) != b['size_bytes'] or sha(data) != b['sha256']:
        raise ValueError('Input binding mismatch: ' + str(p))
    return data

def rows(data):
    return list(csv.DictReader(io.StringIO(data.decode('utf-8-sig'))))

policy = json.loads(blob(ROOT / 'SEALED_INPUTS.json'))
ledger_data = verify(policy['inputs']['analysis_inputs/LEDGER.csv'], 'analysis_inputs/LEDGER.csv')
geometry_data = verify(policy['inputs']['analysis_inputs/FROZEN_GEOMETRY.npz'], 'analysis_inputs/FROZEN_GEOMETRY.npz')
miss_data = blob(ROOT / 'diagnosis_v1/MISSED_EXTERNAL_67.csv')
misses = rows(miss_data)
ids = [int(r['global_index']) for r in misses]
assert len(ids) == 67 and ids == sorted(set(ids))
assert all(r['LOCAL_prediction'] == 'D' for r in misses)
ledger = {int(r['global_index']): r for r in rows(ledger_data)}
assert all(ledger[i]['human_state'] == 'EXTERNAL_OCCLUDED' for i in ids)
OUT = guard(OUT)
OUT.mkdir(exist_ok=True)
compact = OUT / 'ALL67_CROPS.npz'
previous_complete = OUT / 'COMPLETE.json'
raw_read = False
if compact.exists() and previous_complete.exists():
    old = json.loads(blob(previous_complete))
    cb = old['outputs']['ALL67_CROPS.npz']
    compact_data = blob(compact)
    assert sha(compact_data) == cb['sha256'] and len(compact_data) == cb['size_bytes']
    with np.load(io.BytesIO(compact_data), allow_pickle=False) as z:
        assert z['global_indices'].tolist() == ids
        rgb67 = z['RGB96_HWC'].copy()
else:
    raw_data = verify(policy['raw_crop_source'])
    raw_read = True
    with np.load(io.BytesIO(raw_data), allow_pickle=False) as z:
        rgb67 = z['RGB96_initial_geometry'][ids].transpose(0, 2, 3, 1).copy()
    np.savez_compressed(compact, RGB96_HWC=rgb67, global_indices=np.asarray(ids, dtype=np.int64))
assert rgb67.shape == (67, 96, 96, 3) and rgb67.dtype == np.uint8
with np.load(io.BytesIO(geometry_data), allow_pickle=False) as z:
    probability = z['probability'][ids].copy()
    assert probability.shape == (67, 16, 16)

font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
f15 = ImageFont.truetype(font_path, 15)
f17 = ImageFont.truetype(font_path, 17)
f14 = ImageFont.truetype(font_path, 14)
# Viridis is a display lookup table. Its brightness is not visibility probability.
from matplotlib import colormaps
lut = (colormaps['viridis'](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
markers = {'original_query': 'cyan circle', 'inherited_E_reference': 'magenta cross if inside original crop', 'frozen_JOINT_proposal': 'yellow cross'}
category_text = {'REFERENCE_OUTSIDE_CROP': 'REF OUTSIDE ORIGINAL CROP', 'INSIDE_REF_NEAR_LE3': 'INSIDE / PROPOSAL-REF <=3px', 'INSIDE_REF_BORDER_3_TO6': 'INSIDE / PROPOSAL-REF 3-6px', 'INSIDE_REF_FAR_GT6': 'INSIDE / PROPOSAL-REF >6px'}
records = []
tiles = []
for ordinal, (index, mr, rgb, prob) in enumerate(zip(ids, misses, rgb67, probability), start=1):
    r = ledger[index]
    assert r['row_id'] == mr['row_id']
    q = np.array([float(r['query_x']), float(r['query_y'])])
    ref = np.array([float(r['human_reference_x']), float(r['human_reference_y'])])
    proposed = np.array(json.loads(r['JOINT_SELECTED__LOC_proposed_xy']))
    side = float(r['JOINT_SELECTED__crop_L'])
    assert abs(np.linalg.norm(proposed - ref) - float(mr['proposal_reference_distance_px'])) < 1e-8
    clean = Image.fromarray(rgb).resize((250, 250), Image.Resampling.NEAREST)
    pmin, pmax = float(prob.min()), float(prob.max())
    scaled = (prob - pmin) / max(pmax - pmin, 1e-12)
    scaled250 = np.asarray(Image.fromarray(scaled).resize((250, 250), Image.Resampling.BILINEAR))
    colors = lut[(scaled250.clip(0, 1) * 255).astype(np.uint8)]
    overlay = Image.fromarray((0.62 * np.asarray(clean) + 0.38 * colors).astype(np.uint8))
    tile = Image.new('RGB', (600, 430), '#ffffff')
    tile.paste(clean, (15, 88)); tile.paste(overlay, (315, 88))
    d = ImageDraw.Draw(tile)
    d.text((12, 7), f'[{ordinal:02d}/67] G{index}  {r["row_id"]}', font=f17, fill='#142739')
    d.text((12, 32), f'{mr["group"]} | E reference: {mr["coordinate_source"]}', font=f15, fill='#33475c')
    d.text((15, 62), 'Frozen RGB96 (original q crop)', font=f14, fill='#33475c')
    d.text((315, 62), 'Frozen heatmap / min-max display', font=f14, fill='#33475c')
    # Markers are added only to the display; clean RGB pixels above remain unmodified.
    for point, color, style in [(q, '#00ffff', 'o'), (ref, '#ff00dc', '+'), (proposed, '#ffff00', '+')]:
        uv = (point - q) / side + 0.5
        if np.all(uv >= 0) and np.all(uv <= 1):
            x, y = 315 + uv[0] * 250 - 0.5, 88 + uv[1] * 250 - 0.5
            if style == 'o':
                d.ellipse((x - 4, y - 4, x + 4, y + 4), outline=color, width=2)
            else:
                d.line((x - 6, y, x + 6, y), fill=color, width=2)
                d.line((x, y - 6, x, y + 6), fill=color, width=2)
    initial_dist = float(mr['initial_reference_distance_px'])
    proposed_dist = float(mr['proposal_reference_distance_px'])
    d.text((12, 350), f'Initial-ref {initial_dist:.2f}px | Proposal-ref {proposed_dist:.2f}px', font=f15, fill='#142739')
    d.text((12, 375), category_text[mr['category']], font=f15, fill='#b32222' if mr['reference_inside_crop'] == 'False' else '#142739')
    extra = ' | USER-DISPUTED E LABEL' if mr['disputed_annotation'] == 'True' else ''
    d.text((12, 400), f'GLOBAL {mr["GLOBAL_prediction"]} | LOCAL D | p(E)={float(mr["LOCAL_p_E"]):.4f}' + extra, font=f14, fill='#7c2350' if extra else '#33475c')
    d.rectangle((0, 0, 599, 429), outline='#aab6c2', width=1)
    tiles.append(tile)
    records.append({'ordinal': ordinal, 'global_index': index, 'row_id': r['row_id'], 'RGB96_HWC_sha256': sha(rgb.tobytes()), 'sheet': f'CONTACT_{(ordinal - 1) // 12 + 1:02d}.png', 'tile_zero_based': (ordinal - 1) % 12, 'reference_coordinate_authority': mr['coordinate_source'], 'reference_inside_original_crop': mr['reference_inside_crop'] == 'True', 'initial_reference_distance_px': initial_dist, 'proposal_reference_distance_px': proposed_dist, 'category': mr['category'], 'user_disputed_annotation': mr['disputed_annotation'] == 'True'})
for page in range(6):
    sheet = Image.new('RGB', (1200, 2580), '#e9eef3')
    for k, tile in enumerate(tiles[page * 12:(page + 1) * 12]):
        sheet.paste(tile, ((k % 2) * 600, (k // 2) * 430))
    guard(OUT / f'CONTACT_{page + 1:02d}.png'); sheet.save(OUT / f'CONTACT_{page + 1:02d}.png')
(OUT / 'MISSED_EXTERNAL_67.csv').write_bytes(miss_data)
(OUT / 'PIXEL_SHA256.json').write_text(json.dumps({'RGB_layout': 'HWC uint8 RGB96', 'ordering': 'all 67 original global indices sorted, no visual selection', 'source_member': 'RGB96_initial_geometry', 'markers': markers, 'reference_caution': 'Inherited occluded-point references are not direct visible-location gold. Do not infer raw relabels from this gallery.', 'cases': records}, ensure_ascii=False, indent=2) + '\n')
sheets_html = ''.join(f'<section id="sheet{i}"><h2>{i}/6 · cases {(i-1)*12+1}–{min(i*12,67)}</h2><a href="CONTACT_{i:02d}.png"><img src="CONTACT_{i:02d}.png" alt="all67 contact sheet {i}"></a></section>' for i in range(1,7))
index = '''<!doctype html><html lang="ko"><meta charset="utf-8"><title>V265 외부 가림 누락 67점 전체 crop</title><style>body{font:16px/1.65 sans-serif;max-width:1250px;margin:28px auto;padding:18px;color:#172b3a}aside{background:#fff0d7;padding:16px}img{width:100%;height:auto}nav{position:sticky;top:0;background:#fff;padding:10px}a{color:#145eac}</style><h1>V265 · 외부 가림 누락 67점 전체 crop</h1><aside>기존 E 주석을 LOCAL이 D로 판단한 67점을 원본 global_index 순서로 모두 표시합니다. 새 학습·추론·PnP·주석 변경·GT 중심 재crop은 없습니다. 가려진 점의 기존 참조 좌표와 가깝다는 것은 정확한 직접 보임 위치를 입증하지 않습니다. G996은 사용자와 기존 주석 해석이 다른 사례입니다.</aside><p>왼쪽: 실제 동결 RGB96 입력. 오른쪽: 동결 V263 JOINT heatmap의 crop별 min/max 표시. 청록 ○ 초기 query · 자홍 + 기존 E 참조 · 노랑 + 위치 후보. 참조가 원래 crop 밖이면 자홍 점을 강제로 잘라 표시하지 않고 OUTSIDE로 적었습니다. 색 밝기는 보임 확률이 아닙니다.</p><p>분류 category는 기존 위치 거리 구간이며 육안 원인 정답이 아닙니다. 부분 노출과 native 대응의 모호함을 자동으로 새 라벨로 바꾸지 않았습니다.</p><nav>''' + ' · '.join(f'<a href="#sheet{i}">{i}/6</a>' for i in range(1,7)) + ''' · <a href="MISSED_EXTERNAL_67.csv">67점 수치 CSV</a> · <a href="PIXEL_SHA256.json">픽셀 SHA·대응표</a> · <a href="COMPLETE.json">재현·입력 바인딩</a></nav>''' + sheets_html + '</html>'
(OUT / 'INDEX_KO.html').write_text(index)
outputs = {name: binding(OUT / name) for name in ['ALL67_CROPS.npz', 'MISSED_EXTERNAL_67.csv', 'PIXEL_SHA256.json', 'INDEX_KO.html', *[f'CONTACT_{i:02d}.png' for i in range(1, 7)]]}
complete = {'completed': True, 'PASS': True, 'cases': 67, 'contact_sheets': 6, 'cases_per_sheet': 12, 'order': ids, 'code': binding(Path(__file__)), 'input_bindings': {'SEALED_INPUTS.json': binding(ROOT / 'SEALED_INPUTS.json'), 'MISSED_EXTERNAL_67.csv': binding(ROOT / 'diagnosis_v1/MISSED_EXTERNAL_67.csv'), 'LEDGER.csv': binding(ROOT / 'analysis_inputs/LEDGER.csv'), 'FROZEN_GEOMETRY.npz': binding(ROOT / 'analysis_inputs/FROZEN_GEOMETRY.npz'), 'raw_crop_source_historical_binding': policy['raw_crop_source']}, 'raw_full_crop_source_read_this_run': raw_read, 'portable_replay': 'On rerun, verify existing compact ALL67_CROPS.npz against COMPLETE.json and render with local frozen geometry+ledger. No model required.', 'markers': markers, 'heatmap_display': 'per-crop min-max probability display, not calibrated visibility', 'new_training_model_forward_PnP_relabels_GT_recrops': 0, 'outputs': outputs}
(OUT / 'COMPLETE.json').write_text(json.dumps(complete, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'completed': True, 'cases': 67, 'sheets': 6, 'compact': binding(compact), 'complete': binding(OUT / 'COMPLETE.json')}, ensure_ascii=False))
