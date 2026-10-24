#!/usr/bin/env python3
"""Read frozen predictions and source RGB; write reproducibly selected diagnostics only."""
from pathlib import Path
import csv, hashlib, json, math
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent.parent / 'evaluation_preparation_20261022/PREPARED_QUERY_MANIFEST.csv'
ROWS = list(csv.DictReader(SOURCE.open()))
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
CYAN, MAGENTA = '#00ddff', '#ff31bd'
F = {s: ImageFont.truetype(FONT, s) for s in [15, 17, 19, 22, 24]}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def neg(r): return r['present_at_fixed_0_5'] == 'False'
def pos(r): return not neg(r)
def lp(r): return r['legacy_stored_state'] in ['B', 'I']
def ln(r): return r['legacy_stored_state'] == 'O'
def err(r): return float(r['initial_reference_error_px'])
def norm(r): return err(r) / float(r['v61_roi_side_raw_px'])

def choose(group, state, predicate, order, name, rule):
    pool = [r for r in ROWS if r['same_slot_pipeline_D_vs_ES'] == 'True'
            and r['group'] == group and r['human_state'] == state and predicate(r)]
    assert pool, name
    pool.sort(key=lambda r: (order(r), r['id'], int(r['native_corner'])))
    return name, rule, len(pool), pool[0]

cases = []
for state, tag in [('DIRECT_VISIBLE', 'D'), ('EXTERNAL_OCCLUDED', 'E'), ('SELF_OCCLUDED', 'S')]:
    cases.append(choose('WOOD_PRIMARY105', state, lambda r: neg(r) and lp(r),
                        lambda r: -norm(r), f'wood_{tag}_max_offset_v61neg_legacypos',
                        f'WOOD {tag}, v61 negative / legacy B or I: maximum error / saved ROI side'))
for state, tag in [('DIRECT_VISIBLE', 'D'), ('EXTERNAL_OCCLUDED', 'E')]:
    cases.append(choose('SQUARE118', state, lambda r: pos(r) and ln(r),
                        lambda r: -norm(r), f'square_{tag}_max_offset_v61pos_legacyneg',
                        f'SQUARE {tag}, v61 positive / legacy O: maximum error / saved ROI side'))
for group, tag in [('WOOD_PRIMARY105', 'wood'), ('SQUARE118', 'square')]:
    cases.append(choose(group, 'DIRECT_VISIBLE', lambda r: neg(r) and lp(r), err,
                        f'{tag}_D_min_error_control',
                        f'{tag.upper()} D, v61 negative / legacy B or I: minimum absolute error (small-error control)'))
cases.append(choose('WOOD_PRIMARY105', 'SELF_OCCLUDED', lambda r: pos(r) and r['legacy_stored_state'] == 'S',
                    lambda r: -norm(r), 'wood_S_geometry_max_offset_v61pos',
                    'WOOD S, v61 positive / legacy geometry S: maximum error / saved ROI side'))

def marker(draw, xy, color, kind, size=8, width=3):
    x, y = xy
    if kind == '+':
        draw.line([(x-size, y), (x+size, y)], fill='black', width=width+2)
        draw.line([(x, y-size), (x, y+size)], fill='black', width=width+2)
        draw.line([(x-size, y), (x+size, y)], fill=color, width=width)
        draw.line([(x, y-size), (x, y+size)], fill=color, width=width)
    else:
        draw.line([(x-size, y-size), (x+size, y+size)], fill='black', width=width+2)
        draw.line([(x-size, y+size), (x+size, y-size)], fill='black', width=width+2)
        draw.line([(x-size, y-size), (x+size, y+size)], fill=color, width=width)
        draw.line([(x-size, y+size), (x+size, y-size)], fill=color, width=width)

def text(draw, xy, txt, size=17, fill='#e9eff8'):
    draw.text(xy, txt, font=F[size], fill=fill)

manifest = {
    'source_manifest': str(SOURCE), 'source_manifest_sha256': sha(SOURCE),
    'selection_scope': 'Frozen same_slot_pipeline_D_vs_ES eligible rows only; tie break: id then numeric native_corner.',
    'selection_warning': 'Disagreement-directed diagnostics, not a prevalence-representative sample. No new pixel labels or causal estimate.',
    'crop_rule': 'Exact nominal saved v61 ROI is [query_x-side/2, query_y-side/2, query_x+side/2, query_y+side/2]. Raw diagnostic raster is its floor/ceil integer enclosing box; out-of-image pixels use black. Raw saved raster is unmarked and unscaled. No GT recentering or resizing.',
    'display_warning': 'These are raw RGB diagnostic crops; v61 used saved P3 ROI features, not these raster files. Legacy RGB input/crop differs. No new inference, training, PnP, threshold selection, or reference-based crop adaptation.',
    'marker_legend': {'cyan_plus': 'frozen initial query', 'magenta_x': 'mapped semantic reference (evaluation-only)'},
    'cases': []
}
panels = []
for index, (name, rule, n_pool, row) in enumerate(cases, 1):
    stem = f'{index:02d}_{name}'
    src = Path(row['image'])
    assert sha(src) == row['image_sha256'], src
    image = Image.open(src).convert('RGB')
    assert image.size == (int(row['raw_width']), int(row['raw_height']))
    q = [float(row['query_x']), float(row['query_y'])]
    ref = [float(row['human_reference_x']), float(row['human_reference_y'])]
    side = float(row['v61_roi_side_raw_px'])
    bbox = [q[0]-side/2, q[1]-side/2, q[0]+side/2, q[1]+side/2]
    ibox = [math.floor(bbox[0]), math.floor(bbox[1]), math.ceil(bbox[2]), math.ceil(bbox[3])]
    raw_crop = image.crop(ibox)
    raw_path = ROOT / f'{stem}_raw_crop.png'
    raw_crop.save(raw_path)

    panel = Image.new('RGB', (1400, 530), '#131c2b')
    d = ImageDraw.Draw(panel)
    state = {'DIRECT_VISIBLE': 'D visible', 'EXTERNAL_OCCLUDED': 'E external', 'SELF_OCCLUDED': 'S self'}[row['human_state']]
    text(d, (20, 12), f'{index:02d}  {row["id"]}  k{row["native_corner"]} / ref k{row["reference_corner_id"]}   Human: {state}', 22)
    p = float(row['presence_probability'])
    lp_txt = 'no RGB score (geometry gate)' if not row['legacy_p_DIRECT'] else f'p_DIRECT={float(row["legacy_p_DIRECT"]):.4f}'
    text(d, (20, 45), f'v61 p_present={p:.4f} ({"+" if p >= .5 else "-"})     Legacy {row["legacy_stored_state"]}: {lp_txt}', 19)
    text(d, (20, 74), f'Offset={err(row):.2f}px = {norm(row):.3f} x saved ROI side ({side:.2f}px); reference in nominal crop: {row["human_reference_inside_nominal_crop"]}', 17)
    text(d, (20, 104), 'Full frame + saved nominal ROI', 17)
    text(d, (670, 104), 'Raw RGB crop (unmarked)', 17)
    text(d, (1040, 104), 'Same crop + query/reference', 17)

    # Full context retains the complete frame. Scale is set solely by raw dimensions.
    scale = min(620/image.width, 320/image.height)
    sz = (round(image.width*scale), round(image.height*scale))
    canvas = image.resize(sz, Image.Resampling.BICUBIC)
    cd = ImageDraw.Draw(canvas)
    cd.rectangle([v*scale for v in bbox], outline=CYAN, width=2)
    cd.line([(q[0]*scale,q[1]*scale),(ref[0]*scale,ref[1]*scale)], fill='#ffe17b', width=1)
    marker(cd, [q[0]*scale,q[1]*scale], CYAN, '+', 7, 2)
    marker(cd, [ref[0]*scale,ref[1]*scale], MAGENTA, 'x', 7, 2)
    context_path = ROOT / f'{stem}_context.png'
    canvas.save(context_path)
    ox, oy = 20+(620-sz[0])//2, 138+(320-sz[1])//2
    panel.paste(canvas, (ox,oy))

    # Display geometry uses only the saved query box, never the reference.
    raw_display = raw_crop.resize((300,300), Image.Resampling.NEAREST)
    panel.paste(raw_display, (670,138))
    marked = raw_display.copy()
    md = ImageDraw.Draw(marked)
    fx, fy = 300/raw_crop.width, 300/raw_crop.height
    qd = ((q[0]-ibox[0])*fx, (q[1]-ibox[1])*fy)
    rd = ((ref[0]-ibox[0])*fx, (ref[1]-ibox[1])*fy)
    marker(md, qd, CYAN, '+', 10, 3)
    ref_in_display = 0 <= rd[0] < 300 and 0 <= rd[1] < 300
    if ref_in_display: marker(md, rd, MAGENTA, 'x', 10, 3)
    marked_path = ROOT / f'{stem}_marked_crop.png'
    marked.save(marked_path)
    panel.paste(marked, (1040,138))
    text(d, (1040,448), 'Reference outside crop' if not ref_in_display else 'Reference shown as magenta x', 17, MAGENTA)
    text(d, (20,474), rule, 15)
    text(d, (20,501), 'Cyan + = initial query; magenta x = mapped semantic reference. No real single-pixel presence GT.', 15)
    panel_path = ROOT / f'{stem}_panel.png'
    panel.save(panel_path)
    panels.append(panel)
    manifest['cases'].append({
        'case_number': index, 'selection_name': name, 'selection_rule': rule, 'selection_pool_n': n_pool,
        'id': row['id'], 'group': row['group'], 'native_corner': int(row['native_corner']),
        'reference_corner_id': int(row['reference_corner_id']), 'human_state': row['human_state'],
        'source_image': str(src), 'source_image_sha256': sha(src), 'source_size_wh': image.size,
        'query_xy': q, 'reference_xy': ref, 'query_nearest_pixel_floor_plus_half': [math.floor(t+.5) for t in q],
        'saved_v61_roi_side_raw_px': side, 'exact_nominal_crop_xyxy': bbox,
        'diagnostic_integer_enclosing_crop_xyxy': ibox,
        'raw_crop_size_wh': raw_crop.size, 'reference_in_nominal_crop': row['human_reference_inside_nominal_crop'] == 'True',
        'initial_reference_error_px': err(row), 'error_over_saved_roi_side': norm(row),
        'v61_p_present': p, 'v61_positive': p >= .5,
        'legacy_state': row['legacy_stored_state'], 'legacy_p_DIRECT': float(row['legacy_p_DIRECT']) if row['legacy_p_DIRECT'] else None,
        'legacy_RGB_evaluated': row['legacy_RGB_evaluated'] == 'True',
        'legacy_same_query': row['legacy_query_coordinate_equal'] == 'True',
        'raw_crop_file': str(raw_path), 'raw_crop_sha256': sha(raw_path),
        'marked_crop_file': str(marked_path), 'marked_crop_sha256': sha(marked_path),
        'context_file': str(context_path), 'context_sha256': sha(context_path),
        'panel_file': str(panel_path), 'panel_sha256': sha(panel_path),
    })

for start, letter in [(0,'A'), (4,'B')]:
    sheet = Image.new('RGB', (1400, 530*4), '#131c2b')
    for j, panel in enumerate(panels[start:start+4]): sheet.paste(panel, (0, j*530))
    path = ROOT / f'Crop_Diagnostics_{letter}.png'
    sheet.save(path)
    manifest.setdefault('contact_sheets', []).append({'path': str(path), 'sha256': sha(path)})

(ROOT / 'CROP_DIAGNOSTIC_MANIFEST.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n')
print(json.dumps({'cases': len(manifest['cases']), 'manifest': str(ROOT / 'CROP_DIAGNOSTIC_MANIFEST.json'), 'sheets': manifest['contact_sheets']}, indent=2))
