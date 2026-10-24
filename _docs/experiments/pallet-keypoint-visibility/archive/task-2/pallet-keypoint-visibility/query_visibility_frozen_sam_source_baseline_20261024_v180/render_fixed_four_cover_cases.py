"""Illustrate all four original material/aspect COVER groups, no case search."""
from pathlib import Path
import hashlib
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

V = Path(__file__).resolve().parent
E = V / 'saved_sam_allTRAIN4096_evaluation_c_v3'
A = V / 'source_end_spatial_visual128_v1'
OUT = V / 'scientific_readout_v1'
ids = [8, 40, 521, 553]  # Original designated COVER native in bases0/1/16/17.
rows = {r['original_global_index']: r for r in (json.loads(x) for x in (E / 'ALL_QUERIES_SAM_AND_LITERAL_TARGET_DIAGNOSTICS.jsonl').read_text().splitlines())}
spatial = {r['original_global_index']: r for r in json.loads((A / 'ALL128_SPATIAL_RECORDS.json').read_text())['rows']}
font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 19)
small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
canvas = Image.new('RGB', (1260, 85 + 4 * 390), 'white')
d = ImageDraw.Draw(canvas)
d.text((14, 8), 'Fixed four original COVER groups | SYNTHETIC TRAIN | unchanged q/L | illustrative only', fill='black', font=font)
for x, label in zip([14, 324, 634, 944], ['Original96 + query mark', 'SAM binary mask', 'All-target raster weight', 'Fixed end region overlay']):
    d.text((x, 43), label, fill='black', font=small)
records = []
for number, i in enumerate(ids):
    r, a = rows[i], spatial[i]
    assert r['variant'] == 'COVER' and a['foreign_front_support_target'] == 1
    clean_path = E / f'SCENE_{r["scene_index"]:06d}' / f'NATIVE_{r["native_slot"]}_CLEAN96.png'
    clean = np.array(Image.open(clean_path))
    assert hashlib.sha256(clean.tobytes()).hexdigest() == a['raw96_array_sha256']
    with np.load(E / f'SCENE_{r["scene_index"]:06d}_NATIVE{r["native_slot"]}_SAMPLES.npz', allow_pickle=False) as z:
        pred, truth = z['sam_prediction96'], z['raster_owner_soft5_96']
    sam = np.zeros((96, 96, 3), np.uint8)
    sam[pred == 0] = 255
    sam[pred == 2] = [110, 90, 150]
    gt = np.repeat(np.floor(truth[..., 0, None] * 255 + .5).astype(np.uint8), 3, axis=2)
    overlay = Image.open(a['images']['OVERLAY96']['path']).convert('RGB')
    raw_display = Image.fromarray(clean).resize((288, 288), Image.Resampling.NEAREST)
    mark = ImageDraw.Draw(raw_display)
    center = (47.5 * 3 + 1, 47.5 * 3 + 1)
    mark.line((center[0] - 6, center[1], center[0] + 6, center[1]), fill=(0, 235, 245), width=2)
    mark.line((center[0], center[1] - 6, center[0], center[1] + 6), fill=(0, 235, 245), width=2)
    y = 85 + number * 390
    for x, im in zip([14, 324, 634, 944], [raw_display, Image.fromarray(sam).resize((288, 288), Image.Resampling.NEAREST), Image.fromarray(gt).resize((288, 288), Image.Resampling.NEAREST), overlay.resize((288, 288), Image.Resampling.NEAREST)]):
        canvas.paste(im, (x, y))
    caption = f'G{i:04d} / {r["factors"]["material"]} / {r["factors"]["aspect"]} / native {r["native_slot"]}'
    d.text((14, y + 298), caption, fill='black', font=small)
    d.text((14, y + 324), f'At q: T/F/BG weights={r["point_owner_soft5"][:3]}; SAM={r["prediction_bilinear"]}', fill='black', font=small)
    d.text((14, y + 350), 'End region: foreign-front YES. Red=end covered; green=end visible; blue=self UNKNOWN; gray=outside end.', fill='black', font=small)
    records.append(dict(original_global_index=i, original_query_xy=r['original_query_xy'], original_L=r['original_L'], native_slot=r['native_slot'],
                        point_owner_soft5=r['point_owner_soft5'], prediction_bilinear=r['prediction_bilinear'], source_clean96_sha256=a['raw96_array_sha256'],
                        original_clean96_unchanged=True, cyan_cross_on_display_copy_only=True))
out = OUT / 'FIXED4_COVER_INPUT_SAM_TARGET_END.png'
canvas.save(out)
(OUT / 'FIXED4_COVER_BINDINGS.json').write_text(json.dumps(dict(ids=ids, policy='All four original selected TRAIN material/aspect COVER target roles; fixed16 witness subset, no best/worst search.', rows=records,
                                                             image=dict(path=str(out), sha256=hashlib.sha256(out.read_bytes()).hexdigest())), indent=2) + '\n')
print(out)
