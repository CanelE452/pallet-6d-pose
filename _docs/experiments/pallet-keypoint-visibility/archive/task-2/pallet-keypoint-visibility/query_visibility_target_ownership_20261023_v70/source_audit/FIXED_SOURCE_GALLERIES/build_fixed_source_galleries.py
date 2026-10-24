"""Render previously fixed source cases, joining saved scores only. No model imports."""
from pathlib import Path
import csv
import hashlib
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARENT = ROOT.parent
V68 = PARENT / 'query_visibility_sam21_frozen_20261023_v68'
VFILE = V68 / 'source_audit/FIXED_PUBLIC_VISIBLE_CROPS/PIXEL_AND_PROVENANCE_CHECK.json'
EFILE = V68 / 'source_audit/SAM21_EXTERNAL_FALSE_VISIBLE_CROPS/DIAGNOSIS.json'
RESULT = ROOT / 'source_evaluation/OWNERSHIP_FROZEN_COMPARISON/SOURCE_RESULTS.json'
SCORES = RESULT.parent / 'PER_QUERY.csv'
INPUTS = V68 / 'source_inputs/INPUTS_RGB_ONLY.jsonl'
MODELS = [('V61_FROZEN', 'v61'), ('V65_BASELINE', 'v65'),
          ('V66_CONTROLLED_PLUS_PALLOX', 'v66 PLUS'),
          ('SAM21_BASE_PLUS_FROZEN', 'SAM2.1'), ('V70_OWNERSHIP_LOO_PAIR_V1', 'v70')]
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
BG = '#f5f6f8'
DARK = '#17232f'
MUTED = '#576775'


def require(x, message):
    if not x:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def ref(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def read(path):
    return json.loads(Path(path).read_text())


def checked(binding):
    require(sha(binding['path']) == binding['sha256'], 'Input hash mismatch ' + binding['path'])
    return Path(binding['path'])


def rgb(binding):
    return np.asarray(Image.open(checked(binding)).convert('RGB')).copy()


def strict25(mask, pixel):
    x, y = pixel
    require(2 <= x < mask.shape[1] - 2 and 2 <= y < mask.shape[0] - 2, 'Native strict25 bounds')
    return bool(mask[y-2:y+3, x-2:x+3].all())


def font(size):
    return ImageFont.truetype(FONT, size)


def text(draw, xy, value, size=23, color=DARK):
    draw.text(xy, str(value), font=font(size), fill=color)


def view(array, window, width, height):
    """Nearest native pixel-center display; no model input or RGB interpolation."""
    x0, y0, x1, y1 = window
    xx = np.floor(x0 + (np.arange(width) + .5) * (x1-x0) / width + .5).astype(int)
    yy = np.floor(y0 + (np.arange(height) + .5) * (y1-y0) / height + .5).astype(int)
    inside = (yy[:, None] >= 0) & (yy[:, None] < array.shape[0]) & (xx[None, :] >= 0) & (xx[None, :] < array.shape[1])
    result = array[np.clip(yy, 0, array.shape[0]-1)[:, None], np.clip(xx, 0, array.shape[1]-1)[None, :]].copy()
    result[~inside] = 0
    return Image.fromarray(result), hashlib.sha256(result.tobytes()).hexdigest()


def panel(canvas, array, window, xy, size, q, pixel, side, box=None, title=''):
    x, y = xy
    w, h = size
    im, base_sha = view(array, window, w, h)
    d = ImageDraw.Draw(im)
    x0, y0, x1, y1 = window
    def point(a, b):
        return ((a-x0) / (x1-x0) * w, (b-y0) / (y1-y0) * h)
    def rect(bounds, color, line=2):
        a, b = point(bounds[0], bounds[1]); c, e = point(bounds[2], bounds[3])
        d.rectangle((a, b, c, e), outline=color, width=line)
    if box is not None:
        rect(box, '#00d9ee')
    # Initial model ROI is q-centered and has side L; display is a separate window.
    rect([q[0]-side/2, q[1]-side/2, q[0]+side/2, q[1]+side/2], '#ffba42')
    px, py = pixel
    rect([px-2.5, py-2.5, px+2.5, py+2.5], '#d9ff37')
    qx, qy = point(*q)
    d.line((qx-9, qy, qx+9, qy), fill='#ff10bb', width=3)
    d.line((qx, qy-9, qx, qy+9), fill='#ff10bb', width=3)
    canvas.paste(im, (x, y))
    draw = ImageDraw.Draw(canvas)
    text(draw, (x, y-31), title, 22)
    draw.rectangle((x, y, x+w, y+h), outline='#a9b4bc', width=1)
    return dict(native_display_window_xyxy=list(map(float, window)), displayed_base_rgb_sha256=base_sha,
                display_wh=[w, h], sampling='nearest floor(native_xy+.5), no interpolation',
                overlays='magenta initial q; lime native5x5; orange q-centered input L; cyan input target box')


def saved(score, sid):
    out = {}
    for model, _ in MODELS:
        r = score[(model, sid)]
        require(r['native_valid'] == 'True', 'Fixed case became invalid')
        out[model] = dict(p_visible=float(r['p_visible']), prediction=int(r['prediction']),
                          exact_GT=int(r['exact_GT']), allowed=int(r['allowed']),
                          native_valid=True, image_sha256=r['image_sha256'])
    require(len({r['exact_GT'] for r in out.values()}) == 1, 'Saved GT mismatch')
    return out


def score_table(draw, x, y, after, before=None):
    text(draw, (x, y), '모델', 24)
    if before:
        text(draw, (x+210, y), 'exact pV 원본 → 개입 후', 23)
        text(draw, (x+475, y), '판정', 23)
    else:
        text(draw, (x+210, y), 'exact pV', 23)
        text(draw, (x+345, y), '판정', 23)
    for k, (model, label) in enumerate(MODELS):
        row = after[model]
        yy = y + 40 + k * 39
        text(draw, (x, yy), label, 25)
        if before:
            text(draw, (x+210, yy), f"{before[model]['p_visible']:.3f} → {row['p_visible']:.3f}", 25)
            tx = x+475
        else:
            text(draw, (x+210, yy), f"{row['p_visible']:.3f}", 25)
            tx = x+345
        pred = 'V' if row['prediction'] == 1 else ('nV' if row['prediction'] == 0 else 'U')
        good = row['prediction'] == (1 if row['exact_GT'] == 0 else 0)
        text(draw, (tx, yy), pred, 25, '#10734d' if good else '#c63e36')


def main():
    jsonout = HERE / 'FIXED_SOURCE_GALLERIES.json'
    require(not jsonout.exists(), 'Preserve immutable gallery')
    v = read(VFILE); e = read(EFILE); result = read(RESULT)
    require(v['selection_unchanged'] and v['selection_preceded_SAM21_extraction'], 'Existing V selection proof')
    require(result['completed'] and result['source_only'] and not result['current_inference_allowed'], 'Saved source comparison only')
    require(sha(SCORES) == next(b['sha256'] for b in result['outputs'] if Path(b['path']).name == 'PER_QUERY.csv'), 'Saved score seal')
    old_selection = read(checked(v['selection']))
    vrecords = {r['sample_id']: r for r in old_selection['chosen']}
    with SCORES.open() as f:
        rows = list(csv.DictReader(f))
    score = {(r['model'], r['sample_id']): r for r in rows}
    require(len(rows) == 7*8590 and len(score) == len(rows), 'Complete seven-model saved canvas')
    inputs = {r['global_id']: r for r in map(json.loads, INPUTS.open())}
    vcases = []
    ecases = []
    for example in v['examples']:
        sid = example['sample_id']; record = vrecords[sid]
        original = rgb(example['RGB'])
        vis = np.asarray(Image.open(checked(example['target_visible_mask']))) > 0
        crop = example['crop_box_native']; pixel = example['pixel']
        gid, k = sid.rsplit(':', 1)
        q = record['source_record']['query'][int(k)]; side = record['source_record']['side']; box = record['source_record']['box']
        manifest = inputs[gid]; query = next(r for r in manifest['queries'] if r['sample_id'] == sid)
        require(q == query['query'] and pixel == query['pixel'] and side == manifest['side'] and box == manifest['box'], 'V frozen input parity')
        require(list(np.floor(np.array(q)+.5).astype(int)) == pixel and strict25(vis, pixel), 'V strict25 native truth')
        ss = saved(score, sid)
        require(all(r['exact_GT'] == 0 and r['image_sha256'] == example['RGB']['sha256'] for r in ss.values()), 'V score GT/RGB identity')
        require(ss['V65_BASELINE']['p_visible'] == example['V65_pV'] and ss['V66_CONTROLLED_PLUS_PALLOX']['p_visible'] == example['V66_pV'], 'V old score parity')
        vcases.append(dict(sample_id=sid, query=q, pixel=pixel, side=side, box=box, crop_box_native=crop,
                           RGB=example['RGB'], target_visible_mask=example['target_visible_mask'],
                           GT='V', evidence='BOP original visible mask, strict native25px; conditional source evidence',
                           strict25_V_verified=True, saved_scores=ss, _rgb=original))
    for case in e['cases']:
        sid = case['sample_id']; q = case['query']; pixel = case['pixel']; gid = case['global_id']
        original = rgb(case['RGB_bindings']['original']); controlled = rgb(case['RGB_bindings']['controlled'])
        manifest = inputs[gid]; query = next(r for r in manifest['queries'] if r['sample_id'] == sid)
        require(q == query['query'] and pixel == query['pixel'] and case['side'] == manifest['side'] and case['source_input_box'] == manifest['box'], 'E frozen input parity')
        require(list(np.floor(np.array(q)+.5).astype(int)) == pixel, 'E nearest native pixel')
        with np.load(checked(case['labels']), allow_pickle=False) as z:
            orig_vis = z['original_visible']; paste = z['paste_mask']; new_vis = z['visible_positive']; cert = z['controlled_E_pixel']
        require(strict25(orig_vis, pixel) and strict25(paste, pixel) and strict25(cert, pixel), 'E entire native5x5 causal intervention')
        require(not new_vis[pixel[1]-2:pixel[1]+3, pixel[0]-2:pixel[0]+3].any(), 'E no visible support in25px')
        require(np.array_equal(original[~paste], controlled[~paste]), 'RGB outside unchanged pasted mask')
        require(original[pixel[1], pixel[0]].tolist() == case['query_RGB_original'] and controlled[pixel[1], pixel[0]].tolist() == case['query_RGB_controlled'], 'Prior native RGB pixel parity')
        before_sid = case['original_target_id'] + ':' + str(case['native_corner'])
        before = saved(score, before_sid); after = saved(score, sid)
        require(all(r['exact_GT'] == 0 and r['image_sha256'] == case['RGB_bindings']['original']['sha256'] for r in before.values()), 'E original V score GT/RGB')
        require(all(r['exact_GT'] == 1 and r['image_sha256'] == case['RGB_bindings']['controlled']['sha256'] for r in after.values()), 'E after score GT/RGB')
        ecases.append(dict(sample_id=sid, original_sample_id=before_sid, query=q, pixel=pixel,
                           side=case['side'], box=case['source_input_box'], display=case['display'],
                           RGB_bindings=case['RGB_bindings'], labels=case['labels'],
                           GT_before='V', GT_after='E', strict25_E_verified=True,
                           RGB_outside_paste_bit_exact=True, query_RGB_original=case['query_RGB_original'],
                           query_RGB_controlled=case['query_RGB_controlled'],
                           inside_paste_replay='Previously verified bit-exact V68 replay, bound through DIAGNOSIS; not recomputed here',
                           evidence='Opaque different-instance cover of originally visible native target pixels; conditional BOP/cutout evidence',
                           saved_scores_before=before, saved_scores_after=after, _before=original, _after=controlled))
    require(len(vcases) == len(ecases) == 4, 'Keep all four cases, no score-based filtering')

    visible = Image.new('RGB', (1580, 1695), BG)
    d = ImageDraw.Draw(visible)
    text(d, (30, 20), '고정 공개 V 사례 4개 · V70 저장 점수 비교', 34)
    text(d, (30, 70), 'V66에서 미리 고른 동일 ID · native 5×5 전부 V · 새 점수에 따른 선택/제외 없음', 24)
    text(d, (30, 110), '자홍 q | 연두 5×5 | 주황 입력 L | 청록 target box · 확대는 nearest', 22, MUTED)
    for i, c in enumerate(vcases):
        y = 185 + i*365
        text(d, (30, y-35), f"{i+1}. {c['sample_id']}   GT=V   q=({c['query'][0]:.2f},{c['query'][1]:.2f})  L={c['side']:.1f}px", 25)
        arr = c.pop('_rgb'); h, w = arr.shape[:2]
        c['context_display'] = panel(visible, arr, [-.5, -.5, w-.5, h-.5], (30, y+25), (384, 288), c['query'], c['pixel'], c['side'], c['box'], '원본 RGB / 초기 target box')
        x0, y0, x1, y1 = c['crop_box_native']
        require(x1-x0 == y1-y0 == 96, 'Original native96 crop unchanged')
        c['crop_display'] = panel(visible, arr, [x0-.5, y0-.5, x1-.5, y1-.5], (445, y+25), (288, 288), c['query'], c['pixel'], c['side'], title='동일 native96 crop')
        rawcrop = arr[y0:y1, x0:x1]
        c['native96_RGB_bytes_sha256'] = hashlib.sha256(rawcrop.tobytes()).hexdigest()
        score_table(d, 790, y+26, c['saved_scores'])
    text(d, (30, 1650), 'pV≥.5: V, 그 외 nV. SAM2.1은 저장 native logit>0. nV는 원인 E/S 또는 물리적 부재의 증명이 아닙니다.', 21, MUTED)

    external = Image.new('RGB', (1840, 1695), BG)
    d = ImageDraw.Draw(external)
    text(d, (30, 20), '고정 공개 E 사례 4개 · 원본 → 다른 물체의 불투명 가림', 34)
    text(d, (30, 70), 'V68의 동일 SAM2.1 false-VISIBLE 4개 · q/box/L 동일 · 원본 GT=V → 개입 후 GT=E', 24)
    text(d, (30, 110), '기존 고정 query 표시 창 그대로 · 자홍 q | 연두 5×5 | 주황 입력 L | 청록 target box', 22, MUTED)
    for i, c in enumerate(ecases):
        y = 185 + i*365
        text(d, (30, y-35), f"{i+1}. {c['sample_id']}   L={c['side']:.1f}px  strict25E=True", 23)
        before = c.pop('_before'); after = c.pop('_after')
        win = c['display']['query_display_window_xyxy']; ctx = c['display']['prompt_context_window_xyxy']
        c['context_display'] = panel(external, after, ctx, (30, y+25), (300, 288), c['query'], c['pixel'], c['side'], c['box'], '개입 후 target 문맥')
        c['before_display'] = panel(external, before, win, (355, y+25), (288, 288), c['query'], c['pixel'], c['side'], title='동일 q / 원본 V')
        c['after_display'] = panel(external, after, win, (665, y+25), (288, 288), c['query'], c['pixel'], c['side'], title='동일 q / 개입 후 E')
        score_table(d, 1000, y+26, c['saved_scores_after'], c['saved_scores_before'])
    text(d, (30, 1650), 'E는 개입/원본 visible 소유권에 근거한 source 증거. pV는 exact support 점수이며 nV를 학습된 causal E라고 부르지 않습니다.', 21, MUTED)
    vp = HERE / 'FIXED_VISIBLE_SAVED_COMPARISON.png'; ep = HERE / 'FIXED_EXTERNAL_SAVED_COMPARISON.png'
    require(not vp.exists() and not ep.exists(), 'Preserve previous PNGs')
    visible.save(vp); external.save(ep)
    report = dict(schema='V70_PRESELECTED_FIXED_SOURCE_SAVED_GALLERIES_V1', completed=True,
                  selection_unchanged=True, selected_on_V70_scores=False, all_fixed_cases_retained=True,
                  source_only=True, inputs=dict(V68_fixed_V=ref(VFILE), V68_fixed_E=ref(EFILE),
                  V66_original_V_selection=v['selection'], source_input_manifest=ref(INPUTS),
                  saved_source_comparison=ref(RESULT), saved_scores=ref(SCORES), script=ref(__file__)),
                  visible_cases=vcases, external_cases=ecases, outputs=[ref(vp), ref(ep)],
                  fixed_example_only='Not a representative metric. These are unchanged prior diagnostic cases, not new score-selected examples.',
                  semantic_limit='Native source pixel evidence, not human semantic reference-corner ground truth. Negative score is binary nonvisibility, not inferred causal E/O/S.',
                  RGB_and_crop_provenance_verified=True, new_model_forward_calls=0, new_backbone_or_SAM_calls=0,
                  optimizer_updates=0, PnP_calls=0, current223_RGB_GT_scores_inputs_read=0,
                  old_files_modified=False)
    jsonout.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(report=ref(jsonout), images=[ref(vp), ref(ep)]), ensure_ascii=False))


if __name__ == '__main__':
    main()
