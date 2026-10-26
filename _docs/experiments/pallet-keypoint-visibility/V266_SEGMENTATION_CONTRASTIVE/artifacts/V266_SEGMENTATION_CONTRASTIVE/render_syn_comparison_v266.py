"""Render six deterministically selected SYN DEV examples from saved outputs.

No model forward, training, or ground-truth edit. Real qualitative notes refer
to four already rendered evaluation cards inspected by the reviewer.
"""
from pathlib import Path
import collections
import hashlib
import io
import json
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).absolute().parent
PALETTE = np.asarray([[45, 190, 90], [240, 95, 35], [90, 115, 180]], np.float32)


def safe(value):
    p = Path(value)
    if not p.is_absolute() or any(s in str(p).lower() for s in ('v112', 'cache', '.pyc', '..')):
        raise ValueError(str(p))
    for a in reversed((p, *p.parents)):
        if a.is_symlink():
            raise ValueError(str(a))
    return p


def binding(path, expected=None):
    p = safe(path)
    b = p.read_bytes()
    h = hashlib.sha256(b).hexdigest()
    if expected is not None:
        assert h == expected
    return {'path': str(p), 'sha256': h, 'size_bytes': len(b)}


def checked(b):
    assert binding(b['path']) == b
    return safe(b['path'])


def write_json(path, value):
    p = safe(path)
    assert not p.exists()
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n')


def rgb_crop(rgb, q, side):
    """V90 literal bilinear crop; uint8 RGB rounding retained."""
    h, w = rgb.shape[:2]
    u = (np.arange(96, dtype=np.float64) + .5) / 96 - .5
    xx, yy = np.meshgrid(q[0] + u * side, q[1] + u * side)
    ix, iy = np.floor(xx).astype(int), np.floor(yy).astype(int)
    dx, dy = xx - ix, yy - iy
    acc = np.zeros((96, 96, 3), np.float64)
    for ax, ay, weight in ((0, 0, (1-dx)*(1-dy)), (1, 0, dx*(1-dy)),
                          (0, 1, (1-dx)*dy), (1, 1, dx*dy)):
        x, y = ix + ax, iy + ay
        inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        acc[inside] += rgb[y[inside], x[inside]].astype(np.float64) * weight[inside, None]
    valid = (xx >= -.5) & (xx < w-.5) & (yy >= -.5) & (yy < h-.5)
    result = np.floor(np.clip(acc, 0, 255) + .5).astype(np.uint8)
    result[~valid] = 0
    return result


def owner_image(values, hard=False):
    if hard:
        rgb = PALETTE[values.argmax(0)]
    else:
        rgb = np.einsum('chw,cd->hwd', values.astype(np.float32), PALETTE)
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).resize((256, 256), Image.Resampling.NEAREST)


def main():
    started = time.time()
    out = safe(HERE / 'syn_visual_v1')
    assert not out.exists()
    protocol = json.loads(safe(HERE / 'PROTOCOL_BEFORE_FIT.json').read_text())
    complete = json.loads(safe(HERE / 'syn_training_v1/COMPLETE.json').read_text())
    source_bindings = protocol['source_bindings']
    with np.load(checked(source_bindings['dense_labels']), allow_pickle=False) as z:
        soft, mass, fit = (np.array(z[k]) for k in ('soft_target', 'known_mass', 'fit'))
    with np.load(checked(source_bindings['training_packet']), allow_pickle=False) as z:
        y, uv, base_logit = (np.array(z[k]) for k in ('y', 'uv', 'base_E_logit'))
    rows = [json.loads(l) for l in safe(HERE / 'data_v1/ROW_MANIFEST.jsonl').read_text().splitlines()]
    predictions = {}
    input_bindings = {'protocol': binding(HERE / 'PROTOCOL_BEFORE_FIT.json'),
        'training_complete': binding(HERE / 'syn_training_v1/COMPLETE.json'),
        'dense': source_bindings['dense_labels'], 'training_packet': source_bindings['training_packet'],
        'row_manifest': binding(HERE / 'data_v1/ROW_MANIFEST.jsonl')}
    for arm in ('SEGMENTATION', 'SEGMENTATION_CONTRAST'):
        b = complete['outputs'][arm + '_SYN_DEV_PREDICTIONS.npz']
        with np.load(checked(b), allow_pickle=False) as z:
            predictions[arm] = {k: np.array(z[k]) for k in ('source_index', 'pred', 'owner_probability', 'owner_summary')}
        input_bindings[arm] = b
    indices = predictions['SEGMENTATION']['source_index']
    assert np.array_equal(indices, np.flatnonzero(~fit))
    assert np.array_equal(indices, predictions['SEGMENTATION_CONTRAST']['source_index'])
    dev_position = {int(index): position for position, index in enumerate(indices)}
    foreign_mass = (soft[:, 1].astype(np.float64) * mass.astype(np.float64)).sum((1, 2))
    segpred = np.full(len(y), -2, np.int8)
    conpred = segpred.copy()
    segpred[indices] = predictions['SEGMENTATION']['pred']
    conpred[indices] = predictions['SEGMENTATION_CONTRAST']['pred']
    strata = [
        ('D / FOREIGN absent in crop label', (~fit) & (y == 0) & (foreign_mass == 0)),
        ('D / FOREIGN present in crop label', (~fit) & (y == 0) & (foreign_mass > 0)),
        ('E / FOREIGN absent in crop label', (~fit) & (y == 1) & (foreign_mass == 0)),
        ('E / FOREIGN present in crop label', (~fit) & (y == 1) & (foreign_mass > 0)),
        ('E / both new arms predict E', (~fit) & (y == 1) & (segpred == 1) & (conpred == 1)),
        ('E / both new arms predict D', (~fit) & (y == 1) & (segpred == 0) & (conpred == 0)),
    ]
    selection = []
    for title, condition in strata:
        candidates = np.flatnonzero(condition)
        selection.append({'cell': title, 'count': int(len(candidates)),
                          'source_index': int(candidates.min()) if len(candidates) else None})
    out.mkdir()
    write_json(out / 'SELECTION_BEFORE_SYN_RGB.json', {'completed': True,
        'rule': 'First original source_index per six prespecified DEV D/E x foreign presence or correctness cells. Empty cells retained. Repeated selections retained.',
        'cells': selection, 'RGB_reads_before_selection': 0})
    board = Image.new('RGB', (1220, 1940), 'white')
    fontpath = safe(Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'))
    font = ImageFont.truetype(str(fontpath), 16)
    titlefont = ImageFont.truetype(str(fontpath), 21)
    d = ImageDraw.Draw(board)
    d.text((16, 8), 'V266 합성 DEV: 실제 crop / 분할 정답 / 분할 예측 / 분할+대조학습 예측', font=titlefont, fill='black')
    d.text((16, 40), '초록=대상 팔레트 · 주황=외부 물체 · 파랑=배경. q는 중앙, 노랑+는 고정 heatmap 위치 후보. 색 범위 고정.', font=font, fill='black')
    source_rgb = {}
    examples = []
    for panel, cell in enumerate(selection):
        top = 90 + panel * 304
        d.text((16, top), cell['cell'] + f' | census={cell["count"]}', font=font, fill='black')
        index = cell['source_index']
        if index is None:
            d.text((32, top+48), '해당 셀에 사례 없음 — 다른 사례로 채우지 않음', font=font, fill='black')
            continue
        r = rows[index]
        image_path = r['image']['path']
        if image_path not in source_rgb:
            b = binding(image_path, r['image']['sha256'])
            with np.load(safe(image_path), allow_pickle=False) as z:
                source_rgb[image_path] = np.array(z['RGB_uint8'], copy=True)
            input_bindings[image_path] = b
        rgb = rgb_crop(source_rgb[image_path], np.asarray(r['q0']), float(r['crop_side']))
        clean = safe(out / f'SYN_ROW{index:04d}_CLEAN96.png')
        if not clean.exists():
            Image.fromarray(rgb).save(clean)
        pos = dev_position[index]
        tiles = [Image.fromarray(rgb).resize((256, 256), Image.Resampling.NEAREST),
                 owner_image(soft[index]),
                 owner_image(predictions['SEGMENTATION']['owner_probability'][pos], True),
                 owner_image(predictions['SEGMENTATION_CONTRAST']['owner_probability'][pos], True)]
        for col, tile in enumerate(tiles):
            x, yy = 16 + col*302, top+30
            td = ImageDraw.Draw(tile)
            td.ellipse((124, 124, 132, 132), outline=(0, 240, 255), width=2)
            hp = uv[index] * 256
            td.line((hp[0]-6, hp[1], hp[0]+6, hp[1]), fill='yellow', width=2)
            td.line((hp[0], hp[1]-6, hp[0], hp[1]+6), fill='yellow', width=2)
            board.paste(tile, (x, yy))
        examples.append({'cell': cell, 'row': r, 'clean': binding(clean),
             'RGB96_HWC_sha256': hashlib.sha256(rgb.tobytes()).hexdigest(),
             'native_D_E_label': int(y[index]), 'baseline_E_logit': float(base_logit[index]),
             'foreground_label_token_mass': float(foreign_mass[index]),
             'predictions': {a: {'pred': int(p['pred'][pos]),
                    'owner_q_heatmap_global': p['owner_summary'][pos].reshape(3, 3).tolist()} for a, p in predictions.items()}})
    board.save(safe(out / 'SYN_RGB_GT_PRED_FIXED6_KO.png'))
    current = json.loads(safe(HERE / 'CURRENT_PREDICTIONS_FROZEN.json').read_text())
    with np.load(checked(current['descriptor_predictions']), allow_pickle=False) as z:
        descriptors = {arm: np.array(z[arm]) for arm in ('SEGMENTATION', 'SEGMENTATION_CONTRAST')}
    cases = {c['global_index']: c for c in json.loads(safe(HERE / 'evaluation_v1/VISUAL_CASE_DATA.json').read_text())['cases']}
    notes = {
        370: '실제 콘의 빨간 몸통 일부는 외부 물체로 분할하지만, 흰 띠를 대상 팔레트로 분할함. 같은 콘 내부를 하나의 소유 물체로 묶지 못함.',
        256: '바닥은 배경으로 분리하지만 빨간 콘 몸통의 넓은 부분을 대상 팔레트로 분할함. heatmap은 바닥 경계 쪽임.',
        908: '콘 위 몸통은 외부 물체로, 콘 아래 받침·경계는 팔레트와 연결된 대상으로 분할함. heatmap도 받침 경계 쪽임.',
        996: '드러난 팔레트 조각을 대상으로, 콘 몸통을 외부 물체로 분할하는 해석은 시각적으로 타당한 부분이 있음. 기존 E 주석과 사용자의 보임 해석이 달라 주석 재검토 사례로 유지.'}
    real_notes = []
    for index in (370, 256, 908, 996):
        c = cases[index]
        real_notes.append({'global_index': index, 'row_id': c['row_id'],
           'existing_human_state': c['human_state'], 'coordinate_source': c['annotation_coordinate_source'],
           'qualitative_reviewer_observation': notes[index], 'card': c['card'],
           'owner_probability_readout_class_order': ['TARGET', 'FOREIGN', 'BACKGROUND'],
           'pooled_region_order': ['initial_query_Gaussian_sigma1.5', 'heatmap_Gaussian_sigma1.5', 'global'],
           'models': {a: {'q_heatmap_global_owner_probability': descriptors[a][index, 1:10].reshape(3, 3).tolist(),
                         'point_target_probability_at_q_from_evaluator': c['dense_diagnostics'][a]['target_probability_at_initial_query'],
                         'scalar_prediction': c['predictions'][a]} for a in descriptors}})
    write_json(out / 'SYN_FIXED6_CASE_DATA.json', {'completed': True, 'cases': examples, 'source_RGB_assets_decoded': len(source_rgb)})
    write_json(out / 'REAL4_QUALITATIVE_REVIEW_KO.json', {'completed': True, 'cases': real_notes,
         'real_pixel_segmentation_GT': 'ABSENT; no real IoU/segmentation accuracy claim',
         'common_problem': 'SYN dense ownership can be learned, but real cones are split by local appearance into foreign versus target regions. This small head does not reliably enforce whole-object ownership or depth/instance relations.',
         'not_established': 'These four diagnostic cases do not measure the overall frequency or prove a unique causal explanation. G996 is not asserted as an obvious model mistake.',
         'model_or_label_or_threshold_edits': 0})
    html = '''<!doctype html><meta charset="utf-8"><title>V266 합성·실사 영역분할 진단</title>
<style>body{font:18px sans-serif;max-width:1300px;margin:32px auto;background:#f4f6f8;color:#123}img{max-width:100%;background:white}p{line-height:1.6}a{color:#176bc1}</style>
<h1>합성에서는 영역 감독을 배우지만, 실사 콘의 일부를 팔레트로 읽습니다</h1>
<p>아래는 합성 DEV의 사전 규칙으로 고른 여섯 셀입니다. 같은 사례가 여러 셀에 해당하면 반복하며 빈 셀은 유지합니다. 정답 지도는 저장 renderer owner를 사용합니다. 픽셀·token 분할 정답이 있는 합성과 실사 정성 검토는 구분합니다.</p>
<img src="SYN_RGB_GT_PRED_FIXED6_KO.png">
<p>실사 G370: 콘 흰 띠→팔레트, G256: 콘 몸통→팔레트, G908: 콘 받침→팔레트가 공통 문제입니다. G996은 보이는 팔레트 조각과 콘을 구분한 부분이 있어 기존 E 주석의 재검토 사례로 유지합니다. 실제 분할 GT가 없으므로 실사 분할 IoU로 표현하지 않습니다.</p>
<p><a href="../evaluation_v1/CASE_G0370.png">G370</a> · <a href="../evaluation_v1/CASE_G0256.png">G256</a> · <a href="../evaluation_v1/CASE_G0908.png">G908</a> · <a href="../evaluation_v1/CASE_G0996.png">G996</a> · <a href="REAL4_QUALITATIVE_REVIEW_KO.json">지역별 확률과 관측 기록</a></p>'''
    safe(out / 'INDEX_KO.html').write_text(html)
    output_files = ['SELECTION_BEFORE_SYN_RGB.json', 'SYN_RGB_GT_PRED_FIXED6_KO.png', 'SYN_FIXED6_CASE_DATA.json',
                    'REAL4_QUALITATIVE_REVIEW_KO.json', 'INDEX_KO.html']
    output_files += sorted({Path(c['clean']['path']).name for c in examples})
    write_json(out / 'COMPLETE.json', {'completed': True, 'PASS': True,
        'code': binding(Path(__file__).absolute()), 'inputs': input_bindings,
        'outputs': {name: binding(out/name) for name in output_files},
        'source_RGB_assets_decoded': len(source_rgb), 'current_clean_RGB_decodes': 0,
        'four_existing_real_cards_visually_inspected': [370, 256, 908, 996],
        'model_forwards': 0, 'optimizer_updates': 0, 'threshold_or_label_edits': 0,
        'elapsed_seconds': time.time()-started})
    print(json.dumps({'completed': True, 'PASS': True, 'complete': binding(out/'COMPLETE.json'),
                      'selection': selection, 'real4': real_notes}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
