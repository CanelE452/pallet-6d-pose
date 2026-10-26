#!/usr/bin/env python3
"""Independent, saved-output-only V264 evaluation and diagnostic crop gallery.

Run only after Root has frozen the new configurations and all 1,784 predictions.
No optimizer, encoder, checkpoint, solver, or threshold-selection code is imported.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import html
import io
import json
import math
import os
from pathlib import Path
import stat
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw, ImageFont


P = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
R = P / 'publication_root_preparation'
DEFAULT_ROOT = R / 'V264_HEATMAP_LOCAL_VISIBILITY'
A = R / 'V263_CURRENT_EVALUATION_A'
C = R / 'V263_SPATIAL_HEAD_C/current223_frozen_inference_v1'
D, E, S = 'DIRECT_VISIBLE', 'EXTERNAL_OCCLUDED', 'SELF_OCCLUDED'
ARMS = ('GLOBAL_MATCHED', 'HEATMAP_LOCAL')
GROUPS = ('ALL', 'WOOD_PRIMARY105', 'SQUARE118')
CELLS = ('BOTH_CORRECT', 'LOCAL_ONLY_CORRECT', 'GLOBAL_ONLY_CORRECT', 'BOTH_WRONG')

PINNED = {
    'join': dict(path=str(A / 'actual_evaluation_v1/ALL1784_CLASSIFICATION_LOCATION_JOIN.csv'), sha256='df7b619649ede26af4e21644c22b10e84b334c6bc8708f4e5a668ec373db5175', size_bytes=1916373),
    'old_classification': dict(path=str(A / 'actual_evaluation_v1/CLASSIFICATION.csv'), sha256='15cc01327bddcc825da6ce51ffb73565635471cc1917df5beaa32bf65a8012cc', size_bytes=13994),
    'v263_current_complete': dict(path=str(C / 'COMPLETE.json'), sha256='8fcfa2761b26e3f2fa236f525bf98267d4964d7a650fef5bceb99b7d2ac3e881', size_bytes=6412),
    'old_evaluation_entry': dict(path=str(A / 'ACTUAL_SAVED_CURRENT_EVALUATION_ENTRY.json'), sha256='c7b91a316199671b0c5146b4dd2ade302b21055663f78483616d6aa515b74ecf', size_bytes=10220),
}


def need(value, message):
    if not bool(value):
        raise ValueError(message)


def safe(value, fresh=False):
    p = Path(value)
    need(p.is_absolute() and '..' not in p.parts, 'Absolute nontraversal artifact path required')
    need(not any(x in str(p).casefold() for x in ('v112', 'cache', '.pyc')), 'Forbidden lexical path')
    for ancestor in [*reversed(p.parents), p]:
        try:
            mode = os.lstat(ancestor).st_mode
        except FileNotFoundError:
            continue
        need(not stat.S_ISLNK(mode), 'Artifact symlink/ancestor forbidden: ' + str(ancestor))
    if fresh:
        need(not p.exists(), 'Immutable output already exists: ' + str(p))
    return p


def bind(path):
    p = safe(path)
    raw = p.read_bytes()
    return dict(path=str(p), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))


def checked(binding):
    need(isinstance(binding, dict) and all(k in binding for k in ('path', 'sha256', 'size_bytes')), 'Complete input binding required')
    p = safe(binding['path'])
    raw = p.read_bytes()
    need(len(raw) == binding['size_bytes'] and hashlib.sha256(raw).hexdigest() == binding['sha256'], 'Input SHA/size mismatch: ' + str(p))
    return raw


def walk_bindings(value):
    if isinstance(value, dict):
        if all(k in value for k in ('path', 'sha256', 'size_bytes')):
            yield {k: value[k] for k in ('path', 'sha256', 'size_bytes')}
        for child in value.values():
            yield from walk_bindings(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_bindings(child)


def write_json(path, value):
    p = safe(path, fresh=True)
    with p.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')
    return bind(p)


def write_csv(path, rows):
    p = safe(path, fresh=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with p.open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)
    return bind(p)


def num(value):
    try:
        value = float(value)
    except (ValueError, TypeError):
        return None
    return value if math.isfinite(value) else None


def ratio(n, d):
    return n / d if d else None


def truth(row):
    return 'D' if row['human_state'] == D else 'E'


def manual_d(row):
    return row['human_state'] == D and row['annotation_coordinate_source'] == 'manual_click'


def primary(row):
    return row['paired_same_query_RGB_D_vs_E'] == 'True' and row['human_state'] in (D, E)


def near_d(row):
    error = num(row['initial_reference_error_px'])
    return manual_d(row) and row['original_query_valid'] == 'True' and error is not None and error <= 3


def cm(rows, predictions):
    counts = Counter((truth(r), predictions[int(r['global_index'])]) for r in rows)
    dd, de, du, ed, ee, eu = [counts[k] for k in (('D', 'D'), ('D', 'E'), ('D', 'U'), ('E', 'D'), ('E', 'E'), ('E', 'U'))]
    nd, ne = dd + de + du, ed + ee + eu
    return dict(N=len(rows), truth_D=nd, truth_E=ne, D_D=dd, D_E=de, D_U=du, E_D=ed, E_E=ee, E_U=eu,
                D_precision=ratio(dd, dd + ed), D_recall=ratio(dd, nd), D_F1=ratio(2 * dd, 2 * dd + ed + de + du),
                E_precision=ratio(ee, ee + de), E_recall=ratio(ee, ne), E_F1=ratio(2 * ee, 2 * ee + de + ed + eu),
                U=du + eu, U_rate=ratio(du + eu, len(rows)), coverage=ratio(dd + de + ed + ee, len(rows)),
                accuracy_including_U_as_unresolved=ratio(dd + ee, len(rows)),
                balanced_recall=((dd / nd + ee / ne) / 2 if nd and ne else None),
                accepted_accuracy=ratio(dd + ee, dd + de + ed + ee))


def transition(rows, global_predictions, local_predictions):
    counts = Counter()
    responses = Counter()
    for r in rows:
        i = int(r['global_index'])
        g, l, t = global_predictions[i], local_predictions[i], truth(r)
        gc, lc = g == t, l == t
        key = 'BOTH_CORRECT' if gc and lc else 'LOCAL_ONLY_CORRECT' if lc else 'GLOBAL_ONLY_CORRECT' if gc else 'BOTH_WRONG'
        counts[key] += 1
        responses[(t, g, l)] += 1
    return dict(N=len(rows), **{k: counts[k] for k in CELLS}, net_correct=counts['LOCAL_ONLY_CORRECT'] - counts['GLOBAL_ONLY_CORRECT'],
                changed=sum(v for (t, g, l), v in responses.items() if g != l),
                response_transitions=[dict(truth=t, global_prediction=g, local_prediction=l, N=n) for (t, g, l), n in sorted(responses.items())])


def load_new(root):
    config_path = safe(root / 'NEW_CONFIG_FROZEN_BEFORE_CURRENT.json')
    prediction_freeze_path = safe(root / 'CURRENT_PREDICTIONS_FROZEN.json')
    config_binding, freeze_binding = bind(config_path), bind(prediction_freeze_path)
    config, freeze = json.loads(checked(config_binding)), json.loads(checked(freeze_binding))
    need(config.get('completed') is True or config.get('PASS') is True, 'New configuration freeze must be complete')
    need(freeze.get('completed') is True or freeze.get('PASS') is True, 'All current predictions must be frozen')
    wanted = root / 'CURRENT_ALL1784_PREDICTIONS.jsonl'
    bindings = [b for b in walk_bindings(freeze) if b['path'] == str(wanted)]
    need(len(bindings) >= 1 and all(b == bindings[0] for b in bindings), 'Freeze must bind exact current predictions')
    prediction_binding = bindings[0]
    predictions = [json.loads(line) for line in checked(prediction_binding).splitlines() if line]
    need(len(predictions) == 1784, 'Exactly 1784 new prediction rows required')
    for i, row in enumerate(predictions):
        need(row['global_index'] == i and row['native_slot'] == i % 8, 'New row/native order')
        need(row['row_id'] == row['frame_id'] + ':native' + str(i % 8), 'New row/frame identity')
        need(type(row['available']) is bool, 'Availability must be Boolean')
        for arm in ARMS:
            a = row[arm]
            need(a['pred'] in ('D', 'E', 'U'), 'D/E/U only')
            if row['available']:
                z, p = num(a['E_logit']), num(a['p_E'])
                need(z is not None and p is not None and 0 <= p <= 1, 'Finite supported probability/logit')
                expected = 1 / (1 + math.exp(-max(-700, min(700, z))))
                need(abs(p - expected) <= 2e-6, 'p_E must be sigmoid(E_logit)')
                need(a['pred'] == ('E' if z > 0 else 'D'), 'Frozen logit zero classification required')
            else:
                need(a['pred'] == 'U', 'Unavailable original input must stay U')
    return predictions, dict(config_freeze=config_binding, prediction_freeze=freeze_binding, predictions=prediction_binding)


def evaluate(root, out, new, seals):
    # The first late annotation/baseline read occurs only after load_new succeeds.
    ledger = list(csv.DictReader(io.StringIO(checked(PINNED['join']).decode('utf-8-sig'))))
    need(len(ledger) == 1784, 'Frozen current ledger1784')
    for i, (row, pred) in enumerate(zip(ledger, new)):
        need(int(row['global_index']) == i and int(row['native_corner']) == i % 8 and int(row['frame_index']) == i // 8, 'Frozen ledger order')
        need(row['id'] == pred['frame_id'] and row['row_id'] == pred['row_id'], 'Exact frame and native row join')
        need((row['original_query_valid'] == 'True') == pred['available'], 'No availability-dependent denominator change')
    need(sum(primary(r) for r in ledger) == 1237 and sum(primary(r) and r['human_state'] == D for r in ledger) == 1147, 'Primary1237 D1147/E90 invariant')
    need(sum(near_d(r) for r in ledger) == 376, 'Manual visible initial<=3 subset376')
    need(sum(r['human_state'] == S for r in ledger) == 136, 'SELF136 invariant')

    predictions = {arm: [r[arm]['pred'] for r in new] for arm in ARMS}
    predictions.update(v61=[r['v61_D_E_U'] for r in ledger], CAL_LEGACY=[r['legacy_D_E_U'] for r in ledger],
                       V263_JOINT=[r['JOINT_SELECTED__semantic_VIS_prediction'] for r in ledger],
                       V263_VIS=[r['VIS_ONLY_SELECTED__semantic_VIS_prediction'] for r in ledger])
    baseline_entry = json.loads(checked(PINNED['old_evaluation_entry']))
    baseline_bindings = baseline_entry['V261_saved_predictions']
    for arm in ('V261_CONTROL', 'V261_TREAT'):
        rows = [json.loads(line) for line in checked(baseline_bindings[arm]).splitlines() if line]
        need(len(rows) == 1784, 'V261 saved rows1784')
        for i, (r, meta) in enumerate(zip(rows, ledger)):
            need(r['original_global_index'] == i and r['row_id'] == meta['row_id'] and r['id'] == meta['id'] and r['prediction'] in ('D', 'E', 'U'), 'V261 exact saved correspondence')
        predictions[arm] = [r['prediction'] for r in rows]

    cohorts = {'PRIMARY': [r for r in ledger if primary(r)], 'ALL_REGISTERED_D_E': [r for r in ledger if r['human_state'] in (D, E)],
               'MANUAL_CLICK_D_ALL': [r for r in ledger if manual_d(r)], 'MANUAL_CLICK_D_LE3_VALID_INPUT': [r for r in ledger if near_d(r)]}
    classification, transitions, self_rows = [], [], []
    for group in GROUPS:
        for cohort, rows in cohorts.items():
            take = [r for r in rows if group == 'ALL' or r['group'] == group]
            for arm, pp in predictions.items():
                classification.append(dict(group=group, cohort=cohort, model=arm, **cm(take, pp)))
            t = transition(take, predictions['GLOBAL_MATCHED'], predictions['HEATMAP_LOCAL'])
            transitions.append(dict(group=group, cohort=cohort, **t))
        take = [r for r in ledger if r['human_state'] == S and (group == 'ALL' or r['group'] == group)]
        for arm, pp in predictions.items():
            count = Counter(pp[int(r['global_index'])] for r in take)
            self_rows.append(dict(group=group, model=arm, N=len(take), response_D=count['D'], response_E=count['E'], response_U=count['U'],
                                  interpretation='Human SELF kept separate; binary D/E response is not learned SELF correctness'))

    # Independent exact baseline count replay, including the unbalanced near-D cohort.
    old_metrics = list(csv.DictReader(io.StringIO(checked(PINNED['old_classification']).decode('utf-8-sig'))))
    old_names = {'JOINT_SELECTED': 'V263_JOINT', 'VIS_ONLY_SELECTED': 'V263_VIS'}
    checked_count = 0
    for old in old_metrics:
        model = old_names.get(old['model'], old['model'])
        if old.get('status') or model not in predictions:
            continue
        matches = [r for r in classification if (r['group'], r['cohort'], r['model']) == (old['group'], old['cohort'], model)]
        need(len(matches) == 1, 'Exact baseline metric correspondence')
        for k in ('N', 'truth_D', 'truth_E', 'D_D', 'D_E', 'D_U', 'E_D', 'E_E', 'E_U', 'U'):
            need(matches[0][k] == int(old[k]), 'Baseline count replay failed: ' + model + '/' + k)
        checked_count += 1

    selection = []
    for group in GROUPS[1:]:
        for target in ('D', 'E'):
            for cell in CELLS:
                eligible = []
                for row in cohorts['PRIMARY']:
                    if row['group'] != group or truth(row) != target:
                        continue
                    i = int(row['global_index'])
                    gc, lc = predictions['GLOBAL_MATCHED'][i] == target, predictions['HEATMAP_LOCAL'][i] == target
                    name = 'BOTH_CORRECT' if gc and lc else 'LOCAL_ONLY_CORRECT' if lc else 'GLOBAL_ONLY_CORRECT' if gc else 'BOTH_WRONG'
                    if cell == name:
                        eligible.append(i)
                selection.append(dict(group=group, truth=target, cell=cell, census_N=len(eligible), empty=not eligible,
                                      selected_global_index=min(eligible) if eligible else None,
                                      selection='Minimum original global_index within fixed result cell, before current RGB reads'))
    selection_binding = write_json(out / 'CASE_SELECTION_BEFORE_RGB.json', dict(completed=True, cells=selection, current_RGB_reads_so_far=0,
                                    caveat='Post-result diagnostic selection; not independent validation, training, or model selection. Empty cells preserved.'))
    metrics_binding = write_csv(out / 'CLASSIFICATION.csv', classification)
    self_binding = write_csv(out / 'SELF_RESPONSES.csv', self_rows)
    transitions_binding = write_json(out / 'PAIRED_TRANSITIONS.json', transitions)
    transition_csv = write_csv(out / 'PAIRED_TRANSITIONS.csv', [{k: v for k, v in r.items() if k != 'response_transitions'} for r in transitions])
    joined = []
    for r in ledger:
        i = int(r['global_index'])
        row = {k: r[k] for k in ('global_index', 'row_id', 'id', 'native_corner', 'group', 'human_state', 'annotation_coordinate_source', 'initial_reference_error_px', 'original_query_valid')}
        row['PRIMARY'] = primary(r)
        row['MANUAL_D_Q0_LE3'] = near_d(r)
        for arm, pp in predictions.items():
            row[arm + '_prediction'] = pp[i]
        for arm in ARMS:
            row[arm + '_E_logit'] = new[i][arm]['E_logit']
            row[arm + '_p_E'] = new[i][arm]['p_E']
        joined.append(row)
    joined_binding = write_csv(out / 'ALL1784_COMPARISON.csv', joined)
    return ledger, classification, transitions, self_rows, selection, dict(classification=metrics_binding, self_responses=self_binding,
        transitions=transitions_binding, transitions_csv=transition_csv, joined=joined_binding, selection=selection_binding,
        baseline_count_rows_replayed=checked_count, V261_saved_predictions=baseline_bindings)


FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'


def font(size):
    return ImageFont.truetype(FONT, size)


def wrap(draw, text, xy, max_width, size=23, fill='#172238', spacing=7):
    f = font(size)
    line, y = '', xy[1]
    for char in str(text):
        test = line + char
        if line and draw.textbbox((0, 0), test, font=f)[2] > max_width:
            draw.text((xy[0], y), line, font=f, fill=fill)
            line, y = char, y + size + spacing
        else:
            line = test
    if line:
        draw.text((xy[0], y), line, font=f, fill=fill)
    return y + size + spacing


def marker(draw, xy, color, kind='cross', radius=8):
    x, y = xy
    if not (-20 <= x <= 420 and -20 <= y <= 420):
        return
    if kind == 'circle':
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), outline=color, width=3)
    else:
        draw.line((x-radius, y, x+radius, y), fill=color, width=3)
        draw.line((x, y-radius, x, y+radius), fill=color, width=3)


def rgb_array(arr):
    arr = np.asarray(arr)
    need(arr.shape in ((3, 96, 96), (96, 96, 3)), 'Saved crop96 RGB shape')
    if arr.shape == (3, 96, 96):
        arr = np.moveaxis(arr, 0, -1)
    need(arr.dtype == np.uint8, 'Immutable saved crop must be uint8 RGB')
    return np.ascontiguousarray(arr)


def crop_point(row, xy):
    if xy is None:
        return None
    qx, qy = num(row['query_x']), num(row['query_y'])
    side = num(row['JOINT_SELECTED__crop_L'])
    if any(v is None for v in (qx, qy, side)) or side <= 0:
        return None
    return [(float(xy[0])-qx) * 96/side + 47.5, (float(xy[1])-qy) * 96/side + 47.5]


def image_pair(rgb, heatmap, row):
    clean = Image.fromarray(rgb, 'RGB').resize((400, 400), Image.Resampling.NEAREST)
    h = np.asarray(heatmap, dtype=np.float64)
    need(h.shape == (16, 16) and np.isfinite(h).all(), 'Frozen heatmap16x16 finite')
    scaled = (h-h.min()) / (h.max()-h.min()) if h.max() > h.min() else np.zeros_like(h)
    # Viridis is only a display transform. No classifier decision uses this scaling.
    from matplotlib import colormaps
    rgba = (colormaps['viridis'](scaled)[..., :3] * 255).astype(np.uint8)
    color = Image.fromarray(rgba, 'RGB').resize((400, 400), Image.Resampling.BILINEAR)
    overlay = Image.blend(clean, color, 0.48)
    q = [(47.5 + .5) * 400/96 - .5] * 2
    proposal = json.loads(row['JOINT_SELECTED__LOC_proposed_xy']) if row['JOINT_SELECTED__LOC_proposed_xy'] else None
    ref = [num(row['human_reference_x']), num(row['human_reference_y'])]
    ref = ref if all(v is not None for v in ref) else None
    for im in (clean, overlay):
        draw = ImageDraw.Draw(im)
        marker(draw, q, '#00e5ff', 'circle')
        for xy, color_ in ((crop_point(row, proposal), '#fff33e'), (crop_point(row, ref), '#ff3cd4')):
            if xy is not None:
                pixel = [(v + .5)*400/96 - .5 for v in xy]
                marker(draw, pixel, color_)
    return clean, overlay


CELL_KO = {'BOTH_CORRECT': '둘 다 주석과 일치', 'LOCAL_ONLY_CORRECT': '위치 주변 방식만 일치', 'GLOBAL_ONLY_CORRECT': '전체 평균 방식만 일치', 'BOTH_WRONG': '둘 다 주석과 불일치'}


def card_for_case(cell, row, pred, rgb, heatmap, out, suffix='', caution=None):
    im = Image.new('RGB', (1000, 760), '#ffffff')
    draw = ImageDraw.Draw(im)
    if cell is not None:
        heading = ('목재' if cell['group'] == 'WOOD_PRIMARY105' else '정사각형') + ' · 주석 ' + cell['truth'] + ' · ' + CELL_KO[cell['cell']]
        draw.text((28, 15), heading, font=font(27), fill='#172238')
        draw.text((28, 58), '이 셀 전체 ' + str(cell['census_N']) + '점 · 최소 원본 순번 사례', font=font(22), fill='#566578')
    else:
        draw.text((28, 15), '사용자 피드백 · 단정적 오류 해석을 피할 사례', font=font(27), fill='#172238')
    draw.text((28, 98), 'G' + str(pred['global_index']) + ' · ' + row['row_id'], font=font(24), fill='#172238')
    for x, arm, label in ((28, 'GLOBAL_MATCHED', '전체 평균'), (510, 'HEATMAP_LOCAL', '위치 주변')):
        z = pred[arm]
        ps = '—' if z['p_E'] is None else f"{z['p_E']:.3f}"
        draw.text((x, 140), f"{label}: {z['pred']}  /  P(E)={ps}", font=font(26), fill='#172238')
    raw, overlay = image_pair(rgb, heatmap, row)
    im.paste(raw, (50, 205)); im.paste(overlay, (550, 205))
    draw.text((50, 610), '실제 저장 crop · 표시만 추가', font=font(20), fill='#566578')
    draw.text((550, 610), '고정 JOINT heatmap · min/max 표시', font=font(20), fill='#566578')
    text = caution or ('기존 주석: ' + row['human_state'] + ' / 좌표 출처: ' + row['annotation_coordinate_source'])
    wrap(draw, text, (28, 650), 944, size=22)
    filename = f"CASE_G{pred['global_index']:04d}{suffix}.png"
    p = safe(out / filename, True); im.save(p)
    raw_path = safe(out / f"CLEAN_G{pred['global_index']:04d}{suffix}.png", True)
    Image.fromarray(rgb, 'RGB').save(raw_path)
    return dict(card=bind(p), clean=bind(raw_path), RGB96_HWC_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(), global_index=pred['global_index'],
                row_id=row['row_id'], human_state=row['human_state'], annotation_coordinate_source=row['annotation_coordinate_source'],
                GLOBAL_MATCHED=pred['GLOBAL_MATCHED'], HEATMAP_LOCAL=pred['HEATMAP_LOCAL'], caution=caution)


def empty_card(cell):
    im = Image.new('RGB', (1000, 760), '#f5f7fa'); d = ImageDraw.Draw(im)
    group = '목재' if cell['group'] == 'WOOD_PRIMARY105' else '정사각형'
    wrap(d, group + ' · 주석 ' + cell['truth'] + '\n' + CELL_KO[cell['cell']], (40, 50), 900, 32)
    d.text((40, 320), '해당 셀 0점', font=font(48), fill='#6a7688')
    d.text((40, 400), '대체 사례를 고르지 않고 빈 셀을 유지합니다.', font=font(25), fill='#6a7688')
    return im


def percentage(value):
    return '—' if value is None else f'{100 * value:.1f}'


def table_image(rows, out):
    take = [r for r in rows if r['group'] == 'ALL' and r['cohort'] == 'PRIMARY']
    im = Image.new('RGB', (1800, 860), 'white'); d = ImageDraw.Draw(im)
    d.text((35, 25), 'V264 · 보임/외부 가림 · 동일 1,237점 비교', font=font(42), fill='#172238')
    d.text((35, 90), 'D=보임  E=외부 가림  U=보류 · 정답 D 1,147 / E 90 · 단위 %', font=font(25), fill='#566578')
    columns = [('model', '모델', 35), ('D_recall', 'D recall', 460), ('E_precision', 'E precision', 700), ('E_recall', 'E recall', 950), ('E_F1', 'E F1', 1200), ('U_rate', '보류율', 1420), ('balanced_recall', '균형 recall', 1570)]
    y = 160
    for key, label, x in columns: d.text((x, y), label, font=font(24), fill='#172238')
    for row in take:
        y += 64
        if row['model'] in ARMS: d.rounded_rectangle((20, y-5, 1780, y+52), 8, fill='#eaf4ff')
        for key, label, x in columns:
            text_ = row[key] if key == 'model' else percentage(row[key])
            d.text((x, y), text_, font=font(25), fill='#172238')
    wrap(d, '현재 223장은 반복 사용한 개발 평가입니다. 새 두 방식은 합성 학습·동일 클래스 균형 표본 조건이며, 기존 모델과의 차이에는 학습 조건 차이도 포함됩니다.', (35, 760), 1730, 23)
    p = safe(out / 'PRIMARY_COMPARISON_KO.png', True); im.save(p)
    return bind(p)


def visualize(out, ledger, new, selection, metrics):
    complete = json.loads(checked(PINNED['v263_current_complete']))
    crop_binding = complete['crop_cache']['binding']
    heatmap_binding = complete['heatmaps']['JOINT_SELECTED']['binding']
    # Selection is durably written before either of these current RGB reads.
    crop_npz = np.load(io.BytesIO(checked(crop_binding)), allow_pickle=False)
    heatmap_npz = np.load(io.BytesIO(checked(heatmap_binding)), allow_pickle=False)
    need('RGB96_initial_geometry' in crop_npz.files and 'valid96_initial_geometry' in crop_npz.files, 'Exact saved initial-geometry crop members')
    rgbs = crop_npz['RGB96_initial_geometry']; valid = crop_npz['valid96_initial_geometry']
    hm = heatmap_npz[complete['heatmaps']['JOINT_SELECTED']['probability_member']]
    need(rgbs.shape[0] == valid.shape[0] == 1784 and hm.shape == (1784, 16, 16), 'Saved RGB/heatmap order1784')
    cards, sheets = [], []
    for group in GROUPS[1:]:
        cells = [s for s in selection if s['group'] == group]
        sheet = Image.new('RGB', (2000, 3040), '#e8edf3')
        for j, cell in enumerate(cells):
            if cell['empty']:
                card = empty_card(cell)
            else:
                i = cell['selected_global_index']
                binding = card_for_case(cell, ledger[i], new[i], rgb_array(rgbs[i]), hm[i], out)
                cards.append(dict(selection=cell, **binding))
                with Image.open(binding['card']['path']) as loaded:
                    card = loaded.copy()
            sheet.paste(card, ((j % 2) * 1000, (j // 2) * 760))
        name = ('WOOD' if group == 'WOOD_PRIMARY105' else 'SQUARE') + '_FIXED8_CELLS_KO.png'
        p = safe(out / name, True); sheet.save(p); sheets.append(bind(p))
    cautions = [
        (996, '기존 E 주석과 사용자 시각적 판단이 다릅니다. 명백한 외부 가림 놓침으로 단정하지 않습니다. 원래 주석은 유지하며 주석/대응 재검토 대상으로 분리합니다.'),
        (981, 'V263 위치 오차 1.46→3.65px: 3px 기준 초과인 작은 이동입니다. 현재 가림 판정의 큰 실패를 입증하는 사례가 아닙니다.'),
        (200, 'V263 위치 오차 1.77→4.57px: 3px 기준 초과인 작은 이동입니다. 현재 가림 판정의 큰 실패를 입증하는 사례가 아닙니다.'),
    ]
    for i, caution in cautions:
        cards.append(dict(selection=None, **card_for_case(None, ledger[i], new[i], rgb_array(rgbs[i]), hm[i], out, suffix='_CAUTION', caution=caution)))
    table = table_image(metrics, out)
    case_binding = write_json(out / 'VISUAL_CASE_DATA.json', dict(completed=True, cases=cards, selection_rule='Fixed16 cells x minimum original index; selection persisted before RGB. Three user-feedback examples are separately marked.',
        crop_source=crop_binding, heatmap_source=heatmap_binding, no_new_location_prediction=True,
        heatmap_display='Per-crop min/max normalized probability; display brightness is not visibility confidence.',
        marker_legend=dict(cyan_circle='original query', yellow_cross='frozen V263 JOINT location proposal', magenta_cross='inherited annotation coordinate; E coordinate is not direct visible-point truth')))
    return dict(case_data=case_binding, contact_sheets=sheets, comparison_table=table, crop_source=crop_binding, heatmap_source=heatmap_binding), cards


def report(out, classification, transitions, visuals, cards):
    take = [r for r in classification if r['group'] == 'ALL' and r['cohort'] == 'PRIMARY']
    headers = ['모델', 'D→D/E/U', 'E→D/E/U', 'D precision', 'D recall', 'E precision', 'E recall', 'E F1', '보류율']
    tr = []
    for row in take:
        vals = [row['model'], f"{row['D_D']}/{row['D_E']}/{row['D_U']}", f"{row['E_D']}/{row['E_E']}/{row['E_U']}",
                *[percentage(row[k]) for k in ('D_precision', 'D_recall', 'E_precision', 'E_recall', 'E_F1', 'U_rate')]]
        tr.append('<tr>' + ''.join('<td>' + html.escape(str(v)) + '</td>' for v in vals) + '</tr>')
    transition_all = next(r for r in transitions if r['group'] == 'ALL' and r['cohort'] == 'PRIMARY')
    parts = ['<!doctype html><html lang="ko"><meta charset="utf-8"><title>V264 위치 주변 특징 가림 판별 비교</title>',
             '<style>body{font-family:sans-serif;background:#f5f7fb;color:#172238;max-width:1500px;margin:30px auto;padding:0 24px;line-height:1.6}table{border-collapse:collapse;background:white;width:100%}td,th{padding:8px;border:1px solid #dce3ef}img{max-width:100%;height:auto}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.card{background:white;padding:12px;border-radius:10px}a{color:#165ec9}.note{background:#fff4cf;padding:16px}h1,h2{line-height:1.3}</style>',
             '<h1>Heatmap이 찾은 위치 주변 특징으로 보임/외부 가림 판단</h1>',
             '<p>합성으로 학습한 두 새 분류기: <b>GLOBAL_MATCHED</b>는 crop 전체 특징 평균, <b>HEATMAP_LOCAL</b>은 고정 heatmap 위치 주변 특징을 이용합니다. 동일 클래스 균형 학습·고정 logit 0 기준입니다. 실사 223장은 평가에만 사용했습니다.</p>',
             '<p class="note">223장은 이전에도 확인한 개발 평가입니다. 새 두 방식의 비교가 이번 구조 차이의 대조군입니다. V263·v61·기존 모델과의 차이에는 학습 구조·표본 조건의 차이도 포함됩니다. 기존 모델의 과거 실사 학습 이력은 그대로입니다. 자기 가림은 외부 가림과 합치지 않습니다.</p>',
             '<h2>동일 주 평가 1,237점</h2><p>주석 D 1,147 / E 90. 혼동행렬 각 행의 열 순서는 D/E/U입니다. 비율은 %이며, recall 분모에 보류를 포함합니다.</p>',
             '<table><thead><tr>' + ''.join('<th>'+s+'</th>' for s in headers) + '</tr></thead><tbody>'+''.join(tr)+'</tbody></table>',
             f"<p>새 두 방식의 짝 비교: 둘 다 일치 {transition_all['BOTH_CORRECT']}, 위치 주변만 일치 {transition_all['LOCAL_ONLY_CORRECT']}, 전체 평균만 일치 {transition_all['GLOBAL_ONLY_CORRECT']}, 둘 다 불일치 {transition_all['BOTH_WRONG']}점. 순변화 {transition_all['net_correct']:+d}점.</p>",
             '<p><a href="CLASSIFICATION.csv">재질·평가군별 전체 혼동행렬/precision/recall/F1/보류율 CSV</a> · <a href="SELF_RESPONSES.csv">자기 가림 별도 응답</a> · <a href="PAIRED_TRANSITIONS.json">짝 비교 전이</a> · <a href="ALL1784_COMPARISON.csv">1,784점 행별 결과</a> · <a href="SUMMARY.json">전체 결과 JSON</a></p>',
             '<h2>되는 경우와 안 되는 경우: 고정 16개 셀</h2>',
             '<p>재질 × 정답 D/E × 두 방식의 정오 조합에서 최소 원본 순번을 선택했습니다. RGB를 읽기 전에 사례 목록을 저장했습니다. 셀 전체 개수를 표시하며 빈 셀을 대체하지 않았습니다. 대표 이미지가 전체 빈도를 대신하지 않습니다.</p>',
             '<p>파랑 ○ 초기 위치 / 노랑 + 고정 JOINT 위치 후보 / 분홍 + 기존 주석 좌표. 가림 점의 분홍 좌표는 직접 보이는 위치의 정답을 뜻하지 않습니다. Heatmap은 crop별 min/max 표시이며 밝기는 보임 확률이 아닙니다. P(E)는 별도 분류기 출력입니다.</p>',
             '<div class="grid">']
    for card in cards:
        if card['selection'] is None:
            continue
        parts.append('<div class="card"><a href="'+Path(card['card']['path']).name+'"><img loading="lazy" src="'+Path(card['card']['path']).name+'"></a><a href="'+Path(card['clean']['path']).name+'">표시 없는 실제 crop</a></div>')
    parts.append('</div><p>전체 셀: <a href="WOOD_FIXED8_CELLS_KO.png">목재 8셀</a> · <a href="SQUARE_FIXED8_CELLS_KO.png">정사각형 8셀</a> · <a href="CASE_SELECTION_BEFORE_RGB.json">사례 선택/빈 셀 목록</a></p><h2>사용자 피드백을 반영한 재검토 사례</h2><div class="grid">')
    for card in cards:
        if card['selection'] is not None:
            continue
        parts.append('<div class="card"><img loading="lazy" src="'+Path(card['card']['path']).name+'"><p>'+html.escape(card['caution'])+'</p></div>')
    parts.append('</div><p>새 학습·threshold 검색·PnP는 이 평가 스크립트에 없습니다. 위치 출력은 V263에서 고정된 것을 표시합니다.</p></html>')
    p = safe(out / 'INDEX_KO.html', True)
    with p.open('x', encoding='utf-8') as f:
        f.write('\n'.join(parts))
    return bind(p)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=str(DEFAULT_ROOT))
    parser.add_argument('--out', default='evaluation_v1')
    args = parser.parse_args()
    root = safe(args.root)
    need(Path(args.out).name == args.out and '..' not in args.out, 'Single output directory name only')
    out = safe(root / args.out, fresh=True)
    new, seals = load_new(root)
    out.mkdir()
    consumer = write_json(out / 'CONSUMER_SEAL_BEFORE_LATE_GT_AND_RGB.json', dict(completed=True, PASS=True, **seals,
        all1784_predictions_validated=True, late_GT_baseline_or_current_RGB_reads=0, created_UTC=dt.datetime.now(dt.timezone.utc).isoformat()))
    ledger, classification, transitions, self_rows, selection, outputs = evaluate(root, out, new, seals)
    visuals, cards = visualize(out, ledger, new, selection, classification)
    summary = dict(completed=True, PASS=True, primary_counts=dict(N=1237, D=1147, E=90), rows=1784, frames=223,
        current223_is_repeated_development_evaluation=True, fresh_blind_evaluation=False,
        no_current_annotations_used_for_new_training=True, no_new_threshold_or_PnP_or_location_changes=True,
        threshold=dict(E_logit=0, E_if='logit > 0', U='original input unavailable only'),
        new_head_training_context='Both new heads use synthetic data and matched class-balanced sampling; the new GLOBAL_MATCHED is the causal architecture control. Old head differences include training/architecture changes.',
        confusion_order=dict(rows=['D', 'E'], columns=['D', 'E', 'U']),
        U_interpretation='Original unavailable input, not a calibrated confidence threshold. Recall denominators include U.',
        classification=classification, paired_transitions=transitions, human_SELF=self_rows,
        case_selection=selection, consumer_pre_late_GT_seal=consumer, inputs=PINNED, seals=seals, outputs=outputs, visuals=visuals,
        model_checkpoint_or_encoder_reads=0, optimizer_updates=0, solver_calls=0)
    summary_binding = write_json(out / 'SUMMARY.json', summary)
    report_binding = report(out, classification, transitions, visuals, cards)
    complete = write_json(out / 'COMPLETE.json', dict(completed=True, PASS=True, code=bind(Path(__file__).absolute()),
        consumer_pre_late_GT_seal=consumer, summary=summary_binding, report=report_binding, outputs=outputs, visuals=visuals,
        created_UTC=dt.datetime.now(dt.timezone.utc).isoformat()))
    print(json.dumps(dict(completed=True, output=str(out), COMPLETE=complete), ensure_ascii=False))


if __name__ == '__main__':
    main()
