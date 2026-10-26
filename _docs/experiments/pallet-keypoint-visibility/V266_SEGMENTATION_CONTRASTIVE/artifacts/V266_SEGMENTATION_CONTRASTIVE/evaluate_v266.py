#!/usr/bin/env python3
"""Saved-output-only three-arm evaluation and dense-map diagnostic gallery.

Run only after CURRENT_PREDICTIONS_FROZEN.json seals all 1,784 predictions.
Portable metrics replay: --replay-metrics --inputs evaluation_v1/compact_inputs
                         --out metrics_replay_v1
No model, encoder, checkpoint, training, solver or operating-rule selection.
"""
from pathlib import Path
import argparse
from collections import Counter
import csv
import datetime as dt
import hashlib
import html
import io
import json
import math
import os
import stat
import sys
import types

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W = Path(__file__).absolute().parent
R = W.parent
ARMS = ('CURRENT_V264', 'SEGMENTATION', 'SEGMENTATION_CONTRAST')
GROUPS = ('ALL', 'WOOD_PRIMARY105', 'SQUARE118')
CLASS_NAMES = ('TARGET_PALLET', 'FOREIGN_OBJECT', 'BACKGROUND')
EVALUATOR_SHA = 'e9408144ba952789e487a02f1d4e4d34a9b39c202f25113fc32224e569e2ffeb'
EVALUATOR_PATH = R / 'V264_HEATMAP_LOCAL_VISIBILITY/evaluate_and_visualize_v264.py'
PRIOR_PATH = R / 'V264_HEATMAP_LOCAL_VISIBILITY/CURRENT_ALL1784_PREDICTIONS.jsonl'
PRIOR_SHA = '46fc19cfa3aee116075869382073e7e488c5e604538d233d1070682f2d4c2736'
FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'


def need(value, message):
    if not bool(value):
        raise ValueError(message)


def safe(value, fresh=False):
    p = Path(value)
    need(p.is_absolute() and '..' not in p.parts, 'Absolute nontraversal path required')
    need(not any(t in str(p).casefold() for t in ('v112', 'cache', '.pyc')), 'Forbidden lexical path')
    for a in [*reversed(p.parents), p]:
        try:
            mode = os.lstat(a).st_mode
        except FileNotFoundError:
            continue
        need(not stat.S_ISLNK(mode), 'Symlink or symlink ancestor: ' + str(a))
    if fresh:
        need(not p.exists(), 'Immutable output exists: ' + str(p))
    return p


def raw(path):
    return safe(path).read_bytes()


def bind(path):
    p = safe(path)
    b = p.read_bytes()
    return dict(path=str(p), sha256=hashlib.sha256(b).hexdigest(), size_bytes=len(b))


def checked(binding):
    need(isinstance(binding, dict) and all(k in binding for k in ('path', 'sha256', 'size_bytes')), 'Complete binding required')
    b = raw(binding['path'])
    need(len(b) == binding['size_bytes'] and hashlib.sha256(b).hexdigest() == binding['sha256'], 'SHA/size mismatch: ' + binding['path'])
    return b


def dump(path, value):
    p = safe(path, True)
    with p.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    return bind(p)


def table(path, rows):
    p = safe(path, True)
    need(bool(rows), 'Nonempty output table required')
    with p.open('x', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)
    return bind(p)


def jsonl(data):
    return [json.loads(line) for line in data.splitlines() if line.strip()]


def load_evaluator(path=EVALUATOR_PATH):
    p = safe(path)
    b = p.read_bytes()
    need(hashlib.sha256(b).hexdigest() == EVALUATOR_SHA, 'Pinned V264 evaluator SHA mismatch')
    module = types.ModuleType('v266_pinned_v264_evaluator')
    module.__file__ = str(p)
    sys.modules[module.__name__] = module
    exec(compile(b, str(p), 'exec'), module.__dict__)
    return module


def validate_predictions(predictions):
    need(len(predictions) == 1784, 'Exactly 1784 predictions required')
    for i, row in enumerate(predictions):
        need(row['global_index'] == i and row['native_slot'] == i % 8, 'Prediction order/native mismatch')
        need(row['row_id'] == row['frame_id'] + ':native' + str(i % 8), 'Prediction row identity mismatch')
        need(type(row['available']) is bool, 'Boolean availability required')
        for arm in ARMS:
            a = row[arm]
            z, p = float(a['E_logit']), float(a['p_E'])
            need(math.isfinite(z) and math.isfinite(p) and 0 <= p <= 1, 'Finite E logit/probability')
            expected = 1 / (1 + math.exp(-max(-700, min(700, z))))
            need(abs(p - expected) <= 2e-6, 'p_E must equal sigmoid(E_logit)')
            need(a['pred'] == ('E' if z > 0 else 'D') if row['available'] else a['pred'] == 'U',
                 'Fixed logit0 / original-unavailable-only U')
    need(sum(r['available'] for r in predictions) == 1739, 'Same supported1739 and unsupported45')


def freeze_inputs():
    """No annotation or pixel reads before this returns."""
    seal_path = safe(W / 'CURRENT_PREDICTIONS_FROZEN.json')
    seal = json.loads(raw(seal_path))
    need(seal.get('completed') is True and seal.get('PASS', True) is True, 'Prediction freeze incomplete')
    config = json.loads(checked(seal['configuration']))
    need(config.get('completed') is True or config.get('PASS') is True, 'Configuration not frozen')
    pred_binding = seal['predictions']
    dense_binding = seal['dense_predictions']
    need(Path(pred_binding['path']) == W / 'CURRENT_ALL1784_PREDICTIONS.jsonl', 'Exact prediction payload required')
    need(Path(dense_binding['path']) == W / 'current_inference_v1/DENSE_PROBABILITIES.npz', 'Exact dense payload required')
    predictions = jsonl(checked(pred_binding))
    validate_predictions(predictions)
    # Verify integrity before late GT reads; dense values are model outputs, not RGB.
    dense_bytes = checked(dense_binding)
    with np.load(io.BytesIO(dense_bytes), allow_pickle=False) as z:
        need(all(arm in z.files for arm in ARMS[1:]), 'Two learned dense probability arrays required')
        for arm in ARMS[1:]:
            a = z[arm]
            need(a.shape == (1784, 3, 16, 16), 'Dense order1784/classes3/grid16')
            need(np.isfinite(a).all() and (a >= -1e-6).all() and (a <= 1 + 1e-6).all(), 'Dense values must be probabilities')
            need(np.max(np.abs(a.sum(1) - 1)) <= .003, 'Dense classes must sum to one')
    return predictions, dict(configuration=seal['configuration'], prediction_freeze=bind(seal_path),
                             predictions=pred_binding, dense_predictions=dense_binding)


def validate_prior(new, prior):
    need(len(prior) == 1784, 'Saved V264 prior1784')
    max_diff = 0.
    for i, (a, b) in enumerate(zip(new, prior)):
        need(a['global_index'] == b['global_index'] == i and a['row_id'] == b['row_id'], 'Exact prior row join')
        need(a['available'] == b['available'], 'Prior availability preserved')
        p, q = a['CURRENT_V264'], b['HEATMAP_LOCAL']
        need(p['pred'] == q['pred'], 'CURRENT_V264 must reuse prior decisions exactly')
        max_diff = max(max_diff, abs(float(p['E_logit']) - float(q['E_logit'])))
        need(abs(float(p['p_E']) - float(q['p_E'])) <= 2e-6, 'Prior probability reuse')
    need(max_diff <= 1e-6, 'CURRENT_V264 saved logit reuse required')
    return max_diff


def metrics(e, ledger, predictions):
    need(len(ledger) == 1784, 'Ledger1784')
    for i, (row, p) in enumerate(zip(ledger, predictions)):
        need(int(row['global_index']) == i and row['row_id'] == p['row_id'], 'Ledger/prediction exact join')
        need((row['original_query_valid'] == 'True') == p['available'], 'Same original query availability')
    cohorts = {
        'PRIMARY': [r for r in ledger if e.primary(r)],
        'ALL_REGISTERED_D_E': [r for r in ledger if r['human_state'] in (e.D, e.E)],
        'MANUAL_CLICK_D_ALL': [r for r in ledger if e.manual_d(r)],
        'MANUAL_CLICK_D_LE3_VALID_INPUT': [r for r in ledger if e.near_d(r)],
    }
    need(len(cohorts['PRIMARY']) == 1237 and sum(r['human_state'] == e.D for r in cohorts['PRIMARY']) == 1147, 'PRIMARY1237 D1147/E90')
    need(len(cohorts['MANUAL_CLICK_D_LE3_VALID_INPUT']) == 376, 'Near-manualD376')
    pp = {arm: [r[arm]['pred'] for r in predictions] for arm in ARMS}
    classification, self_rows, transitions = [], [], []
    for group in GROUPS:
        for cohort, rows in cohorts.items():
            take = [r for r in rows if group == 'ALL' or r['group'] == group]
            for arm in ARMS:
                classification.append(dict(group=group, cohort=cohort, model=arm, **e.cm(take, pp[arm])))
            for arm in ARMS[1:]:
                transitions.append(dict(group=group, cohort=cohort, model=arm,
                                        **e.transition(take, pp['CURRENT_V264'], pp[arm])))
        take = [r for r in ledger if r['human_state'] == e.S and (group == 'ALL' or r['group'] == group)]
        for arm in ARMS:
            count = Counter(pp[arm][int(r['global_index'])] for r in take)
            self_rows.append(dict(group=group, model=arm, N=len(take),
                response_D=count['D'], response_E=count['E'], response_U=count['U'],
                interpretation='Human SELF kept separate; binary D/E response is not learned SELF correctness'))
    external = [int(r['global_index']) for r in cohorts['PRIMARY'] if r['human_state'] == e.E]
    prior_misses = [i for i in external if pp['CURRENT_V264'][i] == 'D']
    need(len(external) == 90 and len(prior_misses) == 67, 'Prior E90/missed67 invariant')
    external_summary = []
    external_rows = []
    for i in external:
        row = dict(global_index=i, row_id=ledger[i]['row_id'], group=ledger[i]['group'],
                   annotation_coordinate_source=ledger[i]['annotation_coordinate_source'],
                   prior_missed=i in prior_misses, user_disputed_annotation=i == 996,
                   cone_membership='NOT_INFERRED_FROM_EXTERNAL_LABEL',
                   CURRENT_V264=pp['CURRENT_V264'][i])
        for arm in ARMS[1:]:
            p = pp[arm][i]
            row[arm] = p
            row[arm + '_transition'] = ('RECOVERED_PRIOR_MISS' if p == 'E' else 'REMAINED_PRIOR_MISS') if i in prior_misses else ('KEPT_PRIOR_CORRECT' if p == 'E' else 'REGRESSED_PRIOR_CORRECT')
        external_rows.append(row)
    for group in GROUPS:
        rows = [r for r in external_rows if group == 'ALL' or r['group'] == group]
        for arm in ARMS[1:]:
            count = Counter(r[arm + '_transition'] for r in rows)
            external_summary.append(dict(group=group, model=arm, N=len(rows),
                prior_misses=sum(r['prior_missed'] for r in rows),
                **{key: count[key] for key in ('RECOVERED_PRIOR_MISS', 'REMAINED_PRIOR_MISS', 'KEPT_PRIOR_CORRECT', 'REGRESSED_PRIOR_CORRECT')}))
    # All eight correctness patterns are retained, including empty cells.
    selection = []
    for group in GROUPS[1:]:
        for truth in ('D', 'E'):
            for bits in ('000', '001', '010', '011', '100', '101', '110', '111'):
                ids = []
                for row in cohorts['PRIMARY']:
                    i = int(row['global_index'])
                    if row['group'] != group or e.truth(row) != truth:
                        continue
                    pattern = ''.join('1' if pp[a][i] == truth else '0' for a in ARMS)
                    if pattern == bits:
                        ids.append(i)
                selection.append(dict(group=group, truth=truth, correctness_pattern=bits,
                    census_N=len(ids), selected_global_index=min(ids) if ids else None,
                    empty=not ids))
    return dict(classification=classification, human_SELF=self_rows, paired_transitions=transitions,
                external90_transitions=external_summary, external90_rows=external_rows,
                fixed_pattern_selection=selection, primary_counts=dict(N=1237, D=1147, E=90),
                prior_missed_external_count=67)


def write_metrics(out, result, ledger, new):
    classification = table(out / 'CLASSIFICATION.csv', result['classification'])
    self_rows = table(out / 'SELF_RESPONSES.csv', result['human_SELF'])
    external = table(out / 'EXTERNAL90_TRANSITIONS.csv', result['external90_rows'])
    external_summary = table(out / 'EXTERNAL90_TRANSITION_SUMMARY.csv', result['external90_transitions'])
    transitions = dump(out / 'PAIRED_TRANSITIONS.json', result['paired_transitions'])
    joined = []
    for row, p in zip(ledger, new):
        r = {k: row[k] for k in ('global_index', 'row_id', 'id', 'native_corner', 'group', 'human_state',
              'annotation_coordinate_source', 'initial_reference_error_px', 'original_query_valid')}
        r['PRIMARY'] = row['paired_same_query_RGB_D_vs_E'] == 'True' and row['human_state'] in ('DIRECT_VISIBLE', 'EXTERNAL_OCCLUDED')
        for arm in ARMS:
            r[arm + '_prediction'] = p[arm]['pred']
            r[arm + '_E_logit'] = p[arm]['E_logit']
            r[arm + '_p_E'] = p[arm]['p_E']
        joined.append(r)
    return dict(classification=classification, self_responses=self_rows,
                external90=external, external_summary=external_summary, transitions=transitions,
                joined=table(out / 'ALL1784_COMPARISON.csv', joined))


def compact_inputs(out, e, ledger_bytes, prediction_bytes, prior_bytes):
    dest = safe(out / 'compact_inputs', True)
    dest.mkdir()
    payloads = {'LEDGER.csv': ledger_bytes, 'NEW_PREDICTIONS.jsonl': prediction_bytes,
                'V264_PRIOR_PREDICTIONS.jsonl': prior_bytes,
                'V264_EVALUATOR_CODE.py': raw(EVALUATOR_PATH)}
    integrity = {}
    for name, b in payloads.items():
        p = safe(dest / name, True)
        p.write_bytes(b)
        integrity[name] = dict(sha256=hashlib.sha256(b).hexdigest(), size_bytes=len(b))
    return dump(dest / 'INTEGRITY.json', dict(completed=True, files=integrity,
        evaluator_SHA=EVALUATOR_SHA, no_RGB_or_model_required=True,
        replay='python -B evaluate_v266.py --replay-metrics --inputs evaluation_v1/compact_inputs --out metrics_replay_v1'))


def replay(inputs, out):
    integrity = json.loads(raw(inputs / 'INTEGRITY.json'))
    payloads = {}
    for name in ('LEDGER.csv', 'NEW_PREDICTIONS.jsonl', 'V264_PRIOR_PREDICTIONS.jsonl', 'V264_EVALUATOR_CODE.py'):
        b = raw(inputs / name)
        need(len(b) == integrity['files'][name]['size_bytes'] and hashlib.sha256(b).hexdigest() == integrity['files'][name]['sha256'], 'Compact input mismatch')
        payloads[name] = b
    e = load_evaluator(inputs / 'V264_EVALUATOR_CODE.py')
    predictions = jsonl(payloads['NEW_PREDICTIONS.jsonl'])
    validate_predictions(predictions)
    max_prior_diff = validate_prior(predictions, jsonl(payloads['V264_PRIOR_PREDICTIONS.jsonl']))
    ledger = list(csv.DictReader(io.StringIO(payloads['LEDGER.csv'].decode('utf-8-sig'))))
    result = metrics(e, ledger, predictions)
    outputs = write_metrics(out, result, ledger, predictions)
    summary = dump(out / 'SUMMARY.json', dict(completed=True, PASS=True, mode='COMPACT_METRICS_REPLAY',
        **result, outputs=outputs, original_prior_max_abs_logit_difference=max_prior_diff,
        RGB_reads=0, model_checkpoint_or_encoder_reads=0, optimizer_updates=0, solver_calls=0))
    return dump(out / 'COMPLETE.json', dict(completed=True, PASS=True, summary=summary,
                inputs_integrity=bind(inputs / 'INTEGRITY.json'), code=bind(Path(__file__).absolute())))


def font(size):
    return ImageFont.truetype(FONT_PATH, size)


def label(draw, text, xy, size=20, color='#172238'):
    draw.text(xy, text, font=font(size), fill=color)


def marker(draw, xy, color, circle=False):
    x, y = xy
    if not (0 <= x < 352 and 0 <= y < 352):
        return
    if circle:
        draw.ellipse((x-6, y-6, x+6, y+6), outline=color, width=3)
    else:
        draw.line((x-8, y, x+8, y), fill=color, width=3)
        draw.line((x, y-8, x, y+8), fill=color, width=3)


def make_card(e, index, ledger, new, rgb, hm, dense, out, transition):
    """Categorical colors are fixed; target probability uses a fixed [0,1] scale."""
    row, pred = ledger[index], new[index]
    im = Image.new('RGB', (1200, 1130), 'white')
    d = ImageDraw.Draw(im)
    label(d, f"G{index} · {row['row_id']}", (22, 12), 26)
    label(d, f"기존 주석 {row['human_state']} · 좌표 출처 {row['annotation_coordinate_source']}", (22, 51), 19)
    for j, arm in enumerate(ARMS):
        p = pred[arm]
        label(d, f"{arm}: {p['pred']} · P(E)={float(p['p_E']):.3f}", (22, 84 + j*27), 19)
    raw_rgb = Image.fromarray(rgb, 'RGB').resize((352, 352), Image.Resampling.NEAREST)
    h = np.asarray(hm, float)
    hn = (h - h.min()) / max(float(h.max() - h.min()), 1e-12)
    hc = np.stack((255*hn, 210*hn, 220*(1-hn)), -1).clip(0, 255).astype('uint8')
    hpic = Image.blend(raw_rgb, Image.fromarray(hc).resize((352,352), Image.Resampling.BILINEAR), .45)
    palette = np.array([[30, 210, 85], [255, 110, 25], [65, 95, 150]], np.uint8)
    panels = [('실제 저장 crop', raw_rgb.copy()), ('고정 V263 heatmap · 상대 색', hpic)]
    dense_diagnostics = {}
    for arm in ARMS[1:]:
        p = np.asarray(dense[arm][index], float)
        argmax = np.argmax(p, 0)
        categorical = Image.fromarray(palette[argmax]).resize((352,352), Image.Resampling.NEAREST)
        categorical = Image.blend(raw_rgb, categorical, .48)
        target = p[0]
        # Fixed zero-to-one map shared by both heads and every crop.
        tc = np.stack((20 + 20*target, 30 + 220*target, 25 + 65*target), -1).clip(0,255).astype('uint8')
        tpic = Image.fromarray(tc).resize((352,352), Image.Resampling.BILINEAR)
        panels.extend([(arm + ' · argmax', categorical), (arm + ' · P(target) 0–1', tpic)])
        entropy = -(p * np.log(np.maximum(p, 1e-30))).sum(0) / math.log(3)
        dense_diagnostics[arm] = dict(mean_target_probability=float(target.mean()),
            mean_foreign_probability=float(p[1].mean()), mean_normalized_entropy=float(entropy.mean()),
            target_probability_at_initial_query=float(target[8,8]),
            predicted_class_fraction={CLASS_NAMES[k]: float((argmax == k).mean()) for k in range(3)})
    q = np.array([float(row['query_x']), float(row['query_y'])])
    side = float(row['JOINT_SELECTED__crop_L'])
    ref = np.array([float(row['human_reference_x']), float(row['human_reference_y'])])
    prop = np.array(json.loads(row['JOINT_SELECTED__LOC_proposed_xy']))
    points = [(np.array([47.5,47.5]), '#00e0ff', True),
              ((ref-q)*96/side+47.5, '#ff39cb', False),
              ((prop-q)*96/side+47.5, '#ffff40', False)]
    for j, (title, tile) in enumerate(panels):
        x, y = 24 + (j % 3)*398, 226 + (j // 3)*426
        td = ImageDraw.Draw(tile)
        for p, color, circle in points:
            marker(td, (p+.5)*352/96, color, circle)
        im.paste(tile, (x,y))
        label(d, title, (x,y-35), 16)
        if j >= 2:
            arm = ARMS[1] if j < 4 else ARMS[2]
            a = dense_diagnostics[arm]
            label(d, f"평균 P(T) {a['mean_target_probability']:.2f} · P(F) {a['mean_foreign_probability']:.2f} · H {a['mean_normalized_entropy']:.2f}",
                  (x,y+357), 15)
    label(d, '초록=target pallet · 주황=foreign object · 파랑=background · H=평균 정규화 entropy', (22, 1035), 17)
    label(d, '파랑○ 초기 · 노랑+ 고정 위치 후보 · 분홍+ 상속 참조(E는 직접 보이는 위치 정답 아님)', (22,1063), 16)
    caution = 'G996: 사용자와 기존 E 주석의 해석이 달라 재검토 대상으로 유지합니다.' if index == 996 else '실사 dense 정답은 없습니다. 분할 색·P(target)는 모델 출력이며 보임 정답이나 보정 승인이 아닙니다.'
    label(d, caution, (22,1091), 15, '#8b421e')
    path = safe(out / f'CASE_G{index:04d}.png', True)
    im.save(path)
    clean_path = safe(out / f'CLEAN_G{index:04d}.png', True)
    Image.fromarray(rgb, 'RGB').save(clean_path)
    return dict(global_index=index, row_id=row['row_id'], group=row['group'], human_state=row['human_state'],
        annotation_coordinate_source=row['annotation_coordinate_source'], transition=transition,
        user_disputed_annotation=index==996, RGB96_HWC_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
        card=bind(path), clean=bind(clean_path), dense_diagnostics=dense_diagnostics,
        predictions={a: pred[a] for a in ARMS})


def visualize(e, out, ledger, new, result, seals):
    # Numeric selection is recorded before any saved current RGB is read.
    selected = [r['global_index'] for r in result['external90_rows']]
    selected += [r['selected_global_index'] for r in result['fixed_pattern_selection'] if not r['empty']]
    selected += [370, 256, 908, 996]
    selected = list(dict.fromkeys(selected))
    selection_binding = dump(out / 'CASE_SELECTION_BEFORE_RGB.json', dict(completed=True,
        external90_all=[r['global_index'] for r in result['external90_rows']],
        fixed_patterns=result['fixed_pattern_selection'], predefined=[370,256,908,996],
        selected_indices=selected, prior_RGB_exposure=True,
        selection='All90 PRIMARY E plus min original global index in material x D/E x three-arm correctness pattern. Prior visual exposure means this is not blinded evidence.',
        cone_claim='External E labels do not establish cone occlusion; no automatic all67-cone claim.'))
    complete = json.loads(e.checked(e.PINNED['v263_current_complete']))
    crop_binding = complete['crop_cache']['binding']
    heatmap_binding = complete['heatmaps']['JOINT_SELECTED']['binding']
    with np.load(io.BytesIO(checked(crop_binding)), allow_pickle=False) as z:
        crops = z['RGB96_initial_geometry']
    with np.load(io.BytesIO(checked(heatmap_binding)), allow_pickle=False) as z:
        heatmaps = z[complete['heatmaps']['JOINT_SELECTED']['probability_member']]
    with np.load(io.BytesIO(checked(seals['dense_predictions'])), allow_pickle=False) as z:
        dense = {arm: np.array(z[arm], copy=True) for arm in ARMS[1:]}
    need(heatmaps.shape == (1784,16,16) and len(crops) == 1784, 'Exact saved current crop/heatmap order')
    transitions = {r['global_index']: r for r in result['external90_rows']}
    cards = []
    for i in selected:
        rgb = e.rgb_array(crops[i])
        cards.append(make_card(e, i, ledger, new, rgb, heatmaps[i], dense, out, transitions.get(i)))
    case_data = dump(out / 'VISUAL_CASE_DATA.json', dict(completed=True, cases=cards, selection=selection_binding,
        crop_source=crop_binding, frozen_heatmap_source=heatmap_binding,
        dense_source=seals['dense_predictions'], dense_class_order=list(CLASS_NAMES),
        segmentation_truth_on_current223=False, no_current_dense_GT_or_threshold_selection=True))
    # Only immutable local selected crops are needed to reproduce the diagnostic panels.
    compact = safe(out / 'compact_visual_inputs', True)
    compact.mkdir()
    ids = np.array(selected, np.int64)
    rgbs = np.stack([e.rgb_array(crops[i]) for i in selected])
    np.savez_compressed(safe(compact / 'SELECTED_CROPS_HEATMAP_DENSE.npz', True),
        global_index=ids, RGB96_HWC=rgbs, original_heatmap=heatmaps[ids],
        **{a: dense[a][ids] for a in ARMS[1:]})
    visual_inputs = dump(compact / 'SOURCE_BINDINGS.json', dict(completed=True,
        selected_payload=bind(compact / 'SELECTED_CROPS_HEATMAP_DENSE.npz'),
        crop_source=crop_binding, original_heatmap_source=heatmap_binding,
        dense_source=seals['dense_predictions'], selected_indices=selected))
    return dict(case_data=case_data, selection=selection_binding, compact_visual_inputs=visual_inputs), cards


def report(out, result, cards):
    esc = lambda v: html.escape(str(v))
    def percent(x):
        return '—' if x is None else f'{100*x:.1f}'
    parts = ['<!doctype html><html lang="ko"><meta charset="utf-8"><title>V266 · 합성 분할·대조 학습 비교</title>',
        '<style>body{font:17px/1.6 sans-serif;max-width:1500px;margin:32px auto;padding:0 20px;color:#172238}table{border-collapse:collapse}td,th{border:1px solid #ccd4df;padding:8px}img{max-width:100%}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.case{background:#f3f6fa;padding:10px}.note{background:#fff1cf;padding:16px}a{color:#175cc0}</style>',
        '<h1>같은 평가에서 현재 방식 / 분할 추가 / 분할+대조 학습 비교</h1>',
        '<p class="note">새 학습은 합성만 사용했습니다. 실사223장은 이번 fit·threshold 선택에 쓰지 않았지만, 이전에도 확인한 개발 평가입니다. 외부 가림과 자기 가림을 분리합니다. 분할 색은 모델 출력이고 실사 분할 정답은 없습니다. 숨은 E 참조 좌표는 직접 보이는 위치 정답이 아닙니다.</p>']
    for group in GROUPS:
        parts.append('<h2>'+esc(group)+' · PRIMARY 보임/외부 가림</h2><table><tr><th>방법</th><th>D→D/E/U</th><th>E→D/E/U</th><th>D P/R</th><th>E P/R/F1</th><th>보류율</th></tr>')
        for row in result['classification']:
            if row['group'] != group or row['cohort'] != 'PRIMARY':
                continue
            cells = [row['model'],f"{row['D_D']}/{row['D_E']}/{row['D_U']}",f"{row['E_D']}/{row['E_E']}/{row['E_U']}",
                '/'.join(percent(row[k]) for k in ('D_precision','D_recall')),
                '/'.join(percent(row[k]) for k in ('E_precision','E_recall','E_F1')), percent(row['U_rate'])]
            parts.append('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in cells)+'</tr>')
        parts.append('</table>')
    parts.append('<h2>기존 외부 가림 누락67점의 변화와 새 누락</h2><p>67점 전체가 콘 가림인 것은 아닙니다. 결과 기반 회복/잔존/회귀 구분을 표시하며, 정확한 가림 물체와 native 점은 crop에서 확인해야 합니다. G370·G256·G908은 기존 콘 관련 검토 사례이고 G996은 사용자 주석 이견 사례입니다.</p><table><tr><th>재질</th><th>방법</th><th>기존67중 회복</th><th>남은 누락</th><th>기존 정답 유지</th><th>새 누락</th></tr>')
    for row in result['external90_transitions']:
        cells=[row['group'],row['model'],row['RECOVERED_PRIOR_MISS'],row['REMAINED_PRIOR_MISS'],row['KEPT_PRIOR_CORRECT'],row['REGRESSED_PRIOR_CORRECT']]
        parts.append('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in cells)+'</tr>')
    parts.append('</table><p><a href="CLASSIFICATION.csv">전체 혼동행렬·P/R/F1·보류율</a> · <a href="SELF_RESPONSES.csv">자기 가림 별도 응답</a> · <a href="ALL1784_COMPARISON.csv">1784점 행별 출력</a> · <a href="EXTERNAL90_TRANSITIONS.csv">외부 가림90점 전이</a> · <a href="SUMMARY.json">전체 JSON</a> · <a href="CASE_SELECTION_BEFORE_RGB.json">사례 선정·빈 셀·기존 RGB 노출</a></p>')
    parts.append('<h2>외부 가림90점 전체와 고정 패턴 사례</h2><p>정확도 패턴은 CURRENT/SEG/SEG+CONTRAST 순서입니다. 각 재질×D/E×패턴의 최소 원본 순번을 선택하고 빈 셀을 유지했습니다. 초록/주황/파랑은 target/foreign/background argmax입니다. P(target)는 모든 crop에서 고정0–1색이며 heatmap만 개별 min/max 표시입니다. H는 정규화 entropy로 보임 확률이 아닙니다.</p><div class="grid">')
    for case in cards:
        i=case['global_index']
        t=case['transition']
        desc='고정 패턴 또는 사전 지정 사례' if t is None else ' · '.join(a+': '+t[a+'_transition'] for a in ARMS[1:])
        parts.append('<div class="case"><p>'+esc(f"G{i} · {case['row_id']} · "+desc)+'</p><a href="'+Path(case['card']['path']).name+'"><img loading="lazy" src="'+Path(case['card']['path']).name+'"></a><a href="'+Path(case['clean']['path']).name+'">표시 없는 실제 crop</a></div>')
    parts.append('</div><p>이번 평가·시각화 코드는 새 학습·모델 추론·PnP·운영 규칙 선택을 하지 않습니다. U는 기존 입력 불가45점에만 적용되며 confidence threshold를 뜻하지 않습니다.</p></html>')
    p=safe(out/'INDEX_KO.html',True)
    p.write_text('\n'.join(parts),encoding='utf-8')
    return bind(p)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='evaluation_v1')
    parser.add_argument('--replay-metrics', action='store_true')
    parser.add_argument('--inputs', default='evaluation_v1/compact_inputs')
    parser.add_argument('--metrics-only', action='store_true')
    args=parser.parse_args()
    need(Path(args.out).name==args.out and '..' not in args.out,'Single output directory name')
    out=safe(W/args.out,True)
    if args.replay_metrics:
        rel=Path(args.inputs)
        inputs=safe(rel if rel.is_absolute() else W/rel)
        out.mkdir()
        complete=replay(inputs,out)
    else:
        e=load_evaluator()
        new,seals=freeze_inputs()
        out.mkdir()
        consumer=dump(out/'CONSUMER_SEAL_BEFORE_LATE_GT_AND_RGB.json',dict(completed=True,PASS=True,
            seals=seals,predictions1784_validated=True,late_current_GT_or_RGB_reads=0,
            created_UTC=dt.datetime.now(dt.timezone.utc).isoformat()))
        # All late current annotation reads start here.
        ledger_bytes=e.checked(e.PINNED['join'])
        ledger=list(csv.DictReader(io.StringIO(ledger_bytes.decode('utf-8-sig'))))
        prior_bytes=raw(PRIOR_PATH)
        need(hashlib.sha256(prior_bytes).hexdigest()==PRIOR_SHA,'Pinned current V264 saved prediction SHA')
        prior_diff=validate_prior(new,jsonl(prior_bytes))
        result=metrics(e,ledger,new)
        outputs=write_metrics(out,result,ledger,new)
        compact=compact_inputs(out,e,ledger_bytes,checked(seals['predictions']),prior_bytes)
        visuals,cards=({},[]) if args.metrics_only else visualize(e,out,ledger,new,result,seals)
        summary=dump(out/'SUMMARY.json',dict(completed=True,PASS=True,rows=1784,frames=223,
            arms=list(ARMS),dense_class_order=list(CLASS_NAMES),**result,
            current223_is_repeated_development_evaluation=True,fresh_blind_evaluation=False,
            threshold=dict(E_logit=0,E_if='logit > 0',U='original input unavailable only'),
            confusion_order=dict(rows=['D','E'],columns=['D','E','U']),
            no_current_annotations_used_for_new_training=True,no_new_threshold_or_PnP_or_location_changes=True,
            model_checkpoint_or_encoder_reads=0,optimizer_updates=0,solver_calls=0,
            current_V264_max_abs_saved_logit_difference=prior_diff,
            consumer_pre_late_GT_seal=consumer,seals=seals,outputs=outputs,visuals=visuals,
            compact_metrics_inputs=compact,pinned_evaluator=bind(EVALUATOR_PATH)))
        report_binding=report(out,result,cards)
        complete=dump(out/'COMPLETE.json',dict(completed=True,PASS=True,summary=summary,
            report=report_binding,outputs=outputs,visuals=visuals,code=bind(Path(__file__).absolute()),
            consumer_pre_late_GT_seal=consumer,compact_metrics_inputs=compact))
    print(json.dumps(dict(completed=True,PASS=True,COMPLETE=complete),ensure_ascii=False))


if __name__=='__main__':
    main()
