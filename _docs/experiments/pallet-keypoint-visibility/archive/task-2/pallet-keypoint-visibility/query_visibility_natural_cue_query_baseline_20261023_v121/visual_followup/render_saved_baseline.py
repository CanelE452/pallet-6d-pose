"""Saved public baseline diagnostic. Actual rendering requires a completed prediction seal."""
import argparse
import ast
import hashlib
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

H = Path(__file__).resolve().parent
V = H.parent
P = V.parent
SUMMARY = V / 'input_preparation/SUMMARY.json'
POLICY = V / 'ROOT_PUBLIC_BASELINE_POLICY_BEFORE_FORWARD.json'
TARGET_MATH = P / 'query_visibility_target_context_rgb_20261023_v107/target_context_math.py'
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
ROLES = ('D_SUCCESS', 'E_SUCCESS', 'E_MISSED', 'FALSE_E')
ROLE_KO = dict(FIXED_PAIR00_CENTRE='고정 pair00 중앙 offset', D_SUCCESS='첫 D 정답',
               E_SUCCESS='첫 E 정답', E_MISSED='첫 E 누락(D/U)', FALSE_E='첫 false E')


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def bind(p):
    return dict(path=str(Path(p).resolve()), sha256=sha(p))


def same_binding(a, b):
    return a['path'] == b['path'] and a['sha256'] == b['sha256']


def checked(b):
    if sha(b['path']) != b['sha256']:
        raise ValueError('Binding mismatch: ' + b['path'])
    return Path(b['path'])


def rows(p):
    return [json.loads(s) for s in Path(p).read_text().splitlines() if s]


def write(p, value):
    with p.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def select_cases(inputs, labels, predictions):
    """Original occurrence order only, no aesthetic/quality substitution."""
    bykey = {(r['pair_index'], r['jitter_index'], r['view']): r['occurrence_index'] for r in inputs}
    fixed = bykey[(0, 4, 'before')]
    if inputs[fixed]['offset_xy'] != [0, 0]:
        raise ValueError('Mandatory central offset changed')
    selected = [dict(pair_index=0, jitter_index=4, roles=['FIXED_PAIR00_CENTRE'], trigger_rows=[fixed])]
    missing = []
    for role in ROLES:
        yes = dict(D_SUCCESS=(labels == 0) & (predictions == 0),
                   E_SUCCESS=(labels == 1) & (predictions == 1),
                   E_MISSED=(labels == 1) & (predictions != 1),
                   FALSE_E=(labels == 0) & (predictions == 1))[role]
        candidates = np.flatnonzero(yes)
        if not len(candidates):
            missing.append(role)
            continue
        i = int(candidates[0])
        r = inputs[i]
        matches = [c for c in selected if (c['pair_index'], c['jitter_index']) == (r['pair_index'], r['jitter_index'])]
        if matches:
            matches[0]['roles'].append(role)
            matches[0]['trigger_rows'].append(i)
        else:
            selected.append(dict(pair_index=r['pair_index'], jitter_index=r['jitter_index'], roles=[role], trigger_rows=[i]))
    for c in selected:
        c['indices'] = [bykey[(c['pair_index'], c['jitter_index'], s)] for s in ('before', 'after')]
    return selected, missing


def font(size):
    return ImageFont.truetype(FONT, size)


def text(draw, xy, message, size=17, color='#152236'):
    draw.text(xy, message, font=font(size), fill=color)


def paste_fit(canvas, array, xy, wh, nearest=False):
    img = Image.fromarray(array)
    scale = min(wh[0] / img.width, wh[1] / img.height)
    size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
    img = img.resize(size, Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR)
    x, y = xy
    canvas.paste(img, (x + (wh[0] - img.width) // 2, y + (wh[1] - img.height) // 2))


def diagnostic_figure(cases, missing, inputs, labels, logit, predictions, crops, contexts):
    width, rowheight, top = 1260, 360, 176
    canvas = Image.new('RGB', (width, top + rowheight * len(cases) + 66), '#ffffff')
    d = ImageDraw.Draw(canvas)
    text(d, (18, 12), '공개 자연 cue · frozen C14 · 실제 저장 입력 진단', 26)
    text(d, (18, 54), 'source-emulated q / weak cue visibility. 현재 성능·native corner gold·학습 승인 아님.', 18)
    text(d, (18, 83), 'RGB96와 central32는 저장된 무표시 모델 입력 그대로 확대. Context224는 exact grid 재구성/해시 검산 후 표시.', 16)
    text(d, (18, 110), '37 cue/22 physical pairs는 과거 source 개발에 노출됨. 9 jitters는 독립 데이터 아님. 사후 그림 선택은 진단 전용.', 16)
    text(d, (18, 137), '없는 역할: ' + (', '.join(ROLE_KO[r] for r in missing) if missing else '없음'), 16)
    panelx = (18, 252, 454, 642, 876, 1078)
    for n, case in enumerate(cases):
        y = top + n * rowheight
        a = inputs[case['indices'][0]]
        d.rectangle((10, y, width - 10, y + rowheight - 8), fill='#f3f6fa')
        roles = ' / '.join(ROLE_KO[r] for r in case['roles'])
        text(d, (18, y + 7), f"pair{case['pair_index']:02d}, jitter{case['jitter_index']} {a['offset_xy']} | {roles}", 20)
        text(d, (18, y + 36), f"{a['pair_id']} · {a['raw_split']} · q={np.round(a['query_xy'], 3).tolist()} · L={a['L']:.4f}px", 16)
        for stage, idx in enumerate(case['indices']):
            r = inputs[idx]
            x0, x1, x2 = panelx[stage * 3:stage * 3 + 3]
            decision = {0: 'D', 1: 'E', -1: 'U'}[int(predictions[idx])]
            target = {0: 'D', 1: 'E'}[int(labels[idx])]
            status = f"{r['view'].upper()} weak {target} → {decision}, zE={float(logit[idx]):+.5f}"
            text(d, (x0, y + 66), status, 16)
            for x, title in ((x0, 'Context224 · 재구성'), (x1, '저장 RGB96 ×2'), (x2, '저장 central32 ×4')):
                text(d, (x, y + 94), title, 15)
            paste_fit(canvas, contexts[idx], (x0, y + 118), (224, 224))
            paste_fit(canvas, crops['rgb96'][idx].transpose(1, 2, 0), (x1, y + 118), (192, 192), True)
            paste_fit(canvas, crops['central32'][idx].transpose(1, 2, 0), (x2, y + 118), (128, 128), True)
            text(d, (x1, y + 313), f"valid {crops['valid96'][idx].mean():.3f}", 14)
            text(d, (x2, y + 257), f"valid {crops['central_valid32'][idx].mean():.3f}", 14)
    text(d, (18, canvas.height - 52), '그림의 D/E는 기존 AI weak expected-cue-region 관찰과 고정 raw-logit 부호 비교이며, 각 q의 목재 픽셀 소속 정답이 아닙니다.', 16)
    return canvas


def target_functions():
    tree = ast.parse(TARGET_MATH.read_text())
    names = {'valid_box', 'target_context_grid', 'sample_target_context'}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if len(nodes) != 3:
        raise ValueError('Pinned context math missing')
    ns = {'np': np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(TARGET_MATH), 'exec'), ns)
    return ns['sample_target_context']


def run(seal_path):
    seal = json.loads(Path(seal_path).read_text())
    if seal.get('completed') is not True or seal.get('PASS') is not True:
        raise ValueError('Completed prediction seal required before any score/GT/pixel read')
    policy = json.loads(POLICY.read_text())
    summary = json.loads(SUMMARY.read_text())
    for key in ('source_model_freeze', 'checkpoint'):
        expected = policy['model_freeze' if key == 'source_model_freeze' else key]
        if not same_binding(seal[key], expected):
            raise ValueError('Frozen C14 binding differs')
    if not same_binding(seal['input_manifest'], summary['input_rows']):
        raise ValueError('All666 input manifest differs')
    inputs = rows(checked(seal['input_manifest']))
    if len(inputs) != 666 or [r['occurrence_index'] for r in inputs] != list(range(666)):
        raise ValueError('All666 order required')
    with np.load(checked(seal['predictions']), allow_pickle=False) as f:
        logit = np.array(f['query_logit'], copy=True)
        predictions = np.array(f['prediction'], copy=True)
        available = np.array(f['query_available'], copy=True)
        ids = np.array(f['query_id'], copy=True)
        indices = np.array(f['global_occurrence_index'], copy=True)
    if logit.shape != (666,) or not np.isfinite(logit).all() or not np.array_equal(indices, np.arange(666)):
        raise ValueError('Prediction schema/order invalid')
    if not np.array_equal(ids, np.asarray([r['query_id'] for r in inputs])):
        raise ValueError('Prediction identity differs')
    if not np.array_equal(predictions, np.where(available, (logit >= 0).astype(np.int8), -1)):
        raise ValueError('Raw sign/availability decode differs')
    # Only now join unchanged weak evidence; no label was read by the model.
    evidence = rows(checked(summary['weak_original_evidence']))
    if len(evidence) != 666 or any((e['occurrence_index'], e['query_id']) != (i, inputs[i]['query_id']) for i, e in enumerate(evidence)):
        raise ValueError('Weak evidence identity differs')
    labels = np.array([e['original_teacher_label_unchanged'] for e in evidence], np.int8)
    cases, missing = select_cases(inputs, labels, predictions)
    selected_indices = sorted({i for c in cases for i in c['indices']})
    selection = dict(completed=True, diagnostic_only=True, no_model_or_threshold_selection=True,
                     policy='Mandatory pair00 jitter4; then first original occurrence per D-success/E-success/E-missed(D/U)/falseE, entire matched view pair retained, duplicate cases merged, no substitutions.',
                     prediction_seal=bind(seal_path), input_manifest=seal['input_manifest'], weak_evidence=summary['weak_original_evidence'],
                     selected_cases=cases, missing_roles=missing, selected_indices=selected_indices,
                     selection_written_before_any_RGB_or_crop_payload_read=True)
    write(H / 'SELECTION_BEFORE_VISUAL_PAYLOADS.json', selection)
    facts = rows(checked(seal['inputfacts']))
    byindex = {r['occurrence_index']: r for r in facts}
    if sorted(byindex) != list(range(666)):
        raise ValueError('Inputfacts must retain666')
    with np.load(checked(seal['crops']), allow_pickle=False) as f:
        crops = {key: np.array(f[key], copy=True) for key in ('rgb96', 'valid96', 'central32', 'central_valid32', 'global_occurrence_index')}
    if crops['rgb96'].shape != (666, 3, 96, 96) or crops['rgb96'].dtype != np.uint8:
        raise ValueError('RGB96 schema')
    if crops['valid96'].shape != (666, 1, 96, 96) or crops['valid96'].dtype != bool:
        raise ValueError('valid96 schema')
    if not np.array_equal(crops['global_occurrence_index'], np.arange(666)) or not np.array_equal(crops['central32'], crops['rgb96'][:, :, 32:64, 32:64]) or not np.array_equal(crops['central_valid32'], crops['valid96'][:, :, 32:64, 32:64]):
        raise ValueError('Central32 must be literal saved RGB96 slice')
    sample = target_functions()
    contexts, provenance, rawcache = {}, [], {}
    for idx in selected_indices:
        r, fact = inputs[idx], byindex[idx]
        image_binding = r['full_RGB']
        if image_binding['path'] not in rawcache:
            path = checked(image_binding)
            with Image.open(path) as im:
                rgb = np.array(im.convert('RGB'), copy=True)
            if list(rgb.shape[:2]) != r['raw_hw'] or hashlib.sha256(rgb.tobytes()).hexdigest() != r['decoded_RGB_sha256']:
                raise ValueError('Canonical public native RGB differs')
            rawcache[image_binding['path']] = rgb
        target, valid = sample(rawcache[image_binding['path']], r['context_box_xyxy'])
        th = hashlib.sha256(np.ascontiguousarray(target).tobytes()).hexdigest()
        vh = hashlib.sha256(np.ascontiguousarray(valid).tobytes()).hexdigest()
        if th != fact['target_RGB224_F32_sha256'] or vh != fact['target_valid_sha256']:
            raise ValueError('Reconstructed context224 differs from actual inputfacts')
        contexts[idx] = np.floor(np.clip(target, 0, 255) + .5).astype(np.uint8)
        provenance.append(dict(occurrence_index=idx, query_id=r['query_id'], full_RGB=image_binding,
                               q=r['query_xy'], L=r['L'], context=r['context_box_xyxy'],
                               saved_RGB96_sha256=hashlib.sha256(crops['rgb96'][idx].tobytes()).hexdigest(),
                               saved_valid96_sha256=hashlib.sha256(crops['valid96'][idx].tobytes()).hexdigest(),
                               saved_central32_sha256=hashlib.sha256(crops['central32'][idx].tobytes()).hexdigest(),
                               reconstructed_target224_F32_sha256=th, reconstructed_target_valid_sha256=vh,
                               weak_label=int(labels[idx]), prediction=int(predictions[idx]), raw_E_logit=float(logit[idx])))
    fig = diagnostic_figure(cases, missing, inputs, labels, logit, predictions, crops, contexts)
    figpath = H / 'SOURCE_BASELINE_FIXED_DIAGNOSTIC.png'
    if figpath.exists():
        raise FileExistsError('Preserve existing actual diagnostic')
    fig.save(figpath)
    write(H / 'FIGURE_PROVENANCE.json', dict(selection=bind(H / 'SELECTION_BEFORE_VISUAL_PAYLOADS.json'),
          code=bind(__file__), target_math=bind(TARGET_MATH), prediction_seal=bind(seal_path),
          crops=seal['crops'], inputfacts=seal['inputfacts'], cases=provenance,
          target_display='Exact preprocessor-input context224 F32 hash replay, then floor(+.5) to uint8 for display; not native full frame or actual normalized GPU tensor.',
          RGB96_central32_display='Literal saved uint8 model inputs, unmarked, nearest enlargement; no oracle128 reuse.',
          no_labels_q_L_native_geometry_or_threshold_changed=True, new_model_encoder_fit_current_V112_PnP=0))
    missingtext = ', '.join(ROLE_KO[r] for r in missing) or '없음'
    h = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V121 공개 source 입력 진단</title><style>body{{font:17px/1.65 sans-serif;color:#152236;max-width:1260px;margin:30px auto;padding:0 20px}}img{{width:100%;height:auto}}p{{max-width:1050px}}code{{overflow-wrap:anywhere}}a{{color:#0756a6}}</style><h1>Frozen C14 · 공개 자연 cue baseline 입력 진단</h1><p>고정 pair00 중앙 offset과 원래 순서의 첫 성공·실패 사례입니다. 없는 유형: {html.escape(missingtext)}. 선택은 prediction seal 이후의 사후 설명용이며 모델·임계값 선택이나 학습에 쓰지 않습니다.</p><p>37개 weak cue는22개 physical pair와44개 native RGB에서 왔으며 과거 source 개발에 이미 노출됐습니다. 666행과9개 jitter는 독립 데이터가 아닙니다. 실제 initializer 입력을 보존한 별도 source-emulated q이며 native corner·human gold·현재 성능을 뜻하지 않습니다.</p><img src="{figpath.name}" alt="고정 source 진단 실제 RGB96와 central32 및 context224"><p>RGB96와 central32는 새로 저장된 실제 모델 입력을 무표시로 표시합니다. Context224는 같은 native RGB·고정 context box를 exact grid로 재구성하고 실제 입력 hash를 확인한 뒤 표시한 이미지입니다. 원본 full frame이나 normalized GPU tensor dump와 구분합니다. 기존 oracle128 진단 이미지는 재사용하지 않습니다.</p><p>Weak D/E는 기존 expected structural-cue-region visibility 관찰입니다. q 픽셀의 목재 소속, 정확한 native part 위치 또는 shift마다 독립적인 corner 정답이 아닙니다. 원래 등록·불확실성 한계를 유지합니다.</p><p><a href="SELECTION_BEFORE_VISUAL_PAYLOADS.json">고정 선택 기록</a> · <a href="FIGURE_PROVENANCE.json">입력·그림 provenance</a></p></html>'''
    htmlpath = H / 'READOUT_KO.html'
    with htmlpath.open('x') as f:
        f.write(h)
    write(H / 'VISUALIZATION_RECEIPT.json', dict(completed=True, prediction_seal=bind(seal_path),
          code=bind(__file__), figure=bind(figpath), HTML=bind(htmlpath), provenance=bind(H / 'FIGURE_PROVENANCE.json'),
          selected_cases=len(cases), actual_unique_RGB_decodes=len(rawcache), selected_input_rows=len(selected_indices),
          actual_view_status='NOT_YET_VIEWED_BY_THIS_RENDERER; root/agent actual view recorded separately.',
          no_new_model_encoder_fit_current_V112_PnP=True))
    print(json.dumps(dict(completed=True, receipt=bind(H / 'VISUALIZATION_RECEIPT.json')), indent=2))


def generated():
    inputs = [dict(occurrence_index=i, pair_index=i // 18, jitter_index=(i % 18) // 2,
                   offset_xy=[0, 0] if (i % 18) // 2 == 4 else [-3, -3],
                   view='before' if i % 2 == 0 else 'after') for i in range(36)]
    labels = np.array([i % 2 for i in range(36)], np.int8)
    good = labels.copy()
    cases, missing = select_cases(inputs, labels, good)
    assert cases[0]['indices'] == [8, 9] and missing == ['E_MISSED', 'FALSE_E']
    changed = good.copy()
    changed[1] = 0
    changed[2] = 1
    cases, missing = select_cases(inputs, labels, changed)
    assert not missing
    assert next(c for c in cases if 'E_MISSED' in c['roles'])['trigger_rows'][-1] == 1
    assert next(c for c in cases if 'FALSE_E' in c['roles'])['trigger_rows'][-1] == 2
    assert all(len(c['indices']) == 2 for c in cases)
    changed[1] = -1
    assert 'E_MISSED' in [r for c in select_cases(inputs, labels, changed)[0] for r in c['roles']]
    # In-memory format only: no actual corpus/crop/model reads or generated output PNG.
    for r in inputs:
        r.update(pair_id='invented', raw_split='TRAIN_LAB', query_xy=[20., 20.], L=40.)
    invented = dict(rgb96=np.zeros((36, 3, 96, 96), np.uint8), valid96=np.ones((36, 1, 96, 96), bool),
                    central32=np.zeros((36, 3, 32, 32), np.uint8), central_valid32=np.ones((36, 1, 32, 32), bool))
    contexts = {i: np.full((224, 224, 3), 180, np.uint8) for c in cases for i in c['indices']}
    img = diagnostic_figure(cases, missing, inputs, labels, np.zeros(36, np.float32), changed, invented, contexts)
    assert img.width == 1260 and img.height < 2200
    write(H / 'GENERATED_FORMAT_CHECK.json', dict(PASS=True, code=bind(__file__),
          mandatory_pair00_jitter4=True, original_order_first_roles=True, absent_roles_explicit=True,
          duplicate_cases_merged_without_losing_roles=True, E_abstention_is_miss=True,
          whole_view_pairs_retained=True, in_memory_format_dimensions=list(img.size),
          actual_RGB_scores_GT_features_weights_models_current_V112_reads=0))
    print(json.dumps(dict(PASS=True, generated=bind(H / 'GENERATED_FORMAT_CHECK.json'))))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--prediction-seal', type=Path)
    ap.add_argument('--generated', action='store_true')
    args = ap.parse_args()
    if args.generated:
        generated()
    elif args.prediction_seal:
        run(args.prediction_seal)
    else:
        raise SystemExit('Use --generated or --prediction-seal COMPLETED_SOURCE_PREDICTION_SEAL.json')
