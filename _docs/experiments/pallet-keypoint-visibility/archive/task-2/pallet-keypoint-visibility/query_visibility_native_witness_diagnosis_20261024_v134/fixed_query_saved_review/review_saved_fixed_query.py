"""Independent saved-array arithmetic and two preselected source RGB panels.

No encoder, checkpoint, model, private input, fitting or PnP is used.
Run only after the frozen producer and separate GT join have completed.
"""
from pathlib import Path
import collections
import hashlib
import importlib.util
import json
import numpy as np

H = Path(__file__).resolve().parent
V = H.parent
P = V.parent
V132 = P / 'query_visibility_actual_initializer_source_20261024_v132'
V133 = P / 'query_visibility_native_geometry_control_20261024_v133'
FROZEN = V / 'same_before_query_control_v1'
RESULT = V / 'saved_fixed_query_evaluation_v1/SOURCE_FIXED_QUERY_DIAGNOSIS.json'
SELECTION = H / 'TWO_CASE_SELECTION_BEFORE_RGB.json'
HELPER = V132 / 'report_preparation/render_saved_visibility.py'
MODELS = ('C16', 'C24', 'C25')
ARMS = {'C24': 'C24_ZERO_GEOMETRY_CONTROL', 'C25': 'C25_NATIVE_GEOMETRY42'}


def read(p):
    return json.loads(Path(p).read_text())


def bind(p):
    p = Path(p).resolve()
    return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


def check(b):
    assert bind(b['path']) == b
    return Path(b['path'])


def write(p, x):
    with Path(p).open('x') as f:
        json.dump(x, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def arrays(path, keys):
    with np.load(path, allow_pickle=False) as f:
        return {k: np.array(f[k], copy=True) for k in keys}


def calculate(y, pred, mask):
    selected = list(zip(y[mask].tolist(), pred[mask].tolist()))
    cm = [[sum(a == t and b == p for a, b in selected) for p in (0, 1, -1)] for t in (0, 1)]
    ratio = lambda a, b: a / b if b else None
    return {'CM_D_E_rows_D_E_U_columns': cm,
            'D_recall': ratio(cm[0][0], sum(cm[0])), 'E_recall': ratio(cm[1][1], sum(cm[1])),
            'D_precision': ratio(cm[0][0], cm[0][0] + cm[1][0]), 'E_precision': ratio(cm[1][1], cm[0][1] + cm[1][1]),
            'prediction_U_known': cm[0][2] + cm[1][2],
            'prediction_U_rate_known': ratio(cm[0][2] + cm[1][2], sum(map(sum, cm))),
            'original_rows': int(mask.sum()), 'prediction_U_all': int(np.sum(mask & (pred == -1))),
            'known_DE': sum(map(sum, cm))}


def load():
    sel = read(SELECTION)
    assert sel['RGB_read'] is False and sel['source_only'] and sel['cases'] == [{'base_index': 136, 'native_slot': 3}, {'base_index': 134, 'native_slot': 0}]
    check(sel['code']); check(sel['helper'])
    result = read(RESULT)
    assert result['completed'] and result['PASS'] and result['source_only'] and result['all6144_preserved']
    seal = read(check(result['frozen_forward']))
    assert seal['completed'] and seal['PASS'] and seal['GT_read'] is False and seal['source_input_intervention']
    parity = read(check(seal['BEFORE_parity']))
    assert parity['PASS'] and all(parity['models'][m]['BEFORE1536_raw_exact'] for m in MODELS)
    rows = [json.loads(line) for line in check(result['GT_ledger']).read_text().splitlines()]
    assert len(rows) == 6144 and all(r['global_index'] == i for i, r in enumerate(rows))
    ids = np.array([r['query_id'] for r in rows])
    oldseal = read(V132 / 'frozen_C16_actual_initializer_v1/PREDICTION_SEAL.json')
    oldfacts = read(check(oldseal['input_crop_facts']))['rows']
    facts = read(check(seal['input_crop_facts']))['rows']
    keys = ('global_index', 'occurrence_id', 'query_xy', 'L', 'prediction', 'raw_head_logit_before_U', 'query_available', 'head_forwarded')
    old = {'C16': arrays(check(oldseal['predictions']), keys)}
    fixed = {m: arrays(check(seal['models'][m]['predictions']), keys) for m in MODELS}
    for m in ('C24', 'C25'):
        old[m] = arrays(V133 / 'runs/C24_C25_MATCHED_G42_FIXED8000' / ARMS[m] / 'FINAL_NATIVE6144_SCORES_INPUT_AND_OUTPUT.npz', keys)
    for group in (old, fixed):
        for m in MODELS:
            a = group[m]
            assert np.array_equal(a['global_index'], np.arange(6144)) and np.array_equal(a['occurrence_id'], ids)
            assert np.array_equal(a['prediction'], np.where(a['query_available'], (a['raw_head_logit_before_U'] >= 0).astype(np.int8), -1))
    for m in MODELS:
        assert np.array_equal(fixed[m]['query_xy'], fixed['C16']['query_xy'], equal_nan=True)
        assert np.array_equal(fixed[m]['query_available'], fixed['C16']['query_available'])
    for i, r in enumerate(facts):
        b = (i // 32) * 32 + i % 8
        assert r['occurrence_index'] == i and r['source_BEFORE_occurrence_index'] == b
        for k in ('query_xy', 'L', 'initial_points', 'context_box_xyxy'):
            assert r[k] == oldfacts[b][k]
    return sel, result, seal, rows, oldfacts, facts, old, fixed


def audit(result, rows, old, fixed):
    y = np.array([{'D': 0, 'E': 1}.get(r['raw_state'], -1) for r in rows])
    state = np.array([r['raw_state'] for r in rows]); fold = np.array([r['split'] for r in rows])
    variants = np.array([r['variant'] for r in rows]); family = np.array([r['factors']['foreign_family'] for r in rows])
    area_disagreement = np.array([r['convention_boundary'] for r in rows], bool)
    original_error = np.array([r['error_px'] if r['error_px'] is not None else np.nan for r in rows])
    refs = np.array([r['reference_xy'] for r in rows], np.float64)
    assert refs.shape == (6144, 2) and np.isfinite(refs).all()
    error = np.sqrt(np.sum((fixed['C16']['query_xy'] - refs) ** 2, axis=1))
    groups = {}
    for split in ('ALL', 'TRAIN', 'VALIDATION'):
        mask = np.ones(6144, bool) if split == 'ALL' else fold == split
        groups[split] = mask
        groups[split + '::historical_area_agreement'] = mask & ~area_disagreement
        groups[split + '::actual_error_LE3'] = mask & (original_error <= 3)
        groups[split + '::frozen_error_LE3'] = mask & (error <= 3)
        groups.update({split + '::' + v: mask & (variants == v) for v in set(variants)})
        groups.update({split + '::' + f: mask & (family == f) for f in set(family)})
    assert set(groups) == set(result['results'])
    for name, mask in groups.items():
        expected = result['results'][name]
        assert expected['teacher_states'] == {s: int(np.sum(mask & (state == s))) for s in 'DESU'}
        for m in MODELS:
            a, b = old[m]['prediction'], fixed[m]['prediction']
            common = mask & (a != -1) & (b != -1)
            changes = mask & (a != b)
            output = {'actual_query': calculate(y, a, mask), 'fixed_BEFORE_query': calculate(y, b, mask),
                      'common_available_actual': calculate(y, a, common), 'common_available_fixed': calculate(y, b, common),
                      'prediction_changed': int(changes.sum()),
                      'D_correct_to_E': int(np.sum(changes & (y == 0) & (a == 0) & (b == 1))),
                      'D_E_to_correct': int(np.sum(changes & (y == 0) & (a == 1) & (b == 0))),
                      'E_miss_to_correct': int(np.sum(changes & (y == 1) & (a == 0) & (b == 1))),
                      'E_correct_to_miss': int(np.sum(changes & (y == 1) & (a == 1) & (b == 0)))}
            assert output == expected['models'][m], (name, m)
    return {'groups_checked': len(groups), 'models_checked': list(MODELS), 'all_original_and_common_available_metrics_exact': True,
            'rows': 6144, 'available': int(fixed['C16']['query_available'].sum()),
            'constructed': int(fixed['C16']['head_forwarded'].sum()), 'noquery': int((~fixed['C16']['head_forwarded']).sum()),
            'native_ids_q8_L_context_shared_from_BEFORE_exact': True,
            'VALIDATION': result['results']['VALIDATION'], 'frozen_error_LE3': result['results']['VALIDATION::frozen_error_LE3']}


def show(case, ordinal, rows, oldfacts, facts, old, fixed, pixels, h):
    from PIL import Image, ImageDraw
    im = Image.new('RGB', (1700, 1390), '#f7f9fc'); d = ImageDraw.Draw(im)
    base, slot = case['base_index'], case['native_slot']
    h.text(d, (24, 15), '합성 장면 · BEFORE 초기 좌표를 고정한 비교', 32)
    h.text(d, (24, 65), f'장면 {base:03d} / 키포인트 채널 {slot} · 각 변형의 RGB는 원본 그대로입니다.', 24)
    h.text(d, (24, 106), '노랑 +: 고정 q / 청록 +: 원래 reference. 아래 raw96은 표시를 넣지 않은 저장 crop입니다.', 22)
    h.text(d, (24, 145), '세 모델과 임계값은 동결했습니다. D=보임, E=외부 가림, U=보류. 현재 실사 성능이 아닙니다.', 22)
    for x, title in ((24, '원본 장면 · 고정 q 표시'), (300, '기존 실제 q의 raw96'), (510, '고정 BEFORE q raw96'), (720, '고정 crop 별도 표시본'), (940, '저장 출력: 기존 → 고정')):
        h.text(d, (x, 200), title, 18)
    raw_receipts = []
    for vi in range(4):
        i = base * 32 + vi * 8 + slot; r = rows[i]; q = facts[i]['query_xy']; ref = r['reference_xy']; y = 245 + vi * 270
        scene = Image.fromarray(pixels.scene(facts[i])).resize((240, 240), Image.Resampling.BILINEAR)
        sd = ImageDraw.Draw(scene)
        h.cross(sd, [h.pixel_to_display(t, 240 / 512) for t in q], '#ffe43b', 7, 3)
        h.cross(sd, [h.pixel_to_display(t, 240 / 512) for t in ref], '#00e5e0', 4, 2)
        im.paste(scene, (24, y))
        for x, item, kind in ((300, oldfacts[i], 'ACTUAL'), (510, facts[i], 'FROZEN_BEFORE')):
            rgb = pixels.crop(item)
            rawpath = H / 'raw_crops' / f'CASE{ordinal:02d}_{r["variant"]}_{kind}_IDX{i:04d}_RAW96.png'
            Image.fromarray(rgb).save(rawpath)
            raw_receipts.append({'index': i, 'kind': kind, 'file': bind(rawpath), 'decoded_rgb_sha256': h.ash(rgb)})
            im.paste(Image.fromarray(rgb).resize((192, 192), Image.Resampling.NEAREST), (x, y + 25))
        marked = Image.fromarray(pixels.crop(facts[i])).resize((192, 192), Image.Resampling.NEAREST)
        md = ImageDraw.Draw(marked); h.cross(md, (95.5, 95.5), '#ffe43b', 10, 3)
        local = h.reference_in_crop(q, ref, facts[i]['L'])
        if local is not None and all(-5 < t < 101 for t in local):
            h.cross(md, [h.pixel_to_display(t, 2) for t in local], '#00e5e0', 6, 2)
        im.paste(marked, (720, y + 25))
        err = float(np.linalg.norm(np.array(q) - ref))
        h.text(d, (940, y + 1), f'{r["variant"]} | 원래 정답 {r["raw_state"]}', 24)
        h.text(d, (940, y + 37), '좌표 오차: %.2f → %.2fpx' % (r['error_px'], err), 22)
        for mi, m in enumerate(MODELS):
            names = {-1: 'U', 0: 'D', 1: 'E'}
            a, b = old[m], fixed[m]
            h.text(d, (940, y + 76 + mi * 34), '%s: %s (%+.3f) → %s (%+.3f)' % (m, names[int(a['prediction'][i])], a['raw_head_logit_before_U'][i], names[int(b['prediction'][i])], b['raw_head_logit_before_U'][i]), 22)
        h.text(d, (940, y + 187), 'q %s | L %.2f' % (h.coordinates(q), facts[i]['L']), 20)
        h.text(d, (940, y + 219), 'reference %s' % h.coordinates(ref), 20)
    h.text(d, (24, 1343), 'CF 순서는 이 192개 합성 장면의 원래 학습 규약과 대조했습니다. 실제 적용 가능한 좌표 보정 실험은 아닙니다.', 20, '#53617a')
    path = H / 'figures' / f'CASE{ordinal:02d}_BASE{base:03d}_SLOT{slot}_FROZEN_BEFORE_QUERY.png'
    im.save(path)
    return {'case': case, 'figure': bind(path), 'raw_crops': raw_receipts}


def main():
    assert not (H / 'SAVED_REVIEW.json').exists()
    sel, result, seal, rows, oldfacts, facts, old, fixed = load()
    verified = audit(result, rows, old, fixed)
    # Write the complete arithmetic receipt before the first RGB read.
    write(H / 'ARITHMETIC_REVIEW_BEFORE_RGB.json', dict(completed=True, PASS=True, code=bind(__file__), result=bind(RESULT), checks=verified, RGB_read=False))
    for folder in ('figures', 'raw_crops'):
        (H / folder).mkdir()
    spec = importlib.util.spec_from_file_location('v134_saved_display_helpers', HELPER)
    h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
    pixels = h.SavedPixels()
    panels = [show(c, n + 1, rows, oldfacts, facts, old, fixed, pixels, h) for n, c in enumerate(sel['cases'])]
    out = dict(completed=True, PASS=True, code=bind(__file__), selection=bind(SELECTION), result=bind(RESULT), frozen_seal=bind(FROZEN / 'PREDICTION_SEAL.json'),
               arithmetic=bind(H / 'ARITHMETIC_REVIEW_BEFORE_RGB.json'), panels=panels,
               source_only=True, raw96_untouched_and_saved_input_pixel_hash_verified=True,
               model_checkpoint_fit_private_V112_PnP_calls=0, producer_or_evaluator_mutations=False)
    write(H / 'SAVED_REVIEW.json', out)
    print(json.dumps({'completed': True, 'PASS': True, 'review': bind(H / 'SAVED_REVIEW.json'), 'checks': verified}, ensure_ascii=False))


if __name__ == '__main__':
    main()
