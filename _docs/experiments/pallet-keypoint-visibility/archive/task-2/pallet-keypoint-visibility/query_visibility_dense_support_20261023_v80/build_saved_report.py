"""V80 saved-only report. No raw source RGB, model, weights or Torch imports.

Run only after parent supplies the completed fit SHA. Images are the unchanged
V79 diagnostic crop PNGs, not newly decoded source frames or newly sampled ROIs.
"""
from pathlib import Path
import argparse, hashlib, html, json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

R = Path(__file__).resolve().parent
P = R.parent
V79 = P / 'query_visibility_natural_occlusion_audit_20261023_v79'
GALLERY = V79 / 'natural_failure_gallery'
RUN = R / 'runs/SOURCE_DENSE_AUX_V1'
ANALYSIS = R / 'saved_analysis'
OUT = R / 'fixed_natural_gallery'
EXPECTED = [('YCB:000009:000241:000000', 6),
            ('YCB:000008:001872:000004', 6),
            ('YCB:000008:001299:000002', 1),
            ('YCB:000009:000225:000002', 6)]
PALETTE = np.asarray([[223, 103, 155], [78, 202, 133], [70, 78, 91]], np.uint8)

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for x in iter(lambda: f.read(1 << 20), b''): h.update(x)
    return h.hexdigest()

def bind(p): return {'path': str(Path(p).resolve()), 'sha256': sha(p)}
def load(p): return json.loads(Path(p).read_text())
def check(b): assert sha(b['path']) == b['sha256'], b['path']
def write(p, obj):
    with open(p, 'x') as f: f.write(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
def arrays(p):
    with np.load(p, allow_pickle=False) as z: return {k: z[k] for k in z.files}
def pvisible(z): return float(np.exp(-np.logaddexp(0., -float(z))))
def pct(x): return '—' if x is None else f'{100 * x:.1f}%'
def esc(x): return html.escape(str(x))
def link(p, label): return f'<a href="{esc(p)}">{esc(label)}</a>'

def color_probability(a, valid):
    a = np.clip(a, 0, 1)
    lo = np.asarray([32, 41, 62.])
    hi = np.asarray([245, 205, 75.])
    rgb = np.rint(lo[None, None] * (1 - a[..., None]) + hi[None, None] * a[..., None]).astype(np.uint8)
    rgb[~valid] = [12, 12, 16]
    return rgb

def grid_image(rgb):
    im = Image.fromarray(rgb).resize((252, 252), Image.Resampling.NEAREST)
    d = ImageDraw.Draw(im)
    for j in range(1, 9):
        xy = j * 28
        d.line((xy, 0, xy, 252), fill=(18, 25, 33), width=1)
        d.line((0, xy, 252, xy), fill=(18, 25, 33), width=1)
    d.rectangle((112, 112, 139, 139), outline=(255, 255, 255), width=2)
    return im

def panel(case, maps, scores, n):
    # English on the bitmap avoids silent font fallback. Korean explanations
    # and the full definitions are in the HTML immediately beside each figure.
    font = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    reg = ImageFont.truetype(font, 19)
    small = ImageFont.truetype(font, 16)
    bold = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 24)
    im = Image.new('RGB', (2016, 650), '#111c28'); d = ImageDraw.Draw(im)
    d.text((20, 14), f'{n + 1}. {case["id"]} / native{k(case)} / teacher {"V" if case["GT_label"] else "C"}', font=bold, fill='white')
    d.text((20, 49), f'V79-fixed ID; q-r={case["error_px"]:.3f}px; L={case["L"]:.3f}px; {case["rule"]}', font=reg, fill='#c8d7e7')
    valid = maps['valid9'][n]
    gt = PALETTE[maps['GT_class_CVO'][n]].copy(); gt[~valid] = [12, 12, 16]
    pred = PALETTE[maps['predicted_class_CVO'][n]].copy(); pred[~valid] = [12, 12, 16]
    joint = maps['joint_probability_CVO'][n]
    a = maps['pA'][n]
    conditional = maps['pS'][n]
    # PNGs are the exact V79 input crops, byte-bound before opening.
    for key in ('raw_local_RGB96', 'raw_target_RGB128'): check(case[key])
    local = Image.open(case['raw_local_RGB96']['path']).convert('RGB').resize((252, 252), Image.Resampling.NEAREST)
    target = Image.open(case['raw_target_RGB128']['path']).convert('RGB').resize((252, 252), Image.Resampling.NEAREST)
    ld = ImageDraw.Draw(local)
    ld.line((119, 125, 132, 125), fill='#39e4f2', width=2)
    ld.line((125, 119, 125, 132), fill='#39e4f2', width=2)
    rr = (np.asarray(case['r']) - np.asarray(case['q'])) / case['L']
    px, py = (rr + .5) * 252 - .5
    ld.rectangle((round(px - 3), round(py - 3), round(px + 3), round(py + 3)), outline='#ffb24b', width=2)
    images = [local, target, grid_image(gt), grid_image(pred),
              grid_image(color_probability(a, valid)),
              grid_image(color_probability(conditional, valid)),
              grid_image(color_probability(joint[1], valid))]
    titles = ['Same local RGB96', 'Same target RGB128', 'GT C / V / O', 'Aux joint argmax', 'pA = p(M0)', 'pS = p(V | M0)', 'Joint pV = pA*pS']
    for j, (tile, title) in enumerate(zip(images, titles)):
        x = 20 + j * 284
        d.text((x, 97), title, font=small, fill='white')
        im.paste(tile, (x, 126))
    d.text((20, 396), 'Local markers: q cyan / fixed GT reference r orange. Masks were never input. White map cell is nominal grid center.', font=reg, fill='#c8d7e7')
    d.text((20, 428), 'Map legend: C=M0 minus M4 (pink), V=M4 (green), O=outside M0 (gray), invalid (black). Argmax tie order C,V,O.', font=reg, fill='#c8d7e7')
    for j, name in enumerate(('V76', 'V77', 'V78', 'V80')):
        s = scores[name]
        correct = (s['prediction'] == ('VISIBLE' if case['GT_label'] else 'COVERED'))
        color = '#76e0aa' if correct else '#ffa28e'
        d.text((20 + j * 495, 477), f'{name} primary pV={s["p_visible"]:.6f}', font=bold, fill=color)
        d.text((20 + j * 495, 514), s['prediction'] + '  (fixed threshold 0.5)', font=reg, fill=color)
    d.text((20, 564), 'Auxiliary maps do not replace the primary readout or selector. pS is conditional, not unconditional visible probability.', font=reg, fill='#c8d7e7')
    d.text((20, 595), 'Nominal 9x9 cells have broad receptive fields; this is a weak source support proxy, not human semantic D/E/S.', font=reg, fill='#c8d7e7')
    path = OUT / f'{n + 1}_FIXED_NATURAL_AUX.png'; im.save(path)
    return path

def k(c): return int(c['native_k'])

def table(headers, rows):
    return '<table><thead><tr>' + ''.join(f'<th>{esc(x)}</th>' for x in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{esc(x)}</td>' for x in row) + '</tr>' for row in rows) + '</tbody></table>'

def main(fit_sha):
    # Every data read below is a saved-output read; no inference path is imported.
    assert sha(RUN / 'FIT_COMPLETE.json') == fit_sha
    fit = load(RUN / 'FIT_COMPLETE.json'); assert fit['completed'] and fit['optimizer_updates'] == 2000
    analysis_seal = load(ANALYSIS / 'SEAL.json'); assert analysis_seal['completed']
    for key in ('code', 'primary', 'auxiliary', 'fixed_four_maps', 'gate'): check(analysis_seal[key])
    assert not OUT.exists() and not (R / 'RESULTS_KO.html').exists() and not (R / 'RESULTS_KO.json').exists()
    comparison = load(ANALYSIS / 'PRIMARY_SOURCE_COMPARISON.json')
    auxiliary = load(ANALYSIS / 'AUXILIARY_MAP_DIAGNOSTICS.json')
    gate = load(R / 'SOURCE_ADMISSION_DECISION.json')
    old = load(GALLERY / 'GALLERY_METADATA.json'); cases = old['cases']
    assert [(c['id'], k(c)) for c in cases] == EXPECTED
    maps = arrays(ANALYSIS / 'FIXED_NATURAL_FOUR_MAPS.npz')
    assert list(maps['id']) == [c['id'] for c in cases]
    z = arrays(RUN / 'SELECTED_SOURCE_SCORES.npz')
    where = {str(x): i for i, x in enumerate(z['id'])}
    check(fit['auxiliary_logits'])
    raw_aux = arrays(fit['auxiliary_logits']['path'])
    assert np.array_equal(raw_aux['id'], z['id'])
    chosen = np.stack([raw_aux['aux_mask_logits'][where[c['id']], k(c)].astype(np.float64) for c in cases])
    probability = np.exp(-np.logaddexp(0., -chosen))
    maps['pA'], maps['pS'] = probability[:, 0], probability[:, 1]
    assert np.allclose(maps['pA'] * maps['pS'], maps['joint_probability_CVO'][:, 1], rtol=1e-6, atol=1e-8)
    del raw_aux, chosen, probability
    OUT.mkdir()
    figures = []; case_results = []
    for n, case in enumerate(cases):
        i = where[case['id']]
        scores = dict(case['scores'])
        logit = float(z['support_logit'][i, k(case)])
        scores['V80'] = dict(logit=logit, p_visible=pvisible(logit), prediction='VISIBLE' if logit >= 0 else 'COVERED')
        fig = panel(case, maps, scores, n); figures.append(bind(fig))
        case_results.append(dict(id=case['id'], native_k=k(case), GT_reference_local_teacher='VISIBLE' if case['GT_label'] else 'COVERED', q=case['q'], reference_GT_only=case['r'], L=case['L'], error_px=case['error_px'], scores=scores, same_old_RGB96=case['raw_local_RGB96'], same_old_target128=case['raw_target_RGB128'], figure=bind(fig), map_center=dict(GT_class_CVO=int(maps['GT_class_CVO'][n, 4, 4]), aux_class_CVO=int(maps['predicted_class_CVO'][n, 4, 4]), probabilities_CVO=maps['joint_probability_CVO'][n, :, 4, 4].tolist())))
    contact = Image.new('RGB', (2016, 650 * 4), '#111c28')
    for j, fig in enumerate(figures): contact.paste(Image.open(fig['path']), (0, j * 650))
    contact_path = OUT / 'CONTACT_SHEET.png'; contact.save(contact_path)
    model = load(RUN / 'MODEL_FREEZE.json')
    source_rows = [x for x in comparison['rows'] if x['domain'] == 0 and x['source'] == 'YCB_PUBLIC_REAL']
    pair_rows = [x for x in comparison['pairs'] if x['source'] == 'YCB_PUBLIC_REAL']
    strict = comparison['strict535']
    decision = 'SOURCE PASS' if gate['source_candidate_pass'] else 'HOLD'
    result = dict(completed=True, status=decision, fully_solved=False, model=model, gate=gate,
                  primary_public_natural=source_rows, public_controlled_pairs=pair_rows,
                  strict535=strict, auxiliary=auxiliary, fixed_cases=case_results,
                  figures=figures, contact_sheet=bind(contact_path),
                  bindings=dict(fit=bind(RUN / 'FIT_COMPLETE.json'), analysis=bind(ANALYSIS / 'SEAL.json'), gallery_metadata=bind(GALLERY / 'GALLERY_METADATA.json'), gallery_selection=old['selection_lock'], report_builder=bind(__file__)),
                  interpretation=dict(primary_vs_auxiliary='Primary FP32 support only selects/checks eligibility. Auxiliary FP16 maps are diagnostics.', conditional_ps='pS means visible conditional on expected support M0; joint pV=pA*pS.', source_labels='Reference-local mask-support V/C/U; C is not certified causal external E, O is not physical object absence, and self proxy remains separate.', identifiability='Fifty extra loss-only decoder parameters add supervision, not a new explicit target-instance cue at inference.', alignment='GT9 samples the frozen nominal P3 grid. Fused cells have broad receptive fields; unchanged local RGB branch phase is not a pixel-perfect native mask decoder.', crop_selection='Same four V79 outcome-cell examples, fixed before V80; cannot estimate prevalence.', selector='Same original natural VAL four-stratum BCE, earliest minimum among 20 scheduled steps; no auxiliary or current selection.'),
                  current223_reads=0, source_raw_RGB_decodes=0, model_forward=0, new_fit=0, PnP=0)
    write(R / 'RESULTS_KO.json', result)
    overview = table(['모델', '자연 TRAIN C recall', '자연 VAL C precision', '자연 VAL C recall', '자연 VAL V recall', 'VAL 양쪽 정답'], [[name,
        pct(next(x for x in source_rows if x['model'] == name and x['split'] == 'train')['C_recall']),
        pct(next(x for x in source_rows if x['model'] == name and x['split'] == 'val')['C_precision']),
        pct(next(x for x in source_rows if x['model'] == name and x['split'] == 'val')['C_recall']),
        pct(next(x for x in source_rows if x['model'] == name and x['split'] == 'val')['V_recall']),
        f"{next(x for x in pair_rows if x['model'] == name and x['split'] == 'val')['both']}/{next(x for x in pair_rows if x['model'] == name and x['split'] == 'val')['N']} ({pct(next(x for x in pair_rows if x['model'] == name and x['split'] == 'val')['both_rate'])})"] for name in ('V76', 'V77', 'V78', 'V80')])
    strict_table = table(['모델', '고정 V 정답 / 459', '고정 C 정답 / 76', 'V recall', 'C recall'], [[x['model'], f"{x['CM_rows_GT_CV_cols_pred_CV'][1][1]}/459", f"{x['CM_rows_GT_CV_cols_pred_CV'][0][0]}/76", pct(x['V_recall']), pct(x['C_recall'])] for x in strict])
    check_table = table(['사전 기준', 'V80', '하한', '판정'], [[name, pct(x['value']), pct(x['minimum']), 'PASS' if x['PASS'] else 'FAIL'] for name, x in gate['checks'].items()])
    aux_rows = [x for x in auxiliary['rows'] if x['domain'] == 0 and x['source'] == 'YCB_PUBLIC_REAL' and x['split'] == 'val']
    aux_table = table(['밀집 진단 분모', '유효 cell', 'GT C / V / O cell', 'C / V / O recall'], [[x['population'], x['Ncells'], ' / '.join(map(str, x['GT_pixel_count'])), ' / '.join(pct(t) for t in x['correct_per_GT'])] for x in aux_rows])
    body = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>V80 밀집 보조 감독 결과</title>
<style>body{{font-family:Arial,"NanumGothic",sans-serif;max-width:1480px;margin:32px auto;padding:0 24px;color:#17222b;line-height:1.65}}h1{{font-size:32px}}h2{{margin-top:36px}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:10px;border:1px solid #bdcad3;text-align:left}}th{{background:#e8f0f5}}.notice{{padding:18px;background:#edf4f8;border-left:5px solid #527a90}}.warn{{background:#fff1e9;border-color:#bb653f}}img{{width:100%;height:auto}}small{{color:#4b5c6b}}a{{color:#175575}}code{{background:#edf1f4}}</style>
<h1>V80: {decision} — 밀집 보조 감독만 추가한 통제 실험</h1>
<p class="notice warn">팔레트의 실제 가림 판별 문제가 해결됐다고 주장하지 않습니다. 이번 비교는 공개·합성 source의 저장 출력만 사용했습니다. 현재 평가 223장은 읽거나 추론하지 않았고, PnP·새 backbone 실행도 없습니다.</p>
<p>V77의 같은 6개 입력·41,237개 primary 경로·V75 초기 가중치·96-query sampler·2,000 update·고정 문턱 0.5를 유지했습니다. fused 24×9×9에 1×1 decoder 50개 매개변수만 추가해 총 41,287개입니다. primary BCE에 0.125×M0 BCE + 0.125×조건부 visible BCE를 더했습니다. source 자연 VAL의 네 source×class strata macro BCE만으로 {model['best_update']} update를 선택했습니다.</p>
<h2>자연 가림과 제어된 before/after는 별도 분모</h2>{overview}
<p>자연 VAL은 변경하지 않은 YCB 원본의 admitted V/C 쿼리입니다. 제어된 pair는 같은 q·L에서 원본 V와 합성 opaque after C를 모두 맞힌 비율입니다. 자연과 제어 가림 사이 성능 차이만으로 합성 이음새, donor 정체성 또는 순수 과적합 중 한 원인이 입증되지는 않습니다.</p>
<h2>q-r≤3px · STRONG · nominal 중심 cell의 같은 535개</h2>{strict_table}
<p>535개는 V79에서 고정한 V 459 / C 76입니다. C 76개는 실제 q 중심과 native 3×3도 모두 C-mask였으므로 이 부분집합의 오류를 단순한 q-r 위치 차이만으로 설명할 수 없습니다. C는 reference 주변 expected support가 visible mask에 없는 source proxy이며, 사람의 외부 가림 E 정답과 같지 않습니다.</p>
<h2>사전에 정한 source 기준</h2>{check_table}
<p>체크포인트·문턱을 다시 고르지 않습니다. source PASS라도 실사 semantic 해결이나 자동 current 평가 허가가 아닙니다.</p>
<h2>밀집 출력은 primary와 분리된 진단</h2>{aux_table}
<p>pA=p(M0), pS=p(V|M0)이며 pV=pA×pS, pC=pA×(1−pS), pO=1−pA로 계산했습니다. 표는 유효 native/cache 교집합 9×9 cell의 argmax 진단입니다. O를 C로 학습하지 않았고, invalid·unknown은 loss에서 제외했습니다. pS 자체를 보일 확률로 해석하면 안 됩니다. class-balanced loss의 출력이 자연 cell 빈도에 대해 보정됐다는 주장도 하지 않습니다.</p>
<h2>V79에 고정된 같은 네 사례</h2>
<p>기존 RAW_RGB96·target128 PNG를 그대로 사용했습니다. GT map, conditional pS, joint pV는 별도 패널입니다. 주황 r은 평가용 reference이며 모델 입력이 아닙니다. nominal 9×9 map은 frozen P3 좌표의 감독 격자이고 각 fused feature의 receptive field는 넓습니다. 기존 RGB branch의 phase와 해상도는 바꾸지 않았으므로 native 픽셀을 정확하게 분할한 출력으로 부르지 않습니다.</p>'''
    for c in case_results:
        body += f'<figure><img src="{esc(c["figure"]["path"])}" alt="고정 source 사례 {esc(c["id"])}"><figcaption>{esc(c["id"])} / native {c["native_k"]}: source reference-local {c["GT_reference_local_teacher"]}, q-r {c["error_px"]:.3f}px. 결과를 보고 새로 고른 사례가 아닙니다.</figcaption></figure>'
    body += '<h2>해석 범위와 원자료</h2><p>보조 decoder는 동일한 target crop·query set·context가 target ownership을 구분하도록 추가 감독을 줍니다. inference에 새 target-instance 증거가 생긴 것은 아니며, 인접 target이나 가려진 형상이 같은 입력에서 모호한 문제를 이 실험만으로 해결하지 못합니다. pooled DINO와의 차이에는 backbone·공간 readout·head·초기화가 함께 포함되므로 단일 representation 효과로 환원하지 않습니다.</p><p>'
    refs = [(RUN / 'MODEL_FREEZE.json', '선택·모델 동결'), (RUN / 'HISTORY.json', '20-step 선택 이력'), (ANALYSIS / 'PRIMARY_SOURCE_COMPARISON.json', 'primary 전체 분모'), (ANALYSIS / 'AUXILIARY_MAP_DIAGNOSTICS.json', '밀집 혼동행렬'), (R / 'SOURCE_ADMISSION_DECISION.json', '사전 source 기준'), (GALLERY / 'GALLERY_METADATA.json', '고정 사례·원 crop provenance'), (R / 'EXPERIMENT_DECISIONS_BEFORE_GT_BUILD.json', '실험 전 결정'), (R / 'RESULTS_KO.json', '보고서 JSON')]
    body += ' · '.join(link(p, label) for p, label in refs) + '</p></html>'
    with open(R / 'RESULTS_KO.html', 'x') as f: f.write(body)
    write(OUT / 'GALLERY_METADATA.json', dict(completed=True, cases=case_results, contact_sheet=bind(contact_path), parent_gallery=bind(GALLERY / 'GALLERY_METADATA.json'), fixed_map_source=bind(ANALYSIS / 'FIXED_NATURAL_FOUR_MAPS.npz'), report_builder=bind(__file__), new_raw_RGB_decode=0, model_forward=0, new_fit=0, current223=0, PnP=0))
    write(OUT / 'BUILD_SEAL.json', dict(completed=True, report=bind(R / 'RESULTS_KO.html'), report_JSON=bind(R / 'RESULTS_KO.json'), gallery_metadata=bind(OUT / 'GALLERY_METADATA.json'), figures=figures, contact_sheet=bind(contact_path), source_analysis=bind(ANALYSIS / 'SEAL.json'), fit=bind(RUN / 'FIT_COMPLETE.json'), code=bind(__file__), current_forward_fit_PnP=0))
    print(json.dumps(bind(OUT / 'BUILD_SEAL.json')), flush=True)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--fit-sha', required=True)
    main(ap.parse_args().fit_sha)
