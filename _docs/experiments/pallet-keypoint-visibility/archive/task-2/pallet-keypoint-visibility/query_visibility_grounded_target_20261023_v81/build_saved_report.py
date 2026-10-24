"""Sealed V81 saved-mask report/gallery. No model or feature imports.

The four source IDs and q/L come from V79. Source RGB is decoded only for the
new selected-detection overlay, byte-checked against both prior crop metadata
and the detector's saved RGB digest. No crop/query/detection is selected by GT.
"""
from pathlib import Path
import argparse, hashlib, html, json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

R = Path(__file__).resolve().parent
P = R.parent
V79 = P / 'query_visibility_natural_occlusion_audit_20261023_v79'
V80 = P / 'query_visibility_dense_support_20261023_v80'
G79 = V79 / 'natural_failure_gallery'
OUT = R / 'fixed_natural_gallery'
EXPECTED = [('YCB:000009:000241:000000', 6), ('YCB:000008:001872:000004', 6),
            ('YCB:000008:001299:000002', 1), ('YCB:000009:000225:000002', 6)]

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for x in iter(lambda: f.read(1 << 20), b''): h.update(x)
    return h.hexdigest()
def bind(p): return {'path': str(Path(p).resolve()), 'sha256': sha(p)}
def check(b): assert sha(b['path']) == b['sha256'], b['path']
def read(p): return json.loads(Path(p).read_text())
def lines(p): return [json.loads(x) for x in Path(p).read_text().splitlines()]
def arrays(p):
    with np.load(p, allow_pickle=False) as z: return {k: z[k] for k in z.files}
def write(p, obj):
    with open(p, 'x') as f: json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
def pct(x): return '—' if x is None else f'{100*x:.1f}%'
def decimal(x): return '—' if x is None else f'{x:.3f}'
def esc(x): return html.escape(str(x))
def link(p, text): return f'<a href="{esc(p)}">{esc(text)}</a>'

def sample(mask, q, side, n, hw):
    """Literal old native9/RGB96 coordinates; NN validity kept separately."""
    h, w = hw
    offsets = ((np.arange(n) + .5) / n - .5) * side
    xx, yy = np.meshgrid(q[0] + offsets, q[1] + offsets)
    px, py = np.floor(xx + .5).astype(np.int64), np.floor(yy + .5).astype(np.int64)
    valid = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h) & (px >= 0) & (px < w) & (py >= 0) & (py < h)
    if mask is None: return None, valid
    assert mask.shape == (h, w) and mask.dtype == np.bool_
    out = np.zeros((n, n), bool); out[valid] = mask[py[valid], px[valid]]
    return out, valid

def grid(mask, valid, green=True):
    if mask is None:
        im = Image.new('RGB', (192, 192), '#28283c'); d = ImageDraw.Draw(im)
        for j in range(-180, 384, 20): d.line((j, 0, j+192, 192), fill='#54516e', width=5)
        d.rectangle((38, 75, 154, 115), fill='#222235')
        d.text((47, 85), 'U / NO MASK', fill='white', font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14))
        return im
    a = np.zeros((*mask.shape, 3), np.uint8); a[:] = [63, 69, 79]
    a[mask] = [76, 205, 138] if green else [232, 234, 239]
    a[~valid] = [8, 10, 14]
    im = Image.fromarray(a).resize((192, 192), Image.Resampling.NEAREST)
    if mask.shape == (9, 9):
        d = ImageDraw.Draw(im)
        for j in range(1, 9):
            v = round(j*192/9); d.line((v, 0, v, 192), fill='#1b2730'); d.line((0, v, 192, v), fill='#1b2730')
        d.rectangle((85, 85, 106, 106), outline='white', width=2)
    return im

def marker(d, xy, color, radius=5):
    x, y = map(float, xy); d.line((x-radius, y, x+radius, y), fill=color, width=2); d.line((x, y-radius, x, y+radius), fill=color, width=2)

def render_case(n, case, association, rgb_record, detection, maps, source_rgb, previous_scores, qprediction, target_iou):
    bold = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 23)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)
    im = Image.new('RGB', (1840, 690), '#111c28'); d = ImageDraw.Draw(im)
    gt = 'V' if case['GT_label'] else 'C'
    state = ('NOT_VISIBLE_MASK', 'VISIBLE_MASK', 'U')[qprediction]
    d.text((20, 12), f'{n+1}. {case["id"]} / native{case["native_k"]} / reference support {gt}', fill='white', font=bold)
    d.text((20, 47), f'V81 at unchanged q: {state} | whole target M4 IoU={target_iou:.3f} | q-r={case["error_px"]:.3f}px | L={case["L"]:.3f}px', fill='#d4dfea', font=font)
    matched = association['status'] == 'ASSOCIATED'
    if matched:
        j = association['selected_detection_index']; phrase = str(detection['text_labels'][j])
        detail = f'Input-only association: IoU={association["association_iou"]:.3f}, score={association["selected_detection_score"]:.3f}, predicted phrase="{phrase}"'
    else: phrase = None; detail = 'No input-only association: ' + association['reason'] + ' (U, not zero mask)'
    d.text((20, 80), detail, fill='#d4dfea', font=font)
    h, w = source_rgb.shape[:2]
    scene = Image.fromarray(source_rgb).resize((440, 330), Image.Resampling.BILINEAR)
    sd = ImageDraw.Draw(scene); scale = np.asarray([440/w, 330/h])
    box = np.asarray(association['query_box_xyxy'], float).reshape(2, 2)*scale
    sd.rectangle(tuple(box.ravel()), outline='#53a6ff', width=3)
    if matched:
        box2 = np.asarray(association['selected_box_xyxy_prompt'], float).reshape(2, 2)*scale
        sd.rectangle(tuple(box2.ravel()), outline='#75ed98', width=3)
    marker(sd, np.asarray(case['q'])*scale, '#ffdf52')
    marker(sd, np.asarray(case['r'])*scale, '#3be6fa')
    d.text((20, 123), 'Native scene: input box blue / selected green', fill='white', font=small)
    im.paste(scene, (20, 151))
    check(case['raw_local_RGB96']); check(case['raw_target_RGB128'])
    local = Image.open(case['raw_local_RGB96']['path']).convert('RGB').resize((192, 192), Image.Resampling.NEAREST)
    target = Image.open(case['raw_target_RGB128']['path']).convert('RGB').resize((192, 192), Image.Resampling.NEAREST)
    ld = ImageDraw.Draw(local); marker(ld, [95.5, 95.5], '#ffdf52', 4)
    ref = (np.asarray(case['r'])-np.asarray(case['q']))/case['L']
    marker(ld, (ref+.5)*192-.5, '#3be6fa', 4)
    panels = [(480, 'Same RGB96 / q,r', local), (694, 'Same target RGB128', target),
              (908, 'GT M0 / nominal9', grid(maps['M0_9'], maps['valid9'], False)),
              (1122, 'GT M4 / nominal9', grid(maps['M4_9'], maps['valid9'])),
              (1336, 'SAM mask / nominal9', grid(maps['SAM_9'], maps['valid9'])),
              (1550, 'SAM mask / same L96', grid(maps['SAM_96'], maps['valid96']))]
    for x, title, tile in panels:
        d.text((x, 123), title, fill='white', font=small); im.paste(tile, (x, 151))
    d.text((480, 363), 'GT: M0 expected isolated support; M4 observed visible support.', fill='#d4dfea', font=font)
    d.text((480, 391), 'SAM: green=visible mask, gray=mask0, hatch=U/no mask. Mask0 is not semantic E/S.', fill='#d4dfea', font=font)
    d.text((480, 419), 'q yellow / fixed r cyan (GT-only marker); no recenter, L change or oracle detection.', fill='#d4dfea', font=font)
    d.text((20, 502), 'Stored learned-head outputs (fixed 0.5); V81 above is a binary mask sample, not a learned probability:', fill='#d4dfea', font=font)
    for j, name in enumerate(('V76', 'V77', 'V78', 'V80')):
        s = previous_scores[name]; correct = s['prediction'] == ('VISIBLE' if case['GT_label'] else 'COVERED')
        color = '#76e0aa' if correct else '#ffa28e'
        d.text((20+j*450, 541), f'{name} pV={s["p_visible"]:.6f}', fill=color, font=bold)
        d.text((20+j*450, 578), s['prediction'], fill=color, font=font)
    d.text((20, 625), 'Same four V79 source cases, fixed before V81. Weak source pixel/reference support; not human semantic D/E/S gold.', fill='#d4dfea', font=font)
    path = OUT / f'{n+1}_FIXED_GROUNDED_MASK.png'; im.save(path)
    return path, phrase

def table(headers, rows):
    return '<table><tr>'+''.join('<th>'+esc(x)+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</table>'

def build(sam_seal_path, sam_sha, evaluation_path, evaluation_sha):
    assert sha(sam_seal_path) == sam_sha and sha(evaluation_path) == evaluation_sha
    seal, evseal = read(sam_seal_path), read(evaluation_path)
    assert seal['completed'] and seal['PASS'] and evseal['completed'] and evseal['PASS']
    for file in seal['outputs']: check(file)
    for file in evseal['files']: check(file)
    check(seal['detector_output_seal']); detector_seal = read(seal['detector_output_seal']['path'])
    check(detector_seal['summary']); detector_summary = read(detector_seal['summary']['path'])
    check(seal['summary']); sam_summary = read(seal['summary']['path'])
    evaldir = evaluation_path.parent
    results = read(evaldir/'SOURCE_MASK_RESULTS.json')
    assert results['SAM_output_seal'] == bind(sam_seal_path)
    assert not OUT.exists() and not (R/'RESULTS_KO.html').exists() and not (R/'RESULTS_KO.json').exists()
    old = read(G79/'GALLERY_METADATA.json'); cases = old['cases']
    assert [(x['id'], x['native_k']) for x in cases] == EXPECTED
    rows = lines(seal['output_masks']['path'])
    rgbrows = lines(seal['per_RGB']['path']); byrgb = {x['rgb_input_id']: x for x in rgbrows}
    evaluation = arrays(evaldir/'SAVED_MASK_EVALUATION_PER_QUERY.npz')
    assert len(rows) == 1035 and len(rgbrows) == 435
    by_original = {str(x): i for i, x in enumerate(evaluation['original_id'])}
    prior = read(V80/'fixed_natural_gallery/GALLERY_METADATA.json')
    assert [x['id'] for x in prior['cases']] == [x['id'] for x in cases]
    OUT.mkdir(); panels = []; records = []
    for n, case in enumerate(cases):
        i = by_original[case['id']]; a = rows[i]; rgb = byrgb[a['rgb_input_id']]
        assert evaluation['input_id'][i] == a['input_id']
        assert a['initial_points'][case['native_k']] == case['q'] and a['roi_side'] == case['L']
        assert np.array_equal(np.asarray(a['query_box_xyxy'], np.float64), case['box'])
        check(rgb['detections']); detection = arrays(rgb['detections']['path'])
        check(case['source_RGB']); assert rgb['image_sha256'] == case['source_RGB']['sha256']
        with Image.open(case['source_RGB']['path']) as im: source_rgb = np.asarray(im.convert('RGB')).copy()
        assert list(source_rgb.shape) == [*a['raw_hw'], 3]
        assert hashlib.sha256(source_rgb.tobytes()).hexdigest() == rgb['decoded_RGB_SHA256']
        gt_masks = []
        for key in ('M0', 'M4'):
            check(case['mask_pair'][key])
            with Image.open(case['mask_pair'][key]['path']) as im: gt_masks.append(np.asarray(im)>0)
        assert not (gt_masks[1] & ~gt_masks[0]).any()
        mask = None
        if a['mask_status'] == 'OK':
            assert a['status'] == 'ASSOCIATED'; check(a['mask'])
            mask = arrays(a['mask']['path'])['visible_mask']; assert mask.dtype == np.bool_ and mask.shape == tuple(a['raw_hw'])
        else: assert a['status'] == 'U' and a['mask'] is None
        q, side = np.asarray(case['q']), case['L']
        m0, valid9 = sample(gt_masks[0], q, side, 9, a['raw_hw'])
        m4, _ = sample(gt_masks[1], q, side, 9, a['raw_hw'])
        sam9, _ = sample(mask, q, side, 9, a['raw_hw'])
        sam96, valid96 = sample(mask, q, side, 96, a['raw_hw'])
        prediction = int(evaluation['prediction_CVU'][i, case['native_k']])
        assert prediction == (2 if mask is None else int(sam9[4, 4]))
        maps = dict(M0_9=m0, M4_9=m4, SAM_9=sam9, SAM_96=sam96, valid9=valid9, valid96=valid96)
        scores = prior['cases'][n]['scores']
        target_iou = float(evaluation['IoU_allrows_nomask0'][i])
        path, phrase = render_case(n, case, a, rgb, detection, maps, source_rgb, scores, prediction, target_iou)
        panels.append(bind(path))
        npz_path = OUT/f'{n+1}_FIXED_SAMPLED_MAPS.npz'
        np.savez_compressed(npz_path, M0_9=m0, M4_9=m4, native_valid9=valid9, predicted_mask9=np.full((9,9),2,np.int8) if mask is None else sam9.astype(np.int8), predicted_mask96=np.full((96,96),2,np.int8) if mask is None else sam96.astype(np.int8), native_valid96=valid96, q=np.asarray(case['q']), L=np.asarray(side))
        records.append(dict(id=case['id'], input_id=a['input_id'], native_k=case['native_k'], fixed_q=case['q'], fixed_r_GT_only=case['r'], L=side, unchanged_initial_query_box=a['query_box_xyxy'], association=a, selected_predicted_phrase=phrase, prior_scores=scores, V81_mask_at_q=['NOT_VISIBLE_MASK','VISIBLE_MASK','U'][prediction], GT_reference_support=['C','V'][case['GT_label']], GT_query_support='O' if not m0[4,4] else 'V' if m4[4,4] else 'C', whole_target_M4_IoU_no_mask0=target_iou, GT_nominal9_valid=int(valid9.sum()), SAM_nominal9_positive=None if sam9 is None else int((sam9&valid9).sum()), same_old_RGB96=case['raw_local_RGB96'], same_old_target128=case['raw_target_RGB128'], source_RGB=case['source_RGB'], source_RGB_decode_parity=True, saved_mask=a['mask'], selected_detection=rgb['detections'], figure=bind(path), sampled_maps=bind(npz_path)))
    contact = Image.new('RGB', (1840, 690*4), '#111c28')
    for n, panel in enumerate(panels): contact.paste(Image.open(panel['path']), (0,n*690))
    contact.save(OUT/'CONTACT_SHEET.png')
    benchmarks = results['benchmarks']
    all_metrics = next(x for x in benchmarks if x['source'] == 'ALL')
    report = dict(completed=True, stage='SOURCE_ONLY_FEASIBILITY_PILOT', fully_solved=False,
                  benchmarks=benchmarks, fixed_four=records, figures=panels, contact_sheet=bind(OUT/'CONTACT_SHEET.png'),
                  no_automatic_current_evaluation=True, no_model_or_threshold_selection_from_current=True,
                  target_mismatch='Binary SAM mask sample at initialq is separate from native reference/crop semantic visibility; no learned D/E/S.',
                  no_GT_category_or_mask_selection=True, no_new_fit=True, current223=0, report_builder_model_forward=0, PnP=0,
                  completed_source_run_execution=dict(detector_unique_RGB=detector_summary['unique_RGB'], detector_forward_calls=detector_summary['source_forward_calls'], SAM_embedding_API_calls=sam_summary['embedding_calls'], SAM_mask_predict_API_calls=sam_summary['associated'], detector_state_unchanged=detector_summary['state_unchanged'], SAM_state_unchanged=sam_summary['state_before']==sam_summary['state_after']),
                  source_RGB_decodes_for_figure_only=4, original_inputs_and_labels_unchanged=True,
                  bindings=dict(SAM=bind(sam_seal_path), evaluation=bind(evaluation_path), prior_gallery=bind(G79/'GALLERY_METADATA.json'), prior_selection=old['selection_lock'], prior_scores=bind(V80/'fixed_natural_gallery/GALLERY_METADATA.json'), code=bind(__file__)))
    write(R/'RESULTS_KO.json', report)
    ownership = table(['Source','전체 instance','mask 있음','U/no match','U율','전체 mIoU(U=0)','mask 있음 조건부 mIoU'], [[x['source'],x['target_instances'],x['mask_IoU']['usable_selected_mask_N'],x['mask_IoU']['nomatch_N'],pct(x['mask_IoU']['mask_hold_rate']),decimal(x['mask_IoU']['all_rows_no_mask0_mean']),decimal(x['mask_IoU']['usable_selected_mask_conditional_mean'])] for x in benchmarks])
    def cvtable(key):
        rows=[]
        for x in benchmarks:
            m=x[key]; rows.append([x['source'],m['N'],str(m['CM_rows_GT_CV_cols_pred_CVU']),pct(m['C_precision_known_CV_conditional']),pct(m['C_end_to_end_recall_U_is_miss']),pct(m['V_end_to_end_recall_U_is_miss']),f"{m['hold_N']} ({pct(m['hold_rate'])})"])
        return table(['Source','분모N','GT C/V × pred C/V/U','C precision*','C recall(U miss)','V recall(U miss)','보류'],rows)
    body=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>V81 고정 텍스트·대상 mask source 평가</title><style>body{{font-family:Arial,"NanumGothic",sans-serif;max-width:1480px;margin:32px auto;padding:0 24px;color:#17222b;line-height:1.65}}h1{{font-size:30px}}h2{{margin-top:34px}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:10px;border:1px solid #bdcad3;text-align:left}}th{{background:#e8f0f5}}.notice{{padding:18px;background:#edf4f8;border-left:5px solid #527a90}}.warn{{background:#fff1e9;border-color:#bb653f}}img{{width:100%;height:auto}}small{{color:#4b5c6b}}a{{color:#175575}}</style>
<h1>V81: 고정 category 요청과 입력 box 대응을 이용한 source mask 진단</h1>
<p class="notice warn">현재 223장은 사용하지 않았습니다. 이는 공개·합성 VAL에서 시행한 frozen foundation component 실험이며 사람의 semantic D/E/S 판별이 해결됐다는 결과가 아닙니다. 새 학습·PnP·query 수정·GT category별 prompt 선택은 없습니다.</p>
<p>GroundingDINO tiny에 모든 이미지마다 같은 YCB 21종+pallet의 22개 phrase(77 tokens)를 입력했습니다. 고정 검출 문턱 0.4 / text 0.3, 기존 initial-q box와 검출 box의 최대 IoU≥0.1로만 대응했습니다. 동률은 detector score와 원 query 순서로 정하고 NMS·GT class·GT mask로 고르지 않았습니다. 선택한 box 하나만 frozen SAM2.1 base-plus에 입력하며 point prompt와 mask quality 선택은 없습니다.</p>
<p>완료된 source 실행은 detector RGB {detector_summary['unique_RGB']}개·반복 검산 포함 forward {detector_summary['source_forward_calls']}회, SAM image embedding API {sam_summary['embedding_calls']}회·mask predict {sam_summary['associated']}회입니다. 실패한 API 형식 검사는 별도 보존됐습니다. 모델 상태는 실행 전후 동일하며, 이 보고서 생성에서는 저장 결과만 읽고 모델을 호출하지 않았습니다.</p>
<h2>대상 소유 mask: U를 분모에서 숨기지 않음</h2>{ownership}
<p>전체 instance는 {all_metrics['target_instances']}개이며 U/no-match는 {all_metrics['mask_IoU']['nomatch_N']}개입니다. 전체 mIoU는 no-mask를 0으로 포함하고, mask가 있는 경우의 조건부 평균을 별도로 표시합니다. U는 실제 all-zero mask가 아닙니다. 실제 양쪽 빈 mask만 IoU=1이며 no prediction은 GT가 비어도 0입니다.</p>
<h2>원래 reference-local primary를 유지한 mask-at-q component 비교</h2>{cvtable('REFERENCE_LOCAL_PRIMARY_UNCHANGED_mask_at_q_component')}
<p>*Precision은 GT가 C/V로 알려진 primary subset 조건부 값입니다. Recall은 그 GT class의 U도 miss로 포함했습니다. mask0를 semantic E/S로 바꾸지 않았고 GT O/unknown을 C 분모에 합치지 않았습니다. 원래 학습 head의 scalar 출력과 동일한 target을 학습했다는 뜻이 아닙니다.</p>
<h2>별도 목표: 실제 q pixel이 GT M0 안인 C/V</h2>{cvtable('SEPARATE_QUERY_PIXEL_CV_given_GT_M0_and_native_valid')}
<p>이 표는 q를 floor(q+0.5)로 읽은 native pixel support입니다. reference/corner 의미를 대신하는 primary 변경이 아닙니다. q가 GT M0 밖인 O와 native invalid는 원 JSON에서 별도 보존했습니다.</p>
<h2>같은 strict 535개 — V459 / C76</h2>{cvtable('strict535_same_frozen_IDs')}
<p>V79에 고정한 q-r≤3px·STRONG·nominal 중심 cell의 같은 ID입니다. 이 cohort에서 qGT와 기존 reference label이 일치한다는 사전 검산을 유지했습니다. U 포함 end-to-end와 accepted-only 값을 바꾸어가며 선택하지 않습니다.</p>
<h2>V79에 고정된 실제 네 사례</h2><p>같은 RGB96·target128 PNG, q/L/native slot을 재사용했습니다. scene overlay의 파랑은 입력 q-box, 초록은 실제 선택한 detector box, 노랑 q와 청록 r입니다. r/GT mask는 그림·평가용이며 모델에 입력되지 않았습니다. nominal9 mask와 같은 L의 96 mask를 새 readout 선택 없이 함께 표시합니다. 회색 mask0는 보이지 않는 source mask sample이며 O/E/S 원인을 뜻하지 않습니다.</p>'''
    for record in records:
        body+=f'<figure><img src="{esc(record["figure"]["path"])}" alt="고정 source 사례 {esc(record["id"])}"><figcaption>{esc(record["id"])} / native {record["native_k"]}: GT support {record["GT_reference_support"]}, V81 {record["V81_mask_at_q"]}. 입력·사례 선택은 V81 결과로 바꾸지 않았습니다.</figcaption></figure>'
    body+='''<h2>확인할 수 있는 것과 남는 차이</h2><p>이 실험은 범용 object name 요청과 입력 box 대응이 target-owned visible mask를 만들 수 있는지 조사합니다. 같은 category의 다른 instance, 가린 물체를 가림 물체로 잘못 대응하는 문제, source 밖의 pallet 의미 대응은 남습니다. 문자열은 고정 caption에서 나온 predicted phrase이며 해당 instance의 GT class가 아닙니다. Source pixel mask 성능을 사람의 reference-keypoint semantic visibility 성능으로 옮겨 말하지 않습니다.</p><p>공개 YCB q는 이전 V76의 GT-derived source emulation이며 실제 배포 initializer 성능이 아닙니다. M0/M4는 source의 투영/가시 support proxy입니다. foundation pretraining overlap은 인증되지 않았고 반복 사용한 source VAL 개발 진단이므로 독립 held-out 성공 주장도 하지 않습니다. raw token full logits를 저장하지 않아 text&gt;0.3 phrase 문자열의 독립 재생성은 감사 범위 밖이며, 실제 선택에 쓰는 raw maxscores·boxes·순서·association은 별도 감사합니다.</p><h2>실행 오류와 성능 판단의 분리</h2><p>Detector의 padded token은 공식 구현상 −inf인데 처음의 전값 finite guard가 맞지 않아 실제 source 실행 전에 수정·검증했습니다. 첫 SAM 실행도 공개 API가 내부 bool mask를 float32 0/1로 반환한다는 dtype 경계를 잘못 예상한 guard에서 중단했습니다. 실패 코드·잠금·로그는 보존하고, 공식 API의 정확한 0/1만 bool로 옮기도록 고쳤습니다. 새 문턱·모델·prompt·후보 선택을 도입하지 않았으며, 이 infrastructure 수정은 성능 개선 실험이나 재선택이 아닙니다.</p><p>'''
    refs=[(evaluation_path,'전체 saved 평가 seal'),(evaldir/'SOURCE_MASK_RESULTS.json','전체 분모·혼동행렬 JSON'),(sam_seal_path,'SAM 출력 seal'),(Path(seal['detector_output_seal']['path']),'Detector seal'),(R/'EXPERIMENT_DECISIONS_BEFORE_ASSETS_AND_SOURCE.json','실행 전 고정 결정'),(R/'RESULTS_KO.json','보고서 JSON')]
    body+=' · '.join(link(p,t) for p,t in refs)+'</p></html>'
    with open(R/'RESULTS_KO.html','x') as f:f.write(body)
    write(OUT/'GALLERY_METADATA.json',dict(completed=True,cases=records,figures=panels,contact_sheet=bind(OUT/'CONTACT_SHEET.png'),same_fixed_V79_four=True,no_GT_selection_of_detection_or_masks=True,source_RGB_figure_only_decodes=4,model_forward_fit_current_PnP=0))
    write(OUT/'BUILD_SEAL.json',dict(completed=True,PASS=True,code=bind(__file__),report=bind(R/'RESULTS_KO.html'),report_JSON=bind(R/'RESULTS_KO.json'),gallery=bind(OUT/'GALLERY_METADATA.json'),figures=panels,contact=bind(OUT/'CONTACT_SHEET.png'),SAM=bind(sam_seal_path),evaluation=bind(evaluation_path),model_forward_fit_current_PnP=0))
    print(json.dumps(bind(OUT/'BUILD_SEAL.json')),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--sam-seal',type=Path,required=True);parser.add_argument('--sam-sha',required=True);parser.add_argument('--evaluation-seal',type=Path,required=True);parser.add_argument('--evaluation-sha',required=True)
    a=parser.parse_args();build(a.sam_seal,a.sam_sha,a.evaluation_seal,a.evaluation_sha)
