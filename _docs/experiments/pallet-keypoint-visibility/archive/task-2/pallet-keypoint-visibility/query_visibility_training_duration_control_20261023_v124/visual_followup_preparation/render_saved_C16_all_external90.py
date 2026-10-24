"""All paired external-E cases, saved current crops only; no model imports/calls."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import csv, hashlib, json, math, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import PIL

V = Path(__file__).resolve().parents[1]
P = V.parent
E = V / 'private_evaluation/C16_FROZEN_CURRENT223'
O = V / 'visual_followup/all_external90'
FONT = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'

def binding(p):
    p = Path(p)
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return {'path': str(p), 'sha256': h.hexdigest(), 'bytes': p.stat().st_size}

def checked(b):
    assert binding(b['path'])['sha256'] == b['sha256'], b['path']
    return Path(b['path'])

def read_bound(b):
    return json.loads(checked(b).read_text())

def write(p, value):
    with Path(p).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

def csv_rows(p):
    with Path(p).open(newline='') as f:
        return list(csv.DictReader(f))

def key(r):
    return r['id'], int(r['native_corner'])

def yes(s):
    return str(s).lower() == 'true'

def ah(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def main():
    assert not O.exists(), 'Preserve existing completed/partial outputs; do not overwrite.'
    evp = E / 'evaluation/EVALUATION_SEAL.json'
    ev = json.loads(evp.read_text())
    assert ev['completed'] and ev['PASS']
    inf = read_bound(ev['inference'])
    assert inf['completed'] and inf['PASS'] and inf['model_unchanged']
    assert (inf['slots'], inf['valid'], inf['U'], inf['additional_U']) == (1784, 1739, 45, 0)
    assert inf['D_rule'] == 'z<0' and not inf['current_used_for_selection']
    assert inf['source_model_freeze'] == ev['source_freeze']
    primary_b = next(b for b in ev['files'] if Path(b['path']).name == 'CURRENT_PRIMARY_RESULTS.json')
    primary = read_bound(primary_b)
    assert primary['completed'] and primary['primary_model'] == 'V124_C16_RAW'
    bb = primary['bindings']['bindings']
    score_b = next(b for b in inf['outputs'] if Path(b['path']).name == 'HEAD_ONLY_SCORES.csv')
    scores = csv_rows(checked(score_b))
    truth = csv_rows(checked(bb['evaluation_manifest']))
    prior = [json.loads(s) for s in checked(bb['prior_input_rows']).read_text().splitlines()]
    assert len(scores) == len(truth) == len(prior) == 1784
    tm = {key(r): r for r in truth}
    assert len(tm) == 1784
    oldlock_p = P / 'query_visibility_success_failure_visual_review_20261023_v117/SELECTION_LOCK_BEFORE_PIXELS.json'
    oldlock = json.loads(oldlock_p.read_text())
    packet_b = oldlock['current_original_RGB96_packet_recorded_binding']
    records = []
    paired_D = Counter()
    paired_E = Counter()
    for i, (s, p) in enumerate(zip(scores, prior)):
        assert key(s) == key(p)
        t = tm[key(s)]
        q = [float(s['query_x']), float(s['query_y'])]
        L = float(s['roi_side'])
        assert q == [p['query_x'], p['query_y']] == [float(t['query_x']), float(t['query_y'])]
        assert L == p['roi_side'] == float(t['v61_roi_side_raw_px'])
        assert yes(s['valid']) == p['valid']
        z = float(s['raw_query_aux_hidden_logit']) if yes(s['valid']) else None
        if z is not None:
            assert math.isfinite(z)
        pred = ('D' if z < 0 else 'E') if z is not None else 'U'
        assert s['authoritative_decision'] == ('UNKNOWN' if pred == 'U' else pred)
        if not yes(t['paired_same_query_RGB_D_vs_E']):
            continue
        assert t['human_state'] in ('DIRECT_VISIBLE', 'EXTERNAL_OCCLUDED')
        human = 'D' if t['human_state'] == 'DIRECT_VISIBLE' else 'E'
        (paired_D if human == 'D' else paired_E)[(t['group'], pred)] += 1
        if human == 'D' and pred != 'E':
            continue
        records.append({
            'original_native_row': i, 'id': s['id'], 'native_slot': int(s['native_corner']),
            'group': t['group'], 'material_display': 'WOOD' if t['group'] == 'WOOD_PRIMARY105' else 'SQUARE',
            'cohort': 'paired_same_query_RGB', 'human_label': human, 'human_state': t['human_state'],
            'human_label_source': t['human_label_source'], 'C16_prediction': pred, 'C16_raw_zE': z,
            'query_xy': q, 'original_L': L, 'original_input_valid': p['valid'],
            'image_recorded_binding': {'path': t['image'], 'sha256': t['image_sha256']},
            'reference_xy_recorded': [float(t['human_reference_x']), float(t['human_reference_y'])],
            'initial_reference_error_px_recorded': float(t['initial_reference_error_px']),
            'reference_not_used_to_crop_or_move_q': True,
            'E_reference_is_occluded_estimate_not_verified_coordinate_gold': human == 'E',
            'category': 'E_CORRECT' if human == 'E' and pred == 'E' else 'E_MISSED' if human == 'E' else 'D_FALSE_E',
        })
    external = [r for r in records if r['human_label'] == 'E']
    falseE = [r for r in records if r['human_label'] == 'D']
    assert len(external) == 90 and len(falseE) == 10
    assert Counter(r['group'] for r in external) == {'WOOD_PRIMARY105': 29, 'SQUARE118': 61}
    assert Counter(r['C16_prediction'] for r in external) == {'E': 35, 'D': 55}
    assert Counter(r['group'] for r in falseE) == {'WOOD_PRIMARY105': 10}
    for m in primary['tables']['D_vs_external_E_2x3']:
        if m['cohort'] == 'paired_same_query_RGB' and m['model'] == 'V124_C16_RAW':
            g = m['group']
            assert [paired_D[(g,k)] for k in ('D','E','U')] == [m['D_pred_D'], m['D_pred_hidden'], m['D_pred_U']]
            assert [paired_E[(g,k)] for k in ('D','E','U')] == [m['E_pred_D'], m['E_pred_hidden'], m['E_pred_U']]
    O.mkdir(parents=True)
    inputs = {'C16_evaluation_seal': binding(evp), 'C16_inference_seal': ev['inference'],
              'C16_primary_results': primary_b, 'C16_saved_scores': score_b,
              'evaluation_manifest': bb['evaluation_manifest'], 'original_input_rows': bb['prior_input_rows'],
              'RGB96_packet_from_existing_V117_lock': packet_b, 'V117_original_crop_lock': binding(oldlock_p),
              'existing_crop_display_renderer': binding(P / 'query_visibility_success_failure_visual_review_20261023_v117/render_saved_review.py'),
              'existing_C16_fixed4_renderer': binding(V / 'visual_followup_preparation/render_saved_C16_fixed4.py')}
    lock = {'completed': True, 'rule': 'Every paired human EXTERNAL_OCCLUDED row, plus every paired human DIRECT_VISIBLE predicted E; original 1784-row order retained within each category, no pixel/quality selection.',
            'external_E_rows': [r['original_native_row'] for r in external],
            'D_false_E_rows': [r['original_native_row'] for r in falseE],
            'counts': {'external_E':90,'E_correct':35,'E_missed_as_D':55,'D_false_E':10},
            'bindings':inputs, 'selected_before_any_new_RGB96_array_read':True,
            'diagnosis_only_no_future_training_threshold_or_arm_selection':True}
    write(O / 'ALL_ROWS_SELECTION_BEFORE_CROP_READ.json', lock)
    # Match V117 exactly: access only existing crop display arrays, never features/P3/full RGB.
    checked(packet_b)
    with np.load(packet_b['path'], allow_pickle=False) as z:
        rgb = z['rgb96']
        valid = z['crop_valid96']
    assert rgb.shape == (1784,3,96,96) and rgb.dtype == np.uint8
    assert valid.shape == (1784,1,96,96) and valid.dtype == np.bool_
    for r in records:
        n = r['original_native_row']
        r['RGB96_CHW_sha256'] = ah(rgb[n])
        r['valid96_sha256'] = ah(valid[n])
        r['valid96_fraction'] = float(valid[n].mean())
        r['source_packet_row'] = n
    artifacts = []
    titles = {'E_CORRECT': '외부 가림 E 정답: 인간 E → C16 E',
              'E_MISSED': '외부 가림 E 누락: 인간 E → C16 D',
              'D_FALSE_E': '보이는 D 오판: 인간 D → C16 E'}
    font = lambda n: ImageFont.truetype(FONT,n)
    W, H, COLS, PER = 224, 275, 6, 30
    for category in ('E_CORRECT','E_MISSED','D_FALSE_E'):
        rows = [r for r in records if r['category'] == category]
        for page, start in enumerate(range(0,len(rows),PER),1):
            subset = rows[start:start+PER]
            height = 147 + math.ceil(len(subset)/COLS)*H + 48
            sheet = Image.new('RGB',(COLS*W,height),'#eef3f7')
            d = ImageDraw.Draw(sheet)
            d.text((18,14),titles[category],font=font(28),fill='#142e43')
            d.text((18,55),f'전체 {len(rows)}개 중 {start+1}–{start+len(subset)} · 원행 순서 · {page}쪽',font=font(21),fill='#243d50')
            d.text((18,89),'무표시 실제 RGB96 ×2 · q=(47.5,47.5) · 기존 q/L/라벨 유지 · 재crop/재추론 없음',font=font(19),fill='#243d50')
            d.text((18,118),'D=보임  E=외부 가림 응답 | 진단용: 이후 학습·문턱·모델/arm 선택에 사용하지 않음',font=font(17),fill='#9c3b22')
            fp = O / f'{category}_{page:02d}.png'
            for k,r in enumerate(subset):
                xx,yy = (k%COLS)*W,147+(k//COLS)*H
                d.rectangle((xx+4,yy+3,xx+W-5,yy+H-5),fill='white',outline='#ccd8e1')
                d.text((xx+12,yy+9),f'행 {r["original_native_row"]:04d} · {r["material_display"]}',font=font(19),fill='#16384f')
                d.text((xx+12,yy+36),f'GT {r["human_label"]} → C16 {r["C16_prediction"]} · 슬롯 {r["native_slot"]}',font=font(17),fill='#253e50')
                n = r['original_native_row']
                raw = Image.fromarray(rgb[n].transpose(1,2,0))
                sheet.paste(raw.resize((192,192),Image.Resampling.NEAREST),(xx+16,yy+62))
                d.text((xx+12,yy+255),f'zE {r["C16_raw_zE"]:+.3f} · L {r["original_L"]:.1f}',font=font(13),fill='#39566b')
                r['contact'] = {'path':str(fp),'page':page,'tile_index':k,
                                'crop_display_xyxy':[xx+16,yy+62,xx+208,yy+254],
                                'enlargement':'NEAREST exact 2x; raw pixels unmarked'}
            d.text((18,height-34),'분모 전체 보존. E의 기존 reference는 가려진 지점의 추정값이며, 작은 좌표 오차도 위치 정답 인증이 아닙니다.',font=font(17),fill='#425f73')
            sheet.save(fp)
            # Verify every displayed unmarked crop is exactly the saved array enlarged 2x.
            with Image.open(fp) as saved:
                saved = np.asarray(saved)
                for r in subset:
                    x0,y0,x1,y1 = r['contact']['crop_display_xyxy']
                    expected = np.repeat(np.repeat(rgb[r['original_native_row']].transpose(1,2,0),2,0),2,1)
                    assert np.array_equal(saved[y0:y1,x0:x1],expected)
            artifacts.append({**binding(fp),'category':category,'page':page,'rows':len(subset),
                              'original_native_rows':[r['original_native_row'] for r in subset],
                              'width':COLS*W,'height':height,'saved_unmarked_crops_exact_2x':True})
    assert len(artifacts)==5 and sum(a['rows'] for a in artifacts)==100
    counts = {'paired_human_D_total':1147,'paired_human_external_E_total':90,
              'external_E_correct_E':35,'external_E_missed_D':55,'external_E_U':0,
              'visible_D_misclassified_E':10,
              'by_group':{g:{'human_D':sum(paired_D[(g,k)] for k in ('D','E','U')),
                             'human_E':sum(paired_E[(g,k)] for k in ('D','E','U')),
                             'D_predictions_D_E_U':[paired_D[(g,k)] for k in ('D','E','U')],
                             'E_predictions_D_E_U':[paired_E[(g,k)] for k in ('D','E','U')]}
                          for g in ('WOOD_PRIMARY105','SQUARE118')}}
    output = {'completed':True,'created_UTC':datetime.now(timezone.utc).isoformat(),
              'purpose':'Saved C16 current diagnostic only. Exhaustive paired external-E90 plus all paired D→E10, without quality/sample substitutions.',
              'counts':counts,'external90_original_order':external,'visible_D_false_E10_original_order':falseE,
              'artifacts':artifacts,'inputs':inputs,'selection_lock':binding(O/'ALL_ROWS_SELECTION_BEFORE_CROP_READ.json'),
              'renderer':binding(Path(__file__)),
              'runtime':{'Python':sys.version.split()[0],'NumPy':np.__version__,'Pillow':PIL.__version__,'font':FONT},
              'crop_policy':{'same_saved_V117_rgb96_and_crop_valid96':True,'crop_recomputed':False,
                             'q_L_ID_order_unchanged':True,'q_display_index':[47.5,47.5],
                             'no_marks_in_actual_crop_panels':True,'max_tiles_per_page':30,'columns':6,
                             'all100_displayed_saved_crops_verified_exact_2x':True},
              'limits':['Postfit current diagnostic; never use these cases for later training-data, threshold, model, or arm selection.',
                        'Human labels and estimated occluded references are unchanged. No physical/native association or location correctness is newly certified.',
                        'These crops are C16 RGB96 display inputs, not the complete spatial T/Q/F/C/P feature input. A crop alone cannot establish the model mechanism.',
                        'Repeated frames/slots are correlated; 90 cases do not mean 90 independent scenes.'],
              'new_model_encoder_initializer_training_coordinate_change_PnP':0,
              'V112_read':False,'full_RGB_decodes':0,'root_actual_view_pending_at_write':True}
    write(O/'MAPPING_AND_COUNTS.json',output)
    print(json.dumps({'mapping':binding(O/'MAPPING_AND_COUNTS.json'),'counts':counts,'artifacts':artifacts},ensure_ascii=False))

if __name__ == '__main__':
    main()
