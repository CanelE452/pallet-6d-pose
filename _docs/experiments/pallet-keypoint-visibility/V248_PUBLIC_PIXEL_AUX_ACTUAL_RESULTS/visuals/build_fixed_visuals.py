"""Render only Root's fixed16 SOURCE AUX slots. No model/prediction/GT reader."""
import argparse, ast, csv, hashlib, html, io, json, os, textwrap
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

R = Path(__file__).absolute().parent.parent
H = R/'V248_FIXED_SOURCE_AUX_VISUALS_B'
SCOPE = {'path': str(R/'V248_ROOT_PUBLIC_PIXEL_AUX/FIXED_SOURCE_VISUAL_SCOPE_BEFORE_NEW_HEAD_OUTPUTS.json'),
         'sha256': '0e5f793891c5abd5e680efb594ae1005e016f8dfcd9e165c96213cfac6ff5086', 'size_bytes': 7505}
FEATURE_COMPLETE = {'path': str(R/'V248_ROOT_PUBLIC_PIXEL_AUX/aux_canonical_pixel_features_v1/COMPLETE.json'),
                    'sha256': '153f1bbe2619d970208c48b81cfb3485f083a5ecaff4179db2a45fe7f0b3e0a4', 'size_bytes': 7913}
FIT_COMPLETE = {'path': str(R/'V248_ROOT_PUBLIC_PIXEL_AUX/matched_source_public_pixel_aux_fit_v1/COMPLETE.json'),
                'sha256': '35c169e9263734b6b0faa5d73f71c18f2e5efbe0a195d9c64a15c815d69e870a', 'size_bytes': 13202}
FIT_FREEZE = {'path': str(R/'V248_ROOT_PUBLIC_PIXEL_AUX/matched_source_public_pixel_aux_fit_v1/BOTH_FINAL1000_AND_LOGS_FROZEN_BEFORE_ANY_DEV_OUTPUT.json'),
              'sha256': '7f8f26131a01f534404defae4e44bce2e769738ad1187f704e8431a9b21c182d', 'size_bytes': 3814}
FONT = ImageFont.load_default()


def guard(value):
    s = str(value)
    if any(t in s.casefold() for t in ('v112', 'cache', '.pyc', 'current', 'private', 'held', 'holdout', 'oldval', 'v177', 'v178', 'v182')):
        raise ValueError('Forbidden path before FS')
    p = Path(s)
    if not p.is_absolute() or '..' in p.parts:
        raise ValueError('Absolute traversal-free path required')
    for a in reversed((p, *p.parents)):
        if a.is_symlink():
            raise ValueError('Symlink forbidden')
    return p


def sha(raw): return hashlib.sha256(raw).hexdigest()


def read_bound(b):
    p = guard(b['path'])
    with os.fdopen(os.open(p, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as f:
        raw = f.read()
    if sha(raw) != b['sha256'] or len(raw) != b['size_bytes']:
        raise ValueError('Exact SHA/bytes mismatch')
    return raw


def binding(p):
    raw = guard(p).read_bytes()
    return {'path': str(p), 'sha256': sha(raw), 'size_bytes': len(raw)}


def write_json(p, obj):
    with guard(p).open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
    return binding(p)


def save_png(p, im):
    p = guard(p)
    if p.exists(): raise ValueError('No overwrite')
    im.save(p, format='PNG')
    return binding(p)


def label(im, xy, text, scale=2, color=(230, 235, 245)):
    box = FONT.getbbox(text)
    tile = Image.new('RGBA', (max(1, box[2]-box[0]+4), max(1, box[3]-box[1]+4)), (0, 0, 0, 0))
    ImageDraw.Draw(tile).text((2-box[0], 2-box[1]), text, font=FONT, fill=(*color, 255))
    if scale != 1: tile = tile.resize((tile.width*scale, tile.height*scale), Image.Resampling.NEAREST)
    im.paste(tile, xy, tile)


def run(scope_path):
    if str(guard(scope_path)) != SCOPE['path']: raise ValueError('Only exact Root scope')
    scope = json.loads(read_bound(SCOPE))
    if scope['approved_SOURCE_fixed_visualization'] is not True or scope['selection_fixed_before_any_new_head_outputs'] is not True or scope['output_directory'] != str(H):
        raise ValueError('Root fixed visual scope required')
    if guard(H/'COMPLETE.json').exists(): raise ValueError('No rerun/overwrite')
    complete = json.loads(read_bound(FEATURE_COMPLETE))
    if complete['completed'] is not True or complete['PASS'] is not True: raise ValueError('Feature producer not complete')
    raw_records = [json.loads(s) for s in read_bound(scope['record_manifest']).splitlines() if s]
    feat_records = [json.loads(s) for s in read_bound(complete['record_manifest']).splitlines() if s]
    raw_by = {r['record_id']: r for r in raw_records}; feat_by = {r['record_id']: r for r in feat_records}
    if len(raw_by) != len(raw_records) or set(raw_by) != set(feat_by): raise ValueError('Immutable row IDs changed')
    if len(scope['slots']) != 16 or len(scope['record_ids']) != 12: raise ValueError('Fixed16/12 scope')
    selected = [s['record_id'] for s in scope['slots'] if s['record_id'] is not None]
    if selected != scope['record_ids']: raise ValueError('Fixed record order changed')
    fit = json.loads(read_bound(FIT_COMPLETE)); freeze = json.loads(read_bound(FIT_FREEZE))
    if fit['completed'] is not True or fit['PASS'] is not True or fit['freeze'] != FIT_FREEZE or freeze['PASS'] is not True or freeze['new_original_DEV_or_AUX_DEV_logits_before_bothfinalfreeze'] != 0:
        raise ValueError('Separate Root-authorized both-final barrier required before scores')
    if set(fit['aux_outputs']) != {'CONTROL_NO_PUBLIC_AUX', 'TREAT_PUBLIC_PIXEL_AUX'}: raise ValueError('Exact2 SOURCE AUX score CSVs')
    score_rows = {}
    for arm, b in fit['aux_outputs'].items():
        scores = list(csv.DictReader(io.StringIO(read_bound(b).decode())))
        by = {r['row_id']: r for r in scores}
        if len(scores) != len(by) or set(by) != set(raw_by): raise ValueError('Saved score/record ID closure')
        score_rows[arm] = by
    source = read_bound(scope['crop_math']); crop_nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'crop']
    if len(crop_nodes) != 1: raise ValueError('One pinned V90 crop')
    ns = {'np': np}; exec(compile(ast.Module(body=crop_nodes, type_ignores=[]), scope['crop_math']['path'], 'exec'), ns)
    crop = ns['crop']
    protocol = write_json(H/'PROTOCOL_BEFORE_EXACT8_RGB_READS.json', {
        'completed': True, 'scope': SCOPE, 'feature_complete': FEATURE_COMPLETE,
        'raw_records': scope['record_manifest'], 'feature_records': complete['record_manifest'],
        'postfreeze_saved_score_authority': {'Root_authorized_after_both_final_freeze': True, 'fit_complete': FIT_COMPLETE, 'freeze': FIT_FREEZE, 'only_aux_CSV2': fit['aux_outputs']},
        'RGB_whitelist': scope['RGB_whitelist'], 'crop_math': scope['crop_math'],
        'fixed_slots': 16, 'record_slots': 12, 'empty_slots': 4,
        'operation': 'V90 crop(q_aux,L64,96), compare saved RGB/valid SHA; raw PNG plus display-only marker/enlargement/fullframe q box',
        'new_model_forward_or_native_labels': 0,
        'authority': 'Conditional SAM+mesh+depth pixel teacher under inherited static-camera/calibration/depth-grid assumptions; native and SELF UNKNOWN.'})
    rgb_white = {(b['path'], b['sha256']): b for b in scope['RGB_whitelist']}
    if len(rgb_white) != 8: raise ValueError('Exact8 source RGB required')
    # Only these eight encoded images are read. No other original images or arrays.
    decoded = {}; source_qa = []
    for key, b in rgb_white.items():
        raw = read_bound(b)
        with Image.open(io.BytesIO(raw)) as im: a = np.array(im.convert('RGB'), np.uint8, copy=True)
        if a.shape != (480, 640, 3): raise ValueError('Native480x640 RGB expected')
        decoded[key] = a
        source_qa.append({'binding': b, 'decoded_RGB_HWC_sha256': sha(a.tobytes()), 'shape': list(a.shape)})
    outputs = []; cases = []; panels = []
    for si, slot in enumerate(scope['slots']):
        stem = f'SLOT_{si:02d}'; rid = slot['record_id']; target = int(slot['target'])
        state = 'V0 conditional visible support' if target == 0 else 'E1 conditional external support'
        panel = Image.new('RGB', (1240, 590), (17, 24, 39))
        label(panel, (12, 8), f"{si:02d} {slot['role']} | {slot['pair_id']} | view {slot['view']} | {state}", 2)
        case = dict(slot, slot_index=si, predictions={}, native_label=None, semantic_SELF=None, human_gold=False)
        if rid is None:
            matches = [r for r in raw_records if r['source_role'] == slot['role'] and r['pair_id'] == slot['pair_id'] and r['certificate_view'] == slot['view'] and r['conditional_pixel_target_V0_E1'] == target]
            if matches: raise ValueError('Fixed empty slot is not empty')
            label(panel, (80, 200), 'EMPTY STRATUM - no replacement / no fabricated target', 3, (251, 191, 36))
            label(panel, (80, 255), 'No crop or prediction. Original rows remain unchanged.', 2)
            case.update(status='EMPTY_STRATUM_NO_REPLACEMENT', artifacts={})
        else:
            row, fr = raw_by[rid], feat_by[rid]
            if any(fr[k] != v for k, v in row.items() if k != 'Root_conditional_AUX_FIT_admission'):
                raise ValueError('Raw AUX row changed beyond separate Root admission')
            if any(row[k] != slot[sk] for k, sk in (('source_role', 'role'), ('pair_id', 'pair_id'), ('certificate_view', 'view'), ('conditional_pixel_target_V0_E1', 'target'))):
                raise ValueError('Fixed slot row identity mismatch')
            image = row['image_binding']; key = (image['path'], image['sha256'])
            if key not in rgb_white or row['aux_L'] != 64.0 or row['native_slot'] is not None or row['semantic_SELF'] is not None:
                raise ValueError('Source/geometry/authority changed')
            rgb = decoded[key]; q = np.asarray(row['aux_query_xy'], np.float64)
            pixels, valid = crop(rgb, q, 64.0, 96)
            if pixels.dtype != np.uint8 or valid.dtype != bool or pixels.shape != (3, 96, 96) or valid.shape != (1, 96, 96): raise ValueError('Crop output schema')
            if sha(pixels.tobytes()) != fr['crop_RGB96_CHW_sha256'] or sha(valid[0].tobytes()) != fr['valid96_bool_sha256']:
                raise ValueError('Saved RGB/valid pixel SHA mismatch')
            clean = Image.fromarray(pixels.transpose(1, 2, 0), 'RGB')
            raw_b = save_png(H/(stem+'_RAW96.png'), clean); outputs.append(raw_b)
            with Image.open(raw_b['path']) as im:
                roundtrip = np.array(im.convert('RGB'), np.uint8).transpose(2, 0, 1).copy()
            if sha(roundtrip.tobytes()) != fr['crop_RGB96_CHW_sha256']: raise ValueError('Raw PNG roundtrip SHA')
            enlarged = clean.resize((384, 384), Image.Resampling.NEAREST)
            draw = ImageDraw.Draw(enlarged); center = 191.5
            color = (34, 211, 238) if target == 0 else (244, 114, 182)
            draw.line((center-18, center, center+18, center), fill=color, width=2)
            draw.line((center, center-18, center, center+18), fill=color, width=2)
            marked_b = save_png(H/(stem+'_MARKED4X.png'), enlarged); outputs.append(marked_b)
            full = Image.fromarray(rgb, 'RGB'); d = ImageDraw.Draw(full); x, y = map(float, q)
            d.rectangle((x-32, y-32, x+32, y+32), outline=(250, 204, 21), width=2)
            d.line((x-10, y, x+10, y), fill=color, width=2); d.line((x, y-10, x, y+10), fill=color, width=2)
            whole_b = save_png(H/(stem+'_WHOLEFRAME_Q.png'), full); outputs.append(whole_b)
            panel.paste(full, (12, 65)); panel.paste(clean, (680, 65)); panel.paste(enlarged, (812, 65))
            label(panel, (12, 42), 'Full frame + AUX q / L64 box (display only)', 1)
            label(panel, (680, 42), 'RAW96 clean', 1); label(panel, (812, 42), '4x nearest display + marker at raw (47.5,47.5)', 1)
            label(panel, (680, 180), 'q_aux='+str(row['aux_query_xy']), 1)
            label(panel, (680, 205), 'L_aux=64', 1)
            label(panel, (680, 230), 'native=UNKNOWN', 1)
            label(panel, (680, 255), 'SELF=UNKNOWN', 1)
            for ai, (arm, sr) in enumerate(score_rows.items()):
                sc = sr[rid]
                if sc['pair_id'] != slot['pair_id'] or sc['view'] != slot['view'] or sc['role'] != slot['role'] or int(sc['target']) != target or sc['available'] != 'True':
                    raise ValueError('Selected saved AUX score identity changed')
                logit = float(sc['logit']); prediction = 'E' if logit >= 0 else 'V'
                if not np.isfinite(logit) or sc['prediction'] != prediction: raise ValueError('Fixed threshold >=0 score encoding')
                short = 'CONTROL' if ai == 0 else 'TREAT'
                label(panel, (680, 290+ai*30), short+': '+prediction, 1)
                label(panel, (680, 304+ai*30), f'E logit {logit:+.5f}', 1)
                case['predictions'][arm] = {'saved_logit': logit, 'prediction': prediction, 'truth_conditional_target': target, 'source_CSV': fit['aux_outputs'][arm], 'new_forward': False}
            label(panel, (12, 555), rid, 1)
            case.update(status='FIXED_CONDITIONAL_PIXEL_RECORD', query_xy=row['aux_query_xy'], L=64.0,
                        image_binding=image, certificate_binding=row['certificate_binding'],
                        crop_RGB96_CHW_sha256=sha(pixels.tobytes()), valid96_bool_sha256=sha(valid[0].tobytes()),
                        saved_RGB_and_valid_SHA_exact=True, feature_available=fr['available'],
                        artifacts={'raw96': raw_b, 'marked4x': marked_b, 'wholeframe_q': whole_b})
        label(panel, (12, 570), 'Conditional AUX only; native/SELF UNKNOWN. Saved postfreeze scores; NOT native success/generalization.', 1, (251, 191, 36))
        panel_b = save_png(H/(stem+'_PANEL.png'), panel); outputs.append(panel_b)
        case['panel'] = panel_b; cases.append(case); panels.append(panel)
    boards = []
    for pi in range(4):
        board = Image.new('RGB', (2480, 1180), (17, 24, 39))
        for j in range(4): board.paste(panels[4*pi+j], ((j%2)*1240, (j//2)*590))
        b = save_png(H/f'PAIR_{pi:02d}_BOARD.png', board); outputs.append(b); boards.append((board, b))
    overview = Image.new('RGB', (2480, 1180), (17, 24, 39))
    for i, (im, _) in enumerate(boards): overview.paste(im.resize((1240, 590), Image.Resampling.LANCZOS), ((i%2)*1240, (i//2)*590))
    ov = save_png(H/'OVERVIEW_FIXED16.png', overview); outputs.append(ov)
    case_b = write_json(H/'CASE_DATA.json', {'scope': SCOPE, 'feature_complete': FEATURE_COMPLETE, 'cases': cases, 'SOURCE_RGB_QA': source_qa})
    with guard(H/'CASES.csv').open('x') as f:
        w = csv.DictWriter(f, fieldnames=['slot_index','role','pair_id','view','target','record_id','status','query_xy','L','panel'])
        w.writeheader()
        for c in cases:
            w.writerow({k:(Path(c['panel']['path']).name if k=='panel' else c.get(k)) for k in w.fieldnames})
    csv_b = binding(H/'CASES.csv')
    intro = ('Root가 새 head output 전에 고정한 16개 slot입니다. 12개 record와 빈4개 stratum을 대체 없이 표시합니다. '
             'V/E는 조건부 SAM+mesh+depth pixel teacher이며 physical/native 0–7 gold가 아닙니다. SELF와 native 대응은 UNKNOWN입니다. '
             '보정·static-camera·depth/RGB grid 가정을 상속하며, 그림에서 새 human label을 만들거나 점수·모델 선택을 하지 않았습니다. '
             '이 q는 teacher-selected 새 AUX 좌표이며 원352 initializer q/L/native를 바꾸지 않습니다. '
             '두 final1000 동결 뒤 Root가 허가한 저장 AUX CSV2의 예측만 붙였습니다. 신규 forward는0이며 conditional 학습사례이지 native 성능·독립 일반화 증거가 아닙니다.')
    body = ['<!doctype html><meta charset="utf-8"><title>V248 fixed SOURCE conditional pixel AUX</title>',
            '<style>body{background:#111827;color:#e5e7eb;font:16px system-ui;max-width:1400px;margin:auto;padding:24px}img{max-width:100%;height:auto}a{color:#67e8f9}section{border-top:1px solid #475569;margin:24px 0;padding:16px 0}</style>',
            '<h1>V248 고정 SOURCE pixel AUX 시각화</h1><p>'+html.escape(intro)+'</p><p><a href="CASE_DATA.json">정확 binding·좌표·SHA</a> · <a href="CASES.csv">고정16 slot 표</a></p><img src="OVERVIEW_FIXED16.png" alt="고정16 slot overview">']
    for c in cases:
        body.append('<section><h2>'+html.escape(f"{c['slot_index']:02d} {c['role']} {c['pair_id']} {c['view']} {'V0' if c['target']==0 else 'E1'}")+'</h2><img src="'+Path(c['panel']['path']).name+'" alt="fixed conditional pixel record"><p>'+html.escape(c['status'])+'</p>')
        if c['artifacts']:
            body.append('<p>'+' · '.join('<a href="'+Path(b['path']).name+'">'+html.escape(k)+'</a>' for k,b in c['artifacts'].items())+'</p>')
        body.append('</section>')
    with guard(H/'GALLERY.html').open('x') as f: f.write('\n'.join(body)+'\n')
    html_b = binding(H/'GALLERY.html')
    write_json(H/'COMPLETE.json', {'completed': True, 'PASS': True, 'scope': SCOPE, 'code': binding(__file__), 'protocol': protocol,
        'feature_complete': FEATURE_COMPLETE, 'record_manifest': complete['record_manifest'], 'source_images_exact': 8,
        'fit_complete_metadata': FIT_COMPLETE, 'both_final_freeze_metadata': FIT_FREEZE, 'saved_AUX_score_CSV2': fit['aux_outputs'],
        'fixed_slots': 16, 'record_slots': 12, 'empty_strata': 4, 'RGB_valid_saved_SHA_matches': 12,
        'raw_PNG_roundtrip_CHW_SHA_matches': 12, 'model_weights_or_GT_arrays_read': 0,
        'saved_AUX_CSVs_read_after_both_final_freeze': 2, 'new_model_forward': 0,
        'native_or_SELF_labels_created': 0, 'original_native352_modified': False, 'training_updates': 0,
        'selection_before_newhead_outputs': True, 'case_data': case_b, 'cases_csv': csv_b, 'gallery': html_b,
        'overview': ov, 'pair_boards': [b for _, b in boards], 'derived_PNG_bindings': outputs,
        'tool_image_view': 'PENDING separate receipt; rendering alone does not claim actual tool viewing'})
    print(json.dumps({'completed': True, 'PASS': True, 'complete': binding(H/'COMPLETE.json'), 'overview': ov, 'pair_boards': [b for _,b in boards]}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--scope', required=True); run(p.parse_args().scope)
