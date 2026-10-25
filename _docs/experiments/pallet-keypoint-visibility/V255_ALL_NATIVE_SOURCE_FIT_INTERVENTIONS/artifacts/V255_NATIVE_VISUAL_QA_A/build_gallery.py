"""Fixed first-constructed SOURCE FIT scene gallery. No model or original GT reads."""
import argparse
import ast
from collections import Counter
import csv
import hashlib
import html
import io
import json
import os
from pathlib import Path
import stat
import sys
import traceback
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation')
CROP_CODE = {'path': str(ROOT.parent/'query_visibility_public_cue_inputs_20261023_v90/producer/spatial_input_math.py'),
             'sha256': '6a6c2543e28715e3d47e15c543c25f75ff7d3efa3b881e523bd3451817fc004b', 'size_bytes': 4715}
ROLES = ('BEFORE', 'COVER', 'NEAR_SHAM', 'FAR_SHAM')


def need(ok, message):
    if not bool(ok):
        raise ValueError(message)


def safe(path, exists=True):
    s = str(path)
    need(not any(t in s.casefold() for t in ('v112', 'cache', '.pyc', 'current223',
         'publicheld', 'oldval', 'holdout', 'private_evaluation')), 'Forbidden path before FS')
    p = Path(s)
    need(p.is_absolute() and '..' not in p.parts, 'Exact absolute path')
    for a in [*reversed(p.parents), p]:
        try:
            mode = os.lstat(a).st_mode
        except FileNotFoundError:
            need(not exists, 'Missing exact artifact')
            continue
        need(not stat.S_ISLNK(mode), 'No symlinks')
    return p


def read_bound(b):
    raw = safe(b['path']).read_bytes()
    need(hashlib.sha256(raw).hexdigest() == b['sha256'] and len(raw) == b['size_bytes'], 'Exact bound bytes')
    return raw


def bind(p):
    raw = safe(p).read_bytes()
    return {'path': str(p), 'sha256': hashlib.sha256(raw).hexdigest(), 'size_bytes': len(raw)}


def put_json(p, d):
    p = safe(p, False)
    need(not p.exists(), 'Fresh output file')
    with p.open('x', encoding='utf-8') as f:
        json.dump(d, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')
    return bind(p)


def pure_crop():
    raw = read_bound(CROP_CODE)
    tree = ast.parse(raw)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'crop']
    need(len(functions) == 1, 'Pinned V90 pure crop function only')
    env = {'np': np}
    exec(compile(ast.Module(body=functions, type_ignores=[]), CROP_CODE['path'], 'exec'), env)
    return env['crop']


def select_cases(configs, trials, scenes):
    """First constructed base per template/native. Proof success is not a selector."""
    templates = {int(c['base_index']): int(c['template_index']) for c in configs}
    need(len(templates) == len(configs), 'Unique selected base configs')
    scene_index = {}
    for s in scenes:
        key = (int(s['base_index']), s['target_native_slot'], s['variant'])
        need(key not in scene_index, 'Unique scene key')
        scene_index[key] = s
    trial_keys = set()
    groups, counts = [], {}
    for t in trials:
        key = (int(t['base_index']), int(t['native_slot']))
        need(key not in trial_keys and key[0] in templates and 0 <= key[1] < 8, 'All trial identity')
        trial_keys.add(key)
    need(len(trial_keys) == len(configs)*8, 'All selected native trials retained')
    for template in sorted(set(templates.values())):
        for native in range(8):
            pool = sorted([t for t in trials if templates[int(t['base_index'])] == template and
                           int(t['native_slot']) == native], key=lambda t: int(t['base_index']))
            c = Counter({'trials': len(pool)})
            for t in pool:
                c['alone_'+t['target_alone_point0_state']] += 1
                c['constructed' if t['constructed'] else 'unconstructed'] += 1
                if t['constructed']:
                    c['observed_D_E_D_D' if t['observed_D_E_D_D'] else 'failed_D_E_D_D'] += 1
            key = 'template%d_native%d' % (template, native)
            counts[key] = dict(c)
            candidates = [t for t in pool if t['constructed']]
            row = {'case_id': key, 'template_index': template, 'intervention_native': native,
                   'support': dict(c), 'empty': not candidates, 'selected_trial': None, 'scenes': []}
            if candidates:
                t = candidates[0]
                b = int(t['base_index'])
                keys = [(b, None, 'BEFORE')]+[(b, native, role) for role in ROLES[1:]]
                need(all(k in scene_index for k in keys), 'Constructed case has all four stored scenes')
                row.update(base_index=b, selected_trial=t, scenes=[scene_index[k] for k in keys])
            groups.append(row)
    return {'policy': 'FIRST_CONSTRUCTED_BASE_ASCENDING_PER_TEMPLATE_NATIVE_NO_PROOF_FILTER',
            'groups': groups, 'support': counts, 'case_count': sum(not g['empty'] for g in groups),
            'empty_count': sum(g['empty'] for g in groups), 'selection_before_pixels': True,
            'model_predictions_exist': False, 'training_admission': False}


def crops_for_scene(rgb, qmeta, crop):
    output, valid, supported = [], [], []
    for j in range(8):
        q, L = qmeta['q8'][j], qmeta['L8'][j]
        finite = q is not None and np.asarray(q).shape == (2,) and np.isfinite(q).all()
        finite = bool(finite and np.isfinite(L) and L > 0 and not np.array_equal(q, [-1, -1]))
        if finite:
            x, v = crop(rgb, np.asarray(q, np.float64), float(L), 96)
        else:
            x, v = np.zeros((3, 96, 96), np.uint8), np.zeros((1, 96, 96), bool)
        output.append(x)
        valid.append(v)
        supported.append(bool(qmeta['original_available8'][j] and v.any() and v[0, 47:49, 47:49].all()))
    return np.stack(output), np.stack(valid), supported


def draw_cross(d, x, y, color, size=5):
    d.line((x-size, y, x+size, y), fill=color, width=2)
    d.line((x, y-size, x, y+size), fill=color, width=2)


def make_card(case, scene_data):
    font = ImageFont.load_default(size=15)
    small = ImageFont.load_default(size=13)
    card = Image.new('RGB', (1190, 1880), 'white')
    d = ImageDraw.Draw(card)
    k = case['intervention_native']
    lines = [f"{case['case_id']}  base={case['base_index']}  intervention k={k} (query j remains separate)",
             'Fixed first constructed case; D/E/D/D proof failures are retained. No model predictions or new gold.',
             'Cyan: original q[j]; yellow box: original q[k]/L[k]; red circle: teacher reference r[k], display only.',
             'Crop centre q=(47.5,47.5). Point0/region states describe registered geometry, not literal q or whole part.',
             'All8 display crops use unchanged q/L. avail=0 and unsupported inputs retain U; display pixels do not relabel them.']
    for i, s in enumerate(lines):
        d.text((18, 12+21*i), s, fill='black', font=font)
    for ri, (meta, rgb, crops, valid, supported) in enumerate(scene_data):
        y = 130+ri*430
        qmeta = meta['original_before_input']
        q, L = qmeta['q8'][k], qmeta['L8'][k]
        r = meta['fresh_records_CF'][k]['projected_reference_xy']
        state = meta['point0_combined_states8'][k]
        d.text((18, y), f"{meta['variant']}  target k point0={state}  fresh region={meta['fresh_records_CF'][k]['state']}  original avail[k]={qmeta['original_available8'][k]}", fill='black', font=font)
        whole = Image.fromarray(rgb).resize((384, 384), Image.Resampling.NEAREST)
        wd = ImageDraw.Draw(whole)
        h, w = rgb.shape[:2]
        for j, qj in enumerate(qmeta['q8']):
            if qj is not None and np.isfinite(qj).all():
                xj, yj = (np.asarray(qj)+.5)*np.array([384/w, 384/h])-.5
                draw_cross(wd, xj, yj, '#00ffff')
                wd.text((xj+4, yj-13), str(j), fill='#00ffff', font=small)
        if q is not None and np.isfinite(q).all():
            a = (np.asarray(q)-L/2+.5)*np.array([384/w, 384/h])-.5
            b = (np.asarray(q)+L/2+.5)*np.array([384/w, 384/h])-.5
            wd.rectangle((*a, *b), outline='#ffe100', width=2)
        rx, ry = (np.asarray(r)+.5)*np.array([384/w, 384/h])-.5
        wd.ellipse((rx-5, ry-5, rx+5, ry+5), outline='#ff3030', width=2)
        card.paste(whole, (18, y+28))
        active = Image.fromarray(crops[k].transpose(1, 2, 0)).resize((192, 192), Image.Resampling.NEAREST)
        ad = ImageDraw.Draw(active)
        draw_cross(ad, 95.5, 95.5, '#00ffff', 8)
        if q is not None and np.isfinite(q).all() and np.isfinite(L) and L > 0:
            local_r = (np.asarray(r)-np.asarray(q))/L*96+47.5
            if ((local_r >= -.5) & (local_r < 95.5)).all():
                ax, ay = (local_r+.5)*2-.5
                ad.ellipse((ax-6, ay-6, ax+6, ay+6), outline='#ff3030', width=2)
            rtext = f"r local=({local_r[0]:.2f},{local_r[1]:.2f})"
        else:
            rtext = 'r local unavailable; original q absent'
        card.paste(active, (424, y+48))
        info = [f"Active original q-centred96 (j=k={k})", rtext,
                f"valid fraction={valid[k].mean():.4f}", f"structural supported={supported[k]}",
                'Cyan centre is q, red is r.', 'r never recentres any crop.',
                f"proof actual D/E/D/D={case['selected_trial']['observed_D_E_D_D']}"]
        for ii, text in enumerate(info):
            d.text((420, y+250+20*ii), text, fill='black', font=small)
        for j in range(8):
            tx, ty = 704+(j%4)*116, y+35+(j//4)*184
            tile = Image.fromarray(crops[j].transpose(1, 2, 0))
            td = ImageDraw.Draw(tile)
            draw_cross(td, 47.5, 47.5, '#00ffff', 4)
            card.paste(tile, (tx, ty))
            rec = meta['fresh_records_CF'][j]
            text = [f"j={j} avail={int(qmeta['original_available8'][j])}",
                    f"alone/p0/region", f"{meta['point0_target_alone_states8'][j]}/{meta['point0_combined_states8'][j]}/{rec['state']}",
                    f"support={int(supported[j])}"]
            for ii, s in enumerate(text):
                d.text((tx, ty+100+16*ii), s, fill='black', font=small)
    d.text((18, 1860), 'SOURCE FIT procedural geometry only. Intended part/instance at literal q remains unverified. No physical native gold or adoption.', fill='black', font=small)
    return card


def run(entry_path):
    entry_raw = safe(entry_path).read_bytes()
    e = json.loads(entry_raw)
    need(e.get('approved_actual_native_visualization') is True, 'Root actual visualization entry required')
    own = bind(Path(__file__).absolute())
    need(e['code'] == own, 'Entry code binding')
    proto = json.loads(read_bound(e['protocol']))
    need(proto['code'] == own and proto['crop_code'] == CROP_CODE, 'Frozen display protocol')
    scope = json.loads(read_bound(e['root_scope']))
    need(scope.get('approved_actual_native_visualization') is True and
         scope.get('entry_contract') == {k: v for k, v in e.items() if k != 'root_scope'}, 'Exact Root visual scope')
    need(e.get('new_model_encoder_or_fit') is False and e.get('current_or_held_access') is False,
         'No model/protected access')
    complete = json.loads(read_bound(e['dataset_complete']))
    need(complete.get('completed') is True and complete.get('PASS') is True, 'Actual dataset COMPLETE')
    need(e['selected_configs_manifest'] == complete['selected_configs_manifest'], 'Closed selected FIT config metadata')
    out = safe(e['output_directory'], False)
    need(not out.exists(), 'Fresh gallery output')
    out.mkdir()
    saved = []
    try:
        cfg = json.loads(read_bound(e['selected_configs_manifest']))
        scenes = [json.loads(x) for x in read_bound(complete['outputs']['SCENES.jsonl']).splitlines() if x]
        trials = [json.loads(x) for x in read_bound(complete['outputs']['ALL_NATIVE_TRIALS.jsonl']).splitlines() if x]
        selection = select_cases(cfg['rows'], trials, scenes)
        need(e.get('scene_read_policy') == 'ONLY_METADATA_SELECTED_FIRST_CONSTRUCTED_CASE_SCENES', 'Closed selection policy')
        # This receipt is committed before any scene NPZ is read or NumPy-loaded.
        selected_binding = put_json(out/'CASE_IDS_BEFORE_PIXELS.json', selection)
        saved.append(selected_binding)
        selected_rows = [row for case in selection['groups'] if not case['empty'] for row in case['scenes']]
        need(e['actual_pixel_read_allowlist'] == [row['image'] for row in selected_rows] and
             e['selected_scene_metadata_read_allowlist'] == [row['metadata'] for row in selected_rows],
             'Exact presealed selected scene pixel/metadata whitelist')
        crop = pure_crop()
        case_data, cards = [], []
        for case in selection['groups']:
            if case['empty']:
                case_data.append(case)
                continue
            dataset_rows, crop_rgbs, crop_valid = [], [], []
            data_for_card = []
            for row in case['scenes']:
                meta = json.loads(read_bound(row['metadata']))
                need(meta['image'] == row['image'] and meta['RGB_sha256_HWC'] == row['RGB_sha256_HWC'], 'Scene metadata/image join')
                need(meta['variant'] == row['variant'] and meta['base_index'] == case['base_index'] and
                     meta['target_native_slot'] == row['target_native_slot'], 'Selected scene identity')
                need(meta['original_before_input']['source_role'] == 'FIT', 'SOURCE FIT only')
                with np.load(io.BytesIO(read_bound(row['image'])), allow_pickle=False) as z:
                    need(z.files == ['RGB_uint8_HWC'], 'Only scene RGB pixel member')
                    rgb = z['RGB_uint8_HWC']
                need(rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3, 'Scene U8 HWC')
                sha = hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()
                need(sha == row['RGB_sha256_HWC'], 'Scene decoded HWC SHA exact')
                qmeta = meta['original_before_input']
                if dataset_rows:
                    need(qmeta == dataset_rows[0]['original_before_input'], 'Unchanged original q8/L8/U across four variants')
                crops, valid, support = crops_for_scene(rgb, qmeta, crop)
                k = case['intervention_native']
                clean = out/(case['case_id']+'_'+meta['variant']+'_ACTIVE96_CLEAN.png')
                Image.fromarray(crops[k].transpose(1, 2, 0)).save(safe(clean, False))
                clean_binding = bind(clean)
                saved.append(clean_binding)
                dataset_rows.append({'scene': row, 'original_before_input': qmeta,
                                     'fresh_records_CF': meta['fresh_records_CF'],
                                     'point0_target_alone_states8': meta['point0_target_alone_states8'],
                                     'point0_combined_states8': meta['point0_combined_states8'],
                                     'structural_support8': support, 'clean_active96': clean_binding,
                                     'clean_active96_CHW_sha256': hashlib.sha256(crops[k].tobytes()).hexdigest(),
                                     'valid96_sha256': hashlib.sha256(valid.tobytes()).hexdigest(),
                                     'native_RGB_changed_pixels': meta.get('native_RGB_changed_pixels', 0)})
                crop_rgbs.append(crops)
                crop_valid.append(valid)
                data_for_card.append((meta, rgb, crops, valid, support))
            small_npz = out/(case['case_id']+'_CLEAN_CROPS96.npz')
            with safe(small_npz, False).open('xb') as f:
                np.savez_compressed(f, RGB_uint8_CHW=np.stack(crop_rgbs), valid96=np.stack(crop_valid))
            nb = bind(small_npz)
            saved.append(nb)
            card_path = out/(case['case_id']+'_CARD.png')
            make_card(case, data_for_card).save(safe(card_path, False))
            cb = bind(card_path)
            saved.append(cb)
            cards.append(cb)
            case_data.append(dict(case, displayed_scenes=dataset_rows, clean_crops=nb, card=cb))
        case_binding = put_json(out/'CASE_DATA.json', {'cases': case_data, 'teacher_reference_display_only': True,
                                                      'q_centre_47_5_exact_V90_math': True})
        saved.append(case_binding)
        csv_path = out/'SUPPORT_AND_CASES.csv'
        fields = ['case_id', 'template_index', 'intervention_native', 'empty', 'base_index',
                  'trials', 'alone_D', 'alone_E', 'alone_S', 'alone_U', 'constructed',
                  'unconstructed', 'observed_D_E_D_D', 'failed_D_E_D_D', 'selected_proof_D_E_D_D']
        with safe(csv_path, False).open('x', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for c in selection['groups']:
                w.writerow({**{k: c.get(k) for k in fields[:5]},
                            **{k: c['support'].get(k, 0) for k in fields[5:-1]},
                            'selected_proof_D_E_D_D': None if c['empty'] else c['selected_trial']['observed_D_E_D_D']})
        saved.append(bind(csv_path))
        tiles = []
        for c in case_data:
            tile = Image.new('RGB', (595, 970), 'white')
            td = ImageDraw.Draw(tile)
            td.text((8, 8), c['case_id']+(' EMPTY' if c['empty'] else ' first constructed'), fill='black', font=ImageFont.load_default(size=15))
            if not c['empty']:
                with Image.open(safe(c['card']['path'])) as im:
                    tile.paste(im.resize((595, 940), Image.Resampling.LANCZOS), (0, 30))
            else:
                td.text((8, 45), 'No constructed case. No substitute selected.', fill='black', font=ImageFont.load_default(size=15))
            tiles.append(tile)
        overview = Image.new('RGB', (1190, ((len(tiles)+1)//2)*970), 'white')
        for i, tile in enumerate(tiles):
            overview.paste(tile, ((i%2)*595, (i//2)*970))
        op = out/'OVERVIEW.png'
        overview.save(safe(op, False))
        saved.append(bind(op))
        page = ['<!doctype html><meta charset="utf-8"><title>V255 SOURCE native scene gallery</title>',
                '<style>body{font:16px sans-serif;max-width:1250px;margin:auto}img{max-width:100%}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:5px}</style>',
                '<h1>V255 SOURCE FIT: 원 q/L 고정 native별 첫 constructed 사례</h1>',
                '<p>template×원 native별 base_index 최소 constructed 사례입니다. D/E/D/D 실패를 제외하지 않았고 빈칸을 대체하지 않았습니다. 모델 예측은 없습니다.</p>',
                '<p>청록 q와 노란 q/L 박스는 원 입력, 빨간 r은 등록 합성 teacher reference를 보여주는 표시입니다. r로 crop을 이동하지 않았습니다. point0·region 상태는 literal q, 전체 part 또는 실사 native gold가 아닙니다. 개입 k와 각 query j를 구분합니다.</p>',
                '<p>원 U/불가용 플래그와 target-alone S/U를 유지합니다. 96 crop은 고정 q/L 표시이며 0 availability에서 보이는 픽셀은 학습 허용이나 라벨 변경을 뜻하지 않습니다. 선택 SOURCE FIT 사례 관찰은 모집단 원인이나 모델 성공의 증거가 아닙니다.</p>',
                '<p><a href="CASE_IDS_BEFORE_PIXELS.json">픽셀 전 고정 ID</a> · <a href="SUPPORT_AND_CASES.csv">S/U/실패/빈칸 분모</a> · <a href="CASE_DATA.json">주소·teacher·hash</a></p>']
        for c in case_data:
            page.append('<h2>'+html.escape(c['case_id'])+'</h2>')
            page.append('<p>'+html.escape(json.dumps(c['support'], sort_keys=True))+'</p>')
            if c['empty']:
                page.append('<p>EMPTY: constructed 사례 없음. 대체 없음.</p>')
            else:
                page.append('<p>base '+str(c['base_index'])+' / actual D/E/D/D: '+str(c['selected_trial']['observed_D_E_D_D'])+'</p>')
                page.append('<img src="'+html.escape(Path(c['card']['path']).name)+'">')
        hp = out/'PORTABLE_GALLERY_KO.html'
        with safe(hp, False).open('x', encoding='utf-8') as f:
            f.write('\n'.join(page)+'\n')
        saved.append(bind(hp))
        mb = put_json(out/'MANIFEST.json', {'files': saved, 'treewalk': False})
        complete_out = {'completed': True, 'PASS': True, 'code': own, 'protocol': e['protocol'],
                        'entry': {'path': str(entry_path), 'sha256': hashlib.sha256(entry_raw).hexdigest(), 'size_bytes': len(entry_raw)},
                        'dataset_complete': e['dataset_complete'], 'root_scope': e['root_scope'],
                        'selection': selected_binding, 'case_data': case_binding, 'manifest': mb,
                        'overview': bind(op), 'gallery': bind(hp), 'cards': cards,
                        'counts': {'selected_cases': selection['case_count'], 'empty_buckets': selection['empty_count'],
                                   'selected_scene_NPZ_reads': 4*selection['case_count']},
                        'model_encoder_or_training': 0, 'original_RGB_GT_mesh_archives_read': 0,
                        'training_admission': False, 'physical_or_semantic_gold_certification': False}
        cb = put_json(out/'COMPLETE.json', complete_out)
        print(json.dumps({'completed': True, 'COMPLETE': cb, 'counts': complete_out['counts']}, sort_keys=True), flush=True)
        return complete_out
    except BaseException as exc:
        put_json(out/'FAILURE.json', {'completed': False, 'PASS': False, 'error': str(exc),
                                   'traceback': traceback.format_exc(), 'written_files': saved,
                                   'automatic_retry': False})
        raise


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    ap = argparse.ArgumentParser()
    ap.add_argument('--entry', required=True)
    run(ap.parse_args().entry)
