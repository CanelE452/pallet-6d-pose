"""V238 saved SOURCE DEV fixed12 gallery, explicit Root entry only.

No encoder/head/GT access. IDs are sealed before cache hashes or pixel decoding.
"""
import argparse
import csv
import hashlib
import html
import io
import json
import os
from pathlib import Path
import textwrap
import traceback
import numpy as np
from PIL import Image

H = Path(__file__).absolute().parent
R = H.parent
CONTROL = 'CONTROL_LOCAL_CROP'
TREAT = 'TREAT_CONTEXT_RESIDUAL'
POLICY_PIN = (str(R/'V235_ROOT_PUBLIC_COUNTERFACTUAL/FIXED12_SOURCE_CASE_POLICY_BEFORE_NEW_PREDICTIONS.json'), 'df482d27f87c4e0e918eea0da5104152256a2e29ebea8c82237ce3fc28c26ea5')
INPUT_PIN = 'a9d8d8088f4cd61dac9384f2f25f49de1e48cef20787e3638e3f9c0aa5bb1718'
MANIFEST_PIN = '68827e79feb8115294f147675130091792b6c3be26b28640a07bdcd402cd7f36'


def safe(value):
    value = str(value)
    if any(x in value.casefold() for x in ('v112', '__pycache__', '.pyc', 'current223', 'publicheld', 'oldval', 'v177', 'v178', 'source_holdout', 'v182', 'private_evaluation')) or '..' in Path(value).parts:
        raise ValueError('Forbidden path before FS')
    path = Path(os.path.abspath(value))
    for part in reversed((path, *path.parents)):
        if part.is_symlink(): raise ValueError('Symlink rejected')
    return path


def read(path):
    with os.fdopen(os.open(safe(path), os.O_RDONLY | os.O_NOFOLLOW), 'rb') as stream:
        return stream.read()


def identity(b):
    return str(safe(b['path'])), b['sha256']


def checked(b, whitelist):
    if identity(b) not in {identity(x) for x in whitelist}: raise ValueError('Outside exact Root whitelist')
    digest = hashlib.sha256(); size = 0
    with os.fdopen(os.open(safe(b['path']), os.O_RDONLY | os.O_NOFOLLOW), 'rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block); size += len(block)
    if digest.hexdigest() != b['sha256'] or ('size_bytes' in b and b['size_bytes'] != size): raise ValueError('Changed bytes')
    return safe(b['path'])


def binding(path):
    b = read(path)
    return {'path': str(path), 'sha256': hashlib.sha256(b).hexdigest(), 'size_bytes': len(b)}


def plain(value):
    if isinstance(value, dict): return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)): return [plain(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)): return value.item()
    if isinstance(value, (float, np.floating)): return float(value) if np.isfinite(value) else None
    return value


def write_json(path, value):
    with os.fdopen(os.open(safe(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), 'w') as stream:
        json.dump(plain(value), stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')
    return binding(path)


def parse_predictions(raw):
    rows = list(csv.DictReader(io.StringIO(raw.decode())))
    for i, row in enumerate(rows):
        if int(row['sample_index']) != i or int(row['family']) not in (0, 1) or int(row['target']) not in (0, 1) or row['role'] not in ('FIT', 'DEV') or row['available'] not in ('True', 'False'):
            raise ValueError('Original source CSV schema/order')
        if row['available'] == 'True':
            z = np.array([float(row['logit0']), float(row['logit1'])], np.float32)
            selected = float(row['selected_family_logit'])
            if not np.isfinite(z).all() or not np.isfinite(selected) or np.float32(selected) != z[int(row['family'])] or row['prediction'] != ('E' if selected >= 0 else 'D'):
                raise ValueError('Saved finite zero-threshold decode')
        elif row['prediction'] != 'U' or row['selected_family_logit'] or row['logit0'] or row['logit1']:
            raise ValueError('Original input U remains null/U')
    return rows


def bucket_name(bucket):
    family = 'SYN_MAIN' if bucket['family'] == 0 else 'YCB_AUX'
    if 'target' in bucket:
        return f"{family} target{'D' if bucket['target'] == 0 else 'E'} TREAT_{'correct' if bucket['treatment_correct'] else 'wrong'}"
    return family+' '+bucket['paired_change']


def select_cases(policy, control, treatment):
    if len(control) != len(treatment): raise ValueError('Exact paired row count')
    for a, b in zip(control, treatment):
        if any(a[k] != b[k] for k in ('sample_index', 'row_id', 'family', 'role', 'available', 'target')): raise ValueError('Prediction metadata identity')
    cases = []
    for number, bucket in enumerate(policy['8_primary_buckets']+policy['4_change_buckets'], 1):
        candidates = []
        for a, b in zip(control, treatment):
            if b['role'] != 'DEV' or b['available'] != 'True' or int(b['family']) != bucket['family']: continue
            truth = 'D' if b['target'] == '0' else 'E'
            old, new = a['prediction'] == truth, b['prediction'] == truth
            if 'treatment_correct' in bucket:
                if int(b['target']) != bucket['target'] or new != bucket['treatment_correct']: continue
            elif bucket['paired_change'] == 'RECOVERED':
                if not (not old and new): continue
            elif bucket['paired_change'] == 'DAMAGED':
                if not (old and not new): continue
            else: raise ValueError('Unknown fixed bucket')
            candidates.append(int(b['sample_index']))
        index = min(candidates) if candidates else None
        cases.append({'case_number': number, 'bucket': bucket, 'bucket_name': bucket_name(bucket), 'eligible_population': len(candidates), 'sample_index': index, 'missing': index is None})
    return cases


def display_context(rgb_chw):
    """Display-only half-up quantization; never a model-input rewrite."""
    if rgb_chw.dtype != np.float32 or rgb_chw.shape != (3, 224, 224) or not np.isfinite(rgb_chw).all() or not ((rgb_chw >= 0) & (rgb_chw <= 255)).all():
        raise ValueError('Saved context F32 CHW0..255')
    return np.clip(np.floor(rgb_chw.transpose(1, 2, 0).astype(np.float64)+.5), 0, 255).astype(np.uint8)


def save_rgb_png(chw, path):
    if chw.dtype != np.uint8 or chw.shape != (3, 96, 96): raise ValueError('Saved node CHW96')
    hwc = chw.transpose(1, 2, 0).copy(); Image.fromarray(hwc, 'RGB').save(safe(path))
    with Image.open(safe(path)) as image: decoded = np.asarray(image.convert('RGB')).copy()
    restored = np.ascontiguousarray(decoded.transpose(2, 0, 1))
    if not np.array_equal(restored, chw): raise ValueError('Node PNG exact CHW roundtrip')
    return {'PNG': binding(path), 'cache_CHW_uint8_sha256': hashlib.sha256(np.ascontiguousarray(chw).tobytes()).hexdigest(), 'decoded_CHW_uint8_sha256': hashlib.sha256(restored.tobytes()).hexdigest(), 'exact': True}


def render_card(path, item, node_rgb, node_valid, context_display, group):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    active = int(item['active_slot']); q8 = np.asarray(group['q8'], np.float64); L8 = np.asarray(group['L8'], np.float64)
    avail = np.asarray(group['original_available8'], bool); box = np.asarray(group['box_xyxy'], np.float64)
    support = avail & node_valid[:, 47:49, 47:49].all((1, 2)) & node_valid.any((1, 2))
    fig = plt.figure(figsize=(16, 11), dpi=130)
    grid = fig.add_gridspec(3, 4, height_ratios=[1.2, 1, 1], hspace=.4, wspace=.25)
    def crop(ax, slot, title):
        ax.imshow(node_rgb[slot].transpose(1, 2, 0), interpolation='nearest'); ax.scatter([47.5], [47.5], marker='+', color='cyan', s=100)
        ax.set_xlim(-.5, 95.5); ax.set_ylim(95.5, -.5); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=10)
    crop(fig.add_subplot(grid[0, 0]), active, f'Original active RAW96 | slot {active}\nq cyan (47.5,47.5), unchanged')
    ax = fig.add_subplot(grid[0, 1]); ax.imshow(node_valid[active], vmin=0, vmax=1, cmap='gray', interpolation='nearest'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f'Original validity (not visibility)\nfraction {node_valid[active].mean():.4f}', fontsize=10)
    ax = fig.add_subplot(grid[0, 2:]); ax.imshow(context_display, interpolation='nearest'); ax.set_xlim(-.5, 223.5); ax.set_ylim(223.5, -.5); ax.set_xticks([]); ax.set_yticks([])
    for slot in range(8):
        q = q8[slot]
        if not np.isfinite(q).all() or np.array_equal(q, [-1, -1]): continue
        xy = (q-box[:2])/(box[2:]-box[:2])*224.-.5
        color = 'cyan' if slot == active else '#00ff44' if avail[slot] else '#aaaaaa'
        shown = np.clip(xy, 0, 223)
        ax.scatter([shown[0]], [shown[1]], marker='+' if slot == active else 'x', color=color, s=70)
        ax.annotate(('N' if item['family'] == 0 else 'A')+str(slot)+(' U' if not avail[slot] else ''), shown, xytext=(3, 5), textcoords='offset points', color=color, fontsize=8,
                    bbox={'facecolor': 'black', 'alpha': .35, 'pad': .3, 'edgecolor': 'none'})
    fraction = item['cache_pixel_proof']['context_valid_fraction']
    ax.set_title(f'Same actual-image / original q8-box context224 | validfraction={fraction:.4f}\nDisplay-only quantization; boundary markers clipped for display', fontsize=10)
    others = [slot for slot in range(8) if slot != active]
    for i, slot in enumerate(others):
        row, col = (1, i) if i < 4 else (2, i-4)
        name = 'native' if item['family'] == 0 else 'logical aux (native UNKNOWN)'
        crop(fig.add_subplot(grid[row, col]), slot, f'{name} {slot} | available={bool(avail[slot])}\nencoder support={bool(support[slot])}, L={L8[slot]:.2f}')
    ax = fig.add_subplot(grid[2, 3]); ax.axis('off')
    a, b = item['control'], item['treatment']
    lines = [f"sample {item['sample_index']} / slot {active}", *textwrap.wrap(item['row_id'], 37),
             f"Stored teacher: {item['target_state']}", f"CONTROL {a['prediction']} z={float(a['selected_family_logit']):+.5f}",
             f"CONTEXT {b['prediction']} z={float(b['selected_family_logit']):+.5f}", f"Paired change: {item['paired_change']}",
             f"original q=({q8[active,0]:.3f},{q8[active,1]:.3f})", f"L={L8[active]:.3f}",
             f"Bucket population={item['eligible_population']}", 'Min index; no substitute', 'No GT ref/pose/owner overlay', 'No attention/gold heatmap']
    ax.text(0, 1, '\n'.join(lines), va='top', family='monospace', fontsize=9, linespacing=1.35)
    title = f"V238 B{item['case_number']:02d} {item['bucket_name']}"
    if item.get('invented_toy_only'):
        title = 'INVENTED TOY — NOT SOURCE DATA | '+title
    fig.suptitle(title, fontsize=14, y=.995)
    fig.text(.5, .018, 'Reused SOURCE internal DEV, not blind/current | SYN witness0 vs YCB associated-r/native UNKNOWN | RGB/other material is not new native gold; no new forward', ha='center', fontsize=8.5)
    fig.subplots_adjust(top=.92, bottom=.07, left=.03, right=.98); fig.savefig(safe(path)); plt.close(fig)


def build(entry_path):
    e = json.loads(read(entry_path))
    if e.get('approved_source_context_gallery') is not True or e.get('source_only') is not True or e.get('automatic_retry') is not False:
        raise ValueError('Separate Root SOURCE gallery entry required')
    meta, payload = e['metadata_whitelist'], e['payload_whitelist']
    checked(e['code'], e['code_whitelist']); checked(e['implementation_protocol'], e['code_whitelist']); checked(e['root_scope'], meta)
    if identity(e['code']) != (str(safe(__file__)), binding(__file__)['sha256']): raise ValueError('Exact executed gallery code binding')
    if identity(e['policy']) != POLICY_PIN: raise ValueError('Original fixed12 policy only')
    policy = json.loads(read(checked(e['policy'], meta)))
    fit = json.loads(read(checked(e['fit_complete'], meta)))
    freeze = json.loads(read(checked(e['freeze'], meta))); barrier = json.loads(read(checked(e['global_QA_barrier'], meta)))
    inp = json.loads(read(checked(e['input_complete'], meta))); manifest = json.loads(read(checked(e['input_manifest'], meta)))
    if not all(x.get('completed') is True and x.get('PASS') is True for x in (fit, freeze, barrier, inp, manifest)): raise ValueError('Completed fit/data/freeze/global QA required')
    if identity(e['freeze']) != identity(fit['freeze']) or identity(e['global_QA_barrier']) != identity(fit['global_QA_barrier']) or fit['new_head_DEV_forward_before_global_QA'] != 0 or barrier['DEV_checked'] != 1741 or barrier['CONTROL_replayed'] != 7617:
        raise ValueError('Full V2 global QA barrier before all new DEV output')
    if fit['new_CONTROL_optimizer_updates'] != 0 or fit['new_TREAT_optimizer_updates'] != 1000 or fit['new_V235_CF'] != 0 or e['input_complete']['sha256'] != INPUT_PIN or e['input_manifest']['sha256'] != MANIFEST_PIN: raise ValueError('Original naturalSOURCE paired trial only')
    if identity(e['sample_map']) != identity(manifest['all_rows']) or identity(e['groups_metadata']) != identity(manifest['groups']): raise ValueError('Exact input-only group map')
    for arm, key in ((CONTROL, 'control_csv'), (TREAT, 'treatment_csv')):
        if identity(e[key]) != identity(fit['outputs'][arm]): raise ValueError('Exact frozen CSV closure')
    control = parse_predictions(read(checked(e['control_csv'], meta))); treatment = parse_predictions(read(checked(e['treatment_csv'], meta)))
    if len(control) != 7617: raise ValueError('Original7617 rows')
    cases = select_cases(policy, control, treatment); indices = sorted({x['sample_index'] for x in cases if not x['missing']})
    out = safe(e['output_directory'])
    if not out.is_relative_to(H) or out.exists(): raise ValueError('Fresh owned gallery output')
    out.mkdir()
    selection = write_json(out/'CASE_IDS_BEFORE_PIXELS.json', {'completed': True, 'PASS': True, 'policy': e['policy'], 'fit_complete': e['fit_complete'], 'CSV': [e['control_csv'], e['treatment_csv']], 'cases': cases, 'unique_sample_indices': indices, 'bucket_slots': 12, 'unique_samples': len(indices), 'missing_buckets': sum(x['missing'] for x in cases), 'minimum_sample_index_no_substitution': True, 'sealed_before_pixel_cache_hash_header_or_numerical_read': True})
    try:
        selected = set(indices); sample_map = {}
        for line in read(checked(e['sample_map'], meta)).splitlines():
            row = json.loads(line)
            if row['sample_index'] in selected:
                i = int(row['sample_index'])
                if i in sample_map or row['row_id'] != control[i]['row_id'] or bool(row['fit']) or not bool(row['crop_available']) or int(row['family_code']) != int(control[i]['family']) or int(row['target']) != int(control[i]['target']): raise ValueError('Selected original DEV identity')
                sample_map[i] = row
        if set(sample_map) != selected: raise ValueError('All selected cases mapped once')
        selected_groups = {int(x['group_index']) for x in sample_map.values()}; groups = {}
        for line in read(checked(e['groups_metadata'], meta)).splitlines():
            group = json.loads(line)
            if int(group['group_index']) in selected_groups: groups[int(group['group_index'])] = group
        if set(groups) != selected_groups: raise ValueError('Exact selected input-only groups')
        sealed_inputs = {Path(x['path']).name: x for x in inp['outputs']}
        arrays = {}
        for key, filename, dtype, shape in [('node_rgb','NODE_RGB96.npy',np.uint8,(1767,8,3,96,96)), ('node_valid','NODE_VALID96.npy',bool,(1767,8,96,96)), ('context_rgb','CONTEXT_RGB224_F32.npy',np.float32,(1767,3,224,224)), ('context_valid','CONTEXT_VALID224.npy',bool,(1767,224,224))]:
            if identity(e[key]) != identity(sealed_inputs[filename]): raise ValueError('Exact existing input cache binding')
            arrays[key] = np.load(checked(e[key], payload), mmap_mode='r', allow_pickle=False)
            if arrays[key].dtype != dtype or arrays[key].shape != shape: raise ValueError('Cache shape/dtype')
        saved = {}; decoded = {}; outputs = [selection]
        for gi in sorted(selected_groups):
            folder = out/f'GROUP_{gi:04d}'; folder.mkdir()
            node = np.array(arrays['node_rgb'][gi], copy=True); valid = np.array(arrays['node_valid'][gi], copy=True)
            ctx = np.array(arrays['context_rgb'][gi], copy=True); cv = np.array(arrays['context_valid'][gi], copy=True)
            display = display_context(ctx); proofs = []
            for slot in range(8):
                proof = save_rgb_png(node[slot], folder/f'NODE_{slot}_RAW96.png'); proofs.append(proof); outputs.append(proof['PNG'])
            cp = folder/'CONTEXT224_DISPLAY.png'; Image.fromarray(display, 'RGB').save(cp)
            with Image.open(cp) as image: restored = np.asarray(image.convert('RGB')).copy()
            if not np.array_equal(restored, display): raise ValueError('Context display PNG roundtrip')
            outputs.append(binding(cp))
            decoded[gi] = (node, valid, display)
            saved[gi] = {'node_PNG_proofs': proofs, 'context_display_PNG': binding(cp), 'source_context_F32_CHW_sha256': hashlib.sha256(ctx.tobytes()).hexdigest(), 'display_HWC_uint8_sha256': hashlib.sha256(display.tobytes()).hexdigest(), 'context_display_quantization_only': 'floor(F32+0.5), not an exact-F32 PNG or a model-input change', 'context_valid_fraction': float(cv.mean()), 'node_valid_fraction': valid.mean((1, 2)).tolist()}
        data = []
        for case in cases:
            item = dict(case); i = case['sample_index']
            if i is None: data.append(item); continue
            a, b = control[i], treatment[i]; row = sample_map[i]; gi = int(row['group_index']); slot = int(row['active_slot']); group = groups[gi]
            q = np.asarray(row['q'], np.float64); q8 = np.asarray(group['q8'], np.float64); L = float(row['L'])
            if not np.array_equal(q, q8[slot]) or L != float(group['L8'][slot]): raise ValueError('Original active q/L group slot exact')
            truth = 'D' if b['target'] == '0' else 'E'; old_ok = a['prediction'] == truth; new_ok = b['prediction'] == truth
            change = 'RECOVERED' if not old_ok and new_ok else 'DAMAGED' if old_ok and not new_ok else 'BOTH_CORRECT' if old_ok else 'BOTH_WRONG'
            item.update(row_id=row['row_id'], family=int(row['family_code']), target_state=truth, target_referent='SYN intended-native witness0' if int(row['family_code']) == 0 else 'YCB associated-r AUX; native UNKNOWN', active_slot=slot, group_index=gi, original_q=q.tolist(), original_L=L, q8_original=group['q8'], L8_original=group['L8'], original_available8=group['original_available8'], original_box=group['box_xyxy'], control={k:a[k] for k in ('prediction','logit0','logit1','selected_family_logit')}, treatment={k:b[k] for k in ('prediction','logit0','logit1','selected_family_logit')}, paired_change=change, cache_pixel_proof=saved[gi], role='INTERNAL_SOURCE_DEV_REUSED_NOT_BLIND')
            folder = out/f"CASE_{case['case_number']:02d}_SAMPLE_{i:04d}"; folder.mkdir(); card = folder/'CARD.png'
            render_card(card, item, *decoded[gi], group); item['card'] = binding(card); outputs.append(item['card']); data.append(item)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(4, 3, figsize=(13.5, 17), dpi=140)
        for ax, item in zip(axes.flat, data):
            if item['missing']:
                ax.axis('off'); ax.text(.5, .5, f"B{item['case_number']:02d} {item['bucket_name']}\nMISSING population0\nNo substitute", ha='center', va='center', fontsize=11); continue
            node = decoded[item['group_index']][0][item['active_slot']].transpose(1, 2, 0)
            ax.imshow(node, interpolation='nearest'); ax.scatter([47.5], [47.5], marker='+', color='cyan'); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"B{item['case_number']:02d} {item['bucket_name']} N={item['eligible_population']}\nsample {item['sample_index']} truth {item['target_state']} | CTRL {item['control']['prediction']} -> CTX {item['treatment']['prediction']}\n{item['paired_change']}", fontsize=9)
        fig.suptitle('V238 original SOURCE internal DEV | fixed12 policy / cyan original q\nFull cards show same-image original q8 context and other7 crops; not native gold', fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, .96)); contact = out/'FIXED12_SOURCE_DEV.png'; fig.savefig(contact); plt.close(fig); outputs.append(binding(contact))
        outputs.append(write_json(out/'CASE_DATA.json', {'completed': True, 'PASS': True, 'selection': selection, 'cases': data, 'unique_samples': indices, 'unique_groups': sorted(selected_groups), 'selected_cache_pixel_groups_only': True, 'new_model_forward': 0, 'attention_map_rendered': False, 'GT_reference_pose_ownership_used': False, 'SOURCE_DEV_not_blind': True, 'V1_history': 'V1 was an unrun preflight global-QA ordering repair. Separately, V2 actual attempt stopped at FIT descriptor QA with optimizer0. This gallery requires a later completed trial/freeze/fullQA, never relabels that failed attempt as completed.'}))
        columns = ['case_number','bucket_name','eligible_population','sample_index','missing','row_id','family','target_state','active_slot','group_index','original_L','paired_change']
        cp = out/'CASES.csv'
        with cp.open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader(); writer.writerows({k:item.get(k) for k in columns} for item in data)
        outputs.append(binding(cp))
        body = ['<!doctype html><html lang="ko"><meta charset="utf-8"><title>V238 SOURCE 고정12</title><style>body{font:17px sans-serif;max-width:1700px;margin:28px auto;line-height:1.6}img{max-width:100%}section{border-top:1px solid #aaa;margin-top:28px}</style><h1>V238 SOURCE internal DEV: 원 q8 문맥과 local crop</h1>', f'<p>12 bucket / 고유 {len(indices)}행 / 빈 {sum(x["missing"] for x in cases)} bucket. 사전 고정 최소 sample_index, 대체 없음. 재사용 SOURCE DEV이며 blind/current 평가가 아니다.</p>', '<p>원 q/L/native와 actual image를 유지했다. SYN witness0와 YCB associated-r/native UNKNOWN는 별도 교사다. 원q cyan·다른q green·원U gray; validity는 visibility gold가 아니다. GT reference/pose/ownership 또는 attention gold 표시를 쓰지 않는다. 새 head/encoder forward·학습은 없다.</p>', '<p>V1은 실행 전 전체 QA 장벽 순서 수리였다. 별도로 V2 실제 시도는 FIT descriptor 검사에서 optimizer0으로 종료됐다. 이 gallery는 그 실패를 완료로 쓰지 않으며, 후속 실제 완료·최종 freeze·전체 QA 뒤 저장 CSV만 사용한다.</p>', '<p><a href="CASE_IDS_BEFORE_PIXELS.json">픽셀 전 봉인 ID</a> · <a href="CASE_DATA.json">좌표·입력/PNG SHA</a> · <a href="CASES.csv">CSV</a></p><img src="FIXED12_SOURCE_DEV.png" alt="fixed12 cases">']
        for item in data:
            body.append('<section><h2>'+html.escape(f"B{item['case_number']:02d} {item['bucket_name']} N={item['eligible_population']}")+'</h2>')
            if item['missing']: body.append('<p>빈 bucket: 대체 사례 없음.</p></section>'); continue
            rel = str(Path(item['card']['path']).relative_to(out)); body.append('<p>'+html.escape(item['row_id'])+'</p><img src="'+rel+'" alt="active/context/other7 saved crops"></section>')
        body.append('</html>'); page = out/'PORTABLE_GALLERY_KO.html'; page.write_text('\n'.join(body)); outputs.append(binding(page))
        complete = {'completed': True, 'PASS': True, 'entry': binding(entry_path), 'code': e['code'], 'policy': e['policy'], 'fit_complete': e['fit_complete'], 'global_QA_barrier': e['global_QA_barrier'], 'input_complete': e['input_complete'], 'selection': selection, 'bucket_slots': 12, 'unique_samples': len(indices), 'unique_groups': len(selected_groups), 'missing_buckets': sum(x['missing'] for x in cases), 'numerically_decoded_group_indices_only': sorted(selected_groups), 'node_pixels_per_group': 8, 'only_selected_existing_V237_cache_pixels': True, 'node_PNG_CHW_roundtrip_exact': True, 'context_F32_to_U8_display_only': True, 'outputs': outputs, 'weights_features_prefix_rawRGB_GT_or_protected_reads': 0, 'new_model_forward_or_training': 0, 'new_visibility_label_or_cause_certification': False, 'automatic_retry': False}
        write_json(out/'COMPLETE.json', complete); print(json.dumps({'completed':True,'PASS':True,'COMPLETE':binding(out/'COMPLETE.json'),'contact':binding(contact),'HTML':binding(page)}), flush=True)
    except BaseException:
        write_json(out/'FAILED_EXECUTION.json', {'completed': False, 'PASS': False, 'automatic_retry': False, 'traceback': traceback.format_exc()}); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--entry', required=True); build(parser.parse_args().entry)
