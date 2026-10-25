"""V233 고정 SOURCE crop 조립. 모델/특징/학습/GT 재조회는 하지 않는다."""
import argparse
import ast
from collections import Counter
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat

import numpy as np

P = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
R = P / 'publication_root_preparation'
PROTOCOL = dict(path=str(R/'V233_ROOT_CROP_ADAPTATION/PROTOCOL_BEFORE_SOURCE_PACKET_RGB_WEIGHT_FIT.json'), sha256='6755fecc3fd7927163357eb0caa677f4635e135f7819c954ae52316f3f29b28b')
OUT = R / 'V233_ROOT_CROP_ADAPTATION/crop_samples_v1'
Y = R / 'V232_ROOT_SOURCE_ADAPTATION/ycb_manifest_v1'
CODE = {
    'point_contract': dict(path=str(P/'query_visibility_dense_point_weak_bridge_source_training_20261025_v212/fit_preparation_a/source_target_contract.py'), sha256='d8925908d00f4f120bd80bae7a691a1a3846c4e9096e2e111d71984843e0b738'),
    'condition_contract': dict(path=str(R/'V219_TRAINING_PREPARATION_C/source_training_contract.py'), sha256='fc0a161e8a4c1a413d528ce1e66eed1b2addf2d4593c632506280f10732451d7'),
    'crop_code': dict(path=str(P/'query_visibility_public_cue_inputs_20261023_v90/producer/spatial_input_math.py'), sha256='6a6c2543e28715e3d47e15c543c25f75ff7d3efa3b881e523bd3451817fc004b'),
}
YCB = {
    'ycb_complete': dict(path=str(Y/'COMPLETE.json'), sha256='846aef541791f6b100acaa22dcaa7f74c906bd5e75696fa3b263d9b3172d3df2'),
    'ycb_inputs': dict(path=str(Y/'INPUTS1000.jsonl'), sha256='7f4e4bb8597b71b55395ae9ca014de0744d7f4eb664f2aeaa9a26400a52cdd28'),
    'ycb_targets': dict(path=str(Y/'TARGETS8000_TEACHER_ONLY.jsonl'), sha256='aa479c58c53eda2558f580f675f182a9968eae41dd19054d3b6f8cb49381af45'),
    'ycb_whitelist': dict(path=str(Y/'RGB700_WHITELIST.json'), sha256='790a33ab75214188fbc7a315686b7296755a02e863d5495d2cccf7b7dbb16e6f'),
}
META = ('row_id', 'original_global_index', 'native_slot', 'query_xy', 'L', 'original_query_available')
INPUT8 = set(META) | {'rgb96', 'valid96'}


def need(ok, message):
    if not bool(ok):
        raise ValueError(message)


def safe(path, exists=True):
    # 문자열 거부를 반드시 filesystem 접근보다 먼저 수행한다.
    q = Path(path)
    need(q.is_absolute() and '..' not in q.parts and 'v112' not in str(q).casefold()
         and '__pycache__' not in q.parts, 'Forbidden/nonabsolute path')
    for z in list(q.parents)[::-1] + [q]:
        try:
            mode = os.lstat(z).st_mode
        except FileNotFoundError:
            need(not exists, 'Missing exact path')
            continue
        need(not stat.S_ISLNK(mode), 'Symlink rejected')
    return q


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_bound(binding):
    q = safe(binding['path'])
    with os.fdopen(os.open(q, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as f:
        raw = f.read()
    need(digest(raw) == binding['sha256'], 'Exact SHA mismatch: ' + str(q))
    if 'size_bytes' in binding:
        need(len(raw) == binding['size_bytes'], 'Exact byte size mismatch')
    return raw


def same_binding(a, b):
    return a['path'] == b['path'] and a['sha256'] == b['sha256']


def records(raw):
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def pure(binding, names):
    source = read_bound(binding).decode('utf-8')
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name in names]
    need({n.name for n in nodes} == set(names), 'Pinned pure helper definitions')
    ns = dict(np=np, math=math)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), binding['path']+'::pure', 'exec'), ns)
    return ns


def kernels():
    pt = pure(CODE['point_contract'], ('need', 'finite_pair', 'point_loss_target'))
    cc = pure(CODE['condition_contract'], ('need', 'context32'))
    cr = pure(CODE['crop_code'], ('crop', 'query_validity'))
    return pt['point_loss_target'], cc['context32'], cr['crop'], cr['query_validity']


def finite_q(q):
    return isinstance(q, list) and len(q) == 2 and all(type(v) in (int, float) and math.isfinite(v) for v in q)


def valid_geometry(q, L, hw):
    if not finite_q(q) or not isinstance(L, (int, float)) or not math.isfinite(L) or L <= 0:
        return np.zeros((96, 96), bool)
    a = (np.arange(96, dtype=np.float64)+.5)/96-.5
    xx, yy = np.meshgrid(q[0]+a*L, q[1]+a*L)
    h, w = hw
    return (xx >= -.5) & (xx < w-.5) & (yy >= -.5) & (yy < h-.5)


def availability(original, valid):
    central = bool(valid[47:49, 47:49].all())
    fractions = [float(valid.mean()), float(valid[32:64, 32:64].mean())]
    return bool(original and central and valid.any()), central, fractions


def syn_fit(component):
    need(isinstance(component, str) and len(component) == 64, 'SYN component identity')
    return hashlib.sha256(('V233:'+component).encode()).digest()[0] % 5 != 0


def write_json(path, value):
    safe(path, False)
    with path.open('x', encoding='utf-8') as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)+'\n')


def binding(path):
    safe(path)
    h = hashlib.sha256()
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return dict(path=str(path), sha256=h.hexdigest(), size_bytes=path.stat().st_size)


def run(entry_path):
    with os.fdopen(os.open(safe(entry_path), os.O_RDONLY | os.O_NOFOLLOW), 'rb') as f:
        entry_raw = f.read()
    e = json.loads(entry_raw)
    need(e.get('approved_source_payload_read_and_crop_assembly') is True, 'Actual root entry required')
    need(same_binding(e['protocol'], PROTOCOL) and same_binding(e['code'], binding(Path(__file__).absolute())), 'Code/protocol entry pins')
    protocol = json.loads(read_bound(PROTOCOL))
    expected = dict(syn_inputs=protocol['exact_SYN_read'][0], syn_metadata=protocol['exact_SYN_read'][1], point_teacher=protocol['exact_SYN_read'][2],
                    syn_authority=protocol['exact_SYN_read'][4], syn_complete=protocol['exact_SYN_read'][5], **YCB, **CODE)
    need(set(e['inputs']) == set(expected), 'Closed input binding keys')
    for key in expected:
        need(same_binding(e['inputs'][key], expected[key]), 'Pinned input '+key)
    need(e['output_dir'] == str(OUT), 'Fresh fixed output path')
    need(e.get('no_model_GT_mixed_TARGET5_access') is True, 'No additional input authority')
    # Root RGB subset은 실제 이미지 읽기 전에 경로와 identity를 검사한다.
    allowed = {}
    for b in e['ycb_rgb_read_allowlist']:
        q = str(safe(b['path']))
        need(q not in allowed, 'Unique RGB allowlist')
        allowed[q] = b
    point_target, context32, crop, query_validity = kernels()
    closed = {k: json.loads(read_bound(e['inputs'][k])) for k in ('syn_authority', 'syn_complete', 'ycb_complete')}
    need(all(x.get('completed') is True and x.get('PASS') is True for x in (closed['syn_complete'], closed['ycb_complete'])), 'Completed source closures')
    for item in closed['ycb_complete']['outputs']:
        key = {'INPUTS1000.jsonl':'ycb_inputs', 'TARGETS8000_TEACHER_ONLY.jsonl':'ycb_targets', 'RGB700_WHITELIST.json':'ycb_whitelist'}[item['relative_path']]
        need(item['sha256'] == e['inputs'][key]['sha256'], 'YCB complete output pin')
    sm = records(read_bound(e['inputs']['syn_metadata']))
    pt = records(read_bound(e['inputs']['point_teacher']))
    yi = records(read_bound(e['inputs']['ycb_inputs']))
    yt = records(read_bound(e['inputs']['ycb_targets']))
    white = json.loads(read_bound(e['inputs']['ycb_whitelist']))['bindings']
    need(len(sm) == len(pt) == 7040 and len(yi) == 1000 and len(yt) == 8000 and len(white) == 700, 'Fixed source census')
    wh = {b['path']:b for b in white}
    need(len(wh) == 700 and all(p in wh and same_binding(b, wh[p]) for p,b in allowed.items()), 'Root RGB subset belongs to frozen700')
    wanted = {yi[t['instance_row_index']]['image']['path'] for t in yt if t['external_target'] in (0,1)}
    need(set(allowed) == wanted, 'Exact known D/E RGB subset; no unused image access')
    teachers = {r['row_id']:r for r in pt}
    need(len(teachers) == 7040 and set(teachers) == {m['row_id'] for m in sm}, 'All7040 exact teacher identities')
    targets = []
    for m in sm:
        r = teachers[m['row_id']]
        need(all(m[k] == r[k] for k in META), 'Teacher META6 identity')
        need(m['component_role'] == 'TRAIN' and m['nominal_fit_eligible'] is True and m['lineage_bound'] is True, 'SYN original TRAIN nominal lineage')
        need(r['source_teacher_key'] == [m['source_image_original_global_index']//8, m['native_slot']], 'Actual own-image teacher key')
        need(all(r[k] == m[k] for k in ('source_image_original_global_index','anchor_original_global_index')), 'Image vs q-anchor identity')
        targets.append(point_target(r)[0])
    need(Counter(targets) == {0:5853, 1:243, -1:944}, 'Existing SYN6096 D/E and944 exclusion')
    raw = read_bound(e['inputs']['syn_inputs'])
    with np.load(io.BytesIO(raw), allow_pickle=False) as z:
        need(set(z.files) == INPUT8, 'Only selected INPUT8; no SAM/TARGET5')
        x = {k: z[k] for k in z.files}
    need(x['rgb96'].dtype == np.uint8 and x['rgb96'].shape == (7040,96,96,3), 'SYN RGB HWC')
    need(x['valid96'].dtype == bool and x['valid96'].shape == (7040,96,96), 'SYN valid')
    need(all(x[k].shape == ((7040,2) if k == 'query_xy' else (7040,)) for k in META), 'INPUT6 shapes')
    need(x['query_xy'].dtype == x['L'].dtype == np.float64 and x['original_query_available'].dtype == bool and x['row_id'].dtype.kind == 'U', 'Original q/L/availability/ID dtypes')
    need(x['native_slot'].dtype.kind in 'iu' and x['original_global_index'].dtype.kind in 'iu' and int((~x['original_query_available']).sum()) == 64, 'Original SYN integer IDs / U64 retained')
    need(Counter(t['original_associated_r_state'] for t in yt) == {'D':1181,'E':340,'S':1917,'U':4562}, 'Original YCB teacher census')
    safe(OUT, False); OUT.mkdir(exist_ok=False)
    N = 7617
    rgb = np.lib.format.open_memmap(OUT/'raw_rgb96.npy', mode='w+', dtype=np.uint8, shape=(N,3,96,96))
    valid = np.lib.format.open_memmap(OUT/'raw_valid96.npy', mode='w+', dtype=bool, shape=(N,1,96,96))
    samples, diagnostics, full_rows = [], [], []

    def save_sample(row, im, v, cond):
        j = len(samples)
        need(im.shape == (3,96,96) and im.dtype == np.uint8 and v.shape == (96,96) and v.dtype == bool, 'Exact RGB/valid shapes')
        rgb[j] = im; valid[j,0] = v
        need(np.array_equal(rgb[j], im) and np.array_equal(valid[j,0], v), 'Saved RGB/valid row equality')
        row.update(sample_index=j, pixel_sha256=digest(im.tobytes()), valid_sha256=digest(v.tobytes()))
        samples.append(dict(full_view=2*j, central_view=2*j+1, fractions=row['valid_fractions'], central4=row['central4'], condition=cond,
                            target=row['target'], family=row['family_code'], fit=row['fit'], available=row['available'], row_id=row['row_id']))
        diagnostics.append(dict(row))

    for i, m in enumerate(sm):
        for k in META:
            want = np.asarray([np.nan,np.nan] if m[k] is None and k == 'query_xy' else np.nan if m[k] is None and k == 'L' else m[k])
            need(np.array_equal(np.asarray(x[k][i]), want, equal_nan=True) if k in ('query_xy','L') else x[k][i] == m[k], 'INPUT8 original META6 '+k)
        r = teachers[m['row_id']]; t = targets[i]
        need(m['context_origin'] == 'ORIGINAL_INITIALIZER' and m['context_q8'][m['native_slot']] == m['query_xy'], 'Original active query in anchor q8')
        cond = context32(m['query_xy'], m['L'], m['native_slot'], m['original_query_available'], m['context_q8'], m['context_L8'], m['context_available8'])
        need(digest(cond.tobytes()) == m['context_FP32_native_mask_xy_sha256'], 'Original condition32 exact')
        av, c4, frac = availability(m['original_query_available'], x['valid96'][i])
        ref = r.get('saved_projected_reference_xy'); dist = math.dist(m['query_xy'], ref) if finite_q(m['query_xy']) and finite_q(ref) else None
        fit = syn_fit(m['component_id'])
        row = dict(row_id=m['row_id'], family='SYN_NATIVE_WITNESS0', family_code=0, source_index=i, original_global_index=m['original_global_index'],
                   component_id=m['component_id'], original_component_role='TRAIN', internal_source_role='INTERNAL_SOURCE_FIT' if fit else 'INTERNAL_SOURCE_DEV', fit=fit,
                   native_slot=m['native_slot'], query_xy=m['query_xy'], L=m['L'], original_query_available=m['original_query_available'], target=t,
                   original_state=r['point0_combined_state'], target_alone_state=r['point0_target_alone_state'], region_state_context_only=r.get('region_state_context_only'),
                   reference_xy_GT_diagnostic_only=ref, q_reference_distance_px_diagnostic_only=dist, reference_locality_diagnostic_only=r.get('reference_locality_for_actual_input'),
                   source_teacher_key=r['source_teacher_key'], target_exclusion_reasons=point_target(r)[1], available=av, central4=c4, valid_fractions=frac,
                   validity_basis='EXACT_STORED_INPUT_VALID96', sample_index=None)
        if t in (0,1): save_sample(row, x['rgb96'][i].transpose(2,0,1), x['valid96'][i], cond)
        full_rows.append(row)
    last_path, last_rgb = None, None
    actually_read = {}
    for i, t in enumerate(yt):
        n, k = divmod(i,8); inp = yi[n]
        need(inp['input_id'] == f'Y109_SOURCE_{n:06d}' and inp['instance_row_index'] == n and inp['split'] == 'TRAIN' and inp['native_mapping'] == 'UNKNOWN', 'Original YCB input1000')
        need(t['slot_index'] == i and t['instance_row_index'] == n and t['logical_aux_slot'] == k and t['input_id'] == inp['input_id'] and t['native_mapping'] == 'UNKNOWN' and t['original_split'] == 'TRAIN', 'Exact YCB8000 input/target join')
        scene = int(t['scene_id']); fit = scene in range(5)
        need(scene in (0,1,2,3,4,6,7), 'Frozen YCB scenes')
        role = 'INTERNAL_SOURCE_FIT' if fit else 'INTERNAL_SOURCE_DEV'
        need(t['internal_source_role'] == inp['internal_source_role'] == role, 'YCB split before source read')
        target = {'D':0,'E':1,'S':-1,'U':-1}[t['original_associated_r_state']]
        need(target == t['external_target'], 'Original associated-r target; S/U masked')
        q, L, hw = inp['initial_points'][k], inp['roi_side'], inp['raw_hw']
        original = bool(query_validity([q], hw)[0][0]) if finite_q(q) else False
        v = valid_geometry(q, L, hw); av, c4, frac = availability(original, v)
        row = dict(row_id=t['input_id']+':aux'+str(k), family='YCB_ASSOCIATED_R', family_code=1, source_index=i, input_id=t['input_id'],
                   original_component_role='TRAIN', internal_source_role=role, fit=fit, native_slot=None, native_mapping='UNKNOWN', logical_aux_slot=k,
                   scene_id=t['scene_id'], source_frame_id=t['source_frame_id'], source_instance_id=t['source_instance_id'], object_id=t['object_id'], frame_index=t['frame_index'],
                   query_xy=q, L=L, original_query_available=original, original_state=t['original_associated_r_state'], target=target,
                   reference_xy_GT_diagnostic_only=None, q_reference_distance_px_diagnostic_only=None, reference_absence_reason='TARGET8000 has no r coordinates; no GT follow',
                   available=av, central4=c4, valid_fractions=frac, validity_basis='V90_RAW_HW_GEOMETRY; RGB decoded only for known D/E samples', sample_index=None)
        if target in (0,1):
            b = inp['image']; need(b['path'] in allowed and same_binding(b, allowed[b['path']]), 'Exact image authorized before read')
            if b['path'] != last_path:
                from PIL import Image
                image_raw = read_bound(allowed[b['path']])
                with Image.open(io.BytesIO(image_raw)) as im: last_rgb = np.asarray(im.convert('RGB')).copy()
                last_path = b['path']; actually_read[last_path] = allowed[last_path]
            need(list(last_rgb.shape[:2]) == list(hw) and last_rgb.dtype == np.uint8, 'Original RGB raw_hw')
            im, vv = crop(last_rgb, np.asarray(q,np.float64), L, 96) if finite_q(q) and isinstance(L,(int,float)) and math.isfinite(L) and L > 0 else (np.zeros((3,96,96),np.uint8),np.zeros((1,96,96),bool))
            need(np.array_equal(vv[0], v), 'V90 crop validity equals analytical original geometry')
            row['source_RGB_binding'] = b
            save_sample(row, im, v, np.zeros(32,np.float32))
        full_rows.append(row)
    need(len(samples) == N and len(full_rows) == 15040 and len({r['row_id'] for r in full_rows}) == 15040 and set(actually_read) == set(allowed), 'All rows retained / exact RGB subset')
    rgb.flush(); valid.flush()
    fields = dict(full_view=np.int64,central_view=np.int64,fractions=np.float32,central4=bool,condition=np.float32,target=np.int8,family=np.int8,fit=bool,available=bool,row_id=str)
    packet = {k: np.asarray([s[k] for s in samples], dtype=typ) for k,typ in fields.items()}
    need(packet['condition'].shape == (N,32) and packet['fractions'].shape == (N,2) and np.array_equal(packet['full_view'],np.arange(N)*2), 'Sample schema and order')
    np.savez(OUT/'samples.npz', **packet)
    for name, rows in [('ALL15040_ROWS.jsonl',full_rows), ('SAMPLE_DIAGNOSTICS.jsonl',diagnostics)]:
        with (OUT/name).open('x',encoding='utf-8') as f:
            for row in rows: f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    summary = dict(completed=True,PASS=True,original_rows15040=len(full_rows),known_samples7617=N,unknown_target944plus6479=7423,
                   sample_counts={str(k):v for k,v in sorted(Counter((s['family'],s['fit'],s['target'],s['available']) for s in samples).items())},
                   full_counts={str(k):v for k,v in sorted(Counter((r['family'],r['internal_source_role'],r['original_state'],r['target']) for r in full_rows).items())},
                   original_query_unchanged=True,GT_reference_only_diagnostic=True,known_unavailable_rows_deleted=0,normalization_DINO_model_forward_fit=0,
                   cross_family_donor_independence='UNKNOWN; not established by component or scene split',actual_RGB_reads=list(actually_read.values()))
    write_json(OUT/'SUMMARY.json',summary)
    complete = dict(completed=True,PASS=True,entry=dict(path=str(entry_path),sha256=digest(entry_raw)),protocol=PROTOCOL,code=binding(Path(__file__).absolute()),
                    inputs=e['inputs'],samples=N,all_rows=15040,outputs=[binding(OUT/n) for n in ('samples.npz','raw_rgb96.npy','raw_valid96.npy','ALL15040_ROWS.jsonl','SAMPLE_DIAGNOSTICS.jsonl','SUMMARY.json')],
                    model_weight_feature_array_raw_GT_or_mixed_input_bodies_read=0,training_performed=False)
    write_json(OUT/'COMPLETE.json',complete)
    print(json.dumps(dict(completed=True,PASS=True,samples=N,all_rows=15040,complete=binding(OUT/'COMPLETE.json'))))


def self_test():
    _, context32, crop, query_validity = kernels()
    im = np.arange(16*16*3,dtype=np.uint16).reshape(16,16,3).astype(np.uint8)
    q, L = [7.5,7.5], 16.
    a,v = crop(im,np.asarray(q),L,96)
    need(np.array_equal(v[0],valid_geometry(q,L,[16,16])), 'Toy crop validity')
    need(a.shape == (3,96,96) and a.dtype == np.uint8 and query_validity([q],[16,16])[0][0], 'Toy original query')
    need(availability(True,np.ones((96,96),bool)) == (True,True,[1.,1.]), 'Toy fractions')
    off=np.ones((96,96),bool);off[47,47]=False
    need(not availability(True,off)[0], 'Known target retained even unavailable')
    c=context32([0.,0.],2.,0,True,[[0.,0.],[40.,-40.]]+[[0.,0.]]*6,[2.]*8,[True,True]+[False]*6)
    need(c.shape==(32,)and c.dtype==np.float32 and c[8]==0 and c[9]==1 and np.array_equal(c[18:20],[8.,-8.]), 'Toy C61 context')
    need(syn_fit('a'*64)==syn_fit('a'*64) and 6096+1521==7617 and 7040+8000==15040, 'Toy fixed role/count algebra')
    print(json.dumps(dict(completed=True,PASS=True,groups=5,actual_source_payloads_read=0)))


if __name__ == '__main__':
    cli=argparse.ArgumentParser()
    cli.add_argument('--entry',type=Path)
    cli.add_argument('--self-test',action='store_true')
    args=cli.parse_args()
    if args.self_test: self_test()
    else:
        need(args.entry is not None,'--entry required')
        run(args.entry)
