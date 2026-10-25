"""Two explicitly authorized stages: conditional AUX records, frozen pixel1154.

Import is inert. No original initializer rows, native targets, fitted head or
old feature/prefix arrays are input. Root pins each exact stage before access.
"""
import argparse, ast, hashlib, io, json, os, re, sys, time, traceback, types
from pathlib import Path
from collections import Counter
import numpy as np

sys.dont_write_bytecode = True
R = Path(__file__).absolute().parent.parent
P = R.parent
H = R / 'V248_PUBLIC_PIXEL_AUX_BUILDER_B'
ROOT = R / 'V248_ROOT_PUBLIC_PIXEL_AUX'
PINS = {
 'root_protocol': (str(ROOT/'PROTOCOL_BEFORE_NEW_AUX_ARRAYS_OR_FEATURES_OR_FIT.json'), 'f66d9e29f85141527aa2761311a29e512d611c6ba14b287d21dbb08c589e4fda'),
 'pair_usage': (str(ROOT/'AUX_PAIR_USAGE_FROZEN_BEFORE_ARRAYS.json'), '6e8324f107a4b14903a531b591daac0bf9d4d10e2093b1d219f559a41eb45497'),
 'crop_addendum': (str(ROOT/'CROP_CONTRACT_CLARIFICATION_BEFORE_PAYLOADS.json'), '1272df925a262a60c1246259f576d028353a5935fa20bb2d5320c9e94e6f04a7'),
 'selected_complete': (str(R/'V246_ROOT_PALLOX_TRAIN/train_pair_certificate_metadata_v2/COMPLETE.json'), '15533dcad2a45de544d17b56d737170fb64fe869792c343d39d6ec33540fb488'),
 'certificate_manifest': (str(R/'V246_ROOT_PALLOX_TRAIN/train_pair_certificate_metadata_v2/TRAIN22_CERTIFICATE_PAYLOAD_WHITELIST.json'), 'f74051cba2033f0dcf6727c022fd499624cd6c144bce3f8c1b66cb03a8b61f3f'),
 'selected_pairs': (str(R/'V246_ROOT_PALLOX_TRAIN/train_pair_certificate_metadata_v2/TRAIN22_PAIR_CERTIFICATE_METADATA.jsonl'), '89f18161612472031c98933483b4f7198f17b5e2e514ec071261eadf5a14508a'),
 'sampler': (str(R/'V248_PALLOX_AUX_SAMPLER_READINESS_B/sample_conditional_aux_pixels.py'), 'f78982a5c3cd15abf7144fcd63dc9ea9285b34cb929bc63dcfba204bf3cad3ff'),
 'paired_truth': (str(P/'query_visibility_controlled_real_20261022_v66/contract/paired_truth.py'), '649c93b98085289bb1aa797b3beba141423f5ffbeb055b5c7f2f571e208f0c14'),
 'canonical_producer': (str(R/'V239_CANONICAL_FEATURE_PRODUCER_B/build_canonical_source_features.py'), '29a890a308f232743f65bc525e5aaee346b8702ae14c5db8a60b15d5bd367ff4'),
 'canonical_feature_entry': (str(R/'V239_ROOT_CANONICAL_CONTEXT/ACTUAL_CANONICAL_FEATURE_ENTRY.json'), '755cc5a4f132394010e650f96a46b2bb81f5374aa8c9815eabf5cf950661870a'),
 'crop_math': (str(P/'query_visibility_public_cue_inputs_20261023_v90/producer/spatial_input_math.py'), '6a6c2543e28715e3d47e15c543c25f75ff7d3efa3b881e523bd3451817fc004b'),
 'crop_module': (str(R/'V233_DETERMINISTIC_CROP_FIX_A/crop_modules.py'), '44f54dfcbf1671a401e748e66ec30463096e6c24b0779f07dda85d7c9f38ae88'),
 'preprocess_CODE': (str(P/'query_visibility_frozen_rgb_head_20261023_v105/frozen_rgb_feature_preparation/prepare_frozen_rgb_features.py'), 'cc1e2c331d41aaea5b3d42d144c9abbc40231dd3784d24f52dcc58f41d607578'),
 'model_loader_code': (str(R/'V233_DETERMINISTIC_CROP_FIX_A/train_crop_adaptation.py'), 'ae6e8ceb66ef29a52633ab98511e23a659d7f5b7e8268d300a53b8f4f12d71b1'),
}
OUTPUT = {'sample': ROOT/'aux_sample_records_v1', 'features': ROOT/'aux_canonical_pixel_features_v1'}
CERT_KEYS = ('visible_positive', 'visible_negative', 'full_positive', 'full_negative')


def lexical(value):
    text = str(value)
    if any(t in text.casefold() for t in ('v112', 'cache', '.pyc', 'current', 'held', 'private', 'holdout', 'oldval', 'v177', 'v178', 'v182')):
        raise ValueError('Forbidden path before filesystem access')
    p = Path(text)
    if not p.is_absolute() or '..' in p.parts:
        raise ValueError('Absolute traversal-free path required')
    return p


def safe(value):
    p = lexical(value)
    for a in reversed((p, *p.parents)):
        if a.is_symlink():
            raise ValueError('Bound artifact symlink forbidden')
    return p


def identity(b):
    if not {'path', 'sha256'} <= set(b) <= {'path', 'sha256', 'size_bytes', 'original_relative_path'}:
        raise ValueError('Unexpected binding keys')
    if not isinstance(b['sha256'], str) or re.fullmatch('[0-9a-f]{64}', b['sha256']) is None:
        raise ValueError('SHA256 required')
    if 'size_bytes' in b and (type(b['size_bytes']) is not int or b['size_bytes'] < 0):
        raise ValueError('Nonnegative integer size required')
    return str(lexical(b['path'])), b['sha256']


def identities(rows):
    out = [identity(x) for x in rows]
    if len(out) != len(set(out)):
        raise ValueError('Duplicate declared binding')
    return set(out)


def raw_read(path):
    with os.fdopen(os.open(safe(path), os.O_RDONLY | os.O_NOFOLLOW), 'rb') as f:
        return f.read()


def binding(path):
    p = safe(path); h = hashlib.sha256(); n = 0
    with os.fdopen(os.open(p, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk); n += len(chunk)
    return {'path': str(p), 'sha256': h.hexdigest(), 'size_bytes': n}


def admitted(b, white, pin=None):
    key = identity(b)
    if key not in identities(white) or (pin is not None and key != pin):
        raise ValueError('Exact declared binding required before FS')
    data = raw_read(b['path'])
    if hashlib.sha256(data).hexdigest() != b['sha256'] or ('size_bytes' in b and len(data) != b['size_bytes']):
        raise ValueError('SHA or size changed')
    return data


def metadata(b, e, pin=None):
    return json.loads(admitted(b, e['metadata_whitelist'], pin))


def load_code(b, e, name, pin):
    data = admitted(b, e['code_whitelist'], pin)
    m = types.ModuleType(name); m.__file__ = str(safe(b['path'])); sys.modules[name] = m
    exec(compile(data, m.__file__, 'exec'), m.__dict__)
    return m


def load_crop_math(b, e):
    data = admitted(b, e['code_whitelist'], PINS['crop_math'])
    nodes = [n for n in ast.parse(data).body if isinstance(n, ast.FunctionDef) and n.name == 'crop']
    if len(nodes) != 1:
        raise ValueError('One pinned pure V90 crop function required')
    ns = {'np': np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), b['path'], 'exec'), ns)
    return types.SimpleNamespace(crop=ns['crop'])


def write_json(path, obj, jsonl=False):
    with os.fdopen(os.open(safe(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), 'w') as f:
        if jsonl:
            for row in obj:
                f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
        else:
            json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
    return binding(path)


def validate_usage(obj):
    rows = obj['roles']; by = {r['pair_id']: r for r in rows}
    if len(rows) != len(by) or len(by) != 22:
        raise ValueError('Exactly22 pair assignments required')
    ranked = sorted(by, key=lambda pid: (hashlib.sha256(('V248:AUX_PAIR_USAGE:17:'+pid).encode()).hexdigest(), pid))
    dev = set(ranked[:5])
    for pid, row in by.items():
        digest = hashlib.sha256(('V248:AUX_PAIR_USAGE:17:'+pid).encode()).hexdigest()
        if row['original_source_split'] != 'TRAIN_LAB' or row['role_hash_sha256'] != digest or row['new_AUX_usage'] != ('AUX_DEV' if pid in dev else 'AUX_FIT'):
            raise ValueError('Frozen pair/sibling usage changed')
    return by


def sample_stage(e, out, usage):
    selected = metadata(e['selected_complete'], e, PINS['selected_complete'])
    white = metadata(e['certificate_manifest'], e, PINS['certificate_manifest'])
    if selected.get('PASS') is not True or selected.get('completed') is not True or selected['selected_pairs'] != 22:
        raise ValueError('Sealed TRAIN22 selection required')
    if identity(selected['metadata']) != identity(e['selected_pairs']) or identity(selected['certificate_payload_whitelist']) != identity(e['certificate_manifest']):
        raise ValueError('Selected metadata closure changed')
    pairs = [json.loads(line) for line in admitted(e['selected_pairs'], e['metadata_whitelist'], PINS['selected_pairs']).splitlines() if line]
    if white['TRAIN_only'] is not True or white['pair_count'] != 22 or len(pairs) != 22:
        raise ValueError('Exact TRAIN22 source scope required')
    if identities(e['payload_whitelist']) != identities(white['files']):
        raise ValueError('Sample stage permits only exact22 certificates')
    helper = load_code(e['paired_truth'], e, 'v248_pinned_cert_math', PINS['paired_truth'])
    sampler = load_code(e['sampler'], e, 'v248_frozen_aux_sampler', PINS['sampler'])
    records, strata, frames, field_qa = [], [], [], []
    seen = set(); certificate_map = {identity(x): x for x in white['files']}
    for pi, pair in enumerate(pairs):
        pid = pair['pair_id']
        if pid in seen or pid not in usage or pair['split'] != 'TRAIN_LAB' or pair['tier'] != 'CONDITIONAL_TEACHER_NOT_GOLD':
            raise ValueError('Unexpected pair/role/authority')
        if pair['pixel_RGB_depth_alignment_certified'] is not False or pair['native_depth_is_assumed_RGB_grid'] is not True or type(pair['supplied_pose_and_K_identical']) is not bool:
            raise ValueError('Conditional calibration assumptions changed')
        seen.add(pid); cb = pair['evidence']
        if identity(cb) not in certificate_map:
            raise ValueError('Pair certificate not in exact whitelist')
        blob = admitted(cb, e['payload_whitelist'])
        with np.load(io.BytesIO(blob), allow_pickle=False) as z:
            for vi, view in enumerate(('A', 'B')):
                fields = [z[k+'_'+view] for k in CERT_KEYS]  # only eight allowed members; no stored class/map/stat reader
                if any(a.dtype != np.bool_ or a.shape != (480, 640) for a in fields):
                    raise ValueError('Four Boolean480x640 fields/view required')
                vp, vn, fp, fn = fields
                if fn.any() or (vp & vn).any() or (vp & ~fp).any() or (fp & fn).any():
                    raise ValueError('Certificate conflict or full-negative forbidden')
                empty = not any(a.any() for a in fields)
                if not pair['supplied_pose_and_K_identical'] and not empty:
                    raise ValueError('Ineligible pair cannot supply nonempty support')
                cert = helper.construct_certainty(*fields)
                answer = sampler.select_aux_view(cert, pair_id=pid, view=view, certificate_binding=cb,
                                                  aux_L=64.0, budget_V=32, budget_E=32, seed=17)
                group = 2*pi+vi; im = pair['files'][view]['image']
                if set(im) != {'path', 'sha256', 'bytes'} or type(im['bytes']) is not int or im['bytes'] < 0:
                    raise ValueError('Exact selected source RGB binding schema required')
                rgb = {'path': im['path'], 'sha256': im['sha256'], 'size_bytes': im['bytes']}; identity(rgb)
                frame = {'group_index': group, 'pair_id': pid, 'view': view, 'original_source_split': 'TRAIN_LAB',
                         'source_role': usage[pid]['new_AUX_usage'], 'image': rgb, 'raw_hw': [480, 640],
                         'frame_input_id': 'PALLOX50:'+pid+(':'+('before' if view == 'A' else 'after'))}
                frames.append(frame)
                for row in answer['records']:
                    if row['native_slot'] is not None or row['training_admission'] is not False:
                        raise ValueError('Raw sampler authority must remain separate')
                    records.append(dict(row, record_id=row['aux_query_id'], group_index=group,
                                        original_source_split='TRAIN_LAB', source_role=frame['source_role'],
                                        frame_input_id=frame['frame_input_id'], image_binding=rgb,
                                        conditional_AUX_FIT_candidate=(frame['source_role'] == 'AUX_FIT'),
                                        Root_conditional_AUX_FIT_admission=False, native_training_admission=False))
                strata.extend(dict(s, source_role=frame['source_role']) for s in answer['strata'])
                field_qa.append({'pair_id': pid, 'view': view, 'four_bool_fields': 4, 'shape': [480, 640],
                                 'empty': empty, 'pair_metadata_eligible': pair['supplied_pose_and_K_identical'],
                                 'O_fullnegative_enabled': False, 'native_labels_created': 0})
    if seen != set(usage) or len(frames) != 44 or len(strata) != 88 or len({x['record_id'] for x in records}) != len(records):
        raise ValueError('Pair/view/row closure changed')
    rgb_white = [f['image'] for f in frames]; identities(rgb_white)
    outputs = {}
    for name, obj, lines in (
      ('AUX_QUERY_RECORDS.jsonl', records, True),
      ('AUX_FRAME44_METADATA.json', {'completed': True, 'PASS': True, 'frames': frames, 'RGB_whitelist44': rgb_white}, False),
      ('AUX_STRATA.json', {'completed': True, 'PASS': True, 'strata': strata, 'no_shortfall_refill': True}, False),
      ('ROLE_PROOF.json', {'completed': True, 'PASS': True, 'pair_usage': e['pair_usage'], 'AUX_FIT_pairs': 17, 'AUX_DEV_pairs': 5, 'all_siblings_same_usage': True}, False),
      ('CERTIFICATE_QA.json', {'completed': True, 'PASS': True, 'views': field_qa, 'only_eight_named_Boolean_members_per_pair': True}, False)):
        outputs[name] = write_json(out/name, obj, lines)
    return {'outputs': outputs, 'record_manifest': outputs['AUX_QUERY_RECORDS.jsonl'], 'frames': outputs['AUX_FRAME44_METADATA.json'],
            'strata': outputs['AUX_STRATA.json'], 'roleproof': outputs['ROLE_PROOF.json'], 'certificate_QA': outputs['CERTIFICATE_QA.json'],
            'selected_complete': e['selected_complete'], 'certificate_manifest': e['certificate_manifest'],
            'certificate_whitelist': white['files'], 'RGB_whitelist44_inert_not_read': rgb_white,
            'counts': {'pairs': 22, 'views': 44, 'records': len(records), 'strata': 88,
                       'per_role_target': dict(Counter(r['source_role']+':'+str(r['conditional_pixel_target_V0_E1']) for r in records))},
            'certificate_transport': 'whole NPZ opaque SHA/ZIP bytes; numerical decoding only eight named bool members',
            'RGB_or_official_weight_reads': 0, 'original_native352_loaded_or_modified': False}


def crop_aux96(rgb, query_xy, L, math_module):
    q = np.asarray(query_xy, np.float64)
    if q.shape != (2,) or not np.isfinite(q).all() or float(L) != 64.0:
        raise ValueError('Prospective AUX query/L required')
    crop, valid = math_module.crop(rgb, q, float(L), 96)
    if crop.dtype != np.uint8 or crop.shape != (3, 96, 96) or valid.dtype != np.bool_ or valid.shape != (1, 96, 96):
        raise ValueError('Pinned V90 F64-to-halfup-U8 CHW crop/centre-valid required')
    return np.ascontiguousarray(crop), np.ascontiguousarray(valid[0])


def feature_stage(e, out, usage):
    import torch
    from PIL import Image, __version__ as PIL_VERSION
    sample = metadata(e['sample_complete'], e)
    if sample.get('completed') is not True or sample.get('PASS') is not True or sample['stage'] != 'sample' or identity(sample['code']) != identity(e['code']):
        raise ValueError('Same frozen sampler build must be complete')
    if identity(sample['root_protocol']) != PINS['root_protocol'] or identity(sample['pair_usage']) != PINS['pair_usage']:
        raise ValueError('Frozen sampling/role authority changed')
    rows = [json.loads(s) for s in admitted(sample['record_manifest'], e['metadata_whitelist']).splitlines() if s]
    frame_meta = metadata(sample['frames'], e); frames = frame_meta['frames']
    rgb_white = frame_meta['RGB_whitelist44']
    if len(frames) != 44 or len(rgb_white) != 44 or identities(rgb_white) != identities(e['RGB_whitelist44']):
        raise ValueError('Exact selected RGB44 closure required')
    for name, b in sample['outputs'].items():
        admitted(b, e['metadata_whitelist'])
    ref = metadata(e['canonical_feature_entry'], e, PINS['canonical_feature_entry'])
    cfg = e['model_loader_config']
    if cfg != ref['model_loader_config'] or cfg['asset_identity'] != {'sha256': 'f433177089a681826f849f194ece3bb48f4d63fb38d32fc837e3dc7a4e5641fb', 'vendor_revision': '7764ea0f912e53c92e82eb78a2a1631e92725fc8'}:
        raise ValueError('Original official loader config must be identical')
    official = cfg['official_state']
    if identities(e['payload_whitelist']) != identities(rgb_white) | {identity(official)} or identities(cfg['payload_whitelist']) != {identity(official)}:
        raise ValueError('Only selected RGB44 and official state admitted')
    if not identities(cfg['code_whitelist']) <= identities(e['code_whitelist']):
        raise ValueError('Loader code closure absent from outer whitelist')
    for b in cfg['code_whitelist']:
        admitted(b, e['code_whitelist'])
    if e['approved_conditional_AUX_FIT_admission'] is not True:
        raise ValueError('Separate Root conditional AUX FIT admission flag required')
    canon = load_code(e['canonical_producer'], e, 'v248_canonical_pixel_only', PINS['canonical_producer'])
    crop = load_code(e['crop_module'], e, 'crop_modules', PINS['crop_module'])
    loader = load_code(e['model_loader_code'], e, 'v248_official_loader', PINS['model_loader_code'])
    math_module = load_crop_math(e['crop_math'], e)
    preprocess = canon.load_preprocessor(e['preprocess_CODE'], e['code_whitelist'], 'preprocess_views', PINS['preprocess_CODE'])
    loader.configure_runtime(); model = loader.load_backbone(cfg, e['device'], treatment=False)
    model.float().eval().requires_grad_(False)
    if any(p.requires_grad for p in model.parameters()) or any(m.training for m in model.modules()):
        raise ValueError('Fully frozen FP32 eval official encoder required')
    before = canon.state_sha(model.state_dict()); official_before = binding(official['path'])
    if identity(official_before) != identity(official):
        raise ValueError('Official file changed')
    n = len(rows); values = np.zeros((n, 1154), np.float32)
    available = np.zeros(n, bool); fractions = np.zeros((n, 2), np.float32); central4 = np.zeros(n, bool)
    frame_by = {int(f['group_index']): f for f in frames}
    if set(frame_by) != set(range(44)) or len(frame_by) != 44:
        raise ValueError('Exact44 group map required')
    decoded = {}; cache = {}; repeat = False; seen = set()
    with torch.no_grad():
        for start in range(0, n, 32):
            crops, valids = [], []
            for i in range(start, min(start+32, n)):
                row = rows[i]; gi = int(row['group_index']); f = frame_by[gi]
                if row['record_id'] in seen or row['record_id'] != row['aux_query_id'] or row['native_slot'] is not None or row['semantic_SELF'] is not None or row['training_admission'] is not False:
                    raise ValueError('AUX row/native authority changed')
                seen.add(row['record_id'])
                role = usage[row['pair_id']]['new_AUX_usage']
                if row['pair_id'] != f['pair_id'] or row['certificate_view'] != f['view'] or row['source_role'] != role or row['original_source_split'] != 'TRAIN_LAB' or identity(row['image_binding']) != identity(f['image']) or row['conditional_pixel_target_V0_E1'] not in (0, 1):
                    raise ValueError('AUX pair/view/target/source RGB identity changed')
                if gi not in cache:
                    raw = admitted(f['image'], e['payload_whitelist'])
                    with Image.open(io.BytesIO(raw)) as im:
                        rgb = np.array(im.convert('RGB'), dtype=np.uint8, copy=True)
                    if rgb.shape != (480, 640, 3):
                        raise ValueError('Original native RGB480x640 required')
                    cache[gi] = rgb
                    decoded[gi] = {'group_index': gi, 'image': f['image'], 'decoded_RGB_sha256': hashlib.sha256(rgb.tobytes()).hexdigest(), 'decoder': 'PIL.Image.open(encoded).convert(RGB)', 'PIL_version': PIL_VERSION}
                    if len(cache) > 4:
                        del cache[next(k for k in cache if k != gi)]
                pix, valid = crop_aux96(cache[gi], row['aux_query_xy'], row['aux_L'], math_module)
                crops.append(pix); valids.append(valid)
                central4[i] = bool(valid[47:49, 47:49].all()); available[i] = central4[i] and bool(valid.any())
                fractions[i] = [valid.mean(), valid[32:64, 32:64].mean()]
                row.update(crop_RGB96_CHW_sha256=hashlib.sha256(pix.tobytes()).hexdigest(), valid96_bool_sha256=hashlib.sha256(valid.tobytes()).hexdigest(),
                           central4=bool(central4[i]), fractions=fractions[i].tolist(), available=bool(available[i]),
                           Root_conditional_AUX_FIT_admission=bool(available[i] and role == 'AUX_FIT'), native_training_admission=False)
            take = np.flatnonzero(available[start:start+len(crops)])
            if len(take):
                rgb_t = torch.from_numpy(np.stack(crops)[take]); valid_t = torch.from_numpy(np.stack(valids)[take, None])
                v, fr, c4 = canon.node_pixel_batch(model, rgb_t, valid_t, preprocess, crop, e['device'])
                if not c4.all() or not np.array_equal(fr, fractions[start+take]):
                    raise ValueError('Canonical pixel fractions/central4 mismatch')
                values[start+take] = v
                if not repeat:
                    v2, f2, c2 = canon.node_pixel_batch(model, rgb_t, valid_t, preprocess, crop, e['device'])
                    if v.tobytes() != v2.tobytes() or fr.tobytes() != f2.tobytes() or c4.tobytes() != c2.tobytes():
                        raise ValueError('Same-batch frozen feature repeat must be byteexact')
                    repeat = True
            print(json.dumps({'stage': 'aux_pixel1154', 'rows': min(start+32, n), 'total': n}), flush=True)
    after = canon.state_sha(model.state_dict()); official_after = binding(official['path'])
    if before != after or identity(official_before) != identity(official_after) or np.count_nonzero(values[~available]) or not np.isfinite(values).all() or (available.any() and not repeat):
        raise ValueError('Frozen state/unsupported/repeat/finite verification failed')
    target = np.array([r['conditional_pixel_target_V0_E1'] for r in rows], np.int8)
    fit = np.array([r['source_role'] == 'AUX_FIT' for r in rows], bool)
    samples = {'target': target, 'fit': fit, 'available': available,
               'group_index': np.array([r['group_index'] for r in rows], np.int64),
               'row_id': np.array([r['record_id'] for r in rows], dtype=str),
               'query_xy': np.array([r['aux_query_xy'] for r in rows], np.float64).reshape(n, 2),
               'L': np.array([r['aux_L'] for r in rows], np.float64),
               'pair_id': np.array([r['pair_id'] for r in rows], dtype=str),
               'view': np.array([r['certificate_view'] for r in rows], dtype=str),
               'source_role': np.array([r['source_role'] for r in rows], dtype=str),
               'original_source_split': np.array([r['original_source_split'] for r in rows], dtype=str)}
    np.save(safe(out/'AUX_FEATURES.npy'), values, allow_pickle=False)
    np.savez(safe(out/'AUX_SAMPLES.npz'), **samples)
    features = binding(out/'AUX_FEATURES.npy'); sb = binding(out/'AUX_SAMPLES.npz')
    records = write_json(out/'AUX_FEATURE_RECORDS.jsonl', rows, True)
    decoder = write_json(out/'RGB_DECODE_AND_CROP_QA.json', {'completed': True, 'PASS': True, 'decoded_frames': list(decoded.values()),
                     'crop': 'pinned V90 crop(rgb,q_aux,64,96): F64 accumulate -> floor(clip+.5) uint8 CHW; centre bounds valid',
                     'V90_original_RAW96_parity_claimed': False, 'unsupported_features_zero': True})
    counts = {'records': n, 'supported': int(available.sum()), 'unsupported_retained': int((~available).sum()),
              'AUX_FIT': int(fit.sum()), 'AUX_DEV': int((~fit).sum()), 'AUX_FIT_supported': int((fit & available).sum()),
              'AUX_DEV_supported': int((~fit & available).sum()),
              'per_role_target': dict(Counter(r['source_role']+':'+str(r['conditional_pixel_target_V0_E1']) for r in rows))}
    return {'features': features, 'samples': sb, 'record_manifest': records, 'sample_complete': e['sample_complete'],
            'strata': sample['strata'], 'roleproof': sample['roleproof'], 'counts': counts,
            'outputs': {'AUX_FEATURES.npy': features, 'AUX_SAMPLES.npz': sb, 'AUX_FEATURE_RECORDS.jsonl': records, 'RGB_DECODE_AND_CROP_QA.json': decoder},
            'official_base': official, 'asset_identity': cfg['asset_identity'], 'official_file_before': official_before, 'official_file_after': official_after,
            'base_state_sha256_before': before, 'base_state_sha256_after': after, 'encoder_state_unchanged': True,
            'first_supported_batch_repeat_byteexact': True if repeat else 'NOT_APPLICABLE_NO_SUPPORTED_ROWS',
            'feature_schema': 'FP32[N,1154] fullCLS384/fixedcentreQ384/centreCLS384/fractions2 only; no native/q8/context feature',
            'AUX_DEV_pretrained_features_only_no_fit_statistics': True, 'original_SOURCE_samples_or_features_loaded': False,
            'new_original_native_labels': 0, 'teacher_fields_as_forward_features': False, 'classifier_optimizer_updates': 0}


def run(entry_path):
    e = json.loads(raw_read(entry_path)); stage = e.get('stage')
    if e.get('approved_public_pixel_aux_stage') is not True or e.get('source_only') is not True or e.get('automatic_retry') is not False or stage not in OUTPUT:
        raise ValueError('Separate Root true stage entry required before any input access')
    for white in ('metadata_whitelist', 'code_whitelist', 'payload_whitelist'):
        identities(e[white])  # all lexical binding checks precede child FS
    if identity(e['code']) != identity(binding(__file__)):
        raise ValueError('Exact self-code required')
    admitted(e['code'], e['code_whitelist']); admitted(e['implementation_protocol'], e['code_whitelist'])
    protocol = metadata(e['root_protocol'], e, PINS['root_protocol'])
    metadata(e['crop_addendum'], e, PINS['crop_addendum'])
    pair_usage = metadata(e['pair_usage'], e, PINS['pair_usage']); usage = validate_usage(pair_usage)
    scope = metadata(e['root_scope'], e)
    contract = dict(e); contract.pop('root_scope')
    contract['metadata_whitelist'] = [b for b in contract['metadata_whitelist'] if identity(b) != identity(e['root_scope'])]
    if scope.get('approved') is not True or scope.get('entry_contract') != contract:
        raise ValueError('Exact Root stage scope/whitelist contract required')
    fixed = protocol['sampler_fixed']
    if any(fixed[k] != v for k, v in {'seed': 17, 'budget_V_per_pair_view': 32, 'budget_E_per_pair_view': 32, 'AUX_L_pixels': 64.0}.items()):
        raise ValueError('Predeclared sampling constants changed')
    if e['output_directory'] != str(OUTPUT[stage]) or e.get('batch_size') != 32 or e.get('device') != 'cuda':
        raise ValueError('Fixed stage output/batch32/CUDA required')
    out = safe(e['output_directory'])
    if out.exists():
        raise ValueError('Fresh immutable output; no retry/overwrite')
    out.mkdir(); started = time.monotonic()
    try:
        result = sample_stage(e, out, usage) if stage == 'sample' else feature_stage(e, out, usage)
        result.update(completed=True, PASS=True, stage=stage, code=e['code'], implementation_protocol=e['implementation_protocol'],
                      root_protocol=e['root_protocol'], crop_addendum=e['crop_addendum'], pair_usage=e['pair_usage'], root_scope=e['root_scope'], entry=binding(entry_path),
                      native_labels_created=0, original_native352_modified=False, current_or_held_payloads=0,
                      new_training_updates=0, PnP=0, automatic_retry=False, seconds=time.monotonic()-started)
        write_json(out/'COMPLETE.json', result)
        print(json.dumps({'completed': True, 'PASS': True, 'complete': binding(out/'COMPLETE.json')}), flush=True)
    except BaseException:
        write_json(out/'FAILURE.json', {'completed': False, 'PASS': False, 'stage': stage, 'traceback': traceback.format_exc(), 'automatic_retry': False})
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--entry', required=True); run(p.parse_args().entry)
