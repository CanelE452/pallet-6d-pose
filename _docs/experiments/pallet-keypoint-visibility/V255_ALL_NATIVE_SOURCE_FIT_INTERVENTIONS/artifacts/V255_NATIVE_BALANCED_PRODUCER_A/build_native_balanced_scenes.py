"""Root-entry-only V255 registered SOURCE FIT scene intervention renderer.

Only the selected FIT config file is parsed; old BASE_CONFIGS192/GT/RGB are inert.
The original renderer's BEFORE pixel SHA is a hard gate before any scene emit.
"""
import argparse
import copy
import hashlib
import io
import json
import os
import platform
from pathlib import Path
import stat
import sys
import types
import time
import traceback
from collections import Counter
import numpy as np

P = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
PINNED_CODE = {
    'renderer': (P/'query_visibility_registered_native_diversity_20261023_v129/render_preparation/render_corpus_once.py', 'f6093f950b9af0aff8494f3fc03d633ac6789a53e70e381d85d4e9f7716e0ec6'),
    'core': (P/'query_visibility_registered_native_diversity_20261023_v129/render_preparation/corpus_math.py', '96713b5c8380e7698081d9189d54b3f4911ea6cda14cbd47083a79f4df36817e'),
    'teacher': (P/'query_visibility_spatial_stencil_control_20261023_v127/registered_native_teacher_feasibility/build_registered_pallet.py', '1ac00f53f037ceeebe3d67d6f76f69cbbdf3781461c514ae6b483256f3e7c01d'),
    'raster': (P/'query_visibility_synthetic_part_teacher_20261023_v93/raster_preparation/cpu_part_raster.py', '1ed9801add06cb9fec4c15b0078948d243a08625567c718cbf3ee01851a19d53'),
    'helper': (P/'query_visibility_registered_native_render_20261023_v128/render_preparation/render_registered_six_once.py', 'f82798206265fbff7138dc6860af6da7beb0560327d871b8802f0cd37a94b7fa'),
}
VARIANTS = ('COVER', 'NEAR_SHAM', 'FAR_SHAM')
MESH_KEYS = ('vertices', 'triangles', 'part_owner', 'face_axis', 'face_side',
             'native_anchor3D', 'native_primary_part_id', 'native_incident_part_ids',
             'native_patch_quads', 'native_edge_samples', 'native_witness_points')
ROOT_PROTOCOL = {'path': str(P/'publication_root_preparation/V255_ROOT_NATIVE_COVERAGE/PROTOCOL_BEFORE_NATIVE_3D_RENDER.json'), 'sha256': '340ec90f3389255763a99402aad77d587a321eb352dab39e1f4fef1b171f05c1', 'size_bytes': 8330}
STAGE1_QA = {'path': str(P/'publication_root_preparation/V255_NEXT_INTERVENTION_QA_C/actual_review_v1/REVIEW.json'), 'sha256': '6a96d2f2cc6b315ec52a56cc56da0fa7e49c1b58580104b1e0f65fecfbdee71f', 'size_bytes': 6085}


def need(ok, message):
    if not bool(ok):
        raise ValueError(message)


def safe(path, exists=True):
    raw = str(path)
    need(not any(t in raw.casefold() for t in
         ('v112', 'cache', '.pyc', 'current223', 'publicheld', 'oldval',
          'holdout', 'private_evaluation')), 'Forbidden lexical path before FS')
    p = Path(raw)
    need(p.is_absolute() and '..' not in p.parts, 'Exact absolute path required')
    for q in [*reversed(p.parents), p]:
        try:
            mode = os.lstat(q).st_mode
        except FileNotFoundError:
            need(not exists, 'Exact path missing')
            continue
        need(not stat.S_ISLNK(mode), 'Symlink excluded')
    return p


def load_pinned_modules(root_code_bindings, expected_numpy):
    """CODE-only load; the explicit Root stage2 entry must authorize this call."""
    need(np.__version__ == expected_numpy, 'Original renderer NumPy identity')
    need(set(root_code_bindings) == set(PINNED_CODE), 'Closed five CODE bindings')
    out = {}
    sys.dont_write_bytecode = True
    for key, (path, sha) in PINNED_CODE.items():
        b = root_code_bindings[key]
        need(b['path'] == str(path) and b['sha256'] == sha, 'Pinned CODE identity')
        raw = safe(path).read_bytes()
        need(hashlib.sha256(raw).hexdigest() == sha, 'Pinned CODE bytes')
        if 'size_bytes' in b:
            need(len(raw) == b['size_bytes'], 'Pinned CODE size')
        name = 'v255_frozen_' + key + '_' + sha[:12]
        m = types.ModuleType(name)
        m.__file__ = str(path)
        m.__package__ = None
        sys.modules[name] = m
        exec(compile(raw, str(path), 'exec'), m.__dict__)
        out[key] = m
    return out


def rgb_sha(rgb):
    need(rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3, 'U8 HWC RGB')
    return hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()


def point0_states(records_CF, field):
    need(len(records_CF) == 8, 'Eight fresh registered records')
    states = []
    for k, record in enumerate(records_CF):
        need(record['CF_index'] == k, 'Original CF/native order')
        values = record[field]
        need(len(values) == 37 and all(s in ('D', 'E', 'S', 'U') for s in values), 'Witness37 states')
        states.append(values[0])
    return states


def input_identity(before_input, cfg):
    """Validate shape/identity without changing q/L/U or deriving a teacher query."""
    b = int(cfg['base_index'])
    need(cfg['base_id'] == 'V129_BASE_%03d' % b, 'Original base ID')
    need(cfg['split'] == 'TRAIN' and before_input['source_role'] == 'FIT',
         'Original TRAIN and separately admitted internal SOURCE FIT')
    need(before_input['base_index'] == b and before_input['base_id'] == cfg['base_id'],
         'Original SOURCE FIT base join')
    need(before_input['scene_index'] == 4*b and before_input['variant'] == 'BEFORE',
         'Original BEFORE identity')
    need(before_input['source_base_config']['row_index'] == b, 'Original config row')
    need(before_input['source_base_config']['file']['sha256'] ==
         'a63c1e6931cbff892dc0738c6571a7a9a57c588a8a1df66bcdcd409d638a2e86',
         'Original config binding')
    q8 = before_input['q8']
    need(isinstance(q8, list) and len(q8) == 8, 'Original q8 members')
    # Validate nullable transport without changing the saved list or original U.
    need(all(q is None or (isinstance(q, list) and len(q) == 2) for q in q8),
         'Original nullable q8 shape')
    need(np.asarray(before_input['L8'], np.float64).shape == (8,), 'Original L8 shape')
    av = np.asarray(before_input['original_available8'])
    need(av.shape == (8,) and av.dtype == bool, 'Original availability8 bool')
    need(len(before_input['decoded_RGB_sha256']) == 64, 'Original BEFORE HWC pixel SHA')
    return copy.deepcopy(before_input)


def render_base_interventions(template, cfg, before_input, modules, emit):
    """Pure in-memory generation, streamed via emit(metadata, RGB_uint8_HWC).

    No original Source RGB/GT is read. BEFORE newly-rendered pixel SHA must exactly
    equal its frozen original input metadata before any event is emitted.
    Target-alone S/U trials are kept in the ledger and are never turned into E.
    Crop/context extraction is deliberately a later coherent-RGB-only adapter.
    """
    need(set(template) == set(MESH_KEYS), 'Only original typed geometry members')
    need(set(modules) == set(PINNED_CODE), 'Five frozen CODE modules')
    original_cfg = copy.deepcopy(cfg)
    qmeta = input_identity(before_input, original_cfg)
    renderer, core = modules['renderer'], modules['core']
    teacher, raster, helper = modules['teacher'], modules['raster'], modules['helper']
    data = core.scaled_data(template, original_cfg)
    vertices, tri, pid = data['vertices'], data['triangles'], data['part_owner']
    need(tri.ndim == 2 and tri.shape[1] == 3 and len(pid) == len(tri), 'Mesh face identity')
    # Constructive make_mesh uses contiguous part IDs; equivalent to len(parts).
    part_ids = np.unique(pid)
    need(np.array_equal(part_ids, np.arange(len(part_ids))), 'Registered part IDs')
    camera = renderer.camera_arrays(original_cfg['camera'])
    perm = list(original_cfg['permCF_to_canonical'])
    need(sorted(perm) == list(range(8)), 'Original CF-to-canonical permutation')
    xy = raster.project_camera(raster.world_to_camera(data['native_anchor3D'], camera), camera)
    need(np.array_equal(xy[perm], np.asarray(original_cfg['native8_CF_xy'])),
         'Original geometric reference only, never replacement q')
    F = len(tri)
    depth, faces, _ = raster.rasterize_parts(
        raster.world_to_camera(vertices, camera), tri, pid, np.arange(F, dtype=np.int32),
        np.ones(F, bool), np.ones(F, np.int8), camera, len(part_ids))
    target_digest = helper.digest(depth, faces)
    before_buf = raster.resolve_layers(depth, faces)
    before_rgb = core.shade(before_buf, vertices, tri, original_cfg, np.zeros(3))
    expected_sha = qmeta['decoded_RGB_sha256']
    need(rgb_sha(before_rgb) == expected_sha, 'BEFORE decoded RGB exact SHA hardgate')
    baseline = teacher.region_teacher(data, camera, raster, None)
    baseline_CF = core.CF_records(baseline, perm, data, original_cfg['camera'])
    alone8 = point0_states(baseline_CF, 'target_witness_states')
    combined_before8 = point0_states(baseline_CF, 'combined_witness_states')
    need(alone8 == combined_before8, 'BEFORE has no foreign BOX')
    common = {
        'base_index': int(original_cfg['base_index']), 'base_id': original_cfg['base_id'],
        'source_role': 'FIT', 'original_before_input': qmeta,
        'original_config_provenance': copy.deepcopy(qmeta['source_base_config']),
        'permCF_to_canonical': perm, 'target_layer_digest': target_digest,
        'teacher_authority': 'REGISTERED_SYNTHETIC_POINT0_WHOLE_TARGET_RAY_FOREIGN999',
        'literal_q_or_wholepart_or_real_native_gold': False,
        'training_admission': False,
    }
    emit(dict(common, kind='BEFORE', variant='BEFORE', target_native_slot=None,
              RGB_sha256_HWC=expected_sha, fresh_records_CF=baseline_CF,
              point0_target_alone_states8=alone8,
              point0_combined_states8=combined_before8), before_rgb.copy())
    ledger = []
    for k in range(8):
        trial = {'base_index': int(original_cfg['base_index']), 'native_slot': k,
                 'canonical_index': int(perm[k]), 'target_alone_point0_state': alone8[k],
                 'original_query_available': bool(qmeta['original_available8'][k]),
                 'rendered_variants': [], 'training_admission': False}
        if alone8[k] != 'D':
            trial.update(constructed=False, reason='TARGET_ALONE_POINT0_NOT_D',
                         retained_geometric_state=alone8[k], constructed_target=None)
            ledger.append(trial)
            continue
        intervention_cfg = copy.deepcopy(original_cfg)
        intervention_cfg['selected_CF_index'] = k
        intervention_cfg['selected_canonical_index'] = int(perm[k])
        foreign, translations, placement = core.foreign_placements(data, intervention_cfg)
        actual_target_states = {}
        for variant in VARIANTS:
            fv = foreign[variant]
            fd, ff, _ = raster.rasterize_parts(
                raster.world_to_camera(fv, camera), teacher.BOX_TRI,
                np.zeros(12, np.int32), F+np.arange(12, dtype=np.int32),
                np.ones(12, bool), np.ones(12, np.int8), camera, 1)
            buf = raster.resolve_layers(depth, faces, foreign=(fd[0], ff[0], 999))
            scene_vertices = np.concatenate([vertices, fv])
            scene_triangles = np.concatenate([tri, teacher.BOX_TRI+len(vertices)])
            rgb = core.shade(buf, scene_vertices, scene_triangles,
                             intervention_cfg, translations[variant])
            same = ((buf['part_id'] >= 0) & (buf['part_id'] != 999) &
                    (buf['part_id'] == before_buf['part_id']) &
                    (buf['face_id'] == before_buf['face_id']) &
                    (buf['depth'] == before_buf['depth']))
            need(np.array_equal(rgb[same], before_rgb[same]), 'Visible target RGB unchanged')
            need(helper.digest(depth, faces) == target_digest, 'Target layers immutable')
            fresh = teacher.region_teacher(data, camera, raster, fv)
            fresh_CF = core.CF_records(fresh, perm, data, original_cfg['camera'])
            new_alone8 = point0_states(fresh_CF, 'target_witness_states')
            combined8 = point0_states(fresh_CF, 'combined_witness_states')
            need(new_alone8 == alone8, 'Target-alone witness states immutable')
            actual_target_states[variant] = combined8[k]
            emit(dict(common, kind='NATIVE_INTERVENTION', variant=variant,
                      target_native_slot=k, target_canonical_index=int(perm[k]),
                      placement=copy.deepcopy(placement),
                      RGB_sha256_HWC=rgb_sha(rgb),
                      fresh_records_CF=fresh_CF,
                      point0_target_alone_states8=alone8,
                      point0_combined_states8=combined8,
                      actual_target_point0_state=combined8[k],
                      native_RGB_changed_pixels=int(np.any(rgb != before_rgb, axis=2).sum()),
                      unchanged_target_surface_pixels=int(same.sum())),
                 rgb)
            trial['rendered_variants'].append(variant)
        trial.update(constructed=True, reason=None,
                     actual_target_point0_states=actual_target_states,
                     observed_D_E_D_D=(actual_target_states ==
                         {'COVER': 'E', 'NEAR_SHAM': 'D', 'FAR_SHAM': 'D'}))
        ledger.append(trial)
    need(len(ledger) == 8 and cfg == original_cfg, 'All trials and original config preserved')
    return {'completed': True, 'base_index': int(original_cfg['base_index']),
            'BEFORE_pixel_SHA_exact': True, 'trials': ledger,
            'original_q8_L8_U_unchanged': True, 'new_label_fallback': False,
            'training_admission': False}


def bound_bytes(binding):
    """Only caller-admitted exact bindings are read; embedded pointers are inert."""
    need(isinstance(binding, dict) and set(('path', 'sha256')) <= set(binding), 'Binding schema')
    need(isinstance(binding['sha256'], str) and len(binding['sha256']) == 64,
         'Binding SHA64')
    raw = safe(binding['path']).read_bytes()
    need(hashlib.sha256(raw).hexdigest() == binding['sha256'], 'Bound bytes SHA')
    if 'size_bytes' in binding:
        need(len(raw) == binding['size_bytes'], 'Bound bytes size')
    return raw


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def encoded_json(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False, default=json_default) + '\n').encode('utf-8')


def bind_written(path):
    raw = safe(path).read_bytes()
    return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'size_bytes': len(raw)}


def put_json(path, value):
    p = safe(path, exists=False)
    need(not p.exists(), 'Fresh output file')
    with p.open('xb') as f:
        f.write(encoded_json(value))
    return bind_written(p)


def adapt_before_input(parent, cfg, stage1_complete):
    """Explicit transport aliases; original q8/L8/U/raw_hw/image remain untouched."""
    need(parent['identity_PASS'] is True and not parent['failure_reasons'], 'Stage1 parent identity')
    b = int(cfg['base_index'])
    need(parent['base_index'] == b == parent['config_row_index'], 'Selected config/base row')
    need(parent['source_base_role'] == 'FIT' and parent['all_original_parent_rows'] == 8,
         'All original FIT members retained')
    members = parent['native_members']
    need(len(members) == 8 and sorted(x['native_slot'] for x in members) == list(range(8)),
         'Original native memberships8')
    original = parent['before_input']
    need(original['originalpermCF_to_canonical'] == cfg['permCF_to_canonical'], 'Original native permutation')
    need(original['image']['path'] == original['expected_BEFORE_encoded_RGB_binding']['path'] and
         original['image']['sha256'] == original['expected_BEFORE_encoded_RGB_binding']['sha256'],
         'Original encoded pointer identity only; no archive read')
    need(original['render_raw_hw_declared_separately'] == cfg['raw_hw'], 'Declared render HW')
    qmeta = copy.deepcopy(original)
    qmeta.update(source_role=parent['source_base_role'], base_index=b,
                 base_id=cfg['base_id'], scene_index=4*b, variant='BEFORE',
                 decoded_RGB_sha256=original['expected_BEFORE_decoded_RGB_sha256'],
                 source_base_config={'file': copy.deepcopy(stage1_complete['inputs']['configs192_opaque']),
                                     'row_index': parent['config_row_index']},
                 selected_config_span=copy.deepcopy(parent['selected_config_span']))
    input_identity(qmeta, cfg)
    need(all(qmeta[k] == v for k, v in original.items()), 'Original BEFORE metadata unchanged')
    return qmeta


def run(entry_path):
    started = time.monotonic()
    entry_raw = safe(entry_path).read_bytes()
    entry = json.loads(entry_raw)
    # False entry exits before selected config, geometry, installed modules or renderer.
    need(entry.get('approved_actual_native_balanced_render') is True, 'Root actual entry required')
    need(entry.get('root_protocol') == ROOT_PROTOCOL and entry.get('stage1_QA') == STAGE1_QA,
         'Frozen Root protocol and Stage1 actual QA')
    own = {'path': str(Path(__file__).absolute()),
           'sha256': hashlib.sha256(safe(__file__).read_bytes()).hexdigest(),
           'size_bytes': len(safe(__file__).read_bytes())}
    need(entry.get('code') == own, 'Entry exact producer CODE')
    proto = json.loads(bound_bytes(ROOT_PROTOCOL))
    q1 = json.loads(bound_bytes(STAGE1_QA))
    need(q1.get('completed') is True and q1.get('PASS') is True, 'Stage1 QA PASS')
    q2 = json.loads(bound_bytes(entry['stage2_CODE_QA']))
    need(q2.get('completed') is True and q2.get('PASS') is True, 'Stage2 CODE QA PASS')
    need(q2.get('code') == own, 'Stage2 QA exact producer binding')
    scope_binding = entry['root_scope']
    scope = json.loads(bound_bytes(scope_binding))
    need(scope.get('approved_actual_native_balanced_render') is True and
         scope.get('entry_contract') == {k: v for k, v in entry.items() if k != 'root_scope'},
         'Root exact entry contract')
    need(entry.get('actual_payload_read_allowlist') == proto['mesh_payload_bindings'],
         'Closed two-mesh payload whitelist')
    need(entry.get('automatic_retry') is False and entry.get('current_or_held_access') is False and
         entry.get('old_SCENES_GT_JSON_read_allowed') is False and
         entry.get('full_BASE_CONFIGS192_json_decode_allowed') is False and
         entry.get('new_encoder_CP_model_forward_or_fit') is False, 'Frozen negative access flags')
    need(entry.get('output_directory') == proto['output_directory'], 'Fixed output directory')
    out = safe(proto['output_directory'], exists=False)
    need(not out.exists(), 'Fresh output directory, no resume/retry')
    out.mkdir()
    made, base_receipts, current_base = [], [], None
    trial_count, scene_count = 0, 0
    scene_path, ledger_path = out/'SCENES.jsonl', out/'ALL_NATIVE_TRIALS.jsonl'
    try:
        rt = proto['runtime']
        need(sys.executable == rt['executable'] and platform.python_version() == rt['python'] and
             np.__version__ == rt['NumPy'], 'Original renderer runtime identity')
        stage1 = json.loads(bound_bytes(proto['source_fit_selection_complete']))
        configs = json.loads(bound_bytes(proto['selected_configs_manifest']))
        parents = json.loads(bound_bytes(proto['before_input_manifest']))
        need(stage1.get('completed') is True and stage1.get('PASS') is True and
             configs.get('completed') is True and configs['selected_from_original_SOURCE_FIT_only'] is True and
             parents.get('completed') is True and parents['all_parent_identity_PASS'] is True,
             'Selected FIT metadata PASS')
        need(stage1['outputs']['SELECTED_FIT_CONFIGS.json'] == proto['selected_configs_manifest'] and
             stage1['outputs']['SOURCE_FIT_PARENT_MANIFEST.json'] == proto['before_input_manifest'] and
             q1['stage1_complete'] == proto['source_fit_selection_complete'], 'Stage1 binding closure')
        cfgs = configs['rows']
        parent_by_base = {int(p['base_index']): p for p in parents['parents']}
        nbase = stage1['census']['selected_original_FIT_base_count']
        need(len(cfgs) == len(parent_by_base) == len(parents['parents']) == nbase, 'All selected bases unique')
        need(len({int(c['base_index']) for c in cfgs}) == nbase and
             set(int(c['base_index']) for c in cfgs) == set(parent_by_base), 'Selected config joins')
        need(stage1['census']['selected_original_FIT_beforeanchor_BEFORE_members'] == nbase*8,
             'All selected original native memberships')
        modules = load_pinned_modules(proto['code_bindings'], rt['NumPy'])
        templates = []
        for mesh_binding in proto['mesh_payload_bindings']:
            with np.load(io.BytesIO(bound_bytes(mesh_binding)), allow_pickle=False) as z:
                need(set(MESH_KEYS) <= set(z.files), 'Registered mesh numeric schema')
                templates.append({k: z[k].copy() for k in MESH_KEYS})
        scene_dir, base_dir = out/'scenes', out/'bases'
        safe(scene_dir, exists=False).mkdir()
        safe(base_dir, exists=False).mkdir()
        by_native = [Counter() for _ in range(8)]
        unavailable_retained, masked_retained = 0, 0
        with safe(scene_path, exists=False).open('x', encoding='utf-8') as scene_stream, \
             safe(ledger_path, exists=False).open('x', encoding='utf-8') as trial_stream:
            for cfg in sorted(cfgs, key=lambda c: int(c['base_index'])):
                current_base = int(cfg['base_index'])
                parent = parent_by_base[current_base]
                qmeta = adapt_before_input(parent, cfg, stage1)
                mesh_index = int(cfg['template_index'])
                need(0 <= mesh_index < len(templates) and
                     cfg['template'] == {k: proto['mesh_payload_bindings'][mesh_index][k] for k in ('path', 'sha256')},
                     'Original template identity')
                emitted_for_base = []

                def emit(metadata, rgb):
                    nonlocal scene_count
                    need(metadata['original_before_input'] == qmeta, 'Original inputs frozen before save')
                    need(rgb_sha(rgb) == metadata['RGB_sha256_HWC'], 'Saved HWC byte identity')
                    need(list(rgb.shape[:2]) == qmeta['render_raw_hw_declared_separately'], 'Render HW')
                    native = metadata['target_native_slot']
                    stem = cfg['base_id'] + ('_BEFORE' if native is None else
                           '_NATIVE%d_%s' % (native, metadata['variant']))
                    pixel_path, metadata_path = scene_dir/(stem+'.npz'), scene_dir/(stem+'.json')
                    with safe(pixel_path, exists=False).open('xb') as f:
                        np.savez_compressed(f, RGB_uint8_HWC=rgb)
                    pixel_binding = bind_written(pixel_path)
                    made.append(pixel_binding)
                    saved_metadata = dict(metadata, image=dict(pixel_binding, format='npz', member='RGB_uint8_HWC'),
                                          RGB_shape=list(rgb.shape), RGB_dtype='uint8',
                                          model_input_geometry_source='UNCHANGED_ORIGINAL_BEFORE_Q8_L8_U',
                                          teacher_records_are_model_input=False)
                    metadata_binding = put_json(metadata_path, saved_metadata)
                    made.append(metadata_binding)
                    row = {'scene_index': scene_count, 'scene_id': stem, 'base_index': current_base,
                           'base_id': cfg['base_id'], 'source_role': 'FIT', 'variant': metadata['variant'],
                           'target_native_slot': native, 'image': saved_metadata['image'],
                           'metadata': metadata_binding, 'RGB_sha256_HWC': metadata['RGB_sha256_HWC'],
                           'point0_combined_states8': metadata['point0_combined_states8'],
                           'training_admission': False}
                    scene_stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)+'\n')
                    scene_stream.flush()
                    emitted_for_base.append(row)
                    scene_count += 1

                receipt = render_base_interventions(templates[mesh_index], cfg, qmeta, modules, emit)
                receipt.update(original_parent=copy.deepcopy(parent), scenes=emitted_for_base)
                for trial in receipt['trials']:
                    trial_count += 1
                    k = trial['native_slot']
                    by_native[k]['trials'] += 1
                    by_native[k]['target_alone_'+trial['target_alone_point0_state']] += 1
                    by_native[k]['constructed' if trial['constructed'] else 'not_constructed'] += 1
                    if trial['constructed']:
                        by_native[k]['observed_D_E_D_D' if trial['observed_D_E_D_D'] else 'proof_not_D_E_D_D'] += 1
                    trial_stream.write(json.dumps(trial, ensure_ascii=False, sort_keys=True, allow_nan=False)+'\n')
                trial_stream.flush()
                unavailable_retained += parent['unavailable_native_members_retained']
                masked_retained += parent['masked_native_members_retained']
                rb = put_json(base_dir/(cfg['base_id']+'_COMPLETE.json'), receipt)
                made.append(rb)
                base_receipts.append(rb)
                print(json.dumps({'completed_bases': len(base_receipts), 'selected_bases': nbase,
                                  'base_index': current_base, 'scenes': scene_count,
                                  'native_trials': trial_count}, sort_keys=True), flush=True)
        need(len(base_receipts) == nbase and trial_count == nbase*8, 'All selected trial denominator')
        made.extend([bind_written(scene_path), bind_written(ledger_path)])
        summary = {'completed': True, 'selected_original_FIT_bases': nbase,
                   'all_native_trials': trial_count, 'scenes': scene_count,
                   'per_native': {str(k): dict(v) for k, v in enumerate(by_native)},
                   'original_unavailable_native_members_retained': unavailable_retained,
                   'original_masked_native_members_retained': masked_retained,
                   'all_BEFORE_pixel_SHA_exact': True, 'original_q8_L8_U_unchanged': True,
                   'variant_name_not_used_as_label': True, 'training_admission': False,
                   'literal_q_wholepart_real_native_gold': False,
                   'raw_original_RGB_GT_JSON_BASE192_reads': 0,
                   'new_encoder_model_CP_fit_PnP': 0, 'current_or_held_access': False}
        sb = put_json(out/'SUMMARY.json', summary)
        made.append(sb)
        manifest = put_json(out/'MANIFEST.json', {'files': made, 'treewalk': False})
        entry_binding = {'path': str(entry_path), 'sha256': hashlib.sha256(entry_raw).hexdigest(),
                         'size_bytes': len(entry_raw)}
        complete = {'completed': True, 'PASS': True, 'code': own, 'root_protocol': ROOT_PROTOCOL,
                    'entry': entry_binding, 'root_scope': scope_binding, 'stage1_QA': STAGE1_QA,
                    'stage2_CODE_QA': entry['stage2_CODE_QA'],
                    'source_fit_selection_complete': proto['source_fit_selection_complete'],
                    'selected_configs_manifest': proto['selected_configs_manifest'],
                    'before_input_manifest': proto['before_input_manifest'],
                    'mesh_payload_bindings': proto['mesh_payload_bindings'],
                    'code_bindings': proto['code_bindings'], 'runtime': rt,
                    'outputs': {'SCENES.jsonl': bind_written(scene_path),
                                'ALL_NATIVE_TRIALS.jsonl': bind_written(ledger_path),
                                'SUMMARY.json': sb, 'MANIFEST.json': manifest},
                    'base_receipts': base_receipts, 'counts': summary,
                    'training_admission': False, 'new_encoder_model_CP_fit': 0,
                    'current_or_held_access': False, 'automatic_retry': False,
                    'elapsed_seconds': time.monotonic()-started}
        cb = put_json(out/'COMPLETE.json', complete)
        print(json.dumps({'completed': True, 'PASS': True, 'COMPLETE': cb}, sort_keys=True), flush=True)
        return complete
    except BaseException as exc:
        failure = {'completed': False, 'PASS': False, 'error_type': type(exc).__name__, 'error': str(exc),
                   'traceback': traceback.format_exc(), 'code': own, 'root_protocol': ROOT_PROTOCOL,
                   'current_base_index': current_base, 'completed_base_receipts': base_receipts,
                   'completed_scene_count': scene_count, 'completed_native_trial_count': trial_count,
                   'written_files': made, 'partial_scene_manifest_path': str(scene_path),
                   'partial_trial_ledger_path': str(ledger_path), 'automatic_retry': False,
                   'training_admission': False, 'elapsed_seconds': time.monotonic()-started}
        put_json(out/'FAILURE.json', failure)
        raise


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--entry', required=True)
    args = ap.parse_args()
    run(args.entry)
