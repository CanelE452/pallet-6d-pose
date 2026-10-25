"""V255 pure registered-scene renderer. No actual I/O entry is implemented yet.

The caller supplies only Root-selected SOURCE FIT configs and typed mesh payloads.
This module never reads BASE_CONFIGS192, old scene GT, RGB or source row payloads.
"""
import argparse
import copy
import hashlib
import os
from pathlib import Path
import stat
import sys
import types
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


def need(ok, message):
    if not bool(ok):
        raise ValueError(message)


def safe(path):
    raw = str(path)
    need(not any(t in raw.casefold() for t in
         ('v112', 'cache', '.pyc', 'current223', 'publicheld', 'oldval',
          'holdout', 'private_evaluation')), 'Forbidden lexical path before FS')
    p = Path(raw)
    need(p.is_absolute() and '..' not in p.parts, 'Exact absolute path required')
    for q in [*reversed(p.parents), p]:
        need(not stat.S_ISLNK(os.lstat(q).st_mode), 'Symlink excluded')
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
    need(np.asarray(before_input['q8'], np.float64).shape == (8, 2), 'Original q8 shape')
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


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--entry')
    ap.parse_args()
    raise SystemExit('PREPARATION_ONLY: Root stage2 I/O/whitelist protocol not yet bound; no Source payload read.')
