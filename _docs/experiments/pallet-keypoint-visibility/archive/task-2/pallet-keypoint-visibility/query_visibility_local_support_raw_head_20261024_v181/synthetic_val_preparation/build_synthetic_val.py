"""Existing synthetic VAL: same fixed support operator, evaluation only."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import sys
import time
import numpy as np
import source_end_support_eval as op

H = Path(__file__).resolve().parent
V = H.parent
P = V.parent
C = P / 'query_visibility_full_source_query_causal_census_20261024_v173/census_preparation'
CONFIG = P / 'query_visibility_registered_native_diversity_20261023_v129/render_preparation/BASE_CONFIGS192_BEFORE_RGB.json'
TEACHER = P / 'query_visibility_spatial_stencil_control_20261023_v127/registered_native_teacher_feasibility/build_registered_pallet.py'
OLD_OP = P / 'query_visibility_frozen_sam_source_baseline_20261024_v180/source_end_supervision_preparation_a/source_end_support.py'
SCOPE = V / 'ROOT_SCOPE_BEFORE_NEW_RAW_OR_SAM_CACHE_OR_FIT.json'
PROTOCOL = H / 'PROTOCOL_BEFORE_VAL_PAYLOADS.json'
OUT = V / 'synthetic_VAL2048_support_v1'
FLAGS = ('query_available', 'factual_query_available', 'feature_constructed', 'crop_constructed', 'context_available', 'Q_inside', 'RGB_central4_available', 'native_continuous_valid', 'native_nearest_valid')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def bind(path):
    return dict(path=str(path), sha256=sha(path))


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines()]


def write(path, data):
    with Path(path).open('x') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def jlines(path, data):
    with Path(path).open('x') as f:
        for r in data:
            f.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n')


def prepare():
    assert not PROTOCOL.exists()
    # The sole operator change is accepting the original VALIDATION split for evaluation.
    assert (H / 'source_end_support_eval.py').read_text() == OLD_OP.read_text().replace("record['split'] == 'TRAIN'", "record['split'] in ('TRAIN', 'VALIDATION')")
    inventory = lines(C / 'ALL6144_ORIGINAL_QUERY_METADATA_BEFORE_MASKS.jsonl')
    rows = [r for r in inventory if r['split'] == 'VALIDATION']
    scenes = {s['scene_index']: s for s in lines(C / 'ALL768_SCENE_PATHS_BEFORE_MASKS.jsonl') if s['split'] == 'VALIDATION'}
    cfg = {r['base_index']: r for r in read(CONFIG)['rows'] if r['split'] == 'VALIDATION'}
    assert len(rows) == 2048 and len(scenes) == 256 and len(cfg) == 64
    assert [r['original_global_index'] for r in rows] == list(range(4096, 6144))
    nodes = []
    for n in ast.parse(TEACHER.read_text()).body:
        if isinstance(n, ast.FunctionDef) and n.name in ('add_box', 'make_mesh'):
            nodes.append(n)
        elif isinstance(n, ast.Assign) and any(isinstance(x, ast.Name) and x.id in ('DIM', 'SIGNS') for x in n.targets):
            nodes.append(n)
    assert len(nodes) == 4
    ns = dict(np=np)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(TEACHER) + '::GEOMETRY_ONLY', 'exec'), ns)
    anchors = ns['SIGNS'] * ns['DIM'] / 2
    templates = [ns['make_mesh'](name, anchors)[0] for name in ('WOOD_BOARD_BLOCK', 'POLYMER_RIB_LEG')]
    geometry = []
    for base, c in cfg.items():
        template = templates[c['template_index']]
        scale = np.asarray(c['positive_scale_xyz'], np.float64)
        assert np.array_equal(template['native_anchor3D'] * scale, np.asarray(c['native8_canonical_xyz'], np.float64))
        for slot, canonical in enumerate(c['permCF_to_canonical']):
            geometry.append(dict(geometry_key=f'BASE{base:03d}_NATIVE{slot}', base_index=base, native_slot=slot,
                                 canonical_index=canonical, primary_part_id=int(template['native_primary_part_id'][canonical]),
                                 quads_world=(template['native_patch_quads'][canonical] * scale).tolist(),
                                 camera=c['camera'], raw_hw=c['raw_hw'], native8_CF_xy=c['native8_CF_xy'],
                                 permCF_to_canonical=c['permCF_to_canonical'], GT_supervision_only=True))
    gm = {(r['base_index'], r['native_slot']): r for r in geometry}
    plan = []
    for r in rows:
        g = gm[r['base_index'], r['native_slot']]
        s = scenes[r['scene_index']]
        assert r['declared_canonical_index'] == g['canonical_index'] and r['primary_part_id'] == g['primary_part_id']
        assert s['camera'] == g['camera'] and s['native8_CF_xy'] == g['native8_CF_xy'] and s['permCF_to_canonical'] == g['permCF_to_canonical']
        plan.append(dict(r, geometry_key=g['geometry_key'], scene_buffer=s['scene_buffer'], BEFORE_buffer=s['BEFORE_buffer'], evaluation_only=True))
    jlines(H / 'VAL512_FIXED_PATCH_METADATA.jsonl', geometry)
    jlines(H / 'VAL2048_SUPPORT_INPUT_PLAN.jsonl', plan)
    # Build the SAM input-only manifest using the same source facts as V180.
    source_proto = read(C / 'PROTOCOL_BEFORE_FULL_SOURCE_MASK_BODIES.json')
    sources = {s['scene_index']: s for s in read(source_proto['original_scene_input']['path'])['rows'] if s['split'] == 'VALIDATION'}
    facts = read(source_proto['original_crop_facts']['path'])['rows']
    manifest = []
    for r in rows:
        s, f = sources[r['scene_index']], facts[r['original_global_index']]
        assert f['occurrence_id'] == r['original_occurrence_id'] and f['query_xy'] == r['query_xy'] and f['L'] == r['L']
        manifest.append(dict(original_global_index=r['original_global_index'], original_occurrence_id=r['original_occurrence_id'],
                             scene_index=r['scene_index'], base_index=r['base_index'], native_slot=r['native_slot'], split='VALIDATION', variant=r['variant'], factors=r['factors'],
                             original_query_xy=r['query_xy'], original_L=r['L'], original_input_flags=r['original_input_flags'],
                             original_initial_points8=f['initial_points'], original_q8_context_box_xyxy=f['context_box_xyxy'],
                             full_RGB_packet=s['full_RGB'], full_RGB_key=s['full_RGB_key'], full_RGB_format=s['full_RGB_format'], raw_hw=s['raw_hw'],
                             raw_RGB_PNG=s['raw_RGB_PNG'], decoded_RGB_sha256_inherited=s['decoded_RGB_sha256'],
                             saved_crop_chunk=f['rgb96_chunk'], saved_crop_row=f['rgb96_row'], saved_rgb96_sha256_inherited=f['rgb96_sha256'],
                             saved_valid96_sha256_inherited=f['valid96_sha256'], training_admission=False, evaluation_only=True))
    jlines(H / 'VAL2048_ORIGINAL_INPUTS_METADATA_ONLY.jsonl', manifest)
    protocol = dict(completed=True, prepared_only=True, code=bind(__file__), operator=bind(H / 'source_end_support_eval.py'),
                    original_operator=bind(OLD_OP), scope=bind(SCOPE), plan=bind(H / 'VAL2048_SUPPORT_INPUT_PLAN.jsonl'),
                    geometry=bind(H / 'VAL512_FIXED_PATCH_METADATA.jsonl'), input_manifest=bind(H / 'VAL2048_ORIGINAL_INPUTS_METADATA_ONLY.jsonl'),
                    source_config=bind(CONFIG), source_teacher_code=bind(TEACHER),
                    counts=dict(rows=2048, scenes=256, bases=64, original_U=sum(not r['original_input_flags']['query_available'] for r in rows)),
                    operator_calculation_changed=False, evaluation_only=True, no_fit_or_threshold_selection=True,
                    old_VAL_soft_or_source_pixel_body_reads=0, runtime=dict(executable=sys.executable, numpy=np.__version__), output=str(OUT))
    write(PROTOCOL, protocol)
    print(json.dumps(dict(prepared=True, protocol=bind(PROTOCOL), counts=protocol['counts'])), flush=True)


def load(binding, expected):
    assert sha(binding['path']) == binding['sha256']
    fields = ('part_id', 'camera_depth', 'global_triangle_id', 'ambiguity', 'camera_K', 'camera_R', 'camera_t', 'native8_CF_xy', 'permCF_to_canonical')
    with np.load(binding['path'], allow_pickle=False) as z:
        a = {k: z[k].copy() for k in fields}
    op.validate_buffer_metadata(a, expected['camera'], expected['native8_CF_xy'], expected['permCF_to_canonical'])
    return a


def owner_central4(scene, q, L):
    if q is None or L is None:
        return None
    classes = np.where(scene['part_id'] < 0, 2, np.where(scene['part_id'] == 999, 1, 0)).astype(np.int8)
    classes[scene['ambiguity']] = 3
    points = op.original_crop_positions(q, L).reshape(96, 96, 2)[47:49, 47:49].reshape(-1, 2)
    # Reuse the identical categorical sampler with unused classes4/5.
    wt = op.sample_classes(classes, points).mean(0)
    return [float(wt[0]), float(wt[1]), float(wt[2]), float(wt[3]), float(wt[6])]


def run(entry):
    e = read(entry)
    assert e['approved_VAL_generation'] is True and e['evaluation_only'] is True
    p = read(PROTOCOL)
    assert e['protocol'] == bind(PROTOCOL)
    for key in ('code', 'operator', 'original_operator', 'scope', 'plan', 'geometry'):
        assert bind(p[key]['path']) == p[key]
    assert sys.executable == p['runtime']['executable'] and np.__version__ == p['runtime']['numpy']
    rows, geometry = lines(p['plan']['path']), lines(p['geometry']['path'])
    gm = {(g['base_index'], g['native_slot']): g for g in geometry}
    OUT.mkdir(exist_ok=False)
    write(OUT / 'STARTED.json', dict(entry=bind(entry), protocol=bind(PROTOCOL)))
    results, used = [], {}
    start = time.monotonic()
    for base in range(128, 192):
        rr = [r for r in rows if r['base_index'] == base]
        assert len(rr) == 32
        before = load(rr[0]['BEFORE_buffer'], gm[base, 0])
        used[rr[0]['BEFORE_buffer']['path']] = rr[0]['BEFORE_buffer']
        cache = {slot: dict(op.expected_patch_depth(gm[base, slot]['quads_world'], gm[base, slot]['camera'], [512, 512]),
                            geometry_key=gm[base, slot]['geometry_key']) for slot in range(8)}
        for variant in range(4):
            sr = rr[variant * 8:(variant + 1) * 8]
            scene = before if variant == 0 else load(sr[0]['scene_buffer'], gm[base, 0])
            used[sr[0]['scene_buffer']['path']] = sr[0]['scene_buffer']
            for r in sr:
                result = op.process_one(r, gm[base, r['native_slot']], before, scene, cache[r['native_slot']])
                result.update(owner_central4_soft5=owner_central4(scene, r['query_xy'], r['L']),
                              source_supervision_generated=True, evaluation_only=True, training_admission=False)
                results.append(result)
        if (base + 1) % 16 == 0:
            print(json.dumps(dict(bases=base - 127, rows=len(results), seconds=time.monotonic() - start)), flush=True)
    assert len(results) == 2048 and len(used) == 256
    jlines(OUT / 'ALL2048_VAL_END_SUPPORT.jsonl', results)
    arrays = dict(original_global_index=np.array([r['original_global_index'] for r in results], np.int32),
                  original_occurrence_id=np.array([r['original_occurrence_id'] for r in results]),
                  scene_index=np.array([r['scene_index'] for r in results], np.int32), base_index=np.array([r['base_index'] for r in results], np.int32),
                  native_slot=np.array([r['native_slot'] for r in results], np.int8), split=np.full(2048, 'VALIDATION'),
                  query_xy=np.asarray([[np.nan, np.nan] if r['query_xy'] is None else r['query_xy'] for r in results], np.float64),
                  L=np.asarray([np.nan if r['L'] is None else r['L'] for r in results], np.float64),
                  visible_support_target=np.array([r['visible_support_target'] for r in results], np.int8),
                  foreign_front_support_target=np.array([r['foreign_front_support_target'] for r in results], np.int8),
                  jointmass7=np.asarray([[np.nan] * 7 if r['full96_average_mass'] is None else r['full96_average_mass'] for r in results], np.float64),
                  owner_central4_soft5=np.asarray([[np.nan] * 5 if r['owner_central4_soft5'] is None else r['owner_central4_soft5'] for r in results], np.float64),
                  source_end_class_order=np.array(op.CLASSES), evaluation_only=np.array(True), training_admission=np.array(False))
    arrays.update({flag: np.array([r['original_input_flags'][flag] for r in results], bool) for flag in FLAGS})
    path = OUT / 'VAL2048_SUPPORT_AND_OWNER_ONLY.npz'
    np.savez_compressed(path, **arrays)
    for binding in used.values():
        assert sha(binding['path']) == binding['sha256']
    write(OUT / 'COMPLETE.json', dict(completed=True, PASS=True, entry=bind(entry), protocol=bind(PROTOCOL), rows=2048, scenes=256,
                                     original_input_U=int((~arrays['query_available']).sum()),
                                     rows_output=bind(OUT / 'ALL2048_VAL_END_SUPPORT.jsonl'), sidecar=bind(path),
                                     evaluation_only=True, training_admission=False, original_q_L_native_unchanged=True,
                                     original_native_labels_not_redefined=True, new_model_or_fit_or_PnP=0, seconds=time.monotonic() - start))
    print(json.dumps(dict(completed=True, complete=bind(OUT / 'COMPLETE.json'))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--entry', type=Path)
    args = parser.parse_args()
    assert args.prepare != bool(args.entry)
    prepare() if args.prepare else run(args.entry)
