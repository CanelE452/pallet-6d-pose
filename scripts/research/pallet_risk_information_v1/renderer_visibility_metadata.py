"""Read-only synthetic renderer metadata inventory for B diagnostics only.

This export is deliberately separate from prediction/features. Material variant
labels describe scene assets; they do not identify the owner of a boundary
profile, mesh face, or pixel.
"""
from __future__ import annotations
import argparse, collections, gzip, hashlib, json, resource, time
from pathlib import Path

PLACEMENT_KEYS = (
    'diagnostic_mode', 'cargo_visibility_measured', 'front_visibility_after_cargo',
    'left_opening_visibility_after_cargo', 'right_opening_visibility_after_cargo',
    'explicit_metrics_available', 'explicit_occluder_placed',
    'mask_area_target_only', 'mask_area_after_static', 'mask_area_after_cargo',
    'mask_area_after_context', 'mask_area_visible', 'f_static', 'f_cargo',
    'f_context', 'f_explicit', 'f_total', 'occlusion_decomposition_order',
    'front_face_visibility', 'left_opening_visibility', 'right_opening_visibility',
    'opening_visibility_reason',
)
LABEL_KEYS = (
    'pallet_type', 'material_variant_target', 'material_variant_actual',
    'V_actual', 'V_vis_actual', 'ext_occ_corners_actual', 'f_cargo', 'f_occ',
    'f_total', 'occlusion_fraction', 'n_cargo_actual', 'occluder_asset',
    'occluder_placed', 'mask_area_unocc', 'mask_area_after_cargo',
    'mask_area_visible', 'diagnostic_mode', 'mask_area_target_only',
    'mask_area_amodal', 'mask_area_after_static', 'mask_area_after_context',
    'f_static', 'f_context', 'f_explicit', 'mask_profile',
    'occlusion_decomposition_available', 'occlusion_decomposition_order',
    'mask_invariants_pass', 'front_face_visibility', 'left_opening_visibility',
    'right_opening_visibility', 'opening_visibility_reason',
)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text())

def run(repo, upstream, output, *, diagnostic_GO=False):
    if not diagnostic_GO: raise PermissionError('Separate synthetic diagnostic GO required')
    start, cpu = time.monotonic(), time.process_time()
    repo, upstream, output = map(lambda p: Path(p).resolve(), (repo, upstream, output))
    output.mkdir(parents=True, exist_ok=False)
    target_path = upstream / 'synthetic_v1/SYNTH_GT_LABELS.jsonl.gz'
    manifest_path = repo / 'data/pallet/results/pallet_line_pose_v1/SOURCE_MANIFEST.json'
    split_path = upstream / 'synthetic_v1/DATA_SPLIT_LOCK.json'
    original = {str(p): sha(p) for p in (target_path, manifest_path, split_path)}
    split = read(split_path)
    if original[str(manifest_path)] != split['original_source_sha256']:
        raise ValueError('Original renderer source manifest binding differs')
    source = {r['id']: r for r in read(manifest_path)['records']}
    expected = {r['id']: r['split'] for r in split['all_rows']}
    rows, presence, nonnull, material = [], collections.Counter(), collections.Counter(), collections.Counter()
    with gzip.open(target_path, 'rt') as f:
        for line in f:
            target = json.loads(line)
            fid = target['id']
            if target['split'] != expected[fid]: raise ValueError('Split differs: ' + fid)
            path = repo / source[fid]['renderer_annotation_locator_provenance_only']
            digest = sha(path)
            if digest != target['renderer_annotation_sha256']: raise ValueError('Annotation SHA differs: ' + fid)
            obj = read(path)['objects'][0]
            values = {}
            for key in ('front_visibility_cos',):
                if key in obj: values['objects[0].' + key] = obj[key]
            for container, keys in (('scene_placement_v2', PLACEMENT_KEYS), ('v2_labels', LABEL_KEYS)):
                for key in keys:
                    if key in obj.get(container, {}):
                        values['objects[0].' + container + '.' + key] = obj[container][key]
            for key, value in values.items():
                presence[key] += 1
                if value is not None: nonnull[key] += 1
            if 'material_variant_actual' in obj.get('v2_labels', {}):
                material[json.dumps(obj['v2_labels']['material_variant_actual'], sort_keys=True)] += 1
            rows.append(dict(id=fid, split=target['split'], annotation_locator=str(path),
                             annotation_sha256=digest, metadata_by_exact_key_path=values))
    if len(rows) != 1003 or {r['id'] for r in rows} != set(expected):
        raise ValueError('Full original 1003-row coverage differs')
    if any(sha(p) != digest for p, digest in original.items()):
        raise ValueError('Original input changed during read-only export')
    row_path = output / 'B_RENDERER_VISIBILITY_METADATA.jsonl.gz'
    with row_path.open('xb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as gz:
            for row in rows: gz.write((json.dumps(row, sort_keys=True, allow_nan=False) + '\n').encode())
    receipt = dict(schema='SYNTHETIC_RENDERER_METADATA_DIAGNOSTIC_V1', status='PASS', rows=1003,
                   row_sha256=sha(row_path), source_sha256=sha(__file__), input_bindings=original,
                   present_count_by_exact_key_path=dict(presence), nonnull_count_by_exact_key_path=dict(nonnull),
                   material_variant_actual_counts=dict(material),
                   source_scope='Original SHA-bound synthetic renderer annotations only; no rendering/forward/pose solve.',
                   material_metadata='Scene-level material variant labels exist; no per-profile, mesh-face or pixel-owner association is established by this export.',
                   physical_boundary_owner='UNVERIFIED', front_visibility_cos_is_geometric_frontview_metadata=True,
                   coordinates_exported=False, production_feature_or_candidate_insertion=False, real_reference_reads=0,
                   cost=dict(wall_seconds=time.monotonic()-start, CPU_seconds=time.process_time()-cpu,
                             peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, threads=1))
    with (output/'RECEIPT.json').open('x') as f:
        json.dump(receipt, f, indent=2, sort_keys=True, allow_nan=False); f.write('\n')
    return receipt

if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--repo-root', required=True, type=Path)
    p.add_argument('--upstream', required=True, type=Path); p.add_argument('--output', required=True, type=Path)
    p.add_argument('--diagnostic-GO', action='store_true'); a=p.parse_args()
    r=run(a.repo_root, a.upstream, a.output, diagnostic_GO=a.diagnostic_GO)
    print(json.dumps({k:r[k] for k in ('status','rows','row_sha256','cost')}))
