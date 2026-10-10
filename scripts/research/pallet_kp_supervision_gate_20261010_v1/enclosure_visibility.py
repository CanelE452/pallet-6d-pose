"""Source-only sufficient certificate; an enclosure is not a physical mask.

If the camera-to-source segment first enters an enclosing box at its endpoint,
no triangle entirely within that enclosure can precede the endpoint. Point
membership in the actual mesh and boundary ownership remain separate gates.
No model, optimizer, raycast or evaluation truth is loaded here.
"""
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PRIVATE = Path('/dev/shm/pallet-kp-supervision-gate-private-20261010')
EPS = 1e-12


def interval(camera, source, dimensions):
    near, far = -float('inf'), float('inf')
    for origin, target, size in zip(camera, source, dimensions):
        lower, upper = -size / 2, size / 2
        direction = target - origin
        if direction == 0:
            if origin < lower or origin > upper:
                return None
            continue
        a, b = (lower - origin) / direction, (upper - origin) / direction
        near, far = max(near, min(a, b)), min(far, max(a, b))
    if near > far + EPS or far < 0 or near > 1 + EPS:
        return None
    return [near, far]


def checks():
    assert interval([0, 0, -3], [0, 0, -.5], [1, 1, 1])[0] == 1
    assert interval([0, 0, -3], [0, 0, .5], [1, 1, 1])[0] < 1
    assert interval([.5, 0, -3], [.5, 0, -.5], [1, 1, 1])[0] == 1
    assert interval([2, 0, -3], [2, 0, -.5], [1, 1, 1]) is None
    return dict(front_endpoint=True, preceding_enclosure=True,
                closed_boundary_segment=True, outside_parallel_segment=True)


def main():
    protocol_path = PRIVATE / 'ENCLOSURE_PROTOCOL.json'
    protocol = json.loads(protocol_path.read_text())
    inp = REPO / protocol['input']['path']
    assert hashlib.sha256(inp.read_bytes()).hexdigest() == protocol['input']['sha256']
    target = PRIVATE / 'ENCLOSURE_ROWS.jsonl.gz'
    assert not target.exists(), 'Preserve completed source-only diagnostic'
    records = []
    counts = Counter()
    math_checks = checks()
    with gzip.open(inp, 'rt') as handle:
        for line in handle:
            row = json.loads(line)
            R, t = row['R'], row['t']
            camera = [-sum(R[k][j] * t[k] for k in range(3)) for j in range(3)]
            queries = []
            for query in row['queries']:
                if not (query.get('ray_eligible') and
                        query.get('actual_mesh_point_within_original_tolerance') and
                        query.get('in_image') and
                        query.get('supplied_visible_mask_support_3x3') and
                        query.get('whole_edge_has_physical_sample') and
                        query['cached_target'] == 'NONE' and
                        query.get('original_finite') is False):
                    continue
                counts['examined_original_infinite_NONE'] += 1
                point = query['source_X']
                bounds = interval(camera, point, row['dimensions'])
                first_entry = bounds is not None and abs(bounds[0] - 1) <= EPS
                inside = all(abs(x) <= d / 2 + EPS * max(1, d)
                             for x, d in zip(point, row['dimensions']))
                certified = first_entry and inside
                witness = bool(query.get('mesh_and_mask_supported_original_miss_rescued'))
                counts['own_mesh_visibility_certified'] += certified
                counts['certified_with_fixed_offset_depth_witness'] += certified and witness
                counts['certified_without_fixed_offset_depth_witness'] += certified and not witness
                counts['witness_without_enclosure_certificate'] += witness and not certified
                queries.append(dict(query=query['query'], edge=query['edge'],
                                    camera_object_frame=camera, source_X=point,
                                    enclosure_segment_interval=bounds,
                                    source_inside_enclosure=inside,
                                    own_mesh_visibility_certificate=certified,
                                    existing_fixed_offset_depth_witness=witness,
                                    actual_triangle_membership_not_tested_here=True,
                                    target_changed=False))
            records.append(dict(id=row['id'], index=row['index'], queries=queries))
    assert len(records) == 128
    with gzip.open(target, 'wt') as handle:
        for row in records:
            handle.write(json.dumps(row, separators=(',', ':')) + '\n')
    result = dict(schema='source_mesh_enclosure_certificate_v1', complete=True,
                  created_utc=datetime.now(timezone.utc).isoformat(), families=128,
                  counts=dict(counts), checks=math_checks, code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
                  rows_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                  targets_changed=False, new_models=0, new_mesh_rays=0,
                  new_PnP_or_optimization=0, new_training_updates=0,
                  complete_supervision_proven=False)
    (PRIVATE / 'ENCLOSURE_VALIDATION.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
