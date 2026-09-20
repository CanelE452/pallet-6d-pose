"""Recover JSON summary and independently check object-center facing convention."""
import hashlib
import json
import zipfile
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from . import renderer_front_visibility_audit as V

C = V.C


def center_visibility(q, cam):
    q = np.asarray(q, float); cam = np.asarray(cam, float)
    center = q.mean(0); ray = cam-center
    # Diagonal cross product is independent of the first audit's edge pair.
    p = q[np.asarray(V.A.FACES)]
    normals = np.cross(p[:, 2]-p[:, 0], p[:, 3]-p[:, 1])
    norms = np.linalg.norm(normals, axis=1)
    if not (norms > 1e-12).all() or np.linalg.norm(ray) < 1e-12:
        raise ValueError('Degenerate cuboid or camera')
    normals /= norms[:, None]
    normals *= np.where(np.sum(normals*(p.mean(1)-center), axis=1) < 0, -1, 1)[:, None]
    center_cos = normals @ ray / np.linalg.norm(ray)
    face_ray = cam-p.mean(1)
    face_cos = np.sum(normals*face_ray, axis=1)/np.linalg.norm(face_ray, axis=1)
    return center_cos, face_cos


def main():
    rows = C.read(V.RAW/'ROWS.json')
    assert len(rows) == 60000 and all(r['available'] for r in rows)
    protocol = C.read(V.DOC/'PROTOCOL.json')
    for b in protocol['sources']: C.verify(b)
    C.freeze(V.DOC/'CENTER_FOLLOWUP_PROTOCOL.json', dict(
        status='POST_AUDIT_SOURCE_DIAGNOSTIC_NOT_PREREGISTERED_HYPOTHESIS',
        question='Are all stored fronts maximal for a ray from the object center rather than each face center?',
        population='All60000same source records; raw bytes verified against first audit',
        why='799face-center maxima disagreements motivated this followup; do not disguise it as a preplanned test.',
        independence='Normals recomputed from face diagonals, outward oriented by object centroid.',
        tolerance=1e-6, prediction_changes=0, new_annotations=0, real_GT_used=False,
        sources=[C.bound(__file__), C.bound(Path(__file__).with_name('test_finalize_renderer_front_audit.py')),
                 C.bound(V.RAW/'ROWS.json'), C.bound(V.DOC/'PROTOCOL.json')]))
    output = []; buckets = defaultdict(list); elevations = defaultdict(list)
    with ExitStack() as stack:
        archives = {}
        for i, r in enumerate(rows):
            loc = r['locator']
            if '::' in loc:
                archive, member = loc.split('::', 1)
                if archive not in archives: archives[archive] = stack.enter_context(zipfile.ZipFile(archive))
                payload = archives[archive].read(member)
            else: payload = (C.ROOT/loc).read_bytes()
            assert hashlib.sha256(payload).hexdigest() == r['raw_annotation_sha256']
            d = json.loads(payload); q = d['objects'][0]['cuboid']; cam = d['camera_data']['location_worldframe']
            center, face = center_visibility(q, cam)
            discrepancy = float(np.max(np.abs(face-np.asarray(r['cosines']))))
            assert discrepancy < 1e-5, (r['id'], discrepancy)
            output.append(dict(id=r['id'], source=r['source'], center_cosines=center.tolist(),
                center_front_is_max=bool(center[0] >= center.max()-1e-6),
                face_recompute_max_error=discrepancy, face_front_is_max=r['front_is_max']))
            buckets[r['source']].append(r)
            lo = int(np.searchsorted([0,15,30,45,60], r['elevation_deg'], side='right'))
            name = ['[-90,0)', '[0,15)', '[15,30)', '[30,45)', '[45,60)', '[60,90]'][lo]
            elevations[f'{r["source"]}/{name}'].append(r)
            if (i+1)%10000 == 0: print('CENTER_AUDIT', i+1, '/60000', flush=True)
    C.freeze(V.RAW/'CENTER_ROWS.json', output)
    def center_summary(sub):
        return dict(total=len(sub), center_front_is_max=sum(r['center_front_is_max'] for r in sub),
            face_nonmax=sum(not r['face_front_is_max'] for r in sub),
            face_nonmax_center_max=sum(not r['face_front_is_max'] and r['center_front_is_max'] for r in sub),
            independently_recomputed_max_error=max(r['face_recompute_max_error'] for r in sub))
    # Preserve the failed serialization prefix, never pretend it was a valid report.
    path = V.DOC/'RESULTS.json'; partial = V.DOC/'RESULTS_SERIALIZATION_FAILED.partial'
    if path.exists():
        try: C.read(path)
        except json.JSONDecodeError:
            assert not partial.exists()
            path.rename(partial)
    result = dict(status='DIAGNOSTIC_ONLY_NOT_A_CORRECTION',
        recovery='Original RAW/ROWS.json fully saved; summary failed on numpy int64 serialization. Reloading JSON rows restores Python scalar types. Original partial summary retained.',
        all=V.summarize(rows), by_source={k:V.summarize(v) for k,v in buckets.items()},
        by_source_elevation={k:V.summarize(v) for k,v in elevations.items()},
        center_all=center_summary(output),
        center_by_source={k:center_summary([r for r in output if r['source']==k]) for k in buckets},
        evidence=[C.bound(V.DOC/'PROTOCOL.json'), C.bound(V.RAW/'ROWS.json'),
                  C.bound(V.DOC/'CENTER_FOLLOWUP_PROTOCOL.json'), C.bound(V.RAW/'CENTER_ROWS.json'), C.bound(partial)],
        goal_complete=False, real_GT_used=False, predictions_changed=False)
    # Validate serialization completely before exclusive output creation.
    json.dumps(result, allow_nan=False)
    C.freeze(path, result)
    C.freeze(V.DOC/'COMPLETION_AUDIT.json', dict(
        status='SOURCE_DIAGNOSTIC_COMPLETE_GOAL_NOT_COMPLETE', raw_bytes_reverified=60000,
        independent_face_normal_checks=60000, legacy_protocol_bindings_verified=len(protocol['sources']),
        summary=C.bound(path), new_labels=0, new_tags=0, model_changes=0,
        limit='No inference correction, real improvement, labeling-error attribution, or causal error decomposition established.'))
    print(json.dumps(dict(center=result['center_all'], by_source=result['by_source']), indent=2), flush=True)


if __name__ == '__main__': main()
