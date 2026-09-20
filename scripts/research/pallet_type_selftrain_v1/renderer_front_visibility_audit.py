"""Read-only renderer provenance audit; not a relabeler or inference filter."""
import hashlib
import json
import zipfile
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from . import camera_facing_contract_audit as A

C = A.C
PHASE = 'renderer_front_visibility_audit'
DOC = A.R.DOC / PHASE
RAW = A.R.RAW / PHASE


def visibility(cuboid, camera):
    """Outward face-normal cosines; world units and rigid transforms cancel."""
    q = np.asarray(cuboid, dtype=float)
    cam = np.asarray(camera, dtype=float)
    if q.shape != (8, 3) or cam.shape != (3,) or not np.isfinite(q).all() or not np.isfinite(cam).all():
        raise ValueError('Finite world-frame cuboid and camera required')
    center = q.mean(0)
    values = []
    for face in A.FACES:
        p = q[face]
        fc = p.mean(0)
        n = np.cross(p[1] - p[0], p[3] - p[0])
        if np.linalg.norm(n) < 1e-12 or np.linalg.norm(cam - fc) < 1e-12:
            raise ValueError('Degenerate face or camera at face center')
        n /= np.linalg.norm(n)
        if n @ (fc - center) < 0:
            n = -n
        values.append(float(n @ (cam - fc) / np.linalg.norm(cam - fc)))
    up = q[[0, 1, 4, 5]].mean(0) - q[[2, 3, 6, 7]].mean(0)
    ray = cam - center
    if min(np.linalg.norm(up), np.linalg.norm(ray)) < 1e-12:
        raise ValueError('Degenerate height or camera at cuboid center')
    elevation = float(np.degrees(np.arcsin(np.clip(
        up @ ray / (np.linalg.norm(up) * np.linalg.norm(ray)), -1, 1))))
    return np.asarray(values), elevation


def summarize(rows):
    valid = [r for r in rows if r.get('available')]
    paired = [r for r in valid if r['legacy_identity'] is not None]
    errors = [r['stored_cos_error'] for r in valid if r['stored_cos_error'] is not None]
    return dict(total=len(rows), available=len(valid), unavailable=len(rows)-len(valid),
        front_positive=sum(r['cosines'][0] > 0 for r in valid),
        front_is_max=sum(r['front_is_max'] for r in valid),
        front_not_max_examples=[r['id'] for r in valid if not r['front_is_max']][:10],
        stored_cos_count=len(errors), stored_cos_max_error=max(errors) if errors else None,
        stored_cos_matches_4dp=sum(e <= 5.1e-5 for e in errors),
        legacy_complete=len(paired), legacy_identity=sum(r['legacy_identity'] for r in paired),
        legacy_disagrees_but_front_is_max=sum(not r['legacy_identity'] and r['front_is_max'] for r in paired),
        supervised_raw_to_training_max_px=max((r['raw_to_training_max_px'] for r in valid
            if r['raw_to_training_max_px'] is not None), default=None),
        elevation_quantiles=np.quantile([r['elevation_deg'] for r in valid], [0, .1, .5, .9, 1]).tolist() if valid else [])


def main():
    manifest = C.ROOT / 'data/pallet/results/pallet_line_pose_v1/SOURCE_MANIFEST.json'
    old = A.RAW / 'SOURCE_ROWS.json'
    paths = [Path(__file__), Path(__file__).with_name('test_renderer_front_visibility_audit.py'),
             Path(A.__file__), manifest, old, A.DOC / 'RESULTS.json']
    C.freeze(DOC / 'PROTOCOL.json', dict(
        status='SOURCE_ONLY_READ_ONLY_AUDIT', population='All60000source records, no sampling',
        question='Does stored CF face0 maximize outward-normal camera visibility, including legacy-area-rule disagreements?',
        tolerance='front_is_max: max(other)-front<=1e-6; rounded4dp cosine error<=5.1e-5; diagnostic only',
        elevation_bins_deg=[-90, 0, 15, 30, 45, 60, 90],
        source_GT_for_provenance_only=True, real_GT_opened=False,
        new_labels=0, new_tags=0, prediction_changes=0, filter_changes=0,
        warning='Even perfect agreement is evidence of a renderer convention, not proof of author intent, a real-image solver, or a causal explanation of model errors.',
        sources=[C.bound(p) for p in paths]))
    records = C.read(manifest)['records']
    legacy = {r['id']:r['observation'] for r in C.read(old)}
    assert len(records) == len(legacy) == 60000
    rows = []
    with ExitStack() as stack:
        archives = {}
        for i, r in enumerate(records):
            loc = r['renderer_annotation_locator_provenance_only']
            try:
                if '::' in loc:
                    archive, member = loc.split('::', 1)
                    if archive not in archives:
                        archives[archive] = stack.enter_context(zipfile.ZipFile(archive))
                    payload = archives[archive].read(member)
                else:
                    payload = (C.ROOT / loc).read_bytes()
                d = json.loads(payload)
                assert len(d['objects']) == 1
                obj = d['objects'][0]
                cos, el = visibility(obj['cuboid'], d['camera_data']['location_worldframe'])
                p = np.asarray(obj['projected_cuboid'], float)
                t = np.asarray(r['targets'][0]['keypoints_normalized'], float)[:8]
                xy = t[:, :2] * np.asarray(r['prepared_shape_hw'][::-1])
                valid = (t[:, 2] != 0) & np.isfinite(p).all(-1)
                delta = np.linalg.norm(xy - (p + r['reflect_pad_px']), axis=-1)
                obs = legacy[r['id']]
                row = dict(id=r['id'], source=r['source'], partition=r['partition'], available=True,
                    locator=loc, raw_annotation_sha256=hashlib.sha256(payload).hexdigest(),
                    convention=obj.get('keypoint_convention'), cosines=cos.tolist(), elevation_deg=el,
                    stored_cos_error=abs(cos[0]-obj['front_visibility_cos']) if obj.get('front_visibility_cos') is not None else None,
                    front_is_max=bool(cos[0] >= cos.max()-1e-6),
                    legacy_identity=obs['legacy_identity'] if obs.get('rule_available') else None,
                    raw_to_training_max_px=float(delta[valid].max()) if valid.any() else None)
            except (OSError, KeyError, ValueError, AssertionError) as e:
                row = dict(id=r['id'], source=r['source'], partition=r['partition'], available=False,
                           locator=loc, error=f'{type(e).__name__}: {e}')
            rows.append(row)
            if (i+1) % 10000 == 0:
                print('RENDERER_AUDIT', i+1, '/60000', flush=True)
    C.freeze(RAW / 'ROWS.json', rows)
    buckets = defaultdict(list)
    for row in rows:
        buckets[row['source']].append(row)
    elevations = defaultdict(list)
    for row in rows:
        if row['available']:
            lo = int(np.searchsorted([0, 15, 30, 45, 60], row['elevation_deg'], side='right'))
            name = ['[-90,0)', '[0,15)', '[15,30)', '[30,45)', '[45,60)', '[60,90]'][lo]
            elevations[f'{row["source"]}/{name}'].append(row)
    result = dict(status='DIAGNOSTIC_ONLY_NOT_A_CORRECTION', all=summarize(rows),
        by_source={k:summarize(v) for k,v in buckets.items()},
        by_source_elevation={k:summarize(v) for k,v in elevations.items()},
        evidence=[C.bound(DOC/'PROTOCOL.json'), C.bound(RAW/'ROWS.json')],
        goal_complete=False, real_GT_used=False, predictions_changed=False)
    C.freeze(DOC / 'RESULTS.json', result)
    print(json.dumps({k:result[k] for k in ['all','by_source']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
