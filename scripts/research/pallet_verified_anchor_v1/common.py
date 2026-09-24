from pathlib import Path
import ast
import hashlib
import json
import math
import os
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]
NAME = 'pallet_verified_anchor_v1'
DOC = ROOT / '_docs/experiments' / NAME
RAW = ROOT / 'data/pallet/results' / NAME
OUT = ROOT / 'outputs' / NAME
SPLIT = ROOT / '_docs/experiments/pallet_existing_data_transfer_v1/SPLIT_LOCK.json'
STATUSES = ('DIRECT_VISIBLE', 'VIRTUAL_INFERABLE', 'SELF_OCCLUDED',
            'EXTERNAL_OCCLUDED', 'OUT_OF_FRAME', 'UNCERTAIN')
SEVERITIES = ('CLEAN', 'MODERATE_OCCLUSION', 'SEVERE_OCCLUSION')

def now():
    return datetime.now(timezone.utc).isoformat()

def read(path):
    return json.loads(path.read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save_new(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        f.write(obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

def atomic(path, obj):
    # Only mutable human labeling sidecars may be replaced.
    assert path.parent == RAW and path.name in ('LABELS.json', 'ANNOTATION_PROGRESS.json')
    temp = path.with_suffix('.tmp')
    with temp.open('w') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)

def contract():
    """Extract repository definitions without importing model/annotation pipelines."""
    draw = ROOT / 'scripts/annotate/annotate_draw.py'
    geom = ROOT / 'scripts/annotate/annotate_pnp.py'
    constants = {}
    for path in (draw, geom):
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in ('KP_NAMES', 'CUBOID_EDGES', 'PALLET_DIMS'):
                        constants[target.id] = ast.literal_eval(node.value)
    function = next(n for n in ast.parse(geom.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'make_pallet_keypoints_3d_diagram')
    import numpy as np
    namespace = {'np': np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(geom), 'exec'), namespace)
    xyz = namespace[function.name](*constants['PALLET_DIMS'])
    return dict(names=constants['KP_NAMES'][:8], edges=constants['CUBOID_EDGES'],
                dimensions_WDH_m=constants['PALLET_DIMS'], xyz_m=xyz[:8].tolist(),
                sources=[dict(path=str(p.relative_to(ROOT)), sha256=sha(p)) for p in (draw, geom)],
                convention='camera-facing native; near P0..3, far P4..7; local X right,Y down,Z forward; W/D parity may swap with view')

def valid_corner(c, hw):
    status, xy = c.get('status'), c.get('xy')
    if status not in STATUSES:
        return False
    if xy is None:
        return status != 'DIRECT_VISIBLE'
    if status not in ('DIRECT_VISIBLE', 'VIRTUAL_INFERABLE'):
        return False
    h, w = hw
    return (isinstance(xy, list) and len(xy) == 2 and
            all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in xy)
            and 0 <= xy[0] < w and 0 <= xy[1] < h)

def progress(labels, selection):
    sizes = {r['frame_id']: r['hw'] for r in selection['frames']}
    complete = sum(all(valid_corner(c, sizes[r['frame_id']]) for c in r['corners']) for r in labels['frames'])
    count = sum(valid_corner(c, sizes[r['frame_id']]) for r in labels['frames'] for c in r['corners'])
    return dict(status='WAITING_FOR_HUMAN_LABELING' if complete < len(sizes) else 'FIRST_PASS_READY_FOR_LOCK',
                completed_images=complete, total_images=len(sizes), completed_statuses=count,
                total_statuses=len(sizes)*8, updated_utc=now())
