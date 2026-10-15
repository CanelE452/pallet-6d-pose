"""Three predeclared CAL guide transports; model metadata only, no method change."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
from itertools import islice
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / '_docs/experiments/pallet_visibility_curriculum_20261013_v1/nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1'
OBS = BASE / 'geometry_guided_edge_observations_v4_29b'
TASKS = BASE / 'coarse_prior_containment_eval_v4_25'
OUTPUT = BASE / 'CAL_native_guide_transport_diagnosis_v4_29'
CASES = (('wood_184309:000925', 4, 0), ('wood_184309:000804', 1, 2),
         ('eval_cad:1778653017736058368', 7, 3))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    path = Path(path)
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def rows(path):
    with gzip.open(path, 'rt') as stream:
        for line in stream:
            yield json.loads(line)


def declare():
    if OUTPUT.exists():
        raise FileExistsError('Preserve every transport diagnostic attempt')
    index = read(OBS / 'CAL_ROW_INDEX.json')
    entries = {r['id']: r for r in index['entries']}
    paths = [OBS / 'CAL_ROW_INDEX.json', OBS / 'CAL_OBSERVATION_SEAL.json',
             TASKS / 'PROTOCOL.json', TASKS / 'INPUT_SEAL.json']
    for ident, _, _ in CASES:
        path = OBS / 'row_spool' / entries[ident]['path']
        if sha(path) != entries[ident]['sha256']:
            raise ValueError('Exact existing CAL spool required')
        paths.append(path)
    OUTPUT.mkdir()
    plan = dict(schema='CAL_NATIVE_GUIDE_TRANSPORT_PREDECLARATION_V29',
        UTC=datetime.now(timezone.utc).isoformat(), cases=[list(c) for c in CASES],
        case_selection='Exactly the three already declared CAL top-error examples; no new ranking or selection.',
        source_sha256={str(Path(__file__).relative_to(ROOT)): sha(__file__)},
        input_sha256={str(p.relative_to(ROOT)): sha(p) for p in paths},
        task_payload_sha256=read(TASKS / 'INPUT_SEAL.json')['input_sha256'],
        calculations=['Recover R_source=R_guide Q_original Q_source.T.',
            'Fit unit native named-edge basis T over all twelve corresponding edges.',
            'Compare source-camera and transported-original model edge vectors and 2D projected line directions.',
            'Verify common physical R/center and unordered camera cuboid, without assuming per-corner identity.'],
        new_RGB_NN_labels_reference_score_posefit_or_method_changes=0,
        S_U_observed_coordinate_queries=0, model_only_all_twelve_edge_projection=True,
        full238_observation_handoff_has_priority=True)
    write_new(OUTPUT / 'PREDECLARATION.json', plan)
    print(dict(plan_sha256=sha(OUTPUT / 'PREDECLARATION.json')), flush=True)


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def angle(a, b, unoriented=False):
    dot = float(np.clip(unit(a) @ unit(b), -1., 1.))
    return float(np.degrees(np.arccos(abs(dot) if unoriented else dot)))


def project(xyz, R, t, K):
    camera = xyz @ R.T + t
    homogeneous = camera @ K.T
    if np.any(np.abs(homogeneous[:, 2]) <= 1e-12):
        raise ValueError('Finite model projection required for this diagnostic')
    return homogeneous[:, :2] / homogeneous[:, 2, None]


def compare_models(original, source, guide):
    xo, xs = (np.asarray(f['native_xyz'], float) for f in (original, source))
    Qo, Qs = (np.asarray(f['physical_registration_Q'], float) for f in (original, source))
    Rg = np.asarray(guide['R_cf'], float)
    center = np.asarray(guide['centroid'], float)
    Rs = Rg @ Qo @ Qs.T
    tg, ts = center - Rg @ xo.mean(0), center - Rs @ xs.mean(0)
    edges = [(i, j) for i in range(8) for j in range(i + 1, 8)
             if np.count_nonzero(np.abs(xo[j] - xo[i]) > 1e-9) == 1]
    if len(edges) != 12 or any(np.count_nonzero(np.abs(xs[j] - xs[i]) > 1e-9) != 1 for i, j in edges):
        raise ValueError('Same twelve named native cuboid edges required')
    do = unit(np.asarray([xo[j] - xo[i] for i, j in edges]))
    ds = unit(np.asarray([xs[j] - xs[i] for i, j in edges]))
    T = np.linalg.lstsq(do, ds, rcond=None)[0].T
    K = np.asarray(guide['K'], float)
    po, ps = project(xo, Rg, tg, K), project(xs, Rs, ts, K)
    co, cs = xo @ Rg.T + tg, xs @ Rs.T + ts
    distances = np.linalg.norm(co[:, None, :] - cs[None, :, :], axis=2)
    per_edge = [dict(native_edge=[i, j], original_native_unit=do[n].tolist(),
        source_native_unit=ds[n].tolist(),
        camera_directed_angle_deg=angle(Rg @ do[n], Rs @ ds[n]),
        camera_unoriented_angle_deg=angle(Rg @ do[n], Rs @ ds[n], True),
        projected_unoriented_angle_deg=angle(po[j] - po[i], ps[j] - ps[i], True))
        for n, (i, j) in enumerate(edges)]
    return dict(recovered_source_R_cf=Rs.tolist(), guide_original_R_cf=Rg.tolist(),
        source_Q=Qs.tolist(), original_Q=Qo.tolist(),
        unit_native_transport_T=T.tolist(),
        unit_native_transport_identity_maxabs=float(np.max(np.abs(T - np.eye(3)))),
        unit_native_transport_residual_maxabs=float(np.max(np.abs(ds - do @ T.T))),
        physical_registration_transport=(Qs @ Qo.T).tolist(),
        same_physical_rotation_maxabs=float(np.max(np.abs(Rs @ Qs - Rg @ Qo))),
        model_center_camera=center.tolist(),
        unordered_camera_cuboid_gap_m=float(max(distances.min(0).max(), distances.min(1).max())),
        same_label_camera_vertex_maxgap_m=float(np.linalg.norm(co - cs, axis=1).max()),
        all_twelve_edges=per_edge)


def run(*, root_metadata_GO=False):
    if root_metadata_GO is not True:
        raise PermissionError('Root CAL metadata-only diagnostic GO required before data IO')
    plan = read(OUTPUT / 'PREDECLARATION.json')
    for mapping in (plan['source_sha256'], plan['input_sha256']):
        for relative, digest in mapping.items():
            if sha(ROOT / relative) != digest:
                raise ValueError('Frozen diagnostic binding changed: ' + relative)
    if sha(TASKS / 'INPUTS.jsonl.gz') != plan['task_payload_sha256']:
        raise ValueError('Exact frozen task payload required')
    requested = {c[0] for c in CASES}
    families = {}
    for task in islice(rows(TASKS / 'INPUTS.jsonl.gz'), 46):
        if task['partition'] != 'CAL':
            raise ValueError('CAL prefix only')
        if task['id'] in requested:
            families[task['id']] = {name: {k: f[k] for k in ('native_xyz', 'physical_registration_Q')}
                                     for name, f in task['families'].items()}
    if set(families) != requested:
        raise ValueError('Exactly three CAL cases required')
    index = {e['id']: e for e in read(OBS / 'CAL_ROW_INDEX.json')['entries']}
    records = []
    for ident, k, mate in CASES:
        entry = index[ident]
        spool = OBS / 'row_spool' / entry['path']
        candidates = list(rows(spool))
        if len(candidates) != 1 or candidates[0]['id'] != ident:
            raise ValueError('Exactly one existing spool row')
        observation = candidates[0]['observation']
        refs = [r for r in observation['selected_chain_refs']
                if r['native_corner'] == k and r['native_neighbor'] == mate]
        if len(refs) != 1:
            raise ValueError('Exact predeclared selected native edge')
        guide_id = refs[0]['guide_reference']['geometry_ref_id']
        source_family = guide_id.split('/', 1)[0]
        result = compare_models(families[ident]['ORIGINAL_WD'], families[ident][source_family],
                                observation['guide_geometries'][guide_id])
        records.append(dict(id=ident, selected_native_edge=[k, mate], selected_guide_id=guide_id,
                            source_family=source_family, **result))
        del observation, candidates
    output = dict(schema='CAL_NATIVE_GUIDE_TRANSPORT_MODEL_METADATA_DIAGNOSIS_V29',
        plan_sha256=sha(OUTPUT / 'PREDECLARATION.json'), records=records,
        new_RGB_labels_numeric_reference_score_NN_or_posefit_calls=0,
        observed_S_U_coordinate_queries=0, existing_runtime_or_model_changes=0,
        interpretation='Physical-pose/center transport and same native-channel direction preservation are different contracts; this diagnostic does not prove which image association is physically correct.')
    write_new(OUTPUT / 'DIAGNOSIS.json', output)
    write_new(OUTPUT / 'DIAGNOSIS_SEAL.json', dict(source_sha256=sha(__file__),
        plan_sha256=sha(OUTPUT / 'PREDECLARATION.json'), diagnosis_sha256=sha(OUTPUT / 'DIAGNOSIS.json')))
    print(dict(cases=len(records), diagnosis_sha256=sha(OUTPUT / 'DIAGNOSIS.json')), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('declare', 'run'))
    parser.add_argument('--root-metadata-GO', action='store_true')
    args = parser.parse_args()
    declare() if args.stage == 'declare' else run(root_metadata_GO=args.root_metadata_GO)
