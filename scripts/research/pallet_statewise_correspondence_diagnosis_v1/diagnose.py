"""Postseal residual/state/owner and good-input diagnostics, no fitting.

Pixel residuals here use projections of the EXISTING reconstructed reference,
not independently clicked image corners. Physical vertex registration is model
arithmetic and never asserted to validate the image semantic owner. C2 phase
is selected once per whole initial point set, strictly for postseal analysis;
both phase residuals are retained and no pointwise nearest-GT reassignment is
performed. Prediction files are not modified by this module.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import json
from pathlib import Path
import time

import numpy as np

from . import analysis as A


VERTEX_ATOL_M = 1e-8  # Existing e_inputs.physical_vertex_bijection tolerance.
OWNER_STATUS = 'SEMANTIC_OWNER_UNVERIFIED'
STATES = ('B', 'I', 'S', 'O', 'T', 'U')


def cuboid(extents):
    x, y, z = np.asarray(extents, float)/2
    return np.array([[-x, -y, -z], [x, -y, -z], [x, y, -z], [-x, y, -z],
                     [-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z]])


def owners(native_xyz, Q, physical_dimensions):
    """Exact model-coordinate bijection; no image/GT nearest-point matching."""
    native = np.asarray(native_xyz, float)
    if native.shape != (8, 3) or not np.isfinite(native).all():
        raise ValueError('Eight known native model vertices required')
    physical = native@A.rotation(Q)
    known = cuboid(physical_dimensions)
    distance = np.linalg.norm(physical[:, None]-known[None], axis=2)
    match = distance <= VERTEX_ATOL_M
    if not np.all(match.sum(0) == 1) or not np.all(match.sum(1) == 1):
        raise ValueError('No unique original physical-vertex model bijection')
    return np.argmax(match, axis=1).tolist()


def pose_owners(pose, dimensions):
    Q = A.rotation(pose['R_cf']).T@A.rotation(pose['R_physical'])
    return owners(cuboid(pose['cf_extents']), Q, dimensions)


def reference_projections(reference, K):
    """Return both allowed WHOLE-object phases; retain unresolved rotations."""
    if reference['R'] is None:
        return [None, None]
    if reference['order'] != 2:
        raise ValueError('No new C4 equivalence may enter this diagnostic')
    xyz = cuboid(reference['xyz'])
    R, t, K = A.rotation(reference['R']), A.point(reference['t']), np.asarray(K, float)
    if K.shape != (3, 3) or not np.isfinite(K).all():
        raise ValueError('Explicit original camera matrix required')
    result = []
    for S in (np.eye(3), np.diag([-1., 1., -1.])):
        camera = xyz@(R@S).T+t
        homogeneous = camera@K.T
        if np.any(camera[:, 2] <= 0) or np.any(homogeneous[:, 2] <= 0):
            raise ValueError('Reference corners must be in front of original camera')
        result.append(homogeneous[:, :2]/homogeneous[:, 2, None])
    return result


def _xy(value):
    try:
        x = np.asarray(value, float)
    except (ValueError, TypeError):
        return None
    if x.shape != (2,) or not np.isfinite(x).all() or np.all(x == -1):
        return None
    return x


def whole_C2_phase(points, owner, projections):
    """One global C2 phase from the same initial points, never per-point min."""
    values = []
    for projected in projections:
        residuals = [np.linalg.norm(_xy(q)-projected[owner[k]]) for k, q in enumerate(points[:8])
                     if _xy(q) is not None] if projected is not None else []
        values.append(float(np.mean(residuals)) if residuals else None)
    phase = min(range(2), key=lambda k: (values[k], k)) if all(v is not None for v in values) else None
    return phase, values


def point_record(fid, stage, native_id, xy, owner, state, source, kind, active,
                 projections, phase, *, anchor):
    point = _xy(xy)
    errors = [None if point is None or p is None else float(np.linalg.norm(point-p[owner])) for p in projections]
    return dict(id=fid, stage=stage, native_corner=native_id, model_physical_vertex=owner,
                state=state, source=source, observation_type=kind, active_point_factor=active,
                phase0_error_px=errors[0], phase180_error_px=errors[1],
                whole_object_C2_phase=phase, C2_phase_anchor=anchor,
                error_px=None if phase is None else errors[phase],
                finite_source_point=point is not None, semantic_owner_status=OWNER_STATUS,
                reference_kind='EXISTING_GEOMETRY_RECONSTRUCTION_PROJECTION_NOT_MANUAL_2D_GT')


def movement_record(fid, stage, native_id, initial_xy, corrected_xy, owner, state,
                    projections, phase, *, state_source):
    p, q = _xy(initial_xy), _xy(corrected_xy)
    target = None if phase is None else projections[phase]
    valid_motion = p is not None and q is not None
    valid = valid_motion and target is not None and owner is not None
    before = float(np.linalg.norm(p-target[owner])) if valid else None
    after = float(np.linalg.norm(q-target[owner])) if valid else None
    return dict(id=fid, stage=stage, native_corner=native_id, model_physical_vertex=owner,
                state=state, state_source=state_source, whole_object_C2_phase=phase,
                input_error_px=before, output_error_px=after,
                movement_px=float(np.linalg.norm(q-p)) if valid_motion else None,
                good_input_lt5=None if before is None else before < 5.,
                good_to_bad_gt10=None if before is None else before < 5. and after > 10.,
                large_gt20_to_le10=None if before is None else before > 20. and after <= 10.,
                image_semantic_correspondence_verified=False,
                semantic_owner_status=OWNER_STATUS,
                residuals_are_geometric_reference_not_direct_pixel_GT=True)


def movement_summary(rows):
    good = [r for r in rows if r['good_input_lt5'] is True]
    hard = [r for r in rows if r['input_error_px'] is not None and r['input_error_px'] > 20.]
    return dict(rows=len(rows), available_n=sum(r['input_error_px'] is not None for r in rows),
                good_input_lt5_n=len(good), damaged_good_to_gt10_n=sum(r['good_to_bad_gt10'] is True for r in good),
                large_input_gt20_n=len(hard), recovered_large_to_le10_n=sum(r['large_gt20_to_le10'] is True for r in hard),
                all_movement_px=A.distribution(r['movement_px'] for r in rows),
                good_input_movement_px=A.distribution(r['movement_px'] for r in good),
                good_input_output_error_px=A.distribution(r['output_error_px'] for r in good),
                all_rows_kept=True, independent_2D_GT=False)


def analyze_CAL(line, packet, reference, frame):
    fid = line['id']
    if packet['id'] != fid or frame['id'] != fid:
        raise ValueError('CAL lineage/input/typed packet IDs differ')
    if not np.array_equal(frame['xyz'], reference['xyz']):
        raise ValueError('Registered input dimensions and original reference dimensions differ')
    projections = reference_projections(reference, packet['solver_input']['K'])
    original = line['A_original_N3']
    geometry = original['geometry']
    own0 = owners(geometry['native_xyz'], geometry['physical_registration_Q'], frame['xyz'])
    states0 = line['B_original_pre_e']['states']
    points0 = original['points']
    phase, phase_costs = whole_C2_phase(points0, own0, projections)
    family = packet['registered_family']
    ownE = owners(family['native_xyz'], family['physical_registration_Q'], frame['xyz'])
    post = line['D_post_e']['observation']
    ownpost = owners(post['initial_geometry']['native_xyz'],
                     post['initial_geometry']['physical_registration_Q'], frame['xyz'])
    records = []
    for stage, points, own, states in (
        ('N3_INITIAL', points0, own0, states0),
        ('POST_E_REINJECTED_N3', line['D_post_e']['native_points'], ownpost, post['states'])):
        for k in range(8):
            records.append(point_record(fid, stage, k, points[k], own[k], states[k],
                'INHERITED_PREDICTOR', 'predictor_diagnostic_not_visible_measurement', False,
                projections, phase, anchor='N3_INITIAL_whole8'))
    old_complete = set(packet['activation']['complete_ids'])
    for stage, measurements, own, states, active in (
        ('ORIGINAL_E_COMPLETE', packet['solver_input']['visible'], ownE, packet['solver_input']['states'], old_complete),
        ('POST_E_COMPLETE', post['frozen_visible_measurements'], ownpost, post['states'],
         {r['native_corner'] for r in post['frozen_visible_measurements']})):
        for m in measurements:
            k = m['native_corner']
            if k not in active:
                continue  # Partial terms never become a 2D point-error claim.
            if states[k] not in ('B', 'I'):
                raise ValueError('S/O/T/U cannot enter active image-point diagnosis')
            kind = 'inherited_predictor_point' if m.get('source') == 'VISIBLE_BACKBONE_FALLBACK' else 'image_point_source_unverified_semantic_owner'
            records.append(point_record(fid, stage, k, m['xy'], own[k], states[k], m.get('source'),
                                         kind, True, projections, phase, anchor='N3_INITIAL_whole8'))
    partials = []
    for b in packet['solver_input']['boundary_observations']:
        k = b['native_corner']
        for number, observed in enumerate(b['lines']):
            normal = np.asarray(observed['normal_unit_xy'], float)
            values = [None if p is None else float(normal@p[ownE[k]]+observed['offset_px']) for p in projections]
            partials.append(dict(id=fid, native_corner=k, model_physical_vertex=ownE[k], state=b['state'],
                line_number=number, phase0_signed_normal_residual_px=values[0],
                phase180_signed_normal_residual_px=values[1], whole_object_C2_phase=phase,
                signed_normal_residual_px=None if phase is None else values[phase],
                is_complete_2D_point=False, semantic_owner_status=OWNER_STATUS))
    inverse0 = {physical: k for k, physical in enumerate(own0)}
    changes = []
    for m in packet['solver_input']['visible']:
        k = m['native_corner']
        if k not in old_complete:
            continue
        source_k = inverse0[ownE[k]]
        changes.append(movement_record(fid, 'N3_TO_ORIGINAL_E_ACTIVE_COMPLETE', k,
            points0[source_k], m['xy'], ownE[k], states0[source_k], projections, phase,
            state_source='ORIGINAL_N3_STATE_ON_SAME_REGISTERED_PHYSICAL_VERTEX'))
    bbox = np.asarray(frame['arms']['RAW_YOLO'].get('box_xyxy', frame.get('candidate_metadata', {}).get('box_xyxy', [])), float)
    # A cache without its original predicted bbox cannot establish normalized support.
    if bbox.shape != (4,):
        candidate = frame.get('candidate_metadata', {})
        bbox = np.asarray(candidate.get('box_xyxy', []), float)
    diagonal = float(np.linalg.norm(bbox[2:]-bbox[:2])) if bbox.shape == (4,) and np.isfinite(bbox).all() else None
    for record in records:
        record['error_over_original_R0_bbox_diagonal'] = (record['error_px']/diagonal
            if record['error_px'] is not None and diagonal is not None and diagonal > 0 else None)
    return dict(points=records, partials=partials, movements=changes,
                audit=dict(id=fid, whole_object_C2_phase=phase, initial_phase_mean_errors_px=phase_costs,
                           original_N3_model_owners=own0, original_E_model_owners=ownE, post_E_model_owners=ownpost,
                           native_channel_model_owner_changed_n=sum(a != b for a, b in zip(own0, ownE)),
                           state_transitions=dict(Counter(r['original_state']+'->'+r['post_e_same_owner_state'] for r in line['audit'])),
                           typed_partial_to_full_n=sum(r['e_owner_typed_to_post_full'] for r in line['audit']),
                           legacy_partial_to_full_n=sum(r['e_owner_legacy_to_post_full'] for r in line['audit']),
                           original_predicted_bbox_diagonal_px=diagonal, semantic_owner_status=OWNER_STATUS))


def posefix_movements(line, frame, pf_row, reference):
    raw = frame['arms']['RAW_YOLO']
    if raw['coarse_pose'].get('available') is False:
        return [movement_record(line['id'], 'R0_TO_POSEFIX_RAW_FIXED_INPUT_OWNER', k,
            raw['native_points'][k], pf_row['native_points'][k], None, 'U', [None, None], None,
            state_source='UNAVAILABLE_R0_MODEL_REGISTRATION_NO_STATE_TRANSPORT') for k in range(8)]
    projections = reference_projections(reference, frame['K'])
    if not np.array_equal(frame['xyz'], reference['xyz']):
        raise ValueError('Known dimensions/reference mismatch; no fitted registration')
    own = pose_owners(raw['coarse_pose'], frame['xyz'])
    phase, _ = whole_C2_phase(raw['native_points'], own, projections)
    n3geo = line['A_original_N3']['geometry']
    n3own = owners(n3geo['native_xyz'], n3geo['physical_registration_Q'], frame['xyz'])
    inverse = {p: k for k, p in enumerate(n3own)}
    contextual_states = line['B_original_pre_e']['states']
    points = pf_row['native_points']
    return [movement_record(line['id'], 'R0_TO_POSEFIX_RAW_FIXED_INPUT_OWNER', k,
                raw['native_points'][k], points[k], own[k], contextual_states[inverse[own[k]]],
                projections, phase, state_source='MODEL_TRANSPORTED_ORIGINAL_N3_CONTEXT_NOT_PF_OWN_VISIBILITY_GT')
            for k in range(8)]


def _verified_lineage(folder, ids):
    folder = Path(folder)
    seal = A.read(folder/'EXTRACTION_SEAL.json')
    if seal['ordered_ids'] != ids or seal['rows'] != 238:
        raise ValueError('Full ordered238 lineage extraction required')
    result = {}
    for name in ('TYPED_PACKETS.jsonl.gz', 'OBSERVATION_LINEAGE.jsonl.gz'):
        p = folder/name
        if A.sha(p) != seal['files'][name]['sha256']:
            raise ValueError('Original typed lineage seal changed')
        selected = {}
        count = 0
        with gzip.open(p, 'rt') as stream:
            for index, text in enumerate(stream):
                r = json.loads(text)
                if index >= len(ids) or r['id'] != ids[index]:
                    raise ValueError('Lineage journal order changed')
                count += 1
                if name == 'TYPED_PACKETS.jsonl.gz':
                    if r['partition'] == 'CAL':
                        selected[r['id']] = dict(id=r['id'], registered_family=r['registered_family'],
                            activation=r['activation'], solver_input={k:r['solver_input'][k]
                            for k in ('K', 'states', 'visible', 'boundary_observations')})
                elif r['partition'] == 'CAL':
                    selected[r['id']] = r
                else:
                    # PF movement needs only original model geometry and contextual states.
                    selected[r['id']] = dict(id=r['id'], A_original_N3=dict(geometry=r['A_original_N3']['geometry']),
                        B_original_pre_e=dict(states=r['B_original_pre_e']['states']))
        if count != len(ids):
            raise ValueError('All238 lineage rows must remain in source coverage')
        result[name] = selected
    return result, seal


def run(*, folder, source_root, single_folder, lineage_folder, score_folder, reference_manifest,
        expected_prediction_seal_sha256, expected_candidate_seal_sha256, candidate_file,
        expected_reference_manifest_sha256, output, root_scoring_GO=False):
    if root_scoring_GO is not True:
        raise PermissionError('Root postseal diagnostic GO required before any IO')
    start, cpu = time.monotonic(), time.process_time()
    output, score_folder, single_folder = map(Path, (output, score_folder, single_folder))
    if output.exists():
        raise FileExistsError('Preserve every diagnostic attempt')
    new, _, _ = A.prediction_gate(folder, expected_prediction_seal_sha256, expected_candidate_seal_sha256, candidate_file)
    ids = [r['id'] for r in new]
    seal = A.read(score_folder/'SCORE_SEAL.json')
    if (seal['prediction_seal_sha256'] != expected_prediction_seal_sha256
            or seal['candidate_seal_sha256'] != expected_candidate_seal_sha256
            or A.sha(score_folder/'SCORED_ROWS.json') != seal['artifacts']['SCORED_ROWS.json']
            or A.sha(reference_manifest) != expected_reference_manifest_sha256):
        raise ValueError('All postseal metric/reference bindings required')
    scored = A.read(score_folder/'SCORED_ROWS.json')['rows']
    if [r['id'] for r in scored] != ids:
        raise ValueError('Diagnostic and scored IDs differ')
    lineage, extraction = _verified_lineage(lineage_folder, ids)
    line = lineage['OBSERVATION_LINEAGE.jsonl.gz']
    packets = lineage['TYPED_PACKETS.jsonl.gz']
    input_path = single_folder/'inputs_v1/INPUTS.json'
    iseal = A.read(single_folder/'inputs_v1/INPUT_SEAL.json')
    if A.sha(input_path) != iseal['inputs_sha256']:
        raise ValueError('Original component input source changed')
    frames = A.read(input_path)['frames']
    if [r['id'] for r in frames] != ids:
        raise ValueError('Point input IDs differ')
    refs, refbindings, annotations = A.fixed_references(source_root, reference_manifest, new)
    pf = {}
    for relative in (A.PF95, A.PF143):
        for r in A.journal(Path(source_root)/relative/'PREDICTIONS.jsonl.gz'):
            if relative == A.PF95 and r['method'] != 'POSEFIX_RAW':
                continue
            if r['id'] in pf:
                raise ValueError('Duplicate saved PoseFix row')
            pf[r['id']] = r
    if set(pf) != set(ids):
        raise ValueError('All238 saved PoseFix point arrays required')
    points, partials, movements, audits, pfmoves = [], [], [], [], []
    for r, f in zip(new, frames):
        fid = r['id']
        reference = refs[r['partition']][fid]
        if r['partition'] == 'CAL':
            d = analyze_CAL(line[fid], packets[fid], reference, f)
            points += d['points']; partials += d['partials']; movements += d['movements']; audits.append(d['audit'])
        if r['partition'] != 'CAL':
            pfmoves += posefix_movements(line[fid], f, pf[fid], reference)
    point_summary = {stage: {state: A.distribution(r['error_px'] for r in points
        if r['stage'] == stage and r['state'] == state) for state in STATES}
        for stage in sorted({r['stage'] for r in points})}
    radius_comparison = {}
    for stage in ('N3_INITIAL', 'POST_E_REINJECTED_N3'):
        subset = [r for r in points if r['stage'] == stage and r['state'] in ('B', 'I')]
        xs = [r['error_over_original_R0_bbox_diagonal'] for r in subset
              if r['error_over_original_R0_bbox_diagonal'] is not None]
        radius_comparison[stage] = dict(visible_BI_predictor_rows=len(subset), available_n=len(xs),
            below_5pct=sum(x < .05 for x in xs), inside_5_to_15pct=sum(.05 <= x <= .15 for x in xs),
            above_15pct=sum(x > .15 for x in xs),
            interpretation='Historical manual+noise location-radius comparison only; not synthetic-only training-support validation',
            semantic_owner_support_verified=False)
    pf_state = {state: movement_summary([r for r in pfmoves if r['state'] == state]) for state in STATES}
    scoremap = {r['id']: r for r in scored}
    patterns = {}
    for r in scored:
        if r['partition'] == 'CAL':
            continue
        a, b = [r['metrics'][m]['success_le5'] for m in ('POSEFIX_RAW', 'E0')]
        category = 'UNRESOLVED' if a is None or b is None else 'BOTH_SUCCESS' if a and b else 'PF_ONLY' if a else 'E_ONLY' if b else 'BOTH_FAILURE'
        patterns.setdefault(category, []).append(r['id'])
    pose_good = {}
    primary = [r for r in scored if r['partition'] != 'CAL']
    for arm in A.DIAGNOSTICS:
        good = [r for r in primary if r['metrics']['E0']['success_le5'] is True]
        pose_good[arm] = dict(E0_good_input_n=len(good), damaged_ids=[r['id'] for r in good
            if r['metrics'][arm]['success_le5'] is not True],
            delta_translation_cm=A.distribution(None if r['metrics'][arm]['translation_cm'] is None else
                r['metrics'][arm]['translation_cm']-r['metrics']['E0']['translation_cm'] for r in good),
            delta_rotation_deg=A.distribution(None if r['metrics'][arm]['rotation_deg'] is None else
                r['metrics'][arm]['rotation_deg']-r['metrics']['E0']['rotation_deg'] for r in good))
    output.mkdir(parents=True)
    def csvout(name, values, fields):
        with (output/name).open('x', newline='') as stream:
            w = csv.DictWriter(stream, fieldnames=fields); w.writeheader(); w.writerows(values)
    csvout('CAL_POINT_RESIDUALS.csv', points, list(points[0]))
    if partials:
        csvout('CAL_PARTIAL_LINE_RESIDUALS.csv', partials, list(partials[0]))
    csvout('CAL_GOOD_INPUT_MOVEMENTS.csv', movements, list(movements[0]) if movements else
           ['id', 'stage', 'input_error_px', 'output_error_px', 'movement_px'])
    csvout('PRIMARY_POSEFIX_GOOD_INPUT_MOVEMENTS.csv', pfmoves, list(pfmoves[0]))
    A.write(output/'CAL_OWNER_AND_STATE_AUDIT.json', dict(rows=audits, CAL_images=46,
        semantic_owner_status=OWNER_STATUS, manual_semantic_owner_GT_available=False))
    A.write(output/'DIAGNOSTICS.json', dict(schema='STATEWISE_POSTSEAL_GOOD_INPUT_AND_CAL_RESIDUAL_V1',
        CAL_images=46, PRIMARY_images=192, CAL_point_stage_state_errors_px=point_summary,
        CAL_historical_noise_radius_descriptive_comparison=radius_comparison,
        CAL_original_e_complete_good_input=movement_summary(movements),
        PRIMARY_PoseFix_input_R0_good_point=movement_summary(pfmoves),
        PRIMARY_PoseFix_context_state_bins=pf_state,
        PRIMARY_PoseFix_vs_E0_patterns={k: dict(n=len(v), ids=v,
            sessions=dict(Counter(scoremap[i]['session'] for i in v))) for k, v in patterns.items()},
        PRIMARY_PoseFix_pattern_movement={k: movement_summary([r for r in pfmoves if r['id'] in set(v)])
                                       for k, v in patterns.items()},
        PRIMARY_E0_good_pose_diagnostic=pose_good,
        error_support_status='COVERAGE_NOT_VERIFIED', new_learning_proposal_count=0,
        limitations=['Residuals use fixed reconstructed geometry, not independently clicked image corners.',
                     'Model physical vertex bijection does not verify semantic image corner identity.',
                     'PF state bins transport original N3 states through model vertices; not PF own visibility labels.',
                     'A global initial-point C2 phase is privileged postseal analysis, never a selector input.',
                     'B/I roles and visibility states are not human jitter/inversion/swap/miss labels.',
                     'Partial normal residual is one-dimensional; no region center or S projection becomes a 2D measurement.',
                     'Training corruption location radius cannot certify real semantic-owner error coverage.',
                     'All192 are repeated development images, no independent generalization claim.']))
    A.write(output/'DIAGNOSTIC_SEAL.json', dict(rows=238, all_CAL46_and_PRIMARY192_kept=True,
        prediction_seal_sha256=expected_prediction_seal_sha256, candidate_seal_sha256=expected_candidate_seal_sha256,
        score_seal_sha256=A.sha(score_folder/'SCORE_SEAL.json'), extraction_seal_sha256=A.sha(Path(lineage_folder)/'EXTRACTION_SEAL.json'),
        reference_bindings=refbindings, annotation_sha256=annotations,
        source_sha256={'diagnose.py': A.sha(__file__), 'analysis.py': A.sha(A.__file__)},
        new_training_updates=0, new_model_forwards=0, new_pose_solves=0,
        elapsed_seconds=time.monotonic()-start, process_cpu_seconds=time.process_time()-cpu,
        artifacts={p.name: A.sha(p) for p in output.iterdir() if p.is_file()}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('folder', 'source-root', 'single-folder', 'lineage-folder', 'score-folder',
                'reference-manifest', 'candidate-file', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    for key in ('prediction-seal-sha256', 'candidate-seal-sha256', 'reference-manifest-sha256'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--root-scoring-go', action='store_true')
    a = p.parse_args()
    run(folder=a.folder, source_root=a.source_root, single_folder=a.single_folder,
        lineage_folder=a.lineage_folder, score_folder=a.score_folder, reference_manifest=a.reference_manifest,
        candidate_file=a.candidate_file, output=a.output, expected_prediction_seal_sha256=a.prediction_seal_sha256,
        expected_candidate_seal_sha256=a.candidate_seal_sha256,
        expected_reference_manifest_sha256=a.reference_manifest_sha256, root_scoring_GO=a.root_scoring_go)


if __name__ == '__main__':
    main()
