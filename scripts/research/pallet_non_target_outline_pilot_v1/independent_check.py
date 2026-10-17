"""Independent arithmetic, lineage and fixed-bank selection verification.

This module intentionally imports neither the pilot kernel/replay nor the old
selector. It reads only their sealed neutral inputs and output journals. Its
nearest-segment calculation, contribution sums, stable ordering and hierarchy
are implemented here independently. It does not evaluate any new pose cost,
generate observations/candidates, or import a solver or neural-network runtime.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np

ATOL = 1e-9
STAGES = ('G10', 'G13')
FAMILIES = ('ORIGINAL_WD', 'SWAPPED_WD')
VARIANTS = ('BASELINE_E', 'SUPPORT_AWARE_RERANK', 'MASS_MATCHED_CONTROL')
IDS = ('plastic_night_01:038630', 'wood_day_01:005286',
       'wood_night_01:030698', 'plastic_day_01:020955',
       'plastic_day_01:014704', 'wood_night_01:030898')
DENIED = {'reference', 'reference_success', 'references', 'oracle', 'oracles',
          'metric', 'metrics', 'group', 'translation_cm', 'rotation_deg',
          'success_le5', 'final_reference_success', 'SUPPORT_DECISIONS',
          'representative_success_candidate_id',
          'reference_success_candidates_g13_all',
          'reference_success_candidates_g13_selected_family',
          'family_reference_diagnosis', 'g10_reference_success_candidate_ids'}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with (gzip.open(path, 'rt') if str(path).endswith('.gz') else Path(path).open()) as stream:
        return [json.loads(line) for line in stream if line.strip()]


def equal(a, b, label, maximum):
    """Nested exact metadata; arrays/floats use the declared absolute tolerance."""
    if isinstance(a, dict):
        require(isinstance(b, dict) and set(a) == set(b), label+' keys')
        for key in a:
            equal(a[key], b[key], label+'/'+key, maximum)
    elif isinstance(a, (list, tuple)):
        require(isinstance(b, (list, tuple)) and len(a) == len(b), label+' cardinality')
        for index, (left, right) in enumerate(zip(a, b)):
            equal(left, right, label+'/'+str(index), maximum)
    elif isinstance(a, float) or isinstance(b, float):
        require(isinstance(a, (float, int)) and isinstance(b, (float, int)), label+' numeric type')
        error = abs(float(a)-float(b))
        maximum[0] = max(maximum[0], error)
        require(math.isfinite(error) and error <= ATOL, label+' absolute difference '+str(error))
    else:
        require(type(a) is type(b) and a == b, label+' exact metadata')


def non_neutral(value):
    if isinstance(value, dict):
        return [key for key in value if key in DENIED] + [k for child in value.values() for k in non_neutral(child)]
    if isinstance(value, list):
        return [key for child in value for key in non_neutral(child)]
    return []


def bound(package_path, entry):
    path = Path(entry['path'])
    if not path.is_absolute():
        path = Path(package_path).parent/path
    require(digest(path) == entry['sha256'], 'Neutral input byte binding')
    return path


def input_masks(manifest_path, frames):
    if manifest_path is None:
        return {frame['id']: None for frame in frames}
    entries = rows(manifest_path)
    require([e['id'] for e in entries] == list(IDS), 'Frozen input ordered full denominator')
    masks = {}
    for frame, entry in zip(frames, entries):
        require(entry['image_sha256'] == frame['image_sha256'], 'Same source RGB file')
        require(entry['image_hw'] == frame['image_hw'], 'Same raw canvas')
        if entry.get('admissible_for_censoring') is not True:
            masks[frame['id']] = None
            continue
        path = Path(entry['mask_path'])
        if not path.is_absolute():
            path = Path(manifest_path).parent/path
        require(digest(path) == entry['mask_sha256'], 'Frozen mask file SHA')
        if path.suffix == '.npz':
            with np.load(path, allow_pickle=False) as arrays:
                mask = arrays[entry['mask_key']].copy()
        elif path.suffix == '.npy':
            mask = np.load(path, allow_pickle=False)
        else:
            from PIL import Image
            with Image.open(path) as image:
                mask = np.asarray(image).copy()
        require(mask.shape == tuple(frame['image_hw']), 'Frozen mask raw shape')
        require(np.all((mask == 0) | (mask == 1)), 'Supplier binary already fixed')
        require(hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest() == entry['pixel_bytes_sha256'],
                'Frozen mask raw pixel bytes')
        masks[frame['id']] = mask
    return masks


def lineage(outline, mask):
    if not outline.get('supported'):
        return [], [], []
    polygon = np.asarray(outline['polygon_xy'], dtype=float)
    h, w = outline['image_hw']
    margin = float(outline['configuration']['frame_margin_px'])
    spacing = float(outline['configuration']['observed_sample_spacing_px'])
    segments, parents, sample_parent, sample_xy = [], [], [], []
    for edge, start in enumerate(polygon):
        end = polygon[(edge+1) % len(polygon)]
        vector = end-start
        low, high, outside = 0., 1., False
        # Independent line/canvas interval intersection. Preclip vertices below
        # remain untouched and are the only source of mask membership evidence.
        for axis, maximum in ((0, w-1-margin), (1, h-1-margin)):
            if abs(vector[axis]) < 1e-14:
                outside |= not margin <= start[axis] <= maximum
            else:
                times = ((margin-start[axis])/vector[axis], (maximum-start[axis])/vector[axis])
                low, high = max(low, min(times)), min(high, max(times))
        if outside or low >= high:
            continue
        first, last = start+low*vector, start+high*vector
        sid = len(segments)
        count = max(1, int(math.ceil(float(np.linalg.norm(last-first))/spacing)))
        positions = first+((np.arange(count)+.5)/count)[:, None]*(last-first)
        segments.append([first.tolist(), last.tolist()])
        coordinates = np.floor(np.asarray([start, end])+.5).astype(int)
        valid = mask is not None and all(0 <= x < w and 0 <= y < h for x, y in coordinates)
        members = ([bool(mask[y, x]) for x, y in coordinates] if valid else [None, None])
        parents.append(dict(segment_id=sid, original_hull_edge_index=edge,
            full_parent_xy=[start.tolist(), end.tolist()], xy=[first.tolist(), last.tolist()],
            sample_indices=list(range(len(sample_xy), len(sample_xy)+count)),
            endpoint_pixels_xy=coordinates.tolist(), endpoint_mask_membership=members,
            censor=bool(valid and any(members))))
        sample_xy.extend(positions.tolist())
        sample_parent.extend([sid]*count)
    require(np.array_equal(segments, outline['observed_boundary_segments_xy']), 'Finite original segment exact order')
    require(np.array_equal(sample_xy, outline['observed_boundary_samples_xy']), 'Original sample exact coordinates/order')
    return parents, sample_parent, np.asarray(segments, dtype=float)


def nearest(points, segments):
    """Independent finite segment distance; argmin keeps original segment ties."""
    points, segments = np.asarray(points, dtype=float), np.asarray(segments, dtype=float)
    distances, residuals = [], []
    for first, last in segments:
        delta = last-first
        alpha = ((points-first)*delta).sum(axis=1)/max(float((delta*delta).sum()), 1e-20)
        foot = first+np.minimum(1., np.maximum(0., alpha))[:, None]*delta
        residual = points-foot
        residuals.append(residual)
        distances.append((residual*residual).sum(axis=1))
    indices = np.asarray(distances).argmin(axis=0)
    return indices, np.asarray(residuals)[indices, np.arange(len(points))]


def costs(candidate, frame, mask):
    stage, family = candidate['stage'], candidate['family']
    outline, dense = frame['outlines'][stage][family], candidate['dense']
    parents, sample_parent, segments = lineage(outline, mask)
    censor = np.asarray([p['censor'] for p in parents], dtype=bool)
    directions = []
    source = [d for d in dense['per_observation'] if d['loss_family'] == 'outline']
    require(bool(dense['outline_active']) == bool(source), 'Active outline source rows')
    if source:
        require(sorted(d['direction'] for d in source) == ['model_to_observed', 'observed_to_model'], 'Two directions')
    for row in source:
        if row['direction'] == 'observed_to_model':
            require(np.array_equal(row['sample_xy'], outline['observed_boundary_samples_xy']), 'Observed source coordinates')
            parent = np.asarray(sample_parent, dtype=int)
        else:
            parent, residual = nearest(row['sample_xy'], segments)
            require(np.allclose(residual, row['residual_xy_px'], atol=ATOL, rtol=0), 'Original nearest-segment residuals')
        flags = ~censor[parent]
        weights = [float(x) for x in row['arc_or_sample_weights']]
        contributions = [float(x) for x in row['weighted_sample_contributions']]
        require(len(flags) == len(weights) == len(contributions), 'Direction sample cardinality')
        normalizer = float(row['normalizer'])
        require(abs(math.fsum(weights)-normalizer) <= ATOL, 'Original mass normalizer')
        require(all(x > 0 for x in weights), 'Original positive masses')
        original = float(row['weighted_contribution'])
        require(abs(math.fsum(contributions)-original) <= ATOL, 'Original contributions sum')
        kept_cost = math.fsum(x for x, keep in zip(contributions, flags) if keep)
        removed_cost = math.fsum(x for x, keep in zip(contributions, flags) if not keep)
        kept_mass = math.fsum(x for x, keep in zip(weights, flags) if keep)
        ratio = 1. if flags.all() else 0. if not flags.any() else kept_mass/normalizer
        directions.append(dict(direction=row['direction'], original_count=len(flags), retained_count=int(flags.sum()),
            original_mass=normalizer, retained_mass=kept_mass, retained_mass_ratio=ratio,
            original_weighted_cost=original, support_weighted_cost=original if flags.all() else kept_cost,
            censored_weighted_cost=removed_cost, mass_control_weighted_cost=original*ratio,
            sample_parent_segment_ids=parent.tolist(), retained_sample_flags=flags.tolist(),
            original_normalizer_preserved=True, factor_and_half_already_in_saved_contributions=True))
    objective = float(candidate['original_record']['objective'])
    old_outline = float(dense['weighted_components']['outline'])
    require(abs(float(dense['objective'])-objective) <= ATOL, 'Original objective binding')
    require(abs(math.fsum(d['original_weighted_cost'] for d in directions)-old_outline) <= ATOL, 'Original outline sum')
    identity = not censor.any() or not dense['outline_active']
    removed = math.fsum(d['censored_weighted_cost'] for d in directions)
    supported = old_outline if identity else math.fsum(d['support_weighted_cost'] for d in directions)
    mass = old_outline if identity else math.fsum(d['mass_control_weighted_cost'] for d in directions)
    scores = dict(BASELINE_E=objective,
        SUPPORT_AWARE_RERANK=objective if identity else math.fsum([objective, -removed]),
        MASS_MATCHED_CONTROL=objective if identity else math.fsum([objective, -old_outline, mass]))
    require(abs(scores['SUPPORT_AWARE_RERANK']-math.fsum([objective, -old_outline, supported])) <= ATOL,
            'Original-minus-censored equals outline replacement')
    return dict(old_objective=objective, original_outline_cost=old_outline,
        support_outline_cost=supported, mass_control_outline_cost=mass,
        censored_outline_cost=removed, identity_shortcut=bool(identity), scores=scores, directions=directions), parents


def select(frame, candidates, values, variant):
    pools = defaultdict(list)
    for c in candidates:
        pools[c['stage'], c['family']].append(c)
    winners, ranking, family_scores = {}, {}, []
    for stage in STAGES:
        winners[stage], ranking[stage] = {}, {}
        require([m['family'] for m in frame['stages'][stage]['families']] == list(FAMILIES), 'Original family order')
        for family_index, family in enumerate(FAMILIES):
            # Original insertion order is restored first, then Python stable
            # sorting preserves it when the schema comparator has an exact tie.
            ordered = sorted(pools[stage, family], key=lambda c: c['original_outcome_order'])
            def key(c):
                score = values[stage, family, c['candidate_id']]['scores'][variant]
                require(math.isfinite(score), 'Finite original/new scalar')
                return ((score,) if c['solver_schema'] == 'statewise_u_region_solver_v4_5'
                        else (score, c['original_record']['initialization']))
            ordered = sorted(ordered, key=key)
            metadata = frame['stages'][stage]['families'][family_index]
            require(bool(ordered) == metadata['accepted'], 'Unchanged original accepted pool')
            require(all(c['retained'] for c in ordered), 'Original retained only')
            require(len({c['solver_schema'] for c in ordered}) <= 1, 'One solver schema per pool')
            ranking[stage][family] = [c['candidate_id'] for c in ordered]
            winners[stage][family] = ordered[0]['candidate_id'] if ordered else None
            if stage == 'G10' and ordered:
                family_scores.append((key(ordered[0])[0], family_index, family))
    family = min(family_scores)[2] if family_scores else FAMILIES[0]
    cid = winners['G13'][family]
    metadata = frame['stages']['G13']['families'][FAMILIES.index(family)]
    pose = deepcopy(metadata['published_actual_pose'])
    selected = next((c for c in pools['G13', family] if c['candidate_id'] == cid), None)
    if selected is not None:
        pose.update(R_cf=selected['pose']['R_cf'], centroid=selected['pose']['centroid'],
            R_physical=selected['pose']['R_physical'], physical_registration_Q=selected['physical_registration_Q'],
            cf_extents=selected['cf_extents'])
        if 'native_t' in pose:
            pose['native_t'] = selected['pose']['native_t']
        require(np.allclose(np.asarray(pose['R_cf'])@pose['physical_registration_Q'], pose['R_physical'],
                            atol=ATOL, rtol=0), 'Physical same-family Q')
    return dict(id=frame['id'], variant=variant, selected_family=family, candidate_id=cid,
        accepted=metadata['accepted'], fallback=not metadata['accepted'], pose=pose,
        original_constraint_pass=None if selected is None else selected['dense']['original_joint_feasible'],
        recomputed_constraint_pass=None if selected is None else selected['dense']['recomputed_joint_feasible'],
        minimum_constraint_slack=None if selected is None else selected['dense']['minimum_constraint_slack'],
        original_fallback_status=metadata['status'], stage_winner_ids=winners, ranking=ranking,
        same_family_registration_preserved=True, compares_G10_with_G13_objectives=False,
        original_fallback_preserved=True, new_pose_fits=0)


def verify_run(package_path, input_manifest, run_path):
    package, run_path = read(package_path), Path(run_path)
    frames = rows(bound(package_path, package['frames']))
    candidates = rows(bound(package_path, package['candidates']))
    require([f['id'] for f in frames] == list(IDS), 'Exact six fixed order')
    require(len(candidates) == 351, 'All original 351 retained records')
    require(not non_neutral(frames) and not non_neutral(candidates), 'No reference/audit/metric fields in neutral bank')
    masks = input_masks(input_manifest, frames)
    keys = [(c['id'], c['stage'], c['family'], c['candidate_id']) for c in candidates]
    require(len(set(keys)) == len(keys), 'Composite candidate key unique')
    expected = {tuple(c[k] for k in ('id', 'stage', 'family', 'candidate_id')): c
                for c in rows(run_path/'CANDIDATE_COSTS.jsonl.gz')}
    require(set(expected) == set(keys), 'Full retained output denominator')
    parent_rows = rows(run_path/'PARENT_SUPPORT.jsonl')
    parent_actual = {(p['id'], p['stage'], p['family'], p['segment_id']): p for p in parent_rows}
    require(len(parent_actual) == len(parent_rows), 'No duplicate parent records')
    pred_actual = rows(run_path/'PREDICTIONS.jsonl.gz')
    require([(p['id'], p['variant']) for p in pred_actual] == [(fid, v) for fid in IDS for v in VARIANTS],
            'All18 outputs exact full order')
    pred_map = {(p['id'], p['variant']): p for p in pred_actual}
    maximum, independently_predicted, frame_checks, parent_seen = [0.], [], [], set()
    for frame in frames:
        own = [c for c in candidates if c['id'] == frame['id']]
        values, no_input_values = {}, {}
        for candidate in own:
            k = tuple(candidate[k] for k in ('id', 'stage', 'family', 'candidate_id'))
            equal(candidate['R'], candidate['pose']['R_cf'], 'Candidate native rotation', maximum)
            equal(candidate['t'], candidate['pose']['native_t'], 'Candidate native translation', maximum)
            centroid = np.asarray(candidate['R'])@np.asarray(candidate['native_xyz']).mean(axis=0)+candidate['t']
            require(np.allclose(centroid, candidate['pose']['centroid'], atol=ATOL, rtol=0),
                    'Candidate physical center from original native R/t/XYZ')
            value, parents = costs(candidate, frame, masks[frame['id']])
            identity, _ = costs(candidate, frame, None)
            for variant in VARIANTS:
                require(identity['scores'][variant] == candidate['original_record']['objective'], 'Empty exact scalar identity')
            local = k[1:]
            values[local], no_input_values[local] = value, identity
            for field, actual in value.items():
                equal(actual, expected[k][field], 'candidate/'+str(k)+'/'+field, maximum)
            equal(candidate['dense']['weighted_components'], expected[k]['original_weighted_components'],
                  'Original non-outline/outline component preservation', maximum)
            require(expected[k]['original_hard_constraints_preserved'] is True, 'Original hard constraints preserved')
            require(all(expected[k][field] == 0 for field in ('new_observations_created', 'new_model_samples_created', 'new_pose_fits')),
                    'No new target/model samples or fits')
            for parent in parents:
                pk = (frame['id'], candidate['stage'], candidate['family'], parent['segment_id'])
                parent_seen.add(pk)
                require(pk in parent_actual, 'Original parent output present')
                for field, value in parent.items():
                    equal(value, parent_actual[pk][field], 'parent/'+str(pk)+'/'+field, maximum)
        baseline = select(frame, own, values, 'BASELINE_E')
        require(baseline['candidate_id'] == frame['selected_candidate_id'], 'Original e candidate exact')
        require(baseline['selected_family'] == frame['selected_family'], 'Original e family exact')
        require(baseline['accepted'] == frame['original_e_accepted'], 'Original e acceptance')
        equal(baseline['pose'], frame['original_e_actual_pose'], 'Original e pose', maximum)
        for variant in VARIANTS:
            predicted = select(frame, own, values, variant)
            independent_empty = select(frame, own, no_input_values, variant)
            require({k: v for k, v in independent_empty.items() if k != 'variant'} ==
                    {k: v for k, v in baseline.items() if k != 'variant'}, 'Empty exact rank/pose metadata identity')
            equal(predicted, pred_map[frame['id'], variant], 'Final selection/'+frame['id']+'/'+variant, maximum)
            independently_predicted.append(predicted)
        frame_checks.append(dict(id=frame['id'], candidates=len(own), input_present=masks[frame['id']] is not None,
                                 no_input_cost_rank_pose_exact=True, baseline_original_e_parity=True))
    require(parent_seen == set(parent_actual), 'All output parents independent lineage accounted')
    seal = read(run_path/'PREDICTION_SEAL.json')
    require(seal['neutral_package_sha256'] == digest(package_path), 'Prediction sealed neutral package')
    require(seal['input_manifest_sha256'] == (None if input_manifest is None else digest(input_manifest)), 'Prediction sealed frozen input')
    for name, expected_sha in seal['artifacts'].items():
        require(digest(run_path/name) == expected_sha, 'Prediction artifact seal '+name)
    require(seal['frames'] == 6 and seal['candidates'] == 351 and seal['prediction_rows'] == 18, 'Seal complete denominator')
    require(all(seal[k] == 0 for k in ('new_pose_fit_calls', 'new_candidate_calls', 'new_model_forward_calls')), 'Selector no new science')
    require(seal['runtime_guard']['forbidden_function_bodies_executed'] == 0
            and not seal['runtime_guard']['forbidden_blocked_attempts'], 'Selector runtime forbidden scientific calls absent')
    require(not seal['read_guard']['denied_events'] and seal['read_guard']['neutral_or_supplier_or_new_output_only']
            and seal['read_guard']['network_and_subprocess_denied'], 'Selector actual reference-free read boundary')
    return dict(passed=True, independent_kernel_replay_selector_imports=0,
        frames=6, candidates=351, prediction_rows=18, original_parent_records=len(parent_rows),
        maximum_absolute_arithmetic_or_pose_difference=maximum[0], atol=ATOL,
        same_original_sample_indices_normalizers_and_weights=True,
        original_preclip_endpoint_membership_checked=True,
        original_all_segment_nearest_with_first_tie_checked=True,
        new_target_polygon_or_boundary_created=False, all_stage_family_candidates_ranked=True,
        frame_checks=frame_checks,
        input_sha256=None if input_manifest is None else digest(input_manifest),
        prediction_seal_sha256=digest(run_path/'PREDICTION_SEAL.json'))


def protection(start_path, original_root, export_receipt):
    start, root = read(start_path), Path(original_root)
    checked = []
    for relative, expected in start['protected_files_SHA256'].items():
        require(digest(root/relative) == expected, 'Protected user file changed '+relative)
        checked.append(relative)
    for entry in read(export_receipt)['original_assets']:
        require(digest(entry['path']) == entry['sha256'], 'Original RGB/target file changed')
    all_status = subprocess.check_output(['git', 'status', '--porcelain', '-z'], cwd=root)
    tracked = subprocess.check_output(['git', 'status', '--porcelain', '-z', '--untracked-files=no'], cwd=root)
    require(hashlib.sha256(all_status).hexdigest() == start['original_all_status_SHA256'], 'Original all-files status changed')
    require(hashlib.sha256(tracked).hexdigest() == start['original_tracked_status_SHA256'], 'Original tracked status changed')
    require(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root).decode().strip() == start['original_HEAD'], 'Original HEAD changed')
    for item in start['branch_refs']:
        name, expected = item.split()
        current = subprocess.check_output(['git', 'rev-parse', name], cwd=root).decode().strip()
        require(current == expected, 'Existing branch ref changed '+name)
    return dict(passed=True, protected_user_files=checked,
        original_rgb_and_target_assets=len(read(export_receipt)['original_assets']),
        original_NUL_status_and_HEAD_unchanged=True, existing_branch_refs_unchanged=True)


def supplier_check(supplier_path, prompt_path, frozen_input_path):
    """Verify recorded calls and all alternatives; no segmentation is rerun."""
    folder = Path(supplier_path)
    receipt = read(folder/'SUPPLIER_RUN.json')
    prompts = rows(prompt_path)
    supplied = rows(folder/'NON_TARGET_INPUT_MANIFEST.jsonl')
    frozen = rows(frozen_input_path)
    require(len(prompts) == len(supplied) == len(frozen) == 6, 'Supplier full six input denominator')
    require([r['image_id'] for r in supplied] == list(IDS), 'Supplier fixed six order')
    require([r['id'] for r in frozen] == list(IDS), 'Frozen fixed six order')
    expected_calls = sum(p['status'] == 'ADMISSIBLE_NON_TARGET_EVIDENCE' for p in prompts)
    require(receipt['complete'] and receipt['input_rows'] == 6, 'Actual supplier completed all six')
    for name in ('image_encoder_forwards', 'prompt_decoder_calls', 'prompt_encoder_forwards', 'mask_decoder_forwards'):
        require(receipt['counters'][name] == expected_calls, 'One actual forward per admissible prompt '+name)
    require(receipt['prompt_manifest_sha256'] == digest(prompt_path), 'Actual supplier prompt bytes frozen')
    require(receipt['quality_is_calibrated_probability'] is False, 'No quality probability certification')
    admitted = no_prompt = ambiguous = 0
    alternative_count = 0
    for prompt, raw, entry in zip(prompts, supplied, frozen):
        require(raw['rgb_file_sha256'] == prompt['rgb_sha256'] == entry['image_sha256'], 'Supplier source RGB')
        require(raw['supplier_mode'] == 'ASSISTED', 'No automatic input claim')
        require(raw['physical_depth_order_asserted'] is False, 'No unsupported depth-order assertion')
        if prompt['status'] == 'NO_ADMISSIBLE_NON_TARGET_EVIDENCE':
            require(raw['encoder_calls'] == raw['decoder_calls'] == 0, 'Unknown/no-prompt no model call')
            require(raw['selected_mask_path'] is None and not entry['admissible_for_censoring'], 'No invented input for absent cone')
            no_prompt += 1
            continue
        require(raw['encoder_calls'] == raw['decoder_calls'] == 1, 'One encoder+decoder actual call')
        require(raw['prompt_box_xyxy'] == prompt['box_xyxy'] and raw['positive_point_xy'] == prompt['positive_point_xy'],
                'Supplier prompt unchanged after lock')
        path = Path(raw['all_masks_path'])
        require(digest(path) == raw['all_masks_file_sha256'], 'All returned masks archived')
        with np.load(path, allow_pickle=False) as archive:
            masks, qualities = archive['masks'], archive['quality']
            require(masks.shape == (3, *raw['image_hw']) and masks.dtype == np.bool_, 'Three binary original-canvas alternatives')
            require(qualities.shape == (3,) and np.isfinite(qualities).all(), 'Three finite qualities')
            require(np.array_equal(archive['original_indices'], [0, 1, 2]), 'Original alternative order')
            selected = int(np.argmax(qualities))
            require(selected == int(archive['selected_index']) == raw['selected_index'], 'Maximum quality first original tie')
            require(np.allclose(qualities, raw['quality'], atol=0, rtol=0), 'Recorded qualities exact')
            for index, mask in enumerate(masks):
                binary = np.ascontiguousarray(mask, dtype=np.uint8)
                pixel_sha = hashlib.sha256(binary.tobytes()).hexdigest()
                require(pixel_sha == raw['alternative_pixel_sha256'][index], 'All alternative binary pixel SHA')
                require(int(mask.sum()) == raw['foreground_pixels'][index], 'All foreground counts')
                from PIL import Image
                png_path = Path(raw['alternative_mask_paths'][index])
                require(digest(png_path) == raw['alternative_mask_file_sha256'][index], 'All alternative PNG file SHA')
                with Image.open(png_path) as image:
                    encoded = np.asarray(image)
                    require(np.array_equal(encoded, binary*255), 'Archived PNG exact 0/255 encoding of returned binary mask')
            require(raw['selected_pixel_sha256'] == raw['alternative_pixel_sha256'][selected], 'Selected original alternative unchanged')
        alternative_count += 3
        if entry['admissible_for_censoring']:
            require(entry['pixel_bytes_sha256'] == raw['selected_pixel_sha256'], 'Admitted mask is the fixed quality winner')
            admitted += 1
        else:
            ambiguous += 1
    access = read(folder/'SOURCE_ACCESS_AUDIT.json')
    require(access['enforcing'], 'Actual supplier audit-open enforcement active')
    blocked_files = [event for event in access['denied_attempts'] if 'path' in event]
    blocked_runtime = Counter(event.get('event') for event in access['denied_attempts'] if 'event' in event)
    require(not blocked_files, 'Actual supplier scientific/file read boundary violations absent')
    require(set(blocked_runtime) <= {'socket.bind', 'subprocess.Popen'}, 'Only documented optional runtime operations blocked')
    if blocked_runtime:
        require(receipt.get('completed_via_packaging_recovery_without_NN') is True, 'Recovered packaging failure is explicitly preserved')
        require(receipt['execution_status'] == 'RECOVERED_AFTER_POST_FORWARD_PACKAGING_FAILURE'
                and receipt['initial_process_clean_exit'] is False, 'Initial process must not be reported clean')
        require(dict(blocked_runtime) == receipt['blocked_runtime_operations'], 'Exact blocked runtime event counts preserved')
        require(digest(folder/'SUPPLIER_FAILURE.json') == receipt['source_failure_receipt_sha256'], 'Original post-forward failure receipt preserved')
        failure = read(folder/'SUPPLIER_FAILURE.json')
        require(failure['rows_finished'] == 6 and failure['counters'] == receipt['counters'], 'Recovered rows/call counts match original failure')
        require(receipt['new_NN_calls_in_packaging_recovery'] == receipt['new_RGB_reads_in_packaging_recovery'] == 0,
                'No retry/extra inference in packaging recovery')
        require(receipt['model_load_seconds'] is None, 'Unpersisted timing is not invented')
    actual_source = folder/'SUPPLIER_SOURCE_AT_INFERENCE.py'
    if actual_source.exists():
        require(digest(actual_source) == receipt['supplier_source_sha256_at_inference'], 'Exact source-at-inference bytes preserved')
    sentinel = read(folder/'BOUNDARY_SENTINELS.json')
    require(len(sentinel) == 4 and all(s['blocked_before_read'] for s in sentinel), 'Actual forbidden-source sentinel opens blocked')
    forbidden_names = ('SUPPORT_DECISIONS', 'GT.json', 'candidate_bank', 'target_mask')
    require(not any(token in path for path in access['allowed_open_paths'] for token in forbidden_names),
            'No forbidden dataset opened by supplier')
    return dict(passed=True, actual_encoder_calls=expected_calls, actual_decoder_calls=expected_calls,
        frozen_mask_quality_rule_verified=True, archived_alternatives=alternative_count,
        admitted_inputs=admitted, no_admissible_prompt=no_prompt, predicted_but_not_admitted=ambiguous,
        actual_supplier_sentinel_opens_blocked=4, supplier_forbidden_file_accesses=0,
        preserved_blocked_optional_runtime_events=dict(blocked_runtime),
        supplier_execution_status=receipt.get('execution_status', 'COMPLETE_CLEAN'),
        supplier_security_limit=access['security_limit'], no_new_model_calls_by_validator=True,
        supplier_receipt_sha256=digest(folder/'SUPPLIER_RUN.json'))


def csv_rows(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def publication_and_weight_check(prior_bindings, asset_root, export_receipt, supplier_config):
    """Directly check prior publication inventory and unchanged vendor weights."""
    root = Path(asset_root)
    permission = read(prior_bindings)['publication_permission']
    inventory = root/permission['path']
    require(digest(inventory) == permission['sha256'], 'Prior public release inventory bytes')
    # Neutral export may bind an exact readonly asset through the isolated
    # worktree's data/legacy symlink. Resolve bytes to the original asset root.
    wanted = {str(Path(e['path']).resolve().relative_to(root.resolve())): e
              for e in read(export_receipt)['original_assets']}
    found = {}
    with gzip.open(inventory, 'rt') as stream:
        for line in stream:
            entry = json.loads(line)
            if entry['relpath'] in wanted:
                require(entry['relpath'] not in found, 'Public inventory member unique')
                found[entry['relpath']] = entry
    require(set(found) == set(wanted) and len(found) == 12, 'All six RGB and six source masks have inventory entries')
    for relative, entry in found.items():
        require(entry['prior_release_exact'] and entry['prior_release_path_present'], 'Existing public release member exact')
        require(digest(root/relative) == entry['sha256'] == wanted[relative]['sha256'], 'Current public-authorized original bytes')
    config = read(supplier_config)
    vendor = Path(config['vendor_root'])
    for relative, expected in config['runtime_files_sha256'].items():
        require(digest(vendor/relative) == expected, 'Original vendor runtime unchanged '+relative)
    require(digest(config['checkpoint_path']) == config['checkpoint_sha256'], 'Original segmentation checkpoint unchanged')
    return dict(passed=True, existing_publication_inventory_sha256=digest(inventory),
        prior_release_exact_source_RGBs=6, prior_release_exact_source_masks=6,
        vendor_runtime_files_unchanged=len(config['runtime_files_sha256']), checkpoint_unchanged=True,
        no_raw_RGB_NPZ_or_weight_publication_authorized_by_this_check=True,
        publication_scope='Prior released originals permit requested derived small figures; no new rights assertion',
        new_model_forward_calls=0)


def csv_value(value):
    if value == '':
        return None
    if value in ('True', 'False'):
        return value == 'True'
    return value


def evaluation_check(run_path, package_path, evaluation_path, reference_path, reference_seal):
    """Post-seal only: independent center/C2 metrics and paired arithmetic."""
    run_path, evaluation_path = Path(run_path), Path(evaluation_path)
    seal = read(run_path/'PREDICTION_SEAL.json')
    for name, expected in seal['artifacts'].items():
        require(digest(run_path/name) == expected, 'Prediction seal before independent reference read')
    reference_binding = read(reference_seal)
    require(digest(reference_path) == reference_binding['artifacts']['FIXED_REFERENCES.json'], 'Original fixed references sealed')
    references = read(reference_path)['references']
    package = read(package_path)
    frames = {f['id']: f for f in rows(bound(package_path, package['frames']))}
    actual = csv_rows(evaluation_path/'METRICS.csv')
    transitions = csv_rows(evaluation_path/'TRANSITIONS.csv')
    predictions = rows(run_path/'PREDICTIONS.jsonl.gz')
    require(len(actual) == len(predictions) == 18 and len(transitions) == 12, 'Evaluation all18 and paired12 denominator')
    maximum, expected_rows = [0.], []
    for prediction, stored in zip(predictions, actual):
        fid, variant, pose = prediction['id'], prediction['variant'], prediction['pose']
        frame = frames[fid]
        reference = references[frame['partition']][fid]
        require(reference['order'] == 2, 'Existing C2 contract only')
        if pose['available']:
            vector = np.asarray(pose['centroid'], float)-np.asarray(reference['t'], float)
            translation = 100.*math.sqrt(math.fsum(float(v)*float(v) for v in vector))
            if reference['R'] is None:
                angle = None
            else:
                R, G = np.asarray(pose['R_physical'], float), np.asarray(reference['R'], float)
                # An independent Frobenius-dot implementation of trace(G.T R).
                # C2 is exactly the half-turn around the registered vertical Y.
                symmetry = np.diag([-1., 1., -1.])
                cosines = [(math.fsum(float(a)*float(b) for a, b in zip((G@S).ravel(), R.ravel()))-1.)/2.
                           for S in (np.eye(3), symmetry)]
                angle = math.degrees(math.acos(min(1., max(-1., max(cosines)))))
            success = None if angle is None else translation <= 5. and angle <= 5.
        else:
            translation, angle, success = None, None, False
        row = dict(id=fid, ordinal=frame['ordinal'], partition=frame['partition'], variant=variant,
            candidate_id=prediction['candidate_id'], family=prediction['selected_family'],
            accepted=prediction['accepted'], fallback=prediction['fallback'],
            translation_cm=translation, rotation_deg=angle, success_le5=success,
            pose_available=pose['available'], reference_rotation_available=reference['R'] is not None)
        for field, expected in row.items():
            value = csv_value(stored[field])
            if isinstance(expected, float):
                value = float(value)
            elif type(expected) is int:
                value = int(value)
            equal(expected, value, 'Independent metric/'+fid+'/'+variant+'/'+field, maximum)
        expected_rows.append(row)
    metric_map = {(r['id'], r['variant']): r for r in expected_rows}
    transition_map = {(t['id'], t['variant']): t for t in transitions}
    require(len(transition_map) == 12, 'Unique transition keys')
    group_sums, summaries = {}, {}
    for variant in VARIANTS:
        own = [r for r in expected_rows if r['variant'] == variant]
        translations = [r['translation_cm'] for r in own if r['translation_cm'] is not None]
        rotations = [r['rotation_deg'] for r in own if r['rotation_deg'] is not None]
        deltas = defaultdict(lambda: [[], []])
        counts, changed_candidates, changed_families = Counter(), 0, 0
        if variant != 'BASELINE_E':
            for fid in IDS:
                baseline, current = metric_map[fid, 'BASELINE_E'], metric_map[fid, variant]
                before, after = baseline['success_le5'], current['success_le5']
                group = ('RECOVERY' if after and not before else 'DAMAGE' if before and not after else
                         'PRESERVED_SUCCESS' if after else 'PERSISTENT_FAILURE')
                dt, dr = current['translation_cm']-baseline['translation_cm'], current['rotation_deg']-baseline['rotation_deg']
                expected = dict(id=fid, ordinal=frames[fid]['ordinal'], variant=variant,
                    baseline_candidate_id=baseline['candidate_id'], new_candidate_id=current['candidate_id'],
                    candidate_changed=baseline['candidate_id'] != current['candidate_id'],
                    family_changed=baseline['family'] != current['family'], baseline_success=before,
                    new_success=after, transition=group, translation_delta_cm=dt, rotation_delta_deg=dr)
                stored = transition_map[fid, variant]
                for field, value in expected.items():
                    actual_value = csv_value(stored[field])
                    if isinstance(value, float):
                        actual_value = float(actual_value)
                    elif type(value) is int:
                        actual_value = int(actual_value)
                    equal(value, actual_value, 'Independent transition/'+fid+'/'+variant+'/'+field, maximum)
                counts[group] += 1
                changed_candidates += expected['candidate_changed']
                changed_families += expected['family_changed']
                deltas[group][0].append(dt)
                deltas[group][1].append(dr)
            group_sums[variant] = {group: dict(N=counts[group], translation_delta_sum_cm=math.fsum(deltas[group][0]),
                rotation_delta_sum_deg=math.fsum(deltas[group][1]))
                for group in ('RECOVERY', 'DAMAGE', 'PRESERVED_SUCCESS', 'PERSISTENT_FAILURE')}
            require(sum(counts.values()) == 6, 'All transition categories retained')
            require(sum(r['success_le5'] is True for r in own)-sum(metric_map[fid, 'BASELINE_E']['success_le5'] is True for fid in IDS)
                    == counts['RECOVERY']-counts['DAMAGE'], 'Recovery minus damage equals success delta')
            for direction, field in ((0, 'translation_cm'), (1, 'rotation_deg')):
                expected_delta = math.fsum(metric_map[fid, variant][field]-metric_map[fid, 'BASELINE_E'][field] for fid in IDS)
                require(abs(expected_delta-math.fsum(math.fsum(v[direction]) for v in deltas.values())) <= ATOL,
                        'Transition group continuous delta sums partition full denominator')
        def statistics(values):
            mean = math.fsum(values)/len(values)
            sd = math.sqrt(math.fsum((v-mean)**2 for v in values)/(len(values)-1))
            return mean, sd, max(values)
        tm, ts, tx = statistics(translations)
        rm, rs, rx = statistics(rotations)
        summaries[variant] = dict(denominator=6, successes=sum(r['success_le5'] is True for r in own),
            recoveries=counts['RECOVERY'], damages=counts['DAMAGE'], changed_candidates=changed_candidates,
            changed_families=changed_families, translation_mean_cm=tm, translation_sample_SD_cm=ts,
            translation_max_cm=tx, rotation_mean_deg=rm, rotation_sample_SD_deg=rs, rotation_max_deg=rx,
            translation_gt50cm=sum(t > 50 for t in translations), translation_gt100cm=sum(t > 100 for t in translations))
    summary = read(evaluation_path/'SUMMARY.json')
    equal(summaries, summary['metrics'], 'Independent six-case metric summary', maximum)
    require(summary['AUTO238_executed'] is False and summary['populations_combined'] is False, 'No automatic population claim')
    receipt = read(evaluation_path/'EVALUATION_RECEIPT.json')
    require(receipt['prediction_seal_sha256'] == digest(run_path/'PREDICTION_SEAL.json'), 'Evaluator read final prediction seal')
    require(receipt['reference_sha256'] == digest(reference_path), 'Evaluator original reference binding')
    for name, expected in receipt['artifacts'].items():
        require(digest(evaluation_path/name) == expected, 'Evaluation artifact seal')
    return dict(passed=True, direct_independent_center_C2_metrics=18, paired_transitions=12,
        no_quarter_turn_equivalence_added=True, maximum_absolute_metric_difference=maximum[0],
        recovery_minus_damage_identity=True, continuous_delta_group_sums=group_sums,
        fixed_reference_sha256=digest(reference_path), summary=summaries,
        new_pose_calls=0, new_NN_calls=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--neutral-package', required=True)
    parser.add_argument('--input-manifest')
    parser.add_argument('--run', required=True)
    parser.add_argument('--empty-run')
    parser.add_argument('--start-snapshot')
    parser.add_argument('--original-root')
    parser.add_argument('--export-receipt')
    parser.add_argument('--supplier-run')
    parser.add_argument('--prompts')
    parser.add_argument('--evaluation')
    parser.add_argument('--fixed-references')
    parser.add_argument('--reference-seal')
    parser.add_argument('--prior-publication-bindings')
    parser.add_argument('--supplier-config')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    started, cpu = time.monotonic(), time.process_time()
    result = dict(schema='NON_TARGET_PILOT_INDEPENDENT_VALIDATION_V1',
        arithmetic_and_selection=verify_run(args.neutral_package, args.input_manifest, args.run))
    if args.empty_run:
        result['independent_empty_run'] = verify_run(args.neutral_package, None, args.empty_run)
    if args.start_snapshot:
        require(args.original_root and args.export_receipt, 'Protection inputs required together')
        result['original_protection'] = protection(args.start_snapshot, args.original_root, args.export_receipt)
    if args.supplier_run:
        require(args.prompts and args.input_manifest, 'Frozen supplier validation inputs required')
        result['supplier'] = supplier_check(args.supplier_run, args.prompts, args.input_manifest)
    if args.evaluation:
        require(args.fixed_references and args.reference_seal, 'Original post-seal reference bindings required')
        result['postseal_evaluation'] = evaluation_check(args.run, args.neutral_package, args.evaluation,
                                                        args.fixed_references, args.reference_seal)
    if args.prior_publication_bindings:
        require(args.original_root and args.export_receipt and args.supplier_config,
                'Prior publication/weight audit inputs required')
        result['prior_publication_and_weights'] = publication_and_weight_check(
            args.prior_publication_bindings, args.original_root, args.export_receipt, args.supplier_config)
    result.update(passed=True, new_NN_calls=0, new_pose_calls=0, source_sha256=digest(__file__),
                  elapsed_seconds=time.monotonic()-started, process_cpu_seconds=time.process_time()-cpu)
    with Path(args.output).open('x') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
