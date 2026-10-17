"""Reference-free four-arm re-selection of the original 351 retained poses.

Only neutral candidate/cost packets and frozen RGB-only support inputs are
read during this process. Reference evaluation belongs to a later process.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
import time

import numpy as np

from scripts.research.pallet_selection_score_diagnosis_v1.selection import rank_retained
from scripts.research.pallet_selection_score_diagnosis_v1.runtime_guard import FixedPoseOnly
from scripts.research.pallet_non_target_outline_pilot_v1.replay import (
    FAMILIES, STAGES, ReferenceFreeIO as OriginalReferenceFreeIO,
    _bound_file, _deny_fields, _read, _rows, _save_rows, _write, load_package,
    sha, validate_candidate_pose,
)
from .kernel import VARIANTS, evaluate_candidate, sample_visible, validate_polylines


class ReferenceFreeIO(OriginalReferenceFreeIO):
    """Old scientific artifact ban plus legitimate runtime/cache files.

    A cooperative audit boundary, not an OS sandbox. Temporary/library/font
    access is allowed but reference/audit/metric scientific artifacts remain
    forbidden even when their directory otherwise qualifies.
    """
    forbidden_artifact_tokens = OriginalReferenceFreeIO.forbidden_artifact_tokens + (
        'support_decisions', 'support_review', 'candidate_costs', 'candidate_scores',
        'metrics.csv', 'transitions.csv', 'evidence_discrimination',
    )

    def __init__(self, package_path, input_manifest, output):
        super().__init__(package_path, input_manifest, output)
        self.allow(__file__)
        self.allow(Path(__file__).with_name('kernel.py'))
        self.library_roots += tuple(p.resolve() for p in (
            Path(tempfile.gettempdir()), Path('/usr/share/fonts'),
            Path('/usr/share/fontconfig'), Path('/var/cache/fontconfig'),
            Path.home()/'.cache'/'matplotlib', Path.home()/'.cache'/'fontconfig',
        ))

    def _audit(self, event, args):
        # Newly written score files are legitimate outputs, but never selector
        # scientific *inputs*. The inherited artifact ban precedes directory
        # permission, so output access needs this narrow read/write distinction.
        # SciPy/QHull opens its legitimate message-stream temporary file with
        # a bytes pathname. pathlib.Path rejects bytes even though os.open
        # accepts them. Decode losslessly using the OS filesystem convention
        # *before* applying the same scientific-artifact and location checks.
        if event == 'open' and isinstance(args[0], bytes):
            args = (os.fsdecode(args[0]), *args[1:])
        if self.active and event == 'open' and not isinstance(args[0], int):
            path = Path(args[0]).resolve()
            if path == self.output or self.output in path.parents:
                self.open_events += 1
                return
        return super()._audit(event, args)


def choose(frame, candidates, results, variant):
    """G10 family choice, then G13 within that family; no cross-stage scores."""
    if variant not in VARIANTS:
        raise ValueError('Only the four predeclared variants are allowed')
    pools = defaultdict(list)
    for candidate in candidates:
        if candidate['id'] != frame['id']:
            raise ValueError('One frame per selection call required')
        pools[(candidate['stage'], candidate['family'])].append(candidate)
    orders, winners, accepted_scores = {}, {}, []
    for stage in STAGES:
        metadata = frame['stages'][stage]['families']
        if [entry['family'] for entry in metadata] != list(FAMILIES):
            raise ValueError('Original family order required')
        orders[stage], winners[stage] = {}, {}
        for index, family in enumerate(FAMILIES):
            pool = pools[(stage, family)]
            scores = {candidate['candidate_id']: results[(stage, family, candidate['candidate_id'])]['scores'][variant]
                      for candidate in pool}
            ranked = rank_retained(pool, scores)
            if bool(ranked) != metadata[index]['accepted']:
                raise ValueError('Original accepted/fallback and retained pool disagree')
            orders[stage][family] = [candidate['candidate_id'] for candidate in ranked]
            winner = ranked[0] if ranked else None
            winners[stage][family] = None if winner is None else winner['candidate_id']
            if stage == 'G10' and winner is not None:
                accepted_scores.append((float(scores[winner['candidate_id']]), index, family))
    selected_family = min(accepted_scores)[2] if accepted_scores else FAMILIES[0]
    final_id = winners['G13'][selected_family]
    family_meta = frame['stages']['G13']['families'][FAMILIES.index(selected_family)]
    pose = deepcopy(family_meta['published_actual_pose'])
    selected = next((c for c in pools[('G13', selected_family)] if c['candidate_id'] == final_id), None)
    if selected is not None:
        validate_candidate_pose(selected)
        p = selected['pose']
        pose.update(R_cf=deepcopy(p['R_cf']), centroid=deepcopy(p['centroid']),
                    R_physical=deepcopy(p['R_physical']),
                    physical_registration_Q=deepcopy(selected['physical_registration_Q']),
                    cf_extents=deepcopy(selected['cf_extents']))
        # Current six-case publications do not contain this field. Preserve a
        # future publication's native_t without adding keys to the old schema.
        if 'native_t' in pose:
            pose['native_t'] = deepcopy(p['native_t'])
        if not np.allclose(np.asarray(pose['R_cf'])@pose['physical_registration_Q'], pose['R_physical'],
                           atol=1e-9, rtol=0):
            raise ValueError('Same-family physical Q conversion failed')
    return dict(id=frame['id'], variant=variant, selected_family=selected_family, candidate_id=final_id,
                accepted=family_meta['accepted'], fallback=not family_meta['accepted'], pose=pose,
                original_constraint_pass=(None if selected is None else selected['dense']['original_joint_feasible']),
                recomputed_constraint_pass=(None if selected is None else selected['dense']['recomputed_joint_feasible']),
                minimum_constraint_slack=(None if selected is None else selected['dense']['minimum_constraint_slack']),
                original_fallback_status=family_meta['status'], stage_winner_ids=winners, ranking=orders,
                same_family_registration_preserved=True, compares_G10_with_G13_objectives=False,
                original_fallback_preserved=True, new_pose_fits=0)


def _check_baseline(frame, prediction):
    if prediction['selected_family'] != frame['selected_family'] or prediction['candidate_id'] != frame['selected_candidate_id']:
        raise ValueError('BASELINE_E original family/candidate replay failed')
    if prediction['accepted'] != frame['original_e_accepted']:
        raise ValueError('BASELINE_E original acceptance replay failed')
    expected, actual = frame['original_e_actual_pose'], prediction['pose']
    if set(expected) != set(actual):
        raise ValueError('BASELINE_E published pose metadata keys differ')
    for key in expected:
        if isinstance(expected[key], list):
            if not np.allclose(expected[key], actual[key], atol=1e-9, rtol=0):
                raise ValueError('BASELINE_E numeric pose mismatch: '+key)
        elif expected[key] != actual[key] or type(expected[key]) is not type(actual[key]):
            raise ValueError('BASELINE_E exact pose metadata mismatch: '+key)


def load_inputs(path, frames):
    """Read one frozen model/human-assisted RGB-only sheet per original RGB."""
    if path is None:
        return {frame['id']: dict(status='NO_INPUT_IDENTITY_TEST', polylines=[]) for frame in frames}
    entries = _rows(path)
    if [entry['id'] for entry in entries] != [frame['id'] for frame in frames]:
        raise ValueError('Exact same six ordered input identities required')
    result = {}
    for frame, entry in zip(frames, entries):
        if entry['image_sha256'] != frame['image_sha256'] or entry['image_hw'] != frame['image_hw']:
            raise ValueError('Frozen RGB/canvas binding differs from neutral frame')
        if entry['author_kind'].lower() not in {'model', 'human', 'model_assisted', 'human_assisted'}:
            raise ValueError('Pilot input must state its actual model/human assisted author')
        polylines = entry.get('polylines', [])
        validate_polylines(polylines, frame['image_hw'])
        if polylines and entry['status'] != 'AVAILABLE':
            raise ValueError('Available polylines require AVAILABLE status; unresolved geometry cannot be scored')
        # Empty/unavailable input follows the predeclared common identity arm.
        result[frame['id']] = deepcopy(entry)
    return result


def _identity_equivalent(first, baseline):
    excluded = {'variant'}
    return {k: v for k, v in first.items() if k not in excluded} == {
        k: v for k, v in baseline.items() if k not in excluded}


def _run(package_path, input_manifest, output, io_guard):
    started, cpu = time.monotonic(), time.process_time()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    preliminary = _read(package_path)
    for role in ('frames', 'candidates'):
        bound = Path(preliminary[role]['path'])
        io_guard.allow(bound if bound.is_absolute() else Path(package_path).parent/bound)
    package, frames, candidates = load_package(package_path)
    if len(frames) != 6 or len(candidates) != 351:
        raise ValueError('Original fixed six frames and all 351 retained candidates required')
    original_counts = (22, 94, 42, 130, 41, 22)
    inputs = load_inputs(input_manifest, frames)
    by_frame = defaultdict(list)
    for candidate in candidates:
        validate_candidate_pose(candidate)
        by_frame[candidate['id']].append(candidate)
    if tuple(len(by_frame[f['id']]) for f in frames) != original_counts:
        raise ValueError('Original per-frame retained counts differ')
    predictions, costs, checks, observations = [], [], [], []
    with FixedPoseOnly() as guard:
        for frame in frames:
            sheet = inputs[frame['id']]
            polylines = sheet['polylines'] if sheet['status'] == 'AVAILABLE' else []
            visible_by_package = {}
            for stage in STAGES:
                for family in FAMILIES:
                    outline = frame['outlines'][stage][family]
                    visible = (sample_visible(polylines, frame['image_hw'],
                                outline['configuration']['observed_sample_spacing_px'])
                               if polylines and outline.get('supported') else None)
                    visible_by_package[(stage, family)] = visible
                    observations.append(dict(id=frame['id'], stage=stage, family=family,
                        input_status=sheet['status'], outline_supported=outline.get('supported', False),
                        original_outline_configuration=deepcopy(outline.get('configuration')),
                        visible_quadrature=visible))
            values, identities = {}, {}
            for candidate in by_frame[frame['id']]:
                stage, family, cid = candidate['stage'], candidate['family'], candidate['candidate_id']
                outline = frame['outlines'][stage][family]
                value = evaluate_candidate(candidate, outline, frame['K'],
                    visible_by_package[(stage, family)], sheet['status'])
                values[(stage, family, cid)] = value
                costs.append(dict(id=frame['id'], **value))
                identity = evaluate_candidate(candidate, outline, frame['K'])
                identities[(stage, family, cid)] = identity
                if any(identity['scores'][v] != candidate['original_record']['objective'] for v in VARIANTS):
                    raise ValueError('No-input exact scalar identity failed')
            baseline = choose(frame, by_frame[frame['id']], values, 'BASELINE_E')
            _check_baseline(frame, baseline)
            final_predictions = [baseline]
            for variant in VARIANTS[1:]:
                identity = choose(frame, by_frame[frame['id']], identities, variant)
                if not _identity_equivalent(identity, baseline):
                    raise ValueError('No-input exact rank/pose identity failed')
                prediction = choose(frame, by_frame[frame['id']], values, variant)
                if not polylines and not _identity_equivalent(prediction, baseline):
                    raise ValueError('Unavailable actual input changed original ranking or pose')
                final_predictions.append(prediction)
            # Metadata is added only after the no-input exact chooser comparison.
            for prediction in final_predictions:
                key = ('G13', prediction['selected_family'], prediction['candidate_id'])
                selected_value = values.get(key)
                prediction.update(input_status=sheet['status'],
                    treatment_active=(prediction['variant'] != 'BASELINE_E' and selected_value is not None
                                      and selected_value['treatment_active']),
                    new_input_available=bool(polylines),
                    no_input_identity=not bool(polylines),
                    selected_total_score=(None if selected_value is None else selected_value['scores'][prediction['variant']]),
                    selected_r=(None if selected_value is None else selected_value['r']),
                    eligible_candidate_count=len(by_frame[frame['id']]))
                predictions.append(prediction)
            checks.append(dict(id=frame['id'], candidate_count=len(by_frame[frame['id']]),
                original_expected_count=original_counts[len(checks)], baseline_family_candidate_exact=True,
                baseline_pose_numeric_atol=1e-9, baseline_pose_metadata_exact=True,
                no_input_cost_rank_pose_exact=True, input_status=sheet['status'],
                new_input_polyline_count=len(polylines),
                original_native_projection_residual_max_abs=max(
                    (v.get('original_native_projection_residual_max_abs', 0.) for v in values.values()), default=0.)))
    _save_rows(output/'PREDICTIONS.jsonl.gz', predictions)
    _save_rows(output/'CANDIDATE_SCORES.jsonl.gz', costs)
    _save_rows(output/'OBSERVATION_QUADRATURE.jsonl.gz', observations)
    sources = {Path(__file__).name: sha(__file__), 'kernel.py': sha(Path(__file__).with_name('kernel.py'))}
    artifacts = {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}
    receipt = dict(schema='VISIBLE_PALLET_BOUNDARY_PILOT_PREDICTION_SEAL_V1',
        neutral_package_sha256=sha(package_path), input_manifest_sha256=None if input_manifest is None else sha(input_manifest),
        frames=len(frames), candidates=len(candidates), variants=list(VARIANTS), prediction_rows=len(predictions),
        full_original_retained_bank_used=True, reference_free_selection=True, before_new_reference_evaluation=True,
        baseline_checks=checks, runtime_guard=guard.receipt(), read_guard=io_guard.receipt(),
        actual_input_frame_count=sum(bool(entry['polylines']) and entry['status'] == 'AVAILABLE' for entry in inputs.values()),
        new_pose_fit_calls=0, new_candidate_calls=0, new_model_forward_calls=0,
        elapsed_seconds=time.monotonic()-started, process_cpu_seconds=time.process_time()-cpu,
        maximum_resident_set_size_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        source_sha256=sources, artifacts=artifacts,
        read_guard_snapshot='after scientific output/source/artifact hash reads; final seal write excluded')
    _write(output/'PREDICTION_SEAL.json', receipt)
    return receipt


def run(package_path, input_manifest, output):
    io_guard = ReferenceFreeIO(package_path, input_manifest, output)
    with io_guard:
        return _run(package_path, input_manifest, output, io_guard)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--neutral-package', required=True)
    parser.add_argument('--inputs', help='Frozen RGB-only VISIBLE_INPUTS.jsonl; omit only for baseline/identity run')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.neutral_package, args.inputs, args.output), indent=2))


