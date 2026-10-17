"""Reference-free replay of all retained candidates with the original hierarchy.

The package reader accepts only already exported neutral files. No archive,
reference, prior audit labels, metric journal or solver is opened here.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

from scripts.research.pallet_selection_score_diagnosis_v1.selection import rank_retained
from scripts.research.pallet_selection_score_diagnosis_v1.runtime_guard import FixedPoseOnly
from .kernel import VARIANTS, evaluate_outline, lineage_for_outline, parent_support

FAMILIES = ('ORIGINAL_WD', 'SWAPPED_WD')
STAGES = ('G10', 'G13')
DENIED_KEYS = {'reference', 'reference_success', 'references', 'oracle', 'oracles', 'metric', 'metrics',
               'group', 'translation_cm', 'rotation_deg', 'success_le5', 'final_reference_success',
               'representative_success_candidate_id', 'reference_success_candidates_g13_all',
               'reference_success_candidates_g13_selected_family', 'family_reference_diagnosis',
               'g10_reference_success_candidate_ids', 'SUPPORT_DECISIONS'}


class ReferenceFreeIO:
    """Process-local open/network boundary, active only during this selector.

    This is an auditable restriction of this cooperative analysis process,
    not an operating-system sandbox or a security claim about hostile code.
    Only neutral package members, frozen supplier masks, this implementation,
    installed Python libraries, and this new output directory can be opened.
    """
    forbidden_artifact_tokens = ('support_decisions', 'support_review', 'case_assessments',
        'fixed_references', 'original_e_candidate_bank', 'retained_candidates',
        'candidate_metrics', 'row_metrics', 'oracle_rows', 'pair_ledger')

    def __init__(self, package_path, input_manifest, output):
        self.allowed = set()
        self.allow(package_path)
        if input_manifest is not None:
            self.allow(input_manifest)
        self.allow(__file__)
        self.allow(Path(__file__).with_name('kernel.py'))
        self.output = Path(output).resolve()
        self.library_roots = (Path(sys.base_prefix).resolve(), Path(sys.prefix).resolve())
        self.active = False
        self.open_events = 0
        self.denied = []
        self.bound_nonlibrary_reads = set()

    def allow(self, path):
        self.allowed.add(Path(path).resolve())

    def _reject(self, reason):
        self.denied.append(reason)
        raise PermissionError('Reference-free selector IO forbids '+reason)

    def _audit(self, event, args):
        if not self.active:
            return
        if event.startswith('socket.') or event.startswith('subprocess.') or event in {'os.system', 'os.posix_spawn', 'os.exec', 'pty.spawn'}:
            self._reject('network_or_subprocess:'+event)
        if event != 'open' or isinstance(args[0], int):
            return
        path = Path(args[0]).resolve()
        self.open_events += 1
        name = path.name.lower()
        # Python module names are not scientific artifact inputs. Artifact
        # denials win even if a corrupt manifest explicitly binds such a file.
        if not name.endswith(('.py', '.pyc', '.so')) and any(token in name for token in self.forbidden_artifact_tokens):
            self._reject('forbidden_scientific_artifact:'+name)
        is_output = path == self.output or self.output in path.parents
        is_library = any(path == root or root in path.parents for root in self.library_roots)
        if path not in self.allowed and not is_output and not is_library:
            self._reject('unbound_file:'+name)
        if path in self.allowed and not is_library:
            self.bound_nonlibrary_reads.add(str(path))

    def __enter__(self):
        sys.addaudithook(self._audit)
        self.active = True
        return self

    def __exit__(self, *args):
        self.active = False
        return False

    def receipt(self):
        return dict(scope='cooperative selector process only; audit hook active within run',
            open_events=self.open_events, denied_events=list(self.denied),
            nonlibrary_bound_files_opened=len(self.bound_nonlibrary_reads),
            neutral_or_supplier_or_new_output_only=True, network_and_subprocess_denied=True,
            operating_system_sandbox_claimed=False)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _rows(path):
    with (gzip.open(path, 'rt') if str(path).endswith('.gz') else Path(path).open()) as stream:
        return [json.loads(line) for line in stream]


def _write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def _save_rows(path, rows):
    # mtime=0 makes these small prediction seals reproducible from the same input.
    with Path(path).open('xb') as raw:
        with gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as compressed:
            for row in rows:
                compressed.write((json.dumps(row, separators=(',', ':'), ensure_ascii=False, allow_nan=False)+'\n').encode())


def _deny_fields(value, location='neutral'):
    if isinstance(value, dict):
        for key, child in value.items():
            if key in DENIED_KEYS:
                raise ValueError('Non-neutral field rejected before selection: '+location+'/'+key)
            _deny_fields(child, location+'/'+key)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _deny_fields(child, location+'/'+str(index))


def _bound_file(package_path, entry):
    if not isinstance(entry, dict) or set(entry) != {'path', 'sha256'}:
        raise ValueError('Each neutral input must be a path/SHA binding')
    path = Path(entry['path'])
    if not path.is_absolute():
        path = Path(package_path).parent/path
    if sha(path) != entry['sha256']:
        raise ValueError('Neutral package byte binding failed')
    return path


def load_package(path):
    package = _read(path)
    frames = _rows(_bound_file(path, package['frames']))
    candidates = _rows(_bound_file(path, package['candidates']))
    _deny_fields(frames)
    _deny_fields(candidates)
    ids = [frame['id'] for frame in frames]
    if len(ids) != len(set(ids)) or set(ids) != {candidate['id'] for candidate in candidates}:
        raise ValueError('Exact neutral frame/candidate identity binding required')
    keys = [(c['id'], c['stage'], c['family'], c['candidate_id']) for c in candidates]
    if len(keys) != len(set(keys)):
        raise ValueError('Candidate identity must include frame/stage/family/id')
    return package, frames, candidates


def validate_candidate_pose(candidate):
    """The neutral numeric pose must be one unchanged native candidate."""
    pose = candidate['pose']
    if not np.array_equal(pose['R_cf'], candidate['R']):
        raise ValueError('Candidate native R and pose R_cf differ')
    if not np.array_equal(pose['native_t'], candidate['t']):
        raise ValueError('Candidate native t and pose native_t differ')
    expected_center = np.asarray(candidate['R'])@np.asarray(candidate['native_xyz']).mean(axis=0)+candidate['t']
    if not np.array_equal(expected_center, pose['centroid']):
        raise ValueError('Candidate centroid does not match its native XYZ/R/t exactly')
    if not np.allclose(np.asarray(candidate['R'])@candidate['physical_registration_Q'], pose['R_physical'],
                       atol=1e-9, rtol=0):
        raise ValueError('Candidate physical pose and native R/Q differ')


def choose(frame, candidates, results, variant):
    """G10 family choice, then G13 within that family; no cross-stage scores."""
    if variant not in VARIANTS:
        raise ValueError('Only the three fixed variants are allowed')
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
    """Load only frozen, explicitly admissible supplier masks; unknown stays None."""
    if path is None:
        return {f['id']: (None, 'NO_INPUT_IDENTITY_TEST') for f in frames}
    entries = _rows(path)
    if [entry['id'] for entry in entries] != [frame['id'] for frame in frames]:
        raise ValueError('Exactly the same ordered full input denominator required')
    result = {}
    for frame, entry in zip(frames, entries):
        if entry['image_sha256'] != frame['image_sha256'] or entry['image_hw'] != frame['image_hw']:
            raise ValueError('Supplier RGB/canvas binding differs from neutral frame')
        if entry.get('admissible_for_censoring') is not True:
            result[frame['id']] = (None, entry['status'])
            continue
        mask_path = Path(entry['mask_path'])
        if not mask_path.is_absolute():
            mask_path = Path(path).parent/mask_path
        if sha(mask_path) != entry['mask_sha256']:
            raise ValueError('Frozen non-target mask bytes changed')
        if mask_path.suffix == '.npz':
            with np.load(mask_path, allow_pickle=False) as arrays:
                mask = arrays[entry['mask_key']].copy()
        elif mask_path.suffix == '.npy':
            mask = np.load(mask_path, allow_pickle=False)
        else:
            from PIL import Image
            with Image.open(mask_path) as image:
                mask = np.asarray(image).copy()
        digest = hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest()
        if digest != entry['pixel_bytes_sha256']:
            raise ValueError('Frozen binary mask pixel-byte binding failed')
        if mask.shape != tuple(frame['image_hw']) or not np.all((mask == 0) | (mask == 1)):
            raise ValueError('Original canvas binary non-target mask required')
        result[frame['id']] = (mask, entry['status'])
    return result


def _run(package_path, input_manifest, output, io_guard):
    started, cpu = time.monotonic(), time.process_time()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    preliminary = _read(package_path)
    for role in ('frames', 'candidates'):
        bound = Path(preliminary[role]['path'])
        io_guard.allow(bound if bound.is_absolute() else Path(package_path).parent/bound)
    if input_manifest is not None:
        for entry in _rows(input_manifest):
            if entry.get('admissible_for_censoring') is True:
                bound = Path(entry['mask_path'])
                io_guard.allow(bound if bound.is_absolute() else Path(input_manifest).parent/bound)
    package, frames, candidates = load_package(package_path)
    inputs = load_inputs(input_manifest, frames)
    by_frame = defaultdict(list)
    for candidate in candidates:
        validate_candidate_pose(candidate)
        by_frame[candidate['id']].append(candidate)
    predictions, costs, parents, checks = [], [], [], []
    with FixedPoseOnly() as guard:
        for frame in frames:
            lineages, supports = {}, {}
            mask, status = inputs[frame['id']]
            for stage in STAGES:
                for family in FAMILIES:
                    outline = frame['outlines'][stage][family]
                    lineage = lineage_for_outline(outline)
                    support = parent_support(lineage, mask if lineage['supported'] else None, input_status=status)
                    lineages[(stage, family)], supports[(stage, family)] = lineage, support
                    parents.extend(dict(id=frame['id'], stage=stage, family=family, **entry) for entry in support)
            values, identities = {}, {}
            for candidate in by_frame[frame['id']]:
                stage, family, cid = candidate['stage'], candidate['family'], candidate['candidate_id']
                key = (stage, family)
                outline = frame['outlines'][stage][family]
                value = evaluate_outline(candidate, candidate['dense'], outline, lineages[key], supports[key])
                costs.append(dict(id=frame['id'], **value))
                values[(stage, family, cid)] = value
                identity = evaluate_outline(candidate, candidate['dense'], outline, lineages[key],
                    parent_support(lineages[key], None, input_status='NO_INPUT_IDENTITY_TEST'))
                identities[(stage, family, cid)] = identity
                if any(identity['scores'][v] != candidate['original_record']['objective'] for v in VARIANTS):
                    raise ValueError('No-input scalar identity failed')
            baseline = choose(frame, by_frame[frame['id']], values, 'BASELINE_E')
            _check_baseline(frame, baseline)
            predictions.append(baseline)
            for variant in VARIANTS[1:]:
                identity = choose(frame, by_frame[frame['id']], identities, variant)
                if {k: v for k, v in identity.items() if k != 'variant'} != {k: v for k, v in baseline.items() if k != 'variant'}:
                    raise ValueError('No-input exact ranking/pose identity failed')
                prediction = choose(frame, by_frame[frame['id']], values, variant)
                if not any(p['censor'] for key in supports for p in supports[key]):
                    if {k: v for k, v in prediction.items() if k != 'variant'} != {k: v for k, v in baseline.items() if k != 'variant'}:
                        raise ValueError('No-censor exact ranking/pose identity failed')
                predictions.append(prediction)
            checks.append(dict(id=frame['id'], baseline_family_candidate_exact=True,
                baseline_pose_numeric_atol=1e-9, baseline_pose_metadata_exact=True,
                no_input_cost_rank_pose_exact=True, candidate_count=len(by_frame[frame['id']]),
                input_status=status, censored_parent_records=sum(p['censor'] for key in supports for p in supports[key])))
    _save_rows(output/'PREDICTIONS.jsonl.gz', predictions)
    _save_rows(output/'CANDIDATE_COSTS.jsonl.gz', costs)
    with (output/'PARENT_SUPPORT.jsonl').open('x') as stream:
        for record in parents:
            stream.write(json.dumps(record, separators=(',', ':'), allow_nan=False)+'\n')
    source_hashes = {Path(__file__).name: sha(__file__), 'kernel.py': sha(Path(__file__).with_name('kernel.py'))}
    artifact_hashes = {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}
    receipt = dict(schema='NON_TARGET_OUTLINE_FUNCTIONAL_PILOT_PREDICTION_SEAL_V1',
        neutral_package_sha256=sha(package_path), input_manifest_sha256=None if input_manifest is None else sha(input_manifest),
        frames=len(frames), candidates=len(candidates), variants=list(VARIANTS), prediction_rows=len(predictions),
        full_original_retained_bank_used=True, reference_free_selection=True, before_new_reference_evaluation=True,
        baseline_checks=checks, runtime_guard=guard.receipt(), read_guard=io_guard.receipt(),
        new_pose_fit_calls=0, new_candidate_calls=0, new_model_forward_calls=0,
        elapsed_seconds=time.monotonic()-started, process_cpu_seconds=time.process_time()-cpu,
        source_sha256=source_hashes, artifacts=artifact_hashes,
        read_guard_snapshot='after scientific outputs and all source/artifact hash reads; final seal write excluded')
    _write(output/'PREDICTION_SEAL.json', receipt)
    return receipt


def run(package_path, input_manifest, output):
    io_guard = ReferenceFreeIO(package_path, input_manifest, output)
    with io_guard:
        return _run(package_path, input_manifest, output, io_guard)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--neutral-package', required=True)
    parser.add_argument('--input-manifest', help='Omit only for the explicit no-input baseline/identity test')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.neutral_package, args.input_manifest, args.output), indent=2))
