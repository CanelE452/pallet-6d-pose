"""Postseal spatial context of frozen visible support and old censored parents.

This diagnostic does not assign physical edge identities, modify observations, or
rerank poses. It compares the existing visible quadrature with ALL old finite
segments, then labels the nearest original parent's previous censor status.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    with path.open('rb') as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def run(run_folder: str, input_manifest: str, prior_parents: str,
        evaluation_receipt: str, output: str) -> dict:
    started, cpu = time.monotonic(), time.process_time()
    folder, manifest, parent_file = map(Path, (run_folder, input_manifest, prior_parents))
    evaluation_file, out = map(Path, (evaluation_receipt, output))
    seal_file = folder / 'PREDICTION_SEAL.json'
    seal = json.loads(seal_file.read_text())
    evaluation = json.loads(evaluation_file.read_text())
    assert seal['before_new_reference_evaluation'] is True
    assert evaluation['predictions_sealed_before_reference_open'] is True
    assert evaluation['prediction_seal_sha256'] == sha256(seal_file)
    assert seal['input_manifest_sha256'] == sha256(manifest)
    frozen = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    assert len(frozen) == 6 and len({row['id'] for row in frozen}) == 6
    assert all(row['status'] == 'AVAILABLE' for row in frozen)
    order = [row['id'] for row in sorted(frozen, key=lambda row: row['ordinal'])]
    parents = [json.loads(line) for line in parent_file.read_text().splitlines() if line.strip()]
    first = {}
    score_file = folder / 'CANDIDATE_SCORES.jsonl.gz'
    assert seal['artifacts'][score_file.name] == sha256(score_file)
    with gzip.open(score_file, 'rt') as stream:
        for line in stream:
            record = json.loads(line)
            if record['id'] not in first:
                first[record['id']] = record['visible_evidence']
    assert set(first) == set(order)
    records = []
    for identifier in order:
        evidence = first[identifier]
        points = np.asarray(evidence['sample_xy'], dtype=float)
        masses = np.asarray(evidence['sample_masses'], dtype=float)
        assert len(points) == len(masses) and np.all(masses > 0)
        # Keep original file order. np.argmin therefore preserves the first
        # original segment on an exact nearest-distance tie.
        original = [row for row in parents if row['id'] == identifier
                    and row['stage'] == 'G13' and row['family'] == 'ORIGINAL_WD']
        assert original and len({row['segment_id'] for row in original}) == len(original)
        starts = np.asarray([row['xy'][0] for row in original], dtype=float)
        ends = np.asarray([row['xy'][1] for row in original], dtype=float)
        directions = ends - starts
        square_lengths = np.einsum('si,si->s', directions, directions)
        assert np.all(square_lengths > 0)
        offsets = points[:, None, :] - starts[None, :, :]
        fraction = np.clip(np.einsum('nsi,si->ns', offsets, directions)
                           / square_lengths[None, :], 0.0, 1.0)
        distances = np.linalg.norm(offsets - fraction[:, :, None]
                                   * directions[None, :, :], axis=2)
        nearest = np.argmin(distances, axis=1)
        nearest_distance = distances[np.arange(len(points)), nearest]
        censored = np.asarray([row['censor'] for row in original], dtype=bool)[nearest]
        total = float(masses.sum())
        records.append(dict(
            id=identifier, ordinal=order.index(identifier) + 1,
            context_stage='G13', context_family='ORIGINAL_WD',
            original_segments=len(original),
            original_censored_segments=sum(bool(row['censor']) for row in original),
            visible_quadrature_samples=len(points), visible_mass=total,
            nearest_old_censored_mass_fraction=float(masses[censored].sum() / total),
            distance_to_all_old_segments_mass_weighted_mean_px=float(
                np.dot(masses, nearest_distance) / total),
            distance_to_all_old_segments_max_px=float(nearest_distance.max()),
            distance_threshold_used=False, physical_edge_correspondence_certified=False,
            diagnostic_postseal_only=True))
    out.mkdir(parents=True, exist_ok=False)
    table_file = out / 'OBSERVATION_CONTEXT.csv'
    with table_file.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    receipt = dict(
        schema='VISIBLE_OLD_PARENT_SPATIAL_CONTEXT_POSTSEAL_V1',
        source_sha256=sha256(Path(__file__)),
        inputs_sha256={
            'original_frozen_visible_input_manifest': sha256(manifest),
            'sealed_candidate_scores': sha256(score_file),
            'prediction_seal': sha256(seal_file),
            'postseal_evaluation_receipt': sha256(evaluation_file),
            'prior_public_PARENT_SUPPORT': sha256(parent_file)},
        frames=len(records), selection_or_observation_modified=False,
        postseal_status_verified=True, all_old_finite_segments_considered=True,
        nearest_tie='first original file order', distance_thresholds_or_sweeps=0,
        statistical_information_or_physical_identity_claimed=False,
        new_NN=0, new_pose_fit=0, new_candidate=0,
        artifacts={table_file.name: sha256(table_file)},
        elapsed_seconds=time.monotonic() - started,
        process_cpu_seconds=time.process_time() - cpu,
        maximum_resident_set_size_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        timing_scope='run function including input/output; imports excluded')
    (out / 'OBSERVATION_CONTEXT_RECEIPT.json').write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('run-folder', 'input-manifest', 'prior-parents', 'evaluation-receipt', 'output'):
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.run_folder, args.input_manifest, args.prior_parents,
                         args.evaluation_receipt, args.output), ensure_ascii=False))
