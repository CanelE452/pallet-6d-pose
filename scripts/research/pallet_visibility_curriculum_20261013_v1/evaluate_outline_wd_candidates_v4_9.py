"""Select sealed explicit W/D pose hypotheses with whole-image mask evidence.

The original regularized result is retained. An unchanged-model no-prior
alternative requires the previously defined image observability gate. The
explicit swapped W/D family uses its own registered vertices and Q; global
silhouette evidence compares it with the original geometry. All95 choices are
sealed before reused annotation scoring. No new local S/U query or pose fit.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'

from copy import deepcopy
from pathlib import Path
import time

import numpy as np

from . import common as C
from . import statewise_evaluation_v4 as E
from . import evaluate_visibility_pose_v4_8 as PRIOR
from . import evaluate_two_wd_families_v4_9 as WD
from .global_pallet_silhouette_v4_9 import observe_mask_outline, choose_by_outline
from .pose_observability_prior_selector_v4_8 import choose_prior_by_observability
from .scoring import pose_metric, distribution

OUTPUT = PRIOR.OUTPUT.parent / 'outline_wd_family_eval_v4_9'
MASKS = PRIOR.OUTPUT.parent / 'image_evidence_v4_6'


def select_explicit_candidates(observation, original_geometry, prior_result,
                               original_result, swapped_family, swapped_result,
                               *, original_no_prior_admissible):
    """Keep each result's native model attached throughout silhouette selection."""
    candidates = []
    results = {}
    geometries = {}
    for name, result, geometry, eligible in (
        ('INITIAL_REGULARIZED', prior_result, original_geometry, True),
        ('ORIGINAL_NO_PRIOR', original_result, original_geometry, original_no_prior_admissible),
        ('SWAPPED_WD_NO_PRIOR', swapped_result, swapped_family, True),
    ):
        if not eligible or not result['accepted']:
            continue
        pose = result['actual_pose']
        # The result must already be expressed in this family's native basis.
        q = np.asarray(geometry['physical_registration_Q'], float)
        if not np.allclose(np.asarray(pose['R_cf']) @ q, pose['R_physical'], rtol=0, atol=1e-9):
            raise ValueError('Pose and explicit family registration differ')
        if not np.allclose(geometry['cf_extents'], pose['cf_extents'], rtol=0, atol=1e-9):
            raise ValueError('Pose and explicit family extents differ')
        candidates.append(dict(name=name, xyz=geometry['native_xyz'], K=original_geometry['K'],
            R=pose['R_cf'], t=pose['centroid']))
        results[name] = result
        geometries[name] = geometry
    choice = choose_by_outline(observation, candidates)
    name = choice['selected_name']
    if name is None:
        name = 'INITIAL_REGULARIZED'
        result = prior_result
        geometry = original_geometry
    else:
        result = results[name]
        geometry = geometries[name]
    return dict(selected_arm=name, selected_result=deepcopy(result),
        selected_geometry={key: deepcopy(geometry[key]) for key in
            ('native_xyz', 'cf_extents', 'physical_registration_Q')},
        outline_selection=choice, explicit_original_fallback=choice['selected_name'] is None)


def _sealed_rows(directory, protocol, bindings):
    complete = C.read(directory / 'PREDICTION_COMPLETE.json')
    if complete['rows'] != 95 or not complete['before_any_new_pose_scoring']:
        raise ValueError('Complete primary95 sealed predictions required')
    values = []
    for part, count in (('CAL', 46), ('TEST', 49)):
        sealpath = directory / f'{part}_PREDICTION_SEAL.json'
        journal = directory / f'{part}_PREDICTIONS.jsonl.gz'
        seal = C.read(sealpath)
        if (seal['rows'] != count or not seal['before_new_annotation_score']
                or C.sha(journal) != seal['predictions_sha256']
                or C.sha(directory / 'PROTOCOL.json') != seal['protocol_sha256']
                or C.sha(sealpath) != complete['ordered_partition_prediction_seal_sha256'][part]):
            raise ValueError('Upstream prediction seal mismatch')
        rows = list(C.rows(journal))
        if len(rows) != count or any(row['partition'] != part for row in rows):
            raise ValueError('Complete original partition required')
        values.extend(rows)
        for path in (sealpath, journal):
            bindings[str(path.relative_to(C.ROOT))] = C.sha(path)
    for path in (directory / 'PROTOCOL.json', directory / 'PREDICTION_COMPLETE.json'):
        bindings[str(path.relative_to(C.ROOT))] = C.sha(path)
    for key in ('source_sha256', 'input_sha256', 'protected_user_sha256'):
        E._verify_sources(protocol[key])
    return values


def main():
    if OUTPUT.exists():
        raise FileExistsError('Preserve previous sealed evaluation')
    bindings = {}
    prior_protocol = C.read(PRIOR.OUTPUT / 'PROTOCOL.json')
    wd_protocol = C.read(WD.OUTPUT / 'PROTOCOL.json')
    prior_rows = _sealed_rows(PRIOR.OUTPUT, prior_protocol, bindings)
    wd_rows = _sealed_rows(WD.OUTPUT, wd_protocol, bindings)
    if [r['id'] for r in prior_rows] != [r['id'] for r in wd_rows]:
        raise ValueError('Same ordered CAL46/TEST49 required')
    maskseal = C.read(MASKS / 'SEAL.json')
    if (C.sha(MASKS / 'RECORDS.json') != maskseal['records_sha256']
            or C.sha(MASKS / 'PROTOCOL.json') != maskseal['protocol_sha256']):
        raise ValueError('Mask input seal mismatch')
    records = {r['id']: r for r in C.read(MASKS / 'RECORDS.json')}
    if set(records) != {r['id'] for r in prior_rows}:
        raise ValueError('Same full95 mask population required')
    for path in (MASKS / 'SEAL.json', MASKS / 'PROTOCOL.json', MASKS / 'RECORDS.json'):
        bindings[str(path.relative_to(C.ROOT))] = C.sha(path)
    sources = deepcopy(wd_protocol['source_sha256'])
    for name in (Path(__file__).name, 'global_pallet_silhouette_v4_9.py',
                 'pose_observability_prior_selector_v4_8.py'):
        path = Path(__file__).parent / name
        sources[str(path.relative_to(C.ROOT))] = C.sha(path)
    protected = wd_protocol['protected_user_sha256']
    OUTPUT.mkdir(parents=True)
    protocol = dict(schema='GLOBAL_MASK_EXPLICIT_WD_SELECTION_V4_9', CAL=46, TEST=49, severe=0,
        ids=[r['id'] for r in prior_rows], source_sha256=sources, input_sha256=bindings,
        protected_user_sha256=protected,
        rule='supported joint-feasible explicit models; observed hull IoU plus normalized symmetric median distance',
        original_no_prior_requires='same-model structural observability v4.8',
        WD_family_requires='its own explicit registered basis and accepted joint fit; independent global silhouette selection',
        initial_S_state_gate_preserved=True, new_pose_fits=0, new_mask_forwards=0,
        S_local_queries=0, U_local_queries=0, GT_inputs=False,
        annotation_metrics_only_after_all95_choice_seal=True, all95_fallbacks_and_native_flagged_kept=True,
        TEST_is_reused_iterative_exploratory=True, score_is_calibrated_probability=False)
    C.write(OUTPUT / 'PROTOCOL.json', protocol)
    started = time.monotonic()
    values = []
    for index, (row, wdrow) in enumerate(zip(prior_rows, wd_rows)):
        if (row['states'] != wdrow['states'] or row['frozen_BI_fit'] != wdrow['frozen_BI_fit']
                or row['partial_boundary_observation'] != wdrow['partial_boundary_observation']
                or row['u_region_observation'] != wdrow['u_region_observation']):
            raise ValueError('Discrete model comparison changed observations')
        g = row['runtime']['initial_geometry']
        prior = row['arms'][PRIOR.ARMS[0]]
        original = row['arms'][PRIOR.ARMS[1]]
        if original != wdrow['arms'][WD.ARMS[0]]:
            raise ValueError('Exact original-family solution was not preserved')
        gate = choose_prior_by_observability(g['native_xyz'], row['states'], row['frozen_BI_fit'],
            row['partial_boundary_observation']['boundary_observations'],
            prior['solution'], original['solution'])
        swapped = next(f for f in wdrow['families'] if f['name'] == 'SWAPPED_WD')
        record = records[row['id']]
        maskpath = MASKS / record['mask_npz']
        if (C.sha(maskpath) != record['mask_file_sha256']
                or record['image_sha256'] != row['image_sha256']):
            raise ValueError('Actual whole-image mask binding changed')
        with np.load(maskpath) as masks:
            mask = masks['pallet_mask_clean'].copy()
        observation = observe_mask_outline(mask, g['image_hw'])
        choice = select_explicit_candidates(observation, g, prior, original, swapped,
            wdrow['arms'][WD.ARMS[1]], original_no_prior_admissible=gate['no_prior_admissible'])
        value = dict(id=row['id'], partition=row['partition'], image=row['image'], image_sha256=row['image_sha256'],
            states=row['states'], original_no_prior_gate=gate, mask_file_sha256=record['mask_file_sha256'],
            GT_inputs=False, S_local_queries=0, U_local_queries=0, **choice)
        values.append(value)
        print(f"{index+1}/95 {row['id']} selected={value['selected_arm']}", flush=True)
    C.save_rows(OUTPUT / 'PREDICTIONS.jsonl.gz', values)
    for value in (sources, bindings, protected):
        E._verify_sources(value)
    C.write(OUTPUT / 'PREDICTION_SEAL.json', dict(rows=95,
        predictions_sha256=C.sha(OUTPUT / 'PREDICTIONS.jsonl.gz'), protocol_sha256=C.sha(OUTPUT / 'PROTOCOL.json'),
        before_new_annotation_metrics=True, source_sha256=sources, input_sha256=bindings,
        elapsed_seconds=time.monotonic()-started))
    reports = {}
    for part in ('CAL', 'TEST'):
        target = E.COHORT / ('OOF_pose_v1/ISOLATED_CAL_SCORING.json' if part == 'CAL'
                            else 'test_reused_v1/ISOLATED_TEST_SCORING.json')
        labels = C.read(target)['rows']
        scored = []
        for value in values:
            if value['partition'] != part:
                continue
            scored.append(dict(value, metric=pose_metric(value['selected_result']['actual_pose'],
                labels[value['id']]['target']['truth'])))
        report = dict(partition=part, frames=len(scored), all_frames_include_fallbacks=True,
            translation_cm=distribution([r['metric']['translation_cm'] for r in scored]),
            rotation_deg=distribution([r['metric']['rotation_deg'] for r in scored]),
            translation_le5=sum(r['metric']['translation_cm'] <= 5 for r in scored),
            successes_5cm_5deg=sum(r['metric']['success_5cm_5deg'] for r in scored),
            fallback_count=sum(not r['selected_result']['accepted'] for r in scored),
            selected_arm_counts={arm: sum(r['selected_arm'] == arm for r in scored)
                for arm in ('INITIAL_REGULARIZED', 'ORIGINAL_NO_PRIOR', 'SWAPPED_WD_NO_PRIOR')},
            rows=scored, target_binding=E._binding(target))
        C.write(OUTPUT / f'{part}_SCORE.json', report)
        reports[part] = report
    C.write(OUTPUT / 'SCORE_SEAL.json', dict(
        CAL_score_sha256=C.sha(OUTPUT / 'CAL_SCORE.json'), TEST_score_sha256=C.sha(OUTPUT / 'TEST_SCORE.json'),
        source_prediction_seal=E._binding(OUTPUT / 'PREDICTION_SEAL.json'),
        TEST49_all_translation_le5cm=reports['TEST']['translation_le5'] == 49))
    print({part: {key: value for key, value in report.items() if key != 'rows'}
        for part, report in reports.items()}, flush=True)


if __name__ == '__main__':
    main()
