"""Read-only CPU verification; writes only this follow-up's aggregate AUDIT.json.

Run with the pinned pallet-yolo26 Python as a module. This is not a trainer,
does not invoke git, does not repair bindings, and never updates the ledger.
Missing downstream work is PENDING, not a fabricated successful check.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime
import io
import math
from pathlib import Path
import re
import time
import unittest
from urllib.parse import unquote

import numpy as np
from . import common as C


PACKAGE = 'scripts.research.pallet_oracle_mechanism_followup_v1'
C2 = 'C2_REAL_AFFINE_OFF'
C3 = 'C3_MANUAL38_CAPABILITY'
FORBIDDEN_ARRAY_KEYS = {
    'xy', 'gt', 'keypoints', 'keypoints_xy', 'box', 'box_xyxy', 'verified_xy',
    'raw_target', 'ref_target', 'manual_target', 'coordinates', 'target_coordinates',
    'camera_intrinsics', 'camera_matrix', 'K', 'R', 't', 'R_cf', 'R_physical',
    'centroid', 'body_R', 'body_xyz', 'model_xy', 'rvec', 'tvec',
}
REQUIRED_DOCS = (
    'REPORT_KO.md', 'CLI_REPORT_KO.md', 'PURPOSE_AND_BASELINE.md',
    'PRIOR_ATTEMPTS.md', 'RELATED_WORK_AND_TRANSFER.md', 'ROOT_CAUSE_MAP.md',
    'ORACLE_REPORT_KO.md', 'EXPERIMENT_LOG.md', 'CLAIM_IMPACT.md',
    'NEXT_DECISION.md', 'INPUT_BINDINGS.json', 'DATA_ROLE_LEDGER.json',
    'RESOURCE_LEDGER.json', 'STATE.json',
)


def require(condition, message):
    """Messages describe an invariant, never include private coordinate values."""
    if not condition:
        raise AssertionError(message)


def close(a, b):
    from .pose_oracle import D
    D.close(a, b)


def bindings(value):
    if isinstance(value, dict):
        if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str):
            yield value
        else:
            for child in value.values():
                yield from bindings(child)
    elif isinstance(value, list):
        for child in value:
            yield from bindings(child)


def verify_bindings(value):
    found = list(bindings(value))
    for binding in found:
        C.verify(binding)
        if 'bytes' in binding:
            require((C.ROOT / binding['path']).stat().st_size == binding['bytes'], 'Bound file size changed')
    return len(found)


def numeric_array(value):
    """A numeric list, including nested arrays; metric scalars are not arrays."""
    if not isinstance(value, list) or not value:
        return False
    return any(isinstance(v, (int, float)) and not isinstance(v, bool)
               or numeric_array(v) for v in value)


def private_array_paths(value, prefix='$'):
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            here = f'{prefix}.{key}'
            if key in FORBIDDEN_ARRAY_KEYS and numeric_array(child):
                found.append(here)
            found.extend(private_array_paths(child, here))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(private_array_paths(child, f'{prefix}[{index}]'))
    return found


def markdown_image_targets(text):
    # Inline Markdown image links and ordinary links to image files. External
    # URLs are deliberately not fetched; this audit checks local artifacts.
    for target in re.findall(r'!?\[[^\]]*\]\(([^\n]+?)\)', text):
        target = target.strip()
        if target.startswith('<') and '>' in target:
            target = target[1:target.index('>')]
        else:
            target = target.split(' "', 1)[0].split(" '", 1)[0]
        target = unquote(target.split('#', 1)[0])
        if not re.match(r'^(?:https?:|data:|mailto:)', target) and re.search(r'\.(?:png|jpe?g|svg|webp)$', target, re.I):
            yield target


def cap_checks(ledger, observed_fits=()):
    caps = ledger['caps']
    hard_caps = dict(cycles=3, fits=12, optimizer_updates=7680,
                     per_fit_updates=640, gpu_seconds=21600, wall_seconds=36000)
    require(all(caps[k] <= v for k, v in hard_caps.items()), 'Declared resource cap exceeds authorization')
    require(all(event[k] >= 0 for event in ledger['events']
                for k in ('gpu_seconds', 'fits', 'optimizer_updates')), 'Negative resource event cannot refund budget')
    sums = {key: sum(event[key] for event in ledger['events'])
            for key in ('gpu_seconds', 'fits', 'optimizer_updates')}
    for key, value in sums.items():
        require(math.isclose(value, ledger['totals'][key], rel_tol=1e-12, abs_tol=1e-6), 'Resource total does not equal event sum')
        require(0 <= value <= caps[key], 'Recorded resource cap exceeded')
    require(0 <= ledger['totals']['elapsed_wall_seconds'] <= caps['wall_seconds'], 'Recorded wall cap exceeded')
    for fit in observed_fits:
        require(0 < fit['optimizer_steps'] <= caps['per_fit_updates'], 'Per-fit update cap exceeded')
    observed = dict(fits=len(observed_fits), optimizer_updates=sum(f['optimizer_steps'] for f in observed_fits))
    require(observed['fits'] <= caps['fits'] and observed['optimizer_updates'] <= caps['optimizer_updates'], 'Observed fit artifacts exceed cap')
    return dict(event_sums=sums, observed_completed_artifacts=observed,
                ledger_caught_up=all(sums[k] >= observed[k] for k in observed),
                recorded_wall_seconds=ledger['totals']['elapsed_wall_seconds'],
                wall_basis='Recorded last ledger event; audit reruns after completion do not extend execution time')


def input_preservation():
    data = C.read(C.DOC / 'INPUT_BINDINGS.json')
    require(len(data['files']) == 31, 'Expected original 31 input bindings')
    count = verify_bindings(data['files'])
    papers = [b for b in data['files'] if b['path'].startswith('_docs/paper/selftraining_submission_v1/')]
    require({Path(b['path']).name for b in papers} == {'manuscript.tex', 'manuscript.pdf', 'material_extension.tex'}, 'Original paper binding inventory differs')
    return dict(verified_original_bindings=count, original_paper_files=3,
                paper_hashes_unchanged=True, baseline_GT_checkpoint_hashes_unchanged=True,
                original_HEAD=data['head_start'], files=data['files'])


def namespace_scope():
    require(Path(__file__).resolve().is_relative_to(C.ROOT / 'scripts/research' / C.NAME), 'Audit code outside namespace')
    parent_path = C.DOC / 'GIT_SCOPE_AUDIT.json'
    if parent_path.exists():
        parent = C.read(parent_path)
        require(parent['old_tracked_scope_check_pass'] and not parent['outside_namespace_committed']
                and not parent['tracked_current_changes_outside_namespace'], 'Parent Git scope audit reports an out-of-scope change')
        require(parent['start'] == C.read(C.DOC / 'INPUT_BINDINGS.json')['head_start'], 'Parent Git audit used a different starting HEAD')
        return dict(status='PASS', evidence_type='PARENT_GIT_CHECK_ARTIFACT',
                    source=C.bind(parent_path), checked_head=parent['checked_head'], checked_at=parent['utc'],
                    audit_writes_only=str((C.DOC / 'AUDIT.json').relative_to(C.ROOT)),
                    scope='Parent git check reused at the stated HEAD/time; this CPU audit does not independently call git or claim later changes were checked',
                    parent_action='Refresh parent artifact immediately before final commit and separately verify final push/remote HEAD')
    return dict(status='NOT_CHECKED_PARENT_GIT',
                audit_writes_only=str((C.DOC / 'AUDIT.json').relative_to(C.ROOT)),
                reason='No git operation authorized for this subtask. Original 31 bound files are hash-verified separately; they do not prove preservation of every old tracked file.',
                parent_action='Compare final git diff/status to startup tracked changes; stage only the new public code/docs namespace. Confirm private data/weights/RGB are not staged.')


def geometry_reference_preservation():
    path = C.DOC / 'GEOMETRY_REFERENCE_PRESERVATION.json'
    data = C.read(path)
    require(data['preserved_from_prior_oracle_scoring'] and len(data['files']) == 3, 'Geometry reference preservation inventory differs')
    count = verify_bindings(data)
    original = C.read(C.DOC / 'ORACLE_POSE_RESULTS.json')['reference_bindings']
    original_bindings = {b['path']: b for b in bindings(original)}
    require(all(original_bindings.get(b['path']) == b for b in data['files']), 'Preservation references not bound by original oracle scoring')
    return dict(hash_bindings_verified=count, original_reference_files_preserved=3,
                source=C.bind(path), provenance='Legacy geometry-derived; hash preservation does not make it independent physical pose truth')


def fixed_pose_oracle():
    from . import pose_oracle as O
    public = C.read(C.DOC / 'ORACLE_POSE_RESULTS.json')
    private = C.read(O.RAW / 'POSE_ORACLE_RESULTS.json')
    close(public, private)
    lock = C.read(O.RAW / 'CANDIDATES_LOCK.json')
    nbindings = verify_bindings(lock) + verify_bindings(public)
    require(lock['no_reference_coordinates_read'] and lock['inference_only'], 'Candidate lock is not inference-only')
    rows_checked = 0
    counts = {}
    for material, expected in [('PLASTIC', 128), ('WOOD', 45)]:
        candidate = C.read(O.RAW / f'{material}_CANDIDATES.json')
        metrics = C.read(O.RAW / f'{material}_METRICS.json')['arms']
        _, _, original_poses, _ = O.population(material)
        counts[material] = {}
        for arm in O.ARMS[material]:
            require(len(metrics[arm]) == expected, 'Pose denominator changed')
            old_parity = 0
            for fid, record in metrics[arm].items():
                cand = candidate['arms'][arm][fid]
                require({h['name'] for h in cand['hypotheses']} == {h['name'] for h in record['hypotheses']}, 'Oracle candidate membership changed')
                selected = O.oracle_choice(record['hypotheses'])
                reordered = O.oracle_choice(list(reversed(record['hypotheses'])))
                require((selected or {}).get('name') == (reordered or {}).get('name') == record['oracle_name'], 'Oracle choice/order invariant failed')
                close((selected or {}).get('metric', dict(id=fid, available=False)), record['oracle'])
                if arm in original_poses:
                    close(cand['current'], original_poses[arm][fid]); old_parity += 1
                if record['current']['available']:
                    require(record['oracle']['available'], 'Current pose missing from oracle candidate set')
                    require(record['oracle']['ADDsym_normalized'] <= record['current']['ADDsym_normalized'] + 1e-12, 'Oracle worse than current')
                rows_checked += 1
            aggregate = O.aggregate(list(metrics[arm].values()))
            close(aggregate, public['materials'][material]['groups']['ALL'][arm])
            require(aggregate['gap'] >= -1e-12, 'Negative oracle gap')
            counts[material][arm] = dict(frames=expected, existing_pose_numeric_parity=old_parity)
    return dict(frame_arm_records=rows_checked, material_arms=counts, hash_bindings_verified=nbindings,
                AUC_contract='Original normalized ADDsym trapezoid: 1001 thresholds over [0,0.1]; unavailable retains full denominator',
                no_full_solver_rerun='Stored full repeat parity reused by hash; unit tests rerun one native frame/material')


def c1_zero_change():
    from . import pose_oracle as O
    directory = C.DOC / 'cycles/C1_HUBER_D9'
    raw = C.RAW / 'cycles/C1_HUBER_D9'
    result = C.read(directory / 'RESULTS.json')
    lock = C.read(raw / 'PREDICTIONS_LOCK.json')
    verified = verify_bindings(lock) + verify_bindings(result)
    require(lock['GT_reference_read'] is False and lock['same_candidate_set'] and lock['no_fit'], 'C1 information contract changed')
    require(lock['delta_px'] == result['delta_px'] == 12, 'C1 robust scale changed')
    require(result['new_fits'] == result['updates'] == result['GPU_seconds'] == 0, 'C1 unexpected training/GPU')
    cues = C.read(C.ROOT / lock['source']['path'])['materials']
    selections = C.read(raw / 'SELECTIONS.json')
    checked = 0
    for material, arms in cues.items():
        candidates = C.read(O.RAW / f'{material}_CANDIDATES.json')
        for arm, frames in arms.items():
            for fid, hypotheses in frames.items():
                scores = {}
                for name, cue in hypotheses.items():
                    residual = np.asarray(cue['residual9_px'], float)
                    require(residual.shape == (9,) and np.isfinite(residual).all() and (residual >= 0).all(), 'C1 residual9 contract failed')
                    # Algebraically independent expression of the fixed Huber loss.
                    clipped = np.minimum(residual, 12)
                    robust = math.sqrt(float(np.mean(clipped**2 + 24*np.maximum(residual-12, 0))))
                    scores[name] = float(cue['selector_score']) - math.sqrt(float(np.mean(residual**2))) + robust
                choice = min(scores, key=lambda name: (scores[name], name)) if scores else None
                require(choice == selections[material][arm][fid] == candidates['arms'][arm][fid]['selected_name'], 'C1 changed a production choice')
                checked += 1
        for group in result['groups'][material].values():
            for value in group.values():
                require(value['selected_changes'] == 0 and value['delta_AUC'] == 0, 'C1 reported nonzero change')
                close(value['old'], value['new'])
    require(checked == 692, 'C1 expected all 692 frame-arm choices')
    return dict(independently_recomputed_choices=checked, changed_choices=0, hash_bindings_verified=verified)


def c2_completed_pair():
    import torch
    from . import cycle_affine as A
    from . import cycle_affine_eval as E
    from scripts.research.pallet_material_selftrain_closure_v1.infer_eval import assert_detector_parity
    raw = C.RAW / 'cycles' / C2
    doc = C.DOC / 'cycles' / C2
    result = C.read(doc / 'RESULTS.json')
    parity = C.read(doc / 'TRAINING_PARITY.json')
    preflight = C.read(doc / 'PREFLIGHT.json')
    lock = C.read(raw / 'PREDICTIONS_LOCK.json')
    start = C.read(raw / 'SCORING_START.json')
    verified = sum(verify_bindings(x) for x in (result, preflight, lock, start))
    require(lock['no_reference_coordinates_read'], 'C2 predictions were not locked inference-only')
    require(datetime.fromisoformat(lock['created_at']) < datetime.fromisoformat(start['utc']), 'C2 scoring did not follow prediction lock')
    fits = [C.read(raw / f'FIT_{arm}.json') for arm in A.ARMS]
    require(len(fits) == 4 and sum(f['optimizer_steps'] for f in fits) == 1280, 'C2 fit/update budget changed')
    for fit in fits:
        verified += verify_bindings(fit)
        require(fit['complete'] and fit['optimizer_steps'] == 320 and fit['exact_R0_initialization'] and fit['frozen_state_exact'], 'C2 incomplete or incompatible fit')
        require(fit['initialization'] == fits[0]['initialization'], 'C2 initial checkpoints differ')
        require(fit['manual_supervision_added'] == 0 and not fit['GT_training'] and not fit['data_rewritten'], 'C2 supervision contract changed')
    initial = torch.load(C.ROOT / fits[0]['initialization']['path'], map_location='cpu', weights_only=False)['model'].float()
    base = initial.state_dict()
    buffers = {name for name, _ in initial.named_buffers()}
    protected = {name for name in base if name in buffers or not A.pose_parameter(name)}
    require(len(protected) == 747, 'Expected all 747 protected tensors including all buffers')
    for fit in fits:
        final = torch.load(C.ROOT / fit['checkpoint']['path'], map_location='cpu', weights_only=False)['model'].float().state_dict()
        require(all(torch.equal(base[name], final[name]) for name in protected), 'C2 protected checkpoint state changed')
    material_counts = {}
    for material, nframes, corners, matched in [('PLASTIC', 128, 985, 120), ('WOOD', 45, 346, 45)]:
        traces = [C.read(raw / f'TRACE_{material}_{target}.json') for target in ('RAW', 'REF')]
        require(all(len(t) == 320 for t in traces), 'C2 trace length changed')
        exposure = {'SOURCE': Counter(), 'REAL': Counter()}
        for a, b in zip(*traces):
            require(all(a[k] == b[k] for k in ('names', 'images', 'boxes', 'support', 'batch_idx', 'roles')), 'C2 actual paired batches differ')
            for role in exposure:
                exposure[role].update(a['roles'][role])
        require({k: dict(v) for k, v in exposure.items()} == parity[material]['supervision_exposure'], 'C2 exposure sum differs')
        require(exposure['SOURCE']['images'] == exposure['REAL']['images'] == 2560, 'C2 source/real exposure differs')
        p = preflight['materials'][material]
        require(p['source_bit_exact'] == p['real_changed'] == 32 and p['RNG_same'] == p['paired_RGB_mask_boxes_exact'] == 64, 'C2 preflight incomplete')
        groups = result['materials'][material]['groups']
        rows = C.read(raw / f'{material}_METADATA.json')
        predictions = C.read(raw / f'{material}_PREDICTIONS.json')
        frames = C.read(raw / f'{material}_FRAME_METRICS.json')
        fixed_frames = C.read(raw / f'{material}_FIXED_ID_METRICS.json')
        pose_frames = C.read(raw / f'{material}_POSE_METRICS.json')
        require(len(rows) == nframes, 'C2 inference metadata population changed')
        for row in rows:
            C.verify(row['image'])
            for arm in ('NEW_RAW', 'NEW_REF'):
                assert_detector_parity(predictions['R0'][row['id']], predictions[arm][row['id']])
        for group, ids in E.groups(rows).items():
            for arm in predictions:
                close(E.summary2([frames[arm][i] for i in ids]), groups[group][arm]['twoD'])
                close(E.summary2([fixed_frames[arm][i] for i in ids]), groups[group][arm]['fixed_ID'])
                close(E.summary6([pose_frames[arm][i] for i in ids]), groups[group][arm]['sixD'])
        require(set(groups['ALL']) == {'R0', 'OLD_RAW', 'OLD_REF', 'SYN', 'NEW_RAW', 'NEW_REF'}, 'C2 baseline arms missing')
        for arm, value in groups['ALL'].items():
            two, fixed, pose = value['twoD'], value['fixed_ID'], value['sixD']
            require(two['total_frames'] == pose['frames'] == nframes and two['corners'] == fixed['corners'] == corners and two['matched'] == matched, 'C2 full denominators changed')
            require(pose['pose_coverage'] == 1., 'C2 pose coverage changed')
            for threshold in ('5', '10', '20'):
                require(math.isclose(two['PCK'][threshold], two['correct'][threshold]/corners), 'C2 PCK count/fraction mismatch')
            for kind in ('recording:', 'severity:'):
                rows = [g[arm]['twoD'] for name, g in groups.items() if name.startswith(kind)]
                require(rows and sum(r['total_frames'] for r in rows) == nframes and sum(r['corners'] for r in rows) == corners, 'C2 stratified denominator does not sum to full')
                for threshold in ('5', '10', '20'):
                    require(sum(r['correct'][threshold] for r in rows) == two['correct'][threshold], 'C2 stratified correct counts do not sum to full')
        material_counts[material] = dict(frames=nframes, fixed_ID_corners=corners, matched=matched, paired_actual_batches=320)
        for target in ('RAW', 'REF'):
            oldfit = C.read((C.P.REC / 'pose_only' / f'FIT_{target}_LR5.json') if material == 'PLASTIC'
                            else C.M.DOC / f'FIT_WOOD_{target}_LR5.json')
            newfit = next(fit for fit in fits if fit['arm'] == f'{material}_{target}')
            for name, fit in [('OLD_'+target, oldfit), ('NEW_'+target, newfit)]:
                C.verify(fit['results_csv'])
                with (C.ROOT / fit['results_csv']['path']).open() as stream:
                    last = list(csv.DictReader(stream))[-1]
                values = {key.strip(): float(value) for key, value in last.items()
                          if key.strip().startswith(('metrics/', 'val/'))}
                close(values, result['source_validation32']['results'][material][name])
    require(result['verified66']['points'] == 66 and result['verified66']['frames'] == 16, 'C2 verified subset changed')
    require(result['resources']['fits'] == 4 and result['resources']['optimizer_updates'] == 1280, 'C2 public resource counts changed')
    return dict(fits=4, optimizer_updates=1280, protected_tensors_per_checkpoint=len(protected),
                protected_note='747 includes every named buffer, versus 744 in the narrower training-loop check',
                material_counts=material_counts, verified_points=66, hash_bindings_verified=verified,
                frozen_detector_comparisons=346, native_image_hashes_checked=173,
                all_2D_fixedID_6D_groups_reaggregated=True, source32_validation_final_csvs_compared=8)


def coordinate_oracle():
    from . import coordinate_oracle as O
    result = C.read(C.DOC / 'ORACLE_COORDINATE_RESULTS.json')
    protocol = C.read(C.DOC / 'ORACLE_COORDINATE_PROTOCOL.json')
    private = C.read(O.RAW / 'POINT_ERRORS_AND_ORACLE_CHOICES_PRIVATE.json')
    verified = verify_bindings(protocol) + verify_bindings(result)
    require(result['GT_DEPENDENT'] and result['DIAGNOSTIC_ONLY'] and not result['production_training_or_inference_input'], 'Coordinate oracle isolation flags changed')
    truth = C.read(C.P.TRUTH)
    counts = {}
    for material, expected in [('PLASTIC', 985), ('WOOD', 346)]:
        predictions = C.read(O.paths(material) / 'PREDICTIONS.json')
        points, symmetries, parity = O.legacy(material, predictions, truth)
        close(points, private['materials'][material]['points'])
        require(len(points) == expected, 'Coordinate oracle denominator changed')
        for group, rows in O.group_points(points).items():
            summary, _ = O.coordinate_summary(rows, O.ARMS[material])
            close(summary, result['materials'][material]['strict_fixed_ID'][group])
            for threshold in ('5', '10', '20'):
                whole = summary['whole_output_by_objective'][threshold]['PCK'][threshold]['correct']
                require(summary['per_point_fixed_identity']['PCK'][threshold]['correct'] >= whole, 'Per-point bound below whole-output')
                require(all(whole >= a['PCK'][threshold]['correct'] for a in summary['baselines'].values()), 'Whole-output oracle below baseline')
        close(symmetries, result['materials'][material]['legacy_whole_symmetry_supplement'])
        pose, _ = O.pose_summary(material)
        close(pose, result['materials'][material]['whole_production_pose_expert_oracle'])
        counts[material] = dict(points=expected, native_error_parity_comparisons=parity)
    points = O.verified(C.read(O.paths('PLASTIC') / 'PREDICTIONS.json'), truth)
    close(points, private['materials']['PLASTIC_VERIFIED66']['points'])
    for group, rows in O.group_points(points).items():
        summary, _ = O.coordinate_summary(rows, O.ARMS['PLASTIC'])
        close(summary, result['materials']['PLASTIC_VERIFIED66']['groups'][group])
    return dict(material_counts=counts, verified_points=len(points), hash_bindings_verified=verified,
                recomputation='Native fixed-ID errors and all recording/severity summaries, whole-object symmetry supplement, whole-production-pose expert oracle',
                isolation_scope='Diagnostic flags, stored schema, and contract unit tests; not a dynamic information-flow proof of arbitrary future code')


def public_arrays():
    files = [p for p in sorted(C.DOC.rglob('*.json')) if p.name != 'AUDIT.json']
    violations = [{'file': str(p.relative_to(C.ROOT)), 'fields': private_array_paths(C.read(p))}
                  for p in files if private_array_paths(C.read(p))]
    require(not violations, 'Public JSON contains a prohibited numeric coordinate/camera/pose array field')
    return dict(files_scanned=len(files), violations=violations,
                scope='Recursive named-field numeric-array scan, not semantic proof for arbitrary renamed/encoded fields; hash/path metadata and metric-only arrays allowed',
                forbidden_array_field_names=sorted(FORBIDDEN_ARRAY_KEYS), audit_self_excluded=True)


def figure_manifest():
    manifest = C.read(C.DOC / 'FIGURE_MANIFEST.json')
    count = verify_bindings(manifest)
    registered = {(C.ROOT / b['path']).resolve() for b in manifest['files']}
    actual = {p.resolve() for p in (C.DOC / 'figures').rglob('*') if p.is_file()}
    require(actual == registered, 'Figure files differ from manifest')
    for example in manifest['examples']:
        previous = C.read(C.ROOT / example['previous_publication']['path'])
        approved = {x.get('frame_id', x.get('id')) for x in previous['examples']}
        require(example['id'] in approved, 'Figure frame ID not previously approved/published')
        require((C.DOC / 'figures' / (example['name']+'.png')).resolve() in registered, 'Example missing from registered figures')
    return dict(registered_figures=len(registered), previously_published_example_IDs=len(manifest['examples']),
                hash_bindings_verified=count, approval_scope='Existing publication manifest ID membership; source RGB and generated figure hashes checked')


def documentation():
    missing = [name for name in REQUIRED_DOCS if not (C.DOC / name).exists()]
    manifest = C.read(C.DOC / 'FIGURE_MANIFEST.json')
    registered = {(C.ROOT / x['path']).resolve() for x in manifest['files']}
    links = 0
    for path in C.DOC.rglob('*.md'):
        for target in markdown_image_targets(path.read_text()):
            resolved = (path.parent / target).resolve()
            require(resolved.exists(), 'Broken local Markdown figure link')
            if resolved.is_relative_to(C.DOC / 'figures'):
                require(resolved in registered, 'Linked figure not registered')
            links += 1
    return dict(status='PENDING' if missing else 'PASS', missing_required_docs=missing,
                local_figure_links_checked=links, external_links='NOT_CHECKED_NETWORK_NOT_USED')


def resource_caps():
    ledger = C.read(C.DOC / 'RESOURCE_LEDGER.json')
    fit_paths = sorted((C.RAW / 'cycles').glob('*/FIT_*.json'))
    fits = [C.read(p) for p in fit_paths if C.read(p).get('complete')]
    result = cap_checks(ledger, fits)
    cycles = [p for p in (C.DOC / 'cycles').iterdir() if (p / 'SPEC.md').exists()]
    require(len(cycles) <= ledger['caps']['cycles'], 'Cycle cap exceeded')
    result.update(cycle_specs=len(cycles), completed_fit_artifacts=[C.bind(p) for p in fit_paths if C.read(p).get('complete')],
                  status='PASS' if result['ledger_caught_up'] else 'PENDING',
                  ledger_snapshot=C.bind(C.DOC / 'RESOURCE_LEDGER.json'),
                  note='PENDING ledger synchronization may occur while another agent is finishing a fit; no ledger is modified')
    return result


def c3_status():
    doc, raw = C.DOC / 'cycles' / C3, C.RAW / 'cycles' / C3
    required = [doc / 'RESULTS.json', doc / 'REPORT_KO.md', raw / 'FIT_RAW9.json', raw / 'FIT_MANUAL9.json']
    missing = [str(p.relative_to(C.ROOT)) for p in required if not p.exists()]
    if missing:
        return dict(status='PENDING', missing=missing, reason='C3 execution/scoring is not complete; no outcome assumed')
    result = C.read(doc / 'RESULTS.json')
    fits = [C.read(raw / f'FIT_{arm}.json') for arm in ('RAW9', 'MANUAL9')]
    for fit in fits:
        require(fit['complete'] and fit['optimizer_steps'] == 320, 'C3 incomplete fit or unexpected update count')
    count = verify_bindings(result) + sum(verify_bindings(fit) for fit in fits)
    independent_path=C.DOC/'AUDIT_C3_INDEPENDENT.json'
    if independent_path.exists():
        independent=C.read(independent_path)
        require(independent['status']=='PASS_CORE_ARTIFACT_AND_METRIC_CONTRACTS', 'Independent C3 audit not passing')
        count+=verify_bindings(independent)
        require(all(v['protected_tensors_exact']==747 and v['optimizer_steps']==320 for v in independent['weights'].values()), 'C3 protection/update evidence incomplete')
        require(independent['paired_training']['batches']==320 and independent['paired_training']['RGB_box_mask_order_exact'], 'C3 paired trace audit incomplete')
        require(independent['manual_support']['manual38_exact_annotation_values'] and independent['population']['TRAIN_DEV_disjoint_ID_SHA_recording'], 'C3 manual or population audit incomplete')
        require(independent['public_table_validation']['total_rows']==57, 'C3 public table audit incomplete')
        return dict(status='PASS',fits=2,optimizer_updates=640,hash_bindings_verified=count,
                    independent_audit=C.bind(independent_path),
                    scope='Generic checks rerun here plus hash-bound separate postfit audit: 747 protected tensors,320 paired batches, manual38 and evaluation recomputations.',
                    limits=independent['limits'],method_success=False)
    return dict(status='LIMITED', fits=2, optimizer_updates=640, hash_bindings_verified=count,
                scope='Completed artifacts and generic fit/binding check only; C3-specific scientific pairing/capability checks must be supplied by its own audit/tests')


def tests():
    names = ['test_pose_oracle', 'test_coordinate_oracle', 'test_mechanism', 'test_cycle_affine', 'test_final_audit']
    if (Path(__file__).parent / 'test_cycle_manual.py').exists():
        names.append('test_cycle_manual')
    suite = unittest.defaultTestLoader.loadTestsFromNames([PACKAGE+'.'+name for name in names])
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=0).run(suite)
    return dict(status='FAIL' if not result.wasSuccessful() else 'LIMITED' if result.skipped else 'PASS',
                modules=names, run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                skipped=[dict(test=str(test), reason=reason) for test, reason in result.skipped],
                unsuccessful_test_names=[str(test) for test, _ in result.failures+result.errors],
                note='CPU tests only; a skipped artifact test is not counted as verified. Test failure tracebacks omitted from public JSON to avoid leaking private arrays.')


def overall(checks):
    states = {v['status'] for v in checks.values()}
    if 'FAIL' in states:
        return 'FAIL'
    if 'PENDING' in states:
        return 'PENDING'
    if states - {'PASS'}:
        return 'PASS_WITH_EXPLICIT_LIMITATIONS'
    return 'PASS'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-tests', action='store_true', help='Report tests NOT_RUN; never silently count them as passing')
    args = parser.parse_args(argv)
    start = time.monotonic()
    checks = {}
    operations = [input_preservation, namespace_scope, geometry_reference_preservation, fixed_pose_oracle, c1_zero_change,
                  c2_completed_pair, coordinate_oracle, public_arrays, figure_manifest,
                  documentation, resource_caps, c3_status]
    if not args.skip_tests:
        operations.append(tests)
    for operation in operations:
        tick = time.monotonic()
        try:
            result = operation()
            result.setdefault('status', 'PASS')
        except FileNotFoundError as error:
            result = dict(status='PENDING', error_type=type(error).__name__,
                          reason='Required artifact absent; not verified')
        except Exception as error:
            # Assertions from numerical libraries may embed complete private arrays.
            # Do not serialize exception text or traceback into public reports.
            result = dict(status='FAIL', error_type=type(error).__name__,
                          reason='Verification raised an exception; rerun this named check locally to inspect it')
        result['CPU_wall_seconds'] = time.monotonic()-tick
        checks[operation.__name__] = result
        print(operation.__name__, result['status'], flush=True)
    if args.skip_tests:
        checks['tests'] = dict(status='NOT_RUN', reason='Explicit --skip-tests')
    result = dict(created_at=C.now(), status=overall(checks), checks=checks,
                  CPU_wall_seconds=time.monotonic()-start, GPU_seconds=0, new_fits=0, optimizer_updates=0,
                  code=C.bind(Path(__file__)), test_code=C.bind(Path(__file__).with_name('test_final_audit.py')),
                  audit_policy='No git, training, GPU, source-binding repair, ledger/state writes, or historical artifact mutation. Missing and unverified checks remain explicit.',
                  rerun='python -m '+PACKAGE+'.final_audit')
    require(not private_array_paths(result), 'Audit output would contain forbidden arrays')
    C.save(C.DOC / 'AUDIT.json', result)
    print('AUDIT', result['status'], flush=True)
    return 1 if result['status'] == 'FAIL' else 0


if __name__ == '__main__':
    raise SystemExit(main())
