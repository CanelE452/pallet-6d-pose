"""Read-only final B/C checks; no fitting, CNN forwards or pose solving."""
from __future__ import annotations

import ast
from collections import Counter, defaultdict
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.research.pallet_wd_info_fast_20261011 import common as C
from scripts.research.pallet_wd_info_fast_20261011.classifier import ARMS, NAMES, loaded_npz, prep_inputs


def sigmoid(values):
    values = np.asarray(values)
    output = np.empty_like(values, dtype=float)
    positive = values >= 0
    output[positive] = 1 / (1 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    output[~positive] = exponential / (1 + exponential)
    return output


def fixed_weight_probability(checkpoint, data, arm):
    """Plain CPU matrix arithmetic against already fitted fixed weights."""
    x = prep_inputs(data, checkpoint['mean'], checkpoint['scale'], arm)
    if 'coef' in checkpoint:
        logits = x @ checkpoint['coef'].T + checkpoint['intercept']
    else:
        weights = {k: v.cpu().numpy() for k, v in checkpoint['state_dict'].items()}
        hidden = np.maximum(x @ weights['0.weight'].T + weights['0.bias'], 0)
        logits = hidden @ weights['2.weight'].T + weights['2.bias']
    return sigmoid(logits.ravel())


def declared_geometry_functions():
    namespace = {'np': np}
    for relative, name in [('scripts/research/pallet_dim_conditioned_p_v1/pose.py', 'rotations'),
                           ('scripts/paper/pose_metric_closure_v1/run_pose_evaluation.py', 'cuboid')]:
        parsed = ast.parse((C.SOURCE / relative).read_text())
        node = next(n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), relative, 'exec'), namespace)
    return namespace['rotations'], namespace['cuboid']


def symmetry_checks():
    rotations, cuboid = declared_geometry_functions()
    quarter = rotations(4)[1]
    cases = []
    for name, dimensions, order in [('square_C4', [1.1, .15, 1.1], 4),
                                    ('rectangular_C2', [1.2, .15, .8], 2),
                                    ('rectangular_C1', [1.2, .15, .8], 1)]:
        points = np.concatenate([cuboid(*dimensions), np.zeros((1, 3))])
        proper = []; centre = []; legal = []
        for rotation in rotations(order):
            moved = points @ rotation.T
            distance = np.linalg.norm(moved[:, None] - points[None], axis=-1)
            correspondence = distance.argmin(-1)
            proper.append(bool(np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-12) and abs(np.linalg.det(rotation)-1)<1e-12))
            centre.append(bool(correspondence[8] == 8 and distance[8, 8] == 0))
            legal.append(bool(len(set(correspondence)) == 9 and np.max(distance.min(-1)) < 1e-12))
        minimum = min(np.degrees(np.arccos(np.clip((np.trace(rotation.T @ quarter)-1)/2, -1, 1)))
                      for rotation in rotations(order))
        cases.append(dict(case=name, dimensions_XYZ_m=dimensions, registered_order=order,
                          proper_rotations=all(proper), centre_index8_fixed=all(centre),
                          declared_group_preserves_cuboid=all(legal), quarter_turn_error_deg=float(minimum),
                          quarter_turn_equivalent=bool(minimum<1e-8)))
    assert cases[0]['quarter_turn_equivalent'] and not cases[1]['quarter_turn_equivalent']
    assert all(row['proper_rotations'] and row['centre_index8_fixed'] and row['declared_group_preserves_cuboid'] for row in cases)
    return cases


def run():
    started = time.monotonic(); C.lock_ready()
    assert C.sha(ROOT / 'scripts/research/pallet_wd_info_fast_20261011/classifier.py') == C.read(C.DOC / 'C_IO_PREFIT_LOCK.json')['code']['sha256']
    assert C.sha(ROOT / 'scripts/research/pallet_wd_info_fast_20261011/dimension_intervention.py') == C.read(C.DOC / 'B1_CODE_LOCK.json')['code']['sha256']
    allowed_contract = C.read(C.DOC / 'INPUT_CONTRACT.json')
    allowed = set(allowed_contract['allowed_ids']); protected = set(allowed_contract['excluded_protected_ids'])
    real = loaded_npz(C.PRIVATE / 'C_REAL_FEATURES.npz')
    probability = loaded_npz(C.PRIVATE / 'C_REAL_PROBABILITIES_SEALED.npz')
    synthetic = loaded_npz(C.PRIVATE / 'C_SYNTH_PREDICTIONS_SEALED.npz')
    split = C.read(C.PRIVATE / 'C_SYNTH_RECORDS.json')
    fitted = C.read(C.DOC / 'C_FITS.json')
    actual_ids = set(real['ids'].tolist())
    assert actual_ids == allowed and actual_ids.isdisjoint(protected)
    assert len(real['ids']) == 519 and Counter(zip(real['ids'], real['seeds'])) == Counter((fid, seed) for fid in allowed for seed in (1, 2, 3))
    assert np.array_equal(probability['ids'], real['ids']) and np.array_equal(probability['seeds'], real['seeds'])
    groups = {part: {r['renderer_group'] for r in split if r['split']==part} for part in ('TRAIN', 'VAL', 'TEST')}
    hashes = {part: {r['image_sha256'] for r in split if r['split']==part} for part in ('TRAIN', 'VAL', 'TEST')}
    for left, right in [('TRAIN','VAL'), ('TRAIN','TEST'), ('VAL','TEST')]:
        assert groups[left].isdisjoint(groups[right]) and hashes[left].isdisjoint(hashes[right])
    assert all(r['original_partition']=='train' for r in split if r['split']=='TRAIN')
    assert Counter(r['split'] for r in split) == {'TRAIN':4096, 'VAL':1024, 'TEST':1024}
    assert np.array_equal(synthetic['ids'], np.array([r['id'] for r in split]))
    models = {}; swap = {}; max_probability_error = 0.
    # A tagged container can arrive in either order. Serialization always looks
    # up physical names, as the production NAMES loop does. No target reversal.
    tagged_reversed = {NAMES[1]: real['G'][:, 1], NAMES[0]: real['G'][:, 0]}
    canonical = dict(real, G=np.stack([tagged_reversed[name] for name in NAMES], axis=1))
    assert np.array_equal(canonical['G'], real['G'])
    untagged_swap = dict(real, G=real['G'][:, ::-1].copy())
    for arm in ARMS:
        record = fitted['fits'][arm]; checkpoint_path = C.PRIVATE / (arm + '.pt')
        assert C.sha(checkpoint_path) == record['checkpoint']['sha256']
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        actual = fixed_weight_probability(checkpoint, real, arm)
        tagged = fixed_weight_probability(checkpoint, canonical, arm)
        wrong = fixed_weight_probability(checkpoint, untagged_swap, arm)
        difference = float(np.max(np.abs(actual-probability[arm])))
        max_probability_error = max(max_probability_error, difference)
        assert difference < 2e-6 and np.array_equal(actual, tagged)
        swap[arm] = dict(rows=len(actual), stored_probability_maxabs_error=difference,
                         tagged_container_order_probability_maxabs=0., tagged_physical_selection_changes=0,
                         untagged_G_slots_probability_maxabs=float(np.max(np.abs(actual-wrong))),
                         untagged_G_slots_selection_changes=int(((actual>.5)!=(wrong>.5)).sum()),
                         untagged_G_swap_is_valid_input_contract=False,
                         no_output_probability_or_target_reversal=True)
        models[arm] = dict(checkpoint_sha256=record['checkpoint']['sha256'], parameter_count=record['parameter_count'])
    original = C.read(C.DOC / 'B2_SYMMETRY.json'); source_bindings = []
    def scan(value):
        if isinstance(value, dict):
            if {'path','sha256','bytes'} <= value.keys():
                source_bindings.append(value)
            for child in value.values(): scan(child)
        elif isinstance(value, list):
            for child in value: scan(child)
    scan(original)
    checked_source = []
    for binding in source_bindings:
        path = C.SOURCE / binding['path']; actual = C.sha(path)
        assert actual == binding['sha256']
        checked_source.append(dict(path=binding['path'], sha256=actual, matched_pre_execution=True))
    detector = C.SOURCE / 'challenge/yolo_pose_one_model/spatial_concat_scratch/runs/YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42/weights/best.pt'
    assert C.sha(detector) == C.read(C.DOC / 'BC_INPUT_CONTRACT.json')['frozen_detector_sha256']
    base_checkpoint = C.read(C.DOC / 'B1_PREDICTION_SEAL.json')['source_checkpoint']
    assert C.sha(C.SOURCE / base_checkpoint['path']) == base_checkpoint['sha256']
    # Predictor source literal subscripts are checked against the exact whitelist.
    source_ast = ast.parse((ROOT / 'scripts/research/pallet_wd_info_fast_20261011/classifier.py').read_text())
    function = next(n for n in source_ast.body if isinstance(n, ast.FunctionDef) and n.name=='extract_synthetic')
    geometry_fields = {n.slice.value for n in ast.walk(function) if isinstance(n, ast.Subscript)
                       and isinstance(n.value, ast.Name) and n.value.id=='geom' and isinstance(n.slice, ast.Constant)}
    assert geometry_fields <= {'stems','K','dims','pad'}
    dimension_fields = {n.slice.value for n in ast.walk(function) if isinstance(n, ast.Subscript)
                        and isinstance(n.value, ast.Name) and n.value.id=='side' and isinstance(n.slice, ast.Constant)}
    assert dimension_fields <= {'dimensions','order'}
    scored = list(C.rows(C.DOC / 'C_FRAME_METRICS.jsonl.gz'))
    assert len(scored)==5190 and set(r['id'] for r in scored)==allowed
    assert Counter(r['method'] for r in scored)==Counter({arm:519 for arm in (*ARMS,'TRAIN_PRIOR','OFFICIAL')})
    ae = [r for r in C.rows(C.DOC / 'AE_CANDIDATE_METRICS.jsonl.gz') if r['method']=='N3_THEN_SUBPIX']
    assert len(ae)==519 and {r['backbone'] for r in ae}=={'YOLO'}
    b1 = list(C.rows(C.DOC / 'B1_FRAME_ROWS.jsonl.gz'))
    assert len(b1)==692 and set(r['id'] for r in b1)==allowed
    assert set(r['seed'] for r in b1)=={1}
    seal = C.read(C.DOC / 'B1_PREDICTION_SEAL.json')
    assert seal['sealed_before_reference_scoring'] and seal['full_logit_decode_max_error_px']<=1e-4
    assert C.read(C.DOC / 'C_EXTRACTION_RECEIPT.json')['outputs_sealed_before_renderer_labels']
    assert fitted['completed_fit_count']==8 and fitted['probability_seal_before_real_truth']
    b1_result = C.read(C.DOC / 'B1_RESULTS.json')
    actual_call_total = C.read(C.DOC / 'C_SYNTH_TEST_RESULTS.json')['total_C_OpenCV_calls']
    assert actual_call_total==30748<=31000 and b1_result['actual_OpenCV_calls']==3114<=3200
    C.write(C.DOC / 'C_B1_VERIFICATION.json', dict(status='PASS_WITH_EXPLICIT_LIMITATIONS',
        code=C.binding(Path(__file__)), protected_REAL_overlap=0, REAL_frames=173, REAL_seed_rows=519,
        original_TRAIN_only=True, renderer_group_and_image_hash_split_disjoint=True,
        split_counts={'TRAIN':4096,'VAL':1024,'TEST':1024}, fitting_count=8,
        all_model_checkpoint_hashes=models, original_input_bindings_verified=checked_source,
        frozen_detector_unchanged=True, N3_seed1_unchanged=True, model_normalization='TRAIN only',
        prediction_label_separation=dict(extract_geometry_fields=sorted(geometry_fields),
                                         extract_dimension_fields=sorted(dimension_fields),
                                         pose_reference_fields_in_predictor=False,
                                         sealed_before_reference_scoring=True),
        candidate_order_tests=swap, stored_classifier_probabilities_maxabs_error=max_probability_error,
        candidate_order_conclusion='Tagged candidate enumeration is invariant after fixed physical serialization; an untagged94-slot swap violates the fixed input contract and is not an architectural symmetry',
        canonical_dimensions_symmetry_checks=symmetry_checks(),
        B1_full_logit_coordinate_parity_max_error_px=seal['full_logit_decode_max_error_px'],
        REAL_candidate_metric_backbone='YOLO only', PUBLIC_C_rows=5190, PUBLIC_B1_rows=692,
        new_fits=0,new_OpenCV_calls=0,new_backbone_or_N3_forwards=0,
        fixed_small_classifier_CPU_arithmetic_passes=24, synthetic_C1_physical_contract_repaired=False,
        limitations=['Synthetic TEST was previously exposed to frozen estimators; not independent final.',
                     'D TRAIN overlaps C VAL253/TEST271, so cross-task combined synthetic TEST is not independent.',
                     'Existing synthetic C1 physical roundtrip failure retained, no GT-conditioned repair.',
                     'Generic untagged candidate-slot equivariance is not guaranteed.',
                     'BC_INPUT_CONTRACT is a pre-execution snapshot; missing confidences corrected in C_PREEXEC_AMENDMENT.'],
        verification_wall_seconds=time.monotonic()-started,
        verification_process_max_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))
    print('BC_VERIFY_PASS', round(time.monotonic()-started,2), flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    run()
