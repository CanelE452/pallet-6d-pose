"""One fixed source-gated C2 ablation; seal319 local fits before reading truth."""
from collections import Counter
from pathlib import Path
import time
import cv2
import numpy as np
from ..pallet_observation_refiner_20261009_v1 import common as C, point_line as P
from ..pallet_observation_refiner_20261009_v1.inference import hidden_mask
from ..pallet_observation_refiner_20261009_v1.evaluate import packet, score
from . import all_lines as A

DOC = C.WORKTREE / '_docs/experiments/pallet_kp_supervision_gate_20261010_v1'
METHOD = 'IMAGE_ROLE_ORIGINAL_ALL_LINES'


def run():
    protocol = C.read(DOC / 'PROTOCOL.json')
    assert C.read(DOC / 'ALL_LINES_CHECKS.json')['passed']
    assert C.read(DOC / 'SOURCE_OBSERVATION_GATE.json')['complete']
    assert not (DOC / 'GEOMETRY_SEALED.jsonl.gz').exists()
    for binding in protocol['inputs']:
        assert C.sha(C.WORKTREE / binding['path']) == binding['sha256']
    frames = {f['id']: f for f in C.read(C.DOC / 'INPUTS.json')['frames']}
    prior = {r['id']: r for r in C.iter_rows(C.DOC / 'LEARNED_GEOMETRY_SEALED.jsonl.gz')
             if r['method'] == 'IMAGE_ROLE'}
    initial = {r['id']: r['initial_pose'] for r in C.iter_rows(C.DOC / 'OBSERVATIONS.jsonl.gz')
               if r['method'] == 'BASE_NO_MASK_STANDARD'}
    observations = [r for r in C.iter_rows(C.DOC / 'LEARNED_OBSERVATIONS.jsonl.gz')
                    if r['method'] == 'IMAGE_ROLE']
    assert len(frames) == len(prior) == len(initial) == len(observations) == 319
    assert {r['id'] for r in observations} == set(frames)
    counts = Counter()
    optimizer = P.least_squares
    def instrument(function, *args, **kwargs):
        counts['optimizer_starts'] += 1
        def residual(*a, **kw):
            counts['actual_optimizer_residual_evaluations'] += 1
            return function(*a, **kw)
        return optimizer(residual, *args, **kwargs)
    P.least_squares = instrument
    cv2.setNumThreads(1)
    begin = time.monotonic()
    records = []
    try:
        with C.no_truth_reads():
            for i, observation in enumerate(observations):
                fid = observation['id']; f = frames[fid]; init = initial[fid]
                H, diagnostic = hidden_mask(init)
                original = np.asarray(f['points']['BASE'], float)
                answer = A.solve(observation, np.asarray(f['K']), np.asarray(f['xyz']),
                                 init, prior[fid]['solver'], H)
                counts['actual_paths'] += 1
                counts['optimizer_reported_nfev'] += sum(a.get('nfev', 0) for a in answer.get('attempts', []))
                new = bool(answer['available']); fallback = not new and init['available']
                final = answer if new else init
                out = original.copy()
                if new:
                    for corner in observation['corners']:
                        if corner['id'] not in H: out[corner['id']] = corner['xy']
                    if H: out[H] = np.asarray(answer['projected'])[H]
                after, _ = hidden_mask(final) if new else ([], {})
                answer.update(used=[], inliers=[], final_inliers=[], fit_input_ids=[], eligible=[],
                              line_factor_edges=answer['selected_line_edges'], hard_point_consensus_claimed=False)
                result = dict(solver=answer, actual_pose=final, native_points=out,
                    initial_pose=init, pose_available=bool(final['available']),
                    new_pose_estimated=new, fallback_used=fallback, no_pose=not final['available'],
                    hidden_initial=H, hidden_after=after, excluded=H,
                    hidden_reprojected=new and bool(H), reprojected_ids=H if new else [],
                    hidden_set_changed=new and set(H) != set(after),
                    output_status='NEW_POSE' if new else 'BASELINE_FALLBACK' if fallback else 'POSE_FAILURE',
                    oracle=False, inference_GT_input=False, mask_diagnostic=diagnostic,
                    observations_semantic_sha256=C.digest(observation),
                    observation_raw_logits_sha256=observation['raw_logits_sha256'],
                    selected_corner_ids=[c['id'] for c in observation['corners']],
                    selected_line_edges=[l['edge'] for l in observation['lines']],
                    initial_or_hidden_image_points_used_in_fit=False,
                    no_match_points_filled_from_Base=False, reprojections_reused_as_observations=False,
                    independent_four_point_PnP=False, local_line_refinement=True,
                    global_solution_uniqueness_proven=False)
                records.append(packet(f, METHOD, result))
                if i % 52 == 0: print('ALL_LINES', i + 1, 319, round(time.monotonic()-begin, 2), flush=True)
    finally:
        P.least_squares = optimizer
    assert len(records) == 319
    C.save_rows(DOC / 'GEOMETRY_SEALED.jsonl.gz', records)
    execution = dict(complete=True, method=METHOD, counts=dict(counts),
                     new_model_forwards=0, new_mesh_rays=0, new_training_updates=0,
                     new_PnP_calls=0, GT_canary=True, geometry_sealed_before_GT=True,
                     accuracy_solver_seconds=time.monotonic()-begin, deployment_latency=False,
                     geometry=C.binding(DOC / 'GEOMETRY_SEALED.jsonl.gz'),
                     code=[C.binding(Path(__file__)), C.binding(Path(A.__file__)), C.binding(Path(P.__file__))])
    C.write(DOC / 'LINE_EXECUTION_SEALED.json', execution)
    C.source_modules()
    from scripts.research.pallet_training_free_compare_20261007_v1.common import load_real
    E, real, targets, _, _ = load_real(); real = {f['id']: f for f in real}
    scored = [score(E, real[r['id']], targets[r['id']], r) for r in records]
    C.save_rows(DOC / 'PREDICTIONS.jsonl.gz', scored)
    execution.update(scored=C.binding(DOC / 'PREDICTIONS.jsonl.gz'), score_after_seal=True,
                     total_seconds=time.monotonic()-begin)
    C.write(DOC / 'LINE_EXECUTION.json', execution)
    print('ALL_LINES_COMPLETE', len(scored), dict(counts), flush=True)


if __name__ == '__main__':
    run()
